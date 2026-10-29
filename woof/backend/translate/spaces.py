"""Metal address spaces for the pointers and references of a CUDA unit.

CUDA pointers are generic: one ``float*`` parameter may receive a global
buffer from one call and a per-thread array from another.  MSL requires an
address space on every pointer and reference (``device``, ``constant``,
``threadgroup``, ``thread``), and lets functions overload on them.  This pass
infers the spaces from how values actually flow, and gives each device
function one overload per distinct combination it is called with:

* seeds: a kernel's pointer parameters are ``device``; a local array or a
  local scalar bound to a reference is ``thread``; a ``__shared__`` array is
  ``threadgroup``; a program-scope table is ``constant``;
* flow: through pointer arithmetic, ``&x``/``*p``, subscripts, casts, local
  pointer variables (by their initialisers), and into callees' parameters
  (worklist over (function, parameter-spaces) instances, from the kernels);
* struct pointer fields are ``device`` (woof's accessor structs wrap kernel
  buffers), and a member function returning a reference or pointer into one
  returns ``device``.

A value whose space cannot be determined, or that receives two different
spaces where one is required, is a :class:`TranslationError` naming it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEVICE, CONSTANT, THREADGROUP, THREAD, NULL = "device", "constant", "threadgroup", "thread", "null"

_CASTS = ("CSTYLE_CAST_EXPR", "CXX_STATIC_CAST_EXPR", "CXX_REINTERPRET_CAST_EXPR",
          "CXX_CONST_CAST_EXPR", "CXX_FUNCTIONAL_CAST_EXPR")
_INTEGER_KINDS = frozenset({"ULONG", "ULONGLONG", "LONG", "LONGLONG", "UINT", "INT", "USHORT",
                            "SHORT", "UCHAR", "CHAR_U", "CHAR_S", "SCHAR", "BOOL"})

_PASS_THROUGH = ("UNEXPOSED_EXPR", "PAREN_EXPR", "CSTYLE_CAST_EXPR", "CXX_STATIC_CAST_EXPR",
                 "CXX_REINTERPRET_CAST_EXPR", "CXX_CONST_CAST_EXPR", "CXX_FUNCTIONAL_CAST_EXPR")


@dataclass
class FnInfo:
    cursor: object
    name: str
    #: (index, PARM_DECL cursor, "ptr" | "ref") for each pointer/reference parameter
    slots: list
    returns: str | None  # "ptr" | "ref" | None
    template: bool
    decls: list = field(default_factory=list)  # prototypes (non-definitions)


def _indirection(t) -> str | None:
    kind = t.get_canonical().kind.name
    if kind == "POINTER":
        return "ptr"
    if kind in ("INCOMPLETEARRAY", "CONSTANTARRAY", "VARIABLEARRAY", "DEPENDENTSIZEDARRAY"):
        return "array"
    if kind == "LVALUEREFERENCE":
        return "ref"
    return None


class SpaceAnalysis:
    def __init__(self, rewriter):
        self.rw = rewriter
        self.ci = rewriter.ci
        self.functions: dict[str, FnInfo] = {}
        self.by_name: dict[str, list[FnInfo]] = {}
        self.instances: dict[str, set] = {}
        self.returns: dict[tuple, str | None] = {}
        self._work: list = []
        self._analysing: set = set()
        #: (start, end) of a nullptr argument -> the cast's opening text
        self.null_casts: dict = {}
        #: pointer field USR -> spaces aggregate initialisers give it
        self.field_spaces: dict = {}
        #: local pointer/reference variable usr -> spaces it took (all instances)
        self.var_spaces: dict = {}
        #: function usr -> PARM_DECLs copied to a thread local at entry
        self.localised: dict = {}
        #: (function usr, instance spaces) -> {(start, end) of a pointer cast:
        #: spaces its operand took}
        self.cast_spaces: dict = {}
        #: instance -> {local var usr: spaces it took in that instance}
        self.inst_var_spaces: dict = {}
        #: method usr -> non-thread spaces its object lives in at some call
        self.this_spaces: dict = {}
        #: the unit dereferences integer addresses (host-built pointer tables)
        self.address_casts = False
        self._instance = None

    # ---------------------------------------------------------- collection
    def collect(self, cursors):
        K = self.ci.CursorKind
        for c in cursors:
            if c.kind in (K.NAMESPACE, K.LINKAGE_SPEC, K.STRUCT_DECL, K.CLASS_DECL, K.CLASS_TEMPLATE,
                          K.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION):
                self.collect(list(c.get_children()))
                continue
            if c.kind not in (K.FUNCTION_DECL, K.FUNCTION_TEMPLATE, K.CXX_METHOD, K.CONSTRUCTOR) \
                    or self.rw.is_kernel(c):
                continue
            params = [p for p in c.get_children() if p.kind == K.PARM_DECL]
            # An array parameter is a pointer (C's decay); it binds like one.
            slots = [(i, p, "ptr" if _indirection(p.type) == "array" else _indirection(p.type))
                     for i, p in enumerate(params) if _indirection(p.type)]
            if c.kind == K.CONSTRUCTOR:
                returns = None
            elif c.kind in (K.FUNCTION_DECL, K.CXX_METHOD):
                returns = _indirection(c.result_type)
            else:
                returns = _template_return(c)
            # Every function is collected, pointer-free ones too: their
            # bodies pass local addresses to the ones that are not.
            key = c.get_usr()
            info = self.functions.get(key)
            if info is None:
                info = FnInfo(c, c.spelling, slots, returns, c.kind == K.FUNCTION_TEMPLATE)
                self.functions[key] = info
                self.by_name.setdefault(c.spelling, []).append(info)
            if c.is_definition():
                if info.cursor is not c:
                    info.decls.append(info.cursor)
                info.cursor = c
                info.slots = slots
                self.localised[key] = self._reassigned_const_pointers(c, slots)
            else:
                info.decls.append(c)

    def _reassigned_const_pointers(self, fn, slots):
        """Pointer-to-const parameters the body assigns to.  Metal has no
        generic pointer, so `a = b` between parameters that arrive in
        different address spaces cannot compile; such a parameter is read
        through a thread-local copy of its one pointee instead (the pointee
        is const, so no write is lost).  A parameter indexed or offset
        beyond its first element is refused: one copied element would not
        hold the rest."""
        K = self.ci.CursorKind
        params = {p.get_usr(): p for _, p, ind in slots
                  if ind == "ptr" and p.type.get_pointee().is_const_qualified()}
        if not params:
            return []
        assigned = set()
        for node in fn.walk_preorder():
            if node.kind == K.BINARY_OPERATOR:
                kids = list(node.get_children())
                toks = [t.spelling for t in node.get_tokens()]
                if len(kids) == 2 and kids[0].kind == K.DECL_REF_EXPR and kids[0].referenced is not None:
                    usr = kids[0].referenced.get_usr()
                    lhs_end = kids[0].extent.end.offset
                    op = next((t.spelling for t in node.get_tokens()
                               if t.extent.start.offset >= lhs_end), None)
                    if usr in params and op == "=":
                        assigned.add(usr)
        for usr in assigned:
            p = params[usr]
            for node in fn.walk_preorder():
                if node.kind == K.ARRAY_SUBSCRIPT_EXPR:
                    kids = list(node.get_children())
                    base = kids[0] if kids else None
                    while base is not None and base.kind.name in ("UNEXPOSED_EXPR", "PAREN_EXPR"):
                        inner = list(base.get_children())
                        base = inner[0] if inner else None
                    if base is not None and base.kind == K.DECL_REF_EXPR and base.referenced is not None \
                            and base.referenced.get_usr() == usr:
                        idx = self.rw.src(kids[1]).strip()
                        if idx != "0":
                            self.rw.refuse(node, f"parameter {p.spelling!r} is reassigned across address "
                                                 "spaces and indexed past its first element")
        return [params[u] for u in sorted(assigned)]

    def resolve(self, ref) -> FnInfo | None:
        if ref is None:
            return None
        info = self.functions.get(ref.get_usr())
        if info is not None:
            return info
        # A template specialisation: its template, by name and arity.
        candidates = [f for f in self.by_name.get(ref.spelling, []) if f.template]
        if not candidates:
            return None
        nparams = len([a for a in ref.get_arguments()]) if hasattr(ref, "get_arguments") else None
        if len(candidates) > 1 and nparams is not None:
            candidates = [f for f in candidates
                          if len([p for p in f.cursor.get_children()
                                  if p.kind == self.ci.CursorKind.PARM_DECL]) == nparams]
        return candidates[0] if len(candidates) == 1 else None

    def resolve_call(self, call) -> FnInfo | None:
        """The function a CALL_EXPR calls.  A call that depends on template
        parameters has no resolved declaration; then the callee expression's
        declaration, or the overload/template of that name whose parameter
        count fits the call's arguments."""
        if call.referenced is not None:
            # Resolved by clang: one of ours, or a method, operator or
            # builtin this pass does not track.
            return self.resolve(call.referenced)
        kids = list(call.get_children())
        if not kids:
            return None
        callee = kids[0]
        while callee.kind.name in ("UNEXPOSED_EXPR", "PAREN_EXPR") and list(callee.get_children()):
            callee = list(callee.get_children())[0]
        nargs = len(list(call.get_arguments())) or (len(kids) - 1)
        candidates = []
        if callee.kind.name == "OVERLOADED_DECL_REF":
            for i in range(callee.get_num_overloaded_decls()):
                info = self.functions.get(callee.get_overloaded_decl(i).get_usr())
                if info is not None:
                    candidates.append(info)
        else:
            if callee.referenced is not None:
                info = self.functions.get(callee.referenced.get_usr())
                if info is not None:
                    return info
            tokens = list(callee.get_tokens())
            name = tokens[0].spelling if tokens else callee.spelling
            candidates = list(self.by_name.get(name, []))
        fitting = []
        for info in candidates:
            params = [c for c in info.cursor.get_children() if c.kind.name == "PARM_DECL"]
            required = sum(1 for c in params if not any(t.spelling == "=" for t in c.get_tokens()))
            if required <= nargs <= len(params):
                fitting.append(info)
        return fitting[0] if len(fitting) == 1 else None

    # ---------------------------------------------------------- analysis
    def run(self, kernels):
        for fn in kernels:
            self._instance = (fn.get_usr(), ())
            env = {}
            for p in fn.get_children():
                if p.kind == self.ci.CursorKind.PARM_DECL and _indirection(p.type) in ("ptr", "array"):
                    env[p.get_usr()] = DEVICE
            self.walk(fn, env)
        self.drain()
        # Functions no kernel reaches still have to compile: give them the
        # all-device instance and follow their calls too.
        for usr, info in list(self.functions.items()):
            if usr not in self.instances:
                # pointers default to buffers; references to objects (locals)
                self.request(info, tuple(THREAD if ind == "ref" else DEVICE for _, _, ind in info.slots), None)
        self.drain()

    def drain(self):
        while self._work:
            info, spaces = self._work.pop()
            self.analyse_instance(info, spaces)

    def request(self, info: FnInfo, spaces: tuple, at) -> None:
        bucket = self.instances.setdefault(info.cursor.get_usr(), set())
        if spaces not in bucket:
            bucket.add(spaces)
            self._work.append((info, spaces))

    def analyse_instance(self, info: FnInfo, spaces: tuple):
        key = (info.cursor.get_usr(), spaces)
        if key in self._analysing:
            self.rw.refuse(info.cursor, f"recursion through {info.name!r} has no Metal equivalent")
        self._analysing.add(key)
        outer, self._instance = self._instance, key
        env = {}
        local = {p.get_usr() for p in self.localised.get(info.cursor.get_usr(), [])}
        for (index, parm, _), space in zip(info.slots, spaces):
            env[parm.get_usr()] = THREAD if parm.get_usr() in local else space
        ret = self.walk(info.cursor, env, collect_returns=info.returns is not None)
        self.returns[key] = ret
        self._analysing.discard(key)
        self._instance = outer

    def walk(self, fn, env, collect_returns=False):
        K = self.ci.CursorKind
        returned = set()
        stack = [fn]
        while stack:
            node = stack.pop()
            kind = node.kind
            if kind == K.VAR_DECL and node.type.get_canonical().kind.name == "CONSTANTARRAY" \
                    and node.type.get_canonical().element_type.get_canonical().kind.name == "POINTER":
                for il in node.get_children():
                    if il.kind == K.INIT_LIST_EXPR:
                        for e in il.get_children():
                            sp = self.space(e, env)
                            if sp not in (None, NULL):
                                self.var_spaces.setdefault(node.get_usr(), set()).add(sp)
            if kind == K.VAR_DECL and _indirection(node.type) in ("ptr", "ref"):
                init = [c for c in node.get_children() if c.kind not in (K.TYPE_REF, K.TEMPLATE_REF, K.NAMESPACE_REF)]
                if init:
                    space = self.space(init[-1], env, lvalue=_indirection(node.type) == "ref")
                    if space not in (None, NULL):
                        env[node.get_usr()] = space
                        self.var_spaces.setdefault(node.get_usr(), set()).add(space)
                        self.inst_var_spaces.setdefault(self._instance, {}).setdefault(
                            node.get_usr(), set()).add(space)
            elif kind == K.CALL_EXPR:
                self.call(node, env)
            if kind == K.CONDITIONAL_OPERATOR and _indirection(node.type) == "ptr":
                self.space(node, env)       # names a `(T*)0` branch after the other
            if kind == K.INIT_LIST_EXPR:
                self._aggregate_fields(node, env, node.type)
            if kind == K.VAR_DECL:
                # in a template the list itself is untyped: the variable's type
                for il in node.get_children():
                    if il.kind == K.INIT_LIST_EXPR and il.type.get_canonical().kind.name == "VOID":
                        self._aggregate_fields(il, env, node.type)
            if kind.name in _CASTS and _indirection(node.type) == "ptr":
                operand = [c for c in node.get_children()
                           if c.kind not in (K.TYPE_REF, K.TEMPLATE_REF, K.NAMESPACE_REF)]
                if operand:
                    o = operand[-1]
                    if o.type.get_canonical().kind.name in _INTEGER_KINDS:
                        space = NULL if o.kind.name == "INTEGER_LITERAL" else DEVICE  # an address
                        if space == DEVICE:
                            self.address_casts = True
                    else:
                        space = self.space(o, env)
                    key = (node.extent.start.offset, node.extent.end.offset)
                    bucket = self.cast_spaces.setdefault(self._instance, {}).setdefault(key, set())
                    if space not in (None, NULL):
                        bucket.add(space)
            if kind == K.BINARY_OPERATOR and _indirection(node.type) == "ptr":
                kids = list(node.get_children())
                if len(kids) == 2 and kids[0].kind == K.DECL_REF_EXPR and kids[0].referenced is not None \
                        and kids[0].referenced.kind.name == "VAR_DECL":
                    lhs_end = kids[0].extent.end.offset
                    op = next((t.spelling for t in node.get_tokens() if t.extent.start.offset >= lhs_end), None)
                    if op == "=":
                        space = self.space(kids[1], env)
                        if space not in (None, NULL):
                            usr = kids[0].referenced.get_usr()
                            env.setdefault(usr, space)
                            self.var_spaces.setdefault(usr, set()).add(space)
                            self.inst_var_spaces.setdefault(self._instance, {}).setdefault(
                                usr, set()).add(space)
            elif kind == K.RETURN_STMT and collect_returns:
                children = list(node.get_children())
                if children:
                    s = self.space(children[0], env, lvalue=True)
                    if s not in (None, NULL):
                        returned.add(s)
            stack.extend(reversed(list(node.get_children())))
        if len(returned) > 1:
            self.rw.refuse(fn, f"{fn.spelling!r} returns pointers into different address spaces "
                               f"({', '.join(sorted(returned))})")
        return next(iter(returned), None)

    def _aggregate_fields(self, node, env, t):
        """``S s = {p, ...}``: the space each pointer field is given."""
        decl = t.get_canonical().get_declaration()
        if decl is None or decl.kind.name == "NO_DECL_FOUND":
            decl = t.get_declaration()
        if decl is None or decl.kind.name not in ("STRUCT_DECL", "CLASS_DECL", "CLASS_TEMPLATE"):
            return
        fields = [c for c in decl.get_children() if c.kind.name == "FIELD_DECL"]
        if not fields:          # an implicit instantiation lists none: its template's
            pattern = self._class_template(decl.spelling.split("<", 1)[0])
            if pattern is not None:
                fields = [c for c in pattern.get_children() if c.kind.name == "FIELD_DECL"]
        for field, elem in zip(fields, node.get_children()):
            if _indirection(field.type) != "ptr" and "DEPENDENT" not in field.type.get_canonical().kind.name \
                    and field.type.get_canonical().kind.name != "UNEXPOSED":
                continue
            sp = self.space(elem, env)
            if sp in (DEVICE, CONSTANT, THREADGROUP, THREAD):
                self.field_spaces.setdefault(field_key(field), set()).add(sp)

    def _class_template(self, name):
        if not hasattr(self, "_templates"):
            self._templates = {}
            for c in self.rw.tu.cursor.walk_preorder():
                if c.kind.name == "CLASS_TEMPLATE" and c.is_definition():
                    self._templates.setdefault(c.spelling, c)
        return self._templates.get(name)

    def _null_cast_node(self, e):
        """The pointer cast of an integer literal ``e`` is, through parens."""
        while e.kind.name in ("PAREN_EXPR", "UNEXPOSED_EXPR"):
            kids = list(e.get_children())
            if len(kids) != 1:
                return None
            e = kids[0]
        if e.kind.name not in _CASTS or _indirection(e.type) != "ptr":
            return None
        operand = [c for c in e.get_children() if c.kind.name not in ("TYPE_REF", "TEMPLATE_REF", "NAMESPACE_REF")]
        o = operand[-1] if operand else None
        while o is not None and o.kind.name in ("PAREN_EXPR", "UNEXPOSED_EXPR") and len(list(o.get_children())) == 1:
            o = next(o.get_children())
        if o is not None and o.kind.name == "INTEGER_LITERAL":
            return e
        return None

    def _name_cast(self, cast, space):
        key = (cast.extent.start.offset, cast.extent.end.offset)
        self.cast_spaces.setdefault(self._instance, {}).setdefault(key, set()).add(space)

    def _dependent_field(self, e, base):
        """The FIELD_DECL a dependent ``base.f`` / ``base->f`` names, by name."""
        toks = [t.spelling for t in e.get_tokens()]
        if not toks:
            return None
        t = base.type
        if t.get_canonical().kind.name in ("LVALUEREFERENCE", "RVALUEREFERENCE", "POINTER"):
            t = t.get_pointee()
        decl = t.get_declaration()
        if decl is None or decl.kind.name == "NO_DECL_FOUND":
            return None
        tmpl = getattr(decl, "specialized_template", None)
        for d in (decl, tmpl):
            if d is None:
                continue
            for c in d.get_children():
                if c.kind.name == "FIELD_DECL" and c.spelling == toks[-1]:
                    return c
        return None

    def _subscript_operator(self, base):
        """The operator[] (one return space for every overload) of ``base``'s class."""
        t = base.type
        while t.get_canonical().kind.name == "DEPENDENT":
            # `(*this)[i]` in a class template: the type through the parens and the `*`
            kids = list(base.get_children())
            if base.kind.name == "PAREN_EXPR" and len(kids) == 1:
                base = kids[0]
            elif base.kind.name == "UNARY_OPERATOR" and len(kids) == 1 and \
                    kids[0].type.get_canonical().kind.name == "POINTER":
                t = kids[0].type.get_pointee()
                break
            else:
                return None
            t = base.type
        decl = t.get_canonical().get_declaration()
        if decl is None or decl.kind.name == "NO_DECL_FOUND":
            decl = t.get_declaration()
        if decl is None or decl.kind.name == "NO_DECL_FOUND":
            return None
        names = {decl.get_usr()}
        tmpl = getattr(decl, "specialized_template", None)   # a class template's instantiation
        if tmpl is not None:
            names.add(tmpl.get_usr())
        ops = [info for info in self.by_name.get("operator[]", [])
               if info.returns and not info.slots and info.cursor.semantic_parent is not None
               and info.cursor.semantic_parent.get_usr() in names]
        return ops[0] if ops else None

    def _record_this_space(self, node, env):
        """A member call on an object in device/constant/threadgroup memory
        needs a method overload qualified for that space."""
        ref = node.referenced
        if ref is None or ref.kind.name not in ("CXX_METHOD",):
            return
        kids = list(node.get_children())
        if not kids:
            return
        obj = kids[0]
        if obj.kind.name == "MEMBER_REF_EXPR":
            inner = list(obj.get_children())
            if not inner:
                return  # implicit this: the caller's own space
            obj = inner[0]
            if _indirection(obj.type) == "ptr":   # p->f(): the pointee's space
                space = self.space(obj, env)
            else:
                space = self.space(obj, env, lvalue=True)
        else:                                      # an operator call: object first
            space = self.space(obj, env, lvalue=True)
        if space in (DEVICE, CONSTANT, THREADGROUP):
            self.this_spaces.setdefault(ref.get_usr(), set()).add(space)

    def call(self, node, env):
        self._record_this_space(node, env)
        info = self.resolve_call(node)
        if info is None:
            return
        if not info.slots:
            self.request(info, (), node)
            return
        args = list(node.get_arguments())
        spaces = []
        for index, parm, ind in info.slots:
            if index >= len(args) or args[index].location.file is None \
                    or args[index].extent.start.offset == args[index].extent.end.offset:
                # a defaulted argument: the callee's own default (nullptr)
                spaces.append(THREAD if ind == "ref" else DEVICE)
                continue
            s = self.space(args[index], env, lvalue=(ind == "ref"))
            if ind == "ref" and s in (None, NULL):
                s = THREAD  # a temporary (a literal, an rvalue) bound to a const reference
            if s is None:
                if ind == "ref":
                    s = THREAD
                else:
                    self.rw.refuse(args[index], f"cannot tell which address space the argument "
                                                f"for {info.name}({parm.spelling}) lives in")
            spaces.append(s)
        # nullptr converts to every address space: name one, or overloads
        # would be ambiguous.  The nearest pointer of the same parameter
        # type, preceding first (`f(q, nullptr)`: an optional second buffer
        # beside a first, which the callee may choose between), else device.
        def fill_for(k):
            spelled = info.slots[k][1].type.spelling
            order = list(range(k - 1, -1, -1)) + list(range(k + 1, len(spaces)))
            for j in order:
                if spaces[j] != NULL and info.slots[j][1].type.spelling == spelled:
                    return spaces[j]
            return DEVICE
        for k, ((index, parm, _), sp) in enumerate(zip(info.slots, spaces)):
            if sp == NULL:
                fill = fill_for(k)
                cast = self._null_cast_node(args[index])
                if cast is not None:            # `(T*)0`: that cast names the space itself
                    self._name_cast(cast, fill)
                    spaces[k] = fill
                    continue
                pointee = self.rw._spell_type(parm.type.get_pointee(), drop_const=False)
                start, end = self.rw.span(args[index])
                self.null_casts[(start, end)] = f"static_cast<{fill} {pointee}*>("
                spaces[k] = fill
        self.request(info, tuple(spaces), node)

    def space(self, e, env, *, lvalue=False):
        K = self.ci.CursorKind
        kind = e.kind.name
        children = [c for c in e.get_children()
                    if c.kind not in (K.TYPE_REF, K.TEMPLATE_REF, K.NAMESPACE_REF)]
        if kind in _PASS_THROUGH:
            if not children:
                return None
            if kind in _CASTS and _indirection(e.type) == "ptr" \
                    and children[-1].type.get_canonical().kind.name in _INTEGER_KINDS:
                return NULL if children[-1].kind.name == "INTEGER_LITERAL" else DEVICE
            return self.space(children[-1], env, lvalue=lvalue)
        if kind in ("CXX_NULL_PTR_LITERAL_EXPR", "GNU_NULL_EXPR", "INTEGER_LITERAL"):
            return NULL
        if kind == "DECL_REF_EXPR":
            d = e.referenced
            if d is None:
                return None
            usr = d.get_usr()
            if usr in env:
                return env[usr]
            if d.kind.name == "VAR_DECL":
                if any(c.kind == K.ANNOTATE_ATTR and c.spelling == "woof_shared" for c in d.get_children()):
                    return THREADGROUP
                parent = d.semantic_parent
                if parent is not None and parent.kind.name in ("TRANSLATION_UNIT", "NAMESPACE", "LINKAGE_SPEC"):
                    return CONSTANT
                if parent is not None and parent.kind.name in ("STRUCT_DECL", "CLASS_DECL", "CLASS_TEMPLATE",
                                                               "CLASS_TEMPLATE_PARTIAL_SPECIALIZATION"):
                    return CONSTANT     # a static data member: program scope, so `constant`
                if _indirection(d.type) == "ptr":
                    return None  # a local pointer we could not resolve
                return THREAD
            if d.kind.name == "PARM_DECL":
                if _indirection(d.type):
                    return None
                return THREAD
            return None
        if kind == "ARRAY_SUBSCRIPT_EXPR":
            if children and _indirection(children[0].type) is None:
                # `obj[i]` on a class object inside a template, where clang
                # leaves the operator call unresolved: its operator[]'s space
                op = self._subscript_operator(children[0])
                if op is not None:
                    key = (op.cursor.get_usr(), ())
                    if key not in self.returns and key not in self._analysing:
                        self.request(op, (), e)
                        self.analyse_instance(op, ())
                    return self.returns.get(key)
            return self.space(children[0], env) if children else None
        if kind == "MEMBER_REF_EXPR":
            member = e.referenced
            if member is None and children:
                member = self._dependent_field(e, children[0])   # `s.f` on a template's struct
            if member is not None and member.kind.name == "FIELD_DECL" and not lvalue \
                    and _indirection(member.type) == "ptr":
                return self.rw._field_space(member)
            if not children:
                return THREAD  # implicit this->member inside a method
            base = children[0]
            if _indirection(base.type) == "ptr":
                return self.space(base, env)
            return self.space(base, env, lvalue=True)
        if kind == "UNARY_OPERATOR":
            if lvalue and _operator(e, children) in ("-", "+", "!", "~"):
                return THREAD           # a prvalue: a temporary binds the reference
            return self.space(children[0], env, lvalue=True) if children else None
        if kind == "BINARY_OPERATOR":
            if e.type.get_canonical().kind.name == "DEPENDENT" and not lvalue \
                    and _operator(e, children) in ("+", "-"):
                # pointer arithmetic in a template: the pointer operand's space
                for c in children:
                    if _indirection(c.type) == "ptr" or c.type.get_canonical().kind.name == "DEPENDENT":
                        sp = self.space(c, env)
                        if sp in (DEVICE, CONSTANT, THREADGROUP, THREAD):
                            return sp
            if _indirection(e.type) == "ptr":
                for c in children:
                    if _indirection(c.type) == "ptr" or c.type.get_canonical().kind.name.endswith("ARRAY"):
                        return self.space(c, env)
            if children and lvalue:
                op = _operator(e, children)
                if op == ",":
                    return self.space(children[-1], env, lvalue=True)
                if op is not None and (op == "=" or (op.endswith("=") and op not in ("==", "!=", "<=", ">="))):
                    return self.space(children[0], env, lvalue=True)
                return THREAD           # arithmetic: a prvalue temporary
            return None
        if kind == "CONDITIONAL_OPERATOR" and len(children) == 3:
            a, b = self.space(children[1], env, lvalue=lvalue), self.space(children[2], env, lvalue=lvalue)
            # `c ? p : (T*)0`: the null cast takes the other branch's space
            for null_side, other in ((children[1], b), (children[2], a)):
                cast = self._null_cast_node(null_side)
                if cast is not None and other not in (None, NULL):
                    self._name_cast(cast, other)
            if a in (None, NULL):
                return b
            if b in (None, NULL) or a == b:
                return a
            self.rw.refuse(e, f"a conditional chooses between {a} and {b} pointers")
        if kind == "CALL_EXPR":
            ref = e.referenced
            info = self.resolve_call(e)
            if info is not None and info.returns:
                args = list(e.get_arguments())
                spaces = tuple((self.space(args[i], env, lvalue=(ind == "ref")) if i < len(args) else None)
                               or (THREAD if ind == "ref" else DEVICE)
                               for i, _, ind in info.slots)
                spaces = tuple(DEVICE if s == NULL else s for s in spaces)
                key = (info.cursor.get_usr(), spaces)
                if key not in self.returns:
                    self.request(info, spaces, e)
                    self.analyse_instance(info, spaces)
                return self.returns.get(key)
            return THREAD if lvalue else None
        if kind in ("COMPOUND_LITERAL_EXPR", "INIT_LIST_EXPR"):
            return THREAD
        return THREAD if lvalue else None


def field_key(field):
    """A field across a class template's pattern and its instantiations
    (whose FIELD_DECLs have their own USRs): owner name and field name."""
    parent = field.semantic_parent
    return ((parent.spelling.split("<", 1)[0] if parent is not None else ""), field.spelling)


def _operator(e, children):
    """The operator token of a unary or binary expression."""
    toks = list(e.get_tokens())
    if not toks:
        return None
    if len(children) == 1:                      # unary: prefix or postfix
        inner = children[0].extent
        for t in toks:
            if t.extent.end.offset <= inner.start.offset or t.extent.start.offset >= inner.end.offset:
                return t.spelling
        return None
    lhs_end = children[0].extent.end.offset
    rhs_start = children[1].extent.start.offset
    for t in toks:
        if lhs_end <= t.extent.start.offset and t.extent.end.offset <= rhs_start:
            return t.spelling
    return None


def _template_return(cursor):
    # FUNCTION_TEMPLATE has no result_type; read the declared return type
    # from its type spelling ("W<T> (T *, int)" -> "W<T>").
    spelled = cursor.type.spelling
    head = spelled.split("(")[0].strip()
    if head.endswith("&"):
        return "ref"
    if head.endswith("*"):
        return "ptr"
    return None
