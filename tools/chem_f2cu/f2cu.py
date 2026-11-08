"""A small Fortran-to-CUDA translator for WRF-Chem routines graded word for word.

Generalises tools/chem_radm2/transcribe_radm2.py from straight-line level
arithmetic to whole subroutines and functions: declarations (with implicit
typing where a routine has no IMPLICIT NONE), module PARAMETERs and DATA
tables, DO loops (labelled or not), IF/ELSE IF chains, CALLs with Fortran's
by-reference arguments, functions, and the one backward GO TO pattern WRF's
chem code uses.  Output is CUDA C++ the Metal backend translates as it does
every other kernel.

What makes the output word-exact against a gfortran -O0 oracle:

* every REAL(4) operation is FADD/FSUB/FMUL/FDIV/FSQRT (rounded, never
  fused) and every REAL(8) one DADD/.../__dsqrt_rn, in the source's own
  association; mixed kinds promote exactly as Fortran does;
* constant subexpressions -- including intrinsics of PARAMETERs, which
  gfortran folds with MPFR -- are folded here too, exactly (IEEE binary32 /
  binary64 per operation, correctly rounded elementary functions), and
  emitted as round-tripping literals;
* constant integer powers are gfortran's own multiplication chains (its
  powi_table); integer-variable powers are libgcc's __powisf2/__powidf2;
  real powers, EXP and LOG call WOOF's glibc 2.39 words (gfk_* in binary32,
  glibc_* in binary64).

The translator refuses anything outside that subset by name rather than
guessing.  Callers hold its output against the pinned source with a
``--check`` regeneration test, and every kernel built from it is graded
against a Fortran oracle.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction

import decimal

# --------------------------------------------------------------------------
# Exact arithmetic for constant folding
# --------------------------------------------------------------------------
_FMT = {"float": (24, -126, 127), "double": (53, -1022, 1023)}


def round_to(value: Fraction, kind: str) -> Fraction:
    """Round an exact rational to binary32/binary64, ties to even."""
    if value == 0:
        return Fraction(0)
    p, emin, emax = _FMT[kind]
    sign = -1 if value < 0 else 1
    v = abs(value)
    e = v.numerator.bit_length() - v.denominator.bit_length()
    if Fraction(2) ** e > v:
        e -= 1
    e = max(e, emin)
    if e > emax:
        raise OverflowError("constant overflows its kind")
    scale = Fraction(2) ** (p - 1 - e)
    q = v * scale
    n = q.numerator // q.denominator
    rem = q - n
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and n % 2):
        n += 1
    return sign * Fraction(n) / scale


def parse_literal(text: str, kind: str) -> Fraction:
    mant = re.sub(r"[EeDd].*$", "", text)
    exp = re.search(r"[EeDd]([+-]?\d+)", text)
    value = Fraction(mant) if mant not in ("", ".") else Fraction(0)
    if exp:
        value *= Fraction(10) ** int(exp.group(1))
    return round_to(value, kind)


def _mp_fn(name, args, kind):
    """Correctly rounded elementary function of exact arguments (gfortran's
    MPFR folding), via 100-digit decimal arithmetic."""
    ctx = decimal.Context(prec=100)
    xs = [ctx.divide(decimal.Decimal(a.numerator), decimal.Decimal(a.denominator))
          for a in args]
    if name == "exp":
        r = ctx.exp(xs[0])
    elif name == "log":
        r = ctx.ln(xs[0])
    elif name == "log10":
        r = ctx.log10(xs[0])
    elif name == "sqrt":
        r = ctx.sqrt(xs[0])
    elif name == "pow":
        r = ctx.power(xs[0], xs[1])
    else:
        raise NotImplementedError(name)
    return round_to(Fraction(r), kind)


def literal(value: Fraction, kind: str) -> str:
    if kind == "int":
        return str(int(value))
    if kind == "bool":
        return "true" if value else "false"
    f = float(value)
    if kind == "float":
        text = repr(float(f"{f:.9g}"))
        assert round_to(Fraction(float(text)), "float") == value
        return _c_float(text) + "f"
    text = repr(f)
    return _c_float(text)


def _c_float(text):
    if "e" in text or "." in text or "inf" in text:
        return text if "." in text or "e" in text else text + ".0"
    return text + ".0"


# --------------------------------------------------------------------------
# Lexing and parsing
# --------------------------------------------------------------------------
TOKEN = re.compile(r"""\s*(?:
    (?P<num>(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][+-]?\d+)?(?:_8|_4)?(?![A-Za-z]))
  | (?P<dotop>\.(?:and|or|not|eqv|neqv|true|false|gt|ge|lt|le|eq|ne)\.)
  | (?P<name>[A-Za-z_]\w*)
  | (?P<op>\*\*|>=|<=|==|/=|//|[-+*/(),<>:=])
  | (?P<str>'[^']*'|"[^"]*")
)""", re.X | re.I)

_DOTOPS = {".gt.": ">", ".ge.": ">=", ".lt.": "<", ".le.": "<=", ".eq.": "==",
           ".ne.": "/=", ".and.": ".and.", ".or.": ".or.", ".not.": ".not.",
           ".eqv.": ".eqv.", ".neqv.": ".neqv.", ".true.": ".true.",
           ".false.": ".false."}


def tokenize(text):
    pos, out = 0, []
    text = text.rstrip()
    while pos < len(text):
        m = TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise SyntaxError(f"cannot tokenize {text[pos:]!r} in {text!r}")
        pos = m.end()
        kind = m.lastgroup
        value = m.group(kind)
        if kind == "dotop":
            value = _DOTOPS[value.lower()]
            kind = "lit" if value in (".true.", ".false.") else "op"
        elif kind == "name":
            value = value.lower()
        out.append((kind, value))
    return out


class Parser:
    def __init__(self, text):
        self.t, self.i = tokenize(text), 0

    def peek(self, k=0):
        j = self.i + k
        return self.t[j] if j < len(self.t) else (None, None)

    def take(self, value=None):
        tok = self.peek()
        if value is not None and tok[1] != value:
            raise SyntaxError(f"expected {value!r}, got {tok!r} in {self.t}")
        self.i += 1
        return tok

    def done(self):
        return self.i >= len(self.t)

    # Fortran precedence, lowest first: .eqv. < .or. < .and. < .not. <
    # relational < + - < * / < unary - < **.
    def logical(self):
        node = self.disj()
        while self.peek()[1] in (".eqv.", ".neqv."):
            op = self.take()[1]
            node = ("log", op, node, self.disj())
        return node

    def disj(self):
        node = self.conj()
        while self.peek()[1] == ".or.":
            self.take()
            node = ("log", ".or.", node, self.conj())
        return node

    def conj(self):
        node = self.neg()
        while self.peek()[1] == ".and.":
            self.take()
            node = ("log", ".and.", node, self.neg())
        return node

    def neg(self):
        if self.peek()[1] == ".not.":
            self.take()
            return ("not", self.neg())
        return self.rel()

    def rel(self):
        left = self.expr()
        if self.peek()[1] in (">=", "<=", "==", "/=", "<", ">"):
            op = self.take()[1]
            return ("cmp", op, left, self.expr())
        return left

    def expr(self):
        if self.peek()[1] in ("+", "-"):
            sign = self.take()[1]
            node = self.term()
            if sign == "-":
                node = ("neg", node)
        else:
            node = self.term()
        while self.peek()[1] in ("+", "-"):
            op = self.take()[1]
            node = ("bin", op, node, self.term())
        return node

    def term(self):
        node = self.factor()
        while self.peek()[1] in ("*", "/"):
            op = self.take()[1]
            node = ("bin", op, node, self.factor())
        return node

    def factor(self):
        base = self.primary()
        if self.peek()[1] == "**":
            self.take()
            if self.peek()[1] in ("-", "+"):
                sign = self.take()[1]
                rhs = self.factor()
                return ("pow", base, ("neg", rhs) if sign == "-" else rhs)
            return ("pow", base, self.factor())
        return base

    def primary(self):
        kind, value = self.take()
        if kind == "num":
            return ("num", value)
        if kind == "lit":
            return ("bool", value == ".true.")
        if value == "(":
            node = self.logical()
            self.take(")")
            return node
        if kind == "name":
            if self.peek()[1] == "(":
                self.take("(")
                args = []
                if self.peek()[1] != ")":
                    args.append(self.logical())
                    while self.peek()[1] == ",":
                        self.take(",")
                        args.append(self.logical())
                self.take(")")
                return ("ref", value, args)
            return ("ref", value, None)
        raise SyntaxError(f"unexpected {value!r} in {self.t}")


def parse_expr(text):
    p = Parser(text)
    node = p.logical()
    if not p.done():
        raise SyntaxError(f"trailing tokens in {text!r}")
    return node


# --------------------------------------------------------------------------
# Source structure
# --------------------------------------------------------------------------
def strip_comment(line):
    out, quote = [], None
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
        elif ch == "!":
            break
        out.append(ch)
    return "".join(out).rstrip()


def logical_lines(lines):
    """(lineno, label, text) per statement; free-form continuations joined."""
    parts, start = [], None
    for n, raw in lines:
        text = strip_comment(raw.replace("\r", "")).strip()
        if not text:
            continue
        if start is None:
            start = n
        if text.startswith("&") and parts:
            text = text[1:].strip()
        cont = text.endswith("&")
        parts.append(text[:-1].strip() if cont else text)
        if cont:
            continue
        stmt = " ".join(parts)
        parts, s = [], start
        start = None
        m = re.match(r"(\d+)\s+(.*)$", stmt)
        label, stmt = (int(m.group(1)), m.group(2)) if m else (None, stmt)
        yield s, label, stmt


_HEADER = re.compile(
    r"^(?:(real(?:\*8|\s*\(\s*kind\s*=\s*8\s*\))?|integer|double\s+precision|logical)\s+)?"
    r"(subroutine|function)\s+(\w+)\s*(?:\((.*)\))?\s*$", re.I)


def find_routines(text):
    """{name: [(lineno, raw), ...]} for every routine, header to END."""
    lines = list(enumerate(text.splitlines(), 1))
    out, cur, buf = {}, None, []
    for n, raw in lines:
        low = strip_comment(raw.replace("\r", "")).strip().lower()
        if cur is None:
            m = re.match(r"^(?:(?:real(?:\*8)?|integer|double\s+precision|logical)\s+)?"
                         r"(subroutine|function)\s+(\w+)", low)
            if m:
                cur, buf = m.group(2), [(n, raw)]
            continue
        buf.append((n, raw))
        m = re.match(r"^end\s*(subroutine|function)\b\s*(\w*)", low)
        if m and m.group(2) in ("", cur):
            out[cur] = buf
            cur = None
    return out


@dataclass
class Var:
    name: str
    kind: str                     # float | double | int | bool | char
    dims: list | None = None      # [(lo_node, hi_node|None for *)]
    param: object = None          # (Fraction|int|bool, kind) for PARAMETERs
    data: list | None = None      # DATA values (Fractions/ints)
    is_arg: bool = False
    explicit: bool = True


_TYPE = re.compile(
    r"^(real\s*\*\s*8|real\s*\(\s*kind\s*=\s*8\s*\)|real\s*\(\s*8\s*\)|double\s+precision|"
    r"real(?:\s*\*\s*4)?|integer|logical|character(?:\s*\*\s*\(?[\w*]+\)?)?)\b(.*)$", re.I)


def _kind_of(word):
    w = re.sub(r"\s", "", word.lower())
    if w.startswith("character"):
        return "char"
    if w in ("real*8", "real(kind=8)", "real(8)", "doubleprecision"):
        return "double"
    if w.startswith("real"):
        return "float"
    return {"integer": "int", "logical": "bool"}[w]


def _data_groups(text):
    """DATA's  obj-list /value-list/ [,] obj-list /value-list/ ...  pairs."""
    out, i = [], 0
    while i < len(text):
        j = text.index("/", i)
        k = text.index("/", j + 1)
        objs = text[i:j].strip().lstrip(",").strip()
        out.append((objs, text[j + 1:k]))
        i = k + 1
    return out


def split_top(text, sep=","):
    out, depth, cur, quote = [], 0, [], None
    for ch in text:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur).strip())
    return out


def parse_dims(text):
    dims = []
    for d in split_top(text):
        if d.strip() == "*":
            dims.append((("num", "1"), None))
        elif ":" in d:
            lo, hi = d.split(":", 1)
            dims.append((parse_expr(lo), None if hi.strip() == "*" else parse_expr(hi)))
        else:
            dims.append((("num", "1"), parse_expr(d)))
    return dims


class Scope:
    """Declarations of one routine (or one module) plus the module scope."""

    def __init__(self, parent=None, implicit=True):
        self.vars: dict[str, Var] = {}
        self.parent = parent
        self.implicit = implicit

    def lookup(self, name):
        if name in self.vars:
            return self.vars[name]
        if self.parent is not None:
            return self.parent.lookup(name)
        return None

    def declare_line(self, stmt, evaluator):
        """Handle a specification statement; return False if executable."""
        low = stmt.lower()
        if re.match(r"^(use|implicit\s+none|intrinsic|save|contains|module|end\s+module)\b", low):
            if low.startswith("implicit"):
                self.implicit = False
            return True
        if low.startswith("external"):
            for n in split_top(stmt[8:]):
                self.vars.setdefault(n.lower(), Var(n.lower(), "external"))
                self.vars[n.lower()].kind = "external"
            return True
        if re.match(r"^format\s*\(", low):
            return True
        m = re.match(r"^parameter\s*\((.*)\)\s*$", stmt, re.I)
        if m:
            for item in split_top(m.group(1)):
                name, value = item.split("=", 1)
                self.set_param(name.strip().lower(), value, evaluator)
            return True
        m = re.match(r"^data\s+(.*)$", stmt, re.I)
        if m:
            for objs, values in _data_groups(m.group(1)):
                vals = []
                for item in split_top(values):
                    if re.match(r"^\s*\d+\s*\*", item):
                        n, v = item.split("*", 1)
                        vals += [v.strip()] * int(n)
                    else:
                        vals.append(item.strip())
                targets = split_top(objs)
                if len(targets) == 1 and "(" not in targets[0]:
                    self.set_data(targets[0].strip().lower(), vals, evaluator)
                    continue
                assert len(targets) == len(vals), stmt
                for t, val in zip(targets, vals):
                    node = parse_expr(t)
                    if node[2] is None:
                        self.set_data(node[1], [val], evaluator)
                    else:
                        self.set_element(node[1], node[2], val, evaluator)
            return True
        m = _TYPE.match(stmt.strip())
        if not m or re.match(r"^(real|integer|logical)\s*(function|\()", low.replace(" ", "")) \
                and "function" in low:
            return False
        kind = _kind_of(m.group(1))
        rest = m.group(2).strip()
        attrs = []
        if "::" in rest:
            spec, rest = rest.split("::", 1)
            attrs = [a.strip().lower() for a in split_top(spec) if a.strip()]
        elif rest.startswith(","):
            return False
        dim_attr = None
        is_param = False
        for a in attrs:
            if a.startswith("dimension"):
                dim_attr = parse_dims(a[a.index("(") + 1:a.rindex(")")])
            if a == "parameter":
                is_param = True
        for item in split_top(rest):
            init = None
            if "=" in item and not item.strip().startswith("("):
                item, init = item.split("=", 1)
            m2 = re.match(r"^(\w+)\s*(?:\((.*)\))?\s*(?:\*\s*\d+)?$", item.strip())
            if not m2:
                raise SyntaxError(f"declaration {item!r} in {stmt!r}")
            name = m2.group(1).lower()
            dims = parse_dims(m2.group(2)) if m2.group(2) else dim_attr
            v = self.vars.get(name)
            if v is None:
                v = Var(name, kind)
                self.vars[name] = v
            v.kind, v.explicit = kind, True
            if dims is not None:
                v.dims = dims
            if init is not None:
                if is_param:
                    self.set_param(name, init, evaluator)
                elif init.strip().startswith("(/"):
                    body = init.strip()[2:-2].strip()
                    ido = re.match(r"^\((.*),\s*(\w+)\s*=\s*([^,]+),\s*([^,]+)\)$", body)
                    if ido:  # implied DO of one constant value
                        lo_v, _ = evaluator.const(parse_expr(ido.group(3)), self)
                        hi_v, _ = evaluator.const(parse_expr(ido.group(4)), self)
                        items = [ido.group(1)] * (hi_v - lo_v + 1)
                    else:
                        items = split_top(body)
                    self.set_data(name, items, evaluator)
                else:
                    self.set_data(name, [init], evaluator)
        return True

    def ensure(self, name):
        v = self.lookup(name)
        if v is None:
            if not self.implicit:
                raise KeyError(f"undeclared {name!r} under IMPLICIT NONE")
            v = Var(name, "int" if name[0] in "ijklmn" else "float", explicit=False)
            self.vars[name] = v
        return v

    def set_param(self, name, text, evaluator):
        v = self.ensure(name)
        if v.kind == "char":
            return
        val, kind = evaluator.const(parse_expr(text), self)
        v.param = (evaluator.convert(val, kind, v.kind), v.kind)

    def set_element(self, name, subs, text, evaluator):
        v = self.ensure(name)
        if v.kind == "char":
            return
        size, offset, stride = 1, 0, 1
        for (lo, hi), sub in zip(v.dims, subs):
            lo_v = evaluator.const(lo, self)[0]
            hi_v = evaluator.const(hi, self)[0]
            offset += (evaluator.const(sub, self)[0] - lo_v) * stride
            stride *= hi_v - lo_v + 1
        if v.data is None:
            v.data = [None] * stride
        val, kind = evaluator.const(parse_expr(text), self)
        v.data[offset] = evaluator.convert(val, kind, v.kind)

    def set_data(self, name, items, evaluator):
        v = self.ensure(name)
        if v.kind == "char":
            return
        vals = []
        for item in items:
            val, kind = evaluator.const(parse_expr(item), self)
            vals.append(evaluator.convert(val, kind, v.kind))
        v.data = vals


# --------------------------------------------------------------------------
# Expressions
# --------------------------------------------------------------------------
# gfortran's trans-expr.cc powi_table (the same table as GCC's tree-ssa-math-opts).
POWI_TABLE = [0, 1, 1, 2, 2, 3, 3, 4, 4, 6, 5, 6, 6, 10, 7, 9,
              8, 16, 9, 16, 10, 12, 11, 13, 12, 17, 13, 18, 14, 24, 15, 26,
              16, 17, 17, 19, 18, 33, 19, 26, 20, 25, 21, 40, 22, 27, 23, 44]

_OPS = {"+": "ADD", "-": "SUB", "*": "MUL", "/": "DIV"}


class Emitter:
    def __init__(self, translator, scope, routine):
        self.tr, self.scope, self.routine = translator, scope, routine

    # ---- constant folding -------------------------------------------------
    def convert(self, val, src, dst):
        return self.tr.convert(val, src, dst)

    def const(self, node, scope=None):
        return self.tr.const(node, scope or self.scope)

    # ---- code -------------------------------------------------------------
    def emit(self, node):
        """(code, kind) for an expression node; constants are folded."""
        try:
            val, kind = self.const(node)
            return literal(val, kind), kind
        except NotConstant:
            pass
        tag = node[0]
        if tag == "neg":
            code, kind = self.emit(node[1])
            return f"(-{code})", kind
        if tag == "conv":
            return self.cast(self.emit(node[2]), node[1]), node[1]
        if tag == "bin":
            return self.arith(node[1], self.emit(node[2]), self.emit(node[3]))
        if tag == "pow":
            return self.power(node[1], node[2])
        if tag == "cmp":
            a, b = self.emit(node[2]), self.emit(node[3])
            kind = _promote(a[1], b[1])
            op = {"/=": "!="}.get(node[1], node[1])
            return f"({self.cast(a, kind)} {op} {self.cast(b, kind)})", "bool"
        if tag == "log":
            a, b = self.emit(node[2])[0], self.emit(node[3])[0]
            op = {".and.": "&&", ".or.": "||", ".eqv.": "==", ".neqv.": "!="}[node[1]]
            return f"({a} {op} {b})", "bool"
        if tag == "not":
            return f"(!{self.emit(node[1])[0]})", "bool"
        if tag == "ref":
            return self.ref(node[1], node[2])
        raise ValueError(node)

    @staticmethod
    def cast(item, kind):
        code, k = item
        if k == kind or kind == "bool":
            return code
        return f"(({_CTYPE[kind]})({code}))"

    def arith(self, op, a, b):
        ka, kb = a[1], b[1]
        if ka == kb == "int":
            if op == "/":
                return f"({a[0]}/{b[0]})", "int"
            return f"({a[0]}{op}{b[0]})", "int"
        kind = _promote(ka, kb)
        x, y = self.cast(a, kind), self.cast(b, kind)
        return f"{'D' if kind == 'double' else 'F'}{_OPS[op]}({x},{y})", kind

    def power(self, base_node, exp_node):
        base = self.emit(base_node)
        try:
            n, ek = self.const(exp_node)
        except NotConstant:
            n, ek = None, None
        if ek == "int":
            n = int(n)
            if base[1] == "int":
                assert n >= 0
                return self._chain(base, n, "int"), "int"
            code = self._chain(base, abs(n), base[1])
            if n < 0:
                one = "1.0" if base[1] == "double" else "1.0f"
                code = f"{'D' if base[1] == 'double' else 'F'}DIV({one},{code})"
            return code, base[1]
        e = self.emit(exp_node)
        if e[1] == "int":
            fn = "f2cu_powi_d" if base[1] == "double" else "f2cu_powi_f"
            return f"{fn}({base[0]},{e[0]})", base[1]
        kind = _promote(base[1], e[1])
        fn = "glibc_pow" if kind == "double" else "gfk_pow"
        return f"{fn}({self.cast(base, kind)},{self.cast(e, kind)})", kind

    def _chain(self, base, n, kind):
        if n == 0:
            return {"int": "1", "float": "1.0f", "double": "1.0"}[kind]
        memo = {1: base[0]}

        def build(m):
            if m in memo:
                return memo[m]
            d = POWI_TABLE[m] if m < len(POWI_TABLE) else m // 2
            a, b = build(m - d), build(d)
            memo[m] = (f"({a}*{b})" if kind == "int" else
                       f"{'D' if kind == 'double' else 'F'}MUL({a},{b})")
            return memo[m]
        return build(n)

    def ref(self, name, args):
        tr = self.tr
        if args is not None and name in self.routine.stmtfns:
            dummies, body = self.routine.stmtfns[name]
            env = {d: ("conv", self.scope.lookup(d).kind, a) for d, a in zip(dummies, args)}
            code, kind = self.emit(_subst(body, env))
            fk = self.scope.lookup(name).kind
            return self.cast((code, kind), fk), fk
        if args is not None and name in INTRINSICS and self.scope.lookup(name) is None:
            return self.intrinsic(name, args)
        if name in self.routine.ext_alias:
            name = self.routine.ext_alias[name]
        if args is not None and name in tr.routines and (self.scope.lookup(name) is None
                                                         or self.scope.lookup(name).dims is None):
            return tr.call_function(self, name, args)
        v = self.scope.lookup(name)
        if v is None:
            v = self.scope.ensure(name)
        if v.param is not None:
            return literal(*v.param), v.param[1]
        cname = tr.cname(v, self.routine)
        if args is None:
            if v.dims is not None:
                return cname, v.kind  # whole array (actual argument)
            return (f"(*{cname})" if v.is_arg and tr.by_pointer(v) else cname), v.kind
        return f"{cname}[{self.index(v, args)}]", v.kind

    def index(self, v, args):
        if len(args) != len(v.dims):
            raise SyntaxError(f"{v.name}: {len(args)} subscripts for {len(v.dims)} dims")
        terms, stride = [], None
        for (lo, hi), a in zip(v.dims, args):
            off = self.emit(("bin", "-", a, lo))
            assert off[1] == "int", (v.name, a)
            terms.append(off[0] if stride is None else f"({off[0]})*({stride})")
            if hi is not None:
                ext = self.emit(("bin", "+", ("bin", "-", hi, lo), ("num", "1")))[0]
                stride = ext if stride is None else f"({stride})*({ext})"
        return "+".join(terms)

    def intrinsic(self, name, args):
        vals = [self.emit(a) for a in args]
        kinds = [k for _, k in vals]
        if name in ("exp", "log", "alog", "sqrt", "log10", "alog10", "dexp", "dlog", "dsqrt"):
            (code, kind), = vals
            base = name.lstrip("ad") if name not in ("alog10",) else "log10"
            base = {"og": "log"}.get(base, base)
            if kind == "int":
                raise TypeError(f"{name} of an integer")
            if kind == "double":
                fn = {"exp": "glibc_exp", "log": "glibc_log", "sqrt": "__dsqrt_rn"}[base]
            else:
                fn = {"exp": "gfk_exp", "log": "gfk_log", "sqrt": "FSQRT", "log10": "gfk_log10"}[base]
            return f"{fn}({code})", kind
        if name in ("acos", "cos", "sin"):
            (code, kind), = vals
            if kind == "double":
                # Correctly rounded binary64 (CORE-MATH acos/cos, WOOF's wcr_sin):
                # glibc's own binary64 trig is LGPL IBM code, not transcribed.
                return {"acos": "uw_acos", "cos": "uw_cos", "sin": "wcr_sin"}[name] + f"({code})", kind
            if kind != "float":
                raise NotImplementedError(f"{name} of {kind}")
            return f"glibc_{name}f({code})", kind
        if name in ("max", "min", "amax1", "amin1", "max0", "min0", "dmax1", "dmin1"):
            kind = kinds[0]
            for k in kinds[1:]:
                kind = _promote(kind, k)
            big = name.startswith(("max", "amax", "dmax"))
            code = self.cast(vals[0], kind)
            for v in vals[1:]:
                other = self.cast(v, kind)
                # gfortran -O0: MAX(a, b) = a > b ? a : b, MIN(a, b) = a < b ? a : b,
                # folded left to right (measured: MAX(-0., 0.) = 0., MAX(1., NaN)
                # = NaN), which fmaxf/fminf do not promise on every backend.
                fn = "f2cu_" + ("max" if big else "min") + {"int": "i", "float": "f",
                                                            "double": "d"}[kind]
                code = f"{fn}({code},{other})"
            return code, kind
        if name in ("abs", "dabs", "iabs"):
            (code, kind), = vals
            return ({"int": f"abs({code})", "float": f"fabsf({code})",
                     "double": f"fabs({code})"}[kind]), kind
        if name == "sign":
            (a, ka), (b, kb) = vals
            if ka == "int":
                return f"(({b})>=0?abs({a}):-abs({a}))", "int"
            fn = "copysignf" if ka == "float" else "copysign"
            ab = "fabsf" if ka == "float" else "fabs"
            return f"{fn}({ab}({a}),{self.cast((b, kb), ka)})", ka
        if name in ("float", "real", "sngl"):
            return f"((float)({vals[0][0]}))", "float"
        if name == "dble":
            return f"((double)({vals[0][0]}))", "double"
        if name == "int":
            return f"((int)({vals[0][0]}))", "int"
        if name == "nint":
            fn = "roundf" if kinds[0] == "float" else "round"
            return f"((int){fn}({vals[0][0]}))", "int"
        if name == "mod":
            (a, ka), (b, kb) = vals
            if ka == kb == "int":
                return f"(({a})%({b}))", "int"
            fn = "fmodf" if _promote(ka, kb) == "float" else "fmod"
            return f"{fn}({a},{b})", _promote(ka, kb)
        raise NotImplementedError(f"intrinsic {name}")


INTRINSICS = {"acos", "cos", "sin", "exp", "log", "alog", "sqrt", "log10", "alog10", "dexp", "dlog", "dsqrt",
              "max", "min", "amax1", "amin1", "max0", "min0", "dmax1", "dmin1",
              "abs", "dabs", "iabs", "sign", "float", "real", "sngl", "dble", "int",
              "nint", "mod"}
_CTYPE = {"float": "float", "double": "double", "int": "int", "bool": "bool"}


def _promote(a, b):
    order = {"bool": 0, "int": 1, "float": 2, "double": 3}
    return a if order[a] >= order[b] else b


def _subst(node, env):
    if not isinstance(node, tuple):
        return node
    if node[0] == "ref" and node[2] is None and node[1] in env:
        return env[node[1]]
    if node[0] == "ref":
        return ("ref", node[1], None if node[2] is None else [_subst(a, env) for a in node[2]])
    return tuple(_subst(x, env) if isinstance(x, tuple) else x for x in node)


class NotConstant(Exception):
    pass


# --------------------------------------------------------------------------
# Translator
# --------------------------------------------------------------------------
@dataclass
class Routine:
    name: str
    lines: list
    args: list = field(default_factory=list)
    kind: str | None = None           # function result kind
    scope: Scope | None = None
    body: list = field(default_factory=list)
    ext_alias: dict = field(default_factory=dict)
    patches: dict = field(default_factory=dict)   # lineno -> replacement text
    assigned: set = field(default_factory=set)     # names a statement may write
    stmtfns: dict = field(default_factory=dict)    # statement functions


class Translator:
    def __init__(self, module_texts, struct_name="F2cuModule", implicit=True, prefix=""):
        self.implicit = implicit
        #: prepended to every generated routine's C name (no clash with CUDA's fmin etc.)
        self.prefix = prefix
        self.module = Scope(implicit=False)
        self.routines: dict[str, Routine] = {}
        self.struct_name = struct_name
        self.module_vars_used: set[str] = set()
        self.fatal_used = False
        self.dummy_bounds: dict = {}
        #: file-scope __constant__ copies of routines' DATA arrays
        self.constant_tables: list = []
        self.bound_from_module_param = False
        #: routines whose CALL must never execute (a branch the build excludes);
        #: reaching one records a fatal like wrf_error_fatal
        self.unreachable_calls: set = set()
        for text in module_texts:
            for _, _, stmt in logical_lines(enumerate(text.splitlines(), 1)):
                low = stmt.lower()
                if low.startswith(("module ", "end module", "contains")):
                    continue
                self.module.declare_line(stmt, self)

    # ---- constants --------------------------------------------------------
    def convert(self, val, src, dst):
        if src == dst:
            return val
        if dst == "int":
            if src in ("float", "double"):
                return int(val)  # truncation toward zero
            return int(val)
        if dst in ("float", "double"):
            if src == "bool":
                raise TypeError("logical to real")
            return round_to(Fraction(val), dst)
        if dst == "bool":
            return bool(val)
        return val

    def const(self, node, scope):
        tag = node[0]
        if tag == "num":
            text = node[1]
            if re.fullmatch(r"\d+", text):
                return int(text), "int"
            kind = "double" if re.search(r"[Dd]|_8$", text) else "float"
            return parse_literal(text.replace("_8", "").replace("_4", ""), kind), kind
        if tag == "bool":
            return node[1], "bool"
        if tag == "neg":
            v, k = self.const(node[1], scope)
            return -v, k
        if tag == "conv":
            v, k = self.const(node[2], scope)
            return self.convert(v, k, node[1]), node[1]
        if tag == "bin":
            a, ka = self.const(node[2], scope)
            b, kb = self.const(node[3], scope)
            op = node[1]
            if ka == kb == "int":
                if op == "/":
                    q = abs(a) // abs(b)
                    return (q if (a >= 0) == (b >= 0) else -q), "int"
                return {"+": a + b, "-": a - b, "*": a * b}[op], "int"
            kind = _promote(ka, kb)
            a, b = Fraction(a), Fraction(b)
            r = {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else None}[op]
            if r is None:
                raise ZeroDivisionError("constant division by zero")
            return round_to(r, kind), kind
        if tag == "pow":
            b, kb = self.const(node[1], scope)
            e, ke = self.const(node[2], scope)
            if ke == "int":
                if kb == "int":
                    return b ** e, "int"
                # gfortran folds constant integer powers exactly then rounds?  No:
                # it folds with MPFR at the kind's precision per multiplication of
                # its chain, which for exact chains matches repeated IEEE products.
                return self._fold_chain(b, int(e), kb), kb
            kind = _promote(kb, ke)
            return _mp_fn("pow", [Fraction(b), Fraction(e)], kind), kind
        if tag == "ref":
            name, args = node[1], node[2]
            if args is not None and name in INTRINSICS and scope.lookup(name) is None:
                vals = [self.const(a, scope) for a in args]
                return self._fold_intrinsic(name, vals)
            v = scope.lookup(name)
            if v is not None and v.param is not None and args is None:
                return v.param
            raise NotConstant(name)
        raise NotConstant(tag)

    def _fold_chain(self, b, n, kind):
        memo = {1: Fraction(b)}

        def build(m):
            if m in memo:
                return memo[m]
            d = POWI_TABLE[m] if m < len(POWI_TABLE) else m // 2
            memo[m] = round_to(build(m - d) * build(d), kind)
            return memo[m]
        if n == 0:
            return Fraction(1)
        r = build(abs(n))
        return round_to(1 / r, kind) if n < 0 else r

    def _fold_intrinsic(self, name, vals):
        kinds = [k for _, k in vals]
        if name in ("exp", "log", "alog", "sqrt", "log10", "alog10", "dexp", "dlog", "dsqrt"):
            (v, k), = vals
            base = {"alog": "log", "alog10": "log10", "dexp": "exp", "dlog": "log",
                    "dsqrt": "sqrt"}.get(name, name)
            return _mp_fn(base, [Fraction(v)], k), k
        if name in ("max", "min", "amax1", "amin1", "max0", "min0", "dmax1", "dmin1"):
            kind = kinds[0]
            for k in kinds[1:]:
                kind = _promote(kind, k)
            xs = [self.convert(v, k, kind) for v, k in vals]
            return (max(xs) if name.startswith(("max", "amax", "dmax")) else min(xs)), kind
        if name in ("abs", "dabs", "iabs"):
            (v, k), = vals
            return abs(v), k
        if name in ("float", "real", "sngl"):
            return self.convert(vals[0][0], kinds[0], "float"), "float"
        if name == "dble":
            return self.convert(vals[0][0], kinds[0], "double"), "double"
        if name == "int":
            return int(vals[0][0]), "int"
        if name == "sign":
            (a, ka), (b, kb) = vals
            return (abs(a) if b >= 0 else -abs(a)), ka
        raise NotConstant(name)

    # ---- routines ---------------------------------------------------------
    def add_source(self, text, names, *, patches=None, ext_alias=None):
        found = find_routines(text)
        for name in names:
            if name not in found:
                raise KeyError(f"routine {name} not in source")
            r = Routine(name, found[name])
            r.patches = (patches or {}).get(name, {})
            r.ext_alias = (ext_alias or {}).get(name, {})
            self.routines[name] = r
        for name in names:
            self._parse(self.routines[name])
        self.analyse_writes()

    def add_synthetic(self, name, body_lines, first_line=1):
        """A routine made of statements cut from inside another (module constants)."""
        lines = [(first_line - 1, f"      SUBROUTINE {name}")]
        lines += [(first_line + k, raw) for k, raw in enumerate(body_lines)]
        lines.append((first_line + len(body_lines), f"      END SUBROUTINE {name}"))
        r = Routine(name, lines)
        self.routines[name] = r
        self._parse(r)
        self.analyse_writes()

    def prototype(self, r):
        params = [f"{self.struct_name} &mod"]
        for a in r.args:
            if a in r.ext_alias:
                continue
            v = r.scope.vars[a]
            if v.kind == "char":
                raise NotImplementedError(f"character dummy {a} of {r.name}")
            ctype = _CTYPE[v.kind]
            if v.dims is not None:
                params.append(f"{'' if a in r.assigned else 'const '}{ctype} *{a}")
            elif self.by_value(r, a):
                params.append(f"{ctype} {a}")
            else:
                params.append(f"{ctype} &{a}")
        ret = _CTYPE[r.kind] if r.kind else "void"
        return f"__device__ {ret} {self.prefix}{r.name}({', '.join(params)})"

    def _parse(self, r: Routine):
        lines = [(n, r.patches.get(n, raw)) for n, raw in r.lines]
        stmts = list(logical_lines(lines))
        n0, _, header = stmts[0]
        m = _HEADER.match(header.strip())
        if not m:
            raise SyntaxError(f"header {header!r}")
        r.args = [a.strip().lower() for a in split_top(m.group(4) or "") if a.strip()]
        scope = Scope(self.module, implicit=self.implicit)
        r.scope = scope
        for a in r.args:
            if a not in r.ext_alias:
                scope.vars[a] = Var(a, "int" if a[0] in "ijklmn" else "float",
                                    is_arg=True, explicit=False)
        if m.group(2).lower() == "function":
            r.kind = _kind_of(m.group(1)) if m.group(1) else None
            scope.vars[r.name] = Var(r.name, r.kind or ("int" if r.name[0] in "ijklmn" else "float"),
                                     explicit=bool(m.group(1)))
        body = []
        exec_started = False
        for n, label, stmt in stmts[1:-1]:
            if not exec_started and label is None and scope.declare_line(stmt, self):
                continue
            sf = re.match(r"^(\w+)\s*\(([\w\s,]+)\)\s*=(.*)$", stmt.strip())
            if not exec_started and sf:
                v = scope.lookup(sf.group(1).lower())
                if v is not None and v.dims is None and sf.group(3)[:1] != "=":
                    dummies = [d.strip().lower() for d in sf.group(2).split(",")]
                    r.stmtfns[sf.group(1).lower()] = (dummies, parse_expr(sf.group(3)))
                    continue
            exec_started = True
            body.append((n, label, stmt))
        for a in r.args:
            if a in scope.vars:
                scope.vars[a].is_arg = True
        if r.kind is None and m.group(2).lower() == "function":
            r.kind = scope.vars[r.name].kind
        r.body = body

    def _call_sites(self, r):
        """(callee, [actual base name or None]) for every CALL and function use."""
        sites = []
        names = set(self.routines) | set(r.ext_alias)
        for _, _, stmt in r.body:
            low = stmt.lower()
            for m in re.finditer(r"\b(\w+)\s*\(", low):
                name = m.group(1)
                if name not in names or name in r.stmtfns:
                    continue
                v = r.scope.vars.get(name)
                if v is not None and v.dims is not None:
                    continue
                k = self._close_paren(low, m.end() - 1)
                actuals = []
                for arg in split_top(low[m.end():k]):
                    mm = re.fullmatch(r"\s*(\w+)\s*(\(.*\))?\s*", arg)
                    actuals.append(mm.group(1) if mm and not re.match(r"\d", mm.group(1)) else None)
                sites.append((r.ext_alias.get(name, name), actuals))
        return sites

    def analyse_writes(self):
        """Which names each routine may write: assignments and DO variables,
        plus every actual argument bound to a dummy its callee may write
        (to a fixed point across the call tree)."""
        sites = {}
        for r in self.routines.values():
            for _, _, stmt in r.body:
                low = stmt.lower().strip()
                m = re.match(r"^do\s+(?:\d+\s*,?\s*)?(\w+)\s*=", low)
                if m:
                    r.assigned.add(m.group(1))
                if low.startswith("if") and re.match(r"^if\s*\(", low):
                    k = self._close_paren(low, low.index("("))
                    low = low[k + 1:].strip()
                parts = self._split_assign(low)
                if parts and not re.match(r"^(do\b|if\s*\(|call\b)", low):
                    mm = re.match(r"^(\w+)", parts[0])
                    if mm:
                        r.assigned.add(mm.group(1))
            sites[r.name] = self._call_sites(r)
        changed = True
        while changed:
            changed = False
            for r in self.routines.values():
                for callee, actuals in sites[r.name]:
                    c = self.routines[callee]
                    dummies = [d for d in c.args if d not in c.ext_alias]
                    for d, actual in zip(dummies, actuals):
                        if actual and d in c.assigned and actual not in r.assigned:
                            r.assigned.add(actual)
                            changed = True

    def by_value(self, r, name):
        v = r.scope.vars[name]
        return v.dims is None and name not in r.assigned

    def by_pointer(self, v):
        return False  # scalar dummies are C++ references

    def cname(self, v, routine):
        if v.name in self.module.vars and routine.scope.vars.get(v.name) is None:
            self.module_vars_used.add(v.name)
            return f"mod.{v.name}"
        if routine.kind is not None and v.name == routine.name:
            return "f2cu_result"
        if v.dims is not None and v.data is not None and not v.is_arg:
            return f"{self.prefix}{routine.name}_{v.name}"
        return v.name

    def call_function(self, em, name, args):
        callee = self.routines[name]
        return f"{self.prefix}{name}({self._actuals(em, callee, args)})", callee.kind

    def _actuals(self, em, callee, args, pre=None):
        out = ["mod"]
        dummies = [a for a in callee.args if a not in callee.ext_alias]
        actual_args = [a for d, a in zip(callee.args, args) if d not in callee.ext_alias]
        for d, a in zip(dummies, actual_args):
            dv = callee.scope.vars[d]
            if dv.dims is not None:
                out.append(self._array_actual(em, a))
            elif self.by_value(callee, d):
                out.append(em.cast(em.emit(a), dv.kind))
            else:
                code, kind = em.emit(a)
                if a[0] == "ref" and kind == dv.kind and self._is_lvalue(em, a):
                    out.append(code)
                else:
                    if pre is None:
                        raise SyntaxError(f"expression actual {a} for reference dummy "
                                          f"{d} of function {callee.name}")
                    tmp = f"f2cu_t{len(pre)}"
                    pre.append(f"{_CTYPE[dv.kind]} {tmp} = {em.cast((code, kind), dv.kind)};")
                    out.append(tmp)
        return ", ".join(out)

    def _is_lvalue(self, em, a):
        v = em.scope.lookup(a[1])
        return v is not None and v.param is None and a[1] not in INTRINSICS

    def _array_actual(self, em, a):
        assert a[0] == "ref", a
        v = em.scope.lookup(a[1])
        name = self.cname(v, em.routine)
        if a[2] is None:
            return name
        return f"&{name}[{em.index(v, a[2])}]"

    # ---- statements -------------------------------------------------------
    def emit_routine(self, r: Routine):
        em = Emitter(self, r.scope, r)
        ret = _CTYPE[r.kind] if r.kind else "void"
        out = [self.prototype(r) + " {"]
        for v in r.scope.vars.values():
            if v.is_arg or v.param is not None or v.kind in ("char", "external"):
                continue
            if r.kind is not None and v.name == r.name:
                out.append(f"    {ret} f2cu_result;")
                continue
            ctype = _CTYPE[v.kind]
            if v.dims is not None:
                size = self._local_size(v, r)
                if v.data is not None:
                    vals = ", ".join(literal(x, v.kind) for x in v.data)
                    # DATA tables live in the constant space, not in every
                    # thread's frame (they are never written: DATA objects
                    # are implicitly SAVEd, and assigning one is refused).
                    self.constant_tables.append(
                        f"__constant__ {ctype} {self.prefix}{r.name}_{v.name}[{size}] = {{{vals}}};")
                else:
                    out.append(f"    {ctype} {v.name}[{size}];")
            elif v.data is not None:
                out.append(f"    const {ctype} {v.name} = {literal(v.data[0], v.kind)};")
            else:
                out.append(f"    {ctype} {v.name};")
        out += self._block(em, r, r.body, 1)
        if r.kind is not None:
            out.append("    return f2cu_result;")
        out.append("}")
        return out

    def _block(self, em, r, stmts, depth):
        out = []
        ind = "    " * depth
        i = 0
        # Backward GO TO: label L ... GO TO L at the same level becomes for(;;).
        while i < len(stmts):
            n, label, stmt = stmts[i]
            low = stmt.lower().strip()
            if label is not None:
                tail = self._find_goto(stmts, i, label)
                if tail is not None and not re.match(r"^continue$", low) or tail is not None:
                    inner = stmts[i:tail]
                    inner = [(inner[0][0], None, inner[0][2])] + inner[1:]
                    out.append(f"{ind}for (;;) {{ // {n}: label {label} ... GO TO {label}")
                    out += self._block(em, r, inner, depth + 1)
                    out.append(f"{ind}}}")
                    i = tail + 1
                    continue
            consumed, code = self._statement(em, r, stmts, i, depth)
            out += code
            i += consumed
        return out

    @staticmethod
    def _find_goto(stmts, start, label):
        depth = 0
        for j in range(start, len(stmts)):
            low = stmts[j][2].lower().strip()
            if re.match(r"^go\s*to\s+%d$" % label, low):
                return j if depth == 0 else None
        return None

    def _statement(self, em, r, stmts, i, depth):
        n, label, stmt = stmts[i]
        ind = "    " * depth
        low = stmt.lower().strip()
        m = re.match(r"^do\s+(\d+)\s*,?\s*(\w+)\s*=\s*(.+)$", low) or \
            re.match(r"^do\s+()(\w+)\s*=\s*(.+)$", low)
        if m:
            end_label = int(m.group(1)) if m.group(1) else None
            var = m.group(2)
            bounds = split_top(m.group(3))
            j, d = i + 1, 0
            while True:
                if j >= len(stmts):
                    raise SyntaxError(f"unterminated DO at {n}")
                nj, lj, sj = stmts[j]
                lowj = sj.lower().strip()
                if end_label is None:
                    if re.match(r"^do\s+(?!\d)\w+\s*=", lowj):
                        d += 1
                    elif re.fullmatch(r"end\s*do", lowj):
                        if d == 0:
                            break
                        d -= 1
                else:
                    if lj == end_label:
                        break
                j += 1
            body = stmts[i + 1:j]
            if end_label is not None:
                last = stmts[j]
                if not re.fullmatch(r"continue|end\s*do", last[2].lower().strip()):
                    body = body + [(last[0], None, last[2])]
                # shared termination: count outer DOs ending on the same label
            v = em.scope.ensure(var)
            lo, hi = (em.emit(parse_expr(b))[0] for b in bounds[:2])
            step = em.emit(parse_expr(bounds[2]))[0] if len(bounds) > 2 else "1"
            cv = self.cname(v, r)
            cmp = "<=" if not step.startswith("(-") and not step.startswith("-") else ">="
            out = [f"{ind}{{ int f2cu_e{depth} = {hi}; "
                   f"for ({cv} = {lo}; {cv} {cmp} f2cu_e{depth}; {cv} += {step}) {{ // {n}"]
            out += self._block(em, r, body, depth + 1)
            out.append(f"{ind}}} }}")
            # Shared-label termination (DO 100 j ... DO 100 i ... 100 CONTINUE)
            consumed = j - i + 1
            return consumed, out
        m = re.match(r"^if\s*\((.*)\)\s*then$", low)
        if m and self._balanced_if(low):
            branches, j = self._if_chain(stmts, i)
            out = []
            for k, (cond, body) in enumerate(branches):
                head = "if" if k == 0 else "} else if"
                if cond is None:
                    out.append(f"{ind}}} else {{")
                else:
                    c = em.emit(parse_expr(cond))[0]
                    out.append(f"{ind}{head} ({c}) {{ // {stmts[i][0]}")
                out += self._block(em, r, body, depth + 1)
            out.append(f"{ind}}}")
            return j - i + 1, out
        if low.startswith("if") and re.match(r"^if\s*\(", low):
            k = self._close_paren(low, low.index("("))
            cond = em.emit(parse_expr(stmt[low.index("(") + 1:k]))[0]
            rest = stmt[k + 1:].strip()
            _, code = self._statement(em, r, [(n, None, rest)], 0, 0)
            return 1, [f"{ind}if ({cond}) {{"] + ["    " * (depth + 1) + c.strip()
                                                  for c in code] + [f"{ind}}}"]
        if low in ("continue",):
            return 1, []
        if low == "return":
            return 1, [f"{ind}return{' f2cu_result' if r.kind else ''}; // {n}"]
        if re.match(r"^(print|write)\b", low) or re.match(r"^format\s*\(", low):
            return 1, [f"{ind}// {n}: {low.split()[0]} omitted"]
        m = re.match(r"^go\s*to\s+(\d+)$", low)
        if m:
            raise NotImplementedError(f"GO TO {m.group(1)} at {n} outside the loop pattern")
        if low.startswith("call "):
            m = re.match(r"^call\s+(\w+)\s*(?:\((.*)\))?$", stmt.strip(), re.I)
            name = m.group(1).lower()
            if name in ("wrf_debug", "wrf_message"):
                return 1, [f"{ind}// {n}: {name} omitted"]
            if name == "wrf_error_fatal" or name in self.unreachable_calls:
                # A kernel cannot abort: record the first fatal line; the
                # launcher raises on it.
                self.fatal_used = True
                return 1, [f"{ind}if (mod.f2cu_fatal == 0) mod.f2cu_fatal = {n}; // {n}: wrf_error_fatal"]
            if name not in self.routines:
                raise KeyError(f"CALL {name} at {n}: routine not translated")
            callee = self.routines[name]
            args = [parse_expr(a) for a in split_top(m.group(2) or "")]
            pre = []
            actuals = self._actuals(em, callee, args, pre)
            body = [f"{ind}{self.prefix}{name}({actuals}); // {n}"]
            if pre:
                return 1, [f"{ind}{{"] + [f"{ind}    {p}" for p in pre] + \
                          ["    " + b for b in body] + [f"{ind}}}"]
            return 1, body
        parts = self._split_assign(stmt)
        if parts:
            lhs, rhs = parts
            lnode = parse_expr(lhs)
            name = lnode[1]
            v = em.scope.lookup(name)
            if v is None:
                v = em.scope.ensure(name)
            if v.kind == "char":
                return 1, [f"{ind}// {n}: character assignment omitted"]
            if v.data is not None:
                raise NotImplementedError(f"{name} is DATA-initialised (implicitly SAVEd) "
                                          f"and assigned at {n}")
            code, kind = em.emit(parse_expr(rhs))
            if v.dims is not None and lnode[2] is None:
                size = self._size(v, r)
                return 1, [f"{ind}for (int f2cu_q = 0; f2cu_q < {size}; ++f2cu_q) "
                           f"{self.cname(v, r)}[f2cu_q] = {em.cast((code, kind), v.kind)}; // {n}"]
            target, tkind = em.emit(lnode) if not (r.kind and name == r.name) else ("f2cu_result", r.kind)
            if target.startswith("(*"):
                target = target[2:-1]
            return 1, [f"{ind}{target} = {self._assign_cast(code, kind, tkind)}; // {n}"]
        raise NotImplementedError(f"statement at {n}: {stmt!r}")

    def _local_size(self, v, r):
        """A local array's element count; an automatic array (extent from a
        dummy) takes the dummy's bound from ``self.dummy_bounds``."""
        size = 1
        for lo, hi in v.dims:
            ext = []
            for node in (lo, hi):
                try:
                    ext.append(self.const(node, r.scope)[0])
                except NotConstant:
                    key = (r.name, node[1]) if node[0] == "ref" else None
                    mod = self.module.vars.get(node[1]) if key else None
                    if key not in self.dummy_bounds and mod is not None and mod.param \
                            and self.bound_from_module_param:
                        # a dummy named after a module PARAMETER, passed that
                        # PARAMETER by every caller (checked by the caller list)
                        self.dummy_bounds[key] = mod.param[0]
                    if key not in self.dummy_bounds:
                        raise NotImplementedError(
                            f"{r.name}: automatic array {v.name} needs a bound for {node}")
                    ext.append(self.dummy_bounds[key])
            size *= ext[1] - ext[0] + 1
        return size

    def _size(self, v, r):
        size = 1
        for lo, hi in v.dims:
            size *= self.const(hi, r.scope)[0] - self.const(lo, r.scope)[0] + 1
        return size

    @staticmethod
    def _assign_cast(code, kind, tkind):
        if kind == tkind:
            return code
        if tkind == "int" and kind in ("float", "double"):
            return f"((int)({code}))"
        return f"(({_CTYPE[tkind]})({code}))"

    @staticmethod
    def _close_paren(text, start):
        d = 0
        for k in range(start, len(text)):
            d += text[k] == "("
            d -= text[k] == ")"
            if d == 0:
                return k
        raise SyntaxError(text)

    def _balanced_if(self, low):
        k = self._close_paren(low, low.index("("))
        return low[k + 1:].strip() == "then"

    def _if_chain(self, stmts, i):
        branches, cond = [], None
        low = stmts[i][2].strip()
        k = self._close_paren(low.lower(), low.lower().index("("))
        cond = low[low.index("(") + 1:k]
        body, d, j = [], 0, i + 1
        while j < len(stmts):
            s = stmts[j][2].strip()
            sl = s.lower()
            if re.match(r"^if\s*\(.*\)\s*then$", sl) and self._balanced_if(sl):
                d += 1
            elif re.fullmatch(r"end\s*if", sl):
                if d == 0:
                    branches.append((cond, body))
                    return branches, j
                d -= 1
            elif d == 0 and re.match(r"^else\s*if\s*\(", sl):
                branches.append((cond, body))
                kk = self._close_paren(sl, sl.index("("))
                cond, body = s[s.index("(") + 1:kk], []
                j += 1
                continue
            elif d == 0 and sl == "else":
                branches.append((cond, body))
                cond, body = None, []
                j += 1
                continue
            body.append(stmts[j])
            j += 1
        raise SyntaxError("unterminated IF")

    @staticmethod
    def _split_assign(text):
        depth = 0
        for i, ch in enumerate(text):
            depth += ch == "("
            depth -= ch == ")"
            if ch == "=" and depth == 0 and text[i - 1] not in "<>/=" and \
                    (i + 1 >= len(text) or text[i + 1] != "="):
                return text[:i].strip(), text[i + 1:].strip()
        return None

    # ---- module struct ----------------------------------------------------
    def module_struct(self, extra=()):
        fields = []
        for name in sorted(self.module_vars_used | set(extra)):
            v = self.module.vars[name]
            ctype = _CTYPE[v.kind]
            if v.dims is not None:
                fields.append(f"    {ctype} {name}[{self._size(v, Routine('m', [], scope=self.module))}];")
            else:
                fields.append(f"    {ctype} {name};")
        if self.fatal_used:
            fields.append("    int f2cu_fatal;  // first source line of a wrf_error_fatal, or 0")
        return [f"struct {self.struct_name} {{"] + fields + ["};"]


HELPERS = r"""
// Fortran MAX/MIN as gfortran -O0 evaluates them (first argument if it
// compares greater/less, else the second), including signed zeros and NaN.
__device__ __forceinline__ float f2cu_maxf(float a, float b) { return a > b ? a : b; }
__device__ __forceinline__ float f2cu_minf(float a, float b) { return a < b ? a : b; }
__device__ __forceinline__ double f2cu_maxd(double a, double b) { return a > b ? a : b; }
__device__ __forceinline__ double f2cu_mind(double a, double b) { return a < b ? a : b; }
__device__ __forceinline__ int f2cu_maxi(int a, int b) { return a > b ? a : b; }
__device__ __forceinline__ int f2cu_mini(int a, int b) { return a < b ? a : b; }
// libgcc's __powisf2 / __powidf2: what gfortran -O0 calls for x**n with a
// variable integer n.
__device__ float f2cu_powi_f(float x, int m) {
    unsigned int n = m < 0 ? -(unsigned int)m : (unsigned int)m;
    float y = (n % 2u) ? x : 1.0f;
    while (n >>= 1) { x = FMUL(x, x); if (n % 2u) y = FMUL(y, x); }
    return m < 0 ? FDIV(1.0f, y) : y;
}
__device__ double f2cu_powi_d(double x, int m) {
    unsigned int n = m < 0 ? -(unsigned int)m : (unsigned int)m;
    double y = (n % 2u) ? x : 1.0;
    while (n >>= 1) { x = DMUL(x, x); if (n % 2u) y = DMUL(y, x); }
    return m < 0 ? DDIV(1.0, y) : y;
}
"""
