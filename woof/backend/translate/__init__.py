"""Translate woof's CUDA C++ kernel units into Metal Shading Language.

The ``.cu`` files and the hex inline strings stay the only source.  This
module turns the exact text NVRTC would compile
(``woof.core.kernels.module_source``) into MSL, and says what each kernel's
parameters are so the launcher can bind them.

Pipeline
--------
1. **Preprocess** with the host clang (``clang -E -P``).  Macros, integer
   defines and constants expand, so every token the rewriter edits has a
   real location.  CUDA's keywords become ``annotate`` attributes
   (:data:`PARSE_DEFINES`) that survive into the AST.
2. **Parse** the preprocessed text as C++17 with libclang, against
   ``cuda_decls.h`` (CUDA's builtins and libm, declarations only).  Any
   error is a :class:`TranslationError`: a unit is translated completely or
   refused by name, never half-translated.
3. **Rewrite**, guided by types:

   * every ``extern "C" __global__`` kernel gets an MSL ``[[kernel]]``
     signature (``device`` pointers, scalars through ``constant`` copies,
     the thread-position builtins as parameters);
   * ``double`` becomes ``f64`` (``woof/backend/metal/prelude.h``): binary64
     words in software, bit-exact; a floating literal of type ``double``
     becomes its exact binary64 word; every implicit conversion out of
     ``double`` (to float, to an integer, to bool) is spelled as an explicit
     cast, because ``f64`` converts out only explicitly;
   * a compound assignment whose computation type is ``double`` but whose
     target is not (``float f; f += d;``) is spelled out as C++ evaluates it;
   * ``__shared__``/``__constant__`` become ``threadgroup``/``constant``;
     ``long long`` becomes ``long``; identifiers that are MSL keywords are
     renamed; ``printf`` calls are dropped.

Refusals (:class:`TranslationError`) name the construct: ``goto``,
recursion, ``long double``, a device function that reads ``threadIdx``.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import struct
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
DECLS = HERE / "cuda_decls.h"
#: Stand-ins for the system headers kernels include; see the files.
INCLUDE = HERE / "include"
METAL_DIR = HERE.parent / "metal"

_GLOBAL = "woof_global"
_SHARED = "woof_shared"
_CONSTANT = "woof_constant"
_SPACE_HINT = "woof_space:"

#: The CUDA keywords, as the preprocessor sees them.
PARSE_DEFINES = (
    f'-D__global__=__attribute__((annotate("{_GLOBAL}")))',
    f'-D__shared__=__attribute__((annotate("{_SHARED}")))',
    f'-D__constant__=__attribute__((annotate("{_CONSTANT}")))',
    "-D__device__=", "-D__host__=", "-D__forceinline__=inline",
    "-D__noinline__=__attribute__((noinline))", "-D__restrict__=", "-D__launch_bounds__(...)=",
    "-D__CUDACC__=1", "-D__CUDACC_RTC__=1", "-D__WOOF_METAL__=1", "-D__align__(n)=alignas(n)",
)

#: Where a unit's own `#include "sibling.cuh"` resolves (NVRTC gets the
#: same directory as an include path from the loaders that need it).
KERNEL_INCLUDE_DIRS = (HERE.parents[1] / "core" / "kernels",)

#: MSL keywords and reserved names a CUDA identifier may collide with.
METAL_RESERVED = frozenset({
    "kernel", "vertex", "fragment", "device", "constant", "thread",
    "threadgroup", "threadgroup_imageblock", "ray_data", "object_data",
    "half", "sampler", "texture", "visible", "stage_in", "patch", "mesh",
    "uchar", "ushort", "uint", "ulong", "bfloat", "vec", "packed_float3",
})

#: struct format and size of each kernel scalar type, by canonical spelling.
SCALAR_FORMATS = {
    "float": "f", "double": "d", "int": "i", "unsigned int": "I",
    "long": "q", "unsigned long": "Q", "long long": "q",
    "unsigned long long": "Q", "short": "h", "unsigned short": "H",
    "char": "b", "signed char": "b", "unsigned char": "B", "bool": "?",
}


#: CUDA vector types whose Metal namesake differs: float3 is 12 bytes in
#: CUDA and 16 in MSL (packed_float3 is 12); MSL has no binary64 vectors.
_VECTOR_RENAMES = {"float3": "packed_float3", "make_float3": "make_packed_float3",
                   "double2": "__woof_double2", "double3": "__woof_double3",
                   "double4": "__woof_double4", "make_double2": "__woof_make_double2",
                   "make_double3": "__woof_make_double3", "make_double4": "__woof_make_double4"}


class TranslationError(RuntimeError):
    """A unit the translator refuses, with the construct named."""


@dataclass(frozen=True)
class Param:
    name: str
    pointer: bool
    #: Canonical element type (pointer) or value type (scalar), C spelling.
    ctype: str
    #: Pointer: the pointee is const (the buffer is never written).
    const: bool = False
    #: Scalar: struct format for packing; None for a by-value struct.
    fmt: str | None = None
    size: int = 0
    #: Scalar: byte offset in the kernel's packed scalar struct.
    offset: int = 0
    #: A module global's name: bound by the loader, not passed by the caller.
    global_name: str | None = None


#: Metal binds at most 31 buffers (0..30).  A kernel with more pointer
#: parameters than fit beside its scalar struct reads them from a table.
MAX_DIRECT_POINTERS = 29


@dataclass(frozen=True)
class KernelSignature:
    name: str
    params: tuple[Param, ...]
    #: Bytes of ``extern __shared__`` dynamic threadgroup memory requested.
    dynamic_shared: bool = False
    #: "direct": pointer i at slot i.  "table": every pointer's GPU address
    #: in a struct at ``table_slot``.
    pointer_mode: str = "direct"
    table_slot: int = -1
    #: Slot and size of the packed scalar struct (-1 / 0 when none).
    scalar_slot: int = -1
    scalar_size: int = 0
    #: Reaches memory through integer addresses (pointer tables built on the
    #: host): MLX cannot see those buffers, so the launch is serialised.
    indirect: bool = False


@dataclass
class TranslatedUnit:
    msl: str
    kernels: dict[str, KernelSignature] = field(default_factory=dict)
    #: sha256 of the MSL text: the Metal library cache key.
    key: str = ""
    #: module globals: name -> {"nbytes", "ctype", "space"}
    globals: dict = field(default_factory=dict)


# ---------------------------------------------------------------- toolchain

@lru_cache(maxsize=1)
def _clang() -> str:
    path = shutil.which("clang") or shutil.which("clang++")
    if path is None:
        raise TranslationError("the CUDA-to-Metal translator needs clang on PATH "
                               "(Xcode Command Line Tools)")
    return path


@lru_cache(maxsize=1)
def _index():
    import clang.cindex as ci
    # The process-lifetime Index outlives libclang at interpreter shutdown;
    # its destructor would then call into an unloaded library.
    ci.Index.__del__ = lambda self: None
    return ci.Index.create()


def preprocess(source: str, defines=()) -> str:
    """``defines``: the -D macros the unit's NVRTC options carry."""
    result = subprocess.run(
        [_clang(), "-E", "-P", "-x", "c++", "-std=c++17", "-nostdinc", "-nostdinc++",
         "-isystem", str(INCLUDE), *[f"-I{d}" for d in KERNEL_INCLUDE_DIRS],
         "-Wno-pragma-once-outside-header", *PARSE_DEFINES, *[f"-D{d}" for d in defines], "-"],
        input=source, capture_output=True, text=True)
    if result.returncode != 0:
        raise TranslationError("preprocessing failed:\n" + result.stderr[-4000:])
    return result.stdout


# ---------------------------------------------------------------- inline PTX

_ASM = re.compile(
    r'\basm\s*(?:volatile\s*)?\(\s*((?:"(?:[^"\\]|\\.)*"\s*)+)'
    r':\s*"=([a-z])"\s*\(([^()]*)\)\s*'
    r':\s*((?:"[a-z]"\s*\([^()]*\)\s*,?\s*)+)\)\s*;')
_ASM_IN = re.compile(r'"[a-z]"\s*\(([^()]*)\)')
_PTX_CMP = {"eq": "(({a}) == ({b}))", "ne": "(({a}) < ({b}) || ({a}) > ({b}))",
            "lt": "(({a}) < ({b}))", "le": "(({a}) <= ({b}))",
            "gt": "(({a}) > ({b}))", "ge": "(({a}) >= ({b}))"}


def rewrite_ptx_asm(text: str, unit: str) -> str:
    """Inline PTX has no Metal meaning; woof's kernels use a handful of
    idioms that pin one IEEE operation, and each becomes the C++ the
    prelude defines with the same rounding.  Anything else is refused."""
    def one(m):
        template = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1)))
        template = template.replace("\\n", " ").replace("\\t", " ").replace("%%", "%")
        body = re.sub(r"\.reg\s+\.pred\s+%?\w+\s*;", "", template)
        body = " ".join(body.replace("{", " ").replace("}", " ").split())
        out = m.group(3).strip()
        ins = [x.strip() for x in _ASM_IN.findall(m.group(4))]
        a = ins[0] if ins else None
        b = ins[1] if len(ins) > 1 else None
        mm = re.fullmatch(r"(add|sub|mul|div)\.rn\.f32 %0, ?%1, ?%2;", body)
        if mm:
            return f"{out} = __f{mm.group(1)}_rn({a}, {b});"
        mm = re.fullmatch(r"setp\.(eq|ne|lt|le|gt|ge)\.f32 %?\w+, ?%1, ?%2; selp\.u32 %0, ?1, ?0, ?%?\w+;", body)
        if mm:
            return f"{out} = {_PTX_CMP[mm.group(1)].format(a=a, b=b)} ? 1u : 0u;"
        mm = re.fullmatch(r"setp\.(eq|ne|lt|le|gt|ge)\.f32 %?\w+, ?%1, ?%2; "
                          r"selp\.f32 %0, ?0f3F800000, ?0f00000000, ?%?\w+;", body)
        if mm:
            return f"{out} = {_PTX_CMP[mm.group(1)].format(a=a, b=b)} ? 1.0f : 0.0f;"
        if re.fullmatch(r"min\.f32 %0, ?%1, ?%2;", body):
            return f"{out} = fminf({a}, {b});"
        if re.fullmatch(r"max\.f32 %0, ?%1, ?%2;", body):
            return f"{out} = fmaxf({a}, {b});"
        if re.fullmatch(r"cvt\.rn\.f32\.f64 %0, ?%1;", body):
            return f"{out} = __double2float_rn({a});"
        raise TranslationError(f"{unit}: inline PTX with no Metal translation: {body!r}")
    out = _ASM.sub(one, text)
    if re.search(r"\basm\s*(volatile\s*)?\(", out):
        line = out[:re.search(r"\basm\s*(volatile\s*)?\(", out).start()].count("\n") + 1
        raise TranslationError(f"{unit}: line {line}: inline asm in a form the translator does not read")
    return out


def parse(text: str):
    import clang.cindex as ci
    tu = _index().parse(
        "unit.cpp",
        args=["-x", "c++", "-std=c++17", "-nostdinc", "-include", str(DECLS),
              "-Wno-everything"],
        unsaved_files=[("unit.cpp", text)],
        options=ci.TranslationUnit.PARSE_SKIP_FUNCTION_BODIES * 0)
    errors = [d for d in tu.diagnostics if d.severity >= ci.Diagnostic.Error]
    if errors:
        lines = text.splitlines()
        shown = []
        for d in errors[:8]:
            line = d.location.line
            ctx = lines[line - 1].strip() if 0 < line <= len(lines) else ""
            shown.append(f"  {line}:{d.location.column}: {d.spelling}\n      {ctx[:160]}")
        raise TranslationError("the unit does not parse as C++ against cuda_decls.h:\n"
                               + "\n".join(shown))
    return tu


# ---------------------------------------------------------------- edits

class Edits:
    """Byte-range edits on one text, applied in one pass.

    Replacements must not overlap each other.  Insertions at the same
    offset are ordered by ``rank``: an outer wrap's opening text before an
    inner one's, an inner wrap's closing text before an outer one's.

    A *region* is a range rendered by a callback (one copy per address-space
    instance of a function): the callback receives ``render(extra)``, which
    returns the range's text with every edit inside it applied plus the
    extra insertions, and returns the text that replaces the range.
    """

    def __init__(self, data: bytes):
        self.data = data
        self.replacements: list[tuple[int, int, str]] = []
        self.inserts: list[tuple[int, int, str]] = []
        self.regions: list[tuple[int, int, object]] = []

    def replace(self, start: int, end: int, text: str):
        self.replacements.append((start, end, text))

    def insert(self, at: int, text: str, rank: int = 0):
        self.inserts.append((at, rank, text))

    def wrap(self, start: int, end: int, before: str, after: str, depth: int):
        self.insert(start, before, rank=depth)
        self.insert(end, after, rank=-depth)

    def region(self, start: int, end: int, callback):
        self.regions.append((start, end, callback))

    def _render(self, start, end, reps, ins) -> str:
        reps = sorted(r for r in reps if start <= r[0] and r[1] <= end)
        ins = sorted((i for i in ins if start <= i[0] <= end), key=lambda t: (t[0], t[1]))
        for (s0, e0, _), (s1, e1, _) in zip(reps, reps[1:]):
            if s1 < e0:
                raise TranslationError(f"overlapping rewrites at bytes {s0}-{e0} and {s1}-{e1}")
        out, pos, ri, ii = [], start, 0, 0
        while ri < len(reps) or ii < len(ins):
            next_rep = reps[ri][0] if ri < len(reps) else None
            next_ins = ins[ii][0] if ii < len(ins) else None
            if next_ins is not None and (next_rep is None or next_ins <= next_rep):
                at, _, text = ins[ii]
                if at < pos:
                    raise TranslationError(f"insertion at byte {at} falls inside a rewrite")
                out.append(self.data[pos:at].decode())
                out.append(text)
                pos = at
                ii += 1
            else:
                s, e, text = reps[ri]
                out.append(self.data[pos:s].decode())
                out.append(text)
                pos = e
                ri += 1
        out.append(self.data[pos:end].decode())
        return "".join(out)

    def apply(self) -> str:
        regions = sorted(self.regions, key=lambda r: (r[0], r[1]))
        for (s0, e0, _), (s1, e1, _) in zip(regions, regions[1:]):
            if s1 < e0:
                raise TranslationError(f"overlapping regions at bytes {s0}-{e0} and {s1}-{e1}")

        def inside(at, end=None):
            end = at if end is None else end
            return any(rs <= at and end <= re_ and not (at == re_) for rs, re_, _ in regions)

        reps = [r for r in self.replacements if not inside(r[0], r[1])]
        ins = [i for i in self.inserts if not inside(i[0])]
        for rs, re_, callback in regions:
            def render(extra=(), rs=rs, re_=re_):
                return self._render(rs, re_, self.replacements,
                                     self.inserts + [(at, rank, text) for at, rank, text in extra])
            reps.append((rs, re_, callback(render)))
        return self._render(0, len(self.data), reps, ins)


def _canonical(t) -> str:
    return t.get_canonical().spelling


def _is_double(t) -> bool:
    c = _canonical(t).replace("const ", "").replace("volatile ", "").strip()
    return c == "double"


def _is_float(t) -> bool:
    return _canonical(t).replace("const ", "").replace("volatile ", "").strip() == "float"


_INTEGRAL_KINDS = frozenset({"BOOL", "CHAR_U", "UCHAR", "CHAR16", "CHAR32", "USHORT", "UINT", "ULONG",
                             "ULONGLONG", "CHAR_S", "SCHAR", "WCHAR", "SHORT", "INT", "LONG", "LONGLONG"})


def _float_literal_value(token: str) -> float:
    """A binary32 literal's exact value: the decimal rounded once, straight
    to binary32 (Python's float() would round to binary64 first, and the
    second rounding can differ at a binary32 tie)."""
    from fractions import Fraction
    text = token.rstrip("fF")
    exact = Fraction(float.fromhex(text)) if text.lower().startswith(("0x", "-0x")) else Fraction(text)
    return _round_fraction_to_binary32(exact)


def _round_fraction_to_binary32(x) -> float:
    from fractions import Fraction
    import math
    if x == 0:
        return 0.0
    sign = -1 if x < 0 else 1
    x = abs(x)
    e = math.floor(math.log2(x))
    while Fraction(2) ** e > x:
        e -= 1
    while Fraction(2) ** (e + 1) <= x:
        e += 1
    e = max(e, -126)                      # subnormal spacing below 2^-126
    ulp = Fraction(2) ** (e - 23)
    q, r = divmod(x, ulp)
    if r * 2 > ulp or (r * 2 == ulp and q % 2 == 1):
        q += 1
    value = float(q * ulp)
    if value > 3.4028234663852886e38:
        value = math.inf
    return sign * value


def _round_to_binary32(v: float) -> float:
    """binary64 -> binary32, round to nearest even (a single rounding)."""
    from fractions import Fraction
    import math
    if math.isnan(v) or math.isinf(v):
        return v
    return _round_fraction_to_binary32(Fraction(v))


def _double_literal(token: str) -> int:
    text = token.rstrip("lL")
    if text.lower().startswith("0x"):
        value = float.fromhex(text)
    else:
        value = float(text)
    return struct.unpack("<Q", struct.pack("<d", value))[0]


# ---------------------------------------------------------------- translate

def _indirection_kind(t):
    k = t.get_canonical().kind.name
    return "ptr" if k == "POINTER" else "ref" if k in ("LVALUEREFERENCE", "RVALUEREFERENCE") else None


def _binary32_of_decimal(text: str):
    """The binary32 a C float literal names (round to nearest even, from
    the exact decimal), as a Python float; None when it is not finite."""
    import math
    try:
        v = _float_literal_value(text)
    except (ValueError, ZeroDivisionError):
        return None
    return None if math.isinf(v) or math.isnan(v) else v


class _Rewriter:
    def __init__(self, text: str, tu, *, unit: str, retain_subnormals: bool = False):
        import clang.cindex as ci
        self.ci = ci
        self.text = text
        self.data = text.encode()
        self.tu = tu
        self.unit = unit
        self.retain = retain_subnormals
        self.edits = Edits(self.data)
        self._indirect_kernels: set = set()
        self.kernels: dict[str, KernelSignature] = {}
        self._handled_literals: set[int] = set()
        #: spans already replaced by a folded constant (not visited again)
        self._folded: set = set()
        #: program-scope constant tables whose `float` stays `float`
        self._keep_float_ranges: list = []
        #: declaration statement start -> (stmt, var usrs, function usr, default space)
        self._qualified_sites: dict = {}
        self._emitted_sites: set = set()
        self._decl_stmt_index = None
        self._kernel_cursors: list = []
        self._constant_starts: set = set()
        self._evaluating: dict = {}
        #: usr -> module global (see module_global)
        self.module_globals: dict = {}
        self._implicit_needs, self._hoisted, self._global_keys = {}, {}, {}
        self._program_vars_done: set = set()
        self._rendered_regions: set = set()

    # -- helpers
    def main_file(self, cursor) -> bool:
        f = cursor.location.file
        return f is not None and f.name == "unit.cpp"

    def span(self, cursor) -> tuple[int, int]:
        return cursor.extent.start.offset, cursor.extent.end.offset

    def src(self, cursor) -> str:
        s, e = self.span(cursor)
        return self.data[s:e].decode()

    def refuse(self, cursor, what: str):
        raise TranslationError(f"{self.unit}: line {cursor.location.line}: {what}")

    # -- the walk
    def run(self) -> TranslatedUnit:
        K = self.ci.CursorKind
        top = [c for c in self.tu.cursor.get_children() if self.main_file(c)]
        from woof.backend.translate.spaces import SpaceAnalysis
        self.spaces = SpaceAnalysis(self)
        self.spaces.collect(top)
        self._kernel_cursors = self._kernels(top)
        self.spaces.run(self._kernel_cursors)
        self.eliminate_gotos()
        self.dynamic_shared = self._dynamic_shared(top)
        self._register_globals(top)
        self.compute_implicit(top)
        for cursor in top:
            self.visit_top(cursor)
        self.emit_spaces()
        self.emit_decl_sites()
        self.thread_builtins(top)
        self.rewrite_tokens()
        msl = self.edits.apply()
        msl = msl.replace(f'__attribute__((annotate("{_GLOBAL}")))', "")
        msl = msl.replace(f'__attribute__((annotate("{_SHARED}")))', "threadgroup")
        msl = msl.replace(f'__attribute__((annotate("{_CONSTANT}")))', "constant")
        msl = re.sub(r'__attribute__\(\(annotate\("' + _SPACE_HINT + r'\w+"\)\)\)', "", msl)
        from dataclasses import replace
        if self.spaces.address_casts:
            self.kernels = {k: replace(v, indirect=True) for k, v in self.kernels.items()}
        self.kernels = {k: replace(v, indirect=True) if k in self._indirect_kernels else v
                        for k, v in self.kernels.items()}
        return TranslatedUnit(msl=msl, kernels=self.kernels, globals={
            g["name"]: {"nbytes": g["nbytes"], "ctype": g["ctype"], "space": g["space"], "init": g.get("init")}
            for g in self.module_globals.values()})

    def visit_top(self, cursor):
        K = self.ci.CursorKind
        if cursor.kind == K.LINKAGE_SPEC:
            self.strip_linkage(cursor)
            for child in cursor.get_children():
                self.visit_top(child)
            return
        if cursor.kind == K.NAMESPACE:
            for child in cursor.get_children():
                self.visit_top(child)
            return
        if cursor.kind in (K.FUNCTION_DECL, K.FUNCTION_TEMPLATE):
            if self.is_kernel(cursor):
                if cursor.is_definition():
                    self.rewrite_kernel(cursor)
                self.visit_expr_tree(cursor, depth=0, in_kernel=True)
            else:
                self.visit_expr_tree(cursor, depth=0, in_kernel=False)
            return
        if cursor.kind == K.VAR_DECL:
            self.program_scope_var(cursor)
        if cursor.kind in (K.STRUCT_DECL, K.CLASS_DECL, K.CLASS_TEMPLATE,
                           K.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION):
            self.struct_fields(cursor)
        self.visit_expr_tree(cursor, depth=0, in_kernel=False)

    def program_scope_var(self, cursor):
        """MSL program-scope variables live in ``constant`` (or are
        ``threadgroup``, which only kernels may declare); host-filled and
        mutable globals become module buffers (module_global)."""
        K = self.ci.CursorKind
        if cursor.get_usr() in self._program_vars_done:
            return
        self._program_vars_done.add(cursor.get_usr())
        if any(c.kind == K.ANNOTATE_ATTR and c.spelling == _CONSTANT for c in cursor.get_children()):
            if not self._has_initialiser(cursor):
                self.module_global(cursor, "constant")   # filled by the host
            return
        if any(c.kind == K.ANNOTATE_ATTR and c.spelling == _SHARED for c in cursor.get_children()):
            return
        t = cursor.type
        base = t
        while base.kind.name in ("CONSTANTARRAY", "INCOMPLETEARRAY"):
            base = base.element_type
        if not (base.is_const_qualified() or t.is_const_qualified()):
            init = None
            if self._has_initialiser(cursor):
                init = self._initial_words(cursor)
                if init is None:
                    self.refuse(cursor, f"mutable program-scope variable {cursor.spelling!r} with an "
                                        "initialiser that is not a list of literals")
            self.module_global(cursor, "device", init=init)
            return
        self.constant_declaration(cursor)

    def _has_initialiser(self, cursor):
        return any(c.kind.name not in ("TYPE_REF", "ANNOTATE_ATTR", "NAMESPACE_REF", "TEMPLATE_REF")
                   and not (c.kind.name == "INTEGER_LITERAL" and cursor.type.get_canonical().kind.name ==
                            "CONSTANTARRAY")
                   for c in cursor.get_children()) and any(t.spelling == "=" for t in cursor.get_tokens())

    def _initial_words(self, cursor):
        """A mutable global's initialiser as its buffer's bytes: every
        element a literal (or a folded constant), the rest zero; None
        otherwise."""
        import numpy as np
        from woof.backend.metal.loader import _GLOBAL_DTYPES
        ctype = _canonical(self._base_type(cursor.type)).replace("const ", "").strip()
        dtype = _GLOBAL_DTYPES.get(ctype)
        if dtype is None:
            return None
        values = []

        def value(node):
            kids = [c for c in node.get_children() if c.kind.name != "TYPE_REF"]
            k = node.kind.name
            if k in ("PAREN_EXPR", "UNEXPOSED_EXPR", "CSTYLE_CAST_EXPR") and len(kids) == 1:
                return value(kids[0])
            if k == "UNARY_OPERATOR" and len(kids) == 1:
                op = [t.spelling for t in node.get_tokens()][:1]
                v = value(kids[0])
                return None if v is None else (-v if op == ["-"] else v if op == ["+"] else None)
            if k == "FLOATING_LITERAL":
                text = self.src(node)
                if node.type.get_canonical().kind.name == "FLOAT":
                    return _binary32_of_decimal(text)
                return float.fromhex(text) if text.lower().startswith("0x") else float(text.rstrip("lL"))
            if k == "INTEGER_LITERAL":
                return int(self.src(node).rstrip("uUlL"), 0)
            if k == "CXX_BOOL_LITERAL_EXPR":
                return self.src(node) == "true"
            return None

        def walk(node):
            if node.kind.name == "INIT_LIST_EXPR":
                for c in node.get_children():
                    if not walk(c):
                        return False
                return True
            v = value(node)
            if v is None:
                return False
            values.append(v)
            return True

        init = [c for c in cursor.get_children()
                if c.kind.name not in ("TYPE_REF", "ANNOTATE_ATTR", "NAMESPACE_REF", "TEMPLATE_REF")]
        if not init or not walk(init[-1]):
            return None
        words = np.zeros(int(cursor.type.get_size()) // np.dtype(dtype).itemsize, dtype)
        if len(values) > words.size:
            return None
        words[:len(values)] = np.array(values, dtype=np.float64 if np.dtype(dtype).kind == "f" else None)
        return words.tobytes()

    def module_global(self, cursor, space, init=None):
        """An uninitialised ``__constant__`` or a mutable ``__device__``
        global: a buffer the module owns.  Its declaration goes; every
        kernel that reaches it binds the buffer as a hidden parameter, and
        device functions receive it as an implicit one (thread_builtins)."""
        t = cursor.type
        elem, dims = self._array_shape(t)
        index = len(self.module_globals)
        self.module_globals[cursor.get_usr()] = {
            "name": cursor.spelling, "elem": elem, "dims": dims, "space": space,
            "nbytes": int(t.get_size()), "param": f"__woof_g_{index}",
            "ctype": _canonical(self._base_type(t)).replace("const ", "").strip(), "init": init}
        tokens = list(cursor.get_tokens())
        start, end = tokens[0].extent.start.offset, self._statement_end(cursor)
        before = self.data[:start].rstrip()
        if before.endswith(b"extern"):
            start = len(before) - len(b"extern")
        self.edits.replace(start, end, "")

    @staticmethod
    def _base_type(t):
        while t.get_canonical().kind.name in ("CONSTANTARRAY", "INCOMPLETEARRAY"):
            t = t.get_canonical().element_type
        return t

    def constant_declaration(self, cursor):
        """``constant`` once per declaration (``const T a = 1, b = 2;`` is
        two VAR_DECLs from one statement), its binary64 initialisers folded
        to exact words.  (In a subnormal-retaining unit binary32 constants
        are f32s, built by its constant-qualified constexpr constructor.)"""
        start = cursor.extent.start.offset
        if start not in self._constant_starts:
            self._constant_starts.add(start)
            self.edits.insert(start, "constant ", rank=-10_000)
        init = [c for c in cursor.get_children() if c.kind.name not in ("TYPE_REF", "ANNOTATE_ATTR")]
        if init and _is_double(cursor.type) and init[-1].kind.name not in ("INIT_LIST_EXPR", "FLOATING_LITERAL"):
            if self.fold_constant_double(init[-1]):
                self._folded.add(self.span(init[-1]))
        if init and self.retain and _is_float(cursor.type) \
                and init[-1].kind.name not in ("INIT_LIST_EXPR", "FLOATING_LITERAL"):
            # f32s arithmetic is not constexpr: the value, as a literal
            v = self.const_eval(init[-1])
            if v is not None and v[0] == "double":
                import math
                if math.isfinite(v[1]):
                    s_, e_ = self.span(init[-1])
                    self.edits.replace(s_, e_, f"{float(v[1]).hex()}f")
                    self._folded.add(self.span(init[-1]))

    def local_pointer_decl(self, node):
        """Local pointers and references need an address space in Metal.

        A plain ``T* p = e;`` becomes ``auto p = e;`` (deduced, so right in
        every overload instance).  Anything else -- a reference, several
        declarators in one statement, a pointer-to-array declarator, a null
        or absent initialiser -- gets the space as a qualifier on its
        declaration statement, from the analysis of each instance.
        """
        K = self.ci.CursorKind
        canon = node.type.get_canonical().kind.name
        if canon == "CONSTANTARRAY" and node.type.get_canonical().element_type.get_canonical().kind.name == "POINTER":
            self.local_pointer_array(node)
            return
        if canon not in ("POINTER", "LVALUEREFERENCE"):
            return
        stmt = self._decl_stmts().get(node.get_usr())
        if stmt is None:
            self.refuse(node, f"cannot find the declaration statement of {node.spelling!r}")
        declared = [c for c in stmt.get_children() if c.kind == K.VAR_DECL]
        children = [c for c in node.get_children()
                    if c.kind not in (K.TYPE_REF, K.NAMESPACE_REF, K.TEMPLATE_REF)]
        tokens = list(node.get_tokens())
        name_index = next((i for i, tok in enumerate(tokens)
                           if tok.spelling == node.spelling and tok.kind.name == "IDENTIFIER"), None)
        simple = (canon == "POINTER" and len(declared) == 1 and children and not self._is_null(children[-1])
                  and name_index and "(" not in [t.spelling for t in tokens[:name_index]]
                  and tokens[0].extent.start.offset == node.extent.start.offset)
        if simple:
            self.edits.replace(tokens[0].extent.start.offset, tokens[name_index].extent.start.offset, "auto ")
            return
        key = self.span(stmt)[0]
        if key in self._qualified_sites:
            return
        fn = node.semantic_parent
        default = "thread" if canon == "LVALUEREFERENCE" else "device"
        self._qualified_sites[key] = (stmt, [d.get_usr() for d in declared
                                             if d.type.get_canonical().kind.name in ("POINTER", "LVALUEREFERENCE")],
                                      fn.get_usr() if fn is not None else None, default)

    def _decl_stmts(self):
        if self._decl_stmt_index is None:
            index = {}
            K = self.ci.CursorKind
            for n in self.tu.cursor.walk_preorder():
                if n.kind == K.DECL_STMT and self.main_file(n):
                    for c in n.get_children():
                        if c.kind == K.VAR_DECL:
                            index[c.get_usr()] = n
            self._decl_stmt_index = index
        return self._decl_stmt_index

    def decl_plan(self, instance):
        """Qualifier insertions for the declaration statements of one
        function instance."""
        plan = []
        fn_usr = instance[0] if instance else None
        spaces_of = self.spaces.inst_var_spaces.get(instance, {})
        for at, (stmt, usrs, owner, default) in self._qualified_sites.items():
            if owner != fn_usr:
                continue
            spaces = set()
            for u in usrs:
                spaces |= spaces_of.get(u, set())
            if len(spaces) > 1:
                self.refuse(stmt, f"one declaration holds {' and '.join(sorted(spaces))} pointers")
            plan.append((at, f"{next(iter(spaces), default)} "))
        return plan

    def emit_decl_sites(self):
        """Statements in kernels and single-instance functions; the
        multi-instance ones ride in each rendered copy (_space_inserts)."""
        multi = {usr for usr, inst in self.spaces.instances.items() if len(inst) > 1}
        done = set()
        for instance in list(self.spaces.inst_var_spaces) + [(k.get_usr(), ()) for k in self._kernel_cursors]:
            if instance is None or instance[0] in multi or instance in done:
                continue
            done.add(instance)
            for at, text in self.decl_plan(instance):
                self.edits.insert(at, text, rank=-20_000)
                self._emitted_sites.add(at)
        # statements no analysed instance owns (methods, lambdas, unused code)
        for at, (stmt, usrs, owner, default) in self._qualified_sites.items():
            if at in self._emitted_sites or owner in multi:
                continue
            spaces = set()
            for u in usrs:
                spaces |= self.spaces.var_spaces.get(u, set())
            if len(spaces) > 1:
                self.refuse(stmt, f"one declaration holds {' and '.join(sorted(spaces))} pointers")
            self.edits.insert(at, f"{next(iter(spaces), default)} ", rank=-20_000)

    # -- goto
    _LOOPS = ("FOR_STMT", "WHILE_STMT", "DO_STMT", "SWITCH_STMT")

    def eliminate_gotos(self):
        """Forward ``goto L;`` -> a structured exit (Metal has no goto).

        The statements of L's block from the first one containing a jump
        to L up to L become ``do { ... } while (false)`` behind a flag;
        each jump sets the flag and breaks; every loop or switch between a
        jump and that wrapper re-breaks on the flag; the label goes.
        Refused by name: backward jumps, a declaration in the wrapped
        statements used after the label, a break/continue in them that
        would retarget, overlapping wrappers.
        """
        K = self.ci.CursorKind
        parents, labels, gotos = {}, {}, []
        stack = [(self.tu.cursor, None)]
        while stack:
            node, parent = stack.pop()
            if parent is not None:
                parents[self.span(node) + (node.kind.name,)] = parent
            if node.kind == K.LABEL_STMT and self.main_file(node):
                labels[node.spelling] = labels.get(node.spelling, []) + [node]
            if node.kind == K.GOTO_STMT and self.main_file(node):
                gotos.append(node)
            for child in node.get_children():
                stack.append((child, node))
        targeted = {(g.referenced.spelling, self.span(g.referenced)) for g in gotos if g.referenced is not None}
        for name, nodes in labels.items():
            for label in nodes:
                if (name, self.span(label)) in targeted:
                    continue
                toks = list(label.get_tokens())
                if len(toks) >= 2 and toks[1].spelling == ":":
                    # a label nothing jumps to: Metal refuses the label itself
                    self.edits.replace(toks[0].extent.start.offset, toks[1].extent.end.offset, "")
        if not gotos:
            return

        def parent(node):
            return parents.get(self.span(node) + (node.kind.name,))

        def chain(node):
            out = []
            while node is not None:
                out.append(node)
                node = parent(node)
            return out

        by_label = {}
        for g in gotos:
            target = g.referenced
            if target is None:
                self.refuse(g, "goto whose label clang could not resolve")
            by_label.setdefault((target.spelling, self.span(target)), []).append(g)
        regions = []
        for (name, (ls, le)), jumps in by_label.items():
            label = next(l for l in labels.get(name, []) if self.span(l) == (ls, le))
            block = parent(label)
            if block is None or block.kind != K.COMPOUND_STMT:
                self.refuse(label, f"label {name!r} is not a statement of a block")
            stmts = list(block.get_children())
            j = next(i for i, st in enumerate(stmts) if self.span(st) == (ls, le))
            if self._backward_goto(name, label, stmts, j, jumps, chain, parent, block, regions):
                continue
            first = j
            inner_loops = []
            for g in jumps:
                anc = chain(g)
                top = next((a for a in anc if parent(a) is not None and self.span(parent(a)) == self.span(block)
                            and parent(a).kind == K.COMPOUND_STMT), None)
                if top is None:
                    self.refuse(g, f"goto {name!r} from outside the label's block")
                i = next(k for k, st in enumerate(stmts) if self.span(st) == self.span(top))
                if i >= j:
                    self.refuse(g, f"backward goto {name!r}")
                first = min(first, i)
                for a in anc[1:]:
                    if self.span(a) == self.span(top) and a.kind == top.kind:
                        if a.kind.name in self._LOOPS:
                            inner_loops.append(a)
                        break
                    if a.kind.name in self._LOOPS:
                        inner_loops.append(a)
            region = stmts[first:j]
            rs, re_ = self.span(region[0])[0], ls
            for (a, b) in regions:
                if a < re_ and rs < b and not (a <= rs and re_ <= b) and not (rs <= a and b <= re_):
                    self.refuse(label, f"goto regions overlap at label {name!r}")
            regions.append((rs, re_))
            # retargeted break/continue, and declarations used after the label
            after = stmts[j:]
            for st in region:
                if st.kind == K.DECL_STMT:
                    for var in st.get_children():
                        usr = var.get_usr()
                        for later in after:
                            if any(n.kind == K.DECL_REF_EXPR and n.referenced is not None
                                   and n.referenced.get_usr() == usr for n in later.walk_preorder()):
                                self.refuse(var, f"{var.spelling!r} is declared between a goto and "
                                                 f"label {name!r} and used after it")
                for n in self._unenclosed_jumps(st):
                    self.refuse(n, f"a {n.kind.name.split('_')[0].lower()} between a goto and label "
                                   f"{name!r} would leave the wrapper instead of its loop")
            flag = f"__woof_goto_{name}"
            self.edits.insert(rs, f"bool {flag} = false; do {{ ", rank=-50_000)
            self.edits.insert(ls, f"}} while (false); ", rank=50_000)
            label_tokens = list(label.get_tokens())
            if len(label_tokens) >= 2 and label_tokens[1].spelling == ":":
                self.edits.replace(label_tokens[0].extent.start.offset, label_tokens[1].extent.end.offset, "")
            for g in jumps:
                gs, ge = self.span(g)
                self.edits.replace(gs, ge, f"{{ {flag} = true; break; }}")
            seen = set()
            for loop in inner_loops:
                key = self.span(loop)
                if key in seen:
                    continue
                seen.add(key)
                end = self._statement_end(loop)
                self.edits.insert(end, f" if ({flag}) break;", rank=50_000)

    def _backward_goto(self, name, label, stmts, j, jumps, chain, parent, block, regions):
        """Backward ``goto L;`` (a Fortran retry: ``100 CONTINUE ... GOTO 100``)
        -> ``while (true) { L's statements ...; break; }`` with each jump a
        ``continue``.  True when handled; forward jumps go back to the caller.
        Refused by name: jumps both ways, a jump inside a loop or switch of
        the region, a break/continue the wrapper would capture, declarations
        in the region used after it."""
        K = self.ci.CursorKind
        tops = []
        for g in jumps:
            top = next((a for a in chain(g) if parent(a) is not None and self.span(parent(a)) == self.span(block)
                        and parent(a).kind == K.COMPOUND_STMT), None)
            if top is None:
                return False            # the forward path refuses it by name
            tops.append((g, next(k for k, st in enumerate(stmts) if self.span(st) == self.span(top)), top))
        if all(i < j for _, i, _ in tops):
            return False
        if any(i < j for _, i, _ in tops):
            self.refuse(label, f"label {name!r} is reached by both forward and backward gotos")
        last = max(i for _, i, _ in tops)
        for g, _, top in tops:
            for a in chain(g)[1:]:
                if self.span(a) == self.span(top) and a.kind == top.kind:
                    break
                if a.kind.name in self._LOOPS:
                    self.refuse(g, f"backward goto {name!r} from inside a loop or switch")
        region = stmts[j:last + 1]
        after = stmts[last + 1:]
        for st in region:
            for n in self._unenclosed_jumps(st):
                self.refuse(n, f"a {n.kind.name.split('_')[0].lower()} between label {name!r} and its "
                               "backward goto would leave the retry loop instead of its own")
            if st.kind == K.DECL_STMT:
                for var in st.get_children():
                    usr = var.get_usr()
                    for later in after:
                        if any(n.kind == K.DECL_REF_EXPR and n.referenced is not None
                               and n.referenced.get_usr() == usr for n in later.walk_preorder()):
                            self.refuse(var, f"{var.spelling!r} is declared between label {name!r} and "
                                             "its backward goto and used after them")
        rs = self.span(label)[0]
        re_ = self._statement_end(stmts[last])
        for (a, b) in regions:
            if a < re_ and rs < b and not (a <= rs and re_ <= b) and not (rs <= a and b <= re_):
                self.refuse(label, f"goto regions overlap at label {name!r}")
        regions.append((rs, re_))
        self.edits.insert(rs, "while (true) { ", rank=-50_000)
        self.edits.insert(re_, " break; }", rank=50_000)
        label_tokens = list(label.get_tokens())
        if len(label_tokens) >= 2 and label_tokens[1].spelling == ":":
            self.edits.replace(label_tokens[0].extent.start.offset, label_tokens[1].extent.end.offset, "")
        for g, _, _ in tops:
            gs, ge = self.span(g)
            self.edits.replace(gs, ge, "continue")
        return True

    def _unenclosed_jumps(self, node):
        """break/continue in ``node`` not enclosed by a loop/switch inside it."""
        K = self.ci.CursorKind
        out = []
        stack = [node]
        while stack:
            n = stack.pop()
            if n.kind in (K.BREAK_STMT, K.CONTINUE_STMT):
                out.append(n)
                continue
            if n.kind.name in self._LOOPS and n is not node:
                # a continue inside a switch still targets the outer loop
                if n.kind == K.SWITCH_STMT:
                    stack.extend(c for c in n.walk_preorder() if c.kind == K.CONTINUE_STMT)
                continue
            if n.kind.name in self._LOOPS and n is node:
                continue
            stack.extend(n.get_children())
        return out

    def _statement_end(self, node):
        end = node.extent.end.offset
        rest = self.data[end:end + 64]
        stripped = rest.lstrip()
        if stripped[:1] == b";":
            return end + (len(rest) - len(stripped)) + 1
        return end

    def _dynamic_shared(self, top):
        """``extern __shared__ T name[];``: threadgroup memory sized at
        dispatch.  Metal passes it as a ``[[threadgroup(0)]]`` kernel
        parameter, so the declaration goes and each kernel that names it
        receives it.  {usr: (name, element type)}"""
        K = self.ci.CursorKind
        found = {}
        for node in self.tu.cursor.walk_preorder():
            if node.kind != K.VAR_DECL or not self.main_file(node):
                continue
            if not any(c.kind == K.ANNOTATE_ATTR and c.spelling == _SHARED for c in node.get_children()):
                continue
            if node.type.get_canonical().kind.name != "INCOMPLETEARRAY":
                continue
            elem = self._spell_type(node.type.get_canonical().element_type, drop_const=True)
            found[node.get_usr()] = (node.spelling, elem)
            tokens = list(node.get_tokens())
            start = tokens[0].extent.start.offset
            # the declaration's extent stops before its ';'
            end = node.extent.end.offset
            rest = self.data[end:end + 1]
            self.edits.replace(start, end + (1 if rest == b";" else 0), "")
            # `extern` precedes the decl's own extent
            before = self.data[:start].rstrip()
            if before.endswith(b"extern"):
                self.edits.replace(len(before) - len(b"extern"), len(before), "")
        if not found:
            return found
        return found

    def _register_globals(self, cursors):
        K = self.ci.CursorKind
        for c in cursors:
            if c.kind in (K.NAMESPACE, K.LINKAGE_SPEC):
                self._register_globals(list(c.get_children()))
            elif c.kind == K.VAR_DECL:
                self._registering = True
                self.program_scope_var(c)
                self._registering = False

    def _kernels(self, cursors):
        K = self.ci.CursorKind
        found = []
        for c in cursors:
            if c.kind in (K.LINKAGE_SPEC, K.NAMESPACE):
                found += self._kernels(list(c.get_children()))
            elif c.kind == K.FUNCTION_DECL and self.is_kernel(c) and c.is_definition():
                found.append(c)
        return found

    # -- thread-position builtins in device functions
    _BUILTINS = ("threadIdx", "blockIdx", "blockDim", "gridDim")
    _BUILTIN_PARAMS = ("uint3 __woof_tid, uint3 __woof_bid, uint3 __woof_bdim, uint3 __woof_gdim")
    _BUILTIN_ARGS = "__woof_tid, __woof_bid, __woof_bdim, __woof_gdim"

    _SYNC_REDUCTIONS = ("__syncthreads_or", "__syncthreads_and", "__syncthreads_count")

    def thread_builtins(self, top):
        """Implicit parameters: what only a kernel can have, passed down.

        MSL gives a thread its position only as kernel parameters, and
        threadgroup memory can be declared only in a kernel.  So a device
        function receives, as leading parameters (leading, so its default
        arguments still bind), every resource it or its callees use:

        * ``builtins``: threadIdx, blockIdx, blockDim, gridDim;
        * ``sync``: the scratch word __syncthreads_or/_and/_count reduce in;
        * each ``__shared__`` array a device function declares, hoisted to
          the kernels that reach it (declared there, passed down by name).
        """
        K = self.ci.CursorKind
        functions = self.spaces.functions
        kernels = list(self._kernel_cursors)
        needs, hoisted = self._implicit_needs, self._hoisted
        self._emit_implicit(K, functions, kernels, needs, hoisted)

    def compute_implicit(self, top):
        """The implicit-parameter closure (see thread_builtins), computed
        before kernels are rewritten: a kernel's hidden global parameters
        are part of its signature."""
        K = self.ci.CursorKind
        functions = self.spaces.functions
        kernels = list(self._kernel_cursors)
        own, calls, hoisted = {}, {}, {}
        global_keys = {usr: f"g:{i}" for i, usr in enumerate(self.module_globals)}
        for usr, info in list(functions.items()) + [(k.get_usr(), None) for k in kernels]:
            cursor = info.cursor if info is not None else next(k for k in kernels if k.get_usr() == usr)
            uses, callees = set(), set()
            for node in cursor.walk_preorder():
                if node.kind == K.DECL_REF_EXPR and node.spelling in self._BUILTINS:
                    uses.add("builtins")
                elif node.kind == K.DECL_REF_EXPR and node.referenced is not None \
                        and node.referenced.get_usr() in global_keys:
                    uses.add(global_keys[node.referenced.get_usr()])
                elif node.kind == K.DECL_REF_EXPR and node.referenced is not None \
                        and node.referenced.get_usr() in self.dynamic_shared and info is not None:
                    # dynamic shared memory read in a device function: passed down by name
                    uses.add("dyn:" + node.referenced.get_usr())
                elif node.kind == K.CALL_EXPR:
                    if node.spelling in self._SYNC_REDUCTIONS:
                        uses.add("sync")
                    callee = self.spaces.resolve_call(node)
                    if callee is not None:
                        callees.add(callee.cursor.get_usr())
                elif node.kind == K.VAR_DECL and info is not None and any(
                        c.kind == K.ANNOTATE_ATTR and c.spelling == _SHARED for c in node.get_children()):
                    if node.type.get_canonical().kind.name == "INCOMPLETEARRAY":
                        continue  # dynamic shared memory: _dynamic_shared
                    key = f"sh:{len(hoisted)}"
                    hoisted[key] = (node, usr)
                    uses.add(key)
            own[usr], calls[usr] = uses, callees
        needs = {u: set(v) for u, v in own.items()}
        changed = True
        while changed:
            changed = False
            for usr, callees in calls.items():
                for c in callees:
                    extra = needs.get(c, set()) - needs[usr]
                    if extra:
                        needs[usr] |= extra
                        changed = True

        self._implicit_needs, self._hoisted, self._global_keys = needs, hoisted, global_keys

    @staticmethod
    def _implicit_order(keys):
        rank = {"builtins": 0, "sync": 1}

        def key(k):
            if k in rank:
                return (rank[k], 0, "")
            kind, _, rest = k.partition(":")
            return ({"g": 2, "dyn": 3, "sh": 4}[kind], int(rest) if rest.isdigit() else 0, rest)
        return sorted(keys, key=key)

    def _emit_implicit(self, K, functions, kernels, needs, hoisted):
        order = self._implicit_order
        globals_by_key = {key: usr for usr, key in self._global_keys.items()}

        def param(key):
            if key.startswith("dyn:"):
                name, elem = self.dynamic_shared[key[4:]]
                return f"threadgroup {elem}* {name}"
            if key == "builtins":
                return self._BUILTIN_PARAMS
            if key == "sync":
                return "threadgroup metal::atomic_int* __woof_sync"
            if key.startswith("g:"):
                return self._global_param(globals_by_key[key])
            node, _ = hoisted[key]
            elem, dims = self._array_shape(node.type)
            name = f"__woof_sh_{key[3:]}"
            if len(dims) <= 1:
                return f"threadgroup {elem}* {name}"
            return f"threadgroup {elem} (*{name})" + "".join(f"[{d}]" for d in dims[1:])

        def arg(key, in_kernel=False):
            if key.startswith("sh:") and in_kernel and not self._array_shape(hoisted[key][0].type)[1]:
                return f"&__woof_sh_{key[3:]}"     # a hoisted scalar: the kernel holds the object
            if key.startswith("dyn:"):
                return self.dynamic_shared[key[4:]][0]
            if key.startswith("g:"):
                return self.module_globals[globals_by_key[key]]["param"]
            return {"builtins": self._BUILTIN_ARGS, "sync": "__woof_sync"}.get(key, f"__woof_sh_{key[3:]}")

        # module globals: every reference names the bound buffer
        for usr, g in self.module_globals.items():
            for node in self.tu.cursor.walk_preorder():
                if node.kind == K.DECL_REF_EXPR and node.referenced is not None \
                        and node.referenced.get_usr() == usr and self.main_file(node):
                    rs, re_ = self.span(node)
                    self.edits.replace(rs, re_, g["param"] if g["dims"] else f"(*{g['param']})")
        # hoist each device function's __shared__ arrays out of its body
        for key, (node, owner) in hoisted.items():
            name = f"__woof_sh_{key[3:]}"
            stmt = self._decl_stmt_of(node)
            if stmt is None:
                self.refuse(node, f"__shared__ {node.spelling!r} is not a declaration statement")
            ss, se = self.span(stmt)
            self.edits.replace(ss, self._statement_end(stmt), "")
            scalar = not node.type.get_canonical().kind.name.endswith("ARRAY")
            for ref in functions[owner].cursor.walk_preorder():
                if ref.kind == K.DECL_REF_EXPR and ref.referenced is not None \
                        and ref.referenced.get_usr() == node.get_usr():
                    rs, re_ = self.span(ref)
                    self.edits.replace(rs, re_, f"(*{name})" if scalar else name)
        # device functions: leading parameters
        for usr, info in functions.items():
            keys = order(needs.get(usr, ()))
            if not keys:
                continue
            text = ", ".join(param(k) for k in keys)
            seen_parens = set()
            for cursor in [info.cursor] + info.decls:
                paren = self._open_paren(cursor)
                if paren in seen_parens:
                    continue            # one declaration reached twice
                seen_parens.add(paren)
                if paren is None:
                    self.refuse(cursor, f"cannot find the parameter list of {cursor.spelling!r}")
                has_params = any(c.kind == K.PARM_DECL for c in cursor.get_children())
                # before the first parameter's address-space qualifier
                self.edits.insert(paren, text + (", " if has_params else ""), rank=-40_000)
        # kernels: declare the threadgroup resources their callees need
        for k in kernels:
            keys = order(k_ for k_ in needs.get(k.get_usr(), ()) - {"builtins"}
                         if not k_.startswith(("g:", "dyn:")))
            body = next((c for c in k.get_children() if c.kind == K.COMPOUND_STMT), None)
            decls = []
            for key in keys:
                if key == "sync":
                    decls.append("threadgroup metal::atomic_int __woof_sync[1];")
                else:
                    node, owner = hoisted[key]
                    elem, dims = self._array_shape(node.type)
                    dims = self._bound_template_dims(functions[owner].cursor, dims)
                    decls.append(f"threadgroup {elem} __woof_sh_{key[3:]}" + "".join(f"[{d}]" for d in dims) + ";")
            if decls and body is not None:
                self.edits.insert(body.extent.start.offset + 1, "\n    " + " ".join(decls), rank=9_000)
        # every call: pass them on; sync reductions read the scratch
        for fn in kernels + [f.cursor for f in functions.values()]:
            for node in fn.walk_preorder():
                if node.kind != K.CALL_EXPR:
                    continue
                if node.spelling in self._SYNC_REDUCTIONS:
                    kids = list(node.get_children())
                    callee_tok = next(t for t in node.get_tokens() if t.spelling == node.spelling)
                    self.edits.replace(callee_tok.extent.start.offset, callee_tok.extent.end.offset,
                                       "__woof" + node.spelling[1:])
                    paren = self._call_paren(node)
                    self.edits.insert(paren, "__woof_sync, ", rank=-40_000)
                    continue
                callee = self.spaces.resolve_call(node)
                if callee is None:
                    continue
                keys = order(needs.get(callee.cursor.get_usr(), ()))
                if not keys:
                    continue
                paren = self._call_paren(node)
                if paren is None:
                    self.refuse(node, f"cannot find the argument list of a call to {node.spelling!r}")
                has_args = len(list(node.get_arguments())) > 0
                in_kernel = fn in kernels
                self.edits.insert(paren, ", ".join(arg(k, in_kernel) for k in keys) + (", " if has_args else ""),
                                  rank=-40_000)

    def _bound_template_dims(self, owner, dims):
        """A hoisted ``__shared__ T x[KMAX]`` of a function template is
        declared in the kernel, where ``KMAX`` means nothing: substitute the
        explicit template argument its callers pass (one instantiation
        only -- two would need one array per instance)."""
        params = [c.spelling for c in owner.get_children()      # every template parameter, in order
                  if c.kind.name in ("TEMPLATE_NON_TYPE_PARAMETER", "TEMPLATE_TYPE_PARAMETER",
                                     "TEMPLATE_TEMPLATE_PARAMETER")]
        if not any(re.search(rf"\b{re.escape(p)}\b", str(d)) for p in params for d in dims):
            return dims
        seen = set()
        for node in self.tu.cursor.walk_preorder():
            if node.kind.name != "CALL_EXPR" or node.spelling != owner.spelling or not self.main_file(node):
                continue
            toks = [t.spelling for t in node.get_tokens()]
            if owner.spelling not in toks:
                continue
            i = toks.index(owner.spelling) + 1
            if i >= len(toks) or toks[i] != "<":
                self.refuse(node, f"a call to {owner.spelling!r} without explicit template arguments "
                                  "sizes a hoisted __shared__ array")
            depth, args, cur = 0, [], []
            for t in toks[i + 1:]:
                if t == "<":
                    depth += 1
                elif t == ">":
                    if depth == 0:
                        break
                    depth -= 1
                if t == "," and depth == 0:
                    args.append(" ".join(cur)); cur = []
                else:
                    cur.append(t)
            args.append(" ".join(cur))
            seen.add(tuple(args))
        if len(seen) != 1:
            self.refuse(owner, f"{owner.spelling!r} sizes a __shared__ array by a template parameter "
                               f"and is instantiated {len(seen)} ways")
        values = dict(zip(params, next(iter(seen))))
        out = []
        for d in dims:
            for name, v in values.items():
                d = re.sub(rf"\b{re.escape(name)}\b", f"({v})", str(d))
            out.append(d)
        return out

    def _global_param(self, usr):
        g = self.module_globals[usr]
        if len(g["dims"]) <= 1:
            return f"{g['space']} {g['elem']}* {g['param']}"
        return f"{g['space']} {g['elem']} (*{g['param']})" + "".join(f"[{d}]" for d in g["dims"][1:])

    def _array_shape(self, t):
        if "DEPENDENTSIZEDARRAY" in (t.kind.name, t.get_canonical().kind.name) or (
                t.get_canonical().kind.name == "CONSTANTARRAY"
                and "DEPENDENTSIZEDARRAY" == t.get_canonical().element_type.get_canonical().kind.name):
            # a bound that names a template parameter: read the bounds as spelled
            m = re.fullmatch(r"(.*?)\s*((?:\[[^\]]+\])+)", t.spelling)
            if m is None:
                raise TranslationError(f"{self.unit}: cannot read the array bounds of {t.spelling!r}")
            return m.group(1), re.findall(r"\[([^\]]+)\]", m.group(2))
        dims = []
        while t.get_canonical().kind.name == "CONSTANTARRAY":
            dims.append(t.get_canonical().element_count)
            t = t.get_canonical().element_type
        return self._spell_type(t, drop_const=False), dims

    def _decl_stmt_of(self, var):
        K = self.ci.CursorKind
        for node in self.tu.cursor.walk_preorder():
            if node.kind == K.DECL_STMT and self.main_file(node):
                if any(c.get_usr() == var.get_usr() for c in node.get_children()):
                    return node
        return None

    def _open_paren(self, cursor):
        """Byte offset just after the '(' opening ``cursor``'s parameters."""
        tokens = list(cursor.get_tokens())
        name = cursor.spelling.split("<", 1)[0] if not cursor.spelling.startswith("operator") else cursor.spelling
        for i, t in enumerate(tokens):
            if t.spelling == name or (cursor.spelling.startswith("operator") and t.spelling == "operator"):
                for u in tokens[i + 1:]:
                    if u.spelling == "(":
                        return u.extent.end.offset
                return None
        return None

    def _call_paren(self, call):
        kids = list(call.get_children())
        if not kids:
            return None
        callee_end = kids[0].extent.end.offset
        for t in call.get_tokens():
            if t.extent.start.offset >= callee_end and t.spelling == "(":
                return t.extent.end.offset
        return None

    # -- address spaces
    def _cast_insert_at(self, start, end):
        tokens = list(self.tu.get_tokens(extent=self.tu.get_extent("unit.cpp", (start, end))))
        if not tokens:
            return None
        if tokens[0].spelling == "(":                       # C-style: (T*)x
            return tokens[1].extent.start.offset
        if tokens[0].spelling in ("static_cast", "reinterpret_cast", "const_cast"):
            return next(t for t in tokens if t.spelling == "<").extent.end.offset
        return None                                         # functional cast T(x)

    def cast_plan(self, instance):
        """Insertions naming each pointer cast's space in one function
        instance (an integer address is device memory; null takes device)."""
        plan = []
        for (start, end), spaces in self.spaces.cast_spaces.get(instance, {}).items():
            if len(spaces) > 1:
                self.refuse_at(start, f"a pointer cast reads {' and '.join(sorted(spaces))} pointers")
            at = self._cast_insert_at(start, end)
            if at is not None:
                plan.append((at, f"{next(iter(spaces), 'device')} "))
        return plan

    def emit_casts(self):
        """Casts in kernels and in single-instance functions are emitted
        here; multi-instance functions carry theirs per copy
        (_space_inserts).  Casts the analysis never reached (struct
        methods) take their operand's space read without context."""
        multi = {usr for usr, inst in self.spaces.instances.items() if len(inst) > 1}
        reached = set()
        for instance, casts in self.spaces.cast_spaces.items():
            reached.update(casts)
            if instance is None or instance[0] in multi:
                continue
            for at, text in self.cast_plan(instance):
                self.edits.insert(at, text, rank=-25_000)
        K = self.ci.CursorKind
        from woof.backend.translate.spaces import _CASTS, _INTEGER_KINDS, _indirection
        for node in self.tu.cursor.walk_preorder():
            if node.kind.name not in _CASTS or not self.main_file(node) or _indirection(node.type) != "ptr":
                continue
            key = (node.extent.start.offset, node.extent.end.offset)
            if key in reached:
                continue
            operand = [c for c in node.get_children() if c.kind not in (K.TYPE_REF, K.TEMPLATE_REF)]
            space = "device"
            if operand and operand[-1].type.get_canonical().kind.name not in _INTEGER_KINDS:
                space = self.spaces.space(operand[-1], {}) or "device"
                if space == "null":
                    space = "device"
            at = self._cast_insert_at(*key)
            if at is not None:
                self.edits.insert(at, f"{space} ", rank=-25_000)

    def refuse_at(self, offset, what):
        line = self.data[:offset].count(b"\n") + 1
        raise TranslationError(f"{self.unit}: line {line}: {what}")

    def emit_spaces(self):
        from woof.backend.translate.spaces import DEVICE
        self.emit_casts()
        for (start, end), opening in self.spaces.null_casts.items():
            self.edits.wrap(start, end, opening, ")", 1_000)
        for usr, params in self.spaces.localised.items():
            info = self.spaces.functions[usr]
            body = next((c for c in info.cursor.get_children()
                         if c.kind == self.ci.CursorKind.COMPOUND_STMT), None)
            if body is None:
                continue
            prologue = []
            for parm in params:
                name = parm.spelling
                pointee = self._spell_type(parm.type.get_pointee(), drop_const=False)
                at = parm.location.offset
                self.edits.replace(at, at + len(name), f"__woof_in_{name}")
                prologue.append(f"{pointee} __woof_copy_{name} = *__woof_in_{name}; "
                                f"thread {pointee} *{name} = &__woof_copy_{name};")
            self.edits.insert(body.extent.start.offset + 1, "\n    " + "\n    ".join(prologue), rank=10_001)
        for usr, info in self.spaces.functions.items():
            this_spaces = sorted(self.spaces.this_spaces.get(usr, ()))
            if not info.slots and not info.returns and not this_spaces:
                continue
            instances = sorted(self.spaces.instances.get(usr) or ())
            if not instances:
                instances = [tuple(DEVICE for _ in info.slots)]
            for cursor in [info.cursor] + info.decls:
                plans = [self._space_inserts(info, cursor, spaces) for spaces in instances]
                if this_spaces:
                    at = self._method_qualifier_at(cursor)
                    if at is None:
                        self.refuse(cursor, f"cannot place an address-space qualifier on {cursor.spelling!r}")
                    plans = plans + [plan + [(at, f" {space}")] for plan in plans for space in this_spaces]
                if len(plans) == 1:
                    for at, text in plans[0]:
                        self.edits.insert(at, text, rank=-20_000)
                    continue
                start, end = self.span(cursor)
                if cursor.is_definition() is False:
                    end = self._decl_end(cursor)
                if (start, end) in self._rendered_regions:
                    continue      # the same declaration reached twice
                self._rendered_regions.add((start, end))

                def render_all(render, plans=plans):
                    return "\n".join(render([(at, -20_000, text) for at, text in plan]) for plan in plans)
                self.edits.region(start, end, render_all)

    def _method_qualifier_at(self, cursor):
        """After the parameter list's ')' (and a following ``const``)."""
        tokens = list(cursor.get_tokens())
        name = cursor.spelling
        i = next((k for k, t in enumerate(tokens) if t.spelling == name or
                  (name.startswith("operator") and t.spelling == "operator")), None)
        if i is None:
            return None
        while i < len(tokens) and tokens[i].spelling != "(":
            i += 1
        if name == "operator()":           # operator() ( params )
            i += 1
            while i < len(tokens) and tokens[i].spelling != "(":
                i += 1
        depth = 0
        for k in range(i, len(tokens)):
            if tokens[k].spelling == "(":
                depth += 1
            elif tokens[k].spelling == ")":
                depth -= 1
                if depth == 0:
                    at = tokens[k].extent.end.offset
                    if k + 1 < len(tokens) and tokens[k + 1].spelling == "const":
                        at = tokens[k + 1].extent.end.offset
                    return at
        return None

    def _decl_end(self, cursor):
        # a prototype's extent stops before its ';'
        end = cursor.extent.end.offset
        rest = self.data[end:end + 4]
        return end + 1 if rest[:1] == b";" else end

    def _space_inserts(self, info, cursor, spaces):
        K = self.ci.CursorKind
        params = [p for p in cursor.get_children() if p.kind == K.PARM_DECL]
        plan = []
        for (index, _, _), space in zip(info.slots, spaces):
            if index < len(params):
                plan.append((params[index].extent.start.offset, f"{space} "))
        if cursor is info.cursor and len(self.spaces.instances.get(info.cursor.get_usr(), ())) > 1:
            plan += self.cast_plan((info.cursor.get_usr(), spaces))
            plan += self.decl_plan((info.cursor.get_usr(), spaces))
        if info.returns:
            ret = self.spaces.returns.get((info.cursor.get_usr(), spaces)) or "device"
            start = self._return_type_start(cursor)
            if start is not None:
                plan.append((start, f"{ret} "))
        return plan

    def _return_type_start(self, cursor):
        tokens = list(cursor.get_tokens())
        name = cursor.spelling
        idx = None
        for i, t in enumerate(tokens):
            if t.spelling == name or (name.startswith("operator") and t.spelling == "operator"):
                idx = i
                break
        if idx is None:
            return None
        i = 0
        if tokens and tokens[0].spelling == "template":
            depth = 0
            for j, t in enumerate(tokens):
                if t.spelling == "<":
                    depth += 1
                elif t.spelling == ">":
                    depth -= 1
                    if depth == 0:
                        i = j + 1
                        break
        skip = {"static", "inline", "constexpr", "extern", "__forceinline__", "__device__", "__host__"}
        while i < idx and tokens[i].spelling in skip:
            i += 1
        return tokens[i].extent.start.offset if i < idx else None

    def struct_fields(self, cursor):
        """Pointer fields take their initialiser's space (else device),
        reference members thread; static data members are constant."""
        K = self.ci.CursorKind
        from woof.backend.translate.spaces import _indirection
        seen = set()
        for c in cursor.get_children():
            if c.kind == K.FIELD_DECL and _indirection(c.type) == "ptr":
                # `T *a, *b;` declares two fields from one decl-specifier
                if c.extent.start.offset not in seen:
                    seen.add(c.extent.start.offset)
                    space = self._field_space(c)
                    self.edits.insert(c.extent.start.offset, f"{space} ", rank=-20_000)
            elif c.kind == K.FIELD_DECL and _indirection(c.type) == "ref":
                # a reference member binds a workspace object of the caller's
                # (woof's are locals): thread memory
                if c.extent.start.offset not in seen:
                    seen.add(c.extent.start.offset)
                    self.edits.insert(c.extent.start.offset, "thread ", rank=-20_000)
            # (methods and constructors: the space analysis instances them
            # like free functions -- parameters, returns, per-copy edits)
            elif c.kind == K.VAR_DECL:
                # a static data member is program-scope storage: constant
                self.constant_declaration(c)
            elif c.kind == K.CONSTRUCTOR and c.is_definition():
                self._arithmetic_constructor(cursor, c)
            elif c.kind in (K.STRUCT_DECL, K.CLASS_DECL, K.CLASS_TEMPLATE,
                            K.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION):
                self.struct_fields(c)

    def local_pointer_array(self, node):
        """``T* a[N] = {&x, p, ...}``: the elements' address space, from the
        initialisers (one space for the whole array)."""
        K = self.ci.CursorKind
        spaces = set(self.spaces.var_spaces.get(node.get_usr(), set()))
        if len(spaces) > 1:
            self.refuse(node, f"array of pointers {node.spelling!r} holds {' and '.join(sorted(spaces))} pointers")
        tokens = list(node.get_tokens())
        if tokens:
            self.edits.insert(tokens[0].extent.start.offset, f"{next(iter(spaces), 'thread')} ", rank=-20_000)

    def _is_null(self, e):
        while e.kind.name in ("UNEXPOSED_EXPR", "PAREN_EXPR", "CSTYLE_CAST_EXPR") and list(e.get_children()):
            e = list(e.get_children())[-1]
        return e.kind.name in ("CXX_NULL_PTR_LITERAL_EXPR", "GNU_NULL_EXPR") or \
            (e.kind.name == "INTEGER_LITERAL" and self.src(e).strip() in ("0", "0L", "0UL"))

    def _field_space(self, field):
        """A pointer field's space: its in-class initialiser's when it has
        one (a pointer into a constant table), else device -- woof's
        accessor structs wrap kernel buffers.  A source can name it outright:
        ``__attribute__((annotate("woof_space:thread")))`` on the field."""
        for c in field.get_children():
            if c.kind.name == "ANNOTATE_ATTR" and c.spelling.startswith(_SPACE_HINT):
                return c.spelling[len(_SPACE_HINT):]
        from woof.backend.translate.spaces import field_key
        given = self.spaces.field_spaces.get(field_key(field), set()) if self.spaces is not None else set()
        if len(given) > 1:
            self.refuse(field, f"pointer field {field.spelling!r} is given {' and '.join(sorted(given))} "
                               "pointers; a Metal struct field has one address space")
        if given:
            return next(iter(given))
        inits = [c for c in field.get_children() if c.kind.name not in ("TYPE_REF", "ANNOTATE_ATTR")]
        if inits:
            sp = self.spaces.space(inits[-1], {})
            if sp in ("constant", "device", "threadgroup", "thread"):
                return sp
        # `this->f = e` / `f = e` / `: f(e)` in the struct's own constructors
        parent = field.semantic_parent
        spaces = set()
        if parent is not None:
            for m in parent.get_children():
                if m.kind.name not in ("CONSTRUCTOR", "CXX_METHOD"):
                    continue
                for n in m.walk_preorder():
                    if n.kind.name == "BINARY_OPERATOR":
                        kids = list(n.get_children())
                        lhs = kids[0] if len(kids) == 2 else None
                        # a dependent `this->f` (class templates) resolves to nothing: match the name
                        names_field = lhs is not None and lhs.kind.name == "MEMBER_REF_EXPR" and (
                            (lhs.referenced is not None and lhs.referenced.get_usr() == field.get_usr())
                            or (lhs.referenced is None and
                                [t.spelling for t in lhs.get_tokens()][-1:] == [field.spelling]))
                        op = [t.spelling for t in n.get_tokens()
                              if lhs is not None and t.extent.start.offset >= lhs.extent.end.offset][:1]
                        if names_field and op == ["="]:
                            sp = self.spaces.space(kids[1], {})
                            if sp in ("constant", "device", "threadgroup", "thread"):
                                spaces.add(sp)
        if len(spaces) == 1:
            return spaces.pop()
        return "device"

    def _arithmetic_constructor(self, struct, ctor):
        """``S(double x)``: CUDA converts any arithmetic argument to double
        (a standard conversion) and then to S.  On Metal double is f64, a
        class, and two user-defined conversions are not allowed, so S gets
        a delegating constructor template for every arithmetic type."""
        K = self.ci.CursorKind
        params = [p for p in ctor.get_children() if p.kind == K.PARM_DECL]
        if len(params) != 1:
            return
        canon = _canonical(params[0].type).replace("const ", "").strip()
        target = "f64" if canon == "double" else ("f32s" if canon == "float" and self.retain else None)
        if target is None:
            return
        name = struct.spelling
        end = self._statement_end(ctor)
        self.edits.insert(end, f" template <WOOF_ARITH(__WoofT)> {name}(__WoofT __woof_x) : "
                               f"{name}({target}(__woof_x)) {{}}", rank=50_000)

    def is_kernel(self, cursor) -> bool:
        K = self.ci.CursorKind
        return any(c.kind == K.ANNOTATE_ATTR and c.spelling == _GLOBAL
                   for c in cursor.get_children())

    def strip_linkage(self, cursor):
        tokens = list(cursor.get_tokens())
        if len(tokens) >= 2 and tokens[0].spelling == "extern" and tokens[1].spelling.startswith('"'):
            start = tokens[0].extent.start.offset
            end = tokens[1].extent.end.offset
            if len(tokens) > 2 and tokens[2].spelling == "{":
                end = tokens[2].extent.end.offset
                last = tokens[-1]
                if last.spelling == "}":
                    self.edits.replace(last.extent.start.offset, last.extent.end.offset, "")
            self.edits.replace(start, end, "")

    # -- kernels
    def rewrite_kernel(self, fn):
        """``extern "C" __global__ void f(...)`` -> an MSL kernel.

        Scalars arrive packed in one ``constant`` struct (C layout, offsets
        from libclang, so the launcher packs the same bytes); pointers bind
        one per slot, or -- past :data:`MAX_DIRECT_POINTERS` -- through a
        table of GPU addresses.  Both are copied into locals named as the
        CUDA parameters, so the body is untouched.
        """
        K = self.ci.CursorKind
        T = self.ci.TypeKind
        body = next((c for c in fn.get_children() if c.kind == K.COMPOUND_STMT), None)
        if body is None:
            return
        params = [c for c in fn.get_children() if c.kind == K.PARM_DECL]
        name = fn.spelling
        pointers, scalars, meta = [], [], []
        offset, max_align = 0, 1
        addr_tables = []
        for index, p in enumerate(params):
            t = p.type
            pname = self._renamed(p.spelling or f"__woof_unnamed{index}")
            if t.get_canonical().kind.name in ("INCOMPLETEARRAY", "CONSTANTARRAY"):
                self.refuse(p, f"kernel {name}: array parameter {pname!r}; declare it as a pointer")
            if t.get_canonical().kind == T.POINTER and t.get_pointee().get_canonical().kind == T.POINTER:
                # a table of device addresses (T**): a buffer of device pointers.
                # What it points at is invisible to MLX's hazard tracking.
                # Metal has no pointer-to-pointer buffers: the buffer is the
                # words, read back as addresses (__woof_addr_table).
                inner = t.get_pointee()
                target = inner.get_pointee()
                tspace = "const device" if target.is_const_qualified() else "device"
                pointers.append((f"__woof_tab_{pname}", "const device ulong*"))
                addr_tables.append(f"__woof_addr_table<{tspace} {self._spell_type(target, drop_const=True)}*> "
                                   f"{pname}{{__woof_tab_{pname}}};")
                meta.append(Param(pname, True, "unsigned long long", True))
                self._indirect_kernels.add(name)
            elif t.get_canonical().kind == T.POINTER:
                pointee = t.get_pointee()
                const = pointee.is_const_qualified()
                elem = self._spell_type(pointee, drop_const=True)
                pointers.append((pname, ("const device" if const else "device") + f" {elem}*"))
                meta.append(Param(pname, True, _canonical(pointee).replace("const ", "").strip(), const))
            else:
                canon = _canonical(t).replace("const ", "").strip()
                spelled = self._spell_type(t, drop_const=True)
                size, align = int(t.get_size()), max(int(t.get_align()), 1)
                offset = (offset + align - 1) // align * align
                scalars.append((pname, spelled))
                meta.append(Param(pname, False, canon, fmt=SCALAR_FORMATS.get(canon),
                                  size=size, offset=offset))
                offset += size
                max_align = max(max_align, align)
        global_rows = []
        for key in self._implicit_order(k for k in self._implicit_needs.get(fn.get_usr(), ()) if k.startswith("g:")):
            usr = next(u for u, kk in self._global_keys.items() if kk == key)
            g = self.module_globals[usr]
            const = g["space"] == "constant"
            space = "constant" if const else "device"
            if len(g["dims"]) > 1:
                # bound flat; the body indexes rows: T (*g)[d2]...
                rows = "".join(f"[{d}]" for d in g["dims"][1:])
                pointers.append((g["param"] + "_flat", f"{space} {g['elem']}*"))
                global_rows.append(f"{space} {g['elem']} (*{g['param']}){rows} = "
                                   f"({space} {g['elem']} (*){rows}){g['param']}_flat;")
                meta.append(Param(g["param"] + "_flat", True, g["ctype"], const, global_name=g["name"]))
            else:
                pointers.append((g["param"], f"{space} {g['elem']}*"))
                meta.append(Param(g["param"], True, g["ctype"], const, global_name=g["name"]))
        scalar_size = (offset + max_align - 1) // max_align * max_align if scalars else 0
        table = len(pointers) > MAX_DIRECT_POINTERS
        sig_params, prologue, structs = [], [], []
        if table:
            fields = "\n".join(f"    {ty} {pn};" for pn, ty in pointers)
            structs.append(f"struct __woof_table_{name} {{\n{fields}\n}};")
            sig_params.append(f"constant __woof_table_{name}& __woof_table [[buffer(0)]]")
            prologue += [f"{ty} {pn} = __woof_table.{pn};" for pn, ty in pointers]
            table_slot, scalar_slot = 0, 1
        else:
            sig_params += [f"{ty} {pn} [[buffer({i})]]" for i, (pn, ty) in enumerate(pointers)]
            table_slot, scalar_slot = -1, len(pointers)
        prologue += addr_tables
        if scalars:
            fields = "\n".join(f"    {ty} {pn};" for pn, ty in scalars)
            structs.append(f"struct __woof_scalars_{name} {{\n{fields}\n}};")
            sig_params.append(f"constant __woof_scalars_{name}& __woof_scalars [[buffer({scalar_slot})]]")
            prologue += [f"{ty} {pn} = __woof_scalars.{pn};" for pn, ty in scalars]
        else:
            scalar_slot = -1
        uses = sorted({(n.referenced.get_usr()) for n in fn.walk_preorder()
                       if n.kind == K.DECL_REF_EXPR and n.referenced is not None
                       and n.referenced.get_usr() in self.dynamic_shared}
                      | {k[4:] for k in self._implicit_needs.get(fn.get_usr(), ()) if k.startswith("dyn:")})
        if len(uses) > 1:
            self.refuse(fn, f"kernel {name} names {len(uses)} dynamic shared arrays; CUDA gives "
                            "them one base, Metal one threadgroup(0) buffer")
        for usr in uses:
            sname, selem = self.dynamic_shared[usr]
            sig_params.append(f"threadgroup {selem}* {sname} [[threadgroup(0)]]")
        sig_params += [
            "uint3 __woof_tid [[thread_position_in_threadgroup]]",
            "uint3 __woof_bid [[threadgroup_position_in_grid]]",
            "uint3 __woof_bdim [[threads_per_threadgroup]]",
            "uint3 __woof_gdim [[threadgroups_per_grid]]",
        ]
        # The signature: from the name token to the body's brace.
        tokens = list(fn.get_tokens())
        name_tok = next(t for t in tokens if t.spelling == name and t.kind.name == "IDENTIFIER")
        void_tok = tokens[tokens.index(name_tok) - 1]
        decl_start = fn.extent.start.offset
        if fn.semantic_parent is not None and fn.semantic_parent.kind == K.LINKAGE_SPEC:
            decl_start = min(decl_start, fn.semantic_parent.extent.start.offset)
        if structs:
            self.edits.insert(decl_start, "\n".join(structs) + "\n", rank=-30_000)
        self.edits.insert(void_tok.extent.start.offset, "[[kernel]] ", rank=-10_000)
        self.edits.replace(name_tok.extent.start.offset, body.extent.start.offset,
                           f"{name}(\n    " + ",\n    ".join(sig_params) + ")\n")
        prologue += global_rows
        if prologue:
            self.edits.insert(body.extent.start.offset + 1, "\n    " + "\n    ".join(prologue), rank=10_000)
        self.kernels[name] = KernelSignature(
            name, tuple(meta), dynamic_shared=bool(uses),
            pointer_mode="table" if table else "direct", table_slot=table_slot,
            scalar_slot=scalar_slot, scalar_size=scalar_size)

    def _spell_type(self, t, *, drop_const: bool) -> str:
        spelled = t.spelling
        if drop_const:
            spelled = re.sub(r"\bconst\b", "", spelled)
        spelled = re.sub(r"\b__restrict__\b|\brestrict\b|\bvolatile\b", "", spelled)
        spelled = re.sub(r"\bunsigned long long\b", "ulong", spelled)
        spelled = re.sub(r"\blong long\b", "long", spelled)
        spelled = re.sub(r"\bdouble\b", "f64", spelled)
        if self.retain:
            spelled = re.sub(r"\bfloat\b", "f32s", spelled)
        return " ".join(spelled.split())

    # -- expressions
    def visit_expr_tree(self, cursor, *, depth: int, in_kernel: bool):
        K = self.ci.CursorKind
        stack = [(cursor, depth)]
        while stack:
            node, d = stack.pop()
            if self.span(node) in self._folded:
                continue
            kind = node.kind
            if kind == K.CALL_EXPR and node.spelling == "printf":
                s, e = self.span(node)
                self.edits.replace(s, e, "((void)0)")
                continue
            if kind == K.DECL_REF_EXPR and node.spelling in ("threadIdx", "blockIdx", "blockDim", "gridDim"):
                # Kernels have these as parameters; device functions that
                # read them receive them (thread_builtins below).
                s, e = self.span(node)
                self.edits.replace(s, e, {"threadIdx": "__woof_tid", "blockIdx": "__woof_bid",
                                          "blockDim": "__woof_bdim", "gridDim": "__woof_gdim"}[node.spelling])
                continue
            if kind == K.INIT_LIST_EXPR:
                self.fold_init_list(node)
            if kind == K.UNARY_OPERATOR and _is_double(node.type):
                inner = list(node.get_children())
                toks = [t.spelling for t in node.get_tokens()]
                if (len(inner) == 1 and inner[0].kind == K.FLOATING_LITERAL and toks and toks[0] == "-"
                        and _is_double(inner[0].type)):
                    s, e = self.span(node)
                    word = _double_literal(self.src(inner[0])) ^ (1 << 63)
                    self.edits.replace(s, e, f"__woof_f64_lit(0x{word:016X}UL)")
                    continue
            if kind == K.FLOATING_LITERAL and _is_double(node.type):
                s, e = self.span(node)
                token = self.data[s:e].decode()
                self.edits.replace(s, e, f"__woof_f64_lit(0x{_double_literal(token):016X}UL)")
                continue
            if kind == K.UNEXPOSED_EXPR and _is_double(node.type):
                folded = self.fold_to_double(node)
                if folded is not None:
                    s, e = self.span(node)
                    self.edits.replace(s, e, folded)
                    continue
            if kind == K.UNEXPOSED_EXPR and self.retain:
                # f32s converts out to float and the integer types only
                # explicitly: spell C++'s implicit conversions as casts.
                children = list(node.get_children())
                if (len(children) == 1 and _is_float(children[0].type)
                        and node.type.get_canonical().kind.name in _INTEGRAL_KINDS):
                    s, e = self.span(children[0])
                    if s < e:
                        self.edits.wrap(s, e, f"(({self._spell_type(node.type, drop_const=True)})(", "))", d)
            if kind == K.UNEXPOSED_EXPR:
                children = list(node.get_children())
                if len(children) == 1 and _is_double(children[0].type) and not _is_double(node.type):
                    target = _canonical(node.type).replace("const ", "").strip()
                    if target and target not in ("void",) and node.type.kind.name not in ("RECORD", "ELABORATED"):
                        s, e = self.span(children[0])
                        if s < e:
                            spelled = self._spell_type(node.type, drop_const=True)
                            self.edits.wrap(s, e, f"(({spelled})(", "))", d)
            if kind == K.COMPOUND_ASSIGNMENT_OPERATOR:
                self.compound_assign(node, d)
            if kind == K.LAMBDA_EXPR:
                self.lambda_pointer_params(node)
            if kind == K.VAR_DECL and d > 0:
                self.local_pointer_decl(node)
                if node.storage_class == self.ci.StorageClass.STATIC and node.semantic_parent is not None \
                        and node.semantic_parent.kind.name in self._FUNCTION_KINDS:
                    self.static_local(node)
            for child in reversed(list(node.get_children())):
                stack.append((child, d + 1))

    _FUNCTION_KINDS = ("FUNCTION_DECL", "CXX_METHOD", "CONSTRUCTOR", "DESTRUCTOR",
                       "FUNCTION_TEMPLATE", "LAMBDA_EXPR")

    def lambda_pointer_params(self, node):
        """A lambda's pointer parameter takes whatever space its callers'
        arguments have: it becomes ``auto`` (a generic lambda).  Overload
        resolution is unchanged -- a lambda has one call operator."""
        for parm in node.get_children():
            if parm.kind.name != "PARM_DECL" or _indirection_kind(parm.type) != "ptr":
                continue
            toks = list(parm.get_tokens())
            name = next((t for t in toks if t.spelling == parm.spelling), None)
            if name is None or not toks:
                self.refuse(parm, f"cannot find the type of lambda parameter {parm.spelling!r}")
            self.edits.replace(toks[0].extent.start.offset, name.extent.start.offset, "auto ")
        # `-> T&` / `-> T*`: the space comes from the returned expression
        body = next((c for c in node.get_children() if c.kind.name == "COMPOUND_STMT"), None)
        if body is None:
            return
        toks = [t for t in node.get_tokens() if t.extent.end.offset <= body.extent.start.offset]
        arrow = next((k for k in range(len(toks) - 1, -1, -1) if toks[k].spelling == "->"), None)
        if arrow is None or arrow + 1 >= len(toks):
            return
        ret = [t.spelling for t in toks[arrow + 1:]]
        if ret[-1] == "*":
            self.edits.replace(toks[arrow + 1].extent.start.offset, toks[-1].extent.end.offset, "auto")
        elif ret[-1] == "&":
            # decltype(auto) of a parenthesised lvalue is that lvalue's reference, space and all
            self.edits.replace(toks[arrow + 1].extent.start.offset, toks[-1].extent.end.offset, "decltype(auto)")
            for r in body.walk_preorder():
                kids = list(r.get_children())
                if r.kind.name == "RETURN_STMT" and kids:
                    rs, re_ = self.span(kids[0])
                    self.edits.wrap(rs, re_, "(", ")", 0)

    def static_local(self, node):
        """MSL has no function-scope statics.  A constant one is the same
        value initialised per call, so ``static`` just goes; a mutable one
        (state kept across calls) has no spelling and is refused."""
        t = node.type
        while t.get_canonical().kind.name in ("CONSTANTARRAY", "INCOMPLETEARRAY"):
            t = t.get_canonical().element_type
        if not t.is_const_qualified():
            self.refuse(node, f"function-scope static {node.spelling!r} is mutable")
        tok = next((t for t in node.get_tokens() if t.spelling == "static"), None)
        if tok is not None:
            self.edits.replace(tok.extent.start.offset, tok.extent.end.offset, "")

    def fold_to_double(self, node):
        """``(double)<integer or float literal>``, implicit: the exact word,
        so the conversion is a constant expression (f64's widening is not)."""
        children = list(node.get_children())
        if len(children) != 1:
            return None
        child = children[0]
        while child.kind.name == "PAREN_EXPR" and len(list(child.get_children())) == 1:
            child = next(child.get_children())
        sign = 1
        if child.kind.name == "UNARY_OPERATOR":
            toks = [t.spelling for t in child.get_tokens()]
            inner = list(child.get_children())
            if len(inner) != 1 or not toks or toks[0] != "-":
                return None
            sign, child = -1, inner[0]
            while child.kind.name == "PAREN_EXPR" and len(list(child.get_children())) == 1:
                child = next(child.get_children())
        if child.kind.name == "INTEGER_LITERAL":
            text = self.src(child).rstrip("uUlL")
            if text.startswith(("0x", "0X")):
                value = int(text, 16)
            elif text.startswith(("0b", "0B")):
                value = int(text, 2)
            elif len(text) > 1 and text.startswith("0"):
                value = int(text, 8)
            else:
                value = int(text)
            word = struct.unpack("<Q", struct.pack("<d", float(sign * value)))[0]
            if float(sign * value) != sign * value:
                return None  # not exactly representable: leave it to f64's rounding
        elif child.kind.name == "FLOATING_LITERAL" and child.type.get_canonical().kind.name == "FLOAT":
            value = _binary32_of_decimal(self.src(child))
            if value is None:
                return None
            word = struct.unpack("<Q", struct.pack("<d", sign * value))[0]
        else:
            return None
        return f"__woof_f64_lit(0x{word:016X}UL)"

    def const_eval(self, node):
        """Evaluate a constant expression of literals: ("double", float) or
        ("int", value, bits, signed), or None.  binary64 + - * / are Python's
        (IEEE round-to-nearest); integers follow C's widths and wrapping.
        Float (binary32) literals are not evaluated: Python cannot parse
        decimal to binary32 without a double rounding."""
        kind = node.kind.name
        kids = [c for c in node.get_children() if c.kind.name not in ("TYPE_REF",)]
        t = node.type.get_canonical()
        if kind in ("PAREN_EXPR", "UNEXPOSED_EXPR", "CSTYLE_CAST_EXPR", "CXX_STATIC_CAST_EXPR",
                    "CXX_FUNCTIONAL_CAST_EXPR"):
            if len(kids) != 1:
                return None
            inner = self.const_eval(kids[0])
            return None if inner is None else self._convert(inner, t)
        if kind == "INTEGER_LITERAL":
            text = self.src(node).rstrip("uUlL")
            value = int(text, 8) if len(text) > 1 and text[0] == "0" and text[1] not in "xXbB" else int(text, 0)
            return self._convert(("int", value, 64, True), t)
        if kind == "FLOATING_LITERAL":
            if _is_double(node.type):
                return ("double", struct.unpack("<d", struct.pack("<Q", _double_literal(self.src(node))))[0])
            if _is_float(node.type):
                return self._convert(("double", _float_literal_value(self.src(node))), t)
            return None
        if kind == "DECL_REF_EXPR" and node.referenced is not None and node.referenced.kind.name == "VAR_DECL":
            var = node.referenced
            if not (var.type.is_const_qualified() or any(tk.spelling == "constexpr" for tk in var.get_tokens())):
                return None
            inits = [c for c in var.get_children() if c.kind.name not in ("TYPE_REF", "ANNOTATE_ATTR")]
            if not inits or self._evaluating.get(var.get_usr()):
                return None
            self._evaluating[var.get_usr()] = True
            try:
                v = self.const_eval(inits[-1])
            finally:
                self._evaluating[var.get_usr()] = False
            return None if v is None else self._convert(v, t)
        if kind == "UNARY_OPERATOR" and len(kids) == 1:
            op = next((tk.spelling for tk in node.get_tokens()), None)
            v = self.const_eval(kids[0])
            if v is None:
                return None
            if op == "-":
                return ("double", -v[1]) if v[0] == "double" else self._convert(("int", -v[1], v[2], v[3]), t)
            if op == "+":
                return v
            if op == "~" and v[0] == "int":
                return self._convert(("int", ~v[1], v[2], v[3]), t)
            return None
        if kind == "BINARY_OPERATOR" and len(kids) == 2:
            a, b = self.const_eval(kids[0]), self.const_eval(kids[1])
            if a is None or b is None:
                return None
            lhs_end = kids[0].extent.end.offset
            op = next((tk.spelling for tk in node.get_tokens() if tk.extent.start.offset >= lhs_end), None)
            if a[0] == "double" or b[0] == "double":
                x, y = float(a[1]), float(b[1])
                try:
                    r = {"+": lambda: x + y, "-": lambda: x - y, "*": lambda: x * y,
                         "/": lambda: x / y}[op]()
                except (KeyError, ZeroDivisionError, OverflowError):
                    return None
                # a binary32 operation: the binary64 result rounded once more is
                # the correctly rounded one (53 >= 2*24 + 2 bits)
                return self._convert(("double", r), t) if t.kind.name == "FLOAT" else ("double", r)
            x, y = a[1], b[1]
            ops = {"+": lambda: x + y, "-": lambda: x - y, "*": lambda: x * y,
                   "<<": lambda: x << y, ">>": lambda: x >> y, "|": lambda: x | y,
                   "&": lambda: x & y, "^": lambda: x ^ y,
                   "/": lambda: int(x / y) if y else None, "%": lambda: (x - y * int(x / y)) if y else None}
            if op not in ops:
                return None
            r = ops[op]()
            return None if r is None else self._convert(("int", r, 64, True), t)
        return None

    @staticmethod
    def _convert(value, t):
        kind = t.kind.name
        if kind == "DOUBLE":
            return ("double", float(value[1]))
        if kind == "FLOAT":
            return ("double", _round_to_binary32(float(value[1])))
        ints = {"INT": (32, True), "UINT": (32, False), "LONG": (64, True), "ULONG": (64, False),
                "LONGLONG": (64, True), "ULONGLONG": (64, False), "SHORT": (16, True),
                "USHORT": (16, False), "CHAR_S": (8, True), "SCHAR": (8, True), "UCHAR": (8, False),
                "CHAR_U": (8, False), "BOOL": (1, False)}
        if kind in ints:
            bits, signed = ints[kind]
            v = int(value[1]) if value[0] == "int" else int(value[1])  # C truncation toward zero
            if kind == "BOOL":
                return ("int", int(v != 0), 1, False)
            v &= (1 << bits) - 1
            if signed and v >> (bits - 1):
                v -= 1 << bits
            return ("int", v, bits, signed)
        return None

    def fold_constant_double(self, node):
        """Replace a binary64 constant expression by its exact word."""
        if not _is_double(node.type):
            return False
        if node.kind.name == "FLOATING_LITERAL":
            return False  # the literal rule spells it
        v = self.const_eval(node)
        if v is None or v[0] != "double":
            return False
        word = struct.unpack("<Q", struct.pack("<d", v[1]))[0]
        s, e = self.span(node)
        self.edits.replace(s, e, f"__woof_f64_lit(0x{word:016X}UL)")
        return True

    def fold_init_list(self, node):
        """Integer literals initialising binary64 elements or fields:
        ``{1.0, 0, -2.5}`` for a double table, as exact words (a constant
        initialiser cannot run f64's converting constructor)."""
        K = self.ci.CursorKind
        t = node.type.get_canonical()
        kids = list(node.get_children())
        if t.kind.name in ("CONSTANTARRAY", "INCOMPLETEARRAY"):
            targets = [t.element_type] * len(kids)
        elif t.kind.name == "RECORD":
            decl = t.get_declaration()
            targets = [f.type for f in decl.get_children() if f.kind == K.FIELD_DECL]
        else:
            return
        for child, target in zip(kids, targets):
            if not _is_double(target):
                continue
            if child.kind.name not in ("INIT_LIST_EXPR", "FLOATING_LITERAL") and self.fold_constant_double(child):
                self._folded.add(self.span(child))
                continue
            literal, sign = child, 1
            if literal.kind.name == "UNEXPOSED_EXPR":
                inner = list(literal.get_children())
                literal = inner[0] if len(inner) == 1 else literal
            if literal.kind.name == "UNARY_OPERATOR":
                toks = [tk.spelling for tk in literal.get_tokens()]
                inner = list(literal.get_children())
                if toks and toks[0] == "-" and len(inner) == 1:
                    sign, literal = -1, inner[0]
            if literal.kind.name != "INTEGER_LITERAL":
                continue
            text = self.src(literal).rstrip("uUlL")
            value = sign * int(text, 0) if not (len(text) > 1 and text[0] == "0" and text[1] not in "xXbB") \
                else sign * int(text, 8)
            if float(value) != value:
                continue
            word = struct.unpack("<Q", struct.pack("<d", float(value)))[0]
            s, e = self.span(child)
            self.edits.replace(s, e, f"__woof_f64_lit(0x{word:016X}UL)")

    def compound_assign(self, node, depth):
        children = list(node.get_children())
        if len(children) != 2:
            return
        lhs, rhs = children
        if _is_double(lhs.type) or not _is_double(rhs.type):
            return
        lhs_text = self.src(lhs)
        if not re.fullmatch(r"[\w\s\.\[\]\(\)\->+*]*", lhs_text) or "++" in lhs_text or "--" in lhs_text:
            self.refuse(node, f"compound assignment with a double right side to {lhs_text!r} "
                              "would evaluate the target twice")
        tokens = [t.spelling for t in node.get_tokens()]
        op = next((t for t in tokens if t in ("+=", "-=", "*=", "/=")), None)
        if op is None:
            self.refuse(node, "unsupported compound assignment with a double right side")
        spelled = self._spell_type(lhs.type, drop_const=True)
        s, _ = self.span(lhs)
        rs, re_ = self.span(rhs)
        # lhs OP= rhs  ->  lhs = (T)((f64)(lhs) OP (rhs))
        self.edits.replace(lhs.extent.end.offset, rs, f" = (({spelled})(f64({lhs_text}) {op[0]} (")
        self.edits.insert(re_, ")))", rank=-depth)

    # -- tokens
    def rewrite_tokens(self):
        import clang.cindex as ci
        tokens = list(self.tu.get_tokens(extent=self.tu.cursor.extent))
        n = len(tokens)
        i = 0
        while i < n:
            t = tokens[i]
            if t.location.file is None or t.location.file.name != "unit.cpp":
                i += 1
                continue
            sp = t.spelling
            if t.kind == ci.TokenKind.KEYWORD:
                if sp == "float" and self.retain:
                    at = t.extent.start.offset
                    if any(a <= at < b for a, b in self._keep_float_ranges):
                        i += 1
                        continue
                    if not self._inside_replacement(t.extent.start.offset):
                        self.edits.replace(t.extent.start.offset, t.extent.end.offset, "f32s")
                elif sp == "double":
                    if i > 0 and tokens[i - 1].spelling == "long":
                        raise TranslationError(f"{self.unit}: line {t.location.line}: long double")
                    if not self._inside_replacement(t.extent.start.offset):
                        self.edits.replace(t.extent.start.offset, t.extent.end.offset, "f64")
                elif sp == "long" and i + 1 < n and tokens[i + 1].spelling == "long":
                    start = t.extent.start.offset
                    unsigned = i > 0 and tokens[i - 1].spelling == "unsigned"
                    if unsigned:
                        start = tokens[i - 1].extent.start.offset
                    if not self._inside_replacement(start):
                        self.edits.replace(start, tokens[i + 1].extent.end.offset,
                                           "ulong" if unsigned else "long")
                    i += 2
                    continue
            elif t.kind == ci.TokenKind.IDENTIFIER and sp in _VECTOR_RENAMES:
                if not self._inside_replacement(t.extent.start.offset):
                    self.edits.replace(t.extent.start.offset, t.extent.end.offset, _VECTOR_RENAMES[sp])
            elif t.kind == ci.TokenKind.IDENTIFIER and sp in METAL_RESERVED:
                if not self._inside_replacement(t.extent.start.offset):
                    self.edits.replace(t.extent.start.offset, t.extent.end.offset, self._renamed(sp))
            i += 1

    def _inside_replacement(self, offset: int) -> bool:
        return any(s <= offset < e for s, e, _ in self.edits.replacements)

    @staticmethod
    def _renamed(name: str) -> str:
        return name + "_w" if name in METAL_RESERVED else name


def prelude(*, contract: bool, retain_subnormals: bool = False) -> str:
    soft = (METAL_DIR / "f64_soft.h").read_text()
    text = (METAL_DIR / "prelude.h").read_text().replace("// F64_SOFT_H_TEXT", soft)
    if retain_subnormals:
        text += "\n" + (METAL_DIR / "f32s.h").read_text()
    # The prelude always compiles uncontracted: __fmul_rn and kin promise one
    # rounding each, and LLVM records contraction per instruction, so their
    # bodies stay unfusable after inlining into a unit compiled contract(fast).
    after = "\n#pragma METAL fp contract(fast)\n" if contract else ""
    return "#pragma METAL fp contract(off)\n" + text + after


VENDOR_LIBM64 = HERE / "vendor_libm64.cuh"
F64CR = HERE.parents[1] / "core" / "kernels" / "f64cr.cuh"
_VENDOR_CALL = re.compile(r"\b(exp|log|log10|sin|cos|atan|pow|atan2|hypot|log2|expm1|tanh|tgammaf?|lgamma|log1pf?|cbrtf?)\s*\(")
CR_CBRT = HERE.parents[1] / "core" / "kernels" / "cr_cbrt_flt64.cuh"


def compose(source: str) -> str:
    """Prepend the binary64 libm (vendor_libm64.cuh, with f64cr.cuh) to a
    unit that calls one of CUDA's binary64 libm names."""
    if not _VENDOR_CALL.search(source):
        return source
    f64cr = F64CR.read_text(encoding="utf-8")
    hook = "#define WCR_PLATFORM_SIN(x) sin(x)"
    if hook not in f64cr:
        raise TranslationError("f64cr.cuh no longer defines its CUDA platform-sin hook as expected")
    f64cr = f64cr.replace(hook, "#define WCR_PLATFORM_SIN(x) wlm_sin_huge(x)")
    cbrt = ""
    if re.search(r"\bcbrt\s*\(", source) and "cr_cbrt(" not in source:
        # binary64 cbrt: CORE-MATH's correctly rounded cr_cbrt
        cbrt = (CR_CBRT.read_text(encoding="utf-8")
                + "\n__device__ __forceinline__ double cbrt(double x) { return g64_cbrt::cr_cbrt(x); }\n")
    return ("__device__ double wlm_sin_huge(double x);\n" + f64cr + "\n"
            + VENDOR_LIBM64.read_text(encoding="utf-8") + "\n" + cbrt + source)


def option_defines(options) -> tuple[str, ...]:
    """The ``-DNAME[=VALUE]`` macros among NVRTC options."""
    out, it = [], iter(options or ())
    for o in it:
        if o in ("-D", "--define-macro"):
            out.append(next(it, ""))
        elif o.startswith("--define-macro="):
            out.append(o.split("=", 1)[1])
        elif o.startswith("-D") and len(o) > 2:
            out.append(o[2:])
    return tuple(d for d in out if d)


def flatten_inheritance(text: str, unit: str) -> str:
    """Metal has no derived classes.  ``struct D : B<A...> {...}`` becomes
    D with B's members copied in (B's template parameters replaced by the
    base clause's arguments), and every function parameter of type
    ``[const] B<...>&`` becomes a deduced template parameter, so it accepts
    B and D alike as the derived-to-base binding did.  Single, non-virtual
    bases defined in the unit; anything else is refused by name."""
    import clang.cindex as ci
    if not re.search(r"\b(struct|class)\s+\w+\s*:\s*(public\s+)?\w", text):
        return text
    tu = parse(text)
    K = ci.CursorKind
    data = text.encode()
    edits = Edits(data)

    def src(c):
        return data[c.extent.start.offset:c.extent.end.offset].decode()

    def main(c):
        return c.location.file is not None and c.location.file.name == "unit.cpp"

    def definition(c):
        d = c.get_definition() if c is not None else None
        return d

    bases = {}   # base definition usr -> base cursor
    for d in tu.cursor.walk_preorder():
        if d.kind not in (K.STRUCT_DECL, K.CLASS_DECL, K.CLASS_TEMPLATE) or not main(d) or not d.is_definition():
            continue
        specs = [c for c in d.get_children() if c.kind == K.CXX_BASE_SPECIFIER]
        if not specs:
            continue
        if len(specs) > 1:
            raise TranslationError(f"{unit}: line {d.location.line}: {d.spelling!r} has several bases")
        spec = specs[0]
        if any(t.spelling == "virtual" for t in spec.get_tokens()):
            raise TranslationError(f"{unit}: line {d.location.line}: virtual base of {d.spelling!r}")
        base_ref = next((c for c in spec.get_children() if c.kind in (K.TEMPLATE_REF, K.TYPE_REF)), None)
        base = base_ref.referenced if base_ref is not None else None
        if base is None or not main(base):
            raise TranslationError(f"{unit}: line {d.location.line}: base of {d.spelling!r} is not in the unit")
        base_def = definition(base) or base
        bases[base_def.get_usr()] = base_def
        # the base's members, its template parameters replaced by the arguments
        spec_tokens = [t.spelling for t in spec.get_tokens()]
        args = []
        if "<" in spec_tokens:
            inner = spec_tokens[spec_tokens.index("<") + 1:]
            depth, cur = 0, []
            for t in inner:
                if t == "<":
                    depth += 1
                elif t == ">":
                    if depth == 0:
                        break
                    depth -= 1
                if t == "," and depth == 0:
                    args.append(" ".join(cur)); cur = []
                else:
                    cur.append(t)
            args.append(" ".join(cur))
        params = [c.spelling for c in base_def.get_children()
                  if c.kind in (K.TEMPLATE_TYPE_PARAMETER, K.TEMPLATE_NON_TYPE_PARAMETER)]
        # members only: constructors and destructors are not inherited in C++
        # (copied, a base constructor would parse as a deduction guide)
        parts, seen_starts = [], set()
        for m in base_def.get_children():
            if m.extent.start.offset in seen_starts:
                continue          # `int a, b;`: one declaration, several FIELD_DECLs
            seen_starts.add(m.extent.start.offset)
            if m.kind in (K.CONSTRUCTOR, K.DESTRUCTOR, K.TEMPLATE_TYPE_PARAMETER,
                          K.TEMPLATE_NON_TYPE_PARAMETER, K.CXX_BASE_SPECIFIER, K.ANNOTATE_ATTR):
                continue
            if m.kind.name == "CXX_ACCESS_SPEC_DECL":
                continue
            if m.kind == K.FIELD_DECL:
                # the whole declaration, every declarator, through its ';'
                semi = data.index(b";", m.extent.end.offset)
                parts.append(data[m.extent.start.offset:semi + 1].decode())
                continue
            text_m = src(m)
            end = m.extent.end.offset
            if data[end:end + 1] == b";":
                text_m += ";"
            parts.append(text_m)
        # each base constructor becomes `void __woof_base_init(params)`: its
        # member initialisers as assignments, then its body; a derived
        # constructor's `: B<...>(args)` becomes a call at the top of its body
        for ctor in (m for m in base_def.get_children() if m.kind == K.CONSTRUCTOR and m.is_definition()):
            ctoks = list(ctor.get_tokens())
            lp = next(i for i, t in enumerate(ctoks) if t.spelling == "(")
            depth, rp = 0, lp
            for i in range(lp, len(ctoks)):
                depth += ctoks[i].spelling == "("
                depth -= ctoks[i].spelling == ")"
                if depth == 0:
                    rp = i
                    break
            params_text = data[ctoks[lp].extent.start.offset:ctoks[rp].extent.end.offset].decode()
            body = next((c for c in ctor.get_children() if c.kind == K.COMPOUND_STMT), None)
            inits_text = data[ctoks[rp].extent.end.offset:body.extent.start.offset].decode().strip()
            assigns = []
            if inits_text.startswith(":"):
                depth, cur = 0, ""
                for ch in inits_text[1:] + ",":
                    if ch in "({[<":
                        depth += 1
                    elif ch in ")}]>":
                        depth -= 1
                    if ch == "," and depth == 0:
                        item = cur.strip()
                        if item:
                            brace, paren = item.find("{"), item.find("(")
                            if brace >= 0 and (paren < 0 or brace < paren):   # m{a, b}
                                mname, rest = item[:brace], item[brace:]
                                assigns.append(f"this->{mname.strip()} = {rest};")
                            else:                                             # m(a)
                                mname, _, rest = item.partition("(")
                                assigns.append(f"this->{mname.strip()} = ({rest.rsplit(')', 1)[0]});")
                        cur = ""
                    else:
                        cur += ch
            body_text = src(body)
            parts.append(f"void __woof_base_init{params_text} {{ {' '.join(assigns)} {body_text} }}")
        members = "\n    ".join(parts)
        for name, arg in zip(params, args):
            members = re.sub(rf"\b{re.escape(name)}\b", arg.strip(), members)
        # derived constructors: `: B<..>(args)` -> __woof_base_init(args); in the body
        for dctor in (m for m in d.get_children() if m.kind == K.CONSTRUCTOR and m.is_definition()):
            dtoks = list(dctor.get_tokens())
            body = next((c for c in dctor.get_children() if c.kind == K.COMPOUND_STMT), None)
            if body is None:
                continue
            head = [t for t in dtoks if t.extent.end.offset <= body.extent.start.offset]
            for i, t in enumerate(head):
                if t.spelling == base_def.spelling and i > 0 and head[i - 1].spelling in (":", ","):
                    # the base initialiser: through its closing parenthesis
                    j = i
                    while head[j].spelling != "(":
                        j += 1
                    depth = 0
                    for k in range(j, len(head)):
                        depth += head[k].spelling == "("
                        depth -= head[k].spelling == ")"
                        if depth == 0:
                            break
                    call_args = data[head[j].extent.start.offset:head[k].extent.end.offset].decode()
                    start = head[i - 1].extent.start.offset
                    end = head[k].extent.end.offset
                    if head[i - 1].spelling == ":" and k + 1 < len(head) and head[k + 1].spelling == ",":
                        end = head[k + 1].extent.end.offset   # keep a leading ':' for what follows
                        edits.replace(head[i].extent.start.offset, end, "")
                    else:
                        edits.replace(start, end, "")
                    edits.insert(body.extent.start.offset + 1, f" this->__woof_base_init{call_args};", 0)
                    break
        # `this->x` stays valid in D; base-qualified names do not occur in woof
        d_tokens = list(d.get_tokens())
        colon = next(t for t in d_tokens if t.spelling == ":")
        brace = next(t for t in d_tokens if t.spelling == "{")
        edits.replace(colon.extent.start.offset, brace.extent.start.offset, " ")
        edits.insert(brace.extent.end.offset, f"\n    /* members of {base_def.spelling} */ {members}\n", 0)
    if not bases:
        return text
    # parameters bound to a base by reference become deduced template types
    counter = 0
    for fn in tu.cursor.walk_preorder():
        if fn.kind not in (K.FUNCTION_DECL, K.FUNCTION_TEMPLATE, K.CXX_METHOD) or not main(fn):
            continue
        generic = []
        for p in fn.get_children():
            if p.kind != K.PARM_DECL or p.type.get_canonical().kind.name != "LVALUEREFERENCE":
                continue
            target = p.type.get_canonical().get_pointee().get_declaration()
            tdef = target.get_definition() if target is not None else None
            usr = (tdef.get_usr() if tdef is not None else None)
            template = getattr(tdef, "specialized_template", None) if tdef is not None else None
            if usr not in bases and not (template is not None and template.get_usr() in bases) \
                    and not any(target.spelling == b.spelling for b in bases.values()):
                continue
            counter += 1
            name = f"__WoofBase{counter}"
            const = p.type.get_pointee().is_const_qualified()
            ptoks = list(p.get_tokens())
            name_tok = next(t for t in reversed(ptoks) if t.spelling == p.spelling)
            edits.replace(ptoks[0].extent.start.offset, name_tok.extent.start.offset,
                          f"{'const ' if const else ''}{name}& ")
            generic.append(name)
        if generic:
            head = ", ".join(f"class {g}" for g in generic)
            toks = list(fn.get_tokens())
            if fn.kind == K.FUNCTION_TEMPLATE and toks[0].spelling == "template":
                lt = next(t for t in toks if t.spelling == "<")
                edits.insert(lt.extent.end.offset, head + ", ", 0)
            else:
                edits.insert(fn.extent.start.offset, f"template <{head}> ", 0)
    return edits.apply()


def translate(source: str, *, unit: str = "<unit>", contract: bool = False,
              retain_subnormals: bool = False, defines=()) -> TranslatedUnit:
    """CUDA unit text -> :class:`TranslatedUnit` (MSL plus kernel signatures).

    ``retain_subnormals``: the unit's ``float`` is ``f32s`` (gradual
    underflow in software where the GPU would flush; woof/backend/metal/
    f32s.h) -- CUDA's --ftz=false.
    """
    text = rewrite_ptx_asm(preprocess(compose(source), defines), unit)
    text = flatten_inheritance(text, unit)
    tu = parse(text)
    result = _Rewriter(text, tu, unit=unit, retain_subnormals=retain_subnormals).run()
    result.msl = (prelude(contract=contract, retain_subnormals=retain_subnormals)
                  + "\n// ---- translated unit: " + unit + "\n" + result.msl)
    result.key = hashlib.sha256(result.msl.encode()).hexdigest()
    return result
