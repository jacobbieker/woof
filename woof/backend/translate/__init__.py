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
import os
import pickle
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

#: The __CUDA_ARCH__ a translated unit is preprocessed for: the CUDA
#: comparison machine's (sm_86).  NVRTC compiles device code only, so CUDA
#: runs a unit's __CUDA_ARCH__ branches (TEMPO's glibc libm, MPAS WSM6's
#: global workspace); the facade's NVRTC architecture says the same.
CUDA_ARCH = 860

#: The CUDA keywords, as the preprocessor sees them.
PARSE_DEFINES = (
    f'-D__global__=__attribute__((annotate("{_GLOBAL}")))',
    f'-D__shared__=__attribute__((annotate("{_SHARED}")))',
    f'-D__constant__=__attribute__((annotate("{_CONSTANT}")))',
    "-D__device__=", "-D__host__=", "-D__forceinline__=inline",
    "-D__noinline__=__attribute__((noinline))", "-D__restrict__=", "-D__launch_bounds__(...)=",
    # register-count hints have no Metal meaning; __ldg is a plain load
    "-D__maxnreg__(...)=", "-D__ldg(p)=(*(p))",
    "-D__CUDACC__=1", "-D__CUDACC_RTC__=1", "-D__WOOF_METAL__=1", "-D__align__(n)=alignas(n)",
    # A by-value kernel parameter is constant data in Metal already:
    # __grid_constant__ is moot.
    f"-D__CUDA_ARCH__={CUDA_ARCH}", "-D__grid_constant__=",
    # NVRTC's own spelling of the keywords (a unit that #undefs __global__
    # restores it as `__location__(global)`)
    "-D__location__(x)=__woof_location_##x",
    f'-D__woof_location_global=__attribute__((annotate("{_GLOBAL}")))',
    f'-D__woof_location_shared=__attribute__((annotate("{_SHARED}")))',
    f'-D__woof_location_constant=__attribute__((annotate("{_CONSTANT}")))',
    "-D__woof_location_device=", "-D__woof_location_host=",
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


def preprocess(source: str, defines=(), include_dirs=()) -> str:
    """``defines``: the -D macros the unit's NVRTC options carry;
    ``include_dirs``: its -I directories, searched before woof's kernels."""
    result = subprocess.run(
        [_clang(), "-E", "-P", "-x", "c++", "-std=c++17", "-nostdinc", "-nostdinc++",
         "-isystem", str(INCLUDE), *[f"-I{d}" for d in (*include_dirs, *KERNEL_INCLUDE_DIRS)],
         "-Wno-pragma-once-outside-header", *PARSE_DEFINES, *[f"-D{d}" for d in defines], "-"],
        input=source, capture_output=True, text=True)
    if result.returncode != 0:
        raise TranslationError("preprocessing failed:\n" + result.stderr[-4000:])
    return result.stdout


# ---------------------------------------------------------------- inline PTX

#: An operand expression: parentheses nest three deep
#: (``"l"((unsigned long long)__double_as_longlong(a))``).
_ASM_EXPR = r'(?:[^()]|\((?:[^()]|\((?:[^()]|\([^()]*\))*\))*\))*'
_ASM = re.compile(
    r'\basm\s*(?:volatile\s*)?\(\s*((?:"(?:[^"\\]|\\.)*"\s*)+)'
    r':\s*"=([a-z])"\s*\((' + _ASM_EXPR + r')\)\s*'
    r':\s*((?:"[a-z]"\s*\(' + _ASM_EXPR + r'\)\s*,?\s*)+)\)\s*;')
_ASM_IN = re.compile(r'"([a-z])"\s*\((' + _ASM_EXPR + r')\)')
_PTX_CMP = {"eq": "(({a}) == ({b}))", "ne": "(({a}) < ({b}) || ({a}) > ({b}))",
            "lt": "(({a}) < ({b}))", "le": "(({a}) <= ({b}))",
            "gt": "(({a}) > ({b}))", "ge": "(({a}) >= ({b}))"}


#: Register widths of the asm operand constraints the bit translation reads.
_PTX_BITS = {"f": 32, "r": 32, "d": 64, "l": 64}
_PTX_WIDTH = {"pred": 1, "b32": 32, "u32": 32, "s32": 32, "f32": 32,
              "b64": 64, "u64": 64, "s64": 64, "f64": 64}
_PTX_BITOP = {"and": "&", "or": "|", "xor": "^"}


def _ptx_bit_block(template: str, out_constraint: str, out: str, constraints, ins):
    """Straight-line PTX on register bits -> a C++ block, or None.

    Reads ``.reg`` declarations, ``mov``, ``and``/``or``/``xor`` on
    ``.b32``/``.b64``, ``setp.<cmp>.f32|f64`` (no ``.ftz``) and ``selp``.
    Every register is its bits (``unsigned int``, ``unsigned long long``;
    a predicate is ``bool``), and an ``f``/``d`` operand enters and leaves by
    its bit pattern, so a move or a select keeps a subnormal, a signed zero
    and a NaN payload as the PTX does.  A binary32 comparison is
    ``__woof_ieee_<cmp>`` (f32s.h), which reads a subnormal as itself, as
    ``setp`` without ``.ftz`` does; Metal's own compare reads it as zero.
    Breakage it prevents: the MPAS physics units' sign flips (``xor.b32
    ..., 0x80000000``), bit moves and NaN-ordered min/max selects were
    refused, so the hex lane's physics could not load on Metal."""
    operands = [(out_constraint, out)] + list(zip(constraints, ins))
    if any(c not in _PTX_BITS for c, _ in operands):
        return None
    regs: dict[str, int] = {}
    decls, code = [], []

    def name(reg):
        return "__woof_ptx_" + reg

    def ctype(width):
        return {1: "bool", 32: "unsigned int", 64: "unsigned long long"}[width]

    def value(tok, width):
        m = re.fullmatch(r"%(\d+)", tok)
        if m:
            k = int(m.group(1))
            if k == 0 or k >= len(operands) or _PTX_BITS[operands[k][0]] != width:
                return None
            c, e = operands[k]
            return {"f": f"__float_as_uint({e})", "r": f"((unsigned int)({e}))",
                    "d": f"((unsigned long long)__double_as_longlong({e}))",
                    "l": f"((unsigned long long)({e}))"}[c]
        m = re.fullmatch(r"%?([A-Za-z_]\w*)", tok)
        if m:
            return name(m.group(1)) if regs.get(m.group(1)) == width else None
        m = re.fullmatch(r"0[fF]([0-9a-fA-F]{8})|0[dD]([0-9a-fA-F]{16})", tok)
        if m:
            digits = m.group(1) if width == 32 else m.group(2)
            return None if digits is None else f"0x{digits}{'u' if width == 32 else 'ull'}"
        m = re.fullmatch(r"0[xX][0-9a-fA-F]+|\d+", tok)
        if m:
            return f"{tok}{'u' if width == 32 else 'ull'}"
        return None

    def target(tok, width):
        if tok == "%0":
            if _PTX_BITS[out_constraint] != width:
                return None
            regs.setdefault("0", width)
            return name("0")
        m = re.fullmatch(r"%?([A-Za-z_]\w*)", tok)
        return name(m.group(1)) if m and regs.get(m.group(1)) == width else None

    def compare(cmp, s, t, width):
        if width == 32:
            return f"__woof_ieee_{cmp}(__uint_as_float({s}), __uint_as_float({t}))"
        return _PTX_CMP[cmp].format(a=f"__longlong_as_double((long long)({s}))",
                                    b=f"__longlong_as_double((long long)({t}))")

    for instr in (s.strip() for s in template.replace("{", " ").replace("}", " ").split(";")):
        if not instr:
            continue
        op, _, rest = instr.partition(" ")
        args = [t.strip() for t in rest.split(",")] if rest.strip() else []
        parts = op.split(".")
        if op == ".reg":
            kind, _, names = rest.strip().partition(" ")
            if kind.lstrip(".") not in _PTX_WIDTH:
                return None
            for reg in names.split(","):
                reg = reg.strip().lstrip("%")
                if not re.fullmatch(r"[A-Za-z_]\w*", reg) or reg in regs:
                    return None
                regs[reg] = _PTX_WIDTH[kind.lstrip(".")]
                decls.append(f"{ctype(regs[reg])} {name(reg)};")
            continue
        width = _PTX_WIDTH.get(parts[-1])
        if width in (None, 1):
            return None
        if parts[0] == "mov" and len(parts) == 2 and len(args) == 2:
            d, s = target(args[0], width), value(args[1], width)
            if d is None or s is None:
                return None
            code.append(f"{d} = {s};")
        elif parts[0] in ("and", "or", "xor") and len(parts) == 2 and len(args) == 3:
            d, s, t = target(args[0], width), value(args[1], width), value(args[2], width)
            if None in (d, s, t):
                return None
            code.append(f"{d} = {s} {_PTX_BITOP[parts[0]]} {t};")
        elif (parts[0] == "setp" and len(parts) == 3 and parts[1] in _PTX_CMP
              and parts[2] in ("f32", "f64") and len(args) == 3):
            p = args[0].lstrip("%")
            s, t = value(args[1], width), value(args[2], width)
            if regs.get(p) != 1 or None in (s, t):
                return None
            code.append(f"{name(p)} = {compare(parts[1], s, t, width)};")
        elif parts[0] == "selp" and len(parts) == 2 and len(args) == 4:
            p = args[3].lstrip("%")
            d, s, t = target(args[0], width), value(args[1], width), value(args[2], width)
            if regs.get(p) != 1 or None in (d, s, t):
                return None
            code.append(f"{d} = {name(p)} ? {s} : {t};")
        else:
            return None
    if regs.get("0") is None:
        return None
    result = name("0")
    store = {"f": f"__uint_as_float({result})", "r": result,
             "d": f"__longlong_as_double((long long){result})", "l": result}[out_constraint]
    return ("{ " + " ".join([f"{ctype(regs['0'])} {result};", *decls, *code, f"{out} = {store};"])
            + " }")


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
        constraints = [c for c, _ in _ASM_IN.findall(m.group(4))]
        ins = [x.strip() for _, x in _ASM_IN.findall(m.group(4))]
        a = ins[0] if ins else None
        b = ins[1] if len(ins) > 1 else None
        # No .ftz: CUDA keeps binary32 subnormals here whatever -ftz says,
        # so these are the IEEE ops in any unit (f32s.h __woof_ieee_*).
        mm = re.fullmatch(r"(add|sub|mul|div)\.rn\.f32 %0, ?%1, ?%2;", body)
        if mm:
            return f"{out} = __woof_ieee_{mm.group(1)}({a}, {b});"
        mm = re.fullmatch(r"setp\.(eq|ne|lt|le|gt|ge)\.f32 %?\w+, ?%1, ?%2; selp\.u32 %0, ?1, ?0, ?%?\w+;", body)
        if mm:
            return f"{out} = {_PTX_CMP[mm.group(1)].format(a=a, b=b)} ? 1u : 0u;"
        mm = re.fullmatch(r"setp\.(eq|ne|lt|le|gt|ge)\.f32 %?\w+, ?%1, ?%2; "
                          r"selp\.f32 %0, ?0f3F800000, ?0f00000000, ?%?\w+;", body)
        if mm:
            return f"{out} = {_PTX_CMP[mm.group(1)].format(a=a, b=b)} ? 1.0f : 0.0f;"
        if re.fullmatch(r"min\.f32 %0, ?%1, ?%2;", body):
            return f"{out} = __woof_ieee_min({a}, {b});"
        if re.fullmatch(r"max\.f32 %0, ?%1, ?%2;", body):
            return f"{out} = __woof_ieee_max({a}, {b});"
        if re.fullmatch(r"cvt\.rn\.f32\.f64 %0, ?%1;", body):
            return f"{out} = __double2float_rn({a});"
        block = _ptx_bit_block(template, m.group(2), out, constraints, ins)
        if block is not None:
            return block
        raise TranslationError(f"{unit}: inline PTX with no Metal translation: {body!r}")
    out = _ASM.sub(one, text)
    # `asm("trap;")`: the prelude's __trap, as CUDA's own spelling.
    out = re.sub(r'\basm\s*(?:volatile\s*)?\(\s*"\s*trap\s*;\s*"\s*\)\s*;', "__trap();", out)
    # `asm volatile("" ::: "memory")`: a compiler barrier (no instruction).
    # Values cross it through memory: a fence on this thread's device
    # memory, the same promise to Metal's compiler (acoustic_fusion's
    # phase separator).
    out = re.sub(r'\basm\s*(?:volatile\s*)?\(\s*""\s*:\s*:\s*:\s*"memory"\s*\)\s*;',
                 "__woof_compiler_fence();", out)
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

#: The rank of a goto wrapper's closing text.  A wrapper that closes where
#: another opens (a label right after the statement a region starts at)
#: closes first; inner closings still precede outer ones (rank + 1).
_GOTO_CLOSE = -200_000_000


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

    @staticmethod
    def _drop_deleted(reps):
        """Rewrites inside a range another rewrite deletes are moot: a
        module global's declaration goes (its initialiser becomes the
        buffer's bytes) while a literal in it is also respelled (f2c's
        ``doublereal c_b2 = 10.;`` in gfdleta_co2_init and cam3_emissivity).
        ``reps`` sorted; returned sorted."""
        if not any(r[2] == "" for r in reps):
            return reps
        out, cut = [], None
        for r in sorted(reps, key=lambda r: (r[0], -r[1])):
            if cut is not None and r[1] <= cut[1] and (r[0], r[1]) != cut:
                continue
            out.append(r)
            if r[2] == "" and (cut is None or r[1] > cut[1]):
                cut = (r[0], r[1])
        return sorted(out)

    def _render(self, start, end, reps, ins) -> str:
        reps = sorted(r for r in reps if start <= r[0] and r[1] <= end)
        reps = self._drop_deleted(reps)
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
    def __init__(self, text: str, tu, *, unit: str, retain_subnormals: bool = False, macros=None):
        import clang.cindex as ci
        self.ci = ci
        self.text = text
        self.data = text.encode()
        self.tu = tu
        self.unit = unit
        self.retain = retain_subnormals
        #: the largest integer value each macro is #defined to in the unit as given
        self.macros = dict(macros or {})
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
        self._split_decls: set = set()
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
        self._kernel_cursors = self._kernels(top)
        self.generic_structs = self._space_generic_structs(top)
        self.spaces = SpaceAnalysis(self)
        # a field given constant data anywhere (the probe saw every function)
        # is constant before the first call passes it on
        self.spaces.field_spaces.update(self._constant_fields)
        self.spaces.collect(top)
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
        self.inline_pointer_functions()
        self.table_parameters()
        self.table_field_stores(top)
        self.generic_constructions(top)
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

    def _space_generic_structs(self, top) -> set:
        """Structs whose pointer field is given pointers in two address
        spaces: a probe of the space analysis records every field's spaces
        (refusing nothing for them), and those become space-generic --
        templated on the space of their data (generic_struct)."""
        from woof.backend.translate.spaces import SpaceAnalysis
        self.generic_structs = set()
        self._constant_fields = {}
        self._probing_fields = True
        try:
            probe = self.spaces = SpaceAnalysis(self)
            probe.collect(top)
            probe.run(self._kernel_cursors)
        except TranslationError:
            return set()        # the real pass names what it cannot do
        finally:
            self._probing_fields = False
            self.spaces = None
        self._constant_fields = {key: set(spaces) for key, spaces in probe.field_spaces.items()
                                 if spaces == {"constant"}}
        return {owner for (owner, _field), spaces in probe.field_spaces.items() if len(spaces) > 1}

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
        # A file-scope static_assert names the launch builtins only in
        # unevaluated operands (decltype(blockIdx.x)): their types, not a
        # thread's parameters, which exist only inside functions.
        self._file_scope_builtins = cursor.kind == K.STATIC_ASSERT
        try:
            self.visit_expr_tree(cursor, depth=0, in_kernel=False)
        finally:
            self._file_scope_builtins = False

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

    def program_var_space(self, cursor) -> str:
        """The space a program-scope variable's storage has in the MSL
        (program_scope_var): a mutable ``__device__`` global is a module
        buffer in ``device`` memory, everything else ``constant``."""
        K = self.ci.CursorKind
        if any(c.kind == K.ANNOTATE_ATTR and c.spelling == _CONSTANT for c in cursor.get_children()):
            return "constant"
        t = cursor.type
        base = t
        while base.kind.name in ("CONSTANTARRAY", "INCOMPLETEARRAY"):
            base = base.element_type
        return "constant" if (base.is_const_qualified() or t.is_const_qualified()) else "device"

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
        cell = None
        if not dims and elem.rstrip().endswith("*") and space != "constant":
            # a pointer global (`__device__ T* p;`, set by one kernel and
            # read by others): Metal binds no buffer of pointers, so the
            # buffer is one address word and every reference a
            # __woof_ptr_cell (prelude).  It points into device memory.
            cell = f"device {elem.rstrip()}"
            elem = "ulong"
        index = len(self.module_globals)
        self.module_globals[cursor.get_usr()] = {
            "name": cursor.spelling, "elem": elem, "dims": dims, "space": space,
            "nbytes": int(t.get_size()), "param": f"__woof_g_{index}",
            "ctype": _canonical(self._base_type(t)).replace("const ", "").strip(), "init": init, "cell": cell}
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
        if key in self._qualified_sites or key in self._split_decls:
            return
        if self._split_mixed_declaration(stmt, declared):
            self._split_decls.add(key)
            return
        fn = node.semantic_parent
        default = "thread" if canon == "LVALUEREFERENCE" else "device"
        self._qualified_sites[key] = (stmt, [d.get_usr() for d in declared
                                             if d.type.get_canonical().kind.name in ("POINTER", "LVALUEREFERENCE")],
                                      fn.get_usr() if fn is not None else None, default)

    def _split_mixed_declaration(self, stmt, declared):
        """``T *a = p, b, *c = q;``: pointers and non-pointers in one
        declaration, where a space qualifier on the statement would qualify
        ``b`` too (cam3_emissivity: automatic variable qualified with an
        address space).  When every pointer declarator has an initialiser the
        statement becomes one declaration each, the pointers ``auto`` (their
        space deduced, right in every instance).  False otherwise."""
        K = self.ci.CursorKind
        if len(declared) < 2:
            return False
        kinds = [d.type.get_canonical().kind.name for d in declared]
        if "POINTER" not in kinds or all(k == "POINTER" for k in kinds) or "LVALUEREFERENCE" in kinds:
            return False
        for d, k in zip(declared, kinds):
            init = [c for c in d.get_children() if c.kind not in (K.TYPE_REF, K.NAMESPACE_REF, K.TEMPLATE_REF)]
            if k == "POINTER" and (not init or self._is_null(init[-1])):
                return False
        toks = list(stmt.get_tokens())
        names = []
        for d in declared:
            at = next((i for i, t in enumerate(toks) if t.kind.name == "IDENTIFIER" and t.spelling == d.spelling
                       and t.extent.start.offset >= d.extent.start.offset
                       and (not names or i > names[-1])), None)
            if at is None:
                return False
            names.append(at)
        start = names[0]
        while start > 0 and toks[start - 1].spelling in ("*", "const", "__restrict__"):
            start -= 1
        if start == 0:
            return False
        if any(t.spelling in ("const", "static", "volatile") for t in toks[:start]):
            return False
        # the non-pointer declarators' type, spelled as the translator spells
        # types (f32s, f64), since inserted text is not rewritten
        scalars = [d for d, k in zip(declared, kinds) if k not in ("POINTER", "CONSTANTARRAY", "INCOMPLETEARRAY")]
        if scalars:
            base = self._spell_type(scalars[0].type, drop_const=True)
        else:
            array = next(d for d, k in zip(declared, kinds) if k != "POINTER")
            base = self._spell_type(array.type.get_canonical().element_type, drop_const=True)
        depth, commas = 0, []
        for i in range(names[0], len(toks)):
            sp = toks[i].spelling
            if sp in ("(", "[", "{", "<"):
                depth += 1
            elif sp in (")", "]", "}", ">"):
                depth -= 1
            elif sp == "," and depth == 0:
                commas.append(i)
        if len(commas) != len(declared) - 1 or any(not (commas[j] < names[j + 1]) for j in range(len(commas))):
            return False
        starts = [start] + [c + 1 for c in commas]
        for j, (d, k) in enumerate(zip(declared, kinds)):
            seg = starts[j]
            if j == 0:
                if k == "POINTER":
                    # the written type stays (its tokens are respelled
                    # elsewhere) and declares an unused placeholder
                    dummy = f" __woof_decl{len(self._split_decls)}; auto "
                    if names[0] > start:
                        self.edits.replace(toks[start].extent.start.offset, toks[names[0]].extent.start.offset,
                                           dummy)
                    else:
                        self.edits.insert(toks[names[0]].extent.start.offset, dummy, rank=-20_001)
                continue
            self.edits.replace(toks[commas[j - 1]].extent.start.offset, toks[commas[j - 1]].extent.end.offset,
                               "; " + ("auto " if k == "POINTER" else base + " "))
            if k == "POINTER" and names[j] > seg:
                self.edits.replace(toks[seg].extent.start.offset, toks[names[j]].extent.start.offset, "")
        return True

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
        for at, (stmt, usrs, owner) in self._generic_local_sites().items():
            if owner != fn_usr:
                continue
            spaces = set()
            for u in usrs:
                spaces |= spaces_of.get(u, set())
            if len(spaces) > 1:
                self.refuse(stmt, f"one declaration holds data in {' and '.join(sorted(spaces))} memory")
            if spaces:
                plan.append((at, f", {self.SPACE_INDEX[next(iter(spaces))]}"))
        return plan

    def _generic_local_sites(self):
        """Local declarations of a space-generic struct type: where each
        statement's ``S<...>`` takes its space, keyed by offset."""
        if getattr(self, "_generic_sites", None) is None:
            self._generic_sites = {}
            if self.generic_structs:
                K = self.ci.CursorKind
                stmts = self._decl_stmts()
                for n in self.tu.cursor.walk_preorder():
                    if n.kind != K.VAR_DECL or not self.main_file(n):
                        continue
                    name = self.spaces.generic_name(n.type)
                    owner = n.semantic_parent
                    if name is None or owner is None or owner.kind.name not in (
                            "FUNCTION_DECL", "FUNCTION_TEMPLATE", "CXX_METHOD", "CONSTRUCTOR"):
                        continue
                    if owner.semantic_parent is not None and \
                            owner.semantic_parent.spelling.split("<", 1)[0] in self.generic_structs:
                        continue        # inside the struct: __WS (generic_struct)
                    stmt = stmts.get(n.get_usr())
                    if stmt is None:
                        continue
                    at = self.generic_type_close(list(stmt.get_tokens()), name)
                    if at is None:
                        self.refuse(n, f"cannot place the address space of {name!r} local {n.spelling!r}")
                    site = self._generic_sites.setdefault(at, (stmt, [], owner.get_usr()))
                    site[1].append(n.get_usr())
        return self._generic_sites

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
        for at, (stmt, usrs, owner) in self._generic_local_sites().items():
            if at in self._emitted_sites or owner in multi:
                continue
            spaces = set()
            for u in usrs:
                spaces |= self.spaces.var_spaces.get(u, set())
            if len(spaces) == 1:
                self.edits.insert(at, f", {self.SPACE_INDEX[next(iter(spaces))]}", rank=-20_000)
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
    _LOOPS = ("FOR_STMT", "WHILE_STMT", "DO_STMT", "CXX_FOR_RANGE_STMT", "SWITCH_STMT")

    def eliminate_gotos(self):
        """Gotos -> structured control flow (Metal has no goto).

        Each label's jumps are made structured (:meth:`_structured_gotos`).
        A label whose jumps that refuses -- crossing regions, a backward
        jump from inside a loop, a break or continue among the wrapped
        statements -- joins its block's dispatch group instead
        (:meth:`_dispatch_block`), with every label of the block whose
        region crosses the group's statements, and the structured pass runs
        again for the rest.  Breakage it prevents: deng's and kf_old's
        Fortran (``IF (..) GOTO 255`` / ``175 ...`` / ``GOTO 255`` /
        ``255 ...`` / ``GOTO 175``) was refused as overlapping.
        """
        saved = (list(self.edits.replacements), list(self.edits.inserts), list(self.edits.regions))
        #: block span -> [block, {label spans}]
        dispatch = {}
        while True:
            self._goto_block = self._goto_label = None
            try:
                self._structured_gotos(dispatch)
                return
            except TranslationError:
                block, label = self._goto_block, self._goto_label
                if block is None or label is None or self.span(label) in dispatch.get(self.span(block), (None, ()))[1]:
                    raise
            self.edits.replacements, self.edits.inserts, self.edits.regions = (
                list(saved[0]), list(saved[1]), list(saved[2]))
            dispatch.setdefault(self.span(block), [block, set()])[1].add(self.span(label))

    def _structured_gotos(self, dispatch=None):
        """Forward ``goto L;`` -> a structured exit.

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
        targeted = set()
        for g in gotos:
            node = g.referenced
            while node is not None and node.kind == K.LABEL_STMT:
                targeted.add((node.spelling, self.span(node)))      # and the labels it is stacked under
                node = parents.get(self.span(node) + (node.kind.name,))
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
        dispatch = dispatch or {}
        to_dispatch = {key: [] for key in dispatch}
        #: goto span -> the label statement of its block it reaches
        self._goto_target = {}
        stacked = set()
        for g in gotos:
            target = g.referenced
            if target is None:
                self.refuse(g, "goto whose label clang could not resolve")
            label = next((l for l in labels.get(target.spelling, []) if self.span(l) == self.span(target)), None)
            # stacked labels (`L400: L401: s;`, f2c's): a jump to the inner
            # one is a jump to the outer, which is the block's statement
            while label is not None and parent(label) is not None and parent(label).kind == K.LABEL_STMT:
                if self.span(label) not in stacked:
                    stacked.add(self.span(label))
                    toks = list(label.get_tokens())
                    if len(toks) >= 2 and toks[1].spelling == ":":
                        self.edits.replace(toks[0].extent.start.offset, toks[1].extent.end.offset, "")
                label = parent(label)
            target = label if label is not None else target
            self._goto_target[self.span(g)] = target
            home = None if label is None else parent(label)
            if home is not None and self.span(label) in dispatch.get(self.span(home), (None, ()))[1]:
                to_dispatch[self.span(home)].append(g)
                continue
            by_label.setdefault((target.spelling, self.span(target)), []).append(g)
        # a label some jump reaches from outside its block (a goto from the
        # then arm into the else arm: Noah-MP's dynamic-VIC label_1003) is
        # entered through a re-entry loop instead (_reenter_label)
        reenter = {}
        for key, jumps in list(by_label.items()):
            label = next((l for l in labels.get(key[0], []) if self.span(l) == key[1]), None)
            home = None if label is None else parent(label)
            if home is None or home.kind != K.COMPOUND_STMT:
                continue
            hs, he = self.span(home)
            if any(not (hs <= self.span(g)[0] and self.span(g)[1] <= he) for g in jumps):
                reenter[key] = (label, by_label.pop(key))
        # each dispatch group's statements: [start, end) offsets
        group_spans = {key: self._dispatch_range(block, to_dispatch[key], parent, chain)[2:]
                       for key, (block, _) in dispatch.items()}
        self._group_spans = group_spans
        starts, entries = self._nested_goto_starts(by_label, labels, parent, chain)
        regions, wrappers = [], []
        for (name, (ls, le)), jumps in by_label.items():
            label = next(l for l in labels.get(name, []) if self.span(l) == (ls, le))
            block = parent(label)
            if block is None or block.kind != K.COMPOUND_STMT:
                self.refuse(label, f"label {name!r} is not a statement of a block")
            self._goto_block, self._goto_label = block, label
            stmts = list(block.get_children())
            j = next(i for i, st in enumerate(stmts) if self.span(st) == (ls, le))
            if self._backward_goto(name, label, stmts, j, jumps, chain, parent, block, regions, wrappers):
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
                    if self.span(a) == self.span(block) and a.kind == block.kind:
                        break       # the jump is itself the block's statement (top is g)
                    if self.span(a) == self.span(top) and a.kind == top.kind:
                        if a.kind.name in self._LOOPS:
                            inner_loops.append(a)
                        break
                    if a.kind.name in self._LOOPS:
                        inner_loops.append(a)
            first = min(first, starts.get((name, (ls, le)), first))
            # a jump into a retry loop's body: the wrapper ends at the loop's
            # label, and the loop's statements before this label are skipped
            # on that first pass (_nested_goto_starts)
            entry = entries.get((name, (ls, le)))
            end = j if entry is None else entry
            region = stmts[first:end]
            rs, re_ = self.span(region[0])[0], (ls if entry is None else self.span(stmts[entry])[0])
            for (a, b) in regions:
                if a < re_ and rs < b and not (a <= rs and re_ <= b) and not (rs <= a and b <= re_):
                    self.refuse(label, f"goto regions overlap at label {name!r}")
            self._outside_dispatch(label, name, block, rs, re_, group_spans)
            regions.append((rs, re_))
            # retargeted break/continue, and declarations used after the label
            after = stmts[end:]
            for st in region + ([] if entry is None else stmts[entry:j]):
                if st not in region:
                    # skipped by the entry guard, a block: scoped to it
                    if st.kind == K.DECL_STMT:
                        for var in st.get_children():
                            usr = var.get_usr()
                            if any(n.kind == K.DECL_REF_EXPR and n.referenced is not None
                                   and n.referenced.get_usr() == usr
                                   for later in stmts[j:] for n in later.walk_preorder()):
                                self.refuse(var, f"{var.spelling!r} is declared in a loop before "
                                                 f"label {name!r} and used after it")
                    continue
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
            # Wrappers opening at one offset nest outer first: the longer one.
            self.edits.insert(rs, f"bool {flag} = false; do {{ ", rank=-50_000 - (re_ - rs))
            if entry is None:
                self.edits.insert(ls, f"}} while (false); ", rank=_GOTO_CLOSE)
            else:
                # closed before the loop opens; the guard opens inside it
                self.edits.insert(re_, f"}} while (false); ", rank=-100_000_000)
                self.edits.insert(re_, f"if (!{flag}) {{ ", rank=-49_999)
                self.edits.insert(ls, f"}} {flag} = false; ", rank=-100_000_000)
            wrappers.append({"kind": "forward", "rs": rs, "close": re_, "flag": flag, "name": name,
                             "jumps": [self.span(g)[0] for g in jumps], "node": label})
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
                # before the wrapper's close when the loop ends at the label
                self.edits.insert(end, f" if ({flag}) break;", rank=_GOTO_CLOSE - 1)
            wrappers[-1]["block"] = block
        self._goto_block = self._goto_label = None
        # innermost first: an outer dispatch loop's jumps cross the inner ones
        for n, key in enumerate(sorted(dispatch, key=lambda k: k[1] - k[0])):
            self._dispatch_block(dispatch[key][0], to_dispatch[key], n, parent, chain, wrappers)
        # innermost re-entry statement first: an outer one's jumps cross it
        ordered = sorted(reenter.values(), key=lambda v: self._reenter_extent(v[0], v[1], chain))
        for n, (label, jumps) in enumerate(ordered):
            self._reenter_label(label, jumps, n, parent, chain, wrappers)
        self._nested_wrapper_exits(wrappers)

    def _reenter_extent(self, label, jumps, chain):
        """Byte length of the statement a re-entry loop wraps (its order)."""
        stmt = self._reenter_statement(label, jumps, chain)
        if stmt is None:
            return 0
        s, e = self.span(stmt)
        return e - s

    def _reenter_statement(self, label, jumps, chain):
        """(the path from the label up, its index of S) for _reenter_label;
        S is the innermost statement of a block holding the label and every
        jump to it.  None when there is none."""
        K = self.ci.CursorKind
        path = chain(label)
        spans = [self.span(a) for a in path]
        common = None
        for g in jumps:
            for a in chain(g):
                if self.span(a) in spans:
                    i = spans.index(self.span(a))
                    common = i if common is None else max(common, i)
                    break
        i = common
        while i is not None and i + 1 < len(path) and (path[i].kind == K.COMPOUND_STMT
                                                       or path[i + 1].kind != K.COMPOUND_STMT):
            i += 1
        if i is None or i + 1 >= len(path) or path[i + 1].kind != K.COMPOUND_STMT:
            return None
        return path[i]

    def _reenter_label(self, label, jumps, n, parent, chain, wrappers):
        """A label entered from outside its block -> a re-entry loop around
        the innermost statement S (of a block) holding the label and every
        jump to it: ``while (true) { S break; }``.  A jump sets a flag and
        continues the loop (out of every loop and wrapper between it and S);
        with the flag set, each ``if`` on the way down to the label takes the
        arm that holds it and each statement before it on the way is
        skipped; the label clears the flag.  Refused by name: a loop or
        switch between S and the label (re-entering it would rerun it), a
        declaration among the skipped statements, a break or continue in S
        meant for a loop around it."""
        K = self.ci.CursorKind
        name = label.spelling
        path = chain(label)                 # label, its block, ..., the root
        spans = [self.span(a) for a in path]
        common = None
        for g in jumps:
            for a in chain(g):
                if self.span(a) in spans:
                    i = spans.index(self.span(a))
                    common = i if common is None else max(common, i)
                    break
        # S: the outermost ancestor up to the common one that is a block's statement
        i = common
        while i is not None and i + 1 < len(path) and (path[i].kind == K.COMPOUND_STMT
                                                       or path[i + 1].kind != K.COMPOUND_STMT):
            i += 1
        if i is None or i + 1 >= len(path) or path[i + 1].kind != K.COMPOUND_STMT:
            self.refuse(label, f"goto {name!r} from outside the label's block")
        stmt = path[i]
        flag = f"__woof_in{n}"
        for a in path[1:i + 1]:
            if a.kind.name in self._LOOPS:
                self.refuse(label, f"goto {name!r} into a loop or switch")
        for j in range(1, i + 1):
            a, below = path[j], path[j - 1]
            if a.kind == K.COMPOUND_STMT:
                skipped = []
                for st in a.get_children():
                    if self.span(st) == self.span(below):
                        break
                    skipped.append(st)
                if skipped:
                    if any(st.kind == K.DECL_STMT for st in skipped):
                        self.refuse(label, f"a declaration before label {name!r} in a block the goto enters")
                    self.edits.insert(self.span(skipped[0])[0], f"if (!{flag}) {{ ", rank=-60_000)
                    # after any `if (flag) continue/break;` a crossed loop or
                    # wrapper ending here carries (ranks _GOTO_CLOSE - 1, + 2)
                    self.edits.insert(self._statement_end(skipped[-1]), " } ", rank=_GOTO_CLOSE + 3)
            elif a.kind == K.IF_STMT:
                kids = list(a.get_children())
                if len(kids) < 2:
                    self.refuse(a, f"cannot read the if statement around label {name!r}")
                cond, arms = kids[0], kids[1:]
                into_then = self.span(arms[0]) == self.span(below)
                cs, ce = self.span(cond)
                self.edits.insert(cs, f"({flag} || (" if into_then else f"(!{flag} && (", rank=-60_000)
                self.edits.insert(ce, "))", rank=60_000)
            elif a.kind != K.LABEL_STMT:
                self.refuse(a, f"goto {name!r} enters a {a.kind.name.lower()}")
        for j in self._unenclosed_jumps(stmt):
            self.refuse(j, f"a {j.kind.name.split('_')[0].lower()} in the statement a goto re-enters "
                           f"for label {name!r}")
        rs, re_ = self.span(stmt)[0], self._statement_end(stmt)
        inner = [w for w in wrappers if rs <= w["rs"] and w["close"] <= re_]
        for g in jumps:
            at = self.span(g)[0]
            crossed = []
            for a in chain(g)[1:]:
                if self.span(a) == self.span(stmt) and a.kind == stmt.kind:
                    if a.kind.name in self._LOOPS:
                        crossed.append((self.span(a)[0], self._statement_end(a), _GOTO_CLOSE - 1))
                    break
                if a.kind.name in self._LOOPS:
                    crossed.append((self.span(a)[0], self._statement_end(a), _GOTO_CLOSE - 1))
            crossed += [(w["rs"], w["close"], _GOTO_CLOSE + 2) for w in inner if w["rs"] <= at < w["close"]]
            crossed.sort()
            for k, (_, end, rank) in enumerate(crossed):
                self.edits.insert(end, f" if ({flag}) {'continue' if k == 0 else 'break'};", rank=rank)
            gs, ge = self.span(g)
            self.edits.replace(gs, ge, f"{{ {flag} = true; {'break' if crossed else 'continue'}; }}")
        toks = list(label.get_tokens())
        if len(toks) < 2 or toks[1].spelling != ":":
            self.refuse(label, f"cannot find label {name!r}")
        self.edits.replace(toks[0].extent.start.offset, toks[1].extent.end.offset, f"{flag} = false;")
        self.edits.insert(rs, f"{{ bool {flag} = false; while (true) {{ ", rank=-50_000 - (re_ - rs))
        self.edits.insert(re_, " break; } }", rank=_GOTO_CLOSE - 1)
        wrappers.append({"kind": "reenter", "rs": rs, "close": re_, "flag": None, "name": name,
                         "jumps": [], "node": label, "block": path[i + 1]})

    def _outside_dispatch(self, label, name, block, rs, re_, group_spans):
        """A structured region of a block with a dispatch group must hold
        the group's statements whole or lie apart from them: a case inside
        a wrapper, or a wrapper's flag declared among the cases (a case
        would jump over it), cannot be built.  Refused with the label
        named, so it joins the group."""
        span = group_spans.get(self.span(block))
        if span is None:
            return
        a, b = span
        if a < re_ and rs < b and not (rs <= a and b <= re_):
            self.refuse(label, f"goto region of label {name!r} crosses a dispatched goto block")

    def _dispatch_range(self, block, jumps, parent, chain):
        """A dispatch group's (first, last) statement indices of ``block``
        and their [start, end) offsets: from the first jump or label of the
        group to the last."""
        K = self.ci.CursorKind
        stmts = list(block.get_children())
        index = {self.span(st): i for i, st in enumerate(stmts)}
        marks = []
        for g in jumps:
            j = index.get(self.span(self._goto_target[self.span(g)]))
            top = next((a for a in chain(g) if parent(a) is not None and self.span(parent(a)) == self.span(block)
                        and parent(a).kind == K.COMPOUND_STMT), None)
            if j is None or top is None:
                self.refuse(g, f"goto {self._goto_target[self.span(g)].spelling!r} from outside the label's block")
            marks += [j, index[self.span(top)]]
        first, last = min(marks), max(marks)
        return first, last, self.span(stmts[first])[0], self._statement_end(stmts[last])

    def _dispatch_block(self, block, jumps, n, parent, chain, wrappers):
        """The labels of one block, whatever the shape of their jumps ->
        ``while (true) { switch (st) { case 0: ... case k: ... } break; }``.

        The statements from the first jump or label to the last become the
        switch body; each label is a ``case``; a jump sets ``st`` and
        continues the loop, or, from inside a loop, switch or wrapper of its
        own, sets a flag and breaks out of each (``if (jmp) break;`` after
        every one).  A break or continue among them meant for a loop around
        the block leaves the dispatch loop first.  Refused by name: a
        declaration among the dispatched statements (a case would jump over
        it), a jump from outside the block's statements.
        """
        K = self.ci.CursorKind
        stmts = list(block.get_children())
        first, last, rs, re_ = self._dispatch_range(block, jumps, parent, chain)
        cases, tops = {}, []
        for g in jumps:
            target = self._goto_target[self.span(g)]
            cases.setdefault(self.span(target), (None, target))
            tops.append((g, next(a for a in chain(g) if parent(a) is not None
                                 and self.span(parent(a)) == self.span(block) and parent(a).kind == K.COMPOUND_STMT)))
        region = stmts[first:last + 1]
        for st in region:
            if st.kind == K.DECL_STMT:
                self.refuse(st, "a declaration among the statements of a dispatched goto block")
        st_, jmp, out = f"__woof_st{n}", f"__woof_jmp{n}", f"__woof_out{n}"
        number = {}
        for k, (span, (j, label)) in enumerate(sorted(cases.items()), start=1):
            number[span] = k
            toks = list(label.get_tokens())
            if len(toks) >= 2 and toks[1].spelling == ":":
                self.edits.replace(toks[0].extent.start.offset, toks[1].extent.end.offset, f"case {k}:")
            else:
                self.refuse(label, f"cannot find label {label.spelling!r}")
        inner = [w for w in wrappers if rs <= w["rs"] and w["close"] <= re_]

        def leave(node, action, exit=False):
            """``node``'s jump out of every loop, switch and wrapper between
            it and the dispatch loop; ``action`` is what it sets first."""
            at = self.span(node)[0]
            crossed = []
            for a in chain(node)[1:]:
                if self.span(a) == self.span(block) and a.kind == block.kind:
                    break
                if a.kind.name in self._LOOPS:
                    crossed.append((self.span(a)[0], self._statement_end(a), _GOTO_CLOSE - 1))
            crossed += [(w["rs"], w["close"], _GOTO_CLOSE + 2) for w in inner if w["rs"] <= at < w["close"]]
            if not crossed:
                return f"{{ {action} break; }}" if exit else f"{{ {action} continue; }}"
            crossed.sort()
            for k, (_, end, rank) in enumerate(crossed):
                if k == 0:
                    self.edits.insert(end, f" if ({jmp}) {{ if ({out}) break; {jmp} = false; continue; }}",
                                      rank=rank)
                else:
                    self.edits.insert(end, f" if ({jmp}) break;", rank=rank)
            return f"{{ {action} {jmp} = true; break; }}"

        for g, top in tops:
            gs, ge = self.span(g)
            self.edits.replace(gs, ge, leave(g, f"{st_} = {number[self.span(self._goto_target[self.span(g)])]};"))
        exits = set()
        for st in region:
            for j in self._unenclosed_jumps(st):
                if any(w["rs"] <= self.span(j)[0] < w["close"] for w in inner):
                    self.refuse(j, f"a {j.kind.name.split('_')[0].lower()} leaves a goto wrapper inside "
                                   "a dispatched goto block")
                code = 1 if j.kind == K.BREAK_STMT else 2
                exits.add(code)
                js, je = self.span(j)
                self.edits.replace(js, je, leave(j, f"{out} = {code};", exit=True))
        self.edits.insert(rs, f"{{ int {st_} = 0; bool {jmp} = false; int {out} = 0; (void){jmp}; (void){out}; "
                              f"while (true) {{ switch ({st_}) {{ case 0: ", rank=-50_000 - (re_ - rs))
        post = "".join(f" if ({out} == {c}) {'break' if c == 1 else 'continue'};" for c in sorted(exits))
        self.edits.insert(re_, f" }} break; }}{post} }}", rank=_GOTO_CLOSE - 1)
        wrappers.append({"kind": "dispatch", "rs": rs, "close": re_, "flag": None, "name": f"dispatch{n}",
                         "jumps": [], "node": block, "block": parent(block)})

    def _nested_goto_starts(self, by_label, labels, parent, chain):
        """Where each forward region starts so that the regions of one block
        nest.  Fortran's spaghetti (``IF (..) GOTO 270`` ... ``GOTO 320``
        ``270 ...``, or a ``GOTO 1002`` out of a ``1001 ... GOTO 1001`` retry
        loop) gives regions that cross: one starts inside another and ends
        after it.  Starting the later region at the earlier one's start makes
        it the outer wrapper; its jumps out of the inner wrapper leave both
        (:meth:`_nested_wrapper_exits`).  A backward region starts at its
        label and is never moved.  Breakage it prevents: MM5 sfclay's
        regime branches and Noah's frh2o iteration were refused as
        overlapping."""
        K = self.ci.CursorKind
        spans = {}
        for (name, (ls, le)), jumps in by_label.items():
            label = next((l for l in labels.get(name, []) if self.span(l) == (ls, le)), None)
            block = None if label is None else parent(label)
            if block is None or block.kind != K.COMPOUND_STMT:
                continue
            stmts = list(block.get_children())
            j = next(i for i, st in enumerate(stmts) if self.span(st) == (ls, le))
            tops = []
            for g in jumps:
                top = next((a for a in chain(g) if parent(a) is not None and self.span(parent(a)) == self.span(block)
                            and parent(a).kind == K.COMPOUND_STMT), None)
                if top is None:
                    break
                tops.append(next(k for k, st in enumerate(stmts) if self.span(st) == self.span(top)))
            else:
                if tops and all(i < j for i in tops):
                    spans[(name, (ls, le))] = [self.span(block), "forward", min(tops), j, max(tops)]
                elif tops and all(i > j for i in tops):
                    spans[(name, (ls, le))] = [self.span(block), "backward", j, max(tops) + 1]
        changed = True
        while changed:
            changed = False
            for f in spans.values():
                if f[1] != "forward":
                    continue
                for r in spans.values():
                    if r is not f and r[0] == f[0] and r[2] < f[2] < r[3] <= f[3]:
                        f[2] = r[2]
                        changed = True
        # A forward jump into a retry loop from before it (RUC's ``IF (..)
        # GOTO 100`` / ``90 ...`` / ``100 ...`` / ``IF (..) GOTO 90``): the
        # loop's statements before the label are guarded by the jump's flag,
        # which the label clears.  Only when every jump to the label comes
        # before the loop (a jump from the guarded statements would break out
        # of the retry loop instead), and no other region of the block starts
        # or ends among the guarded statements.
        entries = {}
        for key, f in spans.items():
            if f[1] != "forward":
                continue
            for r in spans.values():
                if not (r[1] == "backward" and r[0] == f[0] and f[2] < r[2] < f[3] < r[3]
                        and f[4] < r[2]):
                    continue
                clear = all(o is f or o is r or o[0] != f[0] or o[3] <= r[2] or o[2] >= f[3]
                            or (o[2] <= r[2] and o[3] >= r[3]) for o in spans.values())
                if clear:
                    entries[key] = r[2]
        return {key: s[2] for key, s in spans.items() if s[1] == "forward"}, entries

    def _nested_wrapper_exits(self, wrappers):
        """A jump that crosses another region's wrapper on its way out: the
        wrapper (a do/while(false) or retry loop the elimination made, not in
        the AST) is left by the jump's break, and the enclosing region must be
        left too -- ``if (flag) break;`` right after the inner wrapper closes.
        Breakage it prevents: lin_ggamma's ``goto __label_000020`` out of a
        loop also nested in L_28's wrapper fell into ``return 0.0f``, so
        Γ(b+4) was 0, Lin's rain never fell, and its radar was 84 dBZ off."""
        done = set()
        for outer in wrappers:
            for at in outer["jumps"]:
                for inner in wrappers:
                    if inner is outer or not (inner["rs"] <= at < inner["close"]):
                        continue
                    if not (outer["rs"] <= inner["rs"] and inner["close"] <= outer["close"]):
                        continue
                    if outer["kind"] == "backward":
                        self._goto_block, self._goto_label = outer.get("block"), outer["node"]
                        self.refuse(outer["node"], f"backward goto {outer['name']!r} from inside the "
                                                   f"structured region of label {inner['name']!r}")
                    key = (id(inner), id(outer))
                    if key not in done:
                        done.add(key)
                        self.edits.insert(inner["close"], f" if ({outer['flag']}) break; ", rank=_GOTO_CLOSE + 1)

    def _backward_goto(self, name, label, stmts, j, jumps, chain, parent, block, regions, wrappers):
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
                # the block's own statement too: a jump from a loop that is
                # one would `continue` that loop (ntu's retry inside its DO)
                if a.kind.name in self._LOOPS:
                    self.refuse(g, f"backward goto {name!r} from inside a loop or switch")
                if self.span(a) == self.span(top) and a.kind == top.kind:
                    break
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
        self._outside_dispatch(label, name, block, rs, re_, getattr(self, "_group_spans", {}))
        regions.append((rs, re_))
        self.edits.insert(rs, "while (true) { ", rank=-50_000 - (re_ - rs))
        self.edits.insert(re_, " break; }", rank=_GOTO_CLOSE)
        wrappers.append({"kind": "backward", "rs": rs, "close": re_, "flag": None, "name": name,
                         "jumps": [self.span(g)[0] for g, _, _ in tops], "node": label, "block": block})
        label_tokens = list(label.get_tokens())
        if len(label_tokens) >= 2 and label_tokens[1].spelling == ":":
            self.edits.replace(label_tokens[0].extent.start.offset, label_tokens[1].extent.end.offset, "")
        for g, _, _ in tops:
            gs, ge = self.span(g)
            self.edits.replace(gs, ge, "continue")
        return True

    def _unenclosed_jumps(self, node):
        """break/continue in ``node`` that leave it: not enclosed by a loop
        inside it, and for a break, not by a switch either (a switch, even
        ``node`` itself, captures break but not continue)."""
        K = self.ci.CursorKind
        out = []
        stack = [(node, False)]             # (cursor, a switch captures its breaks)
        while stack:
            n, switched = stack.pop()
            if n.kind == K.BREAK_STMT:
                if not switched:
                    out.append(n)
                continue
            if n.kind == K.CONTINUE_STMT:
                out.append(n)
                continue
            if n.kind.name in ("FOR_STMT", "WHILE_STMT", "DO_STMT", "CXX_FOR_RANGE_STMT"):
                continue                    # its own break and continue
            if n.kind == K.SWITCH_STMT:
                switched = True
            stack.extend((c, switched) for c in n.get_children())
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
                    self.edits.replace(rs, re_, g["param"] if g["dims"] else
                                       f"(__woof_ptr_cell<{g['cell']}>({g['param']}))" if g.get("cell")
                                       else f"(*{g['param']})")
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
            dims = re.findall(r"\[([^\]]+)\]", m.group(2))
            base = t
            for _ in dims:                      # the element type, spelled as every other type is
                nxt = base.get_array_element_type()
                if nxt.kind.name == "INVALID":
                    break
                base = nxt
            elem = self._spell_type(base, drop_const=False) if base is not t else m.group(1)
            return elem, dims
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

    def _address_table_casts(self):
        """``(const T* const*)address``: a table of device addresses reached by
        an integer address.  Metal has no pointer-to-pointer it can spell
        here, so the cast builds the table wrapper over the address's words:
        ``__woof_addr_table<const device T*>(reinterpret_cast<const device ulong*>(address))``."""
        from woof.backend.translate.spaces import _CASTS, _INTEGER_KINDS, _is_table
        K = self.ci.CursorKind
        done = set()
        for node in self.tu.cursor.walk_preorder():
            if node.kind.name not in _CASTS or not self.main_file(node) or not _is_table(node.type):
                continue
            operand = [c for c in node.get_children() if c.kind not in (K.TYPE_REF, K.TEMPLATE_REF)]
            if not operand or operand[-1].type.get_canonical().kind.name not in _INTEGER_KINDS:
                continue
            key = (node.extent.start.offset, node.extent.end.offset)
            if key in done:
                continue
            done.add(key)
            target = node.type.get_canonical().get_pointee().get_pointee()
            tspace = "const device" if target.is_const_qualified() else "device"
            elem = self._spell_type(target, drop_const=True)
            op_start = operand[-1].extent.start.offset
            self.edits.replace(key[0], op_start,
                               f"__woof_addr_table<{tspace} {elem}*>(reinterpret_cast<const device ulong*>(")
            self.edits.insert(key[1], "))", rank=60_000)
        return done

    def cast_plan(self, instance):
        """Insertions naming each pointer cast's space in one function
        instance (an integer address is device memory; null takes device)."""
        plan = []
        for (start, end), spaces in self.spaces.cast_spaces.get(instance, {}).items():
            if (start, end) in getattr(self, "_table_casts", ()):
                continue
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
        self._table_casts = self._address_table_casts()
        multi = {usr for usr, inst in self.spaces.instances.items() if len(inst) > 1}
        reached = set(self._table_casts)
        for instance, casts in self.spaces.cast_spaces.items():
            reached.update(casts)
            if instance is None or instance[0] in multi:
                continue
            for at, text in self.cast_plan(instance):
                self.edits.insert(at, text, rank=-25_000)
        K = self.ci.CursorKind
        from woof.backend.translate.spaces import _CASTS, _INTEGER_KINDS, _indirection
        generic = [self.span(c) for c in self.tu.cursor.walk_preorder()
                   if c.kind == K.CLASS_TEMPLATE and c.spelling in getattr(self, "generic_structs", ())
                   and self.main_file(c)]
        for node in self.tu.cursor.walk_preorder():
            if node.kind.name not in _CASTS or not self.main_file(node) or _indirection(node.type) != "ptr":
                continue
            key = (node.extent.start.offset, node.extent.end.offset)
            if key in reached:
                continue
            if node.kind.name in ("CXX_STATIC_CAST_EXPR", "CXX_REINTERPRET_CAST_EXPR", "CXX_CONST_CAST_EXPR") \
                    and any(gs <= key[0] and key[1] <= ge for gs, ge in generic):
                # inside a space-generic struct (gsfc_ptr's
                # `const_cast<T *>(value)`): the struct's own space
                toks = list(node.get_tokens())
                if len(toks) > 2 and toks[1].spelling == "<":
                    close = self._closing_angle(toks, 1)
                    star = next((t for t in reversed(toks[2:close]) if t.spelling == "*"), None) \
                        if close is not None else None
                    if star is not None:
                        self.edits.insert(toks[2].extent.start.offset, "__woof_sptr<", rank=-25_000)
                        self.edits.replace(star.extent.start.offset, star.extent.end.offset, ", __WS>")
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
        for usr, params in self.spaces.variant.items():
            info = self.spaces.functions[usr]
            body = next((c for c in info.cursor.get_children()
                         if c.kind == self.ci.CursorKind.COMPOUND_STMT), None)
            if body is None:
                continue
            prologue = []
            for parm in params:
                name = parm.spelling
                elem = self._spell_type(parm.type.get_pointee(), drop_const=True)
                at = parm.location.offset
                self.edits.replace(at, at + len(name), f"__woof_vp_{name}")
                prologue.append(f"__woof_vptr<{elem}> {name}(__woof_vp_{name});")
            self.edits.insert(body.extent.start.offset + 1, "\n    " + "\n    ".join(prologue), rank=10_001)
        for usr, info in self.spaces.functions.items():
            this_spaces = sorted(self.spaces.this_spaces.get(usr, ()))
            if not info.slots and not info.returns and not this_spaces:
                continue
            instances = sorted(self.spaces.instances.get(usr) or ())
            if not instances:
                instances = [tuple(DEVICE for _ in info.slots)]
            seen_decls = set()
            for cursor in [info.cursor] + info.decls:
                if self.span(cursor) in seen_decls:
                    continue        # one prototype reached twice (collect sees it per scope)
                seen_decls.add(self.span(cursor))
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
        for (index, _, ind), space in zip(info.slots, spaces):
            if index >= len(params):
                continue
            if ind == "gstruct":
                name = self.spaces.generic_name(params[index].type)
                at = self.generic_type_close(list(params[index].get_tokens()), name)
                if at is None:
                    self.refuse(params[index], f"cannot place the address space of {name!r} parameter "
                                               f"{params[index].spelling!r} (spell it {name}<...>)")
                plan.append((at, f", {self.SPACE_INDEX[space]}"))
                continue
            plan.append((params[index].extent.start.offset, f"{space} "))
        multi = len(self.spaces.instances.get(info.cursor.get_usr(), ())) > 1
        if cursor is info.cursor and multi:
            plan += self.cast_plan((info.cursor.get_usr(), spaces))
            plan += self.decl_plan((info.cursor.get_usr(), spaces))
        if multi:
            # A call that omits a defaulted pointer is the device instance's
            # (spaces.call); other instances drop the default, or such a call
            # would match them all.
            # (C++ defaults are trailing: such an instance drops all of them)
            if any(space != "device" and index < len(params)
                   and any(t.spelling == "=" for t in params[index].get_tokens())
                   for (index, _, _), space in zip(info.slots, spaces)):
                for parm in params:
                    eq = next((t for t in parm.get_tokens() if t.spelling == "="), None)
                    if eq is not None:
                        plan.append((eq.extent.start.offset, "/* "))
                        plan.append((parm.extent.end.offset, " */"))
        if info.returns:
            ret = self.spaces.returns.get((info.cursor.get_usr(), spaces)) or "device"
            start = self._return_type_start(cursor)
            if start is not None:
                plan.append((start, f"{ret} "))
        return plan

    def table_field_stores(self, top):
        """``s->table[k] = p`` into a ``T**`` struct field (cam3's
        ``state->abs_args[0] = pbr``, filling the per-column descriptor
        row the kernel reads back): the field is a ``__woof_addr_table``
        (struct_fields), whose ``[k]`` is a value, so the store is spelled
        ``s->table.__woof_set(k, p)``, which writes the entry's word."""
        K = self.ci.CursorKind
        from woof.backend.translate.spaces import _is_table

        def bare(e):
            while e.kind in (K.UNEXPOSED_EXPR, K.PAREN_EXPR) and len(list(e.get_children())) == 1:
                e = next(e.get_children())
            return e
        for c in top:
            for node in c.walk_preorder():
                if node.kind != K.BINARY_OPERATOR or not self.main_file(node):
                    continue
                kids = list(node.get_children())
                if len(kids) != 2:
                    continue
                lhs = bare(kids[0])
                if lhs.kind != K.ARRAY_SUBSCRIPT_EXPR:
                    continue
                parts = list(lhs.get_children())
                if len(parts) != 2:
                    continue
                base = bare(parts[0])
                field = base.referenced if base.kind == K.MEMBER_REF_EXPR else None
                if field is None or field.kind != K.FIELD_DECL or not _is_table(field.type):
                    continue
                op = next((t for t in node.get_tokens()
                           if t.extent.start.offset >= lhs.extent.end.offset and t.spelling in ("=",)), None)
                if op is None or op.extent.start.offset >= kids[1].extent.start.offset:
                    continue
                be, (is_, ie) = base.extent.end.offset, self.span(parts[1])
                self.edits.replace(be, is_, ".__woof_set(")
                self.edits.replace(ie, op.extent.end.offset, ", ")
                self.edits.insert(kids[1].extent.end.offset, ")", rank=50_000)

    def table_parameters(self):
        """A device function's ``T**`` / ``T* const*`` parameter receives a
        kernel's address table (rewrite_kernel): it is declared as the same
        ``__woof_addr_table<device T*>``, whose ``F[k]`` is the device pointer.

        A parameter every call gives a local array of device pointers
        (nwp_windows' ``float* out[12]``) is a pointer into thread memory
        instead: ``device T* const thread*``.  One given both is refused."""
        from woof.backend.translate.spaces import _is_table, _local_pointer_array
        K = self.ci.CursorKind
        kinds = (K.FUNCTION_DECL, K.FUNCTION_TEMPLATE, K.CXX_METHOD)
        seen = set()
        #: (function usr, parameter index) -> {"thread", "device"}: where the
        #: tables its calls pass live
        storage = {}
        forwards = []                       # (caller parameter, callee parameter)
        for call in self.tu.cursor.walk_preorder():
            if call.kind != K.CALL_EXPR or not self.main_file(call):
                continue
            fn = call.referenced
            if fn is None or fn.kind not in kinds:
                continue
            for i, arg in enumerate(call.get_arguments()):
                if not _is_table(arg.type):
                    continue
                if _local_pointer_array(arg) is not None:
                    storage.setdefault((fn.get_usr(), i), set()).add("thread")
                    continue
                # a table parameter passed on: its own storage (below)
                ref = arg
                while ref.kind.name in ("UNEXPOSED_EXPR", "PAREN_EXPR") and len(list(ref.get_children())) == 1:
                    ref = list(ref.get_children())[0]
                d = ref.referenced if ref.kind.name == "DECL_REF_EXPR" else None
                owner = d.semantic_parent if d is not None and d.kind == K.PARM_DECL else None
                if owner is not None and owner.kind in kinds and not self.is_kernel(owner):
                    index = [p.get_usr() for p in owner.get_children() if p.kind == K.PARM_DECL].index(d.get_usr())
                    forwards.append(((owner.get_usr(), index), (fn.get_usr(), i)))
                    storage.setdefault((fn.get_usr(), i), set())
                else:
                    storage.setdefault((fn.get_usr(), i), set()).add("device")
        changed = True
        while changed:
            changed = False
            for src, dst in forwards:
                got = storage.get(src, set())
                if not got <= storage[dst]:
                    storage[dst] |= got
                    changed = True
        for key, where in storage.items():
            if not where:
                where.add("device")         # only ever passed on, from a parameter nothing resolves
        for node in self.tu.cursor.walk_preorder():
            if node.kind not in kinds or not self.main_file(node) or self.is_kernel(node):
                continue
            parms = [p for p in node.get_children() if p.kind == K.PARM_DECL]
            for index, parm in enumerate(parms):
                if not _is_table(parm.type):
                    continue
                key = self.span(parm)
                if key in seen:
                    continue
                seen.add(key)
                target = parm.type.get_canonical().get_pointee().get_pointee()
                tspace = "const device" if target.is_const_qualified() else "device"
                if "type-parameter-" in target.spelling:
                    # a template's `const T* const*`: the canonical type has
                    # lost the parameter's name, which the written type keeps
                    target = parm.type.get_pointee().get_pointee()
                elem = self._spell_type(target, drop_const=True)
                toks = list(parm.get_tokens())
                name = next((t for t in toks if t.spelling == parm.spelling), None)
                if name is None or not toks:
                    self.refuse(parm, f"cannot find the type of table parameter {parm.spelling!r}")
                where = storage.get((node.get_usr(), index), set())
                if where == {"thread"}:
                    inner = " const" if parm.type.get_canonical().get_pointee().is_const_qualified() else ""
                    self.edits.replace(toks[0].extent.start.offset, name.extent.start.offset,
                                       f"{tspace} {elem}*{inner} thread* ")
                    continue
                if len(where) > 1:
                    self.refuse(parm, f"table parameter {parm.spelling!r} is given both a local array "
                                      f"of pointers and a device address table")
                self.edits.replace(toks[0].extent.start.offset, name.extent.start.offset,
                                   f"__woof_addr_table<{tspace} {elem}*> ")

    #: Body size (tokens) at or under which WOOF_INLINE_POLICY=size inlines.
    INLINE_TOKENS = 300

    #: Units whose fully inlined kernel Metal's compiler cannot build: its
    #: service crashes (XPC_ERROR_CONNECTION_INTERRUPTED) on goddard's
    #: radiation column, which compiles in a second with the large
    #: functions out of line.  Its device oracle holds it to WRF's words.
    #: Fully inlined, lake, mskf and mskf_aerosol_replay did not compile in
    #: the census's 30 minutes and flg_streams crashed the compiler service;
    #: out of line they build in 2-5 minutes, and their Metal tests (lake's
    #: column history and f64cr grades, mskf_kernel, the flg replays) give
    #: the words they give on main.  NOT nssl2: out of line its rain
    #: sedimentation wrote zeros on Metal (the out-of-line hazard above),
    #: and cammgmp and rcon have no Metal test to hold them to.
    INLINE_POLICY_BY_UNIT = {"goddard": "size", "lake": "size", "mskf": "size",
                             "mskf_aerosol_replay": "size", "flg_streams": "size"}

    #: The same, for units known by a kernel they define (a RawModule's unit
    #: name is a hash of its text).  TEMPO's column kernel did not finish
    #: compiling fully inlined in 35 minutes; its column oracle holds the
    #: out-of-line build to CUDA's words.
    INLINE_POLICY_BY_KERNEL = {"tempo_columns": "size"}

    #: Units (known by a kernel they define) whose source ``__noinline__``
    #: functions are inlined too.  FAST_SBM's tile kernel, with its 23
    #: ``__noinline__`` helpers out of line, gave wrong words in 3 of 49
    #: columns of its kc2 replay (th off by 4e-3 K, number concentrations
    #: by orders of magnitude); fully inlined it is CUDA's --fmad=false
    #: words but for 1-ulp deltas in 8 liquid-bin words.  Not the default:
    #: the other units with ``__noinline__`` (noah, sfclay, mynn_surface,
    #: kf, urban_ucm480, the MPAS ports) give the same results either way,
    #: and fully inlined sfclay took 46 minutes to compile cold.
    #: WOOF_INLINE_NOINLINE=1|0 forces either way, for diagnosis.
    INLINE_NOINLINE_BY_KERNEL = frozenset({"fsbm_tile_f64", "fsbm_tile_f32"})

    #: The same, only for a tier: (kernel, macro, smallest value) -> policy.
    #: Thompson's column kernel crashes Metal's compiler service fully
    #: inlined (XPC_ERROR_CONNECTION_INTERRUPTED after ~3 min) at
    #: THOMPSON_COLUMN_KMAX 128 and 256, and at the default 64 as well, so
    #: the tier gate covers every tier.  Out of line (size or none: the same
    #: words) the three tiers agree word for word on the same columns.
    INLINE_POLICY_BY_KERNEL_TIER = {("thompson_column_step", "THOMPSON_COLUMN_KMAX", 64): "size"}

    def _unit_inline_policy(self) -> str:
        policy = self.INLINE_POLICY_BY_UNIT.get(self.unit.split("[", 1)[0], "")
        if policy:
            return policy
        tiers = {k: (m, v, p) for (k, m, v), p in self.INLINE_POLICY_BY_KERNEL_TIER.items()}
        for node in self.tu.cursor.walk_preorder():
            if node.kind != self.ci.CursorKind.FUNCTION_DECL:
                continue
            if node.spelling in self.INLINE_POLICY_BY_KERNEL and self.main_file(node) \
                    and self.is_kernel(node):
                return self.INLINE_POLICY_BY_KERNEL[node.spelling]
            if node.spelling in tiers and self.main_file(node) and self.is_kernel(node):
                macro, least, tier_policy = tiers[node.spelling]
                if self.macros.get(macro, 0) >= least:
                    return tier_policy
        return ""

    def inline_pointer_functions(self):
        """Every device function of the unit is ``always_inline`` (a source
        ``__noinline__`` stands): CUDA's own default, and the one policy under
        which Metal has given CUDA's words everywhere it was tried.

        Left to its heuristics, Metal's compiler gave wrong values (Noah-MP's
        water kernel: varying with the threadgroup size, vanishing under
        shader validation).  Explicit out-of-line calls were no cure: with
        small functions inlined into out-of-line ones, Morrison's mo_process
        gave different words on repeated launches of the same inputs; with
        every function out of line it was right alone and wrong after another
        test file ran in the process.  Fully inlined, both are deterministic
        and CUDA's word for word.  The price is compile time on the largest
        units (CLM lake ~20 min, NSSL's fused unit ~13 min cold).
        WOOF_INLINE_POLICY=size|none and WOOF_KEEP_CALL=f,g select the
        alternatives, for diagnosis only."""
        K = self.ci.CursorKind
        kinds = (K.FUNCTION_DECL, K.FUNCTION_TEMPLATE, K.CXX_METHOD, K.CONSTRUCTOR, K.DESTRUCTOR)
        policy = os.environ.get("WOOF_INLINE_POLICY", "") or self._unit_inline_policy()
        keep_call = set(os.environ.get("WOOF_KEEP_CALL", "").split(","))
        forced = os.environ.get("WOOF_INLINE_NOINLINE", "")
        if forced in ("0", "1"):
            honour_noinline = forced == "0"
        else:
            honour_noinline = not any(
                n.kind == K.FUNCTION_DECL and n.spelling in self.INLINE_NOINLINE_BY_KERNEL
                and self.main_file(n) and self.is_kernel(n)
                for n in self.tu.cursor.walk_preorder())
        done = set()
        for node in self.tu.cursor.walk_preorder():
            if node.kind not in kinds or not node.is_definition() or not self.main_file(node):
                continue
            if self.is_kernel(node):
                continue
            key = self.span(node)
            if key in done:
                continue
            done.add(key)
            tokens = list(node.get_tokens())
            noinline = next((t for t in tokens[:40] if t.spelling == "noinline"), None)
            if noinline is not None:
                if not honour_noinline and policy not in ("size", "none") \
                        and not self._inside_replacement(noinline.extent.start.offset):
                    self.edits.replace(noinline.extent.start.offset, noinline.extent.end.offset,
                                       "always_inline")
                continue
            at = self._attribute_site(node, tokens)
            if at is None:
                continue
            body = next((c for c in node.get_children() if c.kind == K.COMPOUND_STMT), None)
            size = len(list(body.get_tokens())) if body is not None else 0
            if policy == "none" or node.spelling in keep_call:
                inline = False
            elif policy == "size":
                inline = size <= self.INLINE_TOKENS
            else:
                inline = True
            self.edits.insert(at, "__attribute__((always_inline)) " if inline
                              else "__attribute__((noinline)) ", rank=-70_000)

    def _attribute_site(self, node, tokens):
        """Where a function attribute goes: before the return type, or
        before a constructor's / destructor's name."""
        if node.kind.name in ("CONSTRUCTOR", "DESTRUCTOR"):
            base = node.spelling.lstrip("~").split("<", 1)[0]
            for i, t in enumerate(tokens):
                if t.spelling == base:
                    j = i - 1 if node.kind.name == "DESTRUCTOR" and i and tokens[i - 1].spelling == "~" else i
                    return tokens[j].extent.start.offset
            return None
        return self._return_type_start(node)

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

    def _static_member_definition(self, owner, member):
        """``constant constexpr T C::m;`` after the class: MSL is C++14, where
        a static constexpr member that is odr-used (bound to a const&
        parameter of a function kept out of line) needs a definition."""
        toks = [t.spelling for t in member.get_tokens()]
        if "constexpr" not in toks or owner.kind.name not in ("STRUCT_DECL", "CLASS_DECL"):
            return                      # (class templates: their members stay folded)
        if any(c.kind.name.startswith("TEMPLATE") for c in owner.get_children()):
            return
        first = next(owner.get_tokens(), None)
        if first is not None and first.spelling == "template":
            return                      # an explicit specialization (``template<> struct S<int>``): folded too
        end = self._statement_end(owner)
        spelled = self._spell_type(member.type, drop_const=True)
        self.edits.insert(end, f"\nconstant constexpr {spelled} {owner.spelling}::{member.spelling};",
                          rank=65_000)

    #: a space-generic struct's template argument for each address space
    #: (prelude __woof_space_of)
    SPACE_INDEX = {"thread": 0, "device": 1, "constant": 2, "threadgroup": 3}

    @staticmethod
    def _closing_angle(tokens, i):
        """The index of the ``>`` closing the ``<`` at ``tokens[i]``."""
        depth = 0
        for j in range(i, len(tokens)):
            t = tokens[j].spelling
            if t == "<":
                depth += 1
            elif t in (">", ">>"):
                depth -= len(t)
                if depth <= 0:
                    return j
            elif t in (";", "{"):
                return None
        return None

    def generic_struct(self, cursor):
        """A space-generic struct (spaces.py): ``template<..., int __WS = 0>``,
        its pointer field ``__woof_sptr<T, __WS>``, a method's reference or
        pointer return ``__woof_sref``/``__woof_sptr<T, __WS>``, and every
        ``S<...>`` it names carries ``__WS`` on."""
        K = self.ci.CursorKind
        from woof.backend.translate.spaces import _indirection, _template_return
        if cursor.kind != K.CLASS_TEMPLATE:
            self.refuse(cursor, f"{cursor.spelling!r} holds pointers in two address spaces; only a "
                                "class template can carry the space as a template parameter")
        toks = list(cursor.get_tokens())
        if not toks or toks[0].spelling != "template":
            self.refuse(cursor, f"cannot find the template header of {cursor.spelling!r}")
        close = self._closing_angle(toks, 1)
        if close is None:
            self.refuse(cursor, f"cannot find the template header of {cursor.spelling!r}")
        self.edits.insert(toks[close].extent.start.offset, ", int __WS = 0", rank=-30_000)
        for c in cursor.get_children():
            if c.kind == K.FIELD_DECL and _indirection(c.type) == "ptr":
                ftoks = list(c.get_tokens())
                star = next((t for t in ftoks if t.spelling == "*"), None)
                if star is None:
                    self.refuse(c, f"cannot respell pointer field {c.spelling!r}")
                self.edits.insert(ftoks[0].extent.start.offset, "__woof_sptr<", rank=-30_000)
                self.edits.replace(star.extent.start.offset, star.extent.end.offset, ", __WS> ")
            if c.kind in (K.CONSTRUCTOR, K.CXX_METHOD, K.FUNCTION_TEMPLATE):
                # its parameters: a pointer is into the struct's space
                # (flg_column's `FLGPtr(T* v, int s = 1)`), a reference
                # (the converting copy's `const FLGPtr<U>& a`) a local
                for parm in (p for p in c.get_children() if p.kind == K.PARM_DECL):
                    ptoks = list(parm.get_tokens())
                    ind = _indirection(parm.type)
                    if not ptoks or ind not in ("ptr", "ref"):
                        continue
                    if ind == "ref":
                        self.edits.insert(ptoks[0].extent.start.offset, "thread ", rank=-30_000)
                        continue
                    star = next((t for t in ptoks if t.spelling == "*"), None)
                    if star is None:
                        self.refuse(parm, f"cannot respell pointer parameter {parm.spelling!r}")
                    self.edits.insert(ptoks[0].extent.start.offset, "__woof_sptr<", rank=-30_000)
                    self.edits.replace(star.extent.start.offset, star.extent.end.offset, ", __WS> ")
            if c.kind in (K.CXX_METHOD, K.FUNCTION_TEMPLATE):
                kind = _template_return(c) if c.kind == K.FUNCTION_TEMPLATE else _indirection(c.result_type)
                if kind not in ("ref", "ptr"):
                    continue
                mtoks = list(c.get_tokens())
                start = self._return_type_start(c)
                name = next((i for i, t in enumerate(mtoks) if t.spelling == c.spelling or
                             (c.spelling.startswith("operator") and t.spelling == "operator")), None)
                if start is None or name is None or mtoks[name - 1].spelling not in ("&", "*"):
                    self.refuse(c, f"cannot respell the return type of {c.spelling!r}")
                mark = mtoks[name - 1]
                if mark.spelling == "&" and c.kind == K.CXX_METHOD \
                        and c.result_type.get_pointee().get_declaration().spelling == cursor.spelling:
                    # `S& operator+=(...) { ...; return *this; }`: the object
                    # is a local (a thread value), whatever space it points into
                    self.edits.insert(start, "thread ", rank=-30_000)
                    continue
                alias = "__woof_sref" if mark.spelling == "&" else "__woof_sptr"
                self.edits.insert(start, f"{alias}<", rank=-30_000)
                self.edits.replace(mark.extent.start.offset, mark.extent.end.offset, ", __WS> ")
        # S<...> inside the class: the same space
        for i in range(close + 1, len(toks) - 1):
            if toks[i].spelling == cursor.spelling and toks[i + 1].spelling == "<":
                j = self._closing_angle(toks, i + 1)
                if j is not None:
                    self.edits.insert(toks[j].extent.start.offset, ", __WS", rank=-30_000)

    def generic_constructions(self, top):
        """``S<T>(p, ...)`` naming a space-generic struct outside it
        (gsfcsw's ``gsfc_ptr<real>((real *)(workspace + 0), 32)``): the
        temporary takes its pointer argument's space as ``S<T, space>``."""
        K = self.ci.CursorKind
        names = set(getattr(self, "generic_structs", ()))
        if not names:
            return
        from woof.backend.translate.spaces import _indirection
        done = set()
        for c in top:
            if c.kind == K.CLASS_TEMPLATE and c.spelling in names:
                continue
            for node in c.walk_preorder():
                if node.kind not in (K.CXX_FUNCTIONAL_CAST_EXPR, K.CALL_EXPR, K.UNEXPOSED_EXPR) \
                        or not self.main_file(node):
                    continue
                toks = list(node.get_tokens())
                if len(toks) < 4 or toks[0].spelling not in names or toks[1].spelling != "<":
                    continue
                close = self._closing_angle(toks, 1)
                if close is None or close + 1 >= len(toks) or toks[close + 1].spelling != "(":
                    continue
                at = toks[close].extent.start.offset
                if at in done or any(t.spelling == "," for t in toks[2:close]):
                    continue
                args = [a for a in node.walk_preorder() if a is not node
                        and a.extent.start.offset > toks[close + 1].extent.start.offset
                        and _indirection(a.type) == "ptr"]
                if not args:
                    continue
                space = self.spaces.space(args[0], {}) or "device"
                if space == "null":
                    space = "device"
                done.add(at)
                self.edits.insert(at, f", {self.SPACE_INDEX[space]}", rank=-30_000)

    def generic_type_close(self, tokens, name):
        """Where a ``name<...>`` among ``tokens`` takes its space argument."""
        for i in range(len(tokens) - 1):
            if tokens[i].spelling == name and tokens[i + 1].spelling == "<":
                j = self._closing_angle(tokens, i + 1)
                if j is not None:
                    return tokens[j].extent.start.offset
        return None

    def struct_fields(self, cursor):
        """Pointer fields take their initialiser's space (else device),
        reference members thread; static data members are constant."""
        K = self.ci.CursorKind
        from woof.backend.translate.spaces import _indirection
        if cursor.spelling in getattr(self, "generic_structs", ()):
            self.generic_struct(cursor)
            return
        seen = set()
        groups = {}
        for c in cursor.get_children():
            if c.kind == K.FIELD_DECL:
                groups.setdefault(c.extent.start.offset, []).append(c)
        for fields in groups.values():
            kinds = {(_indirection(f.type), self._field_space(f) if _indirection(f.type) == "ptr" else None)
                     for f in fields}
            if (len(fields) > 1 and len(kinds) > 1 and any(k[0] == "ptr" for k in kinds)
                    and self._split_field_group(fields)):
                seen.add(fields[0].extent.start.offset)
        from woof.backend.translate.spaces import _is_table
        for c in cursor.get_children():
            if c.kind == K.FIELD_DECL and _is_table(c.type) and len(groups.get(c.extent.start.offset, ())) == 1:
                # a T** field holds a kernel's address table (cam3's
                # CamState::abs_args = descriptors + worker*41): the table type
                target = c.type.get_canonical().get_pointee().get_pointee()
                tspace = "const device" if target.is_const_qualified() else "device"
                elem = self._spell_type(target, drop_const=True)
                toks = list(c.get_tokens())
                name = next((t for t in toks if t.spelling == c.spelling), None)
                if name is not None and toks:
                    self.edits.replace(toks[0].extent.start.offset, name.extent.start.offset,
                                       f"__woof_addr_table<{tspace} {elem}*> ")
                    seen.add(c.extent.start.offset)
                    continue
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
                self._static_member_definition(cursor, c)
            elif c.kind == K.CONSTRUCTOR and c.is_definition():
                self._arithmetic_constructor(cursor, c)
            elif c.kind in (K.STRUCT_DECL, K.CLASS_DECL, K.CLASS_TEMPLATE,
                            K.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION):
                self.struct_fields(c)

    def _split_field_group(self, fields):
        """``T *a, b, *c;`` with pointer and non-pointer fields (or pointers in
        different spaces) under one decl-specifier: one declaration per field,
        each pointer qualified with its own space, since a space in the shared
        decl-specifier would qualify the non-pointer fields too (cam3's
        CamState: field may not be qualified with an address space).  False
        when the declarators cannot be found."""
        from woof.backend.translate.spaces import _indirection
        toks = [t for t in self.tu.get_tokens(extent=self.ci.SourceRange.from_locations(
            fields[0].extent.start, fields[-1].extent.end))]
        first_name = next((i for i, t in enumerate(toks) if t.spelling == fields[0].spelling), None)
        if first_name is None:
            return False
        base_end = first_name
        while base_end > 0 and toks[base_end - 1].spelling in ("*", "&", "const", "__restrict__"):
            base_end -= 1
        if base_end == 0:
            return False
        base = " ".join(t.spelling for t in toks[:base_end])
        depth, commas = 0, []
        for t in toks[first_name:]:
            if t.spelling in ("(", "[", "{"):
                depth += 1
            elif t.spelling in (")", "]", "}"):
                depth -= 1
            elif t.spelling == "," and depth == 0:
                commas.append(t)
        if len(commas) != len(fields) - 1:
            return False
        def prefix(f):
            return f"{self._field_space(f)} " if _indirection(f.type) == "ptr" else ""

        def spelled(f):
            # inserted text is not respelled: the element type as the
            # translator spells types (f64, f32s)
            t = f.type.get_canonical()
            while t.kind.name in ("POINTER", "CONSTANTARRAY", "INCOMPLETEARRAY"):
                t = t.get_pointee() if t.kind.name == "POINTER" else t.element_type
            return self._spell_type(t, drop_const=False)
        self.edits.insert(fields[0].extent.start.offset, prefix(fields[0]), rank=-20_000)
        for comma, f in zip(commas, fields[1:]):
            self.edits.replace(comma.extent.start.offset, comma.extent.end.offset, f"; {prefix(f)}{spelled(f)} ")
        return True

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
        if len(given) > 1 and getattr(self, "_probing_fields", False):
            return sorted(given)[0]
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
                if g.get("cell"):
                    # what the stored address points at is invisible to
                    # MLX's hazard tracking, as a pointer table's targets are
                    self._indirect_kernels.add(name)
        scalar_size = (offset + max_align - 1) // max_align * max_align if scalars else 0
        table = len(pointers) > MAX_DIRECT_POINTERS
        sig_params, prologue, structs = [], [], []
        repointed = []
        # A const pointer parameter the kernel points elsewhere (a layout
        # trial's `dnw = __shared_cache;`): a variant pointer, bound to the
        # buffer and reassignable to another space (prelude __woof_vptr).
        for parm in self._kernel_params_repointed(fn, params):
            pname = parm.spelling
            slot = next((k for k, (pn, _) in enumerate(pointers) if pn == pname), None)
            if slot is None:
                continue
            pointers[slot] = (f"__woof_vp_{pname}", pointers[slot][1])
            elem = self._spell_type(parm.type.get_pointee(), drop_const=True)
            repointed.append(f"__woof_vptr<{elem}> {pname}(__woof_vp_{pname});")
        if table:
            fields = "\n".join(f"    {ty} {pn};" for pn, ty in pointers)
            structs.append(f"struct __woof_table_{name} {{\n{fields}\n}};")
            sig_params.append(f"constant __woof_table_{name}& __woof_table [[buffer(0)]]")
            prologue += [f"{ty} {pn} = __woof_table.{pn};" for pn, ty in pointers]
            table_slot, scalar_slot = 0, 1
        else:
            sig_params += [f"{ty} {pn} [[buffer({i})]]" for i, (pn, ty) in enumerate(pointers)]
            table_slot, scalar_slot = -1, len(pointers)
        prologue += addr_tables + repointed
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
        # CUDA's binary64 vector types, which MSL reserves (a double2*
        # kernel parameter: global_polar's spectrum)
        spelled = re.sub(r"\bdouble([234])\b", r"__woof_double\1", spelled)
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
            if kind == K.DECL_REF_EXPR and node.spelling in ("threadIdx", "blockIdx", "blockDim", "gridDim") \
                    and not self._shadows_builtin(node):
                # Kernels have these as parameters; device functions that
                # read them receive them (thread_builtins below).
                s, e = self.span(node)
                if getattr(self, "_file_scope_builtins", False):
                    self.edits.replace(s, e, "uint3()")
                    continue
                self.edits.replace(s, e, {"threadIdx": "__woof_tid", "blockIdx": "__woof_bid",
                                          "blockDim": "__woof_bdim", "gridDim": "__woof_gdim"}[node.spelling])
                continue
            if kind == K.INIT_LIST_EXPR:
                self.fold_init_list(node)
            if kind == K.ARRAY_SUBSCRIPT_EXPR:
                self.opaque_constant_index(node)
            if kind == K.UNARY_OPERATOR and not self.retain and _is_float(node.type):
                self.flushing_negation(node)
            if kind == K.CONDITIONAL_OPERATOR and _is_float(node.type):
                self.select_minmax(node)
            if kind == K.CXX_STATIC_CAST_EXPR:
                self.volatile_load(node)
            if kind == K.UNARY_OPERATOR and _is_double(node.type):
                inner = list(node.get_children())
                toks = [t.spelling for t in node.get_tokens()]
                if (len(inner) == 1 and inner[0].kind == K.FLOATING_LITERAL and toks and toks[0] == "-"
                        and _is_double(inner[0].type)):
                    s, e = self.span(node)
                    word = _double_literal(self.src(inner[0])) ^ (1 << 63)
                    self.edits.replace(s, e, f"__woof_f64_lit(0x{word:016X}UL)")
                    continue
            if kind == K.CALL_EXPR and self.retain:
                self.qualify_namespace_call(node)
                self.f32s_literal_arguments(node)
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
            if self.retain and self.f32s_vector_component(node, d):
                pass
            elif kind == K.COMPOUND_ASSIGNMENT_OPERATOR:
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

    _PURE = ("DECL_REF_EXPR", "MEMBER_REF_EXPR", "ARRAY_SUBSCRIPT_EXPR", "FLOATING_LITERAL",
             "INTEGER_LITERAL", "PAREN_EXPR", "UNEXPOSED_EXPR")

    def select_minmax(self, node):
        """``a > b ? a : b`` (and its mirrors): NVCC emits max.f32 / min.f32,
        which order -0 below +0, where the conditional as written returns b
        for equal zeros.  Rewritten to __woof_selmax/__woof_selmin: the
        conditional's own result except on equal operands, where CUDA's
        signed-zero order holds.  (RRTMG's prep preflight asserts
        rp_max(+0, -0) == +0 on the card.)"""
        kids = list(node.get_children())
        if len(kids) != 3 or kids[0].kind.name != "BINARY_OPERATOR":
            return
        cond, yes, no = kids
        ops = list(cond.get_children())
        if len(ops) != 2:
            return
        op = next((t.spelling for t in cond.get_tokens()
                   if ops[0].extent.end.offset <= t.extent.start.offset < ops[1].extent.start.offset), None)
        if op not in (">", ">=", "<", "<="):
            return
        text = lambda c: "".join(t.spelling for t in c.get_tokens())
        def pure(c):
            return all(n.kind.name in self._PURE or n.kind.name.startswith("TYPE") for n in c.walk_preorder())
        if not all(pure(c) for c in (ops[0], ops[1], yes, no)):
            return
        l, r, y, n = text(ops[0]), text(ops[1]), text(yes), text(no)
        if (y, n) == (l, r):
            kind = "max" if op in (">", ">=") else "min"
        elif (y, n) == (r, l):
            kind = "min" if op in (">", ">=") else "max"
        else:
            return
        s, e = self.span(node)
        ys, ye = self.span(yes)
        ns, ne = self.span(no)
        self.edits.replace(s, ys, f"__woof_sel{kind}(")
        self.edits.replace(ye, ns, ", ")
        self.edits.replace(ne, e, ")")

    def volatile_load(self, node):
        """``static_cast<volatile T&>(x)`` read as a value (convective_rrfs
        keeps a load from sinking this way): MSL's reference cast must name
        x's address space, which a by-value accessor's ``operator[]`` in a
        template does not tell; ``__woof_vload(x)`` finds it by overload and
        loads through a volatile pointer.  A write through the cast is
        refused."""
        toks = list(node.get_tokens())
        spelled = [t.spelling for t in toks]
        if len(spelled) < 5 or spelled[0] != "static_cast" or spelled[1] != "<":
            return
        depth = 0
        for close, t in enumerate(spelled):
            depth += t == "<"
            depth -= t == ">"
            if depth == 0 and close > 1:
                break
        target = spelled[2:close]
        if not target or target[-1] != "&" or "volatile" not in target or close + 1 >= len(toks) \
                or spelled[close + 1] != "(":
            return
        after = self.tu.get_tokens(extent=self.tu.get_extent(
            "unit.cpp", (toks[-1].extent.end.offset, toks[-1].extent.end.offset + 3)))
        following = next(iter(after), None)
        if following is not None and following.spelling in ("=", "+=", "-=", "*=", "/=", "++", "--"):
            self.refuse(node, "a write through static_cast<volatile T&> has no Metal spelling")
        self.edits.replace(toks[0].extent.start.offset, toks[close + 1].extent.end.offset, "__woof_vload(")

    def flushing_negation(self, node):
        """``-x`` on a binary32 value in a -ftz=true unit: CUDA's neg.ftz
        flushes a subnormal x to -0, Metal's negation flips the bit and keeps
        it.  ``-<literal>`` is a constant and stays as written."""
        kids = list(node.get_children())
        toks = list(node.get_tokens())
        if len(kids) != 1 or not toks or toks[0].spelling != "-":
            return
        inner = kids[0]
        while inner.kind.name in ("PAREN_EXPR", "UNEXPOSED_EXPR") and len(list(inner.get_children())) == 1:
            inner = next(inner.get_children())
        if inner.kind.name in ("FLOATING_LITERAL", "INTEGER_LITERAL") or self.const_eval(inner) is not None:
            return                  # a constant (and constant initialisers must stay constexpr)
        s, e = self.span(node)
        self.edits.replace(toks[0].extent.start.offset, toks[0].extent.end.offset, "__woof_neg(")
        self.edits.insert(e, ")", rank=60_000)

    def static_local(self, node):
        """MSL has no function-scope statics.  A constant one is the same
        value initialised per call, so ``static`` just goes; a mutable one
        (state kept across calls) has no spelling and is refused."""
        t = node.type
        const = t.is_const_qualified()
        while t.get_canonical().kind.name in ("CONSTANTARRAY", "INCOMPLETEARRAY"):
            # libclang puts an array's const on the canonical array and can
            # drop it from that array's element type (``static const double
            # tr[] = {...}`` in core-math's log10f read as mutable)
            const = const or t.get_canonical().is_const_qualified()
            t = t.get_canonical().element_type
        if not (const or t.is_const_qualified()):
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

    def f32s_literal_arguments(self, call):
        """In a subnormal-retaining unit ``float`` is ``f32s``, but a float
        literal is still ``float``: passed to ``g(f64, float)`` beside an
        overload ``g(f64, f64)``, both need a user-defined conversion and
        Metal calls the call ambiguous.  CUDA chose the float parameter, so
        the literal argument of a parameter clang resolved as ``float``
        becomes ``f32s(...)`` (goddard's g_div / g_pow)."""
        ref = call.referenced
        if ref is None or ref.kind.name not in ("FUNCTION_DECL", "CXX_METHOD") or not self.main_file(ref):
            return
        params = [c for c in ref.get_children() if c.kind.name == "PARM_DECL"]
        for arg, parm in zip(call.get_arguments(), params):
            if parm.type.get_canonical().kind.name != "FLOAT":
                continue
            inner = arg
            while inner.kind.name in ("UNEXPOSED_EXPR", "PAREN_EXPR") and len(list(inner.get_children())) == 1:
                inner = next(inner.get_children())
            literal = inner
            if inner.kind.name == "UNARY_OPERATOR" and len(list(inner.get_children())) == 1:
                literal = next(inner.get_children())
            if literal.kind.name != "FLOATING_LITERAL" or literal.type.get_canonical().kind.name != "FLOAT":
                # a constant expression of float literals (`1.f/3000.f`) is
                # a Metal float too (chem_drydep_wesely's WFloat operators)
                if not self._float_literal_arithmetic(inner):
                    continue
            s, e = self.span(inner)
            self.edits.wrap(s, e, "f32s(", ")", 900)

    @staticmethod
    def _float_literal_arithmetic(node) -> bool:
        if node.kind.name not in ("BINARY_OPERATOR", "UNARY_OPERATOR"):
            return False
        for n in node.walk_preorder():
            k = n.kind.name
            if k in ("BINARY_OPERATOR", "UNARY_OPERATOR", "PAREN_EXPR", "UNEXPOSED_EXPR"):
                if n.type.get_canonical().kind.name != "FLOAT":
                    return False
            elif k != "FLOATING_LITERAL" or n.type.get_canonical().kind.name != "FLOAT":
                return False
        return True

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

    #: Private arrays longer than this many elements take constant indices
    #: through __woof_opaque_index (opaque_constant_index).
    PRIVATE_ARRAY_CONSTANT_INDEX_LIMIT = 32

    def opaque_constant_index(self, node):
        """``a[0]`` on a function-local array of more than 32 elements ->
        ``a[__woof_opaque_index(0)]``.  Apple's shader compiler (macOS 26.5,
        M3 Max) reads such an element as 0 when the array was filled and
        read by loops of runtime trip count before it (``float r[64]; for (k
        < nz) r[k] = ...; for (k < nz) t += r[k]; out = r[0];`` gives 0, in
        a plain mx.fast.metal_kernel too; 32 elements or a runtime index read
        it right).  NSSL-2's rain sedimentation exported rho[0] * flux as 0
        through it.  The index's value is unchanged."""
        kids = list(node.get_children())
        if len(kids) != 2:
            return
        base, index = kids
        while base.kind.name in ("UNEXPOSED_EXPR", "PAREN_EXPR") and len(list(base.get_children())) == 1:
            base = list(base.get_children())[0]
        d = base.referenced if base.kind.name == "DECL_REF_EXPR" else None
        if d is None or d.kind.name != "VAR_DECL" or d.semantic_parent is None \
                or d.semantic_parent.kind.name not in self._FUNCTION_KINDS:
            return
        if any(c.kind.name == "ANNOTATE_ATTR" for c in d.get_children()):
            return          # __shared__ (threadgroup) or another marked array
        t = d.type.get_canonical()
        if t.kind.name == "CONSTANTARRAY":
            if t.element_count <= self.PRIVATE_ARRAY_CONSTANT_INDEX_LIMIT:
                return
        elif t.kind.name not in ("DEPENDENTSIZEDARRAY",):
            return
        literal = index
        while literal.kind.name in ("UNEXPOSED_EXPR", "PAREN_EXPR") and len(list(literal.get_children())) == 1:
            literal = list(literal.get_children())[0]
        if literal.kind.name != "INTEGER_LITERAL":
            return
        s, e = self.span(index)
        if s < e:
            self.edits.wrap(s, e, "__woof_opaque_index(", ")", 0)

    def qualify_namespace_call(self, node):
        """An unqualified call to a namespace's function, in an f32s unit:
        ``N::f(...)``.  f32s is a class at global scope, so argument-dependent
        lookup adds a global function of the same name and signature to the
        candidates and the call is ambiguous, where CUDA's float arguments
        bring no ADL (the selected Noah-MP water unit's d_canwater, global and
        in namespace noahmp_water_options)."""
        ref = node.referenced
        if ref is None or ref.kind.name not in ("FUNCTION_DECL", "FUNCTION_TEMPLATE"):
            return
        scopes, owner = [], ref.semantic_parent
        while owner is not None and owner.kind.name == "NAMESPACE":
            scopes.append(owner.spelling)
            owner = owner.semantic_parent
        if not scopes or owner is None or owner.kind.name != "TRANSLATION_UNIT" or "" in scopes:
            return
        toks = list(node.get_tokens())
        if not toks or toks[0].spelling != ref.spelling or not self.main_file(node):
            return
        start = toks[0].extent.start.offset
        if self.data[max(0, start - 2):start] == b"::":
            return
        self.edits.insert(start, "::" + "::".join(reversed(scopes)) + "::", rank=-65_000)

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
            if _is_float(target) and t.kind.name == "RECORD" and t.get_declaration().kind.name == "UNION_DECL":
                # a binary64 literal initialising a union's float member
                # (gsfcgce4ice's `union{float f; unsigned u;} st[] =
                # {{0x1.2a05f2p+33}, ...}`): its binary32 rounding, spelled
                literal = child
                while literal.kind.name in ("UNEXPOSED_EXPR", "PAREN_EXPR") and len(list(literal.get_children())) == 1:
                    literal = list(literal.get_children())[0]
                if literal.kind.name == "FLOATING_LITERAL" and _is_double(literal.type):
                    word = _double_literal(self.src(literal))
                    try:
                        value = struct.unpack("<f", struct.pack("<f", struct.unpack("<d", struct.pack("<Q", word))[0]))[0]
                    except OverflowError:
                        continue
                    s, e = self.span(literal)
                    self.edits.replace(s, e, f"{value.hex()}f")
                    self._folded.add((s, e))
                continue
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

    def _kernel_params_repointed(self, fn, params):
        """Pointer-to-const kernel parameters assigned a pointer that is not
        in device memory."""
        K = self.ci.CursorKind
        consts = {p.get_usr(): p for p in params
                  if p.type.get_canonical().kind.name == "POINTER"
                  and p.type.get_pointee().is_const_qualified()}
        out = {}
        if not consts:
            return []
        for node in fn.walk_preorder():
            if node.kind != K.BINARY_OPERATOR:
                continue
            kids = list(node.get_children())
            if len(kids) != 2 or kids[0].kind != K.DECL_REF_EXPR or kids[0].referenced is None:
                continue
            usr = kids[0].referenced.get_usr()
            if usr not in consts:
                continue
            op = next((t.spelling for t in node.get_tokens()
                       if t.extent.start.offset >= kids[0].extent.end.offset), None)
            if op == "=" and self.spaces.space(kids[1], {}) not in ("device", None):
                out[usr] = consts[usr]
        return list(out.values())

    @staticmethod
    def _shadows_builtin(node) -> bool:
        """``const dim3 blockIdx(...)`` in a kernel (the ensemble batch
        adapter's virtual block coordinates) is a local that hides CUDA's
        builtin: a reference to it stays a reference to the local."""
        ref = node.referenced
        if ref is None or ref.kind.name not in ("VAR_DECL", "PARM_DECL"):
            return False
        sp = ref.semantic_parent
        return sp is not None and sp.kind.name in _Rewriter._FUNCTION_KINDS

    @staticmethod
    def _float_vector_component(node) -> bool:
        """``v.x`` / ``p->y`` of a CUDA float vector (float1..float4).  Its
        Metal namesake keeps ``float`` components in a unit whose scalar
        ``float`` is ``f32s``."""
        if node.kind.name != "MEMBER_REF_EXPR" or not _is_float(node.type) \
                or node.spelling not in ("x", "y", "z", "w"):
            return False
        kids = list(node.get_children())
        if len(kids) != 1:
            return False
        t = kids[0].type.get_canonical()
        if t.kind.name == "POINTER":
            t = t.get_pointee().get_canonical()
        name = t.spelling.replace("const ", "").replace("volatile ", "").replace("struct ", "").strip()
        return re.fullmatch(r"float[1-4]", name) is not None

    def f32s_vector_component(self, node, depth) -> bool:
        """``retain_subnormals``: float vector components stay ``float`` (a
        Metal vector of f32s does not exist), while the unit's scalar float
        is ``f32s``, which converts to ``float`` only explicitly.  So

        * a component read as a value is ``f32s(v.x)`` -- arithmetic on it
          keeps subnormals, as CUDA's --ftz=false does;
        * ``v.x = e`` stores ``float(e)`` (the same bits);
        * ``v.x op= e`` is ``v.x = float(f32s(f32s(v.x) op (e)))``;
        * a vector built by ``make_floatN(...)`` or ``{...}`` takes
          ``float(e)`` elements.

        Returns True when the node was rewritten as a compound assignment
        (so the binary64 compound rewrite does not also apply)."""
        K = self.ci.CursorKind
        kind = node.kind
        kids = list(node.get_children())
        if kind == K.UNEXPOSED_EXPR and len(kids) == 1 and _is_float(node.type) \
                and self._float_vector_component(kids[0]):
            s, e = self.span(kids[0])
            if s < e:
                self.edits.wrap(s, e, "f32s(", ")", depth + 1)
            return False
        if kind in (K.BINARY_OPERATOR, K.COMPOUND_ASSIGNMENT_OPERATOR) and len(kids) == 2 \
                and self._float_vector_component(kids[0]):
            lhs, rhs = kids
            op = next((t.spelling for t in node.get_tokens()
                       if lhs.extent.end.offset <= t.extent.start.offset < rhs.extent.start.offset), None)
            rs, re_ = self.span(rhs)
            if op == "=":
                self.edits.wrap(rs, re_, "float(", ")", depth + 1)
                return False
            if op in ("+=", "-=", "*=", "/="):
                lhs_text = self.src(lhs)
                if not re.fullmatch(r"[\w\s\.\[\]\(\)\->+*]*", lhs_text) or "++" in lhs_text or "--" in lhs_text:
                    self.refuse(node, f"compound assignment to vector component {lhs_text!r} "
                                      "would evaluate the target twice")
                self.edits.replace(lhs.extent.end.offset, rs,
                                   f" = float(f32s(f32s({lhs_text}) {op[0]} (")
                self.edits.insert(re_, ")))", rank=-depth)
                return True
            return False
        building = False
        if kind == K.CALL_EXPR and re.fullmatch(r"make_float[1-4]", node.spelling or ""):
            building, args = True, kids[1:]
        elif kind == K.INIT_LIST_EXPR:
            name = node.type.get_canonical().spelling.replace("const ", "").replace("struct ", "").strip()
            building, args = re.fullmatch(r"float[1-4]", name) is not None, kids
        if building:
            for a in args:
                if _is_float(a.type):
                    s, e = self.span(a)
                    if s < e:
                        self.edits.wrap(s, e, "float(", ")", depth + 1)
        return False

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

    @staticmethod
    def _volatile_float_value(tokens, i) -> bool:
        """tokens[i] is `volatile` qualifying a float *value* (not the
        pointee of a pointer or reference): `volatile [const] float x`."""
        quals = ("const", "volatile")
        j = i
        while j > 0 and tokens[j - 1].spelling in quals:
            j -= 1
        k = i + 1
        while k < len(tokens) and tokens[k].spelling in quals:
            k += 1
        if k < len(tokens) and tokens[k].spelling == "float":
            k += 1
        elif not (j > 0 and tokens[j - 1].spelling == "float"):
            return False
        while k < len(tokens) and tokens[k].spelling in quals:
            k += 1
        return k < len(tokens) and tokens[k].kind.name == "IDENTIFIER"

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
                if sp == "volatile" and self.retain and self._volatile_float_value(tokens, i):
                    # `volatile float x` pins a rounding in CUDA (no
                    # contraction across it).  f32s arithmetic is calls,
                    # never contracted, and a volatile f32s has no copy
                    # constructor: the qualifier goes.
                    if not self._inside_replacement(t.extent.start.offset):
                        self.edits.replace(t.extent.start.offset, t.extent.end.offset, "")
                    i += 1
                    continue
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


def prelude(*, contract: bool, retain_subnormals: bool = False, ieee_helpers: bool = False,
            f64: str = "soft") -> str:
    """``retain_subnormals``: the unit is --ftz=false (float is f32s);
    ``ieee_helpers``: it uses the subnormal-keeping ops of non-.ftz PTX,
    which f32s.h also defines, in a unit that otherwise flushes;
    ``f64``: ``soft`` (strict: f64_soft.h, with f64_df64.h's exact division
    and square root) or ``df64`` (fast numerics, :func:`_df64_prelude`)."""
    soft = (METAL_DIR / "f64_soft.h").read_text()
    if f64 == "df64":
        return _df64_prelude(soft, contract=contract, retain_subnormals=retain_subnormals,
                             ieee_helpers=ieee_helpers)
    if f64 != "soft":
        raise ValueError(f"f64 must be soft or df64, not {f64!r}")
    # strict: soft-float, but division and square root from f64_df64.h's
    # exact df64-seeded words (the soft-float words, tools/df64_grade.py
    # strict), without its libm
    exact = ("#define WOOF_DF64_ARITH 1\n#define WOOF_DF64_NO_LIBM 1\n"
             + (METAL_DIR / "f64_df64.h").read_text())
    text = (METAL_DIR / "prelude.h").read_text().replace("// F64_SOFT_H_TEXT", soft + "\n" + exact)
    if retain_subnormals:
        text = "#define WOOF_RETAIN_SUBNORMALS 1\n" + text
    if retain_subnormals or ieee_helpers:
        text += "\n" + (METAL_DIR / "f32s.h").read_text()
    # The prelude always compiles uncontracted: __fmul_rn and kin promise one
    # rounding each, and LLVM records contraction per instruction, so their
    # bodies stay unfusable after inlining into a unit compiled contract(fast).
    after = "\n#pragma METAL fp contract(fast)\n" if contract else ""
    return "#pragma METAL fp contract(off)\n" + text + after


def _df64_prelude(soft: str, *, contract: bool, retain_subnormals: bool, ieee_helpers: bool) -> str:
    """The fast numerics' prelude: f64_soft.h, then f64_df64.h (division and
    square root through its exact df64-seeded words, WOOF_DF64_ARITH), and
    after the prelude the df64 libm's overloads on ``f64``
    (f64_df64_bind.h)."""
    df64 = "#define WOOF_DF64_ARITH 1\n" + (METAL_DIR / "f64_df64.h").read_text()
    text = (METAL_DIR / "prelude.h").read_text().replace("// F64_SOFT_H_TEXT", soft + "\n" + df64)
    if retain_subnormals:
        text = "#define WOOF_RETAIN_SUBNORMALS 1\n" + text
    if retain_subnormals or ieee_helpers:
        text += "\n" + (METAL_DIR / "f32s.h").read_text()
    text += "\n" + (METAL_DIR / "f64_df64_bind.h").read_text()
    after = "\n#pragma METAL fp contract(fast)\n" if contract else ""
    return "#pragma METAL fp contract(off)\n" + text + after


VENDOR_LIBM64 = HERE / "vendor_libm64.cuh"
#: WOOF's correctly rounded powf, the MLX facade's strict float32 power
#: (facade/_ops.py; WOOF_STRICT_POWF=cuda selects the opt-in instead)
STRICT_POWF = HERE / "strict_powf.cuh"
F64CR = HERE.parents[1] / "core" / "kernels" / "f64cr.cuh"
_VENDOR_CALL = re.compile(r"\b(exp|log|log10|sin|cos|tan|atan|pow|atan2|hypot|log2|expm1|tanh|tgammaf?|lgamma|log1pf?|cbrtf?|erfc?)\s*\(")
CR_CBRT = HERE.parents[1] / "core" / "kernels" / "cr_cbrt_flt64.cuh"
#: binary64 functions the df64 prelude provides (f64_df64_bind.h): under
#: f64="df64" the vendored libm keeps only its other entry points
DF64_NATIVE = ("exp", "expm1", "log", "log2", "log10", "log1p", "pow", "sin", "cos", "tan",
               "atan", "atan2", "tanh", "hypot")
_VENDOR_CALL_DF64 = re.compile(r"\b(tgammaf?|lgamma|log1pf|cbrtf|erfc?)\s*\(")


def compose(source: str, expanded: str | None = None, f64: str = "soft") -> str:
    """Prepend the binary64 libm (vendor_libm64.cuh, with f64cr.cuh) to a
    unit that calls one of CUDA's binary64 libm names, in its own text or
    (``expanded``, the preprocessed unit) in a header it includes.

    ``f64="df64"``: the df64 prelude defines :data:`DF64_NATIVE` and cbrt
    itself, so only a unit calling one of the rest (tgamma, lgamma, erf,
    erfc and the binary32 log1pf, cbrtf, tgammaf) gets the vendored text,
    without its wrappers for the native names."""
    if f64 == "df64":
        return _compose_df64(source, expanded)
    if not _VENDOR_CALL.search(source) and not (expanded and _VENDOR_CALL.search(expanded)):
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


def _compose_df64(source: str, expanded: str | None) -> str:
    if not _VENDOR_CALL_DF64.search(source) and not (expanded and _VENDOR_CALL_DF64.search(expanded)):
        return source
    f64cr = F64CR.read_text(encoding="utf-8")
    hook = "#define WCR_PLATFORM_SIN(x) sin(x)"
    if hook not in f64cr:
        raise TranslationError("f64cr.cuh no longer defines its CUDA platform-sin hook as expected")
    f64cr = f64cr.replace(hook, "#define WCR_PLATFORM_SIN(x) wlm_sin_huge(x)")
    vendor, kept = [], 0
    wrapper = re.compile(r"^__device__ __forceinline__ (?:double|float) (\w+)\(")
    for line in VENDOR_LIBM64.read_text(encoding="utf-8").splitlines():
        m = wrapper.match(line)
        if m and m.group(1) in DF64_NATIVE:
            continue
        kept += bool(m and m.group(1) in ("tgamma", "lgamma", "erf", "erfc"))
        vendor.append(line)
    if kept != 4:
        raise TranslationError("vendor_libm64.cuh no longer defines its tgamma/lgamma/erf/erfc wrappers as expected")
    return ("__device__ double wlm_sin_huge(double x);\n" + f64cr + "\n" + "\n".join(vendor) + "\n" + source)


def _integer_macros(source: str, defines=()) -> dict:
    """Each macro the unit (or a -D option) defines to an integer: the
    largest value given (a tier's ``#define KMAX 128`` ahead of a header's
    ``#ifndef`` default of 64 reads 128)."""
    out: dict = {}
    pairs = re.findall(r"^[ \t]*#[ \t]*define[ \t]+(\w+)[ \t]+(\d+)[ \t]*$", source, re.M)
    pairs += [tuple(d.split("=", 1)) for d in defines if "=" in d]
    for name, value in pairs:
        if str(value).isdigit():
            out[name] = max(out.get(name, 0), int(value))
    return out


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


def option_include_dirs(options) -> tuple[str, ...]:
    """The ``-I DIR`` include directories among NVRTC options."""
    out, it = [], iter(options or ())
    for o in it:
        if o in ("-I", "--include-path"):
            out.append(next(it, ""))
        elif o.startswith("--include-path="):
            out.append(o.split("=", 1)[1])
        elif o.startswith("-I") and len(o) > 2:
            out.append(o[2:])
    return tuple(d for d in out if d)


def explicit_float_to_integral(text: str) -> str:
    """``retain_subnormals`` with inheritance: ``f32s`` converts to the
    integer types only explicitly, and the translator spells C++'s implicit
    float -> integer conversions as casts where it sees them.  But
    :func:`flatten_inheritance` makes every function taking a base class a
    template, and in that dependent code no conversion is visible (CLM
    lake's ``snl(c) = snl2d(i,j)``).  So the casts are spelled first, on
    the unit as written."""
    import clang.cindex as ci
    if not re.search(r"\b(struct|class)\s+\w+\s*:\s*(public\s+)?\w", text):
        return text
    tu = parse(text)
    edits = Edits(text.encode())
    for n in tu.cursor.walk_preorder():
        if n.kind != ci.CursorKind.UNEXPOSED_EXPR or n.location.file is None \
                or n.location.file.name != "unit.cpp":
            continue
        if n.type.get_canonical().kind.name not in _INTEGRAL_KINDS:
            continue
        kids = list(n.get_children())
        if len(kids) != 1 or not _is_float(kids[0].type):
            continue
        s, e = kids[0].extent.start.offset, kids[0].extent.end.offset
        if s < e:
            spelled = n.type.get_canonical().spelling.replace("const ", "").strip()
            edits.wrap(s, e, f"(({spelled})(", "))", 0)
    return edits.apply()


def hoist_block_prototypes(text: str) -> str:
    """f2c declares the functions a body calls inside it (``extern int
    f_(integer *, ...);``).  The translator gives a function's declarations
    its address spaces and implicit parameters, but a declaration inside a
    body that is itself rendered once per address-space instance would be
    a region inside that region.  So each such declaration statement moves
    to file scope, just before the function that holds it (the same
    namespace, so it names the same function)."""
    import clang.cindex as ci
    if not re.search(r"[;{]\s*extern\s[^;{}()]*\w\s*\(", text):
        return text
    tu = parse(text)
    K = ci.CursorKind
    data = text.encode()
    edits = Edits(data)
    for fn in tu.cursor.walk_preorder():
        if fn.kind not in (K.FUNCTION_DECL, K.FUNCTION_TEMPLATE) or not fn.is_definition() \
                or fn.location.file is None or fn.location.file.name != "unit.cpp" \
                or fn.semantic_parent is None \
                or fn.semantic_parent.kind not in (K.TRANSLATION_UNIT, K.NAMESPACE):
            continue
        moved = []
        for stmt in fn.walk_preorder():
            if stmt.kind != K.DECL_STMT:
                continue
            kids = list(stmt.get_children())
            if kids and all(k.kind == K.FUNCTION_DECL for k in kids):
                s, e = stmt.extent.start.offset, stmt.extent.end.offset
                # `extern int f(...), g(...);`: one declaration each (each
                # is rendered per instance on its own)
                prefix = data[s:kids[0].location.offset]
                ends = [kids[0].extent.end.offset] + [k.extent.end.offset for k in kids[1:]]
                if any(b"*" in data[ends[i - 1]:k.location.offset] or b"&" in data[ends[i - 1]:k.location.offset]
                       for i, k in enumerate(kids) if i):
                    moved.append(data[s:e].decode())
                else:
                    moved.extend((prefix + data[k.location.offset:k.extent.end.offset]).decode() + ";"
                                 for k in kids)
                edits.replace(s, e, "")
        if moved:
            at = data.rfind(b"\n", 0, fn.extent.start.offset) + 1
            edits.insert(at, "\n".join(moved) + "\n", 0)
    return edits.apply()


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


#: Environment override for the translation cache directory ("off": none).
TRANSLATION_DIR_ENV = "WOOF_METAL_TRANSLATION_DIR"


@lru_cache(maxsize=1)
def translator_digest() -> str:
    """Everything a translation depends on besides its preprocessed text:
    the translator's own sources, the Metal headers its prelude carries and
    the libclang it parses with."""
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    files = sorted(root.glob("*.py")) + sorted(root.glob("*.h")) + sorted(root.glob("*.cuh")) \
        + sorted((root.parent / "metal").glob("*.h"))
    for path in files:
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    try:
        from importlib.metadata import version
        digest.update(version("libclang").encode())
    except Exception:  # noqa: BLE001 -- the bindings' version is best effort
        pass
    return digest.hexdigest()


def _translation_dir() -> Path | None:
    env = os.environ.get(TRANSLATION_DIR_ENV)
    if env in ("off", "0"):
        return None
    if env:
        return Path(env).expanduser()
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return (base / "woof" / "metal-translations").expanduser()


def _translation_cache(text, unit, contract, retain_subnormals, source, defines, f64):
    """``(cached unit or None, store)``: translations persist across processes
    under :func:`_translation_dir`, keyed by the PREPROCESSED text (so an
    edited header is a new key), the unit's name and flags (``f64`` among them), the integer
    macros the source and options define, and :func:`translator_digest`.
    Parsing and rewriting a large unit costs seconds; a hit costs a read."""
    directory = _translation_dir()
    if directory is None:
        return None, lambda result: None
    key = hashlib.sha256("\0".join((
        translator_digest(), unit, str(bool(contract)), str(bool(retain_subnormals)), f64,
        repr(sorted(_integer_macros(source, defines).items())), text)).encode()).hexdigest()
    path = directory / f"{key}.pickle"
    try:
        with open(path, "rb") as f:
            return pickle.load(f), lambda result: None
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ImportError):
        pass

    def store(result: TranslatedUnit) -> None:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(f"{path.name}.{os.getpid()}.partial")
            with open(partial, "wb") as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(partial, path)
        except OSError:
            pass
    return None, store


def translate(source: str, *, unit: str = "<unit>", contract: bool = False,
              retain_subnormals: bool = False, defines=(), include_dirs=(),
              f64: str = "soft") -> TranslatedUnit:
    """CUDA unit text -> :class:`TranslatedUnit` (MSL plus kernel signatures).

    ``retain_subnormals``: the unit's ``float`` is ``f32s`` (gradual
    underflow in software where the GPU would flush; woof/backend/metal/
    f32s.h) -- CUDA's --ftz=false.
    ``f64``: ``soft``, every ``double`` operation's word exact (strict), or
    ``df64``, the fast numerics' division, square root and libm through
    double-float (woof/backend/metal/f64_df64.h).  The translated body is
    the same text in both; the prelude and the vendored libm differ.
    """
    text = preprocess(source, defines, include_dirs)
    if compose(source, text, f64) is not source:
        text = preprocess(compose(source, text, f64), defines, include_dirs)
    cached, store = _translation_cache(text, unit, contract, retain_subnormals, source, defines, f64)
    if cached is not None:
        return cached
    result = _translate_preprocessed(text, source, unit=unit, contract=contract,
                                     retain_subnormals=retain_subnormals, defines=defines, f64=f64)
    store(result)
    return result


def _translate_preprocessed(text: str, source: str, *, unit: str, contract: bool,
                            retain_subnormals: bool, defines, f64: str) -> TranslatedUnit:
    text = rewrite_ptx_asm(text, unit)
    if retain_subnormals:
        text = explicit_float_to_integral(text)
    text = flatten_inheritance(text, unit)
    text = hoist_block_prototypes(text)
    tu = parse(text)
    result = _Rewriter(text, tu, unit=unit, retain_subnormals=retain_subnormals,
                       macros=_integer_macros(source, defines)).run()
    result.msl = (prelude(contract=contract, retain_subnormals=retain_subnormals,
                          ieee_helpers="__woof_ieee_" in result.msl, f64=f64)
                  + "\n// ---- translated unit: " + unit + "\n" + result.msl)
    result.key = hashlib.sha256(result.msl.encode()).hexdigest()
    return result
