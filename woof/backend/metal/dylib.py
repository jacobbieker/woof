"""A translated unit as a Metal dynamic library and a thin kernel library.

Metal compiles a library from source on every process whose OS shader cache
has lost it, and only a *dynamic* library can be written to disk and loaded
back without compiling (``MTLDynamicLibrary.serializeToURL`` /
``newDynamicLibraryWithURL``).  A dynamic library cannot hold kernels, so
:func:`split` cuts a translated unit in two:

* the **body**: the whole unit, each ``[[kernel]] void K(...)`` turned into
  an ordinary function, plus an exported wrapper ``__woof_body_K`` whose
  parameters are opaque (``device uchar*``, ``constant uchar*`` and the
  thread-position builtins), so its name mangles the same in both halves;
* the **thin** library: ``<metal_stdlib>``, one ``extern`` declaration per
  wrapper and one kernel per wrapper with the original binding attributes,
  which only forwards its arguments.  It compiles in a fraction of a second.

``launch.cpp`` builds the body once as a dynamic library (the same compile a
monolithic library costs), serializes it under the library directory
(:func:`woof.backend.metal.ext.library_root`), and every later process loads
it in milliseconds and links the thin library against it.  The arithmetic is
the body's, compiled with the same options, so no word changes.

A unit :func:`split` cannot cut returns ``None`` and is built monolithically
as before: one that declares ``threadgroup`` memory (a non-kernel function
cannot declare it), or whose kernel parameters are not the translator's
buffer, scalar-block and builtin forms.

A dynamic library compiles every kernel of its unit at once, where pipelines
compile only the kernels a run launches.  A unit with many kernels pays for
kernels it never launches, so only a unit with at most :func:`max_kernels`
kernels is cut.  The strict NSSL-2 unit (32 kernels, one of which had not
compiled after an hour) took 37 to 49 minutes to fail as a dynamic library
on an M3 Max, where the oracle's run of it compiles in minutes.
"""

from __future__ import annotations

import os
import re

#: Environment override for :func:`max_kernels`.
MAX_KERNELS_ENV = "WOOF_METAL_DYLIB_MAX_KERNELS"


def max_kernels() -> int:
    """The most kernels a unit may have to be cut (``$WOOF_METAL_DYLIB_MAX_KERNELS``,
    default 4; 0 cuts no unit)."""
    try:
        return int(os.environ.get(MAX_KERNELS_ENV, "4"))
    except ValueError:
        return 4

#: Where the translator's prelude ends and the unit begins (translate()).
UNIT_MARKER = "\n// ---- translated unit: "

_KERNEL = re.compile(r"\[\[kernel\]\]\s+void\s+(\w+)\s*\(([^{;]*?)\)\s*\{", re.S)
_POINTER = re.compile(r"^(const\s+)?(device|constant)\s+(const\s+)?([\w:<>, ]+?)\s*\*\s*(\w+)\s*"
                      r"\[\[\s*buffer\(\d+\)\s*\]\]$")
_REFERENCE = re.compile(r"^(constant)\s+([\w:<>]+)\s*&\s*(\w+)\s*\[\[\s*buffer\(\d+\)\s*\]\]$")
_BUILTIN = re.compile(r"^(uint|uint2|uint3|ushort|ushort2|ushort3)\s+(\w+)\s*\[\[\s*\w+\s*\]\]$")


def _parameters(text: str) -> list[str] | None:
    """The parameter list split at top-level commas."""
    out, depth, start = [], 0, 0
    for i, c in enumerate(text):
        if c in "<([":
            depth += 1
        elif c in ">)]":
            depth -= 1
        elif c == "," and depth == 0:
            out.append(text[start:i].strip())
            start = i + 1
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def split(msl: str) -> tuple[str, str] | None:
    """``(body, thin)`` for a translated unit's MSL, or ``None``."""
    if UNIT_MARKER not in msl:
        return None
    unit = msl.split(UNIT_MARKER, 1)[1]
    if re.search(r"\bthreadgroup\b", unit):
        return None
    kernels = list(_KERNEL.finditer(msl))
    if not kernels:
        return None
    wrappers, thin = [], []
    edits = []
    for m in kernels:
        name, params = m.group(1), _parameters(m.group(2))
        if params is None:
            return None
        bare, opaque, forward, call, binds = [], [], [], [], []
        for p in params:
            p = " ".join(p.split())
            pointer, reference, builtin = _POINTER.match(p), _REFERENCE.match(p), _BUILTIN.match(p)
            attribute = p[p.index("[["):]
            if pointer:
                const = bool(pointer.group(1) or pointer.group(3))
                space, ctype, pname = pointer.group(2), pointer.group(4), pointer.group(5)
                qual = f"{'const ' if const else ''}{space}"
                bare.append(f"{qual} {ctype}* {pname}")
                opaque.append(f"{qual} uchar* {pname}")
                call.append(f"({qual} {ctype}*){pname}")
            elif reference:
                ctype, pname = reference.group(2), reference.group(3)
                bare.append(f"constant {ctype}& {pname}")
                opaque.append(f"constant uchar* {pname}")
                call.append(f"*(constant {ctype}*){pname}")
            elif builtin:
                ctype, pname = builtin.group(1), builtin.group(2)
                bare.append(f"{ctype} {pname}")
                opaque.append(f"{ctype} {pname}")
                call.append(pname)
            else:
                return None
            forward.append(pname)
            binds.append(f"{opaque[-1]} {attribute}")
        inner = f"__woof_kernel_{name}"
        edits.append((m.start(), m.end(), f"static void {inner}(" + ", ".join(bare) + ") {"))
        wrapper = f"void __woof_body_{name}(" + ", ".join(opaque) + ")"
        wrappers.append(wrapper + " { " + inner + "(" + ", ".join(call) + "); }")
        thin.append("extern " + wrapper + ";")
        thin.append(f"[[kernel]] void {name}(" + ", ".join(binds) + ") { __woof_body_" + name
                    + "(" + ", ".join(forward) + "); }")
    body = msl
    for start, end, text in reversed(edits):
        body = body[:start] + text + body[end:]
    body += "\n// ---- dynamic-library entries\n" + "\n".join(wrappers) + "\n"
    thin_msl = "#include <metal_stdlib>\nusing namespace metal;\n" + "\n".join(thin) + "\n"
    return body, thin_msl


__all__ = ["MAX_KERNELS_ENV", "UNIT_MARKER", "max_kernels", "split"]
