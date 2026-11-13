"""The opt-in glibc 2.39 binary64 ``sin``, ``cos`` and ``atan`` for GPU kernels.

glibc's binary64 ``sin``, ``cos`` and ``atan`` (the IBM Accurate
Mathematical Library) are not correctly rounded, so no permissively licensed
function gives their words, and their code is LGPL-2.1-or-later. Kernels that
must equal a gfortran oracle's binary64 ``sin``/``cos``/``atan`` word for
word (MALI at RKIND 8: the strain-rate angle ``eTheta``, the sphere's
velocity reconstruction, the SHMIP forcings, 'seroussi' floating melt)
therefore refuse those configurations by name unless this opt-in is on.

The transcription is ``tools/glibc239_libm_gpu/glibc239_trig_flt64.cuh``, in
the form of glibc's x86-64 FMA variants. It is not package data: the
recast-woof and metal-woof wheels and every default build contain none of it
(see NOTICE). Only a source checkout with ``WOOF_GLIBC239_LIBM64=1`` uses it;
``WOOF_GLIBC239_LIBM64_SOURCE`` may name another copy of the header.

It gives glibc 2.39's words (and the oracle host's glibc 2.44 words, which
come from the same sources) for every ``atan`` argument and every ``sin`` and
``cos`` argument below 105414350 in magnitude; above that glibc 2.44's build
of ``branred.c`` rounds some reductions differently
(``tools/glibc239_libm_gpu/verification.log``).
"""
from __future__ import annotations

import os
from pathlib import Path

VARIABLE = "WOOF_GLIBC239_LIBM64"
SOURCE_VARIABLE = "WOOF_GLIBC239_LIBM64_SOURCE"
#: The macro a kernel unit sees when the transcription is prepended
MACRO = "WOOF_GLIBC239_TRIG64"


def enabled() -> bool:
    """True when ``WOOF_GLIBC239_LIBM64=1`` (the opt-in), False when unset or 0."""
    value = os.environ.get(VARIABLE, "").strip().lower()
    if value in ("", "0", "off", "false"):
        return False
    if value in ("1", "on", "true", "glibc239"):
        return True
    raise ValueError(f"{VARIABLE}={value!r}; expected 1 (opt in) or unset")


def header_path() -> Path:
    return Path(os.environ.get(SOURCE_VARIABLE) or
                Path(__file__).resolve().parents[2] / "tools" / "glibc239_libm_gpu" / "glibc239_trig_flt64.cuh")


def header_text() -> str:
    """The transcription, behind its macro; raises when the checkout lacks it."""
    path = header_path()
    if not path.is_file():
        raise RuntimeError(
            f"{VARIABLE}=1 needs the opt-in glibc 2.39 transcription, {path}, which only a source "
            f"checkout carries (no wheel does; see NOTICE); {SOURCE_VARIABLE} names another copy")
    return f"#define {MACRO} 1\n" + path.read_text(encoding="utf-8") + "\n"


def refusal_suffix() -> str:
    """The sentence a refusal ends with: how to opt in."""
    return (f"; set {VARIABLE}=1 in a source checkout to use the opt-in LGPL transcription of glibc's "
            "(see NOTICE)")
