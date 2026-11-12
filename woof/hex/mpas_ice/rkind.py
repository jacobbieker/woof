"""MALI's two precisions: RKIND 4 (MALI's default build) and RKIND 8.

Every MALI unit (``*_v842.cu``) is written once over ``rk``; defining
``MPAS_RKIND8`` makes it the double unit. The r8 unit is prepended with the
shared binary64 libm transcriptions (``glibc_flt64.cuh``: ``glibc_exp`` and
``glibc_pow``, Arm optimized-routines under MIT; ``cr_tanh_flt64.cuh``:
CORE-MATH's correctly rounded ``tanh`` under MIT, which glibc 2.44 carries),
the words glibc's ``exp``, ``pow`` and ``tanh`` give the oracle host. The
precision a call runs at is the dtype of the arrays it is given.

glibc's binary64 ``sin``, ``cos`` and ``atan`` are LGPL and not correctly
rounded, so the r8 units that call them (the sphere's velocity
reconstruction, the strain-rate angle, the SHMIP forcings, 'seroussi'
floating melt) refuse those configurations by name unless the opt-in
transcription is on (:mod:`woof.core.glibc239_gpu`: ``WOOF_GLIBC239_LIBM64=1``
in a source checkout); :func:`r8_trig_prelude` then prepends it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

PRECISIONS = {"r4": np.float32, "r8": np.float64}
#: The binary64 libm transcriptions every r8 unit is prepended with
R8_MATH_HEADERS = ("glibc_flt64.cuh", "cr_tanh_flt64.cuh")


def precision_of(array) -> str:
    """'r4' for float32 arrays, 'r8' for float64; anything else is refused."""
    for precision, dtype in PRECISIONS.items():
        if array.dtype == dtype:
            return precision
    raise ValueError(f"MALI arrays are float32 (RKIND 4) or float64 (RKIND 8), not {array.dtype}")


def require(precision: str) -> None:
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(PRECISIONS)}")


def r8_prelude() -> str:
    """The text an r8 unit starts with: the RKIND 8 switch and the binary64 libm."""
    import woof.core.kernels as kernels
    root = Path(kernels.__file__).parent
    libm = "\n".join((root / name).read_text(encoding="utf-8") for name in R8_MATH_HEADERS)
    return "#define MPAS_RKIND8 1\n" + libm + "\n"


def trig64() -> bool:
    """True when the opt-in glibc 2.39 binary64 sin/cos/atan is on (``WOOF_GLIBC239_LIBM64=1``)."""
    from woof.core import glibc239_gpu
    return glibc239_gpu.enabled()


def r8_trig_prelude() -> str:
    """The opt-in glibc 2.39 sin/cos/atan transcription behind ``WOOF_GLIBC239_TRIG64``, or nothing."""
    from woof.core import glibc239_gpu
    return glibc239_gpu.header_text() if glibc239_gpu.enabled() else ""


def trig_refusal(what: str) -> str:
    """Why an r8 configuration that calls glibc's binary64 sin, cos or atan is refused without the opt-in."""
    from woof.core import glibc239_gpu
    return (f"at RKIND 8 {what}, which are LGPL and not correctly rounded, so their words cannot be "
            "reproduced under the default licence" + glibc239_gpu.refusal_suffix())
