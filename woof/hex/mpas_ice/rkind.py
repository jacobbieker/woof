"""MALI's two precisions: RKIND 4 (MALI's default build) and RKIND 8.

Every MALI unit (``*_v842.cu``) is written once over ``rk``; defining
``MPAS_RKIND8`` makes it the double unit. The r8 unit is prepended with the
shared binary64 libm transcription (``glibc_flt64.cuh``: ``glibc_exp`` and
``glibc_pow``, Arm optimized-routines under MIT), the words glibc's ``exp``
and ``pow`` give the oracle host. The precision a call runs at is the dtype
of the arrays it is given.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

PRECISIONS = {"r4": np.float32, "r8": np.float64}


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
    libm = (Path(kernels.__file__).parent / "glibc_flt64.cuh").read_text(encoding="utf-8")
    return "#define MPAS_RKIND8 1\n" + libm + "\n"
