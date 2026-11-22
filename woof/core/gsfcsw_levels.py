"""GSFC shortwave level tiers, importable without CuPy.

``kernels/gsfcsw.cu`` and ``gsfcsw_fast.cuh`` size every per-level array of
their device arena from ``GSFC_KMAX``.  The source's own ``#ifndef`` default
is 128, so every column of at most 128 levels compiles the unchanged text
and is word for word the pre-tier kernel; a 129 to 256 level column adds
``-D GSFC_KMAX=256`` through ``get_kernel_int_defines``.  The per-column
arena is affine in the interface count ``np = GSFC_KMAX + 1``; the two
coefficients are bound to the kernel's own ``GSFC_WORKSPACE_WORDS``
expression by ``tests/test_gsfcsw_resources.py``.
"""

from __future__ import annotations

#: The ``#ifndef`` default of ``GSFC_KMAX``.
SHIPPED_KMAX = 128
#: The deep tier, the dynamical core's level ceiling.
DEEP_KMAX = 256
#: ``GSFC_WORKSPACE_WORDS = WORDS_PER_INTERFACE * (GSFC_KMAX + 1) + WORDS_FIXED``
#: float words per column (32677 at the shipped tier, 62885 at the deep tier).
WORKSPACE_WORDS_PER_INTERFACE = 236
WORKSPACE_WORDS_FIXED = 2233
VERTICAL_LEVEL_BOUNDS = (2, DEEP_KMAX)


def kmax_tier(nz: int) -> int:
    """``GSFC_KMAX`` an ``nz``-level column compiles at."""
    nz = int(nz)
    if nz <= SHIPPED_KMAX:
        return SHIPPED_KMAX
    if nz <= DEEP_KMAX:
        return DEEP_KMAX
    raise ValueError(
        f"GSFC shortwave holds at most {DEEP_KMAX} layers per column; "
        f"this domain has {nz}")


def tier_workspace_words(kmax: int) -> int:
    """Float words of arena per column of a ``GSFC_KMAX = kmax`` build."""
    return (WORKSPACE_WORDS_PER_INTERFACE * (int(kmax) + 1)
            + WORKSPACE_WORDS_FIXED)


def workspace_words(nz: int) -> int:
    """Float words of arena per column at the tier of an ``nz``-level column."""
    return tier_workspace_words(kmax_tier(nz))
