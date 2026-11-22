"""QNSE and BouLac per-thread column tiers, importable without CuPy.

``kernels/qnsepbl.cu`` (``QNSE_KMAX``) and ``kernels/boulac.cu``
(``BOULAC_KMAX``) hold the whole column in per-thread local arrays.  Each
compiles at one of two tiers: the source's own ``#ifndef`` default (128) for
every column of at most 128 levels -- the unchanged translation unit, so
those runs are word-for-word the pre-tier kernel -- and 256 above it, the
dynamical core's own level ceiling.  The launchers
(:mod:`woof.core.myjpbl`, :mod:`woof.core.boulac`) and the local-frame
pricing (:mod:`woof.core.preflight`) read this one ladder.
"""

from __future__ import annotations

#: The ``#ifndef`` default of ``QNSE_KMAX`` / ``BOULAC_KMAX``.
SHIPPED_KMAX = 128
#: The deep tier, the dynamical core's level ceiling.
DEEP_KMAX = 256


def kmax_tier(nz: int) -> int:
    """``QNSE_KMAX`` / ``BOULAC_KMAX`` an ``nz``-level column compiles at."""
    nz = int(nz)
    if nz <= SHIPPED_KMAX:
        return SHIPPED_KMAX
    if nz <= DEEP_KMAX:
        return DEEP_KMAX
    raise ValueError(
        f"QNSE/BouLac hold at most {DEEP_KMAX} levels per column; "
        f"this domain has {nz}")


def tiered_kernel(module: str, func: str, define: str, nz: int):
    """``module``'s ``func`` at the tier of an ``nz``-level column: the plain
    loader (unchanged source) up to :data:`SHIPPED_KMAX`, the ``define`` =
    :data:`DEEP_KMAX` build above it.  CuPy is imported only here."""
    from woof.core.kernels import get_kernel, get_kernel_int_defines

    tier = kmax_tier(nz)
    if tier == SHIPPED_KMAX:
        return get_kernel(module, func)
    return get_kernel_int_defines(module, func, ((define, tier),))
