"""WRF v4.6.1 WSM6 coefficient initialization.

The values and operation order below follow ``mp_wsm6_init`` in
``phys/physics_mmm/mp_wsm6.F90``.  Keeping the derived constants on the
host makes the hail/graupel selection explicit and lets restart identity
bind the actual coefficient set instead of only the namelist integer.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math


@dataclass(frozen=True)
class WSM6RimedIceConstants:
    n0g: float
    deng: float
    avtg: float
    bvtg: float
    lamdagmax: float


@dataclass(frozen=True)
class WSM6Coefficients:
    rimed: WSM6RimedIceConstants
    xlv1: float
    qc0: float
    qck1: float
    pidnc: float
    pidn0r: float
    pidn0s: float
    pidn0g: float
    pvtr: float
    pvts: float
    pvtg: float
    pacrr: float
    pacrs: float
    pacrg: float
    pacrc: float
    precr1: float
    precr2: float
    precs1: float
    precs2: float
    precg1: float
    precg2: float
    roqimax: float
    rslopermax: float
    rslopesmax: float
    rslopegmax: float
    rsloperbmax: float
    rslopesbmax: float
    rslopegbmax: float
    rsloper2max: float
    rslopes2max: float
    rslopeg2max: float
    rsloper3max: float
    rslopes3max: float
    rslopeg3max: float


def _rgmma(x: float) -> float:
    """WSM6's Weierstrass-product Gamma implementation, verbatim."""
    if x == 1.0:
        return 0.0
    value = x * math.exp(0.577215664901532 * x)
    for index in range(1, 10001):
        y = float(index)
        value *= (1.0 + x / y) * math.exp(-x / y)
    return 1.0 / value


def rimed_ice_constants(hail_opt: int) -> WSM6RimedIceConstants:
    """Return WRF's five ``hail_opt``-selected WSM6 constants."""
    if int(hail_opt) == 0:
        return WSM6RimedIceConstants(4.0e6, 500.0, 330.0, 0.8, 6.0e4)
    if int(hail_opt) == 1:
        return WSM6RimedIceConstants(4.0e4, 700.0, 285.0, 0.8, 2.0e4)
    raise ValueError(f"WSM6 hail_opt must be 0 or 1, got {hail_opt}")


@lru_cache(maxsize=2)
def coefficients(hail_opt: int = 0) -> WSM6Coefficients:
    """Reproduce ``mp_wsm6_init`` for liquid density 1000 kg m-3."""
    r = rimed_ice_constants(hail_opt)
    pi = 4.0 * math.atan(1.0)
    den0, denr, dens = 1.28, 1000.0, 100.0
    cliq, cpv = 4190.0, 1846.4
    n0r, n0s = 8.0e6, 2.0e6
    avtr, bvtr, avts, bvts = 841.9, 0.8, 11.72, 0.41

    g3r, g4r = _rgmma(3.0 + bvtr), _rgmma(4.0 + bvtr)
    g5r2 = _rgmma(2.5 + 0.5 * bvtr)
    g3s, g4s = _rgmma(3.0 + bvts), _rgmma(4.0 + bvts)
    g5s2 = _rgmma(2.5 + 0.5 * bvts)
    g3g, g4g = _rgmma(3.0 + r.bvtg), _rgmma(4.0 + r.bvtg)
    g5g2 = _rgmma(2.5 + 0.5 * r.bvtg)

    rr, rs, rg = 1.0 / 8.0e4, 1.0 / 1.0e5, 1.0 / r.lamdagmax
    return WSM6Coefficients(
        rimed=r,
        xlv1=cliq - cpv,
        qc0=(4.0 / 3.0) * pi * denr * (0.8e-5 ** 3) * 3.0e8 / den0,
        qck1=(0.104 * 9.8 * 0.55 / ((3.0e8 * denr) ** (1.0 / 3.0))
              / 1.718e-5 * den0 ** (4.0 / 3.0)),
        pidnc=pi * denr / 6.0,
        pidn0r=pi * denr * n0r,
        pidn0s=pi * dens * n0s,
        pidn0g=pi * r.deng * r.n0g,
        pvtr=avtr * g4r / 6.0,
        pvts=avts * g4s / 6.0,
        pvtg=r.avtg * g4g / 6.0,
        pacrr=pi * n0r * avtr * g3r * 0.25,
        pacrs=pi * n0s * avts * g3s * 0.25,
        pacrg=pi * r.n0g * r.avtg * g3g * 0.25,
        pacrc=pi * n0s * avts * g3s * 0.25,
        precr1=2.0 * pi * n0r * 0.78,
        precr2=2.0 * pi * n0r * 0.31 * math.sqrt(avtr) * g5r2,
        precs1=4.0 * n0s * 0.65,
        precs2=4.0 * n0s * 0.44 * math.sqrt(avts) * g5s2,
        precg1=2.0 * pi * r.n0g * 0.78,
        precg2=2.0 * pi * r.n0g * 0.31 * math.sqrt(r.avtg) * g5g2,
        roqimax=2.08e22 * 500.0e-6 ** 8,
        rslopermax=rr, rslopesmax=rs, rslopegmax=rg,
        rsloperbmax=rr ** bvtr, rslopesbmax=rs ** bvts,
        rslopegbmax=rg ** r.bvtg,
        rsloper2max=rr * rr, rslopes2max=rs * rs,
        rslopeg2max=rg * rg,
        rsloper3max=rr ** 3, rslopes3max=rs ** 3,
        rslopeg3max=rg ** 3,
    )


#: ``kernels/wsm6.cu``'s column workspace.  One CUDA thread owns a whole
#: column, and until 2026-10-12 its ~30 column arrays were per-thread locals
#: sized by a compile-time ``WSM6_KMAX`` tier (64 or 80): the driver reserved
#: that frame -- 7,216 B at 64, 9,008 B at 80, +112 B per level -- times the
#: card's WHOLE resident-thread capacity at first launch, and at 256 levels
#: it would have been ~28.7 KB per thread (several GiB).  The arrays now live
#: in a global workspace ``woof/core/wsm6.py`` allocates per call, sized to
#: the columns IN FLIGHT and to the RUNTIME level count, so nothing is
#: compiled per level count and the tier ladder is retired.
#:
#: These live here, not in ``woof/core/wsm6.py``, because the adapter
#: imports CuPy at module scope and the memory estimator (and the hex
#: device admission) must price a configuration on a host with no device.
#:
#: ``WSM6WS_SLOTS`` and ``WSM6_BLOCK`` must match ``wsm6.cu``'s
#: ``WSM6WS_SLOTS`` / ``WSM6WS_LANES``; ``tests/test_wsm6_workspace.py``
#: re-derives them from the source and fails if either side moves alone.
WSM6WS_SLOTS = 30
#: Launch block and the workspace's lane interleave: one warp.
WSM6_BLOCK = 32
#: Blocks per SM the tile is sized for: 16 one-warp blocks is the per-SM
#: block ceiling on sm_86 and the pre-cut kernel's own residency there, and
#: the tile at that occupancy runs as fast as one launch over every column
#: (docs/kernel_local_memory_bounds.md section 10).  The device occupancy
#: query in the launcher only ever lowers it.
WSM6_TILE_BLOCKS_PER_SM = 16
#: The deepest column the kernel admits.  A REFUSAL CEILING only: it sizes
#: nothing, because the workspace extent is the runtime ``nz``.  256 matches
#: the other column microphysics (Thompson, Morrison, NSSL, Kessler).
WSM6_MAX_LEVELS = 256
#: Retained name for :data:`WSM6_MAX_LEVELS`.  It used to be the deepest
#: compiled tier (80); the hex column cap and its tests read it.
WSM6_DEEP_KMAX = WSM6_MAX_LEVELS
#: ``(minimum, maximum)`` levels WSM6 admits: 2 (the scheme needs a layer
#: above and below to sediment between) up to :data:`WSM6_MAX_LEVELS`.
WSM6_VERTICAL_LEVEL_BOUNDS = (2, WSM6_MAX_LEVELS)


def wsm6_workspace_extent(nz: int) -> int:
    """Per-slot level extent: ``nz + 1`` (the interface arrays index nz)."""
    nz = int(nz)
    minimum, maximum = WSM6_VERTICAL_LEVEL_BOUNDS
    if not minimum <= nz <= maximum:
        raise ValueError(
            f"WSM6 requires {minimum} <= nz <= {maximum}, got {nz}")
    return nz + 1


def wsm6_workspace_floats(nz: int, columns: int) -> int:
    """Workspace floats for ``columns`` columns in flight at this ``nz``.

    Rounded up to whole blocks: the kernel interleaves the workspace by lane
    within a block, the way CUDA lays local memory out across a warp, so the
    unit of allocation is one block's region, not one column's.
    """
    blocks = (int(columns) + WSM6_BLOCK - 1) // WSM6_BLOCK
    return blocks * WSM6WS_SLOTS * wsm6_workspace_extent(nz) * WSM6_BLOCK


def wsm6_workspace_bytes_per_column(nz: int) -> int:
    """FP32 workspace bytes one column in flight holds at this ``nz``."""
    return WSM6WS_SLOTS * wsm6_workspace_extent(nz) * 4


__all__ = ["WSM6Coefficients", "WSM6RimedIceConstants",
           "WSM6WS_SLOTS", "WSM6_BLOCK", "WSM6_DEEP_KMAX", "WSM6_MAX_LEVELS",
           "WSM6_TILE_BLOCKS_PER_SM", "WSM6_VERTICAL_LEVEL_BOUNDS",
           "coefficients", "rimed_ice_constants",
           "wsm6_workspace_bytes_per_column", "wsm6_workspace_extent",
           "wsm6_workspace_floats"]
