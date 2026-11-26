"""diff_opt 1 / km_opt 4 TL/AD (P3a item 5): WRF compute_diff_metrics +
cal_deform_and_div (the deformation) and smag2d_km at diff_opt 1 on
woof's strict wrf_smag_deform and wrf_diff_opt1_km4 + wrf_smag_km_bc, and
the horizontal_diffusion rows on wrf_diff_opt1_horizontal
(docs/dev/P3A-WRF-DRYCORE-ADJOINT.md section 5, item 5).

Same conventions as :mod:`woof.grad.wrfplus.acoustic`: dicts of arrays by
name, ``*_tl(state, pert_state, traj, pert, geom, nl=True)`` updates the
perturbation (and the base state when ``nl``), ``*_ad`` adds into the
inputs' adjoints.  The deformation and the coefficients overwrite their
outputs, so their AD consumes (zeroes) the outputs' adjoints; the rows
accumulate into a tendency, whose adjoint their AD leaves as it is.

Deformation: ``u`` and ``v`` are the (coupled-to-map-factor) winds the
primal reads, ``php`` the w-level perturbation geopotential; ``phb`` is
the 3-D base geopotential.  Coefficients: the horizontal eddy viscosity
``xkmh`` and diffusivity ``xkhh = xkmh / prandtl`` from the deformation,
with the 10 * mlen cap and the cold-zero outer row of a bounded axis.
Rows: the u, v and w (xkmh) and theta and scalar (xkhh) tendencies of
the field, the coefficient and the column mass ``mut`` (hybrid mass
c1 mut + c2; theta mixes its departure from the initial field).
Not covered: diff_opt 2's slope limiter, the TC horizontal length, a 1-D
base geopotential and m_opt.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .acoustic import THREADS, _blocks, _check, _dtype
from .acoustic import _kernel as _unit_kernel

#: woof's dycore._PRANDTL, as the primal passes it (the strict-primal test
#: checks the two agree).
PRANDTL = 1.0 / 3.0


def _kernel(name: str, dtype):
    return _unit_kernel("diffusion", name, dtype)


def _named(label, names, d):
    return [(label, n, d[n]) for n in names]


def _axes(geom) -> None:
    for name in ("nz", "ny", "nx"):
        if getattr(geom, name) < 1:
            raise ValueError(f"{name} must be >= 1, got {getattr(geom, name)}")
    for name in ("boundary_x", "boundary_y"):
        if getattr(geom, name) not in (0, 1):
            raise ValueError(f"{name} is 0 or 1, got {getattr(geom, name)}")
    # the zeroing launches take an int32 count (as the primal's grids)
    if (geom.nz + 1) * (geom.ny + 1) * (geom.nx + 1) >= 2 ** 31:
        raise ValueError("the grid is too large for the int32 launches")


# ------------------------------------------------------------- deformation

DF_STATE = ("defor11", "defor22", "defor12")
DF_ACTIVE = ("u", "v", "php")
DF_PASSIVE = ("phb", "msft", "msfu", "msfv", "fnm", "fnp", "dn", "dnw")


@dataclass(frozen=True)
class DFGeometry:
    """wrf_smag_deform's scalars: ``rdx = 1/dx`` and ``rdy = 1/dy`` as the
    primal rounds them, the bottom extrapolation weights cf1-cf3 and the
    bounded (open or specified) axes.  ``seam_cold`` = 1 is WRF's
    compute_diff_metrics at a periodic axis's seam: the slope at the top w
    level keeps its cold zero there (RunConfig.diff_metrics_seam_slope off,
    the primal's boundary flag 2, woof.core.dycore.smag_boundary_flags);
    0 takes the true slope."""
    nz: int
    ny: int
    nx: int
    rdx: float
    rdy: float
    cf1: float
    cf2: float
    cf3: float
    boundary_x: int = 0
    boundary_y: int = 0
    seam_cold: int = 0

    def args(self, dtype) -> tuple:
        i, r = np.int32, np.dtype(dtype).type
        # the primal's flags: 1 bounded, 0 periodic, 2 periodic with WRF's
        # cold seam slope at the top w level
        bx, by = ((1 if b else 2 * int(self.seam_cold))
                  for b in (self.boundary_x, self.boundary_y))
        return (r(self.rdx), r(self.rdy), r(self.cf1), r(self.cf2), r(self.cf3),
                i(bx), i(by), i(self.nz), i(self.ny), i(self.nx))

    def validate(self) -> None:
        _axes(self)
        if self.seam_cold not in (0, 1):
            raise ValueError(f"seam_cold is 0 or 1, got {self.seam_cold}")
        if self.nz < 3:
            raise ValueError(f"the deformation needs nz >= 3 (the bottom "
                             f"extrapolation reads levels 0-2), got {self.nz}")
        if not (np.isfinite(self.rdx) and np.isfinite(self.rdy)
                and self.rdx > 0 and self.rdy > 0):
            raise ValueError("rdx and rdy must be positive and finite")


def df_shapes(geom) -> dict:
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    mass, wlev = (nz, ny, nx), (nz + 1, ny, nx)
    return {"defor11": mass, "defor22": mass, "defor12": mass, "u": (nz, ny, nx + 1),
            "v": (nz, ny + 1, nx), "php": wlev, "phb": wlev, "msft": (ny, nx),
            "msfu": (ny, nx + 1), "msfv": (ny + 1, nx), "fnm": (nz,), "fnp": (nz,),
            "dn": (nz,), "dnw": (nz,)}


def df_tl(state, pert_state, traj, pert, geom: DFGeometry, *, nl=True):
    """WRFPLUS g_compute_diff_metrics + g_cal_deform_and_div on woof's
    strict wrf_smag_deform: defor11, defor22 and defor12 (overwritten) of
    u, v and php."""
    geom.validate()
    _check(geom, _named("state", DF_STATE, state) + _named("pert_state", DF_STATE, pert_state)
           + _named("traj", DF_ACTIVE + DF_PASSIVE, traj)
           + _named("pert", DF_ACTIVE, pert), df_shapes(geom), min_nz=3)
    arrs = (tuple(state[n] for n in DF_STATE) + tuple(pert_state[n] for n in DF_STATE)
            + tuple(a for n in DF_ACTIVE for a in (traj[n], pert[n]))
            + tuple(traj[n] for n in DF_PASSIVE))
    dtype = _dtype(arrs)
    _kernel("df_tl", dtype)(_blocks(geom.nz * geom.ny * geom.nx), (THREADS,),
                            arrs + (np.int32(bool(nl)),) + geom.args(dtype))


def df_ad(adj_state, traj, adj, geom: DFGeometry):
    """Adjoint of :func:`df_tl`: add into ``adj`` (u, v, php) and consume
    the defor11, defor22 and defor12 adjoints."""
    geom.validate()
    _check(geom, _named("adj", DF_STATE, adj_state)
           + _named("traj", DF_ACTIVE + DF_PASSIVE, traj)
           + _named("adj", DF_ACTIVE, adj), df_shapes(geom), min_nz=3)
    arrs = (tuple(adj_state[n] for n in DF_STATE) + tuple(traj[n] for n in DF_ACTIVE)
            + tuple(adj[n] for n in DF_ACTIVE) + tuple(traj[n] for n in DF_PASSIVE))
    dtype = _dtype(arrs)
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    n = nz * ny * (nx + 1) + nz * (ny + 1) * nx + (nz + 1) * ny * nx
    _kernel("df_ad", dtype)(_blocks(n), (THREADS,), arrs + geom.args(dtype))
    m = nz * ny * nx
    _kernel("df_zero3", dtype)(_blocks(m), (THREADS,),
                               tuple(adj_state[n] for n in DF_STATE) + (np.int32(m),))


# -------------------------------------------------- smag2d_km (diff_opt 1)

KM_STATE = ("xkmh", "xkhh")
KM_ACTIVE = DF_STATE
KM_PASSIVE = ("msft",)


@dataclass(frozen=True)
class KMGeometry:
    """wrf_diff_opt1_km4's scalars: the grid spacings, c_s, the Prandtl
    number and the bounded axes (whose outer row keeps zero)."""
    nz: int
    ny: int
    nx: int
    dx: float
    dy: float
    c_s: float = 0.25
    prandtl: float = PRANDTL
    boundary_x: int = 0
    boundary_y: int = 0

    def args(self, dtype) -> tuple:
        i, r = np.int32, np.dtype(dtype).type
        return (r(self.dx), r(self.dy), r(self.c_s), r(self.prandtl), i(self.boundary_x),
                i(self.boundary_y), i(self.nz), i(self.ny), i(self.nx))

    def validate(self) -> None:
        _axes(self)
        if not all(np.isfinite(x) and x > 0 for x in (self.dx, self.dy, self.prandtl)):
            raise ValueError("dx, dy and prandtl must be positive and finite")


def km_shapes(geom) -> dict:
    mass = (geom.nz, geom.ny, geom.nx)
    return {"xkmh": mass, "xkhh": mass, "defor11": mass, "defor22": mass, "defor12": mass,
            "msft": (geom.ny, geom.nx)}


def km_tl(state, pert_state, traj, pert, geom: KMGeometry, *, nl=True):
    """WRFPLUS G_SMAG2D_KM (diff_opt 1) on woof's strict wrf_diff_opt1_km4
    + wrf_smag_km_bc: xkmh and xkhh (overwritten) of the deformation."""
    geom.validate()
    _check(geom, _named("state", KM_STATE, state) + _named("pert_state", KM_STATE, pert_state)
           + _named("traj", KM_ACTIVE + KM_PASSIVE, traj)
           + _named("pert", KM_ACTIVE, pert), km_shapes(geom), min_nz=1)
    arrs = (tuple(state[n] for n in KM_STATE) + tuple(pert_state[n] for n in KM_STATE)
            + tuple(a for n in KM_ACTIVE for a in (traj[n], pert[n])) + (traj["msft"],))
    dtype = _dtype(arrs)
    _kernel("km_tl", dtype)(_blocks(geom.nz * geom.ny * geom.nx), (THREADS,),
                            arrs + (np.int32(bool(nl)),) + geom.args(dtype))


def km_ad(adj_state, traj, adj, geom: KMGeometry):
    """Adjoint of :func:`km_tl`: add into ``adj`` (defor11, defor22,
    defor12) and consume the xkmh and xkhh adjoints."""
    geom.validate()
    _check(geom, _named("adj", KM_STATE, adj_state)
           + _named("traj", KM_ACTIVE + KM_PASSIVE, traj)
           + _named("adj", KM_ACTIVE, adj), km_shapes(geom), min_nz=1)
    arrs = (tuple(adj_state[n] for n in KM_STATE) + tuple(traj[n] for n in KM_ACTIVE)
            + tuple(adj[n] for n in KM_ACTIVE) + (traj["msft"],))
    dtype = _dtype(arrs)
    m = geom.nz * geom.ny * geom.nx
    _kernel("km_ad", dtype)(_blocks(m), (THREADS,), arrs + geom.args(dtype))
    _kernel("km_zero2", dtype)(_blocks(m), (THREADS,),
                               tuple(adj_state[n] for n in KM_STATE) + (np.int32(m),))


# ------------------------------------- horizontal_diffusion (diff_opt 1)

#: The rows of wrf_diff_opt1_horizontal: stagger and whether the field is
#: mixed as a perturbation from a base (horizontal_diffusion_3dmp).
HD_ROWS = {"u": (1, 0), "v": (2, 0), "w": (3, 0), "t": (0, 1), "s": (0, 0)}
HD_STATE = ("tend",)
HD_ACTIVE = ("field", "xk", "mut")
HD_PASSIVE = ("c1", "c2", "msft", "msfu", "msfv")


@dataclass(frozen=True)
class HDGeometry:
    """wrf_diff_opt1_horizontal's scalars: ``rdx = 1/dx`` and ``rdy = 1/dy``
    as the primal rounds them, the row (``u``, ``v``, ``w``; ``t`` theta
    minus its initial field, horizontal_diffusion_3dmp; ``s`` a scalar)
    and the bounded (open or specified) axes."""
    nz: int
    ny: int
    nx: int
    rdx: float
    rdy: float
    row: str = "t"
    boundary_x: int = 0
    boundary_y: int = 0

    def __post_init__(self):
        if self.row not in HD_ROWS:   # before stag/perturb index HD_ROWS
            raise ValueError(f"row is one of {sorted(HD_ROWS)}, got {self.row!r}")

    @property
    def stag(self) -> int:
        return HD_ROWS[self.row][0]

    @property
    def perturb(self) -> int:
        return HD_ROWS[self.row][1]

    def args(self, dtype) -> tuple:
        i, r = np.int32, np.dtype(dtype).type
        return (r(self.rdx), r(self.rdy), i(self.stag), i(self.perturb), i(self.boundary_x),
                i(self.boundary_y), i(self.nz), i(self.ny), i(self.nx))

    def validate(self) -> None:
        _axes(self)   # the row is checked in __post_init__
        if not (np.isfinite(self.rdx) and np.isfinite(self.rdy)
                and self.rdx > 0 and self.rdy > 0):
            raise ValueError("rdx and rdy must be positive and finite")


def hd_passive(geom) -> tuple:
    return HD_PASSIVE + (("base",) if geom.perturb else ())


def hd_shapes(geom) -> dict:
    nz, ny, nx, st = geom.nz, geom.ny, geom.nx, geom.stag
    row = (nz + (st == 3), ny + (st == 2), nx + (st == 1))
    nc = nz + 1 if st == 3 else nz
    return {"tend": row, "field": row, "xk": (nz, ny, nx), "mut": (ny, nx), "c1": (nc,),
            "c2": (nc,), "base": (nz, ny, nx), "msft": (ny, nx), "msfu": (ny, nx + 1),
            "msfv": (ny + 1, nx)}


def _hd_passive_args(traj, geom):
    base = traj["base"] if geom.perturb else traj["field"]
    return (traj["c1"], traj["c2"], base, traj["msft"], traj["msfu"], traj["msfv"])


def hd_tl(state, pert_state, traj, pert, geom: HDGeometry, *, nl=True):
    """WRFPLUS G_HORIZONTAL_DIFFUSION / g_horizontal_diffusion_3dmp on
    woof's strict wrf_diff_opt1_horizontal: ``tend`` (accumulated) of the
    field, the coefficient ``xk`` and ``mut``."""
    geom.validate()
    _check(geom, _named("state", HD_STATE, state) + _named("pert_state", HD_STATE, pert_state)
           + _named("traj", HD_ACTIVE + hd_passive(geom), traj)
           + _named("pert", HD_ACTIVE, pert), hd_shapes(geom), min_nz=1)
    arrs = ((state["tend"], pert_state["tend"])
            + tuple(a for n in HD_ACTIVE for a in (traj[n], pert[n]))
            + _hd_passive_args(traj, geom))
    dtype = _dtype(arrs)
    n = int(np.prod(hd_shapes(geom)["tend"]))
    _kernel("hd_tl", dtype)(_blocks(n), (THREADS,), arrs + (np.int32(bool(nl)),)
                            + geom.args(dtype))


def hd_ad(adj_state, traj, adj, geom: HDGeometry):
    """Adjoint of :func:`hd_tl`: add into ``adj`` (field, xk, mut).  The
    tendency accumulates, so its adjoint is left as it is."""
    geom.validate()
    _check(geom, _named("adj", HD_STATE, adj_state)
           + _named("traj", HD_ACTIVE + hd_passive(geom), traj)
           + _named("adj", HD_ACTIVE, adj), hd_shapes(geom), min_nz=1)
    arrs = ((adj_state["tend"],) + tuple(traj[n] for n in HD_ACTIVE)
            + tuple(adj[n] for n in HD_ACTIVE) + _hd_passive_args(traj, geom))
    dtype = _dtype(arrs)
    sh = hd_shapes(geom)
    n = sum(int(np.prod(sh[k])) for k in HD_ACTIVE)
    _kernel("hd_ad", dtype)(_blocks(n), (THREADS,), arrs + geom.args(dtype))
