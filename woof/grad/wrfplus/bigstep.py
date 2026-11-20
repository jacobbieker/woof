"""Big-step tendency TL/AD (P3a item 5): WRF horizontal_pressure_gradient
and pg_buoy_w on woof's strict slow_pgf and slow_buoyancy
(docs/dev/P3A-WRF-DRYCORE-ADJOINT.md section 5, item 5).

Same conventions as :mod:`woof.grad.wrfplus.acoustic`: dicts of arrays by
name, ``*_tl(state, pert_state, traj, pert, geom, nl=True)`` updates the
perturbation (and the base state when ``nl``), ``*_ad`` adds into the
inputs' adjoints.  Both operators add into the momentum tendencies, so
the tendencies' adjoints pass through unchanged.

``p`` is the perturbation pressure (the strict diagnosis's p'), ``php``
the w-level perturbation geopotential and ``mup`` the perturbation
column mass; ``pb``, ``phb`` and ``mub`` are the (3-D) base state.

Not covered: the moist cqu/cqv/cqw factors (calc_cq), the hydrostatic
option and a 1-D base state.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .acoustic import THREADS, _blocks, _check, _dtype, _real

KERNELS = Path(__file__).resolve().parent / "kernels"


@lru_cache(maxsize=None)
def _device_module(real: str, device: int):
    import cupy as cp
    source = (KERNELS / "bigstep.cu").read_text(encoding="utf-8")
    return cp.RawModule(code=source, options=("--fmad=false", f"-DWRFPLUS_REAL={real}"))


def _kernel(name: str, dtype):
    import cupy as cp
    return _device_module(_real(dtype), cp.cuda.Device().id).get_function(name)


def _named(label, names, d):
    return [(label, n, d[n]) for n in names]


def _axes(geom) -> None:
    for name in ("nz", "ny", "nx"):
        if getattr(geom, name) < 1:
            raise ValueError(f"{name} must be >= 1, got {getattr(geom, name)}")


# ------------------------------------------ horizontal_pressure_gradient

PGF_STATE = ("ru_t", "rv_t")
PGF_ACTIVE = ("p", "al", "alt", "php", "mup")
PGF_PASSIVE = ("pb", "phb", "mub", "c1h", "c2h", "rdnw", "fnm", "fnp")


@dataclass(frozen=True)
class PGFGeometry:
    """slow_pgf's scalars: ``rdx = 1/dx`` and ``rdy = 1/dy`` as the primal
    rounds them, the bottom (cf1-cf3) and lid (cfn, cfn1) extrapolation
    weights, ``top_lid`` and the non-periodic axes.  ``boundary_x`` and
    ``boundary_y`` drop both end faces of their axis, as woof's primal
    does; WRF's separate open_xs/open_xe (one open end) is not ported."""
    nz: int
    ny: int
    nx: int
    rdx: float
    rdy: float
    cf1: float
    cf2: float
    cf3: float
    cfn: float
    cfn1: float
    top_lid: int = 0
    boundary_x: int = 0
    boundary_y: int = 0

    def args(self, dtype) -> tuple:
        i, r = np.int32, np.dtype(dtype).type
        return (r(self.rdx), r(self.rdy), r(self.cf1), r(self.cf2), r(self.cf3),
                r(self.cfn), r(self.cfn1), i(self.top_lid), i(self.boundary_x),
                i(self.boundary_y), i(self.nz), i(self.ny), i(self.nx))

    def validate(self) -> None:
        _axes(self)
        if self.nz < 3:
            raise ValueError(f"the pressure gradient needs nz >= 3 (the bottom "
                             f"extrapolation reads levels 0-2), got {self.nz}")


def pgf_shapes(geom) -> dict:
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    mass, wlev = (nz, ny, nx), (nz + 1, ny, nx)
    return {"ru_t": (nz, ny, nx + 1), "rv_t": (nz, ny + 1, nx), "p": mass,
            "al": mass, "alt": mass, "php": wlev, "mup": (ny, nx), "pb": mass,
            "phb": wlev, "mub": (ny, nx), "c1h": (nz,), "c2h": (nz,),
            "rdnw": (nz,), "fnm": (nz,), "fnp": (nz,)}


def pgf_tl(state, pert_state, traj, pert, geom: PGFGeometry, *, nl=True):
    """WRFPLUS G_HORIZONTAL_PRESSURE_GRADIENT (dry, non-hydrostatic) on
    woof's strict slow_pgf: ru_t and rv_t lose the pressure-gradient
    term of p', al, alt, php and mup."""
    geom.validate()
    _check(geom, _named("state", PGF_STATE, state) + _named("pert", PGF_STATE, pert_state)
           + _named("traj", PGF_ACTIVE + PGF_PASSIVE, traj)
           + _named("pert", PGF_ACTIVE, pert), pgf_shapes(geom), min_nz=3)
    pairs = ()
    for n in PGF_ACTIVE:
        pairs += (traj[n], pert[n])
    arrs = ((state["ru_t"], pert_state["ru_t"], state["rv_t"], pert_state["rv_t"]) + pairs
            + tuple(traj[n] for n in PGF_PASSIVE))
    dtype = _dtype(arrs)
    n = geom.nz * (geom.ny + 1) * (geom.nx + 1)
    _kernel("pgf_tl", dtype)(_blocks(n), (THREADS,),
                             arrs + (np.int32(bool(nl)),) + geom.args(dtype))


def pgf_ad(adj_state, traj, adj, geom: PGFGeometry):
    """Adjoint of :func:`pgf_tl`: add into ``adj`` (p, al, alt, php, mup)
    from the ru_t and rv_t adjoints, which are unchanged."""
    geom.validate()
    _check(geom, _named("adj", PGF_STATE, adj_state)
           + _named("traj", PGF_ACTIVE + PGF_PASSIVE, traj)
           + _named("adj", PGF_ACTIVE, adj), pgf_shapes(geom), min_nz=3)
    import cupy as cp
    ins = tuple(traj[n] for n in PGF_ACTIVE)
    outs = tuple(adj[n] for n in PGF_ACTIVE)
    passive = tuple(traj[n] for n in PGF_PASSIVE)
    seeds = (adj_state["ru_t"], adj_state["rv_t"])
    dtype = _dtype(seeds + ins + outs + passive)
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    # The per-face adjoints aB, aL, aD, aV (see kernels/bigstep.cu);
    # pgf_ad_face writes every word.
    fu = cp.empty((4, nz, ny, nx + 1), dtype=dtype)
    fv = cp.empty((4, nz, ny + 1, nx), dtype=dtype)
    _kernel("pgf_ad_face", dtype)(_blocks(nz * (ny + 1) * (nx + 1)), (THREADS,),
                                  seeds + ins + (fu, fv) + passive + geom.args(dtype))
    a = (fu, fv) + ins + outs + passive + geom.args(dtype)
    st = ny * nx
    _kernel("pgf_ad_cell", dtype)(_blocks(nz * st), (THREADS,), a)
    _kernel("pgf_ad_php", dtype)(_blocks((nz + 1) * st), (THREADS,), a)
    _kernel("pgf_ad_mup", dtype)(_blocks(st), (THREADS,), a)


# ------------------------------------------------------------- pg_buoy_w

BUOY_STATE = ("rw_t",)
BUOY_ACTIVE = ("p", "mup")
BUOY_PASSIVE = ("rdn", "rdnw", "c1f", "msft")


@dataclass(frozen=True)
class BuoyGeometry:
    """slow_buoyancy's scalars (dry).  ``msft`` is read only when
    ``has_msf``, but the trajectory must still carry an (ny, nx) array."""
    nz: int
    ny: int
    nx: int
    has_msf: int = 0

    def args(self) -> tuple:
        i = np.int32
        return (i(self.has_msf), i(self.nz), i(self.ny), i(self.nx))

    def validate(self) -> None:
        _axes(self)


def buoy_shapes(geom) -> dict:
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    return {"rw_t": (nz + 1, ny, nx), "p": (nz, ny, nx), "mup": (ny, nx),
            "rdn": (nz,), "rdnw": (nz,), "c1f": (nz + 1,), "msft": (ny, nx)}


def buoy_tl(state, pert_state, traj, pert, geom: BuoyGeometry, *, nl=True):
    """WRFPLUS g_pg_buoy_w (dry: cqw = 0) on woof's strict slow_buoyancy:
    rw_t gains the vertical pressure gradient and buoyancy of p' and mup."""
    geom.validate()
    _check(geom, _named("state", BUOY_STATE, state) + _named("pert", BUOY_STATE, pert_state)
           + _named("traj", BUOY_ACTIVE + BUOY_PASSIVE, traj)
           + _named("pert", BUOY_ACTIVE, pert), buoy_shapes(geom), min_nz=1)
    arrs = ((state["rw_t"], pert_state["rw_t"], traj["p"], pert["p"], traj["mup"], pert["mup"])
            + tuple(traj[n] for n in BUOY_PASSIVE))
    dtype = _dtype(arrs)
    _kernel("buoy_tl", dtype)(_blocks(geom.nz * geom.ny * geom.nx), (THREADS,),
                              arrs + (np.int32(bool(nl)),) + geom.args())


def buoy_ad(adj_state, traj, adj, geom: BuoyGeometry):
    """Adjoint of :func:`buoy_tl`: add into ``adj`` (p, mup) from the
    rw_t adjoint, which is unchanged."""
    geom.validate()
    _check(geom, _named("adj", BUOY_STATE, adj_state)
           + _named("traj", BUOY_PASSIVE, traj)
           + _named("adj", BUOY_ACTIVE, adj), buoy_shapes(geom), min_nz=1)
    arrs = ((adj_state["rw_t"], adj["p"], adj["mup"])
            + tuple(traj[n] for n in BUOY_PASSIVE))
    dtype = _dtype(arrs)
    _kernel("buoy_ad", dtype)(_blocks(geom.ny * geom.nx), (THREADS,), arrs + geom.args())
