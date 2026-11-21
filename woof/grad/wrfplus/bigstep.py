"""Big-step tendency TL/AD (P3a item 5): WRF horizontal_pressure_gradient,
pg_buoy_w, rhs_ph, coriolis, curvature and w_damp on woof's strict
slow_pgf, slow_buoyancy, slow_geopotential, coriolis_curvature and w_damp
(docs/dev/P3A-WRF-DRYCORE-ADJOINT.md section 5, item 5).

Same conventions as :mod:`woof.grad.wrfplus.acoustic`: dicts of arrays by
name, ``*_tl(state, pert_state, traj, pert, geom, nl=True)`` updates the
perturbation (and the base state when ``nl``), ``*_ad`` adds into the
inputs' adjoints.  The pressure gradient and buoyancy add into the
momentum tendencies, so the tendencies' adjoints pass through unchanged;
rhs_ph adds too but overwrites the top w level, whose adjoint it zeroes;
coriolis/curvature add into all three momentum tendencies and w_damp
into the w tendency.

``p`` is the perturbation pressure (the strict diagnosis's p'), ``php``
the w-level perturbation geopotential and ``mup`` the perturbation
column mass; ``pb``, ``phb`` and ``mub`` are the (3-D) base state.

Moisture (P3a, DA item 4): ``calc_cq`` (cq_tl/cq_ad) forms cqu, cqv and
cqw from qv (n_mass 1) or qv, qc and qr (n_mass 3); the pressure
gradient takes cqu and cqv as active inputs with ``PGFGeometry.moist_cq``
and the buoyancy qv, qc and qr with ``BuoyGeometry.moist`` (the primal's
moist_mode 1).  Not covered: the ice species (moist_mode 2-3, n_mass 6-7),
the hydrostatic option and a 1-D base state.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from woof.core.constants import RERADIUS

from ._mode import tl_options
from .acoustic import THREADS, _blocks, _check, _dtype, _real

KERNELS = Path(__file__).resolve().parent / "kernels"


@lru_cache(maxsize=None)
def _device_module(real: str, device: int, build: tuple = ()):
    import cupy as cp
    source = (KERNELS / "bigstep.cu").read_text(encoding="utf-8")
    return cp.RawModule(code=source,
                        options=("--fmad=false", f"-DWRFPLUS_REAL={real}") + build)


def _kernel(name: str, dtype):
    import cupy as cp
    return _device_module(_real(dtype), cp.cuda.Device().id, tl_options()).get_function(name)


def _named(label, names, d):
    return [(label, n, d[n]) for n in names]


def _rdtype(arrs):
    """The real type of ``arrs``, skipping their int32 flags."""
    return _dtype(tuple(a for a in arrs if not isinstance(a, np.integer)))


def _axes(geom) -> None:
    for name in ("nz", "ny", "nx"):
        if getattr(geom, name) < 1:
            raise ValueError(f"{name} must be >= 1, got {getattr(geom, name)}")


# ------------------------------------------ horizontal_pressure_gradient

PGF_STATE = ("ru_t", "rv_t")
PGF_ACTIVE = ("p", "al", "alt", "php", "mup")
PGF_PASSIVE = ("pb", "phb", "mub", "c1h", "c2h", "rdnw", "fnm", "fnp")
#: the moist pressure gradient's actives: the dry ones and cqu, cqv
PGFM_ACTIVE = PGF_ACTIVE + ("cqu", "cqv")


def pgf_active(geom) -> tuple:
    return PGFM_ACTIVE if geom.moist_cq else PGF_ACTIVE


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
    moist_cq: int = 0

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
            "rdnw": (nz,), "fnm": (nz,), "fnp": (nz,), "cqu": (nz, ny, nx + 1),
            "cqv": (nz, ny + 1, nx)}


def pgf_tl(state, pert_state, traj, pert, geom: PGFGeometry, *, nl=True):
    """WRFPLUS G_HORIZONTAL_PRESSURE_GRADIENT (non-hydrostatic) on woof's
    strict slow_pgf: ru_t and rv_t lose the pressure-gradient term of p',
    al, alt, php and mup, times cqu/cqv with ``geom.moist_cq``."""
    geom.validate()
    act = pgf_active(geom)
    _check(geom, _named("state", PGF_STATE, state) + _named("pert", PGF_STATE, pert_state)
           + _named("traj", act + PGF_PASSIVE, traj)
           + _named("pert", act, pert), pgf_shapes(geom), min_nz=3)
    pairs = ()
    for n in PGF_ACTIVE:
        pairs += (traj[n], pert[n])
    if geom.moist_cq:
        cq = (traj["cqu"], pert["cqu"], traj["cqv"], pert["cqv"], np.int32(1))
    else:   # never read
        cq = (traj["p"],) * 4 + (np.int32(0),)
    arrs = ((state["ru_t"], pert_state["ru_t"], state["rv_t"], pert_state["rv_t"]) + pairs
            + cq + tuple(traj[n] for n in PGF_PASSIVE))
    dtype = _rdtype(arrs)
    n = geom.nz * (geom.ny + 1) * (geom.nx + 1)
    _kernel("pgf_tl", dtype)(_blocks(n), (THREADS,),
                             arrs + (np.int32(bool(nl)),) + geom.args(dtype))


def pgf_ad(adj_state, traj, adj, geom: PGFGeometry):
    """Adjoint of :func:`pgf_tl`: add into ``adj`` (p, al, alt, php, mup;
    cqu, cqv with ``geom.moist_cq``) from the ru_t and rv_t adjoints,
    which are unchanged."""
    geom.validate()
    act = pgf_active(geom)
    _check(geom, _named("adj", PGF_STATE, adj_state)
           + _named("traj", act + PGF_PASSIVE, traj)
           + _named("adj", act, adj), pgf_shapes(geom), min_nz=3)
    import cupy as cp
    ins = tuple(traj[n] for n in PGF_ACTIVE)
    outs = tuple(adj[n] for n in PGF_ACTIVE)
    passive = tuple(traj[n] for n in PGF_PASSIVE)
    seeds = (adj_state["ru_t"], adj_state["rv_t"])
    dtype = _dtype(seeds + ins + outs + passive)
    if geom.moist_cq:
        cq = (traj["cqu"], traj["cqv"], adj["cqu"], adj["cqv"], np.int32(1))
    else:   # never read or written
        cq = (traj["p"], traj["p"], outs[0], outs[0], np.int32(0))
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    # The per-face adjoints aB, aL, aD, aV (see kernels/bigstep.cu);
    # pgf_ad_face writes every word.
    fu = cp.empty((4, nz, ny, nx + 1), dtype=dtype)
    fv = cp.empty((4, nz, ny + 1, nx), dtype=dtype)
    _kernel("pgf_ad_face", dtype)(_blocks(nz * (ny + 1) * (nx + 1)), (THREADS,),
                                  seeds + ins + cq + (fu, fv) + passive + geom.args(dtype))
    a = (fu, fv) + ins + outs + passive + geom.args(dtype)
    st = ny * nx
    _kernel("pgf_ad_cell", dtype)(_blocks(nz * st), (THREADS,), a)
    _kernel("pgf_ad_php", dtype)(_blocks((nz + 1) * st), (THREADS,), a)
    _kernel("pgf_ad_mup", dtype)(_blocks(st), (THREADS,), a)


# ------------------------------------------------------------- pg_buoy_w

BUOY_STATE = ("rw_t",)
BUOY_ACTIVE = ("p", "mup")
BUOY_PASSIVE = ("rdn", "rdnw", "c1f", "msft")
#: the moist buoyancy's actives and the passives it adds (base_mass)
BUOYM_ACTIVE = BUOY_ACTIVE + ("qv", "qc", "qr")
BUOYM_PASSIVE = BUOY_PASSIVE + ("mub", "c2f")


def buoy_active(geom) -> tuple:
    return BUOYM_ACTIVE if geom.moist else BUOY_ACTIVE


def buoy_passive(geom) -> tuple:
    return BUOYM_PASSIVE if geom.moist else BUOY_PASSIVE


@dataclass(frozen=True)
class BuoyGeometry:
    """slow_buoyancy's scalars: dry (moist_mode 0) or ``moist`` (moist_mode
    1: qv, qc and qr).  ``msft`` is read only when ``has_msf``, but the
    trajectory must still carry an (ny, nx) array."""
    nz: int
    ny: int
    nx: int
    has_msf: int = 0
    moist: int = 0

    def args(self) -> tuple:
        i = np.int32
        return (i(self.has_msf), i(self.nz), i(self.ny), i(self.nx))

    def validate(self) -> None:
        _axes(self)
        if self.moist and self.nz < 2:
            raise ValueError(f"the moist buoyancy needs nz >= 2 (the top's cqw pairs "
                             f"levels nz-1 and nz-2), got {self.nz}")


def buoy_shapes(geom) -> dict:
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    mass = (nz, ny, nx)
    return {"rw_t": (nz + 1, ny, nx), "p": mass, "mup": (ny, nx),
            "rdn": (nz,), "rdnw": (nz,), "c1f": (nz + 1,), "msft": (ny, nx),
            "qv": mass, "qc": mass, "qr": mass, "mub": (ny, nx), "c2f": (nz + 1,)}


def _buoy_moist(geom, traj, pert):
    """The moist arguments (qv, qc, qr pairs, mub, c2f, moist): dummies
    that are never read on a dry geometry."""
    if not geom.moist:
        return (traj["p"],) * 8 + (np.int32(0),)
    out = ()
    for n in ("qv", "qc", "qr"):
        out += (traj[n], pert[n])
    return out + (traj["mub"], traj["c2f"], np.int32(1))


def buoy_tl(state, pert_state, traj, pert, geom: BuoyGeometry, *, nl=True):
    """WRFPLUS g_pg_buoy_w on woof's strict slow_buoyancy: rw_t gains the
    vertical pressure gradient and buoyancy of p' and mup (dry: cqw = 0),
    and of qv, qc and qr through WRF's cqw with ``geom.moist``."""
    geom.validate()
    act, pas = buoy_active(geom), buoy_passive(geom)
    _check(geom, _named("state", BUOY_STATE, state) + _named("pert", BUOY_STATE, pert_state)
           + _named("traj", act + pas, traj)
           + _named("pert", act, pert), buoy_shapes(geom), min_nz=1)
    arrs = ((state["rw_t"], pert_state["rw_t"], traj["p"], pert["p"], traj["mup"], pert["mup"])
            + _buoy_moist(geom, traj, pert) + tuple(traj[n] for n in BUOY_PASSIVE))
    dtype = _rdtype(arrs)
    _kernel("buoy_tl", dtype)(_blocks(geom.nz * geom.ny * geom.nx), (THREADS,),
                              arrs + (np.int32(bool(nl)),) + geom.args())


def buoy_ad(adj_state, traj, adj, geom: BuoyGeometry):
    """Adjoint of :func:`buoy_tl`: add into ``adj`` (p, mup; qv, qc, qr
    with ``geom.moist``) from the rw_t adjoint, which is unchanged."""
    geom.validate()
    act, pas = buoy_active(geom), buoy_passive(geom)
    _check(geom, _named("adj", BUOY_STATE, adj_state)
           + _named("traj", ("p",) + (act[2:] if geom.moist else ()) + pas, traj)
           + _named("adj", act, adj), buoy_shapes(geom), min_nz=1)
    if geom.moist:
        moist = (tuple(traj[n] for n in ("qv", "qc", "qr"))
                 + tuple(adj[n] for n in ("qv", "qc", "qr"))
                 + (traj["mub"], traj["c2f"], np.int32(1)))
    else:   # never read or written
        moist = (traj["p"],) * 3 + (adj["p"],) * 3 + (traj["p"], traj["p"], np.int32(0))
    arrs = ((adj_state["rw_t"], traj["p"], adj["p"], adj["mup"]) + moist
            + tuple(traj[n] for n in BUOY_PASSIVE))
    dtype = _rdtype(arrs)
    _kernel("buoy_ad", dtype)(_blocks(geom.ny * geom.nx), (THREADS,), arrs + geom.args())


# ---------------------------------------------------------------- calc_cq

CQ_STATE = ("cqu", "cqv", "cqw")
CQ_SPECIES = ("qv", "qc", "qr")


@dataclass(frozen=True)
class CQGeometry:
    """calc_cq's scalars: ``n_mass`` 1 sums qv, 3 sums qv, qc and qr (the
    primal's moist mass species; the ice species are not ported)."""
    nz: int
    ny: int
    nx: int
    n_mass: int = 1

    def validate(self) -> None:
        _axes(self)
        if self.n_mass not in (1, 3):
            raise ValueError(f"calc_cq's TL/AD covers n_mass 1 (qv) and 3 (qv, qc, qr), "
                             f"got {self.n_mass}")


def cq_species(geom) -> tuple:
    return CQ_SPECIES[:geom.n_mass]


def cq_shapes(geom) -> dict:
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    mass = (nz, ny, nx)
    return {"cqu": (nz, ny, nx + 1), "cqv": (nz, ny + 1, nx), "cqw": (nz + 1, ny, nx),
            "qv": mass, "qc": mass, "qr": mass}


def _cq_species(geom, d):
    """qv, qc, qr from ``d``; qv stands in for the species n_mass 1 never
    reads."""
    return tuple(d[n] if n in d else d["qv"] for n in CQ_SPECIES) \
        if geom.n_mass >= 3 else (d["qv"],) * 3


def cq_tl(state, pert_state, traj, pert, geom: CQGeometry, *, nl=True):
    """WRFPLUS g_calc_cq on woof's strict calc_cq: cqu, cqv and cqw (and
    their perturbations) are overwritten from the moist species."""
    geom.validate()
    sp = cq_species(geom)
    _check(geom, _named("state", CQ_STATE, state) + _named("pert", CQ_STATE, pert_state)
           + _named("traj", sp, traj) + _named("pert", sp, pert), cq_shapes(geom), min_nz=1)
    q, dq = _cq_species(geom, traj), _cq_species(geom, pert)
    arrs = (state["cqu"], pert_state["cqu"], state["cqv"], pert_state["cqv"],
            state["cqw"], pert_state["cqw"], q[0], dq[0], q[1], dq[1], q[2], dq[2])
    dtype = _dtype(arrs)
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    n = max(nz * ny * (nx + 1), nz * (ny + 1) * nx, (nz + 1) * ny * nx)
    i = np.int32
    _kernel("cq_tl", dtype)(_blocks(n), (THREADS,),
                            arrs + (i(bool(nl)), i(geom.n_mass), i(nz), i(ny), i(nx)))


def cq_ad(adj_state, traj, adj, geom: CQGeometry):
    """Adjoint of :func:`cq_tl`: add into ``adj`` (qv; qc, qr with n_mass
    3) from the cqu, cqv and cqw adjoints, then zero those (the primal
    overwrites cq)."""
    geom.validate()
    sp = cq_species(geom)
    _check(geom, _named("adj", CQ_STATE, adj_state) + _named("traj", sp, traj)
           + _named("adj", sp, adj), cq_shapes(geom), min_nz=1)
    q, aq = _cq_species(geom, traj), _cq_species(geom, adj)
    arrs = (adj_state["cqu"], adj_state["cqv"], adj_state["cqw"]) + q + aq
    dtype = _dtype(arrs)
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    i = np.int32
    _kernel("cq_ad", dtype)(_blocks(nz * ny * nx), (THREADS,),
                            arrs + (i(geom.n_mass), i(nz), i(ny), i(nx)))
    for n in CQ_STATE:
        adj_state[n][...] = 0


# ---------------------------------------------------------------- rhs_ph

GEO_STATE = ("rph_t",)
GEO_ACTIVE = ("ww", "w", "u", "v", "php", "mup")
GEO_PASSIVE = ("phb", "mub", "rdnw", "fnm", "fnp", "c1f", "c2f", "msft", "msfu", "msfv")


@dataclass(frozen=True)
class GeoGeometry:
    """slow_geopotential's scalars: ``quarter_rdx = 0.25/dx`` and
    ``quarter_rdy`` as the primal rounds them, the lid extrapolation
    weights cfn/cfn1, ``h_sca_adv_order`` (``order``), open axes and
    ``specified`` (which makes both axes boundaries, as woof's
    _boundary_x/_boundary_y do).  ``open_x``/``open_y`` drop both ends of
    their axis; WRF's separate open_xs/open_xe are not ported.  The map
    factors are read only when ``has_msf``, but the trajectory must still
    carry them."""
    nz: int
    ny: int
    nx: int
    quarter_rdx: float
    quarter_rdy: float
    cfn: float
    cfn1: float
    order: int = 5
    open_x: int = 0
    open_y: int = 0
    specified: int = 0
    has_msf: int = 0

    @property
    def boundary_x(self) -> int:
        return int(bool(self.open_x or self.specified))

    @property
    def boundary_y(self) -> int:
        return int(bool(self.open_y or self.specified))

    def args(self, dtype) -> tuple:
        i, r = np.int32, np.dtype(dtype).type
        return (r(self.quarter_rdx), r(self.quarter_rdy), r(self.cfn), r(self.cfn1),
                i(self.order), i(self.boundary_x), i(self.boundary_y), i(self.specified),
                i(self.has_msf), i(self.nz), i(self.ny), i(self.nx))

    def validate(self) -> None:
        _axes(self)
        if self.order not in (0, 2, 3, 4, 5, 6):
            raise ValueError(f"h_sca_adv_order must be 0, 2, 3, 4, 5 or 6, got {self.order}")
        need = 7 if self.order in (5, 6) else 5 if self.order in (3, 4) else 3
        if min(self.nx, self.ny) < need:
            raise ValueError(f"h_sca_adv_order={self.order} needs nx, ny >= {need}, "
                             f"got {self.nx} x {self.ny}")
        if self.nz < 2:
            raise ValueError(f"rhs_ph needs nz >= 2 (the lid flux reads u levels "
                             f"nz-1 and nz-2), got {self.nz}")


def geo_shapes(geom) -> dict:
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    wlev, col = (nz + 1, ny, nx), (ny, nx)
    return {"rph_t": wlev, "ww": wlev, "w": wlev, "u": (nz, ny, nx + 1),
            "v": (nz, ny + 1, nx), "php": wlev, "mup": col, "phb": wlev, "mub": col,
            "rdnw": (nz,), "fnm": (nz,), "fnp": (nz,), "c1f": (nz + 1,), "c2f": (nz + 1,),
            "msft": col, "msfu": (ny, nx + 1), "msfv": (ny + 1, nx)}


def geo_tl(state, pert_state, traj, pert, geom: GeoGeometry, *, nl=True):
    """WRFPLUS g_rhs_ph (non-hydrostatic) on woof's strict
    slow_geopotential: rph_t gains the geopotential advection by ww, u and
    v and the g w term; the top w level is overwritten."""
    geom.validate()
    _check(geom, _named("state", GEO_STATE, state) + _named("pert", GEO_STATE, pert_state)
           + _named("traj", GEO_ACTIVE + GEO_PASSIVE, traj)
           + _named("pert", GEO_ACTIVE, pert), geo_shapes(geom), min_nz=2)
    pairs = ()
    for n in GEO_ACTIVE:
        pairs += (traj[n], pert[n])
    arrs = ((state["rph_t"], pert_state["rph_t"]) + pairs
            + tuple(traj[n] for n in GEO_PASSIVE))
    dtype = _dtype(arrs)
    _kernel("geo_tl", dtype)(_blocks(geom.nz * geom.ny * geom.nx), (THREADS,),
                             arrs + (np.int32(bool(nl)),) + geom.args(dtype))


def geo_ad(adj_state, traj, adj, geom: GeoGeometry):
    """Adjoint of :func:`geo_tl`: add into ``adj`` (ww, w, u, v, php, mup)
    from the rph_t adjoint, whose top level is then zeroed (the primal
    overwrites it); the other levels pass through."""
    geom.validate()
    _check(geom, _named("adj", GEO_STATE, adj_state)
           + _named("traj", GEO_ACTIVE + GEO_PASSIVE, traj)
           + _named("adj", GEO_ACTIVE, adj), geo_shapes(geom), min_nz=2)
    import cupy as cp
    ins = tuple(traj[n] for n in GEO_ACTIVE)
    outs = tuple(adj[n] for n in GEO_ACTIVE)
    passive = tuple(traj[n] for n in GEO_PASSIVE)
    seed = adj_state["rph_t"]
    dtype = _dtype((seed,) + ins + outs + passive)
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    # Per-cell pass adjoints (see kernels/bigstep.cu); geo_ad_cell writes
    # every word.
    sy = cp.empty((4, nz, ny, nx), dtype=dtype)
    sx = cp.empty((6, nz, ny, nx), dtype=dtype)
    g = geom.args(dtype)
    _kernel("geo_ad_cell", dtype)(_blocks(nz * ny * nx), (THREADS,),
                                  (seed,) + ins + (sy, sx) + passive + g)
    a = (seed,) + ins + (sy, sx) + outs + passive + g
    _kernel("geo_ad_lev", dtype)(_blocks((nz + 1) * ny * nx), (THREADS,), a)
    _kernel("geo_ad_u", dtype)(_blocks(nz * ny * (nx + 1)), (THREADS,), a)
    _kernel("geo_ad_v", dtype)(_blocks(nz * (ny + 1) * nx), (THREADS,), a)
    _kernel("geo_ad_mup", dtype)(_blocks(ny * nx), (THREADS,), a)
    _kernel("geo_ad_top", dtype)(_blocks(ny * nx), (THREADS,),
                                 (seed, np.int32(nz), np.int32(ny), np.int32(nx)))


# ---------------------------------------------------- coriolis + curvature

CC_STATE = ("ru_t", "rv_t", "rw_t")
CC_ACTIVE = ("ru", "rv", "u", "v", "w", "mut")
CC_PASSIVE = ("msft", "msfu", "msfv", "f", "e", "sina", "cosa", "c1f", "c2f", "fnm", "fnp")


@dataclass(frozen=True)
class CCGeometry:
    """coriolis_curvature's scalars: ``rdx = 1/dx`` and ``rdy`` as the
    primal rounds them, and the bounded axes (open or specified), whose
    boundary-normal faces the primal leaves to the lateral boundary code.
    ``rerad`` is woof's RERADIUS (1/6370 km)."""
    nz: int
    ny: int
    nx: int
    rdx: float
    rdy: float
    boundary_x: int = 0
    boundary_y: int = 0
    rerad: float = RERADIUS

    def args(self, dtype) -> tuple:
        i, r = np.int32, np.dtype(dtype).type
        return (r(self.rdx), r(self.rdy), r(self.rerad), i(self.boundary_x),
                i(self.boundary_y), i(self.nz), i(self.ny), i(self.nx))

    def validate(self) -> None:
        _axes(self)
        if self.rerad != RERADIUS:
            raise ValueError(f"rerad {self.rerad!r} is not woof's RERADIUS {RERADIUS!r}")


def cc_shapes(geom) -> dict:
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    u, v, wlev, col = (nz, ny, nx + 1), (nz, ny + 1, nx), (nz + 1, ny, nx), (ny, nx)
    return {"ru_t": u, "rv_t": v, "rw_t": wlev, "ru": u, "rv": v, "u": u, "v": v,
            "w": wlev, "mut": col, "msft": col, "msfu": (ny, nx + 1), "msfv": (ny + 1, nx),
            "f": col, "e": col, "sina": col, "cosa": col, "c1f": (nz + 1,), "c2f": (nz + 1,),
            "fnm": (nz,), "fnp": (nz,)}


def cc_tl(state, pert_state, traj, pert, geom: CCGeometry, *, nl=True):
    """WRFPLUS G_CORIOLIS then G_CURVATURE on woof's strict
    coriolis_curvature: ru_t, rv_t, rw_t gain the Coriolis and curvature
    terms of ru, rv, u, v and rw = (c1f mut + c2f) w / msft."""
    geom.validate()
    _check(geom, _named("state", CC_STATE, state) + _named("pert", CC_STATE, pert_state)
           + _named("traj", CC_ACTIVE + CC_PASSIVE, traj)
           + _named("pert", CC_ACTIVE, pert), cc_shapes(geom), min_nz=1)
    pairs = ()
    for n in CC_STATE:
        pairs += (state[n], pert_state[n])
    for n in CC_ACTIVE:
        pairs += (traj[n], pert[n])
    arrs = pairs + tuple(traj[n] for n in CC_PASSIVE)
    dtype = _dtype(arrs)
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    _kernel("cc_tl", dtype)(_blocks(nz * (ny + 1) * (nx + 1)), (THREADS,),
                            arrs + (np.int32(bool(nl)),) + geom.args(dtype))


def cc_ad(adj_state, traj, adj, geom: CCGeometry):
    """Adjoint of :func:`cc_tl`: add into ``adj`` (ru, rv, u, v, w, mut)
    from the tendencies' adjoints, which pass through unchanged."""
    geom.validate()
    _check(geom, _named("adj", CC_STATE, adj_state)
           + _named("traj", CC_ACTIVE + CC_PASSIVE, traj)
           + _named("adj", CC_ACTIVE, adj), cc_shapes(geom), min_nz=1)
    import cupy as cp
    passive = tuple(traj[n] for n in CC_PASSIVE)
    seeds = tuple(adj_state[n] for n in CC_STATE)
    ins = tuple(traj[n] for n in CC_ACTIVE)
    outs = tuple(adj[n] for n in CC_ACTIVE)
    dtype = _dtype(seeds + ins + outs + passive)
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    # Per-face adjoints (see kernels/bigstep.cu); cc_ad_face writes every word.
    su = cp.empty((4, nz, ny, nx + 1), dtype=dtype)
    sv = cp.empty((4, nz, ny + 1, nx), dtype=dtype)
    sw = cp.empty((4, nz, ny, nx), dtype=dtype)
    sg = cp.empty((nz, ny, nx), dtype=dtype)
    srw = cp.empty((nz + 1, ny, nx), dtype=dtype)
    g = geom.args(dtype)
    _kernel("cc_ad_face", dtype)(_blocks(nz * (ny + 1) * (nx + 1)), (THREADS,),
                                 seeds + ins + (su, sv, sw) + passive + g)
    _kernel("cc_ad_vxgm", dtype)(_blocks(nz * ny * nx), (THREADS,),
                                 (su, sv, sg, np.int32(nz), np.int32(ny), np.int32(nx)))
    a = (traj["w"], traj["mut"], su, sv, sw, sg, srw) + outs + passive + g
    _kernel("cc_ad_u", dtype)(_blocks(nz * ny * (nx + 1)), (THREADS,), a)
    _kernel("cc_ad_v", dtype)(_blocks(nz * (ny + 1) * nx), (THREADS,), a)
    _kernel("cc_ad_w", dtype)(_blocks((nz + 1) * ny * nx), (THREADS,), a)
    _kernel("cc_ad_mut", dtype)(_blocks(ny * nx), (THREADS,), a)


# ---------------------------------------------------------------- w_damp

WD_STATE = ("rw_t",)
WD_ACTIVE = ("ww", "mup")
WD_PASSIVE = ("w", "mub", "c1f", "c2f", "rdnw")

def _w_damp_alpha() -> float:
    """woof's W_DAMP_ALPHA, read from the primal's source (openbc.cu), so
    the TL/AD cannot drift from the constant the forward uses."""
    import re
    src = (Path(__file__).resolve().parents[2] / "core/kernels/openbc.cu").read_text()
    m = re.search(r"^#define\s+W_DAMP_ALPHA\s+([0-9.eE+-]+)f?\s*$", src, re.M)
    if m is None:
        raise RuntimeError("openbc.cu no longer defines W_DAMP_ALPHA")
    return float(m.group(1))


#: WRF share/module_model_constants.F w_alpha (woof openbc.cu W_DAMP_ALPHA).
W_ALPHA = _w_damp_alpha()


@dataclass(frozen=True)
class WDampGeometry:
    """w_damp's scalars: the time step, the onset ``w_damp_on``
    (dycore.w_damp_onset: w_beta = 1, or w_crit_cfl under
    zadvect_implicit) and ``w_crit_cfl``, from which the excess is
    measured."""
    nz: int
    ny: int
    nx: int
    dt: float
    w_damp_on: float = 1.0
    w_crit_cfl: float = 1.0

    def args(self, dtype) -> tuple:
        i, r = np.int32, np.dtype(dtype).type
        return (r(self.dt), r(self.w_damp_on), r(self.w_crit_cfl), r(W_ALPHA),
                i(self.nz), i(self.ny), i(self.nx))

    def validate(self) -> None:
        _axes(self)


def wd_shapes(geom) -> dict:
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    wlev, col = (nz + 1, ny, nx), (ny, nx)
    return {"rw_t": wlev, "ww": wlev, "w": wlev, "mup": col, "mub": col,
            "c1f": (nz + 1,), "c2f": (nz + 1,), "rdnw": (nz,)}


def wd_tl(state, pert_state, traj, pert, geom: WDampGeometry, *, nl=True):
    """WRFPLUS g_w_damp (w_damping = 1) on woof's strict w_damp: where the
    base state's vertical Courant number exceeds the onset, rw_t loses
    sign(w) w_alpha (vert_cfl - w_crit_cfl) times the column mass."""
    geom.validate()
    _check(geom, _named("state", WD_STATE, state) + _named("pert", WD_STATE, pert_state)
           + _named("traj", WD_ACTIVE + WD_PASSIVE, traj)
           + _named("pert", WD_ACTIVE, pert), wd_shapes(geom), min_nz=1)
    arrs = ((state["rw_t"], pert_state["rw_t"], traj["ww"], pert["ww"], traj["mup"],
             pert["mup"]) + tuple(traj[n] for n in WD_PASSIVE))
    dtype = _dtype(arrs)
    _kernel("wd_tl", dtype)(_blocks(max(geom.nz - 1, 1) * geom.ny * geom.nx), (THREADS,),
                            arrs + (np.int32(bool(nl)),) + geom.args(dtype))


def wd_ad(adj_state, traj, adj, geom: WDampGeometry):
    """Adjoint of :func:`wd_tl`: add into ``adj`` (ww, mup) from the rw_t
    adjoint, which is unchanged."""
    geom.validate()
    _check(geom, _named("adj", WD_STATE, adj_state)
           + _named("traj", WD_ACTIVE + WD_PASSIVE, traj)
           + _named("adj", WD_ACTIVE, adj), wd_shapes(geom), min_nz=1)
    arrs = ((adj_state["rw_t"], traj["ww"], traj["mup"], adj["ww"], adj["mup"])
            + tuple(traj[n] for n in WD_PASSIVE))
    dtype = _dtype(arrs)
    _kernel("wd_ad", dtype)(_blocks(geom.ny * geom.nx), (THREADS,), arrs + geom.args(dtype))
