"""Lateral-boundary tendency TL/AD (P3a item 7): WRF relax_bdy_dry and
spec_bdy_dry on woof's strict state_relaxation_multi rows
(docs/dev/P3A-WRF-DRYCORE-ADJOINT.md section 5, item 7).

One call is one row of the primal's launch: a field kind (u, v, theta,
phi, mu, w or qv) and the row's flags (:class:`LBCGeometry`).  The strict
dycore runs these rows per RK stage: on stage 1 the held u, v, theta (and
w) relaxation (``held_compute`` + ``held_store``, no target), then the
specified-zone tendency of u, v, theta (w) (``target``), phi's held row
(stage 1 ``held_compute`` + ``held_store`` + ``divide_msf``, later stages
``held_read``) and mu's direct relaxation (``apply_relax`` on stage 1).

The state is the row's tendency ``tend`` and its held buffer ``held``;
the active inputs are the coupled field's source ``src`` (u, v, theta',
ph', w; unused for mu), mu' and the eight side tables (``west`` and its
tendency ``west_t``, ...), so the LBC perturbation is an AD output.
``*_tl`` updates the perturbation (and the base state when ``nl``);
``*_ad`` maps the tendency and held adjoints back in place (the specified
cells' tendency adjoint is consumed; a stored held row's adjoint is
consumed) and adds into the inputs' adjoints.

qv's row (8k) is the moist array's rk_step-1 capture: a direct row
(``apply_relax``) into qv's held tendency, from the time-t qv and mu'.
Its spec_bdy_final is :func:`lbf_tl` with kind qv.  Hydrometeors without
a table take WRF's flow_dep_bdy (:func:`flow_tl`).

Whole domains only (no streamed tile seams).  Not covered: scalar-array
rows, the nested sources, the non-strict path.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ._mode import tl_options
from .acoustic import THREADS, _blocks, _check, _dtype, _real

KERNELS = Path(__file__).resolve().parent / "kernels"

LB_STATE = ("tend", "held")
LB_TABLES = ("west", "west_t", "east", "east_t", "south", "south_t", "north", "north_t")
LB_ACTIVE = ("src", "mup") + LB_TABLES
LB_PASSIVE = ("mub", "c1h", "c2h", "c1f", "c2f", "msft", "msfu", "msfv", "fcx", "gcx")
#: The primal's kind codes (kernels/lbc_state.cu).  qv (5) is a mass-grid
#: row coupled as theta is under the strict build, ``(c1h mu + c2h) q``,
#: with the time-t qv and mu' as its source and mass (moist.py's
#: rk_step-1 capture) and qv's own tables.
LB_KINDS = {"u": 0, "v": 1, "theta": 2, "phi": 3, "mu": 4, "qv": 5, "w": 6}


@lru_cache(maxsize=None)
def _device_module(real: str, device: int, build: tuple = ()):
    import cupy as cp
    source = (KERNELS / "lbc.cu").read_text(encoding="utf-8")
    return cp.RawModule(code=source,
                        options=("--fmad=false", f"-DWRFPLUS_REAL={real}") + build)


def _kernel(name: str, dtype):
    import cupy as cp
    return _device_module(_real(dtype), cp.cuda.Device().id, tl_options()).get_function(name)


def _rings(ny: int, nx: int, width: int) -> int:
    return sum(2 * (nx - 2 * d) + 2 * max(ny - 2 * d - 2, 0) for d in range(width))


def _cells(geom) -> int:
    """Threads of the point kernels: the frame rings of every level, and
    the interior too for a row with a held pass (lb_cell's order)."""
    fz, fy, fx = geom.field
    if geom.held_compute or geom.held_read:
        return fz * fy * fx
    return fz * _rings(fy, fx, geom.aw)


@dataclass(frozen=True)
class LBCGeometry:
    """One relaxation row: mass-grid dims, the field kind, the zones
    (``spec_zone`` specified rows, relaxation rows ``spec_zone`` ..
    ``relax_zone - 1``), the table width, WRF's dtbc (time since the
    table's start) and the row's flags (see the module docstring).
    ``has_msf``: u is divided by msfu, v multiplied by msfv (the strict
    path's stored inverse)."""
    nz: int
    ny: int
    nx: int
    kind: str
    width: int = 5
    spec_zone: int = 1
    relax_zone: int = 4
    dtbc: float = 0.0
    held_compute: int = 0
    held_store: int = 0
    held_read: int = 0
    target: int = 1
    apply_relax: int = 0
    divide_msf: int = 0
    has_msf: int = 0

    @property
    def field(self) -> tuple[int, int, int]:
        nz, ny, nx = self.nz, self.ny, self.nx
        return {"u": (nz, ny, nx + 1), "v": (nz, ny + 1, nx), "theta": (nz, ny, nx),
                "qv": (nz, ny, nx), "phi": (nz + 1, ny, nx), "w": (nz + 1, ny, nx),
                "mu": (1, ny, nx)}[self.kind]

    @property
    def aw(self) -> int:
        return max(self.spec_zone, self.relax_zone)

    @property
    def w2(self) -> int:
        return self.relax_zone + 1

    def validate(self) -> None:
        for name in ("nz", "ny", "nx"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.kind not in LB_KINDS:
            raise ValueError(f"kind {self.kind!r} is not a ported row")
        if not 1 <= self.spec_zone <= self.relax_zone:
            # the relaxation stencil reads the row outside the zone
            raise ValueError("need 1 <= spec_zone <= relax_zone")
        if self.width < self.spec_zone + self.relax_zone or self.width < self.relax_zone + 1:
            raise ValueError("boundary width is smaller than spec_zone + relax_zone")
        _, fy, fx = self.field
        if min(fy, fx) < 2 * self.w2 + 1:
            raise ValueError("domain too small for the relaxation frames")
        if self.held_compute and self.held_read:
            raise ValueError("a row computes or reads its held buffer, not both")
        if self.held_store and not self.held_compute:
            raise ValueError("held_store needs held_compute")
        if self.apply_relax and (self.held_compute or self.held_read):
            raise ValueError("apply_relax is the direct form (no held buffer)")
        if self.divide_msf and not self.held_compute:
            raise ValueError("divide_msf belongs to the held pass")
        if self.divide_msf and self.kind in ("u", "v"):
            # the primal indexes msft on the row's own grid (phi's use)
            raise ValueError("divide_msf is for mass-grid rows")
        if not (self.target or self.held_compute):
            raise ValueError("the row writes nothing")

    def args(self, dtype) -> tuple:
        i, r = np.int32, np.dtype(dtype).type
        nz, ny, nx = self.field
        return (r(self.dtbc), i(LB_KINDS[self.kind]), i(self.width), i(self.spec_zone),
                i(self.relax_zone), i(self.aw), i(self.w2), i(self.held_compute),
                i(self.held_store), i(self.held_read), i(self.target), i(self.apply_relax),
                i(self.divide_msf), i(self.has_msf), i(nz), i(ny), i(nx))


def lb_shapes(geom: LBCGeometry) -> dict:
    fz, fy, fx = geom.field
    nz, ny, nx, w = geom.nz, geom.ny, geom.nx, geom.width
    sh = {"tend": geom.field, "held": geom.field, "src": geom.field, "mup": (ny, nx),
          "mub": (ny, nx), "c1h": (nz,), "c2h": (nz,), "c1f": (nz + 1,), "c2f": (nz + 1,),
          "msft": (ny, nx), "msfu": (ny, nx + 1), "msfv": (ny + 1, nx), "fcx": (w,),
          "gcx": (w,)}
    for side in ("west", "east"):
        sh[side] = sh[side + "_t"] = (fz, fy, w)
    for side in ("south", "north"):
        sh[side] = sh[side + "_t"] = (fz, w, fx)
    return sh


def lbc_tl(state, pert_state, traj, pert, geom: LBCGeometry, *, nl=True):
    """WRFPLUS g_relax_bdytend / g_spec_bdytend on one strict
    state_relaxation_multi row."""
    geom.validate()
    _check(geom, _named("state", LB_STATE, state) + _named("pert", LB_STATE, pert_state)
           + _named("traj", LB_ACTIVE + LB_PASSIVE, traj)
           + _named("pert", LB_ACTIVE, pert), lb_shapes(geom), min_nz=1)
    arrs = ((state["tend"], pert_state["tend"], state["held"], pert_state["held"],
             traj["src"], pert["src"], traj["mup"], pert["mup"])
            + tuple(traj[n] for n in LB_TABLES) + tuple(pert[n] for n in LB_TABLES)
            + tuple(traj[n] for n in LB_PASSIVE))
    dtype = _dtype(arrs)
    _kernel("lbc_tl", dtype)(_blocks(_cells(geom)), (THREADS,),
                             arrs + (np.int32(bool(nl)),) + geom.args(dtype))


def lbc_ad(adj_state, traj, adj, geom: LBCGeometry):
    """Adjoint of :func:`lbc_tl`: map ``adj_state`` (tend, held) back in
    place and add into ``adj`` (src, mu', the side tables)."""
    import cupy as cp
    geom.validate()
    _check(geom, _named("adj", LB_STATE, adj_state)
           + _named("traj", LB_ACTIVE + LB_PASSIVE, traj)
           + _named("adj", LB_ACTIVE, adj), lb_shapes(geom), min_nz=1)
    passive = tuple(traj[n] for n in LB_PASSIVE)
    dtype = _dtype((adj_state["tend"], adj_state["held"], traj["src"]) + passive
                   + tuple(adj[n] for n in LB_ACTIVE))
    fz, fy, fx = geom.field
    nf, n2 = _rings(fy, fx, geom.aw), _rings(fy, fx, geom.w2)
    # Rows that do not relax (specified target, held read) leave src and
    # mu' alone: no coupled-field or mass gather.
    relaxes = bool(geom.held_compute or geom.apply_relax)
    sf = cp.empty((6, fz, nf), dtype=dtype)
    sm = cp.empty((fz, n2), dtype=dtype) if relaxes else None
    g = geom.args(dtype)
    _kernel("lbc_ad_point", dtype)(_blocks(_cells(geom)), (THREADS,),
                                   (adj_state["tend"], adj_state["held"], sf) + passive
                                   + (np.int32(nf),) + g)
    if relaxes:
        _kernel("lbc_ad_coupled", dtype)(_blocks(fz * n2), (THREADS,),
                                         (traj["src"], traj["mup"], adj["src"], sf, sm) + passive
                                         + (np.int32(nf), np.int32(n2)) + g)
    for side, name in enumerate(("south", "north", "west", "east")):
        n = int(np.prod(adj[name].shape))
        _kernel("lbc_ad_table", dtype)(_blocks(n), (THREADS,),
                                       (adj[name], adj[name + "_t"], sf, np.int32(side),
                                        np.int32(nf)) + g)
    if relaxes:
        _kernel("lbc_ad_mup", dtype)(_blocks(geom.ny * geom.nx), (THREADS,),
                                     (adj["mup"], sm, np.int32(n2)) + g)


def _named(label, names, d):
    return [(label, n, d[n]) for n in names]


# ---------------------------------------------------------------- item 7b
#: spec_bdy_final: the field it overwrites on the specified rows, the
#: acoustic carrier muts and the eight side tables (active), and the
#: coupling constants (passive).
LF_STATE = ("target",)
LF_ACTIVE = ("muts",) + LB_TABLES
LF_PASSIVE = ("c1h", "c2h", "c1f", "c2f", "msft", "msfu", "msfv")
LF_KINDS = ("u", "v", "theta", "phi", "w", "mu", "qv")


@dataclass(frozen=True)
class FinalGeometry:
    """One field of WRF spec_bdy_final as woof's strict finalize_state_field
    (GPUWM_WRF_EXACT: the specified rows only, coupled with the acoustic
    muts) or, for mu, install_mu_boundary.  ``has_msf``: u, v and w take
    msfu, msfv and msft."""
    nz: int
    ny: int
    nx: int
    kind: str
    width: int = 5
    spec_zone: int = 1
    dtbc: float = 0.0
    has_msf: int = 0

    @property
    def field(self) -> tuple[int, int, int]:
        return LBCGeometry(self.nz, self.ny, self.nx, self.kind).field

    def validate(self) -> None:
        for name in ("nz", "ny", "nx"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.kind not in LF_KINDS:
            raise ValueError(f"kind {self.kind!r} is not a ported field")
        if not 1 <= self.spec_zone <= self.width:
            raise ValueError("need 1 <= spec_zone <= width")
        _, fy, fx = self.field
        if min(fy, fx) < 2 * self.spec_zone + 1:
            raise ValueError("domain too small for the specified frame")

    def args(self, dtype) -> tuple:
        i, r = np.int32, np.dtype(dtype).type
        nz, ny, nx = self.field
        return (r(self.dtbc), i(LB_KINDS[self.kind]), i(self.width), i(self.spec_zone),
                i(self.has_msf), i(nz), i(ny), i(nx))


def lf_shapes(geom: FinalGeometry) -> dict:
    fz, fy, fx = geom.field
    nz, ny, nx, w = geom.nz, geom.ny, geom.nx, geom.width
    sh = {"target": geom.field, "muts": (ny, nx), "c1h": (nz,), "c2h": (nz,),
          "c1f": (nz + 1,), "c2f": (nz + 1,), "msft": (ny, nx), "msfu": (ny, nx + 1),
          "msfv": (ny + 1, nx)}
    for side in ("west", "east"):
        sh[side] = sh[side + "_t"] = (fz, fy, w)
    for side in ("south", "north"):
        sh[side] = sh[side + "_t"] = (fz, w, fx)
    return sh


def _lf_cells(geom) -> int:
    fz, fy, fx = geom.field
    return fz * _rings(fy, fx, geom.spec_zone)


def lbf_tl(state, pert_state, traj, pert, geom: FinalGeometry, *, nl=True):
    """WRFPLUS g_spec_bdy_final on one field: the specified rows of
    ``target`` take the coupled table value."""
    geom.validate()
    _check(geom, _named("state", LF_STATE, state) + _named("pert", LF_STATE, pert_state)
           + _named("traj", LF_ACTIVE + LF_PASSIVE, traj)
           + _named("pert", LF_ACTIVE, pert), lf_shapes(geom), min_nz=1)
    arrs = ((state["target"], pert_state["target"])
            + tuple(traj[n] for n in LB_TABLES) + tuple(pert[n] for n in LB_TABLES)
            + (traj["muts"], pert["muts"]) + tuple(traj[n] for n in LF_PASSIVE))
    dtype = _dtype(arrs)
    _kernel("lbf_tl", dtype)(_blocks(_lf_cells(geom)), (THREADS,),
                             arrs + (np.int32(bool(nl)),) + geom.args(dtype))


def lbf_ad(adj_state, traj, adj, geom: FinalGeometry):
    """Adjoint of :func:`lbf_tl`: the specified rows' target adjoint is
    consumed into the tables' and muts' adjoints (added into ``adj``)."""
    geom.validate()
    _check(geom, _named("adj", LF_STATE, adj_state)
           + _named("traj", LF_ACTIVE + LF_PASSIVE, traj)
           + _named("adj", LF_ACTIVE, adj), lf_shapes(geom), min_nz=1)
    passive = tuple(traj[n] for n in LF_PASSIVE)
    tables = tuple(traj[n] for n in LB_TABLES)
    dtype = _dtype((adj_state["target"], traj["muts"]) + tables + passive
                   + tuple(adj[n] for n in LF_ACTIVE))
    g = geom.args(dtype)
    if geom.kind != "mu":
        _kernel("lbf_ad_muts", dtype)(_blocks(geom.ny * geom.nx), (THREADS,),
                                      (adj["muts"], adj_state["target"]) + tables
                                      + (traj["muts"],) + passive + g)
    _kernel("lbf_ad_cell", dtype)(_blocks(_lf_cells(geom)), (THREADS,),
                                  (adj_state["target"],) + tuple(adj[n] for n in LB_TABLES)
                                  + (traj["muts"],) + passive + g)


#: spec_bdyupdate_ph + zero_grad_bdy(w) + the frame's calc_p_rho.
SP_STATE = ("ph_pp", "w_pp", "p_pp", "al_pp")
SP_ACTIVE = ("rph_t", "php", "rmu_t", "muts", "th_pp", "mu_pp", "thp", "alt", "c2a")
SP_PASSIVE = ("rdnw", "c1h", "c2h", "c1f", "c2f")


@dataclass(frozen=True)
class SpecPhiGeometry:
    """woof's strict advance_specified_phi_w: one acoustic substep's
    specified-frame ph'' update (WRF spec_bdyupdate_ph), the frame's
    zero-gradient w'' (zero_grad_bdy) and calc_p_rho on the frame columns
    (the interior's is advance_w's)."""
    nz: int
    ny: int
    nx: int
    dtau: float
    spec_zone: int = 1

    def validate(self) -> None:
        for name in ("nz", "ny", "nx"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.spec_zone < 1:
            raise ValueError("spec_zone must be >= 1")
        if min(self.ny, self.nx) < 2 * self.spec_zone + 1:
            # the zero-gradient source column must be interior
            raise ValueError("domain too small for the specified frame")

    def args(self, dtype) -> tuple:
        i, r = np.int32, np.dtype(dtype).type
        return (r(self.dtau), i(self.spec_zone), i(self.nz), i(self.ny), i(self.nx))


def sp_shapes(geom: SpecPhiGeometry) -> dict:
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    wl, ml, col = (nz + 1, ny, nx), (nz, ny, nx), (ny, nx)
    return {"ph_pp": wl, "w_pp": wl, "p_pp": ml, "al_pp": ml, "rph_t": wl, "php": wl,
            "rmu_t": col, "muts": col, "th_pp": ml, "mu_pp": col, "thp": ml, "alt": ml,
            "c2a": ml, "rdnw": (nz,), "c1h": (nz,), "c2h": (nz,), "c1f": (nz + 1,),
            "c2f": (nz + 1,)}


def sph_tl(state, pert_state, traj, pert, geom: SpecPhiGeometry, *, nl=True):
    """WRFPLUS g_spec_bdyupdate_ph, g_zero_grad_bdy (w) and g_calc_p_rho on
    the specified frame columns."""
    geom.validate()
    _check(geom, _named("state", SP_STATE, state) + _named("pert", SP_STATE, pert_state)
           + _named("traj", SP_ACTIVE + SP_PASSIVE, traj)
           + _named("pert", SP_ACTIVE, pert), sp_shapes(geom), min_nz=1)
    st = tuple(x for n in SP_STATE for x in (state[n], pert_state[n]))
    act = tuple(x for n in SP_ACTIVE for x in (traj[n], pert[n]))
    arrs = st + act + tuple(traj[n] for n in SP_PASSIVE)
    dtype = _dtype(arrs)
    _kernel("sph_tl", dtype)(_blocks(geom.ny * geom.nx), (THREADS,),
                             arrs + (np.int32(bool(nl)),) + geom.args(dtype))


def sph_ad(adj_state, state, traj, adj, geom: SpecPhiGeometry):
    """Adjoint of :func:`sph_tl`: ``adj_state`` maps back in place
    (``state`` holds the input ph''); the :data:`SP_ACTIVE` adjoints are
    added into ``adj``."""
    geom.validate()
    sh = sp_shapes(geom)
    _check(geom, _named("adj", SP_STATE, adj_state) + [("state", "ph_pp", state["ph_pp"])]
           + _named("traj", SP_ACTIVE + SP_PASSIVE, traj)
           + _named("adj", SP_ACTIVE, adj), sh, min_nz=1)
    dtype = _dtype(tuple(adj_state[n] for n in SP_STATE) + (state["ph_pp"],)
                   + tuple(traj[n] for n in SP_ACTIVE + SP_PASSIVE)
                   + tuple(adj[n] for n in SP_ACTIVE))
    g = geom.args(dtype)
    _kernel("sph_ad_w", dtype)(_blocks((geom.nz + 1) * geom.ny * geom.nx), (THREADS,),
                               (adj_state["w_pp"],) + g)
    _kernel("sph_ad_col", dtype)(
        _blocks(geom.ny * geom.nx), (THREADS,),
        (adj_state["ph_pp"], adj_state["w_pp"], adj_state["p_pp"], adj_state["al_pp"])
        + tuple(adj[n] for n in ("rph_t", "php", "rmu_t", "muts", "th_pp", "mu_pp", "thp",
                                 "alt", "c2a"))
        + (state["ph_pp"],) + tuple(traj[n] for n in ("rph_t", "php", "rmu_t", "muts",
                                                     "th_pp", "mu_pp", "thp", "alt", "c2a"))
        + tuple(traj[n] for n in SP_PASSIVE) + g)


# ---------------------------------------------------------------- item 8k
#: WRF flow_dep_bdy (share/module_bc.F) on the specified ring of a
#: hydrometeor without a boundary table: woof's strict
#: ``flow_dependent_batch`` (kernels/lbc_flow.cu), which moist.py runs on
#: qc and qr after every stage's scalar update.  Outflow copies the first
#: interior row or column (zero gradient), inflow takes zero; the y sides
#: own the corners.  The mass fluxes only pick the branch, so they are
#: passive.
FLOW_STATE = ("q",)
FLOW_PASSIVE = ("ru", "rv")


@dataclass(frozen=True)
class FlowGeometry:
    nz: int
    ny: int
    nx: int
    spec_zone: int = 1

    def validate(self) -> None:
        for name in ("nz", "ny", "nx"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.spec_zone < 1 or min(self.ny, self.nx) <= 2 * self.spec_zone:
            raise ValueError("spec_zone leaves no flow-dependent boundary interior")


def flow_shapes(geom: FlowGeometry) -> dict:
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    return {"q": (nz, ny, nx), "ru": (nz, ny, nx + 1), "rv": (nz, ny + 1, nx)}


@lru_cache(maxsize=64)
def _flow_frame(ny: int, nx: int, sz: int) -> tuple:
    """Per frame point (flow_frame_point's order): its cell, the interior
    cell it copies on outflow, the face whose flux picks the branch (flat
    index into ru or rv), which of the two, and the flux sign that is
    outflow."""
    dest, src, face, use_v, sign = [], [], [], [], []
    for d in range(sz):
        ni, nj = nx - 2 * d, max(ny - 2 * d - 2, 0)
        pts = ([(d, d + p) for p in range(ni)] + [(ny - 1 - d, d + p) for p in range(ni)]
               + [(d + 1 + p, d) for p in range(nj)] + [(d + 1 + p, nx - 1 - d) for p in range(nj)])
        for j, i in pts:
            if j == d:
                ci = min(max(i, sz), nx - 1 - sz)
                s, f, v, g = sz * nx + ci, j * nx + i, True, -1.0
            elif j == ny - 1 - d:
                ci = min(max(i, sz), nx - 1 - sz)
                s, f, v, g = (ny - 1 - sz) * nx + ci, (j + 1) * nx + i, True, 1.0
            elif i == d:
                cj = min(max(j, sz), ny - 1 - sz)
                s, f, v, g = cj * nx + sz, j * (nx + 1) + i, False, -1.0
            else:
                cj = min(max(j, sz), ny - 1 - sz)
                s, f, v, g = cj * nx + nx - 1 - sz, j * (nx + 1) + i + 1, False, 1.0
            dest.append(j * nx + i)
            src.append(s)
            face.append(f)
            use_v.append(v)
            sign.append(g)
    return tuple(np.asarray(a) for a in (dest, src, face, use_v, sign))


def _flow_map(traj, geom: FlowGeometry):
    """Device (dest, src, outflow) over every level."""
    import cupy as cp
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    dest, src, face, use_v, sign = _flow_frame(ny, nx, geom.spec_zone)
    k = np.arange(nz)[:, None]
    dest3 = (k * ny * nx + dest[None]).ravel()
    src3 = (k * ny * nx + src[None]).ravel()
    fu = (k * ny * (nx + 1) + face[None]).ravel()
    fv = (k * (ny + 1) * nx + face[None]).ravel()
    pick_v = np.broadcast_to(use_v[None], (nz, use_v.size)).ravel()
    sgn = np.broadcast_to(sign[None], (nz, sign.size)).ravel()
    ru, rv = traj["ru"].reshape(-1), traj["rv"].reshape(-1)
    flux = cp.where(cp.asarray(pick_v), rv[cp.asarray(np.where(pick_v, fv, 0))],
                    ru[cp.asarray(np.where(pick_v, 0, fu))])
    # flux < 0 on the low sides, > 0 on the high ones (strict, as the kernel)
    outflow = cp.where(cp.asarray(sgn) > 0, flux > 0, flux < 0)
    return cp.asarray(dest3), cp.asarray(src3), outflow


def _flow_check(geom, named) -> np.dtype:
    geom.validate()
    _check(geom, named, flow_shapes(geom), min_nz=1)
    return _dtype([a for _, _, a in named])


def flow_tl(state, pert_state, traj, pert, geom: FlowGeometry, *, nl=True):
    """WRFPLUS's flow_dep_bdy on one field: the ring of ``state['q']``
    (when ``nl``) and ``pert_state['q']`` take the interior copy on
    outflow and zero on inflow.  ``pert`` is unused (the fluxes are
    passive)."""
    import cupy as cp
    from ._mode import base_only_active
    tl = not base_only_active()
    _flow_check(geom, [("state", "q", state["q"])] + _named("traj", FLOW_PASSIVE, traj)
                + ([("pert", "q", pert_state["q"])] if tl else []))
    dest, src, out = _flow_map(traj, geom)
    for d in (((state,) if nl else ()) + ((pert_state,) if tl else ())):
        q = d["q"].reshape(-1)
        q[dest] = cp.where(out, q[src], q.dtype.type(0))


def flow_ad(adj_state, traj, adj, geom: FlowGeometry):
    """Adjoint of :func:`flow_tl`: the ring's adjoint moves to the interior
    cells it copied (outflow) or is dropped (inflow), in place."""
    import cupy as cp
    dtype = _flow_check(geom, [("adj", "q", adj_state["q"])] + _named("traj", FLOW_PASSIVE, traj))
    dest, src, out = _flow_map(traj, geom)
    a = adj_state["q"].reshape(-1)
    moved = cp.where(out, a[dest], dtype.type(0))
    a[dest] = 0
    a += cp.bincount(src, weights=moved, minlength=a.size).astype(dtype)
