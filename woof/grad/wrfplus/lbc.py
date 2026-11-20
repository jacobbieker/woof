"""Lateral-boundary tendency TL/AD (P3a item 7): WRF relax_bdy_dry and
spec_bdy_dry on woof's strict state_relaxation_multi rows
(docs/dev/P3A-WRF-DRYCORE-ADJOINT.md section 5, item 7).

One call is one row of the primal's launch: a field kind (u, v, theta,
phi, mu or w) and the row's flags (:class:`LBCGeometry`).  The strict
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

Whole domains only (no streamed tile seams).  Not covered: moist and
scalar rows, the nested (time-t copy) sources, the non-strict path.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .acoustic import THREADS, _blocks, _check, _dtype, _real

KERNELS = Path(__file__).resolve().parent / "kernels"

LB_STATE = ("tend", "held")
LB_TABLES = ("west", "west_t", "east", "east_t", "south", "south_t", "north", "north_t")
LB_ACTIVE = ("src", "mup") + LB_TABLES
LB_PASSIVE = ("mub", "c1h", "c2h", "c1f", "c2f", "msft", "msfu", "msfv", "fcx", "gcx")
LB_KINDS = {"u": 0, "v": 1, "theta": 2, "phi": 3, "mu": 4, "w": 6}


@lru_cache(maxsize=None)
def _device_module(real: str, device: int):
    import cupy as cp
    source = (KERNELS / "lbc.cu").read_text(encoding="utf-8")
    return cp.RawModule(code=source, options=("--fmad=false", f"-DWRFPLUS_REAL={real}"))


def _kernel(name: str, dtype):
    import cupy as cp
    return _device_module(_real(dtype), cp.cuda.Device().id).get_function(name)


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
                "phi": (nz + 1, ny, nx), "w": (nz + 1, ny, nx), "mu": (1, ny, nx)}[self.kind]

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
            raise ValueError(f"kind {self.kind!r} is not a dry row")
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
