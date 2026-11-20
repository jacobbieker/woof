"""Flux-form advection TL/AD (P3a item 6): woof's strict flux_div_scalar,
flux_div_u, flux_div_v and flux_div_w (WRF advect_scalar, advect_u,
advect_v and advect_w), periodic, open and specified lateral boundaries,
map factors, h_*_adv_order 3-6 and v_*_adv_order 2-6
(docs/dev/P3A-WRF-DRYCORE-ADJOINT.md section 5, item 6).

Same conventions as :mod:`woof.grad.wrfplus.acoustic`.  The state is the
tendency, which the primal adds into, so its adjoint passes through
unchanged; ``adv_ad`` adds into the adjoints of the advected field and
the three mass fluxes.  ``rdz`` is ``rdnw`` for scalar, u and v and
``rdn`` for w, as in woof/core/advection.py.

There is no ``has_msf=False``: pass ``msf`` of ones, which gives the
same words (1.0 * rdx is exact).

Not covered: h_*_adv_order 2 (the legacy idealized path), WENO and
monotonic/positive-definite limiters.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .acoustic import _check, _dtype, _real

KERNELS = Path(__file__).resolve().parent / "kernels"
KINDS = ("scalar", "u", "v", "w")
H_ORDERS = (3, 4, 5, 6)
V_ORDERS = (2, 3, 4, 5, 6)
BLOCK = (32, 4, 1)

ADV_STATE = ("tend",)
ADV_ACTIVE = ("q", "ru", "rv", "rw")
ADV_PASSIVE = ("msf", "rdz", "fnm", "fnp")
#: The adjoint's kernels, in launch order: the field adjoint's x, y and z
#: gathers (each adds into it), then the three mass fluxes'.
AD_PASSES = ("qx", "qy", "qz", "ru", "rv", "rw")


@dataclass(frozen=True)
class AdvGeometry:
    """One flux_div launch: the field kind, the grid, ``rdx = 1/dx`` and
    ``rdy = 1/dy``, the lateral boundary options and the orders."""
    kind: str
    nz: int
    ny: int
    nx: int
    rdx: float
    rdy: float
    open_x: int = 0
    open_y: int = 0
    spec: int = 0
    vorder: int = 3
    horder: int = 5

    def args(self, dtype) -> tuple:
        i, r = np.int32, np.dtype(dtype).type
        return (r(self.rdx), r(self.rdy), i(self.nz), i(self.ny), i(self.nx),
                i(self.open_x), i(self.open_y), i(self.spec), i(self.vorder))

    def validate(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if self.horder not in H_ORDERS:
            raise ValueError(f"horder must be one of {H_ORDERS}, got {self.horder!r}")
        if self.vorder not in V_ORDERS:
            raise ValueError(f"vorder must be one of {V_ORDERS}, got {self.vorder!r}")
        for name in ("nz", "ny", "nx"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.kind == "w" and self.nz < 2:
            raise ValueError(f"w advection needs nz >= 2 (the lid flux), got {self.nz}")
        if self.open_x and self.nx < 7:
            raise ValueError(f"open_x advection needs nx >= 7, got {self.nx}")
        if self.open_y and self.ny < 7:
            raise ValueError(f"open_y advection needs ny >= 7, got {self.ny}")
        if self.spec and not (self.open_x or self.open_y):
            raise ValueError("spec needs an open axis")


def adv_shapes(geom: AdvGeometry) -> dict:
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    point = {"scalar": (nz, ny, nx), "u": (nz, ny, nx + 1),
             "v": (nz, ny + 1, nx), "w": (nz + 1, ny, nx)}[geom.kind]
    return {"tend": point, "q": point, "ru": (nz, ny, nx + 1),
            "rv": (nz, ny + 1, nx), "rw": (nz + 1, ny, nx),
            "msf": point[1:], "rdz": (nz,), "fnm": (nz,), "fnp": (nz,)}


@lru_cache(maxsize=None)
def _device_module(real: str, horder: int, device: int):
    import cupy as cp
    source = (KERNELS / "advect.cu").read_text(encoding="utf-8")
    return cp.RawModule(code=source, options=(
        "--fmad=false", f"-DWRFPLUS_REAL={real}", f"-DADV_H={horder}"))


def _kernel(name: str, dtype, horder: int):
    import cupy as cp
    return _device_module(_real(dtype), horder, cp.cuda.Device().id).get_function(name)


def _grid(ni: int, nj: int, nk: int) -> tuple:
    return ((ni + BLOCK[0] - 1) // BLOCK[0], (nj + BLOCK[1] - 1) // BLOCK[1], nk)


def _named(label, names, d):
    return [(label, n, d[n]) for n in names]


def _prepare(geom: AdvGeometry, named) -> None:
    geom.validate()
    _check(geom, named, adv_shapes(geom), min_nz=2 if geom.kind == "w" else 1)


def adv_tl(state, pert_state, traj, pert, geom: AdvGeometry, *, nl=True):
    """TL of tend += -div(F(q; ru, rv, rw)): update ``pert_state['tend']``
    (and ``state['tend']`` when ``nl``) from the perturbations of
    :data:`ADV_ACTIVE`."""
    _prepare(geom, _named("state", ADV_STATE, state)
             + _named("pert", ADV_STATE, pert_state)
             + _named("traj", ADV_ACTIVE + ADV_PASSIVE, traj)
             + _named("pert", ADV_ACTIVE, pert))
    pairs = ()
    for n in ADV_ACTIVE:
        pairs += (traj[n], pert[n])
    arrs = ((state["tend"], pert_state["tend"]) + pairs
            + tuple(traj[n] for n in ADV_PASSIVE))
    dtype = _dtype(arrs)
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    ni, nj, nk = {"scalar": (nx, ny, nz), "u": (nx + 1, ny, nz),
                  "v": (nx, ny + 1, nz), "w": (nx, ny, nz + 1)}[geom.kind]
    _kernel(f"adv_{geom.kind}_tl", dtype, geom.horder)(
        _grid(ni, nj, nk), BLOCK, arrs + (np.int32(bool(nl)),) + geom.args(dtype))


def adv_ad(adj_state, traj, adj, geom: AdvGeometry):
    """Adjoint of :func:`adv_tl`: add into ``adj`` (q, ru, rv, rw) from
    ``adj_state['tend']``, which is unchanged."""
    _prepare(geom, _named("adj", ADV_STATE, adj_state)
             + _named("traj", ADV_ACTIVE + ADV_PASSIVE, traj)
             + _named("adj", ADV_ACTIVE, adj))
    arrs = ((adj_state["tend"],) + tuple(traj[n] for n in ADV_ACTIVE)
            + tuple(adj[n] for n in ADV_ACTIVE)
            + tuple(traj[n] for n in ADV_PASSIVE))
    dtype = _dtype(arrs)
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    grid = _grid(nx + 1, ny + 1, nz + 1)
    for out in AD_PASSES:
        _kernel(f"adv_{geom.kind}_ad_{out}", dtype, geom.horder)(
            grid, BLOCK, arrs + geom.args(dtype))
