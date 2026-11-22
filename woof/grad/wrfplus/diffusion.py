"""diff_opt 1 / km_opt 4 coefficient TL/AD (P3a item 5): WRF
compute_diff_metrics + cal_deform_and_div (the deformation) and smag2d_km
at diff_opt 1 on woof's strict wrf_smag_deform and wrf_diff_opt1_km4 +
wrf_smag_km_bc (docs/dev/P3A-WRF-DRYCORE-ADJOINT.md section 5, item 5).

Same conventions as :mod:`woof.grad.wrfplus.acoustic`: dicts of arrays by
name, ``*_tl(state, pert_state, traj, pert, geom, nl=True)`` updates the
perturbation (and the base state when ``nl``), ``*_ad`` adds into the
inputs' adjoints.  Both operators overwrite their outputs, so the AD
consumes (zeroes) the outputs' adjoints.

Deformation: ``u`` and ``v`` are the (coupled-to-map-factor) winds the
primal reads, ``php`` the w-level perturbation geopotential; ``phb`` is
the 3-D base geopotential.  Coefficients: the horizontal eddy viscosity
``xkmh`` and diffusivity ``xkhh = xkmh / prandtl`` from the deformation,
with the 10 * mlen cap and the cold-zero outer row of a bounded axis.
Not covered: diff_opt 2's slope limiter, the TC horizontal length, a 1-D
base geopotential and m_opt.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ._mode import tl_options
from .acoustic import THREADS, _blocks, _check, _dtype, _real

KERNELS = Path(__file__).resolve().parent / "kernels"

#: woof's dycore._PRANDTL, as the primal passes it.
PRANDTL = 1.0 / 3.0


@lru_cache(maxsize=None)
def _device_module(real: str, device: int, build: tuple = ()):
    import cupy as cp
    source = (KERNELS / "diffusion.cu").read_text(encoding="utf-8")
    return cp.RawModule(code=source,
                        options=("--fmad=false", f"-DWRFPLUS_REAL={real}") + build)


def _kernel(name: str, dtype):
    import cupy as cp
    return _device_module(_real(dtype), cp.cuda.Device().id, tl_options()).get_function(name)


def _named(label, names, d):
    return [(label, n, d[n]) for n in names]


def _axes(geom) -> None:
    for name in ("nz", "ny", "nx"):
        if getattr(geom, name) < 1:
            raise ValueError(f"{name} must be >= 1, got {getattr(geom, name)}")
    for name in ("boundary_x", "boundary_y"):
        if getattr(geom, name) not in (0, 1):
            raise ValueError(f"{name} is 0 or 1, got {getattr(geom, name)}")


# ------------------------------------------------------------- deformation

DF_STATE = ("defor11", "defor22", "defor12")
DF_ACTIVE = ("u", "v", "php")
DF_PASSIVE = ("phb", "msft", "msfu", "msfv", "fnm", "fnp", "dn", "dnw")


@dataclass(frozen=True)
class DFGeometry:
    """wrf_smag_deform's scalars: ``rdx = 1/dx`` and ``rdy = 1/dy`` as the
    primal rounds them, the bottom extrapolation weights cf1-cf3 and the
    bounded (open or specified) axes."""
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

    def args(self, dtype) -> tuple:
        i, r = np.int32, np.dtype(dtype).type
        return (r(self.rdx), r(self.rdy), r(self.cf1), r(self.cf2), r(self.cf3),
                i(self.boundary_x), i(self.boundary_y), i(self.nz), i(self.ny), i(self.nx))

    def validate(self) -> None:
        _axes(self)
        if self.nz < 3:
            raise ValueError(f"the deformation needs nz >= 3 (the bottom "
                             f"extrapolation reads levels 0-2), got {self.nz}")


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
    _check(geom, _named("state", DF_STATE, state) + _named("pert", DF_STATE, pert_state)
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
        if not (self.dx > 0 and self.dy > 0 and self.prandtl > 0):
            raise ValueError("dx, dy and prandtl must be positive")


def km_shapes(geom) -> dict:
    mass = (geom.nz, geom.ny, geom.nx)
    return {"xkmh": mass, "xkhh": mass, "defor11": mass, "defor22": mass, "defor12": mass,
            "msft": (geom.ny, geom.nx)}


def km_tl(state, pert_state, traj, pert, geom: KMGeometry, *, nl=True):
    """WRFPLUS G_SMAG2D_KM (diff_opt 1) on woof's strict wrf_diff_opt1_km4
    + wrf_smag_km_bc: xkmh and xkhh (overwritten) of the deformation."""
    geom.validate()
    _check(geom, _named("state", KM_STATE, state) + _named("pert", KM_STATE, pert_state)
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
