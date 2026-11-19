"""Acoustic-substep TL/AD launchers (WRFPLUS module_small_step_em_{tl,ad}).

Each routine takes its trajectory (the base state the primal read) as a
dict of device arrays in woof's (k, j, i) layout, the same names as the
primal launcher in woof/core/acoustic.py, and the geometry as a
:class:`UVGeometry`.  The real type is the arrays' dtype: float64 (the
test and oracle build, CUDA and the host backend) or float32 (Metal,
which has no binary64).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

KERNELS = Path(__file__).resolve().parent / "kernels"
THREADS = 256

#: Trajectory arrays of advance_uv, in kernel argument order.
UV_TRAJECTORY = ("ru_t", "rv_t", "p_pp", "p_pp_old", "ph_pp", "php", "phb",
                 "alt", "al_pp", "pb", "mup", "mu_pp", "mub2d", "c1h", "c2h",
                 "fnm", "fnp", "rdnw", "cqu", "cqv", "mudf", "msfu", "msfv")

#: The perturbed inputs of advance_uv besides u_pp/v_pp, in kernel order.
UV_ACTIVE = ("ru_t", "rv_t", "p_pp", "p_pp_old", "ph_pp", "php", "alt",
             "al_pp", "mup", "mu_pp", "cqu", "cqv", "mudf")


@dataclass(frozen=True)
class UVGeometry:
    """advance_uv's scalars, as woof/core/acoustic.py passes them.

    ``xscale``/``yscale`` are the strict launcher's ``-emdiv*dx`` and
    ``-emdiv*dy``; ``msfv`` is then MSFV_INV, as in the primal.
    """
    nz: int
    ny: int
    nx: int
    rdx: float
    rdy: float
    dtau: float
    smdiv: float = 0.0
    cf1: float = 0.0
    cf2: float = 0.0
    cf3: float = 0.0
    spec_zone: int = 0
    base3d: int = 1
    top_lid: int = 0
    moist_cq: int = 0
    use_emdiv: int = 0
    has_msf: int = 0
    xscale: float = 0.0
    yscale: float = 0.0

    def args(self, dtype) -> tuple:
        i, r = np.int32, np.dtype(dtype).type
        return (i(self.nz), i(self.ny), i(self.nx), i(self.spec_zone),
                i(self.base3d), i(self.top_lid), i(self.moist_cq),
                i(self.use_emdiv), i(self.has_msf), r(self.rdx), r(self.rdy),
                r(self.dtau), r(self.smdiv), r(self.cf1), r(self.cf2),
                r(self.cf3), r(self.xscale), r(self.yscale))


def _real(dtype) -> str:
    dtype = np.dtype(dtype)
    if dtype == np.float64:
        return "double"
    if dtype == np.float32:
        return "float"
    raise TypeError(f"TL/AD kernels run in float64 or float32, not {dtype}")


@lru_cache(maxsize=None)
def _module(unit: str, real: str):
    import cupy as cp
    source = (KERNELS / f"{unit}.cu").read_text(encoding="utf-8")
    return cp.RawModule(code=source,
                        options=("--fmad=false", f"-DWRFPLUS_REAL={real}"))


def _kernel(unit: str, name: str, dtype):
    return _module(unit, _real(dtype)).get_function(name)


def _blocks(n: int) -> tuple[int]:
    return ((n + THREADS - 1) // THREADS,)


def _dtype(arrays) -> np.dtype:
    kinds = {np.dtype(a.dtype) for a in arrays}
    if len(kinds) != 1:
        raise TypeError(f"mixed real types {sorted(map(str, kinds))}")
    return kinds.pop()


def advance_uv_tl(u_pp, v_pp, u_d, v_d, traj: dict, pert: dict,
                  geom: UVGeometry, *, nl: bool = True) -> None:
    """WRFPLUS g_advance_uv: update u_d/v_d (and u_pp/v_pp when ``nl``).

    ``pert`` holds the perturbations of :data:`UV_ACTIVE`, by name.
    """
    traj_args = tuple(traj[n] for n in UV_TRAJECTORY)
    pert_args = tuple(pert[n] for n in UV_ACTIVE)
    dtype = _dtype((u_pp, v_pp, u_d, v_d) + traj_args + pert_args)
    g = geom
    _kernel("advance_uv", "advance_uv_tl", dtype)(
        _blocks(g.nz * (g.ny + 1) * (g.nx + 1)), (THREADS,),
        (u_pp, v_pp, u_d, v_d) + traj_args + pert_args
        + (np.int32(bool(nl)),) + g.args(dtype))


def advance_uv_ad(u_a, v_a, traj: dict, adj: dict, geom: UVGeometry) -> None:
    """WRFPLUS a_advance_uv: add the adjoints of :data:`UV_ACTIVE` into
    ``adj`` (by name).  ``u_a``/``v_a`` pass through unchanged, because
    the update is ``u_out = u_in + ...``.
    """
    import cupy as cp
    traj_args = tuple(traj[n] for n in UV_TRAJECTORY)
    dtype = _dtype((u_a, v_a) + traj_args + tuple(adj[n] for n in UV_ACTIVE))
    g = geom
    nu = g.nz * g.ny * (g.nx + 1)
    nv = g.nz * (g.ny + 1) * g.nx
    su = cp.empty(5 * nu, dtype=dtype)
    sv = cp.empty(5 * nv, dtype=dtype)
    geo = g.args(dtype)
    _kernel("advance_uv", "advance_uv_ad_face", dtype)(
        _blocks(g.nz * (g.ny + 1) * (g.nx + 1)), (THREADS,),
        (u_a, v_a) + traj_args
        + (adj["ru_t"], adj["rv_t"], adj["cqu"], adj["cqv"], su, sv) + geo)
    _kernel("advance_uv", "advance_uv_ad_gather", dtype)(
        _blocks((g.nz + 1) * g.ny * g.nx), (THREADS,),
        traj_args + (su, sv, adj["ph_pp"], adj["php"], adj["alt"],
                     adj["al_pp"], adj["p_pp"], adj["p_pp_old"]) + geo)
    _kernel("advance_uv", "advance_uv_ad_column", dtype)(
        _blocks(g.ny * g.nx), (THREADS,),
        (traj["c1h"], su, sv, adj["mup"], adj["mu_pp"], adj["mudf"]) + geo)


def uv_shapes(geom: UVGeometry) -> dict[str, tuple[int, ...]]:
    """Array shapes of advance_uv's trajectory and outputs, by name."""
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    mass, wlev, col = (nz, ny, nx), (nz + 1, ny, nx), (ny, nx)
    return {
        "u_pp": (nz, ny, nx + 1), "v_pp": (nz, ny + 1, nx),
        "ru_t": (nz, ny, nx + 1), "rv_t": (nz, ny + 1, nx),
        "p_pp": mass, "p_pp_old": mass, "ph_pp": wlev, "php": wlev,
        "phb": wlev if geom.base3d else (nz + 1,),
        "alt": mass, "al_pp": mass, "pb": mass,
        "mup": col, "mu_pp": col, "mub2d": col,
        "c1h": (nz,), "c2h": (nz,), "fnm": (nz + 1,), "fnp": (nz + 1,),
        "rdnw": (nz,), "cqu": (nz, ny, nx + 1), "cqv": (nz, ny + 1, nx),
        "mudf": col, "msfu": (ny, nx + 1), "msfv": (ny + 1, nx),
    }
