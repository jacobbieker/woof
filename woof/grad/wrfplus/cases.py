"""Seeded synthetic trajectories for the TL/AD tests, the WRFPLUS oracle
fixtures and the cost bench.  Physically scaled, not physical: a sounding
is not needed to check a linearisation.
"""

from __future__ import annotations

import numpy as np

from .acoustic import (UV_ACTIVE, W_ACTIVE, W_STATE, UVGeometry, WGeometry,
                       uv_shapes, w_shapes)

# (centre, spread) per field, roughly a mid-latitude 1-3 km grid.
_UV_SCALES = {
    "u_pp": (0.0, 1.0), "v_pp": (0.0, 1.0),
    "ru_t": (0.0, 5e-2), "rv_t": (0.0, 5e-2),
    "p_pp": (0.0, 50.0), "p_pp_old": (0.0, 50.0),
    "ph_pp": (0.0, 20.0), "php": (0.0, 20.0),
    "alt": (1.0, 0.2), "al_pp": (0.0, 1e-3),
    "mup": (0.0, 200.0), "mu_pp": (0.0, 20.0), "mub2d": (9.0e4, 2e3),
    "cqu": (1.0, 0.01), "cqv": (1.0, 0.01), "mudf": (0.0, 1.0),
    "msfu": (1.0, 0.02), "msfv": (1.0, 0.02),
}


def uv_geometry(nz=7, ny=8, nx=9, **overrides) -> UVGeometry:
    base = dict(nz=nz, ny=ny, nx=nx, rdx=1.0 / 3000.0, rdy=1.0 / 3000.0,
                dtau=3.0, smdiv=0.1, cf1=1.6, cf2=-0.8, cf3=0.2)
    base.update(overrides)
    return UVGeometry(**base)


def uv_trajectory(geom: UVGeometry, seed: int, dtype=np.float64, *,
                  hybrid: bool = False) -> dict:
    """Host (NumPy) trajectory and the u_pp/v_pp start, by name.

    ``hybrid=False`` is WRF's terrain-following coordinate (c1h = 1,
    c2h = 0, hybrid_opt = 0), the only one WRFPLUS linearises.
    """
    rng = np.random.default_rng(seed)
    shapes = uv_shapes(geom)
    out = {}
    for name, shape in shapes.items():
        if name in _UV_SCALES:
            c, s = _UV_SCALES[name]
            out[name] = c + s * rng.standard_normal(shape)
    nz = geom.nz
    eta = np.linspace(1.0, 0.0, nz + 1)
    z = 15000.0 * (1.0 - eta)                                  # w levels
    zm = 0.5 * (z[1:] + z[:-1])
    if geom.base3d:
        terrain = 300.0 * rng.random(shapes["mup"])
        out["phb"] = 9.81 * (z[:, None, None] + terrain[None]
                             * eta[:, None, None])
        out["pb"] = 1.0e5 * np.exp(-(zm[:, None, None] + terrain[None]
                                     * 0.5 * (eta[1:] + eta[:-1])[:, None, None])
                                   / 8000.0)
    else:
        out["phb"] = 9.81 * z
        out["pb"] = np.broadcast_to(1.0e5 * np.exp(-zm / 8000.0)[:, None, None],
                                    shapes["pb"]).copy()
    dnw = np.diff(eta)
    out["rdnw"] = 1.0 / dnw
    w = rng.random(nz + 1)
    out["fnm"] = 0.5 + 0.1 * (w - 0.5)
    out["fnp"] = 1.0 - out["fnm"]
    if hybrid:
        out["c1h"] = np.sqrt(0.5 * (eta[1:] + eta[:-1]))
        out["c2h"] = (1.0 - out["c1h"]) * 5000.0
    else:
        out["c1h"] = np.ones(nz)
        out["c2h"] = np.zeros(nz)
    return {k: np.ascontiguousarray(v, dtype=dtype) for k, v in out.items()}


def uv_perturbation(geom: UVGeometry, seed: int, dtype=np.float64) -> dict:
    """Unit-scaled perturbations of u_pp, v_pp and :data:`UV_ACTIVE`."""
    rng = np.random.default_rng(seed)
    shapes = uv_shapes(geom)
    names = ("u_pp", "v_pp") + UV_ACTIVE
    return {n: np.ascontiguousarray(
        _UV_SCALES.get(n, (0.0, 1.0))[1] * rng.standard_normal(shapes[n]),
        dtype=dtype) for n in names}


# advance_w: theta'' is (mu*theta)'' as woof's strict path carries it and
# thp is theta_t* - 300 (t0); the Thomas factors are WRF-like (|a|, |gam|
# small, alpha near 1) so the sweeps stay well conditioned.
_W_SCALES = {
    "w_pp": (0.0, 1.0), "ph_pp": (0.0, 2.0), "p_pp": (0.0, 50.0),
    "al_pp": (0.0, 1e-3), "rw_t": (0.0, 5e-2), "rph_t": (0.0, 0.5),
    "ww_pp": (0.0, 10.0), "mu_pp": (0.0, 20.0), "mu_pp_old": (0.0, 20.0),
    "th_pp": (0.0, 100.0), "th_pp_old": (0.0, 100.0), "thp": (10.0, 5.0),
    "php": (0.0, 20.0), "alt": (1.0, 0.2), "c2a": (1.4e5, 1e3),
    "a": (-0.1, 0.02), "alpha": (0.8, 0.05), "gam": (-0.1, 0.02),
    "mup": (0.0, 200.0), "u_pp": (0.0, 1.0), "v_pp": (0.0, 1.0),
    "ht": (500.0, 100.0), "mub2d": (9.0e4, 2e3), "cqw": (1.0, 0.01),
}


def w_geometry(nz=7, ny=8, nx=9, **overrides) -> WGeometry:
    base = dict(nz=nz, ny=ny, nx=nx, rdx=1.0 / 3000.0, rdy=1.0 / 3000.0,
                dtau=3.0, epssm=0.1, cf1=1.6, cf2=-0.8, cf3=0.2)
    base.update(overrides)
    return WGeometry(**base)


def w_trajectory(geom: WGeometry, seed: int, dtype=np.float64, *,
                 hybrid: bool = False) -> dict:
    """Host trajectory and state start of advance_w, by name
    (``hybrid=False``: c1 = 1, c2 = 0, the coordinate WRFPLUS has)."""
    rng = np.random.default_rng(seed)
    shapes = w_shapes(geom)
    out = {}
    for name, shape in shapes.items():
        if name in _W_SCALES:
            c, s = _W_SCALES[name]
            out[name] = c + s * rng.standard_normal(shape)
    nz = geom.nz
    eta = np.linspace(1.0, 0.0, nz + 1)
    z = 15000.0 * (1.0 - eta)
    if geom.base3d:
        terrain = 300.0 * rng.random(shapes["mup"])
        out["phb"] = 9.81 * (z[:, None, None] + terrain[None]
                             * eta[:, None, None])
    else:
        out["phb"] = 9.81 * z
    out["rdnw"] = 1.0 / np.diff(eta)
    znu = 0.5 * (eta[1:] + eta[:-1])
    rdn = np.zeros(nz)
    rdn[1:] = 1.0 / np.diff(znu)
    out["rdn"] = rdn
    w = rng.random(nz + 1)
    out["fnm"] = 0.5 + 0.1 * (w - 0.5)
    out["fnp"] = 1.0 - out["fnm"]
    if hybrid:
        out["c1h"] = np.sqrt(znu)
        out["c2h"] = (1.0 - out["c1h"]) * 5000.0
        out["c1f"] = np.sqrt(eta)
        out["c2f"] = (1.0 - out["c1f"]) * 5000.0
    else:
        out["c1h"], out["c2h"] = np.ones(nz), np.zeros(nz)
        out["c1f"], out["c2f"] = np.ones(nz + 1), np.zeros(nz + 1)
    return {k: np.ascontiguousarray(v, dtype=dtype) for k, v in out.items()}


def w_perturbation(geom: WGeometry, seed: int, dtype=np.float64) -> dict:
    """Scaled perturbations of :data:`W_STATE` and :data:`W_ACTIVE`."""
    rng = np.random.default_rng(seed)
    shapes = w_shapes(geom)
    names = W_STATE + W_ACTIVE
    return {n: np.ascontiguousarray(
        _W_SCALES.get(n, (0.0, 1.0))[1] * rng.standard_normal(shapes[n]),
        dtype=dtype) for n in names}
