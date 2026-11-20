"""Seeded synthetic trajectories for the TL/AD tests, the WRFPLUS oracle
fixtures and the cost bench.  Physically scaled, not physical: a sounding
is not needed to check a linearisation.
"""

from __future__ import annotations

import numpy as np

from .acoustic import (CW_ACTIVE, CW_STATE, MT_ACTIVE, MT_ALIAS, MT_STATE,
                       SF_ACTIVE,
                       SF_STATE, SI_ACTIVE, SI_STATE, UV_ACTIVE, W_ACTIVE,
                       W_STATE, CWGeometry, MTGeometry, SSGeometry, UVGeometry,
                       WGeometry, cw_shapes, mt_shapes, ss_shapes, uv_shapes,
                       w_shapes)

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


# calc_coef_w: full pressure and inverse density of a column.
_CW_SCALES = {
    "c2a": (1.4e5, 1e3), "a": (-0.1, 0.02), "alpha": (0.8, 0.05),
    "gam": (-0.1, 0.02), "p": (6.0e4, 2e3), "alt": (1.2, 0.1),
    "mup": (0.0, 200.0), "cqw": (1.0, 0.01), "mub2d": (9.0e4, 2e3),
}


def cw_geometry(nz=7, ny=8, nx=9, **overrides) -> CWGeometry:
    base = dict(nz=nz, ny=ny, nx=nx, dtau=3.0, epssm=0.1)
    base.update(overrides)
    return CWGeometry(**base)


def cw_trajectory(geom: CWGeometry, seed: int, dtype=np.float64, *,
                  hybrid: bool = False) -> dict:
    """Host trajectory of calc_coefs (and the outputs' prior words)."""
    rng = np.random.default_rng(seed)
    shapes = cw_shapes(geom)
    out = {}
    for name, (c, s) in _CW_SCALES.items():
        out[name] = c + s * rng.standard_normal(shapes[name])
    w = w_trajectory(WGeometry(nz=geom.nz, ny=geom.ny, nx=geom.nx, rdx=1.0,
                               rdy=1.0, dtau=geom.dtau), seed, hybrid=hybrid)
    for name in ("rdn", "rdnw", "c1h", "c2h", "c1f", "c2f"):
        out[name] = w[name]
    return {k: np.ascontiguousarray(v, dtype=dtype) for k, v in out.items()}


def cw_perturbation(geom: CWGeometry, seed: int, dtype=np.float64) -> dict:
    """Scaled perturbations of :data:`CW_STATE` and :data:`CW_ACTIVE`."""
    rng = np.random.default_rng(seed)
    shapes = cw_shapes(geom)
    return {n: np.ascontiguousarray(
        _CW_SCALES[n][1] * rng.standard_normal(shapes[n]), dtype=dtype)
        for n in CW_STATE + CW_ACTIVE}


# advance_mu_t: coupled momenta u_pp/v_pp against full-mass reference
# winds u/v; thp is theta - t0; msfv is the strict path's msfv_inv.
_MT_SCALES = {
    "mu_pp": (0.0, 20.0), "th_pp": (0.0, 100.0), "mu_old": (0.0, 20.0),
    "th_old": (0.0, 100.0), "p_old": (0.0, 50.0), "mudf": (0.0, 1.0),
    "ww_pp": (0.0, 10.0), "muts": (9.0e4, 2e3),
    "u_pp": (0.0, 50.0), "v_pp": (0.0, 50.0), "u": (5.0, 5.0),
    "v": (-2.0, 5.0), "mup": (0.0, 200.0), "rmu_t": (0.0, 1.0),
    "thp": (10.0, 5.0), "rth_t": (0.0, 50.0), "ww_ref": (0.0, 10.0),
    "p_pp": (0.0, 50.0), "mub2d": (9.0e4, 2e3), "msft": (1.0, 0.02),
    "msfu": (1.0, 0.02), "msfv": (1.0, 0.02),
}


def mt_geometry(nz=7, ny=8, nx=9, **overrides) -> MTGeometry:
    base = dict(nz=nz, ny=ny, nx=nx, rdx=1.0 / 3000.0, rdy=1.0 / 3000.0,
                dtau=3.0)
    base.update(overrides)
    return MTGeometry(**base)


def mt_trajectory(geom: MTGeometry, seed: int, dtype=np.float64, *,
                  hybrid: bool = False) -> dict:
    """Host trajectory and state start of advance_mu_th, by name
    (``hybrid=False``: c1 = 1, c2 = 0, the coordinate WRFPLUS has).
    Without ``geom.mapped`` the map factors are 1, as advance_mu_th
    passes them."""
    rng = np.random.default_rng(seed)
    shapes = mt_shapes(geom)
    out = {}
    for name, (c, s) in _MT_SCALES.items():
        out[name] = c + s * rng.standard_normal(shapes[name])
    if not geom.mapped:
        for name in ("msft", "msfu", "msfv"):
            out[name] = np.ones(shapes[name])
    nz = geom.nz
    eta = np.linspace(1.0, 0.0, nz + 1)
    dnw = np.diff(eta) * (1.0 + 0.2 * (rng.random(nz) - 0.5))
    out["dnw"] = dnw
    out["rdnw"] = 1.0 / dnw
    w = rng.random(nz + 1)
    out["fnm"] = 0.5 + 0.1 * (w - 0.5)
    out["fnp"] = 1.0 - out["fnm"]
    znu = 0.5 * (eta[1:] + eta[:-1])
    if hybrid:
        out["c1h"] = np.sqrt(znu)
        out["c2h"] = (1.0 - out["c1h"]) * 5000.0
    else:
        out["c1h"], out["c2h"] = np.ones(nz), np.zeros(nz)
    return {k: np.ascontiguousarray(v, dtype=dtype) for k, v in out.items()}


def mt_perturbation(geom: MTGeometry, seed: int, dtype=np.float64) -> dict:
    """Scaled perturbations of :data:`MT_STATE` and :data:`MT_ACTIVE`."""
    rng = np.random.default_rng(seed)
    shapes = mt_shapes(geom)
    return {n: np.ascontiguousarray(
        _MT_SCALES[n][1] * rng.standard_normal(shapes[n]), dtype=dtype)
        for n in MT_STATE + MT_ACTIVE}


# small_step_init/_finish: u0..mup0 the stage-start state, u..mup the
# current one; p is the full pressure (c2a = gamma p / alt), thp/thp0
# theta - t0, muts WRF's full column mass.
_SS_SCALES = {
    "u_pp": (0.0, 50.0), "v_pp": (0.0, 50.0), "w_pp": (0.0, 10.0),
    "th_pp": (0.0, 100.0), "ph_pp": (0.0, 20.0), "mu_pp": (0.0, 20.0),
    "al_pp": (0.0, 1e-3), "p_pp": (0.0, 50.0), "p_pp_old": (0.0, 50.0),
    "u0": (5.0, 5.0), "v0": (-2.0, 5.0), "u": (5.0, 5.0), "v": (-2.0, 5.0),
    "w0": (0.0, 1.0), "w": (0.0, 1.0), "thp0": (10.0, 5.0),
    "thp": (10.0, 5.0), "php0": (0.0, 20.0), "php": (0.0, 20.0),
    "p": (8.0e4, 1e4), "alt": (1.0, 0.2), "mup0": (0.0, 200.0),
    "mup": (0.0, 200.0), "muts": (9.0e4, 2e3), "h_diabatic": (0.0, 1e-3),
    "mub2d": (9.0e4, 2e3), "msft": (1.0, 0.02), "msfu": (1.0, 0.02),
    "msfv": (1.0, 0.02),
}


def ss_geometry(nz=7, ny=8, nx=9, **overrides) -> SSGeometry:
    base = dict(nz=nz, ny=ny, nx=nx)
    base.update(overrides)
    return SSGeometry(**base)


def ss_trajectory(geom: SSGeometry, seed: int, dtype=np.float64, *,
                  hybrid: bool = False) -> dict:
    """Host trajectory (and state start) of small_step_init/_finish, by
    name.  Without ``geom.has_msf`` the map factors are 1."""
    rng = np.random.default_rng(seed)
    shapes = ss_shapes(geom)
    out = {}
    for name, (c, s) in _SS_SCALES.items():
        out[name] = c + s * rng.standard_normal(shapes[name])
    if not geom.has_msf:
        for name in ("msft", "msfu", "msfv"):
            out[name] = np.ones(shapes[name])
    nz = geom.nz
    eta = np.linspace(1.0, 0.0, nz + 1)
    dnw = np.diff(eta) * (1.0 + 0.2 * (rng.random(nz) - 0.5))
    out["rdnw"] = 1.0 / dnw
    znu = 0.5 * (eta[1:] + eta[:-1])
    if hybrid:
        out["c1h"] = np.sqrt(znu)
        out["c2h"] = (1.0 - out["c1h"]) * 5000.0
        out["c1f"] = np.sqrt(eta)
        out["c2f"] = (1.0 - out["c1f"]) * 5000.0
    else:
        out["c1h"], out["c2h"] = np.ones(nz), np.zeros(nz)
        out["c1f"], out["c2f"] = np.ones(nz + 1), np.zeros(nz + 1)
    return {k: np.ascontiguousarray(v, dtype=dtype) for k, v in out.items()}


def ss_perturbation(geom: SSGeometry, seed: int, dtype=np.float64) -> dict:
    """Scaled perturbations of every small_step state and active array."""
    rng = np.random.default_rng(seed)
    shapes = ss_shapes(geom)
    names = dict.fromkeys(SI_STATE + SI_ACTIVE + SF_STATE + SF_ACTIVE)
    return {n: np.ascontiguousarray(
        _SS_SCALES[n][1] * rng.standard_normal(shapes[n]), dtype=dtype)
        for n in names}


# One acoustic substep (advance_uv -> advance_mu_t -> advance_w) on the
# strict primal's names: periodic (spec_zone 0), msf 1, shared dtau and
# off-centring, so the three operators see one consistent state.
def substep_geometries(nz=7, ny=8, nx=9, *, base3d=1, top_lid=0,
                       smdiv=0.1) -> tuple:
    return (uv_geometry(nz, ny, nx, base3d=base3d, top_lid=top_lid,
                        smdiv=smdiv),
            mt_geometry(nz, ny, nx),
            w_geometry(nz, ny, nx, base3d=base3d, top_lid=top_lid))


def _mt_named(d: dict) -> dict:
    return {MT_ALIAS.get(n, n): v for n, v in d.items()}


def substep_trajectory(geoms: tuple, seed: int, dtype=np.float64, *,
                       hybrid: bool = False) -> dict:
    """Host base state of one substep: advance_mu_t's, then advance_uv's,
    then advance_w's draws (the later win on shared names, so the
    vertical constants are advance_w's); dnw = 1/rdnw."""
    uvg, mtg, wg = geoms
    out = _mt_named(mt_trajectory(mtg, seed + 2, dtype, hybrid=hybrid))
    out.update(uv_trajectory(uvg, seed + 1, dtype, hybrid=hybrid))
    out.update(w_trajectory(wg, seed, dtype, hybrid=hybrid))
    out["dnw"] = np.ascontiguousarray(1.0 / out["rdnw"], dtype=dtype)
    return out


def substep_active(geoms: tuple) -> tuple:
    """Every name one substep reads or writes that is perturbed."""
    names = (("u_pp", "v_pp") + UV_ACTIVE
             + tuple(MT_ALIAS.get(n, n) for n in MT_STATE + MT_ACTIVE)
             + W_STATE + W_ACTIVE)
    return tuple(dict.fromkeys(names))


def substep_perturbation(geoms: tuple, seed: int, dtype=np.float64) -> dict:
    uvg, mtg, wg = geoms
    out = _mt_named(mt_perturbation(mtg, seed + 2, dtype))
    out.update(uv_perturbation(uvg, seed + 1, dtype))
    out.update(w_perturbation(wg, seed, dtype))
    return {n: out[n] for n in substep_active(geoms)}
