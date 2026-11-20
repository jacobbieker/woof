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
from .stage import (AT_ACTIVE, AT_ALIAS, AT_PASSIVE, AT_STATE, PRP_ACTIVE,
                    PRP_ALIAS, PRP_PASSIVE, PRP_STATE, RSP_ACTIVE, RSP_ALIAS,
                    RSP_PASSIVE, RSP_STATE, ATGeometry, PRPGeometry,
                    RSPGeometry, StageGeometry, SWGeometry, at_shapes,
                    prp_shapes, rsp_shapes, sw_shapes)

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


# ---------------------------------------------------------------- RK stage
# calc_p_rho_phi, rk_step_prep, rk_addtend_dry and set_w_surface (P3a item
# 4), on woof's strict primal names.

_RK_SCALES = {
    "thp": (10.0, 5.0), "php": (0.0, 20.0), "mup": (0.0, 200.0),
    "mub": (9.0e4, 2e3), "u": (5.0, 5.0), "v": (-2.0, 5.0),
    "w": (0.0, 1.0), "h_diabatic": (0.0, 1e-3),
    "p": (8.0e4, 1e4), "al": (0.0, 1e-3), "alt": (1.0, 0.2),
    "p_pp": (0.0, 50.0), "ru": (0.0, 4e5), "rv": (0.0, 4e5),
    "ww": (0.0, 10.0),
    "ru_t": (0.0, 50.0), "rv_t": (0.0, 50.0), "rw_t": (0.0, 5.0),
    "rph_t": (0.0, 50.0), "rth_t": (0.0, 50.0), "rmu_t": (0.0, 0.5),
    "ru_tf": (0.0, 50.0), "rv_tf": (0.0, 50.0), "rw_tf": (0.0, 5.0),
    "rph_tf": (0.0, 50.0), "rth_tf": (0.0, 50.0), "rmu_tf": (0.0, 0.5),
}


def _vertical(nz, rng, hybrid):
    eta = np.linspace(1.0, 0.0, nz + 1)
    dnw = np.diff(eta) * (1.0 + 0.2 * (rng.random(nz) - 0.5))
    eta = np.concatenate(([1.0], 1.0 + np.cumsum(dnw)))
    znu = 0.5 * (eta[1:] + eta[:-1])
    out = {"dnw": dnw, "rdnw": 1.0 / dnw, "znw": eta, "znu": znu}
    if hybrid:
        out["c1h"] = np.sqrt(np.clip(znu, 0.0, None))
        out["c2h"] = (1.0 - out["c1h"]) * 5000.0
    else:
        out["c1h"], out["c2h"] = np.ones(nz), np.zeros(nz)
    # hypsometric_opt 2's pressure coefficients (non-hybrid: c3 = eta,
    # c4 = 0; hybrid: c3 = c1, c4 = c2 times the column scale, as WRF).
    c1f = np.sqrt(np.clip(eta, 0.0, None)) if hybrid else np.ones(nz + 1)
    out["c3h"] = out["c1h"] * znu if hybrid else znu.copy()
    out["c4h"] = out["c2h"] * znu if hybrid else np.zeros(nz)
    out["c3f"] = c1f * eta if hybrid else eta.copy()
    out["c4f"] = (1.0 - c1f) * 5000.0 * eta if hybrid else np.zeros(nz + 1)
    return out


def prp_geometry(nz=7, ny=8, nx=9, **overrides) -> PRPGeometry:
    base = dict(nz=nz, ny=ny, nx=nx, p_top=5000.0)
    base.update(overrides)
    return PRPGeometry(**base)


def prp_trajectory(geom: PRPGeometry, seed: int, dtype=np.float64, *,
                   hybrid: bool = False) -> dict:
    """Host base state of calc_p_rho_phi: a hydrostatic base (phb, alb,
    pb) built from mub on the hypsometric_opt 2 pressures, so both
    hypsometric options give small perturbations."""
    rng = np.random.default_rng(seed)
    sh = prp_shapes(geom)
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    out = _vertical(nz, rng, hybrid)
    for n in ("thp", "php", "mup", "mub"):
        c, s = _RK_SCALES[n]
        out[n] = c + s * rng.standard_normal(sh[n])
    out["muts"] = out["mub"] + out["mup"] + 20.0 * rng.standard_normal(sh["muts"])
    mb = out["mub"] if geom.base3d else np.full((1, 1), float(out["mub"].mean()))
    pf = out["c3f"][:, None, None] * mb + out["c4f"][:, None, None] + geom.p_top
    ph = out["c3h"][:, None, None] * mb + out["c4h"][:, None, None] + geom.p_top
    alb = RK_T0_RD * (1.0 + 0.05 * rng.standard_normal(ph.shape)) / ph
    dphb = alb * ph * np.log(pf[:-1] / pf[1:])
    phb = np.concatenate((np.zeros((1,) + ph.shape[1:]), np.cumsum(dphb, 0)))
    pb = 1.0e5 * (287.0 * 310.0 / (1.0e5 * alb)) ** 1.4
    if not geom.base3d:
        alb, phb, pb = alb[:, 0, 0], phb[:, 0, 0], pb[:, 0, 0]
    out.update(alb=alb, phb=phb, pb=pb)
    for n in PRP_STATE:
        c, s = _RK_SCALES[n]
        out[n] = c + s * rng.standard_normal(sh[n])
    keep = PRP_STATE + PRP_ACTIVE + PRP_PASSIVE
    return {k: np.ascontiguousarray(out[k], dtype=dtype) for k in keep}


RK_T0_RD = 287.0 * 300.0


def _rk_perturbation(names, shapes, seed, dtype):
    rng = np.random.default_rng(seed)
    out = {}
    for n in names:
        s = _RK_SCALES[n][1] if n != "muts" else 200.0
        out[n] = np.ascontiguousarray(s * rng.standard_normal(shapes[n]),
                                      dtype=dtype)
    return out


def prp_perturbation(geom: PRPGeometry, seed: int, dtype=np.float64) -> dict:
    return _rk_perturbation(PRP_STATE + PRP_ACTIVE, prp_shapes(geom), seed,
                            dtype)


def rsp_geometry(nz=7, ny=8, nx=9, **overrides) -> RSPGeometry:
    base = dict(nz=nz, ny=ny, nx=nx, rdx=1.0 / 3000.0, rdy=1.0 / 2500.0,
                boundary_x=1, boundary_y=1)
    base.update(overrides)
    return RSPGeometry(**base)


def _msf(out, rng, shapes, names, mapped):
    for n in names:
        out[n] = (1.0 + 0.02 * rng.standard_normal(shapes[n]) if mapped
                  else np.ones(shapes[n]))


def rsp_trajectory(geom: RSPGeometry, seed: int, dtype=np.float64, *,
                   hybrid: bool = False) -> dict:
    """Host base state of rk_step_prep; ``msfv`` is msfv_inv."""
    rng = np.random.default_rng(seed)
    sh = rsp_shapes(geom)
    out = _vertical(geom.nz, rng, hybrid)
    for n in RSP_STATE + RSP_ACTIVE + ("mub",):
        c, s = _RK_SCALES[n]
        out[n] = c + s * rng.standard_normal(sh[n])
    _msf(out, rng, sh, ("msfu", "msfv", "msft"), geom.has_msf)
    keep = RSP_STATE + RSP_ACTIVE + RSP_PASSIVE
    return {k: np.ascontiguousarray(out[k], dtype=dtype) for k in keep}


def rsp_perturbation(geom: RSPGeometry, seed: int, dtype=np.float64) -> dict:
    return _rk_perturbation(RSP_STATE + RSP_ACTIVE, rsp_shapes(geom), seed,
                            dtype)


def at_geometry(nz=7, ny=8, nx=9, **overrides) -> ATGeometry:
    base = dict(nz=nz, ny=ny, nx=nx)
    base.update(overrides)
    return ATGeometry(**base)


def at_trajectory(geom: ATGeometry, seed: int, dtype=np.float64, *,
                  hybrid: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    sh = at_shapes(geom)
    out = _vertical(geom.nz, rng, hybrid)
    for n in AT_STATE + AT_ACTIVE + ("mub",):
        c, s = _RK_SCALES[n]
        out[n] = c + s * rng.standard_normal(sh[n])
    _msf(out, rng, sh, ("msft",), geom.has_msf)
    keep = AT_STATE + AT_ACTIVE + AT_PASSIVE
    return {k: np.ascontiguousarray(out[k], dtype=dtype) for k in keep}


def at_perturbation(geom: ATGeometry, seed: int, dtype=np.float64) -> dict:
    return _rk_perturbation(AT_STATE + AT_ACTIVE, at_shapes(geom), seed, dtype)


def sw_geometry(nz=7, ny=8, nx=9, **overrides) -> SWGeometry:
    base = dict(nz=nz, ny=ny, nx=nx, cf1=1.6, cf2=-0.75, cf3=0.15,
                half_rdx=0.5 / 3000.0, half_rdy=0.5 / 2500.0,
                boundary_x=1, boundary_y=1)
    base.update(overrides)
    return SWGeometry(**base)


def sw_trajectory(geom: SWGeometry, seed: int, dtype=np.float64) -> dict:
    rng = np.random.default_rng(seed)
    sh = sw_shapes(geom)
    out = {}
    for n in ("w", "u", "v"):
        c, s = _RK_SCALES[n]
        out[n] = c + s * rng.standard_normal(sh[n])
    out["ht"] = 500.0 + 300.0 * rng.standard_normal(sh["ht"])
    _msf(out, rng, sh, ("msft",), geom.mapped)
    return {k: np.ascontiguousarray(v, dtype=dtype) for k, v in out.items()}


def sw_perturbation(geom: SWGeometry, seed: int, dtype=np.float64) -> dict:
    return _rk_perturbation(("w", "u", "v"), sw_shapes(geom), seed, dtype)


# One RK stage (woof.grad.wrfplus.stage.rk_stage_tl): periodic, msf 1, the
# glue operators' draws under the stage's names, then small_step's,
# calc_coef_w's and the substep's; calc_p_rho_phi's hydrostatic base
# (phb, alb, pb; 3-D, as advance_uv's pb always is) and carried muts are
# kept.
def stage_geometry(nz=7, ny=8, nx=9, *, rk_step=2, nsub=2, top_lid=0,
                   hypso=1) -> StageGeometry:
    sub = substep_geometries(nz, ny, nx, top_lid=top_lid)
    uvg = sub[0]
    return StageGeometry(
        prp=prp_geometry(nz, ny, nx, hypso=hypso),
        rsp=rsp_geometry(nz, ny, nx, boundary_x=0, boundary_y=0,
                         rdx=uvg.rdx, rdy=uvg.rdy),
        at=at_geometry(nz, ny, nx),
        ss=ss_geometry(nz, ny, nx, boundary_x=0, boundary_y=0, rk_step=rk_step),
        cw=cw_geometry(nz, ny, nx, top_lid=top_lid),
        substep=sub, nsub=nsub)


def _aliased(d, alias):
    return {alias.get(n, n): v for n, v in d.items()}


def stage_trajectory(geom: StageGeometry, seed: int, dtype=np.float64, *,
                     hybrid: bool = False) -> dict:
    prp = _aliased(prp_trajectory(geom.prp, seed + 3, dtype, hybrid=hybrid),
                   PRP_ALIAS)
    out = dict(prp)
    out.update(_aliased(rsp_trajectory(geom.rsp, seed + 4, dtype, hybrid=hybrid),
                        RSP_ALIAS))
    out.update(_aliased(at_trajectory(geom.at, seed + 5, dtype, hybrid=hybrid),
                        AT_ALIAS))
    out.update(ss_trajectory(geom.ss, seed + 6, dtype, hybrid=hybrid))
    out.update(cw_trajectory(geom.cw, seed + 7, dtype, hybrid=hybrid))
    out.update(substep_trajectory(geom.substep, seed, dtype, hybrid=hybrid))
    for n in ("phb", "alb", "pb", "muts", "c3h", "c4h", "c3f", "c4f"):
        out[n] = prp[n]
    out["dnw"] = np.ascontiguousarray(1.0 / out["rdnw"], dtype=dtype)
    return out


def stage_active(geom: StageGeometry) -> tuple:
    """Every name one stage reads or writes that is perturbed."""
    names = (tuple(PRP_ALIAS.get(n, n) for n in PRP_STATE + PRP_ACTIVE)
             + tuple(RSP_ALIAS.get(n, n) for n in RSP_STATE + RSP_ACTIVE)
             + AT_STATE + AT_ACTIVE + SI_STATE + SI_ACTIVE + CW_STATE
             + CW_ACTIVE + substep_active(geom.substep) + SF_STATE + SF_ACTIVE)
    return tuple(dict.fromkeys(names))


def stage_perturbation(geom: StageGeometry, seed: int, dtype=np.float64) -> dict:
    out = _aliased(prp_perturbation(geom.prp, seed + 3, dtype), PRP_ALIAS)
    out.update(_aliased(rsp_perturbation(geom.rsp, seed + 4, dtype), RSP_ALIAS))
    out.update(at_perturbation(geom.at, seed + 5, dtype))
    out.update(ss_perturbation(geom.ss, seed + 6, dtype))
    out.update(cw_perturbation(geom.cw, seed + 7, dtype))
    out.update(substep_perturbation(geom.substep, seed, dtype))
    return {n: out[n] for n in stage_active(geom)}


# ----------------------------------------------------------------- advection

def adv_geometry(kind="scalar", nz=7, ny=8, nx=9, **overrides):
    from .advect import AdvGeometry
    base = dict(kind=kind, nz=nz, ny=ny, nx=nx, rdx=1.0 / 3000.0, rdy=1.0 / 2500.0)
    base.update(overrides)
    return AdvGeometry(**base)


_ADV_SCALES = {"tend": (0.0, 1e-3), "q": (0.0, 1.0), "ru": (0.0, 2.0),
               "rv": (0.0, 2.0), "rw": (0.0, 0.5), "msf": (1.0, 0.02)}


def adv_trajectory(geom, seed: int, dtype=np.float64, *, hybrid: bool = False) -> dict:
    """Host trajectory (q, ru, rv, rw, msf, rdz, fnm, fnp) and the tendency
    start.  ``hybrid`` stretches the levels (fnm != fnp)."""
    from .advect import adv_shapes
    rng = np.random.default_rng(seed)
    shapes = adv_shapes(geom)
    out = {}
    for name, (c, s) in _ADV_SCALES.items():
        out[name] = c + s * rng.standard_normal(shapes[name])
    nz = geom.nz
    eta = np.linspace(1.0, 0.0, nz + 1)
    if hybrid:
        eta = eta ** 1.3
    dnw = np.diff(eta)
    znu = 0.5 * (eta[1:] + eta[:-1])
    dn = np.zeros(nz)
    dn[1:] = np.diff(znu)
    fnm = np.zeros(nz)
    fnp = np.zeros(nz)
    fnm[1:] = 0.5 * dnw[:-1] / dn[1:]
    fnp[1:] = 0.5 * dnw[1:] / dn[1:]
    if geom.kind == "w":
        rdz = np.zeros(nz)
        rdz[1:] = 1.0 / dn[1:]
        rdz[0] = rdz[1] if nz > 1 else 1.0
    else:
        rdz = 1.0 / dnw
    out.update(rdz=rdz, fnm=fnm, fnp=fnp)
    return {n: np.ascontiguousarray(v, dtype=dtype) for n, v in out.items()}


def adv_perturbation(geom, seed: int, dtype=np.float64) -> dict:
    from .advect import ADV_ACTIVE, adv_shapes
    rng = np.random.default_rng(seed)
    shapes = adv_shapes(geom)
    return {n: np.ascontiguousarray(_ADV_SCALES[n][1] * rng.standard_normal(shapes[n]),
                                    dtype=dtype) for n in ADV_ACTIVE}
