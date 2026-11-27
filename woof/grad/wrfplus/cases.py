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
                    prp_active, prp_shapes, rsp_shapes, sw_shapes)

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
    _msf(out, rng, shapes, ("msft",), geom.has_msf)   # last: earlier draws unchanged
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
# off-centring, so the three operators see one consistent state.  With
# ``emdiv`` advance_uv adds the external-mode filter of the mudf the
# previous substep's advance_mu_t wrote.
def substep_geometries(nz=7, ny=8, nx=9, *, base3d=1, top_lid=0,
                       smdiv=0.1, emdiv=0.0) -> tuple:
    uvg = uv_geometry(nz, ny, nx, base3d=base3d, top_lid=top_lid, smdiv=smdiv)
    if emdiv:
        uvg = uv_geometry(nz, ny, nx, base3d=base3d, top_lid=top_lid, smdiv=smdiv,
                          use_emdiv=1, xscale=-emdiv / uvg.rdx, yscale=-emdiv / uvg.rdy)
    return (uvg,
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
    "ww": (0.0, 10.0), "qv": (8e-3, 1e-3),
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
    out["c1f"] = c1f
    out["c2f"] = (1.0 - c1f) * 5000.0 if hybrid else np.zeros(nz + 1)
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
    if geom.moist:
        out["qv"] = np.abs(8e-3 + 3e-3 * rng.standard_normal(sh["qv"]))
        keep += ("qv",)
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
    return _rk_perturbation(PRP_STATE + prp_active(geom), prp_shapes(geom), seed,
                            dtype)


def prpm_geometry(nz=7, ny=8, nx=9, **overrides) -> PRPGeometry:
    """The moist diagnosis (qv active)."""
    return prp_geometry(nz, ny, nx, **dict(overrides, moist=1))


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


# ------------------------------------------------------- big-step tendencies
# horizontal_pressure_gradient and pg_buoy_w (P3a item 5).

_BS_SCALES = {"ru_t": (0.0, 50.0), "rv_t": (0.0, 50.0), "rw_t": (0.0, 5.0),
              "p": (0.0, 50.0), "al": (0.0, 1e-4), "alt": (1.0, 0.2),
              "php": (0.0, 20.0), "mup": (0.0, 200.0), "mub": (9.0e4, 2e3)}


def pgf_geometry(nz=7, ny=8, nx=9, **overrides):
    from .bigstep import PGFGeometry
    base = dict(nz=nz, ny=ny, nx=nx, rdx=1.0 / 3000.0, rdy=1.0 / 2500.0, cf1=1.6,
                cf2=-0.8, cf3=0.2, cfn=1.5, cfn1=-0.5)
    base.update(overrides)
    return PGFGeometry(**base)


def _bs_levels(nz, rng, hybrid):
    v = _vertical(nz, rng, hybrid)
    znu, dnw = v["znu"], v["dnw"]
    dn = np.zeros(nz)
    dn[1:] = np.diff(znu)
    fnm, fnp, rdn = np.zeros(nz), np.zeros(nz), np.zeros(nz)
    fnm[1:] = 0.5 * dnw[:-1] / dn[1:]
    fnp[1:] = 0.5 * dnw[1:] / dn[1:]
    rdn[1:] = 1.0 / dn[1:]
    c1f = np.sqrt(np.clip(v["znw"], 0.0, None)) if hybrid else np.ones(nz + 1)
    return dict(c1h=v["c1h"], c2h=v["c2h"], rdnw=v["rdnw"], fnm=fnm, fnp=fnp, rdn=rdn,
                c1f=c1f)


def pgf_trajectory(geom, seed: int, dtype=np.float64, *, hybrid: bool = False) -> dict:
    """Host state of slow_pgf: p' and al small, alt near 1, a base pressure
    falling and a base geopotential rising with height; ``hybrid`` gives
    hybrid c1h/c2h and stretched levels."""
    from .bigstep import PGF_ACTIVE, PGF_PASSIVE, PGF_STATE, pgf_shapes
    rng = np.random.default_rng(seed)
    sh = pgf_shapes(geom)
    out = _bs_levels(geom.nz, rng, hybrid)
    for n in PGF_STATE + PGF_ACTIVE + ("mub",):
        c, s = _BS_SCALES[n]
        out[n] = c + s * rng.standard_normal(sh[n])
    k = np.arange(geom.nz + 1)[:, None, None]
    out["pb"] = 9.0e4 - 6.0e3 * k[:-1] + 300.0 * rng.standard_normal(sh["pb"])
    out["phb"] = 4.0e3 * k + 50.0 * rng.standard_normal(sh["phb"])
    keep = PGF_STATE + PGF_ACTIVE + PGF_PASSIVE
    if getattr(geom, "moist_cq", 0):
        # calc_cq's range: 1/(1 + 0.5 qtot) for qtot of a few g/kg
        for n in ("cqu", "cqv"):
            out[n] = 0.99 + 0.004 * rng.standard_normal(sh[n])
        keep += ("cqu", "cqv")
    return {n: np.ascontiguousarray(out[n], dtype=dtype) for n in keep}


def pgf_perturbation(geom, seed: int, dtype=np.float64) -> dict:
    from .bigstep import pgf_active, pgf_shapes
    rng = np.random.default_rng(seed)
    sh = pgf_shapes(geom)
    scale = {"p": 50.0, "al": 1e-4, "alt": 0.05, "php": 20.0, "mup": 200.0,
             "cqu": 0.004, "cqv": 0.004}
    return {n: np.ascontiguousarray(scale[n] * rng.standard_normal(sh[n]), dtype=dtype)
            for n in pgf_active(geom)}


def pgfm_geometry(nz=7, ny=8, nx=9, **overrides):
    """The moist pressure gradient (cqu, cqv active)."""
    return pgf_geometry(nz, ny, nx, **dict(overrides, moist_cq=1))


def buoy_geometry(nz=7, ny=8, nx=9, **overrides):
    from .bigstep import BuoyGeometry
    base = dict(nz=nz, ny=ny, nx=nx)
    base.update(overrides)
    return BuoyGeometry(**base)


def buoy_trajectory(geom, seed: int, dtype=np.float64, *, hybrid: bool = False) -> dict:
    from .bigstep import BUOY_ACTIVE, BUOY_PASSIVE, BUOY_STATE, buoy_shapes
    rng = np.random.default_rng(seed)
    sh = buoy_shapes(geom)
    out = _bs_levels(geom.nz, rng, hybrid)
    for n in BUOY_STATE + BUOY_ACTIVE:
        c, s = _BS_SCALES[n]
        out[n] = c + s * rng.standard_normal(sh[n])
    out["msft"] = 1.0 + 0.02 * rng.standard_normal(sh["msft"])
    keep = BUOY_STATE + BUOY_ACTIVE + BUOY_PASSIVE
    if getattr(geom, "moist", 0):
        out.update(_moist_species(sh["qv"], rng))
        out["mub"] = 9.0e4 + 2e3 * rng.standard_normal(sh["mub"])
        # hybrid levels: c2f carries the base mass c1f leaves
        out["c2f"] = (1.0 - out["c1f"]) * 5.0e4
        keep += ("qv", "qc", "qr", "mub", "c2f")
    return {n: np.ascontiguousarray(out[n], dtype=dtype) for n in keep}


_Q_SCALES = {"qv": (8e-3, 3e-3), "qc": (4e-4, 3e-4), "qr": (3e-4, 2e-4)}


def _moist_species(shape, rng) -> dict:
    """qv, qc and qr of a cloudy column, all positive."""
    return {n: np.abs(c + s * rng.standard_normal(shape)) for n, (c, s) in _Q_SCALES.items()}


def buoy_perturbation(geom, seed: int, dtype=np.float64) -> dict:
    from .bigstep import buoy_active, buoy_shapes
    rng = np.random.default_rng(seed)
    sh = buoy_shapes(geom)
    scale = {"p": 50.0, "mup": 200.0, "qv": 1e-3, "qc": 1e-4, "qr": 1e-4}
    return {n: np.ascontiguousarray(scale[n] * rng.standard_normal(sh[n]), dtype=dtype)
            for n in buoy_active(geom)}


def buoym_geometry(nz=7, ny=8, nx=9, **overrides):
    """The moist buoyancy (moist_mode 1: qv, qc and qr active)."""
    return buoy_geometry(nz, ny, nx, **dict(overrides, moist=1))


# calc_cq (P3a DA item 4).

def cq_geometry(nz=7, ny=8, nx=9, **overrides):
    from .bigstep import CQGeometry
    base = dict(nz=nz, ny=ny, nx=nx)
    base.update(overrides)
    return CQGeometry(**base)


def cq_trajectory(geom, seed: int, dtype=np.float64, *, hybrid: bool = False) -> dict:
    """qv, qc and qr, and cqu, cqv and cqw holding stale values (calc_cq
    overwrites them).  ``hybrid`` is accepted for the common signature."""
    from .bigstep import CQ_STATE, cq_shapes
    rng = np.random.default_rng(seed)
    sh = cq_shapes(geom)
    out = _moist_species(sh["qv"], rng)
    for n in CQ_STATE:
        out[n] = 5.0 + rng.standard_normal(sh[n])
    return {n: np.ascontiguousarray(v, dtype=dtype) for n, v in out.items()}


def cq_perturbation(geom, seed: int, dtype=np.float64) -> dict:
    """Perturbations of all three species (n_mass 1 reads only qv's), at
    5 g/kg: cq moves by about 0.5 dq, so a smaller dq leaves the binary64
    finite differences at a = 1e-7 in roundoff."""
    from .bigstep import CQ_SPECIES, cq_shapes
    rng = np.random.default_rng(seed)
    sh = cq_shapes(geom)
    return {n: np.ascontiguousarray(5e-3 * rng.standard_normal(sh[n]), dtype=dtype)
            for n in CQ_SPECIES}


# rhs_ph (P3a item 5).

_GEO_SCALES = {"rph_t": (0.0, 5.0), "ww": (0.0, 5.0), "w": (0.0, 1.0), "u": (5.0, 10.0),
               "v": (-3.0, 10.0), "php": (0.0, 20.0), "mup": (0.0, 200.0),
               "mub": (9.0e4, 2e3)}


def geo_geometry(nz=7, ny=8, nx=9, **overrides):
    from .bigstep import GeoGeometry
    base = dict(nz=nz, ny=ny, nx=nx, quarter_rdx=0.25 / 3000.0, quarter_rdy=0.25 / 2500.0,
                cfn=1.5, cfn1=-0.5)
    base.update(overrides)
    return GeoGeometry(**base)


def geo_trajectory(geom, seed: int, dtype=np.float64, *, hybrid: bool = False) -> dict:
    """Host state of slow_geopotential: winds with a mean flow, a base
    geopotential rising with height, map factors near 1; ``hybrid`` gives
    hybrid c1f/c2f and stretched levels."""
    from .bigstep import GEO_ACTIVE, GEO_PASSIVE, GEO_STATE, geo_shapes
    rng = np.random.default_rng(seed)
    sh = geo_shapes(geom)
    out = _bs_levels(geom.nz, rng, hybrid)
    out["c2f"] = (1.0 - out["c1f"]) * 5000.0 if hybrid else np.zeros(geom.nz + 1)
    for n in GEO_STATE + GEO_ACTIVE + ("mub",):
        c, s = _GEO_SCALES[n]
        out[n] = c + s * rng.standard_normal(sh[n])
    k = np.arange(geom.nz + 1)[:, None, None]
    out["phb"] = 4.0e3 * k + 50.0 * rng.standard_normal(sh["phb"])
    for n in ("msft", "msfu", "msfv"):
        out[n] = 1.0 + 0.02 * rng.standard_normal(sh[n])
    keep = GEO_STATE + GEO_ACTIVE + GEO_PASSIVE
    return {n: np.ascontiguousarray(out[n], dtype=dtype) for n in keep}


def geo_perturbation(geom, seed: int, dtype=np.float64) -> dict:
    from .bigstep import GEO_ACTIVE, geo_shapes
    rng = np.random.default_rng(seed)
    sh = geo_shapes(geom)
    scale = {"ww": 5.0, "w": 1.0, "u": 10.0, "v": 10.0, "php": 20.0, "mup": 200.0}
    return {n: np.ascontiguousarray(scale[n] * rng.standard_normal(sh[n]), dtype=dtype)
            for n in GEO_ACTIVE}


_CC_SCALES = {"ru_t": (0.0, 50.0), "rv_t": (0.0, 50.0), "rw_t": (0.0, 20.0),
              "ru": (9.0e5, 9.0e5), "rv": (-3.0e5, 9.0e5), "u": (10.0, 10.0),
              "v": (-3.0, 10.0), "w": (0.0, 1.0), "mut": (9.0e4, 2e3)}


def cc_geometry(nz=7, ny=8, nx=9, **overrides):
    from .bigstep import CCGeometry
    base = dict(nz=nz, ny=ny, nx=nx, rdx=1.0 / 3000.0, rdy=1.0 / 2500.0)
    base.update(overrides)
    return CCGeometry(**base)


def cc_trajectory(geom, seed: int, dtype=np.float64, *, hybrid: bool = False) -> dict:
    """Host state of coriolis_curvature: coupled momenta of mass x wind,
    Coriolis parameters of mid latitudes, a rotated grid (sina up to
    about 0.3) and map factors near 1; ``hybrid`` gives hybrid c1f/c2f
    and stretched levels."""
    from .bigstep import CC_ACTIVE, CC_PASSIVE, CC_STATE, cc_shapes
    rng = np.random.default_rng(seed)
    sh = cc_shapes(geom)
    out = _bs_levels(geom.nz, rng, hybrid)
    out["c2f"] = (1.0 - out["c1f"]) * 5000.0 if hybrid else np.zeros(geom.nz + 1)
    for n in CC_STATE + CC_ACTIVE:
        c, s = _CC_SCALES[n]
        out[n] = c + s * rng.standard_normal(sh[n])
    lat = np.deg2rad(40.0 + 5.0 * rng.standard_normal(sh["f"]))
    out["f"] = 2.0 * 7.292e-5 * np.sin(lat)
    out["e"] = 2.0 * 7.292e-5 * np.cos(lat)
    alpha = 0.3 * rng.uniform(-1.0, 1.0, sh["f"])
    out["sina"], out["cosa"] = np.sin(alpha), np.cos(alpha)
    for n in ("msft", "msfu", "msfv"):
        out[n] = 1.0 + 0.02 * rng.standard_normal(sh[n])
    keep = CC_STATE + CC_ACTIVE + CC_PASSIVE
    return {n: np.ascontiguousarray(out[n], dtype=dtype) for n in keep}


def cc_perturbation(geom, seed: int, dtype=np.float64) -> dict:
    from .bigstep import CC_ACTIVE, cc_shapes
    rng = np.random.default_rng(seed)
    sh = cc_shapes(geom)
    scale = {"ru": 9.0e5, "rv": 9.0e5, "u": 10.0, "v": 10.0, "w": 1.0, "mut": 200.0}
    return {n: np.ascontiguousarray(scale[n] * rng.standard_normal(sh[n]), dtype=dtype)
            for n in CC_ACTIVE}


def wd_geometry(nz=7, ny=8, nx=9, **overrides):
    from .bigstep import WDampGeometry
    base = dict(nz=nz, ny=ny, nx=nx, dt=20.0)
    base.update(overrides)
    return WDampGeometry(**base)


def wd_trajectory(geom, seed: int, dtype=np.float64, *, hybrid: bool = False) -> dict:
    """Host state of w_damp: an eta mass flux ww whose vertical Courant
    number spreads over 0-2.5, so about half the points are damped; no
    point lies within 2% of the onset (so the switch does not move under
    the FD perturbation or a binary32 rounding)."""
    from .bigstep import WD_ACTIVE, WD_PASSIVE, WD_STATE, wd_shapes
    rng = np.random.default_rng(seed)
    sh = wd_shapes(geom)
    out = _bs_levels(geom.nz, rng, hybrid)
    out["c2f"] = (1.0 - out["c1f"]) * 5000.0 if hybrid else np.zeros(geom.nz + 1)
    out["mub"] = 9.0e4 + 2e3 * rng.standard_normal(sh["mub"])
    out["mup"] = 200.0 * rng.standard_normal(sh["mup"])
    out["rw_t"] = 20.0 * rng.standard_normal(sh["rw_t"])
    out["w"] = rng.standard_normal(sh["w"])
    m = out["c1f"][:, None, None] * (out["mub"] + out["mup"]) + out["c2f"][:, None, None]
    scale = np.abs(m / (np.concatenate([out["rdnw"], [out["rdnw"][-1]]])[:, None, None]
                        * geom.dt))
    cfl = rng.uniform(0.0, 2.5, sh["ww"])
    for onset in (geom.w_damp_on, geom.w_crit_cfl):
        near = np.abs(cfl - onset) < 0.02 * onset
        cfl[near] += 0.05 * onset
    out["ww"] = np.where(rng.random(sh["ww"]) < 0.5, -1.0, 1.0) * cfl * scale
    keep = WD_STATE + WD_ACTIVE + WD_PASSIVE
    return {n: np.ascontiguousarray(out[n], dtype=dtype) for n in keep}


def wd_perturbation(geom, seed: int, dtype=np.float64) -> dict:
    from .bigstep import WD_ACTIVE, wd_shapes
    rng = np.random.default_rng(seed)
    sh = wd_shapes(geom)
    scale = {"ww": 50.0, "mup": 200.0}
    return {n: np.ascontiguousarray(scale[n] * rng.standard_normal(sh[n]), dtype=dtype)
            for n in WD_ACTIVE}


# The source field's (centre, spread) per row kind (u, v, theta', ph', w, mu').
_LB_SRC = {"u": (5.0, 8.0), "v": (-2.0, 8.0), "theta": (0.0, 2.0), "phi": (0.0, 100.0),
           "w": (0.0, 0.5), "mu": (0.0, 200.0), "qv": (0.01, 0.003)}


def lbc_geometry(nz=4, ny=12, nx=13, kind="u", **overrides):
    from .lbc import LBCGeometry
    base = dict(nz=nz, ny=ny, nx=nx, kind=kind, width=5, spec_zone=1, relax_zone=4,
                dtbc=600.0)
    base.update(overrides)
    return LBCGeometry(**base)


def lbc_weights(width, spec_zone, relax_zone, dt=20.0):
    """WRF lbc_fcx_gcx's specified weights (spec_exp 0)."""
    fcx, gcx = np.zeros(width), np.zeros(width)
    for loop in range(spec_zone + 1, spec_zone + relax_zone + 1):
        if loop - 1 >= width:
            break
        ramp = (spec_zone + relax_zone - loop) / max(relax_zone - 1, 1)
        fcx[loop - 1] = 0.1 / dt * ramp
        gcx[loop - 1] = 1.0 / dt / 50.0 * ramp
    return fcx, gcx


def lbc_trajectory(geom, seed: int, dtype=np.float64, *, hybrid: bool = False) -> dict:
    """Host state of one relaxation row: the row's field, mu', a base
    column mass near 9e4, hybrid levels (``hybrid``) and side tables
    holding the coupled field (mass times the field) a few per cent off
    the current one, with tendencies over an hour."""
    from .lbc import LB_ACTIVE, LB_PASSIVE, LB_STATE, LB_TABLES, lb_shapes
    rng = np.random.default_rng(seed)
    sh = lb_shapes(geom)
    out = _bs_levels(geom.nz, rng, hybrid)
    out["c2f"] = (1.0 - out["c1f"]) * 5000.0 if hybrid else np.zeros(geom.nz + 1)
    out["mub"] = 9.0e4 + 2e3 * rng.standard_normal(sh["mub"])
    out["mup"] = 200.0 * rng.standard_normal(sh["mup"])
    c, s = _LB_SRC[geom.kind]
    out["src"] = c + s * rng.standard_normal(sh["src"])
    for n in ("msft", "msfu", "msfv"):
        out[n] = 1.0 + 0.02 * rng.standard_normal(sh[n])
    out["fcx"], out["gcx"] = lbc_weights(geom.width, geom.spec_zone, geom.relax_zone)
    scale = 1.0 if geom.kind == "mu" else 9.0e4
    for n in LB_TABLES:
        if n.endswith("_t"):
            out[n] = (scale * s / 3600.0) * rng.standard_normal(sh[n])
        else:
            out[n] = scale * (c + s * rng.standard_normal(sh[n]))
    out["tend"] = 5e-2 * scale * rng.standard_normal(sh["tend"])
    out["held"] = 5e-2 * scale * rng.standard_normal(sh["held"])
    keep = LB_STATE + LB_ACTIVE + LB_PASSIVE
    return {n: np.ascontiguousarray(out[n], dtype=dtype) for n in keep}


def lbc_perturbation(geom, seed: int, dtype=np.float64) -> dict:
    from .lbc import LB_ACTIVE, lb_shapes
    rng = np.random.default_rng(seed)
    sh = lb_shapes(geom)
    c, s = _LB_SRC[geom.kind]
    scale = 1.0 if geom.kind == "mu" else 9.0e4
    if geom.kind == "qv":
        # qv's coupled row is ~1e3 against tables of the same size: a
        # larger step keeps binary64 roundoff out of the 1e-7 FD point
        s *= 10.0
    out = {"src": 0.1 * s * rng.standard_normal(sh["src"]),
           "mup": 20.0 * rng.standard_normal(sh["mup"])}
    for n in LB_ACTIVE[2:]:
        t = 1.0 / 60.0 if n.endswith("_t") else 1.0
        out[n] = 0.1 * scale * s * t * rng.standard_normal(sh[n])
    return {n: np.ascontiguousarray(out[n], dtype=dtype) for n in LB_ACTIVE}


def lf_geometry(nz=4, ny=10, nx=11, kind="u", **overrides):
    from .lbc import FinalGeometry
    base = dict(nz=nz, ny=ny, nx=nx, kind=kind, width=5, spec_zone=1, dtbc=600.0)
    base.update(overrides)
    return FinalGeometry(**base)


def lf_trajectory(geom, seed: int, dtype=np.float64, *, hybrid: bool = False) -> dict:
    """Host state of one spec_bdy_final field: the field, an acoustic muts
    near 9e4, hybrid levels (``hybrid``), map factors near 1 and side
    tables holding the coupled field (mass times the field), with
    tendencies over an hour."""
    from .lbc import LB_TABLES, LF_ACTIVE, LF_PASSIVE, LF_STATE, lf_shapes
    rng = np.random.default_rng(seed)
    sh = lf_shapes(geom)
    out = _bs_levels(geom.nz, rng, hybrid)
    out["c2f"] = (1.0 - out["c1f"]) * 5000.0 if hybrid else np.zeros(geom.nz + 1)
    out["muts"] = 9.0e4 + 2e3 * rng.standard_normal(sh["muts"])
    for n in ("msft", "msfu", "msfv"):
        out[n] = 1.0 + 0.02 * rng.standard_normal(sh[n])
    c, s = _LB_SRC[geom.kind]
    out["target"] = c + s * rng.standard_normal(sh["target"])
    scale = 1.0 if geom.kind == "mu" else 9.0e4
    for n in LB_TABLES:
        if n.endswith("_t"):
            out[n] = (scale * s / 3600.0) * rng.standard_normal(sh[n])
        else:
            out[n] = scale * (c + s * rng.standard_normal(sh[n]))
    keep = LF_STATE + LF_ACTIVE + LF_PASSIVE
    return {n: np.ascontiguousarray(out[n], dtype=dtype) for n in keep}


def lf_perturbation(geom, seed: int, dtype=np.float64) -> dict:
    from .lbc import LB_TABLES, LF_ACTIVE, lf_shapes
    rng = np.random.default_rng(seed)
    sh = lf_shapes(geom)
    _, s = _LB_SRC[geom.kind]
    scale = 1.0 if geom.kind == "mu" else 9.0e4
    out = {"muts": 20.0 * rng.standard_normal(sh["muts"])}
    for n in LB_TABLES:
        t = 1.0 / 60.0 if n.endswith("_t") else 1.0
        out[n] = 0.1 * scale * s * t * rng.standard_normal(sh[n])
    return {n: np.ascontiguousarray(out[n], dtype=dtype) for n in LF_ACTIVE}


def sp_geometry(nz=4, ny=8, nx=9, **overrides):
    from .lbc import SpecPhiGeometry
    base = dict(nz=nz, ny=ny, nx=nx, dtau=4.0, spec_zone=1)
    base.update(overrides)
    return SpecPhiGeometry(**base)


def sp_trajectory(geom, seed: int, dtype=np.float64, *, hybrid: bool = False) -> dict:
    """Host state of one specified-frame substep: phi'' and w'' small, a
    coupled phi tendency, muts near 9e4 with a small mass tendency, the
    coupled theta'' near mu'' theta', alt near 1 and c2a near gamma p /
    alpha; ``hybrid`` gives hybrid levels."""
    from .lbc import SP_ACTIVE, SP_PASSIVE, SP_STATE, sp_shapes
    rng = np.random.default_rng(seed)
    sh = sp_shapes(geom)
    out = _bs_levels(geom.nz, rng, hybrid)
    out["c2f"] = (1.0 - out["c1f"]) * 5000.0 if hybrid else np.zeros(geom.nz + 1)
    n = rng.standard_normal
    out.update(ph_pp=10.0 * n(sh["ph_pp"]), w_pp=0.5 * n(sh["w_pp"]),
               p_pp=50.0 * n(sh["p_pp"]), al_pp=1e-3 * n(sh["al_pp"]),
               rph_t=9.0e4 * n(sh["rph_t"]), php=100.0 * n(sh["php"]),
               rmu_t=5.0 * n(sh["rmu_t"]), muts=9.0e4 + 2e3 * n(sh["muts"]),
               mu_pp=50.0 * n(sh["mu_pp"]), thp=2.0 * n(sh["thp"]),
               alt=1.0 + 0.1 * n(sh["alt"]), c2a=1.4e5 * (1.0 + 0.05 * n(sh["c2a"])))
    out["th_pp"] = out["mu_pp"][None] * out["thp"] + 50.0 * n(sh["th_pp"])
    keep = SP_STATE + SP_ACTIVE + SP_PASSIVE
    return {k: np.ascontiguousarray(out[k], dtype=dtype) for k in keep}


def sp_perturbation(geom, seed: int, dtype=np.float64) -> dict:
    from .lbc import SP_ACTIVE, sp_shapes
    rng = np.random.default_rng(seed)
    sh = sp_shapes(geom)
    scale = dict(rph_t=9.0e3, php=10.0, rmu_t=0.5, muts=20.0, th_pp=5.0, mu_pp=5.0,
                 thp=0.2, alt=0.01, c2a=1.4e3)
    return {k: np.ascontiguousarray(scale[k] * rng.standard_normal(sh[k]), dtype=dtype)
            for k in SP_ACTIVE}


# ------------------------------------------------- moist scalar transport

_SU_SCALES = {"q": (0.01, 0.002), "q0": (0.01, 0.002), "tend": (0.0, 1.0),
              "ru_m": (2e5, 4e5), "rv_m": (-1e5, 4e5), "ww_m": (0.0, 10.0),
              "mup0": (0.0, 200.0), "mup": (0.0, 200.0), "mub2d": (9.0e4, 2e3),
              "held": (0.0, 0.05), "msft": (1.0, 0.02)}


def su_geometry(nz=7, ny=8, nx=9, **overrides):
    """A species' stage update; ``spec`` alone turns on both bounded axes
    with a one-point ring, as a specified domain runs it."""
    from .scalar import ScalarGeometry
    base = dict(nz=nz, ny=ny, nx=nx, rdx=1.0 / 3000.0, rdy=1.0 / 2500.0, dt=12.0)
    if overrides.get("spec"):
        base.update(open_x=1, open_y=1, spec_zone=1)
    base.update(overrides)
    return ScalarGeometry(**base)


def su_trajectory(geom, seed: int, dtype=np.float64, *, hybrid: bool = False) -> dict:
    """Host base state of the update: the species and its time-t copy,
    the sumflux fluxes, both masses, the levels, msft and the held
    boundary tendency."""
    from .scalar import SU_PASSIVE, SU_STATE, su_active, su_shapes
    rng = np.random.default_rng(seed)
    sh = su_shapes(geom)
    lev = _vertical(geom.nz, rng, hybrid)
    adv = adv_trajectory(geom.adv(), seed + 1, hybrid=hybrid)
    out = {"c1h": lev["c1h"], "c2h": lev["c2h"], "rdnw": adv["rdz"],
           "fnm": adv["fnm"], "fnp": adv["fnp"]}
    for n in SU_STATE + su_active(geom) + SU_PASSIVE:
        if n not in out:
            c, s = _SU_SCALES[n]
            out[n] = c + s * rng.standard_normal(sh[n])
    if not geom.has_msf:   # the advection then takes msft of ones
        out["msft"] = np.ones(sh["msft"])
    return {k: np.ascontiguousarray(v, dtype=dtype) for k, v in out.items()}


def su_perturbation(geom, seed: int, dtype=np.float64) -> dict:
    from .scalar import su_active, su_shapes
    rng = np.random.default_rng(seed)
    sh = su_shapes(geom)
    return {n: np.ascontiguousarray(_SU_SCALES[n][1] * rng.standard_normal(sh[n]),
                                    dtype=dtype) for n in su_active(geom)}


def sf_geometry(nz=7, ny=8, nx=9, **overrides):
    from .scalar import SumfluxGeometry
    base = dict(nz=nz, ny=ny, nx=nx, nsub=4)
    base.update(overrides)
    return SumfluxGeometry(**base)


_SF_SCALES = {"ru_m": 4e5, "rv_m": 4e5, "ww_m": 10.0, "u_pp": 4e4, "v_pp": 4e4,
              "ww_pp": 2.0, "ru": 4e5, "rv": 4e5, "ww": 10.0}


def sf_trajectory(geom, seed: int, dtype=np.float64, *, hybrid: bool = False) -> dict:
    from .scalar import sumflux_shapes
    rng = np.random.default_rng(seed)
    return {n: np.ascontiguousarray(s * rng.standard_normal(sumflux_shapes(geom)[n]),
                                    dtype=dtype) for n, s in _SF_SCALES.items()}


def sf_perturbation(geom, seed: int, dtype=np.float64) -> dict:
    return sf_trajectory(geom, seed, dtype)


def flow_geometry(nz=3, ny=7, nx=8, **overrides):
    from .lbc import FlowGeometry
    base = dict(nz=nz, ny=ny, nx=nx, spec_zone=1)
    base.update(overrides)
    return FlowGeometry(**base)


def flow_trajectory(geom, seed: int, dtype=np.float64) -> dict:
    """A hydrometeor and mass fluxes of either sign, so every side has
    outflow and inflow cells."""
    from .lbc import flow_shapes
    rng = np.random.default_rng(seed)
    sh = flow_shapes(geom)
    out = {"q": 1e-3 + 5e-4 * rng.standard_normal(sh["q"]),
           "ru": 4e5 * rng.standard_normal(sh["ru"]),
           "rv": 4e5 * rng.standard_normal(sh["rv"])}
    return {n: np.ascontiguousarray(v, dtype=dtype) for n, v in out.items()}


def df_geometry(nz=7, ny=8, nx=9, **overrides):
    from .diffusion import DFGeometry
    base = dict(nz=nz, ny=ny, nx=nx, rdx=1.0 / 3000.0, rdy=1.0 / 2500.0, cf1=1.6, cf2=-0.8,
                cf3=0.2)
    base.update(overrides)
    return DFGeometry(**base)


def df_trajectory(geom, seed: int, dtype=np.float64, *, hybrid: bool = False) -> dict:
    """Host state of wrf_smag_deform: winds of 10 m/s with 5 m/s noise, a
    base geopotential rising 4e3 per level with 50 noise (so the
    coordinate slopes zx, zy are nonzero), ph' of 20 and map factors near
    1; ``hybrid`` only stretches the levels (the deformation has no
    column mass)."""
    from .diffusion import DF_ACTIVE, DF_PASSIVE, DF_STATE, df_shapes
    rng = np.random.default_rng(seed)
    sh = df_shapes(geom)
    v = _vertical(geom.nz, rng, hybrid)
    znu, dnw = v["znu"], v["dnw"]
    dn = np.zeros(geom.nz)
    dn[1:] = np.diff(znu)
    fnm, fnp = np.zeros(geom.nz), np.zeros(geom.nz)
    fnm[1:] = 0.5 * dnw[:-1] / dn[1:]
    fnp[1:] = 0.5 * dnw[1:] / dn[1:]
    out = dict(dn=dn, dnw=dnw, fnm=fnm, fnp=fnp)
    for n in DF_STATE:
        out[n] = np.zeros(sh[n])
    out["u"] = 10.0 + 5.0 * rng.standard_normal(sh["u"])
    out["v"] = -4.0 + 5.0 * rng.standard_normal(sh["v"])
    out["php"] = 20.0 * rng.standard_normal(sh["php"])
    k = np.arange(geom.nz + 1)[:, None, None]
    out["phb"] = 4.0e3 * k + 50.0 * rng.standard_normal(sh["phb"])
    for n in ("msft", "msfu", "msfv"):
        out[n] = 1.0 + 0.02 * rng.standard_normal(sh[n])
    keep = DF_STATE + DF_ACTIVE + DF_PASSIVE
    return {n: np.ascontiguousarray(out[n], dtype=dtype) for n in keep}


def df_perturbation(geom, seed: int, dtype=np.float64) -> dict:
    from .diffusion import DF_ACTIVE, df_shapes
    rng = np.random.default_rng(seed)
    sh = df_shapes(geom)
    scale = {"u": 1.0, "v": 1.0, "php": 5.0}
    return {n: np.ascontiguousarray(scale[n] * rng.standard_normal(sh[n]), dtype=dtype)
            for n in DF_ACTIVE}


def km_geometry(nz=7, ny=8, nx=9, **overrides):
    from .diffusion import KMGeometry
    base = dict(nz=nz, ny=ny, nx=nx, dx=3000.0, dy=2500.0)
    base.update(overrides)
    return KMGeometry(**base)


def _km_ratio(geom, d11, d22, d12, msft):
    """smag2d_km's uncapped coefficient over its 10 * mlen cap (binary64,
    periodic or clamped neighbours as the primal)."""
    ny, nx = geom.ny, geom.nx
    jp = np.minimum(np.arange(ny) + 1, ny - 1) if geom.boundary_y else (np.arange(ny) + 1) % ny
    ip = np.minimum(np.arange(nx) + 1, nx - 1) if geom.boundary_x else (np.arange(nx) + 1) % nx
    cross = 0.25 * (d12 + d12[:, jp, :] + d12[:, :, ip] + d12[:, jp][:, :, ip])
    strain = np.sqrt(0.25 * (d11 - d22) ** 2 + cross ** 2)
    length = np.sqrt(geom.dx / msft * geom.dy / msft)
    return geom.c_s ** 2 * length ** 2 * strain / (10.0 * length)


def km_trajectory(geom, seed: int, dtype=np.float64, *, hybrid: bool = False) -> dict:
    """Host state of wrf_diff_opt1_km4: deformations whose magnitudes
    spread over 3e-4 to 0.3 /s, so about a quarter of the points take the
    10 * mlen cap; no point lies within 3% of the cap (so the switch does
    not move under the FD perturbation or a binary32 rounding).  The
    levels play no part (``hybrid`` is accepted for the common harness)."""
    from .diffusion import KM_ACTIVE, KM_PASSIVE, KM_STATE, km_shapes
    rng = np.random.default_rng(seed)
    sh = km_shapes(geom)
    out = {n: np.zeros(sh[n]) for n in KM_STATE}
    for n in KM_ACTIVE:
        mag = 10.0 ** rng.uniform(-3.5, -0.5, sh[n])
        out[n] = mag * rng.standard_normal(sh[n])
    out["msft"] = 1.0 + 0.02 * rng.standard_normal(sh["msft"])
    # the outer row of a bounded axis is 0 whatever the deformation
    inner = np.ones(sh["xkmh"], dtype=bool)
    if geom.boundary_x:
        inner[..., 0] = inner[..., -1] = False
    if geom.boundary_y:
        inner[..., 0, :] = inner[..., -1, :] = False
    r = _km_ratio(geom, out["defor11"], out["defor22"], out["defor12"], out["msft"])
    if inner.any() and not (r[inner] < 1.0).any():
        # a capped coefficient has no tangent: keep one point below the cap
        # (the coefficient is homogeneous of degree 1 in the deformation)
        for n in KM_ACTIVE:
            out[n] *= 0.5 / r[inner].min()
    for _ in range(200):
        r = _km_ratio(geom, out["defor11"], out["defor22"], out["defor12"], out["msft"])
        near = np.abs(r - 1.0) < 0.03
        if not near.any():
            break
        for n in ("defor11", "defor22"):
            out[n] = np.where(near, out[n] * 1.3, out[n])
    else:
        raise RuntimeError("km_trajectory: could not clear the cap's neighbourhood")
    if inner.any() and not (r[inner] < 1.0).any():
        raise RuntimeError("km_trajectory: every coefficient takes the cap")
    keep = KM_STATE + KM_ACTIVE + KM_PASSIVE
    return {n: np.ascontiguousarray(out[n], dtype=dtype) for n in keep}


def km_perturbation(geom, seed: int, dtype=np.float64) -> dict:
    from .diffusion import KM_ACTIVE, km_shapes
    rng = np.random.default_rng(seed)
    sh = km_shapes(geom)
    return {n: np.ascontiguousarray(1e-2 * rng.standard_normal(sh[n]), dtype=dtype)
            for n in KM_ACTIVE}


def hd_geometry(nz=7, ny=8, nx=9, **overrides):
    from .diffusion import HDGeometry
    base = dict(nz=nz, ny=ny, nx=nx, rdx=1.0 / 3000.0, rdy=1.0 / 2500.0)
    base.update(overrides)
    return HDGeometry(**base)


#: Row field scales (mean, noise) of wrf_diff_opt1_horizontal's cases.
_HD_FIELD = {"u": (10.0, 5.0), "v": (-4.0, 5.0), "w": (0.0, 0.5), "t": (5.0, 2.0),
             "s": (1.0e-2, 3.0e-3)}


def hd_trajectory(geom, seed: int, dtype=np.float64, *, hybrid: bool = False) -> dict:
    """Host state of wrf_diff_opt1_horizontal: the row's field (winds of
    10 m/s with 5 m/s noise, w of 0.5 m/s, theta - 300 of 5 K with 2 K
    noise and an initial field 1 K (rms) from it, a moisture-like scalar),
    coefficients spread over 30-3000 m2/s, a column mass of 9e4 Pa with
    500 Pa noise, map factors near 1, a nonzero starting tendency and the
    half- (w: full-) level c1, c2 (``hybrid``: c1 < 1, c2 > 0)."""
    from .diffusion import HD_ACTIVE, HD_STATE, hd_passive, hd_shapes
    rng = np.random.default_rng(seed)
    sh = hd_shapes(geom)
    v = _vertical(geom.nz, rng, hybrid)
    mean, noise = _HD_FIELD[geom.row]
    out = {"field": mean + noise * rng.standard_normal(sh["field"]),
           "xk": 10.0 ** rng.uniform(1.5, 3.5, sh["xk"]),
           "mut": 9.0e4 + 500.0 * rng.standard_normal(sh["mut"]),
           "tend": rng.standard_normal(sh["tend"])}
    if geom.perturb:
        out["base"] = out["field"] + rng.standard_normal(sh["base"])
    if geom.stag == 3:
        out["c1"], out["c2"] = v["c1f"], v["c2f"]
    else:
        out["c1"], out["c2"] = v["c1h"], v["c2h"]
    for n in ("msft", "msfu", "msfv"):
        out[n] = 1.0 + 0.02 * rng.standard_normal(sh[n])
    keep = HD_STATE + HD_ACTIVE + hd_passive(geom)
    return {n: np.ascontiguousarray(out[n], dtype=dtype) for n in keep}


def hd_perturbation(geom, seed: int, dtype=np.float64) -> dict:
    from .diffusion import HD_ACTIVE, hd_shapes
    rng = np.random.default_rng(seed)
    sh = hd_shapes(geom)
    scale = {"field": _HD_FIELD[geom.row][1] * 0.2, "xk": 10.0, "mut": 100.0}
    return {n: np.ascontiguousarray(scale[n] * rng.standard_normal(sh[n]), dtype=dtype)
            for n in HD_ACTIVE}


def h2_geometry(nz=7, ny=8, nx=9, **overrides):
    from .diffusion import H2Geometry
    base = dict(nz=nz, ny=ny, nx=nx, rdx=1.0 / 3000.0, rdy=1.0 / 2500.0, cf1=1.6, cf2=-0.8,
                cf3=0.2)
    base.update(overrides)
    return H2Geometry(**base)


def h2_trajectory(geom, seed: int, dtype=np.float64, *, hybrid: bool = False) -> dict:
    """Host state of horizontal_diffusion_2's row: deformations of about
    1e-3 /s, coefficients spread over 30-3000 m2/s, a specific volume
    near 0.9 m3/kg (qv near 1e-2 when moist), winds of 10 m/s and w of
    0.5 m/s, the row's field (theta - 300 of 5 K or a moisture-like
    scalar), a base geopotential rising 4e3 per level with 50 noise (so
    the slopes are nonzero) and ph' of 20, map factors near 1, the
    fnm/fnp/dn/dnw of stretched levels (``hybrid`` only stretches them;
    the row has no column mass) and a nonzero starting tendency."""
    from .diffusion import H2_STATE, h2_active, h2_passive, h2_shapes
    rng = np.random.default_rng(seed)
    sh = h2_shapes(geom)
    v = _vertical(geom.nz, rng, hybrid)
    znu, dnw = v["znu"], v["dnw"]
    dn = np.zeros(geom.nz)
    dn[1:] = np.diff(znu)
    fnm, fnp = np.zeros(geom.nz), np.zeros(geom.nz)
    fnm[1:] = 0.5 * dnw[:-1] / dn[1:]
    fnp[1:] = 0.5 * dnw[1:] / dn[1:]
    out = dict(dn=dn, dnw=dnw, fnm=fnm, fnp=fnp, tend=rng.standard_normal(sh["tend"]))
    for n in ("defor11", "defor22", "defor12"):
        out[n] = 1e-3 * rng.standard_normal(sh[n])
    out["xk"] = 10.0 ** rng.uniform(1.5, 3.5, sh["xk"])
    out["alt"] = 0.9 + 0.05 * rng.standard_normal(sh["alt"])
    out["qv"] = 1e-2 + 2e-3 * rng.standard_normal(sh["qv"])
    out["u"] = 10.0 + 5.0 * rng.standard_normal(sh["u"])
    out["v"] = -4.0 + 5.0 * rng.standard_normal(sh["v"])
    out["w"] = 0.5 * rng.standard_normal(sh["w"])
    mean, noise = (5.0, 2.0) if geom.row in ("t", "vt") else (1.0e-2, 3.0e-3)
    out["field"] = mean + noise * rng.standard_normal(sh["field"])
    out["php"] = 20.0 * rng.standard_normal(sh["php"])
    k = np.arange(geom.nz + 1)[:, None, None]
    out["phb"] = 4.0e3 * k + 50.0 * rng.standard_normal(sh["phb"])
    for n in ("msft", "msfu", "msfv"):
        out[n] = 1.0 + 0.02 * rng.standard_normal(sh[n])
    # the vertical rows' surface fields: USTM near 0.4 m/s, HFX near 100
    # W/m2, QFX near 1e-4 kg/m2/s and a base theta rising 3 K per level
    out["ustm"] = 0.4 + 0.1 * rng.standard_normal(sh["ustm"])
    out["hfx"] = 100.0 + 30.0 * rng.standard_normal(sh["hfx"])
    out["qfx"] = 1e-4 + 3e-5 * rng.standard_normal(sh["qfx"])
    out["thb"] = 290.0 + 3.0 * k[:-1] + 0.5 * rng.standard_normal(sh["thb"])
    keep = H2_STATE + h2_active(geom) + h2_passive(geom)
    return {n: np.ascontiguousarray(out[n], dtype=dtype) for n in keep}


def h2_perturbation(geom, seed: int, dtype=np.float64) -> dict:
    from .diffusion import h2_active, h2_shapes
    rng = np.random.default_rng(seed)
    sh = h2_shapes(geom)
    scale = {"defor11": 2e-4, "defor22": 2e-4, "defor12": 2e-4, "xk": 10.0, "alt": 0.01,
             "qv": 1e-3, "u": 1.0, "v": 1.0, "w": 0.1, "php": 5.0,
             "field": 0.4 if geom.row in ("t", "vt") else 6e-4}
    return {n: np.ascontiguousarray(scale[n] * rng.standard_normal(sh[n]), dtype=dtype)
            for n in h2_active(geom)}
