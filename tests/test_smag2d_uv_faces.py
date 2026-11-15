"""The stored-stress u/v routes add the words wrf_smag_hd_u / wrf_smag_hd_v add.

``woof.core.dycore.launch_wrf_smag2d_hd(stagger="x" | "y")`` reads the
tau11 / tau22 (mass point) and tau12 (corner) stresses that
``launch_wrf_smag_uv_faces`` stored once (``kernels/smag2d_uv_faces.cu``),
where ``wrf_smag_hd_u`` / ``wrf_smag_hd_v`` re-evaluated every stress at each
of the up to ten points that read it.  Both compile with smag2d's IEEE
options (no contraction), so the stored word is the returned word.  These
cases hold the two routes to the same tendency words on sloped terrain with
non-unity map factors, moist density, a non-zero incoming tendency, and
periodic, specified and nested lateral boundaries, with the faces computed
by the launcher itself and handed in by the caller.  Run them with
``GPUWM_WRF_EXACT=1`` as well: that build takes smag2d's other statements.
"""
import numpy as np
import pytest

from conftest import requires_gpu
from woof.config import RunConfig

pytestmark = pytest.mark.gpu


def _state(boundary_mode, nx=23, ny=17, nz=11, seed=7):
    import cupy as cp

    from woof.core.diagnostics import update_diagnostics
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.state import init_at_rest

    extra = {} if boundary_mode == "periodic" else {boundary_mode: True}
    cfg = RunConfig(
        nx=nx, ny=ny, nz=nz, dx=900.0, dy=1100.0, ztop=9000.0,
        dt=2.0, run_seconds=0.0, terrain_opt=1, hybrid_opt=2,
        km_opt=4, bl_pbl_physics=1, c_s=0.25, moist=True, **extra)
    coord = make_vertical_coord(nz, hybrid_opt=cfg.hybrid_opt, etac=cfg.etac)
    xx = 2.0 * np.pi * np.arange(nx) / nx
    yy = 2.0 * np.pi * np.arange(ny) / ny
    terrain = (260.0 + 110.0 * np.sin(xx)[None, :] + 75.0 * np.cos(yy)[:, None]
               + 35.0 * np.sin(yy[:, None] + xx[None, :]))
    base = make_base_state(
        coord, lambda z: 300.0 + 0.004 * np.asarray(z, dtype=np.float64),
        p_surf=cfg.p_surf, ztop=cfg.ztop, terrain_z=terrain)
    state = init_at_rest(cfg, coord, base, terrain_z=terrain)
    msft = 1.08 + 0.055 * np.sin(xx)[None, :] + 0.035 * np.cos(yy)[:, None]
    msfu_core = 0.5 * (msft + np.roll(msft, 1, axis=1))
    msfv_core = 0.5 * (msft + np.roll(msft, 1, axis=0))
    state.set_map_coriolis(
        msft=msft, msfu=np.concatenate([msfu_core, msfu_core[:, :1]], axis=1),
        msfv=np.concatenate([msfv_core, msfv_core[:1, :]], axis=0))
    rng = np.random.default_rng(seed)
    state.u[...] = cp.asarray(6.0 * rng.standard_normal(state.u.shape), dtype=cp.float32)
    state.v[...] = cp.asarray(6.0 * rng.standard_normal(state.v.shape), dtype=cp.float32)
    state.w[...] = cp.asarray(1.5 * rng.standard_normal(state.w.shape), dtype=cp.float32)
    state.qv[...] = cp.asarray(0.01 + 0.004 * rng.random(state.qv.shape), dtype=cp.float32)
    update_diagnostics(state, cfg.hypsometric_opt)
    return cfg, state, rng


@requires_gpu
@pytest.mark.parametrize("boundary_mode", ["periodic", "specified", "nested"])
def test_stored_stresses_add_the_words_wrf_smag_hd_u_and_v_add(boundary_mode, monkeypatch):
    import cupy as cp

    from woof.core.dycore import (launch_wrf_smag2d_hd, launch_wrf_smag2d_km,
                                  launch_wrf_smag_uv_faces)

    cfg, state, rng = _state(boundary_mode)
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    km = state.scratch((nz, ny, nx), "smag_km")
    kh = state.scratch((nz, ny, nx), "smag_kh")
    deformation = launch_wrf_smag2d_km(state, cfg, km, kh, time_t=False)
    incoming = {"x": cp.asarray(rng.standard_normal(state.u.shape), dtype=cp.float32),
                "y": cp.asarray(rng.standard_normal(state.v.shape), dtype=cp.float32)}
    fields = {"x": state.u, "y": state.v}
    tendencies = {}
    for route in ("0", "1", "handed"):
        monkeypatch.setenv("WOOF_SMAG_UV_FACES", "0" if route == "0" else "1")
        # Poison the stress stores: the stored route must write every word
        # it reads.
        for slot, shape in (("smag_t11", (nz, ny, nx)), ("smag_t22", (nz, ny, nx)),
                            ("smag_t12", (nz, ny + 1, nx + 1))):
            state.scratch(shape, slot)[...] = cp.float32(np.nan)
        faces = None
        if route == "handed":
            faces = launch_wrf_smag_uv_faces(state, cfg, km, deformation, time_t=False)
            assert faces is not None
        for stagger in ("x", "y"):
            r = incoming[stagger].copy()
            launch_wrf_smag2d_hd(state, cfg, fields[stagger], km, r, stagger=stagger,
                                 time_t=False, deformation=deformation, faces=faces)
            tendencies[route, stagger] = cp.asnumpy(r).view(np.uint32)
    for stagger in ("x", "y"):
        assert not np.array_equal(tendencies["1", stagger],
                                  cp.asnumpy(incoming[stagger]).view(np.uint32))
        assert np.all(np.isfinite(tendencies["1", stagger].view(np.float32)))
        assert np.array_equal(tendencies["1", stagger], tendencies["0", stagger])
        assert np.array_equal(tendencies["handed", stagger], tendencies["0", stagger])


@requires_gpu
def test_disabled_route_returns_no_faces(monkeypatch):
    from woof.core.dycore import launch_wrf_smag2d_km, launch_wrf_smag_uv_faces

    cfg, state, _rng = _state("periodic")
    km = state.scratch((cfg.nz, cfg.ny, cfg.nx), "smag_km")
    kh = state.scratch((cfg.nz, cfg.ny, cfg.nx), "smag_kh")
    deformation = launch_wrf_smag2d_km(state, cfg, km, kh, time_t=False)
    monkeypatch.setenv("WOOF_SMAG_UV_FACES", "0")
    assert launch_wrf_smag_uv_faces(state, cfg, km, deformation, time_t=False) is None
