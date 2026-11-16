"""The stored-coefficient u/v routes add the words wrf_smag_hd_u / wrf_smag_hd_v add.

``woof.core.dycore.launch_wrf_smag2d_hd(stagger="x" | "y")`` reads the
stress coefficients (``-rho K`` at the mass points, ``-rhoavg kavg`` at the
corners) that ``launch_wrf_smag_uv_faces`` stored once
(``kernels/smag2d_uv_faces.cu``), where ``wrf_smag_hd_u`` / ``wrf_smag_hd_v``
re-evaluated them at each of the up to ten points that read a stress.  These
cases hold the two routes to the same tendency words on sloped terrain with
non-unity map factors, WRF's vertical weights, moist density, a non-zero
incoming tendency, and periodic, specified and nested lateral boundaries,
with the faces computed by the launcher itself and handed in by the caller.

Each case runs in a child process under the default (fast) numerics as well
as strict: the battery's conftest pins ``GPUWM_NUMERICS=strict``, and fast
mode drops ``--fmad=false``, so a stored STRESS (not coefficient) passed the
strict battery while moving about two thirds of a 3 km forecast's u words.
Run them with ``GPUWM_WRF_EXACT=1`` as well: that build takes smag2d's other
statements.
"""
import os
import subprocess
import sys
from pathlib import Path

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
    # WRF-like stretched-level weights (this coordinate's are exactly 0.5,
    # which makes every vertical interpolation product exact).
    fnm = (0.5 - 0.04 * np.sin(np.arange(nz) + 0.3)).astype(np.float32)
    fnm[0] = 0.0
    state.fnm[...] = cp.asarray(fnm)
    state.fnp[...] = cp.asarray(np.where(np.arange(nz) == 0, 0.0, 1.0 - fnm).astype(np.float32))
    rng = np.random.default_rng(seed)
    state.u[...] = cp.asarray(6.0 * rng.standard_normal(state.u.shape), dtype=cp.float32)
    state.v[...] = cp.asarray(6.0 * rng.standard_normal(state.v.shape), dtype=cp.float32)
    state.w[...] = cp.asarray(1.5 * rng.standard_normal(state.w.shape), dtype=cp.float32)
    state.qv[...] = cp.asarray(0.01 + 0.004 * rng.random(state.qv.shape), dtype=cp.float32)
    update_diagnostics(state, cfg.hypsometric_opt)
    return cfg, state, rng


def _identity(boundary_mode):
    """Child-process body: both routes, every stagger, the same words."""
    import cupy as cp

    from woof.core.dycore import (launch_wrf_smag2d_hd, launch_wrf_smag2d_km,
                                  launch_wrf_smag_uv_faces)

    cfg, state, rng = _state(boundary_mode)
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    # 0.5 weights make every interpolation product exact and hide a
    # contraction change (the #475 lesson).
    assert not np.all(cp.asnumpy(state.fnm)[1:] == np.float32(0.5))
    km = state.scratch((nz, ny, nx), "smag_km")
    kh = state.scratch((nz, ny, nx), "smag_kh")
    deformation = launch_wrf_smag2d_km(state, cfg, km, kh, time_t=False)
    incoming = {"x": cp.asarray(rng.standard_normal(state.u.shape), dtype=cp.float32),
                "y": cp.asarray(rng.standard_normal(state.v.shape), dtype=cp.float32)}
    fields = {"x": state.u, "y": state.v}
    tendencies = {}
    for route in ("0", "1", "handed"):
        os.environ["WOOF_SMAG_UV_FACES"] = "0" if route == "0" else "1"
        # Poison the coefficient stores: the stored route must write every
        # word it reads.
        for slot, shape in (("smag_cm", (nz, ny, nx)),
                            ("smag_c12", (nz, ny + 1, nx + 1))):
            state.scratch(shape, slot)[...] = cp.float32(np.nan)
        faces = None
        if route == "handed":
            faces = launch_wrf_smag_uv_faces(state, cfg, km, time_t=False)
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
    print("IDENTICAL")


@requires_gpu
@pytest.mark.parametrize("numerics", ["fast", "strict"])
@pytest.mark.parametrize("boundary_mode", ["periodic", "specified", "nested"])
def test_stored_coefficients_add_the_words_wrf_smag_hd_u_and_v_add(boundary_mode, numerics):
    if numerics == "fast" and os.environ.get("GPUWM_WRF_EXACT", "0") == "1":
        pytest.skip("GPUWM_WRF_EXACT=1 is strict numerics")
    here = Path(__file__).resolve().parent
    # conftest pins strict at import; the child selects its mode after that
    # import and before the first compile, which fixes the process's mode.
    code = (f"import os, sys; sys.path.insert(0, {str(here)!r}); "
            "import test_smag2d_uv_faces as t; "
            f"os.environ['GPUWM_NUMERICS'] = {numerics!r}; "
            "from woof import numerics; "
            f"assert numerics.mode() == {numerics!r}, numerics.mode(); "
            f"t._identity({boundary_mode!r})")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, env=dict(os.environ), timeout=900)
    assert result.returncode == 0 and "IDENTICAL" in result.stdout, (
        result.stdout[-4000:] + result.stderr[-4000:])


@requires_gpu
def test_disabled_route_returns_no_faces(monkeypatch):
    from woof.core.dycore import launch_wrf_smag2d_km, launch_wrf_smag_uv_faces

    cfg, state, _rng = _state("periodic")
    km = state.scratch((cfg.nz, cfg.ny, cfg.nx), "smag_km")
    kh = state.scratch((cfg.nz, cfg.ny, cfg.nx), "smag_kh")
    launch_wrf_smag2d_km(state, cfg, km, kh, time_t=False)
    monkeypatch.setenv("WOOF_SMAG_UV_FACES", "0")
    assert launch_wrf_smag_uv_faces(state, cfg, km, time_t=False) is None
