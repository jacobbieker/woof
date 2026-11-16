"""The stored-face TKE source deposits the words wrf_tke_rhs deposits.

``woof.core.dycore.launch_wrf_tke_km`` (km_opt = 2) stores each defor13 /
defor23 face once (``kernels/smag2d_tke_faces.cu`` wrf_tke_defor_faces) and
forms the TKE source from the stored faces (wrf_tke_rhs_faces), where
``wrf_tke_rhs`` re-evaluated every face at each of the up to four points
that read it.  A face is ``dwdx + dudz``: its last operation is an add, so
the stored word is the returned word even under fast numerics (which drop
``--fmad=false`` and let NVRTC contract a product into the add that consumes
it; a stored product would not contract), and every use squares or sums the
returned float itself.  These cases hold the two routes to the same TKE
tendency (and budget) words with random winds and TKE on a periodic and an
open domain, with the surface drag and heat-flux terms on, in a child
process under the default (fast) numerics as well as strict (the battery's
conftest pins strict; run with ``GPUWM_WRF_EXACT=1`` as well); and pin that
wrf_tke_rhs_faces is wrf_tke_rhs's text with the twelve calls substituted.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from woof.config import RunConfig

pytestmark = pytest.mark.gpu

ROOT = Path(__file__).resolve().parents[1]


def _state(open_boundaries, isfflx, nx=14, ny=9, nz=10, seed=5):
    """A km_opt=2 domain on sloped terrain with non-unity map factors, moist
    density, random winds and TKE (a zero-TKE patch for the seed floors)."""
    import cupy as cp

    from woof.core.diagnostics import update_diagnostics
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.state import init_at_rest

    extra = {"open_x": True, "open_y": True} if open_boundaries else {}
    cfg = RunConfig(
        nx=nx, ny=ny, nz=nz, dx=200.0, dy=220.0, ztop=2000.0, dt=2.0,
        run_seconds=0.0, terrain_opt=1, hybrid_opt=2, km_opt=2,
        bl_pbl_physics=0, isfflx=isfflx, c_k=0.10, tke_heat_flux=0.24,
        tke_drag_coefficient=0.0013, moist=True, **extra)
    coord = make_vertical_coord(nz, hybrid_opt=cfg.hybrid_opt, etac=cfg.etac)
    xx = 2.0 * np.pi * np.arange(nx) / nx
    yy = 2.0 * np.pi * np.arange(ny) / ny
    terrain = (60.0 + 25.0 * np.sin(xx)[None, :] + 18.0 * np.cos(yy)[:, None]
               + 9.0 * np.sin(yy[:, None] + xx[None, :]))
    base = make_base_state(
        coord, lambda z: 300.0 + 0.003 * np.asarray(z, dtype=np.float64),
        p_surf=cfg.p_surf, ztop=cfg.ztop, terrain_z=terrain)
    state = init_at_rest(cfg, coord, base, terrain_z=terrain)
    msft = 1.04 + 0.03 * np.sin(xx)[None, :] + 0.02 * np.cos(yy)[:, None]
    msfu_core = 0.5 * (msft + np.roll(msft, 1, axis=1))
    msfv_core = 0.5 * (msft + np.roll(msft, 1, axis=0))
    state.set_map_coriolis(
        msft=msft, msfu=np.concatenate([msfu_core, msfu_core[:, :1]], axis=1),
        msfv=np.concatenate([msfv_core, msfv_core[:1, :]], axis=0))
    rng = np.random.default_rng(seed)
    u = 3.0 * rng.standard_normal((nz, ny, nx + 1))
    v = 3.0 * rng.standard_normal((nz, ny + 1, nx))
    w = 0.8 * rng.standard_normal((nz + 1, ny, nx))
    w[0] = 0.0
    w[nz] = 0.0
    state.u[...] = cp.asarray(u, dtype=cp.float32)
    state.v[...] = cp.asarray(v, dtype=cp.float32)
    state.w[...] = cp.asarray(w, dtype=cp.float32)
    state.qv[...] = cp.asarray(0.008 + 0.003 * rng.random(state.qv.shape), dtype=cp.float32)
    e = 0.7 * np.abs(rng.standard_normal((nz, ny, nx)))
    e[:, 0, :2] = 0.0
    state.tke[...] = cp.asarray(e, dtype=cp.float32)
    update_diagnostics(state, cfg.hypsometric_opt)
    return cfg, state


def _kernel_text(source: str, name: str) -> str:
    start = source.index(f"void {name}(")
    end = source.index("\n}\n", start)
    return source[start:end + 2]


def test_rhs_faces_is_wrf_tke_rhs_with_the_calls_substituted():
    smag = (ROOT / "woof/core/kernels/smag2d.cu").read_text(encoding="utf-8")
    faces = (ROOT / "woof/core/kernels/smag2d_tke_faces.cu").read_text(encoding="utf-8")
    original = _kernel_text(smag, "wrf_tke_rhs")
    stored = _kernel_text(faces, "wrf_tke_rhs_faces")
    assert original.count("wrf_defor13(q, ") == 6
    assert original.count("wrf_defor23(q, ") == 6
    expected = original.replace(
        "void wrf_tke_rhs(WRF_SMAG_GRID_ARGS,",
        "void wrf_tke_rhs_faces(WRF_SMAG_GRID_ARGS,\n"
        "                 const real* d13, const real* d23,")
    expected = expected.replace("wrf_defor13(q, ", "wrf_tke_d13(q, d13, ")
    expected = expected.replace("wrf_defor23(q, ", "wrf_tke_d23(q, d23, ")
    assert stored == expected
    assert not re.search(r"wrf_defor(13|23)\(", stored)


def _rhs(cfg, state, route):
    import cupy as cp

    from woof.core.dycore import launch_wrf_tke_km
    from woof.core import tke_budget

    os.environ["WOOF_TKE_FACES"] = route
    nz, ny, nx = state.p.shape
    km = state.scratch((nz, ny, nx), "smag_km")
    kh = state.scratch((nz, ny, nx), "smag_kh")
    state.scratch((nz, ny, nx), "smag_rtke")[...] = 0
    # Poison the face stores: the stored route must write every word it reads.
    state.scratch((nz, ny, nx + 1), "smag_fx_1")[...] = cp.float32(np.nan)
    state.scratch((nz, ny + 1, nx), "smag_fy_1")[...] = cp.float32(np.nan)
    launch_wrf_tke_km(state, cfg, km, kh, time_t=False)
    out = {"rtke": cp.asnumpy(state.scratch((nz, ny, nx), "smag_rtke")).view(np.uint32)}
    if tke_budget.enabled(cfg):
        for name in ("shear", "buoyancy", "dissipation", "limiter"):
            out[name] = cp.asnumpy(tke_budget.term(state, cfg, name)).view(np.uint32)
    return out


def _identity(boundary, isfflx):
    """Child-process body: both routes, the same words."""
    cfg, state = _state(boundary == "open", isfflx)
    words = {route: _rhs(cfg, state, route) for route in ("0", "1")}
    assert np.all(np.isfinite(words["1"]["rtke"].view(np.float32)))
    assert np.any(words["1"]["rtke"] != 0)
    for name in words["0"]:
        assert np.array_equal(words["1"][name], words["0"][name]), name
    print("IDENTICAL")


@requires_gpu
@pytest.mark.parametrize("numerics", ["fast", "strict"])
@pytest.mark.parametrize("boundary", ["periodic", "open"])
@pytest.mark.parametrize("isfflx", [0, 1])
def test_stored_faces_deposit_the_words_wrf_tke_rhs_deposits(boundary, isfflx, numerics):
    if numerics == "fast" and os.environ.get("GPUWM_WRF_EXACT", "0") == "1":
        pytest.skip("GPUWM_WRF_EXACT=1 is strict numerics")
    here = Path(__file__).resolve().parent
    # conftest pins strict at import; the child selects its mode after that
    # import and before the first compile, which fixes the process's mode.
    code = (f"import os, sys; sys.path.insert(0, {str(here)!r}); "
            "import test_smag2d_tke_faces as t; "
            f"os.environ['GPUWM_NUMERICS'] = {numerics!r}; "
            "from woof import numerics; "
            f"assert numerics.mode() == {numerics!r}, numerics.mode(); "
            f"t._identity({boundary!r}, {isfflx!r})")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, env=dict(os.environ), timeout=900)
    assert result.returncode == 0 and "IDENTICAL" in result.stdout, (
        result.stdout[-4000:] + result.stderr[-4000:])
