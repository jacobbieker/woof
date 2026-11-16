"""FARMS in one launch (kernels/farms.cu farms_cell_inputs; farms.py
FastSurfaceShortwave._every_step_fused) against the per-step glue it
replaces: the standalone cloud-fraction launch, the MYNN merge ufuncs, the
host radius multiplies, the cloud-fraction publication and the six output
copies.  Every word of the six fluxes and of the published fraction must
match, with and without the MYNN merge, on the first step (mass merge only)
and later (fraction replaced), day and night columns, present and absent
radii.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

import _numerics_child
from conftest import requires_gpu

NZ, NY, NX = 12, 9, 11


def _cfg(*, mynn: bool, aer_opt: int = 0):
    return SimpleNamespace(
        icloud=1, cu_rad_feedback=False, mp_physics=8, shcu_physics=0,
        bl_pbl_physics=5 if mynn else 1, icloud_bl=1, bl_mynn_version="wrf_461",
        moist=True, use_mp_re=1, aer_opt=aer_opt, aer_ssa_val=0.9,
        aer_asy_val=0.7, aer_angexp_val=1.3, couple_farms=False, dt=15.0)


def _fields(rng):
    import cupy as cp

    shape = (NZ, NY, NX)
    u = lambda lo, hi, s=shape: cp.asarray(rng.uniform(lo, hi, s).astype(np.float32))
    atmosphere = {
        "qv": u(0, .02), "qc": u(0, 2e-4), "qi": u(0, 2e-4), "qs": u(0, 2e-4),
        "temperature": u(220, 300), "pressure": u(30000, 100000),
        "rho": u(.4, 1.2), "dz": u(100, 400),
        "p_interface": u(30000, 101000, (NZ + 1, NY, NX)),
    }
    # clear and saturated cells, so the composition's branches are reached
    for key in ("qc", "qi", "qs"):
        atmosphere[key][:, ::3, :] = 0
    atmosphere["qc"][:, 1::3, 1::2] = 0
    fields = {
        "albedo": u(.1, .4, (NY, NX)),
        "qc_bl": u(0, 5e-4), "qi_bl": u(0, 5e-5), "cldfra_bl": u(0, 1),
    }
    fields["cldfra_bl"][:, ::4, :] = 0
    # Poisoned outputs: each path must write every word it publishes.
    for name in ("swdown2", "swddir2", "swddni2", "swddif2", "swdownc2", "swddnic2"):
        fields[name] = cp.full((NY, NX), np.nan, cp.float32)
    state = SimpleNamespace(
        effc=u(2, 30), effi=u(4, 60), effs=u(10, 200), elapsed_seconds=0.0)
    return atmosphere, fields, state


def _driver(fields, *, cldfra):
    import cupy as cp

    return SimpleNamespace(
        fields=dict(fields), aodtot=None, aerosol2d=None,
        cldfra=cp.full((NZ, NY, NX), np.nan, cp.float32) if cldfra else None)


def _farms():
    from woof.core.farms import FastSurfaceShortwave

    lat = np.linspace(40.0, 48.0, NY)[:, None] * np.ones((NY, NX))
    lon = np.linspace(-125.0, -115.0, NX)[None, :] * np.ones((NY, NX))
    return FastSurfaceShortwave(
        start_time=datetime(2026, 6, 21, 12), latitude_deg=lat,
        longitude_deg=lon, shape=(NY, NX))


def _words(driver):
    import cupy as cp

    out = {name: cp.asnumpy(driver.fields[name]).view(np.uint32)
           for name in ("swdown2", "swddir2", "swddni2", "swddif2", "swdownc2", "swddnic2")}
    if driver.cldfra is not None:
        out["cldfra"] = cp.asnumpy(driver.cldfra).view(np.uint32)
    return out


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("mynn,elapsed,cldfra", [
    (False, 0.0, False), (True, 0.0, True), (True, 450.0, True),
    (True, 450.0, False), (False, 7200.0, True),
])
def test_fused_step_matches_the_per_step_glue(monkeypatch, mynn, elapsed, cldfra):
    import cupy as cp

    import woof.core.farms as farms

    rng = np.random.default_rng(21)
    atmosphere, fields, state = _fields(rng)
    state.elapsed_seconds = elapsed
    cfg = _cfg(mynn=mynn)
    results = {}
    for enabled in (False, True):
        monkeypatch.setattr(farms, "FARMS_FUSED_ENABLED", enabled)
        driver = _driver(fields, cldfra=cldfra)
        scheme = _farms()
        before = {k: v.copy() for k, v in atmosphere.items()}
        scheme.every_step(driver, dict(atmosphere), state, cfg)
        # neither path writes the atmosphere it was handed
        for k in before:
            assert cp.array_equal(before[k], atmosphere[k]), k
        results[enabled] = _words(driver)
    assert results[False].keys() == results[True].keys()
    for key in results[False]:
        assert np.array_equal(results[False][key], results[True][key]), key
        assert np.all(np.isfinite(results[True][key].view(np.float32))), key
    # not vacuous: daylight columns carry flux
    assert results[True]["swdown2"].view(np.float32).max() > 0.0
    if cldfra:
        assert results[True]["cldfra"].view(np.float32).max() > 0.0


@requires_gpu
@pytest.mark.gpu
def test_fused_path_is_taken_and_refused_as_documented(monkeypatch):
    import woof.core.farms as farms

    rng = np.random.default_rng(3)
    atmosphere, fields, state = _fields(rng)
    scheme = _farms()
    driver = _driver(fields, cldfra=True)
    assert scheme._fused_inputs(driver, atmosphere, state, _cfg(mynn=True)) is not None
    # an icloud the composition does not cover, the GSD merge, an absent
    # MYNN field and a host array each take the per-step glue
    cfg = _cfg(mynn=True)
    cfg.icloud = 2
    assert scheme._fused_inputs(driver, atmosphere, state, cfg) is None
    cfg = _cfg(mynn=True)
    cfg.bl_mynn_version = "gsd_41"
    assert scheme._fused_inputs(driver, atmosphere, state, cfg) is None
    partial = _driver({k: v for k, v in fields.items() if k != "qi_bl"}, cldfra=True)
    assert scheme._fused_inputs(partial, atmosphere, state, _cfg(mynn=True)) is None
    assert scheme._fused_inputs(partial, atmosphere, state, _cfg(mynn=False)) is not None
    host = dict(atmosphere, qv=np.asarray(atmosphere["qv"].get()))
    assert scheme._fused_inputs(driver, host, state, _cfg(mynn=False)) is None
    # the launch is reached when enabled
    calls = []
    real = farms.farms_surface_columns

    def spy(*args, **kwargs):
        calls.append(kwargs.get("cloud") is not None)
        return real(*args, **kwargs)

    monkeypatch.setattr(farms, "farms_surface_columns", spy)
    monkeypatch.setattr(farms, "FARMS_FUSED_ENABLED", True)
    scheme.every_step(driver, dict(atmosphere), state, _cfg(mynn=True))
    assert calls == [True]
    monkeypatch.setattr(farms, "FARMS_FUSED_ENABLED", False)
    scheme.every_step(driver, dict(atmosphere), state, _cfg(mynn=True))
    assert calls == [True, False]


@requires_gpu
@pytest.mark.gpu
def test_kernel_radius_scale_matches_the_host_multiply():
    import cupy as cp

    from woof.core.farms import farms_surface_columns

    rng = np.random.default_rng(8)
    nz, nc = 14, 37
    u = lambda lo, hi, s=(nz, nc): cp.asarray(rng.uniform(lo, hi, s).astype(np.float32))
    p8w = u(30000, 101000, (nz + 1, nc)); rho = u(.4, 1.2); dz = u(100, 400)
    qv, qi, qs, qc, cf = u(0, .02), u(0, 2e-4), u(0, 2e-4), u(0, 2e-4), u(0, 1)
    microns = (u(2, 30), u(4, 60), u(10, 200))
    s = lambda lo, hi: cp.asarray(rng.uniform(lo, hi, nc).astype(np.float32))
    alb, ssa, asy, aod, ang, cz = s(.1, .4), s(.8, 1), s(.6, .8), s(0, .3), s(1, 2), s(-.2, 1)
    args = (p8w, rho, dz, alb, ssa, asy, aod, ang, cz, qv, qi, qs, qc)
    scaled = [m * np.float32(1.e-6) for m in microns]
    ref = farms_surface_columns(*args, *scaled, cf, julian=172,
                                has_reqc=1, has_reqi=1, has_reqs=1)
    got = farms_surface_columns(*args, *microns, cf, julian=172,
                                has_reqc=1, has_reqi=1, has_reqs=1, re_scale=1.e-6)
    assert np.array_equal(cp.asnumpy(ref).view(np.uint32), cp.asnumpy(got).view(np.uint32))
    assert float(ref.max()) > 0.0


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize(*_numerics_child.MODES)
def test_identity_holds_under_each_numerics(numerics, exact):
    """FARMS in one launch against the per-step glue.
    The battery runs strict; production runs fast, where NVRTC drops
    --fmad=false and Metal keeps fp contract(fast).  Each mode in its own
    child process (tests/_numerics_child.py)."""
    _numerics_child.run(__file__, ['test_fused_step_matches_the_per_step_glue', 'test_kernel_radius_scale_matches_the_host_multiply'], numerics, exact)
