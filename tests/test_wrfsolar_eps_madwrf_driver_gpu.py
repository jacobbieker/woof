"""WRF-Solar EPS and MAD-WRF on a card: the driver seams in short forecasts.

EPS (multi_perturb = 1):
* the patterns move on the first step and every stepsp steps after the
  first-RK physics, on domain 1 only; a nest keeps WRF's zero fields;
* every switched-on hook runs at its WRF seat (FARMS around farms_columns,
  icloud = 3 inputs, Deng around launch_deng, Thompson around the scheme);
* a resumed run is bit-identical to the uninterrupted one.
MAD-WRF:
* mp_physics = 96 runs no process (no rain, no latent heating);
* madwrf_opt = 2 seeds TR_Q* from q at the cold start, nudges q toward
  them inside the window and not after it;
* madwrf_cldinit's column driver runs on a real-shaped state.
The column words are tests/test_wrfsolar_eps_wrf471_oracle.py,
tests/test_wrfsolar_eps_pattern_oracle.py and tests/test_madwrf_wrf471_oracle.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu
from woof.config import validate_run_config
from test_shcu_deng_integration import _cfg as _deng_cfg
from test_shcu_deng_driver_gpu import _build, _snapshot


def _eps_cfg(**kw):
    base = dict(nx=16, ny=16, multi_perturb=1, pert_deng=True, pert_deng_qv=1.0,
                pert_deng_qc=1.0, pert_deng_t=1.0, pert_deng_w=1.0, pert_thom=True,
                pert_thom_qv=1.0, pert_thom_qc=1.0, pert_thom_qi=1.0,
                pert_thom_qs=1.0, pert_thom_ni=1.0, swint_opt=2, icloud=3,
                pert_farms=True, pert_farms_albedo=1.0, pert_farms_qv=1.0,
                pert_farms_qc=1.0, pert_farms_qs=1.0, pert_cld3=True,
                pert_cld3_qv=1.0, pert_cld3_t=1.0, spdt=1.0, use_mp_re=0)
    base.update(kw)
    return _deng_cfg(**base)


@pytest.mark.gpu
@requires_gpu
def test_the_patterns_move_on_their_cadence_and_every_hook_runs(monkeypatch):
    import cupy as cp
    from woof.core import wrfsolar_eps as eps
    from tilestream import harness

    cfg = validate_run_config(_eps_cfg())
    state, driver = _build(cfg)
    mp = driver.multi_perturb
    assert mp is not None and mp.stepsp == 3          # nint(60/20)
    assert set(mp.names) == {"ALBEDO", "AOD", "ANGSTROM", "ASSYMFAC", "QVAPOR",
                             "QCLOUD", "QICE", "QSNOW", "NI", "TH", "W"}
    calls = {}
    for name in ("farms_add", "farms_remove", "deng_add", "deng_remove",
                 "thompson_add", "thompson_remove", "cld3_inputs"):
        real = getattr(eps, name)

        def wrapped(*a, _real=real, _name=name, **k):
            calls[_name] = calls.get(_name, 0) + 1
            return _real(*a, **k)
        monkeypatch.setattr(eps, name, wrapped)
    updates = []
    real_update = mp.update
    monkeypatch.setattr(mp, "update", lambda it: updates.append(it) or real_update(it))
    pats = mp.patterns()
    assert all(float(cp.abs(p.values).max()) == 0.0 for p in pats.values())
    harness.run_steps(state, cfg, 4)
    for name in ("farms_add", "farms_remove", "deng_add", "deng_remove",
                 "thompson_add", "thompson_remove"):
        assert calls.get(name) == 4, (name, calls)
    assert calls.get("cld3_inputs", 0) >= 4
    assert updates == [1, 2, 3, 4]
    for name, p in mp.patterns().items():
        v = cp.asnumpy(p.values)
        row = eps.load_stochpert_table("")[name]
        assert np.isfinite(v).all() and np.abs(v).max() > 0.0, name
        assert np.abs(v).max() <= np.float32(row.cutoff) * np.float32(row.std), name
        assert v.shape[0] == (cfg.nz if row.vertstruc == 1 else 1)


@pytest.mark.gpu
@requires_gpu
def test_a_nest_keeps_zero_patterns():
    import cupy as cp
    from tilestream import harness

    cfg = validate_run_config(_eps_cfg(grid_id=2))
    state, driver = _build(cfg)
    harness.run_steps(state, cfg, 2)
    assert all(float(cp.abs(p.values).max()) == 0.0
               for p in driver.multi_perturb.patterns().values())


@pytest.mark.gpu
@requires_gpu
def test_the_perturbations_move_the_forecast():
    import cupy as cp
    from tilestream import harness

    on_cfg = validate_run_config(_eps_cfg())
    off_cfg = validate_run_config(_eps_cfg(multi_perturb=0, pert_farms=False,
                                           pert_deng=False, pert_thom=False,
                                           pert_cld3=False))
    on, _ = _build(on_cfg)
    off, _ = _build(off_cfg)
    harness.run_steps(on, on_cfg, 4)
    harness.run_steps(off, off_cfg, 4)
    assert not np.array_equal(cp.asnumpy(on.qv), cp.asnumpy(off.qv))


@pytest.mark.gpu
@requires_gpu
def test_a_resumed_eps_run_is_bit_identical(tmp_path):
    from woof.io import restart
    from tilestream import harness

    cfg = validate_run_config(_eps_cfg())
    state, driver = _build(cfg)
    harness.run_steps(state, cfg, 2)
    path = restart.write_restart(tmp_path / "eps.npz", state, cfg)
    harness.run_steps(state, cfg, 3)
    reference = _snapshot(state, driver)
    assert "fields/pqvapor" in reference and "fields/spforcc3d_pqvapor" in reference
    resumed, resumed_driver = _build(cfg)
    restart.restore_restart(path, resumed, cfg)
    harness.run_steps(resumed, cfg, 3)
    got = _snapshot(resumed, resumed_driver)
    bad = [k for k in reference
           if not np.array_equal(reference[k].view(np.uint8), got[k].view(np.uint8))]
    assert not bad, bad


# ---------------------------------------------------------------------------
# MAD-WRF
# ---------------------------------------------------------------------------

def _madwrf_cfg(**kw):
    base = dict(shcu_physics=0, mp_physics=96, madwrf_opt=1, use_mp_re=0)
    base.update(kw)
    return _deng_cfg(**base)


def _clouds(state):
    import cupy as cp
    nz = state.qv.shape[0]
    shape = state.qc.shape
    rng = np.random.default_rng(5)
    for name, scale in (("qc", 2e-4), ("qi", 5e-5), ("qs", 1e-4)):
        a = (rng.random(shape) < 0.3) * rng.uniform(0, scale, shape)
        getattr(state, name)[...] = cp.asarray(a.astype(np.float32))


@pytest.mark.gpu
@requires_gpu
def test_mp96_runs_no_microphysical_process():
    import cupy as cp
    from tilestream import harness

    cfg = validate_run_config(_madwrf_cfg())
    state, driver = _build(cfg)
    _clouds(state)
    harness.run_steps(state, cfg, 3)
    assert float(cp.abs(state.h_diabatic).max()) == 0.0
    assert float(cp.abs(driver.microphysics.rainnc).max()) == 0.0
    assert float(cp.abs(state.qr).max()) == 0.0


@pytest.mark.gpu
@requires_gpu
def test_madwrf_opt2_seeds_and_nudges_inside_the_window_only(monkeypatch):
    import cupy as cp
    from woof.core import madwrf
    from tilestream import harness

    cfg = validate_run_config(_madwrf_cfg(mp_physics=8, madwrf_opt=2, use_mp_re=1,
                                          madwrf_dt_nudge=1.0, madwrf_dt_relax=40.0))
    state = None
    from test_shcu_deng_driver_gpu import _make_state
    from woof.core.physics import initialize_physics
    from datetime import datetime
    state = _make_state(cfg)
    _clouds(state)
    lat = np.full((cfg.ny, cfg.nx), 35.0, np.float32)
    lon = np.full((cfg.ny, cfg.nx), -97.0, np.float32)
    initialize_physics(state, cfg, landmask=1.0, tsk=306.0, pblh=1200.0,
                       radiation_start_time=datetime(2024, 7, 1, 18),
                       radiation_latitude=lat, radiation_longitude=lon)
    tr = madwrf.tracer_fields(state)
    for t, q in zip(tr, (state.qc, state.qi, state.qs)):
        assert np.array_equal(cp.asnumpy(t), cp.asnumpy(q))
    seen = []
    real = madwrf.nudge
    monkeypatch.setattr(madwrf, "nudge", lambda *a, **k: seen.append(k["xtime_minutes"]) or real(*a, **k))
    harness.run_steps(state, cfg, 5)            # 20 s steps; window 1 min
    assert [float(x) < 1.0 for x in seen] == [True, True, True, False, False]
    for t in tr:
        assert np.isfinite(cp.asnumpy(t)).all()


@pytest.mark.gpu
@requires_gpu
def test_cloud_init_runs_on_a_model_state():
    import cupy as cp
    from woof.core import madwrf
    from test_shcu_deng_driver_gpu import _make_state

    cfg = validate_run_config(_madwrf_cfg(madwrf_cldinit=1))
    state = _make_state(cfg)
    qv0 = cp.asnumpy(state.qv).copy()
    nz, ny, nx = state.qv.shape
    from woof.core.physics import _prepare_atmosphere
    p_hyd = _prepare_atmosphere(state)["pressure"]
    receipt = madwrf.cloud_init_state(state, cfg, p_hyd=p_hyd,
                                      terrain=np.zeros((ny, nx), np.float32))
    assert receipt["madwrf_cldinit"]["insert_clouds"] == 0
    assert receipt["madwrf_cldinit"]["cloudy_columns"] == ny * nx
    assert not np.array_equal(cp.asnumpy(state.qv), qv0) or float(cp.asnumpy(state.qc).max()) > 0
