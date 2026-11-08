"""WRF-Solar EPS on a card: the driver seams in short forecasts.

Since the 1.1.6 sync the WRF engine's FARMS, Deng, cal_cldfra3 and MAD-WRF
are upstream's: pert_farms/pert_deng/pert_cld3 are refused by name there
(they run on the MPAS lane) and the fork's MAD-WRF driver tests are retired
with the fork's MAD-WRF port.  What runs here is pert_thom.

EPS (multi_perturb = 1):
* the patterns move on the first step and every stepsp steps after the
  first-RK physics, on domain 1 only; a nest keeps WRF's zero fields;
* the Thompson hook runs around the scheme;
* a resumed run is bit-identical to the uninterrupted one.
The column words are tests/test_wrfsolar_eps_wrf471_oracle.py and
tests/test_wrfsolar_eps_pattern_oracle.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu
from woof.config import validate_run_config
from test_shcu_deng_integration import _cfg as _deng_cfg
from test_shcu_deng_driver_gpu import _build, _snapshot


def _eps_cfg(**kw):
    base = dict(nx=16, ny=16, shcu_physics=0, multi_perturb=1, pert_thom=True,
                pert_thom_qv=1.0, pert_thom_qc=1.0, pert_thom_qi=1.0,
                pert_thom_qs=1.0, pert_thom_ni=1.0, spdt=1.0, use_mp_re=0)
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
    assert set(mp.names) == {"QVAPOR", "QCLOUD", "QICE", "QSNOW", "NI"}
    calls = {}
    for name in ("thompson_add", "thompson_remove"):
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
    for name in ("thompson_add", "thompson_remove"):
        assert calls.get(name) == 4, (name, calls)
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
    off_cfg = validate_run_config(_eps_cfg(multi_perturb=0, pert_thom=False))
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
