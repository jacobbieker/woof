"""shcu_physics = 5 (Deng) on a card: the driver seam in a short forecast.

* Deng runs on every model step after the cumulus slot;
* the driver's call IS ``launch_deng`` on the driver's own fields (the
  glue is held word for word, the carried state included);
* its coupled tendencies reach the RK step (the forecast moves with them
  and does not without them), and the history's RAINSH is Deng's field;
* a checkpoint carries every Deng field and a resumed run is bit-identical
  to the uninterrupted one;
* default off: no Deng field is allocated and nothing launches.

The doors (config, namelist, digest) are tests/test_shcu_deng_integration.py.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from conftest import requires_gpu
from woof.config import validate_run_config
from test_shcu_deng_integration import _cfg

# ---------------------------------------------------------------------------
# on a card
# ---------------------------------------------------------------------------

def _make_state(cfg):
    """A capped, moist mixed layer under a warm surface: Deng's regime."""
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced

    coord = make_vertical_coord(cfg.nz)

    def theta(z):
        z = np.asarray(z)
        return 300.0 + np.where(z < 1200.0, 0.0, 1.5 + 0.0045 * (z - 1200.0))

    def qv(z):
        z = np.asarray(z)
        return np.where(z < 1200.0, 0.0145,
                        0.006 * np.exp(-(z - 1200.0) / 2500.0))

    base = make_base_state(coord, theta, p_surf=cfg.p_surf, ztop=cfg.ztop)
    return init_moist_balanced(cfg, coord, base, qv)


def _build(cfg):
    from woof.core.physics import initialize_physics

    state = _make_state(cfg)
    lat = np.full((cfg.ny, cfg.nx), 35.0, np.float32)
    lon = np.full((cfg.ny, cfg.nx), -97.0, np.float32)
    driver = initialize_physics(
        state, cfg, landmask=1.0, tsk=306.0, pblh=1200.0,
        radiation_start_time=datetime(2024, 7, 1, 18),
        radiation_latitude=lat, radiation_longitude=lon)
    if driver.shallow_cumulus is not None:
        _seed_cloud_history(driver)
    return state, driver


def _seed_cloud_history(driver):
    """A non-buoyant cloud (NBC) left by an earlier updraft near the capping
    inversion -- what Deng's own NBC leapfrog carries between steps -- so the
    dissipation code (edge evaporation, in-cloud and exposed-base diffusion,
    CTEI, drizzle) runs from the first step.  Varies by column."""
    import cupy as cp
    f = driver.fields
    nz, ny, nx = f["cldareab"].shape
    j, i = np.indices((ny, nx), dtype=np.float32)
    weight = cp.asarray(np.float32(0.3) + np.float32(0.05) * i
                        + np.float32(0.02) * j)
    for k, area in ((10, 0.2), (11, 0.6), (12, 0.4)):
        f["cldareab"][k] = cp.float32(area) * weight
        f["cldareaa"][k] = cp.float32(area) * weight
        f["cldliqb"][k] = cp.float32(6.0e-4) * weight
        f["cldliqa"][k] = cp.float32(5.0e-4) * weight
    f["clddpthb"][...] = cp.float32(600.0)
    f["capesave"][...] = cp.float32(40.0)
    f["radsave"][...] = cp.float32(400.0)


def _snapshot(state, driver):
    """Every carrier of the state and its driver, the streaming inventory's
    own list, plus every driver field by name."""
    import cupy as cp
    from tilestream.physics_inventory import carrier_inventory
    out = {key: cp.asnumpy(value)
           for key, value in carrier_inventory(state).items()}
    out.update({f"fields/{k}": cp.asnumpy(v)
                for k, v in driver.fields.items() if hasattr(v, "shape")})
    return out


@pytest.mark.gpu
@requires_gpu
def test_default_off_allocates_nothing_and_never_launches(monkeypatch):
    from woof.core import shcu_deng
    from tilestream import harness

    monkeypatch.setattr(shcu_deng, "launch_deng",
                        lambda **kw: pytest.fail("Deng launched while off"))
    state, driver = _build(_cfg(shcu_physics=0))
    assert driver.shallow_cumulus is None
    for name in shcu_deng.deng_field_shapes(_cfg(), 40, 8, 8):
        if name != "w0avg":
            assert name not in driver.fields, name
    harness.run_steps(state, _cfg(shcu_physics=0), 1)
    assert "shallow_cumulus" not in driver.call_counts


@pytest.mark.gpu
@requires_gpu
def test_the_driver_call_is_launch_deng_on_the_driver_fields(monkeypatch):
    import cupy as cp
    from woof.core import shcu_deng
    from tilestream import harness

    cfg = validate_run_config(_cfg())
    state, driver = _build(cfg)
    real = shcu_deng.launch_deng
    calls = []

    def recording(**kw):
        before = {k: (cp.asnumpy(v) if isinstance(v, cp.ndarray) else v)
                  for k, v in kw.items() if k != "state"}
        before["state"] = {k: cp.asnumpy(v)
                           for k, v in kw["state"].arrays().items()}
        out = real(**kw)
        after = {k: cp.asnumpy(v) for k, v in out.items()}
        after["state"] = {k: cp.asnumpy(v)
                          for k, v in kw["state"].arrays().items()}
        after["mavail"] = cp.asnumpy(kw["mavail"])
        calls.append((before, after))
        return out

    monkeypatch.setattr(shcu_deng, "launch_deng", recording)
    steps = 6
    harness.run_steps(state, cfg, steps)
    assert driver.call_counts["shallow_cumulus"] == steps
    assert len(calls) == steps
    # The scheme is active on this sounding: it is not grading zeros.
    last = calls[-1][1]["state"]
    assert last["capesave"].max() > 0.0 and last["radsave"].max() >= 150.0
    assert np.abs(last["w0avg"]).max() > 0.0
    # Replay each recorded call from its own inputs and compare every word.
    for before, after in calls:
        st = shcu_deng.DengState(**{k: cp.asarray(v) for k, v in
                                    before["state"].items()})
        mavail = cp.asarray(before["mavail"])
        kwargs = {k: (cp.asarray(v) if isinstance(v, np.ndarray) else v)
                  for k, v in before.items()
                  if k not in ("state", "mavail")}
        out = real(state=st, mavail=mavail, **kwargs)
        for name, value in out.items():
            assert np.array_equal(cp.asnumpy(value).view(np.uint8),
                                  after[name].view(np.uint8)), name
        for name, value in st.arrays().items():
            assert np.array_equal(cp.asnumpy(value).view(np.uint8),
                                  after["state"][name].view(np.uint8)), name
        assert np.array_equal(cp.asnumpy(mavail), after["mavail"])
    # Its inputs are the driver's own fields: the PBL's TKE (MYJ), EXCH_H,
    # the held radiative heating, PBLH.
    before = calls[-1][0]
    assert np.array_equal(before["tke"], cp.asnumpy(driver.fields["tke_myj"]))
    assert np.array_equal(before["rthratenlw"], cp.asnumpy(driver.rthratenlw))


@pytest.mark.gpu
@requires_gpu
def test_the_tendencies_reach_the_rk_step_and_rainsh_is_published():
    import cupy as cp
    from tilestream import harness

    cfg = validate_run_config(_cfg())
    on_state, on_driver = _build(cfg)
    harness.run_steps(on_state, cfg, 6)
    on = cp.asnumpy(on_state.total_theta())
    out = on_driver.output_fields()
    assert out["RAINSH"] is on_driver.fields["rainsh"]
    # The same run with the Deng rates withheld (the scheme still runs and
    # carries its state): the theta moves.
    off_state, off_driver = _build(cfg)
    scheme = off_driver.shallow_cumulus

    def withheld(**kw):
        return {k: cp.zeros_like(v) for k, v in scheme(**kw).items()}

    off_driver.shallow_cumulus = withheld
    harness.run_steps(off_state, cfg, 6)
    assert off_driver.call_counts["shallow_cumulus"] == 6
    assert not np.array_equal(on, cp.asnumpy(off_state.total_theta()))


@pytest.mark.gpu
@requires_gpu
def test_a_resumed_run_is_bit_identical(tmp_path):
    from woof.io import restart
    from tilestream import harness

    cfg = validate_run_config(_cfg())
    state, driver = _build(cfg)
    harness.run_steps(state, cfg, 3)
    path = restart.write_restart(tmp_path / "deng.npz", state, cfg)
    harness.run_steps(state, cfg, 3)
    reference = _snapshot(state, driver)
    for name in ("cldareab", "clddpthb", "capesave", "ainckfsa", "ltopb",
                 "w0avg", "cw_rad", "cldfra_sh", "rainsh"):
        assert f"fields/{name}" in reference, name

    resumed, resumed_driver = _build(cfg)
    restart.restore_restart(path, resumed, cfg)
    harness.run_steps(resumed, cfg, 3)
    got = _snapshot(resumed, resumed_driver)
    assert set(got) == set(reference)
    bad = [k for k in reference
           if not np.array_equal(reference[k].view(np.uint8),
                                 got[k].view(np.uint8))]
    assert not bad, bad
