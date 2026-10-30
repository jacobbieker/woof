"""solar_diagnostics = 1 in the WRF step: what the routine reads, when, and
that switching it on changes no forecast word.

The routine is graded word for word by tests/test_solar_diag_wrf471_oracle.py.
Here, on a small moist Thompson state with a cloud deck put in by hand:

* WRF's Solar_step cloud (woof.core.solar_diagnostics.driver_clouds,
  evaluated every step) equals the CLDFRA each RRTMG engine radiated with on
  a radiation step, for icloud 1, 2 and 3 (and MYNN's subgrid merge);
* the end-of-step diagnostics are the routine over exactly the captured
  inputs, GHI_ACCUM sums SWDNB*dt step by step;
* a run with the diagnostics on and one with them off agree on every state
  word;
* history, restart and memory planning carry the 26 fields as WRF names
  them, and the output stack is not checkpointed (WRF has no restart I/O).
"""
import sys
from datetime import datetime

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from test_physics_driver import _full_state


def _state(variant, icloud, solar=1, mynn=False):
    start = datetime(1974, 4, 3, 18)
    lat = cp.full((2, 6), 40.0, cp.float32)
    lon = cp.linspace(-101, -99, 12, dtype=cp.float32).reshape(2, 6)
    extra = (dict(bl_pbl_physics=5, sf_sfclay_physics=5, icloud_bl=1)
             if mynn else {})
    state, cfg, driver = _full_state(
        ra_physics=4, ra_rrtmg_variant=variant, radt_minutes=1.0,
        mp_physics=8, icloud=icloud, solar_diagnostics=solar,
        radiation_start_time=start, radiation_latitude=lat,
        radiation_longitude=lon, **extra)
    state.qv[4:7] *= cp.float32(1.6)
    state.qc[3:5, :, :3] = cp.float32(2e-4)
    state.qi[10:12, :, 2:] = cp.float32(3e-5)
    state.qs[9:11] = cp.float32(2e-5)
    return state, cfg, driver


def _skip_mynn_on_metal(mynn):
    from woof import backend
    if mynn and backend.active() == "mlx":
        pytest.skip("the MYNN PBL unit does not translate to Metal on this "
                    "branch (__shfl_xor_sync); held on CUDA")


@pytest.mark.parametrize("variant", ["rrtmg_legacy", "rte-rrtmgp"])
@pytest.mark.parametrize("icloud,mynn", [(1, False), (2, False), (3, False), (1, True)])
def test_solar_step_cloud_is_the_radiated_cloud(variant, icloud, mynn):
    """On a radiation step the per-step evaluation IS the engine's CLDFRA."""
    _skip_mynn_on_metal(mynn)
    from woof.core.dycore import step
    state, cfg, driver = _state(variant, icloud, mynn=mynn)
    step(state, cfg)                      # itimestep 1: radiation due
    mine = cp.asnumpy(state.existing_scratch("solar_step_cldfra"))
    radiated = cp.asnumpy(driver.cldfra)
    np.testing.assert_array_equal(mine.view(np.uint32), radiated.view(np.uint32))
    assert (mine > 0).any()


def test_legacy_qc_tot_is_the_radiated_condensate(monkeypatch):
    """icloud = 3: QC_TOT/QI_TOT are the qc/qi the legacy wrapper reads."""
    from woof.core import rrtmg_legacy_device as dev
    from woof.core.dycore import step
    seen = []
    real = dev.lwrad_prep_batch_device

    def spy(**kw):
        seen.append({k: cp.asnumpy(kw[k]) for k in ("qc3d", "qi3d")})
        return real(**kw)

    monkeypatch.setattr(dev, "lwrad_prep_batch_device", spy)
    state, cfg, _driver = _state("rrtmg_legacy", 3)
    step(state, cfg)
    nz = state.qv.shape[0]
    for key, slot in (("qc3d", "solar_step_qc_tot"), ("qi3d", "solar_step_qi_tot")):
        got = np.concatenate([s[key] for s in seen])           # (ncol, nz)
        mine = cp.asnumpy(state.existing_scratch(slot)).reshape(nz, -1).T
        np.testing.assert_array_equal(got.view(np.uint32), mine.view(np.uint32))
    # the made-up condensate is there, and the prognostic qc is not touched
    assert (cp.asnumpy(state.existing_scratch("solar_step_qc_tot"))
            != cp.asnumpy(state.qc)).any()


def test_diagnostics_are_the_routine_over_the_captured_inputs():
    from woof.core.dycore import step
    from woof.core.solar_diagnostics import (NOUT, SOLAR_OUTPUTS, history_fields,
                                             run_solar_diag, species_flags)
    state, cfg, driver = _state("rrtmg_legacy", 3)
    swdnb = []
    ghi = np.zeros(state.qv.shape[1:], np.float32)
    for _ in range(3):
        before = cp.asnumpy(history_fields(state)["GHI_ACCUM"]).copy()
        step(state, cfg)
        swdnb.append(cp.asnumpy(driver.fields["swdnb"]).copy())
        ghi = (ghi + (swdnb[-1] * np.float32(cfg.dt)).astype(np.float32)).astype(np.float32)
        np.testing.assert_array_equal(cp.asnumpy(history_fields(state)["GHI_ACCUM"]), ghi)
        assert not np.array_equal(before, ghi)
    # replay the last call from the captured inputs
    nz, ny, nx = state.qv.shape
    m = lambda slot: state.existing_scratch(slot)
    from woof.core.solar_diagnostics import radius_metres
    out = cp.asarray(np.concatenate([np.zeros((2, ny, nx), np.float32),
                                     np.broadcast_to(before, (1, ny, nx)),
                                     np.zeros((NOUT - 3, ny, nx), np.float32)]))
    run_solar_diag(nz=nz, ncol=ny * nx, dt=cfg.dt,
                   flags=species_flags(qv=1, qc=1, qi=1, qs=1, has_reqc=1,
                                       has_reqi=1, has_reqs=1),
                   rho=m("solar_step_rho"), dz8w=m("solar_step_dz"), ph=state.php,
                   phb=cp.ascontiguousarray(cp.broadcast_to(
                       state.phb.reshape(-1, 1, 1) if state.phb.ndim == 1 else state.phb,
                       state.php.shape)),
                   cldfra=m("solar_step_cldfra"), coszen=m("solar_step_coszen"),
                   swdnb=driver.fields["swdnb"], swdnt=driver.fields["swdnt"],
                   qv=state.qv, qc=state.qc, qi=state.qi, qs=state.qs,
                   qc_tot=m("solar_step_qc_tot"), qi_tot=m("solar_step_qi_tot"),
                   re_cloud=radius_metres(state.effc), re_ice=radius_metres(state.effi),
                   re_snow=radius_metres(state.effs), out=out,
                   work=cp.empty((nz, ny * nx), cp.float32))
    want = cp.asnumpy(out)
    got = {k: cp.asnumpy(v) for k, v in history_fields(state).items()}
    for i, (_slot, name, _u, _d) in enumerate(SOLAR_OUTPUTS):
        np.testing.assert_array_equal(got[name].view(np.uint32), want[i].view(np.uint32), err_msg=name)
    # the coszen the diagnostics read is the radiation call's, made now
    assert (cp.asnumpy(m("solar_step_coszen")) > 0).all()


@pytest.mark.parametrize("variant", ["rrtmg_legacy", "rte-rrtmgp"])
def test_switching_the_diagnostics_on_changes_no_forecast_word(variant):
    from woof.core.dycore import run_steps
    on, cfg_on, _ = _state(variant, 3, solar=1)
    off, cfg_off, _ = _state(variant, 3, solar=0)
    run_steps(on, cfg_on, 3)
    run_steps(off, cfg_off, 3)
    for name in ("u", "v", "w", "theta", "php", "mup", "qv", "qc", "qi", "qs", "qr", "qg"):
        a, b = getattr(on, name, None), getattr(off, name, None)
        if a is None:
            continue
        np.testing.assert_array_equal(cp.asnumpy(a).view(np.uint32),
                                      cp.asnumpy(b).view(np.uint32), err_msg=name)


def test_history_restart_and_planning_carry_wrfs_fields():
    from woof.config import RunConfig, validate_run_config, ConfigurationRefused
    from woof.core.preflight import scratch_slot_registry
    from woof.core.solar_diagnostics import SOLAR_OUTPUTS, scratch_shapes
    from woof.io.history_layout import produced_history_shapes
    from woof.io.restart import classify_scratch_slot
    from woof.io.wrf_output_schema import REGISTRY_VAR_META
    import dataclasses
    base = RunConfig(nx=8, ny=6, nz=12, dx=3000., dy=3000., ztop=12000., dt=15.,
                     run_seconds=60., ra_lw_physics=4, ra_sw_physics=4,
                     ra_rrtmg_variant="rrtmg_legacy", mp_physics=8, moist=True)
    cfg = validate_run_config(dataclasses.replace(base, solar_diagnostics=1))
    shapes = produced_history_shapes(cfg)
    off = produced_history_shapes(base)
    for _slot, name, units, description in SOLAR_OUTPUTS:
        assert shapes[name] == (6, 8), name
        assert name not in off
        assert REGISTRY_VAR_META[name] == (description, units)
    registry = scratch_slot_registry(cfg)
    for slot, shape in scratch_shapes(cfg).items():
        assert registry[slot] == shape
        assert classify_scratch_slot(slot) == ("carry" if slot == "solar_out" else "rebuild")
    for bad in (dict(solar_diagnostics=2), dict(solar_diagnostics=1, ra_sw_physics=1)):
        with pytest.raises(ConfigurationRefused):
            validate_run_config(dataclasses.replace(base, **bad))
