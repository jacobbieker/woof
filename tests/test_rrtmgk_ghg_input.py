"""RRTMG-K (14/14) with ghg_input = 1: legacy RRTMG's CAMtr gases.

WRF's rad_rrtmg_driver never reads ghg_input; woof's K takes the
time-varying CO2, N2O, CH4, CFC-11 and CFC-12 legacy RRTMG reads under
ghg_input = 1 (read_CAMgases, woof.core.camtr_gases) into both spectra, a
declared extension beyond WRF.  These tests hold:

* the gases K radiates are legacy RRTMG's REAL(4) profile values of the
  CAMtr table at the call's valid time, in the longwave and the shortwave
  engine, and they move with the date;
* that is the only change: K with ghg_input = 1 whose table returns the
  driver's fixed gases is bit for bit K with ghg_input = 0;
* ghg_input = 0 (unset) passes no gases, so rad_rrtmg_driver's fixed
  values reach the engines as before and the restart identity keeps its
  keys (the columns stay held to compiled WRF 4.7.1 by
  tests/test_rrtmg_k_column_oracle.py).
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest

from woof import config as config_mod
from woof.config import RunConfig
from woof.core import rrtmgk
from woof.core.camtr_gases import camtr_trace_gases

F = np.float32
GHG = rrtmgk.GHG_INPUT_GASES


def _cfg(**kw):
    base = dict(nx=12, ny=12, nz=12, dx=3000., dy=3000., ztop=16000., dt=3.,
                run_seconds=30., moist=True, mp_physics=10, radt=.2,
                ra_lw_physics=14, ra_sw_physics=14, o3input=2)
    base.update(kw)
    return RunConfig(**base)


def test_k_admits_ghg_input_and_stages_the_camtr_table():
    from woof.tables import _families
    config_mod.validate_run_config(_cfg(ghg_input=1))
    assert "camtr" in _families(_cfg(ghg_input=1))
    assert "camtr" not in _families(_cfg())


@pytest.mark.parametrize("when", [(2026, 232.0), (1950, 10.0), (2024, 59.5)])
def test_k_takes_legacy_rrtmgs_profile_values(when):
    """The same REAL(8) -> REAL(4) narrowing legacy RRTMG's LW and SW
    wrappers apply to read_CAMgases' values."""
    from woof.core.rrtmg_legacy_prep import lw_trace_gases
    from woof.core.rrtmg_sw import option4_trace_gases
    from woof.core.trace_gases import (FileGases, LEGACY_LW_GASES,
                                       LEGACY_SW_GASES, trace_gas_subset)
    yr, julian = when[0], F(when[1])
    got = rrtmgk.camtr_gases(yr, julian)
    assert set(got) == set(GHG)
    assert all(type(v) is F for v in got.values())
    table = FileGases(camtr_trace_gases(yr, julian))
    lw = lw_trace_gases(yr, trace_gas_subset(table, LEGACY_LW_GASES))
    for name in GHG:
        assert got[name] == lw[name], name
    co2, ch4, n2o, _ = option4_trace_gases(yr, trace_gas_subset(table, LEGACY_SW_GASES))
    assert (got["co2"], got["ch4"], got["n2o"]) == (co2, ch4, n2o)


# --------------------------------------------------------------------- GPU


def _engine_spies(monkeypatch):
    """Record the gas operands of every K longwave and shortwave call."""
    from woof.core import rrtmg_lw
    import cupy as cp
    seen = {"lw": [], "sw": []}
    lw_inner = rrtmg_lw.gpu_rrtmg_lw_batched_device

    def lw(*args, **kwargs):
        seen["lw"].append({name: cp.asnumpy(args[i]) for i, name in zip(
            range(10, 18), ("co2", "ch4", "n2o", "o2", "cfc11", "cfc12",
                            "cfc22", "ccl4"))})
        return lw_inner(*args, **kwargs)
    monkeypatch.setattr(rrtmg_lw, "gpu_rrtmg_lw_batched_device", lw)
    sw_engine = rrtmgk._engines()[1]
    sw_inner = sw_engine.rrtmg_sw_batched_device

    def sw(*args, **kwargs):
        seen["sw"].append({name: cp.asnumpy(args[i]) for i, name in zip(
            range(10, 14), ("co2", "ch4", "n2o", "o2"))})
        return sw_inner(*args, **kwargs)
    monkeypatch.setattr(sw_engine, "rrtmg_sw_batched_device", sw)
    return seen


def _radiate(cfg, start, monkeypatch, gases_hook=None):
    """One WRF-route radiation call; (result, engine gas operands, adapter)."""
    import cupy as cp
    from tilestream import harness
    from woof.core.radiation_composition import retained_ozone_adapter
    state, driver = harness.make_physics_state(cfg, 104, start_time=start)
    adapter = retained_ozone_adapter(driver.radiation_callable)
    assert isinstance(adapter, rrtmgk.RRTMGKRadiation)
    assert adapter.ghg_input == cfg.ghg_input
    with monkeypatch.context() as patch:
        if gases_hook is not None:
            patch.setattr(rrtmgk, "camtr_gases", gases_hook)
        seen = _engine_spies(patch)
        state.elapsed_seconds = 0.
        driver.compute(state, cfg)
    out = {name: cp.asnumpy(getattr(driver, name)).copy()
           for name in ("rthratenlw", "rthratensw")}
    return out, seen, adapter


def _julian(when):
    hour = when.hour + when.minute / 60.0 + when.second / 3600.0
    return F((when.timetuple().tm_yday - 1) + hour / 24.0)


@pytest.mark.gpu
@pytest.mark.parametrize("start", [datetime(2011, 4, 27, 18),
                                   datetime(1950, 7, 15, 18)])
def test_the_k_engines_radiate_the_camtr_gases(start, monkeypatch):
    out, seen, adapter = _radiate(_cfg(ghg_input=1), start, monkeypatch)
    want = {name: F(v) for name, v in
            camtr_trace_gases(start.year, _julian(start)).items()}
    fixed = rrtmgk.GASES
    assert seen["lw"] and seen["sw"]
    for spectrum, names in (("lw", fixed), ("sw", ("co2", "ch4", "n2o", "o2"))):
        for call in seen[spectrum]:
            for name in names:
                expected = want.get(name, fixed[name])
                assert (call[name] == expected).all(), (spectrum, name)
    # The table moves: 1950 has a fraction of 2011's CFCs and less CO2.
    assert want["co2"] != fixed["co2"]
    assert np.isfinite(out["rthratenlw"]).all()
    identity = adapter.restart_identity()
    assert identity["ghg_input"] == 1 and "ghg_source" in identity


@pytest.mark.gpu
def test_ghg_input_changes_only_the_gas_values(monkeypatch):
    """K at ghg_input = 1 whose table returns the driver's fixed gases is
    bit for bit K at ghg_input = 0; with the real table it differs."""
    start = datetime(2011, 4, 27, 18)
    fixed, seen0, adapter0 = _radiate(_cfg(), start, monkeypatch)
    for call in seen0["lw"]:
        for name, value in rrtmgk.GASES.items():
            assert (call[name] == value).all(), name
    assert "ghg_input" not in adapter0.restart_identity()
    same, _, _ = _radiate(
        _cfg(ghg_input=1), start, monkeypatch,
        gases_hook=lambda yr, julian: {k: rrtmgk.GASES[k] for k in GHG})
    for name in fixed:
        np.testing.assert_array_equal(same[name], fixed[name], name)
    real, _, _ = _radiate(_cfg(ghg_input=1), start, monkeypatch)
    assert not np.array_equal(real["rthratenlw"], fixed["rthratenlw"])


@pytest.mark.gpu
def test_a_k_adapter_refuses_a_config_it_was_not_built_for():
    from woof.core.radiation_composition import make_radiation
    lat = np.zeros((2, 2), F)
    adapter = make_radiation(_cfg(nx=2, ny=2), datetime(2011, 4, 27, 18), lat, lat)
    assert adapter.ghg_input == 0
    with pytest.raises(ValueError, match="ghg_input=0"):
        adapter._check_pins(replace(_cfg(nx=2, ny=2), ghg_input=1), 10)
