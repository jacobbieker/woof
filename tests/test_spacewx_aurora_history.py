"""spacewx_aurora_opt: the auroral oval in every history frame.

Admission, the activity sources, the checkpoint echo, the history inventory and
the refusals (hex, streamed) run on the CPU.  The card tests check that the
published rows are the device kernel's output at each frame's valid time on the
radiation columns, that they are held while the valid time does not move, and
that the atmosphere is untouched.  Every input is synthetic (an axial-dipole
IGRF table, a GFZ-format index file, an in-memory solar wind or a fixed Kp), so
nothing is fetched.
"""
from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta
import inspect

import numpy as np
import pytest

from woof.config import ConfigurationRefused, validate_run_config
from woof.spacewx import aurora as au
from woof.spacewx import aurora_history as ah
from woof.spacewx import igrf, indices, solarwind

from conftest import requires_gpu  # noqa: E402
from test_spacewx_ionosphere import _dipole  # noqa: E402
from test_spacewx_ionosphere_history import START, _cfg, _gfz, _radiation  # noqa: E402

NAMES = ("AURORA_ZONE", "AURORA_DIST", "MAG_LAT", "MAG_LT")
ENERGY = ("AURORA_EFLUX", "AURORA_EMEAN", "AURORA_IFLUX")


# --- admission ----------------------------------------------------------------------

def test_defaults_are_off_and_admitted():
    cfg = _cfg()
    assert (cfg.spacewx_aurora_opt, cfg.spacewx_kp_source) == (0, "")
    validate_run_config(cfg)
    assert ah.history_shapes(cfg) == {} and ah.device_shapes(cfg) == {}


@pytest.mark.parametrize("overrides, match", [
    (dict(spacewx_aurora_opt=3), "spacewx_aurora_opt"),
    (dict(spacewx_aurora_opt=2), "must be 'omni' or 'swpc'"),
    (dict(spacewx_aurora_opt=2, spacewx_kp_source="gfz"), "must be 'omni' or 'swpc'"),
    (dict(spacewx_aurora_opt=2, spacewx_kp_source="5"), "must be 'omni' or 'swpc'"),
    (dict(spacewx_aurora_opt=2, spacewx_kp_source="omni", ra_physics=0, ra_sw_physics=0, ra_lw_physics=0),
     "spacewx_aurora_opt = 2 places"),
    (dict(spacewx_aurora_opt=True), "spacewx_aurora_opt"),
    (dict(spacewx_aurora_opt=1, spacewx_kp_source="9.5"), "not a Kp"),
    (dict(spacewx_aurora_opt=1, spacewx_kp_source="-1"), "not a Kp"),
    (dict(spacewx_aurora_opt=1, spacewx_kp_source="nan"), "not a Kp"),
    (dict(spacewx_kp_source="gfz"), "would do nothing"),
    (dict(spacewx_aurora_opt=1, ra_physics=0, ra_sw_physics=0, ra_lw_physics=0),
     "no radiation scheme"),
])
def test_refusals_name_the_breakage(overrides, match):
    with pytest.raises(ConfigurationRefused, match=match):
        validate_run_config(_cfg(**overrides))


@pytest.mark.parametrize("source", ["", "gfz", "omni", "swpc", "4.33", "/data/indices.txt"])
def test_an_on_config_is_admitted(source):
    validate_run_config(_cfg(spacewx_aurora_opt=1, spacewx_kp_source=source))


@pytest.mark.parametrize("source", ["omni", "swpc"])
def test_the_energy_rows_are_admitted_with_a_solar_wind(source):
    validate_run_config(_cfg(spacewx_aurora_opt=2, spacewx_kp_source=source))


def test_kp_source_forms():
    assert ah.parse_kp_source("") == ("gfz", None)
    assert ah.parse_kp_source(" swpc ") == ("swpc", None)
    assert ah.parse_kp_source("omni") == ("omni", None)
    assert ah.parse_kp_source("0") == ("value", 0.0)
    assert ah.parse_kp_source("a/b.txt") == ("file", "a/b.txt")


# --- the held activity ----------------------------------------------------------------

def test_gfz_holds_the_latest_observed_kp_at_the_start(tmp_path):
    path = _gfz(tmp_path)
    text = path.read_text().splitlines()
    # 2024-05-10: Kp 1, 2, ..., 8 in the eight 3-hour intervals; the 21 UT one is missing.
    for i, line in enumerate(text):
        if line.startswith("2024 05 10"):
            words = line.split()
            words[7:15] = ["1.000", "2.000", "3.000", "4.000", "5.000", "6.000", "7.000", "-1.000"]
            text[i] = " ".join(words)
    path.write_text("\n".join(text) + "\n")
    asked = []

    def loader(source, needed, kp=False):
        assert kp                                # the start's interval, not only its day
        asked.append((source, needed))
        return indices.read(path)

    late = START.replace(hour=22)
    drive = ah.activity("gfz", late, indices_loader=loader)
    # 18-21 UT is the latest interval that has ended by 22 UT; nothing after the start is read.
    assert asked == [("gfz", START)]
    assert drive["kp"] == 7.0 and drive["record"]["spacewx_kp_interval"] == "2024-05-10T18:00Z"
    assert drive["al"] == pytest.approx(au.kp_to_al(7.0))
    assert ah.activity("gfz", START, indices_loader=loader)["kp"] == 6.0      # 15-18, not 18-21
    assert ah.activity("gfz", START.replace(hour=23, minute=59), indices_loader=loader)["kp"] == 7.0
    assert ah.activity(str(path), START.replace(hour=4))["kp"] == 1.0
    with pytest.raises(ah.AuroraHistoryError, match="no observed Kp"):
        ah.activity(str(path), datetime(2024, 6, 1, 12))


def _wind(hours, start, bz=-6.0):
    t = np.array([np.datetime64(start, "h") + np.timedelta64(k, "h") for k in range(hours)])
    full = np.full(hours, 1.0)
    return solarwind.SolarWind(hours=t, by=full * 2.0, bz=full * bz, density=full * 5.0,
                               speed=full * 450.0, kp=full * np.nan, al=full * np.nan,
                               source="synthetic", sha256="0" * 64)


def test_the_solar_wind_kp_is_the_latest_interval_with_data():
    wind = _wind(20, START - timedelta(hours=19))      # ends at the start's hour
    calls = []

    def loader(source, first, last):
        calls.append((source, first, last))
        return wind

    drive = ah.activity("swpc", START, solarwind_loader=loader)
    assert calls[0][0] == "swpc" and calls[0][2] == START
    # The 18 UT interval needs the 19 UT hour the record does not have yet, so
    # the latest complete one (15 UT: hours 14, 15, 16) is taken.
    assert drive["kp"] == pytest.approx(wind.kp_estimate(START - timedelta(hours=3)))
    assert drive["record"]["spacewx_kp_estimated"] is True
    assert drive["record"]["spacewx_kp_interval"] == "2024-05-10T15:00Z"
    with pytest.raises(ah.AuroraHistoryError, match="no swpc solar wind"):
        ah.activity("swpc", START + timedelta(days=3), solarwind_loader=loader)
    # A record running past the start (OMNI2 year files do) is not read past it.
    longer = _wind(30, START - timedelta(hours=19))
    drive = ah.activity("omni", START, solarwind_loader=lambda *a: longer)
    assert drive["record"]["spacewx_kp_interval"] == "2024-05-10T15:00Z"


def test_a_fetched_activity_is_held_for_the_start_and_reused(tmp_path):
    wind = _wind(20, START - timedelta(hours=19))
    calls = []

    def loader(source, first, last):
        calls.append(source)
        return wind

    first = ah.activity("swpc", START, solarwind_loader=loader, memo=tmp_path)
    again = ah.activity("swpc", START, solarwind_loader=lambda *a: 1 / 0, memo=tmp_path)
    assert calls == ["swpc"] and again["kp"] == first["kp"] and again["al"] == first["al"]
    assert again["record"]["spacewx_kp_interval"] == first["record"]["spacewx_kp_interval"]
    other = ah.activity("swpc", START + timedelta(hours=3), solarwind_loader=loader, memo=tmp_path)
    assert calls == ["swpc", "swpc"] and other["record"]["spacewx_kp_interval"] == "2024-05-10T15:00Z"
    for path in tmp_path.iterdir():
        path.write_text("{")                      # damaged: the source is read again
    ah.activity("swpc", START, solarwind_loader=loader, memo=tmp_path)
    assert len(calls) == 3


def test_the_energy_coupling_uses_only_hours_ended_by_the_start(tmp_path):
    from woof.spacewx import ovation
    wind = _wind(20, START - timedelta(hours=19))      # ends with the start's hour
    late = START + timedelta(minutes=40)
    drive = ah.activity("omni", late, solarwind_loader=lambda *a: wind, energy=True)
    # On the hour of the start the hour in progress has weight 0: hours 15-17 UT.
    assert drive["dphidt"] == pytest.approx(ovation.coupling(wind, START))
    assert drive["dphidt"] == pytest.approx(float(solarwind.newell_coupling(450.0, 2.0, -6.0)))
    with pytest.raises(ah.AuroraHistoryError, match="must be 'omni' or 'swpc'"):
        ah.activity("5", START, energy=True)
    # Two of hours 15-17 missing: no coupling, so the energy rows refuse; the Kp
    # alone (from an earlier interval) still runs.
    gap = _wind(20, START - timedelta(hours=19))
    gap.bz[-3:-1] = np.nan
    with pytest.raises(ah.AuroraHistoryError, match="no OVATION coupling"):
        ah.activity("omni", START, solarwind_loader=lambda *a: gap, energy=True)
    assert "dphidt" not in ah.activity("omni", START, solarwind_loader=lambda *a: gap)


def test_the_held_coupling_is_reused_and_added_to_an_older_hold(tmp_path):
    wind = _wind(20, START - timedelta(hours=19))
    calls = []

    def loader(source, first, last):
        calls.append(source)
        return wind

    plain = ah.activity("swpc", START, solarwind_loader=loader, memo=tmp_path)
    assert calls == ["swpc"] and plain["dphidt"] > 0.0           # kept even without the rows
    again = ah.activity("swpc", START, solarwind_loader=lambda *a: 1 / 0, memo=tmp_path, energy=True)
    assert again["dphidt"] == plain["dphidt"] and again["kp"] == plain["kp"]
    # A hold written before the energy rows existed has no dPhi/dt: the Kp is
    # kept and the coupling is read once and added.
    import json
    (memo,) = tmp_path.glob("swpc-*.json")
    doc = json.loads(memo.read_text())
    doc.pop("dphidt")
    doc["kp"] = 2.0
    memo.write_text(json.dumps(doc))
    added = ah.activity("swpc", START, solarwind_loader=loader, memo=tmp_path, energy=True)
    assert calls == ["swpc", "swpc"] and added["kp"] == 2.0 and added["dphidt"] == plain["dphidt"]
    assert json.loads(memo.read_text())["dphidt"] == plain["dphidt"]
    held = ah.activity("swpc", START, solarwind_loader=lambda *a: 1 / 0, memo=tmp_path, energy=True)
    assert held["kp"] == 2.0 and held["dphidt"] == plain["dphidt"]


def test_an_unreachable_gfz_falls_back_to_the_cached_file(tmp_path):
    path = _gfz(tmp_path)
    asked = []

    def loader(source, needed, kp=False):
        asked.append((needed, kp))
        if kp:
            raise OSError("network down")
        return indices.read(path)

    assert ah.activity("gfz", START, indices_loader=loader)["kp"] >= 0.0
    assert asked == [(START - timedelta(hours=3), True), (None, False)]


def test_a_fixed_kp_is_held():
    drive = ah.activity("3.67", START)
    assert drive["kp"] == 3.67 and drive["record"] == {"spacewx_kp_source": "value"}


# --- the checkpoint echo and the history inventory -------------------------------------

def test_checkpoint_echo_omits_the_off_keys_and_keeps_the_on_ones():
    from woof.io import restart
    values = dataclasses.asdict(_cfg())
    restart._drop_default_off_run_keys(values)
    assert not {"spacewx_aurora_opt", "spacewx_kp_source"} & set(values)
    values = dataclasses.asdict(_cfg(spacewx_aurora_opt=1, spacewx_kp_source="5"))
    restart._drop_default_off_run_keys(values)
    assert values["spacewx_aurora_opt"] == 1 and values["spacewx_kp_source"] == "5"
    assert "spacewx_aurora" in restart.DRIVER_REBUILT_ATTRS


def test_produced_history_shapes_carry_the_rows_only_when_on():
    from woof.io.history_layout import produced_history_shapes
    from woof.io.wrf_output_schema import REGISTRY_VAR_META
    off = produced_history_shapes(_cfg())
    assert not set(NAMES) & set(off)
    on = produced_history_shapes(_cfg(spacewx_aurora_opt=1, spacewx_kp_source="5"))
    assert {name: on[name] for name in NAMES} == {name: (4, 6) for name in NAMES}
    assert set(on) - set(off) == set(NAMES)
    assert tuple(ah.HISTORY_FIELDS) == NAMES
    for name, (_key, desc, units) in ah.HISTORY_FIELDS.items():
        assert REGISTRY_VAR_META[name] == (desc, units)
    energy = produced_history_shapes(_cfg(spacewx_aurora_opt=2, spacewx_kp_source="omni"))
    assert set(energy) - set(off) == set(NAMES + ENERGY)
    assert tuple(ah.ENERGY_HISTORY_FIELDS) == ENERGY
    from woof.spacewx.ovation import COLUMN_FIELDS
    for name, (key, desc, units) in ah.ENERGY_HISTORY_FIELDS.items():
        assert REGISTRY_VAR_META[name] == (desc, units) and key in COLUMN_FIELDS


def test_preflight_prices_the_held_arrays_and_the_kernel():
    from woof.core import preflight
    cfg = _cfg(spacewx_aurora_opt=1, spacewx_kp_source="5")
    shapes = ah.device_shapes(cfg)
    assert shapes["spacewx_aurora/history"] == (4, 4, 6) and shapes["spacewx_aurora/frame"] == (9,)
    assert sum(1 for name in shapes if shapes[name] == (24,)) == 2 + len(au.OUTPUTS)
    assert ah.device_shapes(_cfg(spacewx_aurora_opt=2, spacewx_kp_source="omni"))[
        "spacewx_aurora/history"] == (7, 4, 6)
    assert preflight.KERNEL_MAX_LOCAL_SIZE_BYTES["spacewx_aurora"] == 32
    src = inspect.getsource(preflight)
    assert 'modules.add("spacewx_aurora")' in src
    assert "items += _items(\"physics\", aurora_shapes(run))" in src


def test_streamed_domains_refuse_by_name():
    from woof.core import streaming
    assert "spacewx_aurora_opt = 1, whose history rows are computed on the" in inspect.getsource(streaming)


def test_hex_refuses_the_option_and_admits_off():
    from woof.hex.physics_suites import PhysicsSelectionRefusal, resolve_physics
    with pytest.raises(PhysicsSelectionRefusal, match="spacewx_aurora_opt"):
        resolve_physics("arwen_mpas_column", {"spacewx_aurora_opt": 1}, mesh="x1.40962")
    resolve_physics("arwen_mpas_column", {"spacewx_aurora_opt": 0}, mesh="x1.40962")


def test_the_default_driver_path_imports_nothing_new():
    from woof.core import physics
    src = inspect.getsource(physics.PhysicsDriver)
    assert 'if int(getattr(cfg, "spacewx_aurora_opt", 0) or 0) and radiation is not None:' in src


# --- on a card ------------------------------------------------------------------------

@pytest.mark.gpu
@requires_gpu
def test_rows_are_the_kernel_at_each_valid_time_and_are_held_between():
    import cupy as cp
    h = ah.AuroraHistory.for_run(_cfg(spacewx_aurora_opt=1, spacewx_kp_source="6"),
                                 _radiation(), igrf_model=_dipole())
    al = au.kp_to_al(6.0)
    assert h.al == pytest.approx(al)
    first = h.history_fields(0.0)
    assert tuple(first) == NAMES
    assert all(a.dtype == cp.float32 and a.shape == (4, 6) for a in first.values())
    want = h.aurora.evaluate(START, al)
    for name, (key, _d, _u) in ah.HISTORY_FIELDS.items():
        assert np.array_equal(cp.asnumpy(first[name]), cp.asnumpy(want[key]).reshape(4, 6)), name
    held = cp.asnumpy(first["MAG_LT"])
    assert all(h.history_fields(0.0)[k] is first[k] for k in NAMES)
    later = h.history_fields(6 * 3600.0)
    assert all(later[k] is first[k] for k in NAMES)            # one stable allocation
    dmlt = np.mod(cp.asnumpy(later["MAG_LT"]) - held, 24.0)
    np.testing.assert_allclose(dmlt, 6.0, atol=0.3)            # MLT turns with the Sun
    receipt = h.receipt()
    assert receipt["spacewx_kp"] == 6.0 and receipt["spacewx_al_nt"] == pytest.approx(al)


@pytest.mark.gpu
@requires_gpu
def test_driver_publishes_the_rows_and_leaves_the_atmosphere_alone(monkeypatch):
    import cupy as cp
    from woof.core.physics import _prepare_atmosphere
    from test_spacewx_ionosphere_history import _driver

    monkeypatch.setattr(igrf, "load", lambda path=None: _dipole())
    off, state, cfg = _driver()
    on, state_on, cfg_on = _driver(spacewx_aurora_opt=1, spacewx_kp_source="7")
    assert off.spacewx_aurora is None and on.spacewx_aurora is not None
    assert not set(NAMES) & set(off.output_fields())
    for driver, st, c in ((off, state, cfg), (on, state_on, cfg_on)):
        driver._run_radiation(_prepare_atmosphere(st), st, c)
    fields = on.output_fields()
    lat = cp.asnumpy(on.radiation_callable.latitude_deg).astype(np.float64)
    lon = cp.asnumpy(on.radiation_callable.longitude_deg).astype(np.float64)
    want = au.Aurora(lat, lon, when=START, igrf_model=_dipole()).evaluate(START, au.kp_to_al(7.0))
    for name, (key, _d, _u) in ah.HISTORY_FIELDS.items():
        assert np.array_equal(cp.asnumpy(fields[name]), cp.asnumpy(want[key]).reshape(lat.shape)), name
    for name in ("rthratensw", "rthratenlw"):
        assert np.array_equal(cp.asnumpy(getattr(on, name)), cp.asnumpy(getattr(off, name)))
    for name in ("swdown", "glw"):
        assert np.array_equal(cp.asnumpy(on.fields[name]), cp.asnumpy(off.fields[name]))


@pytest.mark.gpu
@requires_gpu
def test_energy_rows_are_ovation_on_the_kernels_frame(tmp_path):
    import cupy as cp
    from types import SimpleNamespace
    from woof.spacewx import ovation
    from test_spacewx_ovation import _premodel
    coef = ovation.read_coefficients(_premodel(tmp_path))
    wind = _wind(20, START - timedelta(hours=19))
    lat, lon = np.meshgrid(np.linspace(45.0, 88.0, 4), np.linspace(-170.0, 170.0, 6), indexing="ij")
    rad = SimpleNamespace(start_time=START, latitude_deg=lat.astype(np.float32),
                          longitude_deg=lon.astype(np.float32))
    h = ah.AuroraHistory.for_run(_cfg(spacewx_aurora_opt=2, spacewx_kp_source="omni"), rad,
                                 solarwind_loader=lambda *a: wind, igrf_model=_dipole(),
                                 ovation_coefficients=coef)
    rows = h.history_fields(3600.0)
    assert tuple(rows) == NAMES + ENERGY
    assert all(a.dtype == cp.float32 and a.shape == (4, 6) for a in rows.values())
    grids = ovation.hemisphere_grids(coef, ovation.coupling(wind, START), START)
    want = ovation.energy_columns(grids, cp.asnumpy(rows["MAG_LAT"]).reshape(-1).astype(np.float64),
                                  cp.asnumpy(rows["MAG_LT"]).reshape(-1).astype(np.float64))
    for name, (key, _d, _u) in ah.ENERGY_HISTORY_FIELDS.items():
        np.testing.assert_array_equal(cp.asnumpy(rows[name]), want[key].astype(np.float32).reshape(4, 6))
    assert cp.asnumpy(rows["AURORA_EFLUX"]).max() > 0.0
    receipt = h.receipt()
    assert receipt["spacewx_ovation_dphidt"] == pytest.approx(ovation.coupling(wind, START))
    assert receipt["spacewx_ovation_sha256"] == coef.sha256 and receipt["spacewx_hp_north_gw"] > 0.0
