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
JOULE = ("JOULE_POWER",)


# --- admission ----------------------------------------------------------------------

def test_defaults_are_off_and_admitted():
    cfg = _cfg()
    assert (cfg.spacewx_aurora_opt, cfg.spacewx_kp_source) == (0, "")
    validate_run_config(cfg)
    assert ah.history_shapes(cfg) == {} and ah.device_shapes(cfg) == {}


@pytest.mark.parametrize("overrides, match", [
    (dict(spacewx_aurora_opt=4), "spacewx_aurora_opt"),
    (dict(spacewx_aurora_opt=3), "must be 'omni', not"),
    (dict(spacewx_aurora_opt=3, spacewx_kp_source="swpc"), "only OMNI2 carries"),
    (dict(spacewx_aurora_opt=3, spacewx_kp_source="4"), "only OMNI2 carries"),
    (dict(spacewx_aurora_opt=3, spacewx_kp_source="omni", ra_physics=0, ra_sw_physics=0, ra_lw_physics=0),
     "spacewx_aurora_opt = 3 places"),
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


def test_the_joule_row_is_admitted_with_omni_only():
    validate_run_config(_cfg(spacewx_aurora_opt=3, spacewx_kp_source="omni"))


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
    # The start's own hour never counts, even toward the three good hours: with
    # 15 UT missing, hours 16, 17 and 18 do not make a coupling.
    early = _wind(20, START - timedelta(hours=19))
    early.bz[-4] = np.nan
    with pytest.raises(ah.AuroraHistoryError, match="no OVATION coupling"):
        ah.activity("omni", START + timedelta(minutes=30), solarwind_loader=lambda *a: early, energy=True)


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
    assert added["record"]["spacewx_dphidt_sha256"] == wind.sha256
    assert json.loads(memo.read_text())["record"]["spacewx_dphidt_sha256"] == wind.sha256
    held = ah.activity("swpc", START, solarwind_loader=lambda *a: 1 / 0, memo=tmp_path, energy=True)
    assert held["kp"] == 2.0 and held["dphidt"] == plain["dphidt"]


def _omni_wind(hours, start, pcn=1.5, dst=-40.0):
    wind = _wind(hours, start)
    full = np.full(hours, 1.0)
    return dataclasses.replace(wind, pcn=full * pcn, dst=full * dst, source="omni2")


def test_the_joule_drivers_are_the_latest_hour_ended_by_the_start():
    from woof.spacewx import ovation
    wind = _omni_wind(30, START - timedelta(hours=19))        # runs past the start
    wind.pcn[19:] = 9.0                                       # the start's hour and later: never read
    wind.dst[18] = np.nan                                     # 17 UT lacks Dst: 16 UT is taken
    drive = ah.activity("omni", START + timedelta(minutes=40), solarwind_loader=lambda *a: wind, joule=True)
    j = drive["joule"]
    assert (j["pcn"], j["dst"], j["hour"], j["season"]) == (1.5, -40.0, "2024-05-10T16:00Z", "summer")
    assert j["north_gw"] == pytest.approx(ovation.joule_power(1.5, -40.0, START))
    assert j["north_gw"] == pytest.approx(29.27 * 1.5 + 8.18 * 2.25 - 0.04 * 40.0 + 0.0126 * 1600.0)
    assert drive["record"]["spacewx_joule_sha256"] == wind.sha256
    # SWPC carries no PC(N) or Dst; a fixed Kp none either.
    for source in ("swpc", "gfz", "5"):
        with pytest.raises(ah.AuroraHistoryError, match="must be 'omni'"):
            ah.activity(source, START, solarwind_loader=lambda *a: wind, joule=True)
    # No hour with both in the lookback: refused by name; the Kp alone still runs.
    empty = _wind(30, START - timedelta(hours=19))
    with pytest.raises(ah.AuroraHistoryError, match="no OMNI2 hour with both PC"):
        ah.activity("omni", START, solarwind_loader=lambda *a: empty, joule=True)
    assert "joule" not in ah.activity("omni", START, solarwind_loader=lambda *a: empty)


def test_the_held_joule_drivers_are_reused_and_added_to_an_older_hold(tmp_path):
    import json
    wind = _omni_wind(20, START - timedelta(hours=19))
    calls = []

    def loader(source, first, last):
        calls.append(source)
        return wind

    plain = ah.activity("omni", START, solarwind_loader=loader, memo=tmp_path, energy=True)
    assert calls == ["omni"] and plain["joule"]["pcn"] == 1.5           # kept even without the row
    held = ah.activity("omni", START, solarwind_loader=lambda *a: 1 / 0, memo=tmp_path, joule=True)
    assert held["joule"] == plain["joule"] and held["dphidt"] == plain["dphidt"]
    (memo,) = tmp_path.glob("omni-*.json")
    doc = json.loads(memo.read_text())
    doc.pop("joule")
    doc["kp"] = 2.0
    memo.write_text(json.dumps(doc))
    added = ah.activity("omni", START, solarwind_loader=loader, memo=tmp_path, energy=True, joule=True)
    assert calls == ["omni", "omni"] and added["kp"] == 2.0 and added["joule"] == plain["joule"]
    assert added["dphidt"] == plain["dphidt"] and added["record"]["spacewx_joule_sha256"] == wind.sha256
    again = json.loads(memo.read_text())
    assert again["joule"] == plain["joule"] and again["kp"] == 2.0 and again["dphidt"] == plain["dphidt"]


def test_a_negative_fit_is_held_at_zero_power():
    # Summer, PC(N) 0, Dst -3 nT: the fit gives -0.04 * 3 + 0.0126 * 9 < 0.
    drive = {"kp": 1.0, "al": -50.0, "record": {},
             "joule": {"pcn": 0.0, "dst": -3.0, "hour": "x", "season": "summer", "north_gw": -0.0066}}
    h = ah.AuroraHistory(None, START, drive, (2, 2))
    assert h.joule_gw == 0.0 and h.names[-1] == "JOULE_POWER"
    assert ah.AuroraHistory(None, START, dict(drive, joule=None), (2, 2)).joule_gw is None


def test_knipp_joule_power_matches_the_published_scale():
    """Physical sanity (Knipp et al. 2004, Sol. Phys. 224, 495): their fit
    gives a 1975-2003 mean global Joule input of 95 GW (twice the northern
    value).  With climatological drivers, |PC(N)| about 1.2 and Dst about
    -17 nT, the annual fit gives a global power of that size; a severe storm
    (PC(N) 6, Dst -250 nT) gives hundreds of GW per hemisphere, the scale of
    AMIE storm peaks (McHarg et al. 2005, JGR 110, A12S16: up to about 300 GW)."""
    from woof.spacewx import ovation
    june = datetime(2024, 6, 1)
    mean_global = 2.0 * ovation.joule_power(1.2, -17.0, june, season="annual")
    assert 0.7 * 95.0 <= mean_global <= 1.3 * 95.0
    storm = ovation.joule_power(6.0, -250.0, june, season="annual")
    assert 200.0 <= storm <= 600.0
    for season in ("winter", "summer", "equinox"):
        quiet = ovation.joule_power(1.2, -17.0, june, season=season)
        assert 20.0 <= quiet <= 70.0
        assert ovation.joule_power(6.0, -250.0, june, season=season) > 4.0 * quiet


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
    joule = produced_history_shapes(_cfg(spacewx_aurora_opt=3, spacewx_kp_source="omni"))
    assert set(joule) - set(off) == set(NAMES + ENERGY + JOULE)
    assert tuple(ah.JOULE_HISTORY_FIELDS) == JOULE
    for name, (desc, units) in ah.JOULE_HISTORY_FIELDS.items():
        assert REGISTRY_VAR_META[name] == (desc, units)


def test_preflight_prices_the_held_arrays_and_the_kernel():
    from woof.core import preflight
    cfg = _cfg(spacewx_aurora_opt=1, spacewx_kp_source="5")
    shapes = ah.device_shapes(cfg)
    assert shapes["spacewx_aurora/history"] == (4, 4, 6) and shapes["spacewx_aurora/frame"] == (9,)
    assert sum(1 for name in shapes if shapes[name] == (24,)) == 2 + len(au.OUTPUTS)
    assert ah.device_shapes(_cfg(spacewx_aurora_opt=2, spacewx_kp_source="omni"))[
        "spacewx_aurora/history"] == (7, 4, 6)
    assert ah.device_shapes(_cfg(spacewx_aurora_opt=3, spacewx_kp_source="omni"))[
        "spacewx_aurora/history"] == (8, 4, 6)
    assert preflight.KERNEL_MAX_LOCAL_SIZE_BYTES["spacewx_aurora"] == 32
    src = inspect.getsource(preflight)
    assert 'modules.add("spacewx_aurora")' in src
    assert "items += _items(\"physics\", aurora_shapes(run))" in src


def test_streamed_domains_refuse_by_name():
    from woof.core import streaming
    assert "spacewx_aurora_opt = {int(cfg.spacewx_aurora_opt)}, whose history rows are computed on the" in (
        inspect.getsource(streaming))


def test_hex_generic_seam_takes_the_keys():
    # The rows themselves: tests/test_hex_spacewx_history_gpu.py.
    from woof.hex.physics_suites import resolve_physics
    got = resolve_physics("arwen_mpas_column", {"spacewx_aurora_opt": 1, "spacewx_kp_source": "5"},
                          mesh="x1.40962").engine_physics
    assert (got["spacewx_aurora_opt"], got["spacewx_kp_source"]) == (1, "5")
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
    assert receipt["spacewx_hp_ion_south_gw"] == pytest.approx(ovation.power_scalars(grids)["hp_ion_south_gw"])
    # A second domain of the same run shares the held tables.
    again = ah.AuroraHistory.for_run(_cfg(spacewx_aurora_opt=2, spacewx_kp_source="omni"), rad,
                                     solarwind_loader=lambda *a: wind, igrf_model=_dipole(),
                                     ovation_coefficients=coef)
    assert again.tables is h.tables


def test_missing_ovation_coefficients_are_refused_by_name(monkeypatch):
    from types import SimpleNamespace
    from woof.spacewx import ovation

    def offline(source=None):
        raise OSError("network down")

    monkeypatch.setattr(ovation, "load", offline)
    wind = _wind(20, START - timedelta(hours=19))
    rad = SimpleNamespace(start_time=START, latitude_deg=np.zeros((2, 2), np.float32),
                          longitude_deg=np.zeros((2, 2), np.float32))
    with pytest.raises(ah.AuroraHistoryError, match="spacewx_aurora_opt = 2 needs the OVATION Prime"):
        ah.AuroraHistory.for_run(_cfg(spacewx_aurora_opt=2, spacewx_kp_source="omni"), rad,
                                 solarwind_loader=lambda *a: wind, igrf_model=_dipole())


@pytest.mark.gpu
@requires_gpu
def test_the_joule_row_is_the_held_knipp_power_in_every_column(tmp_path):
    import cupy as cp
    from types import SimpleNamespace
    from woof.spacewx import ovation
    from test_spacewx_ovation import _premodel
    coef = ovation.read_coefficients(_premodel(tmp_path))
    wind = _omni_wind(20, START - timedelta(hours=19), pcn=3.0, dst=-120.0)
    lat, lon = np.meshgrid(np.linspace(-80.0, 88.0, 4), np.linspace(-170.0, 170.0, 6), indexing="ij")
    rad = SimpleNamespace(start_time=START, latitude_deg=lat.astype(np.float32),
                          longitude_deg=lon.astype(np.float32))
    cfg = _cfg(spacewx_aurora_opt=3, spacewx_kp_source="omni")
    h = ah.AuroraHistory.for_run(cfg, rad, solarwind_loader=lambda *a: wind, igrf_model=_dipole(),
                                 ovation_coefficients=coef)
    want = np.float32(ovation.joule_power(3.0, -120.0, START))
    for elapsed in (0.0, 3600.0, 7200.0):
        rows = h.history_fields(elapsed)
        assert tuple(rows) == NAMES + ENERGY + JOULE
        joule = cp.asnumpy(rows["JOULE_POWER"])
        assert joule.dtype == np.float32 and joule.shape == (4, 6) and np.all(joule == want)
    receipt = h.receipt()
    assert receipt["spacewx_joule_north_gw"] == pytest.approx(float(want))
    assert receipt["spacewx_joule_global_gw"] == pytest.approx(2.0 * float(want))
    assert (receipt["spacewx_joule_pcn"], receipt["spacewx_joule_dst_nt"]) == (3.0, -120.0)
    assert receipt["spacewx_joule_hour"] == "2024-05-10T17:00Z"
    # Option 2 on the same drive writes no Joule row and no Joule receipt.
    two = ah.AuroraHistory.for_run(_cfg(spacewx_aurora_opt=2, spacewx_kp_source="omni"), rad,
                                   solarwind_loader=lambda *a: wind, igrf_model=_dipole(),
                                   ovation_coefficients=coef)
    assert tuple(two.history_fields(0.0)) == NAMES + ENERGY
    assert not any(k.startswith("spacewx_joule") for k in two.receipt())
