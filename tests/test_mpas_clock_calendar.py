"""The MPAS-Ocean and MPAS-Seaice clock on config_calendar_type's calendar.

mpas_timekeeping_init sets the calendar before the clock: the Registry
default gregorian_noleap (a 365-day year without 29 February) or the
proleptic gregorian. The oracle cases (split_*_after_february,
evp_*_after_february) hold the GPU runs to v8.4.2 on both; these check the
clock itself on the host.
"""
from __future__ import annotations

import numpy as np
import pytest

from woof.hex.errors import ConfigurationRefusal
from woof.hex.mpas_ocean import forward_v842 as fw


def test_the_registry_default_is_the_365_day_year():
    clock = fw.Clock.from_configs({"config_start_time": "2000-02-28_23:58:00", "config_dt": "00:02:00"})
    assert clock.calendar == "gregorian_noleap"
    clock.now += clock.step
    assert clock.time_stamp == "2000-03-01_00:00:00"
    gregorian = fw.Clock.from_configs({"config_start_time": "2000-02-28_23:58:00", "config_dt": "00:02:00",
                                       "config_calendar_type": "gregorian"})
    gregorian.now += gregorian.step
    assert gregorian.time_stamp == "2000-02-29_00:00:00"


@pytest.mark.parametrize("calendar, days", [("gregorian_noleap", 28), ("gregorian", 29)])
def test_february_has_the_calendar_s_days(calendar, days):
    elapsed = fw.parse_time("2000-03-01_00:00:00", calendar) - fw.parse_time("2000-02-01_00:00:00", calendar)
    assert elapsed == days * 86400
    # 1900 is not a leap year on either calendar
    assert fw.parse_time("1900-03-01_00:00:00", calendar) - fw.parse_time("1900-02-01_00:00:00", calendar) == \
        28 * 86400


@pytest.mark.parametrize("calendar", fw.CALENDARS)
def test_every_second_formats_back_to_its_string(calendar):
    rs = np.random.RandomState(5)
    for seconds in rs.randint(0, 3000 * 365 * 86400, 2000, dtype=np.int64):
        text = fw.format_time(int(seconds), calendar)
        assert fw.parse_time(text, calendar) == seconds, text
    assert fw.format_time(0, calendar) == "0001-01-01_00:00:00"
    days = 365 if calendar == "gregorian_noleap" else 366
    assert fw.format_time(fw.parse_time("2000-01-01_00:00:00", calendar) + days * 86400 - 1, calendar) == \
        "2000-12-31_23:59:59"


def test_dates_the_calendar_does_not_have_are_refused():
    with pytest.raises(ConfigurationRefusal, match="gregorian_noleap"):
        fw.parse_time("2000-02-29_00:00:00", "gregorian_noleap")
    with pytest.raises(ConfigurationRefusal):
        fw.parse_time("2001-02-29_00:00:00", "gregorian")
    with pytest.raises(ConfigurationRefusal):
        fw.parse_time("2001-04-31_00:00:00", "gregorian")
    assert fw.parse_time("2000-02-29_00:00:00", "gregorian") > 0


def test_other_calendars_are_refused_by_name():
    with pytest.raises(ConfigurationRefusal, match="config_calendar_type"):
        fw.Clock.from_configs({"config_start_time": "2000-01-01_00:00:00", "config_dt": "00:02:00",
                               "config_calendar_type": "360day"})


def test_the_seaice_clock_carries_the_calendar():
    from woof.hex.mpas_seaice import timestep_v842 as ts
    clock = ts.clock_from_configs({"config_start_time": "2000-02-28_23:30:00", "config_dt": 1800.0})
    assert clock.calendar == "gregorian_noleap"
    clock.now += clock.step
    assert clock.time_stamp == "2000-03-01_00:00:00"
    assert clock.now - clock.seconds("2000-02-01_00:00:00") == 28 * 86400


def _restart(tmp_path, calendar):
    """A restart of host pools after one step across 28 February 2000."""
    configs = {"config_start_time": "2000-02-28_23:58:00", "config_dt": "00:02:00",
               "config_calendar_type": calendar}
    clock = fw.Clock.from_configs(configs)
    clock.now += clock.step
    pools = {"diagnostics": {"xtime": clock.time_stamp, "daysSinceStartOfSim": np.array([0.25])},
             "mesh": {"nCells": 4}}
    fw.write_restart(tmp_path / "r.npz", pools=pools, clock=clock)
    return pools, clock


@pytest.mark.parametrize("calendar, stamp", [("gregorian_noleap", "2000-03-01_00:00:00"),
                                             ("gregorian", "2000-02-29_00:00:00")])
def test_a_restart_keeps_the_calendar(tmp_path, calendar, stamp):
    pools, clock = _restart(tmp_path, calendar)
    back_pools, back = fw.read_restart(tmp_path / "r.npz", like=pools, configs={"config_calendar_type": calendar})
    assert back == clock and back.time_stamp == stamp
    assert back_pools["diagnostics"]["xtime"] == stamp and back_pools["mesh"]["nCells"] == 4
    assert np.array_equal(back_pools["diagnostics"]["daysSinceStartOfSim"], [0.25])


def test_the_default_calendar_resumes_a_noleap_restart(tmp_path):
    pools, clock = _restart(tmp_path, "gregorian_noleap")
    _, back = fw.read_restart(tmp_path / "r.npz", like=pools, configs={})
    assert back == clock


@pytest.mark.parametrize("written, configs", [("gregorian", {}),
                                              ("gregorian", {"config_calendar_type": "gregorian_noleap"}),
                                              ("gregorian_noleap", {"config_calendar_type": "gregorian"})])
def test_a_restart_on_another_calendar_is_refused(tmp_path, written, configs):
    pools, _ = _restart(tmp_path, written)
    with pytest.raises(ConfigurationRefusal, match=f"runs on the {written} calendar") as refusal:
        fw.read_restart(tmp_path / "r.npz", like=pools, configs=configs)
    assert refusal.value.knob == "config_calendar_type"


def _legacy(tmp_path):
    """A restart as written before the clock carried its calendar: Gregorian seconds, no clock_calendar."""
    pools, clock = _restart(tmp_path, "gregorian")
    with np.load(tmp_path / "r.npz") as data:
        arrays = {k: data[k] for k in data.files if k != "clock_calendar"}
    np.savez(tmp_path / "legacy.npz", **arrays)
    return pools, clock


def test_a_legacy_restart_resumes_only_under_an_explicit_gregorian(tmp_path):
    pools, clock = _legacy(tmp_path)
    _, back = fw.read_restart(tmp_path / "legacy.npz", like=pools, configs={"config_calendar_type": "gregorian"})
    assert back == clock and back.time_stamp == "2000-02-29_00:00:00"
    for configs in ({}, {"config_calendar_type": "gregorian_noleap"}):
        with pytest.raises(ConfigurationRefusal, match="set to 'gregorian' explicitly") as refusal:
            fw.read_restart(tmp_path / "legacy.npz", like=pools, configs=configs)
        assert refusal.value.knob == "config_calendar_type"


def test_a_clock_on_another_calendar_is_refused_before_a_step():
    clock = fw.Clock.from_configs({"config_start_time": "2000-01-01_00:00:00", "config_dt": "00:02:00",
                                   "config_calendar_type": "gregorian"})
    with pytest.raises(ConfigurationRefusal, match="config_calendar_type"):
        clock.check_calendar({})
    clock.check_calendar({"config_calendar_type": "gregorian"})
    with pytest.raises(ValueError, match="calendar"):
        fw.Clock(0, 0, 60, "360day")


def test_a_bad_simulation_start_time_is_refused_by_its_own_name():
    clock = fw.Clock.from_configs({"config_start_time": "2001-01-01_00:00:00", "config_dt": "00:02:00"})
    with pytest.raises(ConfigurationRefusal) as refusal:
        clock.seconds("2000-02-29_00:00:00", "diagnostics:simulationStartTime")
    assert refusal.value.knob == "diagnostics:simulationStartTime"
