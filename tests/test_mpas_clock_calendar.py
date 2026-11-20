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


def test_a_restart_keeps_the_calendar(tmp_path):
    pytest.importorskip("cupy")
    clock = fw.Clock.from_configs({"config_start_time": "2000-02-28_23:58:00", "config_dt": "00:02:00"})
    clock.now += clock.step
    pools = {"diagnostics": {"xtime": clock.time_stamp}}
    fw.write_restart(tmp_path / "r.npz", pools=pools, clock=clock)
    _, back = fw.read_restart(tmp_path / "r.npz", like=pools)
    assert back == clock and back.time_stamp == "2000-03-01_00:00:00"
