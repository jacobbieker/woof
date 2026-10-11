"""Availability windows of the ~10 km global boundary sources the energy
planners offer (:mod:`woof.energy.source_window`)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from woof.energy import source_window as sw

NOW = datetime(2026, 10, 10, 12)


@pytest.fixture
def no_routes(monkeypatch):
    """Judge on the declared rows alone, whatever route this build has."""

    monkeypatch.setattr(sw, "_route", lambda source: None)
    monkeypatch.setattr(sw, "_rolling_endpoints", lambda source: None)


def test_other_sources_are_not_judged():
    for source in ("gfs", "ecmwf-open-data", "era5", "hrrr", "no-such"):
        assert sw.window_refusal(source, datetime(2001, 1, 1), 999.0,
                                 now=NOW) is None
        assert sw.source_notes(source) == []
        assert sw.registry_gap(source) is None


def test_aliases_name_the_planned_rows():
    assert sw.planned_source("ukmo_global") == "met-office-global"
    assert sw.planned_source("UM-Global") == "met-office-global"
    assert sw.planned_source("icon-global-ml") == "icon-global-ml"
    assert sw.planned_source("gfs") is None


@pytest.mark.parametrize("source", ["icon-global", "icon-global-ml"])
def test_icon_window(no_routes, source):
    ok = sw.window_refusal(source, datetime(2026, 10, 10, 0), 180.0, now=NOW)
    assert ok is None
    early = sw.window_refusal(source, datetime(2026, 8, 25, 6), 6.0, now=NOW)
    assert "2026-08-25T12Z" in early and "source.coop" in early
    assert "about the last 27 h" in early
    assert sw.window_refusal(source, datetime(2026, 8, 25, 12), 6.0,
                             now=NOW) is None
    # 00/12Z to 180 h, 06/18Z to 120 h, on 3 h boundaries
    assert sw.window_refusal(source, datetime(2026, 10, 9, 12), 181.0,
                             now=NOW).count("f183") == 1
    late = sw.window_refusal(source, datetime(2026, 10, 9, 18), 121.0,
                             now=NOW)
    assert "f123" in late and "f120" in late
    assert sw.window_refusal(source, datetime(2026, 10, 9, 18), 120.0,
                             now=NOW) is None
    # not landed: lag of the declared row
    assert "lands about 3.5 h" in sw.window_refusal(
        source, datetime(2026, 10, 10, 12), 6.0, now=NOW + timedelta(hours=2))
    off = sw.window_refusal(source, datetime(2026, 10, 10, 3), 6.0, now=NOW)
    assert "00/06/12/18 UTC only" in off


def test_met_office_window(no_routes):
    source = "met-office-global"
    # 3-hourly to f144 (6-hourly after it), so a 3 h series ends there
    assert sw.window_refusal(source, datetime(2026, 10, 9, 0), 144.0,
                             now=NOW) is None
    assert "f144" in sw.window_refusal(source, datetime(2026, 10, 9, 0),
                                       145.0, now=NOW)
    assert sw.window_refusal(source, datetime(2026, 10, 9, 6), 69.0,
                             now=NOW) is None
    short = sw.window_refusal(source, datetime(2026, 10, 9, 18), 70.0,
                              now=NOW)
    assert "f069" in short and "Shorten --hours" in short
    # lands about 5.5 h after the run time
    assert sw.window_refusal(source, datetime(2026, 10, 10, 6), 6.0,
                             now=datetime(2026, 10, 10, 11, 31)) is None
    pending = sw.window_refusal(source, datetime(2026, 10, 10, 6), 6.0,
                                now=datetime(2026, 10, 10, 11, 29))
    assert "lands about 5.5 h" in pending
    # a rolling two-year archive
    old = sw.window_refusal(source, datetime(2024, 9, 1, 0), 6.0, now=NOW)
    assert "two years" in old and "met-office-global" in old
    assert sw.window_refusal(source, datetime(2024, 10, 11, 0), 6.0,
                             now=NOW) is None
    # an aware clock is read as UTC
    aware = NOW.replace(tzinfo=timezone.utc)
    assert sw.window_refusal(source, datetime(2026, 10, 10, 0), 6.0,
                             now=aware) is None


def test_a_route_of_rolling_publishers_bounds_the_age(monkeypatch):
    monkeypatch.setattr(sw, "_route", lambda source: None)
    monkeypatch.setattr(sw, "_rolling_endpoints",
                        lambda source: (("dwd", 24.0),))
    refused = sw.window_refusal("icon-global", datetime(2026, 10, 8, 0), 6.0,
                                now=NOW)
    assert "only rolling publishers: dwd (about 24 h)" in refused
    assert sw.window_refusal("icon-global", datetime(2026, 10, 10, 0), 6.0,
                             now=NOW) is None


def test_the_route_ladder_and_lag_are_read_when_present():
    from woof import fetch_routes

    route = fetch_routes.table_route("icon-global")
    if route is None:                       # pragma: no cover - in-tree
        pytest.skip("this build has no icon-global route")
    last = fetch_routes.ladder_for(route, datetime(2026, 10, 9, 6))[-1]
    refused = sw.window_refusal("icon-global", datetime(2026, 10, 9, 6),
                                last + 1.0, now=NOW)
    assert f"f{last:03d}" in refused and "lead ladder" in refused
    lag = route.publication_lag_hours(0, 6, settled=True)
    pending = sw.window_refusal(
        "icon-global", datetime(2026, 10, 10, 0), 6.0,
        now=datetime(2026, 10, 10, 0) + timedelta(hours=lag * 0.5))
    assert f"lands about {lag:.1f} h" in pending


def test_a_thinning_ladder_refuses_the_lead_it_skips(monkeypatch):
    import dataclasses

    from woof import fetch_routes

    route = fetch_routes.table_route("icon-global")
    if route is None:                       # pragma: no cover - in-tree
        pytest.skip("this build has no icon-global route")
    # 3-hourly to 144 h, 6-hourly to 168 h
    thinning = dataclasses.replace(
        route, publication_eras=(), ladders=((None, ((144, 3), (168, 6))),))
    monkeypatch.setattr(sw, "_route", lambda source: thinning)
    monkeypatch.setattr(sw, "_rolling_endpoints", lambda source: None)
    start = datetime(2026, 10, 1, 0)
    refused = sw.window_refusal("met-office-global", start, 150.0, now=NOW)
    assert "publishes no f147" in refused and "144 h or less" in refused
    assert sw.window_refusal("met-office-global", start, 144.0,
                             now=NOW) is None


def test_a_defaulted_start_steps_back_to_a_landed_run(no_routes):
    # 08:00Z: the Met Office 06Z run lands about 11:30Z, its 00Z run at 05:30Z
    clock = datetime(2026, 10, 10, 8)
    assert sw.newest_landed_start("met-office-global",
                                  datetime(2026, 10, 10, 6), 6.0,
                                  now=clock) == datetime(2026, 10, 10, 0)
    assert sw.newest_landed_start("icon-global", datetime(2026, 10, 10, 6),
                                  6.0, now=datetime(2026, 10, 10, 10)) == \
        datetime(2026, 10, 10, 6)
    # other sources are never moved
    assert sw.newest_landed_start("gfs", datetime(2026, 10, 10, 6), 6.0,
                                  now=clock) == datetime(2026, 10, 10, 6)


def test_notes_say_when_the_route_reads_only_rolling_publishers(monkeypatch):
    monkeypatch.setattr(sw, "_rolling_endpoints",
                        lambda source: (("dwd", 24.0),))
    notes = " ".join(sw.source_notes("icon-global"))
    assert "reads only rolling publishers (dwd (about 24 h))" in notes
    monkeypatch.setattr(sw, "_rolling_endpoints", lambda source: None)
    assert "rolling publishers" not in " ".join(sw.source_notes("icon-global"))


def test_registry_aliases_name_the_planned_row():
    # icon-global's registry aliases are not in the declared table
    assert sw.planned_source("dwd-icon-global") == "icon-global"
    assert sw.planned_source("icon") == "icon-global"


def test_registry_gap_names_the_missing_row(monkeypatch):
    from woof import source_adapters

    monkeypatch.delitem(source_adapters._ALIASES, "met-office-global",
                        raising=False)
    gap = sw.registry_gap("met-office-global")
    assert "no registry row for 'met-office-global'" in gap
    assert "Met Office global" in gap
    assert sw.canonical_source("met-office-global") == "met-office-global"


def test_met_office_notes_name_the_gdas_land_state(monkeypatch):
    monkeypatch.setattr(sw, "_route", lambda source: None)
    notes = " ".join(sw.source_notes("met-office-global"))
    assert "no soil, land mask or sea ice" in notes
    assert "same-cycle GDAS" in notes
    assert "06/18Z runs reach 69 h" in notes


def test_the_declared_rows_agree_with_this_builds_real_routes():
    """No stand-ins: the route table and registry rows are in the tree."""

    from woof import fetch_routes
    from woof.source_adapters import get_source_adapter

    for source in ("icon-global", "icon-global-ml"):
        route = fetch_routes.route_for(source)
        dwd, coop = route.hosts
        assert dwd.retention_hours == sw.ICON_DWD_RETENTION_HOURS
        assert coop.serves_from == sw.ICON_ARCHIVE_START
        notes = " ".join(sw.source_notes(source))
        assert "about the last 27 h" in notes
        assert "rolling publishers" not in notes
        early = sw.window_refusal(source, datetime(2026, 8, 25, 6), 6.0, now=NOW)
        assert "2026-08-25T12Z" in early
        assert sw.window_refusal(source, datetime(2026, 9, 29, 12), 6.0,
                                 now=NOW) is None
    source = "met-office-global"
    window = sw.WINDOWS[source]
    assert max(window.horizons.values()) == get_source_adapter(
        source).max_forecast_hour == 144
    assert window.retention_hours == fetch_routes.route_for(
        source).hosts[0].retention_hours
    # The rolling bucket is the archive: no "until the route reads the
    # archive" note for it.
    assert "rolling publishers" not in " ".join(sw.source_notes(source))
    assert sw.window_refusal(source, datetime(2026, 10, 9, 0), 144.0,
                             now=NOW) is None
    assert "publishes no f147" in sw.window_refusal(
        source, datetime(2026, 10, 9, 0), 147.0, now=NOW)
