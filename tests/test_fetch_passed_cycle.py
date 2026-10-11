"""A cycle the server has let go is refused as gone, never as "not published yet".

The breakage these guard: ``woof fetch --source icon-global --cycle
2026-09-27T00`` a day and a half later, and ``--source icon-eu --cycle
2026-09-27T06`` a day later, were refused with "cycle ... is not
published through f003 yet; the newest complete ... cycle covering f003
is 2026-09-28T06Z".  Both cycles are OLDER than that newest one: DWD's
open-data door keeps each cycle hour's newest run for about a day and
has no archive behind it, so the user was told to wait for a cycle that
would never appear.

Only the network is replaced, by a stand-in listing of a rolling door
that holds each cycle hour's newest published run.  The probe URLs, the
endpoint ladder, the retention facts and the latest-cycle resolver are
the production ones.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import re

import pytest

import woof.fetch as fetch
from woof import fetch_endpoints, fetch_routes


NOW = datetime(2026, 9, 28, 12)
_STAMP = re.compile(r"_(\d{10})_")


def _rolling_door(source: str, *, lag_hours: int = 3, dropped=()):
    """What DWD's door answers: each cycle hour's newest run published by ``NOW``, less ``dropped``."""

    held = set()
    for hour in fetch_routes.route_for(source).cycle_hours:
        run = NOW.replace(hour=int(hour))
        while run + timedelta(hours=lag_hours) > NOW:
            run -= timedelta(days=1)
        if run not in dropped:
            held.add(f"{run:%Y%m%d%H}")

    def probe(url):
        stamp = _STAMP.search(url)
        return bool(stamp) and stamp.group(1) in held
    return probe


#: The newest run the stand-in door holds that `latest` reaches at ``NOW``.
#: ICON-EU's measured publication lag (f000 at + 2 h 40 min, 2026-09-29) now
#: starts the walk at its 09Z run, which the door holds at + 3 h; the route
#: used to declare 5 h and so never asked about it.  ICON global declares 5 h.
_NEWEST = {"icon-eu": "2026-09-28T09Z"}


@pytest.mark.parametrize("source,cycle,age", [
    ("icon-eu", datetime(2026, 9, 27, 6), 30),
])
def test_a_cycle_past_the_doors_retention_is_refused_as_gone(source, cycle, age):
    refusal = fetch.cycle_publication_refusal(source, cycle, 3, now=NOW,
                                              probe=_rolling_door(source))

    assert refusal is not None
    named = f"{source.upper()} cycle {cycle:%Y-%m-%dT%H}Z"
    assert refusal.startswith(f"{named} is no longer on the server: opendata.dwd.de keeps "
                              f"only about the newest 24 h of {source.upper()} cycles and "
                              f"this one is {age} h old, with no archive behind it, so it "
                              "will not appear by waiting; ")
    assert (f"the newest complete {source.upper()} cycle covering f003 is {_NEWEST[source]} "
            "-- pass that, or --cycle latest") in refusal
    assert "not published" not in refusal
    with pytest.raises(RuntimeError, match="is no longer on the server"):
        fetch.require_published_cycle(source, cycle, 3, now=NOW, probe=_rolling_door(source),
                                      progress=lambda line: None)


@pytest.mark.parametrize("source", ["icon-eu"])
def test_a_cycle_not_yet_published_is_still_told_to_wait(source):
    cycle = datetime(2026, 9, 28, 12)

    refusal = fetch.cycle_publication_refusal(source, cycle, 3, now=NOW,
                                              probe=_rolling_door(source))

    assert refusal == (f"{source.upper()} cycle 2026-09-28T12Z is not published through "
                       f"f003 yet; the newest complete {source.upper()} cycle covering f003 "
                       f"is {_NEWEST[source]} -- pass that, or --cycle latest to resolve it "
                       "automatically")


def test_an_older_cycle_missing_inside_the_retention_will_not_appear_by_waiting():
    """The door answered for the newer run but not this one: it is not still publishing."""

    lost = datetime(2026, 9, 28, 0)
    probe = _rolling_door("icon-eu", dropped={lost})

    refusal = fetch.cycle_publication_refusal("icon-eu", lost, 3, now=NOW, probe=probe)

    assert refusal == ("ICON-EU cycle 2026-09-28T00Z is not on opendata.dwd.de through f003, "
                       "and waiting will not bring it: a newer cycle is already complete; the "
                       f"newest complete ICON-EU cycle covering f003 is {_NEWEST['icon-eu']} -- pass "
                       "that, or --cycle latest to resolve it automatically")


def _rolling_doors():
    """Every route whose hosts all keep a bounded window, read from the table rather than named."""

    return [route for route in fetch_routes.route_ids()
            if fetch.cycle_is_probeable(route) and fetch_endpoints.has_ladder(route)
            and not any(rung.archive for rung in fetch_endpoints.ladder(route))]


def test_the_rolling_doors_are_table_rows():
    # The ICON route the defect was met on is among them by its row alone.
    # ICON global, where it was met too, now has the source.coop mirror of
    # DWD's v1 tree behind it, so it is no longer a rolling door.
    assert "icon-eu" in set(_rolling_doors())
    assert "icon-global" not in set(_rolling_doors())


def test_an_icon_global_run_past_dwds_window_goes_straight_to_the_mirror():
    """The 2026-09-27T00 run the defect was met on is now fetched, from the mirror."""

    cycle = datetime(2026, 9, 27, 0)
    assert [rung.name for rung in fetch_endpoints.serving_ladder(
        "icon-global", cycle=cycle, now=NOW)] == ["source-coop"]
    assert [rung.name for rung in fetch_endpoints.serving_ladder(
        "icon-global", cycle=NOW.replace(hour=6), now=NOW)] == ["dwd"]
    plan = fetch_routes.resolve_request("icon-global", cycle=cycle, hours=3, now=NOW)
    assert fetch_routes.listed_head(plan)
    assert all(obj.url.startswith("https://data.source.coop/bkr/icon/icon/p/")
               for obj in plan.objects)


@pytest.mark.parametrize("source", _rolling_doors())
def test_every_rolling_door_names_its_own_retention(source):
    rungs = fetch_endpoints.ladder(source)
    kept = max(rung.retention_hours for rung in rungs)
    hour = int(fetch_routes.route_for(source).cycle_hours[0])
    cycle = (NOW - timedelta(hours=kept + 24)).replace(hour=hour)

    refusal = fetch.cycle_publication_refusal(source, cycle, 6, now=NOW, probe=lambda url: False)

    assert refusal is not None
    assert "is no longer on the server" in refusal
    assert f"only about the newest {kept:g} h of {source.upper()} cycles" in refusal
    assert "not published" not in refusal


def test_a_pinned_host_names_the_host_that_still_keeps_an_older_cycle():
    """A pinned door without the cycle, and the other host in the row keeps cycles that old."""

    cycle = datetime(2026, 9, 27, 12)
    newer = datetime(2026, 9, 28, 0)
    kept = set(fetch.cycle_probe_urls("ecmwf-open-data", newer, 6, transport="ecmwf"))

    refusal = fetch.cycle_publication_refusal("ecmwf-open-data", cycle, 6, now=NOW,
                                              transport="ecmwf", probe=kept.__contains__)

    assert refusal is not None
    assert refusal.startswith("ECMWF-OPEN-DATA cycle 2026-09-27T12Z is not on data.ecmwf.int "
                              "through f006, and waiting will not bring it: a newer cycle is "
                              "already complete; aws also keeps ECMWF-OPEN-DATA cycles this old "
                              "-- pass --transport aws, or leave --transport off")
    assert "cycle covering f006 is 2026-09-28T00Z" in refusal


# --------------------------------------------------------------------------
# One measurement of each publisher door, read by every question asked of it
# --------------------------------------------------------------------------

def _declared_doors():
    """Every rolling door the date guidance declares, beside its route row's host."""

    from woof import source_adapters

    doors = []
    for adapter in source_adapters.source_adapters():
        if adapter.source_id not in fetch_routes.route_ids():
            continue
        for window in adapter.archive_windows:
            if window.retention_hours is not None:
                doors.append((adapter.source_id, window))
    return doors


def test_the_route_table_declares_each_doors_measured_retention():
    """The breakage: data.ecmwf.int and dd.weather.gc.ca read as archives.

    The date guidance carried their measured windows (72 h and 696 h)
    while the route rows said ``retention_hours: null``, so the fetch
    ladder asked the ECMWF door for a cycle it had dropped days ago and
    ``--cycle latest`` walked back only the 48 h default.
    """

    doors = _declared_doors()
    assert {"ecmwf-open-data", "aifs", "gem-gdps", "icon-eu", "icon-d2",
            "icon-global"} <= {
        source for source, _window in doors}
    for source, window in doors:
        host = fetch_endpoints.endpoint_named(source, window.transport)
        assert host.retention_hours == window.retention_hours, (source, window.transport)


def test_an_archive_rungs_first_run_is_the_date_guidances_first_run():
    """One measurement of where a mirror begins, read by the fetch and the calendar."""

    from woof import source_adapters
    from woof.source_availability import parse_cycle

    checked = 0
    for source in fetch_routes.route_ids():
        windows = {window.transport: window for window in
                   source_adapters.get_source_adapter(source).archive_windows}
        for rung in fetch_endpoints.ladder(source):
            if rung.serves_from is None:
                continue
            assert parse_cycle(windows[rung.name].start) == rung.serves_from, source
            checked += 1
    assert checked >= 1


@pytest.mark.parametrize("source", ["ecmwf-open-data", "aifs"])
def test_an_ecmwf_cycle_past_the_doors_window_goes_straight_to_the_mirror(source):
    recent = NOW - timedelta(hours=48)
    dropped = NOW - timedelta(hours=96)

    assert [rung.name for rung in fetch_endpoints.serving_ladder(
        source, cycle=recent, now=NOW)] == ["ecmwf", "aws"]
    assert [rung.name for rung in fetch_endpoints.serving_ladder(
        source, cycle=dropped, now=NOW)] == ["aws"]
    plan = fetch_routes.resolve_request(source, cycle=dropped.replace(hour=0), hours=6,
                                        now=NOW)
    assert plan.host.name == "aws"
    assert all(obj.url.startswith("https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/")
               for obj in plan.objects)


@pytest.mark.parametrize("source,hours", [
    ("ecmwf-open-data", 72), ("aifs", 72), ("gem-gdps", 696)])
def test_cycle_latest_walks_back_as_far_as_the_door_keeps(source, hours):
    from woof.source_cycles import cycle_grid_for

    assert cycle_grid_for(source).search_hours == hours


def test_a_gem_cycle_past_the_door_is_refused_as_gone():
    cycle = datetime(2026, 8, 28, 0)

    refusal = fetch.cycle_publication_refusal("gem-gdps", cycle, 6, now=NOW,
                                              probe=lambda url: False)

    assert refusal is not None
    assert refusal.startswith("GEM-GDPS cycle 2026-08-28T00Z is no longer on the server: "
                              "dd.weather.gc.ca keeps only about the newest 696 h of "
                              "GEM-GDPS cycles and this one is 756 h old, with no archive "
                              "behind it, so it will not appear by waiting")


def test_a_mirror_run_is_listed_whole_as_the_publication_check(monkeypatch):
    """No lead-by-lead probe of a listed mirror: the run is listed, and a part-run named."""

    cycle = datetime(2026, 9, 27, 0)
    plan = fetch_routes.resolve_request("icon-global", cycle=cycle, hours=3, now=NOW)
    held = {obj.key for obj in plan.objects}
    gone = "p/W_SO/lvt1/106/lv1/7.29/r/2026-09-27T00:00/s/PT003H00M.grib2"
    assert gone in held

    def listed(endpoint, directory):
        return frozenset(key for key in held if key.startswith(directory))

    held.discard(gone)

    def probe(url):
        raise AssertionError("a listed mirror is not probed lead by lead")

    monkeypatch.setattr(fetch_routes, "_listed_keys", listed)
    with pytest.raises(RuntimeError, match="only in part: 1 of 236") as refusal:
        fetch.require_published_cycle("icon-global", cycle, 3, now=NOW, probe=probe,
                                      progress=lambda line: None)
    assert gone in str(refusal.value)
    held.add(gone)
    fetch.require_published_cycle("icon-global", cycle, 3, now=NOW, probe=probe,
                                  progress=lambda line: None)
