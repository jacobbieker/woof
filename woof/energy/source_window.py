"""Whether a 10 km global boundary source serves the cycle a plan asks for.

The energy planners (``wrf-nests``, ``wrf-tiles`` and the ``hex-swath``
experimental chain) offer three ~10 km global sources besides the NCEP and
ECMWF ones they always took:

* ``icon-global`` -- DWD ICON global on pressure levels (18 levels,
  1000 to 30 hPa), 3 h boundaries;
* ``icon-global-ml`` -- the same model on its 120 native model levels;
* ``met-office-global`` -- the Met Office global deterministic model on
  pressure levels (37 for some fields, 33 for others, to 10 hPa), 3 h
  boundaries, with soil, land mask and sea ice taken from the same-cycle
  GDAS analysis.

Each of them publishes only inside a window: a rolling publisher, an
archive with a first cycle, a forecast range that depends on the cycle
hour, and a delay before a run lands.  A plan that names a cycle outside
it would emit configurations whose first command (the fetch) fails, so
:func:`window_refusal` refuses it at plan time and names the bound.

Facts are read from the fetch route table (:mod:`woof.fetch_routes`) and
its endpoint ladder (:mod:`woof.fetch_endpoints`) whenever this build has a
route for the source: cycle hours, the lead ladder per cycle hour, the
measured publication lag and each endpoint's retention.  The declared rows
below (:data:`WINDOWS`) hold what the route table cannot say -- where an
archive starts and how long a rolling archive keeps -- and stand in for a
route this build does not have yet.

Only these three sources are judged here; every other source keeps the
behaviour it had (the fetch is its gate).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Mapping


@dataclass(frozen=True)
class SourceWindow:
    """One boundary source's publication window."""

    source: str
    label: str
    #: Hours between the boundary leads the planners fetch.
    cadence_hours: int
    #: Cycle hour (UTC) -> the last forecast lead that run publishes.
    horizons: Mapping[int, int]
    #: Hours after the run time until the last lead lands.
    publication_lag_hours: float
    #: The first cycle any store this build reads holds; None when the
    #: oldest cycle is set by a rolling retention instead.
    archive_start: datetime | None
    #: How far back a rolling archive keeps, hours; None for no bound.
    retention_hours: float | None
    #: The stores, in words, for the refusal.
    stores: str
    #: Registry aliases (refusal wording only; the registry resolves them).
    aliases: tuple[str, ...] = ()
    #: Fields the source does not publish, and the source that supplies them.
    land_state_from: str | None = None


#: The first cycle of the source.coop ICON archive (``p/`` tree).
ICON_ARCHIVE_START = datetime(2026, 8, 25, 12)
#: DWD open data (v1 tree) keeps about this many hours of runs (the
#: route table's dwd rung, measured 2026-10-11: the last four runs, each
#: dropped about 27 h after its cycle).
ICON_DWD_RETENTION_HOURS = 27.0
#: The Met Office AWS open-data archive is a rolling two-year window.
MET_OFFICE_RETENTION_HOURS = 2 * 365 * 24.0

_ICON_STORES = (
    "DWD open data (opendata.dwd.de/weather/nwp/v1/m/icon/p/) keeps about "
    f"the last {ICON_DWD_RETENTION_HOURS:g} h, and the source.coop archive "
    "(data.source.coop/bkr/icon/icon/, p/ tree) holds "
    f"{ICON_ARCHIVE_START:%Y-%m-%dT%H}Z onward")
_ICON_HORIZONS = {0: 180, 6: 120, 12: 180, 18: 120}

WINDOWS: Mapping[str, SourceWindow] = {
    "icon-global": SourceWindow(
        source="icon-global", label="DWD ICON global (pressure levels)",
        cadence_hours=3, horizons=_ICON_HORIZONS,
        # The route table's measured f180 posting is about 3 h 29 min.
        publication_lag_hours=3.5, archive_start=ICON_ARCHIVE_START,
        retention_hours=None, stores=_ICON_STORES),
    "icon-global-ml": SourceWindow(
        source="icon-global-ml", label="DWD ICON global (model levels)",
        cadence_hours=3, horizons=_ICON_HORIZONS,
        publication_lag_hours=3.5, archive_start=ICON_ARCHIVE_START,
        retention_hours=None, stores=_ICON_STORES),
    "met-office-global": SourceWindow(
        source="met-office-global", label="Met Office global deterministic",
        # The 00/12Z runs reach f168, but 3-hourly only to f144 (6-hourly
        # after it), so a 3 h boundary series ends at f144: the registry
        # row's max_forecast_hour and the route's ladder agree.
        cadence_hours=3, horizons={0: 144, 6: 69, 12: 144, 18: 69},
        publication_lag_hours=5.5, archive_start=None,
        retention_hours=MET_OFFICE_RETENTION_HOURS,
        stores=("the AWS open-data archive "
                "(registry.opendata.aws/met-office-global-deterministic) "
                "keeps a rolling two years"),
        aliases=("metoffice-global", "ukmo-global", "um-global"),
        land_state_from="gdas"),
}


def _utc_naive(moment: datetime | None) -> datetime:
    if moment is None:
        moment = datetime.now(timezone.utc)
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc).replace(tzinfo=None)
    return moment


def planned_source(name: str) -> str | None:
    """The :data:`WINDOWS` id ``name`` spells (id or alias), else None.

    The registry resolves the name first, so every alias a registered row
    declares counts; the declared aliases cover a build without the row.
    """

    from woof.source_adapters import get_source_adapter

    try:
        resolved = get_source_adapter(str(name)).source_id
    except ValueError:
        resolved = None
    if resolved in WINDOWS:
        return resolved
    key = str(name).strip().lower().replace("_", "-")
    for window in WINDOWS.values():
        if key == window.source or key in window.aliases:
            return window.source
    return None


def canonical_source(name: str) -> str:
    """``name`` as its registry id, or as typed when the registry lacks it.

    An unregistered name is the caller's to refuse (:func:`registry_gap`);
    this never raises.
    """

    from woof.source_adapters import get_source_adapter

    try:
        return get_source_adapter(str(name)).source_id
    except ValueError:
        return planned_source(name) or str(name).strip().lower()


def registry_gap(name: str) -> str | None:
    """Why a source the planners offer cannot be planned in this build.

    None when the registry has the row (or the name is not one of
    :data:`WINDOWS`: other names keep the refusal their planner already
    gives).  The three sources here land as separate registry rows; until
    a build carries the row, a plan for it is refused naming the gap.
    """

    from woof.source_adapters import get_source_adapter

    source = planned_source(name)
    if source is None:
        return None
    try:
        get_source_adapter(source)
    except ValueError:
        return (f"--source {name}: {WINDOWS[source].label} is a boundary "
                f"source the energy planners offer, but this build has no "
                f"registry row for {source!r} (woof.source_adapters), so "
                f"neither `woof fetch` nor `woof domain` can take it.  "
                f"Use a build that registers it, or another --source")
    return None


def _route(source: str):
    from woof import fetch_routes

    try:
        return fetch_routes.table_route(source)
    except Exception:                        # noqa: BLE001 - absent is absent
        return None


def _rolling_endpoints(source: str) -> tuple[tuple[str, float], ...] | None:
    """(name, retention) of every endpoint when ALL are rolling, else None.

    None too when the declared row's own rolling retention is no longer
    than what the endpoints keep: then the rolling store IS the archive
    (the Met Office bucket's two years), and there is no other one the
    route could read.
    """

    from woof import fetch_endpoints

    try:
        rungs = fetch_endpoints.ladder(source)
    except Exception:                        # noqa: BLE001 - absent is absent
        return None
    if not rungs or any(rung.retention_hours is None for rung in rungs):
        return None
    declared = WINDOWS[source].retention_hours if source in WINDOWS else None
    if declared is not None and max(
            float(rung.retention_hours) for rung in rungs) >= declared:
        return None
    return tuple((rung.name, float(rung.retention_hours)) for rung in rungs)


def last_lead(source: str, hours: float) -> int:
    """The last boundary lead a ``hours``-long run fetches from ``source``."""

    cadence = WINDOWS[source].cadence_hours
    return cadence * max(1, int(math.ceil(float(hours) / cadence - 1e-9)))


def _lag_hours(source: str, route, start: datetime, lead: int) -> float:
    """Hours after ``start`` until ``lead`` lands: the route's, else the row's."""

    if route is not None:
        try:
            return float(route.publication_lag_hours(start.hour, lead,
                                                     settled=True))
        except Exception:                    # noqa: BLE001 - keep the row's
            pass
    return WINDOWS[source].publication_lag_hours


def newest_landed_start(name: str, start: datetime, hours: float, *,
                        now: datetime | None = None) -> datetime:
    """``start``, stepped back a cycle at a time until its run has landed.

    For a DEFAULTED start only: the planners default to the newest
    00/06/12/18 cycle by the clock, which a source with a publication
    delay has not published for part of every 6 h.  A start the user
    typed is never moved (:func:`window_refusal` refuses it instead).  A
    source outside :data:`WINDOWS` gets ``start`` back unchanged.
    """

    source = planned_source(name)
    if source is None:
        return start
    now = _utc_naive(now)
    route = _route(source)
    lead = last_lead(source, hours)
    candidate = _utc_naive(start)
    for _ in range(8):                       # two days of 6-hourly cycles
        if now >= candidate + timedelta(
                hours=_lag_hours(source, route, candidate, lead)):
            return candidate
        candidate -= timedelta(hours=6)
    return _utc_naive(start)


def window_refusal(name: str, start: datetime, hours: float, *,
                   now: datetime | None = None) -> str | None:
    """Why ``name`` cannot force a ``hours``-long run from ``start``, or None.

    Checked in order: the cycle hour, the forecast range of that cycle
    hour, a run that has not landed yet, and the oldest cycle the stores
    hold.  A source outside :data:`WINDOWS` is not judged (None).
    """

    source = planned_source(name)
    if source is None:
        return None
    window = WINDOWS[source]
    start = _utc_naive(start)
    now = _utc_naive(now)
    head = f"--source {source} --start {start:%Y-%m-%dT%H}"
    lead = last_lead(source, hours)
    route = _route(source)

    # cycle hours and forecast range: the route's ladder when there is one
    cycle_hours = tuple(sorted(route.cycle_hours if route is not None
                               else window.horizons))
    if start.hour not in cycle_hours:
        offered = "/".join(f"{hour:02d}" for hour in cycle_hours)
        return (f"{head}: {window.label} runs at {offered} UTC only, not "
                f"{start:%H}Z")
    if route is not None:
        from woof import fetch_routes

        try:
            ladder = fetch_routes.ladder_for(route, start)
        except ValueError as error:
            return f"{head}: {error}"
        horizon = ladder[-1]
        basis = "the fetch route's lead ladder"
        # every boundary lead must be published, not only the last: a
        # ladder that thins out (6 h steps late in the run) has gaps
        published = set(ladder)
        gaps = [step for step in range(0, min(lead, horizon) + 1,
                                       window.cadence_hours)
                if step not in published]
        if gaps:
            return (f"{head}: --hours {hours:g} needs a boundary every "
                    f"{window.cadence_hours} h, and the {start:%H}Z "
                    f"{window.label} run publishes no f{gaps[0]:03d} "
                    f"({basis}).  Shorten --hours to "
                    f"{max(gaps[0] - window.cadence_hours, 0)} h or less")
    else:
        horizon = window.horizons[start.hour]
        basis = ", ".join(
            f"{'/'.join(f'{h:02d}' for h in sorted(hrs))}Z to {last} h"
            for last, hrs in _by_horizon(window.horizons))
    if lead > horizon:
        return (f"{head}: --hours {hours:g} needs boundaries to f{lead:03d} "
                f"(every {window.cadence_hours} h), and the {start:%H}Z "
                f"{window.label} run ends at f{horizon:03d} ({basis}).  "
                f"Shorten --hours or start from a longer run")

    # publication: the route's measured lag for that lead when there is one
    lag = _lag_hours(source, route, start, lead)
    landed = start + timedelta(hours=lag)
    if now < landed:
        return (f"{head}: the {window.label} run lands about {lag:.1f} h "
                f"after its run time (f{lead:03d} around "
                f"{landed:%Y-%m-%dT%H:%M}Z) and the clock reads "
                f"{now:%Y-%m-%dT%H:%M}Z; start from an earlier cycle")

    # the oldest cycle: the declared archive bounds, then this build's route
    age = (now - start).total_seconds() / 3600.0
    if window.archive_start is not None and start < window.archive_start:
        return (f"{head}: before {window.archive_start:%Y-%m-%dT%H}Z no "
                f"store holds {window.label} runs: {window.stores}")
    if window.retention_hours is not None and age > window.retention_hours:
        return (f"{head}: the run is {age / 24.0:.0f} days old, outside "
                f"the {window.retention_hours / 24.0:.0f}-day window: "
                f"{window.stores}")
    rolling = _rolling_endpoints(source)
    if rolling is not None and age > max(kept for _, kept in rolling):
        hosts = ", ".join(f"{name} (about {kept:g} h)"
                          for name, kept in rolling)
        return (f"{head}: the run is {age:.0f} h old and this build's fetch "
                f"route for {source} reads only rolling publishers: {hosts}.  "
                f"{window.stores}; this build has no route to that archive")
    if route is not None and route.publication_eras:
        from woof import fetch_routes

        try:
            fetch_routes.publication_era(route, start)
        except ValueError as error:
            return f"{head}: {error}"
    return None


def _by_horizon(horizons: Mapping[int, int]):
    grouped: dict[int, list[int]] = {}
    for hour, last in horizons.items():
        grouped.setdefault(last, []).append(hour)
    return sorted(grouped.items(), key=lambda item: -item[0])


def source_notes(name: str) -> list[str]:
    """Plan notes for a source in :data:`WINDOWS` (empty for any other)."""

    source = planned_source(name)
    if source is None:
        return []
    window = WINDOWS[source]
    notes = [f"{source}: {window.label}, {window.cadence_hours} h boundaries; "
             f"{window.stores}; "
             + ", ".join(f"{'/'.join(f'{h:02d}' for h in sorted(hrs))}Z runs "
                         f"reach {last} h"
                         for last, hrs in _by_horizon(window.horizons))]
    rolling = _rolling_endpoints(source)
    if rolling is not None:
        hosts = ", ".join(f"{host} (about {kept:g} h)"
                          for host, kept in rolling)
        notes.append(
            f"{source}: this build's fetch route reads only rolling "
            f"publishers ({hosts}), so only runs that recent are plannable "
            "here; an older run is refused until the route reads the archive")
    if window.land_state_from is not None:
        donor = window.land_state_from
        route = _route(source)
        declared = route is not None and any(
            row.source == donor and row.cycle == "same"
            for row in route.donors)
        notes.append(
            f"{source} publishes no soil, land mask or sea ice; they come "
            f"from the same-cycle {donor.upper()} analysis"
            + (" (a donor of the fetch route, fetched with it)" if declared
               else ""))
    return notes


__all__ = [
    "ICON_ARCHIVE_START", "SourceWindow", "WINDOWS", "canonical_source",
    "last_lead", "newest_landed_start", "planned_source", "registry_gap",
    "source_notes", "window_refusal",
]
