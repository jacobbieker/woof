"""Choose the boundary source for a region and a cycle: ``--source auto``.

A year-long hindcast over every public wind farm cannot name one forcing
source: HRRR reaches only the CONUS, the Met Office global archive is a
rolling two years whose oldest runs cannot be prepared, ECMWF's open data
can be prepared only from 2024-03-18 12Z, and ERA5 reaches everywhere and
everything but at 31 km.  :func:`select_source` walks a priority list and
takes the first candidate that passes every check, recording why each
other candidate was passed over.

Every check reads a table this build already ships; nothing reaches a
server, and the clock is injected (``now``), so the same arguments give
the same answer.  The checks, in the order a receipt lists them:

``registry``
    The name resolves in :mod:`woof.source_adapters` and the row is
    ``runnable``.
``acquisition``
    The fetch route table carries an acquisition route for it
    (:func:`woof.fetch_routes.all_fetchable_sources`) and its launch chain
    does not need bytes already on disk
    (:func:`woof.source_drivability.drivability_for`).
``planner``
    The calling planner's own refusal (``refuse``), e.g. ``wrf-nests``
    cannot emit HRRR's route documents.
``coverage``
    Every sample of the region's bounding box, widened by ``margin_km``,
    lies inside the row's declared native-grid window
    (:func:`woof.source_coverage.points_outside`).  A global row covers
    everything.
``horizon``
    The cycle's run reaches ``hours + spinup_hours``: the route table's
    lead ladder for that cycle when it has one, else the row's cycle-grid
    horizons, else ``max_forecast_hour``.  A reanalysis has no horizon.
``column_top``
    The source reaches the requested model top ``p_top_pa``: no certified
    top declared (the preparation's vertical-coverage check is the gate),
    a certified top at or below it, or a fetch that extends its ladder to
    it (``extendable_source_top_pa``, :func:`fetch_model_top_pa`).
``date``
    The calendar's verdict for the cycle
    (:func:`woof.source_availability.verdict`): the archive's first cycle,
    a rolling door's retention, a publication era that cannot be prepared
    (route ``prep_refusal``), the cycle hours for a run this long, and a
    run not yet published by ``now``.  A row whose archive start is not
    declared answers ``unknown``; that is a rejection unless
    ``accept_unknown_archive`` is set (the selector does not guess).
``window``
    For the sources :mod:`woof.energy.source_window` judges (ICON global,
    Met Office global): the planners' own window refusal, so a chosen
    source is never one the planner then refuses.

Pinned here: :data:`DEFAULT_PRIORITY`, the check names above
(:data:`CHECKS`), the receipt schema :data:`SCHEMA`, and the coverage
sampling (:data:`COVERAGE_STEP_DEG`, :data:`COVERAGE_MAX_PER_AXIS`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from woof.energy.contracts import SOURCE_CHOICE_SCHEMA as SCHEMA

#: Where a planner writes the receipt, beside ``plan.json``.
RECEIPT_FILE = "source-choice.json"

#: A defaulted start (:func:`select_newest`) looks back this many cycles of
#: :data:`LOOKBACK_STEP_HOURS` from the clock's newest 00/06/12/18Z cycle.
LOOKBACK_CYCLES = 8
LOOKBACK_STEP_HOURS = 6

#: Regional sources first (each is ranked out wherever its grid does not
#: reach), then the global ones in order of preference; ERA5 is the last
#: resort that reaches every date since 1940 at 31 km.
DEFAULT_PRIORITY: tuple[str, ...] = (
    "hrrr-native", "hrrr", "rap", "met-office-global", "ecmwf-open-data",
    "gfs", "era5")

#: The checks a candidate is put to, in receipt order.
CHECKS: tuple[str, ...] = (
    "registry", "acquisition", "planner", "coverage", "horizon",
    "column_top", "date", "window")

#: Spacing (degrees) of the grid the region's bounding box is sampled on
#: for the coverage check, edges included.
COVERAGE_STEP_DEG = 0.25
#: Most samples along one axis of that grid (a continent-wide box is
#: sampled more coarsely; its edges are always included).
COVERAGE_MAX_PER_AXIS = 121

#: Kilometres a planner adds to its own estimate of how far its outermost
#: domain reaches past the sites, for the wizard's whole-cell rounding of
#: the root it fits (see each planner's ``_auto_margin_km``).
PLANNER_SLACK_KM = 10.0

#: Kilometres per degree of latitude (spherical, the WPS radius).
_KM_PER_DEG = 6370.0 * math.pi / 180.0


class SourceSelectRefused(ValueError):
    """No candidate qualifies, or the request itself is malformed.

    ``choice`` carries the full receipt when candidates were judged
    (every candidate with its rejections), else None.
    """

    def __init__(self, message: str, choice: "SourceChoice | None" = None):
        super().__init__(message)
        self.choice = choice


@dataclass(frozen=True)
class Rejection:
    """One failed check of one candidate."""

    check: str
    reason: str

    def to_json(self) -> dict[str, str]:
        return {"check": self.check, "reason": self.reason}


@dataclass(frozen=True)
class Candidate:
    """One entry of the priority list, as judged."""

    rank: int
    requested: str
    source: str
    rejections: tuple[Rejection, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def eligible(self) -> bool:
        return not self.rejections

    def to_json(self, chosen: str | None) -> dict[str, Any]:
        status = ("chosen" if self.eligible and self.source == chosen
                  else "eligible" if self.eligible else "rejected")
        return {"rank": self.rank, "requested": self.requested,
                "source": self.source, "status": status,
                "rejections": [r.to_json() for r in self.rejections],
                "notes": list(self.notes)}


@dataclass(frozen=True)
class SourceChoice:
    """The chosen source and the judgement of every candidate.

    ``source`` is None only on the receipt a :class:`SourceSelectRefused`
    carries.
    """

    source: str | None
    cycle: datetime
    hours: float
    spinup_hours: float
    p_top_pa: float | None
    priority: tuple[str, ...]
    priority_origin: str
    region: Mapping[str, Any]
    now: datetime
    accept_unknown_archive: bool
    candidates: tuple[Candidate, ...]
    route_table_sha256: str | None = None
    notes: tuple[str, ...] = field(default=())

    @property
    def horizon_hours(self) -> float:
        return self.hours + self.spinup_hours

    @property
    def rejections(self) -> dict[str, list[str]]:
        """``source -> ["check: reason", ...]`` for every rejected candidate."""

        return {c.source: [f"{r.check}: {r.reason}" for r in c.rejections]
                for c in self.candidates if c.rejections}

    def chosen(self) -> Candidate | None:
        for candidate in self.candidates:
            if candidate.eligible and candidate.source == self.source:
                return candidate
        return None

    def to_json(self) -> dict[str, Any]:
        """The ``woof-energy.source-choice.v1`` receipt."""

        return {
            "schema": SCHEMA,
            "source": self.source,
            "cycle": self.cycle.strftime("%Y-%m-%dT%H"),
            "hours": self.hours,
            "spinup_hours": self.spinup_hours,
            "horizon_hours": self.horizon_hours,
            "p_top_pa": self.p_top_pa,
            "priority": list(self.priority),
            "priority_origin": self.priority_origin,
            "region": dict(self.region),
            "now": self.now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "accept_unknown_archive": self.accept_unknown_archive,
            "checks": list(CHECKS),
            "candidates": [c.to_json(self.source) for c in self.candidates],
            "route_table_sha256": self.route_table_sha256,
            "notes": list(self.notes),
        }

    def summary(self) -> dict[str, Any]:
        """A compact form for a CLI's JSON record (:func:`choice_summary`)."""

        return choice_summary(self.to_json())

    def note(self) -> str:
        """One plan note naming the choice and what was passed over."""

        passed = []
        for candidate in self.candidates:
            if candidate.source == self.source:
                break
            first = candidate.rejections[0]
            passed.append(f"{candidate.source} ({first.check})")
        return (f"--source auto chose {self.source} for the "
                f"{self.cycle:%Y-%m-%dT%H}Z cycle "
                f"({self.horizon_hours:g} h) from "
                f"{','.join(self.priority)}"
                + (f"; passed over {', '.join(passed)}" if passed else "")
                + f"; receipt schema {SCHEMA}")


# --------------------------------------------------------------------------
# inputs


def _utc_naive(moment: datetime | None) -> datetime:
    if moment is None:
        moment = datetime.now(timezone.utc)
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc).replace(tzinfo=None)
    return moment


def parse_cycle(cycle: datetime | str) -> datetime:
    """``cycle`` as a naive UTC datetime on a whole hour, else refused."""

    if isinstance(cycle, datetime):
        parsed = _utc_naive(cycle)
    else:
        text = str(cycle).strip()
        if text.endswith("Z"):
            text = text[:-1]
        parsed = None
        for form in ("%Y-%m-%dT%H", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S"):
            try:
                parsed = datetime.strptime(text, form)
                break
            except ValueError:
                continue
        if parsed is None:
            raise SourceSelectRefused(
                f"cycle {cycle!r} is not YYYY-MM-DDTHH (UTC)")
    if parsed.minute or parsed.second or parsed.microsecond:
        raise SourceSelectRefused(
            f"cycle {cycle!r} is not on a whole hour: it is the source cycle "
            "the run starts from")
    return parsed


def _finite_hours(value, label: str, *, positive: bool) -> float:
    if isinstance(value, bool):
        raise SourceSelectRefused(f"{label} {value!r} is not a number of hours")
    try:
        hours = float(value)
    except (TypeError, ValueError) as error:
        raise SourceSelectRefused(
            f"{label} {value!r} is not a number of hours") from error
    if not math.isfinite(hours) or hours < 0.0 or (positive and hours == 0.0):
        raise SourceSelectRefused(
            f"{label} {value!r} must be a finite "
            f"{'positive' if positive else 'nonnegative'} number of hours")
    return hours


def _coordinates(geometry: Mapping[str, Any]) -> list[tuple[float, float]]:
    kind = geometry.get("type")
    if kind == "FeatureCollection":
        points: list[tuple[float, float]] = []
        for feature in geometry.get("features") or ():
            points += _coordinates(feature)
        return points
    if kind == "Feature":
        return _coordinates(geometry.get("geometry") or {})
    if kind == "GeometryCollection":
        points = []
        for member in geometry.get("geometries") or ():
            points += _coordinates(member)
        return points
    if kind not in ("Point", "MultiPoint", "LineString", "MultiLineString",
                    "Polygon", "MultiPolygon"):
        raise SourceSelectRefused(
            f"region GeoJSON type {kind!r} is not a geometry, Feature or "
            "FeatureCollection")

    def walk(node) -> Iterable[tuple[float, float]]:
        if (isinstance(node, (list, tuple)) and len(node) >= 2
                and all(isinstance(v, (int, float)) for v in node[:2])):
            yield float(node[0]), float(node[1])
            return
        for child in node or ():
            yield from walk(child)

    return list(walk(geometry.get("coordinates")))


def region_bbox(region) -> tuple[float, float, float, float]:
    """``(west, south, east, north)`` of ``region``, refused when empty.

    ``region`` is a GeoJSON geometry, Feature or FeatureCollection; a
    mapping with a ``bbox``; a ``(west, south, east, north)`` 4-tuple; or
    a sequence of ``(lon, lat)`` points.  A region spanning more than 180
    degrees of longitude (one crossing the antimeridian) is refused: plan
    each side of it separately.
    """

    if isinstance(region, Mapping):
        if "bbox" in region and "type" not in region:
            return region_bbox(tuple(region["bbox"]))
        points = _coordinates(region)
    elif (isinstance(region, (list, tuple)) and len(region) == 4
          and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                  for v in region)):
        west, south, east, north = (float(v) for v in region)
        points = [(west, south), (east, north)]
        if not (west <= east and south <= north):
            raise SourceSelectRefused(
                f"region bbox {tuple(region)!r} needs west <= east and "
                "south <= north (west,south,east,north)")
    else:
        try:
            array = np.asarray(region, dtype=float)
        except (TypeError, ValueError) as error:
            raise SourceSelectRefused(
                "region is not GeoJSON, a west,south,east,north box or a "
                "list of (lon, lat) points") from error
        if array.ndim != 2 or array.shape[1] != 2:
            raise SourceSelectRefused(
                "region points must be a list of (lon, lat) pairs")
        points = [(float(lon), float(lat)) for lon, lat in array]
    if not points:
        raise SourceSelectRefused("region has no coordinates")
    lon = np.array([p[0] for p in points], dtype=float)
    lat = np.array([p[1] for p in points], dtype=float)
    if not (np.all(np.isfinite(lon)) and np.all(np.isfinite(lat))):
        raise SourceSelectRefused("region coordinates must be finite")
    if np.any(np.abs(lat) > 90.0) or np.any(np.abs(lon) > 360.0):
        raise SourceSelectRefused(
            "region coordinates are not (lon, lat) degrees")
    lon = (lon + 180.0) % 360.0 - 180.0
    west, east = float(lon.min()), float(lon.max())
    if east - west > 180.0:
        raise SourceSelectRefused(
            "the region spans more than 180 degrees of longitude (it "
            "crosses the antimeridian); select for each side separately")
    return west, float(lat.min()), east, float(lat.max())


def _widen(bbox, margin_km: float) -> tuple[float, float, float, float]:
    west, south, east, north = bbox
    dlat = margin_km / _KM_PER_DEG
    south, north = max(south - dlat, -89.9), min(north + dlat, 89.9)
    widest = math.cos(math.radians(max(abs(south), abs(north))))
    dlon = margin_km / (_KM_PER_DEG * max(widest, 1e-3))
    # Not clamped at the antimeridian: the samples are wrapped instead
    # (:func:`_samples`), so a margin reaching across it is checked.
    return (west - dlon, south, east + dlon, north)


def _samples(bbox) -> tuple[np.ndarray, np.ndarray]:
    west, south, east, north = bbox

    def axis(lo: float, hi: float) -> np.ndarray:
        count = int(math.ceil((hi - lo) / COVERAGE_STEP_DEG - 1e-9)) + 1
        return np.linspace(lo, hi, max(2, min(count, COVERAGE_MAX_PER_AXIS)))

    lon, lat = np.meshgrid(axis(west, east), axis(south, north))
    lon = (lon + 180.0) % 360.0 - 180.0
    return lat.ravel(), lon.ravel()


# --------------------------------------------------------------------------
# the checks


def _resolve(name: str):
    from woof.source_adapters import get_source_adapter

    try:
        return get_source_adapter(str(name))
    except ValueError:
        return None


def _registry_check(adapter) -> str | None:
    if not adapter.runnable:
        return (f"the registry row for {adapter.source_id} is not runnable "
                f"(status {adapter.status.value}); no strict implementation "
                "route exists")
    return None


def _acquisition_check(source: str) -> str | None:
    from woof import fetch_routes
    from woof.source_drivability import drivability_for

    if source not in fetch_routes.all_fetchable_sources():
        return ("no acquisition route: "
                + fetch_routes.acquisition_refusal_reason(source))
    verdict = drivability_for(source) or {}
    if verdict.get("requires_source_root"):
        return ("its launch chain reads bytes already on disk "
                "(requires_source_root); an automatic plan cannot supply them")
    if verdict.get("refusal"):
        return f"no launch chain: {verdict['refusal']}"
    return None


def _coverage_check(adapter, lat: np.ndarray, lon: np.ndarray) -> str | None:
    from woof.source_coverage import points_outside

    window = adapter.coverage_window
    if window is None:
        return None
    outside = points_outside(window, lat, lon)
    if not bool(outside.any()):
        return None
    index = int(np.argmax(outside))
    lat0, lon0 = float(lat[index]), float(lon[index])
    return (f"{int(outside.sum())} of {outside.size} region samples lie "
            f"outside its native grid ({window.describe()}); "
            f"e.g. lat/lon ({lat0:.3f}, {lon0:.3f}) "
            f"{window.locate(lat0, lon0)}")


def _run_horizon(adapter, cycle: datetime) -> tuple[int | None, str]:
    """(last lead the cycle's run publishes, basis), or (None, why)."""

    from woof import fetch_routes

    route = fetch_routes.table_route(adapter.source_id)
    if route is not None:
        try:
            ladder = fetch_routes.ladder_for(route, cycle)
        except ValueError:
            ladder = ()
        if ladder:
            return int(ladder[-1]), "the fetch route's lead ladder"
    grid = adapter.cycle_grid
    if grid is not None and grid.horizons:
        for hours, last in grid.horizons:
            if hours is None or cycle.hour in hours:
                return int(last), "the registry's cycle-grid horizons"
    return int(adapter.max_forecast_hour), "the registry's max_forecast_hour"


def _horizon_check(adapter, cycle: datetime, horizon: float) -> str | None:
    if adapter.max_forecast_hour == 0 or adapter.record_kind != "forecast":
        return None                      # an analysis series: no horizon
    last, basis = _run_horizon(adapter, cycle)
    needed = int(math.ceil(horizon - 1e-9))
    if last is not None and needed > last:
        return (f"the {cycle:%H}Z run ends at f{last:03d} ({basis}) and the "
                f"run needs f{needed:03d} (hours plus spin-up)")
    return None


def _column_top_check(adapter, p_top_pa: float | None
                      ) -> tuple[str | None, str | None]:
    """(rejection, note) for the requested model top."""

    from woof.source_adapters import fetch_model_top_pa

    if p_top_pa is None:
        return None, None
    ceiling = adapter.certified_source_top_pa
    if ceiling is None:
        return None, (f"{adapter.source_id}: no certified source top is "
                      "declared; the preparation's vertical-coverage check "
                      f"gates the {p_top_pa:g} Pa model top")
    if p_top_pa >= float(ceiling):
        return None, None
    reach = adapter.extendable_source_top_pa
    if (fetch_model_top_pa(adapter.source_id, p_top_pa) is not None
            and reach is not None and p_top_pa >= float(reach)):
        return None, (f"{adapter.source_id}: certified ladder stops at "
                      f"{float(ceiling):g} Pa; the fetch extends it to the "
                      f"{p_top_pa:g} Pa model top")
    deepest = float(reach) if reach is not None else float(ceiling)
    return (f"the model top {p_top_pa:g} Pa is above the deepest top it can "
            f"serve ({deepest:g} Pa)"), None


def _date_check(source: str, cycle: datetime, horizon: float, now: datetime,
                accept_unknown: bool) -> tuple[str | None, str | None]:
    """(rejection, note) from the calendar's no-network verdict."""

    from woof.source_availability import availability, verdict

    try:
        document = availability(source, horizon, now=now)
        answer = verdict(document, cycle.strftime("%Y-%m-%dT%H"), now=now)
    except ValueError as error:
        return f"the calendar refuses it: {error}", None
    state = answer.get("state")
    reason = str(answer.get("why") or answer.get("reason") or "").strip()
    if state == "yes":
        return _usual_delay_check(source, cycle, now), None
    if state == "unknown":
        if accept_unknown:
            return None, (f"{source}: {reason} (accepted: "
                          "accept_unknown_archive)")
        return (f"{reason}  The selector does not guess an archive start; "
                "accept_unknown_archive=True lets the download decide"), None
    fix = answer.get("fix")
    return (reason + (f"  {fix}" if fix else "")) or "no data for this cycle", \
        None


def _usual_delay_check(source: str, cycle: datetime,
                       now: datetime) -> str | None:
    """A run younger than its source's usual whole-run publication delay.

    The calendar's verdict leaves a run that recent to the fetch's object
    probe; with no probe here the selector does not take it, so a fresh
    cycle is never chosen for a source that has not posted it yet.
    """

    from woof.source_cycles import cycle_grid_for

    try:
        grid = cycle_grid_for(source)
    except ValueError:
        grid = None
    if grid is None:
        return None
    delay = float(grid.usual_delay)
    if delay <= 0.0 or now >= cycle + timedelta(hours=delay):
        return None
    return (f"a whole {cycle:%H}Z run is usually published {delay:g} h after "
            f"its start (around {cycle + timedelta(hours=delay):%Y-%m-%dT%H:%M}"
            f"Z) and the clock reads {now:%Y-%m-%dT%H:%M}Z; the selector "
            "reaches no server to confirm an earlier posting")


def _window_check(source: str, cycle: datetime, horizon: float,
                  now: datetime) -> str | None:
    from woof.energy.source_window import window_refusal

    return window_refusal(source, cycle, horizon, now=now)


def judge(requested: str, rank: int, *, cycle: datetime, hours: float,
          spinup_hours: float, p_top_pa: float | None, lat: np.ndarray,
          lon: np.ndarray, now: datetime, accept_unknown_archive: bool,
          refuse: Callable[[str], str | None] | None) -> Candidate:
    """One candidate put to every check (see the module docstring)."""

    adapter = _resolve(requested)
    if adapter is None:
        return Candidate(rank=rank, requested=requested,
                         source=str(requested).strip().lower(),
                         rejections=(Rejection(
                             "registry", f"{requested!r} is not a registered "
                             "source"),))
    source = adapter.source_id
    horizon = hours + spinup_hours
    rejections: list[Rejection] = []
    notes: list[str] = []

    def add(check: str, reason: str | None, note: str | None = None):
        if reason:
            rejections.append(Rejection(check, reason))
        if note:
            notes.append(note)

    add("registry", _registry_check(adapter))
    add("acquisition", _acquisition_check(source))
    if refuse is not None:
        add("planner", refuse(source))
    add("coverage", _coverage_check(adapter, lat, lon))
    add("horizon", _horizon_check(adapter, cycle, horizon))
    add("column_top", *_column_top_check(adapter, p_top_pa))
    date_reason, date_note = _date_check(source, cycle, horizon, now,
                                         accept_unknown_archive)
    add("date", date_reason, date_note)
    if date_reason is None:
        # The planners' own window: same facts, asked the way they ask.
        add("window", _window_check(source, cycle, horizon, now))
    return Candidate(rank=rank, requested=requested, source=source,
                     rejections=tuple(rejections), notes=tuple(notes))


# --------------------------------------------------------------------------
# the entry point


def _priority(priority: Sequence[str] | str | None
              ) -> tuple[tuple[str, ...], str]:
    if priority is None:
        return DEFAULT_PRIORITY, "default"
    if isinstance(priority, str):
        names = tuple(p.strip() for p in priority.split(",") if p.strip())
    else:
        names = tuple(str(p).strip() for p in priority if str(p).strip())
    if not names:
        raise SourceSelectRefused("the source priority list is empty")
    if any(name.lower() == "auto" for name in names):
        raise SourceSelectRefused("'auto' cannot appear in a priority list")
    unknown = [name for name in names if _resolve(name) is None]
    if unknown:
        from woof.source_adapters import source_adapters

        raise SourceSelectRefused(
            f"source priority names unregistered sources {unknown}; "
            "registered: "
            + ", ".join(a.source_id for a in source_adapters()))
    seen: dict[str, str] = {}
    for name in names:
        canonical = _resolve(name).source_id
        if canonical in seen:
            raise SourceSelectRefused(
                f"source priority lists {canonical} twice ({seen[canonical]!r}"
                f" and {name!r})")
        seen[canonical] = name
    return names, "caller"


def select_source(region, cycle, hours, p_top_pa=None, priority=None, *,
                  now: datetime | None = None, spinup_hours: float = 0.0,
                  margin_km: float = 0.0,
                  accept_unknown_archive: bool = False,
                  refuse: Callable[[str], str | None] | None = None
                  ) -> SourceChoice:
    """The first source in ``priority`` that can force ``region`` at ``cycle``.

    ``region``: see :func:`region_bbox`.  ``cycle``: the source cycle the
    run starts from (``datetime`` or ``YYYY-MM-DDTHH``, UTC).  ``hours``:
    the run length after spin-up; ``spinup_hours`` is added to it for
    every horizon and publication question.  ``p_top_pa``: the run's
    model top (None skips the column-top check).  ``priority``: source
    ids or aliases, best first (default :data:`DEFAULT_PRIORITY`).
    ``now``: the clock every date question is asked against (default the
    wall clock; injected for reproducibility).  ``margin_km`` widens the
    region's bounding box before the coverage check (a planner's parent
    domain extends past its sites).  ``refuse(source_id)`` is the calling
    planner's own refusal for a candidate, or None.

    Raises :class:`SourceSelectRefused` (its ``choice`` holding every
    candidate's rejections) when none qualifies.  Never reaches a server.
    """

    from woof import fetch_routes

    start = parse_cycle(cycle)
    hours_f = _finite_hours(hours, "hours", positive=True)
    spin = _finite_hours(spinup_hours, "spinup_hours", positive=False)
    margin = float(margin_km)
    if not math.isfinite(margin) or margin < 0.0:
        raise SourceSelectRefused(f"margin_km {margin_km!r} must be >= 0")
    if p_top_pa is not None:
        if isinstance(p_top_pa, bool):
            raise SourceSelectRefused(f"p_top_pa {p_top_pa!r} is not a pressure")
        top = float(p_top_pa)
        if not math.isfinite(top) or top <= 0.0:
            raise SourceSelectRefused(
                f"p_top_pa {p_top_pa!r} must be a positive pressure in Pa")
        p_top_pa = top
    names, origin = _priority(priority)
    clock = _utc_naive(now)
    bbox = region_bbox(region)
    sampled = _widen(bbox, margin) if margin > 0.0 else bbox
    lat, lon = _samples(sampled)
    candidates = tuple(
        judge(name, rank, cycle=start, hours=hours_f, spinup_hours=spin,
              p_top_pa=p_top_pa, lat=lat, lon=lon, now=clock,
              accept_unknown_archive=bool(accept_unknown_archive),
              refuse=refuse)
        for rank, name in enumerate(names, start=1))
    chosen = next((c.source for c in candidates if c.eligible), None)
    try:
        table_sha = fetch_routes.packaged_route_table_sha256()
    except OSError:
        table_sha = None
    choice = SourceChoice(
        source=chosen, cycle=start, hours=hours_f, spinup_hours=spin,
        p_top_pa=p_top_pa, priority=tuple(c.source for c in candidates),
        priority_origin=origin,
        region={"bbox": [round(v, 6) for v in bbox], "margin_km": margin,
                "sampled_bbox": [round(v, 6) for v in sampled],
                "samples": int(lat.size)},
        now=clock, accept_unknown_archive=bool(accept_unknown_archive),
        candidates=candidates, route_table_sha256=table_sha,
        notes=tuple(note for c in candidates if c.source == chosen
                    for note in c.notes))
    if chosen is None:
        lines = "; ".join(
            f"{c.source}: " + " / ".join(f"{r.check}: {r.reason}"
                                         for r in c.rejections)
            for c in candidates)
        raise SourceSelectRefused(
            f"no source in {','.join(choice.priority)} can force this region "
            f"at {start:%Y-%m-%dT%H}Z for {choice.horizon_hours:g} h: "
            f"{lines}", choice)
    return choice


def select_newest(region, newest, hours, p_top_pa=None, priority=None, *,
                  now: datetime | None = None,
                  lookback_cycles: int = LOOKBACK_CYCLES,
                  step_hours: int = LOOKBACK_STEP_HOURS,
                  **kwargs) -> SourceChoice:
    """:func:`select_source` for a start the caller did not name.

    Asks every cycle from ``newest`` back ``lookback_cycles`` steps of
    ``step_hours`` and keeps the choice whose source ranks best in the
    priority list, the newest cycle among equals: a preferred source's
    previous cycle beats a lower-ranked source's newest one, as a named
    source's newest *landed* cycle would.  The receipt's notes name the
    cycles searched.  Refused (with the newest cycle's receipt) when no
    cycle has a qualifying source.
    """

    first = parse_cycle(newest)
    if int(lookback_cycles) < 1 or int(step_hours) < 1:
        raise SourceSelectRefused("lookback_cycles and step_hours must be "
                                  "at least 1")
    best: SourceChoice | None = None
    best_rank = None
    refusal: SourceSelectRefused | None = None
    for back in range(int(lookback_cycles)):
        cycle = first - timedelta(hours=back * int(step_hours))
        try:
            choice = select_source(region, cycle, hours, p_top_pa, priority,
                                   now=now, **kwargs)
        except SourceSelectRefused as error:
            if error.choice is None:
                raise
            refusal = refusal or error
            continue
        rank = choice.chosen().rank
        if best is None or rank < best_rank:
            best, best_rank = choice, rank
        if rank == 1:
            break
    if best is None:
        assert refusal is not None
        raise refusal
    searched = (f"default start: searched {lookback_cycles} cycles every "
                f"{step_hours} h back from {first:%Y-%m-%dT%H}Z and took "
                f"{best.cycle:%Y-%m-%dT%H}Z, where {best.source} (rank "
                f"{best_rank}) is the best-ranked source that qualifies")
    from dataclasses import replace

    return replace(best, notes=best.notes + (searched,))


def record_choice(choice: SourceChoice | None, outdir: str | Path,
                  notes: list[str]) -> dict | None:
    """A planner's bookkeeping for ``--source auto``.

    Writes the receipt to ``outdir/source-choice.json`` and appends the
    plan note when ``choice`` is given; removes a stale receipt when it
    is None.  Returns the receipt for ``Plan.source_choice``.
    """

    path = Path(outdir) / RECEIPT_FILE
    if choice is None:
        path.unlink(missing_ok=True)
        return None
    notes.append(choice.note())
    notes.extend(note for note in choice.notes
                 if note.startswith("default start:"))
    write_receipt(choice, path)
    return choice.to_json()


def write_receipt(choice: SourceChoice, path: str | Path) -> Path:
    """Write ``choice``'s receipt atomically (``.partial`` then rename)."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".partial")
    staging.write_text(json.dumps(choice.to_json(), indent=1) + "\n")
    staging.replace(path)
    return path


def choice_summary(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """The short form of a ``woof-energy.source-choice.v1`` receipt.

    What a planner's stdout record carries: the choice, the ranking and
    ``source -> ["check: reason", ...]`` for every rejected candidate.
    """

    return {"schema": receipt["schema"], "source": receipt["source"],
            "cycle": receipt["cycle"],
            "horizon_hours": receipt["horizon_hours"],
            "priority": list(receipt["priority"]),
            "rejected": {c["source"]: [f"{r['check']}: {r['reason']}"
                                       for r in c["rejections"]]
                         for c in receipt["candidates"]
                         if c["status"] == "rejected"}}


def is_auto(source: str | None) -> bool:
    """Whether ``source`` asks for the selector (``--source auto``)."""

    return source is not None and str(source).strip().lower() == "auto"


__all__ = [
    "CHECKS", "Candidate", "COVERAGE_MAX_PER_AXIS", "COVERAGE_STEP_DEG",
    "choice_summary",
    "DEFAULT_PRIORITY", "LOOKBACK_CYCLES", "LOOKBACK_STEP_HOURS",
    "PLANNER_SLACK_KM", "RECEIPT_FILE", "Rejection", "SCHEMA",
    "SourceChoice", "record_choice", "select_newest",
    "SourceSelectRefused", "is_auto", "judge", "parse_cycle",
    "region_bbox", "select_source", "write_receipt",
]
