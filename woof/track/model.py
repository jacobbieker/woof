"""A 3-D path as one time-ordered track, and what a planner reads off it.

:class:`Track3D` is the contract between path ingestion
(:mod:`woof.track.ingest`) and anything that sizes a simulation around a
moving object -- a balloon, a drone, an aircraft: the cleaned fixes, the
launch and landing found in them, and the quantities a domain planner
needs without re-deriving them -- duration, peak, the time-weighted
height profile, the lat/lon envelope and which regional analysis can
force it.

Nothing here knows a file format.  :func:`track_from_fixes` takes arrays
that are already quality-controlled and time-ordered; the reader owns the
parsing and the QC.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
from typing import Any, Mapping, Sequence

import numpy as np

from woof.explain import layered


TRACK_SCHEMA = "woof-track3d-v1"

#: Launch rule: a climb of at least this many metres above the launch
#: point, with horizontal drift.  Configurable per call; 300 m keeps a
#: payload carried up a hill or a lift test from reading as a launch.
DEFAULT_MIN_CLIMB_M = 300.0
#: The drift half of the rule.  Small on purpose: a low test climb rises a
#: few hundred metres and drifts about a kilometre, and it is still one.
MIN_DRIFT_M = 200.0
#: A fix this far above the ground it left is airborne.  Ground GPS
#: heights scatter by a few metres; a climbing object clears this within
#: seconds.
AIRBORNE_M = 50.0
#: Ground needs this many fixes before the first airborne one for the
#: launch to count as observed.  A single low first fix is as likely the
#: first message of an object already climbing.
MIN_GROUND_FIXES = 2
#: A peak at least this far above launch makes a ``high`` track; below it
#: the track is ``low``, or ``unknown`` when contact was lost while still
#: climbing.
HIGH_PEAK_M = 5000.0
#: No landing site is higher (MSL; an AGL ground is near 0 anyway).  A
#: track that ends at rest above this is floating out of contact.
MAX_GROUND_M = 5000.0
#: At rest: a vertical speed below this, measured over at least
#: REST_SPAN_S so that metres of jitter between fixes a second apart do
#: not read as motion ...
REST_RATE_MS = 1.0
REST_SPAN_S = 60.0
#: ... and, across a run of resting fixes, within this distance and this
#: height of the last one.  The distance is generous because a parachute
#: can drag a landed payload several hundred metres.
REST_RADIUS_M = 1000.0
REST_HEIGHT_M = 50.0
#: The first ground fix is the first of the resting run within this of
#: the run's median height; GPS heights on the ground scatter less.
GROUND_TOLERANCE_M = 10.0
#: A final fix after a silence counts as landed when the height fell, over
#: the silence, at less than this fraction of the speed it was falling at
#: just before: a payload still descending would be far lower by then.
STOPPED_FRACTION = 0.25
#: A measured climb or descent slower than this is not used to place
#: launch or landing inside a gap between messages.
MIN_PLACING_RATE_MS = 0.5
#: The height histogram's bin width and the step it is integrated on.
HISTOGRAM_BIN_M = 250.0
HISTOGRAM_STEP_S = 10.0
#: The margin the boundary-source decision demands around the track: the
#: outer domain is the track envelope plus this, and every column of it
#: must come from one analysis.
BOUNDARY_MARGIN_KM = 60.0
#: A conservative interior of the UKV grid, (south, west, north, east).
#: The real grid reaches well past this on every side; a track inside it
#: (plus margin) is certainly covered.
UKV_INTERIOR = (48.0, -12.0, 60.0, 5.0)

EARTH_RADIUS_KM = 6371.0
BOUNDARY_SOURCES = ("hrrr", "ukv", "era5")
HEIGHT_DATUMS = ("msl", "agl")


@dataclass(frozen=True)
class TrackEvent:
    """Launch or landing: when, where and at what ground height.

    ``window_s`` is the gap between the two messages that bracket the
    event (last ground and first airborne for a launch; last airborne and
    first ground for a landing).  ``time`` sits inside that window,
    placed by the measured climb or descent rate.  An event that was not
    observed is the first (launch) or last (landing) contact, and its
    window is 0.
    """

    time: datetime
    lat: float
    lon: float
    height: float
    window_s: float

    def to_dict(self) -> dict[str, Any]:
        return {"time": _iso(self.time), "lat": self.lat, "lon": self.lon,
                "height": self.height, "window_s": self.window_s}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrackEvent":
        return cls(time=_parse_iso(data["time"]), lat=float(data["lat"]),
                   lon=float(data["lon"]), height=float(data["height"]),
                   window_s=float(data["window_s"]))


@dataclass(frozen=True)
class Track3D:
    """One track: cleaned fixes plus everything a planner reads off them.

    ``times``/``lat``/``lon``/``height`` are the whole quality-controlled
    record; :attr:`launch` and :attr:`landing` bound the moving part
    inside it.  Heights are metres in ``height_datum`` (``msl`` or
    ``agl``), as the input declared them.

    ``height_hist_seconds[k]`` is the time, between launch and landing,
    spent in ``[k * height_hist_bin_m, (k + 1) * height_hist_bin_m)``.
    ``envelope`` is ``(south, west, north, east)`` over every fix;
    ``extent_ew_km`` and ``extent_ns_km`` are its sides measured through
    its centre.  ``boundary_source`` is a suggestion -- ``hrrr``, ``ukv``
    or ``era5`` -- from where the track is (:func:`boundary_source`).
    ``qc`` counts what the reader dropped or repaired, and why.
    """

    name: str
    times: tuple[datetime, ...]
    lat: tuple[float, ...]
    lon: tuple[float, ...]
    height: tuple[float, ...]
    height_datum: str
    launch: TrackEvent
    launch_observed: bool
    landing: TrackEvent
    landing_observed: bool
    peak_class: str
    duration_s: float
    max_height: float
    max_height_time: datetime
    min_climb_m: float
    height_hist_bin_m: float
    height_hist_seconds: tuple[float, ...]
    envelope: tuple[float, float, float, float]
    extent_ew_km: float
    extent_ns_km: float
    boundary_source: str
    qc: dict[str, int] = field(default_factory=dict)
    source_path: str | None = None

    @property
    def duration_h(self) -> float:
        return self.duration_s / 3600.0

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready dict; :meth:`from_dict` inverts it exactly."""

        return {
            "schema": TRACK_SCHEMA,
            "name": self.name,
            "height_datum": self.height_datum,
            "launch": self.launch.to_dict(),
            "launch_observed": self.launch_observed,
            "landing": self.landing.to_dict(),
            "landing_observed": self.landing_observed,
            "peak_class": self.peak_class,
            "duration_s": self.duration_s,
            "max_height": self.max_height,
            "max_height_time": _iso(self.max_height_time),
            "min_climb_m": self.min_climb_m,
            "height_hist_bin_m": self.height_hist_bin_m,
            "height_hist_seconds": list(self.height_hist_seconds),
            "envelope": list(self.envelope),
            "extent_ew_km": self.extent_ew_km,
            "extent_ns_km": self.extent_ns_km,
            "boundary_source": self.boundary_source,
            "qc": dict(self.qc),
            "source_path": self.source_path,
            "times": [_iso(t) for t in self.times],
            "lat": list(self.lat),
            "lon": list(self.lon),
            "height": list(self.height),
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Track3D":
        schema = data.get("schema")
        if schema != TRACK_SCHEMA:
            raise ValueError(layered(
                f"this track document declares schema {schema!r}, not "
                f"{TRACK_SCHEMA!r}.\n"
                "  remedy: regenerate it with `woof track inspect FILE "
                "--json OUT`.",
                "The document is read field by field; a different schema "
                "may name the same fields with different meanings, so it "
                "is refused rather than guessed at."))
        source = data["boundary_source"]
        if source not in BOUNDARY_SOURCES:
            raise ValueError(f"track boundary_source {source!r} is not one of "
                             f"{', '.join(BOUNDARY_SOURCES)}")
        return cls(
            name=data["name"],
            times=tuple(_parse_iso(t) for t in data["times"]),
            lat=tuple(float(v) for v in data["lat"]),
            lon=tuple(float(v) for v in data["lon"]),
            height=tuple(float(v) for v in data["height"]),
            height_datum=data["height_datum"],
            launch=TrackEvent.from_dict(data["launch"]),
            launch_observed=bool(data["launch_observed"]),
            landing=TrackEvent.from_dict(data["landing"]),
            landing_observed=bool(data["landing_observed"]),
            peak_class=data["peak_class"],
            duration_s=float(data["duration_s"]),
            max_height=float(data["max_height"]),
            max_height_time=_parse_iso(data["max_height_time"]),
            min_climb_m=float(data["min_climb_m"]),
            height_hist_bin_m=float(data["height_hist_bin_m"]),
            height_hist_seconds=tuple(float(v) for v in data["height_hist_seconds"]),
            envelope=tuple(float(v) for v in data["envelope"]),
            extent_ew_km=float(data["extent_ew_km"]),
            extent_ns_km=float(data["extent_ns_km"]),
            boundary_source=source,
            qc={str(k): int(v) for k, v in data.get("qc", {}).items()},
            source_path=data.get("source_path"),
        )

    @classmethod
    def from_json(cls, text: str) -> "Track3D":
        return cls.from_dict(json.loads(text))


def track_from_fixes(name: str, times: Sequence[datetime],
                     lat: Sequence[float], lon: Sequence[float],
                     height: Sequence[float], *, height_datum: str = "msl",
                     min_climb_m: float = DEFAULT_MIN_CLIMB_M,
                     margin_km: float = BOUNDARY_MARGIN_KM,
                     qc: Mapping[str, int] | None = None,
                     source_path: str | None = None) -> Track3D:
    """Detect launch and landing in cleaned fixes and derive the rest.

    The fixes must be quality-controlled and strictly time-ordered (UTC,
    timezone-aware).  Vertical speeds are measured between fixes, never
    taken from a transmitter's own report: some firmware sends a constant
    zero and some a heavily lagged average, either of which would put a
    descending payload on the ground.

    Raises ``ValueError`` when the fixes hold no climb of ``min_climb_m``
    with horizontal drift.
    """

    n = len(times)
    if n < 2 or not (len(lat) == len(lon) == len(height) == n):
        raise ValueError(layered(
            f"track {name} has {n} usable fixes; a track needs at least "
            "two, with one lat, lon and height per time.\n"
            "  remedy: check the file holds position rows and that the "
            "column options name its time, lat, lon and height columns.",
            "Launch and landing are found between fixes, so a single fix "
            "cannot bound a track."))
    if height_datum not in HEIGHT_DATUMS:
        raise ValueError(f"height datum {height_datum!r} is not one of "
                         f"{', '.join(HEIGHT_DATUMS)}")
    t = np.array([(x - times[0]).total_seconds() for x in times])
    if np.any(np.diff(t) <= 0):
        raise ValueError(f"track {name}: fixes are not strictly time-ordered; "
                         "sort and de-duplicate them first")
    la = np.asarray(lat, dtype=np.float64)
    lo = np.asarray(lon, dtype=np.float64)
    h = np.asarray(height, dtype=np.float64)
    if lo.max() - lo.min() > 180.0:
        raise ValueError(layered(
            f"track {name} spans {lo.min():.1f} to {lo.max():.1f} degrees "
            "of longitude.\n"
            "  remedy: a track crossing the antimeridian is not supported; "
            "split it there.",
            "The envelope is a plain longitude min/max, which across the "
            "antimeridian would describe the rest of the globe."))
    rest = _at_rest(t, h)

    first_air, launch_t, launch_h, launch_window = _launch(name, t, h, min_climb_m)
    launch_observed = first_air >= MIN_GROUND_FIXES
    if not launch_observed:
        # Climbing from (nearly) the first fix: the launch happened before
        # the record, and the first fix is the best bound on it there is.
        first_air, launch_t, launch_h, launch_window = 0, 0.0, float(h[0]), 0.0
    launch_index = max(first_air - 1, 0)
    drift_km = haversine_km(la[launch_index], lo[launch_index],
                            la[first_air:], lo[first_air:]).max()
    if drift_km * 1000.0 < MIN_DRIFT_M:
        raise ValueError(layered(
            f"track {name} climbs {h.max() - launch_h:.0f} m but never "
            f"moves more than {drift_km * 1000.0:.0f} m from where it "
            "left the ground.\n"
            "  remedy: if this is a real flight, check its positions; a "
            "bench or tethered test has no track to follow.",
            f"A launch is a climb of at least {min_climb_m:.0f} m above "
            "the launch point WITH horizontal drift; a climb in place "
            f"({MIN_DRIFT_M:.0f} m or less) is a bench test or a tether."))

    peak = first_air + int(np.argmax(h[first_air:]))
    landing = _landing(t, la, lo, h, rest, peak, launch_h)
    landing_observed = landing is not None
    if landing is None:
        land_i, land_t, land_h, land_window = n - 1, t[-1], h[-1], 0.0
    else:
        land_i, land_t, land_h, land_window = landing

    if h[peak] - launch_h >= HIGH_PEAK_M:
        peak_class = "high"
    elif not landing_observed and peak == n - 1:
        peak_class = "unknown"
    else:
        peak_class = "low"

    t0 = times[0]
    envelope = (float(la.min()), float(lo.min()), float(la.max()), float(lo.max()))
    ew_km, ns_km = envelope_extent_km(envelope)
    return Track3D(
        name=name,
        times=tuple(times),
        lat=tuple(float(v) for v in la),
        lon=tuple(float(v) for v in lo),
        height=tuple(float(v) for v in h),
        height_datum=height_datum,
        launch=TrackEvent(time=_at(t0, launch_t), lat=float(la[launch_index]),
                          lon=float(lo[launch_index]), height=launch_h,
                          window_s=launch_window),
        launch_observed=launch_observed,
        landing=TrackEvent(time=_at(t0, land_t), lat=float(la[land_i]),
                           lon=float(lo[land_i]), height=float(land_h),
                           window_s=land_window),
        landing_observed=landing_observed,
        peak_class=peak_class,
        duration_s=float(round(land_t - launch_t, 6)),
        max_height=float(h[peak]),
        max_height_time=times[peak],
        min_climb_m=float(min_climb_m),
        height_hist_bin_m=HISTOGRAM_BIN_M,
        height_hist_seconds=height_histogram(t, h, launch_t, land_t),
        envelope=envelope,
        extent_ew_km=ew_km,
        extent_ns_km=ns_km,
        boundary_source=boundary_source(la, lo, margin_km=margin_km),
        qc=dict(qc or {}),
        source_path=source_path,
    )


def _ground(h: np.ndarray) -> float:
    """The ground height the record opens on: the densest cluster among
    the first five fixes (ties to the lowest), so one low glitch cannot
    pull the reference under the real ground."""

    first = h[:5]
    support = (np.abs(first[:, None] - first[None, :]) <= AIRBORNE_M).sum(axis=1)
    best = min(range(first.size), key=lambda k: (-support[k], first[k]))
    return float(np.median(first[np.abs(first - first[best]) <= AIRBORNE_M]))


def _launch(name: str, t: np.ndarray, h: np.ndarray,
            min_climb_m: float) -> tuple[int, float, float, float]:
    """``(first airborne index, launch time s, ground height, window s)``."""

    ground = _ground(h)
    climbed = np.nonzero(h >= ground + min_climb_m)[0]
    if climbed.size == 0:
        raise ValueError(layered(
            f"track {name} never climbs {min_climb_m:.0f} m above where "
            f"it starts (peak {h.max() - ground:.0f} m above it).\n"
            "  remedy: check the height column (and --height-datum), or "
            "lower --min-climb if a smaller climb is a launch for this "
            "kind of track.",
            f"A launch is a climb of at least {min_climb_m:.0f} m above "
            "the launch point with horizontal drift.  Anything less is a "
            "bench test, a ride in a vehicle or a record that starts "
            "after launch, and none of those has a launch to find."))
    first_air = int(climbed[0])
    while first_air > 0 and h[first_air - 1] > ground + AIRBORNE_M:
        first_air -= 1
    if first_air == 0:
        return 0, float(t[0]), float(h[0]), 0.0
    last_ground = first_air - 1
    launch_h = float(np.median(h[max(last_ground - 2, 0):last_ground + 1]))
    climb = _rate(t, h, first_air, first_air + 1)
    t_ground, t_air = float(t[last_ground]), float(t[first_air])
    if climb is None:
        estimate = 0.5 * (t_ground + t_air)
    else:
        estimate = t_air - (h[first_air] - launch_h) / climb
    return (first_air, float(np.clip(estimate, t_ground, t_air)), launch_h,
            t_air - t_ground)


def _landing(t: np.ndarray, la: np.ndarray, lo: np.ndarray, h: np.ndarray,
             rest: np.ndarray, peak: int,
             launch_h: float) -> tuple[int, float, float, float] | None:
    """``(first ground index, landing time s, ground height, window s)``,
    or ``None`` when the record ends before the object is on the ground."""

    last = h.size - 1
    # Down in the lower half of the climb: the ground may be higher than
    # the launch site, but an object at rest near its peak is floating.
    if (last <= peak or h[last] > MAX_GROUND_M
            or h[last] > launch_h + 0.5 * (h[peak] - launch_h)):
        return None
    if not (rest[last] or _stopped_in_gap(t, h, last, peak)):
        return None
    first_ground = last
    while (first_ground - 1 > peak and rest[first_ground - 1]
           and abs(h[first_ground - 1] - h[last]) <= REST_HEIGHT_M
           and haversine_km(la[last], lo[last], la[first_ground - 1],
                            lo[first_ground - 1]) * 1000.0 <= REST_RADIUS_M):
        first_ground -= 1
    land_h = float(np.median(h[first_ground:]))
    # The span-wide rest test admits the last few metres of a slow
    # descent; the ground starts where the height has come down to it.
    while first_ground < last and h[first_ground] > land_h + GROUND_TOLERANCE_M:
        first_ground += 1
    last_air = first_ground - 1
    descent = _rate(t, h, last_air, last_air - 1)
    t_air, t_ground = float(t[last_air]), float(t[first_ground])
    if descent is None:
        estimate = 0.5 * (t_air + t_ground)
    else:
        estimate = t_air + (h[last_air] - land_h) / descent
    return (first_ground, float(np.clip(estimate, t_air, t_ground)), land_h,
            t_ground - t_air)


def _stopped_in_gap(t: np.ndarray, h: np.ndarray, last: int, peak: int) -> bool:
    """Whether a final fix after a long silence shows a descent that
    stopped: over the gap the height fell far slower than it was falling
    just before, so the object reached this height early and stayed."""

    before, prior = last - 1, last - 2
    if prior <= peak or t[last] - t[before] < REST_SPAN_S:
        return False
    falling = (h[prior] - h[before]) / (t[before] - t[prior])
    over_gap = (h[before] - h[last]) / (t[last] - t[before])
    return falling >= MIN_PLACING_RATE_MS and over_gap < STOPPED_FRACTION * falling


def _rate(t: np.ndarray, h: np.ndarray, i: int, j: int) -> float | None:
    """The climb (``j`` after ``i``) or descent (``j`` before ``i``) speed
    at fix ``i``, m/s and positive, measured against fix ``j``; ``None``
    when there is no such fix or the speed is too slow to place by."""

    if not 0 <= j < h.size:
        return None
    speed = (h[j] - h[i]) / abs(t[j] - t[i])
    return float(speed) if speed >= MIN_PLACING_RATE_MS else None


def _at_rest(t: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Per fix: vertical speed below :data:`REST_RATE_MS`, measured
    forward to the first fix at least :data:`REST_SPAN_S` later (back to
    the last one that far earlier, near the end of the record).  Forward,
    so the first fix on the ground is at rest although the descent ended
    at it."""

    n = h.size
    if n < 2:
        return np.ones(n, dtype=bool)
    ahead = np.minimum(np.searchsorted(t, t + REST_SPAN_S), n - 1)
    behind = np.maximum(np.searchsorted(t, t - REST_SPAN_S, side="right") - 1, 0)
    # Within a span of the record's end the forward span is cut short;
    # there, the longer of the two spans measures.  (n >= 2 keeps the
    # last fix's ``behind`` off itself.)
    other = np.where((t[ahead] - t >= REST_SPAN_S)
                     | (t[ahead] - t >= t - t[behind]), ahead, behind)
    speed = np.abs(h[other] - h) / np.abs(t[other] - t)
    return speed < REST_RATE_MS


def height_histogram(t: np.ndarray, h: np.ndarray, start: float,
                     end: float, *, bin_m: float = HISTOGRAM_BIN_M,
                     step_s: float = HISTOGRAM_STEP_S) -> tuple[float, ...]:
    """Seconds spent per ``bin_m`` height bin between ``start`` and
    ``end`` (seconds on ``t``'s axis), integrating the linearly
    interpolated track on a ``step_s`` grid.  Bin 0 starts at the datum;
    heights below it count there."""

    if end <= start:
        return ()
    samples = np.arange(start, end, step_s)
    weights = np.full(samples.size, step_s)
    weights[-1] = end - samples[-1]
    level = np.interp(samples, t, h)
    index = np.floor(np.maximum(level, 0.0) / bin_m).astype(np.int64)
    seconds = np.bincount(index, weights=weights)
    return tuple(round(float(s), 3) for s in seconds)


def haversine_km(lat0, lon0, lat1, lon1) -> np.ndarray:
    """Great-circle distance, km, broadcasting over its arguments."""

    p0, l0, p1, l1 = (np.radians(np.asarray(v, dtype=np.float64))
                      for v in (lat0, lon0, lat1, lon1))
    a = (np.sin((p1 - p0) / 2.0) ** 2
         + np.cos(p0) * np.cos(p1) * np.sin((l1 - l0) / 2.0) ** 2)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def envelope_extent_km(envelope: Sequence[float]) -> tuple[float, float]:
    """``(east-west, north-south)`` sides of a ``(south, west, north,
    east)`` box, km, each measured through the box's centre."""

    south, west, north, east = envelope
    mid_lat, mid_lon = 0.5 * (south + north), 0.5 * (west + east)
    ew = float(haversine_km(mid_lat, west, mid_lat, east))
    ns = float(haversine_km(south, mid_lon, north, mid_lon))
    return round(ew, 3), round(ns, 3)


def boundary_source(lat: Sequence[float], lon: Sequence[float], *,
                    margin_km: float = BOUNDARY_MARGIN_KM) -> str:
    """Which regional analysis can force a domain around this track.

    ``hrrr`` when the track and the perimeter of its lat/lon envelope,
    inset by ``margin_km``, sit inside the native CONUS HRRR mass grid;
    ``ukv`` when the envelope grown by ``margin_km`` sits inside
    :data:`UKV_INTERIOR`; ``era5`` (global) otherwise.
    """

    la = np.asarray(lat, dtype=np.float64)
    lo = np.asarray(lon, dtype=np.float64)
    south, west, north, east = la.min(), lo.min(), la.max(), lo.max()
    # The whole perimeter, not the corners: in the Lambert projection a
    # parallel bows away from the chord between its ends.
    edge = np.linspace(0.0, 1.0, 65)
    ring_lat = np.concatenate((np.full(edge.size, south), np.full(edge.size, north),
                               south + (north - south) * edge,
                               south + (north - south) * edge))
    ring_lon = np.concatenate((west + (east - west) * edge, west + (east - west) * edge,
                               np.full(edge.size, west), np.full(edge.size, east)))
    if _inside_hrrr(np.concatenate((la, ring_lat)),
                    np.concatenate((lo, ring_lon)), margin_km):
        return "hrrr"
    pad_lat = margin_km / 111.195
    pad_lon = margin_km / (111.195 * max(np.cos(np.radians(
        min(max(abs(south), abs(north)) + pad_lat, 89.0))), 1e-6))
    u_south, u_west, u_north, u_east = UKV_INTERIOR
    if (south - pad_lat >= u_south and north + pad_lat <= u_north
            and west - pad_lon >= u_west and east + pad_lon <= u_east):
        return "ukv"
    return "era5"


def _inside_hrrr(lat: np.ndarray, lon: np.ndarray, margin_km: float) -> bool:
    """Every point at least ``margin_km`` inside the HRRR mass grid, as
    the source registry declares it (the one coverage definition)."""

    # Local import: the registry is only needed once a track is placed.
    from woof.source_adapters import source_coverage_window

    window = source_coverage_window("hrrr")
    x, y = window.grid().latlon_to_ij(lat, lon)
    inset = margin_km * 1000.0 / window.dx_m
    return bool(np.all((x >= 1.0 + inset) & (x <= window.nx - inset)
                       & (y >= 1.0 + inset) & (y <= window.ny - inset)))


def _at(origin: datetime, seconds: float) -> datetime:
    return origin + timedelta(seconds=float(seconds))


def _iso(moment: datetime) -> str:
    # Microseconds: datetime's own resolution, so a document read back
    # holds the very instants it was written from.
    text = moment.astimezone(timezone.utc).isoformat(timespec="microseconds")
    return text.replace("+00:00", "Z")


def _parse_iso(text: str) -> datetime:
    moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment
