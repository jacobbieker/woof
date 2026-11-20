"""WIS2 in-situ observations from the ``bkr/obs/wis2.icechunk`` archive (opt-in).

``woof obs wis2 fetch`` reads one time window and one latitude/longitude box
from Source Cooperative's combined WIS2 store
(``us-west-2.opendata.source.coop``, prefix ``bkr/obs/wis2.icechunk``),
anonymously and read-only, and writes:

* a ``gpuwm-obs.table.v2`` table (:mod:`woof.obs.neutral_table`) that the
  ensemble filter's point operators (:func:`woof.da.obs_point.point_batches`)
  read, with each report's Global Cache arrival as its ``received_time``;
* optionally OBSPROC's ``obs_gts`` ASCII, the 3D-Var interface agreed with
  the da.var work (docs/dev/WRFDA-3DVAR-PLAN.md section 12.7), which
  :func:`woof.da.var.obs_wrfda.read_obs_ascii` and WRFDA itself read.

This is the archive, not the live broker: the ``wis2`` stream of
:mod:`woof.obs.streams` subscribes to the broker through ``rw_wis2``.  The
store layout, its QC flags and the family-to-type mapping are in section 12
of docs/dev/WRFDA-3DVAR-PLAN.md.  Rows whose label ``obs_point`` has no
operator for yet (ocean platforms, aircraft) are written with their true
label and refused by name there.  ``aircraft_level`` is used only with the
opt-in ``assimilate_aircraft`` request field (``woof local-da
--assimilate-aircraft``), which applies to every aircraft source alike.
Families read here:

==============  ============================  ======  ==========================
family          groups                        WRFDA   neutral measurement
==============  ============================  ======  ==========================
``land``        land/reports, land/synop_tac  FM-12   screen_temperature_2m, ...
``ocean``       ocean/surface, ocean/ship_tac FM-13/18 platform_* (no operator)
``sonde``       upper_air/soundings           FM-35   sonde_level
``aircraft``    aircraft                      FM-42   aircraft_level (opt-in)
``profiler``    upper_air/wind_profiler       FM-132  (obs_gts only: no pressure)
``amv``         satellite/amv                 FM-88   amv_assigned_pressure
``scat``        satellite/scatterometer       FM-281  scatterometer_wind_10m
``ro``          satellite/gnss_ro             FM-116  ro_refractivity_tangent_point
==============  ============================  ======  ==========================

``amv``, ``scat`` and ``ro`` are read only when named.

Quality control, in order (each refusal is counted by name):

1. The store's own flags: a row or level is refused when any bit of
   ``quality_flag`` / ``level_quality_flag`` other than ``time_too_old``
   (256) is set; a value is refused when its ``<variable>_quality`` is
   non-zero.  The store never alters a value, so these are its only verdict.
2. Gross bounds on every family (temperature and dewpoint 150-350 K,
   pressure up to 1100 hPa, sea-level pressure 850-1100 hPa, station
   elevation -500 to 9000 m; a report with neither a pressure nor a height
   is dropped), sensor heights (2 m screen
   temperature 1-3 m, 10 m wind 8-12 m, as the station referees), wind
   speed 0-120 m/s, a direction of 0 with a non-zero speed.
3. Soundings: the TEMP parts of one launch are merged.  A part whose
   levels sit at ten times their pressure is detected and refused
   (:func:`pressure_scale_fault`): a part that by its own label holds only
   levels above 100 hPa (Part C or D) but has levels below it, any part
   with a level above 1100 hPa, or a part
   whose temperatures disagree with the launch's other parts unless every
   pressure is divided by ten.  The Global Cache archive holds such parts
   (section 12.6); the check keys on the data, not on a centre.
4. Duplicates: corrections replace originals (highest
   ``update_sequence_number``; a flagged correction does not revive the
   original), a text-only report repeating a BUFR report
   of the same station and time is dropped, aircraft by ``report_id``.

The background check is the analysis's (``obs_point``'s or WRFDA's
``check_max_iv``), not this module's.

Errors are WRFDA's: OBSPROC's ``obserr.txt`` through
:mod:`woof.obs.obserr_wrfda`, so the obs_gts file carries what OBSPROC
would have written and the neutral table the same numbers.

``icechunk`` and ``zarr`` are imported only when the store is opened; the
functions below also take any mapping of arrays laid out like the store
(the tests build one in memory), so nothing is fetched in a test.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BUCKET = "us-west-2.opendata.source.coop"
PREFIX = "bkr/obs/wis2.icechunk"
REGION = "us-west-2"
SOURCE = "wis2"
FAMILIES = ("land", "ocean", "sonde", "aircraft", "profiler")
# Satellite families (opt-in by name; not in the default list): the point winds, which
# --thin-km thins, and GNSS-RO refractivity profiles.
SATELLITE_FAMILIES = ("amv", "scat", "ro")
THINNED_FAMILIES = ("amv", "scat")
# quality_flag bits (the store's flag_meanings); every bit but time_too_old refuses.
QUALITY_BITS = ("outside_physical_range", "outside_plausible_range", "internally_inconsistent",
                "vertically_inconsistent", "position_invalid", "position_suspect", "time_invalid",
                "time_in_future", "time_too_old", "duplicate", "provider_flagged")
IGNORED_BITS = 256
SPEED_BOUNDS = (0.0, 120.0)
T2_SENSOR_M = (1.0, 3.0)
WIND_SENSOR_M = (8.0, 12.0)
# BUFR 0 08 042 (extended vertical sounding significance) bits.
VSS_SURFACE = 131072
VSS_STANDARD = 65536
# A sounding filed for hour H is launched up to about 75 min before it; two
# reports of one station launched this close together are one ascent.
SONDE_LAUNCH_LEAD = np.timedelta64(150, "m")
SONDE_SAME_ASCENT = np.timedelta64(90, "m")
# Pressure-scale detection: parts disagreeing by more than this (median, K)
# are inconsistent; dividing the pressures by ten must bring them within the
# second figure for the fault to be named a pressure-scale fault.
PART_DISAGREE_K = 10.0
PART_RESCALED_AGREE_K = 3.0
PART_MIN_OVERLAP = 3
PRESSURE_MAX_PA = 110000.0  # no station pressure exceeds 1100 hPa
# Gross bounds on every family before either writer (OBSPROC's range checks, loosely).
T_BOUNDS = (150.0, 350.0)
SLP_BOUNDS = (85000.0, 110000.0)
ELEV_BOUNDS = (-500.0, 9000.0)
HEIGHT_BOUNDS = (-500.0, 60000.0)
L_OVER_RV = 5418.12  # OBSPROC constants.inc
MISSING = -888888.0
MISSING_QC = -88


class Wis2Error(RuntimeError):
    """The store cannot be read for the request."""


# --------------------------------------------------------------------------- store access
def open_store(*, snapshot: str | None = None):
    """The archive's root group, anonymous and read-only (``icechunk`` and ``zarr`` needed)."""
    try:
        import icechunk
        import zarr
    except ImportError:
        raise Wis2Error("the WIS2 archive reader needs icechunk and zarr "
                        "(`pip install icechunk zarr`); nothing else in woof does") from None
    storage = icechunk.s3_storage(bucket=BUCKET, prefix=PREFIX, region=REGION, anonymous=True)
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session(snapshot_id=snapshot) if snapshot else repo.readonly_session("main")
    return zarr.open_group(session.store, mode="r"), session.snapshot_id


def _group(root, path):
    try:
        return root[path]
    except KeyError:
        raise Wis2Error(f"the store has no group {path!r}") from None


def _fill(attrs):
    value = attrs.get("_FillValue") if hasattr(attrs, "get") else None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _attrs(array):
    return getattr(array, "attrs", {}) or {}


def _read(array, index):
    """``array`` at the sorted row ``index``: one block when dense, else the rows alone."""
    index = np.asarray(index, np.int64)
    if index.size == 0:
        return np.zeros((0,) + tuple(array.shape[1:]))
    lo, hi = int(index[0]), int(index[-1]) + 1
    if hi - lo <= 4 * index.size or not hasattr(array, "oindex"):
        return np.asarray(array[lo:hi])[index - lo]
    return np.asarray(array.oindex[index])


def _numbers(array, index):
    """Float values with the array's fill value as NaN."""
    raw = _read(array, index)
    fill = _fill(_attrs(array))
    out = np.asarray(raw, dtype=float)
    if fill is not None:
        out = np.where(np.asarray(raw) == fill, np.nan, out)
    return out


def _texts(array, index):
    return np.asarray([str(v).strip() for v in _read(array, index).tolist()], dtype=object)


def _times(array, index=None):
    """datetime64[s] from the store's integer times (ns, ms or s since 1970); NaT for fill."""
    raw = np.asarray(array[:] if index is None else _read(array, index))
    units = str(_attrs(array).get("units", "nanoseconds since 1970-01-01"))
    raw = raw.astype(np.int64)
    bad = raw == np.iinfo(np.int64).min
    fill = _fill(_attrs(array))
    if fill is not None:
        bad |= raw == int(fill)
    if units.startswith("nanoseconds"):
        out = (raw // 1_000_000_000).astype("datetime64[s]")
    elif units.startswith("milliseconds"):
        out = (raw // 1_000).astype("datetime64[s]")
    elif units.startswith("seconds"):
        out = raw.astype("datetime64[s]")
    else:
        raise Wis2Error(f"unknown time units {units!r}")
    out[bad] = np.datetime64("NaT", "s")
    return out


def _quality_ok(group, name, index, counts, label):
    """True where ``name`` (a quality_flag) has no refusing bit; counts each bit."""
    if name not in group:
        return np.ones(len(index), bool)
    flags = _numbers(group[name], index)
    flags = np.where(np.isfinite(flags), flags, 0).astype(np.int64)
    for bit, meaning in enumerate(QUALITY_BITS):
        hit = (flags >> bit) & 1
        if (1 << bit) != IGNORED_BITS and hit.any():
            counts[f"{label}_{meaning}"] += int(hit.sum())
    return (flags & ~IGNORED_BITS) == 0


def _value(group, name, index, counts, label):
    """Float values of ``name`` with NaN where missing or where ``<name>_quality`` is set."""
    if name not in group:
        return np.full(len(index), np.nan)
    values = _numbers(group[name], index)
    quality = f"{name}_quality"
    if quality in group:
        q = _numbers(group[quality], index)
        bad = np.isfinite(q) & (q != 0) & np.isfinite(values)
        if bad.any():
            counts[f"{label}_{name}_quality"] += int(bad.sum())
            values = np.where(bad, np.nan, values)
    return values


def in_box(lat, lon, box):
    """Inside (south, north, west, east); longitudes taken in [west, west + 360)."""
    south, north, west, east = box
    lat = np.asarray(lat, float)
    lon = ((np.asarray(lon, float) - west) % 360.0) + west
    return np.isfinite(lat) & (lat >= south) & (lat <= north) & (lon >= west) & (lon <= east)


def _window(start, end):
    return (np.datetime64(_naive(start), "s"), np.datetime64(_naive(end), "s"))


def _naive(when):
    if isinstance(when, str):
        when = dt.datetime.fromisoformat(when.replace("Z", "+00:00"))
    if when.tzinfo is not None:
        when = when.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return when


# --------------------------------------------------------------------------- reports
@dataclass
class Report:
    """One report in OBSPROC's shape: header plus levels (one level for a surface report).

    Missing values are NaN; ``p`` in Pa, ``z`` in m above sea level, ``t``
    and ``td`` in K, ``rh`` in %, ``spd`` in m/s, ``wdir`` in degrees from.
    ``level_time``, ``level_lat`` and ``level_lon`` place each level (a
    sonde's launch plus elapsed time and drift).
    """

    family: str
    fm: int
    station: str
    name: str
    lat: float
    lon: float
    elev: float
    time: np.datetime64
    received: np.datetime64
    p: np.ndarray
    z: np.ndarray
    t: np.ndarray
    td: np.ndarray
    rh: np.ndarray
    spd: np.ndarray
    wdir: np.ndarray
    slp: float = float("nan")
    level_time: np.ndarray | None = None
    level_lat: np.ndarray | None = None
    level_lon: np.ndarray | None = None
    group: str = ""
    surface: bool = True
    extras: dict = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.p)
        for name in ("z", "t", "td", "rh", "spd", "wdir"):
            setattr(self, name, np.asarray(getattr(self, name), float).reshape(n))
        self.p = np.asarray(self.p, float)
        if self.level_time is None:
            self.level_time = np.full(n, self.time)
        if self.level_lat is None:
            self.level_lat = np.full(n, self.lat)
        if self.level_lon is None:
            self.level_lon = np.full(n, self.lon)


def _wind_ok(spd, wdir, counts, label):
    good = np.isfinite(spd) & (spd >= SPEED_BOUNDS[0]) & (spd <= SPEED_BOUNDS[1]) \
        & np.isfinite(wdir) & (wdir >= 0) & (wdir <= 360) & ~((spd > 0) & (wdir == 0))
    bad = (np.isfinite(spd) | np.isfinite(wdir)) & ~good
    if bad.any():
        counts[f"{label}_wind_invalid"] += int(bad.sum())
    return good


def rh_from_dewpoint(t, td):
    """OBSPROC's ``diagnostics_moist``: 100 exp(L/Rv (1/T - 1/Td)), clipped to (1e-6, 100]."""
    t, td = np.asarray(t, float), np.asarray(td, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        rh = 100.0 * np.exp(L_OVER_RV * (1.0 / t - 1.0 / td))
    return np.clip(rh, 1.0e-6, 100.0)


def _moisture(t, td, rh):
    """(td, rh): rh from t and td where it is missing (OBSPROC); td above t refused."""
    td = np.where(np.isfinite(td) & np.isfinite(t) & (td > t + 0.05), np.nan, td)
    rh = np.where(np.isfinite(rh), np.clip(rh, 1.0e-6, 100.0), np.nan)
    derived = ~np.isfinite(rh) & np.isfinite(t) & np.isfinite(td)
    rh = np.where(derived, rh_from_dewpoint(t, td), rh)
    return td, rh


def _select(group, start, end, box, *, time_name="time", lead=np.timedelta64(0, "s")):
    lo, hi = _window(start, end)
    time = _times(group[time_name])
    lat = np.asarray(group["latitude"][:], float)
    lon = np.asarray(group["longitude"][:], float)
    keep = (time >= lo - lead) & (time <= hi) & in_box(lat, lon, box)
    return np.nonzero(keep)[0], time


def _received(group, index):
    for name in ("first_arrival_time", "arrival_time", "arrived"):
        if name in group:
            return _times(group[name], index)
    return np.full(len(index), np.datetime64("NaT"), "datetime64[s]")


# ----------------------------------------------------------------- land: SYNOP BUFR and TAC
def land_reports(root, start, end, box, counts: Counter) -> list[Report]:
    out, seen = [], set()
    for path, label in (("land/reports", "land"), ("land/synop_tac", "synop_tac")):
        if path not in root:
            continue
        group = root[path]
        index, time = _select(group, start, end, box)
        counts[f"{label}_in_window"] += len(index)
        if index.size == 0:
            continue
        ok = _quality_ok(group, "quality_flag", index, counts, label)
        sid = _texts(group["station_id"], index)
        when = time[index]
        seq = _numbers(group["update_sequence_number"], index) if "update_sequence_number" in group \
            else np.zeros(len(index))
        best = {}
        for k in np.lexsort((np.nan_to_num(seq, nan=-1.0), when.astype(np.int64), sid.astype(str))):
            best[(sid[k], when[k])] = k  # the highest sequence number of a (station, time) wins
        counts[f"{label}_superseded_correction"] += len(index) - len(best)
        # The latest correction stands or falls alone: a flagged correction does not revive the original.
        keep = np.array(sorted(k for k in best.values() if ok[k]), np.int64)
        if keep.size == 0:
            continue
        rows = index[keep]
        get = lambda name: _value(group, name, rows, counts, label)  # noqa: E731
        t2, td2, rh2 = get("2m_temperature"), get("2m_dewpoint_temperature"), get("2m_relative_humidity")
        spd, wdir = get("10m_wind_speed"), get("10m_wind_direction")
        ps, slp, elev = get("surface_pressure"), get("mean_sea_level_pressure"), get("station_elevation")
        th, wh = get("temperature_sensor_height"), get("wind_sensor_height")
        t_ok = ~np.isfinite(th) | ((th >= T2_SENSOR_M[0]) & (th <= T2_SENSOR_M[1]))
        w_ok = ~np.isfinite(wh) | ((wh >= WIND_SENSOR_M[0]) & (wh <= WIND_SENSOR_M[1]))
        counts[f"{label}_temperature_sensor_height"] += int((~t_ok & np.isfinite(t2)).sum())
        counts[f"{label}_wind_sensor_height"] += int((~w_ok & np.isfinite(spd)).sum())
        t2 = np.where(t_ok, t2, np.nan)
        td2 = np.where(t_ok, td2, np.nan)
        good_w = _wind_ok(spd, wdir, counts, label) & w_ok
        names = _texts(group["station_name"], rows) if "station_name" in group else sid[keep]
        received = _received(group, rows)
        lat, lon = _numbers(group["latitude"], rows), _numbers(group["longitude"], rows)
        for n, row in enumerate(rows):
            key = (_wmo(sid[keep][n]), when[keep][n])
            if key in seen:
                counts[f"{label}_duplicate_of_bufr"] += 1
                continue
            seen.add(key)
            td_n, rh_n = _moisture(t2[n:n + 1], td2[n:n + 1], rh2[n:n + 1])
            out.append(Report(
                family="land", fm=12, station=_wmo(sid[keep][n]), name=str(names[n]) or sid[keep][n],
                lat=float(lat[n]), lon=float(lon[n]), elev=float(elev[n]), time=when[keep][n], received=received[n],
                p=[ps[n]], z=[elev[n]], t=[t2[n]], td=td_n, rh=rh_n,
                spd=[spd[n] if good_w[n] else np.nan], wdir=[wdir[n] if good_w[n] else np.nan],
                slp=float(slp[n]), group=path))
    return out


def _wmo(station_id: str) -> str:
    """The WMO block and station of a WIGOS ``0-20000-0-NNNNN`` id; other ids unchanged."""
    m = re.fullmatch(r"0-2000[01]-0-(\w+)", station_id)
    return m.group(1) if m else station_id


# --------------------------------------------------------------- ocean: ships, buoys, TAC
BUOY_KINDS = ("buoy", "moored_buoy", "drifting_buoy", "wave_buoy", "uncrewed_surface_vehicle")


def ocean_reports(root, start, end, box, counts: Counter) -> list[Report]:
    out, seen = [], set()
    for path, label in (("ocean/surface", "ocean"), ("ocean/ship_tac", "ship_tac")):
        if path not in root:
            continue
        group = root[path]
        index, time = _select(group, start, end, box)
        counts[f"{label}_in_window"] += len(index)
        if index.size == 0:
            continue
        ok = _quality_ok(group, "quality_flag", index, counts, label)
        rows = index[ok]
        get = lambda name: _value(group, name, rows, counts, label)  # noqa: E731
        t, td, rh = get("air_temperature"), get("dewpoint_temperature"), get("relative_humidity")
        spd, wdir = get("wind_speed"), get("wind_direction")
        ps, slp = get("surface_air_pressure"), get("mean_sea_level_pressure")
        good_w = _wind_ok(spd, wdir, counts, label)
        kind = _texts(group["platform_kind"], rows) if "platform_kind" in group else np.full(len(rows), "ship")
        ident = None
        for name in ("call_sign", "wigos_id", "station_name"):
            if name in group:
                values = _texts(group[name], rows)
                ident = values if ident is None else np.where(ident == "", values, ident)
        ident = ident if ident is not None else np.full(len(rows), "")
        lat, lon = _numbers(group["latitude"], rows), _numbers(group["longitude"], rows)
        received = _received(group, rows)
        for n, row in enumerate(rows):
            if kind[n] not in ("ship",) + BUOY_KINDS:
                counts[f"{label}_platform_{kind[n] or 'unknown'}"] += 1
                continue
            sid = str(ident[n]) or f"{label}-{int(row)}"
            key = (sid, time[row])
            if key in seen:
                counts[f"{label}_duplicate"] += 1
                continue
            seen.add(key)
            buoy = kind[n] in BUOY_KINDS
            td_n, rh_n = _moisture(t[n:n + 1], td[n:n + 1], rh[n:n + 1])
            out.append(Report(
                family="ocean", fm=18 if buoy else 13, station=sid, name=sid, lat=float(lat[n]),
                lon=float(lon[n]), elev=0.0, time=time[row], received=received[n],
                p=[ps[n]], z=[0.0], t=[t[n]], td=td_n, rh=rh_n,
                spd=[spd[n] if good_w[n] else np.nan], wdir=[wdir[n] if good_w[n] else np.nan],
                slp=float(slp[n]), group=path))
    return out


# --------------------------------------------------------------------------- soundings
_PART = re.compile(r"(?:TEMP|PILOT)_(AA|BB|CC|DD)_|\b(?:TT|PP)(AA|BB|CC|DD)(?=\d|\b)")


def part_label(source_key: str, heading: str = "") -> str:
    """'AA'..'DD' from the object name or the TAC heading; '' for a BUFR sounding."""
    m = _PART.search(source_key) or _PART.search(heading)
    return (m.group(1) or m.group(2)) if m else ""


def _interp_lnp(p_ref, x_ref, p, max_ratio=1.15):
    """``x_ref`` at ``p`` by ln p interpolation, only between reference levels at most ``max_ratio`` apart
    in pressure (or within 0.5 hPa of one); NaN elsewhere, as the radiosonde referee's rule."""
    ok = np.isfinite(p_ref) & np.isfinite(x_ref) & (p_ref > 0)
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    if ok.sum() < 1:
        return out
    order = np.argsort(p_ref[ok], kind="stable")
    pp, xx = p_ref[ok][order], x_ref[ok][order]
    for k, target in enumerate(p):
        if not (np.isfinite(target) and target > 0):
            continue
        near = np.nonzero(np.abs(pp - target) <= 50.0)[0]
        if near.size:
            out[k] = xx[near[np.argmin(np.abs(pp[near] - target))]]
            continue
        hi = np.searchsorted(pp, target)
        if 0 < hi < pp.size and pp[hi] / pp[hi - 1] <= max_ratio:
            w = (np.log(target) - np.log(pp[hi - 1])) / (np.log(pp[hi]) - np.log(pp[hi - 1]))
            out[k] = xx[hi - 1] + w * (xx[hi] - xx[hi - 1])
    return out


def pressure_scale_fault(label: str, p, t, ref_p=None, ref_t=None) -> str:
    """Why a TEMP part's pressures are wrong, or '' when they are not.

    * ``'pressure_out_of_range'``: any part with a level above
      :data:`PRESSURE_MAX_PA` or at a non-positive pressure (a whole sounding
      stored at ten times its pressure, with nothing to compare it with).
    * ``'part_above_100hpa'``: a Part C or D holds only levels above
      100 hPa by definition, yet has a level below it (allowing 1 hPa).
    * ``'pressure_scale_x10'``: over the levels where the launch's other
      parts (``ref_p``, ``ref_t``) overlap it, the part's temperatures
      disagree by more than :data:`PART_DISAGREE_K` (median) and agree within
      :data:`PART_RESCALED_AGREE_K` once every pressure is divided by ten.
    * ``'inconsistent_part'``: they disagree, and rescaling does not help.
    """
    p, t = np.asarray(p, float), np.asarray(t, float)
    if np.any(np.isfinite(p) & ((p > PRESSURE_MAX_PA) | (p <= 0.0))):
        return "pressure_out_of_range"
    if label in ("CC", "DD") and np.any(np.isfinite(p) & (p > 10100.0)):
        return "part_above_100hpa"
    if ref_p is None or len(ref_p) == 0:
        return ""
    ref_p, ref_t = np.asarray(ref_p, float), np.asarray(ref_t, float)
    has_t = np.isfinite(p) & np.isfinite(t)
    direct = _interp_lnp(ref_p, ref_t, p[has_t]) - t[has_t]
    direct = direct[np.isfinite(direct)]
    if direct.size < PART_MIN_OVERLAP or np.median(np.abs(direct)) <= PART_DISAGREE_K:
        return ""
    scaled = _interp_lnp(ref_p, ref_t, p[has_t] / 10.0) - t[has_t]
    scaled = scaled[np.isfinite(scaled)]
    if scaled.size >= PART_MIN_OVERLAP and np.median(np.abs(scaled)) <= PART_RESCALED_AGREE_K:
        return "pressure_scale_x10"
    return "inconsistent_part"


def sonde_reports(root, start, end, box, counts: Counter) -> list[Report]:
    group = _group(root, "upper_air/soundings")
    index, launch = _select(group, start, end, box, time_name="launch_time", lead=SONDE_LAUNCH_LEAD)
    counts["sonde_parts_in_window"] += len(index)
    if index.size == 0:
        return []
    ok = _quality_ok(group, "quality_flag", index, counts, "sonde")
    index = index[ok]
    if index.size == 0:
        return []
    first = _numbers(group["first_level"], index).astype(np.int64)
    count = _numbers(group["level_count"], index).astype(np.int64)
    sid = _texts(group["station_id"], index)
    keys = _texts(group["source_key"], index) if "source_key" in group else np.full(len(index), "")
    heads = _texts(group["bulletin_heading"], index) if "bulletin_heading" in group else np.full(len(index), "")
    lat, lon = _numbers(group["latitude"], index), _numbers(group["longitude"], index)
    elev = _numbers(group["station_elevation"], index)
    received = _received(group, index)
    a, b = int(first.min()), int((first + count).max())
    span = np.arange(a, b)
    level_counts = Counter()
    lv = {name: _value(group, name, span, level_counts, "sonde_level") for name in
          ("pressure", "temperature", "dewpoint_temperature", "geopotential_height", "wind_speed",
           "wind_direction", "time_since_launch", "latitude_displacement", "longitude_displacement")}
    lv["ok"] = _quality_ok(group, "level_quality_flag", span, level_counts, "sonde_level")
    lv["vss"] = np.nan_to_num(_numbers(group["vertical_sounding_significance"], span), nan=0).astype(np.int64) \
        if "vertical_sounding_significance" in group else np.zeros(span.size, np.int64)
    counts.update(level_counts)
    parts = {}
    for n in range(len(index)):
        s0, s1 = int(first[n]) - a, int(first[n] + count[n]) - a
        sl = {k: v[s0:s1] for k, v in lv.items()}
        for k in ("pressure", "temperature", "dewpoint_temperature", "geopotential_height", "wind_speed",
                  "wind_direction"):
            sl[k] = np.where(sl["ok"], sl[k], np.nan)
        parts.setdefault((sid[n], launch[index[n]]), []).append((part_label(keys[n], heads[n]), n, sl))
    lo, hi = _window(start, end)
    out = []
    for (station, when), members in sorted(parts.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        accepted = _consistent_parts(_latest_per_label(members, received, counts), counts)
        if not accepted:
            continue
        rep = _merge_parts(station, when, accepted, lat, lon, elev, received, counts)
        if rep is not None and lo - SONDE_LAUNCH_LEAD <= rep.time <= hi:
            out.append(rep)
    return _one_per_ascent(out, counts)


def _consistent_parts(members, counts):
    """The parts of one launch whose pressures agree, by :func:`pressure_scale_fault`.

    The label rule goes first. Then each part is compared with the union
    of the others, and parts are refused one at a time: every part with
    the ten-times signature, else the one that disagrees most, until
    the rest agree. This way, one bad part in the union cannot condemn
    the good ones. With two parts that disagree and no signature, neither
    can be trusted.
    """
    parts = []
    for member in members:
        fault = pressure_scale_fault(member[0], member[2]["pressure"], member[2]["temperature"])
        if fault:
            counts[f"sonde_part_{fault}"] += 1
        else:
            parts.append(member)
    while len(parts) > 1:
        verdicts = []
        for k, (label, _, sl) in enumerate(parts):
            ref_p = np.concatenate([m[2]["pressure"] for j, m in enumerate(parts) if j != k])
            ref_t = np.concatenate([m[2]["temperature"] for j, m in enumerate(parts) if j != k])
            verdicts.append(pressure_scale_fault(label, sl["pressure"], sl["temperature"], ref_p, ref_t))
        scaled = [k for k, v in enumerate(verdicts) if v == "pressure_scale_x10"]
        if scaled:
            counts["sonde_part_pressure_scale_x10"] += len(scaled)
            parts = [m for k, m in enumerate(parts) if k not in scaled]
            continue
        bad = [k for k, v in enumerate(verdicts) if v == "inconsistent_part"]
        if not bad:
            break
        if len(parts) == 2:
            counts["sonde_part_inconsistent_part"] += 2
            return []
        worst = max(bad, key=lambda k: _median_disagreement(parts, k))
        counts["sonde_part_inconsistent_part"] += 1
        parts.pop(worst)
    return parts


def _median_disagreement(parts, k):
    sl = parts[k][2]
    ref_p = np.concatenate([m[2]["pressure"] for j, m in enumerate(parts) if j != k])
    ref_t = np.concatenate([m[2]["temperature"] for j, m in enumerate(parts) if j != k])
    d = _interp_lnp(ref_p, ref_t, sl["pressure"]) - sl["temperature"]
    d = d[np.isfinite(d)]
    return float(np.median(np.abs(d))) if d.size else 0.0


def _latest_per_label(members, received, counts):
    """One part per label (a correction or relay copy replaces an earlier arrival); BUFR soundings kept."""
    by_label, out = {}, []
    for member in members:
        label = member[0]
        if not label:
            out.append(member)
            continue
        prior = by_label.get(label)
        if prior is not None:
            counts["sonde_part_repeated"] += 1
            mine, theirs = received[member[1]], received[prior[1]]
            later = np.isnat(theirs) or (not np.isnat(mine) and mine >= theirs)
            if not later:
                continue
        by_label[label] = member
    return out + [by_label[k] for k in sorted(by_label)]


PAIRED = (("pressure",), ("temperature", "dewpoint_temperature"), ("geopotential_height",),
          ("wind_speed", "wind_direction"), ("time_since_launch",), ("latitude_displacement",),
          ("longitude_displacement",))


def _merge_parts(station, when, accepted, lat, lon, elev, received, counts) -> Report | None:
    """Merge a launch's parts into one profile, levels in decreasing pressure.

    Levels within 0.5 hPa are one level; a standard level's values win, and
    a value missing in one part is taken from another.
    """
    names = ("pressure", "temperature", "dewpoint_temperature", "geopotential_height", "wind_speed",
             "wind_direction", "time_since_launch", "latitude_displacement", "longitude_displacement")
    stack = {k: np.concatenate([sl[k] for _, _, sl in accepted]) for k in names}
    vss = np.concatenate([sl["vss"] for _, _, sl in accepted])
    valid = np.isfinite(stack["pressure"]) & (stack["pressure"] > 0)
    order = np.lexsort((-(vss[valid] & VSS_STANDARD), -stack["pressure"][valid]))
    rows = np.nonzero(valid)[0][order]
    levels = []
    for r in rows:
        p = stack["pressure"][r]
        if levels and abs(levels[-1]["pressure"] - p) <= 50.0:
            merged = levels[-1]
            for group_names in PAIRED:  # a pair (t and td, speed and direction) comes from one part
                if not any(np.isfinite(merged[k]) for k in group_names) \
                        and any(np.isfinite(stack[k][r]) for k in group_names):
                    for k in group_names:
                        merged[k] = stack[k][r]
            merged["vss"] |= int(vss[r])
            continue
        levels.append({**{k: float(stack[k][r]) for k in names}, "vss": int(vss[r])})
    if not levels:
        counts["sonde_no_levels"] += 1
        return None
    n0 = accepted[0][1]
    station_elev = float(elev[n0])
    p = np.array([lev["pressure"] for lev in levels])
    z = np.array([lev["geopotential_height"] for lev in levels])
    for k, lev in enumerate(levels):
        if lev["vss"] & VSS_SURFACE and np.isfinite(station_elev):
            z[k] = station_elev  # WRFDA splits sonde_sfc on |elv - h| < 0.1
    t = np.array([lev["temperature"] for lev in levels])
    t = np.where((t >= 150.0) & (t <= 350.0), t, np.nan)
    td, rh = _moisture(t, np.array([lev["dewpoint_temperature"] for lev in levels]),
                       np.full(len(levels), np.nan))
    spd = np.array([lev["wind_speed"] for lev in levels])
    wdir = np.array([lev["wind_direction"] for lev in levels])
    good = _wind_ok(spd, wdir, counts, "sonde_level")
    since = np.array([lev["time_since_launch"] for lev in levels])
    level_time = when + np.where(np.isfinite(since), np.round(since), 0).astype("timedelta64[s]")
    dlat = np.nan_to_num(np.array([lev["latitude_displacement"] for lev in levels]))
    dlon = np.nan_to_num(np.array([lev["longitude_displacement"] for lev in levels]))
    arrivals = [received[n] for _, n, _ in accepted if not np.isnat(received[n])]
    return Report(
        family="sonde", fm=35, station=station, name=station, lat=float(lat[n0]), lon=float(lon[n0]),
        elev=station_elev, time=when, received=max(arrivals) if arrivals else np.datetime64("NaT"),
        p=p, z=z, t=t, td=td, rh=rh, spd=np.where(good, spd, np.nan), wdir=np.where(good, wdir, np.nan),
        level_time=level_time, level_lat=lat[n0] + dlat, level_lon=lon[n0] + dlon,
        group="upper_air/soundings", surface=False, extras={"parts": [lbl or "bufr" for lbl, _, _ in accepted]})


def _one_per_ascent(reports, counts):
    """A station's reports launched within :data:`SONDE_SAME_ASCENT` are one ascent: keep the richer."""
    kept = []
    for rep in sorted(reports, key=lambda r: (r.station, r.time)):
        if kept and kept[-1].station == rep.station and abs(rep.time - kept[-1].time) <= SONDE_SAME_ASCENT:
            counts["sonde_same_ascent"] += 1
            if np.isfinite(rep.t).sum() > np.isfinite(kept[-1].t).sum():
                kept[-1] = rep
            continue
        kept.append(rep)
    return kept


# --------------------------------------------------------------------------- aircraft
def isa_pressure(height_m):
    """ICAO standard-atmosphere pressure (Pa) of a pressure altitude (m), as neutral_table.isa_pressure_pa."""
    from woof.obs.neutral_table import isa_pressure_pa
    return np.array([isa_pressure_pa(float(h)) if np.isfinite(h) else np.nan for h in np.ravel(height_m)])


def aircraft_reports(root, start, end, box, counts: Counter) -> list[Report]:
    group = _group(root, "aircraft")
    index, time = _select(group, start, end, box)
    counts["aircraft_in_window"] += len(index)
    if index.size == 0:
        return []
    ok = _quality_ok(group, "quality_flag", index, counts, "aircraft")
    rows = index[ok]
    rid = _texts(group["report_id"], rows) if "report_id" in group else np.array([str(r) for r in rows])
    _, first = np.unique(rid.astype(str), return_index=True)
    counts["aircraft_duplicate_report_id"] += len(rows) - len(first)
    rows = rows[np.sort(first)]
    get = lambda name: _value(group, name, rows, counts, "aircraft")  # noqa: E731
    fl, t, td, rh = get("flight_level"), get("temperature"), get("dewpoint_temperature"), get("relative_humidity")
    spd, wdir = get("wind_speed"), get("wind_direction")
    good_fl = np.isfinite(fl) & (fl > -500.0) & (fl < 20000.0)
    counts["aircraft_flight_level_invalid"] += int((~good_fl).sum())
    good_w = _wind_ok(spd, wdir, counts, "aircraft")
    ident = _texts(group["aircraft_identifier"], rows) if "aircraft_identifier" in group else rid
    lat, lon = _numbers(group["latitude"], rows), _numbers(group["longitude"], rows)
    received = _received(group, rows)
    p = isa_pressure(fl)
    out = []
    for n in np.nonzero(good_fl)[0]:
        t_n = t[n] if 150.0 <= t[n] <= 350.0 else np.nan
        td_n, rh_n = _moisture(np.array([t_n]), td[n:n + 1], rh[n:n + 1])
        out.append(Report(
            family="aircraft", fm=42, station=str(ident[n]) or f"aircraft-{int(rows[n])}", name=str(ident[n]),
            lat=float(lat[n]), lon=float(lon[n]), elev=float("nan"), time=time[rows[n]], received=received[n],
            p=[p[n]], z=[fl[n]], t=[t_n], td=td_n, rh=rh_n,
            spd=[spd[n] if good_w[n] else np.nan], wdir=[wdir[n] if good_w[n] else np.nan],
            group="aircraft", surface=False))
    return out


# --------------------------------------------------------------------------- wind profilers
def profiler_reports(root, start, end, box, counts: Counter) -> list[Report]:
    group = _group(root, "upper_air/wind_profiler")
    index, time = _select(group, start, end, box)
    counts["profiler_in_window"] += len(index)
    if index.size == 0:
        return []
    ok = _quality_ok(group, "quality_flag", index, counts, "profiler")
    index = index[ok]
    if index.size == 0:
        return []
    first = _numbers(group["first_level"], index).astype(np.int64)
    count = _numbers(group["level_count"], index).astype(np.int64)
    sid = _texts(group["station_id"], index)
    lat, lon = _numbers(group["latitude"], index), _numbers(group["longitude"], index)
    elev = _numbers(group["station_elevation"], index)
    received = _received(group, index)
    a, b = int(first.min()), int((first + count).max())
    span = np.arange(a, b)
    level_ok = _quality_ok(group, "level_quality_flag", span, counts, "profiler_level")
    height = _value(group, "height", span, counts, "profiler_level")
    # Most profilers report speed and direction (BUFR 0 11 002/0 11 001), some the components.
    spd_all = _value(group, "wind_speed", span, counts, "profiler_level")
    dir_all = _value(group, "wind_direction", span, counts, "profiler_level")
    u = _value(group, "eastward_wind", span, counts, "profiler_level")
    v = _value(group, "northward_wind", span, counts, "profiler_level")
    from_uv = ~(np.isfinite(spd_all) & np.isfinite(dir_all)) & np.isfinite(u) & np.isfinite(v)
    spd_all = np.where(from_uv, np.hypot(u, v), spd_all)
    dir_all = np.where(from_uv, np.mod(np.degrees(np.arctan2(-u, -v)), 360.0), dir_all)
    dir_all = np.where(from_uv & (spd_all > 0) & (dir_all == 0.0), 360.0, dir_all)
    wind_ok = _wind_ok(spd_all, dir_all, counts, "profiler_level")
    out = []
    for n in range(len(index)):
        s0, s1 = int(first[n]) - a, int(first[n] + count[n]) - a
        good = level_ok[s0:s1] & np.isfinite(height[s0:s1]) & wind_ok[s0:s1]
        if not good.any():
            continue
        spd, wdir, zz = spd_all[s0:s1][good], dir_all[s0:s1][good], height[s0:s1][good]
        order = np.argsort(zz)
        m = int(good.sum())
        out.append(Report(
            family="profiler", fm=132, station=_wmo(sid[n]), name=sid[n], lat=float(lat[n]), lon=float(lon[n]),
            elev=float(elev[n]), time=time[index[n]], received=received[n], p=np.full(m, np.nan), z=zz[order],
            t=np.full(m, np.nan), td=np.full(m, np.nan), rh=np.full(m, np.nan), spd=spd[order],
            wdir=wdir[order], group="upper_air/wind_profiler", surface=False))
    return out


# --------------------------------------------------------------------------- satellite point winds
# The satellite groups are hundreds of millions of rows, appended partition by
# partition, so their rows are nearly in time order: a bisection on sampled
# times finds the rows of a window, with a margin for late arrivals.
SAT_SCAN_MARGIN = np.timedelta64(6, "h")
SAT_SCAN_SAMPLE = 1000
SAT_SCAN_SMALL = 2_000_000  # read a group this small whole

# AMVs: the WMO AMV template (310077) only; its percent confidences are
# typed by 0-01-044 "standard generating application" in the same column.
AMV_TEMPLATES = ("310077",)
QI_WITHOUT_FORECAST = 5  # 0-01-044: quality index without forecast consistency
AMV_QI_MIN = 80.0  # EUMETSAT and NWP SAF's threshold for QI without forecast
AMV_PRESSURE_BOUNDS = (10000.0, 100000.0)
# Polar imagers (0-02-019); WRFDA reads an AMV as polaramv when its name has
# MODIS or its id has AVHRR, so other polar AMVs stay out of obs_gts.
POLAR_INSTRUMENTS = {590: "AVHRR", 591: "AVHRR", 389: "MODIS", 616: "VIIRS"}
# Geostationary satellites (0-01-007): Meteosat and MTG, MTSAT and Himawari,
# GOES, COMS and GK-2A.
GEO_SATELLITES = tuple(range(50, 80)) + tuple(range(171, 175)) + tuple(range(257, 280)) + (810, 811, 812)

# Scatterometers: ASCAT's template (312061) only.  ``wind_vector_cell_quality``
# is the 24-bit flag 0-21-155 (bit 1 the most significant); a cell is refused
# when any bit from 1 to 11 or 16 to 18 is set (retrieval, monitoring, quality
# control, land and ice), and kept with 12 to 15 alone (speed classes, rain
# flag unused).  The land and ice fractions are also checked directly.
SCAT_TEMPLATES = ("312061",)
SCAT_WVC_REFUSE = sum(1 << (24 - bit) for bit in list(range(1, 12)) + [16, 17, 18])
SCAT_ICE_MAX = 0.1
SCAT_HEIGHT_M = 10.0  # equivalent-neutral 10 m wind


def _first_at_or_after(times, target, lo=0, hi=None):
    """The first row whose sampled time reaches ``target`` (bisection on a nearly sorted array)."""
    hi = times.shape[0] if hi is None else hi
    while hi - lo > SAT_SCAN_SAMPLE * 4:
        mid = (lo + hi) // 2
        sample = np.array([], "datetime64[s]")
        for probe in range(mid, hi, SAT_SCAN_SAMPLE):  # step past blocks with no time at all
            sample = _times(times, np.arange(probe, min(probe + SAT_SCAN_SAMPLE, times.shape[0])))
            sample = sample[~np.isnat(sample)]
            if sample.size:
                break
        if sample.size and np.median(sample.astype(np.int64)) < target.astype(np.int64):
            lo = mid
        else:
            hi = mid
    return lo


def _sat_select(group, start, end, box, counts, label):
    """Rows of a satellite group in the window and box: (index, time at index)."""
    lo, hi = _window(start, end)
    times = group["time"]
    n = times.shape[0]
    if n <= SAT_SCAN_SMALL:
        a, b = 0, n
    else:
        a = _first_at_or_after(times, lo - SAT_SCAN_MARGIN)
        b = min(n, _first_at_or_after(times, hi + SAT_SCAN_MARGIN, lo=a) + SAT_SCAN_SAMPLE * 4)
    counts[f"{label}_rows_scanned"] += b - a
    when = _times(times, np.arange(a, b))
    rows = np.nonzero((when >= lo) & (when <= hi))[0]
    lat, lon = _column(group["latitude"], rows + a), _column(group["longitude"], rows + a)
    keep = in_box(lat, lon, box)
    return rows[keep] + a, when[rows[keep]]


def _column(array, index, col=0):
    """Column ``col`` of a 1-D or 2-D array at ``index`` (fill as NaN)."""
    values = _numbers(array, index)
    return values[:, col] if values.ndim == 2 else values


def _quality_column(group, name, index, values, counts, label, col=None):
    """``values`` with NaN where ``<name>_quality`` (the same column, or per-row ``col``) is set."""
    quality = f"{name}_quality"
    if quality not in group:
        return values
    q = _numbers(group[quality], index)
    if q.ndim == 2:
        q = q[np.arange(len(index)), col] if col is not None else q[:, 0]
    bad = np.isfinite(q) & (q != 0) & np.isfinite(values)
    if bad.any():
        counts[f"{label}_{name}_quality"] += int(bad.sum())
    return np.where(bad, np.nan, values)


def isa_altitude_m(pressure_pa):
    """The ICAO standard-atmosphere altitude of a pressure (the inverse of neutral_table.isa_pressure_pa)."""
    from woof.obs import neutral_table as nt
    p = np.asarray(pressure_pa, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        low = nt.ISA_SEA_LEVEL_K / nt.ISA_LAPSE_K_M * (1.0 - (p / nt.ISA_SEA_LEVEL_PA) ** (1.0 / nt.ISA_EXPONENT))
        high = nt.ISA_TROPOPAUSE_M - nt.ISA_STRATOSPHERE_SCALE_M * np.log(p / nt.ISA_TROPOPAUSE_PA)
    return np.where(p >= nt.ISA_TROPOPAUSE_PA, low, high)


def _templates_ok(group, index, allowed, counts, label):
    if "bufr_template" not in group:
        return np.ones(len(index), bool)
    templates = _texts(group["bufr_template"], index)
    ok = np.isin(templates.astype(str), allowed)
    for name, n in Counter(templates[~ok].tolist()).items():
        counts[f"{label}_template_{name or 'unknown'}_not_vetted"] += n
    return ok


def amv_reports(root, start, end, box, counts: Counter) -> list[Report]:
    """Atmospheric motion vectors at their assigned pressure (WRFDA geoamv and polaramv)."""
    if "satellite/ahi_amv" in root:
        counts["ahi_amv_group_skipped_no_quality_index"] += 1  # JMA's template decodes no QI here
    group = _group(root, "satellite/amv")
    index, time = _sat_select(group, start, end, box, counts, "amv")
    counts["amv_in_window"] += len(index)
    if index.size == 0:
        return []
    ok = _quality_ok(group, "quality_flag", index, counts, "amv") \
        & _templates_ok(group, index, AMV_TEMPLATES, counts, "amv")
    index, time = index[ok], time[ok]
    if index.size == 0:
        return []
    sat = _column(group["satellite_identifier"], index)
    instrument = _column(group["satellite_instruments"], index) if "satellite_instruments" in group \
        else np.full(len(index), np.nan)
    p = _quality_column(group, "pressure", index, _column(group["pressure"], index), counts, "amv")
    spd = _value(group, "wind_speed", index, counts, "amv")
    wdir = _value(group, "wind_direction", index, counts, "amv")
    apps = np.atleast_2d(_numbers(group["standard_generating_application"], index).reshape(len(index), -1))
    conf = np.atleast_2d(_numbers(group["percent_confidence"], index).reshape(len(index), -1))
    width = min(apps.shape[1], conf.shape[1])
    typed = np.where(apps[:, :width] == QI_WITHOUT_FORECAST, conf[:, :width], np.nan)
    qi = np.array([row[np.isfinite(row)][0] if np.isfinite(row).any() else np.nan for row in typed])
    lat, lon = _column(group["latitude"], index), _column(group["longitude"], index)
    received = _received(group, index)
    good_w = _wind_ok(spd, wdir, counts, "amv")
    out, seen = [], set()
    for n in np.argsort(received.astype(np.int64), kind="stable"):  # the first arrival of a copy wins
        if not np.isfinite(qi[n]):
            counts["amv_qi_missing"] += 1
            continue
        if qi[n] < AMV_QI_MIN:
            counts["amv_qi_below_minimum"] += 1
            continue
        if not (np.isfinite(p[n]) and AMV_PRESSURE_BOUNDS[0] <= p[n] <= AMV_PRESSURE_BOUNDS[1]):
            counts["amv_pressure_out_of_bounds"] += 1
            continue
        if not good_w[n]:
            continue
        sid = int(sat[n]) if np.isfinite(sat[n]) else -1
        imager = POLAR_INSTRUMENTS.get(int(instrument[n]) if np.isfinite(instrument[n]) else -1)
        if imager is None and sid not in GEO_SATELLITES:
            counts[f"amv_platform_{sid}_unknown"] += 1
            continue
        key = (sid, time[n], round(float(lat[n]), 2), round(float(lon[n]), 2), round(float(p[n]) / 10.0))
        if key in seen:
            counts["amv_duplicate_copy"] += 1
            continue
        seen.add(key)
        station = f"{imager}-{sid}" if imager else f"GEO-{sid}"
        out.append(Report(
            family="amv", fm=88, station=station, name=f"{imager or 'GEO'} AMV {sid}", lat=float(lat[n]),
            lon=float(lon[n]), elev=float("nan"), time=time[n], received=received[n], p=[p[n]], z=[np.nan],
            t=[np.nan], td=[np.nan], rh=[np.nan], spd=[spd[n]], wdir=[wdir[n]], group="satellite/amv",
            surface=False, extras={"qi": float(qi[n]), "polar": imager is not None, "imager": imager or "geo"}))
    return out


def scat_reports(root, start, end, box, counts: Counter) -> list[Report]:
    """Scatterometer 10 m winds, the selected ambiguity only (WRFDA qscat)."""
    group = _group(root, "satellite/scatterometer")
    index, time = _sat_select(group, start, end, box, counts, "scat")
    counts["scat_in_window"] += len(index)
    if index.size == 0:
        return []
    ok = _quality_ok(group, "quality_flag", index, counts, "scat") \
        & _templates_ok(group, index, SCAT_TEMPLATES, counts, "scat")
    index, time = index[ok], time[ok]
    if index.size == 0:
        return []
    selected = _column(group["index_of_selected_wind_vector"], index)
    ambiguities = _column(group["number_of_vector_ambiguities"], index)
    usable = np.isfinite(selected) & (selected >= 1) & (selected <= np.nan_to_num(ambiguities, nan=0.0))
    counts["scat_no_selected_wind"] += int((~usable).sum())
    col = np.where(usable, selected - 1, 0).astype(np.int64)
    rows = np.arange(len(index))
    spd = np.atleast_2d(_numbers(group["wind_speed_at10_m"], index).reshape(len(index), -1))[rows, col]
    wdir = np.atleast_2d(_numbers(group["wind_direction_at10_m"], index).reshape(len(index), -1))[rows, col]
    spd = _quality_column(group, "wind_speed_at10_m", index, spd, counts, "scat", col)
    wdir = _quality_column(group, "wind_direction_at10_m", index, wdir, counts, "scat", col)
    wvc = _column(group["wind_vector_cell_quality"], index) if "wind_vector_cell_quality" in group \
        else np.zeros(len(index))
    wvc_bad = ~np.isfinite(wvc) | ((np.nan_to_num(wvc, nan=0.0).astype(np.int64) & SCAT_WVC_REFUSE) != 0)
    land = np.nanmax(np.atleast_2d(_numbers(group["land_fraction"], index).reshape(len(index), -1)), axis=1) \
        if "land_fraction" in group else np.zeros(len(index))
    ice = _column(group["ice_probability"], index) if "ice_probability" in group else np.zeros(len(index))
    sat = _column(group["satellite_identifier"], index)
    lat, lon = _column(group["latitude"], index), _column(group["longitude"], index)
    received = _received(group, index)
    good_w = _wind_ok(np.where(usable, spd, np.nan), np.where(usable, wdir, np.nan), counts, "scat")
    out, seen = [], set()
    for n in range(len(index)):
        if not usable[n]:
            continue
        if wvc_bad[n]:
            counts["scat_wind_vector_cell_quality"] += 1
            continue
        if np.isfinite(land[n]) and land[n] > 0.0:
            counts["scat_land"] += 1
            continue
        if np.isfinite(ice[n]) and ice[n] > SCAT_ICE_MAX:
            counts["scat_ice"] += 1
            continue
        if not good_w[n]:
            continue
        sid = int(sat[n]) if np.isfinite(sat[n]) else -1
        key = (sid, time[n], round(float(lat[n]), 3), round(float(lon[n]), 3))
        if key in seen:
            counts["scat_duplicate_copy"] += 1
            continue
        seen.add(key)
        out.append(Report(
            family="scat", fm=281, station=f"SCAT-{sid}", name=f"ASCAT {sid}", lat=float(lat[n]),
            lon=float(lon[n]), elev=0.0, time=time[n], received=received[n], p=[np.nan], z=[SCAT_HEIGHT_M],
            t=[np.nan], td=[np.nan], rh=[np.nan], spd=[spd[n]], wdir=[wdir[n]], group="satellite/scatterometer"))
    return out


AMV_THIN_LAYER_PA = 5000.0  # AMVs are thinned per 50 hPa layer


def thin_satellite_winds(reports, thin_km: float, counts: Counter) -> list[Report]:
    """Keep one AMV per ``thin_km`` box and 50 hPa layer (the highest QI) and one scatterometer wind
    per ``thin_km`` box (the first); other reports pass. ``thin_km`` 0 keeps every report."""
    if not thin_km:
        return list(reports)
    if thin_km < 0:
        raise Wis2Error("the thinning distance must be positive (km), or 0 for none")
    step = thin_km / 111.2
    best, out = {}, []
    for rep in reports:
        if rep.family not in THINNED_FAMILIES:
            out.append(rep)
            continue
        row = int(np.floor((rep.lat + 90.0) / step))
        edge = max(abs(row * step - 90.0), abs((row + 1) * step - 90.0))  # the row's poleward edge
        ring = max(1, int(np.floor(360.0 * np.cos(np.radians(min(89.9, edge))) / step)))
        col = int(np.floor(((rep.lon + 180.0) % 360.0) / 360.0 * ring))
        layer = int(rep.p[0] // AMV_THIN_LAYER_PA) if rep.family == "amv" else 0
        key = (rep.family, row, col, layer)
        prior = best.get(key)
        if prior is None or rep.extras.get("qi", 0.0) > prior.extras.get("qi", 0.0):
            best[key] = rep
    for family in THINNED_FAMILIES:
        kept = sum(1 for k in best if k[0] == family)
        counts[f"{family}_thinned"] += sum(1 for r in reports if r.family == family) - kept
    return out + list(best.values())


# --------------------------------------------------------------------------- GNSS-RO
RO_TEMPLATES = ("310026", "001016 310026")  # EUMETSAT's second form carries bending angles only
# 0-33-039 (16 bits, bit 1 the most significant): the bits that refuse a profile.
RO_FLAG_BITS = {1: "non_nominal", 4: "excess_phase_non_nominal", 5: "bending_angle_non_nominal",
                6: "refractivity_non_nominal", 15: "background_profile_not_retrieved"}
RO_PCCF_MIN = 100.0  # da_read_obs_bufrgpsro skips a profile below 100 % confidence
RO_HEIGHT_RANGE_M = (0.0, 30000.0)  # WRFDA's bot_km_gpsro and top_km_gpsro defaults
RO_DNDZ_MIN = -50.0  # N/km; da_qc_gpsref's gpsref_qc_dndz_thresh (super-refraction)
RO_DNDZ2_MAX = 100.0  # N/km^2; gpsref_qc_dndz2_thresh
RO_OBS_GTS_MIN_ERROR = 0.01  # obs_gts prints errors as F7.2: a smaller one would read as 0


def ro_gradient_ok(height_m, refractivity):
    """False at levels whose observed dN/dz is below -50 N/km or whose |d2N/dz2| exceeds 100 N/km^2,
    as ``da_qc_gpsref`` computes them (one-sided at the ends, centred inside, per km)."""
    h, n = np.asarray(height_m, float), np.asarray(refractivity, float)
    top = len(h)
    if top <= 2:
        return np.ones(top, bool)
    zd = np.empty(top)
    zd[0], zd[-1] = h[1] - h[0], h[-1] - h[-2]
    zd[1:-1] = h[2:] - h[:-2]
    zd *= 0.001

    def deriv(a):
        d = np.empty(top)
        d[0], d[-1] = (a[1] - a[0]) / zd[0], (a[-1] - a[-2]) / zd[-1]
        d[1:-1] = (a[2:] - a[:-2]) / zd[1:-1]
        return d

    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = deriv(n)
        d2 = deriv(d1)
    return ~((d1 < RO_DNDZ_MIN) | (np.abs(d2) > RO_DNDZ2_MAX))


def ro_reports(root, start, end, box, counts: Counter) -> list[Report]:
    """GNSS-RO refractivity profiles against height above mean sea level (WRFDA gpsref).

    Template 310026 (ROPP) lists the bending-angle levels (each with its
    tangent point's latitude and longitude), then the refractivity levels.
    As WRFDA's BUFR reader does, refractivity level k takes bending-angle
    level k's location when the two counts agree; otherwise every level
    takes the occultation point.  A profile under 100 % confidence or with
    a non-nominal flag is refused whole.  A level is dropped with 0 %
    confidence, outside 0-30 km (WRFDA's defaults), or where the observed
    profile fails ``da_qc_gpsref``'s gradient checks
    (:func:`ro_gradient_ok`; super-refraction).  Each copy of an occultation (satellite, transmitter, time)
    is kept once, the first to arrive.
    """
    group = _group(root, "satellite/gnss_ro")
    index, time = _sat_select(group, start, end, box, counts, "ro")
    counts["ro_in_window"] += len(index)
    if index.size == 0:
        return []
    ok = _quality_ok(group, "quality_flag", index, counts, "ro") \
        & _templates_ok(group, index, RO_TEMPLATES, counts, "ro")
    if "radio_occultation_data_quality_flags" in group:
        flags = _column(group["radio_occultation_data_quality_flags"], index)
        bits = np.where(np.isfinite(flags), flags, 0).astype(np.int64)
        for bit, meaning in RO_FLAG_BITS.items():
            hit = (bits & (1 << (16 - bit))) != 0
            counts[f"ro_flag_{meaning}"] += int((hit & ok).sum())
            ok &= ~hit
    index, time = index[ok], time[ok]
    if index.size == 0:
        return []

    def table(name):
        return np.atleast_2d(_numbers(group[name], index).reshape(len(index), -1))

    sat = _column(group["satellite_identifier"], index)
    ptid = _column(group["platform_transmitter_id_number"], index)
    conf, counts_rep = table("percent_confidence"), table("extended_delayed_descriptor_replication_factor")
    height, refr = table("height"), table("atmospheric_refractivity")
    lat, lon = table("latitude"), table("longitude")
    received = _received(group, index)
    out, seen = [], set()
    for n in np.argsort(received.astype(np.int64), kind="stable"):  # the first arrival of a copy wins
        sid = int(sat[n]) if np.isfinite(sat[n]) else -1
        tx = int(ptid[n]) if np.isfinite(ptid[n]) else -1
        key = (sid, tx, time[n])
        if key in seen:
            counts["ro_duplicate_copy"] += 1
            continue
        seen.add(key)
        if not (np.isfinite(conf[n, 0]) and conf[n, 0] >= RO_PCCF_MIN):
            counts["ro_profile_confidence_below_100"] += 1
            continue
        n_bend, n_ref = counts_rep[n, 0], counts_rep[n, 1] if counts_rep.shape[1] > 1 else np.nan
        if not (np.isfinite(n_bend) and np.isfinite(n_ref)) or n_ref > min(height.shape[1], refr.shape[1] // 2):
            counts["ro_level_counts_missing"] += 1
            continue
        if n_ref < 1:
            counts["ro_bending_angle_only"] += 1  # no refractivity levels; no bending-angle operator here
            continue
        n_bend, n_ref = int(n_bend), int(n_ref)
        h = height[n, :n_ref]
        value = refr[n, 0:2 * n_ref:2]  # each level: N, then its error
        level_conf = conf[n, 1 + n_bend:1 + n_bend + n_ref]
        level_conf = np.concatenate([level_conf, np.full(n_ref - level_conf.size, np.nan)])
        if n_bend == n_ref and lat.shape[1] > n_ref:
            llat, llon = lat[n, 1:1 + n_ref], lon[n, 1:1 + n_ref]
            off = ~(np.isfinite(llat) & np.isfinite(llon))
            llat, llon = np.where(off, lat[n, 0], llat), np.where(off, lon[n, 0], llon)
        else:
            counts["ro_profiles_at_occultation_point"] += 1
            llat, llon = np.full(n_ref, lat[n, 0]), np.full(n_ref, lon[n, 0])
        good = np.isfinite(h) & np.isfinite(value) & (value > 0.0)
        zero = good & (level_conf == 0)
        counts["ro_level_confidence_zero"] += int(zero.sum())
        good &= ~zero
        outside = good & ((h < RO_HEIGHT_RANGE_M[0]) | (h > RO_HEIGHT_RANGE_M[1]))
        counts["ro_level_outside_0_30_km"] += int(outside.sum())
        good &= ~outside
        if good.sum() > 2:
            kept = np.nonzero(good)[0]
            kept = kept[np.argsort(h[kept], kind="stable")]
            steep = ~ro_gradient_ok(h[kept], value[kept])
            counts["ro_level_gradient_check"] += int(steep.sum())
            good[kept[steep]] = False
        if not good.any():
            counts["ro_no_usable_level"] += 1
            continue
        m = int(good.sum())
        nan = np.full(m, np.nan)
        out.append(Report(
            family="ro", fm=116, station=f"{sid:03d}{tx:02d}", name=f"GNSS-RO {sid} G{tx}",
            lat=float(lat[n, 0]), lon=float(lon[n, 0]), elev=float("nan"), time=time[n], received=received[n],
            p=nan, z=h[good], t=nan, td=nan, rh=nan, spd=nan, wdir=nan, level_lat=llat[good],
            level_lon=llon[good], group="satellite/gnss_ro", surface=False,
            extras={"refractivity": value[good]}))
    return out


READERS = {"land": land_reports, "ocean": ocean_reports, "sonde": sonde_reports,
           "aircraft": aircraft_reports, "profiler": profiler_reports, "amv": amv_reports,
           "scat": scat_reports, "ro": ro_reports}


def read_reports(root, start, end, box, families=FAMILIES) -> tuple[list[Report], Counter]:
    """Every requested family's reports in the window and box, with the QC counts."""
    counts = Counter()
    reports = []
    for family in families:
        if family not in READERS:
            raise Wis2Error(f"unknown family {family!r}; known: {', '.join(FAMILIES + SATELLITE_FAMILIES)}")
        part = [rep for rep in READERS[family](root, start, end, box, counts) if _gross_bounds(rep, counts)]
        counts[f"{family}_reports"] += len(part)
        reports.extend(part)
    return reports, counts


def _outside(values, bounds):
    values = np.asarray(values, float)
    return np.isfinite(values) & ((values < bounds[0]) | (values > bounds[1]))


def _gross_bounds(rep: Report, counts: Counter) -> bool:
    """Blank values outside the gross bounds (counted per family and variable); False when no level
    keeps a pressure or a height, which both writers need to place it."""
    label = rep.family
    for name, bounds in (("t", T_BOUNDS), ("td", T_BOUNDS), ("p", (1.0, PRESSURE_MAX_PA)),
                         ("z", HEIGHT_BOUNDS)):
        bad = _outside(getattr(rep, name), bounds)
        if bad.any():
            counts[f"{label}_{name}_out_of_bounds"] += int(bad.sum())
            setattr(rep, name, np.where(bad, np.nan, getattr(rep, name)))
            if name in ("t", "td"):
                rep.rh = np.where(bad, np.nan, rep.rh)  # rh derived from a bad t or td
    if _outside(rep.slp, SLP_BOUNDS):
        counts[f"{label}_slp_out_of_bounds"] += 1
        rep.slp = float("nan")
    if _outside(rep.elev, ELEV_BOUNDS):
        counts[f"{label}_elevation_out_of_bounds"] += 1
        rep.elev = float("nan")
        if rep.surface:
            rep.z = np.full(len(rep.p), np.nan)
    placed = np.isfinite(rep.p) | np.isfinite(rep.z)
    if not placed.any():
        counts[f"{label}_no_pressure_or_height"] += 1
        return False
    return True


# --------------------------------------------------------------------------- neutral table
SURFACE_LABELS = {  # (family, variable) -> measurement
    ("land", "temperature_k"): "screen_temperature_2m",
    ("land", "dewpoint_k"): "screen_dewpoint_2m",
    ("land", "wind"): "anemometer_wind_10m",
    ("land", "surface_pressure_pa"): "station_pressure",
    ("ocean", "temperature_k"): "platform_temperature",
    ("ocean", "dewpoint_k"): "platform_dewpoint",
    ("ocean", "wind"): "platform_wind",
    ("ocean", "surface_pressure_pa"): "station_pressure",
    ("scat", "wind"): "scatterometer_wind_10m",
}
ALOFT_LABELS = {"sonde": "sonde_level", "aircraft": "aircraft_level", "amv": "amv_assigned_pressure"}


def _aware(when):
    if np.isnat(when):
        return None
    return dt.datetime.fromisoformat(str(np.datetime64(when, "s"))).replace(tzinfo=dt.timezone.utc)


def wind_components(spd, wdir):
    """Earth-relative (u, v) from speed and the direction the wind blows from."""
    rad = np.radians(wdir)
    return -spd * np.sin(rad), -spd * np.cos(rad)


def neutral_rows(reports, table) -> tuple[list, Counter]:
    """``gpuwm-obs.table.v2`` rows (:class:`woof.obs.neutral_table.ObsRow`) with WRFDA's errors.

    Profiler levels carry a height and no pressure, and obs_point's point
    operators are pressure-level only, so they stay out of the table (they
    are in the obs_gts file).  GNSS-RO levels are ``refractivity_n`` rows at
    their tangent point, anchored at their height (``elevation_m``), with
    WRFDA's :func:`~woof.obs.obserr_wrfda.gpsref_error`; ``level_pa`` is the
    height's standard-atmosphere pressure, which only keeps the levels'
    identities apart (the operator reads the height).
    """
    from woof.obs.neutral_table import ObsRow
    from woof.obs.obserr_wrfda import gpsref_error
    rows, counts = [], Counter()
    for rep in reports:
        if rep.family == "profiler":
            counts["profiler_obs_gts_only"] += len(rep.p)
            continue
        source = f"{SOURCE}.{rep.family}"
        received = _aware(rep.received)
        if rep.family == "ro":
            ref = rep.extras["refractivity"]
            level_p = isa_pressure(rep.z)
            for k in range(len(rep.z)):
                if not (np.isfinite(rep.z[k]) and np.isfinite(ref[k])):
                    continue
                rows.append(ObsRow(
                    variable="refractivity_n", value=float(ref[k]),
                    error=float(gpsref_error(ref[k], rep.z[k], rep.level_lat[k])),
                    measurement="ro_refractivity_tangent_point", source=source, station_id=rep.station,
                    latitude_deg=float(rep.level_lat[k]), longitude_deg=float(rep.level_lon[k]),
                    elevation_m=float(rep.z[k]), level_pa=float(level_p[k]), valid_time=_aware(rep.level_time[k]),
                    nominal_time=_aware(rep.time), received_time=received, revision=rep.group))
                counts["rows_ro_refractivity_n"] += 1
            continue
        if rep.surface and not np.isfinite(rep.elev):
            counts[f"{rep.family}_surface_without_elevation"] += 1  # the table needs it on every row
            continue
        for k in range(len(rep.p)):
            p = rep.p[k]
            if not (np.isfinite(p) or np.isfinite(rep.z[k])):
                counts[f"{rep.family}_level_without_pressure_or_height"] += 1
                continue
            err = table.errors(rep.fm, p if np.isfinite(p) else None,
                               rep.z[k] if np.isfinite(rep.z[k]) else None)
            when = _aware(rep.level_time[k])
            base = dict(source=source, station_id=rep.station, latitude_deg=float(rep.level_lat[k]),
                        longitude_deg=float(rep.level_lon[k]), valid_time=when, nominal_time=_aware(rep.time),
                        received_time=received, revision=rep.group)
            if rep.surface:
                base.update(elevation_m=float(rep.elev), level_pa=None)
                label = lambda var: SURFACE_LABELS[(rep.family, var)]  # noqa: E731
            else:
                if not np.isfinite(p):
                    continue
                # The table needs an elevation on every row; a level without a height carries the
                # station's, or (an AMV) its pressure's standard-atmosphere altitude. Pressure-level
                # operators key on level_pa, not on it.
                elevation = rep.z[k] if np.isfinite(rep.z[k]) else rep.elev
                if not np.isfinite(elevation):
                    elevation = isa_altitude_m(p)
                base.update(elevation_m=float(elevation), level_pa=float(p))
                label = lambda var: ALOFT_LABELS[rep.family]  # noqa: E731
            values = [("temperature_k", rep.t[k], err["t"]), ("dewpoint_k", rep.td[k], err["td"])]
            if rep.surface:
                values.append(("surface_pressure_pa", p, err["p"]))
            if np.isfinite(rep.spd[k]) and np.isfinite(rep.wdir[k]):
                u, v = wind_components(rep.spd[k], rep.wdir[k])
                values += [("wind_u_m_s", u, err["u"], "wind"), ("wind_v_m_s", v, err["v"], "wind")]
            for item in values:
                variable, value, error = item[:3]
                if not np.isfinite(value):
                    continue
                rows.append(ObsRow(variable=variable, value=float(value), error=float(error),
                                   measurement=label(item[3] if len(item) > 3 else variable), **base))
                counts[f"rows_{rep.family}_{variable}"] += 1
    return rows, counts


def write_table(rows, path) -> None:
    """Write rows as ``gpuwm-obs.table.v2`` CSV (the header :data:`woof.obs.neutral_table.TABLE_HEADER`)."""
    import csv
    from woof.obs.neutral_table import TABLE_HEADER

    def stamp(when):
        return "" if when is None else when.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def number(value):
        return "" if value is None or (isinstance(value, float) and not math.isfinite(value)) else repr(float(value))

    with open(path, "w", newline="", encoding="utf-8") as handle:
        out = csv.writer(handle)
        out.writerow(TABLE_HEADER)
        for r in rows:
            out.writerow([r.source, r.station_id, number(r.latitude_deg), number(r.longitude_deg),
                          number(r.elevation_m), number(r.level_pa), stamp(r.valid_time), r.variable,
                          number(r.value), number(r.error), r.measurement, stamp(r.nominal_time),
                          stamp(r.published_time), stamp(r.received_time), r.revision])


# --------------------------------------------------------------------------- OBSPROC obs_gts
FM_PLATFORM = {12: "FM-12 SYNOP", 13: "FM-13 SHIP", 18: "FM-18 BUOY", 35: "FM-35 TEMP",
               42: "FM-42 AMDAR", 88: "FM-88 SATOB", 116: "FM-116 GPSRF", 132: "FM-132 PROFL",
               281: "FM-281 QSCAT"}
HEADER_COUNTS = (("SYNOP", 12), ("METAR", None), ("SHIP", 13), ("BUOY", 18), ("BOGUS", None),
                 ("TEMP", 35), ("AMDAR", 42), ("AIREP", None), ("TAMDAR", None), ("PILOT", None),
                 ("SATEM", None), ("SATOB", 88), ("GPSPW", None), ("GPSZD", None), ("GPSRF", 116),
                 ("GPSEP", None), ("SSMT1", None), ("SSMT2", None), ("TOVS", None), ("QSCAT", 281),
                 ("PROFL", 132), ("AIRSR", None), ("OTHER", None))
INFO_FMT = "(A12,1X,A19,1X,A40,1X,I6,3(F12.3,11X),6X,A40)"
SRFC_FMT = "(F12.3,I4,F7.2,F12.3,I4,F7.3)"
EACH_FMT = "(3(F12.3,I4,F7.2),11X,3(F12.3,I4,F7.2),11X,3(F12.3,I4,F7.2))"
SLP_ERROR_PA = 200.0  # OBSPROC's sea-level-pressure error in obs_gts files
PW_ERROR_CM = 0.2


def _f(value, width, decimals):
    text = f"{value:{width}.{decimals}f}"
    if len(text) > width:
        raise ValueError(f"{value} does not fit F{width}.{decimals}")
    return text


def _triple(value, error, decimals=2):
    if value is None or not np.isfinite(value):
        return _f(MISSING, 12, 3) + f"{MISSING_QC:4d}" + _f(error, 7, decimals)
    return _f(value, 12, 3) + f"{0:4d}" + _f(error, 7, decimals)


def obs_gts_header(counts: Counter, domain: dict | None = None) -> list[str]:
    """OBSPROC's obs_gts header: totals, per-type counts, the domain lines and the three formats.

    ``domain`` fills OBSPROC's projection lines (``PHIC``, ``XLONC``, ...);
    WRFDA reads only the format lines, so absent values are written as 0.
    """
    d = {"PHIC": 0.0, "XLONC": 0.0, "TRUE1": 0.0, "TRUE2": 0.0, "XIM11": 1.0, "XJM11": 1.0,
         "IXC": 0, "JXC": 0, "IPROJ": 0, "IDD": 1, "MAXNES": 1, "DIS": 0.0, **(domain or {})}
    total = sum(counts.values())
    lines = [f"TOTAL = {total:6d}, MISS. =-888888.,"]
    cells = [f"{name:<6}= {sum(n for fm, n in counts.items() if fm == code) if code else 0:6d}, "
             for name, code in HEADER_COUNTS]
    for i in range(0, len(cells), 6):
        lines.append("".join(cells[i:i + 6]))
    lines += [f"PHIC  = {d['PHIC']:6.2f}, XLONC = {d['XLONC']:6.2f}, TRUE1 = {d['TRUE1']:6.2f}, "
              f"TRUE2 = {d['TRUE2']:6.2f}, XIM11 = {d['XIM11']:6.2f}, XJM11 = {d['XJM11']:6.2f},",
              "base_temp= 290.00, base_lapse=  50.00, PTOP  =  1000., base_pres=100000., "
              "base_tropo_pres= 20000., base_strat_temp=   215.,",
              f"IXC   = {d['IXC']:6d}, JXC   = {d['JXC']:6d}, IPROJ = {d['IPROJ']:6d}, "
              f"IDD   = {d['IDD']:6d}, MAXNES= {d['MAXNES']:6d},",
              f"NESTIX= {d['IXC']:6d}, ", f"NESTJX= {d['JXC']:6d}, ", f"NUMC  = {1:6d}, ",
              f"DIS   = {d['DIS']:6.2f}, ", f"NESTI = {1:6d}, ", f"NESTJ = {1:6d}, ",
              "INFO  = PLATFORM, DATE, NAME, LEVELS, LATITUDE, LONGITUDE, ELEVATION, ID.",
              "SRFC  = SLP, PW (DATA,QC,ERROR).",
              "EACH  = PRES, SPEED, DIR, HEIGHT, TEMP, DEW PT, HUMID (DATA,QC,ERROR)*LEVELS.",
              f"INFO_FMT = {INFO_FMT}", f"SRFC_FMT = {SRFC_FMT}", f"EACH_FMT = {EACH_FMT}",
              "#------------------------------------------------------------------------------#"]
    return lines


def _ascii(text) -> str:
    """Printable ASCII for a fixed-width text field (names arrive in any script)."""
    return "".join(c if " " <= c <= "~" else "?" for c in str(text))


def obs_gts_report(rep: Report, table) -> list[str]:
    """One report as OBSPROC writes it: INFO, SRFC, then one EACH line per level (decreasing pressure)."""
    order = np.nonzero(np.isfinite(rep.p) | np.isfinite(rep.z))[0]
    if rep.fm == 35:
        order = order[np.argsort(-rep.p[order], kind="stable")]
    elif rep.fm in (116, 132):
        order = order[np.argsort(rep.z[order], kind="stable")]
    lon = ((rep.lon + 180.0) % 360.0) - 180.0
    date = str(np.datetime64(rep.time, "s")).replace("T", "_")
    name = _ascii(rep.name)[:40]
    elev = rep.elev if np.isfinite(rep.elev) else MISSING
    info = (f"{FM_PLATFORM[rep.fm]:<12} {date:<19} {name:<40} {len(order):6d}"
            + "".join(_f(x, 12, 3) + " " * 11 for x in (rep.lat, lon, elev)) + " " * 6
            + f"{_ascii(rep.station)[:40]:<40}")
    slp = rep.slp if np.isfinite(rep.slp) else None
    srfc = _triple(slp, SLP_ERROR_PA) + _f(MISSING, 12, 3) + f"{MISSING_QC:4d}" + _f(PW_ERROR_CM, 7, 3)
    lines = [info, srfc]
    if rep.fm == 116:  # WRFDA's ascii gpsref: refractivity in the dew point slot, its height above sea level
        from woof.obs.obserr_wrfda import gpsref_error
        ref = rep.extras["refractivity"]
        for k in order:
            error = max(gpsref_error(ref[k], rep.z[k], rep.lat), RO_OBS_GTS_MIN_ERROR)
            lines.append(_triple(None, 0.0) * 3 + " " * 11 + _triple(rep.z[k], 0.0) + _triple(None, 0.0)
                         + _triple(ref[k], error) + " " * 11 + _triple(None, 0.0))
        return lines
    for k in order:
        p = rep.p[k] if np.isfinite(rep.p[k]) else None
        z = rep.z[k] if np.isfinite(rep.z[k]) else None
        err = table.errors(rep.fm, p, z)
        wind = np.isfinite(rep.spd[k]) and np.isfinite(rep.wdir[k])
        lines.append(_triple(p, err["p"]) + _triple(rep.spd[k] if wind else None, err["speed"])
                     + _triple(rep.wdir[k] if wind else None, err["direction"]) + " " * 11
                     + _triple(z, err["height"]) + _triple(rep.t[k], err["t"]) + _triple(rep.td[k], err["td"])
                     + " " * 11 + _triple(rep.rh[k], err["rh"]))
    return lines


def write_obs_gts(reports, path, table, *, domain: dict | None = None) -> Counter:
    """Write OBSPROC's ``obs_gts`` ASCII for the reports (WRFDA's ``ob.ascii``).

    A polar AMV from an imager other than AVHRR or MODIS is left out: WRFDA
    would read it as a geostationary one (it keys polaramv on those names).
    A GNSS-RO profile is one FM-116 report at its occultation point (the
    ascii reader has one location per report), errors as
    :func:`~woof.obs.obserr_wrfda.gpsref_error` gives them there, but not
    below 0.01 N (the format's resolution; near 30 km the error is smaller).
    """
    kept = []
    for rep in reports:
        if rep.fm == 88 and rep.extras.get("polar") and rep.extras.get("imager") not in ("AVHRR", "MODIS"):
            continue
        if rep.fm in FM_PLATFORM and len(rep.p):
            kept.append(rep)
    counts = Counter(rep.fm for rep in kept)
    body = []
    for rep in sorted(kept, key=lambda r: (r.time, r.fm, r.station)):
        body.extend(obs_gts_report(rep, table))
    Path(path).write_text("\n".join(obs_gts_header(counts, domain) + body) + "\n", encoding="latin-1")
    return counts


# --------------------------------------------------------------------------- CLI
def _parse_time(text):
    return dt.datetime.fromisoformat(text.replace("Z", "+00:00"))


def fetch(start, end, box, *, families=FAMILIES, table_path=None, obs_gts_path=None, domain=None,
          root=None, snapshot=None, thin_km=0.0) -> dict:
    """Read, QC and write; returns the receipt (counts, store snapshot, outputs)."""
    from woof.obs.obserr_wrfda import ObsErrTable

    snapshot_id = snapshot
    if root is None:
        root, snapshot_id = open_store(snapshot=snapshot)
    reports, counts = read_reports(root, start, end, box, families)
    reports = thin_satellite_winds(reports, thin_km, counts)
    errors = ObsErrTable.load()
    receipt = {"schema": "woof.obs.wis2-fetch/v1", "store": f"s3://{BUCKET}/{PREFIX}", "snapshot": snapshot_id,
               "start": str(start), "end": str(end), "box": list(box), "families": list(families),
               "thin_km": thin_km, "errors": "WRF 4.7.1 OBSPROC obserr.txt (woof/obs/data/wrfda_obserr.txt)",
               "qc": dict(counts)}
    if table_path:
        rows, row_counts = neutral_rows(reports, errors)
        write_table(rows, table_path)
        receipt["table"] = {"path": str(table_path), "rows": len(rows), **row_counts}
    if obs_gts_path:
        fm_counts = write_obs_gts(reports, obs_gts_path, errors, domain=domain)
        receipt["obs_gts"] = {"path": str(obs_gts_path), **{FM_PLATFORM[k]: v for k, v in sorted(fm_counts.items())}}
    return receipt


def _cli_fetch(args) -> int:
    box = tuple(args.bbox)
    families = tuple(f.strip() for f in args.families.split(",") if f.strip())
    if not args.table and not args.obs_gts:
        print("woof obs wis2 fetch: name --table and/or --obs-gts", file=sys.stderr)
        return 2
    try:
        receipt = fetch(_parse_time(args.start), _parse_time(args.end), box, families=families,
                        table_path=args.table, obs_gts_path=args.obs_gts, snapshot=args.snapshot,
                        thin_km=args.thin_km)
    except Wis2Error as error:
        print(f"woof obs wis2 fetch: {error}", file=sys.stderr)
        return 1
    text = json.dumps(receipt, indent=2, sort_keys=True)
    if args.receipt:
        Path(args.receipt).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def register_cli(obs_sub) -> None:
    """``woof obs wis2 fetch``: the archive reader (opt-in; never run by a forecast)."""
    wis2 = obs_sub.add_parser("wis2", help="WIS2 Global Cache archive (bkr/obs/wis2.icechunk) in-situ reader")
    sub = wis2.add_subparsers(dest="obs_wis2_command", required=True)
    fetch_p = sub.add_parser("fetch", help="read a window and box, QC, and write a neutral table and/or obs_gts")
    fetch_p.add_argument("--start", required=True, help="window start, ISO 8601 UTC")
    fetch_p.add_argument("--end", required=True, help="window end, ISO 8601 UTC")
    fetch_p.add_argument("--bbox", required=True, nargs=4, type=float, metavar=("SOUTH", "NORTH", "WEST", "EAST"))
    fetch_p.add_argument("--families", default=",".join(FAMILIES),
                         help=f"comma list of {', '.join(FAMILIES + SATELLITE_FAMILIES)}; the satellite "
                              f"families ({', '.join(SATELLITE_FAMILIES)}: point winds and GNSS-RO "
                              f"refractivity) are read only when named")
    fetch_p.add_argument("--table", help="write gpuwm-obs.table.v2 CSV here (for the ensemble filter)")
    fetch_p.add_argument("--obs-gts", help="write OBSPROC obs_gts ASCII here (for 3D-Var / WRFDA)")
    fetch_p.add_argument("--receipt", help="also write the JSON receipt here")
    fetch_p.add_argument("--thin-km", type=float, default=0.0,
                         help="keep one AMV per box of this size and 50 hPa layer (highest QI) and one "
                              "scatterometer wind per box; 0 (the default) keeps them all")
    fetch_p.add_argument("--snapshot", help="read this Icechunk snapshot id instead of main's head")
    fetch_p.set_defaults(func=_cli_fetch)


__all__ = ["FAMILIES", "Report", "Wis2Error", "aircraft_reports", "fetch", "land_reports", "neutral_rows",
           "obs_gts_header", "obs_gts_report", "ocean_reports", "open_store", "part_label", "pressure_scale_fault",
           "profiler_reports", "read_reports", "register_cli", "rh_from_dewpoint", "ro_gradient_ok", "ro_reports", "sonde_reports",
           "write_obs_gts", "write_table"]
