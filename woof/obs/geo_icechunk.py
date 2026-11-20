"""Geostationary imager radiances from the ``bkr/geo`` Icechunk stores (opt-in).

Source Cooperative's ``bkr/geo`` prefix (``us-west-2.opendata.source.coop``,
anonymous) holds geostationary imagery as Icechunk repositories.  For GOES
ABI the readable form is ``bkr/geo/virtualized/``: one repository per
satellite and channel, ``goes{16,17,18,19}_radf_C{01..16}``, whose chunks are
*virtual references* into NOAA's own ABI-L1b-RadF netCDF files in the
``s3://noaa-goes{NN}/`` buckets.  So a read is NOAA's L1b radiance itself,
with its DQF, Planck constants and fixed-grid navigation, and nothing is
re-encoded.  Each channel's archive is split in time: ``..._C08_2018-01-03``
holds scans up to that date, the undated repository the newest segment.
:func:`goes_segments` lists them and :func:`read_goes_scene` picks the one
whose time coordinate holds the case, after reading it.

Himawari AHI is read from ``virtualized/himawari{8,9}_isatss_C{01..16}[_<date>]``
by :func:`read_himawari_scene`.  Those are NOAA/STAR's "sectorized" full-disk
CMI products (brightness temperature in K for channels 7-16, reflectance
factor for 1-6, on the GOES fixed-grid convention with the satellite at
140.7 E), referenced from ``s3://noaa-himawari{8,9}/``.  They carry no
quality flag, no Planck constants and only the scene's start time, so a
pixel is rejected only when it is the fill value or not a physical value,
and the scan's end is nominal (see :data:`AHI_SCAN`).

The rest of ``bkr/geo`` (Satpy-processed per-platform stores for GK-2A,
Meteosat and MTG; the GK-2A virtual stores) is listed by
:func:`list_stores` and is the next reader; docs/dev/WRFDA-3DVAR-PLAN.md
section 11.12 has the inventory.

Every read is anonymous and read-only.  Only NOAA's public buckets are
authorised as virtual-chunk containers; a repository naming any other
container is read without it (and its chunks then fail by name).  The
result is an :class:`~woof.obs.imager_scene.ImagerScene` with the true
geometry: the satellite from the file, the view angles from its actual
sub-point, the instrument scan angles from the fixed grid, and a per-row
scan time.  Subsets are cached under ``~/.woof/cache/geo`` (or
``$WOOF_GEO_CACHE``) with a provenance record; no data is committed or
shipped.  ``icechunk`` and ``zarr`` are imported only here, and refused by
name when missing.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from woof.obs.imager_scene import (
    BANDS, BRIGHTNESS_TEMPERATURE, FLAG_LIMB, FLAG_NIGHT, FLAG_OUT_OF_RANGE, LIMB_ZENITH_DEG,
    ImagerScene, Platform, band, geostationary_view, read_scene_netcdf, solar_angles, utc,
    write_scene_netcdf,
)

BUCKET = "us-west-2.opendata.source.coop"
REGION = "us-west-2"
PREFIX = "bkr/geo"
LIST_URL = "https://data.source.coop/bkr"
ROOT_ENV = "WOOF_GEO_ROOT"
CACHE_ENV = "WOOF_GEO_CACHE"
PROVENANCE_SCHEMA = "woof.obs.geo-icechunk-provenance/v1"
#: Virtual-chunk containers this reader authorises: NOAA's public buckets only.
ALLOWED_CONTAINER = re.compile(r"s3://noaa-[a-z0-9-]+/")
#: How many snapshots back from a branch head a read may fall back to when the
#: head is unreadable (a store being written can name a manifest not yet visible).
SNAPSHOT_FALLBACK = 8
J2000 = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)
GOES_STORE = re.compile(r"^goes(?P<sat>1[6-9])_radf_C(?P<ch>\d\d)(?:_(?P<end>\d{4}-\d\d-\d\d))?$")
LICENCE_EUMETSAT = ("EUMETSAT data policy: Core data (hourly level 1, and all data older than a stated age) are "
                    "CC BY 4.0; higher-frequency level 1 data (15-minute and 5-minute SEVIRI) are 'Recommended' "
                    "data, free for non-commercial use with conditions on redistribution.  bkr/geo names no "
                    "licence of its own.  woof caches locally and never redistributes")
HIMAWARI_STORE = re.compile(
    r"^himawari(?P<sat>[89])_isatss_C(?P<ch>\d\d)(?:_(?P<end>\d{4}-\d\d-\d\d))?$")
#: A Himawari full-disk scan is nominally 10 minutes from the store's scene start time.
AHI_SCAN = timedelta(minutes=10)
LICENCE = "NOAA open data (no restrictions; attribution requested)"
ATTRIBUTION = ("GOES-R series ABI L1b radiances, NOAA/NESDIS, via the NOAA Open Data Dissemination "
               "program; Icechunk references from bkr/geo on Source Cooperative.")
AHI_ATTRIBUTION = ("Himawari-8/9 AHI imagery, JMA, processed by NOAA/NESDIS/STAR (sectorized full-disk CMI) "
                   "and distributed through the NOAA Open Data Dissemination program; Icechunk references "
                   "from bkr/geo on Source Cooperative.")
#: ABI fixed grid: GRS80 ellipsoid and perspective height (GOES-R PUG vol. 3, 5.1.2.8).
_A = 6378137.0
_B = 6356752.31414
_H = 35786023.0 + _A


class GeoIcechunkError(RuntimeError):
    """A store, time, band or box the reader cannot serve, named."""


# --------------------------------------------------------------------------- stores
def _require():
    try:
        import icechunk
        import zarr
    except ImportError:
        raise GeoIcechunkError("the bkr/geo reader needs icechunk and zarr "
                               "(`pip install icechunk zarr`); nothing else in woof does") from None
    return icechunk, zarr


def default_root() -> str:
    """``$WOOF_GEO_ROOT`` (a local directory mirroring ``bkr/geo``) or the public bucket."""

    return os.environ.get(ROOT_ENV) or f"s3://{BUCKET}/{PREFIX}"


def cache_root() -> Path:
    override = os.environ.get(CACHE_ENV)
    return Path(override).expanduser() if override else Path.home() / ".woof" / "cache" / "geo"


def _is_remote(root: str) -> bool:
    return root.startswith("s3://")


def list_stores(subdir: str = "virtualized", root: str | None = None) -> list[str]:
    """Repository names (without ``.icechunk``) directly under ``root/subdir``."""

    root = root or default_root()
    if not _is_remote(root):
        base = Path(root) / subdir if subdir else Path(root)
        return sorted(p.name[:-len(".icechunk")] for p in base.glob("*.icechunk") if p.is_dir())
    bucket_prefix = root[len("s3://"):].split("/", 1)[1] if "/" in root[len("s3://"):] else ""
    # data.source.coop serves the listing per account: strip the account from the prefix.
    account, _, rest = bucket_prefix.partition("/")
    prefix = "/".join(p for p in (rest, subdir) if p) + "/"
    names, token = [], None
    while True:
        query = {"prefix": prefix, "delimiter": "/", "list-type": "2"}
        if token:
            query["continuation-token"] = token
        url = f"https://data.source.coop/{account}?{urllib.parse.urlencode(query)}"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "woof-geo"}),
                                        timeout=60) as response:
                text = response.read().decode()
        except OSError as error:
            raise GeoIcechunkError(f"listing {url}: {error}") from error
        names += [m[len(prefix):-len(".icechunk/")] for m in re.findall(r"<Prefix>([^<]+\.icechunk/)</Prefix>", text)]
        more = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", text)
        if not more:
            break
        token = more.group(1)
    return sorted(n for n in names if "/" not in n)


@dataclass(frozen=True)
class Segment:
    """One time segment of one satellite's channel."""

    name: str
    satellite: int
    channel: int
    end: date | None  # last date the segment holds; None for the newest (undated) one


def goes_segments(satellite: int, channel: int, root: str | None = None,
                  names: Iterable[str] | None = None) -> list[Segment]:
    """The ``goes{NN}_radf_C{cc}`` segments, oldest first, the undated one last."""

    out = []
    for name in names if names is not None else list_stores("virtualized", root):
        m = GOES_STORE.match(name)
        if m and int(m["sat"]) == satellite and int(m["ch"]) == channel:
            out.append(Segment(name, satellite, channel, date.fromisoformat(m["end"]) if m["end"] else None))
    return sorted(out, key=lambda s: (s.end is None, s.end or date.max))


def himawari_segments(satellite: int, channel: int, root: str | None = None,
                      names: Iterable[str] | None = None) -> list[Segment]:
    """The ``himawari{8,9}_isatss_C{cc}`` segments, oldest first, the undated one last."""

    out = []
    for name in names if names is not None else list_stores("virtualized", root):
        m = HIMAWARI_STORE.match(name)
        if m and int(m["sat"]) == satellite and int(m["ch"]) == channel:
            out.append(Segment(name, satellite, channel, date.fromisoformat(m["end"]) if m["end"] else None))
    return sorted(out, key=lambda s: (s.end is None, s.end or date.max))


def _storage(name: str, subdir: str, root: str):
    icechunk, _ = _require()
    if _is_remote(root):
        bucket, _, prefix = root[len("s3://"):].partition("/")
        return icechunk.s3_storage(bucket=bucket, prefix=f"{prefix}/{subdir}/{name}.icechunk".replace("//", "/"),
                                   region=REGION, anonymous=True)
    return icechunk.local_filesystem_storage(str(Path(root) / subdir / f"{name}.icechunk"))


def _open_repository(name: str, subdir: str, root: str):
    """The repository, with NOAA's public buckets (only) authorised for its virtual chunks."""

    icechunk, _ = _require()
    storage = _storage(name, subdir, root)
    try:
        repo = icechunk.Repository.open(storage)
    except Exception as error:  # icechunk raises its own error types
        raise GeoIcechunkError(f"{subdir}/{name}: cannot open ({str(error).splitlines()[0]})") from error
    containers = sorted((repo.config.virtual_chunk_containers or {}).keys())
    allowed = [c for c in containers if ALLOWED_CONTAINER.fullmatch(c)]
    if allowed:
        credentials = icechunk.containers_credentials(
            {c: icechunk.s3_anonymous_credentials() for c in allowed})
        repo = icechunk.Repository.open(storage, authorize_virtual_chunk_access=credentials)
    return repo, containers, allowed


class _Arrays:
    """A snapshot's arrays, addressed by their NOAA (``source_name``) names.

    The stores were renamed after ingest (``Rad`` became
    ``abi_l1b_radiances`` and so on) and keep the original name in each
    array's ``source_name``; older snapshots still carry the original names.
    """

    def __init__(self, group):
        self.group = group
        self.by_source = {}
        for key, array in group.arrays():
            self.by_source.setdefault(array.attrs.get("source_name", key), array)
            self.by_source.setdefault(key, array)

    def __contains__(self, name):
        return name in self.by_source

    def __getitem__(self, name):
        try:
            return self.by_source[name]
        except KeyError:
            raise GeoIcechunkError(f"the store has no {name!r} (by name or source_name)") from None


def _open_snapshot(repo, label: str, time_name: str = "t"):
    """The newest readable snapshot of ``main``: (arrays, snapshot id, written, fallback depth).

    Readable means its time coordinate's first and last chunks can be read; a
    store being written can name a manifest that is not yet visible.
    """

    _, zarr = _require()
    problems = []
    for depth, snapshot in enumerate(repo.ancestry(branch="main")):
        if depth >= SNAPSHOT_FALLBACK:
            break
        try:
            group = zarr.open_group(repo.readonly_session(snapshot_id=snapshot.id).store, mode="r")
            arrays = _Arrays(group)
            t = arrays[time_name]
            if t.shape[0]:
                np.asarray(t[0]), np.asarray(t[t.shape[0] - 1])
        except Exception as error:  # a manifest or chunk the snapshot names but the bucket lacks
            problems.append(f"{snapshot.id}: {str(error).splitlines()[0][:120]}")
            continue
        return arrays, snapshot.id, str(snapshot.written_at), depth
    raise GeoIcechunkError(f"{label}: none of the newest {SNAPSHOT_FALLBACK} snapshots is readable: "
                           + "; ".join(problems))


def _j2000(seconds, epoch: datetime = J2000, unit_s: float = 1.0, floor: float = 0.0) -> np.ndarray:
    """Time values (J2000 seconds by default) to datetime64[ms]; NaT where not finite or <= ``floor``."""

    s = np.asarray(seconds, dtype="f8")
    out = np.full(s.shape, np.datetime64("NaT", "ms"), dtype="datetime64[ms]")
    ok = np.isfinite(s) & (s > floor)
    out[ok] = np.datetime64(epoch.replace(tzinfo=None), "ms") + (s[ok] * unit_s * 1000).astype("timedelta64[ms]")
    return out


def _time_axis(t_array) -> tuple[datetime, float]:
    """The epoch and the seconds per unit of a CF time array (``minutes since 2025-01-01 00:00:00``)."""

    units = str(t_array.attrs.get("units", ""))
    m = re.fullmatch(r"\s*(nanoseconds|seconds|minutes|hours|days)\s+since\s+(\d{4}-\d\d-\d\d)(?:[T ](\d\d:\d\d:\d\d))?"
                     r"\s*(?:UTC|Z)?\s*", units)
    if not m:
        raise GeoIcechunkError(f"the time coordinate's units {units!r} are not '<unit> since <date> [<time>]'")
    epoch = datetime.fromisoformat(f"{m[2]}T{m[3] or '00:00:00'}").replace(tzinfo=timezone.utc)
    return epoch, {"nanoseconds": 1e-9, "seconds": 1.0, "minutes": 60.0, "hours": 3600.0, "days": 86400.0}[m[1]]


def _cf_scaled(array, values=None) -> np.ndarray:
    raw = array[:] if values is None else values
    out = np.asarray(raw, dtype="f8")
    scale = array.attrs.get("scale_factor")
    offset = array.attrs.get("add_offset")
    if scale is not None:
        out = out * float(scale)
    if offset is not None:
        out = out + float(offset)
    return out


# --------------------------------------------------------------------------- geometry
def fixed_grid_latlon(x_rad, y_rad, origin_lon_deg: float, height_m: float | None = None,
                      semi_major_m: float = _A, semi_minor_m: float = _B):
    """Latitude and longitude (degrees, NaN off the disc) of fixed-grid angles (GOES-R PUG).

    ``height_m`` is the perspective height above the ellipsoid; the default is
    ABI's 35786023 m (Himawari's sectorized grid names its own).
    """

    ea, eb = float(semi_major_m), float(semi_minor_m)
    eh = ea + (35786023.0 if height_m is None else float(height_m))
    x, y = np.meshgrid(np.asarray(x_rad, float), np.asarray(y_rad, float))
    a = np.sin(x) ** 2 + np.cos(x) ** 2 * (np.cos(y) ** 2 + (ea / eb) ** 2 * np.sin(y) ** 2)
    b = -2.0 * eh * np.cos(x) * np.cos(y)
    c = eh ** 2 - ea ** 2
    disc = b * b - 4.0 * a * c
    with np.errstate(invalid="ignore"):
        rs = (-b - np.sqrt(disc)) / (2.0 * a)
        sx = rs * np.cos(x) * np.cos(y)
        sy = -rs * np.sin(x)
        sz = rs * np.cos(x) * np.sin(y)
        lat = np.degrees(np.arctan((ea / eb) ** 2 * sz / np.sqrt((eh - sx) ** 2 + sy ** 2)))
        lon = origin_lon_deg - np.degrees(np.arctan(sy / (eh - sx)))
    lat[disc < 0] = np.nan
    lon[disc < 0] = np.nan
    lon = ((lon + 180.0) % 360.0) - 180.0
    return lat, lon


def _in_bbox(lat, lon, bbox):
    west, south, east, north = bbox
    inside = (lat >= south) & (lat <= north)
    if west <= east:
        return inside & (lon >= west) & (lon <= east)
    return inside & ((lon >= west) | (lon <= east))


def _window(x, y, origin, bbox, step=16, height_m=None, ellipsoid=(_A, _B)):
    """Row and column slices of the fixed grid covering ``bbox`` (``None`` = the whole disc)."""

    if bbox is None:
        return slice(0, len(y)), slice(0, len(x))
    rows = np.arange(0, len(y), step)
    cols = np.arange(0, len(x), step)
    lat, lon = fixed_grid_latlon(x[cols], y[rows], origin, height_m, *ellipsoid)
    hit = _in_bbox(lat, lon, bbox)
    if not hit.any():
        raise GeoIcechunkError(f"the box {bbox} is not on this satellite's disc (or is smaller than "
                               f"{step} pixels; widen it)")
    r = np.nonzero(hit.any(axis=1))[0]
    c = np.nonzero(hit.any(axis=0))[0]
    r0, r1 = max(rows[r[0]] - step, 0), min(rows[r[-1]] + step + 1, len(y))
    c0, c1 = max(cols[c[0]] - step, 0), min(cols[c[-1]] + step + 1, len(x))
    return slice(int(r0), int(r1)), slice(int(c0), int(c1))


def _platform(satellite: int, sub_lon: float) -> Platform:
    sector = "goes_west" if sub_lon < -100.0 else "goes_east"
    if satellite == 19:
        return Platform("GOES-19", sector, "abi", "abi_g16", sub_lon,
                        "CRTM v2.4.1 has no readable GOES-19 coefficients; abi_g16 stands in "
                        "(docs/dev/WRFDA-3DVAR-PLAN.md 11.3)")
    return Platform(f"GOES-{satellite}", sector, "abi", f"abi_g{satellite}", sub_lon)


def _ahi_platform(satellite: int) -> Platform:
    return Platform(f"Himawari-{satellite}", "himawari", "ahi", f"ahi_himawari{satellite}", 140.7)


def _band_nm(channel: int, family: str = "abi") -> int:
    for nm, b in BANDS.items():
        if b.channels.get(family) == channel:
            return nm
    raise GeoIcechunkError(f"{family.upper()} channel {channel} has no band in the scene schema")


def _abi_band_nm(channel: int) -> int:
    return _band_nm(channel, "abi")


def _band_channel(value, family: str = "abi") -> int:
    """A channel of ``family`` from ``C08``, ``8`` or a wavelength/name the scene schema knows."""

    text = str(value).strip()
    m = re.fullmatch(r"[Cc]?(\d{1,2})", text)
    if m and 1 <= int(m.group(1)) <= 16:
        return int(m.group(1))
    b = band(text)
    if family not in b.channels:
        raise GeoIcechunkError(f"{text}: {family.upper()} does not carry this band")
    return b.channels[family]


# --------------------------------------------------------------------------- read
@dataclass
class _ChannelRead:
    channel: int
    segment: str
    snapshot: str
    snapshot_written: str
    fallback_depth: int
    index: int
    scan_start: datetime
    scan_end: datetime
    values: np.ndarray       # on its own grid, the window's rows/cols at its resolution
    dqf_bad: int
    sha256: str
    containers: list
    sub_lon: float
    origin: float
    x: np.ndarray
    y: np.ndarray
    first_row: int
    full_rows: int
    rows: slice = None
    cols: slice = None
    height: float | None = None   # perspective height above the ellipsoid (None: ABI's)
    ellipsoid: tuple = (_A, _B)   # semi-major and semi-minor axes of the navigation ellipsoid
    #: Per window row, the fraction of the scan elapsed when it was observed (None: north-first,
    #: by row index, as ABI and AHI sweep).
    row_frac: np.ndarray | None = None


def _nearest(times: np.ndarray, when: datetime, tolerance: timedelta, label: str) -> int:
    """The index of the scan nearest ``when`` in a decoded time array, within ``tolerance``."""

    target = np.datetime64(when.astimezone(timezone.utc).replace(tzinfo=None), "ms")
    finite = ~np.isnat(times)
    if not finite.any():
        raise GeoIcechunkError(f"{label}: the time coordinate is empty")
    delta = np.abs((times - target).astype("timedelta64[ms]").astype("f8"))
    delta[~finite] = np.inf
    i = int(np.argmin(delta))
    if delta[i] > tolerance.total_seconds() * 1000.0:
        span = f"{times[finite].min()}..{times[finite].max()}"
        raise GeoIcechunkError(f"{label}: no scan within {tolerance} of {when.isoformat()} "
                               f"(nearest {times[i]}; the segment holds {span})")
    return i


def _locate(t_array, when: datetime, tolerance: timedelta, label: str, *, epoch: datetime = J2000,
            unit_s: float = 1.0, floor: float = 0.0) -> int:
    """:func:`_nearest` without reading the whole time coordinate.

    The stores chunk time a few dozen scans per chunk, so a full read of a
    multi-year segment is thousands of requests.  Scans are appended in time
    order, so a bisection on single elements (a chunk each) finds the
    neighbourhood, and a window around it is read and searched.  If the
    window holds no scan within ``tolerance`` (a segment appended out of
    order, or a gap) the whole coordinate is read before refusing.
    """

    n = int(t_array.shape[0])
    if n == 0:
        raise GeoIcechunkError(f"{label}: the time coordinate is empty")
    target = (when - epoch).total_seconds() / unit_s

    def at(i):
        # the nearest finite value at or after i (then before), and its index
        for j in list(range(i, min(i + 8, n))) + list(range(i - 1, max(i - 8, -1), -1)):
            v = float(np.asarray(t_array[j]))
            if np.isfinite(v) and v > floor:
                return j, v
        return i, float("nan")

    lo, hi = 0, n - 1
    while hi - lo > 64:
        mid = (lo + hi) // 2
        _, v = at(mid)
        if not np.isfinite(v):
            break
        if v < target:
            lo = mid
        else:
            hi = mid
    w0, w1 = max(lo - 64, 0), min(hi + 65, n)
    window = _j2000(np.asarray(t_array[w0:w1]), epoch, unit_s, floor)
    try:
        return w0 + _nearest(window, when, tolerance, label)
    except GeoIcechunkError:
        return _nearest(_j2000(np.asarray(t_array[:]), epoch, unit_s, floor), when, tolerance, label)


_RES_KM = {1: 1.0, 2: 0.5, 3: 1.0, 5: 1.0}   # ABI channels not listed are 2 km


def _resolution_km(channel: int) -> float:
    return _RES_KM.get(channel, 2.0)


def _segment_order(segments, day: date) -> list:
    """Segment indices by distance (days) from ``day`` to each segment's span.

    Segments are sorted by end date, so a segment spans the day after the
    previous end through its own end (an undated one is open-ended).
    """

    def distance(k):
        lo = segments[k - 1].end + timedelta(days=1) if k else date.min
        hi = segments[k].end or date.max
        if day < lo:
            return (lo - day).days
        if day > hi:
            return (day - hi).days
        return 0

    return sorted(range(len(segments)), key=lambda k: (distance(k), k))


def _read_channel(satellite, channel, when, bbox, tolerance, root, segments, window=None,
                  factor: int = 1, family: str = "abi") -> _ChannelRead:
    label_base = f"{'Himawari' if family == 'ahi' else 'GOES'}-{satellite} C{channel:02d}"
    read_store = _read_ahi_from_store if family == "ahi" else _read_from_store
    if not segments:
        raise GeoIcechunkError(f"{label_base}: no virtualized store under {root}")
    misses = []
    for k in _segment_order(segments, when.date()):
        seg = segments[k]
        snap = None
        try:
            repo, containers, allowed = _open_repository(seg.name, "virtualized", root)
            arrays, snap, written, depth = _open_snapshot(repo, seg.name)
            return read_store(seg.name, arrays, snap, written, depth, containers, channel, when,
                              bbox, tolerance, window, factor)
        except GeoIcechunkError as error:
            misses.append(str(error))
        except Exception as error:  # icechunk / zarr errors from the bisection or the data reads
            misses.append(f"{seg.name} (snapshot {snap}): read failed ({type(error).__name__}: "
                          f"{str(error).splitlines()[0][:160] if str(error) else ''})")
    raise GeoIcechunkError(f"{label_base} at {when.isoformat()}: " + " | ".join(misses))


def _read_from_store(name, arrays, snap, written, depth, containers, channel, when, bbox, tolerance,
                     window, factor) -> _ChannelRead:
    index = _locate(arrays["t"], when, tolerance, name)
    x = _cf_scaled(arrays["x"])
    y = _cf_scaled(arrays["y"])
    sub_lon = float(np.asarray(arrays["nominal_satellite_subpoint_lon"][index]))
    if not (np.isfinite(sub_lon) and -180.0 <= sub_lon <= 180.0):
        raise GeoIcechunkError(f"{name} (snapshot {snap}): nominal_satellite_subpoint_lon is {sub_lon}, "
                               "not a longitude in [-180, 180]")
    # the fixed grid's projection origin is the nominal slot (-75.0, -137.0, -89.5),
    # 0.2 degrees off the satellite's sub-point at most
    origin = round(sub_lon * 2.0) / 2.0
    if window is None:
        rows, cols = _window(x, y, origin, bbox)
    else:
        # the coarsest channel's window, scaled to this grid: block-aligned for the block mean
        rows = slice(window[0].start * factor, min(window[0].stop * factor, len(y)))
        cols = slice(window[1].start * factor, min(window[1].stop * factor, len(x)))
    rad = arrays["Rad"]
    raw = np.asarray(rad[index, rows, cols])
    fill = rad.attrs.get("_FillValue")
    dqf = np.asarray(arrays["DQF"][index, rows, cols])
    radiance = _cf_scaled(rad, raw)
    bad = (dqf != 0) | (raw == fill if fill is not None else False)
    radiance[bad] = np.nan
    if channel >= 7:
        fk1, fk2, bc1, bc2 = (float(np.asarray(arrays[n][index])) for n in
                              ("planck_fk1", "planck_fk2", "planck_bc1", "planck_bc2"))
        with np.errstate(invalid="ignore", divide="ignore"):
            values = (fk2 / np.log(fk1 / radiance + 1.0) - bc1) / bc2
        values[~(radiance > 0)] = np.nan
    else:
        kappa0 = float(np.asarray(arrays["kappa0"][index]))
        values = radiance * kappa0 * 100.0
    bounds = np.asarray(arrays["time_bounds"][index], dtype="f8")
    start, end = (J2000 + timedelta(seconds=float(s)) for s in bounds)
    digest = hashlib.sha256(np.ascontiguousarray(raw).tobytes() + np.ascontiguousarray(dqf).tobytes()).hexdigest()
    return _ChannelRead(channel, name, snap, written, depth, index, start, end,
                        values.astype("f4"), int(bad.sum()), digest, containers, sub_lon, origin,
                        x[cols], y[rows], rows.start, len(y), rows, cols)


#: Himawari channels not on the 2 km grid (1 and 2 and 4 are 1 km, 3 is 0.5 km).
_AHI_RES_KM = {1: 1.0, 2: 1.0, 3: 0.5, 4: 1.0}
_AHI_COLUMNS = 5500    # the 2 km full disk


def _read_ahi_from_store(name, arrays, snap, written, depth, containers, channel, when, bbox, tolerance,
                         window, factor) -> _ChannelRead:
    """One Himawari sectorized-CMI channel at the scan nearest ``when`` (see :func:`_read_from_store`)."""

    t = arrays["t"]
    epoch, unit_s = _time_axis(t)
    index = _locate(t, when, tolerance, name, epoch=epoch, unit_s=unit_s, floor=-1.0)
    start = epoch + timedelta(seconds=float(np.asarray(t[index])) * unit_s)
    projection = dict(arrays["fixedgrid_projection"].attrs)
    if projection.get("sweep_angle_axis") != "y" or projection.get("grid_mapping_name") != "geostationary":
        raise GeoIcechunkError(f"{name} (snapshot {snap}): the projection is not a y-sweep geostationary "
                               f"grid ({projection})")
    if (abs(float(projection.get("semi_major", 0)) - _A) > 1.0
            or abs(float(projection.get("semi_minor", 0)) - _B) > 1.0):
        raise GeoIcechunkError(f"{name} (snapshot {snap}): the ellipsoid is not GRS80 ({projection})")
    origin = float(projection["longitude_of_projection_origin"])
    height = float(projection["perspective_point_height"])
    if not (-180.0 <= origin <= 180.0 and 3.0e7 < height < 4.0e7):
        raise GeoIcechunkError(f"{name} (snapshot {snap}): implausible projection origin {origin} / height {height}")
    scale_to_rad = {"microradian": 1.0e-6, "radian": 1.0}.get(str(arrays["x"].attrs.get("units")))
    if scale_to_rad is None:
        raise GeoIcechunkError(f"{name} (snapshot {snap}): x units {arrays['x'].attrs.get('units')!r} "
                               "are neither microradian nor radian")
    x = np.asarray(arrays["x"][:], dtype="f8") * scale_to_rad
    y = np.asarray(arrays["y"][:], dtype="f8") * scale_to_rad
    expected = _AHI_COLUMNS * int(round(2.0 / _AHI_RES_KM.get(channel, 2.0)))
    if len(x) != expected or len(y) != expected:
        raise GeoIcechunkError(f"{name} (snapshot {snap}): C{channel:02d} has a {len(y)}x{len(x)} grid, "
                               f"expected {expected}x{expected}")
    if window is None:
        rows, cols = _window(x, y, origin, bbox, height_m=height)
    else:
        rows = slice(window[0].start * factor, min(window[0].stop * factor, len(y)))
        cols = slice(window[1].start * factor, min(window[1].stop * factor, len(x)))
    cmi = arrays["Sectorized_CMI"]
    raw = np.asarray(cmi[index, rows, cols])
    fill = cmi.attrs.get("_FillValue")
    values = _cf_scaled(cmi, raw)
    bad = (raw == fill) if fill is not None else np.zeros(raw.shape, dtype=bool)
    units = str(cmi.attrs.get("units"))
    if channel >= 7:
        if units != "kelvin":
            raise GeoIcechunkError(f"{name} (snapshot {snap}): C{channel:02d} units {units!r}, expected kelvin")
        bad |= ~(values > 0.0)
    else:
        if units != "1":
            raise GeoIcechunkError(f"{name} (snapshot {snap}): C{channel:02d} units {units!r}, expected a "
                                   "reflectance factor ('1')")
        values = values * 100.0
        bad |= ~(values >= 0.0)
    values[bad] = np.nan
    digest = hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest()
    return _ChannelRead(channel, name, snap, written, depth, index, start, start + AHI_SCAN,
                        values.astype("f4"), int(bad.sum()), digest, containers, origin, origin,
                        x[cols], y[rows], rows.start, len(y), rows, cols, height)


def _to_coarse(values: np.ndarray, factor: int) -> np.ndarray:
    """Block-mean a finer channel (0.5 or 1 km) onto the 2 km grid; a block with any NaN is NaN."""

    if factor == 1:
        return values
    ny, nx = (values.shape[0] // factor) * factor, (values.shape[1] // factor) * factor
    v = values[:ny, :nx].reshape(ny // factor, factor, nx // factor, factor)
    return v.mean(axis=(1, 3))


def _assemble(reads, *, platform_of, family, when, box, source, attribution, geometry, licence=LICENCE):
    """The :class:`ImagerScene` of one scan's channel reads (all on one fixed grid)."""

    starts = {r.channel: r.scan_start for r in reads}
    if max(starts.values()) - min(starts.values()) > timedelta(seconds=60):
        raise GeoIcechunkError("the channels' nearest scans are different scans: "
                               + ", ".join(f"C{c:02d} {t.isoformat()}" for c, t in sorted(starts.items())))
    # the 2 km grid: the coarsest channel's angles
    base = max(reads, key=lambda r: float(np.abs(np.diff(r.x[:2])).max()) if r.x.size > 1 else 0.0)
    step2 = float(np.abs(np.diff(base.x[:2]))[0]) if base.x.size > 1 else 0.0
    values = {}
    for r in reads:
        step = float(np.abs(np.diff(r.x[:2]))[0]) if r.x.size > 1 else step2
        factor = max(int(round(step2 / step)), 1) if step else 1
        v = _to_coarse(r.values, factor)
        ny, nx = min(v.shape[0], base.y.size), min(v.shape[1], base.x.size)
        out = np.full((base.y.size, base.x.size), np.nan, dtype="f4")
        out[:ny, :nx] = v[:ny, :nx]
        values[_band_nm(r.channel, family)] = out
    lat, lon = fixed_grid_latlon(base.x, base.y, base.origin, base.height, *base.ellipsoid)
    platform = platform_of(base.sub_lon)
    zenith, azimuth, _ = geostationary_view(lat, lon, base.sub_lon)
    xx, yy = np.meshgrid(base.x, base.y)
    scan = np.degrees(np.arccos(np.cos(xx) * np.cos(yy)))
    scene_time = base.scan_start + (base.scan_end - base.scan_start) / 2
    # a full disk is swept north to south; the row time is taken as linear between the
    # scan's bounds over the full disk's rows (a documented approximation of the swath timeline).
    row_frac = (base.first_row + np.arange(base.y.size)) / base.full_rows if base.row_frac is None \
        else base.row_frac
    offset = (base.scan_start - scene_time).total_seconds() + row_frac * (base.scan_end - base.scan_start).total_seconds()
    scan_offset = np.repeat(offset[:, None], base.x.size, axis=1).astype("f4")
    finite = np.isfinite(lat)
    sun_zenith, sun_azimuth = solar_angles(scene_time, np.where(finite, lat, 0.0), np.where(finite, lon, 0.0))
    flags = np.zeros(lat.shape, dtype=np.int16)
    flags[zenith > LIMB_ZENITH_DEG] |= FLAG_LIMB
    flags[sun_zenith > 90.0] |= FLAG_NIGHT
    for v in values.values():
        flags[~np.isfinite(v) & finite] |= FLAG_OUT_OF_RANGE
    platform_index = np.where(finite, 0, -1).astype(np.int16)
    provenance = dict(
        schema=PROVENANCE_SCHEMA, source=source,
        licence=licence, attribution=attribution, satellite=platform.name,
        requested_time=when.isoformat(), bbox=box, geometry=geometry,
        channels={f"C{r.channel:02d}": dict(
            store=r.segment, snapshot=r.snapshot, snapshot_written=r.snapshot_written,
            snapshot_fallback=r.fallback_depth, time_index=r.index,
            scan_start=r.scan_start.isoformat(), scan_end=r.scan_end.isoformat(),
            dqf_rejected=r.dqf_bad, raw_sha256=r.sha256, virtual_containers=r.containers)
            for r in reads},
        cached=False)
    return ImagerScene(time=scene_time, lat=lat, lon=lon, values=values, platforms=(platform,),
                       platform_index=platform_index, view_zenith=zenith, view_azimuth=azimuth,
                       scan_angle=scan, solar_zenith=sun_zenith, solar_azimuth=sun_azimuth,
                       flags=flags, provenance=provenance, scan_offset_s=scan_offset)


def _read_scan(satellite, wanted, when, bbox, tolerance, root, family) -> list[_ChannelRead]:
    """Every wanted channel at the scan nearest ``when``, over one window."""

    segments_of = himawari_segments if family == "ahi" else goes_segments
    res_km = (lambda c: _AHI_RES_KM.get(c, 2.0)) if family == "ahi" else _resolution_km
    names = list_stores("virtualized", root)
    # one channel per thread: each read is a few dozen object-store round trips
    from concurrent.futures import ThreadPoolExecutor

    # the window is found once, on the coarsest requested channel's grid, and every finer channel
    # reads the same window scaled by its resolution ratio (so the block means register)
    coarse_ch = max(wanted, key=lambda c: (res_km(c), -c))
    coarse = _read_channel(satellite, coarse_ch, when, bbox, tolerance, root,
                           segments_of(satellite, coarse_ch, names=names), family=family)
    others = [c for c in wanted if c != coarse_ch]

    def read_other(ch):
        factor = max(int(round(res_km(coarse_ch) / res_km(ch))), 1)
        return _read_channel(satellite, ch, when, bbox, tolerance, root,
                             segments_of(satellite, ch, names=names),
                             window=(coarse.rows, coarse.cols), factor=factor, family=family)

    with ThreadPoolExecutor(max_workers=max(min(len(others), 8), 1)) as pool:
        return [coarse] + list(pool.map(read_other, others))


def read_goes_scene(satellite: int, when, *, channels: Iterable = (8, 9, 10, 14), bbox=None,
                    tolerance: timedelta = timedelta(minutes=5), root: str | None = None,
                    cache: Path | str | None = None, refresh: bool = False,
                    use_cache: bool = True) -> ImagerScene:
    """One GOES ABI full-disk scan near ``when`` over ``bbox``, as an :class:`ImagerScene`.

    Brightness temperatures (K) for channels 7-16 from NOAA's own Planck
    constants; reflectance factors (%) for 1-6 from ``kappa0``.  A pixel
    whose DQF is not 0 is NaN and counted.  Finer channels are block-averaged
    onto the coarsest requested channel's grid, over one window chosen on that grid.
    ``bbox`` is west,south,east,north in degrees.

    The cache is keyed by satellite, time (UTC), channels, box, tolerance and
    root, not by the store's contents; the snapshot each channel was read from
    is recorded in the provenance.  A hit is served as is until ``refresh=True``
    (``--refresh``) re-reads it; a cache entry whose digest no longer matches
    is re-read.
    """

    when = utc(when)
    root = root or default_root()
    wanted = sorted({_band_channel(c) for c in channels})
    if satellite not in (16, 17, 18, 19):
        raise GeoIcechunkError(f"GOES-{satellite}: the virtualized stores hold GOES-16 to 19")
    box = [float(v) for v in bbox] if bbox is not None else None
    key = hashlib.sha256(json.dumps([satellite, when.isoformat(), wanted, box, tolerance.total_seconds(),
                                     root]).encode()).hexdigest()[:24]
    directory = Path(cache).expanduser() if cache else cache_root()
    entry = directory / f"goes{satellite}" / when.strftime("%Y%m%dT%H%M") / key
    if use_cache and not refresh and (entry / "provenance.json").is_file():
        try:
            return _read_cached(entry)
        except (ValueError, KeyError, OSError):
            pass  # a tampered, truncated or half-written entry is a cache miss: re-read and rewrite
    reads = _read_scan(satellite, wanted, when, bbox, tolerance, root, "abi")
    scene = _assemble(reads, platform_of=lambda sub_lon: _platform(satellite, sub_lon), family="abi",
                      when=when, box=box, source=f"s3://{BUCKET}/{PREFIX}/virtualized" if _is_remote(root) else root,
                      attribution=ATTRIBUTION,
                      geometry="fixed-grid navigation from the file; view angles from the actual sub-point; "
                               "row time linear between the scan bounds")
    if use_cache:
        write_scene_netcdf(entry, scene, licence=LICENCE, attribution=ATTRIBUTION)
    return scene


def read_himawari_scene(satellite: int, when, *, channels: Iterable = (8, 9, 10, 14), bbox=None,
                        tolerance: timedelta = timedelta(minutes=5), root: str | None = None,
                        cache: Path | str | None = None, refresh: bool = False,
                        use_cache: bool = True) -> ImagerScene:
    """One Himawari-8 or -9 AHI full-disk scan near ``when`` over ``bbox``, as an :class:`ImagerScene`.

    From ``virtualized/himawari{8,9}_isatss_C{cc}``: brightness temperature
    (K) for channels 7-16 and reflectance factor (%) for 1-6, as the store
    holds them (NOAA/STAR's sectorized CMI, so with no raw radiance or
    quality flag).  A pixel is NaN where the store holds its fill value or
    a non-physical value; the count is in the provenance as
    ``dqf_rejected``.  The scan time is the store's scene start time and its
    end is nominal (:data:`AHI_SCAN`).  Channels are given as ``C08``, ``8``
    or a wavelength; the cache and ``refresh`` work as for
    :func:`read_goes_scene`.
    """

    when = utc(when)
    root = root or default_root()
    if satellite not in (8, 9):
        raise GeoIcechunkError(f"Himawari-{satellite}: the virtualized stores hold Himawari-8 and -9")
    wanted = sorted({_band_channel(c, "ahi") for c in channels})
    box = [float(v) for v in bbox] if bbox is not None else None
    key = hashlib.sha256(json.dumps(["ahi", satellite, when.isoformat(), wanted, box,
                                     tolerance.total_seconds(), root]).encode()).hexdigest()[:24]
    directory = Path(cache).expanduser() if cache else cache_root()
    entry = directory / f"himawari{satellite}" / when.strftime("%Y%m%dT%H%M") / key
    if use_cache and not refresh and (entry / "provenance.json").is_file():
        try:
            return _read_cached(entry)
        except (ValueError, KeyError, OSError):
            pass  # a tampered, truncated or half-written entry is a cache miss: re-read and rewrite
    reads = _read_scan(satellite, wanted, when, bbox, tolerance, root, "ahi")
    scene = _assemble(reads, platform_of=lambda sub_lon: _ahi_platform(satellite), family="ahi",
                      when=when, box=box,
                      source=f"s3://{BUCKET}/{PREFIX}/virtualized" if _is_remote(root) else root,
                      attribution=AHI_ATTRIBUTION,
                      geometry="fixed-grid navigation from the store's projection; view angles from the "
                               "projection's sub-point; scene time is the store's start time, the scan end "
                               f"is nominal ({AHI_SCAN}) with row time linear over it")
    if use_cache:
        write_scene_netcdf(entry, scene, licence=LICENCE, attribution=AHI_ATTRIBUTION)
    return scene


# --------------------------------------------------------------------------- Satpy L1b stores
#: The Satpy-processed top-level stores of ``bkr/geo`` that hold Meteosat SEVIRI (one array per
#: channel, brightness temperature in K or reflectance in %, stored as float16, 3 km).  The
#: stores name the satellite of each scan themselves.
METEOSAT_STORES = {"msg": "msg_3000m", "rss": "rss_3000m", "odegree": "odegree_3000m"}
EPOCH_1970 = datetime(1970, 1, 1, tzinfo=timezone.utc)
SEVIRI_SENSORS = {f"Meteosat-{n}": f"seviri_m{n:02d}" for n in (8, 9, 10, 11)}
SATPY_ATTRIBUTION = ("Contains modified EUMETSAT Meteosat data, processed with Satpy; Icechunk stores from "
                     "bkr/geo on Source Cooperative.")
_SATPY_CHANNEL = re.compile(r"^channel_(\d\d)_(reflectance|brightness_temperature)_\d")


def _ns_datetime(value) -> datetime:
    """A datetime from nanoseconds since 1970 (an int64 or a datetime64[ns])."""

    return EPOCH_1970 + timedelta(microseconds=int(np.asarray(value).astype("i8")) / 1000.0)


def _satpy_scan(arrays: "_Arrays", index: int) -> dict:
    """One scan's platform, projection longitude, actual sub-longitude and observation bounds."""

    if "observation_start_time" in arrays:
        start, end = (_ns_datetime(arrays[k][index]) for k in ("observation_start_time", "observation_end_time"))
        platform = str(np.asarray(arrays["instrument"][index]))
        origin = float(np.asarray(arrays["projection_longitude"][index]))
        actual = float(np.asarray(arrays["satellite_actual_longitude"][index]))
        height = float(np.asarray(arrays["projection_altitude"][index]))
        bounds = "observation start and end"
    else:   # the archive store keeps the orbit as a Python-literal string and only nominal times
        orbit = ast.literal_eval(str(np.asarray(arrays["orbital_parameters"][index])))
        start, end = (_ns_datetime(arrays[k][index]) for k in ("start_time", "end_time"))
        platform = str(np.asarray(arrays["platform_name"][index]))
        origin = float(orbit["projection_longitude"])
        actual = float(orbit["satellite_actual_longitude"])
        height = float(orbit["projection_altitude"])
        bounds = "nominal start and end"
    if not (np.isfinite(origin) and np.isfinite(actual) and -180.0 <= origin <= 180.0):
        raise GeoIcechunkError(f"scan {index}: projection longitude {origin} / satellite longitude {actual}")
    return dict(platform=platform, origin=origin, actual=actual, height=height, start=start, end=end,
                bounds=bounds)


def read_meteosat_scene(service: str, when, *, channels: Iterable = ("6200", "7330", "11200"), bbox=None,
                        tolerance: timedelta = timedelta(minutes=5), root: str | None = None,
                        cache: Path | str | None = None, refresh: bool = False,
                        use_cache: bool = True) -> ImagerScene:
    """One Meteosat SEVIRI scan near ``when`` over ``bbox``, from a ``bkr/geo`` Satpy store.

    ``service`` is ``msg`` (``msg_3000m``, 2025 on), ``rss`` (``rss_3000m``, the
    5-minute rapid scan at 9.5 E) or ``odegree`` (``odegree_3000m``, the 0 degree
    archive, 2018-2022).  The satellite of each scan (Meteosat-8 to 11) is the
    store's own and names the CRTM sensor.  Channels are SEVIRI numbers
    (``C05`` or ``5``) or wavelengths (``6200``); values are brightness
    temperature (K) for channels 4-11 and reflectance (%) for 1-3, as Satpy
    calibrated them from the L1.5 native files, stored as float16 (so a 250 K
    temperature is quantised at 0.125 K).  The scan is matched by the
    store's nominal time label, which is the end of its slot (the scan
    observed 17:45-17:58 is "18:00"); the scene's time is the midpoint of
    the scan's observation bounds (the archive store has only nominal
    bounds), row time runs south to north across them, and a pixel is NaN
    where the store is.  The cache and
    ``refresh`` work as for :func:`read_goes_scene`.
    """

    when = utc(when)
    root = root or default_root()
    if service not in METEOSAT_STORES:
        raise GeoIcechunkError(f"Meteosat service {service!r}: use one of {sorted(METEOSAT_STORES)}")
    wanted = sorted({_band_channel(c, "seviri") for c in channels})
    box = [float(v) for v in bbox] if bbox is not None else None
    key = hashlib.sha256(json.dumps(["seviri", service, when.isoformat(), wanted, box,
                                     tolerance.total_seconds(), root]).encode()).hexdigest()[:24]
    directory = Path(cache).expanduser() if cache else cache_root()
    entry = directory / f"meteosat-{service}" / when.strftime("%Y%m%dT%H%M") / key
    if use_cache and not refresh and (entry / "provenance.json").is_file():
        try:
            return _read_cached(entry)
        except (ValueError, KeyError, OSError):
            pass  # a tampered, truncated or half-written entry is a cache miss: re-read and rewrite
    name = METEOSAT_STORES[service]
    try:
        repo, containers, _ = _open_repository(name, "", root)
        arrays, snap, written, depth = _open_snapshot(repo, name, "time")
        reads, meta = _read_satpy_scan(arrays, name, snap, written, depth, containers, wanted, when, box,
                                       tolerance, "seviri")
    except GeoIcechunkError:
        raise
    except Exception as error:  # icechunk / zarr errors
        raise GeoIcechunkError(f"{name} at {when.isoformat()}: read failed ({type(error).__name__}: "
                               f"{str(error).splitlines()[0][:160] if str(error) else ''})") from error
    sensor = SEVIRI_SENSORS.get(meta["platform"])
    if sensor is None:
        raise GeoIcechunkError(f"{name}: the scan's platform {meta['platform']!r} is not Meteosat-8 to 11")
    origin = meta["origin"]
    sector = "meteosat_rss" if abs(origin - 9.5) < 1.0 else "meteosat_iodc" if origin > 30.0 else "meteosat_0"
    platform = Platform(meta["platform"], sector, "seviri", sensor, origin)
    scene = _assemble(reads, platform_of=lambda sub_lon: platform, family="seviri", when=when, box=box,
                      source=f"s3://{BUCKET}/{PREFIX}" if _is_remote(root) else root,
                      attribution=SATPY_ATTRIBUTION, licence=LICENCE_EUMETSAT,
                      geometry=f"navigation from the store's per-scan geostationary coordinates (MSG "
                               f"ellipsoid); view angles from the satellite's actual longitude; scan time "
                               f"is the midpoint of the {meta['bounds']}, row time linear south to north")
    if use_cache:
        write_scene_netcdf(entry, scene, licence=LICENCE_EUMETSAT, attribution=SATPY_ATTRIBUTION)
    return scene


#: MSG's reference ellipsoid (Satpy's ``a=6378169``, ``rf=295.488065897014``) and FCI's WGS84.
_ELLIPSOIDS = {"seviri": (6378169.0, 6378169.0 * (1.0 - 1.0 / 295.488065897014)), "fci": (_A, _B)}


def _read_satpy_scan(arrays, name, snap, written, depth, containers, wanted, when, box, tolerance, family):
    """The wanted channels of the scan nearest ``when`` in a Satpy store, over one window.

    Returns the per-channel reads (on the store's 3 km grid, navigated with the scan's own
    geostationary coordinates) and the scan's metadata.
    """

    # the scan is matched by its nominal time label, the end of its slot (the 17:45 scan is "18:00")
    index = _locate(arrays["time"], when, tolerance, name, epoch=EPOCH_1970, unit_s=1e-9)
    meta = _satpy_scan(arrays, index)
    ellipsoid = _ELLIPSOIDS[family]
    height = meta["height"]
    x_m = np.asarray(arrays["x_geostationary_coordinates"][index], dtype="f8")
    y_m = np.asarray(arrays["y_geostationary_coordinates"][index], dtype="f8")
    if not (np.isfinite(x_m).all() and np.isfinite(y_m).all() and 3.0e7 < height < 4.0e7):
        raise GeoIcechunkError(f"{name} (snapshot {snap}): scan {index} has no usable geostationary "
                               f"coordinates (height {height})")
    x, y = x_m / height, y_m / height
    rows, cols = _window(x, y, meta["origin"], box, height_m=height, ellipsoid=ellipsoid)
    # the instrument scans south to north: the fraction of the scan elapsed at each row
    row_frac = (y[rows] - y.min()) / (y.max() - y.min())
    by_channel = {}
    for key, array in arrays.by_source.items():
        m = _SATPY_CHANNEL.match(key)
        if m and array.ndim == 3:
            by_channel[int(m[1])] = (key, m[2], array)
    missing = [c for c in wanted if c not in by_channel]
    if missing:
        raise GeoIcechunkError(f"{name} has no channel {missing} (it holds {sorted(by_channel)})")
    from concurrent.futures import ThreadPoolExecutor

    def read(channel):
        key, quantity, array = by_channel[channel]
        raw = np.asarray(array[index, rows, cols])
        values = raw.astype("f4")
        bad = ~np.isfinite(values) | (~(values > 0.0) if quantity == "brightness_temperature" else ~(values >= 0.0))
        values[bad] = np.nan
        digest = hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest()
        return _ChannelRead(channel, name, snap, written, depth, index, meta["start"], meta["end"], values,
                            int(bad.sum()), digest, containers, meta["actual"], meta["origin"], x[cols],
                            y[rows], rows.start, len(y), rows, cols, height, ellipsoid, row_frac)

    with ThreadPoolExecutor(max_workers=max(min(len(wanted), 8), 1)) as pool:
        return list(pool.map(read, wanted)), meta


def _read_cached(entry: Path) -> ImagerScene:
    scene = read_scene_netcdf(entry)
    scene.provenance["cached"] = True
    return scene


# --------------------------------------------------------------------------- inventory
def coverage(name: str, subdir: str = "virtualized", root: str | None = None) -> dict:
    """One repository's time coverage and readable snapshot, for ``woof obs geo list``."""

    root = root or default_root()
    repo, containers, allowed = _open_repository(name, subdir, root)
    time_name = "time" if subdir == "" else "t"   # the Satpy top-level stores name it ``time``
    arrays, snap, written, depth = _open_snapshot(repo, name, time_name)
    try:
        epoch, unit_s = _time_axis(arrays[time_name])
    except GeoIcechunkError:
        epoch, unit_s = J2000, 1.0   # the ABI stores' J2000 seconds
    times = _j2000(np.asarray(arrays[time_name][:]), epoch, unit_s, floor=-1.0 if epoch != J2000 else 0.0)
    finite = times[~np.isnat(times)]
    steps = np.diff(np.sort(finite)).astype("timedelta64[s]").astype("f8") if finite.size > 1 else np.array([])
    return dict(store=name, snapshot=snap, snapshot_written=written, snapshot_fallback=depth,
                scans=int(finite.size), first=str(finite.min()) if finite.size else None,
                last=str(finite.max()) if finite.size else None,
                median_step_s=float(np.median(steps)) if steps.size else None,
                gaps_over_1h=int((steps > 3600).sum()), containers=containers,
                containers_authorised=allowed)


# --------------------------------------------------------------------------- command line
#: Imagers whose radiances woof cannot simulate, so a scene would be unusable for assimilation.
#: Checked against CRTM v2.4.1's coefficient packages on 2026-11-20 (docs/satellite-imagers.md).
NO_CRTM_COEFFICIENTS = {
    "gk2a": "GK-2A AMI: CRTM v2.4.1's coefficient package has no ami_gk2a file (its sensor list holds "
            "only the CRTM 3 name ami_gk2, in a format v2.4 cannot read), so woof has no operator to "
            "simulate AMI radiances and does not read them for assimilation",
}


def _satellite_arg(text: str):
    """``16``-``19`` (GOES, an int), ``himawari8``/``himawari9`` (``h8``, ``h9``) or a Meteosat
    service (``msg``, ``rss``, ``odegree``); the last two kinds are strings."""

    t = str(text).strip().lower()
    if t in ("16", "17", "18", "19"):
        return int(t)
    m = re.fullmatch(r"(?:himawari|h)-?([89])", t)
    if m:
        return f"himawari{m[1]}"
    if t in METEOSAT_STORES:
        return t
    if t in ("gk2a", "gk-2a", "ami"):
        raise argparse.ArgumentTypeError(NO_CRTM_COEFFICIENTS["gk2a"])
    raise argparse.ArgumentTypeError(f"{text!r}: use 16, 17, 18, 19, himawari8, himawari9, msg, rss or odegree")


def fetch_main(args) -> int:
    import sys

    try:
        common = dict(bbox=args.bbox, tolerance=timedelta(minutes=args.tolerance_minutes), root=args.root,
                      cache=args.cache, refresh=args.refresh)
        if args.satellite in METEOSAT_STORES:
            scene = read_meteosat_scene(args.satellite, args.time, channels=args.channels or ("6200", "7330", "11200"),
                                        **common)
        elif isinstance(args.satellite, str):
            scene = read_himawari_scene(int(args.satellite[-1]), args.time,
                                        channels=args.channels or ["C08", "C09", "C10", "C14"], **common)
        else:
            scene = read_goes_scene(args.satellite, args.time, channels=args.channels or ["C08", "C09", "C10", "C14"],
                                    **common)
    except (GeoIcechunkError, KeyError) as error:
        print(f"woof obs geo fetch: {error}", file=sys.stderr)
        return 2
    summary = scene.summary()
    summary["provenance"] = scene.provenance
    print(json.dumps({"time": summary["time"], "shape": summary["shape"],
                      "satellite": scene.platforms[0].name,
                      "bands": {k: v["finite"] for k, v in summary["bands"].items()},
                      "cached": bool(scene.provenance.get("cached"))}))
    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"{'Meteosat' if args.satellite in METEOSAT_STORES else 'Himawari' if isinstance(args.satellite, str) else 'GOES'} "
          f"data: {scene.provenance['licence']}. {scene.provenance['attribution']}", file=sys.stderr)
    return 0


def list_main(args) -> int:
    import sys

    try:
        if args.satellite in METEOSAT_STORES:
            args.subdir = ""
        names = list_stores(args.subdir, args.root)
        if args.satellite:
            if args.satellite in METEOSAT_STORES:
                names = [n for n in names if n == METEOSAT_STORES[args.satellite]]
            elif isinstance(args.satellite, str):
                names = [s.name for s in himawari_segments(int(args.satellite[-1]), args.channel or 8, names=names)]
            else:
                names = [s.name for s in goes_segments(args.satellite, args.channel or 8, names=names)]
        for name in names:
            if args.coverage:
                try:
                    print(json.dumps(coverage(name, args.subdir, args.root)))
                except GeoIcechunkError as error:
                    print(json.dumps({"store": name, "error": str(error)}))
            else:
                print(name)
    except GeoIcechunkError as error:
        print(f"woof obs geo list: {error}", file=sys.stderr)
        return 2
    return 0


def register_cli(obs_sub) -> None:
    """``woof obs geo {list,fetch}`` (``woof fetch geo`` is the fetch door)."""

    from woof.obs.cli import _bbox_wrapping
    from woof.obs.satweave import _satweave_time

    parser = obs_sub.add_parser(
        "geo", help="geostationary imager radiances from the bkr/geo Icechunk stores (GOES ABI L1b, Himawari AHI)")
    sub = parser.add_subparsers(dest="geo_command", required=True)
    fetch = sub.add_parser("fetch", help="read ABI channels near a time over a box into the local cache")
    fetch.add_argument("--satellite", type=_satellite_arg, required=True,
                       help="GOES satellite number (16-19), himawari8 / himawari9, or a Meteosat SEVIRI "
                            "service: msg (0 degrees, 2025 on), rss (rapid scan), odegree (0 degrees, 2018-2022)")
    fetch.add_argument("--time", required=True, type=_satweave_time("--time"),
                       help="UTC time with offset, e.g. 2025-06-01T18:00Z")
    fetch.add_argument("--channels", type=lambda v: [c for c in v.split(",") if c.strip()], default=None,
                       help="comma-separated channels of the instrument (C08 or 8; SEVIRI numbers its own 1-11) or "
                            "band wavelengths (default C08,C09,C10,C14 for ABI and AHI; 6200,7330,11200 for SEVIRI)")
    fetch.add_argument("--bbox", type=_bbox_wrapping, default=None,
                       help="west,south,east,north in degrees (west > east crosses 180); default the full disc")
    fetch.add_argument("--tolerance-minutes", type=float, default=5.0,
                       help="nearest scan accepted within this many minutes (default 5)")
    fetch.add_argument("--root", default=None,
                       help=f"store root: s3://{BUCKET}/{PREFIX} (default) or a local mirror (${ROOT_ENV})")
    fetch.add_argument("--cache", default=None, help=f"cache directory (default ${CACHE_ENV} or ~/.woof/cache/geo)")
    fetch.add_argument("--refresh", action="store_true", help="re-read even when the cache holds the subset")
    fetch.add_argument("--summary-json", default=None, help="write the scene summary and provenance here")
    # a box west of Greenwich starts with '-': let argparse read it as a value (Python < 3.14)
    fetch._negative_number_matcher = re.compile(r"^-\.?\d")
    fetch.set_defaults(func=fetch_main)
    listing = sub.add_parser("list", help="list the bkr/geo stores (and, with --coverage, their time coverage)")
    listing.add_argument("--subdir", default="virtualized", help="sub-prefix to list (default virtualized; '' = top level)")
    listing.add_argument("--satellite", type=_satellite_arg, default=None,
                         help="only this satellite's segments of --channel (16-19, himawari8, himawari9)")
    listing.add_argument("--channel", type=int, default=None,
                         help="ABI or AHI channel for --satellite (default 8)")
    listing.add_argument("--coverage", action="store_true",
                         help="open each store and print its readable snapshot and time coverage (slow)")
    listing.add_argument("--root", default=None, help="store root (as for fetch)")
    listing.set_defaults(func=list_main)
