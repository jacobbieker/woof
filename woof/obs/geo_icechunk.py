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

The rest of ``bkr/geo`` (Satpy-processed per-platform stores for Himawari,
GK-2A, Meteosat and MTG; Himawari and GK-2A virtual stores) is listed by
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
LICENCE = "NOAA open data (no restrictions; attribution requested)"
ATTRIBUTION = ("GOES-R series ABI L1b radiances, NOAA/NESDIS, via the NOAA Open Data Dissemination "
               "program; Icechunk references from bkr/geo on Source Cooperative.")
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


def _open_snapshot(repo, label: str):
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
            t = arrays["t"]
            if t.shape[0]:
                np.asarray(t[0]), np.asarray(t[t.shape[0] - 1])
        except Exception as error:  # a manifest or chunk the snapshot names but the bucket lacks
            problems.append(f"{snapshot.id}: {str(error).splitlines()[0][:120]}")
            continue
        return arrays, snapshot.id, str(snapshot.written_at), depth
    raise GeoIcechunkError(f"{label}: none of the newest {SNAPSHOT_FALLBACK} snapshots is readable: "
                           + "; ".join(problems))


def _j2000(seconds) -> np.ndarray:
    """J2000 seconds to datetime64[ms] (NaT where not finite)."""

    s = np.asarray(seconds, dtype="f8")
    out = np.full(s.shape, np.datetime64("NaT", "ms"), dtype="datetime64[ms]")
    ok = np.isfinite(s) & (s > 0)
    out[ok] = np.datetime64(J2000.replace(tzinfo=None), "ms") + (s[ok] * 1000).astype("timedelta64[ms]")
    return out


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
def fixed_grid_latlon(x_rad, y_rad, origin_lon_deg: float):
    """Latitude and longitude (degrees, NaN off the disc) of ABI fixed-grid angles (GOES-R PUG)."""

    x, y = np.meshgrid(np.asarray(x_rad, float), np.asarray(y_rad, float))
    a = np.sin(x) ** 2 + np.cos(x) ** 2 * (np.cos(y) ** 2 + (_A / _B) ** 2 * np.sin(y) ** 2)
    b = -2.0 * _H * np.cos(x) * np.cos(y)
    c = _H ** 2 - _A ** 2
    disc = b * b - 4.0 * a * c
    with np.errstate(invalid="ignore"):
        rs = (-b - np.sqrt(disc)) / (2.0 * a)
        sx = rs * np.cos(x) * np.cos(y)
        sy = -rs * np.sin(x)
        sz = rs * np.cos(x) * np.sin(y)
        lat = np.degrees(np.arctan((_A / _B) ** 2 * sz / np.sqrt((_H - sx) ** 2 + sy ** 2)))
        lon = origin_lon_deg - np.degrees(np.arctan(sy / (_H - sx)))
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


def _window(x, y, origin, bbox, step=16):
    """Row and column slices of the fixed grid covering ``bbox`` (``None`` = the whole disc)."""

    if bbox is None:
        return slice(0, len(y)), slice(0, len(x))
    rows = np.arange(0, len(y), step)
    cols = np.arange(0, len(x), step)
    lat, lon = fixed_grid_latlon(x[cols], y[rows], origin)
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


def _abi_band_nm(channel: int) -> int:
    for nm, b in BANDS.items():
        if b.channels.get("abi") == channel:
            return nm
    raise GeoIcechunkError(f"ABI channel {channel} has no band in the scene schema")


def _band_channel(value) -> int:
    """An ABI channel from ``C08``, ``8`` or a wavelength/name the scene schema knows."""

    text = str(value).strip()
    m = re.fullmatch(r"[Cc]?(\d{1,2})", text)
    if m and 1 <= int(m.group(1)) <= 16:
        return int(m.group(1))
    b = band(text)
    if "abi" not in b.channels:
        raise GeoIcechunkError(f"{text}: ABI does not carry this band")
    return b.channels["abi"]


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


def _locate(t_array, when: datetime, tolerance: timedelta, label: str) -> int:
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
    target = (when - J2000).total_seconds()

    def at(i):
        # the nearest finite value at or after i (then before), and its index
        for j in list(range(i, min(i + 8, n))) + list(range(i - 1, max(i - 8, -1), -1)):
            v = float(np.asarray(t_array[j]))
            if np.isfinite(v) and v > 0:
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
    window = _j2000(np.asarray(t_array[w0:w1]))
    try:
        return w0 + _nearest(window, when, tolerance, label)
    except GeoIcechunkError:
        return _nearest(_j2000(np.asarray(t_array[:])), when, tolerance, label)


def _read_channel(satellite, channel, when, bbox, tolerance, root, segments) -> _ChannelRead:
    label_base = f"GOES-{satellite} C{channel:02d}"
    if not segments:
        raise GeoIcechunkError(f"{label_base}: no virtualized store under {root}")
    day = when.date()
    # the first segment whose end date is on or after the case, then the others in order of distance
    first = next((k for k, s in enumerate(segments) if s.end is None or s.end >= day), len(segments) - 1)
    order = [first] + [k for k in range(len(segments)) if k != first]
    misses = []
    for k in order:
        seg = segments[k]
        try:
            repo, containers, allowed = _open_repository(seg.name, "virtualized", root)
            arrays, snap, written, depth = _open_snapshot(repo, seg.name)
            index = _locate(arrays["t"], when, tolerance, seg.name)
        except GeoIcechunkError as error:
            misses.append(str(error))
            continue
        break
    else:
        raise GeoIcechunkError(f"{label_base} at {when.isoformat()}: " + " | ".join(misses))
    x = _cf_scaled(arrays["x"])
    y = _cf_scaled(arrays["y"])
    sub_lon = float(np.asarray(arrays["nominal_satellite_subpoint_lon"][index]))
    # the fixed grid's projection origin is the nominal slot (-75.0, -137.0, -89.5),
    # 0.2 degrees off the satellite's sub-point at most
    origin = round(sub_lon * 2.0) / 2.0
    rows, cols = _window(x, y, origin, bbox)
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
    return _ChannelRead(channel, seg.name, snap, written, depth, index, start, end,
                        values.astype("f4"), int(bad.sum()), digest, containers, sub_lon, origin,
                        x[cols], y[rows], rows.start, len(y))


def _to_coarse(values: np.ndarray, factor: int) -> np.ndarray:
    """Block-mean a finer channel (0.5 or 1 km) onto the 2 km grid; a block with any NaN is NaN."""

    if factor == 1:
        return values
    ny, nx = (values.shape[0] // factor) * factor, (values.shape[1] // factor) * factor
    v = values[:ny, :nx].reshape(ny // factor, factor, nx // factor, factor)
    return v.mean(axis=(1, 3))


def read_goes_scene(satellite: int, when, *, channels: Iterable = (8, 9, 10, 14), bbox=None,
                    tolerance: timedelta = timedelta(minutes=5), root: str | None = None,
                    cache: Path | str | None = None, refresh: bool = False,
                    use_cache: bool = True) -> ImagerScene:
    """One GOES ABI full-disk scan near ``when`` over ``bbox``, as an :class:`ImagerScene`.

    Brightness temperatures (K) for channels 7-16 from NOAA's own Planck
    constants; reflectance factors (%) for 1-6 from ``kappa0``.  A pixel
    whose DQF is not 0 is NaN and counted.  Finer channels are block-averaged
    onto the 2 km grid.  ``bbox`` is west,south,east,north in degrees.
    """

    when = utc(when)
    root = root or default_root()
    wanted = sorted({_band_channel(c) for c in channels})
    if satellite not in (16, 17, 18, 19):
        raise GeoIcechunkError(f"GOES-{satellite}: the virtualized stores hold GOES-16 to 19")
    key = hashlib.sha256(json.dumps([satellite, when.isoformat(), wanted, bbox, tolerance.total_seconds(),
                                     root]).encode()).hexdigest()[:24]
    directory = Path(cache).expanduser() if cache else cache_root()
    entry = directory / f"goes{satellite}" / when.strftime("%Y%m%dT%H%M") / key
    if use_cache and not refresh and (entry / "provenance.json").is_file():
        return _read_cached(entry)
    names = list_stores("virtualized", root)
    # one channel per thread: each read is a few dozen object-store round trips
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(len(wanted), 8)) as pool:
        reads = list(pool.map(lambda ch: _read_channel(satellite, ch, when, bbox, tolerance, root,
                                                       goes_segments(satellite, ch, names=names)), wanted))
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
        values[_abi_band_nm(r.channel)] = out
    lat, lon = fixed_grid_latlon(base.x, base.y, base.origin)
    platform = _platform(satellite, base.sub_lon)
    zenith, azimuth, _ = geostationary_view(lat, lon, base.sub_lon)
    xx, yy = np.meshgrid(base.x, base.y)
    scan = np.degrees(np.arccos(np.cos(xx) * np.cos(yy)))
    scene_time = base.scan_start + (base.scan_end - base.scan_start) / 2
    # ABI's full disk is swept north to south; the row time is taken as linear between the
    # scan's bounds over the full disk's rows (a documented approximation of the swath timeline).
    row_frac = (base.first_row + np.arange(base.y.size)) / base.full_rows
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
        schema=PROVENANCE_SCHEMA, source=f"s3://{BUCKET}/{PREFIX}/virtualized" if _is_remote(root) else root,
        licence=LICENCE, attribution=ATTRIBUTION, satellite=platform.name,
        requested_time=when.isoformat(), bbox=list(bbox) if bbox else None,
        geometry="fixed-grid navigation from the file; view angles from the actual sub-point; "
                 "row time linear between the scan bounds",
        channels={f"C{r.channel:02d}": dict(
            store=r.segment, snapshot=r.snapshot, snapshot_written=r.snapshot_written,
            snapshot_fallback=r.fallback_depth, time_index=r.index,
            scan_start=r.scan_start.isoformat(), scan_end=r.scan_end.isoformat(),
            dqf_rejected=r.dqf_bad, raw_sha256=r.sha256, virtual_containers=r.containers)
            for r in reads},
        cached=False)
    scene = ImagerScene(time=scene_time, lat=lat, lon=lon, values=values, platforms=(platform,),
                        platform_index=platform_index, view_zenith=zenith, view_azimuth=azimuth,
                        scan_angle=scan, solar_zenith=sun_zenith, solar_azimuth=sun_azimuth,
                        flags=flags, provenance=provenance, scan_offset_s=scan_offset)
    if use_cache:
        write_scene_netcdf(entry, scene, licence=LICENCE, attribution=ATTRIBUTION)
    return scene


def _read_cached(entry: Path) -> ImagerScene:
    scene = read_scene_netcdf(entry)
    scene.provenance["cached"] = True
    return scene


# --------------------------------------------------------------------------- inventory
def coverage(name: str, subdir: str = "virtualized", root: str | None = None) -> dict:
    """One repository's time coverage and readable snapshot, for ``woof obs geo list``."""

    root = root or default_root()
    repo, containers, allowed = _open_repository(name, subdir, root)
    arrays, snap, written, depth = _open_snapshot(repo, name)
    times = _j2000(np.asarray(arrays["t"][:]))
    finite = times[~np.isnat(times)]
    steps = np.diff(np.sort(finite)).astype("timedelta64[s]").astype("f8") if finite.size > 1 else np.array([])
    return dict(store=name, snapshot=snap, snapshot_written=written, snapshot_fallback=depth,
                scans=int(finite.size), first=str(finite.min()) if finite.size else None,
                last=str(finite.max()) if finite.size else None,
                median_step_s=float(np.median(steps)) if steps.size else None,
                gaps_over_1h=int((steps > 3600).sum()), containers=containers,
                containers_authorised=allowed)


# --------------------------------------------------------------------------- command line
def fetch_main(args) -> int:
    import sys

    try:
        scene = read_goes_scene(args.satellite, args.time, channels=args.channels, bbox=args.bbox,
                                tolerance=timedelta(minutes=args.tolerance_minutes), root=args.root,
                                cache=args.cache, refresh=args.refresh)
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
    print(f"GOES data: {LICENCE}. {ATTRIBUTION}", file=sys.stderr)
    return 0


def list_main(args) -> int:
    import sys

    try:
        names = list_stores(args.subdir, args.root)
        if args.satellite:
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
        "geo", help="geostationary imager radiances from the bkr/geo Icechunk stores (GOES ABI L1b)")
    sub = parser.add_subparsers(dest="geo_command", required=True)
    fetch = sub.add_parser("fetch", help="read ABI channels near a time over a box into the local cache")
    fetch.add_argument("--satellite", type=int, required=True, choices=(16, 17, 18, 19),
                       help="GOES satellite number")
    fetch.add_argument("--time", required=True, type=_satweave_time("--time"),
                       help="UTC time with offset, e.g. 2025-06-01T18:00Z")
    fetch.add_argument("--channels", type=lambda v: [c for c in v.split(",") if c.strip()],
                       default=["C08", "C09", "C10", "C14"],
                       help="comma-separated ABI channels (C08 or 8) or band wavelengths (default C08,C09,C10,C14)")
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
    listing.add_argument("--satellite", type=int, default=None, choices=(16, 17, 18, 19),
                         help="only this GOES satellite's segments of --channel")
    listing.add_argument("--channel", type=int, default=None, help="ABI channel for --satellite (default 8)")
    listing.add_argument("--coverage", action="store_true",
                         help="open each store and print its readable snapshot and time coverage (slow)")
    listing.add_argument("--root", default=None, help="store root (as for fetch)")
    listing.set_defaults(func=list_main)
