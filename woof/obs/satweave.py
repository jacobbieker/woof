"""SatWeave: Google's global geostationary mosaic, read for data assimilation (opt-in).

SatWeave (https://developers.google.com/weathernext/guides/satweave; Google
DeepMind and Google Research, CC BY 4.0) blends GOES-East, GOES-West,
Meteosat 0 deg, Meteosat IODC and Himawari onto one 0.1 deg grid with 17
bands.  This module reads a time, a bounding box and a set of bands, keeps a
local cache of the subset, and returns an :class:`woof.obs.imager_scene.ImagerScene`
with the satellite and view geometry reconstructed per pixel
(docs/dev/WRFDA-3DVAR-PLAN.md sections 11.1-11.2).

Stores (both Zarr v3, consolidated metadata, one ``(1, 1801, 3600)`` chunk per
band per step, ``bytes`` then ``zstd``):

* ``zarr/2016_to_2025/mosaic.zarr``: 30-minute steps, time in
  ``minutes since 2016-01-01``;
* ``zarr/2026_to_present/<YYYYMMDDTHHMMSS>/mosaic.zarr``: one store per hour
  from 2026-01-01, time in ``days since <that hour>``.

Transport needs no Google account and no client library: chunks are read over
anonymous HTTPS from ``storage.googleapis.com`` and decoded with the standard
library's zstd (Python 3.14; ``numcodecs`` on older Pythons).  A
``gs://satweave_1`` root is read through the same public HTTPS endpoint.  A
local directory holding the same layout is also a root
(``WOOF_SATWEAVE_ROOT`` or ``root=``), which is how the tests run.

NaN rules (SatWeave's own): beyond the geostationary disc, for bands a
sector's satellite does not carry, and for reflectances at night.  Values
outside a band's published ``valid_range`` are dropped to NaN and counted.

No SatWeave data is committed or shipped.  The cache and every scene carry
the CC BY 4.0 attribution in their provenance.  Nothing on a forecast path
imports this module.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time as _time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from woof.obs.imager_scene import (BANDS, FLAG_GEOMETRY_GUESSED, FLAG_LIMB, FLAG_NIGHT,
                                   FLAG_NO_SATELLITE, FLAG_OVERLAP,
                                   HORIZON_ZENITH_DEG, LIMB_ZENITH_DEG, OVERLAP_ZENITH_DEG,
                                   ImagerScene, Platform, band, geostationary_view,
                                   solar_angles, utc)

PROVENANCE_SCHEMA = "woof.obs.satweave-provenance/v1"
HTTPS_ROOT = "https://storage.googleapis.com/satweave_1"
ARCHIVE_PATH = "zarr/2016_to_2025/mosaic.zarr"
HOURLY_TEMPLATE = "zarr/2026_to_present/{stamp}/mosaic.zarr"
ARCHIVE_START = datetime(2016, 1, 1, tzinfo=timezone.utc)
HOURLY_START = datetime(2026, 1, 1, tzinfo=timezone.utc)
ARCHIVE_STEP = timedelta(minutes=30)
HOURLY_STEP = timedelta(hours=1)
ROOT_ENV = "WOOF_SATWEAVE_ROOT"
CACHE_ENV = "WOOF_SATWEAVE_CACHE"
GRID_SHAPE = (1801, 3600)
LICENCE = "CC-BY-4.0"
ATTRIBUTION = (
    "SatWeave geostationary mosaic by Google DeepMind and Google Research, licensed CC BY 4.0 "
    "(https://developers.google.com/weathernext/guides/satweave). Contains modified EUMETSAT "
    "Meteosat data; NOAA GOES-R series data via Google Cloud Public Datasets; JMA Himawari-8/9 "
    "data via NOAA NODD on Google Cloud, modified by Google. Subset and reprojected by woof.")

#: The sectors' satellites, by era.  SatWeave publishes no per-pixel satellite;
#: these rows are what the reconstruction assumes.  They start on 2018-11-14,
#: the first date SatWeave states every sector is ABI, AHI or SEVIRI (before
#: it the legacy GOES-13/15 imagers carry five bands): earlier scenes read, but
#: refuse reconstruction.  ``end`` is exclusive; None is open.
#: Sources: NOAA NODD (GOES-19 GOES-East, planned 2025-04-04; GOES-18
#: GOES-West 2023-01-04), JMA (Himawari-9 primary 2022-12-13), EUMETSAT
#: (Meteosat-11 at 0 deg from 2018-02-20; Meteosat-9 IODC at 45.5 E from
#: 2022-06-01, after Meteosat-8 at 41.5 E).
_GOES19_NOTE = ("CRTM v2.4.1's coefficient release carries no GOES-19 file and the separately posted "
                "abi_g19 files do not work with it; GOES-16's ABI coefficients stand in (same ABI "
                "design) and the residual enters the sector bias")
ERA_START = datetime(2018, 11, 14, tzinfo=timezone.utc)


@dataclass(frozen=True)
class _Era:
    platform: Platform
    start: datetime
    end: datetime | None


def _d(text):
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


ERAS: tuple[_Era, ...] = (
    _Era(Platform("GOES-16", "goes_east", "abi", "abi_g16", -75.2), ERA_START, _d("2025-04-04")),
    _Era(Platform("GOES-19", "goes_east", "abi", "abi_g16", -75.2, _GOES19_NOTE), _d("2025-04-04"), None),
    _Era(Platform("GOES-17", "goes_west", "abi", "abi_g17", -137.2), ERA_START, _d("2023-01-04")),
    _Era(Platform("GOES-18", "goes_west", "abi", "abi_g18", -137.2), _d("2023-01-04"), None),
    _Era(Platform("Himawari-8", "himawari", "ahi", "ahi_himawari8", 140.7), ERA_START, _d("2022-12-13")),
    _Era(Platform("Himawari-9", "himawari", "ahi", "ahi_himawari9", 140.7), _d("2022-12-13"), None),
    _Era(Platform("Meteosat-11", "meteosat_0", "seviri", "seviri_m11", 0.0), ERA_START, None),
    _Era(Platform("Meteosat-8", "meteosat_iodc", "seviri", "seviri_m08", 41.5), ERA_START, _d("2022-06-01")),
    _Era(Platform("Meteosat-9", "meteosat_iodc", "seviri", "seviri_m09", 45.5), _d("2022-06-01"), None),
)


class SatWeaveError(RuntimeError):
    """A request SatWeave cannot answer as asked; the message names why."""


def platforms_at(when) -> tuple[Platform, ...]:
    """The five sectors' satellites at ``when``, or a refusal outside the table."""

    when = utc(when)
    if when < ERA_START:
        raise SatWeaveError(
            f"{when:%Y-%m-%dT%H:%MZ} is before {ERA_START:%Y-%m-%d}: SatWeave's sectors then include "
            "the legacy GOES-13/15 imagers, whose satellites and bands this reconstruction does not model")
    out = tuple(e.platform for e in ERAS if e.start <= when and (e.end is None or when < e.end))
    if len(out) != 5:
        raise SatWeaveError(f"the satellite table has {len(out)} rows at {when.isoformat()}, not five")
    return out


# --------------------------------------------------------------------------- transport
def _zstd_decompress(data: bytes) -> bytes:
    try:
        from compression import zstd  # Python 3.14
        return zstd.decompress(data)
    except ImportError:
        pass
    try:
        from numcodecs import Zstd
        return bytes(Zstd().decode(data))
    except ImportError:
        raise SatWeaveError("no zstd decoder: SatWeave chunks are zstd-compressed; use Python 3.14 "
                            "or install numcodecs") from None


class _Transport:
    """Byte reads relative to one root: an HTTPS URL, a gs:// URL, or a directory.

    A ``gs://bucket/prefix`` root is read anonymously through the bucket's public
    HTTPS endpoint, so no Google client library is needed.
    """

    def __init__(self, root: str):
        self.root = root.rstrip("/")
        if self.root.startswith("gs://"):
            self.root = "https://storage.googleapis.com/" + self.root[len("gs://"):]
        if re.match(r"https?://", self.root):
            self.kind = "https"
        else:
            self.kind = "directory"

    def get(self, key: str) -> bytes | None:
        """The object's bytes, or None when it does not exist."""

        if self.kind == "directory":
            path = Path(self.root) / key
            return path.read_bytes() if path.is_file() else None
        url = f"{self.root}/{key}"
        for attempt in range(4):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "woof-satweave"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    return response.read()
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    return None
                if error.code < 500 and error.code != 429 or attempt == 3:
                    raise SatWeaveError(f"GET {url}: HTTP {error.code}") from error
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                if attempt == 3:
                    raise SatWeaveError(f"GET {url}: {error}") from error
            _time.sleep(2.0 * (attempt + 1))
        return None


def default_root() -> str:
    return os.environ.get(ROOT_ENV) or HTTPS_ROOT


def cache_root() -> Path:
    """``$WOOF_SATWEAVE_CACHE`` or ``~/.woof/cache/satweave``."""

    override = os.environ.get(CACHE_ENV)
    return Path(override).expanduser() if override else Path.home() / ".woof" / "cache" / "satweave"


# --------------------------------------------------------------------------- stores
def store_for(when) -> tuple[str, int | None]:
    """The store path holding ``when`` and, for the archive, the time index estimate.

    Archive times are on the half hour; hourly times on the hour.  Anything
    else is refused (``nearest_step`` rounds first, when the caller wants that).
    """

    when = utc(when)
    if when < ARCHIVE_START:
        raise SatWeaveError(f"{when.isoformat()} is before SatWeave's first step, {ARCHIVE_START:%Y-%m-%d}")
    if when >= HOURLY_START:
        if when != when.replace(minute=0, second=0, microsecond=0):
            raise SatWeaveError(f"{when.isoformat()}: the 2026-onward stores are hourly; ask for a whole hour")
        return HOURLY_TEMPLATE.format(stamp=when.strftime("%Y%m%dT%H%M%S")), None
    offset = when - ARCHIVE_START
    if offset % ARCHIVE_STEP:
        raise SatWeaveError(f"{when.isoformat()}: the 2016-2025 archive is half-hourly; ask for :00 or :30")
    return ARCHIVE_PATH, int(offset // ARCHIVE_STEP)


def nearest_step(when) -> datetime:
    """``when`` rounded to the nearest step of the store that covers it."""

    when = utc(when)
    step = HOURLY_STEP if when >= HOURLY_START - ARCHIVE_STEP / 2 else ARCHIVE_STEP
    base = HOURLY_START if step == HOURLY_STEP else ARCHIVE_START
    n = round((when - base) / step)
    return base + n * step


_UNITS = re.compile(r"^\s*(seconds|minutes|hours|days)\s+since\s+(.+?)\s*$")


def decode_cf_times(values, units: str) -> list[datetime]:
    """CF ``<unit> since <epoch>`` integers to UTC datetimes (proleptic Gregorian)."""

    match = _UNITS.match(units)
    if match is None:
        raise SatWeaveError(f"time units {units!r} are not '<unit> since <epoch>'")
    epoch = utc(match.group(2).replace(" ", "T", 1))
    scale = {"seconds": 1, "minutes": 60, "hours": 3600, "days": 86400}[match.group(1)]
    return [epoch + timedelta(seconds=float(v) * scale) for v in np.asarray(values).ravel()]


class _Store:
    def __init__(self, transport: _Transport, path: str):
        self.transport, self.path = transport, path
        raw = transport.get(f"{path}/zarr.json")
        if raw is None:
            raise SatWeaveError(f"no SatWeave store at {transport.root}/{path} (not yet published, or "
                                "outside the archive)")
        meta = json.loads(raw)
        self.metadata_sha256 = hashlib.sha256(raw).hexdigest()
        self.attributes = meta.get("attributes", {})
        self.arrays = (meta.get("consolidated_metadata") or {}).get("metadata") or {}
        if not self.arrays:
            raise SatWeaveError(f"{path}/zarr.json has no consolidated metadata")

    def _array(self, name):
        try:
            return self.arrays[name]
        except KeyError:
            raise SatWeaveError(f"{self.path}: no array {name}; it has {sorted(self.arrays)}") from None

    def _check_codecs(self, name, meta):
        codecs = [c.get("name") for c in meta.get("codecs", [])]
        endian = (meta.get("codecs") or [{}])[0].get("configuration", {}).get("endian")
        if codecs not in (["bytes", "zstd"], ["bytes"]) or endian not in ("little", None):
            raise SatWeaveError(f"{self.path}/{name}: codecs {meta.get('codecs')} are not the "
                                "bytes(little)+zstd this reader decodes")
        sep = meta.get("chunk_key_encoding", {}).get("configuration", {}).get("separator", "/")
        if meta.get("chunk_key_encoding", {}).get("name", "default") != "default" or sep != "/":
            raise SatWeaveError(f"{self.path}/{name}: chunk key encoding {meta.get('chunk_key_encoding')}")
        return "zstd" in codecs

    def chunk(self, name, index: tuple[int, ...], dtype, shape) -> tuple[np.ndarray, str]:
        meta = self._array(name)
        compressed = self._check_codecs(name, meta)
        key = f"{self.path}/{name}/c/" + "/".join(str(i) for i in index)
        raw = self.transport.get(key)
        if raw is None:
            fill = meta.get("fill_value")
            value = np.nan if fill in ("NaN", None) else fill
            return np.full(shape, value, dtype=dtype), "absent"
        data = _zstd_decompress(raw) if compressed else raw
        array = np.frombuffer(data, dtype=np.dtype(dtype).newbyteorder("<"))
        if array.size != int(np.prod(shape)):
            raise SatWeaveError(f"{key}: {array.size} values, expected {int(np.prod(shape))}")
        return array.reshape(shape), hashlib.sha256(raw).hexdigest()

    def coordinate(self, name) -> np.ndarray:
        meta = self._array(name)
        n = int(meta["shape"][0])
        if int(meta["chunk_grid"]["configuration"]["chunk_shape"][0]) != n:
            raise SatWeaveError(f"{self.path}/{name}: coordinate split across chunks")
        values, _ = self.chunk(name, (0,), meta["data_type"], (n,))
        return np.asarray(values, dtype=float)

    def time_at(self, index: int) -> datetime:
        meta = self._array("time")
        size = int(meta["chunk_grid"]["configuration"]["chunk_shape"][0])
        n = int(meta["shape"][0])
        if not 0 <= index < n:
            raise SatWeaveError(f"{self.path}: time index {index} outside 0..{n - 1}")
        block = index // size
        length = min(size, n - block * size)
        values, _ = self.chunk("time", (block,), meta["data_type"], (length,))
        return decode_cf_times([values[index - block * size]], meta["attributes"]["units"])[0]

    def band_array(self, name) -> dict:
        meta = self._array(name)
        if meta.get("data_type") != "float32" or tuple(meta["shape"][1:]) != GRID_SHAPE or \
                tuple(meta["chunk_grid"]["configuration"]["chunk_shape"]) != (1,) + GRID_SHAPE:
            raise SatWeaveError(f"{self.path}/{name}: shape {meta.get('shape')}, chunks "
                                f"{meta['chunk_grid']}, type {meta.get('data_type')} are not the "
                                "published (time, 1801, 3600) float32 layout")
        return meta


# --------------------------------------------------------------------------- subsetting
def _bbox_indices(lat, lon, bbox):
    """Row and column index arrays for ``bbox = (west, south, east, north)``.

    ``west > east`` crosses the antimeridian.  Columns come back in west-to-east
    order and their longitudes are made continuous from ``west``.
    """

    if bbox is None:
        rows = np.arange(lat.size)
        cols = np.arange(lon.size)
        return rows, cols, lon.copy()
    west, south, east, north = (float(v) for v in bbox)
    if not (-90 <= south < north <= 90):
        raise SatWeaveError(f"bbox south {south} / north {north} is not a latitude band")
    rows = np.flatnonzero((lat >= south - 1e-9) & (lat <= north + 1e-9))
    span = (east - west) % 360.0 or 360.0
    rel = (lon - west) % 360.0
    cols = np.flatnonzero(rel <= span + 1e-9)
    cols = cols[np.argsort(rel[cols], kind="stable")]
    if rows.size == 0 or cols.size == 0:
        raise SatWeaveError(f"bbox {bbox} selects no 0.1 degree cell")
    return rows, cols, west + rel[cols]


def _cache_key(store_path, when, bands, bbox) -> str:
    text = json.dumps(dict(store=store_path, time=when.isoformat(), bands=sorted(bands),
                           bbox=None if bbox is None else [round(float(v), 4) for v in bbox]),
                      sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:20]


@dataclass
class SatWeaveSubset:
    """The values read for one time, before geometry reconstruction."""

    time: datetime
    lat: np.ndarray            # (ny,) north to south
    lon: np.ndarray            # (nx,) continuous from the box's west edge
    values: dict               # nm -> (ny, nx) float32
    provenance: dict


def read_subset(when, *, bands: Iterable, bbox=None, root: str | None = None,
                cache: Path | str | None = None, refresh: bool = False,
                use_cache: bool = True) -> SatWeaveSubset:
    """Read ``bands`` at ``when`` inside ``bbox = (west, south, east, north)``.

    The subset is cached under ``cache`` (default :func:`cache_root`) as
    ``<stamp>/<key>/satweave.nc`` with ``provenance.json`` beside it; a later
    identical request reads the cache unless ``refresh``.
    """

    when = utc(when)
    nms = sorted({band(b).wavelength_nm for b in bands})
    if not nms:
        raise SatWeaveError("no bands requested")
    store_path, index = store_for(when)
    cache_dir = None
    if use_cache:
        cache_dir = Path(cache) if cache is not None else cache_root()
        cache_dir = cache_dir / when.strftime("%Y%m%dT%H%M%S") / _cache_key(store_path, when, nms, bbox)
        if not refresh and (cache_dir / "provenance.json").is_file() and (cache_dir / "satweave.nc").is_file():
            return _read_cached(cache_dir)
    transport = _Transport(root or default_root())
    store = _Store(transport, store_path)
    if index is None:
        index = 0
        n = int(store._array("time")["shape"][0])
        if n != 1:
            raise SatWeaveError(f"{store_path}: an hourly store with {n} steps, not one")
    stamp = store.time_at(index)
    if stamp != when:
        raise SatWeaveError(f"{store_path}: step {index} is {stamp.isoformat()}, not {when.isoformat()}")
    lat = store.coordinate("latitude")
    lon = store.coordinate("longitude")
    if lat.shape != (GRID_SHAPE[0],) or lon.shape != (GRID_SHAPE[1],) or not lat[0] > lat[-1]:
        raise SatWeaveError(f"{store_path}: latitude/longitude are not the published north-first 0.1 degree grid")
    rows, cols, lon_out = _bbox_indices(lat, lon, bbox)
    values, chunks, dropped, attrs = {}, {}, {}, {}
    for nm in nms:
        name = band(nm).name
        meta = store.band_array(name)
        full, digest = store.chunk(name, (index, 0, 0), "float32", GRID_SHAPE)
        sub = np.array(full[np.ix_(rows, cols)], dtype=np.float32)
        lo, hi = meta.get("attributes", {}).get("valid_range", (-np.inf, np.inf))
        with np.errstate(invalid="ignore"):
            bad = np.isfinite(sub) & ((sub < lo) | (sub > hi))
        sub[bad] = np.nan
        values[nm], chunks[name], dropped[str(nm)] = sub, digest, int(bad.sum())
        attrs[str(nm)] = {k: meta.get("attributes", {}).get(k)
                          for k in ("units", "satellite_sources", "valid_range", "long_name")}
    provenance = dict(
        schema=PROVENANCE_SCHEMA, source="SatWeave", licence=LICENCE, attribution=ATTRIBUTION,
        guide="https://developers.google.com/weathernext/guides/satweave",
        root=transport.root, transport=transport.kind, store=store_path, time_index=index,
        time=when.isoformat(), bbox=None if bbox is None else [float(v) for v in bbox],
        bands=nms, store_metadata_sha256=store.metadata_sha256, chunk_sha256=chunks,
        dropped_outside_valid_range=dropped, band_attributes=attrs,
        store_attributes={k: store.attributes.get(k) for k in ("title", "institution", "source")},
        read_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    subset = SatWeaveSubset(when, lat[rows], lon_out, values, provenance)
    if cache_dir is not None:
        _write_cache(cache_dir, subset)
    return subset


def _write_cache(directory: Path, subset: SatWeaveSubset) -> None:
    from netCDF4 import Dataset

    directory.mkdir(parents=True, exist_ok=True)
    tmp = directory / "satweave.nc.partial"
    with Dataset(tmp, "w") as ds:
        ds.createDimension("latitude", subset.lat.size)
        ds.createDimension("longitude", subset.lon.size)
        ds.createVariable("latitude", "f8", ("latitude",))[:] = subset.lat
        ds.createVariable("longitude", "f8", ("longitude",))[:] = subset.lon
        for nm, value in subset.values.items():
            var = ds.createVariable(band(nm).name, "f4", ("latitude", "longitude"), zlib=True,
                                    fill_value=np.float32(np.nan))
            var.units = band(nm).units
            var[:] = value
        ds.time = subset.time.isoformat()
        ds.licence = LICENCE
        ds.attribution = ATTRIBUTION
    tmp.replace(directory / "satweave.nc")
    provenance = dict(subset.provenance, cache_sha256=hashlib.sha256(
        (directory / "satweave.nc").read_bytes()).hexdigest())
    (directory / "provenance.json.partial").write_text(json.dumps(provenance, indent=2) + "\n")
    (directory / "provenance.json.partial").replace(directory / "provenance.json")
    subset.provenance = provenance


def _read_cached(directory: Path) -> SatWeaveSubset:
    from netCDF4 import Dataset

    provenance = json.loads((directory / "provenance.json").read_text())
    raw = (directory / "satweave.nc").read_bytes()
    if provenance.get("cache_sha256") != hashlib.sha256(raw).hexdigest():
        raise SatWeaveError(f"{directory}: the cached subset does not match its provenance digest; "
                            "delete the directory or re-read with refresh")
    with Dataset(directory / "satweave.nc") as ds:
        lat = np.array(ds["latitude"][:], dtype=float)
        lon = np.array(ds["longitude"][:], dtype=float)
        values = {nm: np.array(ds[band(nm).name][:].filled(np.nan), dtype=np.float32)
                  for nm in provenance["bands"]}
    return SatWeaveSubset(utc(provenance["time"]), lat, lon, values, dict(provenance, cached=True))


# --------------------------------------------------------------------------- reconstruction
def _check_sources(provenance: dict) -> None:
    """Refuse a store whose band attributes no longer name the instruments assumed here."""

    for nm, attrs in (provenance.get("band_attributes") or {}).items():
        text = (attrs or {}).get("satellite_sources") or ""
        if not text:
            continue
        if "FCI" in text and "SEVIRI" not in text:
            raise SatWeaveError(f"band {nm}: SatWeave now names FCI for Meteosat ({text!r}); "
                                "the satellite table models SEVIRI and must be extended")


def reconstruct(subset: SatWeaveSubset) -> ImagerScene:
    """Attribute each pixel to a satellite and compute its view and solar geometry.

    Band availability first: a finite 6900 nm value means GOES or Himawari, a
    finite 6200 nm value with a NaN 6900 nm one means Meteosat.  Then the
    smallest view zenith among the candidates.  Every pixel gets
    ``FLAG_GEOMETRY_GUESSED``; ``FLAG_OVERLAP`` where a second satellite sees
    it under 70 degrees.  Without 6200 and 6900 in the subset, only geometry
    decides.
    """

    _check_sources(subset.provenance)
    platforms = platforms_at(subset.time)
    lon2, lat2 = np.meshgrid(subset.lon, subset.lat)
    shape = lat2.shape
    zen, azi, scan = [], [], []
    for p in platforms:
        z, a, s = geostationary_view(lat2, lon2, p.sub_longitude_deg)
        zen.append(z), azi.append(a), scan.append(s)
    zen, azi, scan = np.stack(zen), np.stack(azi), np.stack(scan)
    visible = zen < HORIZON_ZENITH_DEG
    candidates = visible.copy()
    v62, v69 = subset.values.get(6200), subset.values.get(6900)
    if v69 is not None:
        not_meteo = np.array([p.family != "seviri" for p in platforms])[:, None, None]
        has69 = np.isfinite(v69)[None]
        candidates &= np.where(has69, not_meteo, True)
        if v62 is not None:
            meteo_only = (np.isfinite(v62) & ~np.isfinite(v69))[None]
            candidates &= np.where(meteo_only, ~not_meteo, True)
    masked = np.where(candidates, zen, np.inf)
    index = np.argmin(masked, axis=0)
    attributed = np.isfinite(np.take_along_axis(masked, index[None], 0)[0])
    index = np.where(attributed, index, -1).astype(np.int16)
    safe = np.maximum(index, 0)[None]
    pick = lambda a: np.where(attributed, np.take_along_axis(a, safe, 0)[0], np.nan).astype(np.float32)
    vza, vaz, sca = pick(zen), pick(azi), pick(scan)
    others = (zen < OVERLAP_ZENITH_DEG) & (np.arange(len(platforms))[:, None, None] != index[None])
    flags = np.full(shape, FLAG_GEOMETRY_GUESSED, dtype=np.uint16)
    flags |= np.where(attributed & others.any(axis=0), FLAG_OVERLAP, 0).astype(np.uint16)
    flags |= np.where(attributed & (vza > LIMB_ZENITH_DEG), FLAG_LIMB, 0).astype(np.uint16)
    flags |= np.where(~attributed, FLAG_NO_SATELLITE, 0).astype(np.uint16)
    # A finite value the attributed satellite does not carry is a contradiction.
    for nm, value in subset.values.items():
        carriers = np.array([p.family in BANDS[nm].channels for p in platforms] + [False])
        carried = carriers[np.where(attributed, index, len(platforms))]
        bad = np.isfinite(value) & ~carried
        flags |= np.where(bad, FLAG_NO_SATELLITE, 0).astype(np.uint16)
    sza, saz = solar_angles(subset.time, lat2, lon2)
    flags |= np.where(sza > 90.0, FLAG_NIGHT, 0).astype(np.uint16)
    provenance = dict(subset.provenance, reconstruction=dict(
        method="band availability (6900 nm: GOES/Himawari; 6200 nm without 6900 nm: Meteosat), "
               "then the smallest view zenith; nominal sub-satellite longitudes",
        overlap_zenith_deg=OVERLAP_ZENITH_DEG, horizon_zenith_deg=HORIZON_ZENITH_DEG,
        used_bands=[nm for nm in (6200, 6900) if nm in subset.values]))
    return ImagerScene(time=subset.time, lat=lat2, lon=lon2, values=dict(subset.values),
                       platforms=platforms, platform_index=index, view_zenith=vza, view_azimuth=vaz,
                       scan_angle=sca, solar_zenith=sza.astype(np.float32),
                       solar_azimuth=saz.astype(np.float32), flags=flags, provenance=provenance)


def read_scene(when, *, bands: Iterable, bbox=None, root: str | None = None,
               cache: Path | str | None = None, refresh: bool = False,
               use_cache: bool = True, identify: bool = True) -> ImagerScene:
    """:func:`read_subset` then :func:`reconstruct`.

    ``identify`` adds 6200 and 6900 nm to the read (two more chunks) so the
    satellite can be attributed from band availability; the scene's
    ``values`` still hold only the requested bands.
    """

    wanted = {band(b).wavelength_nm for b in bands}
    read = wanted | ({6200, 6900} if identify else set())
    subset = read_subset(when, bands=read, bbox=bbox, root=root, cache=cache,
                         refresh=refresh, use_cache=use_cache)
    scene = reconstruct(subset)
    scene.values = {nm: v for nm, v in scene.values.items() if nm in wanted}
    return scene


# --------------------------------------------------------------------------- command line
def _bands_argument(value: str) -> list[int]:
    import argparse

    try:
        return [band(part.strip()).wavelength_nm for part in value.split(",") if part.strip()]
    except KeyError as error:
        raise argparse.ArgumentTypeError(str(error.args[0])) from None


def _times(args) -> list[datetime]:
    start = args.time
    end = args.end or start
    if end < start:
        raise SatWeaveError("--end is before --time")
    out, when = [], nearest_step(start) if args.nearest else start
    while when <= end:
        out.append(when)
        step = HOURLY_STEP if when >= HOURLY_START else ARCHIVE_STEP
        when = when + max(step, timedelta(minutes=args.every_minutes or 0))
    return out


def fetch_main(args) -> int:
    """``woof obs satweave fetch``: cache subsets and print one summary line per time."""

    import sys

    try:
        summaries = []
        for when in _times(args):
            scene = read_scene(when, bands=args.bands, bbox=args.bbox, root=args.root,
                               cache=args.cache, refresh=args.refresh, identify=not args.no_identify)
            summary = scene.summary()
            summary["provenance"] = {k: scene.provenance.get(k) for k in
                                     ("store", "licence", "attribution", "cached", "transport")}
            summaries.append(summary)
            print(json.dumps({"time": summary["time"], "shape": summary["shape"],
                              "pixels_by_platform": summary["pixels_by_platform"],
                              "bands": {k: v["finite"] for k, v in summary["bands"].items()},
                              "cached": bool(scene.provenance.get("cached"))}))
    except SatWeaveError as error:
        print(f"woof obs satweave fetch: {error}", file=sys.stderr)
        return 2
    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"SatWeave data: {LICENCE}. {ATTRIBUTION}", file=sys.stderr)
    return 0


def register_cli(obs_sub) -> None:
    """``woof obs satweave fetch`` (``woof fetch satweave`` is the same door)."""

    from woof.obs.cli import _bbox_wrapping, _utc_time

    parser = obs_sub.add_parser(
        "satweave", help="SatWeave geostationary mosaic (Google; CC BY 4.0): subset and cache")
    sub = parser.add_subparsers(dest="satweave_command", required=True)
    fetch = sub.add_parser(
        "fetch", help="read bands for a time (or a span) and bounding box into the local cache")
    fetch.add_argument("--time", required=True, type=_utc_time,
                       help="UTC time with offset, e.g. 2026-11-18T12:00Z (archive: :00/:30; 2026 on: hourly)")
    fetch.add_argument("--end", default=None, type=_utc_time,
                       help="last time of a span (inclusive); every step from --time")
    fetch.add_argument("--every-minutes", type=float, default=None,
                       help="span step in minutes (default the store's own: 30 or 60)")
    fetch.add_argument("--nearest", action="store_true",
                       help="round --time to the nearest published step instead of refusing")
    fetch.add_argument("--bands", type=_bands_argument, default=[6200, 6900, 7330, 11200],
                       help="comma-separated wavelengths or names (default 6200,6900,7330,11200)")
    fetch.add_argument("--bbox", type=_bbox_wrapping, default=None,
                       help="west,south,east,north in degrees (west > east crosses 180); default the globe")
    fetch.add_argument("--root", default=None,
                       help=f"store root: an https:// or gs:// URL or a directory (default ${ROOT_ENV} or {HTTPS_ROOT})")
    fetch.add_argument("--cache", default=None, help=f"cache directory (default ${CACHE_ENV} or ~/.woof/cache/satweave)")
    fetch.add_argument("--refresh", action="store_true", help="re-read even when the cache holds the subset")
    fetch.add_argument("--no-identify", action="store_true",
                       help="do not read 6200/6900 nm to attribute satellites (geometry only)")
    fetch.add_argument("--summary-json", default=None, help="write every scene's summary here")
    fetch.set_defaults(func=fetch_main)
