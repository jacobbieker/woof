"""Station observations from the Dynamical.org ASOS Parquet archive.

The Iowa Environmental Mesonet's ASOS/METAR archive, as republished by
dynamical.org on Source Cooperative: one Parquet file per year
(``<base>/year=YYYY/data.parquet``, 1940 to the current year), rows sorted
by ``(station, valid)``, about 650 MB for a recent year.  It covers the US
and a dozen other countries (CA, AU, BR, CN, DE, FR, GB, IN, JP, KR, MX, NZ,
RU, ZA), which is what lets a forecast outside the IEM network front door's
reach be scored against surface reports.

:func:`fetch_surface` writes the same two records the ``rw_asos`` front door
writes -- ``stations.json`` (``gpuwm-obs.asos-stations.v1``) and
``surface.json`` (``gpuwm-obs.asos-surface.v2``) -- so the Rust scorer
(``rw_verify``) and :class:`woof.obs.sources.AsosSurfaceSource` read them
unchanged.  The decode rules are ``rw_asos decode``'s, applied to the
archive's own SI-adjacent columns:

* one report per station, the one nearest the valid time within
  :data:`MATCH_SECONDS` (ties to the earlier report);
* ``temperature_2m = tmpc + 273.15``, ``dewpoint_2m = dwpc + 273.15``,
  ``wind_speed_10m = sknt * 1852/3600``, ``mslp = mslp * 100`` (Pa); a null
  is an absent key, never a NaN;
* the gross-error screen: a report whose temperature leaves
  233.15..328.15 K, whose wind leaves 0..75 m/s, or whose dewpoint exceeds
  its temperature is dropped, and a station whose dropped share of the
  reports in the screening window (valid time +/- 1 h, the window the
  ``rw_asos`` route fetches) exceeds :data:`MAX_SCREEN_RATE` is dropped
  whole;
* ``p01i``/``p01m`` are never read: missing and zero precipitation are the
  same value in this archive.

Access is by HTTP range request: only the footer, the row groups whose
``station`` statistics overlap the stations asked for, and the columns in
:data:`COLUMNS` are read.  Every byte range read is cached under
``~/.woof/cache/dynamical-asos/`` (``$WOOF_DYNAMICAL_ASOS_CACHE``), keyed by
the file's URL and ETag, so asking again for the same file version moves no
data.  The current year's file is rewritten twice an hour; a cached version
is reused without asking the host only when it was seen after the hour in
question had settled, and ``refresh=True`` always revalidates.

Attribution: data Iowa Environmental Mesonet (Iowa State University);
original reports NOAA/NWS/FAA (public domain); processing dynamical.org;
hosting Source Cooperative.  The dataset is marked experimental.
"""

from __future__ import annotations

import bisect
from datetime import datetime, timedelta, timezone
import functools
from http.client import IncompleteRead
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request

SOURCE = "dynamical-asos-parquet"
PRODUCT = "iem-asos-metar-dynamical-parquet"
BASE_URL_ENV = "WOOF_DYNAMICAL_ASOS_URL"
DEFAULT_BASE_URL = "https://data.source.coop/dynamical/asos-parquet"
CACHE_ENV = "WOOF_DYNAMICAL_ASOS_CACHE"
USER_AGENT = "gpuwm-fetch/2.5 (+https://github.com/arwenweather)"
ATTRIBUTION = (
    "Data: Iowa Environmental Mesonet (Iowa State University); original "
    "reports NOAA/NWS/FAA (public domain); processing: dynamical.org; "
    "hosting: Source Cooperative. Dataset marked experimental by "
    "dynamical.org.")

STATIONS_SCHEMA = "gpuwm-obs.asos-stations.v1"
SURFACE_SCHEMA = "gpuwm-obs.asos-surface.v2"
STATION_TABLE_SCHEMA = "gpuwm-obs.dynamical-asos-stations.v1"

#: The columns a surface decode reads.  Everything else in the file
#: (``p01i``/``p01m``, the geometry, the text fields) is never fetched.
COLUMNS = ("station", "valid", "latitude", "longitude", "elevation", "name",
           "country", "tmpc", "dwpc", "sknt", "mslp")

#: ``rw_asos decode``'s defaults, so a record from this archive is scored
#: under the same rules as one from the IEM CGI.
MATCH_SECONDS = 600
SCREEN_WINDOW_SECONDS = 3600
MIN_REPORT_RATE = 0.80
MAX_SCREEN_RATE = 0.05
TEMPERATURE_MIN_K = 233.15
TEMPERATURE_MAX_K = 328.15
WIND_MIN_MS = 0.0
WIND_MAX_MS = 75.0
KNOTS_TO_MS = 0.5144444444444445

#: How long after the end of the screening window the archive is taken to
#: hold every report for it.  A cached file version seen later than that is
#: reused without asking the host; an earlier one is revalidated.
SETTLE = timedelta(hours=3)

_TABLE_PATH = Path(__file__).resolve().parent / "data" / "dynamical_asos_stations.json"
_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"


class DynamicalUnavailable(RuntimeError):
    """The archive cannot be read: pyarrow is missing, the host does not
    answer, or the file it served cannot be read as Parquet."""


class _VersionChanged(Exception):
    """The file was rewritten between two reads of it (HTTP 412)."""


class _OutOfTime(Exception):
    """The caller's time budget ran out."""


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------

def base_url() -> str:
    """The archive root, ``$WOOF_DYNAMICAL_ASOS_URL`` or the public one."""

    return (os.environ.get(BASE_URL_ENV) or DEFAULT_BASE_URL).rstrip("/")


def year_url(year: int, base: str | None = None) -> str:
    """The one Parquet file holding every report of ``year``."""

    return f"{(base or base_url()).rstrip('/')}/year={int(year)}/data.parquet"


def cache_root() -> Path:
    """Where fetched byte ranges live: ``$WOOF_DYNAMICAL_ASOS_CACHE`` or
    ``~/.woof/cache/dynamical-asos``."""

    override = os.environ.get(CACHE_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".woof" / "cache" / "dynamical-asos"


def _seam_time(when: datetime) -> str:
    return when.astimezone(timezone.utc).strftime(_TIME_FORMAT)


def _utc(when: datetime) -> datetime:
    if not isinstance(when, datetime):
        raise TypeError(f"valid_time must be a datetime, not {type(when).__name__}")
    if when.tzinfo is None:
        return when.replace(tzinfo=timezone.utc)
    return when.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# The frozen station table
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _station_table() -> tuple[dict, ...]:
    record = json.loads(_TABLE_PATH.read_text(encoding="utf-8"))
    if record.get("schema") != STATION_TABLE_SCHEMA:
        raise ValueError(f"{_TABLE_PATH} declares schema {record.get('schema')!r}, "
                         f"expected {STATION_TABLE_SCHEMA!r}")
    return tuple(record["stations"])


def _wrap(longitude: float) -> float:
    return ((float(longitude) + 180.0) % 360.0) - 180.0


def _bbox_test(west: float, south: float, east: float, north: float):
    """A ``(latitude, longitude) -> bool`` for a box; ``west > east`` crosses
    the antimeridian."""

    south, north = float(south), float(north)
    if south > north:
        raise ValueError(f"bbox south {south} is north of its north {north}")
    west, east = float(west), float(east)
    if east - west >= 360.0:
        return lambda lat, lon: south <= lat <= north
    # The box runs eastward from ``west`` for ``width`` degrees, whichever
    # convention its edges are written in: -8..2, 352..2 and 352..362 are
    # one box, and 170..-170 is the 20 degrees across the antimeridian.
    width = (east - west) % 360.0
    start = _wrap(west)
    return lambda lat, lon: (south <= lat <= north
                             and (_wrap(lon) - start) % 360.0 <= width)


def stations_in_bbox(west: float, south: float, east: float,
                     north: float) -> list[dict]:
    """The frozen table's stations inside the box, sorted by id.

    Keys ``station_id, name, latitude, longitude, elevation_m, country``.
    Reads only the packaged table: no network, no pyarrow.
    """

    inside = _bbox_test(west, south, east, north)
    return [dict(row) for row in _station_table()
            if inside(float(row["latitude"]), float(row["longitude"]))]


def station_count_in_bbox(west: float, south: float, east: float,
                          north: float) -> int:
    """How many archive stations the frozen table places inside the box."""

    inside = _bbox_test(west, south, east, north)
    return sum(1 for row in _station_table()
               if inside(float(row["latitude"]), float(row["longitude"])))


# ---------------------------------------------------------------------------
# HTTP range reads with an on-disk cache
# ---------------------------------------------------------------------------

class _Budget:
    """A wall-clock deadline every request and every retry wait respects."""

    def __init__(self, seconds: float | None):
        self.deadline = None if seconds is None else time.monotonic() + float(seconds)

    def remaining(self) -> float | None:
        if self.deadline is None:
            return None
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise _OutOfTime()
        return left

    def pause(self, seconds: float) -> None:
        left = self.remaining()
        if left is not None and seconds >= left:
            raise _OutOfTime()
        from woof import fetch_pool
        fetch_pool.sleep_unless_stopped(seconds, sleep=time.sleep)


class _Stats:
    """What one fetch moved: requests sent and bytes over the wire vs. from
    the cache.  :func:`last_fetch_stats` returns the latest."""

    def __init__(self):
        self.lock = threading.Lock()
        self.head_requests = 0
        self.range_requests = 0
        self.bytes_downloaded = 0
        self.bytes_from_cache = 0
        self.row_groups_total = 0
        self.row_groups_read = 0

    def as_dict(self) -> dict:
        return {"head_requests": self.head_requests,
                "range_requests": self.range_requests,
                "bytes_downloaded": self.bytes_downloaded,
                "bytes_from_cache": self.bytes_from_cache,
                "row_groups_total": self.row_groups_total,
                "row_groups_read": self.row_groups_read}


_LAST_STATS: dict = {}


def last_fetch_stats() -> dict:
    """Counters from the most recent :func:`fetch_surface` in this process."""

    return dict(_LAST_STATS)


def _endpoint(url: str):
    from woof.fetch_endpoints import Endpoint

    return Endpoint(name="source.coop", base=url, retention_hours=None,
                    why="the Dynamical.org ASOS Parquet archive")


def _ask(url: str, transfer, *, name: str, budget: _Budget):
    """One request under the tree's shared retry (``ask_along_ladder``)."""

    from woof.fetch_endpoints import ask_along_ladder

    _endpoint_used, result = ask_along_ladder(
        (_endpoint(url),), lambda _endpoint: transfer(), label="dynamical-asos",
        name=name, progress=_progress, pause=budget.pause)
    return result


def _open(request: Request, budget: _Budget, timeout: float):
    from woof.nomads_governor import paced_urlopen

    left = budget.remaining()
    per_request = timeout if left is None else max(0.1, min(timeout, left))
    return paced_urlopen(request, timeout=per_request)


class _Version:
    """One version (URL + ETag) of one year file, with its cached extents.

    Extents are the byte ranges already fetched, each its own file named
    ``<start>-<end>.bin`` under the version's directory.  A read takes what
    the extents cover and fetches only the gaps, exactly, so nothing is
    over-read and no byte is fetched twice for one version.
    """

    def __init__(self, url: str, etag: str, last_modified: str, size: int,
                 *, budget: _Budget, stats: _Stats, timeout: float):
        self.url = url
        self.etag = etag
        self.last_modified = last_modified
        self.size = int(size)
        self.budget = budget
        self.stats = stats
        self.timeout = timeout
        key = hashlib.sha256(f"{etag or last_modified}\n{size}".encode()).hexdigest()[:24]
        self.directory = _url_directory(url) / key
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.consumed: list[tuple[int, int]] = []
        self._extents: list[tuple[int, int]] = []
        for path in self.directory.glob("*.bin"):
            try:
                start, end = (int(part) for part in path.stem.split("-"))
            except ValueError:
                continue
            if 0 <= start < end <= self.size and path.stat().st_size == end - start:
                self._extents.append((start, end))
        self._extents.sort()
        meta = self.directory / "version.json"
        if not meta.is_file():
            from woof.fetch_guard import atomic_write_text
            atomic_write_text(meta, json.dumps(
                {"url": url, "etag": etag, "last_modified": last_modified,
                 "size": self.size}, indent=2) + "\n")

    def _extent_path(self, start: int, end: int) -> Path:
        return self.directory / f"{start:015d}-{end:015d}.bin"

    def _fetch(self, start: int, end: int) -> bytes:
        headers = {"User-Agent": USER_AGENT, "Range": f"bytes={start}-{end - 1}"}
        if self.etag:
            headers["If-Match"] = self.etag
        elif self.last_modified:
            headers["If-Unmodified-Since"] = self.last_modified

        def transfer() -> bytes:
            request = Request(self.url, headers=headers)
            try:
                with _open(request, self.budget, self.timeout) as response:
                    status = getattr(response, "status", None) or response.getcode()
                    if status != 206:
                        raise ValueError(
                            f"{self.url} answered a range request with HTTP {status}; "
                            "refusing to read the whole file")
                    served = response.headers.get("ETag")
                    if self.etag and served and served != self.etag:
                        raise _VersionChanged(self.url)
                    modified = response.headers.get("Last-Modified")
                    if (not self.etag and self.last_modified and modified
                            and modified != self.last_modified):
                        raise _VersionChanged(self.url)
                    data = response.read()
            except HTTPError as error:
                if error.code == 412:
                    raise _VersionChanged(self.url) from error
                raise
            if len(data) != end - start:
                raise IncompleteRead(data, end - start - len(data))
            return data

        with self.stats.lock:
            self.stats.range_requests += 1
        data = _ask(self.url, transfer, name=f"{self.url} bytes {start}-{end - 1}",
                    budget=self.budget)
        with self.stats.lock:
            self.stats.bytes_downloaded += len(data)
        from woof.fetch_guard import atomic_write_bytes
        # Another process may have pruned this version meanwhile.
        self.directory.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(self._extent_path(start, end), data)
        with self._lock:
            bisect.insort(self._extents, (start, end))
        return data

    def gaps(self, start: int, end: int) -> list[tuple[int, int]]:
        """The parts of ``[start, end)`` no cached extent covers."""

        end = min(end, self.size)
        with self._lock:
            extents = list(self._extents)
        missing = []
        position = start
        for low, high in extents:
            if high <= position:
                continue
            if low >= end:
                break
            if low > position:
                missing.append((position, low))
            position = max(position, high)
            if position >= end:
                break
        if position < end:
            missing.append((position, end))
        return missing

    def prefetch(self, ranges: list[tuple[int, int]], *, workers: int = 6) -> None:
        """Fetch the uncached parts of ``ranges`` concurrently.

        The host answers each request after most of a second however small
        it is, so the column chunks a decode will read are asked for side
        by side first; pyarrow's own reads then come from the cache.
        """

        wanted = [gap for start, end in ranges for gap in self.gaps(start, end)]
        if not wanted:
            return
        if len(wanted) == 1 or workers <= 1:
            for start, end in wanted:
                self._fetch(start, end)
            return
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(workers, len(wanted))) as pool:
            for future in [pool.submit(self._fetch, start, end) for start, end in wanted]:
                future.result()

    def _read_cached(self, start: int, end: int, extent: tuple[int, int]) -> bytes:
        with open(self._extent_path(*extent), "rb") as stream:
            stream.seek(start - extent[0])
            return stream.read(end - start)

    def read(self, offset: int, length: int, *, record: bool = True) -> bytes:
        end = min(self.size, offset + max(0, length))
        if offset >= end:
            return b""
        if record:
            with self._lock:
                self.consumed.append((offset, end))
        pieces = []
        position = offset
        while position < end:
            with self._lock:
                extents = list(self._extents)
            covering = None
            following = end
            for extent in extents:
                if extent[0] <= position < extent[1]:
                    if covering is None or extent[1] > covering[1]:
                        covering = extent
                elif extent[0] > position:
                    following = min(following, extent[0])
            if covering is not None:
                stop = min(covering[1], end)
                try:
                    pieces.append(self._read_cached(position, stop, covering))
                except FileNotFoundError:
                    # Pruned by another process: fetch the range again.
                    with self._lock:
                        if covering in self._extents:
                            self._extents.remove(covering)
                    continue
                if record:
                    with self.stats.lock:
                        self.stats.bytes_from_cache += stop - position
                position = stop
                continue
            stop = min(following, end)
            pieces.append(self._fetch(position, stop))
            position = stop
        return b"".join(pieces)

    def digest(self, hasher) -> None:
        """Feed every byte range pyarrow read, merged and in file order."""

        merged: list[list[int]] = []
        for start, end in sorted(self.consumed):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        hasher.update(f"{self.url}\t{self.etag or self.last_modified}\t{self.size}\n".encode())
        for start, end in merged:
            hasher.update(f"{start}-{end}\n".encode())
            hasher.update(self.read(start, end - start, record=False))


class _RangeFile(io.RawIOBase):
    """A seekable read-only file over one :class:`_Version`."""

    def __init__(self, version: _Version):
        super().__init__()
        self.version = version
        self.position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.position = offset
        elif whence == io.SEEK_CUR:
            self.position += offset
        elif whence == io.SEEK_END:
            self.position = self.version.size + offset
        else:
            raise ValueError(f"bad whence {whence}")
        if self.position < 0:
            raise ValueError("negative seek position")
        return self.position

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.version.size - self.position
        data = self.version.read(self.position, size)
        self.position += len(data)
        return data

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def size(self) -> int:
        return self.version.size


def _index_path(url: str) -> Path:
    return cache_root() / "index" / f"{hashlib.sha256(url.encode()).hexdigest()[:24]}.json"


def _url_directory(url: str) -> Path:
    return cache_root() / "files" / hashlib.sha256(url.encode()).hexdigest()[:24]


#: A superseded version's cached ranges are removed once nothing has
#: written to them for this long, so a reader still on it is not cut off.
STALE_VERSION_AGE_S = 3600.0


def _prune_superseded(url: str, current: Path) -> None:
    """Remove cached versions of ``url`` other than ``current``.

    The current year's file is rewritten twice an hour, and every version
    a request read would otherwise stay on disk for good.
    """

    import shutil

    now = time.time()
    try:
        siblings = [path for path in _url_directory(url).iterdir()
                    if path.is_dir() and path != current]
    except OSError:
        return
    for path in siblings:
        try:
            if now - path.stat().st_mtime > STALE_VERSION_AGE_S:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            continue


def _resolve(url: str, *, needed_after: datetime, refresh: bool, budget: _Budget,
             stats: _Stats, timeout: float) -> _Version:
    """The version of ``url`` to read: the cached one when it was seen after
    ``needed_after``, otherwise whatever the host serves now (one HEAD)."""

    index = _index_path(url)
    if not refresh:
        try:
            known = json.loads(index.read_text(encoding="utf-8"))
            seen = datetime.fromisoformat(known["checked_at"])
            if known.get("url") == url and seen >= needed_after:
                return _Version(url, known.get("etag", ""), known.get("last_modified", ""),
                                int(known["size"]), budget=budget, stats=stats,
                                timeout=timeout)
        except (OSError, ValueError, KeyError, TypeError):
            pass

    def transfer() -> tuple[str, str, int]:
        request = Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
        with _open(request, budget, timeout) as response:
            length = response.headers.get("Content-Length")
            if length is None:
                raise ValueError(f"{url} did not state its length")
            return (response.headers.get("ETag") or "",
                    response.headers.get("Last-Modified") or "", int(length))

    with stats.lock:
        stats.head_requests += 1
    etag, last_modified, size = _ask(url, transfer, name=url, budget=budget)
    if not etag and not last_modified:
        raise DynamicalUnavailable(f"{url} states neither an ETag nor a Last-Modified; "
                         "a cached copy could not be told from a rewritten one")
    from woof.fetch_guard import atomic_write_text
    index.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(index, json.dumps(
        {"url": url, "etag": etag, "last_modified": last_modified, "size": size,
         "checked_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
    version = _Version(url, etag, last_modified, size, budget=budget, stats=stats,
                       timeout=timeout)
    _prune_superseded(url, version.directory)
    return version


def _import_pyarrow():
    try:
        import pyarrow
        import pyarrow.compute
        import pyarrow.parquet
    except ImportError as error:
        raise DynamicalUnavailable(
            "reading the Dynamical.org ASOS archive needs pyarrow, which is not "
            "installed: pip install 'recast-woof[obs]'") from error
    return pyarrow


def _overlapping_row_groups(metadata, wanted: list[str]) -> list[int]:
    """Row groups whose ``station`` min/max could hold one of ``wanted``.

    A row group without statistics is kept: pruning is an optimisation and
    must never lose a report.
    """

    column = metadata.schema.names.index("station")
    keep = []
    for index in range(metadata.num_row_groups):
        stats = metadata.row_group(index).column(column).statistics
        if stats is None or not stats.has_min_max:
            keep.append(index)
            continue
        low, high = stats.min, stats.max
        if isinstance(low, bytes):
            low, high = low.decode("utf-8", "replace"), high.decode("utf-8", "replace")
        at = bisect.bisect_left(wanted, low)
        if at < len(wanted) and wanted[at] <= high:
            keep.append(index)
    return keep


def _chunk_ranges(metadata, groups: list[int], columns, *,
                  hole: int = 64 * 1024) -> list[tuple[int, int]]:
    """The byte ranges of ``columns``' chunks in ``groups``, with ranges
    closer than ``hole`` bytes merged into one request."""

    wanted = set(columns)
    ranges = []
    for group in groups:
        row_group = metadata.row_group(group)
        for index in range(row_group.num_columns):
            chunk = row_group.column(index)
            if chunk.path_in_schema not in wanted:
                continue
            start = chunk.data_page_offset
            if chunk.has_dictionary_page and chunk.dictionary_page_offset is not None:
                start = min(start, chunk.dictionary_page_offset)
            ranges.append((start, start + chunk.total_compressed_size))
    merged: list[list[int]] = []
    for start, end in sorted(ranges):
        if merged and start - merged[-1][1] <= hole:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _read_rows(version: _Version, wanted: list[str], start: datetime, end: datetime,
               stats: _Stats) -> list[dict]:
    """Every row of the wanted stations with ``start <= valid <= end``."""

    pa = _import_pyarrow()
    pc = pa.compute
    pq = pa.parquet
    try:
        handle = pq.ParquetFile(_RangeFile(version), pre_buffer=True)
        names = handle.schema_arrow.names
        missing = [name for name in COLUMNS if name not in names]
        if missing:
            raise DynamicalUnavailable(
                f"{version.url} has no column(s) {missing}; the archive layout changed")
        groups = _overlapping_row_groups(handle.metadata, wanted)
        with stats.lock:
            stats.row_groups_total += handle.metadata.num_row_groups
            stats.row_groups_read += len(groups)
        version.prefetch(_chunk_ranges(handle.metadata, groups, COLUMNS))
        wanted_set = pa.array(wanted, type=pa.large_string())
        unit = handle.schema_arrow.field("valid").type
        low = pa.scalar(start, type=unit)
        high = pa.scalar(end, type=unit)
        rows: list[dict] = []
        for group in groups:
            table = handle.read_row_group(group, columns=list(COLUMNS), use_threads=False)
            station = table.column("station").cast(pa.large_string())
            mask = pc.and_(pc.is_in(station, value_set=wanted_set),
                           pc.and_(pc.greater_equal(table.column("valid"), low),
                                   pc.less_equal(table.column("valid"), high)))
            rows.extend(table.filter(mask).to_pylist())
        return rows
    except (_VersionChanged, _OutOfTime, DynamicalUnavailable):
        raise
    except Exception as error:
        from woof.fetch_endpoints import TransferRefusal
        if isinstance(error, TransferRefusal):
            raise
        if isinstance(error, pa.ArrowException) or isinstance(error, OSError):
            cause = error.__cause__ or error.__context__
            if isinstance(cause, (_VersionChanged, _OutOfTime)):
                raise cause from None
            if cause is not None and type(cause).__name__ == "TransferRefusal":
                raise cause from None
            raise DynamicalUnavailable(f"{version.url} could not be read as Parquet: {error}") from error
        raise


# ---------------------------------------------------------------------------
# Decode: rw_asos decode's rules on the archive's columns
# ---------------------------------------------------------------------------

def _number(value) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _decode(rows: list[dict], selected: list[str], target: datetime) -> tuple[list, dict, list]:
    """Screen, match and complete; returns ``(reports, screen, kept_ids)``."""

    known = set(selected)
    by_station: dict[str, list[dict]] = {}
    screened: dict[str, list[int]] = {}
    dewpoint_drops = 0
    range_drops = 0
    for row in rows:
        station = row["station"]
        if station not in known or row.get("valid") is None:
            continue
        entry = screened.setdefault(station, [0, 0])
        entry[1] += 1
        temperature = _number(row.get("tmpc"))
        dewpoint = _number(row.get("dwpc"))
        knots = _number(row.get("sknt"))
        fired = False
        if temperature is not None and not (
                TEMPERATURE_MIN_K <= temperature + 273.15 <= TEMPERATURE_MAX_K):
            fired = True
            range_drops += 1
        if temperature is not None and dewpoint is not None and dewpoint > temperature:
            fired = True
            dewpoint_drops += 1
        if knots is not None and not (WIND_MIN_MS <= knots * KNOTS_TO_MS <= WIND_MAX_MS):
            fired = True
            range_drops += 1
        if fired:
            entry[0] += 1
            continue
        by_station.setdefault(station, []).append(row)

    dropped_by_screen = []
    for station in sorted(screened):
        fired, seen = screened[station]
        if seen > 0 and fired / seen > MAX_SCREEN_RATE:
            dropped_by_screen.append(station)
            by_station.pop(station, None)

    valid_times = [target]
    reports = []
    matched: dict[str, int] = {}
    for station in sorted(by_station):
        best = None
        for row in by_station[station]:
            when = _utc(row["valid"])
            distance = abs((when - target).total_seconds())
            if distance > MATCH_SECONDS:
                continue
            if best is None or (distance, when) < (best[0], best[1]):
                best = (distance, when, row)
        if best is None:
            continue
        _distance, when, row = best
        values = {}
        temperature = _number(row.get("tmpc"))
        dewpoint = _number(row.get("dwpc"))
        knots = _number(row.get("sknt"))
        pressure = _number(row.get("mslp"))
        if temperature is not None:
            values["temperature_2m"] = temperature + 273.15
        if dewpoint is not None:
            values["dewpoint_2m"] = dewpoint + 273.15
        if knots is not None:
            values["wind_speed_10m"] = knots * KNOTS_TO_MS
        if pressure is not None:
            values["mslp"] = pressure * 100.0
        if not values:
            continue
        matched[station] = matched.get(station, 0) + 1
        reports.append({"station_id": station, "valid_time": _seam_time(target),
                        "observation_time": _seam_time(when),
                        "values": dict(sorted(values.items())), "flags": []})

    required = math.ceil(MIN_REPORT_RATE * len(valid_times))
    kept = []
    dropped_by_completeness = []
    for station in selected:
        count = matched.get(station, 0)
        if count >= required and count > 0:
            kept.append(station)
        elif station in matched:
            dropped_by_completeness.append(station)
    keep = set(kept)
    reports = sorted((r for r in reports if r["station_id"] in keep),
                     key=lambda r: (r["station_id"], r["valid_time"]))
    screen = {
        "temperature_min_k": TEMPERATURE_MIN_K,
        "temperature_max_k": TEMPERATURE_MAX_K,
        "wind_min_ms": WIND_MIN_MS,
        "wind_max_ms": WIND_MAX_MS,
        "dewpoint_above_temperature_drops": dewpoint_drops,
        "range_drops": range_drops,
        "stations_dropped_by_screen": dropped_by_screen,
        "stations_dropped_by_completeness": dropped_by_completeness,
    }
    return reports, screen, kept


def station_rows_digest(stations: list[dict]) -> str:
    """``rw_asos``'s station-table digest: the rows, canonically spelled."""

    text = "".join(
        f"{row['station_id']}\t{float(row['latitude']):.6f}\t"
        f"{float(row['longitude']):.6f}\t{float(row['elevation_m']):.3f}\n"
        for row in stations)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _station_rows(selected: list[str], rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Station metadata for ``selected``: the archive's latest row in the
    window, else the frozen table.  Stations with no usable coordinates are
    returned separately."""

    frozen = {row["station_id"]: row for row in _station_table()}
    latest: dict[str, dict] = {}
    for row in rows:
        station = row["station"]
        if row.get("valid") is None:
            continue
        if station not in latest or row["valid"] >= latest[station]["valid"]:
            latest[station] = row
    table = []
    unplaced = []
    for station in selected:
        row = latest.get(station, {})
        base = frozen.get(station, {})
        latitude = _number(row.get("latitude"))
        longitude = _number(row.get("longitude"))
        elevation = _number(row.get("elevation"))
        if latitude is None or longitude is None:
            latitude, longitude = _number(base.get("latitude")), _number(base.get("longitude"))
        if elevation is None:
            elevation = _number(base.get("elevation_m"))
        if latitude is None or longitude is None or elevation is None or not -90 <= latitude <= 90:
            if station in latest or station in frozen:
                unplaced.append(station)
            continue
        table.append({
            "station_id": station,
            "name": str(row.get("name") or base.get("name") or ""),
            "latitude": latitude,
            "longitude": _wrap(longitude),
            "elevation_m": elevation,
            "network": SOURCE,
            "state": "",
            "country": str(row.get("country") or base.get("country") or ""),
        })
    return table, unplaced


# ---------------------------------------------------------------------------
# The front door
# ---------------------------------------------------------------------------

def _years(start: datetime, end: datetime) -> list[int]:
    return list(range(start.year, end.year + 1))


def fetch_surface(bbox, valid_time: datetime, folder, *, timeout: float = 120.0,
                  refresh: bool = False, station_ids=None) -> Path:
    """Decode the archive's reports for one hour into ``folder``.

    ``bbox`` is ``(west, south, east, north)`` in degrees (``west > east``
    crosses 180 deg); ``valid_time`` an aware UTC datetime (a naive one is
    read as UTC).  Writes ``folder/stations.json`` and
    ``folder/surface.json`` and returns the latter.  ``station_ids``
    replaces the bbox's frozen-table selection.  ``timeout`` bounds the
    whole fetch in seconds.

    Raises :class:`LookupError` when no report survives the screens and
    :class:`DynamicalUnavailable` when the archive cannot be read.
    """

    target = _utc(valid_time)
    folder = Path(folder)
    if station_ids is not None:
        selected = sorted({str(s).strip() for s in station_ids if str(s).strip()})
    else:
        if bbox is None:
            raise ValueError("fetch_surface needs a bbox or station_ids")
        selected = sorted(row["station_id"] for row in stations_in_bbox(*bbox))
    if not selected:
        raise LookupError(f"the Dynamical.org ASOS archive has no station inside {bbox}")

    _import_pyarrow()
    from woof.fetch_endpoints import TransferRefusal

    stats = _Stats()
    _LAST_STATS.clear()
    budget = _Budget(timeout)
    window_start = target - timedelta(seconds=SCREEN_WINDOW_SECONDS)
    window_end = target + timedelta(seconds=SCREEN_WINDOW_SECONDS)
    base = base_url()
    rows: list[dict] = []
    versions: list[_Version] = []
    try:
        for year in _years(window_start, window_end):
            url = year_url(year, base)
            year_end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
            needed_after = min(window_end, year_end) + SETTLE
            for attempt in range(2):
                try:
                    version = _resolve(url, needed_after=needed_after,
                                       refresh=refresh or attempt > 0, budget=budget,
                                       stats=stats, timeout=timeout)
                    year_rows = _read_rows(version, selected, window_start, window_end, stats)
                    break
                except _VersionChanged:
                    if attempt:
                        raise DynamicalUnavailable(
                            f"{url} was rewritten twice while it was being read; "
                            "try again in a few minutes") from None
                    _progress(f"dynamical-asos: {url} was rewritten while it was "
                              "being read; reading the new version")
                except TransferRefusal as refusal:
                    cause = refusal.__cause__
                    if (year != target.year and isinstance(cause, HTTPError)
                            and cause.code == 404):
                        year_rows, version = [], None
                        break
                    raise DynamicalUnavailable(
                        f"the Dynamical.org ASOS archive did not serve {url}: {refusal}") from refusal
            if version is not None:
                versions.append(version)
            rows.extend(year_rows)
    except _OutOfTime:
        raise DynamicalUnavailable(
            f"reading the Dynamical.org ASOS archive took longer than {timeout:g} s") from None
    finally:
        _LAST_STATS.update(stats.as_dict())

    hasher = hashlib.sha256()
    for version in versions:
        version.digest(hasher)
    stations, unplaced = _station_rows(selected, rows)
    placed = {row["station_id"] for row in stations}
    reports, screen, kept = _decode(rows, [s for s in selected if s in placed], target)
    screen["stations_without_coordinates"] = unplaced
    if not reports:
        raise LookupError(
            f"the Dynamical.org ASOS archive has no report that passes the screens within "
            f"{MATCH_SECONDS} s of {_seam_time(target)} for the {len(selected)} station(s) asked for")

    now = _seam_time(datetime.now(timezone.utc))
    content = station_rows_digest(stations)
    table_record = {
        "schema": STATIONS_SCHEMA,
        "status": "READY",
        "archive": base,
        "networks": [],
        "frozen_at": now,
        "content_sha256": content,
        "source": SOURCE,
        "attribution": ATTRIBUTION,
        "stations": stations,
    }
    kept_set = set(kept)
    primary = year_url(target.year, base)
    provenance = {
        "source": SOURCE,
        "product": PRODUCT,
        "uri": primary if any(v.url == primary for v in versions) else versions[0].url,
        "sha256": hasher.hexdigest(),
        "fetched_at": now,
        "is_stub": False,
        "stub_reason": "",
        "uris": [v.url for v in versions],
        "etags": [v.etag or v.last_modified for v in versions],
        "attribution": ATTRIBUTION,
    }
    record = {
        "schema": SURFACE_SCHEMA,
        "status": "READY",
        "provenance": provenance,
        "station_table_sha256": content,
        "valid_times": [_seam_time(target)],
        "match_seconds": MATCH_SECONDS,
        "min_report_rate": MIN_REPORT_RATE,
        "max_screen_rate": MAX_SCREEN_RATE,
        "screen": screen,
        "stations": [row for row in stations if row["station_id"] in kept_set],
        "reports": reports,
    }
    from woof.fetch_guard import atomic_write_text, hold

    folder.mkdir(parents=True, exist_ok=True)
    surface = folder / "surface.json"
    with hold("dynamical-asos", folder, progress=_progress):
        atomic_write_text(folder / "stations.json", json.dumps(table_record, indent=2) + "\n")
        atomic_write_text(surface, json.dumps(record, indent=2) + "\n")
    return surface


# ---------------------------------------------------------------------------
# Used by tools/freeze_dynamical_asos_stations.py
# ---------------------------------------------------------------------------

def read_station_columns(year: int, *, timeout: float = 3600.0, refresh: bool = False):
    """``(latest, version, stats)`` for the freezer: the last row of every
    station in ``year``'s file (by station id), reading only the station
    metadata columns.  Raises :class:`DynamicalUnavailable` when the archive
    cannot be read."""

    try:
        return _read_station_columns(year, timeout=timeout, refresh=refresh)
    except _OutOfTime:
        raise DynamicalUnavailable(
            f"reading the station columns took longer than {timeout:g} s") from None
    except _VersionChanged:
        raise DynamicalUnavailable(
            f"{year_url(year)} was rewritten while it was being read; run again") from None


def _read_station_columns(year: int, *, timeout: float, refresh: bool):
    pa = _import_pyarrow()
    stats = _Stats()
    budget = _Budget(timeout)
    url = year_url(year)
    version = _resolve(url, needed_after=datetime(year + 1, 1, 1, tzinfo=timezone.utc) + SETTLE,
                       refresh=refresh, budget=budget, stats=stats, timeout=min(timeout, 300.0))
    handle = pa.parquet.ParquetFile(_RangeFile(version), pre_buffer=True)
    columns = ["station", "latitude", "longitude", "elevation", "name", "country"]
    version.prefetch(_chunk_ranges(handle.metadata, list(range(handle.metadata.num_row_groups)),
                                   columns))
    latest: dict[str, dict] = {}
    for group in range(handle.metadata.num_row_groups):
        table = handle.read_row_group(group, columns=columns, use_threads=False)
        stations = table.column("station").to_pylist()
        # Rows are sorted by (station, valid): the last row of a run is the
        # station's latest position.
        last_index = {}
        for index, station in enumerate(stations):
            last_index[station] = index
        picked = table.take(pa.array(sorted(last_index.values()), type=pa.int64()))
        for row in picked.to_pylist():
            latest[row["station"]] = row
        _progress(f"dynamical-asos: row group {group + 1}/{handle.metadata.num_row_groups}, "
                  f"{len(latest)} stations, {stats.bytes_downloaded / 1e6:.1f} MB downloaded")
    with stats.lock:
        stats.row_groups_total = stats.row_groups_read = handle.metadata.num_row_groups
    return latest, version, stats


__all__ = ["ATTRIBUTION", "BASE_URL_ENV", "CACHE_ENV", "COLUMNS", "DEFAULT_BASE_URL",
           "DynamicalUnavailable", "SOURCE", "USER_AGENT", "base_url", "cache_root",
           "fetch_surface", "last_fetch_stats", "read_station_columns",
           "station_count_in_bbox", "station_rows_digest", "stations_in_bbox", "year_url"]
