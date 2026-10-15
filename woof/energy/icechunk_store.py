"""Icechunk stores for the energy hindcast: remote storage, many writers.

A year-long reanalysis at every public wind farm is run as hundreds of
independent jobs (one cluster or leaf domain, one time segment each) on many
GPUs and nodes.  Each job writes its piece straight into a shared Icechunk
repository on S3, GCS or a local/shared filesystem; nothing is staged on the
node's disk.  This module owns that path.

Storage (:func:`storage_from_url`)
    ``s3://bucket/prefix``      AWS S3 or an S3-compatible endpoint (MinIO).
                                Credentials come from the environment: the
                                ``AWS_*`` variables, the shared profile, or
                                the instance / pod role.  ``region`` defaults
                                to ``AWS_REGION`` then ``AWS_DEFAULT_REGION``;
                                ``endpoint_url`` to ``AWS_ENDPOINT_URL_S3``
                                then ``AWS_ENDPOINT_URL``.
    ``gs://bucket/prefix``      Google Cloud Storage (``gcs://`` is an alias).
                                Application-default credentials or workload
                                identity, or ``service_account_file``.
    ``file:///abs/path``        a local or shared (NFS/Lustre) directory; a
                                bare path means the same.
    ``memory://name``           an in-process store, shared by name within
                                the process (tests and dry runs).
    Any other scheme is refused.  ``anonymous=True`` reads public buckets
    without credentials.  No credential is ever written to a summary: URLs
    are reported as given, and a URL carrying a password is refused.

Concurrency (:func:`commit_with_retry`)
    Every writer applies its writes to a fresh writable session at the
    branch tip and commits.  When another writer committed first the commit
    raises ``ConflictError``; the session is rebased with
    ``ConflictDetector``, which succeeds whenever the two writers touched
    disjoint chunks, and the commit is retried.  When they touched the same
    chunk (a site or time chunk shared by two jobs) the rebase fails; the
    session is discarded and the writes are re-applied on a new session from
    the new tip, so the read-modify-write of the shared chunk now sees the
    other job's values and nothing is lost.  Both paths sleep a jittered,
    exponentially growing delay (full jitter, ``base_delay_s * 2**n`` capped
    at ``max_delay_s``) and the total number of commit attempts is bounded by
    ``max_attempts``; exhausting it raises :class:`IcechunkStoreRefused`.
    Abandoned sessions leave unreferenced chunks that
    ``Repository.garbage_collect`` removes; they are never visible.

Site store (``woof-energy.icechunk-sites.v1``)
    :func:`init_site_store` pre-allocates the whole ``(time, site, height)``
    year once: the time axis from start, end (inclusive) and interval, the
    sites in ``sites.v1`` order, the heights, every requested
    ``forecast.v1`` variable as a float32 array filled with NaN (metadata
    only; no data chunk is written), and ``written`` ``(time, site)`` int8,
    1 where a job wrote that time and site.  Initialising again with the
    same axis, sites, heights and variables is a no-op; anything else is
    refused.  :func:`write_site_region` writes one job's ``forecast.v1``
    data into its times and sites with orthogonal selection (only the chunks
    that hold the job's points are touched), sets ``written``, fills the
    per-site coordinates the job owns (``terrain_height``, ``domain_id``,
    ``dx_m``, ``inside``) and commits.  Its times must lie on the axis and
    its sites in the store; its heights must equal the store's.  Notes and
    the job id go into the commit metadata, never into attributes, because
    two jobs changing one attribute would conflict.

Gridded store (``woof-energy.icechunk-gridded.v1``)
    :func:`write_gridded_segment` writes the wrfout frames of one domain and
    one segment into a group (one group per cluster/leaf domain), on a time
    axis pre-allocated at the group's first write.  The variable set is the
    union over the initialising segment's frames (the seed-and-NaN-fill rule
    of ``tools/wrfout_to_netcdf.py``: a field the model first writes after
    t=0, such as ``REFL_10CM``, is a NaN-filled array, not a dropped one);
    a later frame missing a stored field leaves it at its fill value, and a
    frame field the store does not hold is dropped and reported.  ``Times``
    and ``XTIME`` are replaced by the ``Time`` coordinate (``XTIME`` counts
    minutes from each segment's own start, so it has no single meaning on a
    shared axis).  ``frame_written`` ``(Time,)`` is 1 for every frame
    written and ``frame_published_unix_ms`` its source file's mtime.
    Re-running a segment skips frames already written
    (``on_existing="skip"``, the default) or rewrites them
    (``"overwrite"``, or ``"newer"``: only when the source file is newer
    than the one written, so a later run wins).  The live export
    (:mod:`woof.icechunk_export`) also lets the axis grow at its end
    (``extend_axis``) and creates late fields on the frame that first
    carries them (``add_variables``).  Numeric arrays are compressed with
    Blosc zstd and bitshuffle; the time chunk is ``time_chunk`` frames (default 1, so
    segments never share a chunk) and the other dimensions are whole unless
    a chunk would exceed :data:`MAX_CHUNK_BYTES`.

Manifests are split along ``time``/``Time`` every
:data:`MANIFEST_SPLIT_CHUNKS` chunks, set when this module creates the
repository, so a commit rewrites only the manifests its segment touched.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import socket
import sys
import threading
import time as _time
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import urlsplit

import numpy as np

from woof.energy.contracts import (
    FORECAST_COORDINATES,
    FORECAST_SCHEMA,
    FORECAST_VARIABLES,
    PROFILE_DIMS,
    ContractError,
    SiteSet,
    load_sites,
    sha256_file,
)

SITE_STORE_SCHEMA = "woof-energy.icechunk-sites.v1"
GRIDDED_STORE_SCHEMA = "woof-energy.icechunk-gridded.v1"
WRITE_SCHEMA = "woof-energy.icechunk-write.v1"
VERIFY_SCHEMA = "woof-energy.icechunk-verify.v1"

#: URL schemes :func:`storage_from_url` accepts.
SCHEMES = ("s3", "gs", "gcs", "file", "memory")

#: Largest uncompressed chunk the gridded writer makes (bytes).
MAX_CHUNK_BYTES = 64 * 1024 * 1024

#: Manifest split size along ``time``/``Time``, in chunks.
MANIFEST_SPLIT_CHUNKS = 672

#: Chunk length of the ``Time`` coordinate of a gridded group whose axis
#: grows (``write_gridded_segment(extend_axis=True)``), so growing it one
#: frame at a time rewrites one small chunk rather than adding one.
GROWING_TIME_COORD_CHUNK = 4096

#: Gridded-group array holding each frame's publication stamp (its source
#: file's mtime, Unix ms); ``on_existing="newer"`` compares against it.
FRAME_PUBLISHED = "frame_published_unix_ms"

#: Default site-store chunking: one day of steps by 64 sites by all heights.
DEFAULT_SITE_CHUNK = 64
DEFAULT_TIME_CHUNK_SECONDS = 86400

#: Coordinates a job fills for the sites it owns (not known at init).
JOB_SITE_COORDS = ("terrain_height", "domain_id", "dx_m", "inside")

#: wrfout variables the gridded store replaces with its ``Time`` coordinate.
REPLACED_WRF_VARIABLES = ("Times", "XTIME")

_STRING_COORDS = ("site_id", "asset_id", "kind", "domain_id")
_MEMORY: dict[str, Any] = {}
_MEMORY_LOCK = threading.Lock()


class IcechunkStoreRefused(RuntimeError):
    """A store request this module will not carry out (bad URL, a store
    that does not match, writes off the axis, retries exhausted)."""


def _icechunk():
    try:
        import icechunk
    except ModuleNotFoundError as error:
        raise IcechunkStoreRefused(
            "Icechunk output needs the icechunk package in the active "
            "environment (install the pixi environment)") from error
    global _LOGS_QUIETED
    if not _LOGS_QUIETED:
        _LOGS_QUIETED = True
        # Pinned: icechunk's Rust tracing writes to STDOUT, and an aborted
        # rebase -- routine in commit_with_retry -- logs a WARN line, which
        # would corrupt the JSON summaries the doors print there.  Errors
        # still show; ICECHUNK_LOG, when set, is left to rule.
        if not os.environ.get("ICECHUNK_LOG") \
                and hasattr(icechunk, "set_logs_filter"):
            icechunk.set_logs_filter("error")
    return icechunk


_LOGS_QUIETED = False


def _zarr():
    try:
        import zarr
    except ModuleNotFoundError as error:
        raise IcechunkStoreRefused(
            "Icechunk output needs the zarr package in the active "
            "environment") from error
    return zarr


# --------------------------------------------------------------------------
# storage


def is_store_url(text: str | os.PathLike) -> bool:
    """True when ``text`` names a store by URL (``scheme://...``) rather
    than by a bare local path."""

    text = str(text)
    scheme, sep, _ = text.partition("://")
    return bool(sep) and scheme.lower() in SCHEMES


def _bucket_prefix(url: str) -> tuple[str, str | None]:
    parts = urlsplit(url)
    if parts.password or parts.username:
        raise IcechunkStoreRefused(
            f"store URL {parts.scheme}://...: credentials in the URL are "
            "refused; use the environment, a profile or a role")
    bucket = parts.netloc
    if not bucket:
        raise IcechunkStoreRefused(f"store URL {url!r} names no bucket")
    prefix = parts.path.strip("/")
    return bucket, (prefix or None)


def storage_from_url(url: str | os.PathLike, *, region: str | None = None,
                     anonymous: bool = False,
                     endpoint_url: str | None = None,
                     allow_http: bool = False,
                     force_path_style: bool | None = None,
                     requester_pays: bool = False,
                     service_account_file: str | None = None):
    """An ``icechunk.Storage`` for ``url`` (see the module docstring for the
    schemes and where credentials come from).

    Pinned: an S3 ``endpoint_url`` (argument or environment) turns on
    path-style addressing unless ``force_path_style`` says otherwise, and an
    ``http://`` endpoint turns on ``allow_http`` -- what MinIO needs.
    """

    ic = _icechunk()
    text = os.fspath(url)
    if not is_store_url(text):
        if "://" in text:
            raise IcechunkStoreRefused(
                f"store URL scheme {text.split('://', 1)[0]!r} is not one of "
                f"{', '.join(s + '://' for s in SCHEMES)}")
        return ic.local_filesystem_storage(str(Path(text).expanduser()
                                               .resolve()))
    scheme = text.split("://", 1)[0].lower()
    if scheme == "file":
        rest = text.split("://", 1)[1]
        if not rest:
            raise IcechunkStoreRefused("file:// URL names no path")
        return ic.local_filesystem_storage(str(Path(rest).expanduser()
                                               .resolve()))
    if scheme == "memory":
        name = text.split("://", 1)[1]
        with _MEMORY_LOCK:
            if name not in _MEMORY:
                _MEMORY[name] = ic.in_memory_storage()
            return _MEMORY[name]
    bucket, prefix = _bucket_prefix(text)
    if scheme == "s3":
        region = region or os.environ.get("AWS_REGION") \
            or os.environ.get("AWS_DEFAULT_REGION")
        endpoint_url = endpoint_url or os.environ.get("AWS_ENDPOINT_URL_S3") \
            or os.environ.get("AWS_ENDPOINT_URL")
        if endpoint_url and endpoint_url.startswith("http://"):
            allow_http = True
        if force_path_style is None:
            force_path_style = bool(endpoint_url)
        credentials: dict[str, Any] = ({"anonymous": True} if anonymous
                                       else {"from_env": True})
        return ic.s3_storage(
            bucket=bucket, prefix=prefix, region=region,
            endpoint_url=endpoint_url, allow_http=bool(allow_http),
            force_path_style=bool(force_path_style),
            requester_pays=bool(requester_pays), **credentials)
    # gs / gcs
    if anonymous:
        credentials = {"anonymous": True}
    elif service_account_file:
        credentials = {"service_account_file": str(service_account_file)}
    else:
        credentials = {"from_env": True}
    return ic.gcs_storage(bucket=bucket, prefix=prefix, **credentials)


def repository_config(split_chunks: int = MANIFEST_SPLIT_CHUNKS):
    """The configuration this module creates repositories with: manifests
    split along ``time`` and ``Time`` every ``split_chunks`` chunks."""

    ic = _icechunk()
    split = ic.ManifestSplittingConfig.from_dict({
        ic.ManifestSplitCondition.AnyArray(): {
            ic.ManifestSplitDimCondition.DimensionName("time"): split_chunks,
            ic.ManifestSplitDimCondition.DimensionName("Time"): split_chunks,
        }
    })
    return ic.RepositoryConfig(manifest=ic.ManifestConfig(splitting=split))


def local_path(store) -> Path | None:
    """The directory a bare path or ``file://`` URL names; ``None`` for
    object stores, ``memory://`` and ``icechunk.Storage`` objects."""

    if not isinstance(store, (str, os.PathLike)):
        return None
    text = os.fspath(store)
    if not is_store_url(text):
        return Path(text).expanduser().resolve()
    scheme, _, rest = text.partition("://")
    if scheme.lower() == "file" and rest:
        return Path(rest).expanduser().resolve()
    return None


#: Suffix of the lock file beside a filesystem repository.
COMMIT_LOCK_SUFFIX = ".woof-commit.lock"


@contextmanager
def _flock(path: Path):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def commit_lock(store) -> Callable[[], Any]:
    """A context-manager factory serialising branch updates.

    Pinned: Icechunk's local filesystem storage has no atomic
    compare-and-swap for branch refs, so two processes committing at once
    can both "succeed" and one commit is silently dropped (measured: 5 of 30
    two-process races with icechunk 1.1 on a local disk).  For a bare path
    or ``file://`` store the commit is therefore taken under an exclusive
    ``flock`` on ``<repo>.woof-commit.lock`` beside the repository (Icechunk
    creates repositories only in empty prefixes, so it cannot live inside)
    -- held only for the commit itself, and honoured by every writer in
    this module.  The filesystem must support ``flock``: local disks,
    NFSv4, Lustre mounted with ``-o flock``.  S3 and GCS use the object
    store's conditional writes and need no lock."""

    path = local_path(store)
    if path is None:
        return nullcontext
    lock = path.with_name(path.name + COMMIT_LOCK_SUFFIX)
    return lambda: _flock(lock)


def open_repository(store, *, create: bool = True, **storage_options):
    """Open the repository ``store`` names (a URL, a path or an
    ``icechunk.Storage``), creating it when ``create`` and absent.

    Two writers creating the same repository at once is not an error: the
    loser opens what the winner made (a filesystem store is created under
    :func:`commit_lock`)."""

    ic = _icechunk()
    storage = store if not isinstance(store, (str, os.PathLike)) \
        else storage_from_url(store, **storage_options)
    path = local_path(store)
    if path is not None:
        if path.exists() and not path.is_dir():
            raise IcechunkStoreRefused(
                f"{path} exists and is not an Icechunk repository")
        if not create and not path.exists():
            raise IcechunkStoreRefused(f"no Icechunk repository at {store}")
    if ic.Repository.exists(storage):
        return ic.Repository.open(storage)
    if not create:
        raise IcechunkStoreRefused(f"no Icechunk repository at {store}")
    with commit_lock(store)():
        if ic.Repository.exists(storage):
            return ic.Repository.open(storage)
        try:
            return ic.Repository.create(storage, config=repository_config())
        except ic.IcechunkError as error:
            if ic.Repository.exists(storage):
                return ic.Repository.open(storage)
            raise IcechunkStoreRefused(
                f"cannot create an Icechunk repository at {store}: "
                f"{str(error).splitlines()[0] if str(error) else error}"
            ) from error


# --------------------------------------------------------------------------
# time axis


def _parse_instant(text: str | np.datetime64) -> np.datetime64:
    if isinstance(text, np.datetime64):
        return text.astype("datetime64[s]")
    value = str(text).strip()
    if value.endswith("Z"):
        value = value[:-1]
    value = value.replace("_", "T")
    try:
        return np.datetime64(value, "s")
    except ValueError as error:
        raise IcechunkStoreRefused(
            f"cannot read {text!r} as a UTC time (YYYY-MM-DDTHH[:MM[:SS]])"
        ) from error


@dataclass(frozen=True)
class TimeAxis:
    """A regular valid-time axis: ``start`` to ``end`` inclusive every
    ``interval_s`` seconds (UTC, second resolution)."""

    start: np.datetime64
    end: np.datetime64
    interval_s: int

    def __post_init__(self):
        start, end = _parse_instant(self.start), _parse_instant(self.end)
        interval = int(self.interval_s)
        if interval <= 0:
            raise IcechunkStoreRefused("time-axis interval must be positive")
        if end < start:
            raise IcechunkStoreRefused("time-axis end is before its start")
        span = int((end - start) / np.timedelta64(1, "s"))
        if span % interval:
            raise IcechunkStoreRefused(
                f"time-axis end {end}Z is not a whole number of "
                f"{interval} s steps after its start {start}Z")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "interval_s", interval)

    @classmethod
    def parse(cls, text: str) -> "TimeAxis":
        """``START,END,INTERVAL`` -- interval in seconds, or with an ``s``,
        ``min`` or ``h`` suffix."""

        parts = [p.strip() for p in str(text).split(",")]
        if len(parts) != 3:
            raise IcechunkStoreRefused(
                f"time axis takes START,END,INTERVAL; got {text!r}")
        return cls(parts[0], parts[1], parse_interval(parts[2]))

    @property
    def size(self) -> int:
        span = int((self.end - self.start) / np.timedelta64(1, "s"))
        return span // self.interval_s + 1

    def times(self) -> np.ndarray:
        return self.start + np.arange(self.size, dtype=np.int64) \
            * np.timedelta64(self.interval_s, "s")

    def index_of(self, times: Sequence | np.ndarray) -> np.ndarray:
        """Axis indices of ``times``; refused when any is off the axis."""

        values = np.asarray(times, dtype="datetime64[s]")
        offsets = (values - self.start) / np.timedelta64(1, "s")
        offsets = offsets.astype(np.int64)
        off = (offsets % self.interval_s != 0) | (offsets < 0) \
            | (offsets // self.interval_s >= self.size)
        if np.any(off):
            bad = values[off][0]
            raise IcechunkStoreRefused(
                f"time {bad}Z is not on the store's axis "
                f"({self.start}Z..{self.end}Z every {self.interval_s} s)")
        return offsets // self.interval_s

    def to_attrs(self) -> dict[str, Any]:
        return {"time_start": str(self.start) + "Z",
                "time_end": str(self.end) + "Z",
                "time_interval_s": self.interval_s}

    @classmethod
    def from_attrs(cls, attrs: Mapping[str, Any]) -> "TimeAxis":
        try:
            return cls(attrs["time_start"], attrs["time_end"],
                       int(attrs["time_interval_s"]))
        except KeyError as error:
            raise IcechunkStoreRefused(
                f"store group lacks its time-axis attribute {error}") from error

    def units(self) -> str:
        return ("seconds since "
                + str(self.start).replace("T", " "))


def parse_interval(text: str | int | float) -> int:
    """Seconds from ``900``, ``900s``, ``15min`` or ``1h``."""

    value = str(text).strip().lower()
    for suffix, scale in (("min", 60), ("s", 1), ("h", 3600)):
        if value.endswith(suffix):
            value, factor = value[: -len(suffix)], scale
            break
    else:
        factor = 1
    try:
        seconds = float(value) * factor
    except ValueError as error:
        raise IcechunkStoreRefused(
            f"cannot read interval {text!r} (seconds, or s/min/h)") from error
    if seconds <= 0 or seconds != int(seconds):
        raise IcechunkStoreRefused(
            f"interval {text!r} must be a positive whole number of seconds")
    return int(seconds)


# --------------------------------------------------------------------------
# commits


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded commit retries.  Pinned defaults: 12 commit attempts, delays
    drawn uniformly from ``[0, min(max_delay_s, base_delay_s * 2**n)]``."""

    max_attempts: int = 12
    base_delay_s: float = 0.25
    max_delay_s: float = 20.0

    def delay(self, attempt: int, rng: random.Random) -> float:
        ceiling = min(self.max_delay_s,
                      self.base_delay_s * (2 ** max(0, attempt - 1)))
        return rng.uniform(0.0, ceiling)


@dataclass
class CommitResult:
    snapshot: str
    committed: bool
    attempts: int
    rebases: int
    rewrites: int
    info: dict[str, Any]


def commit_with_retry(repo, *, branch: str,
                      apply: Callable[[Any], dict[str, Any] | None],
                      message: str, metadata: Mapping[str, Any],
                      policy: RetryPolicy | None = None,
                      before_commit: Callable[[int], None] | None = None,
                      sleep: Callable[[float], None] = _time.sleep,
                      lock: Callable[[], Any] = nullcontext
                      ) -> CommitResult:
    """Apply ``apply(session)`` on a fresh session at the tip of ``branch``
    and commit it, rebasing or re-applying on conflict (module docstring).

    ``apply`` must be safe to call again on a newer tip; it returns a dict
    recorded as ``info`` (the last call's).  A session ``apply`` left
    unchanged is not committed (``committed=False``, the tip snapshot).
    ``lock()`` is held around each commit (:func:`commit_lock`).
    ``before_commit(attempt)`` is a test hook."""

    ic = _icechunk()
    policy = policy or RetryPolicy()
    rng = random.Random()
    attempts = rebases = rewrites = 0
    while True:
        session = repo.writable_session(branch)
        info = apply(session) or {}
        if not session.has_uncommitted_changes:
            return CommitResult(repo.lookup_branch(branch), False, attempts,
                                rebases, rewrites, info)
        while True:
            attempts += 1
            if before_commit is not None:
                before_commit(attempts)
            try:
                with lock():
                    snapshot = session.commit(message,
                                              metadata=dict(metadata))
                return CommitResult(str(snapshot), True, attempts, rebases,
                                    rewrites, info)
            except ic.ConflictError as error:
                if attempts >= policy.max_attempts:
                    raise IcechunkStoreRefused(
                        f"commit to branch {branch!r} still conflicted after "
                        f"{attempts} attempts ({rebases} rebases, "
                        f"{rewrites} rewrites): {error}") from error
                sleep(policy.delay(attempts, rng))
                try:
                    session.rebase(ic.ConflictDetector())
                    rebases += 1
                except ic.RebaseFailedError:
                    rewrites += 1
                    break


def _job_metadata(job_id: str | None, segment_id: str | None, group: str,
                  extra: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema": WRITE_SCHEMA, "job_id": job_id or "",
            "segment_id": segment_id or "", "group": group,
            "host": socket.gethostname(), "pid": os.getpid(),
            "written_utc": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"),
            **extra}


def _open_group(session, group: str, mode: str = "r+"):
    zarr = _zarr()
    try:
        return zarr.open_group(session.store, path=group, mode=mode)
    except (FileNotFoundError, KeyError):
        return None
    except Exception as error:  # zarr's GroupNotFoundError across versions
        if type(error).__name__ in ("GroupNotFoundError",
                                    "NodeTypeValidationError",
                                    "ContainsArrayError"):
            return None
        raise


def _compressors(compress: bool, clevel: int):
    if not compress:
        return "auto"
    from zarr.codecs import BloscCodec, BloscShuffle

    return [BloscCodec(cname="zstd", clevel=int(clevel),
                       shuffle=BloscShuffle.bitshuffle)]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _selector(index: np.ndarray):
    """A slice when ``index`` is a contiguous ascending run, else itself."""

    index = np.asarray(index, dtype=np.int64)
    if index.size and np.all(np.diff(index) == 1):
        return slice(int(index[0]), int(index[-1]) + 1)
    return index


# --------------------------------------------------------------------------
# site store


def _resolve_sites(sites: SiteSet | str | os.PathLike) -> tuple[SiteSet, str]:
    """The sites and the SHA-256 of their canonical JSON (sorted keys), so
    a path and the same sites loaded in Python hash alike, whatever the
    file's whitespace."""

    import hashlib

    site_set = sites if isinstance(sites, SiteSet) else load_sites(sites)
    text = json.dumps(site_set.to_json(), sort_keys=True)
    return site_set, hashlib.sha256(text.encode("utf-8")).hexdigest()


def default_time_chunk(axis: TimeAxis) -> int:
    return max(1, min(axis.size, DEFAULT_TIME_CHUNK_SECONDS // axis.interval_s))


def init_site_store(store, *, sites: SiteSet | str | os.PathLike,
                    axis: TimeAxis, heights_m: Sequence[float] | None = None,
                    variables: Sequence[str] | None = None,
                    group: str = "forecast", branch: str = "main",
                    time_chunk: int | None = None,
                    site_chunk: int = DEFAULT_SITE_CHUNK,
                    compress: bool = True, zstd_clevel: int = 5,
                    job_id: str | None = None,
                    policy: RetryPolicy | None = None,
                    **storage_options) -> dict[str, Any]:
    """Pre-allocate the year-long ``(time, site, height)`` site store once
    (module docstring).  Returns a summary; ``created`` is False when an
    identical store was already there."""

    try:
        import xarray as xr
    except ModuleNotFoundError as error:
        raise IcechunkStoreRefused(
            "the site store needs xarray in the active environment") from error
    site_set, sites_digest = _resolve_sites(sites)
    heights = np.asarray(heights_m if heights_m is not None
                         else site_set.heights_m, dtype=np.float64)
    if heights.size == 0 or np.any(np.diff(heights) <= 0) \
            or np.any(heights <= 0):
        raise IcechunkStoreRefused(
            "heights must be positive and strictly increasing")
    names = list(variables) if variables else list(FORECAST_VARIABLES)
    unknown = [n for n in names if n not in FORECAST_VARIABLES]
    if unknown:
        raise IcechunkStoreRefused(
            f"{unknown} are not forecast.v1 variables")
    if len(set(names)) != len(names):
        raise IcechunkStoreRefused("variables are named more than once")
    tchunk = int(time_chunk or default_time_chunk(axis))
    schunk = max(1, min(int(site_chunk), len(site_set)))
    if tchunk <= 0:
        raise IcechunkStoreRefused("time_chunk must be positive")
    if len(site_set) == 0:
        raise IcechunkStoreRefused("the sites document has no sites")
    expected = {
        "schema": SITE_STORE_SCHEMA, "forecast_schema": FORECAST_SCHEMA,
        **axis.to_attrs(), "sites_sha256": sites_digest,
        "site_count": len(site_set), "heights_m": heights.tolist(),
        "variables": names,
    }
    repo = open_repository(store, **storage_options)

    def same(attrs: Mapping[str, Any]) -> list[str]:
        return [key for key, value in expected.items()
                if _jsonable(attrs.get(key)) != _jsonable(value)]

    def apply(session) -> dict[str, Any]:
        existing = _open_group(session, group, mode="r")
        if existing is not None:
            differ = same(dict(existing.attrs))
            if differ:
                raise IcechunkStoreRefused(
                    f"group {group!r} already holds a different store "
                    f"(differs in {', '.join(differ)}); refusing to "
                    "replace it")
            return {"created": False}
        arrays = site_set.as_arrays()
        count = len(site_set)
        coords: dict[str, Any] = {
            "time": ("time", axis.times().astype("datetime64[ns]"),
                     {"long_name": FORECAST_COORDINATES["time"][2],
                      "standard_name": "time"}),
            "height": ("height", heights,
                       {"units": "m", "positive": "up",
                        "long_name": FORECAST_COORDINATES["height"][2]}),
        }
        fills = {"terrain_height": np.full(count, np.nan),
                 "domain_id": np.asarray([""] * count, dtype=object),
                 "dx_m": np.full(count, np.nan),
                 "inside": np.zeros(count, dtype=np.int8)}
        site_names = [n for n, (dims, _, _) in FORECAST_COORDINATES.items()
                      if dims == ("site",)]
        for name in site_names:
            dims, units, long_name = FORECAST_COORDINATES[name]
            values = fills[name] if name in fills else arrays[name]
            if name in _STRING_COORDS:
                values = np.asarray([str(v) for v in values], dtype=object)
            attrs = {"long_name": long_name}
            if units:
                attrs["units"] = units
            coords[name] = (dims, values, attrs)
        dataset = xr.Dataset(coords=coords)
        encoding: dict[str, dict[str, Any]] = {
            "time": {"units": axis.units(), "calendar": "proleptic_gregorian",
                     "dtype": "int64"}}
        for name in site_names:
            encoding[name] = {"chunks": (schunk,)}
        dataset.to_zarr(session.store, group=group, mode="w-",
                        consolidated=False, zarr_format=3, encoding=encoding)
        zgroup = _open_group(session, group)
        compressors = _compressors(compress, zstd_clevel)
        coordinates = " ".join(site_names)
        ntime, nheight = axis.size, heights.size
        for name in names:
            dims, units, long_name = FORECAST_VARIABLES[name]
            profile = dims == PROFILE_DIMS
            zgroup.create_array(
                name,
                shape=(ntime, count, nheight) if profile else (ntime, count),
                chunks=(tchunk, schunk, nheight) if profile
                else (tchunk, schunk),
                dtype="float32", fill_value=np.float32(np.nan),
                compressors=compressors, dimension_names=dims,
                attributes={"units": units, "long_name": long_name,
                            "coordinates": coordinates})
        zgroup.create_array(
            "written", shape=(ntime, count), chunks=(tchunk, schunk),
            dtype="int8", fill_value=0, compressors=compressors,
            dimension_names=("time", "site"),
            attributes={"long_name": "1 where a job wrote this time and "
                                     "site",
                        "flag_values": [0, 1],
                        "flag_meanings": "not_written written",
                        "coordinates": coordinates})
        zgroup.attrs.update({
            **expected, "time_chunk": tchunk, "site_chunk": schunk,
            "compression": ("blosc zstd bitshuffle clevel "
                            f"{int(zstd_clevel)}" if compress else "default"),
            "created_utc": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"),
        })
        return {"created": True}

    result = commit_with_retry(
        repo, branch=branch, apply=apply,
        message=(f"woof energy icechunk init: {SITE_STORE_SCHEMA} {group} "
                 f"({axis.size} times x {len(site_set)} sites)"),
        metadata=_job_metadata(job_id, None, group, {"action": "init"}),
        policy=policy, lock=commit_lock(store))
    return {
        "schema": SITE_STORE_SCHEMA, "store": str(store), "group": group,
        "branch": branch, "snapshot": result.snapshot,
        "created": bool(result.info.get("created")),
        "times": axis.size, "sites": len(site_set),
        "heights_m": heights.tolist(), "variables": names,
        "time_chunk": tchunk, "site_chunk": schunk,
        "attempts": result.attempts, **axis.to_attrs(),
    }


def _nan_equal(a: np.ndarray, b: np.ndarray) -> bool:
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        return bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))
    return [str(v) for v in np.asarray(a).ravel()] \
        == [str(v) for v in np.asarray(b).ravel()]


def write_site_region(store, data, *, group: str = "forecast",
                      branch: str = "main", job_id: str | None = None,
                      segment_id: str | None = None,
                      on_existing: str = "overwrite",
                      policy: RetryPolicy | None = None,
                      before_commit: Callable[[int], None] | None = None,
                      **storage_options) -> dict[str, Any]:
    """Write one job's :class:`~woof.energy.extract.ForecastData` into its
    times and sites of an initialised site store and commit (module
    docstring).  ``on_existing="skip"`` drops the times at which every one
    of the job's sites is already written; ``"overwrite"`` (default) writes
    them again -- the same job writes the same values, so a re-run is
    idempotent either way."""

    if on_existing not in ("overwrite", "skip"):
        raise IcechunkStoreRefused(
            f"on_existing {on_existing!r} is not overwrite or skip")
    times = np.asarray(data.times, dtype="datetime64[s]")
    if times.size == 0:
        raise IcechunkStoreRefused("the forecast has no times")
    if np.any(np.diff(times.astype(np.int64)) <= 0):
        raise IcechunkStoreRefused("the forecast times are not increasing")
    site_ids = [str(s) for s in data.coords["site_id"]]
    if len(set(site_ids)) != len(site_ids):
        raise IcechunkStoreRefused("the forecast repeats a site_id")
    unknown_vars = [n for n in data.variables if n not in FORECAST_VARIABLES]
    if unknown_vars:
        raise IcechunkStoreRefused(
            f"{unknown_vars} are not forecast.v1 variables")
    repo = open_repository(store, create=False, **storage_options)

    def apply(session) -> dict[str, Any]:
        zgroup = _open_group(session, group)
        if zgroup is None:
            raise IcechunkStoreRefused(
                f"no site store at group {group!r}; run `woof energy "
                "icechunk init` first")
        attrs = dict(zgroup.attrs)
        if attrs.get("schema") != SITE_STORE_SCHEMA:
            raise IcechunkStoreRefused(
                f"group {group!r} is not a {SITE_STORE_SCHEMA} store")
        axis = TimeAxis.from_attrs(attrs)
        t_index = axis.index_of(times)
        stored_heights = np.asarray(attrs.get("heights_m") or [], float)
        heights = np.asarray(data.heights_m, dtype=np.float64)
        if stored_heights.shape != heights.shape \
                or not np.allclose(stored_heights, heights, atol=1e-6):
            raise IcechunkStoreRefused(
                f"forecast heights {heights.tolist()} differ from the "
                f"store's {stored_heights.tolist()}")
        missing = [n for n in data.variables if n not in attrs["variables"]]
        if missing:
            raise IcechunkStoreRefused(
                f"the store was initialised without {missing}")
        stored_ids = [str(s) for s in zgroup["site_id"][:]]
        position = {sid: i for i, sid in enumerate(stored_ids)}
        absent = [sid for sid in site_ids if sid not in position]
        if absent:
            raise IcechunkStoreRefused(
                f"{len(absent)} site(s) are not in the store, e.g. "
                f"{absent[0]!r}")
        s_index = np.asarray([position[sid] for sid in site_ids], np.int64)
        order = np.argsort(s_index, kind="stable")
        s_sorted = s_index[order]
        keep_t = np.arange(times.size)
        skipped = 0
        if on_existing == "skip":
            written = np.asarray(zgroup["written"].get_orthogonal_selection(
                (_selector(t_index), _selector(s_sorted))))
            done = np.all(written == 1, axis=1)
            keep_t = np.nonzero(~done)[0]
            skipped = int(np.count_nonzero(done))
            if keep_t.size == 0:
                return {"times_written": 0, "times_skipped": skipped}
        t_sel = _selector(t_index[keep_t])
        s_sel = _selector(s_sorted)
        for name, values in data.variables.items():
            values = np.asarray(values)[keep_t][:, order]
            array = zgroup[name]
            if values.shape[2:] != tuple(array.shape[2:]):
                raise IcechunkStoreRefused(
                    f"{name} has shape {values.shape}, the store "
                    f"{array.shape}")
            selection = (t_sel, s_sel) + ((slice(None),)
                                          if values.ndim == 3 else ())
            array.set_orthogonal_selection(selection,
                                           values.astype(np.float32))
        zgroup["written"].set_orthogonal_selection(
            (t_sel, s_sel), np.ones((keep_t.size, s_sorted.size), np.int8))
        for name in JOB_SITE_COORDS:
            if name not in data.coords:
                continue
            ours = np.asarray(data.coords[name])[order]
            array = zgroup[name]
            current = np.asarray(array.get_orthogonal_selection((s_sel,)))
            if not _nan_equal(current, ours):
                if name in _STRING_COORDS:
                    ours = np.asarray([str(v) for v in ours], dtype=object)
                else:
                    ours = ours.astype(array.dtype)
                array.set_orthogonal_selection((s_sel,), ours)
        return {"times_written": int(keep_t.size), "times_skipped": skipped}

    first, last = str(times[0]) + "Z", str(times[-1]) + "Z"
    result = commit_with_retry(
        repo, branch=branch, apply=apply,
        message=(f"woof energy icechunk: job {job_id or '-'} wrote "
                 f"{times.size} times x {len(site_ids)} sites into {group} "
                 f"({first}..{last})"),
        metadata=_job_metadata(job_id, segment_id, group, {
            "action": "write-sites", "first_time": first, "last_time": last,
            "times": int(times.size), "sites": len(site_ids),
            "variables": list(data.variables),
            "notes": list(getattr(data, "notes", []) or []),
            "variable_notes": dict(getattr(data, "variable_notes", {}) or {}),
            "attrs": {k: str(v) for k, v in
                      (getattr(data, "attrs", {}) or {}).items()},
        }),
        policy=policy, before_commit=before_commit,
        lock=commit_lock(store))
    return {
        "schema": WRITE_SCHEMA, "store": str(store), "group": group,
        "branch": branch, "snapshot": result.snapshot,
        "committed": result.committed, "job_id": job_id,
        "segment_id": segment_id, "first_time": first, "last_time": last,
        "times": int(times.size), "sites": len(site_ids),
        "times_written": int(result.info.get("times_written", 0)),
        "times_skipped": int(result.info.get("times_skipped", 0)),
        "attempts": result.attempts, "rebases": result.rebases,
        "rewrites": result.rewrites,
    }


# --------------------------------------------------------------------------
# gridded store (wrfout frames)


@dataclass(frozen=True)
class Frame:
    """One history frame: ``index`` along ``Time`` in ``path``."""

    path: Path
    index: int
    time: np.datetime64


def _netcdf4():
    try:
        import netCDF4
    except ModuleNotFoundError as error:
        raise IcechunkStoreRefused(
            "reading wrfout frames needs netCDF4 in the active "
            "environment") from error
    return netCDF4


def _open_nc(path: Path):
    netCDF4 = _netcdf4()
    nc = netCDF4.Dataset(str(path), "r")
    nc.set_auto_maskandscale(False)
    nc.set_auto_chartostring(False)
    return nc


def _nc_lock():
    """The process-wide netCDF4/HDF5 lock
    (:data:`woof.io.netcdf_serialization.NETCDF4_IO_LOCK`).

    Pinned: HDF5 is not thread-safe, and this module's reader can share
    a process with another netCDF user (the live export run in a host's
    thread beside a writer), so every session here holds it."""

    from woof.io.netcdf_serialization import NETCDF4_IO_LOCK

    return NETCDF4_IO_LOCK


@contextmanager
def _nc_session(path: Path):
    """One read-only netCDF session under :func:`_nc_lock`."""

    with _nc_lock():
        nc = _open_nc(path)
        try:
            yield nc
        finally:
            nc.close()


def _frame_times(nc, path: Path) -> list[np.datetime64]:
    netCDF4 = _netcdf4()
    if "Times" not in nc.variables:
        raise IcechunkStoreRefused(f"{path.name} has no Times variable")
    raw = np.asarray(nc.variables["Times"][:])
    if raw.ndim == 1:
        raw = raw[None, :]
    stamps = netCDF4.chartostring(raw)
    return [_parse_instant(str(s).strip()) for s in np.atleast_1d(stamps)]


def wrfout_frames(paths: Iterable[str | os.PathLike], *,
                  window: tuple[Any, Any] | None = None) -> list[Frame]:
    """Every frame of ``paths`` with ``first <= time < end`` when
    ``window=(first, end)``, sorted by valid time; a time found twice inside
    the window is refused (window out overlapping restart output)."""

    return _select_frames(_all_frames(paths), window)


def _all_frames(paths: Iterable[str | os.PathLike]) -> list[Frame]:
    frames: list[Frame] = []
    for path in sorted(Path(p) for p in paths):
        with _nc_session(path) as nc:
            for index, stamp in enumerate(_frame_times(nc, path)):
                frames.append(Frame(path, index, stamp))
    return frames


def _select_frames(frames: list[Frame], window: tuple[Any, Any] | None
                   ) -> list[Frame]:
    if window is not None:
        lo, hi = _parse_instant(window[0]), _parse_instant(window[1])
        frames = [f for f in frames if lo <= f.time < hi]
    frames = sorted(frames, key=lambda f: f.time)
    seen: dict[np.datetime64, Frame] = {}
    for frame in frames:
        if frame.time in seen:
            raise IcechunkStoreRefused(
                f"valid time {frame.time}Z is in both "
                f"{seen[frame.time].path.name} and {frame.path.name}")
        seen[frame.time] = frame
    return frames


def _chunks(shape: tuple[int, ...], itemsize: int, time_chunk: int
            ) -> tuple[int, ...]:
    chunks = [max(1, min(time_chunk, shape[0]))] + list(shape[1:])
    while math.prod(chunks) * itemsize > MAX_CHUNK_BYTES:
        rest = [i for i in range(1, len(chunks)) if chunks[i] > 1]
        if not rest:
            break
        i = rest[0]
        chunks[i] = math.ceil(chunks[i] / 2)
    return tuple(int(c) for c in chunks)


def _clean_attrs(variable) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for key in variable.ncattrs():
        if key in ("_FillValue",):
            continue
        value = _jsonable(variable.getncattr(key))
        if key == "coordinates" and isinstance(value, str):
            value = " ".join(t for t in value.split()
                             if t not in REPLACED_WRF_VARIABLES)
            if not value:
                continue
        attrs[key] = value
    return attrs


def _scan_variables(frames: Sequence[Frame]) -> dict[str, dict[str, Any]]:
    """name -> {dims, dtype, shape, attrs, path} over the frames' files,
    first file to carry a name wins."""

    found: dict[str, dict[str, Any]] = {}
    for path in dict.fromkeys(f.path for f in frames):
        with _nc_session(path) as nc:
            for name, variable in nc.variables.items():
                if name in found or name in REPLACED_WRF_VARIABLES:
                    continue
                dtype = variable.dtype
                found[name] = {
                    "dims": tuple(variable.dimensions),
                    "dtype": np.dtype(object) if dtype is str
                    else np.dtype(dtype),
                    "shape": tuple(variable.shape),
                    "attrs": _clean_attrs(variable), "path": path,
                }
    return found


def _fill_for(dtype: np.dtype):
    if dtype.kind == "f":
        return dtype.type(np.nan)
    return dtype.type(0)


def _frame_stamp_ms(path: Path) -> int:
    """The publication stamp of a frame file: its mtime, Unix ms.

    Pinned: a wrfout is published by renaming a finished temporary onto
    its name, so its mtime is when it was written -- a frame written again
    (a resumed run, a history rewind) carries a later stamp than the one
    it replaces."""

    return int(os.stat(path).st_mtime_ns // 1_000_000)


def _create_gridded_variable(zgroup, name: str, meta: Mapping[str, Any], *,
                             axis_size: int, tchunk: int,
                             compressors) -> str:
    """Create one wrfout variable in a gridded group: ``"time"`` (on the
    axis, fill-valued), ``"static"`` (written whole from the first file
    carrying it) or ``"skipped"`` (not numeric; nothing created)."""

    dtype = meta["dtype"]
    if dtype.kind not in "fiu":
        return "skipped"
    dims, shape = meta["dims"], meta["shape"]
    if dims and dims[0] == "Time":
        full = (axis_size,) + tuple(shape[1:])
        chunk_shape = (max(axis_size, tchunk),) + tuple(shape[1:])
        zgroup.create_array(
            name, shape=full,
            chunks=_chunks(chunk_shape, dtype.itemsize, tchunk),
            dtype=dtype, fill_value=_fill_for(dtype),
            compressors=compressors, dimension_names=dims,
            attributes=meta["attrs"])
        return "time"
    array = zgroup.create_array(
        name, shape=shape, dtype=dtype, fill_value=_fill_for(dtype),
        compressors=compressors, dimension_names=dims or None,
        attributes=meta["attrs"])
    with _nc_session(meta["path"]) as nc:
        array[...] = np.asarray(nc.variables[name][...], dtype=dtype)
    return "static"


def _init_gridded(session, group: str, frames: Sequence[Frame],
                  axis: TimeAxis, *, time_chunk: int, compress: bool,
                  zstd_clevel: int, grows: bool = False,
                  interval_known: bool = True) -> dict[str, Any]:
    zarr = _zarr()
    found = _scan_variables(frames)
    root = zarr.open_group(session.store, mode="a")
    zgroup = root.require_group(group)
    compressors = _compressors(compress, zstd_clevel)
    # A growing axis starts at the size of its first write, so its chunk
    # is the requested one rather than one capped at that first size.
    tchunk = max(1, time_chunk if grows else min(time_chunk, axis.size))
    skipped: list[str] = []
    time_vars: list[str] = []
    seed = frames[0].path
    for name, meta in found.items():
        kind = _create_gridded_variable(zgroup, name, meta,
                                        axis_size=axis.size, tchunk=tchunk,
                                        compressors=compressors)
        if kind == "skipped":
            skipped.append(name)
        elif kind == "time":
            time_vars.append(name)
    offsets = ((axis.times() - axis.start) / np.timedelta64(1, "s")) \
        .astype(np.int64)
    coord_chunk = max(axis.size, GROWING_TIME_COORD_CHUNK) if grows \
        else axis.size
    zgroup.create_array(
        "Time", shape=(axis.size,), chunks=(coord_chunk,), dtype="int64",
        fill_value=0, dimension_names=("Time",),
        attributes={"units": axis.units(), "calendar": "proleptic_gregorian",
                    "standard_name": "time",
                    "long_name": "valid time (UTC)"})[:] = offsets
    zgroup.create_array(
        "frame_written", shape=(axis.size,), chunks=(tchunk,), dtype="int8",
        fill_value=0, compressors=compressors, dimension_names=("Time",),
        attributes={"long_name": "1 where a wrfout frame was written",
                    "flag_values": [0, 1],
                    "flag_meanings": "not_written written"})
    _create_stamps(zgroup, axis.size, tchunk, compressors)
    with _nc_session(seed) as nc:
        global_attrs = {f"wrf_{k}": _jsonable(nc.getncattr(k))
                        for k in nc.ncattrs()}
    zgroup.attrs.update({
        "schema": GRIDDED_STORE_SCHEMA, **axis.to_attrs(),
        "time_chunk": tchunk, "source": "wrfout",
        "axis_grows": bool(grows),
        "time_interval_inferred": not interval_known,
        "variables": time_vars,
        "replaced_variables": list(REPLACED_WRF_VARIABLES),
        "skipped_variables": skipped,
        "seed_file": seed.name,
        "compression": ("blosc zstd bitshuffle clevel "
                        f"{int(zstd_clevel)}" if compress else "default"),
        "created_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        **global_attrs,
    })
    return {"created": True, "variables": len(time_vars),
            "skipped_variables": skipped}


def _create_stamps(zgroup, size: int, tchunk: int, compressors):
    return zgroup.create_array(
        FRAME_PUBLISHED, shape=(size,), chunks=(tchunk,), dtype="int64",
        fill_value=0, compressors=compressors, dimension_names=("Time",),
        attributes={"long_name": "publication time of the frame written "
                                 "here (its source file's mtime)",
                    "units": "milliseconds since 1970-01-01 00:00:00"})


def _members(zgroup) -> set[str]:
    return {name for name, _ in zgroup.members()}


def _grown_axis(axis: TimeAxis, times: Sequence[np.datetime64], *,
                adopt: bool = False) -> TimeAxis:
    """``axis`` extended at its end to hold ``times`` (on its grid).

    Pinned: an axis only grows at its end -- a time before its start is
    refused, never prepended -- and keeps its start and interval, except
    that a one-step axis whose interval was not given (``adopt``: the
    group's ``time_interval_inferred``) adopts the greatest common divisor
    of the new times' offsets from its start."""

    values = np.asarray(times, dtype="datetime64[s]")
    if values.size == 0:
        return axis
    first, last = values.min(), values.max()
    if first < axis.start:
        raise IcechunkStoreRefused(
            f"frame at {first}Z precedes the store's axis start "
            f"{axis.start}Z; a growing axis is extended at its end, never "
            "before its start")
    if last <= axis.end:
        return axis
    interval = axis.interval_s
    if axis.size == 1 and adopt:
        offsets = [int((t - axis.start) / np.timedelta64(1, "s"))
                   for t in values if t > axis.start]
        interval = math.gcd(*offsets) if len(offsets) > 1 else offsets[0]
    span = int((last - axis.start) / np.timedelta64(1, "s"))
    if span % interval:
        raise IcechunkStoreRefused(
            f"time {last}Z is not on the store's axis ({axis.start}Z every "
            f"{interval} s)")
    return TimeAxis(axis.start, last, interval)


def _extend_gridded(zgroup, old: TimeAxis, new: TimeAxis) -> None:
    """Resize every ``Time`` array of a gridded group from ``old`` to
    ``new`` (same start) and rewrite the ``Time`` coordinate."""

    members = _members(zgroup)
    names = list(zgroup.attrs["variables"]) + ["frame_written"]
    if FRAME_PUBLISHED in members:
        names.append(FRAME_PUBLISHED)
    for name in names:
        array = zgroup[name]
        array.resize((new.size,) + tuple(array.shape[1:]))
    coord = zgroup["Time"]
    coord.resize((new.size,))
    # Only the new steps: the old ones keep their offsets (a one-step
    # axis that adopted its interval has only offset 0).
    offsets = ((new.times() - new.start) / np.timedelta64(1, "s")) \
        .astype(np.int64)
    coord[old.size:] = offsets[old.size:]
    zgroup.attrs.update(new.to_attrs())


def _add_gridded_variables(zgroup, frames: Sequence[Frame], *,
                           axis: TimeAxis, compress: bool,
                           zstd_clevel: int) -> list[str]:
    """Create the variables ``frames`` carry that the group does not hold
    yet (a field first written after the group's seed frame), the way the
    group's creation would have had it seen them: NaN/zero-filled on the
    axis, attributes from the first file carrying them."""

    members = _members(zgroup)
    attrs = dict(zgroup.attrs)
    skipped = list(attrs.get("skipped_variables") or [])
    known = members | set(skipped) | set(REPLACED_WRF_VARIABLES)
    found = {name: meta for name, meta in _scan_variables(frames).items()
             if name not in known}
    if not found:
        return []
    compressors = _compressors(compress, zstd_clevel)
    tchunk = max(1, int(attrs.get("time_chunk") or 1))
    variables = list(attrs["variables"])
    added: list[str] = []
    for name, meta in found.items():
        kind = _create_gridded_variable(zgroup, name, meta,
                                        axis_size=axis.size, tchunk=tchunk,
                                        compressors=compressors)
        if kind == "skipped":
            skipped.append(name)
            continue
        if kind == "time":
            variables.append(name)
        added.append(name)
    zgroup.attrs.update({"variables": variables,
                         "skipped_variables": skipped})
    return added


def _write_frame_values(jobs: Iterable[tuple[Any, int, np.ndarray]],
                        threads: int) -> None:
    """``array[t] = values`` for every job, ``threads`` at a time.

    Reading stays on the calling thread (netCDF/HDF5 is not thread-safe);
    the chunk encoding and the store writes of different arrays run
    concurrently, at most ``2 * threads`` arrays held in memory."""

    if threads <= 1:
        for array, t, values in jobs:
            array[t] = values
        return
    from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait

    pending: set = set()
    with ThreadPoolExecutor(max_workers=threads,
                            thread_name_prefix="icechunk-write") as pool:
        try:
            for array, t, values in jobs:
                if len(pending) >= 2 * threads:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        future.result()
                pending.add(pool.submit(array.__setitem__, t, values))
            for future in pending:
                future.result()
        finally:
            for future in pending:
                future.cancel()


def write_gridded_segment(store, paths: Iterable[str | os.PathLike], *,
                          group: str, axis: TimeAxis | None = None,
                          branch: str = "main", job_id: str | None = None,
                          segment_id: str | None = None,
                          window: tuple[Any, Any] | None = None,
                          on_existing: str = "skip", time_chunk: int = 1,
                          compress: bool = True, zstd_clevel: int = 5,
                          extend_axis: bool = False,
                          add_variables: bool = False,
                          write_threads: int = 1,
                          axis_interval_known: bool = True,
                          message: str | None = None,
                          policy: RetryPolicy | None = None,
                          before_commit: Callable[[int], None] | None = None,
                          **storage_options) -> dict[str, Any]:
    """Write one segment's wrfout frames for one domain into ``group``
    and commit (module docstring).

    ``axis`` is required when the group does not exist yet and must match
    it when it does (``None`` reads it from the group).  ``window`` =
    ``(first, end)`` keeps frames with ``first <= time < end`` (drop a
    segment's spin-up or its overlap with the next); frames off the axis
    are refused.

    The live export (:mod:`woof.icechunk_export`) writes one frame per
    call with three switches the parallel hindcast leaves off:
    ``extend_axis`` makes the axis grow at its end to hold later frames
    (``axis`` is then only the creating axis; a group created this way is
    marked ``axis_grows``; with ``axis_interval_known=False`` its
    interval is a placeholder the second frame replaces, marked
    ``time_interval_inferred``), ``add_variables`` creates a field the group
    does not hold yet instead of dropping it, and ``on_existing="newer"``
    rewrites a frame already written only when its source file is newer
    (``frame_published_unix_ms``) -- a later run wins, the same run is a
    no-op.  ``write_threads`` encodes and stores that many arrays at once.
    """

    if on_existing not in ("overwrite", "skip", "newer"):
        raise IcechunkStoreRefused(
            f"on_existing {on_existing!r} is not overwrite, skip or newer")
    if not group or group.strip("/") != group:
        raise IcechunkStoreRefused(
            f"group {group!r} must be a non-empty relative path")
    if int(time_chunk) < 1:
        raise IcechunkStoreRefused("time_chunk must be at least 1 frame")
    every = _all_frames(paths)
    frames = _select_frames(every, window)
    windowed = len(every) - len(frames)
    if not frames:
        raise IcechunkStoreRefused("no wrfout frames to write")
    stamps_of = {path: _frame_stamp_ms(path)
                 for path in dict.fromkeys(f.path for f in frames)}
    frame_times = [f.time for f in frames]
    repo = open_repository(store, **storage_options)

    def apply(session) -> dict[str, Any]:
        zgroup = _open_group(session, group)
        created: dict[str, Any] = {"created": False}
        extended = False
        if zgroup is None:
            if axis is None:
                raise IcechunkStoreRefused(
                    f"group {group!r} does not exist; give its time axis")
            group_axis = (_grown_axis(axis, frame_times,
                                      adopt=not axis_interval_known)
                          if extend_axis else axis)
            group_axis.index_of(frame_times)
            created = _init_gridded(session, group, frames, group_axis,
                                    time_chunk=int(time_chunk),
                                    compress=compress,
                                    zstd_clevel=zstd_clevel,
                                    grows=bool(extend_axis),
                                    interval_known=bool(axis_interval_known))
            zgroup = _open_group(session, group)
        else:
            attrs = dict(zgroup.attrs)
            if attrs.get("schema") != GRIDDED_STORE_SCHEMA:
                raise IcechunkStoreRefused(
                    f"group {group!r} is not a {GRIDDED_STORE_SCHEMA} store")
            group_axis = TimeAxis.from_attrs(attrs)
            if extend_axis:
                grown = _grown_axis(
                    group_axis, frame_times,
                    adopt=bool(attrs.get("time_interval_inferred")))
                if grown != group_axis:
                    _extend_gridded(zgroup, group_axis, grown)
                    group_axis, extended = grown, True
            elif axis is not None and axis != group_axis:
                raise IcechunkStoreRefused(
                    f"group {group!r} has time axis "
                    f"{group_axis.to_attrs()}, not {axis.to_attrs()}")
        index = group_axis.index_of(frame_times)
        # Only this segment's entries are read: a growing group's masks
        # are one chunk per frame, and reading them whole made every
        # one-frame commit cost a read per frame already written.
        where = (_selector(index),)
        written = np.asarray(
            zgroup["frame_written"].get_orthogonal_selection(where))
        members = _members(zgroup)
        if FRAME_PUBLISHED not in members and on_existing == "newer":
            _create_stamps(zgroup, group_axis.size,
                           max(1, int(zgroup.attrs.get("time_chunk") or 1)),
                           _compressors(compress, zstd_clevel))
            members.add(FRAME_PUBLISHED)
        stamp_array = zgroup[FRAME_PUBLISHED] \
            if FRAME_PUBLISHED in members else None
        stamps = (np.asarray(stamp_array.get_orthogonal_selection(where))
                  if stamp_array is not None
                  else np.zeros(len(index), dtype=np.int64))
        todo: list[tuple[Frame, int]] = []
        replaced = 0
        for n, (f, i) in enumerate(zip(frames, index)):
            i = int(i)
            if written[n] != 1 or on_existing == "overwrite":
                todo.append((f, i))
            elif on_existing == "newer" and stamps_of[f.path] > stamps[n]:
                todo.append((f, i))
                replaced += 1
        skipped = len(frames) - len(todo)
        added = (_add_gridded_variables(
                     zgroup, [f for f, _ in todo], axis=group_axis,
                     compress=compress, zstd_clevel=zstd_clevel)
                 if add_variables and todo else [])
        stored = list(zgroup.attrs["variables"])
        dropped: set[str] = set()
        absent: set[str] = set()
        arrays = {name: zgroup[name] for name in stored}
        lock = _nc_lock()
        handle: list = []

        def jobs():
            # The netCDF lock is held for the open, each read and the
            # close, and released while the values are encoded and stored
            # (``yield``): HDF5 is never entered by two threads, and a
            # frame never holds it for a whole write.  An idle open
            # handle enters nothing.
            for frame, t in todo:
                with lock:
                    nc = _open_nc(frame.path)
                    handle.append(nc)
                    present = {name: tuple(variable.dimensions)
                               for name, variable in nc.variables.items()}
                for name in stored:
                    dims = present.get(name)
                    if not dims or dims[0] != "Time":
                        absent.add(name)
                        continue
                    array = arrays[name]
                    with lock:
                        values = np.asarray(nc.variables[name][frame.index],
                                            dtype=array.dtype)
                    if values.shape != tuple(array.shape[1:]):
                        raise IcechunkStoreRefused(
                            f"{frame.path.name} {name} has shape "
                            f"{values.shape}, the store {array.shape[1:]}")
                    yield array, t, values
                for name, dims in present.items():
                    if name in stored or name in REPLACED_WRF_VARIABLES:
                        continue
                    if dims[:1] == ("Time",):
                        dropped.add(name)
                with lock:
                    handle.pop().close()

        try:
            # Concurrent writes are safe only when no two of them share a
            # chunk: with one frame per time chunk every (array, frame)
            # write is whole chunks of its own.
            tchunk = max(1, int(zgroup.attrs.get("time_chunk") or 1))
            _write_frame_values(jobs(), max(1, int(write_threads))
                                if tchunk == 1 else 1)
        finally:
            while handle:
                with lock:
                    handle.pop().close()
        for frame, t in todo:
            zgroup["frame_written"][t] = 1
            if stamp_array is not None:
                stamp_array[t] = stamps_of[frame.path]
        return {**created, "frames_written": len(todo),
                "time_chunk": int(zgroup.attrs.get("time_chunk") or 1),
                "frames_skipped": skipped, "frames_replaced": replaced,
                "dropped": sorted(dropped), "absent": sorted(absent),
                "added": added, "extended": extended, "axis": group_axis}

    first, last = str(frames[0].time) + "Z", str(frames[-1].time) + "Z"
    result = commit_with_retry(
        repo, branch=branch, apply=apply,
        message=message or (
            f"woof energy icechunk: job {job_id or '-'} segment "
            f"{segment_id or '-'} wrote {len(frames)} wrfout frames "
            f"into {group} ({first}..{last})"),
        metadata=_job_metadata(job_id, segment_id, group, {
            "action": "write-gridded", "first_time": first,
            "last_time": last, "frames": len(frames),
            "files": sorted({f.path.name for f in frames}),
        }),
        policy=policy, before_commit=before_commit,
        lock=commit_lock(store))
    info = result.info
    group_axis = info.get("axis")
    index = group_axis.index_of([f.time for f in frames]) \
        if group_axis is not None else np.zeros(0, np.int64)
    tchunk = max(1, int(info.get("time_chunk") or 1))
    aligned = bool(index.size == 0
                   or (index[0] % tchunk == 0
                       and ((index[-1] + 1) % tchunk == 0
                            or index[-1] + 1 == group_axis.size)))
    return {
        "schema": WRITE_SCHEMA, "store": str(store), "group": group,
        "branch": branch, "snapshot": result.snapshot,
        "committed": result.committed, "job_id": job_id,
        "segment_id": segment_id, "first_time": first, "last_time": last,
        "frames": len(frames), "frames_windowed_out": windowed,
        "frames_written": int(info.get("frames_written", 0)),
        "frames_skipped": int(info.get("frames_skipped", 0)),
        "frames_replaced": int(info.get("frames_replaced", 0)),
        "created": bool(info.get("created")),
        "axis_extended": bool(info.get("extended")),
        "added_variables": info.get("added", []),
        "dropped_variables": info.get("dropped", []),
        "absent_variables": info.get("absent", []),
        "chunk_aligned": aligned,
        "attempts": result.attempts, "rebases": result.rebases,
        "rewrites": result.rewrites,
        "compression": ({"codec": "blosc", "cname": "zstd",
                         "shuffle": "bitshuffle", "clevel": int(zstd_clevel)}
                        if compress else None),
    }


# --------------------------------------------------------------------------
# verification


def _coverage(array, axis: TimeAxis, time_chunk: int) -> dict[str, Any]:
    """Written-step count and first/last written time, read a block of
    time chunks at a time."""

    tchunk = max(1, int(time_chunk))
    block = tchunk * max(1, 4096 // tchunk)
    count = 0
    first = last = None
    for start in range(0, array.shape[0], block):
        values = np.asarray(array[start:start + block])
        rows = values.reshape(values.shape[0], -1).max(axis=1) \
            if values.ndim > 1 else values
        hit = np.nonzero(rows == 1)[0]
        count += int(hit.size)
        if hit.size:
            if first is None:
                first = start + int(hit[0])
            last = start + int(hit[-1])
    times = axis.times()
    return {"steps": axis.size, "steps_written": count,
            "first_written": None if first is None else str(times[first]) + "Z",
            "last_written": None if last is None else str(times[last]) + "Z"}


def _walk_groups(group, prefix: str = ""):
    for name, member in group.members():
        path = f"{prefix}{name}"
        if hasattr(member, "members"):
            yield path, member
            yield from _walk_groups(member, path + "/")


def verify_repository(store, *, branch: str = "main",
                      group: str | None = None, coverage: bool = True,
                      **storage_options) -> dict[str, Any]:
    """Open an existing repository (anonymously with ``anonymous=True``)
    and report its branch tip snapshot, its last commit and, for each
    site or gridded store group (or just ``group``), the axis and how much
    of it is written."""

    zarr = _zarr()
    ic = _icechunk()
    storage = storage_from_url(store, **storage_options) \
        if isinstance(store, (str, os.PathLike)) else store
    if not ic.Repository.exists(storage):
        raise IcechunkStoreRefused(f"no Icechunk repository at {store}")
    repo = ic.Repository.open(storage)
    try:
        snapshot = repo.lookup_branch(branch)
    except ic.IcechunkError as error:
        raise IcechunkStoreRefused(
            f"repository has no branch {branch!r}") from error
    tip = next(iter(repo.ancestry(snapshot_id=snapshot)))
    session = repo.readonly_session(branch=branch)
    root = zarr.open_group(session.store, mode="r")
    if group is not None:
        target = _open_group(session, group, mode="r")
        if target is None:
            raise IcechunkStoreRefused(f"no group {group!r} in the store")
        candidates = [(group, target)]
    else:
        candidates = list(_walk_groups(root))
    groups: dict[str, Any] = {}
    for path, member in candidates:
        attrs = dict(member.attrs)
        schema = attrs.get("schema")
        if schema not in (SITE_STORE_SCHEMA, GRIDDED_STORE_SCHEMA):
            continue
        axis = TimeAxis.from_attrs(attrs)
        entry: dict[str, Any] = {"schema": schema, **axis.to_attrs(),
                                 "variables": len(attrs.get("variables")
                                                  or [])}
        if schema == SITE_STORE_SCHEMA:
            entry["sites"] = attrs.get("site_count")
            mask = "written"
        else:
            mask = "frame_written"
        if coverage:
            entry.update(_coverage(member[mask], axis,
                                   int(attrs.get("time_chunk") or 1)))
        groups[path] = entry
    return {
        "schema": VERIFY_SCHEMA, "store": str(store), "branch": branch,
        "snapshot": str(snapshot),
        "last_commit": {"message": tip.message,
                        "written_at": tip.written_at.isoformat(),
                        "metadata": _jsonable(dict(tip.metadata or {}))},
        "groups": groups,
    }


# --------------------------------------------------------------------------
# woof energy icechunk


def _storage_options(args) -> dict[str, Any]:
    return {"region": getattr(args, "region", None),
            "endpoint_url": getattr(args, "endpoint_url", None),
            "allow_http": bool(getattr(args, "allow_http", False)),
            "anonymous": bool(getattr(args, "anonymous", False))}


def main(args) -> int:
    """``woof energy icechunk init|verify``: JSON summary on stdout."""

    command = getattr(args, "icechunk_command", None)
    try:
        if command == "init":
            axis = TimeAxis(args.start, args.end, parse_interval(args.interval))
            summary = init_site_store(
                args.repo, sites=Path(args.sites), axis=axis,
                heights_m=args.heights_m, variables=args.vars,
                group=args.group, branch=args.branch,
                time_chunk=args.time_chunk, site_chunk=args.site_chunk,
                zstd_clevel=args.zstd_clevel, job_id=args.job_id,
                **_storage_options(args))
        elif command == "verify":
            summary = verify_repository(
                args.repo, branch=args.branch, group=args.group,
                coverage=not args.no_coverage, **_storage_options(args))
        else:
            print("woof energy icechunk: choose init or verify",
                  file=sys.stderr)
            return 2
    except (IcechunkStoreRefused, ContractError) as error:
        print(f"woof energy icechunk {command}: refused: {error}",
              file=sys.stderr)
        return 2
    except OSError as error:
        print(f"woof energy icechunk {command}: failed: {error}",
              file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, default=str))
    return 0


__all__ = [
    "SITE_STORE_SCHEMA", "GRIDDED_STORE_SCHEMA", "WRITE_SCHEMA",
    "VERIFY_SCHEMA", "SCHEMES", "MAX_CHUNK_BYTES", "MANIFEST_SPLIT_CHUNKS",
    "IcechunkStoreRefused", "TimeAxis", "RetryPolicy", "CommitResult",
    "Frame", "FRAME_PUBLISHED", "GROWING_TIME_COORD_CHUNK",
    "is_store_url", "storage_from_url", "repository_config",
    "open_repository", "parse_interval", "commit_with_retry",
    "init_site_store", "write_site_region", "wrfout_frames",
    "write_gridded_segment", "verify_repository", "main",
]
