"""Fetch each forcing cycle once and share it with every job of that cycle.

A year-long wind-farm hindcast runs every regional cluster from the same
global cycle: about 440 clusters x 1,460 six-hourly cycles.  Left alone,
each job's ``woof go`` downloads its own copy of the cycle's global files
(``woof fetch`` into the run's managed download folder), so one cycle is
downloaded 440 times.  This module moves the download to one **prefetch**
per (source, cycle) into a shared object-store **cache**, and has each
forecast job **mirror** only the objects its own fetch request names into
the folder ``woof go`` will use, where ``woof fetch`` then reuses them
without asking any host.

Pinned here
-----------

**What is shared.**  A source whose fetch is a table route
(:func:`woof.fetch_routes.route_ids`: Met Office global, ECMWF open data,
HRRR native, RAP, ICON, ...) moves whole published objects and its request
carries no area: every cluster of a cycle asks for byte-identical files, so
those are cached.  The four hand-written transports
(:data:`woof.fetch_routes.LEGACY_ROUTE_SOURCES`: ``gfs``, ``gdas``,
``hrrr``, ``era5``) are not: their fetch crops to the cluster's area (the
NOMADS grib filter, the ERA5 retrieval) or binds the area in its receipt
(``woof.fetch.check_prior_request``), so their jobs keep fetching for
themselves, and the job document says so (``forcing.mode = "direct"``).
:func:`shareability` is the one answer.

**No lat/lon cropping.**  The published objects are global GRIB2/NetCDF-4
grids; a lat/lon window of one is not a byte range of it, so cutting one
would mean re-encoding, which changes what the decoders read.  The only
subsetting done is by **object**: a cache entry may hold more objects than
a job asks for (a longer window, another job's leads), and the job takes
exactly the objects of its own resolved fetch plan
(:func:`woof.fetch_routes.resolve_request`).  Decoding is untouched: the
job's preparation reads the same bytes from the same relative paths it
would have downloaded.

**Cache layout** (``file:///abs/dir``, ``s3://bucket/prefix`` or
``gs://bucket/prefix``, the backends of :mod:`woof.energy.workqueue`; a
bare path means ``file://``)::

    entries/<source>/<YYYYMMDDTHH>Z/<key16>.json   woof-energy.forcing-cache.v1
    blobs/sha256/<aa>/<sha256>                     file bytes, content-addressed

The entry key is the sha256 of the canonical JSON of the **forcing
request** (:func:`forcing_request`: source, cycle, hours, cadence,
forecast start hour, member, pinned host).  Blobs are named by their
content, so a blob is either the right bytes or absent; uploads skip blobs
already present at the right size, re-runs are idempotent, and two
prefetchers of one cycle cannot corrupt each other (the entry document is
written create-only, last, after every blob).  The entry lists every
object the fetch moved, every file of a declared donor's folder, and the
route manifest and ``SHA256SUMS`` for provenance (relative path, sha256,
bytes, kind ``object`` / ``donor`` / ``receipt``); the rest of the fetch's
handoff names the prefetch's scratch folder and is rewritten by each job's
own fetch, so it is not kept.  Each object carries its route manifest
entry (name, bytes, sha256, url, endpoint); the entry also records the
fetch argv, the route table sha256, how long the fetch took and when it
was retrieved.

**Prefetch** (:func:`prefetch`) runs exactly the ``woof fetch`` a job's
``woof go`` fetch stage would run for that request (the argv
``woof.runplan._fetch_arguments_from_hints`` spells from a ``[fetch]``
table) into a scratch folder, checks the route manifest it wrote, uploads,
and deletes the scratch.  An entry already complete in the cache is not
fetched again (``status: cached``) unless ``force``.

**Mirror** (:func:`mirror_job`) runs after a job is materialized and before
``woof energy run``: for each ``woof go`` step of the plan it reads the
config's ``[fetch]`` table, computes the managed download folder ``woof go``
will choose (``woof.go_cli.managed_download_dir``, the same call with the
same arguments), resolves that request's fetch plan offline, copies each
planned object from the cache (sha256 verified before it is placed; a mismatch
is not placed), copies a declared donor's subfolder whole, and writes the
per-object receipts ``woof fetch`` itself writes
(:func:`woof.fetch_routes.adopt_verified_objects`).  ``woof fetch`` then
finds every object verified and asks no host (``request_cached``); an
object the cache lacks is simply fetched by the job as before, and the
mirror receipt (``forcing-mirror.json`` in the job directory,
``woof-energy.forcing-mirror.v1``) names each one (``fallback``).  A
config whose request differs from the entry's (another source, cycle,
cadence, start hour or member, or more hours than the entry holds) is not
mirrored at all, and says why; the pinned host is not compared
(:func:`family`).

Python boundary: file copies, sha256 and a few JSON documents per job; no
field data is decoded.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse

from woof.energy.contracts import ContractError, _write_json
# The queue tries ids with this prefix first; one spelling for both.
from woof.energy.workqueue import \
    PREFETCH_ID_PREFIX as _QUEUE_PREFETCH_ID_PREFIX

CACHE_SCHEMA = "woof-energy.forcing-cache.v1"
MIRROR_SCHEMA = "woof-energy.forcing-mirror.v1"
PREFETCH_SCHEMA = "woof-energy.forcing-prefetch.v1"

#: Job kinds a ``woof-energy.hindcast-job.v1`` document may declare; a
#: document without ``kind`` is a forecast job (every job before this unit).
PREFETCH_KIND = "prefetch"
FORECAST_KIND = "forecast"
#: Prefetch job ids start with this (the queue tries them first).
PREFETCH_ID_PREFIX = _QUEUE_PREFETCH_ID_PREFIX

#: Receipts, in the job directory.
MIRROR_NAME = "forcing-mirror.json"
PREFETCH_RECEIPT_NAME = "prefetch.json"

ENTRY_PREFIX = "entries"
BLOB_PREFIX = "blobs/sha256"
SCHEMES = ("file", "s3", "gs")

#: ``forcing.mode`` values a forecast job document carries.
MODE_SHARED = "shared"
MODE_DIRECT = "direct"

#: Why each legacy (hand-written) transport is fetched per job.
_LEGACY_REASONS = {
    "gfs": "the gfs fetch crops to the cluster's area at the NOMADS grib "
           "filter, and its archive full-file receipt binds that area; "
           "each job fetches its own",
    "gdas": "the gdas fetch crops to the cluster's area at the NOMADS grib "
            "filter, and its archive full-file receipt binds that area; "
            "each job fetches its own",
    "hrrr": "the hrrr fetch's receipt binds the requested area "
            "(woof.fetch.check_prior_request), so another cluster's folder "
            "is refused; each job fetches its own",
    "era5": "the ERA5 retrieval is cropped to the cluster's area (and its "
            "forcing is a declared input); each job fetches its own",
}

_CHUNK = 8 * 1024 * 1024

#: Top-level receipts of a fetch folder kept in an entry for provenance
#: (the route manifest and its sums); the rest of the handoff names the
#: prefetch's scratch folder and is rewritten by each job's own fetch.
_KEPT_RECEIPTS = ("fetch-manifest.json", "SHA256SUMS")


class ForcingCacheRefused(ContractError):
    """The forcing cache will not prefetch, mirror or open this as asked."""


# --------------------------------------------------------------------------
# small helpers


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical(document: Any) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _encode(document: Mapping[str, Any]) -> bytes:
    return (json.dumps(document, sort_keys=True, indent=1) + "\n").encode(
        "utf-8")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_hashing(source: Path, dest: Path) -> str:
    """Copy ``source`` to ``dest``, hashing the bytes as they pass."""

    digest = hashlib.sha256()
    with open(source, "rb") as reader, open(dest, "wb") as writer:
        for block in iter(lambda: reader.read(_CHUNK), b""):
            digest.update(block)
            writer.write(block)
    return digest.hexdigest()


def _matches(path: Path, size: int, sha256: str) -> bool:
    return (path.is_file() and path.stat().st_size == size
            and _sha256_path(path) == sha256)


def _safe_relpath(text: str) -> str:
    """A cache entry's relative path, refused if it could leave its folder."""

    path = PurePosixPath(text)
    if (not text or path.is_absolute() or ".." in path.parts
            or "\\" in text or text != path.as_posix()):
        raise ForcingCacheRefused(f"cache entry names an unsafe path "
                                  f"{text!r}")
    return text


def blob_key(sha256: str) -> str:
    """Where a file with this sha256 lives in the cache."""

    if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
        raise ForcingCacheRefused(f"{sha256!r} is not a sha256 hex digest")
    return f"{BLOB_PREFIX}/{sha256[:2]}/{sha256}"


# --------------------------------------------------------------------------
# what is shared


def shareability(source: str) -> tuple[str, str]:
    """``("shared", why)`` for a table route, ``("direct", why)`` otherwise."""

    from woof import fetch_routes

    canonical = fetch_routes.canonical_source(str(source))
    if canonical in fetch_routes.route_ids():
        return MODE_SHARED, (
            f"{canonical} is a table route: whole published objects, no "
            "area in the request, byte-identical for every cluster of a "
            "cycle")
    if canonical in _LEGACY_REASONS:
        return MODE_DIRECT, _LEGACY_REASONS[canonical]
    return MODE_DIRECT, (f"{canonical} has no table fetch route; each job "
                         "acquires its own inputs")


def forcing_request(source: str, cycle: str | datetime, hours: int, *,
                    cadence: int | None = None, forecast_start_hour: int = 0,
                    member: str | None = None,
                    transport: str | None = None) -> dict:
    """The canonical forcing request a cache entry is keyed on."""

    from woof import fetch_routes

    if isinstance(cycle, datetime):
        cycle = cycle.strftime("%Y-%m-%dT%H")
    text = str(cycle).strip()
    try:
        moment = datetime.strptime(text[:13], "%Y-%m-%dT%H")
    except ValueError as error:
        raise ForcingCacheRefused(f"forcing cycle {cycle!r} is not "
                                  "YYYY-MM-DDTHH") from error
    if isinstance(hours, bool) or int(hours) != hours or int(hours) <= 0:
        raise ForcingCacheRefused(f"forcing hours {hours!r} must be a "
                                  "positive whole number")
    return {"source": fetch_routes.canonical_source(str(source)),
            "cycle": moment.strftime("%Y-%m-%dT%H"),
            "hours": int(hours),
            "cadence": None if cadence is None else int(cadence),
            "forecast_start_hour": int(forecast_start_hour or 0),
            "member": member or None,
            "transport": transport or None}


def family(request: Mapping[str, Any]) -> dict:
    """Everything of a request but its length and its host: requests of
    one family fetch the same objects (the same keys, at the same relative
    paths, whichever endpoint serves them) for the leads they share.  A
    job whose fetch pins a host (``[fetch] transport``, or the hosted
    ``WOOF_FETCH_POLICY``) still takes the bytes another host served: its
    own fetch re-verifies each one against its sha256."""

    return {k: v for k, v in request.items()
            if k not in ("hours", "transport")}


def predicted_request(source: str, run_start: datetime, run_hours: float
                      ) -> dict:
    """The request a segment's ``[fetch]`` table will make, before it is
    planned: the planner's own ``fetch_window`` and default member
    (:func:`woof.energy.plan_wrf_nests._fetch_and_case`)."""

    from woof import domain_wizard as dw
    from woof import fetch_routes
    from woof.fetch import parse_cycle

    # The planner's own calls, with its own arguments: the cycle goes
    # through parse_cycle (which may resolve it for the source), the window
    # through fetch_window.
    cycle = parse_cycle(run_start.strftime("%Y-%m-%dT%H"), source)
    cadence, hours = dw.fetch_window(source, float(run_hours), 0)
    member = None
    route = fetch_routes.table_route(fetch_routes.canonical_source(source))
    if route is not None and route.members is not None:
        member = fetch_routes.resolve_member(route, None)[0]
    return forcing_request(source, cycle, hours, cadence=cadence,
                           member=member)


def request_key(request: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical({"schema": CACHE_SCHEMA,
                                      "request": dict(request)})).hexdigest()


def entry_key(request: Mapping[str, Any]) -> str:
    """The object key of a request's cache entry."""

    cycle = datetime.strptime(request["cycle"], "%Y-%m-%dT%H")
    return (f"{ENTRY_PREFIX}/{request['source']}/{cycle:%Y%m%dT%H}Z/"
            f"{request_key(request)[:16]}.json")


def fetch_hints(request: Mapping[str, Any]) -> dict:
    """The ``[fetch]`` table that makes this request (keys = fetch flags)."""

    hints: dict[str, Any] = {"source": request["source"],
                             "cycle": request["cycle"],
                             "hours": int(request["hours"])}
    if request.get("cadence") is not None:
        hints["cadence"] = int(request["cadence"])
    if request.get("forecast_start_hour"):
        hints["forecast_start_hour"] = int(request["forecast_start_hour"])
    if request.get("member"):
        hints["member"] = request["member"]
    if request.get("transport"):
        hints["transport"] = request["transport"]
    return hints


def fetch_argv(request: Mapping[str, Any], out: Path | str) -> list[str]:
    """``woof fetch`` arguments (without the program) for this request: the
    argv woof go's fetch stage spells from the same ``[fetch]`` table."""

    from woof.runplan import _fetch_arguments_from_hints

    return list(_fetch_arguments_from_hints(fetch_hints(request),
                                            out=Path(out)))


def config_request(fetch_table: Mapping[str, Any]) -> dict:
    """The forcing request a config's ``[fetch]`` table makes under
    ``woof go`` (no ``--transport`` flag: the table's own pin, if any)."""

    from woof.fetch import parse_cycle
    from woof.go_cli import pinned_transport

    pinned, _ = pinned_transport(dict(fetch_table), None)
    cycle = parse_cycle(str(fetch_table["cycle"]), str(fetch_table["source"]))
    return forcing_request(
        str(fetch_table["source"]), cycle, int(fetch_table["hours"]),
        cadence=fetch_table.get("cadence"),
        forecast_start_hour=int(fetch_table.get("forecast_start_hour") or 0),
        member=fetch_table.get("member"), transport=pinned)


# --------------------------------------------------------------------------
# the cache store


def normalize_url(url: str | os.PathLike) -> str:
    """``file:///abs``, ``s3://b/p`` or ``gs://b/p``; a bare path becomes
    an absolute ``file://`` URL.  Credentials in a URL are refused."""

    text = os.fspath(url).strip()
    if not text:
        raise ForcingCacheRefused("the forcing cache URL is empty")
    if "://" not in text:
        return "file://" + str(Path(text).expanduser().resolve())
    parsed = urlparse(text)
    scheme = parsed.scheme.lower()
    if scheme == "gcs":
        scheme = "gs"
    if scheme not in SCHEMES:
        raise ForcingCacheRefused(
            f"forcing cache URL scheme {parsed.scheme!r} is not one of "
            f"{', '.join(s + '://' for s in SCHEMES)} (a bare path means "
            "file://)")
    if parsed.username or parsed.password:
        raise ForcingCacheRefused("credentials in the forcing cache URL are "
                                  "refused; use the environment, a profile "
                                  "or a role")
    if scheme == "file":
        if parsed.netloc not in ("", "localhost") or not parsed.path.startswith("/"):
            raise ForcingCacheRefused(f"file:// cache URLs take an absolute "
                                      f"path (file:///abs/dir); got {text!r}")
        return "file://" + parsed.path.rstrip("/")
    if not parsed.netloc:
        raise ForcingCacheRefused(f"{scheme}:// cache URL names no bucket: "
                                  f"{text!r}")
    return f"{scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"


class CacheStore:
    """A forcing cache over one :mod:`woof.energy.workqueue` backend, with
    whole-file transfers for blobs (no file is held in memory on the
    ``file://``, S3 or GCS backends).

    The cloud SDKs are imported only when an ``s3://``/``gs://`` cache is
    opened, and their absence is the queue's named refusal
    (``QueueBackendUnavailable``) naming the package to install."""

    def __init__(self, backend):
        self.backend = backend
        self.url = backend.url

    @classmethod
    def open(cls, url: str, **clients) -> "CacheStore":
        from woof.energy.workqueue import open_store

        return cls(open_store(normalize_url(url), **clients))

    # -- documents --------------------------------------------------------

    def get_json(self, key: str) -> dict | None:
        found = self.backend.get(key)
        if found is None:
            return None
        try:
            document = json.loads(found[0].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ForcingCacheRefused(f"{self.url}/{key} is not JSON: "
                                      f"{error}") from error
        if not isinstance(document, dict):
            raise ForcingCacheRefused(f"{self.url}/{key} is not a JSON "
                                      "object")
        return document

    def put_json_if_absent(self, key: str, document: Mapping[str, Any]
                           ) -> bool:
        from woof.energy.workqueue import PreconditionFailed

        try:
            self.backend.put_if_absent(key, _encode(document))
            return True
        except PreconditionFailed:
            return False

    def put_json(self, key: str, document: Mapping[str, Any]) -> None:
        self.backend.put(key, _encode(document))

    # -- blobs ------------------------------------------------------------

    def size(self, key: str) -> int | None:
        """Bytes of ``key``, or None when absent."""

        from woof.energy.workqueue import FileStore, GCSStore, S3Store

        backend = self.backend
        if isinstance(backend, FileStore):
            path = backend._path(key)
            return path.stat().st_size if path.is_file() else None
        if isinstance(backend, S3Store):
            try:
                head = backend.client.head_object(Bucket=backend.bucket,
                                                  Key=backend._key(key))
            except Exception as error:  # botocore ClientError
                from woof.energy.workqueue import _s3_error_code

                code, status = _s3_error_code(error)
                if code in ("NoSuchKey", "404", "NotFound") or status == 404:
                    return None
                raise
            return int(head["ContentLength"])
        if isinstance(backend, GCSStore):
            blob = backend.bucket.get_blob(backend._key(key))
            return None if blob is None else int(blob.size)
        found = backend.get(key)
        return None if found is None else len(found[0])

    def put_file(self, key: str, path: Path) -> None:
        from woof.energy.workqueue import FileStore, GCSStore, S3Store

        backend = self.backend
        if isinstance(backend, FileStore):
            target = backend._path(key)
            target.parent.mkdir(parents=True, exist_ok=True)
            staging = backend.root / ".tmp"
            staging.mkdir(parents=True, exist_ok=True)
            part = staging / f"{target.name}.{os.getpid()}.part"
            shutil.copyfile(path, part)
            os.replace(part, target)
        elif isinstance(backend, S3Store):
            backend.client.upload_file(str(path), backend.bucket,
                                       backend._key(key))
        elif isinstance(backend, GCSStore):
            backend.bucket.blob(backend._key(key)).upload_from_filename(
                str(path))
        else:
            backend.put(key, Path(path).read_bytes())

    def get_file(self, key: str, dest: Path, *, sha256: str,
                 size: int | None = None) -> str | None:
        """Copy ``key`` to ``dest`` if its bytes hash to ``sha256``.

        Lands at ``dest`` only when verified; returns None then, else why
        it was not placed (absent, wrong size, wrong digest)."""

        from woof.energy.workqueue import FileStore, GCSStore, S3Store

        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".cache-part")
        backend = self.backend
        digest = None
        try:
            if isinstance(backend, FileStore):
                source = backend._path(key)
                if not source.is_file():
                    return "absent from the cache"
                digest = _copy_hashing(source, part)
            elif isinstance(backend, S3Store):
                if self.size(key) is None:
                    return "absent from the cache"
                backend.client.download_file(backend.bucket,
                                             backend._key(key), str(part))
            elif isinstance(backend, GCSStore):
                blob = backend.bucket.get_blob(backend._key(key))
                if blob is None:
                    return "absent from the cache"
                blob.download_to_filename(str(part))
            else:
                found = backend.get(key)
                if found is None:
                    return "absent from the cache"
                part.write_bytes(found[0])
            got_size = part.stat().st_size
            if size is not None and got_size != size:
                return f"cache copy is {got_size} bytes, not {size}"
            if digest is None:
                digest = _sha256_path(part)
            if digest != sha256:
                return f"cache copy hashes to {digest[:12]}..., not {sha256[:12]}..."
            os.replace(part, dest)
            return None
        finally:
            part.unlink(missing_ok=True)


def open_cache(url: str | CacheStore, **clients) -> CacheStore:
    return url if isinstance(url, CacheStore) else CacheStore.open(url,
                                                                  **clients)


def read_entry(store: CacheStore, key: str) -> dict | None:
    """The cache entry at ``key``, or None; a foreign document is refused."""

    document = store.get_json(key)
    if document is None:
        return None
    if document.get("schema") != CACHE_SCHEMA:
        raise ForcingCacheRefused(f"{store.url}/{key} is "
                                  f"{document.get('schema')!r}, not "
                                  f"{CACHE_SCHEMA}")
    if not document.get("complete") or not isinstance(document.get("files"),
                                                      list):
        raise ForcingCacheRefused(f"{store.url}/{key} is not a complete "
                                  "entry")
    return document


def missing_blobs(store: CacheStore, entry: Mapping[str, Any]) -> list[str]:
    """Paths of an entry whose blob is absent or the wrong size."""

    missing = []
    for item in entry["files"]:
        if store.size(item["blob"]) != item["bytes"]:
            missing.append(item["path"])
    return missing


# --------------------------------------------------------------------------
# prefetch


def default_fetcher(request: Mapping[str, Any], out: Path) -> dict:
    """Run ``woof fetch`` for ``request`` into ``out`` (a child process, the
    command woof go's fetch stage runs); its output goes to stderr."""

    from woof.energy.run import _child_env

    argv = [sys.executable, "-m", "woof", "fetch",
            *fetch_argv(request, out)]
    completed = subprocess.run(argv, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, env=_child_env(),
                               check=False)
    text = completed.stdout.decode("utf-8", errors="replace")
    if text:
        sys.stderr.write(text)
    if completed.returncode != 0:
        tail = text[-2000:]
        raise ForcingCacheRefused(
            f"woof fetch exited {completed.returncode} for "
            f"{request['source']} {request['cycle']}: {tail.strip()}")
    return {"argv": ["woof", *argv[3:]], "exit": 0}


def describe_fetch(out: Path, request: Mapping[str, Any]) -> dict:
    """The files a finished table-route fetch left in ``out``, classified
    and hashed, with its route manifest's identity."""

    from woof import fetch_routes

    out = Path(out)
    manifest_path = out / fetch_routes.MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ForcingCacheRefused(f"the fetch left no readable "
                                  f"{fetch_routes.MANIFEST_NAME} in {out}: "
                                  f"{error}") from error
    if manifest.get("schema") != fetch_routes.ROUTE_MANIFEST_SCHEMA:
        raise ForcingCacheRefused(
            f"{manifest_path} is {manifest.get('schema')!r}, not a table "
            f"route manifest ({fetch_routes.ROUTE_MANIFEST_SCHEMA})")
    identity = manifest.get("request") or {}
    want_cycle = datetime.strptime(request["cycle"], "%Y-%m-%dT%H")
    if (identity.get("source") != request["source"]
            or identity.get("cycle") != f"{want_cycle:%Y-%m-%dT%H}Z"):
        raise ForcingCacheRefused(
            f"{manifest_path} records {identity.get('source')} "
            f"{identity.get('cycle')}, not {request['source']} "
            f"{request['cycle']}")
    entries = {str(e["relpath"]): e for e in manifest.get("files") or ()
               if isinstance(e, dict) and e.get("relpath")}
    files = []
    for path in sorted(out.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(out).as_posix()
        top = rel.split("/", 1)[0]
        if (top == ".fetch-verified" or top.startswith("quarantine-")
                or path.name.endswith((".part", ".partial"))):
            continue
        if (rel not in entries and not top.startswith("donor-")
                and rel not in _KEPT_RECEIPTS):
            # The handoff (inputs.txt, prep-arguments.json, ...) names this
            # scratch folder and every job's fetch writes its own.
            continue
        digest = _sha256_path(path)
        size = path.stat().st_size
        record: dict[str, Any] = {"path": _safe_relpath(rel),
                                  "sha256": digest, "bytes": size,
                                  "blob": blob_key(digest)}
        if rel in entries:
            entry = entries[rel]
            if entry.get("sha256") != digest or entry.get("bytes") != size:
                raise ForcingCacheRefused(
                    f"{rel} in {out} no longer matches its route manifest "
                    "entry; the fetch folder changed under the prefetch")
            record["kind"] = "object"
            record["fetch_entry"] = {
                key: entry.get(key) for key in ("name", "bytes", "sha256",
                                                "url", "endpoint")}
            record["role"] = entry.get("role")
            record["lead"] = entry.get("lead")
        elif top.startswith("donor-"):
            record["kind"] = "donor"
        else:
            record["kind"] = "receipt"
        files.append(record)
    missing = sorted(set(entries) - {f["path"] for f in files})
    if missing:
        raise ForcingCacheRefused(f"the route manifest in {out} names "
                                  f"{len(missing)} file(s) the folder lacks, "
                                  f"e.g. {missing[0]}")
    return {"manifest": manifest, "files": files}


def prefetch(request: Mapping[str, Any], cache: str | CacheStore, *,
             fetcher: Callable[[Mapping[str, Any], Path], Any] | None = None,
             workdir: Path | str | None = None, force: bool = False,
             job_id: str | None = None, keep_scratch: bool = False) -> dict:
    """Fetch ``request`` once into the cache; return the prefetch receipt
    (``woof-energy.forcing-prefetch.v1``).

    ``status`` is ``cached`` (the entry was already complete; nothing was
    fetched), ``fetched`` (this call fetched and published it) or
    ``fetched-peer-published`` (another prefetcher published the entry
    first; its document is kept, the blobs are the same bytes)."""

    request = dict(request)
    store = open_cache(cache)
    key = entry_key(request)
    started = time.monotonic()
    receipt: dict[str, Any] = {
        "schema": PREFETCH_SCHEMA, "cache_url": store.url, "entry": key,
        "key": request_key(request), "request": request, "job_id": job_id}
    existing = read_entry(store, key)
    if existing is not None and not force:
        missing = missing_blobs(store, existing)
        if not missing:
            receipt.update(status="cached",
                           files=len(existing["files"]),
                           bytes=existing["totals"]["bytes"],
                           seconds=round(time.monotonic() - started, 3),
                           finished_utc=_utc_now())
            return receipt
        receipt["notes"] = [f"entry was complete but {len(missing)} blob(s) "
                            f"are missing or short (e.g. {missing[0]}); "
                            "fetching again"]
    shareable, why = shareability(request["source"])
    if shareable != MODE_SHARED:
        raise ForcingCacheRefused(f"{request['source']} is not cached: {why}")
    base = Path(workdir) if workdir is not None else None
    if base is not None:
        base.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="forcing-prefetch-", dir=base))
    out = scratch / "fetch"
    try:
        fetch_started = time.monotonic()
        fetch_record = (fetcher or default_fetcher)(request, out)
        fetch_s = time.monotonic() - fetch_started
        described = describe_fetch(out, request)
        manifest = described["manifest"]
        uploaded = reused = 0
        uploaded_bytes = 0
        seen: set[str] = set()
        for item in described["files"]:
            if item["sha256"] in seen:
                continue
            seen.add(item["sha256"])
            if store.size(item["blob"]) == item["bytes"]:
                reused += 1
                continue
            store.put_file(item["blob"], out / item["path"])
            uploaded += 1
            uploaded_bytes += item["bytes"]
        objects = [f for f in described["files"] if f["kind"] == "object"]
        import woof

        entry = {
            "schema": CACHE_SCHEMA, "key": request_key(request),
            "entry": key, "complete": True,
            "source": request["source"], "cycle": request["cycle"],
            "request": request,
            "fetch": {
                "argv": ["woof", "fetch", *fetch_argv(request, "OUT")[:-2]],
                "route_manifest_schema": manifest.get("schema"),
                "route_table_sha256": manifest.get("route_table_sha256"),
                "identity": manifest.get("request"),
                "label": manifest.get("label"),
                "mode": manifest.get("mode"),
                "endpoints_served": (manifest.get("endpoints") or {}).get(
                    "served"),
                "seconds": round(fetch_s, 3),
                "fetcher": ("woof fetch" if fetcher is None else
                            getattr(fetcher, "__qualname__", repr(fetcher))),
                "record": (fetch_record if isinstance(
                    fetch_record, (dict, list, str, int, float, bool))
                    or fetch_record is None else repr(fetch_record)),
            },
            "files": described["files"],
            "totals": {"files": len(described["files"]),
                       "bytes": sum(f["bytes"] for f in described["files"]),
                       "objects": len(objects),
                       "object_bytes": sum(f["bytes"] for f in objects)},
            "provenance": {"woof_version": getattr(woof, "__version__", None),
                           "prefetch_job_id": job_id,
                           "host": socket.gethostname()},
            "retrieved_utc": _utc_now(),
        }
        if force or existing is not None:
            store.put_json(key, entry)
            published = True
        else:
            published = store.put_json_if_absent(key, entry)
        receipt.update(
            status="fetched" if published else "fetched-peer-published",
            files=entry["totals"]["files"], bytes=entry["totals"]["bytes"],
            objects=entry["totals"]["objects"],
            uploaded_blobs=uploaded, uploaded_bytes=uploaded_bytes,
            reused_blobs=reused, fetch_seconds=round(fetch_s, 3),
            seconds=round(time.monotonic() - started, 3),
            finished_utc=_utc_now())
        return receipt
    finally:
        if not keep_scratch:
            shutil.rmtree(scratch, ignore_errors=True)


# --------------------------------------------------------------------------
# mirror


def _go_steps(plan) -> list[tuple[Path, Path, list[str]]]:
    """(config, case root, domain ids) of every ``woof go`` step, as
    :func:`woof.energy.run.build_steps` folds them."""

    from woof.energy.run import build_steps

    steps = []
    for step in build_steps(plan):
        if step.kind != "go":
            continue
        argv = step.commands[0]
        config = plan.resolve(argv[1])
        outdir = argv[argv.index("--outdir") + 1]
        steps.append((config, (plan.directory / outdir).resolve(),
                      list(step.domain_ids)))
    return steps


def _mirror_config(config: Path, case_root: Path, *, forcing: Mapping,
                   store: CacheStore, entry_cache: dict) -> dict:
    from woof import fetch_routes
    from woof.fetch import parse_cycle
    from woof.go_cli import (config_fetch_request, managed_download_dir,
                             pin_request, pinned_transport)
    from woof.source_drivability import local_input_requested

    record: dict[str, Any] = {"config": str(config),
                              "case_root": str(case_root)}
    payload = tomllib.loads(config.read_text(encoding="utf-8"))
    fetch = payload.get("fetch")
    if "case_data" in payload or not isinstance(fetch, dict) \
            or not {"source", "cycle", "hours"} <= fetch.keys():
        record.update(status="no-download",
                      reason="the config names its inputs ([case_data]) or "
                             "has no [fetch] table")
        return record
    if local_input_requested(fetch):
        record.update(status="no-download",
                      reason="the config reads a local source root")
        return record
    mode, why = shareability(str(fetch["source"]))
    if mode != MODE_SHARED:
        record.update(status="direct", reason=why)
        return record
    want = dict(forcing["request"])
    have = config_request(fetch)
    record["request"] = have
    if family(have) != family(want) or have["hours"] > want["hours"]:
        differing = sorted(k for k in set(have) | set(want)
                           if have.get(k) != want.get(k))
        record.update(status="request-differs", fallback="direct-fetch",
                      reason="the config's [fetch] request differs from the "
                             f"cache entry's in {', '.join(differing)}; the "
                             "job fetches for itself")
        return record
    pinned, _ = pinned_transport(fetch, None)
    data_dir = managed_download_dir(
        case_root, pin_request(config_fetch_request(payload), pinned))
    record["data_dir"] = str(data_dir)
    plan = fetch_routes.resolve_request(
        have["source"], cycle=parse_cycle(str(fetch["cycle"]), have["source"]),
        hours=have["hours"], cadence=have["cadence"],
        start_hour=have["forecast_start_hour"], host=pinned,
        member=fetch.get("member"), out=data_dir)
    record["objects"] = len(plan.objects)
    if data_dir.is_dir() and fetch_routes.request_cached(plan, data_dir):
        record.update(status="already-local", mirrored=0, missing=[],
                      bytes_from_cache=0)
        return record
    key = forcing["entry"]
    if key not in entry_cache:
        entry_cache[key] = read_entry(store, key)
    entry = entry_cache[key]
    if entry is None:
        record.update(status="entry-missing", fallback="direct-fetch",
                      mirrored=0, missing=[o.relpath for o in plan.objects],
                      reason=f"{store.url}/{key} does not exist (the "
                             "prefetch did not run or failed); the job "
                             "fetches for itself")
        return record
    if family(entry["request"]) != family(have) or \
            entry["request"]["hours"] < have["hours"]:
        record.update(status="request-differs", fallback="direct-fetch",
                      reason="the cache entry was fetched for another "
                             "request")
        return record
    by_path = {item["path"]: item for item in entry["files"]}
    adopted: dict[str, dict] = {}
    missing: list[dict] = []
    copied = copied_bytes = local = 0
    for obj in plan.objects:
        item = by_path.get(obj.relpath)
        if item is None or item.get("kind") != "object":
            missing.append({"path": obj.relpath,
                            "why": "the cache entry holds no such object"})
            continue
        dest = data_dir / _safe_relpath(item["path"])
        if _matches(dest, item["bytes"], item["sha256"]):
            local += 1
        else:
            why = store.get_file(item["blob"], dest, sha256=item["sha256"],
                                 size=item["bytes"])
            if why is not None:
                missing.append({"path": obj.relpath, "why": why})
                continue
            copied += 1
            copied_bytes += item["bytes"]
        adopted[obj.relpath] = dict(item["fetch_entry"])
    donors = []
    for donor in plan.donors:
        prefix = f"donor-{donor.source}/"
        items = [i for i in entry["files"] if i["path"].startswith(prefix)]
        placed = 0
        reasons = []
        for item in items:
            dest = data_dir / _safe_relpath(item["path"])
            if _matches(dest, item["bytes"], item["sha256"]):
                placed += 1
                continue
            why = store.get_file(item["blob"], dest, sha256=item["sha256"],
                                 size=item["bytes"])
            if why is None:
                placed += 1
                copied_bytes += item["bytes"]
            else:
                reasons.append(f"{item['path']}: {why}")
        donors.append({"source": donor.source, "files": len(items),
                       "mirrored": placed})
        if not items or reasons:
            missing.append({"path": prefix, "why": "; ".join(reasons) or
                            "the cache entry holds no donor files"})
    fetch_routes.adopt_verified_objects(plan, data_dir, adopted)
    record.update(
        status="mirrored" if not missing else "partial",
        mirrored=copied, already_local=local, missing=missing,
        donors=donors, bytes_from_cache=copied_bytes,
        receipts_written=len(adopted))
    if missing:
        record["fallback"] = "direct-fetch"
        record["reason"] = (f"{len(missing)} file(s) are not in the cache; "
                            "woof fetch downloads only those")
    return record


def mirror_job(job: Mapping[str, Any], plan_path: Path, *, job_dir: Path,
               cache: str | CacheStore | None = None) -> dict:
    """Mirror the job's forcing from the cache into the download folders its
    plan's ``woof go`` steps will use; write and return the receipt."""

    from woof.energy.contracts import load_plan

    forcing = job.get("forcing")
    started = time.monotonic()
    receipt: dict[str, Any] = {"schema": MIRROR_SCHEMA,
                               "job_id": job.get("job_id"),
                               "mirrored_utc": _utc_now()}
    if not isinstance(forcing, Mapping):
        receipt.update(mode=None, status="no-forcing-cache",
                       reason="the job was planned without --forcing-cache")
        return receipt
    receipt.update(mode=forcing.get("mode"),
                   cache_url=forcing.get("cache_url"),
                   entry=forcing.get("entry"),
                   prefetch_job_id=forcing.get("prefetch_job_id"))
    if forcing.get("mode") != MODE_SHARED:
        receipt.update(status="direct", reason=forcing.get("reason"))
        _write_json(receipt, Path(job_dir) / MIRROR_NAME)
        return receipt
    store = open_cache(cache if cache is not None else forcing["cache_url"])
    plan = load_plan(plan_path)
    configs = []
    entry_cache: dict[str, Any] = {}
    for config, case_root, domains in _go_steps(plan):
        record = _mirror_config(config, case_root, forcing=forcing,
                                store=store, entry_cache=entry_cache)
        record["domains"] = domains
        configs.append(record)
    statuses = {c["status"] for c in configs}
    fallback = [c for c in configs if c.get("fallback")]
    receipt.update(
        configs=configs,
        status=("mirrored" if statuses <= {"mirrored", "already-local",
                                           "no-download"} and configs
                else "fallback" if fallback else
                "nothing-to-mirror" if not configs else "mixed"),
        fallback=bool(fallback),
        totals={"objects": sum(c.get("objects", 0) for c in configs),
                "mirrored": sum(c.get("mirrored", 0) for c in configs),
                "missing": sum(len(c.get("missing") or ()) for c in configs),
                "bytes_from_cache": sum(c.get("bytes_from_cache", 0)
                                        for c in configs)},
        seconds=round(time.monotonic() - started, 3))
    _write_json(receipt, Path(job_dir) / MIRROR_NAME)
    return receipt


# --------------------------------------------------------------------------
# prefetch job documents


def prefetch_job(request: Mapping[str, Any], cache_url: str, *,
                 job_schema: str) -> dict:
    """The prefetch job document for one (source, cycle) request.

    Its id is ``prefetch-<source>-<YYYYMMDDTHH>Z-<sha12>``, the sha256 over
    the request and the cache URL (the two things that decide what it
    writes), so the same plan produces the same document."""

    request = dict(request)
    cache_url = normalize_url(cache_url)
    identity = {"schema": job_schema, "kind": PREFETCH_KIND,
                "request": request, "cache_url": cache_url}
    digest = hashlib.sha256(_canonical(identity)).hexdigest()
    cycle = datetime.strptime(request["cycle"], "%Y-%m-%dT%H")
    slug = "".join(c if c.isalnum() or c in "-." else "-"
                   for c in request["source"]).strip("-") or "source"
    job_id = f"{PREFETCH_ID_PREFIX}{slug}-{cycle:%Y%m%dT%H}Z-{digest[:12]}"
    _, why = shareability(request["source"])
    return {
        "schema": job_schema, "kind": PREFETCH_KIND, "job_id": job_id,
        "identity_sha256": digest,
        "forcing": {"cache_url": cache_url, "entry": entry_key(request),
                    "key": request_key(request), "request": request,
                    "argv": ["woof", "fetch",
                             *fetch_argv(request, "OUT")[:-2]],
                    "shareability": why},
        "segment": {"cycle": f"{cycle:%Y-%m-%dT%H:%M:%SZ}"},
        "source": {"name": request["source"]},
        "resources": {"gpus": 0},
        "paths": {"hindcast_dir": "../..", "receipt": PREFETCH_RECEIPT_NAME},
    }


def is_prefetch(job: Mapping[str, Any]) -> bool:
    return job.get("kind") == PREFETCH_KIND


__all__ = [
    "CACHE_SCHEMA", "MIRROR_SCHEMA", "PREFETCH_SCHEMA", "PREFETCH_KIND",
    "FORECAST_KIND", "PREFETCH_ID_PREFIX", "MIRROR_NAME",
    "PREFETCH_RECEIPT_NAME", "MODE_SHARED", "MODE_DIRECT", "CacheStore",
    "ForcingCacheRefused", "blob_key", "config_request", "default_fetcher",
    "describe_fetch", "entry_key", "family", "fetch_argv", "fetch_hints",
    "forcing_request", "is_prefetch", "mirror_job", "missing_blobs",
    "normalize_url", "open_cache", "predicted_request", "prefetch",
    "prefetch_job", "read_entry", "request_key", "shareability",
]
