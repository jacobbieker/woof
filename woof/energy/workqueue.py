"""Object-store work queue for energy hindcast jobs (``woof energy queue``).

A queue is a prefix in an object store.  Nothing else is shared between
the machines that serve it -- no database, no shared filesystem, no
coordinator process -- so a fleet of GPU nodes in AWS or GCP drains one
queue by talking to the bucket alone.  Three backends implement the same
five operations (get, put, create-only put, compare-and-swap put, list):

``file:///abs/dir``
    A local or POSIX-shared directory.  Create-only writes are a staged
    file hard-linked into place (``link(2)`` fails if the name exists, the
    ``O_EXCL`` guarantee for a whole file); compare-and-swap holds a
    per-key ``flock`` while it compares the content digest and renames.
``s3://bucket/prefix``
    Amazon S3 (or a compatible store that implements conditional writes):
    create-only is ``PutObject`` with ``If-None-Match: *``; compare-and-
    swap is ``PutObject`` with ``If-Match: <ETag>``.  Needs ``boto3``.
``gs://bucket/prefix``
    Google Cloud Storage: create-only is ``ifGenerationMatch=0``;
    compare-and-swap is ``ifGenerationMatch=<generation>``.  Needs
    ``google-cloud-storage``.

Layout under the queue prefix (every document is JSON)::

    queue.json          woof-energy.queue.v1        settings, written once
    jobs/<id>.json      the job spec as pushed      immutable
    leases/<id>.json    woof-energy.queue-lease.v1  owner, expiry, counters
    done/<id>.json      woof-energy.queue-done.v1   create-only: exactly once
    failed/<id>.json    woof-energy.queue-failed.v1 charged failures, error tails
    dead/<id>.json      woof-energy.queue-dead.v1   gave up after max attempts
    logs/<id>/...       runner logs, one object per attempt

What is pinned:

* **Completion happens exactly once.**  ``done/<id>.json`` is written
  create-only.  Two workers can both finish one job -- a lease expired
  under a paused or partitioned host and another worker took it -- and
  exactly one of them writes the marker; the other's record goes to
  ``logs/<id>/duplicate-...json`` and the call reports the loss.
* **A lease is the claim.**  Claiming a job with no lease is a create-only
  write of ``leases/<id>.json``; taking over an expired, released or
  backed-off lease is a compare-and-swap against the version read.  Two
  claimants of one version cannot both win.  Renewal (the heartbeat) is a
  compare-and-swap too, so a worker whose lease was taken over learns it
  on its next renewal (:class:`LeaseLost`).
* **The lease document carries the counters.**  ``attempt`` counts claims,
  ``failures`` counts charged failures (runner exit not 0, timeout, a lease
  that expired under a dead worker) and ``preemptions`` counts leases
  released on SIGTERM, which are not charged.  ``failures`` reaching
  ``max_attempts`` -- or one exit in :data:`PERMANENT_EXIT_CODES` -- writes
  ``dead/<id>.json`` and nothing claims the job again until
  ``woof energy queue requeue``.
* **Backoff** after a charged failure is ``min(backoff_max_s,
  backoff_base_s * 2**(failures - 1))``; the lease waits in state
  ``backoff`` with a ``not_before`` time no worker claims it before.
* **Time is wall-clock UTC seconds** from every host.  Hosts must keep
  their clocks synchronised (NTP) to well inside ``lease_s``; the defaults
  leave minutes of margin.
* **Declared GPU memory.**  A job may declare the device memory it needs
  as ``resources.gpu_mem_gib`` (a positive number of GiB;
  ``woof energy hindcast plan`` writes the planner's priced peak there).
  ``push`` refuses any other value (:class:`JobSpecError`).  A claim may
  pass an ``admit`` callable: a job it rejects is skipped, not leased, so
  a worker packing several jobs on one card only takes what fits
  (``woof energy worker --slots-per-gpu``).  The declared figure of each
  job is read once per queue object and cached (job documents are
  immutable); a claim reads at most :data:`MAX_ADMISSION_READS_PER_CLAIM`
  job documents it has not seen.

Queue-level refusals raise a named :class:`QueueError` subclass; nothing
is coerced.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import hashlib
import json
import math
import os
import random
import re
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol
from urllib.parse import unquote, urlparse

#: Schemas this module writes.
QUEUE_SCHEMA = "woof-energy.queue.v1"
LEASE_SCHEMA = "woof-energy.queue-lease.v1"
DONE_SCHEMA = "woof-energy.queue-done.v1"
FAILED_SCHEMA = "woof-energy.queue-failed.v1"
DEAD_SCHEMA = "woof-energy.queue-dead.v1"
STATUS_SCHEMA = "woof-energy.queue-status.v1"
PUSH_SCHEMA = "woof-energy.queue-push.v1"
REQUEUE_SCHEMA = "woof-energy.queue-requeue.v1"

#: The job documents ``push`` accepts by default (sibling unit's spec).
JOB_SCHEMA = "woof-energy.hindcast-job.v1"

#: Runner exit codes no retry can change: woof's refusal exit is 2.
PERMANENT_EXIT_CODES = frozenset({2})

#: A job id becomes an object key, so it is held to a safe alphabet.
JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")

#: ``failed/<id>.json`` keeps the most recent attempts only.
FAILED_HISTORY_LIMIT = 20

#: Held leases a claim reads at most (sampled at random), so a fleet
#: polling a queue with thousands of running jobs does not GET every lease
#: on every poll.  An expired lease is still found within a few polls.
MAX_LEASE_READS_PER_CLAIM = 64

#: Job documents an admission-filtered claim reads at most (uncached), so
#: a full card does not GET thousands of jobs on one poll.
MAX_ADMISSION_READS_PER_CLAIM = 64

#: Declared-memory figures a queue object caches before it starts over.
ADMISSION_CACHE_LIMIT = 200_000

#: Error tail carried in failed/ and dead/ documents.
ERROR_TAIL_BYTES = 4000

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_LEASE_S = 900.0
DEFAULT_HEARTBEAT_S = 120.0
DEFAULT_BACKOFF_BASE_S = 60.0
DEFAULT_BACKOFF_MAX_S = 3600.0

_KEY_PATTERN = re.compile(r"^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$")


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class QueueError(RuntimeError):
    """The work queue refused an operation."""


class QueueURLError(QueueError, ValueError):
    """The queue URL names no supported backend or is malformed."""


class QueueBackendUnavailable(QueueError):
    """The backend's client library is not installed."""


class QueueNotInitialized(QueueError):
    """No ``queue.json`` at the URL: run ``woof energy queue init`` first."""


class QueueConfigConflict(QueueError):
    """``queue init`` found different settings already in place."""


class QueueSettingsError(QueueError, ValueError):
    """Queue settings that cannot work (heartbeat not inside the lease...)."""


class JobSpecError(QueueError, ValueError):
    """A job document is not one this queue accepts."""


class JobConflict(QueueError):
    """A different document is already queued under the same job id."""


class RequeueRefused(QueueError):
    """The job is done or running; requeue would break exactly-once."""


class PreconditionFailed(QueueError):
    """A create-only or compare-and-swap write lost its race."""


class LeaseLost(QueueError):
    """This worker's lease was taken over (it expired before renewal)."""


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def iso_utc(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    return (datetime.fromtimestamp(epoch, timezone.utc)
            .isoformat(timespec="seconds").replace("+00:00", "Z"))


def encode(document: dict[str, Any]) -> bytes:
    """Canonical JSON bytes: sorted keys, two-space indent, final newline."""

    return (json.dumps(document, sort_keys=True, indent=2) + "\n").encode(
        "utf-8")


def decode(data: bytes, key: str) -> dict[str, Any]:
    try:
        document = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QueueError(f"queue object {key} is not JSON: {error}") from error
    if not isinstance(document, dict):
        raise QueueError(f"queue object {key} is not a JSON object")
    return document


def _check_key(key: str) -> str:
    if not _KEY_PATTERN.fullmatch(key) or ".." in key.split("/"):
        raise QueueError(f"refusing unsafe queue key {key!r}")
    return key


def tail_text(data: bytes, limit: int = ERROR_TAIL_BYTES) -> str:
    """The last ``limit`` bytes of a log as text, cut at a line start."""

    if len(data) > limit:
        data = data[-limit:]
        newline = data.find(b"\n")
        if 0 <= newline < len(data) - 1:
            data = data[newline + 1:]
    return data.decode("utf-8", errors="replace")


# --------------------------------------------------------------------------
# Object stores
# --------------------------------------------------------------------------


class ObjectStore(Protocol):
    """The five operations a queue backend provides.

    ``version`` is opaque: a content digest (file), an ETag (S3) or a
    generation number (GCS).  ``put_if_absent`` and ``put_if_match``
    raise :class:`PreconditionFailed` when they lose; ``put_if_match``
    also loses when the object no longer exists.
    """

    url: str

    def get(self, key: str) -> tuple[bytes, str] | None: ...

    def put(self, key: str, data: bytes) -> str: ...

    def put_if_absent(self, key: str, data: bytes) -> str: ...

    def put_if_match(self, key: str, data: bytes, version: str) -> str: ...

    def delete(self, key: str) -> None: ...

    def list(self, prefix: str) -> list[str]: ...


class FileStore:
    """``file://`` backend: one directory, POSIX ``link``/``rename``/``flock``.

    Shared between hosts only on a filesystem with working ``flock`` and
    atomic ``link`` (local disks; NFSv4 with care).  Staging files live in
    ``.tmp/`` and per-key locks in ``.locks/``; neither appears in listings.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.url = "file://" + str(self.root)

    def _path(self, key: str) -> Path:
        return self.root / _check_key(key)

    @contextlib.contextmanager
    def _locked(self, key: str):
        import fcntl

        locks = self.root / ".locks"
        locks.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        with open(locks / f"{digest}.lock", "a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _stage(self, data: bytes) -> Path:
        staging = self.root / ".tmp"
        staging.mkdir(parents=True, exist_ok=True)
        path = staging / f"{uuid.uuid4().hex}.part"
        with open(path, "xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        return path

    @staticmethod
    def _version(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def get(self, key: str) -> tuple[bytes, str] | None:
        try:
            data = self._path(key).read_bytes()
        except FileNotFoundError:
            return None
        return data, self._version(data)

    def put(self, key: str, data: bytes) -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        staged = self._stage(data)
        try:
            with self._locked(key):
                os.replace(staged, path)
        except BaseException:
            staged.unlink(missing_ok=True)
            raise
        return self._version(data)

    def put_if_absent(self, key: str, data: bytes) -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        staged = self._stage(data)
        try:
            with self._locked(key):
                try:
                    os.link(staged, path)
                except FileExistsError as error:
                    raise PreconditionFailed(f"{key} exists") from error
                except OSError as error:
                    # A filesystem without hard links: the key lock makes
                    # the existence check and the rename one step for every
                    # writer that goes through this class.
                    if error.errno not in (errno.EPERM, errno.ENOTSUP,
                                           errno.EOPNOTSUPP, errno.EXDEV):
                        raise
                    if path.exists():
                        raise PreconditionFailed(f"{key} exists") from None
                    os.replace(staged, path)
        finally:
            staged.unlink(missing_ok=True)
        return self._version(data)

    def put_if_match(self, key: str, data: bytes, version: str) -> str:
        path = self._path(key)
        staged = self._stage(data)
        try:
            with self._locked(key):
                try:
                    current = path.read_bytes()
                except FileNotFoundError:
                    raise PreconditionFailed(f"{key} is gone") from None
                if self._version(current) != version:
                    raise PreconditionFailed(f"{key} changed")
                os.replace(staged, path)
        finally:
            staged.unlink(missing_ok=True)
        return self._version(data)

    def delete(self, key: str) -> None:
        with self._locked(key):
            self._path(key).unlink(missing_ok=True)

    def list(self, prefix: str) -> list[str]:
        base = self.root / prefix.rstrip("/")
        if not base.is_dir():
            return []
        keys = []
        for directory, subdirectories, files in os.walk(base):
            subdirectories[:] = [d for d in subdirectories
                                 if not d.startswith(".")]
            for name in files:
                if name.startswith("."):
                    continue
                full = Path(directory) / name
                keys.append(full.relative_to(self.root).as_posix())
        return sorted(keys)


def _s3_error_code(error: BaseException) -> tuple[str | None, int | None]:
    response = getattr(error, "response", None) or {}
    code = (response.get("Error") or {}).get("Code")
    status = (response.get("ResponseMetadata") or {}).get("HTTPStatusCode")
    return (None if code is None else str(code)), status


_S3_LOST = {"PreconditionFailed", "ConditionalRequestConflict", "412", "409"}
_S3_MISSING = {"NoSuchKey", "404", "NotFound"}


class S3Store:
    """``s3://`` backend over a boto3 S3 client.

    Create-only writes send ``IfNoneMatch='*'`` and compare-and-swap sends
    ``IfMatch=<ETag>`` (S3 conditional writes, general-purpose buckets
    since 2024).  A 412 ``PreconditionFailed`` or 409
    ``ConditionalRequestConflict`` is a lost race.  Credentials, region and
    endpoint come from the usual AWS environment (``AWS_PROFILE``,
    ``AWS_REGION``, ``AWS_ENDPOINT_URL_S3``...).
    """

    def __init__(self, bucket: str, prefix: str, client=None):
        if client is None:
            try:
                import boto3
            except ImportError as error:
                raise QueueBackendUnavailable(
                    "s3:// queues need boto3, which this environment does not "
                    "have; install it (pip install boto3) on every worker "
                    "host") from error
            client = boto3.client("s3")
        self.client = client
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.url = f"s3://{bucket}/{self.prefix}".rstrip("/")

    def _key(self, key: str) -> str:
        _check_key(key)
        return f"{self.prefix}/{key}" if self.prefix else key

    def _lost(self, error: BaseException, key: str, *,
              missing_loses: bool) -> None:
        code, status = _s3_error_code(error)
        if code in _S3_LOST or status in (409, 412):
            raise PreconditionFailed(f"{key}: {code or status}") from error
        if missing_loses and (code in _S3_MISSING or status == 404):
            raise PreconditionFailed(f"{key} is gone") from error

    def get(self, key: str) -> tuple[bytes, str] | None:
        try:
            response = self.client.get_object(Bucket=self.bucket,
                                              Key=self._key(key))
        except Exception as error:
            code, status = _s3_error_code(error)
            if code in _S3_MISSING or status == 404:
                return None
            raise
        return response["Body"].read(), str(response["ETag"])

    def _put(self, key: str, data: bytes, **conditions) -> str:
        response = self.client.put_object(
            Bucket=self.bucket, Key=self._key(key), Body=data,
            ContentType="application/json", **conditions)
        return str(response["ETag"])

    def put(self, key: str, data: bytes) -> str:
        return self._put(key, data)

    def put_if_absent(self, key: str, data: bytes) -> str:
        try:
            return self._put(key, data, IfNoneMatch="*")
        except Exception as error:
            self._lost(error, key, missing_loses=False)
            raise

    def put_if_match(self, key: str, data: bytes, version: str) -> str:
        try:
            return self._put(key, data, IfMatch=version)
        except Exception as error:
            self._lost(error, key, missing_loses=True)
            raise

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=self._key(key))

    def list(self, prefix: str) -> list[str]:
        full = self._key(prefix.rstrip("/")) + "/"
        strip = len(self.prefix) + 1 if self.prefix else 0
        keys: list[str] = []
        token = None
        while True:
            request = {"Bucket": self.bucket, "Prefix": full}
            if token:
                request["ContinuationToken"] = token
            response = self.client.list_objects_v2(**request)
            keys.extend(item["Key"][strip:]
                        for item in response.get("Contents", ()))
            if not response.get("IsTruncated"):
                break
            token = response.get("NextContinuationToken")
        return sorted(keys)


def _gcs_status(error: BaseException) -> int | None:
    code = getattr(error, "code", None)
    return code if isinstance(code, int) else None


class GCSStore:
    """``gs://`` backend over a ``google.cloud.storage`` client.

    Create-only writes pass ``if_generation_match=0``; compare-and-swap
    passes the generation read.  HTTP 412 is a lost race.  Credentials come
    from Application Default Credentials.
    """

    def __init__(self, bucket: str, prefix: str, client=None):
        if client is None:
            try:
                from google.cloud import storage
            except ImportError as error:
                raise QueueBackendUnavailable(
                    "gs:// queues need google-cloud-storage, which this "
                    "environment does not have; install it (pip install "
                    "google-cloud-storage) on every worker host") from error
            client = storage.Client()
        self.client = client
        self.bucket_name = bucket
        self.bucket = client.bucket(bucket)
        self.prefix = prefix.strip("/")
        self.url = f"gs://{bucket}/{self.prefix}".rstrip("/")

    def _key(self, key: str) -> str:
        _check_key(key)
        return f"{self.prefix}/{key}" if self.prefix else key

    def get(self, key: str) -> tuple[bytes, str] | None:
        name = self._key(key)
        for _ in range(5):
            blob = self.bucket.get_blob(name)
            if blob is None:
                return None
            try:
                data = blob.download_as_bytes(
                    if_generation_match=blob.generation)
            except Exception as error:
                status = _gcs_status(error)
                if status == 404:
                    return None
                if status == 412:
                    continue  # rewritten between metadata and read
                raise
            return data, str(blob.generation)
        raise QueueError(f"{key} kept changing while it was read")

    def _put(self, key: str, data: bytes, generation: int | None) -> str:
        blob = self.bucket.blob(self._key(key))
        kwargs = {"content_type": "application/json"}
        if generation is not None:
            kwargs["if_generation_match"] = generation
        try:
            blob.upload_from_string(data, **kwargs)
        except Exception as error:
            if generation is not None and _gcs_status(error) in (404, 412):
                raise PreconditionFailed(f"{key}: lost") from error
            raise
        return str(blob.generation)

    def put(self, key: str, data: bytes) -> str:
        return self._put(key, data, None)

    def put_if_absent(self, key: str, data: bytes) -> str:
        return self._put(key, data, 0)

    def put_if_match(self, key: str, data: bytes, version: str) -> str:
        return self._put(key, data, int(version))

    def delete(self, key: str) -> None:
        try:
            self.bucket.blob(self._key(key)).delete()
        except Exception as error:
            if _gcs_status(error) != 404:
                raise

    def list(self, prefix: str) -> list[str]:
        full = self._key(prefix.rstrip("/")) + "/"
        strip = len(self.prefix) + 1 if self.prefix else 0
        return sorted(blob.name[strip:] for blob in
                      self.client.list_blobs(self.bucket_name, prefix=full))


def open_store(url: str, *, s3_client=None, gcs_client=None) -> ObjectStore:
    """The backend a queue URL names; clients may be injected (tests)."""

    parsed = urlparse(url)
    if parsed.scheme == "file":
        if parsed.netloc not in ("", "localhost"):
            raise QueueURLError(
                f"file:// queue URLs take an absolute path "
                f"(file:///abs/dir); got {url!r}")
        path = unquote(parsed.path)
        if not path.startswith("/"):
            raise QueueURLError(f"file:// queue path must be absolute: {url!r}")
        return FileStore(Path(path))
    if parsed.scheme in ("s3", "gs"):
        if not parsed.netloc:
            raise QueueURLError(f"{parsed.scheme}:// URL names no bucket: {url!r}")
        if parsed.scheme == "s3":
            return S3Store(parsed.netloc, parsed.path, client=s3_client)
        return GCSStore(parsed.netloc, parsed.path, client=gcs_client)
    raise QueueURLError(
        f"queue URL {url!r} names no supported backend; use file:///abs/dir, "
        "s3://bucket/prefix or gs://bucket/prefix")


# --------------------------------------------------------------------------
# Settings and job documents
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class QueueSettings:
    """``queue.json``: written once by ``queue init``, read by every worker."""

    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    lease_s: float = DEFAULT_LEASE_S
    heartbeat_s: float = DEFAULT_HEARTBEAT_S
    backoff_base_s: float = DEFAULT_BACKOFF_BASE_S
    backoff_max_s: float = DEFAULT_BACKOFF_MAX_S

    def validate(self) -> "QueueSettings":
        if self.max_attempts < 1:
            raise QueueSettingsError("max_attempts must be at least 1")
        if not (0 < self.heartbeat_s and self.heartbeat_s * 2 <= self.lease_s):
            raise QueueSettingsError(
                f"heartbeat_s ({self.heartbeat_s:g}) must be positive and at "
                f"most half of lease_s ({self.lease_s:g}) so one missed "
                "renewal does not lose the lease")
        if self.backoff_base_s < 0 or self.backoff_max_s < self.backoff_base_s:
            raise QueueSettingsError(
                "backoff needs 0 <= backoff_base_s <= backoff_max_s")
        return self

    def backoff_s(self, failures: int) -> float:
        if failures < 1:
            return 0.0
        return min(self.backoff_max_s,
                   self.backoff_base_s * 2.0 ** (failures - 1))

    def document(self) -> dict[str, Any]:
        return {"schema": QUEUE_SCHEMA, "max_attempts": self.max_attempts,
                "lease_s": self.lease_s, "heartbeat_s": self.heartbeat_s,
                "backoff_base_s": self.backoff_base_s,
                "backoff_max_s": self.backoff_max_s}

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "QueueSettings":
        if document.get("schema") != QUEUE_SCHEMA:
            raise QueueError(
                f"queue.json schema is {document.get('schema')!r}, expected "
                f"{QUEUE_SCHEMA!r}")
        try:
            return cls(int(document["max_attempts"]),
                       float(document["lease_s"]),
                       float(document["heartbeat_s"]),
                       float(document["backoff_base_s"]),
                       float(document["backoff_max_s"])).validate()
        except KeyError as error:
            raise QueueError(f"queue.json lacks {error.args[0]!r}") from None


def declared_gpu_mem_gib(document: Any, *, source: str = "<job>"
                         ) -> float | None:
    """The job's ``resources.gpu_mem_gib``, or None when it declares none.

    A declared value must be a finite positive number (GiB); anything else
    is :class:`JobSpecError`, never coerced.  ``None`` (or no
    ``resources`` table) is "not declared": the worker prices it with its
    own conservative default.
    """

    if not isinstance(document, dict):
        return None
    resources = document.get("resources")
    if resources is None:
        return None
    if not isinstance(resources, dict):
        raise JobSpecError(f"{source}: resources must be a JSON object")
    value = resources.get("gpu_mem_gib")
    if value is None:
        return None
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or float(value) <= 0):
        raise JobSpecError(
            f"{source}: resources.gpu_mem_gib {value!r} must be a positive "
            "number of GiB (or absent)")
    return float(value)


def job_from_document(document: Any, *, schema: str = JOB_SCHEMA,
                      source: str = "<job>") -> tuple[str, dict[str, Any]]:
    """``(job_id, document)`` for a pushable job, or :class:`JobSpecError`."""

    if not isinstance(document, dict):
        raise JobSpecError(f"{source}: a job document is a JSON object")
    if document.get("schema") != schema:
        raise JobSpecError(
            f"{source}: schema is {document.get('schema')!r}, this push "
            f"accepts {schema!r} (--schema names another)")
    job_id = document.get("job_id")
    if not isinstance(job_id, str) or not JOB_ID_PATTERN.fullmatch(job_id):
        raise JobSpecError(
            f"{source}: job_id {job_id!r} must match "
            f"{JOB_ID_PATTERN.pattern} (it becomes an object key)")
    declared_gpu_mem_gib(document, source=source)
    return job_id, document


def read_job_files(paths: Iterable[str | Path], *,
                   schema: str = JOB_SCHEMA) -> list[tuple[str, dict[str, Any]]]:
    """Job documents from files and directories (``*.json`` inside)."""

    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            found = sorted(path.glob("*.json"))
            if not found:
                raise JobSpecError(f"{path}: directory holds no *.json jobs")
            files.extend(found)
        elif path.is_file():
            files.append(path)
        else:
            raise JobSpecError(f"{path}: no such job file or directory")
    jobs = []
    seen: dict[str, Path] = {}
    for path in files:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise JobSpecError(f"{path}: not JSON: {error}") from error
        job_id, document = job_from_document(document, schema=schema,
                                             source=str(path))
        if job_id in seen:
            raise JobSpecError(
                f"job_id {job_id!r} appears in both {seen[job_id]} and {path}")
        seen[job_id] = path
        jobs.append((job_id, document))
    return jobs


# --------------------------------------------------------------------------
# The queue
# --------------------------------------------------------------------------


@dataclass
class Claim:
    """A job this worker holds a lease on."""

    job_id: str
    job: dict[str, Any]
    job_bytes: bytes
    lease: dict[str, Any]
    version: str
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    lost: bool = False

    @property
    def attempt(self) -> int:
        return int(self.lease["attempt"])

    @property
    def failures(self) -> int:
        return int(self.lease["failures"])


def _ids(keys: Iterable[str], prefix: str) -> set[str]:
    out = set()
    for key in keys:
        name = key[len(prefix):]
        if "/" not in name and name.endswith(".json"):
            out.add(name[:-5])
    return out


class WorkQueue:
    """Queue operations over one :class:`ObjectStore`.

    ``clock`` returns epoch seconds; tests inject one to expire leases
    without sleeping.
    """

    def __init__(self, store: ObjectStore, *,
                 clock: Callable[[], float] = time.time):
        self.store = store
        self.clock = clock
        self._settings: QueueSettings | None = None
        self.last_view: dict[str, set[str]] | None = None
        #: Jobs the last admission-filtered :meth:`claim` skipped because
        #: ``admit`` rejected them (claimable, but not here and now).
        self.last_unadmitted = 0
        self._declared: dict[str, float | None] = {}

    @classmethod
    def open(cls, url: str, **clients) -> "WorkQueue":
        return cls(open_store(url, **clients))

    @property
    def url(self) -> str:
        return self.store.url

    # -- settings ---------------------------------------------------------

    def init(self, settings: QueueSettings) -> dict[str, Any]:
        settings.validate()
        if isinstance(self.store, FileStore):
            self.store.root.mkdir(parents=True, exist_ok=True)
        document = settings.document()
        try:
            self.store.put_if_absent("queue.json", encode(document))
            created = True
        except PreconditionFailed:
            existing = self.settings(refresh=True)
            if existing != settings:
                raise QueueConfigConflict(
                    f"{self.url} is already initialised with "
                    f"{existing.document()}; refusing to change it to "
                    f"{document}") from None
            created = False
        self._settings = settings
        return {"schema": QUEUE_SCHEMA, "queue": self.url, "created": created,
                "settings": document}

    def settings(self, *, refresh: bool = False) -> QueueSettings:
        if self._settings is None or refresh:
            found = self.store.get("queue.json")
            if found is None:
                raise QueueNotInitialized(
                    f"{self.url} has no queue.json; run `woof energy queue "
                    "init` first")
            self._settings = QueueSettings.from_document(
                decode(found[0], "queue.json"))
        return self._settings

    # -- push -------------------------------------------------------------

    def push(self, job_id: str, document: dict[str, Any]) -> str:
        """``queued``, ``already-queued``, or :class:`JobConflict`."""

        self.settings()
        key = f"jobs/{job_id}.json"
        declared_gpu_mem_gib(document, source=key)
        try:
            self.store.put_if_absent(key, encode(document))
            return "queued"
        except PreconditionFailed:
            found = self.store.get(key)
            if found is not None and decode(found[0], key) == document:
                return "already-queued"
            raise JobConflict(
                f"job {job_id!r} is already queued with a different "
                "document; job ids are deterministic, so a changed job "
                "needs a new id") from None

    # -- claims -----------------------------------------------------------

    def _snapshot(self) -> dict[str, set[str]]:
        return {name: _ids(self.store.list(f"{name}/"), f"{name}/")
                for name in ("jobs", "leases", "done", "dead")}

    def snapshot(self) -> dict[str, set[str]]:
        """The job, lease, done and dead listings one :meth:`claim` reads
        (``claim(view=...)`` takes one made earlier)."""

        return self._snapshot()

    def drained(self, *, reuse_last: bool = False) -> bool:
        """Every queued job is done or dead.

        ``reuse_last`` answers from the listing the last :meth:`claim`
        made instead of listing again.
        """

        view = self.last_view if reuse_last and self.last_view else \
            self._snapshot()
        return not (view["jobs"] - view["done"] - view["dead"])

    def is_done(self, job_id: str) -> bool:
        return self.store.get(f"done/{job_id}.json") is not None

    def _lease_document(self, job_id: str, *, state: str,
                        owner: dict[str, Any] | None, attempt: int,
                        failures: int, preemptions: int, revision: int,
                        now: float, expires_at: float | None = None,
                        not_before: float | None = None,
                        claimed_at: float | None = None,
                        note: str | None = None) -> dict[str, Any]:
        return {
            "schema": LEASE_SCHEMA, "job_id": job_id, "state": state,
            "owner": owner, "attempt": attempt, "failures": failures,
            "preemptions": preemptions, "revision": revision,
            "claimed_at": claimed_at, "claimed_at_utc": iso_utc(claimed_at),
            "updated_at": now, "updated_at_utc": iso_utc(now),
            "expires_at": expires_at, "expires_at_utc": iso_utc(expires_at),
            "not_before": not_before, "not_before_utc": iso_utc(not_before),
            "note": note,
        }

    def lease_view(self, lease: dict[str, Any], now: float) -> str:
        """``running``, ``expired``, ``backoff`` or ``claimable``."""

        state = lease.get("state")
        if state == "running":
            expires = lease.get("expires_at")
            return "running" if expires is not None and expires > now \
                else "expired"
        if state == "backoff":
            not_before = lease.get("not_before") or 0.0
            return "backoff" if not_before > now else "claimable"
        if state == "dead":
            return "dead"
        return "claimable"  # released

    def declared_gpu_mem(self, job_id: str) -> tuple[bool, float | None]:
        """``(known, gib)``: the job's declared GPU memory, from the cache
        or one GET (``known`` False when the job document is gone)."""

        if job_id in self._declared:
            return True, self._declared[job_id]
        key = f"jobs/{job_id}.json"
        found = self.store.get(key)
        if found is None:
            return False, None
        try:
            gib = declared_gpu_mem_gib(decode(found[0], key), source=key)
        except JobSpecError:
            # Pushed by an older client or written by hand: the worker
            # treats it as undeclared and the runner decides.
            gib = None
        if len(self._declared) >= ADMISSION_CACHE_LIMIT:
            self._declared.clear()
        self._declared[job_id] = gib
        return True, gib

    def claim(self, owner: dict[str, Any], *,
              rng: random.Random | None = None,
              admit: Callable[[str, float | None], bool] | None = None,
              view: dict[str, set[str]] | None = None) -> Claim | None:
        """Take one job, or ``None`` if nothing is claimable right now.

        Jobs with no lease are tried first, in random order (so a fleet
        does not stampede one key); then leases that expired, were
        released, or finished their backoff.

        ``admit(job_id, declared_gpu_mem_gib)`` filters candidates before
        their lease is taken: a rejected job is left for another claimant
        and counted in :attr:`last_unadmitted`.  ``view`` is a listing
        :meth:`snapshot` made earlier (stale is safe: every take is a
        create-only or compare-and-swap write and re-checks completion).
        """

        rng = rng or random.Random()
        view = view if view is not None else self._snapshot()
        self.last_view = view
        self.last_unadmitted = 0
        reads = [0]

        def admitted(job_id: str) -> bool:
            if admit is None:
                return True
            if job_id not in self._declared:
                if reads[0] >= MAX_ADMISSION_READS_PER_CLAIM:
                    self.last_unadmitted += 1
                    return False
                reads[0] += 1
            known, gib = self.declared_gpu_mem(job_id)
            if not known:
                return False
            if admit(job_id, gib):
                return True
            self.last_unadmitted += 1
            return False

        finished = view["done"] | view["dead"]
        fresh = sorted(view["jobs"] - view["leases"] - finished)
        rng.shuffle(fresh)
        for job_id in fresh:
            if not admitted(job_id):
                continue
            claim = self._take(job_id, None, owner)
            if claim is not None:
                return claim
        held = sorted((view["leases"] & view["jobs"]) - finished)
        rng.shuffle(held)
        for job_id in held[:MAX_LEASE_READS_PER_CLAIM]:
            key = f"leases/{job_id}.json"
            found = self.store.get(key)
            if found is None:
                prior = None
            else:
                lease = decode(found[0], key)
                state = self.lease_view(lease, self.clock())
                if state == "dead":
                    # A worker died between marking the lease dead and
                    # writing the marker: finish its job for it.
                    self._mark_dead(job_id,
                                    failures=int(lease.get("failures", 0)),
                                    reason=str(lease.get("note")),
                                    error_tail="")
                    continue
                if state not in ("expired", "claimable"):
                    continue
                prior = (lease, found[1])
            if not admitted(job_id):
                continue
            claim = self._take(job_id, prior, owner)
            if claim is not None:
                return claim
        return None

    def _take(self, job_id: str,
              prior: tuple[dict[str, Any], str] | None,
              owner: dict[str, Any]) -> Claim | None:
        settings = self.settings()
        now = self.clock()
        key = f"leases/{job_id}.json"
        attempt = failures = preemptions = revision = 0
        expired_from = None
        if prior is not None:
            lease, _ = prior
            attempt = int(lease.get("attempt", 0))
            failures = int(lease.get("failures", 0))
            preemptions = int(lease.get("preemptions", 0))
            revision = int(lease.get("revision", 0))
            if self.lease_view(lease, now) == "expired":
                failures += 1
                expired_from = lease
        if expired_from is not None and failures >= settings.max_attempts:
            dead_lease = self._lease_document(
                job_id, state="dead", owner=None, attempt=attempt,
                failures=failures, preemptions=preemptions,
                revision=revision + 1, now=now,
                note="lease expired on the last allowed attempt")
            try:
                self.store.put_if_match(key, encode(dead_lease), prior[1])
            except PreconditionFailed:
                return None
            reason = self._expiry_reason(expired_from)
            self._mark_dead(job_id, failures=failures, reason=reason,
                            error_tail="")
            self._record_failure(job_id, attempt=attempt,
                                 owner=expired_from.get("owner"),
                                 exit_code=None, reason=reason,
                                 error_tail="", log_key=None,
                                 failures=failures)
            return None
        lease = self._lease_document(
            job_id, state="running", owner=owner, attempt=attempt + 1,
            failures=failures, preemptions=preemptions,
            revision=revision + 1, now=now, claimed_at=now,
            expires_at=now + settings.lease_s)
        try:
            if prior is None:
                version = self.store.put_if_absent(key, encode(lease))
            else:
                version = self.store.put_if_match(key, encode(lease), prior[1])
        except PreconditionFailed:
            return None
        if expired_from is not None:
            self._record_failure(job_id, attempt=attempt,
                                 owner=expired_from.get("owner"),
                                 exit_code=None,
                                 reason=self._expiry_reason(expired_from),
                                 error_tail="", log_key=None,
                                 failures=failures)
        claim_ = Claim(job_id, {}, b"", lease, version)
        job = self.store.get(f"jobs/{job_id}.json")
        if (job is None or self.is_done(job_id)
                or self.store.get(f"dead/{job_id}.json") is not None):
            # Finished (or withdrawn) between the listing and the claim.
            self._drop_lease(claim_)
            return None
        claim_.job_bytes = job[0]
        claim_.job = decode(job[0], f"jobs/{job_id}.json")
        return claim_

    @staticmethod
    def _expiry_reason(lease: dict[str, Any]) -> str:
        owner = lease.get("owner") or {}
        return (f"lease-expired: worker {owner.get('worker_id')!r} on "
                f"{owner.get('host')!r} stopped renewing (expired "
                f"{lease.get('expires_at_utc')})")

    def renew(self, claim: Claim) -> None:
        """Extend the lease by ``lease_s``; :class:`LeaseLost` if taken over."""

        settings = self.settings()
        with claim.lock:
            if claim.lost:
                raise LeaseLost(f"lease on {claim.job_id} was lost")
            now = self.clock()
            lease = dict(claim.lease, updated_at=now,
                         updated_at_utc=iso_utc(now),
                         expires_at=now + settings.lease_s,
                         expires_at_utc=iso_utc(now + settings.lease_s),
                         revision=int(claim.lease["revision"]) + 1)
            try:
                claim.version = self.store.put_if_match(
                    f"leases/{claim.job_id}.json", encode(lease),
                    claim.version)
            except PreconditionFailed:
                claim.lost = True
                raise LeaseLost(
                    f"lease on {claim.job_id} was taken over") from None
            claim.lease = lease

    def _drop_lease(self, claim: Claim) -> None:
        """Delete the lease if it is still this claim's version."""

        key = f"leases/{claim.job_id}.json"
        with claim.lock:
            found = self.store.get(key)
            if found is not None and found[1] == claim.version:
                self.store.delete(key)
            claim.lost = True

    def abandon(self, claim: Claim) -> None:
        """Let go of a claim without settling it (done elsewhere)."""

        self._drop_lease(claim)

    def complete(self, claim: Claim, record: dict[str, Any]) -> bool:
        """Write ``done/<id>.json`` create-only.  ``True`` if this call won.

        The loser's record is kept under ``logs/<id>/duplicate-...json``.
        """

        now = self.clock()
        document = {
            "schema": DONE_SCHEMA, "job_id": claim.job_id,
            "attempt": claim.attempt, "owner": claim.lease.get("owner"),
            "claimed_at": claim.lease.get("claimed_at"),
            "completed_at": now, "completed_at_utc": iso_utc(now),
            **record,
        }
        try:
            self.store.put_if_absent(f"done/{claim.job_id}.json",
                                     encode(document))
            won = True
        except PreconditionFailed:
            won = self._done_by(claim)
        if not won:
            worker = (claim.lease.get("owner") or {}).get("worker_id", "worker")
            self.store.put(
                f"logs/{claim.job_id}/duplicate-attempt-{claim.attempt:03d}-"
                f"{_safe(worker)}.json", encode(document))
        self._drop_lease(claim)
        return won

    def _done_by(self, claim: Claim) -> bool:
        """Whether the existing done marker is this very claim's.

        A retried completion whose first write landed (the response was
        lost) finds its own marker; that is a win, not a duplicate.
        """

        found = self.store.get(f"done/{claim.job_id}.json")
        if found is None:
            return False
        done = decode(found[0], f"done/{claim.job_id}.json")
        return (done.get("owner") == claim.lease.get("owner")
                and done.get("attempt") == claim.attempt
                and done.get("claimed_at") == claim.lease.get("claimed_at"))

    def release(self, claim: Claim, reason: str) -> bool:
        """Hand the job back uncharged (SIGTERM/preemption)."""

        now = self.clock()
        with claim.lock:
            if claim.lost:
                return False
            lease = self._lease_document(
                claim.job_id, state="released", owner=None,
                attempt=claim.attempt, failures=claim.failures,
                preemptions=int(claim.lease["preemptions"]) + 1,
                revision=int(claim.lease["revision"]) + 1, now=now,
                note=reason)
            try:
                claim.version = self.store.put_if_match(
                    f"leases/{claim.job_id}.json", encode(lease),
                    claim.version)
            except PreconditionFailed:
                claim.lost = True
                return False
            claim.lease = lease
            claim.lost = True
            return True

    def fail(self, claim: Claim, *, exit_code: int | None, reason: str,
             error_tail: str, log_key: str | None,
             permanent: bool = False) -> str:
        """Charge one failure: ``backoff``, ``dead`` or ``lost``.

        ``lost`` means the lease had already been taken over; the failure
        is recorded in ``failed/`` but the other holder owns the job's state.
        """

        settings = self.settings()
        now = self.clock()
        failures = claim.failures + 1
        dead = permanent or failures >= settings.max_attempts
        owner = claim.lease.get("owner")
        with claim.lock:
            if not claim.lost:
                not_before = None if dead else now + settings.backoff_s(failures)
                lease = self._lease_document(
                    claim.job_id, state="dead" if dead else "backoff",
                    owner=None, attempt=claim.attempt, failures=failures,
                    preemptions=int(claim.lease["preemptions"]),
                    revision=int(claim.lease["revision"]) + 1, now=now,
                    not_before=not_before, note=reason)
                try:
                    claim.version = self.store.put_if_match(
                        f"leases/{claim.job_id}.json", encode(lease),
                        claim.version)
                    claim.lease = lease
                except PreconditionFailed:
                    claim.lost = True
            lost = claim.lost
            claim.lost = True
        if not lost and dead:
            self._mark_dead(
                claim.job_id, failures=failures, error_tail=error_tail,
                reason=(f"permanent: {reason}" if permanent else reason))
        self._record_failure(claim.job_id, attempt=claim.attempt,
                             owner=owner, exit_code=exit_code, reason=reason,
                             error_tail=error_tail, log_key=log_key,
                             failures=failures)
        if lost:
            return "lost"
        return "dead" if dead else "backoff"

    def _record_failure(self, job_id: str, *, attempt: int,
                        owner: dict[str, Any] | None, exit_code: int | None,
                        reason: str, error_tail: str, log_key: str | None,
                        failures: int) -> bool:
        """Append to ``failed/<id>.json``; best effort, never raises.

        The lease carries the counters that decide retries; this document
        is the error history, so losing one entry to a store outage must
        not abandon a claim or leave a dead lease without its marker.
        """

        try:
            self._append_failure(job_id, attempt=attempt, owner=owner,
                                 exit_code=exit_code, reason=reason,
                                 error_tail=error_tail, log_key=log_key,
                                 failures=failures)
            return True
        except Exception as error:
            print(f"woof energy queue: failure record for {job_id} not "
                  f"written ({type(error).__name__}: {error})",
                  file=sys.stderr)
            return False

    def _append_failure(self, job_id: str, *, attempt: int,
                        owner: dict[str, Any] | None, exit_code: int | None,
                        reason: str, error_tail: str, log_key: str | None,
                        failures: int) -> None:
        key = f"failed/{job_id}.json"
        entry = {"attempt": attempt, "owner": owner, "exit_code": exit_code,
                 "reason": reason, "error_tail": error_tail[-ERROR_TAIL_BYTES:],
                 "log_key": log_key, "at": self.clock(),
                 "at_utc": iso_utc(self.clock())}
        for _ in range(20):
            found = self.store.get(key)
            if found is None:
                document = {"schema": FAILED_SCHEMA, "job_id": job_id,
                            "failures": 0, "attempts": []}
            else:
                document = decode(found[0], key)
            document["failures"] = max(int(document.get("failures", 0)) + 1,
                                       failures)
            document["attempts"] = (list(document.get("attempts", []))
                                    + [entry])[-FAILED_HISTORY_LIMIT:]
            try:
                if found is None:
                    self.store.put_if_absent(key, encode(document))
                else:
                    self.store.put_if_match(key, encode(document), found[1])
                return
            except PreconditionFailed:
                time.sleep(random.uniform(0.01, 0.1))
        raise QueueError(f"{key} kept changing; failure record not written")

    def _mark_dead(self, job_id: str, *, failures: int, reason: str,
                   error_tail: str) -> None:
        now = self.clock()
        document = {"schema": DEAD_SCHEMA, "job_id": job_id,
                    "failures": failures,
                    "max_attempts": self.settings().max_attempts,
                    "reason": reason, "error_tail": error_tail,
                    "at": now, "at_utc": iso_utc(now)}
        try:
            self.store.put_if_absent(f"dead/{job_id}.json", encode(document))
        except PreconditionFailed:
            pass

    def put_log(self, job_id: str, name: str, data: bytes) -> str:
        key = f"logs/{job_id}/{_safe(name)}"
        self.store.put(key, data)
        return key

    # -- status and requeue ----------------------------------------------

    def status(self, *, include_jobs: bool = False) -> dict[str, Any]:
        settings = self.settings()
        now = self.clock()
        view = self._snapshot()
        failed = _ids(self.store.list("failed/"), "failed/")
        counts = {name: 0 for name in ("pending", "running", "expired",
                                       "backoff", "done", "dead")}
        jobs = []
        workers = []
        for job_id in sorted(view["jobs"]):
            lease = None
            if job_id in view["done"]:
                state = "done"
            elif job_id in view["dead"]:
                state = "dead"
            elif job_id in view["leases"]:
                found = self.store.get(f"leases/{job_id}.json")
                lease = None if found is None else decode(
                    found[0], f"leases/{job_id}.json")
                state = "pending" if lease is None else self.lease_view(
                    lease, now)
                if state == "claimable":
                    state = "pending"
            else:
                state = "pending"
            counts[state] += 1
            if state == "running" and lease is not None:
                owner = lease.get("owner") or {}
                workers.append({
                    "job_id": job_id, "attempt": lease.get("attempt"),
                    "worker_id": owner.get("worker_id"),
                    "host": owner.get("host"), "pid": owner.get("pid"),
                    "gpu_uuid": owner.get("gpu_uuid"),
                    "expires_in_s": round(lease["expires_at"] - now, 1)})
            if include_jobs:
                row: dict[str, Any] = {"job_id": job_id, "state": state,
                                       "failed_attempts": job_id in failed}
                if lease is not None:
                    row.update({k: lease.get(k) for k in (
                        "attempt", "failures", "preemptions",
                        "expires_at_utc", "not_before_utc", "note")})
                    row["owner"] = lease.get("owner")
                jobs.append(row)
        orphans = sorted((view["done"] | view["dead"]) - view["jobs"])
        summary: dict[str, Any] = {
            "schema": STATUS_SCHEMA, "queue": self.url,
            "settings": settings.document(), "total": len(view["jobs"]),
            "counts": counts, "with_failures": len(failed & view["jobs"]),
            "drained": counts["done"] + counts["dead"] == len(view["jobs"]),
            "active_workers": workers, "orphan_markers": orphans,
            "at_utc": iso_utc(now),
        }
        if include_jobs:
            summary["jobs"] = jobs
        return summary

    def requeue(self, job_ids: Iterable[str] = (), *,
                all_dead: bool = False) -> dict[str, Any]:
        """Make dead or backed-off jobs claimable now.

        A dead job loses its ``dead/`` marker and its lease (the failure
        count starts again; ``failed/`` keeps the history).  A job in
        backoff keeps its count and loses only the wait.  Done and running
        jobs are refused.
        """

        now = self.clock()
        view = self._snapshot()
        targets = list(dict.fromkeys(job_ids))
        if all_dead:
            targets += sorted(view["dead"] - set(targets))
        if not targets:
            raise RequeueRefused("name jobs with --job or pass --all-dead")
        results: dict[str, str] = {}
        for job_id in targets:
            if job_id not in view["jobs"]:
                raise RequeueRefused(f"job {job_id!r} is not in {self.url}")
            if job_id in view["done"]:
                raise RequeueRefused(
                    f"job {job_id!r} is done; completion is exactly once and "
                    "requeue will not undo it")
        for job_id in targets:
            key = f"leases/{job_id}.json"
            if job_id in view["dead"]:
                self.store.delete(key)
                self.store.delete(f"dead/{job_id}.json")
                results[job_id] = "requeued-from-dead"
                continue
            found = self.store.get(key)
            if found is None:
                results[job_id] = "already-pending"
                continue
            lease = decode(found[0], key)
            state = self.lease_view(lease, now)
            if state == "running":
                raise RequeueRefused(
                    f"job {job_id!r} is running (lease until "
                    f"{lease.get('expires_at_utc')}); requeue refuses to "
                    "start a second copy")
            if state == "backoff":
                lease = dict(lease, not_before=now, not_before_utc=iso_utc(now),
                             revision=int(lease.get("revision", 0)) + 1,
                             updated_at=now, updated_at_utc=iso_utc(now))
                try:
                    self.store.put_if_match(key, encode(lease), found[1])
                except PreconditionFailed:
                    raise RequeueRefused(
                        f"job {job_id!r} changed while it was requeued; "
                        "run requeue again") from None
                results[job_id] = "backoff-cleared"
            elif state == "dead":
                self.store.delete(key)
                results[job_id] = "requeued-from-dead"
            else:
                results[job_id] = "already-pending"
        return {"schema": REQUEUE_SCHEMA, "queue": self.url, "jobs": results}


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)[:200] or "_"


# --------------------------------------------------------------------------
# CLI (``woof energy queue ...``)
# --------------------------------------------------------------------------


def _emit(document: dict[str, Any]) -> None:
    print(json.dumps(document, sort_keys=True, indent=2))


def _cmd_init(args) -> int:
    queue = WorkQueue.open(args.queue)
    settings = QueueSettings(
        max_attempts=args.max_attempts, lease_s=args.lease_s,
        heartbeat_s=(args.heartbeat_s if args.heartbeat_s is not None
                     else min(DEFAULT_HEARTBEAT_S, args.lease_s / 4.0)),
        backoff_base_s=args.backoff_base_s,
        backoff_max_s=args.backoff_max_s)
    _emit(queue.init(settings))
    return 0


def _cmd_push(args) -> int:
    queue = WorkQueue.open(args.queue)
    queue.settings()
    jobs = read_job_files(args.jobs, schema=args.schema)
    results: dict[str, str] = {}
    conflicts: list[str] = []
    for job_id, document in jobs:
        try:
            results[job_id] = queue.push(job_id, document)
        except JobConflict as error:
            results[job_id] = "conflict"
            conflicts.append(str(error))
    counts: dict[str, int] = {}
    for outcome in results.values():
        counts[outcome] = counts.get(outcome, 0) + 1
    _emit({"schema": PUSH_SCHEMA, "queue": queue.url, "counts": counts,
           "jobs": results})
    for message in conflicts:
        print(f"woof energy queue push: REFUSED: {message}", file=sys.stderr)
    return 2 if conflicts else 0


def _cmd_status(args) -> int:
    _emit(WorkQueue.open(args.queue).status(include_jobs=args.jobs))
    return 0


def _cmd_requeue(args) -> int:
    queue = WorkQueue.open(args.queue)
    queue.settings()
    _emit(queue.requeue(args.job or (), all_dead=args.all_dead))
    return 0


_COMMANDS = {"init": _cmd_init, "push": _cmd_push, "status": _cmd_status,
             "requeue": _cmd_requeue}


def main(args: argparse.Namespace) -> int:
    command = getattr(args, "queue_command", None)
    if command is None:
        print("woof energy queue: name a subcommand: init, push, status, "
              "requeue", file=sys.stderr)
        return 2
    try:
        return _COMMANDS[command](args)
    except QueueError as error:
        print(f"woof energy queue {command}: REFUSED: "
              f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2


__all__ = [
    "Claim", "DEAD_SCHEMA", "DONE_SCHEMA", "FAILED_SCHEMA", "FileStore",
    "GCSStore", "JOB_SCHEMA", "JobConflict", "JobSpecError", "LEASE_SCHEMA",
    "LeaseLost", "ObjectStore", "PERMANENT_EXIT_CODES", "PreconditionFailed",
    "QUEUE_SCHEMA", "QueueBackendUnavailable", "QueueConfigConflict",
    "QueueError", "QueueNotInitialized", "QueueSettings",
    "QueueSettingsError", "QueueURLError", "RequeueRefused", "S3Store",
    "STATUS_SCHEMA", "WorkQueue", "declared_gpu_mem_gib",
    "job_from_document", "main", "open_store",
    "read_job_files", "tail_text",
]
