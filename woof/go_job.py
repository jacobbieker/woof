"""``woof go-job JOB.json``: one ``woof go`` run as a batch job.

A fleet of independent forecasts -- one track run, one Icechunk
repository each -- is the same command run many times with different
inputs on machines nobody logs into.  ``woof go`` is that command; what it
lacks for a fleet is everything around it: fetching a config and its
companions from object storage onto local scratch, a fixed output contract
(Icechunk, no pictures by default), an exit status a scheduler can act on,
and a small machine-readable record of what happened.  This module is that
wrapper and nothing else: it never changes what ``woof go`` does.

A job document (:data:`JOB_SCHEMA`) names::

    {
      "schema": "woof.go-job.v1",
      "job_id": "run-001",
      "config": "s3://YOUR-BUCKET/inputs/run-001.toml",
      "files": [{"url": "s3://YOUR-BUCKET/inputs/run-001.namelist.wps"}],
      "icechunk_url": "s3://YOUR-BUCKET/outputs/run-001.icechunk",
      "products": "none",
      "go_args": ["--keep-checkpoints", "1"],
      "result_url": "s3://YOUR-BUCKET/outputs/run-001.result.json",
      "resources": {"gpu_mem_gib": 38.5}
    }

``config`` is an ``s3://``, ``https://`` or ``file://`` URL or a path (a
relative path or URL is resolved against the job document's own location);
``config_toml`` instead carries the TOML inline.  ``files`` are companions
staged beside the config under ``path`` (default: the URL's file name), each
fetched from its ``url`` or written from inline ``content``, so a
config's ``<stem>.namelist.wps`` or an eta-level file is found where a
local run would find it.  ``icechunk_url`` defaults to
``$WOOF_ENERGY_OUTPUT_URL/<job_id>.icechunk`` (the variable every rendered
deployment sets).  ``go_args`` is held to :data:`GO_ARGS`; the flags this
wrapper owns (``--icechunk``, ``--products``, ``--outdir``) are refused there.
``resources.gpu_mem_gib`` is what ``woof energy worker`` packs by, and here
it is also checked against the card: a job routed to a GPU smaller than it
declares is refused before anything is fetched.

What ``woof go-job`` does, in order:

1. reads the job (a path or URL; a :data:`LIST_SCHEMA` document or a JSON
   list of jobs is indexed by ``--index``, else ``$AWS_BATCH_JOB_ARRAY_INDEX``
   or ``$WOOF_WORKER_INDEX``, which is how one AWS Batch array job runs a
   whole list of runs);
2. stages the config and companions into ``WORKDIR/inputs``;
3. runs ``woof go CONFIG --outdir WORKDIR/go --icechunk URL --products P
   [go_args]`` in this process group (so a worker's or scheduler's SIGTERM
   reaches it; a SIGTERM delivered to this process is forwarded);
4. when ``woof go`` exits 73 -- the forecast is complete, its Icechunk
   export is not -- resumes the export from the run folder still on local
   disk (``woof icechunk-export``) up to :data:`EXPORT_RESUME_ATTEMPTS`
   times, so a store hiccup at the end does not cost a whole forecast;
5. writes the result (:data:`RESULT_SCHEMA`: status, exit codes, timings,
   the store URL, the last committed frame and snapshot) to ``--result``
   (default ``WORKDIR/result.json``), to stdout, and to the job's
   ``result_url`` when it names one.

Exit status (what ``woof energy worker`` and AWS Batch retry on):
0 complete; 2 refused -- a bad job, an unreachable scheme, a card too
small, an input the store says does not exist, frames the Icechunk store
refuses, or ``woof go``'s own refusal -- never worth a retry
(:data:`woof.energy.workqueue.PERMANENT_EXIT_CODES`); 73 the forecast ran
but its export is still incomplete after the resumes (retryable; the store
is idempotent, but a retry on another machine has no local run folder and
runs the whole forecast again); 75 stopped by SIGTERM, an input that could
not be fetched, or a host without a transfer tool (retryable); anything
else is ``woof go``'s own status, retryable.

As a ``woof energy worker`` runner::

    woof energy worker --queue URL \\
      --runner-cmd "{python} -m woof go-job {job} --workdir {workdir} --result {result}"

**Credentials.**  None are read, written or printed here.  ``s3://``
transfers use boto3 when installed, else ``rclone`` or the ``aws`` CLI,
each with the ambient identity (instance role, job role, ``AWS_PROFILE``,
the usual ``AWS_*`` environment); ``woof go`` and its exporter do the same.
A URL carrying credentials (``user:pass@``) or a query string (a presigned
URL's signature) is refused, so no secret can reach a log or a result.
"""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import unquote, urlsplit

from woof.explain import explain_enabled, layered, render
from woof.explain import split as split_layers

JOB_SCHEMA = "woof.go-job.v1"
LIST_SCHEMA = "woof.go-job-list.v1"
RESULT_SCHEMA = "woof.go-job.result.v1"
PUSH_SCHEMA = "woof.go-job-push.v1"

#: Keys a job document may carry.  An unknown key is refused by name, so a
#: misspelt ``icechunk`` cannot silently fall back to the default store.
JOB_KEYS = frozenset({
    "schema", "job_id", "config", "config_toml", "config_name", "files",
    "icechunk_url", "products", "go_args", "result_url", "resources",
    "depends_on", "description", "labels",
})

#: ``woof go`` flags a job may add, and whether each takes a value.
#: Everything else is refused: a flag that moves outputs, restarts from a
#: path on someone else's disk or turns the run into a dry run is not a
#: per-run knob.
GO_ARGS: dict[str, bool] = {
    "--cycle": True,
    "--transport": True,
    "--keep-checkpoints": True,
    "--devices": True,
    "--late-after-minutes": True,
    "--verify-inputs": True,
    "--whole-cycle": False,
    "--no-memory-gate": False,
    "--no-verify-visuals": False,
}

#: Flags this wrapper sets itself, and the job key that sets them instead.
OWNED_GO_ARGS = {"--icechunk": "icechunk_url", "--products": "products",
                 "--outdir": "the --workdir of woof go-job"}

EXIT_OK = 0
EXIT_REFUSED = 2
#: :data:`woof.icechunk_export.EXPORT_INCOMPLETE_EXIT_CODE`, spelled here
#: so importing this module does not import the exporter.
EXIT_EXPORT_INCOMPLETE = 73
#: sysexits EX_TEMPFAIL: stopped by SIGTERM, or an input that would not
#: come; ``woof energy worker`` stops with the same code.
EXIT_TEMPFAIL = 75

#: Export resumes after ``woof go`` exits 73, and the waits between them.
EXPORT_RESUME_ATTEMPTS = 3
EXPORT_RESUME_WAIT_S = (30.0, 120.0, 300.0)

#: Attempts at one input transfer, with 2, 4, 8 s between them.
TRANSFER_ATTEMPTS = 4
HTTP_TIMEOUT_S = 60.0

#: Variables a scheduler sets for an array child's index, in the order read.
INDEX_ENV = ("AWS_BATCH_JOB_ARRAY_INDEX", "WOOF_WORKER_INDEX")
#: Attempt counters a scheduler sets (the worker's, then AWS Batch's).
ATTEMPT_ENV = ("WOOF_ENERGY_JOB_ATTEMPT", "AWS_BATCH_JOB_ATTEMPT")
#: The output prefix every ``woof energy deploy render`` worker gets.
OUTPUT_ENV = "WOOF_ENERGY_OUTPUT_URL"

_URL_SCHEMES = ("s3", "gs", "https", "file")
_STORE_SCHEMES = ("s3", "gs", "file")


class GoJobRefused(ValueError):
    """The job cannot run as written; the message says what to change."""


class TransferFailed(RuntimeError):
    """An input or result transfer failed after its retries (retryable)."""


class HostUnequipped(TransferFailed):
    """This host has no tool for a scheme: a property of the machine, not
    the job, so retryable elsewhere and not worth retrying here."""


class InputMissing(GoJobRefused):
    """The store answered that the object does not exist (404): no retry
    can make it appear."""


#: S3 error codes for an object that does not exist
#: (:data:`woof.energy.workqueue._S3_MISSING`).
_S3_MISSING = frozenset({"NoSuchKey", "404", "NotFound"})


def _note(message: str) -> None:
    print(f"woof go-job: {message}", file=sys.stderr, flush=True)


def _utc(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# URLs


def _is_url(text: str) -> bool:
    head = text.split("://", 1)[0]
    return "://" in text and head.isalnum()


def check_url(url: str, what: str, *, schemes=_URL_SCHEMES) -> str:
    """``url``'s scheme, or :class:`GoJobRefused` naming what is wrong.

    Credentials in the authority and query strings are refused outright:
    a presigned URL's signature or a ``user:pass@`` would otherwise be
    printed in a log line and kept in the result document."""

    parts = urlsplit(url)
    scheme = parts.scheme
    if scheme not in schemes:
        raise GoJobRefused(layered(
            f"{what}: {url.split('://', 1)[0]}:// is not one of "
            f"{', '.join(s + '://' for s in schemes)}; use one of those or "
            "a local path",
            "A batch job fetches its inputs with the machine's own identity "
            "(instance or job role) from object storage or a public HTTPS "
            "host; any other scheme would need a credential in the job."))
    if parts.username or parts.password or "@" in parts.netloc:
        raise GoJobRefused(layered(
            f"{what}: the URL carries credentials; remove them and let the "
            "instance role, job role or AWS_PROFILE authenticate",
            "Job documents are stored in queues and copied into logs and "
            "results; a credential in one would leak to everyone who can "
            "read those."))
    if parts.query or parts.fragment:
        raise GoJobRefused(layered(
            f"{what}: the URL has a query string or fragment; name the "
            "object plainly (s3://bucket/key) and grant the job role read "
            "access",
            "A presigned URL's signature is a credential, and this job's "
            "URLs are printed in logs and kept in its result."))
    if any(c.isspace() for c in url):
        raise GoJobRefused(f"{what}: {url!r} contains whitespace")
    if scheme in ("s3", "gs") and not parts.netloc:
        raise GoJobRefused(f"{what}: {url!r} names no bucket")
    if scheme in ("s3", "gs", "https") and not parts.path.strip("/"):
        raise GoJobRefused(f"{what}: {url!r} names no object")
    if scheme == "file" and (parts.netloc not in ("", "localhost")
                             or not parts.path.startswith("/")):
        raise GoJobRefused(f"{what}: a file:// URL takes an absolute path "
                           f"(file:///abs/path); got {url!r}")
    return scheme


def _file_url_path(url: str) -> Path:
    return Path(unquote(urlsplit(url).path))


def resolve_ref(ref: str, base: str | Path | None) -> str:
    """``ref`` as an absolute URL or path: a relative path is taken against
    ``base`` (a job document's directory or URL prefix)."""

    if _is_url(ref) or base is None:
        return ref
    if os.path.isabs(ref):
        return ref
    if isinstance(base, Path):
        return str(base / ref)
    parts = urlsplit(base)
    joined = posixpath.normpath(posixpath.join(parts.path, ref))
    return f"{parts.scheme}://{parts.netloc}{joined}"


def _parent_of(location: str) -> str | Path:
    """The directory a job document's relative references resolve against."""

    if _is_url(location):
        parts = urlsplit(location)
        if parts.scheme == "file":
            return _file_url_path(location).parent
        parent = posixpath.dirname(parts.path) or "/"
        return f"{parts.scheme}://{parts.netloc}{parent}"
    return Path(location).expanduser().resolve().parent


def _basename(ref: str) -> str:
    path = urlsplit(ref).path if _is_url(ref) else ref
    return PurePosixPath(path.replace("\\", "/")).name


def _plain_name(name: Any, what: str) -> str:
    """A relative file path inside the staging directory: no absolute
    path, no ``..``, no empty parts."""

    if not isinstance(name, str) or not name.strip():
        raise GoJobRefused(f"{what} must be a non-empty relative path; "
                           f"got {name!r}")
    pure = PurePosixPath(name)
    if (pure.is_absolute() or "\\" in name or ".." in pure.parts
            or any(c in name for c in "\n\0") or name.startswith("~")):
        raise GoJobRefused(f"{what}: {name!r} must be a relative path "
                           "inside the job's input folder (no '..', no "
                           "absolute path)")
    return str(pure)


# --------------------------------------------------------------------------
# Transfers


def _rclone_remote(url: str) -> str:
    parts = urlsplit(url)
    region = os.environ.get("AWS_REGION") or os.environ.get(
        "AWS_DEFAULT_REGION")
    path = f"{parts.netloc}{parts.path}"
    if parts.scheme == "gs":
        return f":gcs,env_auth=true,bucket_policy_only=true:{path}"
    if region:
        return f":s3,provider=AWS,env_auth=true,region={region}:{path}"
    return f":s3,provider=AWS,env_auth=true:{path}"


def _cloud_copy(source: str, dest: str, *, upload: bool) -> None:
    """One object to or from ``s3://``/``gs://``: boto3 (s3) if installed,
    else rclone, else the aws/gcloud CLI; ambient identity throughout."""

    url = dest if upload else source
    parts = urlsplit(url)
    if parts.scheme == "s3":
        try:
            import boto3
        except ImportError:
            boto3 = None
        if boto3 is not None:
            from woof.energy.workqueue import _s3_error_code

            client = boto3.client("s3")
            key = parts.path.lstrip("/")
            try:
                if upload:
                    client.upload_file(source, parts.netloc, key)
                else:
                    client.download_file(parts.netloc, key, dest)
            except Exception as error:
                code, status = _s3_error_code(error)
                if not upload and (code in _S3_MISSING or status == 404):
                    raise InputMissing(f"{url} does not exist") from error
                raise
            return
    if shutil.which("rclone"):
        remote = _rclone_remote(url)
        argv = (["rclone", "copyto", source, remote] if upload
                else ["rclone", "copyto", remote, dest])
        argv += ["--retries", "1", "--low-level-retries", "5"]
    elif parts.scheme == "s3" and shutil.which("aws"):
        argv = ["aws", "s3", "cp", "--only-show-errors", source, dest]
    elif parts.scheme == "gs" and shutil.which("gcloud"):
        argv = ["gcloud", "storage", "cp", source, dest]
    else:
        # Retryable: the job is fine, this machine is not equipped.
        raise HostUnequipped(
            f"{parts.scheme}:// transfers need boto3, rclone or the "
            f"{'aws' if parts.scheme == 's3' else 'gcloud'} CLI, and this "
            "host has none; install one (the deploy/Dockerfile image ships "
            "boto3 and rclone) or stage the file and give a local path")
    # The CLIs' exit status does not tell a missing object from a network
    # failure, so through them a missing input is retried and ends as
    # TransferFailed (retryable); boto3 above names it InputMissing.
    completed = subprocess.run(argv, stdin=subprocess.DEVNULL,
                               capture_output=True, text=True)
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip()[-400:]
        raise TransferFailed(f"{argv[0]} exited {completed.returncode}: "
                             f"{tail}")


def _https_get(url: str, dest: Path) -> None:
    import urllib.error
    import urllib.request

    request = urllib.request.Request(url, headers={
        "User-Agent": "woof-go-job"})
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) \
                as response, open(dest, "wb") as out:
            shutil.copyfileobj(response, out, 1 << 20)
    except urllib.error.HTTPError as error:
        if error.code in (404, 410):
            raise InputMissing(f"{url} answered HTTP {error.code}") from error
        raise


def _retrying(action: Callable[[], None], what: str,
              sleep: Callable[[float], None] = time.sleep) -> None:
    for attempt in range(1, TRANSFER_ATTEMPTS + 1):
        try:
            action()
            return
        except (GoJobRefused, HostUnequipped):
            raise
        except Exception as error:  # noqa: BLE001 - network, throttling,
            # a CLI failure: all transient until the attempts run out.
            if attempt == TRANSFER_ATTEMPTS:
                raise TransferFailed(
                    f"{what} failed {attempt} times; last: "
                    f"{type(error).__name__}: {error}") from error
            wait = 2.0 ** attempt
            _note(f"{what} failed ({type(error).__name__}); retrying in "
                  f"{wait:g} s")
            sleep(wait)


def fetch(ref: str, dest: Path, *, what: str,
          sleep: Callable[[float], None] = time.sleep) -> int:
    """Copy one input (URL or path) to ``dest`` atomically; its size."""

    dest.parent.mkdir(parents=True, exist_ok=True)
    if _is_url(ref):
        scheme = check_url(ref, what)
        source_path = _file_url_path(ref) if scheme == "file" else None
    else:
        scheme, source_path = "path", Path(ref).expanduser()
    if source_path is not None and not source_path.is_file():
        raise GoJobRefused(f"{what}: {source_path} is not a file")
    fd, staging = tempfile.mkstemp(prefix=f".{dest.name}.", dir=dest.parent)
    os.close(fd)
    staged = Path(staging)
    try:
        if source_path is not None:
            shutil.copyfile(source_path, staged)
        elif scheme == "https":
            _retrying(lambda: _https_get(ref, staged), f"fetching {what}",
                      sleep)
        else:
            _retrying(lambda: _cloud_copy(ref, str(staged), upload=False),
                      f"fetching {what}", sleep)
        os.replace(staged, dest)
    finally:
        staged.unlink(missing_ok=True)
    return dest.stat().st_size


def publish(path: Path, url: str, *, what: str,
            sleep: Callable[[float], None] = time.sleep) -> None:
    """Copy a local file to an ``s3://``, ``gs://`` or ``file://`` URL."""

    scheme = check_url(url, what, schemes=_STORE_SCHEMES)
    if scheme == "file":
        from woof.fetch_guard import atomic_write_bytes

        target = _file_url_path(url)
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(target, path.read_bytes())
        return
    _retrying(lambda: _cloud_copy(str(path), url, upload=True),
              f"writing {what}", sleep)


# --------------------------------------------------------------------------
# The job document


@dataclass(frozen=True)
class Companion:
    """One file staged beside the config: fetched from ``ref``, or written
    from ``content`` carried inline in the job document."""

    ref: str | None
    path: str
    content: str | None = None


@dataclass(frozen=True)
class GoJob:
    """A validated job document (see the module docstring)."""

    job_id: str
    config_ref: str | None
    config_toml: str | None
    config_name: str
    files: tuple[Companion, ...]
    icechunk_url: str
    products: str
    go_args: tuple[str, ...]
    result_url: str | None
    gpu_mem_gib: float | None
    document: dict = field(repr=False, default_factory=dict)


def _go_args(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(a, str) for a in raw):
        raise GoJobRefused(f"go_args must be a list of strings; got {raw!r}")
    out: list[str] = []
    index = 0
    while index < len(raw):
        token = raw[index]
        if any(c in token for c in "\n\0"):
            raise GoJobRefused(f"go_args entry {token!r} has a newline")
        flag, eq, value = token.partition("=")
        if flag in OWNED_GO_ARGS:
            raise GoJobRefused(
                f"go_args may not carry {flag}: set {OWNED_GO_ARGS[flag]} "
                "instead (go-job passes it to woof go itself)")
        if flag not in GO_ARGS:
            raise GoJobRefused(layered(
                f"go_args: {flag!r} is not a flag a job may pass to woof go; "
                f"allowed: {', '.join(sorted(GO_ARGS))}",
                "A job runs unattended on a machine its author never sees, "
                "so the flags it may set are a short list that cannot move "
                "its outputs, point it at another disk's checkpoint or turn "
                "it into a dry run."))
        index += 1
        if GO_ARGS[flag]:
            if not eq:
                if index >= len(raw) or raw[index].startswith("--"):
                    raise GoJobRefused(f"go_args: {flag} needs a value")
                value = raw[index]
                index += 1
            if any(c in value for c in "\n\0") or not value:
                raise GoJobRefused(f"go_args: {flag} value {value!r} is "
                                   "empty or has a newline")
            out.append(f"{flag}={value}")
        else:
            if eq:
                raise GoJobRefused(f"go_args: {flag} takes no value")
            out.append(flag)
    return tuple(out)


def _store_url(url: Any, what: str) -> str:
    if not isinstance(url, str) or not url:
        raise GoJobRefused(f"{what} must be a URL string; got {url!r}")
    check_url(url, what, schemes=_STORE_SCHEMES)
    return url


def parse_job(document: Any, *, base: str | Path | None = None,
              output_prefix: str | None = None,
              source: str = "<job>") -> GoJob:
    """A :class:`GoJob` from a parsed document, or :class:`GoJobRefused`."""

    from woof.energy.workqueue import JobSpecError, job_from_document

    try:
        job_id, document = job_from_document(document, schema=JOB_SCHEMA,
                                             source=source)
    except JobSpecError as error:
        raise GoJobRefused(str(error)) from error
    unknown = sorted(set(document) - JOB_KEYS)
    if unknown:
        raise GoJobRefused(f"{source}: unknown key(s) {', '.join(unknown)}; "
                           f"a {JOB_SCHEMA} job takes "
                           f"{', '.join(sorted(JOB_KEYS))}")
    config, inline = document.get("config"), document.get("config_toml")
    if (config is None) == (inline is None):
        raise GoJobRefused(f"{source}: give exactly one of config (a URL or "
                           "path) and config_toml (the TOML itself)")
    if config is not None and (not isinstance(config, str) or not config):
        raise GoJobRefused(f"{source}: config must be a URL or path string")
    if inline is not None:
        if not isinstance(inline, str) or not inline.strip():
            raise GoJobRefused(f"{source}: config_toml must be TOML text")
        import tomllib
        try:
            tomllib.loads(inline)
        except tomllib.TOMLDecodeError as error:
            raise GoJobRefused(f"{source}: config_toml is not TOML: "
                               f"{error}") from error
    config_ref = None if config is None else resolve_ref(config, base)
    if config_ref is not None and _is_url(config_ref):
        check_url(config_ref, f"{source}: config")
    name = document.get("config_name")
    if name is None:
        name = (_basename(config_ref) if config_ref else "") or \
            f"{job_id}.toml"
    name = _plain_name(name, f"{source}: config_name")
    if "/" in name:
        raise GoJobRefused(f"{source}: config_name {name!r} must be a file "
                           "name, not a path")

    raw_files = document.get("files") or []
    if not isinstance(raw_files, list):
        raise GoJobRefused(f"{source}: files must be a list of "
                           "{url, path} or {path, content} tables")
    files: list[Companion] = []
    seen = {name}
    for index, entry in enumerate(raw_files):
        where = f"{source}: files[{index}]"
        if isinstance(entry, str):
            entry = {"url": entry}
        if (not isinstance(entry, Mapping)
                or set(entry) - {"url", "path", "content"}
                or ("url" in entry) == ("content" in entry)):
            raise GoJobRefused(f"{where} must be a URL string, a {{url, "
                               "path}} table or a {path, content} table")
        if "content" in entry:
            if not isinstance(entry["content"], str) or not entry.get("path"):
                raise GoJobRefused(f"{where}: inline content is a string "
                                   "and needs a path")
            ref, content = None, entry["content"]
            default_name = ""
        else:
            if not isinstance(entry["url"], str):
                raise GoJobRefused(f"{where}: url must be a string")
            ref, content = resolve_ref(entry["url"], base), None
            if _is_url(ref):
                check_url(ref, where)
            default_name = _basename(ref)
        path = _plain_name(entry.get("path") or default_name,
                           f"{where}.path")
        if path in seen:
            raise GoJobRefused(f"{where}: {path!r} is staged twice (or is "
                               "the config's own name)")
        seen.add(path)
        files.append(Companion(ref, path, content))

    icechunk = document.get("icechunk_url")
    if icechunk is None:
        prefix = output_prefix or os.environ.get(OUTPUT_ENV)
        if not prefix:
            raise GoJobRefused(layered(
                f"{source}: no icechunk_url, and no --output-prefix or "
                f"{OUTPUT_ENV} to derive one from; add "
                f"\"icechunk_url\": \"s3://BUCKET/PREFIX/{job_id}.icechunk\"",
                "Every run writes its own repository; go-job names it "
                f"<prefix>/<job_id>.icechunk when the job leaves it out."))
        icechunk = f"{prefix.rstrip('/')}/{job_id}.icechunk"
    icechunk = _store_url(icechunk, f"{source}: icechunk_url")
    result_url = document.get("result_url")
    if result_url is not None:
        result_url = _store_url(result_url, f"{source}: result_url")

    products = document.get("products", "none")
    if (not isinstance(products, str) or not products.strip()
            or products.startswith("-") or any(c.isspace() for c in products)):
        raise GoJobRefused(f"{source}: products must be 'none', 'all' or a "
                           f"comma-separated list; got {products!r}")
    from woof.energy.workqueue import declared_gpu_mem_gib
    try:
        need = declared_gpu_mem_gib(document, source=source)
    except JobSpecError as error:
        raise GoJobRefused(str(error)) from error
    labels = document.get("labels")
    if labels is not None and not (isinstance(labels, dict) and all(
            isinstance(v, str) for v in labels.values())):
        raise GoJobRefused(f"{source}: labels must be a table of strings")
    return GoJob(job_id=job_id, config_ref=config_ref, config_toml=inline,
                 config_name=name, files=tuple(files),
                 icechunk_url=icechunk, products=products,
                 go_args=_go_args(document.get("go_args")),
                 result_url=result_url, gpu_mem_gib=need, document=document)


def _read_document(ref: str, scratch: Path) -> tuple[Any, str | Path]:
    """A job or list document (path or URL) and its resolution base."""

    if _is_url(ref):
        check_url(ref, "the job document")
        if urlsplit(ref).scheme == "file":
            path = _file_url_path(ref)
        else:
            path = scratch / "job-source.json"
            fetch(ref, path, what="the job document")
    else:
        path = Path(ref).expanduser()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise GoJobRefused(f"{ref}: no such job document") from None
    try:
        return json.loads(text), _parent_of(ref)
    except json.JSONDecodeError as error:
        raise GoJobRefused(f"{ref}: not JSON: {error}") from error


def array_index(explicit: int | None,
                environ: Mapping[str, str] = os.environ) -> int | None:
    """``--index``, else the scheduler's array index, else None."""

    if explicit is not None:
        return explicit
    for name in INDEX_ENV:
        value = environ.get(name, "").strip()
        if value:
            if not value.isdigit():
                raise GoJobRefused(f"{name}={value!r} is not an array index")
            return int(value)
    return None


def select_job(document: Any, base: str | Path, index: int | None,
               scratch: Path, *, source: str,
               explicit: bool = False) -> tuple[Any, str | Path, str]:
    """The one job a document names: itself, or entry ``index`` of a list
    (an entry may be a job document or a reference to one).  ``explicit``
    says the index came from ``--index`` rather than the environment: a
    worker fleet's array index is inherited by every runner and means
    nothing to a single job document."""

    entries = None
    if isinstance(document, list):
        entries = document
    elif isinstance(document, dict) and document.get("schema") == LIST_SCHEMA:
        entries = document.get("jobs")
        if not isinstance(entries, list):
            raise GoJobRefused(f"{source}: a {LIST_SCHEMA} document needs a "
                               "jobs list")
    if entries is None:
        if explicit and index not in (None, 0):
            # One job and an array index above 0: running it again in
            # every child would write one store from many machines.
            raise GoJobRefused(f"{source} is one job; array index {index} "
                               "needs a job list")
        return document, base, source
    if not entries:
        raise GoJobRefused(f"{source}: the job list is empty")
    if index is None:
        if len(entries) != 1:
            raise GoJobRefused(layered(
                f"{source} lists {len(entries)} jobs; pick one with --index "
                "N (0-based), or run it as an array job "
                "(AWS_BATCH_JOB_ARRAY_INDEX)",
                "A job list is how one AWS Batch array job runs many "
                "runs: child N runs entry N."))
        index = 0
    if not 0 <= index < len(entries):
        raise GoJobRefused(f"{source}: index {index} is outside the list's "
                           f"{len(entries)} jobs (0-{len(entries) - 1})")
    entry = entries[index]
    where = f"{source}[{index}]"
    if isinstance(entry, str):
        ref = resolve_ref(entry, base)
        document, base = _read_document(ref, scratch)
        return document, base, ref
    return entry, base, where


# --------------------------------------------------------------------------
# Running


def card_memory_gib(environ: Mapping[str, str] = os.environ) -> float | None:
    """The visible card's total memory (GiB), or None when unreadable or
    when this process may not touch a GPU."""

    from woof.local_gpu import no_local_gpu

    if no_local_gpu():
        return None
    from woof.core.preflight import device_physical_total_bytes

    visible = environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    device = visible if visible and visible != "-1" else None
    total = device_physical_total_bytes(device_id=device)
    return None if total is None else total / 1024 ** 3


def check_card(job: GoJob, card_gib: float | None) -> None:
    if job.gpu_mem_gib is None or card_gib is None:
        return
    if job.gpu_mem_gib > card_gib:
        raise GoJobRefused(layered(
            f"job {job.job_id} declares resources.gpu_mem_gib "
            f"{job.gpu_mem_gib:g} GiB and this card holds {card_gib:.1f} GiB; "
            "queue it for a larger GPU (the H100 tier in "
            "deploy/examples/aws-track-runs.toml) instead",
            "A forecast that cannot fit fails after its whole preparation; "
            "refusing here costs nothing and is never retried on the same "
            "too-small card."))


def go_command(job: GoJob, config: Path, outdir: Path) -> list[str]:
    """The ``woof go`` argv this job runs."""

    return [sys.executable, "-m", "woof", "go", str(config),
            "--outdir", str(outdir), "--icechunk", job.icechunk_url,
            "--products", job.products, *job.go_args]


def export_status(outdir: Path) -> tuple[Path | None, dict[str, Any]]:
    """The newest ``icechunk-export/status.json`` under ``outdir``."""

    found = sorted(outdir.rglob("icechunk-export/status.json"),
                   key=lambda p: p.stat().st_mtime) if outdir.is_dir() else []
    for path in reversed(found):
        try:
            status = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(status, dict):
            return path.parent, status
    return None, {}


def export_summary(status: Mapping[str, Any]) -> dict[str, Any]:
    """The result's view of an exporter status: state, counts, the last
    committed frame (the latest exported valid time over every group)."""

    groups = status.get("groups") if isinstance(status.get("groups"),
                                                dict) else {}
    latest = {name: entry.get("latest_exported")
              for name, entry in sorted(groups.items())
              if isinstance(entry, dict)}
    stamps = [v for v in latest.values() if isinstance(v, str)]
    return {
        "export_state": status.get("state"),
        "frames_exported": status.get("frames_exported"),
        "frames_found": status.get("frames_found"),
        "frames_refused": status.get("frames_refused"),
        "last_committed_frame": max(stamps) if stamps else None,
        "last_committed_frame_by_group": latest,
        "last_snapshot": status.get("last_snapshot"),
    }


class _Child:
    """The running subprocess, so a SIGTERM to go-job reaches it too."""

    def __init__(self) -> None:
        self.process: subprocess.Popen | None = None
        self.stopped_by: int | None = None

    def handle(self, signum, _frame) -> None:
        self.stopped_by = signum
        process = self.process
        if process is not None and process.poll() is None:
            try:
                process.send_signal(signum)
            except ProcessLookupError:
                pass

    def run(self, argv: Sequence[str], *, cwd: Path) -> int:
        if self.stopped_by is not None:
            # Stopped while staging or between stages: start nothing.
            return -int(self.stopped_by)
        environment = {**os.environ, "PYTHONSAFEPATH": "1"}
        # Not a new session: a worker's or scheduler's SIGTERM to this
        # process group must reach woof go and its stages directly.
        self.process = subprocess.Popen(list(argv), cwd=cwd, env=environment,
                                        stdin=subprocess.DEVNULL)
        if self.stopped_by is not None:
            # The signal landed while Popen ran and found no child to pass
            # it to; pass it now.
            self.handle(self.stopped_by, None)
        try:
            while True:
                try:
                    return self.process.wait()
                except KeyboardInterrupt:
                    # The terminal sent SIGINT to the whole group; wait for
                    # woof go to finish its own interrupt handling.
                    self.stopped_by = signal.SIGINT
        finally:
            self.process = None

    def pause(self, seconds: float) -> None:
        """Sleep, cut short by a stop (a plain sleep resumes after the
        handler returns, and the scheduler's grace would run out first)."""

        deadline = time.monotonic() + seconds
        while self.stopped_by is None and time.monotonic() < deadline:
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))


def _exit_of(code: int) -> int:
    """A child's status as a process exit status (signals as 128+N)."""

    return 128 - code if code < 0 else code


def _attempt(environ: Mapping[str, str] = os.environ) -> int | None:
    for name in ATTEMPT_ENV:
        value = environ.get(name, "").strip()
        if value.isdigit():
            return int(value)
    return None


@dataclass
class Outcome:
    exit_code: int
    status: str
    go_exit_code: int | None = None
    error: str | None = None


def run_job(job: GoJob, workdir: Path, *, child: _Child | None = None,
            sleep: Callable[[float], None] | None = None,
            card_gib: Callable[[], float | None] = card_memory_gib
            ) -> tuple[Outcome, dict[str, Any]]:
    """Stage, run and settle one job; ``(outcome, result fields)``.

    ``sleep`` (default: the child's stop-aware pause) waits between
    transfer retries and before each export resume."""

    child = child or _Child()
    sleep = sleep or child.pause
    timings: dict[str, float] = {}
    fields: dict[str, Any] = {"timings_s": timings}
    started = time.monotonic()

    check_card(job, card_gib())
    inputs = workdir / "inputs"
    config = inputs / job.config_name
    staged: list[dict[str, Any]] = []
    try:
        if job.config_toml is not None:
            from woof.fetch_guard import atomic_write_text

            inputs.mkdir(parents=True, exist_ok=True)
            atomic_write_text(config, job.config_toml)
            staged.append({"path": job.config_name, "source": "config_toml",
                           "bytes": config.stat().st_size})
        else:
            size = fetch(job.config_ref, config, what="config", sleep=sleep)
            staged.append({"path": job.config_name, "source": job.config_ref,
                           "bytes": size})
        for companion in job.files:
            target = inputs / companion.path
            if companion.content is not None:
                from woof.fetch_guard import atomic_write_text

                target.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_text(target, companion.content)
                size, origin = target.stat().st_size, "inline content"
            else:
                size = fetch(companion.ref, target, what=companion.path,
                             sleep=sleep)
                origin = companion.ref
            staged.append({"path": companion.path, "source": origin,
                           "bytes": size})
    except TransferFailed as error:
        timings["stage"] = round(time.monotonic() - started, 3)
        fields["inputs"] = staged
        return Outcome(EXIT_TEMPFAIL, "input-unavailable",
                       error=str(error)), fields
    fields["inputs"] = staged
    for entry in staged:
        _note(f"staged {entry['path']} ({entry['bytes']} bytes) from "
              f"{entry['source']}")
    timings["stage"] = round(time.monotonic() - started, 3)

    outdir = workdir / "go"
    outdir.mkdir(parents=True, exist_ok=True)
    argv = go_command(job, config, outdir)
    fields["go_command"] = ["woof", *argv[3:]]
    _note(f"running woof go for {job.job_id} -> {job.icechunk_url}")
    go_started = time.monotonic()
    code = _exit_of(child.run(argv, cwd=workdir))
    timings["go"] = round(time.monotonic() - go_started, 3)
    go_code = code

    export_dir, status = export_status(outdir)
    if export_dir is not None:
        fields["run_dir"] = str(export_dir.parent)
    export_refused = False
    if code == EXIT_EXPORT_INCOMPLETE and export_dir is not None \
            and child.stopped_by is None:
        resume_started = time.monotonic()
        for attempt in range(1, EXPORT_RESUME_ATTEMPTS + 1):
            wait = EXPORT_RESUME_WAIT_S[min(attempt, len(
                EXPORT_RESUME_WAIT_S)) - 1]
            _note(f"the forecast finished and its Icechunk export did not; "
                  f"resuming it in {wait:g} s (attempt {attempt} of "
                  f"{EXPORT_RESUME_ATTEMPTS})")
            sleep(wait)
            resumed = _exit_of(child.run(
                [sys.executable, "-m", "woof", "icechunk-export",
                 str(export_dir)], cwd=workdir))
            if resumed == 0:
                code = EXIT_OK
                break
            if child.stopped_by is not None:
                break
            if resumed == EXIT_REFUSED:
                # Refused frames (a grid or time the store will not take)
                # come back the same on every try, and a retry elsewhere
                # would rerun the whole forecast to meet the same refusal.
                export_refused = True
                break
        timings["export_resume"] = round(time.monotonic() - resume_started, 3)
        export_dir, status = export_status(outdir)
    fields.update(export_summary(status))

    if code == EXIT_OK:
        # Complete is complete, even with a stop arriving just after: a
        # retry would run a finished run again.
        outcome = Outcome(EXIT_OK, "complete", go_code)
    elif child.stopped_by is not None:
        name = signal.Signals(child.stopped_by).name
        outcome = Outcome(EXIT_TEMPFAIL if child.stopped_by != signal.SIGINT
                          else 130, "stopped", go_code,
                          f"stopped by {name}")
    elif export_refused:
        outcome = Outcome(EXIT_REFUSED, "export-refused", go_code,
                          "the forecast finished; the Icechunk store "
                          "refused frames (see the export log)")
    elif code == EXIT_REFUSED:
        outcome = Outcome(EXIT_REFUSED, "refused", go_code,
                          "woof go refused the job (see its log)")
    elif code == EXIT_EXPORT_INCOMPLETE:
        outcome = Outcome(EXIT_EXPORT_INCOMPLETE, "export-incomplete",
                          go_code, "the forecast finished; its Icechunk "
                                   "export is incomplete")
    else:
        outcome = Outcome(code, "failed", go_code,
                          f"woof go exited {code}")
    return outcome, fields


def default_workdir(job_id: str,
                    environ: Mapping[str, str] = os.environ) -> Path:
    scratch = environ.get("WOOF_SCRATCH")
    if scratch:
        return Path(scratch) / "go-job" / job_id
    return Path.cwd() / f"go-job-{job_id}"


def main(args) -> int:
    started = time.time()
    clock = time.monotonic()
    child = _Child()
    previous = signal.signal(signal.SIGTERM, child.handle)
    result: dict[str, Any] = {"schema": RESULT_SCHEMA,
                              "started_at_utc": _utc(started),
                              "host": socket.gethostname(),
                              "attempt": _attempt()}
    job: GoJob | None = None
    workdir: Path | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="woof-go-job-") as scratch:
            document, base = _read_document(str(args.job), Path(scratch))
            index = array_index(args.index)
            document, base, source = select_job(
                document, base, index, Path(scratch), source=str(args.job),
                explicit=args.index is not None)
            if index is not None and source != str(args.job):
                result["index"] = index
            job = parse_job(document, base=base,
                            output_prefix=args.output_prefix, source=source)
        result.update({"job_id": job.job_id,
                       "icechunk_url": job.icechunk_url})
        workdir = (args.workdir or default_workdir(job.job_id)).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        result["workdir"] = str(workdir)
        outcome, fields = run_job(job, workdir, child=child)
        result.update(fields)
    except GoJobRefused as error:
        outcome = Outcome(EXIT_REFUSED, "refused", error=str(error))
        print(render(str(error), explain=explain_enabled(args),
                     command="woof go-job"), file=sys.stderr)
    except TransferFailed as error:
        outcome = Outcome(EXIT_TEMPFAIL, "input-unavailable",
                          error=str(error))
        _note(str(error))
    except OSError as error:
        # Scratch full or unwritable, a process that would not start: a
        # property of this machine, so worth a retry elsewhere.
        outcome = Outcome(1, "failed", error=f"{type(error).__name__}: "
                                             f"{error}")
    finally:
        signal.signal(signal.SIGTERM, previous)
    finished = time.time()
    result.setdefault("timings_s", {})["total"] = round(
        time.monotonic() - clock, 3)
    # The action half only: the result is read by programs, the
    # explanation is for a person at a terminal.
    error = split_layers(outcome.error)[0] if outcome.error else None
    result.update({"status": outcome.status, "exit_code": outcome.exit_code,
                   "go_exit_code": outcome.go_exit_code, "error": error,
                   "finished_at_utc": _utc(finished)})
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    result_path = args.result or (workdir / "result.json" if workdir
                                  else None)
    if result_path is not None:
        from woof.fetch_guard import atomic_write_text

        result_path = Path(result_path)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(result_path, text)
        if job is not None and job.result_url:
            try:
                publish(result_path, job.result_url, what="the result")
            except (TransferFailed, GoJobRefused) as failure:
                # The run's outcome stands; the copy is a convenience.
                _note(f"could not write the result to {job.result_url}: "
                      f"{split_layers(str(failure))[0]}")
    sys.stdout.write(text)
    sys.stdout.flush()
    if outcome.status != "complete":
        _note(f"{outcome.status} (exit {outcome.exit_code})"
              + (f": {error}" if error else ""))
    return outcome.exit_code


# --------------------------------------------------------------------------
# Getting inputs to the machines


def push_inputs(local: Path, prefix: str, *, dry_run: bool = False,
                sleep: Callable[[float], None] = time.sleep
                ) -> list[dict[str, Any]]:
    """Copy every file under ``local`` (dot files skipped) to
    ``prefix/<relative path>``; the manifest of what went where.

    This is how inputs that live only on one workstation -- generated
    configs, their companions, a path a nest follows -- reach batch
    machines: through object storage the job role can read, named by the
    caller, never through the repository."""

    import hashlib

    check_url(prefix, "the input prefix", schemes=_STORE_SCHEMES)
    root = Path(local).expanduser()
    if not root.is_dir():
        raise GoJobRefused(f"{root} is not a directory")
    files = sorted(path for path in root.rglob("*") if path.is_file()
                   and not any(part.startswith(".")
                               for part in path.relative_to(root).parts))
    if not files:
        raise GoJobRefused(f"{root} holds no files to push")
    manifest = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        url = f"{prefix.rstrip('/')}/{relative}"
        data = path.read_bytes()
        manifest.append({"path": relative, "url": url, "bytes": len(data),
                         "sha256": hashlib.sha256(data).hexdigest()})
        if not dry_run:
            publish(path, url, what=relative, sleep=sleep)
            _note(f"pushed {relative} ({len(data)} bytes) -> {url}")
    return manifest


def push_main(args) -> int:
    try:
        manifest = push_inputs(args.local, args.prefix,
                               dry_run=bool(args.dry_run))
    except GoJobRefused as error:
        print(render(str(error), explain=explain_enabled(args),
                     command="woof go-job-push"), file=sys.stderr)
        return EXIT_REFUSED
    except TransferFailed as error:
        _note(str(error))
        return 1
    print(json.dumps({"schema": PUSH_SCHEMA, "prefix": args.prefix,
                      "dry_run": bool(args.dry_run), "files": manifest},
                     indent=2, sort_keys=True))
    return EXIT_OK


def register_cli(subparsers) -> argparse.ArgumentParser:
    from woof.cli_numbers import nonnegative_int

    push = subparsers.add_parser(
        "go-job-push",
        help="copy a local folder of job inputs (configs, companions) to an "
             "s3://, gs:// or file:// prefix for woof go-job to fetch",
        description=(
            "Upload every file under LOCAL_DIR to PREFIX/<relative path> "
            "with this machine's own identity (boto3, rclone or the aws "
            "CLI), and print a manifest of URLs, sizes and SHA-256 sums to "
            "reference from job documents.  Dot files are skipped."))
    push.add_argument("local", type=Path, metavar="LOCAL_DIR",
                      help="the folder whose files are pushed")
    push.add_argument("prefix", metavar="PREFIX",
                      help="where they go: s3://BUCKET/PREFIX, "
                           "gs://BUCKET/PREFIX or file:///DIR")
    push.add_argument("--dry-run", action="store_true",
                      help="print the manifest and upload nothing")
    push.set_defaults(func=push_main)

    parser = subparsers.add_parser(
        "go-job",
        help="run one woof go job document (config URL, companions, "
             "Icechunk store) as a batch job: stage, run, export, result "
             "JSON, scheduler exit codes",
        description=(
            "Stage a job's config and companion files onto local disk, run "
            "woof go with its Icechunk store and products, resume an "
            "unfinished export, and write a result JSON.  Exit 0 complete, "
            "2 refused (never retry), 73 export incomplete, 75 stopped or "
            "inputs unavailable; any other status is woof go's own.  As a "
            "worker runner: woof energy worker --runner-cmd \"{python} -m "
            "woof go-job {job} --workdir {workdir} --result {result}\"."))
    parser.add_argument(
        "job", metavar="JOB",
        help=f"a {JOB_SCHEMA} job document, or a {LIST_SCHEMA} list (or "
             "JSON array) of jobs indexed by --index; a path or an s3://, "
             "gs://, https:// or file:// URL")
    parser.add_argument(
        "--workdir", type=Path, default=None, metavar="DIR",
        help="where inputs are staged and woof go runs (default "
             "$WOOF_SCRATCH/go-job/<job_id>, else ./go-job-<job_id>)")
    parser.add_argument(
        "--result", type=Path, default=None, metavar="PATH",
        help=f"where the {RESULT_SCHEMA} JSON is written (default "
             "WORKDIR/result.json); it is also printed on stdout")
    parser.add_argument(
        "--index", type=nonnegative_int, default=None, metavar="N",
        help="which entry of a job list to run (0-based; default "
             "$AWS_BATCH_JOB_ARRAY_INDEX, else $WOOF_WORKER_INDEX)")
    parser.add_argument(
        "--output-prefix", default=None, metavar="URL",
        help=f"store prefix for a job without icechunk_url: "
             f"URL/<job_id>.icechunk (default ${OUTPUT_ENV})")
    parser.set_defaults(func=main)
    return parser


__all__ = [
    "EXIT_EXPORT_INCOMPLETE", "EXIT_OK", "EXIT_REFUSED", "EXIT_TEMPFAIL",
    "GO_ARGS", "GoJob", "GoJobRefused", "JOB_SCHEMA", "LIST_SCHEMA",
    "RESULT_SCHEMA", "TransferFailed", "array_index", "check_url",
    "export_summary", "fetch", "go_command", "main", "parse_job", "publish",
    "push_inputs", "push_main", "register_cli", "resolve_ref", "run_job",
    "select_job",
]
