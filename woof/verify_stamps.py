"""Verify an input's bytes once, then reuse the digest while it is unchanged.

A re-run of ``woof go`` over a finished ICON cycle (33 GB in the DWD v1
tree, 10 767 files on a FUSE mount reading 8-18 MB/s) spent about 66
minutes in the fetch stage re-reading files whose SHA-256 it had already
proven, before preparation began; the forecast preflight and the
post-forecast check then read the prepared inputs again.  The bytes had not
moved.  This module is the one place that decides when a digest may be
reused instead of re-read, and says so in the receipts.

THE RULE (policy ``stamp``, the default off Windows).  After a file -- or
one prepared cache array -- has been read in full and its digest proven, a
*stamp* is
recorded: ``(device, inode, size, mtime_ns, ctime_ns) -> digest``, with the
algorithm and when it was read.  A later verification of the same path
whose ``os.stat`` tuple is identical to the stamp's reuses the digest
without reading the bytes.  Anything else -- a different size, modification
time, change time, inode or device, a stamp that is missing, unreadable,
malformed, for another algorithm or for another path -- is a full read, and
the new digest is stamped in turn.  A file whose modification or change
time was within :data:`SETTLED_NS` (2 s) of the read is never stamped: a
write inside its timestamps' resolution could otherwise go unseen.

What the stat tuple guards.  Every write through the filesystem moves
``ctime`` (the kernel sets it; ``touch``/``os.utime`` can rewind ``mtime``
but cannot rewind ``ctime``), a replacement moves the inode, and a
truncation moves the size.  What it cannot see is a STAT-PRESERVING change:
bytes altered underneath the filesystem (a raw block-device write, a
filesystem with coarse or untrustworthy change times -- some network and
FUSE filesystems, FAT/exFAT, and Windows, whose ``st_ctime`` is the
CREATION time), a system clock set back so a rewrite lands on the same
nanoseconds, or a stamp file forged by someone who can already write the
inputs and their receipts.  Policy ``full`` covers every one of those.

``WOOF_VERIFY=full`` (or ``--verify-inputs full`` on ``woof fetch``/``woof go``)
reuses no digest at all: every verification reads every byte, the
in-process memo is bypassed too, and stamps are neither read nor written.
That is the audit mode; it costs what the integrity checks cost before
stamps existed.  It is also the default on Windows, where ``st_ctime`` is
the creation time and so no stat field is guaranteed to move on a rewrite
(:func:`default_policy`); ``WOOF_VERIFY=stamp`` opts in there.

Where stamps live.  One small JSON file per (path, algorithm) in a
per-user cache -- ``$XDG_CACHE_HOME/woof/verify-stamps`` (``~/.cache``
when unset; ``%LOCALAPPDATA%`` on Windows) -- keyed by the file's
absolute path (symlinks unresolved), so read-only input trees (a shared mount, a colleague's
download) are never written to.  ``WOOF_VERIFY_STAMPS=<dir>`` moves the
store; ``WOOF_VERIFY_STAMPS=off`` disables it (the in-process memo still
applies under ``stamp``).  Each stamp is published by an atomic rename; a
torn or corrupt one reads as absent and is replaced after the next full
read.  A store that cannot be written costs only the reuse: the
verification itself never fails because a stamp could not be kept, and the
first such failure is said once on stderr.

What is pinned.  A reused digest is exactly the digest an earlier full
read produced for a file with the identical stat tuple, under the same
algorithm and absolute path.  Every receipt that consumed one says so:
``how`` is ``hashed`` (read by this verification), ``memo`` (read earlier
by this same process, stat-identical since) or ``stamp`` (read by an
earlier process, stat-identical since), and the policy and stamp store are
recorded beside the counts (:func:`receipt`).
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import os
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

#: The setting that chooses the policy: ``stamp`` (default) or ``full``.
POLICY_ENV = "WOOF_VERIFY"
#: Where stamps are kept, or ``off``.
STAMPS_ENV = "WOOF_VERIFY_STAMPS"
#: How many files are read at once when a full read is needed.
WORKERS_ENV = "WOOF_VERIFY_WORKERS"

STAMP = "stamp"
FULL = "full"
POLICIES = (STAMP, FULL)

#: How a digest was obtained, as every receipt spells it.
HASHED = "hashed"
MEMO = "memo"
STAMPED = "stamp"
HOWS = (HASHED, MEMO, STAMPED)

STAMP_SCHEMA = "woof-verify-stamp-v1"
RECEIPT_SCHEMA = "woof-verify-receipt-v1"

#: A file's own bytes, SHA-256.
FILE_SHA256 = "sha256"

#: The rule one sentence long, recorded in receipts beside the counts.
RULE = {
    STAMP: ("a digest is reused without reading only when the file's "
            "(device, inode, size, mtime_ns, ctime_ns) is identical to "
            "the stamp an earlier full read recorded; any difference, or a "
            "missing or corrupt stamp, is a full read. A stat-preserving "
            "change (raw device write, untrustworthy change times, a "
            "rewound clock) is not seen; WOOF_VERIFY=full reads every byte"),
    FULL: "every verification read every byte; no digest was reused",
}

#: A file changed this recently can still change again inside its
#: timestamps' resolution without either one moving, so it is never
#: stamped (and a stamp claiming otherwise is not believed).
SETTLED_NS = 2_000_000_000

#: Files smaller than this are always read: hashing them costs less than
#: consulting a stamp, and a receipt-sized file gains nothing from one.
STAMP_MIN_BYTES = 1 << 20

#: Read size for a full read.  Large, so a slow mount sees few requests
#: and ``hashlib`` (which releases the GIL above 2 KiB) runs off-lock.
READ_BYTES = 16 << 20

#: The wall clock the settling rule reads (a test seam).
_clock_ns: Callable[[], int] = time.time_ns


def default_policy() -> str:
    """``stamp``, except on Windows, where it is ``full``.

    Windows reports the CREATION time as ``st_ctime``, so nothing there
    moves when a file is rewritten and its modification time put back:
    the guard the stamp rule leans on does not exist.  ``WOOF_VERIFY=stamp``
    opts in anyway, knowing that.
    """

    return FULL if os.name == "nt" else STAMP


def policy() -> str:
    """The verification policy this process runs under.

    Refuses, naming the setting, a value that is neither ``stamp`` nor
    ``full`` -- a typo in an audit request must not quietly mean "reuse".
    """

    raw = os.environ.get(POLICY_ENV, "").strip().lower()
    if not raw:
        return default_policy()
    if raw not in POLICIES:
        raise ValueError(
            f"{POLICY_ENV}={os.environ.get(POLICY_ENV)!r} is not a "
            f"verification policy; use {POLICY_ENV}=stamp (reuse a digest "
            f"while the file's stat is identical; the default except on "
            f"Windows) or {POLICY_ENV}=full (read every byte on every verification)")
    return raw


def set_policy(value: str | None) -> None:
    """Set the policy for this process and every child it starts.

    For a command-line ``--verify-inputs`` flag: the setting is exported, so a
    stage the command runs as a subprocess verifies under the same one.
    ``None`` leaves the inherited setting alone.
    """

    if value is None:
        return
    value = str(value).strip().lower()
    if value not in POLICIES:
        raise ValueError(
            f"--verify-inputs {value!r} is not a verification policy; use "
            "'stamp' or 'full'")
    os.environ[POLICY_ENV] = value


VERIFY_FLAG_HELP = (
    "how inputs already on disk are verified: 'stamp' (the default, "
    "except on Windows) reuses a file's sha256 without reading it while its device, inode, "
    "size, mtime and ctime are identical to the full read that last "
    "proved it, and reads it again on any difference; 'full' reads every "
    "byte at every check (an audit; slow on a large cycle). Sets "
    f"{POLICY_ENV} for every stage this command starts; receipts record "
    "which ran")


def add_argument(parser) -> None:
    """``--verify-inputs {stamp,full}`` on a command that verifies inputs."""

    parser.add_argument("--verify-inputs", choices=POLICIES, default=None,
                        metavar="{stamp,full}", help=VERIFY_FLAG_HELP)


def apply(args) -> str:
    """Adopt a command's ``--verify-inputs`` and refuse a bad ``WOOF_VERIFY`` now.

    At the door, so a typo in an audit request is refused before anything
    is fetched or read rather than when the first receipt is written.
    Returns the policy in force.
    """

    set_policy(getattr(args, "verify_inputs", None))
    return policy()


def store_root() -> Path | None:
    """The stamp directory, or ``None`` when stamps are switched off."""

    raw = os.environ.get(STAMPS_ENV, "").strip()
    if raw.lower() in ("off", "0", "false", "none"):
        return None
    if raw:
        return Path(raw).expanduser()
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        base = Path(os.environ["LOCALAPPDATA"])
    elif os.environ.get("XDG_CACHE_HOME"):
        base = Path(os.environ["XDG_CACHE_HOME"])
    else:
        base = Path.home() / ".cache"
    return base / "woof" / "verify-stamps"


def default_workers(jobs: int | None = None) -> int:
    """Readers for a batch of full reads: ``WOOF_VERIFY_WORKERS`` or 8.

    Bounded by the CPU count (SHA-256 is CPU work once the bytes are in)
    and by the number of jobs.  A slow mount gains most from several reads
    in flight; a fast disk is CPU-bound well before eight.
    """

    raw = os.environ.get(WORKERS_ENV, "").strip()
    if raw:
        try:
            workers = int(raw)
        except ValueError:
            raise ValueError(
                f"{WORKERS_ENV}={raw!r} is not a worker count; pass a "
                "positive integer") from None
        if workers < 1:
            raise ValueError(
                f"{WORKERS_ENV}={raw!r} is not a worker count; pass a "
                "positive integer")
    else:
        workers = min(8, os.cpu_count() or 1)
    if jobs is not None:
        workers = min(workers, max(1, int(jobs)))
    return max(1, workers)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Identity:
    """Every stat field that moves when a file's bytes can."""

    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def of(cls, path) -> "Identity":
        status = os.stat(path)
        return cls(status.st_dev, status.st_ino, status.st_size,
                   status.st_mtime_ns, status.st_ctime_ns)

    def settled(self, at_ns: int) -> bool:
        return at_ns - max(self.mtime_ns, self.ctime_ns) >= SETTLED_NS

    def as_dict(self) -> dict:
        return {"device": self.device, "inode": self.inode,
                "size": self.size, "mtime_ns": self.mtime_ns,
                "ctime_ns": self.ctime_ns}


@dataclass(frozen=True)
class Verified:
    """One verification's answer: the digest and how it was obtained."""

    digest: str
    how: str
    size: int
    extra: Mapping[str, object] | None = None


# ---------------------------------------------------------------------------
# Tallies: what a receipt says about one verification step
# ---------------------------------------------------------------------------

class Tally:
    """Counts of files (or arrays) and bytes by how each was verified."""

    def __init__(self):
        self._lock = threading.Lock()
        self.counts = {how: 0 for how in HOWS}
        self.bytes = {how: 0 for how in HOWS}

    def add(self, how: str, size: int) -> None:
        with self._lock:
            self.counts[how] += 1
            self.bytes[how] += int(size)

    def merge(self, other: "Tally") -> None:
        for how in HOWS:
            with self._lock:
                self.counts[how] += other.counts[how]
                self.bytes[how] += other.bytes[how]

    def snapshot(self) -> "Tally":
        """A copy, to measure what a later step added (:meth:`since`)."""
        copy = Tally()
        copy.merge(self)
        return copy

    def since(self, earlier: "Tally") -> "Tally":
        """What was counted after ``earlier`` (a :meth:`snapshot`)."""
        delta = Tally()
        with self._lock:
            for how in HOWS:
                delta.counts[how] = self.counts[how] - earlier.counts[how]
                delta.bytes[how] = self.bytes[how] - earlier.bytes[how]
        return delta

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def as_dict(self) -> dict:
        return {
            "hashed": self.counts[HASHED],
            "hashed_bytes": self.bytes[HASHED],
            "memo_reused": self.counts[MEMO],
            "memo_reused_bytes": self.bytes[MEMO],
            "stamp_reused": self.counts[STAMPED],
            "stamp_reused_bytes": self.bytes[STAMPED],
        }


_COLLECTORS: contextvars.ContextVar[tuple[Tally, ...]] = (
    contextvars.ContextVar("woof_verify_collectors", default=()))


@contextmanager
def collecting():
    """Collect every verification this thread makes into one tally.

    Nested collectors each see every verification inside them.  Workers a
    verification starts report through the thread that started it, so a
    parallel batch is counted once and in full.
    """

    tally = Tally()
    token = _COLLECTORS.set(_COLLECTORS.get() + (tally,))
    try:
        yield tally
    finally:
        _COLLECTORS.reset(token)


def _sinks(tally: Tally | None) -> tuple[Tally, ...]:
    active = _COLLECTORS.get()
    return active + ((tally,) if tally is not None and tally not in active
                     else ())


def _count(sinks: Sequence[Tally], verified: Verified) -> None:
    for sink in sinks:
        sink.add(verified.how, verified.size)


def report(tally: Tally) -> None:
    """Hand a batch's counts to every collector open on this thread.

    For a verifier that counted into its own :class:`Tally` on worker
    threads (which see no collector): called once, on the thread that
    started the batch, so the batch is counted once and in full.
    """
    for sink in _COLLECTORS.get():
        sink.merge(tally)


def map_bounded(function, items: Sequence, *, workers: int | None = None
                ) -> list:
    """``[function(item) for item in items]`` on a bounded thread pool.

    ``workers`` threads, :func:`default_workers` when None, never more
    than there are items; one item or one worker runs inline.  Order is
    kept and the first exception, in item order, propagates.
    """
    items = list(items)
    count = (default_workers(len(items)) if workers is None
             else max(1, min(int(workers), len(items) or 1)))
    if count <= 1 or len(items) <= 1:
        return [function(item) for item in items]
    with ThreadPoolExecutor(max_workers=count,
                            thread_name_prefix="woof-verify") as pool:
        return list(pool.map(function, items))


def receipt(tally: Tally | None = None, *, active: str | None = None) -> dict:
    """The policy statement a receipt records, with ``tally``'s counts."""

    active = policy() if active is None else active
    root = store_root() if active == STAMP else None
    payload = {
        "schema": RECEIPT_SCHEMA,
        "policy": active,
        "rule": RULE[active],
        "stamp_store": None if root is None else str(root),
        "settled_seconds": SETTLED_NS / 1e9,
    }
    if tally is not None:
        payload.update(tally.as_dict())
    return payload


# ---------------------------------------------------------------------------
# The in-process memo and the stamp store
# ---------------------------------------------------------------------------

#: ``(path, algorithm, identity) -> (digest, extra, origin)``.  ``origin``
#: is what a later hit reports: ``memo`` when this process read the bytes,
#: ``stamp`` when the digest came from a stamp an earlier process wrote --
#: a stamp consulted twice in one process is still a stamp, not a read.
_MEMO: OrderedDict[tuple, tuple[str, Mapping | None, str]] = OrderedDict()
_MEMO_LOCK = threading.Lock()
_MEMO_ENTRIES = 65536
_WARNED = threading.Event()


def clear_memo() -> None:
    """Forget every in-process digest (a test seam)."""

    with _MEMO_LOCK:
        _MEMO.clear()


def _resolved(path) -> str:
    """The stamp key: the absolute path, without resolving symlinks.

    ``realpath`` costs one ``lstat`` per path component, which on a
    network mount is a round trip each, for ten thousand files.  A file
    reached through two spellings simply has two stamps; what a stamp
    vouches for is pinned by the device and inode, not by the name.
    """
    return os.path.abspath(os.fspath(path))


def _memo_key(resolved: str, algorithm: str, identity: Identity) -> tuple:
    return (resolved, algorithm, identity)


def _stamp_path(root: Path, resolved: str, algorithm: str) -> Path:
    token = hashlib.sha256(
        f"{algorithm}\0{resolved}".encode("utf-8", "surrogatepass")
    ).hexdigest()
    return root / token[:2] / f"{token[2:]}.json"


def _is_digest(value) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def lookup(path, algorithm: str = FILE_SHA256, *,
           identity: Identity | None = None) -> Verified | None:
    """The digest a stamp holds for ``path`` if its stat is identical.

    ``None`` under policy ``full``, with stamps off, or for any stamp that
    is missing, unreadable, malformed, for another path or algorithm,
    unsettled, or whose stat tuple differs in any field.  Checks the
    in-process memo first (answering ``memo`` for a digest this process
    read, ``stamp`` for one a stamp supplied), then the store (``stamp``).
    """

    if policy() != STAMP:
        return None
    resolved = _resolved(path)
    identity = Identity.of(path) if identity is None else identity
    with _MEMO_LOCK:
        known = _MEMO.get(_memo_key(resolved, algorithm, identity))
        if known is not None:
            _MEMO.move_to_end(_memo_key(resolved, algorithm, identity))
            return Verified(known[0], known[2], identity.size, known[1])
    root = store_root()
    if root is None:
        return None
    try:
        document = json.loads(
            _stamp_path(root, resolved, algorithm).read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    try:
        if (document.get("schema") != STAMP_SCHEMA
                or document.get("path") != resolved
                or document.get("algorithm") != algorithm
                or not _is_digest(document.get("digest"))):
            return None
        recorded = Identity(**{key: document[key] for key in (
            "device", "inode", "size", "mtime_ns", "ctime_ns")})
        verified_at = document["verified_at_ns"]
        if (any(not isinstance(value, int) or isinstance(value, bool)
                for value in (*recorded.as_dict().values(), verified_at))
                or recorded != identity
                or not recorded.settled(verified_at)):
            return None
        extra = document.get("extra")
        if extra is not None and not isinstance(extra, dict):
            return None
    except (AttributeError, KeyError, TypeError):
        return None
    _remember(resolved, algorithm, identity, document["digest"], extra,
              STAMPED)
    return Verified(document["digest"], STAMPED, identity.size, extra)


def _remember(resolved, algorithm, identity, digest, extra, origin) -> None:
    with _MEMO_LOCK:
        _MEMO[_memo_key(resolved, algorithm, identity)] = (
            digest, extra, origin)
        while len(_MEMO) > _MEMO_ENTRIES:
            _MEMO.popitem(last=False)


def record(path, digest: str, *, before: Identity,
           algorithm: str = FILE_SHA256,
           extra: Mapping[str, object] | None = None) -> bool:
    """Stamp a digest a full read of ``path`` just proved.

    ``before`` is the stat taken before the read began.  Nothing is kept
    -- and ``False`` returned -- under policy ``full``, when the file's stat
    moved during the read, when it was changed within :data:`SETTLED_NS`
    of now, or when the digest is not a SHA-256 hex digest.  A store that
    cannot be written keeps the memo only and says so once.
    """

    if policy() != STAMP or not _is_digest(digest):
        return False
    try:
        after = Identity.of(path)
    except OSError:
        return False
    now = _clock_ns()
    if after != before or not before.settled(now):
        return False
    resolved = _resolved(path)
    extra = None if extra is None else dict(extra)
    _remember(resolved, algorithm, before, digest, extra, MEMO)
    root = store_root()
    if root is None:
        return True
    document = {
        "schema": STAMP_SCHEMA,
        "path": resolved,
        "algorithm": algorithm,
        **before.as_dict(),
        "digest": digest,
        "extra": extra,
        "verified_at": datetime.fromtimestamp(
            now / 1e9, timezone.utc).isoformat(),
        "verified_at_ns": now,
    }
    target = _stamp_path(root, resolved, algorithm)
    staging = target.with_name(
        f".{target.name}.{os.getpid()}.{threading.get_ident()}.partial")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        staging.write_text(json.dumps(document, sort_keys=True) + "\n",
                           encoding="utf-8")
        os.replace(staging, target)
    except OSError as error:
        try:
            staging.unlink(missing_ok=True)
        except OSError:
            pass
        if not _WARNED.is_set():
            _WARNED.set()
            print(f"verify: could not keep a digest stamp in {root} "
                  f"({error.strerror or error}); this process re-reads "
                  f"unchanged inputs next time. Set {STAMPS_ENV} to a "
                  "writable directory, or to 'off' to silence this",
                  file=sys.stderr)
    return True


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def sha256_stream(path) -> str:
    """SHA-256 of a file's bytes, read in :data:`READ_BYTES` blocks."""

    digest = hashlib.sha256()
    with open(path, "rb", buffering=0) as stream:
        while True:
            block = stream.read(READ_BYTES)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def file_sha256(path, *, tally: Tally | None = None,
                hasher: Callable[[object], str] | None = None,
                min_bytes: int = STAMP_MIN_BYTES) -> Verified:
    """SHA-256 of ``path``, reused from a stamp when stat-identical.

    ``hasher`` replaces the full read (a caller's own reader, kept so its
    seams stay where tests expect them).  Files under ``min_bytes`` are
    always read.  ``OSError`` from the stat or the read propagates: a file
    that cannot be read has no digest to report.
    """

    sinks = _sinks(tally)
    identity = Identity.of(path)
    if identity.size >= min_bytes:
        reused = lookup(path, identity=identity)
        if reused is not None:
            _count(sinks, reused)
            return reused
    digest = (hasher or sha256_stream)(path)
    if identity.size >= min_bytes:
        record(path, digest, before=identity)
    verified = Verified(digest, HASHED, identity.size)
    _count(sinks, verified)
    return verified


def files_sha256(paths: Iterable, *, tally: Tally | None = None,
                 workers: int | None = None,
                 min_bytes: int = STAMP_MIN_BYTES) -> list[Verified]:
    """:func:`file_sha256` for many files, full reads in parallel.

    Stamp lookups are answered first, in order and without a pool; only
    the files left to read go to ``workers`` threads.  Results are in the
    order given.  The first ``OSError`` propagates.
    """

    paths = list(paths)
    sinks = _sinks(tally)
    results: list[Verified | None] = [None] * len(paths)
    pending: list[tuple[int, Identity]] = []
    for index, path in enumerate(paths):
        identity = Identity.of(path)
        reused = (lookup(path, identity=identity)
                  if identity.size >= min_bytes else None)
        if reused is not None:
            results[index] = reused
        else:
            pending.append((index, identity))

    def read(item: tuple[int, Identity]) -> Verified:
        index, identity = item
        digest = sha256_stream(paths[index])
        if identity.size >= min_bytes:
            record(paths[index], digest, before=identity)
        return Verified(digest, HASHED, identity.size)

    for (index, _identity), verified in zip(
            pending, map_bounded(read, pending, workers=workers)):
        results[index] = verified
    for verified in results:
        _count(sinks, verified)
    return results  # type: ignore[return-value]


__all__ = [
    "FILE_SHA256", "FULL", "HASHED", "HOWS", "Identity", "MEMO", "POLICIES",
    "POLICY_ENV", "READ_BYTES", "RECEIPT_SCHEMA", "RULE", "SETTLED_NS",
    "STAMP", "STAMPED", "STAMPS_ENV", "STAMP_MIN_BYTES", "STAMP_SCHEMA",
    "Tally", "Verified", "WORKERS_ENV", "add_argument", "apply",
    "clear_memo", "collecting", "default_policy", "default_workers",
    "file_sha256", "files_sha256", "lookup", "map_bounded", "policy",
    "receipt", "record", "report", "set_policy", "sha256_stream",
    "store_root",
]
