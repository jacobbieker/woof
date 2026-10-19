"""Export a forecast's history frames to Icechunk while the forecast runs.

``woof go`` with ``[output] icechunk = { url = "..." }`` (or ``--icechunk
URL``) starts this module as a separate process beside the forecast.  The
forecast publishes each ``wrfout_dNN_*`` by atomic rename and then a
``ready/<frame>.json`` marker (``docs/public/PROGRESS.md``); the exporter
polls the markers -- and walks the forecast folder for frames whose marker
is missing -- and writes each frame it has not exported yet into one
group per domain (``d01``, ``d02``, ...) of one repository, one commit per
frame.  When the forecast ends, ``woof go`` writes ``finish`` into the
export folder; the exporter drains what is left and exits, and ``woof
go`` waits for it before declaring the run done.

What is pinned:

* **One writer implementation.**  Each frame goes through
  :func:`woof.energy.icechunk_store.write_gridded_segment` -- the writer
  ``tools/wrfout_to_netcdf.py --icechunk-group`` and the hindcast use --
  with ``extend_axis`` (the group's axis starts at its first frame and
  grows at its end), ``add_variables`` (a field first written after t=0,
  such as ``REFL_10CM``, is created on the frame that first carries it
  and is NaN before it) and ``on_existing="newer"``.  So a store
  exported live equals one exported after the fact from the same frames.
* **The model never waits.**  The exporter is its own process, started at
  ``nice`` 10, that reads only published frames; nothing in the forecast
  waits on it, and a failed or killed exporter leaves the forecast
  running.  Its lag (frames pending, model time behind, the oldest
  pending frame's wait) is in ``status.json``.
* **Resumable and idempotent.**  ``frame_written`` and
  ``frame_published_unix_ms`` (the source file's mtime) in the store
  decide what is done: a frame already written from the same file is
  skipped, a frame written again later -- a resumed run's overlap, a
  history rewind -- replaces it (the later run wins).  ``state.json``
  only saves re-reading files on a restart.  ``woof icechunk-export
  <run folder or export folder>`` resumes or completes an export.
* **Named refusals.**  A frame the store refuses (a different grid, a
  time before the group's start or off its interval) is recorded with
  the reason and the export ends ``incomplete``; it is never dropped
  silently.
* **A long run fits its disk.**  ``domains = [3]`` exports only
  those grids (the others' frames are counted as skipped, so the run
  still finishes ``complete``), and ``delete_after_commit = true``
  unlinks each exported file once the store holds it (see
  :meth:`Exporter.delete_exported` for what makes that safe).  ``url``
  may name the run with ``{name}`` (``[experiment] name``, or
  ``woof icechunk-export --name``), one repository per experiment.
* **Moving nests.**  A group's ``wrf_*`` attributes are its first
  frame's; the nest's placement at every frame is in ``(Time,)`` arrays
  (``wrf_I_PARENT_START``, ``wrf_CEN_LAT``, ...;
  :data:`woof.energy.icechunk_store.PLACEMENT_ATTRS`).

Files in the export folder (``<run>/icechunk-export``):
``export.json`` (:data:`SPEC_SCHEMA`, what to export where),
``state.json`` (:data:`STATE_SCHEMA`), ``status.json``
(:data:`STATUS_SCHEMA`, live progress and the final receipt),
``export.log`` and ``finish``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import urlsplit

SPEC_SCHEMA = "woof.icechunk-export.spec.v1"
STATE_SCHEMA = "woof.icechunk-export.state.v1"
STATUS_SCHEMA = "woof.icechunk-export.status.v1"

EXPORT_DIRNAME = "icechunk-export"
SPEC_NAME = "export.json"
STATE_NAME = "state.json"
STATUS_NAME = "status.json"
FINISH_NAME = "finish"
LOG_NAME = "export.log"

#: Seconds between looks at the forecast folder.
DEFAULT_POLL_SECONDS = 2.0
#: Arrays encoded and stored at once per frame.  Pinned at 4: measured
#: 2026-10-15 on a 540 MB frame (30 x 50 x 300 x 300 float32, file://
#: repository, 32-core host running two forecasts), 1 thread 11-15 s,
#: 4 threads 3-6.5 s, 8 threads 3-4.3 s; four leave the forecast's host
#: threads their cores.
DEFAULT_THREADS = 4
#: The ``nice`` increment the exporter process takes (POSIX).
EXPORTER_NICE = 10
#: The interval a group's axis is created with before its second frame
#: shows the real one (a one-step axis adopts the first gap it sees).
PLACEHOLDER_INTERVAL_S = 3600
#: Attempts at one frame when the store fails (not refuses) before the
#: export gives up, with 1, 2, 4, 8 s between them.
TRANSIENT_ATTEMPTS = 5
#: ``woof go``'s exit code when the forecast finished but its Icechunk
#: export did not (sysexits EX_CANTCREAT: an output could not be made).
EXPORT_INCOMPLETE_EXIT_CODE = 73

#: Keys ``[output] icechunk = { ... }`` accepts.
OPTION_KEYS = frozenset({"url", "branch", "compress", "zstd_clevel",
                         "group_prefix", "threads", "delete_after_commit",
                         "domains"})

#: Placeholders ``url`` may carry: ``{name}``, filled by ``woof go`` with
#: the config's ``[experiment] name`` (or by ``woof icechunk-export
#: --name``), so a series of configs can share one url template.
URL_PLACEHOLDERS = ("name",)
_PLACEHOLDER = re.compile(r"\{([^{}]*)\}")
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

_FRAME_NAME = re.compile(
    r"^wrfout_d(\d+)_(\d{4})-(\d{2})-(\d{2})_(\d{2})[_:](\d{2})[_:](\d{2})")


class IcechunkExportRefused(ValueError):
    """An export request this module will not carry out (bad options, a
    folder with nothing to export, options that disagree)."""


# --------------------------------------------------------------------------
# options


@dataclass(frozen=True)
class ExportOptions:
    """Where and how to export.  ``url`` is a store URL (``s3://``,
    ``gs://``, ``file://``) or a local directory; Blosc zstd with
    bitshuffle at ``zstd_clevel`` unless ``compress`` is false; domain N
    goes to group ``f"{group_prefix}d{N:02d}"``.  ``domains`` (grid ids)
    limits the export to those grids, ``None`` exports every one;
    ``delete_after_commit`` unlinks each exported wrfout once the store
    holds it."""

    url: str
    branch: str = "main"
    compress: bool = True
    zstd_clevel: int = 5
    group_prefix: str = ""
    threads: int = DEFAULT_THREADS
    delete_after_commit: bool = False
    domains: tuple[int, ...] | None = None

    @classmethod
    def from_mapping(cls, value, *, source: str = "<config>",
                     base_dir: Path | None = None,
                     allow_memory: bool = False,
                     names: Mapping[str, Any] | None = None,
                     allow_template: bool = False
                     ) -> "ExportOptions | None":
        """Validate ``[output] icechunk``: a URL string or a table of
        :data:`OPTION_KEYS`.  ``None`` (absent) gives ``None``.

        ``{name}`` in ``url`` is filled from ``names`` (``woof go``
        passes ``[experiment] name``); with
        ``allow_template`` (the config's own validation, which has no
        name yet) they are checked and kept, and otherwise a url still
        carrying one is refused."""

        if value is None:
            return None
        where = f"[output] icechunk of {source}"
        if isinstance(value, str):
            value = {"url": value}
        if not isinstance(value, Mapping):
            raise IcechunkExportRefused(
                f"{where} must be a store URL or a table, e.g. icechunk = "
                '{ url = "s3://bucket/prefix" }, got ' f"{value!r}")
        unknown = sorted(set(value) - OPTION_KEYS)
        if unknown:
            raise IcechunkExportRefused(
                f"unknown key(s) {unknown} in {where}; known keys: "
                f"{sorted(OPTION_KEYS)}")
        url = value.get("url")
        if not isinstance(url, str) or not url.strip():
            raise IcechunkExportRefused(
                f"{where} needs url = \"s3://...\", \"gs://...\", "
                "\"file:///...\" or a directory")
        url = _filled_url(url.strip(), where=where, names=names,
                          allow_template=allow_template)
        url = _checked_url(url, where=where, base_dir=base_dir,
                           allow_memory=allow_memory)
        branch = value.get("branch", "main")
        if not isinstance(branch, str) or not branch:
            raise IcechunkExportRefused(f"{where} branch must be a name")
        compress = value.get("compress", True)
        if not isinstance(compress, bool):
            raise IcechunkExportRefused(
                f"{where} compress must be true or false")
        clevel = value.get("zstd_clevel", 5)
        if isinstance(clevel, bool) or not isinstance(clevel, int) \
                or not 1 <= clevel <= 9:
            raise IcechunkExportRefused(
                f"{where} zstd_clevel must be an integer 1-9")
        prefix = value.get("group_prefix", "")
        if not isinstance(prefix, str) or prefix.startswith("/") \
                or (prefix and not prefix.endswith("/")
                    and not re.fullmatch(r"[A-Za-z0-9_.-]+", prefix)):
            raise IcechunkExportRefused(
                f"{where} group_prefix must be a relative group path "
                "ending in '/' (e.g. \"uk/\") or a plain name prefix")
        threads = value.get("threads", DEFAULT_THREADS)
        if isinstance(threads, bool) or not isinstance(threads, int) \
                or not 1 <= threads <= 64:
            raise IcechunkExportRefused(
                f"{where} threads must be an integer 1-64")
        delete = value.get("delete_after_commit", False)
        if not isinstance(delete, bool):
            raise IcechunkExportRefused(
                f"{where} delete_after_commit must be true or false")
        domains = value.get("domains")
        if domains is not None:
            if not isinstance(domains, (list, tuple)) or not domains or any(
                    isinstance(d, bool) or not isinstance(d, int) or d < 1
                    for d in domains) or len(set(domains)) != len(domains):
                raise IcechunkExportRefused(
                    f"{where} domains must be a non-empty list of distinct "
                    f"grid ids, e.g. domains = [3], got {domains!r}")
            domains = tuple(sorted(int(d) for d in domains))
        return cls(url=url, branch=branch, compress=compress,
                   zstd_clevel=clevel, group_prefix=prefix, threads=threads,
                   delete_after_commit=delete, domains=domains)

    def group(self, domain: int) -> str:
        return f"{self.group_prefix}d{int(domain):02d}"

    def exports(self, domain: int) -> bool:
        """Whether domain ``domain``'s frames go to the store."""

        return self.domains is None or int(domain) in self.domains

    def to_mapping(self) -> dict[str, Any]:
        mapping = asdict(self)
        if self.domains is not None:
            mapping["domains"] = list(self.domains)
        return mapping


def _filled_url(url: str, *, where: str, names: Mapping[str, Any] | None,
                allow_template: bool) -> str:
    """``url`` with its :data:`URL_PLACEHOLDERS` filled from ``names``."""

    from woof.explain import layered

    found = _PLACEHOLDER.findall(url)
    if not found:
        return url
    unknown = sorted(set(found) - set(URL_PLACEHOLDERS))
    if unknown:
        raise IcechunkExportRefused(layered(
            f"{where}: url placeholder(s) "
            f"{', '.join('{' + u + '}' for u in unknown)} are not known; "
            f"use {', '.join('{' + p + '}' for p in URL_PLACEHOLDERS)}.",
            "Only the experiment's name is filled into a store URL: each "
            "run of a series is its own experiment, so one url template "
            "gives every run its own repository."))
    if names is None:
        if allow_template:
            return url
        raise IcechunkExportRefused(layered(
            f"{where}: url {url!r} still carries a placeholder; give the "
            "store's own URL here, or the name with --name.",
            "{name} is filled from the config's [experiment] name by woof "
            "go; nothing here knows that name."))
    filled = url
    for key in sorted(set(found)):
        value = names.get(key)
        if not isinstance(value, str) or not value:
            raise IcechunkExportRefused(layered(
                f"{where}: url uses {{{key}}} but there is no "
                "[experiment] name to fill it with. Next: set [experiment] "
                "name, or write the store's URL out.",
                "The placeholder names one repository per experiment; "
                "without a name every run would share one url."))
        if not _SAFE_NAME.fullmatch(value) or value in (".", ".."):
            raise IcechunkExportRefused(layered(
                f"{where}: url uses {{{key}}} and [experiment] name "
                f"{value!r} is not a safe store path segment (letters, "
                "digits, '.', '_' and '-', not starting with a "
                "punctuation mark). Next: rename the experiment, or write "
                "the store's URL out.",
                "The name becomes part of an object-store key or a "
                "directory; a '/' would nest repositories and a space or "
                "'..' would name a different place than it reads."))
        filled = filled.replace("{" + key + "}", value)
    return filled


def _checked_url(url: str, *, where: str, base_dir: Path | None,
                 allow_memory: bool) -> str:
    from woof.energy.icechunk_store import SCHEMES, is_store_url

    if "://" in url and not is_store_url(url):
        raise IcechunkExportRefused(
            f"{where}: scheme {url.split('://', 1)[0]!r} is not one of "
            f"{', '.join(s + '://' for s in SCHEMES)}")
    if is_store_url(url):
        parts = urlsplit(url)
        if parts.username or parts.password:
            raise IcechunkExportRefused(
                f"{where}: credentials in the URL are refused; use the "
                "environment, a profile or a role")
        if parts.scheme.lower() == "memory" and not allow_memory:
            raise IcechunkExportRefused(
                f"{where}: memory:// is an in-process store and the export "
                "runs in its own process; use file:// or a directory")
        return url
    path = Path(url).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    return str(path.resolve())


def history_intervals(config: Path | str | None) -> dict[int, int]:
    """Each domain's whole-second history interval from a config TOML
    (``[[domain]] history_interval_s``, or a legacy ``[run]
    output_interval_s`` for domain 1); best effort, ``{}`` when unread.
    Only the interval a group's axis starts with before its second frame
    is read from it."""

    if config is None:
        return {}
    try:
        import tomllib

        raw = tomllib.loads(Path(config).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    found: dict[int, int] = {}
    domains = raw.get("domain")
    if isinstance(domains, list):
        for entry in domains:
            if not isinstance(entry, Mapping):
                continue
            gid, interval = entry.get("grid_id"), entry.get(
                "history_interval_s")
            if isinstance(gid, int) and _whole(interval):
                found[gid] = int(interval)
    run = raw.get("run")
    if not found and isinstance(run, Mapping) \
            and _whole(run.get("output_interval_s")):
        found[1] = int(run["output_interval_s"])
    return found


def _aside(path: Path) -> Path:
    """Where :meth:`Exporter.delete_exported` moves a frame before it
    unlinks it: a hidden name in the same folder, which neither the frame
    walk nor a ready marker names."""

    return Path(path).with_name(f".{Path(path).name}.icechunk-exported")


def _whole(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0 and value == int(value))


# --------------------------------------------------------------------------
# frames


@dataclass(frozen=True)
class SourceFrame:
    """One published history file: its domain and the valid time its
    name carries (used for ordering and lag; the store reads ``Times``)."""

    path: Path
    domain: int
    valid: datetime
    size: int
    mtime_ns: int
    inode: tuple[int, int] | None = None

    @property
    def key(self) -> str:
        return str(self.path)

    def signature(self) -> dict[str, int]:
        return {"size": self.size, "mtime_ns": self.mtime_ns}


def parse_frame_name(name: str) -> tuple[int, datetime] | None:
    match = _FRAME_NAME.match(name)
    if match is None:
        return None
    values = [int(v) for v in match.groups()]
    try:
        return values[0], datetime(*values[1:], tzinfo=timezone.utc)
    except ValueError:
        return None


def discover_frames(sources: Iterable[Path], *,
                    markers: dict[str, tuple[int, str | None]] | None = None
                    ) -> list[SourceFrame]:
    """Every published frame under ``sources``: those the history walk
    (:func:`woof.io.wrfout.iter_wrfout_files`) finds and those named by
    ``ready/*.json`` markers.

    One file reached by two spellings (a marker's resolved path and the
    walk's path through a symlink) is one frame, under the walk's
    spelling.  A frame that disappears between the listing and its
    ``stat`` (a history rewind deleting it) is left out.  ``markers`` is
    a cache of parsed markers by path and mtime, kept by a caller that
    looks again and again."""

    from woof.io.wrfout import iter_wrfout_files
    from woof.progress_log import FRAME_MARKER_DIRNAME, FRAME_MARKER_SCHEMA

    cache = markers if markers is not None else {}
    candidates: list[Path] = []
    for root in sources:
        root = Path(root)
        candidates.extend(Path(os.path.abspath(path)) for path in
                          iter_wrfout_files(root, "wrfout_d*",
                                            include_temporaries=False))
        try:
            entries = sorted((root / FRAME_MARKER_DIRNAME).glob(
                "wrfout_d*.json"))
        except OSError:
            entries = []
        for marker in entries:
            try:
                stamp = marker.stat().st_mtime_ns
            except OSError:
                continue
            cached = cache.get(str(marker))
            if cached is None or cached[0] != stamp:
                named = None
                data = _read_json(marker)
                if data is not None \
                        and data.get("schema") == FRAME_MARKER_SCHEMA:
                    path = Path(str(data.get("path", "")))
                    named = str(path) if path.is_absolute() else None
                cached = cache[str(marker)] = (stamp, named)
            if cached[1] is not None:
                candidates.append(Path(cached[1]))
    frames: dict[tuple[int, int], SourceFrame] = {}
    for path in candidates:
        parsed = parse_frame_name(path.name)
        if parsed is None:
            continue
        try:
            info = path.stat()
        except OSError:
            continue
        frames.setdefault((info.st_dev, info.st_ino), SourceFrame(
            path, parsed[0], parsed[1], int(info.st_size),
            int(info.st_mtime_ns), (int(info.st_dev), int(info.st_ino))))
    return sorted(frames.values(),
                  key=lambda f: (f.valid, f.domain, f.mtime_ns, f.key))


# --------------------------------------------------------------------------
# the exporter


def _now_ms() -> int:
    return int(time.time() * 1000)


def _utc(ms: int | None = None) -> str:
    stamp = time.time() if ms is None else ms / 1000
    return datetime.fromtimestamp(stamp, timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    from woof.supervisor import atomic_write_json

    atomic_write_json(path, dict(payload))


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _alive(pid: int | None, created: str | None = None) -> bool:
    """Whether the process ``pid`` (created at ``created``, when given) is
    still that process (:func:`woof.ownership.identity_alive`: a reused
    pid is not the parent)."""

    if not pid:
        return True
    from woof.ownership import identity_alive

    return identity_alive({"pid": int(pid), "created": created})


def _created(pid: int) -> str | None:
    from woof.ownership import process_created

    value = process_created(pid)
    return None if value is None else str(value)


class ExportFailed(RuntimeError):
    """The store failed (not refused) a frame :data:`TRANSIENT_ATTEMPTS`
    times; the export stops and can be resumed."""


class Exporter:
    """Export the frames under ``sources`` into ``options.url``.

    :meth:`run` drains once (``follow=False``: the resume command and the
    after-the-fact export) or follows a running forecast until
    ``finish_file`` appears or ``parent_pid`` exits, then drains."""

    def __init__(self, sources: Sequence[Path], options: ExportOptions, *,
                 state_dir: Path, intervals: Mapping[int, int] | None = None,
                 poll_seconds: float = DEFAULT_POLL_SECONDS,
                 finish_file: Path | None = None,
                 parent_pid: int | None = None,
                 parent_created: str | None = None,
                 say: Callable[[str], None] | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 write: Callable[..., dict] | None = None):
        self.sources = [Path(s) for s in sources]
        self.options = options
        self.state_dir = Path(state_dir)
        self.intervals = {int(k): int(v) for k, v in (intervals or {}).items()}
        self.poll_seconds = max(0.05, float(poll_seconds))
        self.finish_file = None if finish_file is None else Path(finish_file)
        self.parent_pid = parent_pid
        self.parent_created = parent_created
        self.say = say or (lambda line: print(line, flush=True))
        self.sleep = sleep
        if write is None:
            from woof.energy.icechunk_store import write_gridded_segment
            write = write_gridded_segment
        self._write = write
        self.started_ms = _now_ms()
        self.frames: dict[str, dict[str, Any]] = {}
        self.refused: dict[str, dict[str, Any]] = {}
        self.found: list[SourceFrame] = []
        #: Frames of domains ``options.domains`` leaves out: seen, never
        #: written, never deleted, and no reason for the run not to finish.
        self.skipped: list[SourceFrame] = []
        self.last_snapshot: str | None = None
        self.last_frame_seconds: float | None = None
        self.state = "starting"
        self.error: str | None = None
        self.notes: list[str] = []
        #: Why ``state.json`` could not be checked against the store.
        self.unconfirmed: str | None = None
        self._markers: dict[str, tuple[int, str | None]] = {}
        self._load_state()
        self._reconcile_with_store()

    # -- state -------------------------------------------------------------

    @property
    def state_path(self) -> Path:
        return self.state_dir / STATE_NAME

    @property
    def status_path(self) -> Path:
        return self.state_dir / STATUS_NAME

    def _load_state(self) -> None:
        saved = _read_json(self.state_path)
        if saved is None or saved.get("schema") != STATE_SCHEMA:
            return
        if saved.get("url") != self.options.url \
                or saved.get("branch") != self.options.branch \
                or saved.get("group_prefix", "") != self.options.group_prefix:
            # Another store: what it recorded says nothing about this one.
            return
        frames = saved.get("frames")
        if isinstance(frames, dict):
            self.frames = {str(k): dict(v) for k, v in frames.items()
                           if isinstance(v, dict)}
        self.last_snapshot = saved.get("last_snapshot")

    def _reconcile_with_store(self) -> None:
        """Keep only the saved frames the store itself confirms.

        Pinned: ``state.json`` is a cache, the store is the record.  A
        saved frame counts as exported only when its group marks that
        valid time written with this file's publication stamp or a newer
        one (a later run's frame superseded it: marked ``superseded``,
        which :meth:`delete_exported` never deletes); a repository
        deleted, recreated or reset since leaves nothing confirmed.  A
        store that cannot be reached is retried like a frame, then the
        records are kept for :meth:`run` to end ``failed`` (resumably):
        frames deleted after their commit exist only in this record and
        the store, and a passing outage must not forget them."""

        if not self.frames:
            return
        import numpy as np

        from woof.energy.icechunk_store import (FRAME_PUBLISHED,
                                                IcechunkStoreRefused,
                                                TimeAxis, _members,
                                                _open_group, open_repository)

        delay = 1.0
        for attempt in range(1, TRANSIENT_ATTEMPTS + 1):
            try:
                repo = open_repository(self.options.url, create=False)
                break
            except IcechunkStoreRefused:   # no repository: nothing done
                self.frames, self.last_snapshot = {}, None
                return
            except Exception as error:  # noqa: BLE001 - a store failure
                if attempt == TRANSIENT_ATTEMPTS:
                    self.unconfirmed = (
                        f"the store could not be opened to confirm "
                        f"state.json ({type(error).__name__}: {error})")
                    return
                self.sleep(delay)
                delay *= 2
        try:
            session = repo.readonly_session(branch=self.options.branch)
        except Exception:  # noqa: BLE001 - no branch: nothing done
            self.frames, self.last_snapshot = {}, None
            return
        by_group: dict[str, list[tuple[str, dict]]] = {}
        for key, record in self.frames.items():
            by_group.setdefault(str(record.get("group")), []).append(
                (key, record))
        confirmed: dict[str, dict[str, Any]] = {}
        for group, entries in by_group.items():
            zgroup = _open_group(session, group, mode="r")
            if zgroup is None or FRAME_PUBLISHED not in _members(zgroup):
                continue
            try:
                axis = TimeAxis.from_attrs(dict(zgroup.attrs))
            except IcechunkStoreRefused:
                continue
            written = np.asarray(zgroup["frame_written"][:])
            stamps = np.asarray(zgroup[FRAME_PUBLISHED][:])
            for key, record in entries:
                try:
                    valid = np.datetime64(
                        str(record.get("valid_time", "")).rstrip("Z"), "s")
                    index = int(axis.index_of([valid])[0])
                except (IcechunkStoreRefused, ValueError):
                    continue
                stamp = int(record.get("mtime_ns", 0)) // 1_000_000
                if written[index] == 1 and stamps[index] >= stamp:
                    if stamps[index] > stamp:
                        record["outcome"] = "superseded"
                    confirmed[key] = record
        self.frames = confirmed

    def _save_state(self) -> None:
        _write_json(self.state_path, {
            "schema": STATE_SCHEMA, "url": self.options.url,
            "branch": self.options.branch,
            "group_prefix": self.options.group_prefix,
            "last_snapshot": self.last_snapshot,
            "frames": self.frames,
            "refused": self.refused,
        })

    def _exported(self, frame: SourceFrame) -> bool:
        record = self.frames.get(frame.key)
        return record is not None and record.get("size") == frame.size \
            and record.get("mtime_ns") == frame.mtime_ns

    def _done(self, frame: SourceFrame) -> bool:
        if self._exported(frame):
            return True
        record = self.refused.get(frame.key)
        return record is not None and record.get("size") == frame.size \
            and record.get("mtime_ns") == frame.mtime_ns

    def _deleted_records(self) -> dict[str, dict[str, Any]]:
        """Exported frames this export deleted (``delete_after_commit``)
        that are no longer on disk: still found and exported, for the
        counts, though the folder no longer lists them."""

        on_disk = {f.key for f in self.found}
        return {key: record for key, record in self.frames.items()
                if record.get("deleted_utc") and key not in on_disk}

    # -- deleting exported frames -------------------------------------------

    def delete_exported(self, frame: SourceFrame) -> bool:
        """Unlink ``frame``'s file when ``delete_after_commit`` asks and
        the store holds it; returns whether it was deleted.

        Pinned, so that no frame is lost:

        * Only a frame recorded as exported from this very file (its size
          and mtime) is deleted, and :meth:`_reconcile_with_store` keeps a
          record only when the store confirms it -- a frame written to a
          store since deleted or reset is exported again, never deleted
          unexported.  A refused frame stays on disk, and so does one the
          store skipped because it already held a newer file's frame for
          that time (``superseded``: this file's bytes are not there).
        * The file is moved aside (renamed to a hidden name the frame
          walk never lists) and checked there: only the very file that
          was exported (inode, size, mtime) is unlinked.  A file published
          onto the name since -- a history rewind republishing it -- is
          put back under its name (never over a newer one) and exported in
          its turn.  A symlink is kept (unlinking it frees nothing).
        * ``deleted_utc`` is saved before the file is touched, and every
          pass finishes what a killed exporter left (an exported frame
          still on disk, a file still moved aside), so an exporter killed
          anywhere in between deletes or restores it when resumed, and one
          killed after it still counts the frame.
        * Only exported domains' frames are ever touched."""

        if not self.options.delete_after_commit \
                or not self.options.exports(frame.domain) \
                or not self._exported(frame):
            return False
        record = self.frames[frame.key]
        if record.get("outcome") == "superseded":
            return False
        try:
            if frame.path.is_symlink():
                return False
            info = frame.path.stat()
        except OSError:
            return False
        if not self._same_file(info, frame.size, frame.mtime_ns,
                               frame.inode):
            return False
        record["deleted_utc"] = _utc()
        self._save_state()
        aside = _aside(frame.path)
        try:
            os.replace(frame.path, aside)
        except FileNotFoundError:
            return False
        except OSError as error:
            return self._kept(frame.path, record, error)
        return self._finish_aside(frame.path, record, frame.inode)

    @staticmethod
    def _same_file(info, size: int, mtime_ns: int,
                   inode: tuple[int, int] | None) -> bool:
        return (int(info.st_size), int(info.st_mtime_ns)) == (size, mtime_ns) \
            and (inode is None
                 or (int(info.st_dev), int(info.st_ino)) == tuple(inode))

    def _finish_aside(self, path: Path, record: dict[str, Any],
                      inode: tuple[int, int] | None) -> bool:
        """Unlink the file moved aside from ``path`` when it is the one
        ``record`` exported; otherwise put it back (never over a file
        published since)."""

        aside = _aside(path)
        try:
            info = aside.stat()
        except FileNotFoundError:
            return False
        if self._same_file(info, int(record.get("size", -1)),
                           int(record.get("mtime_ns", -1)), inode):
            try:
                aside.unlink()
            except FileNotFoundError:
                return False
            except OSError as error:
                return self._kept(aside, record, error)
            return True
        # Not the exported file: a publication landed between the look
        # and the move.  Back under its name unless a newer one is there.
        record.pop("deleted_utc", None)
        self._save_state()
        try:
            os.link(aside, path)
        except FileExistsError:
            note = (f"a file published onto {path.name} after its export "
                    f"was replaced again since; it is kept as {aside.name}")
            if note not in self.notes:
                self.notes.append(note)
                self.say(f"icechunk export: {note}")
            return False
        except OSError as error:
            return self._kept(aside, record, error)
        aside.unlink()
        return False

    def _kept(self, path: Path, record: dict[str, Any],
              error: OSError) -> bool:
        record.pop("deleted_utc", None)
        self._save_state()
        note = (f"could not delete {path.name} after its commit "
                f"({type(error).__name__}: {error}); it is kept")
        if note not in self.notes:
            self.notes.append(note)
            self.say(f"icechunk export: {note}")
        return False

    def _delete_exported_on_disk(self) -> None:
        """:meth:`delete_exported` every exported frame still on disk (a
        resumed export's, or one whose unlink an earlier pass missed), and
        finish any file a killed exporter left moved aside."""

        if not self.options.delete_after_commit:
            return
        for key, record in list(self.frames.items()):
            if record.get("deleted_utc") and _aside(Path(key)).exists():
                self._finish_aside(Path(key), record, None)
        for frame in self.found:
            if self._exported(frame):
                self.delete_exported(frame)

    # -- progress ----------------------------------------------------------

    def progress(self) -> dict[str, Any]:
        """Frames found, exported and pending, and the lag: per group the
        newest valid time found vs exported, the largest of those gaps in
        model seconds, and how long the oldest pending frame has waited."""

        groups: dict[str, dict[str, Any]] = {}
        first_found: dict[str, datetime] = {}
        pending = [f for f in self.found if not self._done(f)]
        exported = 0
        now_ns = time.time_ns()
        entries = [(self.options.group(f.domain), f.valid,
                    f.key in self.frames and self._done(f))
                   for f in self.found]
        # Deleted after their commit: exported, and gone from the folder.
        for key, record in self._deleted_records().items():
            parsed = parse_frame_name(Path(key).name)
            if parsed is not None:
                entries.append((str(record.get("group")), parsed[1], True))
        for name, valid, done in entries:
            entry = groups.setdefault(name, {
                "found": 0, "exported": 0, "latest_found": valid,
                "latest_exported": None})
            first_found[name] = min(first_found.get(name, valid), valid)
            entry["found"] += 1
            entry["latest_found"] = max(entry["latest_found"], valid)
            if done:
                exported += 1
                entry["exported"] += 1
                entry["latest_exported"] = valid \
                    if entry["latest_exported"] is None \
                    else max(entry["latest_exported"], valid)
        lag = 0.0
        for name, entry in groups.items():
            # Nothing exported yet: behind by the whole span found.
            done = entry["latest_exported"] or first_found[name]
            lag = max(lag, (entry["latest_found"] - done).total_seconds())
            entry["latest_found"] = _iso(entry["latest_found"])
            entry["latest_exported"] = _iso(entry["latest_exported"])
        oldest = min((f.mtime_ns for f in pending), default=None)
        return {
            "frames_found": len(entries),
            "frames_exported": exported,
            "frames_pending": len(pending),
            "frames_refused": len(self.refused),
            "frames_deleted": sum(1 for r in self.frames.values()
                                  if r.get("deleted_utc")),
            "frames_skipped": len(self.skipped),
            "skipped_domains": sorted({f.domain for f in self.skipped}),
            "lag_model_seconds": lag,
            "oldest_pending_wait_seconds": (
                None if oldest is None else max(0.0, (now_ns - oldest) / 1e9)),
            "groups": groups,
        }

    def write_status(self) -> dict[str, Any]:
        status = {
            "schema": STATUS_SCHEMA, "state": self.state, "pid": os.getpid(),
            "url": self.options.url, "branch": self.options.branch,
            "domains": (None if self.options.domains is None
                        else list(self.options.domains)),
            "delete_after_commit": self.options.delete_after_commit,
            "sources": [str(s) for s in self.sources],
            "started_utc": _utc(self.started_ms),
            "updated_unix_ms": _now_ms(),
            "last_snapshot": self.last_snapshot,
            "last_frame_seconds": self.last_frame_seconds,
            "error": self.error, "notes": list(self.notes),
            "refused": {Path(k).name: v.get("reason")
                        for k, v in self.refused.items()},
            "command": list(sys.argv),
            **self.progress(),
        }
        if self.state in ("complete", "incomplete", "empty", "failed",
                          "interrupted"):
            status["finished_utc"] = _utc()
        _write_json(self.status_path, status)
        return status

    # -- one frame ---------------------------------------------------------

    def interval_for(self, domain: int) -> int | None:
        """The history interval a new group's axis is created with: the
        configured one, else the greatest common divisor of the gaps
        between the domain's frames found so far, else ``None`` (one
        frame: a placeholder its second frame replaces)."""

        if domain in self.intervals:
            return self.intervals[domain]
        stamps = sorted({f.valid for f in self.found if f.domain == domain})
        gaps = [int((b - a).total_seconds())
                for a, b in zip(stamps, stamps[1:])]
        gaps = [g for g in gaps if g > 0]
        return math.gcd(*gaps) if gaps else None

    def export_frame(self, frame: SourceFrame) -> dict[str, Any] | None:
        """Write one frame; ``None`` when it vanished before it was read
        (a history rewind), a refusal is recorded, a store failure is
        retried then raised as :class:`ExportFailed`."""

        from woof.energy.icechunk_store import (IcechunkStoreRefused,
                                                TimeAxis, wrfout_frames)

        group = self.options.group(frame.domain)
        delay = 1.0
        for attempt in range(1, TRANSIENT_ATTEMPTS + 1):
            started = time.monotonic()
            try:
                times = [f.time for f in wrfout_frames([frame.path])]
                if not times:
                    raise IcechunkStoreRefused(
                        f"{frame.path.name} holds no frame")
                interval = self.interval_for(frame.domain)
                axis = TimeAxis(times[0], times[0],
                                interval or PLACEHOLDER_INTERVAL_S)
                summary = self._write(
                    self.options.url, [frame.path], group=group, axis=axis,
                    branch=self.options.branch,
                    job_id="woof-icechunk-export",
                    segment_id=frame.path.name, on_existing="newer",
                    extend_axis=True, add_variables=True,
                    compress=self.options.compress,
                    zstd_clevel=self.options.zstd_clevel,
                    write_threads=self.options.threads,
                    axis_interval_known=interval is not None,
                    message=(f"woof icechunk export: {frame.path.name} "
                             f"into {group}"))
            except IcechunkStoreRefused as error:
                if not frame.path.exists():
                    return None
                self.refused[frame.key] = {**frame.signature(),
                                           "group": group,
                                           "reason": str(error)}
                self.say(f"icechunk export: refused {frame.path.name}: "
                         f"{error}")
                return None
            except Exception as error:  # noqa: BLE001 - a store failure
                # of any kind (network, throttling, an icechunk error) is
                # retried, then ends the export resumably; a frame that
                # vanished (a history rewind) is simply gone.
                if not frame.path.exists():
                    return None
                if attempt == TRANSIENT_ATTEMPTS:
                    raise ExportFailed(
                        f"{frame.path.name} into {group} failed "
                        f"{attempt} times; last: {type(error).__name__}: "
                        f"{error}") from error
                self.say(f"icechunk export: {frame.path.name} failed "
                         f"({type(error).__name__}: {error}); retrying in "
                         f"{delay:g} s")
                self.sleep(delay)
                delay *= 2
                continue
            self.last_frame_seconds = round(time.monotonic() - started, 3)
            self.last_snapshot = summary["snapshot"]
            outcome = ("replaced" if summary.get("frames_replaced")
                       else "written" if summary.get("frames_written")
                       else "superseded" if summary.get("frames_superseded")
                       else "already-written")
            self.frames[frame.key] = {
                **frame.signature(), "group": group,
                "valid_time": _iso(frame.valid), "outcome": outcome,
                "snapshot": summary["snapshot"],
                "added_variables": summary.get("added_variables", []),
            }
            self.refused.pop(frame.key, None)
            if summary.get("added_variables"):
                self.say(f"icechunk export: {group} gained "
                         f"{', '.join(summary['added_variables'])} at "
                         f"{frame.path.name} (NaN before)")
            return summary
        return None

    # -- the loop ----------------------------------------------------------

    def _finish_requested(self) -> bool:
        if self.finish_file is not None and self.finish_file.exists():
            return True
        return not _alive(self.parent_pid, self.parent_created)

    def run(self, *, follow: bool = False) -> dict[str, Any]:
        """Export until done; returns the final status (also
        ``status.json``).  ``state`` is ``complete``, ``incomplete``
        (refused frames), ``empty`` (no frame found), ``failed`` (store
        failures) or ``interrupted``."""

        self.state_dir.mkdir(parents=True, exist_ok=True)
        if self.unconfirmed is not None:
            self.state, self.error = "failed", self.unconfirmed
            return self.write_status()
        self.state = "following" if follow else "draining"
        self.write_status()
        try:
            while True:
                finishing = (not follow) or self._finish_requested()
                if finishing and follow and self.state != "finishing":
                    self.state = "finishing"
                every = discover_frames(self.sources, markers=self._markers)
                self.found = [f for f in every
                              if self.options.exports(f.domain)]
                self.skipped = [f for f in every
                                if not self.options.exports(f.domain)]
                self._delete_exported_on_disk()
                pending = [f for f in self.found if not self._done(f)]
                if not pending:
                    self.write_status()
                    if finishing:
                        break
                    self.sleep(self.poll_seconds)
                    continue
                # Oldest first; look again once a poll interval has passed,
                # so the frames published meanwhile are counted and the lag
                # in status.json stays current.
                looked = time.monotonic()
                for frame in pending:
                    self.export_frame(frame)
                    self._save_state()
                    # Only once the commit and its record are saved.
                    self.delete_exported(frame)
                    self.write_status()
                    if time.monotonic() - looked >= self.poll_seconds:
                        break
        except KeyboardInterrupt:
            self.state = "interrupted"
            self._save_state()
            return self.write_status()
        except ExportFailed as error:
            self.state, self.error = "failed", str(error)
            self._save_state()
            return self.write_status()
        if not self.found and not self._deleted_records():
            # Not "complete": a store holding nothing because the frames
            # were written somewhere else must not read as a finished
            # export.
            self.error = ("no history frames were found under "
                          + ", ".join(str(s) for s in self.sources))
            if self.skipped:
                self.error += (
                    f" of domain(s) {list(self.options.domains or ())}; "
                    f"{len(self.skipped)} frame(s) of domain(s) "
                    f"{sorted({f.domain for f in self.skipped})} were "
                    "skipped")
            self.state = "empty"
            self._save_state()
            return self.write_status()
        self.state = "incomplete" if self.refused else "complete"
        self._save_state()
        return self.write_status()


# --------------------------------------------------------------------------
# the spec, and woof go's side


def write_spec(state_dir: Path, options: ExportOptions,
               sources: Sequence[Path], *,
               intervals: Mapping[int, int] | None = None,
               written_by: str = "woof go") -> Path:
    path = Path(state_dir) / SPEC_NAME
    _write_json(path, {
        "schema": SPEC_SCHEMA, "options": options.to_mapping(),
        "sources": [str(Path(s)) for s in sources],
        "intervals": {str(k): int(v) for k, v in (intervals or {}).items()},
        "written_by": written_by, "written_utc": _utc(),
    })
    return path


def read_spec(path: Path) -> tuple[ExportOptions, list[Path], dict[int, int]]:
    data = _read_json(path)
    if data is None or data.get("schema") != SPEC_SCHEMA:
        raise IcechunkExportRefused(
            f"{path} is not a {SPEC_SCHEMA} export description")
    options = ExportOptions.from_mapping(data.get("options"),
                                         source=str(path), allow_memory=True)
    sources = [Path(s) for s in data.get("sources") or []]
    intervals = {int(k): int(v) for k, v in
                 (data.get("intervals") or {}).items()}
    return options, sources, intervals


def resume_command(state_dir: Path) -> str:
    from woof.bridges import shell_line

    return shell_line("woof", "icechunk-export", str(state_dir))


def _words(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    if minutes:
        return f"{minutes} min {secs:02d} s"
    return f"{secs} s"


def progress_line(status: Mapping[str, Any] | None) -> str:
    """One line for a terminal: committed/found, pending and lag."""

    if not status:
        return "icechunk export: starting"
    line = (f"icechunk export: {status.get('frames_exported', 0)}/"
            f"{status.get('frames_found', 0)} frames committed")
    pending = int(status.get("frames_pending") or 0)
    if pending:
        line += (f", {pending} pending ("
                 f"{_words(status.get('lag_model_seconds'))} of model time "
                 "behind, oldest waiting "
                 f"{_words(status.get('oldest_pending_wait_seconds'))})")
    if status.get("frames_refused"):
        line += f", {status['frames_refused']} refused"
    return line


def experiment_names(config: Path | str | None) -> dict[str, str]:
    """The values :data:`URL_PLACEHOLDERS` take for a config: its
    ``[experiment] name``; ``{}`` when it has none."""

    if config is None or not Path(config).is_file():
        return {}
    import tomllib

    try:
        raw = tomllib.loads(Path(config).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return _names_in(raw)


def _names_in(raw: Mapping[str, Any]) -> dict[str, str]:
    experiment = raw.get("experiment")
    name = experiment.get("name") if isinstance(experiment, Mapping) \
        else None
    if not isinstance(name, str) or not name:
        return {}
    return {key: name for key in URL_PLACEHOLDERS}


def go_export_options(args, config: Path | None) -> ExportOptions | None:
    """``woof go``'s export request: ``--icechunk URL`` over the config's
    ``[output] icechunk``; ``None`` when neither asks.  A flag URL
    replaces only the table's ``url``.  ``{name}`` in either is filled
    with the config's ``[experiment] name``."""

    table = None
    raw: dict = {}
    if config is not None and Path(config).is_file():
        import tomllib

        try:
            raw = tomllib.loads(Path(config).read_text(encoding="utf-8-sig"))
        except ValueError:
            raw = {}
        output = raw.get("output")
        if isinstance(output, Mapping):
            table = output.get("icechunk")
    flag = getattr(args, "icechunk", None)
    if flag is not None:
        if isinstance(table, Mapping):
            table = {**table, "url": flag}
        else:
            table = {"url": flag}
        source = "--icechunk"
        base = Path.cwd()
    else:
        source = str(config)
        base = None if config is None else Path(config).resolve().parent
    options = ExportOptions.from_mapping(table, source=source, base_dir=base,
                                         names=_names_in(raw))
    # A [simulated_radar] table is enabled unless it says otherwise
    # (SimulatedRadarOptions.from_mapping).
    radar = raw.get("simulated_radar")
    if options is not None and options.delete_after_commit \
            and isinstance(radar, Mapping) and radar.get("enabled", True):
        from woof.explain import layered

        raise IcechunkExportRefused(layered(
            f"[output] icechunk delete_after_commit of {source} and "
            "[simulated_radar] cannot run together. Next: drop one, or set "
            "[simulated_radar] enabled = false.",
            "Simulated radar scans each history frame after the next one "
            "on its grid lands, from the file; the exporter would already "
            "have deleted it, and the radar's failure stops the forecast."))
    return options


#: Set by :meth:`LiveExport.start` for a delete-after-commit export, so
#: the child stages' output-disk reviews price the exported grids'
#: history as the export's backlog (:mod:`woof.output_disk`).  JSON:
#: ``{"domains": [grid ids] | null}`` (null: every grid).
LIVE_EXPORT_DELETE_ENV = "WOOF_LIVE_EXPORT_DELETES"


class LiveExport:
    """``woof go``'s handle on the exporter process: :meth:`start` before
    the forecast, :meth:`finish` when it ends (however it ends),
    :meth:`wait` before the run is declared done.  Nothing here can stop
    the forecast: a failure to start or a failed exporter is recorded and
    said, with the resume command."""

    REPORT_SECONDS = 300.0
    WAIT_LINE_SECONDS = 15.0
    #: How long :meth:`wait` lets the exporter go without a word (no
    #: ``status.json`` update) before it is stopped and the export left
    #: for the resume command.  Pinned at 30 min: a store call that
    #: neither returns nor fails must not hold ``go`` for ever, and a 3 GB
    #: frame takes minutes, not half an hour, on any link worth exporting
    #: over.
    STALL_SECONDS = 1800.0

    def __init__(self, options: ExportOptions, *, sources: Sequence[Path],
                 state_dir: Path, intervals: Mapping[int, int] | None = None,
                 say: Callable[[str], None] | None = None,
                 poll_seconds: float = DEFAULT_POLL_SECONDS,
                 env: Mapping[str, str] | None = None):
        self.options = options
        self.sources = [Path(s) for s in sources]
        self.state_dir = Path(state_dir)
        self.intervals = dict(intervals or {})
        self.say = say or (lambda line: print(line, flush=True))
        self.poll_seconds = poll_seconds
        self.env = dict(env) if env is not None else None
        self.process: subprocess.Popen | None = None
        self.start_error: str | None = None
        self._stop = threading.Event()
        self._finishing = threading.Event()
        self._launch_lock = threading.Lock()
        self._starter: threading.Thread | None = None
        self._reporter: threading.Thread | None = None
        self._abandoned = False
        self._log = None

    @property
    def status_path(self) -> Path:
        return self.state_dir / STATUS_NAME

    @property
    def finish_path(self) -> Path:
        return self.state_dir / FINISH_NAME

    def command(self) -> list[str]:
        return [sys.executable, "-m", "woof.icechunk_export",
                str(self.state_dir / SPEC_NAME), "--follow",
                "--finish-file", str(self.finish_path),
                "--parent-pid", str(os.getpid()),
                *(["--parent-created", str(created)]
                  if (created := _created(os.getpid())) is not None else []),
                "--poll-seconds", str(self.poll_seconds)]

    def start(self, *, after: Path | None = None) -> bool:
        """Start the exporter now, or -- with ``after`` -- once that path
        exists; a run that ends before it does starts none.

        ``after`` is for a launcher that refuses an output folder holding
        anything it did not write (``woof run-plan``'s prepared launch):
        the export folder appears only once the run has claimed it."""

        self.say(f"go: exporting each history frame to Icechunk "
                 f"{self.options.url} as it lands (its own process; the "
                 f"model never waits on it). Progress: {self.status_path}")
        if self.options.delete_after_commit:
            # The stages run as child processes and price their own output
            # disk: tell them which grids' frames this export unlinks.
            os.environ[LIVE_EXPORT_DELETE_ENV] = json.dumps(
                {"domains": (None if self.options.domains is None
                             else sorted(int(g) for g in self.options.domains))})
        if after is None:
            return self._launch()
        self._starter = threading.Thread(
            target=self._launch_after, args=(Path(after),),
            name="icechunk-export-start", daemon=True)
        self._starter.start()
        return True

    def _launch_after(self, after: Path) -> None:
        while not after.exists():
            if self._finishing.is_set():
                # The run ended (or was stopped) before it claimed its
                # folder: nothing was published, and nothing is created.
                self.start_error = (f"the run ended before it made {after}, "
                                    "so no exporter was started")
                return
            self._finishing.wait(min(1.0, self.poll_seconds))
        if self._abandoned:
            return
        self._launch()

    def _launch(self) -> bool:
        with self._launch_lock:
            if self.process is not None or self.start_error is not None:
                return self.process is not None
            try:
                self.state_dir.mkdir(parents=True, exist_ok=True)
                self.finish_path.unlink(missing_ok=True)
                write_spec(self.state_dir, self.options, self.sources,
                           intervals=self.intervals)
                self._log = open(self.state_dir / LOG_NAME, "ab")
                self.process = subprocess.Popen(
                    self.command(), stdin=subprocess.DEVNULL,
                    stdout=self._log, stderr=subprocess.STDOUT, env=self.env)
            except OSError as error:
                self.start_error = f"{type(error).__name__}: {error}"
                self.say(f"go: WARNING -- the Icechunk export could not "
                         f"start ({self.start_error}); the forecast runs "
                         "without it. Next, after the run: "
                         f"{resume_command(self.state_dir)}")
                return False
            if self._finishing.is_set():
                self.finish()
        self._reporter = threading.Thread(target=self._report,
                                          name="icechunk-export-progress",
                                          daemon=True)
        self._reporter.start()
        return True

    def status(self) -> dict | None:
        return _read_json(self.status_path)

    def _report(self) -> None:
        said = None
        while not self._stop.wait(self.REPORT_SECONDS):
            status = self.status()
            if status is None:
                continue
            key = (status.get("frames_exported"), status.get("frames_found"))
            if key != said:
                said = key
                self.say("go: " + progress_line(status))

    def finish(self) -> None:
        """Tell the exporter the forecast has ended (idempotent)."""

        self._finishing.set()
        if self.process is None:
            return
        try:
            _write_json(self.finish_path, {"finished_utc": _utc()})
        except OSError:
            pass   # the exporter also sees this process exit

    def abandon(self) -> None:
        """An interrupted run: stop reporting and say how to resume."""

        self._abandoned = True
        self._stop.set()
        self._finishing.set()
        if self._starter is not None:
            self._starter.join()
        if self.process is not None:
            self.say("go: the Icechunk export stopped with the run; "
                     f"resume it: {resume_command(self.state_dir)}")

    def wait(self) -> dict[str, Any]:
        """Wait for the exporter to drain and exit; a line every
        :attr:`WAIT_LINE_SECONDS`, and the exporter stopped after
        :attr:`STALL_SECONDS` without progress.  Returns the final status
        with ``ok`` (complete) and ``exit_code``."""

        self._stop.set()
        self.finish()
        if self._starter is not None:
            self._starter.join()
        if self.process is None:
            self.say("go: WARNING -- the Icechunk export did not run "
                     f"({self.start_error}). The forecast's own outputs are "
                     f"unaffected. Next: {resume_command(self.state_dir)}")
            return {"ok": False, "state": "not-started",
                    "error": self.start_error}
        said_at = 0.0
        last_line = None
        progressed = time.monotonic()
        last_update = None
        stalled = None
        while True:
            code = self.process.poll()
            if code is not None:
                break
            now = time.monotonic()
            status = self.status()
            update = None if status is None else status.get("updated_unix_ms")
            if update != last_update:
                last_update, progressed = update, now
            elif now - progressed >= self.STALL_SECONDS:
                stalled = (f"the exporter made no progress for "
                           f"{_words(now - progressed)} and was stopped")
                self.process.terminate()
                try:
                    code = self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    code = self.process.wait()
                break
            line = "go: " + progress_line(status) + ", finishing"
            if now - said_at >= self.WAIT_LINE_SECONDS and line != last_line:
                self.say(line)
                said_at, last_line = now, line
            time.sleep(0.5)
        if self._log is not None:
            self._log.close()
        status = self.status() or {}
        state = status.get("state")
        ok = code == 0 and state == "complete" and stalled is None
        result = {**status, "ok": ok, "exit_code": code}
        if ok:
            self.say(f"go: icechunk export complete -- "
                     f"{status.get('frames_exported', 0)} frames in "
                     f"{self.options.url} (snapshot "
                     f"{status.get('last_snapshot')})")
            return result
        if stalled is not None:
            reason = stalled
        elif status.get("error"):
            reason = str(status["error"])
        elif status.get("frames_refused"):
            reason = (f"{status['frames_refused']} frame(s) refused: "
                      + "; ".join(f"{k}: {v}" for k, v in
                                  (status.get("refused") or {}).items()))
        else:
            reason = (f"the exporter exited {code} in state {state!r}; "
                      f"see {self.state_dir / LOG_NAME}")
        self.say(f"go: WARNING -- the Icechunk export is INCOMPLETE "
                 f"({status.get('frames_exported', 0)}/"
                 f"{status.get('frames_found', 0)} frames): {reason}. "
                 "The forecast's own outputs are unaffected. Next: "
                 f"{resume_command(self.state_dir)}")
        return result


# --------------------------------------------------------------------------
# woof icechunk-export


def _spec_for(path: Path) -> Path | None:
    """The ``export.json`` a PATH names: itself, inside it, in its
    ``icechunk-export`` folder, or -- for a forecast folder ``woof go``
    exported (``<run>/run`` or ``<run>/chain/run``) -- in its parent's or
    grandparent's ``icechunk-export`` when that export's sources include
    it."""

    path = Path(path).resolve()
    if path.is_file() and path.name == SPEC_NAME:
        return path
    for candidate in (path / SPEC_NAME, path / EXPORT_DIRNAME / SPEC_NAME):
        if candidate.is_file():
            return candidate
    for ancestor in (path.parent, path.parent.parent):
        candidate = ancestor / EXPORT_DIRNAME / SPEC_NAME
        if candidate.is_file():
            data = _read_json(candidate) or {}
            if str(path) in [str(Path(s).resolve())
                             for s in data.get("sources") or []]:
                return candidate
    return None


def _plan(args) -> tuple[ExportOptions, list[Path], dict[int, int], Path]:
    specs: list[tuple[ExportOptions, list[Path], dict[int, int], Path]] = []
    loose: list[Path] = []
    for raw in args.paths:
        path = Path(raw)
        if not path.exists():
            raise IcechunkExportRefused(f"{path} does not exist")
        spec = _spec_for(path)
        if spec is None:
            loose.append(path.resolve())
            continue
        options, sources, intervals = read_spec(spec)
        specs.append((options, sources, intervals, spec.parent))
    flags = {"branch": args.branch, "zstd_clevel": args.zstd_clevel,
             "group_prefix": args.group_prefix, "threads": args.threads,
             "domains": args.domains}
    if args.no_compress:
        flags["compress"] = False
    if args.delete_after_commit:
        flags["delete_after_commit"] = True
    flags = {k: v for k, v in flags.items() if v is not None}
    if specs:
        options = specs[0][0]
        for other, *_ in specs[1:]:
            if other != options:
                raise IcechunkExportRefused(
                    "the export folders name different stores or options: "
                    f"{options.url} and {other.url}; export them separately")
        mapping = options.to_mapping()
        if args.url is not None:
            mapping["url"] = args.url
    else:
        if args.url is None:
            raise IcechunkExportRefused(
                "no export.json in " + ", ".join(map(str, args.paths))
                + "; give the store with --url s3://..., gs://..., "
                "file:///... or a directory")
        mapping = {"url": args.url}
    mapping.update(flags)
    options = ExportOptions.from_mapping(
        mapping, source="woof icechunk-export", base_dir=Path.cwd(),
        allow_memory=True,
        names=None if args.name is None else {"name": args.name})
    sources: list[Path] = []
    intervals: dict[int, int] = {}
    for _o, spec_sources, spec_intervals, _d in specs:
        sources.extend(spec_sources)
        intervals.update(spec_intervals)
    sources.extend(loose)
    sources = list(dict.fromkeys(sources))
    if args.state_dir is not None:
        state_dir = Path(args.state_dir).resolve()
    elif specs:
        state_dir = specs[0][3]
    else:
        state_dir = loose[0] / EXPORT_DIRNAME
    return options, sources, intervals, state_dir


def _grid_ids(text: str) -> list[int]:
    import argparse

    try:
        values = [int(part) for part in str(text).split(",") if part.strip()]
    except ValueError:
        values = []
    if not values:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a comma-separated list of grid ids, e.g. 3 "
            "or 2,3")
    return values


def add_arguments(parser) -> None:
    parser.add_argument(
        "paths", nargs="+", type=Path, metavar="PATH",
        help="a run's export folder (icechunk-export, with export.json), a "
             "woof go run folder, or forecast output folders; several "
             "folders of one forecast (a run and its resumed continuation) "
             "go into one store, the later file winning an overlap")
    parser.add_argument("--url", default=None, metavar="URL",
                        help="the store: s3://bucket/prefix, gs://bucket/"
                             "prefix, file:///path or a directory (default: "
                             "the export.json's); {name} in it is --name")
    parser.add_argument("--name", default=None, metavar="NAME",
                        help="the value of {name} in --url (woof go fills "
                             "it with the config's [experiment] name)")
    parser.add_argument("--branch", default=None, metavar="NAME",
                        help="Icechunk branch (default main)")
    parser.add_argument("--zstd-clevel", type=int, default=None, metavar="N",
                        help="zstd level for the Blosc zstd+bitshuffle "
                             "compression (default 5)")
    parser.add_argument("--no-compress", action="store_true",
                        help="store arrays with zarr's default codec instead "
                             "of Blosc zstd+bitshuffle")
    parser.add_argument("--group-prefix", default=None, metavar="PREFIX",
                        help="put domain N in group PREFIXdNN (default dNN)")
    parser.add_argument("--threads", type=int, default=None, metavar="N",
                        help=f"arrays encoded at once (default "
                             f"{DEFAULT_THREADS})")
    parser.add_argument("--domains", type=_grid_ids, default=None,
                        metavar="N[,N...]",
                        help="export only these grid ids (e.g. 3); the "
                             "other domains' frames are skipped (default: "
                             "every domain, or the export.json's)")
    parser.add_argument("--delete-after-commit", action="store_true",
                        help="unlink each wrfout once the store holds it "
                             "(default: the export.json's, else keep)")
    parser.add_argument("--state-dir", type=Path, default=None, metavar="DIR",
                        help="where state.json and status.json go (default: "
                             "the export folder, else PATH/icechunk-export)")
    parser.add_argument("--follow", action="store_true",
                        help="keep exporting frames as they land until "
                             "--finish-file appears or --parent-pid exits")
    parser.add_argument("--finish-file", type=Path, default=None,
                        metavar="FILE", help="with --follow: drain and stop "
                                             "once FILE exists")
    parser.add_argument("--parent-pid", type=int, default=None, metavar="PID",
                        help="with --follow: drain and stop once PID exits")
    parser.add_argument("--parent-created", default=None, metavar="STAMP",
                        help="with --parent-pid: that process's creation "
                             "stamp, so a reused pid is not mistaken for it")
    parser.add_argument("--poll-seconds", type=float,
                        default=DEFAULT_POLL_SECONDS, metavar="S",
                        help=f"with --follow: seconds between looks "
                             f"(default {DEFAULT_POLL_SECONDS:g})")
    parser.add_argument("--json", action="store_true",
                        help="print the final status document on stdout")


def register_cli(sub) -> None:
    parser = sub.add_parser(
        "icechunk-export",
        help="export a forecast's history frames to Icechunk, or resume "
             "or complete the export woof go ran beside it",
        description=(
            "Write every wrfout frame of a run into an Icechunk repository, "
            "one group per domain (dNN), one commit per frame, Blosc "
            "zstd+bitshuffle.  Frames already in the store from the same "
            "file are skipped; a newer file of the same valid time "
            "replaces the older (a resumed run wins its overlap).  "
            "Exit 0 complete, 1 store failure, 2 refused."))
    add_arguments(parser)
    parser.set_defaults(func=main)


def main(args) -> int:
    try:
        options, sources, intervals, state_dir = _plan(args)
    except IcechunkExportRefused as error:
        print(f"woof icechunk-export: refused: {error}", file=sys.stderr)
        return 2
    if args.follow and hasattr(os, "nice"):
        try:
            os.nice(EXPORTER_NICE)
        except OSError:
            pass
    exporter = Exporter(sources, options, state_dir=state_dir,
                        intervals=intervals,
                        poll_seconds=args.poll_seconds,
                        finish_file=args.finish_file,
                        parent_pid=args.parent_pid,
                        parent_created=args.parent_created,
                        say=(lambda line: print(line, file=sys.stderr,
                                                flush=True)) if args.json
                        else None)
    if not args.json:
        print(f"icechunk export: {', '.join(map(str, sources))} -> "
              f"{options.url} (state {state_dir})", flush=True)
    status = exporter.run(follow=bool(args.follow))
    if args.json:
        print(json.dumps(status, indent=2, sort_keys=True))
    else:
        print(progress_line(status) + f"; {status['state']}", flush=True)
        for note in status.get("notes") or []:
            print(f"icechunk export: {note}", flush=True)
        if status.get("error"):
            print(f"icechunk export: {status['error']}", file=sys.stderr)
    return {"complete": 0, "incomplete": 2, "empty": 2,
            "interrupted": 130}.get(status["state"], 1)


def _module_main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m woof.icechunk_export",
        description="the exporter woof go starts; see woof icechunk-export")
    add_arguments(parser)
    return main(parser.parse_args(argv))


__all__ = [
    "SPEC_SCHEMA", "STATE_SCHEMA", "STATUS_SCHEMA", "EXPORT_DIRNAME",
    "EXPORT_INCOMPLETE_EXIT_CODE", "DEFAULT_THREADS", "OPTION_KEYS",
    "ExportOptions", "Exporter", "ExportFailed", "IcechunkExportRefused",
    "LiveExport", "SourceFrame", "URL_PLACEHOLDERS", "discover_frames",
    "experiment_names", "go_export_options",
    "history_intervals", "parse_frame_name", "progress_line", "read_spec",
    "register_cli", "resume_command", "write_spec", "main",
]


if __name__ == "__main__":
    raise SystemExit(_module_main())
