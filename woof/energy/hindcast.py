"""``woof energy hindcast``: a long reanalysis as a chain of cold-started segments.

A year of 100-300 m wind-farm weather cannot be one run: no boundary
source forecasts a year ahead, and a run restarted from its own checkpoint
cannot switch to a new forcing cycle (only the HRRR route chains restarts
across cycles).  So a hindcast is a chain of **re-initialised segments**,
each a fresh run from one source cycle whose first hours are discarded as
spin-up.  This module expands *clusters x date range* into one job per
segment and drives each job through the existing planner, ``woof energy
run`` and ``woof energy extract``.

Pinned here
-----------

**Segments.**  Segment ``k`` of a request runs from ``first + k * keep``
for ``segment_hours`` (default :data:`DEFAULT_SEGMENT_HOURS`, 6 h), where
``keep = segment_hours - spinup_hours`` (default 6 h) and ``first`` is
``--start`` at ``--cycle-hour`` (default 00 UTC).  Segments are generated
while their run start is before ``--end`` (exclusive), so ``--start
2025-11-01 --end 2025-11-04`` is twelve 6-hourly segments.  Its first
``spinup_hours`` (default :data:`DEFAULT_SPINUP_HOURS`, 0 h) are spin-up;
the **kept window** is ``[run_start + spinup, run_end)``, half-open, so
consecutive segments tile time with no gap and no duplicate frame.
Everything is naive UTC arithmetic on :class:`datetime.datetime`: month
ends, 29 February and daylight-saving changes need no special case, and
an input with a non-UTC offset is refused rather than converted.

**Job identity.**  A job's id is ``<cluster>-<YYYYMMDDTHH>Z-<sha12>``,
where the sha256 is over the canonical JSON of every input that changes
what the job computes: the cluster's site content hash, leaf spacing and
chain, the segment window, the chosen source, the planner topology and
arguments, the history settings and the output targets.  The same request
produces byte-identical ``job.json`` files; nothing time-dependent is in
them.

**Source per segment.**  The source is chosen per job by a selector
``selector(region, cycle, hours, p_top_pa=..., priority=..., now=...)``
returning an object with ``.source`` (and optionally ``.reasons``).  The
default is :func:`woof.energy.source_select.select_source` when that
module is importable; without it the built-in :func:`fallback_select`
takes the first of :data:`DEFAULT_FALLBACK_PRIORITY` whose publication
window (:func:`woof.energy.source_window.window_refusal`) serves the
segment.  ``--source`` pins one source for every job.  Every job records
the selector, its reasons, the previous segment's source and whether it
changed, because a source change between segments is a discontinuity a
user of the reanalysis must be able to find.

**History.**  On ``wrf-nests`` every ``[[domain]]`` of the emitted
configuration gets ``history_begin_s = spinup_hours * 3600``, leaves write
every ``leaf_history_interval_s`` (default 900 s; finer such as 300 s is
allowed) and parents every ``parent_history_interval_s`` (default 3600 s).
On ``wrf-tiles`` the parent run's history IS the tiles' boundary forcing
and the tiles start where it starts, so the parent keeps its frames from
t = 0 at the planner's cadence (an explicit parent interval is refused);
the offline tiles read a legacy ``[run]`` table with no history window,
so their spin-up frames are written and the kept window is applied at
extraction (and handed to the writer).  A leaf tile's ``output_interval_s``
is patched to the leaf interval and its ``--child-config-sha256`` binding
recomputed.

**What a leaf writes.**  ``--leaf-output-preset`` (default
:data:`DEFAULT_LEAF_OUTPUT_PRESET`, ``windfarm``) is the ``[output]`` of
every leaf (a ``wrf-nests`` leaf's ``output = {...}``, a ``wrf-tiles``
leaf tile's ``[output]`` table): ``windfarm`` writes only 2-D planes --
the hub-height wind and scalars, the near-surface state and the gust
family (``woof.io.history_selection.WINDFARM_FIELD_REASONS``) -- with
``hub_heights_m`` set to the cluster sites document's ``heights_m``, so
``woof energy extract`` reads the forecast heights straight off the hub
planes; ``energy`` and ``full`` keep their 3-D volumes plus
``hub_wind = true`` (the default six heights).  The preset and the hub
heights are recorded in the job's ``history`` and bind its identity.  A
job document written before this key existed carries no preset and
keeps the ``full`` leaves it was planned with.  Every interval must divide the kept window.  Patched files
are re-read through the door's own loader before they replace the
planner's.

**Files.**  Under the hindcast directory::

    hindcast-manifest.json          woof-energy.hindcast-manifest.v1
    clusters/<cluster>/sites-<sha12>.json  the cluster's sites, by content
    jobs/<job_id>/job.json          woof-energy.hindcast-job.v1
    jobs/<job_id>/plan/plan.json    woof-energy.plan.v1 (materialize)
    jobs/<job_id>/materialized.json woof-energy.hindcast-materialized.v1
    jobs/<job_id>/sites-forecast.nc forecast.v1, kept window (run-local)

Paths inside job documents are relative to the job directory.  Every JSON
document is written to ``<name>.partial`` and renamed.  The manifest is
updated under an advisory ``flock`` so jobs run by several workers on one
filesystem do not lose each other's status.

**Statuses.**  ``pending`` (planned), ``materialized`` (plan written),
``running``, ``complete`` (every plan domain complete and the post-step
returned), ``failed`` (with a reason).  ``run-local`` skips a complete job
unless forced; a failed job reruns with ``--resume`` passed on to
``woof energy run``.

Python boundary: orchestration over a few hundred records per cluster; no
field data is touched except the time slice of the extracted site series.
"""

from __future__ import annotations

import contextlib
import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import string
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

from woof.energy.contracts import (ContractError, Plan, Site, SiteSet,
                                   dump_plan, load_plan, load_sites,
                                   sha256_file)

JOB_SCHEMA = "woof-energy.hindcast-job.v1"
MANIFEST_SCHEMA = "woof-energy.hindcast-manifest.v1"
MATERIALIZED_SCHEMA = "woof-energy.hindcast-materialized.v1"
PARTITION_SCHEMA = "woof-energy.partition.v1"

MANIFEST_NAME = "hindcast-manifest.json"
LOCK_NAME = ".hindcast-manifest.lock"
JOBS_DIR = "jobs"
CLUSTERS_DIR = "clusters"
JOB_NAME = "job.json"
PLAN_DIR = "plan"
MATERIALIZED_NAME = "materialized.json"
SITES_FORECAST_NAME = "sites-forecast.nc"

#: Job states a manifest records.
JOB_STATUSES = ("pending", "materialized", "running", "complete", "failed")
#: Topologies a hindcast drives (hex-swath is an experimental lane).
TOPOLOGIES = ("wrf-nests", "wrf-tiles")
_PLANNER_MODULES = {"wrf-nests": "woof.energy.plan_wrf_nests",
                    "wrf-tiles": "woof.energy.plan_wrf_tiles"}

#: One forecast per global cycle: each 6 h run starts at a 00/06/12/18 UTC
#: init and is kept whole, half-open, so the 6-hourly forecasts stitch into
#: one continuous analysis ([00, 06), [06, 12), ...) with every frame from
#: the freshest init.  ``--spinup-hours`` > 0 trades that alignment for a
#: discarded lead: 7 h runs with 1 h spin-up keep [01, 07), [07, 13), ...
DEFAULT_SEGMENT_HOURS = 6
DEFAULT_SPINUP_HOURS = 0
DEFAULT_CYCLE_HOUR = 0
DEFAULT_TOPOLOGY = "wrf-nests"
DEFAULT_DX_M = 100.0
DEFAULT_CORRIDOR_KM = 2.0
#: Leaf history cadence (s): quarter-hourly energy products.
DEFAULT_LEAF_HISTORY_INTERVAL_S = 900.0
#: Parent history cadence (s) on wrf-nests; wrf-tiles keeps the planner's
#: (the parent's history is the tiles' boundary forcing).
DEFAULT_NESTS_PARENT_HISTORY_INTERVAL_S = 3600.0
#: Model top the planners emit (about 20 km); passed to the selector and
#: checked against the emitted configuration.
#: The wind-energy column's top (woof.energy.vertical.DEFAULT_P_TOP_PA):
#: 5500 Pa is 19.97 km in the standard atmosphere, inside the hindcast's
#: 20 km ceiling, where WRF's usual 5000 Pa is 20.58 km.
DEFAULT_P_TOP_PA = 5500.0
#: Built-in selector order when woof.energy.source_select is absent:
#: the Met Office global model, then GFS.
DEFAULT_FALLBACK_PRIORITY = ("met-office-global", "gfs")
#: Output roots without a URL scheme are relative to the hindcast dir.
DEFAULT_OUTPUT_ROOT = "icechunk"
#: What a leaf domain writes (``--leaf-output-preset``): the wind-farm 2-D
#: set by default; ``energy``/``full`` keep their 3-D volumes.
LEAF_OUTPUT_PRESETS = ("windfarm", "energy", "full")
DEFAULT_LEAF_OUTPUT_PRESET = "windfarm"
#: The preset a job document without ``history.leaf_output_preset`` was
#: planned with (every job before the key existed).
LEGACY_LEAF_OUTPUT_PRESET = "full"
#: One gridded store per cluster leaf for the whole hindcast; each job
#: writes its kept window into it.
DEFAULT_GRIDDED_TEMPLATE = "{root}/gridded/{cluster_id}/{leaf}.icechunk"
#: One site-series store per cluster.
DEFAULT_SITES_TEMPLATE = "{root}/sites/{cluster_id}.icechunk"
_TEMPLATE_FIELDS = {"root", "cluster_id", "leaf", "topology", "dx_m"}


class HindcastRefused(ContractError):
    """``woof energy hindcast`` will not plan or run this as asked."""


@dataclass(frozen=True)
class SourceChoice:
    """The boundary source for one segment, and why."""

    source: str
    reasons: tuple[str, ...] = ()
    selector: str = "fallback"


@dataclass(frozen=True)
class Cluster:
    """One set of sites planned together (a partition cluster or a whole
    sites document)."""

    cluster_id: str
    site_ids: tuple[str, ...]
    region: Any
    origin: str                         # "sites" | "partition"
    leaf_dx_m: float | None = None
    chain: tuple[int, ...] | None = None
    topology: str | None = None


@dataclass(frozen=True)
class Segment:
    index: int
    run_start: datetime
    run_end: datetime
    keep_start: datetime
    keep_end: datetime

    def to_json(self, spinup_hours: int) -> dict:
        return {"index": self.index, "cycle": _iso(self.run_start),
                "run_start": _iso(self.run_start),
                "run_end": _iso(self.run_end),
                "run_hours": _hours(self.run_end - self.run_start),
                "spinup_hours": spinup_hours,
                "keep_start": _iso(self.keep_start),
                "keep_end": _iso(self.keep_end),
                "keep_hours": _hours(self.keep_end - self.keep_start),
                "history_begin_s": float(spinup_hours * 3600)}


@dataclass
class HindcastRequest:
    """Everything ``woof energy hindcast plan`` was asked for."""

    outdir: Path
    start: datetime
    end: datetime
    sites_path: Path | None = None
    partition_path: Path | None = None
    clusters: tuple[str, ...] | None = None
    cluster_id: str | None = None
    topology: str = DEFAULT_TOPOLOGY
    segment_hours: int = DEFAULT_SEGMENT_HOURS
    spinup_hours: int = DEFAULT_SPINUP_HOURS
    cycle_hour: int = DEFAULT_CYCLE_HOUR
    start_has_hour: bool = False
    dx_m: float | None = None
    corridor_km: float = DEFAULT_CORRIDOR_KM
    parent_dx_m: float | None = None
    nz: int | None = None
    card: str | None = None
    vram_gib: float | None = None
    max_domains: int | None = None
    p_top_pa: float = DEFAULT_P_TOP_PA
    source: str | None = None
    source_priority: tuple[str, ...] | None = None
    leaf_history_interval_s: float = DEFAULT_LEAF_HISTORY_INTERVAL_S
    parent_history_interval_s: float | None = None
    leaf_output_preset: str = DEFAULT_LEAF_OUTPUT_PRESET
    output_root: str = DEFAULT_OUTPUT_ROOT
    gridded_template: str = DEFAULT_GRIDDED_TEMPLATE
    sites_template: str = DEFAULT_SITES_TEMPLATE

    def __post_init__(self):
        self.outdir = Path(self.outdir)
        if self.sites_path is None and self.partition_path is None:
            raise HindcastRefused("give --sites (one cluster) or --partition")
        if self.cluster_id is not None and self.partition_path is not None:
            raise HindcastRefused("--cluster-id names the single cluster of "
                                  "--sites; a partition names its own")
        if self.clusters is not None and self.partition_path is None:
            raise HindcastRefused("--clusters selects partition clusters; "
                                  "it needs --partition")
        if self.topology not in TOPOLOGIES:
            raise HindcastRefused(
                f"--topology {self.topology!r}: a hindcast drives "
                f"{', '.join(TOPOLOGIES)} (hex-swath is an experimental lane "
                "with no history window)")
        for name in ("segment_hours", "spinup_hours", "cycle_hour"):
            value = getattr(self, name)
            if isinstance(value, float) and value != int(value):
                raise HindcastRefused(f"--{name.replace('_', '-')} {value!r} "
                                      "must be a whole number of hours")
            setattr(self, name, int(value))
        if self.segment_hours <= 0:
            raise HindcastRefused("--segment-hours must be positive")
        if not 0 <= self.spinup_hours < self.segment_hours:
            raise HindcastRefused(
                f"--spinup-hours {self.spinup_hours} must be at least 0 and "
                f"less than --segment-hours {self.segment_hours}: something "
                "must be kept")
        if not 0 <= self.cycle_hour <= 23:
            raise HindcastRefused(f"--cycle-hour {self.cycle_hour} is not an "
                                  "hour of the day (0..23)")
        if self.start_has_hour and self.start.hour != self.cycle_hour:
            raise HindcastRefused(
                f"--start {_iso(self.start)} is at {self.start.hour:02d}Z but "
                f"--cycle-hour is {self.cycle_hour:02d}; give --start as a "
                "date or make the two agree (segments start on the cycle "
                "hour, never shifted silently)")
        if self.end <= self.start:
            raise HindcastRefused(f"--end {_iso(self.end)} must be after "
                                  f"--start {_iso(self.start)}")
        keep_s = self.keep_hours * 3600.0
        for name in ("leaf_history_interval_s", "parent_history_interval_s"):
            value = getattr(self, name)
            if value is None:
                continue
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                raise HindcastRefused(f"--{name.replace('_', '-')} must be "
                                      "positive seconds")
            if not _divides(value, keep_s):
                raise HindcastRefused(
                    f"--{name.replace('_', '-')} {value:g} does not divide "
                    f"the {self.keep_hours} h kept window, so frames would "
                    "not land on the segment boundaries")
            setattr(self, name, value)
        if (self.spinup_hours * 3600.0) and not _divides(
                self.leaf_history_interval_s, self.spinup_hours * 3600.0):
            raise HindcastRefused(
                f"--leaf-history-interval-s {self.leaf_history_interval_s:g} "
                f"does not divide the {self.spinup_hours} h spin-up, so the "
                "first kept frame would not fall on the kept window's start")
        if (self.topology == "wrf-tiles"
                and self.parent_history_interval_s is not None):
            raise HindcastRefused(
                "--parent-history-interval-s with --topology wrf-tiles: the "
                "parent's history is the tiles' boundary forcing, written at "
                "the planner's cadence from t = 0; it is not a hindcast knob")
        if self.leaf_output_preset not in LEAF_OUTPUT_PRESETS:
            raise HindcastRefused(
                f"--leaf-output-preset {self.leaf_output_preset!r} is not one "
                f"of {', '.join(LEAF_OUTPUT_PRESETS)}")
        if not math.isfinite(float(self.p_top_pa)) or self.p_top_pa <= 0:
            raise HindcastRefused("--p-top-pa must be positive pascals")
        self.p_top_pa = float(self.p_top_pa)
        for template, what in ((self.gridded_template, "--gridded-url"),
                               (self.sites_template, "--sites-url")):
            _check_template(template, what)
        if "{leaf}" not in self.gridded_template:
            raise HindcastRefused("--gridded-url needs {leaf}: every leaf "
                                  "domain gets its own store")

    @property
    def keep_hours(self) -> int:
        return self.segment_hours - self.spinup_hours

    def to_json(self) -> dict:
        record = {}
        for item in dataclasses.fields(self):
            value = getattr(self, item.name)
            if isinstance(value, datetime):
                value = _iso(value)
            elif isinstance(value, Path):
                value = str(value)
            elif isinstance(value, tuple):
                value = list(value)
            record[item.name] = value
        record.pop("outdir", None)
        return record


# --------------------------------------------------------------------------
# small helpers


def _iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _hours(delta: timedelta) -> float:
    return delta.total_seconds() / 3600.0


def _divides(step: float, total: float) -> bool:
    ratio = total / step
    return abs(ratio - round(ratio)) < 1e-9 * max(1.0, ratio)


def _utc_now() -> str:
    return _iso(datetime.now(timezone.utc).replace(tzinfo=None))


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", str(text)).strip("-")
    if not slug:
        raise HindcastRefused(f"cluster id {text!r} has no usable characters")
    return slug


def _canonical(document: Any) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, default=str).encode("utf-8")


def _json_bytes(document: Mapping[str, Any]) -> bytes:
    return (json.dumps(document, indent=1, default=str) + "\n").encode("utf-8")


def _write_bytes(path: Path, data: bytes) -> Path:
    """Write ``data`` to ``path`` through ``<name>.partial`` and rename."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".partial")
    with open(staging, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(staging, path)
    return path


def _write_json(path: Path, document: Mapping[str, Any]) -> Path:
    return _write_bytes(path, _json_bytes(document))


def _read_json(path: Path, what: str) -> dict:
    try:
        document = json.loads(Path(path).read_text())
    except OSError as error:
        raise HindcastRefused(f"cannot read {what} {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise HindcastRefused(f"{what} {path} is not JSON: {error}") from error
    if not isinstance(document, dict):
        raise HindcastRefused(f"{what} {path} is not a JSON object")
    return document


def _relpath(path: Path, start: Path) -> str:
    return Path(os.path.relpath(Path(path).resolve(),
                                Path(start).resolve())).as_posix()


def parse_utc(text: str, what: str) -> tuple[datetime, bool]:
    """``text`` as naive UTC on a whole hour, and whether it named an hour.

    Accepts ``YYYY-MM-DD`` and ``YYYY-MM-DDTHH[:MM[:SS]]`` with an optional
    ``Z`` or ``+00:00``.  Any other offset is refused: a hindcast is
    DST-agnostic UTC and a local time would shift every segment.
    """

    raw = str(text).strip()
    body = raw
    if body.endswith("Z"):
        body = body[:-1]
    elif re.search(r"[+-]\d\d:?\d\d$", body) and "T" in body:
        offset = body[-6:] if ":" in body[-6:] else body[-5:]
        if offset.replace(":", "") not in ("+0000", "-0000"):
            raise HindcastRefused(
                f"{what} {raw!r} carries the UTC offset {offset}; give UTC "
                "(a hindcast never converts local or daylight-saving time)")
        body = body[:-len(offset)]
    for form, has_hour in (("%Y-%m-%d", False), ("%Y-%m-%dT%H", True),
                           ("%Y-%m-%dT%H:%M", True),
                           ("%Y-%m-%dT%H:%M:%S", True)):
        try:
            moment = datetime.strptime(body, form)
        except ValueError:
            continue
        if moment.minute or moment.second:
            raise HindcastRefused(f"{what} {raw!r} must fall on a whole hour")
        return moment, has_hour
    raise HindcastRefused(f"{what} {raw!r} is not YYYY-MM-DD[THH] (UTC)")


def _check_template(template: str, what: str) -> None:
    try:
        names = {name for _, name, _, _ in string.Formatter().parse(template)
                 if name is not None}
    except ValueError as error:
        raise HindcastRefused(f"{what} {template!r}: {error}") from error
    unknown = names - _TEMPLATE_FIELDS
    if unknown:
        raise HindcastRefused(
            f"{what} {template!r} names {sorted(unknown)}; it may use "
            f"{', '.join('{' + n + '}' for n in sorted(_TEMPLATE_FIELDS))}")


def _render(template: str, **values: Any) -> str:
    """Fill the fields given; leave the others as ``{name}`` for later."""

    class _Keep(dict):
        def __missing__(self, key):
            return "{" + key + "}"

    return string.Formatter().vformat(template, (), _Keep(values))


# --------------------------------------------------------------------------
# segments


def segment_windows(start: datetime, end: datetime, *, segment_hours: int,
                    spinup_hours: int, cycle_hour: int) -> list[Segment]:
    """Every segment whose run starts in ``[start at cycle_hour, end)``."""

    keep = timedelta(hours=segment_hours - spinup_hours)
    if keep <= timedelta(0):
        raise HindcastRefused("segments keep nothing (spin-up >= length)")
    first = datetime(start.year, start.month, start.day, cycle_hour)
    if first < start:
        raise HindcastRefused(
            f"--start {_iso(start)} is after {cycle_hour:02d}Z of its day")
    segments = []
    run_start = first
    while run_start < end:
        segments.append(Segment(
            index=len(segments), run_start=run_start,
            run_end=run_start + timedelta(hours=segment_hours),
            keep_start=run_start + timedelta(hours=spinup_hours),
            keep_end=run_start + timedelta(hours=segment_hours)))
        run_start = run_start + keep
    if not segments:
        raise HindcastRefused(
            f"no segment starts in [{_iso(first)}, {_iso(end)})")
    return segments


# --------------------------------------------------------------------------
# clusters


def _bbox_region(sites: Sequence[Site]) -> dict:
    lons = [s.lon for s in sites]
    lats = [s.lat for s in sites]
    # The form woof.energy.source_select.region_bbox reads.
    return {"bbox": [min(lons), min(lats), max(lons), max(lats)]}


def _subset(site_set: SiteSet, site_ids: Sequence[str], *, cluster_id: str,
            source_ref: dict) -> SiteSet:
    by_id = {site.site_id: site for site in site_set.sites}
    missing = [s for s in site_ids if s not in by_id]
    if missing:
        raise HindcastRefused(
            f"cluster {cluster_id}: {len(missing)} member site id(s) are not "
            f"in the sites document, e.g. {missing[:5]}")
    wanted = set(site_ids)
    # The sites document's own order, so the subset bytes are stable.
    members = [site for site in site_set.sites if site.site_id in wanted]
    provenance = dict(site_set.provenance)
    provenance["hindcast_cluster"] = {"cluster_id": cluster_id,
                                      "from": source_ref}
    return SiteSet(sites=members, heights_m=site_set.heights_m,
                   spacing_m=site_set.spacing_m, assets_ref=None,
                   provenance=provenance)


def _partition_clusters(document: Mapping[str, Any], path: Path
                        ) -> list[Cluster]:
    if document.get("schema") != PARTITION_SCHEMA:
        raise HindcastRefused(f"{path} declares schema "
                              f"{document.get('schema')!r}; --partition "
                              f"reads {PARTITION_SCHEMA}")
    raw = document.get("clusters")
    if isinstance(raw, Mapping):
        raw = [dict(record, cluster_id=record.get("cluster_id", key))
               for key, record in raw.items()]
    if not isinstance(raw, list) or not raw:
        raise HindcastRefused(f"{path} has no clusters")
    clusters = []
    for index, record in enumerate(raw):
        if not isinstance(record, Mapping):
            raise HindcastRefused(f"{path}: cluster {index} is not an object")
        cluster_id = record.get("cluster_id", record.get("id"))
        site_ids = (record.get("site_ids") or record.get("member_site_ids")
                    or record.get("members"))
        if not cluster_id or not isinstance(site_ids, list) or not site_ids:
            raise HindcastRefused(
                f"{path}: cluster {index} needs a cluster_id and a non-empty "
                "site_ids list")
        dx = record.get("leaf_dx_m", record.get("dx_m"))
        chain = record.get("chain")
        topology = record.get("topology")
        if topology is not None and topology not in TOPOLOGIES:
            raise HindcastRefused(
                f"{path}: cluster {cluster_id} asks for topology "
                f"{topology!r}; a hindcast drives {', '.join(TOPOLOGIES)}")
        clusters.append(Cluster(
            cluster_id=str(cluster_id),
            site_ids=tuple(str(s) for s in site_ids),
            region=record.get("region"), origin="partition",
            leaf_dx_m=None if dx is None else float(dx),
            chain=None if chain is None else tuple(int(r) for r in chain),
            topology=topology))
    ids = [c.cluster_id for c in clusters]
    if len(set(ids)) != len(ids):
        raise HindcastRefused(f"{path}: cluster ids are not unique")
    return clusters


def _partition_sites_path(document: Mapping[str, Any], path: Path
                          ) -> tuple[Path | None, str | None]:
    ref = document.get("sites")
    if isinstance(ref, Mapping) and ref.get("path"):
        candidate = Path(ref["path"])
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        return candidate, ref.get("sha256")
    if isinstance(document.get("sites_path"), str):
        candidate = Path(document["sites_path"])
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        return candidate, None
    return None, None


def load_clusters(request: HindcastRequest
                  ) -> tuple[list[Cluster], SiteSet, Path, dict | None]:
    """The request's clusters, the sites document, its path, and the
    partition reference (None for a single sites set)."""

    partition_ref = None
    sites_path = request.sites_path
    if request.partition_path is not None:
        path = Path(request.partition_path)
        document = _read_json(path, "partition")
        clusters = _partition_clusters(document, path)
        named, recorded = _partition_sites_path(document, path)
        if sites_path is None:
            if named is None:
                raise HindcastRefused(f"{path} names no sites document; "
                                      "pass --sites")
            sites_path = named
        if recorded and Path(sites_path).is_file() \
                and sha256_file(sites_path) != recorded:
            raise HindcastRefused(
                f"sites document {sites_path} does not match the sha256 the "
                f"partition {path} recorded")
        if request.clusters is not None:
            wanted = set(request.clusters)
            unknown = wanted - {c.cluster_id for c in clusters}
            if unknown:
                raise HindcastRefused(
                    f"--clusters names {sorted(unknown)}, not in {path}")
            clusters = [c for c in clusters if c.cluster_id in wanted]
        partition_ref = {"path": str(path), "sha256": sha256_file(path)}
    sites_path = Path(sites_path)
    try:
        site_set = load_sites(sites_path)
    except OSError as error:
        raise HindcastRefused(f"cannot read sites {sites_path}: {error}"
                              ) from error
    if len(site_set) == 0:
        raise HindcastRefused(f"sites document {sites_path} has no sites")
    if request.partition_path is None:
        cluster_id = request.cluster_id or sites_path.stem
        clusters = [Cluster(
            cluster_id=cluster_id,
            site_ids=tuple(site.site_id for site in site_set.sites),
            region=_bbox_region(site_set.sites), origin="sites")]
    slugs = [_slug(c.cluster_id) for c in clusters]
    if len(set(slugs)) != len(slugs):
        raise HindcastRefused("two cluster ids reduce to the same directory "
                              "name; rename one")
    return clusters, site_set, sites_path, partition_ref


# --------------------------------------------------------------------------
# source selection


def fallback_select(region: Any, cycle: datetime, hours: float,
                    p_top_pa: float | None = None,
                    priority: Sequence[str] | None = None, *,
                    now: datetime | None = None) -> SourceChoice:
    """The first source in ``priority`` whose publication window serves
    ``[cycle, cycle + hours]`` (windows judged by
    :func:`woof.energy.source_window.window_refusal`; a source that module
    does not judge is accepted, its fetch is the gate)."""

    from woof.energy.source_window import window_refusal

    reasons = ["woof.energy.source_select is not in this build; the "
               "built-in fallback selector chose"]
    for name in priority or DEFAULT_FALLBACK_PRIORITY:
        refusal = window_refusal(name, cycle, hours, now=now)
        if refusal is None:
            reasons.append(f"{name}: publication window serves "
                           f"{_iso(cycle)} + {hours:g} h")
            return SourceChoice(source=name, reasons=tuple(reasons),
                                selector="fallback")
        reasons.append(f"{name} skipped: {refusal}")
    raise HindcastRefused("no source in the priority list serves "
                          f"{_iso(cycle)}: " + "; ".join(reasons[1:]))


def default_source_selector() -> tuple[Callable[..., Any], str]:
    """``woof.energy.source_select.select_source`` when importable, else
    :func:`fallback_select`; and the selector's name."""

    try:
        module = importlib.import_module("woof.energy.source_select")
    except ImportError:
        return fallback_select, "fallback"
    selector = getattr(module, "select_source", None)
    if selector is None:
        return fallback_select, "fallback"
    return selector, "woof.energy.source_select.select_source"


def _choose_source(selector: Callable[..., Any], name: str, *, region: Any,
                   cycle: datetime, hours: float, p_top_pa: float,
                   priority: Sequence[str] | None, now: datetime | None,
                   what: str) -> SourceChoice:
    try:
        choice = selector(region, cycle, hours, p_top_pa=p_top_pa,
                          priority=None if priority is None else
                          tuple(priority), now=now)
    except HindcastRefused:
        raise
    except Exception as error:  # the selector's refusal, named per segment
        raise HindcastRefused(f"{what}: the source selector ({name}) "
                              f"refused: {error}") from error
    source = choice if isinstance(choice, str) else getattr(choice, "source",
                                                            None)
    if not isinstance(source, str) or not source:
        raise HindcastRefused(f"{what}: the source selector ({name}) "
                              f"returned no source ({choice!r})")
    raw = getattr(choice, "reasons", None)
    if raw is None:
        raw = getattr(choice, "reason", None)
    if raw is None:
        reasons: tuple[str, ...] = ()
    elif isinstance(raw, str):
        reasons = (raw,)
    else:
        reasons = tuple(str(r) for r in raw)
    selector_name = getattr(choice, "selector", None) or name
    return SourceChoice(source=source, reasons=reasons,
                        selector=str(selector_name))


# --------------------------------------------------------------------------
# cost


def default_cost_estimate(job: Mapping[str, Any], plan: Plan | None = None
                          ) -> dict:
    """What a job costs, as far as the driver can say without a profiler.

    Before materialize: the run and kept hours.  After: the plan's domain
    count, horizontal cells per domain and the largest priced VRAM
    envelope the planner recorded.  ``gpu_hours`` stays None: a
    throughput model is a caller's to inject (``cost_estimator``)."""

    segment = job["segment"]
    record = {"basis": "segment", "run_hours": segment["run_hours"],
              "kept_hours": segment["keep_hours"],
              "spinup_fraction": round(segment["spinup_hours"]
                                       / segment["run_hours"], 6),
              "gpu_hours": None}
    if plan is None:
        return record
    cells = {}
    peak = None
    for domain in plan.domains:
        nx, ny = domain.extra.get("nx"), domain.extra.get("ny")
        if isinstance(nx, int) and isinstance(ny, int):
            cells[domain.domain_id] = nx * ny
        envelope = domain.extra.get("peak_envelope_gib")
        if isinstance(envelope, (int, float)):
            peak = envelope if peak is None else max(peak, envelope)
    record.update(basis="plan", domains=len(plan.domains),
                  leaves=sum(1 for d in plan.domains if d.role == "child"),
                  horizontal_cells=sum(cells.values()),
                  horizontal_cells_by_domain=cells,
                  peak_envelope_gib=peak)
    return record


# --------------------------------------------------------------------------
# plan


def _planner_kwargs(request: HindcastRequest, cluster: Cluster,
                    segment: Segment, source: str) -> tuple[str, dict]:
    topology = cluster.topology or request.topology
    dx = request.dx_m if request.dx_m is not None else (
        cluster.leaf_dx_m if cluster.leaf_dx_m is not None else DEFAULT_DX_M)
    if (request.dx_m is not None and cluster.leaf_dx_m is not None
            and not math.isclose(request.dx_m, cluster.leaf_dx_m)):
        raise HindcastRefused(
            f"cluster {cluster.cluster_id}: --dx-m {request.dx_m:g} "
            f"disagrees with the partition's leaf_dx_m "
            f"{cluster.leaf_dx_m:g}")
    parent_dx = request.parent_dx_m
    if parent_dx is None and cluster.chain:
        parent_dx = dx * math.prod(cluster.chain)
    kwargs = {"dx_m": float(dx), "corridor_km": float(request.corridor_km),
              "parent_dx_m": None if parent_dx is None else float(parent_dx),
              "start": segment.run_start.strftime("%Y-%m-%dT%H"),
              "hours": float(request.segment_hours), "source": source,
              "card": request.card, "vram_gib": request.vram_gib,
              "max_domains": request.max_domains, "nz": request.nz,
              # Every domain carries the wind-energy column at this top
              # (woof.energy.vertical.wind_energy_column: 12 m lowest layer,
              # at least 8 layers below 250 m, refused above 20 km).
              "column_request": {"p_top_pa": float(request.p_top_pa)}}
    return topology, kwargs


def _cluster_sites_bytes(site_set: SiteSet) -> bytes:
    # The bytes woof.energy.contracts._write_json would write.
    return (json.dumps(site_set.to_json(), indent=1, sort_keys=False)
            + "\n").encode("utf-8")


def _build_jobs(request: HindcastRequest, clusters: list[Cluster],
                site_set: SiteSet, sites_path: Path,
                partition_ref: dict | None, *, selector: Callable[..., Any],
                selector_name: str, cost_estimator: Callable[..., dict],
                now: datetime | None
                ) -> tuple[list[dict], dict[str, bytes], dict]:
    """Job documents, cluster site files (relative path -> bytes), and the
    per-cluster summary."""

    outdir = request.outdir
    segments = segment_windows(request.start, request.end,
                               segment_hours=request.segment_hours,
                               spinup_hours=request.spinup_hours,
                               cycle_hour=request.cycle_hour)
    # Name and digest only: how the path was spelled must not reach the
    # cluster bytes, and so the job ids.
    sites_ref = {"name": Path(sites_path).name,
                 "sha256": sha256_file(sites_path)}
    jobs: list[dict] = []
    files: dict[str, bytes] = {}
    summary: dict[str, dict] = {}
    for cluster in clusters:
        slug = _slug(cluster.cluster_id)
        subset = _subset(site_set, cluster.site_ids,
                         cluster_id=cluster.cluster_id, source_ref=sites_ref)
        data = _cluster_sites_bytes(subset)
        content_sha = hashlib.sha256(data).hexdigest()
        # Content-addressed, so a re-plan from edited sites never rewrites
        # the file an older job is bound to.
        cluster_sites_rel = (f"{CLUSTERS_DIR}/{slug}/"
                             f"sites-{content_sha[:12]}.json")
        files[cluster_sites_rel] = data
        region = (cluster.region if cluster.region is not None
                  else _bbox_region(subset.sites))
        leaf_hub_heights = _leaf_hub_heights(request.leaf_output_preset,
                                             subset, cluster.cluster_id)
        previous: str | None = None
        changes = []
        for segment in segments:
            what = (f"cluster {cluster.cluster_id} segment "
                    f"{_iso(segment.run_start)}")
            if request.source is not None:
                choice = SourceChoice(source=request.source,
                                      reasons=("fixed by --source",),
                                      selector="fixed")
            else:
                choice = _choose_source(
                    selector, selector_name, region=region,
                    cycle=segment.run_start,
                    hours=float(request.segment_hours),
                    p_top_pa=request.p_top_pa,
                    priority=request.source_priority, now=now, what=what)
            topology, kwargs = _planner_kwargs(request, cluster, segment,
                                               choice.source)
            parent_interval = request.parent_history_interval_s
            if parent_interval is None and topology == "wrf-nests":
                parent_interval = DEFAULT_NESTS_PARENT_HISTORY_INTERVAL_S
            if topology == "wrf-tiles" and \
                    request.parent_history_interval_s is not None:
                raise HindcastRefused(
                    f"cluster {cluster.cluster_id} plans wrf-tiles, whose "
                    "parent history is the tiles' forcing; drop "
                    "--parent-history-interval-s")
            gridded = _render(request.gridded_template,
                              root=request.output_root,
                              cluster_id=slug, topology=topology,
                              dx_m=f"{kwargs['dx_m']:g}")
            sites_url = _render(request.sites_template,
                                root=request.output_root, cluster_id=slug,
                                topology=topology,
                                dx_m=f"{kwargs['dx_m']:g}")
            history = {
                "history_begin_s": float(request.spinup_hours * 3600),
                "leaf_interval_s": request.leaf_history_interval_s,
                "parent_interval_s": parent_interval,
                "leaf_output_preset": request.leaf_output_preset,
                "leaf_hub_heights_m": leaf_hub_heights,
                "applies": (
                    "history_begin_s on every [[domain]]" if topology ==
                    "wrf-nests" else
                    "parent keeps every frame from t = 0 (tile forcing); "
                    "leaf tiles write the spin-up too and the kept window "
                    "is applied at extraction")}
            outputs = {
                "root": request.output_root,
                "root_relative_to": (None if "://" in request.output_root
                                     else "hindcast_dir"),
                "gridded_template": gridded,
                "sites_url": sites_url,
                "time_window": [_iso(segment.keep_start),
                                _iso(segment.keep_end)],
                "time_window_closed": "left"}
            identity = {
                "schema": JOB_SCHEMA,
                "cluster": {"cluster_id": cluster.cluster_id,
                            "sites_content_sha256": content_sha,
                            "leaf_dx_m": cluster.leaf_dx_m,
                            "chain": (None if cluster.chain is None
                                      else list(cluster.chain))},
                "segment": {"run_start": _iso(segment.run_start),
                            "run_hours": request.segment_hours,
                            "spinup_hours": request.spinup_hours},
                "source": choice.source,
                "planner": {"topology": topology, "kwargs": kwargs,
                            "p_top_pa": request.p_top_pa},
                "history": {k: v for k, v in history.items()
                            if k != "applies"},
                "outputs": {"gridded_template": gridded,
                            "sites_url": sites_url,
                            "root": request.output_root}}
            digest = hashlib.sha256(_canonical(identity)).hexdigest()
            job_id = f"{slug}-{segment.run_start:%Y%m%dT%H}Z-{digest[:12]}"
            job_dir_rel = f"{JOBS_DIR}/{job_id}"
            changed = previous is not None and previous != choice.source
            if changed:
                changes.append({"job_id": job_id,
                                "cycle": _iso(segment.run_start),
                                "from": previous, "to": choice.source})
            job = {
                "schema": JOB_SCHEMA,
                "job_id": job_id,
                "identity_sha256": digest,
                "cluster": {
                    "cluster_id": cluster.cluster_id,
                    "origin": cluster.origin,
                    "site_count": len(subset),
                    "sites": {"path": f"../../{cluster_sites_rel}",
                              "sha256": content_sha},
                    "partition": (None if partition_ref is None else {
                        "path": _relpath(Path(partition_ref["path"]),
                                         outdir / job_dir_rel),
                        "sha256": partition_ref["sha256"]}),
                    "region": region,
                    "leaf_dx_m": cluster.leaf_dx_m,
                    "chain": (None if cluster.chain is None
                              else list(cluster.chain))},
                "segment": segment.to_json(request.spinup_hours),
                "source": {"name": choice.source,
                           "selector": choice.selector,
                           "reasons": list(choice.reasons),
                           "previous_segment_source": previous,
                           "changed_from_previous_segment": changed},
                "planner": {"topology": topology,
                            "module": _PLANNER_MODULES[topology],
                            "kwargs": kwargs,
                            "p_top_pa": request.p_top_pa},
                "history": history,
                "outputs": outputs,
                "resources": {"gpus": 1, "card": request.card,
                              "vram_gib": request.vram_gib},
                "paths": {"hindcast_dir": "../..", "plan": f"{PLAN_DIR}/plan.json",
                          "materialized": MATERIALIZED_NAME,
                          "sites_forecast": SITES_FORECAST_NAME},
            }
            job["cost"] = cost_estimator(job, None)
            jobs.append(job)
            previous = choice.source
        summary[cluster.cluster_id] = {
            "slug": slug, "origin": cluster.origin,
            "sites": cluster_sites_rel, "site_count": len(subset),
            "sites_content_sha256": content_sha, "region": region,
            "source_changes": changes}
    return jobs, files, summary


def _manifest_record(job: Mapping[str, Any]) -> dict:
    return {"cluster_id": job["cluster"]["cluster_id"],
            "cycle": job["segment"]["cycle"],
            "keep_start": job["segment"]["keep_start"],
            "keep_end": job["segment"]["keep_end"],
            "source": job["source"]["name"],
            "source_changed": job["source"]["changed_from_previous_segment"],
            "topology": job["planner"]["topology"],
            "job": f"{JOBS_DIR}/{job['job_id']}/{JOB_NAME}",
            "status": "pending", "attempts": 0, "start_utc": None,
            "end_utc": None, "reason": None, "notes": []}


@contextlib.contextmanager
def _manifest_lock(hindcast_dir: Path):
    """An advisory exclusive lock for a read-modify-write of the manifest."""

    hindcast_dir.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError:  # no flock on this platform: single writer assumed
        yield
        return
    with open(hindcast_dir / LOCK_NAME, "a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_manifest(hindcast_dir: Path) -> dict | None:
    path = Path(hindcast_dir) / MANIFEST_NAME
    if not path.exists():
        return None
    document = _read_json(path, "hindcast manifest")
    if document.get("schema") != MANIFEST_SCHEMA:
        raise HindcastRefused(f"{path} is not a {MANIFEST_SCHEMA} document; "
                              "move it aside")
    if not isinstance(document.get("jobs"), dict):
        raise HindcastRefused(f"{path} has no jobs table; move it aside")
    return document


def _write_manifest(hindcast_dir: Path, manifest: dict) -> Path:
    manifest["updated_utc"] = _utc_now()
    order = sorted(manifest["jobs"],
                   key=lambda j: (manifest["jobs"][j]["cluster_id"],
                                  manifest["jobs"][j]["cycle"]))
    manifest["order"] = order
    counts: dict[str, int] = {}
    for record in manifest["jobs"].values():
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    manifest["counts"] = counts
    return _write_json(Path(hindcast_dir) / MANIFEST_NAME, manifest)


def update_job(hindcast_dir: Path | None, job_id: str, **changes: Any
               ) -> dict | None:
    """Update one job's manifest record under the lock; None without a
    manifest (a job document carried away from its hindcast directory)."""

    if hindcast_dir is None:
        return None
    hindcast_dir = Path(hindcast_dir)
    if not (hindcast_dir / MANIFEST_NAME).exists():
        return None
    with _manifest_lock(hindcast_dir):
        manifest = read_manifest(hindcast_dir)
        record = manifest["jobs"].get(job_id)
        if record is None:
            raise HindcastRefused(f"job {job_id} is not in "
                                  f"{hindcast_dir / MANIFEST_NAME}")
        notes = changes.pop("add_notes", None)
        if "status" in changes and changes["status"] not in JOB_STATUSES:
            raise HindcastRefused(f"status {changes['status']!r} is not one "
                                  f"of {JOB_STATUSES}")
        record.update(changes)
        if notes:
            record["notes"] = list(record.get("notes") or []) + list(notes)
        _write_manifest(hindcast_dir, manifest)
        return dict(record)


def plan_hindcast(request: HindcastRequest, *, dry_run: bool = False,
                  resume: bool = False,
                  source_selector: Callable[..., Any] | None = None,
                  cost_estimator: Callable[..., dict] | None = None,
                  now: datetime | None = None) -> dict:
    """Expand the request into jobs and write them with the manifest.

    Re-running an identical request is a no-op that keeps every job's
    status.  A different request into a directory that already holds a
    manifest is refused unless ``resume``: then new jobs are added, jobs
    with the same id keep their records, and a job that a new request
    would plan differently for the same cluster and cycle replaces the old
    one unless the old one is complete (kept, and noted).  Job directories
    are never deleted.  ``dry_run`` writes nothing and returns the
    manifest that would be written with every job document.
    """

    clusters, site_set, sites_path, partition_ref = load_clusters(request)
    if source_selector is not None:
        selector, selector_name = source_selector, getattr(
            source_selector, "__qualname__", "injected")
    else:
        selector, selector_name = default_source_selector()
    jobs, files, cluster_summary = _build_jobs(
        request, clusters, site_set, sites_path, partition_ref,
        selector=selector, selector_name=selector_name,
        cost_estimator=cost_estimator or default_cost_estimate, now=now)
    outdir = request.outdir
    request_record = request.to_json()
    request_record["sites_path"] = str(Path(sites_path).resolve())
    # --start 2025-11-01 and --start 2025-11-01T00 are the same request.
    request_record.pop("start_has_hour", None)
    if request.partition_path is not None:
        request_record["partition_path"] = str(
            Path(request.partition_path).resolve())
    request_sha = hashlib.sha256(_canonical(request_record)).hexdigest()
    notes: list[str] = []
    if request.source is None and selector is fallback_select:
        notes.append("woof.energy.source_select is not in this build; "
                     "sources came from the built-in fallback selector "
                     f"({', '.join(request.source_priority or DEFAULT_FALLBACK_PRIORITY)})")
    for cluster_id, summary in cluster_summary.items():
        for change in summary["source_changes"]:
            notes.append(f"cluster {cluster_id}: source changes "
                         f"{change['from']} -> {change['to']} at "
                         f"{change['cycle']} ({change['job_id']})")
    fresh = {job["job_id"]: _manifest_record(job) for job in jobs}

    if dry_run:
        manifest = _new_manifest(request_record, request_sha,
                                 cluster_summary, fresh, notes)
        manifest["counts"] = {"pending": len(fresh)}
        manifest["order"] = [job["job_id"] for job in jobs]
        return {"dry_run": True, "hindcast_dir": str(outdir),
                "manifest": manifest, "jobs": jobs}

    with _manifest_lock(outdir):
        previous = read_manifest(outdir)
        if previous is not None and not resume and \
                previous.get("request_sha256") != request_sha:
            raise HindcastRefused(
                f"{outdir / MANIFEST_NAME} was planned from a different "
                "request; pass --resume to merge this one into it (statuses "
                "kept), or plan into a new directory")
        for relative, data in files.items():
            target = outdir / relative
            if not target.exists() or target.read_bytes() != data:
                _write_bytes(target, data)
        if previous is None:
            manifest = _new_manifest(request_record, request_sha,
                                     cluster_summary, fresh, notes)
        else:
            manifest = _merge(previous, request_record, request_sha,
                              cluster_summary, fresh, notes)
        # Only jobs the manifest adopted get a job.json: a planned job a
        # complete one was kept in place of is not written.
        adopted = [job for job in jobs if job["job_id"] in manifest["jobs"]]
        for job in adopted:
            # A job id binds its identity: an existing job.json is the one
            # its runs used (and holds the source history it was planned
            # with), so it is never rewritten.
            target = outdir / JOBS_DIR / job["job_id"] / JOB_NAME
            if not target.exists():
                _write_json(target, job)
        _write_manifest(outdir, manifest)
    manifest["manifest_path"] = str(outdir / MANIFEST_NAME)
    manifest["planned_job_ids"] = [job["job_id"] for job in adopted]
    manifest["not_adopted_job_ids"] = [job["job_id"] for job in jobs
                                       if job["job_id"] not in manifest["jobs"]]
    return manifest


def _new_manifest(request_record: dict, request_sha: str,
                  clusters: dict, jobs: dict, notes: list[str]) -> dict:
    return {"schema": MANIFEST_SCHEMA, "job_schema": JOB_SCHEMA,
            "created_utc": _utc_now(), "updated_utc": None,
            "request": request_record, "request_sha256": request_sha,
            "requests": [{"sha256": request_sha, "planned_utc": _utc_now()}],
            "clusters": clusters, "jobs": jobs, "notes": list(notes)}


def _merge(previous: dict, request_record: dict, request_sha: str,
           clusters: dict, fresh: dict, notes: list[str]) -> dict:
    manifest = dict(previous)
    jobs = dict(previous["jobs"])
    by_key = {(r["cluster_id"], r["cycle"]): job_id
              for job_id, r in jobs.items()}
    merge_notes = []
    for job_id, record in fresh.items():
        if job_id in jobs:
            continue
        key = (record["cluster_id"], record["cycle"])
        old_id = by_key.get(key)
        if old_id is not None:
            old = jobs[old_id]
            if old["status"] in ("complete", "running"):
                merge_notes.append(
                    f"{old_id} is {old['status']}; kept although the new "
                    f"request plans {job_id} for {key[0]} {key[1]}")
                continue
            merge_notes.append(f"{old_id} ({old['status']}) superseded by "
                               f"{job_id}; its directory is left on disk")
            del jobs[old_id]
        jobs[job_id] = record
        by_key[key] = job_id
    manifest["jobs"] = jobs
    manifest["clusters"] = {**previous.get("clusters", {}), **clusters}
    if previous.get("request_sha256") != request_sha:
        manifest["request"] = request_record
        manifest["request_sha256"] = request_sha
        manifest["requests"] = list(previous.get("requests") or []) + [
            {"sha256": request_sha, "planned_utc": _utc_now()}]
    seen = set(previous.get("notes") or [])
    manifest["notes"] = list(previous.get("notes") or []) + [
        n for n in notes + merge_notes if n not in seen]
    return manifest


# --------------------------------------------------------------------------
# jobs on disk


def locate_job(job: str | Path, *, hindcast_dir: Path | None = None
               ) -> Path:
    """``job.json`` from a path to it, a job directory, or a job id with
    ``hindcast_dir``."""

    candidate = Path(job)
    if candidate.is_file():
        return candidate
    if candidate.is_dir() and (candidate / JOB_NAME).is_file():
        return candidate / JOB_NAME
    if hindcast_dir is not None:
        path = Path(hindcast_dir) / JOBS_DIR / str(job) / JOB_NAME
        if path.is_file():
            return path
    raise HindcastRefused(
        f"no hindcast job {job!r}: give a job.json, a job directory, or a "
        "job id with --dir HINDCAST_DIR")


def load_job(path: Path) -> dict:
    document = _read_json(path, "hindcast job")
    if document.get("schema") != JOB_SCHEMA:
        raise HindcastRefused(f"{path} declares schema "
                              f"{document.get('schema')!r}; this woof reads "
                              f"{JOB_SCHEMA}")
    for key in ("job_id", "cluster", "segment", "source", "planner",
                "history", "outputs", "paths"):
        if key not in document:
            raise HindcastRefused(f"{path} lacks {key!r}")
    return document


def _hindcast_dir_of(job_path: Path, job: Mapping[str, Any]) -> Path | None:
    relative = job["paths"].get("hindcast_dir")
    if not relative:
        return None
    candidate = (job_path.parent / relative).resolve()
    return candidate if (candidate / MANIFEST_NAME).exists() else None


# --------------------------------------------------------------------------
# materialize


def _filter_kwargs(function: Callable[..., Any], kwargs: dict) -> dict:
    """The kwargs ``function`` accepts (all of them under ``**kwargs``)."""

    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind is inspect.Parameter.VAR_KEYWORD
           for p in parameters.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in parameters}


def _plan_path_from(result: Any, plan_dir: Path) -> Path:
    if isinstance(result, Plan) and result.path is not None:
        return Path(result.path)
    if isinstance(result, (str, Path)):
        return Path(result)
    if isinstance(result, Mapping) and result.get("plan"):
        return Path(result["plan"])
    return plan_dir / "plan.json"


def default_cluster_planner(job: Mapping[str, Any], plan_dir: Path, *,
                            job_dir: Path,
                            now: datetime | None = None) -> Path:
    """Plan the job's cluster for its segment into ``plan_dir``.

    A partition cluster goes through
    ``woof.energy.partition.plan_cluster(partition, cluster_id, **kwargs)``
    when that module is in this build (kwargs it does not take are not
    passed).  Otherwise the cluster's sites file (sha256-checked) is planned by the
    topology's own ``build_plan``.  Returns the written plan's path."""

    kwargs = dict(job["planner"]["kwargs"])
    topology = job["planner"]["topology"]
    sites_path = (job_dir / job["cluster"]["sites"]["path"]).resolve()
    recorded = job["cluster"]["sites"].get("sha256")
    if not sites_path.is_file():
        raise HindcastRefused(f"the job's cluster sites {sites_path} are "
                              "missing")
    if recorded and sha256_file(sites_path) != recorded:
        raise HindcastRefused(
            f"the job's cluster sites {sites_path} no longer have the "
            f"sha256 it was planned with ({recorded[:12]}...); re-plan")
    if job["cluster"]["origin"] == "partition" and job["cluster"]["partition"]:
        try:
            partition_mod = importlib.import_module("woof.energy.partition")
        except ImportError:
            partition_mod = None
        plan_cluster = getattr(partition_mod, "plan_cluster", None)
        if plan_cluster is not None:
            partition_path = (job_dir
                              / job["cluster"]["partition"]["path"]).resolve()
            loader = getattr(partition_mod, "load_partition", None)
            partition = (loader(partition_path) if loader is not None
                         else _read_json(partition_path, "partition"))
            offered = {"outdir": plan_dir, "topology": topology,
                       "sites_path": sites_path, "now": now,
                       **{k: v for k, v in kwargs.items() if v is not None}}
            result = plan_cluster(partition, job["cluster"]["cluster_id"],
                                  **_filter_kwargs(plan_cluster, offered))
            return _plan_path_from(result, plan_dir)
    module = importlib.import_module(_PLANNER_MODULES[topology])
    sites = load_sites(sites_path)
    extra: dict[str, Any] = {"now": now}
    if topology == "wrf-nests":
        extra["sites_path"] = sites_path
        extra["name"] = f"{_slug(job['cluster']['cluster_id'])}-" \
                        f"{kwargs['start'].replace('-', '')}"
    plan = module.build_plan(sites, outdir=plan_dir, **kwargs, **extra)
    path = Path(plan.path) if plan.path is not None else plan_dir / "plan.json"
    if topology == "wrf-tiles":
        # The planner's CLI binds the sites document; build_plan does not.
        plan = load_plan(path)
        plan.sites_ref = {"path": _relpath(sites_path, path.parent),
                          "sha256": sha256_file(sites_path),
                          **(plan.sites_ref or {})}
        dump_plan(plan, path)
    return path


def _domain_blocks(text: str) -> list[tuple[int, int]]:
    """``[start, end)`` line ranges of every top-level ``[[domain]]``."""

    lines = text.split("\n")
    blocks, begin = [], None
    for index, line in enumerate(lines):
        if not _TABLE_HEADER.match(line):
            continue
        if begin is not None:
            blocks.append((begin, index))
            begin = None
        if _DOMAIN_HEADER.match(line):
            begin = index
    if begin is not None:
        blocks.append((begin, len(lines)))
    return blocks


#: A TOML table header (``[name]`` / ``[[name]]``, optional comment); an
#: array continuation line such as ``[1.0, 2.0],`` is not one.
_TABLE_HEADER = re.compile(
    r'^\s*\[\[?\s*[A-Za-z_"][\w."\- ]*\]\]?\s*(#.*)?$')
_DOMAIN_HEADER = re.compile(r"^\s*\[\[\s*domain\s*\]\]\s*(#.*)?$")
_RUN_HEADER = re.compile(r"^\s*\[\s*run\s*\]\s*(#.*)?$")


def _key_pattern(key: str) -> re.Pattern:
    return re.compile(rf"^\s*{re.escape(key)}\s*=")


class _TomlLiteral(str):
    """A value :func:`patch_domain_tables` writes verbatim (an inline table)."""


def _leaf_hub_heights(preset: str, site_set: SiteSet, cluster_id: str
                      ) -> list[float] | None:
    """The hub heights a ``windfarm`` leaf computes: the cluster's site
    heights, so the forecast ``woof energy extract`` builds from the hub
    planes lands on the heights the site store was made with.  ``None``
    for a preset that keeps its volumes (those leaves take hub_wind's
    defaults).  Refused, never trimmed, outside the diagnostic's limits."""

    if preset != "windfarm":
        return None
    from woof.io.history_selection import MAX_HUB_HEIGHT_M, MAX_HUB_HEIGHTS

    heights = sorted({float(h) for h in site_set.heights_m})
    bad = [h for h in heights if not 0.0 < h <= MAX_HUB_HEIGHT_M]
    if not heights or bad or len(heights) > MAX_HUB_HEIGHTS:
        raise HindcastRefused(
            f"cluster {cluster_id}: --leaf-output-preset windfarm computes "
            f"the hub-height planes at the sites document's heights_m "
            f"{heights}, which must be 1 to {MAX_HUB_HEIGHTS} heights in "
            f"(0, {MAX_HUB_HEIGHT_M:g}] m; rebuild the sites with a coarser "
            "hub_height_step_m, or use --leaf-output-preset energy")
    return heights


def _leaf_output_items(history: Mapping[str, Any]) -> list[str]:
    """``key = value`` items of a leaf's ``[output]``, from the job's
    recorded ``history`` (module docstring, *What a leaf writes*)."""

    preset = history.get("leaf_output_preset", LEGACY_LEAF_OUTPUT_PRESET)
    if preset not in LEAF_OUTPUT_PRESETS:
        raise HindcastRefused(f"the job records leaf_output_preset "
                              f"{preset!r}, not one of "
                              f"{', '.join(LEAF_OUTPUT_PRESETS)}")
    if preset != "windfarm":
        return [f'preset = "{preset}"', "hub_wind = true"]
    heights = history.get("leaf_hub_heights_m")
    if not heights:
        raise HindcastRefused("the job's windfarm leaves record no "
                              "leaf_hub_heights_m; plan it again")
    listed = ", ".join(f"{float(h)!r}" for h in heights)
    return ['preset = "windfarm"', f"hub_heights_m = [{listed}]"]


def leaf_output(history: Mapping[str, Any]) -> _TomlLiteral:
    """The ``output = {...}`` inline table a ``wrf-nests`` leaf gets."""

    return _TomlLiteral("{ " + ", ".join(_leaf_output_items(history)) + " }")


def _leaf_output_table(history: Mapping[str, Any]) -> list[str]:
    """The same selection as a ``wrf-tiles`` leaf's ``[output]`` table."""

    return ["[output]", *_leaf_output_items(history)]


def patch_domain_tables(text: str, settings: Mapping[int, Mapping[str, Any]]
                        ) -> str:
    """Set keys on ``[[domain]]`` tables by ``grid_id`` (others untouched).

    A key already present has its line replaced; a missing key is written
    after ``history_interval_s`` (or after ``grid_id``)."""

    lines = text.split("\n")
    found: set[int] = set()
    for begin, end in reversed(_domain_blocks(text)):
        block = lines[begin:end]
        grid_id = None
        for line in block:
            match = re.match(r"^\s*grid_id\s*=\s*(\d+)", line)
            if match:
                grid_id = int(match.group(1))
                break
        if grid_id is None or grid_id not in settings:
            continue
        found.add(grid_id)
        for key, value in settings[grid_id].items():
            rendered = (f"{key} = {value}" if isinstance(value, _TomlLiteral)
                        else f"{key} = {float(value)!r}")
            pattern = _key_pattern(key)
            hits = [i for i, line in enumerate(block) if pattern.match(line)]
            if hits:
                block[hits[0]] = rendered
                for extra in reversed(hits[1:]):
                    del block[extra]
                continue
            anchor = next((i for i, line in enumerate(block)
                           if _key_pattern("history_interval_s").match(line)),
                          None)
            if anchor is None:
                anchor = next(i for i, line in enumerate(block)
                              if _key_pattern("grid_id").match(line))
            block.insert(anchor + 1, rendered)
        lines[begin:end] = block
    missing = set(settings) - found
    if missing:
        raise HindcastRefused(f"no [[domain]] table with grid_id "
                              f"{sorted(missing)} to set history on")
    return "\n".join(lines)


def _replace_atomically_after(path: Path, text: str,
                              check: Callable[[Path], None]) -> None:
    """Write ``text`` beside ``path``, ``check`` it, then rename over."""

    staging = path.with_name(path.stem + ".hindcast-partial" + path.suffix)
    staging.write_text(text, encoding="utf-8", newline="\n")
    try:
        check(staging)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    os.replace(staging, path)


def _apply_nests_history(plan: Plan, job: Mapping[str, Any]) -> list[str]:
    from woof.experiment import load_experiment

    history = job["history"]
    begin = float(history["history_begin_s"])
    leaf_s = float(history["leaf_interval_s"])
    parent_s = float(history["parent_interval_s"])
    configs = sorted({d.config for d in plan.domains})
    parents = {d.parent for d in plan.domains if d.parent is not None}
    notes = []
    for config in configs:
        members = [d for d in plan.domains if d.config == config]
        settings = {}
        for domain in members:
            if domain.grid_id is None:
                raise HindcastRefused(f"{domain.domain_id} has no grid_id")
            values = {"history_interval_s":
                      leaf_s if domain.role == "child" else parent_s}
            if begin:
                values["history_begin_s"] = begin
            if domain.role == "child" and domain.domain_id not in parents:
                values["output"] = leaf_output(history)
            settings[int(domain.grid_id)] = values
        path = plan.resolve(config)
        text = patch_domain_tables(path.read_text(encoding="utf-8"),
                                   settings)
        text = _enable_wind_diagnostics(text)

        def check(candidate: Path, settings=settings) -> None:
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    experiment = load_experiment(candidate)
            except (ValueError, TypeError, KeyError) as error:
                raise HindcastRefused(
                    f"the segment history settings make {config} invalid: "
                    f"{error}") from error
            got = {int(dc.grid_id): dc for dc in experiment.domains}
            for grid_id, values in settings.items():
                for key, want in values.items():
                    if isinstance(want, _TomlLiteral):
                        continue  # the loader above accepted it
                    have = float(getattr(got[grid_id], key))
                    if not math.isclose(have, want):
                        raise HindcastRefused(
                            f"{config} d{grid_id:02d}: {key} = {want:g} was "
                            f"written but the loader reads {have:g}")
            if not math.isclose(float(experiment.run_seconds),
                                job["segment"]["run_hours"] * 3600.0):
                raise HindcastRefused(
                    f"{config} runs {experiment.run_seconds:g} s, not the "
                    f"segment's {job['segment']['run_hours']:g} h")

        _replace_atomically_after(path, text, check)
        notes.append(f"{config}: history_begin_s = {begin:g} on every "
                     f"domain, leaves every {leaf_s:g} s, parents every "
                     f"{parent_s:g} s, leaves output = "
                     f"{leaf_output(history)}, nwp_diagnostics = 1 "
                     "(woof energy hindcast)")
    return notes


def _replace_strings(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return new if value == old else value
    if isinstance(value, list):
        return [_replace_strings(v, old, new) for v in value]
    if isinstance(value, tuple):
        return tuple(_replace_strings(v, old, new) for v in value)
    if isinstance(value, dict):
        return {k: _replace_strings(v, old, new) for k, v in value.items()}
    return value


def _set_run_key(text: str, key: str, value: float) -> str:
    """Set ``key`` inside the ``[run]`` table of a legacy child config."""

    lines = text.split("\n")
    in_run, hits = False, []
    for index, line in enumerate(lines):
        if _TABLE_HEADER.match(line):
            in_run = bool(_RUN_HEADER.match(line))
            continue
        if in_run and _key_pattern(key).match(line):
            hits.append(index)
    if len(hits) != 1:
        raise HindcastRefused(f"the child config's [run] table has "
                              f"{len(hits)} {key} lines; expected one")
    lines[hits[0]] = f"{key} = {float(value)!r}"
    return "\n".join(lines)


def _ensure_run_key(text: str, key: str, value: int) -> str:
    """Set ``key`` in the ``[run]`` table of a legacy child config, adding
    the line when the table does not carry it yet."""

    lines = text.split("\n")
    in_run, header, hits = False, None, []
    for index, line in enumerate(lines):
        if _TABLE_HEADER.match(line):
            in_run = bool(_RUN_HEADER.match(line))
            if in_run:
                header = index
            continue
        if in_run and _key_pattern(key).match(line):
            hits.append(index)
    if len(hits) > 1:
        raise HindcastRefused(f"the child config's [run] table has "
                              f"{len(hits)} {key} lines; expected one")
    if hits:
        lines[hits[0]] = f"{key} = {int(value)}"
    elif header is not None:
        lines.insert(header + 1, f"{key} = {int(value)}")
    else:
        raise HindcastRefused("the child config has no [run] table")
    return "\n".join(lines)


def _enable_wind_diagnostics(text: str) -> str:
    """Switch on the in-model wind diagnostics in an experiment TOML:
    ``[shared] nwp_diagnostics = 1`` (WSPD10MAX, U10MAX/V10MAX,
    W_UP_MAX/W_DN_MAX, UPP GUST; woof.core.wind_gust_diag) and
    ``[output] hub_wind = true`` (hub-height wind and its interval
    MAX/MIN/MEAN; woof.core.hub_wind_diag) unless the file already names
    its hub heights.  The output presets only select these fields; the
    run configuration is what computes them."""

    lines = text.split("\n")
    table = None
    shared_at = output_at = None
    nwp_at, hub_named = None, False
    for index, line in enumerate(lines):
        if _TABLE_HEADER.match(line):
            table = line.strip()
            if table == "[shared]":
                shared_at = index
            elif table == "[output]":
                output_at = index
            continue
        if table == "[shared]" and _key_pattern("nwp_diagnostics").match(line):
            nwp_at = index
        if table == "[output]" and (_key_pattern("hub_wind").match(line)
                                     or _key_pattern("hub_heights_m").match(line)):
            hub_named = True
    if nwp_at is not None:
        lines[nwp_at] = "nwp_diagnostics = 1"
    elif shared_at is not None:
        lines.insert(shared_at + 1, "nwp_diagnostics = 1")
        if output_at is not None and output_at > shared_at:
            output_at += 1
    else:
        while lines and lines[-1] == "":
            lines.pop()
        lines += ["", "[shared]", "nwp_diagnostics = 1", ""]
    if not hub_named:
        if output_at is not None:
            lines.insert(output_at + 1, "hub_wind = true")
        else:
            # A table header after the last [[domain]] closes that array
            # entry, so a new [output] table is valid at the end.
            while lines and lines[-1] == "":
                lines.pop()
            lines += ["", "[output]", "hub_wind = true", ""]
    return "\n".join(lines)


def _set_output_table(text: str, lines: Sequence[str]) -> str:
    """Replace the ``[output]`` table of a child config with ``lines`` (or
    append it): the leaf tile's whole selection, never merged key by key."""

    rows = text.split("\n")
    start = next((i for i, line in enumerate(rows)
                  if re.match(r"^\s*\[\s*output\s*\]\s*(#.*)?$", line)), None)
    if start is None:
        while rows and rows[-1] == "":
            rows.pop()
        return "\n".join(rows + ["", *lines, ""])
    end = next((i for i in range(start + 1, len(rows))
                if _TABLE_HEADER.match(rows[i])), len(rows))
    tail = rows[end:]
    return "\n".join(rows[:start] + list(lines) + ([""] + tail if tail else [""]))


def _apply_tiles_history(plan: Plan, job: Mapping[str, Any]
                         ) -> tuple[Plan, list[str]]:
    from woof.config import load_history_selection
    from woof.offline_child import resolve_child_run_config

    leaf_s = float(job["history"]["leaf_interval_s"])
    recorded = "leaf_output_preset" in job["history"]
    output_lines = _leaf_output_table(job["history"]) if recorded else []
    want_preset = job["history"].get("leaf_output_preset",
                                     LEGACY_LEAF_OUTPUT_PRESET)
    notes = []
    domains = []
    for domain in plan.domains:
        extra = dict(domain.extra)
        if domain.role != "child" or domain.parent is None:
            domains.append(domain)
            continue
        relative = extra.get("child_config")
        old_sha = extra.get("child_config_sha256")
        if not relative or not old_sha:
            raise HindcastRefused(f"{domain.domain_id} carries no "
                                  "child_config / child_config_sha256")
        path = plan.resolve(relative)
        original = path.read_text(encoding="utf-8")
        text = _set_run_key(original, "output_interval_s", leaf_s)
        # The tiles' interval wind maxima and UPP gust (wind_gust_diag).
        text = _ensure_run_key(text, "nwp_diagnostics", 1)
        # The leaf's [output]: the job's preset with its hub-height planes.
        # A job planned before the key existed keeps the planner's table.
        if recorded:
            text = _set_output_table(text, output_lines)
        if text == original:
            domains.append(domain)
            continue

        def check(candidate: Path) -> None:
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    cfg = resolve_child_run_config(candidate)
            except (ValueError, TypeError, KeyError) as error:
                raise HindcastRefused(
                    f"output_interval_s = {leaf_s:g} makes {relative} "
                    f"invalid for woof downscale: {error}") from error
            if not math.isclose(float(cfg.output_interval_s), leaf_s):
                raise HindcastRefused(f"{relative}: output_interval_s did "
                                      "not read back as written")
            if int(getattr(cfg, "nwp_diagnostics", 0)) != 1:
                raise HindcastRefused(f"{relative}: nwp_diagnostics = 1 did "
                                      "not read back as written")
            try:
                selection = load_history_selection(candidate)
            except ValueError as error:
                raise HindcastRefused(
                    f"the leaf [output] makes {relative} invalid for woof "
                    f"downscale: {error}") from error
            if recorded and (selection.preset != want_preset
                             or selection.hub_wind is None):
                raise HindcastRefused(
                    f"{relative}: [output] preset = {want_preset!r} with the "
                    "hub-height diagnostic did not read back as written")

        _replace_atomically_after(path, text, check)
        new_sha = sha256_file(path)
        extra = _replace_strings(extra, old_sha, new_sha)
        history = dict(extra.get("history") or {})
        history["interval_s"] = leaf_s
        if recorded:
            history["preset"] = want_preset
        extra["history"] = history
        spec = plan.resolve(domain.config) if domain.config else None
        if spec is not None and spec.suffix == ".json" and spec.is_file():
            document = _replace_strings(json.loads(spec.read_text()),
                                        old_sha, new_sha)
            if isinstance(document.get("history"), dict):
                document["history"]["interval_s"] = leaf_s
                if recorded:
                    document["history"]["preset"] = want_preset
            _write_json(spec, document)
        domains.append(dataclasses.replace(domain, extra=extra))
        notes.append(f"{relative}: output_interval_s = {leaf_s:g}, "
                     "nwp_diagnostics = 1"
                     + ("; " + " ".join(output_lines) if recorded else "")
                     + " "
                     f"(woof energy hindcast); --child-config-sha256 "
                     f"rebound {old_sha[:12]}... -> {new_sha[:12]}...")
    plan.domains = domains
    notes.append("wrf-tiles: the parent keeps every history frame from t = 0 "
                 "(it is the tiles' boundary forcing) and the tiles write "
                 "their spin-up; the kept window "
                 f"[{job['segment']['keep_start']}, "
                 f"{job['segment']['keep_end']}) is applied at extraction")
    return plan, notes


def _check_model_top(plan: Plan, job: Mapping[str, Any]) -> list[str]:
    import tomllib

    want = float(job["planner"]["p_top_pa"])
    roots = sorted({d.config for d in plan.domains if d.parent is None})
    notes = []
    for config in roots:
        path = plan.resolve(config)
        if path.suffix != ".toml":
            continue
        shared = tomllib.loads(path.read_text(encoding="utf-8")).get(
            "shared") or {}
        p_top = shared.get("p_top")
        if p_top is None:
            notes.append(f"{config} states no p_top; the model top was not "
                         "checked")
            continue
        if not math.isclose(float(p_top), want):
            raise HindcastRefused(
                f"{config} has p_top = {float(p_top):g} Pa but the hindcast "
                f"asked for {want:g} Pa; the planner did not apply the "
                "wind-energy column (woof.energy.vertical)")
        ztop = shared.get("ztop")
        notes.append(f"{config}: p_top = {float(p_top):g} Pa"
                     + ("" if ztop is None else f", ztop = {float(ztop):g} m"))
    return notes


def _leaf_targets(plan: Plan, job: Mapping[str, Any]) -> dict[str, str]:
    template = job["outputs"]["gridded_template"]
    return {d.domain_id: _render(template, leaf=_slug(d.domain_id))
            for d in plan.domains if d.role == "child"}


def _move_aside(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for ordinal in range(1, 100):
        suffix = "" if ordinal == 1 else f"-{ordinal:02d}"
        aside = path.with_name(f"{path.name}.previous-{stamp}{suffix}")
        if not aside.exists():
            path.rename(aside)
            return aside
    raise HindcastRefused(f"no free name to move {path} aside")


def materialize_job(job_ref: str | Path, *, hindcast_dir: Path | None = None,
                    cluster_planner: Callable[..., Any] | None = None,
                    cost_estimator: Callable[..., dict] | None = None,
                    force: bool = False, now: datetime | None = None,
                    allow_running: bool = False) -> dict:
    """Write the job's segment plan into ``<job dir>/plan`` and return the
    ``materialized.json`` receipt.

    Idempotent: a plan already materialized for this job identity, whose
    plan.json still has the recorded sha256, is returned as is.  Anything
    else already in ``plan/`` (a refused or edited attempt, or ``force``)
    is renamed to ``plan.previous-<UTC>`` first, never deleted."""

    job_path = locate_job(job_ref, hindcast_dir=hindcast_dir)
    job = load_job(job_path)
    job_dir = job_path.parent
    home = _hindcast_dir_of(job_path, job)
    plan_dir = job_dir / PLAN_DIR
    receipt_path = job_dir / MATERIALIZED_NAME
    if plan_dir.exists() and receipt_path.exists() and not force:
        receipt = _read_json(receipt_path, "materialized receipt")
        plan_path = job_dir / receipt.get("plan", "")
        if (receipt.get("schema") == MATERIALIZED_SCHEMA
                and receipt.get("identity_sha256") == job["identity_sha256"]
                and plan_path.is_file()
                and sha256_file(plan_path) == receipt.get("plan_sha256")):
            receipt = dict(receipt, reused=True)
            return receipt
    if home is not None and not (force or allow_running):
        record = read_manifest(home)["jobs"].get(job["job_id"]) or {}
        if record.get("status") == "running":
            raise HindcastRefused(
                f"{job['job_id']} is running (since "
                f"{record.get('start_utc')}) and its plan would be "
                "replaced; pass --force if that run is dead")
    if home is not None and not force:
        record = read_manifest(home)["jobs"].get(job["job_id"]) or {}
        if record.get("status") == "complete":
            raise HindcastRefused(
                f"{job['job_id']} is complete but its materialized plan is "
                "missing or changed; pass --force to plan it again")
    notes = []
    if receipt_path.exists():
        aside = _move_aside(receipt_path)
        notes.append(f"earlier receipt moved to {aside.name}")
    if plan_dir.exists():
        aside = _move_aside(plan_dir)
        notes.append(f"earlier plan directory moved to {aside.name}")
    plan_dir.mkdir(parents=True)

    planner = cluster_planner or default_cluster_planner
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = planner(job, plan_dir, job_dir=job_dir, now=now)
        plan_path = _plan_path_from(result, plan_dir)
        if not plan_path.is_file():
            raise HindcastRefused(f"the cluster planner wrote no plan at "
                                  f"{plan_path}")
        plan = load_plan(plan_path)
        topology = job["planner"]["topology"]
        if plan.topology not in TOPOLOGIES:
            raise HindcastRefused(f"the planner returned a {plan.topology} "
                                  "plan; a hindcast drives "
                                  f"{', '.join(TOPOLOGIES)}")
        if plan.topology != topology:
            notes.append(f"the cluster planner chose {plan.topology}; the "
                         f"job asked for {topology}")
        want_start = datetime.strptime(job["segment"]["run_start"],
                                       "%Y-%m-%dT%H:%M:%SZ")
        got_start, _ = parse_utc(plan.start, "plan start")
        if got_start != want_start or not math.isclose(
                plan.hours, job["segment"]["run_hours"]):
            raise HindcastRefused(
                f"the plan runs {plan.start} for {plan.hours:g} h, not the "
                f"segment {job['segment']['run_start']} for "
                f"{job['segment']['run_hours']:g} h")
        notes += _check_model_top(plan, job)
        if plan.topology == "wrf-nests":
            notes += _apply_nests_history(plan, job)
            history_begin = {
                d.domain_id: float(job["history"]["history_begin_s"])
                for d in plan.domains}
        else:
            plan, more = _apply_tiles_history(plan, job)
            notes += more
            history_begin = {d.domain_id: None for d in plan.domains}
        domains = []
        for domain in plan.domains:
            leaf = domain.role == "child"
            record = {
                "history_begin_s": history_begin[domain.domain_id],
                "history_interval_s": (
                    float(job["history"]["leaf_interval_s"]) if leaf
                    else (job["history"]["parent_interval_s"]
                          if plan.topology == "wrf-nests" else
                          domain.extra.get("history_interval_s"))),
                "keep_window": list(job["outputs"]["time_window"])}
            domains.append(dataclasses.replace(
                domain, extra={**domain.extra, "hindcast": record}))
        plan.domains = domains
        plan.notes = list(plan.notes) + [
            f"woof energy hindcast job {job['job_id']}: segment "
            f"{job['segment']['run_start']} + {job['segment']['run_hours']:g} "
            f"h, spin-up {job['segment']['spinup_hours']} h, kept "
            f"[{job['segment']['keep_start']}, {job['segment']['keep_end']})"]
        dump_plan(plan, plan_path)
    except Exception as error:
        update_job(home, job["job_id"], status="failed",
                   reason=f"materialize refused: {error}",
                   end_utc=_utc_now())
        raise
    estimator = cost_estimator or default_cost_estimate
    targets = _leaf_targets(plan, job)
    receipt = {
        "schema": MATERIALIZED_SCHEMA,
        "job_id": job["job_id"],
        "identity_sha256": job["identity_sha256"],
        "plan": _relpath(plan_path, job_dir),
        "plan_sha256": sha256_file(plan_path),
        "topology": plan.topology,
        "source": plan.source,
        "start": plan.start, "hours": plan.hours,
        "domains": [{"domain_id": d.domain_id, "role": d.role,
                     "dx_m": d.dx_m, "grid_id": d.grid_id,
                     "parent": d.parent, "run_dir": d.run_dir,
                     "config": d.config, "sites": len(d.site_ids),
                     **d.extra["hindcast"]} for d in plan.domains],
        "targets": {"gridded": targets,
                    "sites": job["outputs"]["sites_url"],
                    "root_relative_to": job["outputs"]["root_relative_to"],
                    "time_window": job["outputs"]["time_window"]},
        "cost": estimator(job, plan),
        "materialized_utc": _utc_now(),
        "notes": notes,
    }
    _write_json(receipt_path, receipt)
    update_job(home, job["job_id"], status="materialized", reason=None,
               plan=f"{JOBS_DIR}/{job['job_id']}/{receipt['plan']}")
    return receipt


# --------------------------------------------------------------------------
# run-local


def _slice_window(data, keep_start: datetime, keep_end: datetime):
    """``ForecastData`` restricted to ``[keep_start, keep_end)``."""

    import numpy as np

    times = data.times.astype("datetime64[s]")
    lo = np.datetime64(keep_start.strftime("%Y-%m-%dT%H:%M:%S"), "s")
    hi = np.datetime64(keep_end.strftime("%Y-%m-%dT%H:%M:%S"), "s")
    mask = (times >= lo) & (times < hi)
    if not mask.any():
        raise HindcastRefused(
            f"the run wrote no site frame in the kept window "
            f"[{_iso(keep_start)}, {_iso(keep_end)}); its frames span "
            f"{times.min() if times.size else 'nothing'} .. "
            f"{times.max() if times.size else ''}")
    variables = {name: values[mask] for name, values in data.variables.items()}
    return dataclasses.replace(
        data, times=data.times[mask], variables=variables,
        notes=list(data.notes) + [
            f"woof energy hindcast kept [{_iso(keep_start)}, "
            f"{_iso(keep_end)}): {int(mask.sum())} of {times.size} frames"])


def default_post_step(job: Mapping[str, Any], *, plan_path: Path,
                      job_dir: Path, receipt: Mapping[str, Any],
                      extract: bool = True,
                      writer: Callable[..., Any] | None = None) -> dict:
    """Extract the kept window at the cluster's sites, then hand it on.

    With ``extract`` the plan's runs are sampled
    (:func:`woof.energy.extract.build_forecast`), cut to the kept window and
    written to ``sites-forecast.nc`` in the job directory.  ``writer``
    (the Icechunk writer, or any callable) is then called as
    ``writer(job=..., job_dir=..., plan_path=..., forecast=..., targets=...)``
    with the in-memory kept-window forecast (None without ``extract``) and
    the materialized targets; what it returns is recorded."""

    record: dict[str, Any] = {}
    forecast = None
    if extract:
        from woof.energy.extract import build_forecast, write_netcdf

        keep_start = datetime.strptime(job["segment"]["keep_start"],
                                       "%Y-%m-%dT%H:%M:%SZ")
        keep_end = datetime.strptime(job["segment"]["keep_end"],
                                     "%Y-%m-%dT%H:%M:%SZ")
        forecast = _slice_window(build_forecast(plan_path), keep_start,
                                 keep_end)
        written = write_netcdf(forecast, job_dir / SITES_FORECAST_NAME)
        record["extract"] = {
            "output": _relpath(written, job_dir),
            "times": int(forecast.times.size),
            "variables": list(forecast.variables)}
    if writer is not None:
        result = writer(job=job, job_dir=job_dir, plan_path=plan_path,
                        forecast=forecast, targets=receipt.get("targets"))
        record["writer"] = result if result is None or isinstance(
            result, (dict, list, str, int, float, bool)) else repr(result)
    return record


def _claim(home: Path | None, job_id: str, *, force: bool) -> str:
    """Mark the job running under the manifest lock, so two workers
    cannot both start it; return the status found."""

    if home is None:
        return "pending"
    with _manifest_lock(home):
        manifest = read_manifest(home)
        record = manifest["jobs"].get(job_id)
        if record is None:
            raise HindcastRefused(f"job {job_id} is not in "
                                  f"{home / MANIFEST_NAME}")
        found = record["status"]
        if not force and found == "complete":
            return found
        if not force and found == "running":
            raise HindcastRefused(f"{job_id} is running (since "
                                  f"{record.get('start_utc')}); pass "
                                  "--force if that run is dead")
        record.update(status="running", reason=None, start_utc=_utc_now(),
                      end_utc=None,
                      attempts=int(record.get("attempts") or 0) + 1)
        _write_manifest(home, manifest)
        return found


def run_local(job_ref: str | Path, *, hindcast_dir: Path | None = None,
              dry_run: bool = False, resume: bool = False,
              force: bool = False, extract: bool = True,
              writer: Callable[..., Any] | None = None,
              runner: Callable[..., dict] | None = None,
              post_step: Callable[..., dict] | None = None,
              cluster_planner: Callable[..., Any] | None = None,
              now: datetime | None = None) -> dict:
    """Materialize (if needed), run every plan domain on this machine with
    :func:`woof.energy.run.run_plan`, then the post-step; return a record.

    A job the manifest calls complete is skipped unless ``force``.
    ``dry_run`` materializes (CPU only) and returns ``run_plan``'s ordered
    argv without launching anything or changing the job's status beyond
    ``materialized``."""

    job_path = locate_job(job_ref, hindcast_dir=hindcast_dir)
    job = load_job(job_path)
    job_dir = job_path.parent
    home = _hindcast_dir_of(job_path, job)
    if home is not None and not force:
        record = read_manifest(home)["jobs"].get(job["job_id"]) or {}
        if record.get("status") == "complete" and not dry_run:
            return {"job_id": job["job_id"], "skipped": True,
                    "reason": "complete in the manifest (pass --force to "
                              "run it again)"}
        if record.get("status") == "running":
            raise HindcastRefused(
                f"{job['job_id']} is running (since "
                f"{record.get('start_utc')}); pass --force if that run is "
                "dead")
    receipt = materialize_job(job_path, cluster_planner=cluster_planner,
                              now=now, allow_running=force)
    plan_path = job_dir / receipt["plan"]
    if runner is None:
        from woof.energy.run import run_plan as runner
    if dry_run:
        run = runner(plan_path, dry_run=True, resume=resume)
        return {"job_id": job["job_id"], "dry_run": True,
                "plan": str(plan_path), "materialized": receipt,
                "run": run,
                "post_step": ("extract the kept window "
                              f"{job['outputs']['time_window']} to "
                              f"{SITES_FORECAST_NAME}" if extract
                              else "no extract")
                             + ("; then the writer" if writer else "")}
    if _claim(home, job["job_id"], force=force) == "complete":
        return {"job_id": job["job_id"], "skipped": True,
                "reason": "completed by another worker meanwhile"}
    try:
        from woof.energy.run import RunRefusal

        try:
            run = runner(plan_path, resume=resume)
        except (RunRefusal, ContractError) as error:
            raise HindcastRefused(f"woof energy run refused: {error}"
                                  ) from error
        domains = run.get("domains") or {}
        bad = {d: r.get("reason") or r.get("status")
               for d, r in domains.items() if r.get("status") != "complete"}
        if bad or run.get("interrupted"):
            reason = ("interrupted" if run.get("interrupted") else
                      "domains not complete: " + "; ".join(
                          f"{d}: {why}" for d, why in sorted(bad.items())))
            update_job(home, job["job_id"], status="failed", reason=reason,
                       end_utc=_utc_now())
            return {"job_id": job["job_id"], "ok": False, "reason": reason,
                    "run_manifest": run.get("manifest_path")}
        step = post_step or default_post_step
        post = step(job, plan_path=plan_path, job_dir=job_dir,
                    receipt=receipt, extract=extract, writer=writer)
    except KeyboardInterrupt:
        update_job(home, job["job_id"], status="failed",
                   reason="interrupted (Ctrl-C)", end_utc=_utc_now())
        raise
    except Exception as error:
        update_job(home, job["job_id"], status="failed",
                   reason=f"{type(error).__name__}: {error}",
                   end_utc=_utc_now())
        raise
    update_job(home, job["job_id"], status="complete", reason=None,
               end_utc=_utc_now(), post_step=post)
    return {"job_id": job["job_id"], "ok": True,
            "run_manifest": run.get("manifest_path"), "post_step": post}


# --------------------------------------------------------------------------
# status


def hindcast_status(hindcast_dir: Path, *, cluster: str | None = None,
                    status: str | None = None, list_jobs: bool = False
                    ) -> dict:
    """Counts by status, per-cluster coverage of the kept windows (gaps and
    overlaps), source changes, and optionally the job records."""

    hindcast_dir = Path(hindcast_dir)
    manifest = read_manifest(hindcast_dir)
    if manifest is None:
        raise HindcastRefused(f"{hindcast_dir} holds no {MANIFEST_NAME}")
    jobs = manifest["jobs"]
    if cluster is not None:
        jobs = {k: v for k, v in jobs.items() if v["cluster_id"] == cluster}
        if not jobs:
            raise HindcastRefused(f"no jobs for cluster {cluster!r}")
    counts: dict[str, int] = {}
    per_cluster: dict[str, dict] = {}
    for job_id, record in sorted(jobs.items(),
                                 key=lambda kv: (kv[1]["cluster_id"],
                                                 kv[1]["keep_start"])):
        counts[record["status"]] = counts.get(record["status"], 0) + 1
        entry = per_cluster.setdefault(record["cluster_id"], {
            "jobs": 0, "counts": {}, "first_kept": record["keep_start"],
            "last_kept": record["keep_end"], "gaps": [], "overlaps": [],
            "sources": {}, "source_changes": 0, "_end": None,
            "_source": None})
        entry["jobs"] += 1
        entry["counts"][record["status"]] = entry["counts"].get(
            record["status"], 0) + 1
        entry["sources"][record["source"]] = entry["sources"].get(
            record["source"], 0) + 1
        if entry["_end"] is not None:
            if record["keep_start"] > entry["_end"]:
                entry["gaps"].append([entry["_end"], record["keep_start"]])
            elif record["keep_start"] < entry["_end"]:
                entry["overlaps"].append([record["keep_start"],
                                          entry["_end"]])
        if entry["_source"] is not None and entry["_source"] != record["source"]:
            entry["source_changes"] += 1
        entry["_end"] = max(entry["_end"] or "", record["keep_end"])
        entry["_source"] = record["source"]
        entry["last_kept"] = entry["_end"]
    for entry in per_cluster.values():
        entry.pop("_end")
        entry.pop("_source")
    out = {"schema": MANIFEST_SCHEMA, "hindcast_dir": str(hindcast_dir),
           "jobs": len(jobs), "counts": counts, "clusters": per_cluster,
           "updated_utc": manifest.get("updated_utc"),
           "notes": manifest.get("notes") or []}
    if status is not None:
        out["selected"] = {k: v for k, v in jobs.items()
                           if v["status"] == status}
    elif list_jobs:
        out["selected"] = jobs
    return out


# --------------------------------------------------------------------------
# CLI


def _load_writer(spec: str | None) -> Callable[..., Any] | None:
    if spec is None:
        return None
    module_name, _, attribute = spec.partition(":")
    if not module_name or not attribute:
        raise HindcastRefused(f"--writer {spec!r} must be MODULE:CALLABLE")
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise HindcastRefused(f"--writer: cannot import {module_name}: "
                              f"{error}") from error
    writer = getattr(module, attribute, None)
    if not callable(writer):
        raise HindcastRefused(f"--writer: {spec} is not callable")
    return writer


def _request_from_args(args) -> HindcastRequest:
    start, start_has_hour = parse_utc(args.start, "--start")
    end, _ = parse_utc(args.end, "--end")
    return HindcastRequest(
        outdir=Path(args.outdir), start=start, end=end,
        sites_path=None if args.sites is None else Path(args.sites),
        partition_path=(None if args.partition is None
                        else Path(args.partition)),
        clusters=args.clusters, cluster_id=args.cluster_id,
        topology=args.topology, segment_hours=args.segment_hours,
        spinup_hours=args.spinup_hours, cycle_hour=args.cycle_hour,
        start_has_hour=start_has_hour, dx_m=args.dx_m,
        corridor_km=args.corridor_km, parent_dx_m=args.parent_dx_m,
        nz=args.nz, card=args.card, vram_gib=args.vram_gib,
        max_domains=args.max_domains, p_top_pa=args.p_top_pa,
        source=args.source, source_priority=args.source_priority,
        leaf_history_interval_s=args.leaf_history_interval_s,
        parent_history_interval_s=args.parent_history_interval_s,
        leaf_output_preset=getattr(args, "leaf_output_preset",
                                   DEFAULT_LEAF_OUTPUT_PRESET),
        output_root=args.output_root, gridded_template=args.gridded_url,
        sites_template=args.sites_url)


def _print(document: Any) -> None:
    print(json.dumps(document, indent=2, default=str))


def _main_plan(args) -> int:
    request = _request_from_args(args)
    result = plan_hindcast(request, dry_run=bool(args.dry_run),
                           resume=bool(args.resume))
    if result.get("dry_run"):
        jobs = result["jobs"]
        _print({"command": "woof energy hindcast plan", "dry_run": True,
                "hindcast_dir": result["hindcast_dir"],
                "jobs": len(jobs), "manifest": result["manifest"],
                "example_job": jobs[0] if jobs else None})
        return 0
    _print({"command": "woof energy hindcast plan", "dry_run": False,
            "manifest": result["manifest_path"],
            "jobs_planned": len(result["planned_job_ids"]),
            "jobs_in_manifest": len(result["jobs"]),
            "counts": result["counts"],
            "first_job": (f"{JOBS_DIR}/{result['planned_job_ids'][0]}/"
                          f"{JOB_NAME}" if result["planned_job_ids"]
                          else None),
            "notes": result["notes"]})
    return 0


def _main_status(args) -> int:
    _print(hindcast_status(Path(args.dir), cluster=args.cluster,
                           status=args.status, list_jobs=bool(args.jobs)))
    return 0


def _main_materialize(args) -> int:
    receipt = materialize_job(args.job, hindcast_dir=args.dir,
                              force=bool(args.force))
    _print({"command": "woof energy hindcast materialize", **receipt})
    return 0


def _main_run_local(args) -> int:
    record = run_local(args.job, hindcast_dir=args.dir,
                       dry_run=bool(args.dry_run), resume=bool(args.resume),
                       force=bool(args.force), extract=not args.no_extract,
                       writer=_load_writer(args.writer))
    _print({"command": "woof energy hindcast run-local", **record})
    if record.get("dry_run") or record.get("skipped"):
        return 0
    return 0 if record.get("ok") else 1


_COMMANDS = {"plan": _main_plan, "status": _main_status,
             "materialize": _main_materialize, "run-local": _main_run_local}


def main(args) -> int:
    command = getattr(args, "hindcast_command", None)
    if command is None:
        print(__doc__.strip().split("\n\n")[0], file=sys.stderr)
        print("subcommands: " + ", ".join(_COMMANDS), file=sys.stderr)
        return 2
    try:
        return _COMMANDS[command](args)
    except (ContractError, ValueError, RuntimeError, NotImplementedError,
            OSError) as error:
        # ContractError/HindcastRefused, the planners' refusals (ValueError
        # subclasses), woof energy run's RunRefusal (RuntimeError) and
        # unreadable files: one refusal line, exit 2.
        print(f"woof energy hindcast {command}: refused "
              f"({type(error).__name__}): {error}", file=sys.stderr)
        return 2


__all__ = [
    "JOB_SCHEMA", "MANIFEST_SCHEMA", "MATERIALIZED_SCHEMA",
    "PARTITION_SCHEMA", "JOB_STATUSES", "TOPOLOGIES", "Cluster",
    "HindcastRefused", "HindcastRequest", "Segment", "SourceChoice",
    "default_cluster_planner", "default_cost_estimate",
    "default_post_step", "default_source_selector", "fallback_select",
    "hindcast_status", "load_clusters", "load_job", "locate_job",
    "materialize_job", "parse_utc", "patch_domain_tables", "plan_hindcast",
    "read_manifest", "run_local", "segment_windows", "update_job", "main",
]
