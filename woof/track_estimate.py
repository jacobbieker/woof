"""Price a set of moving-nest LES runs along tracks before renting a card.

``woof estimate-tracks`` prices one woof run per track: a static 1.5 km d01
over the track's horizontal envelope plus a margin, a 300 m d02 and a 100 m
d03 that move along the track, all on one card in one process, with the d03
history written every 120 s for the track's duration.  Per track and in
total it reports cell-steps per domain, wall hours on each card, the GPU
memory the tree holds, the history bytes (raw and after compression) and the
AWS on-demand cost.

**Every number is an estimate.**  One card is calibrated: the RTX A6000
rows come from ``tools/bench_les_256.py`` (100 m, 256 levels, LES +
RTE-RRTMGP + Thompson, dt 0.5 s; the conditions and the measured numbers
are in :data:`CALIBRATION` and docs/energy-hindcast/cost.md).  Every other
card is the A6000 figure scaled by memory bandwidth and is labelled
``uncalibrated`` wherever it appears.  The prices in :data:`AWS_PRICES` are
approximate us-west-2 on-demand list prices, labelled "verify before
launch"; edit that table, or pass ``--prices`` with a JSON copy, before
quoting a figure.

Tracks come in one of two shapes, and their names come only from the input:

* ``--tracks JSON``: a list (or ``{"tracks": [...]}``) of ``{"name",
  "duration_h", "envelope_km": [ew, ns]}``, so this module depends on no
  track parser;
  Track3D documents (``woof track inspect --json``) are read too;
* ``--paths FILE|DIR ...``: 3-D path files, one track per file, read and
  quality-controlled by :mod:`woof.track` from the time, latitude,
  longitude and height columns named on the command line (the same
  options as ``woof track inspect``).  A track is named by its file stem,
  and priced over its launch-to-landing duration.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from woof.explain import layered

SCHEMA = "woof.track-estimate.v1"
GIB = 1024 ** 3


class EstimateRefusal(ValueError):
    """The estimate cannot be made; the message names what is missing."""


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------
#: The measured card.  ``tools/bench_les_256.py`` on this project's RTX
#: A6000 (48 GB, 768 GB/s, 84 SMs) with the card otherwise idle; see the
#: A6000 row of docs/energy-hindcast/cost.md for the receipt.
#:
#: * ``cell_steps_per_s_ordinary``: the 325 x 325 x 256 leaf alone, 599
#:   ordinary steps at 0.935 s, setup and compilation excluded.  Applied to
#:   d01 and d02 as well: the 3-domain tree (d01 140 x 140, d02 241 x 241,
#:   d03 325 x 325, all 256 levels, stepped 1 : 5 : 15, no radiation step
#:   inside the timed window) measured 29.1 M ordinary cell-steps/s over
#:   the whole tree.
#: * ``radiation_step_ratio``: a forced RTE-RRTMGP step costs 3.18 s
#:   against 0.935 s, so each domain pays ``(ratio - 1) / (radt / dt)``
#:   more per step: 0.4% on the leaf (one step in 600), 1.2% on d02 and 6%
#:   on d01.  ``cell_steps_per_s`` is the leaf's measured cadence mean.
#: * ``bytes_per_cell`` / ``fixed_gib``: a straight line through the
#:   3-domain tree's process high-water mark (nvidia-smi at the end of the
#:   case; the CuPy pool keeps what it allocated) at d03 325, 380 and 400 on
#:   a side: 37.6, 43.7 and 46.2 GiB.
CALIBRATION: Mapping[str, Any] = {
    "card": "rtx-a6000",
    "cell_steps_per_s_ordinary": 28.92e6,
    "radiation_step_ratio": 3.40,
    "cell_steps_per_s": 28.8e6,
    "bytes_per_cell": 662.0,
    "fixed_gib": 8.7,
    "receipt": "tools/bench_les_256.py; docs/energy-hindcast/cost.md "
               "(calibrated A6000 row)",
    "workload": "idealized periodic flat domain, WK82 sounding, 100 m, "
                "256 levels, km_opt=2 diff_opt=2 PBL off, Thompson, "
                "RTE-RRTMGP 4/4 radt 5 min, Noah, dt 0.5 s; no lateral-"
                "boundary, nest or output kernels",
}

#: The measured card's own row (woof.card_catalog has no workstation
#: Ampere parts).  Memory is what nvidia-smi reported on it.
A6000 = {"key": "rtx-a6000", "name": "NVIDIA RTX A6000 48GB",
         "memory_mib": 49_140, "sm_count": 84, "bandwidth_gbs": 768.0}

#: Cards priced by default: the calibrated one first, then the cloud parts
#: (woof.card_catalog rows) scaled by bandwidth.
DEFAULT_CARDS = ("rtx-a6000", "l40s-48gb", "a100-40gb", "a100-80gb",
                 "h100-sxm-80gb", "h200-141gb")

#: Throughput factors (optimistic, central, pessimistic) on the calibrated
#: rate.  The A6000's band covers what the idealized benchmark leaves out
#: (boundary and nest kernels, moving-nest relocation, a real sounding and
#: surface); the bandwidth-scaled cards add the scaling itself, which no
#: datacentre timing in this tree supports.
BAND_CALIBRATED = (1.10, 0.90, 0.70)
BAND_UNCALIBRATED = (1.25, 0.90, 0.45)
BANDS = ("optimistic", "central", "pessimistic")

#: Card memory held back from a track's tree (GiB): the runtime, the
#: driver and fragmentation.
MEMORY_MARGIN_GIB = 1.0


def overheads() -> dict[str, float]:
    """woof.energy.cost's launch-to-first-step fit and device-to-host rate.

    Read from that module (lazily: every ``woof`` invocation builds this
    door's parser) so the two estimators cannot drift apart.  Preparation
    is charged to the card because it is rented while preparation runs;
    history frames leave the card at the staged-copy rate, and compression
    and the store write are assumed to overlap the integration.
    """
    from woof.energy.cost import D2H_GBPS, PREP_INTERCEPT_S, PREP_S_PER_MCELL

    return {"prep_intercept_s": PREP_INTERCEPT_S,
            "prep_s_per_mcell": PREP_S_PER_MCELL, "d2h_gbps": D2H_GBPS}


# --------------------------------------------------------------------------
# prices -- EDIT HERE (or pass --prices PATH with the same shape)
# --------------------------------------------------------------------------
#: AWS on-demand, us-west-2, USD per instance-hour.  APPROXIMATE list
#: prices after AWS's June 2025 GPU price cut, written down by hand: VERIFY
#: BEFORE LAUNCH.  A card with no AWS instance (the A6000) is priced at
#: nothing and says so.
AWS_PRICES: Mapping[str, Any] = {
    "as_of": "2025-07 list prices, approximate",
    "label": "verify before launch",
    "region": "us-west-2",
    "market": "on-demand",
    "source": "https://aws.amazon.com/ec2/pricing/on-demand/",
    "instances": {
        "l40s-48gb": {"instance": "g6e.xlarge", "gpus": 1,
                      "usd_per_hour": 1.861},
        "a100-40gb": {"instance": "p4d.24xlarge", "gpus": 8,
                      "usd_per_hour": 21.96},
        "a100-80gb": {"instance": "p4de.24xlarge", "gpus": 8,
                      "usd_per_hour": 27.45},
        "h100-sxm-80gb": {"instance": "p5.48xlarge", "gpus": 8,
                          "usd_per_hour": 55.04},
        "h200-141gb": {"instance": "p5en.48xlarge", "gpus": 8,
                       "usd_per_hour": 63.30},
    },
    # S3 Standard, first tier.
    "s3_usd_per_gb_month": 0.023,
}


# --------------------------------------------------------------------------
# the design
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DomainDesign:
    """One domain of the per-track tree.

    ``nx``/``ny`` of ``None`` size the domain from the track's envelope
    plus ``margin_km`` on every side (the static d01).  ``spinup_h`` is how
    long the domain runs before the track starts.
    """

    name: str
    dx_m: float
    dt_s: float
    spinup_h: float
    nx: int | None = None
    ny: int | None = None
    margin_km: float = 0.0


@dataclass(frozen=True)
class TrackDesign:
    """The per-track tree, its levels, its output and the compression band."""

    nz: int = 256
    domains: tuple[DomainDesign, ...] = (
        DomainDesign("d01", 1500.0, 7.5, 3.0, margin_km=60.0),
        DomainDesign("d02", 300.0, 1.5, 3.0, nx=241, ny=241),
        DomainDesign("d03", 100.0, 0.5, 1.0, nx=325, ny=325),
    )
    output_domain: str = "d03"
    output_interval_s: float = 120.0
    output_preset: str = "balloon"
    compression: tuple[float, float] = (0.26, 0.40)
    #: Lateral rows a parent keeps around its child (spec + relax zones).
    parent_rim_cells: int = 10
    #: Radiation cadence on every domain (minutes).
    radt_minutes: float = 5.0


@dataclass(frozen=True)
class Track:
    name: str
    duration_h: float
    envelope_km: tuple[float, float]
    max_height_m: float | None = None
    #: What the path reader (woof.track) dropped or repaired, by reason.
    qc: Mapping[str, int] | None = None


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------
def _track(row: Any, where: str) -> Track:
    try:
        name = str(row["name"])
        duration = float(row["duration_h"])
        env = row["envelope_km"]
        if isinstance(env, Mapping):
            env = (env["ew"], env["ns"])
        ew, ns = (float(v) for v in env)
        top = row.get("max_height_m")
        top = None if top is None else float(top)
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise EstimateRefusal(layered(
            f"{where}: a track needs name, duration_h and envelope_km "
            f"[ew, ns]; got {row!r}",
            f"The estimator prices each track from those three facts and "
            f"nothing else ({type(error).__name__}: {error}).")) from error
    if not (duration > 0 and ew >= 0 and ns >= 0
            and all(math.isfinite(v) for v in (duration, ew, ns))):
        raise EstimateRefusal(
            f"{where}: track {name}: duration_h must be positive and the "
            f"envelope non-negative; got {duration}, ({ew}, {ns})")
    return Track(name, duration, (ew, ns), top)


def load_tracks_json(path: Path) -> list[Track]:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as error:
        raise EstimateRefusal(f"{path}: cannot read ({error})") from error
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise EstimateRefusal(f"{path}: not JSON ({error})") from error
    rows = doc.get("tracks", [doc]) if isinstance(doc, Mapping) else doc
    if not isinstance(rows, list):
        raise EstimateRefusal(
            f"{path}: expected a JSON list of tracks or {{\"tracks\": "
            f"[...]}}, got {type(doc).__name__}")
    return [_row(r, str(path)) for r in rows]


def _row(row: Any, where: str) -> Track:
    """A ``{name, duration_h, envelope_km}`` row, or a Track3D document
    (``woof track inspect --json``)."""
    from woof.track.model import TRACK_SCHEMA, Track3D

    if isinstance(row, Mapping) and row.get("schema") == TRACK_SCHEMA:
        try:
            t3d = Track3D.from_dict(row)
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise EstimateRefusal(f"{where}: {error}") from error
        return from_track3d(t3d, where)
    return _track(row, where)


def from_track3d(t, where: str = "track") -> Track:
    """A :class:`Track` from a ``woof.track`` Track3D: its launch-to-landing
    duration, its envelope's sides and its peak height, held to the same
    checks as a ``{name, duration_h, envelope_km}`` row."""
    track = _track({"name": t.name, "duration_h": t.duration_h,
                    "envelope_km": [t.extent_ew_km, t.extent_ns_km],
                    "max_height_m": t.max_height}, where)
    return replace(track, qc=dict(t.qc))


def read_path_files(sources: Iterable[Path], mapping=None, *,
                    min_climb_m: float | None = None) -> list[Track]:
    """One track per 3-D path file, read and cleaned by ``woof.track``.

    ``mapping`` is a :class:`woof.track.ColumnMapping` naming the time,
    latitude, longitude and height columns; directories contribute every
    readable file directly inside them.  ``min_climb_m`` is the climb that
    makes a launch (``--min-climb``; woof.track's default when ``None``).
    """
    from woof.track import load_track
    from woof.track.ingest import supported

    sources = [Path(s) for s in sources]
    paths = supported(sources)
    if not paths:
        raise EstimateRefusal(
            f"no path files in {', '.join(str(s) for s in sources)}")
    tracks = []
    for path in paths:
        try:
            kwargs = {} if min_climb_m is None else {"min_climb_m": min_climb_m}
            tracks.append(from_track3d(load_track(path, mapping, **kwargs),
                                       str(path)))
        except (ValueError, OSError, UnicodeDecodeError, csv.Error) as error:
            raise EstimateRefusal(f"{path}: {error}") from error
    return tracks


def select_tracks(tracks: Sequence[Track], patterns: str | None) -> list[Track]:
    """Tracks whose name matches any comma-separated shell pattern."""
    if not patterns:
        return list(tracks)
    pats = [p.strip() for p in patterns.split(",") if p.strip()]
    chosen = [t for t in tracks
              if any(fnmatch.fnmatchcase(t.name, p) for p in pats)]
    if not chosen:
        raise EstimateRefusal(f"--select {patterns!r} matches none of the "
                              f"{len(tracks)} tracks")
    return chosen


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------
def card(key: str) -> dict:
    """The card's memory and bandwidth, and whether it is calibrated."""
    from woof.card_catalog import _normalise, card_row

    if _normalise(key) in ("a6000", "a600048gb"):
        row = dict(A6000)
    else:
        found = card_row(key)
        if found is None:
            raise EstimateRefusal(
                f"--cards: {key!r} is not a card woof knows; use "
                f"{', '.join(DEFAULT_CARDS)} or a woof.card_catalog spelling")
        row = {"key": found.key, "name": found.name,
               "memory_mib": found.memory_mib, "sm_count": found.sm_count,
               "bandwidth_gbs": found.bandwidth_gbs}
    calibrated = row["key"] == CALIBRATION["card"]
    scale = row["bandwidth_gbs"] / A6000["bandwidth_gbs"]
    row.update(
        memory_gib=row["memory_mib"] / 1024.0,
        calibrated=calibrated,
        calibration=("calibrated (A6000, tools/bench_les_256.py)" if calibrated
                     else f"uncalibrated: A6000 x {scale:.2f} by memory "
                          "bandwidth"),
        cell_steps_per_s=CALIBRATION["cell_steps_per_s_ordinary"] * scale,
        band=BAND_CALIBRATED if calibrated else BAND_UNCALIBRATED)
    return row


def domain_shape(d: DomainDesign, track: Track, design: TrackDesign,
                 child: DomainDesign | None) -> tuple[int, int]:
    """``(nx, ny)``: fixed, or the envelope plus margin and the child's box."""
    if d.nx is not None and d.ny is not None:
        return int(d.nx), int(d.ny)
    ew, ns = track.envelope_km
    nx = math.ceil((ew + 2 * d.margin_km) * 1000.0 / d.dx_m) + 1
    ny = math.ceil((ns + 2 * d.margin_km) * 1000.0 / d.dx_m) + 1
    if child is not None and child.nx is not None:
        # The moving child is centred on the track, so at the envelope's
        # edges it reaches half its box beyond them, plus the parent's rim.
        rim = 2 * design.parent_rim_cells
        nx = max(nx, math.ceil((ew * 1000.0 + child.nx * child.dx_m)
                               / d.dx_m) + rim)
        ny = max(ny, math.ceil((ns * 1000.0 + child.ny * child.dx_m)
                               / d.dx_m) + rim)
    return nx, ny


def check_design(design: TrackDesign) -> None:
    """Refuse a design that cannot run: bad numbers or a child too big."""
    if design.nz < 2:
        raise EstimateRefusal(f"--nz must be at least 2; got {design.nz}")
    if not design.radt_minutes > 0:
        raise EstimateRefusal("radt_minutes must be positive")
    if not design.output_interval_s > 0:
        raise EstimateRefusal("--output-interval-s must be positive; got "
                              f"{design.output_interval_s}")
    for d in design.domains:
        if not (d.dx_m > 0 and d.dt_s > 0 and d.spinup_h >= 0
                and d.margin_km >= 0):
            raise EstimateRefusal(f"{d.name}: dx, dt must be positive and "
                                  "spin-up, margin non-negative")
        if d.nx is not None and (d.nx < 1 or d.ny < 1):
            raise EstimateRefusal(f"{d.name}: {d.nx} x {d.ny} columns is "
                                  "not a domain (--d03-n must be positive)")
    for parent, child in zip(design.domains, design.domains[1:]):
        if parent.nx is None or child.nx is None:
            continue
        room = (parent.nx - 2 * design.parent_rim_cells) * parent.dx_m
        if child.nx * child.dx_m > room or child.ny * child.dx_m > (
                parent.ny - 2 * design.parent_rim_cells) * parent.dx_m:
            raise EstimateRefusal(layered(
                f"{child.name} ({child.nx} x {child.ny} at {child.dx_m:g} m) "
                f"does not fit inside {parent.name} ({parent.nx} x "
                f"{parent.ny} at {parent.dx_m:g} m) less its "
                f"{design.parent_rim_cells}-row rim; shrink --d03-n",
                "A nest must sit inside its parent's interior, clear of the "
                "specified and relaxation rows the parent forces it from."))


#: The leaf's physics, as tools/bench_les_256.py times it, for pricing its
#: history frame.
LEAF_PHYSICS: Mapping[str, Any] = {
    "moist": True, "mp_physics": 8, "bl_pbl_physics": 0, "km_opt": 2,
    "diff_opt": 2, "sf_sfclay_physics": 1, "sf_surface_physics": 2,
    "ra_lw_physics": 4, "ra_sw_physics": 4, "radt": 5.0,
}


def output_frame_bytes(design: TrackDesign, nx: int, ny: int) -> int:
    """Bytes of one history frame of the design's preset on the output
    domain, as the disk planner prices it (woof.disk_budget): the writer's
    own inventory for an LES leaf with this tree's physics, staggered faces
    and 2-D planes included, before compression."""
    d = next(d for d in design.domains if d.name == design.output_domain)
    return _frame_bytes(design.output_preset, int(nx), int(ny),
                        int(design.nz), float(d.dx_m), float(d.dt_s))


@lru_cache(maxsize=None)
def _frame_bytes(preset: str, nx: int, ny: int, nz: int, dx: float,
                 dt: float) -> int:
    from woof.config import RunConfig
    from woof.disk_budget import history_frame_bytes
    from woof.io.history_selection import HistorySelection

    cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=dx, dy=dx, ztop=25000.0, dt=dt,
                    run_seconds=3600.0, **LEAF_PHYSICS)
    try:
        selection = HistorySelection.from_mapping({"preset": preset})
        return int(history_frame_bytes(cfg, selection))
    except ValueError as error:
        raise EstimateRefusal(f"--output-preset {preset!r}: {error}") from error


def track_work(track: Track, design: TrackDesign) -> dict:
    """Cells, steps and cell-steps per domain, and the track's history."""
    domains = []
    for i, d in enumerate(design.domains):
        child = design.domains[i + 1] if i + 1 < len(design.domains) else None
        nx, ny = domain_shape(d, track, design, child)
        cells = nx * ny * design.nz
        run_s = (d.spinup_h + track.duration_h) * 3600.0
        steps = math.ceil(run_s / d.dt_s - 1e-9)
        # Ordinary-step equivalents: one radiation step in `every` costs
        # radiation_step_ratio ordinary ones.
        every = max(1, round(design.radt_minutes * 60.0 / d.dt_s))
        factor = 1.0 + (CALIBRATION["radiation_step_ratio"] - 1.0) / every
        domains.append({"name": d.name, "nx": nx, "ny": ny, "nz": design.nz,
                        "dx_m": d.dx_m, "dt_s": d.dt_s, "cells": cells,
                        "run_h": run_s / 3600.0, "steps": steps,
                        "cell_steps": cells * steps,
                        "radiation_factor": factor})
    out = next((d for d in domains if d["name"] == design.output_domain), None)
    if out is None:
        raise EstimateRefusal(f"output domain {design.output_domain!r} is "
                              "not in the design")
    frames = math.floor(track.duration_h * 3600.0 / design.output_interval_s
                        + 1e-9) + 1
    frame = output_frame_bytes(design, out["nx"], out["ny"])
    lo, hi = design.compression
    return {
        "domains": domains,
        "cells": sum(d["cells"] for d in domains),
        "cell_steps": sum(d["cell_steps"] for d in domains),
        "ordinary_cell_steps": sum(d["cell_steps"] * d["radiation_factor"]
                                   for d in domains),
        "output": {"domain": out["name"], "frames": frames,
                   "frame_raw_bytes": frame, "raw_bytes": frames * frame,
                   "stored_bytes": [frames * frame * lo, frames * frame * hi]},
    }


def track_on_card(work: dict, c: dict, o: Mapping[str, float]) -> dict:
    """Wall hours (band), peak memory and whether the tree fits ``c``."""
    peak_gib = (CALIBRATION["fixed_gib"]
                + work["cells"] * CALIBRATION["bytes_per_cell"] / GIB)
    fits = peak_gib <= c["memory_gib"] - MEMORY_MARGIN_GIB
    overhead_s = (o["prep_intercept_s"]
                  + o["prep_s_per_mcell"] * work["cells"] / 1e6
                  + work["output"]["raw_bytes"] / (o["d2h_gbps"] * 1e9))
    hours = {}
    for label, factor in zip(BANDS, c["band"]):
        compute_s = (work["ordinary_cell_steps"]
                     / (c["cell_steps_per_s"] * factor))
        hours[label] = (compute_s + overhead_s) / 3600.0
    return {"wall_hours": hours, "peak_gib": peak_gib, "fits": fits}


def _price(c: dict, prices: Mapping[str, Any]) -> dict | None:
    row = (prices.get("instances") or {}).get(c["key"])
    if row is None:
        return None
    try:
        per_gpu = float(row["usd_per_hour"]) / int(row["gpus"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise EstimateRefusal(
            f"price table row for {c['key']} needs usd_per_hour and a "
            f"positive gpus count; got {row!r}") from error
    return {**row, "usd_per_gpu_hour": per_gpu}


def estimate_tracks(tracks: Iterable[Track | Mapping[str, Any]],
                    design: TrackDesign | None = None, *,
                    cards: Sequence[str] = DEFAULT_CARDS,
                    prices: Mapping[str, Any] | None = None) -> dict:
    """Per-track and total work, wall hours, memory, bytes and cost."""
    design = design or TrackDesign()
    prices = AWS_PRICES if prices is None else prices
    tracks = [t if isinstance(t, Track) else _track(t, "tracks")
              for t in tracks]
    if not tracks:
        raise EstimateRefusal("no tracks to price")
    names = [t.name for t in tracks]
    if len(set(names)) != len(names):
        raise EstimateRefusal("track names must be unique; repeated: "
                              + ", ".join(sorted({n for n in names
                                                  if names.count(n) > 1})))
    lo, hi = design.compression
    if not 0 < lo <= hi <= 1.5:
        raise EstimateRefusal(f"compression band must be 0 < low <= high "
                              f"<= 1.5; got {design.compression}")
    check_design(design)
    if not isinstance(prices, Mapping) or not isinstance(
            prices.get("instances", {}), Mapping):
        raise EstimateRefusal("the price table must be a JSON object whose "
                              "\"instances\" is an object keyed by card")
    o = overheads()
    card_rows = []
    for k in cards:
        row = card(k)
        if row["key"] not in {c["key"] for c in card_rows}:
            card_rows.append(row)
    if not card_rows:
        raise EstimateRefusal("--cards names no card")
    rows = []
    for t in tracks:
        work = track_work(t, design)
        rows.append({"name": t.name, "duration_h": t.duration_h,
                     "envelope_km": list(t.envelope_km),
                     "max_height_m": t.max_height_m,
                     "qc": dict(t.qc or {}), **work,
                     "cards": {c["key"]: track_on_card(work, c, o)
                               for c in card_rows}})

    storage_tb = [sum(r["output"]["stored_bytes"][i] for r in rows) / 1e12
                  for i in (0, 1)]
    tops = [r["max_height_m"] for r in rows if r["max_height_m"] is not None]
    totals: dict[str, Any] = {
        "tracks": len(rows),
        "track_hours": sum(r["duration_h"] for r in rows),
        "max_height_m": max(tops) if tops else None,
        "qc": _sum_qc(r["qc"] for r in rows),
        "cell_steps": sum(r["cell_steps"] for r in rows),
        "cell_steps_by_domain": {
            d.name: sum(x["cell_steps"] for r in rows for x in r["domains"]
                        if x["name"] == d.name) for d in design.domains},
        "output_frames": sum(r["output"]["frames"] for r in rows),
        "output_raw_tb": sum(r["output"]["raw_bytes"] for r in rows) / 1e12,
        "output_stored_tb": storage_tb,
        "s3_usd_per_month": [tb * 1000 * prices.get("s3_usd_per_gb_month", 0)
                             for tb in storage_tb],
        "cards": {},
    }
    for c in card_rows:
        per = [r["cards"][c["key"]] for r in rows]
        gpu_h = {b: sum(p["wall_hours"][b] for p in per) for b in BANDS}
        price = _price(c, prices)
        totals["cards"][c["key"]] = {
            "name": c["name"], "calibration": c["calibration"],
            "calibrated": c["calibrated"],
            "ordinary_cell_steps_per_s_central": (c["cell_steps_per_s"]
                                                  * c["band"][1]),
            "memory_gib": c["memory_gib"],
            "gpu_hours": gpu_h,
            "longest_track_wall_hours": max(p["wall_hours"]["central"]
                                            for p in per),
            "peak_gib_max": max(p["peak_gib"] for p in per),
            "tracks_that_do_not_fit": [r["name"] for r, p in zip(rows, per)
                                       if not p["fits"]],
            "aws": None if price is None else {
                **price,
                "usd": {b: h * price["usd_per_gpu_hour"]
                        for b, h in gpu_h.items()},
                "label": prices.get("label", "verify before launch")},
        }
    return {
        "schema": SCHEMA,
        "kind": "estimate",
        "design": _design_dict(design),
        "calibration": dict(CALIBRATION),
        "bands": {"calibrated": BAND_CALIBRATED,
                  "uncalibrated": BAND_UNCALIBRATED, "order": list(BANDS)},
        "prices": {k: v for k, v in prices.items() if k != "instances"},
        "assumptions": [
            "one track = one woof process on one card, d01/d02/d03 "
            "stepped in sequence (no concurrency between domains)",
            "d01 is static over the track envelope plus its margin on "
            "every side; d02 and d03 have fixed shapes and move",
            "every domain runs at the calibrated leaf rate per ordinary "
            "cell-step, plus its own radiation steps at the measured "
            "radiation-step ratio",
            "output is the output domain's history preset as the disk "
            "planner prices it (woof.disk_budget), every interval over the "
            "track's duration, compressed by the band",
            f"preparation {o['prep_intercept_s']:g} s + "
            f"{o['prep_s_per_mcell']:g} s per million cells and history "
            f"copies at {o['d2h_gbps']:g} GB/s are charged to the card "
            "(woof.energy.cost's figures)",
            "a card's totals include tracks that do not fit it "
            "(tracks_that_do_not_fit); those need a bigger card",
            "AWS cost = GPU-hours x instance price / GPUs per instance "
            "(8-GPU instances assume 8 tracks share one)",
        ],
        "tracks": rows,
        "totals": totals,
    }


def _sum_qc(counts: Iterable[Mapping[str, int]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for qc in counts:
        for reason, n in qc.items():
            out[reason] = out.get(reason, 0) + int(n)
    return out


def _design_dict(design: TrackDesign) -> dict:
    out = asdict(design)
    out["domains"] = [asdict(d) for d in design.domains]
    return out


# --------------------------------------------------------------------------
# the door
# --------------------------------------------------------------------------
def _fmt_usd(x: float) -> str:
    return f"${x / 1e3:,.1f}k" if x >= 1e3 else f"${x:,.0f}"


def render_table(doc: dict) -> str:
    t = doc["totals"]
    lines = [
        f"{t['tracks']} tracks, {t['track_hours']:.1f} track-hours, "
        f"{t['cell_steps'] / 1e12:,.1f} T cell-steps "
        + "(" + ", ".join(f"{k} {v / 1e12:,.1f} T" for k, v in
                          t["cell_steps_by_domain"].items()) + ")",
        f"history: {t['output_frames']} frames, {t['output_raw_tb']:.1f} TB "
        f"raw, {t['output_stored_tb'][0]:.1f}-{t['output_stored_tb'][1]:.1f}"
        f" TB stored ({doc['design']['output_preset']} preset); S3 "
        f"{_fmt_usd(t['s3_usd_per_month'][0])}-"
        f"{_fmt_usd(t['s3_usd_per_month'][1])} per month",
        "",
        f"{'card':<15} {'GPU-h':>9} {'band':>15} {'peak GiB':>9} "
        f"{'fit':>5} {'AWS on-demand':>26}  calibration",
    ]
    for key, c in t["cards"].items():
        g = c["gpu_hours"]
        nofit = len(c["tracks_that_do_not_fit"])
        aws = c["aws"]
        cost = ("-" if aws is None else
                f"{_fmt_usd(aws['usd']['central'])} "
                f"({_fmt_usd(aws['usd']['optimistic'])}-"
                f"{_fmt_usd(aws['usd']['pessimistic'])})")
        lines.append(
            f"{key:<15} {g['central']:>9,.0f} "
            f"{g['optimistic']:>7,.0f}-{g['pessimistic']:<7,.0f} "
            f"{c['peak_gib_max']:>9.1f} "
            f"{'yes' if not nofit else f'{nofit} no':>5} {cost:>26}  "
            f"{c['calibration']}")
    lines.append("")
    if t["qc"]:
        lines.append("path QC (woof.track): " + ", ".join(
            f"{k} {v}" for k, v in sorted(t["qc"].items())))
    lines.append(f"prices: AWS {doc['prices'].get('region')} "
                 f"{doc['prices'].get('market')}, "
                 f"{doc['prices'].get('as_of')} -- "
                 f"{doc['prices'].get('label')}.  Every figure is an "
                 "ESTIMATE; uncalibrated cards are the A6000 scaled by "
                 "memory bandwidth.")
    return "\n".join(lines)


def design_from_args(args) -> TrackDesign:
    design = TrackDesign()
    doms = list(design.domains)
    if args.d03_n is not None:
        doms[2] = replace(doms[2], nx=args.d03_n, ny=args.d03_n)
    if args.d01_margin_km is not None:
        doms[0] = replace(doms[0], margin_km=args.d01_margin_km)
    return replace(design, domains=tuple(doms), nz=args.nz,
                   output_interval_s=args.output_interval_s,
                   output_preset=args.output_preset,
                   compression=tuple(args.compression))


def load_tracks(args) -> list[Track]:
    if bool(args.tracks) == bool(args.paths):
        raise EstimateRefusal(layered(
            "give exactly one of --tracks JSON or --paths FILE|DIR ...",
            "--tracks reads ready-made {name, duration_h, envelope_km} rows "
            "or Track3D documents; --paths reads 3-D path files by the "
            "column names given."))
    if args.tracks:
        return load_tracks_json(args.tracks)
    from woof.track.cli import mapping_from_args

    try:
        mapping = mapping_from_args(args)
    except (ValueError, OSError) as error:
        raise EstimateRefusal(str(error)) from error
    return read_path_files(args.paths, mapping, min_climb_m=args.min_climb)


def run(args) -> dict:
    tracks = select_tracks(load_tracks(args), args.select)
    prices = AWS_PRICES
    if args.prices is not None:
        try:
            prices = json.loads(Path(args.prices).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise EstimateRefusal(f"--prices {args.prices}: {error}") from error
    cards = [c.strip() for c in args.cards.split(",") if c.strip()]
    return estimate_tracks(tracks, design_from_args(args), cards=cards,
                           prices=prices)


def main(args) -> int:
    from woof.explain import explain_enabled, render

    try:
        doc = run(args)
    except EstimateRefusal as error:
        print(render(f"woof estimate-tracks: REFUSED: {error}",
                     explain=explain_enabled(args),
                     command="woof estimate-tracks"), file=sys.stderr)
        return 2
    print(render_table(doc), file=sys.stderr)
    if not args.per_track:
        doc = {**doc, "tracks": [{"name": r["name"]} for r in doc["tracks"]]}
    text = json.dumps(doc, indent=1)
    if args.output:
        try:
            Path(args.output).write_text(text + "\n", encoding="utf-8")
        except OSError as error:
            print(f"woof estimate-tracks: REFUSED: -o {args.output}: {error}",
                  file=sys.stderr)
            return 2
    else:
        print(text)
    return 0


def _pair(text: str) -> tuple[float, float]:
    try:
        lo, hi = (float(v) for v in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected LOW,HIGH") from error
    return lo, hi


def register_cli(subparsers):
    p = subparsers.add_parser(
        "estimate-tracks",
        help="estimate GPU-hours, memory, history bytes and AWS cost of "
             "moving-nest LES runs along a set of tracks",
        description=__doc__.split("\n\n")[0])
    p.add_argument("--tracks", type=Path, metavar="JSON",
                   help="JSON list of {name, duration_h, envelope_km: [ew, "
                        "ns]} rows or Track3D documents (or {\"tracks\": "
                        "[...]}, or one Track3D document)")
    p.add_argument("--paths", type=Path, nargs="+", metavar="FILE|DIR",
                   help="3-D path files (.csv, .tsv, .json, .parquet), one "
                        "track each, read by woof.track (a directory "
                        "contributes the files directly inside it)")
    # The column options are `woof track inspect`'s, from the same helper
    # and read back by the same woof.track.cli.mapping_from_args.
    from woof.track.cli import add_mapping_arguments

    add_mapping_arguments(p)
    p.add_argument("--select", metavar="PATTERNS",
                   help="comma-separated shell patterns on track name")
    p.add_argument("--cards", default=",".join(DEFAULT_CARDS), metavar="LIST",
                   help="comma-separated cards to price (default: every "
                        "card in DEFAULT_CARDS, A6000 first)")
    p.add_argument("--nz", type=int, default=TrackDesign.nz, metavar="N",
                   help="mass levels on every domain (default: 256)")
    p.add_argument("--d03-n", type=int, metavar="N",
                   help="d03 columns per side (default: 325)")
    p.add_argument("--d01-margin-km", type=float, metavar="KM",
                   help="d01 margin on every side of the track envelope "
                        "(default: 60)")
    p.add_argument("--output-interval-s", type=float,
                   default=TrackDesign.output_interval_s, metavar="S",
                   help="d03 history interval in seconds (default: 120)")
    p.add_argument("--output-preset", default=TrackDesign.output_preset,
                   metavar="PRESET",
                   help="[output] history preset the d03 frames are priced "
                        "with, by the disk planner (default: balloon)")
    p.add_argument("--compression", type=_pair, metavar="LOW,HIGH",
                   default=TrackDesign.compression,
                   help="stored/raw compression band (default: 0.26,0.40)")
    p.add_argument("--prices", type=Path, metavar="JSON",
                   help="price table with the shape of "
                        "woof.track_estimate.AWS_PRICES")
    p.add_argument("--per-track", action="store_true",
                   help="keep every track's rows in the JSON")
    p.add_argument("-o", "--output", type=Path, metavar="PATH",
                   help="write the JSON here instead of stdout")
    p.set_defaults(func=main)
    return p
