#!/usr/bin/env python3
"""Throughput of a woof run and of Breeze's runs, in the same units.

Units, for both models:

* ``sim_hours_per_wall_hour`` -- simulated time per wall time (the
  "x realtime" factor);
* ``cell_steps_per_second`` -- grid cells x model time steps per wall
  second, which divides out the time step and so compares the cost of one
  step of one cell.  It is NOT a work-equivalence: a woof step at 12 km is
  a WRF RK3 step with acoustic substeps and a column of physics, a Breeze
  step is a Wicker-Skamarock RK3 step with its own adaptive substep count,
  and the two take very different time steps (woof ~60 s, Breeze <= 10 s).

Three kinds of number come out of here and each is labelled:

* ``measured`` -- read from a run on this machine (woof's
  ``progress.jsonl``, or the JSON the Breeze timing driver writes);
* ``documented`` -- a figure a Breeze/NumericalEarth source states, see
  :mod:`breeze_cases`;
* ``estimate`` -- a documented figure scaled to another card by the ratio
  of peak DRAM bandwidth.  Both codes are memory-bound stencil codes at
  this size, so bandwidth is the first-order scaling, and it ignores
  clocks, caches, launch overhead and the shared-card contention every
  measured number on this machine carries.

Usage::

    python tools/compare/era5_nest/nest_throughput.py woof RUN_DIR --case mc3e
    python tools/compare/era5_nest/nest_throughput.py breeze BREEZE.json --case mc3e
    python tools/compare/era5_nest/nest_throughput.py published --case ar \\
        --scale-to rtx-a6000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import breeze_cases as cases  # noqa: E402

#: Peak DRAM bandwidth, GB/s, from the vendors' data sheets.
BANDWIDTH_GBPS = {
    "a100-40gb": 1555.0,   # NVIDIA A100 40GB PCIe/SXM data sheet
    "a100-80gb": 2039.0,   # A100 80GB SXM
    "rtx-a6000": 768.0,    # NVIDIA RTX A6000 data sheet
    "h100-sxm": 3350.0,
}

#: Steps excluded from the steady-state window by default: the first steps
#: carry kernel compilation and allocator warm-up.
DEFAULT_WARMUP_STEPS = 10


#: The run-plan event stream `woof go` writes in every run folder.
RUN_PLAN_SCHEMA = "gpuwm.run-plan.event.v1"


def _records(path: Path) -> list[dict]:
    """Records of a woof step log or run-plan event stream."""

    lines = [json.loads(line) for line in path.read_text().splitlines()
             if line.strip()]
    if lines and lines[0].get("schema_version") == RUN_PLAN_SCHEMA:
        return lines
    try:
        from woof.progress_log import read_step_log
    except ImportError:  # pragma: no cover - the tool runs from the repo
        return lines
    return read_step_log(path)


def find_progress_log(run: Path) -> Path:
    """The step timing of one run: ``progress.jsonl`` (``woof run``/``sim``
    with a step log) or else the run folder's ``events.jsonl`` (``woof go``),
    whose ``model_progress`` events carry the same per-step clock."""

    if run.is_file():
        return run
    for name in ("progress.jsonl", "events.jsonl"):
        direct = run / name
        if direct.is_file():
            return direct
    for name in ("progress.jsonl", "events.jsonl"):
        found = sorted(run.rglob(name))
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            raise FileNotFoundError(
                f"{len(found)} {name} files under {run}; name one run folder")
    raise FileNotFoundError(f"no progress.jsonl or events.jsonl under {run}")


def _step_points(records: list[dict], domain: int) -> list[dict]:
    """(step, model_seconds, wall_seconds) per step, from either stream.

    The step log stamps each record's emission (``emitted_unix_ms``); the
    run-plan stream's ``model_progress`` carries ``wall_seconds`` since the
    forecast stage began, measured after the step's device sync.
    """

    points = []
    for r in records:
        if r.get("domain") != domain:
            continue
        if r.get("event") == "step":
            points.append({"step": r["step"],
                           "model_seconds": r["model_seconds"],
                           "wall": r["emitted_unix_ms"] / 1000.0})
        elif r.get("event") == "model_progress" and r.get("outer_step", 0) > 0:
            points.append({"step": r["outer_step"],
                           "model_seconds": r["model_seconds"],
                           "wall": r["wall_seconds"]})
    return points


def stage_seconds(records: list[dict]) -> dict[str, float]:
    """Wall seconds of each finished ``woof go`` stage (prepare, forecast...)."""

    return {r["stage"]: r["wall_seconds"] for r in records
            if r.get("event") == "stage_finished" and "wall_seconds" in r}


def _rates(*, cells: int, n_steps: int, model: float, wall: float) -> dict:
    """The common units from one timed window."""

    if wall <= 0.0 or n_steps <= 0:
        raise ValueError("the timing window is empty")
    return {
        "cells": cells,
        "window_steps": n_steps,
        "window_model_seconds": model,
        "window_wall_seconds": wall,
        "mean_dt_seconds": model / n_steps,
        "seconds_per_step": wall / n_steps,
        "sim_hours_per_wall_hour": model / wall,
        "cell_steps_per_second": cells * n_steps / wall,
    }


def woof_throughput(records: list[dict], *, cells: int, domain: int = 1,
                    warmup_steps: int = DEFAULT_WARMUP_STEPS) -> dict:
    """Throughput of one domain from its step records.

    The steady window runs from the first step at or past ``warmup_steps``
    to the last step; both ends are wall clocks the log stamped, so the
    window includes everything the run did between them -- physics,
    radiation, boundary updates and history writes -- and nothing before
    (preparation, kernel compile) or after (draining writers, rendering).
    """

    steps = _step_points(records, domain)
    if len(steps) < 2:
        raise ValueError(f"domain {domain} has {len(steps)} step records; "
                         "need two to time a window")
    run_end = next((r for r in records if r.get("event") == "run_end"), None)
    start = next((r for r in steps if r["step"] >= warmup_steps), None)
    steady = start is not None and start is not steps[-1]
    if not steady:
        start = steps[0]
    end = steps[-1]
    result = {
        "kind": "measured",
        "model": "woof",
        "window_first_step": start["step"],
        "window_last_step": end["step"],
        "steady_window": steady,
        **_rates(cells=cells, n_steps=end["step"] - start["step"],
                 model=end["model_seconds"] - start["model_seconds"],
                 wall=end["wall"] - start["wall"]),
        "total_steps": end["step"],
        "total_model_seconds": end["model_seconds"],
    }
    if not steady:
        result["note"] = (f"the run has no step past the {warmup_steps}-step "
                          "warm-up, so the window starts at step 1 and "
                          "includes kernel compile and allocator warm-up")
    stages = stage_seconds(records)
    if stages:
        result["stage_wall_seconds"] = stages
        if stages.get("forecast"):
            result["forecast_stage_sim_hours_per_wall_hour"] = (
                end["model_seconds"] / stages["forecast"])
    if run_end is not None:
        result["run_status"] = run_end.get("status")
        result["run_wall_seconds"] = run_end.get("wall_seconds")
        result["first_step_excess_seconds"] = run_end.get(
            "first_step_excess_seconds")
        if run_end.get("wall_seconds"):
            result["run_sim_hours_per_wall_hour"] = (
                end["model_seconds"] / run_end["wall_seconds"])
    return result


def breeze_throughput(summary: dict, *, cells: int) -> dict:
    """Same quantities from the Breeze timing driver's JSON summary.

    The driver records the iteration count, simulated seconds and wall
    seconds of its steady window (after its own warm-up iterations).
    """

    result = {
        "kind": "measured",
        "model": "breeze",
        **_rates(cells=cells, n_steps=summary["window_iterations"],
                 model=summary["window_model_seconds"],
                 wall=summary["window_wall_seconds"]),
    }
    if summary.get("warmup_iterations", 1) == 0:
        result["steady_window"] = False
        result["note"] = ("the run ended before its warm-up count, so the "
                          "window includes JIT compilation")
    return result


#: The A100 variants a source that says only "A100" may have meant.
A100_VARIANTS = ("a100-40gb", "a100-80gb")


def published_throughput(case: cases.Case, timing: cases.PublishedTiming,
                         *, scale_to: str | None = None,
                         assume_card: str | None = None) -> dict:
    """A documented Breeze figure in the common units, optionally scaled.

    A per-iteration figure has no simulated time of its own; the mean step
    of the run it belongs to is borrowed from the case's whole-run figure
    and the result says so.
    """

    seconds_per_step = timing.seconds_per_iteration
    mean_dt = None
    dt_note = None
    whole_runs = [t for t in case.published
                  if t.iterations and t.simulated_hours and t.wall_hours]
    if timing.iterations and timing.simulated_hours and timing.wall_hours:
        seconds_per_step = timing.wall_hours * 3600.0 / timing.iterations
        mean_dt = timing.simulated_hours * 3600.0 / timing.iterations
    elif whole_runs:
        reference = whole_runs[0]
        mean_dt = reference.simulated_hours * 3600.0 / reference.iterations
        dt_note = ("mean step borrowed from the whole-run figure "
                   f"({reference.simulated_hours:g} h in "
                   f"{reference.iterations} iterations)")
    result: dict = {
        "kind": "documented",
        "model": "breeze",
        "hardware": timing.hardware,
        "gpus": timing.gpus,
        "cells": case.cells,
        "source": timing.source,
        "with_output": timing.with_output,
    }
    if timing.simulated_hours and timing.wall_hours:
        result["sim_hours_per_wall_hour"] = (timing.simulated_hours
                                             / timing.wall_hours)
    if seconds_per_step is not None:
        result["seconds_per_step"] = seconds_per_step
        result["cell_steps_per_second"] = case.cells / seconds_per_step
        if mean_dt is not None:
            result["mean_dt_seconds"] = mean_dt
            result.setdefault("sim_hours_per_wall_hour",
                              mean_dt / seconds_per_step)
    if dt_note:
        result["note"] = dt_note
    if scale_to is not None:
        source_card = timing.card or assume_card
        if source_card is None:
            raise ValueError(
                f"{timing.hardware!r} does not say which card variant it was; "
                "pass assume_card to scale it, and label the assumption")
        result = scale_by_bandwidth(result, source_card=source_card,
                                    target_card=scale_to)
        if timing.card is None:
            result["assumed_source_card"] = source_card
    return result


def scale_by_bandwidth(result: dict, *, source_card: str,
                       target_card: str) -> dict:
    """Scale a rate by the ratio of peak DRAM bandwidth (an estimate)."""

    ratio = BANDWIDTH_GBPS[target_card] / BANDWIDTH_GBPS[source_card]
    scaled = dict(result)
    scaled["kind"] = "estimate"
    scaled["scaled_from"] = result.get("hardware", source_card)
    scaled["hardware"] = target_card
    scaled["bandwidth_ratio"] = ratio
    for key in ("sim_hours_per_wall_hour", "cell_steps_per_second"):
        if key in scaled:
            scaled[key] = scaled[key] * ratio
    if "seconds_per_step" in scaled:
        scaled["seconds_per_step"] = scaled["seconds_per_step"] / ratio
    return scaled


def equal_step_wall_hours(seconds_per_step: float, iterations: int) -> float:
    """Wall hours a model would need to take another model's step count.

    Used to separate "cheaper per step" from "takes fewer steps": woof at
    Breeze's 47,600 AR iterations, for instance.
    """

    return seconds_per_step * iterations / 3600.0


def format_row(label: str, result: dict) -> str:
    def number(key, fmt):
        value = result.get(key)
        return "--" if value is None else format(value, fmt)

    return (f"| {label} | {result['kind']} | "
            f"{number('sim_hours_per_wall_hour', '.1f')} | "
            f"{number('cell_steps_per_second', '.3g')} | "
            f"{number('seconds_per_step', '.3f')} | "
            f"{number('mean_dt_seconds', '.1f')} |")


TABLE_HEADER = ("| run | kind | simulated h / wall h | cell-steps/s | "
                "s / step | mean dt (s) |\n|---|---|---|---|---|---|")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    woof = sub.add_parser("woof", help="a woof run directory or progress.jsonl")
    woof.add_argument("run", type=Path)
    woof.add_argument("--case", choices=sorted(cases.CASES), required=True)
    woof.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    woof.add_argument("--cells", type=int,
                      help="cells actually run (default: the case's; a "
                           "--shrink grid needs its own count)")
    breeze = sub.add_parser("breeze", help="the Breeze timing driver's JSON")
    breeze.add_argument("summary", type=Path)
    breeze.add_argument("--case", choices=sorted(cases.CASES), required=True)
    breeze.add_argument("--cells", type=int,
                        help="cells actually run (default: the case's)")
    published = sub.add_parser("published", help="the documented figures")
    published.add_argument("--case", choices=sorted(cases.CASES),
                           required=True)
    published.add_argument("--scale-to", choices=sorted(BANDWIDTH_GBPS))
    parser.add_argument("--json", action="store_true",
                        help="print JSON instead of a markdown table")
    args = parser.parse_args(argv)
    case = cases.CASES[args.case]
    rows: list[tuple[str, dict]] = []
    if args.mode == "woof":
        log = find_progress_log(args.run)
        rows.append((f"woof {case.key}", woof_throughput(
            _records(log), cells=args.cells or case.cells,
            warmup_steps=args.warmup_steps)))
    elif args.mode == "breeze":
        summary = json.loads(args.summary.read_text())
        rows.append((f"Breeze {case.key}", breeze_throughput(
            summary, cells=args.cells or summary.get("cells") or case.cells)))
    else:
        for timing in case.published:
            label = f"Breeze {case.key}, {timing.gpus}x {timing.hardware}"
            rows.append((label, published_throughput(case, timing)))
            # One card's bandwidth says nothing about a two-card run.
            if not args.scale_to or timing.gpus != 1:
                continue
            if timing.card is not None:
                rows.append((f"{label} -> {args.scale_to}",
                             published_throughput(case, timing,
                                                  scale_to=args.scale_to)))
                continue
            # The source names no variant: one row per variant it may be.
            for variant in A100_VARIANTS:
                rows.append((f"{label} -> {args.scale_to} (if {variant})",
                             published_throughput(case, timing,
                                                  scale_to=args.scale_to,
                                                  assume_card=variant)))
    if args.json:
        print(json.dumps([{"label": label, **row} for label, row in rows],
                         indent=2))
    else:
        print(TABLE_HEADER)
        for label, row in rows:
            print(format_row(label, row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
