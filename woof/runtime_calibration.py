"""Measure a configuration on THIS runner, and extrapolate the wall.

``woof estimate CONFIG --calibrate N`` is the honest answer for a card
nobody has timed: it launches the real run (``woof go``, or ``woof go
--prepared-root`` when a bundle is given), reads the per-step stream the
runner already writes (``progress.jsonl``, :mod:`woof.progress_log`),
discards ``--warmup`` steps (the kernel compile and first-touch steps),
measures ``N`` more, stops the run it started, and extrapolates.  The
report quotes the measurement against the prediction, and the result is
stored as a LOCAL calibration (:func:`calibration_path`, never in the
repository) that the next ``woof estimate`` on this runner prices from,
labelled ``calibrated``.

THE EXTRAPOLATION (:func:`summarize_steps`, :func:`extrapolate`).  The
per-root-step wall is the gap between consecutive root ``step`` records'
``emitted_unix_ms`` -- so a nest's steps and a history write landing on
the step are inside it, as they are inside the wall the user waits.  A
measurement spanning at least one whole radiation period is averaged
over whole periods, which prices the radiation steps at their measured
cost; a shorter one prices them from the settled median with the
estimator's measured radiation-step ratio bracket.

The same arithmetic drives the live ETA a forecast logs
(:func:`eta_seconds`, called by :class:`woof.progress_log.StepLog`):
remaining model seconds at the measured model-seconds-per-wall-second
since the run settled.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

CALIBRATION_SCHEMA = "woof.runtime-calibration.v1"

#: A step this many times the settled median is a radiation (or
#: output) step when the estimator counts them.
OUTLIER_FACTOR = 2.0


def calibration_path() -> Path:
    """Where local calibrations live: ``$WOOF_RUNTIME_CALIBRATION``, else
    ``$XDG_CACHE_HOME/woof/runtime-calibration.json`` (``~/.cache``)."""
    override = os.environ.get("WOOF_RUNTIME_CALIBRATION", "").strip()
    if override:
        return Path(override).expanduser()
    root = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(root).expanduser() / "woof" / "runtime-calibration.json"


def load_calibrations(path: Path | None = None) -> list[dict[str, Any]]:
    path = calibration_path() if path is None else Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    rows = document.get("calibrations") if isinstance(document, dict) else None
    return [r for r in rows or [] if isinstance(r, dict)]


def record_calibration(entry: Mapping[str, Any], path: Path | None = None
                       ) -> Path:
    """Append ``entry``; keep the newest 50."""
    path = calibration_path() if path is None else Path(path)
    rows = load_calibrations(path)
    rows.append(dict(entry))
    rows = rows[-50:]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"schema": CALIBRATION_SCHEMA,
                               "calibrations": rows}, indent=2),
                   encoding="utf-8")
    os.replace(tmp, path)
    return path


def _runner_key(runner) -> str:
    return str(runner.card_key or runner.name)


def matching_calibration(runner, *, route: str, config: str | Path | None,
                         rung: str | None = None,
                         path: Path | None = None) -> dict[str, Any] | None:
    """The newest stored calibration for this runner, route and rung.

    ``rung`` is read from ``config`` when not given.  A calibration is
    per card model, route and physics rung: it prices any grid of that
    rung (per column-step at nz=49), not only the configuration it ran.
    """
    if rung is None and config is not None:
        try:
            from tilestream import autoplan

            from woof.experiment import load_experiment

            rung = autoplan.rung_of(load_experiment(config).domains[0].run)
        except Exception:  # noqa: BLE001 - unreadable: no match
            return None
    key = _runner_key(runner)
    for row in reversed(load_calibrations(path)):
        if (row.get("runner_key") == key and row.get("route") == route
                and row.get("rung") == rung
                and row.get("backend") == runner.backend):
            return row
    return None


# --------------------------------------------------------------------------
# reading the step stream
# --------------------------------------------------------------------------
def read_progress(path: Path) -> list[dict[str, Any]]:
    records = []
    try:
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except ValueError:
                    continue  # a line being written
    except OSError:
        return []
    return records


def root_steps(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The root domain's ``step`` records, in order."""
    steps = [dict(r) for r in records if r.get("event") == "step"]
    if not steps:
        return []
    root = min(int(r.get("domain", 1)) for r in steps)
    return [r for r in steps if int(r.get("domain", 1)) == root]


@dataclass(frozen=True)
class StepSummary:
    measured_steps: int
    median_s: float
    mean_s: float
    whole_period_mean_s: float | None
    radiation_ratio: float | None
    walls: tuple[float, ...]


def step_walls(steps: Sequence[Mapping[str, Any]], warmup: int
               ) -> list[float]:
    """Per-root-step wall after ``warmup`` steps, from record stamps.

    Falls back to ``step_wall_seconds`` where the stamps do not advance
    (a log thinned by ``--progress-every`` has gaps of several steps,
    which are divided out)."""
    walls = []
    for prev, cur in zip(steps, steps[1:]):
        if int(cur.get("step", 0)) <= warmup:
            continue
        gap = int(cur.get("step", 0)) - int(prev.get("step", 0))
        dt_ms = (cur.get("emitted_unix_ms") or 0) - (
            prev.get("emitted_unix_ms") or 0)
        if gap >= 1 and dt_ms > 0:
            walls.extend([dt_ms / 1000.0 / gap] * gap)
        else:
            walls.append(float(cur.get("step_wall_seconds", 0.0) or 0.0))
    return walls


def summarize_steps(walls: Sequence[float], *, steps_per_radiation: int | None
                    ) -> StepSummary:
    if not walls:
        raise ValueError("no settled steps were measured")
    median = statistics.median(walls)
    mean = statistics.fmean(walls)
    whole = None
    if steps_per_radiation and len(walls) >= steps_per_radiation:
        n = (len(walls) // steps_per_radiation) * steps_per_radiation
        whole = statistics.fmean(walls[:n])
    slow = [w for w in walls if median > 0 and w > OUTLIER_FACTOR * median]
    ratio = (statistics.fmean(slow) / median) if slow and median > 0 else None
    return StepSummary(len(walls), median, mean, whole, ratio, tuple(walls))


def extrapolate(summary: StepSummary, *, steps: int, radiation_fraction: float,
                ratio_bracket: tuple[float, float]) -> tuple[float, float, float]:
    """(central, low, high) forecast-loop wall for ``steps`` root steps.

    Whole radiation periods measured: their mean prices every step
    (central = low = high).  Otherwise the settled median prices the
    ordinary steps and radiation steps cost the measured ratio if one
    was seen, else the bracket.
    """
    if summary.whole_period_mean_s is not None:
        total = summary.whole_period_mean_s * steps
        return total, total, total
    base = summary.median_s * steps
    if radiation_fraction <= 0:
        return base, base, base
    if summary.radiation_ratio is not None:
        ratios = (summary.radiation_ratio,) * 2
    else:
        ratios = ratio_bracket
    lo = base * (1 + (ratios[0] - 1) * radiation_fraction)
    hi = base * (1 + (ratios[1] - 1) * radiation_fraction)
    return math.sqrt(lo * hi), lo, hi


def eta_seconds(*, model_seconds: float, run_seconds: float,
                anchor_wall: float, anchor_model: float,
                now_wall: float) -> float | None:
    """Wall seconds left at the pace measured since ``anchor``.

    ``None`` until the run has advanced past its anchor in both model
    and wall time.  The live ETA on ``woof go``'s forecast heartbeat
    (:class:`LiveEta`) and ``--calibrate`` both extrapolate this way: remaining model
    seconds over measured model seconds per wall second.
    """
    model_done = model_seconds - anchor_model
    wall_done = now_wall - anchor_wall
    if model_done <= 0 or wall_done <= 0:
        return None
    remaining = max(0.0, float(run_seconds) - float(model_seconds))
    return remaining * wall_done / model_done


class LiveEta:
    """The ETA a running forecast's heartbeat line carries.

    Fed the forecast's own ``progress.json`` payload each heartbeat (the
    heartbeat cadence is the rate limit).  The first payload with model
    time on the clock is the anchor -- set-up and the warm first steps
    are behind it, so the pace is the stepping pace -- and every later
    one extrapolates with :func:`eta_seconds`.  Empty until there is a
    pace to extrapolate, and for a payload without model time.
    """

    def __init__(self, run_seconds: float | None) -> None:
        self.run_seconds = (float(run_seconds)
                            if run_seconds and run_seconds > 0 else None)
        self.anchor: tuple[float, float] | None = None

    def note(self, payload: Mapping | None, now_wall: float) -> str:
        if self.run_seconds is None or not isinstance(payload, Mapping):
            return ""
        model = payload.get("model_elapsed_seconds")
        if not isinstance(model, (int, float)) or model <= 0:
            return ""
        if self.anchor is None:
            self.anchor = (float(now_wall), float(model))
            return ""
        left = eta_seconds(model_seconds=float(model),
                           run_seconds=self.run_seconds,
                           anchor_wall=self.anchor[0],
                           anchor_model=self.anchor[1], now_wall=now_wall)
        if left is None:
            return ""
        from woof.runtime_estimate import fmt_duration
        pct = min(100.0, 100.0 * float(model) / self.run_seconds)
        return f", {pct:.0f}% of the forecast, ETA ~{fmt_duration(left)}"


# --------------------------------------------------------------------------
# running it
# --------------------------------------------------------------------------
def _descendants(pid: int) -> list[int]:
    """Every descendant PID of ``pid`` (Linux ``/proc``; ``pgrep -P``
    elsewhere).  Ours, and only ours, are killed."""
    out: list[int] = []
    frontier = [pid]
    while frontier:
        parent = frontier.pop()
        children: list[int] = []
        try:
            text = subprocess.run(["pgrep", "-P", str(parent)],
                                  capture_output=True, text=True,
                                  timeout=5).stdout
            children = [int(x) for x in text.split() if x.isdigit()]
        except (OSError, subprocess.TimeoutExpired):
            pass
        out.extend(children)
        frontier.extend(children)
    return out


def stop_process_tree(process: subprocess.Popen, *, grace_s: float = 30.0
                      ) -> None:
    """SIGTERM the run this module started and every descendant, then
    SIGKILL what remains after ``grace_s``."""
    if process.poll() is not None:
        return
    pids = [process.pid, *_descendants(process.pid)]
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline and process.poll() is None:
        time.sleep(0.5)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def go_command(config: Path, outdir: Path, *, prepared_root: Path | None
               ) -> list[str]:
    command = [sys.executable, "-m", "woof", "go", str(config),
               "--outdir", str(outdir), "--products", "none"]
    if prepared_root is not None:
        command += ["--prepared-root", str(prepared_root)]
    return command


def measure(config: Path, *, steps: int, warmup: int, workdir: Path,
            prepared_root: Path | None = None, timeout_s: float = 3600.0,
            launch: Callable[..., subprocess.Popen] = subprocess.Popen,
            clock: Callable[[], float] = time.monotonic,
            sleep: Callable[[float], None] = time.sleep,
            poll_s: float = 2.0) -> dict[str, Any]:
    """Run ``config`` until ``warmup + steps`` root steps, then stop it.

    Returns the root step records, the launch-to-first-step wall and the
    command.  ``launch``/``clock``/``sleep`` are for tests.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    command = go_command(config, workdir / "run", prepared_root=prepared_root)
    log = open(workdir / "go.log", "w", encoding="utf-8")
    started = clock()
    process = launch(command, stdout=log, stderr=subprocess.STDOUT,
                     cwd=str(workdir))
    first_step_at = None
    progress: Path | None = None
    records: list[dict[str, Any]] = []
    reason = "steps measured"
    try:
        while True:
            if progress is None:
                found = sorted((workdir / "run").rglob("progress.jsonl"))
                progress = found[0] if found else None
            if progress is not None:
                records = root_steps(read_progress(progress))
                if records and first_step_at is None:
                    first_step_at = clock()
                if len(records) >= warmup + steps + 1:
                    break
            if process.poll() is not None:
                if progress is not None:
                    records = root_steps(read_progress(progress))
                reason = f"the run exited ({process.returncode}) first"
                break
            if clock() - started > timeout_s:
                reason = f"timed out after {timeout_s:g} s"
                break
            sleep(poll_s)
    finally:
        stop_process_tree(process)
        log.close()
    return {"command": command, "records": records, "reason": reason,
            "launch_to_first_step_s": (None if first_step_at is None
                                       else first_step_at - started),
            "log": str(workdir / "go.log")}


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def report_from_records(records: Sequence[Mapping[str, Any]], *,
                        estimate, warmup: int,
                        launch_to_first_step_s: float | None = None
                        ) -> dict[str, Any]:
    """Measured against predicted, from root step records (pure)."""
    from woof.core.pace import REFERENCE_NZ
    from woof.runtime_estimate import RADIATION_STEP_RATIO

    root = estimate.domains[0]
    period = (None if root.radt_min is None else
              max(1, round(root.radt_min * 60.0 / root.dt_s)))
    walls = step_walls(records, warmup)
    summary = summarize_steps(walls, steps_per_radiation=period)
    central, low, high = extrapolate(
        summary, steps=root.steps, radiation_fraction=root.radiation_fraction,
        ratio_bracket=RADIATION_STEP_RATIO)
    predicted = [p for p in estimate.phases
                 if p.name in ("model steps", "radiation steps", "output")]
    pred_lo = sum(p.low_s for p in predicted)
    pred_hi = sum(p.high_s for p in predicted)
    # Columns one root step carries, nests included at their step ratio.
    weighted = sum(d.columns * d.nz / REFERENCE_NZ * d.steps / root.steps
                   for d in estimate.domains)
    ratio = summary.radiation_ratio
    ordinary = summary.median_s / weighted
    return {
        "warmup_steps": warmup,
        "measured_steps": summary.measured_steps,
        "measured_seconds_per_step": summary.median_s,
        "measured_mean_seconds_per_step": summary.mean_s,
        "whole_radiation_period_mean_s": summary.whole_period_mean_s,
        "measured_radiation_step_ratio": ratio,
        "launch_to_first_step_s": launch_to_first_step_s,
        "extrapolated_forecast_s": central,
        "extrapolated_forecast_low_s": low,
        "extrapolated_forecast_high_s": high,
        "predicted_forecast_low_s": pred_lo,
        "predicted_forecast_high_s": pred_hi,
        "inside_bracket": pred_lo <= central <= pred_hi,
        "ordinary_s_per_column_step_nz49": ordinary,
        "predicted_provenance": estimate.rate.provenance,
    }


def calibrate(config, *, steps: int, warmup: int, estimate, runner,
              prepared_root: Path | None = None, record: bool = True,
              timeout_s: float = 3600.0, workdir: Path | None = None,
              keep: bool = False, path: Path | None = None,
              measure_fn: Callable[..., dict[str, Any]] = measure
              ) -> dict[str, Any]:
    """Measure, report, and (``record``) store a local calibration."""
    config = Path(config)
    own_dir = workdir is None
    workdir = Path(workdir or tempfile.mkdtemp(prefix="woof-calibrate-"))
    try:
        run = measure_fn(config, steps=steps, warmup=warmup, workdir=workdir,
                         prepared_root=prepared_root, timeout_s=timeout_s)
    finally:
        if own_dir and not keep:
            import shutil

            shutil.rmtree(workdir, ignore_errors=True)
    if len(run["records"]) <= warmup + 1:
        raise ValueError(f"--calibrate measured no settled step: "
                         f"{run['reason']} (see {run.get('log')})")
    report = report_from_records(
        run["records"], estimate=estimate, warmup=warmup,
        launch_to_first_step_s=run.get("launch_to_first_step_s"))
    report["stop_reason"] = run["reason"]
    report["command"] = " ".join(run["command"])
    if record:
        entry = {
            "runner_key": _runner_key(runner), "card": runner.name,
            "backend": runner.backend, "route": estimate.route,
            "rung": estimate.domains[0].rung,
            "config": str(config), "config_sha256": _sha256(config),
            "recorded_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds"),
            "steps": report["measured_steps"], "warmup": warmup,
            "ordinary_s_per_column_step_nz49":
                report["ordinary_s_per_column_step_nz49"],
            "radiation_step_ratio": report["measured_radiation_step_ratio"],
            "measured_seconds_per_step": report["measured_seconds_per_step"],
            "launch_to_first_step_s": report["launch_to_first_step_s"],
        }
        report["recorded_to"] = str(record_calibration(entry, path))
    return report


__all__ = [
    "CALIBRATION_SCHEMA", "LiveEta", "OUTLIER_FACTOR", "StepSummary",
    "calibrate",
    "calibration_path", "eta_seconds", "extrapolate", "go_command",
    "load_calibrations", "matching_calibration", "measure",
    "read_progress", "record_calibration", "report_from_records",
    "root_steps", "step_walls", "stop_process_tree", "summarize_steps",
]
