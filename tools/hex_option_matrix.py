"""Hex forecast smoke matrix: every generic-seam column option through a real
``woof hex forecast`` (hex parity plan item 1.3).

WHAT THIS IS.  ``tests/mpas_column_physics_cases.selections()`` is the menu of
WRF column options the R1 generic seam accepts (one option changed at a time
from the frozen default, with WRF's own pairing partners).  The column sweep
(tests/test_mpas_column_physics_gpu.py) runs each one for 3 steps on synthetic
columns.  This tool runs each one as a FORECAST on a staged hex case, through
the forecast door's own subprocess, and records per row:

- the outcome: ``ran``, ``refused`` (with the door's named reason; a missing
  per-cell input of the surface-input contract is ``refused (needs X)``),
  ``non-finite``, ``budget`` or ``error``;
- finiteness of every floating history field in every frame;
- the dry-mass and water budgets over the history window (below);
- the receipt's resolved physics: the seam, the sealed engine assignment and
  the per-scheme call counts;
- wall time.

THE CASE.  Nothing site-specific is in this file.  ``--case CASE.json`` names
the staged inputs:

    {"forecast_args": ["--mesh", "...", "--grid", "...", "--static", "...",
                       "--init", "...", "--lbc-dir", "...",
                       "--init-source", "...", "--start-time", "...",
                       "--dt", "row", "--experimental-dt"],
     "static": "<the static file, for areaCell and bdyMaskCell>",
     "suite": "arwen_mpas_column",
     "base_physics": {"config_gwdo_scheme": "off"},
     "env": {"WOOF_HEX_MESH_ROWS": "..."},
     "hours": 1,
     "label": "a description without paths, for the results"}

``base_physics`` are ``--physics`` assignments every row carries (a row's own
assignment of the same engine key wins; a drag row drops a base
``config_gwdo_scheme``).

BUDGETS.  On a limited-area case neither budget closes by itself: the lateral
boundaries carry mass and water in and out.  The tool therefore reports the
raw terms and grades each row AGAINST THE BASELINE ROW of the same backend
(the frozen default selection, ``microphysics.wsm6-mp6``), which shares the
dycore, the boundaries and the initial state:

- dry mass ``M = sum(areaCell * sum_k rho * dz)`` (MPAS's diagnosed ``rho``
  is the dry-air density); graded on ``dM/M0`` minus the baseline's.  Column
  physics on the R1 seam does not change dry mass, so a row whose relative
  change departs from the baseline's by more than ``DRY_TOL`` moved mass the
  dynamics did not;
- water ``W = sum(areaCell * sum_k rho * (qv+qc+qr+qi+qs+qg+qh) * dz)`` plus
  the accumulated precipitation ``sum(areaCell * (rainc + rainnc))`` (kg; MPAS
  ``rainnc`` already holds every frozen category), minus the surface
  evaporation estimated from the end-point ``qfx`` (trapezoid); graded on the
  residual relative to ``W0`` minus the baseline's, at ``WATER_TOL``.  Both are
  also reported over interior cells only (``bdyMaskCell == 0``).

Usage (pixi; mac Metal ``-e mlx``, pb CUDA ``-e default``)::

    python tools/hex_option_matrix.py plan
    python tools/hex_option_matrix.py run --case CASE.json --out DIR \\
        --backend cuda [--only NAME ...] [--components microphysics ...]
    python tools/hex_option_matrix.py publish RESULTS.json [...] -o EVIDENCE.json
    python tools/hex_option_matrix.py table EVIDENCE.json

``run`` writes ``DIR/results-<backend>.json`` after every row and skips rows
already recorded there (resumable).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCHEMA = "woof.hex-option-matrix/v1"

#: The selections the column sweep itself leaves out (the dycore owns
#: turbulence on hex: les/horiz-mixing, refused via dycore_owned).
SKIPPED_COMPONENTS = ("turbulence",)

#: The row every other row's budgets are graded against: the frozen default
#: selection (it changes nothing from MPAS_DEFAULT_PHYSICS).
BASELINE = "microphysics.wsm6-mp6"

#: Grading tolerances against the baseline (see the module docstring).
DRY_TOL = 1.0e-3
WATER_TOL = 5.0e-2

WATER_SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg", "qh")


def _cases():
    """tests/mpas_column_physics_cases, the selection menu's one home."""

    tests = str(REPO / "tests")
    if tests not in sys.path:
        sys.path.insert(0, tests)
    import mpas_column_physics_cases as cases  # noqa: PLC0415
    return cases


def _value(value) -> str:
    if isinstance(value, bool):
        return str(int(value))
    return str(value)


def overrides(physics: dict, base: dict) -> dict:
    """The ``--physics`` assignments turning ``base`` into ``physics``.

    ``base`` is the engine's MPAS_DEFAULT_PHYSICS, which the R1 suite
    ``arwen_mpas_column`` resolves to.  A radiation row drops ``ra_physics``
    for its own ``ra_lw_physics``/``ra_sw_physics`` pair, which the resolver
    takes in place of the suite's (it is how the door spells it too).
    """

    return {key: _value(value) for key, value in physics.items()
            if key not in base or base[key] != value}


def plan(components=None) -> list[dict]:
    """One row per selection: ``{"option", "physics", "overrides"}``."""

    cases = _cases()
    base = dict(cases.suite.MPAS_DEFAULT_PHYSICS)
    rows = []
    for name, physics in cases.selections(components).items():
        if name.split(".")[0] in SKIPPED_COMPONENTS:
            continue
        rows.append({"option": name, "physics": physics,
                     "overrides": overrides(physics, base)})
    return rows


def physics_args(row: dict, base_physics: dict) -> list[str]:
    assigned = dict(base_physics)
    if "gwd_opt" in row["overrides"]:
        assigned.pop("config_gwdo_scheme", None)
    assigned.update(row["overrides"])
    out = []
    for key, value in assigned.items():
        out += ["--physics", f"{key}={value}"]
    return out


# ---------------------------------------------------------------------------
# reading a run
# ---------------------------------------------------------------------------
_NEEDS = re.compile(r"reads (.+?)(?: per cell|,| which|;|$)")


def classify_log(text: str, returncode: int) -> dict:
    """The door's verdict from its log: ran, or the named refusal/error."""

    if re.search(r"^DOOR .*status=passed", text, re.M) and returncode == 0:
        return {"outcome": "ran"}
    missing = [line.split("missing surface input:", 1)[1].strip()
               for line in text.splitlines()
               if "missing surface input:" in line]
    if missing:
        needs = []
        for problem in missing:
            match = _NEEDS.search(problem)
            needs.append(match.group(1).strip() if match else problem[:80])
        return {"outcome": "refused", "needs": needs,
                "reason": "; ".join(missing)[:1500]}
    # The regional admission's own status line names a refusal the
    # experimental lane then overrides; it is not the run's verdict.
    lines = [line for line in text.splitlines() if line.strip()
             and not line.startswith("{")
             and not line.startswith("REGIONAL class=")
             and "the hex model (preview)" not in line]
    stopped = [line for line in lines
               if "the forecast driver stopped with" in line]
    nonfinite = [line for line in lines if re.search(
        r"non-finite|nonfinite|NaN|not finite", line)]
    refusal = [line for line in lines if re.search(
        r"refus|not admitted|admits only|cannot run|is not supported|"
        r"unsupported", line, re.I)]
    if nonfinite:
        return {"outcome": "non-finite", "reason": nonfinite[-1][:1500]}
    if stopped:
        return {"outcome": "error", "reason": stopped[-1][:1500]}
    if refusal:
        return {"outcome": "refused", "reason": refusal[0][:1500]}
    tail = "\n".join(lines[-6:])
    return {"outcome": "error", "reason": tail[-1500:]}


def _sum_column(area, rho, dz, mask=None):
    column = np.sum(rho * dz, axis=1, dtype=np.float64)
    if mask is not None:
        return float(np.sum(area[mask] * column[mask]))
    return float(np.sum(area * column))


def history_budgets(frames: list[Path], static: Path, hours: float) -> dict:
    """Finiteness and the dry/water budget terms over the history window."""

    import netCDF4  # noqa: PLC0415

    with netCDF4.Dataset(static) as ds:
        area = np.asarray(ds.variables["areaCell"][:], np.float64)
        radius = float(getattr(ds, "sphere_radius", 1.0))
        if radius == 1.0:
            area = area * 6371229.0 ** 2
        interior = (np.asarray(ds.variables["bdyMaskCell"][:]) == 0
                    if "bdyMaskCell" in ds.variables else None)
    nonfinite = []
    terms = []
    species_seen = None
    for path in frames:
        with netCDF4.Dataset(path) as ds:
            ds.set_auto_mask(False)
            for name, var in ds.variables.items():
                if var.dtype.kind != "f":
                    continue
                data = np.asarray(var[:])
                if not np.all(np.isfinite(data)):
                    nonfinite.append(f"{path.name}:{name}")
            zgrid = np.asarray(ds.variables["zgrid"][:], np.float64)
            dz = np.diff(zgrid, axis=1)
            rho = np.asarray(ds.variables["rho"][0], np.float64)
            species = [name for name in WATER_SPECIES if name in ds.variables]
            species_seen = species
            q = sum(np.asarray(ds.variables[name][0], np.float64)
                    for name in species)
            q_min = {name: float(np.min(ds.variables[name][0]))
                     for name in species}
            precip = sum(np.asarray(ds.variables[name][0], np.float64)
                         for name in ("rainc", "rainnc")
                         if name in ds.variables)
            qfx = (np.asarray(ds.variables["qfx"][0], np.float64)
                   if "qfx" in ds.variables else np.zeros_like(area))
        row = {"dry": _sum_column(area, rho, dz),
               "water_atm": _sum_column(area, rho * q, dz),
               "precip": float(np.sum(area * precip)),
               "qfx": float(np.sum(area * qfx)), "q_min": q_min}
        if interior is not None:
            row.update(
                dry_interior=_sum_column(area, rho, dz, interior),
                water_atm_interior=_sum_column(area, rho * q, dz, interior),
                precip_interior=float(np.sum(area[interior]
                                             * precip[interior])))
        terms.append(row)
    first, last = terms[0], terms[-1]
    seconds = hours * 3600.0
    evap = 0.5 * (first["qfx"] + last["qfx"]) * seconds
    out = {
        "frames": [path.name for path in frames],
        "nonfinite": nonfinite[:20],
        "species": species_seen,
        "dry_mass_kg": [first["dry"], last["dry"]],
        "dry_rel_change": (last["dry"] - first["dry"]) / first["dry"],
        "water_atm_kg": [first["water_atm"], last["water_atm"]],
        "precip_kg": last["precip"] - first["precip"],
        "evap_estimate_kg": evap,
        "water_residual_rel": (
            (last["water_atm"] - first["water_atm"])
            + (last["precip"] - first["precip"]) - evap) / first["water_atm"],
        "q_min_end": last["q_min"],
    }
    if "dry_interior" in first:
        out["dry_rel_change_interior"] = (
            (last["dry_interior"] - first["dry_interior"])
            / first["dry_interior"])
        out["water_residual_rel_interior"] = (
            (last["water_atm_interior"] - first["water_atm_interior"])
            + (last["precip_interior"] - first["precip_interior"])
        ) / first["water_atm_interior"]
    return out


def _dig(tree, *path):
    for key in path:
        if not isinstance(tree, dict) or key not in tree:
            return None
        tree = tree[key]
    return tree


def receipt_physics(receipt: Path) -> dict:
    """The receipt's resolved physics: seam, sealed assignment, call counts."""

    data = json.loads(receipt.read_text())
    resolution = _dig(data, "physics", "resolution") or {}
    sealed = _dig(data, "driver_receipt", "forecast", "host_preparation",
                  "constructor", "sealed_host_contract_audit", "physics")
    return {
        "seam": resolution.get("seam"),
        "backend_row": resolution.get("backend_row"),
        "selectors": resolution.get("selectors"),
        "engine_physics": resolution.get("engine_physics"),
        "sealed_physics": sealed,
        "call_counts": _dig(data, "physics", "cadence",
                            "engine_call_counts_at_last_step"),
        "status": data.get("status"),
        "integration_seconds": _dig(data, "driver_receipt",
                                    "execution_seconds"),
    }


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------
def run_row(row: dict, case: dict, work: Path, backend: str) -> dict:
    hours = float(case.get("hours", 1))
    rdir = work / backend / row["option"]
    if rdir.exists():
        subprocess.run(["rm", "-rf", str(rdir)], check=True)
    rdir.mkdir(parents=True)
    out, scratch, log = rdir / "out", rdir / "scratch", rdir / "log"
    argv = [sys.executable, "-m", "woof", "hex", "forecast",
            *case["forecast_args"], "--hours", f"{hours:g}",
            "--history-every-minutes", f"{int(hours * 60)}",
            "--out", str(out), "--scratch", str(scratch), "--no-products",
            f"--physics-suite={case.get('suite', 'arwen_mpas_column')}",
            *physics_args(row, case.get("base_physics", {}))]
    env = dict(os.environ, **case.get("env", {}))
    env["PYTHONPATH"] = str(REPO)
    started = time.monotonic()
    with log.open("w") as handle:
        proc = subprocess.run(argv, cwd=REPO, env=env, stdout=handle,
                              stderr=subprocess.STDOUT)
    wall = time.monotonic() - started
    text = log.read_text(errors="replace")
    result = {"option": row["option"], "backend": backend,
              "overrides": row["overrides"],
              "physics_args": physics_args(row, case.get("base_physics", {})),
              "returncode": proc.returncode, "wall_seconds": round(wall, 1)}
    result.update(classify_log(text, proc.returncode))
    receipt = out / "forecast-receipt.json"
    if receipt.is_file():
        try:
            result["receipt"] = receipt_physics(receipt)
        except (OSError, ValueError) as error:
            result["receipt_error"] = str(error)
    if result["outcome"] == "ran":
        frames = sorted(out.glob("*history.*.nc"))
        try:
            result["budgets"] = history_budgets(
                frames, Path(case["static"]), hours)
            if result["budgets"]["nonfinite"]:
                result["outcome"] = "non-finite"
        except Exception as error:  # noqa: BLE001 -- the row is the report
            result["outcome"] = "error"
            result["reason"] = f"budget read: {type(error).__name__}: {error}"
    if not case.get("keep_history"):
        _prune(out, scratch)
    return result


#: What a row keeps of its run: the door's receipt and traceback (the log
#: sits beside ``out``).  History frames, step receipts and the scratch tree
#: are tens of MB a row, and a host sweeping a hundred rows has no room.
_KEEP = ("forecast-receipt.json", "forecast-traceback.log")


def _prune(out: Path, scratch: Path) -> None:
    subprocess.run(["rm", "-rf", str(scratch)], check=False)
    if not out.is_dir():
        return
    for path in out.iterdir():
        if path.name in _KEEP:
            continue
        if path.is_dir():
            subprocess.run(["rm", "-rf", str(path)], check=False)
        else:
            path.unlink()


def grade(results: dict) -> dict:
    """Grade every ``ran`` row's budgets against the baseline row."""

    rows = results["rows"]
    base = rows.get(BASELINE)
    if not base or base.get("outcome") not in ("ran", "budget"):
        return results
    b = base["budgets"]
    for row in rows.values():
        budgets = row.get("budgets")
        if not budgets or row.get("outcome") not in ("ran", "budget"):
            continue
        dry = budgets["dry_rel_change"] - b["dry_rel_change"]
        water = budgets["water_residual_rel"] - b["water_residual_rel"]
        row["graded"] = {"dry_vs_baseline": dry, "water_vs_baseline": water,
                         "dry_tol": DRY_TOL, "water_tol": WATER_TOL}
        bad = []
        if abs(dry) > DRY_TOL:
            bad.append(f"dry mass {dry:+.3e} vs baseline")
        if abs(water) > WATER_TOL:
            bad.append(f"water residual {water:+.3e} vs baseline")
        row["outcome"] = "budget" if bad else "ran"
        if bad:
            row["reason"] = "; ".join(bad)
    return results


def _save(path: Path, results: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(grade(results), indent=1, sort_keys=True,
                              default=str))
    tmp.replace(path)


def command_run(args) -> int:
    case = json.loads(Path(args.case).read_text())
    work = Path(args.out).absolute()
    work.mkdir(parents=True, exist_ok=True)
    path = work / f"results-{args.backend}.json"
    results = (json.loads(path.read_text()) if path.is_file() else
               {"schema": SCHEMA, "backend": args.backend,
                "case": case.get("label", ""),
                "hours": case.get("hours", 1), "rows": {}})
    rows = plan(args.components)
    # The baseline first: every other row is graded against it.
    rows.sort(key=lambda row: row["option"] != BASELINE)
    if args.only:
        rows = [row for row in rows
                if row["option"] in args.only or row["option"] == BASELINE]
    for row in rows:
        done = results["rows"].get(row["option"])
        if done and not args.rerun and done.get("outcome") != "error":
            continue
        # A row whose assignment equals one already run (the frozen
        # default's own options: noah-mp, ysu, revised-mm5 ...) IS that run.
        twin = next((other for other in results["rows"].values()
                     if other.get("overrides") == row["overrides"]
                     and "same_run_as" not in other
                     and other["option"] != row["option"]), None)
        if twin is not None:
            result = dict(twin, option=row["option"],
                          same_run_as=twin["option"])
        else:
            result = run_row(row, case, work, args.backend)
        results["rows"][row["option"]] = result
        _save(path, results)
        print(json.dumps({key: result.get(key) for key in
                          ("option", "outcome", "wall_seconds", "needs",
                           "reason")}, default=str)[:600], flush=True)
    _save(path, results)
    return 0


# ---------------------------------------------------------------------------
# the published evidence and its table
# ---------------------------------------------------------------------------
_PATH = re.compile(r"(?:/[\w.@+-]+){2,}")


def _scrub(text: str | None) -> str | None:
    """A reason without the run's paths (they name scratch directories)."""

    if text is None:
        return None
    return _PATH.sub("<path>", text)


def _published_row(row: dict) -> dict:
    receipt = row.get("receipt") or {}
    budgets = row.get("budgets") or {}
    out = {"outcome": row.get("outcome"),
           "wall_seconds": row.get("wall_seconds")}
    for key in ("needs", "same_run_as"):
        if row.get(key):
            out[key] = row[key]
    if row.get("outcome") != "ran" and row.get("reason"):
        out["reason"] = _scrub(row["reason"])[:600]
    if receipt.get("seam"):
        out["seam"] = receipt["seam"]
        out["sealed_physics"] = receipt.get("sealed_physics")
        out["call_counts"] = receipt.get("call_counts")
    if budgets:
        out["budgets"] = {
            key: budgets[key] for key in (
                "dry_rel_change", "water_residual_rel", "precip_kg",
                "dry_rel_change_interior", "water_residual_rel_interior",
                "species", "q_min_end") if key in budgets}
    if row.get("graded"):
        out["graded"] = row["graded"]
    return out


def publish(loaded: list[dict]) -> dict:
    """Merge per-backend results into the committed evidence file."""

    names = sorted({name for data in loaded for name in data["rows"]})
    rows = {}
    for name in names:
        first = next(data["rows"][name] for data in loaded
                     if name in data["rows"])
        rows[name] = {"overrides": first["overrides"], "backends": {
            data["backend"]: _published_row(data["rows"][name])
            for data in loaded if name in data["rows"]}}
    return {"schema": SCHEMA,
            "case": next((data.get("case") for data in loaded), ""),
            "hours": next((data.get("hours") for data in loaded), None),
            "baseline": BASELINE,
            "tolerances": {"dry": DRY_TOL, "water": WATER_TOL},
            "backends": [data["backend"] for data in loaded],
            "rows": rows}


def verdict(row: dict) -> str:
    """One option's matrix status across the backends that ran it."""

    outcomes = {b["outcome"] for b in row["backends"].values()}
    if outcomes == {"ran"}:
        return "supported"
    if outcomes == {"refused"}:
        first = next(iter(row["backends"].values()))
        if first.get("needs"):
            return "refused (needs " + ", ".join(first["needs"]) + ")"
        return "refused"
    bad = outcomes - {"ran", "refused"}
    return " / ".join(sorted(bad or outcomes))


def _cell(row: dict | None) -> str:
    if row is None:
        return "not run"
    outcome = row.get("outcome")
    if outcome == "ran":
        return f"ran ({row.get('wall_seconds') or 0:.0f} s)"
    if outcome == "refused" and row.get("needs"):
        return "refused (needs " + ", ".join(row["needs"]) + ")"
    return outcome


def _notes(row: dict) -> str:
    bits = []
    for backend in row["backends"].values():
        if backend.get("graded"):
            graded = backend["graded"]
            bits.append(f"dM {graded['dry_vs_baseline']:+.1e} "
                        f"dW {graded['water_vs_baseline']:+.1e}")
            break
    for backend in row["backends"].values():
        if backend.get("same_run_as"):
            bits.append(f"same run as {backend['same_run_as']}")
            break
    for backend in row["backends"].values():
        if backend.get("reason") and not backend.get("needs"):
            bits.append(backend["reason"].replace("|", "/")
                        .replace("\n", " ")[:200])
            break
    return "; ".join(bits)


def table(evidence: dict) -> str:
    backends = evidence["backends"]
    lines = ["| option | --physics | status | " + " | ".join(backends)
             + " | notes |", "|---|---|---|" + "---|" * len(backends)
             + "---|"]
    counts = {}
    for name, row in evidence["rows"].items():
        over = " ".join(f"{k}={v}" for k, v in row["overrides"].items())
        status = verdict(row)
        key = status.split(" (")[0]
        counts[key] = counts.get(key, 0) + 1
        cells = [_cell(row["backends"].get(b)) for b in backends]
        lines.append(f"| {name} | {over or '(default)'} | {status} | "
                     + " | ".join(cells) + f" | {_notes(row)} |")
    lines.append("")
    lines.append("Counts: " + ", ".join(
        f"{key} {value}" for key, value in sorted(counts.items())))
    return "\n".join(lines)


def command_publish(args) -> int:
    loaded = [json.loads(Path(path).read_text()) for path in args.results]
    evidence = publish(loaded)
    Path(args.output).write_text(json.dumps(evidence, indent=1,
                                            sort_keys=True) + "\n")
    return 0


def command_table(args) -> int:
    print(table(json.loads(Path(args.evidence).read_text())))
    return 0


def command_plan(args) -> int:
    for row in plan(args.components):
        print(f"{row['option']}\t" + " ".join(
            f"{k}={v}" for k, v in row["overrides"].items()))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--components", nargs="*")
    p.set_defaults(func=command_plan)
    p = sub.add_parser("run")
    p.add_argument("--case", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--backend", required=True,
                   help="the label the results file carries (cuda, metal)")
    p.add_argument("--components", nargs="*")
    p.add_argument("--only", nargs="*")
    p.add_argument("--rerun", action="store_true")
    p.set_defaults(func=command_run)
    p = sub.add_parser("publish")
    p.add_argument("results", nargs="+")
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(func=command_publish)
    p = sub.add_parser("table")
    p.add_argument("evidence")
    p.set_defaults(func=command_table)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
