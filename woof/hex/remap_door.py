"""``woof hex remap``: an MPAS state from one mesh onto another.

The adaptive cycle regenerates its mesh around what the last cycle found,
and the state that cycle ended on has to start the next one.  The cell sets
share nothing, so this is a remap, not a subset: ``rw_mpas_remap`` does the
numbers (Voronoi-polygon intersections, a conservative vertical remap onto
the target's own terrain-following levels, the cell-vector route for the
edge wind, and the hydrostatic rebalance ``rw_mpas_init`` itself builds a
column with).  This door owns argument resolution, the fail-closed checks
that need no array math, and the provenance receipt.

The output is an init-class file laid out from the TARGET template:
``--to-vertical`` (the artifact ``woof hex vertical`` or ``woof hex init``
constructs), else ``--to-grid`` when it already carries a vertical grid.

Refusals carry the engine's own coverage report: a target cell the source
does not cover (a regional source smaller than the new mesh) is refused,
never extrapolated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import MpasPortError

#: The door receipt's schema name.
RECEIPT_SCHEMA = "gpuwm-hex.remap-door/v1"

#: What a target cell must have covered by the source, unless told.
DEFAULT_MIN_COVERAGE = 1.0 - 1.0e-6

#: The virtual-factor arm the init writer and the dycore's equation of
#: state share (``theta_m = theta (1 + 1.61 qv)``).
DEFAULT_VIRTUAL_FACTOR = "reproduce-fortran"


class RemapDoorRefusal(MpasPortError):
    """A named refusal: wrong result prevented, then remedy."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _described(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def resolve_engine(explicit: Path | None) -> Path:
    """The ``rw_mpas_remap`` binary through the shared engine ladder."""

    from .engines import REMAP, EngineRefusal, resolve

    try:
        return resolve(REMAP, explicit)
    except EngineRefusal as error:
        raise RemapDoorRefusal(str(error)) from error


def _existing(flag: str, value: Path | None, *, required: bool) -> Path | None:
    if value is None:
        if required:
            raise RemapDoorRefusal(f"{flag} is required")
        return None
    # resolve(), not absolute(): a `..` or a symlinked directory must not
    # let two spellings of one file pass the same-file guards below.
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise RemapDoorRefusal(f"{flag} {path} is not a file")
    return path


def _netcdf_variables(path: Path) -> set[str] | None:
    """The variable names in a netCDF file, or ``None`` when unreadable here."""

    try:
        from netCDF4 import Dataset
    except ImportError:  # pragma: no cover - environment-specific
        return None
    try:
        with Dataset(path) as dataset:
            return set(dataset.variables)
    except OSError:
        return None


def build_argv(arguments: argparse.Namespace, engine: Path) -> tuple[list[str], dict[str, Path | None], Path, Path]:
    """Resolve and check every path.

    Returns the engine argv, the resolved inputs, the door receipt path and
    the engine receipt path.
    """

    from_grid = _existing("--from-grid", arguments.from_grid, required=True)
    from_state = _existing("--from-state", arguments.from_state, required=True)
    to_grid = _existing("--to-grid", arguments.to_grid, required=True)
    to_static = _existing("--to-static", arguments.to_static, required=False)
    to_vertical = _existing("--to-vertical", arguments.to_vertical, required=False)
    if arguments.out is None:
        raise RemapDoorRefusal("-o/--out is required")
    out = Path(arguments.out).expanduser().resolve()
    if not out.parent.is_dir():
        raise RemapDoorRefusal(f"the output directory {out.parent} does not exist; create it")
    inputs = {"from_grid": from_grid, "from_state": from_state, "to_grid": to_grid,
              "to_static": to_static, "to_vertical": to_vertical}
    for name, path in inputs.items():
        if path is not None and path == out:
            raise RemapDoorRefusal(
                f"-o {out} is the same file as --{name.replace('_', '-')}; the remap never "
                "overwrites one of its own inputs")
    if out.exists() and not arguments.clobber:
        raise RemapDoorRefusal(f"{out} exists; pass --clobber to replace it")
    if from_grid == to_grid and to_vertical is None:
        raise RemapDoorRefusal(
            "--from-grid and --to-grid are the same file; a remap onto the same mesh is the "
            "identity, and a state with a different vertical grid needs --to-vertical")
    if to_vertical is None:
        names = _netcdf_variables(to_grid)
        if names is not None and "zgrid" not in names:
            raise RemapDoorRefusal(
                f"--to-grid {to_grid} carries no zgrid and no --to-vertical was given: the target "
                "has no vertical grid to remap onto.  Build one with `woof hex vertical --grid "
                f"{to_grid} --static {to_static or 'B.static.nc'} --vertical-spec SPEC -o "
                "B.vertical.nc` and pass it as --to-vertical")
    min_coverage = float(arguments.min_coverage)
    if not 0.0 < min_coverage <= 1.0:
        raise RemapDoorRefusal(f"--min-coverage {min_coverage} is outside (0, 1]")
    receipt = (Path(arguments.receipt).expanduser().resolve() if arguments.receipt
               else out.with_name(out.name + ".receipt.json"))
    engine_receipt = receipt.with_name(receipt.stem + ".engine.json")
    if engine_receipt == receipt:  # pragma: no cover - a receipt named *.engine
        engine_receipt = receipt.with_name(receipt.name + ".engine.json")
    taken = {out, *(path for path in inputs.values() if path is not None)}
    for flag, path in (("--receipt", receipt), ("--receipt (engine copy)", engine_receipt)):
        if path in taken:
            raise RemapDoorRefusal(
                f"{flag} {path} is the output or one of the inputs; the receipt would be "
                "written over a state file")
    argv = [
        str(engine),
        "--from-grid", str(from_grid),
        "--from-state", str(from_state),
        "--to-grid", str(to_grid),
    ]
    if to_static is not None:
        argv += ["--to-static", str(to_static)]
    if to_vertical is not None:
        argv += ["--to-vertical", str(to_vertical)]
    argv += [
        "--out", str(out),
        "--balance", arguments.balance,
        "--virtual-factor", arguments.virtual_factor,
        "--min-coverage", repr(min_coverage),
        "--receipt", str(engine_receipt),
    ]
    return argv, inputs, receipt, engine_receipt


def _summary(engine_payload: dict[str, Any]) -> dict[str, Any]:
    budgets = engine_payload.get("budgets", {})
    coverage = engine_payload.get("coverage", {})
    return {
        "status": engine_payload.get("status"),
        "min_coverage": coverage.get("min_coverage"),
        "cells_below_required": coverage.get("cells_below_required"),
        "horizontal_stage_dry_air_relative_change": budgets.get("horizontal_stage_dry_air_relative_change"),
        "dry_air_relative_change": budgets.get("dry_air_relative_change"),
        "water_vapour_relative_change": budgets.get("water_vapour_relative_change"),
        "kinetic_energy_relative_change": budgets.get("kinetic_energy_relative_change"),
    }


def run_remap(arguments: argparse.Namespace) -> int:
    engine = resolve_engine(arguments.remap_exe)
    argv, inputs, receipt_path, engine_receipt = build_argv(arguments, engine)
    out = Path(argv[argv.index("--out") + 1])
    engine_receipt.unlink(missing_ok=True)
    started = time.perf_counter()
    completed = subprocess.run(argv, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    payload: dict[str, Any] | None = None
    if engine_receipt.is_file():
        payload = json.loads(engine_receipt.read_text(encoding="utf-8"))
    door = {
        "schema": RECEIPT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "engine": _described(engine),
        "argv": argv,
        "seconds": round(elapsed, 2),
        "exit_code": completed.returncode,
        "inputs": {name: _described(path) for name, path in inputs.items()},
        "balance": arguments.balance,
        "virtual_factor": arguments.virtual_factor,
        "summary": _summary(payload) if payload else None,
        "engine_receipt": payload,
        "output": _described(out) if completed.returncode == 0 and out.is_file() else None,
    }
    receipt_path.write_text(json.dumps(door, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8", newline="\n")
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()
        refusal = payload.get("refusal") if payload else None
        raise RemapDoorRefusal(
            f"rw_mpas_remap exited {completed.returncode}: "
            + (refusal or (tail[-1] if tail else "no message"))
            + f".  Receipt: {receipt_path}")
    if not out.is_file():
        raise RemapDoorRefusal(f"rw_mpas_remap exited 0 and wrote no {out}")
    print(json.dumps({"out": str(out), "receipt": str(receipt_path), "seconds": round(elapsed, 2),
                      **(_summary(payload) if payload else {})}, indent=2))
    return 0


def add_remap_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--from-grid", type=Path, required=True, metavar="A.grid.nc",
                        help="the source mesh")
    parser.add_argument("--from-state", type=Path, required=True, metavar="A.state.nc",
                        help="the source state: an init, restart or single-frame history on "
                             "--from-grid, carrying rho, theta, qv (and zgrid, or zgrid in the grid)")
    parser.add_argument("--to-grid", type=Path, required=True, metavar="B.grid.nc",
                        help="the target mesh")
    parser.add_argument("--to-static", type=Path, default=None, metavar="B.static.nc",
                        help="the target static; its landmask masks the soil remap by surface type")
    parser.add_argument("--to-vertical", type=Path, default=None, metavar="B.vertical.nc",
                        help="the target's vertical artifact; the output is laid out from it.  "
                             "Without it --to-grid must carry a vertical grid (an init on B does)")
    parser.add_argument("-o", "--out", type=Path, required=True, metavar="B.state.nc",
                        help="the remapped init-class state")
    parser.add_argument("--balance", choices=("hydrostatic", "carry"), default="hydrostatic",
                        help="hydrostatic (default) rebuilds rho in hydrostatic balance on the "
                             "target with rw_mpas_init's column routine; carry keeps the "
                             "conservatively remapped rho so the mass budget closes")
    parser.add_argument("--virtual-factor", choices=("reproduce-fortran", "consistent"),
                        default=DEFAULT_VIRTUAL_FACTOR,
                        help="the theta_m factor of the rebalance (default reproduce-fortran, "
                             "the arm the dycore's equation of state matches)")
    parser.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE,
                        help="fraction of each target cell the source must cover; below it "
                             "the remap is refused with a coverage report")
    parser.add_argument("--receipt", type=Path, default=None, metavar="JSON",
                        help="door receipt path (default: <out>.receipt.json)")
    parser.add_argument("--clobber", action="store_true", help="replace an existing -o file")
    parser.add_argument("--remap-exe", type=Path, default=None, metavar="FILE",
                        help="the rw_mpas_remap binary (default: the shared engine ladder)")


def add_remap_parser(commands: Any) -> None:
    parser = commands.add_parser(
        "remap",
        help="remap an MPAS init/restart state from one mesh onto another (rw_mpas_remap)",
        description=(
            "Remap an MPAS state between two arbitrary meshes: first-order conservative "
            "Voronoi-polygon intersections for the prognostic state, conservative in height "
            "onto the target's levels, the edge wind through the cell-centre vector, and a "
            "hydrostatic rebalance.  Refuses a target the source does not cover."
        ),
    )
    add_remap_arguments(parser)
    parser.set_defaults(handler=run_remap)
