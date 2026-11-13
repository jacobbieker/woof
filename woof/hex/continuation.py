"""``woof resume`` and ``woof branch`` for hex runs: thin verbs over ``woof hex forecast``.

The WRF path's verbs (:mod:`woof.resume`, :mod:`woof.branch`) continue a run from a
``gpuwmrst_*.npz`` checkpoint.  A hex forecast writes its own checkpoint
(``--checkpoint-hours`` -> ``checkpoint.<label>.pkl``) and continues from it
(``--restart-from``), and those two flags are the whole mechanism.  These verbs add
nothing to the numerics.  They LOCATE a checkpoint, rebuild the forecast's own
command line from the run directory, and call the forecast door with ``--restart-from``.
Every safety property (checkpoint schema, mesh and step-size match, valid time against
the schedule, physics backend state) stays the driver's own.

A hex run is recognised from its directory's receipts: ``forecast-receipt.json``
(``gpuwm-hex.forecast-run/v1``), ``forecast-start.json`` or the driver receipt.  The
door's command line is recorded beside them in ``forecast-request.json`` (the door hands it to the
forecast writers, which record it with the start record, :mod:`woof.hex.forecast_start`), because the receipt records the DRIVER's
argv, which the door derived and cannot be turned back into door flags.  A run that
predates the record is refused by name.

``woof resume RUNDIR`` continues into a sibling leg directory (the door never writes into an existing run).  ``woof branch RUNDIR --outdir
NEW`` writes a new run directory (``branch_receipt.json``, ``branch_manifest.json``, the
checkpoint's hash, written once the door has made the directory) and integrates it from the checkpoint with the settings the restart
state does not bind; the source run is read only.

Refused by name (never relaxed): a ``--set`` of anything that shapes the restart state
(physics, scheme selectors, mesh, timestep, init, lateral boundaries, numerics), a
DA-cycling segment, a ``--members`` roster, and a directory that is not a hex run.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _datetime
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REQUEST_NAME = "forecast-request.json"
REQUEST_SCHEMA = "gpuwm-hex.forecast-request/1"
RUN_RECEIPT_NAME = "forecast-receipt.json"
RUN_RECEIPT_SCHEMA = "gpuwm-hex.forecast-run/v1"
START_RECORD_NAME = "forecast-start.json"
DRIVER_RECEIPT_NAME = "cuda-v841-forecast-receipt.json"
RESUME_RECEIPT_NAME = "resume_receipt.json"
RESUME_RECEIPT_SCHEMA = "gpuwm-hex.resume-receipt/1"
BRANCH_RECEIPT_NAME = "branch_receipt.json"
BRANCH_RECEIPT_SCHEMA = "gpuwm-branch-receipt/v1"
BRANCH_MANIFEST_NAME = "branch_manifest.json"
BRANCH_MANIFEST_SCHEMA = "gpuwm-branch-manifest/v1"


#: ``checkpoint.<label>.pkl``: the R1 route labels a checkpoint with its valid time
#: (``2025-05-31_00.30.00``), the R2 route with ``fHHH`` (whole hours) or ``tSSSSSSSS`` (seconds).
_CHECKPOINT_NAME = re.compile(
    r"^checkpoint\.(?:(?P<valid>\d{4}-\d{2}-\d{2}_\d{2}\.\d{2}\.\d{2})|f(?P<hours>\d+)|t(?P<seconds>\d+))\.pkl$")

#: Door flags a branch may change, and the reason the rest may not.  Each takes one value.
CHANGEABLE = {
    "hours": "--hours",
    "checkpoint_hours": "--checkpoint-hours",
}
CHANGEABLE_SETTINGS = "hours (extend the forecast), checkpoint_hours"

_PHYSICS_KEYS = frozenset({
    "physics", "physics_suite", "physics_mode", "physics_backend", "physics_mesh", "preset", "convection",
    "pbl", "pbl_cadence", "radiation_seconds", "les_model", "les_surface", "stochastic", "sppt", "skeb",
    "horiz_mixing", "monotonic", "time_integration_order", "acoustic_substeps", "wave", "ocean",
    "land_surface", "surface_layer", "microphysics", "cumulus", "chem", "wrf_hydro"})
_PHYSICS_REASON = (
    "the checkpoint carries the state its physics and dynamics wrote (the backend_state layout, the scheme "
    "scalars, the held tendencies); a branch under other physics would restore a layout the new "
    "configuration does not produce or miss the one it needs")
_RUN_REASON = (
    "it is part of what the run IS (mesh, timestep, initial state, boundaries, start time), so the "
    "checkpoint's own mesh/step/valid-time guards would refuse the restore")


def _refuse(text: str) -> ValueError:
    return ValueError(text)


# ---------------------------------------------------------------------------
# the request record and run detection
# ---------------------------------------------------------------------------


def _strip_flag(argv: Sequence[str], flag: str) -> list[str]:
    """``argv`` without every ``flag VALUE`` / ``flag=VALUE`` occurrence."""
    out: list[str] = []
    skip = False
    for token in argv:
        if skip:
            skip = False
            continue
        if token == flag:
            skip = True
            continue
        if token.startswith(flag + "="):
            continue
        out.append(token)
    return out


def _flag_value(argv: Sequence[str], flag: str) -> str | None:
    value = None
    for index, token in enumerate(argv):
        if token == flag and index + 1 < len(argv):
            value = argv[index + 1]
        elif token.startswith(flag + "="):
            value = token.split("=", 1)[1]
    return value


REQUEST_ENV = "WOOF_HEX_DOOR_REQUEST"


def request_text(argv: Sequence[str]) -> str:
    """The door's own command line (``forecast ...``) as the record's text.

    ``--restart-from`` is not recorded: the record is the run's command line, and the restart selection is what a
    continuation supplies.  ``--backend`` is consumed before the door's parser and travels as the environment,
    so it is kept here.
    """
    record = {"schema": REQUEST_SCHEMA, "argv": _strip_flag(list(argv), "--restart-from"), "cwd": os.getcwd(),
              "backend": os.environ.get("WOOF_BACKEND")}
    return json.dumps(record, indent=1, sort_keys=True) + "\n"


def write_request_text(out: Path, text: str) -> Path:
    target = Path(out) / REQUEST_NAME
    partial = target.with_name(target.name + ".partial")
    partial.write_text(text, encoding="utf-8")
    partial.replace(target)
    return target


def write_request(out: Path, argv: Sequence[str]) -> Path:
    return write_request_text(out, request_text(argv))


def is_hex_run(path: Path) -> bool:
    """True when ``path`` is a hex forecast directory, judged from its receipts."""
    path = Path(path)
    if not path.is_dir():
        return False
    receipt = path / RUN_RECEIPT_NAME
    if receipt.is_file():
        try:
            return json.loads(receipt.read_text(encoding="utf-8")).get("schema") == RUN_RECEIPT_SCHEMA
        except (OSError, ValueError):
            return False
    return (path / START_RECORD_NAME).is_file() or (path / DRIVER_RECEIPT_NAME).is_file()


def read_request(run: Path) -> dict[str, Any]:
    path = Path(run) / REQUEST_NAME
    if not path.is_file():
        raise _refuse(
            f"{run} is a hex run, but it has no {REQUEST_NAME}: it was started before the door recorded its "
            "command line, so the command that continues it cannot be rebuilt.  Continue it by hand with "
            "`woof hex forecast <the original flags> --restart-from <checkpoint>` "
            "(--start-time/--hours name the ORIGINAL run)")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema") != REQUEST_SCHEMA or not record.get("argv"):
        raise _refuse(f"{path} is not a {REQUEST_SCHEMA} record")
    argv = [str(token) for token in record["argv"]]
    if argv[0] != "forecast":
        raise _refuse(f"{path} records `{argv[0]}`, not a forecast door command line")
    if "--da-analysis" in argv or any(t.startswith("--da-analysis=") for t in argv):
        raise _refuse(
            "this run is a DA-cycling segment (--da-analysis): resuming it would apply the analysis again "
            "to a state that already carries it.  Start the next cycle with `woof hex analysis`")
    if "--members" in argv or any(t.startswith("--members=") for t in argv):
        raise _refuse(
            "this run is a --members roster: each member would restore one checkpoint and run the same "
            "forecast, published with zero spread.  Resume a member's own directory instead")
    record["argv"] = argv
    return record


# ---------------------------------------------------------------------------
# checkpoint location
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    label: str
    sort_key: tuple[int, Any]

    def describe(self) -> str:
        return self.path.name


def discover_checkpoints(run: Path) -> list[Checkpoint]:
    """Every complete ``checkpoint.<label>.pkl`` in ``run``, newest first.

    ``write_checkpoint`` publishes by atomic rename of a ``.part`` file, so a name that matches is a
    complete file; a zero-byte one is skipped as unreadable.
    """
    found = []
    for path in Path(run).glob("checkpoint.*.pkl"):
        match = _CHECKPOINT_NAME.fullmatch(path.name)
        if match is None or not path.is_file() or path.stat().st_size == 0:
            continue
        if match.group("valid"):
            key = (0, match.group("valid"))
        elif match.group("hours"):
            key = (1, int(match.group("hours")) * 3600)
        else:
            key = (1, int(match.group("seconds")))
        found.append(Checkpoint(path, path.name[len("checkpoint."):-len(".pkl")], key))
    return sorted(found, key=lambda c: c.sort_key, reverse=True)


def resolve_checkpoint(run: Path, spec: str | Path = "latest") -> Checkpoint:
    if str(spec) == "latest":
        found = discover_checkpoints(run)
        if not found:
            raise _refuse(
                f"{run} holds no checkpoint.*.pkl: the run was not started with --checkpoint-hours, so there "
                "is nothing to continue from.  Re-run it with `woof hex forecast ... --checkpoint-hours H`")
        return found[0]
    path = Path(spec)
    if not path.is_absolute() and not path.exists() and (Path(run) / path).exists():
        path = Path(run) / path
    match = _CHECKPOINT_NAME.fullmatch(path.name)
    if match is None or not path.is_file():
        raise _refuse(f"--from {spec} is not a hex checkpoint file (checkpoint.<label>.pkl)")
    for known in discover_checkpoints(path.parent):
        if known.path == path:
            return known
    return Checkpoint(path, path.name[len("checkpoint."):-len(".pkl")], (2, path.name))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# the plans (pure: no device, no writes)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContinuationPlan:
    verb: str
    argv: tuple[str, ...]          # the forecast door's argv, `forecast` first
    cwd: str
    backend: str | None
    checkpoint: Path
    run: Path
    outdir: Path
    receipt: Mapping[str, Any]


def _with_option(argv: Sequence[str], flag: str, value: str) -> list[str]:
    return _strip_flag(argv, flag) + [flag, value]


def plan_resume(run: Path, spec: str | Path = "latest", leg_out: Path | None = None) -> ContinuationPlan:
    """Continue ``run`` from a checkpoint, writing the continuation leg to its own directory.

    The forecast door refuses an existing ``--out`` (a second run must never be read as the first one's
    continuation, nor mix two trajectories' frames), so a resume does not write into the run: the leg goes to
    ``<run>-resume-<label>`` (or ``--leg-out``), holds the frames AFTER the checkpoint (the frame at the
    checkpoint is the run's own) and its own receipts.  ``--hours`` stays the ORIGINAL run's length.
    """
    run = Path(run).resolve()
    if not is_hex_run(run):
        raise _refuse(f"{run} is not a hex forecast directory (no {RUN_RECEIPT_NAME})")
    request = read_request(run)
    checkpoint = resolve_checkpoint(run, spec)
    leg = (Path(leg_out).resolve() if leg_out is not None
           else run.with_name(f"{run.name}-resume-{checkpoint.label}"))
    _check_target(leg, run)
    argv = _strip_flag(request["argv"], "--restart-from")
    argv = _strip_flag(argv, "--receipt")                        # the source run's explicit receipt path
    argv = _with_option(argv, "--out", str(leg))
    argv = _with_option(argv, "--restart-from", str(checkpoint.path.resolve()))
    receipt = {"schema": RESUME_RECEIPT_SCHEMA, "verb": "resume", "engine": "hex",
               "created_utc": _utc(), "run_directory": str(run), "leg_directory": str(leg),
               "checkpoint": {"path": str(checkpoint.path.resolve()), "sha256": _sha256(checkpoint.path),
                              "label": checkpoint.label},
               "command": ["woof", "hex", *argv]}
    return ContinuationPlan("resume", tuple(argv), request["cwd"], request.get("backend"),
                            checkpoint.path.resolve(), run, leg, receipt)


def _check_target(outdir: Path, run: Path) -> None:
    if outdir == run or run in outdir.parents:
        raise _refuse(f"--outdir {outdir} is inside the run being continued ({run}); the new run must not "
                      "write into its own source")
    if outdir in run.parents:
        raise _refuse(f"--outdir {outdir} contains the run being continued ({run})")
    if outdir.exists():
        raise _refuse(f"{outdir} exists; the forecast door writes a continuation into a directory that does not "
                      "(name another with --leg-out / --outdir)")


def parse_setting(text: str) -> tuple[str, str]:
    """One ``KEY=VALUE``: the changeable door flag and its value, or the named refusal."""
    if "=" not in text:
        raise _refuse(f"`{text}` is not a setting: --set takes KEY=VALUE, for example --set hours=12")
    key, value = text.split("=", 1)
    spelled = key.strip().lstrip("-").replace("-", "_")
    if spelled in CHANGEABLE:
        return spelled, value.strip()
    if spelled in _PHYSICS_KEYS or spelled.startswith("physics"):
        reason = _PHYSICS_REASON
    elif spelled in {"mesh", "grid", "static", "init", "lbc_dir", "start_time", "dt", "out", "init_source",
                     "history_every_minutes", "history_from_minutes", "restart_from", "da_analysis"}:
        reason = _RUN_REASON
    else:
        raise _refuse(
            f"`{key}` is not a setting a hex branch knows; changeable from a checkpoint: "
            f"{CHANGEABLE_SETTINGS}.  Everything else is refused by name (a typo is refused as a typo)")
    raise _refuse(
        f"`{key}` is pinned by the checkpoint and cannot be changed by a hex branch: {reason}.\n"
        f"  changeable from a checkpoint: {CHANGEABLE_SETTINGS}\n"
        f"  remedy: to change {key}, run the forecast from its start instead -- `woof hex forecast ... "
        "--out <new dir>` -- which integrates the new setting from t=0")


def plan_branch(run: Path, outdir: Path, spec: str | Path = "latest",
                settings: Sequence[str] = ()) -> ContinuationPlan:
    run = Path(run).resolve()
    outdir = Path(outdir).resolve()
    if not is_hex_run(run):
        raise _refuse(f"{run} is not a hex forecast directory (no {RUN_RECEIPT_NAME})")
    overrides = [parse_setting(text) for text in settings]       # classified before anything is made
    _check_target(outdir, run)
    request = read_request(run)
    checkpoint = resolve_checkpoint(run, spec)
    argv = _strip_flag(request["argv"], "--restart-from")
    argv = _strip_flag(argv, "--receipt")                        # the source run's explicit receipt path
    stamped = []
    for key, value in overrides:
        flag = CHANGEABLE[key]
        stamped.append({"setting": key, "from": _flag_value(argv, flag), "to": value})
        argv = _with_option(argv, flag, value)
    argv = _with_option(argv, "--out", str(outdir))
    argv = _with_option(argv, "--restart-from", str(checkpoint.path.resolve()))
    receipt = {
        "schema": BRANCH_RECEIPT_SCHEMA, "engine": "hex", "created_utc": _utc(),
        "parent": {"run_directory": str(run),
                   "checkpoint": {"path": str(checkpoint.path.resolve()), "sha256": _sha256(checkpoint.path),
                                  "label": checkpoint.label}},
        "branch": {"run_directory": str(outdir)},
        "overrides": stamped,
        "command": ["woof", "hex", *argv],
    }
    return ContinuationPlan("branch", tuple(argv), request["cwd"], request.get("backend"),
                            checkpoint.path.resolve(), run, outdir, receipt)


def _utc() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).replace(microsecond=0).isoformat()


def _publish(path: Path, payload: Mapping[str, Any]) -> None:
    partial = path.with_name(f".{path.name}.partial-{os.getpid()}")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(partial, path)


def write_receipts(plan: ContinuationPlan) -> None:
    if plan.verb == "resume":
        _publish(plan.outdir / RESUME_RECEIPT_NAME, plan.receipt)
        return
    manifest = {"schema": BRANCH_MANIFEST_SCHEMA, "handle": str(plan.checkpoint),
                "members": [{"grid_id": 0, "path": str(plan.checkpoint), "bytes": plan.checkpoint.stat().st_size,
                             "sha256": plan.receipt["parent"]["checkpoint"]["sha256"]}],
                "access": "read-only"}
    _publish(plan.outdir / BRANCH_RECEIPT_NAME, plan.receipt)
    _publish(plan.outdir / BRANCH_MANIFEST_NAME, manifest)


# ---------------------------------------------------------------------------
# execution and the command line
# ---------------------------------------------------------------------------


def _hex_main(argv: Sequence[str]) -> int:
    from woof.hex.cli import main
    return int(main(list(argv)))


@contextlib.contextmanager
def _in(cwd: str, backend: str | None):
    saved_cwd = os.getcwd()
    saved_backend = os.environ.get("WOOF_BACKEND")
    try:
        if Path(cwd).is_dir():
            os.chdir(cwd)
        if backend and saved_backend is None:
            os.environ["WOOF_BACKEND"] = backend
        yield
    finally:
        os.chdir(saved_cwd)
        if saved_backend is None:
            os.environ.pop("WOOF_BACKEND", None)
        else:
            os.environ["WOOF_BACKEND"] = saved_backend


def execute(plan: ContinuationPlan, runner: Callable[[Sequence[str]], int] | None = None) -> int:
    runner = runner or _hex_main
    with _in(plan.cwd, plan.backend):
        return runner(list(plan.argv))


def hex_target(tokens: Sequence[str]) -> Path | None:
    """The hex run a ``resume``/``branch`` command line names, or None when it is the WRF path's.

    The run is the CONFIG argument when that is a directory holding hex receipts, else ``--outdir`` (resume)
    or ``--from-run`` (branch).
    """
    verb = tokens[0] if tokens else None
    if verb not in ("resume", "branch"):
        return None
    candidates: list[str] = []
    rest = list(tokens[1:])
    for index, token in enumerate(rest):
        if token in ("--outdir", "--from-run") and index + 1 < len(rest):
            candidates.append(rest[index + 1])
        elif token.startswith(("--outdir=", "--from-run=")):
            candidates.append(token.split("=", 1)[1])
    positional = next((t for t in rest if not t.startswith("-")), None)
    for text in ([positional] if positional else []) + candidates:
        path = Path(text)
        if is_hex_run(path):
            return path
    return None


def _parser(verb: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"woof {verb}", description=(
            "continue a hex run from its newest checkpoint (thin over `woof hex forecast --restart-from`)"
            if verb == "resume" else
            "start a NEW hex run from an existing run's checkpoint (the source run is read only)"))
    parser.add_argument("config", nargs="?", type=Path, metavar="RUNDIR",
                        help="the hex run directory (a CONFIG file is the WRF path's)")
    parser.add_argument("--from", dest="from_checkpoint", default="latest", metavar="CKPT|latest",
                        help="explicit checkpoint.<label>.pkl, or 'latest' (default)")
    if verb == "resume":
        parser.add_argument("--outdir", type=Path, default=None, metavar="OUT",
                            help="the interrupted run's directory (default: RUNDIR)")
        parser.add_argument("--leg-out", type=Path, default=None, metavar="DIR",
                            help="where the continuation is written (default: <run>-resume-<checkpoint>); the "
                                 "forecast door never writes into an existing run directory")
    else:
        parser.add_argument("--from-run", dest="from_run", type=Path, default=None, metavar="RUNDIR",
                            help="the source run's directory (default: RUNDIR)")
        parser.add_argument("--outdir", type=Path, required=True, metavar="OUT",
                            help="the NEW run's directory; it must be empty and outside the source run")
        parser.add_argument("--set", dest="settings", action="append", default=[], metavar="KEY=VALUE",
                            help=f"a setting to change, repeatable.  Changeable: {CHANGEABLE_SETTINGS}")
        parser.add_argument("--prepare-only", action="store_true",
                            help="write the branch directory and its receipts, then stop without integrating")
    return parser


def main(tokens: Sequence[str], runner: Callable[[Sequence[str]], int] | None = None) -> int:
    """``woof resume|branch`` on a hex run.  ``tokens[0]`` is the verb.

    The forecast door requires a directory that does not exist, so the decision receipts are written into the
    continuation directory after the door has made it (also when the run fails, so the attempt is on record).
    """
    verb = tokens[0]
    args = _parser(verb).parse_args(list(tokens[1:]))
    try:
        if verb == "resume":
            run = args.outdir or args.config
            if run is None:
                raise _refuse("woof resume needs the hex run directory (RUNDIR or --outdir)")
            plan = plan_resume(run, args.from_checkpoint, args.leg_out)
            print(f"resume: continuing {plan.run} from {plan.checkpoint} into {plan.outdir}")
            prepare_only = False
        else:
            run = args.from_run or args.config
            if run is None:
                raise _refuse("woof branch needs the source hex run directory (RUNDIR or --from-run)")
            plan = plan_branch(run, args.outdir, args.from_checkpoint, args.settings)
            print(f"branch: new run {plan.outdir} from {plan.checkpoint} ({len(plan.receipt['overrides'])} "
                  "setting(s) changed)")
            for row in plan.receipt["overrides"]:
                print(f"branch:   {row['setting']}: {row['from']!r} -> {row['to']!r}")
            prepare_only = args.prepare_only
    except ValueError as error:
        print(f"woof {verb}: {error}", file=sys.stderr)
        return 2
    if prepare_only:
        print(f"branch: --prepare-only writes nothing (the forecast door needs a directory that does not "
              f"exist); the command is:\n  woof hex {' '.join(plan.argv)}")
        return 0
    try:
        return execute(plan, runner)
    finally:
        if plan.outdir.is_dir():
            write_receipts(plan)
            print(f"{verb}: receipts in {plan.outdir}")


def intercept(tokens: Sequence[str]) -> int | None:
    """The front door's hook: handle a hex run's ``resume``/``branch``, else None (the WRF path)."""
    if hex_target(tokens) is None:
        return None
    # A run directory named where CONFIG goes is the only positional; --outdir/--from-run may name it too.
    return main(tokens)


__all__ = ["CHANGEABLE", "Checkpoint", "ContinuationPlan", "discover_checkpoints", "execute", "hex_target",
           "intercept", "is_hex_run", "main", "parse_setting", "plan_branch", "plan_resume", "read_request",
           "resolve_checkpoint", "write_request"]
