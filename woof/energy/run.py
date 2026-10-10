"""Run every domain of a plan, in parent-first order, one at a time.

``woof energy run PLAN.json`` drives the existing front doors per plan
domain and records what happened in a run manifest
(``runs/manifest.json`` next to the plan, schema :data:`MANIFEST_SCHEMA`).
It does no numerical work itself: every forecast is a ``woof`` subprocess.

**Steps.**  The plan's domains are folded into *steps*, each one or more
``woof`` command lines:

* WRF root domains (``parent`` is ``None``; every ``wrf-nests`` domain and
  each ``wrf-tiles`` parent) that share one ``config`` are produced by ONE
  ``woof go <config> --outdir <run_dir>``, where ``run_dir`` is the domain
  with the lowest ``grid_id``.  Every domain of that config must name the
  same ``run_dir``, because one run writes one directory.
* A WRF domain with a ``parent`` (a ``wrf-tiles`` child, which may itself
  force further tiles) is ``woof downscale <parent frames> <downscale_args>``
  where ``downscale_args`` is ``PlanDomain.extra["downscale_args"]`` and must
  carry the child's create-only ``--out`` (``run_dir`` or beneath it).  An
  ``--out`` already holding an earlier attempt is renamed to a dated
  ``<out>.previous-<UTC>`` sibling before the rerun, never deleted, and the
  manifest notes the move.  ``<parent frames>`` is the one
  directory that holds the parent's recorded history frames, because
  ``woof downscale`` reads frames directly inside the directory it is given
  and ``woof go`` writes them below run-stamped folders; it is the parent's
  ``run_dir`` only until the parent has run (a dry run shows that, and
  says so in its notes).
* A ``hex-swath`` domain runs each argv in ``PlanDomain.extra["commands"]``
  in order.  ``extra["env_paths"]`` (environment name -> plan-relative
  path, e.g. ``WOOF_HEX_MESH_ROWS``) is exported as absolute paths to
  every one of them.

Each command runs as ``[sys.executable, "-m", "woof", *argv]`` with the
plan's directory as the working directory, so the relative paths a planner
emits resolve as written.  When this process imported woof from a source
checkout, that checkout leads ``PYTHONPATH`` so the child runs the same
code.  Steps run strictly one after another (one GPU).

**Completion.**  A domain is complete only when every command of its step
exited 0 AND its ``output_glob`` matched at least one file in its
``run_dir`` written since the step started.  The glob is matched in
``run_dir``; a glob with no directory part that matches nothing there is
searched beneath ``run_dir`` as well (``woof go`` puts frames under
``run-<stamp>/run/wrfout/``), and the manifest notes when that happened.
Files the glob matches that predate the step (an earlier attempt) do not
count and are noted.

**Failure.**  A failed step marks its domains ``failed`` and every domain
that depends on them, transitively, ``skipped`` with the reason;
independent steps still run.  The exit status is non-zero.  Under
``--only``, a dependent outside the list keeps its record, but one that was
complete returns to ``pending`` once its parent runs again, because its
output came from the parent run that was replaced.  Ctrl-C marks
the running domain ``failed`` (interrupted), saves the manifest and exits
130.

**Logs.**  Each step's output streams to ``runs/logs/<step>.log`` while it
runs and is copied to ``<run_dir>/energy-run.log`` when it ends (when the
run made that directory).  The live log lives outside ``run_dir`` because
``woof downscale --out`` is create-only and would refuse a directory that
already held a log.

**Manifest.**  Written atomically after every state change.  It binds the
plan's sha256; ``--resume`` refuses a manifest written for different plan
bytes, and skips domains the manifest calls complete whose recorded
outputs still exist with the same sizes (a child whose parent reruns
reruns too).  ``--only`` runs the named domains
(and the domains that share their step) and refuses when a parent outside
the list is not complete.  ``--dry-run`` prints the ordered argv per step
and writes nothing.

The per-domain bookkeeping here is plain Python over a handful of records;
there is no bulk array math in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Sequence

from woof.energy.contracts import (
    ContractError,
    Plan,
    PlanDomain,
    load_plan,
    sha256_file,
)

MANIFEST_SCHEMA = "woof-energy.run-manifest.v1"

#: Domain states a manifest may record.
STATUSES = ("pending", "running", "complete", "failed", "skipped")

#: Log file each step leaves in its ``run_dir``.
LOG_NAME = "energy-run.log"

#: Manifest location relative to the plan's directory.
MANIFEST_RELATIVE = Path("runs") / "manifest.json"

#: Live logs, relative to the plan's directory.
LOG_DIR_RELATIVE = Path("runs") / "logs"

#: Seconds of filesystem timestamp slack when deciding whether an output
#: was written by the step that just ran.
_FRESH_SLACK_S = 2.0

#: Seconds a child gets to stop on its own after Ctrl-C before it is
#: terminated, then killed.
_INTERRUPT_GRACE_S = 30.0


class RunRefusal(RuntimeError):
    """``woof energy run`` will not run this plan as asked."""


@dataclass
class Step:
    """One unit of execution: one or more ``woof`` argv for some domains."""

    step_id: str
    kind: str                       # "go" | "downscale" | "hex"
    domain_ids: list[str]
    commands: list[list[str]]       # woof argv, without the interpreter
    run_dir: str
    parent: str | None = None       # domain whose output forces this step
    notes: list[str] = field(default_factory=list)
    #: environment name -> plan-relative path, exported as an absolute
    #: path to every command of the step (``extra["env_paths"]``).
    env_paths: dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------
# helpers


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def _write_manifest(manifest: dict, path: Path) -> None:
    from woof.supervisor import atomic_write_json

    manifest["updated_utc"] = _utc_now()
    atomic_write_json(path, manifest)


def _read_manifest(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        document = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise RunRefusal(f"run manifest {path} is not JSON: {error}; move it "
                         "aside to start over") from error
    if not isinstance(document, dict) or document.get("schema") != MANIFEST_SCHEMA:
        raise RunRefusal(f"{path} is not a {MANIFEST_SCHEMA} document; move "
                         "it aside to start over")
    if not isinstance(document.get("domains"), dict) or not all(
            isinstance(r, dict) for r in document["domains"].values()):
        raise RunRefusal(f"run manifest {path} has no domains table; move "
                         "it aside to start over")
    if not isinstance(document.get("plan"), dict):
        raise RunRefusal(f"run manifest {path} binds no plan; move it aside "
                         "to start over")
    return document


def _relative(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path)


def _flag_value(argv: Sequence[str], flag: str) -> str | None:
    """Value of ``flag`` in ``argv`` (``--out X`` or ``--out=X``), or None."""

    for index, token in enumerate(argv):
        if token == flag and index + 1 < len(argv):
            return argv[index + 1]
        if token.startswith(flag + "="):
            return token[len(flag) + 1:]
    return None


def _string_argv(value: Any, what: str) -> list[str]:
    if (not isinstance(value, (list, tuple)) or not value
            or not all(isinstance(token, str) for token in value)):
        raise RunRefusal(f"{what} must be a non-empty list of strings; got "
                         f"{value!r}")
    return list(value)


def _step_env(step: "Step", plan_dir: Path) -> dict[str, str]:
    """The step's ``env_paths`` as absolute paths (empty for most steps)."""

    return {name: str((plan_dir / path).resolve())
            for name, path in step.env_paths.items()}


def _child_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for a child ``woof``: this process's, plus its checkout.

    When woof was imported from a source checkout (a ``pyproject.toml``
    beside the package), that checkout leads ``PYTHONPATH`` so ``-m woof``
    in the plan directory runs the same code as this process.  An
    installed woof needs nothing.  ``extra`` (a step's exported paths) is
    laid over this process's environment.
    """

    env = dict(os.environ)
    env.update(extra or {})
    import woof

    root = Path(woof.__file__).resolve().parent.parent
    if (root / "pyproject.toml").exists():
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (str(root) if not existing
                             else os.pathsep.join([str(root), existing]))
    return env


def _execute(argv: list[str], cwd: Path, log,
             env: dict[str, str] | None = None) -> int:
    """Run one command to completion, output to ``log``; return its code.

    Tests replace this.  On Ctrl-C the child (which received the same
    SIGINT from the terminal) is given a grace period, then terminated,
    then killed, and the interrupt is re-raised.  ``env`` is passed only
    for a step that exports paths.
    """

    process = subprocess.Popen(argv, cwd=cwd, stdout=log,
                               stderr=subprocess.STDOUT, env=_child_env(env))
    try:
        return process.wait()
    except KeyboardInterrupt:
        _stop(process)
        raise


def _stop(process: subprocess.Popen) -> None:
    """Let an interrupted child stop, then terminate, then kill it.

    A second Ctrl-C during the grace period skips straight to
    terminate, and nothing here returns while the child still runs.
    """

    try:
        process.wait(timeout=_INTERRUPT_GRACE_S)
        return
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        pass
    process.terminate()
    try:
        process.wait(timeout=10.0)
        return
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        pass
    process.kill()
    process.wait()


# --------------------------------------------------------------------------
# steps


def build_steps(plan: Plan) -> list[Step]:
    """Fold the plan's domains into ordered steps, refusing bad plans."""

    order = plan.run_order()
    by_id = {domain.domain_id: domain for domain in plan.domains}
    steps: list[Step] = []
    grouped: set[str] = set()
    for domain in order:
        if domain.domain_id in grouped:
            continue
        if plan.topology == "hex-swath":
            steps.append(_hex_step(domain))
        elif domain.parent is None:
            members = sorted(
                (d for d in plan.domains
                 if d.parent is None and d.config == domain.config),
                key=lambda d: (d.grid_id or 0, d.domain_id))
            steps.append(_go_step(plan, members))
            grouped.update(d.domain_id for d in members)
            continue
        else:
            steps.append(_downscale_step(plan, domain, by_id[domain.parent]))
        grouped.add(domain.domain_id)
    return steps


def _go_step(plan: Plan, members: list[PlanDomain]) -> Step:
    root = members[0]
    run_dirs = sorted({Path(d.run_dir).as_posix() for d in members})
    if len(run_dirs) > 1:
        raise RunRefusal(
            f"domains {[d.domain_id for d in members]} share config "
            f"{root.config} and so come from one woof go run, which writes "
            f"one directory, but they name different run_dirs {run_dirs}")
    if not plan.resolve(root.config).is_file():
        raise RunRefusal(f"{root.domain_id}: config {root.config} does not "
                         f"exist beside the plan ({plan.resolve(root.config)})")
    notes = []
    if len(members) > 1:
        notes.append(f"one woof go run of {root.config} produces "
                     f"{', '.join(d.domain_id for d in members)}")
    return Step(step_id=root.domain_id, kind="go",
                domain_ids=[d.domain_id for d in members],
                commands=[["go", root.config, "--outdir", root.run_dir]],
                run_dir=root.run_dir, notes=notes)


def _downscale_step(plan: Plan, domain: PlanDomain,
                    parent: PlanDomain) -> Step:
    raw = domain.extra.get("downscale_args")
    if raw is None:
        raise RunRefusal(
            f"{domain.domain_id}: a domain with parent {parent.domain_id} "
            "runs as woof downscale and needs extra['downscale_args']; the "
            "plan has none")
    args = _string_argv(raw, f"{domain.domain_id}: extra['downscale_args']")
    out = _flag_value(args, "--out")
    if out is None:
        raise RunRefusal(f"{domain.domain_id}: extra['downscale_args'] "
                         "carries no --out (woof downscale requires one)")
    run_dir = os.path.normpath(os.path.abspath(plan.resolve(domain.run_dir)))
    out_abs = os.path.normpath(os.path.abspath(plan.directory / out))
    if out_abs != run_dir and not out_abs.startswith(run_dir + os.sep):
        raise RunRefusal(
            f"{domain.domain_id}: downscale --out {out} is not run_dir "
            f"{domain.run_dir} or beneath it, so output_glob could never "
            "see the child's frames")
    return Step(step_id=domain.domain_id, kind="downscale",
                domain_ids=[domain.domain_id],
                commands=[["downscale", parent.run_dir, *args]],
                run_dir=domain.run_dir, parent=parent.domain_id)


def _hex_step(domain: PlanDomain) -> Step:
    raw = domain.extra.get("commands")
    if raw is None:
        raise RunRefusal(f"{domain.domain_id}: a hex-swath domain needs "
                         "extra['commands'] (the woof argv lists to run); "
                         "the plan has none")
    if not isinstance(raw, (list, tuple)) or not raw:
        raise RunRefusal(f"{domain.domain_id}: extra['commands'] must be a "
                         f"non-empty list of argv lists; got {raw!r}")
    commands, notes = [], []
    for index, command in enumerate(raw):
        argv = _string_argv(command,
                            f"{domain.domain_id}: extra['commands'][{index}]")
        if argv[0] == "woof":
            argv = argv[1:]
            notes.append(f"command {index} began with 'woof'; it runs as "
                         "python -m woof without it")
            if not argv:
                raise RunRefusal(f"{domain.domain_id}: extra['commands']"
                                 f"[{index}] is only 'woof'")
        commands.append(argv)
    env_paths = domain.extra.get("env_paths") or {}
    if not isinstance(env_paths, dict) or not all(
            isinstance(k, str) and k and isinstance(v, str) and v
            and not Path(v).is_absolute() for k, v in env_paths.items()):
        raise RunRefusal(f"{domain.domain_id}: extra['env_paths'] must map "
                         "environment names to plan-relative paths; got "
                         f"{env_paths!r}")
    if env_paths:
        notes.append("exports " + ", ".join(
            f"{k}=<plan>/{v}" for k, v in sorted(env_paths.items()))
            + " to every command")
    parent = domain.parent
    return Step(step_id=domain.domain_id, kind="hex",
                domain_ids=[domain.domain_id], commands=commands,
                run_dir=domain.run_dir, parent=parent, notes=notes,
                env_paths=dict(env_paths))


# --------------------------------------------------------------------------
# outputs


def _match_outputs(plan: Plan, domain: PlanDomain, since: float
                   ) -> tuple[list[tuple[Path, int]], list[str]]:
    """``(path, size)`` of the outputs written since ``since``, and notes.

    The glob is honoured in ``run_dir`` first.  Only when that finds no
    file written since ``since`` and the glob has no directory part is it
    searched beneath ``run_dir`` too (``woof go`` writes run-stamped
    folders), and that is noted.  Matches older than ``since`` are counted
    in a note and never returned.
    """

    run_dir = plan.resolve(domain.run_dir)
    if not run_dir.is_dir():
        return [], []
    cutoff = since - _FRESH_SLACK_S

    def scan(pattern: str) -> tuple[list[tuple[Path, int]], int]:
        fresh, stale = [], 0
        for path in sorted(run_dir.glob(pattern)):
            try:
                info = path.stat()
            except OSError:
                continue
            if not path.is_file():
                continue
            if info.st_mtime >= cutoff:
                fresh.append((path, info.st_size))
            else:
                stale += 1
        return fresh, stale

    notes: list[str] = []
    fresh, stale = scan(domain.output_glob)
    basename = "/" not in domain.output_glob and "**" not in domain.output_glob
    if not fresh and basename:
        fresh, stale = scan("**/" + domain.output_glob)
        if fresh:
            notes.append(f"output_glob {domain.output_glob!r} matched "
                         f"beneath {domain.run_dir}, not in it")
    if stale:
        notes.append(f"{stale} file(s) matching output_glob predate this run "
                     "and were not counted")
    return fresh, notes


def _outputs_intact(plan: Plan, record: dict) -> bool:
    outputs = record.get("outputs") or []
    if not outputs:
        return False
    for entry in outputs:
        try:
            path = plan.resolve(entry["path"])
            if not path.is_file() or path.stat().st_size != entry["size"]:
                return False
        except (OSError, KeyError, TypeError):
            return False
    return True


def _parent_frames_dir(plan: Plan, record: dict) -> Path:
    """The single directory holding a parent's recorded history frames."""

    dirs = sorted({str(plan.resolve(entry["path"]).parent)
                   for entry in record.get("outputs") or []})
    if len(dirs) != 1:
        raise RunRefusal(f"parent outputs sit in {len(dirs)} directories "
                         f"{dirs}; woof downscale reads one")
    return Path(dirs[0])


# --------------------------------------------------------------------------
# manifest


def _blank_record() -> dict:
    return {"status": "pending", "step": None, "argv": [], "start_utc": None,
            "end_utc": None, "returncode": None, "log": None, "outputs": [],
            "reason": None, "notes": []}


def _new_manifest(plan: Plan, plan_sha: str, steps: list[Step]) -> dict:
    domains = {}
    for step in steps:
        for domain_id in step.domain_ids:
            record = _blank_record()
            record["step"] = step.step_id
            record["argv"] = [list(c) for c in step.commands]
            domains[domain_id] = record
    return {"schema": MANIFEST_SCHEMA,
            "plan": {"path": plan.path.name, "sha256": plan_sha},
            "topology": plan.topology,
            "python": sys.executable,
            "created_utc": _utc_now(), "updated_utc": None,
            "order": [s.step_id for s in steps],
            "notes": [], "domains": domains}


def _dependents(plan: Plan, failed: set[str]) -> dict[str, str]:
    """Every domain downstream of ``failed`` -> the failed root it waits on."""

    children: dict[str, list[str]] = {}
    for domain in plan.domains:
        if domain.parent is not None:
            children.setdefault(domain.parent, []).append(domain.domain_id)
    out: dict[str, str] = {}
    for root in failed:
        stack = list(children.get(root, ()))
        while stack:
            current = stack.pop()
            if current in out or current in failed:
                continue
            out[current] = root
            stack.extend(children.get(current, ()))
    return out


# --------------------------------------------------------------------------
# the run


def run_plan(plan_path: Path, *, dry_run: bool = False,
             only: Sequence[str] | None = None, resume: bool = False) -> dict:
    """Run the plan's domains in parent-first order; return the manifest.

    With ``dry_run`` nothing is written or launched and the returned record
    lists the ordered argv per step instead.  Raises :class:`RunRefusal`
    (or :class:`~woof.energy.contracts.ContractError` for a bad plan) when
    the request cannot be honoured.
    """

    plan_path = Path(plan_path)
    plan = load_plan(plan_path)
    plan_dir = plan.directory
    plan_sha = sha256_file(plan_path)
    steps = build_steps(plan)
    step_of = {d: s for s in steps for d in s.domain_ids}
    manifest_path = plan_dir / MANIFEST_RELATIVE

    previous = _read_manifest(manifest_path)
    previous_matches = (previous is not None
                        and previous.get("plan", {}).get("sha256") == plan_sha)
    if resume and previous is not None and not previous_matches:
        raise RunRefusal(
            f"{manifest_path} was written for plan sha256 "
            f"{previous.get('plan', {}).get('sha256')}, but {plan_path} is now "
            f"{plan_sha}; the plan changed, so its old runs cannot be "
            "resumed (rerun without --resume to start over, after moving the "
            "old run directories aside)")
    if only and previous is not None and not previous_matches:
        raise RunRefusal(
            f"{manifest_path} belongs to a different plan (sha256 "
            f"{previous.get('plan', {}).get('sha256')} vs {plan_sha}); --only "
            "cannot trust its parent records")

    manifest = _new_manifest(plan, plan_sha, steps)
    if previous_matches and (resume or only):
        for domain_id, record in previous["domains"].items():
            if domain_id in manifest["domains"]:
                manifest["domains"][domain_id].update(record)
        manifest["created_utc"] = previous.get("created_utc",
                                               manifest["created_utc"])
    if resume and previous is None:
        manifest["notes"].append("--resume found no manifest; started fresh")

    def complete_now(domain_id: str) -> bool:
        record = manifest["domains"][domain_id]
        return record["status"] == "complete" and _outputs_intact(plan, record)

    # Which steps run.
    selected_steps = list(steps)
    if only:
        unknown = [d for d in only if d not in step_of]
        if unknown:
            raise RunRefusal(f"--only names domains not in the plan: {unknown}")
        wanted = {step_of[d].step_id for d in only}
        selected_steps = [s for s in steps if s.step_id in wanted]
        chosen = {d for s in selected_steps for d in s.domain_ids}
        for step in selected_steps:
            if (step.parent is not None and step.parent not in chosen
                    and not complete_now(step.parent)):
                status = manifest["domains"][step.parent]["status"]
                raise RunRefusal(
                    f"--only {step.step_id}: its parent {step.parent} is not "
                    f"complete (manifest status {status!r}, or its recorded "
                    "outputs changed); run the parent first or add it to "
                    "--only")

    resumable = set()
    if resume:
        selected_ids = {s.step_id for s in selected_steps}
        for step in selected_steps:
            # A child whose parent reruns is forced by new parent output,
            # so it reruns too even when its own outputs are intact.
            parent_step = (step_of[step.parent].step_id
                           if step.parent is not None else None)
            parent_settled = (parent_step is None
                              or parent_step not in selected_ids
                              or parent_step in resumable)
            if parent_settled and all(complete_now(d)
                                      for d in step.domain_ids):
                resumable.add(step.step_id)
            else:
                for domain_id in step.domain_ids:
                    record = manifest["domains"][domain_id]
                    if record["status"] == "complete":
                        record["notes"] = list(record.get("notes") or []) + [
                            "recorded outputs changed or vanished; rerun"]
                    record["status"] = "pending"

    if dry_run:
        return _dry_run_record(plan, plan_sha, steps, selected_steps,
                               resumable, manifest, manifest_path)

    # Reset what is about to run so stale records do not linger.
    for step in selected_steps:
        if step.step_id in resumable:
            for domain_id in step.domain_ids:
                manifest["domains"][domain_id]["notes"] = list(
                    manifest["domains"][domain_id].get("notes") or []) + [
                    "complete in the manifest with intact outputs; "
                    "skipped by --resume"]
            continue
        for domain_id in step.domain_ids:
            record = _blank_record()
            record["step"] = step.step_id
            record["argv"] = [list(c) for c in step.commands]
            manifest["domains"][domain_id] = record
    _write_manifest(manifest, manifest_path)

    selected_domains = {d for s in selected_steps for d in s.domain_ids}
    manifest["interrupted"] = False
    for step in selected_steps:
        if step.step_id in resumable:
            continue
        records = [manifest["domains"][d] for d in step.domain_ids]
        skipped = [r for r in records if r["status"] == "skipped"]
        if skipped:
            # A parent failed earlier in this invocation; the cascade below
            # already recorded why.
            for record in records:
                record["status"] = "skipped"
                record["reason"] = skipped[0]["reason"]
            _write_manifest(manifest, manifest_path)
            continue
        try:
            ok = _run_step(plan, step, manifest, manifest_path)
        except KeyboardInterrupt:
            for record in records:
                record["status"] = "failed"
                record["reason"] = "interrupted (Ctrl-C)"
                record["end_utc"] = record["end_utc"] or _utc_now()
            manifest["interrupted"] = True
            _write_manifest(manifest, manifest_path)
            break
        for domain_id, root in _dependents(plan, set(step.domain_ids)).items():
            record = manifest["domains"][domain_id]
            if domain_id in selected_domains:
                if not ok:
                    why = manifest["domains"][root]["reason"] or "failed"
                    record["status"] = "skipped"
                    record["reason"] = f"parent {root} failed: {why}"
            elif record["status"] == "complete":
                # Outside --only: its output came from the parent run this
                # one replaced, so it is due to run again.
                record["status"] = "pending"
                record["notes"] = list(record.get("notes") or []) + [
                    f"parent {root} ran again at {_utc_now()}; these outputs "
                    "were forced by an earlier parent run"]
        _write_manifest(manifest, manifest_path)
    manifest["manifest_path"] = str(manifest_path)
    manifest["selected"] = [d for s in selected_steps for d in s.domain_ids]
    _write_manifest(manifest, manifest_path)
    return manifest


def _resolved_commands(plan: Plan, step: Step, manifest: dict
                       ) -> list[list[str]]:
    """The step's argv with a downscale parent replaced by its frame dir."""

    if step.kind != "downscale":
        return [list(c) for c in step.commands]
    parent_record = manifest["domains"][step.parent]
    frames = _parent_frames_dir(plan, parent_record)
    command = list(step.commands[0])
    command[1] = _relative(frames, plan.directory)
    return [command]


def _run_step(plan: Plan, step: Step, manifest: dict,
              manifest_path: Path) -> bool:
    plan_dir = plan.directory
    records = [manifest["domains"][d] for d in step.domain_ids]

    def fail(reason: str, returncode: int | None = None) -> bool:
        for record in records:
            record["status"] = "failed"
            record["reason"] = reason
            record["returncode"] = returncode
            record["end_utc"] = _utc_now()
        _write_manifest(manifest, manifest_path)
        return False

    try:
        commands = _resolved_commands(plan, step, manifest)
    except RunRefusal as error:
        return fail(f"cannot locate the parent's history frames: {error}")
    notes = list(step.notes)
    if step.kind == "downscale":
        out = plan_dir / _flag_value(commands[0], "--out")
        occupied = out.is_symlink() or (out.exists() and (
            not out.is_dir() or any(out.iterdir())))
        if occupied:
            try:
                aside = _move_aside(out)
            except (RunRefusal, OSError) as error:
                return fail(f"--out {_relative(out, plan_dir)} holds an "
                            "earlier attempt and could not be moved aside: "
                            f"{error}")
            notes.append(f"--out {_relative(out, plan_dir)} held an earlier "
                         f"attempt (woof downscale's --out is create-only); "
                         f"moved to {_relative(aside, plan_dir)}")

    live_log = plan_dir / LOG_DIR_RELATIVE / f"{step.step_id}.log"
    live_log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for record in records:
        record["status"] = "running"
        record["argv"] = commands
        record["start_utc"] = _utc_now()
        record["log"] = _relative(live_log, plan_dir)
        record["notes"] = list(notes)
    _write_manifest(manifest, manifest_path)

    returncode = 0
    exported = _step_env(step, plan_dir)
    extra = {"env": exported} if exported else {}
    try:
        with open(live_log, "w") as log:
            for argv in commands:
                full = [sys.executable, "-m", "woof", *argv]
                log.write(f"# {_utc_now()} woof energy run [{step.step_id}]: "
                          f"woof {' '.join(argv)}\n")
                log.flush()
                returncode = _execute(full, plan_dir, log, **extra)
                log.write(f"# {_utc_now()} exit {returncode}\n")
                log.flush()
                if returncode != 0:
                    break
    finally:
        _publish_log(plan, step, live_log, records)

    end = _utc_now()
    ok = True
    for domain_id, record in zip(step.domain_ids, records):
        domain = plan.domain(domain_id)
        record["returncode"] = returncode
        record["end_utc"] = end
        fresh, found_notes = _match_outputs(plan, domain, started)
        record["notes"].extend(found_notes)
        record["outputs"] = [{"path": _relative(p, plan_dir), "size": size}
                             for p, size in fresh]
        if returncode != 0:
            record["status"] = "failed"
            record["reason"] = f"exit status {returncode}"
            ok = False
        elif not fresh:
            record["status"] = "failed"
            record["reason"] = (f"exit status 0 but output_glob "
                                f"{domain.output_glob!r} matched no new file "
                                f"in {domain.run_dir}")
            ok = False
        else:
            record["status"] = "complete"
            record["reason"] = None
    _write_manifest(manifest, manifest_path)
    return ok


def _move_aside(path: Path) -> Path:
    """Rename an occupied output directory to a dated sibling; return it."""

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ")
    for ordinal in range(1, 100):
        suffix = "" if ordinal == 1 else f"-{ordinal:02d}"
        aside = path.with_name(f"{path.name}.previous-{stamp}{suffix}")
        if not aside.exists() and not aside.is_symlink():
            path.rename(aside)
            return aside
    raise RunRefusal(f"could not find a free name to move {path} aside")


def _publish_log(plan: Plan, step: Step, live_log: Path,
                 records: list[dict]) -> None:
    """Copy the live log to ``<run_dir>/energy-run.log`` when it exists."""

    run_dir = plan.resolve(step.run_dir)
    if not run_dir.is_dir() or not live_log.exists():
        return
    target = run_dir / LOG_NAME
    try:
        shutil.copyfile(live_log, target)
    except OSError:
        return
    for record in records:
        record["log"] = _relative(target, plan.directory)


def _dry_run_record(plan: Plan, plan_sha: str, steps: list[Step],
                    selected: list[Step], resumable: set[str],
                    manifest: dict, manifest_path: Path) -> dict:
    out_steps = []
    for step in selected:
        notes = list(step.notes)
        if step.kind == "downscale":
            parent_record = manifest["domains"][step.parent]
            commands = None
            if _outputs_intact(plan, parent_record):
                try:
                    commands = _resolved_commands(plan, step, manifest)
                except RunRefusal:
                    commands = None
            if commands is None:
                commands = [list(c) for c in step.commands]
                notes.append(f"the parent argument {step.commands[0][1]} is "
                             f"replaced at run time by the directory holding "
                             f"{step.parent}'s history frames")
        else:
            commands = [list(c) for c in step.commands]
        out_steps.append({
            "step": step.step_id, "kind": step.kind,
            "domains": list(step.domain_ids), "run_dir": step.run_dir,
            "parent": step.parent,
            "action": "skip (complete, --resume)" if step.step_id in resumable
            else "run",
            "argv": commands, "notes": notes})
    return {"schema": MANIFEST_SCHEMA, "dry_run": True,
            "plan": {"path": str(plan.path), "sha256": plan_sha},
            "topology": plan.topology, "cwd": str(plan.directory),
            "python": [sys.executable, "-m", "woof"],
            "manifest_path": str(manifest_path),
            "steps": out_steps, "notes": list(manifest["notes"])}


# --------------------------------------------------------------------------
# CLI


def main(args) -> int:
    try:
        result = run_plan(Path(args.plan), dry_run=bool(args.dry_run),
                          only=args.only, resume=bool(args.resume))
    except (RunRefusal, ContractError) as error:
        print(f"woof energy run: refused: {error}", file=sys.stderr)
        return 2
    if result.get("dry_run"):
        print(json.dumps(result, indent=2, default=str))
        return 0
    selected = result["selected"]
    statuses = {d: result["domains"][d]["status"] for d in selected}
    failed = {d: result["domains"][d]["reason"] for d in selected
              if statuses[d] in ("failed", "skipped")}
    summary = {
        "command": "woof energy run",
        "plan": str(args.plan),
        "plan_sha256": result["plan"]["sha256"],
        "manifest": result["manifest_path"],
        "ok": not failed and not result.get("interrupted"),
        "interrupted": bool(result.get("interrupted")),
        "domains": statuses,
        "problems": failed,
        "notes": result["notes"],
    }
    print(json.dumps(summary, indent=2, default=str))
    if result.get("interrupted"):
        return 130
    return 0 if summary["ok"] else 1


__all__ = ["MANIFEST_SCHEMA", "STATUSES", "LOG_NAME", "RunRefusal", "Step",
           "build_steps", "run_plan", "main"]
