"""Concurrent, process-isolated production forecast orchestration.

The orchestrator resolves physical devices through the supervisor's
``nvidia-smi`` inventory and starts each fresh child with one physical UUID in
``CUDA_VISIBLE_DEVICES``.  Check and prepared-runner wrappers acquire the
existing machine-wide UUID lock before importing their CUDA-facing target.
Inside a masked child, CUDA device ordinal zero is therefore the selected
process-local device; isolation does not depend on the import state of the
calling ``woof.cli`` process.

Each run gets independent output and temporary directories.  GPU kernel
caches (CuPy's and the CUDA driver's JIT cache) are shared and persistent
by default (``[cache] kernels`` / ``--kernel-cache``,
:class:`woof.gpu_packing.KernelCache`), so only the first run on a machine
compiles; ``per-run`` restores one new, empty cache per run.  A module-form
run that names a prepared root but no binding digests is bound at plan load
exactly as ``woof sim`` binds that root (:func:`_bind_prepared_runner`).
A versioned JSON summary is published from a fully written
sibling through atomic create-only publication after every child has finished
(or after the parent is interrupted).  Interrupts do not terminate children:
a supervised ``woof run`` may itself own a fresh CUDA worker, so killing only
the visible supervisor could orphan work.  The interrupted summary records
every PID whose completion was not observed.

Packing several runs onto one GPU (``[devices] runs_per_gpu`` /
``--runs-per-gpu N|auto``) is opt-in; the default, 1, is the behaviour
above, unchanged: one run per physical GPU and a second run resolving to
the same card is refused.  With N > 1 or ``auto`` (at most
:data:`woof.gpu_packing.AUTO_MAX_SLOTS_PER_GPU` per card) runs on one card
are admitted by memory (:data:`PACKED_ADMISSION_RULE`), queue while they
do not fit, and each child gets its own ``WOOF_GPU_LOCK_ROOT`` so the
siblings' own ``woof run`` UUID locks do not refuse each other, while the
parent holds each packed card's machine-wide UUID lock (when no other woof
process already does) so no other woof run starts beside them; every
child's supervisor still prices itself against the card's measured free
memory (:func:`woof.supervisor.shared_gpu_admission`) before its worker
starts.  ``--mps`` adds the CUDA MPS control daemon
(:class:`woof.gpu_packing.MpsDaemon`).
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import tomllib
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

from woof.filesystem_paths import publish_new
from woof.gpu_packing import (AUTO_MAX_SLOTS_PER_GPU, DEFAULT_MEM_MARGIN_GIB,
                              SLOTS_AUTO, KernelCache, MPSUnavailable,
                              MpsDaemon, parse_count_or_auto,
                              parse_kernel_cache)
from woof.supervisor import (GPUFileLock, GPUIdentity, GPU_LOCK_ROOT_ENV,
                              SHARED_ADMISSION_BASIS,
                              SHARED_INPUT_AUTHORITY_ROOT_ENV,
                              measured_device_memory,
                              preflight_exclusive_gpu,
                              priced_reservation_bytes, query_gpus, utc_now)


PLAN_SCHEMA = "gpuwm.multi-run-plan/v1"
SUMMARY_SCHEMA = "gpuwm.multi-run-summary/v1"
PREFLIGHT_MODES = ("estimate", "alloc", "off")
AUTHORITY_MONITOR_POLL_SECONDS = 0.05
SUPPORTED_PREPARED_RUNNERS = frozenset({
    "woof.prepared_domain_tree_forecast",
    "woof.prepared_single_domain_forecast",
})
_RUNNER_PATH_FLAGS = {
    "woof.prepared_domain_tree_forecast": {
        "required": ("--prepared-root", "--experiment-config"),
        "optional": ("--restart",),
    },
    "woof.prepared_single_domain_forecast": {
        "required": (
            "--prepared-root", "--experiment-config", "--wps-namelist"),
        "optional": ("--domain-bundle",),
    },
}
_NON_FORECAST_ARGUMENTS = frozenset({
    "--materialize-authorities", "--show-capabilities",
})
_TOP_LEVEL_KEYS = frozenset(
    {"schema", "summary", "preflight", "devices", "cache", "run"})
_DEVICES_KEYS = frozenset({"runs_per_gpu", "mem_margin_gib", "mps"})
_CACHE_KEYS = frozenset({"kernels"})
#: The digest options each prepared runner binds its preparation with.
#: Their values are lowercase SHA-256 digests, never paths, so they are
#: outside the ``{inputN}`` path rules.
_RUNNER_BINDING_FLAGS = {
    "woof.prepared_single_domain_forecast": (
        "--prepared-head-sha256", "--proof-sha256",
        "--source-manifest-sha256", "--prepared-content-sha256"),
    "woof.prepared_domain_tree_forecast": (
        "--prepared-head-sha256", "--preparation-receipt-sha256",
        "--experiment-config-sha256"),
}
_HEAD_BINDING_FLAG = "--prepared-head-sha256"
_SEALED_BINDING_FLAGS = {
    "woof.prepared_single_domain_forecast": (
        "--proof-sha256", "--prepared-content-sha256"),
    "woof.prepared_domain_tree_forecast": ("--preparation-receipt-sha256",),
}
_TREE_RUNNER = "woof.prepared_domain_tree_forecast"
#: How a module run's preparation binding is derived, verbatim in receipts.
PREPARED_BINDING_BASIS = (
    "woof.stage_cli.sim_command's binding of the declared --prepared-root, "
    "the one `woof sim` and `woof go --prepared-root` give the same root: "
    "a sealed preparation binds its proof (single domain: proof, source "
    "manifest and prepared-cache digests; tree: its preparation receipt "
    "and the experiment config digest), a chained preparation still being "
    "prepared binds its head. Digests the plan gives are kept and must "
    "equal these; a given --prepared-head-sha256 on a sealed chained "
    "preparation must be the head its proof records")
#: Seconds between re-measurements of a card while the run at the head of
#: its queue waits: a sibling finishing re-evaluates at once, this catches
#: memory freed by processes this multi-run does not own.
ADMISSION_RECHECK_SECONDS = 5.0
#: The packed admission rule, recorded verbatim in every packed receipt.
PACKED_ADMISSION_RULE = (
    "A run starts on its GPU only when (1) fewer than runs_per_gpu of this "
    "multi-run's runs hold the card; (2) the priced peaks of this "
    "multi-run's live runs on the card, plus this run's priced peak, plus "
    "the margin, fit the card's NVML total; and (3) this run's priced peak "
    "plus the margin fit the card's MEASURED free memory less the growth "
    "its live siblings may still make (each sibling's priced peak minus "
    "the memory NVML attributes to its process tree, never below zero), so "
    "memory held by processes this multi-run does not own is respected. "
    "Runs on one card are admitted in plan order. A run that does not fit "
    "waits for a sibling on its card to finish; one that does not fit "
    "while none of this multi-run's runs hold the card is refused, because "
    "only processes this multi-run does not own could free that memory, "
    "except within one recheck of a sibling on the card finishing, when "
    "it waits for that sibling's memory to be released. An unpriced run "
    "runs alone on its card. Alloc-mode config checks are admitted by the "
    "same rule before the forecasts; estimate-mode checks hold no forecast "
    "memory and are held only to runs_per_gpu.")
#: What a config-form run's priced peak is.
CONFIG_PRICING_BASIS = (
    "woof.supervisor.priced_reservation_bytes of the captured config "
    "bytes: the admission_estimate peak envelope woof check reviews and "
    "the run's own supervisor re-prices at launch")
#: What a module-form run's priced peak is.
MODULE_PRICING_BASIS = (
    "the prepared runner's own reservation: priced_reservation_bytes of "
    "its --experiment-config with its forcing source and prepared land "
    "cover, the number its locked worker re-prices at launch")
_COMMON_RUN_KEYS = frozenset(
    {"name", "device", "outdir", "scratch", "cache"})
_RUN_KEYS = frozenset(
    {*_COMMON_RUN_KEYS, "config", "module", "args", "inputs"})


@dataclass(frozen=True)
class InputBinding:
    """Pre-launch content authority for one declared read-only input."""

    path: Path
    kind: str
    sha256: str | None
    captured_bytes: bytes | None = field(default=None, repr=False,
                                         compare=False)
    captured_signature: tuple[int, int, int, int, int] | None = field(
        default=None, repr=False, compare=False)

    def as_dict(self) -> dict[str, object]:
        return {
            "binding": (
                "prelaunch_sha256" if self.sha256 is not None
                else "delegated_to_prepared_runner_receipt"),
            "kind": self.kind,
            "path": str(self.path),
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class RunSpec:
    """One validated plan entry before its selector is resolved."""

    name: str
    device_selector: str
    outdir: Path
    scratch_dir: Path
    cache: Path
    config: Path | None
    module: str | None
    arguments: tuple[str, ...]
    inputs: tuple[Path, ...]
    input_bindings: tuple[InputBinding, ...] = ()
    #: The argv template as the plan wrote it; ``arguments`` is the one
    #: launched (with any derived binding digests inserted).
    plan_arguments: tuple[str, ...] | None = None
    #: What :func:`_bind_prepared_runner` derived and verified.
    prepared_binding: Mapping[str, object] | None = field(
        default=None, compare=False, hash=False)

    @property
    def log(self) -> Path:
        return self.scratch_dir / "gpuwm-run.log"

    @property
    def preflight_log(self) -> Path:
        return self.scratch_dir / "gpuwm-check.log"

    @property
    def captured_config(self) -> Path:
        """Create-only immutable payload consumed by every config phase."""

        return self.scratch_dir / "captured-config.toml"

    @property
    def is_config_run(self) -> bool:
        return self.config is not None


@dataclass(frozen=True)
class PackingRequest:
    """How many of a plan's runs may share one physical GPU.

    ``runs_per_gpu`` 1 (the default) is exclusive: today's one run per
    card.  Above 1, or ``auto`` (at most :data:`AUTO_MAX_SLOTS_PER_GPU`),
    runs share a card under :data:`PACKED_ADMISSION_RULE` with
    ``mem_margin_gib`` kept free beside their priced peaks (default
    :data:`DEFAULT_MEM_MARGIN_GIB`, an assumption: driver, MPS server and
    allocator fragmentation a priced peak does not hold).  ``sources``
    names where each setting came from (the plan's ``[devices]`` table,
    a flag, or the default), for the receipt.
    """

    runs_per_gpu: int | str = 1
    mem_margin_gib: float | None = None
    mps: bool = False
    mps_dir: Path | None = None
    mps_active_thread_percentage: int | None = None
    sources: tuple[tuple[str, str], ...] = ()

    @property
    def packed(self) -> bool:
        return self.runs_per_gpu != 1

    @property
    def cap(self) -> int:
        """Most of this plan's runs one card holds at once."""

        return (AUTO_MAX_SLOTS_PER_GPU if self.runs_per_gpu == SLOTS_AUTO
                else int(self.runs_per_gpu))

    @property
    def margin_gib(self) -> float:
        return (DEFAULT_MEM_MARGIN_GIB if self.mem_margin_gib is None
                else float(self.mem_margin_gib))

    @property
    def margin_bytes(self) -> int:
        return int(round(self.margin_gib * 1024 ** 3))


@dataclass(frozen=True)
class MultiRunPlan:
    """A parsed plan whose paths are absolute and collision-checked."""

    path: Path
    sha256: str
    captured_signature: tuple[int, int, int, int, int]
    summary: Path
    input_authority_store: Path
    preflight: str
    runs: tuple[RunSpec, ...]
    packing: PackingRequest = PackingRequest()
    kernel_cache: KernelCache = KernelCache()


@dataclass(frozen=True)
class PlannedRun:
    """A plan entry bound to one physical GPU.

    ``environment_overrides`` are applied last by :func:`child_environment`
    (a packed plan's CUDA MPS variables).  ``kernel_cache`` says whether
    the child shares the persistent GPU kernel cache (the default) or
    starts from its own empty ``<cache>/cupy`` and ``<cache>/cuda``."""

    spec: RunSpec
    gpu: GPUIdentity
    lock_root: Path | None = None
    input_authority_store: Path | None = None
    environment_overrides: tuple[tuple[str, str], ...] = ()
    kernel_cache: KernelCache = KernelCache()


@dataclass
class ProcessRecord:
    """Observed lifecycle of one check or run child."""

    name: str
    phase: str
    command: tuple[str, ...]
    log: Path
    status: str = "not_started"
    pid: int | None = None
    started_at_utc: str | None = None
    completed_at_utc: str | None = None
    duration_seconds: float | None = None
    exit_code: int | None = None
    error: str | None = None
    _started_monotonic: float | None = field(default=None, repr=False)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "command": list(self.command),
            "completed_at_utc": self.completed_at_utc,
            "duration_seconds": self.duration_seconds,
            "exit_code": self.exit_code,
            "log": str(self.log),
            "pid": self.pid,
            "started_at_utc": self.started_at_utc,
            "status": self.status,
        }
        if self.error is not None:
            payload["error"] = self.error
        return payload


@dataclass(frozen=True)
class GroupOutcome:
    """Results from one concurrently launched process group."""

    records: tuple[ProcessRecord, ...]
    interrupted: bool
    started_at_utc: str | None = None
    completed_at_utc: str | None = None
    wall_seconds: float | None = None


class GroupExecutionError(RuntimeError):
    """Guaranteed carrier for a partial group ledger and original failure."""

    def __init__(self, phase: str, outcome: GroupOutcome,
                 original: BaseException):
        super().__init__(f"{type(original).__name__}: {original}")
        self.phase = phase
        self.outcome = outcome
        self.original = original


@dataclass
class OrchestrationState:
    """Mutable stage ledger used for accurate interruption receipts."""

    stage: str = "plan_load"
    summary_capable: bool = False
    claimed_directories: list[Path] = field(default_factory=list)
    error: str | None = None


def _orchestration_record(
        state: OrchestrationState, plan: MultiRunPlan,
        preflight: GroupOutcome | None,
        execution: GroupOutcome | None) -> dict[str, object]:
    records = []
    if preflight is not None:
        records.extend(preflight.records)
    if execution is not None:
        records.extend(execution.records)
    known_pids = sorted({
        record.pid for record in records if record.pid is not None})
    unobserved_pids = sorted({
        record.pid for record in records
        if record.pid is not None and record.status in {
            "launch_unobserved", "running_unobserved"}})
    existing_run_directories = []
    for run in plan.runs:
        for path in (
                run.outdir, run.scratch_dir, run.cache,
                run.cache / "cupy", run.cache / "cuda"):
            if path.is_dir():
                existing_run_directories.append(str(path))
    payload: dict[str, object] = {
        "existing_run_directories": sorted(set(existing_run_directories)),
        "known_child_pids": known_pids,
        "shared_input_authority_store": str(plan.input_authority_store),
        "parent_claimed_directories": [
            str(path) for path in state.claimed_directories],
        "stage": state.stage,
        "summary_create_only_capability_proven": state.summary_capable,
        "unobserved_child_pids": unobserved_pids,
    }
    if state.error is not None:
        payload["error"] = state.error
    return payload


def _input_authority_store_record(plan: MultiRunPlan) -> dict[str, object]:
    """Receipt for the plan-wide, SHA-keyed immutable content store."""

    entries = []
    if plan.input_authority_store.is_dir():
        for path in sorted(plan.input_authority_store.iterdir()):
            if (not path.is_file() or len(path.name) != 64
                    or any(c not in "0123456789abcdef" for c in path.name)):
                continue
            entries.append({
                "bytes": path.stat().st_size,
                "path": str(path),
                "sha256": path.name,
            })
    return {
        "content_deduplication": "one_create_only_file_per_sha256",
        "entries": entries,
        "path": str(plan.input_authority_store),
        "status": (
            "MISSING" if not plan.input_authority_store.is_dir()
            else "POPULATED" if entries else "EMPTY"),
    }


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    result = value.strip()
    if any(character in result for character in "\r\n"):
        raise ValueError(f"{label} must fit on one line")
    return result


def _device_selector(value: object, label: str) -> str:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a GPU index or full GPU UUID")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{label} GPU index must be nonnegative")
        return str(value)
    selector = _nonempty_string(value, label)
    if selector.isdigit() or selector.startswith("GPU-"):
        return selector
    raise ValueError(
        f"{label} must be a nonnegative GPU index or full GPU UUID, got "
        f"{selector!r}")


def _module_name(value: object, label: str) -> str:
    module = _nonempty_string(value, label)
    if module not in SUPPORTED_PREPARED_RUNNERS:
        raise ValueError(
            f"{label} must be one of the supported prepared forecast "
            f"runners {sorted(SUPPORTED_PREPARED_RUNNERS)}, got {module!r}")
    return module


def _argument_list(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a TOML array of argv strings")
    arguments = tuple(
        _nonempty_string(argument, f"{label}[{index}]")
        for index, argument in enumerate(value))
    return arguments


def _canonical_outdir_value(arguments: Sequence[str], label: str) -> str:
    alternates = [
        argument for argument in arguments
        if argument.startswith("--out") and argument != "--outdir"]
    if alternates:
        raise ValueError(
            f"{label} has alternate output option(s) {alternates}; use "
            "exactly '--outdir', followed by its separate value")
    positions = [
        index for index, argument in enumerate(arguments)
        if argument == "--outdir"]
    if len(positions) != 1:
        raise ValueError(
            f"{label} must contain exactly one canonical --outdir option, "
            f"got {len(positions)}")
    index = positions[0]
    if index + 1 >= len(arguments):
        raise ValueError(f"{label} --outdir is missing its value")
    return arguments[index + 1]


def _validate_outdir_template(arguments: Sequence[str], label: str) -> None:
    value = _canonical_outdir_value(arguments, label)
    if value != "{outdir}" or sum(
            argument.count("{outdir}") for argument in arguments) != 1:
        raise ValueError(
            f"{label} must use exactly '--outdir', '{{outdir}}' once; no "
            "fixed, embedded, duplicate, or alternate output path is allowed")


def _validate_rendered_outdir(arguments: Sequence[str], outdir: Path,
                              label: str) -> None:
    value = _canonical_outdir_value(arguments, label)
    if value != str(outdir):
        raise ValueError(
            f"{label} --outdir value {value!r} does not match the "
            f"collision-checked output root {str(outdir)!r}")


def _runner_path_values(module: str, arguments: Sequence[str],
                        label: str) -> dict[str, str]:
    if any(argument in _NON_FORECAST_ARGUMENTS for argument in arguments):
        raise ValueError(
            f"{label} selects a non-forecast mode; multi-run only launches "
            "prepared forecasts")
    specification = _RUNNER_PATH_FLAGS[module]
    required = specification["required"]
    optional = specification["optional"]
    flags = (*required, *optional)
    for argument in arguments:
        if not argument.startswith("--"):
            continue
        option = argument.split("=", 1)[0]
        for flag in flags:
            if argument.startswith(f"{flag}=") or (
                    option != flag and flag.startswith(option)):
                raise ValueError(
                    f"{label} uses alternate path option {argument!r}; "
                    f"use the complete separate form {flag!r}, VALUE")
    values: dict[str, str] = {}
    for flag in flags:
        positions = [
            index for index, argument in enumerate(arguments)
            if argument == flag]
        maximum = 1
        minimum = 1 if flag in required else 0
        if not minimum <= len(positions) <= maximum:
            expectation = "exactly once" if minimum else "at most once"
            raise ValueError(
                f"{label} must contain {flag} {expectation}, got "
                f"{len(positions)}")
        if not positions:
            continue
        index = positions[0]
        if index + 1 >= len(arguments):
            raise ValueError(f"{label} {flag} is missing its value")
        values[flag] = arguments[index + 1]
    return values


def _validate_runner_input_templates(
        module: str, arguments: Sequence[str], input_count: int,
        label: str) -> None:
    values = _runner_path_values(module, arguments, label)
    used: list[str] = []
    for flag, value in values.items():
        match = re.fullmatch(r"\{input(\d+)\}", value)
        if match is None:
            raise ValueError(
                f"{label} {flag} value must be one exact declared "
                f"{{inputN}} placeholder, got {value!r}")
        index = int(match.group(1))
        if index >= input_count:
            raise ValueError(
                f"{label} {flag} refers to undeclared input {index}")
        used.append(match.group(0))
    occurrences = [
        match.group(0) for argument in arguments
        for match in re.finditer(r"\{input\d+\}", argument)]
    if sorted(occurrences) != sorted(used):
        raise ValueError(
            f"{label} may use {{inputN}} placeholders only as exact values "
            "of the prepared runner's declared path options")
    missing = sorted(
        set(range(input_count))
        - {int(token[6:-1]) for token in used})
    if missing:
        raise ValueError(
            f"{label} does not bind declared input index(es) {missing} to a "
            "prepared-runner path option")


def _validate_rendered_runner_inputs(
        module: str, arguments: Sequence[str], inputs: Sequence[Path],
        label: str) -> None:
    values = _runner_path_values(module, arguments, label)
    keys = [os.path.normcase(str(path)) for path in inputs]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{label} contains duplicate declared input paths")
    declared = {str(path) for path in inputs}
    used = set(values.values())
    escaped = sorted(used - declared)
    if escaped:
        raise ValueError(
            f"{label} contains path value(s) outside its declared inputs: "
            f"{escaped}")
    missing = sorted(declared - used)
    if missing:
        raise ValueError(
            f"{label} does not bind declared input path(s): {missing}")


def _validate_placeholders(arguments: Sequence[str], input_count: int,
                           label: str) -> None:
    fixed = {"{outdir}", "{scratch}", "{cache}"}
    for argument in arguments:
        recognized = fixed | set(re.findall(r"\{input\d+\}", argument))
        for match in re.finditer(r"\{[A-Za-z_][A-Za-z0-9_]*\}", argument):
            token = match.group(0)
            if token not in recognized:
                raise ValueError(
                    f"{label} argument {argument!r} has unknown "
                    f"placeholder {token!r}")
        for match in re.finditer(r"\{input(\d+)\}", argument):
            if int(match.group(1)) >= input_count:
                raise ValueError(
                    f"{label} argument {argument!r} refers to undeclared "
                    f"input {match.group(1)}")
        remainder = argument
        for token in recognized:
            remainder = remainder.replace(token, "")
        if any(prefix in remainder for prefix in (
                "{outdir", "{scratch", "{cache", "{input")):
            raise ValueError(
                f"{label} argument {argument!r} has a malformed reserved "
                "placeholder")


def _input_paths(base: Path, value: object, label: str) -> tuple[Path, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(
            f"{label} must be a non-empty TOML array of read-only paths")
    paths = tuple(_plan_path(base, item, f"{label}[{index}]")
                  for index, item in enumerate(value))
    keys = [os.path.normcase(str(path)) for path in paths]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{label} contains a duplicate declared input path")
    return paths


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _authority_signature(path: Path) -> tuple[int, int, int, int, int]:
    metadata = path.stat()
    return (
        int(metadata.st_dev), int(metadata.st_ino), int(metadata.st_size),
        int(metadata.st_mtime_ns), int(metadata.st_ctime_ns))


def _bind_input(path: Path, *, capture_bytes: bool = False) -> InputBinding:
    if path.is_file():
        before = _authority_signature(path)
        payload = path.read_bytes() if capture_bytes else None
        digest = (hashlib.sha256(payload).hexdigest()
                  if payload is not None else _file_sha256(path))
        after = _authority_signature(path)
        if before != after:
            raise ValueError(
                f"declared input changed while it was captured: {path}")
        return InputBinding(path, "file", digest, payload, after)
    if path.is_dir():
        return InputBinding(path, "directory", None)
    raise ValueError(
        f"declared input is neither a regular file nor directory: {path}")


def _plan_path(base: Path, value: object, label: str) -> Path:
    text = _nonempty_string(value, label)
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def _same_or_nested(first: Path, second: Path) -> bool:
    """Whether two resolved paths alias or one contains the other."""

    if first.exists() and second.exists():
        try:
            if os.path.samefile(first, second):
                return True
        except OSError:
            pass
    first_key = os.path.normcase(str(first))
    second_key = os.path.normcase(str(second))
    try:
        common = os.path.commonpath((first_key, second_key))
    except ValueError:  # different Windows drives
        return False
    return common == first_key or common == second_key


def _validate_path_isolation(plan_path: Path, summary: Path,
                             input_authority_store: Path,
                             runs: Sequence[RunSpec]) -> None:
    mutable: list[tuple[str, Path]] = []
    inputs: list[tuple[str, Path]] = []
    for run in runs:
        mutable.extend((
            (f"run {run.name!r} outdir", run.outdir),
            (f"run {run.name!r} scratch", run.scratch_dir),
            (f"run {run.name!r} cache", run.cache),
        ))
        inputs.extend(
            (f"run {run.name!r} input {index}", path)
            for index, path in enumerate(run.inputs))
    for index, (left_label, left_path) in enumerate(mutable):
        for right_label, right_path in mutable[index + 1:]:
            if _same_or_nested(left_path, right_path):
                raise ValueError(
                    f"{left_label} {left_path} overlaps {right_label} "
                    f"{right_path}; every mutable outdir/scratch/cache root "
                    "must be distinct and non-overlapping")

    if summary == plan_path:
        raise ValueError("multi-run summary must not overwrite its plan")
    if summary.exists():
        raise ValueError(
            f"multi-run summary {summary} already exists; choose a new path "
            "so an earlier production summary is not overwritten")
    if input_authority_store.exists():
        raise ValueError(
            f"multi-run input authority store {input_authority_store} "
            "already exists; choose a new summary so earlier captured "
            "inputs are never mixed into this plan")
    if _same_or_nested(input_authority_store, plan_path):
        raise ValueError(
            f"multi-run input authority store {input_authority_store} "
            f"overlaps the plan {plan_path}")
    if _same_or_nested(input_authority_store, summary):
        raise ValueError(
            f"multi-run input authority store {input_authority_store} "
            f"overlaps the summary {summary}")
    for label, path in mutable:
        if _same_or_nested(plan_path, path):
            raise ValueError(
                f"{label} {path} overlaps the plan {plan_path}")
        if _same_or_nested(summary, path):
            raise ValueError(
                f"{label} {path} overlaps the summary {summary}")
        if _same_or_nested(input_authority_store, path):
            raise ValueError(
                f"multi-run input authority store {input_authority_store} "
                f"overlaps {label} {path}")

    for input_label, input_path in inputs:
        if not input_path.exists():
            raise ValueError(
                f"{input_label} does not exist: {input_path}")
        if _same_or_nested(summary, input_path):
            raise ValueError(
                f"summary {summary} overlaps read-only {input_label} "
                f"{input_path}")
        if _same_or_nested(input_authority_store, input_path):
            raise ValueError(
                f"multi-run input authority store {input_authority_store} "
                f"overlaps read-only {input_label} {input_path}")
        for mutable_label, mutable_path in mutable:
            if _same_or_nested(input_path, mutable_path):
                raise ValueError(
                    f"{mutable_label} {mutable_path} overlaps read-only "
                    f"{input_label} {input_path}")

    for run in runs:
        if run.config is not None and not run.config.is_file():
            raise ValueError(
                f"run {run.name!r} config is not a file: {run.config}")
        for kind, path in (("outdir", run.outdir),
                           ("scratch", run.scratch_dir),
                           ("cache", run.cache)):
            if path.exists():
                raise ValueError(
                    f"run {run.name!r} {kind} already exists: {path}; "
                    "multi-run requires new directories so it cannot mix "
                    "with earlier output or temporary state")


def _margin_gib(value: object, label: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or float(value) < 0):
        raise ValueError(
            f"{label} must be a finite, nonnegative number of GiB, not "
            f"{value!r}")
    return float(value)


def _plan_packing(value: object) -> PackingRequest:
    """The plan's optional ``[devices]`` table, validated; never coerced."""

    if value is None:
        return PackingRequest()
    if not isinstance(value, dict):
        raise ValueError("multi-run plan [devices] must be a TOML table")
    unknown = sorted(set(value) - _DEVICES_KEYS)
    if unknown:
        raise ValueError(f"multi-run plan [devices] has unknown keys: "
                         f"{unknown}; known: {sorted(_DEVICES_KEYS)}")
    sources = []
    runs_per_gpu: int | str = 1
    if "runs_per_gpu" in value:
        runs_per_gpu = parse_count_or_auto(
            value["runs_per_gpu"], flag="[devices] runs_per_gpu")
        sources.append(("runs_per_gpu", "plan [devices]"))
    margin = None
    if "mem_margin_gib" in value:
        margin = _margin_gib(value["mem_margin_gib"],
                             "[devices] mem_margin_gib")
        sources.append(("mem_margin_gib", "plan [devices]"))
    mps = False
    if "mps" in value:
        if not isinstance(value["mps"], bool):
            raise ValueError(
                f"[devices] mps must be true or false, not "
                f"{value['mps']!r}")
        mps = value["mps"]
        sources.append(("mps", "plan [devices]"))
    return PackingRequest(runs_per_gpu=runs_per_gpu, mem_margin_gib=margin,
                          mps=mps, sources=tuple(sources))


def packing_request(plan_request: PackingRequest,
                    args: argparse.Namespace | None = None
                    ) -> PackingRequest:
    """The plan's ``[devices]`` with the command-line flags laid over it.

    A flag wins over the plan and the receipt says which one did.  Memory
    and MPS settings without packing are named refusals, as are the MPS
    sub-settings without MPS: a setting that would silently do nothing is
    refused rather than dropped.
    """

    request = plan_request
    sources = dict(request.sources)
    flags = {
        "runs_per_gpu": getattr(args, "runs_per_gpu", None),
        "mem_margin_gib": getattr(args, "mem_margin_gib", None),
        "mps": getattr(args, "mps", None),
        "mps_dir": getattr(args, "mps_dir", None),
        "mps_active_thread_percentage": getattr(
            args, "mps_active_thread_percentage", None),
    }
    if flags["runs_per_gpu"] is not None:
        request = replace(request, runs_per_gpu=parse_count_or_auto(
            flags["runs_per_gpu"], flag="--runs-per-gpu"))
        sources["runs_per_gpu"] = "--runs-per-gpu"
    if flags["mem_margin_gib"] is not None:
        request = replace(request, mem_margin_gib=_margin_gib(
            flags["mem_margin_gib"], "--mem-margin-gib"))
        sources["mem_margin_gib"] = "--mem-margin-gib"
    if flags["mps"] is not None:
        request = replace(request, mps=bool(flags["mps"]))
        sources["mps"] = "--mps" if flags["mps"] else "--no-mps"
    if flags["mps_dir"] is not None:
        request = replace(request, mps_dir=Path(flags["mps_dir"]))
        sources["mps_dir"] = "--mps-dir"
    if flags["mps_active_thread_percentage"] is not None:
        request = replace(request, mps_active_thread_percentage=int(
            flags["mps_active_thread_percentage"]))
        sources["mps_active_thread_percentage"] = (
            "--mps-active-thread-percentage")
    request = replace(request, sources=tuple(sorted(sources.items())))
    if not request.packed:
        if request.mem_margin_gib is not None:
            raise ValueError(
                f"mem_margin_gib ({sources['mem_margin_gib']}) prices runs "
                "on a shared card; it applies only with runs_per_gpu above "
                "1 or auto")
        if request.mps:
            raise ValueError(
                f"mps ({sources['mps']}) lets several runs' kernels share a "
                "GPU; with one run per GPU there is nothing to share (set "
                "[devices] runs_per_gpu or --runs-per-gpu N|auto)")
    if request.mps_dir is not None and not request.mps:
        raise ValueError("--mps-dir applies only with --mps")
    percentage = request.mps_active_thread_percentage
    if percentage is not None:
        if not request.mps:
            raise ValueError(
                "--mps-active-thread-percentage applies only with --mps")
        if not 1 <= percentage <= 100:
            raise ValueError(
                f"--mps-active-thread-percentage must be 1-100, not "
                f"{percentage}")
    return request


def _plan_kernel_cache(value: object, base: Path) -> KernelCache:
    """The plan's optional ``[cache]`` table; ``kernels`` defaults shared."""

    if value is None:
        return KernelCache()
    if not isinstance(value, dict):
        raise ValueError("multi-run plan [cache] must be a TOML table")
    unknown = sorted(set(value) - _CACHE_KEYS)
    if unknown:
        raise ValueError(f"multi-run plan [cache] has unknown keys: "
                         f"{unknown}; known: {sorted(_CACHE_KEYS)}")
    if "kernels" not in value:
        return KernelCache()
    return parse_kernel_cache(
        value["kernels"], flag="[cache] kernels", base=base,
        source="plan [cache]")


def kernel_cache_request(plan: MultiRunPlan,
                         args: argparse.Namespace | None = None
                         ) -> KernelCache:
    """The plan's ``[cache] kernels`` with ``--kernel-cache`` laid over it.

    A shared directory must lie outside every path the plan owns (it is
    written by every child and kept after them); refused by name if not.
    """

    request = plan.kernel_cache
    flag = getattr(args, "kernel_cache", None)
    if flag is not None:
        request = parse_kernel_cache(
            flag, flag="--kernel-cache", base=Path.cwd(),
            source="--kernel-cache")
    if request.directory is not None:
        _validate_outside_plan_paths(
            plan, request.directory, "shared kernel cache",
            "give [cache] kernels / --kernel-cache a directory outside every "
            "plan, input, summary, output, scratch, and cache path")
    return request


def _is_digest(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


@functools.lru_cache(maxsize=None)
def _runner_option_strings(module: str) -> frozenset[str]:
    """Every option string the prepared runner's own parser accepts."""

    runner = importlib.import_module(module)
    return frozenset(runner.build_parser()._option_string_actions)


def _given_binding(module: str, arguments: Sequence[str],
                   label: str) -> dict[str, str]:
    """The binding digests a run's argv gives, in their one canonical form.

    ``--flag=VALUE``, an abbreviation argparse would expand into a binding
    flag, a repeated flag, a missing value or a value that is not a
    lowercase SHA-256 digest is refused by name: a digest the runner would
    read differently from this reading is never inserted beside.
    """

    flags = _RUNNER_BINDING_FLAGS[module]
    known = _runner_option_strings(module)
    for argument in arguments:
        if not argument.startswith("--") or argument == "--":
            continue
        option = argument.split("=", 1)[0]
        if option in flags and "=" in argument:
            raise ValueError(
                f"{label} gives {option} as {argument!r}; use the separate "
                f"form {option!r}, VALUE")
        if option not in known:
            expanded = [flag for flag in flags if flag.startswith(option)]
            if expanded:
                raise ValueError(
                    f"{label} abbreviates {expanded} as {option!r}; spell "
                    "the binding option out")
    values: dict[str, str] = {}
    for flag in flags:
        positions = [index for index, argument in enumerate(arguments)
                     if argument == flag]
        if len(positions) > 1:
            raise ValueError(
                f"{label} gives {flag} {len(positions)} times; at most once")
        if not positions:
            continue
        index = positions[0]
        if index + 1 >= len(arguments):
            raise ValueError(f"{label} {flag} is missing its value")
        value = arguments[index + 1]
        if not _is_digest(value):
            raise ValueError(
                f"{label} {flag} must be a lowercase SHA-256 digest (64 hex "
                f"characters), got {value!r}; omit it and multi-run derives "
                "it from the prepared root")
        values[flag] = value
    return values


def _binding_from_command(command: Sequence[str],
                          flags: Sequence[str]) -> dict[str, str]:
    """The binding digests a composed runner command carries, in order."""

    found = {}
    for index, token in enumerate(command[:-1]):
        if token in flags:
            found[token] = str(command[index + 1])
    return found


def _execution_input_check(
        bundle: Mapping[str, object], key: str, declared: Path,
        declared_sha256: str | None, label: str,
        prepared_root: Path) -> dict[str, object]:
    """The declared config/namelist held to the preparation's own record.

    A mapped preparation records the experiment config and WPS namelist it
    was prepared from in ``execution_inputs`` (path, bytes, sha256); its
    runner refuses any other bytes, after the card was admitted.  Refused
    here by name instead, pointing at the file that does match when one
    is still on disk.  Routes whose proof records no such entry bind the
    file through their own source manifest; that is recorded, not checked.
    """

    payload = bundle.get("payload")
    execution = (payload.get("execution_inputs")
                 if isinstance(payload, Mapping) else None)
    recorded = (execution.get(key) if isinstance(execution, Mapping)
                else None)
    recorded_sha256 = (recorded.get("sha256")
                       if isinstance(recorded, Mapping) else None)
    row: dict[str, object] = {
        "declared": str(declared), "declared_sha256": declared_sha256,
        "recorded_path": (recorded.get("path")
                          if isinstance(recorded, Mapping) else None),
        "recorded_sha256": recorded_sha256,
    }
    if not _is_digest(recorded_sha256):
        row.update(status="NOT_RECORDED", basis=(
            f"the preparation document records no execution_inputs.{key}; "
            "the runner binds this file through its own source manifest"))
        return row
    if declared_sha256 == recorded_sha256:
        row["status"] = "MATCH"
        return row
    candidates = []
    recorded_path = row["recorded_path"]
    if isinstance(recorded_path, str) and recorded_path:
        candidates.append(Path(recorded_path))
    own = {"experiment_config": ("experiment.toml",),
           "wps_namelist": ("namelist.wps",)}.get(key, ())
    candidates.extend(prepared_root / name for name in own)
    matching = []
    for candidate in candidates:
        try:
            if candidate.is_file() and _file_sha256(candidate) \
                    == recorded_sha256:
                matching.append(str(candidate))
        except OSError:
            continue
    where = (f" {matching[0]} still holds those bytes: declare it as this "
             "run's input instead" if matching else
             f" No file holding those bytes was found (the record names "
             f"{recorded_path!r}); prepare again from the file this run "
             "declares, or restore the one it was prepared from")
    raise ValueError(
        f"{label}: declared {key.replace('_', ' ')} {declared} (sha256 "
        f"{declared_sha256}) is not the file the preparation at "
        f"{prepared_root} was made from: its document records sha256 "
        f"{recorded_sha256} in execution_inputs.{key}, and the runner "
        f"refuses any other bytes.{where}.")


def _bind_prepared_runner(spec: RunSpec) -> RunSpec:
    """Bind a module run to its preparation the way ``woof sim`` does.

    Reads the declared ``--prepared-root`` (:func:`woof.stage_cli.
    resolve_bundle`, or the live head of a chained preparation still being
    prepared, :func:`woof.stage_cli.unsealed_head_bundle`) and composes the
    runner command with :func:`woof.stage_cli.sim_command`; its binding
    digests (:data:`PREPARED_BINDING_BASIS`) are the derived ones.

    What is pinned:

    * no digest given: every derived digest is inserted after the
      ``--prepared-root`` pair, and the single-domain runner's absent
      ``--source`` is the bundle's own;
    * digests given: each is kept and must equal the derived one, or the
      plan is refused naming flag, given, expected and where it came
      from.  A given head binding on a sealed chained preparation must be
      the head its proof records (``boundary_stream.head_sha256``), and is
      read back through that head's own digest check
      (:func:`woof.stage_cli.resolve_head_bundle`);
    * the declared experiment config and WPS namelist must be the bytes
      the preparation records in ``execution_inputs``
      (:func:`_execution_input_check`);
    * a root that cannot be bound is refused here, before any directory is
      claimed or device touched, with the reason ``woof sim`` gives.

    Digests are values, not paths: the ``{inputN}`` rules are unchanged.
    """

    assert spec.module is not None
    from woof import stage_cli

    module = spec.module
    label = f"run {spec.name!r}"
    flags = _RUNNER_BINDING_FLAGS[module]
    given = _given_binding(module, spec.arguments, f"{label} args")
    rendered = _render_arguments(spec)
    paths = _runner_path_values(module, rendered, f"{label} args")
    root = Path(paths["--prepared-root"])
    config = Path(paths["--experiment-config"])
    wps_value = paths.get("--wps-namelist")
    wps = None if wps_value is None else Path(wps_value)
    layout = "tree" if module == _TREE_RUNNER else "single"
    try:
        bundle, live = stage_cli.resolve_sim_bundle(root)
        if layout == "tree" and bundle["layout"] != "tree":
            raise ValueError(
                f"{label}: {bundle['document']} is a single-domain "
                f"preparation ({bundle['schema']}); its runner is "
                "woof.prepared_single_domain_forecast")

        def compose(chosen: Mapping[str, object]) -> dict[str, str]:
            return _binding_from_command(stage_cli.sim_command(
                chosen, experiment_config=config, wps_namelist=wps,
                outdir=spec.outdir, runner=layout), flags)

        derived = compose(bundle)
        head_given = _HEAD_BINDING_FLAG in given
        sealed_given = [flag for flag in _SEALED_BINDING_FLAGS[module]
                        if flag in given]
        if head_given and sealed_given:
            raise ValueError(
                f"{label} gives {_HEAD_BINDING_FLAG} and {sealed_given}; a "
                "prepared forecast binds a head or a sealed proof, not both")
        expected = derived
        expected_from = ("derived from the "
                         + ("live chained head" if live else "sealed")
                         + f" preparation document {bundle['document']}")
        if head_given and not live:
            stream = bundle["payload"].get("boundary_stream")
            recorded = (stream.get("head_sha256")
                        if isinstance(stream, Mapping) else None)
            if not _is_digest(recorded):
                raise ValueError(
                    f"{label} gives {_HEAD_BINDING_FLAG}, but "
                    f"{bundle['document']} is sealed and records no chained "
                    "head (boundary_stream.head_sha256), so there is no head "
                    "to bind; omit the binding digests and multi-run binds "
                    "its proof")
            if given[_HEAD_BINDING_FLAG] != recorded:
                raise ValueError(
                    f"{label} {_HEAD_BINDING_FLAG} {given[_HEAD_BINDING_FLAG]}"
                    f" is not the head {bundle['document']} records "
                    f"(boundary_stream.head_sha256 {recorded}); omit the "
                    "binding digests and multi-run binds the preparation")
            expected = compose(stage_cli.resolve_head_bundle(root, recorded))
            expected_from = (
                f"read from the head {recorded} that {bundle['document']} "
                "records (boundary-stream/head.json, checked against its "
                "own digest)")
        elif sealed_given and live:
            raise ValueError(
                f"{label} gives {sealed_given}, but {root} is still being "
                "prepared and has no sealed document to bind yet; omit the "
                "binding digests and multi-run binds its head")
    except (stage_cli.StageRefusal, OSError) as refusal:
        raise ValueError(
            f"{label}: the prepared root {root} cannot be bound: "
            f"{refusal}") from None

    mode = ("prepared_head" if _HEAD_BINDING_FLAG in expected
            else "sealed_receipt" if layout == "tree" else "sealed_proof")
    digests = []
    inserted: list[str] = []
    for flag in flags:
        want = expected.get(flag)
        have = given.get(flag)
        if have is None:
            if want is not None:
                inserted.extend((flag, want))
                digests.append({"flag": flag, "sha256": want,
                                "source": "derived", "basis": expected_from})
            continue
        if want is None:
            if mode == "prepared_head" and flag == "--source-manifest-sha256":
                # An as-posted head names no manifest: it binds its input
                # plan, which the runner holds the given value to.
                digests.append({
                    "flag": flag, "sha256": have, "source": "given",
                    "basis": ("not verified here: an as-posted head names no "
                              "source manifest until its seal; the runner "
                              "holds this value to the head's input plan")})
                continue
            raise ValueError(
                f"{label} gives {flag}, which is not part of the {mode} "
                f"binding of {root} ({expected_from}); omit it")
        if have != want:
            raise ValueError(
                f"{label} {flag} {have} does not match the preparation: "
                f"expected {want}, {expected_from}. Omit the binding digests "
                "and multi-run derives them, or point --prepared-root at the "
                "preparation these digests name")
        digests.append({"flag": flag, "sha256": have,
                        "source": "given, verified", "basis": expected_from})

    arguments = list(spec.arguments)
    source_record = None
    if layout == "single":
        bundle_source = str(bundle["source"])
        given_source = _runner_option_value(arguments, "--source")
        if given_source is None:
            arguments[0:0] = ["--source", bundle_source]
            source_record = {"value": bundle_source, "source": "derived",
                             "basis": f"the source {bundle['document']} "
                                      "names (woof.stage_cli.resolve_bundle)"}
        elif bundle_source != "mapped" and given_source != bundle_source:
            raise ValueError(
                f"{label} --source {given_source!r}, but "
                f"{bundle['document']} was prepared from {bundle_source!r}; "
                "omit --source and multi-run takes the bundle's own")
        else:
            source_record = {"value": given_source, "source": (
                "given, verified" if given_source == bundle_source
                else "given; the bundle names no packaged profile, the "
                     "runner verifies it")}
    if inserted:
        anchor = arguments.index("--prepared-root") + 2
        arguments[anchor:anchor] = inserted

    by_path = {binding.path: binding for binding in spec.input_bindings}
    execution_inputs: dict[str, object] = {}
    for key, path in (("experiment_config", config), ("wps_namelist", wps)):
        if path is None:
            continue
        if layout == "tree":
            # The tree runner holds the recorded config digest as
            # preparation identity only: a forecast may extend its stop
            # time, and its typed per-domain checks compare what changes
            # the prepared state.  Byte equality would refuse runs it takes.
            execution_inputs[key] = {
                "declared": str(path), "status": "NOT_CHECKED",
                "basis": ("the tree runner binds this config by "
                          "--experiment-config-sha256 and checks each "
                          "per-domain setting itself; it need not be the "
                          "bytes the preparation recorded")}
            continue
        binding = by_path.get(path)
        execution_inputs[key] = _execution_input_check(
            bundle, key, path, None if binding is None else binding.sha256,
            label, root)
    record = {
        "basis": PREPARED_BINDING_BASIS,
        "bundle_source": str(bundle["source"]),
        "digests": digests,
        "document": str(bundle["document"]),
        "execution_inputs": execution_inputs,
        "layout": bundle["layout"],
        "mode": mode,
        "prepared_root": str(root),
        "schema": bundle["schema"],
        "source_option": source_record,
        "state": "being_prepared" if live else "sealed",
    }
    return replace(spec, arguments=tuple(arguments),
                   plan_arguments=spec.arguments, prepared_binding=record)


def load_plan(path: str | Path, *, summary_override: str | Path | None = None,
              preflight_override: str | None = None) -> MultiRunPlan:
    """Parse and path-check a versioned TOML plan without touching CUDA.

    Module-form runs are bound to their preparation here
    (:func:`_bind_prepared_runner`), so a plan needs no hand-copied
    digests and a wrong one is refused before anything is claimed."""

    plan_path = Path(path).expanduser().resolve()
    if not plan_path.is_file():
        raise ValueError(f"multi-run plan is not a file: {plan_path}")
    try:
        plan_signature_before = _authority_signature(plan_path)
        plan_bytes = plan_path.read_bytes()
        plan_signature_after = _authority_signature(plan_path)
        if plan_signature_before != plan_signature_after:
            raise ValueError(
                f"multi-run plan changed while it was captured: {plan_path}")
        raw = tomllib.loads(plan_bytes.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"multi-run plan {plan_path} is invalid TOML: "
                         f"{error}") from error
    plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    if not isinstance(raw, dict):
        raise ValueError("multi-run plan must be a TOML table")
    unknown = sorted(set(raw) - _TOP_LEVEL_KEYS)
    if unknown:
        raise ValueError(f"multi-run plan has unknown top-level keys: "
                         f"{unknown}")
    if raw.get("schema") != PLAN_SCHEMA:
        raise ValueError(
            f"multi-run plan schema must be {PLAN_SCHEMA!r}, got "
            f"{raw.get('schema')!r}")

    base = plan_path.parent
    summary_value: object
    if summary_override is not None:
        summary_value = str(summary_override)
    else:
        summary_value = raw.get(
            "summary", f"{plan_path.stem}.summary.json")
    summary = _plan_path(base, summary_value, "summary")
    input_authority_store = summary.with_name(
        f"{summary.name}.input-authorities").resolve()

    preflight_value = (raw.get("preflight", "estimate")
                       if preflight_override is None
                       else preflight_override)
    preflight = _nonempty_string(preflight_value, "preflight")
    if preflight not in PREFLIGHT_MODES:
        raise ValueError(
            f"preflight must be one of {PREFLIGHT_MODES}, got "
            f"{preflight!r}")

    packing = _plan_packing(raw.get("devices"))
    kernel_cache = _plan_kernel_cache(raw.get("cache"), base)

    entries = raw.get("run")
    if not isinstance(entries, list) or not entries:
        raise ValueError("multi-run plan must contain at least one [[run]]")
    runs: list[RunSpec] = []
    names: set[str] = set()
    for index, entry in enumerate(entries, start=1):
        label = f"run {index}"
        if not isinstance(entry, dict):
            raise ValueError(f"{label} must be a TOML table")
        unknown = sorted(set(entry) - _RUN_KEYS)
        if unknown:
            raise ValueError(f"{label} has unknown keys: {unknown}")
        missing = sorted(_COMMON_RUN_KEYS - set(entry))
        if missing:
            raise ValueError(f"{label} is missing required keys: {missing}")
        has_config = "config" in entry
        has_module = "module" in entry
        if has_config == has_module:
            raise ValueError(
                f"{label} must choose exactly one runner form: config for "
                "woof run, or module plus args and inputs")
        if has_config:
            forbidden = sorted(set(entry) & {"module", "args", "inputs"})
            if forbidden:
                raise ValueError(
                    f"{label} config form cannot also set {forbidden}")
        else:
            missing_module = sorted({"args", "inputs"} - set(entry))
            if missing_module:
                raise ValueError(
                    f"{label} module form is missing required keys: "
                    f"{missing_module}")
        name = _nonempty_string(entry["name"], f"{label} name")
        if name in names:
            raise ValueError(f"multi-run name {name!r} is duplicated")
        names.add(name)
        config = (_plan_path(base, entry["config"], f"run {name!r} config")
                  if has_config else None)
        module = (_module_name(entry["module"], f"run {name!r} module")
                  if has_module else None)
        arguments = (_argument_list(entry["args"], f"run {name!r} args")
                     if has_module else ())
        inputs = (_input_paths(base, entry["inputs"],
                               f"run {name!r} inputs")
                  if has_module else (config,))
        if has_module:
            assert module is not None
            _validate_outdir_template(arguments, f"run {name!r} args")
            _validate_runner_input_templates(
                module, arguments, len(inputs), f"run {name!r} args")
            _validate_placeholders(
                arguments, len(inputs), f"run {name!r} args")
        runs.append(RunSpec(
            name=name,
            device_selector=_device_selector(
                entry["device"], f"run {name!r} device"),
            outdir=_plan_path(base, entry["outdir"],
                              f"run {name!r} outdir"),
            scratch_dir=_plan_path(base, entry["scratch"],
                                   f"run {name!r} scratch"),
            cache=_plan_path(base, entry["cache"],
                             f"run {name!r} cache"),
            config=config,
            module=module,
            arguments=arguments,
            inputs=inputs,
        ))
    _validate_path_isolation(
        plan_path, summary, input_authority_store, runs)
    binding_cache: dict[Path, InputBinding] = {}
    config_paths = {run.config for run in runs if run.config is not None}
    bound_runs = []
    for run in runs:
        bindings = []
        for path in run.inputs:
            if path not in binding_cache:
                binding_cache[path] = _bind_input(
                    path, capture_bytes=path in config_paths)
            bindings.append(binding_cache[path])
        bound = replace(run, input_bindings=tuple(bindings))
        if bound.module is not None:
            bound = _bind_prepared_runner(bound)
        bound_runs.append(bound)
    return MultiRunPlan(
        plan_path, plan_sha256, plan_signature_after, summary,
        input_authority_store, preflight, tuple(bound_runs), packing,
        kernel_cache)


def resolve_devices(
        plan: MultiRunPlan, inventory: Sequence[GPUIdentity], *,
        lock_root: Path | None = None) -> tuple[PlannedRun, ...]:
    """Resolve every selector; reject aliases to one device unless packed.

    With the plan's packing at its default (``runs_per_gpu`` 1) two
    selectors that resolve to one physical UUID are refused, exactly as
    before packing existed.  A packed plan lets them share the card; its
    admission (:func:`plan_packing`) decides when each one starts.
    """

    by_index = {str(gpu.index): gpu for gpu in inventory
                if gpu.index is not None}
    by_uuid = {gpu.uuid: gpu for gpu in inventory}
    if len(by_uuid) != len(inventory):
        raise ValueError("nvidia-smi reported duplicate GPU UUIDs")
    indexed = [gpu for gpu in inventory if gpu.index is not None]
    if len(by_index) != len(indexed):
        raise ValueError("nvidia-smi reported duplicate GPU indices")
    resolved: list[PlannedRun] = []
    used: dict[str, str] = {}
    for spec in plan.runs:
        gpu = (by_index.get(spec.device_selector)
               if spec.device_selector.isdigit()
               else by_uuid.get(spec.device_selector))
        if gpu is None:
            available = [
                {"index": item.index, "uuid": item.uuid, "name": item.name}
                for item in inventory]
            raise ValueError(
                f"run {spec.name!r} requested device "
                f"{spec.device_selector!r}, not present in {available}")
        previous = used.get(gpu.uuid)
        if previous is not None and not plan.packing.packed:
            raise ValueError(
                f"runs {previous!r} and {spec.name!r} both resolve to GPU "
                f"{gpu.uuid}; every run needs a unique physical device "
                "unless the plan packs runs onto one card ([devices] "
                "runs_per_gpu or --runs-per-gpu N|auto)")
        used.setdefault(gpu.uuid, spec.name)
        resolved.append(PlannedRun(
            spec, gpu, lock_root, plan.input_authority_store,
            kernel_cache=plan.kernel_cache))
    return tuple(resolved)


def _parent_lock_root(environment: Mapping[str, str]) -> Path:
    """The supervisor lock root before child temp variables are isolated."""

    configured = environment.get(GPU_LOCK_ROOT_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    if os.name == "nt":
        base = (environment.get("PROGRAMDATA")
                or environment.get("TEMP")
                or environment.get("TMP")
                or environment.get("TMPDIR")
                or str(Path.cwd()))
    else:
        base = (environment.get("TMPDIR")
                or environment.get("TEMP")
                or environment.get("TMP")
                or "/tmp")
    return (Path(base).expanduser().resolve() / "woof" / "locks")


def _validate_outside_plan_paths(plan: MultiRunPlan, path: Path,
                                 label: str, remedy: str) -> Path:
    """Prove a host-wide directory is outside every path the plan owns."""

    root = path.expanduser().resolve()
    protected: list[tuple[str, Path]] = [
        ("plan", plan.path), ("summary", plan.summary)]
    for run in plan.runs:
        protected.extend((
            (f"run {run.name!r} outdir", run.outdir),
            (f"run {run.name!r} scratch", run.scratch_dir),
            (f"run {run.name!r} cache", run.cache),
        ))
        protected.extend(
            (f"run {run.name!r} input {index}", item)
            for index, item in enumerate(run.inputs))
    protected.append(("shared input authority store",
                      plan.input_authority_store))
    for what, item in protected:
        if _same_or_nested(root, item):
            raise ValueError(f"{label} {root} overlaps {what} {item}; "
                             f"{remedy}")
    return root


def _validate_lock_root_isolation(
        plan: MultiRunPlan, lock_root: Path) -> Path:
    """Prove the shared UUID-lock namespace is outside every run path."""

    root = _validate_outside_plan_paths(
        plan, lock_root, "shared GPU lock root",
        f"set {GPU_LOCK_ROOT_ENV} to a stable machine-wide directory "
        "outside every plan, input, summary, output, scratch, and cache path")
    root.mkdir(parents=True, exist_ok=True)
    return root


def child_environment(run: PlannedRun,
                      parent: Mapping[str, str] | None = None) -> dict[str, str]:
    """Copy the parent environment and isolate device/temp state.

    TMP/TEMP/TMPDIR are always the run's own scratch.  The GPU kernel
    caches follow ``run.kernel_cache``: shared (inherited, or a plan-level
    directory) by default, or the run's own ``<cache>/cupy`` and
    ``<cache>/cuda`` with ``per-run``."""

    from woof.config_authority import (CONFIG_PAYLOAD_ENV,
                                        CONFIG_SHA256_ENV,
                                        CONFIG_SOURCE_ENV)

    environment = dict(os.environ if parent is None else parent)
    for name in (CONFIG_PAYLOAD_ENV, CONFIG_SOURCE_ENV, CONFIG_SHA256_ENV):
        environment.pop(name, None)
    lock_root = (run.lock_root if run.lock_root is not None
                 else _parent_lock_root(environment))
    scratch = str(run.spec.scratch_dir)
    run.kernel_cache.apply(environment, run.spec.cache)
    environment.update({
        "CUDA_VISIBLE_DEVICES": run.gpu.uuid,
        GPU_LOCK_ROOT_ENV: str(lock_root),
        "TEMP": scratch,
        "TMP": scratch,
        "TMPDIR": scratch,
    })
    if run.input_authority_store is not None:
        environment[SHARED_INPUT_AUTHORITY_ROOT_ENV] = str(
            run.input_authority_store)
    environment.update(dict(run.environment_overrides))
    if run.spec.config is not None:
        from woof.config_authority import authority_environment

        binding = run.spec.input_bindings[0]
        assert binding.sha256 is not None
        environment.update(authority_environment(
            source=run.spec.config,
            payload_path=run.spec.captured_config,
            sha256=binding.sha256))
    return environment


def _check_command(run: PlannedRun, mode: str) -> tuple[str, ...]:
    if run.spec.config is None:
        raise ValueError(
            f"run {run.spec.name!r} uses a production module, which owns "
            "its own preflight")
    return (
        sys.executable, "-m", "woof.multi_run", "check-worker",
        "--gpu-uuid", run.gpu.uuid, "--config", str(run.spec.config),
        "--mode", mode,
    )


def _render_arguments(spec: RunSpec) -> tuple[str, ...]:
    replacements = {
        "{outdir}": str(spec.outdir),
        "{scratch}": str(spec.scratch_dir),
        "{cache}": str(spec.cache),
    }
    replacements.update({
        f"{{input{index}}}": str(path)
        for index, path in enumerate(spec.inputs)
    })
    rendered = []
    for argument in spec.arguments:
        value = argument
        for token, replacement in replacements.items():
            value = value.replace(token, replacement)
        if "{input" in value:
            raise ValueError(
                f"run {spec.name!r} argument {argument!r} refers to an "
                "input placeholder that was not declared")
        rendered.append(value)
    return tuple(rendered)


def _run_command(run: PlannedRun) -> tuple[str, ...]:
    if run.spec.config is not None:
        return (
            sys.executable, "-m", "woof.cli", "run", str(run.spec.config),
            "--outdir", str(run.spec.outdir), "--gpu-uuid", run.gpu.uuid,
        )
    assert run.spec.module is not None
    return (
        sys.executable, "-m", "woof.multi_run", "worker",
        "--gpu-uuid", run.gpu.uuid,
        "--module", run.spec.module,
        "--outdir", str(run.spec.outdir),
        *(item for path in run.spec.inputs
          for item in ("--input", str(path))),
        "--", *_render_arguments(run.spec),
    )


def _worker_reservation_bytes(module: str,
                              arguments: Sequence[str]) -> int | None:
    """This worker's priced peak envelope, or None when it cannot be priced.

    Read off the prepared runner's own ``--experiment-config`` value, which
    the argument contract above has already pinned to one declared input,
    and priced through :func:`woof.supervisor.priced_reservation_bytes` --
    the same function ``woof run`` and ``woof stream`` price a shared
    card from.  ONE function, every door: a multi-run worker that priced
    its own reservation differently would be admitted onto a card the
    supervisor it wraps refuses, or refused off one that supervisor admits.

    Unpriceable is None, never a number and never a refusal: an unpriced
    run is admitted against the device's measured free memory and says so.

    Priced with the source the runner forecasts from, so the analysed
    hydrometeor tables that source puts on the root's boundary are in the
    reservation (:func:`woof.stage_cli.boundary_pricing_source`: the
    single-domain runner's own ``--source``, else the one its prepared
    root names).  Without it a multi-run of HRRR-forced forecasts reserved
    a shared card short by those tables.
    Verified prepared land cover supplies the urban column counts; an
    unreadable prepared bundle retains the configuration's workspace bound.
    """

    values = _runner_path_values(module, arguments, "worker arguments")
    config = values.get("--experiment-config")
    if config is None:
        return None
    from woof.stage_cli import boundary_pricing_source

    root = values.get("--prepared-root")
    source = _runner_option_value(arguments, "--source")
    pricing = {"source": (source if root is None else
                          boundary_pricing_source(Path(root), source))}
    if root is not None:
        from woof.core.preflight import _prepared_check_inputs

        wps = values.get("--wps-namelist")
        try:
            prepared = _prepared_check_inputs(argparse.Namespace(
                config=Path(config), prepared_root=Path(root),
                wps_namelist=None if wps is None else Path(wps)))
        except (OSError, ValueError, RuntimeError, ImportError):
            # Reservation pricing is advisory. The runner verifies the
            # bundle before restore; unverified land cover keeps the bound.
            prepared = None
        if prepared is not None and prepared.urban_columns is not None:
            pricing["urban_columns"] = prepared.urban_columns
    return priced_reservation_bytes(Path(config), **pricing)


def _runner_option_value(arguments: Sequence[str], flag: str) -> str | None:
    """The value a runner's argument list gives ``flag``, or ``None``."""

    for index, argument in enumerate(arguments):
        if argument == flag and index + 1 < len(arguments):
            return str(arguments[index + 1])
        if argument.startswith(f"{flag}="):
            return argument.split("=", 1)[1]
    return None


def _locked_module_main(*, gpu_uuid: str, module_name: str, outdir: Path,
                        inputs: Sequence[Path],
                        arguments: Sequence[str]) -> int:
    """Run one production module while holding the existing UUID lock."""

    module_name = _module_name(module_name, "worker module")
    _validate_rendered_outdir(arguments, outdir, "worker arguments")
    _validate_rendered_runner_inputs(
        module_name, arguments, inputs, "worker arguments")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != gpu_uuid:
        raise RuntimeError(
            "multi-run worker device mask does not match its physical UUID: "
            f"CUDA_VISIBLE_DEVICES={visible!r}, UUID={gpu_uuid!r}")
    reservation_bytes = _worker_reservation_bytes(module_name, arguments)
    if outdir.exists():
        raise ValueError(
            f"multi-run worker outdir already exists: {outdir}; refusing "
            "to mix production outputs")
    with GPUFileLock(gpu_uuid, run_id=f"multi-run-{os.getpid()}"):
        # Another invocation can finish while this worker waits for the UUID
        # lock.  Recheck inside the lock before importing any CUDA-facing
        # target; prepared runners also retain their own atomic mkdir claim.
        if outdir.exists():
            raise ValueError(
                f"multi-run worker outdir already exists: {outdir}; "
                "refusing to mix production outputs")
        # Priced against THIS worker's own configuration, so a co-tenant
        # that fits beside it is admitted here exactly as it is at
        # ``woof run``, and one that does not is refused at both.
        preflight_exclusive_gpu(
            gpu_uuid, approved_pids={os.getpid()},
            reservation_bytes=reservation_bytes)
        module = importlib.import_module(module_name)
        target = getattr(module, "main", None)
        if not callable(target):
            raise ValueError(
                f"production module {module_name!r} has no callable main")
        result = target(list(arguments))
    return 0 if result is None else int(result)


def _locked_check_main(*, gpu_uuid: str, config: Path, mode: str) -> int:
    """Run the full config check inside its selected UUID lock and mask."""

    if mode not in PREFLIGHT_MODES[:2]:
        raise ValueError(
            f"check-worker mode must be estimate or alloc, got {mode!r}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != gpu_uuid:
        raise RuntimeError(
            "multi-run check-worker device mask does not match its physical "
            f"UUID: CUDA_VISIBLE_DEVICES={visible!r}, UUID={gpu_uuid!r}")
    reservation_bytes = priced_reservation_bytes(config)
    with GPUFileLock(gpu_uuid, run_id=f"multi-run-check-{os.getpid()}"):
        # Plan review prices the card from the configuration it is
        # reviewing, because a review that cleared a shared card the run
        # door then refused would be a plan review that reviewed a
        # different question than the run.
        preflight_exclusive_gpu(
            gpu_uuid, approved_pids={os.getpid()},
            reservation_bytes=reservation_bytes)
        cli = importlib.import_module("woof.cli")
        command = ["check", str(config)]
        if mode == "alloc":
            command.append("--alloc")
        result = cli.main(command)
    return 0 if result is None else int(result)


def _module_entry(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m woof.multi_run")
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--gpu-uuid", required=True)
    worker.add_argument("--module", required=True)
    worker.add_argument("--outdir", type=Path, required=True)
    worker.add_argument("--input", type=Path, action="append", required=True)
    worker.add_argument("arguments", nargs=argparse.REMAINDER)
    check_worker = subparsers.add_parser("check-worker")
    check_worker.add_argument("--gpu-uuid", required=True)
    check_worker.add_argument("--config", type=Path, required=True)
    check_worker.add_argument(
        "--mode", choices=PREFLIGHT_MODES[:2], required=True)
    args = parser.parse_args(argv)
    if args.command == "check-worker":
        return _locked_check_main(
            gpu_uuid=args.gpu_uuid, config=args.config.resolve(),
            mode=args.mode)
    arguments = list(args.arguments)
    if arguments[:1] == ["--"]:
        arguments.pop(0)
    return _locked_module_main(
        gpu_uuid=args.gpu_uuid, module_name=args.module,
        outdir=args.outdir.resolve(),
        inputs=tuple(path.resolve() for path in args.input),
        arguments=arguments)


def _new_process_group_kwargs() -> dict[str, object]:
    """Keep a terminal interrupt in the parent from signalling children."""

    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _last_log_line(path: Path, limit: int = 8192) -> str | None:
    """The last non-empty line of a child's log, or None (advisory)."""

    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - limit))
            tail = stream.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    return lines[-1][:400] if lines else None


def _print_failure_line(phase: str, record: ProcessRecord) -> None:
    """Say why a child failed where the operator is looking.

    A refusal inside a child (a wrong digest, a missing input) used to
    show only as ``exit=2`` here, its reason buried in the run log."""

    if record.exit_code in (None, 0):
        return
    line = _last_log_line(record.log)
    if line is not None:
        print(f"multi-run {phase} {record.name}: last log line: {line}",
              file=sys.stderr, flush=True)


def _execute_group(
        runs: Sequence[PlannedRun], *, phase: str,
        command_for: Callable[[PlannedRun], tuple[str, ...]],
        log_for: Callable[[PlannedRun], Path],
        popen: Callable[..., subprocess.Popen] | None = None,
        poll_seconds: float = 0.10) -> GroupOutcome:
    """Launch every child before waiting, then report completion live."""

    if poll_seconds <= 0.0:
        raise ValueError("poll_seconds must be positive")
    group_started_at_utc = utc_now()
    group_started_monotonic = time.monotonic()
    launch = subprocess.Popen if popen is None else popen
    records: list[ProcessRecord] = []
    active: list[tuple[ProcessRecord, subprocess.Popen, object]] = []
    launch_index = -1
    pending_record: ProcessRecord | None = None
    pending_process: subprocess.Popen | None = None
    pending_stream = None
    try:
        for run in runs:
            records.append(ProcessRecord(
                run.spec.name, phase, command_for(run), log_for(run)))
        for launch_index, (run, record) in enumerate(zip(runs, records)):
            pending_record = record
            record.started_at_utc = utc_now()
            record._started_monotonic = time.monotonic()
            stream = None
            try:
                stream = record.log.open("xb")
                pending_stream = stream
                process = launch(
                    list(record.command), env=child_environment(run),
                    stdin=subprocess.DEVNULL, stdout=stream,
                    stderr=subprocess.STDOUT, close_fds=True,
                    **_new_process_group_kwargs())
                pending_process = process
            except OSError as error:
                if stream is not None:
                    stream.close()
                pending_record = None
                pending_stream = None
                record.status = "launch_failed"
                record.completed_at_utc = utc_now()
                record.duration_seconds = round(
                    time.monotonic() - record._started_monotonic, 6)
                record.error = f"{type(error).__name__}: {error}"
                print(
                    f"multi-run {phase} {record.name}: launch failed; "
                    f"see {record.log}", file=sys.stderr, flush=True)
                continue
            record.pid = process.pid
            record.status = "running"
            active.append((record, process, stream))
            pending_record = None
            pending_process = None
            pending_stream = None
            print(
                f"multi-run {phase} {record.name}: started "
                f"{record.started_at_utc} pid={record.pid} "
                f"device={run.gpu.uuid} log={record.log}", flush=True)

        while active:
            remaining: list[tuple[ProcessRecord, subprocess.Popen, object]] = []
            for record, process, stream in active:
                exit_code = process.poll()
                if exit_code is None:
                    remaining.append((record, process, stream))
                    continue
                record.exit_code = int(exit_code)
                record.status = "succeeded" if exit_code == 0 else "failed"
                record.completed_at_utc = utc_now()
                record.duration_seconds = round(
                    time.monotonic() - record._started_monotonic, 6)
                stream.close()
                print(
                    f"multi-run {phase} {record.name}: completed "
                    f"{record.completed_at_utc} exit={record.exit_code} "
                    f"duration={record.duration_seconds:.3f}s "
                    f"log={record.log}", flush=True)
                _print_failure_line(phase, record)
            active = remaining
            if active:
                time.sleep(poll_seconds)
        group_completed_at_utc = utc_now()
        return GroupOutcome(
            tuple(records), False, group_started_at_utc,
            group_completed_at_utc,
            round(time.monotonic() - group_started_monotonic, 6))
    except KeyboardInterrupt:
        observed_at = utc_now()
        now = time.monotonic()
        if (pending_record is not None
                and any(record is pending_record
                        for record, _process, _stream in active)):
            pending_record = None
            pending_process = None
            pending_stream = None
        if pending_stream is not None:
            pending_stream.close()
        if pending_record is not None:
            pending_record.status = (
                "launch_unobserved" if pending_process is None
                else "running_unobserved")
            pending_record.pid = (None if pending_process is None
                                  else pending_process.pid)
            pending_record.duration_seconds = round(
                now - (pending_record._started_monotonic or now), 6)
        for record, _process, stream in active:
            stream.close()
            record.status = "running_unobserved"
            record.duration_seconds = round(
                now - (record._started_monotonic or now), 6)
        not_started_from = launch_index + 1
        for record in records[not_started_from:]:
            record.status = "not_started"
        pids = [record.pid for record, _process, _stream in active]
        if pending_record is not None:
            pids.append(pending_record.pid)
        print(
            f"multi-run {phase}: interrupted at {observed_at}; child "
            f"processes were not terminated, unobserved pids={pids}",
            file=sys.stderr, flush=True)
        return GroupOutcome(
            tuple(records), True, group_started_at_utc, observed_at,
            round(now - group_started_monotonic, 6))
    except BaseException as error:
        # Preserve everything known when a non-interrupt failure escapes
        # orchestration (including a second Popen raising after the first
        # child was launched).  Child ownership is deliberately untouched:
        # no terminate/kill is safe when a child may itself supervise a
        # forecast worker.
        observed_at = utc_now()
        now = time.monotonic()
        if (pending_record is not None
                and any(record is pending_record
                        for record, _process, _stream in active)):
            pending_record = None
            pending_process = None
            pending_stream = None
        if pending_stream is not None:
            try:
                pending_stream.close()
            except BaseException:
                pass
        if pending_record is not None:
            pending_record.status = (
                "launch_unobserved" if pending_process is None
                else "running_unobserved")
            if pending_process is not None:
                try:
                    pending_record.pid = pending_process.pid
                except BaseException:
                    pending_record.pid = None
            pending_record.duration_seconds = round(
                now - (pending_record._started_monotonic or now), 6)
            pending_record.error = f"{type(error).__name__}: {error}"
        for record, _process, stream in active:
            try:
                stream.close()
            except BaseException:
                pass
            if record.status not in {"succeeded", "failed"}:
                record.status = "running_unobserved"
                record.duration_seconds = round(
                    now - (record._started_monotonic or now), 6)
        for record in records[launch_index + 1:]:
            record.status = "not_started"
        outcome = GroupOutcome(
            tuple(records), False, group_started_at_utc, observed_at,
            round(now - group_started_monotonic, 6))
        pids = [record.pid for record in records
                if record.status == "running_unobserved"
                and record.pid is not None]
        print(
            f"multi-run {phase}: orchestration failed at {observed_at}; "
            f"child processes were not terminated, unobserved pids={pids}",
            file=sys.stderr, flush=True)
        raise GroupExecutionError(phase, outcome, error) from error


# ---------------------------------------------------------------------------
# Packing several runs onto one GPU (``runs_per_gpu`` above 1 or ``auto``)
# ---------------------------------------------------------------------------


def _gib_text(value: int | float | None) -> str:
    """:func:`woof.supervisor._gib`, the shared-GPU receipt's formatter,
    with ``unknown`` for an unmeasured figure."""

    from woof.supervisor import _gib

    return "unknown" if value is None else _gib(value)


@dataclass(frozen=True)
class RunPricing:
    """One run's priced forecast peak and the basis it was priced on."""

    need_bytes: int | None
    basis: str

    def as_dict(self) -> dict[str, object]:
        return {
            "basis": self.basis,
            "priced_peak_bytes": self.need_bytes,
            "priced_peak_gib": (
                None if self.need_bytes is None
                else round(self.need_bytes / 1024 ** 3, 3)),
        }


def price_run(spec: RunSpec) -> RunPricing:
    """Price one run's forecast peak the way its own run door will.

    Config form: :func:`woof.supervisor.priced_reservation_bytes` of the
    experiment loaded from the bytes captured at plan load (never the
    source read a second time) -- the number ``woof run``'s supervisor
    prices the card from at launch and ``woof check`` reviews.  Module
    form: the prepared runner's own :func:`_worker_reservation_bytes`, the
    number its locked worker re-prices.  CPU only: no device is asked.
    Unpriceable is ``None`` with the reason in the basis, never a number;
    such a run is packed alone on its card.
    """

    try:
        if spec.config is not None:
            from woof.config_authority import ConfigAuthority
            from woof.experiment import load_experiment

            binding = spec.input_bindings[0]
            if binding.captured_bytes is None or binding.sha256 is None:
                return RunPricing(
                    None, "unpriced: the config bytes were not captured")
            experiment = load_experiment(spec.config, authority=ConfigAuthority(
                binding.captured_bytes, spec.config, binding.sha256,
                spec.captured_config))
            need = priced_reservation_bytes(experiment)
            basis = CONFIG_PRICING_BASIS
        else:
            assert spec.module is not None
            need = _worker_reservation_bytes(
                spec.module, _render_arguments(spec))
            basis = MODULE_PRICING_BASIS
    except Exception as error:  # noqa: BLE001 - pricing is advisory
        return RunPricing(None, f"unpriced: {type(error).__name__}: {error}")
    if need is None:
        return RunPricing(
            None, f"unpriced: no peak envelope could be priced ({basis})")
    return RunPricing(int(need), basis)


def _process_trees(roots: Sequence[int]) -> dict[int, set[int]]:
    """Each root PID with its live descendants (Linux ``/proc``).

    Elsewhere, or when ``/proc`` cannot be read, a root is only itself, so
    less memory is attributed to a sibling and admission is more
    conservative, never less."""

    children: dict[int, list[int]] = {}
    proc = Path("/proc")
    try:
        entries = [entry for entry in os.listdir(proc) if entry.isdigit()]
    except OSError:
        entries = []
    for entry in entries:
        try:
            text = (proc / entry / "stat").read_text(
                encoding="utf-8", errors="replace")
        except OSError:
            continue
        # ``pid (comm) state ppid ...``; comm may itself hold ')'.
        fields = text.rsplit(")", 1)[-1].split()
        if len(fields) >= 2 and fields[1].isdigit():
            children.setdefault(int(fields[1]), []).append(int(entry))
    trees: dict[int, set[int]] = {}
    for root in roots:
        found = {int(root)}
        stack = [int(root)]
        while stack:
            for child in children.get(stack.pop(), ()):
                if child not in found:
                    found.add(child)
                    stack.append(child)
        trees[int(root)] = found
    return trees


class CardMeter:
    """How a packed multi-run measures its cards (NVML through nvidia-smi).

    ``measure`` is :func:`woof.supervisor.measured_device_memory`, the
    numbers every child's own supervisor admits itself against.
    ``process_bytes`` is the per-process framebuffer figure the run doors'
    co-tenant check reads (:func:`woof.supervisor.query_compute_processes`:
    compute-apps rows with pmon's figure where compute-apps has none),
    which :meth:`PackingPlan.decide` attributes to a sibling through its
    process tree so memory a sibling already holds is not counted twice.
    When NVML cannot say (a PID namespace, an unreported row), nothing is
    attributed and admission only gets more conservative.  Tests
    substitute a fake.
    """

    def card_total_bytes(self, gpu_uuid: str) -> int | None:
        from woof.core.preflight import device_physical_total_bytes

        return device_physical_total_bytes(device_id=gpu_uuid)

    def measure(self, gpu_uuid: str) -> Mapping[str, object]:
        return measured_device_memory(gpu_uuid)

    def process_bytes(self, gpu_uuid: str) -> dict[int, int]:
        from woof.supervisor import query_compute_processes

        try:
            rows = query_compute_processes(gpu_uuid)
        except Exception:  # noqa: BLE001 - attribution is advisory
            return {}
        return {row.pid: int(row.used_gpu_memory_mib) * 1024 ** 2
                for row in rows if row.used_gpu_memory_mib}

    def process_trees(self, roots: Sequence[int]) -> dict[int, set[int]]:
        return _process_trees(roots)


def _card_meter() -> CardMeter:
    return CardMeter()


def packed_admission(*, need_bytes: int | None, margin_bytes: int,
                     card_total_bytes: int, cap: int,
                     live: Sequence[Mapping[str, object]],
                     observe: Callable[[], Mapping[str, int]],
                     measure: Callable[[], Mapping[str, object]],
                     settling: bool = False) -> dict[str, object]:
    """Decide one run's admission onto one card (:data:`PACKED_ADMISSION_RULE`).

    ``live`` rows are this multi-run's runs holding the card: ``name`` and
    ``priced_bytes`` (None when unpriced).  Only when the measured rule is
    reached is ``observe`` called (sibling name -> bytes NVML attributes to
    its process tree, 0 when it cannot say) and THEN ``measure``
    (:func:`woof.supervisor.measured_device_memory`'s mapping): memory a
    sibling allocates between the two samples is in the used figure and
    still counted as its outstanding growth -- counted twice, never
    missed.  ``settling`` is true while a sibling on this card finished
    less than one recheck ago: its memory may not be released yet, so a
    run that does not fit then waits instead of being refused.

    Returns the decision: ``verdict`` (``admitted``,
    ``admitted-unpriced-alone``, ``waiting`` or ``refused``), the ``rule``
    that decided it, a ``reason`` sentence and every number it used.
    """

    rows = [dict(row) for row in live]
    names = [str(row["name"]) for row in rows]
    decision: dict[str, object] = {
        "card_total_bytes": int(card_total_bytes),
        "headroom_bytes": None,
        "margin_bytes": int(margin_bytes),
        "measured_free_bytes": None,
        "measured_used_bytes": None,
        "measurement_basis": None,
        "need_bytes": need_bytes,
        "runs_per_gpu_cap": int(cap),
        "siblings": rows,
        "siblings_outstanding_bytes": None,
        "siblings_priced_bytes": None,
    }

    def verdict(value: str, rule: str, reason: str) -> dict[str, object]:
        decision.update(verdict=value, rule=rule, reason=reason)
        return decision

    def settle_or_refuse(rule: str, reason: str) -> dict[str, object]:
        if settling:
            return verdict(
                "waiting", "sibling_memory_settling",
                reason + "; a sibling on this card has just finished and "
                "its memory may not be released yet, so the card is "
                "measured again before the run is refused")
        return verdict("refused", rule, reason)

    if len(rows) >= cap:
        return verdict(
            "waiting", "runs_per_gpu_cap",
            f"{len(rows)} of this multi-run's runs ({names}) hold the card, "
            f"the runs_per_gpu cap of {cap}")
    if need_bytes is None:
        if rows:
            return verdict(
                "waiting", "unpriced_runs_alone",
                f"this run could not be priced, so it runs alone on its "
                f"card; it waits for {names} to finish")
        return verdict(
            "admitted-unpriced-alone", "unpriced_runs_alone",
            "this run could not be priced, so it runs alone on its card; "
            "its own supervisor prices the card's measured free memory at "
            "launch")
    unpriced = [str(row["name"]) for row in rows
                if row.get("priced_bytes") is None]
    if unpriced:
        return verdict(
            "waiting", "unpriced_sibling_alone",
            f"unpriced sibling(s) {unpriced} hold the card alone")
    need = int(need_bytes)
    margin = int(margin_bytes)
    priced_sum = sum(int(row["priced_bytes"]) for row in rows)
    decision["siblings_priced_bytes"] = priced_sum
    if priced_sum + need + margin > card_total_bytes:
        return verdict(
            "waiting", "card_total",
            f"siblings' priced peaks {_gib_text(priced_sum)} + this run's "
            f"{_gib_text(need)} + margin {_gib_text(margin)} exceed the "
            f"card's {_gib_text(card_total_bytes)}; waits for {names} to "
            "finish")
    observed = dict(observe()) if rows else {}
    for row in rows:
        row["observed_bytes"] = int(observed.get(str(row["name"]), 0))
    try:
        measured = measure()
    except Exception as error:  # noqa: BLE001 - named in the decision
        detail = f"{type(error).__name__}: {error}"
        if rows:
            return verdict(
                "waiting", "measurement_failed",
                f"the card's memory could not be measured ({detail}); "
                "re-measured when a sibling finishes")
        return settle_or_refuse(
            "measurement_failed",
            "the card's memory could not be measured, so the run is not "
            f"admitted (an unmeasured card never reads as headroom): "
            f"{detail}")
    used = int(measured["device_wide_used_bytes"])
    free = measured.get("device_free_bytes")
    free = (max(0, int(card_total_bytes) - used) if free is None
            else int(free))
    outstanding = sum(
        max(0, int(row["priced_bytes"]) - int(row["observed_bytes"]))
        for row in rows)
    headroom = free - outstanding
    decision.update(
        headroom_bytes=headroom, measured_free_bytes=free,
        measured_used_bytes=used,
        measurement_basis=measured.get("basis", SHARED_ADMISSION_BASIS),
        siblings_outstanding_bytes=outstanding)
    sentence = (
        f"priced peak {_gib_text(need)} + margin {_gib_text(margin)} "
        f"against {_gib_text(headroom)} = measured free {_gib_text(free)} "
        f"less siblings' outstanding growth {_gib_text(outstanding)}")
    if need + margin <= headroom:
        return verdict("admitted", "fits", sentence + ": fits")
    if rows:
        return verdict(
            "waiting", "measured_free",
            sentence + f": does not fit; waits for {names} to finish")
    return settle_or_refuse(
        "measured_free",
        sentence + f": does not fit, and none of this multi-run's runs "
        f"hold the card ({_gib_text(used)} is used device-wide by "
        "processes this multi-run does not own), so waiting cannot free "
        "it")


@dataclass
class PackingPlan:
    """A packed plan's cards, pricing, admission ledger, locks and MPS."""

    request: PackingRequest
    cards: dict[str, dict[str, object]]
    pricing: dict[str, RunPricing]
    meter: CardMeter
    lock_parent: Path
    events: list[dict[str, object]] = field(default_factory=list)
    decisions: dict[str, dict[str, dict[str, object]]] = field(
        default_factory=dict)
    mps: MpsDaemon | None = None
    shared_locks: dict[str, dict[str, object]] = field(default_factory=dict)
    _held_locks: list[GPUFileLock] = field(default_factory=list, repr=False)
    _logged: dict[tuple[str, str], tuple[object, object]] = field(
        default_factory=dict, repr=False)

    def hold_shared_locks(self) -> None:
        """Hold each packed card's machine-wide UUID lock for the run.

        The children lock under their own roots, so without this a packed
        card carries no lock the host's other woof processes can see: a
        ``woof run``, an energy worker or an exclusive multi-run would
        price the card while the siblings are still growing toward their
        peaks.  Holding the shared lock keeps them out exactly as one
        exclusive run on the card does.  A lock another process already
        holds (a forecast started earlier on this card) is recorded, not
        refused: that process is priced by measurement like any other
        co-tenant, as its own run door priced the card it found.
        """

        from woof.supervisor import GPUAlreadyLockedError

        for gpu_uuid in self.cards:
            lock = GPUFileLock(
                gpu_uuid, run_id=f"multi-run-packed-{os.getpid()}")
            try:
                lock.acquire()
            except GPUAlreadyLockedError as error:
                self.shared_locks[gpu_uuid] = {
                    "held_by_this_multi_run": False, "path": str(lock.path),
                    "detail": (
                        f"{error}; another woof process holds this card's "
                        "shared lock, so it is priced as a measured "
                        "co-tenant")}
                print("multi-run packing: GPU "
                      f"{gpu_uuid}: "
                      f"{self.shared_locks[gpu_uuid]['detail']}", flush=True)
                continue
            self._held_locks.append(lock)
            self.shared_locks[gpu_uuid] = {
                "held_by_this_multi_run": True, "path": str(lock.path),
                "detail": ("held until every child is observed finished, so "
                           "no other woof process starts on the card beside "
                           "packed runs it cannot see growing")}

    def release_shared_locks(self) -> None:
        while self._held_locks:
            self._held_locks.pop().release()

    def decide(self, phase: str, run: PlannedRun,
               live: Sequence[tuple[PlannedRun, int | None]], *,
               settling: bool = False, priced: bool = True
               ) -> dict[str, object]:
        """Admission of ``run`` beside its live siblings ``(run, pid)``.

        ``priced`` false is an estimate-mode config check, which holds no
        forecast memory: only the ``runs_per_gpu`` cap applies to it."""

        gpu_uuid = run.gpu.uuid
        if not priced:
            names = [sibling.spec.name for sibling, _pid in live]
            decision: dict[str, object] = {
                "runs_per_gpu_cap": self.request.cap, "siblings": names}
            if len(live) >= self.request.cap:
                decision.update(
                    verdict="waiting", rule="runs_per_gpu_cap",
                    reason=(f"{len(live)} of this multi-run's checks "
                            f"({names}) hold the card, the runs_per_gpu cap "
                            f"of {self.request.cap}"))
            else:
                decision.update(
                    verdict="admitted", rule="estimate_check_cap_only",
                    reason=("an estimate check holds no forecast memory, so "
                            "only the runs_per_gpu cap applies"))
            return self._record(phase, run, decision)
        rows = [{"name": sibling.spec.name, "pid": pid,
                 "priced_bytes": self.pricing[sibling.spec.name].need_bytes}
                for sibling, pid in live]

        def observe() -> dict[str, int]:
            process_bytes = self.meter.process_bytes(gpu_uuid)
            pids = [pid for _sibling, pid in live if pid is not None]
            trees = self.meter.process_trees(pids) if pids else {}
            return {
                sibling.spec.name: sum(
                    process_bytes.get(member, 0)
                    for member in (() if pid is None
                                   else trees.get(pid, {pid})))
                for sibling, pid in live}

        decision = packed_admission(
            need_bytes=self.pricing[run.spec.name].need_bytes,
            margin_bytes=self.request.margin_bytes,
            card_total_bytes=int(self.cards[gpu_uuid]["total_bytes"]),
            cap=self.request.cap, live=rows, observe=observe,
            measure=lambda: self.meter.measure(gpu_uuid), settling=settling)
        return self._record(phase, run, decision)

    def _record(self, phase: str, run: PlannedRun,
                decision: dict[str, object]) -> dict[str, object]:
        gpu_uuid = run.gpu.uuid
        decision.update({
            "at_utc": utc_now(), "gpu_index": run.gpu.index,
            "gpu_uuid": gpu_uuid, "phase": phase, "run": run.spec.name,
        })
        self.decisions.setdefault(run.spec.name, {})[phase] = decision
        key = (phase, run.spec.name)
        signature = (decision["verdict"], decision["rule"])
        if self._logged.get(key) != signature:
            # A waiting run is re-measured until it starts; the ledger
            # records each change of verdict or rule, not every poll.
            self._logged[key] = signature
            self.events.append(decision)
            print(
                f"multi-run {phase} {run.spec.name}: {decision['verdict']} "
                f"on GPU {gpu_uuid}: {decision['reason']}", flush=True)
        return decision

    def receipt(self) -> dict[str, object]:
        return {
            "admission_rule": PACKED_ADMISSION_RULE,
            "cards": [
                {
                    "index": card["gpu"].index,
                    "name": card["gpu"].name,
                    "runs": [
                        {"name": name, **self.pricing[name].as_dict()}
                        for name in card["runs"]],
                    "shared_gpu_lock": self.shared_locks.get(gpu_uuid),
                    "total_basis": card["total_basis"],
                    "total_bytes": card["total_bytes"],
                    "uuid": gpu_uuid,
                }
                for gpu_uuid, card in self.cards.items()],
            "events": list(self.events),
            "measurement_basis": SHARED_ADMISSION_BASIS,
            "mem_margin_bytes": self.request.margin_bytes,
            "mem_margin_gib": self.request.margin_gib,
            "mode": "packed",
            "mps": None if self.mps is None else self.mps.summary(),
            "per_run_gpu_lock_roots": str(self.lock_parent),
            "runs_per_gpu": self.request.runs_per_gpu,
            "runs_per_gpu_cap": self.request.cap,
            "settings_source": dict(self.request.sources),
        }


def _exclusive_packing_receipt(request: PackingRequest) -> dict[str, object]:
    return {
        "basis": ("one run per physical GPU; a second run resolving to the "
                  "same card is refused at device resolution"),
        "mode": "exclusive",
        "runs_per_gpu": 1,
        "settings_source": dict(request.sources),
    }


def plan_packing(plan: MultiRunPlan, runs: Sequence[PlannedRun], *,
                 lock_root: Path, meter: CardMeter
                 ) -> tuple[PackingPlan, tuple[PlannedRun, ...]]:
    """Price every run, budget every card, refuse what can never fit.

    Each card's budget is its NVML ``memory.total``; a card NVML cannot
    size is a named refusal (packing would have no budget).  A run whose
    priced peak plus the margin exceeds its card's total can never be
    admitted there, so the whole plan is refused before anything is
    claimed or launched, naming every such run.  Each run gets its own
    ``WOOF_GPU_LOCK_ROOT`` under ``lock_root`` (already proven outside
    every plan path), so siblings' ``woof run`` UUID locks do not refuse
    each other.  Prints the per-card assignment and pricing.
    """

    request = plan.packing
    pricing = {run.spec.name: price_run(run.spec) for run in runs}
    cards: dict[str, dict[str, object]] = {}
    for run in runs:
        card = cards.get(run.gpu.uuid)
        if card is None:
            total = meter.card_total_bytes(run.gpu.uuid)
            if total is None or int(total) <= 0:
                raise ValueError(
                    f"GPU {run.gpu.uuid} (index {run.gpu.index}): NVML did "
                    "not report its memory.total, so packing has no budget "
                    "for this card; restore nvidia-smi or run one run per "
                    "GPU (runs_per_gpu = 1)")
            card = cards[run.gpu.uuid] = {
                "gpu": run.gpu, "runs": [], "total_basis": "NVML memory.total",
                "total_bytes": int(total)}
        card["runs"].append(run.spec.name)
    margin = request.margin_bytes
    problems = []
    for gpu_uuid, card in cards.items():
        total = int(card["total_bytes"])
        for name in card["runs"]:
            need = pricing[name].need_bytes
            if need is not None and need + margin > total:
                problems.append(
                    f"run {name!r} is priced at {_gib_text(need)} "
                    f"({pricing[name].basis}); with the "
                    f"{_gib_text(margin)} margin it needs "
                    f"{_gib_text(need + margin)}, more than GPU "
                    f"{gpu_uuid}'s {_gib_text(total)} total, so it can "
                    "never be admitted there")
    if problems:
        raise ValueError(
            "multi-run cannot pack this plan: " + "; ".join(problems)
            + ". Put the run on a larger card, make it smaller, or lower "
            "the margin (mem_margin_gib)")
    lock_parent = (lock_root.expanduser().resolve()
                   / f"multi-run-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    packed = tuple(
        replace(run, lock_root=lock_parent / f"run-{index:03d}")
        for index, run in enumerate(runs))
    for gpu_uuid, card in cards.items():
        listed = ", ".join(
            f"{name} {_gib_text(pricing[name].need_bytes)}"
            if pricing[name].need_bytes is not None
            else f"{name} unpriced (runs alone)"
            for name in card["runs"])
        print(
            f"multi-run packing: GPU {card['gpu'].index} {gpu_uuid} "
            f"({_gib_text(card['total_bytes'])} NVML total, runs_per_gpu "
            f"{request.runs_per_gpu}, cap {request.cap}, margin "
            f"{_gib_text(margin)}): {listed}", flush=True)
    return PackingPlan(request, cards, pricing, meter, lock_parent), packed


def _execute_packed_group(
        runs: Sequence[PlannedRun], *, phase: str,
        command_for: Callable[[PlannedRun], tuple[str, ...]],
        log_for: Callable[[PlannedRun], Path],
        packing: PackingPlan,
        popen: Callable[..., subprocess.Popen] | None = None,
        poll_seconds: float = 0.10,
        recheck_seconds: float | None = None,
        priced: bool = True) -> GroupOutcome:
    """Launch each run when its card admits it; report completion live.

    :func:`_execute_group` with a per-card queue in plan order: the run at
    the head of a card's queue is offered to :meth:`PackingPlan.decide`
    whenever a sibling on that card starts or finishes, and otherwise
    every ``recheck_seconds`` (default :data:`ADMISSION_RECHECK_SECONDS`)
    so memory freed by processes this multi-run does not own is noticed.
    For one recheck after a sibling on a card finishes, a run there that
    does not fit waits rather than being refused (its memory may not be
    released yet).  ``priced`` false (estimate-mode checks) applies only
    the ``runs_per_gpu`` cap.
    A refused run is recorded ``not_admitted`` with the decision's reason
    and never launched.  Interrupt and failure semantics are
    :func:`_execute_group`'s: children are never terminated; queued runs
    are recorded ``not_started``.
    """

    if poll_seconds <= 0.0:
        raise ValueError("poll_seconds must be positive")
    recheck = (ADMISSION_RECHECK_SECONDS if recheck_seconds is None
               else float(recheck_seconds))
    group_started_at_utc = utc_now()
    group_started_monotonic = time.monotonic()
    launch = subprocess.Popen if popen is None else popen
    records: list[ProcessRecord] = []
    queues: dict[str, list[int]] = {}
    live: dict[str, list[tuple[int, ProcessRecord, subprocess.Popen,
                               object]]] = {}
    dirty: dict[str, bool] = {}
    evaluated_at: dict[str, float] = {}
    finished_at: dict[str, float] = {}
    pending_record: ProcessRecord | None = None
    pending_process: subprocess.Popen | None = None
    pending_stream = None

    def unobserved(now: float) -> None:
        nonlocal pending_record, pending_process, pending_stream
        if pending_record is not None and any(
                record is pending_record for entries in live.values()
                for _index, record, _process, _stream in entries):
            # Already in the live ledger: the live loop below records it.
            pending_record = pending_process = pending_stream = None
        if pending_stream is not None:
            try:
                pending_stream.close()
            except BaseException:
                pass
        if pending_record is not None:
            pending_record.status = (
                "launch_unobserved" if pending_process is None
                else "running_unobserved")
            try:
                pending_record.pid = (
                    None if pending_process is None else pending_process.pid)
            except BaseException:
                pending_record.pid = None
            pending_record.duration_seconds = round(
                now - (pending_record._started_monotonic or now), 6)
        for entries in live.values():
            for _index, record, _process, stream in entries:
                try:
                    stream.close()
                except BaseException:
                    pass
                if record.status not in {"succeeded", "failed"}:
                    record.status = "running_unobserved"
                    record.duration_seconds = round(
                        now - (record._started_monotonic or now), 6)
        for queue in queues.values():
            for index in queue:
                records[index].status = "not_started"

    try:
        for index, run in enumerate(runs):
            records.append(ProcessRecord(
                run.spec.name, phase, command_for(run), log_for(run)))
            queues.setdefault(run.gpu.uuid, []).append(index)
            live.setdefault(run.gpu.uuid, [])
            dirty[run.gpu.uuid] = True
            evaluated_at[run.gpu.uuid] = float("-inf")
        while (any(queues.values())
               or any(entries for entries in live.values())):
            progressed = False
            for gpu_uuid, queue in queues.items():
                while queue:
                    now = time.monotonic()
                    if (not dirty[gpu_uuid]
                            and now - evaluated_at[gpu_uuid] < recheck):
                        break
                    index = queue[0]
                    run, record = runs[index], records[index]
                    decision = packing.decide(phase, run, [
                        (runs[item], record_.pid)
                        for item, record_, _process, _stream
                        in live[gpu_uuid]],
                        settling=(now - finished_at.get(
                            gpu_uuid, float("-inf")) < recheck),
                        priced=priced)
                    evaluated_at[gpu_uuid] = now
                    dirty[gpu_uuid] = False
                    if decision["verdict"] == "waiting":
                        break
                    queue.pop(0)
                    progressed = True
                    dirty[gpu_uuid] = True
                    if decision["verdict"] == "refused":
                        record.status = "not_admitted"
                        record.error = str(decision["reason"])
                        print(
                            f"multi-run {phase} {record.name}: not admitted "
                            f"on GPU {gpu_uuid}; nothing was launched",
                            file=sys.stderr, flush=True)
                        continue
                    pending_record = record
                    record.started_at_utc = utc_now()
                    record._started_monotonic = time.monotonic()
                    stream = None
                    try:
                        stream = record.log.open("xb")
                        pending_stream = stream
                        process = launch(
                            list(record.command), env=child_environment(run),
                            stdin=subprocess.DEVNULL, stdout=stream,
                            stderr=subprocess.STDOUT, close_fds=True,
                            **_new_process_group_kwargs())
                        pending_process = process
                    except OSError as error:
                        if stream is not None:
                            stream.close()
                        pending_record = None
                        pending_stream = None
                        record.status = "launch_failed"
                        record.completed_at_utc = utc_now()
                        record.duration_seconds = round(
                            time.monotonic() - record._started_monotonic, 6)
                        record.error = f"{type(error).__name__}: {error}"
                        print(
                            f"multi-run {phase} {record.name}: launch failed; "
                            f"see {record.log}", file=sys.stderr, flush=True)
                        continue
                    record.pid = process.pid
                    record.status = "running"
                    live[gpu_uuid].append((index, record, process, stream))
                    pending_record = None
                    pending_process = None
                    pending_stream = None
                    print(
                        f"multi-run {phase} {record.name}: started "
                        f"{record.started_at_utc} pid={record.pid} "
                        f"device={gpu_uuid} log={record.log}", flush=True)
            for gpu_uuid, entries in live.items():
                remaining = []
                for entry in entries:
                    _index, record, process, stream = entry
                    exit_code = process.poll()
                    if exit_code is None:
                        remaining.append(entry)
                        continue
                    record.exit_code = int(exit_code)
                    record.status = "succeeded" if exit_code == 0 else "failed"
                    record.completed_at_utc = utc_now()
                    record.duration_seconds = round(
                        time.monotonic() - record._started_monotonic, 6)
                    stream.close()
                    dirty[gpu_uuid] = True
                    finished_at[gpu_uuid] = time.monotonic()
                    progressed = True
                    print(
                        f"multi-run {phase} {record.name}: completed "
                        f"{record.completed_at_utc} exit={record.exit_code} "
                        f"duration={record.duration_seconds:.3f}s "
                        f"log={record.log}", flush=True)
                    _print_failure_line(phase, record)
                live[gpu_uuid] = remaining
            if not progressed and (
                    any(queues.values())
                    or any(entries for entries in live.values())):
                time.sleep(poll_seconds)
        return GroupOutcome(
            tuple(records), False, group_started_at_utc, utc_now(),
            round(time.monotonic() - group_started_monotonic, 6))
    except KeyboardInterrupt:
        observed_at = utc_now()
        now = time.monotonic()
        unobserved(now)
        pids = [record.pid for record in records
                if record.status == "running_unobserved"]
        print(
            f"multi-run {phase}: interrupted at {observed_at}; child "
            f"processes were not terminated, unobserved pids={pids}",
            file=sys.stderr, flush=True)
        return GroupOutcome(
            tuple(records), True, group_started_at_utc, observed_at,
            round(now - group_started_monotonic, 6))
    except BaseException as error:
        observed_at = utc_now()
        now = time.monotonic()
        unobserved(now)
        if pending_record is not None:
            pending_record.error = f"{type(error).__name__}: {error}"
        outcome = GroupOutcome(
            tuple(records), False, group_started_at_utc, observed_at,
            round(now - group_started_monotonic, 6))
        pids = [record.pid for record in records
                if record.status == "running_unobserved"
                and record.pid is not None]
        print(
            f"multi-run {phase}: orchestration failed at {observed_at}; "
            f"child processes were not terminated, unobserved pids={pids}",
            file=sys.stderr, flush=True)
        raise GroupExecutionError(phase, outcome, error) from error


def _release_packing(packing: PackingPlan,
                     outcomes: Sequence[GroupOutcome | None]) -> None:
    """Release the cards' shared locks, stop an MPS daemon this run
    started and remove its per-run lock roots.

    The shared locks go with this process whatever happens (an exiting
    process releases them anyway).

    The daemon and the lock roots are left in place while any child's
    completion was not observed
    (an interrupt, an orchestration failure): such a child may still be
    an MPS client, and a lock file a live process holds must stay where
    that process can find it.  Saying so is the receipt's job; the PIDs
    are already recorded there."""

    packing.release_shared_locks()
    if any(outcome is not None and any(
            record.status in {"running", "running_unobserved",
                              "launch_unobserved"}
            for record in outcome.records) for outcome in outcomes):
        if packing.mps is not None and packing.mps.owned:
            print(
                f"multi-run: the MPS control daemon at "
                f"{packing.mps.pipe_dir} is left running for children whose "
                "completion was not observed; stop it with `echo quit | "
                f"CUDA_MPS_PIPE_DIRECTORY={packing.mps.pipe_dir} "
                "nvidia-cuda-mps-control` once they finish",
                file=sys.stderr, flush=True)
        return
    if packing.mps is not None:
        packing.mps.stop()
    shutil.rmtree(packing.lock_parent, ignore_errors=True)


def _prepare_directories(
        plan: MultiRunPlan,
        state: OrchestrationState | None = None) -> None:
    """Claim config-run outputs and prepare every isolated runtime root."""

    try:
        plan.input_authority_store.mkdir(parents=True, exist_ok=False)
        if state is not None:
            state.claimed_directories.append(plan.input_authority_store)
    except OSError as error:
        raise ValueError(
            "could not create shared input authority store "
            f"{plan.input_authority_store}: {error}; no child was launched") \
            from error

    try:
        # A shared kernel-cache directory persists across plans: created
        # if absent, never claimed (nothing here may remove it).
        plan.kernel_cache.prepare()
    except OSError as error:
        raise ValueError(
            f"could not create the shared kernel cache "
            f"{plan.kernel_cache.directory}: {error}; no child was "
            "launched") from error
    for run in plan.runs:
        directories = [
            ("scratch", run.scratch_dir),
            ("cache", run.cache),
        ]
        if not plan.kernel_cache.shared:
            directories.extend((
                ("CuPy cache", run.cache / "cupy"),
                ("CUDA cache", run.cache / "cuda"),
            ))
        # ``woof run`` accepts an existing output directory, so creating it
        # is an atomic claim. Prepared production runners intentionally claim
        # their own absent --outdir; leave it absent for that contract.
        if run.is_config_run:
            directories.insert(0, ("outdir", run.outdir))
        for kind, path in directories:
            try:
                path.mkdir(parents=True, exist_ok=False)
                if state is not None:
                    state.claimed_directories.append(path)
            except OSError as error:
                raise ValueError(
                    f"could not create run {run.name!r} {kind} {path}: "
                    f"{error}; no child was launched") from error
        if run.is_config_run:
            binding = run.input_bindings[0]
            if binding.captured_bytes is None or binding.sha256 is None:
                raise RuntimeError(
                    f"run {run.name!r} has no captured config authority")
            try:
                with run.captured_config.open("xb") as stream:
                    stream.write(binding.captured_bytes)
                    stream.flush()
                    os.fsync(stream.fileno())
            except OSError as error:
                raise ValueError(
                    f"could not publish run {run.name!r} captured config "
                    f"{run.captured_config}: {error}; no child was "
                    "launched") from error


def _prepare_summary_destination(
        path: Path, state: OrchestrationState | None = None) -> None:
    """Prove create-only atomic publication works before any child launch."""

    parent_existed = path.parent.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not parent_existed and state is not None:
        state.claimed_directories.append(path.parent)
    if path.exists():
        raise ValueError(
            f"multi-run summary {path} already exists; refusing to "
            "overwrite it")
    token = f"{os.getpid()}.{uuid.uuid4().hex}"
    source = path.with_name(f".{path.name}.{token}.link-source")
    destination = path.with_name(f".{path.name}.{token}.link-destination")
    source_created = False
    destination_created = False
    try:
        with source.open("x", encoding="utf-8", newline="\n") as stream:
            source_created = True
            stream.write("woof multi-run create-only publication probe\n")
            stream.flush()
            os.fsync(stream.fileno())
        publish_new(source, destination)
        destination_created = True
        if state is not None:
            state.summary_capable = True
    except OSError as error:
        raise ValueError(
            f"multi-run summary destination {path.parent} does not support "
            f"atomic create-only publication: {error}; no child was "
            "launched") from error
    finally:
        if destination_created:
            destination.unlink()
        if source_created:
            # publish_new renames the source where the volume has no hard links.
            source.unlink(missing_ok=True)
    if path.exists():
        raise ValueError(
            f"multi-run summary appeared during destination preparation: "
            f"{path}; refusing to overwrite it")


def _write_summary(path: Path, payload: Mapping[str, object]) -> Path:
    """Durably publish JSON create-only (``publish_new``)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        encoded = json.dumps(
            payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # The temporary is a fully written sibling on the same volume.
            # Publishing it is atomic and create-only (a hard link, or a
            # no-replace rename where the volume has no links): unlike
            # os.replace, a summary created by a racing invocation is never
            # overwritten.
            publish_new(temporary, path)
        except FileExistsError as error:
            raise ValueError(
                f"multi-run summary appeared before publication: {path}; "
                "refusing to overwrite it") from error
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


class FileAuthorityMonitor:
    """Sticky periodic mutation detector spanning every launched child."""

    def __init__(self, plan: MultiRunPlan, *, poll_seconds: float):
        if poll_seconds <= 0.0:
            raise ValueError("authority monitor poll_seconds must be positive")
        self.poll_seconds = poll_seconds
        authorities: list[
            tuple[str, Path, str, tuple[int, int, int, int, int] | None]
        ] = [("plan", plan.path, plan.sha256, plan.captured_signature)]
        seen = {os.path.normcase(str(plan.path))}
        for run in plan.runs:
            for binding in run.input_bindings:
                if binding.sha256 is None:
                    continue
                key = os.path.normcase(str(binding.path))
                if key in seen:
                    continue
                seen.add(key)
                authorities.append((
                    "input", binding.path, binding.sha256,
                    binding.captured_signature))
            if run.is_config_run and run.captured_config.is_file():
                binding = run.input_bindings[0]
                assert binding.sha256 is not None
                authorities.append((
                    "captured_config", run.captured_config, binding.sha256,
                    self._signature(run.captured_config)))
        self._records = [{
            "authority": authority,
            "baseline_signature": (
                None if signature is None else list(signature)),
            "expected_sha256": expected,
            "last_signature": signature,
            "observations": [],
            "path": str(path),
        } for authority, path, expected, signature in authorities]
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._failed = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at_utc: str | None = None
        self._stopped_at_utc: str | None = None
        self._poll_count = 0
        self._monitor_errors: list[dict[str, str]] = []

    @staticmethod
    def _signature(path: Path) -> tuple[int, int, int, int, int]:
        metadata = path.stat()
        if not path.is_file():
            raise OSError("path is no longer a regular file")
        return (
            int(metadata.st_dev), int(metadata.st_ino), int(metadata.st_size),
            int(metadata.st_mtime_ns), int(metadata.st_ctime_ns))

    def _record_failure(self, error: BaseException) -> None:
        with self._lock:
            self._monitor_errors.append({
                "error": f"{type(error).__name__}: {error}",
                "observed_at_utc": utc_now(),
            })
        self._failed.set()

    def _observe(self, event: str, *, force_hash: bool) -> None:
        observed_at = utc_now()
        for record in self._records:
            path = Path(record["path"])
            signature = None
            observed_sha256 = None
            error_text = None
            try:
                signature = self._signature(path)
                changed = (record["last_signature"] is not None
                           and signature != record["last_signature"])
                # Content was read exactly once during plan capture.  After
                # that boundary, the original path is metadata-only: polling
                # its sticky inode/size/mtime/ctime signature detects even a
                # same-size mutate/use/restore without reopening mutable
                # bytes.  Children consume the separate captured payload.
            except OSError as error:
                changed = True
                error_text = f"{type(error).__name__}: {error}"
            first = not record["observations"]
            mismatch = False
            if first or changed or force_hash or error_text is not None:
                observation: dict[str, object] = {
                    "event": event if not changed else "metadata_changed",
                    "observed_at_utc": observed_at,
                    "observed_sha256": observed_sha256,
                    "signature": (
                        None if signature is None else list(signature)),
                    "status": (
                        "CHANGED" if changed or mismatch or error_text
                        else "MATCH"),
                }
                if error_text is not None:
                    observation["error"] = error_text
                with self._lock:
                    record["observations"].append(observation)
            if (record["baseline_signature"] is None
                    and signature is not None):
                record["baseline_signature"] = list(signature)
            record["last_signature"] = signature
            if changed or mismatch or error_text is not None:
                self._failed.set()

    def _run(self) -> None:
        self._started_at_utc = utc_now()
        try:
            self._observe("monitor_started", force_hash=True)
        except BaseException as error:  # keep failures observable to parent
            self._record_failure(error)
        finally:
            self._ready.set()
        while not self._stop.wait(self.poll_seconds):
            try:
                self._poll_count += 1
                self._observe("poll", force_hash=False)
            except BaseException as error:
                self._record_failure(error)

    def start(self) -> "FileAuthorityMonitor":
        if self._thread is not None:
            raise RuntimeError("file authority monitor was already started")
        self._thread = threading.Thread(
            target=self._run, name="gpuwm-file-authority-monitor",
            daemon=False)
        self._thread.start()
        self._ready.wait()
        return self

    def wait_for_failure(self, timeout: float) -> bool:
        return self._failed.wait(timeout)

    @property
    def failed(self) -> bool:
        return self._failed.is_set()

    def stop(self) -> dict[str, object]:
        if self._thread is None:
            raise RuntimeError("file authority monitor was not started")
        self._stop.set()
        self._thread.join()
        try:
            self._observe("monitor_stopped", force_hash=True)
        except BaseException as error:
            self._record_failure(error)
        self._stopped_at_utc = utc_now()
        return self.result()

    def result(self) -> dict[str, object]:
        with self._lock:
            files = []
            for record in self._records:
                observations = list(record["observations"])
                files.append({
                    "authority": record["authority"],
                    "baseline_signature": record["baseline_signature"],
                    "expected_sha256": record["expected_sha256"],
                    "observations": observations,
                    "path": record["path"],
                    "status": (
                        "PASS" if observations
                        and all(item["status"] == "MATCH"
                                for item in observations)
                        else "FAIL"),
                })
            errors = list(self._monitor_errors)
        return {
            "errors": errors,
            "files": files,
            "poll_count": self._poll_count,
            "poll_seconds": self.poll_seconds,
            "schema": "gpuwm.file-authority-monitor/v1",
            "started_at_utc": self._started_at_utc,
            "status": "FAIL" if self.failed else "PASS",
            "stopped_at_utc": self._stopped_at_utc,
        }


def _summary_payload(
        plan: MultiRunPlan, runs: Sequence[PlannedRun], *,
        started_at_utc: str, completed_at_utc: str,
        duration_seconds: float, status: str, exit_code: int,
        preflight: GroupOutcome | None,
        execution: GroupOutcome | None,
        file_verification: Mapping[str, object],
        orchestration: Mapping[str, object],
        packing: PackingPlan | None = None) -> dict[str, object]:
    preflight_by_name = (
        {} if preflight is None
        else {record.name: record for record in preflight.records})
    execution_by_name = (
        {} if execution is None
        else {record.name: record for record in execution.records})
    rows = []
    for run in runs:
        spec = run.spec
        process = execution_by_name.get(spec.name)
        cupy_cache, cuda_cache = run.kernel_cache.paths(
            spec.cache, os.environ)
        rows.append({
            "arguments": list(spec.arguments),
            "plan_arguments": (None if spec.plan_arguments is None
                               else list(spec.plan_arguments)),
            "prepared_binding": (None if spec.prepared_binding is None
                                 else dict(spec.prepared_binding)),
            "kernel_cache": {"cuda": cuda_cache, "cupy": cupy_cache,
                             "mode": run.kernel_cache.label},
            "cache": str(spec.cache),
            "config": None if spec.config is None else str(spec.config),
            "config_sha256": (
                None if spec.config is None
                else spec.input_bindings[0].sha256),
            "device": {
                "index": run.gpu.index,
                "name": run.gpu.name,
                "selector": spec.device_selector,
                "uuid": run.gpu.uuid,
            },
            "log": str(spec.log),
            "gpu_lock_root": (
                None if run.lock_root is None else str(run.lock_root)),
            "module": spec.module,
            "name": spec.name,
            "outdir": str(spec.outdir),
            "input_authorities": [
                binding.as_dict() for binding in spec.input_bindings],
            "shared_input_authority_store": str(
                plan.input_authority_store),
            "read_only_inputs": [str(path) for path in spec.inputs],
            "preflight": (
                None if spec.name not in preflight_by_name
                else preflight_by_name[spec.name].as_dict()),
            "process": (
                {"status": "not_started", "exit_code": None,
                 "log": str(spec.log)}
                if process is None else process.as_dict()),
            "scratch": str(spec.scratch_dir),
        })
        if packing is not None:
            rows[-1]["packing"] = {
                "admission": dict(packing.decisions.get(spec.name, {})),
                "pricing": packing.pricing[spec.name].as_dict(),
            }
    execution_timing = _execution_timing(execution)
    if file_verification.get("status") != "PASS":
        execution_timing["overlap_ratio"] = None
        execution_timing["overlap_ratio_basis"] = (
            "unavailable: a declared file authority changed")
    return {
        "completed_at_utc": completed_at_utc,
        "duration_seconds": duration_seconds,
        "execution_timing": execution_timing,
        "exit_code": exit_code,
        "file_authority_verification": dict(file_verification),
        "input_authority_store": _input_authority_store_record(plan),
        "kernel_cache": plan.kernel_cache.receipt(),
        "orchestration": dict(orchestration),
        "packing": (_exclusive_packing_receipt(plan.packing)
                    if packing is None else packing.receipt()),
        "plan": str(plan.path),
        "plan_sha256": plan.sha256,
        "preflight_mode": plan.preflight,
        "runs": rows,
        "schema": SUMMARY_SCHEMA,
        "started_at_utc": started_at_utc,
        "status": status,
    }


def _execution_timing(
        execution: GroupOutcome | None) -> dict[str, object]:
    """Summarize overlap without claiming speedup from incomplete work."""

    if execution is None:
        return {
            "concurrent_wall_seconds": None,
            "overlap_ratio": None,
            "overlap_ratio_basis": "unavailable: execution not started",
            "observed_child_duration_sum_seconds": 0.0,
            "completed_child_duration_sum_seconds": None,
            "window_completed_at_utc": None,
            "window_started_at_utc": None,
        }
    durations = [
        record.duration_seconds for record in execution.records
        if record.duration_seconds is not None]
    observed_sum = round(sum(durations), 6)
    all_forecast_attempts_completed = bool(execution.records) and all(
        record.status in {"succeeded", "failed"}
        and record.duration_seconds is not None
        for record in execution.records)
    completed_sum = (
        observed_sum if all_forecast_attempts_completed else None)
    all_succeeded = bool(execution.records) and all(
        record.status == "succeeded" and record.exit_code == 0
        for record in execution.records)
    if execution.interrupted:
        overlap_ratio = None
        basis = "unavailable: execution was interrupted"
    elif not all_succeeded:
        overlap_ratio = None
        basis = "unavailable: not every forecast succeeded"
    elif completed_sum is None:
        overlap_ratio = None
        basis = "unavailable: one or more forecast durations were incomplete"
    elif execution.wall_seconds is None or execution.wall_seconds <= 0.0:
        overlap_ratio = None
        basis = "unavailable: concurrent execution window was not measured"
    else:
        overlap_ratio = round(completed_sum / execution.wall_seconds, 6)
        basis = (
            "completed_child_duration_sum_seconds / "
            "concurrent_wall_seconds; this measures process overlap, not "
            "performance against a serial baseline")
    return {
        "concurrent_wall_seconds": execution.wall_seconds,
        "overlap_ratio": overlap_ratio,
        "overlap_ratio_basis": basis,
        "observed_child_duration_sum_seconds": observed_sum,
        "completed_child_duration_sum_seconds": completed_sum,
        "window_completed_at_utc": execution.completed_at_utc,
        "window_started_at_utc": execution.started_at_utc,
    }


def _unstarted_authority_verification(reason: str) -> dict[str, object]:
    """Describe accurately why no continuous authority monitor was started."""

    return {
        "errors": [],
        "files": [],
        "poll_count": 0,
        "poll_seconds": AUTHORITY_MONITOR_POLL_SECONDS,
        "reason": reason,
        "schema": "gpuwm.file-authority-monitor/v1",
        "started_at_utc": None,
        "status": "NOT_STARTED",
        "stopped_at_utc": None,
    }


DRY_RUN_SCHEMA = "gpuwm.multi-run-dry-run/v1"


def _print_dry_run(plan: MultiRunPlan) -> None:
    """Print what the plan would launch; touch no device, claim nothing.

    The plan has been loaded exactly as a real launch loads it (paths,
    inputs, prepared bindings, packing and kernel-cache settings), so a
    plan that passes here fails later only on device, memory or runtime
    grounds.  Device selectors are printed as written: no ``nvidia-smi``
    is asked."""

    runs = []
    for spec in plan.runs:
        if spec.config is not None:
            command = ["woof", "run", str(spec.config), "--outdir",
                       str(spec.outdir), "--gpu-uuid", "<resolved at launch>"]
        else:
            command = [sys.executable, "-m", str(spec.module),
                       *_render_arguments(spec)]
        cupy, cuda = plan.kernel_cache.paths(spec.cache, os.environ)
        runs.append({
            "command": command,
            "device": spec.device_selector,
            "inputs": [binding.as_dict() for binding in spec.input_bindings],
            "kernel_cache": {"cuda": cuda, "cupy": cupy},
            "name": spec.name,
            "outdir": str(spec.outdir),
            "plan_arguments": (None if spec.plan_arguments is None
                               else list(spec.plan_arguments)),
            "prepared_binding": (None if spec.prepared_binding is None
                                 else dict(spec.prepared_binding)),
            "scratch": str(spec.scratch_dir),
        })
    print(json.dumps({
        "kernel_cache": plan.kernel_cache.receipt(),
        "packing": {"runs_per_gpu": plan.packing.runs_per_gpu,
                    "mem_margin_gib": plan.packing.margin_gib,
                    "mps": plan.packing.mps,
                    "settings_source": dict(plan.packing.sources)},
        "plan": str(plan.path),
        "plan_sha256": plan.sha256,
        "preflight": plan.preflight,
        "runs": runs,
        "schema": DRY_RUN_SCHEMA,
        "summary": str(plan.summary),
    }, indent=2, sort_keys=True), flush=True)
    print("multi-run --dry-run: the plan loads and every prepared run is "
          "bound; no device was queried and nothing was claimed or "
          "launched", file=sys.stderr, flush=True)


def multi_run_main(args: argparse.Namespace) -> int:
    """CLI implementation for ``woof multi-run PLAN.toml``."""

    state = OrchestrationState()
    started_at_utc = utc_now()
    started_monotonic = time.monotonic()
    plan: MultiRunPlan | None = None
    runs: tuple[PlannedRun, ...] = ()
    preflight: GroupOutcome | None = None
    execution: GroupOutcome | None = None
    monitor: FileAuthorityMonitor | None = None
    packing: PackingPlan | None = None
    verification: Mapping[str, object] = _unstarted_authority_verification(
        "plan and runtime paths have not yet been prepared")

    def stop_monitor() -> Mapping[str, object]:
        nonlocal verification
        if monitor is None:
            return verification
        if verification.get("stopped_at_utc") is None:
            verification = monitor.stop()
        return verification

    def publish(*, status: str, exit_code: int,
                receipt_stage: str) -> int:
        assert plan is not None
        completed_at_utc = utc_now()
        state.stage = receipt_stage
        payload = _summary_payload(
            plan, runs, started_at_utc=started_at_utc,
            completed_at_utc=completed_at_utc,
            duration_seconds=round(
                time.monotonic() - started_monotonic, 6),
            status=status, exit_code=exit_code,
            preflight=preflight, execution=execution,
            file_verification=verification,
            orchestration=_orchestration_record(
                state, plan, preflight, execution),
            packing=packing)
        state.stage = "summary_publication"
        try:
            _write_summary(plan.summary, payload)
        except (OSError, ValueError) as error:
            print(
                f"multi-run: summary publication failed; any summary "
                f"written by another invocation was preserved: {error}",
                file=sys.stderr, flush=True)
            return 130 if exit_code == 130 else 1
        print(f"multi-run: summary {plan.summary}", flush=True)
        return exit_code

    try:
        state.stage = "plan_load"
        plan = load_plan(
            args.plan, summary_override=getattr(args, "summary", None),
            preflight_override=getattr(args, "preflight", None))
        plan = replace(plan, packing=packing_request(plan.packing, args),
                       kernel_cache=kernel_cache_request(plan, args))
        if getattr(args, "dry_run", False):
            _print_dry_run(plan)
            return 0

        state.stage = "lock_root_resolution"
        lock_root = _validate_lock_root_isolation(
            plan, _parent_lock_root(os.environ))

        state.stage = "device_discovery"
        try:
            inventory = query_gpus()
        except RuntimeError as error:
            raise ValueError(str(error)) from error
        runs = resolve_devices(plan, inventory, lock_root=lock_root)
        if plan.packing.packed:
            state.stage = "packing_admission_plan"
            packing, runs = plan_packing(
                plan, runs, lock_root=lock_root, meter=_card_meter())
            packing.hold_shared_locks()
            if plan.packing.mps:
                state.stage = "mps_start"
                daemon = MpsDaemon(
                    plan.packing.mps_dir, sorted(packing.cards),
                    thread_percentage=(
                        plan.packing.mps_active_thread_percentage))
                if not daemon.external:
                    _validate_outside_plan_paths(
                        plan, daemon.pipe_dir.parent, "MPS directory",
                        "give --mps-dir a directory outside every plan, "
                        "input, summary, output, scratch, and cache path")
                try:
                    daemon.start()
                except MPSUnavailable as error:
                    raise ValueError(f"multi-run --mps: {error}") from error
                packing.mps = daemon
                overrides = tuple(sorted(daemon.environment().items()))
                runs = tuple(replace(run, environment_overrides=overrides)
                             for run in runs)

        def execute_group(group: Sequence[PlannedRun],
                          **kwargs) -> GroupOutcome:
            if packing is None:
                return _execute_group(group, **kwargs)
            # An estimate check holds no forecast memory; an alloc check
            # allocates the priced peak, so it is admitted like the run.
            priced = not (kwargs.get("phase") == "check"
                          and plan.preflight == "estimate")
            return _execute_packed_group(
                group, packing=packing, priced=priced, **kwargs)

        state.stage = "summary_probe"
        _prepare_summary_destination(plan.summary, state)
        state.stage = "directory_claim"
        _prepare_directories(plan, state)

        state.stage = "authority_monitor_start"
        monitor = FileAuthorityMonitor(
            plan, poll_seconds=AUTHORITY_MONITOR_POLL_SECONDS)
        monitor.start()
        verification = monitor.result()
        if monitor.failed:
            verification = stop_monitor()
            print(
                "multi-run: declared file authority changed before any "
                "child was launched", file=sys.stderr, flush=True)
            return publish(
                status="failed", exit_code=1,
                receipt_stage="authority_monitor_start")

        if plan.preflight == "off":
            print(
                "WARNING: multi-run preflight is off; launching without a "
                "woof check sizing report.", file=sys.stderr, flush=True)
        else:
            state.stage = "config_checks"
            checkable = tuple(run for run in runs if run.spec.is_config_run)
            for run in runs:
                if not run.spec.is_config_run:
                    print(
                        f"multi-run check {run.spec.name}: delegated to "
                        f"production module {run.spec.module}; its own input "
                        "and memory gates run inside the locked child.",
                        flush=True)
            if checkable:
                preflight = execute_group(
                    checkable, phase="check",
                    command_for=lambda run: _check_command(
                        run, plan.preflight),
                    log_for=lambda run: run.spec.preflight_log)
            else:
                preflight = GroupOutcome((), False)
            if preflight.interrupted:
                verification = stop_monitor()
                return publish(
                    status="interrupted", exit_code=130,
                    receipt_stage="config_checks")
            for record in preflight.records:
                if record.status != "succeeded":
                    print(
                        f"WARNING: woof check for {record.name!r} reported "
                        f"{record.status} (exit={record.exit_code}); launch "
                        f"continues because check reports sizing and does "
                        f"not gate woof run. See {record.log}.",
                        file=sys.stderr, flush=True)

        if monitor.failed:
            verification = stop_monitor()
            print(
                "multi-run: declared file authority changed while config "
                "checks ran; no forecast child was launched",
                file=sys.stderr, flush=True)
            return publish(
                status="failed", exit_code=1,
                receipt_stage="config_checks")

        state.stage = "forecast"
        execution = execute_group(
            runs, phase="run", command_for=_run_command,
            log_for=lambda run: run.spec.log)

        state.stage = "authority_monitor_stop"
        verification = stop_monitor()
        if execution.interrupted:
            exit_code = 130
            status = "interrupted"
        else:
            succeeded = all(
                record.status == "succeeded" and record.exit_code == 0
                for record in execution.records)
            exit_code = 0 if succeeded else 1
            check_warning = (preflight is not None and any(
                record.status != "succeeded"
                for record in preflight.records))
            status = (
                "complete_with_warnings" if succeeded and check_warning
                else "complete" if succeeded else "failed")
        if verification["status"] != "PASS":
            print(
                "multi-run: declared file authority changed while children "
                "ran; receipt marks the execution failed",
                file=sys.stderr, flush=True)
            if exit_code != 130:
                exit_code = 1
                status = "failed"

        published = publish(
            status=status, exit_code=exit_code,
            receipt_stage=("forecast" if exit_code == 130 else "complete"))
        success_count = sum(
            record.status == "succeeded" for record in execution.records)
        print(
            f"multi-run: {success_count}/{len(execution.records)} runs "
            f"completed successfully; aggregate exit={published}; "
            f"summary {plan.summary}", flush=True)
        return published
    except KeyboardInterrupt:
        interrupted_stage = state.stage
        if monitor is not None:
            try:
                verification = stop_monitor()
            except BaseException as error:
                verification = monitor.result()
                verification = dict(verification)
                errors = list(verification.get("errors", []))
                errors.append({
                    "error": (
                        "monitor stop during interruption failed: "
                        f"{type(error).__name__}: {error}"),
                    "observed_at_utc": utc_now(),
                })
                verification["errors"] = errors
                verification["status"] = "FAIL"
        state.stage = interrupted_stage
        print(
            f"multi-run: interrupted during {interrupted_stage}; child "
            "processes were not terminated", file=sys.stderr, flush=True)
        if (not state.summary_capable or plan is None or not runs):
            print(
                "multi-run: no interrupted summary was published because "
                "create-only summary capability had not been established",
                file=sys.stderr, flush=True)
            return 130
        completed_at_utc = utc_now()
        payload = _summary_payload(
            plan, runs, started_at_utc=started_at_utc,
            completed_at_utc=completed_at_utc,
            duration_seconds=round(
                time.monotonic() - started_monotonic, 6),
            status="interrupted", exit_code=130,
            preflight=preflight, execution=execution,
            file_verification=verification,
            orchestration=_orchestration_record(
                state, plan, preflight, execution),
            packing=packing)
        try:
            _write_summary(plan.summary, payload)
        except (OSError, ValueError) as error:
            print(
                "multi-run: interrupted summary publication failed; any "
                f"summary written by another invocation was preserved: "
                f"{error}", file=sys.stderr, flush=True)
        else:
            print(
                f"multi-run: interrupted summary {plan.summary}",
                file=sys.stderr, flush=True)
        return 130
    except BaseException as error:
        failure_stage = state.stage
        reported_error = (
            error.original if isinstance(error, GroupExecutionError)
            else error)
        if isinstance(error, GroupExecutionError):
            if error.phase == "check":
                preflight = error.outcome
            elif error.phase == "run":
                execution = error.outcome
        state.error = f"{type(reported_error).__name__}: {reported_error}"
        if monitor is not None:
            try:
                verification = stop_monitor()
            except BaseException as monitor_error:
                verification = dict(monitor.result())
                errors = list(verification.get("errors", []))
                errors.append({
                    "error": (
                        "monitor stop after orchestration failure failed: "
                        f"{type(monitor_error).__name__}: {monitor_error}"),
                    "observed_at_utc": utc_now(),
                })
                verification["errors"] = errors
                verification["status"] = "FAIL"
        state.stage = failure_stage
        print(
            f"multi-run: failed during {failure_stage}: "
            f"{type(reported_error).__name__}: {reported_error}",
            file=sys.stderr, flush=True)
        if state.summary_capable and plan is not None and runs:
            try:
                completed_at_utc = utc_now()
                payload = _summary_payload(
                    plan, runs, started_at_utc=started_at_utc,
                    completed_at_utc=completed_at_utc,
                    duration_seconds=round(
                        time.monotonic() - started_monotonic, 6),
                    status="failed", exit_code=1,
                    preflight=preflight, execution=execution,
                    file_verification=verification,
                    orchestration=_orchestration_record(
                        state, plan, preflight, execution),
                    packing=packing)
                _write_summary(plan.summary, payload)
            except BaseException as publication_error:
                print(
                    "multi-run: failed-summary publication also failed; any "
                    "summary written by another invocation was preserved: "
                    f"{type(publication_error).__name__}: "
                    f"{publication_error}", file=sys.stderr, flush=True)
            else:
                print(
                    f"multi-run: failed summary {plan.summary}",
                    file=sys.stderr, flush=True)
        raise
    finally:
        if packing is not None:
            _release_packing(packing, (preflight, execution))
        # The monitor is deliberately non-daemon: never let an unexpected
        # exception strand it and keep interpreter shutdown alive.  Cleanup
        # errors are reported but cannot replace the active exception.
        if (monitor is not None and monitor._thread is not None
                and verification.get("stopped_at_utc") is None):
            try:
                verification = monitor.stop()
            except BaseException as cleanup_error:
                print(
                    "multi-run: authority-monitor cleanup failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}",
                    file=sys.stderr, flush=True)


def register_cli(subparsers: argparse._SubParsersAction) -> None:
    """Register the process-level multi-GPU orchestration command."""

    parser = subparsers.add_parser(
        "multi-run",
        help=("launch independent production forecasts, one per GPU or "
              "packed several to a GPU by memory (--runs-per-gpu)"))
    parser.add_argument(
        "plan", type=Path, metavar="PLAN.toml",
        help="versioned plan with one or more [[run]] entries")
    parser.add_argument(
        "--summary", type=Path, default=None, metavar="SUMMARY.json",
        help="summary path relative to the plan (default PLAN.summary.json)")
    parser.add_argument(
        "--preflight", choices=PREFLIGHT_MODES, default=None,
        help="override the plan's woof check mode: estimate, alloc, or off")
    parser.add_argument(
        "--runs-per-gpu", default=None, metavar="N|auto",
        help=("let up to N runs (auto: up to "
              f"{AUTO_MAX_SLOTS_PER_GPU}) share one GPU, each admitted when "
              "its priced forecast peak plus the margin fits the card's "
              "measured free memory beside its siblings; runs that do not "
              "fit wait for a sibling to finish (overrides the plan's "
              "[devices] runs_per_gpu; default 1, one run per GPU)"))
    parser.add_argument(
        "--mem-margin-gib", type=float, default=None, metavar="GiB",
        help=("card memory kept free beside packed runs' priced peaks "
              f"(default {DEFAULT_MEM_MARGIN_GIB:g}; only with "
              "--runs-per-gpu above 1 or auto)"))
    parser.add_argument(
        "--mps", action=argparse.BooleanOptionalAction, default=None,
        help=("start (or reuse) the CUDA MPS control daemon so packed runs' "
              "kernels share the card instead of time-slicing; stopped on "
              "exit if started here (overrides the plan's [devices] mps)"))
    parser.add_argument(
        "--mps-dir", type=Path, default=None, metavar="DIR",
        help="where the MPS pipe and log directories go (with --mps)")
    parser.add_argument(
        "--mps-active-thread-percentage", type=int, default=None,
        metavar="P",
        help="CUDA_MPS_ACTIVE_THREAD_PERCENTAGE for each run (with --mps)")
    parser.add_argument(
        "--kernel-cache", default=None, metavar="shared|per-run|DIR",
        help=("where runs keep compiled GPU kernels: shared (default: the "
              "persistent CuPy and CUDA caches an ordinary woof run uses, "
              "so only the first run compiles), per-run (each run's own "
              "empty <cache>/cupy and <cache>/cuda, for cross-device "
              "comparisons and forensics) or a shared directory DIR "
              "(overrides the plan's [cache] kernels)"))
    parser.add_argument(
        "--dry-run", action="store_true",
        help=("load and bind the plan, print every run's command, derived "
              "prepared digests and kernel caches as JSON, and stop: no "
              "device is queried and nothing is claimed or launched"))
    parser.set_defaults(func=multi_run_main)


if __name__ == "__main__":
    sys.exit(_module_entry())


__all__ = [
    "PLAN_SCHEMA", "PREFLIGHT_MODES", "SUMMARY_SCHEMA",
    "SUPPORTED_PREPARED_RUNNERS", "DRY_RUN_SCHEMA",
    "PREPARED_BINDING_BASIS", "GroupExecutionError", "GroupOutcome",
    "MultiRunPlan", "PlannedRun", "ProcessRecord", "RunSpec",
    "ADMISSION_RECHECK_SECONDS", "CardMeter", "PACKED_ADMISSION_RULE",
    "PackingPlan", "PackingRequest", "RunPricing",
    "child_environment", "kernel_cache_request", "load_plan",
    "multi_run_main", "packed_admission",
    "packing_request", "plan_packing", "price_run", "register_cli",
    "resolve_devices",
]
