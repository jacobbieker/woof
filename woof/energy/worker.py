"""``woof energy worker``: drain a hindcast work queue, one loop per GPU.

The parent process resolves its devices and starts one **slot** process
per GPU (``python -m woof.energy.worker --slot-config FILE``).  A slot

1. takes a slot lock for its GPU UUID (a :class:`~woof.supervisor.GPUFileLock`
   at ``energy-worker-<digest>.lock`` beside the supervisor's own UUID
   lock), so two workers on one host never share a card;
2. runs with the environment :func:`woof.multi_run.child_environment`
   builds -- ``CUDA_VISIBLE_DEVICES=<uuid>`` and its own temp, CuPy and
   CUDA caches;
3. before each claim, probes the supervisor's UUID lock and waits while
   another woof process (a forecast outside the queue) holds the card;
4. claims a job (:meth:`woof.energy.workqueue.WorkQueue.claim`), renews
   the lease from a heartbeat thread, runs the **runner**, uploads the
   runner's log to ``logs/<id>/attempt-NNN-<claimed>-<worker>.log`` and marks the
   job done, failed (backoff or dead) or released.

The slot lock is a separate file because the runner -- ``woof energy
hindcast run-local`` -> ``woof run`` -- takes the supervisor's UUID lock
itself; holding that lock in the slot would refuse the slot's own job.

The runner is any callable ``RunnerRequest -> RunnerOutcome``.  The CLI's
is :class:`SubprocessRunner` over a command template; the default is
:data:`DEFAULT_RUNNER_TEMPLATE`, ``--runner-cmd`` names another.  Template
placeholders are :data:`RUNNER_PLACEHOLDERS`; any other ``{name}`` is
refused.  The runner also sees ``WOOF_ENERGY_JOB_ID``,
``WOOF_ENERGY_JOB_ATTEMPT``, ``WOOF_ENERGY_JOB_PATH``,
``WOOF_ENERGY_RESULT_PATH`` and ``WOOF_ENERGY_QUEUE`` in its environment.
A runner that writes JSON to ``{result}`` has it embedded in the job's
``done/`` record.

What is pinned:

* exit 0 completes the job (exactly once, see :mod:`woof.energy.workqueue`);
  an exit in :data:`~woof.energy.workqueue.PERMANENT_EXIT_CODES` (woof's
  refusal, 2) is dead at once; any other exit, a timeout, or a runner that
  could not start is a charged failure;
* SIGTERM (spot reclaim, preemptible VM, ``systemctl stop``) or SIGINT:
  the slot forwards SIGTERM to the runner's process group, waits
  ``--term-grace-s`` (default 20 s, inside GCP's 30 s preemption notice),
  then SIGKILLs it and **releases** the lease uncharged.  A runner that
  exits 0 inside the grace still completes;
* a heartbeat that finds the lease taken over keeps the runner going (the
  completion marker decides which copy counts); a heartbeat that finds the
  job already done elsewhere cancels the runner;
* a slot whose parent died stops claiming after its current job; a slot
  whose last :data:`MAX_CONSECUTIVE_JOB_FAILURES` jobs all failed stops
  claiming (exit 1);
* the default runner is refused at start-up when this install has no
  ``woof energy hindcast run-local`` (its exit 2 would kill every job).

Exit codes: 0 drained or idle, 1 a slot failed, 2 refused, 75 stopped by
SIGTERM (work remains), 130 stopped by SIGINT.

Packing several jobs on one GPU (``--slots-per-gpu N|auto``)
------------------------------------------------------------

Most hindcast clusters are one small farm whose domains (0.3-2 M cells)
cannot fill a datacentre card, so one job per GPU rents a card that idles
between kernel launches.  ``--slots-per-gpu`` starts several slots per
GPU.  The default, 1, is the behaviour above, unchanged.  With N > 1 or
``auto``:

* the parent holds the per-UUID worker lock (``energy-worker-<digest>
  .lock``, the file a single-slot worker's slot holds) and hands it to its
  slots, so one worker owns a card whatever its slot count and a
  single-slot worker and a packed one still exclude each other;
* slot *k* of a card holds ``energy-worker-<digest>.slot-<k>.lock`` and,
  while it runs a job, writes the job's memory reservation beside it
  (``.slot-<k>.json``, :data:`RESERVATION_SCHEMA`).  The runner inherits
  the slot lock's descriptor, so a reservation stays live while its runner
  lives, even if the slot itself is killed; a reservation whose lock
  nobody holds is stale and ignored;
* **memory-aware admission.**  Under the card's admission lock
  (``.admit.lock``) a slot reads the live reservations and claims only a
  job whose declared memory fits: ``sum(reserved) + need + margin <=
  card``, where ``need`` is the job's ``resources.gpu_mem_gib`` (the
  planner's priced peak, which includes the CUDA context) and ``margin``
  is ``--mem-margin-gib`` (default :data:`DEFAULT_MEM_MARGIN_GIB`, an
  assumption: driver, MPS server and allocator fragmentation the priced
  peak does not hold).  A job that declares nothing is priced at
  ``--default-job-mem-gib``, or -- the conservative default -- the whole
  card, so it only ever runs alone.  An idle card admits any one job,
  however large, so every job can always run (as it would with one slot).
  A job declaring 0 GiB -- a ``kind: prefetch`` forcing download, which
  counts as 0 even undeclared -- is never held back and reserves nothing.
  The queue gates on dependencies (``depends_on``) before memory, so a
  forecast job waits for its cycle's prefetch first;
* ``auto`` starts ``min(8, floor((card - margin) / 3.4 GiB), CPUs per
  GPU)`` slots: 3.4 GiB is the smallest job the cost model's memory fit
  prices (its fixed intercept, :data:`woof.energy.cost.VRAM_INTERCEPT_GIB`)
  and each slot drives its kernels from one CPU core;
* each slot's runner gets its own ``WOOF_GPU_LOCK_ROOT``
  (``.slot-<k>.runner-locks``), so the runners' own ``woof run`` UUID
  locks do not refuse each other.  A ``woof run`` still sees the sibling
  runners as CUDA co-tenants and prices its own reservation against the
  device's measured free memory (:func:`woof.supervisor.shared_gpu_admission`),
  a second, measured check behind the declared one;
* the slot still waits while a woof process outside the queue holds the
  card's supervisor lock; one started while packed jobs run is admitted or
  refused by that same measured shared-GPU admission;
* when a slot dies unexpectedly the parent kills the runner it recorded
  before restarting it (its lease would expire anyway).

Card memory comes from NVML (``nvidia-smi``) or ``--gpu-mem-gib``, which
``--fake-gpus`` needs.  A large job can wait while small ones keep a card
busy; it runs once the queue's small jobs drain or a card goes idle.

**CUDA MPS** (``--mps``).  Without MPS, kernels from separate processes
time-slice: the card runs one context at a time, so packing hides launch
gaps, host-side work and I/O but never runs two small grids' kernels side
by side.  The Multi-Process Service funnels every slot's work through one
server context, so kernels from different processes run concurrently and
small grids share the SMs.  ``--mps`` starts ``nvidia-cuda-mps-control
-d`` with ``CUDA_MPS_PIPE_DIRECTORY``/``CUDA_MPS_LOG_DIRECTORY`` under
``--mps-dir`` (default a short directory under the system temp: the
control socket path must stay under :data:`MPS_PIPE_PATH_MAX` characters),
gives the slots the same variables, optionally
``CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`` (``--mps-active-thread-percentage``),
and stops the daemon (``quit``) on exit.  When ``CUDA_MPS_PIPE_DIRECTORY``
is already set (a daemon this process does not own, such as the NVIDIA
device plugin's MPS sharing) or a daemon already answers at that pipe
directory, it is used and left running.  The worker refuses when the
control binary is not on ``PATH``.  MPS has not been timed on hardware by
this tree; ``woof energy estimate --slots-per-gpu --mps`` models it.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import json
import math
import os
import random
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from woof.energy.packing import (AUTO_MAX_SLOTS_PER_GPU,
                                 DEFAULT_MEM_MARGIN_GIB, SLOTS_AUTO,
                                 parse_slots_per_gpu)
from woof.energy.workqueue import (
    PERMANENT_EXIT_CODES, Claim, JobSpecError, LeaseLost, QueueError,
    WorkQueue, declared_gpu_mem_gib, iso_utc, tail_text)

WORKER_SUMMARY_SCHEMA = "woof-energy.worker-summary.v1"
SLOT_SUMMARY_SCHEMA = "woof-energy.worker-slot.v1"
SLOT_CONFIG_SCHEMA = "woof-energy.worker-slot-config.v1"
RESERVATION_SCHEMA = "woof-energy.worker-reservation.v1"

#: ``--slots-per-gpu auto``, its slot cap and the default memory margin
#: (GiB) live in :mod:`woof.energy.packing`, shared with the estimate and
#: the deploy renderers.
#: The MPS control binary ``--mps`` starts.
MPS_CONTROL = "nvidia-cuda-mps-control"
#: Longest ``<pipe dir>/control`` socket path accepted (AF_UNIX paths
#: stop at 107 bytes on Linux; the rest is margin).
MPS_PIPE_PATH_MAX = 100
#: Seconds an MPS control command may take before it is a failure.
MPS_COMMAND_TIMEOUT_S = 30.0

#: File in a slot's work directory holding the running runner's
#: process-group id (the parent kills it on a second signal).
RUNNER_PGID_FILE = "runner.pgid"

#: The sibling hindcast unit's door: run one job spec on this host.
DEFAULT_RUNNER_TEMPLATE = "{python} -m woof energy hindcast run-local {job}"

#: Every ``{name}`` a runner template may use.
RUNNER_PLACEHOLDERS = ("python", "job", "job_id", "attempt", "workdir",
                       "result", "gpu_uuid", "queue")

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED = 2
EXIT_STOPPED = 75
EXIT_INTERRUPTED = 130

#: A slot that crashes (not drained, not refused, not stopped) is started
#: again at most this many times.
MAX_SLOT_RESTARTS = 3

#: Consecutive queue errors a slot rides out before it gives up.
MAX_CONSECUTIVE_QUEUE_ERRORS = 20

#: Consecutive jobs a slot may fail before it stops claiming: a broken
#: runner or a sick GPU should not burn through the whole queue.
MAX_CONSECUTIVE_JOB_FAILURES = 5

#: Attempts at settling a finished job (done/failed/released) through
#: transient store errors, with capped exponential waits between them.
SETTLE_ATTEMPTS = 8

#: Logs larger than this upload their tail only.
LOG_UPLOAD_MAX_BYTES = 32 * 1024 * 1024

DEFAULT_TIMEOUT_S = 24 * 3600.0
DEFAULT_TERM_GRACE_S = 20.0
DEFAULT_POLL_S = 30.0


class WorkerError(RuntimeError):
    """The worker refused to start."""


class GPUSelectionError(WorkerError, ValueError):
    """``--gpus`` names devices this host does not have."""


class RunnerTemplateError(WorkerError, ValueError):
    """``--runner-cmd`` is empty or uses an unknown placeholder."""


class PackingError(WorkerError, ValueError):
    """``--slots-per-gpu`` and its memory flags cannot be honoured as asked."""


class MPSUnavailable(WorkerError):
    """``--mps`` was asked for and the MPS control daemon cannot run."""


def _note(message: str) -> None:
    print(f"woof energy worker: {message}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RunnerRequest:
    job_id: str
    job_path: Path
    attempt: int
    workdir: Path
    log_path: Path
    result_path: Path
    env: dict[str, str]
    timeout_s: float | None
    cancel: threading.Event
    gpu_uuid: str | None
    queue_url: str
    #: Called with the runner's process-group id once it starts, so the
    #: worker can kill a runner that outlives its slot.
    on_start: Callable[[int], None] | None = None
    #: Descriptors the runner process inherits: a packed slot's lock, so
    #: the slot's memory reservation stays live while the runner does.
    pass_fds: tuple[int, ...] = ()


@dataclass(frozen=True)
class RunnerOutcome:
    exit_code: int | None
    timed_out: bool = False
    cancelled: bool = False
    error: str | None = None


Runner = Callable[[RunnerRequest], RunnerOutcome]

_PLACEHOLDER = re.compile(r"\{([^{}]*)\}")


def _signal_group(process: subprocess.Popen, signum: int) -> None:
    try:
        os.killpg(process.pid, signum)
    except (ProcessLookupError, PermissionError):
        pass


class SubprocessRunner:
    """Run a command template in its own session, log to a file.

    Cancellation and timeout send SIGTERM to the runner's process group,
    then SIGKILL after ``term_grace_s``.
    """

    def __init__(self, template: str = DEFAULT_RUNNER_TEMPLATE, *,
                 term_grace_s: float = DEFAULT_TERM_GRACE_S):
        try:
            parts = shlex.split(template)
        except ValueError as error:
            raise RunnerTemplateError(
                f"--runner-cmd is not a shell word list: {error}") from error
        if not parts:
            raise RunnerTemplateError("--runner-cmd is empty")
        unknown = sorted({name for part in parts
                          for name in _PLACEHOLDER.findall(part)
                          if name not in RUNNER_PLACEHOLDERS})
        if unknown:
            raise RunnerTemplateError(
                f"--runner-cmd uses unknown placeholder(s) "
                f"{', '.join('{' + n + '}' for n in unknown)}; known: "
                f"{', '.join('{' + n + '}' for n in RUNNER_PLACEHOLDERS)}")
        self.template = template
        self.parts = parts
        self.term_grace_s = term_grace_s

    def argv(self, request: RunnerRequest) -> list[str]:
        values = {"python": sys.executable, "job": str(request.job_path),
                  "job_id": request.job_id, "attempt": str(request.attempt),
                  "workdir": str(request.workdir),
                  "result": str(request.result_path),
                  "gpu_uuid": request.gpu_uuid or "",
                  "queue": request.queue_url}
        return [_PLACEHOLDER.sub(lambda m: values[m.group(1)], part)
                for part in self.parts]

    def __call__(self, request: RunnerRequest) -> RunnerOutcome:
        argv = self.argv(request)
        with open(request.log_path, "ab") as log:
            log.write(f"# woof energy worker runs: {shlex.join(argv)}\n"
                      .encode("utf-8"))
            log.flush()
            try:
                process = subprocess.Popen(
                    argv, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, env=request.env,
                    cwd=request.workdir, start_new_session=True,
                    pass_fds=tuple(request.pass_fds))
            except OSError as error:
                return RunnerOutcome(None, error=f"runner could not start: "
                                                 f"{error}")
            if request.on_start is not None:
                request.on_start(process.pid)
            deadline = (None if request.timeout_s is None
                        else time.monotonic() + request.timeout_s)
            timed_out = cancelled = False
            kill_at: float | None = None
            while True:
                try:
                    code = process.wait(timeout=0.25)
                    break
                except subprocess.TimeoutExpired:
                    pass
                now = time.monotonic()
                if kill_at is None:
                    if request.cancel.is_set():
                        cancelled = True
                    elif deadline is not None and now >= deadline:
                        timed_out = True
                    if cancelled or timed_out:
                        _signal_group(process, signal.SIGTERM)
                        kill_at = now + self.term_grace_s
                elif now >= kill_at:
                    _signal_group(process, signal.SIGKILL)
                    kill_at = float("inf")
            if cancelled or timed_out:
                # Children the runner left behind in its session.
                _signal_group(process, signal.SIGKILL)
        return RunnerOutcome(code, timed_out=timed_out, cancelled=cancelled)


# --------------------------------------------------------------------------
# Devices
# --------------------------------------------------------------------------


def fake_gpus(count: int):
    """``count`` stand-in GPU identities for CPU testing (no device read)."""

    from woof.supervisor import GPUIdentity

    return tuple(GPUIdentity(f"GPU-fake-{index:04d}", "fake", "fake GPU", index)
                 for index in range(count))


def select_gpus(spec: str, inventory) -> tuple:
    """``all``, a count ``N`` (the first N by index), or a UUID comma list."""

    ordered = sorted(inventory, key=lambda gpu: (gpu.index is None,
                                                 gpu.index or 0))
    if spec == "all":
        return tuple(ordered)
    if spec.isdigit():
        count = int(spec)
        if not 1 <= count <= len(ordered):
            raise GPUSelectionError(
                f"--gpus {spec} asks for {count} GPUs; this host has "
                f"{len(ordered)}")
        return tuple(ordered[:count])
    wanted = [part.strip() for part in spec.split(",") if part.strip()]
    by_uuid = {gpu.uuid: gpu for gpu in ordered}
    unknown = [name for name in wanted if name not in by_uuid]
    if not wanted or unknown:
        raise GPUSelectionError(
            f"--gpus takes all, a count, or GPU UUIDs; {unknown or spec!r} "
            f"not among {sorted(by_uuid)}")
    if len(set(wanted)) != len(wanted):
        raise GPUSelectionError(f"--gpus names a UUID twice: {spec!r}")
    return tuple(by_uuid[name] for name in wanted)


def default_runner_available() -> bool:
    """Whether this install has ``woof energy hindcast run-local``."""

    from woof.cli import build_parser

    def child(parser, name):
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                return action.choices.get(name)
        return None

    parser = build_parser()
    for name in ("energy", "hindcast", "run-local"):
        parser = child(parser, name)
        if parser is None:
            return False
    return True


def worker_lock_path(gpu_uuid: str) -> Path:
    """The slot lock, beside the supervisor's UUID lock, not the same file."""

    from woof.supervisor import default_lock_path

    path = default_lock_path(gpu_uuid)
    return path.with_name("energy-worker-" + path.name)


def gpu_is_free(gpu_uuid: str) -> bool:
    """Whether the supervisor's UUID lock is free right now (probe only)."""

    from woof.supervisor import GPUAlreadyLockedError, GPUFileLock

    lock = GPUFileLock(gpu_uuid, run_id=f"energy-worker-probe-{os.getpid()}")
    try:
        lock.acquire()
    except GPUAlreadyLockedError:
        return False
    lock.release()
    return True


# --------------------------------------------------------------------------
# Packing (several slots per GPU)
# --------------------------------------------------------------------------


def gpu_is_free_shared(gpu_uuid: str) -> bool:
    """Whether no process holds the supervisor's UUID lock exclusively.

    A packed card's slots probe with a SHARED lock, so sibling slots
    probing at the same instant never see each other as "another woof
    process" (:func:`gpu_is_free` takes the lock exclusively, which is
    right for a card's only slot).  Nothing is written."""

    import fcntl

    from woof.supervisor import default_lock_path

    try:
        fd = os.open(default_lock_path(gpu_uuid), os.O_RDONLY)
    except FileNotFoundError:
        return True
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except OSError as error:
        if error.errno in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
            return False
        raise
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True
    finally:
        os.close(fd)


def _lock_held(path: Path) -> bool:
    """Whether another open file description holds ``path``'s flock."""

    import fcntl

    try:
        fd = os.open(path, os.O_RDWR)
    except FileNotFoundError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        if error.errno in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
            return True
        raise
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


class CardLedger:
    """One packed slot's view of its card: its lock, its reservation and
    the admission rule (see the module docstring)."""

    def __init__(self, gpu_uuid: str, index: int, *, card_gib: float,
                 margin_gib: float, default_job_gib: float | None):
        base = worker_lock_path(gpu_uuid)
        stem = base.name[:-len(".lock")] if base.name.endswith(".lock") \
            else base.name
        self.gpu_uuid = gpu_uuid
        self.index = index
        self.card_gib = float(card_gib)
        self.margin_gib = float(margin_gib)
        self.default_job_gib = default_job_gib
        self.directory = base.parent
        self.stem = stem
        self.slot_lock_path = self.directory / f"{stem}.slot-{index}.lock"
        self.reservation_path = self.directory / f"{stem}.slot-{index}.json"
        self.admit_lock_path = self.directory / f"{stem}.admit.lock"
        self.runner_lock_root = (self.directory
                                 / f"{stem}.slot-{index}.runner-locks")

    def need_gib(self, declared: float | None) -> tuple[float, str]:
        """What a job is priced at, and on what basis."""

        if declared is not None:
            return float(declared), "declared"
        if self.default_job_gib is not None:
            return float(self.default_job_gib), "--default-job-mem-gib"
        return self.card_gib, "undeclared: the whole card"

    def live_reservations(self) -> list[dict[str, Any]]:
        """Reservations of the card's other slots whose lock is held."""

        rows = []
        for path in sorted(self.directory.glob(f"{self.stem}.slot-*.json")):
            if path == self.reservation_path:
                continue
            if not _lock_held(path.with_suffix(".lock")):
                continue  # stale: nobody holds that slot any more
            try:
                row = json.loads(path.read_text("utf-8"))
            except (OSError, ValueError):
                continue  # removed or being replaced: not a reservation
            if (isinstance(row, dict) and row.get("schema") == RESERVATION_SCHEMA
                    and isinstance(row.get("gpu_mem_gib"), (int, float))
                    and row["gpu_mem_gib"] > 0):
                rows.append(row)
        return rows

    def fits(self, declared: float | None,
             live: list[dict[str, Any]]) -> bool:
        if declared is not None and declared <= 0:
            return True  # no GPU (a prefetch job): never held back
        if not live:
            return True  # an idle card admits any one job
        used = sum(float(row["gpu_mem_gib"]) for row in live)
        need, _ = self.need_gib(declared)
        return used + need + self.margin_gib <= self.card_gib + 1e-9

    def reserve(self, claim: Claim, declared: float | None,
                live: list[dict[str, Any]]) -> dict[str, Any]:
        need, basis = self.need_gib(declared)
        if need <= 0:
            # No GPU (a prefetch job): nothing to reserve, so it never
            # counts against the card or keeps an idle card from a big job.
            self.clear()
            return {"gpu_mem_gib": 0.0, "basis": "no GPU (declared 0)",
                    "reserved_beside_gib": round(sum(
                        float(r["gpu_mem_gib"]) for r in live), 3)}
        document = {
            "schema": RESERVATION_SCHEMA, "gpu_uuid": self.gpu_uuid,
            "slot": self.index, "pid": os.getpid(), "job_id": claim.job_id,
            "attempt": claim.attempt, "gpu_mem_gib": need, "basis": basis,
            "reserved_beside_gib": round(sum(float(r["gpu_mem_gib"])
                                             for r in live), 3),
            "card_gib": self.card_gib, "margin_gib": self.margin_gib,
            "at_utc": iso_utc(time.time())}
        from woof.supervisor import atomic_write_json

        atomic_write_json(self.reservation_path, document)
        return document

    def clear(self) -> None:
        self.reservation_path.unlink(missing_ok=True)

    @contextlib.contextmanager
    def admission(self, stop: threading.Event, poll_s: float = 0.05):
        """The card's admission lock; yields False if ``stop`` came first."""

        import fcntl

        self.directory.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.admit_lock_path, os.O_RDWR | os.O_CREAT, 0o666)
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as error:
                    if error.errno not in (errno.EAGAIN, errno.EACCES,
                                           errno.EWOULDBLOCK):
                        raise
                if stop.wait(poll_s):
                    yield False
                    return
            try:
                yield True
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def auto_slots_per_gpu(card_gib: float, margin_gib: float, gpus: int,
                       cpus: int | None = None) -> int:
    """``--slots-per-gpu auto``: memory-, CPU- and :data:`AUTO_MAX_SLOTS_PER_GPU`-
    bound (module docstring)."""

    from woof.energy.cost import VRAM_INTERCEPT_GIB

    if cpus is None:
        try:
            cpus = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            cpus = os.cpu_count() or 1
    by_memory = int(math.floor((card_gib - margin_gib) / VRAM_INTERCEPT_GIB))
    by_cpu = max(1, int(cpus) // max(1, gpus))
    return max(1, min(AUTO_MAX_SLOTS_PER_GPU, by_memory, by_cpu))


def card_memory_gib(gpu, override: float | None) -> tuple[float, str]:
    """The admission budget of one card and where it came from."""

    if override is not None:
        return float(override), "--gpu-mem-gib"
    if gpu.driver_version == "fake":
        raise PackingError(
            "--fake-gpus have no memory to read; give --gpu-mem-gib with "
            "--slots-per-gpu above 1 or auto")
    from woof.core.preflight import device_physical_total_bytes

    total = device_physical_total_bytes(device_id=gpu.uuid)
    if total is None:
        raise PackingError(
            f"GPU {gpu.uuid}: NVML did not report its memory, so packing "
            "has no budget; give --gpu-mem-gib")
    return total / 1024 ** 3, "NVML memory.total"


class MpsDaemon:
    """The CUDA MPS control daemon a packed worker uses (module docstring)."""

    def __init__(self, root: Path | None, devices: list[str], *,
                 thread_percentage: int | None = None):
        external = os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
        self.external = bool(external)
        if external:
            self.pipe_dir = Path(external)
            self.log_dir = Path(os.environ.get("CUDA_MPS_LOG_DIRECTORY")
                                or self.pipe_dir)
        else:
            root = Path(root) if root is not None else (
                Path(tempfile.gettempdir()) / f"woof-mps-{os.getpid()}")
            self.pipe_dir = root / "pipe"
            self.log_dir = root / "log"
        self.devices = list(devices)
        self.thread_percentage = thread_percentage
        self.control: str | None = None
        self.owned = False
        self.started = False
        self.reused = False

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(self.environment())
        env.pop("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", None)
        if self.devices:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(self.devices)
        return env

    def environment(self) -> dict[str, str]:
        """What every slot (and so every runner) gets."""

        env = {"CUDA_MPS_PIPE_DIRECTORY": str(self.pipe_dir),
               "CUDA_MPS_LOG_DIRECTORY": str(self.log_dir)}
        if self.thread_percentage is not None:
            env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(
                self.thread_percentage)
        return env

    def _command(self, *argv: str, stdin: str | None = None
                 ) -> subprocess.CompletedProcess:
        assert self.control is not None
        return subprocess.run([self.control, *argv], input=stdin,
                              capture_output=True, text=True,
                              env=self._env(), timeout=MPS_COMMAND_TIMEOUT_S)

    def answers(self) -> bool:
        try:
            return self._command(stdin="get_server_list\n").returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def start(self) -> "MpsDaemon":
        self.control = shutil.which(MPS_CONTROL)
        if self.control is None:
            raise MPSUnavailable(
                f"--mps needs {MPS_CONTROL} on PATH (the NVIDIA driver's MPS "
                "control daemon; in a container, the NVIDIA container "
                "toolkit mounts it with the compute and utility "
                "capabilities); none was found")
        socket_path = str(self.pipe_dir / "control")
        if len(socket_path) > MPS_PIPE_PATH_MAX:
            raise MPSUnavailable(
                f"the MPS control socket {socket_path} is "
                f"{len(socket_path)} characters, over the "
                f"{MPS_PIPE_PATH_MAX} a UNIX socket path allows here; give "
                "a shorter --mps-dir")
        if self.external:
            if not self.answers():
                raise MPSUnavailable(
                    f"CUDA_MPS_PIPE_DIRECTORY={self.pipe_dir} is set but no "
                    "MPS control daemon answers there")
            self.reused = True
            return self
        self.pipe_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if (self.pipe_dir / "control").exists() and self.answers():
            self.reused = True
            return self
        try:
            started = self._command("-d")
        except (OSError, subprocess.TimeoutExpired) as error:
            raise MPSUnavailable(f"{MPS_CONTROL} -d failed: {error}") \
                from error
        if started.returncode != 0:
            raise MPSUnavailable(
                f"{MPS_CONTROL} -d exited {started.returncode}: "
                f"{(started.stderr or started.stdout).strip()[-500:]}")
        self.owned = self.started = True
        deadline = time.monotonic() + MPS_COMMAND_TIMEOUT_S
        while not self.answers():
            if time.monotonic() >= deadline:
                self.stop()
                raise MPSUnavailable(
                    f"the MPS control daemon started at {self.pipe_dir} "
                    "does not answer get_server_list")
            time.sleep(0.2)
        return self

    def stop(self) -> None:
        if not self.owned:
            return
        self.owned = False
        try:
            self._command(stdin="quit\n")
        except (OSError, subprocess.TimeoutExpired) as error:
            _note(f"MPS control daemon did not stop cleanly ({error})")

    def summary(self) -> dict[str, Any]:
        return {"pipe_dir": str(self.pipe_dir), "log_dir": str(self.log_dir),
                "control": self.control, "started_here": self.started,
                "reused_running_daemon": self.reused,
                "external": self.external,
                "active_thread_percentage": self.thread_percentage}


# --------------------------------------------------------------------------
# Slot
# --------------------------------------------------------------------------


@dataclass
class SlotConfig:
    queue: str
    slot: int
    worker_id: str
    workdir: str
    gpu_uuid: str | None = None
    runner_cmd: str = DEFAULT_RUNNER_TEMPLATE
    timeout_s: float | None = DEFAULT_TIMEOUT_S
    exit_when_empty: bool = False
    max_idle_s: float | None = None
    poll_s: float = DEFAULT_POLL_S
    term_grace_s: float = DEFAULT_TERM_GRACE_S
    parent_pid: int | None = None
    #: Whether this slot packs (module docstring): set by the parent from
    #: the request, so a card ``auto`` gives one slot still packs (the
    #: parent holds the card's worker lock either way).
    packed: bool = False
    #: Slots on this slot's GPU.
    slots_per_gpu: int = 1
    #: This slot's index among its GPU's slots.
    gpu_slot: int = 0
    #: Packed admission: the card's budget, the margin kept free and the
    #: price of a job that declares no memory (None: the whole card).
    card_mem_gib: float | None = None
    mem_margin_gib: float = DEFAULT_MEM_MARGIN_GIB
    default_job_mem_gib: float | None = None

    def document(self) -> dict[str, Any]:
        return {"schema": SLOT_CONFIG_SCHEMA, **asdict(self)}

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "SlotConfig":
        if document.get("schema") != SLOT_CONFIG_SCHEMA:
            raise WorkerError(f"slot config schema {document.get('schema')!r}")
        values = {key: value for key, value in document.items()
                  if key != "schema"}
        return cls(**values)


class Heartbeat(threading.Thread):
    """Renew one claim every ``interval`` s; spot completion elsewhere."""

    def __init__(self, queue: WorkQueue, claim: Claim, interval: float,
                 cancel: threading.Event):
        super().__init__(name=f"heartbeat-{claim.job_id}", daemon=True)
        self.queue = queue
        self.claim = claim
        self.interval = interval
        self.cancel = cancel
        self.superseded = False
        self.renewals = 0
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.wait(self.interval):
            if not self.claim.lost:
                try:
                    self.queue.renew(self.claim)
                    self.renewals += 1
                except LeaseLost:
                    _note(f"lease on {self.claim.job_id} was taken over; the "
                          "runner continues and the done marker decides")
                except Exception as error:  # transient store trouble
                    _note(f"lease renewal for {self.claim.job_id} failed "
                          f"({type(error).__name__}: {error}); retrying")
            try:
                if self.queue.is_done(self.claim.job_id):
                    self.superseded = True
                    _note(f"{self.claim.job_id} completed elsewhere; "
                          "cancelling this copy")
                    self.cancel.set()
                    return
            except Exception:
                pass

    def stop(self) -> None:
        self._halt.set()
        if self.is_alive():
            self.join()


@dataclass
class SlotResult:
    exit_code: int
    reason: str
    counts: dict[str, int] = field(default_factory=dict)


class Slot:
    """One queue loop bound to one GPU (or none)."""

    def __init__(self, config: SlotConfig, *, queue: WorkQueue | None = None,
                 runner: Runner | None = None,
                 stop: threading.Event | None = None):
        self.config = config
        self.queue = queue or WorkQueue.open(config.queue)
        self.runner = runner or SubprocessRunner(
            config.runner_cmd, term_grace_s=config.term_grace_s)
        self.stop = stop or threading.Event()
        self.stop_signal: int | None = None
        self._cancel: threading.Event | None = None
        self.counts: dict[str, int] = {}
        self.rng = random.Random(f"{config.worker_id}-{time.time_ns()}")
        self.owner = {"worker_id": config.worker_id,
                      "host": socket.gethostname(), "pid": os.getpid(),
                      "slot": config.slot, "gpu_uuid": config.gpu_uuid}
        self.ledger: CardLedger | None = None
        self._slot_lock = None
        if config.packed:
            if not config.gpu_uuid or config.card_mem_gib is None:
                raise PackingError(
                    f"slot {config.slot} packs but its configuration names "
                    "no GPU or no card_mem_gib")
            self.owner["gpu_slot"] = config.gpu_slot
            self.ledger = CardLedger(
                config.gpu_uuid, config.gpu_slot,
                card_gib=config.card_mem_gib,
                margin_gib=config.mem_margin_gib,
                default_job_gib=config.default_job_mem_gib)

    def request_stop(self, signum: int | None = None) -> None:
        self.stop_signal = signum
        self.stop.set()
        cancel = self._cancel
        if cancel is not None:
            cancel.set()

    def install_signal_handlers(self) -> None:
        def handler(signum, _frame):
            self.request_stop(signum)
        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)

    def run(self) -> SlotResult:
        from woof.supervisor import GPUAlreadyLockedError, GPUFileLock

        try:
            settings = self.queue.settings()
        except QueueError as error:
            return SlotResult(EXIT_REFUSED, f"{type(error).__name__}: {error}")
        lock = None
        if self.ledger is not None:
            # Packed: the parent holds the card's worker lock; this slot
            # holds its own index.  A runner a previous incarnation of this
            # slot left behind still holds it (it inherited the
            # descriptor), and its memory with it: wait for that runner.
            lock = GPUFileLock(self.config.gpu_uuid,
                               path=self.ledger.slot_lock_path,
                               run_id=self.config.worker_id)
            waited = False
            while True:
                try:
                    lock.acquire()
                    break
                except GPUAlreadyLockedError:
                    if not waited:
                        _note(f"slot {self.config.slot}: "
                              f"{self.ledger.slot_lock_path.name} is held (a "
                              "runner left by an earlier slot process?); "
                              "waiting for it")
                        waited = True
                    if self.stop.wait(self.config.poll_s):
                        return SlotResult(
                            EXIT_STOPPED if self.stop_signal != signal.SIGINT
                            else EXIT_INTERRUPTED, "stopped", self.counts)
            self._slot_lock = lock
            self.ledger.clear()
        elif self.config.gpu_uuid:
            lock = GPUFileLock(self.config.gpu_uuid,
                               path=worker_lock_path(self.config.gpu_uuid),
                               run_id=self.config.worker_id)
            try:
                lock.acquire()
            except GPUAlreadyLockedError:
                return SlotResult(
                    EXIT_REFUSED, f"GPU {self.config.gpu_uuid} already has an "
                                  "energy worker on this host")
        try:
            reason = self._loop(settings)
        finally:
            if self.ledger is not None:
                self.ledger.clear()
            if lock is not None:
                lock.release()
            self._slot_lock = None
        if reason == "error":
            return SlotResult(EXIT_FAILED, reason, self.counts)
        if self.stop_signal is not None:
            code = (EXIT_INTERRUPTED if self.stop_signal == signal.SIGINT
                    else EXIT_STOPPED)
            return SlotResult(code, reason, self.counts)
        return SlotResult(EXIT_OK, reason, self.counts)

    def _wait(self) -> None:
        self.stop.wait(self.config.poll_s * self.rng.uniform(0.75, 1.25))

    def _loop(self, settings) -> str:
        idle_since = time.monotonic()
        errors = 0
        failed_in_a_row = 0
        while not self.stop.is_set():
            if (self.config.parent_pid is not None
                    and os.getppid() != self.config.parent_pid):
                _note(f"slot {self.config.slot}: parent worker is gone; "
                      "stopping")
                return "parent-gone"
            if self.config.gpu_uuid and not self._gpu_free():
                _note(f"slot {self.config.slot}: GPU {self.config.gpu_uuid} "
                      "is held by another woof process; waiting")
                self._wait()
                continue
            try:
                claim = self._claim()
                errors = 0
            except Exception as error:
                errors += 1
                _note(f"slot {self.config.slot}: claim failed "
                      f"({type(error).__name__}: {error})")
                if errors >= MAX_CONSECUTIVE_QUEUE_ERRORS:
                    return "error"
                self._wait()
                continue
            if claim is None:
                if self.ledger is not None and self.queue.last_unadmitted:
                    # Work is waiting for room on this card: not idle.
                    idle_since = time.monotonic()
                try:
                    if (self.config.exit_when_empty
                            and self.queue.drained(reuse_last=True)):
                        return "drained"
                except Exception as error:
                    _note(f"slot {self.config.slot}: drain check failed "
                          f"({type(error).__name__}: {error})")
                if (self.config.max_idle_s is not None
                        and time.monotonic() - idle_since
                        >= self.config.max_idle_s):
                    return "idle"
                self._wait()
                continue
            if self.config.gpu_uuid and not self._gpu_free():
                # Taken while the claim listed the queue: hand it back
                # uncharged rather than let the runner's lock refusal
                # (exit 2) kill the job.
                try:
                    self._settle(lambda: self.queue.release(
                        claim, "GPU taken by another woof process before "
                               "start"))
                finally:
                    if self.ledger is not None:
                        self.ledger.clear()
                self._wait()
                continue
            try:
                outcome = self.execute(claim, settings)
            except Exception as error:
                _note(f"slot {self.config.slot}: worker error on "
                      f"{claim.job_id} ({type(error).__name__}: {error})")
                tail = traceback.format_exc()
                reason = f"worker error: {type(error).__name__}: {error}"
                try:
                    self._settle(lambda: self.queue.fail(
                        claim, exit_code=None, reason=reason,
                        error_tail=tail, log_key=None))
                except Exception as settle_error:
                    _note(f"slot {self.config.slot}: could not record the "
                          f"failure ({settle_error}); the lease will expire")
                outcome = "failed-worker-error"
            finally:
                if self.ledger is not None:
                    self.ledger.clear()
            self.counts[outcome] = self.counts.get(outcome, 0) + 1
            idle_since = time.monotonic()
            if outcome.startswith("failed"):
                failed_in_a_row += 1
                if failed_in_a_row >= MAX_CONSECUTIVE_JOB_FAILURES:
                    _note(f"slot {self.config.slot}: {failed_in_a_row} jobs "
                          "failed in a row; stopping this slot (check the "
                          "runner and the GPU)")
                    return "error"
            else:
                failed_in_a_row = 0
        return "stopped"

    def _gpu_free(self) -> bool:
        if self.ledger is not None:
            return gpu_is_free_shared(self.config.gpu_uuid)
        return gpu_is_free(self.config.gpu_uuid)

    def _claim(self) -> Claim | None:
        """One claim; packed, under the card's admission lock and only of
        a job that fits beside the live reservations."""

        ledger = self.ledger
        if ledger is None:
            return self.queue.claim(self.owner, rng=self.rng)
        # The listing (the expensive part on an object store) happens
        # outside the card's admission lock; a stale view is safe, because
        # taking a lease is a create-only or compare-and-swap write and a
        # finished job is checked again when it is taken.  Inside the lock:
        # the reservations, declared-memory reads (cached per job) and the
        # lease write -- the check and the reservation must be one step.
        view = self.queue.snapshot()
        with ledger.admission(self.stop) as entered:
            if not entered:
                return None
            live = ledger.live_reservations()
            claim = self.queue.claim(
                self.owner, rng=self.rng, view=view,
                admit=lambda _job_id, gib: ledger.fits(gib, live))
            if claim is None:
                return None
            try:
                declared = declared_gpu_mem_gib(claim.job)
            except JobSpecError:
                declared = None
            try:
                reservation = ledger.reserve(claim, declared, live)
            except Exception:
                # Without its reservation this job could be packed past
                # the card's budget: hand it back rather than run it.
                self._settle(lambda: self.queue.release(
                    claim, "could not record the slot's memory reservation"))
                raise
        _note(f"slot {self.config.slot}: reserved "
              f"{reservation['gpu_mem_gib']:g} GiB ({reservation['basis']}) "
              f"for {claim.job_id} beside "
              f"{reservation['reserved_beside_gib']:g} GiB on "
              f"{ledger.card_gib:g} GiB")
        return claim

    def _settle(self, action: Callable[[], Any]) -> Any:
        """Run a queue settle call through transient store errors."""

        for attempt in range(SETTLE_ATTEMPTS):
            try:
                return action()
            except QueueError:
                raise
            except Exception as error:
                if attempt == SETTLE_ATTEMPTS - 1:
                    raise
                wait = min(30.0, 2.0 ** attempt) * self.rng.uniform(0.5, 1.0)
                _note(f"slot {self.config.slot}: queue write failed "
                      f"({type(error).__name__}: {error}); retrying in "
                      f"{wait:.1f} s")
                time.sleep(wait)
        return None

    def execute(self, claim: Claim, settings) -> str:
        """Run one claimed job and settle it.  Returns the outcome name."""

        config = self.config
        attempt_dir = (Path(config.workdir) / "jobs" / claim.job_id
                       / f"attempt-{claim.attempt:03d}")
        if attempt_dir.exists():
            shutil.rmtree(attempt_dir)
        attempt_dir.mkdir(parents=True)
        job_path = attempt_dir / f"{claim.job_id}.json"
        job_path.write_bytes(claim.job_bytes)
        log_path = attempt_dir / "runner.log"
        result_path = attempt_dir / "result.json"
        cancel = threading.Event()
        self._cancel = cancel
        if self.stop.is_set():
            cancel.set()
        env = dict(os.environ)
        env.update({"WOOF_ENERGY_JOB_ID": claim.job_id,
                    "WOOF_ENERGY_JOB_ATTEMPT": str(claim.attempt),
                    "WOOF_ENERGY_JOB_PATH": str(job_path),
                    "WOOF_ENERGY_RESULT_PATH": str(result_path),
                    "WOOF_ENERGY_QUEUE": self.queue.url})
        pass_fds: tuple[int, ...] = ()
        if self.ledger is not None:
            from woof.supervisor import GPU_LOCK_ROOT_ENV

            # The runner's own `woof run` locks under this slot's root, so
            # sibling runners on the card do not refuse each other.
            self.ledger.runner_lock_root.mkdir(parents=True, exist_ok=True)
            env[GPU_LOCK_ROOT_ENV] = str(self.ledger.runner_lock_root)
            env["WOOF_ENERGY_GPU_SLOT"] = str(config.gpu_slot)
            stream = getattr(self._slot_lock, "_stream", None)
            if stream is not None and os.name == "posix":
                pass_fds = (stream.fileno(),)
        pgid_path = Path(config.workdir) / RUNNER_PGID_FILE
        request = RunnerRequest(
            claim.job_id, job_path, claim.attempt, attempt_dir, log_path,
            result_path, env, config.timeout_s, cancel, config.gpu_uuid,
            self.queue.url,
            on_start=lambda pgid: pgid_path.write_text(f"{pgid}\n"),
            pass_fds=pass_fds)
        _note(f"slot {config.slot}: running {claim.job_id} attempt "
              f"{claim.attempt}")
        heartbeat = Heartbeat(self.queue, claim, settings.heartbeat_s, cancel)
        heartbeat.start()
        started = time.time()
        try:
            outcome = self.runner(request)
        except Exception as error:
            outcome = RunnerOutcome(
                None, error=f"runner raised {type(error).__name__}: {error}\n"
                            + traceback.format_exc())
        finally:
            heartbeat.stop()
            self._cancel = None
            pgid_path.unlink(missing_ok=True)
        finished = time.time()
        log_bytes = log_path.read_bytes() if log_path.exists() else b""
        if outcome.error:
            log_bytes += f"\n# woof energy worker: {outcome.error}\n".encode()
        log_key = self._upload_log(claim, log_bytes)
        record = {"exit_code": outcome.exit_code,
                  "started_at_utc": iso_utc(started),
                  "finished_at_utc": iso_utc(finished),
                  "duration_s": round(finished - started, 3),
                  "log_key": log_key}
        if heartbeat.superseded:
            self._settle(lambda: self.queue.abandon(claim))
            return "superseded"
        if outcome.exit_code == 0 and not outcome.timed_out:
            result = None
            result_error = None
            if result_path.exists():
                try:
                    result = json.loads(result_path.read_text("utf-8"))
                except (OSError, UnicodeDecodeError,
                        json.JSONDecodeError) as error:
                    result_error = f"result.json unreadable: {error}"
            if result_error is None:
                won = self._settle(lambda: self.queue.complete(
                    claim, dict(record, result=result)))
                shutil.rmtree(attempt_dir, ignore_errors=True)
                _note(f"slot {config.slot}: {claim.job_id} "
                      f"{'completed' if won else 'finished a duplicate copy'}")
                return "completed" if won else "duplicate"
            outcome = RunnerOutcome(0, error=result_error)
        if self.stop.is_set():
            # Stopped (SIGTERM/SIGINT): whatever the runner did on the way
            # down -- killed by us, or by the same signal reaching its
            # cgroup first -- the job is handed back uncharged.
            name = (signal.Signals(self.stop_signal).name
                    if self.stop_signal else "stop")
            released = self._settle(lambda: self.queue.release(
                claim, f"worker stopped by {name} (preempted); runner exit "
                       f"{outcome.exit_code}"))
            _note(f"slot {config.slot}: released {claim.job_id} on {name}")
            return "released" if released else "released-lost"
        if outcome.timed_out:
            reason = f"timeout after {config.timeout_s:g} s"
        elif outcome.error:
            reason = outcome.error.splitlines()[0]
        elif outcome.exit_code is not None and outcome.exit_code < 0:
            reason = f"runner killed by {signal.Signals(-outcome.exit_code).name}"
        else:
            reason = f"runner exit {outcome.exit_code}"
        permanent = (not outcome.timed_out
                     and outcome.exit_code in PERMANENT_EXIT_CODES)
        state = self._settle(lambda: self.queue.fail(
            claim, exit_code=outcome.exit_code, reason=reason,
            error_tail=tail_text(log_bytes), log_key=log_key,
            permanent=permanent))
        _note(f"slot {config.slot}: {claim.job_id} failed ({reason}) -> {state}")
        return f"failed-{state}"

    def _upload_log(self, claim: Claim, data: bytes) -> str | None:
        if len(data) > LOG_UPLOAD_MAX_BYTES:
            data = (f"# woof energy worker: log truncated to its last "
                    f"{LOG_UPLOAD_MAX_BYTES} bytes\n").encode() \
                + data[-LOG_UPLOAD_MAX_BYTES:]
        claimed = int(claim.lease.get("claimed_at") or time.time())
        name = (f"attempt-{claim.attempt:03d}-{claimed}-"
                f"{self.config.worker_id}.log")
        try:
            return self.queue.put_log(claim.job_id, name, data)
        except Exception as error:
            _note(f"log upload for {claim.job_id} failed "
                  f"({type(error).__name__}: {error})")
            return None


def _slot_entry(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m woof.energy.worker")
    parser.add_argument("--slot-config", required=True)
    args = parser.parse_args(argv)
    path = Path(args.slot_config)
    config = SlotConfig.from_document(json.loads(path.read_text("utf-8")))
    try:
        slot = Slot(config)
    except QueueError as error:
        _note(f"slot {config.slot}: REFUSED: {type(error).__name__}: {error}")
        return EXIT_REFUSED
    slot.install_signal_handlers()
    result = slot.run()
    summary = {"schema": SLOT_SUMMARY_SCHEMA, "slot": config.slot,
               "worker_id": config.worker_id, "gpu_uuid": config.gpu_uuid,
               "pid": os.getpid(), "exit_code": result.exit_code,
               "reason": result.reason, "counts": result.counts}
    (path.parent / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n", "utf-8")
    if result.exit_code == EXIT_REFUSED:
        _note(f"slot {config.slot}: REFUSED: {result.reason}")
    return result.exit_code


# --------------------------------------------------------------------------
# Parent (``woof energy worker``)
# --------------------------------------------------------------------------


def _devices(args) -> list:
    from woof.local_gpu import NO_LOCAL_GPU_ENV, no_local_gpu

    if args.no_gpu:
        if args.gpus is not None:
            raise GPUSelectionError("--gpus selects GPUs; --no-gpu has none")
        return [None] * (args.slots or 1)
    if args.slots is not None:
        raise GPUSelectionError(
            "--slots applies to --no-gpu; GPU workers run one slot per GPU")
    if args.fake_gpus:
        inventory = fake_gpus(args.fake_gpus)
    else:
        if no_local_gpu():
            raise GPUSelectionError(
                f"{NO_LOCAL_GPU_ENV} is set, so this host's GPUs are not "
                "read; use --no-gpu (CPU runners) or --fake-gpus N (testing)")
        from woof.supervisor import query_gpus

        inventory = query_gpus()
    return list(select_gpus(args.gpus or "all", inventory))


def _slot_environment(gpu, slot: int, slot_dir: Path) -> dict[str, str]:
    from woof.local_gpu import NO_LOCAL_GPU_ENV

    scratch = slot_dir / "tmp"
    cache = slot_dir / "cache"
    scratch.mkdir(parents=True, exist_ok=True)
    if gpu is None:
        environment = dict(os.environ)
        environment.update({"TMP": str(scratch), "TEMP": str(scratch),
                            "TMPDIR": str(scratch),
                            "CUDA_VISIBLE_DEVICES": "-1",
                            NO_LOCAL_GPU_ENV: "1"})
        return environment
    from woof.multi_run import PlannedRun, RunSpec, child_environment

    spec = RunSpec(name=f"energy-worker-slot-{slot}",
                   device_selector=gpu.uuid, outdir=slot_dir,
                   scratch_dir=scratch, cache=cache, config=None,
                   module=None, arguments=(), inputs=())
    return child_environment(PlannedRun(spec, gpu))


@dataclass
class _SlotProcess:
    config: SlotConfig
    config_path: Path
    env: dict[str, str]
    process: subprocess.Popen | None = None
    restarts: int = 0
    exit_code: int | None = None
    pids: list[int] = field(default_factory=list)
    #: Descriptors the slot inherits (the card's worker lock when packed).
    pass_fds: tuple[int, ...] = ()
    #: Set when the slot was never started (its card was refused).
    refused: str | None = None

    def start(self) -> None:
        self.process = subprocess.Popen(
            [sys.executable, "-m", "woof.energy.worker", "--slot-config",
             str(self.config_path)],
            env=self.env, stdin=subprocess.DEVNULL, start_new_session=True,
            pass_fds=self.pass_fds)
        self.pids.append(self.process.pid)


@dataclass
class _Packing:
    """The parent's resolved packing request."""

    slots_per_gpu: int | str = 1
    mem_margin_gib: float = DEFAULT_MEM_MARGIN_GIB
    gpu_mem_gib: float | None = None
    default_job_mem_gib: float | None = None
    mps: bool = False
    mps_dir: str | None = None
    mps_active_thread_percentage: int | None = None

    @property
    def packed(self) -> bool:
        return self.slots_per_gpu != 1


def _packing(args, devices: list) -> _Packing:
    """Validate the packing flags against the devices; named refusals."""

    raw = getattr(args, "slots_per_gpu", 1)
    try:
        slots = parse_slots_per_gpu(str(raw)) if raw is not None else 1
    except ValueError as error:
        raise PackingError(str(error)) from None
    packing = _Packing(
        slots_per_gpu=slots,
        mem_margin_gib=getattr(args, "mem_margin_gib", None),
        gpu_mem_gib=getattr(args, "gpu_mem_gib", None),
        default_job_mem_gib=getattr(args, "default_job_mem_gib", None),
        mps=bool(getattr(args, "mps", False)),
        mps_dir=getattr(args, "mps_dir", None),
        mps_active_thread_percentage=getattr(
            args, "mps_active_thread_percentage", None))
    memory_flags = {"--mem-margin-gib": packing.mem_margin_gib,
                    "--gpu-mem-gib": packing.gpu_mem_gib,
                    "--default-job-mem-gib": packing.default_job_mem_gib}
    if not packing.packed:
        given = [flag for flag, value in memory_flags.items()
                 if value is not None]
        if given:
            raise PackingError(
                f"{', '.join(given)} price jobs on a shared card; they apply "
                "only with --slots-per-gpu above 1 or auto")
        if packing.mps:
            raise PackingError(
                "--mps lets several slots' kernels share a GPU; with one slot "
                "per GPU there is nothing to share (add --slots-per-gpu N or "
                "auto)")
    if packing.mps_dir is not None and not packing.mps:
        raise PackingError("--mps-dir applies only with --mps")
    if packing.mps_active_thread_percentage is not None:
        if not packing.mps:
            raise PackingError(
                "--mps-active-thread-percentage applies only with --mps")
        if not 1 <= packing.mps_active_thread_percentage <= 100:
            raise PackingError(
                "--mps-active-thread-percentage must be 1-100, not "
                f"{packing.mps_active_thread_percentage}")
    if not packing.packed:
        packing.mem_margin_gib = DEFAULT_MEM_MARGIN_GIB
        return packing
    if any(gpu is None for gpu in devices):
        raise PackingError(
            "--slots-per-gpu packs GPUs; --no-gpu runs CPU slots (use "
            "--slots for more of them)")
    if os.name != "posix":
        raise PackingError(
            "--slots-per-gpu above 1 needs POSIX flock and descriptor "
            "inheritance; run one slot per GPU on this platform")
    if packing.mem_margin_gib is None:
        packing.mem_margin_gib = DEFAULT_MEM_MARGIN_GIB
    for flag, value in memory_flags.items():
        if value is not None and (not math.isfinite(value) or value < 0
                                  or (value == 0 and flag != "--mem-margin-gib")):
            raise PackingError(f"{flag} {value!r} must be a positive number "
                               "of GiB")
    return packing


def main(args: argparse.Namespace) -> int:
    # A SIGTERM/SIGINT before the slots' handlers exist (the MPS daemon can
    # take seconds to answer) is remembered, not fatal, so a daemon this
    # worker started is always stopped.
    early: list[int] = []
    previous = {signum: signal.signal(signum,
                                      lambda signum, _f: early.append(signum))
                for signum in (signal.SIGTERM, signal.SIGINT)}
    try:
        return _main(args, early)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _main(args: argparse.Namespace, early: list[int]) -> int:
    mps: MpsDaemon | None = None
    try:
        args.queue = _queue_url(args)
        queue = WorkQueue.open(args.queue)
        queue.settings()
        SubprocessRunner(args.runner_cmd, term_grace_s=args.term_grace_s)
        if (args.runner_cmd == DEFAULT_RUNNER_TEMPLATE
                and not default_runner_available()):
            raise RunnerTemplateError(
                "the default runner `woof energy hindcast run-local` is not "
                "in this install; install a woof with the hindcast door or "
                "name a runner with --runner-cmd (a missing door exits 2, "
                "which would mark every job dead)")
        devices = _devices(args)
        packing = _packing(args, devices)
        cards: dict[str, dict[str, Any]] = {}
        if packing.packed:
            for gpu in devices:
                budget, basis = card_memory_gib(gpu, packing.gpu_mem_gib)
                if budget <= packing.mem_margin_gib:
                    raise PackingError(
                        f"GPU {gpu.uuid}: {budget:g} GiB leaves nothing "
                        f"beside the {packing.mem_margin_gib:g} GiB margin")
                count = (auto_slots_per_gpu(budget, packing.mem_margin_gib,
                                            len(devices))
                         if packing.slots_per_gpu == SLOTS_AUTO
                         else int(packing.slots_per_gpu))
                cards[gpu.uuid] = {"card_mem_gib": round(budget, 3),
                                   "card_mem_basis": basis, "slots": count,
                                   "mem_margin_gib": packing.mem_margin_gib,
                                   "default_job_mem_gib":
                                       packing.default_job_mem_gib}
        if packing.mps:
            mps = MpsDaemon(
                None if packing.mps_dir is None else Path(packing.mps_dir),
                [gpu.uuid for gpu in devices],
                thread_percentage=packing.mps_active_thread_percentage)
            mps.start()
    except (QueueError, WorkerError) as error:
        print(f"woof energy worker: REFUSED: {type(error).__name__}: {error}",
              file=sys.stderr)
        return EXIT_REFUSED
    except Exception as error:
        from woof.supervisor import GPUPreflightError

        if isinstance(error, GPUPreflightError):
            print(f"woof energy worker: REFUSED: {error}", file=sys.stderr)
            return EXIT_REFUSED
        raise
    try:
        return _run_slots(args, queue, devices, packing, cards, mps, early)
    finally:
        if mps is not None:
            mps.stop()


def _queue_url(args) -> str:
    positional = getattr(args, "queue", None)
    flag = getattr(args, "queue_flag", None)
    if positional and flag and positional != flag:
        raise WorkerError(f"the queue is named twice, differently: "
                          f"{positional!r} and --queue {flag!r}")
    url = positional or flag
    if not url:
        raise WorkerError("name the queue: woof energy worker QUEUE_URL (or "
                          "--queue QUEUE_URL)")
    return url


def _run_slots(args, queue: WorkQueue, devices: list, packing: _Packing,
               cards: dict[str, dict[str, Any]],
               mps: "MpsDaemon | None", early: list[int] | None = None) -> int:
    from woof.supervisor import GPUAlreadyLockedError, GPUFileLock

    host = socket.gethostname()
    run_id = f"{_short(host)}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
    root = Path(args.workdir) if args.workdir else (
        Path(tempfile.gettempdir()) / "woof-energy-worker" / run_id)
    root.mkdir(parents=True, exist_ok=True)
    slots: list[_SlotProcess] = []
    card_locks: list = []
    index = 0
    for gpu in devices:
        card = cards.get(gpu.uuid) if gpu is not None else None
        count = card["slots"] if card else 1
        pass_fds: tuple[int, ...] = ()
        refused = None
        if card is not None:
            # One worker owns a card: the lock a single-slot worker's slot
            # holds, held here and inherited by every slot of the card.
            lock = GPUFileLock(gpu.uuid, path=worker_lock_path(gpu.uuid),
                               run_id=run_id)
            try:
                lock.acquire()
                card_locks.append(lock)
                pass_fds = (lock._stream.fileno(),)
            except GPUAlreadyLockedError:
                refused = (f"GPU {gpu.uuid} already has an energy worker on "
                           "this host")
                _note(f"REFUSED: {refused}; its slots do not start")
        for gpu_slot in range(count):
            slot_dir = root / f"slot-{index}"
            slot_dir.mkdir(parents=True, exist_ok=True)
            config = SlotConfig(
                queue=args.queue, slot=index, worker_id=f"{run_id}-s{index}",
                workdir=str(slot_dir),
                gpu_uuid=None if gpu is None else gpu.uuid,
                runner_cmd=args.runner_cmd, timeout_s=args.timeout_s,
                exit_when_empty=args.exit_when_empty,
                max_idle_s=args.max_idle_s, poll_s=args.poll_s,
                term_grace_s=args.term_grace_s, parent_pid=os.getpid(),
                packed=card is not None,
                slots_per_gpu=count, gpu_slot=gpu_slot,
                card_mem_gib=card["card_mem_gib"] if card else None,
                mem_margin_gib=packing.mem_margin_gib,
                default_job_mem_gib=packing.default_job_mem_gib)
            config_path = slot_dir / "slot-config.json"
            config_path.write_text(
                json.dumps(config.document(), indent=2) + "\n", "utf-8")
            (slot_dir / "summary.json").unlink(missing_ok=True)
            env = _slot_environment(gpu, index, slot_dir)
            if mps is not None:
                env.update(mps.environment())
            slots.append(_SlotProcess(config, config_path, env,
                                      pass_fds=pass_fds, refused=refused))
            index += 1

    stopping: list[int] = list(early or [])

    def forward(signum, _frame):
        stopping.append(signum)
        kill = signal.SIGKILL if len(stopping) > 1 else signal.SIGTERM
        for slot in slots:
            if kill == signal.SIGKILL:
                _kill_runner(slot)
            if slot.process is not None and slot.process.poll() is None:
                try:
                    slot.process.send_signal(kill)
                except ProcessLookupError:
                    pass

    previous = {signum: signal.signal(signum, forward)
                for signum in (signal.SIGTERM, signal.SIGINT)}
    runnable = [slot for slot in slots if slot.refused is None]
    _note(f"{len(runnable)} slot(s) on {host} draining {queue.url} "
          f"(workdir {root})"
          + ("" if not cards else "; packed: " + ", ".join(
              f"{uuid_} {card['slots']} slot(s) on {card['card_mem_gib']:g} "
              f"GiB ({card['card_mem_basis']})"
              for uuid_, card in cards.items()))
          + ("" if mps is None else f"; CUDA MPS at {mps.pipe_dir}"))
    try:
        live = []
        for slot in runnable:
            if stopping:
                break  # a signal arrived during start-up
            slot.start()
            live.append(slot)
            if stopping:  # it arrived while this slot was being spawned
                slot.process.send_signal(signal.SIGTERM)
        while live:
            time.sleep(0.2)
            for slot in list(live):
                code = slot.process.poll()
                if code is None:
                    continue
                if (code not in (EXIT_OK, EXIT_REFUSED, EXIT_STOPPED,
                                 EXIT_INTERRUPTED)
                        and not stopping
                        and slot.restarts < MAX_SLOT_RESTARTS):
                    slot.restarts += 1
                    # Its lease is no longer renewed and, packed, its
                    # runner still holds the slot: stop that runner first.
                    _kill_runner(slot)
                    _note(f"slot {slot.config.slot} exited {code}; restart "
                          f"{slot.restarts}/{MAX_SLOT_RESTARTS}")
                    slot.start()
                    continue
                slot.exit_code = code
                live.remove(slot)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        for lock in card_locks:
            lock.release()
    rows = []
    for slot in slots:
        summary_path = Path(slot.config.workdir) / "summary.json"
        try:
            summary = json.loads(summary_path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            summary = {}
        if slot.refused is not None:
            slot.exit_code = EXIT_REFUSED
            summary = {"reason": slot.refused}
        row = {"slot": slot.config.slot,
               "worker_id": slot.config.worker_id,
               "gpu_uuid": slot.config.gpu_uuid,
               "exit_code": slot.exit_code, "restarts": slot.restarts,
               "pids": slot.pids, "reason": summary.get("reason"),
               "counts": summary.get("counts", {})}
        if slot.config.packed:
            row["gpu_slot"] = slot.config.gpu_slot
        rows.append(row)
    totals: dict[str, int] = {}
    for row in rows:
        for name, count in row["counts"].items():
            totals[name] = totals.get(name, 0) + count
    if stopping:
        code = (EXIT_INTERRUPTED if stopping[0] == signal.SIGINT
                else EXIT_STOPPED)
    elif any(row["exit_code"] == EXIT_REFUSED for row in rows) and not totals:
        code = EXIT_REFUSED
    elif any(row["exit_code"] not in (EXIT_OK, EXIT_REFUSED) for row in rows):
        code = EXIT_FAILED
    else:
        code = EXIT_OK
    document = {"schema": WORKER_SUMMARY_SCHEMA, "queue": queue.url,
                "host": host, "workdir": str(root), "exit_code": code,
                "totals": totals, "slots": rows}
    if packing.packed:
        document["packing"] = {
            "slots_per_gpu": packing.slots_per_gpu,
            "cards": cards,
            "admission": "sum(reserved) + need + margin <= card; an idle "
                         "card admits any one job",
            "undeclared_job_mem": (
                f"{packing.default_job_mem_gib:g} GiB "
                "(--default-job-mem-gib)"
                if packing.default_job_mem_gib is not None
                else "the whole card (runs alone)")}
    if mps is not None:
        document["mps"] = mps.summary()
    print(json.dumps(document, sort_keys=True, indent=2))
    return code


def _kill_runner(slot: "_SlotProcess") -> None:
    """SIGKILL the runner session a slot recorded (it outlives the slot)."""

    try:
        pgid = int((Path(slot.config.workdir) / RUNNER_PGID_FILE)
                   .read_text().strip())
        os.killpg(pgid, signal.SIGKILL)
    except (OSError, ValueError):
        pass


def _short(host: str) -> str:
    return re.sub(r"[^A-Za-z0-9-]", "-", host)[:40] or "host"


if __name__ == "__main__":
    sys.exit(_slot_entry(sys.argv[1:]))
