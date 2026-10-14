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
"""

from __future__ import annotations

import argparse
import json
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

from woof.energy.workqueue import (
    PERMANENT_EXIT_CODES, Claim, LeaseLost, QueueError, WorkQueue, iso_utc,
    tail_text)

WORKER_SUMMARY_SCHEMA = "woof-energy.worker-summary.v1"
SLOT_SUMMARY_SCHEMA = "woof-energy.worker-slot.v1"
SLOT_CONFIG_SCHEMA = "woof-energy.worker-slot-config.v1"

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
                    cwd=request.workdir, start_new_session=True)
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
        if self.config.gpu_uuid:
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
            if lock is not None:
                lock.release()
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
            if self.config.gpu_uuid and not gpu_is_free(self.config.gpu_uuid):
                _note(f"slot {self.config.slot}: GPU {self.config.gpu_uuid} "
                      "is held by another woof process; waiting")
                self._wait()
                continue
            try:
                claim = self.queue.claim(self.owner, rng=self.rng)
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
            if self.config.gpu_uuid and not gpu_is_free(self.config.gpu_uuid):
                # Taken while the claim listed the queue: hand it back
                # uncharged rather than let the runner's lock refusal
                # (exit 2) kill the job.
                self._settle(lambda: self.queue.release(
                    claim, "GPU taken by another woof process before start"))
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
        pgid_path = Path(config.workdir) / RUNNER_PGID_FILE
        request = RunnerRequest(
            claim.job_id, job_path, claim.attempt, attempt_dir, log_path,
            result_path, env, config.timeout_s, cancel, config.gpu_uuid,
            self.queue.url,
            on_start=lambda pgid: pgid_path.write_text(f"{pgid}\n"))
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

    def start(self) -> None:
        self.process = subprocess.Popen(
            [sys.executable, "-m", "woof.energy.worker", "--slot-config",
             str(self.config_path)],
            env=self.env, stdin=subprocess.DEVNULL, start_new_session=True)
        self.pids.append(self.process.pid)


def main(args: argparse.Namespace) -> int:
    try:
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
    host = socket.gethostname()
    run_id = f"{_short(host)}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
    root = Path(args.workdir) if args.workdir else (
        Path(tempfile.gettempdir()) / "woof-energy-worker" / run_id)
    root.mkdir(parents=True, exist_ok=True)
    slots: list[_SlotProcess] = []
    for index, gpu in enumerate(devices):
        slot_dir = root / f"slot-{index}"
        slot_dir.mkdir(parents=True, exist_ok=True)
        config = SlotConfig(
            queue=args.queue, slot=index, worker_id=f"{run_id}-s{index}",
            workdir=str(slot_dir), gpu_uuid=None if gpu is None else gpu.uuid,
            runner_cmd=args.runner_cmd, timeout_s=args.timeout_s,
            exit_when_empty=args.exit_when_empty, max_idle_s=args.max_idle_s,
            poll_s=args.poll_s, term_grace_s=args.term_grace_s,
            parent_pid=os.getpid())
        config_path = slot_dir / "slot-config.json"
        config_path.write_text(json.dumps(config.document(), indent=2) + "\n",
                               "utf-8")
        (slot_dir / "summary.json").unlink(missing_ok=True)
        slots.append(_SlotProcess(config, config_path,
                                  _slot_environment(gpu, index, slot_dir)))

    stopping: list[int] = []

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
    _note(f"{len(slots)} slot(s) on {host} draining {queue.url} "
          f"(workdir {root})")
    try:
        live = []
        for slot in slots:
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
                    _note(f"slot {slot.config.slot} exited {code}; restart "
                          f"{slot.restarts}/{MAX_SLOT_RESTARTS}")
                    slot.start()
                    continue
                slot.exit_code = code
                live.remove(slot)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    rows = []
    for slot in slots:
        summary_path = Path(slot.config.workdir) / "summary.json"
        try:
            summary = json.loads(summary_path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            summary = {}
        rows.append({"slot": slot.config.slot,
                     "worker_id": slot.config.worker_id,
                     "gpu_uuid": slot.config.gpu_uuid,
                     "exit_code": slot.exit_code, "restarts": slot.restarts,
                     "pids": slot.pids, "reason": summary.get("reason"),
                     "counts": summary.get("counts", {})})
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
    print(json.dumps({"schema": WORKER_SUMMARY_SCHEMA, "queue": queue.url,
                      "host": host, "workdir": str(root), "exit_code": code,
                      "totals": totals, "slots": rows},
                     sort_keys=True, indent=2))
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
