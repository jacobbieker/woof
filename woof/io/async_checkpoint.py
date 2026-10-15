"""Asynchronous tree checkpoints: snapshot now, write in the background.

A checkpoint used to stop the integration for as long as it took to copy
the whole domain off the card (pageable, blocking), hash it, ``np.savez``
it, ``fsync`` it and rename it -- a median 386 s per hourly checkpoint of a
294 x 506 x 256 domain on a FUSE volume, with the GPU idle throughout.
This module splits the checkpoint at the one place it can be split:

1. ON THE MODEL THREAD, between two steps (unchanged in what it decides):
   every refusal, the whole header and the member inventory
   (``woof.io.restart._prepare_restart``), then a SNAPSHOT of every device
   member into page-locked host memory on a side stream.  The model's own
   stream waits on that copy before its next kernel (``wait_event``), so
   the snapshot is the state at the checkpoint instant byte for byte and
   the host thread returns as soon as the copies are queued.
2. ON A WRITER THREAD: wait for the copy, write each member through the
   same ``np.savez`` + ``fsync`` + ``os.replace`` + directory ``fsync`` the
   synchronous writer uses (children first, the root -- the set's commit
   marker -- last), after every history frame the checkpoint is ordered
   after has become durable.
3. BACK ON THE MODEL THREAD, at the next period commit (or the next
   checkpoint, or the end of the run): PUBLISH -- ``model._last_checkpoint``,
   retirement of superseded sets, the route's own bookkeeping (progress
   ``last_checkpoint``, the step log).  A checkpoint is a resume point
   only after it is durable; a failed write publishes nothing and fails
   the run, as a failed synchronous write did.

PINNED contract (tests/test_async_checkpoints.py):

* AT MOST ONE checkpoint is in flight.  A checkpoint that comes due while
  the previous one is still writing waits for it -- and says so, on stdout
  and in the receipt -- before taking its snapshot.
* The run DRAINS on every exit -- success, failure, signal: the in-flight
  checkpoint is finished and published (its snapshot predates the
  failure), or its partial members are removed.  No half checkpoint is
  ever published; a write that fails leaves the previous resume point.
* The snapshot is a copy; the state is never modified, so the numerics are
  bit-identical to the synchronous mode, and so are the archives (same
  members, same order, same serializer; only ``created`` differs).
* History ORDERING, not a drain: the synchronous mode waits for every
  queued wrfout frame (and its read-back identity) before writing.  Here
  the checkpoint records a durability barrier over the frames already
  submitted -- everything at or before its instant -- and its root member
  is renamed into place only after those frames are durable, so a resume
  never skips or repeats a frame.  A writer set that cannot offer the
  barrier (one with a live simulated radar) is drained on the model
  thread instead, as before.
* NAMED FALLBACKS, never silent.  The checkpoint is written synchronously,
  with the reason in the receipt and on stdout, when the mode is off
  (``[output] async_checkpoints = false`` or ``WOOF_ASYNC_CHECKPOINTS=0``),
  a domain streams from a host store (its store is the snapshot already),
  the sealed forcing extension is in force (it rebuilds state first), the
  state spans devices, host RAM cannot hold the snapshot beside the run,
  or page-locked memory cannot be had.
* The snapshot arena is allocated ONCE per run leg, sized from the state
  (grown, never shrunk, if a later checkpoint is larger), page-locked
  (``cupy.cuda.PinnedMemory``, released when the writer closes -- never
  parked in CuPy's pinned pool) and never reused while a write that points
  into it is in flight (at most one is).

The run receipt carries ``checkpoints`` (:data:`RECEIPT_SCHEMA`): which
mode was requested and from where, which mode each checkpoint actually ran
in, and its model-loop stall.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

#: Operator switch.  Set, it wins over the config key: ``0``/``false``/
#: ``off``/``no``/``sync`` writes every checkpoint synchronously,
#: ``1``/``true``/``on``/``yes``/``async`` asks for the asynchronous
#: writer.  Anything else is refused by name.
ASYNC_CHECKPOINTS_ENV = "WOOF_ASYNC_CHECKPOINTS"

RECEIPT_SCHEMA = "woof.checkpoint-writes/v1"

_TRUE = frozenset({"1", "true", "on", "yes", "async"})
_FALSE = frozenset({"0", "false", "off", "no", "sync"})

#: The named reasons a checkpoint is written synchronously.
FALLBACK_REASONS = {
    "disabled": "asynchronous checkpoints are switched off for this run",
    "streamed-domain": (
        "a domain streams from a pinned host store; its store writer is "
        "already copy-free and runs synchronously"),
    "sealed-forcing-extension": (
        "the sealed forcing extension refreshes streamed state before it "
        "writes and is written synchronously"),
    "multi-device-state": (
        "the state spans more than one CUDA device; the snapshot stream "
        "serves one"),
    "insufficient-host-ram": (
        "host RAM cannot hold the checkpoint snapshot beside the run"),
    "pinned-allocation-failed": (
        "page-locked host memory for the checkpoint snapshot could not be "
        "allocated"),
}

#: Host RAM the snapshot must leave free: the larger of these two.
_RAM_RESERVE_BYTES = 2 << 30
_RAM_RESERVE_FRACTION = 0.05
_ALIGN = 4096
_POLL_SECONDS = 0.5


class CheckpointWriteError(RuntimeError):
    """An asynchronous checkpoint could not be made durable."""


def resolve_checkpoint_mode(experiment=None, *, environ=None) -> dict:
    """``{"requested": "async"|"sync", "source": ...}`` for one run.

    The environment variable wins when it is set (it is the operator's
    switch for a run already configured); the experiment's tree-wide
    ``[output] async_checkpoints`` next; asynchronous is the default.
    """
    environ = os.environ if environ is None else environ
    raw = environ.get(ASYNC_CHECKPOINTS_ENV)
    if raw is not None and raw.strip():
        word = raw.strip().lower()
        if word in _TRUE:
            return {"requested": "async", "source": f"env {ASYNC_CHECKPOINTS_ENV}"}
        if word in _FALSE:
            return {"requested": "sync", "source": f"env {ASYNC_CHECKPOINTS_ENV}"}
        raise ValueError(
            f"{ASYNC_CHECKPOINTS_ENV}={raw!r} is not a switch; use 1/true/on "
            "for asynchronous checkpoints or 0/false/off for synchronous ones")
    output = getattr(experiment, "output", None)
    value = getattr(output, "async_checkpoints", None)
    if value is not None:
        return {"requested": "async" if bool(value) else "sync",
                "source": "[output] async_checkpoints"}
    return {"requested": "async", "source": "default"}


def host_ram_available() -> tuple[int, int | None] | None:
    """``(available, total)`` host RAM in bytes, or ``None`` where unknown.

    ``available`` is :func:`woof.core.preflight.host_available_bytes`:
    MemAvailable capped by the memory cgroup this process runs in, so a
    container's limit is honoured and not the host's free RAM.
    """
    from woof.core.preflight import (_host_total_bytes_or_none,
                                     host_available_bytes)

    available = host_available_bytes()
    if available is None:
        return None
    return int(available), _host_total_bytes_or_none()


def _align(nbytes: int) -> int:
    return (int(nbytes) + _ALIGN - 1) // _ALIGN * _ALIGN


class _Job:
    """One checkpoint between its snapshot and its publication."""

    def __init__(self, *, plans, payloads, directory, root_path, valid_time,
                 event, device_refs, barrier, on_published, row):
        self.plans = plans
        self.payloads = payloads
        self.directory = Path(directory)
        self.root_path = Path(root_path)
        self.valid_time = valid_time
        self.event = event
        self.device_refs = device_refs
        self.barrier = barrier
        self.on_published = on_published
        self.row = row
        self.done = threading.Event()
        self.error: BaseException | None = None
        self.written: list[Path] = []
        self.submitted = time.perf_counter()
        self.finished = None
        self.thread: threading.Thread | None = None
        #: Set when the model thread stops waiting for this write (an
        #: interrupt during the drain); the writer then publishes nothing
        #: more and removes what it wrote.
        self.cancelled = threading.Event()

    def refuse_if_cancelled(self) -> None:
        if self.cancelled.is_set():
            raise InterruptedError(
                "the checkpoint write was abandoned by an interrupt")


class CheckpointWriter:
    """The run's checkpoint writer: asynchronous, or synchronous by name.

    One per :func:`woof.core.model.execute_experiment` call that writes
    checkpoints, published on the model as ``_checkpoint_writer`` so
    :func:`woof.io.restart.write_tree_restart` finds it, and closed (drained)
    by the executor on every exit.  ``receipt`` is the run's
    :data:`RECEIPT_SCHEMA` dict, shared across the legs of one run.
    """

    def __init__(self, model, mode: dict, *, receipt: dict | None = None):
        self.model = model
        self.mode = dict(mode)
        self.asynchronous = self.mode.get("requested") == "async"
        if receipt is None:
            receipt = {"schema": RECEIPT_SCHEMA,
                       "requested": self.mode.get("requested"),
                       "source": self.mode.get("source"),
                       "ran": None, "fallbacks": {}, "in_flight_waits": 0,
                       "in_flight_wait_seconds": 0.0,
                       "snapshot_arena_bytes": 0, "snapshot_arena_pinned": None,
                       "max_loop_stall_seconds": None, "checkpoints": []}
        self.receipt = receipt
        from woof.io.restart import hold_staging_buffer
        # A synchronous checkpoint of this leg reuses one staging buffer
        # (woof.io.restart._staging_buffer) until close().
        hold_staging_buffer()
        self._arena = None
        self._stream = None
        self._job: _Job | None = None
        self._progress = None
        self._op_rows: list[dict] = []
        self.closed = False

    # ------------------------------------------------------------------
    # decisions taken on the model thread before anything is written
    # ------------------------------------------------------------------

    def refusal(self, nodes, *, sealed_forcing_extension: bool) -> str | None:
        """The named reason this checkpoint cannot be asynchronous, if any."""
        if not self.asynchronous:
            return "disabled"
        if sealed_forcing_extension:
            return "sealed-forcing-extension"
        if any(getattr(node.state, "_streamed_domain", None) is not None
               for node in nodes):
            return "streamed-domain"
        return None

    def order_history(self, io_manager):
        """The durability barrier this checkpoint is ordered after.

        ``None`` when there is no history writer.  A writer set that cannot
        offer a barrier is drained here, on the model thread, exactly as
        the synchronous mode does before every checkpoint.
        """
        if io_manager is None:
            return None
        make = getattr(io_manager, "durability_barrier", None)
        barrier = None if make is None else make()
        if barrier is None:
            io_manager.drain()
            status = getattr(self.model, "_runtime_status", None)
            if status is not None:
                status.pending_d2h = int(io_manager.pending)
        return barrier

    def synchronous(self, reason: str, *, io_manager) -> None:
        """Prepare for a synchronous write: nothing in flight, history drained."""
        self.wait_idle(beat=False)
        if io_manager is not None:
            io_manager.drain()
            status = getattr(self.model, "_runtime_status", None)
            if status is not None:
                status.pending_d2h = int(io_manager.pending)
        if reason != "disabled":
            print(f"checkpoint: written synchronously ({reason}: "
                  f"{FALLBACK_REASONS.get(reason, reason)})", flush=True)

    def record_synchronous(self, *, valid_time, reason: str, path,
                           model_thread_seconds: float) -> dict:
        row = {"valid_time": _iso(valid_time), "mode": "sync",
               "reason": reason, "status": "published",
               "path": None if path is None else str(path),
               "model_thread_seconds": float(model_thread_seconds),
               "wall_seconds": float(model_thread_seconds)}
        self._add_row(row)
        if reason != "disabled":
            fallbacks = self.receipt["fallbacks"]
            fallbacks[reason] = int(fallbacks.get(reason, 0)) + 1
        return row

    def wrote_this_op(self) -> bool:
        """Whether a checkpoint was written since the last loop-stall record."""
        return bool(self._op_rows)

    def record_loop_stall(self, seconds: float) -> None:
        """The whole restart op's wall time on the model loop (drain included)."""
        for row in self._op_rows:
            row["loop_stall_seconds"] = float(seconds)
        if self._op_rows:
            worst = self.receipt.get("max_loop_stall_seconds")
            self.receipt["max_loop_stall_seconds"] = max(
                float(seconds), 0.0 if worst is None else float(worst))
        self._op_rows = []

    def _add_row(self, row: dict) -> None:
        self.receipt["checkpoints"].append(row)
        self._op_rows.append(row)
        modes = {r["mode"] for r in self.receipt["checkpoints"]}
        self.receipt["ran"] = (modes.pop() if len(modes) == 1 else "mixed")

    # ------------------------------------------------------------------
    # the asynchronous path
    # ------------------------------------------------------------------

    def submit(self, plans, *, directory, root_path, valid_time, barrier,
               on_published=None, progress_callback=None,
               model_thread_started=None) -> str | None:
        """Snapshot ``plans`` and hand them to a writer thread.

        Returns ``None`` when the checkpoint is in flight, or the named
        reason it must be written synchronously instead (nothing has been
        copied or written in that case).  ``plans`` are in write order:
        children first, the root -- the set's commit marker -- last.
        """
        from woof.io.restart import _is_device_array

        started = (time.perf_counter() if model_thread_started is None
                   else model_thread_started)
        if progress_callback is not None:
            self._progress = progress_callback
        # Inside the route's own ``writing:checkpoint`` beat already.
        waited = self.wait_idle(next_valid_time=valid_time, beat=False)

        device, host, device_bytes, host_bytes = [], [], 0, 0
        for index, plan in enumerate(plans):
            for key, value in plan.members:
                if _is_device_array(value):
                    device.append((index, key, value))
                    device_bytes += _align(value.nbytes)
                else:
                    host.append((index, key, value))
                    host_bytes += int(getattr(value, "nbytes", 0) or 0)
        if device:
            import cupy as cp
            current = cp.cuda.Device().id
            if any(int(value.device.id) != int(current)
                   for _i, _k, value in device):
                return "multi-device-state"
        need = device_bytes + host_bytes
        if self._arena is None or self._arena.nbytes < device_bytes or (
                self._arena.pinned != bool(device)):
            reason = self._allocate_arena(device_bytes, need,
                                          pinned=bool(device))
            if reason is not None:
                return reason

        payloads = [{} for _ in plans]
        device_refs = []
        event = None
        if device:
            import cupy as cp
            if self._stream is None:
                self._stream = cp.cuda.Stream(non_blocking=True)
            producer = cp.cuda.get_current_stream()
            try:
                ready = cp.cuda.Event()
                ready.record(producer)
                self._stream.wait_event(ready)
                offset = 0
                with self._stream:
                    for index, key, value in device:
                        view = self._arena.view(value.shape, value.dtype,
                                                offset)
                        offset += _align(value.nbytes)
                        source = (value if value.flags.c_contiguous
                                  else cp.ascontiguousarray(value))
                        source.get(out=view, stream=self._stream,
                                   blocking=False)
                        device_refs.append(source)
                        payloads[index][key] = view
                    event = cp.cuda.Event()
                    event.record(self._stream)
                # The model's next kernel waits for the snapshot; the host
                # thread does not.
                producer.wait_event(event)
            finally:
                producer.use()
        for index, key, value in host:
            # Host members are copied now: the state may change them in
            # place.  Fortran-only arrays keep their order, so np.savez
            # writes the same header and bytes it writes for the original.
            order = ("F" if value.flags.f_contiguous
                     and not value.flags.c_contiguous else "C")
            payloads[index][key] = np.array(value, copy=True, order=order)
        # np.savez writes members in payload order: the plan's (sorted) order.
        ordered = []
        for plan, payload in zip(plans, payloads):
            from woof.io.restart import _HEADER_KEY, _header_member
            body = {_HEADER_KEY: _header_member(plan.header)}
            for key, _value in plan.members:
                body[key] = payload[key]
            ordered.append(body)

        row = {"valid_time": _iso(valid_time), "mode": "async",
               "reason": None, "status": "in-flight",
               "path": str(root_path), "snapshot_bytes": int(need),
               "waited_for_previous_seconds": float(waited)}
        job = _Job(plans=tuple(plans), payloads=ordered, directory=directory,
                   root_path=root_path, valid_time=valid_time, event=event,
                   device_refs=device_refs, barrier=barrier,
                   on_published=on_published, row=row)
        job.thread = threading.Thread(
            target=self._write, args=(job,), daemon=True,
            name=f"woof-checkpoint-{Path(root_path).stem}")
        job.thread.start()
        self._job = job
        row["model_thread_seconds"] = time.perf_counter() - started
        self._add_row(row)
        return None

    def _allocate_arena(self, device_bytes: int, need: int, *,
                        pinned: bool) -> str | None:
        previous = self._arena
        self._arena = None
        if previous is not None:
            previous.release()
        ram = host_ram_available()
        if ram is not None:
            available, total = ram
            reserve = max(_RAM_RESERVE_BYTES,
                          int(_RAM_RESERVE_FRACTION * (total or 0)))
            if need > available - reserve:
                print(f"checkpoint: {FALLBACK_REASONS['insufficient-host-ram']}"
                      f" (snapshot {need / 2**30:.2f} GiB, available "
                      f"{available / 2**30:.2f} GiB, reserve "
                      f"{reserve / 2**30:.2f} GiB)", flush=True)
                return "insufficient-host-ram"
        from woof.io.restart import HostBuffer
        try:
            # PINNED: one arena per leg, page-locked when it backs device
            # copies, freed at close(); never reused while a write reading
            # it is in flight (at most one is).
            self._arena = HostBuffer(max(int(device_bytes), _ALIGN),
                                     pinned=pinned)
        except Exception as exc:          # noqa: BLE001 - named fallback
            print(f"checkpoint: {FALLBACK_REASONS['pinned-allocation-failed']}"
                  f" ({device_bytes / 2**30:.2f} GiB: "
                  f"{type(exc).__name__}: {exc})", flush=True)
            return "pinned-allocation-failed"
        self.receipt["snapshot_arena_bytes"] = int(self._arena.nbytes)
        self.receipt["snapshot_arena_pinned"] = bool(self._arena.pinned)
        return None

    def _write(self, job: _Job) -> None:
        """Writer thread: make every member durable, the root last."""
        from woof.io.restart import _publish_npz

        try:
            if job.event is not None:
                job.event.synchronize()
            # The snapshot has landed: no copy reads the device arrays any
            # more, so their references (and the contiguous temporaries)
            # can go from any thread.
            job.device_refs = ()
            last = len(job.plans) - 1
            for index, (plan, body) in enumerate(zip(job.plans, job.payloads)):
                job.refuse_if_cancelled()
                if index == last and job.barrier is not None:
                    ordered = time.perf_counter()
                    job.barrier.wait()
                    job.row["history_order_wait_seconds"] = (
                        time.perf_counter() - ordered)
                _publish_npz(plan.path, body, unlink_published_on_failure=True,
                             before_replace=job.refuse_if_cancelled)
                job.written.append(plan.path)
        except BaseException as exc:      # noqa: BLE001 - reported on publish
            # The member names are unique to this generation, so removing
            # them can never touch a set an older root marker references.
            for path in job.written:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            job.error = exc
        finally:
            job.payloads = None
            job.finished = time.perf_counter()
            job.done.set()

    def wait_idle(self, *, next_valid_time=None, phase="checkpoint-drain",
                  beat: bool = True) -> float:
        """Wait for the in-flight checkpoint and publish it; seconds waited.

        ``beat`` publishes a ``writing:<phase>`` record for the wait, sized
        by the checkpoint's bytes, for a caller not already inside one.
        """
        job = self._job
        if job is None:
            return 0.0
        waited = 0.0
        if not job.done.is_set():
            started = time.perf_counter()
            what = ("the next checkpoint" if next_valid_time is None
                    else f"the {_iso(next_valid_time)} checkpoint")
            print(f"checkpoint: the {_iso(job.valid_time)} checkpoint is "
                  f"still being written; {what} waits for it (at most one "
                  "checkpoint is in flight)", flush=True)
            from woof.supervisor import writing_progress
            remaining = sum(plan.nbytes for plan in job.plans)
            with (writing_progress(self._progress, phase,
                                   work_bytes=remaining)
                  if beat else nullcontext()):
                while not job.done.wait(_POLL_SECONDS):
                    if job.thread is not None and not job.thread.is_alive():
                        break
            waited = time.perf_counter() - started
            print(f"checkpoint: waited {waited:.2f} s for the "
                  f"{_iso(job.valid_time)} checkpoint to become durable",
                  flush=True)
            if next_valid_time is not None:
                self.receipt["in_flight_waits"] = int(
                    self.receipt["in_flight_waits"]) + 1
                self.receipt["in_flight_wait_seconds"] = float(
                    self.receipt["in_flight_wait_seconds"]) + waited
        self.poll()
        return waited

    def poll(self) -> Path | None:
        """Publish the in-flight checkpoint if it has become durable.

        Model thread only.  Raises :class:`CheckpointWriteError` for a
        write that failed: nothing of it was published.
        """
        job = self._job
        if job is None or not job.done.is_set():
            return None
        self._job = None
        row = job.row
        row["background_write_seconds"] = job.finished - job.submitted
        if job.error is not None:
            row["status"] = "failed"
            row["error"] = f"{type(job.error).__name__}: {job.error}"
            raise CheckpointWriteError(
                f"the asynchronous checkpoint at {_iso(job.valid_time)} could "
                f"not be made durable ({row['error']}); nothing of it was "
                "published and the previous checkpoint remains the resume "
                "point") from job.error
        self.model._last_checkpoint = job.root_path
        # Only after the whole new set is durable: the sets it supersedes
        # go when the run's retention says so (woof.resume), never before.
        from woof.resume import retire_superseded_checkpoints
        retire_superseded_checkpoints(job.directory)
        row["status"] = "published"
        row["wall_seconds"] = time.perf_counter() - job.submitted
        if job.on_published is not None:
            job.on_published(job.root_path, dict(row))
        return job.root_path

    def wait_published(self) -> None:
        """Block until nothing is in flight and everything is published."""
        self.wait_idle(beat=False)

    def close(self, error: BaseException | None = None) -> None:
        """Drain: finish and publish the in-flight checkpoint, then release.

        With ``error`` (the run is already failing) a checkpoint failure is
        attached to it as a note instead of replacing it.
        """
        try:
            self.wait_idle()
        except Exception as exc:
            if error is None:
                raise
            note = getattr(error, "add_note", None)
            if note is not None:
                note(f"the in-flight checkpoint also failed: "
                     f"{type(exc).__name__}: {exc}")
        except BaseException:
            # An interrupt while waiting: stop waiting, but leave nothing
            # behind that could still be published -- the writer thread
            # checks this before every member and removes what it wrote.
            job = self._job
            if job is not None:
                job.cancelled.set()
            raise
        finally:
            from woof.io.restart import unhold_staging_buffer
            if self._arena is not None:
                # Views a write still holds keep their memory alive.
                self._arena.release()
            self._arena = None
            self._stream = None
            if not self.closed:
                unhold_staging_buffer()
            self.closed = True
            if getattr(self.model, "_checkpoint_writer", None) is self:
                self.model._checkpoint_writer = None


def open_checkpoint_writer(model, *, experiment=None) -> CheckpointWriter:
    """A writer for one executor leg, sharing the run's receipt."""
    mode = resolve_checkpoint_mode(experiment)
    receipt = getattr(model, "_checkpoint_receipt", None)
    writer = CheckpointWriter(model, mode, receipt=receipt)
    model._checkpoint_receipt = writer.receipt
    model._checkpoint_writer = writer
    if mode["requested"] == "sync":
        print(f"checkpoint: synchronous checkpoints ({mode['source']})",
              flush=True)
    return writer


def _iso(value) -> str | None:
    if value is None:
        return None
    iso = getattr(value, "isoformat", None)
    return iso() if iso is not None else str(value)


__all__ = [
    "ASYNC_CHECKPOINTS_ENV", "CheckpointWriteError", "CheckpointWriter",
    "FALLBACK_REASONS", "RECEIPT_SCHEMA", "host_ram_available",
    "open_checkpoint_writer", "resolve_checkpoint_mode",
]
