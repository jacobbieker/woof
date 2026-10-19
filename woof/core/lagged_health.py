"""Lagged per-step health checks: the CPU never waits on the step it just queued.

THE COST THIS REMOVES.  ``execute_experiment`` gates every resident model
step on :func:`woof.core.dycore.stability_report` (NaN in u/w/theta', a
non-finite vertical Courant number), and that report ends in one blocking
device-to-host copy.  A blocking copy after every step means the CPU cannot
queue step k+1 until the GPU has finished step k: the launch queue drains
to empty once per step, and on a shared card each drain hands the device to
the other contexts for a time slice.  Small, launch-bound domains (the
wind-farm hindcast leaves) pay the most.  The same is true of the scheme
status words physics reads back inside the step (microphysics surface
diagnostics, YSU, KF, RRTMGP; :mod:`woof.core.health_ledger`) and of the
every-fourth-step full-state validator.

WHAT LAGGED MEANS.  The reductions still run on every step, on the device,
in the same stream position; only the READ moves.  After step k the compact
records (the stability record and the health ledger's status words) are
copied asynchronously into page-locked host memory and an event is
recorded.  The previous op's records are then checked: the CPU blocks on
step k-1's event while step k is already queued, so the device never runs
dry.  A failure at step k is therefore raised at most one model step (one
STEP op of any domain) late, with the same exception type and message
prefix, the failed step's number, and a sentence saying it was detected
late and how to stop on the exact step.

MANDATORY CHECKPOINTS OF TRUTH.  Every pending record is resolved
synchronously -- :meth:`LaggedHealthMonitor.flush` -- before anything
outside the integration can observe the state: before a history frame is
taken, before a checkpoint is written, at every stop/history/restart period
boundary, before a relocation or a delayed child reads the state, at the
end of the run, and on the way out of a failed run.  No frame or checkpoint
can hold a state whose step has not been checked.

WHAT IS PINNED.  Numerics: nothing is computed differently -- the checks
are read-only reductions into scratch the model never reads, launched where
the strict path launches them.  Detection: at most one op late, never past
a flush point.  The strict path (``--health-check strict``,
``WOOF_HEALTH_CHECK=strict``, and always under ``--health-debug``) is the
historical code, unchanged.  A streamed (``[tiles]``) domain folds its
health inside the tile sweep and stays strict.
"""

from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from woof.core.health_ledger import HealthLedger, refuse

#: The environment opt-out, read when no explicit mode is passed.
ENV = "WOOF_HEALTH_CHECK"
LAGGED = "lagged"
STRICT = "strict"
MODES = (LAGGED, STRICT)
SCHEMA = "woof-health-check-policy-v1"

_STRICT_HINT = ("--health-check strict (or WOOF_HEALTH_CHECK=strict) stops "
                "on the exact step")

# One pinned slot: the 8-word stability record, the 2-word full-state
# validator record and the ledger's status words, at fixed offsets.
_LEDGER_CAPACITY = 64
_STABILITY_OFFSET = 0          # 8 x float32
_VALIDATOR_OFFSET = 64         # 2 x uint64
_LEDGER_OFFSET = 128           # _LEDGER_CAPACITY x uint32
_SLOT_BYTES = _LEDGER_OFFSET + 4 * _LEDGER_CAPACITY


def stability_failure(report: dict, *, grid_id: int, step: int) -> str | None:
    """The step gate's verdict on one stability report: a message, or None.

    Shared by the strict gate (``execute_experiment``'s ``on_step``) and the
    lagged one, so the text a failure carries -- which downstream readers
    match on -- is one string in both modes.
    """
    if report["nan"]:
        return (f"d{int(grid_id):02d} integration produced a non-finite "
                f"state at model step {int(step)}")
    cfl = report["cfl"]
    if cfl is not None and not math.isfinite(float(cfl)):
        return (f"d{int(grid_id):02d} integration produced a non-finite "
                f"vertical Courant number at model step {int(step)}: "
                "a model layer's live thickness is non-positive or non-finite")
    return None


class StepStabilityError(RuntimeError):
    """The per-step stability gate's verdict, raised as a typed failure.

    Still a :class:`RuntimeError` carrying :func:`stability_failure`'s
    text, so every reader that matched the old untyped error still
    matches.  Typed so that the nested health recovery
    (:mod:`woof.stability_recovery`) can tell a failed model state from
    any other runtime error: a blow-up nearly always trips this gate
    before the every-fourth-step full-state validator sees it, and while
    it was untyped the recovery never saw the failure at all -- its
    receipt read ``attempts: []`` and ``failures: []`` with no reason.
    """

    def __init__(self, message, *, grid_id=None, step=None, variable=None):
        super().__init__(message)
        self.grid_id = None if grid_id is None else int(grid_id)
        self.step = None if step is None else int(step)
        self.variable = variable

    @property
    def phase(self) -> str:
        """Where the gate fired, in the full-state validator's spelling."""
        return ("step" if self.grid_id is None
                else f"step.d{self.grid_id:02d}")


def stability_error(report: dict, *, grid_id: int, step: int,
                    note: str | None = None) -> StepStabilityError | None:
    """:func:`stability_failure` as the typed error to raise, or None."""
    failure = stability_failure(report, grid_id=grid_id, step=step)
    if failure is None:
        return None
    variable = ("u/w/theta'" if report["nan"]
                else "vertical Courant number")
    message = failure if note is None else f"{failure} ({note})"
    return StepStabilityError(message, grid_id=grid_id, step=step,
                              variable=variable)


class HealthCheckModeRefused(ValueError):
    """A health-check mode this executor does not implement was requested."""


def resolve_health_check(requested: str | None = None, *,
                         health_debug: bool = False,
                         environ=None) -> dict[str, object]:
    """The run's health-check policy: mode, where it came from, and why.

    Precedence: the explicit argument, then ``WOOF_HEALTH_CHECK``, then the
    default ``"lagged"``.  ``health_debug`` forces ``"strict"`` (attribution
    mode validates every phase synchronously, so a lagged step gate would
    only blur the attribution it exists for).  Anything else is refused by
    name rather than silently mapped onto a mode.
    """
    environ = os.environ if environ is None else environ
    if requested is not None:
        raw, source = requested, "argument"
    elif str(environ.get(ENV, "")).strip():
        raw, source = environ[ENV], f"environment {ENV}"
    else:
        raw, source = LAGGED, "default"
    mode = str(raw).strip().lower()
    if mode not in MODES:
        raise HealthCheckModeRefused(
            f"health check mode {raw!r} (from {source}) is not one of "
            f"{', '.join(MODES)}")
    reason = {
        LAGGED: ("Per-step health records are copied to pinned memory "
                 "asynchronously and checked one op later; every pending "
                 "record is checked synchronously before history, "
                 "checkpoints, stop and the end of the run."),
        STRICT: "Every step's health record is read back before the next "
                "step is queued.",
    }[mode]
    if health_debug and mode == LAGGED:
        mode = STRICT
        reason = ("--health-debug validates every phase synchronously "
                  "(attribution mode), so the step gate is strict too.")
    return {"schema": SCHEMA, "mode": mode, "requested": str(raw),
            "source": source, "reason": reason,
            "max_detection_lag": ("one STEP op; never past a history, "
                                  "checkpoint or stop boundary")
            if mode == LAGGED else "none"}


class _DeferredLedger(HealthLedger):
    """A :class:`HealthLedger` whose device buffer is allocated on first use.

    A run whose steps record nothing (a dry dycore, a CPU test double)
    never allocates it.
    """

    def __init__(self) -> None:  # noqa: D107 - deliberately no allocation
        self._buffer = None
        self._capacity = _LEDGER_CAPACITY
        # Unordered: the snapshot copies exactly _LEDGER_CAPACITY words.
        self._ordered = False
        self._index = {}
        self._describe = {}
        self._label = "lagged-health"
        self.records = 0
        self.drains = 0
        self._pending = 0
        self._sequence = 0
        # The carrier check reads the land/ice footprint only on a path
        # that is about to refuse (never inside a capture: this ledger is
        # entered around eager STEP ops), so an all-water domain is checked
        # exactly as under --health-check strict.
        self.footprint_reads = True
        self.admit = None

    @property
    def _slots(self):
        if self._buffer is None:
            import cupy as cp
            self._buffer = cp.zeros(_LEDGER_CAPACITY, dtype=cp.uint32)
        return self._buffer

    @property
    def allocated(self) -> bool:
        return self._buffer is not None


@dataclass
class _Record:
    grid_id: int
    step: int
    slot: Any
    event: Any
    stability: Callable[[np.ndarray], dict] | None = None
    validator: Callable[[np.ndarray], Any] | None = None
    ledger_sites: tuple = ()
    ledger_describe: dict = field(default_factory=dict)
    label: str = ""


class _Slot:
    """One page-locked host buffer the device copies a step's records into."""

    def __init__(self) -> None:
        import cupy as cp

        self.memory = cp.cuda.alloc_pinned_memory(_SLOT_BYTES)
        raw = np.frombuffer(self.memory, dtype=np.uint8, count=_SLOT_BYTES)
        self.stability = raw[_STABILITY_OFFSET:_STABILITY_OFFSET + 32].view(
            np.float32)
        self.validator = raw[_VALIDATOR_OFFSET:_VALIDATOR_OFFSET + 16].view(
            np.uint64)
        self.ledger = raw[_LEDGER_OFFSET:].view(np.uint32)
        # Re-recorded each time the slot is reused: recording an event
        # that has completed is legal, and it saves a create per step.
        self.event = cp.cuda.Event(disable_timing=True)


class LaggedHealthMonitor:
    """The FIFO of queued, unread health records for one executor run."""

    def __init__(self) -> None:
        self._pending: deque[_Record] = deque()
        self._free: list[_Slot] = []
        self.ledger = _DeferredLedger()
        self.submitted = 0
        self.lagged_resolutions = 0
        self.flushes = 0
        self.flush_reasons: dict[str, int] = {}
        #: ``(grid_id, step)`` of the STEP op in progress or last taken, so
        #: ledger words recorded by a step that then raised can be named.
        self._stepping: tuple[int, int] | None = None
        #: The failure this monitor raised, if any (by identity).
        self._raised: BaseException | None = None
        #: ``ledger.records`` at the last snapshot: unchanged means every
        #: word is still the zero the last snapshot's fill left.
        self._ledger_records_seen = 0

    # -- submission ------------------------------------------------------

    def _slot(self) -> _Slot:
        return self._free.pop() if self._free else _Slot()

    def _ledger_snapshot(self, slot: _Slot, stream):
        """Queue the ledger's words into ``slot`` and clear them, in order."""
        ledger = self.ledger
        if (not ledger.allocated or not ledger._index
                or ledger.records == self._ledger_records_seen):
            # Nothing recorded since the last snapshot cleared the words.
            return (), {}
        import cupy as cp

        self._ledger_records_seen = ledger.records
        ledger._slots.get(stream=stream, out=slot.ledger, blocking=False)
        ledger._slots.fill(cp.uint32(0))
        return tuple(ledger._index.items()), dict(ledger._describe)

    def _enqueue(self, record: _Record, stream) -> None:
        record.ledger_sites, record.ledger_describe = self._ledger_snapshot(
            record.slot, stream)
        record.event = record.slot.event
        record.event.record(stream)
        self._pending.append(record)
        self.submitted += 1

    def begin_step(self, *, grid_id: int, step: int) -> None:
        """Note the STEP op about to run (see :meth:`flush_after_error`)."""
        self._stepping = (int(grid_id), int(step))

    def step_sink(self, *, grid_id: int, step: int, cfg, boundary_width):
        """The ``defer`` callable :func:`woof.core.streaming.step_health` takes.

        Handed the device stability record of the step just queued; copies
        it (and the step's ledger words) behind the step, then checks every
        older record.  ``dt``/``dx`` are frozen here because an adaptive
        driver rewrites ``cfg.dt`` before the record is decoded.
        """
        from types import SimpleNamespace

        frozen = SimpleNamespace(dt=cfg.dt, dx=cfg.dx)

        def sink(device_record) -> None:
            import cupy as cp
            from woof.core.dycore import decode_stability_record

            stream = cp.cuda.get_current_stream()
            slot = self._slot()
            device_record.get(stream=stream, out=slot.stability,
                              blocking=False)
            record = _Record(
                grid_id=int(grid_id), step=int(step), slot=slot, event=None,
                stability=lambda host: decode_stability_record(
                    host, frozen, boundary_width=boundary_width),
                label=f"d{int(grid_id):02d} step {int(step)}")
            self._enqueue(record, stream)
            self._resolve(keep=1, detected=(
                f"after d{int(grid_id):02d} step {int(step)} was queued"))

        return sink

    def drain_ledger(self, *, grid_id: int, step: int) -> None:
        """Check, now, ledger words no step record has taken yet."""
        import cupy as cp

        if (not self.ledger.allocated or not self.ledger._index
                or self.ledger.records == self._ledger_records_seen):
            return
        stream = cp.cuda.get_current_stream()
        record = _Record(grid_id=int(grid_id), step=int(step),
                         slot=self._slot(), event=None,
                         label=f"d{int(grid_id):02d} step {int(step)}")
        self._enqueue(record, stream)
        self.flush("after a failure")

    def submit_validation(self, validator, *, phase: str, grid_id: int,
                          step: int) -> None:
        """The periodic full-state gate, lagged when the validator can be.

        A validator without ``submit`` (a host-store scan, a test double)
        runs synchronously exactly as before.
        """
        import cupy as cp

        try:
            submit = validator.submit
        except AttributeError:
            validator.require_healthy(phase=phase)
            return
        slot = self._slot()
        stream = cp.cuda.get_current_stream()
        decode = submit(phase=phase, out=slot.validator, stream=stream)
        if decode is None:
            self._free.append(slot)
            return
        record = _Record(grid_id=int(grid_id), step=int(step), slot=slot,
                         event=None, validator=decode,
                         label=f"{phase} after root step {int(step)}")
        # Not resolved here: the step this gate follows is still pending,
        # and waiting on it now would stall the queue behind a gate that is
        # itself microseconds of device work.  The next step's submission
        # (or the next flush) checks both, in order.
        self._enqueue(record, stream)

    # -- resolution ------------------------------------------------------

    def flush(self, reason: str) -> None:
        """Check every pending record now (a mandatory checkpoint of truth)."""
        if self._pending:
            self.flushes += 1
            self.flush_reasons[reason] = (
                self.flush_reasons.get(reason, 0) + 1)
        self._resolve(keep=0, detected=f"at the synchronous check {reason}")

    def discard(self) -> None:
        """Forget pending records (their failure has already been raised)."""
        while self._pending:
            record = self._pending.popleft()
            try:
                record.event.synchronize()
            except Exception:  # noqa: BLE001 - the run is already failing
                continue
            self._free.append(record.slot)

    def flush_after_error(self, error: BaseException) -> None:
        """On a failed run: if an older step already went bad, raise THAT.

        The step whose record was still pending ran BEFORE whatever raised,
        so its failure is the root cause and the one strict checking would
        have reported.  Re-raised ``from`` the later error so both survive.
        Failures this monitor raised itself are not re-examined.
        """
        seen, link = set(), error
        while link is not None and id(link) not in seen:
            if link is self._raised:
                # Ours (possibly re-wrapped on the way out): already the
                # earliest failure.
                self.discard()
                return
            seen.add(id(link))
            link = link.__cause__ or link.__context__
        try:
            self.flush("after a failure")
            if self._stepping is not None:
                # Status words a scheme recorded inside the step that then
                # raised: strict checking would have stopped on them first.
                grid_id, step = self._stepping
                self.drain_ledger(grid_id=grid_id, step=step)
        except Exception as earlier:  # noqa: BLE001 - classified below
            self.discard()
            if earlier is self._raised:
                # An earlier step's health failure: the root cause.
                raise earlier from error
            # Anything else (a sticky device fault surfacing at the event
            # wait, say) says nothing about an earlier step; the original
            # error stands.
            error.add_note(
                "lagged health check: pending step records could not be "
                f"read after this failure ({type(earlier).__name__}: "
                f"{earlier})")

    def _resolve(self, *, keep: int, detected: str) -> None:
        while len(self._pending) > keep:
            record = self._pending.popleft()
            record.event.synchronize()
            if keep:
                self.lagged_resolutions += 1
            try:
                self._check(record, detected)
            finally:
                self._free.append(record.slot)

    def _check(self, record: _Record, detected: str) -> None:
        try:
            self._check_record(record, detected)
        except Exception as error:
            self._raised = error
            try:
                error.lagged_health = {
                    "grid_id": record.grid_id, "failed_step": record.step,
                    "detected": detected}
            except AttributeError:  # an exception type without a __dict__
                pass
            raise

    def _check_record(self, record: _Record, detected: str) -> None:
        note = (f"lagged health check: {record.label} failed and was "
                f"detected {detected}; {_STRICT_HINT}")
        for site, index in record.ledger_sites:
            flags = int(record.slot.ledger[index])
            if not flags:
                continue
            try:
                refuse(site, flags, record.ledger_describe.get(site))
            except Exception as error:
                error.add_note(note)
                raise
        if record.stability is not None:
            failure = stability_error(
                record.stability(record.slot.stability.copy()),
                grid_id=record.grid_id, step=record.step, note=note)
            if failure is not None:
                raise failure
        if record.validator is not None:
            try:
                record.validator(record.slot.validator.copy())
            except Exception as error:
                error.add_note(note)
                raise

    def receipt(self, policy: dict) -> dict:
        """The policy plus what this run's monitor actually did."""
        out = dict(policy)
        out.update(submitted=int(self.submitted),
                   lagged_resolutions=int(self.lagged_resolutions),
                   synchronous_flushes=int(self.flushes),
                   flush_reasons=dict(sorted(self.flush_reasons.items())))
        return out


__all__ = ["ENV", "HealthCheckModeRefused", "LAGGED", "LaggedHealthMonitor",
           "MODES", "STRICT", "resolve_health_check"]
