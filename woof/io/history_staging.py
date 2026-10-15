"""How a history frame leaves the model loop: the budgets and the receipt.

The asynchronous wrfout writer (:class:`woof.io.wrfout.AsyncDomainWrfoutWriter`)
used to make the model's own stream wait for the whole device-to-host copy of
every frame: the copy ran on a side stream, and ``producer.wait_event(done)``
put the next kernel behind it.  It now takes a DEVICE SNAPSHOT instead.  The
frame's fields are copied device to device, on the producer stream, into
buffers the writer keeps; the host copy reads those buffers on the side
stream; and the producer waits only for the device copy, which runs at device
bandwidth rather than PCIe bandwidth.  This module decides, by name, whether a
domain may hold those buffers, how many frames its writer may queue, and
whether a ``[tiles]`` host-store frame may be snapshotted instead of drained.
It also collects what happened into the run receipt's ``history_writes``
block.

PINNED contract (tests/test_history_output_path.py):

* The device snapshot is EXTRA device memory: one copy of the frame's
  selected device fields, held for the writer's life.  It is admitted only
  inside the run's PRICED PLAN.  That means the room the door's resident
  admission (:func:`woof.core.streaming.admit_resident_road`) left between the
  card's free memory and the forecast's peak envelope, shared by every writer
  on that card, and never more than
  :data:`woof.core.devices_memory.FRAME_SNAPSHOT_LIMIT_BYTES` per writer (the
  ``[devices]`` road's cap).  Without an admission record
  the mode is ``direct (unpriced)``.  A refused or failed allocation is
  ``direct`` with its reason, and direct is exactly the old path: the producer
  waits for the host copy.
* ``WOOF_ASYNC_HISTORY=off`` restores every old path (direct device copies,
  drained ``[tiles]`` frames, synchronous hex writes); ``on`` admits the
  device snapshot against the card's live free memory when no plan was
  priced (for benchmarks), still under the cap.
* Queue depth: ``[output] history_queue_depth`` (default 1, which is the old
  bound) or ``WOOF_HISTORY_QUEUE_DEPTH``.  The page-locked frames a writer can
  hold (queued + written + staged) must fit host RAM beside the run, with the
  async checkpoint's reserve (the larger of 2 GiB and 5% of RAM).  A depth
  that does not fit is lowered, and the receipt says so.
* The page-locked pool is released at the writers' drain points, never in
  the middle of a write.
* Nothing here changes a byte of any file: the snapshot is a copy, in the
  same order, of the same arrays.
"""
from __future__ import annotations

import os
import threading
from statistics import median

#: Operator switch for every asynchronous history path.  ``off`` (or
#: 0/false/no/sync) restores the old behaviour; ``auto`` (default) admits
#: by budget; ``on`` also admits an UNPRICED device snapshot against live
#: free memory.
ASYNC_HISTORY_ENV = "WOOF_ASYNC_HISTORY"
QUEUE_DEPTH_ENV = "WOOF_HISTORY_QUEUE_DEPTH"
RECEIPT_SCHEMA = "woof.history-writes/v1"

MAX_QUEUE_DEPTH = 8

_OFF = frozenset({"0", "false", "off", "no", "sync"})
_ON = frozenset({"1", "true", "on", "yes", "force"})
_AUTO = frozenset({"", "auto"})

#: Host RAM the page-locked frames and snapshots must leave free is the
#: asynchronous checkpoint's rule, imported, so the two never disagree.
from woof.io.async_checkpoint import (  # noqa: E402
    _RAM_RESERVE_BYTES, _RAM_RESERVE_FRACTION)

#: Device memory a live-memory (``on``, unpriced) snapshot leaves free.
_LIVE_DEVICE_RESERVE_BYTES = 1 << 30

#: Why a writer copies straight from the live arrays (the old path).
DIRECT_REASONS = {
    "disabled": "asynchronous history is switched off for this run",
    "unpriced": ("no resident admission priced this card, so there is no "
                 "plan the device snapshot can be admitted inside"),
    "outside-priced-plan": ("the device snapshot does not fit the room the "
                            "run's priced plan leaves on the card"),
    "over-cap": "the frame is larger than the per-writer snapshot cap",
    "device-allocation-failed": "the device snapshot buffers could not be allocated",
    "no-live-free-memory": "the card's live free memory cannot hold the snapshot",
}


def async_history_mode(environ=None) -> str:
    """``"auto"``, ``"on"`` or ``"off"`` from :data:`ASYNC_HISTORY_ENV`."""
    environ = os.environ if environ is None else environ
    raw = str(environ.get(ASYNC_HISTORY_ENV, "")).strip().lower()
    if raw in _AUTO:
        return "auto"
    if raw in _OFF:
        return "off"
    if raw in _ON:
        return "on"
    raise ValueError(
        f"{ASYNC_HISTORY_ENV}={raw!r} is not a switch; use auto (default), "
        "off to restore the synchronous/fenced history paths, or on to "
        "admit an unpriced device snapshot against live free memory")


def validate_queue_depth(value, where: str) -> int:
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{where} must be an integer from 1 to {MAX_QUEUE_DEPTH}, "
            f"got {value!r}")
    if not 1 <= int(value) <= MAX_QUEUE_DEPTH:
        raise ValueError(
            f"{where} must be from 1 to {MAX_QUEUE_DEPTH} (frames queued per "
            f"domain behind the one being written), got {value}")
    return int(value)


def resolve_queue_depth(selection=None, *, environ=None) -> dict:
    """``{"requested": n, "source": ...}``: env, then ``[output]``, then 1."""
    environ = os.environ if environ is None else environ
    raw = environ.get(QUEUE_DEPTH_ENV)
    if raw is not None and str(raw).strip():
        return {"requested": validate_queue_depth(str(raw), QUEUE_DEPTH_ENV),
                "source": f"env {QUEUE_DEPTH_ENV}"}
    value = getattr(selection, "history_queue_depth", None)
    if value is not None:
        return {"requested": int(value),
                "source": "[output] history_queue_depth"}
    return {"requested": 1, "source": "default"}


def host_ram_budget() -> tuple[int, int] | None:
    """``(available - reserve, reserve)`` host bytes, or ``None`` if unknown."""
    from woof.io.async_checkpoint import host_ram_available

    ram = host_ram_available()
    if ram is None:
        return None
    available, total = ram
    reserve = max(_RAM_RESERVE_BYTES, int(_RAM_RESERVE_FRACTION * (total or 0)))
    return int(available) - reserve, reserve


def effective_queue_depth(requested: int, frame_bytes: int, *,
                          budget=None) -> tuple[int, str | None]:
    """The depth whose page-locked frames fit host RAM, and why if lowered.

    A writer holding depth ``d`` keeps up to ``d`` queued frames, the one it
    is writing and the one being staged: ``d + 2`` frames of page-locked
    memory.  Depth 1 is never refused: it is the old bound, which every run
    already paid.
    """
    requested = int(requested)
    if requested <= 1 or frame_bytes <= 0:
        return max(1, requested), None
    if budget is None:
        budget = host_ram_budget()
    if budget is None:
        return requested, None
    room = budget[0]
    fits = int(room // max(1, int(frame_bytes))) - 2
    if fits >= requested:
        return requested, None
    depth = max(1, fits)
    return depth, (f"depth-capped-by-pinned-budget: {requested} frames of "
                   f"{frame_bytes / 2**20:.1f} MiB do not fit "
                   f"{max(room, 0) / 2**30:.2f} GiB of host RAM beside the "
                   f"run; using {depth}")


def admit_host_snapshot(nbytes: int, *, budget=None) -> tuple[bool, str | None]:
    """May a host-RAM snapshot of ``nbytes`` be held? ``(ok, reason)``."""
    if async_history_mode() == "off":
        return False, "disabled"
    if budget is None:
        budget = host_ram_budget()
    if budget is None:
        return True, None
    if int(nbytes) > budget[0]:
        return False, (f"insufficient-host-ram: snapshot "
                       f"{nbytes / 2**30:.2f} GiB, room "
                       f"{max(budget[0], 0) / 2**30:.2f} GiB beside the run "
                       f"(reserve {budget[1] / 2**30:.2f} GiB)")
    return True, None


# ---------------------------------------------------------------------------
# The priced plan's room on each card, and the snapshots drawn from it
# ---------------------------------------------------------------------------

_PLAN_LOCK = threading.Lock()
#: device id -> {"slack_bytes", "need_bytes", "free_bytes", "what"}
_PRICED_PLANS: dict[int, dict] = {}
#: device id -> bytes of device snapshot admitted against that plan
_RESERVED: dict[int, int] = {}


def record_priced_plan(record, *, device=None) -> None:
    """Remember the door's resident admission for the card it priced.

    ``record`` is :func:`woof.core.resident_admission.admit`'s return value.
    The room is ``free - need``: what the priced peak envelope leaves on the
    card.  An unread card (``fits`` ``None``) records nothing.
    """
    if not record or record.get("free_bytes") is None:
        return
    if device is None:
        try:
            import cupy as cp
            device = int(cp.cuda.Device().id)
        except Exception:                      # noqa: BLE001 - no runtime
            device = 0
    with _PLAN_LOCK:
        _PRICED_PLANS[int(device)] = {
            "what": record.get("what"),
            "need_bytes": int(record.get("need_bytes") or 0),
            "free_bytes": int(record["free_bytes"]),
            "slack_bytes": int(record["free_bytes"]) - int(record.get("need_bytes") or 0),
            "overridden": bool(record.get("overridden", False)),
        }


def priced_plan(device) -> dict | None:
    with _PLAN_LOCK:
        plan = _PRICED_PLANS.get(int(device))
        return None if plan is None else dict(plan, reserved_bytes=_RESERVED.get(int(device), 0))


def forget_priced_plans() -> None:
    """Test hook: forget every recorded plan and reservation."""
    with _PLAN_LOCK:
        _PRICED_PLANS.clear()
        _RESERVED.clear()


def reserve_device_snapshot(device, nbytes: int, *, writer_bytes: int = 0,
                            live_free=None) -> tuple[bool, str | None, str]:
    """Admit ``nbytes`` more of device snapshot on ``device``.

    ``writer_bytes`` is what the asking writer already holds: the 4 GiB cap
    is per writer, the priced plan's room is per card (every writer on it).
    Returns ``(admitted, reason, basis)``; ``reason`` names the direct
    fallback (a :data:`DIRECT_REASONS` key, with figures) when refused.
    """
    from woof.core.devices_memory import FRAME_SNAPSHOT_LIMIT_BYTES

    mode = async_history_mode()
    if mode == "off":
        return False, "disabled", "switch"
    device = int(device)
    nbytes = int(nbytes)
    with _PLAN_LOCK:
        held = _RESERVED.get(device, 0)
        plan = _PRICED_PLANS.get(device)
        if int(writer_bytes) + nbytes > FRAME_SNAPSHOT_LIMIT_BYTES:
            return False, (f"over-cap: {(int(writer_bytes) + nbytes) / 2**30:.2f} "
                           "GiB of snapshot for one writer against the "
                           f"{FRAME_SNAPSHOT_LIMIT_BYTES / 2**30:.0f} GiB cap"), "cap"
        if plan is not None:
            if held + nbytes > plan["slack_bytes"]:
                return False, (
                    f"outside-priced-plan: needs {(held + nbytes) / 2**20:.0f} "
                    f"MiB, the plan leaves {max(plan['slack_bytes'], 0) / 2**20:.0f} "
                    "MiB on the card"), "priced-plan"
            _RESERVED[device] = held + nbytes
            return True, None, "priced-plan"
        if mode != "on":
            return False, "unpriced", "none"
        if live_free is None:
            from woof.core.resident_admission import device_free_bytes
            live_free = device_free_bytes()
        if live_free is None or nbytes + _LIVE_DEVICE_RESERVE_BYTES > live_free:
            return False, "no-live-free-memory", "live-free-memory"
        _RESERVED[device] = held + nbytes
        return True, None, "live-free-memory"


def release_device_snapshot(device, nbytes: int) -> None:
    with _PLAN_LOCK:
        held = _RESERVED.get(int(device), 0) - int(nbytes)
        if held > 0:
            _RESERVED[int(device)] = held
        else:
            _RESERVED.pop(int(device), None)


def free_pinned_pool() -> int:
    """Release CuPy's unused page-locked blocks; the count released.

    Only FREE blocks are released (``free_all_blocks``), so a frame still
    being written keeps its memory; called where the writers are drained.
    """
    try:
        import cupy as cp
        pool = cp.get_default_pinned_memory_pool()
        blocks = int(pool.n_free_blocks())
        pool.free_all_blocks()
        return blocks
    except Exception:                          # noqa: BLE001 - no runtime
        return 0


# ---------------------------------------------------------------------------
# What each writer did, for the run receipt
# ---------------------------------------------------------------------------

class FrameStallLog:
    """Per-frame model-loop cost of one writer's submits.

    ``wall_seconds`` is the host time the model thread spent inside the
    submit.  ``stream_seconds`` is the time the producer stream spent
    between two events recorded around it: the frame's derivations, the
    device snapshot copy, and in direct mode the wait for the host copy.
    The events are resolved lazily, when they have completed, so recording
    them never synchronises the device.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.wall_seconds: list[float] = []
        self.stream_seconds: list[float] = []
        self._pending: list[tuple[object, object]] = []
        self.modes: dict[str, int] = {}
        self.frame_bytes: list[int] = []

    def record(self, *, wall_seconds, mode, nbytes=0, events=None) -> None:
        with self._lock:
            self.wall_seconds.append(float(wall_seconds))
            self.modes[mode] = self.modes.get(mode, 0) + 1
            self.frame_bytes.append(int(nbytes))
            if events is not None:
                self._pending.append(events)

    def resolve(self) -> None:
        with self._lock:
            pending, self._pending = self._pending, []
        keep = []
        for start, end in pending:
            try:
                import cupy as cp
                if not end.done:
                    keep.append((start, end))
                    continue
                seconds = cp.cuda.get_elapsed_time(start, end) / 1e3
            except Exception:                  # noqa: BLE001 - timing only
                continue
            with self._lock:
                self.stream_seconds.append(float(seconds))
        if keep:
            with self._lock:
                self._pending.extend(keep)

    def summary(self) -> dict:
        self.resolve()

        def stats(values):
            if not values:
                return None
            return {"frames": len(values), "median": round(median(values), 6),
                    "max": round(max(values), 6), "total": round(sum(values), 6)}

        with self._lock:
            return {"submit_wall_seconds": stats(self.wall_seconds),
                    "producer_stream_seconds": stats(self.stream_seconds),
                    "frames_by_mode": dict(self.modes),
                    "frame_bytes_max": max(self.frame_bytes, default=0)}
