"""Deferred readback of the scheme health status words.

Four schemes end their step by writing a bitmask of "this output went
non-finite / out of range" into a one-word device buffer and reading it
straight back to the host:

============================  ==============================================
site                          what the word means
============================  ==============================================
``microphysics.py``           per-output finite flags for the canonical
                              surface diagnostics, plus SR below/above range
``ysu.py``                    per-output finite flags in launcher order
``kf.py``                     per-output finite flags for the KF arrays
``rrtmgp.py``                 the fused input-plausibility guard
============================  ==============================================

Each readback is a blocking device-to-host copy, and PERF-FINDINGS.md measures
one at ~94 us of pipeline drain -- 50 queued full-size axpy kernels went from
0.566 to 0.660 ms/iteration with a single trailing ``int(status[0].item())``.
That is the small half of the cost.  The large half is that a blocking read is
a HOST BRANCH OVER DEVICE DATA, and a host branch over device data means the
step's launch sequence is a function of the VALUES in the arrays.  While that
is true no CUDA graph can be captured (capture refuses the transfer outright)
and, more fundamentally, no cache key computable on the host can be correct,
because the host cannot see what the branch saw.

So this module does not merely make the reads cheaper.  It is what makes the
step's topology a function of the clock, which is the precondition
:mod:`tilestream.graphcap` rests on.


HOW THE DEFERRAL WORKS, AND WHAT IT COSTS
-----------------------------------------
A :class:`HealthLedger` owns a small uint32 device buffer, one slot per site.
``record`` ORs the site's status word into its slot -- a device-to-device
operation, capturable, and commutative, so many steps (or many TILES) can
accumulate into one slot without ordering.  ``drain`` reads the whole buffer
in ONE copy at a synchronisation the caller already pays for, and raises the
same exception the blocking site would have raised, built by the same closure.

What changes for a HEALTHY run: nothing.  Not the arithmetic, not the field
bytes, not the number of steps.  The gate proves it.

What changes for a SICK run: the failure is reported later -- at the next
drain rather than inside the step -- so the model executes more work before it
aborts.  The exception type and the offending field name are preserved; the
message says when it was detected, because a report that pretends to be
instantaneous when it is not is worse than a late one.  This is the same
trade-off the codebase already accepted when it replaced per-substep mass-flux
readbacks with :class:`~woof.core.dycore.MassFluxAccumulator`.

WHERE A LEDGER IS ACTIVE
------------------------
Three places install one, and with none every site behaves EXACTLY as
before: it reads its status word and raises immediately.

* :func:`deferring`, which the tiled driver enters around a step and leaves
  at the sweep's synchronisation point.
* ``execute_experiment``'s lagged health check (:mod:`woof.core
  .lagged_health`, the default for resident domains there), which enters
  :func:`deferring` around each resident STEP op and reads the words with
  that step's stability record, one op later; ``--health-check strict``
  removes it.  Its ledger reads the consumer footprint on a refusing path
  (``footprint_reads``), so the carrier check is exact there too.
* :func:`resident_step`, which ``woof.core.dycore.step`` enters around every
  resident step and DRAINS AT THE END OF THAT STEP.  A failure is therefore
  still reported by the step that produced it -- before the run loop's
  stability gate, output, checkpoint or nest coupling can observe the state
  -- and the only thing that moves is how far into that step the model got
  before saying so.  MEASURED (docs/dev/perf-launch-overhead-20261015.md):
  an ordinary wind-farm-suite step took seven blocking reads, each a full
  pipeline drain, and on a card shared with other work each of those waited
  for the co-tenants' time slice as well.  ``WOOF_HEALTH_READBACK=immediate``
  restores the per-site reads; :func:`resident_mode` names the mode for
  receipts.  When :func:`deferring` is already active (a tiled sweep, a
  graph capture, an executor that lags its health check) the resident step
  defers to it and does not drain.  Only the sites named in
  :data:`RESIDENT_SITES` are deferred here; on a FAILING step a deferred
  input guard lets its scheme run once on the inputs it rejected (MYNN's
  loops are bounded by the column depth) before the drain raises, and if
  that run faults the device, the drain never replaces the step's own
  exception with an unreadable one.

The resident ledger is THREAD-LOCAL: multi-card runs step each slab on its
own host thread, and one slab's drain must never read, clear or raise another
slab's words.  :func:`active` answers the calling thread's resident ledger
first and the process-wide :func:`deferring` ledger otherwise.

DRAIN OR IT DID NOT HAPPEN
--------------------------
A ledger that is recorded into and never drained silently converts a fatal
error into no error at all.  :meth:`HealthLedger.close` therefore drains, and
:func:`deferring` closes on the way out, including on an exception path.  The
one API that can lose a failure is calling :meth:`record` outside any drain,
which is why ``record`` is not public API and the sites reach it through
:func:`read_status`.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
import threading
from typing import Callable


#: The ledger the schemes report into, or ``None`` for the historical
#: read-immediately behaviour.  Module-level because the reporting sites are
#: four levels down inside three schemes and threading a parameter through
#: every one of them would be a larger and more fragile change than this.
_ACTIVE: "HealthLedger | None" = None

#: The calling thread's resident-step ledger (:func:`resident_step`).
_LOCAL = threading.local()

#: Environment switch for the resident-step deferral and its two values.
#: ``deferred`` is the default; ``immediate`` is the historical per-site read.
RESIDENT_ENV = "WOOF_HEALTH_READBACK"
RESIDENT_MODES = ("deferred", "immediate")
RESIDENT_CAPACITY = 256

#: The sites the resident step defers, by name prefix.  Each is either an
#: OUTPUT check (the scheme has already run; nothing reads the verdict but
#: the refusal) or an input check whose scheme stays bounded on rejected
#: inputs.  Every other site -- RRTMGP's input guard, whose eager path also
#: replays the detailed validators, Grell-Freitas, WDM6 and the urban
#: models -- keeps its immediate read on the resident path; the tiled/graph
#: ledger (:func:`deferring`) still defers them all.
RESIDENT_SITES = (
    "microphysics",          # canonical surface-output finite/range flags
    "ysu",                   # YSU output finite flags
    "kain-fritsch",          # KF output finite flags
    "validated:",            # physics._validated_array scheme outputs
    "carrier-contract:",     # the LSM's radiative-carrier finiteness
    "mynn-",                 # MYNN guards: loops bounded by nz (mynn_pbl.cu)
)


class HealthLedger:
    """Accumulate scheme status words on the device; read them once, later.

    ``capacity`` bounds the number of distinct SITES, not the number of
    calls: a site's slot is ORed into, so a hundred steps and a hundred tiles
    share one word.  A full-physics step reports from about twenty -- four
    scheme status words plus one per validated radiation/cumulus output --
    and the default leaves room.
    """

    def __init__(self, capacity: int = 64, *, label: str = "",
                 footprint_reads: bool = False, ordered: bool = False,
                 admit: tuple[str, ...] | None = None) -> None:
        import cupy as cp

        self._capacity = int(capacity)
        #: ``ordered``: a second word per slot holds the sequence number of
        #: the first FAILING record since the last drain, so the drain
        #: raises the failure that happened first in program order -- the
        #: one an immediate read would have raised -- rather than the site
        #: that happened to be registered first.
        self._ordered = bool(ordered)
        self._slots = cp.zeros(self._capacity * (2 if ordered else 1),
                               dtype=cp.uint32)
        if ordered:
            self._slots[self._capacity:].fill(cp.uint32(_NO_ORDER))
        self._index: dict[str, int] = {}
        self._describe: dict[str, Callable[[int], None]] = {}
        self._label = label
        self.records = 0
        self.drains = 0
        self._pending = 0
        self._sequence = 0
        #: True when a site may still take a blocking read on a path that is
        #: about to REFUSE (never on a healthy path) to reproduce the eager
        #: answer exactly.  The resident ledger sets it; the tiled/graph
        #: ledger does not, because a blocking read inside capture is refused
        #: outright (woof.core.radiation_carriers' consumer footprint).
        self.footprint_reads = bool(footprint_reads)
        #: Site-name prefixes this ledger defers; ``None`` defers every site.
        #: A site it does not admit keeps its immediate read.
        self.admit = None if admit is None else tuple(admit)

    @property
    def slots(self):
        return self._slots[:self._capacity]

    def admits(self, site: str) -> bool:
        return self.admit is None or site.startswith(self.admit)

    def _slot(self, site: str) -> int:
        i = self._index.get(site)
        if i is None:
            if len(self._index) >= self._capacity:
                raise RuntimeError(
                    f"health ledger is full at {self._capacity} sites; "
                    f"cannot add {site!r}")
            i = len(self._index)
            self._index[site] = i
        return i

    def record(self, site: str, status, describe: Callable[[int], None]) -> None:
        """OR ``status[0]`` into ``site``'s slot.  Device-to-device, capturable."""
        import cupy as cp
        import numpy as np

        i = self._slot(site)
        if self._ordered:
            self._sequence += 1
            _ordered_record_kernel()(
                (1,), (1,), (self._slots, status[:1].reshape(1),
                             np.int32(i), np.int32(self._capacity),
                             np.uint32(min(self._sequence, _NO_ORDER - 1))))
        else:
            cp.bitwise_or(self._slots[i:i + 1], status[:1].reshape(1),
                          out=self._slots[i:i + 1])
        self._describe[site] = describe
        self.records += 1
        self._pending += 1

    def drain(self, *, if_recorded: bool = False) -> None:
        """One D2H over every slot; raise for the first site that flagged.

        "First" is program order on an ``ordered`` ledger and registration
        order otherwise.  Clearing happens BEFORE the raise so a caller that
        catches and continues does not re-raise the same historical failure
        for ever.  An exception raised by a site's description carries
        ``_woof_deferred_health = True``.

        ``if_recorded=True`` skips the read when :meth:`record` has not run
        since the last drain.  Only a ledger whose records all run as Python
        may use it: a replayed CUDA graph ORs into the slots without calling
        :meth:`record`, so the tiled/graph driver must always read.
        """
        import cupy as cp

        if not self._index:
            return
        if if_recorded and not self._pending:
            return
        self._pending = 0
        self._sequence = 0
        self.drains += 1
        host = cp.asnumpy(self._slots)
        flagged = [(site, i) for site, i in self._index.items() if int(host[i])]
        if not flagged:
            return
        self._slots[:self._capacity].fill(cp.uint32(0))
        if self._ordered:
            self._slots[self._capacity:].fill(cp.uint32(_NO_ORDER))
            flagged.sort(key=lambda item: int(host[self._capacity + item[1]]))
        site, i = flagged[0]
        try:
            refuse(site, int(host[i]), self._describe.get(site))
        except BaseException as exc:
            try:
                exc._woof_deferred_health = True
            except AttributeError:                         # pragma: no cover
                pass
            raise

    def close(self) -> None:
        self.drain()


def refuse(site: str, flags: int, describe: Callable[[int], None] | None
           ) -> None:
    """Raise for one site whose drained status word is non-zero.

    The one place a drained word becomes an exception -- :meth:`HealthLedger
    .drain` and the executor's lagged health check
    (:mod:`woof.core.lagged_health`) both call it -- so the two readers
    cannot drift.  ``describe`` raises the site's own exception; a site
    with no description, or one whose description returns, still refuses.
    """
    if describe is None:                                   # pragma: no cover
        raise FloatingPointError(
            f"{site} reported health flags {flags:#x} and no "
            "description was recorded for it")
    describe(flags)
    raise FloatingPointError(                              # pragma: no cover
        f"{site} reported health flags {flags:#x} but its "
        "description did not raise; a health site must refuse")


#: The "no failure recorded yet" sequence number of an ordered slot.
_NO_ORDER = 0xFFFFFFFF

_ORDERED_RECORD_SOURCE = r"""
extern "C" __global__ void health_ledger_record_ordered(
        unsigned int* slots, const unsigned int* status, const int slot,
        const int capacity, const unsigned int sequence) {
    const unsigned int word = status[0];
    if (word != 0u) {
        slots[slot] |= word;
        if (sequence < slots[capacity + slot])
            slots[capacity + slot] = sequence;
    }
}
"""
_ORDERED_RECORD_KERNEL: list = []


def _ordered_record_kernel():
    import cupy as cp

    if not _ORDERED_RECORD_KERNEL:
        _ORDERED_RECORD_KERNEL.append(cp.RawKernel(
            _ORDERED_RECORD_SOURCE, "health_ledger_record_ordered",
            options=("-std=c++17",)))
    return _ORDERED_RECORD_KERNEL[0]


def active() -> "HealthLedger | None":
    """The ledger the calling thread's health sites report into, if any."""
    resident = getattr(_LOCAL, "ledger", None)
    return resident if resident is not None else _ACTIVE


def resident_mode() -> str:
    """``"deferred"`` or ``"immediate"``, from :data:`RESIDENT_ENV`."""
    raw = os.environ.get(RESIDENT_ENV, "").strip().lower() or "deferred"
    if raw not in RESIDENT_MODES:
        raise ValueError(
            f"{RESIDENT_ENV}={raw!r} is not one of {RESIDENT_MODES}")
    return raw


#: Process-cumulative resident-step bookkeeping for receipts
#: (:func:`resident_receipt`): steps that deferred, drains that read, and
#: steps that kept the immediate reads, by named reason.
_RESIDENT_LOCK = threading.Lock()
_RESIDENT_STATS = {"deferred_steps": 0, "drains": 0, "immediate_steps": {}}

#: Why a resident step kept the per-site reads.
IMMEDIATE_REASONS = {
    "mode-immediate": f"{RESIDENT_ENV}=immediate",
    "outer-ledger": "a tiled sweep or graph capture owns the ledger and "
                    "its drain",
    "capturing": "the step is being captured into a CUDA graph with no "
                 "ledger of its own",
}


def _count_immediate(reason: str) -> None:
    with _RESIDENT_LOCK:
        table = _RESIDENT_STATS["immediate_steps"]
        table[reason] = table.get(reason, 0) + 1


def resident_receipt() -> dict:
    """The resident health-readback mode and its process-cumulative counts.

    Pinned shape: ``{"mode", "source", "env", "deferred_steps", "drains",
    "immediate_steps": {reason: steps}}``; ``source`` is ``"default"`` or
    ``"environment"``.  Counts cover every ``dycore.step`` in the process.
    """
    with _RESIDENT_LOCK:
        stats = {"deferred_steps": _RESIDENT_STATS["deferred_steps"],
                 "drains": _RESIDENT_STATS["drains"],
                 "immediate_steps": dict(_RESIDENT_STATS["immediate_steps"])}
    return {"mode": resident_mode(),
            "source": ("environment" if os.environ.get(RESIDENT_ENV, "").strip()
                       else "default"),
            "env": RESIDENT_ENV, **stats}


class resident_step:
    """Defer the health reads of ONE resident step and drain at its end.

    ``with resident_step():`` around the body of ``dycore.step``.  Yields the
    ledger, or ``None`` when nothing is deferred: the mode is
    ``immediate``, a ledger is already active (a tiled sweep or a graph
    capture owns the drain; nesting would drain inside their capture), or
    the current stream is capturing.

    On the healthy path the exit is one device-to-host read of the slot
    buffer, and none at all if no site recorded.  If the step body RAISES,
    the ledger is still drained first: a deferred failure recorded earlier
    in the step is what an eager run would have raised first, so it is
    raised (chained to the later exception) in preference to it.
    """

    def __init__(self) -> None:
        self.ledger = None

    def __enter__(self):
        if active() is not None:
            _count_immediate("outer-ledger")
            return None
        if resident_mode() == "immediate":
            _count_immediate("mode-immediate")
            return None
        import cupy as cp

        if cp.cuda.get_current_stream().is_capturing():
            # A capture with no ledger of its own: the drain would be a
            # device-to-host read inside it.  Keep the historical reads,
            # which the capture refuses loudly, rather than a silent one.
            _count_immediate("capturing")
            return None

        ledgers = getattr(_LOCAL, "ledgers", None)
        if ledgers is None:
            ledgers = _LOCAL.ledgers = {}
        device = int(cp.cuda.runtime.getDevice())
        ledger = ledgers.get(device)
        if ledger is None:
            # 256 sites: a site keeps its slot for the life of the thread
            # (radiation, cumulus and initialisation sites included), and a
            # full ledger refuses rather than dropping a site.
            ledger = ledgers[device] = HealthLedger(
                RESIDENT_CAPACITY, label=f"resident-step:cuda{device}",
                footprint_reads=True, ordered=True, admit=RESIDENT_SITES)
        _LOCAL.ledger = ledger
        self.ledger = ledger
        return ledger

    def __exit__(self, exc_type, exc, tb):
        ledger = self.ledger
        if ledger is None:
            return False
        _LOCAL.ledger = None
        self.ledger = None
        drains = ledger.drains
        try:
            self._drain(ledger, exc)
        finally:
            with _RESIDENT_LOCK:
                _RESIDENT_STATS["deferred_steps"] += 1
                _RESIDENT_STATS["drains"] += ledger.drains - drains
        return False

    @staticmethod
    def _drain(ledger, exc) -> None:
        _LOCAL.draining = ledger
        try:
            if exc is None:
                ledger.drain(if_recorded=True)
                return
            try:
                ledger.drain(if_recorded=True)
            except Exception as deferred:
                # Only a deferred HEALTH refusal replaces the step's own
                # exception (it happened first).  A drain that cannot read
                # at all -- a sticky CUDA fault from the same failure --
                # must not hide the exception that explains it.
                if getattr(deferred, "_woof_deferred_health", False):
                    raise deferred from exc
        finally:
            _LOCAL.draining = None


@contextmanager
def deferring(ledger: "HealthLedger | None"):
    """Route the health sites into ``ledger`` for the duration.

    ``None`` restores the historical immediate reads, which is what makes
    this safe to wrap around anything.
    """
    global _ACTIVE
    previous = _ACTIVE
    _ACTIVE = ledger
    try:
        yield ledger
    finally:
        _ACTIVE = previous


def read_status(status, *, site: str, describe: Callable[[int], None] | None) -> int:
    """The status word -- read now, or recorded for later and reported as 0.

    ``describe=None`` forces the immediate read, so a caller that has not
    said how to report a failure never gets a deferral it cannot describe.
    Returning ``0`` on the deferred path is exactly right: zero means "no
    fault seen", which is what every caller does nothing about, and the fault
    -- if there is one -- is reported by the drain instead.
    """
    ledger = active()
    if ledger is None or describe is None or not _admits(ledger, site):
        return int(status[0].item())
    ledger.record(site, status, describe)
    return 0


#: The most spans :func:`record_span_bits` folds into one ledger word.
SPAN_BITS_MAX = 32

_SPAN_BITS_SOURCE = r"""
extern "C" __global__ void health_span_any_bits(
        const int* __restrict__ words, const int* __restrict__ spans,
        const int nspans, unsigned int* __restrict__ out) {
    unsigned int bit = 0u;
    const int i = threadIdx.x;
    if (i < nspans) {
        for (int w = spans[2 * i]; w < spans[2 * i + 1]; ++w) {
            if (words[w] != 0) { bit = 1u << i; break; }
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        bit |= __shfl_xor_sync(0xffffffffu, bit, offset);
    if (i == 0) out[0] = bit;
}
"""

#: ``(device, spans) -> device int32 [lo0, hi0, lo1, hi1, ...]`` and
#: ``(device, stream, site) -> one uint32 word``.  A call site's spans are a function
#: of its check list, so each table is uploaded once per device; the word is
#: rewritten by every call and ORed into the ledger in stream order.
_SPAN_TABLES: dict = {}
_SPAN_WORDS: dict = {}
_SPAN_KERNEL: list = []


def record_span_bits(words, spans, *, site: str, describe) -> bool:
    """Record ``bit i = any(words[lo_i:hi_i] != 0)`` into the active ledger.

    ``words`` is a device int32 flag block a validator has just written
    (one word per checked array); ``spans`` is a sequence of ``(lo, hi)``
    ranges of it, one per check, in the order the immediate read would
    have tested them.  Two launches (the fold and the ledger's OR) replace
    one blocking read.  ``describe(bits)`` runs at the drain with the OR of
    every word this site recorded since the last drain, and must raise.

    Returns ``False`` and launches nothing when no ledger is active or there
    are more than :data:`SPAN_BITS_MAX` spans; the caller then keeps its
    immediate read.
    """
    import cupy as cp
    import numpy as np

    ledger = active()
    if (ledger is None or not spans or len(spans) > SPAN_BITS_MAX
            or not _admits(ledger, site)):
        return False
    if not _SPAN_KERNEL:
        _SPAN_KERNEL.append(cp.RawKernel(
            _SPAN_BITS_SOURCE, "health_span_any_bits",
            options=("-std=c++17",)))
    device = int(cp.cuda.runtime.getDevice())
    spans = tuple((int(lo), int(hi)) for lo, hi in spans)
    table = _SPAN_TABLES.get((device, spans))
    if table is None:
        table = cp.asarray(np.asarray(spans, dtype=np.int32).reshape(-1))
        _SPAN_TABLES[(device, spans)] = table
    # Per (device, stream, site): two streams folding the same site before
    # their ORs must not share the word between them.
    key = (device, int(cp.cuda.get_current_stream().ptr), site)
    out = _SPAN_WORDS.get(key)
    if out is None:
        # Write-before-read: the kernel's thread 0 always stores out[0].
        out = cp.empty(1, dtype=cp.uint32)
        _SPAN_WORDS[key] = out
    _SPAN_KERNEL[0]((1,), (32,), (words, table, np.int32(len(spans)), out))
    ledger.record(site, out, describe)
    return True


def _array_module(cp, array):
    """``cupy`` or ``numpy``, whichever OWNS ``array``.

    ``cupy.get_array_module`` is the canonical answer and is used when it is
    there.  It is not always there: this package is driven from CPU harnesses
    that substitute a numpy-backed stand-in for the ``cp`` name
    (tests/test_wrf_legacy_radiation.py's ``_NumpyCupy`` is one), and such a
    stand-in implements the array API without implementing cupy's dispatch
    helper -- so the canonical call raises AttributeError from inside a
    finiteness CHECK, which is the one place an exception is indistinguishable
    from the thing being checked for.  The fallback answers the same question
    from the array's own type.
    """
    getter = getattr(cp, "get_array_module", None)
    if getter is not None:
        return getter(array)
    import numpy as _np

    return _np if isinstance(array, _np.ndarray) else cp


def check_finite(array, *, site: str, message: str) -> bool:
    """``bool(isfinite(array).all())`` -- read now, or recorded for later.

    The same trade as :func:`read_status`, for the eight-to-fifteen scheme
    outputs the driver validates one at a time on a radiation- or
    cumulus-due step (``physics.py:_validated_array``).  Each one was a full
    reduction FOLLOWED BY A BLOCKING READ; deferred, the reduction still
    runs -- the check is not weakened -- and only the read moves.

    Returns True when the array is known finite (or when the verdict has
    been deferred, which is the same thing as far as every caller's
    behaviour on a healthy step is concerned).
    """
    import cupy as cp

    # The array's OWN module, not this one's.  ArWen drives several of these
    # sites from a CPU harness that monkeypatches the scheme module's ``cp``
    # to numpy (tests/test_physics_driver.py's WSM6 SR case), and a hard
    # ``cupy.isfinite`` there raises ``TypeError: Unsupported type
    # numpy.ndarray``.  Found by their own suite, which is what it is for.
    xp = _array_module(cp, array)
    ledger = active()
    if ledger is None or xp is not cp or not _admits(ledger, site):
        return bool(xp.isfinite(array).all())
    status = cp.logical_not(cp.isfinite(array).all()).astype(
        cp.uint32).reshape(1)
    ledger.record(site, status,
                  lambda _flags: _raise_nonfinite(message))
    return True


def _raise_nonfinite(message: str) -> None:
    raise FloatingPointError(message + deferred_note())


def _admits(ledger, site: str) -> bool:
    admits = getattr(ledger, "admits", None)
    return True if admits is None else bool(admits(site))


def deferred_note(ledger: "HealthLedger | None" = None) -> str:
    """The sentence appended to a deferred failure, so nobody is misled."""
    if getattr(_LOCAL, "draining", None) is not None:
        return (" (detected by the deferred health read at the end of this "
                "model step, so the rest of the step ran after the check "
                f"failed; {RESIDENT_ENV}=immediate stops at the check "
                "itself)")
    return (" (detected at a deferred health drain, so the run advanced past "
            "the step that produced it; re-run with the ledger disabled to "
            "stop on the exact step)")


def masked_clear(mask, array) -> None:
    """``array[...] = where(mask, 0, array)`` as ONE predicated in-place pass.

    Same written values, no temporary, and -- the reason it is here -- no
    host branch: the caller that used to skip this work when nothing was
    masked can now run it unconditionally, which is what makes its launch
    sequence independent of the data.  When the mask is empty the kernel
    reads the mask and writes nothing.

    Bit-exact by construction: it is a select-and-store with no arithmetic,
    so a lane that is not cleared keeps the bits it already had.
    """
    import cupy as cp

    xp = _array_module(cp, array)
    if xp is not cp:
        # Same reason as :func:`check_finite`: a CPU harness reaches here
        # with numpy arrays, and the point of this function is the written
        # VALUES, which numpy's own ``where`` produces identically.
        array[...] = xp.where(mask, xp.float32(0.0), array)
        return

    global _MASKED_CLEAR
    if _MASKED_CLEAR is None:
        _MASKED_CLEAR = cp.ElementwiseKernel(
            "bool m", "float32 a", "if (m) { a = 0.0f; }",
            "gpuwm_masked_clear")
    # The KF mask arrives as (1, ny, nx) so it broadcasts over the 3-D rates;
    # a 2-D carrier needs it back at (ny, nx), because an ElementwiseKernel
    # broadcasts its INPUTS but writes its output in place and will not
    # broadcast the destination.
    if mask.ndim == array.ndim + 1 and mask.shape[0] == 1:
        mask = mask[0]
    _MASKED_CLEAR(mask, array)


_MASKED_CLEAR = None
