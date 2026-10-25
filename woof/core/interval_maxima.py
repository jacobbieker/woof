"""The history-interval running-max diagnostics a history writer must reset.

WRF zeroes its ``nwp_diagnostics`` running maxima once per history interval
(module_diag_nwp.F ``reset_arrays``); woof zeroes them immediately after the
frame is durable.  ``UP_HELI_MAX`` keeps its own entry point
(its own reset in :mod:`woof.core.uh_diag`, pinned by
tests/test_uh_lifecycle.py); every OTHER per-interval accumulator is a row
of :data:`HISTORY_INTERVAL_SLOTS`, and each history publisher calls
:func:`reset_history_interval_maxima` right beside the UP_HELI_MAX reset.

A new diagnostic of this kind is one row here: its owner module and the
serialized ``(ny, nx)`` scratch slots it folds.  Nothing else in the reset
path changes, and tests/test_wind_gust_diag.py pins that every
history-publishing runner calls the reset.
"""
from __future__ import annotations

from woof.core.wind_gust_diag import WIND_MAXIMA_SLOTS, zero_domain_slot

#: owner module -> the scratch slots it folds every step and the history
#: writer zeroes after every frame.  Every slot named here is
#: restart-serialized (woof/io/restart.py:SERIALIZED_SCRATCH_SLOTS).
HISTORY_INTERVAL_SLOTS: dict[str, tuple[str, ...]] = {
    "woof.core.wind_gust_diag": WIND_MAXIMA_SLOTS,
}


def history_interval_slots() -> tuple[str, ...]:
    """Every registered slot, in registration order."""
    return tuple(slot for slots in HISTORY_INTERVAL_SLOTS.values()
                 for slot in slots)


#: Slots upstream 1.1.0's GSD diagnostics (woof.core.gsd_diagnostics) also
#: fold, under the same names, on their own ``reset_interval1`` window.
#: Where the GSD block is live it owns their window, so the history reset
#: leaves them alone; its own WSPD10UMAX/VMAX and W_MEAN keep the same span.
GSD_SHARED_SLOTS = ("wspd10max", "w_up_max", "w_dn_max")
#: A slot only the GSD block allocates: its presence marks the block live.
_GSD_MARKER = "wspd10umax"


def _gsd_owns(state) -> bool:
    existing = getattr(state, "existing_scratch", None)
    try:
        return existing is not None and existing(_GSD_MARKER) is not None
    except Exception:  # noqa: BLE001 -- a test double without the pool
        return False


def reset_history_interval_maxima(state) -> None:
    """Zero every registered running max wherever the domain keeps it.

    Through :func:`woof.core.wind_gust_diag.zero_domain_slot`, so a
    streamed domain's store copy is zeroed as well as the state's; a slot
    the state never allocated (``nwp_diagnostics = 0``, a test double) is a
    no-op.
    """
    skip = GSD_SHARED_SLOTS if _gsd_owns(state) else ()
    for slot in history_interval_slots():
        if slot not in skip:
            zero_domain_slot(state, slot)


def reset_streamed_history_interval_maxima(stepper) -> None:
    """The same reset on a streamed stepper's store (``None``: resident).

    The run loop's twin of the streamed UP_HELI_MAX reset in runtime: a
    streamed domain whose loop state was never marked keeps its accumulators
    only in ``stepper.store``.
    """
    if stepper is None:
        return
    store = getattr(stepper, "store", None)
    if store is None:
        return
    skip = (GSD_SHARED_SLOTS
            if store.get("scratch/" + _GSD_MARKER) is not None else ())
    for slot in history_interval_slots():
        if slot in skip:
            continue
        buffer = store.get("scratch/" + slot)
        if buffer is not None:
            buffer[...] = 0.0
