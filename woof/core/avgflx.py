"""WRF ``do_avgflx_em = 1``: history-interval means of the coupled wind fluxes.

WRF v4.7.1 Registry/registry.avgflx declares three restart-and-history
(``rh``) fields and a restart-only (``r``) counter::

    AVGFLX_RUM  "hist-time-averaged mu-coupled u"        Pa m s-1  (X stag)
    AVGFLX_RVM  "hist-time-averaged mu-coupled v"        Pa m s-1  (Y stag)
    AVGFLX_WWM  "hist-time-averaged mu-coupled eta-dot"  Pa s-1    (Z stag)
    avgflx_count  "Counter for time-averaged mu-coupled velocities"

solve_em (dyn_em/solve_em.F:3498-3557), after the Runge-Kutta loop and
before the microphysics, folds the LAST RK stage's sumflux fluxes ``ru_m``,
``rv_m``, ``ww_m`` into a running mean with ``upd_avgflx``
(dyn_em/module_avgflx_em.F:66-134; kernels/avgflx.cu), after resetting it
(``zero_avgflx``, count = 0) on the first step after a history write
(``CurrTime < previous history ring + dt``).  start_em (:1300-1315) sets
the count to 0 on every start, and zeroes the arrays unless restarting.

woof keeps the three means as serialized scratch slots (a checkpoint carries
them, as WRF's restart does) and the counter as a one-word int32 device slot
REBUILT at zero on a restore -- start_em's ``avgflx_count = 0`` on every
start, restart included.  The pending reset after a history step is the
counter's value -1: the kernel then reads the mean as zero and the count as
0, which is zero_avgflx followed by upd_avgflx word for word, with no host
read.

``do_avgflx_cugd = 1`` (the Grell-Devenyi convective mass fluxes CFU1/CFD1/
DFU1/EFU1/DFD1/EFD1 of module_cu_gd.F, cu_physics = 93) is refused by name:
Grell-Devenyi is not ported, so the fluxes it would average do not exist.
MPAS has no such option (woof.hex refuses it).
"""

from __future__ import annotations

import numpy as np

#: Serialized scratch slots, by history name.
SLOTS = {"AVGFLX_RUM": "avgflx_rum", "AVGFLX_RVM": "avgflx_rvm",
         "AVGFLX_WWM": "avgflx_wwm"}
#: The device counter (rebuilt at 0: start_em's avgflx_count = 0).
COUNT_SLOT = "avgflx_count"

#: Registry/registry.avgflx descriptions and units, verbatim.
HISTORY_FIELDS = {
    "AVGFLX_RUM": ("hist-time-averaged mu-coupled u", "Pa m s-1", "X"),
    "AVGFLX_RVM": ("hist-time-averaged mu-coupled v", "Pa m s-1", "Y"),
    "AVGFLX_WWM": ("hist-time-averaged mu-coupled eta-dot", "Pa s-1", "Z"),
}

_TPB = 256


def active(cfg) -> bool:
    return int(getattr(cfg, "do_avgflx_em", 0) or 0) == 1


def refusal(cfg) -> str | None:
    value = getattr(cfg, "do_avgflx_em", 0)
    if isinstance(value, bool) or value not in (0, 1):
        return ("do_avgflx_em must be 0 or 1 (Registry/registry.avgflx), "
                f"got {value!r}")
    return None


def _shapes(nz: int, ny: int, nx: int) -> dict[str, tuple[int, ...]]:
    return {"avgflx_rum": (nz, ny, nx + 1), "avgflx_rvm": (nz, ny + 1, nx),
            "avgflx_wwm": (nz + 1, ny, nx)}


def scratch_shapes(cfg) -> dict[str, tuple[int, ...]]:
    """Every float32 scratch slot the switch allocates, for memory planning
    (the one-word int32 counter is priced as nothing)."""
    if not active(cfg):
        return {}
    return _shapes(int(cfg.nz), int(cfg.ny), int(cfg.nx))


def initialize(state, cfg) -> None:
    """start_em's zero_avgflx and ``avgflx_count = 0`` (start_em.F:1300-1315)
    on a new state, so the t = 0 history frame carries WRF's zeros."""
    if not active(cfg):
        return
    _arrays(state)
    _count(state)


def _arrays(state):
    """The three means; created zeroed (start_em's zero_avgflx on a fresh
    start; a restore has already put the checkpoint's words here)."""
    nz, ny, nx = state.p.shape
    out = {}
    for slot, shape in _shapes(nz, ny, nx).items():
        fresh = state.existing_scratch(slot) is None
        out[slot] = state.scratch(shape, slot)
        if fresh:
            out[slot][...] = 0
    return out


def _count(state):
    """avgflx_count, created at 0 (start_em sets it on every start)."""
    fresh = state.existing_scratch(COUNT_SLOT) is None
    count = state.scratch((1,), COUNT_SLOT, dtype=np.int32)
    if fresh:
        count[...] = 0
    return count


def update(state, cfg, ru_m, rv_m, ww_m, *, history_due: bool) -> None:
    """solve_em's avgflx block for one completed step.

    ``ru_m``/``rv_m``/``ww_m`` are the last RK stage's sumflux fluxes;
    ``history_due`` says a history frame is written at this step's end, so
    the next step starts a new mean."""
    if not active(cfg):
        return
    from woof.core.kernels import get_kernel
    kernel = get_kernel("avgflx", "upd_avgflx")
    count = _count(state)
    arrays = _arrays(state)
    for slot, flux in (("avgflx_rum", ru_m), ("avgflx_rvm", rv_m),
                       ("avgflx_wwm", ww_m)):
        avg = arrays[slot]
        if flux.shape != avg.shape:
            raise ValueError(f"{slot}: flux {flux.shape} vs mean {avg.shape}")
        n = avg.size
        kernel(((n + _TPB - 1) // _TPB,), (_TPB,),
               (avg, flux, count, np.int64(n)))
    # grid%avgflx_count = grid%avgflx_count + 1 (from 0 after a reset).
    import cupy as cp
    cp.add(cp.maximum(count, 0), 1, out=count)
    if history_due:
        count[...] = -1


def history_fields(state) -> dict[str, object]:
    """The three means under their WRF history names (views)."""
    existing = getattr(state, "existing_scratch", None)
    out = {}
    for name, slot in SLOTS.items():
        array = (existing(slot) if existing is not None
                 else getattr(state, "_scratch", {}).get(slot))
        if array is not None:
            out[name] = array
    return out
