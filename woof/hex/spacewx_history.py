"""Space-weather history rows on hex cells (``spacewx_ionosphere_opt``, ``spacewx_aurora_opt``).

The rows are the WRF route's own: the climatological ionosphere (woof.spacewx.history.IonosphereHistory) and
the auroral oval with its OVATION energy flux (woof.spacewx.aurora_history.AuroraHistory), evaluated on the
cell centres at each history frame's valid time.  Hex writes each as one float32 value per cell,
``(Time, nCells)``, under its wrfout name and units (woof.io.wrf_output_schema.REGISTRY_VAR_META).

The generic column seam runs WRF's PhysicsDriver on the cells, which builds the history objects from its
RunConfig and radiation geometry (woof.core.physics) as a WRF domain's does; :func:`seam_holders` reads them.

Diagnostic only: nothing the step reads.  A frame's valid time is the schedule's, so a resumed run writes the
rows its uninterrupted run would.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np


def seam_holders(seam) -> tuple:
    """The generic seam's history objects (empty when the options are off or there is no seam)."""
    driver = getattr(seam, "_driver", None)
    return tuple(h for h in (getattr(driver, "spacewx_ionosphere", None), getattr(driver, "spacewx_aurora", None))
                 if h is not None)


def refuse_unwired(backend, mode: str) -> None:
    """Refuse a launch mode whose history writer does not carry these rows when the seam asks for them, rather than
    run and write frames without them."""
    if seam_holders(getattr(backend, "_seam", None)):
        raise NotImplementedError(f"spacewx_ionosphere_opt / spacewx_aurora_opt with {mode}: that path's history "
                                  "writer does not carry the space-weather rows yet; run on one card without active "
                                  "local time stepping")


def _when(valid_time) -> datetime:
    from woof.spacewx.history import _naive_utc
    if not isinstance(valid_time, datetime):
        valid_time = datetime.strptime(str(valid_time), "%Y-%m-%d_%H:%M:%S")
    return _naive_utc(valid_time)


def preload(holders, valid_times) -> None:
    """Load every monthly ionosphere map the frames blend before the time loop (the generic seam's RunConfig holds
    no run length, so IonosphereHistory.for_run loaded only the start's), so no frame fetches one mid-forecast."""
    from woof.spacewx.ionosphere import month_weights
    for holder in holders:
        iono = getattr(holder, "ionosphere", None)
        if iono is None:
            continue
        for month in sorted({m for when in valid_times for m, _w in month_weights(_when(when))}):
            iono.maps(month)


def cell_fields(holders, valid_time, n_cells: int) -> dict:
    """wrfout name -> ``(float32 (n_cells,), ("nCells",), units)`` at ``valid_time`` (a datetime or MPAS's
    ``YYYY-MM-DD_hh:mm:ss``).  The columns past ``n_cells`` (a regional mesh's padded physics column) are
    dropped."""
    if not holders:
        return {}
    import cupy as cp
    from woof.io.wrf_output_schema import REGISTRY_VAR_META
    when = _when(valid_time)
    out = {}
    for holder in holders:
        elapsed = (when - holder.start_time).total_seconds()
        for name, row in holder.history_fields(elapsed).items():
            values = np.ascontiguousarray(cp.asnumpy(row).reshape(-1)[:int(n_cells)], dtype=np.float32)
            if values.size != int(n_cells):
                raise ValueError(f"spacewx row {name}: {values.size} columns for {n_cells} cells")
            out[name] = (values, ("nCells",), REGISTRY_VAR_META[name][1])
    return out


def long_name(name: str) -> str:
    from woof.io.wrf_output_schema import REGISTRY_VAR_META
    return REGISTRY_VAR_META[name][0]


__all__ = ["cell_fields", "long_name", "preload", "refuse_unwired", "seam_holders"]
