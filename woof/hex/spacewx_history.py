"""Space-weather history rows on hex cells (``spacewx_ionosphere_opt``, woof.spacewx.history).

The rows are the WRF route's own: the climatological ionosphere (woof.spacewx.history.IonosphereHistory)
evaluated on the cell centres at each history frame's valid time.  Hex writes each as one float32 value per
cell, ``(Time, nCells)``, under its wrfout name and units (woof.io.wrf_output_schema.REGISTRY_VAR_META).

* The generic column seam runs WRF's PhysicsDriver on the cells, which builds the history object from its
  RunConfig and radiation geometry (woof.core.physics) as a WRF domain's does; :func:`seam_holders` reads it.
* The MPAS v8.4.2 suite route has no WRF driver: :func:`suite_holders` builds the same object from the
  ``--physics`` spacewx keys the door passes it, on the mesh's cell centres and the case start.

Diagnostic only: nothing the step reads.  A frame's valid time is the schedule's, so a resumed run writes the
rows its uninterrupted run would.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np

#: The --physics keys the suite route hands to these rows instead of to MPAS (its Registry has none of them).
SUITE_KEYS = ("spacewx_ionosphere_opt", "spacewx_f107_source", "spacewx_f107_mode", "spacewx_ionosphere_maps")
#: The RunConfig defaults of SUITE_KEYS.
_DEFAULTS = {"spacewx_ionosphere_opt": 0, "spacewx_f107_source": "", "spacewx_f107_mode": "81day",
             "spacewx_ionosphere_maps": "ccir"}


def seam_holders(seam) -> tuple:
    """The generic seam's history objects (empty when the options are off or there is no seam)."""
    driver = getattr(seam, "_driver", None)
    return tuple(h for h in (getattr(driver, "spacewx_ionosphere", None),) if h is not None)


def suite_settings(settings) -> dict:
    """SUITE_KEYS of ``settings`` (the --physics overrides), typed and judged as the WRF route judges them,
    with the defaults filled; ``{}`` when none is given."""
    given = {key: settings[key] for key in SUITE_KEYS if key in settings}
    if not given:
        return {}
    from woof.config import ConfigurationRefused, validate_spacewx_ionosphere_keys
    values = {**_DEFAULTS, **given}
    try:
        validate_spacewx_ionosphere_keys(values["spacewx_ionosphere_opt"], values["spacewx_f107_source"],
                                         values["spacewx_f107_mode"], values["spacewx_ionosphere_maps"])
    except ConfigurationRefused as exc:
        raise ValueError(str(exc)) from None
    return values


def suite_holders(settings, *, latitude_deg, longitude_deg, start_time, run_seconds, **loaders) -> tuple:
    """The suite route's history objects on the cell centres (degrees), from :func:`suite_settings`."""
    if not int(settings.get("spacewx_ionosphere_opt", 0) or 0):
        return ()
    import json
    from woof.spacewx.history import IonosphereHistory
    lat = np.asarray(latitude_deg, dtype=np.float64).reshape(1, -1)
    lon = np.asarray(longitude_deg, dtype=np.float64).reshape(1, -1)
    cfg = SimpleNamespace(**settings, run_seconds=float(run_seconds))
    geometry = SimpleNamespace(start_time=start_time, latitude_deg=lat, longitude_deg=lon)
    holder = IonosphereHistory.for_run(cfg, geometry, **loaders)
    print("[spacewx ionosphere] " + json.dumps(holder.receipt(), default=str), flush=True)
    return (holder,)


def _when(valid_time) -> datetime:
    if isinstance(valid_time, datetime):
        when = valid_time
    else:
        when = datetime.strptime(str(valid_time), "%Y-%m-%d_%H:%M:%S")
    if when.tzinfo is not None:
        when = when.astimezone(timezone.utc).replace(tzinfo=None)
    return when


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


__all__ = ["SUITE_KEYS", "cell_fields", "long_name", "seam_holders", "suite_holders", "suite_settings"]
