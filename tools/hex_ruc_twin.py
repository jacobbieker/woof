"""Run a hex forecast with a WRF RUC-LSM twin beside every RUC call.

Usage::

    python tools/hex_ruc_twin.py SUMMARY.json -- <woof hex forecast arguments>

The companion of tools/hex_lake_twin.py for ``sf_surface_physics = 3`` (RUC,
mosaic or not).  The forecast runs in this process, unchanged; the engine's
``ruc_lsm_step`` (WRF's ``CASE (RUCLSMSCHEME)`` arm: the RUC column, the mosaic
tile loop over LANDUSEF/SOILCTOP, SFCDIAGS_RUCLSM) is wrapped in memory only.
Every call is replayed on a twin holding copies of every field and atmosphere
array the seam's call received, with the same parameters and clock, through
the engine's REFERENCE arm (``woof.core.ruc_runtime._ruc_lsm_step_reference``,
the per-routine transcription the fused production step is graded against),
not the callable the forecast ran, and every field either call can write is
compared word for word.  RUC's whole state lives in those fields, so a twin
that matches after each call shows the seam's fused column equals the reference
transcription on the same inputs.  (Under ``spp_lsm = 1`` production already
runs the reference arm, and the comparison is then a repeat; the summary says
which path ran.)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _install(summary: dict):
    """Wrap ruc_lsm_step; returns a callable that unwraps it."""
    import numpy as np

    from woof.core import physics as physics_module
    from woof.core.ruc_runtime import _ruc_lsm_step_reference
    from woof.core.surface_forcing import SurfacePrecipitationForcing

    original = physics_module.ruc_lsm_step

    def cupy():
        import cupy as cp

        return cp

    def copy_value(value):
        cp = cupy()
        if isinstance(value, cp.ndarray) or hasattr(value, "__cuda_array_interface__"):
            return cp.array(value, copy=True)
        if isinstance(value, np.ndarray):
            return value.copy()
        return value

    def words(a, b) -> int:
        cp = cupy()
        a, b = cp.asnumpy(cp.asarray(a)), cp.asnumpy(cp.asarray(b))
        if a.shape != b.shape:
            return -1
        if a.dtype.itemsize in (4, 8):
            view = np.uint32 if a.dtype.itemsize == 4 else np.uint64
            return int(np.count_nonzero(a.view(view) != b.view(view)))
        return int(np.count_nonzero(a != b))

    def wrapper(fields, atmosphere, **kwargs):
        twin_fields = {key: copy_value(value) for key, value in fields.items()}
        twin_atm = {key: copy_value(value) for key, value in atmosphere.items()}
        twin_kwargs = dict(kwargs)
        twin_kwargs["precipitation"] = SurfacePrecipitationForcing.from_fields(twin_fields)
        if kwargs.get("field_sf") is not None:
            twin_kwargs["field_sf"] = twin_fields.get("field_sf")
        census = original(fields, atmosphere, **kwargs)
        # The reference arm takes no alb_sol: ruc_lsm_step applies that remap
        # before dispatching, so the twin applies it the same way.
        alb_sol = int(twin_kwargs.pop("alb_sol", 0) or 0)
        twin_call = (dict(twin_fields, albedo=twin_fields["albsol"],
                          albbck=twin_fields["albbcksol"])
                     if alb_sol == 1 else twin_fields)
        _ruc_lsm_step_reference(twin_call, twin_atm, **twin_kwargs)
        summary["reference_is_production"] = bool(int(kwargs.get("spp_lsm", 0)))
        summary["calls"] = summary.get("calls", 0) + 1
        summary["mosaic_lu"] = int(kwargs.get("mosaic_lu", 0))
        summary["mosaic_soil"] = int(kwargs.get("mosaic_soil", 0))
        counts = summary.setdefault("step_words_differing", {})
        for name, value in fields.items():
            if hasattr(value, "shape") and name in twin_fields:
                counts[name] = counts.get(name, 0) + words(value, twin_fields[name])
        return census

    physics_module.ruc_lsm_step = wrapper

    def restore():
        physics_module.ruc_lsm_step = original

    return restore


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] != "--":
        print(__doc__)
        return 2
    target = Path(argv[0])
    # Select the run's backend (--backend / WOOF_BACKEND) before the wrapper
    # imports the engine, which binds the array module at import time.
    from woof.backend import consume_backend_option, install

    forecast_argv = consume_backend_option(argv[2:])
    install()
    summary: dict = {"argv": argv[2:]}
    _install(summary)
    from woof.hex.cli import main as hex_main

    rc = None
    try:
        rc = int(hex_main(["forecast", *forecast_argv]))
    finally:
        summary["exit_code"] = rc
        differing = {key: value for key, value in summary.get(
            "step_words_differing", {}).items() if value}
        summary["fields_differing"] = differing
        summary["fields_compared"] = len(summary.get("step_words_differing", {}))
        summary["bitwise"] = bool(summary.get("calls") and not differing)
        target.write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(json.dumps({key: summary.get(key) for key in (
        "calls", "mosaic_lu", "mosaic_soil", "fields_compared", "fields_differing",
        "bitwise")}, sort_keys=True))
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
