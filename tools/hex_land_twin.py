"""Run a hex forecast with a twin beside every slab or Pleim-Xiu land call.

Usage::

    python tools/hex_land_twin.py {slab,px} SUMMARY.json -- <woof hex forecast arguments>

The companion of tools/hex_ruc_twin.py for the thermal-diffusion slab
(``sf_surface_physics = 1``, ``woof.core.slab.launch_slab``) and the Pleim-Xiu
LSM (``sf_surface_physics = 7``, ``woof.core.pxlsm.launch_pxlsm``), the WRF
column ports those schemes' own fixture tests grade word for word against the
Fortran.  The forecast runs in this process, unchanged; the launch is wrapped
in memory only.  Every call is replayed on a twin holding copies of every field
and atmosphere array the seam's call received, and every field either call can
write is compared word for word: the seam hands the column port exactly the
inputs it was graded on, and nothing between calls touches its state.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

#: scheme -> (module holding the name the driver calls, attribute)
TARGETS = {"slab": ("woof.core.physics", "launch_slab"),
           "px": ("woof.core.pxlsm_runtime", "launch_pxlsm")}


def _install(scheme: str, summary: dict):
    """Wrap the scheme's launch; returns a callable that unwraps it."""
    import importlib

    import numpy as np

    module = importlib.import_module(TARGETS[scheme][0])
    name = TARGETS[scheme][1]
    original = getattr(module, name)

    def cupy():
        import cupy as cp

        return cp

    def copy_value(value):
        cp = cupy()
        if isinstance(value, cp.ndarray) or hasattr(value, "__cuda_array_interface__"):
            return cp.array(value, copy=True)
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, dict):
            return {key: copy_value(item) for key, item in value.items()}
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

    def wrapper(fields, *args, **kwargs):
        twin_fields = {key: copy_value(value) for key, value in fields.items()}
        twin_args = tuple(copy_value(value) for value in args)
        # A keyword that is one of the fields (PX's landusef/soilcbot) is the
        # twin's own copy; every other array keyword is copied.
        by_id = {id(value): key for key, value in fields.items()}
        twin_kwargs = {key: (twin_fields[by_id[id(value)]] if id(value) in by_id
                             else copy_value(value))
                       for key, value in kwargs.items()}
        result = original(fields, *args, **kwargs)
        original(twin_fields, *twin_args, **twin_kwargs)
        summary["calls"] = summary.get("calls", 0) + 1
        counts = summary.setdefault("step_words_differing", {})
        for key, value in fields.items():
            if hasattr(value, "shape") and key in twin_fields:
                counts[key] = counts.get(key, 0) + words(value, twin_fields[key])
        return result

    setattr(module, name, wrapper)

    def restore():
        setattr(module, name, original)

    return restore


def main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[0] not in TARGETS or argv[2] != "--":
        print(__doc__)
        return 2
    scheme, target = argv[0], Path(argv[1])
    # Select the run's backend (--backend / WOOF_BACKEND) before the wrapper
    # imports the engine, which binds the array module at import time.
    from woof.backend import consume_backend_option, install

    forecast_argv = consume_backend_option(argv[3:])
    install()
    summary: dict = {"scheme": scheme, "argv": argv[3:]}
    _install(scheme, summary)
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
        "scheme", "calls", "fields_compared", "fields_differing", "bitwise")},
        sort_keys=True))
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
