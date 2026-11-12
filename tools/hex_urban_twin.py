"""Run a hex forecast with a WRF urban-canopy twin beside the seam's urban model.

Usage::

    python tools/hex_urban_twin.py SUMMARY.json -- <woof hex forecast arguments>

The companion of tools/hex_lake_twin.py for ``sf_urban_physics``.  The forecast
runs in this process, unchanged; engine functions are wrapped in memory only:

* ``woof.core.physics._attach_urban``: after WRF's urban_param_init /
  urban_var_init build the seam's urban state (from the staged FRC_URB2D), a
  TWIN is built by the same function from copies of the same fields, and the
  initial urban arrays are compared;
* ``UrbanCoupler.update_solar`` / ``split_shortwave``: the twin is given the
  same held sun;
* ``UrbanCoupler.after_lsm`` / ``after_surface_diagnostics``: every call is
  replayed on the twin through WRF's urban model with copies of the fields and
  atmosphere the seam's call received, and every field either call can write
  is compared word for word.

The twin carries its own urban arrays from urban_var_init on.  The summary
records the urban cell count, the calls compared and the differing words per
field (all zero is the pass).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _install(summary: dict):
    """Wrap the urban entry points; returns a callable that unwraps them."""
    import numpy as np

    from woof.core import physics as physics_module
    from woof.core.urban_driver import UrbanCoupler

    original_attach = physics_module._attach_urban
    originals = {name: getattr(UrbanCoupler, name) for name in
                 ("update_solar", "split_shortwave", "after_lsm",
                  "after_surface_diagnostics")}

    def cupy():
        import cupy as cp

        return cp

    def copy_value(value):
        cp = cupy()
        if hasattr(value, "__cuda_array_interface__") or isinstance(value, cp.ndarray):
            return cp.array(value, copy=True)
        if isinstance(value, np.ndarray):
            return value.copy()
        return value

    def copy_mapping(mapping):
        return {key: copy_value(value) for key, value in mapping.items()}

    def words(a, b) -> int:
        cp = cupy()
        a = cp.asnumpy(cp.asarray(a))
        b = cp.asnumpy(cp.asarray(b))
        if a.shape != b.shape:
            return -1
        if a.dtype.itemsize == 4:
            return int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32)))
        if a.dtype.itemsize == 8:
            return int(np.count_nonzero(a.view(np.uint64) != b.view(np.uint64)))
        return int(np.count_nonzero(a != b))

    def compare(tag, seam_fields, twin_fields, names):
        counts = summary.setdefault(tag, {})
        for name in names:
            if name not in seam_fields or name not in twin_fields:
                continue
            seam_value, twin_value = seam_fields[name], twin_fields[name]
            if not hasattr(seam_value, "shape"):
                continue
            counts[name] = counts.get(name, 0) + words(seam_value, twin_value)

    class _Holder:
        pass

    def attach(driver, state, cfg, f, **kwargs):
        before = copy_mapping(f)
        original_attach(driver, state, cfg, f, **kwargs)
        holder = _Holder()
        original_attach(holder, state, cfg, before, **{
            key: copy_value(value) for key, value in kwargs.items()})
        seam, twin = driver.urban_coupler, holder.urban_coupler
        seam._woof_twin = twin
        names = tuple(seam.state.names)
        summary["urban_arrays"] = len(names)
        summary["urban_cells"] = int(cupy().count_nonzero(
            cupy().asarray(f["frc_urb2d"]) > 0)) if "frc_urb2d" in f else None
        compare("initial_words_differing", f, before, names)

    def mirrored(name):
        method = originals[name]

        def wrapper(self, *args, **kwargs):
            twin = getattr(self, "_woof_twin", None)
            result = method(self, *args, **kwargs)
            if twin is not None:
                if name == "split_shortwave":
                    fields = args[0]
                    shadow = {key: copy_value(fields[key]) for key in
                              ("swdown", "swddif", "swddir") if key in fields}
                    method(twin, shadow, **kwargs)
                    compare("step_words_differing", fields, shadow, ("swddif", "swddir"))
                else:
                    method(twin, *args, **kwargs)
            return result
        return wrapper

    def stepped(name):
        method = originals[name]

        def wrapper(self, fields, atmosphere, cfg, **kwargs):
            twin = getattr(self, "_woof_twin", None)
            if twin is None:
                return method(self, fields, atmosphere, cfg, **kwargs)
            urban_names = set(twin.state.names)
            twin_fields = twin.state.fields
            for key, value in fields.items():
                if key not in urban_names:
                    twin_fields[key] = copy_value(value)
            twin_atm = copy_mapping(atmosphere)
            method(self, fields, atmosphere, cfg, **kwargs)
            method(twin, twin_fields, twin_atm, cfg, **kwargs)
            summary[f"calls_{name}"] = summary.get(f"calls_{name}", 0) + 1
            compare("step_words_differing", fields, twin_fields, tuple(fields))
        return wrapper

    def restore():
        physics_module._attach_urban = original_attach
        for name, method in originals.items():
            setattr(UrbanCoupler, name, method)

    physics_module._attach_urban = attach
    UrbanCoupler.update_solar = mirrored("update_solar")
    UrbanCoupler.split_shortwave = mirrored("split_shortwave")
    UrbanCoupler.after_lsm = stepped("after_lsm")
    UrbanCoupler.after_surface_diagnostics = stepped("after_surface_diagnostics")
    return restore


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] != "--":
        print(__doc__)
        return 2
    target = Path(argv[0])
    summary: dict = {"argv": argv[2:]}
    _install(summary)
    from woof.hex.cli import main as hex_main

    rc = None
    try:
        rc = int(hex_main(["forecast", *argv[2:]]))
    finally:
        summary["exit_code"] = rc
        differing = {key: value for key, value in summary.get(
            "step_words_differing", {}).items() if value}
        summary["fields_differing"] = differing
        summary["bitwise"] = bool(
            summary.get("calls_after_lsm") and not differing
            and not any(summary.get("initial_words_differing", {}).values()))
        target.write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(json.dumps({key: summary.get(key) for key in (
        "urban_cells", "urban_arrays", "calls_after_lsm",
        "calls_after_surface_diagnostics", "fields_differing", "bitwise")},
        sort_keys=True))
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
