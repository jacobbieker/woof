"""Run a hex forecast with a WRF Noah-mosaic twin beside the seam's tiles.

Usage::

    python tools/hex_mosaic_twin.py SUMMARY.json -- <woof hex forecast arguments>

The companion of tools/hex_lake_twin.py and tools/hex_urban_twin.py for
``sf_surface_mosaic = 1``.  The forecast runs in this process, unchanged;
engine functions are wrapped in memory only:

* ``woof.core.noah_mosaic_door.attach_noah_mosaic_to_driver``: after WOOF's
  door builds the seam's tile state from the staged LANDUSEF (real.exe's
  edits, lsm_mosaic_init), a TWIN tile state is built by the same door from
  copies of the same fields and the same LANDUSEF, and compared;
* ``woof.core.noah_mosaic.launch_noah_mosaic``: every mosaic call is replayed
  on the twin through WRF's lsm_mosaic port with copies of the fields the
  seam's call received (the twin carries its own tile state), and every field
  is compared word for word.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _install(summary: dict):
    """Wrap the mosaic entry points; returns a callable that unwraps them."""
    import numpy as np

    from woof.core import noah_mosaic as mosaic_module
    from woof.core import noah_mosaic_door as door_module

    original_attach = door_module.attach_noah_mosaic_to_driver
    original_launch = mosaic_module.launch_noah_mosaic

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

    def compare(tag, seam, twin, names):
        counts = summary.setdefault(tag, {})
        for name in names:
            if name in seam and name in twin and hasattr(seam[name], "shape"):
                counts[name] = counts.get(name, 0) + words(seam[name], twin[name])

    class _Holder:
        pass

    twins: dict = {}

    def attach(driver, cfg, **kwargs):
        before = {key: copy_value(value) for key, value in driver.fields.items()}
        original_attach(driver, cfg, **kwargs)
        owned = tuple(key for key in driver.fields if key not in before)
        holder = _Holder()
        holder.fields = before
        holder.noah_params = driver.noah_params
        original_attach(holder, cfg, **{key: copy_value(value)
                                        for key, value in kwargs.items()})
        twins[id(driver.fields)] = (holder.fields, owned)
        summary["tile_arrays"] = len(owned)
        compare("initial_words_differing", driver.fields, holder.fields,
                tuple(driver.fields))

    def launch(dev, *args, **kwargs):
        twin = twins.get(id(dev))
        if twin is None:
            return original_launch(dev, *args, **kwargs)
        twin_fields, owned = twin
        for key, value in dev.items():
            if key not in owned:
                twin_fields[key] = copy_value(value)
        original_launch(dev, *args, **kwargs)
        original_launch(twin_fields, *args, **kwargs)
        summary["calls"] = summary.get("calls", 0) + 1
        compare("step_words_differing", dev, twin_fields, tuple(dev))

    def restore():
        door_module.attach_noah_mosaic_to_driver = original_attach
        mosaic_module.launch_noah_mosaic = original_launch

    door_module.attach_noah_mosaic_to_driver = attach
    mosaic_module.launch_noah_mosaic = launch
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
            summary.get("calls") and not differing
            and not any(summary.get("initial_words_differing", {}).values()))
        target.write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(json.dumps({key: summary.get(key) for key in (
        "tile_arrays", "calls", "fields_differing", "bitwise")}, sort_keys=True))
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
