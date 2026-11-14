"""Run a hex forecast with a WRF CLM lake twin beside the seam's lake.

Usage::

    python tools/hex_lake_twin.py SUMMARY.json -- <woof hex forecast arguments>

The forecast runs in this process, unchanged.  Two engine functions are
wrapped (nothing is written to any file the run pins):

* ``woof.core.lake.initialize_lake``: after WRF's lakeini runs for the seam,
  a TWIN lake is initialised by the same function from host copies of the
  inputs the seam's call received (the lake mask, LAKE_DEPTH, its flag, the
  soil category, skin temperature, snow and ice, latitude), and its initial
  state is compared with the seam's;
* ``woof.core.lake.LakeModel.step``: every lake call the forecast makes is
  replayed on the twin through WRF's lake column kernel, with copies of the
  forcing the seam's lake received, and the twin's lake state and surface
  outputs are compared with the seam's, word for word, on the lake cells.

The twin carries its own state from lakeini on, so anything that touched the
seam's lake columns between calls (other than the lake kernel) would show as a
difference.  The summary records the lake cell count, the calls compared and
the number of differing words per field (all zero is the pass).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _install(summary: dict) -> None:
    import numpy as np

    from woof.core import lake as lake_module
    from woof.core.lake_schema import LAKE_OUTPUT_LAYOUT

    original_init = lake_module.initialize_lake
    original_step = lake_module.LakeModel.step
    outputs = tuple(name for name, _ in LAKE_OUTPUT_LAYOUT)

    def cupy():
        # Imported at the first lake call: on Metal the backend installs its
        # cupy facade when the forecast starts, after this tool's own imports.
        import cupy as cp

        return cp

    def copy(value):
        cp = cupy()
        return None if value is None else cp.array(cp.asarray(value), copy=True)

    def words(a, b) -> int:
        cp = cupy()
        a = cp.asnumpy(cp.asarray(a)).astype(np.float32, copy=False)
        b = cp.asnumpy(cp.asarray(b)).astype(np.float32, copy=False)
        return int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32)))

    def initialize_lake(fields, *, latitude, **kwargs):
        cp = cupy()
        seed = {name: copy(fields[name]) for name in
                ("tsk", "isltyp", "snow", "xice", "lakemask") if name in fields}
        twin_kwargs = {key: (copy(value) if key == "lake_depth" else value)
                       for key, value in kwargs.items()}
        owner = original_init(fields, latitude=latitude, **kwargs)
        twin_fields = dict(seed)
        twin = original_init(twin_fields, latitude=copy(latitude), **twin_kwargs)
        index = owner.indices
        summary["lake_cells"] = int(owner.column_count)
        summary["initial_words_differing"] = {
            name: words(cp.take(fields[name].reshape(fields[name].shape[0], -1), index, axis=1),
                        cp.take(twin_fields[name].reshape(twin_fields[name].shape[0], -1), index, axis=1))
            for name in ("lake_columns", "lake_static")}
        owner._woof_twin = (twin, twin_fields)
        return owner

    def step(self, fields, atmosphere, *, dt, precipitation):
        cp = cupy()
        pair = getattr(self, "_woof_twin", None)
        if pair is None:
            return original_step(self, fields, atmosphere, dt=dt, precipitation=precipitation)
        twin, twin_fields = pair
        for name in (*outputs, "glw", "emiss", "swdown"):
            twin_fields[name] = copy(fields[name])
        twin_atm = {name: copy(atmosphere[name][:2]) for name in
                    ("temperature", "p_interface", "dz", "qv", "u", "v")}
        rain = copy(precipitation)
        original_step(self, fields, atmosphere, dt=dt, precipitation=precipitation)
        original_step(twin, twin_fields, twin_atm, dt=dt, precipitation=rain)
        index = self.indices
        calls = summary.setdefault("calls", 0) + 1
        summary["calls"] = calls
        differing = summary.setdefault("step_words_differing", {})
        for name in ("lake_columns", *outputs):
            seam_value = fields[name].reshape(-1 if name != "lake_columns" else (fields[name].shape[0], -1))
            twin_value = twin_fields[name].reshape(-1 if name != "lake_columns" else (twin_fields[name].shape[0], -1))
            axis = 1 if name == "lake_columns" else 0
            count = words(cp.take(seam_value, index, axis=axis), cp.take(twin_value, index, axis=axis))
            differing[name] = differing.get(name, 0) + count
        if calls == 1:
            summary["first_call_lake_tsk"] = cp.asnumpy(
                cp.take(fields["tsk"].reshape(-1), index)).astype(float).tolist()

    lake_module.initialize_lake = initialize_lake
    lake_module.LakeModel.step = step


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

    try:
        rc = int(hex_main(["forecast", *forecast_argv]))
    finally:
        summary["exit_code"] = locals().get("rc")
        differing = summary.get("step_words_differing", {})
        summary["bitwise"] = bool(
            summary.get("calls") and not any(differing.values())
            and not any(summary.get("initial_words_differing", {}).values()))
        target.write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(json.dumps({key: summary.get(key) for key in (
        "lake_cells", "calls", "initial_words_differing", "step_words_differing",
        "bitwise")}, sort_keys=True))
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
