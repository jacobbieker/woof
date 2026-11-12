"""Run a hex forecast with a WRF irrigation twin beside the seam's irrigation.

Usage::

    python tools/hex_irrigation_twin.py SUMMARY.json -- <woof hex forecast arguments>

The companion of tools/hex_lake_twin.py for ``sf_surf_irr_scheme``.  The
forecast runs in this process, unchanged; ``PhysicsDriver.apply_irrigation``
(WRF's module_irrigation column, every scheme: channel, drip, sprinkler) is
wrapped in memory only.  Every call is replayed on a twin holding copies of the
fields the seam's call received (IRRIGATION, IRR_RAND_FIELD, RAINBL,
IRRIGATION_CHANNEL and, for the sprinkler, QR/RHO/DZ) and the same clock, and
every array either call can write is compared word for word.  The summary
also counts the calls that added water, so a window that never irrigates is
visible as such.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace


def _install(summary: dict):
    """Wrap apply_irrigation; returns a callable that unwraps it."""
    import numpy as np

    from woof.core.physics import PhysicsDriver

    original = PhysicsDriver.apply_irrigation

    def cupy():
        import cupy as cp

        return cp

    def words(a, b) -> int:
        cp = cupy()
        a, b = cp.asnumpy(cp.asarray(a)), cp.asnumpy(cp.asarray(b))
        if a.shape != b.shape:
            return -1
        view = np.uint32 if a.dtype.itemsize == 4 else np.uint64
        return int(np.count_nonzero(a.view(view) != b.view(view)))

    def wrapper(self, cfg, dt, *, rho=None, dz=None, qr=None, ring=0):
        if not cfg.sf_surf_irr_scheme:
            return original(self, cfg, dt, rho=rho, dz=dz, qr=qr, ring=ring)
        cp = cupy()
        names = ("irrigation", "irr_ph_field", "rainbl", "irrigation_channel")
        shadow_fields = dict(self.fields)
        for name in names:
            shadow_fields[name] = cp.array(self.fields[name], copy=True)
        twin_qr = None if qr is None else cp.array(qr, copy=True)
        before_rain = cp.array(self.fields["rainbl"], copy=True)
        before_qr = None if qr is None else cp.array(qr, copy=True)
        shadow = SimpleNamespace(
            fields=shadow_fields, irrigation_start_time=self.irrigation_start_time,
            state=SimpleNamespace(elapsed_seconds=self.state.elapsed_seconds))
        original(self, cfg, dt, rho=rho, dz=dz, qr=qr, ring=ring)
        original(shadow, cfg, dt, rho=rho, dz=dz, qr=twin_qr, ring=ring)
        summary["calls"] = summary.get("calls", 0) + 1
        added = bool(cp.any(self.fields["rainbl"] != before_rain)) or (
            qr is not None and bool(cp.any(qr != before_qr)))
        summary["calls_adding_water"] = summary.get("calls_adding_water", 0) + int(added)
        counts = summary.setdefault("step_words_differing", {})
        for name in names:
            counts[name] = counts.get(name, 0) + words(self.fields[name], shadow_fields[name])
        if qr is not None:
            counts["qr"] = counts.get("qr", 0) + words(qr, twin_qr)

    PhysicsDriver.apply_irrigation = wrapper

    def restore():
        PhysicsDriver.apply_irrigation = original

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
        summary["bitwise"] = bool(summary.get("calls") and not differing)
        target.write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(json.dumps({key: summary.get(key) for key in (
        "calls", "calls_adding_water", "fields_differing", "bitwise")}, sort_keys=True))
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
