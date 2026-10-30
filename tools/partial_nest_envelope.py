"""Write the partial-depth nest (Phase D3) envelope receipt.

    pixi run -e mlx python tools/partial_nest_envelope.py [OUT.json]
    pixi run -e default python tools/partial_nest_envelope.py [OUT.json]

Runs the three envelope cases of :mod:`woof.verify.partial_nest_envelope`
(rest, warm bubble, gravity wave) each three ways -- parent alone, parent +
partial two-way child, all-fine -- and two comparison rows on the bubble
(the interface budget through D2's dmdt_adj instead of the interface flux,
and a one-way child), and writes one JSON receipt with the device, the commit and
every metric.  ``tests/test_partial_nest_envelope.py`` re-runs the cases
and holds them inside the envelope this receipt measured.
Default output: tests/data/receipts/partial-nest-d3/envelope-<backend>.json.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _device() -> dict:
    import os
    backend = os.environ.get("WOOF_BACKEND", "cuda")
    out = {"backend": backend, "platform": platform.platform()}
    try:
        import cupy as cp
        if backend == "cuda":
            props = cp.cuda.runtime.getDeviceProperties(0)
            out["device"] = props["name"].decode() if isinstance(props["name"], bytes) \
                else str(props["name"])
    except Exception as exc:          # Metal facade: no CUDA runtime
        out["device_note"] = type(exc).__name__
    if backend == "mlx":
        out["device"] = platform.processor() or "Apple silicon"
    return out


def main(argv) -> int:
    from woof.verify.partial_nest_envelope import CASES, GEOMETRY, run_case
    dev = _device()
    out_path = Path(argv[1]) if len(argv) > 1 else (
        REPO / "tests/data/receipts/partial-nest-d3" / f"envelope-{dev['backend']}.json")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                            text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                cwd=REPO, capture_output=True, text=True).stdout.strip())
    receipt = {
        "schema": "woof-partial-nest-envelope-v1",
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": commit, "tree_dirty": dirty, **dev,
        "geometry": GEOMETRY,
        "cases": {},
        "variants": {},
    }
    for name in CASES:
        t0 = time.perf_counter()
        receipt["cases"][name] = run_case(name)
        print(f"{name}: {time.perf_counter() - t0:.1f} s", flush=True)

    def adj(nest):
        nest.mass_route = "adj"

    receipt["variants"]["bubble_mass_route_adj"] = run_case(
        "bubble", configure=adj, timing=False)
    receipt["variants"]["bubble_one_way"] = run_case(
        "bubble", band_overrides={"feedback": 0}, timing=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
