"""Write the partial-depth nest (Phases D3-D4) envelope receipt.

    pixi run -e mlx python tools/partial_nest_envelope.py [OUT.json]
    pixi run -e default python tools/partial_nest_envelope.py [OUT.json]

Runs every envelope case of :mod:`woof.verify.partial_nest_envelope` (rest,
warm bubble, gravity wave, the moist bubble with radiation, and the
two-band chain's rest and gravity wave) each three ways -- root alone, root
+ partial two-way band(s), all-fine -- and comparison rows: the bubble with
the interface budget through D2's dmdt_adj and one-way, the gravity wave
one-way and in D3's configuration (upwinded interface face, mu/u/v/theta
feedback: the "before"), and the chain's gravity wave one-way.  Writes one
JSON receipt with the device, the commit and every metric;
``tests/test_partial_nest_envelope.py`` re-runs the cases and holds them
inside the envelope this receipt measured.
Default output: tests/data/receipts/partial-nest-d4/envelope-<backend>.json.
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
    from woof.verify.partial_nest_envelope import (CASES, CHAIN_GEOMETRY, GEOMETRY,
                                                   run_case)
    dev = _device()
    out_path = Path(argv[1]) if len(argv) > 1 else (
        REPO / "tests/data/receipts/partial-nest-d4" / f"envelope-{dev['backend']}.json")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                            text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                cwd=REPO, capture_output=True, text=True).stdout.strip())
    receipt = {
        "schema": "woof-partial-nest-envelope-v2",
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": commit, "tree_dirty": dirty, **dev,
        "geometry": GEOMETRY,
        "chain_geometry": {k: (list(v) if isinstance(v, tuple) else v)
                           for k, v in CHAIN_GEOMETRY.items()},
        "cases": {},
        "variants": {},
    }
    for name in CASES:
        t0 = time.perf_counter()
        receipt["cases"][name] = run_case(name)
        print(f"{name}: {time.perf_counter() - t0:.1f} s", flush=True)

    def adj(nest):
        nest.mass_route = "adj"

    def d3(nest):
        nest.face_scheme = "upwind"

    variants = {
        "bubble_mass_route_adj": dict(case="bubble", configure=adj),
        "bubble_one_way": dict(case="bubble", band_overrides={"feedback": 0}),
        "gravity_wave_one_way": dict(case="gravity_wave", band_overrides={"feedback": 0}),
        "gravity_wave_d3": dict(case="gravity_wave", configure=d3,
                                band_overrides={"feedback_fields": ("mu", "u", "v", "t")}),
        "chain_gravity_wave_one_way": dict(case="chain_gravity_wave",
                                           band_overrides={"feedback": 0}),
    }
    for name, kw in variants.items():
        t0 = time.perf_counter()
        receipt["variants"][name] = run_case(kw.pop("case"), timing=False, **kw)
        print(f"{name}: {time.perf_counter() - t0:.1f} s", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(receipt, indent=1, sort_keys=True, default=list) + "\n")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
