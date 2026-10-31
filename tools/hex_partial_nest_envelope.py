"""Write the hex partial-depth band (Phase D5) envelope receipt.

    pixi run -e default python tools/hex_partial_nest_envelope.py [OUT.json]
    pixi run -e mlx python tools/hex_partial_nest_envelope.py [OUT.json]

Runs every case of :mod:`woof.verify.hex_partial_nest_envelope` (rest,
gravity wave, warm bubble) three ways -- the parent alone, the parent with
its partial two-way band, all-fine -- plus comparison rows: the gravity wave
and the bubble one-way (no feedback), and the gravity wave with D3's
upwinded interface face (the "before" D4 measured).  Writes one JSON receipt
with the device, the commit and every metric;
components/hex/tests/test_partial_nest_envelope_v841.py re-runs the cases
and holds them inside the envelope this receipt measured.  The meshes come
from tools/rustwx's rw_mpas_mesh (WOOF_HEX_PN_MESH_DIR caches them).
Default output: tests/data/receipts/partial-nest-d5/envelope-<backend>.json.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

#: Comparison rows: (name, case, band overrides).
VARIANTS = (
    ("gravity_wave_one_way", "gravity_wave", {"feedback": 0}),
    ("gravity_wave_upwind_face", "gravity_wave", {"face_scheme": "upwind"}),
    ("bubble_one_way", "bubble", {"feedback": 0}),
)


def _device() -> dict:
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
    from woof.hex import numerics
    out["hex_numerics"] = numerics.mode()
    return out


def main(argv) -> int:
    from woof.verify.hex_partial_nest_envelope import (CASES, GEOMETRY, MIXING,
                                                       MOIST_REFUSAL, run_case)
    dev = _device()
    out_path = Path(argv[1]) if len(argv) > 1 else (
        REPO / "tests/data/receipts/partial-nest-d5" / f"envelope-{dev['backend']}.json")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                            text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                cwd=REPO, capture_output=True, text=True).stdout.strip())
    receipt = {
        "schema": "woof-hex-partial-nest-envelope-v1",
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": commit, "tree_dirty": dirty, **dev,
        "geometry": {k: (list(v) if isinstance(v, tuple) else v) for k, v in GEOMETRY.items()},
        "mixing": MIXING,
        "refused": {"moist_bubble": MOIST_REFUSAL},
        "cases": {},
        "variants": {},
    }
    for name in CASES:
        t0 = time.perf_counter()
        receipt["cases"][name] = run_case(name)
        print(f"{name}: {time.perf_counter() - t0:.1f} s", flush=True)
    for name, case, overrides in VARIANTS:
        t0 = time.perf_counter()
        receipt["variants"][name] = run_case(case, band_overrides=overrides, timing=False)
        print(f"{name}: {time.perf_counter() - t0:.1f} s", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
