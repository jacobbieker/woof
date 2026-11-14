"""Run the built MPAS-Seaice incremental-remap oracle on every case and write the fixture.

    pixi run -e default python tools/mpas_seaice_oracle/make_remap_fixture.py BUILD_DIR [--out DIR]

BUILD_DIR holds ``run_incremental_remap_r4`` and ``run_incremental_remap_r8``
from build.sh. Writes into ``tests/data/oracles/mpas_seaice_remap`` by default:

- ``oracle-r4.npz`` and ``oracle-r8.npz``: per case (``<case>/<pool>/<field>``)
  every output field whose words differ from the input's;
- ``remap-logs.json``: per case and precision the ordered tracer list, the
  warnings the routines logged and stage 4's error code;
- ``inputs.sha256``: every case's input-stream digest at each precision.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from woof.verify import mpas_seaice_remap_oracle as ro  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("--out", type=Path, default=ro.FIXTURE_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    digests, logs = [], {}
    for precision in ro.PRECISIONS:
        arrays: dict[str, np.ndarray] = {}
        for case in ro.cases(precision):
            outputs, log = ro.run_oracle(args.build_dir, case)
            for key, value in ro.changed_fields(case, outputs).items():
                arrays[f"{case['name']}/{key}"] = value
            logs.setdefault(case["name"], {})[precision] = log
            digests.append(f"{ro.input_digest(case)}  {case['name']}/{precision}")
        np.savez_compressed(args.out / f"oracle-{precision}.npz", **arrays)
    (args.out / "remap-logs.json").write_text(json.dumps(logs, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    (args.out / "inputs.sha256").write_text("\n".join(digests) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
