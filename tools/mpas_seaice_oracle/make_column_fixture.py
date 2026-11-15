"""Run the built MPAS-Seaice column-package oracle on every case and write the fixture.

    pixi run -e default python tools/mpas_seaice_oracle/make_column_fixture.py BUILD_DIR [--out DIR]
        [--cases init|therm]

BUILD_DIR holds ``run_column_r8`` and ``run_init_thermo_r8`` from build.sh.
Writes into ``tests/data/oracles/mpas_seaice_column`` by default:
``oracle-r8.npz`` (``<case>/<pool>/<field>``, the fields each case changed),
``column-logs.json`` (each case's tracer object and log lines),
``init-thermo-r8.npy`` (colpkg_init_thermo's profiles for 1 to
``THERMO_LAYERS`` ice layers) and ``inputs.sha256`` (every case's
input-stream digest). ``--cases therm`` runs the vertical thermodynamics'
cases (``VT_CASES``) into ``tests/data/oracles/mpas_seaice_column_therm``
instead, without the profiles.
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

from woof.verify import mpas_seaice_column_oracle as co  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--cases", choices=("init", "therm"), default="init")
    args = parser.parse_args()
    therm = args.cases == "therm"
    args.out = args.out or (co.VT_FIXTURE_DIR if therm else co.FIXTURE_DIR)
    args.out.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    logs, digests = {}, []
    for case in co.cases("r8", co.VT_CASES if therm else co.CASES):
        outputs, log = co.run_oracle(args.build_dir, case)
        for key, value in co.changed_fields(case, outputs).items():
            arrays[f"{case['name']}/{key}"] = value
        logs[case["name"]] = log
        digests.append(f"{co.input_digest(case)}  {case['name']}/r8")
    np.savez_compressed(args.out / "oracle-r8.npz", **arrays)
    (args.out / "column-logs.json").write_text(json.dumps(logs, indent=1, sort_keys=True) + "\n",
                                               encoding="utf-8")
    if not therm:
        np.save(args.out / "init-thermo-r8.npy", co.oracle_thermo_profiles(args.build_dir))
    (args.out / "inputs.sha256").write_text("\n".join(digests) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
