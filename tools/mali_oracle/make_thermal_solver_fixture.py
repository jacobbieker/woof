"""Run the built MALI thermal solver oracle on every case and write the fixture.

    pixi run -e default python tools/mali_oracle/make_thermal_solver_fixture.py BUILD_DIR [--out DIR]

BUILD_DIR holds ``run_thermal_solver_r4`` and ``run_thermal_solver_r8`` from build.sh.  Writes
``oracle-r4.npz`` and ``oracle-r8.npz`` (every case's outputs at each
precision, ``<case>/<field>``) and ``inputs.sha256`` (the digest of every
case's input stream, named ``<case>`` at RKIND 4 and ``<case>/r8`` at
RKIND 8) into ``tests/data/oracles/mali_thermal_solver`` by default.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from woof.verify import mali_thermal_solver_oracle as so  # noqa: E402


def run_case(binary: Path, case: dict) -> dict[str, np.ndarray]:
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
        src.write_bytes(so.input_bytes(case))
        subprocess.run([str(binary), str(src), str(dst)], check=True)
        return so.read_outputs(dst.read_bytes(), case)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("--out", type=Path, default=so.FIXTURE_DIR)
    args = parser.parse_args()
    arrays: dict[str, dict[str, np.ndarray]] = {"r4": {}, "r8": {}}
    digests = []
    for precision in ("r4", "r8"):
        binary = args.build_dir / f"run_thermal_solver_{precision}"
        for case in so.cases(precision):
            for field, value in run_case(binary, case).items():
                arrays[precision][f"{case['name']}/{field}"] = value
            # the RKIND 4 digests keep their bare case names
            digests.append(f"{so.input_digest(case)}  {case['name']}" + ("/r8" if precision == "r8" else ""))
    args.out.mkdir(parents=True, exist_ok=True)
    for precision, named in arrays.items():
        np.savez_compressed(args.out / f"oracle-{precision}.npz", **named)
    (args.out / "inputs.sha256").write_text("\n".join(digests) + "\n", encoding="utf-8")
    print(f"wrote {args.out} (oracle-r4.npz, oracle-r8.npz: {sum(map(len, arrays.values()))} arrays)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
