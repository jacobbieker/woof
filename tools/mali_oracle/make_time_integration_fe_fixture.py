"""Run the built MALI forward-Euler step oracle on every case and write the fixture.

    pixi run -e default python tools/mali_oracle/make_time_integration_fe_fixture.py BUILD_DIR [--out DIR]

BUILD_DIR holds ``run_time_integration_fe_r4`` and ``run_time_integration_fe_r8`` from build.sh.
Writes ``oracle-r4.npz`` and ``oracle-r8.npz`` (every case's outputs at each
precision, ``<case>/<pool>/<field>``), ``oracle-r8-glibc239.npz`` (the RKIND 8
cases the opt-in glibc 2.39 sin/cos/atan transcription lifts, nothing
refused) and ``inputs.sha256`` (the digest of every case's input stream,
named ``<case>`` at RKIND 4, ``<case>/r8`` at RKIND 8 and
``<case>/r8-glibc239``) into ``tests/data/oracles/mali_time_integration_fe``
by default.
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

from woof.verify import mali_time_integration_fe_oracle as fo  # noqa: E402


def run_case(binary: Path, case: dict) -> dict[str, np.ndarray]:
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
        src.write_bytes(fo.input_bytes(case))
        subprocess.run([str(binary), str(src), str(dst)], check=True)
        return fo.read_outputs(dst.read_bytes(), case)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("--out", type=Path, default=fo.FIXTURE_DIR)
    args = parser.parse_args()
    arrays: dict[str, dict[str, np.ndarray]] = {"r4": {}, "r8": {}, fo.R8_GLIBC239: {}}
    digests = []
    for precision in arrays:
        binary = args.build_dir / f"run_time_integration_fe_{'r4' if precision == 'r4' else 'r8'}"
        for case in fo.cases(precision):
            for field, value in run_case(binary, case).items():
                arrays[precision][f"{case['name']}/{field}"] = value
            digests.append(f"{fo.input_digest(case)}  {case['name']}" + ("" if precision == "r4" else f"/{precision}"))
    args.out.mkdir(parents=True, exist_ok=True)
    for precision, named in arrays.items():
        np.savez_compressed(args.out / f"oracle-{precision}.npz", **named)
    (args.out / "inputs.sha256").write_text("\n".join(digests) + "\n", encoding="utf-8")
    print(f"wrote {args.out} (oracle-r4.npz, oracle-r8.npz, oracle-{fo.R8_GLIBC239}.npz: "
          f"{sum(map(len, arrays.values()))} arrays)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
