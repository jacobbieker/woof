"""Run the built MPAS-Ocean tendency oracle on every case and write the fixture.

    pixi run -e default python tools/mpas_ocean_oracle/make_fixture.py BUILD_DIR [--out DIR]

BUILD_DIR holds ``run_oracle_r4`` and ``run_oracle_r8`` from build.sh.  Writes
``oracle-r4.npz`` and ``oracle-r8.npz`` (``<case>/<field>``) and
``inputs.sha256`` (every case's input-stream digest at each precision) into
``tests/data/oracles/mpas_ocean_tend`` by default.
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

from woof.verify import mpas_ocean_tend_oracle as oo  # noqa: E402


def run_case(binary: Path, case: dict, precision: str) -> dict[str, np.ndarray]:
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
        src.write_bytes(oo.input_bytes(case, precision))
        subprocess.run([str(binary), str(src), str(dst)], check=True)
        return oo.read_outputs(dst.read_bytes(), case, precision)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("--out", type=Path, default=oo.FIXTURE_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    digests = []
    for precision in oo.PRECISIONS:
        arrays: dict[str, np.ndarray] = {}
        for case in oo.cases():
            for field, value in run_case(args.build_dir / f"run_oracle_{precision}", case, precision).items():
                arrays[f"{case['name']}/{field}"] = value
            digests.append(f"{oo.input_digest(case, precision)}  {case['name']}/{precision}")
        np.savez_compressed(args.out / f"oracle-{precision}.npz", **arrays)
    (args.out / "inputs.sha256").write_text("\n".join(digests) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
