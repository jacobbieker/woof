"""Run the built MPAS-Seaice time-step oracle on every case and write the fixture.

    pixi run -e default python tools/mpas_seaice_oracle/make_step_fixture.py BUILD_DIR [--out DIR]

BUILD_DIR holds ``run_seaice_step_r4`` and ``run_seaice_step_r8`` from
build.sh. Writes ``oracle-r4.npz`` and ``oracle-r8.npz`` (``<case>/<pool>/<field>``)
and ``inputs.sha256`` (every case's input-stream digest at each precision)
into ``tests/data/oracles/mpas_seaice_step`` by default.
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

from woof.verify import mpas_seaice_step_oracle as so  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("--out", type=Path, default=so.FIXTURE_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    digests = []
    for precision in so.PRECISIONS:
        arrays: dict[str, np.ndarray] = {}
        for case in so.cases(precision):
            with tempfile.TemporaryDirectory() as tmp:
                src, dst = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
                src.write_bytes(so.input_bytes(case))
                subprocess.run([str(args.build_dir / f"run_seaice_step_{precision}"), str(src), str(dst)],
                               check=True)
                for field, value in so.read_outputs(dst.read_bytes(), case).items():
                    arrays[f"{case['name']}/{field}"] = value
            digests.append(f"{so.input_digest(case)}  {case['name']}/{precision}")
        np.savez_compressed(args.out / f"oracle-{precision}.npz", **arrays)
    (args.out / "inputs.sha256").write_text("\n".join(digests) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
