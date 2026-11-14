"""Run the built MPAS-Seaice time-step oracle on the cases with advection on and write the fixture.

    pixi run -e default python tools/mpas_seaice_oracle/make_step_remap_fixture.py BUILD_DIR [--out DIR]

BUILD_DIR holds ``run_seaice_step_r4`` and ``run_seaice_step_r8`` from
build.sh. Writes ``oracle-r4.npz`` and ``oracle-r8.npz`` (``<case>/<pool>/<field>``,
the fields whose words the run changed, the remap's triangle values left out)
and ``inputs.sha256`` into ``tests/data/oracles/mpas_seaice_step_remap`` by default.
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

from woof.verify import mpas_seaice_remap_oracle as ro  # noqa: E402
from woof.verify import mpas_seaice_step_remap_oracle as sr  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("--out", type=Path, default=sr.FIXTURE_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    digests = []
    for precision in sr.PRECISIONS:
        arrays: dict[str, np.ndarray] = {}
        for case in sr.cases(precision):
            with tempfile.TemporaryDirectory() as tmp:
                src, dst = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
                src.write_bytes(sr.input_bytes(case))
                subprocess.run([str(args.build_dir / f"run_seaice_step_{precision}"), str(src), str(dst)],
                               check=True)
                outputs = sr.read_outputs(dst.read_bytes(), case)
            for key, value in ro.changed_fields(case, outputs).items():
                arrays[f"{case['name']}/{key}"] = value
            digests.append(f"{sr.input_digest(case)}  {case['name']}/{precision}")
        np.savez_compressed(args.out / f"oracle-{precision}.npz", **arrays)
    (args.out / "inputs.sha256").write_text("\n".join(digests) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
