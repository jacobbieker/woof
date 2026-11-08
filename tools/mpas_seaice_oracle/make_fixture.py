"""Run the built MPAS-Seaice EVP oracle on every case and write the fixture.

    pixi run -e default python tools/mpas_seaice_oracle/make_fixture.py BUILD_DIR [--out DIR]

BUILD_DIR holds ``run_oracle_r4`` and ``run_oracle_r8`` from build.sh.
Writes ``oracle-r4.npz`` and ``oracle-r8.npz`` (every case's outputs at that
precision, ``<case>/<field>``) and ``inputs.sha256`` (the digest of every
case's input stream, ``<case>/<precision>``) into
``tests/data/oracles/mpas_seaice_evp`` by default.
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

from woof.verify import mpas_seaice_evp_oracle as so  # noqa: E402


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
    args.out.mkdir(parents=True, exist_ok=True)
    digests = []
    for precision in so.PRECISIONS:
        binary = args.build_dir / f"run_oracle_{precision}"
        arrays: dict[str, np.ndarray] = {}
        for case in so.cases(precision):
            for field, value in run_case(binary, case).items():
                arrays[f"{case['name']}/{field}"] = value
            digests.append(f"{so.input_digest(case)}  {case['name']}/{precision}")
        np.savez_compressed(args.out / f"oracle-{precision}.npz", **arrays)
        print(f"wrote {args.out / f'oracle-{precision}.npz'} ({len(arrays)} arrays)")
    (args.out / "inputs.sha256").write_text("\n".join(digests) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
