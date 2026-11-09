"""Run the built MALI advection oracle on every case and write the fixture.

    pixi run -e default python tools/mali_oracle/make_advection_fixture.py BUILD_DIR [--out DIR]

BUILD_DIR holds ``run_advection_r4`` from build.sh.  Writes ``oracle-r4.npz``
(every case's outputs, ``<case>/<field>``) and ``inputs.sha256`` (the digest
of every case's input stream) into ``tests/data/oracles/mali_advection`` by
default.
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

from woof.verify import mali_advection_oracle as so  # noqa: E402


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
    binary = args.build_dir / "run_advection_r4"
    arrays: dict[str, np.ndarray] = {}
    digests = []
    for case in so.cases():
        for field, value in run_case(binary, case).items():
            arrays[f"{case['name']}/{field}"] = value
        digests.append(f"{so.input_digest(case)}  {case['name']}")
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "oracle-r4.npz", **arrays)
    (args.out / "inputs.sha256").write_text("\n".join(digests) + "\n", encoding="utf-8")
    print(f"wrote {args.out / 'oracle-r4.npz'} ({len(arrays)} arrays)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
