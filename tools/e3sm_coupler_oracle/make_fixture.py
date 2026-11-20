"""Run the built E3SM ocean-merge oracle on every case and write the fixture.

    python tools/e3sm_coupler_oracle/make_fixture.py BUILD_DIR [--out DIR]

BUILD_DIR holds ``run_ocean_merge`` from build.sh. Writes ``oracle-r8.npz``
(``<case>/call<i>``: x2o's rAttr after each call, ``(lsize, nflds)``),
``aborts.json`` (each case that must stop: its exit status is nonzero and the
line E3SM or MCT writes) and ``inputs.sha256`` (every case's input stream),
into ``tests/data/oracles/e3sm_ocean_merge`` unless ``--out`` names another
directory.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from woof.verify import e3sm_ocean_merge_oracle as oo  # noqa: E402


def run_case(binary: Path, case: dict) -> tuple[int, list[np.ndarray] | None, str]:
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
        src.write_bytes(oo.input_bytes(case))
        done = subprocess.run([str(binary), str(src), str(dst)], capture_output=True, text=True)
        if done.returncode != 0:
            return done.returncode, None, done.stderr
        return 0, oo.read_outputs(dst.read_bytes(), case), done.stderr


def abort_line(stderr: str) -> str:
    for line in stderr.splitlines():
        if "shr_sys_abort" in line or "FATAL" in line:
            return line.strip()
    raise RuntimeError(f"no abort line in {stderr!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("--out", type=Path, default=oo.FIXTURE_DIR)
    args = parser.parse_args()
    binary = args.build_dir / "run_ocean_merge"
    args.out.mkdir(parents=True, exist_ok=True)
    arrays, aborts, digests = {}, {}, []
    for case in oo.cases():
        status, outputs, stderr = run_case(binary, case)
        if case["aborts"]:
            if status == 0:
                raise RuntimeError(f"{case['name']} was meant to stop and did not")
            aborts[case["name"]] = abort_line(stderr)
        else:
            if status != 0:
                raise RuntimeError(f"{case['name']} stopped ({status}): {stderr}")
            for i, x2o in enumerate(outputs):
                arrays[f"{case['name']}/call{i}"] = x2o
        digests.append(f"{oo.input_digest(case)}  {case['name']}")
    np.savez(args.out / "oracle-r8.npz", **arrays)
    (args.out / "aborts.json").write_text(json.dumps(aborts, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (args.out / "inputs.sha256").write_text("\n".join(digests) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
