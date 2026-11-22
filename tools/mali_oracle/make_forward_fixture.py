"""Run the built MALI forward-mode oracle on every case and write the fixture.

    pixi run -e default python tools/mali_oracle/make_forward_fixture.py BUILD_DIR [--out DIR]

BUILD_DIR holds ``run_forward_r4`` and ``run_forward_r8`` from build_forward.sh.
Writes ``oracle-r4.npz`` and ``oracle-r8.npz`` (every case's outputs at each
precision, ``<case>/<pool>/<field>``, with the error words and
``clock/now``) and ``inputs.sha256`` (the digest of every case's input
stream, named ``<case>`` at RKIND 4 and ``<case>/r8`` at RKIND 8) into
``tests/data/oracles/mali_forward`` by default. A case whose clock,
initialisation or run reports an error is not written.
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

from woof.verify import mali_forward_oracle as fo  # noqa: E402


def run_case(binary: Path, case: dict) -> dict[str, np.ndarray]:
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
        src.write_bytes(fo.input_bytes(case))
        subprocess.run([str(binary), str(src), str(dst)], check=True, cwd=tmp, stdout=subprocess.DEVNULL)
        return fo.read_outputs(dst.read_bytes(), case)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("--out", type=Path, default=fo.FIXTURE_DIR)
    args = parser.parse_args()
    arrays: dict[str, dict[str, np.ndarray]] = {"r4": {}, "r8": {}}
    digests = []
    for precision in arrays:
        for case in fo.cases(precision):
            out = run_case(args.build_dir / f"run_forward_{precision}", case)
            errors = {k: int(out[k][0]) for k in ("errClock", "errInit", "errRun") if out[k][0]}
            if errors:
                raise SystemExit(f"{case['name']} ({precision}): the oracle reports {errors}")
            for field, value in out.items():
                arrays[precision][f"{case['name']}/{field}"] = value
            digests.append(f"{fo.input_digest(case)}  {case['name']}" + ("" if precision == "r4" else "/r8"))
    args.out.mkdir(parents=True, exist_ok=True)
    for precision, named in arrays.items():
        np.savez_compressed(args.out / f"oracle-{precision}.npz", **named)
    (args.out / "inputs.sha256").write_text("\n".join(digests) + "\n", encoding="utf-8")
    print(f"wrote {args.out} (oracle-r4.npz, oracle-r8.npz: {sum(map(len, arrays.values()))} arrays)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
