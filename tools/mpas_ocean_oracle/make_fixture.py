"""Run a built MPAS-Ocean oracle on every case and write the fixture.

    pixi run -e default python tools/mpas_ocean_oracle/make_fixture.py BUILD_DIR [--piece P] [--out DIR]

BUILD_DIR holds the binaries from build.sh.  ``--piece tend`` (the default)
runs ``run_oracle_{r4,r8}`` into ``tests/data/oracles/mpas_ocean_tend``;
``--piece pgrad_hmix`` runs ``run_pgrad_hmix_{r4,r8}`` into
``tests/data/oracles/mpas_ocean_pgrad_hmix``.  Writes ``oracle-r4.npz`` and
``oracle-r8.npz`` (``<case>/<field>``) and ``inputs.sha256`` (every case's
input-stream digest at each precision).
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

from woof.verify import mpas_ocean_pgrad_hmix_oracle, mpas_ocean_tend_oracle  # noqa: E402

#: piece -> (case module, binary stem)
PIECES = {"tend": (mpas_ocean_tend_oracle, "run_oracle"),
          "pgrad_hmix": (mpas_ocean_pgrad_hmix_oracle, "run_pgrad_hmix")}


def run_case(oo, binary: Path, case: dict, precision: str) -> dict[str, np.ndarray]:
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
        src.write_bytes(oo.input_bytes(case, precision))
        subprocess.run([str(binary), str(src), str(dst)], check=True)
        return oo.read_outputs(dst.read_bytes(), case, precision)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("--piece", choices=sorted(PIECES), default="tend")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    oo, stem = PIECES[args.piece]
    args.out = args.out or oo.FIXTURE_DIR
    args.out.mkdir(parents=True, exist_ok=True)
    digests = []
    for precision in oo.PRECISIONS:
        arrays: dict[str, np.ndarray] = {}
        for case in oo.cases():
            for field, value in run_case(oo, args.build_dir / f"{stem}_{precision}", case, precision).items():
                arrays[f"{case['name']}/{field}"] = value
            digests.append(f"{oo.input_digest(case, precision)}  {case['name']}/{precision}")
        np.savez_compressed(args.out / f"oracle-{precision}.npz", **arrays)
    (args.out / "inputs.sha256").write_text("\n".join(digests) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
