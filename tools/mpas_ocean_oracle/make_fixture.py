"""Run a built MPAS-Ocean oracle on every case and write the fixture.

    pixi run -e default python tools/mpas_ocean_oracle/make_fixture.py BUILD_DIR [--piece P] [--out DIR]

BUILD_DIR holds the binaries from build.sh.  ``--piece tend`` (the default)
runs ``run_oracle_{r4,r8}`` into ``tests/data/oracles/mpas_ocean_tend``;
``--piece pgrad_hmix`` runs ``run_pgrad_hmix_{r4,r8}`` into
``tests/data/oracles/mpas_ocean_pgrad_hmix``; ``--piece tendency`` runs
``run_tendency_{r4,r8}`` into ``tests/data/oracles/mpas_ocean_tendency`` (a
case that lists its precisions is run at those only); ``--piece eos`` runs
``run_eos_{r4,r8}`` into ``tests/data/oracles/mpas_ocean_eos``; ``--piece
diagnostics`` runs ``run_diagnostics_{r4,r8}`` into
``tests/data/oracles/mpas_ocean_diagnostics`` and ``--piece tracer_advection``
``run_tracer_advection_{r4,r8}`` into ``tests/data/oracles/mpas_ocean_tracer_advection``,
``--piece tracer_hmix`` ``run_tracer_hmix_{r4,r8}`` into ``tests/data/oracles/mpas_ocean_tracer_hmix``,
``--piece tend_tracer`` ``run_tend_tracer_{r4,r8}`` into ``tests/data/oracles/mpas_ocean_tend_tracer``,
``--piece vmix`` ``run_vmix_{r4,r8}`` into ``tests/data/oracles/mpas_ocean_vmix``,
``--piece vert_transport`` ``run_vert_transport_{r4,r8}`` into ``tests/data/oracles/mpas_ocean_vert_transport``,
``--piece btr_subcycle`` ``run_btr_subcycle_{r4,r8}`` into ``tests/data/oracles/mpas_ocean_btr_subcycle``,
``--piece split_step`` ``run_split_step_{r4,r8}`` into ``tests/data/oracles/mpas_ocean_split_step``,
``--piece forward`` ``run_forward_{r4,r8}`` into ``tests/data/oracles/mpas_ocean_forward``
(their cases are built per precision).  A piece's ``EXTRA_PRECISIONS`` (the
tendency's ``r8-glibc239``: its RKIND 8 bulk wind stress) add fixtures run by
the binary they name.  Writes ``oracle-r4.npz`` and
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

from woof.verify import (  # noqa: E402
    mpas_ocean_btr_subcycle_oracle, mpas_ocean_diagnostics_oracle, mpas_ocean_forward_oracle,
    mpas_ocean_split_step_oracle, mpas_ocean_eos_oracle, mpas_ocean_pgrad_hmix_oracle, mpas_ocean_tend_oracle,
    mpas_ocean_tend_tracer_oracle, mpas_ocean_tendency_oracle, mpas_ocean_tracer_advection_oracle,
    mpas_ocean_tracer_hmix_oracle, mpas_ocean_vert_transport_oracle, mpas_ocean_vmix_oracle)

#: piece -> (case module, binary stem)
PIECES = {"tend": (mpas_ocean_tend_oracle, "run_oracle"),
          "pgrad_hmix": (mpas_ocean_pgrad_hmix_oracle, "run_pgrad_hmix"),
          "tendency": (mpas_ocean_tendency_oracle, "run_tendency"),
          "eos": (mpas_ocean_eos_oracle, "run_eos"),
          "diagnostics": (mpas_ocean_diagnostics_oracle, "run_diagnostics"),
          "tracer_advection": (mpas_ocean_tracer_advection_oracle, "run_tracer_advection"),
          "tracer_hmix": (mpas_ocean_tracer_hmix_oracle, "run_tracer_hmix"),
          "tend_tracer": (mpas_ocean_tend_tracer_oracle, "run_tend_tracer"),
          "vmix": (mpas_ocean_vmix_oracle, "run_vmix"),
          "vert_transport": (mpas_ocean_vert_transport_oracle, "run_vert_transport"),
          "btr_subcycle": (mpas_ocean_btr_subcycle_oracle, "run_btr_subcycle"),
          "split_step": (mpas_ocean_split_step_oracle, "run_split_step"),
          "forward": (mpas_ocean_forward_oracle, "run_forward")}


def run_case(oo, binary: Path, case: dict, precision: str) -> dict[str, np.ndarray]:
    per_precision = getattr(oo, "CASES_PER_PRECISION", False)
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
        src.write_bytes(oo.input_bytes(case) if per_precision else oo.input_bytes(case, precision))
        subprocess.run([str(binary), str(src), str(dst)], check=True)
        if per_precision:
            return oo.read_outputs(dst.read_bytes(), case)
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
    extra = getattr(oo, "EXTRA_PRECISIONS", {})
    for precision in (*oo.PRECISIONS, *extra):
        arrays: dict[str, np.ndarray] = {}
        per_precision = getattr(oo, "CASES_PER_PRECISION", False)
        for case in (oo.cases(precision) if per_precision else oo.cases()):
            if precision not in case.get("precisions", oo.PRECISIONS):
                continue
            binary = args.build_dir / f"{stem}_{extra.get(precision, precision)}"
            for field, value in run_case(oo, binary, case, precision).items():
                arrays[f"{case['name']}/{field}"] = value
            digest = oo.input_digest(case) if per_precision else oo.input_digest(case, precision)
            digests.append(f"{digest}  {case['name']}/{precision}")
        np.savez_compressed(args.out / f"oracle-{precision}.npz", **arrays)
    (args.out / "inputs.sha256").write_text("\n".join(digests) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
