"""Run a built E3SM coupler oracle on every case and write its fixture.

    python tools/e3sm_coupler_oracle/make_fixture.py BUILD_DIR [--oracle merge|coupling|seaice|frac|flux|ocnalb|icemerge|accum|interval] [--out DIR]

BUILD_DIR holds build.sh's binaries. ``--oracle merge`` (the default) runs
``run_ocean_merge`` and writes ``oracle-r8.npz`` (``<case>/call<i>``: x2o's
rAttr after each call, ``(lsize, nflds)``) into
``tests/data/oracles/e3sm_ocean_merge``; ``--oracle coupling`` runs
``run_ocean_coupling`` and writes ``oracle-r8.npz`` (``<case>/o2x`` after the
export and ``<case>/<pool>/<name>`` for every array after the import and
export) into ``tests/data/oracles/e3sm_ocean_coupling``; ``--oracle seaice``
runs ``run_seaice_coupling`` and writes ``<case>/i2x``, ``<case>/<pool>/<name>``
and ``<case>/trig`` (the stress rotation's sine and cosine words) into
``tests/data/oracles/e3sm_seaice_coupling``; ``--oracle frac`` runs
``run_seq_frac`` and writes ``<case>/fractions_i``, ``_o``, ``_a`` and
``<case>/reported`` (the stages ``seq_frac_check`` reported without
stopping) into ``tests/data/oracles/e3sm_seq_frac``; ``--oracle flux`` runs
``run_flux_atmocn`` and writes ``<case>/xao`` into
``tests/data/oracles/e3sm_flux_atmocn``; ``--oracle ocnalb`` runs
``run_flux_ocnalb`` and writes ``<case>/call<i>/xao`` and ``/fractions`` into
``tests/data/oracles/e3sm_ocnalb``; ``--oracle icemerge`` runs
``run_ice_merge`` and writes ``<case>/call<i>`` (x2i after each call) into
``tests/data/oracles/e3sm_ice_merge``; ``--oracle accum`` runs
``run_ocn_accum`` and writes ``<case>/op<i>/acc``, ``/x2o`` and
``<case>/counts`` into ``tests/data/oracles/e3sm_ocn_accum``; ``--oracle
interval`` runs ``run_coupler_interval`` and writes ``<case>/step<i>/xao``,
``/acc``, ``/x2o``, ``/x2i``, ``/fractions_o``, ``_i``, ``_a`` and
``<case>/counts`` into ``tests/data/oracles/e3sm_coupler_interval``. Each writes
``aborts.json`` (each case that must stop: the line E3SM or MCT writes as it
stops) and ``inputs.sha256`` (every case's input stream). ``--out`` names
another directory.
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

from woof.verify import e3sm_coupler_interval_oracle as ci  # noqa: E402
from woof.verify import e3sm_flux_atmocn_oracle as fx  # noqa: E402
from woof.verify import e3sm_ice_merge_oracle as im  # noqa: E402
from woof.verify import e3sm_ocean_coupling_oracle as oc  # noqa: E402
from woof.verify import e3sm_ocean_merge_oracle as oo  # noqa: E402
from woof.verify import e3sm_ocn_accum_oracle as ac  # noqa: E402
from woof.verify import e3sm_ocnalb_oracle as oa  # noqa: E402
from woof.verify import e3sm_seaice_coupling_oracle as si  # noqa: E402
from woof.verify import e3sm_seq_frac_oracle as sf  # noqa: E402

ORACLES = {"merge": (oo, "run_ocean_merge"), "coupling": (oc, "run_ocean_coupling"),
           "seaice": (si, "run_seaice_coupling"), "frac": (sf, "run_seq_frac"),
           "flux": (fx, "run_flux_atmocn"), "ocnalb": (oa, "run_flux_ocnalb"),
           "icemerge": (im, "run_ice_merge"), "accum": (ac, "run_ocn_accum"),
           "interval": (ci, "run_coupler_interval")}


def run_case(binary: Path, case: dict, oo=oo) -> tuple[int, object, str]:
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
        src.write_bytes(oo.input_bytes(case))
        done = subprocess.run([str(binary), str(src), str(dst)], capture_output=True, text=True)
        # MCT's die ends through mpi-serial's MPI_Abort, which exits with status 0:
        # a stop is a nonzero status or a FATAL line with no calls written
        if done.returncode != 0 or ("FATAL" in done.stderr and (not dst.exists() or not dst.stat().st_size)):
            return done.returncode or -1, None, done.stderr
        outputs = oo.read_outputs(dst.read_bytes(), case)
        if hasattr(oo, "read_log"):
            outputs.update(oo.read_log(done.stderr))
        return 0, outputs, done.stderr


def abort_line(stderr: str) -> str:
    for line in stderr.splitlines():
        if "shr_sys_abort" in line or "FATAL" in line:
            return line.strip()
    raise RuntimeError(f"no abort line in {stderr!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("--oracle", choices=sorted(ORACLES), default="merge")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    oo, name = ORACLES[args.oracle]
    args.out = args.out or oo.FIXTURE_DIR
    binary = args.build_dir / name
    args.out.mkdir(parents=True, exist_ok=True)
    arrays, aborts, digests = {}, {}, []
    for case in oo.cases():
        status, outputs, stderr = run_case(binary, case, oo)
        if case["aborts"]:
            if status == 0:
                raise RuntimeError(f"{case['name']} was meant to stop and did not")
            aborts[case["name"]] = getattr(oo, "abort_line", abort_line)(stderr)
        else:
            if status != 0:
                raise RuntimeError(f"{case['name']} stopped ({status}): {stderr}")
            if isinstance(outputs, dict):
                for key, value in outputs.items():
                    arrays[f"{case['name']}/{key}"] = value
            else:
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
