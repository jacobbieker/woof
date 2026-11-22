"""Run the built MPAS-Seaice column-package oracle on every case and write the fixture.

    pixi run -e default python tools/mpas_seaice_oracle/make_column_fixture.py BUILD_DIR [--out DIR]
        [--cases init|therm|itd|ridge|radiation|dedd|step|coupled|oml|bgc|carry|zsal|zsal_solve|zsal_carry|zsal_solve_carry
                 |nonreactive|nonreactive_carry|nonreactive_step|zaero]

BUILD_DIR holds ``run_column_r8`` and ``run_init_thermo_r8`` from build.sh.
Writes into ``tests/data/oracles/mpas_seaice_column`` by default:
``oracle-r8.npz`` (``<case>/<pool>/<field>``, the fields each case changed),
``column-logs.json`` (each case's tracer object and log lines),
``init-thermo-r8.npy`` (colpkg_init_thermo's profiles for 1 to
``THERMO_LAYERS`` ice layers) and ``inputs.sha256`` (every case's
input-stream digest). ``--cases therm`` runs the vertical thermodynamics'
cases (``VT_CASES``) into ``tests/data/oracles/mpas_seaice_column_therm``
instead, without the profiles; ``--cases itd`` the ITD thermodynamics'
(``ITD_CASES``) into ``tests/data/oracles/mpas_seaice_column_itd``; ``--cases ridge``
the ridging's (``RIDGE_CASES``) into ``tests/data/oracles/mpas_seaice_column_ridge``; ``--cases radiation``
the shortwave's (``RADIATION_CASES``) into ``tests/data/oracles/mpas_seaice_column_radiation``;
``--cases dedd`` the delta-Eddington shortwave's (``DEDD_CASES``) into
``tests/data/oracles/mpas_seaice_column_dedd``; ``--cases step`` the column step's
(``STEP_CASES``) into ``tests/data/oracles/mpas_seaice_column_step``; ``--cases coupled``
the time step's with the column package and the velocity solver (``COUPLED_CASES``)
into ``tests/data/oracles/mpas_seaice_coupled_step``; ``--cases oml``
the ocean mixed layer's, alone and in the column step (``OML_CASES``,
``OML_STEP_CASES``), into ``tests/data/oracles/mpas_seaice_ocean_mixed_layer``;
``--cases bgc`` the column biogeochemistry's (the init and the brine
height's step, ``BGC_CASES``) into ``tests/data/oracles/mpas_seaice_bgc_brine``;
``--cases carry`` the brine fraction carried through the ITD thermodynamics,
the ridging and the steps (``BGC_CARRY_CASES``, ``BGC_CARRY_COUPLED_CASES``) into
``tests/data/oracles/mpas_seaice_bgc_brine_carry``; ``--cases zsal`` the vertical
salinity in the init, column_biogeochemistry and the vertical thermodynamics
(``BGC_ZSAL_CASES``) into ``tests/data/oracles/mpas_seaice_bgc_zsal``; ``--cases
zsal_solve`` the same under the opt-in correction ``zsal_solve``
(``BGC_ZSAL_SOLVE_CASES``) into ``tests/data/oracles/mpas_seaice_bgc_zsal_solve``,
from a build made with ``WOOF_ORACLE_PATCH=zsal_solve``; ``--cases zsal_carry``
v8.4.2's inert vertical salinity through the ITD thermodynamics, the ridging and
the steps (``BGC_ZSAL_CARRY_CASES``, ``BGC_ZSAL_CARRY_COUPLED_CASES``) into
``tests/data/oracles/mpas_seaice_bgc_zsal_carry``; ``--cases zsal_solve_carry``
the same under ``zsal_solve`` (``BGC_ZSAL_SOLVE_CARRY_CASES``,
``BGC_ZSAL_SOLVE_CARRY_COUPLED_CASES``) into
``tests/data/oracles/mpas_seaice_bgc_zsal_solve_carry``, from the patched build;
``--cases nonreactive`` the vertical tracers' nonreactive tracer in the init and
column_biogeochemistry (``BGC_NONREACTIVE_CASES``) into
``tests/data/oracles/mpas_seaice_bgc_nonreactive``; ``--cases nonreactive_carry``
the nonreactive tracer through the ITD thermodynamics and the ridging
(``BGC_NONREACTIVE_CARRY_CASES``) into
``tests/data/oracles/mpas_seaice_bgc_nonreactive_carry``; ``--cases nonreactive_step``
the nonreactive tracer through the time steps (``BGC_NONREACTIVE_STEP_CASES``) into
``tests/data/oracles/mpas_seaice_bgc_nonreactive_step``; ``--cases zaero`` the
z-aerosols in the init and column_biogeochemistry (``BGC_ZAERO_CASES``) into
``tests/data/oracles/mpas_seaice_bgc_zaero``; ``--cases zaero_carry`` the
z-aerosols through the ITD thermodynamics and the ridging
(``BGC_ZAERO_CARRY_CASES``) into ``tests/data/oracles/mpas_seaice_bgc_zaero_carry``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from woof.verify import mpas_seaice_column_oracle as co  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--cases", choices=("init", "therm", "itd", "ridge", "radiation", "dedd", "step", "coupled",
                                              "oml", "bgc", "carry", "zsal", "zsal_solve", "zsal_carry",
                                              "zsal_solve_carry", "nonreactive", "nonreactive_carry",
                                              "nonreactive_step", "zaero", "zaero_carry"),
                        default="init")
    args = parser.parse_args()
    rows, fixture_dir = {"init": (co.CASES, co.FIXTURE_DIR), "therm": (co.VT_CASES, co.VT_FIXTURE_DIR),
                         "itd": (co.ITD_CASES, co.ITD_FIXTURE_DIR),
                         "ridge": (co.RIDGE_CASES, co.RIDGE_FIXTURE_DIR),
                         "radiation": (co.RADIATION_CASES, co.RADIATION_FIXTURE_DIR),
                         "dedd": (co.DEDD_CASES, co.DEDD_FIXTURE_DIR),
                         "step": (co.STEP_CASES, co.STEP_FIXTURE_DIR),
                         "coupled": (co.COUPLED_CASES, co.COUPLED_FIXTURE_DIR),
                         "oml": (co.OML_CASES + co.OML_STEP_CASES, co.OML_FIXTURE_DIR),
                         "bgc": (co.BGC_CASES, co.BGC_FIXTURE_DIR),
                         "carry": (co.BGC_CARRY_CASES + co.BGC_CARRY_COUPLED_CASES,
                                   co.BGC_CARRY_FIXTURE_DIR),
                         "zsal": (co.BGC_ZSAL_CASES, co.BGC_ZSAL_FIXTURE_DIR),
                         "zsal_solve": (co.BGC_ZSAL_SOLVE_CASES, co.BGC_ZSAL_SOLVE_FIXTURE_DIR),
                         "zsal_carry": (co.BGC_ZSAL_CARRY_CASES + co.BGC_ZSAL_CARRY_COUPLED_CASES,
                                        co.BGC_ZSAL_CARRY_FIXTURE_DIR),
                         "zsal_solve_carry": (co.BGC_ZSAL_SOLVE_CARRY_CASES + co.BGC_ZSAL_SOLVE_CARRY_COUPLED_CASES,
                                              co.BGC_ZSAL_SOLVE_CARRY_FIXTURE_DIR),
                         "nonreactive": (co.BGC_NONREACTIVE_CASES, co.BGC_NONREACTIVE_FIXTURE_DIR),
                         "nonreactive_carry": (co.BGC_NONREACTIVE_CARRY_CASES,
                                               co.BGC_NONREACTIVE_CARRY_FIXTURE_DIR),
                         "nonreactive_step": (co.BGC_NONREACTIVE_STEP_CASES,
                                              co.BGC_NONREACTIVE_STEP_FIXTURE_DIR),
                         "zaero": (co.BGC_ZAERO_CASES, co.BGC_ZAERO_FIXTURE_DIR),
                         "zaero_carry": (co.BGC_ZAERO_CARRY_CASES, co.BGC_ZAERO_CARRY_FIXTURE_DIR)}[args.cases]
    args.out = args.out or fixture_dir
    args.out.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    logs, digests = {}, []
    for case in co.cases("r8", rows):
        outputs, log = co.run_oracle(args.build_dir, case)
        if outputs is not None:         # None: v8.4.2 died (log["crash"])
            for key, value in co.changed_fields(case, outputs).items():
                arrays[f"{case['name']}/{key}"] = value
        logs[case["name"]] = log
        digests.append(f"{co.input_digest(case)}  {case['name']}/r8")
    np.savez_compressed(args.out / "oracle-r8.npz", **arrays)
    (args.out / "column-logs.json").write_text(json.dumps(logs, indent=1, sort_keys=True) + "\n",
                                               encoding="utf-8")
    if args.cases == "init":
        np.save(args.out / "init-thermo-r8.npy", co.oracle_thermo_profiles(args.build_dir))
    (args.out / "inputs.sha256").write_text("\n".join(digests) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
