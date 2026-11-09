"""The MPAS-Seaice velocity solver's pre-subcycle routines, word for word against v8.4.2, at RKIND 4 and 8.

The reference is MPAS-Model v8.4.2's own ``mpas_seaice_velocity_solver.F``,
with ``seaice_mesh``'s interpolation routines and the stand-alone column
constants, compiled verbatim with gfortran -O0 -ffp-contract=off at RKIND 4
and RKIND 8. ``tools/mpas_seaice_oracle/run_presubcycle.F90`` drives
``velocity_solver_pre_subcycle``: the aggregates, the masks, the new-ice
velocities, the Hibler ice strength, the air stress, the Coriolis
coefficient, the ocean stress, the surface tilt and the subcycle
initialisation (without the ice strength under the column vertical
thermodynamics, which the port refuses by name). The oracle's outputs are
committed (tests/data/oracles/mpas_seaice_presubcycle, PROVENANCE.md), so the
GPU comparison runs without gfortran. When the oracle can be built here
(``GFORTRAN`` or gfortran on PATH, and ``WOOF_MPAS_V842_SOURCE`` naming the
MPAS-Model v8.4.2 tree), the fixture itself is regenerated and compared too.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with MPAS only to a tolerance. Every output word, the
   integer masks included, must be bit-identical, including the halo and
   unsolved entries the routines leave alone, at both precisions.
2. A case generator whose bytes drift. The input digests are pinned.
3. A gate that cannot fail. Mutating one evaluation order or one comparison in
   the kernel must produce mismatches at each precision.
4. A port that silently runs the column package's ice strength it does not
   carry.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from woof.verify import mpas_seaice_presubcycle_oracle as po

ROOT = Path(__file__).resolve().parents[1]


def test_case_inputs_are_the_recorded_bytes():
    recorded = {}
    for line in (po.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        recorded[name] = digest
    assert recorded == {f"{case['name']}/{p}": po.input_digest(case, p)
                        for p in po.PRECISIONS for case in po.cases()}


def test_fixture_covers_the_branches():
    cases = {case["name"]: case for case in po.cases()}
    assert {c["tilt"] for c in cases.values()} == set(po.TILTS)
    assert {c["useAirStress"] for c in cases.values()} == {False, True}
    assert {c["useOceanStress"] for c in cases.values()} == {False, True}
    assert {(c["useColumnPackage"], c["useColumnVerticalThermodynamics"]) for c in cases.values()} == {
        (False, False), (True, False), (True, True)}
    for precision in po.PRECISIONS:
        fixture = po.load_fixture(precision)
        assert set(fixture) == set(cases)
        for name, out in fixture.items():
            case = cases[name]
            nvs = case["nVerticesSolve"]
            assert all(np.isfinite(v).all() for v in out.values())
            # both mask values, the neighbour search, and newly solved vertices
            assert set(np.unique(out["solveStress"])) == {0, 1}
            assert set(np.unique(out["solveVelocity"][:nvs])) == {0, 1}
            own = ((case["iceAreaCellInitial"] if case["useColumnVerticalThermodynamics"] and
                    case["useColumnPackage"] else out["iceAreaCellInitial"]) > 0.001)
            assert ((out["solveStress"] == 1) & ~own).any()
            assert ((out["solveVelocity"] == 1) & (case["solveVelocityPrevious"] == 0))[:nvs].any()
            assert ((out["solveVelocity"] == 1) & (case["solveVelocityPrevious"] == 1))[:nvs].any()
        # the initial area exactly at the minimum, where the column supplies it
        thermo = cases["column_thermo_ssh_tilt"]
        assert (np.asarray(thermo["iceAreaCellInitial"], po.PRECISIONS[precision])
                == po.PRECISIONS[precision](0.001)).any()
    for case in cases.values():
        f = case["fVertex"][: case["nVerticesSolve"]]
        assert (np.signbit(f) & (f == 0)).any() and (~np.signbit(f) & (f == 0)).any()
        assert case["nVerticesSolve"] < case["nVertices"] and case["nCellsSolve"] < case["nCells"]


def test_the_column_ice_strength_is_refused_by_name():
    from woof.hex.errors import ConfigurationRefusal
    from woof.hex.mpas_seaice import presubcycle_v842 as pre
    with pytest.raises(ConfigurationRefusal, match="colpkg_ice_strength"):
        pre.refuse_unported(use_column_package=True, use_column_vertical_thermodynamics=True)
    # pre_subcycle refuses before it touches an array
    with pytest.raises(ConfigurationRefusal):
        pre.pre_subcycle(mesh={}, boundary={}, tracers={}, aggregate={}, icestate={}, ocean={}, atmos={},
                         solver={}, variational={}, use_column_package=True,
                         use_column_vertical_thermodynamics=True, use_air_stress=True, use_ocean_stress=True,
                         tilt="none")
    pre.refuse_unported(use_column_package=True, use_column_vertical_thermodynamics=False)
    pre.refuse_unported(use_column_package=False, use_column_vertical_thermodynamics=True)


@requires_gpu
@pytest.mark.parametrize("precision", sorted(po.PRECISIONS))
def test_gpu_matches_the_v842_oracle_word_for_word(precision):
    report = po.measure(precision)
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize("precision", sorted(po.PRECISIONS))
@pytest.mark.parametrize("original, mutated, field, case", [
    # the interpolation's division by the summed area taken as a reciprocal
    ("variableVertex[iVertex] = D(value, totalArea);",
     "variableVertex[iVertex] = M(value, D((rk)1.0, totalArea));", "totalMassVertex", "standalone_geostrophic"),
    # (P* * volume) * exp regrouped as P* * (volume * exp)
    ("M(M(SI_PSTAR, iceVolumeCell[iCell]), EXP(arg));",
     "M(SI_PSTAR, M(iceVolumeCell[iCell], EXP(arg)));", "icePressure", "standalone_geostrophic"),
    # the air stress's last two factors regrouped
    ("airStressCellU[iCell] = M(M(scale, u), iceAreaCell[iCell]);",
     "airStressCellU[iCell] = M(scale, M(u, iceAreaCell[iCell]));", "airStressCellU", "standalone_geostrophic"),
    # the snow mass added before the ice mass's product is rounded
    ("totalMassCell[iCell] = A(M(ice, SI_RHOI), M(snow, SI_RHOS));",
     "totalMassCell[iCell] = M(A(ice, D(M(snow, SI_RHOS), SI_RHOI)), SI_RHOI);", "totalMassCell",
     "column_no_thermo_no_air"),
    # the area minimum taken as inclusive (the column supplies areas of exactly 0.001)
    ("return area > (rk)SI_AREA_MIN && mass > (rk)SI_MASS_MIN && landIce == 0;",
     "return area >= (rk)SI_AREA_MIN && mass > (rk)SI_MASS_MIN && landIce == 0;", "solveStress",
     "column_thermo_ssh_tilt"),
    # the sea-surface tilt force's mass and tilt multiplied first
    ("surfaceTiltForceU[iVertex] = -M(gm, seaSurfaceTiltVertexU[iVertex]);",
     "surfaceTiltForceU[iVertex] = -M(SI_GRAVITY, M(totalMassVertex[iVertex], seaSurfaceTiltVertexU[iVertex]));",
     "surfaceTiltForceU", "standalone_ssh_no_air"),
])
def test_a_mutated_evaluation_order_is_caught(monkeypatch, precision, original, mutated, field, case):
    from woof.hex.mpas_seaice import presubcycle_v842
    source = presubcycle_v842.module_source(precision)
    assert source.count(original) == 1
    monkeypatch.setattr(presubcycle_v842, "module_source", lambda p="r4": source.replace(original, mutated))
    presubcycle_v842._module_on.cache_clear()
    try:
        row = next(c for c in po.cases() if c["name"] == case)
        got = po.run_gpu(row, precision)
        assert po.bit_mismatches(got[field], po.load_fixture(precision)[case][field]) > 0
    finally:
        presubcycle_v842._module_on.cache_clear()


def _oracle_toolchain() -> tuple[str, Path] | None:
    source = os.environ.get("WOOF_MPAS_V842_SOURCE")
    gfortran = os.environ.get("GFORTRAN") or shutil.which("gfortran")
    if not source or not gfortran or not Path(source, "src", "core_seaice").is_dir():
        return None
    return gfortran, Path(source)


@pytest.mark.skipif(_oracle_toolchain() is None,
                    reason="the MPAS-Seaice oracle build needs gfortran (pixi -e mpas-oracle) and "
                           "WOOF_MPAS_V842_SOURCE naming an MPAS-Model v8.4.2 tree")
def test_the_oracle_rebuilt_here_writes_the_fixture(tmp_path):
    gfortran, source = _oracle_toolchain()
    subprocess.run(["bash", str(ROOT / "tools" / "mpas_seaice_oracle" / "build.sh"), str(source),
                    str(tmp_path / "build")], check=True, env={**os.environ, "GFORTRAN": gfortran})
    subprocess.run([sys.executable, str(ROOT / "tools" / "mpas_seaice_oracle" / "make_presubcycle_fixture.py"),
                    str(tmp_path / "build"), "--out", str(tmp_path / "fixture")], check=True)
    for precision in po.PRECISIONS:
        fresh = np.load(tmp_path / "fixture" / f"oracle-{precision}.npz")
        pinned = np.load(po.FIXTURE_DIR / f"oracle-{precision}.npz")
        assert sorted(fresh.files) == sorted(pinned.files)
        for key in pinned.files:
            assert po.bit_mismatches(fresh[key], pinned[key]) == 0, f"{precision}:{key}"
