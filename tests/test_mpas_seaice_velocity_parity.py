"""The MPAS-Seaice EVP velocity solve, word for word against v8.4.2, at RKIND 4 and 8.

The reference is MPAS-Model v8.4.2's own ``mpas_seaice_velocity_solver.F``,
with the variational internal stress, the constitutive relation and the
stand-alone column constants, compiled verbatim with gfortran -O0
-ffp-contract=off at RKIND 4 and RKIND 8. ``tools/mpas_seaice_oracle/run_subcycle.F90``
drives it: the pre-subcycle Coriolis coefficient, ocean stress and subcycle
initialisation, then either whole elastic subcycles (internal stress, ocean
stress coefficient, velocity solve) or velocity solves against a fixed
stress divergence, then the final divergence/shear,
the principal stresses and the final ocean stress (``ocean_stress_final``,
with ``seaice_mesh``'s vertex-to-cell interpolation). The oracle's outputs are committed
(tests/data/oracles/mpas_seaice_velocity, PROVENANCE.md), so the GPU
comparison runs without gfortran. When the oracle can be built here
(``GFORTRAN`` or gfortran on PATH, and ``WOOF_MPAS_V842_SOURCE`` naming the
MPAS-Model v8.4.2 tree), the fixture itself is regenerated and compared too.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with MPAS only to a tolerance. Every output word must be
   bit-identical, including the halo and unsolved entries the routines leave
   alone, at both precisions.
2. A case generator whose bytes drift. The input digests are pinned.
3. A gate that cannot fail. Mutating one evaluation order in the kernel must
   produce mismatches at each precision.
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
from woof.verify import mpas_seaice_velocity_oracle as vo

ROOT = Path(__file__).resolve().parents[1]


def test_case_inputs_are_the_recorded_bytes():
    recorded = {}
    for line in (vo.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        recorded[name] = digest
    assert recorded == {f"{case['name']}/{p}": vo.input_digest(case, p)
                        for p in vo.PRECISIONS for case in vo.cases() if p in case["precisions"]}


def test_fixture_covers_the_branches():
    for precision in vo.PRECISIONS:
        fixture = vo.load_fixture(precision)
        assert set(fixture) == {row[0] for row in vo.CASES if precision in row[-1]}
        for out in fixture.values():
            assert all(np.isfinite(v).all() for v in out.values())
    cases = {case["name"]: case for case in vo.cases()}
    assert {c["revisedEVP"] for c in cases.values()} == {False, True}
    assert {c["useOceanStress"] for c in cases.values()} == {False, True}
    assert {c["full"] for c in cases.values()} == {False, True}
    for case in cases.values():
        f = case["fVertex"][: case["nVerticesSolve"]]
        # both signed zeros reach sign(1, f)
        assert (np.signbit(f) & (f == 0)).any() and (~np.signbit(f) & (f == 0)).any()
        assert (case["solveVelocity"][: case["nVerticesSolve"]] == 0).any()
        assert case["nVerticesSolve"] < case["nVertices"] and case["nCellsSolve"] < case["nCells"]
    # the vertex-to-cell interpolation meets cells with no interior vertex
    for case in cases.values():
        lonely = [c for c in range(case["nCellsSolve"])
                  if not case["interiorVertex"][case["verticesOnCell"][c, : case["nEdgesOnCell"][c]] - 1].any()]
        assert lonely and (case["interiorVertex"] == 0).mean() > 0.05
    # the final ocean stress is zeroed whole without config_use_ocean_stress
    off = vo.load_fixture("r8")["solve_no_ocean_stress"]
    assert not off["oceanStressCellU"].any() and not off["oceanStressU"].any()
    # the principal stresses take both branches
    out = vo.load_fixture("r4")["subcycle"]
    assert (out["principalStress1"] == np.float32(1.0e30)).any()
    assert (np.abs(out["principalStress1"]) < 1.0e29).any()


@requires_gpu
@pytest.mark.parametrize("precision", sorted(vo.PRECISIONS))
def test_gpu_matches_the_v842_oracle_word_for_word(precision):
    report = vo.measure(precision)
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
def test_the_seaice_door_verify_grades_every_piece():
    from woof.hex.mpas_seaice.verify import verify
    report = verify()
    assert report["passed"] and report["status"] == "stub"
    assert [p["name"] for p in report["pieces"]] == ["evp_variational_internal_stress",
                                                    "evp_velocity_solve_r4", "evp_velocity_solve_r8",
                                                    "velocity_pre_subcycle_r4", "velocity_pre_subcycle_r8"]


@requires_gpu
@pytest.mark.parametrize("precision", sorted(vo.PRECISIONS))
@pytest.mark.parametrize("original, mutated, field", [
    # the right-hand side's ocean-stress term added after the inertia term
    ("A(A(A(A(stressDivergenceU[iVertex], airStressVertexU[iVertex]), surfaceTiltForceU[iVertex]),\n"
     "                      M(coeff, oceanStressU[iVertex])), inertiaU);",
     "A(A(A(A(stressDivergenceU[iVertex], airStressVertexU[iVertex]), surfaceTiltForceU[iVertex]),\n"
     "                      inertiaU), M(coeff, oceanStressU[iVertex]));", "uVelocity"),
    # (drag*rho)*area regrouped as drag*(rho*area)
    ("M(M(M(SI_DRAGIO, SI_RHOW), iceAreaVertex[iVertex]),",
     "M(M(SI_DRAGIO, M(SI_RHOW, iceAreaVertex[iVertex])),", "oceanStressCoeff"),
    # the principal-stress discriminant summed in the other order
    ("A(S(M(t, t), M(M((rk)4.0, s11), s22)), M((rk)4.0, M(s12, s12)));",
     "S(A(M(t, t), M((rk)4.0, M(s12, s12))), M(M((rk)4.0, s11), s22));", "principalStress1"),
    # the final stress per unit ice area taken through a reciprocal
    ("oceanStressU[iVertex] = D(su, iceAreaVertex[iVertex]);",
     "oceanStressU[iVertex] = M(su, D((rk)1.0, iceAreaVertex[iVertex]));", "oceanStressCellU"),
    # a cell with no interior vertex divided anyway
    ("if (totalArea > (rk)0.0) value = D(value, totalArea);", "value = D(value, totalArea);",
     "oceanStressCellV"),
    # the vertex stress not scaled back to per unit grid area
    ("oceanStressU[iVertex] = M(oceanStressU[iVertex], iceAreaVertex[iVertex]);", "", "oceanStressU"),
])
def test_a_mutated_evaluation_order_is_caught(monkeypatch, precision, original, mutated, field):
    from woof.hex.mpas_seaice import velocity_v842
    source = velocity_v842.module_source(precision)
    assert source.count(original) == 1
    monkeypatch.setattr(velocity_v842, "module_source", lambda p="r4": source.replace(original, mutated))
    velocity_v842._module_on.cache_clear()
    try:
        case = next(c for c in vo.cases() if c["name"] == "solve")
        got = vo.run_gpu(case, precision)
        assert vo.bit_mismatches(got[field], vo.load_fixture(precision)["solve"][field]) > 0
    finally:
        velocity_v842._module_on.cache_clear()


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
    subprocess.run([sys.executable, str(ROOT / "tools" / "mpas_seaice_oracle" / "make_velocity_fixture.py"),
                    str(tmp_path / "build"), "--out", str(tmp_path / "fixture")], check=True)
    for precision in vo.PRECISIONS:
        fresh = np.load(tmp_path / "fixture" / f"oracle-{precision}.npz")
        pinned = np.load(vo.FIXTURE_DIR / f"oracle-{precision}.npz")
        assert sorted(fresh.files) == sorted(pinned.files)
        for key in pinned.files:
            assert vo.bit_mismatches(fresh[key], pinned[key]) == 0, f"{precision}:{key}"
