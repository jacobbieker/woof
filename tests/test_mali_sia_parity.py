"""MALI's shallow-ice velocity solver, word for word against MPAS-Model v8.4.2.

The reference is one ``li_sia_solve`` call. It is compiled verbatim, together
with every land-ice module and framework operator it reaches (li_constants,
li_setup, li_mask, li_thermal, li_diagnostic_vars, mpas_constants,
mpas_vector_operations, mpas_matrix_operations, mpas_geometry_utils), by
``tools/mali_oracle`` with gfortran -O0 -ffp-contract=off at RKIND 4. The
outputs are committed (tests/data/oracles/mali_sia, PROVENANCE.md). With
``GFORTRAN`` (or gfortran on PATH) and ``WOOF_MPAS_V842_SOURCE`` set, the
fixture itself is rebuilt and compared.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with MALI only to a tolerance. That includes a
   ``powf``/``expf`` that is not the reference libm's.
2. A case generator whose bytes drift. The input digests are pinned.
3. A gate that cannot fail. A mutated evaluation order must produce mismatches.
4. Albany's external dycores (FO, L1L2, Stokes) silently running as SIA. They
   are refused by name.
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
from woof.hex.errors import ConfigurationRefusal
from woof.verify import mali_sia_oracle as mo

ROOT = Path(__file__).resolve().parents[1]


def test_case_inputs_are_the_recorded_bytes():
    recorded = dict(reversed(line.split()) for line in
                    (mo.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines())
    assert recorded == {case["name"]: mo.input_digest(case) for case in mo.cases()}


def test_fixture_covers_the_branches():
    fixture = mo.load_fixture()
    assert {c["tangentSlopeCalculation"] for c in mo.cases()} == set(mo.TANGENT)
    assert {c["flowParamACalculation"] for c in mo.cases()} == set(mo.FLOW_A)
    for case in mo.cases():
        out = fixture[case["name"]]
        assert out["err"].tolist() == [0, 0]
        assert all(np.isfinite(out[k]).all() for k in mo.OUTPUT_FIELDS)
        assert (case["thickness"] == 0).any()
        dyn_edge = (case["edgeMask"] & 2) > 0
        assert (~dyn_edge[:case["nEdgesSolve"]]).any()
        dyn_cell = (case["cellMask"] & 2) > 0
        one_sided = dyn_edge & (dyn_cell[case["cellsOnEdge"][:, 0] - 1] != dyn_cell[case["cellsOnEdge"][:, 1] - 1])
        assert one_sided.any()
        t = case["temperature"]
        assert (t > 273.15).any() and (t < 223.15).any()


def test_albany_and_unported_velocity_solvers_are_refused_by_name():
    from woof.hex.mpas_ice import sia_v842
    sia_v842.require_velocity_solver("sia")
    for solver in ("FO", "L1L2", "Stokes"):
        with pytest.raises(ConfigurationRefusal) as exc:
            sia_v842.require_velocity_solver(solver)
        assert "Albany" in str(exc.value) and "config_velocity_solver" in str(exc.value)
    for solver in ("simple", "none"):
        with pytest.raises(ConfigurationRefusal):
            sia_v842.require_velocity_solver(solver)


@requires_gpu
def test_gpu_matches_the_v842_oracle_word_for_word():
    report = mo.measure()
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
def test_the_ice_door_verify_grades_the_ported_piece():
    from woof.hex.mpas_ice.verify import verify
    report = verify()
    assert report["passed"] and report["status"] == "stub"
    assert [p["name"] for p in report["pieces"]] == ["sia_velocity_solver", "geometry_diagnostics",
                                                     "thermal_init_and_conversions"]


@requires_gpu
@pytest.mark.parametrize("original, mutated, field, case_index", [
    # the level-independent factor associated the other way
    ("const float levelIndependentFactor = M(M(gfk_pow(slopeEdge[iEdge], nMinus1), normalSlopeEdge[iEdge]),\n"
     "                                           gfk_pow(thicknessEdge, nPlus1));",
     "const float levelIndependentFactor = M(gfk_pow(slopeEdge[iEdge], nMinus1),\n"
     "                                           M(normalSlopeEdge[iEdge], gfk_pow(thicknessEdge, nPlus1)));",
     "normalVelocity", 0),
    # CUDA's expf in place of glibc's
    ("flowParamA[base + k] = M(A0, gfk_exp(D(M(-1.0f, Q),",
     "flowParamA[base + k] = M(A0, expf(D(M(-1.0f, Q),", "flowParamA", 0),
])
def test_a_mutated_operation_is_caught(monkeypatch, original, mutated, field, case_index):
    from woof.hex.mpas_ice import sia_v842
    source = sia_v842.module_source()
    assert source.count(original) == 1
    monkeypatch.setattr(sia_v842, "module_source", lambda: source.replace(original, mutated))
    sia_v842._module_on.cache_clear()
    try:
        case = mo.cases()[case_index]
        got = mo.run_gpu(case)
        assert mo.bit_mismatches(got[field], mo.load_fixture()[case["name"]][field]) > 0
    finally:
        sia_v842._module_on.cache_clear()


def _oracle_toolchain() -> tuple[str, Path] | None:
    source = os.environ.get("WOOF_MPAS_V842_SOURCE")
    gfortran = os.environ.get("GFORTRAN") or shutil.which("gfortran")
    if not source or not gfortran or not Path(source, "src", "core_landice").is_dir():
        return None
    return gfortran, Path(source)


@pytest.mark.skipif(_oracle_toolchain() is None,
                    reason="the MALI oracle build needs gfortran (pixi -e mpas-oracle) and "
                           "WOOF_MPAS_V842_SOURCE naming an MPAS-Model v8.4.2 tree")
def test_the_oracle_rebuilt_here_writes_the_fixture(tmp_path):
    gfortran, source = _oracle_toolchain()
    subprocess.run(["bash", str(ROOT / "tools" / "mali_oracle" / "build.sh"), str(source),
                    str(tmp_path / "build")], check=True, env={**os.environ, "GFORTRAN": gfortran})
    subprocess.run([sys.executable, str(ROOT / "tools" / "mali_oracle" / "make_fixture.py"),
                    str(tmp_path / "build"), "--out", str(tmp_path / "fixture")], check=True)
    fresh = np.load(tmp_path / "fixture" / "oracle-r4.npz")
    pinned = np.load(mo.FIXTURE_DIR / "oracle-r4.npz")
    assert sorted(fresh.files) == sorted(pinned.files)
    for key in pinned.files:
        if key.endswith("/err"):
            assert (fresh[key] == pinned[key]).all(), key
        else:
            assert mo.bit_mismatches(fresh[key], pinned[key]) == 0, key
