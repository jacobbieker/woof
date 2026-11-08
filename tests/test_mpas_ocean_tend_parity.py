"""MPAS-Ocean tendency terms, word for word against v8.4.2, at RKIND 4 and 8.

The reference is MPAS-Model v8.4.2's own ``ocn_vel_coriolis_tend``,
``ocn_vel_vadv_tend``, ``ocn_thick_hadv_tend`` and ``ocn_thick_vadv_tend``
(with their ``_init`` routines). They are compiled verbatim with gfortran -O0
-ffp-contract=off at MPAS's default single precision and at its double
precision, and driven by ``tools/mpas_ocean_oracle/run_oracle.F90``. The
outputs are committed (tests/data/oracles/mpas_ocean_tend, PROVENANCE.md). When
``GFORTRAN`` (or gfortran on PATH) and ``WOOF_MPAS_V842_SOURCE`` are set, the
fixture itself is rebuilt and compared.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with MPAS only to a tolerance, in either precision.
2. A case generator whose bytes drift. The input digests are pinned.
3. A gate that cannot fail. A mutated evaluation order must produce mismatches.
4. An unknown time integrator silently taking an RK4On value. It is refused
   by name.
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
from woof.verify import mpas_ocean_tend_oracle as oo

ROOT = Path(__file__).resolve().parents[1]


def test_case_inputs_are_the_recorded_bytes():
    recorded = dict(reversed(line.split()) for line in
                    (oo.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines())
    assert recorded == {f"{case['name']}/{p}": oo.input_digest(case, p)
                        for p in oo.PRECISIONS for case in oo.cases()}


def test_fixture_covers_the_branches():
    for case in oo.cases():
        assert (case["maxLevelEdgeTop"] == 0).any()
        assert (case["maxLevelEdgeBot"] > case["maxLevelEdgeTop"]).any()
        assert (case["maxLevelCell"] < case["nVertLevels"]).any()
        inside = np.arange(case["nVertLevels"])[None, :] < case["maxLevelEdgeTop"][:, None]
        assert (inside & (case["edgeMask"] == 0)).any()
        assert {-1, 1} <= set(np.unique(case["edgeSignOnCell"]))
    assert {case["rk4"] for case in oo.cases()} == {True, False}
    for precision in oo.PRECISIONS:
        fixture = oo.load_fixture(precision)
        assert set(fixture) == {row[0] for row in oo.CASES}
        assert all(np.isfinite(v).all() for c in fixture.values() for v in c.values())


def test_an_unknown_time_integrator_is_refused_by_name():
    from woof.hex.mpas_ocean import tend_v842
    assert tend_v842.rk4_on("RK4") == 1 and tend_v842.rk4_on("split_explicit") == 0
    with pytest.raises(ConfigurationRefusal) as exc:
        tend_v842.rk4_on("split_implicit")
    assert "config_time_integrator" in str(exc.value)
    assert not tend_v842.vel_vadv_on("impermeable_interfaces", False)


@requires_gpu
@pytest.mark.parametrize("precision", sorted(oo.PRECISIONS))
def test_gpu_matches_the_v842_oracle_word_for_word(precision):
    report = oo.measure(precision)
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
def test_the_ocean_door_verify_grades_the_ported_piece():
    from woof.hex.mpas_ocean.verify import verify
    report = verify()
    assert report["passed"] and report["status"] == "stub"
    assert [p["name"] for p in report["pieces"]] == ["tendency_terms_r4", "tendency_terms_r8",
                                                     "pressure_gradient_hmix_r4", "pressure_gradient_hmix_r8"]


@requires_gpu
@pytest.mark.parametrize("precision", sorted(oo.PRECISIONS))
@pytest.mark.parametrize("original, mutated, field", [
    # the thickness flux formed as u*(dv*h)
    ("const rk flux = M(M(normalVelocity[iEdge * nVertLevels + k], dvEdge[iEdge]),\n"
     "                              layerThicknessEdge[iEdge * nVertLevels + k]);",
     "const rk flux = M(normalVelocity[iEdge * nVertLevels + k],\n"
     "                              M(dvEdge[iEdge], layerThicknessEdge[iEdge * nVertLevels + k]));",
     "tendThickHadv"),
    # the Coriolis edge sum reassociated
    ("q = A(q, M(M(M(edgeWeight, normalVelocity[pe]), workVorticity), layerThicknessEdge[pe]));",
     "q = A(q, M(M(edgeWeight, normalVelocity[pe]), M(workVorticity, layerThicknessEdge[pe])));",
     "tendVelCoriolis"),
])
def test_a_mutated_evaluation_order_is_caught(monkeypatch, precision, original, mutated, field):
    from woof.hex.mpas_ocean import tend_v842
    real = tend_v842.module_source
    assert real("r4").count(original) == 1
    monkeypatch.setattr(tend_v842, "module_source", lambda p="r4": real(p).replace(original, mutated))
    tend_v842._module_on.cache_clear()
    try:
        case = oo.cases()[0]
        got = oo.run_gpu(case, precision)
        assert oo.bit_mismatches(got[field], oo.load_fixture(precision)[case["name"]][field]) > 0
    finally:
        tend_v842._module_on.cache_clear()


def _oracle_toolchain() -> tuple[str, Path] | None:
    source = os.environ.get("WOOF_MPAS_V842_SOURCE")
    gfortran = os.environ.get("GFORTRAN") or shutil.which("gfortran")
    if not source or not gfortran or not Path(source, "src", "core_ocean").is_dir():
        return None
    return gfortran, Path(source)


@pytest.mark.skipif(_oracle_toolchain() is None,
                    reason="the MPAS-Ocean oracle build needs gfortran (pixi -e mpas-oracle) and "
                           "WOOF_MPAS_V842_SOURCE naming an MPAS-Model v8.4.2 tree")
def test_the_oracle_rebuilt_here_writes_the_fixture(tmp_path):
    gfortran, source = _oracle_toolchain()
    subprocess.run(["bash", str(ROOT / "tools" / "mpas_ocean_oracle" / "build.sh"), str(source),
                    str(tmp_path / "build")], check=True, env={**os.environ, "GFORTRAN": gfortran})
    subprocess.run([sys.executable, str(ROOT / "tools" / "mpas_ocean_oracle" / "make_fixture.py"),
                    str(tmp_path / "build"), "--out", str(tmp_path / "fixture")], check=True)
    for precision in oo.PRECISIONS:
        fresh = np.load(tmp_path / "fixture" / f"oracle-{precision}.npz")
        pinned = np.load(oo.FIXTURE_DIR / f"oracle-{precision}.npz")
        assert sorted(fresh.files) == sorted(pinned.files)
        for key in pinned.files:
            assert oo.bit_mismatches(fresh[key], pinned[key]) == 0, (precision, key)
