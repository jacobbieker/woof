"""MPAS-Ocean's Gent-McWilliams bolus velocity, word for word against v8.4.2, at RKIND 4 and 8.

The reference is MPAS-Model v8.4.2's own ``ocn_gm_init`` and
``ocn_gm_compute_Bolus_velocity`` (``mpas_ocn_gm.F`` compiled verbatim with
gfortran -O0 -ffp-contract=off at MPAS's default single precision and at its
double precision), driven by ``tools/mpas_ocean_oracle/run_gm.F90``. The
outputs are committed (tests/data/oracles/mpas_ocean_gm, PROVENANCE.md). When
``GFORTRAN`` (or gfortran on PATH) and ``WOOF_MPAS_V842_SOURCE`` are set, the
fixture itself is rebuilt and compared.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with MPAS only to a tolerance, in either precision.
2. A case generator whose bytes drift. The input digests are pinned.
3. A fixture that misses a branch: the slope clip, the max-slope taper, the
   surface and bottom tapering, ``config_disable_redi_k33``, edges too shallow
   for the stream function's solve.
4. A gate that cannot fail: a regrouped slope, taper or tridiagonal
   coefficient must produce mismatches.
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
from woof.verify import mpas_ocean_gm_oracle as oo

ROOT = Path(__file__).resolve().parents[1]
BOTH = ("r4", "r8")
BOLUS = "diagnostics/normalGMBolusVelocity"


def test_case_inputs_are_the_recorded_bytes():
    recorded = dict(reversed(line.split()) for line in
                    (oo.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines())
    assert recorded == {f"{case['name']}/{p}": oo.input_digest(case) for p in BOTH for case in oo.cases(p)}


@pytest.mark.parametrize("precision", BOTH)
def test_fixture_covers_the_branches(precision):
    fixture = oo.load_fixture(precision)
    cases = {case["name"]: case for case in oo.cases(precision)}
    assert set(fixture) == set(cases)
    clipped = tapered = shallow = False
    for name, out in fixture.items():
        case = cases[name]
        nl = case["dims"]["nVertLevels"]
        slope = out["diagnostics/relativeSlopeTopOfEdge"][:case["dims"]["nEdges"], :nl]
        clipped |= bool((np.abs(slope) == 1.0).any())
        taper = out["diagnostics/relativeSlopeTaperingCell"][:case["dims"]["nCells"], :nl]
        tapered |= bool(((taper > 0) & (taper < 1)).any())
        mlet = case["pools"]["mesh"]["maxLevelEdgeTop"][:case["dims"]["nEdges"]]
        shallow |= bool(((mlet >= 1) & (mlet < 3)).any())
        assert np.isfinite(out[BOLUS]).all(), name
        assert (out[BOLUS] != 0).any(), name
        k33 = out["diagnostics/k33"][:case["dims"]["nCells"]]
        assert (k33 != 0).any() == (not case["config"]["config_disable_redi_k33"]), name
    assert clipped and tapered and shallow
    assert any(c["config"]["config_use_Redi_surface_layer_tapering"] for c in cases.values())


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
def test_gpu_matches_the_v842_oracle_word_for_word(precision):
    report = oo.measure(precision)
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
@pytest.mark.parametrize("original, mutated, case_name", [
    # the constant-z density gradient with the product's operands regrouped into a division
    ("M(dDensityDzTopOfEdge[pe + k], gradZMidTopOfEdge[pe + k])",
     "D(gradZMidTopOfEdge[pe + k], C(1.0) / dDensityDzTopOfEdge[pe + k])", "default"),
    # the RMS slope as the ratio of square roots
    ("SQRT(D(sc[k], sum[k]))", "D(SQRT(sc[k]), SQRT(sum[k]))", "default"),
    # the tridiagonal's off-diagonal with the thickness sum divided first
    ("D(D(M(C(2.0), g2), h[f - 1]), A(h[f - 1], h[f]))", "D(M(C(2.0), g2), M(h[f - 1], A(h[f - 1], h[f])))",
     "default"),
    # the surface taper through the depth's reciprocal
    ("-D(z[k], depth)", "-M(z[k], D(C(1.0), depth))", "tapering"),
    # the bottom taper through the depth's reciprocal
    ("D(S(z[k], z[mlc - 1]), A(bottomDepth, EPS_GM))", "M(S(z[k], z[mlc - 1]), D(C(1.0), A(bottomDepth, EPS_GM)))",
     "tapering"),
])
def test_a_mutated_port_is_caught(monkeypatch, precision, original, mutated, case_name):
    from woof.hex.mpas_ocean import gm_v842
    real = gm_v842.module_source
    assert real(precision).count(original) == 1
    monkeypatch.setattr(gm_v842, "module_source", lambda p="r4": real(p).replace(original, mutated))
    gm_v842._module_on.cache_clear()
    try:
        case = next(c for c in oo.cases(precision) if c["name"] == case_name)
        got = oo.run_gpu(case)
        want = oo.load_fixture(precision)[case_name]
        assert sum(oo.word_mismatches(got[k], want[k]) for k in want) > 0
    finally:
        gm_v842._module_on.cache_clear()


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
                    str(tmp_path / "build"), "--piece", "gm", "--out", str(tmp_path / "fixture")],
                   check=True)
    for precision in BOTH:
        fresh = np.load(tmp_path / "fixture" / f"oracle-{precision}.npz")
        pinned = np.load(oo.FIXTURE_DIR / f"oracle-{precision}.npz")
        assert sorted(fresh.files) == sorted(pinned.files)
        for key in pinned.files:
            assert oo.word_mismatches(fresh[key], pinned[key]) == 0, (precision, key)
