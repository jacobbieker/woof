"""MPAS-Ocean's Redi isoneutral tracer mixing, word for word against v8.4.2, at RKIND 4 and 8.

The reference is MPAS-Model v8.4.2's own ``ocn_tracer_hmix_tend`` with
``ocn_tracer_hmix_redi_tend`` on (``config_use_standardGM``; the four mixing
modules compiled verbatim with gfortran -O0 -ffp-contract=off at MPAS's
default single precision and at its double precision), driven by
``tools/mpas_ocean_oracle/run_tracer_hmix.F90``. The outputs are committed
(tests/data/oracles/mpas_ocean_redi, PROVENANCE.md). When ``GFORTRAN`` (or
gfortran on PATH) and ``WOOF_MPAS_V842_SOURCE`` are set, the fixture itself
is rebuilt and compared. Redi's vertical term is graded with the vertical
mixing (tests/test_mpas_ocean_vmix_parity.py).

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with MPAS only to a tolerance, in either precision, or
   only under one numerics mode.
2. A case generator whose bytes drift. The input digests are pinned.
3. A fixture that misses a branch: each horizontal term switched off, Redi
   after del2, nested halo extents, one-level columns, mixing disabled.
4. A gate that cannot fail: a regrouped flux or gradient must produce
   mismatches.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from _numerics_child import MODES, run as run_child
from conftest import requires_gpu
import _oracle_toolchains
from woof.verify import mpas_ocean_redi_oracle as oo

ROOT = Path(__file__).resolve().parents[1]
BOTH = ("r4", "r8")
TEND = oo.TEND


def test_case_inputs_are_the_recorded_bytes():
    recorded = dict(reversed(line.split()) for line in
                    (oo.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines())
    assert recorded == {f"{case['name']}/{p}": oo.input_digest(case) for p in BOTH for case in oo.cases(p)}


@pytest.mark.parametrize("precision", BOTH)
def test_fixture_covers_the_branches(precision):
    fixture = oo.load_fixture(precision)
    cases = {case["name"]: case for case in oo.cases(precision)}
    assert set(fixture) == set(cases)
    for name, out in fixture.items():
        case = cases[name]
        changed = oo.word_mismatches(out[TEND], case["pools"]["tracersTend"]["activeTracersTend"])
        assert (changed > 0) == (not case["config"]["config_disable_tr_hmix"]), name
        assert np.isfinite(out[TEND]).all(), name
    terms = {t for c in cases.values() for t in (1, 2, 3) if c["config"][f"config_disable_redi_horizontal_term{t}"]}
    assert terms == {1, 2, 3}
    nested = cases["nested"]["pools"]["mesh"]
    assert len(set(map(int, nested["nCellsArray"]))) == 4 and len(set(map(int, nested["nEdgesArray"]))) == 4
    assert (cases["term3_off"]["pools"]["mesh"]["maxLevelCell"][:-1] == 1).any()


@requires_gpu
def test_one_level_meshes_raise():
    import cupy as cp
    from woof.hex.mpas_ocean import tracer_hmix_v842 as th
    case = oo.cases("r8")[0]
    dims = case["dims"]
    t = cp.zeros((dims["nCells"] + 1, 1, 3))
    with pytest.raises(ValueError, match="level 2"):
        th.tracer_hmix_tend(tracers=t, layer_thickness_edge=cp.zeros((dims["nEdges"] + 1, 1)),
                            mesh=dict(case["pools"]["mesh"], nVertLevels=1), tend=t.copy(),
                            configs=case["config"], diagnostics={})


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
def test_gpu_matches_the_v842_oracle_word_for_word(precision):
    report = oo.measure(precision)
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize(*MODES)
def test_word_exact_in_fast_and_strict_children(numerics, exact):
    run_child(__file__, ["test_gpu_matches_the_v842_oracle_word_for_word"], numerics, exact)


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
@pytest.mark.parametrize("original, mutated, case_name", [
    # term 1's coefficient with dv/dc taken first
    ("D(M(kappa, dvEdge[iEdge]), dcEdge[iEdge])", "M(kappa, D(dvEdge[iEdge], dcEdge[iEdge]))", "term2_off"),
    # term 2's flux with the thickness inside
    ("rk flux = M(M(C(0.5), dvEdge[iEdge]), A(s_tmpU, s_tmpD));\n            flux = M(flux, layerThicknessEdge[iEdge * nl + k]);",
     "rk flux = M(M(C(0.5), dvEdge[iEdge]), M(A(s_tmpU, s_tmpD), layerThicknessEdge[iEdge * nl + k]));", "term1_off"),
    # the interface gradient's thickness weighting divided term by term
    ("D(A(M(h2, ge[k - 1]), M(h1, ge[k])), A(h1, h2))",
     "A(D(M(h2, ge[k - 1]), A(h1, h2)), D(M(h1, ge[k]), A(h1, h2)))", "redi"),
    # the vertical derivative over the thickness-weighted distance's other sign
    ("S(zMid[iCell * nl + k - 1], zMid[iCell * nl + k])", "S(zMid[iCell * nl + k], zMid[iCell * nl + k - 1])", "redi"),
    # term 3 without areaCellSum's 1e-34 start
    ("for (int k = 0; k < nl; ++k) acs[k] = C(1.0e-34);", "for (int k = 0; k < nl; ++k) acs[k] = C(0.0);", "redi"),
])
def test_a_mutated_port_is_caught(monkeypatch, precision, original, mutated, case_name):
    from woof.hex.mpas_ocean import tracer_hmix_v842
    real = tracer_hmix_v842.module_source
    assert real(precision).count(original) == 1
    monkeypatch.setattr(tracer_hmix_v842, "module_source", lambda p="r4": real(p).replace(original, mutated))
    tracer_hmix_v842._module_on.cache_clear()
    try:
        case = next(c for c in oo.cases(precision) if c["name"] == case_name)
        got = oo.run_gpu(case)
        assert oo.word_mismatches(got[TEND], oo.load_fixture(precision)[case_name][TEND]) > 0
    finally:
        tracer_hmix_v842._module_on.cache_clear()


def _oracle_toolchain() -> tuple[str, Path] | None:
    return _oracle_toolchains.mpas_v842("core_ocean")


@pytest.mark.skipif(_oracle_toolchain() is None,
                    reason="the MPAS-Ocean oracle build needs gfortran (pixi -e mpas-oracle) and "
                           "WOOF_MPAS_V842_SOURCE naming an MPAS-Model v8.4.2 tree")
def test_the_oracle_rebuilt_here_writes_the_fixture(tmp_path):
    gfortran, source = _oracle_toolchain()
    subprocess.run(["bash", str(ROOT / "tools" / "mpas_ocean_oracle" / "build.sh"), str(source),
                    str(tmp_path / "build")], check=True, env={**os.environ, "GFORTRAN": gfortran})
    subprocess.run([sys.executable, str(ROOT / "tools" / "mpas_ocean_oracle" / "make_fixture.py"),
                    str(tmp_path / "build"), "--piece", "redi", "--out", str(tmp_path / "fixture")],
                   check=True)
    for precision in BOTH:
        fresh = np.load(tmp_path / "fixture" / f"oracle-{precision}.npz")
        pinned = np.load(oo.FIXTURE_DIR / f"oracle-{precision}.npz")
        assert sorted(fresh.files) == sorted(pinned.files)
        for key in pinned.files:
            assert oo.word_mismatches(fresh[key], pinned[key]) == 0, (precision, key)
