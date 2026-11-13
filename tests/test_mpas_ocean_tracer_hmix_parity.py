"""MPAS-Ocean's horizontal tracer mixing (del2, del4), word for word against v8.4.2, at RKIND 4 and 8.

The reference is MPAS-Model v8.4.2's own ``ocn_tracer_hmix_init``,
``ocn_tracer_hmix_redi_init`` and ``ocn_tracer_hmix_tend`` (the four mixing
modules compiled verbatim with gfortran -O0 -ffp-contract=off at MPAS's
default single precision and at its double precision), driven by
``tools/mpas_ocean_oracle/run_tracer_hmix.F90``. The outputs are committed
(tests/data/oracles/mpas_ocean_tracer_hmix, PROVENANCE.md). When ``GFORTRAN``
(or gfortran on PATH) and ``WOOF_MPAS_V842_SOURCE`` are set, the fixture
itself is rebuilt and compared.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with MPAS only to a tolerance, in either precision.
2. A case generator whose bytes drift. The input digests are pinned.
3. A fixture that misses a branch: del2, del4, both, each asked for with its
   switch or coefficient off, mixing disabled, del4's masked edge levels.
4. A gate that cannot fail: a regrouped coefficient or flux must produce
   mismatches.
5. Redi isoneutral mixing silently skipped: it is refused by name.
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
from woof.verify import mpas_ocean_tracer_hmix_oracle as oo

ROOT = Path(__file__).resolve().parents[1]
BOTH = ("r4", "r8")
TEND = "tracersTend/activeTracersTend"


def test_case_inputs_are_the_recorded_bytes():
    recorded = dict(reversed(line.split()) for line in
                    (oo.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines())
    assert recorded == {f"{case['name']}/{p}": oo.input_digest(case) for p in BOTH for case in oo.cases(p)}


@pytest.mark.parametrize("precision", BOTH)
def test_fixture_covers_the_branches(precision):
    from woof.hex.mpas_ocean import tracer_hmix_v842 as th
    fixture = oo.load_fixture(precision)
    cases = {case["name"]: case for case in oo.cases(precision)}
    assert set(fixture) == set(cases)
    seen = set()
    for name, out in fixture.items():
        case = cases[name]
        del2, del4 = th.switches(case["config"])
        on = not case["config"]["config_disable_tr_hmix"] and (del2 is not None or del4 is not None)
        seen.add((del2 is not None, del4 is not None, on))
        changed = oo.word_mismatches(out[TEND], case["pools"]["tracersTend"]["activeTracersTend"])
        assert (changed > 0) == on, name
        mesh = case["pools"]["mesh"]
        inside = np.arange(case["dims"]["nVertLevels"])[None, :] < mesh["maxLevelEdgeTop"][:, None]
        assert (inside & (mesh["edgeMask"] == 0)).any()
    assert {(True, False, True), (False, True, True), (True, True, True)} <= seen
    assert any(not on for _, _, on in seen)


def test_redi_is_refused_by_name():
    from woof.hex.mpas_ocean import tracer_hmix_v842 as th
    th.check_config(dict(oo.BASE_CONFIG))
    with pytest.raises(ConfigurationRefusal, match="config_use_standardGM"):
        th.check_config(dict(oo.BASE_CONFIG, config_use_standardGM=True))


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
def test_gpu_matches_the_v842_oracle_word_for_word(precision):
    report = oo.measure(precision)
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
@pytest.mark.parametrize("original, mutated, case_name", [
    # del2's coefficient with dv/dc taken first
    ("D(M(M(meshScalingDel2[iEdge], eddyDiff2), dvEdge[iEdge]), dcEdge[iEdge])",
     "M(M(meshScalingDel2[iEdge], eddyDiff2), D(dvEdge[iEdge], dcEdge[iEdge]))", "del2"),
    # del2's flux with the thickness last
    ("M(M(layerThicknessEdge[iEdge * nl + k], turb), r_tmp)",
     "M(layerThicknessEdge[iEdge * nl + k], M(turb, r_tmp))", "del2"),
    # del4's Laplacian divided by the area instead of multiplied by its inverse
    ("M(M((rk)edgeSignOnCell[iCell * maxEdges + i], S(r2, r1)), invAreaCell1)",
     "D(M((rk)edgeSignOnCell[iCell * maxEdges + i], S(r2, r1)), areaCell[iCell])", "del4"),
    # del4's coefficient with eddyDiff4/dc taken first
    ("D(M(M(meshScalingDel4[iEdge], dvEdge[iEdge]), eddyDiff4), dcEdge[iEdge])",
     "M(M(meshScalingDel4[iEdge], dvEdge[iEdge]), D(eddyDiff4, dcEdge[iEdge]))", "del2_del4"),
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
                    str(tmp_path / "build"), "--piece", "tracer_hmix", "--out", str(tmp_path / "fixture")],
                   check=True)
    for precision in BOTH:
        fresh = np.load(tmp_path / "fixture" / f"oracle-{precision}.npz")
        pinned = np.load(oo.FIXTURE_DIR / f"oracle-{precision}.npz")
        assert sorted(fresh.files) == sorted(pinned.files)
        for key in pinned.files:
            assert oo.word_mismatches(fresh[key], pinned[key]) == 0, (precision, key)
