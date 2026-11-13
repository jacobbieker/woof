"""MPAS-Ocean's monotone tracer advection, word for word against v8.4.2, at RKIND 4 and 8.

The reference is MPAS-Model v8.4.2's own ``ocn_tracer_advection_init`` and
``ocn_tracer_advection_tend`` (the flux-corrected monotone scheme, with the
vertical stencils of ``mpas_tracer_advection_helpers.F``), compiled with
gfortran -O0 -ffp-contract=off at MPAS's default single precision and at its
double precision, and driven by ``tools/mpas_ocean_oracle/run_tracer_advection.F90``.
The outputs are committed (tests/data/oracles/mpas_ocean_tracer_advection,
PROVENANCE.md). When ``GFORTRAN`` (or gfortran on PATH) and
``WOOF_MPAS_V842_SOURCE`` are set, the fixture itself is rebuilt and compared.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with MPAS only to a tolerance, in either precision.
2. A case generator whose bytes drift. The input digests are pinned.
3. A fixture that misses a branch: each vertical order, 2nd/3rd/4th order
   horizontal blending, budgets on and off, a three-level column, the scheme
   switched off.
4. A gate that cannot fail: a limiter dropped, a stencil regrouped, an upwind
   side swapped must produce mismatches. (The scheme's MAX/MIN ties and SIGN
   of a signed-zero flux only ever decide the sign of a zero term added to
   others, so no output word can show them; the fixture's signed-zero fluxes
   prove that much.)
5. Configurations v8.4.2 cannot run silently running anyway: refused by name.
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
from woof.verify import mpas_ocean_tracer_advection_oracle as oo

ROOT = Path(__file__).resolve().parents[1]
BOTH = ("r4", "r8")


def test_case_inputs_are_the_recorded_bytes():
    recorded = dict(reversed(line.split()) for line in
                    (oo.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines())
    assert recorded == {f"{case['name']}/{p}": oo.input_digest(case) for p in BOTH for case in oo.cases(p)}


@pytest.mark.parametrize("precision", BOTH)
def test_fixture_covers_the_branches(precision):
    fixture = oo.load_fixture(precision)
    cases = {case["name"]: case for case in oo.cases(precision)}
    assert set(fixture) == set(cases)
    configs = [c["config"] for c in cases.values()]
    assert {c["config_vert_tracer_adv_order"] for c in configs} == {2, 3, 4}
    assert {c["config_horiz_tracer_adv_order"] for c in configs} == {2, 3, 4}
    assert {c["config_compute_active_tracer_budgets"] for c in configs} == {True, False}
    for name, out in fixture.items():
        case = cases[name]
        tend = out["tracersTend/activeTracersTend"]
        before = case["pools"]["tracersTend"]["activeTracersTend"]
        changed = int(oo.word_mismatches(tend, before))
        if case["config"]["config_disable_tr_adv"]:
            assert changed == 0
        else:
            assert changed > tend.size // 2
            assert np.isfinite(tend).all()
            flux = case["pools"]["diagnostics"]["normalThicknessFlux"]
            assert (np.signbit(flux) & (flux == 0)).any()
        if case["config"]["config_compute_active_tracer_budgets"]:
            for field in oo.BUDGET_FIELDS:
                assert np.isfinite(out[f"diagnostics/{field}"]).all()


def test_what_v842_cannot_run_is_refused_by_name():
    from woof.hex.mpas_ocean import tracer_advection_v842 as ta
    base = dict(oo.BASE_CONFIG)
    ta.check_config(base)
    for knob, value in (("config_monotonic", False), ("config_num_halos", 2), ("config_horiz_tracer_adv_order", 5)):
        with pytest.raises(ConfigurationRefusal) as exc:
            ta.check_config(dict(base, **{knob: value}))
        assert knob in str(exc.value)


@requires_gpu
def test_columns_v842_reads_outside_are_refused():
    import cupy as cp
    from woof.hex.mpas_ocean import tracer_advection_v842 as ta
    case = oo.cases("r8")[0]
    mlc = cp.asarray(case["pools"]["mesh"]["maxLevelCell"])
    ta._check_columns(mlc, case["dims"]["nCells"], case["dims"]["nVertLevels"])
    mlc[5] = 1
    with pytest.raises(ConfigurationRefusal, match="maxLevelCell"):
        ta._check_columns(mlc, case["dims"]["nCells"], case["dims"]["nVertLevels"])
    with pytest.raises(ConfigurationRefusal, match="nVertLevels"):
        ta._check_columns(mlc, case["dims"]["nCells"], 2)


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
def test_gpu_matches_the_v842_oracle_word_for_word(precision):
    report = oo.measure(precision)
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


TEND = "tracersTend/activeTracersTend"


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
@pytest.mark.parametrize("original, mutated, case_name, field", [
    # the vertical upwind flux taking each side's tracer from the other
    ("const rk upwind = A(M(fminw(C(0.0), ww[kk]), cur[kk - 1]), M(fmaxw(C(0.0), ww[kk]), cur[kk]));",
     "const rk upwind = A(M(fminw(C(0.0), ww[kk]), cur[kk]), M(fmaxw(C(0.0), ww[kk]), cur[kk - 1]));",
     "defaults_3rd_order", TEND),
    # the horizontal antidiffusive flux left unlimited
    ("    *slot = A(M(fmaxw(C(0.0), flux), fminw(fluxOutgoing[cell1 * nl + k], fluxIncoming[cell2 * nl + k])),\n"
     "              M(fminw(C(0.0), flux), fminw(fluxIncoming[cell1 * nl + k], fluxOutgoing[cell2 * nl + k])));",
     "    *slot = flux;", "defaults_3rd_order", TEND),
    # the vertical antidiffusive flux left unlimited
    ("        hof[kk] = A(M(fmaxw(C(0.0), flux), fminw(fout[kk], fin[kk - 1])),\n"
     "                    M(fminw(C(0.0), flux), fminw(fout[kk - 1], fin[kk])));", "        hof[kk] = flux;",
     "defaults_3rd_order", TEND),
    # the 3rd-order vertical stencil's division distributed over its two terms
    ("    return D(S(M(w, S(M(C(7.0), A(q_i, q_im1)), A(q_ip1, q_im2))),\n"
     "               M(M(coef, absw(w)), S(S(q_ip1, q_im2), M(C(3.0), S(q_i, q_im1))))), C(12.0));",
     "    return S(D(M(w, S(M(C(7.0), A(q_i, q_im1)), A(q_ip1, q_im2))), C(12.0)),\n"
     "             D(M(M(coef, absw(w)), S(S(q_ip1, q_im2), M(C(3.0), S(q_i, q_im1)))), C(12.0)));",
     "defaults_3rd_order", TEND),
    # the 4th-order vertical stencil divided first
    ("return D(M(w, S(M(C(7.0), A(q_i, q_im1)), A(q_ip1, q_im2))), C(12.0));",
     "return M(D(w, C(12.0)), S(M(C(7.0), A(q_i, q_im1)), A(q_ip1, q_im2)));", "vert4_horiz4_budgets", TEND),
    # the provisional thickness's transport regrouped
    ("M(M(M(M(dt, invAreaCell1), dvEdge[iEdge]), (rk)edgeSignOnCell[iCell * maxEdges + i])",
     "M(M(M(dt, M(invAreaCell1, dvEdge[iEdge])), (rk)edgeSignOnCell[iCell * maxEdges + i])",
     "vert2_horiz2", TEND),
    # the horizontal budget taken after the vertical pass would add to it
    ("        if (withBudget) budget[t * nTracers + iTracer] = work;\n", "",
     "vert4_horiz4_budgets", "diagnostics/activeTracerHorizontalAdvectionTendency"),
])
def test_a_mutated_port_is_caught(monkeypatch, precision, original, mutated, case_name, field):
    from woof.hex.mpas_ocean import tracer_advection_v842
    real = tracer_advection_v842.module_source
    assert real(precision).count(original) == 1
    monkeypatch.setattr(tracer_advection_v842, "module_source", lambda p="r4": real(p).replace(original, mutated))
    tracer_advection_v842._module_on.cache_clear()
    try:
        case = next(c for c in oo.cases(precision) if c["name"] == case_name)
        got = oo.run_gpu(case)
        assert oo.word_mismatches(got[field], oo.load_fixture(precision)[case_name][field]) > 0
    finally:
        tracer_advection_v842._module_on.cache_clear()


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
                    str(tmp_path / "build"), "--piece", "tracer_advection", "--out", str(tmp_path / "fixture")],
                   check=True)
    for precision in BOTH:
        fresh = np.load(tmp_path / "fixture" / f"oracle-{precision}.npz")
        pinned = np.load(oo.FIXTURE_DIR / f"oracle-{precision}.npz")
        assert sorted(fresh.files) == sorted(pinned.files)
        for key in pinned.files:
            assert oo.word_mismatches(fresh[key], pinned[key]) == 0, (precision, key)
