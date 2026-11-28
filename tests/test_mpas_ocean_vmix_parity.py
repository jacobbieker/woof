"""MPAS-Ocean's implicit vertical mixing, word for word against v8.4.2, at RKIND 4 and 8.

The reference is MPAS-Model v8.4.2's own ``ocn_vmix_init`` and
``ocn_vmix_implicit`` (``mpas_ocn_vmix.F`` and its const, tanh, Richardson and
Redi coefficient modules compiled verbatim, the real equation of state, a
stand-in for CVMix), with gfortran -O0 -ffp-contract=off at MPAS's default
single precision and at its double precision, driven by
``tools/mpas_ocean_oracle/run_vmix.F90``. The outputs are committed
(tests/data/oracles/mpas_ocean_vmix, PROVENANCE.md). When ``GFORTRAN`` (or
gfortran on PATH) and ``WOOF_MPAS_V842_SOURCE`` are set, the fixture itself
is rebuilt and compared.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with MPAS only to a tolerance, in either precision.
2. A case generator whose bytes drift. The input digests are pinned.
3. A fixture that misses a branch: constant and Richardson coefficients (both
   signs of the Richardson number), the bottom drag on and off, one-level
   edges, the budget, velocity or tracer mixing switched off.
4. A port that "fixes" v8.4.2: the bottom drag's level, the single-precision
   -1e34 below a column.
5. A gate that cannot fail: a mutated solve or coefficient must produce
   mismatches.
6. A coefficient scheme this port does not carry silently skipped: refused by name.
7. Redi's vertical diffusion "fixed" by default: v8.4.2 adds k33 without
   config_Redi_kappa; only ``woof_redi_k33 = 'kappa'`` scales it, and that
   opt-in adds exactly kappa * k33.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
import _oracle_toolchains
from woof.hex.errors import ConfigurationRefusal
from woof.verify import mpas_ocean_vmix_oracle as oo

ROOT = Path(__file__).resolve().parents[1]
BOTH = ("r4", "r8")
VEL = "state/normalVelocity"
TRACERS = "tracers/activeTracers"


def test_case_inputs_are_the_recorded_bytes():
    recorded = dict(reversed(line.split()) for line in
                    (oo.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines())
    assert recorded == {f"{case['name']}/{p}": oo.input_digest(case) for p in BOTH for case in oo.cases(p)}


@pytest.mark.parametrize("precision", BOTH)
def test_fixture_covers_the_branches(precision):
    fixture = oo.load_fixture(precision)
    cases = {case["name"]: case for case in oo.cases(precision)}
    assert set(fixture) == set(cases)
    sentinel = np.float32(-1e34).astype(oo.PRECISIONS[precision])
    positive = negative = one_level_edges = 0
    for name, out in fixture.items():
        case = cases[name]
        c = case["config"]
        mesh = case["pools"]["mesh"]
        nc = case["dims"]["nCells"]
        if c["config_use_rich_visc"] or c["config_use_rich_diff"]:
            ri = out["diagnostics/RiTopOfEdge"]
            positive += int((ri > 0).sum())
            negative += int((ri < 0).sum())
        if not c["config_disable_tr_vmix"]:
            below = np.arange(case["dims"]["nVertLevels"])[None, :] >= mesh["maxLevelCell"][:nc, None]
            assert (out[TRACERS][:nc][below] == sentinel).all()
        if not c["config_disable_vel_vmix"]:
            one_level_edges += int((mesh["maxLevelEdgeTop"] == 1).sum())
        assert all(np.isfinite(v).all() for k, v in out.items() if k != TRACERS)
    assert positive > 0 and negative > 0 and one_level_edges > 0


def test_schemes_not_ported_are_refused_by_name():
    from woof.hex.mpas_ocean import vmix_v842 as vm
    vm.check_config(dict(oo.BASE_CONFIG))
    vm.check_config(dict(oo.BASE_CONFIG, config_use_standardGM=True))
    for knob in ("config_use_tanh_visc", "config_use_tanh_diff"):
        with pytest.raises(ConfigurationRefusal) as exc:
            vm.check_config(dict(oo.BASE_CONFIG, **{knob: True}))
        assert knob in str(exc.value)
    # CVMix is ported (tests/test_mpas_ocean_cvmix_parity.py) at RKIND 8 only
    from woof.hex.mpas_ocean import cvmix_v842
    vm.check_config(dict(oo.BASE_CONFIG, config_use_cvmix=True))
    with pytest.raises(ConfigurationRefusal, match="config_use_cvmix"):
        cvmix_v842.check_config(dict(oo.BASE_CONFIG, config_use_cvmix=True), "r4")
    with pytest.raises(ConfigurationRefusal, match="woof_redi_k33"):
        vm.check_config(dict(oo.BASE_CONFIG, config_use_standardGM=True, woof_redi_k33="E3SM"))
    # KPP without CVMix: v8.4.2 ignores the option silently, so it is refused naming both
    for check in (lambda c: vm.check_config(c), lambda c: cvmix_v842.check_config(c, "r8")):
        with pytest.raises(ConfigurationRefusal, match="config_use_cvmix_kpp") as exc:
            check(dict(oo.BASE_CONFIG, config_use_cvmix=False, config_use_cvmix_kpp=True))
        assert "config_use_cvmix off" in str(exc.value)


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
def test_the_redi_k33_kappa_scaling_is_opt_in(precision):
    case = next(c for c in oo.cases(precision) if c["name"] == "redi_const")
    real = oo.PRECISIONS[precision]
    nc, diff = case["dims"]["nCells"], "diagnostics/vertDiffTopOfCell"
    k33 = case["pools"]["diagnostics"]["k33"][:nc]
    kappa = real(case["config"]["config_Redi_kappa"])
    without = oo.run_gpu(dict(case, config=dict(case["config"], config_use_standardGM=False)))[diff][:nc]
    v842 = oo.run_gpu(case)[diff][:nc]
    scaled = oo.run_gpu(dict(case, config=dict(case["config"], woof_redi_k33="kappa")))[diff][:nc]
    assert oo.word_mismatches(v842, (without + k33).astype(real)) == 0
    assert oo.word_mismatches(scaled, (without + (kappa * k33).astype(real)).astype(real)) == 0
    assert oo.word_mismatches(scaled, v842) > 0


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
def test_gpu_matches_the_v842_oracle_word_for_word(precision):
    report = oo.measure(precision)
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
@pytest.mark.parametrize("original, mutated, case_name, field, precisions", [
    # the bottom drag read at the column's bottom level ("fixing" v8.4.2's level 2 for one-level edges)
    ("    const int kd = (N > 2 ? N : 2) - 1;", "    const int kd = N - 1;", "rich_diff_only_one_level_columns",
     VEL, BOTH),
    # the velocity back substitution divided term by term
    ("for (int i = N - 2; i >= 0; --i) x[i] = D(S(rT[i], M(Cc[i], x[i + 1])), bT[i]);",
     "for (int i = N - 2; i >= 0; --i) x[i] = S(D(rT[i], bT[i]), D(M(Cc[i], x[i + 1]), bT[i]));",
     "const", VEL, BOTH),
    # the tracer diagonal formed as 1 - (A + C)
    ("for (int k = 0; k < N; ++k) Bb[k] = S(S(C(1.0), Aa[k]), Cc[k]);",
     "for (int k = 0; k < N; ++k) Bb[k] = S(C(1.0), A(Aa[k], Cc[k]));", "const", TRACERS, BOTH),
    # the Richardson viscosity's denominator squared by two divisions
    ("rk v = A(A(vertViscTopOfEdge[at], bkrdVisc), D(richMix, M(x, x)));",
     "rk v = A(A(vertViscTopOfEdge[at], bkrdVisc), D(D(richMix, x), x));", "rich_budgets",
     "diagnostics/vertViscTopOfEdge", BOTH),
    # the cell shear divided by the area instead of multiplied by its inverse
    ("M(M(M(M(C(0.5), dcEdge[iEdge]), dvEdge[iEdge]), du2TopOfEdge[iEdge * (nl + 1) + k]),\n"
     "                           invAreaCell)",
     "D(M(M(M(C(0.5), dcEdge[iEdge]), dvEdge[iEdge]), du2TopOfEdge[iEdge * (nl + 1) + k]),\n"
     "                           areaCell[iCell])", "rich_budgets", "diagnostics/RiTopOfCell", BOTH),
    # the sentinel below a column as an RKIND literal: RKIND 8 widens the binary32 -1e34
    ("#define SENTINEL_DEFAULT ((rk)-1e34f)", "#define SENTINEL_DEFAULT C(-1e34)", "const", TRACERS, ("r8",)),
    # Redi's k33 scaled by config_Redi_kappa by default ("fixing" v8.4.2)
    ("kappaScaled ? M(kappa, k33[t]) : k33[t]", "M(kappa, k33[t])", "redi_const",
     "diagnostics/vertDiffTopOfCell", BOTH),
])
def test_a_mutated_port_is_caught(monkeypatch, precision, original, mutated, case_name, field, precisions):
    if precision not in precisions:
        pytest.skip(f"the {precision} words do not depend on this expression")
    from woof.hex.mpas_ocean import vmix_v842
    real = vmix_v842.module_source
    assert real(precision).count(original) == 1
    monkeypatch.setattr(vmix_v842, "module_source", lambda p="r4": real(p).replace(original, mutated))
    vmix_v842._module_on.cache_clear()
    try:
        case = next(c for c in oo.cases(precision) if c["name"] == case_name)
        got = oo.run_gpu(case)
        assert oo.word_mismatches(got[field], oo.load_fixture(precision)[case_name][field]) > 0
    finally:
        vmix_v842._module_on.cache_clear()


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
                    str(tmp_path / "build"), "--piece", "vmix", "--out", str(tmp_path / "fixture")], check=True)
    for precision in BOTH:
        fresh = np.load(tmp_path / "fixture" / f"oracle-{precision}.npz")
        pinned = np.load(oo.FIXTURE_DIR / f"oracle-{precision}.npz")
        assert sorted(fresh.files) == sorted(pinned.files)
        for key in pinned.files:
            assert oo.word_mismatches(fresh[key], pinned[key]) == 0, (precision, key)
