"""MPAS-Ocean's CVMix vertical mixing, word for word against v8.4.2 with CVMix v0.84-beta, at RKIND 8.

The reference is MPAS-Model v8.4.2's own ``ocn_vmix_init`` and
``ocn_vmix_implicit`` with ``mpas_ocn_vmix.F``, its coefficient modules and
``mpas_ocn_vmix_cvmix.F`` compiled verbatim against CVMix v0.84-beta's ten
modules (verbatim, downloaded and sha256-pinned, not in the repository), the
real equation of state, with gfortran -O0 -ffp-contract=off at RKIND 8, driven
by ``tools/mpas_ocean_oracle/run_vmix.F90`` (``run_vmix_cvmix_r8``). CVMix
does not compile at RKIND 4, which is refused. The outputs are committed
(tests/data/oracles/mpas_ocean_cvmix, PROVENANCE.md). When ``GFORTRAN`` (or
gfortran on PATH), ``WOOF_MPAS_V842_SOURCE`` and ``CVMIX_SOURCE`` are set,
the fixture itself is rebuilt and compared.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with MPAS and CVMix only to a tolerance.
2. A case generator whose bytes drift. The input digests are pinned.
3. A fixture that misses a branch: both KPP matchings, the three
   interpolations, the boundary layer at a crossing, at the column's bottom,
   at half the top layer, under sea ice and fixed; stable and unstable
   forcing, zero friction velocity; the enhanced diffusivity with the
   layer's base above and below the last centre; PP and KPP shear; the step
   and the (dead) negative-trigger convection; the smoothing on and off;
   distinct halo extents; KPP's non-local flux in the implicit solve.
4. A port that "fixes" v8.4.2: the bottom Nsqr copied over at a full column,
   convection from the top without KPP.
5. A gate that cannot fail: a mutated expression must produce mismatches.
6. An option this port does not carry silently run: refused by name (RKIND 4,
   the v0.84-beta-only Ekman cap and unenhanced diffusivity, the options MPAS
   itself stops on).
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
from woof.verify import mpas_ocean_cvmix_oracle as oo

ROOT = Path(__file__).resolve().parents[1]
VISC = "diagnostics/vertViscTopOfCell"
DIFF = "diagnostics/vertDiffTopOfCell"
NONLOCAL = "diagnostics/vertNonLocalFlux"
BLD = "diagnostics/boundaryLayerDepth"
KOBL = "diagnostics/indexBoundaryLayerDepth"
TRACERS = "tracers/activeTracers"


def test_case_inputs_are_the_recorded_bytes():
    recorded = dict(reversed(line.split()) for line in
                    (oo.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines())
    assert recorded == {f"{case['name']}/r8": oo.input_digest(case) for case in oo.cases("r8")}


def _zt(case) -> np.ndarray:
    """|zt_cntr(maxLevelCell)| of each cell, as the Fortran builds it."""
    h = case["pools"]["state"]["layerThickness"]
    mlc = case["pools"]["mesh"]["maxLevelCell"]
    out = np.empty(len(mlc) - 1)
    for i in range(len(out)):
        zw = 0.0
        for k in range(mlc[i] - 1):
            zw = zw - h[i, k]
        out[i] = abs(zw - h[i, mlc[i] - 1] / 2.0)
    return out


def test_fixture_covers_the_branches():
    fixture = oo.load_fixture("r8")
    cases = {case["name"]: case for case in oo.cases("r8")}
    assert set(fixture) == set(cases)
    quarter = three_quarters = at_bottom = at_half = under_ice = stable = unstable = calm = 0
    for name, out in fixture.items():
        case = cases[name]
        c = case["config"]
        n3 = int(case["pools"]["mesh"]["nCellsArray"][2])
        assert all(np.isfinite(v).all() for k, v in out.items() if k != TRACERS)
        if not c["config_use_cvmix_kpp"]:
            assert (out[NONLOCAL][:n3] == 0).all()
            continue
        kobl = out[KOBL][:n3]
        frac = kobl - np.floor(kobl)
        quarter += int((frac == 0.25).sum())
        three_quarters += int((frac == 0.75).sum())
        bld = out[BLD][:n3]
        at_bottom += int((bld == _zt(case)[:n3]).sum())
        at_half += int((bld == case["pools"]["state"]["layerThickness"][:n3, 0] / 2.0).sum())
        ice = case["pools"]["forcing"]["iceFraction"][:n3]
        under_ice += int(((ice > 0.15) & (bld == c["configure_cvmix_kpp_minimum_OBL_under_sea_ice"])).sum())
        b = case["pools"]["diagnostics"]["surfaceBuoyancyForcing"][:n3]
        u = case["pools"]["diagnostics"]["surfaceFrictionVelocity"][:n3]
        stable += int((b > 0).sum())
        unstable += int((b <= 0).sum())
        calm += int((u == 0).sum())
        # the non-local flux is non-zero only under unstable forcing
        nonlocal_rows = np.abs(out[NONLOCAL][:n3, :, 0]).sum(axis=1) > 0
        assert not (nonlocal_rows & (b > 0)).any() and (nonlocal_rows & (b <= 0)).any()
    assert min(quarter, three_quarters, at_bottom, at_half, under_ice, stable, unstable, calm) > 0


def test_options_not_ported_are_refused_by_name():
    from woof.hex.mpas_ocean import cvmix_v842 as cv
    from woof.hex.mpas_ocean import vmix_v842 as vm
    for _, _, _, _, overrides in oo.CASES:
        vm.check_config(dict(oo.BASE_CONFIG, **{k: v for k, v in overrides.items() if not k.startswith("_")}))
    kpp = dict(oo.BASE_CONFIG, **oo.KPP)
    with pytest.raises(ConfigurationRefusal, match="config_use_cvmix.*RKIND 4"):
        cv.check_config(kpp, "r4")
    refused = (
        ("config_cvmix_kpp_EkmanOBL", True), ("config_cvmix_kpp_use_enhanced_diff", False),
        ("config_cvmix_kpp_matching", "MatchGradient"), ("config_cvmix_kpp_matching", "ParabolicNonLocal"),
        ("config_cvmix_kpp_interpolationOMLType", "LMD94"), ("config_cvmix_shear_mixing_scheme", "KPP94"),
        ("config_cvmix_convective_basedOnBVF", False), ("config_cvmix_kpp_surface_layer_extent", 0.0),
        ("config_cvmix_kpp_surface_layer_extent", 1.5))
    for knob, value in refused:
        with pytest.raises(ConfigurationRefusal) as exc:
            vm.check_config(dict(kpp, **{knob: value}))
        assert knob in str(exc.value)
    with pytest.raises(ConfigurationRefusal, match="config_cvmix_shear_mixing_scheme"):
        vm.check_config(dict(kpp, config_use_cvmix_background=False))
    # Ekman and the enhancement only matter with KPP
    vm.check_config(dict(oo.BASE_CONFIG, config_cvmix_kpp_EkmanOBL=True, config_cvmix_kpp_use_enhanced_diff=False))


def test_a_caller_without_the_kpp_inputs_is_refused_by_name():
    from woof.hex.mpas_ocean import vmix_v842 as vm
    state = {"layerThickness": np.zeros((2, 2)), "normalVelocity": np.zeros((2, 2))}
    for knob, config in (("config_use_cvmix_kpp", dict(oo.BASE_CONFIG, **oo.KPP)),
                         ("config_cvmix_kpp_nonlocal_with_implicit_mix",
                          dict(oo.BASE_CONFIG, config_cvmix_kpp_nonlocal_with_implicit_mix=True))):
        with pytest.raises(ConfigurationRefusal, match=knob):
            vm.vmix_implicit(dt=1.0, mesh={}, state=state, tracers={}, diagnostics={}, configs=config)


@requires_gpu
def test_the_non_local_flux_keeps_its_group_in_a_wider_pool():
    """vertNonLocalFlux(nGroup, nVertLevelsP1, nCells) with more than one group:
    the coefficients write group index_vertNonLocalFluxTemp and the implicit
    solve reads group 1, as in v8.4.2's one-member var_array."""
    import copy
    case = copy.deepcopy(next(c for c in oo.cases("r8") if c["name"] == "kpp_nonlocal_implicit_budgets"))
    diag = case["pools"]["diagnostics"]
    one = diag["vertNonLocalFlux"]
    diag["vertNonLocalFlux"] = np.concatenate([one, one[..., :1] + 7.0], axis=-1)
    got = oo.run_gpu(case)
    ref = oo.load_fixture("r8")["kpp_nonlocal_implicit_budgets"]
    assert oo.word_mismatches(got[NONLOCAL][..., :1], ref[NONLOCAL]) == 0
    assert oo.word_mismatches(got[TRACERS], ref[TRACERS]) == 0


@requires_gpu
def test_gpu_matches_the_v842_oracle_word_for_word():
    report = oo.measure("r8")
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
def test_cvmix_at_rkind4_is_refused_on_the_run():
    from woof.verify import mpas_ocean_vmix_oracle as vo
    case = next(c for c in vo.cases("r4") if c["name"] == "const")
    case = dict(case, config=dict(case["config"], **oo.CVMIX_CONFIG))
    with pytest.raises(ConfigurationRefusal, match="RKIND 4"):
        vo.run_gpu(case)


@requires_gpu
@pytest.mark.parametrize("original, mutated, case_name, field", [
    # the unresolved shear's bottom Nsqr "fixed" (v8.4.2 copies the one above over a full column's)
    ("const int k = mlc + 1 < nl ? mlc + 1 : nl;", "const int k = mlc + 1;", "kpp_simple_shapes",
     "diagnostics/bulkRichardsonNumber"),
    # the unresolved shear's scale divided by the von Karman constant twice
    ("const rk vtc = D(SQRT(D(0.2, M(p.c_s, p.sle))), M(p.vonkar, p.vonkar));",
     "const rk vtc = D(D(SQRT(D(0.2, M(p.c_s, p.sle))), p.vonkar), p.vonkar);",
     "kpp_linear_unsmoothed", "diagnostics/bulkRichardsonNumber"),
    # MatchBoth's G(1) scaled by twice the depth
    ("mAt1 = wmObl != 0.0 ? D(mObl, M(wmObl, obl)) : 0.0;",
     "mAt1 = wmObl != 0.0 ? D(mObl, M(wmObl, M(2.0, obl))) : 0.0;",
     "kpp_match_both_quadratic", VISC),
    # the bulk Richardson number's numerator in another order
    ("const rk num = -M(M(scaling, zt[kt]), rib[kt - 1]);", "const rk num = -M(scaling, M(zt[kt], rib[kt - 1]));",
     "kpp_simple_shapes", "diagnostics/bulkRichardsonNumber"),
    # the quadratic interpolation's coefficients accumulated the other way round
    ("for (int k2 = 1; k2 <= 3; ++k2) c[k2] = A(c[k2], M(mi[4 - k2][k], rhs[k]));",
     "for (int k2 = 1; k2 <= 3; ++k2) c[k2] = A(c[k2], M(mi[k2][k], rhs[k]));",
     "kpp_simple_shapes", BLD),
    # the cubic root found to a looser tolerance
    ("if (fabs(f) < 1e-12) break;", "if (fabs(f) < 1e-6) break;", "kpp_only_cubic", BLD),
    # the enhanced diffusivity skipped
    ("        // cvmix_kpp_compute_enhanced_diff (ktup is kwup or kwup - 1)\n        {",
     "        // cvmix_kpp_compute_enhanced_diff (ktup is kwup or kwup - 1)\n        if (0) {", "kpp_simple_shapes",
     DIFF),
    # the non-local flux under stable forcing too
    ("if (!lstable) tnl[kw] = M(p.nonlocal, cv_eval_cubic(tshape, sigma[kw]));",
     "tnl[kw] = M(p.nonlocal, cv_eval_cubic(tshape, sigma[kw]));", "kpp_simple_shapes", NONLOCAL),
    # the smoothing weights each neighbour once, not twice
    ("smooth = A(smooth, M(M(M(2.0, (rk)mask), areaCell[n]), boundaryLayerDepth[n]));",
     "smooth = A(smooth, M(M((rk)mask, areaCell[n]), boundaryLayerDepth[n]));", "kpp_simple_shapes", BLD),
    # convection without KPP from below the "boundary layer" ("fixing" the unset kOBL_depth)
    ("const int oblInd = (int)floor(kobl + 0.5) + 1;", "const int oblInd = 1 << 20;", "convection_shear_no_kpp",
     DIFF),
    # PP's background left in
    ("                Md[k] = S(Md[k], rp[RP_BG_VISC]);", "                Md[k] = Md[k];", "convection_shear_no_kpp",
     VISC),
    # the Richardson number filter as a Gauss-Seidel sweep
    ("for (int k = 2; k <= mlc; ++k) ris[k] = D(A(A(rit[k - 1], M(2.0, rit[k])), rit[k + 1]), 4.0);",
     "for (int k = 2; k <= mlc; ++k) ris[k] = D(A(A(ris[k - 1], M(2.0, rit[k])), rit[k + 1]), 4.0);",
     "kpp_simple_shapes", VISC),
    # the edge viscosity averaged cell1 + cell2
    ("M(0.5, A(vertViscTopOfCell[(size_t)c2 * (nl + 1) + k], vertViscTopOfCell[(size_t)c1 * (nl + 1) + k]))",
     "D(A(vertViscTopOfCell[(size_t)c2 * (nl + 1) + k], vertViscTopOfCell[(size_t)c1 * (nl + 1) + k]), 3.0)",
     "kpp_simple_shapes", "diagnostics/vertViscTopOfEdge"),
])
def test_a_mutated_port_is_caught(monkeypatch, original, mutated, case_name, field):
    from woof.hex.mpas_ocean import cvmix_v842
    real = cvmix_v842.module_source
    assert real("r8").count(original) == 1
    monkeypatch.setattr(cvmix_v842, "module_source", lambda p="r8": real(p).replace(original, mutated))
    cvmix_v842._module_on.cache_clear()
    try:
        case = next(c for c in oo.cases("r8") if c["name"] == case_name)
        got = oo.run_gpu(case)
        assert oo.word_mismatches(got[field], oo.load_fixture("r8")[case_name][field]) > 0
    finally:
        cvmix_v842._module_on.cache_clear()


@requires_gpu
def test_a_mutated_nonlocal_rhs_is_caught(monkeypatch):
    from woof.hex.mpas_ocean import vmix_v842
    real = vmix_v842.module_source
    original = "for (int j = 0; j < nTracers; ++j) x[j] = A(tr[j], D(M(M(dt, sf[j]), -f[nGroup]), h[0]));"
    assert real("r8").count(original) == 1
    monkeypatch.setattr(vmix_v842, "module_source",
                        lambda p="r4": real(p).replace(original, original.replace("-f[nGroup]", "-f[0]")))
    vmix_v842._module_on.cache_clear()
    try:
        case = next(c for c in oo.cases("r8") if c["name"] == "kpp_nonlocal_implicit_budgets")
        got = oo.run_gpu(case)
        assert oo.word_mismatches(got[TRACERS], oo.load_fixture("r8")[case["name"]][TRACERS]) > 0
    finally:
        vmix_v842._module_on.cache_clear()


def _oracle_toolchain() -> tuple[str, Path, Path] | None:
    source = os.environ.get("WOOF_MPAS_V842_SOURCE")
    cvmix = os.environ.get("CVMIX_SOURCE")
    gfortran = os.environ.get("GFORTRAN") or shutil.which("gfortran")
    if not source or not cvmix or not gfortran or not Path(source, "src", "core_ocean").is_dir():
        return None
    return gfortran, Path(source), Path(cvmix)


@pytest.mark.skipif(_oracle_toolchain() is None,
                    reason="the CVMix oracle build needs gfortran (pixi -e mpas-oracle), WOOF_MPAS_V842_SOURCE "
                           "naming an MPAS-Model v8.4.2 tree and CVMIX_SOURCE naming CVMix v0.84-beta's src/shared")
def test_the_oracle_rebuilt_here_writes_the_fixture(tmp_path):
    gfortran, source, cvmix = _oracle_toolchain()
    subprocess.run(["bash", str(ROOT / "tools" / "mpas_ocean_oracle" / "build.sh"), str(source),
                    str(tmp_path / "build")], check=True,
                   env={**os.environ, "GFORTRAN": gfortran, "CVMIX_SOURCE": str(cvmix)})
    subprocess.run([sys.executable, str(ROOT / "tools" / "mpas_ocean_oracle" / "make_fixture.py"),
                    str(tmp_path / "build"), "--piece", "cvmix", "--out", str(tmp_path / "fixture")], check=True)
    fresh = np.load(tmp_path / "fixture" / "oracle-r8.npz")
    pinned = np.load(oo.FIXTURE_DIR / "oracle-r8.npz")
    assert sorted(fresh.files) == sorted(pinned.files)
    for key in pinned.files:
        assert oo.word_mismatches(fresh[key], pinned[key]) == 0, key
