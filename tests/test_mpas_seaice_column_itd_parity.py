"""MPAS-Seaice's column ITD thermodynamics (colpkg stage 3), word for word against v8.4.2.

The reference is MPAS-Model v8.4.2's ``column_itd_thermodynamics``
(``mpas_seaice_column.F``) and the column package under it
(``colpkg_step_therm2``: ``linear_itd``, ``add_new_ice``, ``lateral_melt``,
``reduce_area``, ``cleanup_itd``), compiled with gfortran -O0
-ffp-contract=off at RKIND 8. ``tools/mpas_seaice_oracle/run_column.F90``
(stage ``itd_thermodynamics``) runs ``colpkg_init_thermo`` and
``init_column_itd`` and then the routine at the clock's start; its ierr, abort
message and warnings are logged. The oracle's outputs are committed
(tests/data/oracles/mpas_seaice_column_itd, PROVENANCE.md), so the GPU
comparison runs without gfortran. The routine calls no libm (only
init_column_itd's category bounds take tanh, the correctly rounded one on the
GPU), so the default build runs it.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with MPAS only to a tolerance: every word of every
   field the routine may write must be bit-identical, an aborting case's
   included (the cells after the abort keep their words).
2. v8.4.2's quirks "fixed": linear_itd returns from a shift_ice abort with
   the snow enthalpies still shifted by rhos*Lfresh, and its
   hin_max(ncat) = 999.9 lands in categoryThicknessLimits.
3. An abort that the port reports differently: the first aborting owned
   cell, its ``indexToCellID`` and its message must be MPAS's, and the
   zap_snow_temperature warnings as many as MPAS logs.
4. A gate that cannot fail: a mutated evaluation order must produce
   mismatches.
5. A configuration the port does not carry running silently: each is
   refused by name.
6. A case generator whose bytes drift: the input digests are pinned.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
import _oracle_toolchains
from woof.hex.errors import ConfigurationRefusal
from woof.hex.mpas_seaice import column_itd_v842 as citd
from woof.verify import mpas_seaice_column_oracle as co

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = co.ITD_FIXTURE_DIR
ROUTINE = "column_itd_thermodynamics"


def _case(name):
    return co.make_case(*next(row for row in co.ITD_CASES if row[0] == name))


@pytest.fixture
def fresh_module():
    citd._module_on.cache_clear()
    yield
    citd._module_on.cache_clear()


def test_case_inputs_are_the_recorded_bytes():
    recorded = {}
    for line in (FIXTURE / "inputs.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        recorded[name] = digest
    assert recorded == {f"{case['name']}/r8": co.input_digest(case) for case in co.cases("r8", co.ITD_CASES)}


def test_fixture_covers_the_branches():
    fixture, logs = co.load_fixture(FIXTURE), co.load_logs(FIXTURE)
    cases = {c["name"]: c for c in co.cases("r8", co.ITD_CASES)}
    assert set(logs) == set(cases)
    outcomes = {name: co.oracle_abort(log, ROUTINE) for name, log in logs.items()}
    # the first aborting owned cell (indexToCellID), before a later one MPAS never reaches
    assert outcomes.pop("itd_abort_shift_ice") == (citd.SEAICE_ERROR_COL_ITD_THERM, 6,
                                                   f"{ROUTINE}: {citd.STOP_LABELS[1]}")
    assert outcomes.pop("itd_abort_area_bounds") == (citd.SEAICE_ERROR_COL_ITD_THERM, 1,
                                                     f"{ROUTINE}: {citd.STOP_LABELS[5]}")
    assert outcomes.pop("itd_delta_abort_zap_negative") == (citd.SEAICE_ERROR_COL_ITD_THERM, 7,
                                                            f"{ROUTINE}: {citd.STOP_LABELS[6]}")
    assert set(outcomes.values()) == {(0, None, None)}
    # zap_snow_temperature: a snow layer too warm and one too cold
    assert logs["itd_mushy_level_ponds"]["log"].count(citd.ZAP_WARNING) == 2
    for name, out in fixture.items():
        for key, value in out.items():
            assert np.isfinite(value).all(), (name, key)
    n_solve = co.MESH["nCellsSolve"]

    def got(name, key):
        return fixture[name][key]

    # linear_itd's hin_max(ncat) = 999.9 reaches categoryThicknessLimits; the delta function's does not
    for name in ("itd_mushy_level_ponds", "itd_abort_area_bounds"):
        assert got(name, "initial/categoryThicknessLimits")[-1] == 999.9
    assert got("itd_bl99_delta_function", "initial/categoryThicknessLimits")[-1] != 999.9
    # cell 4 (row 3): sub-puny categories zapped, flagged as newly formed
    plain = "itd_mushy_level_ponds"
    assert (got(plain, "tracers/iceAreaCategory")[3] == 0).all()
    assert (got(plain, "biogeochemistry/newlyFormedIce")[3] == 1).all()
    # cell 8 (row 7): a total area in (1, 1 + puny) scaled back to 1
    assert cases[plain]["pools"]["tracers_aggregate"]["iceAreaCell"][7] > 1.0
    assert got(plain, "tracers_aggregate/iceAreaCell")[7] == 1.0
    assert got(plain, "icestate/openWaterArea")[7] == 0.0
    # new ice in open water (row 1)
    assert got(plain, "tracers/iceAreaCategory")[1, 0, 0] > 0
    # the mushy frazil returned to the ocean only without config_update_ocean_fluxes
    assert "melt_growth_rates/frazilGrowthDiagnostic" in fixture[plain]
    assert "melt_growth_rates/frazilGrowthDiagnostic" not in fixture["itd_mushy_three_layers_update_ocean"]
    # pond water to the ocean only with topo ponds
    assert "ponds/pondFreshWaterFlux" in fixture["itd_mushy_topo_ponds_bare"]
    assert "ponds/pondFreshWaterFlux" not in fixture["itd_bl99_cesm_ponds"]
    # an abort leaves the later cells as they were
    shift = cases["itd_abort_shift_ice"]["pools"]["tracers"]["iceAreaCategory"]
    assert (got("itd_abort_shift_ice", "tracers/iceAreaCategory")[6:n_solve] == shift[6:n_solve]).all()


@pytest.mark.parametrize("overrides, match", [
    ({"config_thermodynamics_type": "zero layer"}, "zerolayer_check"),
    ({"config_use_aerosols": True}, "aerosols"),
    ({"config_use_column_biogeochemistry": True, "config_use_skeletal_biochemistry": True},
     "config_use_skeletal_biochemistry"),
])
def test_unported_configurations_are_refused_by_name(overrides, match):
    case = _case("itd_mushy_level_ponds")
    with pytest.raises(ConfigurationRefusal, match=match):
        citd.refuse_unported({**case["config"], **overrides}, case["dims"])


def test_more_categories_than_a_thread_holds_are_refused():
    case = _case("itd_mushy_level_ponds")
    with pytest.raises(ConfigurationRefusal, match="thickness categories"):
        citd.refuse_unported(case["config"], {**case["dims"], "nCategories": citd.ITD_MAXCAT + 1})


@requires_gpu
def test_gpu_matches_the_v842_oracle_word_for_word(fresh_module):
    report = co.measure_itd()
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize("original, mutated, field, case", [
    # the frazil volume's division moved onto dt
    ("double vi0new = -D(M(fnew, dt), qi0new);", "double vi0new = -M(fnew, D(dt, qi0new));",
     "melt_growth_rates/frazilFormation", "itd_mushy_level_ponds"),
    # the frazil spread over the categories (hsurp): the division distributed
    ("tr[ix[TI_QICE] + k] = D(A(M(tr[ix[TI_QICE] + k], vtmp), M(qi0new, vsurp)), vicen[n]);",
     "tr[ix[TI_QICE] + k] = A(D(M(tr[ix[TI_QICE] + k], vtmp), vicen[n]), D(M(qi0new, vsurp), vicen[n]));",
     "tracers/iceEnthalpy", "itd_bl99_update_ocean"),
    # fit_line's g1 associated the other way
    ("*g1 = M(M(M(2.0, dhr), wk1), S(wk2, 0.5));", "*g1 = M(M(2.0, dhr), M(wk1, S(wk2, 0.5)));",
     "tracers/iceAreaCategory", "itd_mushy_level_ponds"),
    # the mushy regridding's layer average
    ("trc2[k2 - 1] = D(M(rnilyr, t2), h2);", "trc2[k2 - 1] = M(rnilyr, D(t2, h2));",
     "tracers/iceSalinity", "itd_mushy_level_ponds"),
    # zap_small_areas' excess-area branch skipped (a total area in (1, 1 + puny) kept)
    ("} else if (*aice > 1.0 && *aice < A(1.0, PUNY)) {", "} else if (false) {",
     "tracers/iceAreaCategory", "itd_mushy_level_ponds"),
    # lateral melt's snow heat flux on the ice volume (a reordering of the ocean heat flux's
    # sum only moves words below the accumulator's last place, so the gate is a slip)
    ("double dfhocn = D(M(D(M(tr[ix[TI_QSNO] + k], rside), dt), vsnon[n]), (double)ITD_NSLYR);",
     "double dfhocn = D(M(D(M(tr[ix[TI_QSNO] + k], rside), dt), vicen[n]), (double)ITD_NSLYR);",
     "ocean_fluxes/oceanHeatFlux", "itd_bl99_cesm_ponds"),
    # zap_snow's fresh water at the ice density
    ("double xtmp = D(M(RHOS, *vsnon), dt);", "double xtmp = D(M(RHOI, *vsnon), dt);",
     "ocean_fluxes/oceanFreshWaterFlux", "itd_mushy_level_ponds"),
    # linear_itd's abort "fixed": the snow enthalpies shifted back before returning
    ("if (stop) return stop;   // the snow enthalpies still shifted, as MPAS returns",
     "if (stop) { for (int m = 0; m < ncat; ++m) trcrn[m * nt + T.idx[TI_QSNO]] = "
     "S(trcrn[m * nt + T.idx[TI_QSNO]], rhosL); return stop; }",
     "tracers/snowEnthalpy", "itd_abort_shift_ice"),
])
def test_a_mutated_evaluation_order_is_caught(monkeypatch, fresh_module, original, mutated, field, case):
    source = citd.module_source
    assert source(7).count(original) == 1
    monkeypatch.setattr(citd, "module_source", lambda n: source(n).replace(original, mutated))
    citd._module_on.cache_clear()
    got, _ = co.run_gpu(_case(case))
    want = co.load_fixture(FIXTURE)[case][field]
    assert co.word_mismatches(got[field], want) > 0


def _oracle_toolchain() -> tuple[str, Path] | None:
    return _oracle_toolchains.mpas_v842("core_seaice")


@pytest.mark.skipif(_oracle_toolchain() is None or platform.machine() != "x86_64"
                    or platform.libc_ver() != ("glibc", "2.44"),
                    reason="colpkg_init_itd's category bounds are the x86-64 glibc 2.44 host's tanh "
                           "(the fixture's, PROVENANCE.md); it needs gfortran (pixi -e mpas-oracle) and "
                           "WOOF_MPAS_V842_SOURCE")
def test_the_oracle_rebuilt_here_writes_the_fixture(tmp_path):
    gfortran, source = _oracle_toolchain()
    subprocess.run(["bash", str(ROOT / "tools" / "mpas_seaice_oracle" / "build.sh"), str(source),
                    str(tmp_path / "build")], check=True, env={**os.environ, "GFORTRAN": gfortran})
    subprocess.run([sys.executable, str(ROOT / "tools" / "mpas_seaice_oracle" / "make_column_fixture.py"),
                    str(tmp_path / "build"), "--cases", "itd", "--out", str(tmp_path / "fixture")], check=True)
    fresh = np.load(tmp_path / "fixture" / "oracle-r8.npz")
    pinned = np.load(FIXTURE / "oracle-r8.npz")
    assert sorted(fresh.files) == sorted(pinned.files)
    for key in pinned.files:
        assert co.word_mismatches(fresh[key], pinned[key]) == 0, key
    for name in ("column-logs.json", "inputs.sha256"):
        assert (tmp_path / "fixture" / name).read_text() == (FIXTURE / name).read_text()
