"""MPAS-Seaice's column ridging (colpkg stage 4), word for word against v8.4.2.

The reference is MPAS-Model v8.4.2's ``column_ridging``
(``mpas_seaice_column.F``) and the column package under it
(``colpkg_step_ridge``: ``ridge_ice`` with ``ridge_prep``, ``ridge_itd`` and
``ridge_shift``, then ``cleanup_itd``), compiled with gfortran -O0
-ffp-contract=off at RKIND 8. ``tools/mpas_seaice_oracle/run_column.F90``
(stage ``ridging``) runs ``colpkg_init_thermo`` and ``init_column_itd`` and
then the routine; its ierr, abort message and warnings are logged. The
oracle's outputs are committed (tests/data/oracles/mpas_seaice_column_ridge,
PROVENANCE.md), so the GPU comparison runs without gfortran. The routine's
exp is glibc's, which the GPU transcribes under MIT (glibc_flt64.cuh), so the
default build runs it.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with MPAS only to a tolerance: every word of every
   field the routine may write must be bit-identical, an aborting case's
   included (the cells after the abort keep their words).
2. v8.4.2's quirks "fixed": ridge_prep's hin_max(ncat) = 1e8 lands in
   categoryThicknessLimits; the redistribution diagnostics are written only
   for the thinnest category's ridges under the exponential redistribution;
   a ridging abort leaves the categories already ridged changed.
3. An abort that the port reports differently: the first aborting owned
   cell, its ``indexToCellID`` and its message must be MPAS's, and the
   ``Repeat ridging`` and zap_snow_temperature warnings the lines MPAS logs.
   Where v8.4.2 itself dies (ridge_shift's aborts write a 512-character
   label into a 256-character warning), the port must stop with the same
   runtime error rather than invent an outcome.
4. A gate that cannot fail: a mutated evaluation order must produce
   mismatches.
5. A configuration the port does not carry running silently: each is
   refused by name.
6. A case generator whose bytes drift: the input digests are pinned.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from woof.hex.errors import ConfigurationRefusal
from woof.hex.mpas_seaice import column_itd_v842 as citd
from woof.hex.mpas_seaice import column_ridge_v842 as crdg
from woof.verify import mpas_seaice_column_oracle as co

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = co.RIDGE_FIXTURE_DIR
ROUTINE = "column_ridging"


def _case(name):
    return co.make_case(*next(row for row in co.RIDGE_CASES if row[0] == name))


@pytest.fixture
def fresh_module():
    crdg._module_on.cache_clear()
    yield
    crdg._module_on.cache_clear()


def test_case_inputs_are_the_recorded_bytes():
    recorded = {}
    for line in (FIXTURE / "inputs.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        recorded[name] = digest
    assert recorded == {f"{case['name']}/r8": co.input_digest(case) for case in co.cases("r8", co.RIDGE_CASES)}


def test_fixture_covers_the_branches():
    fixture, logs = co.load_fixture(FIXTURE), co.load_logs(FIXTURE)
    cases = {c["name"]: c for c in co.cases("r8", co.RIDGE_CASES)}
    assert set(logs) == set(cases)
    # ridge_shift's open-water abort: v8.4.2 dies writing its stop label (no outputs)
    assert logs["ridge_negative_opening_dies"]["crash"] == crdg.RUNTIME_ERRORS[2]
    assert "IERR" not in logs["ridge_negative_opening_dies"]["tracer_object"]
    assert not any(key.startswith("ridge_negative_opening_dies/") for key in np.load(FIXTURE / "oracle-r8.npz").files)
    outcomes = {name: co.oracle_abort(log, ROUTINE) for name, log in logs.items() if "crash" not in log}
    # the first aborting owned cell (indexToCellID), before a later one MPAS never reaches
    assert outcomes.pop("ridge_abort_iterations") == (crdg.SEAICE_ERROR_COL_RIDGING, 6,
                                                      f"{ROUTINE}: {crdg.STOP_LABELS[1]}")
    assert outcomes.pop("ridge_abort_cleanup") == (crdg.SEAICE_ERROR_COL_RIDGING, 8,
                                                   f"{ROUTINE}: {crdg.STOP_LABELS[5]}")
    assert set(outcomes.values()) == {(0, None, None)}
    for name, out in fixture.items():
        for key, value in out.items():
            if key != "ridging/ratioRidgeThicknessToIce":
                assert np.isfinite(value).all(), (name, key)
    # ice without volume (row 8) under Hibler's redistribution: krdg = hrmean/0 = +inf, MPAS's word
    for name in ("ridge_bl99_thorndike_hibler", "ridge_exponential_hibler_topo_ponds"):
        krdg = fixture[name]["ridging/ratioRidgeThicknessToIce"]
        assert np.isposinf(krdg[8, 0]) and np.isfinite(np.delete(krdg.ravel(), 8 * krdg.shape[1])).all()
    n_solve = co.MESH["nCellsSolve"]

    def got(name, key):
        return fixture[name][key]

    plain = "ridge_mushy_level_ponds"
    warnings = co.oracle_ridge_warnings(logs[plain])
    # ridge_ice repeats where the closing was cut back; zap_snow_temperature's warning (row 4)
    assert any(line.startswith("WARN  Repeat ridging") for line in warnings)
    assert citd.ZAP_WARNING in warnings
    assert len([line for line in co.oracle_ridge_warnings(logs["ridge_abort_iterations"])
                if line.startswith("WARN  Repeat ridging")]) >= 20
    # ridge_prep's hin_max(ncat) = 1e8 reaches categoryThicknessLimits
    for name in fixture:
        assert got(name, "initial/categoryThicknessLimits")[-1] == crdg.BIG
    # ridging happened: area lost where the forcing closes (row 0), none without forcing (row 7)
    assert got(plain, "ridging/areaLossRidge")[0] > 0
    assert got(plain, "ridging/areaLossRidge")[7] == 0
    # divergence opens water (row 3)
    assert got(plain, "ridging/openingRateRidge")[3] > 0
    # the redistribution diagnostics: written under the exponential redistribution, kept under Hibler's
    assert "ridging/fractionNewRidgeArea" in fixture[plain]
    assert "ridging/fractionNewRidgeArea" not in fixture["ridge_bl99_thorndike_hibler"]
    # rafting under the exponential redistribution only
    assert "ridging/raftingIceArea" in fixture[plain]
    # the pond water lost with the ridging ice: topo ponds only
    assert "ponds/pondFreshWaterFlux" in fixture["ridge_exponential_hibler_topo_ponds"]
    assert "ponds/pondFreshWaterFlux" not in fixture["ridge_thorndike_exponential_cesm_ponds"]
    # an abort leaves the later cells as they were
    for name, first in (("ridge_abort_iterations", 6), ("ridge_abort_cleanup", 8)):
        start = cases[name]["pools"]["tracers"]["iceAreaCategory"]
        assert (got(name, "tracers/iceAreaCategory")[first:n_solve] == start[first:n_solve]).all()


@pytest.mark.parametrize("overrides, match", [
    ({"config_thermodynamics_type": "zero layer"}, "zerolayer_check"),
    ({"config_use_aerosols": True}, "aerosols"),
    ({"config_use_column_biogeochemistry": True, "config_use_skeletal_biochemistry": True},
     "config_use_skeletal_biochemistry"),
])
def test_unported_configurations_are_refused_by_name(overrides, match):
    case = _case("ridge_mushy_level_ponds")
    with pytest.raises(ConfigurationRefusal, match=match):
        crdg.refuse_unported({**case["config"], **overrides}, case["dims"])


def test_more_categories_than_a_thread_holds_are_refused():
    case = _case("ridge_mushy_level_ponds")
    with pytest.raises(ConfigurationRefusal, match="thickness categories"):
        crdg.refuse_unported(case["config"], {**case["dims"], "nCategories": citd.ITD_MAXCAT + 1})


@requires_gpu
def test_gpu_matches_the_v842_oracle_word_for_word(fresh_module):
    report = co.measure_ridge()
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize("original, mutated, field, case", [
    # the exponential participation divided by 1 - exp(-1/astar) instead of times its folded reciprocal
    ("G[n + 1] = M(glibc_exp(-M(G[n + 1], astari)), xtmp);",
     "G[n + 1] = D(glibc_exp(-M(G[n + 1], astari)), S(1.0, glibc_exp(-astari)));",
     "ridging/ridgeParticipationFunction", "ridge_mushy_level_ponds"),
    # Thorndike's participation at Gstar distributed
    ("apartic[n] = M(M(gstari, S(RDG_GSTAR, G[n])), S(2.0, M(gstari, A(RDG_GSTAR, G[n]))));",
     "apartic[n] = M(M(gstari, S(RDG_GSTAR, G[n])), S(2.0, A(M(gstari, RDG_GSTAR), M(gstari, G[n]))));",
     "ridging/ridgeParticipationFunction", "ridge_bl99_thorndike_hibler"),
    # the exponential ridge's thickness ratio distributed
    ("krdg[i] = D(A(hrmin[i], hrexp[i]), hi);", "krdg[i] = A(D(hrmin[i], hi), D(hrexp[i], hi));",
     "ridging/ratioRidgeThicknessToIce", "ridge_mushy_level_ponds"),
    # Hibler's thickness ratio through the reciprocal
    ("krdg[i] = D(hrmean, hi);", "krdg[i] = M(hrmean, D(1.0, hi));",
     "ridging/ratioRidgeThicknessToIce", "ridge_bl99_thorndike_hibler"),
    # the exponential volume fraction's division distributed
    ("fvol = D(S(M(A(hL, hexp), expL), M(A(hR, hexp), expR)), A(hi1, hexp));",
     "fvol = S(D(M(A(hL, hexp), expL), A(hi1, hexp)), D(M(A(hR, hexp), expR), A(hi1, hexp)));",
     "tracers/iceVolumeCategory", "ridge_mushy_level_ponds"),
    # the redistribution diagnostics "fixed" to every ridging category, not the thinnest only
    ("if (n == 1) {", "if (true) {", "ridging/fractionNewRidgeArea", "ridge_mushy_level_ponds"),
    # the tracers with ancestors carried onto the new ridges (v8.4.2 keeps only the brine fraction's)
    ("else work = 0.0;", "else work = M(work, tr[ntr]);", "tracers/pondArea", "ridge_mushy_level_ponds"),
    # the topo ponds' water lost with the new ridges' area instead of the ridging ice's
    ("*mpond = A(*mpond, M(M(ardg1n, tr[ix[TI_APND]]), tr[ix[TI_HPND]]));",
     "*mpond = A(*mpond, M(M(ardg2n, tr[ix[TI_APND]]), tr[ix[TI_HPND]]));",
     "ponds/pondFreshWaterFlux", "ridge_exponential_hibler_topo_ponds"),
    # the snow melt flux averaged over the dynamics step, not the thermodynamic one
    ("dti = D(1.0, M((double)ndtd, dt));", "dti = D(1.0, dt);",
     "ocean_fluxes/oceanFreshWaterFlux", "ridge_subcycles_topo_ponds"),
    # ridge_prep's shear share (Cs) taken as a half
    ("double closing_net = A(M(RDG_CS, rdg_shear), rdg_conv);",
     "double closing_net = A(M(0.5, rdg_shear), rdg_conv);",
     "ridging/areaLossRidge", "ridge_mushy_level_ponds"),
])
def test_a_mutated_evaluation_order_is_caught(monkeypatch, fresh_module, original, mutated, field, case):
    source = crdg.module_source
    assert source(7).count(original) == 1
    monkeypatch.setattr(crdg, "module_source", lambda n: source(n).replace(original, mutated))
    crdg._module_on.cache_clear()
    got, _ = co.run_gpu(_case(case))
    want = co.load_fixture(FIXTURE)[case][field]
    assert co.word_mismatches(got[field], want) > 0


def _oracle_toolchain() -> tuple[str, Path] | None:
    source = os.environ.get("WOOF_MPAS_V842_SOURCE")
    gfortran = os.environ.get("GFORTRAN") or shutil.which("gfortran")
    if not source or not gfortran or not Path(source, "src", "core_seaice").is_dir():
        return None
    return gfortran, Path(source)


@pytest.mark.skipif(_oracle_toolchain() is None or platform.machine() != "x86_64"
                    or platform.libc_ver() != ("glibc", "2.44"),
                    reason="the fixture is the x86-64 glibc 2.44 host's (its exp and colpkg_init_itd's tanh, "
                           "PROVENANCE.md); it needs gfortran (pixi -e mpas-oracle) and WOOF_MPAS_V842_SOURCE")
def test_the_oracle_rebuilt_here_writes_the_fixture(tmp_path):
    gfortran, source = _oracle_toolchain()
    subprocess.run(["bash", str(ROOT / "tools" / "mpas_seaice_oracle" / "build.sh"), str(source),
                    str(tmp_path / "build")], check=True, env={**os.environ, "GFORTRAN": gfortran})
    subprocess.run([sys.executable, str(ROOT / "tools" / "mpas_seaice_oracle" / "make_column_fixture.py"),
                    str(tmp_path / "build"), "--cases", "ridge", "--out", str(tmp_path / "fixture")], check=True)
    fresh = np.load(tmp_path / "fixture" / "oracle-r8.npz")
    pinned = np.load(FIXTURE / "oracle-r8.npz")
    assert sorted(fresh.files) == sorted(pinned.files)
    for key in pinned.files:
        assert co.word_mismatches(fresh[key], pinned[key]) == 0, key
    for name in ("column-logs.json", "inputs.sha256"):
        assert (tmp_path / "fixture" / name).read_text() == (FIXTURE / name).read_text()
