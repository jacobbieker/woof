"""MPAS-Seaice's column biogeochemistry framework and brine height, word for word against v8.4.2.

The reference is MPAS-Model v8.4.2 with ``config_use_column_biogeochemistry``
and ``config_use_brine`` (every other biogeochemistry flag off): the init's
biogeochemistry (the tracer object's brine fraction, ``colpkg_init_hbrine``,
``colpkg_init_zbgc``, ``init_column_biogeochemistry_profiles``, the coupler
fields' ocean concentrations and the ocean flux resets) and
``column_biogeochemistry`` (``colpkg_biogeochemistry``'s brine path:
``preflushing_changes``, ``compute_microS_mushy``, ``update_hbrine``),
compiled with gfortran -O0 -ffp-contract=off at RKIND 8 on an x86-64 glibc
2.44 host. ``tools/mpas_seaice_oracle/run_column.F90`` runs the stages
``init`` and ``biogeochemistry``; the outputs are committed
(tests/data/oracles/mpas_seaice_bgc_brine, PROVENANCE.md), so the GPU
comparison runs without gfortran. ``exp`` is the shared glibc transcription
(MIT), so no LGPL opt-in is needed.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with MPAS only to a tolerance: every word of every
   field the routines may write must be bit-identical.
2. v8.4.2's quirks "fixed": the gravity drainage compares the brine height
   with the critical Rayleigh number, and colpkg_init_zbgc gives the diatoms
   the proteins' carbon to nitrogen ratio (kept; the opt-in correction
   ``zbgc_c2n_proteins`` is tested on its own).
3. An abort the port reports differently: v8.4.2 does not check
   ``compute_microS_mushy``'s stop, so the next ice category clears it, and
   a stop that reaches the wrapper is logged at ``MPAS_LOG_CRIT``, which
   terminates MPAS: the port must run through the first and stop at the
   same cell with the same message on the second.
4. A tracer object that drifts: the brine fraction's index, parent and
   ancestors must be the oracle's.
5. A gate that cannot fail: a mutated evaluation order must produce
   mismatches.
6. A configuration the port does not carry running silently: each other
   biogeochemistry flag, the biogeochemistry in a stage that does not carry
   it, too many biology layers and the stage without the brine height are
   refused by name.
7. A case generator whose bytes drift: the input digests are pinned.
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
from woof.hex.mpas_seaice import column_bgc_v842 as cbgc
from woof.hex.mpas_seaice import column_v842 as cv
from woof.verify import mpas_seaice_column_oracle as co

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = co.BGC_FIXTURE_DIR
PUNY = 1.0e-11


def _case(name, **overrides):
    name_, seed, stage, config = next(row for row in co.BGC_CASES if row[0] == name)
    return co.make_case(name_, seed, stage, {**config, **overrides})


def _dims(case, **overrides):
    return {**case["dims"], **overrides}


def test_case_inputs_are_the_recorded_bytes():
    recorded = {}
    for line in (FIXTURE / "inputs.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        recorded[name] = digest
    assert recorded == {f"{case['name']}/r8": co.input_digest(case) for case in co.cases("r8", co.BGC_CASES)}


def test_fixture_covers_the_branches():
    fixture, logs = co.load_fixture(FIXTURE), co.load_logs(FIXTURE)
    cases = {c["name"]: c for c in co.cases("r8", co.BGC_CASES)}
    assert set(logs) == set(cases)
    for name, out in fixture.items():
        for key, value in out.items():
            assert np.isfinite(value).all(), (name, key)
    # v8.4.2 dies in one case at its eighth cell (indexToCellID 8, before a later aborting
    # cell); the rest run through, the cleared remap stop too
    assert logs["bgc_brine_abort_remap"]["crash"] == {
        "error": "column_biogeochemistry: " + cbgc.STOP_LABELS[2], "critical": "iCell: $i", "ints": [8]}
    assert "bgc_brine_abort_remap" not in fixture
    stage = {name: co.oracle_abort(logs[name], "column_biogeochemistry")
             for name, case in cases.items() if case["stage"] == "biogeochemistry" and "abort" not in name}
    assert set(stage.values()) == {(0, None, None)}
    n_solve = co.MESH["nCellsSolve"]

    def got(name, key):
        return co.expected_outputs(cases[name], fixture.get(name, {}))[key][:n_solve]

    def given(name, key):
        pool, field = key.split("/")
        return np.asarray(cases[name]["pools"][pool][field])[:n_solve]

    for name in ("bgc_brine", "bgc_brine_bl99_half_hour", "bgc_brine_short_drainage_length",
                 "bgc_brine_four_layers_three_categories", "bgc_brine_remap_stop_cleared"):
        iced = given(name, "tracers/iceAreaCategory")[:, :, 0] > PUNY
        darcy = got(name, "biogeochemistry/darcyVelocityBio")
        fbri = got(name, "tracers/brineFraction")[:, :, 0]
        # brine above sea level runs off (cell 0), below it under heavy snow floods (cell 1)
        assert (darcy[0][iced[0]] < 0).all() and (darcy[1][iced[1]] > 0).any(), name
        # no ice: new ice, no thickness before the step (cell 2)
        assert (got(name, "biogeochemistry/newlyFormedIce")[2] == 1).all(), name
        assert (got(name, "icestate/iceThicknessCategoryInitial")[2] == 0).all(), name
        # below thinS the brine height follows the ice (cell 3)
        assert (fbri[3] != given(name, "tracers/brineFraction")[3, :, 0]).all(), name
        # a non-positive brine fraction restarts at 1 and stays positive (cell 5)
        assert (fbri[5][given(name, "tracers/brineFraction")[5, :, 0] <= 0] > 0).all(), name
        # new ice in the first category clears the Rayleigh criterion (cell 4); another cell keeps it
        rayleigh = got(name, "biogeochemistry/rayleighCriteriaReal")
        assert rayleigh[4] == 0 and (rayleigh > 0.5).any(), name
        hbri = got(name, "biogeochemistry/netBrineHeight") != given(name, "biogeochemistry/netBrineHeight")
        assert (hbri == iced.any(axis=1)).all(), name
    # the cleared remap stop: the stopped category keeps its profiles, its brine height still moves
    name = "bgc_brine_remap_stop_cleared"
    for key in ("biogeochemistry/bioTemperature", "biogeochemistry/bioPorosity", "biogeochemistry/zSalinityIceDensity"):
        assert (got(name, key)[4, 2] == given(name, key)[4, 2]).all(), key
        assert (got(name, key)[4, 3] != given(name, key)[4, 3]).any(), key
    assert got(name, "biogeochemistry/darcyVelocityBio")[4, 2] != given(name, "biogeochemistry/darcyVelocityBio")[4, 2]
    # the init: brine fraction 1 and new ice everywhere, unless the brine height restarts
    for name in ("init_bgc_brine", "init_bgc_brine_bl99_c2n", "init_bgc_brine_three_layers_cesm_ponds"):
        assert (got(name, "tracers/brineFraction") == 1.0).all(), name
        assert (got(name, "biogeochemistry/newlyFormedIce") == 1).all(), name
    assert (got("init_bgc_brine_restart_hbrine", "biogeochemistry/newlyFormedIce")
            == given("init_bgc_brine_restart_hbrine", "biogeochemistry/newlyFormedIce")).all()
    # colpkg_init_zbgc's R_C2N(1) is the proteins' ratio, R_C2N_DON its static zero
    c2n = co.expected_outputs(cases["init_bgc_brine_bl99_c2n"], fixture.get("init_bgc_brine_bl99_c2n", {}))
    assert c2n["biogeochemistry/carbonToNitrogenRatioAlgae"][:3].tolist() == [5.5, 7.25, 7.75]
    assert c2n["biogeochemistry/carbonToNitrogenRatioDON"][0] == 0.0


def test_the_tracer_object_is_the_oracle_s():
    logs = co.load_logs(FIXTURE)
    for case in co.cases("r8", co.BGC_CASES):
        oracle = logs[case["name"]]["tracer_object"]
        mine = cv.tracer_object(case["config"], {**case["dims"], "nSnowLayers": 1}).log_rows()
        assert {k: oracle.get(k) for k in mine} == mine, case["name"]


def test_the_brine_fraction_is_the_last_tracer_with_the_ice_volume_parent():
    case = _case("bgc_brine")
    dims = {**case["dims"], "nSnowLayers": 1}
    physics = cv.tracer_object({**case["config"], "config_use_brine": False}, dims)
    brine = cv.tracer_object(case["config"], dims)
    assert brine.nt_fbri == brine.n_tracers == physics.n_tracers + 1 and physics.nt_fbri == 0
    assert brine.parent_index[brine.nt_fbri - 1] == 1
    assert [row[brine.nt_fbri - 1] for row in brine.first_ancestor_mask] == [0, 1, 0]
    assert brine.ancestor_number[brine.nt_fbri - 1] == 0
    assert brine.layout[-1] == ("brineFraction", brine.nt_fbri, 1)


@pytest.mark.parametrize("flag", [f for f in cv._BGC_FLAGS if f not in cv.BGC_PORTED and f != cv.ZSAL_FLAG])
def test_every_other_biogeochemistry_flag_is_refused_by_name(flag):
    case = _case("bgc_brine")
    with pytest.raises(ConfigurationRefusal, match=flag):
        cbgc.refuse_unported({**case["config"], flag: True}, _dims(case, nSnowLayers=1))


def test_stages_without_the_biogeochemistry_refuse_it():
    case = _case("bgc_brine")
    with pytest.raises(ConfigurationRefusal, match="config_use_column_biogeochemistry"):
        cv.refuse_unported(case["config"], _dims(case, nSnowLayers=1))
    cv.refuse_unported(case["config"], _dims(case, nSnowLayers=1), biogeochemistry=True)


@pytest.mark.parametrize("dims", [{"nBioLayers": 0}, {"nBioLayers": cv.MAX_BIO_LAYERS + 1},
                                  {"nBioLayers": cv.MAX_BIO_LAYERS, "nIceLayers": 1}])
def test_biology_layers_past_compute_microS_mushy_s_buffer_are_refused(dims):
    case = _case("bgc_brine")
    with pytest.raises(ConfigurationRefusal, match="nBioLayers"):
        cbgc.refuse_unported(case["config"], _dims(case, nSnowLayers=1, **dims))


def test_the_bio_layer_bound_is_v842_s_own_tracer_count():
    """trtmp is ntrcr + 2 long with ntrcr the configured count, the brine
    fraction included (ice_brine.F90:234): the last layer count that fits is
    accepted, one more is refused."""
    case = _case("bgc_brine")
    dims = _dims(case, nSnowLayers=1, nIceLayers=1)
    to = cv.tracer_object(case["config"], dims)
    fits = to.n_tracers + 2 - to.index["iceSalinity"] + 1
    assert 1 <= fits <= cv.MAX_BIO_LAYERS
    cbgc.refuse_unported(case["config"], {**dims, "nBioLayers": fits})
    with pytest.raises(ConfigurationRefusal, match="nBioLayers"):
        cbgc.refuse_unported(case["config"], {**dims, "nBioLayers": fits + 1})


@requires_gpu
def test_a_missing_brine_fraction_array_is_an_init_error():
    import types

    import cupy as cp
    case = _case("bgc_brine")
    dims = _dims(case, nSnowLayers=1)
    grids = cv.hbrine_grids(dims["nBioLayers"], dims["nIceLayers"])
    pools = {"biogeochemistry": {**{k: cp.zeros(v.shape) for k, v in grids.items()},
                                 "newlyFormedIce": cp.zeros((dims["nCells"] + 1, dims["nCategories"]), cp.int32)},
             "tracers": {}}
    state = types.SimpleNamespace(dims=dims)
    with pytest.raises(ValueError, match="brineFraction"):
        cv.init_biogeochemistry_profiles(pools, case["config"], state)


def test_the_stage_without_the_brine_height_and_too_many_categories_are_refused():
    case = _case("bgc_brine")
    with pytest.raises(ConfigurationRefusal, match="config_use_brine"):
        cbgc.refuse_unported({**case["config"], "config_use_brine": False}, _dims(case, nSnowLayers=1))
    with pytest.raises(ConfigurationRefusal, match="config_nCategories"):
        cbgc.refuse_unported(case["config"], _dims(case, nSnowLayers=1, nCategories=cbgc.MAX_CATEGORIES + 1))


@requires_gpu
def test_gpu_matches_the_v842_oracle_word_for_word():
    report = co.measure_bgc()
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
def test_the_c2n_correction_gives_each_ratio_its_own_value():
    """``zbgc_c2n_proteins``: the diatoms take config_ratio_C_to_N_diatoms and
    the dissolved organic nitrogen config_ratio_C_to_N_proteins, where v8.4.2
    gives the diatoms the proteins' ratio and DON zero."""
    got, _ = co.run_gpu(_case("init_bgc_brine_bl99_c2n", woof_v842_corrections=("zbgc_c2n_proteins",)))
    assert got["biogeochemistry/carbonToNitrogenRatioAlgae"][:3].tolist() == [6.5, 7.25, 7.75]
    assert got["biogeochemistry/carbonToNitrogenRatioDON"][0] == 5.5


@requires_gpu
@pytest.mark.parametrize("original, mutated, field, case", [
    # the runoff's Darcy velocity: the porosity applied before the time step
    ("*darcy_V = -bg_sign(M(D(S(hbrocn, hbrocn_new), dt), *bphi_min), hbrocn);",
     "*darcy_V = -bg_sign(D(M(S(hbrocn, hbrocn_new), *bphi_min), dt), hbrocn);",
     "biogeochemistry/darcyVelocityBio", "bgc_brine"),
    # the gravity drainage's length scale multiplied last
    ("iDin[k] = A(iDin[k], D(M(D(M(M(l_sk, ikin), BG_GRAVIT), BG_VISCOS_DYNAMIC), drho[k]), hb2));",
     "iDin[k] = A(iDin[k], D(M(M(l_sk, D(M(ikin, BG_GRAVIT), BG_VISCOS_DYNAMIC)), drho[k]), hb2));",
     "biogeochemistry/bioDiffusivity", "bgc_brine"),
    # the brine height against the critical Rayleigh number "fixed" away
    ("if (*hbr_old >= BG_RA_C)", "if (*hbr_old >= 1.0e30)",
     "biogeochemistry/bioDiffusivity", "bgc_brine"),
    # temperature_mush's quadratic coefficient summed right to left
    ("M(A(A(M(BS2, zSin), M(BQ2, zqin)), BC2), t_low)",
     "M(A(M(BS2, zSin), A(M(BQ2, zqin), BC2)), t_low)",
     "biogeochemistry/bioTemperature", "bgc_brine_bl99_half_hour"),
])
def test_a_mutated_evaluation_order_is_caught(monkeypatch, original, mutated, field, case):
    source = cbgc.module_source
    assert source(7, 7).count(original) == 1
    monkeypatch.setattr(cbgc, "module_source", lambda *a: source(*a).replace(original, mutated))
    cbgc._module_on.cache_clear()
    try:
        c = _case(case)
        got, _ = co.run_gpu(c)
        want = co.expected_outputs(c, co.load_fixture(FIXTURE)[case])
        assert co.word_mismatches(got[field], want[field]) > 0
    finally:
        cbgc._module_on.cache_clear()


@requires_gpu
def test_the_abort_stops_where_v842_terminates():
    with pytest.raises(cbgc.CriticalAbort) as error:
        co.run_gpu(_case("bgc_brine_abort_remap"))
    assert error.value.crash == co.load_logs(FIXTURE)["bgc_brine_abort_remap"]["crash"]
    assert (error.value.ierr, error.value.cell) == (cbgc.SEAICE_ERROR_COL_BGC, 8)


@requires_gpu
def test_checking_compute_microS_mushy_s_stop_is_caught(monkeypatch):
    """A port that stops where compute_microS_mushy does (v8.4.2 does not
    check it) dies in the case v8.4.2 runs through."""
    original = "                if (!l_stop) {\n"
    mutated = "                if (l_stop) { status[iCell] = l_stop; return; }\n                {\n"
    source = cbgc.module_source
    assert source(7, 7).count(original) == 1
    monkeypatch.setattr(cbgc, "module_source", lambda *a: source(*a).replace(original, mutated))
    cbgc._module_on.cache_clear()
    try:
        with pytest.raises(cbgc.CriticalAbort) as error:
            co.run_gpu(_case("bgc_brine_remap_stop_cleared"))
        assert error.value.cell == 5
    finally:
        cbgc._module_on.cache_clear()


def _oracle_toolchain() -> tuple[str, Path] | None:
    source = os.environ.get("WOOF_MPAS_V842_SOURCE")
    gfortran = os.environ.get("GFORTRAN") or shutil.which("gfortran")
    if not source or not gfortran or not Path(source, "src", "core_seaice").is_dir():
        return None
    return gfortran, Path(source)


@pytest.mark.skipif(_oracle_toolchain() is None or platform.machine() != "x86_64"
                    or platform.libc_ver() != ("glibc", "2.44"),
                    reason="the column oracle's words are the x86-64 glibc 2.44 host's libm (exp; "
                           "the fixture's, PROVENANCE.md); "
                           "it needs gfortran (pixi -e mpas-oracle) and WOOF_MPAS_V842_SOURCE")
def test_the_oracle_rebuilt_here_writes_the_fixture(tmp_path):
    gfortran, source = _oracle_toolchain()
    subprocess.run(["bash", str(ROOT / "tools" / "mpas_seaice_oracle" / "build.sh"), str(source),
                    str(tmp_path / "build")], check=True, env={**os.environ, "GFORTRAN": gfortran})
    subprocess.run([sys.executable, str(ROOT / "tools" / "mpas_seaice_oracle" / "make_column_fixture.py"),
                    str(tmp_path / "build"), "--cases", "bgc", "--out", str(tmp_path / "fixture")], check=True)
    fresh = np.load(tmp_path / "fixture" / "oracle-r8.npz")
    pinned = np.load(FIXTURE / "oracle-r8.npz")
    assert sorted(fresh.files) == sorted(pinned.files)
    for key in pinned.files:
        assert co.word_mismatches(fresh[key], pinned[key]) == 0, key
    for name in ("column-logs.json", "inputs.sha256"):
        assert (tmp_path / "fixture" / name).read_text() == (FIXTURE / name).read_text()
