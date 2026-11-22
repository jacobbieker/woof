"""MPAS-Seaice's vertical salinity (zsalinity), word for word against v8.4.2 and its ``zsal_solve`` correction.

The reference is MPAS-Model v8.4.2 with ``config_use_column_biogeochemistry``,
``config_use_brine`` and ``config_use_vertical_zsalinity`` (BL99), compiled
with gfortran -O0 -ffp-contract=off at RKIND 8 on an x86-64 glibc 2.44 host,
driven by ``tools/mpas_seaice_oracle/run_column.F90`` through the init,
``column_biogeochemistry`` and ``column_vertical_thermodynamics``.

In v8.4.2 the vertical salinity is inert: ``colpkg_init_parameters`` never
sets the column package's ``solve_zsal`` (ice_colpkg.F90:4325-4345) and the
pool's ``verticalSalinity`` is copied to and from the tracer arrays only with
the vertical tracers. The default port is that, word for word
(tests/data/oracles/mpas_seaice_bgc_zsal). The opt-in correction
``zsal_solve`` solves it as the column package is written to and is held to
the oracle built with ``WOOF_ORACLE_PATCH=zsal_solve``
(``tools/mpas_seaice_oracle/zsal_solve.patch``;
tests/data/oracles/mpas_seaice_bgc_zsal_solve). Both fixtures are committed
(PROVENANCE.md), so the GPU comparison runs without gfortran. The vertical
thermodynamics' ``atan`` is glibc's (LGPL), so those cases run under the
opt-in transcription (``WOOF_GLIBC239_LIBM64=1``, source checkouts only).

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with MPAS only to a tolerance: every word of every
   field the routines may write must be bit-identical, inert or solved.
2. v8.4.2's inert vertical salinity "fixed" by default: without
   ``zsal_solve`` the step is the brine height's, the pool's vertical
   salinity keeps its words and BL99's tolerance is the brine height's.
3. A correction that drifts from the column package: under ``zsal_solve``
   every branch (first ice, a brine height above ``maxhbr``, old ice, the
   convective and the add/melt branch, the stops) must be the patched
   oracle's words.
4. An abort the port reports differently: the remap stop that
   ``compute_microS`` checks and ``bSin > max_salin`` stop at the oracle's
   cell with its message.
5. A tracer object that drifts: the zsal tracers' index, parents and
   ancestors must be the oracle's.
6. A gate that cannot fail: mutated orders, the dropped dTemp band and a
   dropped stop must produce mismatches.
7. A configuration the port does not carry running silently: the solved
   vertical salinity in a stage that does not carry it, or the vertical
   salinity without BL99, is refused by name.
8. A case generator whose bytes drift: the input digests are pinned.
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
from woof.hex.mpas_seaice import column_bgc_v842 as cbgc
from woof.hex.mpas_seaice import column_therm_v842 as ct
from woof.hex.mpas_seaice import column_v842 as cv
from woof.verify import mpas_seaice_column_oracle as co

ROOT = Path(__file__).resolve().parents[1]
INERT, SOLVE = co.BGC_ZSAL_FIXTURE_DIR, co.BGC_ZSAL_SOLVE_FIXTURE_DIR
SETS = {"inert": (co.BGC_ZSAL_CASES, INERT), "solve": (co.BGC_ZSAL_SOLVE_CASES, SOLVE)}
PUNY = 1.0e-11


def _row(name):
    return next(row for rows, _ in SETS.values() for row in rows if row[0] == name)


def _case(name, **overrides):
    name_, seed, stage, config = _row(name)
    return co.make_case(name_, seed, stage, {**config, **overrides})


def _fixture(name):
    return INERT if any(row[0] == name for row in co.BGC_ZSAL_CASES) else SOLVE


@pytest.fixture
def glibc239(monkeypatch):
    """The opt-in glibc 2.39 binary64 atan transcription (LGPL, source checkout only)."""
    from woof.core import glibc239_gpu
    if not glibc239_gpu.header_path().is_file():
        pytest.skip("the opt-in glibc 2.39 transcription is only in a source checkout")
    monkeypatch.setenv("WOOF_GLIBC239_LIBM64", "1")
    ct._module_on.cache_clear()
    yield
    ct._module_on.cache_clear()


@pytest.mark.parametrize("which", sorted(SETS))
def test_case_inputs_are_the_recorded_bytes(which):
    rows, fixture = SETS[which]
    recorded = {}
    for line in (fixture / "inputs.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        recorded[name] = digest
    assert recorded == {f"{case['name']}/r8": co.input_digest(case) for case in co.cases("r8", rows)}


def _outputs(which):
    rows, fixture = SETS[which]
    ref, logs = co.load_fixture(fixture), co.load_logs(fixture)
    cases = {c["name"]: c for c in co.cases("r8", rows)}
    return cases, ref, logs


def test_v842_s_vertical_salinity_is_inert():
    """The pool's vertical salinity, the ice salinity and the salt fluxes keep
    their words; the step is the brine height's (bioDiffusivity from
    compute_microS_mushy); the init clears the Rayleigh criterion only."""
    cases, ref, logs = _outputs("inert")
    assert set(logs) == set(cases)
    for name, case in cases.items():
        if "crash" in logs[name]:
            continue
        changed = set(ref.get(name, {}))
        keys = ["tracers/verticalSalinity", "tracers_aggregate/verticalSalinityCell",
                "biogeochemistry/totalVerticalSalinity", "biogeochemistry/zSalinityFlux",
                "biogeochemistry/zSalinityGDFlux", "biogeochemistry/bioPermeability"]
        if case["stage"] == co._BGC_STAGE:
            keys.append("tracers/iceSalinity")       # the init sets it from the profile
        for key in keys:
            assert key not in changed, (name, key)
        if case["stage"] == "init":
            assert "biogeochemistry/rayleighCriteriaReal" in changed, name
            n_solve = co.MESH["nCellsSolve"]
            assert (ref[name]["biogeochemistry/rayleighCriteriaReal"][:n_solve] == 0).all()
        if case["stage"] == co._BGC_STAGE:
            assert "biogeochemistry/bioDiffusivity" in changed, name
    # the default's remap stop is the brine height's: compute_microS_mushy's is cleared, the last category's aborts
    assert logs["bgc_zsal_abort_remap"]["crash"] == {
        "error": "column_biogeochemistry: " + cbgc.STOP_LABELS[2], "critical": "iCell: $i", "ints": [8]}


def test_zsal_solve_covers_the_branches():
    cases, ref, logs = _outputs("solve")
    assert set(logs) == set(cases)
    n_solve = co.MESH["nCellsSolve"]
    for name, out in ref.items():
        for key, value in out.items():
            assert np.isfinite(value).all(), (name, key)
    # compute_microS's remap stop is checked: the first stopping cell aborts (indexToCellID 5)
    assert logs["bgc_zsal_solve_abort_remap"]["crash"] == {
        "error": "column_biogeochemistry: " + cbgc.STOP_LABELS[2], "critical": "iCell: $i", "ints": [5]}
    # cold, salty, old ice: bSin > max_salin stops in cell 1
    assert logs["bgc_zsal_solve_abort_max_salinity"]["crash"] == {
        "error": "column_biogeochemistry: " + cbgc.STOP_LABELS[3], "critical": "iCell: $i", "ints": [1]}

    def got(name, key):
        return co.expected_outputs(cases[name], ref.get(name, {}))[key][:n_solve]

    def given(name, key):
        pool, field = key.split("/")
        return np.asarray(cases[name]["pools"][pool][field])[:n_solve]

    for name in ("bgc_zsal_solve", "bgc_zsal_solve_half_hour_sublayer_drainage_scale", "bgc_zsal_solve_short_step",
                 "bgc_zsal_solve_four_layers_three_categories"):
        iced = given(name, "tracers/iceAreaCategory")[:, :, 0] > PUNY
        zsal = got(name, "tracers/verticalSalinity")
        # the vertical salinity is solved where there is ice
        assert (zsal[iced.any(axis=1)] != given(name, "tracers/verticalSalinity")[iced.any(axis=1)]).any(), name
        # no ice before the step (cell 2): new ice, its vertical salinity zeroed
        assert (zsal[2] == 0.0).all(), name
        # and remapped back onto the ice salinity, with the fluxes and the total
        assert (got(name, "tracers/iceSalinity")[iced.any(axis=1)]
                != given(name, "tracers/iceSalinity")[iced.any(axis=1)]).any(), name
        for key in ("biogeochemistry/zSalinityFlux", "biogeochemistry/zSalinityGDFlux",
                    "biogeochemistry/totalVerticalSalinity"):
            assert (got(name, key) != given(name, key)).any(), (name, key)
        # new ice in the first category (cell 4): its vertical salinity starts at zero ice
        assert (got(name, "biogeochemistry/newlyFormedIce")[2] == 1).all(), name
    # the init writes sss*salt_loss on every owned cell, unless the vertical salinity restarts
    for name in ("init_zsal_solve", "init_zsal_solve_three_layers_four_categories"):
        sss = given(name, "ocean_coupling/seaSurfaceSalinity")
        assert (got(name, "tracers/verticalSalinity") == (sss * 0.4)[:, None, None]).all(), name
    assert (got("init_zsal_solve_restart_zsalinity", "tracers/verticalSalinity")
            == given("init_zsal_solve_restart_zsalinity", "tracers/verticalSalinity")).all()


@pytest.mark.parametrize("which", sorted(SETS))
def test_the_tracer_object_is_the_oracle_s(which):
    rows, fixture = SETS[which]
    logs = co.load_logs(fixture)
    for case in co.cases("r8", rows):
        oracle = logs[case["name"]]["tracer_object"]
        mine = cv.tracer_object(case["config"], {**case["dims"], "nSnowLayers": 1}).log_rows()
        assert "colpkg_zsal" in mine
        assert {k: oracle.get(k) for k in mine} == mine, case["name"]


def test_only_zsal_solve_maps_the_vertical_salinity():
    dims = {**_case("bgc_zsal")["dims"], "nSnowLayers": 1}
    inert = cv.tracer_object(_case("bgc_zsal")["config"], dims)
    solved = cv.tracer_object(_case("bgc_zsal_solve")["config"], dims)
    assert inert.n_tracers == solved.n_tracers == inert.nt_fbri + dims["nBioLayers"]
    assert inert.nt_bgc_s == solved.nt_bgc_s == inert.nt_fbri + 1
    assert "verticalSalinity" not in [name for name, _, _ in inert.layout]
    assert solved.layout[-1] == ("verticalSalinity", solved.nt_bgc_s, dims["nBioLayers"])
    assert not cv.solves_zsal(_case("bgc_zsal")["config"]) and cv.solves_zsal(_case("bgc_zsal_solve")["config"])


def test_every_biogeochemistry_stage_takes_the_vertical_salinity():
    dims = {**_case("bgc_zsal")["dims"], "nSnowLayers": 1}
    # v8.4.2's inert vertical salinity and the solved one: every biogeochemistry stage carries both
    cv.refuse_unported(_case("bgc_zsal")["config"], dims, biogeochemistry=True)
    cv.refuse_unported(_case("bgc_zsal_solve")["config"], dims, biogeochemistry=True)


@pytest.mark.parametrize("change", [{"config_thermodynamics_type": "mushy",
                                     "config_sea_freezing_temperature_type": "mushy"},
                                    {"config_use_brine": False}])
def test_the_vertical_salinity_without_bl99_or_the_brine_height_is_v842_s_critical(change):
    case = _case("bgc_zsal")
    with pytest.raises(ValueError, match="vertical zSalinity requires config_use_brine = true and 'BL99'"):
        cv.check_column_package_configs({**case["config"], **change}, case["dims"]["nCategories"])


@requires_gpu
@pytest.mark.parametrize("which", sorted(SETS))
def test_gpu_matches_the_oracle_word_for_word(glibc239, which):
    rows, fixture = SETS[which]
    report = co.measure_zsal(rows=rows, fixture_dir=fixture)
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


def _mutated(monkeypatch, module, original, mutated):
    source = module.module_source
    assert source(7, 7).count(original) == 1
    monkeypatch.setattr(module, "module_source", lambda *a: source(*a).replace(original, mutated))
    module._module_on.cache_clear()


@requires_gpu
@pytest.mark.parametrize("original, mutated, field, case", [
    # solve_S_dt's flux terms summed right to left
    ("Q_s[k] = A(A(A(M(V_s[k], M(pre_sin[k], pre_sin[k])), U_s[k]), C_s[k]), F_s[k]);",
     "Q_s[k] = A(M(V_s[k], M(pre_sin[k], pre_sin[k])), A(U_s[k], A(C_s[k], F_s[k])));",
     "tracers/verticalSalinity", "bgc_zsal_solve"),
    # column_sum_zsal divides by the layer count before the ice volume
    ("zsal_totn = A(zsal_totn, D(M(M(M(BG_RHOSI, zsal[k]), fbri), vicen), (double)COL_NBLYR));",
     "zsal_totn = A(zsal_totn, M(D(M(M(BG_RHOSI, zsal[k]), fbri), (double)COL_NBLYR), vicen));",
     "biogeochemistry/totalVerticalSalinity", "bgc_zsal_solve"),
])
def test_a_mutated_solve_is_caught(monkeypatch, original, mutated, field, case):
    _mutated(monkeypatch, cbgc, original, mutated)
    try:
        c = _case(case)
        got, _ = co.run_gpu(c)
        want = co.expected_outputs(c, co.load_fixture(SOLVE)[case])
        assert co.word_mismatches(got[field], want[field]) > 0
    finally:
        cbgc._module_on.cache_clear()


@requires_gpu
def test_the_aborts_stop_where_the_oracle_terminates():
    for which, name in (("inert", "bgc_zsal_abort_remap"), ("solve", "bgc_zsal_solve_abort_remap"),
                        ("solve", "bgc_zsal_solve_abort_max_salinity")):
        with pytest.raises(cbgc.CriticalAbort) as error:
            co.run_gpu(_case(name))
        assert error.value.crash == co.load_logs(SETS[which][1])[name]["crash"], name


@requires_gpu
def test_dropping_the_max_salinity_stop_is_caught(monkeypatch):
    _mutated(monkeypatch, cbgc, "l_stop = BG_MAXSALIN_STOP;", ";")
    try:
        rows = tuple(row for row in co.BGC_ZSAL_SOLVE_CASES if row[0] == "bgc_zsal_solve_abort_max_salinity")
        assert co.measure_zsal(rows=rows, fixture_dir=SOLVE) == {"bgc_zsal_solve_abort_max_salinity": {"crash": 1}}
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
                    reason="the column oracle's words are the x86-64 glibc 2.44 host's libm (the fixtures', "
                           "PROVENANCE.md); it needs gfortran (pixi -e mpas-oracle) and WOOF_MPAS_V842_SOURCE")
@pytest.mark.parametrize("which, patch", [("zsal", ""), ("zsal_solve", "zsal_solve"), ("zsal_carry", ""),
                                          ("zsal_solve_carry", "zsal_solve"), ("nonreactive", ""),
                                          ("nonreactive_carry", ""), ("nonreactive_step", ""), ("zaero", ""),
                                          ("zaero_carry", ""), ("zaero_step", ""), ("zaero_shortwave", ""),
                                          ("algae", "")])
def test_the_oracle_rebuilt_here_writes_the_fixture(tmp_path, which, patch):
    gfortran, source = _oracle_toolchain()
    fixture = {"zsal": INERT, "zsal_solve": SOLVE, "zsal_carry": co.BGC_ZSAL_CARRY_FIXTURE_DIR,
               "zsal_solve_carry": co.BGC_ZSAL_SOLVE_CARRY_FIXTURE_DIR,
               "nonreactive": co.BGC_NONREACTIVE_FIXTURE_DIR,
               "nonreactive_carry": co.BGC_NONREACTIVE_CARRY_FIXTURE_DIR,
               "nonreactive_step": co.BGC_NONREACTIVE_STEP_FIXTURE_DIR, "zaero": co.BGC_ZAERO_FIXTURE_DIR,
               "zaero_carry": co.BGC_ZAERO_CARRY_FIXTURE_DIR,
               "zaero_step": co.BGC_ZAERO_STEP_FIXTURE_DIR,
               "zaero_shortwave": co.BGC_ZAERO_SW_FIXTURE_DIR, "algae": co.BGC_ALGAE_FIXTURE_DIR}[which]
    subprocess.run(["bash", str(ROOT / "tools" / "mpas_seaice_oracle" / "build.sh"), str(source),
                    str(tmp_path / "build")], check=True,
                   env={**os.environ, "GFORTRAN": gfortran, "WOOF_ORACLE_PATCH": patch})
    subprocess.run([sys.executable, str(ROOT / "tools" / "mpas_seaice_oracle" / "make_column_fixture.py"),
                    str(tmp_path / "build"), "--cases", which, "--out", str(tmp_path / "fixture")], check=True)
    fresh = np.load(tmp_path / "fixture" / "oracle-r8.npz")
    pinned = np.load(fixture / "oracle-r8.npz")
    assert sorted(fresh.files) == sorted(pinned.files)
    for key in pinned.files:
        assert co.word_mismatches(fresh[key], pinned[key]) == 0, key
    for name in ("column-logs.json", "inputs.sha256"):
        assert (tmp_path / "fixture" / name).read_text() == (fixture / name).read_text()


@requires_gpu
@pytest.mark.parametrize("case", ["vt_zsal_solve_bl99", "vt_zsal_solve_bl99_melting"])
def test_dropping_the_dtemp_band_is_caught(glibc239, monkeypatch, case):
    """BL99's thickness_changes tolerance is 0.1 under the solved vertical
    salinity (ice_therm_bl99.F90:287): a port that keeps the brine height's
    0.01 there differs from the patched oracle."""
    monkeypatch.setattr(ct.cv, "solves_zsal", lambda configs: False)
    ct._module_on.cache_clear()
    c = _case(case)
    got, _ = co.run_gpu(c)
    want = co.expected_outputs(c, co.load_fixture(SOLVE)[case])
    assert sum(co.word_mismatches(got[k], want[k]) for k in want) > 0
