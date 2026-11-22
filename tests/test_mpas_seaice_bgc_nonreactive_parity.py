"""MPAS-Seaice's vertical tracers with the nonreactive tracer, word for word against v8.4.2.

The reference is MPAS-Model v8.4.2 with ``config_use_column_biogeochemistry``,
``config_use_brine``, ``config_use_vertical_tracers`` and
``config_use_nonreactive``, compiled with gfortran -O0 -ffp-contract=off at
RKIND 8 on an x86-64 glibc 2.44 host, driven by
``tools/mpas_seaice_oracle/run_column.F90`` through the init and
``column_biogeochemistry``. Its outputs are committed
(tests/data/oracles/mpas_seaice_bgc_nonreactive, PROVENANCE.md), so the GPU
comparison runs without gfortran.

The column package's ``z_tracers`` with one tracer (PON):
- the init: ``colpkg_init_zbgc``'s tracer layout, retention and release times,
  mobile fraction and ocean concentration, and ``colpkg_init_bgc``'s profiles;
- ``column_biogeochemistry``'s ``zbio``: ``update_snow_bgc``'s scavenging,
  ``z_biogeochemistry``'s mobile and stationary phases (the FCT solve, its
  correction and ``regrid_stationary``, or ``thin_ice_flux``) and
  ``merge_bgc_fluxes``; the shortwave on the bio grid summed into
  ``bioShortwaveFluxCell``.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with MPAS only to a tolerance: every word of every
   field the stages may write must be bit-identical.
2. A tracer object that drifts: the vertical tracers' index, mobile fraction
   index and ``bio_index``/``bio_index_o`` must be the oracle's.
3. An abort the port reports differently: ``zbgc initialization error`` and
   ``negative bgc`` stop at the oracle's cell with its message.
4. A gate that cannot fail: each mutated branch must produce mismatches.
5. A configuration the port does not carry running silently: the vertical
   tracers in a stage that does not carry them, without the brine height or
   the nonreactive tracer, with the vertical salinity, or with the mushy
   salinity-scaled profiles, are refused by name.
6. A case generator whose bytes drift: the input digests are pinned.
"""
from __future__ import annotations

import pytest

from conftest import requires_gpu
from woof.hex.errors import ConfigurationRefusal
from woof.hex.mpas_seaice import column_bgc_v842 as cbgc
from woof.hex.mpas_seaice import column_v842 as cv
from woof.verify import mpas_seaice_column_oracle as co

FIXTURE = co.BGC_NONREACTIVE_FIXTURE_DIR
ROWS = co.BGC_NONREACTIVE_CASES
CONC = "tracers/verticalNonreactiveConc"
FRAC = "tracers/mobileFraction"
ABORTS = {"bgc_nonreactive_abort_init": "column_biogeochemistry: zbgc initialization error",
          "bgc_nonreactive_abort_negative": "column_biogeochemistry: negative bgc"}
_BGC_NAMES = tuple(row[0] for row in ROWS if row[2] == "biogeochemistry" and row[0] not in ABORTS)


def _row(name):
    return next(row for row in ROWS if row[0] == name)


def _case(name, **overrides):
    name_, seed, stage, config = _row(name)
    return co.make_case(name_, seed, stage, {**config, **overrides})


def _dims(case, **overrides):
    return {**case["dims"], "nSnowLayers": 1, **overrides}


def test_case_inputs_are_the_recorded_bytes():
    recorded = {}
    for line in (FIXTURE / "inputs.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        recorded[name] = digest
    assert recorded == {f"{case['name']}/r8": co.input_digest(case) for case in co.cases("r8", ROWS)}


def test_fixture_shows_the_nonreactive_tracer():
    fixture, logs = co.load_fixture(FIXTURE), co.load_logs(FIXTURE)
    names = {row[0] for row in ROWS}
    assert set(logs) == names
    for name in names - set(ABORTS):
        assert logs[name].get("crash") is None, name
        assert not [line for line in logs[name]["log"] if line.startswith("ERR")], name
    for name in _BGC_NAMES:
        for field in (CONC, FRAC, "biogeochemistry/oceanBioFluxes", "biogeochemistry/totalVerticalBiologyIce",
                      "biogeochemistry/bioShortwaveFluxCell"):
            assert field in fixture[name], (name, field)
    # the init sets the profiles and the mobile fraction, except on a bgc restart
    for name in ("init_bgc_nonreactive", "init_bgc_nonreactive_bl99_mobile_frazil",
                 "init_bgc_nonreactive_three_layers_four_categories"):
        assert CONC in fixture[name] and FRAC in fixture[name], name
    assert CONC not in fixture["init_bgc_nonreactive_restart_bgc"]
    assert FRAC not in fixture["init_bgc_nonreactive_restart_bgc"]


def test_the_tracer_object_is_the_oracle_s():
    logs = co.load_logs(FIXTURE)
    for case in co.cases("r8", ROWS):
        oracle = logs[case["name"]]["tracer_object"]
        mine = cv.tracer_object(case["config"], _dims(case)).log_rows()
        assert "colpkg_zbgc" in mine
        assert {k: oracle.get(k) for k in mine} == mine, case["name"]


def test_the_init_and_the_biogeochemistry_carry_the_vertical_tracers():
    for name in ("init_bgc_nonreactive", "bgc_nonreactive"):
        case = _case(name)
        cv.refuse_unported(case["config"], _dims(case), biogeochemistry=True, z_tracers=True)
        cbgc.refuse_unported(case["config"], _dims(case))


def test_other_stages_refuse_the_vertical_tracers():
    case = _case("bgc_nonreactive")
    with pytest.raises(ConfigurationRefusal, match=cv.VT_FLAG):
        cv.refuse_unported(case["config"], _dims(case), biogeochemistry=True)


@pytest.mark.parametrize("change, knob", [
    ({"config_use_brine": False}, "config_use_brine"),
    ({"config_use_nonreactive": False}, "config_use_nonreactive"),
    ({"config_use_vertical_zsalinity": True, **co._BL99}, cv.ZSAL_FLAG),
    ({"config_scale_initial_vertical_bgc": True}, "config_scale_initial_vertical_bgc"),
    ({"config_use_nitrate": True}, "config_use_nitrate"),
])
def test_what_the_vertical_tracers_port_does_not_carry_is_refused(change, knob):
    case = _case("bgc_nonreactive")
    config = {**case["config"], **change}
    if knob == "config_scale_initial_vertical_bgc":
        if config["config_thermodynamics_type"] != "mushy":
            pytest.skip("the cases' thermodynamics are not mushy")
    with pytest.raises(ConfigurationRefusal, match=knob):
        cv.refuse_unported(config, _dims(case), biogeochemistry=True, z_tracers=True)


@requires_gpu
def test_gpu_matches_the_oracle_word_for_word():
    report = co.measure_bgc(rows=ROWS, fixture_dir=FIXTURE)
    assert set(report) == {row[0] for row in ROWS}
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
def test_the_aborts_stop_where_the_oracle_terminates():
    logs = co.load_logs(FIXTURE)
    for name, message in ABORTS.items():
        assert logs[name]["crash"]["error"] == message
        with pytest.raises(cbgc.CriticalAbort) as error:
            co.run_gpu(_case(name))
        assert error.value.crash == logs[name]["crash"], name


def _mismatches(names):
    report = co.measure_bgc(rows=tuple(_row(name) for name in names), fixture_dir=FIXTURE)
    return sum(sum(row.values()) for row in report.values())


@requires_gpu
@pytest.mark.parametrize("names, original, mutated", [
    # update_snow_bgc's scavenging from the surface snow layer
    (_BGC_NAMES, "if (dzssl > PUNY) sloss1 = D(M(M(ZB_KSCAVZ, a1), col_min(-dhs_melts, dzssl)), dzssl);",
     "if (dzssl > PUNY) sloss1 = D(M(ZB_KSCAVZ, col_min(-dhs_melts, dzssl)), dzssl);"),
    # the mobile phase retained at the release time
    (_BGC_NAMES, "M(icm[k], S(glibc_exp(-M(dt, rtau_ret)), 1.0))", "M(icm[k], S(glibc_exp(-M(dt, rtau_rel)), 1.0))"),
    # the FCT flux without the atmospheric source
    (_BGC_NAMES, "*flux_bio = D(A(S(c_init, c_new), source), dt);", "*flux_bio = D(S(c_init, c_new), dt);"),
    # the mobile fraction not updated from the solve
    (_BGC_NAMES, "if (sum_tot > 0.0 && mobile > 0.0) *frac = D(sum_new, sum_tot);", ""),
    # merge_bgc_fluxes' snow layers swapped
    (_BGC_NAMES, "*snow_bio_net = A(A(*snow_bio_net, M(trc[NB1], dvssl)), M(trc[NB1 + 1], dvint));",
     "*snow_bio_net = A(A(*snow_bio_net, M(trc[NB1], dvint)), M(trc[NB1 + 1], dvssl));"),
    # the cell's bio-grid shortwave not weighted by the category area
    (_BGC_NAMES, "M(zfswin[((size_t)iCell * ncat + n) * NB1 + k],\n"
                 "                                                           aicen[iCell * ncat + n]));",
     "zfswin[((size_t)iCell * ncat + n) * NB1 + k]);"),
    # the new ice's mobile fraction not reset
    (_BGC_NAMES, "if (z_tracers) ztr_frac[(size_t)c * nzbgc] = zpar[ZP_FRAC_INIT];", ""),
    # the stops dropped
    (("bgc_nonreactive_abort_init",), "if (*t < 0.0) return BG_ZBGC_INIT;", ""),
    (("bgc_nonreactive_abort_negative",), "else if (bio_tmp < 0.0) return BG_NEGATIVE_BGC;", ""),
])
def test_a_mutated_vertical_tracer_is_caught(monkeypatch, names, original, mutated):
    source = cbgc.module_source
    assert source(7, 7).count(original) == 1
    monkeypatch.setattr(cbgc, "module_source", lambda *a: source(*a).replace(original, mutated))
    cbgc._module_on.cache_clear()
    try:
        assert _mismatches(names) > 0
    finally:
        cbgc._module_on.cache_clear()
