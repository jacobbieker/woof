"""MPAS-Seaice's vertical biochemistry (algae and nutrients), word for word against v8.4.2.

The reference is MPAS-Model v8.4.2 with ``config_use_column_biogeochemistry``,
``config_use_brine``, ``config_use_vertical_tracers``,
``config_use_vertical_biochemistry`` and ``config_use_nitrate`` (with and
without ``config_use_ammonium`` and ``config_use_silicate``, the
z-aerosols and the nonreactive tracer), compiled with gfortran -O0
-ffp-contract=off at RKIND 8 on an x86-64 glibc 2.44 host, driven by
``tools/mpas_seaice_oracle/run_column.F90`` through the init and
``column_biogeochemistry``. Its outputs are committed
(tests/data/oracles/mpas_seaice_bgc_algae, PROVENANCE.md), so the GPU
comparison runs without gfortran.

The column package's ``z_tracers`` with ``solve_zbgc``: the Registry's three
algae (diatoms, small plankton, phaeocystis), then nitrate, ammonium and
silicate, before the nonreactive tracer and the z-aerosols:
- the init: ``colpkg_init_zbgc``'s tracer layout (``bio_index_o`` 1-4, 12
  and 13), each tracer's transport type, and ``colpkg_init_bgc``'s profiles;
- ``column_biogeochemistry``: ``z_biogeochemistry`` with the first alga's
  own release and retention (``algal_vel``) and its stationary part
  regridded only on fast bottom melt or an area below 0.001, ``algal_dyn``
  per biology layer (light, nutrient and silicate limitation, uptake capped
  at ``max_loss``, respiration, grazing, mortality, nitrification, the
  nitrogen lost to the zooplankton in ``verticalNitrogenLosses``) at the
  interface temperature ``iTin`` colpkg_biogeochemistry keeps across
  categories, and ``merge_bgc_fluxes``' uptake, growth, primary production
  and chlorophyll sums.

v8.4.2 kept word for word: ``totalChlorophyll`` sums the cell's running
``totalVerticalBiologyIce`` once per category, so each category's
chlorophyll is counted again with every later one; ``netNitrateUptake`` and
``netAmmoniumUptake`` are never reset; the diatoms' carbon to nitrogen ratio
in ``primaryProduction`` is the proteins' unless the opt-in correction
``zbgc_c2n_proteins``. The opt-in correction ``total_chlorophyll`` counts
each category once. Both are tested here for what they change, not against
the oracle.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with MPAS only to a tolerance: every word of every
   field the stages may write must be bit-identical.
2. A tracer object that drifts: the algae's and nutrients' indices must be
   the oracle's.
3. A gate that cannot fail: each mutated branch must produce mismatches.
4. A configuration the port does not carry running silently: the vertical
   biochemistry in a stage that does not carry it is refused by name.
5. A case generator whose bytes drift: the input digests are pinned.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu
from woof.hex.errors import ConfigurationRefusal
from woof.hex.mpas_seaice import column_bgc_v842 as cbgc
from woof.hex.mpas_seaice import column_itd_v842 as citd
from woof.hex.mpas_seaice import column_radiation_v842 as crad
from woof.hex.mpas_seaice import column_therm_v842 as ctherm
from woof.hex.mpas_seaice import column_v842 as cv
from woof.verify import mpas_seaice_column_oracle as co

FIXTURE = co.BGC_ALGAE_FIXTURE_DIR
ROWS = co.BGC_ALGAE_CASES
ALGAE = "tracers/verticalAlgaeConc"
NITRATE = "tracers/verticalNitrateConc"
BGC = "biogeochemistry/"
SUMS = tuple(BGC + name for name in ("netNitrateUptake", "netSpecificAlgalGrowthRate", "primaryProduction",
                                     "totalChlorophyll", "verticalNitrogenLosses"))
_BGC_NAMES = tuple(row[0] for row in ROWS if row[2] == "biogeochemistry" and "without_algae" not in row[0])
WITHOUT = "bgc_nutrients_nonreactive_without_algae"
_INIT_NAMES = tuple(row[0] for row in ROWS if row[2] == "init")
P3 = 10                                 # nBioLayersP3


def _row(name):
    return next(row for row in ROWS if row[0] == name)


def _case(name, *corrections):
    case = co.make_case(*_row(name))
    if corrections:
        case["config"]["woof_v842_corrections"] = (*cv.corrections(case["config"]), *corrections)
    return case


def _dims(case):
    return {**case["dims"], "nSnowLayers": 1}


def test_case_inputs_are_the_recorded_bytes():
    recorded = {}
    for line in (FIXTURE / "inputs.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        recorded[name] = digest
    assert recorded == {f"{case['name']}/r8": co.input_digest(case) for case in co.cases("r8", ROWS)}


def test_the_tracer_object_is_the_oracle_s():
    logs = co.load_logs(FIXTURE)
    for case in co.cases("r8", ROWS):
        oracle = logs[case["name"]]["tracer_object"]
        mine = cv.tracer_object(case["config"], _dims(case)).log_rows()
        assert {k: oracle.get(k) for k in mine} == mine, case["name"]
        # the algae's, nitrate's, ammonium's and silicate's places in the ocean concentrations
        c = case["config"]
        places = ((1, 2, 3) if c["config_use_vertical_biochemistry"] else ()) + (4,) \
            + ((12,) if c["config_use_ammonium"] else ()) + ((13,) if c["config_use_silicate"] else ())
        assert cv.tracer_object(case["config"], _dims(case)).bio_index_o[:len(places)] == places, case["name"]


def test_fixture_shows_the_algae_growing():
    fixture, logs = co.load_fixture(FIXTURE), co.load_logs(FIXTURE)
    assert set(logs) == {row[0] for row in ROWS}
    for name in logs:
        assert logs[name].get("crash") is None, name
        assert not [line for line in logs[name]["log"] if line.startswith("ERR")], name
    for name in _INIT_NAMES:
        assert ALGAE in fixture[name] and NITRATE in fixture[name], name
    for name in _BGC_NAMES:
        case = _case(name)
        n = case["dims"]["nCellsSolve"]
        for field in (ALGAE, NITRATE) + SUMS:
            assert field in fixture[name], (name, field)
        # every alga's ice layers move
        for j in range(3):
            assert not np.array_equal(fixture[name][ALGAE][:n, :, j * P3:j * P3 + 8],
                                      case["pools"]["tracers"]["verticalAlgaeConc"][:n, :, j * P3:j * P3 + 8])
        # primary production is summed onto its input, positive growth
        before = case["pools"]["biogeochemistry"]["primaryProduction"][:n]
        assert (fixture[name][BGC + "primaryProduction"][:n] >= before).all(), name
    # the nutrients without the algae are carried, with no reactions: nothing sums
    assert NITRATE in fixture[WITHOUT] and not set(SUMS) & set(fixture[WITHOUT])


def test_the_init_and_the_biogeochemistry_carry_the_algae():
    for name in ("init_bgc_algae", "bgc_algae_nutrients_zaero_nonreactive_four_layers_three_categories"):
        case = _case(name)
        cv.refuse_unported(case["config"], _dims(case), biogeochemistry=True, z_tracers=cv.VT_INIT_PORTED)
        cbgc.refuse_unported(case["config"], _dims(case))


@pytest.mark.parametrize("refuse", [
    lambda c, d: citd.refuse_unported(c, d),
    lambda c, d: ctherm.refuse_unported(c, d), lambda c, d: crad.refuse_unported(c, d),
    lambda c, d: cv.refuse_unported(c, d, biogeochemistry=True, z_tracers=True),
])
def test_the_other_stages_refuse_the_vertical_biochemistry_by_name(refuse):
    case = _case("bgc_algae_nutrients_c2n")
    with pytest.raises(ConfigurationRefusal, match="config_use_vertical_biochemistry"):
        refuse(case["config"], _dims(case))


def test_total_chlorophyll_without_the_vertical_biochemistry_is_refused():
    case = _case("bgc_algae", cbgc.TOTAL_CHLOROPHYLL)
    cbgc.refuse_unported(case["config"], _dims(case))
    case = _case(WITHOUT, cbgc.TOTAL_CHLOROPHYLL)
    with pytest.raises(ConfigurationRefusal, match=cbgc.TOTAL_CHLOROPHYLL):
        cbgc.refuse_unported(case["config"], _dims(case))
    # under woof_quirk_corrections = 'all' alone it is on but changes nothing there: not refused
    umbrella = {**case["config"], "woof_v842_corrections": (), "woof_quirk_corrections": "all"}
    assert cbgc.TOTAL_CHLOROPHYLL in cv.corrections(umbrella)
    cbgc.refuse_unported(umbrella, _dims(case))


def test_the_algae_parameters_follow_colpkg_init_zbgc():
    case = _case("bgc_algae_nutrients_c2n")
    to = cv.tracer_object(case["config"], _dims(case))
    alg, ap = cbgc.algae_constants(case["config"], to)
    assert alg.tolist() == [3, 0, 1, 2, 3, 4, 5, 1]
    # R_C2N(1) is the proteins' by default, the diatoms' under the correction
    assert ap[6:9].tolist() == [5.5, 7.25, 7.75]
    _, fixed = cbgc.algae_constants(_case("bgc_algae_nutrients_c2n", "zbgc_c2n_proteins")["config"], to)
    assert fixed[6:9].tolist() == [6.5, 7.25, 7.75]


@requires_gpu
def test_gpu_matches_the_oracle_word_for_word():
    report = co.measure_bgc(rows=ROWS, fixture_dir=FIXTURE)
    assert set(report) == {row[0] for row in ROWS}
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


def _mismatches(names):
    report = co.measure_bgc(rows=tuple(_row(name) for name in names), fixture_dir=FIXTURE)
    return sum(sum(row.values()) for row in report.values())


@requires_gpu
@pytest.mark.parametrize("names, original, mutated", [
    # the light limitation without its inhibition
    (_BGC_NAMES, "glibc_exp(-M(ap[AP_BETA2MAX + k], Iavg_loc)))", "1.0)"),
    # the attenuated light under the bioabsorption
    (("bgc_algae_nutrients_bioabsorption",), "op_dep > ap[AP_OP_DEP_MIN] && !dEdd_algae",
     "op_dep > ap[AP_OP_DEP_MIN]"),
    # the nitrate uptake not capped at max_loss
    (("bgc_algae_nutrients_nutrient_limited",), "    U_Nit_tot = col_min(U_Nit_tot, D(M(max_loss, Nitin), dt));\n",
     ""),
    # the temperature of the layer's interface
    (_BGC_NAMES, "grow_alg + (k - 1) * ALG_MAX, iTin[k],", "grow_alg + (k - 1) * ALG_MAX, iTin[1],"),
    # the ammonium's mortality share left with the zooplankton
    (("bgc_algae_nutrients_c2n",), "Zoo_s_m = S(M(mort_N, dt), Am_s_mo);", "Zoo_s_m = M(mort_N, dt);"),
    # nitrification
    (("bgc_algae_nutrients_bl99_types_nitrification",), "Am_r = A(Am_r, M(nitrif, dt));", ""),
    # the first alga's release and retention by the ice's thinning alone
    (_BGC_NAMES, "m == n1 ? hin_old > A(hin, algal_vel_dt) : hin_old > hin", "hin_old > hin"),
    # its stationary part always regridded
    (_BGC_NAMES, "m != n1 || meltb > algal_vel_dt || aicen < 0.001", "1"),
    # the diatoms' carbon to nitrogen ratio in the primary production
    (("bgc_algae_nutrients_c2n",), "ap[AP_R_C2N + mm]), ZB_R_GC2MOLC", "ap[AP_R_C2N + 1]), ZB_R_GC2MOLC"),
    # each alga's chlorophyll ratio
    (_BGC_NAMES, "M(net, ap[AP_R_CHL2N + mm])", "M(net, ap[AP_R_CHL2N])"),
])
def test_a_mutated_algal_branch_is_caught(monkeypatch, names, original, mutated):
    source = cbgc.module_source
    assert source(7, 7, 6).count(original) == 1
    monkeypatch.setattr(cbgc, "module_source", lambda *a: source(*a).replace(original, mutated))
    cbgc._module_on.cache_clear()
    try:
        # a mutation may trip the kernel's negative bgc stop (v8.4.2's MPAS_LOG_CRIT) before any
        # word is compared: the uncapped nitrate uptake takes the nutrient-limited cells negative
        try:
            caught = _mismatches(names) > 0
        except cbgc.CriticalAbort as stop:
            caught = "negative bgc" in str(stop)
        assert caught
    finally:
        cbgc._module_on.cache_clear()


@requires_gpu
def test_zbgc_c2n_proteins_moves_only_the_primary_production():
    name = "bgc_algae_nutrients_c2n"
    plain, _ = co.run_gpu(_case(name))
    fixed, _ = co.run_gpu(_case(name, "zbgc_c2n_proteins"))
    changed = {k for k in plain if not np.array_equal(plain[k], fixed[k], equal_nan=True)}
    assert changed == {BGC + "primaryProduction"}


@requires_gpu
def test_total_chlorophyll_counts_each_category_once():
    name = "bgc_algae_nutrients_zaero_nonreactive_four_layers_three_categories"
    case = _case(name)
    plain, _ = co.run_gpu(case)
    fixed, _ = co.run_gpu(_case(name, cbgc.TOTAL_CHLOROPHYLL))
    changed = {k for k in plain if not np.array_equal(plain[k], fixed[k], equal_nan=True)}
    assert changed == {BGC + "totalChlorophyll"}
    # v8.4.2's sum re-adds the earlier categories (and the input's running total): the
    # corrected one is smaller wherever a cell has more than one ice category
    n = case["dims"]["nCellsSolve"]
    before = case["pools"]["biogeochemistry"]["totalChlorophyll"][:n]
    added_plain = plain[BGC + "totalChlorophyll"][:n] - before
    added_fixed = fixed[BGC + "totalChlorophyll"][:n] - before
    assert (added_fixed <= added_plain).all() and (added_fixed < added_plain).any()
