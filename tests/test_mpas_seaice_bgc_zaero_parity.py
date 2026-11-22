"""MPAS-Seaice's vertical tracers with the z-aerosols, word for word against v8.4.2.

The reference is MPAS-Model v8.4.2 with ``config_use_column_biogeochemistry``,
``config_use_brine``, ``config_use_vertical_tracers`` and
``config_use_zaerosols`` (alone and with ``config_use_nonreactive``),
compiled with gfortran -O0 -ffp-contract=off at RKIND 8 on an x86-64 glibc
2.44 host, driven by ``tools/mpas_seaice_oracle/run_column.F90`` through the
init and ``column_biogeochemistry``. Its outputs are committed
(tests/data/oracles/mpas_seaice_bgc_zaero, PROVENANCE.md), so the GPU
comparison runs without gfortran.

The column package's ``z_tracers`` with the Registry's three z-aerosols
(black carbon 1 and 2, the first dust), after the nonreactive tracer when it
is on:
- the init: ``colpkg_init_zbgc``'s tracer layout (``bio_index``,
  ``bio_index_o`` 23-25), each aerosol's transport type, retention and
  release times and mobile fraction, and ``colpkg_init_bgc``'s profiles in
  ``verticalAerosolsConc`` (nBioLayersP3 layers per aerosol);
- ``column_biogeochemistry``: the uncoupled deposition into
  ``atmosBioFluxes`` (the whole array zeroed on every cell's turn, so only
  the last owned cell keeps it), ``update_snow_bgc`` over every tracer (its
  negative-snow clamp spans them all), ``z_biogeochemistry``'s phases, each
  over every tracer before the next, and ``merge_bgc_fluxes``.

v8.4.2 scavenges every tracer from melting snow with ``kscavz(k)``, ``k``
the tracer's place among the bio tracers, where the table is laid out by
``bio_index_o``: the aerosols' own coefficients are never used. That is kept
word for word; the opt-in correction ``zaero_snow_scavenging`` is tested
here for what it changes, not against the oracle.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with MPAS only to a tolerance: every word of every
   field the stages may write must be bit-identical.
2. A tracer object that drifts: the aerosols' indices must be the oracle's.
3. A port that runs each tracer through z_biogeochemistry before the next:
   ``abort_order`` stops on the second aerosol's initialization error, where
   such a port meets the nonreactive tracer's negative bgc first.
4. A gate that cannot fail: each mutated branch must produce mismatches.
5. A configuration the port does not carry running silently: the
   z-aerosols in a stage that does not carry them, and the correction
   without them, are refused by name.
6. A case generator whose bytes drift: the input digests are pinned.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu
from woof.hex.errors import ConfigurationRefusal
from woof.hex.mpas_seaice import column_bgc_v842 as cbgc
from woof.hex.mpas_seaice import column_itd_v842 as citd
from woof.hex.mpas_seaice import column_v842 as cv
from woof.verify import mpas_seaice_column_oracle as co

FIXTURE = co.BGC_ZAERO_FIXTURE_DIR
ROWS = co.BGC_ZAERO_CASES
AERO = "tracers/verticalAerosolsConc"
NONREACTIVE = "tracers/verticalNonreactiveConc"
FRAC = "tracers/mobileFraction"
FLUX = "biogeochemistry/oceanBioFluxes"
ATMOS = "biogeochemistry/atmosBioFluxes"
ABORTS = {"bgc_zaero_nonreactive_abort_negative": "column_biogeochemistry: negative bgc",
          "bgc_zaero_nonreactive_abort_order": "column_biogeochemistry: zbgc initialization error"}
_BGC_NAMES = tuple(row[0] for row in ROWS if row[2] == "biogeochemistry" and row[0] not in ABORTS)
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
        n_bio = mine["colpkg_zbgc"][0]
        # the aerosols' places in the ocean concentrations: 2*max_algae + max_doc + max_dic +
        # max_don + 2*max_fe + 7 + mm
        assert mine["colpkg_zbgc"][-3:] == [23, 24, 25], case["name"]
        assert n_bio == 3 + bool(case["config"]["config_use_nonreactive"])


def test_fixture_shows_the_z_aerosols():
    fixture, logs = co.load_fixture(FIXTURE), co.load_logs(FIXTURE)
    assert set(logs) == {row[0] for row in ROWS}
    for name in set(logs) - set(ABORTS):
        assert logs[name].get("crash") is None, name
        assert not [line for line in logs[name]["log"] if line.startswith("ERR")], name
    for name in _INIT_NAMES:
        assert AERO in fixture[name] and FRAC in fixture[name], name
    for name in _BGC_NAMES:
        case = _case(name)
        n = case["dims"]["nCellsSolve"]
        for field in (AERO, FRAC, FLUX, ATMOS, "biogeochemistry/totalVerticalBiologySnow"):
            assert field in fixture[name], (name, field)
        # every aerosol's ice layers move
        for j in range(3):
            assert not np.array_equal(fixture[name][AERO][:n, :, j * P3:j * P3 + 8],
                                      case["pools"]["tracers"]["verticalAerosolsConc"][:n, :, j * P3:j * P3 + 8])
        # atmosBioFluxes keeps only the last owned cell's deposition, at the aerosols' places
        atmos = fixture[name][ATMOS]
        first = int(case["config"]["config_use_nonreactive"])
        assert not atmos[:n - 1].any() and not atmos[n:].any(), name
        assert atmos[n - 1].tolist()[first:first + 3] == [1.0e-12, 1.0e-12, 1.0e-13], name
        assert not atmos[n - 1, :first].any() and not atmos[n - 1, first + 3:].any(), name
    for name, message in ABORTS.items():
        assert logs[name]["crash"]["error"] == message


def test_the_init_the_biogeochemistry_and_the_itd_carry_the_z_aerosols():
    for name in ("init_bgc_zaero", "bgc_zaero_nonreactive"):
        case = _case(name)
        cv.refuse_unported(case["config"], _dims(case), biogeochemistry=True, z_tracers=True)
        cbgc.refuse_unported(case["config"], _dims(case))
        citd.refuse_unported(case["config"], _dims(case))


def test_the_stages_without_the_vertical_tracers_refuse_them():
    case = _case("bgc_zaero_nonreactive")
    with pytest.raises(ConfigurationRefusal, match=cv.VT_FLAG):
        cv.refuse_unported(case["config"], _dims(case), biogeochemistry=True)


def test_the_vertical_tracers_without_a_tracer_are_refused():
    case = _case("bgc_zaero")
    config = {**case["config"], cv.ZAERO_FLAG: False}
    with pytest.raises(ConfigurationRefusal, match="config_use_nonreactive"):
        cv.refuse_unported(config, _dims(case), biogeochemistry=True, z_tracers=True)


def test_zaero_snow_scavenging_without_the_z_aerosols_is_refused():
    case = _case("bgc_zaero_nonreactive", cv.ZAERO_KSCAVZ)
    cbgc.refuse_unported(case["config"], _dims(case))
    config = {**case["config"], cv.ZAERO_FLAG: False}
    with pytest.raises(ConfigurationRefusal, match=cv.ZAERO_KSCAVZ):
        cbgc.refuse_unported(config, _dims(case))


def test_kscavz_is_v842_s_by_default_and_the_table_s_under_the_correction():
    case = _case("bgc_zaero_nonreactive")
    to = cv.tracer_object(case["config"], _dims(case))
    assert cv.zbgc_parameters(case["config"], to)["kscavz"] == [0.03] * 4
    case = _case("bgc_zaero_nonreactive", cv.ZAERO_KSCAVZ)
    assert cv.zbgc_parameters(case["config"], to)["kscavz"] == [0.03, 0.03, 0.20, 0.02]


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
        with pytest.raises(cbgc.CriticalAbort) as error:
            co.run_gpu(_case(name))
        assert error.value.crash == logs[name]["crash"], name


def _mismatches(names):
    report = co.measure_bgc(rows=tuple(_row(name) for name in names), fixture_dir=FIXTURE)
    return sum(sum(row.values()) for row in report.values())


@requires_gpu
@pytest.mark.parametrize("names, original, mutated", [
    # the deposition onto the wrong aerosols
    (_BGC_NAMES, "src < 2 ? atm_bc[iCell * 2 + src] : atm_dust[iCell * 4 + src - 2]",
     "src < 1 ? atm_bc[iCell * 2 + src] : atm_dust[iCell * 4]"),
    # each tracer's own scavenging coefficient
    (_BGC_NAMES, "flux_bio_atm[m], zpar[m * ZP_N + ZP_KSCAVZ]", "flux_bio_atm[m], zpar[ZP_KSCAVZ] * (m + 1)"),
    # another tracer's atmospheric deposition
    (_BGC_NAMES, "A(A(zbgc_snow[m], zbgc_atm[m]), 0.0)", "A(A(zbgc_snow[m], zbgc_atm[0]), 0.0)"),
    # each tracer's ocean concentration
    (_BGC_NAMES, "ocean_conc[(size_t)iCell * nzbgc + bio_o[m]]", "ocean_conc[(size_t)iCell * nzbgc + bio_o[0]]"),
    # the new ice's mobile fractions: only the first reset
    (_BGC_NAMES, "for (int m = 0; m < NBT; ++m) ztr_frac[(size_t)c * nzbgc + m]",
     "for (int m = 0; m < 1; ++m) ztr_frac[(size_t)c * nzbgc + m]"),
    # the phases run tracer by tracer: the initialization check after the others
    (("bgc_zaero_nonreactive_abort_order",), "            if (*t < 0.0) return BG_ZBGC_INIT;\n", ""),
])
def test_a_mutated_z_aerosol_is_caught(monkeypatch, names, original, mutated):
    source = cbgc.module_source
    assert source(7, 7, 4).count(original) == 1
    monkeypatch.setattr(cbgc, "module_source", lambda *a: source(*a).replace(original, mutated))
    cbgc._module_on.cache_clear()
    try:
        assert _mismatches(names) > 0
    finally:
        cbgc._module_on.cache_clear()


@requires_gpu
def test_zaero_snow_scavenging_scavenges_the_aerosols_at_their_own_rates():
    name = "bgc_zaero_nonreactive_bl99_types"
    plain, _ = co.run_gpu(_case(name))
    fixed, _ = co.run_gpu(_case(name, cv.ZAERO_KSCAVZ))
    changed = {k for k in plain if not np.array_equal(plain[k], fixed[k], equal_nan=True)}
    assert changed and changed <= {AERO, FLUX, FRAC, "biogeochemistry/snowIceBioFluxes",
                                   "biogeochemistry/atmosIceBioFluxes", "biogeochemistry/totalVerticalBiologyIce",
                                   "biogeochemistry/totalVerticalBiologySnow"}
    # the first black carbon keeps 0.03, as does the nonreactive tracer
    assert np.array_equal(plain[NONREACTIVE], fixed[NONREACTIVE])
    assert np.array_equal(plain[AERO][..., :P3], fixed[AERO][..., :P3])
    assert np.array_equal(plain[FLUX][:, :2], fixed[FLUX][:, :2])
    # the second black carbon (0.20) and the dust (0.02) do not
    assert not np.array_equal(plain[AERO][..., P3:2 * P3], fixed[AERO][..., P3:2 * P3])
    assert not np.array_equal(plain[AERO][..., 2 * P3:], fixed[AERO][..., 2 * P3:])
