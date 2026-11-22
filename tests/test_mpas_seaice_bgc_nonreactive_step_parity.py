"""MPAS-Seaice's nonreactive vertical tracer through the column step, word for word against v8.4.2.

The reference is MPAS-Model v8.4.2 with ``config_use_column_biogeochemistry``,
``config_use_brine``, ``config_use_vertical_tracers`` and
``config_use_nonreactive``, compiled with gfortran -O0 -ffp-contract=off at
RKIND 8 on an x86-64 glibc 2.44 host, driven by
``tools/mpas_seaice_oracle/run_column.F90``'s ``column_step`` (the column
package's part of ``seaice_timestep`` over several steps). Its outputs are
committed (tests/data/oracles/mpas_seaice_bgc_nonreactive_step,
PROVENANCE.md), so the GPU comparison runs without gfortran. The cases use
the delta-Eddington shortwave, so they need the LGPL opt-in
``WOOF_GLIBC239_LIBM64=1`` (set here).

With the vertical tracers the step separates each tracer's snow and ice
layers into its ``*Snow`` and ``*Ice`` fields after the pre-dynamics
(``column_separate_snow_ice_tracers``) and combines them back before each
dynamics subcycle's ridging (``column_combine_snow_ice_tracers``); the
vertical thermodynamics, the shortwave and ``coupling_prep`` take the
tracer (``coupling_prep`` zeroes the per-species ocean fluxes, which have
no nonreactive species).

Two v8.4.2 quirks are kept word for word, each with an opt-in correction
(``woof_v842_corrections``) tested here for physical sense, not against
the oracle:
- ``subcycle_snow_ice_tracers``: the combine runs every subcycle but the
  separate once a step, so with two subcycles the first subcycle's ridging
  of the tracer is overwritten;
- ``bgc_snow_new_category``: without the vertical thermodynamics
  ``iceAreaCategoryInitial`` is never refreshed, and ``update_snow_bgc``'s
  0/0 for a category the last step opened makes ``oceanBioFluxes`` NaN.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with MPAS only to a tolerance: every word of every
   field the step may write must be bit-identical.
2. A step that skips the separate or the combine, or does not reset the
   ocean biogeochemistry fluxes (the mutations below).
3. A stage that refuses the vertical tracers it now carries, or a
   correction that runs where it means nothing.
4. A correction that does not do what it says.
5. A case generator whose bytes drift: the input digests are pinned.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu
from woof.hex.errors import ConfigurationRefusal
from woof.hex.mpas_seaice import column_bgc_v842 as cbgc
from woof.hex.mpas_seaice import column_radiation_v842 as crad
from woof.hex.mpas_seaice import column_step_v842 as cstep
from woof.hex.mpas_seaice import column_therm_v842 as ct
from woof.hex.mpas_seaice import column_v842 as cv
from woof.hex.mpas_seaice import coupling_prep_v842 as cpl
from woof.verify import mpas_seaice_column_oracle as co

FIXTURE = co.BGC_NONREACTIVE_STEP_FIXTURE_DIR
ROWS = co.BGC_NONREACTIVE_STEP_CASES
CONC = "tracers/verticalNonreactiveConc"
SNOW = "tracers/verticalNonreactiveSnow"
ICE = "tracers/verticalNonreactiveIce"
FLUX = "biogeochemistry/oceanBioFluxes"
DEDD = "step_nonreactive_bl99_dedd"
SUBCYCLES = "step_nonreactive_bl99_dedd_subcycles_gregorian"
NO_THERM = "step_nonreactive_mushy_level_ponds_no_vertical_thermodynamics"
P1 = 8                                  # nBioLayers + 1


def _row(name):
    return next(row for row in ROWS if row[0] == name)


def _case(name, *corrections):
    case = co.make_case(*_row(name))
    if corrections:
        case["config"]["woof_v842_corrections"] = (*cv.corrections(case["config"]), *corrections)
    return case


@pytest.fixture
def glibc239(monkeypatch):
    monkeypatch.setenv("WOOF_GLIBC239_LIBM64", "1")


def test_case_inputs_are_the_recorded_bytes():
    recorded = {}
    for line in (FIXTURE / "inputs.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        recorded[name] = digest
    assert recorded == {f"{case['name']}/r8": co.input_digest(case) for case in co.cases("r8", ROWS)}


def test_fixture_moves_the_tracer_through_the_step():
    fixture, logs = co.load_fixture(FIXTURE), co.load_logs(FIXTURE)
    cases = {c["name"]: c for c in co.cases("r8", ROWS)}
    assert set(logs) == set(cases) == set(fixture)
    for name, case in cases.items():
        n = case["dims"]["nCellsSolve"]
        out = fixture[name]
        assert cv.vertical_tracers(case["config"]), name
        assert logs[name].get("crash") is None and logs[name]["tracer_object"]["IERR"] == [0], name
        for field in (CONC, SNOW, ICE, FLUX):
            assert field in out, (name, field)
        # the separated fields are written, then the dynamics change the tracer after them
        assert not np.array_equal(out[SNOW][:n], out[CONC][:n, :, P1:P1 + 2]), name
        # coupling_prep zeroes the per-species ocean fluxes; the tracer's own flux is the nonreactive slot
        for flux in cpl.BGC_OCEAN_FLUXES:
            assert not np.any(out[f"biogeochemistry/{flux}"][:n]), (name, flux)
        assert np.any(np.nan_to_num(out[FLUX][:n, 0], nan=1.0)) and not np.any(out[FLUX][:n, 1:]), name
    assert cases[SUBCYCLES]["config"]["config_dynamics_subcycle_number"] == 2
    # v8.4.2's 0/0 in update_snow_bgc without the vertical thermodynamics (bgc_snow_new_category)
    assert np.argwhere(np.isnan(fixture[NO_THERM][FLUX])).tolist() == [[1, 0]]
    assert not cases[NO_THERM]["pools"]["icestate"]["iceAreaCategoryInitial"][1].any()


@pytest.mark.parametrize("refuse", [
    lambda c: cstep.check_config(c["config"]),
    lambda c: cpl.check_config(c["config"]),
    lambda c: ct.refuse_unported(c["config"], c["dims"]),
    lambda c: crad.refuse_unported(c["config"], c["dims"]),
    lambda c: cbgc.refuse_unported(c["config"], c["dims"]),
])
def test_the_step_s_stages_take_the_vertical_tracers(refuse, glibc239):
    refuse(_case(DEDD))


def test_the_stages_that_do_not_carry_them_still_refuse_them():
    case = _case(DEDD)
    with pytest.raises(ConfigurationRefusal, match=cv.VT_FLAG):
        cv.refuse_unported(case["config"], {**case["dims"], "nSnowLayers": 1}, biogeochemistry=True)


@pytest.mark.parametrize("correction, refuse", [
    (cstep.SUBCYCLE_SNOW_ICE, lambda c: cstep.check_config(c["config"])),
    (cbgc.SNOW_NEW_CATEGORY, lambda c: cbgc.refuse_unported(c["config"], c["dims"])),
])
def test_a_correction_without_the_vertical_tracers_is_refused(correction, refuse):
    case = _case(DEDD, correction)
    refuse(case)
    case["config"] = {**case["config"], cv.VT_FLAG: False, "config_use_nonreactive": False}
    with pytest.raises(ConfigurationRefusal, match=correction):
        refuse(case)
    # under woof_quirk_corrections = 'all' alone it is on but changes nothing, so not refused
    case["config"] = {**case["config"], "woof_v842_corrections": (), "woof_quirk_corrections": "all"}
    assert correction in cv.corrections(case["config"])
    refuse(case)


@requires_gpu
def test_gpu_matches_the_oracle_word_for_word(glibc239):
    report = co.measure_carry(rows=ROWS, fixture_dir=FIXTURE)
    assert set(report) == {row[0] for row in ROWS}
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


def test_a_nan_matches_any_nan_but_nothing_else():
    # the NaN's sign and payload are the processor's (Metal's soft float flips the sign of this one)
    nan, neg = np.float64(np.nan), -np.float64(np.nan)
    assert co.nan_word_mismatches(np.array([nan, 1.0]), np.array([neg, 1.0])) == 0
    assert co.nan_word_mismatches(np.array([nan, 0.0]), np.array([1.0, -0.0])) == 2


def _mismatches(names, *fields):
    report = co.measure_carry(rows=tuple(_row(name) for name in names), fixture_dir=FIXTURE)
    return sum(row.get(f, 0) for row in report.values() for f in fields + ("abort",))


@requires_gpu
def test_a_step_without_the_separate_is_caught(monkeypatch, glibc239):
    monkeypatch.setattr(cstep, "separate_snow_ice_tracers", lambda pools, configs, state: None)
    # the combine then hands the tracer the input's *Snow and *Ice, and the next step's
    # z_biogeochemistry stops on them where v8.4.2 runs on
    with pytest.raises(cbgc.CriticalAbort, match="zbgc initialization error"):
        _mismatches((DEDD,), SNOW, ICE)


@requires_gpu
def test_a_step_without_the_subcycles_combine_is_caught(monkeypatch, glibc239):
    monkeypatch.setattr(cstep, "combine_snow_ice_tracers", lambda pools, configs, state: None)
    assert _mismatches((SUBCYCLES,), CONC) > 0


@requires_gpu
def test_a_step_that_keeps_the_tracer_s_ocean_flux_is_caught(monkeypatch, glibc239):
    monkeypatch.setattr(cstep, "_BGC_FIELDS", tuple(f for f in cstep._BGC_FIELDS if f != "oceanBioFluxes"))
    assert _mismatches((DEDD,), FLUX) > 0


@requires_gpu
def test_subcycle_snow_ice_tracers_keeps_each_subcycle_s_ridging(glibc239):
    n = 9
    plain, _ = co.run_gpu(_case(SUBCYCLES))
    fixed, _ = co.run_gpu(_case(SUBCYCLES, cstep.SUBCYCLE_SNOW_ICE))
    # the first subcycle's ridging of the tracer survives the second combine
    assert not np.array_equal(plain[CONC][:n], fixed[CONC][:n])
    # the separated fields are the tracer after the last subcycle
    assert np.array_equal(fixed[SNOW][:n], fixed[CONC][:n, :, P1:P1 + 2])
    assert np.array_equal(fixed[ICE][:n], fixed[CONC][:n, :, :P1])
    # with one subcycle only the separated fields change
    one, _ = co.run_gpu(_case(DEDD))
    one_fixed, _ = co.run_gpu(_case(DEDD, cstep.SUBCYCLE_SNOW_ICE))
    assert {k for k in one if not np.array_equal(one[k], one_fixed[k], equal_nan=True)} <= {SNOW, ICE}
    assert np.array_equal(one_fixed[SNOW][:n], one_fixed[CONC][:n, :, P1:P1 + 2])


@requires_gpu
def test_bgc_snow_new_category_gives_a_finite_flux(glibc239):
    plain, _ = co.run_gpu(_case(NO_THERM))
    fixed, _ = co.run_gpu(_case(NO_THERM, cbgc.SNOW_NEW_CATEGORY))
    assert np.isnan(plain[FLUX][1, 0]) and np.isfinite(fixed[FLUX]).all()
    # the category had no snow before it opened: nothing else moves
    same = np.isfinite(plain[FLUX])
    assert np.array_equal(plain[FLUX][same], fixed[FLUX][same])
    assert {k for k in plain if not np.array_equal(plain[k], fixed[k], equal_nan=True)} == {FLUX}
