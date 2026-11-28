"""MPAS-Seaice's z-aerosols through the column step and the coupled step, word for word against v8.4.2.

The reference is MPAS-Model v8.4.2 with ``config_use_column_biogeochemistry``,
``config_use_brine``, ``config_use_vertical_tracers`` and
``config_use_zaerosols`` (alone and with ``config_use_nonreactive``),
compiled with gfortran -O0 -ffp-contract=off at RKIND 8 on an x86-64 glibc
2.44 host, driven by ``tools/mpas_seaice_oracle/run_column.F90``'s
``column_step`` and coupled step over several steps. Its outputs are
committed (tests/data/oracles/mpas_seaice_bgc_zaero_step, PROVENANCE.md), so
the GPU comparison runs without gfortran. The cases use the delta-Eddington
shortwave, so they need the LGPL opt-in ``WOOF_GLIBC239_LIBM64=1`` (set
here).

With the z-aerosols the step separates each aerosol's snow and ice layers
into ``verticalAerosolsSnow`` and ``verticalAerosolsIce`` after the
pre-dynamics and combines them back before each dynamics subcycle's ridging
(each aerosol nBioLayersP3 layers in the Conc field, 2 in the Snow and
nBioLayersP1 in the Ice). ``coupling_prep`` maps the first dust's ocean flux
into ``oceanDustIronFlux`` (summed over the four dust entries onto a zero,
``mpas_seaice_column.F:4018-4021``) and skips the black carbons'.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with MPAS only to a tolerance: every word of every
   field the step may write must be bit-identical.
2. A step that skips the aerosols' separate or combine, or a
   ``coupling_prep`` that drops the dust's flux or maps the wrong tracer
   (the mutations below).
3. A stage that refuses the z-aerosols it now carries.
4. A case generator whose bytes drift: the input digests are pinned.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu
from woof.hex.mpas_seaice import column_bgc_v842 as cbgc
from woof.hex.mpas_seaice import column_radiation_v842 as crad
from woof.hex.mpas_seaice import column_step_v842 as cstep
from woof.hex.mpas_seaice import column_therm_v842 as ct
from woof.hex.mpas_seaice import column_v842 as cv
from woof.hex.mpas_seaice import coupling_prep_v842 as cpl
from woof.verify import mpas_seaice_column_oracle as co

FIXTURE = co.BGC_ZAERO_STEP_FIXTURE_DIR
ROWS = co.BGC_ZAERO_STEP_CASES
CONC = "tracers/verticalAerosolsConc"
SNOW = "tracers/verticalAerosolsSnow"
ICE = "tracers/verticalAerosolsIce"
FLUX = "biogeochemistry/oceanBioFluxes"
DUST = "biogeochemistry/oceanDustIronFlux"
DEDD = "step_zaero_bl99_dedd"
SUBCYCLES = "step_zaero_nonreactive_types_bl99_dedd_subcycles_gregorian"
COUPLED = "seaice_zaero_nonreactive_bl99_dedd_rothrock"
P1, P3 = 8, 10                          # nBioLayers + 1, + 3


def _row(name):
    return next(row for row in ROWS if row[0] == name)


@pytest.fixture
def glibc239(monkeypatch):
    monkeypatch.setenv("WOOF_GLIBC239_LIBM64", "1")


def test_case_inputs_are_the_recorded_bytes():
    recorded = {}
    for line in (FIXTURE / "inputs.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        recorded[name] = digest
    assert recorded == {f"{case['name']}/r8": co.input_digest(case) for case in co.cases("r8", ROWS)}


def test_fixture_moves_the_aerosols_through_the_step():
    fixture, logs = co.load_fixture(FIXTURE), co.load_logs(FIXTURE)
    cases = {c["name"]: c for c in co.cases("r8", ROWS)}
    assert set(logs) == set(cases) == set(fixture)
    for name, case in cases.items():
        n = case["dims"]["nCellsSolve"]
        out = fixture[name]
        nr = bool(case["config"]["config_use_nonreactive"])
        assert logs[name].get("crash") is None and logs[name]["tracer_object"]["IERR"] == [0], name
        assert logs[name]["tracer_object"]["colpkg_zbgc"][0] == 3 + nr, name
        for field in (CONC, SNOW, ICE, FLUX, DUST):
            assert field in out, (name, field)
        # every aerosol's separated snow is written, then the dynamics change the tracer after it
        for j in range(3):
            assert not np.array_equal(out[SNOW][:n, :, 2 * j:2 * j + 2],
                                      out[CONC][:n, :, j * P3 + P1:j * P3 + P1 + 2]), (name, j)
        # of the per-species ocean fluxes only the dust's is set
        for flux in cpl.BGC_OCEAN_FLUXES:
            if flux != "oceanDustIronFlux":
                assert not np.any(out[f"biogeochemistry/{flux}"][:n]), (name, flux)
        assert np.any(out[DUST][:n]), name
        assert np.isfinite(out[FLUX]).all(), name
    assert cases[SUBCYCLES]["config"]["config_dynamics_subcycle_number"] == 2


@pytest.mark.parametrize("refuse", [
    lambda c: cstep.check_config(c["config"]),
    lambda c: cpl.check_config(c["config"]),
    lambda c: ct.refuse_unported(c["config"], c["dims"]),
    lambda c: crad.refuse_unported(c["config"], c["dims"]),
    lambda c: cbgc.refuse_unported(c["config"], c["dims"]),
])
def test_the_step_s_stages_take_the_z_aerosols(refuse, glibc239):
    refuse(co.make_case(*_row(DEDD)))


def test_the_dust_is_the_third_or_fourth_bio_tracer():
    assert cpl._dust_tracer(co.make_case(*_row(DEDD))["config"]) == 2
    assert cpl._dust_tracer(co.make_case(*_row(SUBCYCLES))["config"]) == 3
    assert cpl._dust_tracer({"config_use_vertical_tracers": True, "config_use_nonreactive": True}) is None


@requires_gpu
def test_gpu_matches_the_oracle_word_for_word(glibc239):
    report = co.measure_carry(rows=ROWS, fixture_dir=FIXTURE)
    assert set(report) == {row[0] for row in ROWS}
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


def _mismatches(names, *fields):
    report = co.measure_carry(rows=tuple(_row(name) for name in names), fixture_dir=FIXTURE)
    return sum(row.get(f, 0) for row in report.values() for f in fields + ("abort",))


@requires_gpu
def test_a_step_without_the_aerosols_separate_is_caught(monkeypatch, glibc239):
    monkeypatch.setattr(cstep, "_SNOW_ICE_TRACERS", cstep._SNOW_ICE_TRACERS[1:])
    assert _mismatches((DEDD,), SNOW, ICE, CONC) > 0


@requires_gpu
def test_coupling_prep_without_the_dust_is_caught(monkeypatch, glibc239):
    monkeypatch.setattr(cpl, "_dust_tracer", lambda configs: None)
    assert _mismatches((DEDD, COUPLED), DUST) > 0


@requires_gpu
def test_coupling_prep_with_a_black_carbon_for_the_dust_is_caught(monkeypatch, glibc239):
    dust = cpl._dust_tracer
    monkeypatch.setattr(cpl, "_dust_tracer", lambda configs: dust(configs) - 1)
    assert _mismatches((DEDD, COUPLED), DUST) > 0
