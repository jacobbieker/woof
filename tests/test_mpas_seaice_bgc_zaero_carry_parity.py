"""MPAS-Seaice's z-aerosols through the ITD thermodynamics and the ridging, word for word against v8.4.2.

The reference is MPAS-Model v8.4.2 with ``config_use_column_biogeochemistry``,
``config_use_brine``, ``config_use_vertical_tracers`` and
``config_use_zaerosols`` (alone and with ``config_use_nonreactive``),
compiled with gfortran -O0 -ffp-contract=off at RKIND 8 on an x86-64 glibc
2.44 host, driven by ``tools/mpas_seaice_oracle/run_column.F90``. Its outputs
are committed (tests/data/oracles/mpas_seaice_bgc_zaero_carry, PROVENANCE.md),
so the GPU comparison runs without gfortran.

The ITD thermodynamics and the ridging run the vertical tracers' branches
(``add_new_ice_bgc``, ``adjust_tracer_profile``, ``lateral_melt``'s snow,
``lateral_melt_bgc``, ``zap_small_bgc`` and ``zap_snow``) over every tracer,
each with its own ``bio_index``, ocean concentration (``bio_index_o``),
``zbgc_init_frac`` and ``oceanBioFluxes`` entry. With a transport type per
aerosol the tracers' ``zbgc_init_frac`` differ (a type below 0 takes
``config_new_ice_fraction_biotracer``, the others
``config_fraction_biotracer_in_frazil``).

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with MPAS only to a tolerance: every word of every
   field the stages may write must be bit-identical.
2. A port that runs the first tracer's index, ocean concentration, frazil
   fraction or flux entry for every tracer (the mutations below).
3. ``regrid_stationary``'s remap stop not raised as the ITD's abort.
4. A case generator whose bytes drift: the input digests are pinned.
"""
from __future__ import annotations

import pytest

from conftest import requires_gpu
from woof.hex.mpas_seaice import column_itd_v842 as citd
from woof.hex.mpas_seaice import column_ridge_v842 as crdg
from woof.hex.mpas_seaice import column_v842 as cv
from woof.verify import mpas_seaice_column_oracle as co

FIXTURE = co.BGC_ZAERO_CARRY_FIXTURE_DIR
ROWS = co.BGC_ZAERO_CARRY_CASES
AERO = "tracers/verticalAerosolsConc"
FLUX = "biogeochemistry/oceanBioFluxes"
ABORT = "itd_zaero_nonreactive_abort_regrid"
_ITD_NAMES = tuple(row[0] for row in ROWS if row[2] == "itd_thermodynamics" and row[0] != ABORT)
_RIDGE_NAMES = tuple(row[0] for row in ROWS if row[2] == "ridging")


def _row(name):
    return next(row for row in ROWS if row[0] == name)


def test_case_inputs_are_the_recorded_bytes():
    recorded = {}
    for line in (FIXTURE / "inputs.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        recorded[name] = digest
    assert recorded == {f"{case['name']}/r8": co.input_digest(case) for case in co.cases("r8", ROWS)}


def test_fixture_moves_the_z_aerosols():
    fixture, logs = co.load_fixture(FIXTURE), co.load_logs(FIXTURE)
    cases = {c["name"]: c for c in co.cases("r8", ROWS)}
    assert set(logs) == set(cases) == set(fixture)
    for name, case in cases.items():
        assert logs[name].get("crash") is None, name
        nbio = 4 if case["config"]["config_use_nonreactive"] else 3
        assert logs[name]["tracer_object"]["colpkg_zbgc"][0] == nbio, name
    # regrid_stationary's remap stop on the first tracer: add_new_ice_bgc's abort, cell 2 (indexToCellID 3)
    assert logs[ABORT]["log"] == ["ERR column_itd_thermodynamics: ice: remap_layers_bgc error", "ERR iCell: $i",
                                  "INTS 3"]
    for name in _ITD_NAMES + _RIDGE_NAMES:
        assert logs[name]["tracer_object"]["IERR"] == [0], name
        assert AERO in fixture[name] and FLUX in fixture[name], name


def test_the_itd_and_the_ridging_take_the_z_aerosols():
    for name in (_ITD_NAMES[0], _ITD_NAMES[1], _RIDGE_NAMES[0]):
        case = co.make_case(*_row(name))
        crdg.refuse_unported(case["config"], {**case["dims"], "nSnowLayers": 1})


def test_the_frazil_fractions_differ_by_transport_type():
    case = co.make_case(*_row("itd_zaero_nonreactive_types_frazil_three_layers"))
    state = cv.init_parameters(case["pools"], case["config"], biogeochemistry=True, z_tracers=True)
    # the nonreactive tracer and the dust not mobile (0.7), the black carbons mobile (0.6)
    assert cv.zbgc_parameters(case["config"], state.tracers)["init_frac"] == [0.7, 0.6, 0.6, 0.7]


@requires_gpu
def test_gpu_matches_the_oracle_word_for_word():
    report = co.measure_carry(rows=ROWS, fixture_dir=FIXTURE)
    assert set(report) == {row[0] for row in ROWS}
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


def _mismatches(names):
    report = co.measure_carry(rows=tuple(_row(name) for name in names), fixture_dir=FIXTURE)
    return sum(sum(row.values()) for row in report.values())


@requires_gpu
@pytest.mark.parametrize("names, original, mutated", [
    # every tracer's ocean concentration the first's
    (_ITD_NAMES, "return T.ocean_conc[T.bio_o[m]];", "return T.ocean_conc[T.bio_o[0]];"),
    # every tracer's frazil fraction the first's
    (("itd_zaero_nonreactive_types_frazil_three_layers",),
     "M(M(D(vi0_init, dt), itd_ocean_bio(T, m)), T.init_frac[m])",
     "M(M(D(vi0_init, dt), itd_ocean_bio(T, m)), T.init_frac[0])"),
    # zap_small_bgc on the first tracer's layers for every tracer
    (_ITD_NAMES + _RIDGE_NAMES, "dflux_bio[m] = A(dflux_bio[m], D(M(tr[T.bio[m] + k - 1], bvol), dt));",
     "dflux_bio[m] = A(dflux_bio[m], D(M(tr[T.bio[0] + k - 1], bvol), dt));"),
    # cleanup_itd's zapped tracers all added to the first flux
    (_ITD_NAMES + _RIDGE_NAMES, "for (int m = 0; m < T.nbio; ++m) T.flux_bio[m] = A(T.flux_bio[m], dflux_bio[m]);",
     "for (int m = 0; m < T.nbio; ++m) T.flux_bio[0] = A(T.flux_bio[0], dflux_bio[m]);"),
    # lateral_melt_bgc on the first tracer only
    (_ITD_NAMES, "    for (int m = 0; m < T.nbio; ++m)\n        for (int n = 0; n < ncat; ++n) {",
     "    for (int m = 0; m < 1; ++m)\n        for (int n = 0; n < ncat; ++n) {"),
])
def test_a_mutated_tracer_loop_is_caught(monkeypatch, names, original, mutated):
    source = citd.module_source
    assert source(7).count(original) == 1
    monkeypatch.setattr(citd, "module_source", lambda nil: source(nil).replace(original, mutated))
    citd._module_on.cache_clear()
    crdg._module_on.cache_clear()
    try:
        assert _mismatches(names) > 0
    finally:
        citd._module_on.cache_clear()
        crdg._module_on.cache_clear()
