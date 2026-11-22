"""MPAS-Seaice's nonreactive vertical tracer through the ITD thermodynamics and the ridging, word for word against v8.4.2.

The reference is MPAS-Model v8.4.2 with ``config_use_column_biogeochemistry``,
``config_use_brine``, ``config_use_vertical_tracers`` and
``config_use_nonreactive``, compiled with gfortran -O0 -ffp-contract=off at
RKIND 8 on an x86-64 glibc 2.44 host, driven by
``tools/mpas_seaice_oracle/run_column.F90``. Its outputs are committed
(tests/data/oracles/mpas_seaice_bgc_nonreactive_carry, PROVENANCE.md), so the
GPU comparison runs without gfortran.

With the column package's ``z_tracers`` the ITD thermodynamics run:
- ``add_new_ice_bgc``: the frazil's tracer taken from ``oceanBioFluxes`` and
  ``adjust_tracer_profile`` (``regrid_stationary`` on the bio grid under a
  thickness surplus, or the frazil mixed throughout);
- ``lateral_melt``'s snow tracers and ``lateral_melt_bgc``'s ice layers;
- ``cleanup_itd``: ``zap_small_areas``' ``zap_small_bgc`` and ``zap_snow``'s
  snow tracers (also in ``zap_snow_temperature``), into ``oceanBioFluxes``.
The ridging carries the tracers through the tracer object and runs the same
cleanup.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with MPAS only to a tolerance: every word of every
   field the stages may write must be bit-identical.
2. Each branch above dropped or changed (the mutations below).
3. ``regrid_stationary``'s remap stop not raised as the ITD's abort.
4. A case generator whose bytes drift: the input digests are pinned.
"""
from __future__ import annotations

import pytest

from conftest import requires_gpu
from woof.hex.errors import ConfigurationRefusal
from woof.hex.mpas_seaice import column_itd_v842 as citd
from woof.hex.mpas_seaice import column_ridge_v842 as crdg
from woof.hex.mpas_seaice import column_v842 as cv
from woof.verify import mpas_seaice_column_oracle as co

FIXTURE = co.BGC_NONREACTIVE_CARRY_FIXTURE_DIR
ROWS = co.BGC_NONREACTIVE_CARRY_CASES
CONC = "tracers/verticalNonreactiveConc"
FLUX = "biogeochemistry/oceanBioFluxes"
ABORT = "itd_nonreactive_abort_regrid"
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


def test_fixture_moves_the_nonreactive_tracer():
    fixture, logs = co.load_fixture(FIXTURE), co.load_logs(FIXTURE)
    cases = {c["name"]: c for c in co.cases("r8", ROWS)}
    assert set(logs) == set(cases) == set(fixture)
    for name, case in cases.items():
        assert cv.vertical_tracers(case["config"]), name
        assert logs[name].get("crash") is None, name
        assert logs[name]["tracer_object"]["colpkg_zbgc"][0] == 1, name
    # regrid_stationary's remap stop on the tracer: add_new_ice_bgc's abort, cell 2 (indexToCellID 3)
    assert logs[ABORT]["log"] == ["ERR column_itd_thermodynamics: ice: remap_layers_bgc error", "ERR iCell: $i",
                                  "INTS 3"]
    for name in _ITD_NAMES + _RIDGE_NAMES:
        assert logs[name]["tracer_object"]["IERR"] == [0], name
        assert CONC in fixture[name] and FLUX in fixture[name], name


def test_the_itd_and_the_ridging_take_the_vertical_tracers():
    for name in (_ITD_NAMES[0], _RIDGE_NAMES[0]):
        case = co.make_case(*_row(name))
        citd.refuse_unported(case["config"], {**case["dims"], "nSnowLayers": 1})


def test_the_other_stages_still_refuse_them():
    case = co.make_case(*_row(_ITD_NAMES[0]))
    with pytest.raises(ConfigurationRefusal, match=cv.VT_FLAG):
        cv.refuse_unported(case["config"], {**case["dims"], "nSnowLayers": 1}, biogeochemistry=True)


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
    # the frazil's tracer from its final volume, not add_new_ice's first
    (_ITD_NAMES, "M(M(D(vi0_init, dt), itd_ocean_bio(T, m)), T.init_frac[m])",
     "M(M(D(vi0new, dt), itd_ocean_bio(T, m)), T.init_frac[m])"),
    # regrid_stationary's top concentration without zbgc_init_frac
    (_ITD_NAMES, "const double top_conc = M(itd_ocean_bio(T, m), T.init_frac[m]);",
     "const double top_conc = itd_ocean_bio(T, m);"),
    # the frazil mixed through the profile without its tracer
    (_ITD_NAMES, "zbio[k - 1] = D(A(M(zbio[k - 1], vtmp), M(M(itd_ocean_bio(T, m), T.init_frac[m]), vsurp)), vbrin);",
     "zbio[k - 1] = D(M(zbio[k - 1], vtmp), vbrin);"),
    # lateral_melt without the snow tracers
    (_ITD_NAMES, "T.flux_bio[m] = A(T.flux_bio[m], D(M(A(M(tr[sn], dvssl), M(tr[sn + 1], dvint)), rside), dt));", ""),
    # lateral_melt_bgc on the volume after the melt
    (_ITD_NAMES, "T.flux_bio[m] = A(T.flux_bio[m], D(M(M(M(M(tr[ix[TI_FBRI]], vicen_init[n]), zspace)",
     "T.flux_bio[m] = A(T.flux_bio[m], D(M(M(M(M(tr[ix[TI_FBRI]], vicen[n]), zspace)"),
    # zap_small_bgc dropped
    (_ITD_NAMES + _RIDGE_NAMES, "dflux_bio[m] = A(dflux_bio[m], D(M(tr[T.bio[m] + k - 1], bvol), dt));", ""),
    # zap_snow without the snow tracers
    (_ITD_NAMES + _RIDGE_NAMES, "dflux_bio[m] = A(dflux_bio[m], D(A(M(tr[sn], dvssl), M(tr[sn + 1], dvint)), dt));",
     ""),
    # cleanup_itd's zapped tracer not added to the flux
    (_ITD_NAMES + _RIDGE_NAMES, "for (int m = 0; m < T.nbio; ++m) T.flux_bio[m] = A(T.flux_bio[m], dflux_bio[m]);",
     ""),
    # the tracer's regrid_stationary stop not checked
    ((ABORT,), "            int stop = itd_regrid_points(cs, nbl + 1, ig, hbri_old, hbri, top_conc);\n"
               "            if (stop) return stop;\n",
     "            (void)itd_regrid_points(cs, nbl + 1, ig, hbri_old, hbri, top_conc);\n"),
])
def test_a_mutated_vertical_tracer_branch_is_caught(monkeypatch, names, original, mutated):
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
