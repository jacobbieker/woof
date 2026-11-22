"""MPAS-Seaice's solved vertical salinity (``zsal_solve``) through the ITD, the ridging and the time step, word for word against the patched oracle.

v8.4.2 never solves the vertical salinity; its default words are
tests/test_mpas_seaice_bgc_zsal_carry_parity.py's. The opt-in correction
``zsal_solve`` (``woof.hex.mpas_seaice.column_v842.CORRECTIONS``) runs the
column package's ``solve_zsal`` branches as they are written. The reference
is MPAS-Model v8.4.2 patched with ``tools/mpas_seaice_oracle/zsal_solve.patch``
(``WOOF_ORACLE_PATCH=zsal_solve``), compiled with gfortran -O0
-ffp-contract=off at RKIND 8 on an x86-64 glibc 2.44 host. Its outputs are
committed (tests/data/oracles/mpas_seaice_bgc_zsal_solve_carry,
PROVENANCE.md), so the GPU comparison runs without gfortran. The step cases
use the delta-Eddington shortwave, so they need the LGPL opt-in
``WOOF_GLIBC239_LIBM64=1`` (set here).

Under ``solve_zsal`` the ITD thermodynamics (and the ridging's cleanup) run:
- ``add_new_ice``'s salt loss with the frazil into ``zSalinityFlux``;
- ``add_new_ice_bgc``'s ``adjust_tracer_profile``: ``regrid_stationary``
  under a thickness surplus, the frazil mixed throughout, and the
  ``remap_zbgc`` back into ``sice``;
- the first category's surface temperature from that ``sice``;
- ``lateral_melt_bgc``'s and ``zap_small_areas``' salt into ``zSalinityFlux``.
The steps also run ``zsalinity`` in ``column_biogeochemistry``.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with the patched column package only to a tolerance:
   every word of every field the stages may write must be bit-identical.
2. Each ``solve_zsal`` branch dropped or changed (the mutations below).
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

FIXTURE = co.BGC_ZSAL_SOLVE_CARRY_FIXTURE_DIR
ROWS = co.BGC_ZSAL_SOLVE_CARRY_CASES + co.BGC_ZSAL_SOLVE_CARRY_COUPLED_CASES
ZSAL = "tracers/verticalSalinity"
SICE = "tracers/iceSalinity"
FLUX = "biogeochemistry/zSalinityFlux"
ABORT = "itd_zsal_solve_abort_regrid"
_ITD_NAMES = tuple(row[0] for row in co.BGC_ZSAL_SOLVE_CARRY_CASES if row[2] == "itd_thermodynamics")
_RIDGE_NAMES = tuple(row[0] for row in co.BGC_ZSAL_SOLVE_CARRY_CASES if row[2] == "ridging")


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


def test_every_case_solves_the_vertical_salinity():
    for case in co.cases("r8", ROWS):
        assert cv.solves_zsal(case["config"]), case["name"]


def test_fixture_shows_the_solved_vertical_salinity():
    fixture, logs = co.load_fixture(FIXTURE), co.load_logs(FIXTURE)
    cases = {c["name"]: c for c in co.cases("r8", ROWS)}
    assert set(logs) == set(cases) == set(fixture)
    for name in cases:
        assert logs[name].get("crash") is None, name
    # regrid_stationary's remap stop: add_new_ice_bgc's abort, cell 2 (indexToCellID 3)
    assert logs[ABORT]["log"] == ["ERR column_itd_thermodynamics: ice: remap_layers_bgc error", "ERR iCell: $i",
                                  "INTS 3"]
    for name in cases:
        if name != ABORT:
            assert logs[name]["tracer_object"]["IERR"] == [0], name
            # copied from the tracer arrays and moved
            assert ZSAL in fixture[name], name
    for name in _ITD_NAMES + _RIDGE_NAMES:
        if name != ABORT:
            # the new ice's salt loss, lateral melt and zapped ice reach the salt flux
            assert FLUX in fixture[name], name


@requires_gpu
def test_gpu_matches_the_patched_oracle_word_for_word(glibc239):
    report = co.measure_carry(rows=ROWS, fixture_dir=FIXTURE)
    assert set(report) == {row[0] for row in ROWS}
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


def _mismatches(names):
    report = co.measure_carry(rows=tuple(_row(name) for name in names), fixture_dir=FIXTURE)
    return sum(sum(row.values()) for row in report.values())


@requires_gpu
@pytest.mark.parametrize("names, original, mutated", [
    # add_new_ice's salt loss with the frazil dropped
    (_ITD_NAMES, "if (T.zsal) *fzsal = S(*fzsal, M(M(M(D(M(ITD_RHOSI, vi0new), dt), P001), sss), ITD_SALT_LOSS));",
     ""),
    # regrid_stationary's top concentration without salt_loss
    (_ITD_NAMES, "const double top_conc = M(sss, ITD_SALT_LOSS);", "const double top_conc = sss;"),
    # the frazil mixed through the profile without its salt
    (_ITD_NAMES, "zsal[k - 1] = D(A(M(zsal[k - 1], vtmp), M(M(sss, ITD_SALT_LOSS), vsurp)), vbrin);",
     "zsal[k - 1] = D(M(zsal[k - 1], vtmp), vbrin);"),
    # sice not remapped from the new vertical salinity
    (_ITD_NAMES, "for (int k = 1; k <= COL_NILYR; ++k) tr[T.idx[TI_SICE] + k - 1] = out[k];", ""),
    # the first category's surface temperature left alone
    (_ITD_NAMES, "trcrn[T.idx[TI_TSFC]] = bg_Tin_from_qin(trcrn[T.idx[TI_QICE]], Tmlts);", ""),
    # lateral_melt_bgc on the volume after the melt
    (_ITD_NAMES, "vicen_init[n]), P001), zspace)", "vicen[n]), P001), zspace)"),
    # cleanup_itd without zap_small_areas' salt
    (_ITD_NAMES + _RIDGE_NAMES, "if (T.zsal) *fzsal = A(*fzsal, dfzsal);", ""),
    # regrid_stationary's remap stop not checked
    ((ABORT,), "            int stop = itd_regrid_stationary(T, cs, hbri_old, hbri, top_conc);\n"
               "            if (stop) return stop;\n",
     "            (void)itd_regrid_stationary(T, cs, hbri_old, hbri, top_conc);\n"),
])
def test_mutated_solved_salinity_is_caught(monkeypatch, names, original, mutated):
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


@requires_gpu
def test_a_step_that_does_not_solve_the_vertical_salinity_is_caught(monkeypatch, glibc239):
    monkeypatch.setattr(cv, "solves_zsal", lambda configs: False)
    assert _mismatches(("step_zsal_solve_bl99_dedd",)) > 0

