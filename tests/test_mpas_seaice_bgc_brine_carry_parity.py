"""MPAS-Seaice's brine fraction carried through the ITD, the ridging and the time step, word for word against v8.4.2.

The reference is MPAS-Model v8.4.2 with ``config_use_column_biogeochemistry``
and ``config_use_brine`` (every other biogeochemistry flag off), compiled with
gfortran -O0 -ffp-contract=off at RKIND 8 on an x86-64 glibc 2.44 host:
the ITD thermodynamics (``add_new_ice``'s ``add_new_ice_bgc``, the brine
fraction set to 1 in ``zap_small_areas``), the ridging (the brine fraction a
volume tracer), the column time step (the column biogeochemistry after the
vertical thermodynamics; ``seaice_column_reinitialize_diagnostics_bgc``;
``coupling_prep``'s and ``scale_fluxes``'s ocean biogeochemistry fluxes) and
the coupled time step with the velocity solver.
``tools/mpas_seaice_oracle/run_column.F90`` runs the stages; the outputs are
committed (tests/data/oracles/mpas_seaice_bgc_brine_carry, PROVENANCE.md),
so the GPU comparison runs without gfortran. The step cases use the
delta-Eddington shortwave, so they need the LGPL opt-in
``WOOF_GLIBC239_LIBM64=1`` (set here).

v8.4.2's ``seaice_column_reinitialize_diagnostics_bgc`` zeroes
``totalChlorophyll``, whose packages (the algae's) are off with the brine
height alone: MPAS has no array for it and the write goes to address 0, so
v8.4.2 dies on every step. The step refuses the brine height without the
opt-in correction ``bgc_inactive_diagnostics`` (zero only the fields that
exist); the oracle gives the field an array that is not compared.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with MPAS only to a tolerance: every word of every
   field the stages may write must be bit-identical.
2. A brine fraction the ITD drops: the new ice's brine volume, the surplus
   thickness's, the empty category's 1 and the zapped category's 1 must be
   v8.4.2's.
3. A step that skips the column biogeochemistry, or ocean biogeochemistry
   fluxes that are not zeroed and scaled as coupling_prep and scale_fluxes do.
4. v8.4.2's null-array write running silently, or "corrected" without the
   opt-in: the step refuses it by name.
5. A configuration the port does not carry running silently: each other
   biogeochemistry flag is refused by name in each stage.
6. A gate that cannot fail: mutated brine arithmetic must produce
   mismatches.
7. A case generator whose bytes drift: the input digests are pinned.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu
from woof.hex.errors import ConfigurationRefusal
from woof.hex.mpas_seaice import column_itd_v842 as citd
from woof.hex.mpas_seaice import column_ridge_v842 as crdg
from woof.hex.mpas_seaice import column_step_v842 as cstep
from woof.hex.mpas_seaice import column_v842 as cv
from woof.hex.mpas_seaice import coupling_prep_v842 as cpl
from woof.verify import mpas_seaice_column_oracle as co

FIXTURE = co.BGC_CARRY_FIXTURE_DIR
ROWS = co.BGC_CARRY_CASES + co.BGC_CARRY_COUPLED_CASES
FBRI = "tracers/brineFraction"


def _row(name):
    return next(row for row in ROWS if row[0] == name)


def _case(name):
    return co.make_case(*_row(name))


@pytest.fixture
def glibc239(monkeypatch):
    monkeypatch.setenv("WOOF_GLIBC239_LIBM64", "1")


def test_case_inputs_are_the_recorded_bytes():
    recorded = {}
    for line in (FIXTURE / "inputs.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        recorded[name] = digest
    assert recorded == {f"{case['name']}/r8": co.input_digest(case) for case in co.cases("r8", ROWS)}


def test_fixture_covers_the_branches():
    fixture, logs = co.load_fixture(FIXTURE), co.load_logs(FIXTURE)
    cases = {c["name"]: c for c in co.cases("r8", ROWS)}
    assert set(logs) == set(cases) == set(fixture)
    for name, out in fixture.items():
        for key, value in out.items():
            # (the ridging's thickness ratio is v8.4.2's own, unset where nothing ridges)
            if key == FBRI or key.startswith("biogeochemistry/"):
                assert np.isfinite(value).all(), (name, key)
        assert logs[name].get("crash") is None, name
    n_solve = co.MESH["nCellsSolve"]

    def got(name, key):
        return co.expected_outputs(cases[name], fixture[name])[key][:n_solve]

    def given(name, key):
        pool, field = key.split("/")
        return np.asarray(cases[name]["pools"][pool][field])[:n_solve]

    # every case moves the brine fraction
    for name in cases:
        assert (got(name, FBRI) != given(name, FBRI)).any(), name
    # the ITD: the zapped cell and the new ice's empty category restart at 1
    for name, _, stage, _ in co.BGC_CARRY_CASES:
        if stage == "itd_thermodynamics" and cases[name]["dims"]["nCategories"] > 1:
            assert (got(name, FBRI)[3, :, 0] == 1).all(), name
    assert got("itd_bgc_brine_bl99_delta_function", FBRI)[2, 1, 0] == 1
    # the steps write the brine height's diagnostics, and every ocean biogeochemistry flux
    # starts nonzero and ends zeroed on the owned cells (the aggregate pool's are the same
    # cells in a step; tests/test_mpas_seaice_coupling_prep_parity.py has them apart)
    for name in [r[0] for r in ROWS if r[2] in ("column_step", "seaice_step")]:
        solve = int(cases[name]["pools"]["tracers"]["nCellsSolve"])
        assert solve == int(cases[name]["pools"]["tracers_aggregate"]["nCellsSolve"])
        moved = {key for key in ("biogeochemistry/netBrineHeight", "biogeochemistry/brineTopChange",
                                 "biogeochemistry/brineBottomChange")
                 if key in fixture[name] and (got(name, key) != given(name, key)).any()}
        assert moved, name
        for flux in cpl.BGC_OCEAN_FLUXES:
            start = np.asarray(cases[name]["pools"]["biogeochemistry"][flux])[:solve]
            end = co.expected_outputs(cases[name], fixture[name])[f"biogeochemistry/{flux}"][:solve]
            assert (start != 0).all() and (end == 0).all(), (name, flux)
    assert co.stand_ins(cases["step_bgc_brine_bl99_dedd"]["config"]) == {("biogeochemistry", "totalChlorophyll")}
    assert "biogeochemistry/totalChlorophyll" not in co.expected_outputs(cases["step_bgc_brine_bl99_dedd"],
                                                                         fixture["step_bgc_brine_bl99_dedd"])


def test_the_inactive_diagnostic_is_the_algae_s_chlorophyll():
    assert cstep.inactive_bgc_diagnostics(_case("step_bgc_brine_bl99_dedd")["config"]) == ("totalChlorophyll",)


def test_the_step_refuses_v842_s_null_array_write_without_the_correction():
    config = {k: v for k, v in _case("step_bgc_brine_bl99_dedd")["config"].items() if k != "woof_v842_corrections"}
    with pytest.raises(ConfigurationRefusal, match="config_use_column_biogeochemistry.*totalChlorophyll"):
        cstep.check_config(config)
    cstep.check_config(_case("step_bgc_brine_bl99_dedd")["config"])


def test_the_correction_zeroes_only_the_fields_that_exist():
    case = _case("step_bgc_brine_bl99_dedd")
    bgc = case["pools"]["biogeochemistry"]
    bgc["totalChlorophyll"] = None
    bgc["netBrineHeight"][...] = 7.0
    cstep.reinitialize_diagnostics_bgc(case["pools"], case["config"])
    assert (bgc["netBrineHeight"] == 0).all()
    # word for word, v8.4.2 writes to the array it does not have: refused by name, not a crash
    word_exact = {k: v for k, v in case["config"].items() if k != "woof_v842_corrections"}
    with pytest.raises(ConfigurationRefusal, match="config_use_column_biogeochemistry.*totalChlorophyll"):
        cstep.reinitialize_diagnostics_bgc(case["pools"], word_exact)
    # a tuple naming the correction is the same declaration as the bare name
    assert co.stand_ins({**case["config"], "woof_v842_corrections": ("bgc_inactive_diagnostics",)}) == \
        co.stand_ins(case["config"]) == {("biogeochemistry", "totalChlorophyll")}


@pytest.mark.parametrize("name, refuse", [
    ("itd_bgc_brine_mushy_level_ponds", lambda c: citd.refuse_unported(c["config"], c["dims"])),
    ("ridge_bgc_brine_mushy_level_ponds", lambda c: crdg.refuse_unported(c["config"], c["dims"])),
    ("step_bgc_brine_bl99_dedd", lambda c: cstep.check_config(c["config"])),
    ("step_bgc_brine_bl99_dedd", lambda c: cpl.check_config(c["config"])),
])
def test_the_brine_height_is_carried_and_every_other_flag_refused_by_name(name, refuse):
    case = _case(name)
    refuse(case)
    # (the vertical salinity is carried too: tests/test_mpas_seaice_bgc_zsal_carry_parity.py)
    for flag in (f for f in cv._BGC_FLAGS if f not in cv.BGC_PORTED and f != cv.ZSAL_FLAG):
        other = {**case, "config": {**case["config"], flag: True}}
        with pytest.raises(ConfigurationRefusal, match=flag):
            refuse(other)


@requires_gpu
def test_the_ocean_biogeochemistry_fluxes_are_zeroed_then_scaled():
    import cupy as cp
    n, n_agg = 3, 5
    bgc = {name: cp.full(n_agg, 2.0) for name in cpl.BGC_OCEAN_FLUXES}
    cpl._bgc_ocean_fluxes(bgc, cp.asarray([0.5, 0.0, 0.25, 0.5, 0.0]), n, n_agg)
    for name in cpl.BGC_OCEAN_FLUXES:
        assert cp.asnumpy(bgc[name]).tolist() == [0.0, 0.0, 0.0, 4.0, 0.0], name


@requires_gpu
def test_gpu_matches_the_v842_oracle_word_for_word(glibc239):
    report = co.measure_carry()
    assert set(report) == {row[0] for row in ROWS}
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


def _mismatches(names, field=FBRI):
    report = co.measure_carry(rows=tuple(_row(name) for name in names))
    return sum(row.get(field, 0) + row.get("abort", 0) for row in report.values())


_ITD_NAMES = tuple(row[0] for row in co.BGC_CARRY_CASES if row[2] == "itd_thermodynamics")


@requires_gpu
@pytest.mark.parametrize("original, mutated", [
    # add_new_ice_bgc not called
    ("    if (ix[TI_FBRI] >= 0)\n        return itd_add_new_ice_bgc(", "    if (false)\n        return itd_add_new_ice_bgc("),
    # an empty category under a thickness surplus keeps its brine fraction
    ("            else trcrn[n * nt + fb] = 1.0;\n", "\n"),
    # the new ice's brine volume left out
    ("vbrin[0] = A(vbrin[0], vi0new);", "vbrin[0] = vbrin[0];"),
    # zap_small_areas zeroes the brine fraction like the other tracers
    ("tr[it] = it == ix[TI_FBRI] ? 1.0 : 0.0;", "tr[it] = 0.0;"),
])
def test_mutated_brine_arithmetic_is_caught(monkeypatch, original, mutated):
    source = citd.module_source
    assert source(7).count(original) == 1
    monkeypatch.setattr(citd, "module_source", lambda nil: source(nil).replace(original, mutated))
    citd._module_on.cache_clear()
    try:
        assert _mismatches(_ITD_NAMES) > 0
    finally:
        citd._module_on.cache_clear()


@requires_gpu
def test_a_step_without_the_column_biogeochemistry_is_caught(monkeypatch, glibc239):
    monkeypatch.setattr(cstep.cbgc, "biogeochemistry", lambda pools, configs, state: (0, None, None))
    assert _mismatches(("step_bgc_brine_bl99_dedd",), "biogeochemistry/netBrineHeight") > 0


@requires_gpu
@pytest.mark.parametrize("flux", cpl.BGC_OCEAN_FLUXES)
def test_an_ocean_biogeochemistry_flux_the_step_leaves_alone_is_caught(monkeypatch, glibc239, flux):
    monkeypatch.setattr(cpl, "BGC_OCEAN_FLUXES", tuple(f for f in cpl.BGC_OCEAN_FLUXES if f != flux))
    assert _mismatches(("step_bgc_brine_bl99_dedd",), f"biogeochemistry/{flux}") > 0
