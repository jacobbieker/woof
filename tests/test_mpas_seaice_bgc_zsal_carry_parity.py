"""MPAS-Seaice's vertical salinity carried through the ITD, the ridging and the time step, word for word against v8.4.2.

The reference is MPAS-Model v8.4.2 with ``config_use_column_biogeochemistry``,
``config_use_brine`` and ``config_use_vertical_zsalinity`` (BL99; every other
biogeochemistry flag off), compiled with gfortran -O0 -ffp-contract=off at
RKIND 8 on an x86-64 glibc 2.44 host, unpatched. v8.4.2 never sets the column
package's ``solve_zsal`` (ice_colpkg.F90:4325-4345) and copies
``verticalSalinity`` to and from the tracer arrays only with the vertical
tracers (mpas_seaice_column.F:6292,6664), so its vertical salinity is inert:
the tracers are counted, the ITD's and the ridging's ``cleanup_itd`` add a
zero ``dfzsal`` to ``zSalinityFlux``, the step's
``seaice_column_reinitialize_diagnostics_bgc`` zeroes the salinity fluxes,
and with the velocity solver the incremental remap advects the pool's
``verticalSalinity`` (pkgTracerZSalinityActive, its parent the brine
fraction). The solved vertical salinity (the ``zsal_solve`` correction) is
refused by these stages. ``tools/mpas_seaice_oracle/run_column.F90`` runs
the stages; the outputs are committed
(tests/data/oracles/mpas_seaice_bgc_zsal_carry, PROVENANCE.md), so the GPU
comparison runs without gfortran. The step cases use the delta-Eddington
shortwave, so they need the LGPL opt-in ``WOOF_GLIBC239_LIBM64=1`` (set here).

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with MPAS only to a tolerance: every word of every
   field the stages may write must be bit-identical.
2. A port that "solves" or copies the inert vertical salinity: the pool's
   verticalSalinity must keep its words through the ITD, the ridging and the
   column step.
3. A time step whose remap drops the vertical salinity, or a step that does
   not zero the salinity fluxes.
4. The solved vertical salinity running in a stage that does not carry it:
   refused by name.
5. A case generator whose bytes drift: the input digests are pinned.
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
from woof.hex.mpas_seaice import remap_v842
from woof.verify import mpas_seaice_column_oracle as co

FIXTURE = co.BGC_ZSAL_CARRY_FIXTURE_DIR
ROWS = co.BGC_ZSAL_CARRY_CASES + co.BGC_ZSAL_CARRY_COUPLED_CASES
ZSAL = "tracers/verticalSalinity"
FBRI = "tracers/brineFraction"
COLUMN = tuple(row[0] for row in co.BGC_ZSAL_CARRY_CASES)


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


def test_fixture_shows_v842_s_inert_vertical_salinity():
    fixture, logs = co.load_fixture(FIXTURE), co.load_logs(FIXTURE)
    cases = {c["name"]: c for c in co.cases("r8", ROWS)}
    assert set(logs) == set(cases) == set(fixture)
    for name in cases:
        assert logs[name].get("crash") is None, name
    n_solve = co.MESH["nCellsSolve"]

    def got(name, key):
        return co.expected_outputs(cases[name], fixture[name])[key][:n_solve]

    def given(name, key):
        pool, field = key.split("/")
        return np.asarray(cases[name]["pools"][pool][field])[:n_solve]

    for name in cases:
        # the brine fraction still moves with the vertical salinity counted
        assert (got(name, FBRI) != given(name, FBRI)).any(), name
    for name in COLUMN:
        # never copied from the tracer arrays: the pool keeps its words
        assert ZSAL not in fixture[name], name
    for name, _, stage, _ in co.BGC_ZSAL_CARRY_CASES:
        if stage in ("itd_thermodynamics", "ridging"):
            # cleanup_itd's fzsal + dfzsal with dfzsal = 0
            assert (got(name, "biogeochemistry/zSalinityFlux") == given(name, "biogeochemistry/zSalinityFlux")).all()
    steps = [row[0] for row in ROWS if row[2] in ("column_step", "seaice_step")]
    for name in steps:
        for key in ("biogeochemistry/zSalinityFlux", "biogeochemistry/zSalinityGDFlux"):
            assert (given(name, key) != 0).all() and (got(name, key) == 0).all(), (name, key)
    # the remap advects the pool's vertical salinity
    coupled = co.BGC_ZSAL_CARRY_COUPLED_CASES[0][0]
    assert (got(coupled, ZSAL) != given(coupled, ZSAL)).any()


@pytest.mark.parametrize("name, refuse", [
    ("itd_bgc_zsal_bl99_update_ocean", lambda c: citd.refuse_unported(c["config"], c["dims"])),
    ("ridge_bgc_zsal_bl99", lambda c: crdg.refuse_unported(c["config"], c["dims"])),
    # the column step's and the time step's init (init_parameters with the biogeochemistry)
    ("step_bgc_zsal_bl99_dedd",
     lambda c: cv.refuse_unported(c["config"], {**c["dims"], "nSnowLayers": 1}, biogeochemistry=True)),
])
def test_the_inert_vertical_salinity_is_carried_and_the_solved_one_refused(name, refuse):
    case = _case(name)
    refuse(case)
    solved = {**case, "config": {**case["config"], "woof_v842_corrections": (
        *cv.corrections(case["config"]), cv.ZSAL_SOLVE)}}
    with pytest.raises(ConfigurationRefusal, match=cv.ZSAL_SOLVE):
        refuse(solved)


def test_the_step_and_coupling_prep_take_the_vertical_salinity():
    config = _case("step_bgc_zsal_bl99_dedd")["config"]
    cstep.check_config(config)
    # colpkg_init_zbgc leaves the vertical salinity out of nbtrcr: no ocean flux of its own
    cpl.check_config(config)


@requires_gpu
def test_gpu_matches_the_v842_oracle_word_for_word(glibc239):
    report = co.measure_carry(rows=ROWS, fixture_dir=FIXTURE)
    assert set(report) == {row[0] for row in ROWS}
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


def _mismatches(names, field):
    report = co.measure_carry(rows=tuple(_row(name) for name in names), fixture_dir=FIXTURE)
    return sum(row.get(field, 0) + row.get("abort", 0) for row in report.values())


@requires_gpu
def test_a_port_that_copies_the_inert_vertical_salinity_is_caught(monkeypatch, glibc239):
    original = cv.tracer_object

    def copying(configs, dims):
        to = original(configs, dims)
        to.layout.append(("verticalSalinity", to.nt_bgc_s, dims["nBioLayers"]))
        return to

    monkeypatch.setattr(cv, "tracer_object", copying)
    assert _mismatches(("itd_bgc_zsal_bl99_update_ocean", "ridge_bgc_zsal_bl99"), ZSAL) > 0


@requires_gpu
def test_a_step_that_keeps_the_salinity_flux_is_caught(monkeypatch, glibc239):
    monkeypatch.setattr(cstep, "_BGC_FIELDS", tuple(f for f in cstep._BGC_FIELDS if f != "zSalinityFlux"))
    assert _mismatches(("step_bgc_zsal_bl99_dedd",), "biogeochemistry/zSalinityFlux") > 0


@requires_gpu
def test_a_remap_without_the_vertical_salinity_is_caught(monkeypatch, glibc239):
    monkeypatch.setattr(remap_v842, "TRACER_TABLE",
                        tuple(row for row in remap_v842.TRACER_TABLE if row[0] != "verticalSalinity"))
    assert _mismatches((co.BGC_ZSAL_CARRY_COUPLED_CASES[0][0],), ZSAL) > 0
