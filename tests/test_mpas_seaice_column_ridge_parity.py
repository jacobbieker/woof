"""MPAS-Seaice's column ridging (colpkg stage 4), word for word against v8.4.2.

The reference is MPAS-Model v8.4.2's ``column_ridging``
(``mpas_seaice_column.F``) and the column package under it
(``colpkg_step_ridge``: ``ridge_ice`` with ``ridge_prep``, ``ridge_itd`` and
``ridge_shift``, then ``cleanup_itd``), compiled with gfortran -O0
-ffp-contract=off at RKIND 8. ``tools/mpas_seaice_oracle/run_column.F90``
(stage ``ridging``) runs ``colpkg_init_thermo`` and ``init_column_itd`` and
then the routine; its ierr, abort message and warnings are logged. The
oracle's outputs are committed (tests/data/oracles/mpas_seaice_column_ridge,
PROVENANCE.md), so the GPU comparison runs without gfortran. The routine's
exp is glibc's, which the GPU transcribes under MIT (glibc_flt64.cuh), so the
default build runs it.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with MPAS only to a tolerance: every word of every
   field the routine may write must be bit-identical, an aborting case's
   included (the cells after the abort keep their words).
2. v8.4.2's quirks "fixed": ridge_prep's hin_max(ncat) = 1e8 lands in
   categoryThicknessLimits; the redistribution diagnostics are written only
   for the thinnest category's ridges under the exponential redistribution;
   a ridging abort leaves the categories already ridged changed.
3. An abort that the port reports differently: the first aborting owned
   cell, its ``indexToCellID`` and its message must be MPAS's, and the
   ``Repeat ridging`` and zap_snow_temperature warnings the lines MPAS logs.
4. A gate that cannot fail: a mutated evaluation order must produce
   mismatches.
5. A configuration the port does not carry running silently: each is
   refused by name.
6. A case generator whose bytes drift: the input digests are pinned.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from woof.hex.errors import ConfigurationRefusal
from woof.hex.mpas_seaice import column_itd_v842 as citd
from woof.hex.mpas_seaice import column_ridge_v842 as crdg
from woof.verify import mpas_seaice_column_oracle as co

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = co.RIDGE_FIXTURE_DIR
ROUTINE = "column_ridging"


def _case(name):
    return co.make_case(*next(row for row in co.RIDGE_CASES if row[0] == name))


@pytest.fixture
def fresh_module():
    crdg._module_on.cache_clear()
    yield
    crdg._module_on.cache_clear()


def test_case_inputs_are_the_recorded_bytes():
    recorded = {}
    for line in (FIXTURE / "inputs.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        recorded[name] = digest
    assert recorded == {f"{case['name']}/r8": co.input_digest(case) for case in co.cases("r8", co.RIDGE_CASES)}


def test_fixture_covers_the_branches():
    fixture, logs = co.load_fixture(FIXTURE), co.load_logs(FIXTURE)
    cases = {c["name"]: c for c in co.cases("r8", co.RIDGE_CASES)}
    assert set(logs) == set(cases)
    outcomes = {name: co.oracle_abort(log, ROUTINE) for name, log in logs.items()}
    # the first aborting owned cell (indexToCellID), before a later one MPAS never reaches
    assert outcomes.pop("ridge_negative_opening_dies") == (crdg.SEAICE_ERROR_COL_RIDGING, 4,
                                                      f"{ROUTINE}: {crdg.STOP_LABELS[2]}")
    assert outcomes.pop("ridge_abort_iterations") == (crdg.SEAICE_ERROR_COL_RIDGING, 6,
                                                      f"{ROUTINE}: {crdg.STOP_LABELS[1]}")
    assert outcomes.pop("ridge_abort_cleanup") == (crdg.SEAICE_ERROR_COL_RIDGING, 8,
                                                   f"{ROUTINE}: {crdg.STOP_LABELS[5]}")
    assert set(outcomes.values()) == {(0, None, None)}
    for name, out in fixture.items():
        for key, value in out.items():
            if key != "ridging/ratioRidgeThicknessToIce":       # ice without volume: krdg = hrmean/0
                assert np.isfinite(value).all(), (name, key)
    n_solve = co.MESH["nCellsSolve"]

    def got(name, key):
        return fixture[name][key]

    plain = "ridge_mushy_level_ponds"
    warnings = co.oracle_ridge_warnings(logs[plain])
    # ridge_ice repeats where the closing was cut back; zap_snow_temperature's warning (row 4)
    assert any(line.startswith("WARN  Repeat ridging") for line in warnings)
    assert citd.ZAP_WARNING in warnings
    assert len([line for line in co.oracle_ridge_warnings(logs["ridge_abort_iterations"])
                if line.startswith("WARN  Repeat ridging")]) >= 20
    # ridge_prep's hin_max(ncat) = 1e8 reaches categoryThicknessLimits
    for name in cases:
        assert got(name, "initial/categoryThicknessLimits")[-1] == crdg.BIG
    # ridging happened: area lost where the forcing closes (row 0), none without forcing (row 7)
    assert got(plain, "ridging/areaLossRidge")[0] > 0
    assert got(plain, "ridging/areaLossRidge")[7] == 0
    # divergence opens water (row 3)
    assert got(plain, "ridging/openingRateRidge")[3] > 0
    # the redistribution diagnostics: written under the exponential redistribution, kept under Hibler's
    assert "ridging/fractionNewRidgeArea" in fixture[plain]
    assert "ridging/fractionNewRidgeArea" not in fixture["ridge_bl99_thorndike_hibler"]
    # rafting under the exponential redistribution only
    assert "ridging/raftingIceArea" in fixture[plain]
    # the pond water lost with the ridging ice: topo ponds only
    assert "ponds/pondFreshWaterFlux" in fixture["ridge_exponential_hibler_topo_ponds"]
    assert "ponds/pondFreshWaterFlux" not in fixture["ridge_thorndike_exponential_cesm_ponds"]
    # an abort leaves the later cells as they were
    for name, first in (("ridge_negative_opening_dies", 4), ("ridge_abort_iterations", 6), ("ridge_abort_cleanup", 8)):
        start = cases[name]["pools"]["tracers"]["iceAreaCategory"]
        assert (got(name, "tracers/iceAreaCategory")[first:n_solve] == start[first:n_solve]).all()


@pytest.mark.parametrize("overrides, match", [
    ({"config_thermodynamics_type": "zero layer"}, "zerolayer_check"),
    ({"config_use_aerosols": True}, "aerosols"),
    ({"config_use_column_biogeochemistry": True}, "biogeochemistry"),
])
def test_unported_configurations_are_refused_by_name(overrides, match):
    case = _case("ridge_mushy_level_ponds")
    with pytest.raises(ConfigurationRefusal, match=match):
        crdg.refuse_unported({**case["config"], **overrides}, case["dims"])


def test_more_categories_than_a_thread_holds_are_refused():
    case = _case("ridge_mushy_level_ponds")
    with pytest.raises(ConfigurationRefusal, match="thickness categories"):
        crdg.refuse_unported(case["config"], {**case["dims"], "nCategories": citd.ITD_MAXCAT + 1})


@requires_gpu
def test_gpu_matches_the_v842_oracle_word_for_word(fresh_module):
    report = co.measure_ridge()
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad
