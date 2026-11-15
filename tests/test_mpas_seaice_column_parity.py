"""MPAS-Seaice's column package (colpkg) initialisation and aggregation, word for word against v8.4.2.

The reference is MPAS-Model v8.4.2's ``mpas_seaice_column.F`` and column
package (``src/core_seaice/column``), with ``mpas_seaice_initialize.F``'s and
``mpas_seaice_forcing.F``'s routines cut verbatim, compiled with gfortran -O0
-ffp-contract=off at RKIND 8 (the column package is dbl_kind; the wrappers do
not compile at RKIND 4). ``tools/mpas_seaice_oracle/run_column.F90`` runs
``seaice_init_column_physics_package_parameters`` then ``seaice_init``'s
column steps, ``seaice_column_aggregate`` or ``seaice_column_update_state``;
``run_init_thermo.F90`` writes ``colpkg_init_thermo``'s profile for every
ice-layer count the port accepts. The oracle's outputs are committed
(tests/data/oracles/mpas_seaice_column, PROVENANCE.md), so the GPU comparison
runs without gfortran. On an x86-64 glibc host with the toolchain
(``GFORTRAN``, ``WOOF_MPAS_V842_SOURCE``) the fixture is regenerated and
compared too.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with MPAS only to a tolerance. Every word of every
   field the routines may write, halo and untouched entries included, must
   be bit-identical.
2. A tracer object (indices, parents, ancestor masks and lists) that differs
   from the one v8.4.2 builds, so the aggregate weights a tracer by the wrong
   quantity.
3. A cos, pow or tanh that differs from the oracle host's in one word:
   every ice-layer count the port accepts is graded.
4. A gate that cannot fail. A mutated evaluation order or comparison must
   produce mismatches.
5. A configuration the port does not carry running silently: each is
   refused by name; MPAS's own configuration aborts are raised.
6. A case generator whose bytes drift. The input digests are pinned.
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
from woof.hex.mpas_seaice import column_v842 as cv
from woof.verify import mpas_seaice_column_oracle as co

ROOT = Path(__file__).resolve().parents[1]


def _case(name):
    return co.make_case(*next(row for row in co.CASES if row[0] == name))


def test_case_inputs_are_the_recorded_bytes():
    recorded = {}
    for line in (co.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        recorded[name] = digest
    assert recorded == {f"{case['name']}/r8": co.input_digest(case) for case in co.cases()}


def test_the_host_tracer_object_is_the_oracle_s():
    logs = co.load_logs()
    for case in co.cases():
        state = cv.init_parameters(case["pools"], case["config"])
        assert state.tracers.log_rows() == logs[case["name"]]["tracer_object"], case["name"]
        assert logs[case["name"]]["log"] == [], case["name"]


def test_fixture_covers_the_branches():
    fixture = co.load_fixture()
    cases = {c["name"]: c for c in co.cases()}
    assert set(fixture) == set(cases)
    # BL99 with a fixed surface temperature over fresh water (salinity 0: Tf = -0.0, so the layer temperature is
    # 0 and Tprofile/Ti is infinite) gives v8.4.2's infinite enthalpy; everything else is finite
    infinite = {("init_bl99_fixed_surface_temperature", "tracers/iceEnthalpy"),
                ("init_bl99_fixed_surface_temperature", "tracers_aggregate/iceEnthalpyCell")}
    for name, out in fixture.items():
        for key, value in out.items():
            assert np.isfinite(value).all() or (name, key) in infinite, (name, key)
    assert not np.isfinite(fixture["init_bl99_fixed_surface_temperature"]["tracers/iceEnthalpy"]).all()
    stages = {c["stage"] for c in cases.values()}
    assert stages == {"init", "aggregate", "update_state"}
    configs = [c["config"] for c in cases.values()]
    assert {c["config_thermodynamics_type"] for c in configs} == {"zero layer", "BL99", "mushy"}
    assert {c["config_category_bounds_type"] for c in configs} == {"single category", "original", "new", "WMO",
                                                                     "asymptotic"}
    assert {c["config_itd_conversion_type"] for c in configs} == {"delta function", "linear remap"}
    assert {c["config_initial_condition_type"] for c in configs} == {"cice_default", "uniform", "no_ice"}
    assert {c["config_initial_velocity_type"] for c in configs} == {"uniform", "none"}
    # an ice-free cell takes the ocean freezing point as its surface temperature
    agg = fixture["aggregate_default"]["tracers_aggregate/surfaceTemperatureCell"]
    assert (agg[:cases["aggregate_default"]["dims"]["nCellsSolve"]] == -1.8).sum() >= 2
    # cice_default grows ice in some cells and leaves others ice-free
    area = fixture["init_default"]["tracers/iceAreaCategory"]
    n_solve = cases["init_default"]["dims"]["nCellsSolve"]
    per_cell = area[:n_solve].sum(axis=(1, 2))
    assert (per_cell > 0).any() and (per_cell == 0).any()
    # the ice age's offset skips the cells without ice age
    age = fixture["update_state_transport_age_offset"]["diagnostics/iceAgeTendencyTransport"]
    before = cases["update_state_transport_age_offset"]["pools"]["diagnostics"]["iceAgeTendencyTransport"]
    assert (age[:n_solve] == before[:n_solve]).any() and (age[:n_solve] != before[:n_solve]).any()


def test_the_initial_air_drag_is_gfortran_s_folded_constant():
    drag = co.load_fixture()["init_default"]["drag/airDragCoefficient"]
    assert (drag == cv._folded_constants()["airDragCoefficient"]).all()


@pytest.mark.parametrize("overrides, match", [
    ({"config_use_column_biogeochemistry": True}, "biogeochemistry"),
    ({"config_nSnowLayers": 2}, "snow layer"),
    ({"config_nIceLayers": cv.MAX_ICE_LAYERS + 1}, "cos"),
    ({"config_nIceLayers": 11}, "glibc cos"),
    ({"config_do_restart": True}, "restart"),
    ({"config_initial_condition_type": "circle"}, "initial conditions"),
    ({"config_initial_velocity_type": "ocean"}, "initial velocities"),
    ({"config_use_column_package": False}, "column package"),
    ({"config_use_level_ice": False}, "level-ice tracer"),
])
def test_unported_configurations_are_refused_by_name(overrides, match):
    case = co.make_case("refused", 1, "init", overrides)
    with pytest.raises(ConfigurationRefusal, match=match):
        cv.init_parameters(case["pools"], case["config"])


@pytest.mark.parametrize("overrides, match", [
    ({"config_sea_freezing_temperature_type": "linear_salt"}, "mushy"),
    ({"config_use_cesm_meltponds": True}, "More than one melt pond scheme"),
    ({"config_nCategories": 1}, "nCategories=1"),
    ({"config_category_bounds_type": "single category"}, "single category"),
    ({"config_thermodynamics_type": "BL99"}, "BL99"),
    ({"config_use_form_drag": True, "config_use_level_meltponds": False, "config_use_cesm_meltponds": True,
      "config_pond_refreezing_type": "cesm"}, "form_drag"),
])
def test_mpas_configuration_aborts_are_raised(overrides, match):
    configs = co.configs_of(overrides, "init")
    with pytest.raises(ValueError, match=match):
        cv.check_column_package_configs(configs, configs["config_nCategories"])


@requires_gpu
def test_gpu_matches_the_v842_oracle_word_for_word():
    report = co.measure()
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
def test_every_accepted_ice_layer_count_s_profile_is_the_oracle_s():
    # the refused counts are exactly the ones glibc's cos rounds differently
    bad = {n: v for n, v in co.thermo_profile_mismatches(co.load_thermo_profiles()).items() if v}
    assert set(bad) == cv.COS_REFUSED_LAYERS, bad


@requires_gpu
def test_the_wmo_bounds_need_five_to_seven_categories():
    case = _case("init_bounds_wmo6")
    configs = {**case["config"], "config_nCategories": 4}
    dev = co.on_device(case)
    state = cv.init_parameters(dev, case["config"])
    state.dims["nCategories"] = 4
    import cupy as cp
    dev["initial"]["categoryThicknessLimits"] = cp.zeros(5, np.float64)
    with pytest.raises(ValueError, match="WMO"):
        cv.init_variables(dev, configs, state)


@requires_gpu
@pytest.mark.parametrize("original, mutated, field, case", [
    # an area tracer's mean taken through the reciprocal of the area
    ("trcr[it] = D(atrcr[it], aice);", "trcr[it] = M(atrcr[it], D(1.0, aice));",
     "tracers_aggregate/surfaceTemperatureCell", "aggregate_default"),
    # the ice-free test made strict at zero (a cell of sub-puny area)
    ("if (aice > PUNY) {", "if (aice > 0.0) {", "tracers_aggregate/surfaceTemperatureCell", "aggregate_default"),
    # the mushy enthalpy's latent term regrouped
    ("M(M(S(1.0, phi), RHOI), lfresh())", "M(S(1.0, phi), M(RHOI, lfresh()))",
     "tracers/iceEnthalpy", "init_default"),
    # the salinity profile without its floor (min_salin binds near the surface of 16 layers)
    ("sprofile[k - 1] = col_max(s, MIN_SALIN);", "sprofile[k - 1] = col_max(s, 0.0);",
     "initial/initialSalinityProfile", "init_bl99_sixteen_layers"),
    # the category areas normalised without puny
    ("D(categoryArea[n], A(sum, D(PUNY, (double)nCategories)))", "D(categoryArea[n], sum)",
     "tracers/iceAreaCategory", "init_default"),
    # the ice-age offset applied where the ice age is zero
    ("if (ageCell[c] > 0.0)", "if (ageCell[c] >= 0.0)",
     "diagnostics/iceAgeTendencyTransport", "update_state_transport_age_offset"),
])
def test_a_mutated_evaluation_order_is_caught(monkeypatch, original, mutated, field, case):
    source = cv.module_source()
    assert source.count(original) == 1
    monkeypatch.setattr(cv, "module_source", lambda: source.replace(original, mutated))
    cv._module_on.cache_clear()
    try:
        got, _ = co.run_gpu(_case(case))
        want = co.load_fixture()[case][field]
        assert co.word_mismatches(got[field], want) > 0
    finally:
        cv._module_on.cache_clear()


def _oracle_toolchain() -> tuple[str, Path] | None:
    source = os.environ.get("WOOF_MPAS_V842_SOURCE")
    gfortran = os.environ.get("GFORTRAN") or shutil.which("gfortran")
    if not source or not gfortran or not Path(source, "src", "core_seaice").is_dir():
        return None
    return gfortran, Path(source)


@pytest.mark.skipif(_oracle_toolchain() is None or platform.machine() != "x86_64"
                    or platform.libc_ver() != ("glibc", "2.44"),
                    reason="the column oracle's words are the x86-64 glibc 2.44 host's libm (cos, pow, tanh; "
                           "the fixture's, PROVENANCE.md); it needs gfortran (pixi -e mpas-oracle) and "
                           "WOOF_MPAS_V842_SOURCE")
def test_the_oracle_rebuilt_here_writes_the_fixture(tmp_path):
    gfortran, source = _oracle_toolchain()
    subprocess.run(["bash", str(ROOT / "tools" / "mpas_seaice_oracle" / "build.sh"), str(source),
                    str(tmp_path / "build")], check=True, env={**os.environ, "GFORTRAN": gfortran})
    subprocess.run([sys.executable, str(ROOT / "tools" / "mpas_seaice_oracle" / "make_column_fixture.py"),
                    str(tmp_path / "build"), "--out", str(tmp_path / "fixture")], check=True)
    fresh = np.load(tmp_path / "fixture" / "oracle-r8.npz")
    pinned = np.load(co.FIXTURE_DIR / "oracle-r8.npz")
    assert sorted(fresh.files) == sorted(pinned.files)
    for key in pinned.files:
        assert co.word_mismatches(fresh[key], pinned[key]) == 0, key
    assert co.word_mismatches(np.load(tmp_path / "fixture" / "init-thermo-r8.npy"), co.load_thermo_profiles()) == 0
    for name in ("column-logs.json", "inputs.sha256"):
        assert (tmp_path / "fixture" / name).read_text() == (co.FIXTURE_DIR / name).read_text()
