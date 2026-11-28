"""MPAS-Seaice's time step with the column package and the velocity solver, word for word against v8.4.2.

The reference is MPAS-Model v8.4.2's ``seaice_timestep``
(``mpas_seaice_time_integration.F``) with ``config_use_column_package``: the
time diagnostics, the re-initialised thermodynamic and bgc diagnostics,
``seaice_column_predynamics_time_integration``, per dynamics subcycle the
re-initialised dynamic diagnostics, ``seaice_run_velocity_solver`` (its ice
strength ``colpkg_ice_strength``, Rothrock75's or Hibler79's) and
``seaice_column_dynamics_time_integration``, then
``seaice_column_postdynamics_time_integration``, with
``seaice_timestep_finalize`` (``seaice_column_reinitialize_fluxes``) after
each step. They are compiled with gfortran -O0 -ffp-contract=off at RKIND 8
on an x86-64 glibc 2.44 host. ``tools/mpas_seaice_oracle/run_column.F90``
(stage ``seaice_step``) runs them ``oracle_n_steps`` times on a culled
hexagonal torus; the committed outputs
(tests/data/oracles/mpas_seaice_coupled_step, PROVENANCE.md) let the GPU
comparison run without gfortran. The BL99 vertical thermodynamics, the
delta-Eddington shortwave and Hibler79's strength call glibc's binary64
libm, so the GPU comparison runs under the opt-in transcription
(``WOOF_GLIBC239_LIBM64=1``, source checkouts only).

THE BREAKAGES THESE PREVENT
---------------------------
1. A coupled step that agrees with MPAS only to a tolerance: every word of
   every field the step may write must be bit-identical after the last
   step, an aborting case's included.
2. The velocity solver's ice strength not the column package's: Rothrock75's
   ridging-energy strength (Cf = 17, the participation and redistribution
   functions) and Hibler79's must be colpkg_ice_strength's.
3. The stages in another order, or the finalisation dropped or run after the
   coupler's last step.
4. An abort that the step reports differently, or a step that keeps going
   after one: ``ColumnAbort`` carries the step, the ierr, the first aborting
   cell's ``indexToCellID`` and MPAS's message.
5. A configuration the coupled step does not carry running silently: each is
   refused by name.
6. A case generator whose bytes drift: the input digests are pinned.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
import _oracle_toolchains
from woof.hex.errors import ConfigurationRefusal
from woof.hex.mpas_seaice import column_radiation_v842 as crad
from woof.hex.mpas_seaice import column_ridge_v842 as crdg
from woof.hex.mpas_seaice import column_step_v842 as cstep
from woof.hex.mpas_seaice import column_therm_v842 as ct
from woof.hex.mpas_seaice import timestep_v842 as ts
from woof.verify import mpas_seaice_column_oracle as co

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = co.COUPLED_FIXTURE_DIR
ABORT = "seaice_bl99_abort_salinity"


def _case(name):
    return co.make_case(*next(row for row in co.COUPLED_CASES if row[0] == name))


@pytest.fixture
def glibc239(monkeypatch):
    """The opt-in glibc 2.39 binary64 transcriptions (LGPL, source checkout only)."""
    from woof.core import glibc239_gpu
    if not glibc239_gpu.header_path().is_file():
        pytest.skip("the opt-in glibc 2.39 transcription is only in a source checkout")
    monkeypatch.setenv("WOOF_GLIBC239_LIBM64", "1")
    caches = [ct._module_on, crad._module_on]
    for cache in caches:
        cache.cache_clear()
    yield
    for cache in caches:
        cache.cache_clear()


def test_case_inputs_are_the_recorded_bytes():
    recorded = dict(reversed(line.split()) for line in
                    (FIXTURE / "inputs.sha256").read_text(encoding="utf-8").splitlines())
    assert recorded == {f"{case['name']}/r8": co.input_digest(case) for case in co.cases("r8", co.COUPLED_CASES)}


def test_fixture_covers_the_branches():
    fixture, logs = co.load_fixture(FIXTURE), co.load_logs(FIXTURE)
    cases = {c["name"]: c for c in co.cases("r8", co.COUPLED_CASES)}
    assert set(logs) == set(cases)
    outcomes = {name: co.oracle_step_abort(log) for name, log in logs.items()}
    ierr, _, message = outcomes.pop(ABORT)
    assert ierr == ct.SEAICE_ERROR_COL_VERT_THERM and message.startswith("column_vertical_thermodynamics")
    assert logs[ABORT]["tracer_object"]["STEP"][0] == 1
    assert set(outcomes.values()) == {(0, None, None)}
    for name, out in fixture.items():
        for key, value in out.items():
            if key != "ridging/ratioRidgeThicknessToIce":
                assert np.isfinite(value).all(), (name, key)
    for name, case in cases.items():
        if name == ABORT:
            continue
        out, n_solve = fixture[name], case["pools"]["mesh"]["nCellsSolve"]
        # the velocity solver ran on the column package's strength and moved the ice
        assert (out["velocity_solver/icePressure"][:n_solve] > 0).any(), name
        assert np.count_nonzero(out["velocity_solver/uVelocity"]) > 0, name
        # the ridging saw the solver's convergence
        assert "ridging/areaLossRidge" in out, name
    finalised = fixture["seaice_bl99_dedd_rothrock"]
    kept = fixture["seaice_bl99_dedd_last_step_unfinalized"]
    n_solve = cases["seaice_bl99_dedd_rothrock"]["pools"]["mesh"]["nCellsSolve"]
    assert (finalised["ocean_fluxes/oceanFreshWaterFlux"][:n_solve] == 0).all()
    n_solve = cases["seaice_bl99_dedd_last_step_unfinalized"]["pools"]["mesh"]["nCellsSolve"]
    assert (kept["ocean_fluxes/oceanFreshWaterFlux"][:n_solve] != 0).any()
    quiet = fixture["seaice_mushy_level_ponds_no_vertical_thermodynamics"]
    assert (quiet["melt_growth_rates/congelation"] == 0).all()


@pytest.mark.parametrize("overrides, knob", [
    ({"config_ratio_ridging_work_to_PE": 15.0}, "config_ratio_ridging_work_to_PE"),
    ({"config_use_column_biogeochemistry": True}, "config_use_column_biogeochemistry"),
    ({"config_use_ocean_mixed_layer": True, "config_atmos_boundary_method": "constant"}, "atmo_boundary_const"),
    ({"config_use_vertical_tracers": True}, "config_use_vertical_tracers"),
    ({"config_stress_divergence_scheme": "weak"}, "config_stress_divergence_scheme"),
])
def test_unported_configurations_are_refused_by_name(overrides, knob):
    config = _case("seaice_bl99_dedd_rothrock")["config"]
    ts.check_config(config)
    with pytest.raises(ConfigurationRefusal, match=knob):
        ts.check_config({**config, **overrides})


def test_the_column_package_at_rkind_4_is_refused():
    pools = {"velocity_solver": {"uVelocity": np.zeros(3, np.float32)}}
    with pytest.raises(ConfigurationRefusal, match="RKIND 8"):
        ts._column_precision(pools)


_MISSING = object()


@pytest.mark.parametrize("knob", crdg.NAMED_OPTIONS)
@pytest.mark.parametrize("value", ["Hibler1979", 2, _MISSING])
def test_a_strength_or_ridging_name_v842_does_not_have_is_refused_up_front(knob, value):
    """Before predynamics touches a pool: at check_config, not in the solver."""
    config = dict(_case("seaice_bl99_dedd_rothrock")["config"])
    if value is _MISSING:
        del config[knob]
    else:
        config[knob] = value
    with pytest.raises(ConfigurationRefusal, match=knob):
        ts.check_config(config)


@pytest.mark.parametrize("knob", ["config_ice_strength_formulation", "config_ridging_redistribution_function"])
def test_the_strength_kernel_never_sees_an_unnamed_option(knob):
    """The kernel's redistribution branch has no third case: ice_strength (and
    the ridging) refuse before any pool is read, as check_config does."""
    config = {**_case("seaice_bl99_dedd_rothrock")["config"], knob: "Hibler1979"}
    with pytest.raises(ConfigurationRefusal, match=knob):
        crdg.ice_strength({}, config, 7)
    with pytest.raises(ConfigurationRefusal, match=knob):
        crdg.ridging({}, config, None)


@pytest.mark.parametrize("param, written", [
    ("double *__restrict__ x", True), ("const double *__restrict__ x", False), ("double const *x", False),
    ("double *__restrict__ const x", True), ("int n", False), ("const int *mask", False),
])
def test_the_abort_snapshot_reads_which_parameters_the_kernel_writes(param, written):
    assert ct._is_written(param) is written


def test_the_abort_names_its_step_cell_and_message():
    abort = ts.ColumnAbort(2, 3, 17, "column_vertical_thermodynamics: ice: Vertical thermo error")
    assert (abort.step, abort.ierr, abort.cell) == (2, 3, 17)
    assert "17" in str(abort) and "column_vertical_thermodynamics" in str(abort)


@requires_gpu
def test_gpu_matches_the_v842_oracle_word_for_word_over_the_steps(glibc239):
    report = co.measure_coupled()
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
def test_the_abort_raises_column_abort(glibc239):
    from woof.verify.mpas_seaice_column_oracle import on_device
    case = _case(ABORT)
    dev, config = on_device(case), case["config"]
    dev["diagnostics"]["xtime"] = config["diagnostics:xtime"]
    dev["diagnostics"]["simulationStartTime"] = config["diagnostics:simulationStartTime"]
    clock = ts.initialize(pools=dev, configs=config, basis="input")
    with pytest.raises(ts.ColumnAbort) as raised:
        ts.advance(pools=dev, configs=config, clock=clock, n_steps=int(config["oracle_n_steps"]))
    log = co.load_logs(FIXTURE)[ABORT]
    assert (raised.value.ierr, raised.value.cell, raised.value.message) == co.oracle_step_abort(log)
    assert raised.value.step == log["tracer_object"]["STEP"][0]


def _no_finalize(monkeypatch, case):
    monkeypatch.setattr(cstep, "reinitialize_fluxes", lambda pools, configs: None)
    return case


def _no_dynamics_reinit(monkeypatch, case):
    monkeypatch.setattr(cstep, "reinitialize_diagnostics_dynamics", lambda pools, configs: None)
    return case


def _hibler_strength(monkeypatch, case):
    real = crdg.ice_strength
    monkeypatch.setattr(crdg, "ice_strength", lambda pools, configs, n: real(
        pools, {**configs, "config_ice_strength_formulation": "Hibler79"}, n))
    return case


def _non_column_strength(monkeypatch, case):
    from woof.hex.mpas_seaice import presubcycle_v842 as pre
    real = pre.pre_subcycle

    def drop(*args, column_ice_strength=None, **kwargs):
        kwargs["use_column_vertical_thermodynamics"] = False
        return real(*args, **kwargs)
    monkeypatch.setattr(pre, "pre_subcycle", drop)
    return case


def _finalize_last(monkeypatch, case):
    case["config"] = {**case["config"], "oracle_finalize_last": True}
    return case


@requires_gpu
@pytest.mark.parametrize("mutate, field, name", [
    # seaice_timestep_finalize skipped between the steps: the fluxes accumulate
    (_no_finalize, "atmos_fluxes/sensibleHeatFlux", "seaice_bl99_dedd_rothrock"),
    # the dynamic diagnostics not re-initialised before each subcycle
    (_no_dynamics_reinit, "ridging/areaLossRidge", "seaice_bl99_dedd_two_subcycles_exponential_gregorian"),
    # Hibler79's strength where Rothrock75's is configured
    (_hibler_strength, "velocity_solver/icePressure", "seaice_bl99_dedd_rothrock"),
    # the velocity solver's own (non-column) ice strength
    (_non_column_strength, "velocity_solver/icePressure", "seaice_bl99_dedd_rothrock"),
    # the coupler's last step finalised: its fluxes are gone
    (_finalize_last, "ocean_fluxes/oceanFreshWaterFlux", "seaice_bl99_dedd_last_step_unfinalized"),
])
def test_a_mutated_step_is_caught(monkeypatch, glibc239, mutate, field, name):
    case = mutate(monkeypatch, _case(name))
    got, _ = co.run_gpu(case)
    want = co.load_fixture(FIXTURE)[name][field]
    assert co.word_mismatches(got[field], want) > 0


def _oracle_toolchain() -> tuple[str, Path] | None:
    return _oracle_toolchains.mpas_v842("core_seaice")


@pytest.mark.skipif(_oracle_toolchain() is None or platform.machine() != "x86_64"
                    or platform.libc_ver() != ("glibc", "2.44"),
                    reason="the fixture is the x86-64 glibc 2.44 host's (its atan, exp, log, pow, sin, cos and "
                           "asin, PROVENANCE.md); it needs gfortran (pixi -e mpas-oracle) and WOOF_MPAS_V842_SOURCE")
def test_the_oracle_rebuilt_here_writes_the_fixture(tmp_path):
    gfortran, source = _oracle_toolchain()
    subprocess.run(["bash", str(ROOT / "tools" / "mpas_seaice_oracle" / "build.sh"), str(source),
                    str(tmp_path / "build")], check=True, env={**os.environ, "GFORTRAN": gfortran})
    subprocess.run([sys.executable, str(ROOT / "tools" / "mpas_seaice_oracle" / "make_column_fixture.py"),
                    str(tmp_path / "build"), "--cases", "coupled", "--out", str(tmp_path / "fixture")], check=True)
    fresh = np.load(tmp_path / "fixture" / "oracle-r8.npz")
    pinned = np.load(FIXTURE / "oracle-r8.npz")
    assert sorted(fresh.files) == sorted(pinned.files)
    for key in pinned.files:
        assert co.word_mismatches(fresh[key], pinned[key]) == 0, key
    for name in ("column-logs.json", "inputs.sha256"):
        assert (tmp_path / "fixture" / name).read_text() == (FIXTURE / name).read_text()
