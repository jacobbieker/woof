"""MPAS-Seaice's column-package time step, word for word against v8.4.2 over several steps.

The reference is MPAS-Model v8.4.2's column part of ``seaice_timestep``
(``mpas_seaice_time_integration.F``): the re-initialised thermodynamic and
bgc diagnostics, ``seaice_column_predynamics_time_integration``, each
dynamics subcycle's re-initialised dynamic diagnostics and
``seaice_column_dynamics_time_integration``, then
``seaice_column_postdynamics_time_integration``, with
``seaice_timestep_finalize``'s ``seaice_column_reinitialize_fluxes`` between
the steps. They are compiled with gfortran -O0 -ffp-contract=off at RKIND 8
on an x86-64 glibc 2.44 host. ``tools/mpas_seaice_oracle/run_column.F90``
(stage ``column_step``) runs them ``oracle_n_steps`` times on the clock; the
committed outputs (tests/data/oracles/mpas_seaice_column_step,
PROVENANCE.md) let the GPU comparison run without gfortran. The BL99
vertical thermodynamics and the delta-Eddington shortwave call glibc's
binary64 atan, sin, cos and asin, so the GPU comparison runs under the
opt-in transcription (``WOOF_GLIBC239_LIBM64=1``, source checkouts only).

THE BREAKAGES THESE PREVENT
---------------------------
1. A step that agrees with MPAS only to a tolerance: every word of every
   field the column step may write must be bit-identical after the last
   step, an aborting case's included.
2. Stages run in another order, or the re-initialisations dropped or
   misplaced: the fluxes finalised between the steps, the dynamic
   diagnostics before each subcycle, the drag coefficients reset each step.
3. A stage read on the wrong clock: the day of the year and the seconds into
   the day are the advanced clock's on ``config_calendar_type``'s calendar.
4. An abort that the step reports differently, or a step that keeps going
   after one: the first aborting cell, its ``indexToCellID`` and its message
   must be MPAS's, and the cells after it keep their words.
5. A configuration the step does not carry running silently: each is
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
from woof.hex.mpas_seaice import column_step_v842 as cstep
from woof.hex.mpas_seaice import column_therm_v842 as ct
from woof.hex.mpas_seaice import column_v842 as cv
from woof.verify import mpas_seaice_column_oracle as co

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = co.STEP_FIXTURE_DIR


def _case(name):
    return co.make_case(*next(row for row in co.STEP_CASES if row[0] == name))


@pytest.fixture
def glibc239(monkeypatch):
    """The opt-in glibc 2.39 binary64 transcriptions (LGPL, source checkout only)."""
    from woof.core import glibc239_gpu
    if not glibc239_gpu.header_path().is_file():
        pytest.skip("the opt-in glibc 2.39 transcription is only in a source checkout")
    monkeypatch.setenv("WOOF_GLIBC239_LIBM64", "1")
    ct._module_on.cache_clear()
    crad._module_on.cache_clear()
    yield
    ct._module_on.cache_clear()
    crad._module_on.cache_clear()


def test_case_inputs_are_the_recorded_bytes():
    recorded = dict(reversed(line.split()) for line in
                    (FIXTURE / "inputs.sha256").read_text(encoding="utf-8").splitlines())
    assert recorded == {f"{case['name']}/r8": co.input_digest(case) for case in co.cases("r8", co.STEP_CASES)}


def test_fixture_covers_the_branches():
    fixture, logs = co.load_fixture(FIXTURE), co.load_logs(FIXTURE)
    cases = {c["name"]: c for c in co.cases("r8", co.STEP_CASES)}
    assert set(logs) == set(cases)
    outcomes = {name: co.oracle_step_abort(log) for name, log in logs.items()}
    # the abort stops the step in its first stage, at the first aborting cell (indexToCellID 6)
    assert outcomes.pop("step_bl99_abort_salinity") == (
        ct.SEAICE_ERROR_COL_VERT_THERM, 6,
        "column_vertical_thermodynamics: ice: Vertical thermo error: " + ct.STOP_LABELS[3])
    assert set(outcomes.values()) == {(0, None, None)}
    for name, out in fixture.items():
        for key, value in out.items():
            if key != "ridging/ratioRidgeThicknessToIce":
                assert np.isfinite(value).all(), (name, key)
    n_solve = co.MESH["nCellsSolve"]
    folded = cv._folded_constants()["airDragCoefficient"]
    for name, out in fixture.items():
        if name == "step_bl99_abort_salinity":
            continue
        # every stage ran: the drag coefficients reset, ridging, the coupler's area fluxes
        assert (out["drag/oceanDragCoefficient"] == cstep.ICE_OCEAN_DRAG_COEFFICIENT).all(), name
        assert (out["drag/airDragCoefficient"] == folded).all(), name
        assert (out["ridging/areaLossRidge"][:n_solve] > 0).any(), name
        assert "ocean_fluxes/oceanHeatFluxArea" in out, name
        # the last step is not finalised: the coupler reads its fluxes
        assert (out["ocean_fluxes/oceanFreshWaterFlux"][:n_solve] != 0).any(), name
        for key in ("albedoVisibleDirectCell", "albedoIRDiffuseCell"):
            albedo = out[f"shortwave/{key}"][:n_solve]
            assert ((albedo >= 0) & (albedo <= 1)).all(), (name, key)
    # without the vertical thermodynamics the thermodynamic fluxes stay re-initialised
    quiet = fixture["step_mushy_level_ponds_no_vertical_thermodynamics"]
    assert (quiet["atmos_fluxes/surfaceHeatFlux"] == 0).all()
    assert (quiet["melt_growth_rates/congelation"] == 0).all()
    # the abort leaves the owned cells after it as they were
    start = cases["step_bl99_abort_salinity"]["pools"]["tracers"]["iceVolumeCategory"]
    assert (fixture["step_bl99_abort_salinity"]["tracers/iceVolumeCategory"][6:n_solve] == start[6:n_solve]).all()


@pytest.mark.parametrize("overrides, match", [
    ({"config_use_vertical_tracers": True}, "config_use_vertical_tracers.*without the column biogeochemistry"),
    ({"config_use_column_biogeochemistry": True}, "biogeochemistry"),
    ({"config_use_ocean_mixed_layer": True, "config_atmos_boundary_method": "constant"}, "atmo_boundary_const"),
    ({"config_stress_divergence_scheme": "weak"}, "velocity_weak"),
    ({"config_use_column_package": False}, "column package on"),
])
def test_unported_configurations_are_refused_by_name(overrides, match):
    with pytest.raises(ConfigurationRefusal, match=match):
        cstep.check_config({**_case("step_bl99_dedd")["config"], **overrides})


def test_the_step_reads_the_advanced_clock_on_its_calendar():
    noleap = {"config_start_time": "2000-02-28_23:00:00", "config_dt": 3600.0}
    assert cstep.time_stamps(noleap, 2) == ["2000-03-01_00:00:00", "2000-03-01_01:00:00"]
    gregorian = {**noleap, "config_calendar_type": "gregorian"}
    assert cstep.time_stamps(gregorian, 2) == ["2000-02-29_00:00:00", "2000-02-29_01:00:00"]
    assert cstep.clock_day({"config_calendar_type": "gregorian_noleap"}, "2000-03-01_01:30:15") == (60.0, 5415)
    assert cstep.clock_day({"config_calendar_type": "gregorian"}, "2000-03-01_01:30:15") == (61.0, 5415)


def test_the_fluxes_finalised_without_the_column_package_are_the_air_stress_and_references():
    n = 4
    pools = {"velocity_solver": {"airStressCellU": np.ones(n), "airStressCellV": np.ones(n)},
             "atmos_coupling": {k: np.ones(n) for k in ("atmosReferenceTemperature2m", "atmosReferenceHumidity2m",
                                                        "atmosReferenceSpeed10m")},
             "atmos_fluxes": {"sensibleHeatFlux": np.ones(n)}, "ocean_fluxes": {"oceanHeatFlux": np.ones(n)}}
    cstep.reinitialize_fluxes(pools, {"config_use_column_package": False})
    assert all((a == 0).all() for p in ("velocity_solver", "atmos_coupling") for a in pools[p].values())
    assert (pools["atmos_fluxes"]["sensibleHeatFlux"] == 1).all()
    assert (pools["ocean_fluxes"]["oceanHeatFlux"] == 1).all()


@requires_gpu
def test_gpu_matches_the_v842_oracle_word_for_word_over_the_steps(glibc239):
    report = co.measure_step()
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


def _rasm_drag(monkeypatch):
    monkeypatch.setattr(cstep, "ICE_OCEAN_DRAG_COEFFICIENT", 0.00962)


def _no_finalize(monkeypatch):
    monkeypatch.setattr(cstep, "reinitialize_fluxes", lambda pools, configs: None)


def _no_dynamics_reinit(monkeypatch):
    monkeypatch.setattr(cstep, "reinitialize_diagnostics_dynamics", lambda pools, configs: None)


def _start_day(monkeypatch):
    monkeypatch.setattr(cstep, "clock_day", lambda configs, now: (
        ct.day_of_year(configs["config_start_time"], configs["config_calendar_type"]),
        crad.seconds_into_day(configs["config_start_time"])))


@requires_gpu
@pytest.mark.parametrize("mutate, field, case", [
    # the RASM build's ice-ocean drag coefficient
    (_rasm_drag, "drag/oceanDragCoefficient", "step_bl99_dedd"),
    # seaice_timestep_finalize skipped between the steps: the fluxes accumulate
    (_no_finalize, "atmos_fluxes/sensibleHeatFlux", "step_bl99_dedd"),
    # the dynamic diagnostics not re-initialised before each subcycle
    (_no_dynamics_reinit, "ridging/areaLossRidge", "step_bl99_dedd_subcycles_gregorian"),
    # the shortwave read at the start's time, not the advanced clock's
    (_start_day, "shortwave/albedoVisibleDirectCell", "step_bl99_dedd"),
])
def test_a_mutated_step_is_caught(monkeypatch, glibc239, mutate, field, case):
    mutate(monkeypatch)
    got, _ = co.run_gpu(_case(case))
    want = co.load_fixture(FIXTURE)[case][field]
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
                    str(tmp_path / "build"), "--cases", "step", "--out", str(tmp_path / "fixture")], check=True)
    fresh = np.load(tmp_path / "fixture" / "oracle-r8.npz")
    pinned = np.load(FIXTURE / "oracle-r8.npz")
    assert sorted(fresh.files) == sorted(pinned.files)
    for key in pinned.files:
        assert co.word_mismatches(fresh[key], pinned[key]) == 0, key
    for name in ("column-logs.json", "inputs.sha256"):
        assert (tmp_path / "fixture" / name).read_text() == (FIXTURE / name).read_text()
