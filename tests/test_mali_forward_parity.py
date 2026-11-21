"""The MALI forward mode (clock, initialisation, initial solve and time-step
loop), word for word against v8.4.2.

The reference is MALI v8.4.2's land-ice core (``mpas_li_core.F``,
``mpas_li_time_integration.F`` and every module they reach) with v8.4.2's
own ``mpas_timekeeping.F`` and ESMF time library, compiled verbatim with
gfortran -O0 -ffp-contract=off at RKIND 4 and RKIND 8.
``tools/mali_oracle/run_forward.F90`` runs a whole cold start:
``li_simulation_clock_init``, ``li_core_init`` and ``li_core_run`` to the
stop time. The oracle's outputs are committed
(tests/data/oracles/mali_forward, PROVENANCE.md), so the GPU comparison runs
without gfortran.

THE BREAKAGES THESE PREVENT
---------------------------
1. A run that agrees with MALI only to a tolerance. Every mesh, geometry,
   thermal, velocity and hydro word, the clock's time stamps, the step
   count, the time step and the days since the start included, must be
   bit-identical after the last step.
2. A case generator whose bytes drift. The input digests are pinned.
3. A fixture that misses a branch: the dome on fixed and adaptive time steps
   (the adaptive one bound by the force alarm), with calving, EISMINT2 A
   with the temperature and the enthalpy solver, the subglacial hydrology
   (and grounded ice below sea level), the enthalpy benchmark; each on a
   uniform and a variable resolution mesh; both calendars.
4. A gate that cannot fail. Mutating the initialisation's arithmetic, the
   barycentric weights' operand order, or the force alarm's reset must
   produce mismatches.
5. A restart that is not bitwise: a run stopped, written, read and resumed
   must end on the straight run's words.
6. Outputs that depend on the model's uninitialised scratch or on prior
   diagnostics: the oracle fills them with random words.
7. Statistics that agree only to the printed digits. Every word
   li_compute_statistics's reductions return (``stats/real``, ``stats/int``,
   recorded by the oracle's mpas_dmpar) and every character of its reports
   (``stats/log``) must match; a pairwise sum in place of the Fortran's
   sequential one, or MAX with the other tie rule, must be caught.
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
from woof.verify import mali_forward_oracle as fo

ROOT = Path(__file__).resolve().parents[1]


def test_case_inputs_are_the_recorded_bytes():
    recorded = {}
    for line in (fo.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        recorded[name] = digest
    assert recorded == {**{case["name"]: fo.input_digest(case) for case in fo.cases()},
                        **{case["name"] + "/r8": fo.input_digest(case) for case in fo.cases("r8")}}


@pytest.mark.parametrize("precision", ["r4", "r8"])
def test_fixture_covers_the_branches(precision):
    fixture = fo.load_fixture(precision)
    cases = {case["name"]: case for case in fo.cases(precision)}
    assert set(fixture) == set(cases) == {f"{n}_{s}" for n, _, _ in fo.CASES for s, _ in fo.MESHES}
    configs = [c["config"] for c in cases.values()]
    assert {c["config_velocity_solver"] for c in configs} == {"sia", "simple", "none"}
    assert {c.get("config_simple_velocity_type") for c in configs} == {None, "uniform", "radial"}
    assert {c["config_calving"] for c in configs if c["config_velocity_solver"] == "simple"} == {
        "none", "floating", "topographic_threshold"}
    assert {c["config_thermal_solver"] for c in configs} == {"none", "temperature", "enthalpy"}
    assert {c["config_adaptive_timestep"] for c in configs} == {False, True}
    assert {c["config_calendar_type"] for c in configs} == {"gregorian", "gregorian_noleap"}
    assert {c["config_SGH"] for c in configs} == {False, True}
    assert {c["config_sia_tangent_slope_calculation"] for c in configs} >= {"from_vertex_barycentric",
                                                                            "from_normal_slope"}
    for name, out in fixture.items():
        case = cases[name]
        # the oracle ran clean; its only logged errors are the barycentric
        # weights across the periodic seam, which the model only logs
        assert not any(out[k][0] for k in ("errClock", "errInit", "errRun"))
        bary = (case["config"]["config_velocity_solver"] == "sia"
                and case["config"]["config_sia_tangent_slope_calculation"] == "from_vertex_barycentric")
        assert (out["logErrors"][0] > 0) == bary, name
        assert all(np.isfinite(v).all() for k, v in out.items() if v.dtype.kind == "f" and k != "velocity/eTheta")
        # the run reached its stop time, one step at a time
        assert str(out["clock/now"]) == str(out["mesh/xtime"])
        assert int(out["mesh/timestepNumber"][()]) >= 2
        assert str(out["mesh/simulationStartTime"]) == case["config"]["config_start_time"]
        # the mesh's own words are left as they were
        for field in fo.MESH_INPUTS:
            if field not in case["pools"]["mesh"]:
                continue
            assert fo.word_mismatches(out[f"mesh/{field}"], case["pools"]["mesh"][field]) == 0, (name, field)
        # the time levels shifted after the last step: the old level holds the step's mask
        assert (out["geometry/vertexMask@2"] != 0).any()
    # the adaptive step: bound by the 20-day force alarm on the uniform mesh, by the CFL on the varres one
    adaptive = fixture["dome_adaptive_uniform"]
    assert int(adaptive["mesh/timestepNumber"][()]) > 40 // 8
    assert fixture["dome_adaptive_varres"]["mesh/deltat"][()] < adaptive["mesh/deltat"][()]
    # calving took the floating margin
    assert (fixture["dome_calving_uniform"]["geometry/calvingThickness"] > 0).any()
    # 'simple': the prescribed velocity moved the ice, the circular-shelf test reshaped the bed,
    # and the Dirichlet mask it never reads came back as it went in
    cases = {case["name"]: case for case in fo.cases(precision)}
    for mesh, _ in fo.MESHES:
        for name in ("simple_uniform", "simple_radial_shelf", "simple_rim"):
            out, case = fixture[f"{name}_{mesh}"], cases[f"{name}_{mesh}"]
            assert (out["velocity/normalVelocity"] != 0).any()
            assert fo.word_mismatches(out["velocity/dirichletVelocityMask"],
                                      case["pools"]["velocity"]["dirichletVelocityMask"]) == 0
            changed = fo.word_mismatches(out["geometry/bedTopography"], case["pools"]["geometry"]["bedTopography"])
            assert (changed > 0) == (name != "simple_uniform"), name
        assert (fixture[f"simple_radial_shelf_{mesh}"]["geometry/bedTopography"] == -880.0).any()
    # the statistics: a report on startup (when asked) and every interval-th step, and
    # none from the cases without them
    for name, out in fixture.items():
        config = cases[name]["config"]
        if not fo.stats_on(config):
            assert not any(k.startswith("stats/") for k in out), name
            continue
        steps = int(out["mesh/timestepNumber"][()])
        expected = int(config["config_write_stats_on_startup"]) + steps // config["config_stats_interval"]
        log = str(out["stats/log"])
        assert log.count("Global statistics: time =") == expected, name
        assert out["stats/int"].size == 12 * expected, name
        found = config["config_stats_cell_ID"] != 0
        assert ("Local cell ID, block, proc =       0" in log) != found, name
    # 'none' leaves every normal velocity 0: the magnitude is MAX of two zeros
    assert "Max velocity (m/yr), edge, level      0.0000000000000000" in str(
        fixture["stats_benchmark_uniform"]["stats/log"])
    # the gregorian calendar's leap day lies inside EISMINT2's run
    assert str(fixture["eismint2_a_temperature_uniform"]["clock/now"]) == "2000-03-11_00:00:00"


def _report(precision: str, names=None) -> dict:
    reference = fo.load_fixture(precision)
    report = {}
    for case in fo.cases(precision):
        if names is not None and case["name"] not in names:
            continue
        got = fo.run_gpu(case)
        report[case["name"]] = {k: fo.word_mismatches(got[k], reference[case["name"]][k]) for k in got}
    return report


@requires_gpu
@pytest.mark.parametrize("precision", ["r4", "r8"])
def test_gpu_matches_the_v842_oracle_word_for_word(precision):
    report = _report(precision)
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


def _caught(precision: str, names, field: str) -> int:
    reference = fo.load_fixture(precision)
    total = 0
    for case in fo.cases(precision):
        if case["name"] in names:
            total += fo.word_mismatches(fo.run_gpu(case)[field], reference[case["name"]][field])
    return total


_CU_MUTATIONS = [
    # li_calculate_mask_init's ice extent taken at zero thickness too: the bit stays in the mask
    ("if (thickness[i] > LI_C(0.0)) mask = mask | LI_INITIAL_ICE_EXTENT;",
     "if (thickness[i] >= LI_C(0.0)) mask = mask | LI_INITIAL_ICE_EXTENT;",
     "geometry/cellMask", ("dome_uniform",)),
    # the till water left above config_SGH_till_max: the sheet's till storage starts wrong
    ("tillWaterThickness[i] = li_min(tillMax, till);", "tillWaterThickness[i] = till;",
     "hydro/tillWaterThickness", ("hydro_radial_uniform", "hydro_marine_varres")),
]


@requires_gpu
@pytest.mark.parametrize("precision", ["r4", "r8"])
@pytest.mark.parametrize("original, mutated, field, names", _CU_MUTATIONS)
def test_a_mutated_kernel_is_caught(monkeypatch, precision, original, mutated, field, names):
    from woof.hex.mpas_ice import forward_v842 as fw
    source = fw.module_source(precision)
    assert source.count(original) == 1
    monkeypatch.setattr(fw, "module_source", lambda p="r4": source.replace(original, mutated))
    fw._module_on.cache_clear()
    try:
        assert _caught(precision, names, field) > 0
    finally:
        fw._module_on.cache_clear()


@requires_gpu
@pytest.mark.parametrize("precision", ["r4", "r8"])
@pytest.mark.parametrize("mutation, field, names", [
    # the layer centres' sum regrouped
    ("vertical_grid", "mesh/layerCenterSigma", ("eismint2_a_temperature_uniform", "enthalpy_benchmark_uniform")),
    # the barycentric areas taken from the triangle's second corner
    ("_signed_area", "mesh/baryWeightsOnVertex", ("dome_uniform", "dome_varres")),
    # the force alarm never reset: past its first ring the step is bound by a time gone
    ("reset_force_alarm", "mesh/deltat", ("dome_adaptive_uniform",)),
])
def test_a_mutated_host_step_is_caught(monkeypatch, precision, mutation, field, names):
    from woof.hex.mpas_ice import forward_v842 as fw
    grid, area = fw.vertical_grid, fw._signed_area

    def regrouped(fractions, real):
        f, centre, interface = grid(fractions, real)
        for k in range(1, f.size):
            centre[k] = centre[k - 1] + (real(0.5) * f[k - 1] + real(0.5) * f[k])
        return f, centre, interface

    reset = fw.Clock.reset_force_alarm

    def reset_at_init_only(self):
        # clock_init's reset at the start time stays; the loop's resets are gone
        if self.now == self.start:
            reset(self)

    target, name, value = {"vertical_grid": (fw, "vertical_grid", regrouped),
                           "_signed_area": (fw, "_signed_area", lambda a, b, c, real: area(b, c, a, real)),
                           "reset_force_alarm": (fw.Clock, "reset_force_alarm", reset_at_init_only)}[mutation]
    monkeypatch.setattr(target, name, value)
    if mutation == "reset_force_alarm":
        # past the first ring the seconds to the next ring go to 0 and below:
        # the model's own check stops the run there, and not before
        case = _case(names[0], precision)
        pools = fo.device_pools(case)
        clock = fw.cold_start(pools=pools, config=case["config"])
        ring = clock.force_prev + clock.force_interval
        steps = 0
        with pytest.raises(ValueError, match="config_min_adaptive_timestep"):
            while True:
                steps += fw.advance(pools=pools, config=case["config"], clock=clock, n_steps=1)
        assert clock.now >= ring and steps >= 2, (clock.now, ring, steps)
        return
    assert _caught(precision, names, field) > 0


def _case(name: str, precision: str = "r4") -> dict:
    return next(c for c in fo.cases(precision) if c["name"] == name)


@requires_gpu
@pytest.mark.parametrize("name", ["dome_adaptive_varres", "hydro_radial_uniform", "eismint2_a_temperature_uniform"])
def test_a_restarted_run_ends_on_the_straight_runs_words(tmp_path, name):
    from woof.hex.mpas_ice import forward_v842 as fw
    case = _case(name)
    straight = fo.run_gpu(case)
    pools = fo.device_pools(case)
    clock = fw.cold_start(pools=pools, config=case["config"])
    assert fw.advance(pools=pools, config=case["config"], clock=clock, n_steps=2) == 2
    fw.write_restart(tmp_path / "restart.npz", pools=pools, clock=clock)
    del pools
    pools, clock = fw.read_restart(tmp_path / "restart.npz", config=case["config"])
    assert fw.advance(pools=pools, config=case["config"], clock=clock) >= 1
    resumed = fo.outputs_of(case, pools, clock)
    assert set(resumed) == set(straight)
    bad = {k: fo.word_mismatches(resumed[k], straight[k]) for k in straight}
    assert not any(bad.values()), {k: v for k, v in bad.items() if v}
    other = {"gregorian": "gregorian_noleap", "gregorian_noleap": "gregorian"}[case["config"]["config_calendar_type"]]
    with pytest.raises(Exception, match="config_calendar_type"):
        fw.read_restart(tmp_path / "restart.npz", config=dict(case["config"], config_calendar_type=other))


@requires_gpu
@pytest.mark.parametrize("precision", ["r4", "r8"])
def test_an_adaptive_restart_across_the_force_alarms_ring(tmp_path, precision):
    # restarts before the first ring, on it and after it: the clock (its
    # calendar and alarm) comes back as it was and the run ends on the
    # straight run's words
    from woof.hex.errors import ConfigurationRefusal
    from woof.hex.mpas_ice import forward_v842 as fw
    case = _case("dome_adaptive_uniform", precision)
    straight = fo.run_gpu(case)
    pools = fo.device_pools(case)
    clock = fw.cold_start(pools=pools, config=case["config"])
    first_ring = clock.force_prev + clock.force_interval
    sides = set()
    while not clock.stopped:
        fw.advance(pools=pools, config=case["config"], clock=clock, n_steps=1)
        path = tmp_path / f"restart-{clock.now}.npz"
        fw.write_restart(path, pools=pools, clock=clock)
        restored_pools, restored = fw.read_restart(path, config=case["config"], precision=precision)
        assert restored == clock and restored.calendar == "gregorian_noleap"
        assert restored.time_stamp == pools["mesh"]["xtime"]
        sides.add((clock.now > first_ring) - (clock.now < first_ring))
        fw.advance(pools=restored_pools, config=case["config"], clock=restored)
        resumed = fo.outputs_of(case, restored_pools, restored)
        bad = {k: fo.word_mismatches(resumed[k], straight[k]) for k in straight}
        assert not any(bad.values()), (clock.time_stamp, {k: v for k, v in bad.items() if v})
    assert sides == {-1, 0, 1}
    with pytest.raises(ConfigurationRefusal, match="config_calendar_type"):
        fw.read_restart(path, config=dict(case["config"], config_calendar_type="gregorian"))
    other = "r8" if precision == "r4" else "r4"
    with pytest.raises(ConfigurationRefusal, match="precision"):
        fw.read_restart(path, config=case["config"], precision=other)
    with pytest.raises(ConfigurationRefusal, match="config_adaptive_timestep_force_interval"):
        fw.read_restart(path, config=dict(case["config"], config_adaptive_timestep_force_interval="10_00:00:00"))
    with pytest.raises(ConfigurationRefusal, match="config_adaptive_timestep"):
        fw.read_restart(path, config=dict(case["config"], config_adaptive_timestep=False))
    with pytest.raises(ConfigurationRefusal, match="config_velocity_solver"):
        fw.read_restart(path, config=dict(case["config"], config_velocity_solver="FO"))


@requires_gpu
def test_a_fixed_step_restart_refuses_another_time_step(tmp_path):
    from woof.hex.errors import ConfigurationRefusal
    from woof.hex.mpas_ice import forward_v842 as fw
    case = _case("dome_uniform")
    pools = fo.device_pools(case)
    clock = fw.cold_start(pools=pools, config=case["config"])
    fw.advance(pools=pools, config=case["config"], clock=clock, n_steps=1)
    fw.write_restart(tmp_path / "restart.npz", pools=pools, clock=clock)
    with pytest.raises(ConfigurationRefusal, match="config_dt"):
        fw.read_restart(tmp_path / "restart.npz", config=dict(case["config"], config_dt="0000-00-10_00:00:00"))


@requires_gpu
def test_the_sgh_init_mask_correction_is_opt_in(tmp_path):
    # v8.4.2's li_SGH_init reads a cell mask holding only the initial-extent bit,
    # so grounded ice below sea level takes the ocean's water pressure
    from woof.hex.mpas_ice import forward_v842 as fw
    case = _case("hydro_marine_uniform")
    default = fo.run_gpu(case)
    pinned = fo.load_fixture("r4")[case["name"]]
    assert fo.word_mismatches(default["hydro/waterPressure"], pinned["hydro/waterPressure"]) == 0
    corrected = fo.run_gpu(case, {"woof_sgh_init_mask": "corrected"})
    assert fo.word_mismatches(corrected["hydro/waterPressure"], pinned["hydro/waterPressure"]) > 0
    from woof.hex.errors import ConfigurationRefusal
    with pytest.raises(ConfigurationRefusal, match="woof_sgh_init_mask"):
        fw.run(pools=fo.device_pools(case), config=dict(case["config"], woof_sgh_init_mask="other"))


@requires_gpu
@pytest.mark.parametrize("precision", ["r4", "r8"])
@pytest.mark.parametrize("mesh", ["uniform", "varres"])
def test_the_enthalpy_advection_units_correction_is_opt_in(precision, mesh):
    # v8.4.2's tracer_setup/tracer_finish convert the advected enthalpy with
    # Kelvin temperatures against li_thermal's Celsius melting point, so the
    # ice turns temperate and melts; 'corrected' converts in Celsius
    from woof.hex.errors import ConfigurationRefusal
    from woof.hex.mpas_ice import forward_v842 as fw
    case = _case(f"eismint2_a_enthalpy_{mesh}", precision)
    pinned = fo.load_fixture(precision)[case["name"]]
    default = fo.run_gpu(case)
    assert all(fo.word_mismatches(default[k], pinned[k]) == 0 for k in default if k in pinned)
    corrected = fo.run_gpu(case, {"woof_enthalpy_advection_units": "corrected"})
    h0 = np.asarray(case["pools"]["geometry"]["thickness"], np.float64)
    ice = h0 > 0
    for out, physical in ((default, False), (corrected, True)):
        h = np.asarray(out["geometry/thickness"], np.float64)
        t = np.asarray(out["thermal/temperature"], np.float64)[ice]
        w = np.asarray(out["thermal/waterfrac"], np.float64)[ice]
        kept = h[ice].sum() / h0[ice].sum()
        stays = bool(kept > 0.99 and w.max() < 1e-3 and np.isfinite(t).all() and 200.0 < t.min()
                     and t.max() <= 273.15)
        assert stays == physical, (physical, kept, w.max(), t.min(), t.max())
    with pytest.raises(ConfigurationRefusal, match="woof_enthalpy_advection_units"):
        fw.run(pools=fo.device_pools(case), config=dict(case["config"], woof_enthalpy_advection_units="other"))


def _refused_on_host(case: dict, config: dict) -> None:
    """What cold_start checks before it touches a device array."""
    from woof.hex.mpas_ice import forward_v842 as fw
    fw.check_config(config, case["precision"], case["pools"]["velocity"])
    fw.clock_init(fw.setup_config_options(config))


@pytest.mark.parametrize("knob, value", [
    ("config_do_restart", True), ("config_time_integration", "rk4"), ("config_write_albany_ascii_mesh", True),
    ("woof_statistics_extrema", "other"),
    ("config_AM_oceanStats_enable", True), ("woof_analysis_empty", "other"), ("config_year_digits", 5), ("config_velocity_solver", "FO"),
    ("config_dt", "0000-01-00_00:00:00"), ("config_dt", "00:00:10.5"), ("config_start_time", "file"),
])
def test_the_pieces_not_ported_are_refused_by_name(knob, value):
    from woof.hex.errors import ConfigurationRefusal
    case = _case("dome_uniform")
    with pytest.raises(ConfigurationRefusal, match=knob):
        _refused_on_host(case, dict(case["config"], **{knob: value}))


@pytest.mark.parametrize("knob, value", [
    ("config_dt", "00:00:00"), ("config_dt", "0000-00-00_00:00:00"),
    ("config_stop_time", "0001-06-01_00:00:00"),
])
def test_a_clock_that_never_stops_or_runs_backwards_is_refused(knob, value):
    from woof.hex.errors import ConfigurationRefusal
    case = _case("dome_uniform")
    config = dict(case["config"], **{knob: value})
    if knob == "config_stop_time":
        config.update(config_run_duration="none", config_start_time="0002-01-01_00:00:00")
    with pytest.raises(ConfigurationRefusal, match=knob):
        _refused_on_host(case, config)


def test_years_are_refused_on_the_gregorian_calendar():
    from woof.hex.errors import ConfigurationRefusal
    case = _case("eismint2_a_temperature_uniform")
    with pytest.raises(ConfigurationRefusal, match="config_run_duration"):
        _refused_on_host(case, dict(case["config"], config_run_duration="0001-00-00_00:00:00"))


@requires_gpu
@pytest.mark.parametrize("name", ["coeffs_reconstruct", "edgeNormalVectors"])
def test_a_mesh_without_its_reconstruction_coefficients_is_refused(name):
    # mpas_init_reconstruct is not ported: cold_start takes them as given
    from woof.hex.errors import ConfigurationRefusal
    from woof.hex.mpas_ice import forward_v842 as fw
    case = _case("dome_uniform")
    pools = fo.device_pools(case)
    pools["mesh"][name][...] = 0
    with pytest.raises(ConfigurationRefusal, match=name):
        fw.cold_start(pools=pools, config=case["config"])
    del pools["mesh"][name]
    with pytest.raises(ConfigurationRefusal, match=name):
        fw.cold_start(pools=pools, config=case["config"])


def test_the_clock_reads_mpas_intervals():
    from woof.hex.mpas_ice import forward_v842 as fw
    assert fw.parse_interval("0001-02-03_04:05:06", "k") == (1, 2, ((3 * 24 + 4) * 60 + 5) * 60 + 6)
    assert fw.parse_interval("10_00:00:00", "k") == (0, 0, 10 * 86400)
    assert fw.parse_interval("06:00:00", "k") == (0, 0, 6 * 3600)
    assert fw.parse_interval("90", "k") == (0, 0, 90)
    assert fw.parse_interval("01:30", "k") == (0, 0, 90)
    assert fw.parse_interval("00:00:1126.000000000", "k") == (0, 0, 1126)
    assert fw.interval_seconds("0002-00-00_00:00:00", "gregorian_noleap", "k") == 2 * 365 * 86400
    # dt = (days * 24 * 60 * 60) + seconds, at RKIND: 10^8 + 1 s is not a float32
    assert fw.interval_dt(100_000_001, np.float32) == np.float32(100_000_000)
    assert fw.interval_dt(100_000_001, np.float64) == 100_000_001.0
    config = dict(fo.BASE_CONFIG, config_adaptive_timestep=True,
                  config_adaptive_timestep_force_interval="0000-00-20_00:00:00")
    clock = fw.clock_init(config)
    assert (clock.force_prev, clock.seconds_to_next_force_time()) == (clock.start, 20 * 86400)
    clock.now += 23 * 86400
    clock.reset_force_alarm()
    assert clock.force_prev == clock.start + 20 * 86400
    assert clock.time_stamp == "0001-01-24_00:00:00"


def test_the_vertical_grid_fixes_a_small_fraction_error_and_refuses_a_large_one():
    from woof.hex.mpas_ice import forward_v842 as fw
    f, centre, interface = fw.vertical_grid(np.full(3, np.float32(1.0) / np.float32(3.0)), np.float32)
    assert interface[-1] == 1 and centre[0] == np.float32(0.5) * f[0]
    with pytest.raises(ValueError, match="layerThicknessFractions"):
        fw.vertical_grid(np.full(4, 0.3), np.float64)


def _oracle_toolchain() -> tuple[str, Path] | None:
    source = os.environ.get("WOOF_MPAS_V842_SOURCE")
    gfortran = os.environ.get("GFORTRAN") or shutil.which("gfortran")
    if not source or not gfortran or not Path(source, "src", "core_landice").is_dir():
        return None
    return gfortran, Path(source)


@pytest.mark.skipif(_oracle_toolchain() is None,
                    reason="the MALI oracle build needs gfortran (pixi -e mpas-oracle) and "
                           "WOOF_MPAS_V842_SOURCE naming an MPAS-Model v8.4.2 tree")
@pytest.mark.skipif(platform.machine() not in ("x86_64", "AMD64"),
                    reason="the fixture is x86-64 gfortran's: its MAX is maxss/maxsd, which keeps an operand on a "
                           "tie of signed zeros, where arm64 gfortran's fmaxnm orders -0 below +0")
def test_the_oracle_rebuilt_here_writes_the_fixture(tmp_path):
    gfortran, source = _oracle_toolchain()
    subprocess.run(["bash", str(ROOT / "tools" / "mali_oracle" / "build_forward.sh"), str(source),
                    str(tmp_path / "build")], check=True, env={**os.environ, "GFORTRAN": gfortran})
    subprocess.run([sys.executable, str(ROOT / "tools" / "mali_oracle" / "make_forward_fixture.py"),
                    str(tmp_path / "build"), "--out", str(tmp_path / "fixture")], check=True)
    for precision in ("r4", "r8"):
        fresh = np.load(tmp_path / "fixture" / f"oracle-{precision}.npz")
        pinned = np.load(fo.FIXTURE_DIR / f"oracle-{precision}.npz")
        assert sorted(fresh.files) == sorted(pinned.files)
        for key in pinned.files:
            assert fo.word_mismatches(fresh[key], pinned[key]) == 0, key


@requires_gpu
@pytest.mark.parametrize("name", ["simple_radial_shelf_uniform", "simple_rim_varres"])
def test_the_simple_velocity_bed_correction_is_opt_in(name):
    # v8.4.2's li_velocity_simple_block_init writes its circular-shelf test's bed
    # (a -880 m spike, or a deepened rim) over the input's whenever calving is
    # 'floating' or 'topographic_threshold'; 'corrected' keeps the input bed
    from woof.hex.errors import ConfigurationRefusal
    from woof.hex.mpas_ice import forward_v842 as fw
    case = _case(name)
    pinned = fo.load_fixture("r4")[case["name"]]
    default = fo.run_gpu(case)
    assert all(fo.word_mismatches(default[k], pinned[k]) == 0 for k in default)
    corrected = fo.run_gpu(case, {"woof_simple_velocity_bed": "corrected"})
    bed = case["pools"]["geometry"]["bedTopography"]
    assert fo.word_mismatches(corrected["geometry/bedTopography"], bed) == 0
    assert fo.word_mismatches(default["geometry/bedTopography"], bed) > 0
    # the prescribed velocity does not depend on the bed
    assert fo.word_mismatches(corrected["velocity/normalVelocityInitial"], pinned["velocity/normalVelocityInitial"]) == 0
    assert all(np.isfinite(v).all() for k, v in corrected.items() if v.dtype.kind == "f")
    with pytest.raises(ConfigurationRefusal, match="woof_simple_velocity_bed"):
        fw.run(pools=fo.device_pools(case), config=dict(case["config"], woof_simple_velocity_bed="other"))


@requires_gpu
@pytest.mark.parametrize("precision", ["r4", "r8"])
def test_the_simple_seam_edges_take_cell_twos_vector(monkeypatch, precision):
    # edges stored across the periodic seam are longer than dcEdge from
    # cellsOnEdge(1): the model takes the vector into cellsOnEdge(2), negated.
    # Ignoring that branch leaves the oracle's words.
    from woof.hex.mpas_ice import velocity_simple_v842 as vs
    names = {f"simple_uniform_{m}" for m, _ in fo.MESHES}
    mesh = fo.mesh_of(0.0)
    x_edge, y_edge = fo.edge_points_of(0.0)
    c1 = mesh["cellsOnEdge"][:, 0] - 1
    far = np.hypot(x_edge - mesh["xCell"][c1], y_edge - mesh["yCell"][c1]) > mesh["dcEdge"]
    assert far.any() and not far.all()
    real = fo.PRECISIONS[precision]
    original = vs.normal_velocity_initial

    def never_far(**kw):
        return original(**dict(kw, dc_edge=np.full(np.shape(kw["dc_edge"]), np.inf, real)))

    monkeypatch.setattr(vs, "normal_velocity_initial", never_far)
    assert _caught(precision, names, "velocity/normalVelocityInitial") > 0


@pytest.mark.parametrize("value, w, d, text", [
    # gfortran 16's own output (x86-64): ties to even on the exact binary value,
    # the sign of -0 kept, a three-digit exponent without its E, overflow as asterisks
    (2.0 ** -17, 24, 16, "      0.0000076293945312"), (-0.0, 24, 16, "     -0.0000000000000000"),
    (-1.0e-20, 24, 16, "     -0.0000000000000000"), (12345678.9, 24, 16, "*" * 24),
    (np.float32(0.1), 24, 16, "      0.1000000014901161"), (np.float32(0.1), 6, 4, "0.1000"),
    (0.00005, 6, 4, "0.0001"), (0.00015, 6, 4, "0.0001"),
])
def test_the_f_edit_descriptor_is_gfortrans(value, w, d, text):
    from woof.hex.mpas_ice import statistics_v842 as st
    assert st.fmt_f(value, w, d) == text


@pytest.mark.parametrize("value, text", [
    (2.0 ** -17, "  0.7629394531250000E-05"), (-0.0, " -0.0000000000000000E+00"),
    (1.0e120, "  0.1000000000000000+121"), (1.0e-300, "  0.1000000000000000-299"),
    (np.float32(123456.789), "  0.1234567890625000E+06"), (0.99999999999999999, "  0.1000000000000000E+01"),
    (9.99999999999999999e22, "  0.9999999999999999E+23"),
])
def test_the_e_edit_descriptor_is_gfortrans(value, text):
    from woof.hex.mpas_ice import statistics_v842 as st
    assert st.fmt_e(value, 24, 16) == text


def test_max_keeps_x86_gfortrans_tie_rule():
    # MAX(localMax, -localMin) compiles to maxss with -localMin as the destination:
    # a tie of signed zeros keeps localMax (gfortran 16 -O0, x86-64)
    from woof.hex.mpas_ice import statistics_v842 as st
    z = np.float32(0.0)
    assert not np.signbit(st._fmax(z, -z)) and np.signbit(st._fmax(-z, z))


@requires_gpu
@pytest.mark.parametrize("precision", ["r4", "r8"])
@pytest.mark.parametrize("mutation, field", [
    ("pairwise", "stats/real"), ("max_ties", "stats/log"), ("last_location", "stats/int"),
])
def test_a_mutated_statistic_is_caught(monkeypatch, precision, mutation, field):
    from woof.hex.mpas_ice import statistics_v842 as st
    if mutation == "pairwise":
        monkeypatch.setattr(st, "_sequential_sum", lambda terms, real: real(np.sum(np.asarray(terms, real))))
    elif mutation == "max_ties":
        monkeypatch.setattr(st, "_fmax", lambda a, b: a if a > b else b)
    else:
        original = st._first_extreme

        def last(values, init, larger, eligible=None):
            flat = np.asarray(values).ravel()
            value, at = original(values, init, larger, eligible)
            if at:
                at = int(np.flatnonzero(flat == value)[-1]) + 1
            return value, at
        monkeypatch.setattr(st, "_first_extreme", last)
    names = {f"{n}_{m}" for n in ("stats_dome", "stats_eismint2", "stats_simple_shelf", "stats_benchmark")
             for m, _ in fo.MESHES}
    assert _caught(precision, names, field) > 0


@requires_gpu
def test_the_statistics_extrema_correction_is_opt_in():
    # v8.4.2 takes the extrema over every owned cell and edge, ice or not: the
    # dome's ice-free margin reports a minimum thickness of 0. 'ice' takes the
    # ice cells alone; the state's words do not depend on it
    case = _case("stats_dome_uniform")
    pinned = fo.load_fixture("r4")[case["name"]]
    corrected = fo.run_gpu(case, {"woof_statistics_extrema": "ice"})
    for key in pinned:
        if key.startswith("stats/") or key in ("errClock", "errInit", "errRun", "logErrors"):
            continue
        assert fo.word_mismatches(corrected[key], pinned[key]) == 0, key
    default_min = [line for line in str(pinned["stats/log"]).splitlines() if line.startswith("Min thickness")]
    fixed_min = [line for line in str(corrected["stats/log"]).splitlines() if line.startswith("Min thickness")]
    assert all(" 0.0000000000000000" in line for line in default_min)
    assert not any(" 0.0000000000000000" in line for line in fixed_min)
    assert str(corrected["stats/log"]).count("Global statistics") == str(pinned["stats/log"]).count(
        "Global statistics")
