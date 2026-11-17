"""solar_tsi_opt: time-varying TSI on every shortwave output (woof.spacewx).

The admission, the TSI series readers and the scaling run on the CPU (the scaling on
NumPy arrays); the driver test runs the real ``initialize_physics`` + RRTMG radiation
call on a card (CUDA or Metal) and checks that the shortwave outputs carry exactly the
TSI factor while the longwave is untouched, and that the default path runs no new code.
"""
from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta
import inspect

import numpy as np
import pytest

from woof.config import ConfigurationRefused, RunConfig, validate_run_config
from woof.spacewx import solar_irradiance as si

from conftest import requires_gpu  # noqa: E402


def _cfg(**overrides):
    values = dict(nx=6, ny=4, nz=5, dx=3000.0, dy=3000.0, ztop=20000.0,
                  dt=20.0, run_seconds=3600.0, ra_physics=4)
    values.update(overrides)
    return RunConfig(**values)


def _csv(tmp_path, rows):
    path = tmp_path / "tsi.csv"
    path.write_text("date,tsi\n" + "".join(f"{d},{v}\n" for d, v in rows), encoding="utf-8")
    return path


# --- admission ----------------------------------------------------------------------

def test_defaults_are_off_and_admitted():
    cfg = _cfg()
    assert (cfg.solar_tsi_opt, cfg.solar_tsi_source, cfg.solar_tsi_mode) == (0, "", "absolute")
    validate_run_config(cfg)


@pytest.mark.parametrize("overrides, match", [
    (dict(solar_tsi_opt=2), "solar_tsi_opt"),
    (dict(solar_tsi_opt=True), "solar_tsi_opt"),
    (dict(solar_tsi_opt=1), "needs solar_tsi_source"),
    (dict(solar_tsi_opt=1, solar_tsi_source="x.csv", solar_tsi_mode="spectral"), "solar_tsi_mode"),
    (dict(solar_tsi_source="x.csv"), "would do nothing"),
    (dict(solar_tsi_mode="relative"), "would do nothing"),
    (dict(solar_tsi_opt=1, solar_tsi_source="x.csv", ra_physics=0, ra_sw_physics=0, ra_lw_physics=0),
     "no shortwave"),
])
def test_refusals_name_the_breakage(overrides, match):
    with pytest.raises(ConfigurationRefused, match=match):
        validate_run_config(_cfg(**overrides))


def test_farms_is_refused_by_name():
    cfg = dataclasses.replace(_cfg(), solar_tsi_opt=1, solar_tsi_source="x.csv", swint_opt=2)
    with pytest.raises(ConfigurationRefused, match="FARMS"):
        si_validate(cfg)


def si_validate(cfg):
    from woof.config import _validate_solar_tsi
    _validate_solar_tsi(cfg)


def test_an_on_config_is_admitted():
    validate_run_config(_cfg(solar_tsi_opt=1, solar_tsi_source="nrlssi2", solar_tsi_mode="relative"))


# --- the series ---------------------------------------------------------------------

def test_csv_interpolates_between_daily_noons_and_refuses_extrapolation(tmp_path):
    series = si.read_series(_csv(tmp_path, [("2024-05-10", 1361.0), ("2024-05-11", 1363.0)]))
    assert series.at(datetime(2024, 5, 10, 12)) == pytest.approx(1361.0)
    assert series.at(datetime(2024, 5, 11, 0)) == pytest.approx(1362.0)
    assert series.at(datetime(2024, 5, 10, 3)) == pytest.approx(1361.0)    # half day before the first noon
    assert series.at(datetime(2024, 5, 11, 23, 59)) == pytest.approx(1363.0)
    with pytest.raises(si.SolarForcingError, match="outside"):
        series.at(datetime(2024, 5, 12, 0, 1))
    with pytest.raises(si.SolarForcingError, match="outside"):
        series.at(datetime(2024, 5, 9, 23))


def test_a_gap_is_refused(tmp_path):
    series = si.read_series(_csv(tmp_path, [("2024-05-10", 1361.0), ("2024-05-13", 1363.0)]))
    with pytest.raises(si.SolarForcingError, match="gap"):
        series.at(datetime(2024, 5, 11, 12))


def test_bad_rows_and_units_are_refused(tmp_path):
    with pytest.raises(si.SolarForcingError, match="1300-1400"):
        si.read_series(_csv(tmp_path, [("2024-05-10", 1.361)]))
    with pytest.raises(si.SolarForcingError, match="repeat"):
        si.read_series([_csv(tmp_path, [("2024-05-10", 1361.0)])] * 2)
    with pytest.raises(si.SolarForcingError, match="does not exist"):
        si.read_series(tmp_path / "missing.csv")


def test_noaa_cdr_netcdf_layout(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    path = tmp_path / "tsi_v03r00_daily_s20240101_e20241231_c20250221.nc"
    start = (datetime(2024, 5, 10) - datetime(1610, 1, 1)).days
    with netCDF4.Dataset(path, "w") as data:
        data.createDimension("time", 3)
        data.createDimension("bounds", 2)
        t = data.createVariable("time", "f8", ("time",))
        t.units = "days since 1610-01-01 00:00:00"
        t[:] = start + np.arange(3)
        b = data.createVariable("time_bnds", "f8", ("time", "bounds"))
        b[:] = np.stack([start + np.arange(3), start + 1 + np.arange(3)], axis=1)
        v = data.createVariable("TSI", "f4", ("time",))
        v[:] = [1360.5, 1361.5, 1362.5]
    series = si.read_series(path)
    assert series.at(datetime(2024, 5, 11, 12)) == pytest.approx(1361.5)
    assert series.at(datetime(2024, 5, 11, 0)) == pytest.approx(1361.0)


def test_nrlssi2_fetch_takes_the_newest_listed_year_file_once(tmp_path):
    import io
    calls = []
    listing = ('<a href="tsi_v03r00_daily_s20240101_e20241231_c20250101.nc">'
               '<a href="tsi_v03r00_daily_s20240101_e20241231_c20250221.nc">')

    class Reply(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def opener(url):
        calls.append(url)
        return Reply(listing.encode() if url.endswith("/") else b"payload")

    paths = si.fetch_nrlssi2([2024], cache=tmp_path, opener=opener)
    assert [p.name for p in paths] == ["tsi_v03r00_daily_s20240101_e20241231_c20250221.nc"]
    assert paths[0].read_bytes() == b"payload"
    again = si.fetch_nrlssi2([2024], cache=tmp_path, opener=opener)
    assert again == paths and len(calls) == 2           # cached: no second download
    with pytest.raises(si.SolarForcingError, match="no complete NOAA CDR"):
        si.fetch_nrlssi2([2031], cache=tmp_path, opener=opener)


# --- the factor and the scaling -----------------------------------------------------

def test_modes_and_for_run_check_the_whole_window(tmp_path):
    path = _csv(tmp_path, [("2024-05-10", 1361.0), ("2024-05-11", 1361.0)])
    cfg = _cfg(solar_tsi_opt=1, solar_tsi_source=str(path), run_seconds=6 * 3600.0)
    sv = si.SolarVariability.for_run(cfg, datetime(2024, 5, 10, 6))
    assert sv.factor(0.0) == pytest.approx(1361.0 / 1370.0)
    rel = si.SolarVariability.for_run(dataclasses.replace(cfg, solar_tsi_mode="relative"),
                                      datetime(2024, 5, 10, 6))
    assert rel.factor(0.0) == pytest.approx(1361.0 / 1360.8)
    with pytest.raises(si.SolarForcingError, match="outside"):
        si.SolarVariability.for_run(dataclasses.replace(cfg, run_seconds=3 * 86400.0),
                                    datetime(2024, 5, 10, 6))
    assert si.SolarVariability.for_run(_cfg(), datetime(2024, 5, 10)) is None


def test_scale_moves_every_shortwave_output_and_no_longwave(tmp_path):
    from types import SimpleNamespace
    sv = si.SolarVariability(si.read_series(_csv(tmp_path, [("2024-05-10", 1360.0)])),
                             datetime(2024, 5, 10, 12))
    f = np.float32(1360.0 / 1370.0)
    rng = np.random.default_rng(3)
    a = lambda *shape: rng.uniform(1, 2, shape).astype(np.float32)
    lw, sw = a(3, 2, 2), a(3, 2, 2)
    result = SimpleNamespace(
        rthratenlw=lw.copy(), rthratensw=sw.copy(), rthraten_total=lw + sw,
        swdown=a(2, 2), gsw=a(2, 2), swddir=None, swddif=a(2, 2), glw=a(2, 2), olr=a(2, 2),
        boundary_fluxes={"swupt": a(2, 2), "lwupt": a(2, 2)},
        flux_diagnostics={"SWDNB": a(2, 2), "LWDNB": a(2, 2)},
        spectral_fluxes=None, spectral_surface=None)
    before = {k: (v.copy() if isinstance(v, np.ndarray) else
                  {n: w.copy() for n, w in v.items()} if isinstance(v, dict) else v)
              for k, v in vars(result).items()}
    sv.scale(result, 0.0)
    assert np.array_equal(result.rthratensw, before["rthratensw"] * f)
    for name in ("swdown", "gsw", "swddif"):
        assert np.array_equal(getattr(result, name), before[name] * f), name
    assert result.swddir is None
    for name in ("rthratenlw", "glw", "olr"):
        assert np.array_equal(getattr(result, name), before[name]), name
    assert np.array_equal(result.boundary_fluxes["swupt"], before["boundary_fluxes"]["swupt"] * f)
    assert np.array_equal(result.boundary_fluxes["lwupt"], before["boundary_fluxes"]["lwupt"])
    assert np.array_equal(result.flux_diagnostics["SWDNB"], before["flux_diagnostics"]["SWDNB"] * f)
    assert np.array_equal(result.flux_diagnostics["LWDNB"], before["flux_diagnostics"]["LWDNB"])
    np.testing.assert_allclose(result.rthraten_total, before["rthratenlw"] + before["rthratensw"] * f,
                               rtol=2e-7, atol=0)


def test_the_default_driver_path_imports_and_runs_nothing_new():
    """Off is no new code: the hook is a None check, the package is never imported."""
    from woof.core import physics
    src = inspect.getsource(physics.PhysicsDriver)
    assert 'if int(getattr(cfg, "solar_tsi_opt", 0) or 0) and radiation is not None:' in src
    assert 'if getattr(self, "solar_variability", None) is not None:' in src
    assert "import woof.spacewx" not in inspect.getsource(physics).split("class PhysicsDriver")[0]


def test_checkpoint_echo_omits_the_off_keys():
    from woof.io import restart
    values = dataclasses.asdict(_cfg())
    restart._drop_default_off_run_keys(values)
    assert not {"solar_tsi_opt", "solar_tsi_source", "solar_tsi_mode"} & set(values)
    values = dataclasses.asdict(_cfg(solar_tsi_opt=1, solar_tsi_source="t.csv"))
    restart._drop_default_off_run_keys(values)
    assert values["solar_tsi_opt"] == 1 and values["solar_tsi_source"] == "t.csv"


# --- on the card --------------------------------------------------------------------

def _driver(tmp_path, **overrides):
    from test_physics_driver import _base_config
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced
    from woof.core.physics import initialize_physics

    cfg = _base_config(ra_physics=0, ra_lw_physics=4, ra_sw_physics=4,
                       sf_surface_physics=2, sf_sfclay_physics=91, bl_pbl_physics=1,
                       run_seconds=600.0, **overrides)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: 298.0 + 0.004 * np.asarray(z),
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(cfg, coord, base,
                                lambda z: 0.010 * np.exp(-np.asarray(z, np.float64) / 2400.0))
    driver = initialize_physics(
        state, cfg, radiation_start_time=datetime(2011, 4, 26, 17),
        radiation_latitude=np.full((cfg.ny, cfg.nx), 33.8),
        radiation_longitude=np.full((cfg.ny, cfg.nx), -87.29))
    return driver, state, cfg


@pytest.mark.gpu
@requires_gpu
def test_rrtmg_radiation_call_carries_the_tsi_factor(tmp_path):
    import cupy as cp
    from woof.core.physics import _prepare_atmosphere

    tsi = _csv(tmp_path, [("2011-04-26", 1361.25)])
    off, state, cfg = _driver(tmp_path)
    assert off.solar_variability is None
    on, state_on, cfg_on = _driver(tmp_path, solar_tsi_opt=1, solar_tsi_source=str(tsi))
    assert on.solar_variability is not None
    for driver, st, c in ((off, state, cfg), (on, state_on, cfg_on)):
        driver._run_radiation(_prepare_atmosphere(st), st, c)
    f = np.float32(1361.25 / 1370.0)
    sw_off, sw_on = cp.asnumpy(off.rthratensw), cp.asnumpy(on.rthratensw)
    assert np.abs(sw_off).max() > 0, "the call must be in daylight"
    assert np.array_equal(sw_on, sw_off * f)
    assert np.array_equal(cp.asnumpy(on.rthratenlw), cp.asnumpy(off.rthratenlw))
    down_off, down_on = cp.asnumpy(off.fields["swdown"]), cp.asnumpy(on.fields["swdown"])
    assert down_off.min() > 100.0
    assert np.array_equal(down_on, down_off * f)
    assert np.array_equal(cp.asnumpy(on.fields["glw"]), cp.asnumpy(off.fields["glw"]))


# --- the doors ----------------------------------------------------------------------

def test_experiment_toml_shared_value_reaches_every_domain(tmp_path):
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "tests"))
    from test_gf_closure_choice import _grell_tree_toml
    from woof.experiment import load_experiment
    path = tmp_path / "tree.toml"
    shared = ('ra_physics = 4\nsolar_tsi_opt = 1\nsolar_tsi_source = "nrlssi2"\n'
              'solar_tsi_mode = "relative"\n')
    path.write_text(_grell_tree_toml(shared=shared), encoding="utf-8")
    got = [(d.run.solar_tsi_opt, d.run.solar_tsi_source, d.run.solar_tsi_mode)
           for d in load_experiment(path).domains]
    assert got == [(1, "nrlssi2", "relative")] * 2


def test_hex_generic_seam_reaches_the_keys():
    import json
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    argv = [sys.executable, "-m", "woof.cli", "hex", "physics", "resolve",
            "--physics-suite", "arwen_mpas_column", "--mesh", "x1.40962",
            "--physics", "solar_tsi_opt=1", "--physics", "solar_tsi_source=nrlssi2",
            "--physics", "solar_tsi_mode=relative"]
    run = subprocess.run(argv, cwd=root, capture_output=True, text=True, check=False)
    got = json.loads(run.stdout[run.stdout.index("{"):])
    assert got["resolved"] is True, got
    assert got["engine_physics"]["solar_tsi_opt"] == 1
    assert got["engine_physics"]["solar_tsi_source"] == "nrlssi2"
    assert got["engine_physics"]["solar_tsi_mode"] == "relative"
