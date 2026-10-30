"""The woof.radtran spectral diagnostic on WRF domains (RunConfig radtran_*).

* config: off by default, refused where the solver cannot run;
* fields: allocated only when on, with their documented shapes;
* the carrier's words equal woof.radtran.model.compute run on the same
  columns through the host build (the device and host solver words agree,
  tests/test_radtran_kernel_words*.py);
* output-only: the physics tendencies of a step are bit-identical with the
  diagnostic on and off;
* cadence, restart round trip, and the history writer's axes.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime

import numpy as np
import pytest

import cupy as cp  # noqa: F401  (gpu module; the facade on MLX)

from conftest import requires_gpu

START = datetime(2025, 6, 21, 18)


def _need_data():
    from woof.radtran import data
    try:
        data.data_dir()
    except data.LibradtranDataMissing as exc:
        pytest.skip(str(exc))


def _cfg(**kw):
    from woof.config import RunConfig
    base = dict(nx=4, ny=3, nz=12, dx=3000.0, dy=3000.0, ztop=16000.0, dt=6.0, run_seconds=60.0,
                moist=True, mp_physics=1, sf_sfclay_physics=0, sf_surface_physics=0, bl_pbl_physics=0,
                ra_physics=0)
    base.update(kw)
    return RunConfig(**base)


ON = dict(radtran_wavelengths=(500.0, 760.5), radtran_umu=(0.6, -0.8), radtran_phi=(0.0, 90.0),
          radtran_interval_s=30.0)


def test_config_is_off_by_default_and_refuses_what_cannot_run():
    from woof.config import ConfigurationRefused, validate_run_config
    from woof.core.physics_inventory import radtran_fields
    cfg = _cfg()
    assert cfg.radtran_wavelengths == () and radtran_fields(cfg, 3, 4) == {}
    on = validate_run_config(dataclasses.replace(cfg, **ON))
    shapes = radtran_fields(on, 3, 4)
    assert shapes["rt_edir"] == (2, 3, 4) and shapes["rt_eup_toa"] == (2, 3, 4)
    assert shapes["rt_rad"] == (2, 4, 3, 4) and shapes["rt_umu"] == (4,)
    assert shapes["rt_wavelength"] == (2,)
    for bad in (dict(radtran_wavelengths=(50.0,)), dict(radtran_wavelengths=(500.0,), radtran_interval_s=0.0),
                dict(radtran_wavelengths=(500.0,), radtran_solver="twostr", radtran_umu=(0.5,)),
                dict(radtran_wavelengths=(500.0,), radtran_umu=(0.0,)),
                dict(radtran_wavelengths=(500.0,), radtran_mol_abs_param="kato2"),
                dict(radtran_wavelengths=(500.0,), radtran_streams=7),
                dict(radtran_wavelengths=(500.0,), radtran_aerosol="volcanic"),
                dict(radtran_wavelengths=(240.0,), radtran_aerosol="urban"),
                # Thompson aerosol numbers exist only with mp_physics = 28
                dict(radtran_wavelengths=(500.0,), radtran_aerosol="thompson")):
        with pytest.raises(ConfigurationRefused):
            validate_run_config(dataclasses.replace(cfg, **bad))
    assert cfg.radtran_aerosol == "none"
    validate_run_config(dataclasses.replace(cfg, radtran_wavelengths=(500.0,), radtran_aerosol="maritime_clean"))


def test_frame_sink_routes_radtran_fields_by_name():
    from woof.io.frame_sink import frame_dims
    assert frame_dims("RT_EDIR", (12, 3, 4), nx=4, ny=3, nz=12) == ("rt_wavelength", "south_north", "west_east")
    assert frame_dims("RT_RAD", (2, 4, 3, 4), nx=4, ny=3, nz=12) == (
        "rt_wavelength", "rt_direction", "south_north", "west_east")
    assert frame_dims("RT_WAVELENGTH", (2,), nx=4, ny=3, nz=12) == ("rt_wavelength",)
    assert frame_dims("RT_UMU", (4,), nx=4, ny=3, nz=12) == ("rt_direction",)


@pytest.mark.parametrize("engine", ["python", "rust"])
def test_history_writer_names_the_radtran_axes(tmp_path, engine):
    netCDF4 = pytest.importorskip("netCDF4")
    from woof.io.wrfout import WrfoutWriter
    ny, nx, nz = 3, 4, 12
    fields = {"RT_EDIR": np.full((12, ny, nx), 1.5, np.float32),     # 12 == nz: not bottom_top
              "RT_RAD": np.full((12, 2, ny, nx), 0.25, np.float32),
              "RT_WAVELENGTH": np.linspace(400, 900, 12).astype(np.float32),
              "RT_UMU": np.array([0.5, -0.5], np.float32)}
    path = tmp_path / f"wrfout_rt_{engine}"
    try:
        w = WrfoutWriter(path, nx=nx, ny=ny, nz=nz, dx=3000.0, dy=3000.0, engine=engine)
    except (RuntimeError, FileNotFoundError, OSError) as exc:
        pytest.skip(f"wrfout engine {engine} unavailable: {exc}")
    w.write_frame("2025-06-21_18:00:00", fields)
    w.close()
    with netCDF4.Dataset(path) as ds:
        assert ds["RT_EDIR"].dimensions == ("Time", "rt_wavelength", "south_north", "west_east")
        assert ds["RT_RAD"].dimensions == ("Time", "rt_wavelength", "rt_direction", "south_north", "west_east")
        assert ds["RT_WAVELENGTH"].dimensions == ("Time", "rt_wavelength")
        assert ds["RT_UMU"].dimensions == ("Time", "rt_direction")
        assert ds["RT_EDIR"].units == "W m-2 nm-1" and "WOOF.RADTRAN" in ds["RT_EDIR"].description
        assert np.all(np.asarray(ds["RT_RAD"][0]) == np.float32(0.25))


def _driver(cfg):
    from test_physics import _balanced_state
    from woof.core.physics import initialize_physics
    state = _balanced_state(cp, cfg)
    if state.qc is not None:
        # a cloud layer in the middle columns (kg/kg)
        qc = cp.zeros_like(state.qc)
        qc[3:6, 1, 1:3] = 2e-4
        state.qc[...] = qc
    lat = np.linspace(30.0, 40.0, cfg.ny * cfg.nx, dtype=np.float32).reshape(cfg.ny, cfg.nx)
    lon = np.linspace(-110.0, -80.0, cfg.ny * cfg.nx, dtype=np.float32).reshape(cfg.ny, cfg.nx)
    driver = initialize_physics(state, cfg, radiation_start_time=START, radiation_latitude=lat,
                                radiation_longitude=lon)
    return state, driver


@requires_gpu
def test_carrier_words_equal_the_library_on_the_same_columns():
    _need_data()
    from woof.core.physics import _prepare_atmosphere
    from woof.radtran import model
    cfg = _cfg(**ON)
    state, driver = _driver(cfg)
    atmos = _prepare_atmosphere(state)
    driver.radtran.run(driver, state, cfg, atmos)
    rt = {k: cp.asnumpy(driver.fields[k]) for k in ("rt_edir", "rt_edn", "rt_eup", "rt_eup_toa", "rt_rad")}
    assert np.all(np.isfinite(rt["rt_edir"])) and rt["rt_edir"].max() > 0.1
    assert np.array_equal(cp.asnumpy(driver.fields["rt_wavelength"]), np.float32([500.0, 760.5]))
    # the same columns through the library and the host build
    nz = cfg.nz
    cols = lambda a, n: np.asarray(cp.asnumpy(a), dtype=np.float64).reshape(n, -1).T
    qv = cols(atmos["qv"], nz)
    mu0 = driver.radtran.coszen(float(state.elapsed_seconds), "wrf_480").reshape(-1)
    ref = model.compute(driver.radtran.diag, z_w_m=cols(atmos["z_interface"], nz + 1),
                        p_w_pa=cols(atmos["p_interface"], nz + 1), t_mass=cols(atmos["temperature"], nz),
                        qv_mass=qv, umu0=mu0, albedo=np.clip(cp.asnumpy(driver.fields["albedo"]).reshape(-1), 0, 1),
                        qc_mass=cols(atmos["qc"], nz), qi_mass=cols(atmos["qi"], nz),
                        rho_dry=cols(atmos["rho"], nz) / (1 + qv), day_of_year=START.timetuple().tm_yday,
                        backend="host")
    ny, nx = cfg.ny, cfg.nx
    for name, key in (("rt_edir", "edir"), ("rt_edn", "edn"), ("rt_eup", "eup"), ("rt_eup_toa", "eup_toa")):
        assert np.array_equal(rt[name], np.float32(ref[key].reshape(-1, ny, nx))), name
    assert np.array_equal(rt["rt_rad"], np.float32(ref["rad"].reshape(2, 4, ny, nx)))
    # the cloudy columns see less direct and more diffuse light at 500 nm
    clear, cloudy = (0, 1), (1, 1)
    assert rt["rt_edir"][0][cloudy] < 0.5 * rt["rt_edir"][0][clear]


@requires_gpu
def test_opac_aerosol_carrier_equals_the_library_and_dims_the_beam():
    _need_data()
    from woof.core.physics import _prepare_atmosphere
    from woof.radtran import model
    cfg = _cfg(**ON, radtran_aerosol="continental_average")
    state, driver = _driver(cfg)
    atmos = _prepare_atmosphere(state)
    driver.radtran.run(driver, state, cfg, atmos)
    rt = {k: cp.asnumpy(driver.fields[k]) for k in ("rt_edir", "rt_edn", "rt_rad")}
    nz = cfg.nz
    cols = lambda a, n: np.asarray(cp.asnumpy(a), dtype=np.float64).reshape(n, -1).T
    qv = cols(atmos["qv"], nz)
    mu0 = driver.radtran.coszen(float(state.elapsed_seconds), "wrf_480").reshape(-1)
    kw = dict(z_w_m=cols(atmos["z_interface"], nz + 1), p_w_pa=cols(atmos["p_interface"], nz + 1),
              t_mass=cols(atmos["temperature"], nz), qv_mass=qv, umu0=mu0,
              albedo=np.clip(cp.asnumpy(driver.fields["albedo"]).reshape(-1), 0, 1),
              qc_mass=cols(atmos["qc"], nz), qi_mass=cols(atmos["qi"], nz),
              rho_dry=cols(atmos["rho"], nz) / (1 + qv), day_of_year=START.timetuple().tm_yday, backend="host")
    ref = model.compute(driver.radtran.diag, **kw)
    clean = model.compute(dataclasses.replace(driver.radtran.diag, aerosol="none"), **kw)
    ny, nx = cfg.ny, cfg.nx
    assert np.array_equal(rt["rt_edir"], np.float32(ref["edir"].reshape(-1, ny, nx)))
    assert np.array_equal(rt["rt_edn"], np.float32(ref["edn"].reshape(-1, ny, nx)))
    assert np.array_equal(rt["rt_rad"], np.float32(ref["rad"].reshape(2, 4, ny, nx)))
    # continental aerosol (tau ~ 0.2 at 500 nm) dims the beam everywhere and adds
    # diffuse light under clear sky (column 0)
    assert np.all(ref["edir"] < clean["edir"]) and np.all(ref["edn"][:, 0] > clean["edn"][:, 0])


@requires_gpu
def test_the_diagnostic_moves_no_other_word_and_keeps_its_cadence():
    _need_data()
    off_cfg = _cfg()
    on_cfg = _cfg(**ON)
    s_off, d_off = _driver(off_cfg)
    s_on, d_on = _driver(on_cfg)
    calls = []
    real = d_on.radtran.run
    d_on.radtran.run = lambda *a: (calls.append(float(a[1].elapsed_seconds)), real(*a))
    for step in range(7):
        t_off = d_off.compute(s_off, off_cfg)
        t_on = d_on.compute(s_on, on_cfg)
        for name in dataclasses.fields(t_off):
            a, b = getattr(t_off, name.name), getattr(t_on, name.name)
            if hasattr(a, "shape"):
                assert cp.asnumpy(a).tobytes() == cp.asnumpy(b).tobytes(), name.name
        for key, value in d_off.fields.items():
            assert cp.asnumpy(value).tobytes() == cp.asnumpy(d_on.fields[key]).tobytes(), key
        for s in (s_off, s_on):
            s.elapsed_seconds += off_cfg.dt
    # dt = 6 s, interval 30 s: the first step and the step crossing 30 s
    assert calls == [0.0, 24.0]


@requires_gpu
def test_restart_carries_the_radtran_fields(tmp_path):
    _need_data()
    from woof.core.physics import _prepare_atmosphere
    from woof.io import restart
    cfg = _cfg(**ON)
    state, driver = _driver(cfg)
    driver.radtran.run(driver, state, cfg, _prepare_atmosphere(state))
    path = restart.write_restart(tmp_path / "rt.npz", state, cfg)
    s2, d2 = _driver(cfg)
    restart.restore_restart(path, s2, cfg)
    for name in ("rt_edir", "rt_edn", "rt_eup", "rt_eup_toa", "rt_rad", "rt_wavelength", "rt_umu"):
        assert cp.asnumpy(d2.fields[name]).tobytes() == cp.asnumpy(driver.fields[name]).tobytes(), name
    assert d2.radtran is not None and d2.radtran is not driver.radtran
