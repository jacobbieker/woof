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

import _numerics_child
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


def test_band_groups_bound_each_solves_wavelengths():
    from types import SimpleNamespace
    from woof.radtran import driver
    iwvl = {0: [1, 2], 1: [2, 3], 2: [4]}
    rep = SimpleNamespace(band=lambda b: SimpleNamespace(iwvl=iwvl[b]))
    assert driver.band_groups(rep, [0, 1, 2], 10, 10 ** 9) == [[0, 1, 2]]
    assert driver.band_groups(rep, [0, 1, 2], 10, 30) == [[0, 1], [2]]
    # a band over the bound still solves, alone
    assert driver.band_groups(rep, [0, 1, 2], 10, 1) == [[0], [1], [2]]


def _grouped_and_whole(monkeypatch):
    from woof.core.physics import _prepare_atmosphere
    from woof.radtran import model
    cfg = _cfg(radtran_wavelengths=(450.0, 500.0, 760.5), radtran_umu=(0.6, -0.8),
               radtran_phi=(0.0, 90.0), radtran_interval_s=30.0)
    state, driver = _driver(cfg)
    atmos = _prepare_atmosphere(state)
    nz = cfg.nz
    cols = lambda a, n: np.asarray(cp.asnumpy(a), dtype=np.float64).reshape(n, -1).T
    qv = cols(atmos["qv"], nz)
    kw = dict(z_w_m=cols(atmos["z_interface"], nz + 1), p_w_pa=cols(atmos["p_interface"], nz + 1),
              t_mass=cols(atmos["temperature"], nz), qv_mass=qv,
              umu0=driver.radtran.coszen(float(state.elapsed_seconds), "wrf_480").reshape(-1),
              albedo=np.clip(cp.asnumpy(driver.fields["albedo"]).reshape(-1), 0, 1),
              qc_mass=cols(atmos["qc"], nz), qi_mass=cols(atmos["qi"], nz),
              rho_dry=cols(atmos["rho"], nz) / (1 + qv), day_of_year=START.timetuple().tm_yday)
    out = {}
    for label, budget in (("whole", str(1 << 40)), ("grouped", "1")):
        monkeypatch.setenv("WOOF_RADTRAN_GROUP_BYTES", budget)
        out[label] = model.compute(driver.radtran.diag, **kw)
    return out


@requires_gpu
@pytest.mark.parametrize("mode_threads", [None, 0])
def test_band_groups_keep_the_words(monkeypatch, mode_threads):
    """One solve per band (the bound at 1 byte) gives the words of one solve
    for every band, on both device paths (radiances over Fourier-mode threads
    and one thread per problem): the problems are independent and the path is
    chosen from the whole call's problem count."""
    _need_data()
    from woof.radtran import solver
    if mode_threads is not None:
        monkeypatch.setattr(solver, "MODE_THREADS", mode_threads)
    out = _grouped_and_whole(monkeypatch)
    whole, grouped = out["whole"], out["grouped"]
    assert set(whole) == set(grouped) and "rad" in whole
    assert np.max(whole["edir"]) > 0.1
    for key, value in whole.items():
        assert np.array_equal(np.asarray(value).view(np.uint64), np.asarray(grouped[key]).view(np.uint64)), key


@pytest.mark.parametrize(*_numerics_child.MODES)
def test_band_groups_keep_the_words_in_every_numerics_mode(numerics, exact):
    _numerics_child.run(__file__, ["test_band_groups_keep_the_words"], numerics, exact)


def _synthetic_columns(ncol, nz=12, seed=5):
    rng = np.random.default_rng(seed)
    zs = rng.uniform(0, 2500, ncol)
    eta = np.linspace(0, 1, nz + 1) ** 1.3
    z_w = zs[:, None] + eta[None] * (20000 - zs[:, None]) + rng.uniform(0, 50, (ncol, 1)) * eta[None]
    p_w = 101325 * np.exp(-z_w / 7500)
    t = np.maximum(290 - 3.25e-3 * (z_w[:, 1:] + z_w[:, :-1]), 215) + rng.normal(0, 1, (ncol, nz))
    qv = 0.01 * np.exp(-0.25 * (z_w[:, 1:] + z_w[:, :-1]) / 2500)
    qc = np.where(rng.uniform(size=qv.shape) < 0.2, rng.uniform(0, 5e-4, qv.shape), 0.0)
    qi = np.where(rng.uniform(size=qv.shape) < 0.1, rng.uniform(0, 2e-4, qv.shape), 0.0)
    # The highest model top is a NIGHT column: dropping it must not move the
    # AFGL levels the day columns receive.
    mu0 = np.where(rng.uniform(size=ncol) < 0.5, rng.uniform(0.05, 0.95, ncol), -rng.uniform(0, 0.9, ncol))
    top = int(np.argmax(z_w[:, -1]))
    mu0[top] = -0.3
    z_w[top, -1] += 1500.0
    return dict(z_w_m=z_w, p_w_pa=p_w, t_mass=t, qv_mass=qv, umu0=mu0,
                albedo=rng.uniform(0.05, 0.4, ncol), qc_mass=qc, qi_mass=qi,
                rho_dry=p_w[:, 1:] / (287.0 * t), day_of_year=172)


@requires_gpu
@pytest.mark.parametrize("aerosol", ["none", "continental_average"])
def test_day_column_subset_keeps_the_words(monkeypatch, aerosol):
    """Preparing only the day columns (model.SUBSET_DAY_COLUMNS) gives every
    output word of preparing all of them, night columns zero, with the
    highest model top on a night column."""
    _need_data()
    from woof.radtran import model
    diag = model.Diagnostic(wavelengths=(450.0, 760.5), umu=(0.6, -0.8), phi=(0.0, 90.0), aerosol=aerosol)
    kw = _synthetic_columns(64)
    out = {}
    for subset in (False, True):
        monkeypatch.setattr(model, "SUBSET_DAY_COLUMNS", subset)
        out[subset] = model.compute(diag, **kw)
    night = kw["umu0"] <= 0
    assert set(out[False]) == set(out[True]) and "rad" in out[True]
    assert np.max(out[True]["edir"]) > 0.1 and not np.any(out[True]["edn"][..., night])
    for key, value in out[False].items():
        assert np.array_equal(np.asarray(value).view(np.uint64), np.asarray(out[True][key]).view(np.uint64)), key


@pytest.mark.gpu
@pytest.mark.parametrize(*_numerics_child.MODES)
def test_day_column_subset_keeps_the_words_in_every_numerics_mode(numerics, exact):
    _numerics_child.run(__file__, ["test_day_column_subset_keeps_the_words"], numerics, exact)


def test_model_columns_top_km_selects_the_wider_sets_levels():
    from woof.radtran import atmosphere
    _need_data()
    kw = _synthetic_columns(8)
    z = kw["z_w_m"][:, ::-1] * 1e-3
    p = kw["p_w_pa"][:, ::-1] * 1e-2
    t = np.repeat(kw["t_mass"][:, :1], z.shape[1], axis=1)
    whole = atmosphere.model_columns(z, p, t)
    part = atmosphere.model_columns(z[:3], p[:3], t[:3], top_km=float(z[:, 0].max()))
    assert part.z.shape[1] == whole.z.shape[1]
    assert np.array_equal(part.z, whole.z[:3]) and np.array_equal(part.dens["O3"], whole.dens["O3"][:3])
    with pytest.raises(ValueError, match="top_km"):
        atmosphere.model_columns(z, p, t, top_km=float(z[:, 0].max()) - 1.0)
