"""Turbine icing at hub height (``woof.energy.icing_turbine``).

The physics is graded against values worked by hand from the documented
formulas (written out here as plain arithmetic, not by calling the module's
helpers), and the plumbing end to end: a forecast through ``woof energy
rating --products turbine-icing``; a fake-sampled plan and a real synthetic
wrfout through ``woof energy extract`` and then ``rating``, read back with
the ``forecast.v1`` loader; and the gridded xarray path on synthetic
datasets, which must agree with the site path.  No network; the GPU test is
marked ``gpu``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")
xr = pytest.importorskip("xarray")

from woof.energy import extract, icing_turbine as IT, products as P
from woof.energy.contracts import (FORECAST_VARIABLES, Plan, PlanDomain,
                                   Site, SiteSet, dump_plan, dump_sites,
                                   file_ref)

from test_energy_products import _parse, open_nc, write_forecast

KELVIN = 273.15


def _mu_air(t_k):
    return 1.458e-6 * t_k ** 1.5 / (t_k + 110.4)


def _h(diameter, speed, t_k, rho):
    """Makkonen h = 0.032 Re^0.85 k_a / D, transcribed by hand."""
    tc = t_k - KELVIN
    re = rho * speed * diameter / _mu_air(t_k)
    k_air = 2.424e-2 + 7.477e-5 * tc - 4.407e-9 * tc ** 2
    return 0.032 * re ** 0.85 * k_air / diameter


# ------------------------------------------------------------ droplet spectrum


def test_thompson_shape_rounds_like_fortran_nint():
    n = np.array([1e8, 3e8, 2e9, 4e8, 1e6, 1e10, 0.0])
    # 1e9/N = 10, 3.33, 0.5 (NINT -> 1), 2.5 (-> 3), 1000, 0.1, undefined
    assert IT.thompson_cloud_shape(n).tolist() == [12, 5, 3, 5, 15, 2, 15]


def test_droplet_mvd_hand_value():
    # N = 1.25e8 m-3 -> mu = nint(8) + 2 = 10; w = 0.25 g m-3
    lam = (math.pi * 1000.0 / 6.0 * 11 * 12 * 13 * 1.25e8 / 2.5e-4) ** (1 / 3)
    expected = (10 + 3.67) / lam
    assert expected == pytest.approx(17.85e-6, abs=0.01e-6)
    got = IT.droplet_mvd(np.array([2.5e-4, 0.0, 2.5e-4]),
                         np.array([1.25e8, 1.25e8, 0.0]))
    assert got[0] == pytest.approx(expected, rel=1e-12)
    assert np.isnan(got[1]) and np.isnan(got[2])
    # very few, very large droplets are clipped to the documented ceiling
    assert IT.droplet_mvd(np.array([1e-3]), np.array([1e6]))[0] == 50e-6


def test_collision_efficiency_matches_the_conductor_icing_fit():
    d = np.array([0.003, 0.03, 0.1, 0.3])
    v = np.array([10.0, 10.0, 50.0, 3.0])
    t = np.full(4, 268.0)
    rho = np.full(4, 1.3)
    expected = P.finstad_collision_efficiency(d, v, t, rho, 20e-6)
    got = IT.collision_efficiency(d, v, t, rho, np.full(4, 20e-6))
    np.testing.assert_allclose(got, expected, rtol=1e-14)
    assert IT.collision_efficiency(np.array([0.03]), np.array([0.0]),
                                   np.array([268.0]), np.array([1.3]),
                                   np.array([20e-6]))[0] == 0.0


# ------------------------------------------------------------- blade proxy


def test_blade_relative_speed_hand_values():
    speed, running = IT.blade_relative_speed(
        np.array([8.0, 12.0, 2.0, 30.0, np.nan]))
    # 8 m/s: tip 7 * 8 = 56, section 0.85 * 56 = 47.6, axial 8 * 2/3
    assert speed[0] == pytest.approx(math.hypot(47.6, 16.0 / 3.0), rel=1e-12)
    assert speed[0] == pytest.approx(47.8979, abs=1e-4)
    # 12 m/s: tip 84 capped at 80 -> 68 at the section, axial 8
    assert speed[1] == pytest.approx(math.hypot(68.0, 8.0), rel=1e-12)
    # parked below cut-in and above cut-out: the hub wind
    assert speed[2] == 2.0 and speed[3] == 30.0
    assert np.isnan(speed[4])
    assert running.tolist() == [True, True, False, False, False]


def test_blade_proxy_refuses_nonsense():
    with pytest.raises(IT.TurbineIcingRefused):
        IT.BladeProxy(radial_fraction=1.5)
    with pytest.raises(IT.TurbineIcingRefused):
        IT.BladeProxy(cut_in_m_s=10.0, cut_out_m_s=5.0)
    with pytest.raises(IT.TurbineIcingRefused):
        IT.IcingLossModel(trace_kg_m=2.0, stop_kg_m=1.0)


# ------------------------------------------------------------- heat balance


def test_wet_growth_frozen_fraction_hand_value():
    t, v, d, rho, flux = 272.15, 10.0, 0.03, 1.29, 1e-3
    h = _h(d, v, t, rho)
    balance = (h * math.pi * d * 1.0 + flux * 4218.0 * 1.0
               - h * math.pi * d * 0.79 * 100.0 / (2 * 1004.0)
               - flux * 100.0 / 2.0)
    n = balance / (flux * 3.34e5)
    assert 0.0 < n < 1.0
    got_balance, got_h = IT.surface_heat_balance(
        np.array([flux]), d, np.array([v]), np.array([t]), np.array([rho]))
    assert got_h[0] == pytest.approx(h, rel=1e-12)
    assert IT.freezing_fraction(got_balance, np.array([flux]))[0] == \
        pytest.approx(n, rel=1e-12)


def test_dry_growth_single_interval_hand_value():
    t, v, d, rho, w = 263.15, 10.0, 0.03, 1.3, 2e-4
    shape = (2, 1)
    result = IT.accrete(
        time_seconds=[0.0, 900.0], air_temp_k=np.full(shape, t),
        speed=np.full(shape, v), cloud_lwc=np.full(shape, w), rain_lwc=None,
        mvd_m=np.full(shape, 20e-6), air_density=np.full(shape, rho),
        diameter_m=d)
    a1 = P.finstad_collision_efficiency(np.array([d]), np.array([v]),
                                        np.array([t]), np.array([rho]))[0]
    assert result["mass"][0, 0] == 0.0
    assert result["rate"][1, 0] == pytest.approx(a1 * w * v * d, rel=1e-12)
    assert result["mass"][1, 0] == pytest.approx(a1 * w * v * d * 900.0,
                                                 rel=1e-12)
    assert result["rime"][1, 0] == 1
    # the rime volume (500 kg m-3) is the end state
    assert result["volume_end"][0] == pytest.approx(
        result["mass"][1, 0] / 500.0, rel=1e-12)


def test_melting_removes_ice_at_the_heat_surplus():
    t, v, rho, m0 = 275.15, 5.0, 1.27, 0.1
    d = math.sqrt(0.03 ** 2 + 4.0 * (m0 / 500.0) / math.pi)
    h = _h(d, v, t, rho)
    melt = h * math.pi * d * (2.0 + 0.79 * v ** 2 / (2 * 1004.0)) / 3.34e5
    shape = (2, 1)
    result = IT.accrete(
        time_seconds=[0.0, 900.0], air_temp_k=np.full(shape, t),
        speed=np.full(shape, v), cloud_lwc=np.zeros(shape), rain_lwc=None,
        mvd_m=np.full(shape, np.nan), air_density=np.full(shape, rho),
        diameter_m=0.03, initial_mass=m0)
    assert result["mass"][0, 0] == m0
    assert result["removal"][1, 0] == pytest.approx(melt, rel=1e-12)
    assert result["mass"][1, 0] == pytest.approx(m0 - melt * 900.0,
                                                 rel=1e-12)
    # a long warm spell clears it, never below zero
    long = IT.accrete(
        time_seconds=[0.0, 864000.0], air_temp_k=np.full(shape, t),
        speed=np.full(shape, v), cloud_lwc=np.zeros(shape), rain_lwc=None,
        mvd_m=np.full(shape, np.nan), air_density=np.full(shape, rho),
        diameter_m=0.03, initial_mass=m0)
    assert long["mass"][1, 0] == 0.0 and long["volume_end"][0] == 0.0


def test_sublimation_in_dry_cold_air_hand_value():
    t, v, rho, p, rh, m0 = 263.15, 5.0, 1.19, 9e4, 50.0, 0.1
    d = math.sqrt(0.03 ** 2 + 4.0 * (m0 / 500.0) / math.pi)
    e_si = 611.2 * math.exp(22.46 * -10.0 / (-10.0 + 272.62))
    e_sw = 611.2 * math.exp(17.67 * -10.0 / (-10.0 + 243.5))
    rate = _h(d, v, t, rho) * math.pi * d * 0.622 * (
        e_si - 0.5 * e_sw) / (1004.0 * p)
    shape = (2, 1)
    result = IT.accrete(
        time_seconds=[0.0, 3600.0], air_temp_k=np.full(shape, t),
        speed=np.full(shape, v), cloud_lwc=np.zeros(shape), rain_lwc=None,
        mvd_m=np.full(shape, np.nan), air_density=np.full(shape, rho),
        diameter_m=0.03, air_pressure=np.full(shape, p),
        vapour_pressure=np.full(shape, 0.5 * e_sw), initial_mass=m0)
    assert result["removal"][1, 0] == pytest.approx(rate, rel=1e-12)
    assert result["mass"][1, 0] == pytest.approx(m0 - rate * 3600.0,
                                                 rel=1e-12)


def test_unknown_weather_makes_ice_unknown_from_then_on():
    shape = (4, 1)
    t = np.full(shape, 263.15)
    t[1, 0] = np.nan
    result = IT.accrete(
        time_seconds=[0.0, 900.0, 1800.0, 2700.0], air_temp_k=t,
        speed=np.full(shape, 8.0), cloud_lwc=np.full(shape, 2e-4),
        rain_lwc=None, mvd_m=np.full(shape, 20e-6),
        air_density=np.full(shape, 1.3), diameter_m=0.03)
    assert result["mass"][0, 0] == 0.0
    assert np.isnan(result["mass"][1:, 0]).all()


# ---------------------------------------------------------- classifications


def test_production_loss_piecewise():
    m = np.array([0.0, 0.005, 0.01, 0.5, 0.999, 1.0, 3.0, np.nan])
    got = IT.production_loss_fraction(m)
    np.testing.assert_allclose(got, [0, 0, 0.005, 0.25, 0.4995, 1, 1, np.nan])


def test_iea_ice_class_bounds():
    f = np.array([0.0, 0.005, 0.0051, 0.03, 0.031, 0.05, 0.1, 0.1001, np.nan])
    assert IT.iea_ice_class(f).tolist() == [1, 1, 2, 2, 3, 3, 4, 5, -1]


def test_iso12494_class_matches_the_conductor_product():
    masses = np.array([0.0, 0.0005, 0.01, 0.5, 0.51, 0.9, 1.6, 2.81, 50.0,
                       50.1, np.nan])
    assert IT.iso12494_class(masses).tolist() == \
        P.iso12494_class(masses).tolist()


def _kernel(n_time=24, **overrides):
    shape = (n_time, 3)
    inputs = dict(
        time_seconds=np.arange(n_time) * 900.0,
        air_temperature=np.array([268.15, 272.65, 275.0])[None, :]
        * np.ones(shape),
        wind_speed=np.full(shape, 8.0), air_density=np.full(shape, 1.25),
        cloud_mixing_ratio=np.full(shape, 2e-4),
        air_pressure=np.full(shape, 9e4),
        relative_humidity=np.full(shape, 100.0),
        power_curve=P.load_power_curve())
    inputs.update(overrides)
    return inputs, IT.turbine_icing_arrays(**inputs)


def test_kernel_flags_fractions_and_loss():
    _, r = _kernel()
    # LWC 0.25 g m-3 everywhere; only the two sub-zero columns are icing
    np.testing.assert_allclose(r["lwc"], 2.5e-4)
    assert (r["met_icing"] == np.array([1, 1, 0])).all()
    np.testing.assert_allclose(r["met_icing_fraction"], [1, 1, 0])
    assert r["iea_class"].tolist() == [5, 5, 1]
    # default droplet number 1e8 m-3 -> mu 12
    lam = (math.pi * 1000 / 6 * 13 * 14 * 15 * 1e8 / 2.5e-4) ** (1 / 3)
    np.testing.assert_allclose(r["mvd"], 15.67 / lam, rtol=1e-12)
    # at -0.5 degC the blade's aerodynamic heating (0.79 v^2 / 2 c_p ~ 0.9 K)
    # keeps it ice-free while the still reference collector ices
    assert r["blade_mass"][-1, 1] == 0.0
    assert r["reference_mass"][-1, 1] > 0.0
    assert r["blade_mass"][-1, 0] > r["reference_mass"][-1, 0]
    assert np.all(np.diff(r["reference_mass"][:, 0]) > 0)
    assert r["instrumental_icing"][0, 0] == 0
    assert r["instrumental_icing"][-1, 0] == 1
    np.testing.assert_allclose(
        r["loss_fraction"], IT.production_loss_fraction(r["blade_mass"]))
    # energy-weighted loss: the power curve is flat in time here
    np.testing.assert_allclose(r["energy_loss_fraction"][:2],
                               r["loss_fraction"][:, :2].mean(axis=0))
    assert r["energy_loss_fraction"][2] == 0.0
    assert r["rotor_icing_fraction"][2] == 0.0


def test_kernel_uses_the_droplet_number_when_given():
    n_time = 4
    _, r = _kernel(n_time, cloud_number=np.full((n_time, 3), 1e8))
    np.testing.assert_allclose(
        r["mvd"], IT.droplet_mvd(np.full((n_time, 3), 2.5e-4),
                                 np.full((n_time, 3), 1.25e8)))


def test_rain_adds_liquid_water_and_glaze():
    n_time = 8
    shape = (n_time, 3)
    _, dry = _kernel(n_time, cloud_mixing_ratio=np.zeros(shape))
    _, wet = _kernel(n_time, cloud_mixing_ratio=np.zeros(shape),
                     rain_mixing_ratio=np.full(shape, 3e-4))
    assert np.all(dry["reference_mass"] == 0.0)
    assert np.all(wet["reference_mass"][-1, :2] > 0.0)
    assert wet["reference_mass"][-1, 2] == 0.0
    np.testing.assert_allclose(wet["lwc"], 3e-4 * 1.25)


def test_chunked_hindcast_continues_exactly():
    inputs, whole = _kernel(9)
    first_in = {k: (v[:5] if isinstance(v, np.ndarray) and v.ndim == 2
                    else v) for k, v in inputs.items()}
    first_in["time_seconds"] = inputs["time_seconds"][:5]
    first = IT.turbine_icing_arrays(**first_in)
    # the next chunk starts at the previous chunk's last time
    second_in = {k: (v[4:] if isinstance(v, np.ndarray) and v.ndim == 2
                     else v) for k, v in inputs.items()}
    second_in["time_seconds"] = inputs["time_seconds"][4:]
    second_in["initial_state"] = {
        key.replace("_end", ""): first[key] for key in IT.STATE_OUTPUTS}
    second = IT.turbine_icing_arrays(**second_in)
    for key in ("reference_mass", "blade_mass", "loss_fraction"):
        np.testing.assert_allclose(second[key], whole[key][4:], rtol=1e-12,
                                   err_msg=key)


def test_kernel_refuses_mismatched_shapes_and_times():
    with pytest.raises(IT.TurbineIcingRefused, match="shapes"):
        _kernel(4, wind_speed=np.ones((3, 3)))
    with pytest.raises(IT.TurbineIcingRefused, match="increasing"):
        _kernel(3, time_seconds=np.array([0.0, 0.0, 900.0]))


# ---------------------------------------------------- woof energy rating


SITES = [
    dict(site_id="T1", kind="turbine", lat=51.7, lon=-3.6, hub_height_m=80.0,
         capacity_mw=3.0),
    dict(site_id="T2", kind="turbine", lat=51.7, lon=-3.6, capacity_mw=3.0),
    dict(site_id="L1", kind="line_sample", lat=51.6, lon=-3.4,
         bearing_deg=90.0, voltage_kv=400.0),
    dict(site_id="W1", kind="plant", lat=51.8, lon=-3.5, hub_height_m=100.0,
         capacity_mw=30.0),
]
COLD = {"air_temperature": 265.15, "wind_speed": 8.0,
        "cloud_liquid_mixing_ratio": 2e-4, "air_density": 1.25,
        "air_pressure": 90000.0}


def test_rating_writes_turbine_icing_for_turbines_only(tmp_path):
    fc = write_forecast(tmp_path / "fc.nc", sites=SITES, values=COLD)
    out = P.compute_products(fc, output=tmp_path / "p.nc",
                             products=("turbine-icing",))
    with open_nc(out) as ds:
        assert ds.schema == "woof-energy.products.v1"
        for key, (name, units, _) in IT.SERIES_OUTPUTS.items():
            assert ds[name].dimensions == ("time", "site"), name
        for key, (name, units, _) in IT.SUMMARY_OUTPUTS.items():
            assert ds[name].dimensions == ("site",), name
        lwc = ds["turbine_lwc"][:]
        np.testing.assert_allclose(lwc[:, [0, 3]], 2.5e-4)
        assert np.isnan(lwc[:, [1, 2]]).all()
        assert (ds["turbine_meteorological_icing"][:, [1, 2]] == -1).all()
        assert (ds["turbine_meteorological_icing"][:, [0, 3]] == 1).all()
        assert ds["turbine_iea_ice_class"][:].tolist() == [5, -1, -1, 5]
        assert "default N_c" in ds.notes
        assert "no hub_height_m" in ds.notes
        assert "IEA ice class is defined on a year" in ds.notes
        assert "Thompson" in ds["turbine_droplet_mvd"].method
        assert ds["turbine_icing_energy_loss_fraction"].power_curve == \
            "generic_iec_class_ii"
        # T1 at 80 m between the 50 and 100 m levels of a uniform forecast
        # is the kernel run on the same numbers
        expected = IT.turbine_icing_arrays(
            time_seconds=np.arange(24) * 3600.0,
            air_temperature=np.full((24, 1), 265.15),
            wind_speed=np.full((24, 1), 8.0),
            air_density=np.full((24, 1), 1.25),
            cloud_mixing_ratio=np.full((24, 1), 2e-4),
            rain_mixing_ratio=np.zeros((24, 1)),
            air_pressure=np.full((24, 1), 9e4),
            power_curve=P.load_power_curve())
        np.testing.assert_allclose(ds["turbine_blade_ice_mass"][:, 0],
                                   expected["blade_mass"][:, 0], rtol=1e-12)
        np.testing.assert_allclose(ds["turbine_reference_ice_mass"][:, 0],
                                   expected["reference_mass"][:, 0],
                                   rtol=1e-12)


def test_rating_uses_the_forecast_droplet_number(tmp_path):
    values = dict(COLD, cloud_number_mixing_ratio=1e8)
    fc = write_forecast(tmp_path / "fc.nc", sites=SITES[:1], values=values)
    out = P.compute_products(fc, output=tmp_path / "p.nc",
                             products=("turbine-icing",))
    with open_nc(out) as ds:
        mvd = ds["turbine_droplet_mvd"][:, 0]
        assert "default N_c" not in ds.notes
        assert "QNCLOUD" in ds["turbine_droplet_mvd"].droplet_number
    np.testing.assert_allclose(mvd, IT.droplet_mvd(np.array([2.5e-4]),
                                                   np.array([1.25e8]))[0])


def test_rating_refuses_turbine_icing_without_hub_heights(tmp_path):
    fc = write_forecast(tmp_path / "fc.nc", sites=SITES, values=COLD,
                        drop_coords=("hub_height_m",))
    with pytest.raises(P.ProductsRefused, match="hub_height_m"):
        P.compute_products(fc, output=tmp_path / "p.nc",
                           products=("turbine-icing",))


def test_rating_skips_turbine_icing_without_turbines(tmp_path):
    fc = write_forecast(tmp_path / "fc.nc", sites=SITES[2:3], values=COLD)
    out = P.compute_products(fc, output=tmp_path / "p.nc",
                             products=("icing", "turbine-icing"))
    with open_nc(out) as ds:
        assert "turbine-icing: skipped" in ds.notes
        assert "turbine_lwc" not in ds.variables


# ------------------------------------------- extract -> rating end to end


def _rating_cli(forecast: Path, output: Path, capsys) -> dict:
    args = _parse(["energy", "rating", str(forecast), "-o", str(output),
                   "--products", "turbine-icing,wind-power"])
    assert args.func(args) == 0
    return json.loads(capsys.readouterr().out)


def test_extract_then_rating_with_a_fake_sampler(tmp_path, monkeypatch,
                                                 capsys):
    from test_energy_extract import FakeSampler, _args, make_plan
    from woof.energy import sample, sample_hex

    sampler = FakeSampler(theta=265.0)
    for module, name in ((sample, "sample_wrfout"),
                         (sample_hex, "sample_mpas")):
        monkeypatch.setattr(module, name, sampler.sample)
        monkeypatch.setattr(module, "available_variables",
                            sampler.available_variables)
    plan = make_plan(tmp_path)
    forecast = tmp_path / "forecast.nc"
    assert extract.main(_args(plan, forecast)) == 0
    capsys.readouterr()
    assert "QNCLOUD" in sampler.calls[0]["profile"]
    loaded = P.read_forecast(forecast)
    assert loaded.variables["cloud_number_mixing_ratio"].shape == (
        4, 33, 4)
    np.testing.assert_allclose(loaded.variables["cloud_number_mixing_ratio"],
                               1e8)
    summary = _rating_cli(forecast, tmp_path / "products.nc", capsys)
    assert summary["products"] == ["turbine-icing", "wind-power"]
    assert "turbine_blade_ice_mass" in summary["variables"]
    with open_nc(tmp_path / "products.nc") as ds:
        kind = list(ds["kind"][:])
        turbines = [i for i, k in enumerate(kind) if k == "turbine"]
        assert len(turbines) == 4
        assert (ds["turbine_meteorological_icing"][:, turbines] == 1).all()
        assert np.isfinite(ds["turbine_droplet_mvd"][:, turbines]).all()
        others = [i for i in range(len(kind)) if i not in turbines]
        assert (ds["turbine_iea_ice_class"][others] == -1).all()


sample_tests = pytest.importorskip("test_energy_sample")


@sample_tests.needs_library
def test_extract_then_rating_on_a_synthetic_thompson_wrfout(tmp_path,
                                                            capsys):
    """A real wrfout through the Rust sampler, extract, rating, loaders."""

    from woof.cli import build_parser

    run_dir = tmp_path / "runs" / "tile0"
    run_dir.mkdir(parents=True)
    wrfout = run_dir / "wrfout_d01_2026-01-01_00:00:00"
    times = ["2026-01-01_00:00:00", "2026-01-01_00:15:00",
             "2026-01-01_00:30:00"]
    sample_tests.write_wrfout(wrfout, times)
    nz, ny, nx = sample_tests.NZ, sample_tests.NY, sample_tests.NX
    with netCDF4.Dataset(wrfout, "a") as ds:
        for n in range(len(times)):
            ds["T"][n] = np.full((nz, ny, nx), -30.0)       # theta 270 K
            ds["P"][n] = np.zeros((nz, ny, nx))
            ds["QVAPOR"][n] = np.full((nz, ny, nx), 3e-3)  # saturated
            ds["U"][n] = np.full((nz, ny, nx + 1), 8.0)
            ds["V"][n] = np.zeros((nz, ny + 1, nx))
            ds["QCLOUD"][n] = np.full((nz, ny, nx), 3e-4)
            ds["QRAIN"][n] = np.zeros((nz, ny, nx))
            ds["QNCLOUD"][n] = np.full((nz, ny, nx), 1e8)
    lat, lon = sample_tests._sites([(4.5, 4.5), (6.2, 3.3)])
    sites = SiteSet(sites=[
        Site(site_id=f"T{i}", asset_id=f"wf/{i}", kind="turbine",
             lat=float(lat[i]), lon=float(lon[i]), hub_height_m=90.0,
             capacity_mw=4.0) for i in range(2)],
        heights_m=(60.0, 90.0, 150.0))
    sites_path = dump_sites(sites, tmp_path / "sites.json")
    ring = ((-3.5, 51.5), (-3.3, 51.5), (-3.3, 51.7), (-3.5, 51.7),
            (-3.5, 51.5))
    plan = Plan(topology="wrf-tiles", dx_m=100.0, start="2026-01-01T00",
                hours=0.5, domains=[PlanDomain(
                    domain_id="tile0", topology="wrf-tiles", role="child",
                    dx_m=100.0, run_dir="runs/tile0",
                    output_glob="wrfout_d01_*", footprint=ring,
                    config="tile0.toml", grid_id=1, site_ids=("T0", "T1"))],
                sites_ref=file_ref(sites_path, relative_to=tmp_path))
    plan_path = dump_plan(plan, tmp_path / "plan.json")
    forecast = tmp_path / "forecast.nc"
    args = build_parser().parse_args(
        ["energy", "extract", str(plan_path), "-o", str(forecast)])
    assert args.func(args) == 0
    capsys.readouterr()
    loaded = P.read_forecast(forecast)
    np.testing.assert_allclose(loaded.variables["cloud_number_mixing_ratio"],
                               1e8, rtol=1e-6)
    summary = _rating_cli(forecast, tmp_path / "products.nc", capsys)
    assert "turbine-icing" in summary["products"]
    rho = loaded.variables["air_density"][:, :, 1]          # 90 m
    with open_nc(tmp_path / "products.nc") as ds:
        np.testing.assert_allclose(ds["turbine_lwc"][:], 3e-4 * rho,
                                   rtol=1e-6)
        np.testing.assert_allclose(
            ds["turbine_droplet_mvd"][:],
            IT.droplet_mvd(3e-4 * rho, 1e8 * rho), rtol=1e-6)
        assert (ds["turbine_meteorological_icing"][:] == 1).all()
        assert np.all(ds["turbine_reference_ice_mass"][-1] > 0)
        assert np.all(ds["turbine_blade_ice_mass"][-1] > 0)
        wind = loaded.variables["wind_speed"][:, :, 1]
        np.testing.assert_allclose(wind, 8.0, rtol=1e-3)
        np.testing.assert_allclose(ds["turbine_blade_relative_speed"][:],
                                   np.hypot(0.85 * 7.0 * wind, wind * 2 / 3),
                                   rtol=1e-9)

    # the gridded path on the same file agrees with the site path at the
    # sampled heights' columns: same hub fields give the same icing
    grid = xr.open_dataset(wrfout)
    out = IT.turbine_icing_dataset(grid, hub_height_m=90.0)
    assert out.attrs["schema"] == IT.TURBINE_ICING_SCHEMA
    assert out["turbine_lwc"].dims == ("Time", "south_north", "west_east")
    assert out["Time"].values[1] == np.datetime64("2026-01-01T00:15:00")
    assert (out["turbine_meteorological_icing"] == 1).all()
    hub = IT.wrfout_hub_fields(grid, 90.0)
    j, i = 4, 5
    np.testing.assert_allclose(out["turbine_lwc"][:, j, i],
                               3e-4 * hub["air_density"][:, j, i],
                               rtol=1e-6)


# --------------------------------------------------------------- gridded


def _grid_dataset(n_time=6, height=False):
    times = np.datetime64("2026-01-01T00:00") + np.arange(n_time) * \
        np.timedelta64(15, "m")
    t = np.array([[268.15, 272.65], [275.0, 263.15]])
    base = {
        "air_temperature": t, "wind_speed": np.array([[8.0, 12.0],
                                                      [5.0, 2.0]]),
        "air_density": np.full((2, 2), 1.25),
        "cloud_liquid_mixing_ratio": np.full((2, 2), 2e-4),
        "air_pressure": np.full((2, 2), 9e4),
        "relative_humidity": np.full((2, 2), 95.0),
    }
    data = {}
    for name, field in base.items():
        values = np.broadcast_to(field, (n_time, 2, 2)).copy()
        if height:
            # linear in height for everything but the wind; 90 m is hub
            values = np.stack([values, values], axis=1)
            if name == "wind_speed":
                values[:, 0] *= 0.8
                values[:, 1] *= 1.1
            else:
                values[:, 1] += 1.0 if name == "air_temperature" else 0.0
            data[name] = (("time", "height", "y", "x"), values)
        else:
            data[name] = (("time", "y", "x"), values)
    coords = {"time": times, "y": [0, 1], "x": [0, 1]}
    if height:
        coords["height"] = [60.0, 120.0]
    return xr.Dataset(data, coords=coords)


def test_gridded_matches_the_site_kernel():
    ds = _grid_dataset()
    out = IT.turbine_icing_dataset(ds)
    assert out.attrs["schema"] == IT.TURBINE_ICING_SCHEMA
    assert out["turbine_blade_ice_mass"].dims == ("time", "y", "x")
    assert out["turbine_iea_ice_class"].dims == ("y", "x")
    assert out["turbine_meteorological_icing"].dtype == np.int8
    flat = {name: ds[name].values.reshape(6, 4) for name in ds.data_vars}
    expected = IT.turbine_icing_arrays(
        time_seconds=np.arange(6) * 900.0,
        air_temperature=flat["air_temperature"],
        wind_speed=flat["wind_speed"], air_density=flat["air_density"],
        cloud_mixing_ratio=flat["cloud_liquid_mixing_ratio"],
        air_pressure=flat["air_pressure"],
        relative_humidity=flat["relative_humidity"],
        power_curve=P.load_power_curve())
    for key, (name, _, _) in {**IT.SERIES_OUTPUTS,
                              **IT.SUMMARY_OUTPUTS}.items():
        got = out[name].values
        want = np.asarray(expected[key], dtype=np.float64)
        np.testing.assert_allclose(got.reshape(want.shape), want,
                                   rtol=1e-12, err_msg=name)


def test_gridded_height_interpolation_to_the_hub():
    ds = _grid_dataset(height=True)
    hub = IT.height_hub_fields(ds, 90.0, ["air_temperature", "wind_speed"])
    # linear: halfway between 60 and 120 m
    np.testing.assert_allclose(hub["air_temperature"][0],
                               ds["air_temperature"][0, 0] + 0.5)
    # log: weight ln(90/60) / ln(120/60)
    weight = math.log(1.5) / math.log(2.0)
    lo, hi = ds["wind_speed"][0, 0], ds["wind_speed"][0, 1]
    np.testing.assert_allclose(hub["wind_speed"][0], lo + (hi - lo) * weight)
    out = IT.turbine_icing_dataset(ds, hub_height_m=90.0)
    assert out.attrs["hub_height_m"] == 90.0
    with pytest.raises(IT.TurbineIcingRefused, match="hub_height_m"):
        IT.turbine_icing_dataset(ds)


def test_gridded_continues_from_a_previous_chunk():
    ds = _grid_dataset(9)
    whole = IT.turbine_icing_dataset(ds)
    first = IT.turbine_icing_dataset(ds.isel(time=slice(0, 5)))
    second = IT.turbine_icing_dataset(ds.isel(time=slice(4, None)),
                                      initial_state=first)
    np.testing.assert_allclose(second["turbine_blade_ice_mass"].values,
                               whole["turbine_blade_ice_mass"][4:].values,
                               rtol=1e-12)


def test_gridded_refuses_what_it_cannot_do():
    ds = _grid_dataset().drop_vars("cloud_liquid_mixing_ratio")
    with pytest.raises(IT.TurbineIcingRefused, match="cloud_liquid"):
        IT.turbine_icing_dataset(ds)
    with pytest.raises(IT.TurbineIcingRefused, match="device"):
        IT.turbine_icing_dataset(_grid_dataset(), device="tpu")
    with pytest.raises(IT.TurbineIcingRefused, match="time dimension"):
        IT.turbine_icing_dataset(_grid_dataset().rename(time="t"))


def test_gridded_dask_chunks_agree_with_eager():
    pytest.importorskip("dask")
    ds = _grid_dataset()
    eager = IT.turbine_icing_dataset(ds)
    lazy = IT.turbine_icing_dataset(ds.chunk({"x": 1, "time": 2}))
    xr.testing.assert_allclose(lazy.compute(), eager)


def test_wrfout_column_interpolation_hand_value():
    z = np.array([[10.0, 50.0, 130.0], [10.0, 50.0, 80.0]])
    f = np.array([[1.0, 2.0, 4.0], [1.0, 2.0, 3.0]])
    got = IT.interp_columns(f, z, 90.0, axis=-1)
    assert got[0] == pytest.approx(2.0 + 2.0 * 40.0 / 80.0)
    assert np.isnan(got[1])                  # 90 m is above that column
    got = IT.interp_columns(f, z, 90.0, axis=-1, log=True)
    assert got[0] == pytest.approx(
        2.0 + 2.0 * math.log(90 / 50) / math.log(130 / 50))


@pytest.mark.gpu
def test_gpu_kernel_matches_cpu():
    cupy = pytest.importorskip("cupy")
    inputs, cpu = _kernel(6)
    gpu_inputs = {k: (cupy.asarray(v) if isinstance(v, np.ndarray)
                      and v.ndim == 2 else v) for k, v in inputs.items()}
    gpu = IT.turbine_icing_arrays(**gpu_inputs)
    for key in ("blade_mass", "reference_mass", "loss_fraction", "mvd"):
        np.testing.assert_allclose(cupy.asnumpy(gpu[key]), cpu[key],
                                   rtol=1e-10, err_msg=key)
    out = IT.turbine_icing_dataset(_grid_dataset(), device="gpu")
    ref = IT.turbine_icing_dataset(_grid_dataset(), device="cpu")
    xr.testing.assert_allclose(out, ref)


def test_forecast_contract_carries_the_droplet_number():
    dims, units, _ = FORECAST_VARIABLES["cloud_number_mixing_ratio"]
    assert dims == ("time", "site", "height") and units == "kg-1"
    assert extract.REQUIREMENTS["cloud_number_mixing_ratio"] == (
        ("QNCLOUD",),)



# ------------------------------------------------- review regressions


def test_rating_derives_density_from_pressure(tmp_path):
    values = {k: v for k, v in COLD.items() if k != "air_density"}
    fc = write_forecast(tmp_path / "fc.nc", sites=SITES[:1], values=values,
                        drop=("air_density",))
    out = P.compute_products(fc, output=tmp_path / "p.nc",
                             products=("turbine-icing",))
    with open_nc(out) as ds:
        assert "density from air_pressure" in ds.notes
        rho = 90000.0 / (287.05 * 265.15)
        np.testing.assert_allclose(ds["turbine_lwc"][:, 0], 2e-4 * rho,
                                   rtol=1e-12)


def test_zero_droplet_number_at_a_cloudy_point_takes_the_default():
    _, given = _kernel(3, cloud_number=np.zeros((3, 3)))
    _, default = _kernel(3)
    np.testing.assert_allclose(given["mvd"], default["mvd"])


def test_no_sublimation_during_wet_growth():
    # -1 degC, heavy cloud water, RH 95 %: glaze grows; the heat balance
    # carries the evaporation, so nothing extra is removed
    shape = (2, 1)
    common = dict(
        time_seconds=[0.0, 900.0], air_temp_k=np.full(shape, 272.15),
        speed=np.full(shape, 10.0), cloud_lwc=np.full(shape, 3e-3),
        rain_lwc=None, mvd_m=np.full(shape, 20e-6),
        air_density=np.full(shape, 1.29), diameter_m=0.03,
        air_pressure=np.full(shape, 9e4), initial_mass=0.1)
    e_sw = 611.2 * math.exp(17.67 * -1.0 / 242.5)
    result = IT.accrete(vapour_pressure=np.full(shape, 0.95 * e_sw), **common)
    assert result["rime"][1, 0] == 0          # wet growth
    assert result["removal"][1, 0] == 0.0
    assert result["mass"][1, 0] == pytest.approx(
        0.1 + result["rate"][1, 0] * 900.0, rel=1e-12)


def test_gridded_notes_a_hub_outside_the_heights():
    out = IT.turbine_icing_dataset(_grid_dataset(height=True),
                                   hub_height_m=150.0)
    assert "outside the dataset heights" in out.attrs["notes"]
    out = IT.turbine_icing_dataset(_grid_dataset(), hub_height_m=90.0)
    assert "no height dimension" in out.attrs["notes"]


def test_gridded_accepts_numpy_scalars_and_cpu_device():
    ds = _grid_dataset()
    ref = IT.turbine_icing_dataset(ds)
    out = IT.turbine_icing_dataset(ds, device="cpu")
    xr.testing.assert_allclose(out, ref)
    assert IT.interp_columns(np.array([1.0, 2.0]), np.array([10.0, 20.0]),
                             np.float32(15.0), axis=-1) == 1.5
