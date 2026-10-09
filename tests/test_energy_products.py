"""``woof energy rating``: line rating, icing, wind and PV power products.

Every test builds a small synthetic ``woof-energy.forecast.v1`` netCDF in a
temporary directory with :func:`write_forecast`; nothing touches the network
or a GPU.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

from woof.energy import products as P
from woof.energy.contracts import FORECAST_SCHEMA, FORECAST_VARIABLES

T0 = "2026-06-21 00:00:00"

DEFAULT_SITES = [
    dict(site_id="L400", kind="line_sample", lat=51.6, lon=-3.4,
         bearing_deg=90.0, voltage_kv=400.0),
    dict(site_id="L132", kind="line_sample", lat=51.6, lon=-3.3,
         bearing_deg=45.0, voltage_kv=132.0),
    dict(site_id="T1", kind="turbine", lat=51.7, lon=-3.6, hub_height_m=80.0,
         capacity_mw=3.0),
    dict(site_id="PV1", kind="pv", lat=51.5, lon=-3.2, capacity_mw=10.0),
]

DEFAULT_VALUES = {
    "air_temperature": 293.15,
    "wind_speed": 5.0,
    "wind_attack_angle": 90.0,
    "air_density": 1.2,
    "air_pressure": 100000.0,
    "cloud_liquid_mixing_ratio": 0.0,
    "rain_mixing_ratio": 0.0,
    "t2": 293.15,
    "wind_speed_10m": 3.0,
    "ghi": 0.0,
    "dni": 0.0,
    "dhi": 0.0,
}


def write_forecast(path: Path, *, sites=None, hours=None, heights=(10.0, 50.0,
                   100.0), values=None, drop=(), schema=FORECAST_SCHEMA,
                   drop_coords=()) -> Path:
    """Write a synthetic forecast.v1 file; ``values`` override variables.

    A value may be a scalar, a (time,) array, or a full-shape array.
    """

    sites = DEFAULT_SITES if sites is None else sites
    hours = list(range(24)) if hours is None else list(hours)
    merged = dict(DEFAULT_VALUES)
    merged.update(values or {})
    nt, ns, nh = len(hours), len(sites), len(heights)
    with netCDF4.Dataset(str(path), "w") as ds:
        ds.createDimension("time", nt)
        ds.createDimension("site", ns)
        ds.createDimension("height", nh)
        t = ds.createVariable("time", "f8", ("time",))
        t.units = f"hours since {T0}"
        t.calendar = "standard"
        t[:] = hours
        h = ds.createVariable("height", "f8", ("height",))
        h[:] = heights
        for name in ("site_id", "kind", "asset_id"):
            if name in drop_coords:
                continue
            var = ds.createVariable(name, str, ("site",))
            for i, site in enumerate(sites):
                var[i] = site.get(name, site["site_id"] if name == "asset_id"
                                  else "")
        for name in ("lat", "lon", "bearing_deg", "terrain_height",
                     "voltage_kv", "hub_height_m", "capacity_mw", "inside"):
            if name in drop_coords:
                continue
            default = {"terrain_height": 100.0, "inside": 1.0}.get(name,
                                                                  np.nan)
            var = ds.createVariable(name, "f8", ("site",))
            var[:] = [site.get(name, default) for site in sites]
        for name, value in merged.items():
            if name in drop:
                continue
            dims = FORECAST_VARIABLES[name][0]
            shape = (nt, ns, nh) if len(dims) == 3 else (nt, ns)
            arr = np.asarray(value, dtype=float)
            if arr.ndim == 1 and arr.size == nt:
                arr = arr.reshape((nt,) + (1,) * (len(shape) - 1))
            var = ds.createVariable(name, "f8", dims)
            var[:] = np.broadcast_to(arr, shape)
        ds.schema = schema
        ds.plan_sha256 = "p" * 64
        ds.sites_sha256 = "s" * 64
    return path


def open_nc(path) -> "netCDF4.Dataset":
    ds = netCDF4.Dataset(str(path))
    ds.set_auto_mask(False)
    return ds


def run(tmp_path, **kwargs):
    fc = write_forecast(tmp_path / "fc.nc", **kwargs.pop("forecast", {}))
    out = P.compute_products(fc, output=tmp_path / "products.nc", **kwargs)
    return open_nc(out)


def site_index(ds, site_id):
    return list(ds.variables["site_id"][:]).index(site_id)


# --------------------------------------------------------------------------
# IEEE 738


def test_ieee738_annex_b_drake_reference():
    """IEEE 738-2012 Annex B: Drake, 40 degC air, 100 degC, 0.61 m/s -> 1025 A."""

    hc, zc = P.ieee738_solar_angles(30.0, 161, 11.0)
    assert hc == pytest.approx(74.9, abs=0.1)
    assert zc == pytest.approx(114.0, abs=0.2)
    qse = P.ieee738_clear_sky_flux(hc, 0.0)
    assert qse == pytest.approx(1027, rel=0.01)
    drake = P.load_conductor_table().get("acsr_drake")
    assert drake.resistance(100.0) == pytest.approx(9.390e-5, rel=1e-3)
    theta = np.arccos(np.cos(np.radians(hc)) * np.cos(np.radians(zc - 90.0)))
    qs = drake.absorptivity * qse * np.sin(theta) * drake.diameter_m
    result = P.ieee738_steady_state(
        air_temp_c=40.0, wind_speed=0.61, attack_angle_deg=90.0,
        elevation_m=0.0, solar_heat_w_m=qs, diameter_m=drake.diameter_m,
        resistance_ohm_m=drake.resistance(100.0), conductor_temp_c=100.0,
        emissivity=drake.emissivity)
    assert float(result["q_c"]) == pytest.approx(81.93, rel=0.01)
    assert float(result["q_r"]) == pytest.approx(39.1, rel=0.01)
    assert float(result["q_s"]) == pytest.approx(22.45, rel=0.02)
    assert float(result["amps"]) == pytest.approx(1025.0, rel=0.01)
    assert int(result["limiting_term"]) == P.LIMIT_FORCED_LOW


def test_ieee738_wind_angle_and_no_headroom():
    common = dict(air_temp_c=20.0, wind_speed=2.0, elevation_m=0.0,
                  solar_heat_w_m=0.0, diameter_m=0.028,
                  resistance_ohm_m=9e-5, conductor_temp_c=75.0,
                  emissivity=0.8)
    perpendicular = P.ieee738_steady_state(attack_angle_deg=90.0, **common)
    parallel = P.ieee738_steady_state(attack_angle_deg=0.0, **common)
    assert float(parallel["amps"]) < float(perpendicular["amps"])
    hot = P.ieee738_steady_state(attack_angle_deg=90.0,
                                 **{**common, "air_temp_c": 80.0})
    assert float(hot["amps"]) == 0.0
    assert int(hot["limiting_term"]) == P.LIMIT_NO_HEADROOM
    calm = P.ieee738_steady_state(attack_angle_deg=90.0,
                                  **{**common, "wind_speed": 0.0})
    assert int(calm["limiting_term"]) == P.LIMIT_NATURAL


def test_dlr_from_file_auto_conductor_and_mva(tmp_path):
    with run(tmp_path, products=("dlr",),
             forecast={"drop": ("ghi", "dni", "dhi")}) as ds:
        amps = ds["rating_amps"][:]
        mva = ds["rating_mva"][:]
        i400, i132 = site_index(ds, "L400"), site_index(ds, "L132")
        assert ds["conductor"][i400] == "aaac_araucaria_twin"
        assert ds["conductor"][i132] == "acsr_lynx"
        assert np.all(np.isfinite(amps[:, [i400, i132]]))
        assert np.all(np.isnan(amps[:, site_index(ds, "T1")]))
        assert np.allclose(mva[:, i400],
                           math.sqrt(3) * 400e3 * amps[:, i400] / 1e6)
        # Night (no sun) rates higher than noon under the clear-sky model.
        assert amps[0, i132] > amps[12, i132]
        assert "clear-atmosphere" in ds["rating_amps"].solar_method
        assert "one circuit" in ds.notes
        assert ds.schema == "woof-energy.products.v1"
        assert len(ds.forecast_sha256) == 64
        assert len(ds.conductor_table_sha256) == 64
        assert ds.plan_sha256 == "p" * 64


def test_dlr_bundle_doubles_rating(tmp_path):
    single = run(tmp_path, products=("dlr",), conductor="aaac_araucaria")
    with single:
        a1 = single["rating_amps"][:, 0].copy()
    (tmp_path / "products.nc").unlink()
    with run(tmp_path, products=("dlr",),
             conductor="aaac_araucaria_twin") as twin:
        assert np.allclose(twin["rating_amps"][:, 0], 2 * a1)


def test_dlr_uses_forecast_irradiance_when_present(tmp_path):
    sun = np.zeros(24)
    sun[9:16] = 700.0
    with run(tmp_path, products=("dlr",), forecast={
            "values": {"ghi": sun, "dni": sun, "dhi": sun * 0.1}}) as ds:
        assert "CIGRE" in ds["rating_amps"].solar_method
        amps = ds["rating_amps"][:, 0]
        assert amps[12] < amps[3]


def test_dlr_auto_unknown_voltage_is_nan_with_note(tmp_path):
    sites = [dict(site_id="L", kind="line_sample", lat=52.0, lon=-1.0,
                  bearing_deg=10.0),
             dict(site_id="D", kind="line_sample", lat=52.0, lon=-1.0,
                  bearing_deg=10.0, voltage_kv=11.0),
             dict(site_id="N", kind="tower", lat=52.0, lon=-1.0,
                  voltage_kv=132.0)]
    with run(tmp_path, products=("dlr",),
             forecast={"sites": sites}) as ds:
        assert np.all(np.isnan(ds["rating_amps"][:]))
        assert "unknown voltage" in ds.notes
        assert "no bearing_deg" in ds.notes


def test_dlr_conductor_height_interpolated_and_nearest_noted(tmp_path):
    table = json.loads(P.DEFAULT_CONDUCTOR_TABLE.read_text())
    table["conductors"]["acsr_lynx"]["height_m"] = 500.0
    path = tmp_path / "table.json"
    path.write_text(json.dumps(table))
    with run(tmp_path, products=("dlr",), conductor="acsr_lynx",
             conductor_table=path) as ds:
        assert "nearest height" in ds.notes


# --------------------------------------------------------------------------
# icing


def test_icing_zero_above_freezing(tmp_path):
    with run(tmp_path, products=("icing",), forecast={"values": {
            "air_temperature": 274.15, "cloud_liquid_mixing_ratio": 5e-4,
            "rain_mixing_ratio": 5e-4}}) as ds:
        assert np.all(ds["ice_mass"][:] == 0.0)
        assert np.all(ds["icing_class"][:] == 0)
        assert np.all(ds["freezing_rain_flag"][:] == 0)


def test_icing_grows_with_liquid_water(tmp_path):
    masses = []
    for lwc in (1e-4, 3e-4, 6e-4):
        out = tmp_path / f"p{lwc}.nc"
        fc = write_forecast(tmp_path / f"fc{lwc}.nc", values={
            "air_temperature": 263.15, "cloud_liquid_mixing_ratio": lwc})
        P.compute_products(fc, output=out, products=("icing",))
        with open_nc(out) as ds:
            mass = ds["ice_mass"][:]
            assert np.all(np.diff(mass, axis=0) >= 0)
            assert mass[0].max() == 0.0
            masses.append(mass[-1])
    assert np.all(masses[0] > 0)
    assert np.all(masses[1] > masses[0]) and np.all(masses[2] > masses[1])


def test_icing_wet_growth_below_dry_growth_rate():
    common = dict(diameter_m=np.array([0.03]), speed=np.array([10.0]),
                  air_density=np.array([1.25]), air_pressure=None)
    flux = np.array([5e-4])
    dry = P.accretion_efficiency(flux, air_temp_k=np.array([263.15]), **common)
    wet = P.accretion_efficiency(flux, air_temp_k=np.array([272.65]), **common)
    warm = P.accretion_efficiency(flux, air_temp_k=np.array([275.0]), **common)
    assert dry[0] == 1.0
    assert 0.0 < wet[0] < 1.0
    assert warm[0] == 0.0


def test_finstad_collision_efficiency_ranges():
    d = np.array([0.003, 0.03, 0.3])
    e = P.finstad_collision_efficiency(d, np.full(3, 10.0), np.full(3, 268.0),
                                       np.full(3, 1.3))
    assert np.all((e >= 0) & (e <= 1))
    assert e[0] > e[1] > e[2]
    assert P.finstad_collision_efficiency(
        np.array([0.03]), np.array([0.0]), np.array([268.0]),
        np.array([1.3]))[0] == 0.0


def test_iso12494_class_thresholds():
    masses = np.array([0.0, 0.0005, 0.01, 0.5, 0.51, 0.9, 0.91, 1.6, 2.8,
                       2.81, 5.0, 8.9, 16.0, 28.0, 50.0, 50.1, np.nan])
    expected = [0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7, 8, 9, 10, -1]
    assert P.iso12494_class(masses).tolist() == expected


def test_freezing_rain_flag_and_glaze(tmp_path):
    with run(tmp_path, products=("icing",), forecast={"values": {
            "air_temperature": 271.15, "rain_mixing_ratio": 3e-4}}) as ds:
        assert np.all(ds["freezing_rain_flag"][:] == 1)
        mass = ds["ice_mass"][-1]
        thick = ds["ice_thickness"][-1]
        assert np.all(mass > 0) and np.all(thick > 0)
        assert "Jones" in ds["ice_mass"].assumptions


# --------------------------------------------------------------------------
# wind power


def _turbine_power(tmp_path, speed, density=1.225, name="w"):
    sites = [dict(site_id="T", kind="turbine", lat=55.0, lon=-3.0,
                  hub_height_m=50.0, capacity_mw=4.0)]
    fc = write_forecast(tmp_path / f"{name}.nc", sites=sites, hours=[0],
                        values={"wind_speed": speed, "air_density": density})
    out = tmp_path / f"{name}-p.nc"
    P.compute_products(fc, output=out, products=("wind-power",))
    with open_nc(out) as ds:
        return (float(ds["wind_power_mw"][0, 0]),
                float(ds["hub_equivalent_wind_speed"][0, 0]))


def test_wind_power_curve_regions(tmp_path):
    assert _turbine_power(tmp_path, 2.5, name="a")[0] == 0.0
    assert _turbine_power(tmp_path, 13.0, name="b")[0] == pytest.approx(4.0)
    assert _turbine_power(tmp_path, 25.0, name="c")[0] == pytest.approx(4.0)
    assert _turbine_power(tmp_path, 26.0, name="d")[0] == 0.0
    mid = _turbine_power(tmp_path, 8.0, name="e")[0]
    assert 0.0 < mid < 4.0


def test_wind_power_density_correction(tmp_path):
    power_ref, veq_ref = _turbine_power(tmp_path, 8.0, 1.225, name="r")
    power_thin, veq_thin = _turbine_power(tmp_path, 8.0, 1.225 * 0.9,
                                          name="t")
    assert veq_ref == pytest.approx(8.0)
    assert veq_thin == pytest.approx(8.0 * 0.9 ** (1 / 3))
    assert power_thin < power_ref


def test_wind_hub_height_log_interpolation(tmp_path):
    sites = [dict(site_id="T", kind="turbine", lat=55.0, lon=-3.0,
                  hub_height_m=30.0, capacity_mw=2.0)]
    speeds = np.array([4.0, 8.0])
    fc = write_forecast(tmp_path / "fc.nc", sites=sites, hours=[0],
                        heights=(10.0, 100.0),
                        values={"wind_speed": speeds[None, None, :]})
    out = P.compute_products(fc, output=tmp_path / "p.nc",
                             products=("wind-power",))
    with open_nc(out) as ds:
        expected = 4.0 + 4.0 * math.log(3.0) / math.log(10.0)
        assert float(ds["hub_wind_speed"][0, 0]) == pytest.approx(expected)


def test_wind_density_from_pressure_and_temperature(tmp_path):
    sites = [dict(site_id="T", kind="turbine", lat=55.0, lon=-3.0,
                  hub_height_m=50.0, capacity_mw=2.0)]
    fc = write_forecast(tmp_path / "fc.nc", sites=sites, hours=[0],
                        drop=("air_density",))
    out = P.compute_products(fc, output=tmp_path / "p.nc",
                             products=("wind-power",))
    with open_nc(out) as ds:
        assert "R_d" in ds.notes
        assert np.isfinite(ds["wind_power_mw"][0, 0])


# --------------------------------------------------------------------------
# PV


def _clear_day():
    hours = np.arange(24)
    # A crude clear June day at 51.5 N: sun from about 04 to 20 UTC.
    shape = np.clip(np.sin(np.pi * (hours - 4) / 16.0), 0.0, None)
    return 850.0 * shape, 900.0 * shape ** 0.3 * (shape > 0), 110.0 * shape


def test_pv_zero_at_night_and_plausible_at_noon(tmp_path):
    ghi, dni, dhi = _clear_day()
    with run(tmp_path, products=("pv-power",), forecast={"values": {
            "ghi": ghi, "dni": dni, "dhi": dhi}}) as ds:
        ipv = site_index(ds, "PV1")
        power = ds["pv_power_mw"][:, ipv]
        assert power[0] == 0.0 and power[23] == 0.0
        assert 6.0 < power[12] < 10.0
        assert np.all(power <= 10.0)
        assert np.all(np.isnan(ds["pv_power_mw"][:, site_index(ds, "T1")]))
        cell = ds["cell_temperature"][12, ipv]
        assert cell > 293.15


def test_pv_erbs_used_and_labelled_without_dni_dhi(tmp_path):
    ghi, _, _ = _clear_day()
    with run(tmp_path, products=("pv-power",), forecast={
            "values": {"ghi": ghi}, "drop": ("dni", "dhi")}) as ds:
        assert "Erbs" in ds["pv_power_mw"].assumptions
        assert "Erbs" in ds.notes
        assert ds["pv_power_mw"][12, site_index(ds, "PV1")] > 3.0


def test_erbs_decomposition_conserves_ghi():
    ghi = np.array([600.0, 100.0, 0.0])
    cosz = np.array([0.8, 0.3, -0.1])
    dni, dhi = P.erbs_decomposition(ghi, cosz, np.array([172.0]))
    assert np.allclose(dni * np.clip(cosz, 0, None) + dhi, ghi)


def test_solar_position_noon_and_night():
    import datetime as dt
    times = [dt.datetime(2026, 6, 21, 12, 0), dt.datetime(2026, 6, 21, 0, 0)]
    cosz, az = P.solar_position(times, np.array([51.5]), np.array([0.0]))
    assert np.degrees(np.arccos(cosz[0, 0])) == pytest.approx(28.1, abs=0.5)
    assert az[0, 0] == pytest.approx(180.0, abs=3.0)
    assert cosz[1, 0] < 0


# --------------------------------------------------------------------------
# refusals and controls


def test_refuses_wrong_schema(tmp_path):
    fc = write_forecast(tmp_path / "fc.nc", schema="woof-energy.forecast.v0")
    with pytest.raises(P.ProductsRefused, match="schema"):
        P.compute_products(fc, output=tmp_path / "p.nc")


def test_refuses_missing_inputs_with_list(tmp_path):
    fc = write_forecast(tmp_path / "fc.nc",
                        drop=("cloud_liquid_mixing_ratio", "ghi"))
    with pytest.raises(P.ProductsRefused) as info:
        P.compute_products(fc, output=tmp_path / "p.nc")
    message = str(info.value)
    assert "icing needs cloud_liquid_mixing_ratio" in message
    assert "pv-power needs ghi" in message
    assert not (tmp_path / "p.nc").exists()


def test_refuses_missing_coordinate(tmp_path):
    fc = write_forecast(tmp_path / "fc.nc", drop_coords=("terrain_height",))
    with pytest.raises(P.ProductsRefused, match="coordinate terrain_height"):
        P.compute_products(fc, output=tmp_path / "p.nc", products=("dlr",))


def test_refuses_auto_without_voltage(tmp_path):
    fc = write_forecast(tmp_path / "fc.nc", drop_coords=("voltage_kv",))
    with pytest.raises(P.ProductsRefused, match="voltage_kv"):
        P.compute_products(fc, output=tmp_path / "p.nc", products=("dlr",))
    out = P.compute_products(fc, output=tmp_path / "p.nc", products=("dlr",),
                             conductor="acsr_zebra")
    with open_nc(out) as ds:
        assert np.all(np.isnan(ds["rating_mva"][:]))
        assert np.isfinite(ds["rating_amps"][0, 0])


def test_refuses_unknown_conductor(tmp_path):
    fc = write_forecast(tmp_path / "fc.nc")
    with pytest.raises(P.ProductsRefused, match="unknown conductor 'Moose'"):
        P.compute_products(fc, output=tmp_path / "p.nc", conductor="Moose")


def test_refuses_bad_conductor_table(tmp_path):
    table = json.loads(P.DEFAULT_CONDUCTOR_TABLE.read_text())
    table["conductors"]["acsr_lynx"]["emissivity"] = 1.5
    path = tmp_path / "table.json"
    path.write_text(json.dumps(table))
    fc = write_forecast(tmp_path / "fc.nc")
    with pytest.raises(P.ProductsRefused, match="emissivity"):
        P.compute_products(fc, output=tmp_path / "p.nc",
                           conductor_table=path)
    path.write_text(json.dumps({"schema": "nope"}))
    with pytest.raises(P.ProductsRefused, match="conductors.v1"):
        P.compute_products(fc, output=tmp_path / "p.nc",
                           conductor_table=path)


def test_refuses_output_over_forecast(tmp_path):
    fc = write_forecast(tmp_path / "fc.nc")
    with pytest.raises(P.ProductsRefused, match="overwrite"):
        P.compute_products(fc, output=fc)


def test_refuses_missing_file(tmp_path):
    with pytest.raises(P.ProductsRefused, match="does not exist"):
        P.compute_products(tmp_path / "nope.nc", output=tmp_path / "p.nc")


def test_skips_product_without_its_sites(tmp_path):
    sites = [dict(site_id="L", kind="line_sample", lat=52.0, lon=-1.0,
                  bearing_deg=10.0, voltage_kv=275.0)]
    with run(tmp_path, forecast={"sites": sites}) as ds:
        assert "pv-power: skipped" in ds.notes
        assert "wind-power: skipped" in ds.notes
        assert "pv_power_mw" not in ds.variables
        assert "rating_amps" in ds.variables
        assert ds.skipped_products == "wind-power,pv-power"


def test_refuses_when_no_product_applies(tmp_path):
    sites = [dict(site_id="S", kind="substation", lat=52.0, lon=-1.0)]
    fc = write_forecast(tmp_path / "fc.nc", sites=sites)
    with pytest.raises(P.ProductsRefused, match="none of"):
        P.compute_products(fc, output=tmp_path / "p.nc",
                           products=("dlr", "pv-power"))


def _parse(argv):
    from woof.energy.cli import register_cli

    parser = argparse.ArgumentParser()
    register_cli(parser.add_subparsers(dest="command"))
    return parser.parse_args(argv)


def test_main_prints_summary_and_refusal(tmp_path, capsys):
    fc = write_forecast(tmp_path / "fc.nc")
    args = _parse(["energy", "rating", str(fc), "-o",
                   str(tmp_path / "p.nc"), "--products", "dlr,wind-power"])
    assert args.func(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["schema"] == "woof-energy.products.v1"
    assert summary["products"] == ["dlr", "wind-power"]
    assert summary["stats"]["rating_amps"]["min"] > 0
    args = _parse(["energy", "rating", str(fc), "-o",
                   str(tmp_path / "p.nc"), "--conductor", "Moose"])
    assert args.func(args) == 2
    assert "unknown conductor" in json.loads(capsys.readouterr().out)["refused"]


def test_char_array_strings_are_read(tmp_path):
    fc = write_forecast(tmp_path / "fc.nc", drop_coords=("kind",))
    with netCDF4.Dataset(str(fc), "a") as ds:
        ds.createDimension("nchar", 12)
        var = ds.createVariable("kind", "S1", ("site", "nchar"))
        var._Encoding = "ascii"
        var[:] = np.array([s["kind"] for s in DEFAULT_SITES], dtype="S12")
    out = P.compute_products(fc, output=tmp_path / "p.nc",
                             products=("wind-power",))
    with open_nc(out) as ds:
        assert np.isfinite(ds["wind_power_mw"][0, 2])


def test_packaged_tables_are_complete():
    table = P.load_conductor_table()
    for key in ("acsr_drake", "acsr_lynx", "acsr_zebra", "aaac_rubus",
                "aaac_araucaria", "aaac_upas"):
        assert key in table.conductors
    assert table.for_voltage(400.0).subconductors == 2
    assert table.for_voltage(132.0).key == "acsr_lynx"
    assert table.for_voltage(33.0) is None
    curve = P.load_power_curve()
    assert curve["cut_in_ms"] == 3.0 and curve["cut_out_ms"] == 25.0
    assert np.interp(12.0, curve["speeds"], curve["fraction"]) == 1.0


def test_icing_unknown_temperature_is_undefined_not_zero(tmp_path):
    temps = np.full(24, 263.15)
    temps[5] = np.nan
    with run(tmp_path, products=("icing",), forecast={"values": {
            "air_temperature": temps,
            "cloud_liquid_mixing_ratio": 2e-4}}) as ds:
        mass = ds["ice_mass"][:]
        assert np.all(mass[:5] >= 0) and np.all(np.isfinite(mass[:5]))
        assert np.all(np.isnan(mass[5:]))
        assert np.all(ds["icing_class"][5:] == -1)


def test_icing_reference_collector_ignores_wind_angle(tmp_path):
    sites = [dict(site_id="D", kind="line_sample", lat=52.0, lon=-1.0,
                  bearing_deg=10.0, voltage_kv=11.0)]
    masses = []
    for angle in (90.0, 5.0):
        out = tmp_path / f"p{angle}.nc"
        fc = write_forecast(tmp_path / f"fc{angle}.nc", sites=sites, values={
            "air_temperature": 263.15, "cloud_liquid_mixing_ratio": 2e-4,
            "wind_attack_angle": angle})
        P.compute_products(fc, output=out, products=("icing",))
        with open_nc(out) as ds:
            masses.append(float(ds["ice_mass"][-1, 0]))
            assert float(ds["icing_collector_diameter"][0]) == 0.03
    assert masses[0] == pytest.approx(masses[1])
    assert masses[0] > 0


def test_icing_on_conductor_uses_normal_wind(tmp_path):
    masses = []
    for angle in (90.0, 5.0):
        out = tmp_path / f"p{angle}.nc"
        fc = write_forecast(tmp_path / f"fc{angle}.nc", values={
            "air_temperature": 263.15, "cloud_liquid_mixing_ratio": 2e-4,
            "wind_attack_angle": angle})
        P.compute_products(fc, output=out, products=("icing",))
        with open_nc(out) as ds:
            masses.append(float(ds["ice_mass"][-1, 0]))
    assert masses[1] < masses[0]


def test_conductor_name_not_checked_when_unused(tmp_path):
    fc = write_forecast(tmp_path / "fc.nc")
    out = P.compute_products(fc, output=tmp_path / "p.nc",
                             products=("wind-power",), conductor="Moose")
    assert out.exists()


def test_refuses_empty_time_and_undecodable_calendar(tmp_path):
    fc = write_forecast(tmp_path / "fc.nc", hours=[])
    with pytest.raises(P.ProductsRefused, match="empty"):
        P.compute_products(fc, output=tmp_path / "p.nc")
    fc = write_forecast(tmp_path / "fc2.nc")
    with netCDF4.Dataset(str(fc), "a") as ds:
        ds["time"].calendar = "360_day"
    with pytest.raises(P.ProductsRefused, match="cannot be decoded"):
        P.compute_products(fc, output=tmp_path / "p.nc")
    assert not (tmp_path / "p.nc.partial").exists()
