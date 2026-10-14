"""``woof energy power``: turbine library, matching, per-turbine and plant
power with wakes.

Synthetic forecasts are written to a temporary directory; nothing touches the
network or a GPU.  The fixture ``assets_windfarm_power.geojson`` holds one
plant per match rung (exact, fuzzy, size, generic) plus a plant known only by
its outline.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

from woof.cli import build_parser
from woof.energy import power_farm as PF
from woof.energy import turbine_library as TL
from woof.energy.contracts import (
    FORECAST_SCHEMA, Asset, AssetCollection, ContractError, load_assets)
from woof.energy.sites import build_sites

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "energy" / "assets_windfarm_power.geojson"
DOC = REPO / "docs" / "energy-hindcast" / "power.md"
GUIDE = REPO / "docs" / "energy-forecasts.md"


# --------------------------------------------------------------------------
# helpers


def write_forecast(path: Path, sites, heights, speed, *, density=1.225,
                   direction=None, extra=None, time_hours=None,
                   schema=FORECAST_SCHEMA, drop=()) -> Path:
    """``sites``: dicts with site_id/asset_id/kind/lat/lon/hub/capacity.

    ``speed`` is (time,) at 100 m with a 1/7 power-law shear, or a full
    (time, site, height) array.
    """

    heights = np.asarray(heights, float)
    speed = np.asarray(speed, float)
    nt = speed.shape[0]
    ns = len(sites)
    if speed.ndim == 1:
        speed = speed[:, None, None] * (heights / 100.0)[None, None, :] \
            ** (1.0 / 7.0)
        speed = np.broadcast_to(speed, (nt, ns, heights.size)).copy()
    hours = np.arange(nt) * 0.25 if time_hours is None else time_hours
    with netCDF4.Dataset(str(path), "w") as ds:
        ds.createDimension("time", nt)
        ds.createDimension("site", ns)
        ds.createDimension("height", heights.size)
        t = ds.createVariable("time", "f8", ("time",))
        t.units = "hours since 2025-01-01 00:00:00"
        t.calendar = "standard"
        t[:] = hours
        ds.createVariable("height", "f8", ("height",))[:] = heights
        for name in ("site_id", "asset_id", "kind"):
            var = ds.createVariable(name, str, ("site",))
            for i, site in enumerate(sites):
                var[i] = site[name]
        for name in ("lat", "lon", "hub_height_m", "capacity_mw"):
            ds.createVariable(name, "f8", ("site",))[:] = [
                np.nan if site.get(name) is None else site[name]
                for site in sites]
        fields = {"wind_speed": speed,
                  "air_density": np.full_like(speed, density)}
        if direction is not None:
            rad = np.radians(np.broadcast_to(
                np.asarray(direction, float).reshape(-1, 1, 1), speed.shape))
            fields["u"] = -speed * np.sin(rad)
            fields["v"] = -speed * np.cos(rad)
        for name, data in fields.items():
            if name in drop:
                continue
            ds.createVariable(name, "f8", ("time", "site", "height"))[:] = data
        for name, data in (extra or {}).items():
            ds.createVariable(name, "f8", ("time", "site"))[:] = data
        ds.schema = schema
    return path


def sites_from_assets(assets) -> list[dict]:
    site_set = build_sites(assets, heights_m=(10.0, 30.0, 60.0, 100.0,
                                              120.0, 150.0))
    return [dict(site_id=s.site_id, asset_id=s.asset_id, kind=s.kind,
                 lat=s.lat, lon=s.lon, hub_height_m=s.hub_height_m,
                 capacity_mw=s.capacity_mw) for s in site_set.sites], \
        site_set.heights_m


def run_cli(argv: list[str], capsys) -> tuple[int, dict]:
    args = build_parser().parse_args(["energy", "power"] + argv)
    code = args.func(args)
    out = capsys.readouterr().out
    return code, json.loads(out)


def single_turbine(asset_id="t1", **tags) -> AssetCollection:
    return AssetCollection(assets=[Asset(
        asset_id, "generator", {"type": "Point", "coordinates": [0.0, 52.0]},
        "synthetic", generator_source="wind", capacity_mw=3.0,
        hub_height_m=100.0, rotor_diameter_m=112.0, tags=tags)])


TURBINE_SITE = dict(site_id="t1", asset_id="t1", kind="turbine", lat=52.0,
                    lon=0.0, hub_height_m=100.0, capacity_mw=3.0)


# --------------------------------------------------------------------------
# the library


def test_library_loads_with_licence_and_valid_curves():
    lib = TL.load_turbine_library()
    assert lib.license == "ODbL-1.0"
    assert "Reiner Lemoine" in lib.attribution
    assert len(lib) >= 150
    measured = [c for c in lib.curves.values() if c.curve_source == "oedb"]
    assert len(measured) >= 60
    for curve in lib.curves.values():
        assert curve.power_kw.max() <= curve.rated_kw * 1.0001
        assert curve.fraction(curve.cut_out_ms) > 0
    # the upstream rating error is corrected from the type suffix
    assert lib.get("GE103/2850").rated_kw == 2850.0
    # offshore machines are flagged
    assert lib.get("V164/8000").offshore is True
    assert lib.get("V112/3000").offshore is False


def test_library_refuses_a_wrong_schema(tmp_path):
    bad = tmp_path / "lib.json"
    bad.write_text(json.dumps({"schema": "nope"}))
    with pytest.raises(TL.TurbineLibraryRefused):
        TL.load_turbine_library(bad)


@pytest.mark.parametrize("key,value", [
    ("cut_out_ms", "x"), ("restart_ms", -1), ("offshore", "yes"),
    ("names", "V1"), ("id", None)])
def test_library_refuses_malformed_fields(tmp_path, key, value):
    document = json.loads(TL.DEFAULT_TURBINE_LIBRARY.read_text())
    document["turbines"][0][key] = value
    path = tmp_path / "lib.json"
    path.write_text(json.dumps(document))
    with pytest.raises(TL.TurbineLibraryRefused):
        TL.load_turbine_library(path)


def test_library_refuses_a_curve_above_its_rating(tmp_path):
    document = json.loads(TL.DEFAULT_TURBINE_LIBRARY.read_text())
    document["turbines"][0]["power_kw"][-1] = \
        document["turbines"][0]["rated_kw"] * 2
    path = tmp_path / "lib.json"
    path.write_text(json.dumps(document))
    with pytest.raises(TL.TurbineLibraryRefused, match="rated_kw"):
        TL.load_turbine_library(path)


@pytest.mark.parametrize("text,prefix,rotor,rated", [
    ("V112-3.0 MW", "v", 112, 3000),
    ("SWT-3.6-107", "swt", 107, 3600),
    ("E-101/3050 E2", "e", 101, 3050),
    ("GE1.5-77", "ge", 77, 1500),
    ("3.4M104", "", 104, 3400),
    ("Haliade-X 12 MW", "haliade", None, 12000),
    ("GE 1.6-82.5", "ge", 82.5, 1600),
    ("AW 3000/116", "aw", 116, 3000),
    ("V27-225", "v", 27, 225),
    ("SG 14-222 DD", "sg", 222, 14000),
    ("MM92", "mm", 92, None),
])
def test_parse_model(text, prefix, rotor, rated):
    key = TL.parse_model(text)
    assert key == TL.ModelKey(prefix, rotor, rated)


@pytest.mark.parametrize("text,kw", [
    ("3 MW", 3000), ("3000 kW", 3000), ("3,000 kW", 3000), ("2.3", 2300),
    ("850", 850), ("25", 25), ("nonsense", None)])
def test_parse_power(text, kw):
    assert TL.parse_power_kw(text) == kw


def test_uswtdb_capacity_is_kw():
    asset = Asset("uswtdb:1", "generator",
                  {"type": "Point", "coordinates": [-100, 40]}, "uswtdb",
                  generator_source="wind", tags={"t_cap": "25.5"})
    assert TL.spec_from_asset(asset).rated_kw == 25.5


@pytest.mark.parametrize("text,cls", [
    ("IEC IIA/IEC S", "II"), ("IEC la", "I"), ("IEC lla +", "II"),
    ("IEC IIIA", "III"), ("S", "S"), ("", None)])
def test_parse_iec_class(text, cls):
    assert TL.parse_iec_class(text) == cls


@pytest.mark.parametrize("spec,quality,curve_id", [
    (TL.TurbineSpec("Vestas", "V112-3.0 MW", 3000, 112), "exact",
     "V112/3000"),
    (TL.TurbineSpec("MHI Vestas Offshore Wind", "V164-8.0 MW"), "exact",
     "V164/8000"),
    (TL.TurbineSpec("GE Wind Energy", "GE1.5-77", 1500), "exact",
     "GE77/1500"),
    (TL.TurbineSpec("Acciona", "AW116/3000"), "exact", "AW116/3000"),
    (TL.TurbineSpec("Siemens Gamesa", "SWT3.6/120 DD"), "fuzzy",
     "SWT120/3600"),
    (TL.TurbineSpec("Enercon", "E-82", 2300), "fuzzy", "E-82/2300"),
    (TL.TurbineSpec(None, None, 3000, 112), "size", "V112/3000"),
    (TL.TurbineSpec(None, "Brand X-1", 2000), "generic", "generic_iec_II"),
    (TL.TurbineSpec(None, None, 8000, offshore=True), "generic",
     "generic_offshore"),
    (TL.TurbineSpec(None, None, 2000, iec_class="III"), "generic",
     "generic_iec_III"),
])
def test_the_matching_ladder(spec, quality, curve_id):
    match = TL.load_turbine_library().match(spec)
    assert match.quality == quality, match.reason
    assert match.curve.curve_id == curve_id
    assert match.reason


def test_a_manufacturer_mismatch_blocks_an_exact_name():
    lib = TL.load_turbine_library()
    match = lib.match(TL.TurbineSpec("Enercon", "V112-3.0 MW", 3000, 112))
    assert match.quality == "size"


def test_generic_with_rotor_uses_the_asset_specific_power():
    lib = TL.load_turbine_library()
    match = lib.match(TL.TurbineSpec(None, None, 5000, 100))  # 637 W/m2
    assert match.quality == "generic"
    assert match.curve.curve_id == "parametric_sp637"
    assert match.curve.rotor_diameter_m == 100


def test_no_rating_is_none():
    match = TL.load_turbine_library().match(TL.TurbineSpec())
    assert match.quality == "none" and match.curve is None


def test_spec_reads_osm_and_uswtdb_tags():
    asset = Asset("osm:node/1", "generator",
                  {"type": "Point", "coordinates": [0, 52]}, "osm",
                  generator_source="wind",
                  tags={"generator:output:electricity": "2.3 MW",
                        "rotor:diameter": "101", "t_manu": "Siemens",
                        "t_model": "SWT-2.3-101", "offshore": "yes"})
    spec = TL.spec_from_asset(asset)
    assert spec == TL.TurbineSpec("Siemens", "SWT-2.3-101", 2300.0, 101.0,
                                  None, None, True, None)


def test_parametric_curve_shape():
    v = np.arange(0, 26, 0.5)
    f = TL.parametric_fraction(v, 300.0)
    assert np.all(f[v < 3] == 0)
    assert np.all(np.diff(f) >= 0)
    assert f[-1] == 1.0
    # rated speed of the cubic for SP 300: ~10.4 m/s; 99 % a little later
    assert 10.5 < v[np.argmax(f >= 0.99)] < 13.0


def test_thrust_coefficient_is_physical():
    curve = TL.load_turbine_library().get("V112/3000")
    v = np.array([0.0, 2.0, 5.0, 8.0, 12.0, 20.0, 25.0])
    ct = curve.thrust_coefficient(v)
    assert np.all((ct >= 0) & (ct <= 8 / 9 + 1e-9))
    assert ct[0] == 0 and ct[1] == 0
    assert 0.75 < ct[3] <= 8 / 9
    assert ct[4] > ct[5] > ct[6]


# --------------------------------------------------------------------------
# per-turbine physics


def test_rews_of_a_uniform_profile_is_the_speed():
    heights = np.array([40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0])
    speed = np.full((2, 1, heights.size), 8.0)
    rews, used = PF.rotor_equivalent_wind_speed(speed, heights,
                                                np.array([100.0]),
                                                np.array([112.0]))
    assert used.all()
    assert np.allclose(rews, 8.0)


def test_rews_by_hand():
    # three heights inside a 100 m rotor at 100 m: bands split at 75/125 m
    heights = np.array([50.0, 100.0, 150.0])
    v = np.array([6.0, 8.0, 10.0])
    rews, used = PF.rotor_equivalent_wind_speed(
        v[None, None, :], heights, np.array([100.0]), np.array([100.0]))
    r = 50.0

    def above(z):
        return r * r * (math.acos(z / r) - z / r * math.sqrt(1 - (z / r) ** 2))

    area = math.pi * r * r
    w = np.array([above(-r) - above(-25.0), above(-25.0) - above(25.0),
                  above(25.0) - above(r)]) / area
    assert w.sum() == pytest.approx(1.0)
    assert rews[0, 0] == pytest.approx(float(np.cbrt((w * v ** 3).sum())))
    _, used = PF.rotor_equivalent_wind_speed(
        v[None, None, :2], heights[:2], np.array([100.0]),
        np.array([100.0]))
    assert not used.any()


def test_rews_ignores_nan_outside_the_rotor():
    heights = np.array([10.0, 60.0, 90.0, 120.0, 150.0, 300.0])
    speed = np.full((1, 1, heights.size), 8.0)
    speed[0, 0, -1] = np.nan
    rews, used = PF.rotor_equivalent_wind_speed(
        speed, heights, np.array([90.0]), np.array([120.0]))
    assert used.all() and rews[0, 0] == pytest.approx(8.0)
    speed[0, 0, 2] = np.nan        # inside the disc: NaN is honest
    rews, _ = PF.rotor_equivalent_wind_speed(
        speed, heights, np.array([90.0]), np.array([120.0]))
    assert np.isnan(rews[0, 0])


def test_plant_keys_by_id_name_polygon_and_tag():
    square = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1],
                                                  [0, 1], [0, 0]]]}
    plant = Asset("osm:way/1", "plant", square, "osm", name="Alta Wind",
                  generator_source="wind")

    def turbine(ident, lon, **tags):
        return Asset(ident, "generator",
                     {"type": "Point", "coordinates": [lon, 0.5]}, "x",
                     generator_source="wind", tags=tags)

    keys = PF.plant_keys([turbine("a", 5, plant="osm:way/1"),
                          turbine("b", 5, p_name="alta wind"),
                          turbine("c", 0.5),
                          turbine("d", 5, p_name="Other"),
                          turbine("e", 5)], [plant])
    assert keys == {"a": "osm:way/1", "b": "osm:way/1", "c": "osm:way/1",
                    "d": "Other", "e": "e"}


def test_running_state_hysteresis():
    speed = np.array([10, 26, 24, 23, 21.5, 24, 26, np.nan, 20])[:, None]
    state = PF.running_state(speed, np.array([25.0]), np.array([22.0]))
    assert state[:, 0].tolist() == [True, False, False, False, True, True,
                                    False, False, True]
    # starting between restart and cut-out: running
    assert PF.running_state(np.array([[23.0]]), np.array([25.0]),
                            np.array([22.0]))[0, 0]


def test_gust_shutdown():
    speed = np.array([[15.0], [15.0], [15.0]])
    gust = np.array([[20.0], [35.0], [25.0]])
    state = PF.running_state(speed, np.array([25.0]), np.array([22.0]),
                             gust=gust, gust_limit=np.array([32.0]))
    assert state[:, 0].tolist() == [True, False, True]


def test_one_turbine_by_hand_with_density_correction(tmp_path, capsys):
    """V112-3.0 at 9 m/s hub wind and 1.10 kg/m3, no wakes and no losses."""

    speeds = np.array([4.0, 9.0, 12.0, 30.0, 23.5, 21.0])
    heights = (60.0, 100.0, 140.0)
    fc = write_forecast(tmp_path / "fc.nc", [TURBINE_SITE], heights,
                        speeds, density=1.10)
    assets = tmp_path / "a.geojson"
    from woof.energy.contracts import dump_assets

    dump_assets(single_turbine(manufacturer="Vestas", model="V112/3000"),
                assets)
    code, summary = run_cli([str(fc), "--assets", str(assets), "--hub-only",
                             "--electrical-losses", "0", "--per-turbine",
                             "-o", str(tmp_path / "p.nc")], capsys)
    assert code == 0, summary
    assert summary["match_quality"] == {"exact": 1}
    doc = PF.load_power(tmp_path / "p.nc")
    curve = TL.load_turbine_library().get("V112/3000")
    veq = speeds * (1.10 / 1.225) ** (1 / 3)
    expected = np.interp(veq, curve.speeds, curve.power_kw) / 1000.0
    expected[3] = 0.0   # 30 m/s: cut-out
    expected[4] = 0.0   # 23.5 m/s: above restart (22), still stopped
    got = doc.data["power_mw"][:, 0]
    assert got == pytest.approx(expected, abs=1e-9)
    assert doc.data["turbine_running"][:, 0].tolist() == [1, 1, 1, 0, 0, 1]
    # hand check at 9 m/s: v_eq = 9 (1.10/1.225)^(1/3) = 8.6828 m/s
    assert veq[1] == pytest.approx(8.6828, abs=1e-4)
    lo = np.searchsorted(curve.speeds, veq[1]) - 1
    w = (veq[1] - curve.speeds[lo]) / (curve.speeds[lo + 1] - curve.speeds[lo])
    by_hand = curve.power_kw[lo] + w * (curve.power_kw[lo + 1]
                                        - curve.power_kw[lo])
    assert got[1] * 1000 == pytest.approx(by_hand)
    assert doc.data["capacity_factor"][1, 0] == pytest.approx(got[1] / 3.0)


# --------------------------------------------------------------------------
# wakes


def _two_v112(spacing_d=5.0):
    curve = TL.load_turbine_library().get("V112/3000")
    x = np.array([0.0, spacing_d * 112.0])
    return x, np.zeros(2), np.full(2, 100.0), [curve, curve], \
        np.full(2, 112.0)


def test_jensen_deficit_by_hand():
    x, y, hub, curves, rotor = _two_v112()
    table = PF.wake_ratio_table(x, y, hub, curves, rotor, model="jensen",
                                sigma_deg=0.0)
    west = list(PF.WAKE_DIRECTIONS_DEG).index(270.0)
    south = list(PF.WAKE_DIRECTIONS_DEG).index(180.0)
    v = list(PF.WAKE_SPEEDS_MS).index(8.0)
    ct = float(curves[0].thrust_coefficient(8.0))
    k = PF.JENSEN_K["onshore"]
    expected = 1 - (1 - math.sqrt(1 - ct)) * (56.0 / (56.0 + k * 560.0)) ** 2
    assert table[west, v, 0] == pytest.approx(1.0)
    assert table[west, v, 1] == pytest.approx(expected)
    assert np.allclose(table[south, v, :], 1.0)
    # from the east the first turbine is the waked one
    east = list(PF.WAKE_DIRECTIONS_DEG).index(90.0)
    assert table[east, v, 0] == pytest.approx(expected)


def test_bastankhah_deficit_is_bounded_and_decays():
    x, y, hub, curves, rotor = _two_v112(4.0)
    near = PF.wake_ratio_table(x, y, hub, curves, rotor, model="bastankhah",
                               sigma_deg=0.0)
    far = PF.wake_ratio_table(x * 3, y, hub, curves, rotor,
                              model="bastankhah", sigma_deg=0.0)
    west = list(PF.WAKE_DIRECTIONS_DEG).index(270.0)
    v = list(PF.WAKE_SPEEDS_MS).index(8.0)
    assert 0.0 < near[west, v, 1] < far[west, v, 1] < 1.0


def test_apply_table_interpolates_direction_and_speed():
    table = np.ones((4, 3, 1))
    table[1, :, 0] = 0.5   # 90 degrees
    out = PF.apply_wake_table(table, np.array([1.0, 1.0, np.nan]),
                              np.array([45.0, 315.0, 90.0]),
                              directions_deg=np.array([0.0, 90, 180, 270]),
                              speeds_ms=np.array([0.0, 1.0, 2.0]))
    assert out[0, 0] == pytest.approx(0.75)
    assert out[1, 0] == pytest.approx(1.0)
    assert np.isnan(out[2, 0])


def test_fallback_wake_table():
    assert PF.fallback_wake_loss(1, False) == 0.0
    assert PF.fallback_wake_loss(17, False) == 0.06
    assert PF.fallback_wake_loss(17, True) == pytest.approx(0.08)
    assert PF.fallback_wake_loss(500, False) == 0.13


# --------------------------------------------------------------------------
# end to end on the fixture


@pytest.fixture
def farm(tmp_path):
    assets = load_assets(FIXTURE)
    sites, heights = sites_from_assets(assets)
    nt = 4 * 24
    t = np.arange(nt) * 0.25
    ref = 15.0 - 15.0 * np.cos(2 * np.pi * t / 24.0)   # 0 -> 30 -> 0 m/s
    direction = np.where(t < 12, 270.0, 180.0)
    gust = np.broadcast_to((ref * 1.4)[:, None], (nt, len(sites)))
    fc = write_forecast(tmp_path / "fc.nc", sites, heights, ref,
                        density=1.20, direction=direction,
                        extra={"wind_gust": gust})
    return tmp_path, fc, sites


def test_cli_end_to_end(farm, capsys):
    tmp, fc, _ = farm
    out = tmp / "power.nc"
    code, summary = run_cli([str(fc), "--assets", str(FIXTURE),
                             "--per-turbine", "--matches",
                             str(tmp / "m.json"), "-o", str(out)], capsys)
    assert code == 0, summary
    assert summary["match_quality"] == {"exact": 3, "fuzzy": 2, "size": 2,
                                        "generic": 18}
    doc = PF.load_power(out)
    matches = PF.load_matches(tmp / "m.json")
    by_asset = {row["asset_id"]: row for row in matches["turbines"]}
    assert by_asset["synthetic:generator/101"]["curve_id"] == "V112/3000"
    assert by_asset["synthetic:generator/111"]["curve_id"] == "SWT120/3600"
    assert by_asset["synthetic:generator/121"]["match_quality"] == "size"
    assert by_asset["synthetic:generator/131"]["curve_id"] == \
        "generic_iec_II"
    outline = by_asset["synthetic:plant/14"]
    assert outline["count"] == 17 and outline["match_quality"] == "generic"
    plants = list(doc.data["plant_id"])
    assert plants[:4] == ["synthetic:plant/10", "synthetic:plant/11",
                          "synthetic:plant/12", "synthetic:generator/131"]
    methods = dict(zip(plants, doc.data["wake_method"]))
    assert methods["synthetic:plant/10"] == "layout-jensen"
    assert methods["synthetic:plant/14"] == "fallback-table"
    assert methods["synthetic:generator/131"] == "none"
    cap = dict(zip(plants, doc.data["capacity_mw"]))
    assert cap["synthetic:plant/10"] == pytest.approx(9.0)
    assert cap["synthetic:plant/14"] == pytest.approx(50.0)
    power = doc.data["power_mw"]
    gross = doc.data["gross_power_mw"]
    assert np.all(power <= gross + 1e-9)
    assert np.all(doc.data["wake_loss_mw"] >= -1e-9)
    assert np.all((doc.data["capacity_factor"] >= 0)
                  & (doc.data["capacity_factor"] <= 1))
    # the west-east row is waked from the west only (first 12 h)
    k = plants.index("synthetic:plant/10")
    loss = doc.data["wake_loss_mw"][:, k]
    nt = power.shape[0]
    assert loss[: nt // 2].max() > 0.1
    assert loss[nt // 2 + 4:].max() < 0.25 * loss[: nt // 2].max()
    # shutdown near the 30 m/s peak
    assert doc.data["running_fraction"][nt // 2 - 1, k] == 0.0
    assert doc.attrs["turbine_library_license"] == "ODbL-1.0"


def test_without_wakes_net_is_gross_times_losses(farm, capsys):
    tmp, fc, _ = farm
    code, _ = run_cli([str(fc), "--assets", str(FIXTURE), "--wake-model",
                       "none", "--electrical-losses", "0.05",
                       "--availability", "0.9", "-o", str(tmp / "p.nc")],
                      capsys)
    assert code == 0
    doc = PF.load_power(tmp / "p.nc")
    assert np.allclose(doc.data["power_mw"],
                       doc.data["gross_power_mw"] * 0.9 * 0.95)
    assert np.allclose(doc.data["wake_loss_mw"], 0.0)
    assert set(doc.data["wake_method"]) == {"none"}


def test_wakes_reduce_energy(farm, capsys):
    tmp, fc, _ = farm
    _, waked = run_cli([str(fc), "--assets", str(FIXTURE), "-o",
                        str(tmp / "a.nc")], capsys)
    _, free = run_cli([str(fc), "--assets", str(FIXTURE), "--wake-model",
                       "none", "-o", str(tmp / "b.nc")], capsys)
    for a, b in zip(waked["per_plant"], free["per_plant"]):
        assert a["energy_mwh"] <= b["energy_mwh"] + 1e-9
    assert waked["per_plant"][0]["energy_mwh"] < \
        free["per_plant"][0]["energy_mwh"]


def test_bastankhah_runs_and_is_deterministic(farm, capsys):
    tmp, fc, _ = farm
    for name in ("a", "b"):
        code, _ = run_cli([str(fc), "--assets", str(FIXTURE), "--wake-model",
                           "bastankhah", "-o", str(tmp / f"{name}.nc")],
                          capsys)
        assert code == 0
    a, b = PF.load_power(tmp / "a.nc"), PF.load_power(tmp / "b.nc")
    assert np.array_equal(a.data["power_mw"], b.data["power_mw"])


def test_gust_shutdown_stops_earlier(farm, capsys):
    tmp, fc, _ = farm
    run_cli([str(fc), "--assets", str(FIXTURE), "-o", str(tmp / "a.nc")],
            capsys)
    code, summary = run_cli([str(fc), "--assets", str(FIXTURE),
                             "--gust-shutdown", "-o", str(tmp / "b.nc")],
                            capsys)
    assert code == 0
    assert any("wind_gust" in note for note in summary["notes"])
    a, b = PF.load_power(tmp / "a.nc"), PF.load_power(tmp / "b.nc")
    assert (b.data["running_fraction"] <= a.data["running_fraction"]).all()
    assert (b.data["running_fraction"] < a.data["running_fraction"]).any()


def test_icing_loss_is_applied(farm, capsys):
    tmp, fc, sites = farm
    nt = 96
    with netCDF4.Dataset(str(tmp / "icing.nc"), "w") as ds:
        ds.createDimension("time", nt)
        ds.createDimension("site", len(sites))
        var = ds.createVariable("site_id", str, ("site",))
        for i, site in enumerate(sites):
            var[i] = site["site_id"]
        ds.createVariable("icing_loss_fraction", "f8",
                          ("time", "site"))[:] = 0.25
    run_cli([str(fc), "--assets", str(FIXTURE), "-o", str(tmp / "a.nc")],
            capsys)
    code, _ = run_cli([str(fc), "--assets", str(FIXTURE), "--icing-loss",
                       str(tmp / "icing.nc"), "-o", str(tmp / "b.nc")],
                      capsys)
    assert code == 0
    a, b = PF.load_power(tmp / "a.nc"), PF.load_power(tmp / "b.nc")
    assert np.allclose(b.data["power_mw"], a.data["power_mw"] * 0.75)
    assert b.attrs["icing_loss"] == "icing.nc:icing_loss_fraction"


def test_without_assets_every_turbine_is_generic(farm, capsys):
    tmp, fc, _ = farm
    code, summary = run_cli([str(fc), "-o", str(tmp / "p.nc")], capsys)
    assert code == 0
    assert set(summary["match_quality"]) == {"generic"}
    assert summary["plants"] == 8   # turbines alone; the outline needs assets
    assert any("no assets document" in n for n in summary["notes"])


@pytest.mark.parametrize("change,needle", [
    (dict(drop=("wind_speed",)), "wind_speed"),
    (dict(drop=("air_density",)), "air_density"),
    (dict(schema="woof-energy.other.v1"), "schema"),
])
def test_refusals_of_the_forecast(tmp_path, capsys, change, needle):
    fc = write_forecast(tmp_path / "fc.nc", [TURBINE_SITE], (60.0, 100.0),
                        np.full(4, 8.0), **change)
    code, summary = run_cli([str(fc), "-o", str(tmp_path / "p.nc")], capsys)
    assert code == 2
    assert needle in summary["refused"]


def test_refuses_assets_that_do_not_hold_the_sites(tmp_path, capsys):
    fc = write_forecast(tmp_path / "fc.nc", [TURBINE_SITE], (60.0, 100.0),
                        np.full(4, 8.0))
    code, summary = run_cli([str(fc), "--assets", str(FIXTURE), "-o",
                             str(tmp_path / "p.nc")], capsys)
    assert code == 2
    assert "assets document does not hold" in summary["refused"]


def test_a_missing_icing_file_is_named(farm, capsys):
    tmp, fc, _ = farm
    code, summary = run_cli([str(fc), "--icing-loss", "missing.nc", "-o",
                             str(tmp / "p.nc")], capsys)
    assert code == 2 and "missing.nc" in summary["refused"]


def test_a_plant_site_and_its_turbines_are_not_counted_twice(tmp_path,
                                                            capsys):
    assets = load_assets(FIXTURE)
    sites, heights = sites_from_assets(assets)
    sites.append(dict(site_id="synthetic:plant/10", asset_id="synthetic:plant/10",
                      kind="plant", lat=52.1, lon=-3.59, hub_height_m=None,
                      capacity_mw=9.0))
    fc = write_forecast(tmp_path / "fc.nc", sites, heights, np.full(8, 10.0))
    code, summary = run_cli([str(fc), "--assets", str(FIXTURE), "-o",
                             str(tmp_path / "p.nc")], capsys)
    assert code == 0
    assert any("skipped because their turbines" in n
               for n in summary["notes"])
    first = summary["per_plant"][0]
    assert first["plant_id"] == "synthetic:plant/10"
    assert first["capacity_mw"] == 9.0 and first["n_turbines"] == 3


def test_refuses_an_icing_loss_out_of_range(farm, capsys):
    tmp, fc, sites = farm
    with netCDF4.Dataset(str(tmp / "icing.nc"), "w") as ds:
        ds.createDimension("time", 96)
        ds.createDimension("site", len(sites))
        ds.createVariable("loss", "f8", ("time", "site"))[:] = 1.5
    code, summary = run_cli([str(fc), "--icing-loss",
                             f"{tmp / 'icing.nc'}:loss", "-o",
                             str(tmp / "p.nc")], capsys)
    assert code == 2 and "[0, 1]" in summary["refused"]


def test_cli_refuses_bad_factors_at_the_parser():
    parser = build_parser()
    for argv in (["--availability", "0"], ["--availability", "1.5"],
                 ["--electrical-losses", "0.9"],
                 ["--electrical-losses", "nan"]):
        with pytest.raises(SystemExit):
            parser.parse_args(["energy", "power", "fc.nc", "-o", "p.nc"]
                              + argv)


def test_load_power_refuses_other_files(tmp_path):
    path = tmp_path / "x.nc"
    with netCDF4.Dataset(str(path), "w") as ds:
        ds.schema = "woof-energy.products.v1"
    with pytest.raises(ContractError):
        PF.load_power(path)


# --------------------------------------------------------------------------
# the build script


def test_build_script_corrections(tmp_path):
    from tools import build_turbine_library as B

    upstream = tmp_path / "up"
    upstream.mkdir()
    (upstream / "turbine_data.csv").write_text(
        "turbine_type,id,turbine_id,manufacturer,name,nominal_power,"
        "rotor_diameter,rotor_area,hub_height,max_speed_drive,"
        "wind_class_iec,wind_zone_dibt,power_density,power_density_2,"
        "calculated,has_power_curve,has_cp_curve,has_ct_curve,source\n"
        "X100/2000,1,1,Vestas,X100-2.0 MW,2000000,100,7854,80; 100,,"
        "IEC IIA,,,,,True,False,False,https://example.org/x\n"
        "Y90/2850,2,2,GE Wind,Y 3.4-90,3850000,90,6362,,,,,,,,False,False,"
        "False,\n")
    (upstream / "power_curves.csv").write_text(
        "turbine_type,0.0,4.0,10.0,25.0,26.0\n"
        "X100/2000,0.0,100000.0,2050000.0,2000000.0,0.0\n")
    document = B.build(upstream)
    rendered = B.render(document)
    path = tmp_path / "lib.json"
    path.write_text(rendered)
    lib = TL.load_turbine_library(path)
    x = lib.get("X100/2000")
    assert x.curve_source == "oedb" and x.iec_class == "II"
    assert x.power_kw.max() == 2000.0           # clipped
    assert x.speeds[-1] == 25.0                 # trailing zero removed
    assert x.hub_heights_m == (80.0, 100.0)
    y = lib.get("Y90/2850")
    assert y.rated_kw == 2850.0                 # type suffix wins
    assert y.curve_source == "parametric"
    assert "Y 3.4-90" not in y.names            # names another rating
    assert len(lib) == 2 + len(B.SUPPLEMENT)
    assert document["sources"][0]["commit"] == B.UPSTREAM_COMMIT


def test_committed_library_counts_agree():
    document = json.loads(TL.DEFAULT_TURBINE_LIBRARY.read_text())
    turbines = document["turbines"]
    assert document["counts"]["turbines"] == len(turbines)
    assert [t["id"] for t in turbines] == sorted(t["id"] for t in turbines)


# --------------------------------------------------------------------------
# docs


def _table(text: str, header: str) -> dict[str, list[str]]:
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(header))
    rows = {}
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows[cells[0].strip("`")] = cells[1:]
    return rows


def test_the_power_page_lists_every_variable_and_coordinate():
    text = DOC.read_text(encoding="utf-8")
    variables = _table(text, "| Variable | Dimensions |")
    assert list(variables) == list(PF.POWER_VARIABLES) + \
        list(PF.TURBINE_VARIABLES)
    for name, (dims, units, long_name) in {**PF.POWER_VARIABLES,
                                           **PF.TURBINE_VARIABLES}.items():
        assert variables[name] == [", ".join(dims), units, long_name], name
    coords = _table(text, "| Coordinate | Dimensions |")
    assert list(coords) == list(PF.POWER_COORDINATES)
    for name, (dims, units, long_name) in PF.POWER_COORDINATES.items():
        assert coords[name] == [", ".join(dims), units or "-", long_name]


def test_the_power_page_states_the_pinned_constants():
    text = DOC.read_text(encoding="utf-8")
    for value in (f"{PF.JENSEN_K['onshore']:g}", f"{PF.JENSEN_K['offshore']:g}",
                  f"{PF.BASTANKHAH_K['onshore']:g}",
                  f"{PF.BASTANKHAH_K['offshore']:g}",
                  f"{TL.RESTART_MARGIN_MS:g} m/s", f"{TL.GUST_MARGIN_MS:g} m/s",
                  "ODbL-1.0"):
        assert value in text, value
    for limit, loss in PF.FALLBACK_WAKE_LOSS[1:-1]:
        assert re.search(rf"\b{limit}\b", text), limit


def test_the_guide_names_the_power_product():
    text = GUIDE.read_text(encoding="utf-8")
    assert "woof energy power" in text
    assert "energy-hindcast/power.md" in text
