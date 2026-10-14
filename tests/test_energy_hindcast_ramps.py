"""Ramps and wind-energy products (``woof.energy.ramps``).

Synthetic series with known ramps, shear, veer and turbulence; the
``wind-resource`` and ``ramps`` products through ``woof energy rating`` on
the hourly products fixture and on a 1-minute forecast; the ramp-event
contract loader; the xarray entry points; and doc/code parity of the two
tables in ``docs/energy-hindcast/ramps.md``.  No network, no GPU.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

from woof.energy import products as P
from woof.energy import ramps as R
from woof.energy.contracts import (
    RAMP_EVENT_COLUMNS,
    ContractError,
    RampEvent,
    load_ramp_events,
)

from test_energy_products import _parse, open_nc, write_forecast

REPO = Path(__file__).resolve().parents[1]
RAMPS_DOC = REPO / "docs" / "energy-hindcast" / "ramps.md"
MINUTE = 60.0


def trapezoid():
    """1-min day: 0.2, up to 0.8 over 10:00-10:30, down to 0.1 over
    14:00-14:10."""

    t = np.arange(24 * 60) * MINUTE
    m = t / MINUTE
    p = np.full(t.size, 0.2)
    p = np.where((m >= 600) & (m <= 630), 0.2 + 0.02 * (m - 600), p)
    p = np.where((m > 630) & (m < 840), 0.8, p)
    p = np.where((m >= 840) & (m <= 850), 0.8 - 0.07 * (m - 840), p)
    return t, np.where(m > 850, 0.1, p)


def by_method(events, method, window=None):
    return [e for e in events if e["method"] == method
            and (window is None or e["window_min"] == window)]


# --------------------------------------------------------------------------
# ramp detection


def test_known_ramps_detected_by_every_window_and_the_swinging_door():
    t, p = trapezoid()
    result = R.detect_ramps(p, t)
    assert [w.minutes for w in result.windows] == [15.0, 60.0, 240.0]
    for method, window in (("threshold", 15.0), ("threshold", 60.0),
                           ("threshold", 240.0), ("swinging_door", None)):
        events = by_method(result.events, method, window)
        assert [(e["direction"], e["start"], e["end"]) for e in events] == \
            [(1, 600, 630), (-1, 840, 850)], (method, window)
        up, down = events
        assert up["magnitude"] == pytest.approx(0.6)
        assert up["duration_min"] == 30.0
        assert up["max_rate"] == pytest.approx(1.2)
        assert down["magnitude"] == pytest.approx(0.7)
        assert down["max_rate"] == pytest.approx(4.2)
    flag = result.flag[:, 0, 1]
    assert set(np.flatnonzero(flag == 1)) == set(range(600, 631))
    assert set(np.flatnonzero(flag == -1)) == set(range(840, 851))
    assert np.array_equal(result.sda_flag[:, 0], flag)
    assert result.delta[630, 0, 1] == pytest.approx(0.6)
    assert np.isnan(result.delta[59, 0, 1])
    assert result.rate[605, 0] == pytest.approx(1.2)


def test_ramp_below_threshold_is_not_an_event():
    t, p = trapezoid()
    result = R.detect_ramps(p, t, windows="1h:0.65", threshold=0.75)
    assert [(e["method"], e["direction"]) for e in result.events] == \
        [("threshold", -1)]


def test_window_threshold_pairs_are_independent():
    t, p = trapezoid()
    # 15 min sees at most 0.3 of the 30-minute up ramp
    result = R.detect_ramps(p, t, windows="15m:0.35,1h:0.5")
    up = [e for e in result.events if e["direction"] == 1
          and e["method"] == "threshold"]
    assert [e["window_min"] for e in up] == [60.0]


def test_noisy_ramp_is_found_within_minutes():
    t, p = trapezoid()
    rng = np.random.default_rng(1)
    noisy = np.clip(p + rng.normal(0.0, 0.01, p.size), 0, 1)
    result = R.detect_ramps(noisy, t, sda_tolerance=0.05)
    for method, window in (("threshold", 60.0), ("swinging_door", None)):
        events = by_method(result.events, method, window)
        assert [e["direction"] for e in events] == [1, -1], method
        up, down = events
        assert up["magnitude"] == pytest.approx(0.6, abs=0.06)
        assert down["magnitude"] == pytest.approx(0.7, abs=0.06)
        if method == "swinging_door":
            # the door tolerance absorbs the noise: corners within minutes
            assert abs(up["start"] - 600) <= 5 and abs(up["end"] - 630) <= 5
            assert abs(down["start"] - 840) <= 5
            assert abs(down["end"] - 850) <= 5
        else:
            # the extremes may sit anywhere in the plateau noise of the
            # window, but the event covers the ramp and stays in the window
            assert 600 - 60 <= up["start"] <= 600 and 630 <= up["end"] <= 690
            assert 840 - 60 <= down["start"] <= 840
            assert 850 <= down["end"] <= 850 + 60


def test_irregular_time_steps_and_multiple_sites():
    t, p = trapezoid()
    keep = np.unique(np.concatenate([np.arange(0, t.size, 7),
                                     [600, 630, 840, 850]]))
    t2, p2 = t[keep], p[keep]
    signal = np.stack([p2, np.full(p2.size, 0.5), 1.0 - p2], axis=1)
    result = R.detect_ramps(signal, t2, windows="1h")
    starts = {(e["site"], e["method"], e["direction"]):
              (t2[e["start"]] / MINUTE, t2[e["end"]] / MINUTE)
              for e in result.events}
    assert starts[(0, "threshold", 1)] == (600, 630)
    assert starts[(2, "threshold", -1)] == (600, 630)
    assert starts[(0, "swinging_door", -1)] == (840, 850)
    assert not [e for e in result.events if e["site"] == 1]


def test_nan_gap_splits_segments_without_spurious_ramps():
    t, p = trapezoid()
    p = p.copy()
    p[300:320] = np.nan
    result = R.detect_ramps(p, t)
    assert len(by_method(result.events, "swinging_door")) == 2
    keep = result.sda_breakpoints[:, 0]
    assert keep[299] and keep[320]


def test_swinging_door_keeps_corners_within_tolerance():
    t, p = trapezoid()
    keep = R.swinging_door(p, t, 0.01)[:, 0]
    points = np.flatnonzero(keep)
    rebuilt = np.interp(np.arange(t.size), points, p[points])
    assert np.max(np.abs(rebuilt - p)) <= 0.02 + 1e-12
    assert points[0] == 0 and points[-1] == t.size - 1


def test_swinging_door_runs_split_by_a_slow_monotone_stretch_are_one_ramp():
    t = np.arange(30) * 3600.0
    p = np.concatenate([np.linspace(0.0, 0.5, 6),
                        0.5 + 0.001 * np.arange(1, 20),
                        np.linspace(0.52, 1.0, 5)])
    _, _, events = R.swinging_door_ramps(p, t, tolerance=0.05,
                                         threshold=0.2)
    assert [(e["direction"], e["start"], e["end"]) for e in events] == \
        [(1, 0, 29)]


def test_parse_ramp_windows():
    windows = R.parse_ramp_windows("4h, 15m:0.1,90s,30", threshold=0.3)
    assert [(w.minutes, w.threshold) for w in windows] == [
        (1.5, 0.3), (15.0, 0.1), (30.0, 0.3), (240.0, 0.3)]
    for bad in ("15x", "1h:abc", "0m", "1h,60m", "1h:-1"):
        with pytest.raises(R.RampsRefused):
            R.parse_ramp_windows(bad)


def test_refusals():
    t, p = trapezoid()
    with pytest.raises(R.RampsRefused, match="shorter than the median"):
        R.detect_ramps(p[::15], t[::15], windows="5m")
    with pytest.raises(R.RampsRefused, match="strictly increasing"):
        R.detect_ramps(p, t[::-1])
    with pytest.raises(R.RampsRefused, match="tolerance"):
        R.swinging_door(p, t, 0.0)
    assert isinstance(R.RampsRefused("x"), ContractError)


# --------------------------------------------------------------------------
# rotor, shear, veer, turbulence, power


def test_rotor_segments_are_the_disc():
    centres, weights = R.rotor_segments([100.0, 80.0], [120.0, 80.0], 10)
    assert weights.sum() == pytest.approx(1.0)
    assert np.allclose(weights, weights[::-1])
    assert centres[0, 0] == pytest.approx(46.0)
    assert centres[1, -1] == pytest.approx(116.0)


def test_rews_uniform_sheared_and_veered():
    heights = np.array([20.0, 50.0, 80.0, 110.0, 140.0, 170.0])
    uniform = np.full((1, 1, heights.size), 9.0)
    rews, outside = R.rotor_equivalent_wind_speed(uniform, heights, [100.0],
                                                  [120.0])
    assert rews[0, 0] == pytest.approx(9.0)
    assert outside[0] == 0
    # a power-law profile: REWS against a fine numerical integral over the
    # disc of the same ln-linear interpolated profile
    profile = 8.0 * (heights / 100.0) ** 0.3
    sheared = profile[None, None, :]
    rews, _ = R.rotor_equivalent_wind_speed(sheared, heights, [100.0],
                                            [120.0], n_segments=400)
    z = np.linspace(-60.0, 60.0, 20001)
    chord = 2.0 * np.sqrt(np.maximum(60.0 ** 2 - z ** 2, 0.0))
    speed = np.interp(np.log(100.0 + z), np.log(heights), profile)
    exact = (np.trapezoid(speed ** 3 * chord, z)
             / np.trapezoid(chord, z)) ** (1 / 3)
    assert rews[0, 0] == pytest.approx(exact, rel=1e-4)
    coarse, _ = R.rotor_equivalent_wind_speed(sheared, heights, [100.0],
                                              [120.0])
    assert coarse[0, 0] == pytest.approx(exact, rel=2e-3)
    # veer: the REWS falls with the cosine of the turning
    turn = np.radians(np.linspace(-30.0, 30.0, heights.size))
    u = (-9.0 * np.sin(turn))[None, None, :]
    v = (-9.0 * np.cos(turn))[None, None, :]
    veered, _ = R.rotor_equivalent_wind_speed(uniform, heights, [95.0],
                                              [120.0], u=u, v=v)
    assert veered[0, 0] < 9.0
    assert veered[0, 0] > 9.0 * math.cos(math.radians(30.0))


def test_shear_exponent_recovers_power_law():
    heights = np.array([10.0, 40.0, 80.0, 120.0, 160.0, 250.0])
    alpha = np.array([0.1, 0.25, -0.05])
    speed = 7.0 * (heights[None, :] / 100.0) ** alpha[:, None]
    field = speed[None, :, :]
    got = R.shear_exponent(field, heights, [100.0, 100.0, 100.0],
                           [120.0, 120.0, 120.0])
    assert np.allclose(got[0], alpha)
    # a 20 m rotor holds no forecast height: the two nearest the hub
    got = R.shear_exponent(field, heights, [100.0] * 3, [20.0] * 3)
    assert np.allclose(got[0], alpha)
    calm = R.shear_exponent(np.zeros((1, 1, heights.size)), heights, [100.0],
                            [120.0])
    assert np.isnan(calm[0, 0])
    # an invalid (NaN) rotor has no shear, like every rotor quantity
    unknown = R.shear_exponent(field, heights, [100.0] * 3,
                               [np.nan, 120.0, 120.0])
    assert np.isnan(unknown[0, 0]) and np.isfinite(unknown[0, 1])


def test_wind_veer_signs_and_wraps():
    heights = np.array([20.0, 100.0, 180.0])

    def uv(directions):
        rad = np.radians(directions)
        return (-np.sin(rad))[None, None, :], (-np.cos(rad))[None, None, :]

    u, v = uv([350.0, 0.0, 10.0])
    veer = R.wind_veer(u, v, heights, [100.0], [160.0])
    assert veer[0, 0] == pytest.approx(20.0)
    u, v = uv([30.0, 20.0, 10.0])
    assert R.wind_veer(u, v, heights, [100.0], [160.0])[0, 0] == \
        pytest.approx(-20.0)
    calm = R.wind_veer(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)), heights,
                       [100.0], [160.0])
    assert np.isnan(calm[0, 0])


def test_turbulence_intensity_from_tke_and_resolved():
    ti = R.turbulence_intensity_from_tke(np.array([1.5, 1.5]),
                                         np.array([10.0, 0.5]))
    assert ti[0] == pytest.approx(0.1)
    assert np.isnan(ti[1])
    t = np.arange(60) * MINUTE
    speed = (10.0 + np.where(np.arange(60) % 2, 1.0, -1.0))[:, None]
    resolved = R.resolved_turbulence_intensity(speed, t, 10 * MINUTE)
    assert np.isnan(resolved[1, 0])
    # 10 alternating samples of +-1: sample std sqrt(10/9)
    assert resolved[30, 0] == pytest.approx(math.sqrt(10 / 9) / 10.0)


def test_density_correction_and_generic_curves():
    assert R.density_corrected_speed(10.0, 1.225) == pytest.approx(10.0)
    assert R.density_corrected_speed(10.0, 1.225 * 0.729) == \
        pytest.approx(9.0)
    for key, rated in (("generic_iec_class_i", 11.4),
                       ("generic_iec_class_iii", 9.8),
                       ("generic_offshore", 10.59)):
        curve = P.load_power_curve(name=key)
        assert curve["rated_ms"] == rated
        assert curve["source"] and curve["reference_turbine"]
        assert np.all(np.diff(curve["fraction"][curve["speeds"] <= 25]) >= 0)
        assert R.curve_power_fraction(rated, curve) == pytest.approx(1.0)
        assert R.curve_power_fraction(25.5, curve) == 0.0
        assert R.curve_power_fraction(curve["cut_in_ms"], curve) == 0.0
        implied = R.rated_speed_for_specific_power(
            curve["specific_power_w_m2"])
        assert implied == pytest.approx(rated, rel=1e-3)
        reference = curve["reference_turbine"]
        sp = R.specific_power(reference["rated_power_mw"],
                              reference["rotor_diameter_m"])
        assert sp == pytest.approx(reference["specific_power_w_m2"], rel=1e-3)
        assert R.rated_speed_for_specific_power(sp) == \
            pytest.approx(rated, rel=0.03)
    curve = P.load_power_curve()
    stretched = R.curve_power_fraction([7.5, 10.0, 11.0, 24.0, 26.0], curve,
                                       rated_ms=10.0)
    assert stretched[1] == 1.0 and stretched[2] == 1.0
    assert stretched[0] == pytest.approx(
        R.curve_power_fraction(3.0 + 4.5 * 9.0 / 7.0, curve))
    assert stretched[3] == 1.0 and stretched[4] == 0.0
    assert np.isnan(R.curve_power_fraction(np.nan, curve))
    with pytest.raises(R.RampsRefused):
        R.curve_power_fraction(5.0, curve, rated_ms=2.0)


def test_capacity_factor():
    power = np.array([[1.0, np.nan], [3.0, np.nan]])
    cf = R.capacity_factor(power, np.array([4.0, 2.0]))
    assert cf[0] == pytest.approx(0.5)
    assert np.isnan(cf[1])


# --------------------------------------------------------------------------
# woof energy rating


TURBINES = [
    dict(site_id="T1", kind="turbine", lat=55.0, lon=-3.0, hub_height_m=100.0,
         capacity_mw=4.0),
    dict(site_id="WF", kind="plant", lat=55.1, lon=-3.1, hub_height_m=90.0,
         capacity_mw=120.0),
    dict(site_id="L1", kind="line_sample", lat=55.2, lon=-3.2,
         bearing_deg=0.0, voltage_kv=400.0),
]


def minute_wind():
    """1-min hub wind with one up ramp (10:01-10:10) and one down ramp
    (14:00-14:10) of the full capacity on the class II curve."""

    m = np.arange(24 * 60)
    wind = np.full(m.size, 2.0)
    wind[601] = 2.9
    wind[602:610] = np.arange(4.0, 12.0)
    wind[610:840] = 13.0
    wind[840:850] = np.linspace(13.0, 3.5, 10)
    wind[841] = 11.0
    wind[850:] = 2.0
    return wind


def write_minute_forecast(path, wind):
    write_forecast(path, sites=TURBINES, hours=np.arange(wind.size),
                   heights=(10.0, 50.0, 100.0, 150.0, 200.0),
                   values={"wind_speed": wind, "air_density": 1.225})
    with netCDF4.Dataset(str(path), "a") as ds:
        ds["time"].units = "minutes since 2026-06-21 00:00:00"
    return path


def test_rating_on_the_hourly_fixture(tmp_path):
    speed = 6.0 + 4.0 * np.sin(np.arange(24) / 3.0)
    fc = write_forecast(tmp_path / "fc.nc", values={"wind_speed": speed})
    out = P.compute_products(fc, output=tmp_path / "p.nc",
                             products=("wind-power", "wind-resource",
                                       "ramps"))
    with open_nc(out) as ds:
        assert ds.schema == "woof-energy.products.v1"
        t1 = list(ds["site_id"][:]).index("T1")
        for name in R.WIND_RESOURCE_VARIABLES + R.RAMPS_VARIABLES:
            dims, units, _ = R.PRODUCT_VARIABLES[name]
            assert ds[name].dimensions == dims, name
            assert ds[name].units == units, name
        # uniform profile: REWS is the hub wind, power the wind-power power
        assert np.allclose(ds["rotor_equivalent_wind_speed"][:, t1], speed)
        assert np.allclose(ds["rews_power_mw"][:, t1],
                           ds["wind_power_mw"][:, t1])
        assert np.allclose(ds["shear_exponent"][:, t1], 0.0)
        assert np.all(np.isnan(ds["turbulence_intensity"][:, t1]))
        assert 0 < ds["capacity_factor"][t1] < 1
        assert np.isnan(ds["capacity_factor"][0])
        assert list(ds["ramp_window"][:]) == [60.0, 240.0]
        notes = ds.notes
    assert "15 min skipped" in notes
    assert "no TKE" in notes
    assert "rotor diameter assumed equal to the hub height" in notes
    events = load_ramp_events(out)
    assert events and all(e.site_id == "T1" for e in events)
    assert {e.method for e in events} == {"threshold", "swinging_door"}


def test_rating_on_a_one_minute_series_finds_the_known_ramps(tmp_path):
    fc = write_minute_forecast(tmp_path / "fc.nc", minute_wind())
    out = P.compute_products(fc, output=tmp_path / "p.nc",
                             products=("wind-resource", "ramps"),
                             wind_options={"ramp_windows": "5m,15m,1h,4h"})
    events = load_ramp_events(out)
    t0 = dt.datetime(2026, 6, 21)
    expected = [(1, t0 + dt.timedelta(minutes=601),
                 t0 + dt.timedelta(minutes=610)),
                (-1, t0 + dt.timedelta(minutes=840),
                 t0 + dt.timedelta(minutes=850))]
    for site in ("T1", "WF"):
        for method, window in (("threshold", 15.0), ("threshold", 60.0),
                               ("threshold", 240.0),
                               ("swinging_door", None)):
            rows = [e for e in events if e.site_id == site
                    and e.method == method and e.window_min == window]
            assert [(e.direction, e.start, e.end) for e in rows] == \
                expected, (site, method, window)
            assert all(e.magnitude == pytest.approx(1.0) for e in rows)
            assert all(isinstance(e, RampEvent) for e in rows)
    # the 5 min window sees part of each ramp and trims to the same extremes
    five = [e for e in events if e.window_min == 5.0 and e.site_id == "T1"]
    assert [(e.direction, e.start, e.end) for e in five] == expected
    with open_nc(out) as ds:
        t1 = list(ds["site_id"][:]).index("T1")
        assert ds["ramp_flag"][605, t1, 2] == 1
        assert ds["ramp_flag"][845, t1, 2] == -1
        assert ds["ramp_flag"][700, t1, 2] == 0
        assert ds["ramp_flag_sda"][605, t1] == 1
        assert ds["ramp_signal"][700, t1] == 1.0
        assert ds["ramp_event_start"].units.startswith("minutes since")
        # 1-min data resolve TI: zero on the constant plateau
        assert ds["turbulence_intensity"][700, t1] == \
            pytest.approx(0.0, abs=1e-6)
        assert ds["turbulence_intensity"][606, t1] > 0
        assert np.all(ds["ramp_flag"][:, 2, :] == 0)


def test_rotor_diameter_coordinate_scales_rated_speed(tmp_path):
    wind = np.full(60, 10.0)
    fc = write_forecast(tmp_path / "fc.nc", sites=TURBINES[:1],
                        hours=np.arange(60) / 60.0,
                        values={"wind_speed": wind, "air_density": 1.225})
    out = P.compute_products(fc, output=tmp_path / "a.nc",
                             products=("wind-resource",))
    with open_nc(out) as ds:
        generic = ds["rews_power_mw"][0, 0]
    with netCDF4.Dataset(str(fc), "a") as ds:
        var = ds.createVariable("rotor_diameter_m", "f8", ("site",))
        # 4 MW on 150 m: 226 W/m2, rated about 9.4 m/s
        var[:] = [150.0]
    out = P.compute_products(fc, output=tmp_path / "b.nc",
                             products=("wind-resource",))
    with open_nc(out) as ds:
        assert ds["rews_power_mw"][0, 0] == pytest.approx(4.0)
        assert ds["rotor_diameter_m"][0] == 150.0
        assert "rated speed scaled by specific power" in ds.notes
    assert generic < 4.0


def test_cli_summary_and_refusals(tmp_path, capsys):
    fc = write_minute_forecast(tmp_path / "fc.nc", minute_wind())
    args = _parse(["energy", "rating", str(fc), "-o", str(tmp_path / "p.nc"),
                   "--products", "wind-resource,ramps", "--ramp-windows",
                   "15m,1h:0.5", "--ramp-threshold", "0.3",
                   "--power-curve", "generic_offshore",
                   "--sda-tolerance", "0.02", "--rotor-diameter-m", "120"])
    assert args.func(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["products"] == ["wind-resource", "ramps"]
    assert summary["ramp_events"] == {"swinging_door": 4,
                                      "threshold 15 min": 4,
                                      "threshold 60 min": 4}
    assert any("--rotor-diameter-m 120 m" in n for n in summary["notes"])
    with open_nc(tmp_path / "p.nc") as ds:
        assert list(ds["ramp_threshold"][:]) == [0.3, 0.5]
        assert "generic_offshore" in ds["rews_power_mw"].method
        assert ds.power_curve == "generic_offshore"
    for argv, match in ((["--ramp-windows", "15q"], "not a duration"),
                        (["--power-curve", "nope"], "power curve 'nope'")):
        args = _parse(["energy", "rating", str(fc), "-o",
                       str(tmp_path / "q.nc"), "--products", "ramps", *argv])
        assert args.func(args) == 2
        assert match in json.loads(capsys.readouterr().out)["refused"]
    hourly = write_forecast(tmp_path / "h.nc")
    args = _parse(["energy", "rating", str(hourly), "-o",
                   str(tmp_path / "r.nc"), "--products", "ramps",
                   "--ramp-windows", "15m"])
    assert args.func(args) == 2
    assert "shorter than the forecast time step" in \
        json.loads(capsys.readouterr().out)["refused"]
    # a misspelt curve is refused even when no wind product runs
    args = _parse(["energy", "rating", str(hourly), "-o",
                   str(tmp_path / "s.nc"), "--products", "dlr",
                   "--power-curve", "generic_iec_class_1"])
    assert args.func(args) == 2
    assert "generic_iec_class_1" in \
        json.loads(capsys.readouterr().out)["refused"]
    # the default product set is unchanged
    args = _parse(["energy", "rating", str(hourly), "-o",
                   str(tmp_path / "d.nc")])
    assert args.products == P.DEFAULT_PRODUCTS


def test_no_ramps_writes_an_empty_valid_table(tmp_path):
    fc = write_minute_forecast(tmp_path / "fc.nc", np.full(120, 8.0))
    out = P.compute_products(fc, output=tmp_path / "p.nc",
                             products=("ramps",))
    assert load_ramp_events(out) == []
    with open_nc(out) as ds:
        assert len(ds.dimensions["ramp_event"]) == 0


# --------------------------------------------------------------------------
# contract


def _event(**overrides):
    base = dict(site_id="T1", method="threshold", window_min=60.0,
                direction=1, start=dt.datetime(2026, 1, 1, 10),
                end=dt.datetime(2026, 1, 1, 10, 30), magnitude=0.5,
                duration_min=30.0, max_rate_per_h=1.0)
    base.update(overrides)
    return RampEvent(**base)


def test_ramp_event_contract_refusals():
    assert _event().window_min == 60.0
    assert _event(method="swinging_door", window_min=float("nan")) \
        .window_min is None
    for bad in (dict(method="kamath"), dict(direction=0),
                dict(window_min=None), dict(window_min=-5.0),
                dict(method="swinging_door"),
                dict(end=dt.datetime(2026, 1, 1, 9)),
                dict(duration_min=20.0), dict(magnitude=-0.1),
                dict(max_rate_per_h=float("inf")), dict(site_id=""),
                dict(start=0.0, end=1800.0)):
        with pytest.raises(ContractError):
            _event(**bad)
    # durations come from whole-second times: a second of slack
    assert _event(end=dt.datetime(2026, 1, 1, 10, 30, 0, 500000))


def test_load_ramp_events_refuses_files_without_the_table(tmp_path):
    fc = write_forecast(tmp_path / "fc.nc")
    with pytest.raises(ContractError, match="schema"):
        load_ramp_events(fc)
    out = P.compute_products(fc, output=tmp_path / "p.nc",
                             products=("wind-power",))
    with pytest.raises(ContractError, match="no ramp-event table"):
        load_ramp_events(out)
    with pytest.raises(ContractError, match="does not exist"):
        load_ramp_events(tmp_path / "missing.nc")


# --------------------------------------------------------------------------
# xarray


def test_ramps_dataset_on_xarray():
    xr = pytest.importorskip("xarray")
    t, p = trapezoid()
    times = np.datetime64("2026-01-01T00:00") + (t / MINUTE).astype(int) \
        * np.timedelta64(1, "m")
    da = xr.DataArray(np.stack([p * 50.0, p * 100.0], axis=1),
                      dims=("time", "site"),
                      coords={"time": times, "site": ["A", "B"]})
    ds = R.ramps_dataset(da, capacity=xr.DataArray([50.0, 100.0],
                                                   dims="site"),
                         windows="1h")
    events = load_ramp_events(ds)
    assert len(events) == 8
    first = [e for e in events if e.site_id == "B"
             and e.method == "threshold"][0]
    assert first.start == dt.datetime(2026, 1, 1, 10)
    assert first.end == dt.datetime(2026, 1, 1, 10, 30)
    assert first.magnitude == pytest.approx(0.6)
    assert ds["ramp_flag"].sel(site="A", ramp_window=60.0)[620] == 1
    # a hub-wind series without capacity: thresholds in m/s
    wind = xr.DataArray(4.0 + 10.0 * p, dims="time", coords={"time": times},
                        name="hub")
    ds = R.ramps_dataset(wind, windows="1h:5", threshold=5.0)
    assert {e.site_id for e in load_ramp_events(ds)} == {"hub"}


def test_wind_resource_dataset_matches_numpy():
    xr = pytest.importorskip("xarray")
    heights = np.array([20.0, 60.0, 100.0, 140.0, 180.0])
    n_time = 30
    speed = (8.0 * (heights / 100.0) ** 0.2)[None, None, :] \
        * (1.0 + 0.05 * np.sin(np.arange(n_time)))[:, None, None]
    speed = np.repeat(speed, 2, axis=1)
    rad = np.radians(200.0 + 0.1 * heights)
    u = -speed * np.sin(rad)
    v = -speed * np.cos(rad)
    times = np.datetime64("2026-01-01T00:00") + np.arange(n_time) \
        * np.timedelta64(1, "m")
    ds = xr.Dataset({"wind_speed": (("time", "site", "height"), speed),
                     "u": (("time", "site", "height"), u),
                     "v": (("time", "site", "height"), v)},
                    coords={"time": times, "site": ["A", "B"],
                            "height": heights})
    curve = P.load_power_curve()
    out = R.wind_resource_dataset(ds, hub_height_m=[100.0, 90.0],
                                  rotor_diameter_m=[160.0, 100.0],
                                  curve=curve, capacity_mw=[3.0, 2.0])
    rews, _ = R.rotor_equivalent_wind_speed(speed, heights, [100.0, 90.0],
                                            [160.0, 100.0], u=u, v=v)
    assert np.allclose(out["rotor_equivalent_wind_speed"].values, rews)
    assert np.allclose(out["shear_exponent"].values, 0.2)
    assert np.allclose(out["wind_veer"].sel(site="A").values, 16.0)
    assert out["rews_power_mw"].dims == ("time", "site")
    assert np.allclose(out["rews_power_mw"].values,
                       R.curve_power_fraction(rews, curve)
                       * np.array([3.0, 2.0]))
    assert np.isnan(out["turbulence_intensity"].values[0, 0])
    assert np.isfinite(out["turbulence_intensity"].values[20, 0])
    assert "resolved" in out["turbulence_intensity"].attrs["method"]
    ds["turbulent_kinetic_energy"] = ds["wind_speed"] * 0 + 0.6
    out = R.wind_resource_dataset(ds, hub_height_m=100.0,
                                  rotor_diameter_m=120.0)
    hub = out["turbulence_intensity"].values
    assert hub[0, 0] == pytest.approx(math.sqrt(0.4) / speed[0, 0, 2])


# --------------------------------------------------------------------------
# documentation


def _doc_table(header: str) -> dict[str, list[str]]:
    lines = RAMPS_DOC.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(header))
    rows = {}
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        rows[cells[0].strip("`")] = cells[1:]
    return rows


def test_doc_lists_every_product_variable_exactly():
    rows = _doc_table("| Variable | Dimensions |")
    assert list(rows) == list(R.PRODUCT_VARIABLES)
    for name, (dims, units, long_name) in R.PRODUCT_VARIABLES.items():
        assert rows[name][:3] == [", ".join(dims), units, long_name], name


def test_doc_lists_every_ramp_event_column_exactly():
    rows = _doc_table("| Column | Units |")
    assert list(rows) == list(RAMP_EVENT_COLUMNS)
    for name, (units, meaning) in RAMP_EVENT_COLUMNS.items():
        assert rows[name] == [units, meaning], name


def test_guide_lists_the_new_products():
    guide = (REPO / "docs" / "energy-forecasts.md").read_text("utf-8")
    for product in P.PRODUCTS:
        assert f"| `{product}` |" in guide, product
