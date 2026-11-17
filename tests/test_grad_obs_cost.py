"""woof.grad.obs_cost: the observation cost J_o on hand-built history frames.

Each operator is checked against a value worked by hand on a WRF-layout and
an MPAS-layout file written here; FGAT pairing, the background check and
the frozen observation set (``--accept``) are checked on the same files;
the Abel inversion is checked on an analytic exponential atmosphere; and
the import graph is checked so no forecast path loads ``woof.grad``.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

from woof.core import constants as c  # noqa: E402
from woof.grad import obs_cost as oc  # noqa: E402
from woof.grad import obs_stores  # noqa: E402

T0 = datetime(2026, 11, 7, 0, tzinfo=timezone.utc)
NY, NX = 14, 16
P_LEVELS = np.array([100000.0, 92500.0, 85000.0, 70000.0, 50000.0, 30000.0])
Z_W = np.array([0.0, 700.0, 1500.0, 3000.0, 5500.0, 9000.0, 12000.0])  # interfaces, above MSL


def _lat_lon():
    lat = 55.0 + 0.1 * np.arange(NY)
    lon = -4.0 + 0.1 * np.arange(NX)
    return np.meshgrid(lat, lon, indexing="ij")


def _t2(hour, lat, lon):
    return 280.0 + hour + 2.0 * (lat - 55.0) + 1.0 * (lon + 4.0)


def write_wrf(path, hours=(0, 1, 2), *, sina=0.0, cosa=1.0, theta_shift=0.0):
    """A WRF-layout history: theta 300 K + shift, the levels above, grid winds (4, -2), T2 a ramp."""
    nz = P_LEVELS.size
    lat, lon = _lat_lon()
    with netCDF4.Dataset(path, "w") as d:
        for name, size in (("Time", None), ("DateStrLen", 19), ("south_north", NY), ("west_east", NX),
                           ("bottom_top", nz), ("bottom_top_stag", nz + 1),
                           ("west_east_stag", NX + 1), ("south_north_stag", NY + 1)):
            d.createDimension(name, size)

        def var(name, dims, value):
            v = d.createVariable(name, "f4" if name != "Times" else "S1", dims)
            v[:] = value

        stamps = [(T0 + timedelta(hours=h)).strftime("%Y-%m-%d_%H:%M:%S") for h in hours]
        var("Times", ("Time", "DateStrLen"), np.array([list(s) for s in stamps], dtype="S1"))
        three = ("Time", "bottom_top", "south_north", "west_east")
        two = ("Time", "south_north", "west_east")
        nt = len(hours)
        var("XLAT", two, np.broadcast_to(lat, (nt, NY, NX)))
        var("XLONG", two, np.broadcast_to(lon, (nt, NY, NX)))
        var("HGT", two, np.full((nt, NY, NX), 50.0))
        var("SINALPHA", two, np.full((nt, NY, NX), sina))
        var("COSALPHA", two, np.full((nt, NY, NX), cosa))
        var("PB", three, np.broadcast_to(P_LEVELS[None, :, None, None], (nt, nz, NY, NX)))
        var("P", three, np.zeros((nt, nz, NY, NX)))
        var("T", three, np.full((nt, nz, NY, NX), theta_shift))
        var("QVAPOR", three, np.full((nt, nz, NY, NX), 0.004))
        var("QCLOUD", three, np.full((nt, nz, NY, NX), 1.0e-4))
        var("U", ("Time", "bottom_top", "south_north", "west_east_stag"), np.full((nt, nz, NY, NX + 1), 4.0))
        var("V", ("Time", "bottom_top", "south_north_stag", "west_east"), np.full((nt, nz, NY + 1, NX), -2.0))
        var("PHB", ("Time", "bottom_top_stag", "south_north", "west_east"),
            np.broadcast_to((Z_W * c.G)[None, :, None, None], (nt, nz + 1, NY, NX)))
        var("PH", ("Time", "bottom_top_stag", "south_north", "west_east"), np.zeros((nt, nz + 1, NY, NX)))
        var("T2", two, np.stack([_t2(h, lat, lon) for h in hours]))
        var("Q2", two, np.full((nt, NY, NX), 0.005))
        var("U10", two, np.full((nt, NY, NX), 3.0))
        var("V10", two, np.full((nt, NY, NX), 4.0))
        var("PSFC", two, np.full((nt, NY, NX), 100500.0))
        var("MU", two, np.zeros((nt, NY, NX)))
        var("MUB", two, np.full((nt, NY, NX), 90000.0))
        var("C1H", ("Time", "bottom_top"), np.ones((nt, nz)))
        var("C2H", ("Time", "bottom_top"), np.zeros((nt, nz)))
        var("DNW", ("Time", "bottom_top"), np.full((nt, nz), -1.0 / nz))
    return path


def write_mpas(path, hours=(0, 1, 2)):
    """An MPAS-layout history on the same points (cells), with bdyMaskCell framing a 2-cell rim."""
    lat, lon = _lat_lon()
    ring = np.minimum.reduce([np.arange(NY)[:, None] + 0 * lon.astype(int), (NY - 1 - np.arange(NY))[:, None]
                              + 0 * lon.astype(int), np.arange(NX)[None, :] + 0 * lat.astype(int),
                              (NX - 1 - np.arange(NX))[None, :] + 0 * lat.astype(int)])
    bdy = np.where(ring < 2, 1, 0).reshape(-1)
    ncell, nz = NY * NX, P_LEVELS.size
    with netCDF4.Dataset(path, "w") as d:
        for name, size in (("Time", None), ("StrLen", 64), ("nCells", ncell), ("nVertLevels", nz),
                           ("nVertLevelsP1", nz + 1)):
            d.createDimension(name, size)
        nt = len(hours)
        x = d.createVariable("xtime", "S1", ("Time", "StrLen"))
        x[:] = np.array([list((T0 + timedelta(hours=h)).strftime("%Y-%m-%d_%H:%M:%S").ljust(64))
                         for h in hours], dtype="S1")

        def var(name, dims, value, kind="f8"):
            v = d.createVariable(name, kind, dims)
            v[:] = value

        var("latCell", ("nCells",), np.radians(lat).reshape(-1))
        var("lonCell", ("nCells",), np.radians(lon).reshape(-1))
        var("bdyMaskCell", ("nCells",), bdy, "i4")
        var("zgrid", ("nCells", "nVertLevelsP1"), np.broadcast_to(Z_W, (ncell, nz + 1)))
        three = ("Time", "nCells", "nVertLevels")
        two = ("Time", "nCells")
        var("pressure", three, np.broadcast_to(P_LEVELS, (nt, ncell, nz)))
        var("theta", three, np.full((nt, ncell, nz), 300.0))
        var("qv", three, np.full((nt, ncell, nz), 0.004))
        var("qc", three, np.full((nt, ncell, nz), 1.0e-4))
        var("rho", three, np.full((nt, ncell, nz), 1.0))
        var("uReconstructZonal", three, np.full((nt, ncell, nz), 4.0))
        var("uReconstructMeridional", three, np.full((nt, ncell, nz), -2.0))
        var("t2m", two, np.stack([_t2(h, lat, lon).reshape(-1) for h in hours]))
        var("q2", two, np.full((nt, ncell), 0.005))
        var("u10", two, np.full((nt, ncell), 3.0))
        var("v10", two, np.full((nt, ncell), 4.0))
        var("surface_pressure", two, np.full((nt, ncell), 100500.0))
    return path


def _obs(rows):
    cols = {k: [r.get(k, np.nan if k not in ("platform", "kind", "station", "time") else "") for r in rows]
            for k in oc.OBS_FIELDS}
    return oc.ObsSet.from_columns(**cols)


def _at(hour, minute=0):
    return T0 + timedelta(hours=hour, minutes=minute)


# --------------------------------------------------------------------------- table
def test_the_shipped_error_table_reads_and_is_labelled_uncalibrated():
    table = oc.read_error_table()
    assert table.row("synop", "surface_temperature").error == 1.5
    doc = json.loads(oc.DEFAULT_TYPES.read_text())
    assert "UNCALIBRATED" in json.dumps(doc)
    obs = _obs([{"platform": "ro", "kind": "refractivity", "time": T0, "value": 300.0, "height_m": 0.0},
                {"platform": "ro", "kind": "refractivity", "time": T0, "value": 10.0, "height_m": 12000.0},
                {"platform": "synop", "kind": "surface_temperature", "time": T0, "value": 280.0, "error": 0.5}])
    np.testing.assert_allclose(table.sigma(obs), [6.0, 0.02, 0.5])


def test_a_table_row_needs_exactly_one_error(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"schema": oc.TYPES_SCHEMA, "types": [
        {"platform": "x", "kind": "surface_temperature", "error": 1.0,
         "relative_error": {"height_m": [0], "fraction": [0.1]}}]}))
    with pytest.raises(oc.ObsCostError, match="exactly one"):
        oc.read_error_table(path)


# --------------------------------------------------------------------------- operators, WRF layout
def test_wrf_operators_match_hand_values(tmp_path):
    frames = oc.read_frames([write_wrf(tmp_path / "wrfout_d01_x")])
    assert [f.valid for f in frames] == [_at(0), _at(1), _at(2)]
    lat, lon = 55.62, -3.31  # owned by row 6, column 7 (55.6, -3.3)
    rows = [
        {"platform": "synop", "kind": "surface_temperature", "time": _at(1), "lat": lat, "lon": lon,
         "elevation_m": 60.0, "value": 280.0},
        {"platform": "synop", "kind": "surface_dewpoint", "time": _at(1), "lat": lat, "lon": lon,
         "elevation_m": 60.0, "value": 270.0},
        {"platform": "synop", "kind": "surface_wind_u", "time": _at(1), "lat": lat, "lon": lon,
         "elevation_m": 60.0, "value": 2.0},
        {"platform": "synop", "kind": "surface_wind_v", "time": _at(1), "lat": lat, "lon": lon,
         "elevation_m": 60.0, "value": 2.0},
        {"platform": "synop", "kind": "surface_wind_speed", "time": _at(1), "lat": lat, "lon": lon,
         "elevation_m": 60.0, "value": 4.0},
        {"platform": "sonde", "kind": "upper_temperature", "time": _at(1), "lat": lat, "lon": lon,
         "pressure_pa": 60000.0, "value": 258.0},
        {"platform": "sonde", "kind": "upper_wind_u", "time": _at(1), "lat": lat, "lon": lon,
         "pressure_pa": 60000.0, "value": 3.0},
        {"platform": "ro", "kind": "refractivity", "time": _at(1), "lat": lat, "lon": lon,
         "height_m": 2000.0, "value": 250.0},
        {"platform": "goes", "kind": "cloud_water_path", "time": _at(1), "lat": lat, "lon": lon,
         "value": 900.0, "error": 50.0},
    ]
    obs = _obs(rows)
    result = oc.evaluate(obs, frames, table=oc.read_error_table())
    assert result.n == len(rows), result.counts
    h = dict(zip(result.kind.tolist(), result.h.tolist()))
    assert h["surface_temperature"] == pytest.approx(_t2(1, 55.6, -3.3), abs=1e-4)
    q2 = 0.005 / 1.005
    from woof.obs.column_operators import dewpoint_from_specific_humidity, refractivity_n
    assert h["surface_dewpoint"] == pytest.approx(float(dewpoint_from_specific_humidity(q2, 100500.0)), abs=1e-6)
    assert (h["surface_wind_u"], h["surface_wind_v"], h["surface_wind_speed"]) == pytest.approx((3.0, 4.0, 5.0))
    # 600 hPa: linear in ln p between 700 and 500 hPa of T = 300 (p/P0)^RCP
    t7, t5 = (300.0 * (p / c.P0) ** c.RCP for p in (70000.0, 50000.0))
    w = (np.log(70000.0) - np.log(60000.0)) / (np.log(70000.0) - np.log(50000.0))
    assert h["upper_temperature"] == pytest.approx(t7 + w * (t5 - t7), abs=1e-4)
    assert h["upper_wind_u"] == pytest.approx(4.0)
    # 2000 m sits between mass levels 1100 m (925 hPa) and 2250 m (850 hPa)
    zm = 0.5 * (Z_W[:-1] + Z_W[1:])
    f = (2000.0 - zm[1]) / (zm[2] - zm[1])
    p = np.exp(np.log(92500.0) + f * (np.log(85000.0) - np.log(92500.0)))
    t = (1 - f) * 300.0 * (92500.0 / c.P0) ** c.RCP + f * 300.0 * (85000.0 / c.P0) ** c.RCP
    assert h["refractivity"] == pytest.approx(float(refractivity_n(p, t, 0.004 / 1.004)), rel=1e-6)
    # CWP: 1e-4 kg/kg over 90000 Pa of dry mass, in g m-2
    assert h["cloud_water_path"] == pytest.approx(1.0e-4 * 90000.0 / c.G * 1000.0, rel=1e-6)
    by = result.by_type()
    assert by["synop/surface_temperature"]["J"] == pytest.approx(0.5 * ((h["surface_temperature"] - 280.0) / 1.5) ** 2)
    assert result.J == pytest.approx(sum(v["J"] for v in by.values()))


def test_wrf_winds_are_rotated_to_earth_relative(tmp_path):
    frames = oc.read_frames([write_wrf(tmp_path / "wrfout_d01_x", hours=(0,), sina=1.0, cosa=0.0)])
    # grid (3, 4) at 10 m and (4, -2) aloft, rotated a quarter turn: u = -v_grid, v = u_grid
    np.testing.assert_allclose(frames[0].u10, -4.0)
    np.testing.assert_allclose(frames[0].v10, 3.0)
    np.testing.assert_allclose(frames[0].u, 2.0)
    np.testing.assert_allclose(frames[0].v, 4.0)


def test_mpas_frames_agree_with_wrf_on_the_same_columns(tmp_path):
    wrf = oc.read_frames([write_wrf(tmp_path / "wrfout_d01_x")])
    mpas = oc.read_frames([write_mpas(tmp_path / "history.nc")])
    for a, b in zip(wrf, mpas):
        assert a.valid == b.valid
        for name in ("t", "p", "q", "u", "v", "z", "t2", "q2", "u10", "v10", "psfc"):
            np.testing.assert_allclose(getattr(b, name), getattr(a, name), rtol=1e-6, err_msg=name)
    # hex CWP is sum(q rho dz): 1e-4 * 1 kg m-3 * 12000 m, in g m-2
    np.testing.assert_allclose(mpas[0].cwp, 1.0e-4 * 12000.0 * 1000.0)
    # the 2-cell boundary ring is not usable
    assert mpas[0].usable.reshape(NY, NX)[1, 5] == False  # noqa: E712
    assert mpas[0].usable.reshape(NY, NX)[2, 5] == True  # noqa: E712


# --------------------------------------------------------------------------- pairing
def test_fgat_pairs_each_observation_with_its_nearest_frame(tmp_path):
    frames = oc.read_frames([write_wrf(tmp_path / "wrfout_d01_x")])
    base = {"platform": "synop", "kind": "surface_temperature", "lat": 55.6, "lon": -3.3,
            "elevation_m": 50.0, "value": 283.0}
    obs = _obs([dict(base, time=_at(1, 10)), dict(base, time=_at(1, 40)), dict(base, time=_at(2, 31)),
                dict(base, time=_at(1), lat=55.0), dict(base, time=_at(1), elevation_m=400.0),
                dict(base, time=_at(1), lat=40.0)])
    result = oc.evaluate(obs, frames)
    np.testing.assert_array_equal(result.obs_index, [0, 1])
    np.testing.assert_allclose(result.h, [_t2(1, 55.6, -3.3), _t2(2, 55.6, -3.3)], atol=1e-4)
    counts = result.counts
    assert counts["no_frame_within_offset"] == 1  # 02:31 is beyond half the hourly interval
    assert counts["outside_usable_columns"] == 2  # the WRF rim and a point off the grid
    assert counts["surface_elevation_mismatch"] == 1
    assert oc.evaluate(obs, frames, end=_at(1, 20)).obs_index.tolist() == [0]


def test_the_background_check_rejects_and_the_frozen_set_does_not(tmp_path):
    ref = oc.read_frames([write_wrf(tmp_path / "wrfout_ref")])
    warm = oc.read_frames([write_wrf(tmp_path / "wrfout_warm", theta_shift=0.2)])
    base = {"platform": "sonde", "kind": "upper_temperature", "lat": 55.6, "lon": -3.3, "time": _at(1),
            "pressure_pa": 70000.0}
    t7 = 300.0 * (70000.0 / c.P0) ** c.RCP
    obs = _obs([dict(base, value=t7 + 1.0), dict(base, value=t7 - 4.9), dict(base, value=t7 + 5.1)])
    reference = oc.evaluate(obs, ref)
    assert reference.obs_index.tolist() == [0, 1] and reference.counts["gross_check"] == 1
    pairs = tmp_path / "pairs.npz"
    reference.save_pairs(pairs)
    # unfrozen, the warmer run moves the third obs inside 5 sigma and the second outside: J jumps
    loose = oc.evaluate(obs, warm)
    assert loose.obs_index.tolist() == [0, 2]
    frozen = oc.evaluate(obs, warm, accept=oc.accepted_from(pairs))
    assert frozen.obs_index.tolist() == [0, 1]
    assert frozen.counts["lost_from_reference"] == 0 and frozen.counts["not_in_reference"] == 1
    dt = 0.2 * (70000.0 / c.P0) ** c.RCP
    assert frozen.J == pytest.approx(0.5 * ((1.0 - dt) ** 2 + (4.9 + dt) ** 2), rel=1e-5)
    other = _obs([dict(base, value=t7)])
    with pytest.raises(oc.ObsCostError, match="another observation set"):
        oc.evaluate(other, warm, accept=oc.accepted_from(pairs))


def test_obs_sets_round_trip_without_pickle(tmp_path):
    obs = _obs([{"platform": "synop", "kind": "surface_wind_u", "station": "03166", "time": _at(1),
                 "lat": 55.9, "lon": -3.2, "elevation_m": 20.0, "value": 1.5}])
    path = tmp_path / "obs.npz"
    obs.save(path)
    back = oc.ObsSet.load(path)
    assert back.digest() == obs.digest() and back.station.tolist() == ["03166"]
    with pytest.raises(oc.ObsCostError, match="unknown observation kind"):
        _obs([{"platform": "x", "kind": "radiance", "time": T0, "value": 1.0}])


# --------------------------------------------------------------------------- readers
def test_abel_inversion_recovers_an_exponential_atmosphere():
    n0, scale, radius = 300.0e-6, 7000.0, 6371.0e3
    a = radius + np.arange(0.0, 60000.0, 50.0)
    # the thin-atmosphere forward Abel transform of n - 1 = n0 exp(-(a - R)/H)
    alpha = n0 * np.sqrt(2.0 * np.pi * a / scale) * np.exp(-(a - radius) / scale)
    n = obs_stores.abel_refractivity(a, alpha)
    expected = n0 * np.exp(-(a - radius) / scale)
    low = a - radius < 30000.0
    np.testing.assert_allclose((n[low] - 1.0) / expected[low], 1.0, atol=0.01)
    rows = obs_stores.ro_profile_rows(a, alpha, np.full(a.size, 56.0), np.full(a.size, -3.0),
                                      r_curve=radius, undulation=0.0, time=T0)
    assert np.all(np.diff(rows.height_m) >= 500.0) and rows.height_m.max() <= 30000.0
    a_row = (rows.height_m + radius) * (1.0 + 1.0e-6 * rows.value)  # a = n r
    np.testing.assert_allclose(rows.value, 1.0e6 * n0 * np.exp(-(a_row - radius) / scale), rtol=0.01)


def test_land_rows_split_wind_and_keep_the_latest_update():
    cols = {"station_id": np.array(["A", "A", "B"]),
            "time": np.array(["2026-11-07T01:00"] * 3, "datetime64[s]"),
            "lat": np.array([55.6, 55.6, 55.7]), "lon": np.array([-3.3, -3.3, -3.2]),
            "station_elevation": np.array([50.0, 50.0, 60.0]), "update_sequence_number": np.array([1, 0, 0]),
            "2m_temperature": np.array([281.0, 280.0, 400.0]),
            "temperature_sensor_height": np.array([2.0, 2.0, 2.0]),
            "2m_dewpoint_temperature": np.array([np.nan, np.nan, 275.0]),
            "10m_wind_speed": np.array([5.0, 5.0, 3.0]), "10m_wind_direction": np.array([270.0, 270.0, 0.0]),
            "wind_sensor_height": np.array([10.0, 10.0, 30.0])}
    obs = obs_stores.land_rows(cols)
    assert obs.value[(obs.station == "A") & (obs.kind == "surface_temperature")].tolist() == [281.0]
    assert obs.value[(obs.station == "A") & (obs.kind == "surface_wind_u")] == pytest.approx([5.0])
    b = obs.kind[obs.station == "B"].tolist()
    assert b == ["surface_dewpoint"]  # 400 K fails the bounds; a 30 m mast is not a 10 m wind


def test_wind_components_follow_the_from_convention():
    u, v = obs_stores.wind_components(np.array([10.0, 10.0]), np.array([0.0, 90.0]))
    np.testing.assert_allclose(u, [0.0, -10.0], atol=1e-12)
    np.testing.assert_allclose(v, [-10.0, 0.0], atol=1e-12)


def test_mandatory_levels_take_near_reports_and_interpolate_close_brackets():
    p = np.array([100020.0, 95000.0, 87000.0, 50000.0])
    out = obs_stores.mandatory_levels(p, {"t": np.array([285.0, 283.0, 279.0, 255.0])})
    assert out[1000]["t"] == 285.0
    w = (np.log(92500.0) - np.log(87000.0)) / (np.log(95000.0) - np.log(87000.0))
    assert out[925]["t"] == pytest.approx(279.0 + w * 4.0)
    assert 850 not in out and 700 not in out  # 870 and 500 hPa are more than 15 % apart
    assert out[500]["t"] == 255.0


# --------------------------------------------------------------------------- the door
def test_the_grad_door_answers_its_own_help(capsys):
    from woof.cli import main as cli_main

    assert cli_main(["grad", "--help"]) == 0
    assert "usage: woof grad" in capsys.readouterr().out


def test_the_cost_command_writes_a_summary(tmp_path, capsys):
    from woof.grad.cli import main

    history = write_wrf(tmp_path / "wrfout_d01_2026-11-07_00:00:00")
    obs = _obs([{"platform": "synop", "kind": "surface_temperature", "time": _at(1), "lat": 55.6, "lon": -3.3,
                 "elevation_m": 50.0, "value": 283.0}])
    obs.save(tmp_path / "obs.npz")
    assert main(["cost", "--history", str(tmp_path), "--obs", str(tmp_path / "obs.npz"),
                 "--out", str(tmp_path / "j.json"), "--pairs-out", str(tmp_path / "pairs.npz"), "--quiet"]) == 0
    summary = json.loads((tmp_path / "j.json").read_text())
    assert summary["n"] == 1 and summary["schema"] == oc.SCHEMA
    assert history.exists() and (tmp_path / "pairs.npz").exists()


def test_no_forecast_path_imports_woof_grad():
    code = ("import sys, woof.cli, woof.components, woof.branch, woof.da.letkf\n"
            "bad = sorted(m for m in sys.modules if m == 'woof.grad' or m.startswith('woof.grad.'))\n"
            "print(','.join(bad))\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""
