"""Deterministic tests for the surface-station 10 m wind referee (no network, no GPU)."""
from datetime import datetime, timezone

import numpy as np
import pytest

from woof.hex import station_wind_referee as r

T0 = datetime(2026, 10, 28, 0, tzinfo=timezone.utc)
T6 = datetime(2026, 10, 28, 6, tzinfo=timezone.utc)


def _table(rows):
    keys = r.FIELDS
    cols = {k: [] for k in keys}
    for row in rows:
        for k, v in zip(keys, row):
            cols[k].append(v)
    cols["time"] = np.array(cols["time"], dtype="datetime64[ns]")
    return {k: np.asarray(v) for k, v in cols.items()}


def test_select_reports_filters_height_bounds_and_time():
    t0 = np.datetime64("2026-10-28T00:00")
    rows = [
        ("A", 10.0, 20.0, 100.0, 10.0, 5.0, t0),
        ("B", 11.0, 21.0, 100.0, 30.0, 5.0, t0),         # mast too tall
        ("C", 12.0, 22.0, 100.0, np.nan, 80.0, t0),      # speed out of bounds
        ("D", 13.0, 23.0, 100.0, np.nan, 3.0, t0),       # no stated height: standard 10 m
        ("E", 14.0, 24.0, 100.0, 10.0, 4.0, t0 + np.timedelta64(30, "m")),  # off the hour
        ("A", 10.0, 20.0, 100.0, 10.0, 6.0, t0),         # an update: the last row wins
    ]
    out = r.select_reports(_table(rows), [T0, T6])
    assert [x[0] for x in out[T0]] == ["A", "D"]
    assert out[T0][0][4] == 6.0
    assert out[T6] == []


def _mesh(n=2000):
    k = np.arange(n) + 0.5
    lat = np.degrees(np.arcsin(1.0 - 2.0 * k / n))
    lon = np.degrees(np.pi * (1.0 + 5.0 ** 0.5) * k) % 360.0 - 180.0
    return lat, lon


def test_pairing_drops_stations_whose_cell_misrepresents_the_site():
    lat, lon = _mesh()
    xyz = r.unit_vectors(lat, lon)
    ter = np.zeros(lat.size)
    frame = {"speed": np.full(lat.size, 7.0)}
    reports = [("low", float(lat[10]), float(lon[10]), 50.0, 5.0),
               ("peak", float(lat[20]), float(lon[20]), 2500.0, 5.0),
               ("unknown", float(lat[30]), float(lon[30]), np.nan, 5.0)]
    rows = r.pair_frame(frame, reports, xyz, None, ter)
    assert [row[0] for row in rows] == ["low"]
    assert rows[0][3:] == (5.0, 7.0)


def test_summarise_bias_rmse_and_drift():
    pairs = [(0, "s1", 45.0, 0.0, 5.0, 6.0), (0, "s2", -45.0, 0.0, 4.0, 4.0),
             (24, "s1", 45.0, 0.0, 5.0, 7.0), (24, "s2", -45.0, 0.0, 4.0, 6.0)]
    t = r.summarise(pairs)
    assert t["GL"]["0"]["n"] == 2
    assert t["GL"]["0"]["bias"] == pytest.approx(0.5)
    assert t["GL"]["24"]["rmse"] == pytest.approx(2.0)
    assert t["GL"]["24"]["obs_mean"] == pytest.approx(4.5)
    assert t["GL"]["drift_m_s_per_24h"] == pytest.approx(1.5)
    assert set(t) == {"GL", "NH", "SH"}


def test_initial_blacklist_uses_lead_zero_only():
    pairs = [(0, "bad", 0.0, 0.0, 1.0, 30.0), (6, "late", 0.0, 0.0, 1.0, 30.0), (0, "ok", 0.0, 0.0, 5.0, 6.0)]
    assert r.initial_blacklist(pairs) == {"bad"}


def _record(case, arm, bias):
    return {"case": case, "arm": arm,
            "table": {"GL": {"24": {"n": 10, "bias": bias, "rmse": abs(bias) + 1.0, "obs_mean": 4.0}}}}


def test_compare_pairs_cases_and_is_seeded():
    a = [_record(c, "topo", b) for c, b in (("c1", 0.2), ("c2", 0.1), ("c3", 0.3), ("c4", 0.0))]
    b = [_record(c, "base", b) for c, b in (("c1", 1.2), ("c2", 1.0), ("c3", 1.4), ("c4", 0.9))]
    out = r.compare(a, b, replicates=500, seed=7)
    cell = out["table"]["GL"]["24"]["bias"]
    assert out["cases"] == ["c1", "c2", "c3", "c4"]
    assert cell["mean_diff"] == pytest.approx(-0.975)
    assert cell["ci95"][1] < 0.0
    assert cell == r.compare(a, b, replicates=500, seed=7)["table"]["GL"]["24"]["bias"]
    with pytest.raises(ValueError, match="share no case"):
        r.compare(a, [_record("other", "base", 0.0)])


def _history(tmp_path, with_ter=True):
    netCDF4 = pytest.importorskip("netCDF4")
    lat, lon = _mesh()
    for hour, speed in ((0, 5.0), (6, 8.0)):
        with netCDF4.Dataset(tmp_path / f"history.2026-10-28_{hour:02d}.00.00.nc", "w") as d:
            d.createDimension("Time", None)
            d.createDimension("nCells", lat.size)
            d.createDimension("StrLen", 64)
            x = d.createVariable("xtime", "S1", ("Time", "StrLen"))
            x[0, :] = np.frombuffer(f"2026-10-28_{hour:02d}:00:00".ljust(64).encode(), dtype="S1")
            d.createVariable("latCell", "f8", ("nCells",))[:] = np.radians(lat)
            d.createVariable("lonCell", "f8", ("nCells",))[:] = np.radians(lon)
            if with_ter:
                d.createVariable("ter", "f4", ("nCells",))[:] = 0.0
            d.createVariable("u10", "f4", ("Time", "nCells"))[0] = speed * 0.6
            d.createVariable("v10", "f4", ("Time", "nCells"))[0] = speed * 0.8
    return lat, lon


def test_score_on_a_synthetic_history(tmp_path):
    lat, lon = _history(tmp_path)
    station = ("s", float(lat[100]), float(lon[100]), 10.0)
    reports = {T0: [station + (4.0,)], T6: [station + (6.0,)]}
    rec = r.score(str(tmp_path), "c", "a", reports=reports)
    assert rec["pairs"] == 2
    assert rec["table"]["GL"]["0"]["bias"] == pytest.approx(1.0, abs=1e-5)
    assert rec["table"]["GL"]["6"]["bias"] == pytest.approx(2.0, abs=1e-5)
    assert rec["table"]["GL"]["drift_m_s_per_24h"] == pytest.approx(4.0, abs=1e-4)


def test_score_refuses_a_history_without_terrain_unless_given_a_file(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    lat, lon = _history(tmp_path, with_ter=False)
    station = ("s", float(lat[100]), float(lon[100]), 10.0)
    reports = {T0: [station + (4.0,)], T6: [station + (6.0,)]}
    with pytest.raises(ValueError, match="carries no ter"):
        r.score(str(tmp_path), "c", "a", reports=reports)
    static = tmp_path / "static.nc"
    with netCDF4.Dataset(static, "w") as d:
        d.createDimension("nCells", lat.size)
        d.createVariable("ter", "f4", ("nCells",))[:] = 1000.0
    rec = r.score(str(tmp_path), "c", "a", reports=reports, terrain_file=str(static))
    assert rec["pairs"] == 0  # the station sits 990 m below its cell: dropped, not scored
