"""Deterministic tests for the radiosonde profile referee (no network, no GPU)."""
from datetime import datetime, timezone

import numpy as np
import pytest

from woof.hex import sonde_profile_referee as r


def test_nearest_cells_respects_boundary_mask():
    lat = np.array([0.0, 0.0, 10.0])
    lon = np.array([0.0, 1.0, 0.0])
    xyz = r.unit_vectors(lat, lon)
    assert list(r.nearest_cells(xyz, [0.1], [0.9])) == [1]
    usable = np.array([True, False, True])
    assert list(r.nearest_cells(xyz, [0.1], [0.9], usable)) == [0]


def test_nearest_cells_never_returns_a_masked_cell():
    # antipode of (0, 0): the masked sentinel used to outscore every usable cell
    xyz = r.unit_vectors(np.array([0.0, 0.0]), np.array([0.0, 90.0]))
    assert list(r.nearest_cells(xyz, [0.0], [180.0], np.array([False, True]))) == [1]
    with pytest.raises(ValueError):
        r.nearest_cells(xyz, [0.0], [0.0], np.array([False, False]))


def test_interval_uses_the_cases_that_supply_the_metric():
    out = r.paired_interval(np.array([0.4, np.nan, np.nan, np.nan]), replicates=20, seed=1)
    assert out == {"mean_diff": 0.4, "ci95": None, "cases": 1, "exploratory": True}
    two = r.paired_interval(np.array([0.0, np.nan, 1.0]), replicates=200, seed=1)
    assert two["cases"] == 2 and two["ci95"][0] < two["ci95"][1]


def test_column_interpolation_is_linear_in_lnp_and_never_extrapolates():
    pressure = np.array([100000.0, 50000.0, 10000.0])
    # theta chosen so that T is 300, 250, 200 K at those pressures
    temps = np.array([300.0, 250.0, 200.0])
    theta = temps / (pressure / 1.0e5) ** r.RD_OVER_CP
    target = np.sqrt(100000.0 * 50000.0)  # midway in ln p
    assert r.column_temperature_at(theta, pressure, target) == pytest.approx(275.0)
    assert np.isnan(r.column_temperature_at(theta, pressure, 5000.0))
    assert np.isnan(r.column_temperature_at(theta, pressure, 101000.0))


def test_mandatory_levels_from_reported_levels():
    p = np.array([101000.0, 100000.0, 92500.0, 87000.0, 85000.0, 70000.0, 9000.0, 1000.0])
    t = np.array([290.0, 289.0, 285.0, 9999.0, 280.0, 270.0, 210.0, 230.0])
    got = r.mandatory_from_levels(p, t)
    assert got[1000] == 289.0 and got[925] == 285.0 and got[850] == 280.0 and got[700] == 270.0
    # 100 hPa lies between 90 and ... nothing within 15 %: not interpolated across a gap
    assert 100 not in got and 500 not in got
    assert got[10] == 230.0
    # a mandatory level absent but bracketed closely is interpolated in ln p
    p2 = np.array([26000.0, 24000.0])
    t2 = np.array([226.0, 222.0])
    w = (np.log(25000.0) - np.log(24000.0)) / (np.log(26000.0) - np.log(24000.0))
    assert r.mandatory_from_levels(p2, t2)[250] == pytest.approx(222.0 + w * 4.0)


def _record(case, arm, bias_by_lead):
    table = {"TR": {"250": {str(k): {"n": 10, "bias": v, "rmse": abs(v) + 1.0} for k, v in bias_by_lead.items()}}}
    return {"case": case, "arm": arm, "table": table}


def test_summarise_bias_rmse_and_drift():
    pairs = [(0, "s", 0.0, 0.0, 250, 220.0, 220.0), (24, "s", 0.0, 0.0, 250, 220.0, 221.0),
             (48, "s", 0.0, 0.0, 250, 220.0, 222.0), (24, "n", 70.0, 0.0, 250, 220.0, 219.0)]
    table = r.summarise(pairs)
    assert table["TR"]["250"]["24"] == {"n": 1, "bias": 1.0, "rmse": 1.0}
    assert table["TR"]["250"]["drift_k_per_24h"] == pytest.approx(1.0)
    assert table["NP"]["250"]["24"]["bias"] == -1.0
    assert table["GL"]["250"]["24"]["n"] == 2 and table["GL"]["250"]["24"]["bias"] == 0.0


def test_compare_is_case_paired_and_seeded():
    a = [_record(f"c{i}", "A", {0: 0.0, 24: 1.0 + 0.1 * i}) for i in range(5)]
    b = [_record(f"c{i}", "B", {0: 0.0, 24: 0.5}) for i in range(5)]
    one = r.compare(a, b, replicates=500, seed=7)
    two = r.compare(a, b, replicates=500, seed=7)
    assert one == two
    cell = one["table"]["TR"]["250"]["24"]["bias"]
    assert cell["mean_diff"] == pytest.approx(0.7)
    assert cell["cases"] == 5 and cell["ci95"][0] > 0.5 and cell["ci95"][1] < 0.9
    single = r.compare(a[:1], b[:1], replicates=10, seed=1)
    assert single["table"]["TR"]["250"]["24"]["bias"]["ci95"] is None
    with pytest.raises(ValueError):
        r.compare(a[:1], [_record("other", "B", {0: 0.0})])


def test_score_pairs_frames_at_synoptic_hours(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    lat = np.array([0.0, 45.0, -70.0])
    lon = np.array([0.0, 10.0, 100.0])
    pressure = np.tile(np.array([100000.0, 50000.0, 1000.0]), (3, 1))
    for hour, offset in ((0, 0.0), (6, 9.0), (12, 2.0)):
        path = tmp_path / f"cuda-history.2025-01-15_{hour:02d}.00.00.nc"
        with netCDF4.Dataset(path, "w") as d:
            d.createDimension("Time", None)
            d.createDimension("nCells", 3)
            d.createDimension("nVertLevels", 3)
            d.createDimension("StrLen", 64)
            x = d.createVariable("xtime", "S1", ("Time", "StrLen"))
            x[0, :] = np.frombuffer(f"2025-01-15_{hour:02d}:00:00".ljust(64).encode(), dtype="S1")
            d.createVariable("latCell", "f8", ("nCells",))[:] = np.radians(lat)
            d.createVariable("lonCell", "f8", ("nCells",))[:] = np.radians(lon)
            temps = np.tile(np.array([280.0, 250.0, 220.0]), (3, 1)) + offset
            d.createVariable("theta", "f4", ("Time", "nCells", "nVertLevels"))[0] = temps / (pressure / 1e5) ** r.RD_OVER_CP
            d.createVariable("pressure", "f4", ("Time", "nCells", "nVertLevels"))[0] = pressure
    t0 = datetime(2025, 1, 15, 0, tzinfo=timezone.utc)
    t12 = datetime(2025, 1, 15, 12, tzinfo=timezone.utc)
    sondes = {t0: [("S1", 44.0, 11.0, {500: 251.0, 10: 220.0})], t12: [("S1", 44.0, 11.0, {500: 251.0})]}
    out = r.score(str(tmp_path), "igra16", "case", "arm", soundings=sondes, pairs_out=str(tmp_path / "p.csv"))
    assert out["frames"] == ["cuda-history.2025-01-15_00.00.00.nc", "cuda-history.2025-01-15_12.00.00.nc"]
    nh = out["table"]["NH"]["500"]
    assert nh["0"]["bias"] == pytest.approx(-1.0, abs=1e-3)
    assert nh["12"]["bias"] == pytest.approx(1.0, abs=1e-3)
    assert nh["drift_k_per_24h"] == pytest.approx(4.0, abs=1e-2)
    assert out["table"]["NH"]["10"]["0"]["bias"] == pytest.approx(0.0, abs=1e-3)
    assert (tmp_path / "p.csv").read_text().count("\n") == 4


def test_initial_blacklist_uses_lead_zero_only():
    good = [(0, "A", 0.0, 0.0, lev, 250.0, 250.5) for lev in (850, 500, 250)]
    bad = [(0, "B", 0.0, 0.0, lev, 250.0, 262.0) for lev in (850, 500, 250)]
    late = [(48, "C", 0.0, 0.0, lev, 250.0, 270.0) for lev in (850, 500, 250)]
    short = [(0, "D", 0.0, 0.0, 850, 250.0, 270.0)]
    assert r.initial_blacklist(good + bad + late + short) == {"B"}
