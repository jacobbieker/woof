"""Deterministic tests for the cell-mean QPE referee (no network, no GPU)."""
import json
import struct
import hashlib
from datetime import datetime, timezone

import numpy as np
import pytest

from woof.hex import qpe_cell_referee as q


def test_cell_means_drop_partially_observed_cells():
    # three cells; cell 2 has half its pixels missing, cell 1 only a sliver of a full cell
    assignment = np.array([0] * 10 + [1] * 3 + [2] * 10)
    values = np.array([2.0] * 10 + [5.0] * 3 + [1.0] * 10)
    valid = np.array([1] * 10 + [1] * 3 + [1] * 5 + [0] * 5)
    mean, scored = q.cell_means(assignment, values, valid, 4)
    assert mean[0] == 2.0 and mean[1] == 5.0 and mean[2] == 1.0 and np.isnan(mean[3])
    assert list(scored) == [True, False, False, False]


def test_contingency_and_window_stats():
    model = np.array([0.0, 2.0, 12.0, 0.5])
    obs = np.array([0.0, 0.0, 11.0, 3.0])
    fbi, ets = q.contingency(model, obs, 1.0)
    assert fbi == pytest.approx(1.0)  # hits 1, false 1, misses 1
    chance = 2 * 2 / 4
    assert ets == pytest.approx((1 - chance) / (3 - chance))
    stats = q.window_stats(model, obs)
    assert stats["bias"] == pytest.approx(0.125) and stats["ratio"] == pytest.approx(14.5 / 14.0)
    assert stats["fbi_10"] == pytest.approx(1.0) and stats["cells"] == 4


def test_assign_pixels_honours_the_usable_mask():
    pytest.importorskip("scipy")
    lat = np.array([0.0, 0.0, 10.0])
    lon = np.array([0.0, 1.0, 0.0])
    assert list(q.assign_pixels(lat, lon, [0.1, 0.1], [0.6, 0.4])) == [1, 0]
    # the pixel a masked cell owns is dropped, not handed to its usable neighbour
    assert list(q.assign_pixels(lat, lon, [0.1, 0.1], [0.6, 0.4], np.array([True, False, True]))) == [-1, 0]


def test_compare_is_case_paired_and_seeded():
    def rec(case, arm, bias):
        return {"case": case, "arm": arm, "init": "2026-11-01T00:00:00+00:00", "hours": 24,
                "windows": {"24": {"bias": bias, "rmse": abs(bias) + 1, "end": "2026-11-02T00:00:00+00:00"}}}
    a = [rec(f"c{i}", "A", 1.0 + 0.1 * i) for i in range(5)]
    b = [rec(f"c{i}", "B", 0.5) for i in range(5)]
    one = q.compare(a, b, replicates=500, seed=3, stats=("bias", "rmse"))
    assert one == q.compare(a, b, replicates=500, seed=3, stats=("bias", "rmse"))
    assert one["stats"]["bias"]["mean_diff"] == pytest.approx(0.7)
    lo, hi = one["stats"]["bias"]["ci95"]
    assert 0.5 < lo < 0.7 < hi < 0.9
    assert q.compare(a[:1], b[:1], replicates=10, stats=("bias",))["stats"]["bias"]["ci95"] is None


def test_assign_pixels_rejects_pixels_outside_every_footprint():
    pytest.importorskip("scipy")
    lat = np.array([0.0, 0.0, 1.0])
    lon = np.array([0.0, 1.0, 0.0])
    # inside cell 1's footprint, then far outside the mesh
    assert list(q.assign_pixels(lat, lon, [0.1, 40.0], [0.9, 60.0])) == [1, -1]
    # a pixel beside a masked cell is not handed to a distant usable cell
    assert list(q.assign_pixels(lat, lon, [0.0], [1.0], np.array([True, False, True]))) == [-1]
    mean, scored = q.cell_means(np.array([0, 0, -1, -1]), np.array([1.0, 3.0, 50.0, 50.0]),
                                np.ones(4), 3)
    assert mean[0] == 2.0 and list(scored) == [True, False, False]


def _window_rec(case, arm, windows, init="2026-11-01T00:00:00+00:00", hours=24):
    return {"case": case, "arm": arm, "init": init, "hours": hours,
            "windows": {k: dict(v, end=f"lead{k}") for k, v in windows.items()}}


def test_compare_averages_only_the_windows_both_arms_scored():
    a = [_window_rec("c0", "A", {"24": {"bias": 0.0}, "48": {"bias": 10.0}})]
    b = [_window_rec("c0", "B", {"24": {"bias": 0.0}})]
    out = q.compare(a, b, replicates=10, stats=("bias",))
    assert out["windows"] == {"c0": ["24"]}
    assert out["stats"]["bias"]["mean_diff"] == 0.0
    with pytest.raises(ValueError, match="init or window length"):
        q.compare(a, [_window_rec("c0", "B", {"24": {"bias": 0.0}}, hours=6)], stats=("bias",))


def test_compare_pairs_finite_cases_for_means_and_intervals():
    a = [_window_rec("c0", "A", {"24": {"fbi_10": 2.0}}),
         _window_rec("c1", "A", {"24": {"fbi_10": 10.0}})]
    b = [_window_rec("c0", "B", {"24": {"fbi_10": 1.0}}),
         _window_rec("c1", "B", {"24": {"fbi_10": None}})]
    row = q.compare(a, b, replicates=50, stats=("fbi_10",))["stats"]["fbi_10"]
    assert (row["a"], row["b"], row["mean_diff"], row["cases"]) == (2.0, 1.0, 1.0, 1)
    assert row["ci95"] is None and row["exploratory"]


def test_compare_pairs_windows_where_both_arms_define_the_statistic():
    a = [_window_rec("c0", "A", {"24": {"ets_1": None}, "48": {"ets_1": 1.0}})]
    b = [_window_rec("c0", "B", {"24": {"ets_1": 0.0}, "48": {"ets_1": 1.0}})]
    row = q.compare(a, b, replicates=10, stats=("ets_1",))["stats"]["ets_1"]
    assert (row["a"], row["b"], row["mean_diff"]) == (1.0, 1.0, 0.0)


def _pack(path, schema, meta, arrays):
    payload = b""
    index = {}
    for key, arr in arrays.items():
        dtype = "<u1" if arr.dtype == np.uint8 else "<f8"
        data = np.ascontiguousarray(arr.astype(dtype)).tobytes()
        index[key] = {"dtype": dtype, "shape": list(arr.shape), "offset": len(payload), "bytes": len(data)}
        payload += data
    meta = dict(meta, schema=schema, arrays=index, content_sha256=hashlib.sha256(payload).hexdigest())
    blob = json.dumps(meta).encode()
    header = b"GPWMOBS1" + struct.pack("<II", 1, len(blob)) + struct.pack("<Q", len(payload))
    path.write_bytes(header.ljust(64, b"\0") + blob + payload)


def test_score_differences_frames_across_the_window(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    pytest.importorskip("scipy")
    lat = np.array([30.0, 40.0])
    lon = np.array([-100.0, -90.0])
    for day, (rc, rn) in ((1, (0.0, 1.0)), (2, (2.0, 5.0))):
        path = tmp_path / f"cuda-history.2026-11-0{day}_12.00.00.nc"
        with netCDF4.Dataset(path, "w") as d:
            d.createDimension("Time", None)
            d.createDimension("nCells", 2)
            d.createDimension("StrLen", 64)
            x = d.createVariable("xtime", "S1", ("Time", "StrLen"))
            x[0, :] = np.frombuffer(f"2026-11-0{day}_12:00:00".ljust(64).encode(), dtype="S1")
            d.createVariable("latCell", "f8", ("nCells",))[:] = np.radians(lat)
            d.createVariable("lonCell", "f8", ("nCells",))[:] = np.radians(lon)
            d.createVariable("rainc", "f4", ("Time", "nCells"))[0] = [rc, 0.0]
            d.createVariable("rainnc", "f4", ("Time", "nCells"))[0] = [rn, 3.0 * day]
    plat = np.array([[29.9, 30.1], [39.9, 40.1]])
    plon = np.array([[-100.1, -99.9], [-90.1, -89.9]])
    _pack(tmp_path / "geometry.obspack", "gpuwm-obs.obs-geo.v1", {}, {"latitude": plat, "longitude": plon})
    _pack(tmp_path / "ST4_2026110212_24h.obspack", "gpuwm-obs.obs-grid.v1",
          {"valid_time": "2026-11-02T12:00:00", "accumulation_hours": 24},
          {"values": np.array([[4.0, 6.0], [1.0, 1.0]]), "valid": np.ones((2, 2), dtype=np.uint8)})
    record = q.score(str(tmp_path), str(tmp_path), "case", "arm")
    w = record["windows"]["24"]
    # model: cell 0 6 mm, cell 1 3 mm; observed: 5 mm and 1 mm
    assert w["cells"] == 2 and w["model_mean"] == pytest.approx(4.5) and w["obs_mean"] == pytest.approx(3.0)
    assert w["bias"] == pytest.approx(1.5) and w["end"].startswith("2026-11-02T12")
