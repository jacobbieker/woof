"""UKV from the Source Cooperative archive: the plane, the window, the files.

Everything here runs on synthetic arrays; the archive itself is read only by
the network test at the bottom (WOOF_NETWORK_TESTS=1).
"""
from __future__ import annotations

from datetime import datetime, timedelta
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from woof import ukv_source as ukv
from woof.fetch import Area


# --------------------------------------------------------------------------
# The plane
# --------------------------------------------------------------------------

def test_projection_origin_and_round_trip():
    x, y = ukv.laea_forward(ukv.LAEA_LATITUDE_ORIGIN, ukv.LAEA_LONGITUDE_ORIGIN)
    assert abs(float(x)) < 1e-6 and abs(float(y)) < 1e-6
    lat, lon = ukv.laea_inverse(0.0, 0.0)
    assert lat == pytest.approx(54.9, abs=1e-7) and lon == pytest.approx(-2.5)
    rng = np.random.default_rng(1)
    lat = rng.uniform(46.0, 61.0, 500)
    lon = rng.uniform(-16.0, 8.0, 500)
    back_lat, back_lon = ukv.laea_inverse(*ukv.laea_forward(lat, lon))
    assert np.abs(back_lat - lat).max() < 1e-7
    assert np.abs(back_lon - lon).max() < 1e-9


def test_projection_is_ellipsoidal_equal_area():
    # The authalic sphere preserves area: a small cell maps to
    # the ellipsoidal area element, which a spherical formula misses by
    # about 0.7 percent at these latitudes.
    lat, lon, d = 53.2, -0.5, 1e-4
    xs, ys = ukv.laea_forward([lat, lat, lat + d], [lon, lon + d, lon])
    planar = abs((xs[1] - xs[0]) * (ys[2] - ys[0]) - (ys[1] - ys[0]) * (xs[2] - xs[0]))
    a, b = ukv.LAEA_SEMI_MAJOR, ukv.LAEA_SEMI_MINOR
    e2 = 1 - (b / a) ** 2
    phi = math.radians(lat + d / 2)
    m = a * (1 - e2) / (1 - e2 * math.sin(phi) ** 2) ** 1.5
    n = a / math.sqrt(1 - e2 * math.sin(phi) ** 2)
    ellipsoidal = m * math.radians(d) * n * math.cos(phi) * math.radians(d)
    assert planar == pytest.approx(ellipsoidal, rel=1e-4)


def test_grid_corners_match_the_measured_plane():
    # Inverse of the archive's corner cell centres, as read from the store's
    # projection coordinates on 2026-10-18.
    x_last = ukv.GRID_X0 + (ukv.GRID_NX - 1) * ukv.GRID_SPACING
    y_last = ukv.GRID_Y0 + (ukv.GRID_NY - 1) * ukv.GRID_SPACING
    assert (x_last, y_last) == (924000.0, 902000.0)
    lat, lon = ukv.laea_inverse([ukv.GRID_X0, x_last, ukv.GRID_X0, x_last],
                                [ukv.GRID_Y0, ukv.GRID_Y0, y_last, y_last])
    assert np.allclose(lat, [44.517, 44.900, 61.319, 61.921], atol=2e-3)
    assert np.allclose(lon, [-17.117, 9.213, -24.510, 15.280], atol=2e-3)


def test_registry_window_is_the_plane_box_less_the_fetch_margin():
    from woof.source_adapters import get_source_adapter
    window = get_source_adapter("ukv").coverage_window
    south, west, north, east = ukv.coverage_box()
    step, margin = ukv.TARGET_STEP_DEGREES, ukv.MARGIN_DEGREES
    expected = (math.ceil((south + margin) / step) * step,
                math.ceil((west + margin) / step) * step,
                math.floor((north - margin) / step) * step,
                math.floor((east - margin) / step) * step)
    assert window.envelope() == pytest.approx(expected, abs=1e-9)
    assert window.nx == round((window.east - window.west) / step) + 1
    assert window.ny == round((window.north - window.south) / step) + 1
    # Every point of the window plus the fetch's margin lands on the plane:
    # the fetch's own regrid accepts the whole box.
    lat, lon = ukv.target_axes(Area(window.south, window.west, window.north, window.east))
    ukv.Regridder(lat, lon)


# --------------------------------------------------------------------------
# The window
# --------------------------------------------------------------------------

#: A fixed "now" for the live bucket's rolling two years.
NOW = datetime(2026, 10, 18, 12)


def test_window_inside_the_archive():
    times = ukv.validate_window(datetime(2024, 6, 18, 6), 3)
    assert times == tuple(datetime(2024, 6, 18, 6) + timedelta(hours=h) for h in range(4))
    assert ukv.validate_window(datetime(2024, 6, 18, 6), 6, 3)[-1] == datetime(2024, 6, 18, 12)
    plan = ukv.plan_window(ukv.COVERAGE_END, 0, now=NOW)
    assert (plan.store, plan.times) == ("archive", (ukv.COVERAGE_END,))


@pytest.mark.parametrize("hour, last", [(3, 120), (15, 120), (0, 54), (6, 54), (9, 54),
                                        (12, 54), (18, 54), (21, 54), (1, 12), (23, 12)])
def test_live_run_ladders(hour, last):
    leads = ukv.run_leads(hour)
    assert max(leads) == last and set(range(0, min(last, 54) + 1)) <= leads
    if last == 120:
        assert 55 not in leads and 57 in leads and 117 in leads


def test_window_past_the_archive_is_one_live_run():
    plan = ukv.plan_window(datetime(2026, 10, 16, 6), 6, now=NOW)
    assert plan.store == "live" and plan.runs[0] == datetime(2026, 10, 16, 6)
    # Every candidate run holds every valid time.
    for run in plan.runs:
        leads = ukv.run_leads(run.hour)
        assert all(int((valid - run).total_seconds() // 3600) in leads for valid in plan.times)
    # A window crossing the archive's end is read live too.
    assert ukv.plan_window(datetime(2025, 8, 13, 21), 2, now=NOW).store == "live"


def test_a_long_window_takes_the_long_runs_at_three_hours():
    # 66 h from 00Z: no run is hourly that far, the 03Z/15Z runs are at 3 h.
    with pytest.raises(ValueError, match="--cadence 3"):
        ukv.plan_window(datetime(2026, 10, 25, 0), 66, now=datetime(2026, 10, 26))
    plan = ukv.plan_window(datetime(2026, 10, 25, 0), 66, 3, now=datetime(2026, 10, 26))
    assert plan.store == "live"
    assert plan.runs[:2] == (datetime(2026, 10, 24, 15), datetime(2026, 10, 24, 3))


@pytest.mark.parametrize("cycle, hours", [
    (datetime(2023, 12, 31, 23), 1),
    (datetime(2022, 1, 1, 0), 0),
])
def test_window_outside_both_stores_refuses(cycle, hours):
    with pytest.raises(ValueError, match="leaves the archive") as error:
        ukv.plan_window(cycle, hours, now=NOW)
    assert "2024-01-01 00Z to 2025-08-13 22Z" in str(error.value)


def test_the_archive_hour_without_wind_direction_is_read_live_or_refused():
    plan = ukv.plan_window(datetime(2025, 5, 16, 12), 6, now=NOW)
    assert plan.store == "live"
    # Once the bucket has dropped that run, nothing serves the hour.
    with pytest.raises(ValueError, match="no pressure-level wind direction"):
        ukv.plan_window(datetime(2025, 5, 16, 12), 6, now=datetime(2028, 1, 1))
    # A cadence that steps over the hour stays in the archive.
    plan = ukv.plan_window(datetime(2025, 5, 16, 14), 2, 2, now=NOW)
    assert plan.store == "archive" and ukv.WIND_DIRECTION_GAP not in plan.times


def test_latest_is_the_newest_complete_run(monkeypatch):
    monkeypatch.setattr(ukv, "datetime", _FrozenDatetime)
    complete = {datetime(2026, 10, 18, 9)}
    monkeypatch.setattr(ukv, "live_missing",
                        lambda run, times: [] if run in complete else ["absent"])
    assert ukv.latest_cycle(6) == datetime(2026, 10, 18, 9)


class _FrozenDatetime(datetime):
    @classmethod
    def utcnow(cls):
        return cls(2026, 10, 18, 12, 30)


@pytest.mark.parametrize("hours, cadence", [(-1, None), (3, 2), (2, 0)])
def test_window_shape_refusals(hours, cadence):
    with pytest.raises(ValueError):
        ukv.validate_window(datetime(2024, 6, 18, 6), hours, cadence)


def test_target_axes_are_step_multiples_with_margin():
    lat, lon = ukv.target_axes(Area(52.9, -1.0, 53.5, 0.0))
    assert lat[0] <= 52.9 - ukv.MARGIN_DEGREES and lat[-1] >= 53.5 + ukv.MARGIN_DEGREES
    assert lon[0] <= -1.0 - ukv.MARGIN_DEGREES and lon[-1] >= 0.0 + ukv.MARGIN_DEGREES
    assert np.allclose(np.diff(lat), ukv.TARGET_STEP_DEGREES)
    assert np.allclose(lat / ukv.TARGET_STEP_DEGREES, np.round(lat / ukv.TARGET_STEP_DEGREES))
    # The margin is about a margin, not a whole extra step.
    assert lat[0] > 52.9 - ukv.MARGIN_DEGREES - ukv.TARGET_STEP_DEGREES


def test_target_axes_need_an_area():
    with pytest.raises(ValueError, match="needs --area"):
        ukv.target_axes(None)


@pytest.mark.parametrize("area", [Area(40.0, -4.0, 41.0, -3.0),     # Iberia
                                  Area(53.0, 20.0, 54.0, 21.0)])     # Baltic
def test_domain_off_the_plane_refuses(area):
    lat, lon = ukv.target_axes(area)
    with pytest.raises(ValueError, match="cannot serve this domain") as error:
        ukv.Regridder(lat, lon)
    assert "leave the UKV plane" in str(error.value)


# --------------------------------------------------------------------------
# Regrid, winds, terrain
# --------------------------------------------------------------------------

def test_bilinear_regrid_reproduces_a_linear_field():
    lat, lon = ukv.target_axes(Area(53.0, -1.0, 53.4, -0.4))
    regrid = ukv.Regridder(lat, lon)
    rows, cols = regrid.native_slices()
    y = ukv.GRID_Y0 + ukv.GRID_SPACING * np.arange(rows.start, rows.stop)
    x = ukv.GRID_X0 + ukv.GRID_SPACING * np.arange(cols.start, cols.stop)
    native = 3.0 * x[None, :] - 2.0 * y[:, None] + 7.0
    stacked = np.stack([native, 2.0 * native])
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    px, py = ukv.laea_forward(lat2, lon2)
    expected = 3.0 * px - 2.0 * py + 7.0
    assert np.allclose(regrid(native), expected, rtol=0, atol=1e-5)
    assert np.allclose(regrid(stacked)[1], 2.0 * expected, rtol=0, atol=1e-5)


def test_wind_components_from_true_north_from_direction():
    u, v = ukv.wind_components([10.0, 10.0, 10.0, 0.0], [0.0, 90.0, 270.0, 123.0])
    assert np.allclose(u, [0.0, -10.0, 10.0, 0.0], atol=1e-12)
    assert np.allclose(v, [-10.0, 0.0, 0.0, 0.0], atol=1e-12)


def test_surface_height_of_an_isothermal_column():
    temperature = 260.0
    scale = ukv._RD * temperature / ukv._GRAVITY
    levels = np.array(ukv.PRESSURE_LEVELS_PA, dtype=float)
    shape = (len(levels), 2, 3)
    height = (scale * np.log(100000.0 / levels))[:, None, None] * np.ones(shape)
    psfc = np.array([[101500.0, 100000.0, 97000.0], [88000.0, 64000.0, 30500.0]])
    dry = np.zeros(shape)
    result = ukv.surface_height(levels, height, np.full(shape, temperature), dry,
                                psfc, np.full(psfc.shape, temperature), np.zeros(psfc.shape))
    assert np.allclose(result, scale * np.log(100000.0 / psfc), atol=1e-6)


def test_surface_height_refuses_a_surface_above_the_ladder():
    levels = np.array(ukv.PRESSURE_LEVELS_PA, dtype=float)
    ones = np.ones((len(levels), 1, 1))
    with pytest.raises(ValueError, match="above the highest published level"):
        ukv.surface_height(levels, ones, ones * 250, ones * 0, np.array([[900.0]]),
                           np.array([[250.0]]), np.array([[0.0]]))


def test_era5_nearest_index_takes_the_lower_index_on_ties():
    rows, cols = ukv.era5_nearest_indices(np.array([90.0, 53.125, 53.13, -90.0]),
                                          np.array([-0.125, 0.0, 359.9, -180.0]))
    assert rows.tolist() == [0, 147, 147, 720]
    # -0.125 sits halfway between 359.75 (1439) and 0 (0): the lower index.
    assert cols.tolist() == [0, 0, 0, 720]


# --------------------------------------------------------------------------
# The store's measured layout
# --------------------------------------------------------------------------

class _Array:
    """A lazy stand-in for one archive array: values computed per selection."""

    def __init__(self, shape, value=None, attrs=None):
        self.shape = shape
        self.attrs = attrs or {}
        self._value = value

    def __getitem__(self, selection):
        if self._value is not None and not callable(self._value):
            return self._value[selection]
        # One index vector per axis, the integer axes dropped, broadcast
        # against each other the way an outer product is.
        vectors = [np.atleast_1d(np.arange(size)[part])
                   for size, part in zip(self.shape, selection)]
        kept = [not isinstance(part, int) for part in selection]
        grids = np.ix_(*vectors)
        values = self._value(*grids)
        return np.asarray(values, dtype=np.float64).reshape(
            [len(vector) for vector, keep in zip(vectors, kept) if keep])


def _store(*, short_direction=True, levels=33):
    count = int((ukv.COVERAGE_END - ukv.COVERAGE_START).total_seconds() // 3600) + 1
    origin = int((ukv.COVERAGE_START - datetime(1970, 1, 1)).total_seconds() * 1000)
    stamps = origin + 3_600_000 * np.arange(count, dtype=np.int64)
    ny, nx = ukv.GRID_NY, ukv.GRID_NX
    pressure = np.array(ukv.PRESSURE_LEVELS_PA, dtype=float)

    def level_field(function):
        def make(t, k, j, i):
            return function(t, pressure[k], j, i)
        return make

    rd_g = ukv._RD * 280.0 / ukv._GRAVITY
    fields = {
        "time": _Array((count,), stamps, {"units": "milliseconds since 1970-01-01"}),
        "pressure": _Array((levels,), pressure.astype(np.float32), {"units": "Pa"}),
        "projection_x_coordinate": _Array(
            (nx,), (ukv.GRID_X0 + ukv.GRID_SPACING * np.arange(nx)).astype(np.float32)),
        "projection_y_coordinate": _Array(
            (ny,), (ukv.GRID_Y0 + ukv.GRID_SPACING * np.arange(ny)).astype(np.float32)),
        "air_temperature": _Array((count, levels, ny, nx), level_field(
            lambda t, p, j, i: 280.0 + 0 * (t + j + i) + 0 * p)),
        "geopotential_height": _Array((count, levels, ny, nx), level_field(
            lambda t, p, j, i: rd_g * np.log(100000.0 / p) + 0 * (t + j + i))),
        "relative_humidity": _Array((count, levels, ny, nx), level_field(
            lambda t, p, j, i: 0.5 + 0 * (t + j + i + p))),
        "wind_speed": _Array((count, levels, ny, nx), level_field(
            lambda t, p, j, i: 10.0 + 0 * (t + j + i + p))),
        "wind_from_direction": _Array(
            (count - 1 if short_direction else count, levels, ny, nx), level_field(
                lambda t, p, j, i: 270.0 + 0 * (t + j + i + p))),
        "surface_air_pressure": _Array((count, ny, nx),
                                       lambda t, j, i: 98000.0 + 0 * (t + j + i)),
        "surface_temperature": _Array((count, ny, nx), lambda t, j, i: 285.0 + 0 * (t + j + i)),
        "air_temperature_1.5m": _Array((count, ny, nx), lambda t, j, i: 280.0 + 0 * (t + j + i)),
        "relative_humidity_1.5m": _Array((count, ny, nx), lambda t, j, i: 0.5 + 0 * (t + j + i)),
        "lwe_thickness_of_surface_snow_amount": _Array((count, ny, nx),
                                                       lambda t, j, i: 0.0 * (t + j + i)),
        "wind_speed_10.0m": _Array((count, ny, nx), lambda t, j, i: 4.0 + 0 * (t + j + i)),
        "wind_from_direction_10.0m": _Array((count, ny, nx),
                                            lambda t, j, i: 180.0 + 0 * (t + j + i)),
    }
    units = {"air_temperature": "K", "geopotential_height": "m", "relative_humidity": "1",
             "wind_speed": "m s-1", "wind_from_direction": "degrees",
             "surface_air_pressure": "Pa", "surface_temperature": "K",
             "air_temperature_1.5m": "K", "relative_humidity_1.5m": "1",
             "lwe_thickness_of_surface_snow_amount": "m", "wind_speed_10.0m": "m s-1",
             "wind_from_direction_10.0m": "degrees"}
    for name, unit in units.items():
        fields[name].attrs["units"] = unit
    return fields


def test_store_indices_shift_the_direction_after_the_gap():
    store = _store()
    before = datetime(2025, 5, 16, 14)
    after = datetime(2025, 5, 16, 16)
    indices = ukv.store_indices(store, (ukv.COVERAGE_START, before, after, ukv.COVERAGE_END))
    assert indices[ukv.COVERAGE_START] == (0, 0)
    assert indices[before][0] == indices[before][1]
    assert indices[after][1] == indices[after][0] - 1
    last = indices[ukv.COVERAGE_END]
    assert last[1] == store["wind_from_direction"].shape[0] - 1


def test_store_indices_refuse_a_changed_layout():
    with pytest.raises(ValueError, match="no longer has the layout") as error:
        ukv.store_indices(_store(short_direction=False), (ukv.COVERAGE_START,))
    assert "array shapes" in str(error.value)
    percent = _store()
    percent["relative_humidity"].attrs["units"] = "%"
    with pytest.raises(ValueError, match="field units"):
        ukv.store_indices(percent, (ukv.COVERAGE_START,))
    flipped = _store()
    flipped["pressure"] = _Array((33,), np.array(ukv.PRESSURE_LEVELS_PA[::-1], dtype=float),
                                 {"units": "Pa"})
    with pytest.raises(ValueError, match="top first"):
        ukv.store_indices(flipped, (ukv.COVERAGE_START,))


def _fake_era5(valid, latitude, longitude, *, workers=4):
    shape = (len(latitude), len(longitude))
    fields = {output: np.full(shape, value) for output, value in (
        ("land_area_fraction", 1.0), ("sea_ice_area_fraction", np.nan))}
    for k in range(1, 5):
        fields[f"soil_temperature_layer_{k}"] = np.full(shape, 285.0 - k)
        fields[f"soil_moisture_layer_{k}"] = np.full(shape, 0.3)
    return fields, {"source": "era5", "store": "synthetic", "valid_time": valid.isoformat(),
                    "negative_soil_water_floored": 0}


_REAL_LAND_STATE_SOURCE = ukv.land_state_source


@pytest.fixture(autouse=True)
def _era5_holds_every_window(monkeypatch):
    """No test asks ARCO for its bounds: ERA5 is taken to hold the window."""

    monkeypatch.setattr(ukv, "land_state_source", lambda times, **_: "era5")


def test_acquire_writes_the_files_and_front_door(tmp_path, monkeypatch):
    from netCDF4 import Dataset
    from woof import fetch_routes
    monkeypatch.setattr(ukv, "open_store", lambda: (_store(), "synthetic-snapshot"))
    monkeypatch.setattr(ukv, "read_era5", _fake_era5)
    area = Area(53.1, -0.6, 53.3, -0.4)
    manifest = ukv.acquire(cycle=datetime(2025, 5, 16, 12), hours=2, cadence=None,
                           area=area, out=tmp_path, progress=lambda text: None)
    names = [entry["path"] for entry in manifest["files"]]
    assert names == ["ukv-20250516T12Z.nc", "ukv-20250516T13Z.nc", "ukv-20250516T14Z.nc"]
    assert manifest["store_snapshots"] == ["synthetic-snapshot"]
    with Dataset(tmp_path / names[0]) as dataset:
        assert dataset.variables["time"].units == "seconds since 1970-01-01 00:00:00"
        assert dataset.variables["pressure"][:].tolist() == list(ukv.PRESSURE_LEVELS_PA)
        u = dataset.variables["eastward_wind"][0]
        assert np.allclose(u, 10.0, atol=1e-4)          # from the west
        v10 = dataset.variables["northward_wind_10m"][0]
        assert np.allclose(v10, 4.0, atol=1e-4)          # from the south
        terrain = dataset.variables["surface_altitude"][0]
        expected = ukv._RD * 280.0 / ukv._GRAVITY * math.log(100000.0 / 98000.0)
        assert np.allclose(terrain, expected, rtol=1e-3)
        assert dataset.variables["sea_ice_area_fraction"][0].mask.all()
        assert dataset.getncattr("land_state_source") == "era5"
    arguments = json.loads((tmp_path / fetch_routes.PREP_ARGUMENTS_NAME).read_text())
    assert arguments["source"] == "ukv"
    supplements = [arguments["argv"][i + 1] for i, token in enumerate(arguments["argv"])
                   if token == "--supplement"]
    assert supplements == [f"{ukv.TERRAIN_ROLE}={tmp_path / name}" for name in names]
    inputs = (tmp_path / fetch_routes.INPUT_LIST_NAME).read_text().split()
    assert inputs == [str(tmp_path / name) for name in names]
    # The same request reuses the verified files; another request refetches.
    again = ukv.acquire(cycle=datetime(2025, 5, 16, 12), hours=2, cadence=None,
                        area=area, out=tmp_path, progress=lambda text: None)
    assert again == manifest
    ukv.check_prior_request(tmp_path, cycle=datetime(2025, 5, 16, 12), hours=2,
                            cadence=None, area=area)
    with pytest.raises(ValueError, match="another acquisition request"):
        ukv.check_prior_request(tmp_path, cycle=datetime(2025, 5, 16, 12), hours=1,
                                cadence=None, area=area)


def test_acquire_resumes_an_interrupted_window(tmp_path, monkeypatch):
    from woof import fetch_routes
    monkeypatch.setattr(ukv, "open_store", lambda: (_store(), "s"))
    calls = []

    def flaky(valid, latitude, longitude, *, workers=4):
        calls.append(valid)
        if valid.hour == 8 and len(calls) == 3:
            raise RuntimeError("ERA5 read failed")
        return _fake_era5(valid, latitude, longitude)

    monkeypatch.setattr(ukv, "read_era5", flaky)
    area = Area(53.1, -0.6, 53.2, -0.5)
    with pytest.raises(RuntimeError, match="ERA5 read failed"):
        ukv.acquire(cycle=datetime(2024, 6, 18, 6), hours=2, cadence=None, area=area,
                    out=tmp_path, progress=lambda text: None)
    first = (tmp_path / "ukv-20240618T06Z.nc").read_bytes()
    manifest = ukv.acquire(cycle=datetime(2024, 6, 18, 6), hours=2, cadence=None, area=area,
                           out=tmp_path, progress=lambda text: None)
    # The two finished hours were kept, the failed one read again.
    assert [valid.hour for valid in calls] == [6, 7, 8, 8]
    assert (tmp_path / "ukv-20240618T06Z.nc").read_bytes() == first
    assert len(manifest["files"]) == 3
    # A lost handoff is written again when the verified window is reused.
    (tmp_path / fetch_routes.PREP_ARGUMENTS_NAME).unlink()
    ukv.acquire(cycle=datetime(2024, 6, 18, 6), hours=2, cadence=None, area=area,
                out=tmp_path, progress=lambda text: None)
    assert (tmp_path / fetch_routes.PREP_ARGUMENTS_NAME).is_file()


def test_acquire_reads_the_direction_one_record_early_after_the_gap(tmp_path, monkeypatch):
    store = _store()
    seen = []
    direction = store["wind_from_direction"]
    original = direction.__getitem__

    class Recording(_Array):
        def __getitem__(self, selection):
            seen.append(selection[0])
            return original(selection)

    store["wind_from_direction"] = Recording(direction.shape, direction._value, direction.attrs)
    monkeypatch.setattr(ukv, "open_store", lambda: (store, "s"))
    monkeypatch.setattr(ukv, "read_era5", _fake_era5)
    ukv.acquire(cycle=datetime(2025, 5, 16, 16), hours=0, cadence=None,
                area=Area(53.1, -0.6, 53.2, -0.5), out=tmp_path, progress=lambda text: None)
    index = int((datetime(2025, 5, 16, 16) - ukv.COVERAGE_START).total_seconds() // 3600)
    assert seen == [index - 1]


# --------------------------------------------------------------------------
# The profile and the doors
# --------------------------------------------------------------------------

def test_profile_documents_load_and_agree_with_the_row():
    from woof.mapped_composition import load_composition
    from woof.mapped_source import load_mapping
    from woof.source_adapters import get_source_adapter
    from woof.source_authorities import packaged_authorities
    adapter = get_source_adapter("ukv")
    assert adapter.runnable and adapter.packaged_profile == "ukv-sc-netcdf-v1"
    paths = packaged_authorities(adapter.packaged_profile)
    mapping = load_mapping(paths["mapping"])
    load_composition(paths["composition"], paths["mapping"])
    vertical = mapping["coordinates"]["vertical"]
    assert sorted(vertical["levels"]) == sorted(ukv.PRESSURE_LEVELS_PA)
    assert adapter.certified_source_top_pa == min(vertical["levels"])
    assert adapter.extendable_source_top_pa is None
    assert adapter.forcing_interval_seconds == mapping["target"]["boundary_interval_seconds"]
    assert adapter.forcing_interval_seconds == ukv.CADENCE_HOURS * 3600
    composition = json.loads(Path(paths["composition"]).read_text())
    assert composition["supplements"]["terrain_height"]["data_role"] == ukv.TERRAIN_ROLE
    # Every variable the mapping selects is one the fetch writes.
    written = set(ukv._UNITS)
    for name, field in mapping["fields"].items():
        for selector in field.get("selectors", []):
            assert set(selector["name"]) <= written, name


def test_fetch_doors_know_the_source(monkeypatch):
    from woof import fetch, fetch_routes
    assert fetch.native_cf_fetch_contract("ukv") == ukv.CONTRACT
    assert fetch.native_cf_fetch_contract("met-office-ukv") == ukv.CONTRACT
    assert fetch.fetch_accepts_area("ukv")
    assert "ukv" in fetch_routes.all_fetchable_sources()
    assert fetch_routes.publishes_prep_handoff("ukv")
    sentinel = datetime(2026, 10, 18, 9)
    monkeypatch.setattr(ukv, "latest_cycle", lambda hours, cadence=None: sentinel)
    assert fetch.resolve_latest_cycle("ukv", 3) == sentinel


def test_synthetic_file_decodes_through_the_packaged_mapping(tmp_path, monkeypatch):
    from woof.netcdf_bridge import resolve_netcdf_bin
    try:
        resolve_netcdf_bin()
    except Exception as error:                      # noqa: BLE001 - optional bridge
        pytest.skip(f"no rw_netcdf bridge here: {error}")
    from woof.mapped_source import inspect_mapped_source
    from woof.source_authorities import packaged_authorities
    monkeypatch.setattr(ukv, "open_store", lambda: (_store(), "s"))
    monkeypatch.setattr(ukv, "read_era5", _fake_era5)
    manifest = ukv.acquire(cycle=datetime(2024, 6, 18, 6), hours=1, cadence=None,
                           area=Area(53.1, -0.6, 53.2, -0.5), out=tmp_path,
                           progress=lambda text: None)
    mapping = packaged_authorities("ukv-sc-netcdf-v1")["mapping"]
    try:
        report = inspect_mapped_source(
            mapping, [tmp_path / entry["path"] for entry in manifest["files"]])
    except FileNotFoundError as error:
        pytest.skip(f"no mapped engine here: {error}")
    assert len(report["frames"]) == 2
    fields = report["frames"][0]["fields"]
    for name in ("air_temperature", "eastward_wind", "terrain_height", "soil_temperature",
                 "land_fraction", "surface_pressure", "air_temperature_2m"):
        assert fields[name]["missing"] == 0, name
    assert fields["eastward_wind"]["minimum"] == pytest.approx(10.0, abs=1e-3)


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("WOOF_NETWORK_TESTS") != "1",
                    reason="reads the Source Cooperative and ARCO archives")
def test_live_archive_window(tmp_path):
    manifest = ukv.acquire(cycle=datetime(2024, 6, 18, 6), hours=0, cadence=None,
                           area=Area(53.1, -0.6, 53.3, -0.4), out=tmp_path,
                           progress=lambda text: None)
    assert manifest["files"][0]["store_record"] == 4062
    assert 0.0 < manifest["terrain"]["max_m"] < 400.0


# --------------------------------------------------------------------------
# The land state
# --------------------------------------------------------------------------

def test_era5_layers_are_remapped_to_noah_by_layer_means():
    layers = [np.array([1.0]), np.array([2.0]), np.array([3.0]), np.array([4.0])]
    noah = ukv.layer_mean_remap(layers, ukv.ERA5_SOIL_LAYERS, ukv.NOAH_SOIL_LAYERS)
    expected = [(0.07 * 1 + 0.03 * 2) / 0.1, (0.18 * 2 + 0.12 * 3) / 0.3, 3.0, 4.0]
    assert np.allclose([float(layer[0]) for layer in noah], expected)
    # Noah's own layers pass through.
    same = ukv.layer_mean_remap(layers, ukv.NOAH_SOIL_LAYERS, ukv.NOAH_SOIL_LAYERS)
    assert [float(layer[0]) for layer in same] == [1.0, 2.0, 3.0, 4.0]
    with pytest.raises(ValueError, match="do not cover"):
        ukv.layer_mean_remap(layers[:2], ukv.NOAH_SOIL_LAYERS[:2], ukv.NOAH_SOIL_LAYERS)


def test_gfs_takes_over_past_era5s_last_hour():
    times = (datetime(2026, 10, 11, 12), datetime(2026, 10, 12, 0))
    assert _REAL_LAND_STATE_SOURCE(times, era5_through=datetime(2026, 10, 12)) == "era5"
    assert _REAL_LAND_STATE_SOURCE(times, era5_through=datetime(2026, 10, 11, 23)) == "gfs"
    assert ukv.gfs_cycle_for(datetime(2026, 10, 24, 15)) == datetime(2026, 10, 24, 12)
    assert ukv.gfs_lead(datetime(2026, 10, 24, 12), datetime(2026, 10, 25, 3)) == 15
    # Past f120 the posted three-hourly lead at or before the valid time.
    assert ukv.gfs_lead(datetime(2026, 10, 24, 12), datetime(2026, 10, 29, 14)) == 120


def test_gfs_land_state_reads_only_its_records(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from woof import mapped_source

    names = ("UGRD:10 m above ground", *ukv.GFS_RECORDS, "TMP:surface")
    idx, body, offset = [], b"", 0
    for number, name in enumerate(names, 1):
        idx.append(f"{number}:{offset}:d=2026101612:{name}:3 hour fcst:")
        payload = f"<{name}>".encode()
        body += payload
        offset += len(payload)
    requests = []

    def fake_get(url, headers=None):
        requests.append((url, (headers or {}).get("Range")))
        if url.endswith(".idx"):
            return "\n".join(idx).encode()
        start, end = headers["Range"][len("bytes="):].split("-")
        return body[int(start):(int(end) + 1 if end else None)]

    written = {}

    def fake_records(path, inventory, dump, wanted):
        written["bytes"] = Path(path).read_bytes()
        latitude = np.arange(-90.0, 90.01, 0.25)
        longitude = np.arange(-180.0, 180.0, 0.25)
        records = []
        for position, name in enumerate(ukv.GFS_RECORDS):
            value = np.full((latitude.size, longitude.size), float(position))
            if name.startswith(("TSOIL", "SOILW")):
                value[:, longitude > 1.0] = np.nan          # sea east of 1E
            records.append(SimpleNamespace(values=value, latitude=latitude, longitude=longitude))
        return tuple(records)

    monkeypatch.setattr(ukv, "_http_get", fake_get)
    monkeypatch.setattr(mapped_source, "_build_grib2_tools", lambda: ("inv", "dump"))
    monkeypatch.setattr(mapped_source, "_grib2_inventory",
                        lambda path, tool: [{"index": str(i)} for i in range(1, 11)])
    monkeypatch.setattr(mapped_source, "_grib2_records", fake_records)
    latitude, longitude = np.array([51.5, 51.52]), np.array([-0.12, -0.1])
    fields, receipt = ukv.read_gfs(datetime(2026, 10, 16, 15), datetime(2026, 10, 16, 12),
                                   latitude, longitude, tmp_path)
    # Exactly the ten land-state records, in the declared order, nothing else.
    assert written["bytes"] == b"".join(f"<{name}>".encode() for name in ukv.GFS_RECORDS)
    assert all("f003" in url for url, _ in requests)
    assert receipt["lead"] == 3 and receipt["source"] == "gfs"
    assert float(fields["land_area_fraction"][0, 0]) == 0.0
    assert float(fields["soil_moisture_layer_4"][0, 0]) == 9.0
    assert set(fields) == set(ukv._LAND_STATE_FIELDS)


# --------------------------------------------------------------------------
# The live bucket, on a small stand-in plane
# --------------------------------------------------------------------------

_SMALL = dict(x0=0.0, y0=-430000.0, nx=60, ny=50)


def _small_plane(monkeypatch):
    monkeypatch.setattr(ukv, "GRID_X0", _SMALL["x0"])
    monkeypatch.setattr(ukv, "GRID_Y0", _SMALL["y0"])
    monkeypatch.setattr(ukv, "GRID_NX", _SMALL["nx"])
    monkeypatch.setattr(ukv, "GRID_NY", _SMALL["ny"])


def _live_object(path, run, valid, product):
    from netCDF4 import Dataset
    variable, units, levelled = ukv._LIVE_OBJECTS[product]
    nx, ny = _SMALL["nx"], _SMALL["ny"]
    with Dataset(path, "w", format="NETCDF4") as dataset:
        dataset.createDimension("projection_x_coordinate", nx)
        dataset.createDimension("projection_y_coordinate", ny)
        x = dataset.createVariable("projection_x_coordinate", "f4", ("projection_x_coordinate",))
        x[:] = _SMALL["x0"] + 2000.0 * np.arange(nx)
        y = dataset.createVariable("projection_y_coordinate", "f4", ("projection_y_coordinate",))
        y[:] = _SMALL["y0"] + 2000.0 * np.arange(ny)
        for name, when in (("time", valid), ("forecast_reference_time", run)):
            var = dataset.createVariable(name, "i8")
            var.units = "seconds since 1970-01-01 00:00:00"
            var.assignValue(int((when - datetime(1970, 1, 1)).total_seconds()))
        dims = ("projection_y_coordinate", "projection_x_coordinate")
        constant = {"air_temperature": 280.0, "geopotential_height": 1000.0,
                    "relative_humidity": 0.5, "wind_speed": 10.0, "wind_from_direction": 270.0,
                    "surface_air_pressure": 98000.0, "surface_temperature": 285.0,
                    "lwe_thickness_of_surface_snow_amount": 0.0, "surface_altitude": 42.0}[variable]
        if levelled:
            dataset.createDimension("pressure", 33)
            pressure = dataset.createVariable("pressure", "f4", ("pressure",))
            ladder = np.array(ukv.PRESSURE_LEVELS_PA[::-1], dtype=float)     # bottom first
            pressure[:] = ladder
            data = dataset.createVariable(variable, "f4", ("pressure", *dims), zlib=True)
            value = np.full((33, ny, nx), constant)
            if variable == "air_temperature":
                value = value + (ladder / 1000.0)[:, None, None]             # label each level
            data[:] = value
        else:
            data = dataset.createVariable(variable, "f4", dims, zlib=True)
            data[:] = np.full((ny, nx), constant)
        data.units = units


def _live_bucket(tmp_path, monkeypatch, run, times):
    store = {}
    for valid in times:
        for product in ukv._LIVE_OBJECTS:
            key = ukv.live_key(run, valid, product)
            path = tmp_path / "bucket" / Path(key).name
            path.parent.mkdir(parents=True, exist_ok=True)
            _live_object(path, run, valid, product)
            store[key] = path.read_bytes()
    monkeypatch.setattr(ukv, "_http_get",
                        lambda url, headers=None: store[url[len(ukv.LIVE_HOST) + 1:]])
    monkeypatch.setattr(ukv, "live_listing",
                        lambda probe, valid: set(store) if probe == run else set())
    return store


def test_live_window_is_read_from_one_run(tmp_path, monkeypatch):
    from netCDF4 import Dataset
    _small_plane(monkeypatch)
    run = datetime(2026, 10, 16, 3)
    times = (datetime(2026, 10, 16, 6), datetime(2026, 10, 16, 9))
    _live_bucket(tmp_path, monkeypatch, run, times)
    monkeypatch.setattr(ukv, "read_era5", _fake_era5)
    lat, lon = ukv.laea_inverse(_SMALL["x0"] + 60000.0, _SMALL["y0"] + 50000.0)
    area = Area(float(lat) - 0.1, float(lon) - 0.1, float(lat) + 0.1, float(lon) + 0.1)
    manifest = ukv.acquire(cycle=times[0], hours=3, cadence=3, area=area,
                           out=tmp_path / "out", progress=lambda text: None)
    assert manifest["store"] == "live" and manifest["run"] == run.isoformat()
    assert [entry["lead"] for entry in manifest["files"]] == [3, 6]
    assert manifest["terrain"]["from"] == "published orography"
    with Dataset(tmp_path / "out" / manifest["files"][0]["path"]) as dataset:
        temperature = dataset.variables["air_temperature"][0]
        # Levels are relabelled by value: the 1000 Pa slot holds 1000 Pa's temperature.
        assert np.allclose(temperature[0], 281.0) and np.allclose(temperature[-1], 380.0)
        assert np.allclose(dataset.variables["eastward_wind"][0], 10.0, atol=1e-4)
        assert np.allclose(dataset.variables["surface_altitude"][0], 42.0)
    assert not (tmp_path / "out" / ".ukv-staging").exists()


def test_live_run_selection_skips_a_run_still_posting(monkeypatch):
    plan = ukv.plan_window(datetime(2026, 10, 16, 6), 0, now=NOW)
    complete = datetime(2026, 10, 16, 5)
    monkeypatch.setattr(ukv, "live_missing",
                        lambda run, times: [] if run == complete else ["k"])
    assert ukv.select_live_run(plan) == complete
    monkeypatch.setattr(ukv, "live_missing", lambda run, times: ["uk-deterministic-2km/x.nc"])
    with pytest.raises(ValueError, match="Wait for the run to finish posting"):
        ukv.select_live_run(plan)


def test_live_object_with_another_layout_refuses(tmp_path, monkeypatch):
    _small_plane(monkeypatch)
    run = datetime(2026, 10, 16, 6)
    store = _live_bucket(tmp_path, monkeypatch, run, (run,))
    monkeypatch.setattr(ukv, "GRID_X0", _SMALL["x0"] + 2000.0)       # the plane moved
    lat, lon = ukv.laea_inverse(_SMALL["x0"] + 60000.0, _SMALL["y0"] + 50000.0)
    regrid = ukv.Regridder(np.array([float(lat)]), np.array([float(lon)]))
    with pytest.raises(ValueError, match="does not have the layout"):
        ukv.read_live(run, run, regrid, tmp_path / "staging")
    assert store


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("WOOF_NETWORK_TESTS") != "1",
                    reason="reads the Met Office bucket and the GFS mirror")
def test_live_bucket_window(tmp_path, monkeypatch):
    monkeypatch.setattr(ukv, "land_state_source", _REAL_LAND_STATE_SOURCE)
    manifest = ukv.acquire(cycle=datetime(2026, 10, 16, 6), hours=0, cadence=None,
                           area=Area(51.4, -0.3, 51.6, 0.1), out=tmp_path,
                           progress=lambda text: None)
    assert manifest["store"] == "live" and manifest["run"] == "2026-10-16T06:00:00"
    assert manifest["land_state_source"] == "gfs"
    assert manifest["terrain"]["from"] == "published orography"


def test_a_dateless_window_is_checked_against_the_live_ladders():
    ukv.check_window_shape(54)
    ukv.check_window_shape(120, 3)
    with pytest.raises(ValueError, match="--cadence 3"):
        ukv.check_window_shape(60)
    with pytest.raises(ValueError, match="multiple"):
        ukv.check_window_shape(7, 3)


def test_a_window_whose_run_has_not_started_refuses_by_name():
    future = datetime.utcnow().replace(minute=0, second=0, microsecond=0) + timedelta(days=3)
    plan = ukv.plan_window(future, 3)
    with pytest.raises(ValueError, match="has not run yet"):
        ukv.select_live_run(plan)


def test_gfs_cycle_falls_back_while_the_newest_has_not_posted(monkeypatch):
    from urllib.error import HTTPError
    asked = []

    def fake_urlopen(request, timeout=None):
        asked.append(request.full_url)
        if "/gfs.20261016/12/" in request.full_url:
            raise HTTPError(request.full_url, 404, "absent", None, None)
        return open(os.devnull, "rb")

    monkeypatch.setattr(ukv, "urlopen", fake_urlopen)
    times = (datetime(2026, 10, 16, 15), datetime(2026, 10, 16, 18))
    assert ukv.choose_gfs_cycle(datetime(2026, 10, 16, 15), times) == datetime(2026, 10, 16, 6)
    assert asked[-1].endswith("gfs.t06z.pgrb2.0p25.f012.idx")
