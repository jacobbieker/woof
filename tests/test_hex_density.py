"""``woof hex density``: woof-hex.density.v1 rasters from corridors, terrain
and forecast fields (woof/hex/density.py)."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from woof.hex import density as dens
from woof.hex.density import DensityRefusal

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "energy"
SITES = FIXTURES / "sites_wales.json"
ASSETS = FIXTURES / "assets_wales.geojson"

#: Small rasters keep every build here around a second.
SMALL = dict(max_raster_cells=400_000)


def _grid_of(lat: np.ndarray, lon: np.ndarray) -> dens.RasterGrid:
    dlat = float(lat[1] - lat[0])
    dlon = float(lon[1] - lon[0])
    return dens.RasterGrid(lat0=float(lat[0]) - dlat / 2,
                           lon0=float(lon[0]) - dlon / 2, dlat=dlat,
                           dlon=dlon, ny=lat.size, nx=lon.size)


def _value_at(raster: dens.DensityRaster, lat: float, lon: float) -> float:
    r = int(raster.grid.rows_of(np.array([lat]))[0])
    c = int(raster.grid.cols_of(np.array([lon]))[0])
    return float(raster.spacing_km[r, c])


@pytest.fixture(scope="module")
def corridor() -> dens.DensityRaster:
    source = dens.corridor_from_sites(SITES, fine_km=0.5, corridor_km=2.0)
    return dens.build_density([source], background_km=5.0, **SMALL)


# ---------------------------------------------------------------------------
# the gradient bound comes from where the gates live
# ---------------------------------------------------------------------------


def test_default_bound_is_the_sizing_table_smoothness_refusal():
    from woof.mpas_mesh import load_sizing

    policy = dens.gradient_policy()
    smooth = load_sizing().smoothness
    assert policy.bound_per_cell == pytest.approx(
        smooth.refuse_above_percent_per_cell / 100.0)
    assert policy.bound_per_cell == pytest.approx(0.0306)
    assert policy.grade_per_cell == policy.bound_per_cell
    assert policy.workaround is None
    assert policy.bound_status == smooth.status


def test_allow_rough_is_the_generator_ceiling_and_says_workaround():
    from woof.hex.mesh_spec_gates import MAX_GRADIENT_PER_CELL

    policy = dens.gradient_policy(allow_rough=True)
    assert policy.bound_per_cell == pytest.approx(MAX_GRADIENT_PER_CELL)
    assert policy.workaround.startswith("WORKAROUND")
    assert policy.as_dict()["allow_rough"] is True


def test_a_grade_steeper_than_the_bound_is_refused():
    with pytest.raises(DensityRefusal, match="steeper than"):
        dens.gradient_policy(grade_percent_per_cell=5.0)
    with pytest.raises(DensityRefusal, match="positive"):
        dens.gradient_policy(grade_percent_per_cell=0.0)
    gentle = dens.gradient_policy(grade_percent_per_cell=1.5)
    assert gentle.grade_per_cell == pytest.approx(0.015)


# ---------------------------------------------------------------------------
# the limiter
# ---------------------------------------------------------------------------


def test_limiter_guarantees_the_neighbour_bound_on_a_rough_field():
    rng = np.random.default_rng(7)
    lat = 51.0 + 0.01 * np.arange(120)
    lon = -4.0 + 0.015 * np.arange(140)
    grid = _grid_of(lat, lon)
    raw = rng.uniform(0.2, 20.0, size=(grid.ny, grid.nx))
    limited = raw.copy()
    g = 0.0306
    dens.limit_gradient(limited, grid, g)
    assert dens.max_neighbour_gradient(raw, grid) > 10 * g
    assert dens.max_neighbour_gradient(limited, grid) <= g * (1 + 1e-9)
    # A request is never coarsened, and the finest request survives.
    assert np.all(limited <= raw + 1e-12)
    assert limited.min() == pytest.approx(raw.min())
    # The rw_mpas_mesh probe reads within the bound as well.
    assert dens.probe_gradient_per_cell(limited, grid, 20.0) <= g * 1.0005


@pytest.mark.parametrize("lat0", [-60.0, 40.0, -10.0])
def test_limiter_holds_the_bound_in_either_hemisphere(lat0):
    # Long lateral ramps make the shortest path detour poleward, where dx is
    # narrower; the sweeps must run poleward first or the bound breaks.
    grid = dens.RasterGrid(lat0=lat0, lon0=-20.0, dlat=0.2, dlon=0.2,
                           ny=100, nx=600)
    h = np.full((grid.ny, grid.nx), 1000.0)
    h[50, 300] = 1.0
    dens.limit_gradient(h, grid, 0.05)
    assert dens.max_neighbour_gradient(h, grid) <= 0.05 * (1 + 1e-9)


def test_limiter_is_the_exact_envelope_for_one_point():
    lat = 52.0 + 0.002 * np.arange(81)
    lon = -3.0 + 0.003 * np.arange(81)
    grid = _grid_of(lat, lon)
    h = np.full((grid.ny, grid.nx), 50.0)
    h[40, 40] = 0.1
    g = 0.03
    dens.limit_gradient(h, grid, g)
    dx = grid.dx_km_rows[40]
    # Along the row and the column the ramp is exactly fine + g d.
    expect_row = 0.1 + g * dx * np.abs(np.arange(81) - 40)
    np.testing.assert_allclose(h[40], np.minimum(expect_row, 50.0), rtol=1e-12)
    expect_col = 0.1 + g * grid.dy_km * np.abs(np.arange(81) - 40)
    np.testing.assert_allclose(h[:, 40], expect_col, rtol=1e-12)
    # Everywhere: between the Euclidean cone and the L1 cone (the lower side
    # within the half-percent the row-to-row change of dx makes over 18 km).
    yy = (np.arange(81) - 40)[:, None] * grid.dy_km
    xx = (np.arange(81) - 40)[None, :] * grid.dx_km_rows[:, None]
    assert np.all(h >= 0.1 + 0.995 * g * np.hypot(xx, yy))
    assert np.all(h <= 0.1 + g * (np.abs(xx) + np.abs(yy)) + 1e-9)


# ---------------------------------------------------------------------------
# corridors
# ---------------------------------------------------------------------------


def test_corridor_is_fine_on_the_lines_and_background_far_away(corridor):
    from woof.energy.contracts import load_sites

    sites = load_sites(SITES)
    for site in sites.sites:
        assert _value_at(corridor, site.lat, site.lon) == pytest.approx(0.5)
    # The raster rim is the background: no jump where the generator takes
    # over, and the far field asks for nothing.
    h = corridor.spacing_km
    rim = np.concatenate((h[0], h[-1], h[:, 0], h[:, -1]))
    np.testing.assert_allclose(rim, 5.0)
    assert corridor.min_spacing_km == pytest.approx(0.5)
    assert corridor.max_spacing_km == pytest.approx(5.0)


def test_corridor_grades_geometrically_outward(corridor):
    # Due south of the southern line at -3.62: spacing grows linearly with
    # distance (geometric in cell count) at the enforced grade.
    g = corridor.policy.enforced_per_cell
    lat_line, lon = 51.6, -3.62
    readings = []
    for d_km in (5.0, 20.0, 60.0):
        lat = lat_line - d_km / dens.KM_PER_DEG
        readings.append(_value_at(corridor, lat, lon))
    assert readings[0] < readings[1] < readings[2] <= 5.0
    slope = (readings[1] - readings[0]) / 15.0
    assert slope == pytest.approx(g, rel=0.08)
    assert corridor.limiter["max_neighbour_gradient_percent_after"] <= \
        corridor.policy.grade_per_cell * 100.0 * (1 + 1e-9)
    assert corridor.limiter["estimated_cells_after"] > \
        corridor.limiter["estimated_cells_before"]
    assert corridor.limiter["raster_cells_lowered"] > 0


def test_assets_corridor_covers_lines_points_and_areas():
    source = dens.corridor_from_assets(ASSETS, fine_km=0.5, corridor_km=1.0)
    assert source.polylines and source.points and source.areas
    raster = dens.build_density([source], background_km=5.0, **SMALL)
    from woof.energy.contracts import load_assets

    for asset in load_assets(ASSETS).assets:
        geometry = asset.geometry
        if geometry["type"] == "Point":
            lon, lat = geometry["coordinates"]
            assert _value_at(raster, lat, lon) == pytest.approx(0.5)


def test_sources_combine_with_minimum_spacing():
    coarse = dens.corridor_from_sites(SITES, fine_km=1.0, corridor_km=2.0)
    lat = np.array([51.75])
    lon = np.array([-4.2])
    fields = dens.PointSource(lat=lat, lon=lon, spacing_km=np.array([0.4]),
                              support_km=1.0)
    raster = dens.build_density([coarse, fields], background_km=5.0, **SMALL)
    assert _value_at(raster, 51.75, -4.2) == pytest.approx(0.4)
    assert _value_at(raster, 51.6, -3.62) == pytest.approx(1.0)
    assert [s["kind"] for s in raster.sources] == ["sites-corridor", "fields"]


# ---------------------------------------------------------------------------
# the contract on disk
# ---------------------------------------------------------------------------


def test_written_raster_meets_the_contract_exactly(corridor, tmp_path):
    import netCDF4

    path = corridor.write(tmp_path / "density.nc")
    assert not (tmp_path / "density.nc.partial").exists()
    with netCDF4.Dataset(path) as ds:
        assert ds.getncattr("schema") == "woof-hex.density.v1"
        assert ds.getncattr("min_spacing_km") == pytest.approx(
            float(np.min(ds.variables["spacing_km"][:])))
        assert set(ds.dimensions) == {"lat", "lon"}
        var = ds.variables["spacing_km"]
        assert var.dtype == np.float64
        assert var.dimensions == ("lat", "lon")
        assert var.units == "km"
        for name, units in (("lat", "degrees_north"), ("lon", "degrees_east")):
            axis = ds.variables[name]
            assert axis.dimensions == (name,)
            assert axis.units == units
            assert axis.dtype == np.float64
            assert np.all(np.diff(axis[:]) > 0)
        sources = json.loads(ds.getncattr("sources"))
        limiter = json.loads(ds.getncattr("limiter"))
    assert sources[0]["kind"] == "sites-corridor"
    assert limiter["bound_percent_per_cell"] == pytest.approx(3.06)
    lat, lon, spacing, attrs = dens.read_density_raster(path)
    np.testing.assert_array_equal(spacing, corridor.spacing_km)
    np.testing.assert_allclose(lat, corridor.lat)
    assert attrs["background_km"] == pytest.approx(5.0)


def test_reader_refuses_an_off_contract_file(tmp_path):
    import netCDF4

    path = tmp_path / "bad.nc"
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("lat", 2)
        ds.createDimension("lon", 2)
        ds.createVariable("lat", "f8", ("lat",))[:] = [1.0, 2.0]
        ds.createVariable("lon", "f8", ("lon",))[:] = [1.0, 2.0]
        ds.createVariable("spacing_km", "f4", ("lat", "lon"))[:] = 1.0
        ds.schema = "woof-hex.density.v1"
        ds.min_spacing_km = 1.0
    with pytest.raises(DensityRefusal, match="float64"):
        dens.read_density_raster(path)


def test_summary_reports_extent_cells_and_limiter_effect(corridor):
    summary = corridor.summary("x.nc")
    assert summary["schema"] == dens.SUMMARY_SCHEMA
    assert set(summary["extent"]) == {"lat_min", "lat_max", "lon_min",
                                      "lon_max"}
    assert summary["estimated_cells"] == pytest.approx(
        dens.estimate_cells(corridor.spacing_km, corridor.grid))
    assert summary["limiter"]["cell_ratio_after_over_before"] > 1.0
    assert summary["resolution"]["target_cells_per_finest"] == pytest.approx(4)
    json.dumps(summary, default=dens._json_default)


def test_estimate_cells_matches_a_uniform_field():
    lat = 10.0 + 0.01 * np.arange(100)
    lon = 20.0 + 0.01 * np.arange(100)
    grid = _grid_of(lat, lon)
    h = np.full((grid.ny, grid.nx), 2.0)
    area = (np.radians(1.0) * dens.EARTH_RADIUS_KM) ** 2 * np.cos(
        np.radians(10.5))
    expect = area / (dens.HEX_AREA_FACTOR * 4.0)
    assert dens.estimate_cells(h, grid) == pytest.approx(expect, rel=1e-3)


# ---------------------------------------------------------------------------
# resolution choice and refusals
# ---------------------------------------------------------------------------


def test_resolution_meets_the_target_when_it_fits():
    lat = np.array([51.7])
    lon = np.array([-3.5])
    point = dens.PointSource(lat=lat, lon=lon, spacing_km=np.array([1.0]),
                             support_km=1.0)
    raster = dens.build_density(
        [point], background_km=2.0,
        policy=dens.gradient_policy(allow_rough=True), max_raster_cells=10**6)
    assert raster.plan.coarsened_for_size is False
    assert raster.plan.as_dict()["cells_per_finest"] == pytest.approx(4.0)


def test_too_large_a_raster_is_refused_with_numbers():
    source = dens.corridor_from_sites(SITES, fine_km=0.1, corridor_km=2.0)
    with pytest.raises(DensityRefusal, match="max-raster-cells"):
        dens.build_density([source], background_km=25.0,
                           max_raster_cells=100_000, interp_tolerance=0.01)


def test_empty_sites_are_refused(tmp_path):
    document = json.loads(SITES.read_text())
    document["columns"] = {name: [] for name in document["columns"]}
    document["count"] = 0
    empty = tmp_path / "sites.json"
    empty.write_text(json.dumps(document))
    with pytest.raises(DensityRefusal, match="no sites"):
        dens.corridor_from_sites(empty, fine_km=0.1)


def test_fine_not_finer_than_background_is_refused():
    with pytest.raises(DensityRefusal, match="finer than"):
        dens._check_spacings(5.0, 5.0)


def test_a_raster_across_the_antimeridian_is_refused():
    point = dens.PointSource(lat=np.array([10.0]), lon=np.array([179.9]),
                             spacing_km=np.array([1.0]), support_km=1.0)
    with pytest.raises(DensityRefusal, match="antimeridian"):
        dens.build_density([point], background_km=5.0, **SMALL)


def test_a_polar_raster_is_refused():
    point = dens.PointSource(lat=np.array([84.0]), lon=np.array([0.0]),
                             spacing_km=np.array([1.0]), support_km=1.0)
    with pytest.raises(DensityRefusal, match="latitude"):
        dens.build_density([point], background_km=5.0, **SMALL)


# ---------------------------------------------------------------------------
# terrain (synthetic DEM, no network)
# ---------------------------------------------------------------------------


def _hill_dem():
    lat = 51.70 + 0.0009 * np.arange(160)
    lon = -3.60 + 0.0014 * np.arange(160)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    y = (la - lat.mean()) * dens.KM_PER_DEG * 1000.0
    x = (lo - lon.mean()) * dens.KM_PER_DEG * 1000.0 * np.cos(
        np.radians(lat.mean()))
    # A 900 m ridge, 1 km half-width, flat plain around it.
    z = 900.0 * np.exp(-((x / 1000.0) ** 2) - (y / 4000.0) ** 2)
    return lat, lon, z, x, y


def test_terrain_spacing_is_finer_on_steep_flanks_than_on_flat_ground():
    lat, lon, z, x, y = _hill_dem()
    spacing, diag = dens.terrain_spacing(lat, lon, z, fine_km=0.2,
                                         background_km=5.0)
    flank = (np.abs(np.abs(x) - 707.0) < 150.0) & (np.abs(y) < 500.0)
    plain = np.hypot(x / 1000.0, y / 4000.0) > 3.0
    assert np.nanmax(spacing[flank]) < 1.0
    assert np.nanmin(spacing[plain]) == pytest.approx(5.0)
    assert diag["slope_deg_max"] > 20.0
    assert diag["thresholds"]["status"].startswith("heuristic")


def test_terrain_source_builds_a_limited_raster():
    lat, lon, z, _, _ = _hill_dem()
    source = dens.terrain_from_dem(lat, lon, z, fine_km=0.2, background_km=5.0)
    raster = dens.build_density([source], background_km=5.0, **SMALL)
    assert raster.min_spacing_km < 0.5
    assert raster.limiter["max_neighbour_gradient_percent_after"] <= 3.06
    assert raster.sources[0]["kind"] == "terrain-gradient"


def test_terrain_holes_make_no_artificial_cliffs():
    lat = 51.0 + 0.001 * np.arange(60)
    lon = -3.0 + 0.0015 * np.arange(60)
    z = np.full((60, 60), 800.0)  # a high flat plateau
    z[20:40, 20:40] = np.nan  # a void in it
    spacing, diag = dens.terrain_spacing(lat, lon, z, fine_km=0.2,
                                         background_km=5.0)
    assert np.all(np.isnan(spacing[20:40, 20:40]))
    np.testing.assert_allclose(spacing[~np.isnan(spacing)], 5.0)
    assert diag["no_data_cells"] == 400


def test_flat_terrain_asks_for_nothing_and_is_refused():
    lat, lon, z, _, _ = _hill_dem()
    with pytest.raises(DensityRefusal, match="no refinement"):
        dens.terrain_from_dem(lat, lon, np.zeros_like(z), fine_km=0.2,
                              background_km=5.0)


def test_terrain_thresholds_out_of_order_are_refused():
    lat, lon, z, _, _ = _hill_dem()
    with pytest.raises(DensityRefusal, match="slope-fine-deg"):
        dens.terrain_spacing(lat, lon, z, fine_km=0.2, background_km=5.0,
                             slope_coarse_deg=30.0, slope_fine_deg=10.0)


def test_terrain_offline_with_an_empty_cache_refuses(tmp_path):
    with pytest.raises(DensityRefusal, match="--offline"):
        dens.terrain_from_glo30(-3.55, 51.60, -3.50, 51.62, fine_km=0.25,
                                background_km=5.0, cache_root=tmp_path,
                                offline=True)


@pytest.mark.network
def test_tiny_glo30_fetch(tmp_path):
    if os.environ.get("WOOF_NETWORK_TESTS") != "1":
        pytest.skip("live GLO-30 fetch needs WOOF_NETWORK_TESTS=1")
    lat, lon, z, provenance = dens.fetch_glo30_elevation(
        -3.50, 51.65, -3.45, 51.68, analysis_km=0.1, cache_root=tmp_path)
    assert np.all(np.diff(lat) > 0) and np.all(np.diff(lon) > 0)
    assert np.isfinite(z).all()
    assert 0.0 < float(np.nanmax(z)) < 1000.0  # South Wales valleys
    assert provenance["tiles"] == ["Copernicus_DSM_COG_10_N51_00_W004_00_DEM.tif"]
    assert provenance["tiles_fetched_bytes"] > 0


# ---------------------------------------------------------------------------
# forecast fields (the adaptive driver's API)
# ---------------------------------------------------------------------------


def _field_points():
    lat = 52.0 + 0.02 * np.arange(40)
    lon = -3.0 + 0.03 * np.arange(40)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    wind = np.where((np.abs(la - 52.4) < 0.05) & (np.abs(lo + 2.4) < 0.08),
                    30.0, 5.0)
    return la, lo, wind


def test_fields_refine_where_the_criterion_is_met():
    la, lo, wind = _field_points()
    raster = dens.density_from_fields(
        la, lo, {"wind_ms": wind}, {"wind_ms": (10.0, 25.0)}, fine_km=0.5,
        background_km=5.0, **SMALL)
    assert _value_at(raster, 52.4, -2.4) == pytest.approx(0.5)
    assert _value_at(raster, 52.0, -3.0) > 0.5
    assert raster.sources[0]["criteria"]["wind_ms"]["points_at_fine"] > 0
    assert raster.limiter["max_neighbour_gradient_percent_after"] <= 3.06


def test_fields_lower_is_finer_nan_and_per_criterion_fine():
    la, lo, wind = _field_points()
    visibility = np.where(wind > 10.0, 200.0, 20000.0)
    visibility[0, 0] = np.nan
    spacing, report = dens.criteria_spacing(
        {"wind_ms": wind, "visibility_m": visibility},
        {"wind_ms": (10.0, 25.0),
         "visibility_m": dens.Criterion(5000.0, 500.0, fine_km=0.3)},
        fine_km=0.5, background_km=5.0)
    spacing = spacing.reshape(la.shape)
    assert spacing[np.unravel_index(np.argmax(wind), wind.shape)] == \
        pytest.approx(0.3)
    assert spacing[0, 0] == pytest.approx(5.0)
    assert report["visibility_m"]["fine_km"] == 0.3


def test_point_source_extent_follows_only_the_asking_points():
    lat = np.array([51.0, 40.0, 60.0])
    lon = np.array([-3.0, 10.0, -20.0])
    point = dens.PointSource(lat=lat, lon=lon,
                             spacing_km=np.array([1.0, 5.0, np.nan]),
                             support_km=2.0, background_km=5.0)
    south, north, west, east = point.bounds()
    assert 50.9 < south < north < 51.1 and -3.1 < west < east < -2.9
    assert point.finest_km() == 1.0


def test_fields_that_ask_for_nothing_are_refused():
    la, lo, wind = _field_points()
    with pytest.raises(DensityRefusal, match="no refinement"):
        dens.density_from_fields(la, lo, {"wind_ms": wind},
                                 {"wind_ms": (50.0, 60.0)}, fine_km=0.5,
                                 background_km=5.0)


def test_fields_with_missing_thresholds_or_sizes_are_refused():
    la, lo, wind = _field_points()
    with pytest.raises(DensityRefusal, match="no thresholds"):
        dens.density_from_fields(la, lo, {"wind_ms": wind}, {},
                                 fine_km=0.5, background_km=5.0)
    with pytest.raises(DensityRefusal, match="values for"):
        dens.density_from_fields(la, lo, {"wind_ms": wind[:3]},
                                 {"wind_ms": (10.0, 25.0)}, fine_km=0.5,
                                 background_km=5.0)


# ---------------------------------------------------------------------------
# the door
# ---------------------------------------------------------------------------


def test_cli_writes_the_raster_and_prints_the_summary(tmp_path, capsys):
    from woof.hex.cli import main

    out = tmp_path / "density.nc"
    code = main(["density", "--sites", str(SITES), "--fine-km", "0.5",
                 "--background-km", "5", "--max-raster-cells", "400000",
                 "-o", str(out)])
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["path"] == str(out)
    assert summary["min_spacing_km"] == pytest.approx(0.5)
    assert summary["estimated_cells"] > 0
    assert "total" in summary["timings_s"]
    dens.read_density_raster(out)


@pytest.mark.parametrize("argv, message", [
    (["--fine-km", "0.5", "--background-km", "5"], "at least one source"),
    (["--sites", str(SITES), "--fine-km", "5", "--background-km", "5"],
     "finer than"),
    (["--sites", str(SITES), "--bbox=-3.6,51.7,-3.3,51.9", "--fine-km", "0.5",
      "--background-km", "5"], "--terrain-gradient"),
    (["--terrain-gradient", "--fine-km", "0.5", "--background-km", "5"],
     "--bbox"),
    (["--sites", str(SITES), "--fine-km", "0.5", "--background-km", "5",
      "--grade-percent-per-cell", "4"], "steeper than"),
    (["--sites", str(SITES), "--fine-km", "0.5", "--background-km", "5",
      "--slope-fine-deg", "30", "--offline"], "--offline, --slope-fine-deg"),
])
def test_cli_refusals_exit_2(tmp_path, capsys, argv, message):
    from woof.hex.cli import main

    code = main(["density", *argv, "-o", str(tmp_path / "x.nc")])
    assert code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "x.nc").exists()


def test_cli_parser_needs_fine_background_and_output():
    from woof.hex.cli import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["density", "--sites", str(SITES)])
    args = parser.parse_args(["density", "--sites", str(SITES), "--fine-km",
                              "0.1", "--background-km", "25", "-o", "d.nc"])
    assert args.corridor_km == 2.0
    assert args.cells_per_finest == dens.DEFAULT_CELLS_PER_FINEST
    assert args.max_raster_cells == dens.DEFAULT_MAX_RASTER_CELLS
    assert args.interp_tolerance == dens.DEFAULT_INTERP_TOLERANCE
    assert args.slope_fine_deg == dens.DEFAULT_SLOPE_FINE_DEG
    assert args.curvature_coarse == dens.DEFAULT_CURVATURE_COARSE_PER_M
    assert math.isclose(args.corridor_km, dens.DEFAULT_CORRIDOR_KM)
