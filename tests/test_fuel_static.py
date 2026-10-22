"""Fire fuel statics: NFUEL_CAT on both meshes, fire-mesh slope, no drift."""
from dataclasses import fields as dataclass_fields
from types import SimpleNamespace

import numpy as np
import pytest

from woof.static import build as static_build
from woof.static.build import GeogSelection
from woof.static.fuel import (
    FUEL_FIELDS, NO_FUEL, FuelStaticsRequest, build_fuel_fields,
    MISSING_CODE, crosswalk_table, dominant_fuel, fire_mesh_axis,
    fire_mesh_slope, fire_mesh_spacing,
    fuel_request_for, fuel_summary, landfire_codes_to_fuel,
    landuse_fuel_lookup, nfuel_from_lu_index, refine_nearest, refine_zsf,
    refresh_fuel_fields, with_fuel_statics)

MODIS = {"MMINLU": "MODIFIED_IGBP_MODIS_NOAH", "ISWATER": 17, "ISLAKE": 21,
         "ISICE": 15, "ISURBAN": 13, "NUM_LAND_CAT": 21}
USGS = {"MMINLU": "USGS", "ISWATER": 16, "ISLAKE": -1, "ISICE": 24,
        "ISURBAN": 1, "NUM_LAND_CAT": 24}
NLCD40 = {"MMINLU": "NLCD40", "ISWATER": 17, "ISLAKE": -1, "ISICE": 15,
          "ISURBAN": 13, "NUM_LAND_CAT": 40}


@pytest.fixture(autouse=True)
def _host_only(monkeypatch):
    """These are the CPU tests: no device is opened, whatever the host has.
    tests/test_fuel_static_gpu.py holds the GPU body to this one."""
    from woof.static import fuel
    monkeypatch.setattr(fuel, "gpu_available", lambda: False)


def _grid(nx=6, ny=5, dx=1000.0):
    from woof.static.lambert import LambertGrid
    return LambertGrid(ref_lat=40.0, ref_lon=-105.0, truelat1=30.0,
                       truelat2=60.0, stand_lon=-105.0, dx=dx, dy=dx,
                       e_we=nx + 1, e_sn=ny + 1)


# --- the request -------------------------------------------------------------

def test_no_shipped_configuration_requests_fuel(tmp_path):
    selection = GeogSelection.fallback(tmp_path)
    assert selection.fuel is None
    for cfg in (None, SimpleNamespace(), SimpleNamespace(ifire=0, sr_x=4)):
        assert with_fuel_statics(selection, cfg) is selection


def test_fire_namelist_names_request_the_mesh(tmp_path):
    selection = GeogSelection.fallback(tmp_path)
    result = with_fuel_statics(selection, SimpleNamespace(ifire=2, sr_x=4,
                                                          sr_y=3))
    assert result.fuel == FuelStaticsRequest(sr_x=4, sr_y=3)
    assert result.fuel.landfire is None
    direct = with_fuel_statics(selection, FuelStaticsRequest(sr_x=2, sr_y=2))
    assert direct.fuel.sr_x == 2


def test_a_fire_run_without_a_mesh_is_refused(tmp_path):
    with pytest.raises(ValueError, match="sr_x must be 1 or more"):
        fuel_request_for(SimpleNamespace(ifire=1))
    for bad in (0, -1, 2.5, True, "4"):
        with pytest.raises(ValueError, match="sr_x"):
            FuelStaticsRequest(sr_x=bad, sr_y=1)
    with pytest.raises(ValueError, match="names no LANDFIRE raster"):
        FuelStaticsRequest(sr_x=1, sr_y=1, landfire_kind="fbfm40")
    with pytest.raises(ValueError, match="device must be one of"):
        FuelStaticsRequest(sr_x=1, sr_y=1, device="tpu")
    # Both devices give the same words, so the device is not identity.
    assert (FuelStaticsRequest(sr_x=2, sr_y=2, device="gpu")
            == FuelStaticsRequest(sr_x=2, sr_y=2, device="cpu"))
    assert (hash(FuelStaticsRequest(sr_x=2, sr_y=2, device="gpu"))
            == hash(FuelStaticsRequest(sr_x=2, sr_y=2)))
    with pytest.raises(ValueError, match="landfire_kind must be one of"):
        FuelStaticsRequest(sr_x=1, sr_y=1, landfire=tmp_path / "f.tif",
                           landfire_kind="fbfm99")


def test_selection_identity_is_unchanged_without_a_request(tmp_path):
    """The memo keys carry the selection; no request, the same identity."""
    a = GeogSelection.fallback(tmp_path)
    b = GeogSelection.fallback(tmp_path)
    assert a == b and hash(a) == hash(b)
    # appended last, so positional construction is unchanged
    assert [f.name for f in dataclass_fields(GeogSelection)][-1] == "fuel"
    with_fuel = with_fuel_statics(a, FuelStaticsRequest(sr_x=2, sr_y=2))
    assert with_fuel != a
    # The nest and child routes overlay [static.highres] with no fuel
    # refresh, so the drag/lake chain they share does not request fuel.
    from woof.static.orographic import with_terrain_drag_statics
    for cfg in (SimpleNamespace(topo_wind=0, gwd_opt=0, sf_lake_physics=0),
                SimpleNamespace(topo_wind=0, gwd_opt=0, ifire=1, sr_x=3,
                                sr_y=3)):
        assert with_terrain_drag_statics(a, cfg) is a


def _fake_static(ny, nx):
    lu = np.full((ny, nx), 10.0)
    lu[0, :] = 17.0
    lu[1, 1] = 1.0
    return {"HGT_M": np.arange(ny * nx, dtype=np.float64).reshape(ny, nx),
            "LU_INDEX": lu, "LANDMASK": (lu != 17.0).astype(np.float64)}


def test_static_build_without_request_is_the_same_field_set(
        tmp_path, monkeypatch):
    grid = _grid()
    built = _fake_static(5, 6)
    monkeypatch.setattr(static_build, "_build_static_routed",
                        lambda *a, **k: dict(built))
    monkeypatch.setattr(GeogSelection, "landuse_global_attrs",
                        lambda self: dict(MODIS))
    plain = static_build.build_static(
        grid, tmp_path, selection=GeogSelection.fallback(tmp_path))
    assert sorted(plain) == sorted(built)
    for name in built:
        assert plain[name] is built[name]
    report = {}
    fired = static_build.build_static(
        grid, tmp_path, source_coverage_report=report,
        selection=with_fuel_statics(GeogSelection.fallback(tmp_path),
                                    FuelStaticsRequest(sr_x=2, sr_y=3)))
    assert set(fired) == set(built) | set(FUEL_FIELDS)
    assert fired["NFUEL_CAT_FIRE"].shape == (15, 12)
    assert report["fuel"]["source"] == "landuse"
    assert report["fuel"]["crosswalk_sha256"] == crosswalk_table()["sha256"]


# --- the crosswalk -----------------------------------------------------------

@pytest.mark.parametrize("attrs", [MODIS, USGS, NLCD40])
def test_water_urban_ice_and_barren_have_no_fuel(attrs):
    lookup = landuse_fuel_lookup(attrs["MMINLU"])
    for key in ("ISWATER", "ISLAKE", "ISICE", "ISURBAN"):
        if attrs[key] > 0:
            assert lookup[attrs[key]] == NO_FUEL, key
    names = crosswalk_table()["landuse"][attrs["MMINLU"]]["categories"]
    for code, row in names.items():
        name = row["name"].lower()
        if any(word in name for word in ("water", "lake", "urban", "ice",
                                         "barren", "developed", "lcz")):
            assert row["fuel"] == NO_FUEL, (code, row)
    # every category the legend declares has a row
    assert all(lookup[1:attrs["NUM_LAND_CAT"] + 1] > 0)


def test_lu_index_maps_through_the_legend():
    lu = np.array([[1, 7, 10], [13, 17, 21]], dtype=np.float64)
    np.testing.assert_array_equal(
        nfuel_from_lu_index(lu, MODIS), [[10, 5, 1], [14, 14, 14]])
    np.testing.assert_array_equal(
        nfuel_from_lu_index(np.array([[16.0, 14.0, 2.0]]), USGS),
        [[14, 10, 1]])
    alias = dict(MODIS, MMINLU="modis")
    np.testing.assert_array_equal(nfuel_from_lu_index(lu, alias),
                                  nfuel_from_lu_index(lu, MODIS))


def test_crosswalk_refuses_what_it_cannot_vouch_for():
    with pytest.raises(ValueError, match="no land-use-to-fuel crosswalk"):
        nfuel_from_lu_index(np.ones((2, 2)), dict(MODIS, MMINLU="SSIB"))
    with pytest.raises(ValueError, match=r"categories \[30\]"):
        nfuel_from_lu_index(np.array([[30.0]]), MODIS)
    with pytest.raises(ValueError, match="ISWATER=5 maps to 10"):
        nfuel_from_lu_index(np.ones((2, 2)), dict(MODIS, ISWATER=5))
    with pytest.raises(ValueError, match="whole land-use categories"):
        nfuel_from_lu_index(np.array([[1.5]]), MODIS)


# --- LANDFIRE codes ----------------------------------------------------------

def test_fbfm40_codes_take_the_crosswalk():
    codes = np.array([101, 103, 122, 145, 165, 181, 188, 203, 91, 98, np.nan,
                      -9999])
    fuel, valid, kind = landfire_codes_to_fuel(codes)
    assert kind == "fbfm40"
    np.testing.assert_array_equal(valid, [True] * 10 + [False, False])
    np.testing.assert_array_equal(fuel[:10],
                                  [1, 3, 2, 4, 10, 8, 9, 13, 14, 14])


def test_fbfm13_codes_are_themselves():
    codes = np.array([1.0, 4.0, 13.0, 93.0, 99.0])
    fuel, valid, kind = landfire_codes_to_fuel(codes)
    assert kind == "fbfm13" and valid.all()
    np.testing.assert_array_equal(fuel, [1, 4, 13, 14, 14])
    # every crosswalk row lands in 1..14
    for row in crosswalk_table()["landfire"]["fbfm40"].values():
        assert 1 <= row["fuel"] <= NO_FUEL


def test_landfire_codes_that_are_not_fuel_models_are_refused():
    with pytest.raises(ValueError, match="mixes FBFM13"):
        landfire_codes_to_fuel(np.array([1.0, 101.0]))
    with pytest.raises(ValueError, match=r"codes \[50\]"):
        landfire_codes_to_fuel(np.array([50.0]))
    with pytest.raises(ValueError, match="holds FBFM40"):
        landfire_codes_to_fuel(np.array([101.0]), "fbfm13")
    with pytest.raises(ValueError, match="holds FBFM13"):
        landfire_codes_to_fuel(np.array([1.0]), "fbfm40")


# --- the fire mesh -----------------------------------------------------------

def test_fire_sub_cells_tile_their_parent_cell():
    np.testing.assert_allclose(fire_mesh_axis(2, 4),
                               [0.625, 0.875, 1.125, 1.375,
                                1.625, 1.875, 2.125, 2.375])
    np.testing.assert_allclose(fire_mesh_axis(3, 1), [1.0, 2.0, 3.0])
    parent = np.array([[1, 2], [3, 4]])
    refined = refine_nearest(parent, 3, 2)
    assert refined.shape == (4, 6) and refined.dtype == np.uint8
    np.testing.assert_array_equal(refined[:2, :3], 1)
    np.testing.assert_array_equal(refined[2:, 3:], 4)


def test_refinement_keeps_a_plane_and_its_slope():
    """WRF's refinement is bilinear with a linear continuation past the
    edge, so a plane stays a plane (to REAL(4)), edge band included.  The
    words themselves are held to WRF by tests/test_fire_mesh_wrf471_parity.py.
    """
    ny, nx, dx, dy = 5, 7, 1000.0, 500.0
    j, i = np.mgrid[0:ny, 0:nx].astype(np.float64)
    hgt = 100.0 + 3.0 * i + 5.0 * j
    zsf = refine_zsf(hgt, 4, 2)
    assert zsf.shape == (ny * 2, nx * 4) and zsf.dtype == np.float32
    yf, xf = np.meshgrid(fire_mesh_axis(ny, 2) - 1.0,
                         fire_mesh_axis(nx, 4) - 1.0, indexing="ij")
    np.testing.assert_allclose(zsf, 100.0 + 3.0 * xf + 5.0 * yf,
                               rtol=0, atol=2e-4)
    dzdx, dzdy = fire_mesh_slope(zsf, fire_mesh_spacing(dx, 4),
                                 fire_mesh_spacing(dy, 2))
    assert dzdx.dtype == dzdy.dtype == np.float32
    np.testing.assert_allclose(dzdx, 3.0 / dx, rtol=2e-4)
    np.testing.assert_allclose(dzdy, 5.0 / dy, rtol=2e-4)
    # sr = 1 is the atmospheric grid itself, in REAL(4)
    np.testing.assert_array_equal(refine_zsf(hgt, 1, 1),
                                  hgt.astype(np.float32))
    assert fire_mesh_spacing(333.3, 4) == np.float32(333.3) / np.float32(4)


def test_a_one_cell_axis_is_refused():
    with pytest.raises(ValueError, match="at least 2 x 2"):
        refine_zsf(np.array([[7.0, 9.0]]), 2, 3)
    with pytest.raises(ValueError, match="at least 2 x 2"):
        fire_mesh_slope(np.ones((1, 4), np.float32), 10.0, 10.0)


def test_dominant_category_breaks_ties_low():
    fire = np.array([[3, 3, 14, 14],
                     [9, 9, 14, 2]], dtype=np.float64)
    np.testing.assert_array_equal(dominant_fuel(fire, 2, 2), [[3, 14]])
    fire = np.array([[5, 1], [5, 1]], dtype=np.float64)
    np.testing.assert_array_equal(dominant_fuel(fire, 2, 2), [[1]])


def test_build_on_a_grid_gives_both_meshes_and_a_receipt():
    grid = _grid(nx=6, ny=5)
    built = _fake_static(5, 6)
    report = {}
    out = build_fuel_fields(grid, built, FuelStaticsRequest(sr_x=4, sr_y=4),
                            landuse_attrs=MODIS, coverage_report=report)
    assert out["NFUEL_CAT"].shape == (5, 6)
    assert out["NFUEL_CAT_FIRE"].shape == (20, 24)
    assert out["NFUEL_CAT"].dtype == np.float64
    assert out["NFUEL_CAT_FIRE"].dtype == np.uint8
    assert all(out[name].dtype == np.float32
               for name in ("ZSF", "DZDXF", "DZDYF"))
    assert report["fuel"]["device"] == "cpu"
    np.testing.assert_array_equal(out["NFUEL_CAT"][0], NO_FUEL)
    assert out["NFUEL_CAT"][1, 1] == 10.0  # evergreen needleleaf
    np.testing.assert_array_equal(
        dominant_fuel(out["NFUEL_CAT_FIRE"], 4, 4), out["NFUEL_CAT"])
    # HGT_M rises 1 m per cell in x and nx m per cell in y
    np.testing.assert_allclose(out["DZDXF"], 1.0 / 1000.0, rtol=1e-5)
    np.testing.assert_allclose(out["DZDYF"], 6.0 / 1000.0, rtol=1e-5)
    assert report["fuel"]["fire_mesh_shape"] == [20, 24]
    summary = fuel_summary(out)
    assert sum(row["cells"] for row in summary["fire_mesh"]) == 480
    assert summary["slope"]["magnitude_max"] == pytest.approx(
        np.hypot(1.0, 6.0) / 1000.0, rel=1e-5)
    with pytest.raises(ValueError, match="device='gpu'"):
        build_fuel_fields(grid, built, FuelStaticsRequest(sr_x=1, sr_y=1),
                          landuse_attrs=MODIS, device="gpu")
    with pytest.raises(ValueError, match="duplicate"):
        build_fuel_fields(grid, built, FuelStaticsRequest(sr_x=1, sr_y=1),
                          landuse_attrs=MODIS, coverage_report=report)
    with pytest.raises(ValueError, match="mass points"):
        build_fuel_fields(_grid(nx=7, ny=5), built,
                          FuelStaticsRequest(sr_x=1, sr_y=1),
                          landuse_attrs=MODIS)


def test_refresh_is_a_no_op_without_a_request(tmp_path):
    fields = _fake_static(5, 6)
    selection = GeogSelection.fallback(tmp_path)
    assert refresh_fuel_fields(fields, _grid(), selection,
                               landuse_attrs=MODIS) is fields
    fired = with_fuel_statics(selection, FuelStaticsRequest(sr_x=2, sr_y=2))
    out = refresh_fuel_fields(fields, _grid(), fired, landuse_attrs=MODIS)
    assert set(out) == set(fields) | set(FUEL_FIELDS)
    assert set(fields) == {"HGT_M", "LU_INDEX", "LANDMASK"}


# --- the LANDFIRE raster -----------------------------------------------------

def test_landfire_raster_is_sampled_nearest_at_fire_cell_centres(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    pytest.importorskip("pyproj")
    from rasterio.transform import from_origin

    grid = _grid(nx=4, ny=3, dx=3000.0)
    sr = 2
    # A 0.001-degree geographic raster whose code encodes its column, over
    # the domain plus a margin.  Every fire cell must read the pixel its
    # centre falls in.
    lat, lon = grid.ij_to_latlon(*np.meshgrid(fire_mesh_axis(4, sr),
                                              fire_mesh_axis(3, sr)))
    west, north = lon.min() - 0.05, lat.max() + 0.05
    res = 0.001
    width = int((lon.max() + 0.05 - west) / res)
    height = int((north - (lat.min() - 0.05)) / res)
    codes = np.array([101, 145, 181, 202, 98], dtype=np.int16)
    data = codes[np.arange(width) % codes.size][None, :].repeat(height, 0)
    data[: height // 2, :] = np.where(
        np.arange(width) % 7 == 0, -9999, data[: height // 2, :])
    path = tmp_path / "fbfm40.tif"
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height,
                       count=1, dtype="int16", crs="EPSG:4326",
                       nodata=-9999,
                       transform=from_origin(west, north, res, res)) as out:
        out.write(data, 1)
    with rasterio.open(path) as dataset:
        inverse = ~dataset.transform
    col = np.floor(inverse.a * lon + inverse.c).astype(int)
    row = np.floor(inverse.e * lat + inverse.f).astype(int)
    expected_raw = data[row, col].astype(np.int32)
    expected_raw[expected_raw == -9999] = MISSING_CODE

    from woof.static.fuel import sample_landfire
    sampled = sample_landfire(path, grid, sr, sr)
    assert sampled.dtype == np.int32
    np.testing.assert_array_equal(sampled, expected_raw)

    built = _fake_static(3, 4)
    report = {}
    out = build_fuel_fields(
        grid, built, FuelStaticsRequest(sr_x=sr, sr_y=sr, landfire=path),
        landuse_attrs=MODIS, coverage_report=report)
    fuel, valid, _ = landfire_codes_to_fuel(expected_raw)
    landuse = refine_nearest(nfuel_from_lu_index(built["LU_INDEX"], MODIS),
                             sr, sr)
    np.testing.assert_array_equal(out["NFUEL_CAT_FIRE"],
                                  np.where(valid, fuel, landuse))
    assert report["fuel"]["landfire"]["kind"] == "fbfm40"
    import hashlib
    assert (report["fuel"]["landfire"]["sha256"]
            == hashlib.sha256(path.read_bytes()).hexdigest())
    assert (report["fuel"]["landfire"]["landuse_filled_cells"]
            == int((~valid).sum()))


def test_a_covering_raster_needs_no_land_use_table(tmp_path, monkeypatch):
    """The crosswalk only fills gaps, so a legend it lacks is no refusal
    when the raster covers every fire cell, and is one when it does not."""
    from woof.static import fuel
    raster = tmp_path / "fbfm13.tif"
    raster.write_bytes(b"stand-in")
    grid = _grid(nx=2, ny=2)
    codes = np.array([[1, 2, 8, 91]] * 4, dtype=np.int32)
    cells = np.indices(codes.shape)
    monkeypatch.setattr(fuel, "landfire_window",
                        lambda path, grid, sr_x, sr_y: (codes.copy(),
                                                        cells[0], cells[1]))
    built = {"LU_INDEX": np.full((2, 2), 5.0), "HGT_M": np.zeros((2, 2))}
    assert MISSING_CODE < 0  # the missing marker is never a valid code
    other_legend = dict(MODIS, MMINLU="SSIB")
    request = FuelStaticsRequest(sr_x=2, sr_y=2, landfire=raster)
    out = build_fuel_fields(grid, built, request, landuse_attrs=other_legend)
    np.testing.assert_array_equal(out["NFUEL_CAT_FIRE"][0], [1, 2, 8, 14])
    np.testing.assert_array_equal(out["NFUEL_CAT"], [[1, 8], [1, 8]])
    codes[0, 0] = MISSING_CODE
    with pytest.raises(ValueError, match="no land-use-to-fuel crosswalk"):
        build_fuel_fields(grid, built, request, landuse_attrs=other_legend)
    out = build_fuel_fields(grid, built, request, landuse_attrs=MODIS)
    assert out["NFUEL_CAT_FIRE"][0, 0] == 10.0  # MODIS 5 -> 10, filled


def _write(path, data, *, dtype, nodata, grid):
    rasterio = pytest.importorskip("rasterio")
    pytest.importorskip("pyproj")
    from rasterio.transform import from_origin
    lat, lon = grid.latlon_mass()
    west, north = lon.min() - 0.05, lat.max() + 0.05
    with rasterio.open(path, "w", driver="GTiff", width=data.shape[1],
                       height=data.shape[0], count=1, dtype=dtype,
                       crs="EPSG:4326", nodata=nodata,
                       transform=from_origin(west, north, 0.01, 0.01)) as out:
        out.write(data.astype(dtype), 1)


def test_raster_words_are_masked_before_they_are_checked(tmp_path):
    """A no-data word outside int32, an undeclared float fill, or a float
    product holding whole codes, is read; a value between codes is refused
    where a cell samples it, and only there."""
    from woof.static.fuel import (BAD_CODE, landfire_codes_to_fuel,
                                  sample_landfire)
    grid = _grid(nx=3, ny=3)
    codes = np.full((30, 30), 101.0)
    codes[:, ::4] = np.nan
    path = tmp_path / "float.tif"
    _write(path, codes, dtype="float32", nodata=np.nan, grid=grid)
    got = sample_landfire(path, grid, 2, 2)
    assert set(np.unique(got)) <= {101, int(MISSING_CODE)}
    wide = np.where(np.isnan(codes), -2147483648, 101)
    path = tmp_path / "int32.tif"
    _write(path, wide, dtype="int32", nodata=-2147483648, grid=grid)
    np.testing.assert_array_equal(sample_landfire(path, grid, 2, 2), got)
    filled = np.where(np.isnan(codes), -3.4028235e38, 101.0)
    path = tmp_path / "undeclared_fill.tif"
    _write(path, filled, dtype="float32", nodata=None, grid=grid)
    np.testing.assert_array_equal(sample_landfire(path, grid, 2, 2), got)
    path = tmp_path / "resampled.tif"
    _write(path, np.full((30, 30), 101.5), dtype="float32", nodata=None,
           grid=grid)
    sampled = sample_landfire(path, grid, 2, 2)
    assert (sampled == BAD_CODE).all()
    with pytest.raises(ValueError, match="no whole fuel-model code"):
        landfire_codes_to_fuel(sampled)


# --- the command -------------------------------------------------------------

def test_fuel_preview_is_a_command():
    from woof.cli import build_parser
    args = build_parser().parse_args(
        ["fuel", "preview", "case.toml", "--sr-x", "4", "--sr-y", "2",
         "--json"])
    assert (args.sr_x, args.sr_y, args.json) == (4, 2, True)
    assert args.landfire is None and args.landfire_kind == "auto"
    assert args.device == "auto"
    assert args.func.__module__ == "woof.fuel_cli"


def test_preparation_identity_is_unchanged_without_a_request(tmp_path):
    """The shared-preparation static key serializes every selection field.
    The fuel field is left out while it is None, so the key keeps its
    bytes."""
    from dataclasses import asdict
    from woof.ensemble.runtime_preparation import _digest, _plain
    selection = GeogSelection.fallback(tmp_path)
    assert "fuel" not in _plain(selection)
    legacy = {key: value for key, value in asdict(selection).items()
              if key != "fuel"}
    assert _digest(selection) == _digest(legacy)
    fired = with_fuel_statics(selection, FuelStaticsRequest(sr_x=2, sr_y=2))
    assert _plain(fired)["fuel"] == {"sr_x": 2, "sr_y": 2, "landfire": None,
                                     "landfire_kind": "auto"}
    on_gpu = with_fuel_statics(
        selection, FuelStaticsRequest(sr_x=2, sr_y=2, device="gpu"))
    assert _digest(on_gpu) == _digest(fired)  # where it runs is no identity
    assert _digest(fired) != _digest(selection)
