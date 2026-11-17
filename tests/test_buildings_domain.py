"""``woof buildings``: the buildings product for one WRF domain."""
from __future__ import annotations

import json

import numpy as np
import pytest

from woof.static import buildings as b
from woof.static import buildings_domain as bd

LAT, LON = 47.0, -100.0
NX = NY = 40
DX = 50.0


def _namelist(tmp_path, *, nest=False):
    path = tmp_path / "namelist.wps"
    if nest:
        layout = (" max_dom = 2,\n/\n&geogrid\n"
                  " parent_id = 1, 1,\n parent_grid_ratio = 1, 5,\n"
                  " i_parent_start = 1, 4,\n j_parent_start = 1, 4,\n"
                  f" e_we = 13, {NX + 1},\n e_sn = 13, {NY + 1},\n"
                  f" dx = {DX * 5},\n dy = {DX * 5},\n")
    else:
        layout = (" max_dom = 1,\n/\n&geogrid\n parent_id = 1,\n"
                  " parent_grid_ratio = 1,\n i_parent_start = 1,\n"
                  f" j_parent_start = 1,\n e_we = {NX + 1},\n"
                  f" e_sn = {NY + 1},\n dx = {DX},\n dy = {DX},\n")
    path.write_text(
        "&share\n wrf_core = 'ARW',\n" + layout
        + " map_proj = 'lambert',\n"
        f" ref_lat = {LAT},\n ref_lon = {LON},\n truelat1 = 30.0,\n"
        f" truelat2 = 60.0,\n stand_lon = {LON},\n/\n")
    return path


def _grid(tmp_path, **kw):
    from woof.static.projection import grids_from_wps_namelist
    return grids_from_wps_namelist(_namelist(tmp_path, **kw))[-1]


def _square(grid, i, j, half_m, **props):
    """A lon/lat square of side 2*half_m centred on mass point (i, j)."""
    lat, lon = grid.ij_to_latlon(np.array([float(i)]), np.array([float(j)]))
    lat, lon = float(np.ravel(lat)[0]), float(np.ravel(lon)[0])
    dlat = half_m / 111_195.0
    dlon = dlat / np.cos(np.radians(lat))
    ring = [[lon - dlon, lat - dlat], [lon + dlon, lat - dlat],
            [lon + dlon, lat + dlat], [lon - dlon, lat + dlat],
            [lon - dlon, lat - dlat]]
    return {"type": "Feature", "properties": props,
            "geometry": {"type": "Polygon", "coordinates": [ring]}}


def _geojson(tmp_path, features, name="fp.geojson"):
    path = tmp_path / name
    path.write_text(json.dumps({"type": "FeatureCollection",
                                "features": features}))
    return path


def _built_area(morph):
    return float((morph.lambda_p * morph.sampled_area).sum())


def test_domain_bbox_encloses_every_cell(tmp_path):
    grid = _grid(tmp_path)
    box = bd.domain_bbox(grid, halo_cells=0.0)
    i, j = np.meshgrid(np.arange(1, NX + 1) + 0.0, np.arange(1, NY + 1) + 0.0)
    lat, lon = grid.ij_to_latlon(i.ravel(), j.ravel())
    assert box.lat_min < np.min(lat) and np.max(lat) < box.lat_max
    assert box.lon_min < np.min(lon) and np.max(lon) < box.lon_max
    wide = bd.domain_bbox(grid, halo_cells=4.0)
    assert wide.lat_min < box.lat_min and wide.lon_max > box.lon_max


def test_footprint_spec():
    assert bd.parse_footprint_spec("a.geojson:osm") == (
        bd.Path("a.geojson"), "osm")
    assert bd.parse_footprint_spec("dir/a.parquet") == (
        bd.Path("dir/a.parquet"), "user")
    with pytest.raises(ValueError, match="unknown footprint source"):
        bd.parse_footprint_spec("a.geojson:nope")


def test_cli_footprints_only(tmp_path, capsys):
    pytest.importorskip("rasterio")
    from woof.cli import main
    grid = _grid(tmp_path)
    fp = _geojson(tmp_path, [_square(grid, 20, 20, 50.0, height=30.0),
                             _square(grid, 8, 30, 20.0, **{
                                 "building:levels": 4}),
                             _square(grid, 30, 8, 20.0)])
    out = tmp_path / "bld.nc"
    code = main(["buildings", "--namelist", str(tmp_path / "namelist.wps"),
                 "--output", str(out), "--footprints", f"{fp}:osm",
                 "--no-gem15"])
    assert code == 0
    morph = b.read_product(out)
    assert morph.shape == (NY, NX)
    # 100 m x 100 m plus 40 m x 40 m; the footprint with no height and no
    # storeys is left out (no --default-height).
    assert _built_area(morph) == pytest.approx(1.0e4 + 1.6e3, rel=0.03)
    tall = morph.mean_height[morph.lambda_p > 0]
    assert set(np.round(tall).astype(int).tolist()) == {12, 30}
    assert morph.sources == ("osm",)
    receipt = json.loads(capsys.readouterr().out.split("\n", 1)[1])
    burn = receipt["footprint_files"][0]
    assert burn["source"] == "osm" and burn["file"] == "fp.geojson"
    assert burn["height_from"]["attribute"] == 1
    assert burn["height_from"]["gem15_ndsm_median"] == 0


def test_cli_picks_the_innermost_domain_and_refuses_a_missing_one(
        tmp_path, capsys):
    pytest.importorskip("rasterio")
    from woof.cli import main
    grid = _grid(tmp_path, nest=True)
    fp = _geojson(tmp_path, [_square(grid, 20, 20, 50.0, height=10.0)])
    out = tmp_path / "bld.nc"
    nl = str(tmp_path / "namelist.wps")
    assert main(["buildings", "--namelist", nl, "--output", str(out),
                 "--footprints", str(fp), "--no-gem15"]) == 0
    assert b.read_product(out).shape == (NY, NX)
    assert main(["buildings", "--namelist", nl, "--domain", "3",
                 "--output", str(out), "--footprints", str(fp),
                 "--no-gem15"]) == 2
    assert "--domain 3 is not in" in capsys.readouterr().err
    assert main(["buildings", "--namelist", nl, "--output", str(out),
                 "--no-gem15"]) == 2


class _Fetched:
    def __init__(self, path):
        self.path = path

    def receipt(self):
        return {"path": self.path.name}


def _gem15_windows(tmp_path, monkeypatch, grid, *, raise_m=25.0):
    """Synthetic GEM-15 dsm/dtm windows over the domain with a raised
    patch 120 m square at mass point (20, 20); fetch_gem15_windows is
    replaced so nothing touches the network."""
    rasterio = pytest.importorskip("rasterio")
    from pyproj import Transformer
    from rasterio.transform import from_origin
    crs = "EPSG:32614"
    lat, lon = grid.ij_to_latlon(np.array([20.0]), np.array([20.0]))
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(
        float(np.ravel(lon)[0]), float(np.ravel(lat)[0]))
    n, px = 240, 15.0
    x0, y0 = x - n * px / 2, y + n * px / 2
    dtm = np.full((n, n), 400.0, np.float32)
    dsm = dtm.copy()
    dsm[116:124, 116:124] += raise_m          # 120 m x 120 m at the centre
    paths = {}
    for name, data in (("dsm", dsm), ("dtm", dtm)):
        path = tmp_path / f"gem-{name}.tif"
        with rasterio.open(path, "w", driver="GTiff", height=n, width=n,
                           count=1, dtype="float32", crs=crs,
                           transform=from_origin(x0, y0, px, px)) as dst:
            dst.write(data, 1)
        paths[f"gem15-{name}"] = path
    calls = []

    def fake(bbox, root, source_id, **kw):
        calls.append((source_id, kw.get("offline")))
        return (_Fetched(paths[source_id]),), {}

    from woof.static import gem15
    monkeypatch.setattr(gem15, "fetch_gem15_windows", fake)
    return calls


def test_gem15_supplies_missing_footprint_heights(tmp_path, monkeypatch):
    grid = _grid(tmp_path)
    calls = _gem15_windows(tmp_path, monkeypatch, grid)
    fp = _geojson(tmp_path, [_square(grid, 20, 20, 50.0),
                             _square(grid, 8, 8, 20.0, height=7.0)])
    path, receipt = bd.build_domain_buildings(
        grid, tmp_path / "bld.nc", footprints=[(fp, "osm")],
        cache_root=tmp_path / "cache", offline=True)
    assert calls == [("gem15-dsm", True), ("gem15-dtm", True)]
    morph = b.read_product(path)
    heights = np.round(morph.mean_height[morph.lambda_p > 0]).astype(int)
    assert set(heights.tolist()) == {7, 25}
    assert morph.sources == ("gem15-ndsm", "osm")
    burn = receipt["footprint_files"][0]
    assert burn["height_from"]["gem15_ndsm_median"] == 1
    assert burn["height_from"]["attribute"] == 1
    # Only footprints are burned: the nDSM patch around the footprint
    # (120 m against the footprint's 100 m) adds no area.
    assert _built_area(morph) == pytest.approx(1.0e4 + 1.6e3, rel=0.03)


def test_gem15_alone(tmp_path, monkeypatch):
    grid = _grid(tmp_path)
    _gem15_windows(tmp_path, monkeypatch, grid)
    path, receipt = bd.build_domain_buildings(
        grid, tmp_path / "bld.nc", cache_root=tmp_path / "cache")
    morph = b.read_product(path)
    assert _built_area(morph) == pytest.approx(120.0 * 120.0, rel=0.05)
    assert np.nanmax(morph.mean_height) == pytest.approx(25.0)
    assert morph.sources == ("gem15-ndsm",)
    assert receipt["gem15"]["ndsm"][0]["building_pixels"] == 64


def test_gem15_alone_drops_small_objects(tmp_path, monkeypatch):
    grid = _grid(tmp_path)
    _gem15_windows(tmp_path, monkeypatch, grid)
    path, receipt = bd.build_domain_buildings(
        grid, tmp_path / "bld.nc", cache_root=tmp_path / "cache",
        min_area_m2=20000.0)
    assert _built_area(b.read_product(path)) == 0.0
    assert receipt["gem15"]["ndsm"][0]["building_pixels"] == 0


def test_nothing_to_build_from(tmp_path):
    with pytest.raises(ValueError, match="nothing to build from"):
        bd.build_domain_buildings(_grid(tmp_path), tmp_path / "x.nc",
                                  gem15=False)
