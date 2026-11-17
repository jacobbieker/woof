"""Building morphology (woof.static.buildings): filters, footprints, the
per-cell aggregation, WRF's URB_PARAM layout and the product file.

Synthetic geometry only: generic coordinates, no real tiles or footprints.
"""
from __future__ import annotations

import json
import math
import struct

import numpy as np
import pytest

from woof.static import buildings as b


def _cart(nx=6, ny=6, d=10.0):
    return b.CartesianGrid(nx, ny, d, d)


def test_block_totals_are_conserved():
    # One 20 x 30 m, 10 m tall block straddling cell faces.
    grid = _cart()
    raster = b.block_raster(6, 6, 10.0, 10.0, [(15.0, 35.0, 12.0, 42.0, 10.0)])
    m = b.aggregate([raster], grid)
    area = m.sampled_area.astype(np.float64)
    assert area.sum() == pytest.approx(3600.0)
    assert (m.lambda_p * area).sum() == pytest.approx(600.0, rel=1e-5)
    # Windward walls: flow along x meets the 30 m wide face, along y the 20 m.
    assert (m.frontal_x * area).sum() == pytest.approx(300.0, rel=1e-5)
    assert (m.frontal_y * area).sum() == pytest.approx(200.0, rel=1e-5)
    wall = ((m.lambda_b - m.lambda_p) * area).sum()
    assert wall == pytest.approx(2 * (20 + 30) * 10.0, rel=1e-5)
    built = m.lambda_p > 0
    assert np.allclose(m.mean_height[built], 10.0)
    assert np.allclose(m.std_height[built], 0.0, atol=1e-3)
    # bin [10, 12) holds everything, at its true height
    k = int(np.searchsorted(b.LES_BIN_EDGES, 10.0, side="right") - 1)
    assert np.allclose(m.bldg_hist[k], m.lambda_p)
    assert np.allclose(m.bldg_hbin[k][built], 10.0)
    assert np.allclose(np.delete(m.bldg_hist, k, axis=0), 0.0)
    # BEP class 2 (hb_u = 10 m) holds every building pixel.
    assert np.allclose(m.hi[1][built], 1.0)


def test_solid_fraction_on_layers():
    grid = _cart(2, 1)
    raster = b.block_raster(2, 1, 10.0, 10.0, [(0.0, 10.0, 0.0, 10.0, 10.0),
                                              (10.0, 15.0, 0.0, 10.0, 4.0)])
    m = b.aggregate([raster], grid)
    assert m.solid_fraction(0.0, 2.0)[0].tolist() == pytest.approx([1.0, 0.5])
    assert m.solid_fraction(2.0, 6.0)[0].tolist() == pytest.approx([1.0, 0.25])
    assert m.solid_fraction(8.0, 12.0)[0].tolist() == pytest.approx([0.5, 0.0])
    assert m.solid_fraction(12.0, 20.0)[0].tolist() == pytest.approx([0.0, 0.0])


def test_urb_param_layout_matches_wrf_real_init():
    grid = _cart(3, 3)
    raster = b.block_raster(3, 3, 10.0, 10.0, [(0.0, 30.0, 10.0, 20.0, 23.0)])
    m = b.aggregate([raster], grid)
    up = b.urb_param(m)
    assert up.shape == (132, 3, 3)
    assert np.array_equal(up[90], m.lambda_p)          # 91 LP_URB2D
    assert np.array_equal(up[91], m.mean_height)       # 92 MH_URB2D
    assert np.array_equal(up[92], m.std_height)        # 93 STDH_URB2D
    assert np.array_equal(up[93], m.mean_height)       # 94 HGT_URB2D
    assert np.array_equal(up[94], m.lambda_b)          # 95 LB_URB2D
    assert np.array_equal(up[95], m.frontal_y)         # 96 LF(1): N-S flow
    assert np.array_equal(up[96], m.frontal_d45)       # 97 LF(2): +x+y
    assert np.array_equal(up[97], m.frontal_x)         # 98 LF(3): E-W flow
    assert np.array_equal(up[98], m.frontal_d135)      # 99 LF(4): +x-y
    # Axis-aligned walls: each diagonal sees both faces at 45 degrees.
    assert np.allclose(up[96], (m.frontal_x + m.frontal_y) / math.sqrt(2))
    assert np.allclose(up[98], up[96])
    assert np.array_equal(up[117:132], m.hi)           # 118-132 HI_URB2D
    assert up[118 + 3, 1, 1] == pytest.approx(1.0)     # 23 m -> 25 m class
    untouched = np.ones(132, bool)
    untouched[[90, 91, 92, 93, 94, 95, 96, 97, 98, *range(117, 132)]] = False
    assert not up[untouched].any()


def test_ndsm_filters():
    dtm = np.full((6, 6), 100.0)
    dsm = dtm.copy()
    dsm[1:3, 1:3] = 112.0               # a 2x2 building
    dsm[4, 4] = 109.0                   # one-pixel object
    dsm[0, 5] = 101.0                   # below the height window
    dsm[5, 0] = np.nan                  # hole
    dsm[4, 0:2] = 115.0                 # a tree pair (rough)
    spread = np.zeros_like(dsm)
    spread[4, 0:2] = 12.0
    h, rec = b.ndsm_heights(dsm, dtm, pixel_area_m2=225.0, min_area_m2=400.0,
                            spread=spread)
    assert np.isnan(h[5, 0])
    assert np.allclose(h[1:3, 1:3], 12.0)
    assert h[4, 4] == 0.0 and h[0, 5] == 0.0 and np.all(h[4, 0:2] == 0.0)
    assert rec["rejected_spread"] == 2 and rec["rejected_small_objects"] == 1
    assert rec["building_pixels"] == 4 and rec["holes"] == 1
    mask = np.zeros((6, 6), bool)
    mask[1, 1] = True
    h2, _ = b.ndsm_heights(dsm, dtm, pixel_area_m2=225.0, min_area_m2=0.0,
                           footprint_mask=mask)
    assert np.count_nonzero(h2[np.isfinite(h2)]) == 1


def _ring(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]


def _wkb_polygon(rings, order="<", z=False):
    code = 1003 if z else 3
    out = struct.pack(order + "BI", 1 if order == "<" else 0, code)
    out += struct.pack(order + "I", len(rings))
    for ring in rings:
        out += struct.pack(order + "I", len(ring))
        for x, y in ring:
            out += struct.pack(order + ("ddd" if z else "dd"),
                               *((x, y, 7.0) if z else (x, y)))
    return out


def test_parse_wkb_polygon_and_multipolygon():
    ring = _ring(0.0, 0.0, 2.0, 1.0)
    for order in "<>":
        for z in (False, True):
            polys = b.parse_wkb(_wkb_polygon([ring], order, z))
            assert polys == (((tuple(map(tuple, ring))),),)
    multi = struct.pack("<BII", 1, 6, 2) + _wkb_polygon([ring]) + \
        _wkb_polygon([_ring(5, 5, 6, 6)], ">")
    polys = b.parse_wkb(multi)
    assert len(polys) == 2 and polys[1][0][0] == (5.0, 5.0)
    point = struct.pack("<BIdd", 1, 1, 0.0, 0.0)
    assert b.parse_wkb(point) == ()


LON, LAT = 10.0, 45.0      # generic synthetic location
DEG_M = 111_320.0


def _lonlat_box(dx_m, dy_m, w_m, h_m):
    lon0 = LON + dx_m / (DEG_M * math.cos(math.radians(LAT)))
    lat0 = LAT + dy_m / DEG_M
    lon1 = lon0 + w_m / (DEG_M * math.cos(math.radians(LAT)))
    lat1 = lat0 + h_m / DEG_M
    return _ring(lon0, lat0, lon1, lat1)


def _features():
    return [
        dict(props={"height": 30.0}, ring=_lonlat_box(20, 20, 40, 40)),
        dict(props={}, ring=_lonlat_box(100, 20, 30, 30)),             # ndsm
        dict(props={"num_floors": 4}, ring=_lonlat_box(20, 100, 30, 30)),
        dict(props={}, ring=_lonlat_box(100, 100, 30, 30)),            # none
    ]


def _like():
    return b.utm_raster(LON - 0.001, LAT - 0.001, LON + 0.003, LAT + 0.003,
                        2.0)


def _ndsm_for(like, footprints, crs):
    # A synthetic nDSM: 17 m everywhere the second building sits.
    probe, _ = b.rasterize_footprints([footprints[1]], crs, like,
                                      default_height_m=17.0)
    return np.where(probe.height > 0, 17.0, 0.0)


@pytest.mark.parametrize("kind", ["geojson", "parquet"])
def test_footprints_read_and_height_precedence(tmp_path, kind):
    feats = _features()
    if kind == "geojson":
        path = tmp_path / "fp.geojson"
        path.write_text(json.dumps({"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": f["props"],
             "geometry": {"type": "Polygon", "coordinates": [f["ring"]]}}
            for f in feats]}))
    else:
        pa = pytest.importorskip("pyarrow")
        import pyarrow.parquet as pq
        table = pa.table({
            "geometry": [_wkb_polygon([f["ring"]]) for f in feats],
            "height": [f["props"].get("height") for f in feats],
            "num_floors": [f["props"].get("num_floors") for f in feats]})
        geo = {"version": "1.1.0", "primary_column": "geometry",
               "columns": {"geometry": {"encoding": "WKB",
                                        "geometry_types": ["Polygon"]}}}
        table = table.replace_schema_metadata({"geo": json.dumps(geo)})
        path = tmp_path / "fp.parquet"
        pq.write_table(table, path)
    fps, crs = b.read_footprints(path, source="overture")
    assert crs == "OGC:CRS84" and len(fps) == 4
    assert fps[0].height == 30.0 and fps[2].levels == 4.0
    like = _like()
    ndsm = _ndsm_for(like, fps, crs)
    raster, rec = b.rasterize_footprints(fps, crs, like, ndsm=ndsm)
    assert rec["height_from"] == {"attribute": 1, "ndsm_median": 1,
                                  "storeys": 1, "default": 0, "dropped": 1}
    values = set(np.unique(raster.height).tolist())
    assert values == {0.0, 12.0, 17.0, 30.0}
    # The nDSM median supplied one height, so GEM-15 is a source.
    assert raster.sources == ("gem15-ndsm", "overture")
    # Plan area of the 40 x 40 m building within a pixel row of 2 m.
    assert np.count_nonzero(raster.height == 30.0) * 4.0 == \
        pytest.approx(1600.0, rel=0.12)
    _, rec2 = b.rasterize_footprints(fps, crs, like, default_height_m=9.0)
    assert rec2["height_from"]["default"] == 2
    # bbox filter
    keep, _ = b.read_footprints(path, bbox=(LON - 1e-4, LAT - 1e-4,
                                            LON + 4.5e-4, LAT + 4.5e-4))
    assert len(keep) == 1


def test_projection_grid_places_buildings():
    from woof.static.lambert import LambertGrid
    grid = b.ProjectionGrid(LambertGrid(LAT, LON, 30.0, 60.0, LON, 20.0, 20.0,
                                        41, 41))
    fps = [b.Footprint((tuple(map(tuple, [_lonlat_box(-20, -20, 40, 40)])),),
                       height=25.0, source="osm")]
    like = b.utm_raster(LON - 0.006, LAT - 0.004, LON + 0.006, LAT + 0.004,
                        2.0)
    raster, _ = b.rasterize_footprints(fps, "OGC:CRS84", like)
    m = b.aggregate([raster], grid)
    area = m.sampled_area.astype(np.float64)
    assert (m.lambda_p * area).sum() == pytest.approx(1600.0, rel=0.08)
    rows, cols = np.nonzero(m.lambda_p > 0.5)
    # The 40 m square centred on the domain centre covers the 2x2 central
    # mass cells (the centre point is the corner they share).
    assert sorted(set(rows.tolist())) == [19, 20]
    assert sorted(set(cols.tolist())) == [19, 20]
    assert m.sources == ("osm",)
    assert abs(grid.axis_rotation(raster)) < math.radians(3.0)


def test_geographic_raster_refused():
    raster = b.HeightRaster(np.zeros((2, 2)), 0.0, 1.0, 0.1, 0.1, "EPSG:4326")
    with pytest.raises(ValueError, match="geographic"):
        b.aggregate([raster], _cart())


def test_product_round_trip(tmp_path):
    grid = _cart(4, 3)
    raster = b.block_raster(4, 3, 10.0, 10.0, [(5.0, 25.0, 5.0, 15.0, 33.0)])
    raster = b.HeightRaster(raster.height, raster.x0, raster.y0, raster.px,
                            raster.py, None, ("gem15-ndsm", "osm"))
    m = b.aggregate([raster], grid)
    path = b.write_product(tmp_path / "bldg.nc", m, grid_note="test")
    back = b.read_product(path)
    for name in ("lambda_p", "frontal_x", "bldg_hist", "bldg_hbin", "hi"):
        assert np.array_equal(getattr(back, name), getattr(m, name)), name
    assert back.sources == ("gem15-ndsm", "osm")
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        assert "produced by Google" in ds.attribution
        assert json.loads(ds.licenses) == {"gem15-ndsm": "CC-BY-4.0",
                                           "osm": "ODbL-1.0"}
        assert np.array_equal(ds["URB_PARAM"][90], m.lambda_p)
    (tmp_path / "bad.nc").write_bytes(path.read_bytes())
    with netCDF4.Dataset(tmp_path / "bad.nc", "a") as ds:
        ds.schema = "other"
    with pytest.raises(ValueError, match="woof-buildings.v1"):
        b.read_product(tmp_path / "bad.nc")


def test_gem15_window_pairs(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin
    dtm = np.full((8, 8), 50.0, np.float32)
    dsm = dtm.copy()
    dsm[2:5, 2:5] = 70.0
    paths = []
    for name, data in (("dsm", dsm), ("dtm", dtm)):
        path = tmp_path / f"{name}.tif"
        with rasterio.open(path, "w", driver="GTiff", height=8, width=8,
                           count=1, dtype="float32", crs="EPSG:32632",
                           transform=from_origin(500000.0, 5000000.0, 15, 15)
                           ) as dst:
            dst.write(data, 1)
        paths.append(path)
    rasters, receipts = b.gem15_building_rasters([paths[0]], [paths[1]])
    assert rasters[0].crs == "EPSG:32632" and rasters[0].sources == (
        "gem15-ndsm",)
    assert np.count_nonzero(rasters[0].height == 20.0) == 9
    assert receipts[0]["building_pixels"] == 9
    with rasterio.open(paths[1], "r+") as dst:
        dst.transform = from_origin(500015.0, 5000000.0, 15, 15)
    with pytest.raises(ValueError, match="not one grid"):
        b.gem15_building_rasters([paths[0]], [paths[1]])


def test_registry_rows_name_licences():
    for key, row in b.BUILDING_SOURCES.items():
        assert row["license_id"] and row["attribution"], key
    assert b.BUILDING_SOURCES["overture"]["license_id"] == "ODbL-1.0"
    assert b.BUILDING_SOURCES["gem15-ndsm"]["attribution"].startswith(
        "This dataset is produced by Google")
    with pytest.raises(KeyError):
        b.HeightRaster(np.zeros((1, 1)), 0, 0, 1, 1, None, ("nope",))


class _RotatedCart(b.CartesianGrid):
    """A Cartesian grid whose rasters are rotated by ``angle`` onto it
    (only the frontal projection reads the rotation)."""

    def __init__(self, *args, angle):
        super().__init__(*args)
        self.angle = angle

    def axis_rotation(self, raster):
        return self.angle


def test_diagonal_frontal_area_follows_wall_orientation():
    # A 20 m square, 10 m tall: one face is C = 200 m2.  Rotated 45 degrees
    # onto the model axes, flow along one diagonal meets one face square on.
    raster = b.block_raster(6, 6, 10.0, 10.0, [(20.0, 40.0, 20.0, 40.0, 10.0)])
    for angle in (math.pi / 4, -math.pi / 4):
        m = b.aggregate([raster], _RotatedCart(6, 6, 10.0, 10.0, angle=angle))
        area = m.sampled_area.astype(np.float64)
        total = lambda plane: float((plane * area).sum())
        assert total(m.frontal_d45) == pytest.approx(200.0, rel=1e-5)
        assert total(m.frontal_d135) == pytest.approx(200.0, rel=1e-5)
        assert total(m.frontal_x) == pytest.approx(200.0 * math.sqrt(2),
                                                   rel=1e-5)
    # A 40 x 20 m slab rotated +45 degrees: its 40 m faces' normals lie at
    # +45 degrees in the model plane and its 20 m faces' at -45, so the
    # +x+y flow meets a 40 m face and the +x-y flow a 20 m face.
    slab = b.block_raster(6, 6, 10.0, 10.0, [(10.0, 50.0, 20.0, 40.0, 10.0)])
    m = b.aggregate([slab], _RotatedCart(6, 6, 10.0, 10.0, angle=math.pi / 4))
    area = m.sampled_area.astype(np.float64)
    assert float((m.frontal_d45 * area).sum()) == pytest.approx(400.0, rel=1e-5)
    assert float((m.frontal_d135 * area).sum()) == pytest.approx(200.0,
                                                                rel=1e-5)


def _split(raster, col):
    left = b.HeightRaster(raster.height[:, :col], raster.x0, raster.y0,
                          raster.px, raster.py, raster.crs, raster.sources)
    right = b.HeightRaster(raster.height[:, col:],
                           raster.x0 + col * raster.px, raster.y0, raster.px,
                           raster.py, raster.crs, raster.sources)
    return left, right


def test_tiles_on_one_lattice_mosaic_seams_and_overlaps():
    grid = _cart(3, 3, 1.0)
    whole = b.HeightRaster(np.array([[0, 0, 0], [0, 10, 0], [0, 0, 0]],
                                    np.float32), 0.0, 3.0, 1.0, 1.0)
    ref = b.aggregate([whole], grid, min_samples=1)
    wall = lambda m: float(((m.lambda_b - m.lambda_p) * m.sampled_area).sum())
    assert wall(ref) == pytest.approx(40.0)
    tiles = _split(whole, 2)
    for rasters in (tiles, tiles[::-1], [whole, *tiles], [whole, whole]):
        m = b.aggregate(rasters, grid, min_samples=1)
        assert wall(m) == pytest.approx(40.0)
        for name in ("sampled_area", "lambda_p", "frontal_x", "frontal_y",
                     "bldg_hist"):
            assert np.array_equal(getattr(m, name), getattr(ref, name)), name
    # Overlapping known pixels: the taller wins.
    low = b.HeightRaster(np.full((3, 3), 4.0, np.float32), 0.0, 3.0, 1.0, 1.0)
    m = b.aggregate([low, whole], grid, min_samples=1)
    assert m.mean_height[1, 1] == pytest.approx(10.0)
    assert m.mean_height[0, 0] == pytest.approx(4.0)


def test_overlap_across_lattices_refuses():
    grid = _cart(3, 3, 1.0)
    a = b.HeightRaster(np.zeros((3, 3), np.float32), 0.0, 3.0, 1.0, 1.0)
    shifted = b.HeightRaster(np.zeros((3, 3), np.float32), 0.5, 3.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="different pixel lattices overlap"):
        b.aggregate([a, shifted], grid, min_samples=1)
    finer = b.HeightRaster(np.zeros((2, 6), np.float32), 0.0, 1.0, 0.5, 0.5)
    top = b.HeightRaster(np.zeros((2, 3), np.float32), 0.0, 3.0, 1.0, 1.0)
    b.aggregate([top, finer], grid, min_samples=1)      # disjoint: fine


def test_footprint_medians_ignore_overlap_and_order():
    like = _like()
    ring = _lonlat_box(20, 20, 40, 40)
    no_height = b.Footprint(((tuple(map(tuple, ring)),),), source="osm")
    ten = b.Footprint(((tuple(map(tuple, ring)),),), height=10.0,
                      source="osm")
    ndsm = b.HeightRaster(np.full(like.shape, 20.0, np.float32), like.x0,
                          like.y0, like.px, like.py, like.crs,
                          ("gem15-ndsm",))
    for fps in ([no_height, ten], [ten, no_height]):
        raster, rec = b.rasterize_footprints(fps, "OGC:CRS84", like,
                                             ndsm=ndsm)
        assert rec["height_from"]["ndsm_median"] == 1
        assert set(np.unique(raster.height).tolist()) == {0.0, 20.0}
        assert raster.sources == ("gem15-ndsm", "osm")
    # No nDSM height used: GEM-15 is not a source.
    raster, _ = b.rasterize_footprints([ten], "OGC:CRS84", like, ndsm=ndsm)
    assert raster.sources == ("osm",)


def test_utm_raster_covers_curved_edges():
    from pyproj import Transformer
    like = b.utm_raster(2.0, 45.0, 4.0, 46.0, 15.0)
    tr = Transformer.from_crs("EPSG:4326", like.crs, always_xy=True)
    x, y = tr.transform(3.0, 45.0001)
    rows, cols = like.shape
    assert like.x0 <= x <= like.x0 + cols * like.px
    assert like.y0 - rows * like.py <= y <= like.y0


def _parquet_with_crs(tmp_path, crs_entry):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    column = {"encoding": "WKB", "geometry_types": ["Polygon"], **crs_entry}
    table = pa.table({"geometry": [_wkb_polygon([_ring(0, 0, 10, 10)])]})
    table = table.replace_schema_metadata({"geo": json.dumps(
        {"version": "1.1.0", "primary_column": "geometry",
         "columns": {"geometry": column}})})
    path = tmp_path / "fp.parquet"
    pq.write_table(table, path)
    return path


def test_geoparquet_crs_rules(tmp_path):
    from pyproj import CRS
    with pytest.raises(ValueError, match="null"):
        b.read_footprints(_parquet_with_crs(tmp_path, {"crs": None}))
    custom = CRS.from_proj4("+proj=tmerc +lat_0=0 +lon_0=7 +k=1 +x_0=0 "
                            "+y_0=0 +ellps=WGS84 +units=m").to_json_dict()
    custom.pop("id", None)
    _, crs = b.read_footprints(_parquet_with_crs(tmp_path, {"crs": custom}))
    parsed = CRS.from_user_input(crs)
    assert parsed.is_projected and not parsed.is_geographic
    tagged = CRS.from_epsg(32632).to_json_dict()
    _, crs = b.read_footprints(_parquet_with_crs(tmp_path, {"crs": tagged}))
    assert crs == "EPSG:32632"
