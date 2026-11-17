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
    assert np.array_equal(up[97], m.frontal_x)         # 98 LF(3): E-W flow
    assert np.allclose(up[96], (m.frontal_x + m.frontal_y) / math.sqrt(2))
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
    assert raster.sources == ("overture",)
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
