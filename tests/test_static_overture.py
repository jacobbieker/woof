"""The Overture Maps buildings fetcher, against a fake local release."""
from __future__ import annotations

import json
import os
import struct

import pytest

import pyarrow as pa  # noqa: E402
import pyarrow.fs as pafs  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from woof.static import buildings as b  # noqa: E402
from woof.static import overture as ov  # noqa: E402
from woof.static.highres_fetch import FootprintBBox  # noqa: E402


def _wkb(x0, y0, x1, y1):
    ring = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    out = struct.pack("<BII", 1, 3, 1) + struct.pack("<I", len(ring))
    for x, y in ring:
        out += struct.pack("<dd", x, y)
    return out


SRC = pa.list_(pa.struct([("property", pa.string()),
                          ("dataset", pa.string()),
                          ("license", pa.string())]))
ROWS = [  # id, box, height, floors, sources
    ("a", (-74.011, 40.7035, -74.0108, 40.7037), 12.0, None,
     [("", "OpenStreetMap", "ODbL-1.0")]),
    ("b", (-74.0095, 40.7045, -74.0093, 40.7047), None, 4,
     [("", "Microsoft ML Buildings", "ODbL-1.0"),
      ("", "OpenStreetMap", "ODbL-1.0")]),
    ("far", (-73.5, 40.0, -73.4, 40.1), 30.0, None,
     [("", "Esri Community Maps", "CC-BY-4.0")]),
]


def _release(root, release="2026-11-18.0"):
    part = root / release / ov.THEME_PATH
    part.mkdir(parents=True)
    box = pa.struct([("xmin", pa.float64()), ("xmax", pa.float64()),
                     ("ymin", pa.float64()), ("ymax", pa.float64())])
    table = pa.table({
        "id": [r[0] for r in ROWS],
        "geometry": pa.array([_wkb(*r[1]) for r in ROWS], pa.binary()),
        "bbox": pa.array([dict(xmin=r[1][0], xmax=r[1][2], ymin=r[1][1],
                               ymax=r[1][3]) for r in ROWS], box),
        "height": pa.array([r[2] for r in ROWS], pa.float64()),
        "num_floors": pa.array([r[3] for r in ROWS], pa.int32()),
        "sources": pa.array([[dict(property=s[0], dataset=s[1],
                                   license=s[2]) for s in r[4]]
                             for r in ROWS], SRC),
        "subtype": ["x"] * len(ROWS),
    })
    pq.write_table(table, part / "part-0.parquet")


BOX = FootprintBBox(40.703, 40.706, -74.012, -74.008)


def _fetch(tmp_path, **kw):
    return ov.fetch_overture_buildings(
        BOX, tmp_path / "cache", filesystem=pafs.LocalFileSystem(),
        base=str(tmp_path / "release"), **kw)


def test_bbox_filter_columns_and_provenance(tmp_path):
    _release(tmp_path / "release")
    path, side = _fetch(tmp_path, release="2026-11-18.0")
    table = pq.read_table(path)
    assert sorted(table.column("id").to_pylist()) == ["a", "b"]
    assert set(table.column_names) == set(ov.COLUMNS)
    assert side["release"] == "2026-11-18.0"
    assert side["n_footprints"] == 2
    assert side["sources_by_dataset"] == {"Microsoft ML Buildings": 1,
                                          "OpenStreetMap": 2}
    assert side["sources_by_license"] == {"ODbL-1.0": 2}
    assert side["theme_license"] == "ODbL-1.0"
    assert side["attribution"] == ("© OpenStreetMap contributors, "
                                   "Overture Maps Foundation")
    assert side["bbox"] == dict(lon_min=-74.012, lat_min=40.703,
                                lon_max=-74.008, lat_max=40.706)
    assert side["location"].endswith(f"2026-11-18.0/{ov.THEME_PATH}/")
    assert side["fetched_utc"].endswith("Z")
    on_disk = json.loads(path.with_suffix(".json").read_text())
    assert on_disk == side


def test_read_footprints_reads_the_output(tmp_path):
    _release(tmp_path / "release")
    path, _ = _fetch(tmp_path, release="latest")
    fps, crs = b.read_footprints(path, source="overture")
    assert crs == "OGC:CRS84" and len(fps) == 2
    by_height = sorted((fp.height, fp.levels) for fp in fps
                       if fp.height == fp.height or fp.levels == fp.levels)
    assert any(h == 12.0 for h, _ in by_height)
    assert any(lv == 4 for _, lv in by_height)
    assert {fp.source for fp in fps} == {"overture"}


def test_latest_picks_the_newest_release(tmp_path):
    for name in ("2026-09-23.1", "2026-10-21.0", "2026-10-21.10",
                 "2026-10-21.9"):
        (tmp_path / "release" / name).mkdir(parents=True)
    (tmp_path / "release" / "notes").mkdir()
    assert ov.latest_release(pafs.LocalFileSystem(),
                             str(tmp_path / "release")) == "2026-10-21.10"


def test_second_call_is_cached_and_offline_uses_it(tmp_path):
    _release(tmp_path / "release")
    path, side = _fetch(tmp_path, release="2026-11-18.0")
    mtime = os.stat(path).st_mtime_ns
    # The release is gone: only the cache can answer now.
    import shutil
    shutil.rmtree(tmp_path / "release")
    again, side2 = _fetch(tmp_path, release="2026-11-18.0")
    assert again == path and side2 == side
    assert os.stat(path).st_mtime_ns == mtime
    off, side3 = ov.fetch_overture_buildings(
        BOX, tmp_path / "cache", release="latest", offline=True)
    assert off == path and side3 == side


def test_offline_refuses_cleanly_when_not_cached(tmp_path):
    with pytest.raises(ov.OvertureRefusal, match="--offline"):
        ov.fetch_overture_buildings(BOX, tmp_path / "cache", offline=True)
    _release(tmp_path / "release")
    _fetch(tmp_path, release="2026-11-18.0")
    other = FootprintBBox(10.0, 10.1, 10.0, 10.1)
    with pytest.raises(ov.OvertureRefusal, match="--offline"):
        ov.fetch_overture_buildings(other, tmp_path / "cache", offline=True)


def test_bad_or_missing_release_refuses(tmp_path):
    _release(tmp_path / "release")
    with pytest.raises(ov.OvertureRefusal):
        _fetch(tmp_path, release="../x")
    with pytest.raises(ov.OvertureRefusal, match="cannot read"):
        _fetch(tmp_path, release="1999-01-01.0")


def test_empty_box_writes_an_empty_readable_file(tmp_path):
    _release(tmp_path / "release")
    path, side = ov.fetch_overture_buildings(
        FootprintBBox(0.0, 0.1, 0.0, 0.1), tmp_path / "cache",
        release="2026-11-18.0", filesystem=pafs.LocalFileSystem(),
        base=str(tmp_path / "release"))
    assert side["n_footprints"] == 0 and side["sources_by_dataset"] == {}
    assert b.read_footprints(path, source="overture")[0] == []


def _grid(tmp_path):
    from woof.static.projection import grids_from_wps_namelist
    path = tmp_path / "namelist.wps"
    path.write_text(
        "&share\n wrf_core = 'ARW',\n max_dom = 1,\n/\n&geogrid\n"
        " parent_id = 1,\n parent_grid_ratio = 1,\n i_parent_start = 1,\n"
        " j_parent_start = 1,\n e_we = 41,\n e_sn = 41,\n dx = 50.0,\n"
        " dy = 50.0,\n map_proj = 'lambert',\n ref_lat = 40.7045,\n"
        " ref_lon = -74.01,\n truelat1 = 30.0,\n truelat2 = 60.0,\n"
        " stand_lon = -74.01,\n/\n")
    return grids_from_wps_namelist(path)[-1]


def test_domain_build_carries_the_overture_receipt(tmp_path, monkeypatch):
    pytest.importorskip("rasterio")
    from woof.static import buildings_domain as bd

    grid = _grid(tmp_path)
    box = bd.domain_bbox(grid, halo_cells=4.0)
    root = tmp_path / "release"
    part = root / "2026-11-18.0" / ov.THEME_PATH
    part.mkdir(parents=True)
    lon, lat = (box.lon_min + box.lon_max) / 2, (box.lat_min + box.lat_max) / 2
    boxt = pa.struct([("xmin", pa.float64()), ("xmax", pa.float64()),
                      ("ymin", pa.float64()), ("ymax", pa.float64())])
    d = 0.0004
    pq.write_table(pa.table({
        "id": ["x"],
        "geometry": pa.array([_wkb(lon - d, lat - d, lon + d, lat + d)],
                             pa.binary()),
        "bbox": pa.array([dict(xmin=lon - d, xmax=lon + d, ymin=lat - d,
                               ymax=lat + d)], boxt),
        "height": [15.0], "num_floors": pa.array([None], pa.int32()),
        "sources": pa.array([[dict(property="", dataset="OpenStreetMap",
                                   license="ODbL-1.0")]], SRC)}),
        part / "p.parquet")
    out = tmp_path / "bld.nc"
    orig = ov.fetch_overture_buildings

    def fetch(bbox, cache, **kw):
        return orig(bbox, cache, base=str(root), **kw)
    monkeypatch.setattr(ov, "fetch_overture_buildings", fetch)
    _, receipt = bd.build_domain_buildings(
        grid, out, gem15=False, cache_root=tmp_path / "c",
        overture="latest", overture_filesystem=pafs.LocalFileSystem())
    assert receipt["overture"]["release"] == "2026-11-18.0"
    assert receipt["overture"]["n_footprints"] == 1
    assert b.read_product(out).sources == ("overture",)
    assert receipt["footprint_files"][0]["source"] == "overture"


@pytest.mark.skipif(not os.environ.get("WOOF_OVERTURE_NETWORK"),
                    reason="set WOOF_OVERTURE_NETWORK=1 to fetch for real")
def test_real_fetch_of_a_tiny_box(tmp_path):
    path, side = ov.fetch_overture_buildings(BOX, tmp_path)
    assert side["n_footprints"] > 0
    assert b.read_footprints(path, source="overture")[0]


def test_cli_offline_overture_refuses_without_a_cache(tmp_path, capsys):
    from woof.cli import main
    _grid(tmp_path)
    rc = main(["buildings", "--namelist", str(tmp_path / "namelist.wps"),
               "--output", str(tmp_path / "o.nc"), "--no-gem15",
               "--overture", "--offline", "--cache", str(tmp_path / "c")])
    assert rc == 2
    assert "--offline" in capsys.readouterr().err
