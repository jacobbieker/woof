"""``woof hex static-highres --terrain gem15-dtm|gem15-dsm``.

Offline: a synthetic GEM-15 cell (written here, in UTM zone 30N over the
test mesh's patch) is staged in the block cache with its sidecars, the
pinned index is replaced by one naming it, and the whole door runs with
``offline=True`` through the Rust warp.  No real GEM-15 bytes.
"""

from __future__ import annotations

import base64
import json

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")
pytest.importorskip("scipy")

from test_static_highres import (PATCH_LAT, PATCH_LON,  # noqa: E402,F401
                                 _rust_available, patch_static)
from woof.hex import static_highres as sh  # noqa: E402
from woof.static import cog_range, gem15, utm  # noqa: E402
from woof.static import highres_fetch as hf  # noqa: E402

EPSG = 32630
BLOCK = 512
N = 4 * BLOCK
SLOPE = 0.018          # metres of height per metre of northing
CRC = base64.b64encode(b"\x0a\x0b\x0c\x0d").decode()


def _stage_cell(cache, monkeypatch, tmp_path):
    """Write a synthetic DTM cell around the patch, stage its head and
    blocks as cached ranges, and pin an index naming it."""
    px, py = utm.forward(EPSG, PATCH_LON, PATCH_LAT)
    x0 = float(np.floor((px - N * 15.0 / 2) / 15.0) * 15.0)
    y0 = float(np.ceil((py + N * 15.0 / 2) / 15.0) * 15.0)
    rows = np.arange(N)[:, None]
    northing = y0 - 15.0 * (rows + 0.5)
    values = np.broadcast_to(250.0 + SLOPE * (northing - py),
                             (N, N)).astype(np.float32)
    remote = tmp_path / "remote.tif"
    cog_range.write_tiled_geotiff(
        remote, (values[r * BLOCK:(r + 1) * BLOCK, c * BLOCK:(c + 1) * BLOCK]
                 for r in range(4) for c in range(4)),
        rows=4, cols=4, block=BLOCK,
        transform=(15.0, 0.0, x0, 0.0, -15.0, y0), epsg=EPSG)
    body = remote.read_bytes()
    obj = gem15.Gem15Object(
        url="https://storage.invalid/gem-15/p50_m10/wm4s52/dtm/zall/mosaic.tif",
        bytes=len(body), md5_base64=None, crc32c_base64=CRC, generation="1",
        shape=(N, N), transform=(15.0, 0.0, x0, 0.0, -15.0, y0))
    cell = gem15.Gem15Cell(cell_id="wm4s52", group="p50_m10", lat_sw=52,
                           lon_sw=-4, epsg=EPSG,
                           objects={"dtm": obj, "dsm": obj})
    index = gem15.Gem15Index(snapshot_utc="2026-11-13T00:00:00Z",
                             bucket_url="https://storage.invalid/gem-15",
                             cells={(52, -4): cell})
    monkeypatch.setattr(gem15, "load_index", lambda: index)
    folder = cache / "gem15" / "wm4s52" / "dtm" / obj.crc32c
    folder.mkdir(parents=True)
    size = min(cog_range.HEAD_BYTES, len(body))
    pieces = {f"head_{size}.bin": (0, size - 1)}
    layout = cog_range.parse_head(body)
    for r in range(4):
        for c in range(4):
            offset, count = layout.block(r, c)
            pieces[f"block_r{r:03d}_c{c:03d}.bin"] = (offset,
                                                      offset + count - 1)
    for name, (start, end) in pieces.items():
        (folder / name).write_bytes(body[start:end + 1])
        hf.record_local_artifact(folder / name,
                                 url=f"{obj.url}#bytes={start}-{end}")
    return py


def test_choices_and_pixel():
    assert sh.TERRAIN_CHOICES["gem15-dtm"] == "gem15-dtm"
    assert sh.TERRAIN_CHOICES["gem15-dsm"] == "gem15-dsm"
    assert sh.SOURCE_PIXEL_M["gem15-dtm"] == 15.0
    assert sh.SOURCE_PIXEL_M["gem15-dsm"] == 15.0


def test_excluded_region_falls_back_to_glo30_offline(tmp_path, capsys):
    # Moscow: GEM-15 is excluded, so GLO-30 is used instead; offline with
    # an empty cache, that is GLO-30's own refusal, before any network.
    with pytest.raises(hf.HighresFetchRefusal, match="GLO-30"):
        sh.fetch_terrain(hf.FootprintBBox(55.7, 55.8, 37.5, 37.7), tmp_path,
                         "gem15-dtm", offline=True)
    err = capsys.readouterr().err
    assert "EXPERIMENTAL" in err
    assert "copernicus-dem-glo30 instead" in err and "Russia" in err


def test_offline_without_cached_blocks_refuses(tmp_path, monkeypatch):
    _stage_cell(tmp_path / "staged", monkeypatch, tmp_path)
    with pytest.raises(hf.HighresFetchRefusal, match="offline"):
        sh.fetch_terrain(hf.FootprintBBox(52.2, 52.3, -3.7, -3.5),
                         tmp_path / "empty", "gem15-dtm", offline=True)


@pytest.mark.skipif(not _rust_available(),
                    reason="static-fields bridge not staged")
def test_end_to_end_gem15_through_the_rust_warp(patch_static, tmp_path,
                                                monkeypatch, capsys):
    path, points, _, _ = patch_static
    cache = tmp_path / "cache"
    py = _stage_cell(cache, monkeypatch, tmp_path)
    out = tmp_path / "gem15.static.nc"
    receipt = sh.run_static_highres(path, out, terrain="gem15-dtm",
                                    max_spacing_km=0.5, cache_root=cache,
                                    offline=True)
    with netCDF4.Dataset(path) as ds:
        geometry = sh.read_cell_geometry(ds)
    selected = sh.select_cells(geometry, sh.cell_spacing_m(geometry), 0.5)
    lat_c = np.degrees(np.arcsin(points[:, 2]))
    lon_c = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    _, northing = utm.forward(EPSG, lon_c, lat_c)
    expected = 250.0 + SLOPE * (northing - py)
    with netCDF4.Dataset(out) as ds:
        ter = np.asarray(ds["ter"][:], dtype=float)
        attrs = {k: ds.getncattr(k) for k in ds.ncattrs()}
    err = np.abs(ter[selected] - expected[selected])
    # A 15 m staircase of a 1.8 % slope (0.27 m per pixel), averaged over
    # 100 m cells.
    assert np.median(err) < 0.2 and err.max() < 1.0
    assert receipt["terrain"]["source"] == "gem15-dtm"
    sources = receipt["sources"]
    assert sources["terrain_vertical_datum"].startswith("EGM96")
    assert sources["terrain_source_selection"]["used"] == "gem15-dtm"
    assert attrs["highres_terrain_source"] == "gem15-dtm"
    assert attrs["highres_terrain_attribution"].startswith(
        "This dataset is produced by Google")
    assert json.loads(attrs["highres_terrain_source_selection"])[
        "experimental"] is True
    assert "EXPERIMENTAL" in capsys.readouterr().err
