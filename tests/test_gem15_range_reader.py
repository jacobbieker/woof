"""GEM-15 range reader: COG head parsing, verified block fetch, block cache,
UTM windows.

Offline and deterministic: a synthetic BigTIFF COG (written here, not by
the module under test) is served by a local HTTP server that honours byte
ranges and sends the ``x-goog-hash`` / ``x-goog-generation`` headers GCS
sends.  No real GEM-15 bytes are used or committed.
"""
from __future__ import annotations

import base64
import http.server
import struct
import threading
import zlib
from pathlib import Path

import numpy as np
import pytest

from woof.static import cog_range, gem15, utm
from woof.static import highres_fetch as hf
from woof.static.highres_fetch import FootprintBBox
from woof.static.highres_refusal import HighresRefusal

EPSG = 32614
BLOCK = 512
NBLOCKS = 4                      # 4 x 4 blocks of 512 -> 2048 x 2048
SIZE = BLOCK * NBLOCKS
X0, Y0 = 422400.0, 4500480.0     # upper-left corner, metres (inside 40-41 N, 100-99 W)
SPARSE = (3, 0)                  # a block the producer left out


def _values(spp: int) -> np.ndarray:
    rows, cols = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    bands = [1000.0 + rows * 0.5 + cols * 0.25 + 1000.0 * k
             for k in range(spp)]
    data = np.stack(bands, axis=-1).astype("<f4")
    data[100:140, 700:760, :] = np.nan       # a producer hole (water)
    r, c = SPARSE
    data[r * BLOCK:(r + 1) * BLOCK, c * BLOCK:(c + 1) * BLOCK, :] = np.nan
    return data


def _bigtiff_cog(data: np.ndarray) -> bytes:
    """A minimal BigTIFF COG: head first, then Deflate 512 blocks."""
    spp = data.shape[2]
    n = NBLOCKS * NBLOCKS
    payloads = []
    for r in range(NBLOCKS):
        for c in range(NBLOCKS):
            if (r, c) == SPARSE:
                payloads.append(b"")
                continue
            tile = data[r * BLOCK:(r + 1) * BLOCK, c * BLOCK:(c + 1) * BLOCK]
            payloads.append(zlib.compress(np.ascontiguousarray(tile).tobytes()))
    geokeys = (1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, EPSG)
    entries = [(256, 3, [SIZE]), (257, 3, [SIZE]), (258, 3, [32] * spp),
               (259, 3, [8]), (262, 3, [1]), (277, 3, [spp]), (284, 3, [1]),
               (317, 3, [1]), (322, 3, [BLOCK]), (323, 3, [BLOCK]),
               (324, 16, [0] * n), (325, 16, [0] * n), (339, 3, [3] * spp),
               (33550, 12, [15.0, 15.0, 0.0]),
               (33922, 12, [0.0, 0.0, 0.0, X0, Y0, 0.0]),
               (34735, 3, list(geokeys)), (42113, 2, b"nan\0")]
    fmt = {3: "H", 16: "Q", 12: "d"}
    ifd_at = 16
    data_at = ifd_at + 8 + 20 * len(entries) + 8
    blobs, cursor = {}, data_at
    for tag, kind, values in entries:
        size = len(values) * {3: 2, 16: 8, 12: 8, 2: 1}[kind]
        if size > 8:
            blobs[tag] = cursor
            cursor += size
    offsets, counts, at = [], [], cursor
    for payload in payloads:
        offsets.append(at if payload else 0)
        counts.append(len(payload))
        at += len(payload)
    values_of = {324: offsets, 325: counts}
    out = bytearray(b"II" + struct.pack("<HHHQ", 43, 8, 0, ifd_at))
    out += struct.pack("<Q", len(entries))
    tail = bytearray()
    for tag, kind, values in entries:
        values = values_of.get(tag, values)
        raw = values if kind == 2 else struct.pack(
            f"<{len(values)}{fmt[kind]}", *values)
        if tag in blobs:
            out += struct.pack("<HHQQ", tag, kind, len(values), blobs[tag])
            tail += raw
        else:
            out += struct.pack("<HHQ", tag, kind, len(values)) + raw.ljust(
                8, b"\0")
    out += struct.pack("<Q", 0)
    assert len(out) == data_at
    out += tail
    for payload in payloads:
        out += payload
    return bytes(out)


class _Bucket(http.server.ThreadingHTTPServer):
    daemon_threads = True


def _handler(objects, log, faults):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            body, crc, md5, generation = objects[self.path]
            log.append(self.path)
            if faults:
                code = faults.pop(0)
                self.send_response(code)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            start, end = (int(v) for v in self.headers["Range"].split(
                "=")[1].split("-"))
            chunk = body[start:end + 1]
            self.send_response(206)
            self.send_header("Content-Range",
                             f"bytes {start}-{end}/{len(body)}")
            self.send_header("Content-Length", str(len(chunk)))
            self.send_header("x-goog-hash", f"crc32c={crc}")
            if md5:
                self.send_header("x-goog-hash", f"md5={md5}")
            self.send_header("x-goog-generation", generation)
            self.end_headers()
            self.wfile.write(chunk)
    return Handler


CRC = base64.b64encode(b"\x01\x02\x03\x04").decode()
MD5 = base64.b64encode(bytes(range(16))).decode()


@pytest.fixture()
def bucket():
    dtm, dsm = _values(1), _values(5)
    objects = {
        "/p40_m100/wm100s40/dtm/zall/mosaic.tif":
            (_bigtiff_cog(dtm), CRC, MD5, "17"),
        "/p40_m100/wm100s40/dsm/zall/mosaic.tif":
            (_bigtiff_cog(dsm), CRC, None, "18"),
    }
    log, faults = [], []
    server = _Bucket(("127.0.0.1", 0), _handler(objects, log, faults))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def obj(asset, crc=CRC, md5=MD5, generation=None):
        body = objects[f"/p40_m100/wm100s40/{asset}/zall/mosaic.tif"][0]
        return gem15.Gem15Object(
            url=f"{base}/p40_m100/wm100s40/{asset}/zall/mosaic.tif",
            bytes=len(body), md5_base64=(md5 if asset == "dtm" else None),
            crc32c_base64=crc,
            generation=generation or {"dtm": "17", "dsm": "18"}[asset],
            shape=(SIZE, SIZE),
            transform=(15.0, 0.0, X0, 0.0, -15.0, Y0))

    def index(**overrides):
        cell = gem15.Gem15Cell(
            cell_id="wm100s40", group="p40_m100", lat_sw=40, lon_sw=-100,
            epsg=EPSG, objects={"dtm": obj("dtm", **overrides),
                                "dsm": obj("dsm")})
        return gem15.Gem15Index(snapshot_utc="2026-11-13T00:00:00Z",
                                bucket_url=base, cells={(40, -100): cell})

    yield {"index": index, "log": log, "faults": faults,
           "dtm": dtm, "dsm": dsm}
    server.shutdown()
    server.server_close()


def _footprint(col0, col1, row0, row1) -> FootprintBBox:
    """Geographic box of a pixel rectangle of the synthetic cell."""
    cols = np.array([col0, col1, col0, col1], dtype=float)
    rows = np.array([row0, row0, row1, row1], dtype=float)
    lon, lat = utm.inverse(EPSG, X0 + 15.0 * cols, Y0 - 15.0 * rows)
    return FootprintBBox(float(lat.min()), float(lat.max()),
                         float(lon.min()), float(lon.max()))


def _read_window(path):
    rasterio = pytest.importorskip("rasterio")
    with rasterio.open(path) as src:
        return src.read(1), src.transform, src.crs.to_epsg()


# --------------------------------------------------------------------------
# Head parsing
# --------------------------------------------------------------------------

def test_parse_head_of_bigtiff_cog():
    body = _bigtiff_cog(_values(5))
    layout = cog_range.parse_head(body[:cog_range.HEAD_BYTES])
    assert (layout.width, layout.height, layout.block_w) == (SIZE, SIZE, BLOCK)
    assert layout.samples_per_pixel == 5 and layout.epsg == EPSG
    assert layout.transform == (15.0, 0.0, X0, 0.0, -15.0, Y0)
    assert layout.nodata == "nan"
    assert layout.block(*SPARSE)[1] == 0
    with pytest.raises(cog_range.CogHeadTooShort) as short:
        cog_range.parse_head(body[:200])
    assert short.value.needed > 200


def test_parse_head_refuses_what_it_cannot_read():
    body = bytearray(_bigtiff_cog(_values(1)))
    with pytest.raises(cog_range.CogFormatError):
        cog_range.parse_head(b"MM" + bytes(body[2:]))
    # Predictor 3 (floating point) is not read: patch tag 317's value.
    at = bytes(body).index(struct.pack("<HHQ", 317, 3, 1))
    body[at + 12:at + 14] = struct.pack("<H", 3)
    with pytest.raises(cog_range.CogFormatError, match="predictor"):
        cog_range.parse_head(bytes(body))


def test_decode_block_selects_band_and_fills_sparse():
    data = _values(5)
    body = _bigtiff_cog(data)
    layout = cog_range.parse_head(body)
    offset, count = layout.block(1, 2)
    block = cog_range.decode_block(layout, body[offset:offset + count], 3)
    np.testing.assert_array_equal(block, data[512:1024, 1024:1536, 3])
    assert np.isnan(cog_range.decode_block(layout, b"", 0)).all()


def test_writer_round_trips_through_rasterio(tmp_path):
    tiles = [np.full((BLOCK, BLOCK), float(k), np.float32) for k in range(6)]
    out = tmp_path / "w.tif"
    cog_range.write_tiled_geotiff(out, iter(tiles), rows=2, cols=3,
                                  block=BLOCK, transform=(15.0, 0, 1000.0, 0,
                                                          -15.0, 9000.0),
                                  epsg=32733)
    values, transform, epsg = _read_window(out)
    assert values.shape == (2 * BLOCK, 3 * BLOCK) and epsg == 32733
    assert values[0, 0] == 0.0 and values[-1, -1] == 5.0
    assert (transform.a, transform.c, transform.e, transform.f) == (
        15.0, 1000.0, -15.0, 9000.0)
    layout = cog_range.parse_head(out.read_bytes())
    assert layout.epsg == 32733 and layout.samples_per_pixel == 1


# --------------------------------------------------------------------------
# UTM
# --------------------------------------------------------------------------

def test_utm_matches_pyproj():
    pyproj = pytest.importorskip("pyproj")
    rng = np.random.default_rng(1)
    for epsg, lon0, lats in ((32614, -99.0, (0.0, 84.0)),
                             (32719, -69.0, (-61.0, 0.0)),
                             (32601, -177.0, (0.0, 80.0))):
        lon = lon0 + rng.uniform(-7.0, 7.0, 4000)
        lat = rng.uniform(*lats, 4000)
        x, y = pyproj.Transformer.from_crs(
            4326, epsg, always_xy=True).transform(lon, lat)
        ux, uy = utm.forward(epsg, lon, lat)
        assert np.max(np.abs(ux - x)) < 1e-6 and np.max(np.abs(uy - y)) < 1e-6
        back_lon, back_lat = utm.inverse(epsg, ux, uy)
        assert np.max(np.abs(back_lat - lat)) < 1e-10
        assert np.max(np.abs((back_lon - lon + 180) % 360 - 180)) < 1e-10
    with pytest.raises(ValueError):
        utm.zone_of_epsg(4326)


def test_window_blocks():
    layout = cog_range.parse_head(_bigtiff_cog(_values(1)))
    blocks = gem15.window_blocks(layout, EPSG, _footprint(600, 1300, 300, 900))
    assert blocks == (0, 1, 1, 2)
    far = FootprintBBox(40.0, 40.1, -95.0, -94.9)
    assert gem15.window_blocks(layout, EPSG, far) is None


# --------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------

def test_fetch_reads_only_blocks_under_the_footprint(bucket, tmp_path):
    bbox = _footprint(600, 1300, 300, 900)
    windows, manifest = gem15.fetch_gem15_windows(
        bbox, tmp_path, "gem15-dtm", index=bucket["index"]())
    assert len(windows) == 1
    # One head read plus the four blocks rows 0-1 x cols 1-2.
    assert len(bucket["log"]) == 5
    values, transform, epsg = _read_window(windows[0].path)
    assert epsg == EPSG
    np.testing.assert_array_equal(values, bucket["dtm"][0:1024, 512:1536, 0])
    assert (transform.c, transform.f) == (X0 + 512 * 15.0, Y0)
    cell = manifest["gem15_cells"][0]
    assert cell["blocks"] == {"rows": [0, 1], "cols": [1, 2], "read": 4,
                              "sha256": cell["blocks"]["sha256"]}
    assert cell["object"]["crc32c"] == "01020304"
    assert manifest["terrain_bytes_fetched"] > 0
    assert manifest["gem15_unpublished_squares"] == [] or all(
        sq != [40, -100] for sq in manifest["gem15_unpublished_squares"])


def test_dsm_reads_band_zero_of_five(bucket, tmp_path):
    bbox = _footprint(600, 700, 300, 400)
    windows, manifest = gem15.fetch_gem15_windows(
        bbox, tmp_path, "gem15-dsm", index=bucket["index"]())
    values, _, _ = _read_window(windows[0].path)
    np.testing.assert_array_equal(values, bucket["dsm"][0:512, 512:1024, 0])
    assert manifest["gem15_band"] == "dsm band 0"


def test_sparse_blocks_are_holes_and_not_requested(bucket, tmp_path):
    bbox = _footprint(10, 900, 1300, 1900)    # block rows 2-3, cols 0-1
    windows, _ = gem15.fetch_gem15_windows(
        bbox, tmp_path, "gem15-dtm", index=bucket["index"]())
    values, _, _ = _read_window(windows[0].path)
    assert np.isnan(values[512:, :512]).all()              # the sparse block
    np.testing.assert_array_equal(values[:512, :],
                                  bucket["dtm"][1024:1536, 0:1024, 0])
    assert len(bucket["log"]) == 1 + 3                      # head + 3 blocks


def test_cache_is_reused_and_verified(bucket, tmp_path):
    bbox = _footprint(600, 1300, 300, 900)
    index = bucket["index"]()
    first, _ = gem15.fetch_gem15_windows(bbox, tmp_path, "gem15-dtm",
                                         index=index)
    requests = len(bucket["log"])
    again, manifest = gem15.fetch_gem15_windows(bbox, tmp_path, "gem15-dtm",
                                                index=index)
    assert len(bucket["log"]) == requests          # nothing fetched again
    assert again[0].cache_hit and again[0].sha256 == first[0].sha256
    assert manifest["terrain_bytes_fetched"] == 0
    # A cached block whose bytes no longer match its recorded SHA-256 is
    # fetched again, never trusted.
    block = next((tmp_path / "gem15").rglob("block_r000_c001.bin"))
    block.write_bytes(b"corrupt" + block.read_bytes()[7:])
    gem15.fetch_gem15_windows(bbox, tmp_path, "gem15-dtm", index=index)
    assert len(bucket["log"]) == requests + 1


def test_offline(bucket, tmp_path):
    bbox = _footprint(600, 1300, 300, 900)
    index = bucket["index"]()
    with pytest.raises(hf.HighresFetchRefusal, match="offline"):
        gem15.fetch_gem15_windows(bbox, tmp_path, "gem15-dtm", index=index,
                                  offline=True)
    assert bucket["log"] == []
    gem15.fetch_gem15_windows(bbox, tmp_path, "gem15-dtm", index=index)
    requests = len(bucket["log"])
    windows, _ = gem15.fetch_gem15_windows(bbox, tmp_path, "gem15-dtm",
                                           index=index, offline=True)
    assert len(bucket["log"]) == requests and windows[0].cache_hit


@pytest.mark.parametrize("override,named", [
    ({"crc": base64.b64encode(b"\xff\xff\xff\xff").decode()}, "CRC32C"),
    ({"md5": base64.b64encode(b"\x00" * 16).decode()}, "MD5"),
    ({"generation": "99"}, "generation"),
])
def test_a_republished_object_is_refused(bucket, tmp_path, override, named):
    with pytest.raises(HighresRefusal, match=named) as refused:
        gem15.fetch_gem15_windows(_footprint(600, 700, 300, 400), tmp_path,
                                  "gem15-dtm", index=bucket["index"](**override))
    assert refused.value.reason == "gem15-object-changed"


def test_transient_fault_is_asked_again(bucket, tmp_path, monkeypatch):
    waits = []
    monkeypatch.setattr(hf, "_sleep", waits.append)
    bucket["faults"].extend([503])
    windows, _ = gem15.fetch_gem15_windows(
        _footprint(600, 700, 300, 400), tmp_path, "gem15-dtm",
        index=bucket["index"]())
    assert waits and len(windows) == 1


def test_missing_object_is_absent_not_retried(bucket, tmp_path):
    bucket["faults"].extend([404])
    with pytest.raises(hf.SourceAbsent):
        gem15.fetch_gem15_windows(_footprint(600, 700, 300, 400), tmp_path,
                                  "gem15-dtm", index=bucket["index"]())


def test_footprint_without_published_cell_fetches_nothing(bucket, tmp_path):
    windows, manifest = gem15.fetch_gem15_windows(
        FootprintBBox(10.2, 10.4, -30.4, -30.2), tmp_path, "gem15-dtm",
        index=bucket["index"]())
    assert windows == () and bucket["log"] == []
    assert manifest["gem15_unpublished_squares"] == [[10, -31]]
