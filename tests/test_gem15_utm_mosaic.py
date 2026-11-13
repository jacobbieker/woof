"""GEM-15 mosaic: per-cell UTM windows onto one geographic lattice.

Synthetic windows only: two cells either side of the 6 E zone boundary
(UTM 31N and 32N), written with :func:`woof.static.cog_range.
write_tiled_geotiff`.  The Rust ``static-fields`` body (the default) is
held to an independent numpy reference (each lattice centre projected with
:mod:`woof.static.utm`, the containing source pixel taken) and to the
rasterio/GDAL parity body.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from woof.static import cog_range, gem15, utm
from woof.static import highres_fetch as hf
from woof.static.highres_fetch import FootprintBBox

RES = hf.GEM15_LAT_STEP_DEG
BBOX = FootprintBBox(46.0, 46.08, 5.94, 6.06)
BLOCK = 512


def _window(tmp_path: Path, epsg: int, keep, name: str):
    """A 3 x 3-block 15 m window in ``epsg`` covering BBOX; pixels whose
    centre fails ``keep(lon, lat)`` are NaN (outside the cell), plus a
    producer hole."""
    lon = np.linspace(BBOX.lon_min - 0.02, BBOX.lon_max + 0.02, 9)
    lat = np.linspace(BBOX.lat_min - 0.02, BBOX.lat_max + 0.02, 9)
    x, y = utm.forward(epsg, *np.meshgrid(lon, lat))
    x0 = np.floor(x.min() / 15.0) * 15.0
    y0 = np.ceil(y.max() / 15.0) * 15.0
    n = 3 * BLOCK
    assert x0 + 15.0 * n > x.max() and y0 - 15.0 * n < y.min()
    rows, cols = np.mgrid[0:n, 0:n]
    cx = x0 + 15.0 * (cols + 0.5)
    cy = y0 - 15.0 * (rows + 0.5)
    clon, clat = utm.inverse(epsg, cx, cy)
    values = (500.0 + (epsg % 100) * 1000.0 + rows * 0.25 + cols * 0.5
              ).astype(np.float32)
    values[~keep(clon, clat)] = np.nan
    values[200:230, 300:340] = np.nan
    tiles = (values[r * BLOCK:(r + 1) * BLOCK, c * BLOCK:(c + 1) * BLOCK]
             for r in range(3) for c in range(3))
    path = tmp_path / f"{name}.tif"
    cog_range.write_tiled_geotiff(path, tiles, rows=3, cols=3, block=BLOCK,
                                  transform=(15.0, 0.0, x0, 0.0, -15.0, y0),
                                  epsg=epsg)
    return hf.record_local_artifact(path, url=f"test:{name}"), (x0, y0, values)


@pytest.fixture()
def windows(tmp_path):
    west, west_grid = _window(tmp_path, 32631, lambda lon, lat: lon < 6.0,
                              "cell_w")
    east, east_grid = _window(tmp_path, 32632, lambda lon, lat: lon >= 6.0,
                              "cell_e")
    return [west, east], {32631: west_grid, 32632: east_grid}


def _reference(grids, shape, transform):
    """Independent nearest reference: first writer wins, west then east."""
    ny, nx = shape
    cols, rows = np.meshgrid(np.arange(nx), np.arange(ny))
    lon = transform.c + (cols + 0.5) * transform.a
    lat = transform.f + (rows + 0.5) * transform.e
    out = np.full(shape, np.nan, dtype=np.float32)
    for epsg in (32631, 32632):
        x0, y0, values = grids[epsg]
        x, y = utm.forward(epsg, lon, lat)
        sc = np.floor((x - x0) / 15.0).astype(np.int64)
        sr = np.floor((y0 - y) / 15.0).astype(np.int64)
        inside = ((sc >= 0) & (sr >= 0) & (sc < values.shape[1])
                  & (sr < values.shape[0]))
        picked = np.full(shape, np.nan, dtype=np.float32)
        picked[inside] = values[sr[inside], sc[inside]]
        empty = np.isnan(out)
        out[empty] = picked[empty]
    return out


def _read(path):
    rasterio = pytest.importorskip("rasterio")
    with rasterio.open(path) as src:
        return src.read(1), src.transform, src.crs.to_epsg()


def _require_rust():
    from woof.static import rust_bridge

    reason = rust_bridge.unavailable_reason()
    if reason is not None:
        pytest.fail(f"the Rust static-fields bridge is not loadable: {reason}")


def test_rust_mosaic_matches_independent_reference(windows, tmp_path,
                                                   monkeypatch):
    _require_rust()
    monkeypatch.delenv("WOOF_STATIC_PYTHON", raising=False)
    fetched, grids = windows
    window, audit = gem15.derive_gem15_window(fetched, BBOX, tmp_path)
    values, transform, epsg = _read(window.path)
    assert epsg == 4326
    # The declared lattice: pixel centres on whole multiples of RES.
    assert transform.a == pytest.approx(RES) and transform.e == pytest.approx(-RES)
    centre = (transform.c + 0.5 * RES) / RES
    assert centre == pytest.approx(round(centre), abs=1e-6)
    expected = _reference(grids, values.shape, transform)
    same = (values == expected) | (np.isnan(values) & np.isnan(expected))
    assert same.mean() > 0.9999, same.mean()
    # Both zones reached the mosaic, either side of 6 E.
    assert np.nanmin(values) < 32000.0 < np.nanmax(values)
    assert audit["output_shape"] == list(values.shape)
    assert audit["no_data_pixels_outside_coverage"] == int(np.isnan(values).sum())
    assert audit["vertical_datum"] == hf.GEM15_VERTICAL_DATUM
    assert audit["vertical_datum"].startswith("EGM96")


def test_rust_matches_rasterio_parity_body(windows, tmp_path, monkeypatch):
    _require_rust()
    pytest.importorskip("rasterio")
    fetched, _ = windows
    monkeypatch.delenv("WOOF_STATIC_PYTHON", raising=False)
    rust, _ = gem15.derive_gem15_window(fetched, BBOX, tmp_path / "rust")
    monkeypatch.setenv("WOOF_STATIC_PYTHON", "1")
    ref, ref_audit = gem15.derive_gem15_window(fetched, BBOX,
                                               tmp_path / "python")
    a, at, _ = _read(rust.path)
    b, bt, _ = _read(ref.path)
    assert a.shape == b.shape and at == bt
    both = np.isfinite(a) & np.isfinite(b)
    # GDAL's nearest picks the same source pixel except at ties on a
    # source pixel edge; the measured agreement is far above this floor.
    assert both.mean() > 0.95
    assert (a[both] == b[both]).mean() > 0.999
    assert np.mean(np.isnan(a) != np.isnan(b)) < 0.001
    assert ref_audit["output_shape"] == list(b.shape)


def test_window_is_cached(windows, tmp_path):
    _require_rust()
    fetched, _ = windows
    first, _ = gem15.derive_gem15_window(fetched, BBOX, tmp_path)
    again, audit = gem15.derive_gem15_window(fetched, BBOX, tmp_path)
    assert again.cache_hit and again.sha256 == first.sha256
    assert audit["resampling"].startswith("nearest")


def test_first_window_wins_where_cells_overlap(windows, tmp_path):
    _require_rust()
    fetched, grids = windows
    # East first: east's pixels (lon >= 6) win; west fills the rest.
    window, _ = gem15.derive_gem15_window(fetched[::-1], BBOX, tmp_path)
    values, transform, _ = _read(window.path)
    lon = transform.c + (np.arange(values.shape[1]) + 0.5) * transform.a
    east = values[:, lon > 6.001]
    west = values[:, lon < 5.999]
    assert np.nanmin(east) > 32000.0 and np.nanmax(west) < 32000.0


def test_dateline_window_is_refused(windows, tmp_path):
    from woof.static.highres_refusal import HighresRefusal

    fetched, _ = windows
    with pytest.raises(HighresRefusal):
        gem15.derive_gem15_window(
            fetched, FootprintBBox(10.0, 10.1, 179.95, 180.05), tmp_path)
