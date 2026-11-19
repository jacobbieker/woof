"""SatWeave reader: store layouts, subsetting, cache, NaN rules and the geometry reconstruction.

Every test reads a local directory written in the published layout (Zarr v3,
consolidated metadata, one (1, 1801, 3600) bytes+zstd chunk per band per
step), so nothing here touches the network or SatWeave data.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

zstd = pytest.importorskip("compression.zstd")

from woof.obs import satweave as sw  # noqa: E402
from woof.obs.imager_scene import (FLAG_GEOMETRY_GUESSED, FLAG_LIMB, FLAG_NIGHT,  # noqa: E402
                                   FLAG_OVERLAP, ImagerScene, band, geostationary_view)

UTC = timezone.utc
LAT = np.round(90.0 - 0.1 * np.arange(1801), 1)
LON = np.round(0.1 * np.arange(3600), 1)


def _codecs():
    return [{"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "zstd", "configuration": {"level": 0, "checksum": False}}]


def _array(shape, chunks, dtype, attrs, fill="NaN"):
    return {"shape": list(shape), "data_type": dtype, "zarr_format": 3, "node_type": "array",
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": list(chunks)}},
            "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
            "fill_value": fill, "codecs": _codecs(), "attributes": attrs,
            "dimension_names": None, "storage_transformers": []}


def _put(root, key, array):
    path = root / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(zstd.compress(np.ascontiguousarray(array).astype(array.dtype.newbyteorder("<")).tobytes()))


def _fields(lat2, lon2):
    """Smooth synthetic bands: 6900 only over the GOES/Himawari longitudes."""

    lon180 = ((lon2 + 180.0) % 360.0) - 180.0
    disc = np.abs(lat2) < 70.0
    v62 = np.where(disc, 230.0 + 5.0 * np.cos(np.radians(lat2)), np.nan).astype(np.float32)
    meteo = (lon180 > -30.0) & (lon180 < 75.0)
    v69 = np.where(disc & ~meteo, 240.0 + 0.01 * lon180, np.nan).astype(np.float32)
    v112 = np.where(disc, 280.0, np.nan).astype(np.float32)
    v112[0:5, 0:5] = 400.0  # outside the published valid range, but off-disc rows anyway
    v112[900, 100] = 120.0  # below 150 K: dropped and counted
    return {6200: v62, 6900: v69, 11200: v112}


def _write_store(root, path, *, n_times, index, units, value, sources=None):
    lat2 = np.broadcast_to(LAT[:, None], (1801, 3600))
    lon2 = np.broadcast_to(LON[None, :], (1801, 3600))
    fields = _fields(lat2, lon2)
    meta = {}
    time_chunk = 43836 if n_times > 1 else 1
    meta["time"] = _array((n_times,), (time_chunk,), "int64", {"units": units}, fill=0)
    meta["latitude"] = _array((1801,), (1801,), "float64", {"units": "degrees_north"})
    meta["longitude"] = _array((3600,), (3600,), "float64", {"units": "degrees_east"})
    for nm, data in fields.items():
        attrs = {"units": "K", "valid_range": [150.0, 350.0],
                 "satellite_sources": sources or "GOES-East (ABI), Meteosat Prime & IODC (SEVIRI)"}
        meta[band(nm).name] = _array((n_times, 1801, 3600), (1, 1801, 3600), "float32", attrs)
        _put(root, f"{path}/{band(nm).name}/c/{index}/0/0", data)
    block = index // time_chunk
    length = min(time_chunk, n_times - block * time_chunk)
    times = np.zeros(length, np.int64)
    times[index - block * time_chunk] = value
    _put(root, f"{path}/time/c/{block}", times)
    _put(root, f"{path}/latitude/c/0", LAT.astype(np.float64))
    _put(root, f"{path}/longitude/c/0", LON.astype(np.float64))
    doc = {"zarr_format": 3, "node_type": "group", "attributes": {"title": "test mosaic"},
           "consolidated_metadata": {"kind": "inline", "must_understand": False, "metadata": meta}}
    (root / path).mkdir(parents=True, exist_ok=True)
    (root / path / "zarr.json").write_text(json.dumps(doc))


@pytest.fixture(scope="module")
def archive_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("satweave")
    when = datetime(2024, 7, 1, 12, 30, tzinfo=UTC)
    index = int((when - sw.ARCHIVE_START) / sw.ARCHIVE_STEP)
    _write_store(root, sw.ARCHIVE_PATH, n_times=175344, index=index,
                 units="minutes since 2016-01-01 00:00:00", value=index * 30)
    hour = datetime(2026, 11, 18, 12, tzinfo=UTC)
    _write_store(root, sw.HOURLY_TEMPLATE.format(stamp="20261118T120000"), n_times=1, index=0,
                 units="days since 2026-11-18 12:00:00", value=0)
    return root, when, hour


def test_store_paths_and_refusals():
    path, index = sw.store_for(datetime(2016, 1, 1, 0, 30, tzinfo=UTC))
    assert (path, index) == (sw.ARCHIVE_PATH, 1)
    assert sw.store_for("2026-03-04T05:00Z") == ("zarr/2026_to_present/20260304T050000/mosaic.zarr", None)
    for bad in ("2026-03-04T05:30Z", "2020-01-01T00:10Z", "2015-12-31T23:30Z"):
        with pytest.raises(sw.SatWeaveError):
            sw.store_for(bad)
    assert sw.nearest_step("2020-01-01T00:14Z") == datetime(2020, 1, 1, tzinfo=UTC)
    assert sw.nearest_step("2026-02-01T10:31Z") == datetime(2026, 2, 1, 11, tzinfo=UTC)


def test_cf_time_units():
    out = sw.decode_cf_times([0, 90], "minutes since 2016-01-01 00:00:00")
    assert out == [datetime(2016, 1, 1, tzinfo=UTC), datetime(2016, 1, 1, 1, 30, tzinfo=UTC)]
    assert sw.decode_cf_times([0.5], "days since 2026-11-18 12:00:00") == [datetime(2026, 11, 19, tzinfo=UTC)]
    with pytest.raises(sw.SatWeaveError):
        sw.decode_cf_times([0], "fortnights after lunch")


def test_archive_subset_bbox_valid_range_and_cache(archive_root, tmp_path):
    root, when, _ = archive_root
    subset = sw.read_subset(when, bands=["mosaic_11200_nm", 6200], bbox=(5.0, -5.0, 15.0, 5.0),
                            root=str(root), cache=tmp_path)
    assert subset.lat[0] == 5.0 and subset.lat[-1] == -5.0
    assert subset.lon[0] == pytest.approx(5.0) and subset.lon[-1] == pytest.approx(15.0)
    assert subset.values[11200].shape == (101, 101)
    assert subset.values[11200][10, 10] == 280.0
    assert np.isnan(subset.values[11200][50, 50])  # row 900 col 100 (lat 0, lon 10): 120 K dropped
    assert subset.provenance["dropped_outside_valid_range"]["11200"] == 1
    assert subset.provenance["licence"] == "CC-BY-4.0" and "Google DeepMind" in subset.provenance["attribution"]
    again = sw.read_subset(when, bands=[6200, 11200], bbox=(5.0, -5.0, 15.0, 5.0), root="/nonexistent",
                           cache=tmp_path)
    assert again.provenance.get("cached") is True
    np.testing.assert_array_equal(again.values[6200], subset.values[6200])
    # A tampered cache is refused, not read.
    nc = next(tmp_path.rglob("satweave.nc"))
    nc.write_bytes(nc.read_bytes() + b"x")
    with pytest.raises(sw.SatWeaveError, match="digest"):
        sw.read_subset(when, bands=[6200, 11200], bbox=(5.0, -5.0, 15.0, 5.0), root=str(root), cache=tmp_path)


def test_antimeridian_and_prime_meridian_boxes(archive_root):
    root, when, _ = archive_root
    across = sw.read_subset(when, bands=[6200], bbox=(170.0, 0.0, -170.0, 1.0), root=str(root), use_cache=False)
    assert across.values[6200].shape == (11, 201)
    assert np.all(np.diff(across.lon) > 0) and across.lon[0] == pytest.approx(170.0)
    zero = sw.read_subset(when, bands=[6200], bbox=(-1.0, 0.0, 1.0, 1.0), root=str(root), use_cache=False)
    assert zero.lon[0] == pytest.approx(-1.0) and zero.lon[-1] == pytest.approx(1.0)
    assert zero.values[6200].shape == (11, 21)


def test_hourly_store_and_missing_hour(archive_root):
    root, _, hour = archive_root
    subset = sw.read_subset(hour, bands=[6900], bbox=(-100.0, 30.0, -90.0, 40.0), root=str(root), use_cache=False)
    assert np.isfinite(subset.values[6900]).all()
    with pytest.raises(sw.SatWeaveError, match="no SatWeave store"):
        sw.read_subset(hour + timedelta(hours=1), bands=[6900], root=str(root), use_cache=False)
    with pytest.raises(sw.SatWeaveError, match="not"):
        sw.read_subset(datetime(2024, 7, 1, 13, tzinfo=UTC), bands=[6200], root=str(root), use_cache=False)


def test_reconstruction_attributes_sectors(archive_root):
    root, when, _ = archive_root
    scene = sw.read_scene(when, bands=[11200], bbox=(-180.0, -60.0, 179.9, 60.0),
                          root=str(root), use_cache=False)
    assert isinstance(scene, ImagerScene) and set(scene.values) == {11200}
    names = [p.name for p in scene.platforms]
    assert names == ["GOES-16", "GOES-18", "Himawari-9", "Meteosat-11", "Meteosat-9"]

    def at(lat, lon):
        j = int(np.argmin(np.abs(scene.lat[:, 0] - lat)))
        i = int(np.argmin(np.abs(((scene.lon[0] - lon + 180) % 360) - 180)))
        return j, i

    for (lat, lon), name in (((30.0, -80.0), "GOES-16"), ((20.0, -150.0), "GOES-18"),
                             ((0.0, 140.0), "Himawari-9"), ((10.0, 5.0), "Meteosat-11"),
                             ((0.0, 50.0), "Meteosat-9")):
        j, i = at(lat, lon)
        assert names[scene.platform_index[j, i]] == name, (lat, lon)
    # 6900 present over 60E-75E, west of Himawari's sub-point: availability says
    # not Meteosat, so the pixel goes to Himawari despite IODC's smaller zenith.
    j, i = at(0.0, 80.0)
    assert names[scene.platform_index[j, i]] == "Himawari-9"
    # Between GOES-East and GOES-West both see the pixel well: overlap.
    j, i = at(10.0, -106.0)
    assert scene.flags[j, i] & FLAG_OVERLAP
    assert np.all(scene.flags & FLAG_GEOMETRY_GUESSED)
    j, i = at(59.9, -75.2)
    assert scene.flags[j, i] & FLAG_LIMB
    # 12:30 UTC: night at 150 W, day at 0 E.
    assert scene.flags[at(20.0, -150.0)] & FLAG_NIGHT and not scene.flags[at(10.0, 5.0)] & FLAG_NIGHT
    assert scene.view_zenith[at(0.0, -75.2)] < 1.0


def test_view_geometry_closed_form():
    zen, az, scan = geostationary_view(np.array([0.0, 0.0, 45.0]), np.array([0.0, 30.0, 0.0]), 0.0)
    assert zen[0] == pytest.approx(0.0, abs=1e-6) and scan[0] == pytest.approx(0.0, abs=1e-6)
    # Due east of the sub-point the satellite is to the west.
    assert az[1] == pytest.approx(270.0, abs=1e-6)
    # Due north: the satellite is to the south, and the scan angle is the
    # nadir angle sin(scan) = (R/r) sin(zenith) for a sphere, close on WGS84.
    assert az[2] == pytest.approx(180.0, abs=1e-6)
    ratio = np.sin(np.radians(scan[2])) / np.sin(np.radians(zen[2]))
    assert ratio == pytest.approx(6371.0 / 42164.16, rel=5e-3)


def test_refusals_before_the_era_and_for_fci():
    with pytest.raises(sw.SatWeaveError, match="legacy"):
        sw.platforms_at("2017-06-01T00:00Z")
    subset = sw.SatWeaveSubset(datetime(2026, 1, 1, tzinfo=UTC), LAT[:3], LON[:3], {},
                               {"band_attributes": {"6200": {"satellite_sources": "Meteosat (FCI WV6.3)"}}})
    with pytest.raises(sw.SatWeaveError, match="FCI"):
        sw.reconstruct(subset)
    assert [p.crtm_sensor for p in sw.platforms_at("2025-06-01T00:00Z")][0] == "abi_g16"
    assert "GOES-19" in [p.name for p in sw.platforms_at("2025-06-01T00:00Z")]


def test_fetch_alias_and_door(archive_root, tmp_path, capsys):
    from woof.cli import _rewrite_version_alias, main

    assert _rewrite_version_alias(["fetch", "satweave", "--time", "x"]) == ["obs", "satweave", "fetch", "--time", "x"]
    assert _rewrite_version_alias(["fetch", "--source", "gfs"]) == ["fetch", "--source", "gfs"]
    root, when, _ = archive_root
    summary = tmp_path / "summary.json"
    code = main(["fetch", "satweave", "--time", when.strftime("%Y-%m-%dT%H:%MZ"), "--bands", "6200,11200",
                 "--bbox", "-100,30,-90,40", "--root", str(root), "--cache", str(tmp_path / "cache"),
                 "--summary-json", str(summary)])
    assert code == 0
    doc = json.loads(summary.read_text())
    assert doc[0]["pixels_by_platform"]["GOES-16"] == 101 * 101
    assert "CC BY 4.0" in capsys.readouterr().err
    assert main(["obs", "satweave", "fetch", "--time", "2026-03-04T05:30Z", "--root", str(root),
                 "--cache", str(tmp_path / "cache")]) == 2
