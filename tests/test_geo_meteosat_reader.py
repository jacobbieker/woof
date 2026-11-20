"""The Meteosat SEVIRI reader of woof/obs/geo_icechunk.py against a local mirror.

The mirror is built with icechunk's local storage in the layout of the
Satpy-processed top-level ``bkr/geo`` stores (``msg_3000m``, ``rss_3000m``,
``odegree_3000m``): one float16 array per channel, per-scan geostationary
coordinates, the scan's platform and bounds, on a coarse grid.  No network,
no data in the repository.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

icechunk = pytest.importorskip("icechunk")
zarr = pytest.importorskip("zarr")

from woof.obs import geo_icechunk as geo  # noqa: E402
from woof.obs.imager_scene import FLAG_OUT_OF_RANGE, geostationary_view  # noqa: E402

N = 300
HEIGHT = 35785831.0
HALF_M = 0.1556 * HEIGHT      # the full 3712 grid spans about +-5.57e6 m; the mirror is coarser
NS = 1_000_000_000
A_MSG = 6378169.0


def _ns(when):
    return int(when.timestamp()) * NS


def _write(path, scans, *, platform="Meteosat-10", origin=0.0, archive=False, bt=287.5, bad_rows=()):
    """``scans``: (nominal end, observation start, observation end) triples."""

    repo = icechunk.Repository.create(icechunk.local_filesystem_storage(str(path)))
    session = repo.writable_session("main")
    root = zarr.group(store=session.store)
    nt = len(scans)

    def array(name, data, attrs=None, chunks=None):
        a = root.create_array(name, shape=data.shape, dtype=data.dtype, chunks=chunks or data.shape,
                              attributes=dict(attrs or {}))
        a[...] = data

    grid = np.linspace(-HALF_M, HALF_M, N)
    for nm, quantity, value in (("05_brightness_temperature_6_2um", "bt", 240.0),
                                ("09_brightness_temperature_10_8um", "bt", bt),
                                ("01_reflectance_0_6um", "refl", 12.5)):
        data = np.full((nt, N, N), value, dtype="f2")
        data[:, N // 2, :] = np.nan            # a missing row
        for r in bad_rows:
            data[:, r, :] = -1.0 if quantity == "bt" else np.nan
        array(f"channel_{nm}", data, chunks=(1, N, N))
    array("time", np.array([_ns(s[0]) for s in scans], dtype="i8"), {"units": "nanoseconds since 1970-01-01"})
    array("x_geostationary_coordinates", np.tile(grid, (nt, 1)))
    array("y_geostationary_coordinates", np.tile(grid, (nt, 1)))
    if not archive:
        array("observation_start_time", np.array([_ns(s[1]) for s in scans], dtype="i8"),
              {"units": "nanoseconds since 1970-01-01"})
        array("observation_end_time", np.array([_ns(s[2]) for s in scans], dtype="i8"),
              {"units": "nanoseconds since 1970-01-01"})
        array("projection_longitude", np.full(nt, origin))
        array("projection_altitude", np.full(nt, HEIGHT))
        array("satellite_actual_longitude", np.full(nt, origin - 0.1))
        array("instrument", np.array([platform] * nt, dtype="U26"))
    else:
        array("start_time", np.array([_ns(s[1]) for s in scans], dtype="i8").astype("datetime64[ns]"))
        array("end_time", np.array([_ns(s[2]) for s in scans], dtype="i8").astype("datetime64[ns]"))
        orbit = repr({"projection_longitude": str(origin), "projection_altitude": str(HEIGHT),
                      "satellite_actual_longitude": str(origin + 0.2)})
        array("orbital_parameters", np.array([orbit] * nt, dtype="U512"))
        array("platform_name", np.array([platform] * nt, dtype="U12"))
    session.commit("test store")


def _scan(hour, minute=0, day=1):
    end = datetime(2025, 3, day, hour, minute, tzinfo=timezone.utc)
    return end, end - timedelta(minutes=14, seconds=48), end - timedelta(minutes=2, seconds=17)


@pytest.fixture()
def mirror(tmp_path):
    root = tmp_path / "geo"
    root.mkdir()
    _write(root / "msg_3000m.icechunk", [_scan(17, 45), _scan(18), _scan(18, 15)], bad_rows=(60,))
    _write(root / "rss_3000m.icechunk", [_scan(18, 5)], platform="Meteosat-11", origin=9.5)
    _write(root / "odegree_3000m.icechunk", [_scan(2, 15, day=2)], platform="Meteosat-11", archive=True)
    return root


def test_the_scan_labelled_18_is_the_one_observed_before_18(mirror, tmp_path):
    scene = geo.read_meteosat_scene("msg", "2025-03-01T18:00Z", channels=["C05", "11200"],
                                    root=str(mirror), cache=tmp_path / "cache")
    assert set(scene.values) == {6200, 11200}
    start, end = _scan(18)[1:]
    assert scene.time == start + (end - start) / 2
    v = scene.values[11200]
    finite = np.isfinite(v)
    assert finite.any() and np.allclose(v[finite], 287.5)
    prov = scene.provenance["channels"]["C09"]
    assert prov["time_index"] == 1 and prov["store"] == "msg_3000m" and prov["scan_start"] == start.isoformat()
    # the missing row and the non-physical row are NaN and flagged, and counted
    assert prov["dqf_rejected"] > 0
    assert (scene.flags[~finite & np.isfinite(scene.lat)] & FLAG_OUT_OF_RANGE).all()
    assert "EUMETSAT" in scene.provenance["licence"] and "EUMETSAT" in scene.provenance["attribution"]


def test_the_platform_and_sector_come_from_the_store(mirror):
    msg = geo.read_meteosat_scene("msg", "2025-03-01T18:00Z", channels=[9], bbox=(-10, 20, 25, 50),
                                  root=str(mirror), use_cache=False)
    p = msg.platforms[0]
    assert (p.name, p.family, p.sector, p.crtm_sensor) == ("Meteosat-10", "seviri", "meteosat_0", "seviri_m10")
    rss = geo.read_meteosat_scene("rss", "2025-03-01T18:05Z", channels=[9], bbox=(0, 40, 25, 60),
                                  root=str(mirror), use_cache=False)
    p = rss.platforms[0]
    assert (p.name, p.sector, p.crtm_sensor, p.sub_longitude_deg) == ("Meteosat-11", "meteosat_rss", "seviri_m11", 9.5)
    old = geo.read_meteosat_scene("odegree", "2025-03-02T02:15Z", channels=[9], bbox=(-10, 20, 25, 50),
                                  root=str(mirror), use_cache=False)
    assert old.platforms[0].crtm_sensor == "seviri_m11"
    # the archive store has only nominal bounds, and says so
    assert "nominal" in old.provenance["geometry"]


def test_geometry_is_navigated_from_the_scan_and_viewed_from_the_actual_longitude(mirror):
    scene = geo.read_meteosat_scene("msg", "2025-03-01T18:00Z", channels=[9], root=str(mirror), use_cache=False)
    ok = np.isfinite(scene.lat)
    assert ok.sum() > 1000
    near = np.nanargmin(np.abs(scene.lat) + np.abs(scene.lon))
    assert abs(scene.lat.ravel()[near]) < 1.0 and abs(scene.lon.ravel()[near]) < 1.0
    zen, _, _ = geostationary_view(scene.lat, scene.lon, -0.1)
    assert np.allclose(scene.view_zenith[ok], zen[ok])
    # the instrument scans south to north: rows later in the scan are further north
    south, north = np.nanargmin(scene.lat[:, N // 2]), np.nanargmax(scene.lat[:, N // 2])
    assert scene.scan_offset_s[south, 0] < scene.scan_offset_s[north, 0]
    start, end = _scan(18)[1:]
    span = (end - start).total_seconds()
    assert abs(scene.scan_offset_s.max() - scene.scan_offset_s.min()) <= span


def test_the_ellipsoid_is_msgs(mirror):
    lat, lon = geo.fixed_grid_latlon([0.0], [0.1], 0.0, HEIGHT, A_MSG, A_MSG * (1 - 1 / 295.488065897014))
    ref_lat, _ = geo.fixed_grid_latlon([0.0], [0.1], 0.0, HEIGHT)
    assert abs(lat[0, 0] - ref_lat[0, 0]) < 0.01 and lat[0, 0] != ref_lat[0, 0]


def test_refusals_are_named(mirror):
    with pytest.raises(geo.GeoIcechunkError, match="no scan within"):
        geo.read_meteosat_scene("msg", "2025-03-01T21:00Z", channels=[9], bbox=(-10, 20, 25, 50),
                                root=str(mirror), use_cache=False)
    with pytest.raises(geo.GeoIcechunkError, match="use one of"):
        geo.read_meteosat_scene("iodc", "2025-03-01T18:00Z", root=str(mirror), use_cache=False)
    with pytest.raises(geo.GeoIcechunkError, match="SEVIRI does not carry"):
        geo.read_meteosat_scene("msg", "2025-03-01T18:00Z", channels=[6900], root=str(mirror), use_cache=False)
    with pytest.raises(geo.GeoIcechunkError, match="not on this satellite's disc"):
        geo.read_meteosat_scene("msg", "2025-03-01T18:00Z", channels=[9], bbox=(100, 20, 120, 40),
                                root=str(mirror), use_cache=False)
    with pytest.raises(geo.GeoIcechunkError, match="has no channel"):
        geo.read_meteosat_scene("msg", "2025-03-01T18:00Z", channels=[4], bbox=(-10, 20, 25, 50),
                                root=str(mirror), use_cache=False)


def test_the_cache_round_trips(mirror, tmp_path):
    args = dict(channels=[9], bbox=(-10, 20, 25, 50), root=str(mirror), cache=tmp_path / "cache")
    first = geo.read_meteosat_scene("msg", "2025-03-01T18:00Z", **args)
    again = geo.read_meteosat_scene("msg", "2025-03-01T18:00Z", **args)
    assert again.provenance["cached"] is True and first.provenance["cached"] is False
    assert np.array_equal(np.isnan(first.values[11200]), np.isnan(again.values[11200]))
    assert again.platforms == first.platforms


def test_fetch_door_and_list_door(mirror, tmp_path, capsys):
    from woof.cli import main

    code = main(["fetch", "geo", "--satellite", "msg", "--time", "2025-03-01T18:00Z", "--channels", "C09",
                 "--bbox", "-10,20,25,50", "--root", str(mirror), "--cache", str(tmp_path / "c")])
    out = capsys.readouterr()
    assert code == 0, out.err
    line = json.loads(out.out.strip().splitlines()[-1])
    assert line["satellite"] == "Meteosat-10" and line["bands"]["11200"] > 0
    assert "EUMETSAT" in out.err
    assert main(["obs", "geo", "list", "--root", str(mirror), "--satellite", "msg", "--coverage"]) == 0
    rows = [json.loads(x) for x in capsys.readouterr().out.strip().splitlines()]
    assert [r["store"] for r in rows] == ["msg_3000m"] and rows[0]["scans"] == 3
