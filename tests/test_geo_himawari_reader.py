"""The Himawari AHI reader of woof/obs/geo_icechunk.py against a local mirror.

The mirror is built with icechunk's local storage in the layout of
``bkr/geo/virtualized/himawari{8,9}_isatss_C{cc}[_<end date>]`` (the
sectorized-CMI arrays, once under their original names and once renamed
with a ``source_name``; time as minutes since the segment's year), on a
coarse fixed grid.  No network, no data in the repository.
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

N = 400                               # 2 km columns per side in the mirror (5500 on the real grid)
HALF = 153.7e3                        # microradian, the disc's half width
ORIGIN, HEIGHT = 140.7, 35785831.0
BT_SCALE, BT_OFFSET = 0.064208984375, 69.0
RF_SCALE = 0.00048828125


@pytest.fixture(autouse=True)
def small_grid(monkeypatch):
    monkeypatch.setattr(geo, "_AHI_COLUMNS", N)


def _minutes(when, year):
    return int((when - datetime(year, 1, 1, tzinfo=timezone.utc)).total_seconds() // 60)


def _write_store(path, times, year, *, channel=13, renamed=False, count=2000, bad_rows=(), fine=1, ramp=False):
    repo = icechunk.Repository.create(icechunk.local_filesystem_storage(str(path)))
    session = repo.writable_session("main")
    root = zarr.group(store=session.store)
    n = N * fine
    reflectance = channel < 7

    def array(new, old, data, attrs=None, chunks=None):
        attrs = dict(attrs or {})
        if renamed:
            attrs["source_name"] = old
        a = root.create_array(new if renamed else old, shape=data.shape, dtype=data.dtype,
                              chunks=chunks or data.shape, attributes=attrs)
        a[...] = data

    cmi = np.full((len(times), n, n), count, dtype=np.int16)
    if ramp:    # the count names the 2 km column it sits in
        cmi[:] = (100 + np.arange(n) // fine).astype(np.int16)[None, None, :]
    for r in bad_rows:
        cmi[:, r, :] = -32767
    array("sectorized_cmi", "Sectorized_CMI", cmi,
          {"units": "1" if reflectance else "kelvin", "scale_factor": RF_SCALE if reflectance else BT_SCALE,
           "add_offset": 0.0 if reflectance else BT_OFFSET, "_FillValue": -32767}, chunks=(1, 50 * fine, n))
    step = 2 * HALF / N / fine
    grid = -HALF + step / 2 + step * np.arange(n)
    array("x_geostationary", "x", grid, {"units": "microradian"})
    array("y_geostationary", "y", -grid, {"units": "microradian"})
    array("time", "t", np.array([_minutes(w, year) for w in times], dtype="i8"),
          {"long_name": "scene start time", "units": f"minutes since {year}-01-01 00:00:00"})
    projection = root.create_array("fixedgrid_projection", shape=(len(times),), dtype="i1", chunks=(len(times),),
                                   attributes={"grid_mapping_name": "geostationary", "sweep_angle_axis": "y",
                                               "latitude_of_projection_origin": 0.0,
                                               "longitude_of_projection_origin": ORIGIN,
                                               "semi_major": 6378137.0, "semi_minor": 6356752.31414,
                                               "perspective_point_height": HEIGHT})
    projection[...] = 0
    session.commit("test store")


def _bt(count):
    return count * BT_SCALE + BT_OFFSET


@pytest.fixture()
def mirror(tmp_path):
    root = tmp_path / "geo"
    base = root / "virtualized"
    base.mkdir(parents=True)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)    # minute 0 of the year is a scan: it must not read as a fill
    scans = [t0, t0 + timedelta(minutes=10), t0 + timedelta(minutes=20)]
    _write_store(base / "himawari9_isatss_C13.icechunk", scans, 2026, bad_rows=(60, 80, 100), renamed=True)
    _write_store(base / "himawari9_isatss_C09.icechunk", scans, 2026, renamed=True, ramp=True, channel=9)
    _write_store(base / "himawari9_isatss_C03.icechunk", scans, 2026, renamed=True, ramp=True, channel=3, fine=4)
    _write_store(base / "himawari9_isatss_C14.icechunk", scans, 2026, renamed=True, count=0, channel=14)
    old = [datetime(2024, 12, 15, 12, tzinfo=timezone.utc)]
    _write_store(base / "himawari9_isatss_C13_2024-12-31.icechunk", old, 2024, count=1500)
    _write_store(base / "himawari8_isatss_C13_2022-12-31.icechunk",
                 [datetime(2022, 6, 1, tzinfo=timezone.utc)], 2022, count=1000)
    return root


def test_segments_are_listed_per_satellite_dated_first(mirror):
    names = geo.list_stores("virtualized", str(mirror))
    assert [s.name for s in geo.himawari_segments(9, 13, names=names)] == [
        "himawari9_isatss_C13_2024-12-31", "himawari9_isatss_C13"]
    assert [s.name for s in geo.himawari_segments(8, 13, names=names)] == ["himawari8_isatss_C13_2022-12-31"]
    assert geo.himawari_segments(9, 13, names=["goes19_radf_C13"]) == []


def test_a_scene_carries_the_store_values_fill_rejection_and_true_geometry(mirror, tmp_path):
    scene = geo.read_himawari_scene(9, "2026-01-01T00:11Z", channels=["C13", "10400"], bbox=(120, -10, 160, 30),
                                    root=str(mirror), cache=tmp_path / "cache")
    assert set(scene.values) == {10400}
    v = scene.values[10400]
    finite = np.isfinite(v)
    assert finite.any() and np.allclose(v[finite], _bt(2000), atol=1e-4)
    prov = scene.provenance["channels"]["C13"]
    assert prov["dqf_rejected"] > 0 and prov["time_index"] == 1 and prov["store"] == "himawari9_isatss_C13"
    assert (scene.flags[~finite & np.isfinite(scene.lat)] & FLAG_OUT_OF_RANGE).all()
    p = scene.platforms[0]
    assert (p.name, p.family, p.sector, p.crtm_sensor) == ("Himawari-9", "ahi", "himawari", "ahi_himawari9")
    inside = (scene.lat >= -10) & (scene.lat <= 30) & (scene.lon >= 120) & (scene.lon <= 160)
    assert inside.sum() > 50
    zen, _, _ = geostationary_view(scene.lat, scene.lon, ORIGIN)
    ok = np.isfinite(scene.lat)
    assert np.allclose(scene.view_zenith[ok], zen[ok])
    # the sub-point is on the equator at the projection's longitude
    near = np.nanargmin(np.abs(scene.lat) + np.abs(scene.lon - ORIGIN))
    assert abs(scene.lon.ravel()[near] - ORIGIN) < 0.5 and abs(scene.lat.ravel()[near]) < 0.5
    assert np.all(np.diff(scene.scan_offset_s[:, 0]) > 0)
    assert scene.time == datetime(2026, 1, 1, 0, 15, tzinfo=timezone.utc)   # start 00:10 plus half the nominal scan
    assert "Himawari" in scene.provenance["attribution"]


def test_the_first_scan_of_a_year_is_a_scan(mirror):
    scene = geo.read_himawari_scene(9, "2026-01-01T00:00Z", channels=[13], bbox=(120, -10, 160, 30),
                                    root=str(mirror), use_cache=False)
    assert scene.provenance["channels"]["C13"]["time_index"] == 0
    assert np.isfinite(scene.values[10400]).any()


def test_the_dated_segment_with_the_original_names_serves_an_older_case(mirror):
    scene = geo.read_himawari_scene(9, "2024-12-15T12:03Z", channels=[13], bbox=(120, -10, 160, 30),
                                    root=str(mirror), use_cache=False)
    assert scene.provenance["channels"]["C13"]["store"] == "himawari9_isatss_C13_2024-12-31"
    v = scene.values[10400]
    assert np.allclose(v[np.isfinite(v)], _bt(1500), atol=1e-4)
    h8 = geo.read_himawari_scene(8, "2022-06-01T00:00Z", channels=[13], bbox=(120, -10, 160, 30),
                                 root=str(mirror), use_cache=False)
    assert h8.platforms[0].crtm_sensor == "ahi_himawari8"


def test_refusals_are_named(mirror):
    with pytest.raises(geo.GeoIcechunkError, match="no scan within"):
        geo.read_himawari_scene(9, "2025-03-01T00:00Z", channels=[13], bbox=(120, -10, 160, 30),
                                root=str(mirror), use_cache=False)
    with pytest.raises(geo.GeoIcechunkError, match="no virtualized store"):
        geo.read_himawari_scene(9, "2026-01-01T00:00Z", channels=[16], root=str(mirror), use_cache=False)
    with pytest.raises(geo.GeoIcechunkError, match="not on this satellite's disc"):
        geo.read_himawari_scene(9, "2026-01-01T00:00Z", channels=[13], bbox=(-100, 20, -80, 40),
                                root=str(mirror), use_cache=False)
    with pytest.raises(geo.GeoIcechunkError, match="Himawari-8 and -9"):
        geo.read_himawari_scene(7, "2026-01-01T00:00Z", root=str(mirror), use_cache=False)
    with pytest.raises(geo.GeoIcechunkError, match="AHI does not carry"):
        geo._band_channel(1370, "ahi")


def test_zero_count_below_the_physical_range_is_nan(mirror):
    # C14 holds count 0, which is 69 K here: physical, so it is kept; a negative scale would not be
    scene = geo.read_himawari_scene(9, "2026-01-01T00:00Z", channels=["C14"], bbox=(120, -10, 160, 30),
                                    root=str(mirror), use_cache=False)
    v = scene.values[11200]
    assert np.allclose(v[np.isfinite(v)], BT_OFFSET, atol=1e-4)


def test_mixed_resolution_channels_register_and_visible_comes_out_in_percent(mirror, tmp_path):
    # C09 (2 km) and C03 (0.5 km) carry the 2 km column index in their counts: after the block
    # mean every pixel of the two bands must come from the same column
    scene = geo.read_himawari_scene(9, "2026-01-01T00:00Z", channels=["C09", "C03"], bbox=(120, -10, 160, 30),
                                    root=str(mirror), cache=tmp_path / "cache")
    bt, refl = scene.values[6900], scene.values[640]
    ok = np.isfinite(bt) & np.isfinite(refl)
    assert ok.sum() > 50
    count = (bt[ok] - BT_OFFSET) / BT_SCALE
    assert np.allclose(refl[ok], count * RF_SCALE * 100.0, atol=2e-2)


def test_the_cache_round_trips(mirror, tmp_path):
    cache = tmp_path / "cache"
    args = dict(channels=[13], bbox=(120, -10, 160, 30), root=str(mirror), cache=cache)
    first = geo.read_himawari_scene(9, "2026-01-01T00:00Z", **args)
    again = geo.read_himawari_scene(9, "2026-01-01T00:00Z", **args)
    assert again.provenance["cached"] is True and first.provenance["cached"] is False
    assert np.array_equal(np.isnan(first.values[10400]), np.isnan(again.values[10400]))
    assert again.platforms == first.platforms


def test_a_projection_that_is_not_a_y_sweep_is_refused(mirror):
    path = mirror / "virtualized" / "himawari9_isatss_C13.icechunk"
    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(str(path)))
    session = repo.writable_session("main")
    zarr.open_group(session.store)["fixedgrid_projection"].attrs["sweep_angle_axis"] = "x"
    session.commit("break it")
    with pytest.raises(geo.GeoIcechunkError, match="y-sweep"):
        geo.read_himawari_scene(9, "2026-01-01T00:00Z", channels=[13], bbox=(120, -10, 160, 30),
                                root=str(mirror), use_cache=False)


def test_fetch_door_and_list_door(mirror, tmp_path, capsys):
    from woof.cli import main

    code = main(["fetch", "geo", "--satellite", "himawari9", "--time", "2026-01-01T00:00Z", "--channels", "C13",
                 "--bbox", "120,-10,160,30", "--root", str(mirror), "--cache", str(tmp_path / "c")])
    out = capsys.readouterr()
    assert code == 0, out.err
    line = json.loads(out.out.strip().splitlines()[-1])
    assert line["satellite"] == "Himawari-9" and line["bands"]["10400"] > 0
    assert "Himawari" in out.err
    assert main(["obs", "geo", "list", "--root", str(mirror), "--satellite", "h9", "--channel", "13",
                 "--coverage"]) == 0
    rows = [json.loads(x) for x in capsys.readouterr().out.strip().splitlines()]
    assert [r["store"] for r in rows] == ["himawari9_isatss_C13_2024-12-31", "himawari9_isatss_C13"]
    assert rows[1]["scans"] == 3 and rows[1]["first"].startswith("2026-01-01T00:00")


def test_gk2a_is_refused_by_name_with_the_missing_coefficients(capsys):
    from woof.cli import main

    with pytest.raises(SystemExit) as exit_info:
        main(["fetch", "geo", "--satellite", "gk2a", "--time", "2025-01-02T00:00Z"])
    assert exit_info.value.code == 2
    assert "no ami_gk2a" in capsys.readouterr().err
