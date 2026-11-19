"""The bkr/geo Icechunk reader (woof/obs/geo_icechunk.py) against a local mirror.

The mirror is built here with icechunk's local storage, in the layout and
naming of ``bkr/geo/virtualized/goes{NN}_radf_C{cc}[_<end date>]`` (renamed
arrays carrying their NOAA ``source_name``, and one store with the original
names), on a coarse fixed grid.  No network, no data in the repository.
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

N = 400                      # fixed-grid pixels per side
STEP = 2 * 0.151844 / N      # rad
SCALE, OFFSET = 0.007104762829840183, -0.5586000084877014
FK1, FK2, BC1, BC2 = 50451.5, 2486.6, 1.6, 0.9973
J2000 = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)


def _j2000(when):
    return (when - J2000).total_seconds()


def _write_store(path, times, *, renamed=True, raw_value=400, bad_rows=(), sub_lon=-75.2):
    storage = icechunk.local_filesystem_storage(str(path))
    repo = icechunk.Repository.create(storage)
    session = repo.writable_session("main")
    root = zarr.group(store=session.store)
    nt = len(times)

    def array(new, old, data, attrs=None, chunks=None):
        name = new if renamed else old
        attrs = dict(attrs or {})
        if renamed:
            attrs["source_name"] = old
        a = root.create_array(name, shape=data.shape, dtype=data.dtype,
                              chunks=chunks or data.shape, attributes=attrs)
        a[...] = data

    rad = np.full((nt, N, N), raw_value, dtype=np.int16)
    dqf = np.zeros((nt, N, N), dtype=np.int8)
    for r in bad_rows:
        dqf[:, r, :] = 2
    array("abi_l1b_radiances", "Rad", rad,
          {"scale_factor": SCALE, "add_offset": OFFSET, "_FillValue": 4095}, chunks=(1, 50, N))
    array("abi_l1b_radiances_data_quality_flags", "DQF", dqf, {"_FillValue": -1}, chunks=(1, 50, N))
    x_raw = np.arange(N, dtype=np.int16)
    array("x_geostationary", "x", x_raw, {"scale_factor": STEP, "add_offset": -0.151844 + STEP / 2})
    array("y_geostationary", "y", x_raw, {"scale_factor": -STEP, "add_offset": 0.151844 - STEP / 2})
    t = np.array([_j2000(w) for w in times], dtype="f8")
    array("time", "t", t, {"units": "seconds since 2000-01-01T12:00:00"})
    array("time_bounds", "time_bounds", np.stack([t - 300, t + 300], axis=1))
    for new, old, value in (("planck_fk1", "planck_fk1", FK1), ("planck_fk2", "planck_fk2", FK2),
                            ("planck_bc1", "planck_bc1", BC1), ("planck_bc2", "planck_bc2", BC2),
                            ("kappa0", "kappa0", 0.0019),
                            ("nominal_satellite_subpoint_longitude", "nominal_satellite_subpoint_lon", sub_lon)):
        array(new, old, np.full(nt, value, dtype="f4"))
    session.commit("test store")


@pytest.fixture()
def mirror(tmp_path):
    root = tmp_path / "geo"
    base = root / "virtualized"
    base.mkdir(parents=True)
    june = datetime(2025, 6, 1, tzinfo=timezone.utc)
    december = datetime(2024, 12, 15, 12, tzinfo=timezone.utc)
    _write_store(base / "goes19_radf_C08.icechunk", [june, june + timedelta(minutes=10)], bad_rows=(60, 80, 100, 120, 200))
    _write_store(base / "goes19_radf_C14.icechunk", [june, june + timedelta(minutes=10)], raw_value=0)
    _write_store(base / "goes19_radf_C08_2025-01-01.icechunk", [december], renamed=False, raw_value=300)
    _write_store(base / "goes18_radf_C08.icechunk", [june], sub_lon=-137.2)
    return root


def _bt(raw):
    radiance = raw * SCALE + OFFSET
    return (FK2 / np.log(FK1 / radiance + 1.0) - BC1) / BC2


def test_listing_and_segments_order_dated_first_then_newest(mirror):
    names = geo.list_stores("virtualized", str(mirror))
    assert "goes19_radf_C08_2025-01-01" in names
    segs = geo.goes_segments(19, 8, names=names)
    assert [s.name for s in segs] == ["goes19_radf_C08_2025-01-01", "goes19_radf_C08"]
    assert segs[0].end.isoformat() == "2025-01-01" and segs[1].end is None


def test_a_scene_carries_noaa_calibration_dqf_and_true_geometry(mirror, tmp_path):
    scene = geo.read_goes_scene(19, "2025-06-01T00:09Z", channels=["C08", "6200"], bbox=(-100, 20, -80, 40),
                                root=str(mirror), cache=tmp_path / "cache")
    assert set(scene.values) == {6200}
    v = scene.values[6200]
    finite = np.isfinite(v)
    assert finite.any()
    assert np.allclose(v[finite], _bt(400), atol=1e-3)
    # the DQF row is dropped, and flagged
    prov = scene.provenance["channels"]["C08"]
    assert prov["dqf_rejected"] > 0 and prov["store"] == "goes19_radf_C08" and prov["time_index"] == 1
    assert (scene.flags[~finite & np.isfinite(scene.lat)] & FLAG_OUT_OF_RANGE).all()
    # the platform is the file's, with GOES-19's coefficient stand-in named
    p = scene.platforms[0]
    assert (p.name, p.sector, p.crtm_sensor) == ("GOES-19", "goes_east", "abi_g16")
    assert "abi_g16" in p.coefficient_note
    # the box is covered, and the angles agree with the closed form at the actual sub-point
    inside = (scene.lat >= 20) & (scene.lat <= 40) & (scene.lon >= -100) & (scene.lon <= -80)
    assert inside.sum() > 50
    zen, _, scan = geostationary_view(scene.lat, scene.lon, -75.2)
    ok = np.isfinite(scene.lat)
    assert np.allclose(scene.view_zenith[ok], zen[ok])
    assert np.allclose(scene.scan_angle[ok], scan[ok], atol=0.05)
    # row time runs north to south inside the scan's bounds
    assert scene.scan_offset_s is not None
    assert np.all(np.diff(scene.scan_offset_s[:, 0]) > 0)
    assert scene.time == datetime(2025, 6, 1, 0, 10, tzinfo=timezone.utc)


def test_the_dated_segment_and_the_original_names_serve_older_cases(mirror):
    scene = geo.read_goes_scene(19, "2024-12-15T12:02Z", channels=[8], bbox=(-90, 0, -60, 20),
                                root=str(mirror), use_cache=False)
    assert scene.provenance["channels"]["C08"]["store"] == "goes19_radf_C08_2025-01-01"
    v = scene.values[6200]
    assert np.allclose(v[np.isfinite(v)], _bt(300), atol=1e-3)


def test_a_time_outside_every_segment_is_refused_by_name(mirror):
    with pytest.raises(geo.GeoIcechunkError, match="no scan within"):
        geo.read_goes_scene(19, "2025-03-01T00:00Z", channels=[8], bbox=(-90, 0, -60, 20),
                            root=str(mirror), use_cache=False)
    with pytest.raises(geo.GeoIcechunkError, match="no virtualized store"):
        geo.read_goes_scene(16, "2025-06-01T00:00Z", channels=[8], root=str(mirror), use_cache=False)


def test_zero_radiance_is_nan_not_a_temperature(mirror):
    scene = geo.read_goes_scene(19, "2025-06-01T00:00Z", channels=["C14"], bbox=(-90, 0, -60, 20),
                                root=str(mirror), use_cache=False)
    assert not np.isfinite(scene.values[11200]).any()


def test_west_sector_and_box_off_the_disc(mirror):
    scene = geo.read_goes_scene(18, "2025-06-01T00:00Z", channels=[8], bbox=(-150, 20, -130, 40),
                                root=str(mirror), use_cache=False)
    assert scene.platforms[0].sector == "goes_west" and scene.platforms[0].crtm_sensor == "abi_g18"
    with pytest.raises(geo.GeoIcechunkError, match="not on this satellite's disc"):
        geo.read_goes_scene(18, "2025-06-01T00:00Z", channels=[8], bbox=(20, 20, 40, 40),
                            root=str(mirror), use_cache=False)


def test_the_cache_round_trips_and_refuses_a_changed_file(mirror, tmp_path):
    cache = tmp_path / "cache"
    first = geo.read_goes_scene(19, "2025-06-01T00:00Z", channels=[8], bbox=(-90, 0, -60, 20),
                                root=str(mirror), cache=cache)
    again = geo.read_goes_scene(19, "2025-06-01T00:00Z", channels=[8], bbox=(-90, 0, -60, 20),
                                root=str(mirror), cache=cache)
    assert again.provenance["cached"] is True and first.provenance["cached"] is False
    assert np.array_equal(np.isnan(first.values[6200]), np.isnan(again.values[6200]))
    assert again.platforms == first.platforms
    scene_file = next(cache.rglob("scene.nc"))
    scene_file.write_bytes(scene_file.read_bytes() + b"x")
    with pytest.raises(ValueError, match="does not match its provenance"):
        geo.read_goes_scene(19, "2025-06-01T00:00Z", channels=[8], bbox=(-90, 0, -60, 20),
                            root=str(mirror), cache=cache)


def test_only_noaa_buckets_are_authorised_as_virtual_containers():
    assert geo.ALLOWED_CONTAINER.fullmatch("s3://noaa-goes19/")
    assert not geo.ALLOWED_CONTAINER.fullmatch("s3://someone-else/")
    assert not geo.ALLOWED_CONTAINER.fullmatch("https://noaa-goes19.s3.amazonaws.com/")


def test_fetch_door_and_list_door(mirror, tmp_path, capsys):
    from woof.cli import main

    code = main(["fetch", "geo", "--satellite", "19", "--time", "2025-06-01T00:00Z", "--channels", "C08",
                 "--bbox", "-90,0,-60,20", "--root", str(mirror), "--cache", str(tmp_path / "c")])
    out = capsys.readouterr()
    assert code == 0, out.err
    line = json.loads(out.out.strip().splitlines()[-1])
    assert line["satellite"] == "GOES-19" and line["bands"]["6200"] > 0
    assert "NOAA" in out.err
    assert main(["obs", "geo", "list", "--root", str(mirror), "--satellite", "19", "--coverage"]) == 0
    rows = [json.loads(x) for x in capsys.readouterr().out.strip().splitlines()]
    assert [r["store"] for r in rows] == ["goes19_radf_C08_2025-01-01", "goes19_radf_C08"]
    assert rows[1]["scans"] == 2
