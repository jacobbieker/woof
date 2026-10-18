"""A moving nest exported to Icechunk (``woof/icechunk_export.py``,
``woof/energy/icechunk_store.py``).

A relocation keeps a nest's shape but moves it: ``XLAT``/``XLONG``/``HGT``
change (every woof wrfout gives them a ``Time`` dimension) and so do the
global attributes that place it (``I_PARENT_START``, ``J_PARENT_START``,
``CEN_LAT``, ``CEN_LON``).  The group's ``wrf_*`` attributes are the first
frame's; the placement at every frame is in ``(Time,)`` arrays.  Synthetic
frames, ``file://`` stores under ``tmp_path``; nothing touches the network.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

ic = pytest.importorskip("icechunk")
zarr = pytest.importorskip("zarr")
netCDF4 = pytest.importorskip("netCDF4")

from woof import icechunk_export as ie  # noqa: E402
from woof.energy import icechunk_store as st  # noqa: E402

T0 = datetime(2026, 6, 1, 12, tzinfo=timezone.utc)
NZ, NY, NX = 2, 5, 6
STEP_S = 120
#: (i_parent_start, j_parent_start) of the nest at each frame: it moves
#: east by one parent cell at frame 2 and north-east at frame 4.
TRACK = [(10, 20), (10, 20), (11, 20), (11, 20), (12, 21), (12, 21)]


def _centre(i: int, j: int) -> tuple[float, float]:
    return 40.0 + 0.01 * j, -105.0 + 0.01 * i


def write_frame(path: Path, index: int, *, placement: bool = True,
                offset: float = 0.0) -> None:
    from woof.io.netcdf_serialization import netcdf4_session

    valid = T0 + timedelta(seconds=index * STEP_S)
    i, j = TRACK[index]
    lat0, lon0 = _centre(i, j)
    with netcdf4_session(), netCDF4.Dataset(path, "w",
                                            format="NETCDF4") as nc:
        nc.createDimension("Time", None)
        nc.createDimension("DateStrLen", 19)
        nc.createDimension("bottom_top", NZ)
        nc.createDimension("south_north", NY)
        nc.createDimension("west_east", NX)
        nc.DX = np.float32(100.0)
        nc.GRID_ID = np.int32(3)
        nc.PARENT_ID = np.int32(2)
        nc.PARENT_GRID_RATIO = np.int32(3)
        nc.MOAD_CEN_LAT = np.float32(39.5)
        if placement:
            nc.I_PARENT_START = np.int32(i)
            nc.J_PARENT_START = np.int32(j)
            nc.CEN_LAT = np.float32(lat0)
            nc.CEN_LON = np.float32(lon0)
        times = nc.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times.set_auto_chartostring(False)
        times[0] = np.frombuffer(
            valid.strftime("%Y-%m-%d_%H:%M:%S").encode("ascii"), dtype="S1")
        lat = lat0 + 0.001 * (np.arange(NY)[:, None] - NY // 2) \
            + np.zeros((NY, NX))
        lon = lon0 + 0.001 * (np.arange(NX)[None, :] - NX // 2) \
            + np.zeros((NY, NX))
        for name, values in (("XLAT", lat), ("XLONG", lon),
                             ("HGT", 1500.0 + 10 * i + j + np.zeros((NY, NX)))):
            nc.createVariable(name, "f4", ("Time", "south_north",
                                           "west_east"))[0] = values
        nc.createVariable("U", "f4", ("Time", "bottom_top", "south_north",
                                      "west_east"))[0] = \
            np.full((NZ, NY, NX), index + offset, np.float32)


def publish(run: Path, index: int, **kwargs) -> Path:
    from woof.progress_log import write_frame_marker

    valid = T0 + timedelta(seconds=index * STEP_S)
    folder = run / "wrfout"
    folder.mkdir(parents=True, exist_ok=True)
    name = f"wrfout_d03_{valid:%Y-%m-%d_%H_%M_%S}"
    temporary = folder / f".{name}.tmp.{os.getpid()}"
    write_frame(temporary, index, **kwargs)
    final = folder / name
    os.replace(temporary, final)
    write_frame_marker(run / "ready", domain=3,
                       valid_time=valid.replace(tzinfo=None), path=final)
    return final


def _url(path: Path) -> str:
    return "file://" + str(path)


def _group(url: str, group: str = "d03"):
    repo = st.open_repository(url, create=False)
    return zarr.open_group(repo.readonly_session("main").store, path=group,
                           mode="r")


def _drain(run: Path, url: str, state: Path, **extra) -> dict:
    options = ie.ExportOptions.from_mapping({"url": url, **extra})
    return ie.Exporter([run], options, state_dir=state,
                       intervals={3: STEP_S}, say=lambda _l: None).run()


def test_placement_is_stored_per_frame_and_attrs_are_frame_zero(tmp_path):
    run = tmp_path / "run"
    for index in range(len(TRACK)):
        publish(run, index)
    url = _url(tmp_path / "nest.icechunk")
    status = _drain(run, url, tmp_path / "state")
    assert status["state"] == "complete"
    assert status["frames_exported"] == len(TRACK)

    group = _group(url)
    attrs = dict(group.attrs)
    # The group attributes are frame 0's and say where the rest are.
    assert attrs["wrf_I_PARENT_START"] == TRACK[0][0]
    assert attrs["wrf_J_PARENT_START"] == TRACK[0][1]
    assert attrs["placement_variables"] == [
        st.placement_array_name(a) for a in st.PLACEMENT_ATTRS]
    i = group["wrf_I_PARENT_START"]
    assert tuple(i.metadata.dimension_names) == ("Time",)
    assert str(i.dtype) == "int32"
    np.testing.assert_array_equal(i[:], [t[0] for t in TRACK])
    np.testing.assert_array_equal(group["wrf_J_PARENT_START"][:],
                                  [t[1] for t in TRACK])
    np.testing.assert_allclose(group["wrf_CEN_LAT"][:],
                               [_centre(*t)[0] for t in TRACK], rtol=1e-6)
    np.testing.assert_allclose(group["wrf_CEN_LON"][:],
                               [_centre(*t)[1] for t in TRACK], rtol=1e-6)
    # Coordinates and terrain follow the nest (they are Time variables).
    xlat = group["XLAT"][:]
    assert xlat.shape == (len(TRACK), NY, NX)
    np.testing.assert_allclose(xlat[:, NY // 2, NX // 2],
                               [_centre(*t)[0] for t in TRACK], rtol=1e-6)
    assert group["HGT"][0, 0, 0] != group["HGT"][4, 0, 0]
    # The placement arrays are not wrfout variables to read per frame.
    assert "wrf_CEN_LAT" not in attrs["variables"]
    assert status["frames_refused"] == 0


def test_xarray_reads_the_track(tmp_path):
    xr = pytest.importorskip("xarray")
    run = tmp_path / "run"
    for index in range(len(TRACK)):
        publish(run, index)
    url = _url(tmp_path / "nest.icechunk")
    _drain(run, url, tmp_path / "state")
    repo = st.open_repository(url, create=False)
    ds = xr.open_zarr(repo.readonly_session("main").store, group="d03",
                      consolidated=False, zarr_format=3)
    assert ds["wrf_I_PARENT_START"].dims == ("Time",)
    assert ds["Time"].values[0] == np.datetime64("2026-06-01T12:00:00")
    moved = ds["wrf_I_PARENT_START"].sel(
        Time=np.datetime64("2026-06-01T12:08:00")).item()
    assert moved == TRACK[4][0]
    assert ds.attrs["wrf_I_PARENT_START"] == TRACK[0][0]


def test_live_equals_batch_and_a_rewrite_replaces_the_placement(tmp_path):
    """The live export (axis growing one frame at a time, placement arrays
    resized with it) equals the batch writer on a fixed axis; a frame
    written again without the attributes writes the fill, not the old
    file's placement."""

    run = tmp_path / "run"
    for index in range(len(TRACK)):
        publish(run, index)
    live = _url(tmp_path / "live")
    _drain(run, live, tmp_path / "state")
    batch = _url(tmp_path / "batch")
    files = sorted((run / "wrfout").glob("wrfout_d03_*"))
    st.write_gridded_segment(
        batch, files, group="d03",
        axis=st.TimeAxis(np.datetime64("2026-06-01T12:00"),
                         np.datetime64("2026-06-01T12:00")
                         + np.timedelta64((len(files) - 1) * STEP_S, "s"),
                         STEP_S))
    a, b = _group(live), _group(batch)
    for name in st.PLACEMENT_ATTRS:
        array = st.placement_array_name(name)
        np.testing.assert_array_equal(a[array][:], b[array][:])

    time.sleep(0.01)
    publish(run, 3, placement=False, offset=100.0)
    _drain(run, live, tmp_path / "state")
    group = _group(live)
    assert group["U"][3, 0, 0, 0] == 103.0
    assert group["wrf_I_PARENT_START"][3] == 0
    assert np.isnan(group["wrf_CEN_LAT"][3])
    assert group["wrf_I_PARENT_START"][4] == TRACK[4][0]


def test_a_store_without_placement_attributes_has_no_placement_arrays(
        tmp_path):
    run = tmp_path / "run"
    for index in range(2):
        publish(run, index, placement=False)
    url = _url(tmp_path / "repo")
    _drain(run, url, tmp_path / "state")
    group = _group(url)
    assert group.attrs["placement_variables"] == []
    assert "wrf_CEN_LAT" not in {name for name, _ in group.members()}
    # A later frame that carries them creates them, fill before.
    publish(run, 2)
    _drain(run, url, tmp_path / "state")
    group = _group(url)
    np.testing.assert_array_equal(group["wrf_I_PARENT_START"][:],
                                  [0, 0, TRACK[2][0]])


def test_a_relocated_frame_of_another_shape_is_still_refused(tmp_path):
    """A relocation keeps the nest's shape; a frame that does not is a
    different grid, refused and kept on disk even with
    delete_after_commit."""

    run = tmp_path / "run"
    publish(run, 0)
    publish(run, 1)
    bad = publish(run, 2)
    from woof.io.netcdf_serialization import netcdf4_session
    with netcdf4_session(), netCDF4.Dataset(bad, "a") as nc:
        nc.renameVariable("U", "U_old")
        nc.createDimension("west_east_wide", NX + 1)
        nc.createVariable("U", "f4", ("Time", "bottom_top", "south_north",
                                      "west_east_wide"))[0] = 0.0
    status = _drain(run, _url(tmp_path / "repo"), tmp_path / "state",
                    delete_after_commit=True)
    assert status["state"] == "incomplete"
    assert "shape" in status["refused"][bad.name]
    assert bad.exists()
    assert sorted(p.name for p in (run / "wrfout").iterdir()) == [bad.name]
