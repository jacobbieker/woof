"""``woof hex export --grib2`` / ``--earth2``: the WRF path's writers over the hex export frames.

The door hands the frames it just filed (``wrfout_d0N``, WRF names, a regular lat-lon grid) to
``woof.grib2_export`` and ``woof.earth2_export`` unchanged.  The first group of tests holds the wiring
(no binary needed); the second runs the real ``rw_grib2export`` / ``rw_mlexport`` over a small synthetic
frame set and reads the files back (eccodes for GRIB2, xarray for the Earth2 zarr), comparing the values
with the NetCDF frames.  They skip where a binary is not built or eccodes is not installed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from woof import bridges, earth2_export, grib2_export, ml_export
from woof.hex import export_door

NX, NY, NZ = 8, 6, 5
FRAME = "wrfout_d01_2026-09-29_{hour:02d}_00_00"


def write_frame(path: Path, hour: int) -> None:
    """A hex export frame in miniature: WRF names, 1-D-axis lat-lon geometry, the surface and 3-D state."""
    netCDF4 = pytest.importorskip("netCDF4")
    with netCDF4.Dataset(path, "w", format="NETCDF3_64BIT_OFFSET") as ds:
        for name, size in (("Time", None), ("DateStrLen", 19), ("west_east", NX), ("south_north", NY),
                           ("bottom_top", NZ), ("west_east_stag", NX + 1), ("south_north_stag", NY + 1),
                           ("bottom_top_stag", NZ + 1)):
            ds.createDimension(name, size)
        for key, value in (("MAP_PROJ", 6), ("GRID_ID", 1), ("PARENT_ID", 0), ("DX", 1e5), ("DY", 1e5),
                           ("POLE_LAT", 90.0), ("POLE_LON", 0.0), ("CEN_LAT", 32.0), ("CEN_LON", -97.5),
                           ("SIMULATION_START_DATE", "2026-09-29_00:00:00")):
            ds.setncattr(key, value)
        times = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[0, :] = np.array(list(f"2026-09-29_{hour:02d}:00:00"), dtype="S1")
        rng = np.random.default_rng(7 + hour)
        lat = np.repeat(np.linspace(30, 34, NY)[:, None], NX, axis=1)
        lon = np.repeat(np.linspace(-100, -95, NX)[None, :], NY, axis=0)
        face = np.array([100_000.0 - k * 18_000.0 for k in range(NZ + 1)])
        mass = np.sqrt(face[:-1] * face[1:])
        plane = ("Time", "south_north", "west_east")
        volume = ("Time", "bottom_top", "south_north", "west_east")

        def put(name, dims, value):
            var = ds.createVariable(name, "f4", dims)
            var[0] = np.ascontiguousarray(value, dtype=np.float32)

        put("XLAT", plane, lat)
        put("XLONG", plane, lon)
        for name, value in {"PSFC": 100000., "T2": 290., "Q2": 0.008, "U10": 4., "V10": -2., "HGT": 100.,
                            "LANDMASK": 1., "RAINNC": 1. * hour,
                            "RAINC": 0.5 * hour, "TSK": 292., "PBLH": 800., "HFX": 50., "LH": 80.,
                            "SWDOWN": 300., "GLW": 350., "SNOWNC": 0., "XLAND": 1., "SMOIS": 0.2,
                            "SST": 290., "SNOWH": 0., "SNOW": 0., "ALBEDO": 0.2, "EMISS": 0.95,
                            "GRDFLX": 5., "UST": 0.3, "CLDFRA2D": 0.1, "REFD_MAX": 20., "MUU": 0.}.items():
            put(name, plane, value + 0.01 * rng.standard_normal((NY, NX)))
        put("SINALPHA", plane, np.zeros((NY, NX)))
        put("COSALPHA", plane, np.ones((NY, NX)))
        put("P", volume, 10.0 * rng.standard_normal((NZ, NY, NX)))
        put("PB", volume, np.broadcast_to(mass[:, None, None], (NZ, NY, NX)))
        put("T", volume, np.linspace(0, 40, NZ)[:, None, None] + rng.standard_normal((NZ, NY, NX)))
        put("QVAPOR", volume, 0.004 + 1e-4 * rng.random((NZ, NY, NX)))
        for name in ("QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "CLDFRA", "TKE_PBL", "REFL_10CM", "EXCH_H"):
            put(name, volume, np.zeros((NZ, NY, NX)))
        put("U", ("Time", "bottom_top", "south_north", "west_east_stag"), 5 + rng.standard_normal((NZ, NY, NX + 1)))
        put("V", ("Time", "bottom_top", "south_north_stag", "west_east"), -3 + rng.standard_normal((NZ, NY + 1, NX)))
        stag = ("Time", "bottom_top_stag", "south_north", "west_east")
        put("W", stag, 0.01 * rng.standard_normal((NZ + 1, NY, NX)))
        put("PH", stag, np.zeros((NZ + 1, NY, NX)))
        put("PHB", stag, np.broadcast_to((287.0 * 250.0 * np.log(1e5 / face))[:, None, None], (NZ + 1, NY, NX)))


@pytest.fixture()
def frames(tmp_path):
    folder = tmp_path / "export" / "global" / "2026-09-29"
    folder.mkdir(parents=True)
    paths = []
    for hour in (0, 1):
        paths.append(folder / FRAME.format(hour=hour))
        write_frame(paths[-1], hour)
    return paths


def _args(**overrides):
    parser = argparse.ArgumentParser()
    export_door.add_export_arguments(parser)
    namespace = parser.parse_args([])
    for key, value in overrides.items():
        setattr(namespace, key, value)
    return namespace


# ---------------------------------------------------------------------------
# Wiring, no binary
# ---------------------------------------------------------------------------

def test_formats_are_off_by_default_and_the_receipt_has_no_formats_row(tmp_path):
    args = _args()
    assert args.grib2 is False and args.earth2 is None
    assert export_door.export_formats(args, [tmp_path / "wrfout_d01_x"], tmp_path) == {}


def test_grib2_goes_to_the_wrf_exporter_with_the_frames_and_its_defaults(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(grib2_export, "main", lambda ns: seen.append(ns) or 0)
    frames = [tmp_path / "wrfout_d01_2026-09-29_00_00_00", tmp_path / "wrfout_d01_2026-09-29_01_00_00"]
    rows = export_door.export_formats(_args(grib2=True, grib2_levels="850,500", grib2_fields="all"), frames, tmp_path)
    (namespace,) = seen
    request = grib2_export.build_request(namespace)
    assert request["inputs"] == [str(f.resolve()) for f in frames]
    assert request["out"] == str((tmp_path / "grib2").resolve())
    assert request["levels"] == [850, 500] and request["fields"] == "all" and request["packing"] == "complex"
    # Everything not named is the WRF path's own default.
    assert request["definitions"] == "woof" and request["winds"] == "grid" and request["post_device"] == "auto"
    assert rows["grib2"]["frames"] == 2 and rows["grib2"]["exit"] == 0 and "earth2" not in rows


def test_earth2_goes_to_the_wrf_exporter_with_profile_consumer_and_channels(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(earth2_export, "main", lambda ns: seen.append(ns) or 0)
    frame = tmp_path / "wrfout_d01_2026-09-29_00_00_00"
    rows = export_door.export_formats(
        _args(earth2="e2s-latlon-0.25", earth2_for="windgust-afno", earth2_channels="t2m", earth2_out=str(tmp_path / "e2")),
        [frame], tmp_path)
    (namespace,) = seen
    request = earth2_export.build_request(namespace)
    assert request["names"] == "earth2" and request["layout"] == "earth2"
    assert request["grid"]["target"]["kind"] == "latlon"
    channels = {row["names"]["earth2"] for row in request["variables"]}
    assert {"t2m", "u100m", "z850"} <= channels
    assert rows["earth2"]["out"] == str(tmp_path / "e2")


def test_a_failing_writer_is_an_export_error_not_a_silent_success(tmp_path, monkeypatch):
    monkeypatch.setattr(grib2_export, "main", lambda ns: 3)
    with pytest.raises(export_door.ExportDoorError, match="export-grib2 exited 3"):
        export_door.export_formats(_args(grib2=True), [tmp_path / "wrfout_d01_x"], tmp_path)


def test_formats_refuse_follow_mode(tmp_path):
    with pytest.raises(export_door.ExportDoorError, match="--follow"):
        export_door.run_export(_args(grib2=True, follow=True, forecast_dir=str(tmp_path)))


# ---------------------------------------------------------------------------
# The real writers, where built
# ---------------------------------------------------------------------------

def _binary(binary):
    try:
        found = binary.find()
    except (bridges.StaleCheckoutBuildError, FileNotFoundError) as error:
        pytest.skip(f"{binary.name} is not current on this machine: {error}")
    if found is None or not binary.probe(found)[0]:
        pytest.skip(f"{binary.name} is not built on this machine")


def test_grib2_reads_back_in_eccodes_and_equals_the_netcdf_frame(frames, tmp_path):
    _binary(grib2_export.BINARY)
    eccodes = pytest.importorskip("eccodes")
    netCDF4 = pytest.importorskip("netCDF4")
    out = tmp_path / "export"
    rows = export_door.export_formats(_args(grib2=True), frames, out)
    folder = Path(rows["grib2"]["out"])
    assert sorted(p.name for p in folder.glob("*.grib2")) == sorted(
        f"wrf{kind}_d01_20260929T{hour:02d}0000Z.grib2" for hour in (0, 1) for kind in ("prs", "sfc"))
    with netCDF4.Dataset(frames[1]) as ds:
        expect = {("2t", 2): ds["T2"][0], ("10u", 10): ds["U10"][0], ("10v", 10): ds["V10"][0],
                  ("sp", 0): ds["PSFC"][0]}
        lat, lon = ds["XLAT"][0], ds["XLONG"][0]
    seen = {}
    with open(folder / "wrfsfc_d01_20260929T010000Z.grib2", "rb") as stream:
        while handle := eccodes.codes_grib_new_from_file(stream):
            key = (eccodes.codes_get(handle, "shortName"), eccodes.codes_get(handle, "level"))
            if key in expect:
                assert eccodes.codes_get(handle, "gridType") == "regular_ll"
                ni, nj = eccodes.codes_get(handle, "Ni"), eccodes.codes_get(handle, "Nj")
                assert (ni, nj) == (NX, NY)
                seen[key] = (eccodes.codes_get_values(handle).reshape(nj, ni),
                             eccodes.codes_get_array(handle, "latitudes").reshape(nj, ni),
                             eccodes.codes_get_array(handle, "longitudes").reshape(nj, ni))
            eccodes.codes_release(handle)
    assert set(seen) == set(expect)
    for key, (values, glat, glon) in seen.items():
        np.testing.assert_allclose(values, np.asarray(expect[key]), rtol=1e-5, err_msg=str(key))
        np.testing.assert_allclose(glat, lat, atol=1e-4)
        np.testing.assert_allclose(np.where(glon > 180, glon - 360, glon), lon, atol=1e-4)


def test_earth2_reads_back_in_xarray_and_equals_the_netcdf_frame(frames, tmp_path):
    _binary(ml_export.BINARY)
    xr = pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    netCDF4 = pytest.importorskip("netCDF4")
    rows = export_door.export_formats(
        _args(earth2="e2s-native", earth2_channels="t2m,u10m,v10m,t500,z500"), frames, tmp_path / "export")
    ds = xr.open_zarr(Path(rows["earth2"]["out"]) / "d01.zarr")
    assert {"t2m", "u10m", "v10m", "t500", "z500"} <= set(ds.data_vars)
    assert ds.attrs["source"] == "woof" and ds.sizes["lead_time"] == 2
    with netCDF4.Dataset(frames[1]) as nc:
        np.testing.assert_allclose(ds.t2m.isel(time=0, lead_time=1).values, nc["T2"][0], rtol=1e-6)
        np.testing.assert_allclose(ds.u10m.isel(time=0, lead_time=1).values, nc["U10"][0], rtol=1e-6)
        np.testing.assert_allclose(ds.v10m.isel(time=0, lead_time=1).values, nc["V10"][0], rtol=1e-6)
    receipt = json.loads((Path(rows["earth2"]["out"]) / "ml-export-receipt.json").read_text(encoding="utf-8"))
    assert [f["input"] for f in receipt["domains"][0]["frames"]] == [p.name for p in frames]
