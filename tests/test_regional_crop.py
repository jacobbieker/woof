"""A small domain from a global NetCDF source is cropped before its decode.

The mapped engine decodes a NetCDF series whole, so a whole-globe source
prepared for a domain a few tens of kilometres wide decoded every cell of
every level of every valid time (a 10 km global grid over a 27-hour window
grew past 68 GB).  ``woof.regional_crop`` cuts each object to the run's
area before the input manifest is authored.  These tests build a small
synthetic global grid and check the crop's bounds, its longitude handling
on either side of the seam, that only the cropped hyperslabs are read, and
the front doors that now carry an ``area`` for such a source.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest

from woof import fetch, regional_crop as rc


STEP = 5.0
LEVELS = (100000.0, 85000.0, 50000.0)


def _axes(convention: str):
    latitude = np.arange(-87.5, 90.0, STEP, dtype=np.float32)
    if convention == "signed":
        longitude = np.arange(-177.5, 180.0, STEP, dtype=np.float32)
    else:
        longitude = np.arange(2.5, 360.0, STEP, dtype=np.float32)
    return latitude, longitude


def _field(latitude, longitude, level: float, offset: float = 0.0):
    # A value that names its own cell, so a wrong column or row shows.
    lat = latitude.astype(np.float64)[:, None]
    lon = np.mod(longitude.astype(np.float64), 360.0)[None, :]
    return (offset + level / 1000.0 + 1000.0 * lat + lon).astype(np.float32)


def write_global(path: Path, *, convention: str = "signed",
                 three_d: bool = True, offset: float = 0.0) -> Path:
    """A Met-Office-shaped object: one variable on a regular global grid."""

    latitude, longitude = _axes(convention)
    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.setncatts({"Conventions": "CF-1.7", "title": "synthetic global grid"})
        if three_d:
            ds.createDimension("pressure", len(LEVELS))
        ds.createDimension("latitude", latitude.size)
        ds.createDimension("longitude", longitude.size)
        ds.createDimension("bnds", 2)
        lat = ds.createVariable("latitude", "f4", ("latitude",))
        lat.setncatts({"axis": "Y", "units": "degrees_north",
                       "standard_name": "latitude", "bounds": "latitude_bnds"})
        lat[:] = latitude
        lon = ds.createVariable("longitude", "f4", ("longitude",))
        lon.setncatts({"axis": "X", "units": "degrees_east",
                       "standard_name": "longitude", "bounds": "longitude_bnds"})
        lon[:] = longitude
        lat_b = ds.createVariable("latitude_bnds", "f4", ("latitude", "bnds"))
        lat_b[:] = np.stack([latitude - STEP / 2, latitude + STEP / 2], axis=1)
        lon_b = ds.createVariable("longitude_bnds", "f4", ("longitude", "bnds"))
        lon_b[:] = np.stack([longitude - STEP / 2, longitude + STEP / 2], axis=1)
        ref = ds.createVariable("forecast_reference_time", "i8", ())
        ref.setncatts({"units": "seconds since 1970-01-01 00:00:00",
                       "standard_name": "forecast_reference_time"})
        ref.assignValue(1_791_000_000)
        if three_d:
            pressure = ds.createVariable("pressure", "f4", ("pressure",))
            pressure.setncatts({"axis": "Z", "units": "Pa", "positive": "down"})
            pressure[:] = np.asarray(LEVELS, dtype=np.float32)
            var = ds.createVariable(
                "air_temperature", "f4", ("pressure", "latitude", "longitude"),
                zlib=True, complevel=1, chunksizes=(1, 8, 8))
            var[:] = np.stack([_field(latitude, longitude, level, offset)
                               for level in LEVELS])
        else:
            var = ds.createVariable("surface_altitude", "f4",
                                    ("latitude", "longitude"),
                                    zlib=True, complevel=1, chunksizes=(8, 8),
                                    fill_value=np.float32(-1.0e30))
            var[:] = _field(latitude, longitude, 0.0, offset)
        var.setncatts({"standard_name": "air_temperature", "units": "K",
                       "coordinates": "forecast_reference_time",
                       "grid_mapping": "latitude_longitude"})
    return path


def _read(path: Path, name: str):
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_maskandscale(False)
        return np.asarray(ds.variables[name][:])


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------

def test_an_interior_box_keeps_its_cells_plus_the_stencil_pad():
    latitude, longitude = _axes("signed")
    box = rc.CropBox(south=40.0, north=50.0, west=20.0, span=15.0)
    plan = rc.plan_crop(latitude, longitude, box, pad=2)
    # Latitudes 42.5 and 47.5 are inside (rows 26, 27); two more each side.
    assert plan.rows == (24, 30)
    # Longitudes 22.5..32.5 are columns 40..42; two more each side.
    assert plan.column_runs == ((38, 45),)
    assert plan.column_shift == (0.0,)
    assert plan.shape == (6, 7)
    assert not plan.is_whole


def test_a_box_wider_than_the_globe_keeps_every_column_in_stored_order():
    latitude, longitude = _axes("signed")
    plan = rc.plan_crop(latitude, longitude,
                        rc.CropBox(-10.0, 10.0, 0.0, 359.0), pad=2)
    assert plan.column_runs == ((0, longitude.size),)
    # Latitudes -7.5..7.5 are rows 16..19; two more each side.
    assert plan.rows == (14, 22)


@pytest.mark.parametrize("convention,area,expected_first,expected_last", [
    # A -180..180 axis and a box over the antimeridian: 167.5 .. 192.5.
    ("signed", "40,170,50,-170", 162.5, 197.5),
    # A 0..360 axis and a box over the prime meridian: 352.5 .. 367.5.
    ("positive", "40,-10,50,10", 342.5, 377.5),
])
def test_a_box_over_the_axis_seam_continues_past_it(
        tmp_path, convention, area, expected_first, expected_last):
    source = write_global(tmp_path / "global.nc", convention=convention)
    latitude, longitude = _axes(convention)
    box = rc.box_from_area(fetch.parse_area(area))
    plan = rc.plan_crop(latitude, longitude, box, pad=2)
    assert len(plan.column_runs) == 2
    assert plan.column_shift == (0.0, 360.0)
    cropped = tmp_path / "cropped.nc"
    rc.crop_object(source, cropped, plan)
    lon = _read(cropped, "longitude").astype(np.float64)
    # Uniform and increasing across the seam, in exact multiples of the step.
    assert np.allclose(np.diff(lon), STEP, rtol=0, atol=0)
    assert lon[0] == expected_first and lon[-1] == expected_last
    bounds = _read(cropped, "longitude_bnds").astype(np.float64)
    assert np.array_equal(bounds[:, 0], lon - STEP / 2)
    # Every value is the source's own value at the same place on the globe.
    lat = _read(cropped, "latitude")
    values = _read(cropped, "air_temperature")
    for k, level in enumerate(LEVELS):
        assert np.array_equal(
            values[k], _field(lat, lon.astype(np.float32), level))


def test_the_crop_copies_metadata_compression_and_fill(tmp_path):
    tmp = tmp_path
    source = write_global(tmp / "orography.nc", three_d=False)
    latitude, longitude = _axes("signed")
    plan = rc.plan_crop(latitude, longitude, rc.CropBox(40.0, 50.0, 20.0, 15.0))
    cropped = tmp / "cropped.nc"
    rc.crop_object(source, cropped, plan)
    with netCDF4.Dataset(source) as src, netCDF4.Dataset(cropped) as dst:
        assert {k: src.getncattr(k) for k in src.ncattrs()} == {
            k: dst.getncattr(k) for k in dst.ncattrs()}
        a, b = src["surface_altitude"], dst["surface_altitude"]
        assert b.dtype == a.dtype and b.dimensions == a.dimensions
        assert {k: a.getncattr(k) for k in a.ncattrs()} == {
            k: b.getncattr(k) for k in b.ncattrs()}
        assert b.filters()["zlib"] and b.filters()["complevel"] == 1
        assert b.chunking() == [6, 7]
        assert dst["forecast_reference_time"].getValue() == 1_791_000_000
        assert dst.dimensions["latitude"].size == 6
        assert dst.dimensions["longitude"].size == 7


def test_only_the_cropped_hyperslabs_are_read():
    """The decode never touches a cell outside the plan."""

    latitude, longitude = _axes("signed")
    plan = rc.plan_crop(latitude, longitude,
                        rc.box_from_area(fetch.parse_area("40,170,50,-170")))
    full = np.stack([_field(latitude, longitude, level) for level in LEVELS])
    requests = []

    class Spy:
        def __getitem__(self, key):
            requests.append(key)
            return full[key]

    dims = ("pressure", "latitude", "longitude")
    joined = rc._take(Spy(), dims, "latitude", "longitude", plan, shift=False)
    assert joined.shape == (len(LEVELS), *plan.shape)
    assert len(requests) == len(plan.column_runs)
    for key, run in zip(requests, plan.column_runs):
        assert key[1] == slice(*plan.rows)
        assert key[2] == slice(*run)
    read_cells = sum((k[1].stop - k[1].start) * (k[2].stop - k[2].start)
                     for k in requests)
    assert read_cells == plan.shape[0] * plan.shape[1]
    assert read_cells < latitude.size * longitude.size / 20


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------

_NAMELIST = """&share
 wrf_core = 'ARW',
 max_dom = 1,
 interval_seconds = 10800,
/
&geogrid
 parent_id         = 1,
 parent_grid_ratio = 1,
 i_parent_start    = 1,
 j_parent_start    = 1,
 e_we              = 41,
 e_sn              = 41,
 geog_data_res     = 'default',
 dx = 3000,
 dy = 3000,
 map_proj = 'lambert',
 ref_lat   = 45.0,
 ref_lon   = {ref_lon},
 truelat1  = 35.0,
 truelat2  = 55.0,
 stand_lon = {ref_lon},
/
"""


def _composition(tmp_path: Path) -> Path:
    path = tmp_path / "composition.json"
    path.write_text(json.dumps({"supplements": {"terrain_height": {
        "data_role": "invariant_surface", "format": "netcdf"}},
        "field_sources": {"donor": {"data_role": "donor_data"}}}))
    return path


def _args(tmp_path: Path, *, ref_lon: float, config: str | None = None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    inputs = [write_global(tmp_path / f"f{lead:03d}-air_temperature.nc",
                           offset=lead) for lead in (3, 6)]
    orography = write_global(tmp_path / "f000-orography.nc", three_d=False)
    donor = tmp_path / "donor.grib2"
    donor.write_bytes(b"GRIB-not-cropped")
    namelist = tmp_path / "namelist.wps"
    namelist.write_text(_NAMELIST.format(ref_lon=ref_lon))
    experiment = None
    if config is not None:
        experiment = tmp_path / "experiment.toml"
        experiment.write_text(config)
    chain = tmp_path / "chain"
    chain.mkdir()
    return SimpleNamespace(
        source="met-office-global", wps_namelist=str(namelist),
        experiment_config=None if experiment is None else str(experiment),
        author_input_manifest=str(chain / "inputs.json"),
        composition=str(_composition(tmp_path)),
        mapped_inputs=list(inputs), input_list=None,
        supplement=[f"invariant_surface={orography}",
                    f"donor_data={donor}"]), inputs, orography, donor


def test_the_stage_rebinds_inputs_and_terrain_to_one_cropped_grid(tmp_path):
    args, inputs, orography, donor = _args(tmp_path, ref_lon=179.0)
    result = rc.crop_packaged_inputs(rc.CROP_KIND, args)
    assert result["cropped"] and result["inputs"] == 3
    assert result["area_from"].startswith("namelist domains")
    directory = Path(result["directory"])
    assert directory.parent == tmp_path / "chain"
    # The manifest binds area-dependent bytes, so it lives in the keyed folder.
    assert Path(args.author_input_manifest) == directory / "inputs.json"
    assert [Path(p).parent for p in args.mapped_inputs] == [directory] * 2
    assert Path(args.input_list).read_text().splitlines() == [
        str(p) for p in args.mapped_inputs]
    terrain = Path(args.supplement[0].split("=", 1)[1])
    assert terrain.parent == directory
    # The GRIB2 donor is bound unchanged.
    assert args.supplement[1] == f"donor_data={donor}"
    # The domain sits on the antimeridian: the crop straddles it.
    lon = _read(args.mapped_inputs[0], "longitude").astype(np.float64)
    assert lon[0] < 180.0 < lon[-1]
    assert np.allclose(np.diff(lon), STEP, rtol=0, atol=0)
    # Primary and terrain share their axes exactly (exact_coordinate_subset).
    for name in ("latitude", "longitude"):
        assert np.array_equal(_read(terrain, name),
                              _read(args.mapped_inputs[0], name))
    # Far smaller than the globe, and the original objects are untouched.
    assert result["cropped_bytes"] < result["input_bytes"]
    assert _read(inputs[0], "air_temperature").shape == (3, 36, 72)
    receipt = json.loads((directory / rc.RECEIPT_NAME).read_text())
    assert receipt["schema"] == rc.RECEIPT_SCHEMA
    assert {row["input"] for row in receipt["objects"]} == {
        str(p.resolve()) for p in (*inputs, orography)}


def test_a_second_preparation_reuses_an_intact_crop(tmp_path):
    args, inputs, _orography, _donor = _args(tmp_path, ref_lon=10.0)
    first = rc.crop_packaged_inputs(rc.CROP_KIND, args)
    cropped = Path(first["directory"]) / inputs[0].name
    stamp = cropped.stat().st_mtime_ns
    args.mapped_inputs = list(inputs)
    args.supplement = [f"invariant_surface={tmp_path / 'f000-orography.nc'}"]
    args.author_input_manifest = str(tmp_path / "chain" / "inputs.json")
    second = rc.crop_packaged_inputs(rc.CROP_KIND, args)
    assert second["directory"] == first["directory"]
    assert cropped.stat().st_mtime_ns == stamp


def test_a_declared_area_is_honoured_and_must_hold_the_domains(tmp_path):
    args, *_ = _args(tmp_path, ref_lon=10.0, config=(
        "[fetch]\nsource = 'met-office-global'\narea = '35,0,55,20'\n"))
    result = rc.crop_packaged_inputs(rc.CROP_KIND, args)
    assert result["area_from"] == "experiment [fetch] area"
    lat = _read(args.mapped_inputs[0], "latitude")
    lon = _read(args.mapped_inputs[0], "longitude")
    # 35..55 N plus two cells, 0..20 E plus two cells.
    assert lat[0] == 27.5 and lat[-1] == 62.5
    assert lon[0] == -7.5 and lon[-1] == 27.5

    small, *_ = _args(tmp_path / "small", ref_lon=10.0, config=(
        "[fetch]\nsource = 'met-office-global'\narea = '44.9,9.9,45.1,10.1'\n"))
    with pytest.raises(ValueError, match="does not hold the namelist's domains"):
        rc.crop_packaged_inputs(rc.CROP_KIND, small)


def test_objects_on_another_grid_are_refused(tmp_path):
    latitude, longitude = _axes("signed")
    plan = rc.plan_crop(latitude, longitude, rc.CropBox(40.0, 50.0, 20.0, 15.0))
    source = write_global(tmp_path / "global.nc")
    elsewhere = rc.CropPlan((35, 72), plan.rows, plan.column_runs, plan.column_shift)
    with pytest.raises(ValueError, match="share its grid"):
        rc.crop_object(source, tmp_path / "out.nc", elsewhere)
    # The same shape on a 0..360 axis would cut another place on the globe.
    positive = write_global(tmp_path / "positive.nc", convention="positive")
    with pytest.raises(ValueError, match="latitude/longitude values"):
        rc.crop_object(positive, tmp_path / "out2.nc", plan,
                       axes=(latitude, longitude))
    with pytest.raises(ValueError, match="stencil pad"):
        rc.plan_crop(latitude, longitude, rc.CropBox(40.0, 50.0, 20.0, 15.0), pad=0)


def test_a_float32_axis_with_an_inexact_step_is_still_planned():
    latitude = (np.arange(1800, dtype=np.float64) * 0.1 - 89.95).astype(np.float32)
    longitude = (np.arange(3600, dtype=np.float64) * 0.1 - 179.95).astype(np.float32)
    plan = rc.plan_crop(latitude, longitude,
                        rc.box_from_area(fetch.parse_area("40,175,41,-175")))
    assert len(plan.column_runs) == 2 and plan.column_shift == (0.0, 360.0)
    assert plan.shape == (14, 104)


def test_listed_input_digests_are_used_and_named(tmp_path):
    args, inputs, _orography, _donor = _args(tmp_path, ref_lon=10.0)
    (tmp_path / "SHA256SUMS").write_text(
        f"{'a' * 64}  {inputs[0].name}\n", encoding="utf-8")
    result = rc.crop_packaged_inputs(rc.CROP_KIND, args)
    receipt = json.loads(Path(result["receipt"]).read_text())
    rows = {Path(row["input"]).name: row for row in receipt["objects"]}
    assert rows[inputs[0].name]["input_sha256"] == "a" * 64
    assert rows[inputs[0].name]["input_sha256_from"].startswith("SHA256SUMS")
    assert rows[inputs[1].name]["input_sha256_from"] == "hashed"
    assert rows[inputs[1].name]["input_sha256"] == rc._sha256(inputs[1])


def test_the_crop_source_does_not_prepare_as_posted():
    from woof.source_cli import as_posted_refusal

    # Each lead is cropped before the window's manifest is authored, so the
    # source never prepares beside an as-posted fetch.
    assert as_posted_refusal("met-office-global") is not None


# ---------------------------------------------------------------------------
# The front doors
# ---------------------------------------------------------------------------

def test_the_global_netcdf_profile_declares_the_crop():
    from woof.source_authorities import packaged_profile

    assert packaged_profile("met-office-global-netcdf")["input_crop"] == rc.CROP_KIND
    assert rc.crops_at_preparation("met-office-global")
    assert rc.crops_at_preparation("metoffice-global")
    assert not rc.crops_at_preparation("gfs")


def test_fetch_and_planners_carry_an_area_for_the_global_netcdf_source():
    from woof import domain_wizard as dw

    assert fetch.fetch_accepts_area("met-office-global")
    assert dw.source_fetch_takes_a_crop_box("met-office-global")
    fetch.validate_fetch_hints(
        {"source": "met-office-global", "cycle": "2026-10-16T18", "hours": 27,
         "cadence": 3, "forecast_start_hour": 3,
         "area": "48.01,27.91,52.80,33.16"}, source="area.toml")
    projection = {"map_proj": "lambert", "ref_lat": 45.0, "ref_lon": 10.0,
                  "truelat1": 35.0, "truelat2": 55.0, "stand_lon": 10.0}
    hint = dw.fetch_area_hint(projection, 46, 46, source="met-office-global",
                              root_dx_m=1875.0)
    area = fetch.parse_area(hint)
    assert area.lat_south < 45.0 < area.lat_north
    assert area.lon_west < 10.0 < area.lon_east


def test_a_whole_object_route_without_a_crop_still_refuses_an_area():
    whole_object = sorted(
        source for source in fetch._fetch_hint_sources()
        if not fetch.fetch_accepts_area(source))
    assert "met-office-global" not in whole_object
    with pytest.raises(ValueError, match="names a crop"):
        fetch.validate_fetch_hints(
            {"source": whole_object[0], "cycle": "2026-07-29T00", "hours": 6,
             "area": "30,-100,40,-90"}, source="whole-object.toml")


def test_the_area_does_not_key_a_whole_object_download():
    from woof.go_cli import managed_download_key

    table = {"source": "met-office-global", "cycle": "2026-10-16T18",
             "hours": 27, "cadence": 3, "forecast_start_hour": 3}
    assert managed_download_key(table) == managed_download_key(
        {**table, "area": "48.01,27.91,52.80,33.16"})
    gfs = {"source": "gfs", "cycle": "2026-10-16T18", "hours": 6}
    assert managed_download_key({**gfs, "area": "30,-100,40,-90"}) != \
        managed_download_key({**gfs, "area": "30,-100,41,-90"})
