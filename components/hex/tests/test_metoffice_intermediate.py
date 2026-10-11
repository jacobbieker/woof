"""The Met Office global door: files, crop, conversion, supplement, receipt, refusals.

Engine-free where it can be: each test writes tiny synthetic Met Office
NetCDF files with ``netCDF4`` (a coarse global grid with the product's
coordinates and scalar times), and a stand-in ``met_intermediate`` (a Python
script that writes the supplement's soil/LANDSEA/SEAICE with this tree's own
record writer) drives the whole door.  One test runs the real binary over
the tiny GDAS fixtures as the supplement when it is built, and skips
otherwise.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import textwrap

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

netCDF4 = pytest.importorskip("netCDF4")

from woof.hex import met_intermediate_door as door  # noqa: E402
from woof.hex import metoffice_intermediate as mo  # noqa: E402
from woof.hex.cli import build_parser, main as hex_main  # noqa: E402
from woof.hex.hrrr_intermediate import IntermediateRefusal, LatLonTarget, wps_record_bytes  # noqa: E402
from woof.hex.met_intermediate_door import GlobalIntermediateRefusal  # noqa: E402
from woof.hex.wps_intermediate import WpsIntermediateReader, inventory  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
GDAS_FIXTURES = REPO / "tests" / "fixtures" / "gdas-process-id"
CYCLE = datetime(2026, 10, 11, 0)
#: A coarse global grid with the product's layout: cell centres, south first.
NX, NY = 48, 24
DLON, DLAT = 360.0 / NX, 180.0 / NY
LATS = -90.0 + DLAT * (np.arange(NY) + 0.5)
LONS = -180.0 + DLON * (np.arange(NX) + 0.5)
LEVELS_37 = (100000.0, 85000.0, 50000.0, 25000.0, 10000.0, 1000.0, 100.0)
LEVELS_RH = LEVELS_37[:-1]


# ---------------------------------------------------------------------------
# synthetic Met Office files
# ---------------------------------------------------------------------------
def _write_nc(path: Path, *, var: str, units: str, values: np.ndarray, run: datetime,
              valid: datetime, levels=None, height=None, standard_name=None,
              time_override: datetime | None = None) -> None:
    with netCDF4.Dataset(str(path), "w") as ds:
        if levels is not None:
            ds.createDimension("pressure", len(levels))
            p = ds.createVariable("pressure", "f4", ("pressure",))
            p.units = "Pa"
            p[:] = np.asarray(levels, dtype=np.float32)
        ds.createDimension("latitude", NY)
        ds.createDimension("longitude", NX)
        lat = ds.createVariable("latitude", "f4", ("latitude",))
        lat.units = "degrees_north"
        lat[:] = LATS
        lon = ds.createVariable("longitude", "f4", ("longitude",))
        lon.units = "degrees_east"
        lon[:] = LONS
        crs = ds.createVariable("latitude_longitude", "i4")
        crs.grid_mapping_name = "latitude_longitude"
        dims = (("pressure",) if levels is not None else ()) + ("latitude", "longitude")
        data = ds.createVariable(var, "f4", dims)
        data.units = units
        data.grid_mapping = "latitude_longitude"
        if standard_name:
            data.standard_name = standard_name
        data[:] = values.astype(np.float32)
        frt = ds.createVariable("forecast_reference_time", "i8")
        frt.units = "seconds since 1970-01-01 00:00:00"
        frt[...] = int((run - datetime(1970, 1, 1)).total_seconds())
        t = ds.createVariable("time", "i8")
        t.units = "seconds since 1970-01-01 00:00:00"
        t[...] = int(((time_override or valid) - datetime(1970, 1, 1)).total_seconds())
        fp = ds.createVariable("forecast_period", "i4")
        fp.units = "seconds"
        fp[...] = int(((time_override or valid) - run).total_seconds())
        if height is not None:
            h = ds.createVariable("height", "f4")
            h.units = "m"
            h[...] = height


def write_run(root: Path, leads=(0, 3), *, old_height_name: bool = False, skip=(),
              run: datetime = CYCLE) -> None:
    """One synthetic Met Office run: every file the door reads, per lead."""

    root.mkdir(parents=True, exist_ok=True)
    shape2 = (NY, NX)
    lat2 = np.repeat(LATS[:, None], NX, axis=1)
    for lead in leads:
        valid = run + timedelta(hours=lead)

        def put(name, var, units, values, **kw):
            if name in skip:
                return
            _write_nc(root / mo.file_name(run, lead, name), var=var, units=units,
                      values=values, run=run, valid=valid, **kw)

        def column(levels, top, bottom):
            frac = np.log(np.asarray(levels) / 100000.0)[:, None, None]
            return np.broadcast_to(bottom + (top - bottom) * frac / np.log(100.0 / 100000.0),
                                   (len(levels), *shape2)).copy()

        put("temperature_on_pressure_levels", "air_temperature", "K",
            column(LEVELS_37, 220.0, 285.0) - 0.1 * np.abs(lat2), levels=LEVELS_37)
        put("height_ASL_on_pressure_levels" if old_height_name else
            "geopotential_height_on_pressure_levels", "geopotential_height", "m",
            column(LEVELS_37, 48000.0, 100.0), levels=LEVELS_37)
        put("relative_humidity_on_pressure_levels", "relative_humidity", "1",
            np.full((len(LEVELS_RH), *shape2), 0.6), levels=LEVELS_RH)
        put("wind_speed_on_pressure_levels", "wind_speed", "m s-1",
            np.full((len(LEVELS_37), *shape2), 10.0), levels=LEVELS_37)
        # From the west everywhere: u = +10, v = 0.
        put("wind_direction_on_pressure_levels", "wind_from_direction", "degrees",
            np.full((len(LEVELS_37), *shape2), 270.0), levels=LEVELS_37)
        put("pressure_at_mean_sea_level", "air_pressure_at_sea_level", "Pa",
            np.full(shape2, 101300.0))
        put("temperature_at_surface", "surface_temperature", "K", np.full(shape2, 288.0))
        if lead == 0:
            put("height_of_orography", "surface_altitude", "m", np.full(shape2, 50.0))
        put("snow_depth_water_equivalent", "lwe_thickness_of_surface_snow_amount", "m",
            np.full(shape2, 0.01), standard_name="lwe_thickness_of_surface_snow_amount")
        put("temperature_at_screen_level", "air_temperature", "K", np.full(shape2, 287.0),
            height=1.5)
        put("relative_humidity_at_screen_level", "relative_humidity", "1",
            np.full(shape2, 0.7), height=1.5)
        put("wind_speed_at_10m", "wind_speed", "m s-1", np.full(shape2, 5.0), height=10.0)
        # From the north: u = 0, v = -5.
        put("wind_direction_at_10m", "wind_from_direction", "degrees", np.zeros(shape2),
            height=10.0)


# ---------------------------------------------------------------------------
# a stand-in met_intermediate for the supplement
# ---------------------------------------------------------------------------
SUPPLEMENT_TARGET = LatLonTarget(south=-90.0, west=0.0, dlat=2.5, dlon=2.5, nx=144, ny=73)


def write_supplement(path: Path, hdate: str, *, drop: str = "") -> int:
    slab = np.ones((SUPPLEMENT_TARGET.ny, SUPPLEMENT_TARGET.nx), dtype=np.float32)
    rows = [("LANDSEA", "proprtn", 1.0), ("SEAICE", "proprtn", 0.0)]
    rows += [(n, "K", 283.0) for n in mo.SUPPLEMENT_SOIL if n.startswith("ST")]
    rows += [(n, "fraction", 0.3) for n in mo.SUPPLEMENT_SOIL if n.startswith("SM")]
    n = 0
    with path.open("wb") as handle:
        for name, units, value in rows:
            if name == drop:
                continue
            handle.write(wps_record_bytes(
                hdate=hdate, xfcst=0.0, map_source="NCEP GFS Analysis", field_name=name,
                units=units, description=name, level=200100.0, target=SUPPLEMENT_TARGET,
                values=slab * value))
            n += 1
    return n


FAKE_ENGINE = textwrap.dedent('''\
    import json, sys
    sys.path[:0] = {paths!r}
    from test_metoffice_intermediate import write_supplement
    argv = sys.argv[1:]
    hdate = argv[argv.index("--date") + 1]
    out = argv[argv.index("--out") + 1]
    stamp = {stamp!r} or hdate
    n = write_supplement(__import__("pathlib").Path(out), stamp, drop={drop!r})
    print(json.dumps({{"schema": {schema!r}, "matched_valid_times": [stamp],
                      "records_written": n, "map_source": {{"label": "ncep-gfs"}}}}))
''')


def fake_engine(tmp_path: Path, *, stamp: str = "", drop: str = "") -> Path:
    script = tmp_path / "fake_met_intermediate.py"
    script.write_text(FAKE_ENGINE.format(paths=[str(Path(__file__).parent), *sys.path],
                                         stamp=stamp, drop=drop,
                                         schema=door.ENGINE_RECEIPT_SCHEMA))
    launcher = tmp_path / "met_intermediate"
    launcher.write_text(f"#!/bin/sh\nexec {sys.executable} {script} \"$@\"\n")
    launcher.chmod(0o755)
    return launcher


def gdas_dir(tmp_path: Path, leads=(0, 3)) -> Path:
    root = tmp_path / "gdas"
    root.mkdir(exist_ok=True)
    for lead in leads:
        (root / f"gdas.t00z.pgrb2.0p25.f{lead:03d}").write_bytes(b"GRIB")
    return root


BOUNDS = (40.0, 10.0, 50.0, 25.0)


def request(tmp_path: Path, *, leads=(0, 3), engine: Path | None = None,
            bounds=BOUNDS, **kw) -> mo.MetOfficeRequest:
    fields = dict(
        source=mo.METOFFICE_SOURCES["met-office-global"], input_dir=tmp_path / "mo",
        cycle=CYCLE, leads=tuple(leads), interval_hours=3, bounds=bounds,
        out_dir=tmp_path / "out", supplement_dir=gdas_dir(tmp_path, leads),
        engine=engine or fake_engine(tmp_path),
    )
    fields.update(kw)
    return mo.MetOfficeRequest(**fields)


# ---------------------------------------------------------------------------
# names, winds, the Vtable cut
# ---------------------------------------------------------------------------
def test_file_names_put_the_valid_time_first() -> None:
    assert mo.file_name(CYCLE, 3, "wind_speed_at_10m") == \
        "20261011T0300Z-PT0003H00M-wind_speed_at_10m.nc"
    assert mo.file_name(datetime(2026, 10, 11, 12), 168, "x") == "20261018T1200Z-PT0168H00M-x.nc"


def test_winds_follow_the_meteorological_from_convention() -> None:
    speed = np.array([10.0, 10.0, 10.0, 10.0])
    u, v = mo.wind_components(speed, np.array([0.0, 90.0, 180.0, 270.0]))
    np.testing.assert_allclose(u, [0.0, -10.0, 0.0, 10.0], atol=1e-12)
    np.testing.assert_allclose(v, [-10.0, 0.0, 10.0, 0.0], atol=1e-12)


def test_the_supplement_vtable_keeps_soil_landsea_and_seaice_only() -> None:
    text = mo.supplement_vtable_text(door.packaged_vtable("Vtable.GFS.rw").read_text())
    rows = [line.split("|") for line in text.splitlines()
            if len(line.split("|")) >= 7 and not line.lstrip().startswith(("#", "-"))]
    names = {cells[4].strip() for cells in rows} - {"metgrid", "Name"}
    assert names == set(mo.SUPPLEMENT_FIELDS)


def test_the_source_is_admitted_and_dispatched() -> None:
    from woof.hex.hrrr_intermediate import INTERMEDIATE_SOURCES

    assert "met-office-global" in INTERMEDIATE_SOURCES
    with pytest.raises(GlobalIntermediateRefusal, match="met-office-global"):
        door.global_source("cmc")


# ---------------------------------------------------------------------------
# the crop
# ---------------------------------------------------------------------------
def test_a_crop_across_the_antimeridian_is_read_in_two_pieces() -> None:
    grid = mo.RegularGrid(south=-89.953125, west=-179.9296875, dlat=0.09375, dlon=0.140625,
                          nx=2560, ny=1920)
    crop = mo.crop_for(grid, (60.0, 178.0, 62.0, 182.0), halo_cells=3, what="test")
    assert len(crop.segments()) == 2
    target = crop.target()
    assert 177.0 < target.west < 178.0
    assert target.west + target.dlon * (target.nx - 1) > 182.0
    assert target.south <= 60.0 - 3 * grid.dlat + 1e-6


def test_a_crop_inside_the_grid_is_one_window_with_its_halo() -> None:
    grid = mo.RegularGrid(south=-89.953125, west=-179.9296875, dlat=0.09375, dlon=0.140625,
                          nx=2560, ny=1920)
    crop = mo.crop_for(grid, (71.88, -42.39, 72.12, -41.61), halo_cells=3, what="test")
    assert len(crop.segments()) == 1
    target = crop.target()
    assert target.south < 71.88 - 2 * grid.dlat and target.north > 72.12 + 2 * grid.dlat
    assert target.west < -42.39 - 2 * grid.dlon and target.east > -41.61 + 2 * grid.dlon


def test_a_crop_beyond_a_regional_grid_refuses() -> None:
    band = mo.RegularGrid(south=55.0, west=0.0, dlat=0.25, dlon=0.25, nx=1440, ny=133)
    with pytest.raises(GlobalIntermediateRefusal, match="does not cover"):
        mo.crop_for(band, (50.0, -45.0, 60.0, -40.0), halo_cells=3, what="gdas supplement")
    patch = mo.RegularGrid(south=30.0, west=-100.0, dlat=0.25, dlon=0.25, nx=8, ny=8)
    with pytest.raises(GlobalIntermediateRefusal, match="leaves"):
        mo.crop_for(patch, (30.8, -99.5, 31.0, -97.0), halo_cells=3, what="gdas supplement")


def test_a_crop_at_the_pole_clips_to_the_last_row() -> None:
    grid = mo.RegularGrid(south=-89.953125, west=-179.9296875, dlat=0.09375, dlon=0.140625,
                          nx=2560, ny=1920)
    crop = mo.crop_for(grid, (89.5, -10.0, 90.0, 10.0), halo_cells=3, what="test")
    assert crop.j1 == grid.ny - 1


def test_mesh_bounds_take_the_short_arc_across_the_prime_meridian(tmp_path: Path) -> None:
    path = tmp_path / "grid.nc"
    with netCDF4.Dataset(str(path), "w") as ds:
        ds.createDimension("nCells", 4)
        lat = ds.createVariable("latCell", "f8", ("nCells",))
        lon = ds.createVariable("lonCell", "f8", ("nCells",))
        lat[:] = np.radians([50.0, 51.0, 50.5, 50.2])
        lon[:] = np.radians([359.0, 1.0, 0.5, 359.5])
    bounds, basis = mo.mesh_bounds(path, margin_km=0.0)
    south, west, north, east = bounds
    assert south == pytest.approx(50.0) and north == pytest.approx(51.0)
    assert west == pytest.approx(-1.0) and east == pytest.approx(1.0)
    assert basis["points"] == 4


def test_a_polar_cap_mesh_takes_the_whole_circle(tmp_path: Path) -> None:
    path = tmp_path / "cap.nc"
    with netCDF4.Dataset(str(path), "w") as ds:
        ds.createDimension("nCells", 720)
        lat = ds.createVariable("latCell", "f8", ("nCells",))
        lon = ds.createVariable("lonCell", "f8", ("nCells",))
        lat[:] = np.radians(np.full(720, 88.0))
        lon[:] = np.radians(np.arange(720) * 0.5)
    bounds, _ = mo.mesh_bounds(path, margin_km=10.0)
    grid = mo.RegularGrid(south=-89.953125, west=-179.9296875, dlat=0.09375, dlon=0.140625,
                          nx=2560, ny=1920)
    crop = mo.crop_for(grid, bounds, halo_cells=3, what="test")
    assert crop.nx == grid.nx and crop.j1 <= grid.ny - 1


# ---------------------------------------------------------------------------
# finding the files
# ---------------------------------------------------------------------------
def test_missing_files_are_named_together(tmp_path: Path) -> None:
    write_run(tmp_path / "mo", leads=(0,), skip=("temperature_at_surface", "wind_speed_at_10m"))
    with pytest.raises(GlobalIntermediateRefusal) as refusal:
        mo.locate_files(tmp_path / "mo", CYCLE, (0, 3))
    text = str(refusal.value)
    assert "20261011T0000Z-PT0000H00M-temperature_at_surface.nc" in text
    assert "20261011T0000Z-PT0000H00M-wind_speed_at_10m.nc" in text
    assert "20261011T0300Z-PT0003H00M-temperature_on_pressure_levels.nc" in text
    assert "woof fetch --source met-office-global" in text


def test_the_old_height_spelling_and_a_run_subdirectory_are_read(tmp_path: Path) -> None:
    write_run(tmp_path / "mo" / "20261011T0000Z", old_height_name=True)
    found = mo.locate_files(tmp_path / "mo", CYCLE, (0, 3))
    assert found[3]["Z"].name.endswith("height_ASL_on_pressure_levels.nc")
    assert found[3]["OROG"].name == "20261011T0000Z-PT0000H00M-height_of_orography.nc"


def test_both_height_spellings_for_one_lead_are_ambiguous(tmp_path: Path) -> None:
    write_run(tmp_path / "mo", leads=(0,))
    write_run(tmp_path / "old", leads=(0,), old_height_name=True)
    old = mo.file_name(CYCLE, 0, "height_ASL_on_pressure_levels")
    (tmp_path / "mo" / old).write_bytes((tmp_path / "old" / old).read_bytes())
    with pytest.raises(GlobalIntermediateRefusal, match="more than one file"):
        mo.locate_files(tmp_path / "mo", CYCLE, (0,))


# ---------------------------------------------------------------------------
# validation: the supplement relaxation
# ---------------------------------------------------------------------------
def _write_mixed(path: Path, *, soil_grid: LatLonTarget) -> None:
    main = LatLonTarget(south=50.0, west=-10.0, dlat=0.25, dlon=0.25, nx=4, ny=3)
    hdate = "2026-10-11_00:00:00"
    with path.open("wb") as handle:
        def put(name, value, level, target, units="x"):
            handle.write(wps_record_bytes(
                hdate=hdate, xfcst=0.0, map_source="T", field_name=name, units=units,
                description=name, level=level, target=target,
                values=np.full((target.ny, target.nx), value, dtype=np.float32)))
        for level in (200100.0, 100000.0, 50000.0):
            for name, value in (("TT", 280.0), ("RH", 50.0), ("UU", 1.0), ("VV", 1.0)):
                put(name, value, level, main)
            if level != 200100.0:
                put("GHT", 1000.0, level, main)
        put("SOILHGT", 10.0, 200100.0, main)
        put("LANDSEA", 1.0, 200100.0, SUPPLEMENT_TARGET)
        put("SEAICE", 0.0, 200100.0, SUPPLEMENT_TARGET)
        for name in mo.SUPPLEMENT_SOIL:
            put(name, 283.0 if name.startswith("ST") else 0.3, 200100.0, soil_grid)


def test_declared_supplement_records_may_sit_on_their_own_grid(tmp_path: Path) -> None:
    path = tmp_path / "MET:2026-10-11_00"
    _write_mixed(path, soil_grid=SUPPLEMENT_TARGET)
    with pytest.raises(GlobalIntermediateRefusal, match="not on one regular lat-lon grid"):
        door.validate_intermediate(path, hdate="2026-10-11_00:00:00")
    facts = door.validate_intermediate(path, hdate="2026-10-11_00:00:00",
                                       supplement_fields=mo.SUPPLEMENT_FIELDS)
    assert facts["grid"]["nx"] == 4 and facts["supplement_grid"]["nx"] == 144
    assert facts["soil_layers"] == 4


def test_soil_off_the_land_mask_grid_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "MET:2026-10-11_00"
    other = LatLonTarget(south=-90.0, west=0.0, dlat=5.0, dlon=5.0, nx=72, ny=37)
    _write_mixed(path, soil_grid=other)
    with pytest.raises(GlobalIntermediateRefusal, match="supplement records are not on one"):
        door.validate_intermediate(path, hdate="2026-10-11_00:00:00",
                                   supplement_fields=mo.SUPPLEMENT_FIELDS)


# ---------------------------------------------------------------------------
# the whole door
# ---------------------------------------------------------------------------
def test_the_door_writes_validates_and_receipts_a_series(tmp_path: Path) -> None:
    write_run(tmp_path / "mo")
    receipt = mo.build_metoffice_intermediates(request(tmp_path), log=lambda _m: None)
    out = tmp_path / "out"
    assert [Path(f["path"]).name for f in receipt["files"]] == [
        "MET:2026-10-11_00", "MET:2026-10-11_03"]
    assert not list(out.glob("*.partial"))
    assert receipt["schema"] == mo.SCHEMA and receipt["source"] == "met-office-global"
    assert "CC BY-SA 4.0" in receipt["licence"]["met_office"]
    assert "share-alike" in receipt["licence"]["met_office"]
    assert "NCEP" in receipt["licence"]["supplement"]
    # RH tops at 1000 Pa, so 100 Pa is not written and the column tops at 10 hPa.
    assert receipt["levels"]["isobaric_pa"] == list(LEVELS_RH)
    assert receipt["levels"]["not_written_pa"]["T"] == [100.0]
    assert receipt["levels"]["screen_level_height_m"] == 1.5
    switches = receipt["init_switches"]
    assert switches["--nfglevels"] == len(LEVELS_RH) + 1
    assert switches["--nfgsoillevels"] == 4
    assert switches["--use-spechumd"] == "no"
    assert switches["--extrap-airtemp"] == "lapse-rate"
    later = receipt["files"][1]
    assert "step-0" in later["orography_from"]
    assert later["supplement"]["input"]["path"].endswith("gdas.t00z.pgrb2.0p25.f003")
    assert later["grid"]["dlat"] == pytest.approx(DLAT) and later["supplement_grid"]["dlat"] == 2.5
    assert len(receipt["supplement"]["vtable"]["sha256"]) == 64
    assert json.loads((out / "intermediate-receipt.json").read_text())["valid_times"] == [
        "2026-10-11_00:00:00", "2026-10-11_03:00:00"]
    assert (out / "met_intermediate.supplement.f003.log").is_file()


def test_the_records_carry_the_converted_values(tmp_path: Path) -> None:
    write_run(tmp_path / "mo", leads=(0,))
    mo.build_metoffice_intermediates(request(tmp_path, leads=(0,)), log=lambda _m: None)
    path = tmp_path / "out" / "MET:2026-10-11_00"
    values: dict[tuple[str, float], np.ndarray] = {}
    sources: dict[str, str] = {}
    with WpsIntermediateReader(path) as reader:
        for item in reader.iter_fields():
            values[(item.field, item.level)] = item.values
            sources[item.field] = item.map_source.strip()
    np.testing.assert_allclose(values[("UU", 50000.0)], 10.0, atol=1e-5)
    np.testing.assert_allclose(values[("VV", 50000.0)], 0.0, atol=1e-5)
    np.testing.assert_allclose(values[("UU", 200100.0)], 0.0, atol=1e-5)
    np.testing.assert_allclose(values[("VV", 200100.0)], -5.0, atol=1e-5)
    np.testing.assert_allclose(values[("RH", 50000.0)], 60.0, atol=1e-4)
    np.testing.assert_allclose(values[("RH", 200100.0)], 70.0, atol=1e-4)
    np.testing.assert_allclose(values[("TT", 200100.0)], 287.0)
    np.testing.assert_allclose(values[("SNOW", 200100.0)], 10.0, rtol=1e-5)
    np.testing.assert_allclose(values[("PMSL", 201300.0)], 101300.0)
    assert ("PSFC", 200100.0) not in values
    assert ("TT", 100.0) not in values
    assert sources["TT"] == mo.MAP_SOURCE and sources["LANDSEA"] == "NCEP GFS Analysis"
    records = inventory(path)["records"]
    main = {(r["nx"], r["ny"]) for r in records if r["field"] not in mo.SUPPLEMENT_FIELDS}
    supp = {(r["nx"], r["ny"]) for r in records if r["field"] in mo.SUPPLEMENT_FIELDS}
    assert len(main) == 1 and len(supp) == 1 and main != supp


def test_a_file_whose_coordinates_disagree_with_its_name_is_refused(tmp_path: Path) -> None:
    write_run(tmp_path / "mo", leads=(0,))
    name = mo.file_name(CYCLE, 0, "temperature_at_surface")
    _write_nc(tmp_path / "mo" / name, var="surface_temperature", units="K",
              values=np.full((NY, NX), 288.0), run=CYCLE, valid=CYCLE,
              time_override=CYCLE + timedelta(hours=6))
    with pytest.raises(GlobalIntermediateRefusal, match="mis-filed"):
        mo.build_metoffice_intermediates(request(tmp_path, leads=(0,)), log=lambda _m: None)
    assert not list((tmp_path / "out").glob("MET*"))


def test_an_unexpected_unit_is_refused(tmp_path: Path) -> None:
    write_run(tmp_path / "mo", leads=(0,))
    name = mo.file_name(CYCLE, 0, "pressure_at_mean_sea_level")
    _write_nc(tmp_path / "mo" / name, var="air_pressure_at_sea_level", units="bar",
              values=np.full((NY, NX), 1.013), run=CYCLE, valid=CYCLE)
    with pytest.raises(GlobalIntermediateRefusal, match="'bar'"):
        mo.build_metoffice_intermediates(request(tmp_path, leads=(0,)), log=lambda _m: None)


def test_a_supplement_missing_sea_ice_publishes_nothing(tmp_path: Path) -> None:
    write_run(tmp_path / "mo")
    with pytest.raises(GlobalIntermediateRefusal, match="carries no SEAICE"):
        mo.build_metoffice_intermediates(
            request(tmp_path, engine=fake_engine(tmp_path, drop="SEAICE")), log=lambda _m: None)
    assert not list((tmp_path / "out").glob("MET*"))
    assert not list((tmp_path / "out").glob(".supplement*"))


def test_a_supplement_for_another_time_is_refused(tmp_path: Path) -> None:
    write_run(tmp_path / "mo", leads=(0,))
    with pytest.raises(GlobalIntermediateRefusal, match="same valid time"):
        mo.build_metoffice_intermediates(
            request(tmp_path, leads=(0,), engine=fake_engine(tmp_path, stamp="2026-10-10_18:00:00")),
            log=lambda _m: None)


def test_a_missing_supplement_lead_names_the_fetch(tmp_path: Path) -> None:
    write_run(tmp_path / "mo")
    req = request(tmp_path)
    (req.supplement_dir / "gdas.t00z.pgrb2.0p25.f003").unlink()
    with pytest.raises(GlobalIntermediateRefusal, match="same-cycle gdas supplement.*f003"):
        mo.build_metoffice_intermediates(req, log=lambda _m: None)


def test_leads_past_gdas_need_the_gfs_supplement(tmp_path: Path) -> None:
    write_run(tmp_path / "mo", leads=(0,))
    with pytest.raises(GlobalIntermediateRefusal, match="--supplement-source gfs"):
        mo.build_metoffice_intermediates(request(tmp_path, leads=(0, 12)), log=lambda _m: None)


def test_leads_past_the_run_horizon_refuse(tmp_path: Path) -> None:
    req = request(tmp_path, leads=(0, 72), cycle=datetime(2026, 10, 11, 6))
    with pytest.raises(GlobalIntermediateRefusal, match="PT0069H"):
        mo.build_metoffice_intermediates(req, log=lambda _m: None)


def test_an_out_dir_holding_intermediates_is_refused(tmp_path: Path) -> None:
    write_run(tmp_path / "mo", leads=(0,))
    out = tmp_path / "out"
    out.mkdir()
    write_supplement(out / "OLD:2026-10-10_00", "2026-10-10_00:00:00")
    with pytest.raises(GlobalIntermediateRefusal, match="fresh --out-dir"):
        mo.build_metoffice_intermediates(request(tmp_path, leads=(0,)), log=lambda _m: None)


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------
def _args(tmp_path: Path, *extra: str) -> argparse.Namespace:
    return build_parser().parse_args([
        "intermediate", "--source", "met-office-global", "--input-dir", str(tmp_path / "mo"),
        "--cycle", "2026-10-11T00", "--supplement-dir", str(tmp_path / "gdas"),
        "--out-dir", str(tmp_path / "out"), *extra,
    ])


def test_the_parser_builds_a_request_from_a_point(tmp_path: Path) -> None:
    req = mo.request_from_arguments(_args(tmp_path, "--point", "72,-42", "--radius-km", "20",
                                          "--hours", "0-6"))
    assert req.leads == (0, 3, 6) and req.supplement_source == "gdas"
    south, west, north, east = req.bounds
    assert south < 72.0 < north and west < -42.0 < east


def test_no_box_is_refused(tmp_path: Path) -> None:
    with pytest.raises(GlobalIntermediateRefusal, match="no box was given"):
        mo.request_from_arguments(_args(tmp_path))


def test_a_mesh_grid_with_another_box_flag_is_refused(tmp_path: Path) -> None:
    with pytest.raises(GlobalIntermediateRefusal, match="the box comes from one"):
        mo.request_from_arguments(_args(tmp_path, "--mesh-grid", str(tmp_path / "g.nc"),
                                        "--point", "72,-42", "--radius-km", "5"))


def test_grib_row_flags_are_refused_for_the_met_office(tmp_path: Path) -> None:
    with pytest.raises(GlobalIntermediateRefusal, match="--grib-dir belong to another"):
        mo.request_from_arguments(_args(tmp_path, "--grib-dir", str(tmp_path),
                                        "--point", "72,-42", "--radius-km", "5"))


def test_met_office_flags_are_refused_for_other_rows(tmp_path: Path) -> None:
    from woof.hex import hrrr_intermediate as hi

    arguments = build_parser().parse_args([
        "intermediate", "--source", "gfs", "--grib-dir", str(tmp_path), "--cycle",
        "2026-09-24T12", "--out-dir", str(tmp_path / "o"), "--input-dir", str(tmp_path),
    ])
    with pytest.raises(GlobalIntermediateRefusal, match="--input-dir belong to another"):
        door.request_from_arguments(arguments)
    arguments = build_parser().parse_args([
        "intermediate", "--source", "hrrr", "--grib-dir", str(tmp_path), "--cycle",
        "2026-09-24T12", "--out-dir", str(tmp_path / "o"), "--mesh-grid", str(tmp_path),
        "--point", "40,-100", "--radius-km", "50",
    ])
    with pytest.raises(IntermediateRefusal, match="--mesh-grid belongs to --source met-office-global"):
        hi.request_from_arguments(arguments)


def test_the_cli_runs_the_door_end_to_end(tmp_path: Path, capsys) -> None:
    write_run(tmp_path / "mo")
    gdas_dir(tmp_path)
    code = hex_main([
        "intermediate", "--source", "met-office-global", "--input-dir", str(tmp_path / "mo"),
        "--cycle", "2026-10-11T00", "--hours", "0-3", "--supplement-dir", str(tmp_path / "gdas"),
        "--point", "45,15", "--radius-km", "100", "--decoder", str(fake_engine(tmp_path)),
        "--out-dir", str(tmp_path / "out"),
    ])
    assert code == 0
    printed = capsys.readouterr().out
    summary = json.loads(printed[printed.index("{\n"):])
    assert "--nfglevels 7 --nfgsoillevels 4 --use-spechumd no --extrap-airtemp lapse-rate" in summary["next"]
    assert "next_lbc" in summary


# ---------------------------------------------------------------------------
# the real engine over real (tiny) GDAS bytes, when it is built
# ---------------------------------------------------------------------------
def _built_engine() -> Path | None:
    try:
        from woof import bridges

        found = bridges.find_bridge("met_intermediate")
    except Exception:  # noqa: BLE001
        return None
    if found is None or not bridges.bridge_abi_matches("met_intermediate", Path(found))[0]:
        return None
    return Path(found)


@pytest.mark.skipif(_built_engine() is None or not GDAS_FIXTURES.is_dir(),
                    reason="met_intermediate (v3 contract) is not built here")
def test_the_real_engine_cuts_the_supplement_from_gdas(tmp_path: Path) -> None:
    vtable = tmp_path / "Vtable.supplement"
    vtable.write_text(mo.supplement_vtable_text(door.packaged_vtable("Vtable.GFS.rw").read_text()))
    lead = mo.run_supplement(
        engine=_built_engine(), vtable=vtable, source=door.GLOBAL_SOURCES["gdas"],
        grib=GDAS_FIXTURES / "nomads-gdas-20260729t12z-f000.grib2", hdate="2026-07-29_12:00:00",
        lead=0, bounds=(30.05, -99.95, 30.2, -99.8), work_dir=tmp_path, halo_cells=0)
    assert {r["field"] for r in lead.records} == set(mo.SUPPLEMENT_FIELDS)
    assert lead.provenance["engine_receipt"]["matched_valid_times"] == ["2026-07-29_12:00:00"]
    assert not list(tmp_path.glob(".supplement*"))
