"""Inventory frames and the RADM2 speciation (woof.chem_inventory_frames).

CPU only, no network: the EDGAR decode/remap path reads a tiny NetCDF this
module writes and zips, served through a stub transport.  The real archives
are exercised by tests/test_chem_inventory_edgar_live.py (opt-in).
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof import chem_inventory_frames as inv
from woof import chem_table
from woof.core.chem_radm2_emiss import AEROSOL_EMISSIONS, EMISSIONS, GAS_TERMS

ROOT = Path(__file__).resolve().parents[1]
WHEN = datetime(2026, 11, 7, tzinfo=timezone.utc)


def _generator():
    path = ROOT / "tools" / "chem_speciation" / "make_radm2_speciation.py"
    spec = importlib.util.spec_from_file_location("make_radm2_speciation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _spec():
    return inv.load_speciation("radm2")[0]


def _weights(name):
    return {t["field"]: t["weight"] for t in _spec()["outputs"][name]["terms"]}


# ----------------------------------------------------------------- speciation

def test_speciation_file_is_the_generators_output():
    text = (ROOT / "woof/data/chem/speciation/radm2.v1.json").read_text(
        encoding="utf-8")
    assert text == _generator().text()


def test_speciation_covers_exactly_what_launch_radm2_emiss_reads():
    doc = _spec()
    assert tuple(doc["outputs"]) == EMISSIONS
    assert set(doc["inputs"]) == {"co", "nox", "nh3", "nmvoc", "so2", "bc", "oc",
                                  "pm25", "pm10"}
    for name, row in doc["outputs"].items():
        gas = name in {e for _, e in GAS_TERMS}
        assert row["units"] == ("mol km-2 hr-1" if gas else "ug m-2 s-1")
        assert row["provenance"]
    zero = {n for n, r in doc["outputs"].items() if not r["terms"]}
    assert zero == {"e_hcl", "e_hono", "e_csl", "e_iso", "e_so4i", "e_so4j",
                    "e_no3i", "e_no3j", "e_naaj", "e_naai"}


def test_gas_unit_conversions():
    g = 1.0e6 * 3600.0 * 1000.0
    assert _weights("e_co")["co"] == pytest.approx(g / (12.011 + 15.999), rel=1e-9)
    assert _weights("e_nh3")["nh3"] == pytest.approx(g / (14.007 + 3 * 1.008), rel=1e-9)
    # e_so2 is converted exactly as the GOCART row converts EDGAR's SO2.
    gocart = json.loads((ROOT / "woof/data/chem/species/gocart.json").read_text())
    so2 = next(r for r in gocart["rows"] if r["name"] == "so2")
    edgar = next(e for e in so2["emissions"] if e["source"] == "edgar-v81")
    assert _weights("e_so2")["so2"] == pytest.approx(edgar["weight"], rel=1e-9)
    # NOx is NO2-equivalent mass; NO + NO2 conserve its moles, 90:10.
    m_no2 = 14.007 + 2 * 15.999
    no, no2 = _weights("e_no")["nox"], _weights("e_no2")["nox"]
    assert no + no2 == pytest.approx(g / m_no2, rel=1e-9)
    assert no / (no + no2) == pytest.approx(0.9, rel=1e-9)


def test_nmvoc_profile_sums():
    gen = _generator()
    doc = _spec()
    mol = doc["constants"]["nmvoc_mol_per_kg"]
    # Every class's weight is its mol per kg folded with the gas conversion.
    for cls, per_kg in mol.items():
        assert _weights(f"e_{cls}")["nmvoc"] == pytest.approx(
            per_kg / 1000.0 * 1.0e6 * 3600.0 * 1000.0, rel=1e-8)
    # The TAR industrial profile is complete, and the emitted mass is
    # what the classes carry back: sum over groups of wt x share.
    # (Table 4.7b's rounded wt% add to 100.1.)
    assert sum(g[2] for g in gen.NMVOC_GROUPS) == pytest.approx(100.1, abs=1e-9)
    mass = sum(g[2] / 100.0 * share * 1000.0 / g[5] * g[5] / 1000.0
               for g in gen.NMVOC_GROUPS if g[5] for share in g[6].values())
    assert mass == pytest.approx(doc["constants"]["nmvoc_mass_fraction_emitted"],
                                 abs=1e-6)
    assert 0.8 < mass < 0.95
    total_mol = sum(mol.values())
    # A plausible mean molar mass of the emitted NMVOC (40-80 g mol-1).
    assert 40.0 < mass * 1000.0 / total_mol < 80.0


def test_aerosol_splits_and_clamps():
    doc = _spec()["outputs"]
    assert _weights("e_eci")["bc"] + _weights("e_ecj")["bc"] == pytest.approx(1e9)
    assert _weights("e_eci")["bc"] == pytest.approx(0.2e9)
    assert _weights("e_orgi")["oc"] + _weights("e_orgj")["oc"] == pytest.approx(1.3e9)
    for name in ("e_pm25i", "e_pm25j", "e_pm_10"):
        assert doc[name]["clamp_at_zero"]
    share = _weights("e_pm25i")
    assert share == pytest.approx({"pm25": 0.2e9, "bc": -0.2e9, "oc": -0.26e9})
    assert _weights("e_pm_10") == pytest.approx({"pm10": 1e9, "pm25": -1e9})
    assert not any(doc[n]["clamp_at_zero"] for n in doc
                   if n not in ("e_pm25i", "e_pm25j", "e_pm_10"))


class _Stub:
    """A provider serving fixed fields (for speciation arithmetic only)."""

    def __init__(self, fields, shape):
        self.fields = fields
        self.shape = shape

    def field(self, source, field, valid_time=None):
        return np.broadcast_to(np.asarray(self.fields[field], np.float64),
                               self.shape)


def test_frames_are_the_speciation_applied_with_the_clamp():
    shape = (2, 3)
    fields = dict(co=1e-9, nox=5e-10, nh3=2e-11, nmvoc=4e-10, so2=1e-10,
                  bc=3e-11, oc=4e-11, pm25=[[1e-10] * 3, [2e-11] * 3],
                  pm10=[[1.5e-10] * 3, [1e-11] * 3])
    frames = inv.radm2_emission_frames(_Stub(fields, shape), "x", WHEN)
    assert tuple(frames) == EMISSIONS
    for name, frame in frames.items():
        assert frame.dtype == np.float32 and frame.shape == shape
        assert frame.flags.c_contiguous
    assert frames["e_co"][0, 0] == np.float32(1e-9 * _weights("e_co")["co"])
    fine = 1e-10 - 3e-11 - 1.3 * 4e-11
    assert frames["e_pm25j"][0, 0] == pytest.approx(0.8e9 * fine, rel=1e-6)
    # Row 1: PM2.5 below BC + OM, PM10 below PM2.5 -> clamped to zero.
    assert np.all(frames["e_pm25i"][1] == 0) and np.all(frames["e_pm25j"][1] == 0)
    assert np.all(frames["e_pm_10"][1] == 0)
    for name in ("e_hcl", "e_iso", "e_so4i", "e_naai"):
        assert not np.any(frames[name])
    assert set(AEROSOL_EMISSIONS) <= set(frames)


# ------------------------------------------------------------------ synthetic

def _domain(lat0=55.95, lon0=-3.19, ny=24, nx=30, step=0.08):
    lat = lat0 + step * (np.arange(ny) - ny / 2)[:, None] + 0 * np.arange(nx)
    lon = lon0 + 1.8 * step * (np.arange(nx) - nx / 2)[None, :] + 0 * lat
    return lat, lon


def test_synthetic_is_deterministic_nonzero_and_shape_agnostic():
    lat, lon = _domain()
    one = inv.radm2_emission_frames(inv.InventoryFrames(lat, lon),
                                    "synthetic-anthro", WHEN)
    two = inv.radm2_emission_frames(inv.InventoryFrames(lat, lon),
                                    "synthetic-anthro", WHEN)
    flat = inv.radm2_emission_frames(
        inv.InventoryFrames(lat.reshape(-1), lon.reshape(-1)), "synthetic-anthro")
    hexlike = inv.radm2_emission_frames(
        inv.InventoryFrames(lat.reshape(1, -1), lon.reshape(1, -1)),
        "synthetic-anthro")
    for name in EMISSIONS:
        assert one[name].tobytes() == two[name].tobytes()
        assert np.array_equal(one[name].reshape(-1), flat[name])
        assert hexlike[name].shape == (1, lat.size)
        assert np.array_equal(one[name].reshape(1, -1), hexlike[name])
    row = chem_table.catalog().sources["synthetic-anthro"]
    co = inv.InventoryFrames(lat, lon).field("synthetic-anthro", "co")
    peak = row.variables["co"]["peak"]
    assert co.min() >= 0.01 * peak and co.max() > 0.7 * peak
    # The urban centre (domain middle) beats the corner background tenfold.
    assert co[12, 15] > 10 * co[0, 0]
    for name in ("e_co", "e_no", "e_no2", "e_nh3", "e_so2", "e_hc3", "e_pm25i",
                 "e_pm25j", "e_pm_10", "e_eci", "e_orgj"):
        assert np.all(one[name] > 0), name


def test_synthetic_follows_any_domain_across_the_dateline():
    lat, lon = _domain(lat0=-41.0, lon0=179.9)
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    co = inv.InventoryFrames(lat, lon).field("synthetic-anthro", "co")
    peak = chem_table.catalog().sources["synthetic-anthro"].variables["co"]["peak"]
    assert co.max() > 0.7 * peak and co.min() > 0


# ----------------------------------------------------------- EDGAR (fixture)

GRID = dict(kind="regular_latlon", spacing_deg=0.1, global_=False,
            lat_first=50.05, lon_first=-8.95, nlat=20, nlon=30)


def _flux(lat, lon, field):
    scale = {"co": 1e-9, "nox": 5e-10, "nh3": 4e-11, "nmvoc": 6e-10,
             "so2": 1e-10, "bc": 1e-11, "oc": 2e-11, "pm25": 8e-11,
             "pm10": 1.2e-10}[field]
    return scale * (1.0 + 0.5 * np.sin(0.5 * lat) * np.cos(0.4 * lon))


def _write_archive(directory: Path, field: str, *, units="kg m-2 s-1",
                   values=None) -> Path:
    import netCDF4

    lat = GRID["lat_first"] + 0.1 * np.arange(GRID["nlat"])
    lon = GRID["lon_first"] + 0.1 * np.arange(GRID["nlon"])
    if values is None:
        values = _flux(lat[:, None], lon[None, :], field)
    nc = directory / f"{field}.nc"
    with netCDF4.Dataset(nc, "w") as ds:
        ds.createDimension("lat", lat.size)
        ds.createDimension("lon", lon.size)
        ds.createVariable("lat", "f8", ("lat",))[:] = lat
        ds.createVariable("lon", "f8", ("lon",))[:] = lon
        var = ds.createVariable("fluxes", "f4", ("lat", "lon"), fill_value=np.nan)
        var.units = units
        var[:] = values
    archive = directory / f"{field}_flx_nc.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.write(nc, f"{field}.nc")
        z.writestr("_readme.html", "<html></html>")
    nc.unlink()
    return archive


def _row(archives: dict) -> chem_table.SourceRow:
    variables = {}
    for field, path in archives.items():
        raw = path.read_bytes()
        variables[field] = {"selector": {
            "url": f"https://example.invalid/{path.name}", "archive": path.name,
            "archive_sha256": hashlib.sha256(raw).hexdigest(),
            "archive_bytes": len(raw), "member": f"{field}.nc",
            "variable": "fluxes"}, "units": "kg m-2 s-1"}
    grid = {k.rstrip("_"): v for k, v in GRID.items()}
    return chem_table.SourceRow(
        name="edgar-fixture", kind="emission", route=None, credential=None,
        format="netcdf", grid=grid, variables=variables,
        time={"cadence_s": 31536000, "latency_s": 0, "persistence": "annual"},
        remap="bilinear", licence="test", attribution="test",
        provenance="test", source_file="sources/edgar-v81.json")


@pytest.fixture
def edgar(tmp_path):
    server = tmp_path / "server"
    server.mkdir()
    archives = {f: _write_archive(server, f) for f in
                ("co", "nox", "nh3", "nmvoc", "so2", "bc", "oc", "pm25", "pm10")}
    row = _row(archives)
    calls = []

    def download(url, dest, size):
        calls.append(url)
        shutil.copyfile(server / Path(url).name, dest)

    catalog = SimpleNamespace(sources={row.name: row})
    return SimpleNamespace(row=row, catalog=catalog, cache=tmp_path / "cache",
                           download=download, calls=calls, server=server)


def _provider(edgar, lat, lon, **kw):
    kw.setdefault("download", edgar.download)
    return inv.InventoryFrames(lat, lon, catalog=edgar.catalog,
                               cache_dir=edgar.cache, **kw)


def test_fixture_archive_is_fetched_once_verified_and_remapped(edgar):
    lat, lon = _domain(lat0=51.0, lon0=-7.5, ny=8, nx=6, step=0.05)
    first = _provider(edgar, lat, lon).field("edgar-fixture", "co", WHEN)
    assert len(edgar.calls) == 1
    again = _provider(edgar, lat, lon, download=_no_network)
    assert np.array_equal(again.field("edgar-fixture", "co"), first)
    # The bilinear sample of a smooth field is within its curvature of truth.
    np.testing.assert_allclose(first, _flux(lat, lon, "co"), rtol=2e-3)


def _no_network(url, dest, size):
    raise OSError("network unreachable")


def test_missing_archive_without_network_is_refused_by_name(edgar):
    lat, lon = _domain(lat0=51.0, lon0=-7.5, ny=4, nx=4, step=0.05)
    provider = _provider(edgar, lat, lon, download=_no_network)
    with pytest.raises(inv.InventoryFrameError, match="nox_flx_nc.zip"):
        provider.field("edgar-fixture", "nox")


def test_changed_cached_bytes_are_refused(edgar):
    lat, lon = _domain(lat0=51.0, lon0=-7.5, ny=4, nx=4, step=0.05)
    _provider(edgar, lat, lon).field("edgar-fixture", "so2")
    cached = edgar.cache / "edgar-fixture" / "so2_flx_nc.zip"
    raw = bytearray(cached.read_bytes())
    raw[-1] ^= 0xFF
    cached.write_bytes(bytes(raw))
    with pytest.raises(inv.InventoryFrameError, match="sha256"):
        _provider(edgar, lat, lon).field("edgar-fixture", "so2")


def test_bad_units_nonfinite_values_and_outside_points_are_refused(edgar, tmp_path):
    lat, lon = _domain(lat0=51.0, lon0=-7.5, ny=4, nx=4, step=0.05)
    bad = tmp_path / "bad"
    bad.mkdir()
    holes = np.ones((GRID["nlat"], GRID["nlon"])) * 1e-10
    holes[3, 4] = np.nan
    archives = {"co": _write_archive(bad, "co", units="kg m-2 h-1"),
                "nox": _write_archive(bad, "nox", values=holes)}
    row = _row(archives)
    catalog = SimpleNamespace(sources={row.name: row})

    def download(url, dest, size):
        shutil.copyfile(bad / Path(url).name, dest)

    provider = inv.InventoryFrames(lat, lon, catalog=catalog,
                                   cache_dir=tmp_path / "c", download=download)
    with pytest.raises(inv.InventoryFrameError, match="units"):
        provider.field("edgar-fixture", "co")
    with pytest.raises(inv.InventoryFrameError, match="1 non-finite"):
        provider.field("edgar-fixture", "nox")
    with pytest.raises(inv.InventoryFrameError, match="no field 'xx'"):
        provider.field("edgar-fixture", "xx")
    far_lat, far_lon = _domain(lat0=40.0, lon0=-7.5, ny=2, nx=2)
    with pytest.raises(inv.InventoryFrameError, match="outside the source grid"):
        _provider(edgar, far_lat, far_lon).field("edgar-fixture", "co")


def test_fixture_frames_are_shape_agnostic(edgar):
    lat, lon = _domain(lat0=51.0, lon0=-7.5, ny=8, nx=6, step=0.05)
    grid2d = inv.radm2_emission_frames(_provider(edgar, lat, lon), "edgar-fixture")
    flat = inv.radm2_emission_frames(
        _provider(edgar, lat.reshape(-1), lon.reshape(-1)), "edgar-fixture")
    row = inv.radm2_emission_frames(
        _provider(edgar, lat.reshape(1, -1), lon.reshape(1, -1)), "edgar-fixture")
    for name in EMISSIONS:
        assert np.array_equal(grid2d[name].reshape(-1), flat[name])
        assert np.array_equal(grid2d[name].reshape(1, -1), row[name])


def test_receipt_names_archives_speciation_and_totals(edgar):
    lat, lon = _domain(lat0=51.0, lon0=-7.5, ny=8, nx=6, step=0.05)
    area = np.full(lat.shape, 9000.0 ** 2)
    provider = _provider(edgar, lat, lon, cell_area_m2=area)
    frames = inv.radm2_emission_frames(provider, "edgar-fixture", WHEN)
    receipt = inv.radm2_emission_receipt(provider, "edgar-fixture", frames, WHEN)
    assert receipt["source"] == "edgar-fixture" and not receipt["synthetic"]
    assert set(receipt["archives"]) == set(_spec()["inputs"])
    assert receipt["archives"]["co"]["archive_sha256"] == \
        edgar.row.variables["co"]["selector"]["archive_sha256"]
    assert receipt["speciation_sha256"] == hashlib.sha256(
        (ROOT / "woof/data/chem/speciation/radm2.v1.json").read_bytes()).hexdigest()
    assert receipt["valid_time"] == WHEN.isoformat()
    co = provider.field("edgar-fixture", "co")
    assert receipt["totals"]["e_co"] == pytest.approx(
        float(np.sum(co * area)) * 3600.0 * 1000.0 / 28.01, rel=1e-6)
    assert receipt["totals"]["e_hcl"] == 0.0
    synthetic = inv.InventoryFrames(lat, lon)
    sframes = inv.radm2_emission_frames(synthetic, "synthetic-anthro")
    sreceipt = inv.radm2_emission_receipt(synthetic, "synthetic-anthro", sframes)
    assert sreceipt["synthetic"] and sreceipt["archives"] == {}
    assert sreceipt["totals"]["e_co"] is None


# --------------------------------------------------------------------- remap

def test_bilinear_is_exact_for_linear_fields_and_conserves_smooth_mass():
    grid = {k.rstrip("_"): v for k, v in GRID.items()}
    lat = grid["lat_first"] + 0.1 * np.arange(grid["nlat"])
    lon = grid["lon_first"] + 0.1 * np.arange(grid["nlon"])
    plane = 2.0 + 0.3 * lat[:, None] - 0.2 * lon[None, :]
    rng = np.random.default_rng(7)
    plat = rng.uniform(lat[0], lat[-1], 4000)
    plon = rng.uniform(lon[0], lon[-1], 4000)
    np.testing.assert_allclose(inv.bilinear(plane, grid, plat, plon),
                               2.0 + 0.3 * plat - 0.2 * plon, rtol=1e-12)
    # Mass sanity: a dense equal-spacing target over the interior carries
    # the source's area integral of a smooth (non-linear) density.
    density = 1.0 + 0.5 * np.sin(2 * lat)[:, None] * np.cos(3 * lon)[None, :]
    tlat, tlon = np.meshgrid(np.linspace(lat[0], lat[-1], 381),
                             np.linspace(lon[0], lon[-1], 581), indexing="ij")
    target = inv.bilinear(density, grid, tlat, tlon)
    weight = np.cos(np.radians(tlat))
    source_weight = np.cos(np.radians(lat))[:, None] * np.ones_like(density)
    trap = np.ones_like(density)
    trap[[0, -1], :] *= 0.5
    trap[:, [0, -1]] *= 0.5
    source_mass = np.sum(density * source_weight * trap) / np.sum(source_weight * trap)
    target_mass = np.sum(target * weight) / np.sum(weight)
    assert target_mass == pytest.approx(source_mass, rel=2e-4)


def test_global_grid_wraps_in_longitude():
    grid = dict(spacing_deg=90.0, lat_first=-45.0, lon_first=-135.0, nlat=2,
                nlon=4, **{"global": True})
    values = np.array([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]])
    got = inv.bilinear(values, grid, np.array([0.0, 0.0, 80.0]),
                       np.array([180.0, -180.0, 135.0]))
    np.testing.assert_allclose(got, [1.5, 1.5, 3.0])


# ---------------------------------------------------------------------- door

def _species(processes):
    return SimpleNamespace(processes=tuple(processes))


@pytest.mark.parametrize("source", ["edgar-v81", "synthetic-anthro"])
def test_door_serves_radm2_rows_reading_speciation_fields(source):
    row = chem_table.catalog().sources[source]
    assert inv.serves_source(row)
    radm2 = _species(["emission.radm2", "drydep.wesely"])
    assert inv.serves_emission(radm2, {"source": source, "field": "e_co"}, row)
    assert inv.serves_emission(radm2, {"source": source, "field": "e_pm25j"}, row)
    # Not an output of the speciation, or not an inventory process: unserved.
    assert not inv.serves_emission(radm2, {"source": source, "field": "co"}, row)
    gocart = _species(["emission.inventory", "chem.sulfur"])
    assert not inv.serves_emission(gocart, {"source": source, "field": "e_co"}, row)


def test_door_does_not_serve_sources_missing_a_speciation_input():
    row = chem_table.catalog().sources["edgar-v81"]
    partial = dataclasses.replace(row, variables={
        k: v for k, v in row.variables.items() if k != "nmvoc"})
    assert inv.serves_source(partial)
    assert not inv.serves_emission(_species(["emission.radm2"]),
                                   {"source": "edgar-v81", "field": "e_co"}, partial)
    rave = chem_table.catalog().sources["rave-3km"]
    assert not inv.serves_source(rave)


def test_every_edgar_selector_is_pinned_over_https():
    row = chem_table.catalog().sources["edgar-v81"]
    assert set(row.variables) == set(_spec()["inputs"])
    for field, spec in row.variables.items():
        selector = spec["selector"]
        assert selector["url"].startswith(
            "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/EDGAR/datasets/"
            "v81_FT2022_AP_new/")
        assert selector["url"].endswith("/TOTALS/flx_nc/" + selector["archive"])
        assert len(selector["archive_sha256"]) == 64
        assert 15_000_000 < selector["archive_bytes"] < 17_000_000
        assert selector["member"] == selector["archive"].replace("_nc.zip", ".nc")
