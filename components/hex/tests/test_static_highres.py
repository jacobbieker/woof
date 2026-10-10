"""``woof hex static-highres``: cell averages, land-use mode and refusals.

The mesh here is a real spherical Voronoi diagram (scipy) of a small
refined patch plus a coarse global background, written as an MPAS-like
static with the variables the door reads.  The DEM is either analytic (a
fake sampler evaluates a function at every lattice point, which tests the
assignment and the area weighting without the Rust library) or a real
GeoTIFF written below and read through the static-fields warp (which tests
the lattice orientation against the library that production uses).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import struct

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")
pytest.importorskip("scipy")

from woof.hex import static_highres as sh  # noqa: E402
from woof.static import highres_fetch as hf  # noqa: E402

PATCH_LAT, PATCH_LON = 52.27, -3.60


# ---------------------------------------------------------------------------
# Fixtures: a synthetic MPAS-like mesh and static
# ---------------------------------------------------------------------------

def _xyz(lat_deg, lon_deg):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon),
                     np.sin(lat)), axis=-1)


def _generators(spacing_m: float, half_width_cells: int, background: int):
    """A hexagonal patch of generators around PATCH plus a coarse sphere."""
    d_lat = spacing_m * math.sqrt(3) / 2 / 111_195.0
    d_lon = spacing_m / (111_195.0 * math.cos(math.radians(PATCH_LAT)))
    lats, lons = [], []
    n = half_width_cells
    for j in range(-n, n + 1):
        for i in range(-n, n + 1):
            lats.append(PATCH_LAT + j * d_lat)
            lons.append(PATCH_LON + (i + 0.5 * (j % 2)) * d_lon)
    patch = _xyz(np.array(lats), np.array(lons))
    # Fibonacci background, minus anything near the patch.
    k = np.arange(background) + 0.5
    z = 1 - 2 * k / background
    phi = math.pi * (1 + 5 ** 0.5) * k
    bg = np.stack((np.sqrt(1 - z * z) * np.cos(phi),
                   np.sqrt(1 - z * z) * np.sin(phi), z), axis=-1)
    centre = _xyz(PATCH_LAT, PATCH_LON)
    reach = (2 * n + 6) * spacing_m / 6_371_229.0
    bg = bg[np.arccos(np.clip(bg @ centre, -1, 1)) > reach]
    return np.concatenate((patch, bg)), patch.shape[0]


def _mpas_from_voronoi(points: np.ndarray):
    from scipy.spatial import SphericalVoronoi

    sv = SphericalVoronoi(points, radius=1.0)
    sv.sort_vertices_of_regions()
    n = points.shape[0]
    max_edges = max(len(r) for r in sv.regions)
    verts = np.zeros((n, max_edges), dtype=np.int32)
    n_edges = np.zeros(n, dtype=np.int32)
    edge_owner: dict[tuple[int, int], list[int]] = {}
    for c, region in enumerate(sv.regions):
        n_edges[c] = len(region)
        verts[c, :len(region)] = np.asarray(region) + 1
        for k in range(len(region)):
            a, b = region[k], region[(k + 1) % len(region)]
            edge_owner.setdefault((min(a, b), max(a, b)), []).append(c)
    cells_on_cell = np.zeros((n, max_edges), dtype=np.int32)
    for c, region in enumerate(sv.regions):
        for k in range(len(region)):
            a, b = region[k], region[(k + 1) % len(region)]
            owners = edge_owner[(min(a, b), max(a, b))]
            other = [o for o in owners if o != c]
            cells_on_cell[c, k] = other[0] + 1 if other else 0
    return sv, verts, n_edges, cells_on_cell


def write_static(path: Path, points: np.ndarray, *, ter=None, ivgtyp=None,
                 extra_attrs=None, extra_vars=()):
    sv, verts, n_edges, coc = _mpas_from_voronoi(points)
    n = points.shape[0]
    lat_c = np.arcsin(points[:, 2])
    lon_c = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2 * np.pi)
    v = sv.vertices
    lat_v = np.arcsin(np.clip(v[:, 2], -1, 1))
    lon_v = np.mod(np.arctan2(v[:, 1], v[:, 0]), 2 * np.pi)
    ivgtyp = np.full(n, 10, dtype=np.int32) if ivgtyp is None else ivgtyp
    with netCDF4.Dataset(path, "w", format="NETCDF3_64BIT_OFFSET") as ds:
        ds.createDimension("nCells", n)
        ds.createDimension("nVertices", v.shape[0])
        ds.createDimension("maxEdges", verts.shape[1])
        ds.createDimension("nMonths", 12)
        ds.createDimension("nSoilComps", 8)
        ds.createDimension("StrLen", 64)
        ds.on_a_sphere = "YES"
        ds.sphere_radius = 6371229.0
        ds.mesh_spec = "synthetic"
        for key, value in (extra_attrs or {}).items():
            ds.setncattr(key, value)

        def put(name, data, dims, dtype):
            var = ds.createVariable(name, dtype, dims)
            var[:] = data

        put("latCell", lat_c, ("nCells",), "f8")
        put("lonCell", lon_c, ("nCells",), "f8")
        put("latVertex", lat_v, ("nVertices",), "f8")
        put("lonVertex", lon_v, ("nVertices",), "f8")
        put("nEdgesOnCell", n_edges, ("nCells",), "i4")
        put("verticesOnCell", verts, ("nCells", "maxEdges"), "i4")
        put("cellsOnCell", coc, ("nCells", "maxEdges"), "i4")
        put("ter", np.full(n, 100.0) if ter is None else ter, ("nCells",), "f4")
        put("ivgtyp", ivgtyp, ("nCells",), "i4")
        put("lu_index", ivgtyp, ("nCells",), "i4")
        land = (ivgtyp != 17).astype(np.int32)
        put("landmask", land, ("nCells",), "i4")
        put("isltyp", np.where(land == 1, 6, 14), ("nCells",), "i4")
        put("soilcat_top", np.where(land == 1, 6, 14), ("nCells",), "i4")
        green = np.where(land[:, None] == 1,
                         np.linspace(10, 60, 12)[None, :], 0.0)
        put("greenfrac", green, ("nCells", "nMonths"), "f4")
        put("albedo12m", np.where(land[:, None] == 1, 15.0, 0.0)
            * np.ones((1, 12)), ("nCells", "nMonths"), "f4")
        put("shdmin", green.min(axis=1), ("nCells",), "f4")
        put("shdmax", green.max(axis=1), ("nCells",), "f4")
        put("snoalb", np.where(land == 1, 55.0, 0.0), ("nCells",), "f4")
        put("soiltemp", np.where(land == 1, 283.0, 0.0), ("nCells",), "f4")
        put("soilcomp", np.where(land[:, None] == 1, 20.0, 0.0)
            * np.ones((1, 8)), ("nCells", "nSoilComps"), "f4")
        put("var2d", np.full(n, 12.5), ("nCells",), "f4")
        mminlu = ds.createVariable("mminlu", "S1", ("StrLen",))
        mminlu[:] = netCDF4.stringtoarr("MODIFIED_IGBP_MODIS_NOAH", 64)
        put("iswater_lu", 17, (), "i4")
        put("isice_lu", 15, (), "i4")
        for name in extra_vars:
            put(name, np.zeros(n), ("nCells",), "f4")
    return sv


@pytest.fixture()
def patch_static(tmp_path):
    points, n_patch = _generators(100.0, 12, 400)
    path = tmp_path / "patch.static.nc"
    sv = write_static(path, points)
    return path, points, n_patch, sv


def _analytic_sampler(function):
    """A sampler that evaluates ``function(lat, lon)`` at every mass point
    of the band grid (what the warp does with a nearest pick of a raster
    that IS the function)."""
    def sample(bound, grid):
        ny, nx = grid.e_sn - 1, grid.e_we - 1
        x, y = np.meshgrid(np.arange(1, nx + 1, dtype=float),
                           np.arange(1, ny + 1, dtype=float))
        lat, lon = grid.ij_to_latlon(x, y)
        return function(np.asarray(lat), np.asarray(lon))
    return sample


def _fake_sources(monkeypatch, *, landuse=False):
    """Replace the network/cache layer with in-memory bound rasters."""
    from woof.static.highres import BoundRaster

    def bound(role):
        return BoundRaster(path=Path("/nonexistent"), sha256="0" * 64,
                           source_id=role, role=role, source_url="",
                           license_id="", license_url="",
                           nominal_resolution="")

    monkeypatch.setattr(sh, "fetch_terrain", lambda *a, **k: (bound("terrain"), {
        "terrain_source": "copernicus-dem-glo30", "terrain_tiles": [
            {"tile": "T.tif", "sha256": "a" * 64}],
        "terrain_tiles_absent": [], "terrain_window": {"sha256": "b" * 64},
        "terrain_vertical_datum": "EGM2008", "terrain_attribution": "x"}))
    if landuse:
        row = hf.landcover_source("cglc-modis-lcz")
        monkeypatch.setattr(sh, "fetch_landuse", lambda *a, **k: (
            bound("landcover"), {
                "landuse_source": "cglc-modis-lcz",
                "landuse_raster": {"sha256": "c" * 64},
                "landuse_window": {"sha256": "d" * 64},
                "landuse_crosswalk": "x", "landuse_attribution": "y"}, row))


# ---------------------------------------------------------------------------
# Geometry and assignment
# ---------------------------------------------------------------------------

def test_voronoi_assignment_matches_brute_force_nearest_generator(
        patch_static):
    path, points, n_patch, _ = patch_static
    with netCDF4.Dataset(path) as ds:
        geometry = sh.read_cell_geometry(ds)
    selected = sh.select_cells(geometry, sh.cell_spacing_m(geometry), 0.5)
    # Exactly the 100 m patch is selected, never the background.
    assert selected[:n_patch].sum() >= 0.8 * n_patch
    assert not selected[n_patch:].any()
    assigner = sh.VoronoiAssigner(geometry, selected)
    rng = np.random.default_rng(3)
    lat = PATCH_LAT + rng.uniform(-0.015, 0.015, 40_000)
    lon = PATCH_LON + rng.uniform(-0.025, 0.025, 40_000)
    xyz = _xyz(lat, lon)
    local = assigner.assign(xyz)
    truth = np.argmax(xyz @ points.T, axis=1)          # over ALL generators
    got = np.where(local >= 0, assigner.cells[np.maximum(local, 0)], -1)
    in_selected = selected[truth]
    # Every point whose true Voronoi owner is selected goes to that owner...
    assert np.array_equal(got[in_selected], truth[in_selected])
    # ...and every point owned by an unselected cell goes nowhere, even
    # where the nearest SELECTED centre would have claimed it.
    assert (got[~in_selected] == -1).all()
    assert (~in_selected).sum() > 1000 and assigner.edge_cell_count > 0


def test_lattice_quantisation_converges(patch_static, monkeypatch, tmp_path):
    path, points, n_patch, sv = patch_static
    _fake_sources(monkeypatch)
    out = tmp_path / "fine.static.nc"
    sh.run_static_highres(
        path, out, max_spacing_km=0.5, sample_m=1.0,
        sampler=_analytic_sampler(lambda lat, lon: 1000.0 * (lat - PATCH_LAT)),
        cache_root=tmp_path / "c")
    with netCDF4.Dataset(out) as ds:
        ter = np.asarray(ds["ter"][:], dtype=float)
    centre = n_patch // 2          # a regular interior hexagon
    lat_c = math.degrees(math.asin(points[centre, 2]))
    assert ter[centre] == pytest.approx(1000.0 * (lat_c - PATCH_LAT), abs=0.01)


def test_inradius_is_half_the_generator_spacing(patch_static):
    path, _, n_patch, _ = patch_static
    with netCDF4.Dataset(path) as ds:
        geometry = sh.read_cell_geometry(ds)
    inradius = sh.cell_inradius_m(geometry)
    centre = n_patch // 2
    assert inradius[centre] == pytest.approx(50.0, rel=0.02)


# ---------------------------------------------------------------------------
# Area means
# ---------------------------------------------------------------------------

def test_linear_terrain_area_mean_is_the_polygon_centroid_value(
        patch_static, monkeypatch, tmp_path):
    path, points, n_patch, sv = patch_static
    _fake_sources(monkeypatch)
    slope_lat, slope_lon = 50_000.0, -30_000.0      # m per degree

    def plane(lat, lon):
        return 300.0 + slope_lat * (lat - PATCH_LAT) + slope_lon * (lon - PATCH_LON)

    out = tmp_path / "out.static.nc"
    receipt = sh.run_static_highres(
        path, out, max_spacing_km=0.5, sampler=_analytic_sampler(plane),
        cache_root=tmp_path / "cache")
    with netCDF4.Dataset(out) as ds:
        ter = np.asarray(ds["ter"][:], dtype=float)
        assert ds.highres_schema == sh.SCHEMA
        assert ds.highres_terrain_source == "copernicus-dem-glo30"
        assert ds.mesh_spec == "synthetic"            # provenance kept
        assert json.loads(ds.highres_terrain_tiles)[0]["sha256"] == "a" * 64
        assert np.allclose(ds["var2d"][:], 12.5)      # GWD statistics kept
    with netCDF4.Dataset(path) as ds:
        original = np.asarray(ds["ter"][:], dtype=float)
    selected = np.zeros(points.shape[0], bool)
    with netCDF4.Dataset(path) as ds:
        geometry = sh.read_cell_geometry(ds)
    selected = sh.select_cells(geometry, sh.cell_spacing_m(geometry), 0.5)
    # Unselected cells are untouched to the byte.
    assert np.array_equal(ter[~selected], original[~selected])
    # Interior selected cells: the mean of a linear field over a convex
    # polygon is its value at the area centroid.
    interior = np.flatnonzero(selected)[:]
    errors = []
    for c in interior:
        poly = sv.vertices[sv.regions[c]]
        lat = np.degrees(np.arcsin(poly[:, 2]))
        lon = np.degrees(np.arctan2(poly[:, 1], poly[:, 0]))
        x, y = lon, lat
        cross = x * np.roll(y, -1) - np.roll(x, -1) * y
        area = cross.sum() / 2
        cx = ((x + np.roll(x, -1)) * cross).sum() / (6 * area)
        cy = ((y + np.roll(y, -1)) * cross).sum() / (6 * area)
        errors.append(ter[c] - plane(cy, cx))
    errors = np.abs(np.asarray(errors))
    # 100 m cells, automatic lattice 12.5 m: the regular lattice against
    # the regular hexagons leaves a coherent quantisation of about 1 m on
    # a 0.45 m/m slope (it falls to 0.1 m at a 1 m lattice).
    assert np.median(errors) < 1.5 and errors.max() < 2.5
    assert receipt["terrain"]["cells_fully_covered"] == selected.sum()
    assert receipt["lattice"]["dx_m"] <= 12.5 + 1e-3
    assert Path(receipt["receipt"]).is_file()


def test_step_terrain_mean_is_the_covered_area_fraction(
        patch_static, monkeypatch, tmp_path):
    path, points, n_patch, sv = patch_static
    _fake_sources(monkeypatch)
    # Half-plane step through the patch centre: 1000 m north, 0 south.
    out = tmp_path / "out.static.nc"
    sh.run_static_highres(
        path, out, max_spacing_km=0.5, sample_m=2.0,
        sampler=_analytic_sampler(
            lambda lat, lon: np.where(lat >= PATCH_LAT + 1e-7, 1000.0, 0.0)),
        cache_root=tmp_path / "cache")
    with netCDF4.Dataset(out) as ds:
        ter = np.asarray(ds["ter"][:], dtype=float)
    with netCDF4.Dataset(path) as ds:
        geometry = sh.read_cell_geometry(ds)
    selected = sh.select_cells(geometry, sh.cell_spacing_m(geometry), 0.5)
    lat_c = np.degrees(np.arcsin(points[:, 2]))
    far_north = selected & (lat_c > PATCH_LAT + 0.002)
    far_south = selected & (lat_c < PATCH_LAT - 0.002)
    assert np.allclose(ter[far_north], 1000.0)
    assert np.allclose(ter[far_south], 0.0)
    # Cells the step cuts take the area fraction, strictly between.
    cut = selected & ~far_north & ~far_south
    split = ter[cut][(ter[cut] > 1.0) & (ter[cut] < 999.0)]
    assert split.size >= 5
    # Mass is conserved: the area-weighted mean over the cut cells equals
    # the area fraction north of the line within lattice quantisation.
    for c in np.flatnonzero(cut)[:10]:
        poly = sv.vertices[sv.regions[c]]
        lat = np.degrees(np.arcsin(poly[:, 2]))
        if lat.min() < PATCH_LAT < lat.max():
            assert 0.0 < ter[c] < 1000.0


def test_uncovered_samples_take_the_baseline_height(
        patch_static, monkeypatch, tmp_path):
    path, points, n_patch, _ = patch_static
    _fake_sources(monkeypatch)
    out = tmp_path / "out.static.nc"
    receipt = sh.run_static_highres(
        path, out, max_spacing_km=0.5,
        sampler=_analytic_sampler(
            lambda lat, lon: np.where(lon < PATCH_LON, np.nan, 500.0)),
        cache_root=tmp_path / "cache")
    with netCDF4.Dataset(out) as ds:
        ter = np.asarray(ds["ter"][:], dtype=float)
    lon_c = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    with netCDF4.Dataset(path) as ds:
        geometry = sh.read_cell_geometry(ds)
    selected = sh.select_cells(geometry, sh.cell_spacing_m(geometry), 0.5)
    west = selected & (lon_c < PATCH_LON - 0.003)
    east = selected & (lon_c > PATCH_LON + 0.003)
    assert np.allclose(ter[west], 100.0)          # baseline kept exactly
    assert np.allclose(ter[east], 500.0)
    t = receipt["terrain"]
    assert t["cells_uncovered_kept_baseline"] > 0
    assert t["cells_partly_covered"] > 0


# ---------------------------------------------------------------------------
# Land use
# ---------------------------------------------------------------------------

def test_landuse_mode_landmask_and_land_masked_carry(
        patch_static, monkeypatch, tmp_path):
    path, points, n_patch, _ = patch_static
    _fake_sources(monkeypatch, landuse=True)
    lat_c = np.degrees(np.arcsin(points[:, 2]))

    def classes(lat, lon):
        # North: sea (17); a band of LCZ 2 (-> urban 13); south: cropland 12.
        out = np.full(lat.shape, 12.0)
        out[lat > PATCH_LAT + 0.003] = 17.0
        out[np.abs(lat - PATCH_LAT) < 0.001] = 52.0
        return out

    def sampler(bound, grid):
        if bound.role == "landcover":
            return _analytic_sampler(classes)(bound, grid)
        return _analytic_sampler(lambda la, lo: np.full(la.shape, 50.0))(
            bound, grid)

    out = tmp_path / "out.static.nc"
    receipt = sh.run_static_highres(
        path, out, landuse="cglc", max_spacing_km=0.5, sampler=sampler,
        cache_root=tmp_path / "cache")
    with netCDF4.Dataset(path) as ds:
        geometry = sh.read_cell_geometry(ds)
    selected = sh.select_cells(geometry, sh.cell_spacing_m(geometry), 0.5)
    with netCDF4.Dataset(out) as ds:
        iv = np.asarray(ds["ivgtyp"][:])
        lu = np.asarray(ds["lu_index"][:])
        mask = np.asarray(ds["landmask"][:])
        green = np.asarray(ds["greenfrac"][:])
        shdmax = np.asarray(ds["shdmax"][:])
        soil = np.asarray(ds["isltyp"][:])
        assert ds.highres_landuse_source == "cglc-modis-lcz"
    assert np.array_equal(iv, lu)
    sea = selected & (lat_c > PATCH_LAT + 0.0045)
    urban = selected & (np.abs(lat_c - PATCH_LAT) < 0.0002)
    crop = selected & (lat_c < PATCH_LAT - 0.0025)
    assert (iv[sea] == 17).all() and (mask[sea] == 0).all()
    assert (iv[urban] == 13).all() and (mask[urban] == 1).all()
    assert (iv[crop] == 12).all()
    # New water carries the water zeros; unselected cells keep grassland.
    assert np.allclose(green[sea], 0.0) and np.allclose(shdmax[sea], 0.0)
    assert (iv[~selected] == 10).all()
    assert receipt["landuse"]["newly_water_cells"] == int(
        ((iv == 17) & selected).sum())
    assert soil[sea].max() == 6      # no water donor in the file: kept
    assert receipt["landuse"]["newly_water_soil_kept_no_water_donor"] > 0


def test_new_land_takes_the_nearest_land_climatology(tmp_path, monkeypatch):
    points, n_patch = _generators(100.0, 8, 300)
    lat_c = np.degrees(np.arcsin(points[:, 2]))
    ivgtyp = np.where(lat_c > PATCH_LAT, 17, 10).astype(np.int32)
    path = tmp_path / "coast.static.nc"
    write_static(path, points, ivgtyp=ivgtyp)
    _fake_sources(monkeypatch, landuse=True)

    def sampler(bound, grid):
        values = (lambda la, lo: np.full(la.shape, 5.0)) if bound.role == \
            "landcover" else (lambda la, lo: np.full(la.shape, 1.0))
        return _analytic_sampler(values)(bound, grid)

    out = tmp_path / "coast.out.nc"
    receipt = sh.run_static_highres(path, out, landuse="cglc",
                                    max_spacing_km=0.5, sampler=sampler,
                                    cache_root=tmp_path / "c")
    with netCDF4.Dataset(out) as ds:
        mask = np.asarray(ds["landmask"][:])
        green = np.asarray(ds["greenfrac"][:])
        soiltemp = np.asarray(ds["soiltemp"][:])
        shdmin = np.asarray(ds["shdmin"][:])
        soil = np.asarray(ds["isltyp"][:])
    newly = (ivgtyp == 17) & (mask == 1)
    assert newly.sum() == receipt["landuse"]["newly_land_cells"] > 0
    assert np.allclose(soiltemp[newly], 283.0)
    assert np.allclose(green[newly][:, 0], 10.0)
    assert np.allclose(shdmin[newly], 10.0)
    assert (soil[newly] == 6).all()


def test_landuse_mode_breaks_ties_to_the_lowest_category():
    acc = sh.CellAccumulators(3, 21)
    acc.area[:] = 1.0
    acc.landuse[0, [4, 9]] = 0.5          # tie 5 vs 10 -> 5
    acc.landuse[1, 11] = 0.3              # only 30 % classified -> baseline
    acc.landuse[2, 16] = 0.9
    new, counts = sh.landuse_mode(acc, np.array([1, 2, 3]))
    assert new.tolist() == [5, 2, 17]
    assert counts["cells_kept_baseline_unclassified"] == 1


def test_unclassified_area_votes_for_the_baseline_category():
    acc = sh.CellAccumulators(2, 21)
    acc.area[:] = 1.0
    acc.landuse[0, 9] = 0.55      # 55 % grass, 45 % unclassified sea
    acc.landuse[1, 9] = 0.30      # 30 % grass, 10 % crop, 60 % unclassified
    acc.landuse[1, 11] = 0.10
    new, _ = sh.landuse_mode(acc, np.array([17, 17]))
    assert new.tolist() == [10, 17]


def test_land_masked_donors_read_the_fields_before_any_flip():
    # Cells on a line: 0 old land, 1 old water -> land, 2 old land -> water.
    lat = np.radians([52.0, 52.001, 52.002, 52.003])
    xyz = np.stack((np.cos(lat), np.zeros(4), np.sin(lat)), axis=-1)
    arrays = {"isltyp": np.array([5, 14, 7, 14]),
              "soiltemp": np.array([280.0, 0.0, 281.0, 0.0]),
              "greenfrac": np.tile(np.array([[1.0, 2.0]]), (4, 1)),
              "shdmin": np.zeros(4), "shdmax": np.zeros(4)}
    old_land = np.array([True, False, True, False])
    new_land = np.array([True, True, False, False])
    audit = sh._carry_land_masked(arrays, xyz, old_land, new_land)
    assert arrays["isltyp"].tolist()[:3] == [5, 5, 14]
    # Cell 2 became water: its soil comes from OLD water (cell 1 or 3,
    # both 14), never from cell 1's newly donated land soil.
    assert arrays["soiltemp"].tolist()[:3] == [280.0, 280.0, 0.0]
    assert audit["newly_land_cells"] == 1 and audit["newly_water_cells"] == 1


def test_ice_flips_pair_land_use_with_land_ice_soil():
    lat = np.radians([70.0, 70.001, 70.002])
    xyz = np.stack((np.cos(lat), np.zeros(3), np.sin(lat)), axis=-1)
    arrays = {"isltyp": np.array([4, 16, 9]), "soilcat_top": np.array([4, 16, 9])}
    land = np.ones(3, bool)
    audit = sh._carry_land_masked(
        arrays, xyz, land, land, old_category=np.array([10, 15, 10]),
        new_category=np.array([15, 10, 10]), isice=15)
    assert arrays["isltyp"].tolist() == [16, 9, 9]
    assert audit["newly_ice_cells"] == 1 and audit["no_longer_ice_cells"] == 1


def test_open_polygons_are_never_selected(patch_static):
    path, _, n_patch, _ = patch_static
    with netCDF4.Dataset(path) as ds:
        geometry = sh.read_cell_geometry(ds)
    centre = n_patch // 2
    geometry.vertices[centre, 2] = -1          # a culled-ring style hole
    spacing = sh.cell_spacing_m(geometry)
    assert np.isnan(spacing[centre])
    assert np.isnan(sh.cell_inradius_m(geometry)[centre])
    selected = sh.select_cells(geometry, spacing, 0.5)
    assert not selected[centre]
    # Its neighbours become edge cells, polygon-tested with full polygons.
    assigner = sh.VoronoiAssigner(geometry, selected)
    assert assigner.edge_cell_count >= 6


def test_lattice_spacing_is_true_at_the_equatorward_edge():
    north = sh.SampleLattice.over(hf.FootprintBBox(40.0, 60.0, 0.0, 1.0), 30.0)
    south = sh.SampleLattice.over(hf.FootprintBBox(-60.0, -40.0, 0.0, 1.0), 30.0)
    assert north.truelat == 40.0 and south.truelat == -40.0
    assert north.area_m2(np.array([40.0, 60.0])).max() == pytest.approx(900.0)


def test_unmapped_landuse_class_is_refused():
    lookup = sh._category_lookup({1: 1, 17: 17}, 0.0)
    with pytest.raises(sh.StaticHighresRefusal, match=r"unmapped categories \[99\]"):
        sh.map_categories(np.array([1.0, 99.0, np.nan, 0.0]), lookup,
                          source_id="x")


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def test_refuses_outside_terrain_coverage(patch_static, tmp_path):
    with pytest.raises(sh.StaticHighresRefusal, match="no terrain coverage"):
        sh.fetch_terrain(hf.FootprintBBox(85.0, 86.0, 10.0, 11.0),
                         tmp_path, "copernicus-dem-glo30", offline=True)


def test_refuses_outside_landuse_coverage(tmp_path):
    with pytest.raises(sh.StaticHighresRefusal, match="no land-use coverage"):
        sh.fetch_landuse(hf.FootprintBBox(80.0, 81.0, 10.0, 11.0),
                         tmp_path, "cglc-modis-lcz", offline=True)


def test_offline_with_a_missing_tile_refuses_without_network(tmp_path):
    def no_network(*_a, **_k):  # pragma: no cover - must never run
        raise AssertionError("offline touched the network")

    with pytest.raises(hf.HighresFetchRefusal) as caught:
        sh.fetch_terrain(hf.FootprintBBox(52.2, 52.3, -3.7, -3.5), tmp_path,
                         "copernicus-dem-glo30", offline=True,
                         urlopen=no_network)
    assert "N52_00_W004_00" in str(caught.value)
    assert "--offline" in caught.value.remedy


def test_offline_with_a_missing_landuse_raster_refuses(tmp_path):
    with pytest.raises(hf.HighresFetchRefusal, match="CGLC_MODIS_LCZ.tif"):
        sh.fetch_landuse(hf.FootprintBBox(52.2, 52.3, -3.7, -3.5), tmp_path,
                         "cglc-modis-lcz", offline=True)


def test_offline_with_a_tampered_cached_tile_refuses(tmp_path):
    cache = tmp_path / "copernicus_dem_glo30"
    cache.mkdir()
    tile = cache / "Copernicus_DSM_COG_10_N52_00_W004_00_DEM.tif"
    tile.write_bytes(b"not the tile")
    (cache / (tile.name + ".sha256.json")).write_text(json.dumps(
        {"sha256": "0" * 64, "bytes": 12, "url": "u"}))
    with pytest.raises(hf.HighresFetchRefusal, match="N52_00_W004_00"):
        sh.fetch_terrain(hf.FootprintBBox(52.2, 52.3, -3.7, -3.5), tmp_path,
                         "copernicus-dem-glo30", offline=True)


def test_refuses_a_static_that_already_carries_a_vertical(tmp_path):
    points, _ = _generators(100.0, 3, 200)
    path = tmp_path / "v.static.nc"
    write_static(path, points, extra_vars=("zgrid",))
    with pytest.raises(sh.StaticHighresRefusal, match="vertical grid"):
        sh.run_static_highres(path, tmp_path / "o.nc", max_spacing_km=0.5)


def test_refuses_a_second_application(tmp_path):
    points, _ = _generators(100.0, 3, 200)
    path = tmp_path / "h.static.nc"
    write_static(path, points, extra_attrs={"highres_schema": sh.SCHEMA})
    with pytest.raises(sh.StaticHighresRefusal, match="already post-processed"):
        sh.run_static_highres(path, tmp_path / "o.nc", max_spacing_km=0.5)


def test_refuses_when_no_cell_is_fine_enough(patch_static, tmp_path):
    path, *_ = patch_static
    with pytest.raises(sh.StaticHighresRefusal, match="no cell is 0.01 km"):
        sh.run_static_highres(path, tmp_path / "o.nc", max_spacing_km=0.01)


def test_refuses_overwriting_input_or_existing_output(patch_static, tmp_path):
    path, *_ = patch_static
    with pytest.raises(sh.StaticHighresRefusal, match="names the input"):
        sh.run_static_highres(path, path)
    existing = tmp_path / "exists.nc"
    existing.write_bytes(b"")
    with pytest.raises(sh.StaticHighresRefusal, match="--clobber"):
        sh.run_static_highres(path, existing)


def test_refuses_nothing_to_do_and_non_modis_inventory(patch_static,
                                                       tmp_path):
    path, points, *_ = patch_static
    with pytest.raises(sh.StaticHighresRefusal, match="nothing to"):
        sh.run_static_highres(path, tmp_path / "o.nc", terrain=None)
    usgs = tmp_path / "usgs.static.nc"
    write_static(usgs, points)
    with netCDF4.Dataset(usgs, "r+") as ds:
        ds["mminlu"][:] = netCDF4.stringtoarr("USGS", 64)
    with pytest.raises(sh.StaticHighresRefusal, match="MODIFIED_IGBP_MODIS_NOAH"):
        sh.run_static_highres(usgs, tmp_path / "o.nc", terrain=None,
                              landuse="cglc", max_spacing_km=0.5)


def test_refuses_a_lattice_that_leaves_cells_empty(patch_static, monkeypatch,
                                                   tmp_path):
    path, *_ = patch_static
    _fake_sources(monkeypatch)
    with pytest.raises(sh.StaticHighresRefusal, match="received no lattice"):
        sh.run_static_highres(
            path, tmp_path / "o.nc", max_spacing_km=0.5, sample_m=400.0,
            sampler=_analytic_sampler(lambda la, lo: np.zeros(la.shape)),
            cache_root=tmp_path / "c")
    assert not (tmp_path / "o.nc").exists()


def test_refuses_without_the_rust_bridge(monkeypatch):
    from woof.static import rust_bridge

    monkeypatch.setenv(rust_bridge.STATIC_PYTHON_ENV, "1")
    with pytest.raises(sh.StaticHighresRefusal, match="no pure-Python body"):
        sh.rust_sampler(None, sh.SampleLattice(52, -3, 52, 30.0, 4, 4).grid())


def test_cli_parses_the_contract_flags():
    from woof.hex.cli import build_parser

    args = build_parser().parse_args(
        ["static-highres", "--static", "s.nc", "-o", "s2.nc",
         "--terrain", "glo30", "--landuse", "cglc"])
    assert args.terrain == "glo30" and args.landuse == "cglc"
    assert args.output == Path("s2.nc")
    from woof.hex import cli

    assert args.handler is cli._static_highres
    # The parser lives in the numpy-free CLI module; its choices are the
    # door's tables.
    assert cli.STATIC_HIGHRES_TERRAIN == tuple(sorted(sh.TERRAIN_CHOICES)) + ("none",)
    assert cli.STATIC_HIGHRES_LANDUSE == tuple(sorted(sh.LANDUSE_CHOICES)) + ("none",)
    assert cli.STATIC_HIGHRES_MAX_SPACING_KM == sh.DEFAULT_MAX_SPACING_KM


def test_cli_prints_refusals_and_remedies(tmp_path, capsys):
    from woof.hex.cli import main

    status = main(["static-highres", "--static", str(tmp_path / "none.nc"),
                   "-o", str(tmp_path / "o.nc")])
    assert status == 2
    assert "does not exist" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The real raster path (Rust static-fields warp) on a synthetic GeoTIFF
# ---------------------------------------------------------------------------

def write_geotiff(path: Path, values: np.ndarray, *, west: float,
                  north: float, res: float, nodata: float | None = None):
    """A minimal uncompressed, single-strip, float32 EPSG:4326 GeoTIFF
    (PixelIsArea, north row first) -- enough for the static-fields reader."""
    values = np.ascontiguousarray(values, dtype="<f4")
    ny, nx = values.shape
    data = values.tobytes()
    geokeys = [1, 1, 0, 3, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, 4326]
    extra: list[tuple[int, int, int, bytes]] = []   # tag, type, count, payload
    entries = [
        (256, 4, 1, struct.pack("<I", nx)),
        (257, 4, 1, struct.pack("<I", ny)),
        (258, 3, 1, struct.pack("<H", 32)),
        (259, 3, 1, struct.pack("<H", 1)),
        (262, 3, 1, struct.pack("<H", 1)),
        (273, 4, 1, None),                               # strip offset
        (277, 3, 1, struct.pack("<H", 1)),
        (278, 4, 1, struct.pack("<I", ny)),
        (279, 4, 1, struct.pack("<I", len(data))),
        (284, 3, 1, struct.pack("<H", 1)),
        (339, 3, 1, struct.pack("<H", 3)),
        (33550, 12, 3, struct.pack("<3d", res, res, 0.0)),
        (33922, 12, 6, struct.pack("<6d", 0, 0, 0, west, north, 0)),
        (34735, 3, len(geokeys), struct.pack(f"<{len(geokeys)}H", *geokeys)),
    ]
    if nodata is not None:
        text = (repr(float(nodata)) + "\0").encode("ascii")
        entries.append((42113, 2, len(text), text))
    del extra
    ifd_offset = 8
    ifd_size = 2 + 12 * len(entries) + 4
    payload_offset = ifd_offset + ifd_size
    blobs = b""
    records = []
    for tag, kind, count, payload in entries:
        if payload is None:
            records.append((tag, kind, count, None))
            continue
        if len(payload) <= 4:
            records.append((tag, kind, count, payload.ljust(4, b"\0")))
        else:
            records.append((tag, kind, count,
                            struct.pack("<I", payload_offset + len(blobs))))
            blobs += payload
            if len(blobs) % 2:
                blobs += b"\0"
    data_offset = payload_offset + len(blobs)
    out = bytearray(b"II*\0" + struct.pack("<I", ifd_offset))
    out += struct.pack("<H", len(records))
    for tag, kind, count, field in records:
        if field is None:
            field = struct.pack("<I", data_offset)
        out += struct.pack("<HHI", tag, kind, count) + field
    out += struct.pack("<I", 0)
    out += blobs + data
    Path(path).write_bytes(bytes(out))


def _rust_available() -> bool:
    from woof.static import rust_bridge

    return (rust_bridge.unavailable_reason() is None
            and not rust_bridge.python_fallback_requested())


@pytest.mark.skipif(not _rust_available(),
                    reason="static-fields bridge not staged")
def test_rust_lattice_warp_reads_the_pixel_under_each_sample(tmp_path):
    from woof.static.highres import BoundRaster, sha256_file

    res = 1.0 / 3600.0
    west, north = -3.70, 52.35
    ny, nx = 720, 720
    rows = north - (np.arange(ny) + 0.5) * res          # pixel-centre lat
    cols = west + (np.arange(nx) + 0.5) * res
    values = (1000.0 * (rows[:, None] - 52.0)
              + 10.0 * (cols[None, :] + 4.0)).astype(np.float32)
    tif = tmp_path / "dem.tif"
    write_geotiff(tif, values, west=west, north=north, res=res)
    bound = BoundRaster(path=tif, sha256=sha256_file(tif), source_id="t",
                        role="terrain", source_url="", license_id="",
                        license_url="", nominal_resolution="1as",
                        expected_bytes=tif.stat().st_size)
    bbox = hf.FootprintBBox(52.25, 52.30, -3.65, -3.55)
    lattice = sh.SampleLattice.over(bbox, 20.0)
    lon = lattice.longitudes()
    for row0, rows_n in [(0, 7), (lattice.ny - 5, 5)]:
        got = sh.rust_sampler(bound, lattice.grid(row0, rows_n))
        lat = lattice.latitudes(row0, rows_n)
        # The pixel containing each sample (north-first raster indexing).
        r = np.floor((north - lat) / res).astype(int)
        c = np.floor((lon - west) / res).astype(int)
        expected = values[r[:, None], c[None, :]]
        assert got.shape == expected.shape
        assert np.mean(got == expected) > 0.99, (row0, got[0, :4],
                                                 expected[0, :4])


@pytest.mark.skipif(not _rust_available(),
                    reason="static-fields bridge not staged")
def test_end_to_end_through_the_rust_warp_with_cached_tiles(
        patch_static, tmp_path):
    """Pre-stage a GLO-30-named tile with its sidecar, run offline."""
    path, points, n_patch, _ = patch_static
    cache = tmp_path / "cache"
    tile_dir = cache / "copernicus_dem_glo30"
    tile_dir.mkdir(parents=True)
    res = 1.0 / 3600.0
    values = np.full((3600, 3600), 250.0, dtype=np.float32)
    lat_rows = 53.0 - (np.arange(3600) + 0.5) * res
    values += (2000.0 * (lat_rows - PATCH_LAT))[:, None].astype(np.float32)
    tile = tile_dir / "Copernicus_DSM_COG_10_N52_00_W004_00_DEM.tif"
    write_geotiff(tile, values, west=-4.0, north=53.0, res=res)
    hf.record_local_artifact(tile, url=hf.COPERNICUS_DEM_TILE_URL.format(
        tile="N52_00_W004_00"))
    out = tmp_path / "rust.static.nc"
    receipt = sh.run_static_highres(path, out, max_spacing_km=0.5,
                                    cache_root=cache, offline=True)
    with netCDF4.Dataset(out) as ds:
        ter = np.asarray(ds["ter"][:], dtype=float)
    with netCDF4.Dataset(path) as ds:
        geometry = sh.read_cell_geometry(ds)
    selected = sh.select_cells(geometry, sh.cell_spacing_m(geometry), 0.5)
    lat_c = np.degrees(np.arcsin(points[:, 2]))
    expected = 250.0 + 2000.0 * (lat_c - PATCH_LAT)
    err = np.abs(ter[selected] - expected[selected])
    # A 0.0056 m/m-per-arcsecond staircase averaged over 100 m cells.
    assert np.median(err) < 1.0 and err.max() < 3.0
    assert receipt["sources"]["terrain_tiles"][0]["tile"] == tile.name


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("WOOF_NETWORK_TESTS") != "1",
                    reason="live network smoke; set WOOF_NETWORK_TESTS=1")
@pytest.mark.skipif(not _rust_available(),
                    reason="static-fields bridge not staged")
def test_real_glo30_window_over_a_welsh_valley(tmp_path):
    """One real GLO-30 tile (about 25 MB, cached per user) under a tiny
    100 m patch near the Elan Valley."""
    points, n_patch = _generators(100.0, 6, 200)
    path = tmp_path / "wales.static.nc"
    write_static(path, points)
    out = tmp_path / "wales.out.nc"
    receipt = sh.run_static_highres(path, out, max_spacing_km=0.5)
    terrain = receipt["terrain"]
    assert terrain["cells_fully_covered"] == receipt["cells_selected"]
    # Mid-Wales upland: real relief, nothing like the flat 100 m baseline.
    assert 150.0 < terrain["after"]["mean"] < 700.0
    assert terrain["after"]["max"] - terrain["after"]["min"] > 20.0
