"""WRF-Hydro coupled to hex columns through the conservative cell <-> grid exchange (woof.hex.wrf_hydro_hex).

The routing domain is the synthetic WRF-Hydro domain of tests/wrf_hydro_synthetic.py with its geo_em given
cell corners (XLAT_C/XLONG_C); the hex cells are a Voronoi tessellation around it whose edges are great-circle
arcs (a planar Voronoi diagram in the gnomonic projection, which maps great circles to straight lines).  Each
coupling cycle must

* conserve across the exchange: the soil water the cells gain equals the routing grid's soil change, the
  ponded head the cells receive equals the routing's head, and the runoff the routing receives equals the
  covered cells' runoff;
* be the opt-in's own WrfHydroCoupled on the remapped inputs, word for word (a second routing driven directly
  on the grid with the same inputs carries the same bytes);

and without a hydro namelist nothing is attached, so a hex run without the coupling is the run it was.
"""
from __future__ import annotations

import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from woof import optin

import wrf_hydro_synthetic as synthetic

REASON = optin.unavailable_reason("wrf_hydro")
needs_optin = pytest.mark.skipif(REASON is not None, reason=f"opt-in rule: {REASON}")
NX, NY = 8, 6
LAT0, DLAT, LON0, DLON = 40.0, 0.008, -100.0, 0.012
DZS = [0.1, 0.3, 0.6, 1.0]
START = datetime.datetime(2020, 1, 1)


def _corners():
    """geo_em's XLAT_C/XLONG_C for the synthetic domain's XLAT_M/XLONG_M (cell centres +- half a step)."""
    j, i = np.meshgrid(np.arange(NY + 1) - 0.5, np.arange(NX + 1) - 0.5, indexing="ij")
    return LAT0 + DLAT * j, LON0 + DLON * i


def _domain(path):
    import netCDF4
    synthetic.write(path, nx=NX, ny=NY)
    lat, lon = _corners()
    with netCDF4.Dataset(path / "geo_em.d01.nc", "a") as ds:
        ds.createDimension("south_north_stag", NY + 1)
        ds.createDimension("west_east_stag", NX + 1)
        for name, value in (("XLAT_C", lat), ("XLONG_C", lon)):
            ds.createVariable(name, "f4", ("Time", "south_north_stag", "west_east_stag"))[0] = value
    return path / "hydro.namelist"


def _hex_mesh(spacing_deg=0.006, seed=9):
    """Voronoi cells over the domain and a margin, great-circle edged (gnomonic plane Voronoi)."""
    from scipy.spatial import Voronoi
    lat_c, lon_c = np.radians(LAT0 + DLAT * NY / 2), np.radians(LON0 + DLON * NX / 2)
    centre = np.array([np.cos(lat_c) * np.cos(lon_c), np.cos(lat_c) * np.sin(lon_c), np.sin(lat_c)])
    east = np.array([-np.sin(lon_c), np.cos(lon_c), 0.0])
    north = np.cross(centre, east)
    rng = np.random.default_rng(seed)
    half = np.radians(np.array([DLON * NX * np.cos(lat_c), DLAT * NY])) * 0.5 + np.radians(4 * spacing_deg)
    count = int(np.prod(2 * half) / np.radians(spacing_deg) ** 2)
    points = rng.uniform(-1.0, 1.0, (count, 2)) * half
    vor = Voronoi(points)
    to_sphere = lambda xy: (centre + xy[:, :1] * east + xy[:, 1:] * north)  # noqa: E731
    keep, regions = [], []
    for k, r in enumerate(vor.point_region):
        region = vor.regions[r]
        if -1 in region or not region or np.any(np.abs(points[k]) > half - np.radians(1.5 * spacing_deg)):
            continue
        v = vor.vertices[region] - points[k]
        regions.append(np.asarray(region)[np.argsort(np.arctan2(v[:, 1], v[:, 0]))])
        keep.append(k)
    cells = to_sphere(points[keep])
    vertex = to_sphere(vor.vertices)
    latlon = lambda xyz: (np.arcsin(xyz[:, 2] / np.linalg.norm(xyz, axis=1)),  # noqa: E731
                          np.arctan2(xyz[:, 1], xyz[:, 0]))
    counts = np.array([len(r) for r in regions])
    on_cell = np.full((len(regions), counts.max()), -1, np.int64)
    for i, r in enumerate(regions):
        on_cell[i, :len(r)] = r
    (lat, lon), (latv, lonv) = latlon(cells), latlon(vertex)
    return SimpleNamespace(n_cells=len(regions), attrs={}, arrays=dict(
        latCell=lat, lonCell=lon, latVertex=latv, lonVertex=lonv, nEdgesOnCell=counts, verticesOnCell=on_cell))


def test_without_a_namelist_nothing_is_attached():
    from woof.hex.wrf_hydro_hex import attach
    driver = SimpleNamespace(fields={})
    assert attach(driver, None, namelist=None, start_time=START, dzs=DZS, lsm=4) is None
    assert vars(driver) == {"fields": {}}


def test_the_host_attaches_only_when_the_launch_asks(monkeypatch):
    from woof import wrf_hydro_route as route
    from woof.hex.wrf_hydro_hex import attach_host
    monkeypatch.delenv(route.ENV_NAMELIST, raising=False)
    backend = SimpleNamespace()
    assert attach_host(backend, {}, start_time=START) is None and vars(backend) == {}
    monkeypatch.setenv(route.ENV_NAMELIST, "/x/hydro.namelist")
    with pytest.raises(NotImplementedError, match="checkpoint"):
        attach_host(backend, {}, start_time=START, cycling={"restart_from": None, "checkpoint_steps": {6}})


def test_the_wrf_attach_leaves_hex_columns_to_the_hex_host(monkeypatch):
    from woof import wrf_hydro_route as route
    from woof.core.mpas_column_batch import _ColumnBatchState
    monkeypatch.setenv(route.ENV_NAMELIST, "/x/hydro.namelist")
    driver = SimpleNamespace(fields={})
    assert _ColumnBatchState.hex_columns is True
    assert route.attach(driver, SimpleNamespace(hex_columns=True), SimpleNamespace(sf_surface_physics=4),
                        start_time=START, dzs=DZS) is None
    assert vars(driver) == {"fields": {}}


@pytest.fixture
def coupled(tmp_path):
    cp = pytest.importorskip("cupy")
    if REASON is not None:
        pytest.skip(f"opt-in rule: {REASON}")
    from importlib import import_module
    from woof.hex.hydro_remap import mesh_cells
    from woof.hex.wrf_hydro_hex import attach
    from woof.wrf_hydro_route import soil_centres
    namelist = _domain(tmp_path / "dom")
    mesh = _hex_mesh()
    cells = mesh_cells(mesh)
    nc = len(cells.area)
    water = np.degrees(mesh.arrays["lonCell"]) > LON0 + DLON * (NX - 1.5)
    fields = {"smois": cp.zeros((4, 1, nc), cp.float32), "sh2o": cp.zeros((4, 1, nc), cp.float32),
              "ivgtyp": cp.asarray(np.where(water, 17, 10).reshape(1, nc), dtype=cp.int32),
              "isltyp": cp.asarray(np.where(water, 14, 6).reshape(1, nc), dtype=cp.int32)}
    driver = SimpleNamespace(fields=fields, bldt_seconds=60.0)
    seam = attach(driver, cells, namelist=namelist, start_time=START, dzs=DZS, lsm=2,
                  output_dir=str(tmp_path / "out"))
    # The routing driven directly on the grid: the same domain, classes and inputs.
    module = import_module(optin.load("wrf_hydro").__name__ + ".coupled")
    veg = seam.remap.dominant(np.where(water, 17, 10))
    sol = seam.remap.dominant(np.where(water, 14, 6))
    reference = module.WrfHydroCoupled(namelist, ix=NX, jx=NY, zs=soil_centres(DZS), vegtyp=veg,
                                       soltyp=sol, hydro_dt=60.0, start=START, output_dir=tmp_path / "ref")
    return SimpleNamespace(cp=cp, seam=seam, driver=driver, reference=reference, nc=nc, cells=cells)


@needs_optin
def test_every_cycle_conserves_and_is_the_routing_on_the_remapped_grid(coupled):
    cp, seam, ex = coupled.cp, coupled.seam, coupled.seam.remap
    rng = np.random.default_rng(3)
    f32 = np.float32
    for cycle in range(4):
        smois = rng.uniform(0.2, 0.4, (4, coupled.nc)).astype(f32)
        sh2o = (smois * f32(0.9)).astype(f32)
        infx = rng.uniform(0.0, 6.0, coupled.nc).astype(f32)
        sold = rng.uniform(0.0, 0.2, coupled.nc).astype(f32)
        fields = coupled.driver.fields
        fields["smois"][...] = cp.asarray(smois.reshape(4, 1, -1))
        fields["sh2o"][...] = cp.asarray(sh2o.reshape(4, 1, -1))
        seam.planes["infxsrt"][...] = cp.asarray(infx.reshape(1, -1))
        seam.planes["soldrain"][...] = cp.asarray(sold.reshape(1, -1))
        grid = {name: cp.asarray(ex.to_grid(value.astype(np.float64)).astype(f32))
                for name, value in (("smois", smois), ("sh2o", sh2o), ("infxsrt", infx), ("soldrain", sold))}
        head = cp.zeros((NY, NX), cp.float32)
        coupled.reference.exchange(smois=grid["smois"], sh2o=grid["sh2o"], infxsrt=grid["infxsrt"],
                                   soldrain=grid["soldrain"], sfcheadrt=head)
        seam.exchange(fields)
        got, want = seam.routing.digest_arrays(), coupled.reference.digest_arrays()
        assert set(got) == set(want)
        for name in want:
            assert cp.asnumpy(got[name]).tobytes() == cp.asnumpy(want[name]).tobytes(), (cycle, name)
        budget = seam.budget[-1]
        # The soil water moved: exact in float64 (to the summation's rounding of the whole column store), and
        # the float32 state cast within its own measured envelope.
        total = budget["soil_water_cells_m3"]
        exact, grid_change = budget["soil_change_cells_exact_m3"], budget["soil_change_grid_m3"]
        assert abs(exact - grid_change) <= 1e-12 * total, (cycle, exact, grid_change)
        cast = abs(budget["soil_change_cells_m3"] - exact)
        assert cast <= budget["soil_cast_envelope_m3"] + 1e-12 * total, (cycle, cast)
        # The runoff the routing receives: exact before its float32 cast, the cast within its envelope.
        runoff = budget["runoff_cells_m3"]
        assert abs(runoff - budget["runoff_grid_exact_m3"]) <= 1e-12 * runoff, cycle
        cast = abs(budget["runoff_grid_m3"] - budget["runoff_grid_exact_m3"])
        assert cast <= budget["runoff_cast_envelope_m3"] + 1e-12 * runoff, cycle
        head_cells, head_grid = budget["head_cells_m3"], budget["head_grid_m3"]
        assert abs(head_cells - head_grid) <= 1e-12 * max(abs(head_grid), 1e-9), (cycle, head_cells, head_grid)
        returned = ex.to_cells(cp.asnumpy(head).astype(np.float64)).astype(f32)
        assert cp.asnumpy(seam.planes["sfcheadrt"])[0].tobytes() == returned.tobytes()
    assert abs(budget["soil_change_grid_m3"]) > 0.0, "the routing moved no soil water: the test is vacuous"
    assert any(b["head_grid_m3"] > 0.0 for b in seam.budget), "no ponded head came back: the test is vacuous"
