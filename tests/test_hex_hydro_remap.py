"""The hex cell <-> regular LSM grid exchange (woof.hex.hydro_remap), on a spherical Voronoi mesh.

The cells are a real Voronoi tessellation of the sphere (scipy SphericalVoronoi on seeded points, convex
great-circle polygons like an MPAS mesh), the grid a regular latitude-longitude block inside it given by its
corner points like a geo_em's XLAT_C/XLONG_C.  The checks are the exchange's contract: constants are kept,
every grid cell is covered, what returns to the cells conserves the grid integral to rounding, increments of
a fully covered cell are the grid's, the dominant category is the largest overlap, and the grid shape and
the coverage are refused when they do not hold.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import SphericalVoronoi

from woof.hex.hydro_remap import CellGridExchange, grid_cells, mesh_cells

RADIUS = 6371229.0


def _mesh(n=6000, seed=4):
    rng = np.random.default_rng(seed)
    points = rng.standard_normal((n, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    sv = SphericalVoronoi(points)
    sv.sort_vertices_of_regions()
    counts = np.array([len(r) for r in sv.regions])
    on_cell = np.full((n, counts.max()), -1, np.int64)
    for i, region in enumerate(sv.regions):
        on_cell[i, :len(region)] = region
    lat = lambda xyz: np.arcsin(np.clip(xyz[:, 2], -1, 1))  # noqa: E731
    lon = lambda xyz: np.arctan2(xyz[:, 1], xyz[:, 0])  # noqa: E731
    arrays = {"latCell": lat(points), "lonCell": lon(points), "latVertex": lat(sv.vertices),
              "lonVertex": lon(sv.vertices), "nEdgesOnCell": counts, "verticesOnCell": on_cell}
    return SimpleNamespace(arrays=arrays, n_cells=n, attrs={"sphere_radius_m": RADIUS})


def _grid(ny=6, nx=8, lat0=10.0, lon0=20.0, d=1.5):
    lat = lat0 + d * np.arange(ny + 1)
    lon = lon0 + d * np.arange(nx + 1)
    return np.meshgrid(lat, lon, indexing="ij"), (ny, nx)


@pytest.fixture(scope="module")
def exchange():
    (lat, lon), shape = _grid()
    cells = mesh_cells(_mesh())
    return CellGridExchange.build(cells, grid_cells(lat, lon, radius_m=RADIUS), shape), cells


def test_the_grid_is_covered_and_constants_are_kept(exchange):
    ex, cells = exchange
    grid = grid_cells(*_grid()[0], radius_m=RADIUS)
    np.testing.assert_allclose(ex.grid_area, grid.area, rtol=1e-12)
    np.testing.assert_allclose(ex.to_grid(np.full(len(cells.area), 0.3)), 0.3, rtol=1e-12)
    fraction = ex.covered_fraction
    assert 0 < ex.receipt["fully_covered_cells"] < ex.receipt["covered_cells"] < len(cells.area)
    assert fraction.min() == 0.0 and abs(fraction.max() - 1.0) < 1e-9


def test_what_returns_to_the_cells_conserves_the_grid_integral(exchange):
    ex, cells = exchange
    rng = np.random.default_rng(1)
    for field in (rng.uniform(0.0, 50.0, (2,) + ex.shape), np.full((1,) + ex.shape, 3.0)):
        back = ex.to_cells(field)
        np.testing.assert_allclose(ex.integral_cells(back), ex.integral_grid(field), rtol=1e-13)
        assert np.all(back[:, ex.covered_fraction == 0] == 0.0)


def test_forward_conserves_the_covered_cell_mass(exchange):
    ex, cells = exchange
    q = np.random.default_rng(2).uniform(0.1, 0.4, (4, len(cells.area)))
    covered = np.asarray(ex.overlap.sum(axis=0)).ravel()
    np.testing.assert_allclose(ex.integral_grid(ex.to_grid(q)), q @ covered, rtol=1e-13)


def test_an_increment_reaches_a_fully_covered_cell_whole(exchange):
    ex, cells = exchange
    before = np.random.default_rng(3).uniform(0.1, 0.4, (3,) + ex.shape)
    delta = ex.increment(before, before + 0.05)
    full = ex.covered_fraction > 1.0 - 1e-9
    np.testing.assert_allclose(delta[:, full], 0.05, rtol=1e-9)
    partial = (ex.covered_fraction > 0) & ~full
    want = np.broadcast_to(0.05 * ex.covered_fraction[partial], delta[:, partial].shape)
    np.testing.assert_allclose(delta[:, partial], want, rtol=1e-9)


def test_the_dominant_category_is_the_largest_overlap(exchange):
    ex, cells = exchange
    lat = np.degrees(mesh_cells(_mesh()).centers[:, 2])
    categories = np.where(lat > 14.5, 7, 2)
    dominant = ex.dominant(categories)
    assert set(np.unique(dominant)) == {2, 7}
    assert np.all(dominant[0] == 2) and np.all(dominant[-1] == 7)
    assert np.array_equal(ex.dominant(np.full(len(cells.area), 5)), np.full(ex.shape, 5))


def test_a_grid_outside_the_mesh_and_a_wrong_shape_are_refused():
    cells = mesh_cells(_mesh())
    (lat, lon), shape = _grid()
    grid = grid_cells(lat, lon, radius_m=RADIUS)
    with pytest.raises(ValueError, match="shape"):
        CellGridExchange.build(cells, grid, (shape[0] + 1, shape[1]))
    keep = np.degrees(np.arcsin(cells.centers[:, 2])) < 15.0
    partial = type(cells)(cells.centers[keep], cells.vertices[keep], cells.counts[keep], cells.area[keep],
                          cells.radius_m)
    with pytest.raises(ValueError, match="coverage"):
        CellGridExchange.build(partial, grid, shape)
