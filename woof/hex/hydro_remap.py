"""Conservative exchange between hex (Voronoi) cells and a regular land-surface grid.

WRF-Hydro couples its routing to an LSM on a regular grid: the routing domain's ``geo_em`` (``GEO_STATIC_FLNM``)
names the LSM grid, each LSM cell is ``aggfactrt`` x ``aggfactrt`` routing cells, and ``wrf_cpl_HYDRO`` hands
SMOIS, SH2O, INFXSRT and SOLDRAIN to the routing and takes SMOIS, SH2O and SFCHEADRT back cell for cell.  On
hex the land surface runs on Voronoi cells, so the exchange crosses two tessellations.  This module is that
crossing, first-order conservative and built on the hex moving-mesh operator
(:class:`woof.hex.moving_remap.ConservativeMap`, exact great-circle overlaps of convex spherical cells):

* the grid's cells are its ``geo_em`` corner points (``XLAT_C``/``XLONG_C``) joined by great-circle arcs, so
  both sides are convex spherical polygons and every overlap ``W[g, c]`` (m2) is the exact spherical area of
  their intersection;
* the LSM grid must lie inside the hex mesh (every grid cell covered, the moving operator's
  ``target_subdomain``): a grid value is the overlap-weighted mean of the cells under it,
  ``q_g = sum_c W[g, c] q_c / A_g``, with ``A_g = sum_c W[g, c]``;
* what the routing hands back returns through the transpose, ``q_c = sum_g W[g, c] q_g / A_c`` with ``A_c`` the
  cell's own area, so a cell only partly under the grid takes the routed water of its covered part only and
  ``sum_c A_c q_c = sum_g A_g q_g`` holds exactly, whatever the field;
* a state the routing updates (soil moisture) returns as its increment, ``q_c += back(q_g' - q_g)``: a cell's
  own value is kept where the routing changed nothing and the water the routing moved is conserved;
* a category (vegetation, soil type) takes, per grid cell, the class with the largest overlap area.

The weights are float64 on the host; the exchange runs at the LSM interval on planes of a few thousand cells.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .moving_remap import ConservativeMap, SphericalCells, sphere_xyz, spherical_area

#: MPAS's sphere radius when a mesh declares none (mpas_constants.F a)
DEFAULT_RADIUS_M = 6371229.0


def mesh_cells(mesh, *, radius_m=None) -> SphericalCells:
    """The hex cells of ``mesh`` (zero-based ``verticesOnCell``, angles in radians) as convex spherical
    polygons.  Each cell's area is its polygon's (``ConservativeMap`` checks the mesh's ``areaCell`` against it
    to FP32 rounding when it is supplied)."""
    a = mesh.arrays
    radius = float(radius_m if radius_m is not None else
                   getattr(mesh, "attrs", {}).get("sphere_radius_m", DEFAULT_RADIUS_M))
    n = int(getattr(mesh, "n_cells", len(a["latCell"])))
    centers = sphere_xyz(np.asarray(a["latCell"], np.float64)[:n], np.asarray(a["lonCell"], np.float64)[:n])
    vertex = sphere_xyz(np.asarray(a["latVertex"], np.float64), np.asarray(a["lonVertex"], np.float64))
    counts = np.asarray(a["nEdgesOnCell"], np.int64)[:n]
    index = np.asarray(a["verticesOnCell"], np.int64)[:n]
    if np.any(index[np.arange(index.shape[1])[None, :] < counts[:, None]] < 0):
        raise ValueError("a hex cell names an absent vertex")
    vertices = vertex[np.maximum(index, 0)]
    area = np.array([spherical_area(v[:k]) for v, k in zip(vertices, counts)]) * radius ** 2
    return SphericalCells(centers, vertices, counts, area, radius)


def grid_cells(lat_corner_deg, lon_corner_deg, *, radius_m=DEFAULT_RADIUS_M) -> SphericalCells:
    """A regular grid's cells from its ``(ny + 1, nx + 1)`` corner points (degrees; geo_em's ``XLAT_C``,
    ``XLONG_C``), row ``j`` south to north and column ``i`` west to east, as great-circle quadrilaterals in
    counter-clockwise order.  Cell ``g = j * nx + i``."""
    lat = np.radians(np.asarray(lat_corner_deg, np.float64))
    lon = np.radians(np.asarray(lon_corner_deg, np.float64))
    if lat.ndim != 2 or lat.shape != lon.shape or min(lat.shape) < 2:
        raise ValueError("grid corners must be two matching (ny + 1, nx + 1) arrays")
    corner = sphere_xyz(lat, lon)
    quads = np.stack((corner[:-1, :-1], corner[:-1, 1:], corner[1:, 1:], corner[1:, :-1]), axis=2)
    vertices = np.ascontiguousarray(quads.reshape(-1, 4, 3))
    centers = vertices.sum(axis=1)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    counts = np.full(len(vertices), 4, np.int64)
    area = np.array([spherical_area(v) for v in vertices]) * float(radius_m) ** 2
    return SphericalCells(centers, vertices, counts, area, float(radius_m))


def geo_em_grid(path, *, radius_m=DEFAULT_RADIUS_M) -> SphericalCells:
    """The LSM grid of a WPS/WRF-Hydro ``geo_em`` file from its corner coordinates."""
    from netCDF4 import Dataset
    with Dataset(path) as ds:
        missing = [name for name in ("XLAT_C", "XLONG_C") if name not in ds.variables]
        if missing:
            raise ValueError(f"{path} carries no {missing}: the LSM grid's cell corners are needed for a "
                             "conservative exchange with hex cells")
        lat = np.asarray(ds.variables["XLAT_C"][:], np.float64)
        lon = np.asarray(ds.variables["XLONG_C"][:], np.float64)
    return grid_cells(lat.reshape(lat.shape[-2:]), lon.reshape(lon.shape[-2:]), radius_m=radius_m)


@dataclass
class CellGridExchange:
    """The operator between ``ncells`` hex cells and an ``(ny, nx)`` grid (see the module docstring)."""

    overlap: object          # scipy csr (grid rows, cell columns), m2
    cell_area: np.ndarray    # A_c, m2
    grid_area: np.ndarray    # A_g = row sums, m2
    shape: tuple
    receipt: dict

    @classmethod
    def build(cls, cells: SphericalCells, grid: SphericalCells, shape, *, coverage_tolerance=1e-8):
        shape = tuple(int(n) for n in shape)
        if int(np.prod(shape)) != len(grid.area):
            raise ValueError(f"grid shape {shape} does not hold its {len(grid.area)} cells")
        operator = ConservativeMap.build(cells, grid, target_subdomain=True,
                                         coverage_tolerance=coverage_tolerance)
        matrix = operator.overlap.tocsr()
        grid_area = np.asarray(matrix.sum(axis=1)).ravel()
        covered = np.asarray(matrix.sum(axis=0)).ravel()
        fraction = covered / cells.area
        receipt = dict(operator.receipt, exchange="hex cells <-> regular LSM grid",
                       grid_shape=list(shape), covered_cells=int(np.count_nonzero(covered)),
                       fully_covered_cells=int(np.count_nonzero(fraction > 1.0 - 1e-9)),
                       grid_area_m2=float(grid_area.sum()))
        return cls(matrix, cells.area.copy(), grid_area, shape, receipt)

    @property
    def covered_fraction(self) -> np.ndarray:
        """Each cell's area under the grid over its own area (0 outside, 1 inside)."""
        return np.asarray(self.overlap.sum(axis=0)).ravel() / self.cell_area

    def _cells(self, values):
        a = np.asarray(values, np.float64)
        if a.shape[-1] != len(self.cell_area):
            raise ValueError(f"cell field has {a.shape[-1]} cells, the exchange {len(self.cell_area)}")
        return a.reshape(-1, a.shape[-1])

    def _grid(self, values):
        a = np.asarray(values, np.float64)
        if a.shape[-2:] != self.shape:
            raise ValueError(f"grid field has shape {a.shape[-2:]}, the exchange {self.shape}")
        return a.reshape(-1, len(self.grid_area))

    def to_grid(self, values):
        """Overlap-weighted means of a ``(..., ncells)`` field on the ``(..., ny, nx)`` grid (float64)."""
        a = np.asarray(values)
        out = (self.overlap @ self._cells(a).T).T / self.grid_area
        return out.reshape(*a.shape[:-1], *self.shape)

    def to_cells(self, values):
        """A ``(..., ny, nx)`` grid field returned to the cells, ``sum_g W q_g / A_c`` (float64): zero outside
        the grid, the covered part's share on a partly covered cell; ``sum_c A_c q_c = sum_g A_g q_g``."""
        a = np.asarray(values)
        out = (self.overlap.T @ self._grid(a).T).T / self.cell_area
        return out.reshape(*a.shape[:-2], len(self.cell_area))

    def increment(self, before, after):
        """The cells' share of the grid's change ``after - before`` (float64)."""
        return self.to_cells(np.asarray(after, np.float64) - np.asarray(before, np.float64))

    def dominant(self, categories):
        """Per grid cell, the class of ``categories`` (one per hex cell) with the largest overlap area; ties
        go to the smaller class number."""
        cats = np.asarray(categories).reshape(-1)
        if cats.size != len(self.cell_area):
            raise ValueError("one category per hex cell")
        m = self.overlap.tocoo()
        classes, inverse = np.unique(cats, return_inverse=True)
        totals = np.zeros((len(self.grid_area), len(classes)))
        np.add.at(totals, (m.row, inverse[m.col]), m.data)
        return classes[np.argmax(totals, axis=1)].reshape(self.shape)

    def integral_cells(self, values):
        """``sum_c A_c q_c`` per leading index."""
        return self._cells(values) @ self.cell_area

    def integral_grid(self, values):
        """``sum_g A_g q_g`` per leading index."""
        return self._grid(values) @ self.grid_area
