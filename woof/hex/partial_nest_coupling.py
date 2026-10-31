"""The hex partial band's coupling operators, built once on the host.

Coarse to fine: MPAS's own barycentric interpolation
(``src/operators/mpas_geometry_utils.F`` of MPAS-A v8.4.2,
``mpas_calculate_barycentric_weights_for_points`` and the routines it calls)
transcribed statement for statement in binary64 -- MPAS's double-precision
build -- with Python's ``math`` (the platform libm, which gfortran also
calls).  ``components/hex/tools/partial_nest_mpas_oracle`` compiles those
routines and holds these weights to them word for word; the device applies
them with ``mpas_cells_to_points_using_baryweights``'s statement
(``cuda_partial_nest_v841.pn_bary_apply_v841``), held to the same routine
compiled in single precision.

Fine to coarse: the conservative overlap of :mod:`woof.hex.moving_remap`
(convex great-circle cell intersection, binary64) between the child's
interior cells and the parent cells they cover completely, as a CSR table
whose rows are those parent cells.  :func:`remap_reference` is the exact
binary64 statement the device kernel ``pn_remap_rows_v841`` evaluates, entry
for entry in the same order, and it conserves the remapped mass-weighted
integral to round-off.

The interface budget: the overlap area of every parent cell with the whole
child domain (``cover``) and the child's cell areas, for
``pn_budget_residual_v841``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class CellGeometry:
    """A mesh's cell and vertex geometry in binary64, 0-based connectivity
    (-1 for an absent neighbour), positions on the sphere of ``radius``."""

    x_cell: np.ndarray        # (nCells, 3)
    x_vertex: np.ndarray      # (nVertices, 3)
    cells_on_vertex: np.ndarray
    vertices_on_cell: np.ndarray
    n_edges_on_cell: np.ndarray
    area_cell: np.ndarray
    radius: float

    @classmethod
    def from_mesh(cls, mesh, positions64=None) -> "CellGeometry":
        """``positions64``: optional dict of binary64 ``xCell``.. arrays (the
        grid file's own words); otherwise the mesh's arrays are widened."""
        a = mesh.arrays
        src = positions64 or a

        def xyz(suffix):
            return np.stack([np.asarray(src[n + suffix], np.float64) for n in "xyz"], axis=1)

        x_cell = xyz("Cell")
        return cls(x_cell, xyz("Vertex"),
                   counterclockwise_cells_on_vertex(a["cellsOnVertex"], x_cell),
                   np.asarray(a["verticesOnCell"], np.int64),
                   np.asarray(a["nEdgesOnCell"], np.int64),
                   np.asarray(src.get("areaCell", a["areaCell"]), np.float64),
                   float(mesh.attrs["sphere_radius"]))


def counterclockwise_cells_on_vertex(cells_on_vertex, x_cell) -> np.ndarray:
    """``cellsOnVertex`` with every complete triangle counterclockwise seen
    from outside the sphere.

    The MPAS mesh specification orders ``cellsOnVertex`` counterclockwise,
    and ``mpas_calculate_barycentric_weights`` depends on it (a clockwise
    triangle has a negative signed area and gets zero weights; the
    point-in-polygon test rejects every point).  Meshes from
    ``rw_mpas_mesh`` order half their triangles clockwise (MEASURED: 2560 of
    5120 on its 2562-cell mesh), so the table's copy is normalized by
    swapping the last two cells of a clockwise triangle -- the same triangle,
    MPAS's orientation.  The mesh the dycore reads is not touched."""
    cov = np.array(cells_on_vertex, np.int64, copy=True)
    x = np.asarray(x_cell, np.float64)
    ok = np.all((cov >= 0) & (cov < len(x)), axis=1)
    rows = np.flatnonzero(ok)
    a, b, c = x[cov[rows, 0]], x[cov[rows, 1]], x[cov[rows, 2]]
    clockwise = rows[np.einsum("ij,ij->i", np.cross(b - a, c - a), a) < 0.0]
    cov[clockwise, 1], cov[clockwise, 2] = cov[clockwise, 2], cov[clockwise, 1].copy()
    return cov


# ---------------------------------------------------------------------------
# MPAS barycentric weights (mpas_geometry_utils.F, binary64)
# ---------------------------------------------------------------------------

def _arc_length(a, b) -> float:
    # mpas_arc_length (F:131-154)
    cx = b[0] - a[0]
    cy = b[1] - a[1]
    cz = b[2] - a[2]
    r = math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
    c = math.sqrt(cx * cx + cy * cy + cz * cz)
    return r * 2.0 * math.asin(c / (2.0 * r))


def _triangle_signed_area_sphere(a, b, c, radius: float) -> float:
    # mpas_triangle_signed_area_sphere (F:652-691)
    ab = _arc_length(a, b) / radius
    bc = _arc_length(b, c) / radius
    ca = _arc_length(c, a) / radius
    semiperim = 0.5 * (ab + bc + ca)
    tanqe = math.sqrt(max(0.0, math.tan(0.5 * semiperim) * math.tan(0.5 * (semiperim - ab))
                          * math.tan(0.5 * (semiperim - bc)) * math.tan(0.5 * (semiperim - ca))))
    area = 4.0 * radius * radius * math.atan(tanqe)
    ablen = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    aclen = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    d0 = (ablen[1] * aclen[2]) - (ablen[2] * aclen[1])
    d1 = -((ablen[0] * aclen[2]) - (ablen[2] * aclen[0]))
    d2 = (ablen[0] * aclen[1]) - (ablen[1] * aclen[0])
    if (d0 * a[0] + d1 * a[1] + d2 * a[2]) < 0.0:
        area = -area
    return area


def _cross(p1, p2):
    # mpas_cross_product_in_r3 (mpas_vector_operations.F:113-121)
    return (p1[1] * p2[2] - p1[2] * p2[1],
            p1[2] * p2[0] - p1[0] * p2[2],
            p1[0] * p2[1] - p1[1] * p2[0])


def _point_in_polygon(point, polygon) -> bool:
    # mpas_point_in_polygon (F:710-749), on a sphere: normal_vector = point
    eps = 1.0e-12
    n = len(polygon)
    for i in range(n):
        nxt = polygon[(i + 1) % n]
        here = polygon[i]
        vec1 = (nxt[0] - here[0], nxt[1] - here[1], nxt[2] - here[2])
        vec2 = (point[0] - here[0], point[1] - here[1], point[2] - here[2])
        cp = _cross(vec1, vec2)
        if (cp[0] * point[0] + cp[1] * point[1] + cp[2] * point[2]) < (0.0 - eps):
            return False
    return True


def _weights(point, a, b, c, radius):
    # mpas_calculate_barycentric_weights (F:311-372)
    area = _triangle_signed_area_sphere(a, b, c, radius)
    if area > 0.0:
        return (_triangle_signed_area_sphere(point, b, c, radius) / area,
                _triangle_signed_area_sphere(point, c, a, radius) / area,
                _triangle_signed_area_sphere(point, a, b, radius) / area)
    return (0.0, 0.0, 0.0)


def nearest_vertices(geometry: CellGeometry, points: np.ndarray) -> np.ndarray:
    """The search origin MPAS asks its caller for: each point's nearest
    vertex (``mpas_calculate_barycentric_weights_for_points``' docstring)."""
    from scipy.spatial import cKDTree
    _, index = cKDTree(geometry.x_vertex).query(np.asarray(points, np.float64))
    return np.asarray(index, np.int64)


def barycentric_weights(geometry: CellGeometry, points: np.ndarray,
                        search_vertex: np.ndarray | None = None):
    """``mpas_calculate_barycentric_weights_for_points`` (F:423-565):
    ``(cells (n, 3) int64 0-based, weights (n, 3) float64, inside (n,) bool)``.
    A point no candidate triangle owns takes the nearest searched triangle's
    weights (barycentric extrapolation), as MPAS does."""
    points = np.asarray(points, np.float64)
    if search_vertex is None:
        search_vertex = nearest_vertices(geometry, points)
    xc, xv = geometry.x_cell, geometry.x_vertex
    cov, voc, noc = geometry.cells_on_vertex, geometry.vertices_on_cell, geometry.n_edges_on_cell
    n_cells, n_vertices = len(xc), len(xv)
    radius = geometry.radius

    def valid_cell(c):
        return 0 <= c < n_cells

    cells = np.zeros((len(points), 3), np.int64)
    weights = np.zeros((len(points), 3), np.float64)
    inside = np.zeros(len(points), bool)
    for ip in range(len(points)):
        point = tuple(points[ip])
        base = int(search_vertex[ip])
        found = False
        nearest_index = base
        tri = cov[base]
        if all(valid_cell(c) for c in tri):
            found = _point_in_polygon(point, [tuple(xc[c]) for c in tri])
            nearest_distance = _arc_length(point, tuple(xv[base]))
        else:
            nearest_distance = float("inf")
        chosen = tri
        if not found:
            for c in tri:
                if not valid_cell(c):
                    continue
                for t in range(int(noc[c])):
                    it = int(voc[c, t])
                    if not 0 <= it < n_vertices:
                        continue
                    tri2 = cov[it]
                    if all(valid_cell(cc) for cc in tri2):
                        if _point_in_polygon(point, [tuple(xc[cc]) for cc in tri2]):
                            found = True
                            chosen = tri2
                            break
                        distance = _arc_length(point, tuple(xv[it]))
                    else:
                        distance = float("inf")
                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest_index = it
                if found:
                    break
        if not found:
            chosen = cov[nearest_index]
        cells[ip] = chosen
        weights[ip] = _weights(point, tuple(xc[chosen[0]]), tuple(xc[chosen[1]]),
                               tuple(xc[chosen[2]]), radius)
        inside[ip] = found
    if np.any(cells < 0):
        raise ValueError("a barycentric triangle reaches outside the parent mesh: the "
                         "child must lie inside its parent's interior")
    return cells, weights, inside


@dataclass
class BarycentricTable:
    """Device-ready weights: cells int32 (n, 3), weights float32 (n, 3)."""

    cells: np.ndarray
    weights64: np.ndarray
    inside: np.ndarray

    @property
    def weights32(self) -> np.ndarray:
        return np.ascontiguousarray(self.weights64.astype(np.float32))

    @property
    def cells32(self) -> np.ndarray:
        return np.ascontiguousarray(self.cells.astype(np.int32))


def bary_table(parent: CellGeometry, points: np.ndarray) -> BarycentricTable:
    cells, weights, inside = barycentric_weights(parent, points)
    return BarycentricTable(cells, weights, inside)


def bary_apply_reference(table: BarycentricTable, field: np.ndarray) -> np.ndarray:
    """``mpas_cells_to_points_using_baryweights`` in binary32, left to right."""
    f = np.asarray(field, np.float32)
    w = table.weights32
    c = table.cells
    out = (w[:, 0] * f[..., c[:, 0]]).astype(np.float32)
    out = (out + (w[:, 1] * f[..., c[:, 1]]).astype(np.float32)).astype(np.float32)
    out = (out + (w[:, 2] * f[..., c[:, 2]]).astype(np.float32)).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# the conservative overlap (fine to coarse) and the interface budget
# ---------------------------------------------------------------------------

def _spherical_cells(geometry: CellGeometry, subset: np.ndarray):
    from .moving_remap import SphericalCells
    subset = np.asarray(subset, np.int64)
    unit_c = geometry.x_cell / np.linalg.norm(geometry.x_cell, axis=1, keepdims=True)
    unit_v = geometry.x_vertex / np.linalg.norm(geometry.x_vertex, axis=1, keepdims=True)
    voc = geometry.vertices_on_cell[subset]
    return SphericalCells(unit_c[subset], unit_v[np.maximum(voc, 0)],
                          geometry.n_edges_on_cell[subset], geometry.area_cell[subset],
                          geometry.radius)


def overlap_matrix(source: CellGeometry, source_cells, target: CellGeometry, target_cells):
    """(target rows x source cols) overlap areas in m^2, binary64
    (moving_remap's convex great-circle intersection)."""
    from scipy.sparse import csr_matrix
    from .moving_remap import _overlap_groups, _overlap_rows
    s = _spherical_cells(source, source_cells)
    t = _spherical_cells(target, target_cells)
    sr, tr = s.angular_radius(), t.angular_radius()
    rows, cols, values, _ = _overlap_rows(s, t, sr, tr, _overlap_groups(s, sr), None,
                                          0, len(t.area))
    return csr_matrix((values, (rows, cols)), shape=(len(t.area), len(s.area))), s, t


def _candidate_parents(parent: CellGeometry, child: CellGeometry, child_cells) -> np.ndarray:
    """Parent cells within reach of the given child cells."""
    from scipy.spatial import cKDTree
    pts = child.x_cell[np.asarray(child_cells)]
    reach = 3.0 * math.sqrt(float(parent.area_cell.max()))
    hits = cKDTree(parent.x_cell).query_ball_point(pts, reach)
    return np.unique(np.concatenate([np.asarray(h, np.int64) for h in hits]))


@dataclass
class FeedbackRemap:
    """CSR rows over the parent cells the child's interior covers wholly."""

    targets: np.ndarray      # parent cell per row (int64)
    row_ptr: np.ndarray      # int64
    cols: np.ndarray         # child cell per entry (int64)
    overlap: np.ndarray      # m^2 per entry (float64)
    coverage: np.ndarray     # per row: overlap sum / parent polygon area

    @classmethod
    def build(cls, parent: CellGeometry, child: CellGeometry, child_interior,
              *, coverage_floor: float = 1.0 - 1.0e-9) -> "FeedbackRemap":
        from .moving_remap import spherical_area
        interior = np.asarray(child_interior, np.int64)
        candidates = _candidate_parents(parent, child, interior)
        matrix, _, t = overlap_matrix(child, interior, parent, candidates)
        geometric = np.array([spherical_area(v[:n]) * parent.radius ** 2
                              for v, n in zip(t.vertices, t.counts)])
        coverage = np.asarray(matrix.sum(axis=1)).ravel() / geometric
        keep = np.flatnonzero(coverage >= coverage_floor)
        sub = matrix[keep].tocsr()
        sub.sort_indices()
        return cls(targets=candidates[keep], row_ptr=sub.indptr.astype(np.int64),
                   cols=interior[sub.indices].astype(np.int64),
                   overlap=sub.data.astype(np.float64), coverage=coverage[keep])

    def reference(self, rho: np.ndarray, q: np.ndarray) -> np.ndarray:
        """The binary64 remap of ``q`` (mass-weighted by ``rho``) onto the
        target rows, per level: (nlev, nrows) float64 -- the device kernel's
        sums, entry for entry in the same order."""
        rho = np.asarray(rho, np.float32).astype(np.float64)
        q = np.asarray(q, np.float32).astype(np.float64)
        out = np.empty((rho.shape[0], len(self.targets)), np.float64)
        for r in range(len(self.targets)):
            sl = slice(self.row_ptr[r], self.row_ptr[r + 1])
            num = np.zeros(rho.shape[0])
            den = np.zeros(rho.shape[0])
            for j in range(sl.start, sl.stop):
                c = self.cols[j]
                mass = self.overlap[j] * rho[:, c]
                num = num + mass * q[:, c]
                den = den + mass
            out[:, r] = num / den
        return out


@dataclass
class BudgetCover:
    """The parent's cells that overlap the child domain, the area of each
    the child covers, and the child's cell areas (binary64)."""

    parents: np.ndarray
    cover: np.ndarray
    child_area: np.ndarray

    @classmethod
    def build(cls, parent: CellGeometry, child: CellGeometry) -> "BudgetCover":
        cells = np.arange(len(child.area_cell))
        candidates = _candidate_parents(parent, child, cells)
        matrix, s, _ = overlap_matrix(child, cells, parent, candidates)
        cover = np.asarray(matrix.sum(axis=1)).ravel()
        keep = np.flatnonzero(cover > 0.0)
        return cls(parents=candidates[keep], cover=cover[keep],
                   child_area=np.asarray(child.area_cell, np.float64))


def remap_conservation(remap: FeedbackRemap, rho: np.ndarray, q: np.ndarray):
    """``(sum_rows den*R(q), sum_entries O*rho*q)`` per level in binary64 --
    equal to round-off (the remap's conservation statement)."""
    rho64 = np.asarray(rho, np.float32).astype(np.float64)
    q64 = np.asarray(q, np.float32).astype(np.float64)
    result = remap.reference(rho, q)
    lhs = np.zeros(rho64.shape[0])
    rhs = np.zeros(rho64.shape[0])
    for r in range(len(remap.targets)):
        for j in range(remap.row_ptr[r], remap.row_ptr[r + 1]):
            c = remap.cols[j]
            mass = remap.overlap[j] * rho64[:, c]
            lhs += mass * result[:, r]
            rhs += mass * q64[:, c]
    return lhs, rhs


__all__ = ["BarycentricTable", "BudgetCover", "CellGeometry", "FeedbackRemap",
           "bary_apply_reference", "bary_table", "barycentric_weights",
           "counterclockwise_cells_on_vertex",
           "nearest_vertices", "overlap_matrix", "remap_conservation"]
