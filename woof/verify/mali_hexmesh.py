"""A doubly periodic hexagonal MPAS mesh (a torus), uniform or variable resolution, for the MALI oracles.

MALI's forward mode (``woof.hex.mpas_ice.forward_v842``) runs every cell of
a block, so its test meshes have no culled cells and no index past ``nCells``:
:func:`torus` lays ``n`` x ``n`` hexagons on a rhombus with periodic
connectivity. Cell ``c = r*n + q`` sits at axial ``(q, r)``; the plane point of
an axial point ``(a, b)`` is ``X = dc * n * (F(a/n) + F(b/n)/2)``,
``Y = -dc * n * sqrt(3)/2 * F(b/n)`` with ``F(u) = u + stretch * g(u - floor u)``
and ``g(u) = u (1 - u) (1 - 2 u)``. ``stretch = 0`` is the uniform mesh;
``stretch = 1.2`` refines the middle of the rhombus and coarsens its rim, a
spacing ratio of 5.5 (``F' = 1 + stretch * g'`` runs from 2.2 to 0.4), with
the map periodic and smooth across the seam.

The conventions are MPAS's:

- a cell's vertices go counter-clockwise, and ``edgesOnCell(i)`` joins
  ``verticesOnCell(i)`` and ``verticesOnCell(i+1)``, across from
  ``cellsOnCell(i)``;
- ``cellsOnEdge(1)`` lies to the right of ``verticesOnEdge(1)`` ->
  ``verticesOnEdge(2)`` (the edge's normal, cell 1 to cell 2, turned
  counter-clockwise is its tangent);
- ``cellsOnVertex`` goes counter-clockwise round the vertex and
  ``kiteAreasOnVertex(i)`` is ``cellsOnVertex(i)``'s share.

Coordinates are metres, ``z = 0``, each stored inside the rhombus (as MPAS's
periodic planar meshes store them); lengths and areas are measured across the
seam. A kite is the quadrilateral of the cell's centre, the midpoints of its
two edges at the vertex and the vertex, so a cell's kites tile its polygon.
``coeffs_reconstruct`` is the regular hexagon's least-squares reconstruction
(``n_e / 3``), and ``weightsOnEdge`` Thuburn et al. (2009)'s, as MPAS-Tools
forms them. Only correctly rounded operations build the mesh (no libm
trigonometry, no BLAS), so its float64 words are the same on every host.
Arrays are 1-based in MPAS's C-order-reversed layout (``edgesOnCell`` is
``(nCells, maxEdges)``) and float64; the caller casts.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from woof.verify.cases.hex_obstacles import trisk_weights

_R3 = math.sqrt(3.0)
#: the neighbours at 60k degrees, as axial steps
_STEPS = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))
#: the corners at 30 + 60k degrees (between neighbours k and k+1), as axial offsets in thirds
_CORNERS = ((2, -1), (1, -2), (-1, -1), (-2, 1), (-1, 2), (1, 1))
#: the vertex at each corner: (its owner's axial step, 0 for the owner's U vertex or 1 for its D)
_CORNER_VERTEX = (((0, 0), 0), ((0, -1), 1), ((-1, 0), 0), ((-1, 0), 1), ((-1, 1), 0), ((0, 0), 1))
MAX_EDGES = 6


def ordered_sum(values) -> float:
    """Left-to-right float sum (builtin ``sum`` compensates on Python 3.12+)."""
    total = 0.0
    for value in values:
        total += value
    return total


def _stretch_map(u: float, stretch: float) -> float:
    base = math.floor(u)
    f = u - base
    return u + stretch * (f * (1.0 - f) * (1.0 - 2.0 * f))


def _shoelace(points) -> float:
    n = len(points)
    return 0.5 * ordered_sum(points[i][0] * points[(i + 1) % n][1] - points[(i + 1) % n][0] * points[i][1]
                             for i in range(n))


def torus(n: int, dc: float, stretch: float = 0.0) -> dict:
    """The ``n`` x ``n`` periodic hexagonal mesh of nominal spacing ``dc``
    metres (see the module docstring): a dict of MPAS's mesh arrays and
    dimensions, plus ``period`` (the rhombus's two edge vectors)."""
    if n < 3:
        raise ValueError("a periodic hexagonal mesh needs n >= 3")
    nc, ne, nv = n * n, 3 * n * n, 2 * n * n
    span = dc * n

    def point(a: float, b: float) -> tuple[float, float]:
        fa, fb = _stretch_map(a / n, stretch), _stretch_map(b / n, stretch)
        return span * (fa + 0.5 * fb), -span * (0.5 * _R3) * fb

    def cell(q: int, r: int) -> int:
        return (r % n) * n + (q % n)

    def corner(q: int, r: int, k: int) -> tuple[float, float]:
        da, db = _CORNERS[k]
        return point(q + da / 3.0, r + db / 3.0)

    def vertex_of(q: int, r: int, k: int) -> int:
        (dq, dr), d = _CORNER_VERTEX[k]
        return 2 * cell(q + dq, r + dr) + d

    x_cell, y_cell = np.zeros(nc), np.zeros(nc)
    x_vertex, y_vertex = np.zeros(nv), np.zeros(nv)
    cells_on_cell = np.zeros((nc, MAX_EDGES), np.int64)
    edges_on_cell = np.zeros((nc, MAX_EDGES), np.int64)
    vertices_on_cell = np.zeros((nc, MAX_EDGES), np.int64)
    cells_on_edge = np.zeros((ne, 2), np.int64)
    vertices_on_edge = np.zeros((ne, 2), np.int64)
    cells_on_vertex = np.zeros((nv, 3), np.int64)
    edges_on_vertex = np.zeros((nv, 3), np.int64)
    dc_edge, dv_edge = np.zeros(ne), np.zeros(ne)
    normals = np.zeros((ne, 3))
    area_cell = np.zeros(nc)
    kites = np.zeros((nv, 3))

    def edge_of(q: int, r: int, k: int) -> int:
        if k < 3:
            return 3 * cell(q, r) + k
        dq, dr = _STEPS[k]
        return 3 * cell(q + dq, r + dr) + (k - 3)

    def wrapped(a: float, b: float) -> tuple[float, float]:
        return point(a - n * math.floor(a / n), b - n * math.floor(b / n))

    for r in range(n):
        for q in range(n):
            c = cell(q, r)
            x_cell[c], y_cell[c] = point(q, r)
            for d, k in ((0, 0), (1, 5)):
                da, db = _CORNERS[k]
                x_vertex[2 * c + d], y_vertex[2 * c + d] = wrapped(q + da / 3.0, r + db / 3.0)
            corners = [corner(q, r, k) for k in range(6)]
            area_cell[c] = _shoelace(corners)
            for k in range(6):
                dq, dr = _STEPS[k]
                cells_on_cell[c, k] = cell(q + dq, r + dr)
                edges_on_cell[c, k] = edge_of(q, r, k)
                vertices_on_cell[c, k] = vertex_of(q, r, (k - 1) % 6)
            # the three edges this cell owns: to its neighbours at 0, 60 and 120 degrees
            px, py = point(q, r)
            for k in range(3):
                e = 3 * c + k
                dq, dr = _STEPS[k]
                ox, oy = point(q + dq, r + dr)
                ax, ay = corners[(k - 1) % 6]
                bx, by = corners[k]
                cells_on_edge[e] = c, cell(q + dq, r + dr)
                vertices_on_edge[e] = vertex_of(q, r, (k - 1) % 6), vertex_of(q, r, k)
                length = math.sqrt((ox - px) * (ox - px) + (oy - py) * (oy - py))
                dc_edge[e] = length
                dv_edge[e] = math.sqrt((bx - ax) * (bx - ax) + (by - ay) * (by - ay))
                normals[e] = (ox - px) / length, (oy - py) / length, 0.0
    # the vertices: the owner's U (corner 0: the owner, its neighbours 0 and 1) and D
    # (corner 5: the owner, its neighbours 5 and 0), counter-clockwise
    for r in range(n):
        for q in range(n):
            c = cell(q, r)
            for d, ring in ((0, (None, 0, 1)), (1, (None, 5, 0))):
                v = 2 * c + d
                around = [(q, r)] + [(q + _STEPS[k][0], r + _STEPS[k][1]) for k in ring[1:]]
                cells_on_vertex[v] = [cell(a, b) for a, b in around]
                vx, vy = corner(q, r, 0 if d == 0 else 5)
                for i in range(3):
                    a0, b0 = around[i]
                    a1, b1 = around[(i + 1) % 3]
                    a2, b2 = around[(i - 1) % 3]
                    k_next = _STEPS.index((a1 - a0, b1 - b0))
                    k_prev = _STEPS.index((a2 - a0, b2 - b0))
                    edges_on_vertex[v, i] = edge_of(a0, b0, k_next)
                    cx, cy = point(a0, b0)
                    m_next = _midpoint(corner(a0, b0, (k_next - 1) % 6), corner(a0, b0, k_next))
                    m_prev = _midpoint(corner(a0, b0, (k_prev - 1) % 6), corner(a0, b0, k_prev))
                    kites[v, i] = abs(_shoelace([(cx, cy), m_next, (vx, vy), m_prev]))
    mesh = {
        "nCells": nc, "nEdges": ne, "nVertices": nv, "maxEdges": MAX_EDGES, "maxEdges2": 2 * MAX_EDGES,
        "vertexDegree": 3,
        "nEdgesOnCell": np.full(nc, 6, np.int64), "cellsOnCell": cells_on_cell, "edgesOnCell": edges_on_cell,
        "verticesOnCell": vertices_on_cell, "cellsOnEdge": cells_on_edge, "verticesOnEdge": vertices_on_edge,
        "cellsOnVertex": cells_on_vertex, "edgesOnVertex": edges_on_vertex,
        "xCell": x_cell, "yCell": y_cell, "zCell": np.zeros(nc),
        "xVertex": x_vertex, "yVertex": y_vertex, "zVertex": np.zeros(nv),
        "dcEdge": dc_edge, "dvEdge": dv_edge, "areaCell": area_cell, "kiteAreasOnVertex": kites,
        "areaTriangle": np.array([ordered_sum(row) for row in kites]),
        "edgeNormalVectors": normals,
        "period": (point(n, 0), point(0, n)),
    }
    # the reconstruction: the regular hexagon's (sum over its edges of n n^T is 3 I)
    coeffs = np.zeros((nc, MAX_EDGES, 3))
    for c in range(nc):
        for i in range(6):
            coeffs[c, i] = normals[edges_on_cell[c, i]] / 3.0
    mesh["coeffs_reconstruct"] = coeffs
    weights, edges_on_edge, count = trisk_weights(SimpleNamespace(arrays=mesh))
    mesh["weightsOnEdge"] = weights
    mesh["edgesOnEdge"] = edges_on_edge
    mesh["nEdgesOnEdge"] = count
    for key in ("cellsOnCell", "edgesOnCell", "verticesOnCell", "cellsOnEdge", "verticesOnEdge", "cellsOnVertex",
                "edgesOnVertex", "edgesOnEdge"):
        mesh[key] = (mesh[key] + 1).astype(np.int32)
    for key in ("nEdgesOnCell", "nEdgesOnEdge"):
        mesh[key] = mesh[key].astype(np.int32)
    return mesh


def edge_points(n: int, dc: float, stretch: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """:func:`torus`'s ``xEdge`` and ``yEdge``: the plane point of each edge's
    axial midpoint (between ``cellsOnEdge(1)`` at ``(q, r)`` and its neighbour
    one step away), stored inside the rhombus as the cell and vertex points
    are, so an edge on the seam may sit next to ``cellsOnEdge(2)`` rather
    than ``cellsOnEdge(1)``."""
    span = dc * n
    x_edge, y_edge = np.zeros(3 * n * n), np.zeros(3 * n * n)
    for r in range(n):
        for q in range(n):
            c = r * n + q
            for k in range(3):
                dq, dr = _STEPS[k]
                a, b = q + 0.5 * dq, r + 0.5 * dr
                a, b = a - n * math.floor(a / n), b - n * math.floor(b / n)
                fa, fb = _stretch_map(a / n, stretch), _stretch_map(b / n, stretch)
                x_edge[3 * c + k], y_edge[3 * c + k] = span * (fa + 0.5 * fb), -span * (0.5 * _R3) * fb
    return x_edge, y_edge


def _midpoint(a, b) -> tuple[float, float]:
    return 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])


def centre(mesh: dict) -> tuple[float, float]:
    """The plane point of the rhombus's middle (axial (n/2, n/2)), where the stretched mesh is finest."""
    (ax, ay), (bx, by) = mesh["period"]
    return 0.5 * (ax + bx), 0.5 * (ay + by)


def distance(mesh: dict, x, y, x0: float, y0: float) -> np.ndarray:
    """The plane distance from (x0, y0) to each (x, y), across the seam (the nearest periodic image)."""
    (ax, ay), (bx, by) = mesh["period"]
    best = None
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            dx = np.asarray(x, np.float64) + (i * ax + j * bx) - x0
            dy = np.asarray(y, np.float64) + (i * ay + j * by) - y0
            d = np.sqrt(dx * dx + dy * dy)
            best = d if best is None else np.minimum(best, d)
    return best
