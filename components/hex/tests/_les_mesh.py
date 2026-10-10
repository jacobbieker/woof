"""A small closed MPAS-convention Voronoi mesh for the LES tests.

Built in pure Python (scipy ``SphericalVoronoi`` on an icosahedral point
set) so the LES authority can be exercised on a real spherical C-grid with no
mesh asset and no Rust binary.  Conventions follow MPAS: the edge normal
points from ``cellsOnEdge[e,0]`` to ``cellsOnEdge[e,1]``, the tangent
``k x n`` points from ``verticesOnEdge[e,0]`` to ``verticesOnEdge[e,1]``,
``verticesOnCell`` runs counter-clockwise seen from outside, and edge slot
``j`` of a cell joins its vertices ``j`` and ``j+1``.

``radius`` scales the sphere, so a 162-cell mesh on a sphere of radius
1,600 m has an ~250 m spacing: a synthetic LES patch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TinyMesh:
    arrays: dict[str, np.ndarray]
    attrs: dict[str, Any] = field(default_factory=dict)


def _icosahedral_points(level: int) -> np.ndarray:
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    verts = [
        (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
        (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
        (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
    ]
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]
    points = [np.asarray(v, dtype=np.float64) / np.linalg.norm(v) for v in verts]
    cache: dict[tuple[int, int], int] = {}

    def midpoint(a: int, b: int) -> int:
        key = (min(a, b), max(a, b))
        if key not in cache:
            m = points[a] + points[b]
            points.append(m / np.linalg.norm(m))
            cache[key] = len(points) - 1
        return cache[key]

    for _ in range(level):
        refined = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            refined += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = refined
    return np.asarray(points)


def _arc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a * b, axis=-1))


def _triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    # L'Huilier-free formula (Van Oosterom & Strackee) on the unit sphere.
    numerator = abs(np.dot(a, np.cross(b, c)))
    denominator = 1.0 + np.dot(a, b) + np.dot(b, c) + np.dot(c, a)
    return float(2.0 * np.arctan2(numerator, denominator))


def build_tiny_mesh(level: int = 2, *, radius: float = 1600.0) -> TinyMesh:
    from scipy.spatial import SphericalVoronoi

    from woof.hex.vector import initialize_reconstruction_coefficients

    cells = _icosahedral_points(level)
    sv = SphericalVoronoi(cells, radius=1.0)
    sv.sort_vertices_of_regions()
    vertices = sv.vertices / np.linalg.norm(sv.vertices, axis=1)[:, None]
    n_cells = cells.shape[0]
    regions = []
    for i, region in enumerate(sv.regions):
        ring = list(region)
        poly = vertices[ring]
        orient = sum(
            np.dot(np.cross(poly[j] - cells[i], poly[(j + 1) % len(ring)] - cells[i]), cells[i])
            for j in range(len(ring))
        )
        if orient < 0.0:
            ring = ring[::-1]
        regions.append(ring)
    max_edges = max(len(r) for r in regions)

    edge_of: dict[tuple[int, int], int] = {}
    edge_cells: list[list[int]] = []
    edge_verts: list[tuple[int, int]] = []
    for i, ring in enumerate(regions):
        for j in range(len(ring)):
            a, b = ring[j], ring[(j + 1) % len(ring)]
            key = (min(a, b), max(a, b))
            if key not in edge_of:
                edge_of[key] = len(edge_cells)
                edge_cells.append([i])
                edge_verts.append(key)
            else:
                edge_cells[edge_of[key]].append(i)
    n_edges = len(edge_cells)
    cells_on_edge = np.zeros((n_edges, 2), dtype=np.int64)
    vertices_on_edge = np.zeros((n_edges, 2), dtype=np.int64)
    for e, (pair, verts) in enumerate(zip(edge_cells, edge_verts)):
        assert len(pair) == 2
        c1, c2 = sorted(pair)
        cells_on_edge[e] = (c1, c2)
        mid = cells[c1] + cells[c2]
        mid /= np.linalg.norm(mid)
        normal = cells[c2] - cells[c1]
        left = np.cross(mid, normal)
        a, b = verts
        if np.dot(vertices[b] - vertices[a], left) > 0.0:
            vertices_on_edge[e] = (a, b)
        else:
            vertices_on_edge[e] = (b, a)

    edges_on_cell = np.zeros((n_cells, max_edges), dtype=np.int64)
    cells_on_cell = np.zeros((n_cells, max_edges), dtype=np.int64)
    vertices_on_cell = np.zeros((n_cells, max_edges), dtype=np.int64)
    n_edges_on_cell = np.zeros(n_cells, dtype=np.int64)
    for i, ring in enumerate(regions):
        n_edges_on_cell[i] = len(ring)
        for j in range(len(ring)):
            a, b = ring[j], ring[(j + 1) % len(ring)]
            e = edge_of[(min(a, b), max(a, b))]
            edges_on_cell[i, j] = e
            c1, c2 = cells_on_edge[e]
            cells_on_cell[i, j] = c2 if c1 == i else c1
            vertices_on_cell[i, j] = a
        # MPAS pads unused slots with the last used entry.
        for j in range(len(ring), max_edges):
            edges_on_cell[i, j] = edges_on_cell[i, len(ring) - 1]
            cells_on_cell[i, j] = cells_on_cell[i, len(ring) - 1]
            vertices_on_cell[i, j] = vertices_on_cell[i, len(ring) - 1]

    n_vertices = vertices.shape[0]
    edges_on_vertex = [[] for _ in range(n_vertices)]
    cells_on_vertex = [[] for _ in range(n_vertices)]
    for e, (a, b) in enumerate(vertices_on_edge):
        edges_on_vertex[a].append(e)
        edges_on_vertex[b].append(e)
    for i, ring in enumerate(regions):
        for v in ring:
            cells_on_vertex[v].append(i)
    assert all(len(x) == 3 for x in edges_on_vertex)
    assert all(len(x) == 3 for x in cells_on_vertex)
    edges_on_vertex_a = np.asarray(edges_on_vertex, dtype=np.int64)
    cells_on_vertex_a = np.asarray(cells_on_vertex, dtype=np.int64)

    dc = _arc(cells[cells_on_edge[:, 0]], cells[cells_on_edge[:, 1]]) * radius
    dv = _arc(vertices[vertices_on_edge[:, 0]], vertices[vertices_on_edge[:, 1]]) * radius
    area_cell = np.asarray(sv.calculate_areas()) * radius * radius
    area_triangle = np.asarray(
        [
            _triangle_area(*(cells[c] for c in cells_on_vertex_a[v]))
            for v in range(n_vertices)
        ]
    ) * radius * radius
    edge_mid = cells[cells_on_edge[:, 0]] + cells[cells_on_edge[:, 1]]
    edge_mid /= np.linalg.norm(edge_mid, axis=1)[:, None]

    def latlon(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.arcsin(np.clip(points[:, 2], -1.0, 1.0)), np.arctan2(points[:, 1], points[:, 0])

    lat_c, lon_c = latlon(cells)
    lat_e, lon_e = latlon(edge_mid)
    lat_v, lon_v = latlon(vertices)
    arrays: dict[str, np.ndarray] = {
        "cellsOnEdge": cells_on_edge,
        "verticesOnEdge": vertices_on_edge,
        "edgesOnCell": edges_on_cell,
        "cellsOnCell": cells_on_cell,
        "verticesOnCell": vertices_on_cell,
        "nEdgesOnCell": n_edges_on_cell,
        "edgesOnVertex": edges_on_vertex_a,
        "cellsOnVertex": cells_on_vertex_a,
        "dcEdge": dc,
        "dvEdge": dv,
        "areaCell": area_cell,
        "areaTriangle": area_triangle,
        "meshDensity": np.ones(n_cells),
        "nominalMinDc": np.asarray(float(dc.min())),
        "xCell": cells[:, 0] * radius,
        "yCell": cells[:, 1] * radius,
        "zCell": cells[:, 2] * radius,
        "xVertex": vertices[:, 0] * radius,
        "yVertex": vertices[:, 1] * radius,
        "zVertex": vertices[:, 2] * radius,
        "xEdge": edge_mid[:, 0] * radius,
        "yEdge": edge_mid[:, 1] * radius,
        "zEdge": edge_mid[:, 2] * radius,
        "latCell": lat_c,
        "lonCell": lon_c,
        "latEdge": lat_e,
        "lonEdge": lon_e,
        "latVertex": lat_v,
        "lonVertex": lon_v,
    }
    mesh = TinyMesh(
        arrays=arrays,
        attrs={"on_a_sphere": "YES", "sphere_radius": float(radius), "is_periodic": "NO"},
    )
    mesh.arrays["coeffs_reconstruct"] = np.asarray(
        initialize_reconstruction_coefficients(mesh)
    )
    return mesh


def edge_normal_wind(mesh: TinyMesh, zonal: np.ndarray, meridional: np.ndarray) -> np.ndarray:
    """Project a (zonal, meridional) wind per level onto the edge normals.

    ``zonal``/``meridional`` have shape ``(nlev,)`` (uniform horizontally) or
    ``(nlev, nEdges)``; the normal at each edge is the unit tangent-plane
    vector from cellsOnEdge[0] to cellsOnEdge[1].
    """

    a = mesh.arrays
    coe = a["cellsOnEdge"]
    xyz = np.stack([a["xCell"], a["yCell"], a["zCell"]], axis=1)
    mid = np.stack([a["xEdge"], a["yEdge"], a["zEdge"]], axis=1)
    mid = mid / np.linalg.norm(mid, axis=1)[:, None]
    normal = xyz[coe[:, 1]] - xyz[coe[:, 0]]
    normal -= np.sum(normal * mid, axis=1)[:, None] * mid
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    lat, lon = a["latEdge"], a["lonEdge"]
    east = np.stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)], axis=1)
    north = np.stack([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)], axis=1)
    ne = np.sum(normal * east, axis=1)
    nn = np.sum(normal * north, axis=1)
    zonal = np.asarray(zonal, dtype=np.float64)
    meridional = np.asarray(meridional, dtype=np.float64)
    if zonal.ndim == 1:
        zonal = zonal[:, None]
        meridional = meridional[:, None]
    return zonal * ne[None, :] + meridional * nn[None, :]


def edge_tangent_wind(mesh: TinyMesh, zonal: np.ndarray, meridional: np.ndarray) -> np.ndarray:
    """The ``k x n`` component, MPAS's reconstructed tangential velocity ``v``."""

    a = mesh.arrays
    coe = a["cellsOnEdge"]
    xyz = np.stack([a["xCell"], a["yCell"], a["zCell"]], axis=1)
    mid = np.stack([a["xEdge"], a["yEdge"], a["zEdge"]], axis=1)
    mid = mid / np.linalg.norm(mid, axis=1)[:, None]
    normal = xyz[coe[:, 1]] - xyz[coe[:, 0]]
    normal -= np.sum(normal * mid, axis=1)[:, None] * mid
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    tangent = np.cross(mid, normal)
    lat, lon = a["latEdge"], a["lonEdge"]
    east = np.stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)], axis=1)
    north = np.stack([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)], axis=1)
    te = np.sum(tangent * east, axis=1)
    tn = np.sum(tangent * north, axis=1)
    zonal = np.asarray(zonal, dtype=np.float64)
    meridional = np.asarray(meridional, dtype=np.float64)
    if zonal.ndim == 1:
        zonal = zonal[:, None]
        meridional = meridional[:, None]
    return zonal * te[None, :] + meridional * tn[None, :]


def flat_vertical(nlev: int, ncells: int, dz: float) -> dict[str, np.ndarray]:
    """A flat-terrain uniform vertical grid in hex's 0-based convention."""

    zgrid = np.repeat((np.arange(nlev + 1) * dz)[:, None], ncells, axis=1)
    rdzw = np.full(nlev, 1.0 / dz)
    rdzu = np.full(nlev, 1.0 / dz)
    rdzu[0] = 0.0
    fzm = np.full(nlev, 0.5)
    fzp = np.full(nlev, 0.5)
    fzm[0] = 0.0
    fzp[0] = 0.0
    return {
        "zgrid": zgrid,
        "zz": np.ones((nlev, ncells)),
        "rdzw": rdzw,
        "rdzu": rdzu,
        "fzm": fzm,
        "fzp": fzp,
    }
