"""A defect-free variable-resolution spherical mesh, built analytically.

``graded_sphere_mesh`` is the synthetic instrument of the mesh-validation
tests and the source of the sphere the hex obstacle verification cases run
on, so it ships with the package rather than living in the test suite.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import ConvexHull

from woof.hex.mesh import Mesh


#: The radius the published meshes carry.
EARTH_RADIUS = 6371220.0


# ---------------------------------------------------------------------------
# the instrument: a defect-free variable-resolution spherical mesh
# ---------------------------------------------------------------------------
def _icosahedral_points(level: int) -> np.ndarray:
    """Unit-sphere points of the geodesic icosahedron at ``level``."""

    golden = (1.0 + 5.0**0.5) / 2.0
    corners = []
    for first, second in ((1.0, golden), (-1.0, golden), (1.0, -golden), (-1.0, -golden)):
        corners += [(0.0, first, second), (first, second, 0.0), (second, 0.0, first)]
    base = np.array(corners, dtype=np.float64)
    base /= np.linalg.norm(base, axis=1)[:, None]

    steps = 1 << level
    points = []
    for face in ConvexHull(base).simplices:
        first, second, third = base[face]
        for i in range(steps + 1):
            for j in range(steps + 1 - i):
                k = steps - i - j
                points.append((i * first + j * second + k * third) / steps)
    grid = np.array(points, dtype=np.float64)
    grid /= np.linalg.norm(grid, axis=1)[:, None]
    _, unique = np.unique(np.round(grid, 9), axis=0, return_index=True)
    return grid[np.sort(unique)]


def _conformal_zoom(points: np.ndarray, factor: float) -> np.ndarray:
    """Grade the point set by a Möbius zoom -- a conformal sphere map.

    Conformal matters: the map rescales lengths locally without shearing, so a
    near-equilateral triangulation stays near-equilateral and every triangle
    keeps its circumcentre inside itself.  Spacing runs from ``1/factor`` of
    the uniform spacing at one pole to ``factor`` times it at the other, which
    is how a real variable-resolution mesh carries fine and coarse edges in one
    file.
    """

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    denominator = np.where(1.0 - z < 1.0e-15, 1.0e-15, 1.0 - z)
    real = factor * x / denominator
    imaginary = factor * y / denominator
    square = real * real + imaginary * imaginary
    zoomed = np.stack(
        (
            2.0 * real / (1.0 + square),
            2.0 * imaginary / (1.0 + square),
            (square - 1.0) / (1.0 + square),
        ),
        axis=1,
    )
    return zoomed / np.linalg.norm(zoomed, axis=1)[:, None]


def _generic_rotation() -> np.ndarray:
    """A fixed rotation that keeps the refined region off the axes.

    On a coordinate pole two of the three binary32 components are near zero
    and quantize hundreds of times finer than they do anywhere else, and the
    third one's error is radial -- which normalization removes.  A mesh refined
    there would understate its own storage noise by orders of magnitude and the
    instrument would report a defect-free-looking mesh for the wrong reason.
    """

    matrix = np.eye(3)
    for axis, angle in enumerate((0.6435011087932844, 0.9272952180016122, 0.4636476090008061)):
        turn = np.eye(3)
        first, second = [index for index in range(3) if index != axis]
        turn[first, first] = math.cos(angle)
        turn[second, second] = math.cos(angle)
        turn[first, second] = -math.sin(angle)
        turn[second, first] = math.sin(angle)
        matrix = matrix @ turn
    return matrix


def _spherical_triangle_area(first, second, third):
    """Van Oosterom & Strackee excess, on the unit sphere."""

    triple = np.abs(np.einsum("ij,ij->i", first, np.cross(second, third)))
    denominator = (
        1.0
        + np.einsum("ij,ij->i", first, second)
        + np.einsum("ij,ij->i", second, third)
        + np.einsum("ij,ij->i", third, first)
    )
    return 2.0 * np.arctan2(triple, denominator)


def _arc_length(first, second, radius):
    return radius * np.arctan2(
        np.linalg.norm(np.cross(first, second), axis=1),
        np.einsum("ij,ij->i", first, second),
    )


def _angles_and_positions(unit: np.ndarray, radius: float, dtype) -> dict[str, np.ndarray]:
    scaled = unit * radius
    latitude = np.arcsin(np.clip(unit[:, 2], -1.0, 1.0))
    longitude = np.arctan2(unit[:, 1], unit[:, 0])
    return {
        "x": scaled[:, 0].astype(dtype),
        "y": scaled[:, 1].astype(dtype),
        "z": scaled[:, 2].astype(dtype),
        "lat": latitude.astype(dtype),
        "lon": longitude.astype(dtype),
    }


def graded_sphere_mesh(
    *,
    level: int = 5,
    zoom: float = 12.0,
    radius: float = EARTH_RADIUS,
    dtype=np.float32,
) -> Mesh:
    """A defect-free variable-resolution MPAS mesh, stored at ``dtype``.

    Cells are the points, mesh vertices are the Delaunay circumcentres, and
    every metric is the exact binary64 spherical measure of the figure the
    connectivity describes.  Nothing is approximated except by the final cast
    to ``dtype``, which is the point: that cast is what the arc-length check
    has to tolerate.
    """

    base = _icosahedral_points(level)
    # Exact icosahedral symmetry puts cocircular quadruples on the sphere, and
    # a cocircular quadruple gives two triangles one shared circumcentre -- a
    # zero-length dvEdge that is a degeneracy of the construction rather than
    # anything a generator would emit.  A deterministic jitter at a few per
    # cent of the (uniform) base spacing removes it, and the conformal map
    # carries it as the same few per cent of the local spacing everywhere.
    base = base + np.random.default_rng(20260823).normal(scale=1.0e-3, size=base.shape)
    base /= np.linalg.norm(base, axis=1)[:, None]
    unit = _conformal_zoom(base, zoom) @ _generic_rotation().T
    unit /= np.linalg.norm(unit, axis=1)[:, None]

    triangles = ConvexHull(unit, qhull_options="Qt").simplices.astype(np.int64)
    n_cells = unit.shape[0]
    n_vertices = triangles.shape[0]

    corner = [unit[triangles[:, index]] for index in range(3)]
    normal = np.cross(corner[1] - corner[0], corner[2] - corner[0])
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    outward = np.where(np.einsum("ij,ij->i", normal, corner[0]) < 0.0, -1.0, 1.0)
    vertex_unit = normal * outward[:, None]

    pairs = np.sort(
        np.concatenate(
            (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]), axis=0
        ),
        axis=1,
    )
    cells_on_edge, inverse = np.unique(pairs, axis=0, return_inverse=True)
    n_edges = cells_on_edge.shape[0]
    edge_of_pair = {
        (int(low), int(high)): index for index, (low, high) in enumerate(cells_on_edge)
    }

    vertices_on_edge = np.full((n_edges, 2), -1, dtype=np.int64)
    filled = np.zeros(n_edges, dtype=np.int64)
    for position, edge in enumerate(np.asarray(inverse).ravel()):
        vertices_on_edge[edge, filled[edge]] = position % n_vertices
        filled[edge] += 1
    if not np.all(filled == 2):
        raise AssertionError("every Delaunay edge must carry exactly two triangles")

    triangles_on_cell: list[list[int]] = [[] for _ in range(n_cells)]
    for triangle, row in enumerate(triangles):
        for cell in row:
            triangles_on_cell[int(cell)].append(triangle)

    n_edges_on_cell = np.array(
        [len(entry) for entry in triangles_on_cell], dtype=np.int64
    )
    max_edges = int(n_edges_on_cell.max())
    vertices_on_cell = np.full((n_cells, max_edges), -1, dtype=np.int64)
    edges_on_cell = np.full((n_cells, max_edges), -1, dtype=np.int64)
    cells_on_cell = np.full((n_cells, max_edges), -1, dtype=np.int64)

    for cell in range(n_cells):
        centre = unit[cell]
        seed = np.array([1.0, 0.0, 0.0]) if abs(centre[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        first_axis = np.cross(centre, seed)
        first_axis /= np.linalg.norm(first_axis)
        second_axis = np.cross(centre, first_axis)
        incident = np.array(triangles_on_cell[cell], dtype=np.int64)
        offsets = vertex_unit[incident]
        order = np.argsort(
            np.arctan2(offsets @ second_axis, offsets @ first_axis), kind="stable"
        )
        ring = incident[order]
        count = ring.size
        vertices_on_cell[cell, :count] = ring
        for slot in range(count):
            here = set(int(value) for value in triangles[ring[slot]])
            there = set(int(value) for value in triangles[ring[(slot + 1) % count]])
            shared = (here & there) - {cell}
            if len(shared) != 1:
                raise AssertionError(
                    f"cell {cell} slot {slot} does not have one shared neighbour"
                )
            neighbour = shared.pop()
            cells_on_cell[cell, slot] = neighbour
            edges_on_cell[cell, slot] = edge_of_pair[
                (min(cell, neighbour), max(cell, neighbour))
            ]

    edges_on_vertex = np.array(
        [
            [
                edge_of_pair[
                    (
                        min(int(row[index]), int(row[(index + 1) % 3])),
                        max(int(row[index]), int(row[(index + 1) % 3])),
                    )
                ]
                for index in range(3)
            ]
            for row in triangles
        ],
        dtype=np.int64,
    )

    max_edges2 = 2 * max_edges
    n_edges_on_edge = (
        n_edges_on_cell[cells_on_edge[:, 0]] + n_edges_on_cell[cells_on_edge[:, 1]] - 2
    )
    edges_on_edge = np.full((n_edges, max_edges2), -1, dtype=np.int64)
    for edge in range(n_edges):
        stencil = []
        for cell in cells_on_edge[edge]:
            stencil += [
                int(value)
                for value in edges_on_cell[cell, : n_edges_on_cell[cell]]
                if int(value) != edge
            ]
        edges_on_edge[edge, : len(stencil)] = stencil
        if len(stencil) != int(n_edges_on_edge[edge]):
            raise AssertionError("tangential stencil length disagrees with its count")

    # Kites: the quad (cell centre, arc midpoint, circumcentre, arc midpoint).
    # They tile the Delaunay triangle and the Voronoi cell at once, so summing
    # them one way gives areaTriangle and the other way gives areaCell, with no
    # second formula that could disagree with the first.
    kite_areas = np.zeros((n_vertices, 3), dtype=np.float64)
    for index in range(3):
        here = unit[triangles[:, index]]
        left = unit[triangles[:, (index + 1) % 3]]
        right = unit[triangles[:, (index + 2) % 3]]
        first_middle = here + left
        first_middle /= np.linalg.norm(first_middle, axis=1)[:, None]
        second_middle = here + right
        second_middle /= np.linalg.norm(second_middle, axis=1)[:, None]
        kite_areas[:, index] = radius * radius * (
            _spherical_triangle_area(here, first_middle, vertex_unit)
            + _spherical_triangle_area(here, vertex_unit, second_middle)
        )

    area_triangle = kite_areas.sum(axis=1)
    area_cell = np.zeros(n_cells, dtype=np.float64)
    np.add.at(area_cell, triangles.ravel(), kite_areas.ravel())

    # Construction diagnostics, measured in binary64 on the exact figure.
    # They are what proves the instrument: the tests must not re-derive them
    # from the binary32 arrays, because binary32 is the thing under test.
    direct_triangle = radius * radius * _spherical_triangle_area(
        corner[0], corner[1], corner[2]
    )
    exact_dv = _arc_length(
        vertex_unit[vertices_on_edge[:, 0]], vertex_unit[vertices_on_edge[:, 1]], radius
    )
    exact_dc = _arc_length(unit[cells_on_edge[:, 0]], unit[cells_on_edge[:, 1]], radius)
    diagnostics = {
        # A circumcentre outside its triangle would make the three unsigned
        # kites overlap and overshoot the triangle they are meant to tile.
        "kite_tiling_max_relative_error": float(
            np.max(np.abs(kite_areas.sum(axis=1) - direct_triangle) / direct_triangle)
        ),
        "sphere_closure_relative_error": float(
            abs(area_cell.sum() / (4.0 * math.pi * radius * radius) - 1.0)
        ),
        "min_dc_metres": float(exact_dc.min()),
        "max_dc_metres": float(exact_dc.max()),
        "min_dv_metres": float(exact_dv.min()),
        "max_dv_metres": float(exact_dv.max()),
    }

    cell_positions = _angles_and_positions(unit, radius, dtype)
    vertex_positions = _angles_and_positions(vertex_unit, radius, dtype)
    edge_unit = unit[cells_on_edge[:, 0]] + unit[cells_on_edge[:, 1]]
    edge_unit /= np.linalg.norm(edge_unit, axis=1)[:, None]
    edge_positions = _angles_and_positions(edge_unit, radius, dtype)

    arrays: dict[str, np.ndarray] = {
        "nEdgesOnCell": n_edges_on_cell.astype(np.int32),
        "cellsOnCell": cells_on_cell,
        "edgesOnCell": edges_on_cell,
        "verticesOnCell": vertices_on_cell,
        "cellsOnEdge": cells_on_edge,
        "verticesOnEdge": vertices_on_edge,
        "cellsOnVertex": triangles,
        "edgesOnVertex": edges_on_vertex,
        "nEdgesOnEdge": n_edges_on_edge.astype(np.int32),
        "edgesOnEdge": edges_on_edge,
        "weightsOnEdge": np.zeros((n_edges, max_edges2), dtype=dtype),
        "dcEdge": _arc_length(
            unit[cells_on_edge[:, 0]], unit[cells_on_edge[:, 1]], radius
        ).astype(dtype),
        "dvEdge": _arc_length(
            vertex_unit[vertices_on_edge[:, 0]], vertex_unit[vertices_on_edge[:, 1]], radius
        ).astype(dtype),
        "areaCell": area_cell.astype(dtype),
        "areaTriangle": area_triangle.astype(dtype),
        "kiteAreasOnVertex": kite_areas.astype(dtype),
    }
    for suffix, positions in (
        ("Cell", cell_positions),
        ("Edge", edge_positions),
        ("Vertex", vertex_positions),
    ):
        for axis in ("x", "y", "z"):
            arrays[f"{axis}{suffix}"] = positions[axis]
        arrays[f"lat{suffix}"] = positions["lat"]
        arrays[f"lon{suffix}"] = positions["lon"]

    mesh = Mesh(
        arrays=arrays,
        dimensions={
            "nCells": n_cells,
            "nEdges": n_edges,
            "nVertices": n_vertices,
            "maxEdges": max_edges,
            "maxEdges2": max_edges2,
            "vertexDegree": 3,
        },
        attrs={"on_a_sphere": "YES", "sphere_radius": radius},
    )
    mesh.provenance["synthetic_construction"] = diagnostics
    return mesh
