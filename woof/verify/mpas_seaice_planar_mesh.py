"""A planar MPAS mesh block (hexagons or quadrilaterals), for the sea-ice remap oracles.

MPAS-Seaice's incremental remap works in each cell's and each edge's local
plane from the mesh coordinates (``xVertex - xCell``, ``xVertex - xEdge``), so
its test meshes need real geometry, not a periodic torus: :func:`planar_mesh`
lays hexagonal (``vertexDegree`` 3) or square (``vertexDegree`` 4) cells on a
plane, jitters the vertices, culls a few cells as land, and keeps one block of a
decomposition: the owned cells and ``n_halos`` layers of neighbours, numbered by
layer (so ``nCellsSolve`` < ``nCells``), with every slot that points outside the
block at the missing cell ``nCells+1`` (``nEdges+1``, ``nVertices+1``).

The conventions are MPAS's and the ones v8.4.2's remap reads:

- a cell's vertices go counter-clockwise, and ``edgesOnCell(i)`` joins
  ``verticesOnCell(i)`` and ``verticesOnCell(i+1)``, across from
  ``cellsOnCell(i)``;
- ``cellsOnEdge(1)`` lies to the left of ``verticesOnEdge(1)`` ->
  ``verticesOnEdge(2)`` (which cell is first is drawn at random);
- ``cellsOnVertex``/``edgesOnVertex`` go round the vertex.

Coordinates are metres, ``z = 0``; a cell's centre is its vertices' mean.
Only correctly rounded operations build the mesh (no libm trigonometry, no
BLAS), so its float64 words are the same on every host.
Arrays are 1-based in MPAS's C-order-reversed layout (``edgesOnCell`` is
``(nCells+1, maxEdges)``) and float64; the caller casts.
"""
from __future__ import annotations

import math

import numpy as np


_R3 = math.sqrt(3.0)
#: the hexagon's corners at 30 + 60k degrees, radius 1/sqrt(3) (cos and sin in closed form)
_HEX_CORNERS = ((0.5, 0.5 / _R3), (0.0, 1.0 / _R3), (-0.5, 0.5 / _R3), (-0.5, -0.5 / _R3), (0.0, -1.0 / _R3),
                (0.5, -0.5 / _R3))


def _length(dx: float, dy: float) -> float:
    return math.sqrt(dx * dx + dy * dy)


def _lattice(kind: str, extent: int):
    """Cells of an unbounded lattice near the origin: centre key -> its vertex
    keys counter-clockwise, and a vertex key -> position map."""
    cells, positions = {}, {}
    if kind == "quad":
        for i in range(-extent, extent + 1):
            for j in range(-extent, extent + 1):
                corners = [(2 * i - 1, 2 * j - 1), (2 * i + 1, 2 * j - 1), (2 * i + 1, 2 * j + 1),
                           (2 * i - 1, 2 * j + 1)]
                cells[(i, j)] = corners
                for k in corners:
                    positions[k] = (0.5 * k[0], 0.5 * k[1])
    elif kind == "hex":
        for q in range(-extent, extent + 1):
            for r in range(-extent, extent + 1):
                cx, cy = q + 0.5 * r, math.sqrt(3.0) / 2.0 * r
                corners = []
                for k in range(6):
                    # the corner at 30 + 60k degrees, 1/sqrt(3) out (no libm: the same words on every host)
                    dx, dy = _HEX_CORNERS[k]
                    x, y = cx + dx, cy + dy
                    key = (round(x * 6.0), round(y * 6.0 / math.sqrt(3.0)))
                    positions[key] = (x, y)
                    corners.append(key)
                cells[(q, r)] = corners
    else:
        raise ValueError(f"mesh kind is 'hex' or 'quad', not {kind!r}")
    return cells, positions


def planar_mesh(kind: str, rs: np.random.RandomState, *, owned: int = 3, n_halos: int | None = None,
                spacing: float = 1.0e4, jitter: float = 0.06, land: int = 2, offset=(3.0e4, -2.0e4)) -> dict:
    """One block: the owned cells within ``owned`` lattice steps of the origin
    (less ``land`` culled ones), ``n_halos`` (default 2 for hexagons, 3 for
    quadrilaterals, as v8.4.2's remap requires) halo layers around them."""
    vd = 3 if kind == "hex" else 4
    me = 6 if kind == "hex" else 4
    n_halos = (2 if kind == "hex" else 3) if n_halos is None else n_halos
    cells, positions = _lattice(kind, owned + n_halos + 2)
    keys = sorted(cells)

    def owned_cell(key):
        if kind == "quad":
            return max(abs(key[0]), abs(key[1])) <= owned
        q, r = key
        return max(abs(q), abs(r), abs(q + r)) <= owned

    # the edges of the lattice: a vertex pair -> the cells on it
    edge_cells: dict[tuple, list] = {}
    for key in keys:
        cs = cells[key]
        for i in range(len(cs)):
            edge_cells.setdefault(frozenset((cs[i], cs[(i + 1) % len(cs)])), []).append(key)
    neighbours = {key: [] for key in keys}
    for pair, cs in edge_cells.items():
        if len(cs) == 2:
            neighbours[cs[0]].append(cs[1])
            neighbours[cs[1]].append(cs[0])

    culled = set()
    candidates = [key for key in keys if owned_cell(key) and key != (0, 0)]
    for i in rs.permutation(len(candidates))[:land]:
        culled.add(candidates[i])
    layer = {key: 0 for key in keys if owned_cell(key) and key not in culled}
    front = list(layer)
    for depth in range(1, n_halos + 1):
        nxt = []
        for key in front:
            for n in neighbours[key]:
                if n not in layer and n not in culled:
                    layer[n] = depth
                    nxt.append(n)
        front = nxt
    block = sorted(layer, key=lambda k: (layer[k], k))
    nc = len(block)
    cell_id = {key: i + 1 for i, key in enumerate(block)}

    # vertices and edges of the block's cells, numbered by their least layer
    vertex_layer: dict = {}
    edge_layer: dict = {}
    for key in block:
        cs = cells[key]
        for i in range(len(cs)):
            vertex_layer[cs[i]] = min(vertex_layer.get(cs[i], 99), layer[key])
            e = frozenset((cs[i], cs[(i + 1) % len(cs)]))
            edge_layer[e] = min(edge_layer.get(e, 99), layer[key])
    vertices = sorted(vertex_layer, key=lambda v: (vertex_layer[v], v))
    edges = sorted(edge_layer, key=lambda e: (edge_layer[e], sorted(e)))
    nv, ne = len(vertices), len(edges)
    vertex_id = {v: i + 1 for i, v in enumerate(vertices)}
    edge_id = {e: i + 1 for i, e in enumerate(edges)}

    # geometry: jittered vertices, centres at the vertex mean
    xv = np.zeros(nv + 1)
    yv = np.zeros(nv + 1)
    for v, i in vertex_id.items():
        px, py = positions[v]
        xv[i - 1] = offset[0] + spacing * (px + jitter * (2.0 * rs.random_sample() - 1.0))
        yv[i - 1] = offset[1] + spacing * (py + jitter * (2.0 * rs.random_sample() - 1.0))
    xc = np.zeros(nc + 1)
    yc = np.zeros(nc + 1)
    n_edges_on_cell = np.zeros(nc + 1, np.int32)
    vertices_on_cell = np.full((nc + 1, me), nv + 1, np.int32)
    edges_on_cell = np.full((nc + 1, me), ne + 1, np.int32)
    cells_on_cell = np.full((nc + 1, me), nc + 1, np.int32)
    area = np.zeros(nc + 1)
    for key in block:
        c = cell_id[key] - 1
        cs = cells[key]
        n = len(cs)
        n_edges_on_cell[c] = n
        vx = [float(xv[vertex_id[v] - 1]) for v in cs]
        vy = [float(yv[vertex_id[v] - 1]) for v in cs]
        # plain float sums in order (no BLAS, no SIMD reductions: the same words on every host)
        xc[c], yc[c] = sum(vx) / n, sum(vy) / n
        twice = 0.0
        for j in range(n):
            twice += vx[j] * vy[(j + 1) % n] - vy[j] * vx[(j + 1) % n]
        area[c] = 0.5 * abs(twice)
        # start each cell's ring at a random vertex (MPAS meshes do not align them)
        start = rs.randint(n)
        for j in range(n):
            a, b = cs[(start + j) % n], cs[(start + j + 1) % n]
            vertices_on_cell[c, j] = vertex_id[a]
            e = frozenset((a, b))
            edges_on_cell[c, j] = edge_id[e]
            other = [k for k in edge_cells[e] if k != key]
            if other and other[0] in cell_id:
                cells_on_cell[c, j] = cell_id[other[0]]

    cells_on_edge = np.full((ne + 1, 2), nc + 1, np.int32)
    vertices_on_edge = np.full((ne + 1, 2), nv + 1, np.int32)
    xe = np.zeros(ne + 1)
    ye = np.zeros(ne + 1)
    dc = np.zeros(ne + 1)
    dv = np.zeros(ne + 1)
    for e, i in edge_id.items():
        present = [k for k in edge_cells[e] if k in cell_id]
        first = present[rs.randint(len(present))]
        cs = cells[first]
        j = next(j for j in range(len(cs)) if frozenset((cs[j], cs[(j + 1) % len(cs)])) == e)
        # the first cell's counter-clockwise ring runs V1 -> V2, so it lies to their left
        v1, v2 = vertex_id[cs[j]], vertex_id[cs[(j + 1) % len(cs)]]
        vertices_on_edge[i - 1] = [v1, v2]
        others = [k for k in present if k != first]
        cells_on_edge[i - 1] = [cell_id[first], cell_id[others[0]] if others else nc + 1]
        xe[i - 1] = 0.5 * (xv[v1 - 1] + xv[v2 - 1])
        ye[i - 1] = 0.5 * (yv[v1 - 1] + yv[v2 - 1])
        dv[i - 1] = _length(xv[v2 - 1] - xv[v1 - 1], yv[v2 - 1] - yv[v1 - 1])
        c1 = cell_id[first] - 1
        if others:
            c2 = cell_id[others[0]] - 1
            dc[i - 1] = _length(xc[c2] - xc[c1], yc[c2] - yc[c1])
        else:
            dc[i - 1] = 2.0 * _length(xe[i - 1] - xc[c1], ye[i - 1] - yc[c1])

    # round each vertex: its cells and edges counter-clockwise
    cells_on_vertex = np.full((nv + 1, vd), nc + 1, np.int32)
    edges_on_vertex = np.full((nv + 1, vd), ne + 1, np.int32)
    vertex_cells: dict = {}
    vertex_edges: dict = {}
    for e in edge_cells:
        for v in e:
            vertex_edges.setdefault(v, []).append(e)
    for key in keys:
        for v in cells[key]:
            vertex_cells.setdefault(v, []).append(key)
    for v, i in vertex_id.items():
        around = vertex_cells[v]
        px, py = positions[v]

        def angle(point):
            return math.atan2(point[1] - py, point[0] - px)

        centres = {}
        for key in around:
            cx = sum(positions[w][0] for w in cells[key]) / len(cells[key])
            cy = sum(positions[w][1] for w in cells[key]) / len(cells[key])
            centres[key] = (cx, cy)
        ordered = sorted(around, key=lambda k: angle(centres[k]))
        for j, key in enumerate(ordered[:vd]):
            if key in cell_id:
                cells_on_vertex[i - 1, j] = cell_id[key]
        incident = [e for e in vertex_edges[v]]
        mids = {e: tuple(sum(positions[w][t] for w in e) / 2.0 for t in range(2)) for e in incident}
        for j, e in enumerate(sorted(incident, key=lambda e: angle(mids[e]))[:vd]):
            if e in edge_id:
                edges_on_vertex[i - 1, j] = edge_id[e]

    def counts(items, layer_of):
        layers = [layer_of(x) for x in items]
        return np.array([sum(1 for lay in layers if lay <= k) for k in range(n_halos + 1)], np.int32)

    return {
        "kind": kind, "nCells": nc, "nEdges": ne, "nVertices": nv, "maxEdges": me, "vertexDegree": vd,
        "nHalos": n_halos,
        "nCellsArray": counts(block, lambda k: layer[k]),
        "nEdgesArray": counts(edges, lambda e: edge_layer[e]),
        "nVerticesArray": counts(vertices, lambda v: vertex_layer[v]),
        "nEdgesOnCell": n_edges_on_cell, "edgesOnCell": edges_on_cell, "verticesOnCell": vertices_on_cell,
        "cellsOnCell": cells_on_cell, "cellsOnEdge": cells_on_edge, "verticesOnEdge": vertices_on_edge,
        "cellsOnVertex": cells_on_vertex, "edgesOnVertex": edges_on_vertex,
        "xCell": xc, "yCell": yc, "zCell": np.zeros(nc + 1), "xVertex": xv, "yVertex": yv,
        "zVertex": np.zeros(nv + 1), "xEdge": xe, "yEdge": ye, "zEdge": np.zeros(ne + 1),
        "dcEdge": dc, "dvEdge": dv, "areaCell": area,
    }
