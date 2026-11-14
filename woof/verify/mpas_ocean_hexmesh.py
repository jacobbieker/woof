"""A small MPAS-Ocean mesh with consistent connectivity, for the multi-step oracles.

:func:`hex_mesh` lays hexagonal cells on a doubly periodic torus (axial
coordinates mod ``(nq, nr)``), culls a share of them as land the way MPAS's
culled meshes are (an edge, vertex or neighbour slot that pointed at a land
cell points at the missing cell ``nCells+1``; an edge between two land cells,
or a vertex among three, is gone), and derives the vertical extents, masks,
signs and kite indices exactly as v8.4.2's ``ocn_init_routines_compute_max_level``
and ``ocn_init_routines_setup_sign_and_index_fields`` do. The geometry
(lengths, areas, weights) is the caller's to draw: only the topology must hold
for MPAS's sentinels (``-1e34`` below ``maxLevelEdgeBot``) never to be read
where a real mesh does not read them.

Connectivity is 1-based, in MPAS's (C-order reversed) layout: ``edgesOnCell``
is ``(nCells+1, maxEdges)`` and so on.
"""
from __future__ import annotations

import numpy as np

#: the six neighbour directions, in order around a cell
DIRECTIONS = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))


def hex_mesh(nq: int, nr: int, land_fraction: float, n_levels: int, rs: np.random.RandomState,
             min_levels: int = 2, owned_columns: int | None = None, n_halos: int = 3) -> dict:
    """The mesh's integer fields and counts; ``maxLevelCell`` drawn in
    ``[min_levels, n_levels]`` (the first three ocean cells full depth).

    ``owned_columns`` makes the mesh one block of a decomposition: the ocean
    cells of the first ``owned_columns`` torus columns are the block's own,
    ``n_halos`` layers of neighbours (by ocean adjacency) its halo, and the
    rest outside it. Cells, edges and vertices are numbered by halo layer (an
    edge's or vertex's layer the least of its cells'), so ``nCellsArray(i)``
    and the others count those up to layer ``i-1``; what points outside the
    block points at the missing cell, as MPAS's outermost halo does."""
    n_all = nq * nr

    def cell(q, r):
        return (q % nq) * nr + (r % nr)

    def neighbour(c, k):
        q, r = divmod(c, nr)
        dq, dr = DIRECTIONS[k]
        return cell(q + dq, r + dr)

    land = rs.random_sample(n_all) < land_fraction
    land[0] = False
    # the halo layer of each ocean cell (0 owned); everything is owned without a decomposition
    layer = np.zeros(n_all, np.int64)
    if owned_columns is not None:
        layer[:] = n_halos + 1
        front = [c for c in range(n_all) if not land[c] and c // nr < owned_columns]
        layer[front] = 0
        for depth in range(1, n_halos + 1):
            nxt = []
            for c in front:
                for k in range(6):
                    n = neighbour(c, k)
                    if not land[n] and layer[n] > depth:
                        layer[n] = depth
                        nxt.append(n)
            front = nxt
        land = land | (layer > n_halos)
    ocean = np.array(sorted(np.nonzero(~land)[0], key=lambda c: (layer[c], c)), np.int64)
    nc = len(ocean)
    new_cell = np.full(n_all, nc, np.int64)          # 0-based; nc is the missing cell
    new_cell[ocean] = np.arange(nc)

    # every torus edge (c, k) for k in 0..2, every torus vertex U(a)=2a, D(a)=2a+1
    def edge_id(c, k):
        if k < 3:
            return 3 * c + k
        return 3 * neighbour(c, k) + (k - 3)

    def vertex_id(c, k):
        base = (c, c, neighbour(c, 3), neighbour(c, 4), neighbour(c, 4), neighbour(c, 5))[k]
        return 2 * base + (0, 1, 0, 1, 0, 1)[k]

    edge_cells = {3 * c + k: (c, neighbour(c, k)) for c in range(n_all) for k in range(3)}
    vertex_cells = {}
    for c in range(n_all):
        for k in range(6):
            vertex_cells.setdefault(vertex_id(c, k), []).append(c)
    def least_layer(cells):
        return min(layer[c] for c in cells if not land[c])

    kept_edges = sorted((e for e in edge_cells if not (land[edge_cells[e][0]] and land[edge_cells[e][1]])),
                        key=lambda e: (least_layer(edge_cells[e]), e))
    kept_vertices = sorted((v for v in vertex_cells if not all(land[c] for c in vertex_cells[v])),
                           key=lambda v: (least_layer(vertex_cells[v]), v))
    ne, nv = len(kept_edges), len(kept_vertices)
    new_edge = {e: i for i, e in enumerate(kept_edges)}
    new_vertex = {v: i for i, v in enumerate(kept_vertices)}

    me, me2, vd = 6, 10, 3
    edges_on_cell = np.full((nc + 1, me), ne + 1, np.int32)
    vertices_on_cell = np.full((nc + 1, me), nv + 1, np.int32)
    cells_on_cell = np.full((nc + 1, me), nc + 1, np.int32)
    n_edges_on_cell = np.zeros(nc + 1, np.int32)
    for i, c in enumerate(ocean):
        n_edges_on_cell[i] = 6
        for k in range(6):
            edges_on_cell[i, k] = new_edge[edge_id(c, k)] + 1
            vertices_on_cell[i, k] = new_vertex[vertex_id(c, k)] + 1
            cells_on_cell[i, k] = new_cell[neighbour(c, k)] + 1
    cells_on_edge = np.full((ne + 1, 2), nc + 1, np.int32)
    vertices_on_edge = np.full((ne + 1, 2), nv + 1, np.int32)
    edges_on_edge = np.full((ne + 1, me2), ne + 1, np.int32)
    n_edges_on_edge = np.zeros(ne + 1, np.int32)
    for i, e in enumerate(kept_edges):
        c, k = divmod(e, 3)
        cells_on_edge[i] = [new_cell[c] + 1, new_cell[edge_cells[e][1]] + 1]
        vertices_on_edge[i] = [new_vertex[vertex_id(c, (k - 1) % 6)] + 1, new_vertex[vertex_id(c, k)] + 1]
        # the other edges of the edge's ocean cells, going round each from the edge
        others = []
        for owner, kk in ((c, k), (edge_cells[e][1], k + 3)):
            if land[owner]:
                continue
            others += [new_edge[edge_id(owner, (kk + j) % 6)] + 1 for j in range(1, 6)]
        n_edges_on_edge[i] = len(others)
        edges_on_edge[i, :len(others)] = others
    cells_on_vertex = np.full((nv + 1, vd), nc + 1, np.int32)
    edges_on_vertex = np.full((nv + 1, vd), ne + 1, np.int32)
    for i, v in enumerate(kept_vertices):
        cs = vertex_cells[v]
        cells_on_vertex[i] = [new_cell[c] + 1 for c in cs]
        pairs = [(cs[0], cs[1]), (cs[1], cs[2]), (cs[2], cs[0])]
        for j, (a, b) in enumerate(pairs):
            for k in range(6):
                if neighbour(a, k) == b:
                    e = edge_id(a, k)
                    edges_on_vertex[i, j] = new_edge[e] + 1 if e in new_edge else ne + 1

    # the vertical extents and masks (ocn_init_routines_compute_max_level)
    nl = n_levels
    mlc = np.zeros(nc + 1, np.int32)
    mlc[:nc] = rs.randint(min_levels, nl + 1, nc)
    mlc[:min(nc, 3)] = nl
    mlet = np.minimum(mlc[cells_on_edge[:, 0] - 1], mlc[cells_on_edge[:, 1] - 1]).astype(np.int32)
    mleb = np.maximum(mlc[cells_on_edge[:, 0] - 1], mlc[cells_on_edge[:, 1] - 1]).astype(np.int32)
    mlet[ne] = mleb[ne] = 0
    mlvb = mlc[cells_on_vertex - 1].max(axis=1).astype(np.int32)
    mlvt = mlc[cells_on_vertex - 1].min(axis=1).astype(np.int32)
    mlvb[nv] = mlvt[nv] = 0
    level = np.arange(nl)[None, :]
    edge_mask = (level < mlet[:, None]).astype(np.int32)
    boundary_edge = 1 - edge_mask
    boundary_cell = np.zeros((nc + 1, nl), np.int32)
    for e in range(ne):
        for col in (0, 1):
            boundary_cell[cells_on_edge[e, col] - 1] |= boundary_edge[e]

    # signs and kite indices (ocn_init_routines_setup_sign_and_index_fields)
    edge_sign_on_cell = np.zeros((nc + 1, me), np.int32)
    kite_index = np.zeros((nc + 1, me), np.int32)
    for c in range(nc):
        for i in range(n_edges_on_cell[c]):
            e, v = edges_on_cell[c, i] - 1, vertices_on_cell[c, i] - 1
            edge_sign_on_cell[c, i] = -1 if c + 1 == cells_on_edge[e, 0] else 1
            for j in range(vd):
                if cells_on_vertex[v, j] == c + 1:
                    kite_index[c, i] = j + 1
    edge_sign_on_vertex = np.zeros((nv + 1, vd), np.int32)
    for v in range(nv):
        for i in range(vd):
            e = edges_on_vertex[v, i] - 1
            edge_sign_on_vertex[v, i] = -1 if v + 1 == vertices_on_edge[e, 0] else 1

    def counts(items, layer_of):
        layers = [layer_of(x) for x in items]
        return np.array([sum(1 for lay in layers if lay <= i) for i in range(n_halos + 1)], np.int32)

    return {
        "nCellsArray": counts(ocean, lambda c: layer[c]),
        "nEdgesArray": counts(kept_edges, lambda e: least_layer(edge_cells[e])),
        "nVerticesArray": counts(kept_vertices, lambda v: least_layer(vertex_cells[v])),
        "nCells": nc, "nEdges": ne, "nVertices": nv, "maxEdges": me, "maxEdges2": me2, "vertexDegree": vd,
        "nEdgesOnCell": n_edges_on_cell, "edgesOnCell": edges_on_cell, "verticesOnCell": vertices_on_cell,
        "cellsOnCell": cells_on_cell, "cellsOnEdge": cells_on_edge, "verticesOnEdge": vertices_on_edge,
        "nEdgesOnEdge": n_edges_on_edge, "edgesOnEdge": edges_on_edge, "cellsOnVertex": cells_on_vertex,
        "edgesOnVertex": edges_on_vertex, "maxLevelCell": mlc, "maxLevelEdgeTop": mlet, "maxLevelEdgeBot": mleb,
        "maxLevelVertexBot": mlvb, "maxLevelVertexTop": mlvt, "edgeMask": edge_mask,
        "boundaryCell": boundary_cell, "edgeSignOnCell": edge_sign_on_cell, "kiteIndexOnCell": kite_index,
        "edgeSignOnVertex": edge_sign_on_vertex,
    }
