"""MPAS-Seaice v8.4.2: the incremental remap's init geometry, computed on the host.

``seaice_init_advection_incremental_remap`` (``mpas_seaice_advection_incremental_remap.F``)
derives, once, the geometry the remap runs on:

- ``check_halo_layer_number``: the halo layers against the vertex degree;
- ``rotate_global_vectors`` (on a sphere under ``config_rotate_cartesian_grid``:
  the poles turned to the equator, into the ``rotated_mesh`` pool);
- ``define_local_to_global_transformations``: each cell's, vertex's and edge's
  local frame (``trans*`` in the ``incremental_remap`` pool);
- ``get_vertex_on_cell_coordinates``: each cell's vertices in its tangent
  plane (and v8.4.2's checks that they run counter-clockwise);
- ``get_geometry_incremental_remap``: the edges that carry fluxes
  (``remapEdge``), each edge's remap neighbours (``cellsOnEdgeRemap``,
  ``edgesOnEdgeRemap``), the vertices about each edge in its plane
  (``xVertexOnEdge``/``yVertexOnEdge``) and the shortest edge at each vertex;
- ``compute_geometric_cell_averages``: the cells' moments (``xAvgCell`` ...);
- MPAS's RBF reconstruction (``mpas_rbf_interp_initialize`` =
  ``mpas_initialize_vectors``, then ``mpas_init_reconstruct`` with
  ``mpas_rbf_interp_func_3D_plane_vec_const_dir_comp_coeffs``, its
  inverse-multiquadric matrix and Tao Pang's partial-pivoting elimination,
  ``mpas_legs``/``elgs``): ``coeffs_reconstruct`` for the owned cells (zero
  in the halo: one rank, the halo exchange changes nothing), with the mesh
  pool's ``localVerticalUnitVectors``, ``edgeNormalVectors`` and
  ``cellTangentPlane``.

It runs once, on the host, in numpy at the run's precision (float32 for
RKIND 4, float64 for RKIND 8), one IEEE operation at a time in the Fortran's
order (gfortran -O0 -ffp-contract=off: left to right, ``x**2`` = ``x*x``,
``x**3`` = ``(x*x)*x``, ``x**4`` = ``(x*x)*(x*x)``, ``SUM`` from zero), so
its words are v8.4.2's on any host (``tests/test_mpas_seaice_remap_init_parity.py``).
Only + - * / sqrt, comparisons and max are used; numpy does each correctly
rounded.

Refused (raised) where v8.4.2 stops (MPAS_LOG_CRIT): too few halo layers,
cells whose vertices do not run counter-clockwise, an edge whose cell 1 is
not in its left half-plane, a non-positive owned cell area, and topology
v8.4.2 would read outside its arrays with. Periodic planar meshes
(``is_periodic``) are refused by name.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from woof.hex.errors import ConfigurationRefusal

from .remap_v842 import RemapError

SOURCE_FILES = (
    "src/core_seaice/shared/mpas_seaice_advection_incremental_remap.F:61-77,172-823,836-881,955-2330,"
    "8961-8979,9117-9184,9262-9295",
    "src/operators/mpas_vector_operations.F:95-121,652-771",
    "src/operators/mpas_rbf_interpolation.F:131-139,1079-1145,1369-1376,1527-1559,1670-1849",
    "src/operators/mpas_vector_reconstruction.F:61-191",
)

#: the moments compute_geometric_cell_averages fills, in its order
MOMENTS = ("x", "y", "xx", "xy", "yy", "xxx", "xxy", "xyy", "yyy", "xxxx", "xxxy", "xxyy", "xyyy", "yyyy")
#: the remap pool's frames: (local to global, global to local) for cells, vertices, edges
FRAMES = (("transCellToGlobal", "transGlobalToCell"), ("transVertexToGlobal", "transGlobalToVertex"),
          ("transEdgeToGlobal", "transGlobalToEdge"))


def literal(text: str, real) -> object:
    """A Fortran literal written ``_RKIND``: the decimal rounded once to ``real``."""
    exact = Fraction(text)
    if real is np.float64:
        return np.float64(float(exact))
    guess = np.float32(float(exact))
    candidates = (np.nextafter(guess, np.float32(-np.inf)), guess, np.nextafter(guess, np.float32(np.inf)))
    best = min(candidates, key=lambda c: (abs(Fraction(float(c)) - exact),
                                          int(np.asarray(c).view(np.uint32)) & 1))
    return np.float32(best)


class _Constants:
    def __init__(self, real):
        self.real = real
        self.zero, self.one, self.half, self.quarter = real(0.0), real(1.0), real(0.5), real(0.25)
        self.eps11 = literal("1.0e-11", real)
        self.w1 = literal("1.09951743655321885e-01", real)
        self.w2 = literal("2.23381589678011389e-01", real)
        self.q1 = literal("9.15762135097710761e-02", real)
        self.q2 = literal("8.16847572980458514e-01", real)
        self.q3 = literal("1.08103018168070275e-01", real)
        self.q4 = literal("4.45948490915965612e-01", real)
        self.huge = np.finfo(real).max


def _sum3(a1, b1, a2, b2, a3, b3, zero):
    """gfortran's inline SUM of a three-element product: from zero, left to right."""
    return ((zero + a1 * b1) + a2 * b2) + a3 * b3


def _unit_vector_3d(x, y, z, c):
    """``unit_vector_3d`` (remap.F:8961-8979): zero where the magnitude is not positive."""
    magnitude = np.sqrt((x * x + y * y) + z * z)
    positive = magnitude > c.zero
    safe = np.where(positive, magnitude, c.one)
    return (np.where(positive, x / safe, c.zero), np.where(positive, y / safe, c.zero),
            np.where(positive, z / safe, c.zero))


def _unit_vec_in_r3(x, y, z):
    """``mpas_unit_vec_in_r3`` (mpas_vector_operations.F:95-101): no zero guard."""
    mag = np.sqrt((x * x + y * y) + z * z)
    with np.errstate(divide="ignore", invalid="ignore"):
        return x / mag, y / mag, z / mag


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def check_halo_layer_number(configs: dict, vertex_degree: int) -> list[dict]:
    """``check_halo_layer_number``: refused below the optimum (v8.4.2 stops);
    above it, the three warnings v8.4.2 logs (its first line is an
    information message)."""
    optimal = {3: 2, 4: 3}.get(int(vertex_degree))
    if optimal is None:
        raise ConfigurationRefusal(knob="vertexDegree", value=vertex_degree,
                                   reason="v8.4.2's remap knows hexagonal (3) and quadrilateral (4) meshes only",
                                   declaration="vertexDegree = 3 or 4")
    halos = int(configs.get("config_num_halos", optimal))
    if halos < optimal:
        raise ConfigurationRefusal(knob="config_num_halos", value=halos,
                                   reason=f"IR: too few halo layers for incremental remapping (vertexDegree "
                                          f"{vertex_degree} needs {optimal}; v8.4.2 stops)",
                                   declaration=f"config_num_halos >= {optimal}")
    if halos > optimal:
        return [{"message": "-- vertexDegree:     $i", "ints": [int(vertex_degree)], "reals": []},
                {"message": "-- config_num_halos: $i", "ints": [halos], "reals": []},
                {"message": "-- numHalosOptimal:  $i", "ints": [optimal], "reals": []}]
    return []


def define_frames(x, y, z, n: int, c) -> tuple[np.ndarray, np.ndarray]:
    """``define_local_to_global_transformations`` for elements 1..n: the
    (local to global, global to local) frames, C-order ``(n, 3, 3)`` with
    ``[e, j, i]`` = Fortran ``(i, j, e)``."""
    x, y, z = x[:n], y[:n], z[:n]
    e3 = _unit_vector_3d(x, y, z, c)
    pole = ~(np.abs(x * x + y * y) > c.eps11)
    e1 = _unit_vector_3d(-y, x, np.zeros_like(x), c)
    e2 = _cross(e3, e1)
    north = pole & (z > c.zero)
    south = pole & ~(z > c.zero)
    ones, zeros = np.full_like(x, c.one), np.zeros_like(x)
    e1 = [np.where(north, o, np.where(south, zr, v)) for v, o, zr in zip(e1, (ones, zeros, zeros), (zeros, ones, zeros))]
    e2 = [np.where(north, o, np.where(south, zr, v)) for v, o, zr in zip(e2, (zeros, ones, zeros), (ones, zeros, zeros))]
    # unitVectorLocal(i, k): component i of local axis k
    local = np.stack([np.stack(e1, -1), np.stack(e2, -1), np.stack(e3, -1)], -1)   # (n, i, k)
    # matmul with the identity (libgfortran: from zero, k order): each entry its one product plus zeros
    zero = c.real(0.0)
    l2g = np.empty((n, 3, 3), x.dtype)      # [e, j, i] = sum_k I(k,i) * local(k,j)
    g2l = np.empty((n, 3, 3), x.dtype)      # [e, j, i] = sum_k local(k,i) * I(k,j)
    eye = np.eye(3, dtype=x.dtype)
    for i in range(3):
        for j in range(3):
            acc = np.full(n, zero, x.dtype)
            for k in range(3):
                acc = acc + eye[k, i] * local[:, k, j]
            l2g[:, j, i] = acc
            acc = np.full(n, zero, x.dtype)
            for k in range(3):
                acc = acc + local[:, k, i] * eye[k, j]
            g2l[:, j, i] = acc
    return l2g, g2l


def _to_plane(T, e, w1, w2, w3):
    """Rows 1 and 2 of ``T(:,:,e)`` times ``w`` (remap.F's three-term sums)."""
    return ((T[e, 0, 0] * w1 + T[e, 1, 0] * w2) + T[e, 2, 0] * w3,
            (T[e, 0, 1] * w1 + T[e, 1, 1] * w2) + T[e, 2, 1] * w3)


def vertex_on_cell_coordinates(m: dict, g: dict, sphere: bool, c) -> tuple[np.ndarray, np.ndarray]:
    """``get_vertex_on_cell_coordinates``: written over each cell's vertices;
    raises where v8.4.2's counter-clockwise checks stop it."""
    nc = int(m["nCells"])
    neoc = m["nEdgesOnCell"][:nc]
    voc = m["verticesOnCell"]
    xvoc, yvoc = m["xVertexOnCell"].copy(), m["yVertexOnCell"].copy()
    xV, yV, zV, xC, yC, zC = g["xVertex"], g["yVertex"], g["zVertex"], g["xCell"], g["yCell"], g["zCell"]
    T = g.get("transGlobalToCell")
    for k in range(int(neoc.max(initial=0))):
        cells = np.flatnonzero(k < neoc)
        v = voc[cells, k] - 1
        w1, w2, w3 = xV[v] - xC[cells], yV[v] - yC[cells], zV[v] - zC[cells]
        if sphere:
            xvoc[cells, k], yvoc[cells, k] = _to_plane(T, cells, w1, w2, w3)
        else:
            xvoc[cells, k], yvoc[cells, k] = xV[v] - xC[cells], yV[v] - yC[cells]
    for k in range(int(neoc.max(initial=0))):
        cells = np.flatnonzero(k < neoc)
        kp1 = np.where(k + 1 >= neoc[cells], 0, k + 1)
        if sphere:
            v1, v2 = voc[cells, k] - 1, voc[cells, kp1] - 1
            a = (xV[v1] - xC[cells], yV[v1] - yC[cells], zV[v1] - zC[cells])
            b = (xV[v2] - xC[cells], yV[v2] - yC[cells], zV[v2] - zC[cells])
            w = _cross(a, b)
            dot = (w[0] * xC[cells] + w[1] * yC[cells]) + w[2] * zC[cells]
            if not (dot >= c.zero).all():
                bad = int(cells[np.flatnonzero(~(dot >= c.zero))[0]]) + 1
                raise RemapError(f"IR requires convex cells with vertices arranged CCW around the cell (cell {bad}; "
                                 "v8.4.2 stops, MPAS_LOG_CRIT)")
        cross = xvoc[cells, k] * yvoc[cells, kp1] - yvoc[cells, k] * xvoc[cells, kp1]
        if not (cross >= c.zero).all():
            bad = int(cells[np.flatnonzero(~(cross >= c.zero))[0]]) + 1
            raise RemapError("MPAS-seaice: IR requires convex cells with vertices arranged CCW around the cell: "
                             f"check (cell {bad}; v8.4.2 stops, MPAS_LOG_CRIT)")
    return xvoc, yvoc


def _point_in_half_plane(point, start, end, zero) -> bool:
    v1 = (end[0] - start[0], end[1] - start[1])
    v2 = (point[0] - start[0], point[1] - start[1])
    return bool(v1[0] * v2[1] - v1[1] * v2[0] >= zero)


def geometry(m: dict, g: dict, sphere: bool, c, out: dict) -> None:
    """``get_geometry_incremental_remap`` into ``out`` (the remap pool's
    ``remapEdge``, ``cellsOnEdgeRemap``, ``edgesOnEdgeRemap``,
    ``xVertexOnEdge``/``yVertexOnEdge``, ``minLengthEdgesOnVertex``)."""
    nc, ne, nv, ncs = int(m["nCells"]), int(m["nEdges"]), int(m["nVertices"]), int(m["nCellsSolve"])
    vd = int(m["vertexDegree"])
    neoc, eoc, coe, voe, eov = (m["nEdgesOnCell"], m["edgesOnCell"], m["cellsOnEdge"], m["verticesOnEdge"],
                                m["edgesOnVertex"])
    xV, yV, zV, xE, yE, zE, xC, yC, zC = (g[k] for k in ("xVertex", "yVertex", "zVertex", "xEdge", "yEdge", "zEdge",
                                                         "xCell", "yCell", "zCell"))
    remap_edge = out["remapEdge"]
    remap_edge[:] = 0
    for cell in range(ncs):
        for k in range(int(neoc[cell])):
            remap_edge[eoc[cell, k] - 1] = 1
    for e in range(ne):
        if remap_edge[e] == 1 and any(not 1 <= coe[e, i] <= nc for i in range(2)):
            remap_edge[e] = 0
    edges = [e for e in range(ne) if remap_edge[e] == 1]
    T = g.get("transGlobalToEdge")
    # cell 1 in each edge's left half-plane (point_in_half_plane called (V1, V2, C1), as written)
    for e in edges:
        vs = [voe[e, i] - 1 for i in range(2)]
        if sphere:
            ev = [_to_plane(T, e, xV[v] - xE[e], yV[v] - yE[e], zV[v] - zE[e]) for v in vs]
        else:
            ev = [(xV[v] - xE[e], yV[v] - yE[e]) for v in vs]
        cell = coe[e, 0]
        if 1 <= cell <= nc:
            i = cell - 1
            centre = (_to_plane(T, e, xC[i] - xE[e], yC[i] - yE[e], zC[i] - zE[e]) if sphere
                      else (xC[i] - xE[e], yC[i] - yE[e]))
            if not _point_in_half_plane(ev[0], ev[1], centre, c.zero):
                raise RemapError(f"cell 1 is not in the left half-plane: iEdge {e + 1} (v8.4.2 stops, MPAS_LOG_CRIT)")
    eoer, coer = out["edgesOnEdgeRemap"], out["cellsOnEdgeRemap"]
    eoer[...] = 0
    coer[...] = 0
    main_on_cell = None
    for e in edges:
        for i in range(2):
            coer[e, i] = coe[e, i]
        for i in range(2):
            cell = coe[e, i]
            if not 1 <= cell <= nc:
                continue
            n = int(neoc[cell - 1])
            for k in range(n):
                if eoc[cell - 1, k] == e + 1:
                    main_on_cell = k + 1
                    break
            if main_on_cell is None:
                raise RemapError(f"edge {e + 1} is not among its cell's edges (v8.4.2 would read stale indices)")
            back = main_on_cell - 1 if main_on_cell - 1 >= 1 else main_on_cell - 1 + n
            fwd = main_on_cell + 1 if main_on_cell + 1 <= n else main_on_cell + 1 - n
            if i == 0:
                eoer[e, 0], eoer[e, 1] = eoc[cell - 1, back - 1], eoc[cell - 1, fwd - 1]
            else:
                eoer[e, 2], eoer[e, 3] = eoc[cell - 1, fwd - 1], eoc[cell - 1, back - 1]
        if vd == 4:
            for iv in range(2):
                vertex = voe[e, iv] - 1
                for k in range(vd):
                    nb = eov[vertex, k]
                    if 1 <= nb <= ne:
                        if not any(nb == eoer[e, j] or nb == e + 1 for j in range(4)):
                            eoer[e, iv + 4] = nb
                            break
        if vd == 3:
            coer[e, 2:4] = nc + 1
            for iv in range(2):
                nb = eoer[e, iv]
                if not 1 <= nb <= ne:
                    nb = eoer[e, iv + 2]
                if not 1 <= nb <= ne:
                    raise RemapError(f"edge {e + 1}: neither side edge at vertex {iv + 1} exists (v8.4.2 reads "
                                     "outside cellsOnEdge)")
                for j in range(2):
                    cn = coe[nb - 1, j]
                    if 1 <= cn <= nc and cn != coer[e, 0] and cn != coer[e, 1]:
                        coer[e, iv + 2] = cn
        else:
            coer[e, 2:6] = nc + 1
            for iv in range(2):
                for side, other, slot in ((iv, 0, iv + 2), (iv + 2, 1, iv + 4)):
                    nb = eoer[e, side]
                    if 1 <= nb <= ne:
                        for j in range(2):
                            cn = coe[nb - 1, j]
                            if 1 <= cn <= nc and cn != coer[e, other]:
                                coer[e, slot] = cn
    n_side = 4 if vd == 3 else 6
    xvoe, yvoe = out["xVertexOnEdge"], out["yVertexOnEdge"]
    if sphere:
        all_e = np.arange(ne)
        v1, v2 = voe[:ne, 0] - 1, voe[:ne, 1] - 1
        xv, yv = _to_plane(T, all_e, xV[v2] - xV[v1], yV[v2] - yV[v1], zV[v2] - zV[v1])
        xvoe[:ne, 0], yvoe[:ne, 0] = -c.half * xv, -c.half * yv
        xvoe[:ne, 1], yvoe[:ne, 1] = c.half * xv, c.half * yv
    stale = {}
    for e in edges:
        if not sphere:
            for iv in range(2):
                v = voe[e, iv] - 1
                xvoe[e, iv], yvoe[e, iv] = xV[v] - xE[e], yV[v] - yE[e]
            count = 2
        for k in range(n_side):
            nb = eoer[e, k]
            if not 1 <= nb <= ne:
                continue
            for n in range(2):
                for mm in range(2):
                    if voe[nb - 1, mm] == voe[e, n]:
                        stale.update(n1=n, m1=mm, m2=1 - mm, vertex=voe[nb - 1, 1 - mm] - 1)
                        if not sphere:
                            count += 1
                        break
            if "n1" not in stale:
                raise RemapError(f"edge {e + 1}: a side edge shares no vertex with it (v8.4.2 reads stale indices)")
            if sphere:
                n1, m1, m2 = stale["n1"], stale["m1"], stale["m2"]
                xvoe[e, k + 2] = xvoe[e, n1] + (xvoe[nb - 1, m2] - xvoe[nb - 1, m1])
                yvoe[e, k + 2] = yvoe[e, n1] + (yvoe[nb - 1, m2] - yvoe[nb - 1, m1])
            else:
                v = stale["vertex"]
                xvoe[e, count - 1], yvoe[e, count - 1] = xV[v] - xE[e], yV[v] - yE[e]
    minlen = out["minLengthEdgesOnVertex"]
    minlen[:] = c.huge
    for k in range(vd):
        verts = np.flatnonzero((eov[:nv, k] >= 1) & (eov[:nv, k] <= ne))
        ed = eov[verts, k] - 1
        a, b = voe[ed, 0] - 1, voe[ed, 1] - 1
        w1, w2, w3 = xV[b] - xV[a], yV[b] - yV[a], zV[b] - zV[a]
        length = np.sqrt((w1 * w1 + w2 * w2) + w3 * w3)
        shorter = length < minlen[verts]
        minlen[verts[shorter]] = length[shorter]


def cell_averages(m: dict, xvoc, yvoc, c, out: dict) -> None:
    """``compute_geometric_cell_averages``: the 14 moments of cells 1..nCells."""
    nc = int(m["nCells"])
    neoc, eoc, dc, dv = m["nEdgesOnCell"][:nc], m["edgesOnCell"], m["dcEdge"], m["dvEdge"]
    real = c.real
    kmax = int(neoc.max(initial=0))
    frac = np.zeros((nc, max(kmax, 1)), real)
    total = np.zeros(nc, real)
    for k in range(kmax):
        cells = np.flatnonzero(k < neoc)
        e = eoc[cells, k] - 1
        frac[cells, k] = (c.quarter * dc[e]) * dv[e]
        total[cells] = total[cells] + frac[cells, k]
    for k in range(kmax):
        cells = np.flatnonzero(k < neoc)
        frac[cells, k] = frac[cells, k] / total[cells]
    avg = {name: np.zeros(nc, real) for name in MOMENTS}
    q1, q2, q3, q4 = c.q1, c.q2, c.q3, c.q4
    weights = (c.w1, c.w1, c.w1, c.w2, c.w2, c.w2)
    for k in range(kmax):
        cells = np.flatnonzero(k < neoc)
        kp1 = np.where(k + 1 >= neoc[cells], 0, k + 1)
        x1 = y1 = np.zeros(cells.size, real)
        x2, y2 = xvoc[cells, k], yvoc[cells, k]
        x3, y3 = xvoc[cells, kp1], yvoc[cells, kp1]
        coefficients = ((q1, q1, q2), (q1, q2, q1), (q2, q1, q1), (q3, q4, q4), (q4, q3, q4), (q4, q4, q3))
        qx = [(a * x1 + b * x2) + d * x3 for a, b, d in coefficients]
        qy = [(a * y1 + b * y2) + d * y3 for a, b, d in coefficients]
        sums = {name: np.zeros(cells.size, real) for name in MOMENTS}
        for n in range(6):
            w, x, y = weights[n], qx[n], qy[n]
            x2_, y2_ = x * x, y * y
            x3_, y3_ = x2_ * x, y2_ * y
            x4_, y4_ = x2_ * x2_, y2_ * y2_
            terms = {"x": w * x, "y": w * y, "xx": w * x2_, "xy": (w * x) * y, "yy": w * y2_, "xxx": w * x3_,
                     "xxy": (w * x2_) * y, "xyy": (w * x) * y2_, "yyy": w * y3_, "xxxx": w * x4_,
                     "xxxy": (w * x3_) * y, "xxyy": (w * x2_) * y2_, "xyyy": (w * x) * y3_, "yyyy": w * y4_}
            for name in MOMENTS:
                sums[name] = sums[name] + terms[name]
        for name in MOMENTS:
            avg[name][cells] = avg[name][cells] + frac[cells, k] * sums[name]
    for name in MOMENTS:
        out[f"{name}AvgCell"][:nc] = avg[name]


def initialize_vectors(m: dict, sphere: bool, c, out: dict) -> None:
    """``mpas_initialize_vectors`` (``mpas_rbf_interp_initialize``) on the
    mesh pool's own (unrotated) coordinates: ``localVerticalUnitVectors``,
    ``edgeNormalVectors`` and ``cellTangentPlane``."""
    nc, ne = int(m["nCells"]), int(m["nEdges"])
    real = c.real
    xC, yC, zC, xE, yE, zE = (m[k] for k in ("xCell", "yCell", "zCell", "xEdge", "yEdge", "zEdge"))
    coe, eoc = m["cellsOnEdge"], m["edgesOnCell"]
    up, normal, plane = out["localVerticalUnitVectors"], out["edgeNormalVectors"], out["cellTangentPlane"]
    normal[...] = 0
    up[...] = 0
    if sphere:
        up[:nc, 0], up[:nc, 1], up[:nc, 2] = _unit_vec_in_r3(xC[:nc], yC[:nc], zC[:nc])
    else:
        up[:nc] = np.array([0.0, 0.0, 1.0], real)
    c1, c2 = coe[:ne, 0], coe[:ne, 1]
    if ((c1 < 1) | (c1 > nc + 1) | (c2 < 1) | (c2 > nc + 1) | ((c1 == nc + 1) & (c2 == nc + 1))).any():
        raise RemapError("mpas_initialize_vectors: an edge's cells are outside 1..nCells+1 (MPAS reads outside "
                         "xCell)")
    a, b = c1 - 1, c2 - 1
    first, second = c1 == nc + 1, c2 == nc + 1
    e = np.arange(ne)
    nx = np.where(first, xC[b] - xE[e], np.where(second, xE[e] - xC[a], xC[b] - xC[a]))
    ny = np.where(first, yC[b] - yE[e], np.where(second, yE[e] - yC[a], yC[b] - yC[a]))
    nz = np.where(first, zC[b] - zE[e], np.where(second, zE[e] - zC[a], zC[b] - zC[a]))
    normal[:ne, 0], normal[:ne, 1], normal[:ne, 2] = _unit_vec_in_r3(nx, ny, nz)
    first_edge = eoc[:nc, 0] - 1
    r = (up[:nc, 0], up[:nc, 1], up[:nc, 2])
    n = (normal[first_edge, 0], normal[first_edge, 1], normal[first_edge, 2])
    dot = _sum3(n[0], r[0], n[1], r[1], n[2], r[2], c.zero)
    xhat = _unit_vec_in_r3(n[0] - dot * r[0], n[1] - dot * r[1], n[2] - dot * r[2])
    yhat = _unit_vec_in_r3(*_cross(r, xhat))
    for i in range(3):
        plane[:nc, 0, i] = xhat[i]
        plane[:nc, 1, i] = yhat[i]


def _elimination(A, n: int, c):
    """``elgs`` (Tao Pang's partial-pivoting Gaussian elimination) on a
    stack of matrices ``A[cell, row, col]`` (rows by Fortran's first index),
    in place; returns the pivot order ``INDX`` (0-based)."""
    cells = A.shape[0]
    rows = np.arange(cells)
    indx = np.tile(np.arange(n), (cells, 1))
    scale = np.zeros((cells, n), A.dtype)
    for j in range(n):
        scale = np.maximum(scale, np.abs(A[:, :, j]))   # MAX(C1, ABS(A(I,J))): no NaN, so the order is moot
    for j in range(n - 1):
        best = np.full(cells, c.zero, A.dtype)
        k = np.full(cells, -1)
        for i in range(j, n):
            p = np.abs(A[rows, indx[:, i], j]) / scale[rows, indx[:, i]]
            better = p > best
            best = np.where(better, p, best)
            k = np.where(better, i, k)
        if (k < 0).any():
            raise RemapError("elgs: a singular RBF matrix (no pivot; v8.4.2 would read an unset index)")
        held = indx[:, j].copy()
        indx[:, j] = indx[rows, k]
        indx[rows, k] = held
        for i in range(j + 1, n):
            pj = A[rows, indx[:, i], j] / A[rows, indx[:, j], j]
            A[rows, indx[:, i], j] = pj
            for kk in range(j + 1, n):
                A[rows, indx[:, i], kk] = A[rows, indx[:, i], kk] - pj * A[rows, indx[:, j], kk]
    return indx


def _solve(A, indx, B, n: int):
    """``mpas_legs`` after ``elgs``: the forward and back substitution."""
    cells = A.shape[0]
    rows = np.arange(cells)
    B = B.copy()
    for i in range(n - 1):
        for j in range(i + 1, n):
            B[rows, indx[:, j]] = B[rows, indx[:, j]] - A[rows, indx[:, j], i] * B[rows, indx[:, i]]
    X = np.zeros((cells, n), A.dtype)
    X[:, n - 1] = B[rows, indx[:, n - 1]] / A[rows, indx[:, n - 1], n - 1]
    for i in range(n - 2, -1, -1):
        x = B[rows, indx[:, i]]
        for j in range(i + 1, n):
            x = x - A[rows, indx[:, i], j] * X[:, j]
        X[:, i] = x / A[rows, indx[:, i], i]
    return X


def init_reconstruct(m: dict, c, vectors: dict, coeffs: np.ndarray) -> None:
    """``mpas_init_reconstruct`` (owned cells; ``coeffs_reconstruct`` zeroed
    first, so the halo keeps zeros): per cell the inverse-multiquadric RBF
    fit of a constant vector in the cell's tangent plane from the normal
    components at its edges (``mpas_rbf_interp_func_3D_plane_vec_const_dir_comp_coeffs``)."""
    ncs = int(m["nCellsSolve"])
    real = c.real
    zero = c.zero
    xC, yC, zC, xE, yE, zE = (m[k] for k in ("xCell", "yCell", "zCell", "xEdge", "yEdge", "zEdge"))
    neoc, eoc = m["nEdgesOnCell"][:ncs], m["edgesOnCell"]
    normal, plane = vectors["edgeNormalVectors"], vectors["cellTangentPlane"]
    coeffs[...] = 0
    for count in sorted(set(int(v) for v in neoc)):
        cells = np.flatnonzero(neoc == count)
        e = eoc[cells, :count] - 1                           # (cells, count)
        centre = (xC[cells], yC[cells], zC[cells])
        loc = (xE[e], yE[e], zE[e])
        unit = (normal[e, 0], normal[e, 1], normal[e, 2])
        alpha = np.full(cells.size, zero, real)
        for i in range(count):
            d = [centre[k] - loc[k][:, i] for k in range(3)]
            r = np.sqrt(((zero + d[0] * d[0]) + d[1] * d[1]) + d[2] * d[2])
            alpha = alpha + r
        alpha = alpha / real(count)
        basis = [[plane[cells, a, k] for k in range(3)] for a in range(2)]   # planeBasisVectors(a, k)

        def project(v, a):
            return _sum3(v[0], basis[a][0], v[1], basis[a][1], v[2], basis[a][2], zero)
        ps = [[project([loc[k][:, i] for k in range(3)], a) for a in range(2)] for i in range(count)]
        pu = [[project([unit[k][:, i] for k in range(3)], a) for a in range(2)] for i in range(count)]
        pd = [project(list(centre), a) for a in range(2)]
        size = count + 2
        A = np.zeros((cells.size, size, size), real)
        rhs = np.zeros((cells.size, size, 2), real)
        alpha2 = alpha * alpha
        one = c.one

        def rbf(r2):
            return one / np.sqrt(one + r2)
        for j in range(count):
            for i in range(j, count):
                dx, dy = ps[i][0] - ps[j][0], ps[i][1] - ps[j][1]
                r2 = ((zero + dx * dx) + dy * dy) / alpha2
                dotp = (zero + pu[i][0] * pu[j][0]) + pu[i][1] * pu[j][1]
                A[:, i, j] = rbf(r2) * dotp
                A[:, j, i] = A[:, i, j]
        for j in range(count):
            dx, dy = pd[0] - ps[j][0], pd[1] - ps[j][1]
            r2 = ((zero + dx * dx) + dy * dy) / alpha2
            value = rbf(r2)
            rhs[:, j, 0] = value * pu[j][0]
            rhs[:, j, 1] = value * pu[j][1]
        for i in range(count):
            for a in range(2):
                A[:, i, count + a] = pu[i][a]
                A[:, count + a, i] = A[:, i, count + a]
        rhs[:, count, 0] = one
        rhs[:, count + 1, 1] = one
        indx = _elimination(A, size, c)
        x1 = _solve(A, indx, rhs[:, :, 0], size)
        x2 = _solve(A, indx, rhs[:, :, 1], size)
        for k in range(3):
            coeffs[cells, :count, k] = basis[0][k][:, None] * x1[:, :count] + basis[1][k][:, None] * x2[:, :count]


def initialize(*, mesh: dict, remap: dict, rotated: dict | None, configs: dict, real) -> list[dict]:
    """``seaice_init_advection_incremental_remap``'s geometry, on host arrays
    (numpy, C-order, MPAS's names), in place: the remap pool's frames, vertex
    coordinates, remap neighbours and moments; the rotated vectors; the mesh
    pool's RBF vectors and ``coeffs_reconstruct``. Returns the warnings
    v8.4.2 logs (too many halo layers)."""
    c = _Constants(real)
    if configs.get("mesh:is_periodic", configs.get("is_periodic", False)):
        raise ConfigurationRefusal(knob="is_periodic", value=True,
                                   reason="periodic planar meshes (mpas_fix_periodicity in the RBF set-up) are not "
                                          "ported", declaration="is_periodic = .false.")
    sphere = bool(configs.get("mesh:on_a_sphere", configs.get("on_a_sphere", False)))
    rotate = sphere and bool(configs.get("config_rotate_cartesian_grid", True))
    warnings = check_halo_layer_number(configs, int(mesh["vertexDegree"]))
    nc, ne, nv = int(mesh["nCells"]), int(mesh["nEdges"]), int(mesh["nVertices"])
    g = {}
    for kind, n in (("Cell", nc), ("Vertex", nv), ("Edge", ne)):
        x, y, z = mesh[f"x{kind}"], mesh[f"y{kind}"], mesh[f"z{kind}"]
        if rotate:
            rx, ry, rz = rotated[f"x{kind}Rotate"], rotated[f"y{kind}Rotate"], rotated[f"z{kind}Rotate"]
            rx[:n], ry[:n], rz[:n] = -z[:n], y[:n], x[:n]
            x, y, z = rx, ry, rz
        g[f"x{kind}"], g[f"y{kind}"], g[f"z{kind}"] = x, y, z
    if sphere:
        for (l2g, g2l), kind, n in zip(FRAMES, ("Cell", "Vertex", "Edge"), (nc, nv, ne)):
            a, b = define_frames(g[f"x{kind}"], g[f"y{kind}"], g[f"z{kind}"], n, c)
            remap[l2g][:n], remap[g2l][:n] = a, b
            g[g2l] = remap[g2l]
    view = dict(mesh, xVertexOnCell=remap["xVertexOnCell"], yVertexOnCell=remap["yVertexOnCell"])
    xvoc, yvoc = vertex_on_cell_coordinates(view, g, sphere, c)
    remap["xVertexOnCell"][...], remap["yVertexOnCell"][...] = xvoc, yvoc
    geometry(mesh, g, sphere, c, remap)
    cell_averages(mesh, xvoc, yvoc, c, remap)
    initialize_vectors(mesh, sphere, c, mesh)
    init_reconstruct(mesh, c, mesh, mesh["coeffs_reconstruct"])
    ncs = int(mesh["nCellsSolve"])
    area = mesh["areaCell"][:ncs]
    if not (area > c.zero).all():
        bad = int(np.flatnonzero(~(area > c.zero))[0]) + 1
        raise RemapError(f"IR: Bad cell area, iCell {bad} (v8.4.2 stops, MPAS_LOG_CRIT)")
    return warnings


#: the mesh-pool fields initialize writes (with coeffs_reconstruct)
MESH_OUTPUTS = ("localVerticalUnitVectors", "edgeNormalVectors", "cellTangentPlane", "coeffs_reconstruct")
#: the remap-pool fields initialize writes
REMAP_OUTPUTS = (*(name for pair in FRAMES for name in pair), "xVertexOnCell", "yVertexOnCell", "xVertexOnEdge",
                 "yVertexOnEdge", "remapEdge", "cellsOnEdgeRemap", "edgesOnEdgeRemap", "minLengthEdgesOnVertex",
                 *(f"{name}AvgCell" for name in MOMENTS))


def initialize_pools(pools: dict, configs: dict) -> list[dict]:
    """:func:`initialize` on device pools (cupy arrays, or numpy): copied to
    the host, computed, copied back. An output the pools lack (the RBF
    vectors, ``coeffs_reconstruct``, the rotated coordinates) is allocated."""
    def host(a):
        if a is None or np.isscalar(a) or isinstance(a, (int, float, np.ndarray)):
            return a
        return a.get()
    mesh = {k: host(v) for k, v in pools["mesh"].items()}
    remap = {k: host(v) for k, v in pools["incremental_remap"].items()}
    pools.setdefault("rotated_mesh", {})
    rotated = {k: host(v) for k, v in pools["rotated_mesh"].items()}
    real = remap["xVertexOnCell"].dtype.type
    nc, ne, nv = int(mesh["nCells"]), int(mesh["nEdges"]), int(mesh["nVertices"])
    missing = {"localVerticalUnitVectors": (nc + 1, 3), "edgeNormalVectors": (ne + 1, 3),
               "cellTangentPlane": (nc + 1, 2, 3), "coeffs_reconstruct": (nc + 1, int(mesh["maxEdges"]), 3)}
    for name, shape in missing.items():
        if mesh.get(name) is None:
            mesh[name] = np.zeros(shape, real)
    for kind, n in (("Cell", nc), ("Vertex", nv), ("Edge", ne)):
        for axis in "xyz":
            if rotated.get(f"{axis}{kind}Rotate") is None:
                rotated[f"{axis}{kind}Rotate"] = np.zeros(n + 1, real)
    warnings = initialize(mesh=mesh, remap=remap, rotated=rotated, configs=configs, real=real)
    for pool, values, names in (("mesh", mesh, MESH_OUTPUTS), ("incremental_remap", remap, REMAP_OUTPUTS),
                                ("rotated_mesh", rotated, tuple(rotated))):
        for name in names:
            if name not in values or values[name] is None:
                continue
            target = pools[pool].get(name)
            if isinstance(target, np.ndarray):
                target[...] = values[name]
            elif target is None:
                pools[pool][name] = _like(pools["mesh"]["xCell"], values[name])
            else:
                target.set(np.ascontiguousarray(values[name]))
    return warnings


def _like(template, values: np.ndarray):
    """``values`` where ``template`` lives (numpy, or the device)."""
    if isinstance(template, np.ndarray):
        return np.ascontiguousarray(values)
    import cupy as cp
    return cp.asarray(np.ascontiguousarray(values))
