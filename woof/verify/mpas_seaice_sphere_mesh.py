"""A spherical MPAS mesh block (hexagons or quadrilaterals), for the sea-ice remap oracles.

:func:`sphere_mesh` takes :func:`woof.verify.mpas_seaice_planar_mesh.planar_mesh`'s
block (the same topology, numbering, land and halo layers) and lays it on a
sphere of MPAS-Seaice's radius: optionally stretched about its middle first
(``varres``: cells grow away from the middle, several times larger at the
block's rim), then carried onto the sphere by the inverse gnomonic projection
about the North Pole, and turned to where ``centre`` names (the North Pole, the
South Pole, or 53 degrees north). Cell centres and edge midpoints are the
normalised means of their vertices, ``dvEdge`` and ``dcEdge`` chords, and
``areaCell`` the area of the cell's fan of flat triangles, as a spherical
mesh's magnitudes (the remap reads them as given).

Every number comes from correctly rounded operations (``+ - * /`` and
``sqrt``; no libm trigonometry, no BLAS), so the float64 words are the same on
every host. The planar coordinates the block was built from stay under
``plane`` (the case generator draws the drift and the ice regions on them).
"""
from __future__ import annotations

import math

import numpy as np

from woof.verify.mpas_seaice_planar_mesh import ordered_sum, planar_mesh

#: MPAS-Seaice's sphere radius (m)
RADIUS = 6371229.0
#: where the block's middle goes: a rotation of the North-Pole patch, as rows
#: of a matrix with correctly rounded entries (53 degrees north: cos = 0.8)
CENTRES = {
    "north": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "south": ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
    "midlat": ((1.0, 0.0, 0.0), (0.0, 0.8, -0.6), (0.0, 0.6, 0.8)),
}


def _norm(v) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _on_sphere(v, radius: float):
    n = _norm(v)
    return (radius * (v[0] / n), radius * (v[1] / n), radius * (v[2] / n))


def _chord(a, b) -> float:
    return _norm((b[0] - a[0], b[1] - a[1], b[2] - a[2]))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def sphere_mesh(kind: str, rs: np.random.RandomState, *, owned: int = 3, centre: str = "north",
                varres: bool = False, radius: float = RADIUS, stretch_length: float = 5.0e4, **planar) -> dict:
    """The planar block on the sphere (see the module docstring); ``planar``
    goes to :func:`planar_mesh`."""
    if centre not in CENTRES:
        raise ValueError(f"centre is one of {sorted(CENTRES)}, not {centre!r}")
    m = planar_mesh(kind, rs, owned=owned, **planar)
    nc, ne, nv = m["nCells"], m["nEdges"], m["nVertices"]
    xs, ys = m["xVertex"].copy(), m["yVertex"].copy()
    if varres:
        # stretch about the vertices' mean: p -> c + (p - c) * (1 + |p - c|^2 / L^2)
        cx, cy = ordered_sum(xs[:nv].tolist()) / nv, ordered_sum(ys[:nv].tolist()) / nv
        for i in range(nv):
            dx, dy = float(xs[i]) - cx, float(ys[i]) - cy
            f = 1.0 + (dx * dx + dy * dy) / (stretch_length * stretch_length)
            xs[i], ys[i] = cx + dx * f, cy + dy * f
    rot = CENTRES[centre]
    vertex = []
    for i in range(nv):
        # the inverse gnomonic projection about the North Pole
        p = _on_sphere((float(xs[i]), float(ys[i]), radius), radius)
        vertex.append(tuple(rot[r][0] * p[0] + rot[r][1] * p[1] + rot[r][2] * p[2] for r in range(3)))
    out = dict(m)
    out["plane"] = {"xVertex": xs, "yVertex": ys}
    xv, yv, zv = np.zeros(nv + 1), np.zeros(nv + 1), np.zeros(nv + 1)
    for i, p in enumerate(vertex):
        xv[i], yv[i], zv[i] = p
    cell = []
    xc, yc, zc, area = np.zeros(nc + 1), np.zeros(nc + 1), np.zeros(nc + 1), np.zeros(nc + 1)
    pxc, pyc = np.zeros(nc + 1), np.zeros(nc + 1)
    for c in range(nc):
        n = int(m["nEdgesOnCell"][c])
        ring = [vertex[int(m["verticesOnCell"][c, j]) - 1] for j in range(n)]
        mean = tuple(ordered_sum(p[t] for p in ring) / n for t in range(3))
        p = _on_sphere(mean, radius)
        cell.append(p)
        xc[c], yc[c], zc[c] = p
        total = 0.0
        for j in range(n):
            a, b = ring[j], ring[(j + 1) % n]
            total += 0.5 * _norm(_cross((a[0] - p[0], a[1] - p[1], a[2] - p[2]),
                                        (b[0] - p[0], b[1] - p[1], b[2] - p[2])))
        area[c] = total
        idx = [int(m["verticesOnCell"][c, j]) - 1 for j in range(n)]
        pxc[c] = ordered_sum(float(xs[i]) for i in idx) / n
        pyc[c] = ordered_sum(float(ys[i]) for i in idx) / n
    out["plane"].update({"xCell": pxc, "yCell": pyc})
    xe, ye, ze, dv, dc = (np.zeros(ne + 1) for _ in range(5))
    for e in range(ne):
        v1, v2 = (int(k) - 1 for k in m["verticesOnEdge"][e])
        a, b = vertex[v1], vertex[v2]
        mid = _on_sphere(tuple(0.5 * (a[t] + b[t]) for t in range(3)), radius)
        xe[e], ye[e], ze[e] = mid
        dv[e] = _chord(a, b)
        c1, c2 = (int(k) - 1 for k in m["cellsOnEdge"][e])
        dc[e] = _chord(cell[c1], cell[c2]) if c2 < nc else 2.0 * _chord(mid, cell[c1])
    out.update({"xVertex": xv, "yVertex": yv, "zVertex": zv, "xCell": xc, "yCell": yc, "zCell": zc,
                "xEdge": xe, "yEdge": ye, "zEdge": ze, "dvEdge": dv, "dcEdge": dc, "areaCell": area,
                "radius": radius, "centre": centre, "varres": varres})
    return out


def reconstruction_coefficients(m: dict) -> np.ndarray:
    """``coeffs_reconstruct`` on the sphere: per cell the least-squares map
    from the edges' normal components (cellsOnEdge(1) towards cellsOnEdge(2),
    tangent to the sphere at the edge) to a vector in the cell's tangent plane,
    in global x/y/z (MPAS's RBF reconstruction is an input to the remap; this
    one is in closed form, correctly rounded)."""
    nc, me = m["nCells"], m["maxEdges"]
    coeffs = np.zeros((nc + 1, me, 3))
    for c in range(nc):
        n = int(m["nEdgesOnCell"][c])
        centre = (float(m["xCell"][c]), float(m["yCell"][c]), float(m["zCell"][c]))
        up = _on_sphere(centre, 1.0)
        v0 = int(m["verticesOnCell"][c, 0]) - 1
        d = (float(m["xVertex"][v0]) - centre[0], float(m["yVertex"][v0]) - centre[1],
             float(m["zVertex"][v0]) - centre[2])
        e1 = _on_sphere(_cross(_cross(up, d), up), 1.0)   # d's tangent part, unit
        e2 = _cross(up, e1)
        normals = []
        for i in range(n):
            e = int(m["edgesOnCell"][c, i]) - 1
            v1, v2 = (int(k) - 1 for k in m["verticesOnEdge"][e])
            t = (float(m["xVertex"][v2]) - float(m["xVertex"][v1]), float(m["yVertex"][v2]) - float(m["yVertex"][v1]),
                 float(m["zVertex"][v2]) - float(m["zVertex"][v1]))
            r = (float(m["xEdge"][e]), float(m["yEdge"][e]), float(m["zEdge"][e]))
            nrm = _on_sphere(_cross(t, r), 1.0)   # to the right of V1 -> V2 seen from outside
            normals.append((nrm[0] * e1[0] + nrm[1] * e1[1] + nrm[2] * e1[2],
                            nrm[0] * e2[0] + nrm[1] * e2[1] + nrm[2] * e2[2]))
        sxx = sxy = syy = 0.0
        for nx, ny in normals:
            sxx += nx * nx
            sxy += nx * ny
            syy += ny * ny
        det = sxx * syy - sxy * sxy
        for i, (nx, ny) in enumerate(normals):
            a = (syy * nx - sxy * ny) / det
            b = (sxx * ny - sxy * nx) / det
            for t in range(3):
                coeffs[c, i, t] = a * e1[t] + b * e2[t]
    return coeffs
