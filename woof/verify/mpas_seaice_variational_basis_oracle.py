"""Cases and the stream layout for MPAS-Seaice's variational basis set-up.

The reference is ``tools/mpas_seaice_oracle/run_variational_basis.F90``:
``seaice_init_velocity_solver_variational`` with the Wachspress, PWL,
variational_shared and numerics modules compiled verbatim (build.sh). The
port is :mod:`woof.hex.mpas_seaice.variational_basis_v842` (on the host).

The cases cover:

- every basis (``wachspress``, ``pwl``) and every Wachspress integration rule
  v8.4.2 has (trapezoidal at orders 1, 2 and 5; Dunavant 1-10 and 12, the odd
  ones with their WARN; Fekete 1-6, 8 and 9);
- planar hexagon and quadrilateral blocks (:mod:`.mpas_seaice_planar_mesh`,
  culled land, so boundary vertices whose missing cell is nCells+1) and the
  same on the sphere (:mod:`.mpas_seaice_sphere_mesh`: centred on the north
  and south poles and at mid-latitude, uniform and variable resolution);
- blocks of independent convex polygons of 3 to 9 sides (planar, and on the
  sphere at random places, one cell centred where the rotated ``z`` is 0), so
  ``nEdgesOnCell`` varies cell to cell as at a real mesh's pentagons and
  heptagons;
- ``config_rotate_cartesian_grid`` on and off, ``config_include_metric_terms``
  on and off, and the metric terms on a planar mesh (v8.4.2's quirk: ``NaN``
  where ``sphere_radius`` is 0, ``tan(asin(x/R))/R`` of the rotated ``z = x``
  otherwise).

Every output array is noise on input, so a word v8.4.2 does not write and the
port does show. Only correctly rounded operations build the meshes, so the
input bytes are the same on every host (``inputs.sha256``).
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from woof.verify import mpas_ocean_stream as _stream
from woof.verify.mpas_ocean_tend_oracle import PRECISIONS
from woof.verify.mpas_seaice_planar_mesh import planar_mesh
from woof.verify.mpas_seaice_sphere_mesh import RADIUS, sphere_mesh

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mpas_seaice_variational_basis"
REAL, INT, DIM = _stream.REAL, _stream.INT, _stream.DIM
OUTPUTS = ("basisGradientU", "basisGradientV", "basisIntegralsU", "basisIntegralsV", "basisIntegralsMetric",
           "tanLatVertexRotatedOverRadius", "cellVerticesAtVertex")
FIELDS = (
    ("mesh", "nCells", DIM, ()), ("mesh", "nVertices", DIM, ()), ("mesh", "nEdges", DIM, ()),
    ("mesh", "maxEdges", DIM, ()), ("mesh", "vertexDegree", DIM, ()),
    ("mesh", "nEdgesOnCell", INT, ("C1",)), ("mesh", "verticesOnCell", INT, ("ME", "C1")),
    ("mesh", "edgesOnCell", INT, ("ME", "C1")), ("mesh", "cellsOnVertex", INT, ("VD", "V1")),
    *(("mesh", name, REAL, ("V1",)) for name in ("xVertex", "yVertex", "zVertex")),
    *(("mesh", name, REAL, ("C1",)) for name in ("xCell", "yCell", "zCell", "areaCell")),
    ("mesh", "dvEdge", REAL, ("E1",)),
    *(("velocity_variational", name, REAL, ("ME", "ME", "C1")) for name in OUTPUTS[:5]),
    ("velocity_variational", "tanLatVertexRotatedOverRadius", REAL, ("V1",)),
    ("velocity_variational", "cellVerticesAtVertex", INT, ("VD", "V1")),
)
MESH_ARRAYS = ("nEdgesOnCell", "verticesOnCell", "edgesOnCell", "cellsOnVertex", "xVertex", "yVertex", "zVertex",
               "xCell", "yCell", "zCell", "areaCell", "dvEdge")


def _norm(v) -> float:
    return math.sqrt(float(v[0]) * float(v[0]) + float(v[1]) * float(v[1]) + float(v[2]) * float(v[2]))


def _cross(a, b) -> np.ndarray:
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def polygon_mesh(rs: np.random.RandomState, *, n_cells: int = 24, sphere: bool = False, radius: float = RADIUS,
                 size: float = 2.0e4) -> dict:
    """Independent convex polygons (3 to 9 vertices, counter-clockwise) with
    their own vertices and edges: each vertex's first cell is its polygon, the
    rest nCells+1. Points on a circle are rational ((1-t^2, 2t)/(1+t^2)), so
    only correctly rounded operations make the mesh."""
    me, vd = 9, 3
    sides = [3 + (i % 7) for i in range(n_cells)]
    rs.shuffle(sides)
    nc, nv = n_cells, sum(sides)
    ne = nv
    n_edges_on_cell = np.zeros(nc + 1, np.int32)
    voc = np.zeros((nc + 1, me), np.int32)
    eoc = np.zeros((nc + 1, me), np.int32)
    cov = np.full((nv + 1, vd), nc + 1, np.int32)
    xv, yv, zv = np.zeros(nv + 1), np.zeros(nv + 1), np.zeros(nv + 1)
    xc, yc, zc = np.zeros(nc + 1), np.zeros(nc + 1), np.zeros(nc + 1)
    area, dv = np.zeros(nc + 1), np.zeros(ne + 1)
    v = 0
    for c, n in enumerate(sides):
        # n points on the unit circle, counter-clockwise from t in (-1, 1] then the far side
        t = np.sort(rs.uniform(-2.4, 2.4, n))
        while np.min(np.diff(t)) < 0.15:
            t = np.sort(rs.uniform(-2.4, 2.4, n))
        px = (1.0 - t * t) / (1.0 + t * t)
        py = (2.0 * t) / (1.0 + t * t)
        sx, sy, shear = size * rs.uniform(0.7, 1.3), size * rs.uniform(0.7, 1.3), rs.uniform(-0.3, 0.3)
        local = [(sx * px[k] + shear * sx * py[k], sy * py[k]) for k in range(n)]
        if sphere:
            # the cell's centre on the sphere and two tangent directions there
            if c == 0:
                centre = np.array([0.0, 0.6, 0.8])      # rotated z = x = 0
            else:
                centre = rs.uniform(-1.0, 1.0, 3)
                while _norm(centre) < 0.3:
                    centre = rs.uniform(-1.0, 1.0, 3)
            cn = centre / _norm(centre)
            helper = np.array([0.0, 0.0, 1.0]) if abs(cn[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            e1 = _cross(helper, cn)
            e1 = e1 / _norm(e1)
            e2 = _cross(cn, e1)
            pts = []
            for lx, ly in local:
                p = cn * radius + e1 * lx + e2 * ly
                pts.append(p * (radius / _norm(p)))
            xc[c], yc[c], zc[c] = cn * radius
        else:
            ox, oy = rs.uniform(-1.0e6, 1.0e6, 2)
            pts = [np.array([ox + lx, oy + ly, 0.0]) for lx, ly in local]
            xc[c], yc[c] = ox + rs.uniform(-0.05, 0.05) * size, oy + rs.uniform(-0.05, 0.05) * size
        n_edges_on_cell[c] = n
        doubled = 0.0
        for k in range(n):
            xv[v + k], yv[v + k], zv[v + k] = pts[k]
            voc[c, k] = v + k + 1
            eoc[c, k] = v + k + 1
            cov[v + k, rs.randint(vd)] = c + 1
            a, b = pts[k], pts[(k + 1) % n]
            d = b - a
            dv[v + k] = math.sqrt(float(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]))
            la, lb = local[k], local[(k + 1) % n]
            doubled += la[0] * lb[1] - lb[0] * la[1]
        area[c] = 0.5 * doubled
        v += n
    return {"kind": "polygons", "nCells": nc, "nEdges": ne, "nVertices": nv, "maxEdges": me, "vertexDegree": vd,
            "nEdgesOnCell": n_edges_on_cell, "verticesOnCell": voc, "edgesOnCell": eoc, "cellsOnVertex": cov,
            "xCell": xc, "yCell": yc, "zCell": zc, "xVertex": xv, "yVertex": yv, "zVertex": zv,
            "areaCell": area, "dvEdge": dv}


#: mesh name -> (generator, sphere, sphere_radius)
MESHES = {
    "planar_hex": (lambda rs: planar_mesh("hex", rs, jitter=0.12), False, 0.0),
    "planar_quad": (lambda rs: planar_mesh("quad", rs, jitter=0.12), False, 0.0),
    "planar_polygons": (lambda rs: polygon_mesh(rs), False, 0.0),
    "sphere_hex_north": (lambda rs: sphere_mesh("hex", rs, centre="north"), True, RADIUS),
    "sphere_quad_south": (lambda rs: sphere_mesh("quad", rs, centre="south"), True, RADIUS),
    "sphere_hex_midlat_varres": (lambda rs: sphere_mesh("hex", rs, centre="midlat", varres=True), True, RADIUS),
    "sphere_quad_north_varres": (lambda rs: sphere_mesh("quad", rs, centre="north", varres=True), True, RADIUS),
    "sphere_polygons": (lambda rs: polygon_mesh(rs, sphere=True), True, RADIUS),
}

_RULES = ([("trapezoidal", n) for n in (1, 2, 5)] + [("dunavant", n) for n in (*range(1, 11), 12)]
          + [("fekete", n) for n in (1, 2, 3, 4, 5, 6, 8, 9)])


def _rows() -> list[tuple]:
    """(name, mesh, seed, basis, integration type, order, rotate, metric, sphere_radius override)."""
    rows = []
    for mesh in MESHES:
        rows.append((f"{mesh}_wachspress", mesh, 11, "wachspress", "dunavant", 8, True, True, None))
        rows.append((f"{mesh}_pwl", mesh, 11, "pwl", "dunavant", 8, True, True, None))
    for kind, order in _RULES:
        rows.append((f"planar_polygons_{kind}{order}", "planar_polygons", 12, "wachspress", kind, order, True,
                     False, None))
        rows.append((f"sphere_quad_south_{kind}{order}", "sphere_quad_south", 12, "wachspress", kind, order,
                     True, False, None))
    for basis in ("wachspress", "pwl"):
        rows.append((f"sphere_polygons_{basis}_unrotated", "sphere_polygons", 13, basis, "dunavant", 8, False,
                     True, None))
        rows.append((f"sphere_hex_north_{basis}_no_metric", "sphere_hex_north", 13, basis, "dunavant", 8, True,
                     False, None))
    # the metric terms on a plane with a sphere_radius: asin of |x/R| below and above 1
    rows.append(("planar_hex_metric_radius", "planar_hex", 14, "pwl", "dunavant", 8, True, True, 1.5e5))
    rows.append(("planar_polygons_metric_radius_unrotated", "planar_polygons", 14, "wachspress", "dunavant", 8,
                 False, True, 1.5e6))
    return rows


CASES = tuple(_rows())


def make_case(name, mesh_name, seed, basis, kind, order, rotate, metric, radius=None, *, precision="r4") -> dict:
    real = PRECISIONS[precision]
    generate, sphere, default_radius = MESHES[mesh_name]
    rs = np.random.RandomState(seed)
    m = generate(rs)
    noise = np.random.RandomState(seed + 1000)
    dims = {"nCells": m["nCells"], "nEdges": m["nEdges"], "nVertices": m["nVertices"], "maxEdges": m["maxEdges"],
            "vertexDegree": m["vertexDegree"], "nVertLevels": 1}
    config = {"config_variational_basis": basis, "config_wachspress_integration_type": kind,
              "config_wachspress_integration_order": int(order), "config_rotate_cartesian_grid": bool(rotate),
              "config_include_metric_terms": bool(metric), "mesh:on_a_sphere": bool(sphere),
              "mesh:sphere_radius": float(default_radius if radius is None else radius)}
    case = {"name": name, "precision": precision, "dims": dims, "config": config, "dt": 0.0, "mesh_name": mesh_name}
    pools: dict[str, dict] = {"mesh": {}, "velocity_variational": {}}
    for pool, field, k, fshape in FIELDS:
        if k == DIM:
            value = m[field]
        elif pool == "mesh":
            value = np.asarray(m[field]).astype(real if k == REAL else np.int32)
        elif k == INT:
            value = noise.randint(-9, 99, _stream.shape(case, fshape)).astype(np.int32)
        else:
            value = noise.uniform(-1.0, 1.0, _stream.shape(case, fshape)).astype(real)
        pools[pool][field] = value
    case["pools"] = pools
    return case


def cases(precision: str = "r4") -> list[dict]:
    return [make_case(*row, precision=precision) for row in CASES]


def input_bytes(case: dict) -> bytes:
    return _stream.input_bytes(FIELDS, case)


def input_digest(case: dict) -> str:
    return hashlib.sha256(input_bytes(case)).hexdigest()


def read_outputs(raw: bytes, case: dict) -> dict[str, np.ndarray]:
    out = _stream.read_outputs(FIELDS, raw, case)
    return {k: v for k, v in out.items() if k.startswith("velocity_variational/")}


word_mismatches = _stream.word_mismatches


def load_fixture(precision: str) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, str]]:
    """The oracle's words per case and its log (WARN lines) per case."""
    out: dict[str, dict[str, np.ndarray]] = {}
    logs: dict[str, str] = {}
    with np.load(FIXTURE_DIR / f"oracle-{precision}.npz") as data:
        for key in data.files:
            case, field = key.split("/", 1)
            if field == "log":
                logs[case] = data[key].tobytes().decode("utf-8")
            else:
                out.setdefault(case, {})[field] = data[key]
    return out, logs


def run_port(case: dict, *, metric_libm: str | None = None, corrections=()) -> tuple[dict[str, np.ndarray], list]:
    """The port on ``case``, as run_variational_basis.F90 drives the Fortran."""
    from woof.hex.mpas_seaice import variational_basis_v842 as vb

    real = PRECISIONS[case["precision"]]
    mesh = dict(case["pools"]["mesh"])
    out = {name: np.array(case["pools"]["velocity_variational"][name], copy=True) for name in OUTPUTS}
    out, warnings = vb.initialize(mesh=mesh, configs=case["config"], real=real, out=out, metric_libm=metric_libm,
                                  corrections=corrections)
    return {f"velocity_variational/{name}": out[name] for name in OUTPUTS}, warnings
