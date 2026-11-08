"""Cases and the stream layout for the MPAS-Seaice EVP internal-stress oracle.

``tools/mpas_seaice_oracle`` compiles MPAS-Model v8.4.2's
``mpas_seaice_velocity_solver_constitutive_relation.F`` and
``mpas_seaice_velocity_solver_variational.F`` verbatim (gfortran -O0
-ffp-contract=off, RKIND = 4) and drives them with ``run_oracle.F90``.  This
module writes the driver's input stream, reads its output stream, and builds
the cases.

Array convention: a Fortran array ``a(maxEdges, nCells)`` is the numpy C-order
array of shape ``(nCells, maxEdges)``.  Both have the same bytes, and the GPU
kernels index them the same way.  ``basis*(maxEdges, maxEdges, nCells)`` is
``(nCells, maxEdges, maxEdges)``, and ``cellsOnVertex(vertexDegree, nVertices)``
is ``(nVertices, vertexDegree)``.  Connectivity stays 1-based, as MPAS has it.

The cases come from numpy's legacy ``RandomState``, whose stream numpy keeps
fixed.  They use only ``randint`` and ``random_sample``, which are integer
arithmetic plus one exact float64 scaling, so the input bytes are the same on
every platform.  ``input_digest`` is recorded beside the fixture, and the test
refuses a regenerated case whose bytes differ.

The meshes have random connectivity. Each gather index is valid, but the
geometry is not a real Voronoi mesh: the routines under test are gathers and
pointwise arithmetic over whatever mesh arrays they are given, so word
equality over random gathers is the same proof.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data"
               / "oracles" / "mpas_seaice_evp")

INT_FIELDS = ("nEdgesOnCell", "verticesOnCell", "cellsOnVertex", "cellVerticesAtVertex",
              "solveStress", "solveVelocity")
MESH_REAL_FIELDS = ("dvEdge", "areaCell", "areaTriangle", "uVelocity", "vVelocity",
                    "icePressure", "tanLatVertexRotatedOverRadius")
BASIS_FIELDS = ("basisGradientU", "basisGradientV", "basisIntegralsU", "basisIntegralsV",
                "basisIntegralsMetric")
#: the outputs, in the order run_oracle.F90 reads their prior contents and
#: writes them back
STATE_FIELDS = ("strain11", "strain22", "strain12", "stress11", "stress22", "stress12",
                "replacementPressure", "stressDivergenceU", "stressDivergenceV",
                "divergence", "shear", "ridgeConvergence", "ridgeShear")

#: (name, seed, nCells, maxEdges, revisedEVP, useColumnPackage, nIter, dynamicsTimeStep)
CASES = (
    ("evp", 20261107, 160, 7, False, True, 12, 3600.0),
    ("revised", 20261108, 160, 7, True, True, 12, 1800.0),
    ("evp_nocolumn", 20261109, 96, 6, False, False, 1, 900.0),
)


def _shape(name: str, d: dict) -> tuple[int, ...]:
    nc, nv, ne, me, vd = d["nCells"], d["nVertices"], d["nEdges"], d["maxEdges"], d["vertexDegree"]
    return {
        "nEdgesOnCell": (nc,), "verticesOnCell": (nc, me), "cellsOnVertex": (nv, vd),
        "cellVerticesAtVertex": (nv, vd), "solveStress": (nc,), "solveVelocity": (nv,),
        "dvEdge": (ne,), "areaCell": (nc,), "areaTriangle": (nv,), "uVelocity": (nv,),
        "vVelocity": (nv,), "icePressure": (nc,), "tanLatVertexRotatedOverRadius": (nv,),
        "stressDivergenceU": (nv,), "stressDivergenceV": (nv,), "divergence": (nc,),
        "shear": (nc,), "ridgeConvergence": (nc,), "ridgeShear": (nc,),
    }.get(name, (nc, me, me) if name in BASIS_FIELDS else (nc, me))


def make_case(name: str, seed: int, n_cells: int, max_edges: int, revised: bool,
              use_column: bool, n_iter: int, dynamics_dt: float) -> dict:
    """One deterministic case.

    The magnitudes are those of an MPAS-Seaice mesh of about 10 km:
    edges of 5-20 km, ice velocities up to 0.5 m/s, ice strength up to
    5e4 N/m, basis gradients of order 1/dx and basis integrals of order
    area/dx.  The case also covers:

    - cells with every vertex at rest, which takes Delta below ``puny``
      (the ``max(Delta, puny)`` branch);
    - cells with zero ice strength;
    - unsolved cells and vertices (``solveStress``/``solveVelocity`` = 0),
      whose outputs keep their prior words;
    - pentagons, hexagons and heptagons;
    - the padding past ``nEdgesOnCell``, filled with an out-of-range vertex
      index and sentinel words, which a correct port never reads.
    """
    rs = np.random.RandomState(seed)
    nc = n_cells
    nv = 2 * nc
    ne = 3 * nc
    vd = 3
    me = max_edges
    sides = np.array([5, 6, 6, 6, 6, 6, 7, 7] if me == 7 else [5, 6, 6, 6], np.int64)
    n_edges_on_cell = sides[rs.randint(0, len(sides), nc)].astype(np.int32)
    vertices_on_cell = np.full((nc, me), nv + 7, np.int32)  # padding: never read
    for c in range(nc):
        vertices_on_cell[c, :n_edges_on_cell[c]] = rs.randint(1, nv + 1, n_edges_on_cell[c])
    cells_on_vertex = rs.randint(1, nc + 1, (nv, vd)).astype(np.int32)
    cell_vertices_at_vertex = np.empty((nv, vd), np.int32)
    for v in range(nv):
        for i in range(vd):
            cell_vertices_at_vertex[v, i] = rs.randint(1, n_edges_on_cell[cells_on_vertex[v, i] - 1] + 1)
    solve_stress = (rs.random_sample(nc) < 0.9).astype(np.int32)
    solve_velocity = (rs.random_sample(nv) < 0.85).astype(np.int32)

    def uniform(lo, hi, shape):
        return (lo + (hi - lo) * rs.random_sample(shape)).astype(np.float32)

    dv_edge = uniform(5.0e3, 2.0e4, ne)
    area_cell = uniform(1.0e8, 5.0e8, nc)
    area_triangle = uniform(3.0e7, 1.5e8, nv)
    u = uniform(-0.5, 0.5, nv)
    v = uniform(-0.5, 0.5, nv)
    # tan(lat)/radius for |lat| up to about 85 degrees (drawn directly: a
    # libm tan would make the bytes depend on the platform)
    tan_lat = uniform(-1.8e-6, 1.8e-6, nv)
    # cells at rest: every vertex still, and no metric term
    for c in np.flatnonzero(rs.random_sample(nc) < 0.06):
        idx = vertices_on_cell[c, :n_edges_on_cell[c]] - 1
        u[idx] = 0.0
        v[idx] = 0.0
        tan_lat[idx] = 0.0
    ice_pressure = uniform(0.0, 5.0e4, nc)
    ice_pressure[rs.random_sample(nc) < 0.08] = 0.0
    basis = {
        "basisGradientU": uniform(-3.0e-4, 3.0e-4, (nc, me, me)),
        "basisGradientV": uniform(-3.0e-4, 3.0e-4, (nc, me, me)),
        "basisIntegralsU": uniform(-4.0e4, 4.0e4, (nc, me, me)),
        "basisIntegralsV": uniform(-4.0e4, 4.0e4, (nc, me, me)),
        "basisIntegralsMetric": uniform(-5.0e7, 5.0e7, (nc, me, me)),
    }
    state = {
        "strain11": uniform(-1.0, 1.0, (nc, me)),
        "strain22": uniform(-1.0, 1.0, (nc, me)),
        "strain12": uniform(-1.0, 1.0, (nc, me)),
        "stress11": uniform(-2.0e4, 2.0e4, (nc, me)),
        "stress22": uniform(-2.0e4, 2.0e4, (nc, me)),
        "stress12": uniform(-1.0e4, 1.0e4, (nc, me)),
        "replacementPressure": uniform(-1.0, 1.0, (nc, me)),
        "stressDivergenceU": uniform(-1.0, 1.0, nv),
        "stressDivergenceV": uniform(-1.0, 1.0, nv),
        "divergence": uniform(-1.0, 1.0, nc),
        "shear": uniform(-1.0, 1.0, nc),
        "ridgeConvergence": uniform(-1.0, 1.0, nc),
        "ridgeShear": uniform(-1.0, 1.0, nc),
    }
    case = {
        "name": name, "nCells": nc, "nVertices": nv, "nEdges": ne, "maxEdges": me,
        "vertexDegree": vd, "revisedEVP": bool(revised), "useColumnPackage": bool(use_column),
        "nIter": int(n_iter), "dynamicsTimeStep": np.float32(dynamics_dt),
        "elasticTimeStep": np.float32(dynamics_dt / 120.0),
        "nEdgesOnCell": n_edges_on_cell, "verticesOnCell": vertices_on_cell,
        "cellsOnVertex": cells_on_vertex, "cellVerticesAtVertex": cell_vertices_at_vertex,
        "solveStress": solve_stress, "solveVelocity": solve_velocity,
        "dvEdge": dv_edge, "areaCell": area_cell, "areaTriangle": area_triangle,
        "uVelocity": u, "vVelocity": v, "icePressure": ice_pressure,
        "tanLatVertexRotatedOverRadius": tan_lat,
    }
    case.update(basis)
    case.update(state)
    return case


def cases() -> list[dict]:
    return [make_case(*row) for row in CASES]


def input_bytes(case: dict) -> bytes:
    """run_oracle.F90's input stream for ``case``."""
    head = np.array([case["nCells"], case["nVertices"], case["nEdges"], case["maxEdges"],
                     case["vertexDegree"], int(case["revisedEVP"]), int(case["useColumnPackage"]),
                     case["nIter"]], "<i4")
    parts = [head.tobytes(),
             np.array([case["dynamicsTimeStep"], case["elasticTimeStep"]], "<f4").tobytes()]
    for name in INT_FIELDS:
        parts.append(np.ascontiguousarray(case[name], "<i4").tobytes())
    for name in MESH_REAL_FIELDS + BASIS_FIELDS + STATE_FIELDS:
        parts.append(np.ascontiguousarray(case[name], "<f4").tobytes())
    return b"".join(parts)


def input_digest(case: dict) -> str:
    return hashlib.sha256(input_bytes(case)).hexdigest()


def read_outputs(raw: bytes, case: dict) -> dict[str, np.ndarray]:
    """run_oracle.F90's output stream -> named float32 arrays."""
    out = {"numericalInertiaCoefficient": np.frombuffer(raw, "<f4", 1, 0).copy()}
    off = 4
    for name in STATE_FIELDS:
        shape = _shape(name, case)
        n = int(np.prod(shape))
        out[name] = np.frombuffer(raw, "<f4", n, off).reshape(shape).copy()
        off += 4 * n
    if off != len(raw):
        raise ValueError(f"oracle output: {len(raw) - off} trailing bytes")
    return out


def load_fixture() -> dict[str, dict[str, np.ndarray]]:
    """The committed oracle outputs, keyed by case then field."""
    with np.load(FIXTURE_DIR / "oracle-r4.npz") as data:
        out: dict[str, dict[str, np.ndarray]] = {}
        for key in data.files:
            case, field = key.split("/", 1)
            out.setdefault(case, {})[field] = data[key]
    return out


def bit_mismatches(a: np.ndarray, b: np.ndarray) -> int:
    """Words that differ as bit patterns (so -0.0 != 0.0 and NaN payloads count)."""
    return int(np.count_nonzero(np.asarray(a, np.float32).view(np.uint32)
                                != np.asarray(b, np.float32).view(np.uint32)))


def run_gpu(case: dict) -> dict[str, np.ndarray]:
    """Run the GPU port (woof.hex.mpas_seaice.evp_v842) on ``case`` exactly as
    run_oracle.F90 drives the Fortran, and return the same named outputs."""
    import cupy as cp
    from woof.hex.mpas_seaice import evp_v842 as evp

    dev = {name: cp.asarray(np.ascontiguousarray(case[name]))
           for name in INT_FIELDS + MESH_REAL_FIELDS + BASIS_FIELDS + STATE_FIELDS}
    mesh = {"nEdgesOnCell": dev["nEdgesOnCell"], "verticesOnCell": dev["verticesOnCell"],
            "cellsOnVertex": dev["cellsOnVertex"], "areaTriangle": dev["areaTriangle"],
            "vertexDegree": case["vertexDegree"]}
    solver = {name: dev[name] for name in ("solveStress", "solveVelocity", "uVelocity", "vVelocity",
                                           "icePressure", "stressDivergenceU", "stressDivergenceV",
                                           "divergence", "shear")}
    variational = {name: dev[name] for name in ("strain11", "strain22", "strain12", "stress11",
                                                "stress22", "stress12", "replacementPressure",
                                                "cellVerticesAtVertex",
                                                "tanLatVertexRotatedOverRadius") + BASIS_FIELDS}
    ridging = {"ridgeConvergence": dev["ridgeConvergence"], "ridgeShear": dev["ridgeShear"]}
    consts = evp.init_evp(dev["dvEdge"], case["nEdges"], case["dynamicsTimeStep"],
                          case["elasticTimeStep"])
    for _ in range(case["nIter"]):
        evp.internal_stress_variational(mesh=mesh, solver=solver, variational=variational,
                                        consts=consts, elastic_time_step=case["elasticTimeStep"],
                                        revised_evp=case["revisedEVP"])
    evp.final_divergence_shear_variational(mesh=mesh, solver=solver, variational=variational,
                                           ridging=ridging,
                                           use_column_package=case["useColumnPackage"])
    out = {"numericalInertiaCoefficient": cp.asnumpy(consts[1:2])}
    for name in STATE_FIELDS:
        out[name] = cp.asnumpy(dev[name])
    return out


def measure(reference: dict[str, dict[str, np.ndarray]] | None = None) -> dict[str, dict[str, int]]:
    """Per case and field, the GPU words that differ from the oracle's."""
    reference = load_fixture() if reference is None else reference
    report: dict[str, dict[str, int]] = {}
    for case in cases():
        got = run_gpu(case)
        want = reference[case["name"]]
        report[case["name"]] = {name: bit_mismatches(got[name], want[name]) for name in want}
    return report
