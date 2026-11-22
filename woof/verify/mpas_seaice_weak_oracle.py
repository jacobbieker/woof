"""Cases and the stream layout for MPAS-Seaice's weak-form stress divergence.

Two references:

- stand-alone, ``tools/mpas_seaice_oracle/run_velocity_weak.F90``. It runs
  ``seaice_init_evp`` and ``seaice_init_velocity_solver_weak`` (the normal
  vectors and rotated latitudes), then ``oracle_n_iterations`` passes of
  ``seaice_internal_stress_weak``, then
  ``seaice_final_divergence_shear_weak`` (with and without the column
  package's ridging fields). ``mpas_seaice_velocity_solver_weak.F`` and the
  constitutive relation are compiled verbatim, and ``seaice_normal_vectors``
  is cut verbatim (build.sh). The port is
  :mod:`woof.hex.mpas_seaice.weak_v842`: the init on the host, the rest on
  the GPU.
- inside the forward time step, ``run_seaice_step.F90`` with
  ``config_stress_divergence_scheme = 'weak'`` and ``oracle_compute_basis``.
  Its init runs ``seaice_init_velocity_solver_weak`` as
  ``seaice_init_velocity_solver`` does. Then come N steps of the velocity
  solver (and incremental remap) on
  :mod:`woof.verify.mpas_seaice_step_remap_oracle`'s blocks (see
  :data:`STEP_CASES`).

The blocks are :func:`woof.verify.mpas_seaice_remap_oracle.case_mesh`'s.
They are planar hexagons and quadrilaterals, or on the sphere centred on the
north and south poles and at mid-latitude, with uniform and variable
resolution. Each is one block of a decomposition, owned cells plus halos.
The grid rotation is on and off; the EVP is plain and revised. Every
velocity_weak array is noise on input, so a word v8.4.2 does not write and
the port does shows up.

On a sphere the words depend on the C library (``atan2``, ``asin``, ``sin``,
``cos``, ``tan``; :mod:`~woof.hex.mpas_seaice.weak_v842`). The fixtures come
from a glibc 2.44 x86-64 host with AVX2 and FMA, and the port reproduces
them there with ``metric_libm="host"``: :data:`NEEDS_GLIBC` names those
cases.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from woof.verify import mpas_ocean_stream as _stream
from woof.verify import mpas_seaice_remap_oracle as ro
from woof.verify import mpas_seaice_step_remap_oracle as sr
from woof.verify.mpas_ocean_tend_oracle import PRECISIONS
from woof.verify.mpas_seaice_sphere_mesh import RADIUS

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mpas_seaice_weak"
STEP_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mpas_seaice_step_weak"
REAL, INT, DIM = _stream.REAL, _stream.INT, _stream.DIM
_C, _V, _E = ("C1",), ("V1",), ("E1",)
INIT_OUTPUTS = ("normalVectorPolygon", "normalVectorTriangle", "latCellRotated", "latVertexRotated")
WEAK_FIELDS = (
    ("velocity_weak", "normalVectorPolygon", REAL, (2, "ME", "C1")),
    ("velocity_weak", "normalVectorTriangle", REAL, (2, "VD", "V1")),
    ("velocity_weak", "latCellRotated", REAL, _C), ("velocity_weak", "latVertexRotated", REAL, _V),
    *(("velocity_weak", n, REAL, _C) for n in (
        "strain11", "strain22", "strain12", "stress11", "stress22", "stress12", "replacementPressure",
        "principalStress1", "principalStress2")),
)
FIELDS = (
    *(("mesh", n, DIM, ()) for n in ("nCells", "nEdges", "nVertices", "nCellsSolve", "nEdgesSolve",
                                     "nVerticesSolve", "maxEdges", "vertexDegree")),
    ("mesh", "nEdgesOnCell", INT, _C),
    *(("mesh", n, INT, ("ME", "C1")) for n in ("edgesOnCell", "verticesOnCell")),
    *(("mesh", n, INT, (2, "E1")) for n in ("verticesOnEdge", "cellsOnEdge")),
    *(("mesh", n, INT, ("VD", "V1")) for n in ("cellsOnVertex", "edgesOnVertex")),
    *(("mesh", n, REAL, _C) for n in ("xCell", "yCell", "zCell", "areaCell")),
    *(("mesh", n, REAL, _E) for n in ("xEdge", "yEdge", "zEdge", "dvEdge", "dcEdge")),
    *(("mesh", n, REAL, _V) for n in ("xVertex", "yVertex", "zVertex", "areaTriangle")),
    ("boundary", "interiorVertex", INT, _V),
    ("velocity_solver", "solveStress", INT, _C), ("velocity_solver", "solveVelocity", INT, _V),
    *(("velocity_solver", n, REAL, _V) for n in ("uVelocity", "vVelocity", "stressDivergenceU",
                                                 "stressDivergenceV")),
    *(("velocity_solver", n, REAL, _C) for n in ("icePressure", "divergence", "shear")),
    ("velocity_solver", "dynamicsTimeStep", REAL, ()), ("velocity_solver", "elasticTimeStep", REAL, ()),
    *WEAK_FIELDS,
    ("ridging", "ridgeConvergence", REAL, _C), ("ridging", "ridgeShear", REAL, _C),
)
#: the init's geometry, which the oracle writes once
MESH_INPUTS = ("nEdgesOnCell", "edgesOnCell", "verticesOnCell", "verticesOnEdge", "cellsOnEdge", "cellsOnVertex",
               "edgesOnVertex", "xCell", "yCell", "zCell", "areaCell", "xEdge", "yEdge", "zEdge", "dvEdge",
               "dcEdge", "xVertex", "yVertex", "zVertex")
#: (name, seed, mesh kind, owned radius, field overrides (the sphere), iterations, configs)
CASES = (
    ("weak_planar_hex", 20262201, "hex", 2, {}, 3, {}),
    ("weak_planar_quad_revised_column", 20262202, "quad", 2, {}, 4, {
        "config_revised_evp": True, "config_use_column_package": True}),
    ("weak_planar_hex_unrotated_column", 20262203, "hex", 2, {}, 2, {
        "config_rotate_cartesian_grid": False, "config_use_column_package": True}),
    ("weak_sphere_hex_north", 20262204, "hex", 2, {"sphere": "north"}, 3, {"config_use_column_package": True}),
    ("weak_sphere_quad_south_revised", 20262205, "quad", 2, {"sphere": "south"}, 3, {
        "config_revised_evp": True}),
    ("weak_sphere_hex_midlat_varres_unrotated", 20262206, "hex", 2, {"sphere": "midlat", "varres": True}, 2, {
        "config_rotate_cartesian_grid": False, "config_use_column_package": True}),
    ("weak_sphere_quad_midlat_rotated_column", 20262207, "quad", 2, {"sphere": "midlat"}, 2, {
        "config_use_column_package": True}),
)
#: the cases whose words are the C library's (a sphere)
NEEDS_GLIBC = tuple(row[0] for row in CASES if row[4].get("sphere"))


def _uniform(rs, low, high, size):
    """Host-independent ``RandomState.uniform`` (no contraction)."""
    return np.add(low, np.multiply(high - low, rs.random_sample(size)))


def make_case(name, seed, kind, owned, field_overrides, iterations, configs, *, precision="r4") -> dict:
    """One deterministic case on a 10 km block: velocities up to 0.5 m/s,
    ice pressure up to 3e4 N/m, some cells and vertices not solved for."""
    real = PRECISIONS[precision]
    rs = np.random.RandomState(seed)
    m = ro.case_mesh(kind, rs, owned, field_overrides)
    sphere = bool(field_overrides.get("sphere"))
    nc, ne, nv, me, vd = m["nCells"], m["nEdges"], m["nVertices"], m["maxEdges"], m["vertexDegree"]
    dims = {"nCells": nc, "nEdges": ne, "nVertices": nv, "maxEdges": me, "vertexDegree": vd, "nVertLevels": 1}
    config = {"config_revised_evp": False, "config_use_column_package": False,
              "config_rotate_cartesian_grid": True, **configs, "oracle_n_iterations": int(iterations),
              "mesh:on_a_sphere": sphere, "mesh:sphere_radius": float(m["radius"]) if sphere else 0.0}
    case = {"name": name, "precision": precision, "dims": dims, "config": config, "dt": 0.0}
    noise = np.random.RandomState(seed + 1000)
    pools: dict[str, dict] = {}
    for pool, field, k, fshape in FIELDS:
        if k == DIM:
            value = {"nCellsSolve": int(m["nCellsArray"][0]), "nEdgesSolve": int(m["nEdgesArray"][0]),
                     "nVerticesSolve": int(m["nVerticesArray"][0])}.get(field, m.get(field))
        elif pool == "mesh" and field in MESH_INPUTS:
            value = np.asarray(m[field]).astype(real if k == REAL else np.int32)
        elif field == "areaTriangle":
            value = _uniform(noise, 2.0e7, 6.0e7, nv + 1).astype(real)
        elif field == "interiorVertex":
            value = (m["cellsOnVertex"] <= nc).all(axis=1).astype(np.int32)
            value[nv] = 0
        elif field in ("solveStress", "solveVelocity"):
            value = (noise.random_sample(_stream.shape(case, fshape)) < 0.85).astype(np.int32)
        elif field in ("uVelocity", "vVelocity"):
            value = _uniform(noise, -0.5, 0.5, _stream.shape(case, fshape)).astype(real)
        elif field == "icePressure":
            value = _uniform(noise, 0.0, 3.0e4, _stream.shape(case, fshape)).astype(real)
        elif field == "dynamicsTimeStep":
            value = real(1800.0)
        elif field == "elasticTimeStep":
            value = real(real(1800.0) / real(120.0))
        elif field in ("stress11", "stress22", "stress12"):
            value = _uniform(noise, -1.0e4, 1.0e4, _stream.shape(case, fshape)).astype(real)
        else:
            value = _uniform(noise, -1.0, 1.0, _stream.shape(case, fshape)).astype(real)
        pools.setdefault(pool, {})[field] = value
    case["pools"] = pools
    return case


def cases(precision: str = "r4", rows=CASES) -> list[dict]:
    return [make_case(*row, precision=precision) for row in rows]


def input_bytes(case: dict) -> bytes:
    return _stream.input_bytes(FIELDS, case)


def input_digest(case: dict) -> str:
    return hashlib.sha256(input_bytes(case)).hexdigest()


def read_outputs(raw: bytes, case: dict) -> dict[str, np.ndarray]:
    return _stream.read_outputs(FIELDS, raw, case)


word_mismatches = _stream.word_mismatches


def load_fixture(precision: str, fixture_dir: Path = FIXTURE_DIR) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    with np.load(fixture_dir / f"oracle-{precision}.npz") as data:
        for key in data.files:
            case, field = key.split("/", 1)
            out.setdefault(case, {})[field] = data[key]
    return out


def expected_outputs(case: dict, fixture: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every array's oracle words (the fixture's, or the input's where unchanged)."""
    return {f"{pool}/{name}": fixture.get(f"{pool}/{name}", np.asarray(case["pools"][pool][name]))
            for pool, name, k, _ in FIELDS if k in (REAL, INT)}


def run_init(case: dict, *, metric_libm: str | None = None) -> dict[str, np.ndarray]:
    """The host init on ``case`` (velocity_weak's init outputs)."""
    from woof.hex.mpas_seaice import weak_v842 as wv
    real = PRECISIONS[case["precision"]]
    mesh = {**case["pools"]["mesh"], "interiorVertex": case["pools"]["boundary"]["interiorVertex"]}
    out = {name: np.array(case["pools"]["velocity_weak"][name], copy=True) for name in INIT_OUTPUTS}
    wv.initialize(mesh=mesh, configs=case["config"], real=real, out=out, metric_libm=metric_libm)
    return {f"velocity_weak/{name}": out[name] for name in INIT_OUTPUTS}


def run_gpu(case: dict, fixture: dict[str, np.ndarray] | None = None, *, metric_libm: str | None = None,
            corrections=()) -> dict[str, np.ndarray]:
    """The port on ``case`` as run_velocity_weak.F90 drives the Fortran. With
    ``fixture`` the init's outputs are the fixture's (a host whose C library
    is not the oracle's), else :func:`run_init`'s."""
    import cupy as cp
    from woof.hex.mpas_seaice import evp_v842
    from woof.hex.mpas_seaice import weak_v842 as wv

    if fixture is None:
        init = run_init(case, metric_libm=metric_libm)
    else:
        init = {f"velocity_weak/{n}": fixture[f"velocity_weak/{n}"] for n in INIT_OUTPUTS}
    dev = {pool: {name: (cp.asarray(np.ascontiguousarray(v)) if isinstance(v, np.ndarray) else v)
                  for name, v in fields.items()} for pool, fields in case["pools"].items()}
    for key, value in init.items():
        dev["velocity_weak"][key.split("/", 1)[1]] = cp.asarray(np.ascontiguousarray(value))
    mesh, solver, weak = dev["mesh"], dev["velocity_solver"], dev["velocity_weak"]
    config = case["config"]
    nc, nv, ne = (int(case["dims"][k]) for k in ("nCells", "nVertices", "nEdges"))
    consts = evp_v842.init_evp(mesh["dvEdge"][:ne], int(mesh["nEdgesSolve"]),
                               float(case["pools"]["velocity_solver"]["dynamicsTimeStep"]),
                               float(case["pools"]["velocity_solver"]["elasticTimeStep"]))
    tangents = tuple(cp.asarray(t) for t in wv.metric_tangents(weak["latCellRotated"], weak["latVertexRotated"],
                                                                metric_libm=metric_libm))
    with evp_v842.block_counts(cells=nc, vertices=nv):
        for i in range(int(config["oracle_n_iterations"])):
            wv.internal_stress_weak(mesh=mesh, solver=solver, weak=weak, consts=consts, tangents=tangents,
                                    elastic_time_step=float(case["pools"]["velocity_solver"]["elasticTimeStep"]),
                                    revised_evp=bool(config["config_revised_evp"]),
                                    sphere_radius=config["mesh:sphere_radius"], check=i == 0)
        wv.final_divergence_shear_weak(mesh=mesh, solver=solver, weak=weak, ridging=dev["ridging"],
                                       use_column_package=bool(config["config_use_column_package"]),
                                       corrections=corrections)
    return {f"{pool}/{name}": cp.asnumpy(dev[pool][name]) for pool, name, k, _ in FIELDS if k in (REAL, INT)}


def measure(precision: str, reference: dict | None = None, *, rows=CASES, metric_libm: str | None = "host",
            init_from_fixture: bool = False) -> dict[str, dict[str, int]]:
    reference = load_fixture(precision) if reference is None else reference
    report: dict[str, dict[str, int]] = {}
    for case in cases(precision, rows):
        fixture = reference[case["name"]]
        got = run_gpu(case, fixture if init_from_fixture else None, metric_libm=metric_libm)
        want = expected_outputs(case, fixture)
        report[case["name"]] = {key: word_mismatches(got[key], want[key]) for key in want}
    return report


# --- inside the time step ------------------------------------------------------

def _weak_step(sphere: bool) -> dict:
    return {"config_stress_divergence_scheme": "weak", "oracle_compute_basis": True,
            "config_variational_basis": "wachspress", "config_wachspress_integration_type": "dunavant",
            "config_wachspress_integration_order": 8, "config_include_metric_terms": True,
            "mesh:sphere_radius": RADIUS if sphere else 0.0}


#: sr.CASES' rows (the remap's field overrides naming the sphere) and the weak configs
STEP_CASES = (
    ("weak_step_hex", 20262211, "hex", 2, 3, 1800.0, {}, {}, {}, {}, _weak_step(False)),
    ("weak_step_quad_revised_two_subcycles", 20262212, "quad", 2, 2, 3600.0, {
        "pkgColumnTracerLevelIceActive": True}, {
        "config_revised_evp": True, "config_dynamics_subcycle_number": 2, "config_elastic_subcycle_number": 12},
     {}, {}, _weak_step(False)),
    ("weak_step_sphere_hex_north", 20262213, "hex", 2, 3, 1800.0, {}, {}, {}, {"sphere": "north"},
     _weak_step(True)),
    ("weak_step_sphere_quad_midlat_varres_unrotated", 20262214, "quad", 2, 2, 3600.0, {}, {},
     {"config_rotate_cartesian_grid": False}, {"sphere": "midlat", "varres": True}, _weak_step(True)),
)
STEP_NEEDS_GLIBC = tuple(row[0] for row in STEP_CASES if row[-2].get("sphere"))


def make_step_case(*row, precision="r4") -> dict:
    *step_row, weak = row
    case = sr.make_case(*step_row, precision=precision)
    case["config"] = {**case["config"], **weak}
    noise = np.random.RandomState(int(row[1]) + 2000)
    real = PRECISIONS[precision]
    fields = list(case["fields"])
    have = {(r[0], r[1]) for r in fields}
    for row_ in WEAK_FIELDS:
        if (row_[0], row_[1]) not in have:
            fields.append(row_)
        case["pools"].setdefault("velocity_weak", {})[row_[1]] = _uniform(
            noise, -1.0, 1.0, _stream.shape(case, row_[3])).astype(real)
    if ("mesh", "dcEdge") not in have:
        raise ValueError("the step case's mesh carries no dcEdge")
    case["fields"] = tuple(fields)
    return case


def step_cases(precision: str = "r4", rows=STEP_CASES) -> list[dict]:
    return [make_step_case(*row, precision=precision) for row in rows]


def load_step_fixture(precision: str) -> dict[str, dict[str, np.ndarray]]:
    return sr.load_fixture(precision, STEP_FIXTURE_DIR)


def run_step_gpu(case: dict, fixture: dict[str, np.ndarray], *, metric_libm: str | None = "host",
                 init_from_fixture: bool = False, n_steps: int | None = None) -> dict[str, np.ndarray]:
    """``woof.hex.mpas_seaice.timestep_v842`` on ``case`` as run_seaice_step.F90
    drives the Fortran: the weak init computed (or, ``init_from_fixture``, the
    fixture's words: a host whose C library is not the oracle's)."""
    import cupy as cp
    from woof.hex.mpas_seaice import timestep_v842 as ts

    dev = sr.device_pools(case, fixture)
    config = case["config"]
    if init_from_fixture:
        for name in INIT_OUTPUTS:
            dev["velocity_weak"][name] = cp.asarray(np.ascontiguousarray(fixture[f"velocity_weak/{name}"]))
    clock = ts.initialize(pools=dev, configs=config, remap_geometry="input",
                          basis="input" if init_from_fixture else "compute", metric_libm=metric_libm)
    ts.advance(pools=dev, configs=config, clock=clock, metric_libm=metric_libm,
               n_steps=int(config["oracle_n_steps"]) if n_steps is None else n_steps)
    return {f"{pool}/{name}": cp.asnumpy(dev[pool][name]) for pool, name, kind, _ in case["fields"]
            if kind in (REAL, INT) and dev[pool][name] is not None}


def measure_step(precision: str, reference: dict | None = None, *, rows=STEP_CASES,
                 metric_libm: str | None = "host", init_from_fixture: bool = False) -> dict[str, dict[str, int]]:
    reference = load_step_fixture(precision) if reference is None else reference
    report: dict[str, dict[str, int]] = {}
    for case in step_cases(precision, rows):
        fixture = reference[case["name"]]
        got = run_step_gpu(case, fixture, metric_libm=metric_libm, init_from_fixture=init_from_fixture)
        want = ro.expected_outputs(case, fixture)
        report[case["name"]] = {key: sr.word_mismatches(got[key], want[key]) for key in want
                                if ro.graded(case, key)}
    return report
