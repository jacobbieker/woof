"""Cases and the stream layout for the MPAS-Seaice time-step oracle.

``tools/mpas_seaice_oracle/run_seaice_step.F90`` drives MPAS-Model v8.4.2's
forward model over N steps: ``seaice_init_evp`` and the velocity solver's time
steps, ``seaice_initialize_time_diagnostics``, then per step the clock,
``seaice_timestep`` and ``seaice_timestep_finalize``. These are cut verbatim,
and ``mpas_seaice_velocity_solver.F`` is compiled verbatim. Everything is
built with gfortran -O0 -ffp-contract=off at RKIND 4 and 8. The stream is
:mod:`woof.verify.mpas_ocean_stream`'s, with MPAS-Seaice's pools.

The column package, advection and forcing are off: this is the dynamics-only
step. The meshes are :func:`woof.verify.mpas_ocean_hexmesh.hex_mesh`'s culled
hexagonal tori, as one block of a decomposition (owned cells and three halo
layers, so ``nCellsSolve`` < ``nCells``). The masks MPAS derives from the mesh
(``interiorVertex``, ``cellVerticesAtVertex``) are derived the same way; the
variational basis, normally computed by the Wachspress setup, is drawn at
random with a 10 km mesh's magnitudes.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from woof.verify import mpas_ocean_stream as _stream
from woof.verify import mpas_ocean_tend_oracle as _tend
from woof.verify.mpas_ocean_hexmesh import hex_mesh
from woof.verify.mpas_ocean_tend_oracle import PRECISIONS

FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mpas_seaice_step")
CASES_PER_PRECISION = True
REAL, INT, DIM = _stream.REAL, _stream.INT, _stream.DIM
_C, _V, _E = ("C1",), ("V1",), ("E1",)
FIELDS = (
    ("mesh", "nCells", DIM, ()), ("mesh", "nEdges", DIM, ()), ("mesh", "nVertices", DIM, ()),
    ("mesh", "nCellsSolve", DIM, ()), ("mesh", "nEdgesSolve", DIM, ()), ("mesh", "nVerticesSolve", DIM, ()),
    ("mesh", "maxEdges", DIM, ()), ("mesh", "vertexDegree", DIM, ()), ("mesh", "nCategories", DIM, ()),
    ("mesh", "nEdgesOnCell", INT, _C), ("mesh", "cellsOnCell", INT, ("ME", "C1")),
    ("mesh", "verticesOnCell", INT, ("ME", "C1")), ("mesh", "cellsOnVertex", INT, ("VD", "V1")),
    ("mesh", "dvEdge", REAL, _E), ("mesh", "areaCell", REAL, _C), ("mesh", "areaTriangle", REAL, _V),
    ("mesh", "kiteAreasOnVertex", REAL, ("VD", "V1")), ("mesh", "fVertex", REAL, _V),
    ("mesh", "itimestep", INT, (1,)),
    ("boundary", "interiorVertex", INT, _V),
    ("velocity_solver", "solveStress", INT, _C), ("velocity_solver", "solveVelocity", INT, _V),
    ("velocity_solver", "solveVelocityPrevious", INT, _V),
    *(("velocity_solver", n, REAL, _V) for n in (
        "uVelocity", "vVelocity", "uVelocityInitial", "vVelocityInitial", "totalMassVertexfVertex",
        "oceanStressU", "oceanStressV", "oceanStressCoeff", "uOceanVelocityVertex", "vOceanVelocityVertex",
        "airStressVertexU", "airStressVertexV", "surfaceTiltForceU", "surfaceTiltForceV",
        "seaSurfaceTiltVertexU", "seaSurfaceTiltVertexV", "stressDivergenceU", "stressDivergenceV")),
    *(("velocity_solver", n, REAL, _C) for n in (
        "icePressure", "airStressCellU", "airStressCellV", "divergence", "shear", "oceanStressCellU",
        "oceanStressCellV")),
    ("velocity_solver", "dynamicsTimeStep", REAL, ()), ("velocity_solver", "elasticTimeStep", REAL, ()),
    *(("velocity_variational", n, REAL, ("ME", "C1")) for n in (
        "strain11", "strain22", "strain12", "stress11", "stress22", "stress12", "replacementPressure",
        "principalStress1", "principalStress2")),
    ("velocity_variational", "cellVerticesAtVertex", INT, ("VD", "V1")),
    ("velocity_variational", "tanLatVertexRotatedOverRadius", REAL, _V),
    *(("velocity_variational", n, REAL, ("ME", "ME", "C1")) for n in (
        "basisGradientU", "basisGradientV", "basisIntegralsU", "basisIntegralsV", "basisIntegralsMetric")),
    ("icestate", "iceAreaCellInitial", REAL, _C), ("icestate", "iceAreaVertex", REAL, _V),
    ("icestate", "openWaterArea", REAL, _C), ("icestate", "totalMassCell", REAL, _C),
    ("icestate", "totalMassVertex", REAL, _V),
    *(("tracers", n, REAL, (1, "NCAT", "C1")) for n in ("iceAreaCategory", "iceVolumeCategory",
                                                       "snowVolumeCategory")),
    *(("tracers_aggregate", n, REAL, _C) for n in ("iceAreaCell", "iceVolumeCell", "snowVolumeCell")),
    *(("atmos_coupling", n, REAL, _C) for n in ("airDensity", "uAirVelocity", "vAirVelocity",
                                                "atmosReferenceTemperature2m", "atmosReferenceHumidity2m",
                                                "atmosReferenceSpeed10m")),
    ("ocean_coupling", "landIceMask", INT, _C), ("ocean_coupling", "landIceMaskVertex", INT, _V),
    *(("ocean_coupling", n, REAL, _C) for n in ("seaSurfaceTiltU", "seaSurfaceTiltV", "uOceanVelocity",
                                                "vOceanVelocity")),
    ("ridging", "ridgeConvergence", REAL, _C), ("ridging", "ridgeShear", REAL, _C),
    ("diagnostics", "daysSinceStartOfSim", REAL, ()), ("diagnostics", "timeAverageTestVariable", REAL, ()),
)
BASE_CONFIG = {
    "config_start_time": "0001-01-01_00:00:00", "diagnostics:xtime": "0001-01-01_00:00:00",
    "diagnostics:simulationStartTime": "0001-01-01_00:00:00", "config_do_restart": False,
    "config_use_advection": False, "config_advection_type": "incremental_remap", "config_perform_unit_test": False,
    "config_use_column_package": False,
    "config_use_column_vertical_thermodynamics": False, "config_use_column_biogeochemistry": False,
    "config_use_velocity_solver": True, "config_stress_divergence_scheme": "variational",
    "config_revised_evp": False, "config_use_air_stress": True, "config_use_ocean_stress": True,
    "config_use_surface_tilt": True, "config_geostrophic_surface_tilt": True, "config_use_halo_exch": False,
    "config_aggregate_halo_exch": False, "config_reuse_halo_exch": False, "config_check_state": False,
    "config_use_forcing": False, "config_dynamics_subcycle_number": 1, "config_elastic_subcycle_number": 30,
    "config_use_ice_age": False,
}
#: (name, seed, (nq, nr), owned columns, steps, config_dt [s], config overrides)
CASES = (
    ("evp_three_steps", 20261601, (12, 6), 3, 3, 1800.0, {}),
    ("revised_two_dynamics_subcycles", 20261602, (12, 6), 3, 2, 3600.0, {
        "config_revised_evp": True, "config_dynamics_subcycle_number": 2, "config_elastic_subcycle_number": 12}),
    ("no_air_no_tilt", 20261603, (10, 5), 3, 2, 1800.0, {
        "config_use_air_stress": False, "config_use_surface_tilt": False, "config_use_ocean_stress": False,
        "config_start_time": "2000-02-28_23:30:00", "diagnostics:simulationStartTime": "2000-01-01_00:00:00"}),
)


def _interval(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 86400:04d}_{s % 86400 // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def make_case(name, seed, torus, owned, steps, dt, overrides, precision="r4", mesh=None,
              n_categories: int = 5) -> dict:
    """One deterministic case with a 10 km mesh's magnitudes (``mesh``, if
    given, instead of a culled torus): five thickness categories, an ice-free
    run of cells, fVertex in both hemispheres, ocean currents up to 0.5 m/s,
    winds up to 20 m/s."""
    rs = np.random.RandomState(seed)
    real = PRECISIONS[precision]
    m = hex_mesh(*torus, 0.08, 1, rs, min_levels=1, owned_columns=owned) if mesh is None else mesh
    nc, ne, nv, me, vd = m["nCells"], m["nEdges"], m["nVertices"], m["maxEdges"], m["vertexDegree"]
    ncat = n_categories
    config = {**BASE_CONFIG, **overrides, "config_dt": float(dt), "oracle_clock_step": _interval(dt),
              "oracle_n_steps": steps}

    def uniform(lo, hi, shp):
        return lo + (hi - lo) * rs.random_sample(shp)

    cells_on_vertex, vertices_on_cell = m["cellsOnVertex"], m["verticesOnCell"]
    # seaice_mesh's interior vertices: every cell around it present
    interior = (cells_on_vertex <= nc).all(axis=1).astype(np.int32)
    interior[nv] = 0
    # the variational setup's index of a vertex among each of its cells' vertices
    cell_vertices = np.ones((nv + 1, vd), np.int32)
    for v in range(nv):
        for j in range(vd):
            c = cells_on_vertex[v, j]
            if c <= nc:
                cell_vertices[v, j] = int(np.flatnonzero(vertices_on_cell[c - 1] == v + 1)[0]) + 1
    area = uniform(0.0, 0.2, (nc + 1, ncat))
    free = rs.random_sample(nc + 1) < 0.15
    free[nc // 3: nc // 3 + nc // 6] = True
    area[free] = 0.0
    thickness = uniform(0.1, 4.0, (nc + 1, ncat))
    snow = uniform(0.0, 0.5, (nc + 1, ncat))
    f_vertex = uniform(-1.5e-4, 1.5e-4, nv + 1)
    dims = {"nCells": nc, "nEdges": ne, "nVertices": nv, "maxEdges": me, "vertexDegree": vd, "nCategories": ncat,
            "nVertLevels": 1}
    case = {"name": name, "precision": precision, "dims": dims, "config": config, "dt": float(dt)}
    values = {
        "nCells": nc, "nEdges": ne, "nVertices": nv, "nCellsSolve": int(m["nCellsArray"][0]),
        "nEdgesSolve": int(m["nEdgesArray"][0]), "nVerticesSolve": int(m["nVerticesArray"][0]), "maxEdges": me,
        "vertexDegree": vd, "nCategories": ncat, "nEdgesOnCell": m["nEdgesOnCell"], "cellsOnCell": m["cellsOnCell"],
        "verticesOnCell": vertices_on_cell, "cellsOnVertex": cells_on_vertex,
        "dvEdge": uniform(5.0e3, 2.0e4, ne + 1), "areaCell": uniform(0.5e8, 1.5e8, nc + 1),
        "areaTriangle": uniform(0.3e8, 0.7e8, nv + 1), "kiteAreasOnVertex": uniform(0.1e8, 0.3e8, (nv + 1, vd)),
        "fVertex": f_vertex, "itimestep": np.zeros(1, np.int32), "interiorVertex": interior,
        "cellVerticesAtVertex": cell_vertices,
        "tanLatVertexRotatedOverRadius": uniform(-3.0e-7, 3.0e-7, nv + 1),
        "basisGradientU": uniform(-2.0e-4, 2.0e-4, (nc + 1, me, me)),
        "basisGradientV": uniform(-2.0e-4, 2.0e-4, (nc + 1, me, me)),
        "basisIntegralsU": uniform(-5.0e3, 5.0e3, (nc + 1, me, me)),
        "basisIntegralsV": uniform(-5.0e3, 5.0e3, (nc + 1, me, me)),
        "basisIntegralsMetric": uniform(1.0e6, 2.0e7, (nc + 1, me, me)),
        "iceAreaCategory": area.reshape(nc + 1, ncat, 1),
        "iceVolumeCategory": (area * thickness).reshape(nc + 1, ncat, 1),
        "snowVolumeCategory": (area * snow).reshape(nc + 1, ncat, 1),
        "uOceanVelocity": uniform(-0.5, 0.5, nc + 1), "vOceanVelocity": uniform(-0.5, 0.5, nc + 1),
        "uAirVelocity": uniform(-20.0, 20.0, nc + 1), "vAirVelocity": uniform(-20.0, 20.0, nc + 1),
        "airDensity": uniform(1.1, 1.4, nc + 1), "seaSurfaceTiltU": uniform(-1.0e-5, 1.0e-5, nc + 1),
        "seaSurfaceTiltV": uniform(-1.0e-5, 1.0e-5, nc + 1), "openWaterArea": uniform(0.0, 1.0, nc + 1),
        "landIceMask": np.zeros(nc + 1, np.int32), "landIceMaskVertex": np.zeros(nv + 1, np.int32),
        "uVelocity": uniform(-0.3, 0.3, nv + 1), "vVelocity": uniform(-0.3, 0.3, nv + 1),
        "solveStress": np.zeros(nc + 1, np.int32), "solveVelocity": np.zeros(nv + 1, np.int32),
        "solveVelocityPrevious": (rs.random_sample(nv + 1) < 0.5).astype(np.int32),
        "iceAreaCellInitial": uniform(0.0, 1.0, nc + 1),
        "dynamicsTimeStep": 0.0, "elasticTimeStep": 0.0, "daysSinceStartOfSim": 0.0,
        "timeAverageTestVariable": 0.0,
    }
    pools: dict[str, dict] = {}
    for pool, field, kind, fshape in FIELDS:
        if field in values:
            value = values[field]
        else:
            value = uniform(-1.0, 1.0, _stream.shape(case, fshape))
        if kind == REAL:
            value = np.asarray(value).astype(real)
        elif kind == INT:
            value = np.asarray(value).astype(np.int32)
        pools.setdefault(pool, {})[field] = value
    case["pools"] = pools
    return case


def cases(precision: str = "r4") -> list[dict]:
    return [make_case(*row, precision=precision) for row in CASES]


def input_bytes(case: dict) -> bytes:
    return _stream.input_bytes(FIELDS, case)


def input_digest(case: dict) -> str:
    return hashlib.sha256(input_bytes(case)).hexdigest()


def read_outputs(raw: bytes, case: dict) -> dict[str, np.ndarray]:
    return _stream.read_outputs(FIELDS, raw, case)


word_mismatches = _stream.word_mismatches


def load_fixture(precision: str) -> dict[str, dict[str, np.ndarray]]:
    return _tend.load_fixture(precision, FIXTURE_DIR)


def run_gpu(case: dict, n_steps: int | None = None) -> dict[str, np.ndarray]:
    """Run woof.hex.mpas_seaice.timestep_v842 on ``case`` as run_seaice_step.F90 drives the Fortran."""
    import cupy as cp
    from woof.hex.mpas_seaice import timestep_v842 as ts

    dev = _stream.on_device(FIELDS, case)
    config = case["config"]
    dev["diagnostics"]["xtime"] = config["diagnostics:xtime"]
    dev["diagnostics"]["simulationStartTime"] = config["diagnostics:simulationStartTime"]
    clock = ts.initialize(pools=dev, configs=config)
    ts.advance(pools=dev, configs=config, clock=clock,
               n_steps=int(config["oracle_n_steps"]) if n_steps is None else n_steps)
    return {f"{pool}/{name}": cp.asnumpy(dev[pool][name]) for pool, name, kind, _ in FIELDS
            if kind in (REAL, INT)}


def measure(precision: str, reference: dict | None = None) -> dict[str, dict[str, int]]:
    reference = load_fixture(precision) if reference is None else reference
    report: dict[str, dict[str, int]] = {}
    for case in cases(precision):
        got = run_gpu(case)
        want = reference[case["name"]]
        report[case["name"]] = {name: word_mismatches(got[name], want[name]) for name in want}
    return report
