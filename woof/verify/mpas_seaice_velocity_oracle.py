"""Cases and the stream layout for the MPAS-Seaice EVP velocity-solve oracle.

``tools/mpas_seaice_oracle`` compiles MPAS-Model v8.4.2's
``mpas_seaice_velocity_solver.F`` (with the variational internal stress, the
constitutive relation and the stand-alone column constants) verbatim with
gfortran -O0 -ffp-contract=off at RKIND 4 and RKIND 8, and drives it with
``run_subcycle.F90``. This module writes that driver's input stream, reads
its output stream, and builds the cases.

Two modes (run_subcycle.F90's header has the call lists):

- ``full``: the whole elastic subcycle, with the variational internal stress
  between the velocity solves.
- pieces: the pre-subcycle routines, then ``nSub`` velocity solves against a
  fixed stress-divergence forcing, then the principal stresses. Both
  precisions.

The meshes and the EVP state are those of
:mod:`woof.verify.mpas_seaice_evp_oracle` (random connectivity; the routines
are gathers and pointwise arithmetic, so word equality over random gathers is
the same proof). The velocity-solve fields are drawn in float64 from numpy's
legacy ``RandomState`` with ``random_sample`` only (the EVP fields too,
through ``make_case(dtype=float64)``). The r8 run takes them as
drawn and the r4 run rounds them to float32, so the bytes do not depend on
the platform. ``input_digest`` is pinned beside the fixture.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from woof.verify import mpas_seaice_evp_oracle as evp_oracle

FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data"
               / "oracles" / "mpas_seaice_velocity")
PRECISIONS = {"r4": np.float32, "r8": np.float64}

INT_FIELDS = ("nEdgesOnCell", "verticesOnCell", "cellsOnVertex", "cellVerticesAtVertex",
              "solveStress", "solveVelocity")
INPUT_REAL_FIELDS = ("dvEdge", "areaCell", "areaTriangle", "tanLatVertexRotatedOverRadius", "fVertex",
                     "icePressure", "totalMassVertex", "iceAreaVertex", "uOceanVelocityVertex",
                     "vOceanVelocityVertex", "uVelocityInitial", "vVelocityInitial", "airStressVertexU",
                     "airStressVertexV", "surfaceTiltForceU", "surfaceTiltForceV",
                     "stressDivergenceForcingU", "stressDivergenceForcingV")
BASIS_FIELDS = evp_oracle.BASIS_FIELDS
#: the outputs, in the order run_subcycle.F90 reads their prior contents and
#: writes them back
STATE_FIELDS = ("uVelocity", "vVelocity", "totalMassVertexfVertex", "oceanStressU", "oceanStressV",
                "oceanStressCoeff", "strain11", "strain22", "strain12", "stress11", "stress22", "stress12",
                "replacementPressure", "stressDivergenceU", "stressDivergenceV", "divergence", "shear",
                "ridgeConvergence", "ridgeShear", "principalStress1", "principalStress2")

#: (name, seed, nCells, maxEdges, revisedEVP, useColumnPackage, useOceanStress, full, nSub,
#:  dynamicsTimeStep, precisions)
CASES = (
    ("subcycle", 20261111, 160, 7, False, True, True, True, 120, 3600.0, ("r4", "r8")),
    ("subcycle_revised", 20261112, 160, 7, True, True, True, True, 120, 1800.0, ("r4", "r8")),
    ("solve", 20261113, 160, 7, False, True, True, False, 8, 3600.0, ("r4", "r8")),
    ("solve_revised", 20261114, 160, 7, True, False, True, False, 8, 1800.0, ("r4", "r8")),
    ("solve_no_ocean_stress", 20261115, 96, 6, False, True, False, False, 4, 900.0, ("r4", "r8")),
)


def _shape(name: str, d: dict) -> tuple[int, ...]:
    nc, nv, ne, me, vd = d["nCells"], d["nVertices"], d["nEdges"], d["maxEdges"], d["vertexDegree"]
    if name in BASIS_FIELDS:
        return (nc, me, me)
    if name in ("verticesOnCell", "strain11", "strain22", "strain12", "stress11", "stress22", "stress12",
                "replacementPressure", "principalStress1", "principalStress2"):
        return (nc, me)
    if name in ("cellsOnVertex", "cellVerticesAtVertex"):
        return (nv, vd)
    if name == "dvEdge":
        return (ne,)
    if name in ("nEdgesOnCell", "solveStress", "areaCell", "icePressure", "divergence", "shear",
                "ridgeConvergence", "ridgeShear"):
        return (nc,)
    return (nv,)


def make_case(name: str, seed: int, n_cells: int, max_edges: int, revised: bool, use_column: bool,
              use_ocean_stress: bool, full: bool, n_sub: int, dynamics_dt: float,
              precisions: tuple[str, ...]) -> dict:
    """One deterministic case: the EVP oracle's mesh and stress state, plus
    the velocity-solve fields with the magnitudes of a 10 km mesh:

    - ice mass per area up to 3000 kg/m2, with zero mass on some unsolved
      vertices;
    - fVertex in both hemispheres, including +0.0 and -0.0 (sign(1, -0.0) is
      -1);
    - ocean currents and initial velocities up to 0.5 m/s;
    - air stress up to 0.5 N/m2 and surface tilt up to 0.05 N/m2;
    - ice area from 0.01 to 1.

    nVerticesSolve and nCellsSolve stop short of nVertices and nCells, so the
    halo entries must keep their words.
    """
    case = evp_oracle.make_case(name, seed, n_cells, max_edges, revised, use_column, 1, dynamics_dt,
                                dtype=np.float64)
    rs = np.random.RandomState(seed + 7919)
    nv, nc = case["nVertices"], case["nCells"]

    def uniform(lo, hi, shape):
        return lo + (hi - lo) * rs.random_sample(shape)

    mass = uniform(10.0, 3000.0, nv)
    mass[(case["solveVelocity"] == 0) & (rs.random_sample(nv) < 0.5)] = 0.0
    f_vertex = uniform(-1.5e-4, 1.5e-4, nv)
    zeros = np.flatnonzero(rs.random_sample(nv) < 0.04)
    f_vertex[zeros] = 0.0
    f_vertex[zeros[::2]] = -0.0
    fields = {
        "fVertex": f_vertex,
        "totalMassVertex": mass,
        "iceAreaVertex": uniform(0.01, 1.0, nv),
        "uOceanVelocityVertex": uniform(-0.5, 0.5, nv),
        "vOceanVelocityVertex": uniform(-0.5, 0.5, nv),
        "uVelocityInitial": uniform(-0.5, 0.5, nv),
        "vVelocityInitial": uniform(-0.5, 0.5, nv),
        "airStressVertexU": uniform(-0.5, 0.5, nv),
        "airStressVertexV": uniform(-0.5, 0.5, nv),
        "surfaceTiltForceU": uniform(-0.05, 0.05, nv),
        "surfaceTiltForceV": uniform(-0.05, 0.05, nv),
        "stressDivergenceForcingU": uniform(-1.0, 1.0, nv),
        "stressDivergenceForcingV": uniform(-1.0, 1.0, nv),
        "totalMassVertexfVertex": uniform(-1.0, 1.0, nv),
        "oceanStressU": uniform(-1.0, 1.0, nv),
        "oceanStressV": uniform(-1.0, 1.0, nv),
        "oceanStressCoeff": uniform(-1.0, 1.0, nv),
        "principalStress1": uniform(-1.0, 1.0, (nc, max_edges)),
        "principalStress2": uniform(-1.0, 1.0, (nc, max_edges)),
    }
    case.update(fields)
    case.update({
        "useOceanStress": bool(use_ocean_stress), "full": bool(full), "nSub": int(n_sub),
        "nVerticesSolve": nv - 9, "nCellsSolve": nc - 5,
        "numericalInertiaCoefficient": float(uniform(50.0, 500.0, 1)[0]),
        "precisions": tuple(precisions),
        "dynamicsTimeStep": float(dynamics_dt), "elasticTimeStep": float(dynamics_dt) / 120.0,
    })
    return case


def cases() -> list[dict]:
    return [make_case(*row) for row in CASES]


def input_bytes(case: dict, precision: str) -> bytes:
    """run_subcycle.F90's input stream for ``case`` at ``precision``."""
    real = np.dtype(PRECISIONS[precision]).newbyteorder("<")
    head = np.array([case["nCells"], case["nVertices"], case["nEdges"], case["maxEdges"],
                     case["vertexDegree"], case["nVerticesSolve"], case["nCellsSolve"],
                     int(case["revisedEVP"]), int(case["useColumnPackage"]), int(case["useOceanStress"]),
                     int(case["full"]), case["nSub"]], "<i4")
    # dt/120 is formed in float64 and rounded once to RKIND, as the EVP oracle does
    scalars = [case["dynamicsTimeStep"], case["elasticTimeStep"], case["numericalInertiaCoefficient"]]
    parts = [head.tobytes(), np.array(scalars, real).tobytes()]
    for name in INT_FIELDS:
        parts.append(np.ascontiguousarray(case[name], "<i4").tobytes())
    for name in INPUT_REAL_FIELDS + BASIS_FIELDS + STATE_FIELDS:
        parts.append(np.ascontiguousarray(case[name], real).tobytes())
    return b"".join(parts)


def input_digest(case: dict, precision: str) -> str:
    return hashlib.sha256(input_bytes(case, precision)).hexdigest()


def read_outputs(raw: bytes, case: dict, precision: str) -> dict[str, np.ndarray]:
    """run_subcycle.F90's output stream -> named arrays."""
    real = np.dtype(PRECISIONS[precision]).newbyteorder("<")
    out = {"numericalInertiaCoefficient": np.frombuffer(raw, real, 1, 0).astype(PRECISIONS[precision])}
    off = real.itemsize
    for name in STATE_FIELDS:
        shape = _shape(name, case)
        n = int(np.prod(shape))
        out[name] = np.frombuffer(raw, real, n, off).reshape(shape).astype(PRECISIONS[precision])
        off += real.itemsize * n
    if off != len(raw):
        raise ValueError(f"oracle output: {len(raw) - off} trailing bytes")
    return out


def load_fixture(precision: str) -> dict[str, dict[str, np.ndarray]]:
    """The committed oracle outputs at ``precision``, keyed by case then field."""
    with np.load(FIXTURE_DIR / f"oracle-{precision}.npz") as data:
        out: dict[str, dict[str, np.ndarray]] = {}
        for key in data.files:
            case, field = key.split("/", 1)
            out.setdefault(case, {})[field] = data[key]
    return out


def bit_mismatches(a: np.ndarray, b: np.ndarray) -> int:
    """Words that differ as bit patterns (so -0.0 != 0.0 and NaN payloads count)."""
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype != b.dtype:
        return int(a.size)
    view = np.uint64 if a.dtype == np.float64 else np.uint32
    return int(np.count_nonzero(a.view(view) != b.view(view)))


def run_gpu(case: dict, precision: str) -> dict[str, np.ndarray]:
    """Run the GPU port (woof.hex.mpas_seaice.velocity_v842) on ``case``
    exactly as run_subcycle.F90 drives the Fortran, and return the same
    named outputs."""
    import cupy as cp
    from woof.hex.mpas_seaice import velocity_v842 as vel

    dtype = PRECISIONS[precision]
    raw = input_bytes(case, precision)
    dt = np.frombuffer(raw, np.dtype(dtype).newbyteorder("<"), 3, 48).astype(dtype)
    dev = {name: cp.asarray(np.ascontiguousarray(case[name], np.int32)) for name in INT_FIELDS}
    for name in INPUT_REAL_FIELDS + BASIS_FIELDS + STATE_FIELDS:
        dev[name] = cp.asarray(np.ascontiguousarray(case[name], dtype))
    mesh = {"nEdgesOnCell": dev["nEdgesOnCell"], "verticesOnCell": dev["verticesOnCell"],
            "cellsOnVertex": dev["cellsOnVertex"], "areaTriangle": dev["areaTriangle"],
            "dvEdge": dev["dvEdge"], "fVertex": dev["fVertex"], "vertexDegree": case["vertexDegree"],
            "nEdgesSolve": case["nEdges"], "nVerticesSolve": case["nVerticesSolve"],
            "nCellsSolve": case["nCellsSolve"]}
    icestate = {"totalMassVertex": dev["totalMassVertex"], "iceAreaVertex": dev["iceAreaVertex"]}
    solver = {name: dev[name] for name in (
        "solveStress", "solveVelocity", "uVelocity", "vVelocity", "uVelocityInitial", "vVelocityInitial",
        "icePressure", "totalMassVertexfVertex", "oceanStressU", "oceanStressV", "oceanStressCoeff",
        "uOceanVelocityVertex", "vOceanVelocityVertex", "airStressVertexU", "airStressVertexV",
        "surfaceTiltForceU", "surfaceTiltForceV", "stressDivergenceU", "stressDivergenceV",
        "divergence", "shear")}
    variational = {name: dev[name] for name in (
        "strain11", "strain22", "strain12", "stress11", "stress22", "stress12", "replacementPressure",
        "principalStress1", "principalStress2", "cellVerticesAtVertex",
        "tanLatVertexRotatedOverRadius") + BASIS_FIELDS}
    ridging = {"ridgeConvergence": dev["ridgeConvergence"], "ridgeShear": dev["ridgeShear"]}
    flags = {"use_ocean_stress": case["useOceanStress"]}
    if case["full"]:
        consts = vel.subcycle(mesh=mesh, icestate=icestate, solver=solver, variational=variational,
                              ridging=ridging, n_subcycles=case["nSub"], revised_evp=case["revisedEVP"],
                              use_column_package=case["useColumnPackage"], dynamics_time_step=dt[0],
                              elastic_time_step=dt[1], **flags)
        inertia = consts[1:2]
    else:
        inertia = cp.asarray(dt[2:3])
        vel.pre_subcycle(mesh=mesh, icestate=icestate, solver=solver, variational=variational, **flags)
        for _ in range(case["nSub"]):
            solver["stressDivergenceU"][...] = dev["stressDivergenceForcingU"]
            solver["stressDivergenceV"][...] = dev["stressDivergenceForcingV"]
            vel.velocity_update(mesh=mesh, icestate=icestate, solver=solver, inertia=inertia,
                                revised_evp=case["revisedEVP"], elastic_time_step=dt[1],
                                dynamics_time_step=dt[0], **flags)
        vel.principal_stresses(mesh=mesh, variational=variational)
    out = {"numericalInertiaCoefficient": cp.asnumpy(inertia).astype(dtype)}
    for name in STATE_FIELDS:
        out[name] = cp.asnumpy(dev[name])
    return out


def measure(precision: str, reference: dict | None = None) -> dict[str, dict[str, int]]:
    """Per case and field, the GPU words at ``precision`` that differ from the oracle's."""
    reference = load_fixture(precision) if reference is None else reference
    report: dict[str, dict[str, int]] = {}
    for case in cases():
        if precision not in case["precisions"]:
            continue
        got = run_gpu(case, precision)
        want = reference[case["name"]]
        report[case["name"]] = {name: bit_mismatches(got[name], want[name]) for name in want}
    return report
