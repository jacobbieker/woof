"""Cases and the stream layout for the MPAS-Seaice pre-subcycle oracle.

``tools/mpas_seaice_oracle`` compiles MPAS-Model v8.4.2's
``mpas_seaice_velocity_solver.F`` and ``seaice_mesh``'s interpolation
routines verbatim with gfortran -O0 -ffp-contract=off at RKIND 4 and RKIND 8,
and drives ``velocity_solver_pre_subcycle`` with ``run_presubcycle.F90``
(without ``ice_strength`` under the column vertical thermodynamics, whose
column-package strength the port refuses). This module writes that driver's
input stream, reads its output stream, and builds the cases.

The meshes are random local connectivity (1-based, every index inside the
arrays; a cell's neighbours and a vertex's cells lie a few indices away, so
an ice-free run of cells leaves unsolved cells and vertices): the routines
are gathers, a short neighbour search and pointwise arithmetic, so word
equality over random gathers is the same proof. Fields are drawn in
float64 from numpy's legacy ``RandomState`` with ``random_sample`` only; the
r8 run takes them as drawn and the r4 run rounds them to float32, so the
bytes do not depend on the platform. ``input_digest`` is pinned beside the
fixture.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data"
               / "oracles" / "mpas_seaice_presubcycle")
PRECISIONS = {"r4": np.float32, "r8": np.float64}
TILTS = ("none", "geostrophic", "ssh_gradient")

INT_FIELDS = ("nEdgesOnCell", "cellsOnCell", "verticesOnCell", "cellsOnVertex", "interiorVertex",
              "landIceMask", "landIceMaskVertex")
#: integer outputs, whose prior contents the driver also reads
INT_STATE_FIELDS = ("solveStress", "solveVelocity", "solveVelocityPrevious")
INPUT_REAL_FIELDS = ("areaCell", "areaTriangle", "kiteAreasOnVertex", "fVertex", "iceAreaCategory",
                     "iceVolumeCategory", "snowVolumeCategory", "uOceanVelocity", "vOceanVelocity",
                     "uAirVelocity", "vAirVelocity", "airDensity", "seaSurfaceTiltU", "seaSurfaceTiltV",
                     "openWaterArea")
#: the real outputs, in the order run_presubcycle.F90 reads their prior
#: contents and writes them back
STATE_FIELDS = ("iceAreaCell", "iceVolumeCell", "snowVolumeCell", "totalMassCell", "iceAreaCellInitial",
                "icePressure", "airStressCellU", "airStressCellV", "iceAreaVertex", "totalMassVertex",
                "uOceanVelocityVertex", "vOceanVelocityVertex", "uVelocity", "vVelocity", "uVelocityInitial",
                "vVelocityInitial", "stressDivergenceU", "stressDivergenceV", "oceanStressU", "oceanStressV",
                "oceanStressCoeff", "totalMassVertexfVertex", "airStressVertexU", "airStressVertexV",
                "seaSurfaceTiltVertexU", "seaSurfaceTiltVertexV", "surfaceTiltForceU", "surfaceTiltForceV",
                "strain11", "strain22", "strain12", "stress11", "stress22", "stress12")
CELL_FIELDS = ("nEdgesOnCell", "landIceMask", "solveStress", "areaCell", "uOceanVelocity", "vOceanVelocity",
               "uAirVelocity", "vAirVelocity", "airDensity", "seaSurfaceTiltU", "seaSurfaceTiltV",
               "openWaterArea", "iceAreaCell", "iceVolumeCell", "snowVolumeCell", "totalMassCell",
               "iceAreaCellInitial", "icePressure", "airStressCellU", "airStressCellV")
EDGE_CELL_FIELDS = ("cellsOnCell", "verticesOnCell", "strain11", "strain22", "strain12", "stress11",
                    "stress22", "stress12")

#: (name, seed, nCells, maxEdges, useColumnPackage, useColumnVerticalThermodynamics,
#:  useAirStress, useOceanStress, tilt)
CASES = (
    ("standalone_geostrophic", 20261121, 160, 7, False, False, True, True, "geostrophic"),
    ("column_thermo_ssh_tilt", 20261122, 160, 7, True, True, True, True, "ssh_gradient"),
    ("column_no_thermo_no_air", 20261123, 128, 6, True, False, False, False, "none"),
    ("standalone_ssh_no_air", 20261124, 96, 6, False, False, False, True, "ssh_gradient"),
)


def _shape(name: str, d: dict) -> tuple[int, ...]:
    nc, nv, me, vd, ncat = d["nCells"], d["nVertices"], d["maxEdges"], d["vertexDegree"], d["nCategories"]
    if name in ("iceAreaCategory", "iceVolumeCategory", "snowVolumeCategory"):
        return (nc, ncat)
    if name in EDGE_CELL_FIELDS:
        return (nc, me)
    if name in ("cellsOnVertex", "kiteAreasOnVertex"):
        return (nv, vd)
    if name in CELL_FIELDS:
        return (nc,)
    return (nv,)


def make_case(name: str, seed: int, n_cells: int, max_edges: int, use_column: bool, use_thermo: bool,
              use_air_stress: bool, use_ocean_stress: bool, tilt: str) -> dict:
    """One deterministic case, with the magnitudes of a 10 km mesh:

    - five thickness categories; a run of a quarter of the cells and a
      tenth of the rest ice-free, some with
      ice area or mass just under the masks' minima (0.001, 0.01 kg/m2);
      under the column vertical thermodynamics, where the initial area is an
      input, some initial areas exactly 0.001 (not above the minimum), a few
      of them on lone thin-ice cells inside the ice-free run, where the mask
      turns on that comparison alone;
    - land ice on some cells and vertices, boundary vertices
      (interiorVertex 0);
    - fVertex in both hemispheres, including +0.0 and -0.0;
    - ocean currents up to 0.5 m/s, winds up to 20 m/s, sea-surface tilt up
      to 1e-5.

    nVerticesSolve and nCellsSolve stop short of nVertices and nCells, so the
    halo entries must keep their words. Every prior output word is random.
    """
    if tilt not in TILTS:
        raise ValueError(tilt)
    rs = np.random.RandomState(seed)
    nc, me, vd, ncat = n_cells, max_edges, 3, 5
    nv = 2 * nc + 3

    def uniform(lo, hi, shape):
        return lo + (hi - lo) * rs.random_sample(shape)

    def index(n, shape):
        return (1 + np.floor(n * rs.random_sample(shape))).astype(np.int32)

    def near(centre, spread, n, shape):
        offset = np.floor((2 * spread + 1) * rs.random_sample(shape)).astype(np.int64) - spread
        return (1 + np.clip(centre + offset, 0, n - 1)).astype(np.int32)

    def flags(p, shape):
        return (rs.random_sample(shape) < p).astype(np.int32)

    n_edges_on_cell = (3 + np.floor((me - 2) * rs.random_sample(nc))).astype(np.int32)
    area = uniform(0.0, 0.2, (nc, ncat))
    free = rs.random_sample(nc) < 0.1
    run = np.arange(nc // 5, nc // 5 + nc // 4)
    free[run] = True                                                      # an ice-free run of cells
    lone = run[4:-4:8]                                                    # thin ice alone inside it
    area[free] = 0.0
    thin = np.flatnonzero(~free & (rs.random_sample(nc) < 0.08))
    area[thin[::2]] = uniform(0.0, 0.00019, (len(thin[::2]), ncat))      # total area under 0.001
    area[lone] = 0.0002
    thickness = uniform(0.1, 4.0, (nc, ncat))
    thickness[thin[1::2]] = uniform(0.0, 1.0e-6, (len(thin[1::2]), ncat))  # mass under 0.01
    snow = uniform(0.0, 0.5, (nc, ncat))
    f_vertex = uniform(-1.5e-4, 1.5e-4, nv)
    zeros = np.flatnonzero(rs.random_sample(nv) < 0.04)
    f_vertex[zeros] = 0.0
    f_vertex[zeros[::2]] = -0.0
    case = {
        "name": name, "nCells": nc, "nVertices": nv, "maxEdges": me, "vertexDegree": vd, "nCategories": ncat,
        "nVerticesSolve": nv - 9, "nCellsSolve": nc - 5,
        "useColumnPackage": bool(use_column), "useColumnVerticalThermodynamics": bool(use_thermo),
        "useAirStress": bool(use_air_stress), "useOceanStress": bool(use_ocean_stress), "tilt": tilt,
        "nEdgesOnCell": n_edges_on_cell,
        "cellsOnCell": near(np.arange(nc)[:, None], 3, nc, (nc, me)),
        "verticesOnCell": index(nv, (nc, me)),
        "cellsOnVertex": near(np.arange(nv)[:, None] // 2, 2, nc, (nv, vd)),
        "interiorVertex": 1 - flags(0.1, nv),
        "landIceMask": flags(0.05, nc),
        "landIceMaskVertex": flags(0.05, nv),
        "solveStress": flags(0.5, nc),
        "solveVelocity": flags(0.5, nv),
        "solveVelocityPrevious": flags(0.5, nv),
        "areaCell": uniform(0.5e8, 1.5e8, nc),
        "areaTriangle": uniform(0.3e8, 0.7e8, nv),
        "kiteAreasOnVertex": uniform(0.1e8, 0.3e8, (nv, vd)),
        "fVertex": f_vertex,
        "iceAreaCategory": area,
        "iceVolumeCategory": area * thickness,
        "snowVolumeCategory": area * snow,
        "uOceanVelocity": uniform(-0.5, 0.5, nc),
        "vOceanVelocity": uniform(-0.5, 0.5, nc),
        "uAirVelocity": uniform(-20.0, 20.0, nc),
        "vAirVelocity": uniform(-20.0, 20.0, nc),
        "airDensity": uniform(1.1, 1.4, nc),
        "seaSurfaceTiltU": uniform(-1.0e-5, 1.0e-5, nc),
        "seaSurfaceTiltV": uniform(-1.0e-5, 1.0e-5, nc),
        "openWaterArea": uniform(0.0, 1.0, nc),
    }
    for field in STATE_FIELDS:
        case[field] = uniform(-1.0, 1.0, _shape(field, case))
    initial = uniform(0.0, 1.0, nc)
    initial[rs.random_sample(nc) < 0.2] = 0.0
    initial[rs.random_sample(nc) < 0.06] = 0.001
    initial[run] = 0.0
    initial[lone] = 0.001
    case["iceAreaCellInitial"] = initial
    case["airStressCellU"] = uniform(-0.5, 0.5, nc)
    case["airStressCellV"] = uniform(-0.5, 0.5, nc)
    return case


def cases() -> list[dict]:
    return [make_case(*row) for row in CASES]


def input_bytes(case: dict, precision: str) -> bytes:
    """run_presubcycle.F90's input stream for ``case`` at ``precision``."""
    real = np.dtype(PRECISIONS[precision]).newbyteorder("<")
    head = np.array([case["nCells"], case["nVertices"], case["maxEdges"], case["vertexDegree"],
                     case["nCategories"], case["nVerticesSolve"], case["nCellsSolve"],
                     int(case["useColumnPackage"]), int(case["useColumnVerticalThermodynamics"]),
                     int(case["useAirStress"]), int(case["useOceanStress"]), TILTS.index(case["tilt"])], "<i4")
    parts = [head.tobytes()]
    for name in INT_FIELDS + INT_STATE_FIELDS:
        parts.append(np.ascontiguousarray(case[name], "<i4").tobytes())
    for name in INPUT_REAL_FIELDS + STATE_FIELDS:
        parts.append(np.ascontiguousarray(case[name], real).tobytes())
    return b"".join(parts)


def input_digest(case: dict, precision: str) -> str:
    return hashlib.sha256(input_bytes(case, precision)).hexdigest()


def read_outputs(raw: bytes, case: dict, precision: str) -> dict[str, np.ndarray]:
    """run_presubcycle.F90's output stream -> named arrays."""
    real = np.dtype(PRECISIONS[precision]).newbyteorder("<")
    out: dict[str, np.ndarray] = {}
    off = 0
    for name in INT_STATE_FIELDS:
        shape = _shape(name, case)
        n = int(np.prod(shape))
        out[name] = np.frombuffer(raw, "<i4", n, off).reshape(shape).astype(np.int32)
        off += 4 * n
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
    if a.dtype != b.dtype or a.shape != b.shape:
        return int(max(a.size, b.size))
    view = {8: np.uint64, 4: np.uint32}[a.dtype.itemsize]
    return int(np.count_nonzero(a.view(view) != b.view(view)))


def run_gpu(case: dict, precision: str) -> dict[str, np.ndarray]:
    """Run the GPU port (woof.hex.mpas_seaice.presubcycle_v842) on ``case``
    exactly as run_presubcycle.F90 drives the Fortran, and return the same
    named outputs."""
    import cupy as cp
    from woof.hex.mpas_seaice import presubcycle_v842 as pre
    from woof.hex.mpas_seaice import velocity_v842 as vel

    dtype = PRECISIONS[precision]
    dev = {name: cp.asarray(np.ascontiguousarray(case[name], np.int32))
           for name in INT_FIELDS + INT_STATE_FIELDS}
    for name in INPUT_REAL_FIELDS + STATE_FIELDS:
        dev[name] = cp.asarray(np.ascontiguousarray(case[name], dtype))
    mesh = {name: dev[name] for name in ("nEdgesOnCell", "cellsOnCell", "cellsOnVertex", "areaCell", "fVertex")}
    mesh.update({"vertexDegree": case["vertexDegree"], "nVerticesSolve": case["nVerticesSolve"],
                 "nCellsSolve": case["nCellsSolve"]})
    pools = {
        "mesh": mesh,
        "boundary": {"interiorVertex": dev["interiorVertex"]},
        "tracers": {name: dev[name] for name in ("iceAreaCategory", "iceVolumeCategory", "snowVolumeCategory")},
        "aggregate": {name: dev[name] for name in ("iceAreaCell", "iceVolumeCell", "snowVolumeCell")},
        "icestate": {name: dev[name] for name in ("totalMassCell", "iceAreaCellInitial", "iceAreaVertex",
                                                  "totalMassVertex")},
        "ocean": {name: dev[name] for name in ("landIceMask", "landIceMaskVertex", "uOceanVelocity",
                                               "vOceanVelocity", "seaSurfaceTiltU", "seaSurfaceTiltV")},
        "atmos": {name: dev[name] for name in ("uAirVelocity", "vAirVelocity", "airDensity")},
        "solver": {name: dev[name] for name in (
            "solveStress", "solveVelocity", "solveVelocityPrevious", "uVelocity", "vVelocity",
            "uVelocityInitial", "vVelocityInitial", "icePressure", "totalMassVertexfVertex", "oceanStressU",
            "oceanStressV", "oceanStressCoeff", "uOceanVelocityVertex", "vOceanVelocityVertex",
            "airStressCellU", "airStressCellV", "airStressVertexU", "airStressVertexV",
            "seaSurfaceTiltVertexU", "seaSurfaceTiltVertexV", "surfaceTiltForceU", "surfaceTiltForceV",
            "stressDivergenceU", "stressDivergenceV")},
        "variational": {name: dev[name] for name in ("strain11", "strain22", "strain12", "stress11",
                                                     "stress22", "stress12")},
    }
    flags = {"use_column_package": case["useColumnPackage"],
             "use_column_vertical_thermodynamics": case["useColumnVerticalThermodynamics"]}
    if not (case["useColumnPackage"] and case["useColumnVerticalThermodynamics"]):
        pre.pre_subcycle(**pools, use_air_stress=case["useAirStress"], use_ocean_stress=case["useOceanStress"],
                         tilt=case["tilt"], **flags)
    else:
        # the driver's call list without ice_strength (refused here by name)
        p = pools
        pre.aggregate_and_masks(mesh=mesh, boundary=p["boundary"], tracers=p["tracers"],
                                aggregate=p["aggregate"], icestate=p["icestate"], ocean=p["ocean"],
                                solver=p["solver"], **flags)
        pre.new_ice_velocities(mesh=mesh, ocean=p["ocean"], solver=p["solver"])
        pre.air_stress(mesh=mesh, atmos=p["atmos"], aggregate=p["aggregate"], solver=p["solver"],
                       use_air_stress=case["useAirStress"], **flags)
        vel.coriolis_and_ocean_stress(mesh=mesh, icestate=p["icestate"], solver=p["solver"],
                                      use_ocean_stress=case["useOceanStress"])
        pre.surface_tilt(mesh=mesh, icestate=p["icestate"], ocean=p["ocean"], solver=p["solver"],
                         tilt=case["tilt"])
        vel.init_subcycle_variables(mesh=mesh, solver=p["solver"], variational=p["variational"])
    return {name: cp.asnumpy(dev[name]) for name in INT_STATE_FIELDS + STATE_FIELDS}


def measure(precision: str, reference: dict | None = None) -> dict[str, dict[str, int]]:
    """Per case and field, the GPU words at ``precision`` that differ from the oracle's."""
    reference = load_fixture(precision) if reference is None else reference
    report: dict[str, dict[str, int]] = {}
    for case in cases():
        got = run_gpu(case, precision)
        want = reference[case["name"]]
        report[case["name"]] = {name: bit_mismatches(got[name], want[name]) for name in want}
    return report
