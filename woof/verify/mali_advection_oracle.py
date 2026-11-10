"""Cases and the stream layout for the MALI thickness/tracer advection oracle.

``tools/mali_oracle`` compiles MALI v8.4.2's land-ice modules verbatim with
gfortran -O0 -ffp-contract=off at RKIND 4 and 8, and ``run_advection.F90`` makes a
restart's ``li_thermal_init`` and then one ``li_advection_thickness_tracers``
call. This module writes that driver's input stream, reads its output, and
builds the cases.

The meshes are random connectivity (1-based, every index the routines read
inside the arrays). Fields are drawn in float64 from numpy's legacy
``RandomState`` with ``random_sample`` only and rounded once to the case's precision, so
the bytes do not depend on the platform. ``input_digest`` is pinned beside
the fixture. A case is built at one precision: the RKIND 4 case rounds the
draws to float32 and the RKIND 8 case keeps them in float64, and the stream,
the fixture (``oracle-r4.npz``, ``oracle-r8.npz``) and the GPU run follow it.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mali_advection"
PRECISIONS = {"r4": np.float32, "r8": np.float64}
THERMAL = {"none": 0, "temperature": 1, "enthalpy": 2}

INT_FIELDS = ("nEdgesOnCell", "cellsOnCell", "cellsOnVertex", "cellsOnEdge", "edgesOnCell", "edgeSignOnCell",
              "cellMask", "vertexMask", "edgeMask")
REAL_FIELDS = ("layerThicknessFractions", "layerCenterSigma", "layerInterfaceSigma", "dvEdge", "dcEdge",
               "areaCell", "layerNormalVelocity", "layerThicknessEdge", "bedTopography", "groundedBasalMassBal",
               "floatingBasalMassBal", "surfaceAirTemperature", "basalTemperature")
#: real outputs, in the order run_advection.F90 reads their prior words and writes them
REAL_STATE = ("thickness", "sfcMassBal", "basalMassBal", "dynamicThickening", "layerThickness", "temperature",
              "waterfrac", "enthalpy")
SCRATCH = ("workLevelCell", "workTracerCell", "workTracerCell2", "workTracerLevelCell", "workTracerLevelCell2")
INT_OUTPUTS = ("cellMask", "vertexMask", "edgeMask")

#: (name, seed, nCells, nVertLevels, velocity solver, thermal solver, advect tracers, thickness advection)
CASES = (
    ("fo_temperature", 20261131, 120, 10, "sia", "temperature", True, "fo"),
    ("fo_enthalpy", 20261132, 120, 10, "sia", "enthalpy", True, "fo"),
    ("fo_thickness_only", 20261133, 96, 6, "none", "temperature", False, "fo"),
    ("fo_no_thermal", 20261134, 80, 8, "sia", "none", True, "fo"),
    ("none", 20261135, 64, 5, "sia", "temperature", True, "none"),
)


def make_case(name: str, seed: int, n_cells: int, n_levels: int, solver: str, thermal: str, advect_tracers: bool,
              advection: str, precision: str = "r4") -> dict:
    """One deterministic case: grounded and floating ice, thin ice and open
    ocean on a random mesh, a month's step.

    - ice up to 2500 m, a quarter of the cells ice-free, a tenth thinner than
      a metre, so ablation melts whole layers and accumulation lands on
      ice-free cells;
    - surface and basal mass balance of both signs, up to a few hundred
      metres a year on the thin cells;
    - layer velocities up to 300 m/yr, with exact +0 and -0 (sign(1, -0) is
      -1), and upwind layer thicknesses up to 300 m, so some layers empty
      (the zero-tracer branch);
    - air temperatures on both sides of 273.15 K.

    nCellsSolve stops short of nCells, so the halo cells' words come from the
    routines that run over every cell only. The scratch arrays are random:
    the model leaves them uninitialised.
    """
    rs = np.random.RandomState(seed)
    real = PRECISIONS[precision]
    nc, nl = n_cells, n_levels
    ne, nv, me = 3 * nc, 2 * nc, 7
    dt = 30.0 * 86400.0

    def uniform(lo, hi, shape):
        return (lo + (hi - lo) * rs.random_sample(shape)).astype(real)

    def ints(lo, hi, shape):
        return (lo + np.floor((hi - lo) * rs.random_sample(shape))).astype(np.int32)

    n_edges_on_cell = np.array([5, 6, 6, 6, 7], np.int32)[ints(0, 5, nc)]
    cells_on_cell = ints(1, nc + 1, (nc, me))
    edges_on_cell = ints(1, ne + 1, (nc, me))
    edge_sign = np.where(rs.random_sample((nc, me)) < 0.5, -1, 1).astype(np.int32)
    thickness = uniform(1.0, 2500.0, nc)
    thickness[rs.random_sample(nc) < 0.25] = 0.0
    thin = rs.random_sample(nc) < 0.1
    thickness[thin] = uniform(0.05, 1.0, int(thin.sum()))
    fractions = uniform(0.5, 1.5, nl).astype(np.float64)
    fractions = (fractions / fractions.sum()).astype(real)
    interfaces = np.concatenate([[0.0], np.cumsum(fractions.astype(np.float64))]).astype(real)
    interfaces[-1] = 1.0
    centres = (0.5 * (interfaces[:-1].astype(np.float64) + interfaces[1:])).astype(real)
    year = 365.0 * 86400.0
    velocity = uniform(-300.0 / year, 300.0 / year, (ne, nl))
    zeros = np.flatnonzero(rs.random_sample(ne) < 0.08)
    velocity[zeros, 0] = 0.0
    velocity[zeros[::2], 0] = -0.0
    sfc = uniform(-3.0e-4, 3.0e-4, nc)
    sfc[thin] = uniform(-5.0e-3, 5.0e-3, int(thin.sum()))
    case = {
        "name": name, "precision": precision, "nCells": nc, "nEdges": ne, "nVertices": nv, "nVertLevels": nl,
        "maxEdges": me,
        "nCellsSolve": nc - 4, "velocitySolver": solver, "thermalSolver": thermal,
        "advectTracers": bool(advect_tracers), "thicknessAdvection": advection,
        "dt": real(dt), "iceDensity": real(910.0), "oceanDensity": real(1028.0),
        "seaLevel": real(0.0), "dynamicThickness": real(10.0),
        "nEdgesOnCell": n_edges_on_cell, "cellsOnCell": cells_on_cell,
        "cellsOnVertex": ints(1, nc + 1, (nv, 3)), "cellsOnEdge": ints(1, nc + 1, (ne, 2)),
        "edgesOnCell": edges_on_cell, "edgeSignOnCell": edge_sign,
        "cellMask": ints(0, 512, nc), "vertexMask": ints(0, 512, nv), "edgeMask": ints(0, 512, ne),
        "layerThicknessFractions": fractions, "layerCenterSigma": centres, "layerInterfaceSigma": interfaces,
        "dvEdge": uniform(5.0e3, 2.0e4, ne), "dcEdge": uniform(5.0e3, 2.0e4, ne),
        "areaCell": uniform(2.0e7, 3.0e8, nc),
        "layerNormalVelocity": velocity, "layerThicknessEdge": uniform(0.0, 300.0, (ne, nl)),
        "bedTopography": uniform(-1200.0, 900.0, nc),
        "groundedBasalMassBal": uniform(-1.0e-4, 1.0e-4, nc), "floatingBasalMassBal": uniform(-1.0e-3, 2.0e-4, nc),
        "surfaceAirTemperature": uniform(230.0, 285.0, nc), "basalTemperature": uniform(250.0, 273.0, nc),
        "thickness": thickness, "sfcMassBal": sfc, "basalMassBal": uniform(-1.0, 1.0, nc),
        "dynamicThickening": uniform(-1.0, 1.0, nc), "layerThickness": uniform(-1.0, 1.0, (nc, nl)),
        "temperature": uniform(240.0, 273.0, (nc, nl)), "waterfrac": uniform(0.0, 0.02, (nc, nl)),
        "enthalpy": uniform(-6.0e8, -4.0e8, (nc, nl)),
        "workLevelCell": uniform(-1.0e3, 1.0e3, (nc, nl)), "workTracerCell": uniform(-1.0e3, 1.0e3, nc),
        "workTracerCell2": uniform(-1.0e3, 1.0e3, nc), "workTracerLevelCell": uniform(-1.0e3, 1.0e3, (nc, nl)),
        "workTracerLevelCell2": uniform(-1.0e3, 1.0e3, (nc, nl)),
    }
    return case


def cases(precision: str = "r4") -> list[dict]:
    return [make_case(*row, precision=precision) for row in CASES]


def _real(case: dict) -> np.dtype:
    return np.dtype(PRECISIONS[case["precision"]]).newbyteorder("<")


def input_bytes(case: dict) -> bytes:
    """run_advection.F90's input stream for ``case``."""
    head = np.array([case["nCells"], case["nEdges"], case["nVertices"], case["nVertLevels"], case["maxEdges"],
                     case["nCellsSolve"], int(case["velocitySolver"] == "sia"), THERMAL[case["thermalSolver"]],
                     int(case["advectTracers"]), int(case["thicknessAdvection"] == "fo")], "<i4")
    scalars = np.array([case["dt"], case["iceDensity"], case["oceanDensity"], case["seaLevel"],
                        case["dynamicThickness"]], _real(case))
    parts = [head.tobytes(), scalars.tobytes()]
    parts += [np.ascontiguousarray(case[name], "<i4").tobytes() for name in INT_FIELDS]
    parts += [np.ascontiguousarray(case[name], _real(case)).tobytes() for name in REAL_FIELDS + REAL_STATE + SCRATCH]
    return b"".join(parts)


def input_digest(case: dict) -> str:
    return hashlib.sha256(input_bytes(case)).hexdigest()


def _shape(name: str, case: dict) -> tuple[int, ...]:
    if name == "vertexMask":
        return (case["nVertices"],)
    if name == "edgeMask":
        return (case["nEdges"],)
    if name in ("layerThickness", "temperature", "waterfrac", "enthalpy"):
        return (case["nCells"], case["nVertLevels"])
    return (case["nCells"],)


def read_outputs(raw: bytes, case: dict) -> dict[str, np.ndarray]:
    """run_advection.F90's output stream -> named arrays."""
    head = np.frombuffer(raw, "<i4", 3, 0)
    out = {"errInit": head[0:1].astype(np.int32), "errAdvect": head[1:2].astype(np.int32),
           "logErrors": head[2:3].astype(np.int32)}
    off = 12
    for name, dtype in [(n, np.dtype("<i4")) for n in INT_OUTPUTS] + [(n, _real(case)) for n in REAL_STATE]:
        shape = _shape(name, case)
        n = int(np.prod(shape))
        out[name] = np.frombuffer(raw, dtype, n, off).reshape(shape).astype(dtype.newbyteorder("="))
        off += dtype.itemsize * n
    if off != len(raw):
        raise ValueError(f"oracle output: {len(raw) - off} trailing bytes")
    return out


def load_fixture(precision: str = "r4") -> dict[str, dict[str, np.ndarray]]:
    """The committed oracle outputs at ``precision``, keyed by case then field."""
    with np.load(FIXTURE_DIR / f"oracle-{precision}.npz") as data:
        out: dict[str, dict[str, np.ndarray]] = {}
        for key in data.files:
            case, field = key.split("/", 1)
            out.setdefault(case, {})[field] = data[key]
    return out


def word_mismatches(a: np.ndarray, b: np.ndarray) -> int:
    """Words that differ as bit patterns (so -0.0 != 0.0)."""
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    if a.shape != b.shape or a.dtype.itemsize != b.dtype.itemsize:
        return int(max(a.size, b.size))
    word = np.dtype(f"u{a.dtype.itemsize}")
    return int(np.count_nonzero(a.view(word) != b.view(word)))


def run_gpu(case: dict) -> dict[str, np.ndarray]:
    """Run the GPU port (woof.hex.mpas_ice.advection_v842) on ``case`` as
    run_advection.F90 drives the Fortran; the same named outputs, without
    the error words (the port raises instead)."""
    import cupy as cp
    from woof.hex.mpas_ice import advection_v842 as adv

    dev = {name: cp.asarray(np.ascontiguousarray(case[name])) for name in INT_FIELDS + REAL_FIELDS + REAL_STATE}
    mesh = dict(dev, nCellsSolve=case["nCellsSolve"])
    adv.advect_thickness_tracers(
        mesh=mesh, geometry=dev, velocity=dev, thermal=dev, dt=case["dt"], advect_tracers=case["advectTracers"],
        config_thickness_advection=case["thicknessAdvection"], config_thermal_solver=case["thermalSolver"],
        config_ice_density=case["iceDensity"], config_ocean_density=case["oceanDensity"],
        config_sea_level=case["seaLevel"], config_dynamic_thickness=case["dynamicThickness"],
        config_velocity_solver=case["velocitySolver"])
    return {name: cp.asnumpy(dev[name]) for name in INT_OUTPUTS + REAL_STATE}


def measure(reference: dict | None = None, precision: str = "r4") -> dict[str, dict[str, int]]:
    """Per case and field, the GPU words at ``precision`` that differ from the oracle's."""
    reference = load_fixture(precision) if reference is None else reference
    report: dict[str, dict[str, int]] = {}
    for case in cases(precision):
        got = run_gpu(case)
        want = reference[case["name"]]
        report[case["name"]] = {name: word_mismatches(got[name], want[name]) for name in got}
    return report
