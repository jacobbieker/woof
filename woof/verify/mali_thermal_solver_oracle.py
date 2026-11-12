"""Cases and the stream layout for the MALI thermal solver oracle.

``tools/mali_oracle`` compiles MALI v8.4.2's land-ice modules verbatim with
gfortran -O0 -ffp-contract=off at RKIND 4 and 8, and ``run_thermal_solver.F90``
makes a restart's ``li_thermal_init``, then ``li_basal_melt_floating_ice``
and ``li_thermal_solver``, as ``li_time_integration_fe`` does. This module
writes that driver's input stream, reads its output, and builds the cases.

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

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mali_thermal_solver"
PRECISIONS = {"r4": np.float32, "r8": np.float64}
THERMAL = {"none": 0, "temperature": 1, "enthalpy": 2}
FLOAT_MELT = {"none": 0, "file": 1, "constant": 2, "mismip": 3, "seroussi": 4}

INT_FIELDS = ("nEdgesOnCell", "cellsOnCell", "cellsOnVertex", "cellsOnEdge", "edgesOnCell", "cellMask",
              "vertexMask", "edgeMask")
REAL_FIELDS = ("layerCenterSigma", "layerInterfaceSigma", "dcEdge", "dvEdge", "areaCell", "xCell", "thickness",
               "bedTopography", "upperSurface", "lowerSurface", "basalWaterThickness", "slopeEdge", "flowParamA",
               "basalSpeed", "betaSolve", "basalHeatFlux")
#: real outputs, in the order run_thermal_solver.F90 reads their prior words and writes them
REAL_STATE = ("temperature", "waterfrac", "enthalpy", "surfaceTemperature", "basalTemperature",
              "surfaceAirTemperature", "surfaceConductiveFlux", "basalConductiveFlux", "heatDissipation",
              "basalFrictionFlux", "groundedBasalMassBal", "floatingBasalMassBal")
INT_OUTPUTS = ("cellMask", "vertexMask", "edgeMask")
LEVEL_FIELDS = ("temperature", "waterfrac", "enthalpy", "heatDissipation", "flowParamA")
SCALARS = ("deltat", "daysSinceStart", "iceDensity", "oceanDensity", "seaLevel", "dynamicThickness",
           "thermalThickness", "flowLawExponent", "airValue", "lapseRate", "bmltFlux", "bmltXlimit",
           "serAmplitude", "serPeriod", "serPhase")

#: (name, seed, nCells, nVertLevels, velocity solver, thermal solver, air source, floating melt, calculate bmb)
CASES = (
    ("temperature_sia_constant", 20261141, 120, 10, "sia", "temperature", "file", "constant", True),
    ("enthalpy_sia_mismip", 20261142, 120, 10, "sia", "enthalpy", "lapse", "mismip", True),
    ("temperature_none_seroussi", 20261143, 96, 8, "none", "temperature", "lapse", "seroussi", False),
    ("enthalpy_none_file", 20261144, 80, 6, "none", "enthalpy", "file", "file", True),
    ("no_thermal", 20261145, 64, 5, "sia", "none", "file", "none", True),
    # many floating cells, so MISMIP's tanh is graded on a thousand arguments
    ("no_thermal_mismip", 20261146, 4096, 3, "none", "none", "file", "mismip", False),
)


def _pmp_column(rhoi, thickness, sigma):
    """pressure_melting_point_column at the arguments' precision, the Fortran's order (deg C)."""
    f = np.result_type(rhoi, thickness, sigma).type
    return -(((f(9.7456e-8) * rhoi) * f(9.80616)) * thickness) * sigma


def make_case(name: str, seed: int, n_cells: int, n_levels: int, solver: str, thermal: str, air: str,
              float_melt: str, calculate_bmb: bool, precision: str = "r4") -> dict:
    """One deterministic case: grounded and floating ice, thin ice and open
    ocean on a random mesh, a month's step.

    - ice up to 2500 m, a quarter of the cells ice-free and a tenth thinner
      than the 1 m thermal thickness, over beds from -1200 m to 900 m, so
      the masks find floating and grounded ice and open ocean;
    - column temperatures from 240 K to 273 K, so layers start above the
      pressure melting point (internal melt), with a sixth of the columns'
      basal temperatures and a sixth of their bottom layers set to the
      pressure melting point (the melting-bed boundary rows) and water
      fractions up to 0.03 (drainage above 0.01);
    - geothermal and frictional heating, basal water on half the cells, so
      the bed both melts and freezes on;
    - air temperatures on both sides of 273.15 K.

    nCellsSolve stops short of nCells, so the halo cells' words come from the
    whole-array work only. The scratch arrays are random: the model leaves
    them uninitialised.
    """
    rs = np.random.RandomState(seed)
    real = PRECISIONS[precision]
    nc, nl = n_cells, n_levels
    ne, nv, me = 3 * nc, 2 * nc, 7
    f = real

    def uniform(lo, hi, shape):
        return (lo + (hi - lo) * rs.random_sample(shape)).astype(real)

    def ints(lo, hi, shape):
        return (lo + np.floor((hi - lo) * rs.random_sample(shape))).astype(np.int32)

    rhoi = f(910.0)
    thickness = uniform(1.0, 2500.0, nc)
    thickness[rs.random_sample(nc) < 0.25] = 0.0
    thin = rs.random_sample(nc) < 0.1
    thickness[thin] = uniform(0.05, 1.0, int(thin.sum()))
    fractions = uniform(0.5, 1.5, nl).astype(np.float64)
    fractions = fractions / fractions.sum()
    interfaces = np.concatenate([[0.0], np.cumsum(fractions)]).astype(real)
    interfaces[-1] = 1.0
    centres = (0.5 * (interfaces[:-1].astype(np.float64) + interfaces[1:])).astype(real)
    temperature = uniform(240.0, 273.0, (nc, nl))
    basal_temperature = uniform(250.0, 273.0, nc)
    at_pmp = rs.random_sample(nc) < 1.0 / 6.0
    basal_temperature[at_pmp] = (-((f(9.7456e-8) * rhoi) * f(9.80616)) * thickness[at_pmp]) + f(273.15)
    bottom_at_pmp = rs.random_sample(nc) < 1.0 / 6.0
    temperature[bottom_at_pmp, -1] = _pmp_column(rhoi, thickness[bottom_at_pmp], centres[-1]) + f(273.15)
    water = uniform(0.0, 1.0, nc)
    water[rs.random_sample(nc) < 0.5] = 0.0
    year = 365.0 * 86400.0
    case = {
        "name": name, "precision": precision, "nCells": nc, "nEdges": ne, "nVertices": nv, "nVertLevels": nl,
        "maxEdges": me,
        "nCellsSolve": nc - 4, "velocitySolver": solver, "thermalSolver": thermal, "airSource": air,
        "floatMelt": float_melt, "calculateBmb": bool(calculate_bmb),
        "deltat": f(30.0 * 86400.0), "daysSinceStart": f(400.5), "iceDensity": rhoi, "oceanDensity": f(1028.0),
        "seaLevel": f(0.0), "dynamicThickness": f(10.0), "thermalThickness": f(1.0), "flowLawExponent": f(3.0),
        "airValue": f(268.0), "lapseRate": f(0.006), "bmltFlux": f(100.0), "bmltXlimit": f(1.0e5),
        "serAmplitude": f(50.0), "serPeriod": f(2.0), "serPhase": f(0.7),
        "nEdgesOnCell": np.array([5, 6, 6, 6, 7], np.int32)[ints(0, 5, nc)],
        "cellsOnCell": ints(1, nc + 1, (nc, me)), "cellsOnVertex": ints(1, nc + 1, (nv, 3)),
        "cellsOnEdge": ints(1, nc + 1, (ne, 2)), "edgesOnCell": ints(1, ne + 1, (nc, me)),
        "cellMask": ints(0, 512, nc), "vertexMask": ints(0, 512, nv), "edgeMask": ints(0, 512, ne),
        "layerCenterSigma": centres, "layerInterfaceSigma": interfaces,
        "dcEdge": uniform(5.0e3, 2.0e4, ne), "dvEdge": uniform(5.0e3, 2.0e4, ne),
        "areaCell": uniform(2.0e7, 3.0e8, nc), "xCell": uniform(-5.0e5, 5.0e5, nc), "thickness": thickness,
        "bedTopography": uniform(-1200.0, 900.0, nc), "upperSurface": uniform(0.0, 3000.0, nc),
        "lowerSurface": uniform(-900.0, 0.0, nc), "basalWaterThickness": water,
        "slopeEdge": uniform(-0.05, 0.05, ne), "flowParamA": uniform(1.0e-25, 1.0e-23, (nc, nl)),
        "basalSpeed": uniform(0.0, 300.0 / year, nc), "betaSolve": uniform(0.0, 1.0e3, nc),
        "basalHeatFlux": uniform(0.03, 0.12, nc),
        "temperature": temperature, "waterfrac": uniform(0.0, 0.03, (nc, nl)),
        "enthalpy": uniform(-6.0e8, -4.0e8, (nc, nl)), "surfaceTemperature": uniform(230.0, 273.0, nc),
        "basalTemperature": basal_temperature, "surfaceAirTemperature": uniform(230.0, 285.0, nc),
        "surfaceConductiveFlux": uniform(-1.0, 1.0, nc), "basalConductiveFlux": uniform(-1.0, 1.0, nc),
        "heatDissipation": uniform(0.0, 1.0e-8, (nc, nl)), "basalFrictionFlux": uniform(-1.0, 1.0, nc),
        "groundedBasalMassBal": uniform(-1.0, 1.0, nc), "floatingBasalMassBal": uniform(-1.0, 1.0, nc),
        "workLevelEdge": uniform(-1.0e3, 1.0e3, (ne, nl)), "iceCellMask": ints(-5, 6, nc),
    }
    return case


#: At RKIND 8 the port refuses 'seroussi' (glibc's binary64 sin has no matched
#: transcription), so that case runs 'none'
R8_FLOAT_MELT = {"seroussi": "none"}
#: The fixture "precision" of the RKIND 8 cases run under the opt-in glibc
#: 2.39 sin transcription (WOOF_GLIBC239_LIBM64=1): 'seroussi' as written,
#: its words in oracle-r8-glibc239.npz
R8_GLIBC239 = "r8-glibc239"


def cases(precision: str = "r4") -> list[dict]:
    """The cases at 'r4', 'r8' or :data:`R8_GLIBC239` (the 'seroussi' cases at RKIND 8 under the opt-in)."""
    if precision == R8_GLIBC239:
        return [make_case(*row, precision="r8") for row in CASES if row[7] in R8_FLOAT_MELT]
    rows = CASES
    if precision == "r8":
        rows = [(row[0].replace(row[7], R8_FLOAT_MELT[row[7]]),) + row[1:7] + (R8_FLOAT_MELT[row[7]],) + row[8:]
                if row[7] in R8_FLOAT_MELT else row for row in CASES]
    return [make_case(*row, precision=precision) for row in rows]


def _real(case: dict) -> np.dtype:
    return np.dtype(PRECISIONS[case["precision"]]).newbyteorder("<")


def input_bytes(case: dict) -> bytes:
    """run_thermal_solver.F90's input stream for ``case``."""
    head = np.array([case["nCells"], case["nEdges"], case["nVertices"], case["nVertLevels"], case["maxEdges"],
                     case["nCellsSolve"], int(case["velocitySolver"] == "sia"), THERMAL[case["thermalSolver"]],
                     int(case["airSource"] == "lapse"), FLOAT_MELT[case["floatMelt"]], int(case["calculateBmb"])],
                    "<i4")
    scalars = np.array([case[name] for name in SCALARS], _real(case))
    parts = [head.tobytes(), scalars.tobytes()]
    parts += [np.ascontiguousarray(case[name], "<i4").tobytes() for name in INT_FIELDS]
    parts += [np.ascontiguousarray(case[name], _real(case)).tobytes() for name in REAL_FIELDS + REAL_STATE]
    parts += [np.ascontiguousarray(case["workLevelEdge"], _real(case)).tobytes(),
              np.ascontiguousarray(case["iceCellMask"], "<i4").tobytes()]
    return b"".join(parts)


def input_digest(case: dict) -> str:
    return hashlib.sha256(input_bytes(case)).hexdigest()


def _shape(name: str, case: dict) -> tuple[int, ...]:
    if name == "vertexMask":
        return (case["nVertices"],)
    if name == "edgeMask":
        return (case["nEdges"],)
    if name in LEVEL_FIELDS:
        return (case["nCells"], case["nVertLevels"])
    return (case["nCells"],)


def read_outputs(raw: bytes, case: dict) -> dict[str, np.ndarray]:
    """run_thermal_solver.F90's output stream -> named arrays."""
    head = np.frombuffer(raw, "<i4", 4, 0)
    out = {"errInit": head[0:1].astype(np.int32), "errFloat": head[1:2].astype(np.int32),
           "errSolver": head[2:3].astype(np.int32), "logErrors": head[3:4].astype(np.int32)}
    off = 16
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
    """Run the GPU port (woof.hex.mpas_ice.thermal_solver_v842) on ``case`` as
    run_thermal_solver.F90 drives the Fortran; the same named outputs,
    without the error words (the port raises instead)."""
    import cupy as cp
    from woof.hex.mpas_ice import thermal_init_v842, thermal_solver_v842 as ts

    dev = {name: cp.asarray(np.ascontiguousarray(case[name])) for name in INT_FIELDS + REAL_FIELDS + REAL_STATE}
    mesh = dict(dev, nCellsSolve=case["nCellsSolve"])
    common = {"config_ice_density": case["iceDensity"], "config_ocean_density": case["oceanDensity"],
              "config_sea_level": case["seaLevel"], "config_dynamic_thickness": case["dynamicThickness"],
              "config_velocity_solver": case["velocitySolver"]}
    # li_thermal_init on a restart: with 'lapse', the surface air temperature
    thermal_init_v842.thermal_init(
        thermal=dev, thickness=dev["thickness"], upper_surface=dev["upperSurface"],
        layer_center_sigma=dev["layerCenterSigma"], n_cells_solve=case["nCellsSolve"],
        config_ice_density=case["iceDensity"], config_temperature_init="file",
        config_surface_air_temperature_source=case["airSource"],
        config_surface_air_temperature_value=case["airValue"],
        config_surface_air_temperature_lapse_rate=case["lapseRate"], config_basal_heat_flux_source="file",
        config_basal_heat_flux_value=0.0, config_do_restart=True)
    ts.basal_melt_floating_ice(
        mesh=mesh, geometry=dev, config_basal_mass_bal_float=case["floatMelt"],
        config_thermal_thickness=case["thermalThickness"], config_bmlt_float_flux=case["bmltFlux"],
        config_bmlt_float_xlimit=case["bmltXlimit"], config_basal_mass_bal_seroussi_amplitude=case["serAmplitude"],
        config_basal_mass_bal_seroussi_period=case["serPeriod"],
        config_basal_mass_bal_seroussi_phase=case["serPhase"], days_since_start=case["daysSinceStart"], **common)
    ts.thermal_solver(
        mesh=mesh, geometry=dev, velocity=dev, thermal=dev, deltat=case["deltat"],
        config_thermal_solver=case["thermalSolver"], config_thermal_thickness=case["thermalThickness"],
        config_flowLawExponent=case["flowLawExponent"], config_surface_air_temperature_source=case["airSource"],
        config_surface_air_temperature_value=case["airValue"],
        config_surface_air_temperature_lapse_rate=case["lapseRate"],
        config_thermal_calculate_bmb=case["calculateBmb"], **common)
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
