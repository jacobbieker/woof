"""Cases and the stream layout for the MPAS-Ocean full-tendency oracle.

``tools/mpas_ocean_oracle/run_tendency.F90`` drives MPAS-Model v8.4.2's
``ocn_tend_vel`` and ``ocn_tend_thick`` (``mpas_ocn_tendency.F``, compiled
verbatim with every term they call), and ``ocn_vel_forcing_tend`` and
``ocn_thick_surface_flux_tend`` on their own, at RKIND 4 and 8. This module
writes the driver's input stream, reads its output stream, builds the cases
and runs the GPU port the same way.

A case starts from a :mod:`woof.verify.mpas_ocean_pgrad_hmix_oracle` case
(the pressure-gradient and hmix fields, drawn from the same seed) and draws the
rest from a second ``RandomState`` with the same discipline (``randint`` and
``random_sample`` only; reals in float64, rounded to float32 for r4).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from woof.verify import mpas_ocean_pgrad_hmix_oracle as _ph
from woof.verify import mpas_ocean_tend_oracle as _tend
from woof.verify.mpas_ocean_tend_oracle import PRECISIONS, bit_mismatches

FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data"
               / "oracles" / "mpas_ocean_tendency")
#: The RKIND 8 run of the bulk wind stress, which needs glibc's binary64
#: cos/sin: its words are in oracle-r8-glibc239.npz, and the port runs it only
#: under the opt-in LGPL transcription (WOOF_GLIBC239_LIBM64=1)
R8_GLIBC239 = "r8-glibc239"
#: make_fixture's extra precisions, each with the binary it runs
EXTRA_PRECISIONS = {R8_GLIBC239: "r8"}


def _base(precision: str) -> str:
    return EXTRA_PRECISIONS.get(precision, precision)
#: run_tendency.F90's flag and real config order
FLAGS = ("config_disable_vel_all_tend", "config_disable_thick_all_tend", "config_use_mom_del2",
         "config_use_Leith_del2", "config_use_mom_del4", "config_use_bulk_wind_stress",
         "config_use_bulk_thickness_flux", "config_disable_vel_surface_stress", "config_use_explicit_bottom_drag",
         "config_Rayleigh_friction", "config_Rayleigh_bottom_friction", "config_disable_thick_sflux",
         "config_disable_vel_hmix", "config_disable_vel_pgrad")
REALS = ("config_density0", "config_common_level_weight", "config_mom_del2", "config_mom_del4",
         "config_mom_del4_div_factor", "config_Leith_parameter", "config_Leith_dx", "config_Leith_visc2_max",
         "config_explicit_bottom_drag_coeff", "config_Rayleigh_damping_coeff",
         "config_Rayleigh_bottom_damping_coeff")
INT_FIELDS = ("maxLevelEdgeTop", "maxLevelEdgeBot", "maxLevelCell", "maxLevelVertexTop", "nEdgesOnEdge",
              "nEdgesOnCell", "cellsOnEdge", "verticesOnEdge", "edgesOnEdge", "edgeMask", "edgesOnVertex",
              "edgesOnCell", "edgeSignOnVertex", "edgeSignOnCell")
MESH_REAL_FIELDS = ("dcEdge", "dvEdge", "areaCell", "areaTriangle", "weightsOnEdge", "meshScalingDel2",
                    "meshScalingDel4", "meshScaling", "angleEdge")
STATE_FIELDS = ("normalVelocity", "activeTracers")
DIAG_FIELDS = ("kineticEnergyCell", "layerThicknessEdge", "vertAleTransportTop", "zMid", "relativeVorticity",
               "normalizedRelativeVorticityEdge", "normalizedPlanetaryVorticityEdge", "divergence",
               "montgomeryPotential", "pressure", "density", "potentialDensity", "tangentialVelocity",
               "inSituThermalExpansionCoeff", "inSituSalineContractionCoeff", "normalTransportVelocity",
               "surfaceFluxAttenuationCoefficient")
FORCING_FIELDS = ("windStressZonal", "windStressMeridional", "evaporationFlux", "snowFlux", "seaIceFreshWaterFlux",
                  "riverRunoffFlux", "iceRunoffFlux", "rainFlux", "fractionAbsorbed", "fractionAbsorbedRunoff",
                  "surfaceStressIn", "fluxIn", "runoffIn")
OUT_FIELDS = ("tendNormalVelocity", "surfaceStress", "surfaceStressMagnitude", "viscosity", "tendLayerThickness",
              "surfaceThicknessFlux", "surfaceThicknessFluxRunoff", "tendVelForcing", "tendThickSflux")
_CELL_OUT = {"surfaceStressMagnitude", "tendLayerThickness", "surfaceThicknessFlux", "surfaceThicknessFluxRunoff",
             "tendThickSflux"}
_SCALAR_OUT = {"surfaceStress", "surfaceStressMagnitude", "surfaceThicknessFlux", "surfaceThicknessFluxRunoff"}

_BASE = {"config_mom_del4": 5.0e9, "config_Leith_visc2_max": 25.0, "config_Rayleigh_damping_coeff": 1.0e-4,
         "config_explicit_bottom_drag_coeff": 1.0e-3}

#: (name, seed, nCells, nVertLevels, RK4, every edge wet, precisions, configs)
CASES = (
    ("zmid_del2_split", 20261130, 40, 20, False, False, ("r4", "r8"),
     {"config_pressure_gradient_type": "pressure_and_zmid", "config_use_mom_del2": True,
      "config_use_bulk_thickness_flux": True}),
    # the bulk wind stress needs glibc cosf/sinf: RKIND 4 only
    ("jts_rk4_bulk_drag", 20261131, 36, 24, True, True, ("r4", R8_GLIBC239),
     {"config_pressure_gradient_type": "Jacobian_from_TS", "config_use_mom_del4": True,
      "config_use_Leith_del2": True, "config_use_bulk_wind_stress": True, "config_use_bulk_thickness_flux": True,
      "config_use_explicit_bottom_drag": True, "config_Rayleigh_friction": True,
      "config_Rayleigh_bottom_friction": True}),
    ("jdens_drag_rayleigh", 20261132, 40, 18, False, True, ("r4", "r8"),
     {"config_pressure_gradient_type": "Jacobian_from_density", "config_use_mom_del2": True,
      "config_use_mom_del4": True, "config_use_bulk_thickness_flux": True,
      "config_use_explicit_bottom_drag": True, "config_Rayleigh_friction": True,
      "config_Rayleigh_bottom_friction": True, "config_density0": 1027.5}),
    ("all_tend_off", 20261133, 32, 16, False, True, ("r4", "r8"),
     {"config_disable_vel_all_tend": True, "config_disable_thick_all_tend": True,
      "config_use_explicit_bottom_drag": True, "config_use_bulk_thickness_flux": True}),
    ("mont_forcing_off", 20261134, 32, 22, True, False, ("r4", "r8"),
     {"config_pressure_gradient_type": "MontgomeryPotential", "config_disable_vel_hmix": True,
      "config_disable_vel_pgrad": True,
      "config_use_mom_del2": True, "config_disable_vel_surface_stress": True,
      "config_disable_thick_sflux": True, "config_Rayleigh_friction": True}),
)


def make_case(name, seed, n_cells, n_levels, rk4, all_wet, precisions, configs) -> dict:
    """One deterministic case: a pressure-gradient/hmix case of the same seed
    plus the Coriolis, vertical advection and forcing fields.

    ``all_wet`` lifts every land edge to one level (with its ``edgeMask``
    word), which the explicit bottom drag and Rayleigh bottom friction need
    (forcing_v842 refuses them otherwise). Beyond the pressure-gradient case's
    coverage, a case contains attenuation depths that put the transmission
    under its ``exp(-100)`` floor, absorbed fractions whose column sum is on
    both sides of 1, and both signs of every freshwater flux that has them.
    """
    from woof.hex.mpas_ocean.forcing_v842 import DEFAULT_CONFIGS

    pgrad_type = configs.get("config_pressure_gradient_type", "pressure_and_zmid")
    case = _ph.make_case(name, seed, n_cells, n_levels, pgrad_type, (1, 2), {}, {})
    case["activeTracers"] = case.pop("tracers")
    for key in ("flags", "reals", "pgradType", "tendPgrad", "tendHmix"):
        case.pop(key)
    rs = np.random.RandomState(seed + 7919)
    nc, ne, nl = case["nCells"], case["nEdges"], n_levels
    me2 = 14

    def uniform(lo, hi, shape):
        return lo + (hi - lo) * rs.random_sample(shape)

    if all_wet:
        land = case["maxLevelEdgeTop"] == 0
        case["maxLevelEdgeTop"][land] = 1
        case["edgeMask"][land, 0] = 1
    top = case["maxLevelEdgeTop"]
    case["maxLevelEdgeBot"] = np.minimum(nl, top + rs.randint(0, 4, ne)).astype(np.int32)
    case["nEdgesOnEdge"] = rs.randint(8, me2 + 1, ne).astype(np.int32)
    edges_on_edge = np.full((ne, me2), ne + 9, np.int32)
    for e in range(ne):
        edges_on_edge[e, :case["nEdgesOnEdge"][e]] = rs.randint(1, ne + 1, case["nEdgesOnEdge"][e])
    case["edgesOnEdge"] = edges_on_edge
    case["maxEdges2"] = me2
    case["rk4"] = bool(rk4)
    case["precisions"] = tuple(precisions)
    case["configs"] = {**DEFAULT_CONFIGS, **_BASE, **configs,
                       "config_time_integrator": "RK4" if rk4 else "split_explicit"}
    case.update({
        "weightsOnEdge": uniform(-0.5, 0.5, (ne, me2)),
        "angleEdge": uniform(-np.pi, np.pi, ne),
        "kineticEnergyCell": uniform(0.0, 0.8, (nc, nl)),
        "layerThicknessEdge": uniform(1.0, 250.0, (ne, nl)),
        "vertAleTransportTop": uniform(-2.0e-3, 2.0e-3, (nc, nl + 1)),
        "normalizedRelativeVorticityEdge": uniform(-2.0e-7, 2.0e-7, (ne, nl)),
        "normalizedPlanetaryVorticityEdge": uniform(-1.5e-6, 1.5e-6, (ne, nl)),
        "normalTransportVelocity": uniform(-1.2, 1.2, (ne, nl)),
        "surfaceFluxAttenuationCoefficient": uniform(0.5, 20.0, nc),
        "windStressZonal": uniform(-0.3, 0.3, nc),
        "windStressMeridional": uniform(-0.3, 0.3, nc),
        "evaporationFlux": uniform(-1.0e-4, 0.0, nc),
        "snowFlux": uniform(0.0, 2.0e-5, nc),
        "seaIceFreshWaterFlux": uniform(-5.0e-5, 5.0e-5, nc),
        "riverRunoffFlux": uniform(0.0, 1.0e-4, nc),
        "iceRunoffFlux": uniform(0.0, 1.0e-5, nc),
        "rainFlux": uniform(0.0, 1.0e-4, nc),
        "fractionAbsorbed": uniform(0.0, 0.12, (nc, nl)),
        "fractionAbsorbedRunoff": uniform(0.0, 0.12, (nc, nl)),
        "surfaceStressIn": uniform(-0.2, 0.2, ne),
        "fluxIn": uniform(-1.0e-7, 1.0e-7, nc),
        "runoffIn": uniform(0.0, 1.0e-7, nc),
        "tendNormalVelocity": uniform(-1.0e-4, 1.0e-4, (ne, nl)),
        "surfaceStress": uniform(-1.0, 1.0, ne),
        "surfaceStressMagnitude": uniform(-1.0, 1.0, nc),
        "tendLayerThickness": uniform(-1.0e-3, 1.0e-3, (nc, nl)),
        "surfaceThicknessFlux": uniform(-1.0, 1.0, nc),
        "surfaceThicknessFluxRunoff": uniform(-1.0, 1.0, nc),
        "tendVelForcing": uniform(-1.0e-6, 1.0e-6, (ne, nl)),
        "tendThickSflux": uniform(-1.0e-8, 1.0e-8, (nc, nl)),
    })
    return case


def cases() -> list[dict]:
    return [make_case(*row) for row in CASES]


def _shape(name: str, case: dict) -> tuple[int, ...]:
    n = case["nCells"] if name in _CELL_OUT else case["nEdges"]
    return (n,) if name in _SCALAR_OUT else (n, case["nVertLevels"])


def input_bytes(case: dict, precision: str) -> bytes:
    """run_tendency.F90's input stream for ``case`` at ``precision``."""
    real = np.dtype(PRECISIONS[_base(precision)]).newbyteorder("<")
    c = case["configs"]
    head = np.array([case["nCells"], case["nEdges"], case["nVertices"], case["maxEdges"], case["maxEdges2"],
                     case["vertexDegree"], case["nVertLevels"], case["nTracers"], case["indexT"], case["indexS"],
                     _ph.PGRAD_TYPES.index(c["config_pressure_gradient_type"]) + 1, int(case["rk4"])], "<i4")
    flags = np.array([int(c[k]) for k in FLAGS], "<i4")
    reals = np.array([c[k] for k in REALS]).astype(real)
    parts = [head.tobytes(), flags.tobytes(), reals.tobytes()]
    for name in INT_FIELDS:
        parts.append(np.ascontiguousarray(case[name], "<i4").tobytes())
    for name in MESH_REAL_FIELDS + STATE_FIELDS + DIAG_FIELDS + FORCING_FIELDS + OUT_FIELDS:
        parts.append(np.ascontiguousarray(case[name]).astype(real).tobytes())
    return b"".join(parts)


def input_digest(case: dict, precision: str) -> str:
    return hashlib.sha256(input_bytes(case, precision)).hexdigest()


def read_outputs(raw: bytes, case: dict, precision: str) -> dict[str, np.ndarray]:
    real = np.dtype(PRECISIONS[_base(precision)]).newbyteorder("<")
    out, off = {}, 0
    for name in OUT_FIELDS:
        shape = _shape(name, case)
        n = int(np.prod(shape))
        out[name] = np.frombuffer(raw, real, n, off).reshape(shape).astype(PRECISIONS[_base(precision)])
        off += real.itemsize * n
    if off != len(raw):
        raise ValueError(f"oracle output: {len(raw) - off} trailing bytes")
    return out


def load_fixture(precision: str) -> dict[str, dict[str, np.ndarray]]:
    return _tend.load_fixture(precision, FIXTURE_DIR)


def run_gpu(case: dict, precision: str) -> dict[str, np.ndarray]:
    """Run woof.hex.mpas_ocean.forcing_v842 on ``case`` as run_tendency.F90
    drives the Fortran."""
    import cupy as cp
    from woof.hex.mpas_ocean import forcing_v842 as fv

    dtype = PRECISIONS[_base(precision)]
    dev = {name: cp.asarray(np.ascontiguousarray(case[name], np.int32)) for name in INT_FIELDS}
    for name in MESH_REAL_FIELDS + STATE_FIELDS + DIAG_FIELDS + FORCING_FIELDS + OUT_FIELDS:
        dev[name] = cp.asarray(np.ascontiguousarray(case[name]).astype(dtype))
    diagnostics = {name: dev[name] for name in DIAG_FIELDS}
    diagnostics["viscosity"] = dev["viscosity"]
    forcing = {name: dev[name] for name in FORCING_FIELDS + OUT_FIELDS}
    configs = case["configs"]
    fv.tend_vel(mesh=dev, state=dev, diagnostics=diagnostics, forcing=forcing, tend=dev["tendNormalVelocity"],
                configs=configs, index_temperature=case["indexT"], index_salinity=case["indexS"])
    fv.tend_thick(mesh=dev, diagnostics=diagnostics, forcing=forcing, tend=dev["tendLayerThickness"],
                  configs=configs)
    fv.vel_forcing_tend(mesh=dev, normal_velocity=dev["normalVelocity"],
                        surface_flux_attenuation_coefficient=dev["surfaceFluxAttenuationCoefficient"],
                        surface_stress=dev["surfaceStressIn"], kinetic_energy_cell=dev["kineticEnergyCell"],
                        layer_thickness_edge=dev["layerThicknessEdge"], tend=dev["tendVelForcing"],
                        configs=configs)
    fv.thick_surface_flux_tend(mesh=dev, transmission_coefficients=dev["fractionAbsorbed"],
                               transmission_coefficients_runoff=dev["fractionAbsorbedRunoff"],
                               surface_thickness_flux=dev["fluxIn"], surface_thickness_flux_runoff=dev["runoffIn"],
                               tend=dev["tendThickSflux"], configs=configs)
    return {name: cp.asnumpy(dev[name]) for name in OUT_FIELDS}


def measure(precision: str, reference: dict | None = None) -> dict[str, dict[str, int]]:
    reference = load_fixture(precision) if reference is None else reference
    report: dict[str, dict[str, int]] = {}
    for case in cases():
        if precision not in case["precisions"]:
            continue
        got = run_gpu(case, precision)
        want = reference[case["name"]]
        report[case["name"]] = {name: bit_mismatches(got[name], want[name]) for name in want}
    return report
