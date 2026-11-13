"""Cases and the stream layout for the MPAS-Ocean equation-of-state oracle.

``tools/mpas_ocean_oracle/run_eos.F90`` drives MPAS-Model v8.4.2's
``ocn_equation_of_state_density`` ('linear' and 'jm', every displacement type
MPAS-Ocean calls it with, with and without the expansion coefficients) and
``ocn_freezing_temperature`` / ``ocn_freezing_temperature_salinity_deriv``
(open ocean and land-ice cavity), compiled verbatim (gfortran -O0
-ffp-contract=off) at RKIND 4 and 8. This module writes the driver's input
stream, reads its output stream, builds the cases and runs the GPU port
(:mod:`woof.hex.mpas_ocean.eos_v842`) the same way.

The conventions (Fortran-order arrays read as C-order transposes, legacy
``RandomState`` with ``randint`` and ``random_sample`` only, reals drawn in
float64 and rounded once to float32 for r4) are those of
:mod:`woof.verify.mpas_ocean_tend_oracle`, whose fixture loader and word
comparison this module reuses.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from woof.verify import mpas_ocean_tend_oracle as _tend
from woof.verify.mpas_ocean_tend_oracle import PRECISIONS, bit_mismatches

FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mpas_ocean_eos")
EOS_TYPES = ("linear", "jm")
#: The driver's calls of ocn_equation_of_state_density: (k_displaced, displacement_type,
#: with the expansion coefficients, its output fields). k_displaced None is nVertLevels.
CALLS = (
    (0, "relative", True, ("density", "thermalExpansionCoeff", "salineContractionCoeff")),
    (0, "relative", False, ("densityNoCoeffs",)),
    (1, "absolute", False, ("potentialDensity",)),
    (1, "relative", False, ("displacedDensity",)),
    (0, "surfaceDisplaced", True, ("densitySurfaceDisplaced", "thermalExpansionCoeffSurface",
                                   "salineContractionCoeffSurface")),
    (None, "absolute", False, ("densityDeepest",)),
)
FREEZING_FIELDS = ("freezingTemperatureOpen", "freezingTemperatureCavity",
                   "freezingTemperatureSalinityDerivOpen", "freezingTemperatureSalinityDerivCavity")
OUT_FIELDS = tuple(f for call in CALLS for f in call[3]) + FREEZING_FIELDS
#: The configs the driver reads, in its stream order after config_density0
REALS = ("config_eos_linear_densityref", "config_eos_linear_alpha", "config_eos_linear_beta",
         "config_eos_linear_Tref", "config_eos_linear_Sref",
         "config_open_ocean_freezing_temperature_coeff_0", "config_open_ocean_freezing_temperature_coeff_S",
         "config_open_ocean_freezing_temperature_coeff_p", "config_open_ocean_freezing_temperature_coeff_pS",
         "config_open_ocean_freezing_temperature_reference_pressure",
         "config_land_ice_cavity_freezing_temperature_coeff_0",
         "config_land_ice_cavity_freezing_temperature_coeff_S",
         "config_land_ice_cavity_freezing_temperature_coeff_p",
         "config_land_ice_cavity_freezing_temperature_coeff_pS",
         "config_land_ice_cavity_freezing_temperature_reference_pressure")
#: Registry.xml defaults (v8.4.2)
DEFAULTS = {
    "config_density0": 1026.0, "config_eos_linear_densityref": 1000.0, "config_eos_linear_alpha": 0.2,
    "config_eos_linear_beta": 0.8, "config_eos_linear_Tref": 5.0, "config_eos_linear_Sref": 35.0,
    "config_open_ocean_freezing_temperature_coeff_0": -1.8, "config_open_ocean_freezing_temperature_coeff_S": 0.0,
    "config_open_ocean_freezing_temperature_coeff_p": 0.0, "config_open_ocean_freezing_temperature_coeff_pS": 0.0,
    "config_open_ocean_freezing_temperature_reference_pressure": 0.0,
    "config_land_ice_cavity_freezing_temperature_coeff_0": 6.22e-2,
    "config_land_ice_cavity_freezing_temperature_coeff_S": -5.63e-2,
    "config_land_ice_cavity_freezing_temperature_coeff_p": -7.43e-8,
    "config_land_ice_cavity_freezing_temperature_coeff_pS": -1.74e-10,
    "config_land_ice_cavity_freezing_temperature_reference_pressure": 0.0,
}

#: (name, seed, nCells, nVertLevels, eos type, (indexT, indexS), T range, S range, config overrides)
CASES = (
    ("linear_defaults", 20261201, 48, 24, "linear", (1, 2), (-2.0, 30.0), (30.0, 38.0), {}),
    ("linear_configured", 20261202, 40, 30, "linear", (2, 3), (-2.5, 32.0), (0.0, 41.0),
     {"config_eos_linear_densityref": 1027.5, "config_eos_linear_alpha": 0.17, "config_eos_linear_beta": 0.76,
      "config_eos_linear_Tref": 10.0, "config_eos_linear_Sref": 34.7,
      "config_open_ocean_freezing_temperature_coeff_0": 0.0901,
      "config_open_ocean_freezing_temperature_coeff_S": -0.0575,
      "config_open_ocean_freezing_temperature_coeff_p": -7.61e-8,
      "config_open_ocean_freezing_temperature_coeff_pS": 1.2e-10,
      "config_open_ocean_freezing_temperature_reference_pressure": 1.0e5,
      "config_land_ice_cavity_freezing_temperature_reference_pressure": 2.5e5}),
    ("jm_ocean", 20261203, 64, 40, "jm", (1, 2), (-1.9, 29.0), (32.0, 37.5), {}),
    # T and S past JM's valid range on both sides: the clamps
    ("jm_clamped", 20261204, 48, 30, "jm", (3, 1), (-6.0, 46.0), (-3.0, 47.0), {}),
    # one level: k + 1 and nVertLevels both fall back to level 1
    ("jm_one_level", 20261205, 32, 1, "jm", (1, 2), (-2.0, 30.0), (0.0, 40.0), {}),
    # thin surface layers: exp(-0.025 depth) - 1 cancels, so the reference
    # pressure there carries exp's every bit
    ("jm_deep_short_columns", 20261206, 40, 60, "jm", (2, 1), (-1.5, 25.0), (34.0, 35.5),
     {"config_density0": 1027.0, "_thin_top": 8}),
)


def make_case(name, seed, n_cells, n_levels, eos_type, tracer_index, t_range, s_range, overrides) -> dict:
    """One deterministic case, with an ocean column's magnitudes.

    - ``refBottomDepth`` from a few metres to several kilometres (JM's
      reference pressure at every depth);
    - ``maxLevelCell`` from 1 to ``nVertLevels``: the levels below it keep
      their sentinel words in every output;
    - temperatures and salinities over each case's range (``jm_clamped`` past
      both of JM's clamps), salinity 0 exactly on some cells;
    - freezing (salinity, pressure) pairs on both sides of each reference
      pressure, and exactly at it.
    """
    rs = np.random.RandomState(seed)
    nc, nl, nt = n_cells, n_levels, 3

    def uniform(lo, hi, shape):
        return lo + (hi - lo) * rs.random_sample(shape)

    thickness = uniform(2.0, 250.0, nl)
    thin = overrides.get("_thin_top", 0)
    thickness[:thin] = uniform(0.05, 2.0, thin)
    ref_bottom_depth = np.cumsum(thickness)
    max_level_cell = rs.randint(1, nl + 1, nc).astype(np.int32)
    max_level_cell[: min(nc, 4)] = nl
    tracers = uniform(0.0, 1.0, (nc, nl, nt))
    tracers[:, :, tracer_index[0] - 1] = uniform(*t_range, (nc, nl))
    tracers[:, :, tracer_index[1] - 1] = uniform(*s_range, (nc, nl))
    surface = uniform(0.0, 1.0, (nc, nt))
    surface[:, tracer_index[0] - 1] = uniform(*t_range, nc)
    surface[:, tracer_index[1] - 1] = uniform(*s_range, nc)
    if s_range[0] <= 0.0:
        tracers[rs.random_sample((nc, nl)) < 0.05, tracer_index[1] - 1] = 0.0
    configs = {**DEFAULTS, **{k: v for k, v in overrides.items() if not k.startswith("_")}}
    freeze_s = uniform(0.0, 40.0, (nc, nl))
    freeze_s[rs.random_sample((nc, nl)) < 0.05] = 0.0
    freeze_p = uniform(0.0, 6.0e7, (nc, nl))
    pick = rs.random_sample((nc, nl))
    for lo, hi, where in ((0.0, 0.1, "open_ocean"), (0.2, 0.3, "land_ice_cavity")):
        # a tenth below each reference pressure, and some exactly at it
        reference = configs[f"config_{where}_freezing_temperature_reference_pressure"]
        below = (pick >= lo) & (pick < hi)
        freeze_p[below] = reference - uniform(0.0, 2.0e5, (nc, nl))[below]
        freeze_p[(pick >= hi) & (pick < hi + 0.03)] = reference
    return {
        "name": name, "nCells": nc, "nVertLevels": nl, "nTracers": nt, "eosType": eos_type,
        "indexT": tracer_index[0], "indexS": tracer_index[1], "configs": configs,
        "maxLevelCell": max_level_cell, "refBottomDepth": ref_bottom_depth,
        # activeTracers(nTracers, nVertLevels, nCells): C-order (nCells, nVertLevels, nTracers)
        "activeTracers": tracers, "tracersSurfaceValue": surface,
        "sentinel": uniform(-1.0e3, 1.0e3, (nc, nl)), "freezeS": freeze_s, "freezeP": freeze_p,
    }


def cases() -> list[dict]:
    return [make_case(*row) for row in CASES]


REAL_ARRAYS = ("refBottomDepth", "activeTracers", "tracersSurfaceValue", "sentinel", "freezeS", "freezeP")


def input_bytes(case: dict, precision: str) -> bytes:
    """run_eos.F90's input stream for ``case`` at ``precision``."""
    real = np.dtype(PRECISIONS[precision]).newbyteorder("<")
    head = np.array([case["nCells"], case["nVertLevels"], case["nTracers"], case["indexT"], case["indexS"],
                     EOS_TYPES.index(case["eosType"]) + 1], "<i4")
    reals = np.array([case["configs"]["config_density0"]] + [case["configs"][k] for k in REALS]).astype(real)
    parts = [head.tobytes(), reals.tobytes(), np.ascontiguousarray(case["maxLevelCell"], "<i4").tobytes()]
    for name in REAL_ARRAYS:
        parts.append(np.ascontiguousarray(case[name]).astype(real).tobytes())
    return b"".join(parts)


def input_digest(case: dict, precision: str) -> str:
    return hashlib.sha256(input_bytes(case, precision)).hexdigest()


def read_outputs(raw: bytes, case: dict, precision: str) -> dict[str, np.ndarray]:
    real = np.dtype(PRECISIONS[precision]).newbyteorder("<")
    shape = (case["nCells"], case["nVertLevels"])
    n = int(np.prod(shape))
    out, off = {}, 0
    for name in OUT_FIELDS:
        out[name] = np.frombuffer(raw, real, n, off).reshape(shape).astype(PRECISIONS[precision])
        off += real.itemsize * n
    if off != len(raw):
        raise ValueError(f"oracle output: {len(raw) - off} trailing bytes")
    return out


def load_fixture(precision: str) -> dict[str, dict[str, np.ndarray]]:
    return _tend.load_fixture(precision, FIXTURE_DIR)


def run_gpu(case: dict, precision: str) -> dict[str, np.ndarray]:
    """Run woof.hex.mpas_ocean.eos_v842 on ``case`` as run_eos.F90 drives the Fortran."""
    import cupy as cp
    from woof.hex.mpas_ocean import eos_v842 as eos

    dtype = PRECISIONS[precision]
    real = {name: cp.asarray(np.ascontiguousarray(case[name]).astype(dtype)) for name in REAL_ARRAYS}
    mesh = {"maxLevelCell": cp.asarray(np.ascontiguousarray(case["maxLevelCell"], np.int32)),
            "refBottomDepth": real["refBottomDepth"]}
    configs = dict(case["configs"], config_eos_type=case["eosType"])
    out = {name: real["sentinel"].copy() for name in OUT_FIELDS}
    for k_displaced, kind, coefficients, fields in CALLS:
        eos.equation_of_state_density(
            mesh=mesh, active_tracers=real["activeTracers"], tracers_surface_value=real["tracersSurfaceValue"],
            index_temperature=case["indexT"], index_salinity=case["indexS"], density=out[fields[0]],
            k_displaced=case["nVertLevels"] if k_displaced is None else k_displaced, displacement_type=kind,
            configs=configs, thermal_expansion_coeff=out[fields[1]] if coefficients else None,
            saline_contraction_coeff=out[fields[2]] if coefficients else None)
    for field, cavity, deriv in zip(FREEZING_FIELDS, (False, True, False, True), (False, False, True, True)):
        function = eos.freezing_temperature_salinity_deriv if deriv else eos.freezing_temperature
        function(salinity=real["freezeS"], pressure=real["freezeP"], in_land_ice_cavity=cavity, configs=configs,
                 out=out[field])
    return {name: cp.asnumpy(out[name]) for name in OUT_FIELDS}


def measure(precision: str, reference: dict | None = None) -> dict[str, dict[str, int]]:
    reference = load_fixture(precision) if reference is None else reference
    report: dict[str, dict[str, int]] = {}
    for case in cases():
        got = run_gpu(case, precision)
        want = reference[case["name"]]
        report[case["name"]] = {name: bit_mismatches(got[name], want[name]) for name in want}
    return report
