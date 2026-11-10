"""Cases and the stream layout for the MALI thermal-initialisation oracle.

``tools/mali_oracle/run_thermal.F90`` calls ``li_thermal_init`` and then
``li_compute_pressure_melting_point_fields``, ``li_temperature_to_enthalpy``
and ``li_enthalpy_to_temperature``, all from MPAS-Model v8.4.2 compiled
verbatim (gfortran -O0 -ffp-contract=off, RKIND 4 and 8). This module writes the
driver's input stream, reads its output, builds the cases and runs the GPU
port the same way. The RandomState discipline is that of
:mod:`woof.verify.mpas_seaice_evp_oracle`. A case is built at one precision:
the RKIND 4 case rounds the draws to float32 and the RKIND 8 case keeps them
in float64, and the stream, the fixture and the GPU run follow it.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mali_thermal_init")
PRECISIONS = {"r4": np.float32, "r8": np.float64}

GRID_FIELDS = ("layerCenterSigma", "layerInterfaceSigma", "thickness", "upperSurface")
INIT_FIELDS = ("temperature", "waterfrac", "enthalpy", "surfaceAirTemperature", "surfaceTemperature",
               "basalTemperature", "basalHeatFlux")
CONV_INPUTS = ("inTemperature", "inWaterfrac", "convEnthalpy", "convSurfaceEnthalpy", "convBasalEnthalpy")
OUTPUTS = INIT_FIELDS + ("pmpTemperature", "basalPmpTemperature", "outEnthalpy", "convEnthalpy",
                         "convTemperature", "convWaterfrac", "convSurfaceEnthalpy", "convSurfaceTemperature",
                         "convBasalEnthalpy", "convBasalTemperature")
INIT = {"sfc_air_temperature": 1, "linear": 2, "file": 3}
AIR = {"constant": 1, "lapse": 2}

#: (name, seed, nCells, nVertLevels, temperature init, air temperature source)
CASES = (
    ("sfc_constant", 20261117, 90, 10, "sfc_air_temperature", "constant"),
    ("linear_lapse", 20261118, 70, 12, "linear", "lapse"),
    ("file_lapse", 20261119, 40, 8, "file", "lapse"),
)
_LEVEL = {"temperature", "waterfrac", "enthalpy", "pmpTemperature", "outEnthalpy", "convEnthalpy",
          "convTemperature", "convWaterfrac", "inTemperature", "inWaterfrac"}


def make_case(name, seed, n_cells, n_levels, init, air, precision: str = "r4") -> dict:
    """One deterministic case: an ice sheet up to 3.5 km thick.

    The case covers air temperatures above and below freezing and above the
    pressure melting point, unsolved cells past nCellsSolve, and enthalpies
    either side of each layer's, the surface's (0) and the bed's
    pressure-melting enthalpy.
    """
    rs = np.random.RandomState(seed)
    real = PRECISIONS[precision]
    nc, nl = n_cells, n_levels

    def uniform(lo, hi, shape):
        return (lo + (hi - lo) * rs.random_sample(shape)).astype(real)

    interfaces = np.sort(rs.random_sample(nl - 1))
    sigma_i = np.concatenate(([0.0], interfaces, [1.0])).astype(real)
    sigma_c = ((sigma_i[:-1].astype(np.float64) + sigma_i[1:].astype(np.float64)) * 0.5).astype(real)
    thickness = uniform(0.0, 3500.0, nc)
    thickness[rs.random_sample(nc) < 0.1] = 0.0
    return {
        "name": name, "precision": precision, "nCells": nc, "nCellsSolve": nc - 4, "nVertLevels": nl,
        "temperatureInit": init, "airSource": air,
        "iceDensity": real(910.0), "oceanDensity": real(1028.0),
        "airValue": real(268.0 if air == "constant" else 276.0),
        "lapseRate": real(0.0065), "heatFluxValue": real(0.0585),
        "layerCenterSigma": sigma_c, "layerInterfaceSigma": sigma_i, "thickness": thickness,
        "upperSurface": uniform(0.0, 3800.0, nc),
        "temperature": uniform(230.0, 270.0, (nc, nl)), "waterfrac": uniform(0.0, 0.02, (nc, nl)),
        "enthalpy": uniform(-1.0, 1.0, (nc, nl)), "surfaceAirTemperature": uniform(240.0, 280.0, nc),
        "surfaceTemperature": uniform(-1.0, 1.0, nc), "basalTemperature": uniform(-1.0, 1.0, nc),
        "basalHeatFlux": uniform(-1.0, 1.0, nc),
        "inTemperature": uniform(-30.0, 0.0, (nc, nl)), "inWaterfrac": uniform(0.0, 0.03, (nc, nl)),
        "convEnthalpy": uniform(-6.0e7, 2.0e7, (nc, nl)),
        "convSurfaceEnthalpy": uniform(-4.0e7, 1.0e7, nc),
        "convBasalEnthalpy": uniform(-1.2e7, 1.0e6, nc),
    }


def cases(precision: str = "r4") -> list[dict]:
    return [make_case(*row, precision=precision) for row in CASES]


def _real(case: dict) -> np.dtype:
    return np.dtype(PRECISIONS[case["precision"]]).newbyteorder("<")


def _shape(name: str, case: dict) -> tuple[int, ...]:
    return (case["nCells"], case["nVertLevels"]) if name in _LEVEL else (case["nCells"],)


def input_bytes(case: dict) -> bytes:
    head = np.array([case["nCells"], case["nCellsSolve"], case["nVertLevels"], INIT[case["temperatureInit"]],
                     AIR[case["airSource"]]], "<i4")
    reals = np.array([case["iceDensity"], case["oceanDensity"], case["airValue"], case["lapseRate"],
                      case["heatFluxValue"]], _real(case))
    parts = [head.tobytes(), reals.tobytes()]
    for name in GRID_FIELDS + INIT_FIELDS + CONV_INPUTS:
        parts.append(np.ascontiguousarray(case[name], _real(case)).tobytes())
    return b"".join(parts)


def input_digest(case: dict) -> str:
    return hashlib.sha256(input_bytes(case)).hexdigest()


def read_outputs(raw: bytes, case: dict) -> dict[str, np.ndarray]:
    out = {"err": np.frombuffer(raw, "<i4", 2, 0).copy()}
    off = 8
    for name in OUTPUTS:
        shape = _shape(name, case)
        count = int(np.prod(shape))
        out[name] = np.frombuffer(raw, _real(case), count, off).reshape(shape).astype(PRECISIONS[case["precision"]])
        off += _real(case).itemsize * count
    if off != len(raw):
        raise ValueError(f"oracle output: {len(raw) - off} trailing bytes")
    return out


def load_fixture(precision: str = "r4") -> dict[str, dict[str, np.ndarray]]:
    with np.load(FIXTURE_DIR / f"oracle-{precision}.npz") as data:
        out: dict[str, dict[str, np.ndarray]] = {}
        for key in data.files:
            case, field = key.split("/", 1)
            out.setdefault(case, {})[field] = data[key]
    return out


def bit_mismatches(a: np.ndarray, b: np.ndarray) -> int:
    """Words that differ as bit patterns (so -0.0 != 0.0)."""
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    if a.shape != b.shape or a.dtype.itemsize != b.dtype.itemsize:
        return int(max(a.size, b.size))
    word = np.dtype(f"u{a.dtype.itemsize}")
    return int(np.count_nonzero(a.view(word) != b.view(word)))


def run_gpu(case: dict) -> dict[str, np.ndarray]:
    import cupy as cp
    from woof.hex.mpas_ice import thermal_init_v842 as th

    d = {name: cp.asarray(np.ascontiguousarray(case[name])) for name in GRID_FIELDS + INIT_FIELDS + CONV_INPUTS}
    th.thermal_init(thermal=d, thickness=d["thickness"], upper_surface=d["upperSurface"],
                    layer_center_sigma=d["layerCenterSigma"], n_cells_solve=case["nCellsSolve"],
                    config_ice_density=case["iceDensity"], config_temperature_init=case["temperatureInit"],
                    config_surface_air_temperature_source=case["airSource"],
                    config_surface_air_temperature_value=case["airValue"],
                    config_surface_air_temperature_lapse_rate=case["lapseRate"],
                    config_basal_heat_flux_source="constant", config_basal_heat_flux_value=case["heatFluxValue"])
    nc, nl = case["nCells"], case["nVertLevels"]
    real = PRECISIONS[case["precision"]]
    d["pmpTemperature"] = cp.empty((nc, nl), real)
    d["basalPmpTemperature"] = cp.empty(nc, real)
    th.pressure_melting_point_fields(thickness=d["thickness"], layer_center_sigma=d["layerCenterSigma"],
                                     pmp_temperature=d["pmpTemperature"],
                                     basal_pmp_temperature=d["basalPmpTemperature"], rhoi=case["iceDensity"])
    d["outEnthalpy"] = cp.empty((nc, nl), real)
    th.temperature_to_enthalpy(temperature=d["inTemperature"], waterfrac=d["inWaterfrac"], enthalpy=d["outEnthalpy"],
                               thickness=d["thickness"], layer_center_sigma=d["layerCenterSigma"],
                               rhoi=case["iceDensity"])
    for name in ("convTemperature", "convWaterfrac"):
        d[name] = cp.empty((nc, nl), real)
    for name in ("convSurfaceTemperature", "convBasalTemperature"):
        d[name] = cp.empty(nc, real)
    th.enthalpy_to_temperature(enthalpy=d["convEnthalpy"], temperature=d["convTemperature"],
                               waterfrac=d["convWaterfrac"], surface_enthalpy=d["convSurfaceEnthalpy"],
                               surface_temperature=d["convSurfaceTemperature"],
                               basal_enthalpy=d["convBasalEnthalpy"], basal_temperature=d["convBasalTemperature"],
                               thickness=d["thickness"], layer_center_sigma=d["layerCenterSigma"],
                               rhoi=case["iceDensity"])
    return {name: cp.asnumpy(d[name]) for name in OUTPUTS}


def measure(reference: dict | None = None, precision: str = "r4") -> dict[str, dict[str, int]]:
    reference = load_fixture(precision) if reference is None else reference
    report: dict[str, dict[str, int]] = {}
    for case in cases(precision):
        got = run_gpu(case)
        want = reference[case["name"]]
        report[case["name"]] = {name: bit_mismatches(got[name], want[name]) for name in OUTPUTS}
    return report
