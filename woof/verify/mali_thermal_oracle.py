"""Cases and the stream layout for the MALI thermal-initialisation oracle.

``tools/mali_oracle/run_thermal.F90`` calls ``li_thermal_init`` and then
``li_compute_pressure_melting_point_fields``, ``li_temperature_to_enthalpy``
and ``li_enthalpy_to_temperature``, all from MPAS-Model v8.4.2 compiled
verbatim (gfortran -O0 -ffp-contract=off, RKIND 4). This module writes the
driver's input stream, reads its output, builds the cases and runs the GPU
port the same way. The RandomState discipline is that of
:mod:`woof.verify.mpas_seaice_evp_oracle`.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mali_thermal_init")

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


def make_case(name, seed, n_cells, n_levels, init, air) -> dict:
    """One deterministic case: an ice sheet up to 3.5 km thick.

    The case covers air temperatures above and below freezing and above the
    pressure melting point, unsolved cells past nCellsSolve, and enthalpies
    either side of each layer's, the surface's (0) and the bed's
    pressure-melting enthalpy.
    """
    rs = np.random.RandomState(seed)
    nc, nl = n_cells, n_levels

    def uniform(lo, hi, shape):
        return (lo + (hi - lo) * rs.random_sample(shape)).astype(np.float32)

    interfaces = np.sort(rs.random_sample(nl - 1))
    sigma_i = np.concatenate(([0.0], interfaces, [1.0])).astype(np.float32)
    sigma_c = ((sigma_i[:-1].astype(np.float64) + sigma_i[1:].astype(np.float64)) * 0.5).astype(np.float32)
    thickness = uniform(0.0, 3500.0, nc)
    thickness[rs.random_sample(nc) < 0.1] = 0.0
    return {
        "name": name, "nCells": nc, "nCellsSolve": nc - 4, "nVertLevels": nl,
        "temperatureInit": init, "airSource": air,
        "iceDensity": np.float32(910.0), "oceanDensity": np.float32(1028.0),
        "airValue": np.float32(268.0 if air == "constant" else 276.0),
        "lapseRate": np.float32(0.0065), "heatFluxValue": np.float32(0.0585),
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


def cases() -> list[dict]:
    return [make_case(*row) for row in CASES]


def _shape(name: str, case: dict) -> tuple[int, ...]:
    return (case["nCells"], case["nVertLevels"]) if name in _LEVEL else (case["nCells"],)


def input_bytes(case: dict) -> bytes:
    head = np.array([case["nCells"], case["nCellsSolve"], case["nVertLevels"], INIT[case["temperatureInit"]],
                     AIR[case["airSource"]]], "<i4")
    reals = np.array([case["iceDensity"], case["oceanDensity"], case["airValue"], case["lapseRate"],
                      case["heatFluxValue"]], "<f4")
    parts = [head.tobytes(), reals.tobytes()]
    for name in GRID_FIELDS + INIT_FIELDS + CONV_INPUTS:
        parts.append(np.ascontiguousarray(case[name], "<f4").tobytes())
    return b"".join(parts)


def input_digest(case: dict) -> str:
    return hashlib.sha256(input_bytes(case)).hexdigest()


def read_outputs(raw: bytes, case: dict) -> dict[str, np.ndarray]:
    out = {"err": np.frombuffer(raw, "<i4", 2, 0).copy()}
    off = 8
    for name in OUTPUTS:
        shape = _shape(name, case)
        count = int(np.prod(shape))
        out[name] = np.frombuffer(raw, "<f4", count, off).reshape(shape).copy()
        off += 4 * count
    if off != len(raw):
        raise ValueError(f"oracle output: {len(raw) - off} trailing bytes")
    return out


def load_fixture() -> dict[str, dict[str, np.ndarray]]:
    with np.load(FIXTURE_DIR / "oracle-r4.npz") as data:
        out: dict[str, dict[str, np.ndarray]] = {}
        for key in data.files:
            case, field = key.split("/", 1)
            out.setdefault(case, {})[field] = data[key]
    return out


def bit_mismatches(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(a, np.float32).view(np.uint32)
                                != np.asarray(b, np.float32).view(np.uint32)))


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
    d["pmpTemperature"] = cp.empty((nc, nl), cp.float32)
    d["basalPmpTemperature"] = cp.empty(nc, cp.float32)
    th.pressure_melting_point_fields(thickness=d["thickness"], layer_center_sigma=d["layerCenterSigma"],
                                     pmp_temperature=d["pmpTemperature"],
                                     basal_pmp_temperature=d["basalPmpTemperature"], rhoi=case["iceDensity"])
    d["outEnthalpy"] = cp.empty((nc, nl), cp.float32)
    th.temperature_to_enthalpy(temperature=d["inTemperature"], waterfrac=d["inWaterfrac"], enthalpy=d["outEnthalpy"],
                               thickness=d["thickness"], layer_center_sigma=d["layerCenterSigma"],
                               rhoi=case["iceDensity"])
    for name in ("convTemperature", "convWaterfrac"):
        d[name] = cp.empty((nc, nl), cp.float32)
    for name in ("convSurfaceTemperature", "convBasalTemperature"):
        d[name] = cp.empty(nc, cp.float32)
    th.enthalpy_to_temperature(enthalpy=d["convEnthalpy"], temperature=d["convTemperature"],
                               waterfrac=d["convWaterfrac"], surface_enthalpy=d["convSurfaceEnthalpy"],
                               surface_temperature=d["convSurfaceTemperature"],
                               basal_enthalpy=d["convBasalEnthalpy"], basal_temperature=d["convBasalTemperature"],
                               thickness=d["thickness"], layer_center_sigma=d["layerCenterSigma"],
                               rhoi=case["iceDensity"])
    return {name: cp.asnumpy(d[name]) for name in OUTPUTS}


def measure(reference: dict | None = None) -> dict[str, dict[str, int]]:
    reference = load_fixture() if reference is None else reference
    report: dict[str, dict[str, int]] = {}
    for case in cases():
        got = run_gpu(case)
        want = reference[case["name"]]
        report[case["name"]] = {name: bit_mismatches(got[name], want[name]) for name in OUTPUTS}
    return report
