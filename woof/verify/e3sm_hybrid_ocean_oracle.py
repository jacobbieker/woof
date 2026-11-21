"""The MPAS v8.4.2 ocean under E3SM v3.2.0's coupler: cases, streams, GPU run.

"E3SM v3.2.0 coupler with MPAS v8.4.2 components, not an E3SM
configuration" (:mod:`woof.hex.coupled.hybrid_ocean`).
``tools/e3sm_coupler_oracle/run_hybrid_ocean.F90`` drives v8.4.2's forward
mode (every module verbatim, at RKIND 8) in the order E3SM's
``ocn_init_mct`` and ``ocn_run_mct`` call it, with E3SM's ``ocn_import_mct``
and ``ocn_export_mct`` cut verbatim and compiled against v8.4.2's modules,
with E3SM's build's constants (``run_hybrid_ocean_e3sm``) or v8.4.2's own
(``run_hybrid_ocean_mpas``).

A case is a v8.4.2 forward start (the forward, frazil or land-ice forward
oracle's, at r8) with the fields the coupler's routines look up added (the
ones v8.4.2 lacks absent, ``avgTotalFreshWaterTemperatureFlux`` zero), the
G case's ``x2o`` and an ``o2x`` with the frazil and land-ice fields, and one
x2o per coupling window, drawn per field (the iceberg fluxes zero, the
shortwave sometimes negative, the sea-ice pressure sometimes past its cap).
The oracle writes ``o2x`` after the init's export and after each window's,
then every field.
"""
from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np

from woof.verify import mpas_ocean_forward_frazil_oracle as _fz
from woof.verify import mpas_ocean_forward_land_ice_oracle as _li
from woof.verify import mpas_ocean_forward_oracle as _fw
from woof.verify import mpas_ocean_stream as _stream
from woof.verify.e3sm_ocean_coupling_oracle import LOGICAL_CONFIGS, O2X_BASE
from woof.verify.e3sm_ocean_merge_oracle import GCASE

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "e3sm_hybrid_ocean"
REAL, DIM, ABSENT_REAL = _stream.REAL, _stream.DIM, _stream.REAL
X2O_FIELDS = tuple(GCASE["x2o"])
O2X_FIELDS = tuple(O2X_BASE)

#: fields E3SM's import looks up that v8.4.2's Registry lacks (absent: the import drops them)
_MISSING_1D = ("icebergHeatFlux icebergFreshWaterFlux removedRiverRunoffFlux removedIceRunoffFlux "
               "significantWaveHeight peakWaveFrequency peakWaveDirection z0wav ustarwav charnockSeaState "
               "waveSupportedZonalWindStress waveSupportedMeridionalWindStress waveToOceanZonalWindStress "
               "waveToOceanMeridionalWindStress waveNetOceanZonalWindStress waveNetOceanMeridionalWindStress "
               "windSpeed10m").split()
_MISSING_2D = ("stokesDriftZonalWavenumber", "stokesDriftMeridionalWavenumber")
_LAND_ICE_IMPORT = (("landIceFreshwaterFlux", ("C1",)), ("landIceHeatFlux", ("C1",)),
                    ("landIceInterfaceTracers", (2, "C1")))
_DIMS = ("index_avgSalinitySurfaceValue", "index_avgSurfaceVelocityZonal", "index_avgSurfaceVelocityMeridional",
         "index_landIceInterfaceTemperature", "index_landIceInterfaceSalinity")
_BL_DIMS = ("index_landIceBoundaryLayerTemperature", "index_landIceBoundaryLayerSalinity",
            "index_landIceHeatTransferVelocity", "index_landIceSaltTransferVelocity")

#: (name, base module, base case name, steps per window, windows, constants, config overrides)
CASES = (
    ("hybrid_forward", _fw, "split_three_steps", 2, 2, "e3sm", {}),
    ("hybrid_frazil", _fz, "frazil_split_three_steps", 3, 2, "e3sm", {}),
    # halos, land ice 'pressure_only' with a mask, v8.4.2's own constants, another density and a
    # positive freezing reference pressure (the export's offset stays 0)
    ("hybrid_frazil_land_ice_mpas_constants", _fz, "frazil_land_ice_nested_halos", 1, 3, "mpas", {
        "config_density0": 1027.5, "config_open_ocean_freezing_temperature_reference_pressure": 2.0e4}),
    ("hybrid_land_ice_coupled", _li, "coupled_jenkins_two_steps", 2, 2, "e3sm", {}),
)

#: x2o's ranges per field (others: small positive)
_X2O_RANGES = {
    "Sa_pbot": (9.8e4, 1.03e5), "Sa_pslv": (9.8e4, 1.03e5), "So_duu10n": (0.0, 200.0), "Si_ifrac": (0.0, 1.0),
    "Si_bpress": (0.0, 7.0e4), "Faxa_rain": (0.0, 1.0e-4), "Faxa_snow": (0.0, 2.0e-5), "Faxa_prec": (0.0, 1.2e-4),
    "Faxa_lwdn": (250.0, 420.0), "Foxx_swnet": (-20.0, 400.0), "Foxx_taux": (-0.3, 0.3), "Foxx_tauy": (-0.3, 0.3),
    "Foxx_lat": (-200.0, 0.0), "Foxx_sen": (-50.0, 50.0), "Foxx_lwup": (-450.0, -300.0), "Foxx_evap": (-5.0e-5, 0.0),
    "Fioi_melth": (-30.0, 10.0), "Fioi_meltw": (-1.0e-5, 1.0e-5), "Fioi_salt": (-1.0e-6, 1.0e-6),
    "Foxx_rofl": (0.0, 5.0e-6), "Foxx_rofi": (0.0, 1.0e-6), "PFioi_bergh": (0.0, 0.0), "PFioi_bergw": (0.0, 0.0),
}


def _fields(base) -> tuple:
    present = {(p, n) for p, n, _, _ in base.FIELDS}
    extra = [("forcing", "iceFraction", REAL, ("C1",)), ("forcing", "seaIceEnergy", REAL, ("C1",)),
             ("forcing", "avgTotalFreshWaterTemperatureFlux", REAL, ("C1",))]
    extra += [("forcing", n, REAL, ("C1",)) for n in _MISSING_1D]
    extra += [("forcing", n, REAL, (6, "C1")) for n in _MISSING_2D]
    extra += [("forcing", n, REAL, shp) for n, shp in _LAND_ICE_IMPORT]
    extra += [("forcing", n, DIM, ()) for n in _DIMS]
    extra += [("diagnostics", n, DIM, ()) for n in _BL_DIMS]
    return tuple(base.FIELDS) + tuple(r for r in extra if (r[0], r[1]) not in present)


def _x2o(rs, lsize: int, windows: int) -> np.ndarray:
    out = np.empty((windows, lsize, len(X2O_FIELDS)))
    for k, name in enumerate(X2O_FIELDS):
        lo, hi = _X2O_RANGES.get(name, (0.0, 1.0e-9))
        out[:, :, k] = (rs.random_sample((windows, lsize)) * (hi - lo)) + lo
    return out


def make_case(name, base, base_name, steps_per_window, windows, constants, overrides) -> dict:
    row = next(r for r in base.CASES if r[0] == base_name)
    case = base.make_case(*row, precision="r8")
    case["name"] = name
    case["config"] = {**case["config"], **overrides}
    c = case["config"]
    c.setdefault("config_land_ice_flux_mode", "off")
    c["config_glc_thermal_forcing_coupling_mode"] = "off"
    for knob in LOGICAL_CONFIGS:
        c[knob] = False
    c["package:frazilIceActive"] = bool(c.get("config_use_frazil_ice_formation", False))
    fields = _fields(base)
    rs = np.random.RandomState(int(hashlib.sha256(name.encode()).hexdigest()[:8], 16))
    pools = case["pools"]
    nc = case["dims"]["nCells"]
    frazil = c["package:frazilIceActive"]
    coupled = c["config_land_ice_flux_mode"] == "coupled"
    for pool, field, kind, fshape in fields:
        if field in pools.get(pool, {}):
            continue
        if kind == DIM:
            value = {"index_avgSalinitySurfaceValue": 2, "index_avgSurfaceVelocityZonal": 1,
                     "index_avgSurfaceVelocityMeridional": 2, "index_landIceInterfaceTemperature": 1,
                     "index_landIceInterfaceSalinity": 2, "index_landIceBoundaryLayerTemperature": 1,
                     "index_landIceBoundaryLayerSalinity": 2, "index_landIceHeatTransferVelocity": 1,
                     "index_landIceSaltTransferVelocity": 2}[field]
        elif field == "iceFraction":
            value = rs.random_sample(nc + 1)
        elif field == "seaIceEnergy":
            value = (rs.random_sample(nc + 1) - 0.5) * 1.0e6 if frazil else None
        elif field == "avgTotalFreshWaterTemperatureFlux":
            value = np.zeros(nc + 1)
        elif field in dict(_LAND_ICE_IMPORT):
            value = (rs.random_sample(_stream.shape(case, fshape)) - 0.5) if coupled else None
        else:
            value = None
        pools.setdefault(pool, {})[field] = value
    lsize = int(pools["mesh"]["nCellsSolve"])
    case["fields"] = fields
    case["hybrid"] = {"steps_per_window": steps_per_window, "windows": windows, "constants": constants,
                      "ocn_cpl_dt": steps_per_window * int(case["dt"]), "x2o": _x2o(rs, lsize, windows),
                      "lsize": lsize}
    return case


def cases() -> list[dict]:
    return [make_case(*row) for row in CASES]


def forward_bytes(case: dict) -> bytes:
    return _stream.input_bytes(case["fields"], case)


def coupling_bytes(case: dict) -> bytes:
    h = case["hybrid"]
    x2o_list, o2x_list = ":".join(X2O_FIELDS).encode(), ":".join(O2X_FIELDS).encode()
    parts = [struct.pack("<5i", h["ocn_cpl_dt"], h["windows"], h["lsize"], len(X2O_FIELDS), len(O2X_FIELDS)),
             struct.pack("<i", len(x2o_list)), x2o_list, struct.pack("<i", len(o2x_list)), o2x_list,
             np.ascontiguousarray(h["x2o"], "<f8").tobytes()]
    return b"".join(parts)


def input_digest(case: dict) -> str:
    return hashlib.sha256(forward_bytes(case) + coupling_bytes(case)).hexdigest()


def read_outputs(o2x_raw: bytes, pools_raw: bytes, case: dict) -> dict[str, np.ndarray]:
    h = case["hybrid"]
    n = (h["windows"] + 1) * h["lsize"] * len(O2X_FIELDS)
    if len(o2x_raw) != 8 * n:
        raise ValueError(f"oracle o2x: {len(o2x_raw)} bytes, not {8 * n}")
    o2x = np.frombuffer(o2x_raw, "<f8").reshape(h["windows"] + 1, h["lsize"], len(O2X_FIELDS)).astype(np.float64)
    out = {f"o2x/{w}": o2x[w] for w in range(h["windows"] + 1)}
    out.update(_stream.read_outputs(case["fields"], pools_raw, case))
    return out


word_mismatches = _stream.word_mismatches


def load_fixture() -> dict[str, dict[str, np.ndarray]]:
    out = {}
    for path in sorted(FIXTURE_DIR.glob("*.npz")):
        with np.load(path) as z:
            out[path.stem] = {k.replace("|", "/"): z[k] for k in z.files}
    return out


def run_gpu(case: dict, *, mutate=None) -> dict[str, np.ndarray]:
    """:class:`~woof.hex.coupled.hybrid_ocean.HybridOcean` on ``case`` as the
    oracle drives the Fortran: every o2x and every array by the allocation it
    was given to. ``mutate(ocean)`` runs after construction (tests)."""
    import cupy as cp

    from woof.hex.coupled.hybrid_ocean import HybridOcean
    fields = case["fields"]
    dev = _stream.on_device(fields, case)
    given = {(pool, name): dev[pool][name] for pool, name, kind, _ in fields
             if kind in (REAL, _stream.INT) and dev[pool][name] is not None}
    h = case["hybrid"]
    ocean = HybridOcean(pools=dev, configs=case["config"], x2o_fields=X2O_FIELDS, o2x_fields=O2X_FIELDS,
                        ocn_cpl_dt=h["ocn_cpl_dt"], constants=h["constants"])
    if mutate is not None:
        mutate(ocean)
    out = {"o2x/0": cp.asnumpy(ocean.initial_o2x())}
    for w in range(h["windows"]):
        out[f"o2x/{w + 1}"] = cp.asnumpy(ocean.run_window(cp.asarray(h["x2o"][w])))
    out.update({f"{pool}/{name}": cp.asnumpy(array) for (pool, name), array in given.items()})
    return out


def measure(reference: dict | None = None) -> dict[str, dict[str, int]]:
    reference = load_fixture() if reference is None else reference
    report: dict[str, dict[str, int]] = {}
    for case in cases():
        got, want = run_gpu(case), reference[case["name"]]
        report[case["name"]] = {name: word_mismatches(got[name], want[name]) for name in want}
    return report
