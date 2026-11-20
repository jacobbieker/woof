"""Cases and the stream layout for the MPAS-Ocean frazil oracle.

``tools/mpas_ocean_oracle/run_frazil.F90`` drives MPAS-Model v8.4.2's
``ocn_frazil_forcing_init``, ``ocn_frazil_forcing_build_arrays``,
``ocn_frazil_forcing_layer_thickness`` and ``ocn_frazil_forcing_tracers``
(``mpas_ocn_frazil_forcing.F`` compiled verbatim with the real equation of
state) with gfortran -O0 -ffp-contract=off at RKIND 4 and 8. The stream is
:mod:`woof.verify.mpas_ocean_stream`'s; a field's time level 2 is ``NAME@2``.

The columns are drawn so every branch is taken: layers below and above their
freezing point, frazil formed and then melted higher in the column, the
thickness limit binding, columns skipped (all warmer than the maximum
freezing temperature, or under land ice / in the open ocean when that is
switched off), levels below the maximum depth, empty columns, the land-ice
cavity's freezing coefficients, and halo cells beyond ``nCellsArray(2)`` that
the routines leave alone. The accumulated level-2 fields and the tendencies
start from noise, so a word the routine does not write is checked too.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from woof.verify import mpas_ocean_eos_oracle as _eos
from woof.verify import mpas_ocean_stream as _stream
from woof.verify import mpas_ocean_tend_oracle as _tend
from woof.verify.mpas_ocean_tend_oracle import PRECISIONS

FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mpas_ocean_frazil")
CASES_PER_PRECISION = True
REAL, INT, DIM, DIMS = _stream.REAL, _stream.INT, _stream.DIM, _stream.DIMS
FIELDS = (
    ("mesh", "nVertLevels", DIM, ()), ("mesh", "nCells", DIM, ()), ("mesh", "nCellsArray", DIMS, (4,)),
    ("mesh", "maxLevelCell", INT, ("C1",)),
    ("state", "layerThickness", REAL, ("L", "C1")), ("state", "ssh", REAL, ("C1",)),
    ("state", "accumulatedFrazilIceMass", REAL, ("C1",)), ("state", "accumulatedFrazilIceMass@2", REAL, ("C1",)),
    ("state", "accumulatedFrazilIceSalinity", REAL, ("C1",)),
    ("state", "accumulatedFrazilIceSalinity@2", REAL, ("C1",)),
    ("state", "accumulatedLandIceFrazilMass", REAL, ("C1",)),
    ("state", "accumulatedLandIceFrazilMass@2", REAL, ("C1",)),
    ("tracers", "index_temperature", DIM, ()), ("tracers", "index_salinity", DIM, ()),
    ("tracers", "activeTracers", REAL, ("T", "L", "C1")),
    ("diagnostics", "zMid", REAL, ("L", "C1")), ("diagnostics", "density", REAL, ("L", "C1")),
    ("diagnostics", "pressure", REAL, ("L", "C1")),
    ("forcing", "frazilLayerThicknessTendency", REAL, ("L", "C1")),
    ("forcing", "frazilTemperatureTendency", REAL, ("L", "C1")),
    ("forcing", "frazilSalinityTendency", REAL, ("L", "C1")), ("forcing", "frazilSurfacePressure", REAL, ("C1",)),
    ("forcing", "landIceMask", INT, ("C1",)),
    ("tend", "layerThickness", REAL, ("L", "C1")),
    ("tracersTend", "activeTracersTend", REAL, ("T", "L", "C1")),
)
#: Registry.xml defaults (v8.4.2) of the frazil, equation-of-state and clock configs the driver reads
BASE_CONFIG = {
    **_eos.DEFAULTS, "config_eos_type": "linear",
    "config_start_time": "0001-01-01_00:00:00", "config_dt": "00:10:00",
    "config_use_frazil_ice_formation": True, "config_frazil_in_open_ocean": True,
    "config_frazil_under_land_ice": True, "config_frazil_heat_of_fusion": 3.34e5,
    "config_frazil_ice_density": 1000.0, "config_frazil_fractional_thickness_limit": 0.1,
    "config_specific_heat_sea_water": 3.996e3, "config_frazil_maximum_depth": 100.0,
    "config_frazil_sea_ice_reference_salinity": 4.0, "config_frazil_land_ice_reference_salinity": 0.0,
    "config_frazil_maximum_freezing_temperature": 0.0, "config_frazil_use_surface_pressure": True,
}
_CAVITY = {"config_land_ice_cavity_freezing_temperature_reference_pressure": 1.0e5}
#: (name, seed, nCells, nVertLevels, nTracers, (indexT, indexS), cells solved (nCellsArray(2)), land ice, overrides)
CASES = (
    ("open_ocean", 20262001, 40, 12, 2, (1, 2), 40, False, {}),
    ("land_ice_cavity", 20262002, 44, 10, 2, (1, 2), 44, True, _CAVITY),
    ("land_ice_only", 20262003, 36, 9, 3, (2, 3), 36, True, {"config_frazil_in_open_ocean": False, **_CAVITY}),
    ("open_only_no_pressure_halo", 20262004, 48, 14, 2, (1, 2), 30, True, {
        "config_frazil_under_land_ice": False, "config_frazil_use_surface_pressure": False,
        "config_dt": "00:03:00"}),
    ("configured_limits", 20262005, 42, 16, 2, (2, 1), 42, False, {
        "config_open_ocean_freezing_temperature_coeff_0": 0.0901,
        "config_open_ocean_freezing_temperature_coeff_S": -0.0575,
        "config_open_ocean_freezing_temperature_coeff_p": -7.61e-8,
        "config_open_ocean_freezing_temperature_coeff_pS": 1.2e-10,
        "config_open_ocean_freezing_temperature_reference_pressure": 1.0e5,
        "config_frazil_fractional_thickness_limit": 0.002, "config_frazil_maximum_depth": 40.0,
        "config_frazil_maximum_freezing_temperature": -1.2, "config_frazil_sea_ice_reference_salinity": 40.0,
        "config_frazil_ice_density": 917.0, "config_frazil_heat_of_fusion": 3.337e5,
        "config_specific_heat_sea_water": 3.974e3, "config_density0": 1027.5, "config_dt": "01:00:00"}),
)


def interval_seconds(text: str) -> int:
    days, _, clock = text.rpartition("_")
    h, m, s = (int(x) for x in clock.split(":"))
    return ((int(days or 0) * 24 + h) * 60 + m) * 60 + s


def make_case(name, seed, n_cells, n_levels, n_tracers, indices, solved, land_ice, overrides,
              precision="r4") -> dict:
    rs = np.random.RandomState(seed)
    real = PRECISIONS[precision]
    nc, nl, nt = n_cells, n_levels, n_tracers
    config = dict(BASE_CONFIG, **overrides)

    def uniform(lo, hi, shp):
        return lo + (hi - lo) * rs.random_sample(shp)

    mlc = np.zeros(nc + 1, np.int32)
    mlc[:nc] = rs.randint(1, nl + 1, nc)
    mlc[:nc][rs.random_sample(nc) < 0.05] = 0
    thickness = uniform(2.0, 25.0, (nc + 1, nl))
    ssh = uniform(-4.0, 4.0, nc + 1)  # wide: the depth cutoff moves with the sea surface
    z_mid = ssh[:, None] - (np.cumsum(thickness, axis=1) - 0.5 * thickness)
    tracers = uniform(-1.0, 1.0, (nc + 1, nl, nt))
    it, isal = indices
    # near freezing (about -1.8 C for the default coefficients): cold layers
    # and warm layers in every column, some columns all warm
    temperature = uniform(-2.3, 0.6, (nc + 1, nl))
    warm = rs.random_sample(nc + 1) < 0.15
    temperature[warm] = uniform(0.3, 4.0, (int(warm.sum()), nl))
    tracers[:, :, it - 1] = temperature
    tracers[:, :, isal - 1] = uniform(28.0, 36.0, (nc + 1, nl))
    mask = (rs.random_sample(nc + 1) < 0.4).astype(np.int32) if land_ice else None
    pressure = 1.0e4 * (-z_mid) + (uniform(5.0e5, 8.0e6, nc + 1)[:, None] * mask[:, None] if land_ice else 0.0)
    dims = {"nCells": nc, "nEdges": 0, "nVertLevels": nl, "nTracers": nt}
    case = {"name": name, "precision": precision, "dims": dims, "config": config,
            "dt": float(interval_seconds(config["config_dt"]))}
    values = {
        "nVertLevels": nl, "nCells": nc, "nCellsArray": np.array([solved, solved, nc, nc], np.int32),
        "maxLevelCell": mlc, "layerThickness": thickness, "ssh": ssh,
        "accumulatedFrazilIceMass": uniform(0.0, 50.0, nc + 1),
        "accumulatedFrazilIceSalinity": uniform(0.0, 0.2, nc + 1),
        "accumulatedLandIceFrazilMass": uniform(0.0, 50.0, nc + 1),
        "index_temperature": it, "index_salinity": isal, "activeTracers": tracers,
        "zMid": z_mid, "density": uniform(1020.0, 1030.0, (nc + 1, nl)), "pressure": pressure,
        "landIceMask": mask,
    }
    pools: dict[str, dict] = {}
    for pool, field, kind, fshape in FIELDS:
        if kind in (DIM, DIMS) or field in values:
            value = values[field]
        else:
            value = uniform(-1.0, 1.0, _stream.shape(case, fshape))
        if kind == REAL and value is not None:
            value = np.asarray(value).astype(real)
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


def run_gpu(case: dict) -> dict[str, np.ndarray]:
    """Run woof.hex.mpas_ocean.frazil_v842 on ``case`` as run_frazil.F90 drives the Fortran."""
    import cupy as cp
    from woof.hex.mpas_ocean import frazil_v842 as fz

    dev = _stream.on_device(FIELDS, case)
    tr = dev["tracers"]
    config = case["config"]
    fz.build_arrays(dt=case["dt"], mesh=dev["mesh"], state=dev["state"], tracers=tr,
                    diagnostics=dev["diagnostics"], forcing=dev["forcing"], configs=config)
    fz.layer_thickness_tend(mesh=dev["mesh"], forcing=dev["forcing"], tend=dev["tend"]["layerThickness"],
                            configs=config)
    fz.active_tracers_tend(mesh=dev["mesh"], forcing=dev["forcing"], tend=dev["tracersTend"]["activeTracersTend"],
                           index_temperature=int(tr["index_temperature"]), index_salinity=int(tr["index_salinity"]),
                           configs=config)
    return {f"{pool}/{name}": cp.asnumpy(dev[pool][name]) for pool, name, kind, _ in FIELDS
            if kind in (REAL, INT) and dev[pool][name] is not None}


def measure(precision: str, reference: dict | None = None) -> dict[str, dict[str, int]]:
    reference = load_fixture(precision) if reference is None else reference
    report: dict[str, dict[str, int]] = {}
    for case in cases(precision):
        got = run_gpu(case)
        want = reference[case["name"]]
        report[case["name"]] = {name: word_mismatches(got[name], want[name]) for name in want}
    return report
