"""Cases for the MPAS-Ocean forward mode with frazil ice formation on.

The driver is the forward oracle's (``tools/mpas_ocean_oracle/run_forward.F90``,
:mod:`woof.verify.mpas_ocean_forward_oracle`): per step the absorbed
fractions, ``ocn_frazil_forcing_build_arrays``, ``ocn_timestep`` and the
time-level shift, every module compiled verbatim. The field table is the
forward oracle's plus the frazil package's fields (the accumulated mass and
salinity at both time levels, the three tendencies). The columns' water is
drawn about its freezing point so frazil forms, melts back and accumulates
over the steps, under land ice too (``config_land_ice_flux_mode``
'pressure_only' with a land-ice mask).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from woof.verify import mpas_ocean_forward_oracle as _fw
from woof.verify import mpas_ocean_stream as _stream
from woof.verify import mpas_ocean_tend_oracle as _tend
from woof.verify.mpas_ocean_tend_oracle import PRECISIONS

FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mpas_ocean_forward_frazil")
CASES_PER_PRECISION = True
REAL = _stream.REAL
_FRAZIL = (
    ("state", "accumulatedFrazilIceMass", REAL, ("C1",)), ("state", "accumulatedFrazilIceMass@2", REAL, ("C1",)),
    ("state", "accumulatedFrazilIceSalinity", REAL, ("C1",)),
    ("state", "accumulatedFrazilIceSalinity@2", REAL, ("C1",)),
    ("state", "accumulatedLandIceFrazilMass", REAL, ("C1",)),
    ("state", "accumulatedLandIceFrazilMass@2", REAL, ("C1",)),
    ("forcing", "frazilLayerThicknessTendency", REAL, ("L", "C1")),
    ("forcing", "frazilTemperatureTendency", REAL, ("L", "C1")),
    ("forcing", "frazilSalinityTendency", REAL, ("L", "C1")),
)
FIELDS = _fw.FIELDS + _FRAZIL
_ON = {"config_use_frazil_ice_formation": True}
#: (name, seed, (nq, nr), nVertLevels, steps, config_dt, config_btr_dt, config overrides)
CASES = (
    ("frazil_split_three_steps", 20262101, (7, 6), 8, 3, "00:02:00", "0000_00:00:50", {**_ON}),
    ("frazil_land_ice_nested_halos", 20262102, (12, 6), 8, 3, "00:02:00", "0000_00:00:30", {
        **_ON, "_owned": 3, "config_land_ice_flux_mode": "pressure_only",
        "config_land_ice_cavity_freezing_temperature_reference_pressure": 1.0e5}),
    ("frazil_limits_no_pressure", 20262103, (6, 5), 6, 2, "00:10:00", "0000_00:02:00", {
        **_ON, "config_frazil_use_surface_pressure": False, "config_frazil_fractional_thickness_limit": 0.01,
        "config_frazil_maximum_depth": 60.0, "config_frazil_maximum_freezing_temperature": -1.0,
        "config_frazil_in_open_ocean": True, "config_frazil_sea_ice_reference_salinity": 6.0}),
)


def make_case(name, seed, torus, n_levels, steps, dt, btr_dt, overrides, precision="r4") -> dict:
    """The forward oracle's cold start with the frazil fields added and the
    temperature redrawn about the freezing point (from its own seed)."""
    case = _fw.make_case(name, seed, torus, n_levels, steps, dt, btr_dt, overrides, precision)
    rs = np.random.RandomState(seed + 7)
    real = PRECISIONS[precision]
    nc, nl = case["dims"]["nCells"], case["dims"]["nVertLevels"]

    def uniform(lo, hi, shp):
        return lo + (hi - lo) * rs.random_sample(shp)

    pools = case["pools"]
    temperature = uniform(-2.4, 0.8, (nc + 1, nl))
    warm = rs.random_sample(nc + 1) < 0.15
    temperature[warm] = uniform(0.5, 6.0, (int(warm.sum()), nl))
    tracers = pools["tracers"]["activeTracers"].copy()
    tracers[:, :, 0] = temperature.astype(real)
    pools["tracers"]["activeTracers"] = tracers
    pools["tracers"]["activeTracers@2"] = tracers.copy()
    if pools["forcing"].get("landIceMask") is not None:
        pools["forcing"]["landIceMask"] = (rs.random_sample(nc + 1) < 0.4).astype(np.int32)
        pools["forcing"]["landIcePressure"] = (uniform(1.0e5, 2.0e6, nc + 1)
                                               * pools["forcing"]["landIceMask"]).astype(real)
    for pool, field, kind, fshape in _FRAZIL:
        if field.startswith("accumulated"):
            value = uniform(0.0, 20.0, _stream.shape(case, fshape)) if not field.endswith("@2") else \
                uniform(-1.0, 1.0, _stream.shape(case, fshape))
        else:
            value = uniform(-1.0, 1.0, _stream.shape(case, fshape))
        pools[pool][field] = value.astype(real)
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


def run_gpu(case: dict, n_steps: int | None = None) -> dict[str, np.ndarray]:
    """Run woof.hex.mpas_ocean.forward_v842 on ``case`` as run_forward.F90 drives
    the Fortran; every array by the allocation it was given to."""
    import cupy as cp
    from woof.hex.mpas_ocean import forward_v842 as fw

    dev = _stream.on_device(FIELDS, case)
    given = {(pool, name): dev[pool][name] for pool, name, kind, _ in FIELDS
             if kind in (REAL, _stream.INT) and dev[pool][name] is not None}
    config = case["config"]
    clock = fw.cold_start(pools=dev, configs=config)
    fw.advance(pools=dev, configs=config, clock=clock,
               n_steps=int(config["oracle_n_steps"]) if n_steps is None else n_steps)
    return {f"{pool}/{name}": cp.asnumpy(array) for (pool, name), array in given.items()}


def runs(case: dict) -> bool:
    return _fw.runs(case)


def measure(precision: str, reference: dict | None = None) -> dict[str, dict[str, int]]:
    reference = load_fixture(precision) if reference is None else reference
    report: dict[str, dict[str, int]] = {}
    for case in cases(precision):
        if not runs(case):
            continue
        got = run_gpu(case)
        want = reference[case["name"]]
        report[case["name"]] = {name: word_mismatches(got[name], want[name]) for name in want}
    return report
