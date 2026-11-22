"""Cases for the MPAS-Ocean forward mode with the land-ice fluxes on.

The driver is the forward oracle's (``tools/mpas_ocean_oracle/run_forward.F90``,
:mod:`woof.verify.mpas_ocean_forward_oracle`): per step the absorbed
fractions, ``ocn_surface_land_ice_fluxes_build_arrays``, ``ocn_timestep``
(whose diagnostics compute the land-ice input fields, whose tendencies add the
land-ice terms, whose split step updates the effective density and the
coupler's means in 'coupled') and the time-level shift, every module compiled
verbatim. The field table is the forward oracle's plus the land-ice flux and
coupling packages' fields, ``cellMask`` and the scratch fields. A third of the
cells lie under land ice; the kinetic energy the loops do not reach is drawn
non-negative (it feeds the top drag's square root).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from woof.verify import mpas_ocean_forward_oracle as _fw
from woof.verify import mpas_ocean_land_ice_oracle as _li
from woof.verify import mpas_ocean_stream as _stream
from woof.verify import mpas_ocean_tend_oracle as _tend
from woof.verify.mpas_ocean_tend_oracle import PRECISIONS

FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mpas_ocean_forward_land_ice")
CASES_PER_PRECISION = True
REAL, INT, DIM, SCRATCH = _stream.REAL, _stream.INT, _stream.DIM, _stream.SCRATCH
_LAND_ICE = (
    ("mesh", "cellMask", INT, ("L", "C1")),
    ("state", "accumulatedLandIceMass", REAL, ("C1",)), ("state", "accumulatedLandIceMass@2", REAL, ("C1",)),
    ("state", "accumulatedLandIceHeat", REAL, ("C1",)), ("state", "accumulatedLandIceHeat@2", REAL, ("C1",)),
    ("state", "effectiveDensityInLandIce", REAL, ("C1",)), ("state", "effectiveDensityInLandIce@2", REAL, ("C1",)),
    ("diagnostics", "index_landIceBoundaryLayerTemperature", DIM, ()),
    ("diagnostics", "index_landIceBoundaryLayerSalinity", DIM, ()),
    ("diagnostics", "index_landIceHeatTransferVelocity", DIM, ()),
    ("diagnostics", "index_landIceSaltTransferVelocity", DIM, ()),
    ("diagnostics", "landIceFrictionVelocity", REAL, ("C1",)),
    ("diagnostics", "topDrag", REAL, ("E1",)), ("diagnostics", "topDragMagnitude", REAL, ("C1",)),
    ("diagnostics", "landIceBoundaryLayerTracers", REAL, (2, "C1")),
    ("diagnostics", "landIceTracerTransferVelocities", REAL, (2, "C1")),
    ("forcing", "index_landIceInterfaceTemperature", DIM, ()), ("forcing", "index_landIceInterfaceSalinity", DIM, ()),
    ("forcing", "landIceInterfaceTracers", REAL, (2, "C1")),
    ("forcing", "landIceFreshwaterFlux", REAL, ("C1",)), ("forcing", "landIceHeatFlux", REAL, ("C1",)),
    ("forcing", "heatFluxToLandIce", REAL, ("C1",)),
    ("forcing", "avgLandIceBoundaryLayerTracers", REAL, (2, "C1")),
    ("forcing", "avgLandIceTracerTransferVelocities", REAL, (2, "C1")),
    ("forcing", "avgEffectiveDensityInLandIce", REAL, ("C1",)),
    ("scratch", "boundaryLayerTemperatureScratch", SCRATCH, ("C1",)),
    ("scratch", "boundaryLayerSalinityScratch", SCRATCH, ("C1",)),
    ("scratch", "effectiveDensityScratch", SCRATCH, ("C1",)),
)
FIELDS = _fw.FIELDS + _LAND_ICE
_COUPLING = {"effectiveDensityInLandIce", "avgLandIceBoundaryLayerTracers", "avgLandIceTracerTransferVelocities",
             "avgEffectiveDensityInLandIce"}
_ON = {**{k: v for k, v in _li.BASE_CONFIG.items() if k.startswith("config_land_ice_flux")},
       "config_land_ice_cavity_freezing_temperature_reference_pressure": 1.0e5}
#: (name, seed, (nq, nr), nVertLevels, steps, config_dt, config_btr_dt, config overrides)
CASES = (
    ("standalone_jenkins_three_steps", 20262301, (7, 6), 8, 3, "00:02:00", "0000_00:00:50", {
        **_ON, "config_land_ice_flux_boundaryLayerNeighborWeight": 0.3}),
    ("standalone_isomip_nested_halos", 20262302, (12, 6), 8, 3, "00:02:00", "0000_00:00:30", {
        **_ON, "_owned": 3, "config_land_ice_flux_formulation": "ISOMIP",
        "config_land_ice_flux_boundaryLayerThickness": 30.0}),
    ("coupled_jenkins_two_steps", 20262303, (6, 5), 6, 2, "00:03:00", "0000_00:01:00", {
        **_ON, "config_land_ice_flux_mode": "coupled", "config_land_ice_flux_boundaryLayerNeighborWeight": 1.0,
        "config_land_ice_flux_attenuation_coefficient": 25.0}),
    # no step: the startup build ocn_init_routines_block makes for output
    ("standalone_jenkins_startup", 20262304, (6, 6), 6, 0, "00:02:00", "0000_00:00:40", dict(_ON)),
)


def present(case: dict, name: str) -> bool:
    base = name.split("@")[0]
    if base in _COUPLING:
        return case["config"]["config_land_ice_flux_mode"] == "coupled"
    return True


def make_case(name, seed, torus, n_levels, steps, dt, btr_dt, overrides, precision="r4") -> dict:
    """The forward oracle's cold start with the land-ice fields added (from their own seed)."""
    case = _fw.make_case(name, seed, torus, n_levels, steps, dt, btr_dt, overrides, precision)
    rs = np.random.RandomState(seed + 13)
    real = PRECISIONS[precision]
    nc, nl = case["dims"]["nCells"], case["dims"]["nVertLevels"]

    def uniform(lo, hi, shp):
        return lo + (hi - lo) * rs.random_sample(shp)

    pools = case["pools"]
    m, f, d = pools["mesh"], pools["forcing"], pools["diagnostics"]
    fraction = uniform(0.0, 1.0, nc + 1)
    fraction[rs.random_sample(nc + 1) < 0.5] *= 0.4
    f["landIceFraction"] = fraction.astype(real)
    f["landIcePressure"] = (uniform(1.0e5, 1.0e6, nc + 1) * (fraction >= 0.5)).astype(real)
    f["landIceDraft"] = (-uniform(10.0, 100.0, nc + 1) * (fraction >= 0.5)).astype(real)
    d["kineticEnergyCell"] = uniform(0.0, 0.1, d["kineticEnergyCell"].shape).astype(real)
    # the diagnostics rewrite the mask from the fraction; it starts as integer noise
    f["landIceMask"] = rs.randint(-3, 4, nc + 1).astype(np.int32)
    values = {
        "cellMask": (np.arange(nl)[None, :] < m["maxLevelCell"][:, None]).astype(np.int32),
        "accumulatedLandIceMass": uniform(0.0, 1.0e3, nc + 1), "accumulatedLandIceHeat": uniform(-1.0e8, 1.0e8, nc + 1),
        "effectiveDensityInLandIce": uniform(900.0, 1030.0, nc + 1),
        "index_landIceBoundaryLayerTemperature": 1, "index_landIceBoundaryLayerSalinity": 2,
        "index_landIceHeatTransferVelocity": 1, "index_landIceSaltTransferVelocity": 2,
        "index_landIceInterfaceTemperature": 1, "index_landIceInterfaceSalinity": 2,
        # the coupler's fluxes in 'coupled' (the builder's output in 'standalone')
        "landIceFreshwaterFlux": uniform(-1.0e-4, 1.0e-4, nc + 1), "landIceHeatFlux": uniform(-50.0, 50.0, nc + 1),
        "avgLandIceBoundaryLayerTracers": uniform(-1.0, 35.0, (nc + 1, 2)),
        "avgLandIceTracerTransferVelocities": uniform(0.0, 1.0e-4, (nc + 1, 2)),
        "avgEffectiveDensityInLandIce": uniform(900.0, 1030.0, nc + 1),
    }
    values["cellMask"][nc] = 0
    for pool, field, kind, fshape in _LAND_ICE:
        base = field.split("@")[0]
        if kind == DIM:
            value = values[field]
        elif kind == SCRATCH or not present(case, field):
            value = None
        elif base in values:
            value = values[base]
        else:
            value = uniform(-1.0, 1.0, _stream.shape(case, fshape))
        if kind == REAL and value is not None:
            value = np.asarray(value).astype(real)
        pools.setdefault(pool, {})[field] = value
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
             if kind in (REAL, INT) and dev[pool][name] is not None}
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
