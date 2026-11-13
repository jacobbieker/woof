"""Cases and the stream layout for the MPAS-Ocean ``ocn_tend_tracer`` oracle.

``tools/mpas_ocean_oracle/run_tend_tracer.F90`` drives MPAS-Model v8.4.2's
``ocn_tend_tracer`` with ``mpas_ocn_tendency.F`` compiled verbatim against the
real tracer modules (advection, mixing, surface flux, short-wave absorption,
non-local flux), bulk forcing, diagnostics and equation of state, with
gfortran -O0 -ffp-contract=off at RKIND 4 and 8. The stream is
:mod:`woof.verify.mpas_ocean_stream`'s.

The meshes follow :mod:`woof.verify.mpas_ocean_tracer_advection_oracle`'s
(one block, ``maxLevelCell`` at least 2, ``nVertLevels`` at least 3). The
tracers pool holds ``activeTracers`` (temperature and salinity) alone.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from woof.verify import mpas_ocean_stream as _stream
from woof.verify import mpas_ocean_tend_oracle as _tend
from woof.verify.mpas_ocean_tend_oracle import PRECISIONS
from woof.verify.mpas_ocean_tracer_advection_oracle import SCRATCH_FIELDS as _ADV_SCRATCH

FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mpas_ocean_tend_tracer")
CASES_PER_PRECISION = True
REAL, INT, DIM, DIMS, SCRATCH = _stream.REAL, _stream.INT, _stream.DIM, _stream.DIMS, _stream.SCRATCH
BULK_FIELDS = ("latentHeatFlux", "sensibleHeatFlux", "longWaveHeatFluxUp", "longWaveHeatFluxDown", "seaIceHeatFlux",
               "rainFlux", "snowFlux", "shortWaveHeatFlux", "evaporationFlux", "seaIceFreshWaterFlux",
               "seaIceSalinityFlux", "iceRunoffFlux", "riverRunoffFlux")
BUDGET_FIELDS = ("activeTracerSurfaceFluxTendency", "temperatureShortWaveTendency", "activeTracerNonLocalTendency",
                 "activeTracerHorizontalAdvectionTendency", "activeTracerVerticalAdvectionTendency")
FIELDS = (
    ("mesh", "nVertLevels", DIM, ()), ("mesh", "nEdges", DIM, ()), ("mesh", "nCells", DIM, ()),
    ("mesh", "nCellsSolve", DIM, ()), ("mesh", "maxEdges", DIM, ()),
    ("mesh", "nCellsArray", DIMS, (4,)), ("mesh", "nEdgesArray", DIMS, (4,)),
    ("mesh", "maxLevelCell", INT, ("C1",)), ("mesh", "maxLevelEdgeTop", INT, ("E1",)),
    ("mesh", "latCell", REAL, ("C1",)), ("mesh", "refBottomDepth", REAL, ("L",)),
    ("mesh", "advCoefs", REAL, ("NA", "E1")), ("mesh", "advCoefs3rd", REAL, ("NA", "E1")),
    ("mesh", "highOrderAdvectionMask", INT, ("L", "E1")), ("mesh", "edgeSignOnCell", INT, ("ME", "C1")),
    ("mesh", "nAdvCellsForEdge", INT, ("E1",)), ("mesh", "advCellsForEdge", INT, ("NA", "E1")),
    ("mesh", "dvEdge", REAL, ("E1",)), ("mesh", "dcEdge", REAL, ("E1",)), ("mesh", "cellsOnEdge", INT, (2, "E1")),
    ("mesh", "edgesOnCell", INT, ("ME", "C1")), ("mesh", "cellsOnCell", INT, ("ME", "C1")),
    ("mesh", "areaCell", REAL, ("C1",)), ("mesh", "nEdgesOnCell", INT, ("C1",)),
    ("mesh", "meshScalingDel2", REAL, ("E1",)), ("mesh", "meshScalingDel4", REAL, ("E1",)),
    ("mesh", "edgeMask", INT, ("L", "E1")),
    ("state", "layerThickness", REAL, ("L", "C1")),
    ("tracers", "index_temperature", DIM, ()), ("tracers", "index_salinity", DIM, ()),
    ("tracers", "activeTracers", REAL, ("T", "L", "C1")),
    ("tracersTend", "activeTracersTend", REAL, ("T", "L", "C1")),
    ("tracersSurfaceFlux", "index_temperatureSurfaceFlux", DIM, ()),
    ("tracersSurfaceFlux", "index_salinitySurfaceFlux", DIM, ()),
    ("tracersSurfaceFlux", "activeTracersSurfaceFlux", REAL, ("T", "C1")),
    ("tracersSurfaceFlux", "nonLocalSurfaceTracerFlux", REAL, ("T", "C1")),
    ("tracersSurfaceFlux", "activeTracersSurfaceFluxRunoff", REAL, ("T", "C1")),
    ("tracersSurfaceFlux", "activeTracersSurfaceFluxRemoved", REAL, ("T", "C1")),
    ("diagnostics", "normalTransportVelocity", REAL, ("L", "E1")),
    ("diagnostics", "layerThicknessEdge", REAL, ("L", "E1")),
    ("diagnostics", "vertDiffTopOfCell", REAL, ("LP1", "C1")),
    ("diagnostics", "vertAleTransportTop", REAL, ("LP1", "C1")), ("diagnostics", "zMid", REAL, ("L", "C1")),
    ("diagnostics", "relativeSlopeTopOfEdge", REAL, ("L", "E1")),
    ("diagnostics", "relativeSlopeTapering", REAL, ("L", "E1")),
    ("diagnostics", "relativeSlopeTaperingCell", REAL, ("L", "C1")),
    ("diagnostics", "vertNonLocalFlux", REAL, ("T", "LP1", "C1")),
    ("diagnostics", "penetrativeTemperatureFluxOBL", REAL, ("C1",)),
    ("diagnostics", "activeTracerSurfaceFluxTendency", REAL, ("T", "L", "C1")),
    ("diagnostics", "temperatureShortWaveTendency", REAL, ("L", "C1")),
    ("diagnostics", "activeTracerNonLocalTendency", REAL, ("T", "L", "C1")),
    ("diagnostics", "activeTracerHorizontalAdvectionTendency", REAL, ("T", "L", "C1")),
    ("diagnostics", "activeTracerVerticalAdvectionTendency", REAL, ("T", "L", "C1")),
    ("forcing", "penetrativeTemperatureFlux", REAL, ("C1",)), ("forcing", "fractionAbsorbed", REAL, ("L", "C1")),
    ("forcing", "fractionAbsorbedRunoff", REAL, ("L", "C1")),
    *(("forcing", name, REAL, ("C1",)) for name in BULK_FIELDS),
    ("scratch", "normalThicknessFlux", SCRATCH, ("L", "E1")),
    *(("scratch", name, SCRATCH, ("L", "C1")) for name in _ADV_SCRATCH),
    ("scratch", "highOrderFlux", SCRATCH, ("LP1", "E1")),
    ("scratch", "delsq_tracer", SCRATCH, ("T", "L", "C1")),
)

BASE_CONFIG = {
    "config_density0": 1026.0, "oracle_active_tracers_only": True,
    "config_disable_tr_all_tend": False, "config_use_cvmix_kpp": False,
    "config_cvmix_kpp_nonlocal_with_implicit_mix": False, "config_compute_active_tracer_budgets": False,
    "config_use_activeTracers": True, "config_use_activeTracers_surface_bulk_forcing": False,
    "config_use_activeTracers_surface_restoring": False, "config_use_activeTracers_interior_restoring": False,
    "config_use_activeTracers_exponential_decay": False, "config_use_activeTracers_idealAge_forcing": False,
    "config_use_activeTracers_ttd_forcing": False, "config_land_ice_flux_mode": "off",
    "config_use_frazil_ice_formation": False, "config_use_bulk_wind_stress": False,
    "config_use_bulk_thickness_flux": False,
    # advection
    "config_num_halos": 3, "config_disable_tr_adv": False, "config_dzdk_positive": False,
    "config_check_tracer_monotonicity": False, "config_horiz_tracer_adv_order": 3,
    "config_vert_tracer_adv_order": 3, "config_coef_3rd_order": 0.25, "config_monotonic": True,
    # mixing
    "config_disable_tr_hmix": False, "config_use_tracer_del2": False, "config_tracer_del2": 10.0,
    "config_use_tracer_del4": False, "config_tracer_del4": 0.0, "config_use_standardGM": False,
    # surface flux and short wave
    "config_disable_tr_sflux": False, "config_sw_absorption_type": "none", "config_jerlov_water_type": 3,
    "config_surface_buoyancy_depth": 1.0,
    # the freezing temperature's open-ocean coefficients (the bulk thickness flux)
    "config_open_ocean_freezing_temperature_coeff_0": -1.8, "config_open_ocean_freezing_temperature_coeff_S": 0.0,
    "config_open_ocean_freezing_temperature_coeff_p": 0.0, "config_open_ocean_freezing_temperature_coeff_pS": 0.0,
    "config_open_ocean_freezing_temperature_reference_pressure": 0.0,
}
#: (name, seed, nCells, nVertLevels, config overrides)
CASES = (
    ("advection_only", 20261261, 40, 16, {}),
    ("bulk_jerlov_del2_budgets", 20261262, 48, 20,
     {"config_use_activeTracers_surface_bulk_forcing": True, "config_use_bulk_thickness_flux": True,
      "config_sw_absorption_type": "jerlov", "config_jerlov_water_type": 2, "config_surface_buoyancy_depth": -12.0,
      "config_use_tracer_del2": True, "config_tracer_del2": 100.0, "config_compute_active_tracer_budgets": True,
      "config_open_ocean_freezing_temperature_coeff_0": 0.0901,
      "config_open_ocean_freezing_temperature_coeff_S": -0.0575}),
    ("bulk_salt_limit_del4", 20261263, 44, 24,
     {"config_use_activeTracers_surface_bulk_forcing": True, "config_use_tracer_del4": True,
      "config_tracer_del4": 2.0e9, "config_sw_absorption_type": "jerlov", "config_jerlov_water_type": 5,
      "config_surface_buoyancy_depth": 400.0, "_fresh": True}),
    ("sflux_disabled_runoff", 20261264, 32, 8, {"config_disable_tr_sflux": True, "config_vert_tracer_adv_order": 2}),
    ("active_tracers_off", 20261265, 24, 6, {"config_use_activeTracers": False}),
    ("all_tend_disabled", 20261266, 24, 6, {"config_disable_tr_all_tend": True}),
)


def make_case(name: str, seed: int, n_cells: int, n_levels: int, overrides: dict, precision: str = "r4") -> dict:
    """One deterministic case: an advection mesh, with surface forcing of an
    ocean's size; ``_fresh`` makes some surface salinities small, so the sea-ice
    salt flux is limited."""
    rs = np.random.RandomState(seed)
    real = PRECISIONS[precision]
    nc, nl, nt, me, na = n_cells, n_levels, 2, 7, 14
    ne = 3 * nc
    config = dict(BASE_CONFIG, **{k: v for k, v in overrides.items() if not k.startswith("_")})

    def uniform(lo, hi, shp):
        return lo + (hi - lo) * rs.random_sample(shp)

    def ints(lo, hi, shp):
        return rs.randint(lo, hi, shp).astype(np.int32)

    mlc = np.zeros(nc + 1, np.int32)
    mlc[:nc] = ints(2, nl + 1, nc)
    mlc[: min(nc, 3)] = nl
    cells_on_edge = ints(1, nc + 1, (ne + 1, 2))
    cells_on_edge[rs.random_sample(ne + 1) < 1.0 / 12.0, 1] = nc + 1
    cells_on_edge[ne] = nc + 1
    mlet = np.minimum(mlc[cells_on_edge[:, 0] - 1], mlc[cells_on_edge[:, 1] - 1]).astype(np.int32)
    n_edges_on_cell = np.array([5, 6, 6, 6, 7], np.int32)[rs.randint(0, 5, nc + 1)]
    n_edges_on_cell[nc] = 0
    edges_on_cell = np.full((nc + 1, me), ne + 1, np.int32)
    cells_on_cell = np.full((nc + 1, me), nc + 1, np.int32)
    edge_sign_on_cell = np.zeros((nc + 1, me), np.int32)
    for c in range(nc):
        k = n_edges_on_cell[c]
        edges_on_cell[c, :k] = ints(1, ne + 1, k)
        cells_on_cell[c, :k] = ints(1, nc + 1, k)
        edge_sign_on_cell[c, :k] = 2 * ints(0, 2, k) - 1
    n_adv = ints(6, na + 1, ne + 1)
    n_adv[ne] = 0
    adv_cells = np.full((ne + 1, na), nc + 1, np.int32)
    for e in range(ne):
        adv_cells[e, :n_adv[e]] = ints(1, nc + 1, n_adv[e])
    level = np.arange(nl)[None, :]
    velocity = uniform(-0.5, 0.5, (ne + 1, nl)) * (level < mlet[:, None])
    w = uniform(-1.0e-3, 1.0e-3, (nc + 1, nl + 1))
    w[:, 0] = 0.0
    w[np.arange(nc + 1), mlc] = 0.0
    thickness = uniform(2.0, 60.0, (nc + 1, nl))
    tracers = np.empty((nc + 1, nl, nt))
    tracers[:, :, 0] = uniform(0.5, 28.0, (nc + 1, nl))
    tracers[:, :, 1] = uniform(33.0, 37.0, (nc + 1, nl))
    if overrides.get("_fresh"):
        # salinities this small cannot give up the sea-ice salt flux: it is limited
        fresh = rs.random_sample(nc + 1) < 0.4
        tracers[fresh, 0, 1] = uniform(1.0e-4, 5.0e-3, int(fresh.sum()))
    fraction = uniform(0.0, 1.0, (nc + 1, nl))
    fraction /= fraction.sum(axis=1, keepdims=True) * uniform(0.9, 1.3, (nc + 1, 1))
    dims = {"nCells": nc, "nEdges": ne, "nVertLevels": nl, "nTracers": nt, "maxEdges": me, "nAdvectionCells": na}
    case = {"name": name, "precision": precision, "dims": dims, "config": config, "dt": 600.0}
    values = {
        "nVertLevels": nl, "nEdges": ne, "nCells": nc, "nCellsSolve": nc, "maxEdges": me,
        "nCellsArray": np.array([nc] * 4, np.int32), "nEdgesArray": np.array([ne] * 4, np.int32),
        "maxLevelCell": mlc, "maxLevelEdgeTop": mlet, "latCell": uniform(-1.4, 1.4, nc + 1),
        "refBottomDepth": np.cumsum(uniform(2.0, 60.0, nl)),
        "advCoefs": uniform(-2.0e3, 2.0e3, (ne + 1, na)), "advCoefs3rd": uniform(-1.0e3, 1.0e3, (ne + 1, na)),
        "highOrderAdvectionMask": (rs.random_sample((ne + 1, nl)) < 0.8).astype(np.int32),
        "edgeSignOnCell": edge_sign_on_cell, "nAdvCellsForEdge": n_adv, "advCellsForEdge": adv_cells,
        "dvEdge": uniform(5.0e3, 2.0e4, ne + 1), "dcEdge": uniform(1.0e4, 3.0e4, ne + 1),
        "cellsOnEdge": cells_on_edge, "edgesOnCell": edges_on_cell, "cellsOnCell": cells_on_cell,
        "areaCell": uniform(1.0e8, 8.0e8, nc + 1), "nEdgesOnCell": n_edges_on_cell,
        "meshScalingDel2": uniform(0.3, 1.0, ne + 1), "meshScalingDel4": uniform(0.3, 1.0, ne + 1),
        "edgeMask": (level < mlet[:, None]).astype(np.int32),
        "layerThickness": thickness, "index_temperature": 1, "index_salinity": 2, "activeTracers": tracers,
        "activeTracersTend": uniform(-1.0e-5, 1.0e-5, (nc + 1, nl, nt)),
        "index_temperatureSurfaceFlux": 1, "index_salinitySurfaceFlux": 2,
        "normalTransportVelocity": velocity, "layerThicknessEdge": uniform(2.0, 60.0, (ne + 1, nl)),
        "vertAleTransportTop": w, "fractionAbsorbed": fraction, "fractionAbsorbedRunoff": fraction[::-1].copy(),
        "activeTracersSurfaceFluxRunoff": uniform(-1.0e-6, 1.0e-6, (nc + 1, nt)),
        "latentHeatFlux": uniform(-150.0, 0.0, nc + 1), "sensibleHeatFlux": uniform(-50.0, 50.0, nc + 1),
        "longWaveHeatFluxUp": uniform(-420.0, -300.0, nc + 1), "longWaveHeatFluxDown": uniform(250.0, 400.0, nc + 1),
        "seaIceHeatFlux": uniform(-20.0, 0.0, nc + 1), "rainFlux": uniform(0.0, 1.0e-4, nc + 1),
        "snowFlux": uniform(0.0, 2.0e-5, nc + 1), "shortWaveHeatFlux": uniform(0.0, 300.0, nc + 1),
        "evaporationFlux": uniform(-5.0e-5, 0.0, nc + 1), "seaIceFreshWaterFlux": uniform(-1.0e-5, 1.0e-5, nc + 1),
        "seaIceSalinityFlux": uniform(-5.0e-5, 5.0e-5, nc + 1), "iceRunoffFlux": uniform(0.0, 1.0e-5, nc + 1),
        "riverRunoffFlux": uniform(0.0, 3.0e-4, nc + 1),
    }
    pools: dict[str, dict] = {}
    for pool, field, kind, fshape in FIELDS:
        if kind in (DIM, DIMS):
            value = values[field]
        elif kind == SCRATCH or (field in BUDGET_FIELDS and not config["config_compute_active_tracer_budgets"]):
            value = None
        elif field in values:
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
    """Run woof.hex.mpas_ocean.tend_tracer_v842 on ``case`` as run_tend_tracer.F90 drives the Fortran."""
    import cupy as cp
    from woof.hex.mpas_ocean import tend_tracer_v842 as tt

    dev = _stream.on_device(FIELDS, case)
    tt.tend_tracer(dt=case["dt"], mesh=dev["mesh"], state=dev["state"], tracers=dev["tracers"],
                   tracers_tend=dev["tracersTend"], tracers_surface_flux=dev["tracersSurfaceFlux"],
                   diagnostics=dev["diagnostics"], forcing=dev["forcing"], configs=case["config"])
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
