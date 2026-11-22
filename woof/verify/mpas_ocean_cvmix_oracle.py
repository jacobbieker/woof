"""Cases and the stream layout for the MPAS-Ocean CVMix vertical mixing oracle (RKIND 8).

``tools/mpas_ocean_oracle/run_vmix.F90``, linked as ``run_vmix_cvmix_r8``,
drives MPAS-Model v8.4.2's ``ocn_vmix_init`` and ``ocn_vmix_implicit`` with
``mpas_ocn_vmix.F``, its coefficient modules and ``mpas_ocn_vmix_cvmix.F``
compiled verbatim against CVMix v0.84-beta (its ten modules, verbatim), the
real equation of state, with gfortran -O0 -ffp-contract=off at RKIND 8 only:
``mpas_ocn_vmix_cvmix.F`` does not compile at RKIND 4 (CVMix is real(8)
throughout). The stream is :mod:`woof.verify.mpas_ocean_stream`'s.

The meshes are random connectivity on one block; ``_halos`` gives the cell
and edge halo extents distinct counts (the CVMix loops run over
``nCellsArray(3)``, the smoothing over ``nCellsArray(2)``, the edge average
and the solves over ``nCellsArray(1)``/``nEdgesArray(1)``). The columns are
an ocean's: stratified, with shear, a surface friction velocity and a
buoyancy forcing of either sign (and zero friction velocity for some), the
Brunt-Vaisala frequency negative now and then (MPAS clips it at zero). Every
cell's first neighbour is a real cell, so the smoothing's area sum is never
zero (v8.4.2 divides by it).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from woof.verify import mpas_ocean_stream as _stream
from woof.verify import mpas_ocean_tend_oracle as _tend
from woof.verify import mpas_ocean_vmix_oracle as _vmix

FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mpas_ocean_cvmix")
#: RKIND 8 only: CVMix does not compile at RKIND 4
PRECISIONS = ("r8",)
CASES_PER_PRECISION = True
REAL, INT, DIM, DIMS, SCRATCH = _stream.REAL, _stream.INT, _stream.DIM, _stream.DIMS, _stream.SCRATCH
FIELDS = tuple(row for row in _vmix.FIELDS if row[1] != "vertNonLocalFlux") + (
    ("mesh", "nVertLevelsP1", DIM, ()), ("mesh", "cellsOnCell", INT, ("ME", "C1")),
    ("mesh", "cellMask", INT, ("L", "C1")), ("mesh", "latCell", REAL, ("C1",)), ("mesh", "lonCell", REAL, ("C1",)),
    ("mesh", "fCell", REAL, ("C1",)), ("mesh", "bottomDepth", REAL, ("C1",)),
    ("state", "ssh", REAL, ("C1",)),
    ("diagnostics", "index_vertNonLocalFluxTemp", DIM, ()),
    ("diagnostics", "vertNonLocalFlux", REAL, (1, "LP1", "C1")),
    ("diagnostics", "zTop", REAL, ("L", "C1")), ("diagnostics", "zMid", REAL, ("L", "C1")),
    ("diagnostics", "potentialDensity", REAL, ("L", "C1")),
    ("diagnostics", "BruntVaisalaFreqTop", REAL, ("L", "C1")),
    ("diagnostics", "bulkRichardsonNumber", REAL, ("L", "C1")),
    ("diagnostics", "bulkRichardsonNumberBuoy", REAL, ("L", "C1")),
    ("diagnostics", "bulkRichardsonNumberShear", REAL, ("L", "C1")),
    ("diagnostics", "unresolvedShear", REAL, ("L", "C1")),
    ("diagnostics", "boundaryLayerDepth", REAL, ("C1",)),
    ("diagnostics", "boundaryLayerDepthSmooth", REAL, ("C1",)),
    ("diagnostics", "indexBoundaryLayerDepth", REAL, ("C1",)),
    ("diagnostics", "surfaceFrictionVelocity", REAL, ("C1",)),
    ("diagnostics", "surfaceBuoyancyForcing", REAL, ("C1",)),
    ("forcing", "iceFraction", REAL, ("C1",)),
)
#: v8.4.2's Registry defaults of the CVMix configs, CVMix on
CVMIX_CONFIG = {
    "config_use_cvmix": True, "config_cvmix_prandtl_number": 1.0,
    "config_use_cvmix_background": False, "config_cvmix_background_diffusion": 1.0e-5,
    "config_cvmix_background_viscosity": 1.0e-4,
    "config_use_cvmix_convection": False, "config_cvmix_convective_diffusion": 1.0,
    "config_cvmix_convective_viscosity": 1.0, "config_cvmix_convective_basedOnBVF": True,
    "config_cvmix_convective_triggerBVF": 0.0,
    "config_use_cvmix_shear": False, "config_cvmix_num_ri_smooth_loops": 2, "config_cvmix_use_BLD_smoothing": True,
    "config_cvmix_shear_mixing_scheme": "PP", "config_cvmix_shear_PP_nu_zero": 0.005,
    "config_cvmix_shear_PP_alpha": 5.0, "config_cvmix_shear_PP_exp": 2.0, "config_cvmix_shear_KPP_nu_zero": 0.005,
    "config_cvmix_shear_KPP_Ri_zero": 0.7, "config_cvmix_shear_KPP_exp": 3.0,
    "config_use_cvmix_tidal_mixing": False, "config_use_cvmix_double_diffusion": False,
    "config_use_cvmix_kpp": False, "config_use_cvmix_fixed_boundary_layer": False,
    "config_cvmix_kpp_boundary_layer_depth": 30.0, "config_cvmix_kpp_criticalBulkRichardsonNumber": 0.25,
    "config_cvmix_kpp_matching": "SimpleShapes", "config_cvmix_kpp_EkmanOBL": False,
    "config_cvmix_kpp_MonObOBL": False, "config_cvmix_kpp_interpolationOMLType": "quadratic",
    "config_cvmix_kpp_surface_layer_extent": 0.1, "config_cvmix_kpp_surface_layer_averaging": 5.0,
    "configure_cvmix_kpp_minimum_OBL_under_sea_ice": 10.0, "config_cvmix_kpp_stop_OBL_search": 100.0,
    "config_cvmix_kpp_use_enhanced_diff": True, "config_cvmix_kpp_nonlocal_with_implicit_mix": False,
}
BASE_CONFIG = dict(_vmix.BASE_CONFIG, **CVMIX_CONFIG)
#: E3SM's usual CVMix set: background, Pacanowski-Philander shear, convection, KPP
KPP = {"config_use_cvmix_background": True, "config_use_cvmix_shear": True, "config_use_cvmix_convection": True,
       "config_use_cvmix_kpp": True}
#: (name, seed, nCells, nVertLevels, config overrides; ``_``-keys shape the inputs)
CASES = (
    ("kpp_simple_shapes", 20261301, 48, 24, dict(KPP)),
    ("kpp_nonlocal_implicit_budgets", 20261302, 44, 20,
     dict(KPP, config_cvmix_kpp_nonlocal_with_implicit_mix=True, config_compute_active_tracer_budgets=True)),
    ("kpp_match_both_cubic_kpp_shear_monob", 20261303, 48, 24,
     dict(KPP, config_use_cvmix_background=False, config_cvmix_shear_mixing_scheme="KPP",
          config_cvmix_kpp_matching="MatchBoth", config_cvmix_kpp_interpolationOMLType="cubic",
          config_cvmix_kpp_MonObOBL=True)),
    ("kpp_match_both_quadratic", 20261304, 40, 18,
     dict(KPP, config_cvmix_kpp_matching="MatchBoth", config_cvmix_kpp_nonlocal_with_implicit_mix=True)),
    ("kpp_linear_unsmoothed", 20261305, 40, 16,
     dict(KPP, config_cvmix_kpp_interpolationOMLType="linear", config_cvmix_use_BLD_smoothing=False,
          config_cvmix_num_ri_smooth_loops=0, config_cvmix_kpp_criticalBulkRichardsonNumber=0.3,
          config_cvmix_kpp_surface_layer_extent=0.2)),
    ("kpp_fixed_layer_ice", 20261306, 40, 16,
     dict(KPP, config_use_cvmix_fixed_boundary_layer=True, config_cvmix_kpp_boundary_layer_depth=45.0,
          _ice=True)),
    ("kpp_ice_minimum", 20261307, 44, 20, dict(KPP, configure_cvmix_kpp_minimum_OBL_under_sea_ice=150.0,
                                                    config_cvmix_use_BLD_smoothing=False, _ice=True)),
    ("convection_negative_trigger", 20261308, 32, 12,
     {"config_use_cvmix_convection": True, "config_cvmix_convective_triggerBVF": -1.0e-5}),
    ("convection_shear_no_kpp", 20261309, 36, 14,
     {"config_use_cvmix_background": True, "config_use_cvmix_shear": True, "config_use_cvmix_convection": True}),
    ("background_only", 20261310, 24, 8, {"config_use_cvmix_background": True}),
    ("kpp_shallow_calm", 20261311, 48, 12, dict(KPP, config_cvmix_kpp_MonObOBL=True, _shallow=True, _calm=True)),
    ("kpp_const_rich", 20261312, 40, 16,
     dict(KPP, config_use_const_visc=True, config_use_const_diff=True, config_use_rich_visc=True,
          config_use_rich_diff=True, config_compute_active_tracer_budgets=True)),
    ("kpp_halos", 20261313, 48, 20, dict(KPP, config_cvmix_kpp_nonlocal_with_implicit_mix=True, _halos=True)),
    ("kpp_tidal_ddiff_flags", 20261314, 32, 12,
     dict(KPP, config_use_cvmix_tidal_mixing=True, config_use_cvmix_double_diffusion=True)),
    ("kpp_only_cubic", 20261315, 40, 30,
     {"config_use_cvmix_kpp": True, "config_cvmix_kpp_interpolationOMLType": "cubic",
      "config_cvmix_num_ri_smooth_loops": 5}),
)


def make_case(name: str, seed: int, n_cells: int, n_levels: int, overrides: dict, precision: str = "r8") -> dict:
    """One deterministic case on :func:`woof.verify.mpas_ocean_vmix_oracle.make_case`'s mesh and
    tracers, with the CVMix inputs: an ocean's potential density, stratification and
    surface forcing; ``_shallow`` puts a third of the columns at one or two levels, ``_calm`` a
    third of the friction velocities at zero, ``_ice`` sea ice over half the cells, ``_halos``
    distinct halo extents."""
    case = _vmix.make_case(name, seed, n_cells, n_levels,
                           {k: v for k, v in overrides.items() if k != "_shallow"}, precision)
    case["config"] = dict(BASE_CONFIG, **{k: v for k, v in overrides.items() if not k.startswith("_")})
    rs = np.random.RandomState(seed + 7)
    real = _vmix.PRECISIONS[precision]
    nc, nl, me = n_cells, n_levels, case["dims"]["maxEdges"]
    ne = case["dims"]["nEdges"]
    pools = case["pools"]
    mesh, state, diag = pools["mesh"], pools["state"], pools["diagnostics"]

    def uniform(lo, hi, shp):
        return lo + (hi - lo) * rs.random_sample(shp)

    mlc = mesh["maxLevelCell"]
    if overrides.get("_shallow"):
        shallow = rs.random_sample(nc) < 1.0 / 3.0
        mlc[:nc][shallow] = rs.randint(1, 3, int(shallow.sum()))
        cells_on_edge = mesh["cellsOnEdge"]
        mesh["maxLevelEdgeTop"][:] = np.minimum(mlc[cells_on_edge[:, 0] - 1], mlc[cells_on_edge[:, 1] - 1])
        mesh["maxLevelEdgeBot"][:] = np.maximum(mlc[cells_on_edge[:, 0] - 1], mlc[cells_on_edge[:, 1] - 1])
        level = np.arange(nl)[None, :]
        state["normalVelocity"] = (state["normalVelocity"] * (level < mesh["maxLevelEdgeTop"][:, None])).astype(real)
    if overrides.get("_halos"):
        cells = np.array([nc - 9, nc - 6, nc - 3, nc], np.int32)
        edges = np.array([ne - 20, ne - 12, ne - 5, ne], np.int32)
        mesh["nCellsArray"], mesh["nEdgesArray"] = cells, edges
        mesh["nCellsSolve"], mesh["nEdgesSolve"] = int(cells[0]), int(edges[0])
    n_edges_on_cell = mesh["nEdgesOnCell"]
    cells_on_cell = np.full((nc + 1, me), nc + 1, np.int32)
    for c in range(nc):
        k = n_edges_on_cell[c]
        cells_on_cell[c, :k] = rs.randint(1, nc + 1, k)
        cells_on_cell[c, 1:k][rs.random_sample(k - 1) < 0.15] = nc + 1
    level = np.arange(nl)[None, :]
    cell_mask = (level < mlc[:, None]).astype(np.int32)
    cell_mask[nc] = 0
    # potential density: stable below a mixed layer of random depth, some columns inverted near the top
    h = state["layerThickness"]
    depth = np.cumsum(h, axis=1) - 0.5 * h
    mixed = uniform(5.0, 120.0, (nc + 1, 1))
    pd = 1024.0 + uniform(0.0, 1.0, (nc + 1, 1)) + 0.004 * np.maximum(depth - mixed, 0.0) \
        + uniform(0.0, 2.0e-3, (nc + 1, nl)).cumsum(axis=1)
    inverted = rs.random_sample(nc + 1) < 0.2
    pd[inverted, :3] -= uniform(0.0, 0.05, (int(inverted.sum()), 3)).cumsum(axis=1)
    bvf = uniform(-2.0e-5, 2.0e-4, (nc + 1, nl))
    friction = uniform(2.0e-3, 2.5e-2, nc + 1)
    if overrides.get("_calm"):
        friction[rs.random_sample(nc + 1) < 1.0 / 3.0] = 0.0
    buoyancy = uniform(-2.0e-7, 1.0e-7, nc + 1) * np.where(rs.random_sample(nc + 1) < 0.15, 20.0, 1.0)
    ice = np.zeros(nc + 1)
    if overrides.get("_ice"):
        ice = np.where(rs.random_sample(nc + 1) < 0.5, uniform(0.0, 1.0, nc + 1), 0.0)
    lat = uniform(-1.4, 1.4, nc + 1)
    values = {
        "nVertLevelsP1": nl + 1, "cellsOnCell": cells_on_cell, "cellMask": cell_mask, "latCell": lat,
        "lonCell": uniform(0.0, 6.28, nc + 1), "fCell": 2.0 * 7.292e-5 * np.sin(lat),
        "bottomDepth": h.sum(axis=1), "ssh": uniform(-1.0, 1.0, nc + 1), "index_vertNonLocalFluxTemp": 1,
        "vertNonLocalFlux": uniform(-1.0, 1.0, (nc + 1, nl + 1, 1)), "zTop": -(depth - 0.5 * h), "zMid": -depth,
        "potentialDensity": pd, "BruntVaisalaFreqTop": bvf,
        "bulkRichardsonNumber": uniform(-1.0, 1.0, (nc + 1, nl)),
        "bulkRichardsonNumberBuoy": uniform(-1.0, 1.0, (nc + 1, nl)),
        "bulkRichardsonNumberShear": uniform(-1.0, 1.0, (nc + 1, nl)),
        "unresolvedShear": uniform(-1.0, 1.0, (nc + 1, nl)), "boundaryLayerDepth": uniform(1.0, 80.0, nc + 1),
        "boundaryLayerDepthSmooth": uniform(1.0, 80.0, nc + 1), "indexBoundaryLayerDepth": uniform(1.0, 9.0, nc + 1),
        "surfaceFrictionVelocity": friction, "surfaceBuoyancyForcing": buoyancy, "iceFraction": ice,
        # Ri at the top of the cells: both signs, mostly stable
        "RiTopOfCell": uniform(-0.5, 2.0, (nc + 1, nl + 1)),
    }
    for pool, field, kind, fshape in FIELDS:
        if field in values:
            value = values[field]
            if kind == REAL:
                value = np.asarray(value).astype(real)
            pools.setdefault(pool, {})[field] = value
    return case


def cases(precision: str = "r8") -> list[dict]:
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
    """Run woof.hex.mpas_ocean.vmix_v842 (CVMix through cvmix_v842) on ``case`` as run_vmix.F90 drives the Fortran."""
    import cupy as cp
    from woof.hex.mpas_ocean import vmix_v842 as vm

    dev = _stream.on_device(FIELDS, case)
    vm.vmix_implicit(dt=case["dt"], mesh=dev["mesh"], state=dev["state"], tracers=dev["tracers"],
                     diagnostics=dev["diagnostics"], configs=case["config"], forcing=dev["forcing"],
                     tracers_surface_flux=dev["tracersSurfaceFlux"])
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
