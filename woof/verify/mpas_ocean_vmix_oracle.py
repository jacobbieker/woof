"""Cases and the stream layout for the MPAS-Ocean implicit vertical mixing oracle.

``tools/mpas_ocean_oracle/run_vmix.F90`` drives MPAS-Model v8.4.2's
``ocn_vmix_init`` and ``ocn_vmix_implicit`` with ``mpas_ocn_vmix.F`` and its
coefficient modules compiled verbatim, the real equation of state, and a
stand-in for CVMix, with gfortran -O0 -ffp-contract=off at RKIND 4 and 8. The
stream is :mod:`woof.verify.mpas_ocean_stream`'s.

The meshes are random connectivity on one block (every halo extent the
block's count); ``maxLevelCell`` is at least 1 (v8.4.2 writes outside its
arrays for an empty column), velocities are zero below ``maxLevelEdgeTop``.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from woof.verify import mpas_ocean_stream as _stream
from woof.verify import mpas_ocean_tend_oracle as _tend
from woof.verify.mpas_ocean_tend_oracle import PRECISIONS

FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mpas_ocean_vmix")
CASES_PER_PRECISION = True
REAL, INT, DIM, DIMS, SCRATCH = _stream.REAL, _stream.INT, _stream.DIM, _stream.DIMS, _stream.SCRATCH
FIELDS = (
    ("mesh", "nVertLevels", DIM, ()), ("mesh", "nCellsSolve", DIM, ()), ("mesh", "nEdgesSolve", DIM, ()),
    ("mesh", "nCellsArray", DIMS, (4,)), ("mesh", "nEdgesArray", DIMS, (4,)),
    ("mesh", "maxLevelCell", INT, ("C1",)), ("mesh", "maxLevelEdgeTop", INT, ("E1",)),
    ("mesh", "maxLevelEdgeBot", INT, ("E1",)), ("mesh", "cellsOnEdge", INT, (2, "E1")),
    ("mesh", "dvEdge", REAL, ("E1",)), ("mesh", "dcEdge", REAL, ("E1",)), ("mesh", "areaCell", REAL, ("C1",)),
    ("mesh", "nEdgesOnCell", INT, ("C1",)), ("mesh", "edgesOnCell", INT, ("ME", "C1")),
    ("mesh", "edgeSignOnCell", INT, ("ME", "C1")), ("mesh", "refBottomDepth", REAL, ("L",)),
    ("state", "normalVelocity", REAL, ("L", "E1")), ("state", "layerThickness", REAL, ("L", "C1")),
    ("tracers", "index_temperature", DIM, ()), ("tracers", "index_salinity", DIM, ()),
    ("tracers", "activeTracers", REAL, ("T", "L", "C1")),
    ("diagnostics", "vertViscTopOfEdge", REAL, ("LP1", "E1")),
    ("diagnostics", "vertDiffTopOfCell", REAL, ("LP1", "C1")),
    ("diagnostics", "vertViscTopOfCell", REAL, ("LP1", "C1")),
    ("diagnostics", "kineticEnergyCell", REAL, ("L", "C1")), ("diagnostics", "layerThicknessEdge", REAL, ("L", "E1")),
    ("diagnostics", "RiTopOfEdge", REAL, ("LP1", "E1")), ("diagnostics", "RiTopOfCell", REAL, ("LP1", "C1")),
    ("diagnostics", "density", REAL, ("L", "C1")), ("diagnostics", "displacedDensity", REAL, ("L", "C1")),
    ("diagnostics", "tracersSurfaceValue", REAL, ("T", "C1")),
    ("diagnostics", "activeTracerVertMixTendency", REAL, ("T", "L", "C1")),
    ("diagnostics", "vertNonLocalFlux", REAL, ("T", "LP1", "C1")),
    # Redi's coefficient: its GM package is off, so MPAS leaves it unallocated
    ("diagnostics", "k33", REAL, ("LP1", "C1")),
    ("tracersSurfaceFlux", "nonLocalSurfaceTracerFlux", REAL, ("T", "C1")),
    ("scratch", "ddensityTopOfCell", SCRATCH, ("LP1", "C1")), ("scratch", "du2TopOfCell", SCRATCH, ("LP1", "C1")),
    ("scratch", "ddensityTopOfEdge", SCRATCH, ("LP1", "E1")), ("scratch", "du2TopOfEdge", SCRATCH, ("LP1", "E1")),
)
BASE_CONFIG = {
    "config_density0": 1026.0, "config_eos_type": "jm", "config_eos_linear_densityref": 1000.0,
    "config_eos_linear_alpha": 0.2, "config_eos_linear_beta": 0.8, "config_eos_linear_Tref": 5.0,
    "config_eos_linear_Sref": 35.0,
    "config_disable_vel_vmix": False, "config_disable_tr_vmix": False, "config_use_implicit_bottom_drag": True,
    "config_implicit_bottom_drag_coeff": 1.0e-3, "config_use_const_visc": False, "config_use_const_diff": False,
    "config_vert_visc": 1.0e-4, "config_vert_diff": 1.0e-5, "config_use_rich_visc": False,
    "config_use_rich_diff": False, "config_bkrd_vert_visc": 1.0e-4, "config_bkrd_vert_diff": 1.0e-5,
    "config_rich_mix": 0.005, "config_convective_visc": 1.0, "config_convective_diff": 1.0,
    "config_use_tanh_visc": False, "config_use_tanh_diff": False, "config_use_cvmix": False,
    "config_cvmix_kpp_nonlocal_with_implicit_mix": False, "config_compute_active_tracer_budgets": False,
    "config_use_standardGM": False, "config_Redi_kappa": 0.0,
}
#: (name, seed, nCells, nVertLevels, config overrides)
CASES = (
    ("const", 20261271, 40, 16, {"config_use_const_visc": True, "config_use_const_diff": True}),
    ("rich_budgets", 20261272, 48, 24, {"config_use_rich_visc": True, "config_use_rich_diff": True,
                                        "config_compute_active_tracer_budgets": True}),
    ("const_rich_linear_no_drag", 20261273, 44, 20,
     {"config_use_const_visc": True, "config_use_const_diff": True, "config_use_rich_visc": True,
      "config_use_rich_diff": True, "config_eos_type": "linear", "config_use_implicit_bottom_drag": False}),
    ("rich_diff_only_one_level_columns", 20261274, 36, 12, {"config_use_rich_diff": True, "_shallow": True}),
    ("vel_disabled", 20261275, 24, 8, {"config_use_const_diff": True, "config_disable_vel_vmix": True,
                                       "config_compute_active_tracer_budgets": True}),
    ("tracers_disabled", 20261276, 24, 8, {"config_use_const_visc": True, "config_disable_tr_vmix": True}),
)


def make_case(name: str, seed: int, n_cells: int, n_levels: int, overrides: dict, precision: str = "r4") -> dict:
    """One deterministic case: stratified columns of an ocean's T/S with shear,
    so the Richardson numbers take both signs; ``_shallow`` puts a third of the
    columns at one level (the bottom drag's level-2 read)."""
    rs = np.random.RandomState(seed)
    real = PRECISIONS[precision]
    nc, nl, nt, me = n_cells, n_levels, 2, 7
    ne = 3 * nc
    config = dict(BASE_CONFIG, **{k: v for k, v in overrides.items() if not k.startswith("_")})

    def uniform(lo, hi, shp):
        return lo + (hi - lo) * rs.random_sample(shp)

    def ints(lo, hi, shp):
        return rs.randint(lo, hi, shp).astype(np.int32)

    mlc = np.zeros(nc + 1, np.int32)
    mlc[:nc] = ints(1, nl + 1, nc)
    if overrides.get("_shallow"):
        mlc[:nc][rs.random_sample(nc) < 1.0 / 3.0] = 1
    cells_on_edge = ints(1, nc + 1, (ne + 1, 2))
    cells_on_edge[rs.random_sample(ne + 1) < 1.0 / 12.0, 1] = nc + 1
    cells_on_edge[ne] = nc + 1
    mlet = np.minimum(mlc[cells_on_edge[:, 0] - 1], mlc[cells_on_edge[:, 1] - 1]).astype(np.int32)
    mleb = np.maximum(mlc[cells_on_edge[:, 0] - 1], mlc[cells_on_edge[:, 1] - 1]).astype(np.int32)
    n_edges_on_cell = np.array([5, 6, 6, 6, 7], np.int32)[rs.randint(0, 5, nc + 1)]
    n_edges_on_cell[nc] = 0
    edges_on_cell = np.full((nc + 1, me), ne + 1, np.int32)
    edge_sign_on_cell = np.zeros((nc + 1, me), np.int32)
    for c in range(nc):
        k = n_edges_on_cell[c]
        edges_on_cell[c, :k] = ints(1, ne + 1, k)
        edge_sign_on_cell[c, :k] = 2 * ints(0, 2, k) - 1
    level = np.arange(nl)[None, :]
    tracers = np.empty((nc + 1, nl, nt))
    warm = uniform(10.0, 28.0, (nc + 1, 1))
    tracers[:, :, 0] = warm - np.cumsum(uniform(-0.3, 1.5, (nc + 1, nl)), axis=1)
    tracers[:, :, 1] = uniform(33.0, 37.0, (nc + 1, nl))
    dims = {"nCells": nc, "nEdges": ne, "nVertLevels": nl, "nTracers": nt, "maxEdges": me}
    case = {"name": name, "precision": precision, "dims": dims, "config": config, "dt": 1800.0}
    values = {
        "nVertLevels": nl, "nCellsSolve": nc, "nEdgesSolve": ne, "nCellsArray": np.array([nc] * 4, np.int32),
        "nEdgesArray": np.array([ne] * 4, np.int32), "maxLevelCell": mlc, "maxLevelEdgeTop": mlet,
        "maxLevelEdgeBot": mleb, "cellsOnEdge": cells_on_edge, "dvEdge": uniform(5.0e3, 2.0e4, ne + 1),
        "dcEdge": uniform(1.0e4, 3.0e4, ne + 1), "areaCell": uniform(1.0e8, 8.0e8, nc + 1),
        "nEdgesOnCell": n_edges_on_cell, "edgesOnCell": edges_on_cell, "edgeSignOnCell": edge_sign_on_cell,
        "refBottomDepth": np.cumsum(uniform(2.0, 150.0, nl)),
        "normalVelocity": uniform(-0.6, 0.6, (ne + 1, nl)) * (level < mlet[:, None]),
        "layerThickness": uniform(2.0, 150.0, (nc + 1, nl)), "index_temperature": 1, "index_salinity": 2,
        "activeTracers": tracers, "kineticEnergyCell": uniform(0.0, 0.3, (nc + 1, nl)),
        "vertViscTopOfCell": uniform(0.0, 1.0e-2, (nc + 1, nl + 1)),
    }
    pools: dict[str, dict] = {}
    for pool, field, kind, fshape in FIELDS:
        if kind in (DIM, DIMS):
            value = values[field]
        elif kind == SCRATCH or field == "k33" or (field == "activeTracerVertMixTendency"
                                                   and not config["config_compute_active_tracer_budgets"]):
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
    """Run woof.hex.mpas_ocean.vmix_v842 on ``case`` as run_vmix.F90 drives the Fortran."""
    import cupy as cp
    from woof.hex.mpas_ocean import vmix_v842 as vm

    dev = _stream.on_device(FIELDS, case)
    vm.vmix_implicit(dt=case["dt"], mesh=dev["mesh"], state=dev["state"], tracers=dev["tracers"],
                     diagnostics=dev["diagnostics"], configs=case["config"])
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
