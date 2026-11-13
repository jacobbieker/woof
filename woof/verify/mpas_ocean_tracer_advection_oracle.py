"""Cases and the stream layout for the MPAS-Ocean tracer advection oracle.

``tools/mpas_ocean_oracle/run_tracer_advection.F90`` drives MPAS-Model
v8.4.2's ``ocn_tracer_advection_init`` and ``ocn_tracer_advection_tend`` (the
monotone flux-corrected scheme, ``config_monotonic``), with the three
advection modules compiled verbatim and the two vertical stencils cut out of
``mpas_tracer_advection_helpers.F``, with gfortran -O0 -ffp-contract=off at
RKIND 4 and 8. The stream is :mod:`woof.verify.mpas_ocean_stream`'s.

The meshes are random connectivity on one block (every halo extent the
block's count: the scheme's extents are nested, and random connectivity has no
halo locality to nest). ``maxLevelCell`` is at least 2 and ``nVertLevels`` at
least 3 (v8.4.2 reads outside a column otherwise); a boundary edge, whose
second cell is the missing ``nCells+1``, has ``maxLevelEdgeTop`` 0. Some
fluxes and vertical transports are signed zeros, so the scheme's MAX/MIN
ties show.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from woof.verify import mpas_ocean_stream as _stream
from woof.verify import mpas_ocean_tend_oracle as _tend
from woof.verify.mpas_ocean_tend_oracle import PRECISIONS

FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mpas_ocean_tracer_advection")
CASES_PER_PRECISION = True
REAL, INT, DIM, DIMS, SCRATCH = _stream.REAL, _stream.INT, _stream.DIM, _stream.DIMS, _stream.SCRATCH
SCRATCH_FIELDS = ("tracerCur", "workTendency", "hNewInv", "tracerMin", "tracerMax", "fluxIncoming", "fluxOutgoing",
                  "hProvInv", "hProv")
FIELDS = (
    ("mesh", "nCells", DIM, ()), ("mesh", "nEdges", DIM, ()), ("mesh", "maxEdges", DIM, ()),
    ("mesh", "nCellsArray", DIMS, (4,)), ("mesh", "nEdgesArray", DIMS, (4,)),
    ("mesh", "advCoefs", REAL, ("NA", "E1")), ("mesh", "advCoefs3rd", REAL, ("NA", "E1")),
    ("mesh", "maxLevelCell", INT, ("C1",)), ("mesh", "maxLevelEdgeTop", INT, ("E1",)),
    ("mesh", "highOrderAdvectionMask", INT, ("L", "E1")), ("mesh", "edgeSignOnCell", INT, ("ME", "C1")),
    ("mesh", "nAdvCellsForEdge", INT, ("E1",)), ("mesh", "advCellsForEdge", INT, ("NA", "E1")),
    ("mesh", "dvEdge", REAL, ("E1",)), ("mesh", "cellsOnEdge", INT, (2, "E1")),
    ("mesh", "edgesOnCell", INT, ("ME", "C1")), ("mesh", "cellsOnCell", INT, ("ME", "C1")),
    ("mesh", "areaCell", REAL, ("C1",)), ("mesh", "nEdgesOnCell", INT, ("C1",)),
    ("state", "layerThickness", REAL, ("L", "C1")),
    ("tracers", "activeTracers", REAL, ("T", "L", "C1")),
    ("tracersTend", "activeTracersTend", REAL, ("T", "L", "C1")),
    ("diagnostics", "normalThicknessFlux", REAL, ("L", "E1")),
    ("diagnostics", "vertAleTransportTop", REAL, ("LP1", "C1")),
    ("diagnostics", "activeTracerHorizontalAdvectionTendency", REAL, ("T", "L", "C1")),
    ("diagnostics", "activeTracerVerticalAdvectionTendency", REAL, ("T", "L", "C1")),
    *(("scratch", name, SCRATCH, ("L", "C1")) for name in SCRATCH_FIELDS),
    ("scratch", "highOrderFlux", SCRATCH, ("LP1", "E1")),
)
BUDGET_FIELDS = ("activeTracerHorizontalAdvectionTendency", "activeTracerVerticalAdvectionTendency")

BASE_CONFIG = {
    "config_density0": 1026.0, "config_num_halos": 3, "config_disable_tr_adv": False, "config_dzdk_positive": False,
    "config_check_tracer_monotonicity": False, "config_horiz_tracer_adv_order": 3,
    "config_vert_tracer_adv_order": 3, "config_coef_3rd_order": 0.25, "config_monotonic": True,
    "config_compute_active_tracer_budgets": False,
}
#: (name, seed, nCells, nVertLevels, config overrides)
CASES = (
    ("defaults_3rd_order", 20261241, 48, 20, {}),
    ("vert4_horiz4_budgets", 20261242, 40, 24,
     {"config_vert_tracer_adv_order": 4, "config_horiz_tracer_adv_order": 4,
      "config_compute_active_tracer_budgets": True, "config_check_tracer_monotonicity": True}),
    ("vert2_horiz2", 20261243, 44, 16, {"config_vert_tracer_adv_order": 2, "config_horiz_tracer_adv_order": 2}),
    ("three_levels_coef1", 20261244, 36, 3, {"config_coef_3rd_order": 1.0}),
    ("disabled", 20261245, 24, 8, {"config_disable_tr_adv": True}),
)


def make_case(name: str, seed: int, n_cells: int, n_levels: int, overrides: dict, precision: str = "r4") -> dict:
    """One deterministic case: a column's thicknesses, fluxes of an ocean's
    size, tracers with sharp fronts (the limiter acts) and signed zeros."""
    rs = np.random.RandomState(seed)
    real = PRECISIONS[precision]
    nc, nl, nt, me, na = n_cells, n_levels, 3, 7, 14
    ne = 3 * nc
    config = dict(BASE_CONFIG, **overrides)

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
        cells_on_cell[c, :k][rs.random_sample(k) < 0.05] = nc + 1
        edge_sign_on_cell[c, :k] = 2 * ints(0, 2, k) - 1
    n_adv = ints(6, na + 1, ne + 1)
    n_adv[ne] = 0
    adv_cells = np.full((ne + 1, na), nc + 1, np.int32)
    for e in range(ne):
        adv_cells[e, :n_adv[e]] = ints(1, nc + 1, n_adv[e])
    level = np.arange(nl)[None, :]
    flux = uniform(-2.0e2, 2.0e2, (ne + 1, nl)) * (level < mlet[:, None])
    flux[(rs.random_sample((ne + 1, nl)) < 0.03) & (level < mlet[:, None])] = 0.0
    flux[(rs.random_sample((ne + 1, nl)) < 0.03) & (level < mlet[:, None])] = -0.0
    w = uniform(-1.0e-3, 1.0e-3, (nc + 1, nl + 1))
    w[:, 0] = 0.0
    w[np.arange(nc + 1), mlc] = 0.0
    w[rs.random_sample((nc + 1, nl + 1)) < 0.03] = -0.0
    tracers = uniform(0.0, 1.0, (nc + 1, nl, nt))
    tracers[:, :, 0] = np.where(rs.random_sample((nc + 1, nl)) < 0.3, uniform(-1.8, 2.0, (nc + 1, nl)),
                                uniform(15.0, 28.0, (nc + 1, nl)))
    tracers[:, :, 1] = uniform(33.0, 37.0, (nc + 1, nl))
    dims = {"nCells": nc, "nEdges": ne, "nVertLevels": nl, "nTracers": nt, "maxEdges": me, "nAdvectionCells": na}
    case = {"name": name, "precision": precision, "dims": dims, "config": config, "dt": 600.0}
    values = {
        "nCells": nc, "nEdges": ne, "maxEdges": me, "nCellsArray": np.array([nc] * 4, np.int32),
        "nEdgesArray": np.array([ne] * 4, np.int32),
        "advCoefs": uniform(-2.0e3, 2.0e3, (ne + 1, na)), "advCoefs3rd": uniform(-1.0e3, 1.0e3, (ne + 1, na)),
        "maxLevelCell": mlc, "maxLevelEdgeTop": mlet,
        "highOrderAdvectionMask": (rs.random_sample((ne + 1, nl)) < 0.8).astype(np.int32),
        "edgeSignOnCell": edge_sign_on_cell, "nAdvCellsForEdge": n_adv, "advCellsForEdge": adv_cells,
        "dvEdge": uniform(5.0e3, 2.0e4, ne + 1), "cellsOnEdge": cells_on_edge, "edgesOnCell": edges_on_cell,
        "cellsOnCell": cells_on_cell, "areaCell": uniform(1.0e8, 8.0e8, nc + 1),
        "nEdgesOnCell": n_edges_on_cell, "layerThickness": uniform(2.0, 150.0, (nc + 1, nl)),
        "activeTracers": tracers, "activeTracersTend": uniform(-1.0e-5, 1.0e-5, (nc + 1, nl, nt)),
        "normalThicknessFlux": flux, "vertAleTransportTop": w,
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
            value = uniform(-1.0e3, 1.0e3, _stream.shape(case, fshape))
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
    """Run woof.hex.mpas_ocean.tracer_advection_v842 on ``case`` as
    run_tracer_advection.F90 drives the Fortran."""
    import cupy as cp
    from woof.hex.mpas_ocean import tracer_advection_v842 as ta

    dev = _stream.on_device(FIELDS, case)
    d = dev["diagnostics"]
    ta.tracer_advection_tend(
        tracers=dev["tracers"]["activeTracers"], normal_thickness_flux=d["normalThicknessFlux"],
        w=d["vertAleTransportTop"], layer_thickness=dev["state"]["layerThickness"], dt=case["dt"],
        mesh=dev["mesh"], tend=dev["tracersTend"]["activeTracersTend"], configs=case["config"],
        horizontal_budget=d["activeTracerHorizontalAdvectionTendency"],
        vertical_budget=d["activeTracerVerticalAdvectionTendency"])
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
