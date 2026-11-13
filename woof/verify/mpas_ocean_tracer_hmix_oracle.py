"""Cases and the stream layout for the MPAS-Ocean horizontal tracer mixing oracle.

``tools/mpas_ocean_oracle/run_tracer_hmix.F90`` drives MPAS-Model v8.4.2's
``ocn_tracer_hmix_init``, ``ocn_tracer_hmix_redi_init`` and
``ocn_tracer_hmix_tend`` (del2, del4; Redi off), the four modules compiled
verbatim with gfortran -O0 -ffp-contract=off at RKIND 4 and 8. The stream is
:mod:`woof.verify.mpas_ocean_stream`'s.

The meshes are random connectivity on one block (every halo extent the
block's count); a boundary edge, whose second cell is the missing
``nCells+1``, has ``maxLevelEdgeTop`` 0, and ``edgeMask`` has zeros inside
active columns (del4 skips them).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from woof.verify import mpas_ocean_stream as _stream
from woof.verify import mpas_ocean_tend_oracle as _tend
from woof.verify.mpas_ocean_tend_oracle import PRECISIONS

FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mpas_ocean_tracer_hmix")
CASES_PER_PRECISION = True
REAL, INT, DIM, DIMS, SCRATCH = _stream.REAL, _stream.INT, _stream.DIM, _stream.DIMS, _stream.SCRATCH
FIELDS = (
    ("mesh", "nVertLevels", DIM, ()), ("mesh", "nCellsArray", DIMS, (4,)), ("mesh", "nEdgesArray", DIMS, (4,)),
    ("mesh", "maxLevelEdgeTop", INT, ("E1",)), ("mesh", "maxLevelCell", INT, ("C1",)),
    ("mesh", "cellsOnEdge", INT, (2, "E1")), ("mesh", "areaCell", REAL, ("C1",)), ("mesh", "dvEdge", REAL, ("E1",)),
    ("mesh", "dcEdge", REAL, ("E1",)), ("mesh", "meshScalingDel2", REAL, ("E1",)),
    ("mesh", "meshScalingDel4", REAL, ("E1",)), ("mesh", "nEdgesOnCell", INT, ("C1",)),
    ("mesh", "edgesOnCell", INT, ("ME", "C1")), ("mesh", "edgeSignOnCell", INT, ("ME", "C1")),
    ("mesh", "edgeMask", INT, ("L", "E1")),
    ("tracers", "activeTracers", REAL, ("T", "L", "C1")),
    ("tracersTend", "activeTracersTend", REAL, ("T", "L", "C1")),
    ("diagnostics", "layerThicknessEdge", REAL, ("L", "E1")), ("diagnostics", "zMid", REAL, ("L", "C1")),
    ("diagnostics", "relativeSlopeTopOfEdge", REAL, ("L", "E1")),
    ("diagnostics", "relativeSlopeTapering", REAL, ("L", "E1")),
    ("diagnostics", "relativeSlopeTaperingCell", REAL, ("L", "C1")),
    ("scratch", "delsq_tracer", SCRATCH, ("T", "L", "C1")),
)
BASE_CONFIG = {
    "config_density0": 1026.0, "config_disable_tr_hmix": False, "config_use_tracer_del2": False,
    "config_tracer_del2": 10.0, "config_use_tracer_del4": False, "config_tracer_del4": 0.0,
    "config_use_standardGM": False,
}
#: (name, seed, nCells, nVertLevels, config overrides)
CASES = (
    ("del2", 20261251, 48, 20, {"config_use_tracer_del2": True, "config_tracer_del2": 250.0}),
    ("del4", 20261252, 44, 16, {"config_use_tracer_del4": True, "config_tracer_del4": 3.0e9}),
    ("del2_del4", 20261253, 40, 24, {"config_use_tracer_del2": True, "config_tracer_del2": 10.0,
                                     "config_use_tracer_del4": True, "config_tracer_del4": 1.0e10}),
    # asked for with a zero coefficient (del4) and switched off (del2's coefficient set): nothing runs
    ("off_by_coefficient", 20261254, 32, 8, {"config_use_tracer_del4": True, "config_tracer_del2": 50.0}),
    ("disabled", 20261255, 24, 8, {"config_disable_tr_hmix": True, "config_use_tracer_del2": True}),
)


def make_case(name: str, seed: int, n_cells: int, n_levels: int, overrides: dict, precision: str = "r4") -> dict:
    """One deterministic case with an ocean mesh's magnitudes."""
    rs = np.random.RandomState(seed)
    real = PRECISIONS[precision]
    nc, nl, nt, me = n_cells, n_levels, 3, 7
    ne = 3 * nc
    config = dict(BASE_CONFIG, **overrides)

    def uniform(lo, hi, shp):
        return lo + (hi - lo) * rs.random_sample(shp)

    def ints(lo, hi, shp):
        return rs.randint(lo, hi, shp).astype(np.int32)

    mlc = np.zeros(nc + 1, np.int32)
    mlc[:nc] = ints(1, nl + 1, nc)
    cells_on_edge = ints(1, nc + 1, (ne + 1, 2))
    cells_on_edge[rs.random_sample(ne + 1) < 1.0 / 12.0, 1] = nc + 1
    cells_on_edge[ne] = nc + 1
    mlet = np.minimum(mlc[cells_on_edge[:, 0] - 1], mlc[cells_on_edge[:, 1] - 1]).astype(np.int32)
    level = np.arange(nl)[None, :]
    edge_mask = (level < mlet[:, None]).astype(np.int32)
    edge_mask[rs.random_sample((ne + 1, nl)) < 0.05] = 0
    n_edges_on_cell = np.array([5, 6, 6, 6, 7], np.int32)[rs.randint(0, 5, nc + 1)]
    n_edges_on_cell[nc] = 0
    edges_on_cell = np.full((nc + 1, me), ne + 1, np.int32)
    edge_sign_on_cell = np.zeros((nc + 1, me), np.int32)
    for c in range(nc):
        k = n_edges_on_cell[c]
        edges_on_cell[c, :k] = ints(1, ne + 1, k)
        edge_sign_on_cell[c, :k] = 2 * ints(0, 2, k) - 1
    tracers = uniform(0.0, 1.0, (nc + 1, nl, nt))
    tracers[:, :, 0] = uniform(-1.8, 28.0, (nc + 1, nl))
    tracers[:, :, 1] = uniform(33.0, 37.0, (nc + 1, nl))
    dims = {"nCells": nc, "nEdges": ne, "nVertLevels": nl, "nTracers": nt, "maxEdges": me}
    case = {"name": name, "precision": precision, "dims": dims, "config": config, "dt": 600.0}
    values = {
        "nVertLevels": nl, "nCellsArray": np.array([nc] * 4, np.int32), "nEdgesArray": np.array([ne] * 4, np.int32),
        "maxLevelEdgeTop": mlet, "maxLevelCell": mlc, "cellsOnEdge": cells_on_edge,
        "areaCell": uniform(1.0e8, 8.0e8, nc + 1), "dvEdge": uniform(5.0e3, 2.0e4, ne + 1),
        "dcEdge": uniform(1.0e4, 3.0e4, ne + 1), "meshScalingDel2": uniform(0.3, 1.0, ne + 1),
        "meshScalingDel4": uniform(0.3, 1.0, ne + 1), "nEdgesOnCell": n_edges_on_cell,
        "edgesOnCell": edges_on_cell, "edgeSignOnCell": edge_sign_on_cell, "edgeMask": edge_mask,
        "activeTracers": tracers, "activeTracersTend": uniform(-1.0e-5, 1.0e-5, (nc + 1, nl, nt)),
        "layerThicknessEdge": uniform(1.0, 150.0, (ne + 1, nl)),
    }
    pools: dict[str, dict] = {}
    for pool, field, kind, fshape in FIELDS:
        if kind in (DIM, DIMS):
            value = values[field]
        elif kind == SCRATCH:
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
    """Run woof.hex.mpas_ocean.tracer_hmix_v842 on ``case`` as run_tracer_hmix.F90 drives the Fortran."""
    import cupy as cp
    from woof.hex.mpas_ocean import tracer_hmix_v842 as th

    dev = _stream.on_device(FIELDS, case)
    th.tracer_hmix_tend(tracers=dev["tracers"]["activeTracers"],
                        layer_thickness_edge=dev["diagnostics"]["layerThicknessEdge"], mesh=dev["mesh"],
                        tend=dev["tracersTend"]["activeTracersTend"], configs=case["config"])
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
