"""Cases and the stream layout for the MPAS-Ocean vertical transport oracle.

``tools/mpas_ocean_oracle/run_vert_transport.F90`` drives MPAS-Model v8.4.2's
``ocn_vert_transport_velocity_top`` (cut verbatim out of
``mpas_ocn_diagnostics.F``) with ``mpas_ocn_thick_ale.F`` compiled verbatim,
with gfortran -O0 -ffp-contract=off at RKIND 4 and 8. The stream is
:mod:`woof.verify.mpas_ocean_stream`'s.

The meshes are random connectivity on one block (every halo extent the
block's count); ``maxLevelCell`` is at least 1 (v8.4.2's min/max thickness
writes outside its array for an empty column).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from woof.verify import mpas_ocean_stream as _stream
from woof.verify import mpas_ocean_tend_oracle as _tend
from woof.verify.mpas_ocean_tend_oracle import PRECISIONS

FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mpas_ocean_vert_transport")
CASES_PER_PRECISION = True
REAL, INT, DIM, DIMS, SCRATCH = _stream.REAL, _stream.INT, _stream.DIM, _stream.DIMS, _stream.SCRATCH
FIELDS = (
    ("mesh", "nVertLevels", DIM, ()), ("mesh", "nCells", DIM, ()), ("mesh", "nCellsArray", DIMS, (4,)),
    ("mesh", "nEdgesOnCell", INT, ("C1",)), ("mesh", "areaCell", REAL, ("C1",)),
    ("mesh", "edgesOnCell", INT, ("ME", "C1")), ("mesh", "edgeSignOnCell", INT, ("ME", "C1")),
    ("mesh", "maxLevelCell", INT, ("C1",)), ("mesh", "maxLevelEdgeBot", INT, ("E1",)),
    ("mesh", "dvEdge", REAL, ("E1",)), ("mesh", "vertCoordMovementWeights", REAL, ("L",)),
    ("verticalMesh", "restingThickness", REAL, ("L", "C1")),
    ("state", "layerThickness", REAL, ("L", "C1")), ("state", "normalVelocity", REAL, ("L", "E1")),
    ("state", "ssh", REAL, ("C1",)),
    ("diagnostics", "layerThicknessEdge", REAL, ("L", "E1")),
    ("diagnostics", "vertAleTransportTop", REAL, ("LP1", "C1")),
    ("scratch", "div_hu", SCRATCH, ("L", "C1")), ("scratch", "projectedSSH", SCRATCH, ("C1",)),
    ("scratch", "ALE_Thickness", SCRATCH, ("L", "C1")),
)
BASE_CONFIG = {
    "config_density0": 1026.0, "config_vert_coord_movement": "uniform_stretching",
    "config_use_min_max_thickness": False, "config_max_thickness_factor": 6.0, "config_min_thickness": 1.0,
    "package:thicknessFilterActive": False,
}
#: (name, seed, nCells, nVertLevels, config overrides)
CASES = (
    ("zstar", 20261281, 48, 20, {}),
    ("zstar_min_max", 20261282, 44, 24, {"config_use_min_max_thickness": True, "config_min_thickness": 3.0,
                                         "config_max_thickness_factor": 1.5, "_thin": True}),
    ("fixed_bottom_weights", 20261283, 40, 16, {"_weights": "upper"}),
    ("impermeable_interfaces", 20261284, 24, 8, {"config_vert_coord_movement": "impermeable_interfaces"}),
)


def make_case(name: str, seed: int, n_cells: int, n_levels: int, overrides: dict, precision: str = "r4") -> dict:
    """One deterministic case; ``_thin`` gives thin resting layers and large SSH
    swings so the min/max thickness limits act both ways; ``_weights='upper'``
    moves only the upper half of the column."""
    rs = np.random.RandomState(seed)
    real = PRECISIONS[precision]
    nc, nl, me = n_cells, n_levels, 7
    ne = 3 * nc
    config = dict(BASE_CONFIG, **{k: v for k, v in overrides.items() if not k.startswith("_")})

    def uniform(lo, hi, shp):
        return lo + (hi - lo) * rs.random_sample(shp)

    def ints(lo, hi, shp):
        return rs.randint(lo, hi, shp).astype(np.int32)

    mlc = np.zeros(nc + 1, np.int32)
    mlc[:nc] = ints(1, nl + 1, nc)
    cells_on_edge = ints(1, nc + 1, (ne + 1, 2))
    cells_on_edge[rs.random_sample(ne + 1) < 1.0 / 12.0, 1] = nc + 1
    mleb = np.maximum(mlc[cells_on_edge[:, 0] - 1], mlc[cells_on_edge[:, 1] - 1]).astype(np.int32)
    mleb[ne] = 0
    n_edges_on_cell = np.array([5, 6, 6, 6, 7], np.int32)[rs.randint(0, 5, nc + 1)]
    n_edges_on_cell[nc] = 0
    edges_on_cell = np.full((nc + 1, me), ne + 1, np.int32)
    edge_sign_on_cell = np.zeros((nc + 1, me), np.int32)
    for c in range(nc):
        k = n_edges_on_cell[c]
        edges_on_cell[c, :k] = ints(1, ne + 1, k)
        edge_sign_on_cell[c, :k] = 2 * ints(0, 2, k) - 1
    resting = uniform(2.0, 150.0, (nc + 1, nl))
    ssh = uniform(-3.0, 3.0, nc + 1)
    if overrides.get("_thin"):
        resting = uniform(2.0, 6.0, (nc + 1, nl))
        ssh = uniform(-40.0, 40.0, nc + 1)
    weights = np.ones(nl)
    if overrides.get("_weights") == "upper":
        weights[nl // 2:] = 0.0
    dims = {"nCells": nc, "nEdges": ne, "nVertLevels": nl, "maxEdges": me}
    case = {"name": name, "precision": precision, "dims": dims, "config": config, "dt": 900.0}
    values = {
        "nVertLevels": nl, "nCells": nc, "nCellsArray": np.array([nc] * 4, np.int32),
        "nEdgesOnCell": n_edges_on_cell, "areaCell": uniform(1.0e8, 8.0e8, nc + 1), "edgesOnCell": edges_on_cell,
        "edgeSignOnCell": edge_sign_on_cell, "maxLevelCell": mlc, "maxLevelEdgeBot": mleb,
        "dvEdge": uniform(5.0e3, 2.0e4, ne + 1), "vertCoordMovementWeights": weights,
        "restingThickness": resting, "layerThickness": resting * uniform(0.95, 1.05, (nc + 1, nl)),
        "normalVelocity": uniform(-0.8, 0.8, (ne + 1, nl)), "ssh": ssh,
        "layerThicknessEdge": uniform(2.0, 150.0, (ne + 1, nl)),
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
    """Run woof.hex.mpas_ocean.vert_transport_v842 on ``case`` as run_vert_transport.F90 drives the Fortran."""
    import cupy as cp
    from woof.hex.mpas_ocean import vert_transport_v842 as vt

    dev = _stream.on_device(FIELDS, case)
    s = dev["state"]
    vt.vert_transport_velocity_top(
        mesh=dev["mesh"], resting_thickness=dev["verticalMesh"]["restingThickness"],
        old_layer_thickness=s["layerThickness"], layer_thickness_edge=dev["diagnostics"]["layerThicknessEdge"],
        normal_velocity=s["normalVelocity"], old_ssh=s["ssh"], dt=case["dt"],
        vert_ale_transport_top=dev["diagnostics"]["vertAleTransportTop"], configs=case["config"])
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
