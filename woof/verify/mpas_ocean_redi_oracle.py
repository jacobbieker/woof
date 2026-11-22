"""Cases and the stream layout for MPAS-Ocean's Redi isoneutral mixing oracle.

The driver is ``tools/mpas_ocean_oracle/run_tracer_hmix.F90`` (its stream is
self-describing): MPAS-Model v8.4.2's ``ocn_tracer_hmix_tend`` with
``ocn_tracer_hmix_redi_tend`` (``mpas_ocn_tracer_hmix_redi.F`` compiled
verbatim) on with ``config_use_standardGM``, compiled with gfortran -O0
-ffp-contract=off at RKIND 4 and 8. Here GM's slopes and tapering carry their
Registry's ``nVertLevels+1`` levels, and Redi's scratch fields are declared.

The meshes are :func:`~woof.verify.mpas_ocean_hexmesh.hex_mesh` tori (a sixth
of the cells land); ``nested`` is one block of a decomposition, so Redi's
``nCellsArray(1)``/``(2)`` and ``nEdgesArray(2)`` extents differ.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from woof.verify import mpas_ocean_stream as _stream
from woof.verify import mpas_ocean_tend_oracle as _tend
from woof.verify.mpas_ocean_hexmesh import hex_mesh
from woof.verify.mpas_ocean_tend_oracle import PRECISIONS

FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mpas_ocean_redi")
CASES_PER_PRECISION = True
REAL, INT, DIM, DIMS, SCRATCH = _stream.REAL, _stream.INT, _stream.DIM, _stream.DIMS, _stream.SCRATCH
TEND = "tracersTend/activeTracersTend"
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
    ("diagnostics", "relativeSlopeTopOfEdge", REAL, ("LP1", "E1")),
    ("diagnostics", "relativeSlopeTapering", REAL, ("LP1", "E1")),
    ("diagnostics", "relativeSlopeTaperingCell", REAL, ("LP1", "C1")),
    ("scratch", "delsq_tracer", SCRATCH, ("T", "L", "C1")),
    ("scratch", "gradTracerEdge", SCRATCH, ("L", "E1")), ("scratch", "gradTracerTopOfEdge", SCRATCH, ("LP1", "E1")),
    ("scratch", "gradHTracerSlopedTopOfCell", SCRATCH, ("LP1", "C1")),
    ("scratch", "dTracerdZTopOfCell", SCRATCH, ("LP1", "C1")),
    ("scratch", "dTracerdZTopOfEdge", SCRATCH, ("LP1", "E1")), ("scratch", "areaCellSum", SCRATCH, ("L", "C1")),
)
BASE_CONFIG = {
    "config_density0": 1026.0, "config_disable_tr_hmix": False, "config_use_tracer_del2": False,
    "config_tracer_del2": 10.0, "config_use_tracer_del4": False, "config_tracer_del4": 0.0,
    "config_use_standardGM": True, "config_Redi_kappa": 400.0,
    "config_disable_redi_horizontal_term1": False, "config_disable_redi_horizontal_term2": False,
    "config_disable_redi_horizontal_term3": False,
}
#: (name, seed, torus, nVertLevels, overrides); ``_owned`` makes the mesh one block of a decomposition
CASES = (
    ("redi", 20261131, (7, 6), 12, {}),
    ("redi_del2", 20261132, (6, 6), 16, {"config_use_tracer_del2": True, "config_tracer_del2": 250.0,
                                         "config_Redi_kappa": 900.0}),
    ("term1_off", 20261133, (6, 5), 10, {"config_disable_redi_horizontal_term1": True}),
    ("term2_off", 20261134, (5, 6), 10, {"config_disable_redi_horizontal_term2": True}),
    ("term3_off", 20261135, (6, 5), 8, {"config_disable_redi_horizontal_term3": True, "_min_levels": 1}),
    ("nested", 20261136, (12, 6), 9, {"_owned": 4, "config_Redi_kappa": 1500.0}),
    ("disabled", 20261137, (5, 5), 6, {"config_disable_tr_hmix": True}),
)


def make_case(name: str, seed: int, torus: tuple[int, int], n_levels: int, overrides: dict,
              precision: str = "r4") -> dict:
    """One deterministic case with an ocean's magnitudes."""
    rs = np.random.RandomState(seed)
    real = PRECISIONS[precision]
    mesh = hex_mesh(*torus, 1.0 / 6.0, n_levels, rs, min_levels=overrides.get("_min_levels", 2),
                    owned_columns=overrides.get("_owned"))
    nc, ne, nl, me, nt = mesh["nCells"], mesh["nEdges"], n_levels, mesh["maxEdges"], 3
    config = dict(BASE_CONFIG, **{k: v for k, v in overrides.items() if not k.startswith("_")})

    def uniform(lo, hi, shp):
        return lo + (hi - lo) * rs.random_sample(shp)

    thickness = uniform(4.0, 60.0, (nc + 1, nl))
    z_mid = uniform(-0.4, 0.4, (nc + 1, 1)) - (np.cumsum(thickness, axis=1) - 0.5 * thickness)
    c1, c2 = mesh["cellsOnEdge"][:, 0] - 1, mesh["cellsOnEdge"][:, 1] - 1
    thickness_edge = 0.5 * (thickness[c1] + thickness[c2]) * uniform(0.95, 1.05, (ne + 1, nl))
    tracers = uniform(0.0, 1.0, (nc + 1, nl, nt))
    tracers[:, :, 0] = 20.0 - 0.05 * (-z_mid) + uniform(-1.0, 1.0, (nc + 1, nl))
    tracers[:, :, 1] = uniform(33.0, 37.0, (nc + 1, nl))
    dims = {"nCells": nc, "nEdges": ne, "nVertLevels": nl, "nTracers": nt, "maxEdges": me}
    case = {"name": name, "precision": precision, "dims": dims, "config": config, "dt": 600.0}
    values = {
        **mesh, "nVertLevels": nl,
        "areaCell": uniform(1.0e8, 8.0e8, nc + 1), "dvEdge": uniform(5.0e3, 2.0e4, ne + 1),
        "dcEdge": uniform(1.0e4, 3.0e4, ne + 1), "meshScalingDel2": uniform(0.3, 1.0, ne + 1),
        "meshScalingDel4": uniform(0.3, 1.0, ne + 1), "activeTracers": tracers,
        "activeTracersTend": uniform(-1.0e-5, 1.0e-5, (nc + 1, nl, nt)), "layerThicknessEdge": thickness_edge,
        "zMid": z_mid, "relativeSlopeTopOfEdge": uniform(-0.03, 0.03, (ne + 1, nl + 1)),
        "relativeSlopeTapering": uniform(0.0, 1.0, (ne + 1, nl + 1)),
        "relativeSlopeTaperingCell": uniform(0.0, 1.0, (nc + 1, nl + 1)),
    }
    pools: dict[str, dict] = {}
    for pool, field, kind, fshape in FIELDS:
        if kind in (DIM, DIMS):
            value = values[field]
        elif kind == SCRATCH:
            value = None
        else:
            value = values[field]
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
    from woof.hex.mpas_ocean import tracer_hmix_v842

    dev = _stream.on_device(FIELDS, case)
    tracer_hmix_v842.check_config(case["config"])
    tracer_hmix_v842.tracer_hmix_tend(tracers=dev["tracers"]["activeTracers"],
                                      layer_thickness_edge=dev["diagnostics"]["layerThicknessEdge"],
                                      mesh=dev["mesh"], tend=dev["tracersTend"]["activeTracersTend"],
                                      configs=case["config"], diagnostics=dev["diagnostics"])
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
