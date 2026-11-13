"""Cases and the stream layout for the MPAS-Ocean barotropic-stage oracle.

``tools/mpas_ocean_oracle/run_btr_subcycle.F90`` drives stage 2 of MPAS-Model
v8.4.2's ``ocn_time_integrator_split`` (cut verbatim out of
``mpas_ocn_time_integration_split.F``), with gfortran -O0 -ffp-contract=off at
RKIND 4 and 8. The stream is :mod:`woof.verify.mpas_ocean_stream`'s; a
field's time level 2 is the entry ``NAME@2``.

The meshes are random connectivity on one block. ``nested_halos`` gives the
block four nested halo extents, with each layer's ``edgesOnEdge`` inside the
next layer, so the corrector's copy of the new velocity covers every
neighbour it reads; the other cases have every extent the block's count.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from woof.verify import mpas_ocean_stream as _stream
from woof.verify import mpas_ocean_tend_oracle as _tend
from woof.verify.mpas_ocean_tend_oracle import PRECISIONS

FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mpas_ocean_btr_subcycle")
CASES_PER_PRECISION = True
REAL, INT, DIM, DIMS, SCRATCH = _stream.REAL, _stream.INT, _stream.DIM, _stream.DIMS, _stream.SCRATCH
FIELDS = (
    ("mesh", "nVertLevels", DIM, ()), ("mesh", "nCells", DIM, ()), ("mesh", "nEdges", DIM, ()),
    ("mesh", "nCellsArray", DIMS, (4,)), ("mesh", "nEdgesArray", DIMS, (4,)),
    ("mesh", "nEdgesOnCell", INT, ("C1",)), ("mesh", "edgesOnCell", INT, ("ME", "C1")),
    ("mesh", "edgeSignOnCell", INT, ("ME", "C1")), ("mesh", "areaCell", REAL, ("C1",)),
    ("mesh", "bottomDepth", REAL, ("C1",)), ("mesh", "cellsOnEdge", INT, (2, "E1")),
    ("mesh", "nEdgesOnEdge", INT, ("E1",)), ("mesh", "edgesOnEdge", INT, ("ME2", "E1")),
    ("mesh", "weightsOnEdge", REAL, ("ME2", "E1")), ("mesh", "fEdge", REAL, ("E1",)),
    ("mesh", "dcEdge", REAL, ("E1",)), ("mesh", "dvEdge", REAL, ("E1",)), ("mesh", "edgeMask", INT, ("L", "E1")),
    ("mesh", "maxLevelEdgeTop", INT, ("E1",)), ("mesh", "refBottomDepthTopOfCell", REAL, ("LP1",)),
    ("state", "normalBarotropicVelocity", REAL, ("E1",)), ("state", "normalBarotropicVelocity@2", REAL, ("E1",)),
    ("state", "ssh", REAL, ("C1",)), ("state", "sshSubcycle", REAL, ("C1",)),
    ("state", "sshSubcycle@2", REAL, ("C1",)),
    ("state", "normalBarotropicVelocitySubcycle", REAL, ("E1",)),
    ("state", "normalBarotropicVelocitySubcycle@2", REAL, ("E1",)),
    ("state", "normalVelocity@2", REAL, ("L", "E1")), ("state", "normalBaroclinicVelocity@2", REAL, ("L", "E1")),
    ("diagnostics", "barotropicForcing", REAL, ("E1",)), ("diagnostics", "barotropicThicknessFlux", REAL, ("E1",)),
    ("diagnostics", "normalTransportVelocity", REAL, ("L", "E1")),
    ("diagnostics", "normalGMBolusVelocity", REAL, ("L", "E1")),
    ("diagnostics", "layerThicknessEdge", REAL, ("L", "E1")),
    ("tend", "ssh", REAL, ("C1",)),
    ("scratch", "btrvel_temp", SCRATCH, ("E1",)),
)
#: every config ocn_time_integrator_split looks up before its stages
BASE_CONFIG = {
    "config_time_integrator": "split_explicit", "config_n_bcl_iter_beg": 1, "config_n_bcl_iter_mid": 2,
    "config_n_bcl_iter_end": 2, "config_n_ts_iter": 2, "config_btr_subcycle_loop_factor": 2,
    "config_btr_gam1_velWt1": 0.5, "config_btr_gam2_SSHWt1": 1.0, "config_btr_gam3_velWt2": 1.0,
    "config_btr_solve_SSH2": False, "config_n_btr_cor_iter": 2, "config_filter_btr_mode": False,
    "config_mom_del4": 0.0, "config_use_freq_filtered_thickness": False, "config_vel_correction": True,
    "config_disable_vel_all_tend": False, "config_disable_thick_all_tend": False,
    "config_disable_tr_all_tend": False, "config_prescribe_velocity": False, "config_prescribe_thickness": False,
    "config_use_standardGM": False, "config_use_cvmix_kpp": False, "config_land_ice_flux_mode": "off",
    "config_num_halos": 3, "config_compute_active_tracer_budgets": False, "nBtrSubcycles": 3,
}
#: (name, seed, nCells, nVertLevels, config overrides)
CASES = (
    ("defaults", 20261301, 48, 12, {}),
    ("nested_halos", 20261302, 60, 10, {"_nested": True}),
    ("no_predictor_filtered", 20261303, 40, 8, {
        "config_btr_gam1_velWt1": 0.0, "config_filter_btr_mode": True, "config_n_btr_cor_iter": 1,
        "config_vel_correction": False, "config_btr_gam2_SSHWt1": 0.5, "config_btr_subcycle_loop_factor": 1,
        "nBtrSubcycles": 4}),
    ("ssh_corrector", 20261304, 44, 9, {
        "config_btr_solve_SSH2": True, "config_n_btr_cor_iter": 0, "nBtrSubcycles": 1,
        "config_btr_gam2_SSHWt1": 0.7, "config_btr_gam3_velWt2": 0.6, "config_btr_gam1_velWt1": 0.4}),
    ("unsplit", 20261305, 24, 6, {"config_time_integrator": "unsplit_explicit"}),
)


def make_case(name: str, seed: int, n_cells: int, n_levels: int, overrides: dict, precision: str = "r4") -> dict:
    """One deterministic case; ``_nested`` gives nested halo extents."""
    rs = np.random.RandomState(seed)
    real = PRECISIONS[precision]
    nc, nl, me, me2 = n_cells, n_levels, 7, 10
    ne = 3 * nc
    config = dict(BASE_CONFIG, **{k: v for k, v in overrides.items() if not k.startswith("_")})

    def uniform(lo, hi, shp):
        return lo + (hi - lo) * rs.random_sample(shp)

    def ints(lo, hi, shp):
        return rs.randint(lo, hi, shp).astype(np.int32)

    if overrides.get("_nested"):
        cells_array = np.array([nc - 15, nc - 10, nc - 5, nc], np.int32)
        edges_array = np.array([ne - 36, ne - 24, ne - 12, ne], np.int32)
    else:
        cells_array = np.array([nc] * 4, np.int32)
        edges_array = np.array([ne] * 4, np.int32)
    # each edge's halo layer (0..3); its edgesOnEdge lie within the next layer
    layer = np.searchsorted(edges_array, np.arange(ne), side="right")
    cells_on_edge = ints(1, nc + 1, (ne + 1, 2))
    cells_on_edge[rs.random_sample(ne + 1) < 1.0 / 12.0, 1] = nc + 1
    n_edges_on_edge = ints(6, me2 + 1, ne + 1)
    n_edges_on_edge[ne] = 0
    edges_on_edge = np.full((ne + 1, me2), ne + 1, np.int32)
    for e in range(ne):
        top = edges_array[min(layer[e] + 1, 3)]
        edges_on_edge[e, :n_edges_on_edge[e]] = ints(1, top + 1, n_edges_on_edge[e])
    n_edges_on_cell = np.array([5, 6, 6, 6, 7], np.int32)[rs.randint(0, 5, nc + 1)]
    n_edges_on_cell[nc] = 0
    edges_on_cell = np.full((nc + 1, me), ne + 1, np.int32)
    edge_sign_on_cell = np.zeros((nc + 1, me), np.int32)
    for c in range(nc):
        k = n_edges_on_cell[c]
        edges_on_cell[c, :k] = ints(1, ne + 1, k)
        edge_sign_on_cell[c, :k] = 2 * ints(0, 2, k) - 1
    mlet = ints(0, nl + 1, ne + 1)
    mlet[ne] = 0
    edge_mask = (np.arange(nl)[None, :] < mlet[:, None]).astype(np.int32)
    bottom = uniform(50.0, 4000.0, nc + 1)
    dims = {"nCells": nc, "nEdges": ne, "nVertLevels": nl, "maxEdges": me, "maxEdges2": me2}
    case = {"name": name, "precision": precision, "dims": dims, "config": config, "dt": 120.0}
    values = {
        "nVertLevels": nl, "nCells": nc, "nEdges": ne, "nCellsArray": cells_array, "nEdgesArray": edges_array,
        "nEdgesOnCell": n_edges_on_cell, "edgesOnCell": edges_on_cell, "edgeSignOnCell": edge_sign_on_cell,
        "areaCell": uniform(1.0e8, 8.0e8, nc + 1), "bottomDepth": bottom, "cellsOnEdge": cells_on_edge,
        "nEdgesOnEdge": n_edges_on_edge, "edgesOnEdge": edges_on_edge,
        "weightsOnEdge": uniform(-0.5, 0.5, (ne + 1, me2)), "fEdge": uniform(-1.4e-4, 1.4e-4, ne + 1),
        "dcEdge": uniform(1.0e4, 3.0e4, ne + 1), "dvEdge": uniform(5.0e3, 2.0e4, ne + 1), "edgeMask": edge_mask,
        "maxLevelEdgeTop": mlet, "refBottomDepthTopOfCell": np.linspace(0.0, 4000.0, nl + 1),
        "normalBarotropicVelocity": uniform(-0.3, 0.3, ne + 1),
        "normalBarotropicVelocity@2": uniform(-0.3, 0.3, ne + 1),
        "ssh": uniform(-2.0, 2.0, nc + 1), "sshSubcycle": uniform(-2.0, 2.0, nc + 1),
        "sshSubcycle@2": uniform(-2.0, 2.0, nc + 1),
        "normalBarotropicVelocitySubcycle": uniform(-0.3, 0.3, ne + 1),
        "normalBarotropicVelocitySubcycle@2": uniform(-0.3, 0.3, ne + 1),
        "normalVelocity@2": uniform(-0.6, 0.6, (ne + 1, nl)),
        "normalBaroclinicVelocity@2": uniform(-0.4, 0.4, (ne + 1, nl)),
        "barotropicForcing": uniform(-2.0e-5, 2.0e-5, ne + 1),
        "barotropicThicknessFlux": uniform(-50.0, 50.0, ne + 1),
        "normalTransportVelocity": uniform(-0.6, 0.6, (ne + 1, nl)),
        "normalGMBolusVelocity": uniform(-0.02, 0.02, (ne + 1, nl)),
        "layerThicknessEdge": uniform(2.0, 300.0, (ne + 1, nl)),
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
    """Run woof.hex.mpas_ocean.btr_subcycle_v842 on ``case`` as run_btr_subcycle.F90 drives the Fortran."""
    import cupy as cp
    from woof.hex.mpas_ocean import btr_subcycle_v842 as bt

    dev = _stream.on_device(FIELDS, case)
    s = dev["state"]

    def levels(name: str) -> list:
        return [s.get(name), s[f"{name}@2"]]

    state = {"normalBarotropicVelocity": levels("normalBarotropicVelocity"), "sshSubcycle": levels("sshSubcycle"),
             "normalBarotropicVelocitySubcycle": levels("normalBarotropicVelocitySubcycle"), "ssh": [s["ssh"]],
             "normalVelocity": [None, s["normalVelocity@2"]],
             "normalBaroclinicVelocity": [None, s["normalBaroclinicVelocity@2"]]}
    config = case["config"]
    bt.barotropic_stage(dt=case["dt"], n_btr_subcycles=int(config["nBtrSubcycles"]), mesh=dev["mesh"], state=state,
                        diagnostics=dev["diagnostics"], tend=dev["tend"], configs=config)
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
