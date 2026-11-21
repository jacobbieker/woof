"""Cases and the stream layout for the MPAS-Ocean GM bolus velocity oracle.

``tools/mpas_ocean_oracle/run_gm.F90`` drives MPAS-Model v8.4.2's
``ocn_gm_init`` and ``ocn_gm_compute_Bolus_velocity`` (``mpas_ocn_gm.F``
compiled verbatim) with gfortran -O0 -ffp-contract=off at RKIND 4 and 8. The
stream is :mod:`woof.verify.mpas_ocean_stream`'s.

The meshes are :func:`~woof.verify.mpas_ocean_hexmesh.hex_mesh` tori (a sixth
of the cells culled as land, so boundary edges have ``maxLevelEdgeTop`` 0),
the columns stratified with noise so that some interfaces are unstable (the
density derivative's ``-1e-12`` cap), some slopes reach the [-1, 1] clip and
the Brunt-Vaisala frequency takes both signs. Columns of one and two levels
leave edges whose stream function is not solved. Every output starts from
noise, so a word the routine does not write (the ``nVertLevels+1`` levels it
leaves) is checked too.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from woof.verify import mpas_ocean_stream as _stream
from woof.verify import mpas_ocean_tend_oracle as _tend
from woof.verify.mpas_ocean_hexmesh import hex_mesh
from woof.verify.mpas_ocean_tend_oracle import PRECISIONS

FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mpas_ocean_gm")
CASES_PER_PRECISION = True
REAL, INT, DIM, DIMS, SCRATCH = _stream.REAL, _stream.INT, _stream.DIM, _stream.DIMS, _stream.SCRATCH
FIELDS = (
    ("mesh", "nVertLevels", DIM, ()), ("mesh", "nCellsArray", DIMS, (4,)), ("mesh", "nEdgesArray", DIMS, (4,)),
    ("mesh", "maxLevelEdgeTop", INT, ("E1",)), ("mesh", "maxLevelCell", INT, ("C1",)),
    ("mesh", "cellsOnEdge", INT, (2, "E1")), ("mesh", "areaCell", REAL, ("C1",)), ("mesh", "dcEdge", REAL, ("E1",)),
    ("mesh", "dvEdge", REAL, ("E1",)), ("mesh", "nEdgesOnCell", INT, ("C1",)),
    ("mesh", "edgesOnCell", INT, ("ME", "C1")),
    ("diagnostics", "density", REAL, ("L", "C1")), ("diagnostics", "displacedDensity", REAL, ("L", "C1")),
    ("diagnostics", "zMid", REAL, ("L", "C1")), ("diagnostics", "layerThicknessEdge", REAL, ("L", "E1")),
    ("diagnostics", "BruntVaisalaFreqTop", REAL, ("L", "C1")), ("diagnostics", "boundaryLayerDepth", REAL, ("C1",)),
    ("diagnostics", "hEddyFlux", REAL, ("L", "E1")),
    ("diagnostics", "normalGMBolusVelocity", REAL, ("L", "E1")),
    ("diagnostics", "relativeSlopeTopOfEdge", REAL, ("LP1", "E1")),
    ("diagnostics", "relativeSlopeTopOfCell", REAL, ("LP1", "C1")),
    ("diagnostics", "relativeSlopeTapering", REAL, ("LP1", "E1")),
    ("diagnostics", "relativeSlopeTaperingCell", REAL, ("LP1", "C1")),
    ("diagnostics", "k33", REAL, ("LP1", "C1")),
    ("diagnostics", "gmStreamFuncTopOfEdge", REAL, ("LP1", "E1")),
    ("diagnostics", "gmStreamFuncTopOfCell", REAL, ("L", "C1")),
    ("scratch", "gradDensityEdge", SCRATCH, ("L", "E1")), ("scratch", "gradDensityTopOfEdge", SCRATCH, ("LP1", "E1")),
    ("scratch", "gradDensityConstZTopOfEdge", SCRATCH, ("LP1", "E1")),
    ("scratch", "dDensityDzTopOfCell", SCRATCH, ("LP1", "C1")),
    ("scratch", "dDensityDzTopOfEdge", SCRATCH, ("LP1", "E1")), ("scratch", "gradZMidEdge", SCRATCH, ("L", "E1")),
    ("scratch", "gradZMidTopOfEdge", SCRATCH, ("LP1", "E1")), ("scratch", "areaCellSum", SCRATCH, ("L", "C1")),
)
#: v8.4.2's Registry defaults, GM on with a bolus and a Redi coefficient
BASE_CONFIG = {
    "config_density0": 1026.0, "config_use_standardGM": True, "config_gravWaveSpeed_trunc": 0.3,
    "config_standardGM_tracer_kappa": 600.0, "config_max_relative_slope": 0.01, "config_Redi_kappa": 400.0,
    "config_disable_redi_k33": False, "config_use_Redi_surface_layer_tapering": False,
    "config_Redi_surface_layer_tapering_extent": 0.0, "config_use_Redi_bottom_layer_tapering": False,
    "config_Redi_bottom_layer_tapering_depth": 0.0,
}
#: (name, seed, torus, nVertLevels, overrides); ``_min_levels`` is the shallowest column,
#: ``_steep`` the share of noise in the horizontal density differences, ``_noise`` the
#: level-to-level density noise (large enough to make interfaces unstable)
CASES = (
    ("default", 20261121, (7, 6), 14, {}),
    ("tapering", 20261122, (6, 6), 18, {"config_use_Redi_surface_layer_tapering": True,
                                        "config_Redi_surface_layer_tapering_extent": 30.0,
                                        "config_use_Redi_bottom_layer_tapering": True,
                                        "config_Redi_bottom_layer_tapering_depth": 60.0,
                                        "config_max_relative_slope": 0.3}),
    ("steep_no_k33", 20261123, (5, 7), 10, {"config_disable_redi_k33": True, "config_gravWaveSpeed_trunc": 1.2,
                                            "config_standardGM_tracer_kappa": 1800.0, "_steep": 40.0,
                                            "_noise": 0.2}),
    ("shallow", 20261124, (6, 5), 6, {"_min_levels": 1, "config_max_relative_slope": 0.05, "_noise": 0.6}),
)


def make_case(name: str, seed: int, torus: tuple[int, int], n_levels: int, overrides: dict,
              precision: str = "r4") -> dict:
    """One deterministic case on a hex torus with an ocean's magnitudes."""
    rs = np.random.RandomState(seed)
    real = PRECISIONS[precision]
    mesh = hex_mesh(*torus, 1.0 / 6.0, n_levels, rs, min_levels=overrides.get("_min_levels", 2))
    nc, ne, nl, me = mesh["nCells"], mesh["nEdges"], n_levels, mesh["maxEdges"]
    config = dict(BASE_CONFIG, **{k: v for k, v in overrides.items() if not k.startswith("_")})

    def uniform(lo, hi, shp):
        return lo + (hi - lo) * rs.random_sample(shp)

    thickness = uniform(4.0, 60.0, (nc + 1, nl))
    depth_mid = np.cumsum(thickness, axis=1) - 0.5 * thickness
    z_mid = uniform(-0.4, 0.4, (nc + 1, 1)) - depth_mid
    steep, noise = overrides.get("_steep", 1.0), overrides.get("_noise", 0.03)
    density = (1024.0 + 0.004 * depth_mid + steep * uniform(-0.02, 0.02, (nc + 1, 1))
               + uniform(-noise, noise, (nc + 1, nl)))
    displaced = density + uniform(0.0, 0.01, (nc + 1, nl))
    c1, c2 = mesh["cellsOnEdge"][:, 0] - 1, mesh["cellsOnEdge"][:, 1] - 1
    thickness_edge = 0.5 * (thickness[c1] + thickness[c2]) * uniform(0.95, 1.05, (ne + 1, nl))
    dims = {"nCells": nc, "nEdges": ne, "nVertLevels": nl, "maxEdges": me}
    case = {"name": name, "precision": precision, "dims": dims, "config": config, "dt": 1800.0}
    values = {
        **{k: v for k, v in mesh.items()}, "nVertLevels": nl,
        "areaCell": uniform(1.0e8, 8.0e8, nc + 1), "dcEdge": uniform(1.0e4, 3.0e4, ne + 1),
        "dvEdge": uniform(5.0e3, 2.0e4, ne + 1), "density": density, "displacedDensity": displaced,
        "zMid": z_mid, "layerThicknessEdge": thickness_edge,
        "BruntVaisalaFreqTop": uniform(-2.0e-5, 1.0e-4, (nc + 1, nl)),
        "boundaryLayerDepth": uniform(5.0, 90.0, nc + 1),
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
    """Run woof.hex.mpas_ocean.gm_v842 on ``case`` as run_gm.F90 drives the Fortran."""
    import cupy as cp
    from woof.hex.mpas_ocean import gm_v842

    dev = _stream.on_device(FIELDS, case)
    gm_v842.compute_bolus_velocity(mesh=dev["mesh"], diagnostics=dev["diagnostics"], configs=case["config"])
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
