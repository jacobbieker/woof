"""Cases for one MPAS-Ocean split-explicit step with GM and Redi on.

The driver is the split step oracle's (``tools/mpas_ocean_oracle/run_split_step.F90``,
:mod:`woof.verify.mpas_ocean_split_step_oracle`) with ``config_use_standardGM``
on: stage 2's transport adds the bolus velocity the input carries (the
previous step's), the tracer tendency runs Redi on the input's slopes, GM's
bolus velocity is computed before the implicit vertical mixing (which adds
its ``k33``) and again after the last diagnostics, and
``ocn_reconstruct_gm_vectors`` ends the step; ``mpas_ocn_gm.F`` compiled
verbatim. The field table is the split step's with the slope and tapering
fields at their Registry size (``nVertLevels+1``) and GM's other fields,
the vectors on cells and GM's and Redi's scratch fields added.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from woof.verify import mpas_ocean_split_step_oracle as _ss
from woof.verify import mpas_ocean_stream as _stream
from woof.verify import mpas_ocean_tend_oracle as _tend
from woof.verify.mpas_ocean_tend_oracle import PRECISIONS

FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mpas_ocean_split_step_gm")
CASES_PER_PRECISION = True
REAL, INT, SCRATCH = _stream.REAL, _stream.INT, _stream.SCRATCH
#: the fields the split step's table carries at nVertLevels (GM off reads none of them)
P1_FIELDS = {("diagnostics", "relativeSlopeTopOfEdge"): ("LP1", "E1"),
             ("diagnostics", "relativeSlopeTapering"): ("LP1", "E1"),
             ("diagnostics", "relativeSlopeTaperingCell"): ("LP1", "C1")}
_VECTORS = tuple(("diagnostics", f"{stem}{axis}", REAL, (level, "C1"))
                 for stem, level in (("transportVelocity", "L"), ("GMBolusVelocity", "L"),
                                     ("relativeSlopeTopOfCell", "LP1"), ("GMStreamFunc", "LP1"))
                 for axis in ("X", "Y", "Z", "Zonal", "Meridional"))
GM_FIELDS = (
    ("diagnostics", "boundaryLayerDepth", REAL, ("C1",)),
    ("diagnostics", "hEddyFlux", REAL, ("L", "E1")),
    ("diagnostics", "relativeSlopeTopOfCell", REAL, ("LP1", "C1")),
    ("diagnostics", "gmStreamFuncTopOfEdge", REAL, ("LP1", "E1")),
    ("diagnostics", "gmStreamFuncTopOfCell", REAL, ("L", "C1")),
    *_VECTORS,
    ("scratch", "gradDensityEdge", SCRATCH, ("L", "E1")), ("scratch", "gradDensityTopOfEdge", SCRATCH, ("LP1", "E1")),
    ("scratch", "gradDensityConstZTopOfEdge", SCRATCH, ("LP1", "E1")),
    ("scratch", "dDensityDzTopOfCell", SCRATCH, ("LP1", "C1")),
    ("scratch", "dDensityDzTopOfEdge", SCRATCH, ("LP1", "E1")), ("scratch", "gradZMidEdge", SCRATCH, ("L", "E1")),
    ("scratch", "gradZMidTopOfEdge", SCRATCH, ("LP1", "E1")), ("scratch", "areaCellSum", SCRATCH, ("L", "C1")),
    ("scratch", "gradTracerEdge", SCRATCH, ("L", "E1")), ("scratch", "gradTracerTopOfEdge", SCRATCH, ("LP1", "E1")),
    ("scratch", "gradHTracerSlopedTopOfCell", SCRATCH, ("LP1", "C1")),
    ("scratch", "dTracerdZTopOfCell", SCRATCH, ("LP1", "C1")),
    ("scratch", "dTracerdZTopOfEdge", SCRATCH, ("LP1", "E1")),
)


def field_table(base: tuple) -> tuple:
    """``base`` with the slope fields at ``nVertLevels+1`` and :data:`GM_FIELDS` added."""
    return tuple((p, n, k, P1_FIELDS.get((p, n), f)) for p, n, k, f in base) + GM_FIELDS


FIELDS = field_table(_ss.FIELDS)
GM_ON = {"config_use_standardGM": True, "config_standardGM_tracer_kappa": 600.0, "config_Redi_kappa": 400.0}
#: (name, seed, nCells, nVertLevels, config overrides)
CASES = (
    ("gm_split_defaults", 20261601, 40, 8, {**GM_ON}),
    ("gm_split_jm_rich_tapering_budgets", 20261602, 44, 10, {
        **GM_ON, "config_eos_type": "jm", "config_pressure_gradient_type": "Jacobian_from_TS",
        "config_use_rich_visc": True, "config_use_rich_diff": True, "config_compute_active_tracer_budgets": True,
        "config_use_tracer_del2": True, "config_tracer_del2": 50.0, "config_n_ts_iter": 3,
        "config_use_Redi_surface_layer_tapering": True, "config_Redi_surface_layer_tapering_extent": 30.0,
        "config_use_Redi_bottom_layer_tapering": True, "config_Redi_bottom_layer_tapering_depth": 60.0,
        "config_max_relative_slope": 0.3, "config_gravWaveSpeed_trunc": 1.2, "config_standardGM_tracer_kappa": 1800.0}),
    ("gm_unsplit_const_no_k33_term1_off", 20261603, 32, 6, {
        **GM_ON, "config_time_integrator": "unsplit_explicit", "config_use_const_visc": True,
        "config_use_const_diff": True, "config_vel_correction": False, "config_n_bcl_iter_mid": 1,
        "config_disable_redi_k33": True, "config_disable_redi_horizontal_term1": True}),
)


def add_gm_inputs(case: dict, rs: np.random.RandomState, *, previous_step: bool) -> None:
    """The slope fields at ``nVertLevels+1`` and :data:`GM_FIELDS` in ``case``'s
    pools: noise (words the run must overwrite or keep), or with
    ``previous_step`` the slopes, tapering and bolus velocity a previous step's
    GM leaves, which this step's Redi and stage 2 read first."""
    real = PRECISIONS[case["precision"]]
    pools = case["pools"]
    d = pools["diagnostics"]

    def uniform(lo, hi, shp):
        return lo + (hi - lo) * rs.random_sample(shp)

    for (pool, field), fshape in P1_FIELDS.items():
        pools[pool][field] = uniform(-1.0, 1.0, _stream.shape(case, fshape)).astype(real)
    for pool, field, kind, fshape in GM_FIELDS:
        pools.setdefault(pool, {})[field] = (None if kind == SCRATCH else
                                             uniform(-1.0, 1.0, _stream.shape(case, fshape)).astype(real))
    if previous_step:
        mlet = pools["mesh"]["maxLevelEdgeTop"]
        nl = case["dims"]["nVertLevels"]
        wet = np.arange(nl)[None, :] < mlet[:, None]
        d["relativeSlopeTopOfEdge"] = uniform(-0.05, 0.05, d["relativeSlopeTopOfEdge"].shape).astype(real)
        d["relativeSlopeTapering"] = uniform(0.0, 1.0, d["relativeSlopeTapering"].shape).astype(real)
        d["normalGMBolusVelocity"] = (uniform(-0.02, 0.02, wet.shape) * wet).astype(real)
        d["boundaryLayerDepth"] = uniform(5.0, 60.0, d["boundaryLayerDepth"].shape).astype(real)


def make_case(name: str, seed: int, n_cells: int, n_levels: int, overrides: dict, precision: str = "r4") -> dict:
    """The split step oracle's case with GM's inputs added (from their own seed)."""
    case = _ss.make_case(name, seed, n_cells, n_levels, overrides, precision)
    add_gm_inputs(case, np.random.RandomState(seed + 7), previous_step=True)
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
    """Run woof.hex.mpas_ocean.split_step_v842 on ``case`` as run_split_step.F90
    drives the Fortran: the starting diagnostics, then the step."""
    import cupy as cp
    from woof.hex.mpas_ocean import diagnostics_v842 as dg
    from woof.hex.mpas_ocean import split_step_v842 as ss

    dev = _stream.on_device(FIELDS, case)
    config = case["config"]
    st, tr = dev["state"], dev["tracers"]
    dg.diagnostic_solve(dt=case["dt"], mesh=dev["mesh"],
                        state={n: st[n] for n in ("layerThickness", "normalVelocity", "ssh")},
                        tracers={n: tr[n] for n in ("activeTracers", "index_temperature", "index_salinity")},
                        diagnostics=dev["diagnostics"], forcing=dev["forcing"], configs=config)
    ss.split_step(dt=case["dt"], n_btr_subcycles=int(config["nBtrSubcycles"]), pools=dev, configs=config)
    return {f"{pool}/{name}": cp.asnumpy(dev[pool][name]) for pool, name, kind, _ in FIELDS
            if kind in (REAL, INT) and dev[pool][name] is not None}


def runs(case: dict) -> bool:
    return _ss.runs(case)


def measure(precision: str, reference: dict | None = None) -> dict[str, dict[str, int]]:
    """Word mismatches per case and field, over the cases this checkout runs."""
    reference = load_fixture(precision) if reference is None else reference
    report: dict[str, dict[str, int]] = {}
    for case in cases(precision):
        if not runs(case):
            continue
        got = run_gpu(case)
        want = reference[case["name"]]
        report[case["name"]] = {name: word_mismatches(got[name], want[name]) for name in want}
    return report
