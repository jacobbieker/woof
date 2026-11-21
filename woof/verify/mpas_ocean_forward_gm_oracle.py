"""Cases for the MPAS-Ocean forward mode with GM and Redi on.

The driver is the forward oracle's (``tools/mpas_ocean_oracle/run_forward.F90``,
:mod:`woof.verify.mpas_ocean_forward_oracle`) with ``config_use_standardGM``
on: the cold start sets the boundary layer depth to half the top layer,
computes GM's bolus velocity, adds it to the transport velocity and
reconstructs GM's vectors (``mpas_ocn_init_routines.F``); each step then
carries GM and Redi as :mod:`woof.verify.mpas_ocean_split_step_gm_oracle`
describes. The field table is the forward oracle's with the GM fields of
:func:`~woof.verify.mpas_ocean_split_step_gm_oracle.field_table`. The meshes
are single blocks: v8.4.2's slopes read GM's ``huge`` scratch fill on the
outermost computed halo edges, which a nested-halo block would carry into
its halo columns.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from woof.verify import mpas_ocean_forward_oracle as _fw
from woof.verify import mpas_ocean_split_step_gm_oracle as _ssgm
from woof.verify import mpas_ocean_stream as _stream
from woof.verify import mpas_ocean_tend_oracle as _tend

FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mpas_ocean_forward_gm")
CASES_PER_PRECISION = True
REAL = _stream.REAL
FIELDS = _ssgm.field_table(_fw.FIELDS)
_ON = _ssgm.GM_ON
#: (name, seed, (nq, nr), nVertLevels, steps, config_dt, config_btr_dt, config overrides)
CASES = (
    ("gm_split_three_steps", 20261701, (7, 6), 8, 3, "00:02:00", "0000_00:00:50", {**_ON}),
    ("gm_split_jm_rich_tapering", 20261702, (8, 6), 10, 2, "00:03:00", "0000_00:01:00", {
        **_ON, "config_eos_type": "jm", "config_pressure_gradient_type": "Jacobian_from_TS",
        "config_use_rich_visc": True, "config_use_rich_diff": True, "config_compute_active_tracer_budgets": True,
        "config_use_tracer_del2": True, "config_tracer_del2": 50.0,
        "config_use_Redi_surface_layer_tapering": True, "config_Redi_surface_layer_tapering_extent": 20.0,
        "config_use_Redi_bottom_layer_tapering": True, "config_Redi_bottom_layer_tapering_depth": 60.0,
        "config_max_relative_slope": 0.3}),
    ("gm_unsplit_no_k33", 20261703, (6, 5), 6, 2, "00:01:00", "0000_00:00:20", {
        **_ON, "config_time_integrator": "unsplit_explicit", "config_use_const_visc": True,
        "config_use_const_diff": True, "config_disable_redi_k33": True}),
)


def make_case(name, seed, torus, n_levels, steps, dt, btr_dt, overrides, precision="r4") -> dict:
    """The forward oracle's cold start with GM's fields added as noise (from
    their own seed): the cold start computes every one it reads."""
    case = _fw.make_case(name, seed, torus, n_levels, steps, dt, btr_dt, overrides, precision)
    _ssgm.add_gm_inputs(case, np.random.RandomState(seed + 7), previous_step=False)
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


def run_gpu(case: dict, n_steps: int | None = None) -> dict[str, np.ndarray]:
    """Run woof.hex.mpas_ocean.forward_v842 on ``case`` as run_forward.F90 drives
    the Fortran; every array by the allocation it was given to."""
    import cupy as cp
    from woof.hex.mpas_ocean import forward_v842 as fw

    dev = _stream.on_device(FIELDS, case)
    given = {(pool, name): dev[pool][name] for pool, name, kind, _ in FIELDS
             if kind in (REAL, _stream.INT) and dev[pool][name] is not None}
    config = case["config"]
    clock = fw.cold_start(pools=dev, configs=config)
    fw.advance(pools=dev, configs=config, clock=clock,
               n_steps=int(config["oracle_n_steps"]) if n_steps is None else n_steps)
    return {f"{pool}/{name}": cp.asnumpy(array) for (pool, name), array in given.items()}


def runs(case: dict) -> bool:
    return _fw.runs(case)


def measure(precision: str, reference: dict | None = None) -> dict[str, dict[str, int]]:
    reference = load_fixture(precision) if reference is None else reference
    report: dict[str, dict[str, int]] = {}
    for case in cases(precision):
        if not runs(case):
            continue
        got = run_gpu(case)
        want = reference[case["name"]]
        report[case["name"]] = {name: word_mismatches(got[name], want[name]) for name in want}
    return report
