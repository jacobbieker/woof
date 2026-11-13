"""Cases and the stream layout for the MPAS-Ocean forward-mode oracle.

``tools/mpas_ocean_oracle/run_forward.F90`` drives MPAS-Model v8.4.2's forward
mode over N time steps: ``ocn_time_integration_split_init`` and
``ocn_timestep`` (cut verbatim), the state part of
``ocn_init_routines_block``, and per step the absorbed fractions, the step and
the time-level shift. Every module the step reaches is compiled verbatim with
gfortran -O0 -ffp-contract=off at RKIND 4 and 8. The stream is
:mod:`woof.verify.mpas_ocean_stream`'s, the field table the split step
oracle's (:mod:`woof.verify.mpas_ocean_split_step_oracle`) plus the clock's
fields.

The output is every array by the allocation it was given to: after an odd
number of steps the arrays named ``NAME`` and ``NAME@2`` hold the other time
level, as MPAS's pointer shift leaves them.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from woof.verify import mpas_ocean_split_step_oracle as _ss
from woof.verify import mpas_ocean_stream as _stream
from woof.verify import mpas_ocean_tend_oracle as _tend
from woof.verify.mpas_ocean_tend_oracle import PRECISIONS

FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mpas_ocean_forward")
CASES_PER_PRECISION = True
REAL = _stream.REAL
FIELDS = _ss.FIELDS + (("diagnostics", "daysSinceStartOfSim", REAL, ()),)
CLOCK_CONFIG = {
    "config_start_time": "0001-01-01_00:00:00", "diagnostics:simulationStartTime": "0001-01-01_00:00:00",
    "diagnostics:xtime": "0001-01-01_00:00:00", "config_do_restart": False,
}
#: (name, seed, nCells, nVertLevels, steps, config_dt, config_btr_dt, config overrides)
CASES = (
    ("split_three_steps", 20261501, 36, 8, 3, "00:02:00", "0000_00:00:50", {}),
    ("split_jm_rich_bulk", 20261502, 40, 10, 4, "00:03:00", "0000_00:01:00", {
        "config_eos_type": "jm", "config_pressure_gradient_type": "Jacobian_from_TS",
        "config_use_rich_visc": True, "config_use_rich_diff": True,
        "config_use_activeTracers_surface_bulk_forcing": True, "config_use_bulk_thickness_flux": True,
        "config_sw_absorption_type": "jerlov", "config_compute_active_tracer_budgets": True,
        "config_use_tracer_del2": True, "config_tracer_del2": 50.0}),
    ("unsplit_two_steps", 20261503, 28, 6, 2, "00:01:00", "0000_00:00:20", {
        "config_time_integrator": "unsplit_explicit", "config_use_const_visc": True,
        "config_use_const_diff": True}),
    ("split_filter_btr_mode", 20261504, 32, 7, 2, "00:02:00", "0000_00:00:40", {
        "config_filter_btr_mode": True, "config_start_time": "2000-02-28_23:58:00",
        "diagnostics:simulationStartTime": "2000-01-01_00:00:00"}),
)


def interval_seconds(text: str) -> int:
    """An MPAS interval string '[DDDD_]hh:mm:ss' in seconds."""
    days, _, clock = text.rpartition("_")
    h, m, s = (int(x) for x in clock.split(":"))
    return ((int(days or 0) * 24 + h) * 60 + m) * 60 + s


def make_case(name, seed, n_cells, n_levels, steps, dt, btr_dt, overrides, precision="r4") -> dict:
    case = _ss.make_case(name, seed, n_cells, n_levels, {}, precision)
    config = {**case["config"], **CLOCK_CONFIG, **overrides}
    config.update({"config_dt": dt, "config_btr_dt": btr_dt, "oracle_n_steps": steps})
    case["config"] = config
    case["dt"] = float(interval_seconds(dt))
    case["pools"]["diagnostics"]["daysSinceStartOfSim"] = np.asarray(0.0, PRECISIONS[precision])
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
