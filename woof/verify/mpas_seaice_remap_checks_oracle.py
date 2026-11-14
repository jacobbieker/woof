"""The incremental remap's optional checks against the MPAS-Seaice v8.4.2 oracle.

The same driver, stage (4: ``seaice_run_advection_incremental_remap`` over N
steps) and grading as :mod:`woof.verify.mpas_seaice_remap_oracle`, with
``config_conservation_check`` and ``config_monotonicity_check`` on:

- the conservation check: ``sum_tracers`` before and after the update (the
  ``tracer_conservation`` pool's ``<tracer>Cons`` and ``Cons@2``), then
  ``check_tracer_conservation`` (one rank: ``mpas_dmpar_sum_real`` hands each
  sum back). Its threshold is eps11 relative, which RKIND 4 sums all but
  never meet, so at RKIND 4 it fails (``SEAICE_ERROR_IR_CONS``) and the run
  stops with the thicknesses not yet back to volume;
- the monotonicity check: ``tracer_local_min_max`` before the remap (masks at
  threshold 0), then ``check_tracer_monotonicity``'s extension of the extremes
  by one ring (in place, cell after cell) and its test (``SEAICE_ERROR_IR_MONO``
  on the first value outside them by more than eps11 * max(1, |extreme|)).

On a block with halo cells ice crosses the owned cells' rim, so neither
check holds there (the sums are the owned cells', the halo's local extremes
are left at zero: a one-rank global mesh has no halo); the ``closed`` cases
keep the rim still, so at RKIND 8 both checks pass.

The oracle records the MPAS_LOG_ERR messages (``remap-logs.json``'s
``errors``) and the error code; the GPU's are graded against them.

Planar hexagon and quadrilateral blocks, and spherical ones (about the North
and South Poles with ``config_rotate_cartesian_grid``, at 53 degrees north
without it), two of them stretched to variable resolution.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from woof.verify import mpas_seaice_remap_oracle as ro

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mpas_seaice_remap_checks"
PRECISIONS = ro.PRECISIONS
_PONDS = {"pkgColumnTracerLevelIceActive": True, "pkgColumnTracerPondsActive": True,
          "pkgColumnTracerLidThicknessActive": True}
_CONS = {"config_conservation_check": True}
_MONO = {"config_monotonicity_check": True}
_BOTH = {**_CONS, **_MONO}
#: ro.CASES' rows: (name, seed, stage, mesh kind, owned radius, packages, config overrides, field overrides)
CASES = (
    ("checks_hex_conservation", 20262041, 4, "hex", 2, {}, {**_CONS, "oracle_n_steps": 2}, {"speed": 0.6}),
    ("checks_hex_closed_both_recover", 20262047, 4, "hex", 2, {},
     {**_BOTH, "config_recover_tracer_means_check": True, "oracle_n_steps": 2}, {"speed": 0.6, "closed": True}),
    ("checks_quad_ponds_monotonicity", 20262042, 4, "quad", 2, _PONDS, {**_MONO, "oracle_n_steps": 2},
     {"speed": 1.2}),
    ("checks_quad_closed_ponds_both", 20262048, 4, "quad", 2, _PONDS, {**_BOTH, "oracle_n_steps": 2},
     {"speed": 1.2, "closed": True}),
    ("checks_hex_snow_layers_both_recover", 20262043, 4, "hex", 1, {"pkgTracerBrineActive": True},
     {**_BOTH, "config_recover_tracer_means_check": True}, {"nSnowLayers": 2}),
    ("checks_sphere_hex_north_closed_both", 20262044, 4, "hex", 2, {}, {**_BOTH, "oracle_n_steps": 2},
     {"sphere": "north", "speed": 0.6, "closed": True}),
    ("checks_sphere_quad_midlat_varres_monotonicity", 20262045, 4, "quad", 2, _PONDS,
     {**_MONO, "config_rotate_cartesian_grid": False}, {"sphere": "midlat", "varres": True, "speed": 1.2}),
    ("checks_sphere_hex_south_varres_closed_both", 20262046, 4, "hex", 2, {"pkgTracerBrineActive": True},
     {**_BOTH, "oracle_n_steps": 2}, {"sphere": "south", "varres": True, "nSnowLayers": 2, "closed": True}),
)


def cases(precision: str = "r4") -> list[dict]:
    return ro.cases(precision, CASES)


def load_fixture(precision: str) -> dict[str, dict[str, np.ndarray]]:
    return ro.load_fixture(precision, FIXTURE_DIR)


def load_logs() -> dict:
    return ro.load_logs(FIXTURE_DIR)


def measure(precision: str, reference: dict | None = None) -> dict[str, dict[str, int]]:
    return ro.measure(precision, reference, rows=CASES, fixture_dir=FIXTURE_DIR)
