"""Spherical cases for the MPAS-Seaice time-step oracle with incremental-remap advection.

:mod:`woof.verify.mpas_seaice_step_remap_oracle`'s driver and grading on
:func:`woof.verify.mpas_seaice_sphere_mesh.sphere_mesh` blocks: the velocity
solver, then each dynamics subcycle the spherical remap (the init's geometry is
the GPU's input), over N steps; one block about the North Pole with
``config_rotate_cartesian_grid``, one stretched (variable-resolution) block at
53 degrees north without it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from woof.verify import mpas_seaice_step_remap_oracle as sr

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mpas_seaice_step_remap_sphere"
PRECISIONS = sr.PRECISIONS
#: sr.CASES' rows, with the remap's field overrides naming the sphere
CASES = (
    ("remap_step_sphere_hex_north", 20261951, "hex", 2, 3, 1800.0, {}, {}, {}, {"sphere": "north"}),
    ("remap_step_sphere_quad_midlat_varres_two_subcycles", 20261952, "quad", 2, 2, 3600.0, {
        "pkgColumnTracerLevelIceActive": True, "pkgColumnTracerPondsActive": True}, {
        "config_revised_evp": True, "config_dynamics_subcycle_number": 2, "config_elastic_subcycle_number": 12},
     {"config_rotate_cartesian_grid": False, "config_recover_tracer_means_check": True},
     {"sphere": "midlat", "varres": True}),
)


def cases(precision: str = "r4") -> list[dict]:
    return sr.cases(precision, CASES)


def load_fixture(precision: str) -> dict[str, dict[str, np.ndarray]]:
    return sr.load_fixture(precision, FIXTURE_DIR)


def measure(precision: str, reference: dict | None = None) -> dict[str, dict[str, int]]:
    return sr.measure(precision, reference, rows=CASES, fixture_dir=FIXTURE_DIR)
