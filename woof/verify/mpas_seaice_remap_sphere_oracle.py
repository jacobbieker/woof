"""Spherical cases for the MPAS-Seaice incremental-remap oracle.

The same driver, stages and grading as :mod:`woof.verify.mpas_seaice_remap_oracle`,
on :func:`woof.verify.mpas_seaice_sphere_mesh.sphere_mesh` blocks
(``on_a_sphere``): hexagon and quadrilateral blocks about the North Pole and
the South Pole with ``config_rotate_cartesian_grid`` (MPAS-Seaice's default:
the poles turned to the equator, so the gradient is rotated before it enters a
cell's plane), and at 53 degrees north without it; each mesh kind also comes
stretched (``varres``: cells several times larger at the block's rim than in
its middle). v8.4.2's init (verbatim) derives the remap's spherical geometry:
the local frames (``transGlobalToCell`` ...), the vertices in each cell's and
each edge's tangent plane (the side vertices in the neighbouring edge's frame),
the moments; that geometry is the GPU's input, as on the plane.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from woof.verify import mpas_seaice_remap_oracle as ro

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mpas_seaice_remap_sphere"
PRECISIONS = ro.PRECISIONS
_PONDS = {"pkgColumnTracerLevelIceActive": True, "pkgColumnTracerPondsActive": True,
          "pkgColumnTracerLidThicknessActive": True}
_UNROTATED = {"config_rotate_cartesian_grid": False}
#: ro.CASES' rows: (name, seed, stage, mesh kind, owned radius, packages, config overrides, field overrides)
CASES = (
    ("departure_sphere_hex_north", 20261921, 2, "hex", 3, {}, {}, {"sphere": "north"}),
    ("departure_sphere_quad_midlat_fast", 20261922, 2, "quad", 2, {}, _UNROTATED,
     {"sphere": "midlat", "speed": 1.3}),
    ("departure_sphere_hex_south_varres_fast", 20261923, 2, "hex", 3, {}, {},
     {"sphere": "south", "varres": True, "speed": 0.75}),
    ("departure_sphere_quad_north_varres", 20261924, 2, "quad", 3, {}, {}, {"sphere": "north", "varres": True}),
    ("reconstruct_sphere_hex_north_ponds", 20261931, 3, "hex", 3,
     {**_PONDS, "pkgColumnTracerIceAgeActive": True}, {}, {"sphere": "north"}),
    ("reconstruct_sphere_quad_midlat_snow_layers", 20261932, 3, "quad", 3, {"pkgTracerBrineActive": True},
     _UNROTATED, {"sphere": "midlat", "nSnowLayers": 2}),
    ("reconstruct_sphere_hex_south_varres", 20261933, 3, "hex", 3, {}, {}, {"sphere": "south", "varres": True}),
    ("remap_sphere_hex_north_three_steps", 20261941, 4, "hex", 2, {}, {"oracle_n_steps": 3},
     {"sphere": "north", "speed": 0.6}),
    ("remap_sphere_quad_midlat_ponds_recover", 20261942, 4, "quad", 1, _PONDS,
     {**_UNROTATED, "config_recover_tracer_means_check": True},
     {"sphere": "midlat", "speed": 1.2}),
    ("remap_sphere_hex_south_varres_recover", 20261943, 4, "hex", 1, {"pkgTracerBrineActive": True},
     {"config_recover_tracer_means_check": True, "oracle_n_steps": 2},
     {"sphere": "south", "varres": True, "nSnowLayers": 2}),
    ("remap_sphere_quad_north_varres", 20261944, 4, "quad", 2, {}, {"oracle_n_steps": 2},
     {"sphere": "north", "varres": True, "speed": 0.8}),
)


def cases(precision: str = "r4") -> list[dict]:
    return ro.cases(precision, CASES)


def load_fixture(precision: str) -> dict[str, dict[str, np.ndarray]]:
    return ro.load_fixture(precision, FIXTURE_DIR)


def load_logs() -> dict:
    return ro.load_logs(FIXTURE_DIR)


def measure(precision: str, reference: dict | None = None) -> dict[str, dict[str, int]]:
    return ro.measure(precision, reference, rows=CASES, fixture_dir=FIXTURE_DIR)
