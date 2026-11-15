"""The incremental remap's init geometry, computed by the port, against the MPAS-Seaice v8.4.2 oracle.

The same driver and grading as :mod:`woof.verify.mpas_seaice_remap_oracle`,
built as ``run_incremental_remap_init_{r4,r8}``: MPAS's own RBF set-up
(``mpas_initialize_vectors``, ``mpas_init_reconstruct`` and the routines they
call, cut verbatim out of ``src/operators``) replaces the stand-ins, so
``seaice_init_advection_incremental_remap`` derives every word of the remap's
geometry itself, ``coeffs_reconstruct`` included (owned cells; the halo keeps
the zeros, as on one rank). The GPU side takes none of it from the fixture:
:mod:`woof.hex.mpas_seaice.remap_init_v842` computes it on the host and every
geometry field is graded, with the remap run on it.

Cases (``init`` in the field overrides): stage 5 (the init alone) and stage 4
(the remap over N steps on the port's own geometry), on planar hexagon and
quadrilateral blocks and on spherical ones (about either pole with
``config_rotate_cartesian_grid``, at 53 degrees north without it), uniform
and stretched to variable resolution.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from woof.verify import mpas_seaice_remap_oracle as ro

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mpas_seaice_remap_init"
PRECISIONS = ro.PRECISIONS
_UNROTATED = {"config_rotate_cartesian_grid": False}
_PONDS = {"pkgColumnTracerLevelIceActive": True, "pkgColumnTracerPondsActive": True,
          "pkgColumnTracerLidThicknessActive": True}
#: ro.CASES' rows: (name, seed, stage, mesh kind, owned radius, packages, config overrides, field overrides)
CASES = (
    ("init_hex", 20262101, 5, "hex", 3, {}, {}, {"init": True}),
    ("init_quad", 20262102, 5, "quad", 3, {}, {}, {"init": True}),
    ("init_sphere_hex_north", 20262103, 5, "hex", 3, {}, {}, {"init": True, "sphere": "north"}),
    ("init_sphere_quad_south_varres", 20262104, 5, "quad", 3, {}, {},
     {"init": True, "sphere": "south", "varres": True}),
    ("init_sphere_hex_midlat_varres", 20262105, 5, "hex", 3, {}, _UNROTATED,
     {"init": True, "sphere": "midlat", "varres": True}),
    ("init_sphere_quad_midlat", 20262106, 5, "quad", 2, {}, _UNROTATED, {"init": True, "sphere": "midlat"}),
    ("remap_init_hex_two_steps", 20262141, 4, "hex", 2, {}, {"oracle_n_steps": 2}, {"init": True, "speed": 0.6}),
    ("remap_init_quad_ponds", 20262142, 4, "quad", 2, _PONDS, {}, {"init": True, "speed": 1.2}),
    ("remap_init_sphere_hex_south_varres_recover", 20262143, 4, "hex", 2, {"pkgTracerBrineActive": True},
     {"config_recover_tracer_means_check": True, "oracle_n_steps": 2},
     {"init": True, "sphere": "south", "varres": True, "nSnowLayers": 2}),
    ("remap_init_sphere_quad_midlat_varres", 20262144, 4, "quad", 2, {}, _UNROTATED,
     {"init": True, "sphere": "midlat", "varres": True, "speed": 0.8}),
)


def cases(precision: str = "r4") -> list[dict]:
    return ro.cases(precision, CASES)


def load_fixture(precision: str) -> dict[str, dict[str, np.ndarray]]:
    return ro.load_fixture(precision, FIXTURE_DIR)


def load_logs() -> dict:
    return ro.load_logs(FIXTURE_DIR)


def measure(precision: str, reference: dict | None = None) -> dict[str, dict[str, int]]:
    return ro.measure(precision, reference, rows=CASES, fixture_dir=FIXTURE_DIR)
