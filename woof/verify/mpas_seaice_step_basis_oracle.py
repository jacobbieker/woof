"""Cases for the MPAS-Seaice time-step oracle with the variational basis computed in the init.

:mod:`woof.verify.mpas_seaice_step_remap_oracle`'s driver and grading, with
``oracle_compute_basis``: ``run_seaice_step.F90`` runs
``seaice_init_velocity_solver_variational`` (verbatim) after
``seaice_init_evp``, as v8.4.2's ``seaice_init_velocity_solver`` does, so the
basis gradients and integrals, ``tanLatVertexRotatedOverRadius`` and
``cellVerticesAtVertex`` come from the mesh (their inputs are noise); then N
steps with the velocity solver and incremental remap. The GPU side runs
:func:`woof.hex.mpas_seaice.timestep_v842.initialize` with ``basis="compute"``
(the remap's init geometry is the input, as in the step-remap oracle).

On planar and spherical blocks, both bases, Dunavant and Fekete rules, the
grid rotation on and off and the metric terms on (sphere) and off. At RKIND 8
the metric terms are glibc's binary64 ``asin``/``tan``: the cases with them
run with ``metric_libm="host"`` and are word-exact on a glibc 2.44 host (see
:data:`NEEDS_GLIBC`).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from woof.verify import mpas_seaice_remap_oracle as ro
from woof.verify import mpas_seaice_step_remap_oracle as sr
from woof.verify.mpas_seaice_sphere_mesh import RADIUS

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mpas_seaice_step_basis"
PRECISIONS = sr.PRECISIONS


def _basis(basis, kind, order, metric, sphere):
    return {"oracle_compute_basis": True, "config_variational_basis": basis,
            "config_wachspress_integration_type": kind, "config_wachspress_integration_order": order,
            "config_include_metric_terms": metric, "mesh:sphere_radius": RADIUS if sphere else 0.0}


#: sr.CASES' rows (the remap's field overrides naming the sphere), and the basis configs
CASES = (
    ("basis_step_hex_wachspress", 20262101, "hex", 2, 3, 1800.0, {}, {}, {}, {},
     _basis("wachspress", "dunavant", 8, False, False)),
    ("basis_step_quad_pwl_two_subcycles", 20262102, "quad", 2, 2, 3600.0, {
        "pkgColumnTracerLevelIceActive": True}, {
        "config_revised_evp": True, "config_dynamics_subcycle_number": 2, "config_elastic_subcycle_number": 12},
     {}, {}, _basis("pwl", "dunavant", 8, False, False)),
    ("basis_step_sphere_hex_north_wachspress_metric", 20262103, "hex", 2, 3, 1800.0, {}, {}, {},
     {"sphere": "north"}, _basis("wachspress", "dunavant", 8, True, True)),
    ("basis_step_sphere_quad_midlat_varres_pwl_unrotated", 20262104, "quad", 2, 2, 3600.0, {}, {},
     {"config_rotate_cartesian_grid": False}, {"sphere": "midlat", "varres": True},
     _basis("pwl", "dunavant", 8, True, True)),
    ("basis_step_sphere_hex_south_fekete6_no_metric", 20262105, "hex", 2, 2, 1800.0, {}, {}, {},
     {"sphere": "south"}, _basis("wachspress", "fekete", 6, False, True)),
)
#: the cases whose RKIND 8 words are glibc 2.44's asin and tan (the metric terms)
NEEDS_GLIBC = tuple(row[0] for row in CASES if row[-1]["config_include_metric_terms"])


def make_case(*row, precision="r4") -> dict:
    *step_row, basis = row
    case = sr.make_case(*step_row, precision=precision)
    case["config"] = {**case["config"], **basis}
    return case


def cases(precision: str = "r4", rows=CASES) -> list[dict]:
    return [make_case(*row, precision=precision) for row in rows]


def load_fixture(precision: str) -> dict[str, dict[str, np.ndarray]]:
    return sr.load_fixture(precision, FIXTURE_DIR)


def run_gpu(case: dict, fixture: dict[str, np.ndarray], n_steps: int | None = None, *,
            metric_libm: str | None = "host") -> dict[str, np.ndarray]:
    """Run woof.hex.mpas_seaice.timestep_v842 on ``case`` as run_seaice_step.F90
    drives the Fortran, the basis computed in the init."""
    import cupy as cp
    from woof.hex.mpas_seaice import timestep_v842 as ts

    dev = sr.device_pools(case, fixture)
    config = case["config"]
    clock = ts.initialize(pools=dev, configs=config, remap_geometry="input", basis="compute",
                          metric_libm=metric_libm)
    ts.advance(pools=dev, configs=config, clock=clock,
               n_steps=int(config["oracle_n_steps"]) if n_steps is None else n_steps)
    return {f"{pool}/{name}": cp.asnumpy(dev[pool][name]) for pool, name, kind, _ in case["fields"]
            if kind in (sr.REAL, sr.INT) and dev[pool][name] is not None}


def measure(precision: str, reference: dict | None = None, *, rows=CASES,
            metric_libm: str | None = "host") -> dict[str, dict[str, int]]:
    reference = load_fixture(precision) if reference is None else reference
    report: dict[str, dict[str, int]] = {}
    for case in cases(precision, rows):
        fixture = reference[case["name"]]
        got = run_gpu(case, fixture, metric_libm=metric_libm)
        want = ro.expected_outputs(case, fixture)
        report[case["name"]] = {key: sr.word_mismatches(got[key], want[key]) for key in want
                                if ro.graded(case, key)}
    return report
