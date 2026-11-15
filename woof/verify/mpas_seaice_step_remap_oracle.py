"""Cases for the MPAS-Seaice time-step oracle with incremental-remap advection on.

``tools/mpas_seaice_oracle/run_seaice_step.F90`` (see
:mod:`woof.verify.mpas_seaice_step_oracle`) with ``config_use_advection``:
its init runs ``seaice_init_advection`` (the remap's init, verbatim) and each
dynamics subcycle ``seaice_run_advection`` after the velocity solver, so the
remapped ice feeds the next subcycle's aggregates, masks and ice strength.

A case joins a step case (the velocity solver's fields, from
:func:`woof.verify.mpas_seaice_step_oracle.make_case`) and a remap case (the
mesh's geometry, the tracers and the remap's pools, from
:func:`woof.verify.mpas_seaice_remap_oracle.make_case`) on the same planar
block. The init's geometry is the GPU's input, as in the remap oracle.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from woof.verify import mpas_ocean_stream as _stream
from woof.verify import mpas_seaice_remap_oracle as ro
from woof.verify import mpas_seaice_step_oracle as so

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mpas_seaice_step_remap"
PRECISIONS = ro.PRECISIONS
REAL, INT, DIM = _stream.REAL, _stream.INT, _stream.DIM
#: what the remap's case contributes over the step's: the mesh, the tracers and the remap's pools
_REMAP_POOLS = ("incremental_remap", "tracers", "tracer_masks", "tracer_edge_fluxes", "tracer_products",
                "tracer_triangles", "tracer_barycenter", "tracer_reconstruction", "tracer_conservation",
                "tracer_monotonicity")
#: (name, seed, mesh kind, owned radius, steps, config_dt [s], packages, step overrides, remap overrides[,
#: remap field overrides: the sphere, as in mpas_seaice_remap_oracle.CASES])
CASES = (
    ("remap_step_hex", 20261801, "hex", 2, 3, 1800.0, {}, {}, {}),
    ("remap_step_quad_ponds_two_subcycles", 20261802, "quad", 2, 2, 3600.0, {
        "pkgColumnTracerLevelIceActive": True, "pkgColumnTracerPondsActive": True,
        "pkgColumnTracerIceAgeActive": True}, {
        "config_revised_evp": True, "config_dynamics_subcycle_number": 2, "config_elastic_subcycle_number": 12},
     {"config_recover_tracer_means_check": True}),
)


def make_case(name, seed, kind, owned, steps, dt, packages, step_overrides, remap_overrides, field_overrides=None,
              precision="r4") -> dict:
    field_overrides = dict(field_overrides or {})
    remap = ro.make_case(name, seed, 4, kind, owned, packages, remap_overrides, field_overrides, precision=precision)
    mesh = ro.case_mesh(kind, np.random.RandomState(seed), owned, field_overrides)
    step = so.make_case(name, seed + 7, None, None, steps, dt, step_overrides, precision=precision, mesh=mesh,
                        n_categories=remap["dims"]["nCategories"])
    case = dict(step)
    case["dims"] = {**step["dims"], **remap["dims"]}
    config = {**step["config"], **{k: v for k, v in remap["config"].items() if not k.startswith("oracle_")},
              "config_use_advection": True, "config_advection_type": "incremental_remap"}
    case["config"] = config
    case["stage"] = 4
    case["mesh_kind"] = kind
    case["packages"], case["tracer_fields"], case["absent"] = remap["packages"], remap["tracer_fields"], ()
    for key in ("sphere", "varres"):
        if key in remap:
            case[key] = remap[key]
    rows, seen = [], set()
    for row in [*so.FIELDS, *remap["fields"]]:
        if (row[0], row[1]) not in seen:
            seen.add((row[0], row[1]))
            rows.append(row)
    pools: dict[str, dict] = {}
    for pool, fname, _, _ in rows:
        if pool in _REMAP_POOLS or pool == "mesh" and fname in remap["pools"]["mesh"]:
            value = remap["pools"][pool][fname]
        elif fname in step["pools"].get(pool, {}):
            value = step["pools"][pool][fname]
        else:
            value = remap["pools"][pool][fname]
        pools.setdefault(pool, {})[fname] = value
    case["pools"] = pools
    case["fields"] = rows
    return case


def cases(precision: str = "r4", rows=CASES) -> list[dict]:
    return [make_case(*row, precision=precision) for row in rows]


def input_bytes(case: dict) -> bytes:
    return _stream.input_bytes(case["fields"], case)


def input_digest(case: dict) -> str:
    return hashlib.sha256(input_bytes(case)).hexdigest()


def read_outputs(raw: bytes, case: dict) -> dict[str, np.ndarray]:
    return _stream.read_outputs(case["fields"], raw, case)


word_mismatches = _stream.word_mismatches


def load_fixture(precision: str, fixture_dir: Path = FIXTURE_DIR) -> dict[str, dict[str, np.ndarray]]:
    data = np.load(fixture_dir / f"oracle-{precision}.npz")
    out: dict[str, dict[str, np.ndarray]] = {}
    for key in data.files:
        case, field = key.split("/", 1)
        out.setdefault(case, {})[field] = data[key]
    return out


def device_pools(case: dict, fixture: dict[str, np.ndarray]) -> dict[str, dict]:
    """The case's pools on the device with the init's geometry in place, and
    the time diagnostics' strings."""
    dev = ro.on_device(case, fixture)
    config = case["config"]
    dev["diagnostics"]["xtime"] = config["diagnostics:xtime"]
    dev["diagnostics"]["simulationStartTime"] = config["diagnostics:simulationStartTime"]
    return dev


def run_gpu(case: dict, fixture: dict[str, np.ndarray], n_steps: int | None = None) -> dict[str, np.ndarray]:
    """Run woof.hex.mpas_seaice.timestep_v842 on ``case`` as run_seaice_step.F90 drives the Fortran."""
    import cupy as cp
    from woof.hex.mpas_seaice import timestep_v842 as ts

    dev = device_pools(case, fixture)
    config = case["config"]
    clock = ts.initialize(pools=dev, configs=config, remap_geometry="input")
    ts.advance(pools=dev, configs=config, clock=clock,
               n_steps=int(config["oracle_n_steps"]) if n_steps is None else n_steps)
    return {f"{pool}/{name}": cp.asnumpy(dev[pool][name]) for pool, name, kind, _ in case["fields"]
            if kind in (REAL, INT) and dev[pool][name] is not None}


def measure(precision: str, reference: dict | None = None, *, rows=CASES,
            fixture_dir: Path = FIXTURE_DIR) -> dict[str, dict[str, int]]:
    reference = load_fixture(precision, fixture_dir) if reference is None else reference
    report: dict[str, dict[str, int]] = {}
    for case in cases(precision, rows):
        fixture = reference[case["name"]]
        got = run_gpu(case, fixture)
        want = ro.expected_outputs(case, fixture)
        report[case["name"]] = {key: word_mismatches(got[key], want[key]) for key in want
                                if ro.graded(case, key)}
    return report
