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

from woof.verify import mpas_ocean_diagnostics_oracle as _diag
from woof.verify import mpas_ocean_split_step_oracle as _ss
from woof.verify import mpas_ocean_stream as _stream
from woof.verify import mpas_ocean_tend_oracle as _tend
from woof.verify.mpas_ocean_hexmesh import hex_mesh
from woof.verify.mpas_ocean_tend_oracle import PRECISIONS

FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mpas_ocean_forward")
CASES_PER_PRECISION = True
REAL = _stream.REAL
FIELDS = _ss.FIELDS + (("diagnostics", "daysSinceStartOfSim", REAL, ()),)
CLOCK_CONFIG = {
    "config_start_time": "0001-01-01_00:00:00", "diagnostics:simulationStartTime": "0001-01-01_00:00:00",
    "diagnostics:xtime": "0001-01-01_00:00:00", "config_do_restart": False,
}
#: (name, seed, (nq, nr), nVertLevels, steps, config_dt, config_btr_dt, config overrides)
CASES = (
    ("split_three_steps", 20261501, (7, 6), 8, 3, "00:02:00", "0000_00:00:50", {}),
    ("split_jm_rich_bulk", 20261502, (8, 6), 10, 4, "00:03:00", "0000_00:01:00", {
        "config_eos_type": "jm", "config_pressure_gradient_type": "Jacobian_from_TS",
        "config_use_rich_visc": True, "config_use_rich_diff": True,
        "config_use_activeTracers_surface_bulk_forcing": True, "config_use_bulk_thickness_flux": True,
        "config_sw_absorption_type": "jerlov", "config_compute_active_tracer_budgets": True,
        "config_use_tracer_del2": True, "config_tracer_del2": 50.0}),
    ("unsplit_two_steps", 20261503, (6, 5), 6, 2, "00:01:00", "0000_00:00:20", {
        "config_time_integrator": "unsplit_explicit", "config_use_const_visc": True,
        "config_use_const_diff": True}),
    ("split_filter_btr_mode", 20261504, (6, 6), 7, 2, "00:02:00", "0000_00:00:40", {
        "config_filter_btr_mode": True, "config_start_time": "2000-02-28_23:58:00",
        "diagnostics:simulationStartTime": "2000-01-01_00:00:00"}),
)


def interval_seconds(text: str) -> int:
    """An MPAS interval string '[DDDD_]hh:mm:ss' in seconds."""
    days, _, clock = text.rpartition("_")
    h, m, s = (int(x) for x in clock.split(":"))
    return ((int(days or 0) * 24 + h) * 60 + m) * 60 + s


def make_case(name, seed, torus, n_levels, steps, dt, btr_dt, overrides, precision="r4") -> dict:
    """One deterministic cold start on :func:`~woof.verify.mpas_ocean_hexmesh.hex_mesh`
    (a sixth of the cells culled as land), an ocean's magnitudes, no flow
    through land edges; the geometry, weights and coefficients drawn at random."""
    rs = np.random.RandomState(seed)
    real = PRECISIONS[precision]
    mesh = hex_mesh(*torus, 1.0 / 6.0, n_levels, rs)
    nc, ne, nv, nl = mesh["nCells"], mesh["nEdges"], mesh["nVertices"], n_levels
    me, me2, vd, nt, na = mesh["maxEdges"], mesh["maxEdges2"], mesh["vertexDegree"], 2, 14
    config = {**_ss.BASE_CONFIG, **CLOCK_CONFIG, **overrides, "config_dt": dt, "config_btr_dt": btr_dt,
              "oracle_n_steps": steps}

    def uniform(lo, hi, shp):
        return lo + (hi - lo) * rs.random_sample(shp)

    mlc, mlet = mesh["maxLevelCell"], mesh["maxLevelEdgeTop"]
    # the advection stencil: an edge's cells and their neighbours
    cells_on_cell, cells_on_edge = mesh["cellsOnCell"], mesh["cellsOnEdge"]
    n_adv = np.zeros(ne + 1, np.int32)
    adv_cells = np.full((ne + 1, na), nc + 1, np.int32)
    for e in range(ne):
        stencil = []
        for c in cells_on_edge[e]:
            if c <= nc:
                stencil += [c] + [n for n in cells_on_cell[c - 1] if n <= nc]
        stencil = list(dict.fromkeys(stencil))[:na]
        n_adv[e] = len(stencil)
        adv_cells[e, :len(stencil)] = stencil
    level = np.arange(nl)[None, :]
    resting = uniform(10.0, 80.0, (nc + 1, nl))
    thickness = resting * uniform(0.97, 1.03, (nc + 1, nl))
    bottom_depth = np.array([thickness[c, :mlc[c]].sum() for c in range(nc + 1)])
    ref_bottom = np.cumsum(resting.mean(axis=0))
    velocity = uniform(-0.3, 0.3, (ne + 1, nl)) * (level < mlet[:, None])
    tracers = np.empty((nc + 1, nl, nt))
    tracers[:, :, 0] = uniform(2.0, 25.0, (nc + 1, nl))
    tracers[:, :, 1] = uniform(33.5, 36.0, (nc + 1, nl))
    fraction = uniform(0.0, 1.0, (nc + 1, nl))
    ssh = uniform(-0.5, 0.5, nc + 1)
    dims = {"nCells": nc, "nEdges": ne, "nVertices": nv, "nVertLevels": nl, "nTracers": nt, "maxEdges": me,
            "maxEdges2": me2, "vertexDegree": vd, "nAdvectionCells": na}
    case = {"name": name, "precision": precision, "dims": dims, "config": config,
            "dt": float(interval_seconds(dt))}
    values = {
        **{k: v for k, v in mesh.items()},
        "nVertLevels": nl, "nCellsSolve": nc, "nEdgesSolve": ne, "nVerticesSolve": nv,
        "nCellsArray": np.array([nc] * 4, np.int32), "nEdgesArray": np.array([ne] * 4, np.int32),
        "nVerticesArray": np.array([nv] * 4, np.int32),
        "weightsOnEdge": uniform(-0.3, 0.3, (ne + 1, me2)), "kiteAreasOnVertex": uniform(1.0e7, 1.5e8, (nv + 1, vd)),
        "dcEdge": uniform(1.0e4, 3.0e4, ne + 1), "dvEdge": uniform(5.0e3, 2.0e4, ne + 1),
        "areaCell": uniform(1.0e8, 8.0e8, nc + 1), "areaTriangle": uniform(5.0e7, 4.0e8, nv + 1),
        "bottomDepth": bottom_depth, "fVertex": uniform(-1.4e-4, 1.4e-4, nv + 1),
        "fEdge": uniform(-1.4e-4, 1.4e-4, ne + 1), "derivTwo": np.zeros((ne + 1, 2, 15)),
        "refBottomDepth": ref_bottom, "refBottomDepthTopOfCell": np.concatenate([[0.0], ref_bottom]),
        "latCell": uniform(-1.4, 1.4, nc + 1), "lonCell": uniform(0.0, 6.28, nc + 1),
        "angleEdge": uniform(-3.1, 3.1, ne + 1), "meshScaling": uniform(0.5, 1.0, ne + 1),
        "meshScalingDel2": uniform(0.3, 1.0, ne + 1), "meshScalingDel4": uniform(0.3, 1.0, ne + 1),
        "coeffs_reconstruct": uniform(-0.5, 0.5, (nc + 1, me, 3)),
        "advCoefs": uniform(-0.5, 0.5, (ne + 1, na)), "advCoefs3rd": uniform(-0.2, 0.2, (ne + 1, na)),
        "highOrderAdvectionMask": (rs.random_sample((ne + 1, nl)) < 0.8).astype(np.int32),
        "nAdvCellsForEdge": n_adv, "advCellsForEdge": adv_cells, "vertCoordMovementWeights": np.ones(nl),
        "restingThickness": resting,
        # a cold start: level 2 is initialised from level 1, the barotropic split computed
        "layerThickness": thickness, "layerThickness@2": thickness, "normalVelocity": velocity,
        "normalVelocity@2": velocity, "ssh": ssh, "ssh@2": ssh,
        "index_temperature": 1, "index_salinity": 2, "activeGRP_start": 1, "activeGRP_end": nt,
        "activeTracers": tracers, "activeTracers@2": tracers,
        "index_temperatureSurfaceFlux": 1, "index_salinitySurfaceFlux": 2,
        "fractionAbsorbed": fraction, "fractionAbsorbedRunoff": fraction[::-1].copy(),
        "surfaceFluxAttenuationCoefficient": uniform(5.0, 30.0, nc + 1),
        "surfaceFluxAttenuationCoefficientRunoff": uniform(5.0, 30.0, nc + 1),
        "seaIcePressure": uniform(0.0, 2.0e4, nc + 1), "atmosphericPressure": uniform(9.8e4, 1.03e5, nc + 1),
        "latentHeatFlux": uniform(-150.0, 0.0, nc + 1), "sensibleHeatFlux": uniform(-50.0, 50.0, nc + 1),
        "longWaveHeatFluxUp": uniform(-420.0, -300.0, nc + 1), "longWaveHeatFluxDown": uniform(250.0, 400.0, nc + 1),
        "seaIceHeatFlux": uniform(-20.0, 0.0, nc + 1), "rainFlux": uniform(0.0, 1.0e-4, nc + 1),
        "snowFlux": uniform(0.0, 2.0e-5, nc + 1), "shortWaveHeatFlux": uniform(0.0, 300.0, nc + 1),
        "evaporationFlux": uniform(-5.0e-5, 0.0, nc + 1), "seaIceFreshWaterFlux": uniform(-1.0e-5, 1.0e-5, nc + 1),
        "seaIceSalinityFlux": uniform(-1.0e-6, 1.0e-6, nc + 1), "iceRunoffFlux": uniform(0.0, 1.0e-6, nc + 1),
        "riverRunoffFlux": uniform(0.0, 5.0e-6, nc + 1), "penetrativeTemperatureFlux": uniform(0.0, 0.05, nc + 1),
        "windStressZonal": uniform(-0.2, 0.2, nc + 1), "windStressMeridional": uniform(-0.2, 0.2, nc + 1),
        "index_surfaceVelocityZonal": 1, "index_surfaceVelocityMeridional": 2, "index_SSHGradientZonal": 1,
        "index_SSHGradientMeridional": 2, "nAccumulatedCoupled": 2, "index_avgTemperatureSurfaceValue": 1,
        "index_avgSSHGradientZonal": 1, "index_avgSSHGradientMeridional": 2,
        "avgTracersSurfaceValue": uniform(0.0, 300.0, (nc + 1, nt)),
        "avgSurfaceVelocity": uniform(-0.3, 0.3, (nc + 1, 2)), "avgSSHGradient": uniform(-1.0e-5, 1.0e-5, (nc + 1, 2)),
        "vertNonLocalFlux": np.zeros((nc + 1, nl + 1, nt)), "nonLocalSurfaceTracerFlux": np.zeros((nc + 1, nt)),
        "activeTracersSurfaceFluxRunoff": uniform(-1.0e-6, 1.0e-6, (nc + 1, nt)),
        "normalGMBolusVelocity": np.zeros((ne + 1, nl)), "k33": np.zeros((nc + 1, nl + 1)),
        "relativeSlopeTopOfEdge": np.zeros((ne + 1, nl)), "relativeSlopeTapering": np.zeros((ne + 1, nl)),
        "relativeSlopeTaperingCell": np.zeros((nc + 1, nl)), "barotropicForcing": np.zeros(ne + 1),
        "daysSinceStartOfSim": 0.0,
    }
    pools: dict[str, dict] = {}
    for pool, field, kind, fshape in FIELDS:
        if kind in (_stream.DIM, _stream.DIMS):
            value = values[field]
        elif kind == _stream.SCRATCH or (pool, field) in _ss.ABSENT or not _diag.present(case, field):
            value = None
        elif field in values:
            value = values[field]
        else:
            # a field the run writes before it reads (or never reads)
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
    """Whether this checkout runs the case (the sphere at RKIND 8 needs the glibc opt-in)."""
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
