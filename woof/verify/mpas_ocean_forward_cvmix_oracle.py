"""Cases for the MPAS-Ocean forward mode with CVMix on (RKIND 8), on uniform
and variable-resolution meshes.

The driver is the forward oracle's (``tools/mpas_ocean_oracle/run_forward.F90``,
:mod:`woof.verify.mpas_ocean_forward_oracle`) linked with CVMix v0.84-beta and
``mpas_ocn_vmix_cvmix.F`` verbatim (``run_forward_cvmix_r8``; build.sh): the
cold start sets the boundary layer depth to half the top layer
(``mpas_ocn_init_routines.F``), and each step carries CVMix and KPP as
:mod:`woof.verify.mpas_ocean_split_step_cvmix_oracle` describes, KPP's
boundary layer and non-local flux handed from step to step. The field table
is the forward oracle's with
:func:`~woof.verify.mpas_ocean_split_step_cvmix_oracle.field_table`'s rows.

The meshes are :func:`~woof.verify.mpas_ocean_hexmesh.hex_mesh` tori with a
hexagon's geometry at a cell spacing that is uniform (``_resolution``
'uniform': 20 km everywhere) or varies across the mesh (``'variable'``: from
20 km at the first ocean cell to 80 km at the cells farthest from it, by
graph distance), the edge, cell, triangle and kite areas and the mesh
scaling following it.
"""
from __future__ import annotations

import hashlib
from collections import deque
from pathlib import Path

import numpy as np

from woof.verify import mpas_ocean_forward_oracle as _fw
from woof.verify import mpas_ocean_split_step_cvmix_oracle as _sscv
from woof.verify import mpas_ocean_stream as _stream
from woof.verify import mpas_ocean_tend_oracle as _tend

FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mpas_ocean_forward_cvmix")
#: RKIND 8 only: CVMix does not compile at RKIND 4
PRECISIONS = ("r8",)
CASES_PER_PRECISION = True
REAL = _stream.REAL
FIELDS = _sscv.field_table(_fw.FIELDS)
KPP, CVMIX_ON = _sscv.KPP, _sscv.CVMIX_ON
#: (name, seed, (nq, nr), nVertLevels, steps, config_dt, config_btr_dt, config overrides)
CASES = (
    ("kpp_forward_uniform", 20261901, (7, 6), 12, 3, "00:02:00", "0000_00:00:50",
     {**KPP, "_resolution": "uniform"}),
    ("kpp_forward_variable_jm_bulk", 20261902, (8, 6), 14, 3, "00:03:00", "0000_00:01:00", {
        **KPP, "config_eos_type": "jm", "config_pressure_gradient_type": "Jacobian_from_TS",
        "config_use_activeTracers_surface_bulk_forcing": True, "config_use_bulk_thickness_flux": True,
        "config_sw_absorption_type": "jerlov", "config_compute_active_tracer_budgets": True,
        "_resolution": "variable"}),
    ("kpp_forward_variable_implicit_ice", 20261903, (7, 6), 12, 3, "00:02:00", "0000_00:00:40", {
        **KPP, "config_cvmix_kpp_nonlocal_with_implicit_mix": True, "config_cvmix_kpp_matching": "MatchBoth",
        "_resolution": "variable", "_ice": True}),
    # one block of a decomposition: KPP's smoothing of the boundary layer depth over its halo
    ("kpp_forward_nested_halos_variable", 20261904, (12, 6), 12, 3, "00:02:00", "0000_00:00:30",
     {**KPP, "_owned": 3, "_resolution": "variable"}),
    ("cvmix_unsplit_uniform", 20261905, (6, 5), 10, 2, "00:01:00", "0000_00:00:20", {
        **CVMIX_ON, "config_use_cvmix_background": True, "config_use_cvmix_convection": True,
        "config_use_cvmix_shear": True, "config_cvmix_shear_mixing_scheme": "KPP",
        "config_time_integrator": "unsplit_explicit", "_resolution": "uniform"}),
    # the bulk wind stress (KPP's friction velocity; WOOF_GLIBC239_LIBM64=1 at RKIND 8, as the step's)
    ("kpp_forward_wind_uniform", 20261906, (7, 6), 12, 3, "00:02:00", "0000_00:00:50",
     {**KPP, "config_use_bulk_wind_stress": True, "_resolution": "uniform"}),
    ("kpp_forward_wind_variable_halos", 20261907, (12, 6), 12, 3, "00:02:00", "0000_00:00:30",
     {**KPP, "config_use_bulk_wind_stress": True, "config_compute_active_tracer_budgets": True, "_owned": 3,
      "_resolution": "variable"}),
)
SPACING = 2.0e4


def _cell_scale(case: dict, kind: str) -> np.ndarray:
    """Each cell's spacing over :data:`SPACING`: 1, or 1 to 4 by graph distance
    from the first ocean cell; the missing cell's is 1."""
    mesh = case["pools"]["mesh"]
    nc = case["dims"]["nCells"]
    scale = np.ones(nc + 1)
    if kind == "uniform":
        return scale
    cells_on_cell, n_on = mesh["cellsOnCell"], mesh["nEdgesOnCell"]
    distance = np.full(nc, -1)
    distance[0] = 0
    queue = deque([0])
    while queue:
        c = queue.popleft()
        for n in cells_on_cell[c, :n_on[c]]:
            if n <= nc and distance[n - 1] < 0:
                distance[n - 1] = distance[c] + 1
                queue.append(n - 1)
    distance[distance < 0] = distance.max()
    scale[:nc] = 1.0 + 3.0 * distance / max(int(distance.max()), 1)
    return scale


def set_resolution(case: dict, kind: str) -> None:
    """A hexagon's geometry at the cell spacing :func:`_cell_scale` gives: the
    edge's spacing the mean of its cells', ``dvEdge = dcEdge/sqrt(3)``, a
    hexagon's area on cells, the triangle's on vertices (its kites a third
    each), the mesh scaling the edge's spacing over the finest (del2) and its
    cube (del4)."""
    real = _tend.PRECISIONS[case["precision"]]
    mesh = case["pools"]["mesh"]
    nc, ne, nv = case["dims"]["nCells"], case["dims"]["nEdges"], case["dims"]["nVertices"]
    scale = _cell_scale(case, kind)

    def mean_over_cells(cells):
        """The mean scale of the real cells among ``cells`` (1-based), per row; 1 for none."""
        real_cell = cells <= nc
        total = np.where(real_cell, scale[np.minimum(cells, nc + 1) - 1], 0.0).sum(axis=1)
        count = real_cell.sum(axis=1)
        return np.where(count > 0, total / np.maximum(count, 1), 1.0)

    edge = mean_over_cells(mesh["cellsOnEdge"].astype(np.int64))
    vertex = mean_over_cells(mesh["cellsOnVertex"].astype(np.int64))
    root3 = np.sqrt(3.0)
    dc = SPACING * edge
    triangle = 0.25 * root3 * (SPACING * vertex) ** 2
    geometry = {
        "dcEdge": dc, "dvEdge": dc / root3, "areaCell": 0.5 * root3 * (SPACING * scale) ** 2,
        "areaTriangle": triangle, "kiteAreasOnVertex": np.repeat(triangle[:, None] / 3.0, mesh["cellsOnVertex"].shape[1], 1),
        "meshScaling": edge, "meshScalingDel2": edge, "meshScalingDel4": edge ** 3,
    }
    for name, value in geometry.items():
        assert value.shape[0] in (nc + 1, ne + 1, nv + 1), name
        mesh[name] = value.astype(real)


def make_case(name, seed, torus, n_levels, steps, dt, btr_dt, overrides, precision="r8") -> dict:
    """The forward oracle's cold start with CVMix's inputs added (from their own
    seed) and the mesh's geometry at ``_resolution``'s spacing."""
    case = _fw.make_case(name, seed, torus, n_levels, steps, dt, btr_dt, overrides, precision)
    _sscv.add_cvmix_inputs(case, np.random.RandomState(seed + 7), overrides)
    set_resolution(case, overrides["_resolution"])
    return case


def cases(precision: str = "r8") -> list[dict]:
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
    return _sscv.runs(case)


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
