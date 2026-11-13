"""MPAS-Ocean v8.4.2: tracer advection on the GPU.

:func:`tracer_advection_tend` is ``ocn_tracer_advection_tend`` for one tracer
group on one block: the monotone flux-corrected scheme
(``ocn_tracer_advection_mono_tend``), which adds the horizontal and then the
vertical advective tendency of every tracer to ``tend``, in the Fortran's
order and over each loop's own halo extent. The high-order horizontal flux
blends 2nd and 3rd/4th order (``config_horiz_tracer_adv_order``,
``config_coef_3rd_order``) through the mesh's ``advCoefs``/``advCoefs3rd``;
the vertical one is 2nd, 3rd or 4th order (``config_vert_tracer_adv_order``,
``mpas_tracer_advection_vflux3/4``). With
``config_compute_active_tracer_budgets`` the two partial tendencies go to the
budget arrays.

The kernels (``tracer_advection_v842.cu``) are held word for word to the
v8.4.2 routines, compiled verbatim with gfortran ``-O0 -ffp-contract=off`` at
RKIND 4 and 8 (``tools/mpas_ocean_oracle/run_tracer_advection.F90``,
``tests/test_mpas_ocean_tracer_advection_parity.py``).

Arrays are device float32 (RKIND 4) or float64 (RKIND 8) in the Fortran
layout with MPAS's missing-neighbour element: ``tracers(nTracers,
nVertLevels, nCells+1)`` is C-order ``(nCells+1, nVertLevels, nTracers)``,
``vertAleTransportTop(nVertLevelsP1, nCells+1)`` is ``(nCells+1,
nVertLevels+1)``. ``mesh`` holds the connectivity and the dimensions
(``nCellsArray``, ``nEdgesArray`` as host integers).

Refused by name:

- ``config_monotonic = .false.``: v8.4.2's standard scheme asks the scratch
  pool for fields its Registry does not define, so the model cannot run it;
- fewer than 3 halos (``config_num_halos``), which v8.4.2 aborts on;
- a cell of the block's inner halos with ``maxLevelCell`` below 2, or fewer
  than 3 levels: v8.4.2 reads outside the column there.
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import numpy as np

from woof.hex import numerics as _hex_numerics
from woof.hex.errors import ConfigurationRefusal

from .tend_v842 import COMPILE_OPTIONS, PRECISIONS, _precision_of

SOURCE_FILES = (
    "src/core_ocean/shared/mpas_ocn_tracer_advection.F",
    "src/core_ocean/shared/mpas_ocn_tracer_advection_mono.F",
    "src/operators/mpas_tracer_advection_helpers.F:44-73",
)
MODULE_OPTIONS = COMPILE_OPTIONS
_BLOCK = 128


def module_source(precision: str = "r4") -> str:
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(PRECISIONS)}")
    body = files("woof.hex.mpas_ocean").joinpath("tracer_advection_v842.cu").read_text(encoding="utf-8")
    return ("#define MPAS_RKIND8 1\n" if precision == "r8" else "") + body


@lru_cache(maxsize=None)
def _module_on(device: int, precision: str, options: tuple[str, ...]):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(precision), options, None,
                                   f"ocn_tracer_advection_v842_{precision}.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _launch(precision: str, name: str, n: int, args: tuple) -> None:
    import cupy as cp
    if n > 0:
        options = _hex_numerics.options(COMPILE_OPTIONS, keep_ftz=True)
        _module_on(cp.cuda.Device().id, precision, options).get_function(name)(
            ((n + _BLOCK - 1) // _BLOCK,), (_BLOCK,), args)


def orders(configs: dict) -> tuple[float, int]:
    """(coef_3rd_order, vertical order) as ocn_tracer_advection_mono_init sets them."""
    horizontal = int(configs["config_horiz_tracer_adv_order"])
    vertical = int(configs["config_vert_tracer_adv_order"])
    if horizontal == 3:
        coef = float(configs["config_coef_3rd_order"])
    elif horizontal in (2, 4):
        coef = 0.0
    else:
        raise ConfigurationRefusal(
            knob="config_horiz_tracer_adv_order", value=horizontal,
            reason="v8.4.2 leaves coef_3rd_order unset for any other order",
            declaration="config_horiz_tracer_adv_order = 2 | 3 | 4")
    return coef, vertical if vertical in (3, 4) else 2


def check_config(configs: dict) -> None:
    """Refuse, by name, what v8.4.2 cannot run or this port does not carry."""
    if not configs.get("config_monotonic", True):
        raise ConfigurationRefusal(
            knob="config_monotonic", value=False,
            reason="v8.4.2's standard tracer advection asks the scratch pool for fields (highOrderHorizFlux, "
                   "tracerValue, highOrderVertFlux) its Registry does not define, so the model cannot run it",
            declaration="config_monotonic = .true.")
    if int(configs.get("config_num_halos", 3)) < 3:
        raise ConfigurationRefusal(
            knob="config_num_halos", value=configs["config_num_halos"],
            reason="monotonic advection cannot be used with less than 3 halos (v8.4.2 aborts)",
            declaration="config_num_halos >= 3")
    orders(configs)


def _check_columns(max_level_cell, n_cells: int, nl: int) -> None:
    import cupy as cp
    if nl < 3:
        raise ConfigurationRefusal(
            knob="nVertLevels", value=nl,
            reason="v8.4.2's monotone advection reads a third level of every column",
            declaration="nVertLevels >= 3")
    if n_cells and int(cp.min(max_level_cell[:n_cells])) < 2:
        raise ConfigurationRefusal(
            knob="maxLevelCell", value=int(cp.min(max_level_cell[:n_cells])),
            reason="v8.4.2's monotone advection reads the level above the top of a one-level column, outside it",
            declaration="maxLevelCell >= 2 on nCellsArray(2)")


def tracer_advection_tend(*, tracers, normal_thickness_flux, w, layer_thickness, dt: float, mesh: dict, tend,
                          configs: dict, horizontal_budget=None, vertical_budget=None) -> None:
    """``ocn_tracer_advection_tend`` for one tracer group, in place on ``tend``
    (and the budget arrays when ``config_compute_active_tracer_budgets``)."""
    import cupy as cp
    check_config(configs)
    if configs.get("config_disable_tr_adv", False):
        return
    precision = _precision_of(tracers)
    real = PRECISIONS[precision]
    cells = [int(x) for x in mesh["nCellsArray"]]
    edges = [int(x) for x in mesh["nEdgesArray"]]
    n_rows, nl, nt = tracers.shape
    _check_columns(mesh["maxLevelCell"], cells[1], nl)
    coef, vertical = orders(configs)
    coef, rdt = real(coef), real(dt)
    me = mesh["edgesOnCell"].shape[1]
    na = mesh["advCoefs"].shape[1]
    budgets = bool(configs.get("config_compute_active_tracer_budgets", False))
    i32 = np.int32
    m = mesh
    shape = (n_rows, nl)
    h_prov, h_prov_inv, h_new_inv = (cp.zeros(shape, dtype=real) for _ in range(3))
    cur, work, f_in, f_out, t_min, t_max = (cp.zeros(shape, dtype=real) for _ in range(6))
    hof_edges = cp.zeros((m["dvEdge"].shape[0], nl + 1), dtype=real)
    hof_cells = cp.zeros((n_rows, nl + 1), dtype=real)
    h_budget = horizontal_budget if budgets else tend
    v_budget = vertical_budget if budgets else tend
    connectivity = (m["dvEdge"], m["areaCell"], m["nEdgesOnCell"], m["edgesOnCell"], m["edgeSignOnCell"])
    _launch(precision, "ocn_adv_mono_thickness_v842", cells[-1] * nl, (
        h_prov, h_prov_inv, h_new_inv, layer_thickness, normal_thickness_flux, w, *connectivity,
        m["maxLevelCell"], i32(cells[-1]), i32(nl), i32(me), rdt))
    for i_tracer in range(nt):
        _launch(precision, "ocn_adv_mono_start_v842", cells[-1] * nl, (
            cur, work, f_in, f_out, tracers, i32(cells[-1]), i32(nl), i32(nt), i32(i_tracer)))
        _launch(precision, "ocn_adv_mono_horizontal_bounds_v842", cells[1] * nl, (
            t_min, t_max, cur, m["nEdgesOnCell"], m["cellsOnCell"], m["maxLevelCell"], i32(cells[1]), i32(nl),
            i32(me)))
        _launch(precision, "ocn_adv_mono_horizontal_flux_v842", edges[2] * nl, (
            hof_edges, cur, normal_thickness_flux, m["advCoefs"], m["advCoefs3rd"], m["nAdvCellsForEdge"],
            m["advCellsForEdge"], m["highOrderAdvectionMask"], m["dvEdge"], m["cellsOnEdge"], m["maxLevelCell"],
            m["maxLevelEdgeTop"], i32(edges[2]), i32(nl), i32(na), coef))
        _launch(precision, "ocn_adv_mono_horizontal_limiter_v842", cells[1] * nl, (
            work, f_in, f_out, hof_edges, cur, t_min, t_max, layer_thickness, h_prov_inv, normal_thickness_flux,
            m["dvEdge"], m["areaCell"], m["nEdgesOnCell"], m["edgesOnCell"], m["edgeSignOnCell"],
            m["cellsOnEdge"], m["maxLevelEdgeTop"], m["maxLevelCell"], i32(cells[1]), i32(nl), i32(me), rdt))
        _launch(precision, "ocn_adv_mono_horizontal_limit_v842", edges[1] * nl, (
            hof_edges, f_in, f_out, m["cellsOnEdge"], m["maxLevelEdgeTop"], i32(edges[1]), i32(nl)))
        _launch(precision, "ocn_adv_mono_horizontal_update_v842", cells[0] * nl, (
            cur, work, tend, h_budget, hof_edges, layer_thickness, h_prov_inv, m["areaCell"], m["nEdgesOnCell"],
            m["edgesOnCell"], m["edgeSignOnCell"], m["maxLevelEdgeTop"], m["maxLevelCell"], i32(cells[0]), i32(nl),
            i32(me), i32(nt), i32(i_tracer), i32(budgets), rdt))
        # the vertical pass starts from zeroed fluxes and work over nCellsArray(4)
        hof_cells[: cells[-1]] = 0
        work[: cells[-1]] = 0
        _launch(precision, "ocn_adv_mono_vertical_flux_v842", cells[1], (
            hof_cells, work, t_min, t_max, f_in, f_out, cur, h_prov, w, m["maxLevelCell"], i32(cells[1]), i32(nl),
            i32(vertical), coef))
        _launch(precision, "ocn_adv_mono_vertical_limiter_v842", cells[1] * nl, (
            f_in, f_out, work, cur, t_min, t_max, h_prov, h_new_inv, m["maxLevelCell"], i32(cells[1]), i32(nl), rdt))
        _launch(precision, "ocn_adv_mono_vertical_update_v842", cells[0], (
            hof_cells, work, tend, v_budget, f_in, f_out, m["maxLevelCell"], i32(cells[0]), i32(nl), i32(nt),
            i32(i_tracer), i32(budgets)))
