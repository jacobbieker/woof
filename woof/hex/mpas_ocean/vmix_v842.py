"""MPAS-Ocean v8.4.2: vertical mixing, the implicit step, on the GPU.

:func:`vmix_implicit` is ``ocn_vmix_implicit`` (``mpas_ocn_vmix.F``) for one
block, in the Fortran's order and halo extents:

1. ``ocn_vmix_coefs``: the viscosity and diffusivity zeroed, then the
   constant coefficients (``config_use_const_visc``/``_diff``) and the
   Richardson-number ones (``config_use_rich_visc``/``_diff``: density and
   displaced density through :mod:`~woof.hex.mpas_ocean.eos_v842`, then
   ``ocn_vmix_get_rich_numbers``);
2. ``ocn_vel_vmix_tend_implicit``: the edge thicknesses and a tridiagonal
   solve of each edge's column, with the implicit bottom drag
   (``config_use_implicit_bottom_drag``);
3. ``ocn_tracer_vmix_tend_implicit`` for the active tracers, with their
   vertical-mixing budget (``config_compute_active_tracer_budgets``).

The kernels (``vmix_v842.cu``) are held word for word to the v8.4.2
routines, compiled verbatim with gfortran ``-O0 -ffp-contract=off`` at RKIND
4 and 8 (``tools/mpas_ocean_oracle/run_vmix.F90``,
``tests/test_mpas_ocean_vmix_parity.py``).

v8.4.2's own behaviours kept as written: the bottom drag of a column reads the
kinetic energy and edge thickness at the level the Fortran's loop index ends
on (level 2 for a one-level edge); below a cell's column the tracers become
the single-precision ``-1e34`` widened, and the budget is taken over every
level.

Refused by name, as not ported: the tanh coefficients
(``config_use_tanh_visc``/``_diff``), CVMix (``config_use_cvmix``), and Redi's
vertical diffusion (``config_use_standardGM``). One-level meshes raise (the
bottom drag reads level 2).
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import numpy as np

from woof.hex import numerics as _hex_numerics
from woof.hex.errors import ConfigurationRefusal

from . import eos_v842
from .tend_v842 import COMPILE_OPTIONS, PRECISIONS, _precision_of

SOURCE_FILES = (
    "src/core_ocean/shared/mpas_ocn_vmix.F:88-631,646-813",
    "src/core_ocean/shared/mpas_ocn_vmix_coefs_const.F",
    "src/core_ocean/shared/mpas_ocn_vmix_coefs_rich.F",
)
MODULE_OPTIONS = COMPILE_OPTIONS
GRAVITY = 9.80616
_BLOCK = 128


def module_source(precision: str = "r4") -> str:
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(PRECISIONS)}")
    body = files("woof.hex.mpas_ocean").joinpath("vmix_v842.cu").read_text(encoding="utf-8")
    return ("#define MPAS_RKIND8 1\n" if precision == "r8" else "") + body


@lru_cache(maxsize=None)
def _module_on(device: int, precision: str, options: tuple[str, ...]):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(precision), options, None, f"ocn_vmix_v842_{precision}.cu")
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


def check_config(configs: dict) -> None:
    """Refuse, by name, the coefficient schemes this port does not carry."""
    for knob, reason in (("config_use_tanh_visc", "the tanh coefficients are not ported"),
                         ("config_use_tanh_diff", "the tanh coefficients are not ported"),
                         ("config_use_cvmix", "CVMix (an external library in MPAS-Ocean) is not ported"),
                         ("config_use_standardGM", "Redi's vertical diffusion (with GM) is not ported")):
        if configs.get(knob, False):
            raise ConfigurationRefusal(knob=knob, value=True, reason=reason, declaration=f"{knob} = .false.")
    eos_v842.eos_type(configs)


def vmix_implicit(*, dt: float, mesh: dict, state: dict, tracers: dict, diagnostics: dict, configs: dict) -> None:
    """``ocn_vmix_implicit`` for one block, in place on the state's velocity and
    tracers and the diagnostics' coefficients, edge thicknesses and budget."""
    import cupy as cp
    check_config(configs)
    h = state["layerThickness"]
    u = state["normalVelocity"]
    precision = _precision_of(h)
    real = PRECISIONS[precision]
    i32 = np.int32
    m, d = mesh, diagnostics
    nl = int(m["nVertLevels"])
    if nl < 2:
        raise ValueError("v8.4.2's implicit bottom drag reads level 2: one-level meshes are not run")
    cells = [int(x) for x in m["nCellsArray"]]
    edges = [int(x) for x in m["nEdgesArray"]]
    n_cells_solve, n_edges_solve = int(m["nCellsSolve"]), int(m["nEdgesSolve"])
    rdt = real(dt)
    visc, diff = d["vertViscTopOfEdge"], d["vertDiffTopOfCell"]

    # ocn_vmix_coefs
    visc[: edges[1]] = 0
    diff[: cells[1]] = 0
    if configs.get("config_use_const_visc", False):
        _launch(precision, "ocn_vmix_add_const_v842", n_edges_solve * (nl + 1), (
            visc, i32(n_edges_solve), i32(nl + 1), real(configs["config_vert_visc"])))
    if configs.get("config_use_const_diff", False):
        _launch(precision, "ocn_vmix_add_const_v842", n_cells_solve * (nl + 1), (
            diff, i32(n_cells_solve), i32(nl + 1), real(configs["config_vert_diff"])))
    rich_visc = configs.get("config_use_rich_visc", False)
    rich_diff = configs.get("config_use_rich_diff", False)
    if rich_visc or rich_diff:
        eos_args = dict(mesh={"maxLevelCell": m["maxLevelCell"], "refBottomDepth": m["refBottomDepth"]},
                        active_tracers=tracers["activeTracers"], tracers_surface_value=d["tracersSurfaceValue"],
                        index_temperature=int(tracers["index_temperature"]),
                        index_salinity=int(tracers["index_salinity"]), configs=configs, n_cells=n_cells_solve)
        eos_v842.equation_of_state_density(density=d["density"], k_displaced=0, displacement_type="relative",
                                           **eos_args)
        eos_v842.equation_of_state_density(density=d["displacedDensity"], k_displaced=1,
                                           displacement_type="relative", **eos_args)
        n_rows_c, n_rows_e = h.shape[0], u.shape[0]
        dd_cell = cp.zeros((n_rows_c, nl + 1), dtype=real)
        du2_cell = cp.zeros((n_rows_c, nl + 1), dtype=real)
        dd_edge = cp.zeros((n_rows_e, nl + 1), dtype=real)
        du2_edge = cp.zeros((n_rows_e, nl + 1), dtype=real)
        d["RiTopOfCell"][: cells[-1]] = 0
        d["RiTopOfEdge"][: edges[-1]] = 0
        _launch(precision, "ocn_vmix_ddensity_cell_v842", cells[1] * nl, (
            dd_cell, d["displacedDensity"], d["density"], m["maxLevelCell"], i32(cells[1]), i32(nl)))
        _launch(precision, "ocn_vmix_rich_edge_v842", edges[1] * nl, (
            dd_edge, du2_edge, dd_cell, u, m["cellsOnEdge"], m["maxLevelEdgeTop"], i32(edges[1]), i32(nl)))
        _launch(precision, "ocn_vmix_du2_cell_v842", cells[0] * nl, (
            du2_cell, du2_edge, m["dcEdge"], m["dvEdge"], m["areaCell"], m["nEdgesOnCell"], m["edgesOnCell"],
            m["maxLevelEdgeBot"], i32(cells[0]), i32(nl), i32(m["edgesOnCell"].shape[1])))
        coef = real(-(real(GRAVITY) / real(configs["config_density0"]))) / real(2.0)
        _launch(precision, "ocn_vmix_ri_edge_v842", edges[1] * nl, (
            d["RiTopOfEdge"], dd_edge, du2_edge, d["layerThicknessEdge"], m["maxLevelEdgeTop"], i32(edges[1]),
            i32(nl), coef))
        _launch(precision, "ocn_vmix_ri_cell_v842", cells[0] * nl, (
            d["RiTopOfCell"], dd_cell, du2_cell, h, m["maxLevelCell"], i32(cells[0]), i32(nl), coef))
        if rich_visc:
            _launch(precision, "ocn_vmix_rich_visc_v842", n_edges_solve * nl, (
                visc, d["RiTopOfEdge"], m["maxLevelEdgeTop"], i32(n_edges_solve), i32(nl),
                real(configs["config_rich_mix"]), real(configs["config_bkrd_vert_visc"]),
                real(configs["config_convective_visc"])))
        if rich_diff:
            _launch(precision, "ocn_vmix_rich_diff_v842", n_cells_solve * nl, (
                diff, d["RiTopOfCell"], m["maxLevelCell"], i32(n_cells_solve), i32(nl),
                real(configs["config_rich_mix"]), real(configs["config_bkrd_vert_diff"]),
                real(configs["config_bkrd_vert_visc"]), real(configs["config_convective_diff"])))

    # the implicit velocity and tracer steps
    if not configs.get("config_disable_vel_vmix", False):
        drag = real(configs["config_implicit_bottom_drag_coeff"]) if configs.get(
            "config_use_implicit_bottom_drag", True) else real(0.0)
        work = cp.zeros((max(edges[0], 1), 6 * nl), dtype=real)
        _launch(precision, "ocn_vmix_vel_implicit_v842", edges[0], (
            u, d["layerThicknessEdge"], work, h, visc, d["kineticEnergyCell"], m["cellsOnEdge"],
            m["maxLevelEdgeTop"], i32(edges[0]), i32(nl), rdt, drag))
    group = tracers["activeTracers"]
    nt = group.shape[2]
    budgets = bool(configs.get("config_compute_active_tracer_budgets", False))
    if budgets:
        d["activeTracerVertMixTendency"][: cells[0]] = group[: cells[0]]
    if not configs.get("config_disable_tr_vmix", False):
        work = cp.zeros((max(cells[0], 1), (4 + 2 * nt) * nl), dtype=real)
        _launch(precision, "ocn_vmix_tracer_implicit_v842", cells[0], (
            group, work, h, diff, m["maxLevelCell"], i32(cells[0]), i32(nl), i32(nt), rdt))
    if budgets:
        _launch(precision, "ocn_vmix_budget_v842", cells[0] * nl * nt, (
            d["activeTracerVertMixTendency"], group, i32(cells[0] * nl * nt), rdt))
