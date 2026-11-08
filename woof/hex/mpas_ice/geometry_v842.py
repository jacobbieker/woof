"""MALI v8.4.2: masks, geometry and layer diagnostics on the GPU.

The kernels (``geometry_v842.cu``) transcribe the calls the model makes before
thickness advection:

- ``li_calculate_mask``: the cell, vertex and edge masks, for the 'sia' and
  'none' velocity solvers;
- ``li_update_geometry``: the lower and upper surfaces;
- ``li_calculate_layerThickness``;
- ``li_layer_normal_velocity``: the layer-midpoint velocity and the advective
  CFL time step.

They are held word for word to those routines, compiled verbatim
(``tools/mali_oracle``, ``tests/test_mali_geometry_parity.py``). The
higher-order solvers' mask paths (the Albany-active and Dirichlet bits) are
refused by name with the solvers themselves.
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import numpy as np

from woof.hex import numerics as _hex_numerics
from woof.hex.errors import ConfigurationRefusal

SOURCE_FILES = (
    "src/core_landice/shared/mpas_li_mask.F:30-40,239-626,700-935",
    "src/core_landice/mode_forward/mpas_li_advection.F:533-591,1397-1503",
    "src/core_landice/shared/mpas_li_setup.F:553-609",
)
COMPILE_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false",
                   "--prec-div=true", "--prec-sqrt=true")
MODULE_OPTIONS = COMPILE_OPTIONS
MASK_VELOCITY_SOLVERS = ("sia", "none")
_BLOCK = 128


def module_source() -> str:
    return files("woof.hex.mpas_ice").joinpath("geometry_v842.cu").read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def _module_on(device: int, options: tuple[str, ...]):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(), options, None, "li_geometry_v842.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _function(name: str):
    import cupy as cp
    return _module_on(cp.cuda.Device().id, _hex_numerics.options(COMPILE_OPTIONS, keep_ftz=True)).get_function(name)


def _check(name: str, array, dtype, shape) -> None:
    import cupy as cp
    if (not isinstance(array, cp.ndarray) or array.dtype != dtype or tuple(array.shape) != tuple(shape)
            or not array.flags.c_contiguous):
        raise ValueError(f"{name} must be a contiguous device {np.dtype(dtype).name} array of shape {tuple(shape)}")


def _grid(n: int) -> tuple[int]:
    return (max(1, (n + _BLOCK - 1) // _BLOCK),)


def calculate_mask(*, mesh: dict, geometry: dict, config_ice_density: float, config_ocean_density: float,
                   config_sea_level: float, config_dynamic_thickness: float,
                   config_velocity_solver: str = "sia") -> None:
    """``li_calculate_mask``: update ``geometry["cellMask"]``, ``vertexMask`` and
    ``edgeMask`` in place. ``mesh`` holds ``nEdgesOnCell``, ``cellsOnCell``,
    ``cellsOnVertex`` and ``cellsOnEdge``; ``geometry`` holds ``thickness``
    and ``bedTopography`` as well."""
    import cupy as cp
    if config_velocity_solver not in MASK_VELOCITY_SOLVERS:
        raise ConfigurationRefusal(
            knob="config_velocity_solver", value=config_velocity_solver,
            reason="the higher-order solvers' Albany mask bits need their Dirichlet masks, which are not ported",
            declaration="config_velocity_solver = 'sia' | 'none'")
    n_cells, max_edges = mesh["cellsOnCell"].shape
    n_vertices, vertex_degree = mesh["cellsOnVertex"].shape
    n_edges = mesh["cellsOnEdge"].shape[0]
    i32, f32 = cp.int32, cp.float32
    for name, dtype, shape in (("nEdgesOnCell", i32, (n_cells,)), ("cellsOnCell", i32, (n_cells, max_edges)),
                               ("cellsOnVertex", i32, (n_vertices, vertex_degree)),
                               ("cellsOnEdge", i32, (n_edges, 2))):
        _check(name, mesh[name], dtype, shape)
    for name, dtype, shape in (("thickness", f32, (n_cells,)), ("bedTopography", f32, (n_cells,)),
                               ("cellMask", i32, (n_cells,)), ("vertexMask", i32, (n_vertices,)),
                               ("edgeMask", i32, (n_edges,))):
        _check(name, geometry[name], dtype, shape)
    g = geometry
    cell = g["cellMask"]
    work = cp.empty_like(cell)
    rhoi, rhoo, sl = np.float32(config_ice_density), np.float32(config_ocean_density), np.float32(config_sea_level)
    grid, block = _grid(n_cells), (_BLOCK,)
    _function("li_mask_cell_state_v842")(grid, block, (cell, g["thickness"], g["bedTopography"], rhoi, rhoo, sl,
                                                        np.int32(n_cells)))
    _function("li_mask_margin_v842")(grid, block, (work, cell, g["thickness"], mesh["nEdgesOnCell"],
                                                    mesh["cellsOnCell"], np.float32(config_dynamic_thickness),
                                                    np.int32(n_cells), np.int32(max_edges)))
    _function("li_mask_calving_front_v842")(grid, block, (cell, work, g["thickness"], g["bedTopography"],
                                                           mesh["nEdgesOnCell"], mesh["cellsOnCell"], sl,
                                                           np.int32(n_cells), np.int32(max_edges)))
    _function("li_mask_dynamic_margin_grounding_line_v842")(grid, block, (
        work, cell, g["bedTopography"], mesh["nEdgesOnCell"], mesh["cellsOnCell"], sl, np.int32(n_cells),
        np.int32(max_edges)))
    cell[...] = work
    _function("li_mask_vertex_v842")(_grid(n_vertices), block, (g["vertexMask"], cell, mesh["cellsOnVertex"],
                                                                 np.int32(n_vertices), np.int32(vertex_degree)))
    _function("li_mask_edge_v842")(_grid(n_edges), block, (g["edgeMask"], cell, mesh["cellsOnEdge"],
                                                            np.int32(n_edges)))


def update_geometry_and_layer_thickness(*, mesh: dict, geometry: dict, config_ice_density: float,
                                        config_ocean_density: float, config_sea_level: float) -> None:
    """``li_update_geometry`` then ``li_calculate_layerThickness``, in place:
    ``geometry["lowerSurface"]``, ``upperSurface`` and ``layerThickness``
    (nCells, nVertLevels). ``mesh`` holds ``layerThicknessFractions``."""
    import cupy as cp
    n_cells, n_levels = geometry["layerThickness"].shape
    f32 = cp.float32
    _check("layerThicknessFractions", mesh["layerThicknessFractions"], f32, (n_levels,))
    for name in ("thickness", "bedTopography", "upperSurface", "lowerSurface"):
        _check(name, geometry[name], f32, (n_cells,))
    _check("cellMask", geometry["cellMask"], cp.int32, (n_cells,))
    _check("layerThickness", geometry["layerThickness"], f32, (n_cells, n_levels))
    g = geometry
    _function("li_update_geometry_layer_thickness_v842")(_grid(n_cells), (_BLOCK,), (
        g["upperSurface"], g["lowerSurface"], g["layerThickness"], g["thickness"], g["bedTopography"],
        g["cellMask"], mesh["layerThicknessFractions"], np.float32(config_ice_density),
        np.float32(config_ocean_density), np.float32(config_sea_level), np.int32(n_cells), np.int32(n_levels)))


def layer_normal_velocity(*, normal_velocity, layer_normal_velocity, dc_edge, n_edges_solve: int):
    """``li_layer_normal_velocity``: write ``layer_normal_velocity`` (nEdges,
    nVertLevels) on the solved edges. Returns ``minOfMaxAllowableDt`` as a
    device float32 array of one word."""
    import cupy as cp
    n_edges, n_levels = layer_normal_velocity.shape
    if not 0 <= int(n_edges_solve) <= n_edges:
        raise ValueError("n_edges_solve must lie within nEdges")
    _check("normalVelocity", normal_velocity, cp.float32, (n_edges, n_levels + 1))
    _check("layerNormalVelocity", layer_normal_velocity, cp.float32, (n_edges, n_levels))
    _check("dcEdge", dc_edge, cp.float32, (n_edges,))
    edge_min = cp.empty(max(1, int(n_edges_solve)), dtype=cp.float32)
    out = cp.empty(1, dtype=cp.float32)
    _function("li_layer_normal_velocity_v842")(_grid(n_edges_solve), (_BLOCK,), (
        layer_normal_velocity, edge_min, normal_velocity, dc_edge, np.int32(n_edges_solve), np.int32(n_levels)))
    _function("li_min_dt_v842")((1,), (1,), (out, edge_min, np.int32(n_edges_solve)))
    return out
