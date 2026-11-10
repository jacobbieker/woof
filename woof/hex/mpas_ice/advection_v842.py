"""MALI v8.4.2: thickness and tracer advection on the GPU.

:func:`advect_thickness_tracers` is ``li_advection_thickness_tracers``
(``mpas_li_advection.F``) for ``config_thickness_advection`` 'fo' and
'none', with or without tracer advection (``advectTracersIn``, which
``li_time_integration_fe`` sets from ``config_tracer_advection``). The kernels
(``advection_v842.cu``) transcribe the layer thicknesses, ``tracer_setup``,
the first-order upwind advection, the column thickness and dynamic
thickening, the mass-balance masks, ``apply_mass_balance``, the vertical
remap and ``tracer_finish``. The masks are
:func:`woof.hex.mpas_ice.geometry_v842.calculate_mask` and the enthalpy
conversions :mod:`woof.hex.mpas_ice.thermal_init_v842`'s.

Every word is held to those routines compiled verbatim with gfortran
``-O0 -ffp-contract=off`` at RKIND 4 (MALI's default) and RKIND 8
(``tools/mali_oracle/run_advection.F90``, ``tests/test_mali_advection_parity.py``).

The model's scratch arrays (the old layer thicknesses and tracers and the new
ice's tracers) are private here: nothing the routine returns depends on
them. Arrays are device float32 (RKIND 4) or float64 (RKIND 8) with int32
masks and 1-based connectivity, in
the Fortran layout: ``a(nVertLevels, nCells)`` is C-order ``(nCells,
nVertLevels)``.
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import numpy as np

from woof.hex import numerics as _hex_numerics

from . import geometry_v842, rkind, thermal_init_v842

SOURCE_FILES = (
    "src/core_landice/mode_forward/mpas_li_advection.F:74-517,612-1378,1522-1671",
    "src/core_landice/shared/mpas_li_setup.F:553-609",
)
COMPILE_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false",
                   "--prec-div=true", "--prec-sqrt=true")
MODULE_OPTIONS = COMPILE_OPTIONS
THICKNESS_ADVECTION = ("fo", "none")
THERMAL_SOLVERS = {"none": 0, "temperature": 1, "enthalpy": 2}
MAX_LEVELS = 128
_BLOCK = 128


def module_source(precision: str = "r4") -> str:
    rkind.require(precision)
    body = files("woof.hex.mpas_ice").joinpath("advection_v842.cu").read_text(encoding="utf-8")
    return (rkind.r8_prelude() if precision == "r8" else "") + body


@lru_cache(maxsize=None)
def _module_on(device: int, precision: str, options: tuple[str, ...]):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(precision), options, None, f"li_advection_v842_{precision}.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _function(precision: str, name: str):
    import cupy as cp
    return _module_on(cp.cuda.Device().id, precision,
                      _hex_numerics.options(COMPILE_OPTIONS, keep_ftz=True)).get_function(name)


def _check(name: str, array, dtype, shape) -> None:
    import cupy as cp
    if (not isinstance(array, cp.ndarray) or array.dtype != dtype or tuple(array.shape) != tuple(shape)
            or not array.flags.c_contiguous):
        raise ValueError(f"{name} must be a contiguous device {np.dtype(dtype).name} array of shape {tuple(shape)}")


def _grid(n: int) -> tuple[int]:
    return (max(1, (n + _BLOCK - 1) // _BLOCK),)


def advect_thickness_tracers(*, mesh: dict, geometry: dict, velocity: dict, thermal: dict, dt: float,
                             advect_tracers: bool, config_thickness_advection: str, config_thermal_solver: str,
                             config_ice_density: float, config_ocean_density: float, config_sea_level: float,
                             config_dynamic_thickness: float, config_velocity_solver: str = "sia") -> None:
    """``li_advection_thickness_tracers``, in place.

    ``mesh``: ``nEdgesOnCell``, ``cellsOnCell``, ``cellsOnVertex``,
    ``cellsOnEdge``, ``edgesOnCell``, ``edgeSignOnCell``,
    ``layerThicknessFractions``, ``layerCenterSigma``, ``layerInterfaceSigma``,
    ``dvEdge``, ``areaCell`` and ``nCellsSolve``. ``geometry``: ``thickness``,
    ``bedTopography``, ``sfcMassBal``, ``basalMassBal``,
    ``groundedBasalMassBal``, ``floatingBasalMassBal``, ``layerThickness``,
    ``layerThicknessEdge``, ``cellMask``, ``vertexMask``, ``edgeMask``,
    ``dynamicThickening``. ``velocity``: ``layerNormalVelocity``. ``thermal``:
    ``temperature``, ``waterfrac``, ``enthalpy``, ``surfaceAirTemperature``,
    ``basalTemperature``. ``config_ice_density`` is also li_thermal's
    ``rhoi`` (``li_thermal_init`` copies it).
    """
    import cupy as cp
    if config_thickness_advection not in THICKNESS_ADVECTION:
        raise ValueError(f"{config_thickness_advection} is not a valid option for thickness/tracer advection "
                         f"(one of {THICKNESS_ADVECTION})")
    if config_thermal_solver not in THERMAL_SOLVERS:
        raise ValueError(f"config_thermal_solver must be one of {sorted(THERMAL_SOLVERS)}")
    g, m, t = geometry, mesh, thermal
    n_cells, n_levels = g["layerThickness"].shape
    if n_levels > MAX_LEVELS:
        raise ValueError(f"the vertical remap holds at most {MAX_LEVELS} layers")
    n_edges = velocity["layerNormalVelocity"].shape[0]
    max_edges = m["edgesOnCell"].shape[1]
    ncs = int(m["nCellsSolve"])
    if not 0 <= ncs <= n_cells:
        raise ValueError("nCellsSolve must lie within nCells")
    precision = rkind.precision_of(g["layerThickness"])
    real = rkind.PRECISIONS[precision]
    f32, i32 = real, cp.int32
    for name in ("thickness", "bedTopography", "sfcMassBal", "basalMassBal", "groundedBasalMassBal",
                 "floatingBasalMassBal", "dynamicThickening"):
        _check(name, g[name], f32, (n_cells,))
    _check("layerThicknessEdge", g["layerThicknessEdge"], f32, (n_edges, n_levels))
    _check("layerNormalVelocity", velocity["layerNormalVelocity"], f32, (n_edges, n_levels))
    for name in ("temperature", "waterfrac", "enthalpy"):
        _check(name, t[name], f32, (n_cells, n_levels))
    for name in ("surfaceAirTemperature", "basalTemperature"):
        _check(name, t[name], f32, (n_cells,))
    for name, shape in (("layerThicknessFractions", (n_levels,)), ("layerCenterSigma", (n_levels,)),
                        ("layerInterfaceSigma", (n_levels + 1,)), ("dvEdge", (n_edges,)), ("areaCell", (n_cells,))):
        _check(name, m[name], f32, shape)
    for name, shape in (("edgesOnCell", (n_cells, max_edges)), ("edgeSignOnCell", (n_cells, max_edges)),
                        ("cellsOnEdge", (n_edges, 2)), ("nEdgesOnCell", (n_cells,))):
        _check(name, m[name], i32, shape)
    block = (_BLOCK,)
    cells = _grid(n_cells)
    nc, nl = np.int32(n_cells), np.int32(n_levels)
    _function(precision, "li_calculate_layer_thickness_v842")(cells, block, (
        g["layerThickness"], g["thickness"], m["layerThicknessFractions"], nc, nl))
    if config_thickness_advection == "none":
        return

    rhoi, dtr = real(config_ice_density), real(dt)
    mode = THERMAL_SOLVERS[config_thermal_solver] if advect_tracers else 0
    if mode == 2:
        # li_temperature_to_enthalpy over the owned cells, into the thermal pool
        thermal_init_v842.temperature_to_enthalpy(
            temperature=t["temperature"][:ncs], waterfrac=t["waterfrac"][:ncs], enthalpy=t["enthalpy"][:ncs],
            thickness=g["thickness"][:ncs], layer_center_sigma=m["layerCenterSigma"], rhoi=config_ice_density)
    tracers = cp.empty((n_cells, n_levels), f32)
    surface_tracers = cp.zeros(n_cells, f32)
    basal_tracers = cp.zeros(n_cells, f32)
    _function(precision, "li_tracer_setup_v842")(cells, block, (
        tracers, surface_tracers, basal_tracers, t["temperature"], t["enthalpy"], t["surfaceAirTemperature"],
        t["basalTemperature"], rhoi, np.int32(mode), nc, nl))
    thickness_old = cp.empty_like(g["layerThickness"])
    tracers_old = cp.empty_like(tracers)
    words = np.int32(n_cells * n_levels)
    _function(precision, "li_copy_v842")(_grid(int(words)), block, (thickness_old, g["layerThickness"], words))
    _function(precision, "li_copy_v842")(_grid(int(words)), block, (tracers_old, tracers, words))
    _function(precision, "li_advect_upwind_v842")(_grid(ncs), block, (
        g["layerThickness"], tracers, thickness_old, tracers_old, velocity["layerNormalVelocity"],
        g["layerThicknessEdge"], m["dvEdge"], m["areaCell"], m["nEdgesOnCell"], m["edgesOnCell"],
        m["edgeSignOnCell"], m["cellsOnEdge"], dtr, np.int32(ncs), nl, np.int32(max_edges)))
    _function(precision, "li_column_thickness_v842")(cells, block, (
        g["thickness"], g["dynamicThickening"], g["layerThickness"], dtr, np.int32(1), nc, nl))
    masks = {"config_ice_density": config_ice_density, "config_ocean_density": config_ocean_density,
             "config_sea_level": config_sea_level, "config_dynamic_thickness": config_dynamic_thickness,
             "config_velocity_solver": config_velocity_solver}
    geometry_v842.calculate_mask(mesh=m, geometry=g, **masks)
    _function(precision, "li_mass_balance_masks_v842")(cells, block, (
        g["sfcMassBal"], g["basalMassBal"], g["bedTopography"], g["groundedBasalMassBal"],
        g["floatingBasalMassBal"], g["cellMask"], real(config_sea_level), nc))
    _function(precision, "li_apply_mass_balance_v842")(cells, block, (
        g["layerThickness"], tracers, g["sfcMassBal"], g["basalMassBal"], surface_tracers, basal_tracers,
        dtr, rhoi, nc, nl))
    _function(precision, "li_column_thickness_v842")(cells, block, (
        g["thickness"], g["dynamicThickening"], g["layerThickness"], dtr, np.int32(0), nc, nl))
    geometry_v842.calculate_mask(mesh=m, geometry=g, **masks)
    _function(precision, "li_vertical_remap_v842")(cells, block, (
        g["layerThickness"], tracers, g["thickness"], m["layerThicknessFractions"], m["layerInterfaceSigma"],
        nc, nl))
    if not advect_tracers:
        return
    # tracer_finish
    if config_thermal_solver == "temperature":
        _function(precision, "li_copy_v842")(_grid(int(words)), block, (t["temperature"], tracers, words))
    elif config_thermal_solver == "enthalpy":
        _function(precision, "li_copy_v842")(_grid(int(words)), block, (t["enthalpy"], tracers, words))
        # li_enthalpy_to_temperature without the surface and basal arguments:
        # the column part does not read them, so they are scratch here
        scratch = [cp.zeros(ncs, f32) for _ in range(4)]
        thermal_init_v842.enthalpy_to_temperature(
            enthalpy=t["enthalpy"][:ncs], temperature=t["temperature"][:ncs], waterfrac=t["waterfrac"][:ncs],
            surface_enthalpy=scratch[0], surface_temperature=scratch[1], basal_enthalpy=scratch[2],
            basal_temperature=scratch[3], thickness=g["thickness"][:ncs], layer_center_sigma=m["layerCenterSigma"],
            rhoi=config_ice_density)
