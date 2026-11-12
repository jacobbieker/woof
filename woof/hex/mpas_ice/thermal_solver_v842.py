"""MALI v8.4.2: the vertical thermal solver and the basal melt on the GPU.

:func:`basal_melt_floating_ice` is ``li_basal_melt_floating_ice`` and
:func:`thermal_solver` is ``li_thermal_solver`` (``mpas_li_thermal.F``), which
``li_time_integration_fe`` calls in that order. The kernels
(``thermal_solver_v842.cu``) transcribe:

- ``heat_dissipation_sia`` (with ``config_velocity_solver`` 'sia') and
  ``basal_friction``;
- the per-column solve: the surface and floating basal boundary
  temperatures, ``li_temperature_to_enthalpy``,
  ``temperature_matrix_elements`` or ``enthalpy_matrix_elements`` with
  ``li_thermal_init``'s ``dsigmaTerm``, ``tridiag_solver``, the conductive
  fluxes and ``li_enthalpy_to_temperature``;
- ``basal_melt_grounded_ice`` and the transfer of floating cells' melt;
- ``basal_melt_floating_ice`` for 'none', 'file', 'constant', 'mismip' and
  'seroussi'.

The masks are :func:`woof.hex.mpas_ice.geometry_v842.calculate_mask`. Every
word is held to those routines compiled verbatim with gfortran ``-O0
-ffp-contract=off`` at RKIND 4 (MALI's default) and RKIND 8
(``tools/mali_oracle/run_thermal_solver.F90``,
``tests/test_mali_thermal_solver_parity.py``). At RKIND 8, 'mismip''s
``tanh`` is CORE-MATH's correctly rounded one (``cr_tanh_flt64.cuh``), which
is glibc 2.44's; 'seroussi' is refused by name: glibc's binary64 ``sin`` is
not correctly rounded and is LGPL (``R8_REFUSED_FLOAT_MELT``).

The model's scratch fields (the thermal cell mask, the edge dissipation) are
private here. Arrays are device float32 (RKIND 4) or float64 (RKIND 8) with
int32 masks and 1-based
connectivity, in the Fortran layout: ``a(nVertLevels, nCells)`` is C-order
``(nCells, nVertLevels)``. ``config_ice_density`` and
``config_ocean_density`` are also li_thermal's ``rhoi`` and ``rhoo``
(``li_thermal_init`` copies them).
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import numpy as np

from woof.hex import numerics as _hex_numerics
from woof.hex.errors import ConfigurationRefusal

from . import geometry_v842, rkind

SOURCE_FILES = (
    "src/core_landice/mode_forward/mpas_li_thermal.F:276-296,642-2170,2190-3190,3265-3307",
    "src/core_landice/shared/mpas_li_constants.F:41-69",
)
COMPILE_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false",
                   "--prec-div=true", "--prec-sqrt=true")
MODULE_OPTIONS = COMPILE_OPTIONS
#: The shared libm transcriptions the r4 unit prepends (the r8 unit's are rkind.r8_prelude's)
MATH_HEADERS = ("glibc_flt32.cuh", "glibc_trig_flt32.cuh", "sfire_libm.cuh")
#: Floating melt options whose libm call has no binary64 transcription matched to the oracle host
R8_REFUSED_FLOAT_MELT = {
    "seroussi": "glibc's binary64 sin is not correctly rounded and is LGPL, so its words cannot be reproduced "
                "under the default licence",
}
THERMAL_SOLVERS = {"none": 0, "temperature": 1, "enthalpy": 2}
BASAL_MASS_BAL_FLOAT = {"none": 0, "file": 1, "constant": 2, "mismip": 3, "seroussi": 4}
AIR_TEMPERATURE_SOURCES = ("constant", "lapse", "file")
MAX_LEVELS = 128
_BLOCK = 128


def module_source(precision: str = "r4") -> str:
    rkind.require(precision)
    body = files("woof.hex.mpas_ice").joinpath("thermal_solver_v842.cu").read_text(encoding="utf-8")
    if precision == "r8":
        return rkind.r8_prelude() + body
    import woof.core.kernels as kernels
    root = Path(kernels.__file__).parent
    return "\n".join((root / name).read_text(encoding="utf-8") for name in MATH_HEADERS) + "\n" + body


@lru_cache(maxsize=None)
def _module_on(device: int, precision: str, options: tuple[str, ...]):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(precision), options, None, f"li_thermal_solver_v842_{precision}.cu")
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


def _mask_configs(config_ice_density, config_ocean_density, config_sea_level, config_dynamic_thickness,
                  config_velocity_solver) -> dict:
    return {"config_ice_density": config_ice_density, "config_ocean_density": config_ocean_density,
            "config_sea_level": config_sea_level, "config_dynamic_thickness": config_dynamic_thickness,
            "config_velocity_solver": config_velocity_solver}


def basal_melt_floating_ice(*, mesh: dict, geometry: dict, config_basal_mass_bal_float: str,
                            config_thermal_thickness: float, config_ice_density: float,
                            config_ocean_density: float, config_sea_level: float, config_dynamic_thickness: float,
                            config_bmlt_float_flux: float = 0.0, config_bmlt_float_xlimit: float = 0.0,
                            config_basal_mass_bal_seroussi_amplitude: float = 0.0,
                            config_basal_mass_bal_seroussi_period: float = 1.0,
                            config_basal_mass_bal_seroussi_phase: float = 0.0, days_since_start: float = 0.0,
                            config_velocity_solver: str = "sia") -> None:
    """``li_basal_melt_floating_ice``, in place on ``geometry``:
    ``floatingBasalMassBal`` and, except with 'file', the masks.

    ``mesh``: the mask connectivity (:func:`geometry_v842.calculate_mask`),
    ``xCell`` and ``nCellsSolve``. ``geometry``: ``thickness``,
    ``bedTopography``, ``lowerSurface``, ``cellMask``, ``vertexMask``,
    ``edgeMask``, ``floatingBasalMassBal``.
    """
    try:
        option = BASAL_MASS_BAL_FLOAT[config_basal_mass_bal_float]
    except KeyError:
        raise ConfigurationRefusal(
            knob="config_basal_mass_bal_float", value=config_basal_mass_bal_float,
            reason="not an option basal_melt_floating_ice carries (the model logs it as an error)",
            declaration="config_basal_mass_bal_float = " + " | ".join(map(repr, BASAL_MASS_BAL_FLOAT))) from None
    if option == 1:
        return   # 'file': already set
    if option == 4 and not config_basal_mass_bal_seroussi_period > 0.0:
        raise ValueError("config_basal_mass_bal_seroussi_period has to be a positive real value")
    g = geometry
    n_cells = g["thickness"].shape[0]
    precision = rkind.precision_of(g["thickness"])
    real = rkind.PRECISIONS[precision]
    if precision == "r8" and config_basal_mass_bal_float in R8_REFUSED_FLOAT_MELT:
        raise ConfigurationRefusal(
            knob="config_basal_mass_bal_float", value=config_basal_mass_bal_float,
            reason=f"at RKIND 8, {R8_REFUSED_FLOAT_MELT[config_basal_mass_bal_float]}",
            declaration="config_basal_mass_bal_float = 'none' | 'file' | 'constant' | 'mismip' at RKIND 8")
    ncs = int(mesh["nCellsSolve"])
    if not 0 <= ncs <= n_cells:
        raise ValueError("nCellsSolve must lie within nCells")
    f32 = real
    for name in ("thickness", "bedTopography", "lowerSurface", "floatingBasalMassBal"):
        _check(name, g[name], f32, (n_cells,))
    _check("xCell", mesh["xCell"], f32, (n_cells,))
    geometry_v842.calculate_mask(mesh=mesh, geometry=g, **_mask_configs(
        config_ice_density, config_ocean_density, config_sea_level, config_dynamic_thickness,
        config_velocity_solver))
    r = real
    _function(precision, "li_basal_melt_floating_v842")(_grid(n_cells), (_BLOCK,), (
        g["floatingBasalMassBal"], g["thickness"], g["cellMask"], mesh["xCell"], g["lowerSurface"],
        g["bedTopography"], np.int32(option), r(config_thermal_thickness), r(config_sea_level),
        r(config_bmlt_float_flux), r(config_bmlt_float_xlimit), r(config_basal_mass_bal_seroussi_amplitude),
        r(config_basal_mass_bal_seroussi_period), r(config_basal_mass_bal_seroussi_phase), r(days_since_start),
        r(config_ice_density), np.int32(n_cells), np.int32(ncs)))


def thermal_solver(*, mesh: dict, geometry: dict, velocity: dict, thermal: dict, deltat: float,
                   config_thermal_solver: str, config_thermal_thickness: float, config_ice_density: float,
                   config_ocean_density: float, config_sea_level: float, config_dynamic_thickness: float,
                   config_velocity_solver: str = "sia", config_flowLawExponent: float = 3.0,
                   config_surface_air_temperature_source: str = "file",
                   config_surface_air_temperature_value: float = 273.15,
                   config_surface_air_temperature_lapse_rate: float = 0.0,
                   config_thermal_calculate_bmb: bool = True) -> None:
    """``li_thermal_solver``, in place.

    ``mesh``: the mask connectivity, ``edgesOnCell``, ``layerCenterSigma``,
    ``layerInterfaceSigma``, ``dcEdge``, ``dvEdge``, ``areaCell`` and
    ``nCellsSolve``. ``geometry``: ``thickness``, ``bedTopography``,
    ``upperSurface``, ``basalWaterThickness``, ``slopeEdge``, the three
    masks, ``groundedBasalMassBal`` and ``floatingBasalMassBal``.
    ``velocity``: ``flowParamA``, ``basalSpeed``, ``betaSolve``. ``thermal``:
    ``temperature``, ``waterfrac``, ``enthalpy``, ``heatDissipation``
    ``(nCells, nVertLevels)``; ``surfaceTemperature``, ``basalTemperature``,
    ``surfaceAirTemperature``, ``basalHeatFlux``, ``basalFrictionFlux``,
    ``surfaceConductiveFlux``, ``basalConductiveFlux``.

    Raises where the model aborts: a non-positive ``deltat``, or a column
    temperature outside [-100, 100] deg C after the solve.
    """
    import cupy as cp
    if config_thermal_solver not in THERMAL_SOLVERS:
        raise ConfigurationRefusal(
            knob="config_thermal_solver", value=config_thermal_solver,
            reason="not an option li_thermal_solver carries",
            declaration="config_thermal_solver = " + " | ".join(map(repr, THERMAL_SOLVERS)))
    if config_surface_air_temperature_source not in AIR_TEMPERATURE_SOURCES:
        raise ConfigurationRefusal(
            knob="config_surface_air_temperature_source", value=config_surface_air_temperature_source,
            reason="not an option li_thermal carries",
            declaration="config_surface_air_temperature_source = " + " | ".join(map(repr, AIR_TEMPERATURE_SOURCES)))
    if config_thermal_solver == "none":
        return
    if not deltat > 0.0:
        raise ValueError(f"li_thermal_solver was called with invalid deltat = {deltat}")
    g, m, v, t = geometry, mesh, velocity, thermal
    n_cells, n_levels = t["temperature"].shape
    precision = rkind.precision_of(t["temperature"])
    real = rkind.PRECISIONS[precision]
    if not 1 <= n_levels <= MAX_LEVELS:
        raise ValueError(f"the column solve needs 1 to {MAX_LEVELS} layers, not {n_levels}")
    n_edges = g["slopeEdge"].shape[0]
    max_edges = m["edgesOnCell"].shape[1]
    ncs = int(m["nCellsSolve"])
    if not 0 <= ncs <= n_cells:
        raise ValueError("nCellsSolve must lie within nCells")
    f32, i32 = real, cp.int32
    for name in ("temperature", "waterfrac", "enthalpy", "heatDissipation"):
        _check(name, t[name], f32, (n_cells, n_levels))
    for name in ("surfaceTemperature", "basalTemperature", "surfaceAirTemperature", "basalHeatFlux",
                 "basalFrictionFlux", "surfaceConductiveFlux", "basalConductiveFlux"):
        _check(name, t[name], f32, (n_cells,))
    for name in ("thickness", "upperSurface", "basalWaterThickness", "groundedBasalMassBal",
                 "floatingBasalMassBal"):
        _check(name, g[name], f32, (n_cells,))
    _check("flowParamA", v["flowParamA"], f32, (n_cells, n_levels))
    for name in ("basalSpeed", "betaSolve"):
        _check(name, v[name], f32, (n_cells,))
    for name, shape in (("layerCenterSigma", (n_levels,)), ("layerInterfaceSigma", (n_levels + 1,)),
                        ("dcEdge", (n_edges,)), ("dvEdge", (n_edges,)), ("areaCell", (n_cells,))):
        _check(name, m[name], f32, shape)
    for name, shape in (("edgesOnCell", (n_cells, max_edges)), ("nEdgesOnCell", (n_cells,)),
                        ("cellsOnEdge", (n_edges, 2))):
        _check(name, m[name], i32, shape)
    r = real
    rhoi = r(config_ice_density)
    block = (_BLOCK,)
    cells = _grid(n_cells)
    nc, nl = np.int32(n_cells), np.int32(n_levels)
    geometry_v842.calculate_mask(mesh=m, geometry=g, **_mask_configs(
        config_ice_density, config_ocean_density, config_sea_level, config_dynamic_thickness,
        config_velocity_solver))
    if config_velocity_solver == "sia":
        edge = cp.empty((n_edges, n_levels), f32)
        _function(precision, "li_heat_dissipation_edge_v842")(_grid(n_edges), block, (
            edge, g["thickness"], v["flowParamA"], g["slopeEdge"], m["layerCenterSigma"], m["cellsOnEdge"], rhoi,
            r(config_flowLawExponent), np.int32(n_edges), nc, nl))
        _function(precision, "li_heat_dissipation_cell_v842")(cells, block, (
            t["heatDissipation"], edge, m["dcEdge"], m["dvEdge"], m["areaCell"], m["nEdgesOnCell"],
            m["edgesOnCell"], rhoi, nc, nl, np.int32(max_edges)))
    _function(precision, "li_basal_friction_v842")(cells, block, (
        t["basalFrictionFlux"], v["betaSolve"], v["basalSpeed"], nc))
    error = cp.zeros(1, i32)
    _function(precision, "li_thermal_column_v842")(cells, block, (
        t["temperature"], t["waterfrac"], t["enthalpy"], t["surfaceTemperature"], t["basalTemperature"],
        t["surfaceAirTemperature"], t["surfaceConductiveFlux"], t["basalConductiveFlux"], g["groundedBasalMassBal"],
        g["floatingBasalMassBal"], error, t["heatDissipation"], t["basalHeatFlux"], t["basalFrictionFlux"],
        g["thickness"], g["upperSurface"], g["basalWaterThickness"], g["cellMask"], m["layerCenterSigma"],
        m["layerInterfaceSigma"], r(deltat), rhoi, r(config_ocean_density), r(config_thermal_thickness),
        np.int32(config_surface_air_temperature_source == "lapse"), r(config_surface_air_temperature_value),
        r(config_surface_air_temperature_lapse_rate), np.int32(THERMAL_SOLVERS[config_thermal_solver]),
        np.int32(bool(config_thermal_calculate_bmb)), nc, np.int32(ncs), nl))
    if int(cp.asnumpy(error)[0]):
        raise RuntimeError("li_thermal: a column temperature left [-100, 100] deg C (the model aborts)")
