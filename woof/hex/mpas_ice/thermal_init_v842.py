"""MALI v8.4.2: thermal initialisation and the enthalpy/temperature conversions on the GPU.

The kernels (``thermal_init_v842.cu``) transcribe:

- ``li_thermal_init``'s per-cell work: the surface air temperature
  ('constant', 'lapse'), the basal heat flux ('constant'), and the
  temperature initialisation ('sfc_air_temperature', 'linear' through
  ``li_init_linear_temperature_in_column``, 'file');
- ``li_compute_pressure_melting_point_fields``;
- ``li_temperature_to_enthalpy`` and ``li_enthalpy_to_temperature``,
  including the surface and basal enthalpies.

They are held word for word to those routines, compiled verbatim
(``tools/mali_oracle/run_thermal.F90``, ``tests/test_mali_thermal_init_parity.py``).
The vertical-grid coefficients ``li_thermal_init`` keeps for the solver
(``dsigmaTerm``) are module-private and are ported with ``li_thermal_solver``.
Every value 'file' names is left as read, as the Fortran leaves it.
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import numpy as np

from woof.hex import numerics as _hex_numerics
from woof.hex.errors import ConfigurationRefusal

SOURCE_FILES = (
    "src/core_landice/mode_forward/mpas_li_thermal.F:92-521,537-624,1980-2170,3128-3249",
    "src/core_landice/shared/mpas_li_constants.F:41-63",
)
COMPILE_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false",
                   "--prec-div=true", "--prec-sqrt=true")
MODULE_OPTIONS = COMPILE_OPTIONS
TEMPERATURE_INIT = {"sfc_air_temperature": 1, "linear": 2, "file": 3}
AIR_TEMPERATURE_SOURCE = {"constant": 1, "lapse": 2, "file": 0}
HEAT_FLUX_SOURCE = {"constant": 1, "file": 0}
_BLOCK = 128


def module_source() -> str:
    return files("woof.hex.mpas_ice").joinpath("thermal_init_v842.cu").read_text(encoding="utf-8")


def _option(table: dict, knob: str, value: str) -> int:
    try:
        return table[value]
    except KeyError:
        raise ConfigurationRefusal(knob=knob, value=value, reason="not an option li_thermal_init carries",
                                   declaration=f"{knob} = " + " | ".join(repr(k) for k in table)) from None


@lru_cache(maxsize=None)
def _module_on(device: int, options: tuple[str, ...]):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(), options, None, "li_thermal_init_v842.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _function(name: str):
    import cupy as cp
    return _module_on(cp.cuda.Device().id, _hex_numerics.options(COMPILE_OPTIONS, keep_ftz=True)).get_function(name)


def _check(name: str, array, shape) -> None:
    import cupy as cp
    if (not isinstance(array, cp.ndarray) or array.dtype != cp.float32 or tuple(array.shape) != tuple(shape)
            or not array.flags.c_contiguous):
        raise ValueError(f"{name} must be a contiguous device float32 array of shape {tuple(shape)}")


def _grid(n: int) -> tuple[int]:
    return (max(1, (n + _BLOCK - 1) // _BLOCK),)


def thermal_init(*, thermal: dict, thickness, upper_surface, layer_center_sigma, n_cells_solve: int,
                 config_ice_density: float, config_temperature_init: str,
                 config_surface_air_temperature_source: str, config_surface_air_temperature_value: float,
                 config_surface_air_temperature_lapse_rate: float, config_basal_heat_flux_source: str,
                 config_basal_heat_flux_value: float, config_do_restart: bool = False) -> None:
    """``li_thermal_init``'s per-cell work for one block, in place on
    ``thermal``: ``temperature`` and ``waterfrac`` (nCells, nVertLevels),
    ``surfaceAirTemperature``, ``surfaceTemperature``, ``basalTemperature``
    and ``basalHeatFlux`` (nCells)."""
    init = 0 if config_do_restart else _option(TEMPERATURE_INIT, "config_temperature_init", config_temperature_init)
    air = _option(AIR_TEMPERATURE_SOURCE, "config_surface_air_temperature_source",
                  config_surface_air_temperature_source)
    flux = _option(HEAT_FLUX_SOURCE, "config_basal_heat_flux_source", config_basal_heat_flux_source)
    n_cells, n_levels = thermal["temperature"].shape
    if not 0 <= int(n_cells_solve) <= n_cells:
        raise ValueError("n_cells_solve must lie within nCells")
    for name in ("temperature", "waterfrac"):
        _check(name, thermal[name], (n_cells, n_levels))
    for name in ("surfaceAirTemperature", "surfaceTemperature", "basalTemperature", "basalHeatFlux"):
        _check(name, thermal[name], (n_cells,))
    _check("thickness", thickness, (n_cells,))
    _check("upperSurface", upper_surface, (n_cells,))
    _check("layerCenterSigma", layer_center_sigma, (n_levels,))
    t = thermal
    _function("li_thermal_init_v842")(_grid(n_cells), (_BLOCK,), (
        t["temperature"], t["waterfrac"], t["surfaceAirTemperature"], t["surfaceTemperature"],
        t["basalTemperature"], t["basalHeatFlux"], thickness, upper_surface, layer_center_sigma,
        np.float32(config_ice_density), np.int32(air), np.float32(config_surface_air_temperature_value),
        np.float32(config_surface_air_temperature_lapse_rate), np.int32(flux),
        np.float32(config_basal_heat_flux_value), np.int32(init), np.int32(n_cells), np.int32(n_cells_solve),
        np.int32(n_levels)))


def pressure_melting_point_fields(*, thickness, layer_center_sigma, pmp_temperature, basal_pmp_temperature,
                                  rhoi: float) -> None:
    """``li_compute_pressure_melting_point_fields`` (K), in place."""
    n_cells, n_levels = pmp_temperature.shape
    _check("thickness", thickness, (n_cells,))
    _check("layerCenterSigma", layer_center_sigma, (n_levels,))
    _check("pmpTemperature", pmp_temperature, (n_cells, n_levels))
    _check("basalPmpTemperature", basal_pmp_temperature, (n_cells,))
    _function("li_pressure_melting_point_fields_v842")(_grid(n_cells), (_BLOCK,), (
        pmp_temperature, basal_pmp_temperature, thickness, layer_center_sigma, np.float32(rhoi),
        np.int32(n_cells), np.int32(n_levels)))


def temperature_to_enthalpy(*, temperature, waterfrac, enthalpy, thickness, layer_center_sigma,
                            rhoi: float) -> None:
    """``li_temperature_to_enthalpy`` for every cell's column, in place on ``enthalpy``."""
    n_cells, n_levels = enthalpy.shape
    for name, a in (("temperature", temperature), ("waterfrac", waterfrac), ("enthalpy", enthalpy)):
        _check(name, a, (n_cells, n_levels))
    _check("thickness", thickness, (n_cells,))
    _check("layerCenterSigma", layer_center_sigma, (n_levels,))
    _function("li_temperature_to_enthalpy_v842")(_grid(n_cells), (_BLOCK,), (
        enthalpy, temperature, waterfrac, thickness, layer_center_sigma, np.float32(rhoi), np.int32(n_cells),
        np.int32(n_levels)))


def enthalpy_to_temperature(*, enthalpy, temperature, waterfrac, surface_enthalpy, surface_temperature,
                            basal_enthalpy, basal_temperature, thickness, layer_center_sigma, rhoi: float) -> None:
    """``li_enthalpy_to_temperature`` with the surface and basal enthalpies for
    every cell, in place (temperate surface and basal enthalpies are reset, as
    the Fortran resets them)."""
    n_cells, n_levels = enthalpy.shape
    for name, a in (("enthalpy", enthalpy), ("temperature", temperature), ("waterfrac", waterfrac)):
        _check(name, a, (n_cells, n_levels))
    for name, a in (("surfaceEnthalpy", surface_enthalpy), ("surfaceTemperature", surface_temperature),
                    ("basalEnthalpy", basal_enthalpy), ("basalTemperature", basal_temperature),
                    ("thickness", thickness)):
        _check(name, a, (n_cells,))
    _check("layerCenterSigma", layer_center_sigma, (n_levels,))
    _function("li_enthalpy_to_temperature_v842")(_grid(n_cells), (_BLOCK,), (
        enthalpy, temperature, waterfrac, surface_enthalpy, surface_temperature, basal_enthalpy,
        basal_temperature, thickness, layer_center_sigma, np.float32(rhoi), np.int32(n_cells),
        np.int32(n_levels)))
