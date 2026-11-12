"""MALI v8.4.2: one forward-Euler land-ice time step on the GPU.

:func:`forward_euler_step` is ``li_time_integrator_forwardeuler``
(``mpas_li_time_integration_fe.F``) for one block, in the model's order:

1. ``prepare_advection``: ``li_layer_normal_velocity``,
   ``li_calculate_apparent_diffusivity``, the advective and diffusive CFL
   limits, the (fixed or adaptive) time step and its CFL check;
2. ``li_basal_melt_floating_ice`` and ``li_thermal_solver``;
3. ``advection_solver``: ``calculate_layerThicknessEdge``,
   ``li_advection_thickness_tracers``, negative thickness set to zero, the
   masks and ``li_update_geometry``;
4. subglacial hydrology (``li_SGH_solve``,
   :mod:`~woof.hex.mpas_ice.subglacial_hydro_v842`, with ``config_SGH``), then
   ``li_calve_ice`` or ``li_restore_calving_front``
   (:mod:`~woof.hex.mpas_ice.calving_v842`);
5. ``li_velocity_solve`` for 'sia' and 'none': the masks and geometry, the
   normal surface slope, ``li_sia_solve``, the uphill-margin cleanup,
   ``mpas_reconstruct``, the speeds, ``calculate_strain_rates`` and
   ``remove_fast_ice``;
6. ``li_calculate_diagnostic_vars``: the pressure melting point, ``dHdt``
   and the next step's ``thicknessOld``.

The routines other MALI units port are theirs
(:mod:`~woof.hex.mpas_ice.geometry_v842`, :mod:`~woof.hex.mpas_ice.thermal_init_v842`,
:mod:`~woof.hex.mpas_ice.thermal_solver_v842`, :mod:`~woof.hex.mpas_ice.advection_v842`,
:mod:`~woof.hex.mpas_ice.sia_v842`); ``time_integration_fe_v842.cu`` carries
the step's own arithmetic. Every word is held to the step compiled verbatim
with gfortran ``-O0 -ffp-contract=off`` at RKIND 4 and RKIND 8
(``tools/mali_oracle/run_time_integration_fe.F90``,
``tests/test_mali_time_integration_fe_parity.py``).

State is four dicts named after MALI's pools (``mesh``, ``geometry``,
``thermal``, ``velocity``) of device arrays in the Fortran layout
(``a(nVertLevels, nCells)`` is C-order ``(nCells, nVertLevels)``), float32
(RKIND 4) or float64 (RKIND 8), int32 masks and 1-based connectivity; the mesh
dict also holds ``nCellsSolve`` and ``nEdgesSolve``, and the step's own
scalars (``deltat``, ``allowableDtACFL``, ``allowableDtDCFL``) as one-word
device arrays. ``config`` holds the namelist options by their MPAS names.

At RKIND 8 two things are refused by name unless the opt-in glibc 2.39
transcription is on (``WOOF_GLIBC239_LIBM64=1``, :mod:`woof.core.glibc239_gpu`):
the sphere's reconstruction (``on_a_sphere``: glibc's binary64 cos and sin)
and the strain-rate angle ``eTheta`` (glibc's binary64 atan). Neither glibc
function is correctly rounded (atan differs from the correctly rounded one on
about 3 in 10^4 arguments), and both are LGPL, so their words cannot be
reproduced under the default licence. Without the opt-in, leave ``eTheta``
out of the velocity dict at RKIND 8 to run the step without it; nothing else
in v8.4.2 reads it.
"""
from __future__ import annotations

import math
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import numpy as np

from woof.hex import numerics as _hex_numerics
from woof.hex.errors import ConfigurationRefusal

from . import (advection_v842, calving_v842, geometry_v842, rkind, sia_v842, subglacial_hydro_v842,
               thermal_init_v842, thermal_solver_v842)

SOURCE_FILES = (
    "src/core_landice/mode_forward/mpas_li_time_integration_fe.F:75-952",
    "src/core_landice/mode_forward/mpas_li_diagnostic_vars.F:78-338",
    "src/core_landice/mode_forward/mpas_li_velocity.F:224-666,879-984",
    "src/core_landice/shared/mpas_li_setup.F:626-739",
    "src/operators/mpas_vector_reconstruction.F:205-339,489-588",
)
COMPILE_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false",
                   "--prec-div=true", "--prec-sqrt=true")
MODULE_OPTIONS = COMPILE_OPTIONS
MATH_HEADERS_R4 = ("glibc_trig_flt32.cuh",)
VELOCITY_SOLVERS = ("sia", "none")
ADVECTION = ("fo", "none")
_BLOCK = 128


def module_source(precision: str = "r4") -> str:
    rkind.require(precision)
    body = files("woof.hex.mpas_ice").joinpath("time_integration_fe_v842.cu").read_text(encoding="utf-8")
    if precision == "r8":
        return "#define MPAS_RKIND8 1\n" + rkind.r8_trig_prelude() + body
    import woof.core.kernels as kernels
    root = Path(kernels.__file__).parent
    return "\n".join((root / name).read_text(encoding="utf-8") for name in MATH_HEADERS_R4) + "\n" + body


@lru_cache(maxsize=None)
def _module_on(device: int, precision: str, options: tuple[str, ...], trig64: bool = False):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(precision), options, None,
                                   f"li_time_integration_fe_v842_{precision}.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _function(precision: str, name: str):
    import cupy as cp
    return _module_on(cp.cuda.Device().id, precision, _hex_numerics.options(COMPILE_OPTIONS, keep_ftz=True),
                      precision == "r8" and rkind.trig64()).get_function(name)


def _grid(n: int) -> tuple[int]:
    return (max(1, (n + _BLOCK - 1) // _BLOCK),)


def _refuse(knob: str, value, reason: str, declaration: str):
    raise ConfigurationRefusal(knob=knob, value=value, reason=reason, declaration=declaration)


def _check_config(config: dict, precision: str, velocity: dict) -> None:
    if config["config_velocity_solver"] not in VELOCITY_SOLVERS:
        sia_v842.require_velocity_solver(config["config_velocity_solver"])
        _refuse("config_velocity_solver", config["config_velocity_solver"],
                "'simple' needs the Dirichlet velocity mask the ported masks do not carry",
                "config_velocity_solver = 'sia' | 'none'")
    for knob in ("config_thickness_advection", "config_tracer_advection"):
        if config[knob] not in ADVECTION:
            raise ValueError(f"{knob} must be one of {ADVECTION}")
    if config.get("config_SGH", False):
        subglacial_hydro_v842.check_config(config, precision)
    if not config.get("config_restore_calving_front", False):
        calving_v842.check_config(config)
    if precision == "r8" and config.get("on_a_sphere", False) and not rkind.trig64():
        _refuse("on_a_sphere", True, rkind.trig_refusal("the velocity reconstruction calls glibc's binary64 cos and sin"),
                "on_a_sphere = .false. at RKIND 8")
    if precision == "r8" and "eTheta" in velocity and not rkind.trig64():
        _refuse("eTheta", "computed", rkind.trig_refusal("the strain-rate angle calls glibc's binary64 atan"),
                "leave eTheta out of the velocity pool at RKIND 8")


def _mask_configs(config: dict) -> dict:
    return {name: config[name] for name in ("config_ice_density", "config_ocean_density", "config_sea_level",
                                            "config_dynamic_thickness", "config_velocity_solver")}


def _update_geometry(mesh: dict, geometry: dict, config: dict) -> None:
    """li_update_geometry alone: the shared kernel, with the layer thickness it
    also writes sent to scratch."""
    import cupy as cp
    scratch = dict(geometry, layerThickness=cp.empty_like(geometry["layerThickness"]))
    geometry_v842.update_geometry_and_layer_thickness(
        mesh=mesh, geometry=scratch, config_ice_density=config["config_ice_density"],
        config_ocean_density=config["config_ocean_density"], config_sea_level=config["config_sea_level"])


def _layer_thickness(precision: str, mesh: dict, geometry: dict) -> None:
    n_cells, n_levels = geometry["layerThickness"].shape
    advection_v842._function(precision, "li_calculate_layer_thickness_v842")(_grid(n_cells), (_BLOCK,), (
        geometry["layerThickness"], geometry["thickness"], mesh["layerThicknessFractions"], np.int32(n_cells),
        np.int32(n_levels)))


def _min(precision: str, words, start: float):
    import cupy as cp
    real = rkind.PRECISIONS[precision]
    out = cp.empty(1, dtype=real)
    _function(precision, "li_min_v842")((1,), (1,), (out, words, real(start), np.int32(words.size)))
    return out


def _set_timestep(precision: str, advec_dt, diff_dt, config: dict, seconds_to_next_force_time: int) -> float:
    """set_timestep, on the host at the step's precision."""
    real = rkind.PRECISIONS[precision]
    allowable = min(advec_dt, diff_dt) if config["config_adaptive_timestep_include_DCFL"] else advec_dt
    proposed = min(real(allowable) * real(config["config_adaptive_timestep_CFL_fraction"]),
                   real(config["config_max_adaptive_timestep"]))
    proposed = real(math.floor(proposed))
    dt = min(proposed, real(seconds_to_next_force_time))
    if dt < real(config["config_min_adaptive_timestep"]):
        raise ValueError("New deltat is less than config_min_adaptive_timestep.")
    return dt


def forward_euler_step(*, mesh: dict, geometry: dict, thermal: dict, velocity: dict, config: dict,
                       hydro: dict | None = None,
                       seconds_to_next_force_time: int | None = None) -> None:
    """``li_time_integrator_forwardeuler`` for one block, in place.

    With ``config_adaptive_timestep`` the step sets ``mesh["deltat"]`` from
    the CFL limits and needs ``seconds_to_next_force_time``, the whole
    seconds from the clock's time to the next ring of its
    ``adaptiveTimestepForceInterval`` alarm.

    Raises where the model reports an error: an advective CFL violation,
    or (adaptive) a step below ``config_min_adaptive_timestep``.
    """
    import cupy as cp
    m, g, t, v, c = mesh, geometry, thermal, velocity, config
    precision = rkind.precision_of(g["thickness"])
    real = rkind.PRECISIONS[precision]
    _check_config(c, precision, v)
    n_cells, n_levels = g["layerThickness"].shape
    n_edges = m["dcEdge"].shape[0]
    max_edges = m["edgesOnCell"].shape[1]
    nvi = n_levels + 1
    ncs, nes = int(m["nCellsSolve"]), int(m["nEdgesSolve"])
    if not (0 <= ncs <= n_cells and 0 <= nes <= n_edges):
        raise ValueError("nCellsSolve and nEdgesSolve must lie within nCells and nEdges")
    if ncs < n_cells:
        raise ValueError("the apparent diffusivity reads the reconstructed slope on every cell, which the model "
                         "only sets on the owned ones (the rest is uninitialised scratch): nCellsSolve must be "
                         "nCells")
    sphere = np.int32(bool(c.get("on_a_sphere", False)))
    block = (_BLOCK,)
    masks = _mask_configs(c)

    # === prepare_advection
    advec_dt = geometry_v842.layer_normal_velocity(
        normal_velocity=v["normalVelocity"], layer_normal_velocity=v["layerNormalVelocity"],
        dc_edge=m["dcEdge"], n_edges_solve=nes)
    # li_calculate_apparent_diffusivity
    _layer_thickness(precision, m, g)
    slope = [cp.empty(n_cells, real) for _ in range(5)]   # the model's scratch: only the owned cells are read
    _function(precision, "li_reconstruct_1d_v842")(_grid(ncs), block, (
        *slope, g["normalSlopeEdge"], m["coeffs_reconstruct"], m["nEdgesOnCell"], m["edgesOnCell"],
        m["latCell"], m["lonCell"], sphere, np.int32(ncs), np.int32(max_edges)))
    cell_dt = cp.empty(n_cells, real)
    _function(precision, "li_apparent_diffusivity_v842")(_grid(n_cells), block, (
        g["apparentDiffusivity"], cell_dt, slope[3], slope[4], v["uReconstructZonal"], v["uReconstructMeridional"],
        g["layerThickness"], g["cellMask"], m["nEdgesOnCell"], m["dcEdge"], np.int32(n_cells),
        np.int32(n_levels)))
    diff_dt = _min(precision, cell_dt, 1.0e16)
    advec, diff = real(advec_dt.get()[0]), real(diff_dt.get()[0])
    if c["config_adaptive_timestep"]:
        if seconds_to_next_force_time is None:
            raise ValueError("an adaptive time step needs seconds_to_next_force_time")
        m["deltat"][...] = _set_timestep(precision, advec, diff, c, seconds_to_next_force_time)
    deltat = real(m["deltat"].get()[0])
    if deltat > advec:
        raise ValueError("Advective CFL violation: the time step exceeds the maximum allowable "
                         f"{float(advec)} s")
    m["allowableDtACFL"][...] = advec
    m["allowableDtDCFL"][...] = diff

    # === basal melting for floating ice, vertical thermodynamics
    floats = {"config_bmlt_float_flux": c["config_bmlt_float_flux"],
              "config_bmlt_float_xlimit": c["config_bmlt_float_xlimit"],
              "config_basal_mass_bal_seroussi_amplitude": c["config_basal_mass_bal_seroussi_amplitude"],
              "config_basal_mass_bal_seroussi_period": c["config_basal_mass_bal_seroussi_period"],
              "config_basal_mass_bal_seroussi_phase": c["config_basal_mass_bal_seroussi_phase"]}
    thermal_solver_v842.basal_melt_floating_ice(
        mesh=m, geometry=g, config_basal_mass_bal_float=c["config_basal_mass_bal_float"],
        config_thermal_thickness=c["config_thermal_thickness"],
        days_since_start=float(m["daysSinceStart"].get()[0]), **floats, **masks)
    thermal_solver_v842.thermal_solver(
        mesh=m, geometry=g, velocity=v, thermal=t, deltat=deltat,
        config_thermal_solver=c["config_thermal_solver"], config_thermal_thickness=c["config_thermal_thickness"],
        config_flowLawExponent=c["config_flowLawExponent"],
        config_surface_air_temperature_source=c["config_surface_air_temperature_source"],
        config_surface_air_temperature_value=c["config_surface_air_temperature_value"],
        config_surface_air_temperature_lapse_rate=c["config_surface_air_temperature_lapse_rate"],
        config_thermal_calculate_bmb=c["config_thermal_calculate_bmb"], **masks)

    # === advection_solver
    thickness_fo = c["config_thickness_advection"] == "fo"
    if thickness_fo or c["config_tracer_advection"] == "fo":
        _layer_thickness(precision, m, g)
        _function(precision, "li_layer_thickness_edge_v842")(_grid(n_edges), block, (
            g["layerThicknessEdge"], g["layerThickness"], v["normalVelocity"], m["cellsOnEdge"],
            np.int32(n_edges), np.int32(n_levels)))
    if thickness_fo:
        advection_v842.advect_thickness_tracers(
            mesh=m, geometry=g, velocity=v, thermal=t, dt=deltat,
            advect_tracers=c["config_tracer_advection"] == "fo", config_thickness_advection="fo",
            config_thermal_solver=c["config_thermal_solver"], **masks)
    _function(precision, "li_clamp_thickness_v842")(_grid(n_cells), block, (g["thickness"], np.int32(n_cells)))
    geometry_v842.calculate_mask(mesh=m, geometry=g, **masks)
    _update_geometry(m, g, c)

    # === subglacial hydrology
    if c.get("config_SGH", False):
        if hydro is None:
            raise ValueError("config_SGH needs the hydro pool")
        subglacial_hydro_v842.sgh_solve(mesh=m, geometry=g, thermal=t, velocity=v, hydro=hydro, config=c)

    # === calving
    if c.get("config_restore_calving_front", False):
        calving_v842.restore_calving_front(mesh=m, geometry=g, thermal=t, config=c)
    else:
        calving_v842.calve_ice(mesh=m, geometry=g, velocity=v, config=c)

    # === li_velocity_solve
    geometry_v842.calculate_mask(mesh=m, geometry=g, **masks)
    _update_geometry(m, g, c)
    solver = c["config_velocity_solver"]
    if solver == "sia" or c["config_adaptive_timestep_include_DCFL"]:
        _function(precision, "li_normal_slope_edge_v842")(_grid(n_edges), block, (
            g["normalSlopeEdge"], g["upperSurface"], g["edgeMask"], m["cellsOnEdge"], m["dcEdge"],
            np.int32(n_edges)))
    if solver == "sia":
        sia_v842.sia_solve(
            mesh=m, geometry=g, temperature=t["temperature"], velocity=v, n_edges_solve=nes,
            config_ice_density=c["config_ice_density"], config_flowLawExponent=c["config_flowLawExponent"],
            config_enhancementFactor=c["config_enhancementFactor"],
            config_default_flowParamA=c["config_default_flowParamA"],
            config_flowParamA_calculation=c["config_flowParamA_calculation"],
            config_sia_tangent_slope_calculation=c["config_sia_tangent_slope_calculation"])
    _function(precision, "li_uphill_margin_v842")(_grid(nes), block, (
        v["normalVelocity"], g["edgeMask"], g["cellMask"], g["upperSurface"], m["cellsOnEdge"], np.int32(nes),
        np.int32(nvi)))
    if solver == "sia":
        _function(precision, "li_reconstruct_2d_v842")(_grid(ncs), block, (
            v["uReconstructX"], v["uReconstructY"], v["uReconstructZ"], v["uReconstructZonal"],
            v["uReconstructMeridional"], v["normalVelocity"], m["coeffs_reconstruct"], m["nEdgesOnCell"],
            m["edgesOnCell"], m["latCell"], m["lonCell"], sphere, np.int32(ncs), np.int32(max_edges),
            np.int32(n_levels), np.int32(nvi)))
    else:
        _function(precision, "li_velocity_from_cells_v842")(_grid(n_edges), block, (
            v["normalVelocity"], v["uReconstructX"], v["uReconstructY"], m["edgeNormalVectors"], m["cellsOnEdge"],
            np.int32(n_edges), np.int32(nvi)))
    _function(precision, "li_speeds_v842")(_grid(n_cells), block, (
        v["surfaceSpeed"], v["basalSpeed"], v["uReconstructX"], v["uReconstructY"], np.int32(n_cells),
        np.int32(nvi)))
    for field, x_grad, y_grad in (("uReconstructX", "exx", "exy"), ("uReconstructY", "eyx", "eyy")):
        _function(precision, "li_gradient_2d_v842")(_grid(n_cells), block, (
            v[x_grad], v[y_grad], v[field], m["coeffs_reconstruct"], m["nEdgesOnCell"], m["edgesOnCell"],
            m["cellsOnCell"], m["cellsOnEdge"], m["dcEdge"], np.int32(n_cells), np.int32(max_edges),
            np.int32(nvi)))
    with_theta = "eTheta" in v
    _function(precision, "li_strain_rates_v842")(_grid(n_cells), block, (
        v["eTheta"] if with_theta else v["eMax"], v["eMax"], v["eMin"], v["exx"], v["eyy"], v["exy"], v["eyx"],
        np.int32(with_theta), np.int32(n_cells)))
    _function(precision, "li_remove_fast_ice_v842")(_grid(n_cells), block, (
        g["thickness"], g["calvingThickness"], v["surfaceSpeed"], g["cellMask"], np.int32(n_cells)))

    # === li_calculate_diagnostic_vars
    thermal_init_v842.pressure_melting_point_fields(
        thickness=g["thickness"], layer_center_sigma=m["layerCenterSigma"], pmp_temperature=t["pmpTemperature"],
        basal_pmp_temperature=t["basalPmpTemperature"], rhoi=c["config_ice_density"])
    _function(precision, "li_dhdt_v842")(_grid(n_cells), block, (
        g["dHdt"], g["thicknessOld"], g["thickness"], real(deltat), np.int32(n_cells)))
