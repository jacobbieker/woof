"""MPAS-Ocean v8.4.2: the forcing terms and the full momentum and thickness tendencies.

``forcing_v842.cu`` transcribes the terms ``ocn_tend_vel`` and
``ocn_tend_thick`` (``mpas_ocn_tendency.F``) add beyond those of
``tend_v842`` and ``pgrad_hmix_v842``:

- ``ocn_surface_bulk_forcing_vel`` (the wind stress; at RKIND 8 only under
  the glibc 2.39 opt-in) and
  ``ocn_surface_bulk_forcing_thick``;
- ``ocn_vel_forcing_tend``: the surface stress (with
  ``ocn_forcing_transmission``), the explicit bottom drag and Rayleigh
  friction;
- ``ocn_thick_surface_flux_tend``;
- ``ocn_forcing_build_fraction_absorbed_array`` (:func:`fraction_absorbed`).

:func:`tend_vel` and :func:`tend_thick` compose them with the other ported
terms in the Fortran's order, and zero the arrays the Fortran zeroes. Every
word is held to the v8.4.2 routines compiled verbatim with gfortran
``-O0 -ffp-contract=off``, at RKIND 4 and 8 (``tools/mpas_ocean_oracle``,
``tests/test_mpas_ocean_tendency_parity.py``).

The libm calls take the repository's shared glibc transcriptions: ``expf``
and ``cosf``/``sinf`` at RKIND 4 (``glibc_flt32.cuh``,
``glibc_trig_flt32.cuh``) and ``exp`` at RKIND 8 (``glibc_flt64.cuh``, the
MIT Arm optimized-routines transcription).

Refused by name:

- the bulk wind stress at RKIND 8 (``config_use_bulk_wind_stress``) unless
  the opt-in is on: gfortran calls glibc's double ``cos``/``sin``, which are
  LGPL IBM code and are not correctly rounded, so no permissive transcription
  equals them. ``WOOF_GLIBC239_LIBM64=1`` (:mod:`woof.core.glibc239_gpu`, a
  source checkout) prepends the LGPL transcription of glibc 2.39's and lifts
  the refusal;
- land-ice fluxes (``config_land_ice_flux_mode`` 'standalone' or 'coupled')
  and frazil (``config_use_frazil_ice_formation``), not ported;
- the explicit bottom drag, or Rayleigh bottom friction, on a mesh with an
  edge whose ``maxLevelEdgeTop`` is 0: v8.4.2 indexes level 0 of that edge,
  outside the array (a word of the previous edge's column, or before the
  array for the first edge), so its result is not defined by the source.
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import numpy as np

from woof.hex import numerics as _hex_numerics
from woof.hex.errors import ConfigurationRefusal

from . import pgrad_hmix_v842, tend_v842
from .tend_v842 import COMPILE_OPTIONS, PRECISIONS, _check, _grid, _precision_of

SOURCE_FILES = (
    "src/core_ocean/shared/mpas_ocn_tendency.F:107-350,1034-1049",
    "src/core_ocean/shared/mpas_ocn_surface_bulk_forcing.F:139-306,320-334",
    "src/core_ocean/shared/mpas_ocn_forcing.F:98-175,191-196",
    "src/core_ocean/shared/mpas_ocn_vel_forcing.F",
    "src/core_ocean/shared/mpas_ocn_vel_forcing_surface_stress.F",
    "src/core_ocean/shared/mpas_ocn_vel_forcing_explicit_bottom_drag.F",
    "src/core_ocean/shared/mpas_ocn_vel_forcing_rayleigh.F",
    "src/core_ocean/shared/mpas_ocn_thick_surface_flux.F",
)
MODULE_OPTIONS = COMPILE_OPTIONS
#: The shared libm transcriptions each precision's unit prepends
MATH_HEADERS = {"r4": ("glibc_flt32.cuh", "glibc_trig_flt32.cuh"), "r8": ("glibc_flt64.cuh",)}
_BLOCK = 128

#: Defaults of the configs the composed tendencies read (Registry.xml, v8.4.2)
DEFAULT_CONFIGS = {
    "config_disable_vel_all_tend": False, "config_disable_thick_all_tend": False,
    "config_time_integrator": "split_explicit", "config_vert_coord_movement": "uniform_stretching",
    "config_disable_vel_coriolis": False, "config_disable_vel_vadv": False,
    "config_disable_thick_hadv": False, "config_disable_thick_vadv": False,
    "config_pressure_gradient_type": "pressure_and_zmid", "config_common_level_weight": 0.5,
    "config_density0": 1026.0, "config_disable_vel_pgrad": False,
    "config_disable_vel_hmix": False, "config_use_mom_del2": False, "config_mom_del2": 10.0,
    "config_use_Leith_del2": False, "config_Leith_parameter": 1.0, "config_Leith_dx": 15000.0,
    "config_Leith_visc2_max": 2.5e3, "config_use_mom_del4": False, "config_mom_del4": 5.0e13,
    "config_mom_del4_div_factor": 1.0, "config_use_mom_del2_tensor": False, "config_use_mom_del4_tensor": False,
    "config_use_bulk_wind_stress": False, "config_use_bulk_thickness_flux": False,
    "config_land_ice_flux_mode": "off", "config_use_frazil_ice_formation": False,
    "config_disable_vel_surface_stress": False, "config_use_explicit_bottom_drag": False,
    "config_explicit_bottom_drag_coeff": 1.0e-3, "config_disable_vel_explicit_bottom_drag": False,
    "config_Rayleigh_friction": False, "config_Rayleigh_damping_coeff": 1.0e-4,
    "config_Rayleigh_bottom_friction": False, "config_Rayleigh_bottom_damping_coeff": 1.0e-4,
    "config_disable_thick_sflux": False,
}


def module_source(precision: str = "r4") -> str:
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(PRECISIONS)}")
    import woof.core.kernels as kernels
    root = Path(kernels.__file__).parent
    math = "\n".join((root / name).read_text(encoding="utf-8") for name in MATH_HEADERS[precision])
    body = files("woof.hex.mpas_ocean").joinpath("forcing_v842.cu").read_text(encoding="utf-8")
    if precision == "r8":
        from woof.hex.mpas_ice import rkind
        return "#define MPAS_RKIND8 1\n" + math + "\n" + rkind.r8_trig_prelude() + body
    return math + "\n" + body


@lru_cache(maxsize=None)
def _module_on(device: int, precision: str, options: tuple[str, ...], trig64: bool = False):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(precision), options, None,
                                   f"ocn_forcing_v842_{precision}{'_trig64' if trig64 else ''}.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _function(precision: str, name: str):
    import cupy as cp
    options = _hex_numerics.options(COMPILE_OPTIONS, keep_ftz=True)
    from woof.hex.mpas_ice import rkind
    return _module_on(cp.cuda.Device().id, precision, options, precision == "r8" and rkind.trig64()).get_function(
        name)


def _configs(configs: dict | None) -> dict:
    merged = dict(DEFAULT_CONFIGS)
    if configs:
        unknown = set(configs) - set(DEFAULT_CONFIGS)
        if unknown:
            raise ValueError(f"configs the composed tendencies do not read: {sorted(unknown)}")
        merged.update(configs)
    return merged


def refuse_unported(configs: dict, precision: str) -> None:
    """Refuse, by name, a switch whose term is not ported."""
    if configs["config_land_ice_flux_mode"] in ("standalone", "coupled"):
        raise ConfigurationRefusal(
            knob="config_land_ice_flux_mode", value=configs["config_land_ice_flux_mode"],
            reason="the land-ice flux terms (mpas_ocn_surface_land_ice_fluxes.F) are not ported",
            declaration="config_land_ice_flux_mode = 'off' | 'pressure_only' | 'data'")
    if configs["config_use_frazil_ice_formation"]:
        raise ConfigurationRefusal(
            knob="config_use_frazil_ice_formation", value=True,
            reason="the frazil terms (mpas_ocn_frazil_forcing.F) are not ported",
            declaration="config_use_frazil_ice_formation = .false.")
    if configs["config_use_bulk_wind_stress"] and precision == "r8":
        from woof.hex.mpas_ice import rkind
        if not rkind.trig64():
            raise ConfigurationRefusal(
                knob="config_use_bulk_wind_stress", value=True,
                reason=rkind.trig_refusal("the bulk wind stress calls glibc's double cos and sin"),
                declaration="config_use_bulk_wind_stress = .false. at RKIND 8, or WOOF_GLIBC239_LIBM64=1")
    pgrad_hmix_v842.refuse_hmix_tensor(configs["config_use_mom_del2_tensor"], configs["config_use_mom_del4_tensor"])


def _refuse_level_zero(knob: str, max_level_edge_top) -> None:
    import cupy as cp
    if int(cp.count_nonzero(max_level_edge_top == 0)):
        raise ConfigurationRefusal(
            knob=knob, value=True,
            reason="v8.4.2 indexes level maxLevelEdgeTop = 0 of a land edge, outside the array",
            declaration=f"{knob} only on meshes whose every edge has maxLevelEdgeTop >= 1")


def rho_sw(configs: dict, precision: str):
    """rho_sw as ocn_constants_init sets it (config_density0, in RKIND)."""
    return PRECISIONS[precision](configs["config_density0"])


def vel_forcing_tend(*, mesh: dict, normal_velocity, surface_flux_attenuation_coefficient, surface_stress,
                     kinetic_energy_cell, layer_thickness_edge, tend, configs: dict | None = None) -> None:
    """``ocn_vel_forcing_tend``: surface stress, explicit bottom drag, Rayleigh
    friction, in that order, into ``tend(nVertLevels, nEdges)``. ``mesh`` holds
    ``maxLevelEdgeTop``, ``cellsOnEdge`` and ``edgeMask``."""
    import cupy as cp
    c = _configs(configs)
    precision = _precision_of(tend)
    dtype = PRECISIONS[precision]
    n_edges, n_levels = tend.shape
    n_cells = kinetic_energy_cell.shape[0]
    for name, a in (("normalVelocity", normal_velocity), ("layerThicknessEdge", layer_thickness_edge)):
        _check(name, a, dtype, (n_edges, n_levels))
    _check("kineticEnergyCell", kinetic_energy_cell, dtype, (n_cells, n_levels))
    _check("surfaceFluxAttenuationCoefficient", surface_flux_attenuation_coefficient, dtype, (n_cells,))
    _check("surfaceStress", surface_stress, dtype, (n_edges,))
    _check("maxLevelEdgeTop", mesh["maxLevelEdgeTop"], cp.int32, (n_edges,))
    _check("cellsOnEdge", mesh["cellsOnEdge"], cp.int32, (n_edges, 2))
    _check("edgeMask", mesh["edgeMask"], cp.int32, (n_edges, n_levels))
    grid = (_grid(n_edges), (_BLOCK,))
    if not c["config_disable_vel_surface_stress"]:
        _function(precision, "ocn_vel_forcing_surface_stress_v842")(*grid, (
            tend, surface_flux_attenuation_coefficient, surface_stress, layer_thickness_edge,
            mesh["maxLevelEdgeTop"], mesh["cellsOnEdge"], mesh["edgeMask"], rho_sw(c, precision),
            np.int32(n_edges), np.int32(n_levels)))
    if c["config_use_explicit_bottom_drag"] and not c["config_disable_vel_explicit_bottom_drag"]:
        _refuse_level_zero("config_use_explicit_bottom_drag", mesh["maxLevelEdgeTop"])
        _function(precision, "ocn_vel_forcing_explicit_bottom_drag_v842")(*grid, (
            tend, normal_velocity, kinetic_energy_cell, layer_thickness_edge, mesh["maxLevelEdgeTop"],
            mesh["cellsOnEdge"], dtype(c["config_explicit_bottom_drag_coeff"]), np.int32(n_edges),
            np.int32(n_levels)))
    if c["config_Rayleigh_friction"]:
        bottom = bool(c["config_Rayleigh_bottom_friction"])
        if bottom:
            _refuse_level_zero("config_Rayleigh_bottom_friction", mesh["maxLevelEdgeTop"])
        _function(precision, "ocn_vel_forcing_rayleigh_v842")(*grid, (
            tend, normal_velocity, mesh["maxLevelEdgeTop"], dtype(c["config_Rayleigh_damping_coeff"]),
            np.int32(bottom), np.int32(n_edges), np.int32(n_levels)))


def fraction_absorbed(*, mesh: dict, layer_thickness, diagnostics: dict, forcing: dict) -> None:
    """``ocn_forcing_build_fraction_absorbed_array``: ``fractionAbsorbed`` and
    ``fractionAbsorbedRunoff`` from the layer thicknesses and the two
    attenuation coefficients, over ``nCellsArray(2)``; arrays with MPAS's
    missing-cell element."""
    precision = _precision_of(layer_thickness)
    n_cells, nl = int(mesh["nCellsArray"][1]), int(mesh["nVertLevels"])
    for out, att in (("fractionAbsorbed", "surfaceFluxAttenuationCoefficient"),
                     ("fractionAbsorbedRunoff", "surfaceFluxAttenuationCoefficientRunoff")):
        _function(precision, "ocn_fraction_absorbed_v842")(_grid(n_cells), (_BLOCK,), (
            forcing[out], layer_thickness, diagnostics[att], mesh["maxLevelCell"], np.int32(n_cells), np.int32(nl)))


def thick_surface_flux_tend(*, mesh: dict, transmission_coefficients, transmission_coefficients_runoff,
                            surface_thickness_flux, surface_thickness_flux_runoff, tend,
                            configs: dict | None = None) -> None:
    """``ocn_thick_surface_flux_tend`` into ``tend(nVertLevels, nCells)``.
    ``mesh`` holds ``maxLevelCell``."""
    import cupy as cp
    c = _configs(configs)
    if c["config_disable_thick_sflux"]:
        return
    precision = _precision_of(tend)
    dtype = PRECISIONS[precision]
    n_cells, n_levels = tend.shape
    _check("transmissionCoefficients", transmission_coefficients, dtype, (n_cells, n_levels))
    _check("transmissionCoefficientsRunoff", transmission_coefficients_runoff, dtype, (n_cells, n_levels))
    _check("surfaceThicknessFlux", surface_thickness_flux, dtype, (n_cells,))
    _check("surfaceThicknessFluxRunoff", surface_thickness_flux_runoff, dtype, (n_cells,))
    _check("maxLevelCell", mesh["maxLevelCell"], cp.int32, (n_cells,))
    _function(precision, "ocn_thick_surface_flux_v842")(_grid(n_cells), (_BLOCK,), (
        tend, transmission_coefficients, transmission_coefficients_runoff, surface_thickness_flux,
        surface_thickness_flux_runoff, mesh["maxLevelCell"], np.int32(n_cells), np.int32(n_levels)))


def tend_vel(*, mesh: dict, state: dict, diagnostics: dict, forcing: dict, tend, configs: dict | None = None,
             index_temperature: int = 1, index_salinity: int = 2) -> None:
    """``ocn_tend_vel``: the full momentum tendency into ``tend(nVertLevels, nEdges)``.

    ``state`` holds ``normalVelocity`` and ``activeTracers``; ``diagnostics``
    the fields ``ocn_tend_vel`` reads from the diagnostics pool (``viscosity``
    is written); ``forcing`` holds ``surfaceStress`` and
    ``surfaceStressMagnitude`` (written), and ``windStressZonal``,
    ``windStressMeridional`` for the bulk wind stress. ``mesh`` is the union
    of what the terms read (``tend_v842``, ``pgrad_hmix_v842``, this module,
    and ``angleEdge``).
    """
    c = _configs(configs)
    precision = _precision_of(tend)
    refuse_unported(c, precision)
    tend_v842.rk4_on(c["config_time_integrator"])
    pgrad_hmix_v842.pgrad_type_code(c["config_pressure_gradient_type"])
    dtype = PRECISIONS[precision]
    n_edges, n_levels = tend.shape
    surface_stress, magnitude = forcing["surfaceStress"], forcing["surfaceStressMagnitude"]
    n_cells = magnitude.shape[0]
    _check("surfaceStress", surface_stress, dtype, (n_edges,))
    _check("surfaceStressMagnitude", magnitude, dtype, (n_cells,))
    tend.fill(0)
    surface_stress.fill(0)
    magnitude.fill(0)
    if c["config_disable_vel_all_tend"]:
        return
    if c["config_use_bulk_wind_stress"]:
        import cupy as cp
        for name in ("windStressZonal", "windStressMeridional"):
            _check(name, forcing[name], dtype, (n_cells,))
        _check("angleEdge", mesh["angleEdge"], dtype, (n_edges,))
        _check("cellsOnEdge", mesh["cellsOnEdge"], cp.int32, (n_edges, 2))
        _function(precision, "ocn_bulk_wind_stress_edges_v842")(_grid(n_edges), (_BLOCK,), (
            surface_stress, forcing["windStressZonal"], forcing["windStressMeridional"], mesh["angleEdge"],
            mesh["cellsOnEdge"], np.int32(n_edges)))
        _function(precision, "ocn_bulk_wind_stress_cells_v842")(_grid(n_cells), (_BLOCK,), (
            magnitude, forcing["windStressZonal"], forcing["windStressMeridional"], np.int32(n_cells)))
    d = diagnostics
    tend_v842.vel_coriolis_tend(
        mesh=mesh, normalized_relative_vorticity_edge=d["normalizedRelativeVorticityEdge"],
        normalized_planetary_vorticity_edge=d["normalizedPlanetaryVorticityEdge"],
        layer_thickness_edge=d["layerThicknessEdge"], normal_velocity=state["normalVelocity"],
        kinetic_energy_cell=d["kineticEnergyCell"], tend=tend,
        config_time_integrator=c["config_time_integrator"],
        config_disable_vel_coriolis=c["config_disable_vel_coriolis"])
    tend_v842.vel_vadv_tend(
        mesh=mesh, normal_velocity=state["normalVelocity"], layer_thickness_edge=d["layerThicknessEdge"],
        vert_ale_transport_top=d["vertAleTransportTop"], tend=tend,
        config_vert_coord_movement=c["config_vert_coord_movement"],
        config_disable_vel_vadv=c["config_disable_vel_vadv"])
    pgrad_hmix_v842.vel_pressure_grad_tend(
        mesh=mesh, pressure=d["pressure"], montgomery_potential=d["montgomeryPotential"], z_mid=d["zMid"],
        density=d["density"], potential_density=d["potentialDensity"], tracers=state["activeTracers"],
        index_temperature=index_temperature, index_salinity=index_salinity, tend=tend,
        config_pressure_gradient_type=c["config_pressure_gradient_type"],
        config_common_level_weight=c["config_common_level_weight"], config_density0=c["config_density0"],
        config_disable_vel_pgrad=c["config_disable_vel_pgrad"],
        in_situ_thermal_expansion_coeff=d.get("inSituThermalExpansionCoeff"),
        in_situ_saline_contraction_coeff=d.get("inSituSalineContractionCoeff"))
    pgrad_hmix_v842.vel_hmix_tend(
        mesh=mesh, divergence=d["divergence"], relative_vorticity=d["relativeVorticity"],
        viscosity=d["viscosity"], tend=tend,
        **{k: c[k] for k in ("config_disable_vel_hmix", "config_use_mom_del2", "config_mom_del2",
                             "config_use_Leith_del2", "config_Leith_parameter", "config_Leith_dx",
                             "config_Leith_visc2_max", "config_use_mom_del4", "config_mom_del4",
                             "config_mom_del4_div_factor", "config_use_mom_del2_tensor",
                             "config_use_mom_del4_tensor")})
    vel_forcing_tend(mesh=mesh, normal_velocity=state["normalVelocity"],
                     surface_flux_attenuation_coefficient=d["surfaceFluxAttenuationCoefficient"],
                     surface_stress=surface_stress, kinetic_energy_cell=d["kineticEnergyCell"],
                     layer_thickness_edge=d["layerThicknessEdge"], tend=tend, configs=configs)


def tend_thick(*, mesh: dict, diagnostics: dict, forcing: dict, tend, configs: dict | None = None) -> None:
    """``ocn_tend_thick``: the full thickness tendency into ``tend(nVertLevels, nCells)``.

    ``diagnostics`` holds ``normalTransportVelocity``, ``layerThicknessEdge``
    and ``vertAleTransportTop``; ``forcing`` holds ``surfaceThicknessFlux``
    and ``surfaceThicknessFluxRunoff`` (written), ``fractionAbsorbed``,
    ``fractionAbsorbedRunoff`` and, for the bulk thickness flux, the six
    freshwater fluxes.
    """
    c = _configs(configs)
    precision = _precision_of(tend)
    refuse_unported(c, precision)
    dtype = PRECISIONS[precision]
    n_cells, n_levels = tend.shape
    flux, runoff = forcing["surfaceThicknessFlux"], forcing["surfaceThicknessFluxRunoff"]
    _check("surfaceThicknessFlux", flux, dtype, (n_cells,))
    _check("surfaceThicknessFluxRunoff", runoff, dtype, (n_cells,))
    tend.fill(0)
    flux.fill(0)
    runoff.fill(0)
    if c["config_disable_thick_all_tend"]:
        return
    if c["config_use_bulk_thickness_flux"]:
        names = ("snowFlux", "rainFlux", "evaporationFlux", "seaIceFreshWaterFlux", "iceRunoffFlux",
                 "riverRunoffFlux")
        for name in names:
            _check(name, forcing[name], dtype, (n_cells,))
        _function(precision, "ocn_bulk_thickness_flux_v842")(_grid(n_cells), (_BLOCK,), (
            flux, runoff, *(forcing[n] for n in names), rho_sw(c, precision), np.int32(n_cells)))
    d = diagnostics
    tend_v842.thick_hadv_tend(mesh=mesh, normal_velocity=d["normalTransportVelocity"],
                              layer_thickness_edge=d["layerThicknessEdge"], tend=tend,
                              config_disable_thick_hadv=c["config_disable_thick_hadv"])
    tend_v842.thick_vadv_tend(mesh=mesh, vert_ale_transport_top=d["vertAleTransportTop"], tend=tend,
                              config_disable_thick_vadv=c["config_disable_thick_vadv"])
    thick_surface_flux_tend(mesh=mesh, transmission_coefficients=forcing["fractionAbsorbed"],
                            transmission_coefficients_runoff=forcing["fractionAbsorbedRunoff"],
                            surface_thickness_flux=flux, surface_thickness_flux_runoff=runoff, tend=tend,
                            configs=configs)
