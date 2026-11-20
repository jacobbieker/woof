"""MPAS-Ocean v8.4.2: the land-ice flux modes ('standalone', 'coupled') on the GPU.

The pieces, each the v8.4.2 routine for one block, called where MPAS calls them:

- :func:`input_fields` is the flux part of ``ocn_compute_land_ice_flux_input_fields``
  (``mpas_ocn_diagnostics.F``), which ``ocn_diagnostic_solve`` calls after the
  attenuation coefficients and the land-ice mask: the top drag on edges, its
  magnitude and the friction velocity, the boundary layer's mean temperature
  and salinity (with ``config_land_ice_flux_boundaryLayerNeighborWeight``'s
  neighbours), the Jenkins transfer velocities, and the land-ice attenuation
  coefficient;
- :func:`build_arrays` is ``ocn_surface_land_ice_fluxes_build_arrays``
  (``mpas_ocn_surface_land_ice_fluxes.F``, 'standalone' only), which
  ``ocn_forward_mode_run`` calls before each step: the ISOMIP or Jenkins melt
  fluxes (``compute_melt_fluxes`` without the ice's temperature) and the
  accumulated land-ice mass and heat at time level 2;
- :func:`thick_tend`, :func:`vel_tend` and :func:`tracers_tend` are
  ``ocn_surface_land_ice_fluxes_thick``, ``_vel`` and ``_tracers`` (the
  active tracers), the terms ``ocn_tend_thick``, ``ocn_tend_vel`` and
  ``ocn_tend_tracer`` add after the bulk forcing;
- :func:`effective_density_update` is
  ``ocn_effective_density_in_land_ice_update``
  (``mpas_ocn_effective_density_in_land_ice.F``, 'coupled' only), after the
  split step's last ``ocn_diagnostic_solve``;
- :func:`coupled_means` is the land-ice part of
  ``ocn_time_average_coupled_accumulate`` ('coupled' only).

In 'coupled' mode the melt fluxes are the coupler's (the forcing pool's
inputs); only the pieces above run.

The kernels (``land_ice_v842.cu``) are held word for word to v8.4.2, the
modules compiled verbatim with gfortran ``-O0 -ffp-contract=off`` at RKIND 4
and 8 (``tools/mpas_ocean_oracle/run_land_ice.F90`` and ``run_diagnostics.F90``,
``tests/test_mpas_ocean_land_ice_parity.py``; over forward steps,
``tests/test_mpas_ocean_forward_land_ice_parity.py``).

Refused by name: ``config_land_ice_flux_formulation = 'HollandJenkins'`` (its
transfer velocities call libm's ``log`` and ``pow``, not ported) and
``config_land_ice_flux_useHollandJenkinsAdvDiff`` with the Jenkins fluxes
(``compute_HJ99_melt_fluxes``, not ported).
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import numpy as np

from woof.hex.errors import ConfigurationRefusal

from .tend_v842 import COMPILE_OPTIONS, PRECISIONS, _precision_of, launch_options

SOURCE_FILES = (
    "src/core_ocean/shared/mpas_ocn_diagnostics.F:1541-1830",
    "src/core_ocean/shared/mpas_ocn_surface_land_ice_fluxes.F",
    "src/core_ocean/shared/mpas_ocn_effective_density_in_land_ice.F",
    "src/core_ocean/shared/mpas_ocn_time_average_coupled.F:283-302",
    "src/core_ocean/shared/mpas_ocn_equation_of_state.F:ocn_freezing_temperature",
)
MODULE_OPTIONS = COMPILE_OPTIONS
MODES = ("off", "pressure_only", "standalone", "coupled")
FORMULATIONS = ("ISOMIP", "Jenkins")
_BLOCK = 128
#: ocn_constants_init's cp_sw and latent_heat_fusion_mks (not configs)
CP_SW = 3.996e3
LATENT_HEAT_FUSION_MKS = 3.337e5

#: Registry.xml defaults (v8.4.2) of the configs the land-ice routines read
DEFAULT_CONFIGS = {
    "config_land_ice_flux_mode": "off", "config_land_ice_flux_formulation": "Jenkins",
    "config_land_ice_flux_useHollandJenkinsAdvDiff": False,
    "config_land_ice_flux_attenuation_coefficient": 10.0, "config_land_ice_flux_boundaryLayerThickness": 10.0,
    "config_land_ice_flux_boundaryLayerNeighborWeight": 0.0, "config_land_ice_flux_cp_ice": 2.009e3,
    "config_land_ice_flux_rho_ice": 918.0, "config_land_ice_flux_topDragCoeff": 2.5e-3,
    "config_land_ice_flux_ISOMIP_gammaT": 1.0e-4, "config_land_ice_flux_rms_tidal_velocity": 5.0e-2,
    "config_land_ice_flux_jenkins_heat_transfer_coefficient": 0.011,
    "config_land_ice_flux_jenkins_salt_transfer_coefficient": 3.1e-4,
    "config_density0": 1026.0,
}
#: the state's two-level fields by package: landIceFluxesPKG ('standalone', 'coupled') and landIceCouplingPKG
TWO_LEVEL_FLUXES = ("accumulatedLandIceMass", "accumulatedLandIceHeat")
TWO_LEVEL_COUPLING = ("effectiveDensityInLandIce",)
_FREEZING = ("coeff_0", "coeff_S", "coeff_p", "coeff_pS", "reference_pressure")


def module_source(precision: str = "r4") -> str:
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(PRECISIONS)}")
    body = files("woof.hex.mpas_ocean").joinpath("land_ice_v842.cu").read_text(encoding="utf-8")
    return ("#define MPAS_RKIND8 1\n" if precision == "r8" else "") + body


@lru_cache(maxsize=None)
def _module_on(device: int, precision: str, options: tuple[str, ...]):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(precision), options, None, f"ocn_land_ice_v842_{precision}.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _launch(precision: str, name: str, n: int, args: tuple) -> None:
    import cupy as cp
    if n > 0:
        options = launch_options()
        _module_on(cp.cuda.Device().id, precision, options).get_function(name)(
            ((n + _BLOCK - 1) // _BLOCK,), (_BLOCK,), args)


def _config(configs: dict, name: str):
    return configs.get(name, DEFAULT_CONFIGS[name])


def mode(configs: dict) -> str:
    return _config(configs, "config_land_ice_flux_mode")


def fluxes_on(configs: dict) -> bool:
    """``landIceFluxesOn`` as ``ocn_surface_land_ice_fluxes_init`` sets it."""
    return mode(configs) in ("standalone", "coupled")


def check_config(configs: dict) -> None:
    """Refuse, by name, a land-ice configuration this port does not carry."""
    m = mode(configs)
    if m not in MODES:
        raise ConfigurationRefusal(
            knob="config_land_ice_flux_mode", value=m,
            reason="v8.4.2 allocates no land-ice fields for another mode",
            declaration="config_land_ice_flux_mode = 'off' | 'pressure_only' | 'standalone' | 'coupled'")
    if not fluxes_on(configs):
        return
    formulation = _config(configs, "config_land_ice_flux_formulation")
    if formulation == "HollandJenkins":
        raise ConfigurationRefusal(
            knob="config_land_ice_flux_formulation", value=formulation,
            reason="the Holland and Jenkins transfer velocities call libm's log and pow, which are not ported",
            declaration="config_land_ice_flux_formulation = 'ISOMIP' | 'Jenkins'")
    if formulation not in FORMULATIONS:
        raise ConfigurationRefusal(
            knob="config_land_ice_flux_formulation", value=formulation,
            reason="v8.4.2 stops on a formulation other than 'ISOMIP', 'Jenkins' or 'HollandJenkins'",
            declaration="config_land_ice_flux_formulation = 'ISOMIP' | 'Jenkins'")
    if (m == "standalone" and formulation == "Jenkins"
            and _config(configs, "config_land_ice_flux_useHollandJenkinsAdvDiff")):
        raise ConfigurationRefusal(
            knob="config_land_ice_flux_useHollandJenkinsAdvDiff", value=True,
            reason="compute_HJ99_melt_fluxes (the Holland and Jenkins advection-diffusion fluxes) is not ported",
            declaration="config_land_ice_flux_useHollandJenkinsAdvDiff = .false.")


def two_level(configs: dict) -> tuple[str, ...]:
    """The land-ice state fields with two time levels under ``configs``."""
    m = mode(configs)
    return (TWO_LEVEL_FLUXES if m in ("standalone", "coupled") else ()) + (
        TWO_LEVEL_COUPLING if m == "coupled" else ())


def _cavity(configs: dict, real):
    import cupy as cp
    from .eos_v842 import _configs as eos_configs
    c = eos_configs({k: v for k, v in configs.items() if "freezing_temperature" in k})
    return cp.asarray(np.array([c[f"config_land_ice_cavity_freezing_temperature_{n}"] for n in _FREEZING],
                               dtype=real))


def input_fields(*, mesh: dict, state: dict, tracers: dict, diagnostics: dict, forcing: dict,
                 configs: dict) -> None:
    """The flux part of ``ocn_compute_land_ice_flux_input_fields`` ('standalone',
    'coupled'), in place on ``diagnostics``; the land-ice mask is the caller's.

    ``state`` holds the time level's ``normalVelocity`` and ``layerThickness``;
    ``tracers`` its ``activeTracers`` with their indices; ``mesh`` the counts,
    ``cellsOnEdge``, ``cellsOnCell``, ``nEdgesOnCell``, ``maxLevelCell`` and
    ``cellMask``.
    """
    import cupy as cp
    if not fluxes_on(configs):
        return
    check_config(configs)
    h = state["layerThickness"]
    precision = _precision_of(h)
    real = PRECISIONS[precision]
    i32 = np.int32
    m, d, f = mesh, diagnostics, forcing
    nl = int(m["nVertLevels"])
    n_cells, n_edges = int(m["nCells"]), int(m["nEdges"])
    me = int(m["cellsOnCell"].shape[1])
    act = tracers["activeTracers"]
    rho_sw = real(_config(configs, "config_density0"))
    coeff = real(_config(configs, "config_land_ice_flux_topDragCoeff"))
    _launch(precision, "ocn_li_top_drag_v842", n_edges, (
        d["topDrag"], d["kineticEnergyCell"], f["landIceFraction"], state["normalVelocity"], m["cellsOnEdge"],
        i32(n_edges), i32(nl), rho_sw, coeff))
    _launch(precision, "ocn_li_friction_v842", n_cells, (
        d["topDragMagnitude"], d["landIceFrictionVelocity"], d["kineticEnergyCell"], f["landIceFraction"],
        i32(n_cells), i32(nl), rho_sw, coeff, real(_config(configs, "config_land_ice_flux_rms_tidal_velocity"))))
    # the scratch fields hold every cell and the missing one (zero: its weight is cellMask's 0)
    bl_t = cp.zeros(n_cells + 1, dtype=real)
    bl_s = cp.zeros(n_cells + 1, dtype=real)
    _launch(precision, "ocn_li_boundary_layer_v842", n_cells, (
        bl_t, bl_s, h, act, m["maxLevelCell"], i32(n_cells), i32(nl), i32(act.shape[2]),
        i32(tracers["index_temperature"]), i32(tracers["index_salinity"]),
        real(_config(configs, "config_land_ice_flux_boundaryLayerThickness"))))
    _launch(precision, "ocn_li_boundary_layer_tracers_v842", n_cells, (
        d["landIceBoundaryLayerTracers"], bl_t, bl_s, m["cellsOnCell"], m["nEdgesOnCell"], m["cellMask"],
        i32(n_cells), i32(nl), i32(me), i32(d["index_landIceBoundaryLayerTemperature"]),
        i32(d["index_landIceBoundaryLayerSalinity"]),
        real(_config(configs, "config_land_ice_flux_boundaryLayerNeighborWeight"))))
    if _config(configs, "config_land_ice_flux_formulation") == "Jenkins":
        _launch(precision, "ocn_li_jenkins_transfer_v842", n_cells, (
            d["landIceTracerTransferVelocities"], d["landIceFrictionVelocity"], i32(n_cells),
            i32(d["index_landIceHeatTransferVelocity"]), i32(d["index_landIceSaltTransferVelocity"]),
            real(_config(configs, "config_land_ice_flux_jenkins_heat_transfer_coefficient")),
            real(_config(configs, "config_land_ice_flux_jenkins_salt_transfer_coefficient"))))
    _launch(precision, "ocn_li_attenuation_v842", n_cells, (
        d["surfaceFluxAttenuationCoefficient"], f["landIceMask"], i32(n_cells),
        real(_config(configs, "config_land_ice_flux_attenuation_coefficient"))))


def build_arrays(*, dt: float, mesh: dict, state: dict, diagnostics: dict, forcing: dict, configs: dict) -> None:
    """``ocn_surface_land_ice_fluxes_build_arrays`` ('standalone' only), in place.

    ``state`` holds ``accumulatedLandIceMass`` and ``accumulatedLandIceHeat``
    at both levels (``NAME``, ``NAME@2``); ``diagnostics`` the boundary-layer
    tracers and transfer velocities with their indices; ``forcing`` the
    land-ice pressure, fraction and mask, the interface tracers (with their
    indices) and the three fluxes (written). ``dt`` is the clock's step.
    """
    import cupy as cp
    if mode(configs) != "standalone":
        return
    check_config(configs)
    d, f, st = diagnostics, forcing, state
    precision = _precision_of(f["landIceFreshwaterFlux"])
    real = PRECISIONS[precision]
    i32 = np.int32
    n = int(mesh["nCellsSolve"])
    params = cp.asarray(np.array([
        _config(configs, "config_density0"), CP_SW, LATENT_HEAT_FUSION_MKS,
        _config(configs, "config_land_ice_flux_ISOMIP_gammaT"), _config(configs, "config_land_ice_flux_cp_ice")],
        dtype=real))
    cavity = _cavity(configs, real)
    fluxes = (f["landIceInterfaceTracers"], f["landIceFreshwaterFlux"], f["landIceHeatFlux"], f["heatFluxToLandIce"])
    if _config(configs, "config_land_ice_flux_formulation") == "ISOMIP":
        _launch(precision, "ocn_li_isomip_v842", n, (
            *fluxes, d["landIceBoundaryLayerTracers"], f["landIcePressure"], f["landIceFraction"], f["landIceMask"],
            params, cavity, i32(n), i32(d["index_landIceBoundaryLayerTemperature"]),
            i32(f["index_landIceInterfaceTemperature"])))
    else:
        _launch(precision, "ocn_li_jenkins_melt_v842", n, (
            *fluxes, d["landIceBoundaryLayerTracers"], d["landIceTracerTransferVelocities"], f["landIcePressure"],
            f["landIceFraction"], f["landIceMask"], params, cavity, i32(n),
            i32(d["index_landIceBoundaryLayerTemperature"]), i32(d["index_landIceBoundaryLayerSalinity"]),
            i32(d["index_landIceHeatTransferVelocity"]), i32(d["index_landIceSaltTransferVelocity"]),
            i32(f["index_landIceInterfaceTemperature"]), i32(f["index_landIceInterfaceSalinity"])))
    _launch(precision, "ocn_li_accumulate_v842", n, (
        st["accumulatedLandIceMass@2"], st["accumulatedLandIceHeat@2"], st["accumulatedLandIceMass"],
        st["accumulatedLandIceHeat"], f["landIceFreshwaterFlux"], f["heatFluxToLandIce"], i32(n), real(dt)))


def thick_tend(*, forcing: dict, surface_thickness_flux, n_cells: int, configs: dict) -> None:
    """``ocn_surface_land_ice_fluxes_thick``: the land-ice freshwater flux added
    into ``surface_thickness_flux`` over cells 1..``n_cells`` (the mesh's nCells)."""
    if not fluxes_on(configs):
        return
    precision = _precision_of(surface_thickness_flux)
    real = PRECISIONS[precision]
    _launch(precision, "ocn_li_thick_v842", n_cells, (
        surface_thickness_flux, forcing["landIceFreshwaterFlux"], np.int32(n_cells),
        real(_config(configs, "config_density0"))))


def vel_tend(*, diagnostics: dict, surface_stress, surface_stress_magnitude, n_edges: int, n_cells: int,
             configs: dict) -> None:
    """``ocn_surface_land_ice_fluxes_vel``: the top drag added into the surface
    stress (edges 1..``n_edges``) and its magnitude (cells 1..``n_cells``)."""
    if not fluxes_on(configs):
        return
    precision = _precision_of(surface_stress)
    _launch(precision, "ocn_li_vel_edges_v842", n_edges, (surface_stress, diagnostics["topDrag"], np.int32(n_edges)))
    _launch(precision, "ocn_li_vel_cells_v842", n_cells, (
        surface_stress_magnitude, diagnostics["topDragMagnitude"], np.int32(n_cells)))


def tracers_tend(*, forcing: dict, tracers_surface_flux, n_cells: int, configs: dict) -> None:
    """``ocn_surface_land_ice_fluxes_tracers`` for ``activeTracers``: the land-ice
    heat flux added into the first tracer's surface flux, cells 1..``n_cells``."""
    if not fluxes_on(configs):
        return
    precision = _precision_of(tracers_surface_flux)
    real = PRECISIONS[precision]
    _launch(precision, "ocn_li_tracers_v842", n_cells, (
        tracers_surface_flux, forcing["landIceHeatFlux"], np.int32(n_cells), np.int32(tracers_surface_flux.shape[1]),
        real(_config(configs, "config_density0")), real(CP_SW)))


def effective_density_update(*, mesh: dict, forcing: dict, state: dict, configs: dict) -> None:
    """``ocn_effective_density_in_land_ice_update`` ('coupled' only): level 2's
    ``effectiveDensityInLandIce`` from level 2's ``ssh`` and level 1's density."""
    import cupy as cp
    if mode(configs) != "coupled":
        return
    m, st = mesh, state
    cur = st["effectiveDensityInLandIce"]
    precision = _precision_of(cur)
    i32 = np.int32
    n_cells, nl = int(m["nCells"]), int(m["nVertLevels"])
    scratch = cp.zeros(n_cells + 1, dtype=PRECISIONS[precision])
    _launch(precision, "ocn_li_effective_density_scratch_v842", n_cells, (
        scratch, forcing["landIcePressure"], st["ssh@2"], cur, forcing["landIceMask"], i32(n_cells)))
    _launch(precision, "ocn_li_effective_density_v842", n_cells, (
        st["effectiveDensityInLandIce@2"], scratch, m["cellsOnCell"], m["nEdgesOnCell"], m["cellMask"], i32(n_cells),
        i32(nl), i32(m["cellsOnCell"].shape[1])))


def coupled_means(*, diagnostics: dict, forcing: dict, effective_density, n_cells: int, n_accumulated: int,
                  configs: dict) -> None:
    """The land-ice part of ``ocn_time_average_coupled_accumulate`` ('coupled'
    only): the running means over cells 1..``n_cells`` before the count moves."""
    if mode(configs) != "coupled":
        return
    precision = _precision_of(effective_density)
    _launch(precision, "ocn_li_coupled_means_v842", n_cells, (
        forcing["avgLandIceBoundaryLayerTracers"], forcing["avgLandIceTracerTransferVelocities"],
        forcing["avgEffectiveDensityInLandIce"], diagnostics["landIceBoundaryLayerTracers"],
        diagnostics["landIceTracerTransferVelocities"], effective_density, np.int32(n_cells),
        np.int32(n_accumulated)))
