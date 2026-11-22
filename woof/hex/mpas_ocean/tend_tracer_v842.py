"""MPAS-Ocean v8.4.2: ``ocn_tend_tracer`` on the GPU.

:func:`tend_tracer` is ``ocn_tend_tracer`` (``mpas_ocn_tendency.F``) for the
active tracers on one block, in the Fortran's order:

1. the thickness flux ``normalTransportVelocity * layerThicknessEdge``;
2. the tendency and surface flux zeroed, then the bulk forcing
   (``ocn_surface_bulk_forcing_active_tracers``, with the bulk thickness flux
   and ``ocn_freezing_temperature`` when ``config_use_bulk_thickness_flux``)
   and the land-ice heat flux ('standalone', 'coupled';
   :func:`~woof.hex.mpas_ocean.land_ice_v842.tracers_tend`);
3. advection (:mod:`~woof.hex.mpas_ocean.tracer_advection_v842`) and
   horizontal mixing (:mod:`~woof.hex.mpas_ocean.tracer_hmix_v842`, with
   Redi's isoneutral mixing under ``config_use_standardGM`` on the slopes
   GM left in the diagnostics);
4. the surface flux into the column (``ocn_tracer_surface_flux_tend``, with
   the runoff flux);
5. Jerlov short-wave absorption (``config_sw_absorption_type`` 'jerlov') or
   none;
6. the active-tracer budgets (``config_compute_active_tracer_budgets``);
7. under ``config_use_cvmix_kpp``, ``ocn_compute_KPP_input_fields`` (the
   non-local surface fluxes, the surface buoyancy forcing, the surface
   stress magnitude and the friction velocity, after the equation of state
   at the surface-displaced type) and, unless
   ``config_cvmix_kpp_nonlocal_with_implicit_mix``, KPP's non-local tracer
   flux (``ocn_tracer_nonlocalflux_tend``, off under
   ``config_disable_tr_nonlocalflux``) with its budget;
8. frazil (``ocn_frazil_forcing_tracers``,
   :func:`~woof.hex.mpas_ocean.frazil_v842.active_tracers_tend`) under
   ``config_use_frazil_ice_formation``.

The kernels (``tend_tracer_v842.cu`` and those of the modules above) are held
word for word to the v8.4.2 routines, ``mpas_ocn_tendency.F`` compiled
verbatim with the real tracer modules, with gfortran ``-O0 -ffp-contract=off``
at RKIND 4 and 8 (``tools/mpas_ocean_oracle/run_tend_tracer.F90``,
``tests/test_mpas_ocean_tend_tracer_parity.py``).

The pools are dicts of device arrays named as in MPAS (``mesh``, ``state``,
``tracers``, ``tracersTend``, ``tracersSurfaceFlux``, ``diagnostics``,
``forcing``), with MPAS's missing-neighbour element; dimensions are host
integers, a field whose package is off is ``None``.

Refused by name, as not ported: surface and interior restoring, exponential decay, ideal age and TTD for the
active tracers, the ohlmann00 short-wave absorption, and every other tracer group
(this port carries ``activeTracers``). Short-wave absorption without the bulk
forcing raises, as v8.4.2's init aborts.
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import numpy as np

from woof.hex.errors import ConfigurationRefusal

from . import eos_v842, frazil_v842, land_ice_v842, tracer_advection_v842, tracer_hmix_v842
from .tend_v842 import COMPILE_OPTIONS, PRECISIONS, _precision_of, launch_options

SOURCE_FILES = (
    "src/core_ocean/shared/mpas_ocn_tendency.F:363-893",
    "src/core_ocean/shared/mpas_ocn_surface_bulk_forcing.F:355-500",
    "src/core_ocean/shared/mpas_ocn_tracer_surface_flux_to_tend.F",
    "src/core_ocean/shared/mpas_ocn_tracer_short_wave_absorption.F",
    "src/core_ocean/shared/mpas_ocn_tracer_short_wave_absorption_jerlov.F",
    "src/core_ocean/shared/mpas_ocn_diagnostics.F:1328-1526",
    "src/core_ocean/shared/mpas_ocn_tracer_nonlocalflux.F",
)
MODULE_OPTIONS = COMPILE_OPTIONS
MATH_HEADERS = {"r4": ("glibc_flt32.cuh",), "r8": ("glibc_flt64.cuh",)}
#: ocn_tracer_short_wave_absorption_jerlov's water types: (rfac, depth1, depth2)
JERLOV = ((0.58, 0.35, 23.0), (0.62, 0.60, 20.0), (0.67, 1.00, 17.0), (0.77, 1.50, 14.0), (0.78, 1.40, 7.90))
CP_SW = 3.996e3
#: mpas_constants' gravity
GRAVITY = 9.80616
_UNPORTED_GROUP_SWITCHES = ("surface_restoring", "interior_restoring", "exponential_decay", "idealAge_forcing",
                            "ttd_forcing")
_BLOCK = 128


def module_source(precision: str = "r4") -> str:
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(PRECISIONS)}")
    import woof.core.kernels as kernels
    root = Path(kernels.__file__).parent
    math = "\n".join((root / name).read_text(encoding="utf-8") for name in MATH_HEADERS[precision])
    body = files("woof.hex.mpas_ocean").joinpath("tend_tracer_v842.cu").read_text(encoding="utf-8")
    return ("#define MPAS_RKIND8 1\n" if precision == "r8" else "") + math + "\n" + body


@lru_cache(maxsize=None)
def _module_on(device: int, precision: str, options: tuple[str, ...]):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(precision), options, None, f"ocn_tend_tracer_v842_{precision}.cu")
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


def _refuse(knob: str, value, reason: str, declaration: str):
    raise ConfigurationRefusal(knob=knob, value=value, reason=reason, declaration=declaration)


def check_config(configs: dict, groups=("activeTracers",)) -> None:
    """Refuse, by name, what this port does not carry or v8.4.2 cannot run."""
    for group in groups:
        if group != "activeTracers":
            _refuse("tracer group", group, "only the active tracers' tendency is ported",
                    "the tracers pool holds activeTracers alone")
    for switch in _UNPORTED_GROUP_SWITCHES:
        knob = f"config_use_activeTracers_{switch}"
        if configs.get(knob, False):
            _refuse(knob, True, "not ported", f"{knob} = .false.")
    land_ice_v842.check_config(configs)
    sw = configs.get("config_sw_absorption_type", "none")
    if sw not in ("none", "jerlov"):
        _refuse("config_sw_absorption_type", sw,
                "the variable (ohlmann00) short-wave absorption is not ported; v8.4.2 aborts on any other value",
                "config_sw_absorption_type = 'none' | 'jerlov'")
    if sw != "none" and not configs.get("config_use_activeTracers_surface_bulk_forcing", False):
        raise ValueError("short-wave absorption is on with the bulk forcing off (v8.4.2's init aborts): set "
                         "config_sw_absorption_type to 'none' or enable activeTracers_surface_bulk_forcing")
    if not 1 <= int(configs.get("config_jerlov_water_type", 3)) <= 5:
        raise ValueError("config_jerlov_water_type must lie within 1..5")
    tracer_advection_v842.check_config(configs)
    tracer_hmix_v842.check_config(configs)


def tend_tracer(*, dt: float, mesh: dict, state: dict, tracers: dict, tracers_tend: dict,
                tracers_surface_flux: dict, diagnostics: dict, forcing: dict, configs: dict,
                tracers_level1: dict | None = None) -> None:
    """``ocn_tend_tracer`` for the active tracers, in place on ``tracers_tend``,
    ``tracers_surface_flux``, the forcing's penetrative flux and the budgets.

    With ``config_use_cvmix_kpp``, ``ocn_compute_KPP_input_fields`` reads
    ``tracers_level1`` (time level 1's tracers; ``tracers`` when not given)
    for its surface fluxes, as the Fortran does, and ``tracers`` for its
    equation of state."""
    import cupy as cp
    check_config(configs)
    if configs.get("config_disable_tr_all_tend", False):
        return
    h = state["layerThickness"]
    precision = _precision_of(h)
    real = PRECISIONS[precision]
    i32 = np.int32
    nl = int(mesh["nVertLevels"])
    n_cells, n_edges = int(mesh["nCells"]), int(mesh["nEdges"])
    cells = [int(x) for x in mesh["nCellsArray"]]
    group = tracers["activeTracers"]
    tend = tracers_tend["activeTracersTend"]
    flux = tracers_surface_flux["activeTracersSurfaceFlux"]
    runoff = tracers_surface_flux["activeTracersSurfaceFluxRunoff"]
    removed = tracers_surface_flux["activeTracersSurfaceFluxRemoved"]
    nt = group.shape[2]
    it, isal = int(tracers["index_temperature"]), int(tracers["index_salinity"])
    budgets = bool(configs.get("config_compute_active_tracer_budgets", False))
    d, f = diagnostics, forcing

    thickness_flux = cp.empty_like(d["normalTransportVelocity"])
    _launch(precision, "ocn_tt_thickness_flux_v842", (n_edges + 1) * nl, (
        thickness_flux, d["normalTransportVelocity"], d["layerThicknessEdge"], i32((n_edges + 1) * nl)))
    # the loop's extent is the mesh's nEdges; the missing edge's row follows
    # the same product (MPAS's scratch field holds it uninitialised)
    if not configs.get("config_use_activeTracers", True):
        return
    tend[:n_cells] = 0
    flux[:n_cells] = 0
    rho_sw = real(configs["config_density0"])
    if configs.get("config_use_activeTracers_surface_bulk_forcing", False):
        runoff[:n_cells] = 0
        removed[:n_cells] = 0
        hflux = real(1.0) / (rho_sw * real(CP_SW))
        c = eos_v842._configs(configs)
        freeze = [real(c[f"config_open_ocean_freezing_temperature_{name}"])
                  for name in ("coeff_0", "coeff_S", "coeff_p", "coeff_pS", "reference_pressure")]
        names = ("latentHeatFlux", "sensibleHeatFlux", "longWaveHeatFluxUp", "longWaveHeatFluxDown", "seaIceHeatFlux",
                 "rainFlux", "snowFlux", "shortWaveHeatFlux", "evaporationFlux", "seaIceFreshWaterFlux",
                 "seaIceSalinityFlux", "iceRunoffFlux", "riverRunoffFlux")
        _launch(precision, "ocn_tt_bulk_forcing_active_tracers_v842", cells[2], (
            flux, runoff, removed, f["penetrativeTemperatureFlux"], group, h, *(f[name] for name in names),
            i32(cells[2]), i32(nl), i32(nt), i32(it), i32(isal),
            i32(bool(configs.get("config_use_bulk_thickness_flux", False))), real(dt), hflux, real(1.0), rho_sw,
            *freeze))
    land_ice_v842.tracers_tend(forcing=f, tracers_surface_flux=flux, n_cells=n_cells, configs=configs)
    tracer_advection_v842.tracer_advection_tend(
        tracers=group, normal_thickness_flux=thickness_flux, w=d["vertAleTransportTop"], layer_thickness=h, dt=dt,
        mesh=mesh, tend=tend, configs=configs, horizontal_budget=d.get("activeTracerHorizontalAdvectionTendency"),
        vertical_budget=d.get("activeTracerVerticalAdvectionTendency"))
    tracer_hmix_v842.tracer_hmix_tend(tracers=group, layer_thickness_edge=d["layerThicknessEdge"], mesh=mesh,
                                      tend=tend, configs=configs, diagnostics=d)
    if budgets:
        d["activeTracerSurfaceFluxTendency"][:n_cells] = tend[:n_cells]
    if not configs.get("config_disable_tr_sflux", False):
        for surface, fraction in ((flux, f["fractionAbsorbed"]), (runoff, f["fractionAbsorbedRunoff"])):
            _launch(precision, "ocn_tt_surface_flux_v842", cells[0], (
                tend, surface, fraction, mesh["maxLevelCell"], i32(cells[0]), i32(nl), i32(nt)))
    if budgets:
        sf = d["activeTracerSurfaceFluxTendency"]
        sf[:n_cells] = tend[:n_cells] - sf[:n_cells]
        d["temperatureShortWaveTendency"][:n_cells] = tend[:n_cells, :, it - 1]
    if configs.get("config_sw_absorption_type", "none") == "jerlov":
        rfac, depth1, depth2 = (real(x) for x in JERLOV[int(configs.get("config_jerlov_water_type", 3)) - 1])
        weights = cp.zeros((h.shape[0], nl + 1), dtype=real)
        _launch(precision, "ocn_tt_jerlov_v842", cells[2], (
            tend, d["penetrativeTemperatureFluxOBL"], weights, f["penetrativeTemperatureFlux"], h,
            mesh["maxLevelCell"], i32(cells[2]), i32(nl), i32(nt), i32(it), rfac, depth1, depth2,
            real(abs(real(configs.get("config_surface_buoyancy_depth", 1.0))))))
    if budgets:
        sw = d["temperatureShortWaveTendency"]
        sw[:n_cells] = tend[:n_cells, :, it - 1] - sw[:n_cells]
    if configs.get("config_use_cvmix_kpp", False):
        _kpp_input_fields(precision=precision, mesh=mesh, layer_thickness=h, tracers=tracers,
                          tracers_level1=tracers if tracers_level1 is None else tracers_level1,
                          tracers_surface_flux=tracers_surface_flux, diagnostics=d, forcing=f, configs=configs)
        if not configs.get("config_cvmix_kpp_nonlocal_with_implicit_mix", False):
            if budgets:
                d["activeTracerNonLocalTendency"][:n_cells] = tend[:n_cells]
            if not configs.get("config_disable_tr_nonlocalflux", False):
                vnl = d["vertNonLocalFlux"]
                _launch(precision, "ocn_tt_nonlocal_flux_v842", cells[0], (
                    tend, vnl, tracers_surface_flux["nonLocalSurfaceTracerFlux"], mesh["maxLevelCell"],
                    i32(cells[0]), i32(nl), i32(nt), i32(vnl.shape[-1])))
            if budgets:
                nonlocal_tend = d["activeTracerNonLocalTendency"]
                nonlocal_tend[:n_cells] = tend[:n_cells] - nonlocal_tend[:n_cells]
    frazil_v842.active_tracers_tend(mesh=mesh, forcing=f, tend=tend, index_temperature=it, index_salinity=isal,
                                    configs=configs)


def _kpp_input_fields(*, precision: str, mesh: dict, layer_thickness, tracers: dict, tracers_level1: dict,
                      tracers_surface_flux: dict, diagnostics: dict, forcing: dict, configs: dict) -> None:
    """``ocn_compute_KPP_input_fields`` over ``nCellsArray(3)``: the equation
    of state at the surface-displaced type into scratch (with the in-situ
    coefficients), then the non-local surface fluxes, the surface buoyancy
    forcing, the surface stress magnitude and the friction velocity."""
    import cupy as cp
    real = PRECISIONS[precision]
    i32 = np.int32
    d, f = diagnostics, forcing
    nl = int(mesh["nVertLevels"])
    cells = [int(x) for x in mesh["nCellsArray"]]
    group = tracers["activeTracers"]
    nt = group.shape[2]
    it, isal = int(tracers["index_temperature"]), int(tracers["index_salinity"])
    shape = (group.shape[0], nl)
    density, alpha, beta = (cp.zeros(shape, dtype=real) for _ in range(3))
    eos_v842.equation_of_state_density(
        mesh={"maxLevelCell": mesh["maxLevelCell"], "refBottomDepth": mesh["refBottomDepth"]}, active_tracers=group,
        tracers_surface_value=d["tracersSurfaceValue"], index_temperature=it, index_salinity=isal, density=density,
        k_displaced=0, displacement_type="surfaceDisplaced", configs=configs, thermal_expansion_coeff=alpha,
        saline_contraction_coeff=beta, n_cells=cells[2])
    _launch(precision, "ocn_tt_kpp_input_fields_v842", cells[2], (
        tracers_surface_flux["nonLocalSurfaceTracerFlux"], d["surfaceBuoyancyForcing"], f["surfaceStressMagnitude"],
        d["surfaceFrictionVelocity"], tracers_surface_flux["activeTracersSurfaceFlux"],
        tracers_surface_flux["activeTracersSurfaceFluxRunoff"], f["penetrativeTemperatureFlux"],
        d["penetrativeTemperatureFluxOBL"], f["surfaceThicknessFlux"], tracers_level1["activeTracers"],
        layer_thickness, alpha, beta, f["surfaceStress"], mesh["areaCell"],
        mesh["dcEdge"], mesh["dvEdge"], mesh["nEdgesOnCell"], mesh["edgesOnCell"], i32(cells[2]), i32(nl), i32(nt),
        i32(mesh["edgesOnCell"].shape[1]), i32(tracers_surface_flux["index_temperatureSurfaceFlux"]),
        i32(tracers_surface_flux["index_salinitySurfaceFlux"]),
        real(configs.get("config_flux_attenuation_coefficient", 0.001)),
        real(configs.get("config_flux_attenuation_coefficient_runoff", 0.001)), real(GRAVITY),
        real(configs["config_density0"])))
