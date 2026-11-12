"""MALI v8.4.2: subglacial hydrology (``li_SGH_solve``) on the GPU.

:func:`sgh_solve` is ``li_SGH_solve`` (``mpas_li_subglacial_hydro.F``) for
one block: the flow-law parameter, then subcycles of the hydrology step until
the master time step is used up. Each subcycle:

1. sets the basal melt input ('file', 'thermal' or 'basal_heat') and the SHMIP
   forcing ('none', C1-C4, D1-D5);
2. updates the till;
3. runs ``calc_edge_quantities`` and reconstructs the cell water velocity;
4. runs ``update_channel`` (with channels on);
5. runs ``check_timestep``, which picks the subcycle's step on the host at the
   step's precision;
6. computes the sheet divergence and runs ``evolve_channel``;
7. runs ``calc_pressure`` ('cavity' or 'overburden') with its diagnostics;
8. updates the sheet thickness.

The kernels (``subglacial_hydro_v842.cu``) transcribe the routines. Shared
work comes from the other MALI units:

- ``li_calculate_flowParamA`` and the vertex interpolations are
  :mod:`~woof.hex.mpas_ice.sia_v842`'s;
- ``mpas_reconstruct`` is
  :mod:`~woof.hex.mpas_ice.time_integration_fe_v842`'s.

Real powers use the shared glibc ``powf``/``pow`` transcriptions.

Refused by name:

- at RKIND 8, the SHMIP forcings, whose ``sin``/``cos`` are glibc's binary64
  ones (LGPL, and not correctly rounded);
- the options v8.4.2 reports as errors.

Every word is held to the forward-Euler step compiled verbatim with gfortran
``-O0 -ffp-contract=off`` at RKIND 4 and RKIND 8
(``tests/test_mali_subglacial_hydro_parity.py``).
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import numpy as np

from woof.hex import numerics as _hex_numerics
from woof.hex.errors import ConfigurationRefusal

from . import rkind, sia_v842

SOURCE_FILES = ("src/core_landice/mode_forward/mpas_li_subglacial_hydro.F:206-1755",)
COMPILE_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false",
                   "--prec-div=true", "--prec-sqrt=true")
MODULE_OPTIONS = COMPILE_OPTIONS
MATH_HEADERS_R4 = ("glibc_flt32.cuh", "glibc_trig_flt32.cuh")
BASAL_MELT = {"file": 0, "thermal": 1, "basal_heat": 2}
PRESSURE_CALC = {"cavity": 1, "overburden": 2}
TANGENT = {"from_vertex_barycentric": 1, "from_vertex_barycentric_kiteareas": 2, "from_normal_slope": 3}
SHMIP_C = {"C1": 0.25, "C2": 0.5, "C3": 1.0, "C4": 2.0}
SHMIP_D = {"D1": -4.0, "D2": -2.0, "D3": 0.0, "D4": 2.0, "D5": 4.0}
_BLOCK = 128


def module_source(precision: str = "r4") -> str:
    rkind.require(precision)
    body = files("woof.hex.mpas_ice").joinpath("subglacial_hydro_v842.cu").read_text(encoding="utf-8")
    if precision == "r8":
        return rkind.r8_prelude() + body
    import woof.core.kernels as kernels
    root = Path(kernels.__file__).parent
    return "\n".join((root / name).read_text(encoding="utf-8") for name in MATH_HEADERS_R4) + "\n" + body


@lru_cache(maxsize=None)
def _module_on(device: int, precision: str, options: tuple[str, ...]):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(precision), options, None,
                                   f"li_subglacial_hydro_v842_{precision}.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _function(precision: str, name: str):
    import cupy as cp
    return _module_on(cp.cuda.Device().id, precision,
                      _hex_numerics.options(COMPILE_OPTIONS, keep_ftz=True)).get_function(name)


def _grid(n: int) -> tuple[int]:
    return (max(1, (n + _BLOCK - 1) // _BLOCK),)


def _refuse(knob: str, value, reason: str, declaration: str):
    raise ConfigurationRefusal(knob=knob, value=value, reason=reason, declaration=declaration)


def check_config(config: dict, precision: str) -> None:
    """Refuse what li_SGH_solve reports as an error, or cannot be matched here."""
    for knob, table in (("config_SGH_basal_melt", BASAL_MELT), ("config_SGH_pressure_calc", PRESSURE_CALC),
                        ("config_SGH_tangent_slope_calculation", TANGENT)):
        if config[knob] not in table:
            _refuse(knob, config[knob], "not an option li_SGH_solve carries (the model logs it as an error)",
                    f"{knob} = " + " | ".join(map(repr, table)))
    forcing = config.get("config_SGH_shmip_forcing", "none")
    if forcing != "none" and forcing not in SHMIP_C and forcing not in SHMIP_D:
        _refuse("config_SGH_shmip_forcing", forcing, "not a SHMIP experiment the model carries",
                "config_SGH_shmip_forcing = 'none' | 'C1'..'C4' | 'D1'..'D5'")
    if precision == "r8" and forcing != "none":
        _refuse("config_SGH_shmip_forcing", forcing,
                "at RKIND 8 the SHMIP forcings call glibc's binary64 sin and cos, which are LGPL and not "
                "correctly rounded, so their words cannot be reproduced under the default licence",
                "config_SGH_shmip_forcing = 'none' at RKIND 8")


def sgh_solve(*, mesh: dict, geometry: dict, thermal: dict, velocity: dict, hydro: dict, config: dict) -> int:
    """``li_SGH_solve`` for one block, in place on ``hydro`` (and
    ``velocity["flowParamA"]``). ``mesh`` holds the step's mesh fields as for
    :func:`woof.hex.mpas_ice.time_integration_fe_v842.forward_euler_step`;
    ``hydro`` holds MALI's hydro pool, with ``deltatSGH`` and the five
    ``deltatSGH*`` limits as one-word device arrays. Returns the number of
    subcycles."""
    import cupy as cp
    from . import time_integration_fe_v842 as fe
    if not config.get("config_SGH", False):
        return 0
    m, g, t, v, h, c = mesh, geometry, thermal, velocity, hydro, config
    precision = rkind.precision_of(g["thickness"])
    real = rkind.PRECISIONS[precision]
    check_config(c, precision)
    n_cells, n_levels = t["temperature"].shape
    n_edges = m["dcEdge"].shape[0]
    n_vertices = m["baryCellsOnVertex"].shape[0]
    max_edges = m["edgesOnCell"].shape[1]
    max_edges2 = m["edgesOnEdge"].shape[1]
    ncs, nes = int(m["nCellsSolve"]), int(m["nEdgesSolve"])
    nc, ne, nl = np.int32(n_cells), np.int32(n_edges), np.int32(n_levels)
    cells, edges, block = _grid(n_cells), _grid(n_edges), (_BLOCK,)
    channels = bool(c["config_SGH_chnl_active"])
    r = real
    rhoi, rhoo, sea = r(c["config_ice_density"]), r(c["config_ocean_density"]), r(c["config_sea_level"])
    fn = lambda name: _function(precision, name)  # noqa: E731
    # li_calculate_flowParamA, as li_sia_solve computes it
    flow = sia_v842._option(sia_v842.FLOW_PARAM_A_CALCULATIONS, "config_flowParamA_calculation",
                            c["config_flowParamA_calculation"])
    sia_v842._function(precision, "li_calculate_flowParamA_v842")(cells, block, (
        v["flowParamA"], t["temperature"], g["thickness"], m["layerCenterSigma"], np.int32(flow),
        r(c["config_default_flowParamA"]), r(c["config_enhancementFactor"]), rhoi, nc, nl))
    forcing = c.get("config_SGH_shmip_forcing", "none")
    melt = np.int32(BASAL_MELT[c["config_SGH_basal_melt"]])
    tangent = TANGENT[c["config_SGH_tangent_slope_calculation"]]
    days = r(m["daysSinceStart"].get()[0])
    time_left = r(m["deltat"].get()[0])
    limits = {name: cp.empty(max(1, nes), real) for name in ("advec", "diffu", "pressure", "advecC", "diffuC")}
    word = cp.empty(1, real)
    junk = [cp.empty(n_cells, real) for _ in range(3)]
    subcycles = 0
    while time_left > r(0.0):
        subcycles += 1
        dt = h["deltatSGH"]
        fn("li_sgh_basal_melt_input_v842")(cells, block, (
            h["basalMeltInput"], g["groundedBasalMassBal"], t["basalHeatFlux"], t["basalFrictionFlux"], melt, nc))
        if forcing != "none":
            kind = 1 if forcing in SHMIP_C else 2
            fn("li_sgh_shmip_forcing_v842")(cells, block, (
                h["basalMeltInput"], h["externalWaterInput"], m["areaCell"], g["upperSurface"], days, np.int32(kind),
                r(SHMIP_C.get(forcing, 0.0)), r(SHMIP_D.get(forcing, 0.0)), nc))
        dt_word = r(dt.get()[0])
        fn("li_sgh_till_v842")(cells, block, (
            h["tillWaterThickness"], h["tillWaterThicknessOld"], h["basalMeltInput"], h["externalWaterInput"],
            g["cellMask"], dt_word, r(c["config_SGH_till_drainage"]), r(c["config_SGH_till_max"]), nc))
        # calc_edge_quantities
        fn("li_sgh_edge_normals_v842")(edges, block, (
            h["waterThicknessEdge"], h["hydropotentialBaseSlopeNormal"], h["waterPressureSlopeNormal"],
            h["waterThickness"], h["hydropotentialBase"], h["waterPressure"], g["cellMask"], m["cellsOnEdge"],
            m["dcEdge"], ne))
        if tangent in (1, 2):
            sia_v842._function(precision, "li_upper_surface_vertex_v842")(_grid(n_vertices), block, (
                h["hydropotentialBaseVertex"], h["hydropotentialBase"], m["baryCellsOnVertex"],
                m["baryWeightsOnVertex"], m["cellsOnVertex"], m["kiteAreasOnVertex"], np.int32(tangent),
                np.int32(n_vertices)))
        fn("li_sgh_tangent_v842")(edges, block, (
            h["hydropotentialBaseSlopeTangent"], h["hydropotentialBaseSlopeNormal"], h["hydropotentialBaseVertex"],
            m["verticesOnEdge"], m["dvEdge"], g["edgeMask"], m["nEdgesOnEdge"], m["edgesOnEdge"],
            m["weightsOnEdge"], np.int32(tangent), ne, np.int32(nes), np.int32(max_edges2)))
        fn("li_sgh_edge_fluxes_v842")(edges, block, (
            h["gradMagPhiEdge"], h["effectiveConducEdge"], h["diffusivity"], h["waterVelocity"],
            h["waterThicknessEdgeUpwind"], h["waterFluxAdvec"], h["waterFluxDiffu"], h["waterFlux"],
            h["hydropotentialBaseSlopeNormal"], h["hydropotentialBaseSlopeTangent"], h["waterThicknessEdge"],
            h["waterThickness"], g["cellMask"], h["waterFluxMask"], m["cellsOnEdge"], m["dcEdge"],
            r(c["config_SGH_conduc_coeff"]), r(c["config_SGH_alpha"]), r(c["config_SGH_beta"]), ne))
        # mpas_reconstruct (the step's kernel), then zero off grounded ice
        fe._function(precision, "li_reconstruct_1d_v842")(_grid(ncs), block, (
            h["waterVelocityCellX"], h["waterVelocityCellY"], junk[0], junk[1], junk[2], h["waterVelocity"],
            m["coeffs_reconstruct"], m["nEdgesOnCell"], m["edgesOnCell"], m["latCell"], m["lonCell"],
            np.int32(0), np.int32(ncs), np.int32(max_edges)))
        fn("li_sgh_mask_cell_velocity_v842")(cells, block, (
            h["waterVelocityCellX"], h["waterVelocityCellY"], g["cellMask"], nc))
        if channels:
            fn("li_sgh_update_channel_v842")(edges, block, (
                h["channelDischarge"], h["channelArea"], h["channelVelocity"], h["channelDiffusivity"],
                h["channelMelt"], h["channelPressureFreeze"], h["channelOpeningRate"], h["flowParamAChannel"],
                h["channelEffectivePressure"], h["channelClosingRate"], h["channelChangeRate"],
                h["gradMagPhiEdge"], h["hydropotentialBaseSlopeNormal"], h["waterFlux"],
                h["waterPressureSlopeNormal"], h["waterFluxMask"], g["edgeMask"], v["flowParamA"],
                h["effectivePressure"], m["cellsOnEdge"], r(c["config_SGH_chnl_conduc_coeff"]),
                r(c["config_SGH_chnl_alpha"]), r(c["config_SGH_chnl_beta"]),
                r(c["config_SGH_chnl_creep_coefficient"]), r(c["config_SGH_incipient_channel_width"]), rhoi,
                np.int32(bool(c["config_SGH_include_pressure_melt"])), ne, np.int32(nes), nl))
        # check_timestep
        fn("li_sgh_edge_limits_v842")(_grid(nes), block, (
            limits["advec"], limits["diffu"], limits["pressure"], limits["advecC"], limits["diffuC"], m["dcEdge"],
            h["waterVelocity"], h["diffusivity"], h["channelVelocity"], h["channelDiffusivity"],
            r(c["config_SGH_englacial_porosity"]), np.int32(channels), np.int32(nes)))

        def minimum(name, scale):
            fn("li_sgh_min_v842")((1,), (1,), (word, limits[name], np.int32(nes)))
            block_min = r(scale) * r(word.get()[0])
            big = r(1.0e36)
            return block_min if block_min < big else big   # min(bigNumber, block)

        def fmin(a, b):
            return b if b < a else a

        advec, diffu, pressure = minimum("advec", 0.5), minimum("diffu", 0.25), minimum("pressure", 1.0)
        for name, value in (("deltatSGHadvec", advec), ("deltatSGHdiffu", diffu), ("deltatSGHpressure", pressure)):
            h[name][...] = value
        proposed = fmin(fmin(advec, diffu), pressure)
        if channels:
            advec_c, diffu_c = minimum("advecC", 0.5), minimum("diffuC", 0.25)
            h["deltatSGHadvecChannel"][...] = advec_c
            h["deltatSGHdiffuChannel"][...] = diffu_c
            proposed = fmin(fmin(proposed, advec_c), diffu_c)
        proposed = proposed * r(c["config_SGH_adaptive_timestep_fraction"])
        proposed = fmin(proposed, r(c["config_SGH_max_adaptive_timestep"]))
        proposed = fmin(proposed, time_left)
        time_left = time_left - proposed
        if __import__("os").environ.get("SGH_DEBUG") and subcycles <= 5:
            print("   sub", subcycles, "advec", advec, "diffu", diffu, "pressure", pressure, "proposed", proposed, flush=True)
        if __import__("os").environ.get("SGH_DEBUG") and subcycles > 400:
            raise RuntimeError("too many subcycles")
        h["deltatSGH"][...] = proposed
        dt_word = proposed
        # the sheet divergence, then the channels
        fn("li_sgh_divergence_v842")(cells, block, (
            h["divergence"], h["waterFlux"], m["dvEdge"], m["areaCell"], g["cellMask"], m["nEdgesOnCell"],
            m["edgesOnCell"], m["edgeSignOnCell"], nc, np.int32(ncs), np.int32(max_edges)))
        if channels:
            fn("li_sgh_channel_cells_v842")(cells, block, (
                h["divergenceChannel"], h["channelAreaChangeCell"], h["channelDischarge"], h["channelChangeRate"],
                m["dcEdge"], m["areaCell"], m["nEdgesOnCell"], m["edgesOnCell"], m["edgeSignOnCell"], nc,
                np.int32(ncs), np.int32(max_edges)))
            fn("li_sgh_channel_area_v842")(edges, block, (h["channelArea"], h["channelChangeRate"], dt_word, ne))
        fn("li_sgh_pressure_v842")(cells, block, (
            h["waterPressure"], h["waterPressureOld"], h["waterPressureTendency"], h["openingRate"],
            h["closingRate"], h["zeroOrderSum"], h["effectivePressure"], h["hydropotentialBase"],
            h["hydropotential"], v["basalSpeed"], h["waterThickness"], v["flowParamA"], h["basalMeltInput"],
            h["externalWaterInput"], h["tillWaterThickness"], h["tillWaterThicknessOld"], h["divergence"],
            h["divergenceChannel"], h["channelAreaChangeCell"], g["thickness"], g["bedTopography"], g["cellMask"],
            r(c["config_SGH_bed_roughness"]), r(c["config_SGH_bed_roughness_max"]),
            r(c["config_SGH_creep_coefficient"]), r(c["config_SGH_englacial_porosity"]), rhoi, rhoo, sea, dt_word,
            np.int32(PRESSURE_CALC[c["config_SGH_pressure_calc"]]), nc, nl))
        fn("li_sgh_water_thickness_v842")(cells, block, (
            h["waterThickness"], h["waterThicknessOld"], h["waterThicknessTendency"], h["divergence"],
            h["basalMeltInput"], h["externalWaterInput"], h["divergenceChannel"], h["channelAreaChangeCell"],
            h["tillWaterThickness"], h["tillWaterThicknessOld"], g["cellMask"], dt_word, nc))
    return subcycles
