"""MPAS-Seaice v8.4.2's column shortwave radiation on the GPU.

``column_prep_radiation`` (``mpas_seaice_column.F:1990-2104``: the shortwave
scaling, ``colpkg_prep_radiation``), ``column_radiation``
(``mpas_seaice_column.F:2118-2484``: ``colpkg_step_radiation``) and
``seaice_init_column_shortwave``'s albedo aggregation
(``mpas_seaice_column.F:435-647``), word for word with the gfortran -O0
-ffp-contract=off oracle at RKIND 8. One thread is one owned cell;
``column_radiation_v842.cu`` and ``column_dedd_v842.cu`` are the
transcription.

What it carries: the delta-Eddington shortwave with a computed surface
temperature (``config_shortwave_type = 'dEdd'``, the Registry default:
``colpkg_init_orbit``, ``compute_coszen`` and ``run_dEdd`` with the CESM,
level, topo or no melt ponds), the CCSM3 shortwave
(``config_shortwave_type = 'ccsm3'``: ``shortwave_ccsm3`` with the ``ccsm3``
or ``constant`` albedos and ``absorbed_solar``), and any shortwave type with a
coupled surface temperature (``config_calc_surface_temperature = .false.``:
no shortwave, the topo ponds' effective area), on BL99, mushy or zero-layer
ice.

What it refuses, by name:

- the delta-Eddington shortwave in the default build: the orbit and the
  solar zenith angle call glibc's binary64 ``sin``, ``cos``, ``atan`` and
  ``asin``, which are not correctly rounded and are LGPL, so they compile
  only with ``WOOF_GLIBC239_LIBM64=1`` (:mod:`woof.core.glibc239_gpu`);
- the aerosols and the shortwave bioabsorption (and, as every column stage,
  the biogeochemistry);
- ``shortwave_ccsm3`` with more categories than ice layers: v8.4.2 passes
  ``ncat`` as ``absorbed_solar``'s ``nilyr`` (``ice_shortwave.F90:236``) and
  then writes past the category's layers;
- the ``ccsm3`` albedos (the Registry default) in the default build:
  ``compute_albedos`` calls glibc's binary64 ``atan``, which is not correctly
  rounded and is LGPL, so it compiles only with ``WOOF_GLIBC239_LIBM64=1``
  (:mod:`woof.core.glibc239_gpu`).

v8.4.2's quirks are kept: with fewer categories than ice layers
``absorbed_solar`` spreads the light over ``ncat`` layers of thickness
``hi/ncat`` and the layers below keep 0; ``shortwave_ccsm3`` sets
``solarZenithAngleCosine`` to 0.5; ``colpkg_step_radiation`` zeroes
``bioTracerShortwave``; ``seaice_init_column_shortwave`` adds the bare-ice,
snow and pond albedos and the effective pond area onto their words without
resetting them. In the delta-Eddington shortwave,
``shortwave_dEdd_set_snow``'s ``intent(out)`` snow fraction is set only where
the snow is at least ``hs_min`` deep, so bare ice keeps the 0 ``run_dEdd``
stored (gfortran passes it by reference); ``shortwave_dEdd`` raises the solar
zenith cosine to ``puny`` and ``column_radiation`` writes it back; the
zero-heat-capacity fold of the top layers' absorption into the surface runs
on each surface type's sums so far; categories without ice keep their
bare-ice, snow and pond albedos.

``exp`` is ``glibc_exp`` (the MIT transcription of glibc's binary64 exp).
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import numpy as np

from woof.hex import numerics as _hex_numerics
from woof.hex.errors import ConfigurationRefusal

from . import column_v842 as cv
from .velocity_v842 import COMPILE_OPTIONS, _grid

SOURCE_FILES = (
    "src/core_seaice/shared/mpas_seaice_column.F:435-647,1990-2484",
    "src/core_seaice/column/ice_colpkg.F90:387-412,2547-2972",
    "src/core_seaice/column/ice_orbital.F90:35-680",
    "src/core_seaice/column/ice_shortwave.F90:72-660,673-3563",
)
_BLOCK = 128


def _refuse(knob: str, value, reason: str, declaration: str):
    raise ConfigurationRefusal(knob=knob, value=value, reason=reason, declaration=declaration)


def refuse_unported(configs: dict, dims: dict[str, int]) -> None:
    """Refuse, by name, what the shortwave port does not carry."""
    from woof.hex.mpas_ice import rkind

    c = configs
    cv.refuse_unported(c, dims)
    calc_tsfc = bool(c["config_calc_surface_temperature"])
    dedd = calc_tsfc and c["config_shortwave_type"] == "dEdd"
    if dedd and not rkind.trig64():
        _refuse("config_shortwave_type", "dEdd",
                rkind.trig_refusal("the orbit and the solar zenith angle call glibc's double sin, cos, atan and asin"),
                "WOOF_GLIBC239_LIBM64=1, config_shortwave_type = 'ccsm3' or config_calc_surface_temperature = .false.")
    if c["config_use_aerosols"]:
        _refuse("config_use_aerosols", True, "the aerosols in the shortwave are not ported",
                "config_use_aerosols = .false.")
    if c["config_use_shortwave_bioabsorption"]:
        _refuse("config_use_shortwave_bioabsorption", True,
                "the shortwave bioabsorption (compute_shortwave_trcr) is not ported",
                "config_use_shortwave_bioabsorption = .false.")
    if calc_tsfc and not dedd and dims["nCategories"] > dims["nIceLayers"]:
        _refuse("config_nCategories", dims["nCategories"],
                "shortwave_ccsm3 passes ncat as absorbed_solar's nilyr (ice_shortwave.F90:236), which then "
                "writes past the category's ice layers",
                "config_nCategories <= config_nIceLayers")
    if calc_tsfc and not dedd and c["config_albedo_type"] == "ccsm3" and not rkind.trig64():
        _refuse("config_albedo_type", "ccsm3", rkind.trig_refusal("compute_albedos calls glibc's double atan"),
                "WOOF_GLIBC239_LIBM64=1 or config_albedo_type = 'constant'")


def refuse_state_mismatch(configs: dict, state: "cv.ColumnState") -> None:
    """The kernel takes calc_Tsfc from the state (``init_parameters``) and the
    refusals take it from the configs: refuse a state built from other configs."""
    calc_tsfc = bool(configs["config_calc_surface_temperature"])
    if calc_tsfc != bool(state.calc_tsfc):
        _refuse("config_calc_surface_temperature", calc_tsfc,
                f"the ColumnState was initialised with config_calc_surface_temperature = {bool(state.calc_tsfc)}",
                "the configs the ColumnState was initialised with")


def module_source() -> str:
    from woof.hex.mpas_ice import rkind
    here = files("woof.hex.mpas_seaice")
    body = "\n".join(here.joinpath(name).read_text(encoding="utf-8")
                     for name in ("column_radiation_v842.cu", "column_dedd_v842.cu"))
    return rkind.r8_trig_prelude() + cv.module_source() + "\n" + body


@lru_cache(maxsize=None)
def _module_on(device: int, options: tuple[str, ...], trig: bool):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(), options, None, "seaice_column_radiation_v842.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _function(name: str):
    import cupy as cp
    from woof.hex.mpas_ice import rkind
    options = _hex_numerics.options(COMPILE_OPTIONS, keep_ftz=True)
    return _module_on(cp.cuda.Device().id, options, rkind.trig64()).get_function(name)


def _flags(configs: dict, state: "cv.ColumnState") -> int:
    bits = (state.ktherm != 0, configs["config_albedo_type"] == "constant", state.calc_tsfc,
            configs["config_use_topo_meltponds"])
    return sum(int(bool(b)) << i for i, b in enumerate(bits))


def _shortwave_fields(pools: dict, d: dict) -> dict:
    n1, ncat, nil, nsl = d["nCells"] + 1, d["nCategories"], d["nIceLayers"], d["nSnowLayers"]
    sw = pools["shortwave"]
    need = cv._need
    out = {k: need(sw, k, (n1, ncat)) for k in (
        "albedoVisibleDirectCategory", "albedoVisibleDiffuseCategory", "albedoIRDirectCategory",
        "albedoIRDiffuseCategory", "surfaceShortwaveFlux", "interiorShortwaveFlux", "penetratingShortwaveFlux",
        "bareIceAlbedoCategory", "snowAlbedoCategory", "pondAlbedoCategory", "effectivePondAreaCategory")}
    out["shortwaveLayerPenetration"] = need(sw, "shortwaveLayerPenetration", (n1, ncat, nil + 1))
    out["absorbedShortwaveSnowLayer"] = need(sw, "absorbedShortwaveSnowLayer", (n1, ncat, nsl))
    out["absorbedShortwaveIceLayer"] = need(sw, "absorbedShortwaveIceLayer", (n1, ncat, nil))
    return out


def _down(pools: dict, d: dict) -> list:
    c1 = (d["nCells"] + 1,)
    return [cv._need(pools["atmos_coupling"], k, c1) for k in (
        "shortwaveVisibleDirectDown", "shortwaveVisibleDiffuseDown", "shortwaveIRDirectDown",
        "shortwaveIRDiffuseDown")]


def seconds_into_day(start_time: str) -> int:
    """``get_seconds_into_day`` at the clock's start."""
    return int(start_time[11:13]) * 3600 + int(start_time[14:16]) * 60 + int(start_time[17:19])


def _dedd_parameters(configs: dict, day: float) -> np.ndarray:
    c = configs
    par = np.zeros(12, np.float64)
    par[:] = (c["config_ice_shortwave_tuning_parameter"], c["config_pond_shortwave_tuning_parameter"],
              c["config_snow_shortwave_tuning_parameter"], c["config_temp_change_snow_grain_radius_change"],
              c["config_max_melting_snow_grain_radius"], c["config_algae_absorption_coefficient"],
              c["config_snow_to_ice_transition_depth"], c["config_snow_on_pond_ice_tapering_parameter"],
              c["config_critical_pond_ice_thickness"], c["config_pond_depth_to_fraction_ratio"],
              c["config_dt"], day)
    return par


def orbit() -> "object":
    """``colpkg_init_orbit``'s ``shr_orb_params`` for 1950 on the device:
    eccen, obliqr, lambm0, mvelpp, obliq and mvelp."""
    import cupy as cp
    out = cp.zeros(6, np.float64)
    _function("seaice_column_init_orbit_v842")((1,), (1,), (out,))
    return out


def _radiation_dedd(pools: dict, configs: dict, state: "cv.ColumnState", initialization: bool,
                    day: float | None, sec: int | None) -> None:
    """``column_radiation``'s ``run_dEdd``, with ``colpkg_init_orbit``'s orbit."""
    import cupy as cp
    from . import column_therm_v842 as ct

    c = configs
    d = state.dims
    n1, n_solve, ncat, nil, nsl = d["nCells"] + 1, d["nCellsSolve"], d["nCategories"], d["nIceLayers"], d["nSnowLayers"]
    cat1, cat = (n1, ncat, 1), (n1, ncat)
    day = ct.day_of_year(c["config_start_time"]) if day is None else float(day)
    sec = seconds_into_day(c["config_start_time"]) if sec is None else int(sec)
    t = pools["tracers"]
    area, volume, snow, tsfc = (cv._need(t, k, cat1) for k in ("iceAreaCategory", "iceVolumeCategory",
                                                               "snowVolumeCategory", "surfaceTemperature"))
    cesm, level, topo = (bool(c[k]) for k in ("config_use_cesm_meltponds", "config_use_level_meltponds",
                                              "config_use_topo_meltponds"))
    dummy = cp.zeros(1, np.float64)
    alvl = cv._need(t, "levelIceArea", cat1) if level else dummy
    apnd = cv._need(t, "pondArea", cat1) if cesm or level or topo else dummy
    hpnd = cv._need(t, "pondDepth", cat1) if cesm or level or topo else dummy
    ipnd = cv._need(t, "pondLidThickness", cat1) if level or topo else dummy
    ponds = pools.get("ponds", {})
    dhs = cv._need(ponds, "pondSnowDepthDifference", cat) if level else dummy
    ffrac = cv._need(ponds, "pondLidMeltFluxFraction", cat) if level else dummy
    fsnow = cv._need(pools["atmos_coupling"], "snowfallRate", (n1,)) if level else dummy
    mesh = pools["mesh"]
    lat, lon = (cv._need(mesh, k, (n1,)) for k in ("latCell", "lonCell"))
    f = _shortwave_fields(pools, d)
    sw = pools["shortwave"]
    coszen = cv._need(sw, "solarZenithAngleCosine", (n1,))
    snowfrac = cv._need(sw, "snowFractionCategory", cat)
    bio, nbio = _bio(pools, n1, ncat, dummy)
    flags = (int(cesm) | int(level) << 1 | int(topo) << 2 | int(bool(initialization)) << 3
             | int(state.ktherm != 0) << 4)
    par = cp.asarray(_dedd_parameters(c, day))
    orb = orbit()
    args = (np.int32(ncat), np.int32(nil), np.int32(nsl), np.int32(nbio), np.int32(flags), np.int32(sec),
            np.int32(n_solve), par, orb, lat, lon, area, volume, snow, tsfc, alvl, apnd, hpnd, ipnd, ffrac,
            fsnow, *_down(pools, d), coszen,
            f["albedoVisibleDirectCategory"], f["albedoVisibleDiffuseCategory"], f["albedoIRDirectCategory"],
            f["albedoIRDiffuseCategory"], f["surfaceShortwaveFlux"], f["interiorShortwaveFlux"],
            f["penetratingShortwaveFlux"], f["shortwaveLayerPenetration"], f["absorbedShortwaveIceLayer"],
            f["absorbedShortwaveSnowLayer"], f["bareIceAlbedoCategory"], f["snowAlbedoCategory"],
            f["pondAlbedoCategory"], f["effectivePondAreaCategory"], snowfrac, dhs, bio)
    if n_solve:
        _function("seaice_column_radiation_dedd_v842")(_grid(n_solve), (_BLOCK,), args)


def _bio(pools: dict, n1: int, ncat: int, dummy) -> tuple:
    import cupy as cp
    bio = pools["biogeochemistry"].get("bioTracerShortwave")
    if bio is None:
        return dummy, 0
    if not isinstance(bio, cp.ndarray) or bio.dtype != np.float64 or bio.ndim != 3 \
            or bio.shape[:2] != (n1, ncat) or not bio.flags.c_contiguous:
        raise ValueError("bioTracerShortwave must be a contiguous device float64 (nCells+1, nCategories, n)")
    return bio, int(bio.shape[2])


def radiation(pools: dict, configs: dict, state: "cv.ColumnState", initialization: bool = False,
              day: float | None = None, sec: int | None = None) -> None:
    """``column_radiation(domain, clock, lInitialization)``: the
    delta-Eddington or CCSM3 shortwave, or none with a coupled surface
    temperature. ``day`` and ``sec`` default to ``config_start_time``'s day of
    the year and seconds into the day."""
    import cupy as cp

    d = state.dims
    refuse_state_mismatch(configs, state)
    refuse_unported(configs, d)
    if configs["config_calc_surface_temperature"] and configs["config_shortwave_type"] == "dEdd":
        _radiation_dedd(pools, configs, state, initialization, day, sec)
        return
    n1, n_solve, ncat, nil, nsl = d["nCells"] + 1, d["nCellsSolve"], d["nCategories"], d["nIceLayers"], d["nSnowLayers"]
    cat1 = (n1, ncat, 1)
    t = pools["tracers"]
    area, volume, snow, tsfc = (cv._need(t, k, cat1) for k in ("iceAreaCategory", "iceVolumeCategory",
                                                               "snowVolumeCategory", "surfaceTemperature"))
    topo = bool(configs["config_use_topo_meltponds"])
    dummy = cp.zeros(1, np.float64)
    apnd = cv._need(t, "pondArea", cat1) if topo else dummy
    ipnd = cv._need(t, "pondLidThickness", cat1) if topo else dummy
    f = _shortwave_fields(pools, d)
    coszen = cv._need(pools["shortwave"], "solarZenithAngleCosine", (n1,))
    bio, nbio = _bio(pools, n1, ncat, dummy)
    c = configs
    args = (np.int32(ncat), np.int32(nil), np.int32(nsl), np.int32(nbio), np.int32(_flags(c, state)),
            np.float64(c["config_visible_ice_albedo"]), np.float64(c["config_infrared_ice_albedo"]),
            np.float64(c["config_visible_snow_albedo"]), np.float64(c["config_infrared_snow_albedo"]),
            np.float64(c["config_variable_albedo_thickness_limit"]),
            np.float64(c["config_critical_pond_ice_thickness"]), np.int32(n_solve),
            area, volume, snow, tsfc, apnd, ipnd, *_down(pools, d), coszen,
            f["albedoVisibleDirectCategory"], f["albedoVisibleDiffuseCategory"], f["albedoIRDirectCategory"],
            f["albedoIRDiffuseCategory"], f["surfaceShortwaveFlux"], f["interiorShortwaveFlux"],
            f["penetratingShortwaveFlux"], f["shortwaveLayerPenetration"], f["absorbedShortwaveIceLayer"],
            f["absorbedShortwaveSnowLayer"], f["bareIceAlbedoCategory"], f["snowAlbedoCategory"],
            f["effectivePondAreaCategory"], bio)
    if n_solve:
        _function("seaice_column_radiation_v842")(_grid(n_solve), (_BLOCK,), args)


def prep_radiation(pools: dict, configs: dict, state: "cv.ColumnState") -> None:
    """``column_prep_radiation``: the shortwave fluxes scaled by the ratio of
    the new to the old net shortwave."""
    d = state.dims
    cv.refuse_unported(configs, d)
    n1, n_solve, ncat, nil, nsl = d["nCells"] + 1, d["nCellsSolve"], d["nCategories"], d["nIceLayers"], d["nSnowLayers"]
    sw = pools["shortwave"]
    c1 = (n1,)
    f = _shortwave_fields(pools, d)
    args = (np.int32(ncat), np.int32(nil), np.int32(nsl), np.int32(n_solve),
            cv._need(pools["tracers_aggregate"], "iceAreaCell", c1),
            cv._need(pools["tracers"], "iceAreaCategory", (n1, ncat, 1)), *_down(pools, d),
            *(cv._need(sw, k, c1) for k in ("albedoVisibleDirectArea", "albedoVisibleDiffuseArea",
                                            "albedoIRDirectArea", "albedoIRDiffuseArea", "shortwaveScalingFactor")),
            f["surfaceShortwaveFlux"], f["interiorShortwaveFlux"], f["penetratingShortwaveFlux"],
            f["shortwaveLayerPenetration"], f["absorbedShortwaveSnowLayer"], f["absorbedShortwaveIceLayer"])
    if n_solve:
        _function("seaice_column_prep_radiation_v842")(_grid(n_solve), (_BLOCK,), args)


def init_shortwave(pools: dict, configs: dict, state: "cv.ColumnState", day: float | None = None,
                   sec: int | None = None) -> None:
    """``seaice_init_column_shortwave``: the orbit for the delta-Eddington
    shortwave, ``column_radiation`` at initialisation, then the cell albedos,
    the area albedos and the shortwave scaling factor's net shortwave."""
    d = state.dims
    refuse_unported(configs, d)
    radiation(pools, configs, state, initialization=True, day=day, sec=sec)
    n1, n_solve, ncat = d["nCells"] + 1, d["nCellsSolve"], d["nCategories"]
    sw = pools["shortwave"]
    c1 = (n1,)
    f = _shortwave_fields(pools, d)
    cell = [cv._need(sw, k, c1) for k in (
        "albedoVisibleDirectCell", "albedoVisibleDiffuseCell", "albedoIRDirectCell", "albedoIRDiffuseCell",
        "bareIceAlbedoCell", "snowAlbedoCell", "pondAlbedoCell", "effectivePondAreaCell",
        "albedoVisibleDirectArea", "albedoVisibleDiffuseArea", "albedoIRDirectArea", "albedoIRDiffuseArea",
        "shortwaveScalingFactor")]
    args = (np.int32(ncat), np.int32(n_solve), cv._need(pools["tracers"], "iceAreaCategory", (n1, ncat, 1)),
            cv._need(sw, "solarZenithAngleCosine", c1),
            f["albedoVisibleDirectCategory"], f["albedoVisibleDiffuseCategory"], f["albedoIRDirectCategory"],
            f["albedoIRDiffuseCategory"], f["bareIceAlbedoCategory"], f["snowAlbedoCategory"],
            f["pondAlbedoCategory"], f["effectivePondAreaCategory"], *_down(pools, d), *cell)
    if n_solve:
        _function("seaice_init_column_shortwave_aggregate_v842")(_grid(n_solve), (_BLOCK,), args)
