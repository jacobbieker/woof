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
ice. With the column biogeochemistry's z-aerosols
(``config_use_zaerosols``) the delta-Eddington shortwave carries their
bioabsorption (``config_use_shortwave_bioabsorption``, dEdd_algae):
``compute_shortwave_trcr`` remaps each aerosol's ice layers onto the
shortwave grid into ``bioTracerShortwave`` (``remap_zbgc``, below its two
snow levels), and ``compute_dEdd`` folds their extinction, single scattering
and asymmetry into the snow and the bare and snow-covered ice's layers, from
the aerosols pool's tables: the bulk ones, or with
``config_use_modal_aerosols`` the modal black carbon's (interstitial, and
within the ice with its absorption enhanced by the snow grain size). Without
the z-aerosols the bioabsorption changes nothing (the algae are refused).

What it refuses, by name:

- the delta-Eddington shortwave in the default build: the orbit and the
  solar zenith angle call glibc's binary64 ``sin``, ``cos``, ``atan`` and
  ``asin``, which are not correctly rounded and are LGPL, so they compile
  only with ``WOOF_GLIBC239_LIBM64=1`` (:mod:`woof.core.glibc239_gpu`);
- the NCAR aerosols (``config_use_aerosols``) and, as every column stage, the
  biogeochemistry but its brine height, the vertical salinity, the
  nonreactive tracer and the z-aerosols;
- the bioabsorption without the column biogeochemistry (v8.4.2's
  ``check_column_package_configs`` stops);
- the bioabsorption of the z-aerosols with fewer ice layers than biology
  layers (unless corrected, below): ``column_radiation`` passes
  ``verticalGrid`` (the ice layers' midpoints, ``mpas_seaice_column.F:2415``)
  as ``compute_shortwave_trcr``'s biology interface grid, which then reads
  ``nBioLayers + 1`` points of it, past the array;
- ``shortwave_ccsm3`` with more categories than ice layers: v8.4.2 passes
  ``ncat`` as ``absorbed_solar``'s ``nilyr`` (``ice_shortwave.F90:236``) and
  then writes past the category's layers (unless corrected, below);
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
bare-ice, snow and pond albedos. Ice without volume (``aice > puny``,
``vice = 0``) has layers without thickness, and v8.4.2's
``kalg*(0.5/dzk(klev))`` makes the bottom layer's extinction infinite there,
as the port does: with algae the visible light reaching it is absorbed in ice
that is not there, and with ``kalg = 0`` the category's shortwave is NaN (a
NaN's payload is the host's, so those words are not compared). The day of the year follows
``config_calendar_type`` (``gregorian`` or ``gregorian_noleap``).

Opt-in corrections (``configs["woof_v842_corrections"]``, named in
:data:`column_v842.CORRECTIONS`; the default is v8.4.2's words):
``ccsm3_absorbed_solar_layers`` passes the category's ``nilyr`` to
``absorbed_solar`` (and lifts the ``ncat > nilyr`` refusal), and
``dedd_zero_volume_ice`` gives the delta-Eddington layers without thickness
no chlorophyll absorption, so ice without volume is transparent, and
``shortwave_bio_grid`` passes ``interfaceBiologyGrid`` as
``compute_shortwave_trcr``'s biology grid (and lifts the ice-layer
refusal). v8.4.2's ``verticalGrid`` there is kept by default: with at least
as many ice layers as biology layers it remaps the aerosols from the ice
layers' midpoints' depths. A negative brine height stops ``remap_zbgc``,
and ``compute_shortwave_trcr`` returns with the later aerosols' levels at 0;
``colpkg_step_radiation`` never resets its uninitialised ``l_stop`` (false on
the oracle's entry to each cell), so the cell's later categories return at
their first remap and keep all their levels at 0, as the port does.

``exp`` is ``glibc_exp`` (the MIT transcription of glibc's binary64 exp).
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import numpy as np

from woof.hex import numerics as _hex_numerics
from woof.hex.errors import ConfigurationRefusal

from . import column_therm_v842 as ct
from . import column_v842 as cv
from .velocity_v842 import COMPILE_OPTIONS, _grid

SOURCE_FILES = (
    "src/core_seaice/shared/mpas_seaice_column.F:435-647,1990-2484",
    "src/core_seaice/column/ice_colpkg.F90:387-412,2547-2972",
    "src/core_seaice/column/ice_orbital.F90:35-680",
    "src/core_seaice/column/ice_shortwave.F90:72-660,673-3563,3571-3729",
    "src/core_seaice/column/ice_zbgc_shared.F90:167-333",
)
_BLOCK = 128


def _refuse(knob: str, value, reason: str, declaration: str):
    raise ConfigurationRefusal(knob=knob, value=value, reason=reason, declaration=declaration)


def refuse_unported(configs: dict, dims: dict[str, int]) -> None:
    """Refuse, by name, what the shortwave port does not carry."""
    from woof.hex.mpas_ice import rkind

    c = configs
    cv.refuse_unported(c, dims, biogeochemistry=True, z_tracers=True)
    calc_tsfc = bool(c["config_calc_surface_temperature"])
    dedd = calc_tsfc and c["config_shortwave_type"] == "dEdd"
    if dedd and not rkind.trig64():
        _refuse("config_shortwave_type", "dEdd",
                rkind.trig_refusal("the orbit and the solar zenith angle call glibc's double sin, cos, atan and asin"),
                "WOOF_GLIBC239_LIBM64=1, config_shortwave_type = 'ccsm3' or config_calc_surface_temperature = .false.")
    # a named correction whose shortwave this configuration does not run is
    # refused; under woof_quirk_corrections = 'all' alone it changes nothing
    fixes = cv.named_corrections(c)
    if "ccsm3_absorbed_solar_layers" in fixes and not (calc_tsfc and not dedd):
        _refuse("woof_v842_corrections", "ccsm3_absorbed_solar_layers",
                "it corrects shortwave_ccsm3, which this configuration does not run",
                "config_shortwave_type = 'ccsm3' with config_calc_surface_temperature = .true., or without it")
    if "dedd_zero_volume_ice" in fixes and not dedd:
        _refuse("woof_v842_corrections", "dedd_zero_volume_ice",
                "it corrects the delta-Eddington shortwave, which this configuration does not run",
                "config_shortwave_type = 'dEdd' with config_calc_surface_temperature = .true., or without it")
    if c["config_use_aerosols"]:
        _refuse("config_use_aerosols", True, "the aerosols in the shortwave are not ported",
                "config_use_aerosols = .false.")
    if c["config_use_shortwave_bioabsorption"] and not c["config_use_column_biogeochemistry"]:
        _refuse("config_use_shortwave_bioabsorption", True,
                "check_column_package_configs stops v8.4.2: the shortwave bioabsorption needs the column "
                "biogeochemistry (mpas_seaice_column.F:8159-8167)",
                "config_use_shortwave_bioabsorption = .false. or config_use_column_biogeochemistry = .true.")
    fix_grid = "shortwave_bio_grid" in fixes
    if fix_grid and not (dedd and shortwave_zaerosols(c)):
        _refuse("woof_v842_corrections", "shortwave_bio_grid",
                "it corrects compute_shortwave_trcr's grid, which runs only with the delta-Eddington shortwave's "
                "bioabsorption of the z-aerosols",
                "config_use_shortwave_bioabsorption and config_use_zaerosols with config_shortwave_type = 'dEdd'")
    # the umbrella turns the correction on too, so the read past the array is not reached
    if dedd and shortwave_zaerosols(c) and "shortwave_bio_grid" not in cv.corrections(c) and \
            dims["nIceLayers"] < dims.get("nBioLayers", 7):
        _refuse("config_nIceLayers", dims["nIceLayers"],
                "column_radiation passes verticalGrid (nIceLayers + 1 points) as compute_shortwave_trcr's "
                "biology interface grid, which reads nBioLayers + 1 of them: past the array with fewer ice "
                "layers than biology layers",
                "config_nIceLayers >= nBioLayers, or the opt-in correction "
                "woof_v842_corrections = ('shortwave_bio_grid',)")
    if calc_tsfc and not dedd and dims["nCategories"] > dims["nIceLayers"] and \
            "ccsm3_absorbed_solar_layers" not in cv.corrections(c):
        _refuse("config_nCategories", dims["nCategories"],
                "shortwave_ccsm3 passes ncat as absorbed_solar's nilyr (ice_shortwave.F90:236), which then "
                "writes past the category's ice layers",
                "config_nCategories <= config_nIceLayers, or the opt-in correction "
                "woof_v842_corrections = ('ccsm3_absorbed_solar_layers',)")
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
    from .column_bgc_v842 import zbgc_shared_source
    here = files("woof.hex.mpas_seaice")
    body = "\n".join(here.joinpath(name).read_text(encoding="utf-8")
                     for name in ("column_radiation_v842.cu", "column_dedd_v842.cu"))
    # remap_zbgc (compute_shortwave_trcr's) sized for the most ice layers
    return (rkind.r8_trig_prelude() + f"#define COL_NILYR {cv.MAX_ICE_LAYERS}\n" + cv.module_source() + "\n"
            + zbgc_shared_source() + "\n" + body)


@lru_cache(maxsize=None)
def _module_on(device: int, options: tuple[str, ...], trig: bool):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(), options, None, "seaice_column_radiation_v842.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _module():
    import cupy as cp
    from woof.hex.mpas_ice import rkind
    options = _hex_numerics.options(COMPILE_OPTIONS, keep_ftz=True)
    return _module_on(cp.cuda.Device().id, options, rkind.trig64())


def _function(name: str):
    return _module().get_function(name)


def _flags(configs: dict, state: "cv.ColumnState") -> int:
    bits = (state.ktherm != 0, configs["config_albedo_type"] == "constant", state.calc_tsfc,
            configs["config_use_topo_meltponds"], "ccsm3_absorbed_solar_layers" in cv.corrections(configs))
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


def _orbit() -> "object":
    """``colpkg_init_orbit``'s ``shr_orb_params`` for 1950 on the device:
    eccen, obliqr, lambm0, mvelpp, obliq and mvelp."""
    from woof.hex.mpas_ice import rkind
    if not rkind.trig64():
        _refuse("config_shortwave_type", "dEdd", rkind.trig_refusal("shr_orb_params calls glibc's double sin, cos "
                                                                   "and atan"), "WOOF_GLIBC239_LIBM64=1")
    return _orbit_on(_module())


@lru_cache(maxsize=8)
def _orbit_on(module) -> "object":
    """The orbit, launched once per compiled module (a new module, a new launch)."""
    import cupy as cp
    out = cp.zeros(6, np.float64)
    module.get_function("seaice_column_init_orbit_v842")((1,), (1,), (out,))
    return out


def _radiation_dedd(pools: dict, configs: dict, state: "cv.ColumnState", initialization: bool,
                    day: float | None, sec: int | None) -> None:
    """``column_radiation``'s ``run_dEdd``, with ``colpkg_init_orbit``'s orbit."""
    import cupy as cp

    c = configs
    d = state.dims
    n1, n_solve, ncat, nil, nsl = d["nCells"] + 1, d["nCellsSolve"], d["nCategories"], d["nIceLayers"], d["nSnowLayers"]
    cat1, cat = (n1, ncat, 1), (n1, ncat)
    day = ct.day_of_year(c["config_start_time"], c["config_calendar_type"]) if day is None else float(day)
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
    za = _zaero_optics(pools, c, d, dummy)
    flags = (int(cesm) | int(level) << 1 | int(topo) << 2 | int(bool(initialization)) << 3
             | int(state.ktherm != 0) << 4 | int("dedd_zero_volume_ice" in cv.corrections(c)) << 5
             | int(bool(c.get("config_use_modal_aerosols", False))) << 6)
    par = cp.asarray(_dedd_parameters(c, day))
    orb = _orbit()
    args = (np.int32(ncat), np.int32(nil), np.int32(nsl), np.int32(nbio), np.int32(flags), np.int32(sec),
            np.int32(n_solve), par, orb, lat, lon, area, volume, snow, tsfc, alvl, apnd, hpnd, ipnd, ffrac,
            fsnow, *_down(pools, d), coszen,
            f["albedoVisibleDirectCategory"], f["albedoVisibleDiffuseCategory"], f["albedoIRDirectCategory"],
            f["albedoIRDiffuseCategory"], f["surfaceShortwaveFlux"], f["interiorShortwaveFlux"],
            f["penetratingShortwaveFlux"], f["shortwaveLayerPenetration"], f["absorbedShortwaveIceLayer"],
            f["absorbedShortwaveSnowLayer"], f["bareIceAlbedoCategory"], f["snowAlbedoCategory"],
            f["pondAlbedoCategory"], f["effectivePondAreaCategory"], snowfrac, dhs, bio, *za)
    if n_solve:
        _function("seaice_column_radiation_dedd_v842")(_grid(n_solve), (_BLOCK,), args)


def shortwave_zaerosols(configs: dict) -> bool:
    """``dEdd_algae`` with ``tr_zaero``: the z-aerosols in the delta-Eddington
    shortwave (``compute_shortwave_trcr`` and ``compute_dEdd``'s aerosol
    optics)."""
    c = configs
    return bool(c["config_use_column_biogeochemistry"] and c.get(cv.VT_FLAG, False)
                and c.get(cv.ZAERO_FLAG, False) and c.get("config_use_shortwave_bioabsorption", False))


def _zaero_optics(pools: dict, configs: dict, d: dict, dummy) -> tuple:
    """The kernel's z-aerosol arguments: nz (0 without them), nBioLayers, the
    aerosols' tracer, the brine fraction, the shortwave and biology grids and
    the aerosols pool's optics tables."""
    if not shortwave_zaerosols(configs):
        return (np.int32(0), np.int32(0)) + (dummy,) * 11
    n1, ncat, nil, nbl, nz = d["nCells"] + 1, d["nCategories"], d["nIceLayers"], d["nBioLayers"], d["nzAerosols"]
    need = cv._need
    t, bgc, aer = pools["tracers"], pools["biogeochemistry"], pools["aerosols"]
    # v8.4.2 passes verticalGrid as compute_shortwave_trcr's biology interface grid
    # (mpas_seaice_column.F:2415); the correction shortwave_bio_grid passes interfaceBiologyGrid
    grid = (need(bgc, "interfaceBiologyGrid", (nbl + 1,)) if "shortwave_bio_grid" in cv.corrections(configs)
            else need(bgc, "verticalGrid", (nil + 1,)))
    tables = [need(aer, k, (6, 3)) for k in ("aerosolMassExtinctionCrossSection", "aerosolSingleScatterAlbedo",
                                             "aerosolAsymmetryParameter")]
    tables += [need(aer, k, (10, 3)) for k in ("modalMassExtinctionCrossSection", "modalSingleScatterAlbedo",
                                               "modalAsymmetryParameter")]
    tables.append(need(aer, "modalBCabsorptionParameter", (8, 10, 3)))
    return (np.int32(nz), np.int32(nbl), need(t, "verticalAerosolsConc", (n1, ncat, nz * (nbl + 3))),
            need(t, "brineFraction", (n1, ncat, 1)), need(bgc, "verticalShortwaveGrid", (nil + 1,)), grid, *tables)


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
    cv.refuse_unported(configs, d, biogeochemistry=True, z_tracers=True)
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
