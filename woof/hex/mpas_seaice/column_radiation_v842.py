"""MPAS-Seaice v8.4.2's column shortwave radiation on the GPU, without the
delta-Eddington solution.

``column_prep_radiation`` (``mpas_seaice_column.F:1990-2104``: the shortwave
scaling, ``colpkg_prep_radiation``), ``column_radiation``
(``mpas_seaice_column.F:2118-2484``: ``colpkg_step_radiation``) and
``seaice_init_column_shortwave``'s albedo aggregation
(``mpas_seaice_column.F:435-647``), word for word with the gfortran -O0
-ffp-contract=off oracle at RKIND 8. One thread is one owned cell;
``column_radiation_v842.cu`` is the transcription.

What it carries: the CCSM3 shortwave (``config_shortwave_type = 'ccsm3'``:
``shortwave_ccsm3`` with the ``ccsm3`` or ``constant`` albedos and
``absorbed_solar``), and any shortwave type with a coupled surface
temperature (``config_calc_surface_temperature = .false.``: no shortwave,
the topo ponds' effective area), on BL99, mushy or zero-layer ice.

What it refuses, by name:

- the delta-Eddington shortwave with a computed surface temperature (the
  Registry default; it needs the orbit and is not ported yet);
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
resetting them.

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
    "src/core_seaice/column/ice_colpkg.F90:2547-2972",
    "src/core_seaice/column/ice_shortwave.F90:72-660",
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
    if calc_tsfc and c["config_shortwave_type"] == "dEdd":
        _refuse("config_shortwave_type", "dEdd",
                "the delta-Eddington shortwave (run_dEdd, with the orbit) is not ported",
                "config_shortwave_type = 'ccsm3' or config_calc_surface_temperature = .false.")
    if c["config_use_aerosols"]:
        _refuse("config_use_aerosols", True, "the aerosols in the shortwave are not ported",
                "config_use_aerosols = .false.")
    if c["config_use_shortwave_bioabsorption"]:
        _refuse("config_use_shortwave_bioabsorption", True,
                "the shortwave bioabsorption (compute_shortwave_trcr) is not ported",
                "config_use_shortwave_bioabsorption = .false.")
    if calc_tsfc and dims["nCategories"] > dims["nIceLayers"]:
        _refuse("config_nCategories", dims["nCategories"],
                "shortwave_ccsm3 passes ncat as absorbed_solar's nilyr (ice_shortwave.F90:236), which then "
                "writes past the category's ice layers",
                "config_nCategories <= config_nIceLayers")
    if calc_tsfc and c["config_albedo_type"] == "ccsm3" and not rkind.trig64():
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
    body = files("woof.hex.mpas_seaice").joinpath("column_radiation_v842.cu").read_text(encoding="utf-8")
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


def radiation(pools: dict, configs: dict, state: "cv.ColumnState") -> None:
    """``column_radiation(domain, clock, lInitialization)``: the CCSM3
    shortwave, or none with a coupled surface temperature."""
    import cupy as cp

    d = state.dims
    refuse_state_mismatch(configs, state)
    refuse_unported(configs, d)
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
    bio = pools["biogeochemistry"].get("bioTracerShortwave")
    if bio is not None:
        if not isinstance(bio, cp.ndarray) or bio.dtype != np.float64 or bio.ndim != 3 \
                or bio.shape[:2] != (n1, ncat) or not bio.flags.c_contiguous:
            raise ValueError("bioTracerShortwave must be a contiguous device float64 (nCells+1, nCategories, n)")
        nbio = int(bio.shape[2])
    else:
        bio, nbio = dummy, 0
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


def init_shortwave(pools: dict, configs: dict, state: "cv.ColumnState") -> None:
    """``seaice_init_column_shortwave``: ``column_radiation`` at
    initialisation, then the cell albedos, the area albedos and the
    shortwave scaling factor's net shortwave."""
    d = state.dims
    refuse_unported(configs, d)
    radiation(pools, configs, state)
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
