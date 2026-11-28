"""MPAS-Seaice v8.4.2: the column package's coupling preparation on the GPU.

:func:`coupling_prep` is ``seaice_column_coupling_prep``
(``mpas_seaice_column.F``), the last thing
``seaice_column_postdynamics_time_integration`` does each step when the column
package is on. It works per cell over ``nCellsSolve``:

- under ``config_use_ocean_mixed_layer`` (the Registry default) it first
  runs the stand-alone ocean mixed layer
  (:func:`woof.hex.mpas_seaice.ocean_mixed_layer_v842.ocean_mixed_layer`,
  RKIND 8 and the LGPL libm opt-in only);
- it saves the freezing/melting potential;
- it sums the category albedos (and the bare-ice, snow and pond albedos where
  the sun is up) and the effective pond area into the cell values, weighting
  each category by its ice area;
- it feeds the pond freshwater back into the ocean freshwater flux
  (``config_include_pond_freshwater_feedback``);
- it copies the albedos and ocean fluxes, which are per unit grid-cell area,
  into their ``*Area`` fields;
- it forms the shortwave scaling factor;
- under ``config_use_column_biogeochemistry`` it maps the column package's
  ocean biogeochemical fluxes onto the ``ocean*Flux`` fields: with the brine
  height alone there are none (``nBioTracers`` is 0), and the nonreactive
  vertical tracer's has no field (the mapping skips it), so each is zero.

It then runs ``seaice_column_scale_fluxes``. Where there is ice, that divides
the air stress, the atmospheric and ocean fluxes, the reference
temperature/humidity, the cell albedos and (under the biogeochemistry) the
ocean biogeochemical fluxes by the ice area. Where there is none, it zeroes
them, sets the open-water upward longwave at the sea's freezing point, and
takes the reference values from the air.

The kernels (``coupling_prep_v842.cu``) are held word for word to v8.4.2.
Both routines are cut verbatim out of ``mpas_seaice_column.F`` and compiled
with gfortran ``-O0 -ffp-contract=off`` at RKIND 4 and 8
(``tools/mpas_seaice_oracle/run_coupling_prep.F90``,
``tests/test_mpas_seaice_coupling_prep_parity.py``).

Refused by name:

- what :mod:`~woof.hex.mpas_seaice.ocean_mixed_layer_v842` refuses, under
  ``config_use_ocean_mixed_layer``;
- ``config_use_column_biogeochemistry`` with biogeochemical tracers other
  than the nonreactive vertical tracer (whose ocean flux has no field).
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import numpy as np

from woof.hex.errors import ConfigurationRefusal

from .velocity_v842 import COMPILE_OPTIONS, PRECISIONS, _precision_of, launch_options

SOURCE_FILES = (
    "src/core_seaice/shared/mpas_seaice_column.F:3634-4041",
    "src/core_seaice/shared/mpas_seaice_column.F:4055-4306",
    "src/core_seaice/shared/mpas_seaice_constants.F",
    "src/core_seaice/column/constants/cice/ice_constants_colpkg.F90",
)
MODULE_OPTIONS = COMPILE_OPTIONS
_BLOCK = 128
#: seaice_constants: colpkg dbl_kind parameters (puny, rhofresh, stefan_boltzmann, Tffresh)
PUNY = 1.0e-11
DENSITY_FRESHWATER = 1000.0
STEFAN_BOLTZMANN = 567.0e-10
FRESH_WATER_FREEZING_POINT = 273.15
#: Registry.xml defaults (v8.4.2) of the configs the routines read
DEFAULT_CONFIGS = {"config_use_ocean_mixed_layer": True, "config_include_pond_freshwater_feedback": False,
                   "config_use_column_biogeochemistry": False}
#: the fields each pool must hold
FIELDS = {
    "ocean_coupling": ("freezingMeltingPotential", "seaFreezingTemperature"),
    "diagnostics": ("freezingMeltingPotentialInitial",),
    "tracers": ("iceAreaCategory",),
    "tracers_aggregate": ("iceAreaCell",),
    "shortwave": ("albedoVisibleDirectCell", "albedoVisibleDiffuseCell", "albedoIRDirectCell", "albedoIRDiffuseCell",
                  "albedoVisibleDirectCategory", "albedoVisibleDiffuseCategory", "albedoIRDirectCategory",
                  "albedoIRDiffuseCategory", "albedoVisibleDirectArea", "albedoVisibleDiffuseArea",
                  "albedoIRDirectArea", "albedoIRDiffuseArea", "solarZenithAngleCosine", "bareIceAlbedoCell",
                  "snowAlbedoCell", "pondAlbedoCell", "bareIceAlbedoCategory", "snowAlbedoCategory",
                  "pondAlbedoCategory", "effectivePondAreaCell", "effectivePondAreaCategory",
                  "shortwaveScalingFactor", "absorbedShortwaveFlux"),
    "atmos_coupling": ("shortwaveVisibleDirectDown", "shortwaveVisibleDiffuseDown", "shortwaveIRDirectDown",
                       "shortwaveIRDiffuseDown", "atmosReferenceHumidity2m", "atmosReferenceTemperature2m",
                       "airTemperature", "airSpecificHumidity"),
    "ponds": ("pondFreshWaterFlux",),
    "ocean_fluxes": ("oceanFreshWaterFlux", "oceanSaltFlux", "oceanHeatFlux", "oceanShortwaveFlux",
                     "oceanFreshWaterFluxArea", "oceanSaltFluxArea", "oceanHeatFluxArea", "oceanShortwaveFluxArea"),
    "velocity_solver": ("airStressCellU", "airStressCellV"),
    "atmos_fluxes": ("sensibleHeatFlux", "latentHeatFlux", "evaporativeWaterFlux", "longwaveUp"),
}


def module_source(precision: str = "r4") -> str:
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(PRECISIONS)}")
    body = files("woof.hex.mpas_seaice").joinpath("coupling_prep_v842.cu").read_text(encoding="utf-8")
    return ("#define MPAS_RKIND8 1\n" if precision == "r8" else "") + body


@lru_cache(maxsize=None)
def _module_on(device: int, precision: str, options: tuple[str, ...]):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(precision), options, None,
                                   f"seaice_coupling_prep_v842_{precision}.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _launch(precision: str, name: str, n: int, args: tuple) -> None:
    import cupy as cp
    if n > 0:
        _module_on(cp.cuda.Device().id, precision, launch_options(COMPILE_OPTIONS)).get_function(name)(
            ((n + _BLOCK - 1) // _BLOCK,), (_BLOCK,), args)


def _config(configs: dict, name: str):
    return configs.get(name, DEFAULT_CONFIGS[name])


def check_config(configs: dict) -> None:
    """Refuse, by name, what this port does not carry."""
    if _config(configs, "config_use_ocean_mixed_layer"):
        from . import ocean_mixed_layer_v842 as oml
        oml.check_config(configs)
    if _config(configs, "config_use_column_biogeochemistry"):
        from . import column_v842 as cv
        # an option the caller leaves out is its Registry default, as in MPAS; each is then read strictly
        c = {**cv.default_configs(), **configs}
        if not c["config_use_column_package"]:
            # v8.4.2 checks the biogeochemistry's options and calls seaice_column_coupling_prep only from
            # the column package (seaice_column_postdynamics_time_integration)
            raise ConfigurationRefusal(
                knob="config_use_column_package", value=False,
                reason="the column biogeochemistry runs only with the column package on",
                declaration="config_use_column_package = .true., or config_use_column_biogeochemistry = .false.")
        # the vertical salinity is no biogeochemical tracer: colpkg_init_zbgc leaves it out of
        # nbtrcr (nBioTracers), so its fluxes are not among oceanBioFluxes. The nonreactive
        # tracer is (bio_index_o 17), and coupling_prep's mapping skips that entry (PON has no
        # ocean*Flux field, mpas_seaice_column.F:3998-3999): every ocean*Flux stays zero
        if cv.vertical_tracers(c):
            cv._refuse_z_tracers(c)
        cv.refuse_uncarried_bgc(c, cv.carried_bgc_flags(c),
                                "the column biogeochemistry's ocean fluxes are ported for the brine height, "
                                "the vertical salinity and the nonreactive vertical tracer only")


#: the ocean biogeochemical fluxes seaice_column_coupling_prep maps from the column
#: package's (mpas_seaice_column.F:3928-4028) and seaice_column_scale_fluxes scales
BGC_OCEAN_FLUXES = ("oceanNitrateFlux", "oceanSilicateFlux", "oceanAmmoniumFlux", "oceanDMSPpFlux",
                    "oceanDMSPdFlux", "oceanDMSFlux", "oceanHumicsFlux", "oceanDustIronFlux", "oceanAlgaeFlux",
                    "oceanDOCFlux", "oceanDICFlux", "oceanDONFlux", "oceanParticulateIronFlux",
                    "oceanDissolvedIronFlux")


def _bgc_ocean_fluxes(bgc: dict, ice_area_cell, n_cells_solve: int, n_aggregate: int) -> None:
    """The biogeochemistry's part of both routines without biogeochemical
    tracers: coupling_prep sets each flux of the owned cells to zero (the
    dust's to a sum of zeros), and scale_fluxes multiplies them by the
    inverse ice area where there is ice and zeroes them where there is none,
    so a cell of ice with an area whose inverse overflows gets 0 * inf."""
    import cupy as cp
    area = ice_area_cell[:n_aggregate]
    for name in BGC_OCEAN_FLUXES:
        flux = bgc[name]
        flux[:n_cells_solve] = 0
        rows = flux[:n_aggregate]
        ice = area > 0 if rows.ndim == 1 else (area > 0)[:, None]
        inverse = 1.0 / cp.where(area > 0, area, 1.0)
        scaled = rows * (inverse if rows.ndim == 1 else inverse[:, None])
        rows[...] = cp.where(ice, scaled, cp.zeros_like(rows))


def coupling_prep(*, pools: dict, n_cells_solve: int, n_cells_solve_aggregate: int | None = None,
                  dt: float, configs: dict) -> None:
    """``seaice_column_coupling_prep`` then ``seaice_column_scale_fluxes``, in place on ``pools``.

    ``pools`` maps the Registry struct names in :data:`FIELDS` to their device
    arrays (``iceAreaCategory`` at time level 1). ``n_cells_solve`` is the
    tracers pool's ``nCellsSolve`` and ``n_cells_solve_aggregate`` is
    tracers_aggregate's (the same by default). ``dt`` is ``config_dt``.
    """
    check_config(configs)
    if _config(configs, "config_use_ocean_mixed_layer"):
        from . import ocean_mixed_layer_v842 as oml
        oml.ocean_mixed_layer(pools, {**configs, "config_dt": dt}, n_cells_solve)
    sw, oc, of, ac = pools["shortwave"], pools["ocean_coupling"], pools["ocean_fluxes"], pools["atmos_coupling"]
    area_category = pools["tracers"]["iceAreaCategory"]
    precision = _precision_of(area_category)
    real = PRECISIONS[precision]
    i32 = np.int32
    n_categories = int(area_category.shape[1])
    feedback = bool(_config(configs, "config_include_pond_freshwater_feedback"))
    _launch(precision, "seaice_coupling_prep_v842", n_cells_solve, (
        pools["diagnostics"]["freezingMeltingPotentialInitial"], oc["freezingMeltingPotential"], area_category,
        sw["albedoVisibleDirectCell"], sw["albedoVisibleDiffuseCell"], sw["albedoIRDirectCell"],
        sw["albedoIRDiffuseCell"], sw["albedoVisibleDirectCategory"], sw["albedoVisibleDiffuseCategory"],
        sw["albedoIRDirectCategory"], sw["albedoIRDiffuseCategory"], sw["albedoVisibleDirectArea"],
        sw["albedoVisibleDiffuseArea"], sw["albedoIRDirectArea"], sw["albedoIRDiffuseArea"],
        sw["solarZenithAngleCosine"], sw["bareIceAlbedoCell"], sw["snowAlbedoCell"], sw["pondAlbedoCell"],
        sw["bareIceAlbedoCategory"], sw["snowAlbedoCategory"], sw["pondAlbedoCategory"],
        sw["effectivePondAreaCell"], sw["effectivePondAreaCategory"], pools["ponds"]["pondFreshWaterFlux"],
        of["oceanFreshWaterFlux"], of["oceanSaltFlux"], of["oceanHeatFlux"], of["oceanShortwaveFlux"],
        of["oceanFreshWaterFluxArea"], of["oceanSaltFluxArea"], of["oceanHeatFluxArea"],
        of["oceanShortwaveFluxArea"], sw["shortwaveScalingFactor"], ac["shortwaveVisibleDirectDown"],
        ac["shortwaveVisibleDiffuseDown"], ac["shortwaveIRDirectDown"], ac["shortwaveIRDiffuseDown"],
        i32(n_cells_solve), i32(n_categories), i32(feedback), real(PUNY), real(DENSITY_FRESHWATER), real(dt)))
    vs, af = pools["velocity_solver"], pools["atmos_fluxes"]
    n_aggregate = n_cells_solve if n_cells_solve_aggregate is None else n_cells_solve_aggregate
    _launch(precision, "seaice_scale_fluxes_v842", n_aggregate, (
        pools["tracers_aggregate"]["iceAreaCell"], vs["airStressCellU"], vs["airStressCellV"],
        af["sensibleHeatFlux"], af["latentHeatFlux"], sw["absorbedShortwaveFlux"], af["longwaveUp"],
        af["evaporativeWaterFlux"], ac["atmosReferenceTemperature2m"], ac["atmosReferenceHumidity2m"],
        of["oceanFreshWaterFlux"], of["oceanSaltFlux"], of["oceanHeatFlux"], of["oceanShortwaveFlux"],
        sw["albedoVisibleDirectCell"], sw["albedoIRDirectCell"], sw["albedoVisibleDiffuseCell"],
        sw["albedoIRDiffuseCell"], ac["airTemperature"], ac["airSpecificHumidity"], oc["seaFreezingTemperature"],
        i32(n_aggregate), real(STEFAN_BOLTZMANN), real(FRESH_WATER_FREEZING_POINT)))
    if _config(configs, "config_use_column_biogeochemistry"):
        _bgc_ocean_fluxes(pools["biogeochemistry"], pools["tracers_aggregate"]["iceAreaCell"], n_cells_solve,
                          n_aggregate)
