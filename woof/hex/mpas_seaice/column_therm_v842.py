"""MPAS-Seaice v8.4.2's column vertical thermodynamics on the GPU.

``column_vertical_thermodynamics`` (``mpas_seaice_column.F:1064-1666``) and
the column package under it (``colpkg_step_therm1``: the bottom and lateral
melt, the atmospheric boundary layer, the Bitz and Lipscomb (1999)
temperature solver, the thickness changes and the flux merge), word for word
with the gfortran -O0 -ffp-contract=off oracle at RKIND 8. One thread is one
owned cell; ``column_therm_v842.cu`` is the transcription.

What it carries: ``config_thermodynamics_type = 'BL99'``, both heat
conductivities (``bubbly``, ``MU71``), the constant ocean heat transfer
(MPAS accepts ``Cdn_ocn`` only with form drag), high-frequency coupling, any number of boundary
layer iterations, a computed or a coupled surface temperature
(``config_calc_surface_temperature``; the coupled fluxes are
``alternative_atmos_forcing``'s), the ice age and first-year ice tracers.

What it refuses, by name:

- the mushy and zero-layer thermodynamics, melt ponds (level, CESM, topo;
  level ponds are the Registry default), aerosols, form drag and the
  vertical zsalinity are not ported;
- ``config_atmos_boundary_method = 'constant'`` and
  ``config_calc_surface_stresses = .false.``: ``colpkg_step_therm1``'s
  ``Cdn_atm_ratio_n`` is an uninitialised local that only
  ``atmo_boundary_layer`` with ``calc_strair`` sets, and merge_fluxes adds it
  into ``airOceanDragCoefficientRatio`` either way, so the oracle's word
  there is undefined;
- the default build: ``atmo_boundary_layer``'s ``psimhu`` calls glibc's
  binary64 ``atan`` on every boundary-layer iteration (stable layers
  multiply it by 0 but still evaluate it). glibc's ``atan`` is not correctly
  rounded (graded with ``tools/mpas_seaice_oracle/grade_atmo_atan.cpp``:
  10,886 of 18,874,368 arguments in psimhu's [1, 161**0.25] differ from the
  correctly rounded word) and its code is LGPL, so the unit compiles only
  with ``WOOF_GLIBC239_LIBM64=1`` (:mod:`woof.core.glibc239_gpu`), which
  prepends the opt-in transcription and gives glibc's words for every
  argument.

``exp`` and ``log`` are the shared glibc transcriptions (``glibc_flt64.cuh``),
``deltaT**1.36`` ``glibc_pow``; the constants gfortran folds
(``vonkar/log(zref/iceruf)``, ``log(zref/zTrf)``) are folded on the host,
each operation correctly rounded as MPFR does.

On an abort (``l_stop``), MPAS logs the first aborting cell's message and
returns from the whole routine; :func:`vertical_thermodynamics` returns that
cell and message (the state is then not MPAS's: later cells have run).
"""
from __future__ import annotations

import datetime as _dt
from functools import lru_cache
from importlib.resources import files

import numpy as np

from woof.hex import numerics as _hex_numerics
from woof.hex.errors import ConfigurationRefusal

from . import column_v842 as cv
from .velocity_v842 import COMPILE_OPTIONS, _grid

SOURCE_FILES = (
    "src/core_seaice/shared/mpas_seaice_column.F:1064-1666,3286-3307",
    "src/core_seaice/column/ice_colpkg.F90:1719-2281,3414-3513",
    "src/core_seaice/column/ice_atmo.F90:46-366",
    "src/core_seaice/column/ice_therm_vertical.F90:52-1989",
    "src/core_seaice/column/ice_therm_bl99.F90:53-1498",
    "src/core_seaice/column/ice_therm_shared.F90:47-197",
    "src/core_seaice/column/ice_flux_colpkg.F90:30-283",
    "src/core_seaice/column/ice_age.F90:23-34",
    "src/core_seaice/column/ice_firstyear.F90:36-62",
)
_BLOCK = 128
SEAICE_ERROR_COL_VERT_THERM = 10

#: the stop labels by the kernel's abort code
STOP_LABELS = {
    1: "init_vertical_profile: Starting thermo, zTsn > Tmax",
    2: "init_vertical_profile: Starting thermo, zTsn < Tmin",
    3: "init_vertical_profile: Starting zSin < min_salin, layer",
    4: "init_vertical_profile: Starting thermo, T > Tmax, layer",
    5: "init_vertical_profile: Starting thermo, T < Tmin, layer",
    6: "temperature_changes: Thermo iteration does not converge",
    7: "conservation_check_vthermo: Thermo energy conservation error",
}


def _refuse(knob: str, value, reason: str, declaration: str):
    raise ConfigurationRefusal(knob=knob, value=value, reason=reason, declaration=declaration)


def refuse_unported(configs: dict, dims: dict[str, int]) -> None:
    """Refuse, by name, what the vertical thermodynamics port does not carry."""
    from woof.hex.mpas_ice import rkind

    c = configs
    cv.refuse_unported(c, dims)
    if c["config_thermodynamics_type"] != "BL99":
        _refuse("config_thermodynamics_type", c["config_thermodynamics_type"],
                "the mushy and zero-layer vertical thermodynamics are not ported",
                "config_thermodynamics_type = 'BL99'")
    for knob in ("config_use_level_meltponds", "config_use_cesm_meltponds", "config_use_topo_meltponds"):
        if c[knob]:
            _refuse(knob, True, "the melt ponds are not ported", f"{knob} = .false.")
    for knob, what in (("config_use_aerosols", "the aerosols"), ("config_use_form_drag", "form drag"),
                       ("config_use_vertical_zsalinity", "the vertical zsalinity")):
        if c[knob]:
            _refuse(knob, True, f"{what} in the vertical thermodynamics are not ported", f"{knob} = .false.")
    if c["config_atmos_boundary_method"] != "ccsm3":
        _refuse("config_atmos_boundary_method", c["config_atmos_boundary_method"],
                "colpkg_step_therm1 merges its uninitialised Cdn_atm_ratio_n into "
                "airOceanDragCoefficientRatio unless atmo_boundary_layer sets it",
                "config_atmos_boundary_method = 'ccsm3'")
    if not c["config_calc_surface_stresses"]:
        _refuse("config_calc_surface_stresses", False,
                "colpkg_step_therm1 merges its uninitialised Cdn_atm_ratio_n into "
                "airOceanDragCoefficientRatio unless calc_strair sets it",
                "config_calc_surface_stresses = .true.")
    if c["config_heat_conductivity_type"] not in ("bubbly", "MU71"):
        _refuse("config_heat_conductivity_type", c["config_heat_conductivity_type"],
                "colpkg knows the bubbly and MU71 conductivities", "'bubbly' | 'MU71'")
    if c["config_ocean_heat_transfer_type"] != "constant":
        _refuse("config_ocean_heat_transfer_type", c["config_ocean_heat_transfer_type"],
                "the Cdn_ocn transfer needs form drag, which is not ported", "'constant'")
    if not rkind.trig64():
        _refuse("config_thermodynamics_type", c["config_thermodynamics_type"],
                rkind.trig_refusal("atmo_boundary_layer's psimhu calls glibc's double atan"),
                "WOOF_GLIBC239_LIBM64=1 (the vertical thermodynamics need glibc's atan)")


def day_of_year(start_time: str) -> float:
    """``get_day_of_year`` at the clock's start: the integer day (1 on
    1 January) as a real."""
    date = _dt.date(int(start_time[0:4]), int(start_time[5:7]), int(start_time[8:10]))
    return float(date.timetuple().tm_yday)


@lru_cache(maxsize=None)
def folded_constants() -> dict[str, float]:
    """``vonkar/log(zref/iceruf)`` and ``log(zref/zTrf)``, as gfortran folds them."""
    from decimal import Context, Decimal
    from fractions import Fraction as F

    ctx = Context(prec=60)
    log_ratio = float(ctx.ln(Decimal(float(F(10.0) / F(0.0005)))))
    return {"rdn_ice": float(F(0.4) / F(log_ratio)),
            "al2": float(ctx.ln(Decimal(float(F(10.0) / F(2.0)))))}


def module_source(n_ice_layers: int) -> str:
    from woof.hex.mpas_ice import rkind
    body = files("woof.hex.mpas_seaice").joinpath("column_therm_v842.cu").read_text(encoding="utf-8")
    return (rkind.r8_trig_prelude() + f"#define COL_NILYR {int(n_ice_layers)}\n" + cv.module_source()
            + "\n" + body)


@lru_cache(maxsize=None)
def _module_on(device: int, options: tuple[str, ...], n_ice_layers: int, trig: bool):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(n_ice_layers), options, None,
                                   f"seaice_column_therm_v842_l{n_ice_layers}.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _function(name: str, n_ice_layers: int):
    import cupy as cp
    from woof.hex.mpas_ice import rkind
    options = _hex_numerics.options(COMPILE_OPTIONS, keep_ftz=True)
    return _module_on(cp.cuda.Device().id, options, n_ice_layers, rkind.trig64()).get_function(name)


def _flags(configs: dict, state: "cv.ColumnState") -> int:
    heat_capacity = state.ktherm != 0
    l_brine = 3.2 > 0.1 and heat_capacity          # colpkg_init_thermo: saltmax > min_salin
    bits = (l_brine, state.calc_tsfc, configs["config_calc_surface_stresses"],
            configs["config_use_high_frequency_coupling"], configs["config_heat_conductivity_type"] == "MU71",
            configs["config_use_ice_age"],
            configs["config_use_first_year_ice"])
    return sum(int(bool(b)) << i for i, b in enumerate(bits))


def vertical_thermodynamics(pools: dict, configs: dict, state: "cv.ColumnState",
                            day: float | None = None) -> tuple[int, int | None, str | None]:
    """``column_vertical_thermodynamics``: returns ``(ierr, cell, message)``,
    ``(0, None, None)`` when no cell aborts, else the first aborting owned
    cell's ``indexToCellID`` and its logged message."""
    import cupy as cp

    d = state.dims
    refuse_unported(configs, d)
    n_cells, n_solve, ncat, nil, nsl = (d["nCells"], d["nCellsSolve"], d["nCategories"], d["nIceLayers"],
                                       d["nSnowLayers"])
    day = day_of_year(configs["config_start_time"]) if day is None else day
    folded = folded_constants()
    c1, cc, ci, cs = (n_cells + 1,), (n_cells + 1, ncat), (n_cells + 1, ncat, nil), (n_cells + 1, ncat, nsl)
    cat1 = (n_cells + 1, ncat, 1)
    need = cv._need
    t, ta, ist = pools["tracers"], pools["tracers_aggregate"], pools["icestate"]
    vs, ac, af = pools["velocity_solver"], pools["atmos_coupling"], pools["atmos_forcing"]
    alt, oc, dr = pools["alternative_atmos_forcing"], pools["ocean_coupling"], pools["drag"]
    of, mg, fl, sw, dg = (pools["ocean_fluxes"], pools["melt_growth_rates"], pools["atmos_fluxes"],
                          pools["shortwave"], pools["diagnostics"])
    n_vertices = d["nVertices"]
    tr_iage, tr_fy = configs["config_use_ice_age"], configs["config_use_first_year_ice"]
    dummy = cp.zeros(1, np.float64)
    status = cp.zeros(max(n_solve, 1), np.int32)
    args = (
        status, np.float64(configs["config_dt"]), np.float64(day),
        np.float64(configs["config_min_friction_velocity"]), np.float64(folded["rdn_ice"]),
        np.float64(folded["al2"]), np.int32(ncat), np.int32(_flags(configs, state)),
        np.int32(configs["config_boundary_layer_iteration_number"]), np.int32(n_solve),
        need(pools["mesh"], "latCell", c1), need(ist, "iceAreaCellInitial", c1),
        need(ist, "iceAreaCategoryInitial", cc), need(ist, "iceVolumeCategoryInitial", cc),
        need(ist, "snowVolumeCategoryInitial", cc), need(ta, "iceAreaCell", c1),
        need(t, "iceAreaCategory", cat1), need(t, "iceVolumeCategory", cat1), need(t, "snowVolumeCategory", cat1),
        need(t, "surfaceTemperature", cat1), need(t, "snowEnthalpy", cs), need(t, "iceEnthalpy", ci),
        need(t, "iceSalinity", ci), need(t, "iceAge", cat1) if tr_iage else dummy,
        need(t, "firstYearIceArea", cat1) if tr_fy else dummy,
        need(vs, "uVelocity", (n_vertices + 1,)), need(vs, "airStressCellU", c1), need(vs, "airStressCellV", c1),
        need(vs, "oceanStressCellU", c1), need(vs, "oceanStressCellV", c1),
        need(ac, "uAirVelocity", c1), need(ac, "vAirVelocity", c1), need(ac, "airLevelHeight", c1),
        need(ac, "airSpecificHumidity", c1), need(ac, "airDensity", c1), need(ac, "airPotentialTemperature", c1),
        need(ac, "snowfallRate", c1), need(ac, "longwaveDown", c1), need(ac, "atmosReferenceTemperature2m", c1),
        need(ac, "atmosReferenceHumidity2m", c1), need(ac, "atmosReferenceSpeed10m", c1),
        need(af, "windSpeed", c1),
        need(alt, "latentHeatFluxCouple", cc), need(alt, "sensibleHeatFluxCouple", cc),
        need(alt, "surfaceHeatFluxCouple", cc), need(alt, "surfaceConductiveFluxCouple", cc),
        need(oc, "seaSurfaceTemperature", c1), need(oc, "seaSurfaceSalinity", c1),
        need(oc, "seaFreezingTemperature", c1), need(oc, "freezingMeltingPotential", c1),
        need(dr, "airOceanDragCoefficientRatio", c1), need(dr, "oceanDragCoefficient", c1),
        need(dr, "airDragCoefficient", c1),
        need(of, "oceanHeatFluxIceBottom", c1), need(of, "oceanFreshWaterFlux", c1),
        need(of, "oceanSaltFlux", c1), need(of, "oceanHeatFlux", c1), need(of, "oceanShortwaveFlux", c1),
        need(mg, "lateralIceMeltFraction", c1), need(mg, "surfaceIceMelt", c1),
        need(mg, "surfaceIceMeltCategory", cc), need(mg, "basalIceMelt", c1), need(mg, "basalIceMeltCategory", cc),
        need(mg, "snowMelt", c1), need(mg, "snowMeltCategory", cc), need(mg, "congelation", c1),
        need(mg, "congelationCategory", cc), need(mg, "snowiceFormation", c1),
        need(mg, "snowiceFormationCategory", cc), need(mg, "snowThicknessChangeCategory", cc),
        need(fl, "surfaceHeatFlux", c1), need(fl, "surfaceHeatFluxCategory", cc),
        need(fl, "surfaceConductiveFlux", c1), need(fl, "surfaceConductiveFluxCategory", cc),
        need(fl, "longwaveUp", c1), need(fl, "sensibleHeatFlux", c1), need(fl, "sensibleHeatFluxCategory", cc),
        need(fl, "latentHeatFlux", c1), need(fl, "latentHeatFluxCategory", cc),
        need(fl, "evaporativeWaterFlux", c1),
        need(sw, "surfaceShortwaveFlux", cc), need(sw, "interiorShortwaveFlux", cc),
        need(sw, "penetratingShortwaveFlux", cc), need(sw, "absorbedShortwaveFlux", c1),
        need(sw, "absorbedShortwaveSnowLayer", cs), need(sw, "absorbedShortwaveIceLayer", ci),
        need(dg, "meltOnset", c1), need(dg, "freezeOnset", c1),
    )
    if n_solve:
        _function("seaice_column_vertical_thermodynamics_v842", nil)(_grid(n_solve), (_BLOCK,), args)
    codes = cp.asnumpy(status)[:n_solve]
    bad = np.flatnonzero(codes)
    if not bad.size:
        return 0, None, None
    first = int(bad[0])
    cell = int(cp.asnumpy(pools["mesh"]["indexToCellID"][first:first + 1])[0])
    message = "column_vertical_thermodynamics: ice: Vertical thermo error: " + STOP_LABELS[int(codes[first])]
    return SEAICE_ERROR_COL_VERT_THERM, cell, message
