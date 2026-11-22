"""MPAS-Seaice v8.4.2's column ITD thermodynamics on the GPU.

``column_itd_thermodynamics`` (``mpas_seaice_column.F:1680-1976``) and the
column package under it (``colpkg_step_therm2``: ``linear_itd``'s remapping of
the thickness distribution, ``add_new_ice``, ``lateral_melt``,
``reduce_area`` and ``cleanup_itd``'s ``rebin``, ``zap_small_areas`` and
``zap_snow_temperature``), word for word with the gfortran -O0
-ffp-contract=off oracle at RKIND 8. One thread is one owned cell;
``column_itd_v842.cu`` is the transcription. No libm call is on the path, so
nothing here needs an opt-in.

What it carries: the BL99 and mushy thermodynamics (the mushy frazil's
salinity, liquidus and enthalpy, and its vertical regridding), the linear
remap and the delta function (``config_itd_conversion_type``), any category
bounds, one category (``reduce_area``), the level, CESM and topo melt ponds,
the ice age, first-year ice and level-ice tracers,
``config_update_ocean_fluxes`` either way, and the column biogeochemistry's
brine height (``config_use_brine``): the brine fraction travels with the ice
volume through the remap and rebinning, ``add_new_ice_bgc``
(``ice_zbgc.F90:29-251``) gives the new ice its brine fraction, and
``zap_small_areas`` resets it to 1. v8.4.2's vertical salinity
(``config_use_vertical_zsalinity``) is inert here: its ``solve_zsal`` is never
set, so ``cleanup_itd``'s ``dfzsal`` is zero and ``zSalinityFlux`` keeps its
words; its tracers are counted and not copied. Under the opt-in correction
``zsal_solve`` (the solved vertical salinity, held to the patched oracle)
the column package's ``solve_zsal`` branches run: ``add_new_ice``'s
``fzsal`` loss with the frazil and no mixing of the new ice's ``sice``,
``adjust_tracer_profile``'s ``regrid_stationary`` of the vertical salinity
(its remap's stop an ITD abort) and the ``remap_zbgc`` back into ``sice``
(``ice_zbgc.F90:29-251``, ``ice_brine.F90``), the first category's surface
temperature from that ``sice``, and ``lateral_melt_bgc``'s and
``zap_small_areas``' salt into ``zSalinityFlux``.

What it refuses, by name: the zero-layer thermodynamics (``cleanup_itd``'s
``zerolayer_check`` is not ported), the aerosols, and more than
``ITD_MAXCAT`` categories; of the biogeochemistry everything but the brine
height and the inert vertical salinity
(:func:`woof.hex.mpas_seaice.column_v842.refuse_unported`).

v8.4.2's quirks are kept: ``rebin`` drops ``shift_ice``'s abort (cleanup_itd
never reads it before ``zap_small_areas`` resets it) while ``shift_ice`` still
returns before its transfers; ``zap_small_areas`` reads the total area from
before ``rebin``; ``linear_itd`` returns from an abort with the snow
enthalpies still shifted by ``rhos*Lfresh``; ``linear_itd``'s
``hin_max(ncat) = 999.9`` lands in ``categoryThicknessLimits``.

On an abort (``l_stop``), MPAS writes the aborting cell back, logs its message
and returns from the whole routine, so the later cells keep their words:
:func:`itd_thermodynamics` restores them and returns that cell and message.
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
    "src/core_seaice/shared/mpas_seaice_column.F:295-345,1680-1976,5595-5741",
    "src/core_seaice/column/ice_colpkg.F90:2290-2539",
    "src/core_seaice/column/ice_colpkg_tracers.F90:157-246",
    "src/core_seaice/column/ice_therm_itd.F90:69-1503",
    "src/core_seaice/column/ice_itd.F90:53-658,747-1546",
)
_BLOCK = 128
SEAICE_ERROR_COL_ITD_THERM = 11
#: the categories one thread holds (column_itd_v842.cu)
ITD_MAXCAT = 32
_REMAPPED = 256
#: the line zap_snow_temperature logs per out-of-bounds snow layer
ZAP_WARNING = "WARN  zap_snow_temperature: temperature out of bounds!"
#: the opt-in correction of zap_small_areas' zap_snow area (column_v842.CORRECTIONS)
ZAP_SNOW_AREA = "zap_snow_area"

#: the stop labels by the kernel's abort code
STOP_LABELS = {
    1: "shift_ice: negative daice",
    2: "shift_ice: negative dvice",
    3: "shift_ice: daice > aicen",
    4: "shift_ice: dvice > vicen",
    5: "aggregate ice area out of bounds",
    6: "Zap ice: negative ice area",
    7: "Zap ice: excess ice area",
    8: "ice: remap_layers_bgc error",
}
#: the tracer indices the kernel reads, in its tidx order (TI_* in column_itd_v842.cu)
_TRACER_INDICES = ("surfaceTemperature", "iceEnthalpy", "snowEnthalpy", "iceSalinity", "iceAge",
                   "firstYearIceArea", "levelIceArea", "levelIceVolume", "pondArea", "pondDepth")


def _refuse(knob: str, value, reason: str, declaration: str):
    raise ConfigurationRefusal(knob=knob, value=value, reason=reason, declaration=declaration)


def refuse_unported(configs: dict, dims: dict[str, int]) -> None:
    """Refuse, by name, what the ITD thermodynamics port does not carry."""
    c = configs
    cv.refuse_unported(c, dims, biogeochemistry=True, z_tracers=True)
    if ZAP_SNOW_AREA in cv.corrections(c) and not cv.vertical_tracers(c):
        _refuse("woof_v842_corrections", ZAP_SNOW_AREA,
                "it corrects zap_small_areas' vertical tracer flux, which runs only with the vertical tracers",
                "config_use_vertical_tracers = .true. with config_use_column_biogeochemistry, or without it")
    if c["config_thermodynamics_type"] == "zero layer":
        _refuse("config_thermodynamics_type", "zero layer",
                "cleanup_itd's zerolayer_check (the zero-layer thermodynamics) is not ported",
                "config_thermodynamics_type in ('BL99', 'mushy')")
    if c["config_use_aerosols"]:
        _refuse("config_use_aerosols", True, "the aerosols in the ITD thermodynamics are not ported",
                "config_use_aerosols = .false.")
    if dims["nCategories"] > ITD_MAXCAT:
        _refuse("config_nCategories", dims["nCategories"],
                f"one GPU thread holds at most {ITD_MAXCAT} thickness categories",
                f"config_nCategories <= {ITD_MAXCAT}")


def tracer_indices(to: "cv.TracerObject") -> np.ndarray:
    """The kernel's tidx (TI_* in column_itd_v842.cu): 0-based, -1 for a tracer that is off."""
    return np.array([to.index[k] - 1 for k in _TRACER_INDICES] + [to.nt_fbri - 1, to.nt_bgc_s - 1], np.int32)


def _zsal_args(pools: dict, configs: dict, dims: dict[str, int], written: list, grids: bool = True) -> tuple:
    """The kernels' trailing vertical-salinity arguments: nBioLayers (also
    the vertical tracers'), (the
    ITD's biologyGrid and verticalGrid,) zSalinityFlux. Only under
    ``zsal_solve`` is anything read or written (``zSalinityFlux`` then joins
    ``written``, which an abort restores past the aborting cell)."""
    import cupy as cp
    if not cv.solves_zsal(configs):
        dummy = cp.zeros(1, np.float64)
        # nBioLayers is the vertical tracers' too
        nbl = dims["nBioLayers"] if cv.vertical_tracers(configs) else 0
        return (np.int32(nbl), *((dummy, dummy) if grids else ()), dummy)
    bgc, nbl, nil = pools["biogeochemistry"], dims["nBioLayers"], dims["nIceLayers"]
    flux = cv._need(bgc, "zSalinityFlux", (dims["nCells"] + 1,))
    written.append(flux)
    extra = (cv._need(bgc, "biologyGrid", (nbl + 2,)), cv._need(bgc, "verticalGrid", (nil + 1,))) if grids else ()
    return (np.int32(nbl), *extra, flux)


def _ztr_args(pools: dict, configs: dict, state: "cv.ColumnState", written: list, itd: bool = True) -> tuple:
    """The kernels' trailing vertical-tracer arguments: nbtrcr, bio_index(1)
    (0-based), (the ITD's interfaceBiologyGrid, zbgc_init_frac,
    oceanBioConcentrations,) nZBGCTracers, (the ITD's bio_index_o(1),
    0-based,) oceanBioFluxes. Only with the vertical tracers is anything read
    or written (``oceanBioFluxes`` then joins ``written``, which an abort
    restores past the aborting cell)."""
    import cupy as cp
    d, to = state.dims, state.tracers
    if not cv.vertical_tracers(configs):
        dummy = cp.zeros(1, np.float64)
        if itd:
            return (np.int32(0), np.int32(0), dummy, np.float64(0.0), dummy, np.int32(1), np.int32(0), dummy)
        return (np.int32(0), np.int32(0), np.int32(1), dummy)
    bgc, nz, n1 = pools["biogeochemistry"], d["nZBGCTracers"], d["nCells"] + 1
    flux = cv._need(bgc, "oceanBioFluxes", (n1, nz))
    written.append(flux)
    if to.n_bio_tracers != 1:
        # the kernels carry one vertical tracer (one ocean concentration, flux and bio index)
        raise ValueError(f"the ITD and ridging carry one vertical tracer, not {to.n_bio_tracers}")
    nbio, bio0 = np.int32(to.n_bio_tracers), np.int32(to.bio_index[0] - 1)
    if not itd:
        return (nbio, bio0, np.int32(nz), flux)
    init_frac = cv.zbgc_parameters(configs, to)["init_frac"][0]
    return (nbio, bio0, cv._need(bgc, "interfaceBiologyGrid", (d["nBioLayers"] + 1,)), np.float64(init_frac),
            cv._need(bgc, "oceanBioConcentrations", (n1, nz)), np.int32(nz), np.int32(to.bio_index_o[0] - 1), flux)


def module_source(n_ice_layers: int) -> str:
    from .column_bgc_v842 import zbgc_shared_source
    body = files("woof.hex.mpas_seaice").joinpath("column_itd_v842.cu").read_text(encoding="utf-8")
    return (f"#define COL_NILYR {int(n_ice_layers)}\n" + cv.module_source() + "\n" + zbgc_shared_source() + "\n"
            + body)


@lru_cache(maxsize=None)
def _module_on(device: int, options: tuple[str, ...], n_ice_layers: int):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(n_ice_layers), options, None,
                                   f"seaice_column_itd_v842_l{n_ice_layers}.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _function(name: str, n_ice_layers: int):
    import cupy as cp
    options = _hex_numerics.options(COMPILE_OPTIONS, keep_ftz=True)
    return _module_on(cp.cuda.Device().id, options, n_ice_layers).get_function(name)


def _flags(configs: dict, state: "cv.ColumnState") -> int:
    bits = (configs["config_use_ice_age"], configs["config_use_first_year_ice"], configs["config_use_level_ice"],
            configs["config_use_cesm_meltponds"], configs["config_use_level_meltponds"],
            configs["config_use_topo_meltponds"], configs["config_update_ocean_fluxes"],
            state.ktherm != 0,            # colpkg_init_thermo's heat_capacity
            cv.solves_zsal(configs),      # zsal_solve: the column package's solve_zsal
            ZAP_SNOW_AREA in cv.corrections(configs))
    return sum(int(bool(b)) << i for i, b in enumerate(bits))


def itd_thermodynamics(pools: dict, configs: dict, state: "cv.ColumnState",
                       day: float | None = None) -> tuple[int, int | None, str | None]:
    """``column_itd_thermodynamics``: returns ``(ierr, cell, message)``,
    ``(0, None, None)`` when no cell aborts, else the first aborting owned
    cell's ``indexToCellID`` and its logged message. ``state.warnings`` is the
    number of warnings zap_snow_temperature logged. Needs ``init_column_itd``
    (``state.hi_min``) to have run."""
    import cupy as cp

    d = state.dims
    refuse_unported(configs, d)
    if state.hi_min is None:
        raise ValueError("column_itd_thermodynamics needs init_column_itd (hi_min) first")
    n_cells, n_solve, ncat, nil = d["nCells"], d["nCellsSolve"], d["nCategories"], d["nIceLayers"]
    day = (ct.day_of_year(configs["config_start_time"], configs["config_calendar_type"]) if day is None
           else day)
    to = state.tracers
    nt = to.n_tracers
    c1, cc, cat1 = (n_cells + 1,), (n_cells + 1, ncat), (n_cells + 1, ncat, 1)
    need = cv._need
    t, ist, ta = pools["tracers"], pools["icestate"], pools["tracers_aggregate"]
    oc, mg, of = pools["ocean_coupling"], pools["melt_growth_rates"], pools["ocean_fluxes"]
    limits = need(pools["initial"], "categoryThicknessLimits", (ncat + 1,))
    area, volume, snow = (need(t, k, cat1) for k in ("iceAreaCategory", "iceVolumeCategory",
                                                      "snowVolumeCategory"))
    cell_fields = [need(ist, "openWaterArea", c1), need(ta, "iceAreaCell", c1),
                   need(mg, "lateralIceMelt", c1), need(mg, "frazilFormation", c1),
                   need(pools["ponds"], "pondFreshWaterFlux", c1), need(of, "oceanFreshWaterFlux", c1),
                   need(of, "oceanSaltFlux", c1), need(of, "oceanHeatFlux", c1),
                   need(mg, "frazilGrowthDiagnostic", c1), need(pools["diagnostics"], "freezeOnset", c1)]
    (open_water, ice_area_cell, lateral_melt, frazil, pond_flux, fresh, salt, heat, frazil_diag,
     freeze_onset) = cell_fields
    newly = cv._need_int(pools["biogeochemistry"], "newlyFormedIce", cc)
    tracer_fields = [cv._tracer_field(pools, name)[0] for name, _, _ in to.layout]
    written = [area, volume, snow, newly, *cell_fields, *tracer_fields]
    zsal_args = _zsal_args(pools, configs, d, written)
    ztr_args = _ztr_args(pools, configs, state, written)
    before = [a[:n_solve].copy() for a in written]

    trcrn = cp.zeros((max(n_solve, 1), ncat, nt), np.float64)
    for name, first, layers in to.layout:
        src, sub = cv._tracer_field(pools, name)
        cv._launch("seaice_gather_tracer_v842", n_solve * ncat * layers, trcrn, src, np.int32(first - 1),
                   np.int32(layers), np.int32(sub), np.int32(src.shape[2]), np.int32(nt), np.int32(ncat),
                   np.int32(n_solve))
    atrcrn = cp.zeros_like(trcrn)
    tidx = cp.asarray(tracer_indices(to))
    status = cp.zeros(max(n_solve, 1), np.int32)
    nwarn = cp.zeros(max(n_solve, 1), np.int32)
    args = (
        status, nwarn, np.float64(configs["config_dt"]), np.float64(day), np.float64(state.hi_min),
        np.int32(ncat), np.int32(nt), np.int32(_flags(configs, state)), np.int32(state.ktherm),
        np.int32(state.kitd), np.int32(n_solve), tidx, cp.asarray(to.parent_index),
        cp.asarray(np.ascontiguousarray(to.first_ancestor_mask)), cp.asarray(to.ancestor_number),
        cp.asarray(np.ascontiguousarray(to.ancestor_indices)), limits, trcrn, atrcrn, area, volume, snow,
        need(ist, "iceAreaCategoryInitial", cc), need(ist, "iceVolumeCategoryInitial", cc),
        open_water, ice_area_cell, need(oc, "seaFreezingTemperature", c1), need(oc, "seaSurfaceSalinity", c1),
        need(pools["initial"], "initialSalinityProfile", (n_cells + 1, nil + 1)),
        need(mg, "lateralIceMeltFraction", c1), lateral_melt, need(oc, "freezingMeltingPotential", c1),
        frazil, need(pools["atmos_coupling"], "rainfallRate", c1), pond_flux, fresh, salt, heat, frazil_diag,
        freeze_onset, newly, *zsal_args, *ztr_args,
    )
    if n_solve:
        _function("seaice_column_itd_thermodynamics_v842", nil)(_grid(n_solve), (_BLOCK,), args)
    for name, first, layers in to.layout:
        dst, sub = cv._tracer_field(pools, name)
        _function("seaice_scatter_category_tracer_v842", nil)(
            _grid(max(n_solve * ncat * layers, 1)), (_BLOCK,),
            (dst, trcrn, np.int32(first - 1), np.int32(layers), np.int32(sub), np.int32(dst.shape[2]),
             np.int32(nt), np.int32(ncat), np.int32(n_solve)))
    codes = cp.asnumpy(status)[:n_solve]
    warnings = cp.asnumpy(nwarn)[:n_solve]
    stops = codes & (_REMAPPED - 1)
    bad = np.flatnonzero(stops)
    last = int(bad[0]) if bad.size else n_solve - 1
    # MPAS returns at the first abort: the later cells keep their words
    for array, saved in zip(written, before):
        array[last + 1:n_solve] = saved[last + 1:]
    if (codes[:last + 1] & _REMAPPED).any():
        limits[ncat] = 999.9
    state.warnings = int(warnings[:last + 1].sum())
    if not bad.size:
        return 0, None, None
    cell = int(cp.asnumpy(pools["mesh"]["indexToCellID"][last:last + 1])[0])
    return (SEAICE_ERROR_COL_ITD_THERM, cell,
            "column_itd_thermodynamics: " + STOP_LABELS[int(stops[last])])
