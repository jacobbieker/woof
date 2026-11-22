"""MPAS-Seaice v8.4.2's column biogeochemistry on the GPU: brine height, vertical salinity, nonreactive tracer.

``column_biogeochemistry`` (``mpas_seaice_column.F:2781-3272``) and the
column package under it (``colpkg_biogeochemistry``) with
``config_use_column_biogeochemistry`` and ``config_use_brine``, word for word
with the gfortran -O0 -ffp-contract=off oracle at RKIND 8. Per owned cell and
category: the thickness before the step (``iceThicknessCategoryInitial``), new
ice (``newlyFormedIce``, ``rayleighCriteriaReal``), and on ice the brine
height's ``preflushing_changes``, ``compute_microS_mushy`` (the ice salinity
and enthalpy remapped onto the biology grid, ``temperature_mush``,
``liquid_fraction``, ``prepare_hbrine``, ``calculate_drho`` and the
diffusivity) and ``update_hbrine`` (flushing, flooding and the Darcy
velocity), the brine fraction tracer, ``netBrineHeight`` and the uncoupled
build's black carbon and dust deposition. ``column_bgc_v842.cu`` is the
transcription; the framework (the tracer object's brine fraction, the init's
grids and profiles, the ocean concentrations and flux resets) is
:mod:`~woof.hex.mpas_seaice.column_v842`'s.

The vertical salinity (``config_use_vertical_zsalinity``, BL99 only) is
inert in v8.4.2: ``colpkg_init_parameters`` never sets the column package's
``solve_zsal`` and the pool's ``verticalSalinity`` reaches the tracer arrays
only with the vertical tracers, so the step is the brine height's above and
the init only clears ``rayleighCriteriaReal``; that is the default, word for
word. The opt-in correction ``zsal_solve``
(:data:`~woof.hex.mpas_seaice.column_v842.CORRECTIONS`) solves it as the
column package is written to, held word for word to the oracle built with
``tools/mpas_seaice_oracle/zsal_solve.patch``: ``compute_microS`` (first ice,
a brine height above ``maxhbr`` times the ice, old ice; its stop is checked),
``update_hbrine`` with ``dh_direct``, and ``zsalinity`` (``solve_S_dt``'s
Lapidus-smoothed advection-diffusion in ``max(1, int(dt/dts_b))`` steps, or
its add/melt branch; ``bSin(k) > max_salin`` stops; the remap back to the ice
salinity, ``column_sum_zsal``, ``merge_zsal_fluxes``). There the Rayleigh
criterion: ``rayleighCriteriaReal`` is only ever zeroed, which changes nothing,
since ``solve_S_dt`` forces it where its convective branch runs (the brine
height above ``thinS`` = ``Ra_c``); ``totalVerticalSalinity`` is summed into and
never reset.

The vertical tracers (``config_use_vertical_tracers``) are carried with the
nonreactive tracer alone (``config_use_nonreactive``: the column package's
PON, nbtrcr = 1, no vertical biochemistry). On ice, after the brine height,
``zbio``: ``update_snow_bgc`` (the snow's two layers: evaporation, melt
scavenging, snow ice, snowfall, the surface layer's new thickness),
``z_biogeochemistry`` (first ice from the ocean concentration, the shortwave
remapped onto the biology grid into ``bioShortwaveFlux``, then above
``thinS`` the mobile and stationary exchange by the transport type's
retention and release times with the saturation cap, the flux-corrected
transport's low order solve, ``check_conservation_FCT``'s flux,
``compute_FCT_corr``, ``regrid_stationary`` with the bottom melt and
growth, the new mobile fraction; below it ``thin_ice_flux``; then the bulk
concentration) and ``merge_bgc_fluxes`` into ``oceanBioFluxes``,
``snowIceBioFluxes``, ``atmosIceBioFluxes`` and the column totals.
New ice resets the mobile fraction, and the wrapper's tail sums
``bioShortwaveFluxCell``. Its stops: ``zbgc initialization error`` (a
negative tracer), ``negative bgc``, ``remap_zbgc``'s, and
``check_conservation_FCT``'s negative low order solution, after which v8.4.2
logs a stop label nothing set (whatever ``abortMessage`` held). The FCT's
conservation failure and the snow's warnings are warnings only, not kept.
``update_snow_bgc`` divides by the area before the step unguarded, so ice
new to a category (an initial area of 0) would take NaN; the step never
makes that (the area before the step is the step's own), and the cases do
not.

What it refuses, by name: the biogeochemistry without ``config_use_brine``
(``colpkg_biogeochemistry`` writes the brine changes, which only the brine
height's package allocates), every biogeochemistry flag but ``config_use_brine``
but the vertical tracers' nonreactive tracer (the vertical and skeletal
biochemistry and their species, the shortwave bioabsorption, the z-aerosols:
:func:`~woof.hex.mpas_seaice.column_v842.refuse_unported`), more biology
layers than ``compute_microS_mushy``'s ``trtmp(ntrcr+2)`` holds, and more
than 64 categories. ``exp`` is the shared glibc transcription
(``glibc_flt64.cuh``), ``sqrt`` the correctly rounded instruction.

Upstream behaviour kept word for word:

- ``compute_microS_mushy`` adds gravity drainage when the brine height
  ``hbr_old >= Ra_c``, the critical Rayleigh number (0.05): a length compared
  with a dimensionless number, as upstream;
- ``calculate_drho``'s ``mm`` loop sets ``rho_2a(k)`` to the same expression
  twice (or not at all);
- ``remap_zbgc`` would read ``trdr(0)`` were its first donor point below the
  first receiver point; the remap this routine makes (``hinS = hice``) never
  does.

- ``colpkg_biogeochemistry`` does not check ``compute_microS_mushy``'s
  ``l_stop`` (``remap_zbgc``'s negative thickness, ``ice_zbgc_shared.F90:222``):
  that category's profiles keep their words, ``update_hbrine`` runs with
  ``kavg = bphi_o = 0``, and the next ice category's ``preflushing_changes``
  clears the stop (``ice_brine.F90:100``). Only a stop in a cell's last ice
  category reaches the wrapper.

On an abort that reaches the wrapper (that stop, or ``preflushing_changes``'s
``fbri <= 0``, which the reset to 1 before it makes unreachable), MPAS logs
the message as an error and the cell (``iCell: $i``) at ``MPAS_LOG_CRIT``,
which terminates MPAS (``mpas_seaice_column.F:3229-3231``):
:func:`biogeochemistry` raises :class:`CriticalAbort` with the first such
owned cell and the message, and the arrays are not v8.4.2's after it.
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from importlib.resources import files

import numpy as np

from woof.hex import numerics as _hex_numerics
from woof.hex.errors import MpasPortError

from . import column_v842 as cv
from .velocity_v842 import COMPILE_OPTIONS

SOURCE_FILES = (
    "src/core_seaice/shared/mpas_seaice_column.F:2781-3272",
    "src/core_seaice/column/ice_colpkg.F90:4884-5262",
    "src/core_seaice/column/ice_brine.F90:17-943",
    "src/core_seaice/column/ice_zsalinity.F90:38-425,435-1023,1111-1177",
    "src/core_seaice/column/ice_zbgc_shared.F90:110-133,167-333,370-524",
    "src/core_seaice/column/ice_algae.F90:29-340,747-1417,2159-2611,2617-2710",
    "src/core_seaice/column/ice_aerosol.F90:447-826",
    "src/core_seaice/column/ice_zbgc.F90:528-653",
    "src/core_seaice/column/ice_therm_shared.F90:47-73",
    "src/core_seaice/column/ice_mushy_physics.F90:29-110,361-409",
)
_BLOCK = 64
SEAICE_ERROR_COL_BGC = 13
MAX_CATEGORIES = 64
#: the opt-in correction (column_v842.CORRECTIONS) for update_snow_bgc's 0/0 on a
#: category without an initial area
SNOW_NEW_CATEGORY = "bgc_snow_new_category"
STOP_LABELS = {1: "ice_brine preflushing: fbri <= c0", 2: "ice: remap_layers_bgc error",
               3: "bSin(k) > max_salin", 4: "zbgc initialization error", 5: "negative bgc",
               6: "(check_conservation_FCT's stop: v8.4.2 logs an undefined stop_label)"}
#: z_biogeochemistry's constants as gfortran folds them, each operation rounded once
#: (ice_algae.F90:972-977, pi from ice_constants_colpkg): S_col = 4*pi*r_c**2,
#: P_b = pi*r_bac**2, V_c = 4*pi*r_c**3/3*(1e-6)**3, V_alg = pi/6*r_bac*r_alg**2 with
#: r_c = 3e3, r_bac = 15, r_alg = 10; f_v is the default-real literal 0.7854
_PI = 3.14159265358979323846
_S_COL = (4.0 * _PI) * 9.0e6
_P_B = _PI * 225.0
_V_C = (((4.0 * _PI) * 2.7e10) / 3.0) * float(Fraction(1.0e-6) ** 3)
_V_ALG = ((_PI / 6.0) * 15.0) * 100.0
_F_V = float(np.float32(0.7854))


class CriticalAbort(MpasPortError):
    """``column_biogeochemistry``'s abort: v8.4.2 logs the cell at
    ``MPAS_LOG_CRIT`` and terminates, so the port stops the same way.
    ``crash`` is the oracle's record of it (the error, the critical message
    and its integers)."""

    def __init__(self, cell: int, message: str) -> None:
        self.ierr, self.cell, self.message = SEAICE_ERROR_COL_BGC, cell, message
        self.crash = {"error": message, "critical": "iCell: $i", "ints": [cell]}
        super().__init__(f"{message} (indexToCellID {cell}): MPAS v8.4.2 logs it at MPAS_LOG_CRIT and terminates")


def refuse_unported(configs: dict, dims: dict[str, int]) -> None:
    """Refuse, by name, what the brine height's port does not carry."""
    cv.refuse_unported(configs, dims, biogeochemistry=True, z_tracers=True)
    if configs["config_use_column_biogeochemistry"] and not configs["config_use_brine"]:
        cv._refuse("config_use_brine", False,
                   "colpkg_biogeochemistry writes brineTopChange and brineBottomChange, which only the brine "
                   "height's package allocates", "config_use_brine = .true. with the column biogeochemistry")
    if dims["nCategories"] > MAX_CATEGORIES:
        cv._refuse("config_nCategories", dims["nCategories"],
                   f"the brine kernel holds at most {MAX_CATEGORIES} categories",
                   f"config_nCategories <= {MAX_CATEGORIES}")
    if SNOW_NEW_CATEGORY in cv.corrections(configs) and not cv.vertical_tracers(configs):
        cv._refuse("woof_v842_corrections", SNOW_NEW_CATEGORY,
                   "update_snow_bgc (the correction's routine) runs only with the vertical tracers",
                   "config_use_vertical_tracers = .true., or no bgc_snow_new_category")


def module_source(n_ice_layers: int, n_bio_layers: int) -> str:
    body = files("woof.hex.mpas_seaice").joinpath("column_bgc_v842.cu").read_text(encoding="utf-8")
    return (f"#define COL_NILYR {int(n_ice_layers)}\n#define COL_NBLYR {int(n_bio_layers)}\n"
            + cv.module_source() + "\n" + zbgc_shared_source() + "\n" + body)


def zbgc_shared_source() -> str:
    """column_zbgc_shared_v842.cu: remap_zbgc and calculate_Tin_from_qin, which
    the ITD thermodynamics share under ``zsal_solve``."""
    return files("woof.hex.mpas_seaice").joinpath("column_zbgc_shared_v842.cu").read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def _module_on(device: int, options: tuple[str, ...], n_ice_layers: int, n_bio_layers: int):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(n_ice_layers, n_bio_layers), options, None,
                                   f"seaice_column_bgc_v842_l{n_ice_layers}_b{n_bio_layers}.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _function(name: str, n_ice_layers: int, n_bio_layers: int):
    import cupy as cp
    options = _hex_numerics.options(COMPILE_OPTIONS, keep_ftz=True)
    return _module_on(cp.cuda.Device().id, options, n_ice_layers, n_bio_layers).get_function(name)


def biogeochemistry(pools: dict, configs: dict, state: "cv.ColumnState") -> tuple[int, None, None]:
    """``column_biogeochemistry``: returns ``(0, None, None)`` (its ierr,
    cell and message); raises :class:`CriticalAbort` with the first aborting
    owned cell's ``indexToCellID`` and its logged message where v8.4.2
    terminates."""
    import cupy as cp

    d = state.dims
    refuse_unported(configs, d)
    if not configs["config_use_column_biogeochemistry"]:
        return 0, None, None
    n_cells, n_solve, ncat, nil, nbl = (d["nCells"], d["nCellsSolve"], d["nCategories"], d["nIceLayers"],
                                       d["nBioLayers"])
    need, need_int = cv._need, cv._need_int
    c1, cc, cat1, ci = (n_cells + 1,), (n_cells + 1, ncat), (n_cells + 1, ncat, 1), (n_cells + 1, ncat, nil)
    t, ist, mg, oc, bgc = (pools["tracers"], pools["icestate"], pools["melt_growth_rates"],
                           pools["ocean_coupling"], pools["biogeochemistry"])
    tr_brine = bool(configs["config_use_brine"])
    solve_zsal = cv.solves_zsal(configs)
    z_tracers = cv.vertical_tracers(configs)
    snow_new_category = SNOW_NEW_CATEGORY in cv.corrections(configs)
    flags = int(tr_brine) | int(solve_zsal) << 1 | int(z_tracers) << 2 | int(snow_new_category) << 3
    status = cp.zeros(max(n_solve, 1), np.int32)
    if solve_zsal:
        zsal = need(t, "verticalSalinity", (n_cells + 1, ncat, nbl))
    else:
        zsal = cp.zeros(1, np.float64)       # never read without the vertical salinity
    args = (
        status, np.float64(configs["config_dt"]),
        np.float64(configs["config_bio_gravity_drainage_length_scale"]), np.int32(ncat), np.int32(flags),
        np.int32(n_solve),
        need(bgc, "biologyGrid", (nbl + 2,)), need(bgc, "interfaceBiologyGrid", (nbl + 1,)),
        need(bgc, "verticalGrid", (nil + 1,)),
        need(ist, "iceAreaCategoryInitial", cc), need(ist, "iceVolumeCategoryInitial", cc),
        need(t, "iceAreaCategory", cat1), need(t, "iceVolumeCategory", cat1), need(t, "snowVolumeCategory", cat1),
        need(mg, "basalIceMeltCategory", cc), need(mg, "surfaceIceMeltCategory", cc),
        need(mg, "congelationCategory", cc), need(mg, "snowiceFormationCategory", cc),
        need(mg, "snowMeltCategory", cc),
        need(oc, "seaSurfaceTemperature", c1), need(oc, "seaSurfaceSalinity", c1),
        need(ist, "openWaterArea", c1),
        need(t, "iceEnthalpy", ci), need(t, "iceSalinity", ci),
        need(t, "brineFraction", cat1),
        need(ist, "iceThicknessCategoryInitial", cc), need_int(bgc, "newlyFormedIce", cc),
        need(bgc, "rayleighCriteriaReal", c1),
        need(bgc, "brineTopChange", cc), need(bgc, "brineBottomChange", cc),
        need(bgc, "bioTemperature", (n_cells + 1, ncat, nbl + 2)),
        need(bgc, "bioPorosity", (n_cells + 1, ncat, nbl + 2)),
        need(bgc, "bioDiffusivity", (n_cells + 1, ncat, nbl + 1)),
        need(bgc, "darcyVelocityBio", cc), need(bgc, "zSalinityIceDensity", cc), need(bgc, "netBrineHeight", c1),
        need(bgc, "atmosBlackCarbonFlux", (n_cells + 1, 2)), need(bgc, "atmosDustFlux", (n_cells + 1, 4)),
        need(bgc, "totalSkeletalAlgae", c1), need(bgc, "bioShortwaveFluxCell", (n_cells + 1, nbl + 1)),
        np.float64(cv.zsalinity_dts(nbl)), np.float64(configs["config_zsalinity_gravity_drainage_scale"]),
        np.float64(configs["config_zsalinity_molecular_sublayer"]), need(t, "surfaceTemperature", cat1), zsal,
        need(bgc, "bioPermeability", (n_cells + 1, ncat, nbl + 1)), need(bgc, "totalVerticalSalinity", c1),
        need(bgc, "zSalinityFlux", c1), need(bgc, "zSalinityGDFlux", c1),
        *_z_tracer_args(pools, configs, state),
    )
    if n_solve:
        # atmosBioFluxes(:,:) = 0 on the first cell's turn: the whole array
        cv._fill(need(bgc, "atmosBioFluxes", (n_cells + 1, d["nZBGCTracers"])), 0.0)
        _function("seaice_column_biogeochemistry_v842", nil, nbl)(((n_solve + _BLOCK - 1) // _BLOCK,), (_BLOCK,), args)
    codes = cp.asnumpy(status)[:n_solve]
    bad = np.flatnonzero(codes)
    if not bad.size:
        return 0, None, None
    first = int(bad[0])
    cell = int(cp.asnumpy(pools["mesh"]["indexToCellID"][first:first + 1])[0])
    raise CriticalAbort(cell, "column_biogeochemistry: " + STOP_LABELS[int(codes[first])])


def zbgc_constants(configs: dict, tracers: "cv.TracerObject") -> np.ndarray:
    """The kernel's ``zpar``: the nonreactive tracer's ``colpkg_init_zbgc``
    parameters (:func:`~woof.hex.mpas_seaice.column_v842.zbgc_parameters`),
    ``grid_o`` and z_biogeochemistry's folded constants."""
    par = cv.zbgc_parameters(configs, tracers)
    return np.array([par["type"][0], par["tau_ret"][0], par["tau_rel"][0], par["frac_init"][0],
                     par["init_frac"][0], float(configs["config_biogrid_bottom_molecular_sublayer"]), _F_V, _V_C,
                     _S_COL, _P_B, _V_ALG], np.float64)


def _z_tracer_args(pools: dict, configs: dict, state: "cv.ColumnState") -> tuple:
    """The kernel's vertical tracer arguments (dummies without them)."""
    import cupy as cp

    d, to = state.dims, state.tracers
    nzbgc = d["nZBGCTracers"]
    if not cv.vertical_tracers(configs):
        dummy = cp.zeros(1, np.float64)
        return (dummy, dummy, np.int32(nzbgc), dummy, np.int32(0), dummy, dummy, dummy, dummy, dummy, dummy,
                dummy, dummy, dummy, dummy, dummy, dummy)
    need = cv._need
    n1, ncat, nil, nbl = d["nCells"] + 1, d["nCategories"], d["nIceLayers"], d["nBioLayers"]
    t, ist, bgc = pools["tracers"], pools["icestate"], pools["biogeochemistry"]
    fz = (n1, nzbgc)
    return (
        need(t, "verticalNonreactiveConc", (n1, ncat, nbl + 3)), need(t, "mobileFraction", (n1, ncat, nzbgc)),
        np.int32(nzbgc), cp.asarray(zbgc_constants(configs, to)), np.int32(to.bio_index_o[0] - 1),
        need(bgc, "oceanBioConcentrations", fz), need(pools["atmos_coupling"], "snowfallRate", (n1,)),
        need(ist, "snowVolumeCategoryInitial", (n1, ncat)),
        need(pools["shortwave"], "shortwaveLayerPenetration", (n1, ncat, nil + 1)),
        need(bgc, "interfaceGrid", (nil + 1,)), need(bgc, "bioShortwaveFlux", (n1, ncat, nbl + 1)),
        need(bgc, "oceanBioFluxes", fz), need(bgc, "snowIceBioFluxes", fz), need(bgc, "atmosIceBioFluxes", fz),
        need(bgc, "totalVerticalBiologyIce", fz), need(bgc, "totalVerticalBiologySnow", fz),
        need(bgc, "atmosBioFluxes", fz),
    )
