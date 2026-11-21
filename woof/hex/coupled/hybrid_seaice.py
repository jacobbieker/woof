"""The E3SM v3.2.0 coupler with the MPAS v8.4.2 sea ice, on the GPU, opt-in.

This is not an E3SM configuration. E3SM v3.2.0 builds its own MPAS-Seaice
(``components/mpas-seaice``, on Icepack), which has moved away from
MPAS-Model v8.4.2's sea ice (on the column package). Here the sea ice is
v8.4.2's (:mod:`woof.hex.mpas_seaice`, every piece held word for word to
v8.4.2's standalone build) and only the coupling around it is E3SM's: the
coupler's import and export (:mod:`.ice_coupling_e3sm`) and the order in
which ``ice_init_mct`` and ``ice_run_mct`` (``components/mpas-seaice/driver/
ice_comp_mct.F``) call the component. :data:`HYBRID_NAME` says so wherever
the pairing is named; it is the same pairing as
:mod:`~woof.hex.coupled.hybrid_ocean`'s.

:class:`HybridSeaIce` is the component:

- at start (``ice_init_mct``, ``runtype = 'initial'``): the coupler's
  ``nextsw_cday`` into ``dayOfNextShortwaveCalculation``, the core's init
  (:func:`~woof.hex.mpas_seaice.timestep_v842.initialize`), the coupling
  alarm (``ice_cpl_dt / config_dt`` steps), the column's coupling
  preparation (v8.4.2's ``seaice_column_coupling_prep`` where E3SM calls
  ``seaice_icepack_coupling_prep``), the export
  (:meth:`HybridSeaIce.initial_i2x`) and the import of the coupler's
  still-zero x2i;
- each window (``ice_run_mct``, :meth:`HybridSeaIce.run_window`): the
  column's fluxes re-initialised (v8.4.2's
  ``seaice_column_reinitialize_fluxes``), the import, ``nextsw_cday``, v8.4.2's
  ``post_atmospheric_coupling`` and ``post_oceanic_coupling``
  (:mod:`~woof.hex.mpas_seaice.post_coupling_v842`), then, with the step
  counter restarted at 1, ``ice_cpl_dt / config_dt`` times: the clock
  advanced, on the run's very first step ``seaice_init_post_clock_advance``
  (the column shortwave's init), and ``seaice_timestep`` with no
  ``seaice_timestep_finalize``; then the export.

Where E3SM's sea-ice loop does something v8.4.2's ``seaice_core_run`` does
not (or the other way round), the hybrid ports it, refuses it by name, or
leaves it out with its reason; :data:`DIVERGENCES` lists every one with
E3SM's line. The constants are v8.4.2's standalone column package's
(``column/constants/cice``), the build the port is held to; E3SM's cap
reads them through ``seaice_constants`` and Icepack's interface, which the
oracle supplies from the column package (:data:`V842_CONSTANTS`).

``real(8)`` only, one block. Nothing on the default path calls this. Held
word for word to the hybrid oracle
(``tools/e3sm_coupler_oracle/run_hybrid_seaice.F90``: E3SM's import and
export cut verbatim, compiled against v8.4.2's modules, around v8.4.2's
time step with the column package and the velocity solver;
``tests/test_e3sm_hybrid_seaice_parity.py``).
"""
from __future__ import annotations

import numpy as np

from woof.hex.errors import ConfigurationRefusal

from . import ice_coupling_e3sm as icc
from .hybrid_ocean import E3SM_TAG, HYBRID_NAME, MPAS_TAG, Divergence

__all__ = ("HYBRID_NAME", "E3SM_TAG", "MPAS_TAG", "DIVERGENCES", "V842_CONSTANTS", "HybridSeaIce", "check")

SOURCE_FILES = ("components/mpas-seaice/driver/ice_comp_mct.F:ice_init_mct:640-700,780-930",
                "components/mpas-seaice/driver/ice_comp_mct.F:ice_run_mct:1147-1417",
                "src/core_seaice/shared/mpas_seaice_forcing.F:585-677,1503-1591 (v8.4.2)",
                "src/core_seaice/shared/mpas_seaice_initialize.F:seaice_init_post_clock_advance (v8.4.2)")

#: the column package's constants v8.4.2's standalone build compiles
#: (column/constants/cice/ice_constants_colpkg.F90 and ice_colpkg_shared's
#: phi_init and dSin0_frazil), as E3SM's cap reads them here
_LSUB, _LVAP = 2.835e6, 2.501e6
V842_CONSTANTS = icc.Constants(
    tffresh=273.15, rhoi=917.0, rhow=1026.0, rhos=330.0, gravity=9.80616, cp_ice=2106.0, cp_ocn=4218.0,
    latice=_LSUB - _LVAP, lfresh=_LSUB - _LVAP, ref_salinity=4.0, puny=1.0e-11, depress_t=0.054, tocnfrz=-1.8,
    frazil_salinity_reduction=3.0, frazil_ice_porosity=0.75)


DIVERGENCES = (
    Divergence("seaice_icepack_reinitialize_fluxes once per window, not per step", "ice_comp_mct.F:1224",
               "ported",
               "E3SM re-initialises the column's atmospheric and oceanic fluxes once, before the import; "
               "v8.4.2's seaice_core_run does it after every step (seaice_timestep_finalize). The hybrid "
               "does E3SM's: v8.4.2's seaice_column_reinitialize_fluxes (the same two calls; E3SM's also "
               "zeroes the snow tracers' fluxes, which v8.4.2 lacks and the port refuses) before each "
               "import, and no finalize after a step, so a window of several steps sums their fluxes"),
    Divergence("the step counter restarted each window", "ice_comp_mct.F:1275,1283", "ported",
               "E3SM sets itimestep = 0 before each window's loop, so mesh itimestep runs 1..n per "
               "window; v8.4.2's counts on. Nothing else in v8.4.2's step reads it"),
    Divergence("seaice_icepack_coupling_prep at start", "ice_comp_mct.F:844-856", "ported",
               "E3SM calls it under config_column_physics_type 'icepack'; v8.4.2 has no Icepack, and its "
               "column package's equivalent (the mixed layer, the coupling preparation and the flux "
               "scaling) is seaice_column_coupling_prep, which the hybrid calls there"),
    Divergence("config_column_physics_type", "ice_comp_mct.F:844,1213,2285", "refused",
               "v8.4.2 has only the column package; anything but 'column_package' (E3SM's 'icepack') is "
               "refused", "config_column_physics_type"),
    Divergence("post_atmospheric_coupling and post_oceanic_coupling after the import", "ice_comp_mct.F:1260-1261",
               "ported",
               "v8.4.2 has both (its standalone run calls the atmospheric one from its forcing); the hybrid "
               "runs v8.4.2's after each window's import, as E3SM runs its own"),
    Divergence("nextsw_cday into dayOfNextShortwaveCalculation", "ice_comp_mct.F:659-663,1256-1257", "ported",
               "written at start and each window from the coupler's value, as E3SM does. v8.4.2's standalone "
               "column (no CCSMCOUPLED) computes the solar zenith angle at the step's own day and does not "
               "read it (below)"),
    Divergence("the column built with CCSMCOUPLED (v8.4.2's ESM build)", "core_seaice/build_options.mk "
               "(v8.4.2)", "omitted",
               "under ESM v8.4.2 compiles -Dcoupled -DCCSMCOUPLED and constants/cesm: the zenith angle from "
               "nextsw_cday and shr_orb, BL99's tolerances, hi_min, the coupled BGC ocean arrays and "
               "shr_const's constants. The port is v8.4.2's standalone build (constants/cice), which the "
               "hybrid runs; v8.4.2's ESM build, like E3SM's own sea ice, is later work "
               "(docs/mpas-coupled-roadmap.md)"),
    Divergence("seaice_constants and Icepack in the cap", "ice_comp_mct.F:58-68,78,2366,3190-3260", "ported",
               "E3SM's cap reads seaiceReferenceSalinity, seaiceFrazilSalinityReduction and "
               "seaiceFrazilIcePorosity (absent from v8.4.2's seaice_constants) and Icepack's "
               "sea_freezing_temperature, liquidus_temperature and enthalpy_mush. The hybrid takes them "
               "from v8.4.2's column package (ice_ref_salinity, dSin0_frazil, phi_init and colpkg's "
               "functions of those names, the same arithmetic on the tfrz options v8.4.2 accepts), so the "
               "cap's constants are V842_CONSTANTS"),
    Divergence("first-step forcing and seaice_init_post_clock_advance", "ice_comp_mct.F:1287-1295", "ported",
               "E3SM runs them after the run's first clock advance, as v8.4.2's seaice_core_run does: the "
               "column shortwave's init (seaice_init_column_shortwave) under the column shortwave"),
    Divergence("seaice_forcing_get each step", "ice_comp_mct.F:1288,1294", "refused",
               "it reads v8.4.2's forcing streams, which a coupled sea ice does not use and the port "
               "refuses", "config_use_forcing"),
    Divergence("prescribed ice (ice_prescribed_init, ice_prescribed_run)", "ice_comp_mct.F:924,1305",
               "refused", "E3SM-only (config_use_prescribed_ice); v8.4.2 has no prescribed-ice mode",
               "config_use_prescribed_ice"),
    Divergence("data icebergs (berg_fluxes, PFioi_bergw/bergh)", "ice_comp_mct.F:3000-3010", "refused",
               "v8.4.2 has no berg_fluxes; the export writes the two fields 0 without them",
               "config_use_data_icebergs"),
    Divergence("the floe size distribution and the column waves", "ice_comp_mct.F:2393-2400,3012-3020",
               "refused", "v8.4.2 has no floe_size_distribution or wave_coupling",
               "config_use_floe_size_distribution / config_use_column_waves / wav_ice_coup"),
    Divergence("the dib scaled by the removed ice runoff", "ice_comp_mct.F:645-651,1244-1250", "refused",
               "E3SM-only (berg_forcing's runningMeanRemovedIceRunoff)", "config_scale_dib_by_removed_ice_runoff"),
    Divergence("the coupled biogeochemistry and z-aerosols", "ice_comp_mct.F:2329-2350,2420-2520", "refused",
               "v8.4.2 lacks DOCPoolFractions and the wet and dry dust fluxes, and the port refuses the "
               "biogeochemical tracers", "config_couple_biogeochemistry_fields / config_use_zaerosols"),
    Divergence("the bulk and modal aerosol deposition", "ice_comp_mct.F:2400-2420", "refused",
               "the port refuses the aerosols in the shortwave", "config_use_aerosols"),
    Divergence("the slab ocean (config_ocean_surface_type other than 'free')", "ice_comp_mct.F:2380-2388",
               "refused", "the coupled ocean is MPAS's free surface here; the sea ice's own mixed layer "
               "(config_use_ocean_mixed_layer) is refused too",
               "config_ocean_surface_type / config_use_ocean_mixed_layer"),
    Divergence("seaice_timestep's error stop", "ice_comp_mct.F:1315", "ported",
               "E3SM's seaice_timestep stops the model itself; v8.4.2's returns ierr for "
               "seaice_check_critical_error to stop on. The hybrid raises ColumnAbort there"),
    Divergence("the coupling alarm (ice_cpl_dt a whole number of steps)", "ice_comp_mct.F:797-841", "refused",
               "E3SM stops when ice_cpl_dt is shorter than config_dt or not a multiple of it", "ice_cpl_dt"),
    Divergence("continue and branch runs (restart, nextsw_cday = -1)", "ice_comp_mct.F:663,707-720", "refused",
               "runtype 'initial' only; the port restarts from its own restart", "config_do_restart"),
    Divergence("analysis members, streams, restart and output alarms, timers, the MOAB path",
               "ice_comp_mct.F:1279-1281,1308-1375", "omitted", "they change no state"),
)


def divergence(name: str) -> Divergence:
    for d in DIVERGENCES:
        if d.name == name:
            return d
    raise KeyError(name)


def _refuse(knob: str, value, reason: str, declaration: str):
    raise ConfigurationRefusal(knob=knob, value=value, reason=f"{HYBRID_NAME}: {reason}", declaration=declaration)


#: (knob, reason) refused when true
_REFUSED_TRUE = (
    ("config_use_forcing", "the sea-ice forcing streams (seaice_forcing_get) are not used by a coupled sea ice"),
    ("config_use_prescribed_ice", "prescribed ice is E3SM-only"),
    ("config_use_data_icebergs", "v8.4.2 has no berg_fluxes"),
    ("config_use_floe_size_distribution", "v8.4.2 has no floe_size_distribution"),
    ("config_use_column_waves", "v8.4.2 has no wave_coupling"),
    ("config_scale_dib_by_removed_ice_runoff", "the scaled dib is E3SM-only"),
    ("config_couple_biogeochemistry_fields", "v8.4.2 lacks DOCPoolFractions (the coupled biogeochemistry)"),
    ("config_use_zaerosols", "v8.4.2 lacks atmosWetDustFlux and atmosDryDustFlux (the z-aerosols)"),
    ("config_use_aerosols", "the aerosols in the shortwave are not ported"),
    ("config_use_modal_aerosols", "the modal aerosols are not ported"),
    ("config_use_ocean_mixed_layer", "the coupled ocean is the mixed layer (the sea ice's own one is refused)"),
    ("config_do_restart", "runtype 'initial' only"),
)


def check(configs: dict, *, ice_cpl_dt: int, wav_ice_coup: str = "none") -> None:
    """Refuse, by name, what the hybrid does not carry (:data:`DIVERGENCES`)."""
    if not configs.get("config_use_column_package", False):
        _refuse("config_use_column_package", False, "E3SM's sea ice runs its column physics; the cap reads the "
                "column package's fluxes", "config_use_column_package = .true.")
    physics = str(configs.get("config_column_physics_type", "column_package")).rstrip()
    if physics != "column_package":
        _refuse("config_column_physics_type", physics, "v8.4.2 has only the column package (no Icepack)",
                "config_column_physics_type = 'column_package'")
    for knob, reason in _REFUSED_TRUE:
        if configs.get(knob, False):
            _refuse(knob, True, reason, f"{knob} = .false.")
    surface = str(configs.get("config_ocean_surface_type", "free")).rstrip()
    if surface != "free":
        _refuse("config_ocean_surface_type", surface, "the slab ocean is not carried; the coupled ocean is "
                "MPAS's free surface", "config_ocean_surface_type = 'free'")
    if wav_ice_coup.rstrip() != "none":
        _refuse("wav_ice_coup", wav_ice_coup, "v8.4.2 has no wave_coupling", "wav_ice_coup = 'none'")
    dt = float(configs["config_dt"])
    if int(ice_cpl_dt) != ice_cpl_dt or ice_cpl_dt <= 0 or dt != int(dt) or ice_cpl_dt < dt \
            or ice_cpl_dt % int(dt):
        _refuse("ice_cpl_dt", ice_cpl_dt, "the sea ice steps until its coupling alarm rings; E3SM stops on an "
                "interval shorter than config_dt or not a multiple of it",
                f"ice_cpl_dt a positive multiple of config_dt ({dt} s)")


def options(configs: dict) -> icc.Options:
    """The cap's switches from the sea ice's configs (waves, icebergs and the
    floe size distribution refused by :func:`check`)."""
    c = configs
    return icc.Options(
        config_thermodynamics_type=str(c["config_thermodynamics_type"]),
        config_sea_freezing_temperature_type=str(c["config_sea_freezing_temperature_type"]),
        config_ocean_surface_type=str(c.get("config_ocean_surface_type", "free")),
        config_rotate_cartesian_grid=bool(c["config_rotate_cartesian_grid"]),
        config_use_topo_meltponds=bool(c["config_use_topo_meltponds"]),
        config_use_column_shortwave=bool(c["config_use_column_shortwave"]),
        config_use_column_biogeochemistry=bool(c.get("config_use_column_biogeochemistry", False)))


class HybridSeaIce:
    """The MPAS v8.4.2 sea ice under E3SM v3.2.0's coupler (:data:`HYBRID_NAME`).

    ``pools`` and ``configs`` are a v8.4.2 start with the column package and
    the velocity solver (:mod:`woof.hex.mpas_seaice.timestep_v842`, real(8)),
    with ``ocean_coupling/frazilMassAdjust`` (the cap's own array, which
    v8.4.2 lacks; made if absent); ``x2i_fields`` and ``i2x_fields`` the
    coupler's lists; ``ice_cpl_dt`` the coupling interval in seconds;
    ``nextsw_cday`` the coupler's at start. ``trig`` ``(nCells, 8)`` is the
    rotation's words (:func:`~woof.hex.coupled.ice_coupling_e3sm.rotation_trig`
    over every cell with ``sphere_radius`` when not given). Construction is
    ``ice_init_mct`` for an initial run. ``basis``, ``metric_libm`` and
    ``basis_corrections`` go to
    :func:`~woof.hex.mpas_seaice.timestep_v842.initialize`. The default,
    ``"input"``, keeps the basis the pools carry (the coupled harness's pools
    come from an initialised model, and the HybridSeaIce API takes the
    sphere as ``sphere_radius``, not as the ``mesh:`` configs the set-up
    reads). ``basis="compute"`` computes it from the mesh as v8.4.2's init
    does; at real(8) its metric terms need ``metric_libm``."""

    name = HYBRID_NAME

    def __init__(self, *, pools: dict, configs: dict, x2i_fields, i2x_fields, ice_cpl_dt: int,
                 nextsw_cday: float = -1.0, trig=None, sphere_radius: float | None = None,
                 wav_ice_coup: str = "none", basis: str = "input", metric_libm: str | None = None,
                 basis_corrections=()) -> None:
        import cupy as cp
        from woof.hex.mpas_seaice import coupling_prep_v842 as cprep
        from woof.hex.mpas_seaice import timestep_v842 as ts
        check(configs, ice_cpl_dt=ice_cpl_dt, wav_ice_coup=wav_ice_coup)
        if pools["velocity_solver"]["uVelocity"].dtype != np.float64:
            raise ValueError(f"{HYBRID_NAME}: real(8) only (E3SM runs MPAS at RKIND 8)")
        self.pools, self.configs = pools, configs
        self.idx = icc.indices(x2i_fields, i2x_fields, wav_ice_coup=wav_ice_coup)
        self.opts = options(configs)
        self.opts.check()
        self.constants = V842_CONSTANTS
        m = pools["mesh"]
        self.n_cells, self.n_cells_solve = int(m["nCells"]), int(m["nCellsSolve"])
        oc = pools["ocean_coupling"]
        if oc.get("frazilMassAdjust") is None:
            oc["frazilMassAdjust"] = cp.zeros(self.n_cells + 1, dtype=np.float64)
        if trig is None:
            if sphere_radius is None:
                raise ValueError("trig or sphere_radius is needed (the rotation's words)")
            trig = cp.asarray(icc.rotation_trig(*(cp.asnumpy(m[k]) for k in (
                "latCell", "lonCell", "xCell", "yCell", "zCell")), sphere_radius,
                bool(configs["config_rotate_cartesian_grid"]), self.n_cells))
        if trig.dtype != np.float64 or trig.ndim != 2 or trig.shape[1] != 8 or int(trig.shape[0]) < self.n_cells:
            raise ValueError(f"trig must be float64 (nCells, 8); it is {trig.shape}")
        self.trig = cp.ascontiguousarray(trig[:self.n_cells])
        self.n_steps = int(ice_cpl_dt) // int(float(configs["config_dt"]))
        self._i2x_vector = cp.zeros((self.n_cells_solve, self.idx.ni2x), dtype=np.float64)
        # the export reads the pond arrays only under the topographic ponds (basal_pressure);
        # without the ponds' package v8.4.2 has none, so it sees zeros there
        self._export_pools = dict(pools)
        ta = dict(pools["tracers_aggregate"])
        for name in ("pondDepthCell", "pondLidThicknessCell", "pondAreaCell"):
            if ta.get(name) is None:
                if configs.get("config_use_topo_meltponds", False):
                    raise ValueError(f"tracers_aggregate/{name} is absent under the topographic ponds")
                ta[name] = cp.zeros(self.n_cells + 1, dtype=np.float64)
        self._export_pools["tracers_aggregate"] = ta
        # ice_init_mct: nextsw_cday, then the core's init
        self._nextsw(nextsw_cday)
        self.clock = ts.initialize(pools=pools, configs=configs, basis=basis, metric_libm=metric_libm,
                                   basis_corrections=basis_corrections)
        # the coupling preparation (E3SM's seaice_icepack_coupling_prep; v8.4.2's column one)
        cprep.coupling_prep(pools=pools, n_cells_solve=self.n_cells_solve, dt=float(configs["config_dt"]),
                            configs=configs)
        self._i2x = self._export()
        self._import(cp.zeros((self.n_cells_solve, self.idx.nx2i), dtype=np.float64))
        self.first = True
        self.windows = 0

    # ------------------------------------------------------------ pieces

    def _nextsw(self, nextsw_cday: float) -> None:
        self.pools["shortwave"]["dayOfNextShortwaveCalculation"][...] = float(nextsw_cday)

    def _import(self, x2i) -> None:
        icc.import_x2i(self.idx, self.opts, x2i, n_cells_solve=self.n_cells_solve, pools=self.pools,
                       constants=self.constants)

    def _export(self):
        import cupy as cp
        i2x = self._i2x_vector  # MCT's i2x_i: zeroed once, then the export zeroes and fills it
        icc.export_i2x(self.idx, self.opts, i2x, n_cells_solve=self.n_cells_solve, pools=self._export_pools,
                       trig=self.trig[:self.n_cells_solve], constants=self.constants)
        return cp.array(i2x, copy=True)

    def _post_clock_advance(self, step: int) -> None:
        """``seaice_init_post_clock_advance`` on the run's first step only."""
        from woof.hex.mpas_ocean.forward_v842 import format_time
        from woof.hex.mpas_seaice import column_radiation_v842 as crad
        from woof.hex.mpas_seaice import column_step_v842 as cstep
        from woof.hex.mpas_seaice import timestep_v842 as ts
        del step
        if not self.first:
            return
        self.first = False
        c = self.configs
        if c["config_use_column_package"] and c["config_use_column_shortwave"] and not c["config_do_restart"]:
            day, sec = cstep.clock_day(c, format_time(self.clock.now, self.clock.calendar))
            crad.init_shortwave(self.pools, c, ts._column_state(self.pools, c), day=day, sec=sec)

    # ------------------------------------------------------------ the run

    def initial_i2x(self):
        """The export ``ice_init_mct`` sends."""
        return self._i2x

    def run_window(self, x2i, nextsw_cday: float = -1.0):
        """``ice_run_mct``: one coupling window from ``x2i`` ``(nCellsSolve,
        nx2i)`` (float64, device) and the coupler's ``nextsw_cday``; returns
        the window's ``i2x``. A column abort raises
        :class:`~woof.hex.mpas_seaice.timestep_v842.ColumnAbort`."""
        from woof.hex.mpas_seaice import column_step_v842 as cstep
        from woof.hex.mpas_seaice import post_coupling_v842 as post
        from woof.hex.mpas_seaice import timestep_v842 as ts
        p, c = self.pools, self.configs
        cstep.reinitialize_fluxes(p, c)
        self._import(x2i)
        self._nextsw(nextsw_cday)
        post.post_atmospheric_coupling(pools=p, configs=c, trig=self.trig)
        post.post_oceanic_coupling(pools=p, configs=c, trig=self.trig)
        for step in range(1, self.n_steps + 1):
            ts.advance(pools=p, configs=c, clock=self.clock, n_steps=1, first_step=step, finalize_last=False,
                       after_clock=self._post_clock_advance)
        self._i2x = self._export()
        self.windows += 1
        return self._i2x
