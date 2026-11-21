"""The E3SM v3.2.0 coupler with the MPAS v8.4.2 ocean, on the GPU, opt-in.

This is not an E3SM configuration. E3SM v3.2.0 builds its own MPAS-Ocean
(``components/mpas-ocean``), which has moved away from MPAS-Model v8.4.2
(the land-ice fluxes, frazil, equation of state and constants differ). Here
the ocean is v8.4.2's (:mod:`woof.hex.mpas_ocean`, every piece held word for
word to v8.4.2) and only the coupling around it is E3SM's: the coupler's
import and export (:mod:`.ocn_coupling_e3sm`) and the order in which
``ocn_init_mct`` and ``ocn_run_mct`` (``components/mpas-ocean/driver/
ocn_comp_mct.F``) call the component. :data:`HYBRID_NAME` says so wherever
the pairing is named.

:class:`HybridOcean` is the component:

- at start (``ocn_init_mct``, ``runtype = 'initial'``): v8.4.2's cold start
  (:func:`~woof.hex.mpas_ocean.forward_v842.cold_start`), the absorbed
  fractions, frazil and land-ice arrays built again (E3SM's order), the
  ocean's clock moved to its first coupling alarm without a step (the
  ocean's lag), the coupler's running means reset and accumulated once,
  the export (:meth:`HybridOcean.initial_o2x`), and the import of the
  coupler's still-zero x2o;
- each window (``ocn_run_mct``, :meth:`HybridOcean.run_window`): the import,
  the running means reset (v8.4.2's ``ocn_time_average_coupled_init``), then
  ``ocn_cpl_dt / config_dt`` steps of E3SM's loop around v8.4.2's step (the
  absorbed fractions, frazil, the land-ice fluxes, the clock advanced, the
  split step, ``xtime`` and the days since the start, the time levels
  shifted), and the export.

Where E3SM's ocean loop does something v8.4.2's forward mode does not, the
hybrid ports it, refuses it by name, or leaves it out because it changes
nothing the hybrid carries; :data:`DIVERGENCES` lists every one with its
reason and E3SM's line. The constants are ``ocn_constants_init``'s under
``MPAS_ESM_SHR_CONST`` (as E3SM builds MPAS; ``constants='e3sm'``, the
default) or v8.4.2's own standalone build (``constants='mpas'``); they
differ only in ``T0_Kelvin`` (273.15 against 273.16) and in ``rho_sw``
(1026 against ``config_density0``), so the E3SM build refuses another
``config_density0``.

``real(8)`` only (E3SM runs MPAS at RKIND 8), one block. Nothing on the
default path calls this. Held word for word to the hybrid oracle
(``tools/e3sm_coupler_oracle/run_hybrid_ocean.F90``: E3SM's import and
export cut verbatim, compiled against v8.4.2's modules, around v8.4.2's
forward mode; ``tests/test_e3sm_hybrid_ocean_parity.py``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from woof.hex.errors import ConfigurationRefusal

#: what the pairing is, wherever it is named (code, docs, registry)
HYBRID_NAME = "E3SM v3.2.0 coupler with MPAS v8.4.2 components, not an E3SM configuration"
E3SM_TAG = "v3.2.0"
MPAS_TAG = "v8.4.2"
SOURCE_FILES = ("components/mpas-ocean/driver/ocn_comp_mct.F:ocn_init_mct:781-791,833-846,908,992",
                "components/mpas-ocean/driver/ocn_comp_mct.F:ocn_run_mct:1032-1450",
                "src/core_ocean/shared/mpas_ocn_time_average_coupled.F:48-189 (v8.4.2)")
#: ocn_constants_init's (rho_sw, cp_sw, T0_Kelvin): E3SM's build (MPAS_ESM_SHR_CONST, shr_const_mod)
SHR_RHO_SW, SHR_CP_SW, SHR_T0_KELVIN = 1.026e3, 3.996e3, 273.15
#: v8.4.2's standalone build: rho_sw is config_density0
MPAS_CP_SW, MPAS_T0_KELVIN = 3.996e3, 273.16
CONSTANTS = ("e3sm", "mpas")


@dataclass(frozen=True)
class Divergence:
    """One thing E3SM's ocean driver does that v8.4.2's forward mode does not.

    ``disposition``: 'ported' (the hybrid does it), 'refused' (the named
    knob or field is refused), 'omitted' (left out: ``reason`` says why it
    changes nothing the hybrid carries)."""

    name: str
    where: str
    disposition: str
    reason: str
    knob: str = ""


DIVERGENCES = (
    Divergence("the forcing arrays built frazil first, then the land ice",
               "ocn_comp_mct.F:1187-1197", "ported",
               "E3SM builds the absorbed fractions, frazil, then the land-ice fluxes; v8.4.2's forward mode "
               "builds the land-ice fluxes before frazil. The hybrid keeps E3SM's order (in the modes it "
               "accepts the land-ice build returns at once, so the order changes no word)"),
    Divergence("ocn_surface_land_ice_fluxes_accumulate_fluxes", "ocn_comp_mct.F:1198-1199", "omitted",
               "E3SM-only; it updates accumulatedLandIceMass/Heat only in 'standalone' or 'data', which the "
               "hybrid refuses (config_land_ice_flux_mode), so it does nothing in what the hybrid accepts"),
    Divergence("ocn_eddy_compute_mixed_layer_depth", "ocn_comp_mct.F:1201", "omitted",
               "E3SM-only; it writes the eddy module's dThreshMLD and indMLD, which v8.4.2's step has no "
               "place for and which only GM, the submesoscale eddies and Redi read (refused below)"),
    Divergence("GM bolus velocity, submesoscale eddies, Redi", "ocn_comp_mct.F:1203-1214", "refused",
               "v8.4.2's port refuses GM and Redi (config_use_standardGM); v8.4.2 has no submesoscale "
               "eddies", "config_use_standardGM / config_submesoscale_enable"),
    Divergence("ocn_update_scaled_dismf", "ocn_comp_mct.F:1292", "refused",
               "E3SM-only, under config_scale_dismf_by_removed_ice_runoff, which needs the removed ice "
               "runoff v8.4.2 lacks", "config_scale_dismf_by_removed_ice_runoff"),
    Divergence("ocn_update_scaling_factor", "ocn_comp_mct.F:1333", "refused",
               "E3SM-only, under config_use_precip_scaling (the import's precipitation factor too)",
               "config_use_precip_scaling"),
    Divergence("the running means reset each window (ocn_time_average_coupled_init)",
               "ocn_comp_mct.F:1165", "ported",
               "E3SM resets the coupler's means after the import; v8.4.2 has the routine (its forward "
               "mode never calls it). The hybrid calls v8.4.2's: the surface means and, in 'coupled', the "
               "land-ice means, and the count. E3SM's own also zeroes arrays v8.4.2 lacks (below)"),
    Divergence("ocn_validate_state, analysis members, streams, restart writes",
               "ocn_comp_mct.F:1293-1335", "omitted",
               "they change no state (as in the forward port)"),
    Divergence("forcing streams read in the loop (shortwave data, ecosystem, CFC, monthly salinity)",
               "ocn_comp_mct.F:1225-1257", "refused",
               "the v8.4.2 port refuses those options already (config_sw_absorption_type 'ohlmann00', the "
               "tracer packages, monthly surface salinity restoring)", "config_sw_absorption_type"),
    Divergence("the ocean clock moved to its first coupling alarm at start", "ocn_comp_mct.F:833-837", "ported",
               "runtype 'initial': the clock is advanced one coupling interval without a step"),
    Divergence("x2o PFioi_bergh, PFioi_bergw (iceberg heat and freshwater)", "ocn_comp_mct.F:2506-2518",
               "refused",
               "the coupler always sends them; v8.4.2 has no iceberg forcing, so the import's lookup finds "
               "no field and drops them. They are zero unless the sea ice runs data icebergs: a nonzero "
               "value is refused at the import", "x2o:PFioi_bergh/PFioi_bergw"),
    Divergence("the removed Antarctic runoff", "ocn_comp_mct.F:2457-2465,2523-2545", "refused",
               "v8.4.2 has no removedRiverRunoffFlux or removedIceRunoffFlux",
               "config_remove_ais_river_runoff / config_remove_ais_ice_runoff"),
    Divergence("the 10 m wind speed (KPP theory wave, EKE)", "ocn_comp_mct.F:2467-2470,2616-2620", "refused",
               "v8.4.2 has no windSpeed10m", "config_cvmix_kpp_use_theory_wave / config_eke_correction_enable"),
    Divergence("the waves coupled two ways, the active-wave stress", "ocn_comp_mct.F:2474-2490,2561-2614", "refused",
               "v8.4.2 has no Stokes drift wavenumbers or wave fields", "wav_ocn_coup / "
               "config_momentum_use_active_wave"),
    Divergence("o2x Faoo_h2otemp", "ocn_comp_mct.F:3364", "omitted",
               "the coupler always carries it; v8.4.2 has no avgTotalFreshWaterTemperatureFlux, so the hybrid "
               "gives the export a zero array there and the field is 0. Its only readers are the coupler's "
               "budget diagnostics (seq_diag_mct) and its copy to the atmosphere"),
    Divergence("land-ice 'standalone' and 'data' exports (Foxo_ismw, Foxo_ismh, Foxo_q_li, Foxo_frazil_li)",
               "ocn_comp_mct.F:3298-3301,3367-3370", "refused",
               "v8.4.2 makes no avgLandIceFreshwaterFlux or avgLandIceHeatFlux and has no 'data' mode",
               "config_land_ice_flux_mode"),
    Divergence("the glc 3-D thermal forcing", "ocn_comp_mct.F:3309-3313,3481-3490", "refused",
               "v8.4.2 has no avgThermalForcingAtZLevels", "config_glc_thermal_forcing_coupling_mode"),
    Divergence("the ecosystem, DMS, MacroMolecules and CFC coupling", "ocn_comp_mct.F:ocn_import_mct, ocn_export_mct (the tracer branches)",
               "refused", "not ported (the E3SM cap refuses them too)", "config_use_*Tracers"),
    Divergence("ocn_freezing_temperature in the export", "ocn_comp_mct.F:3404", "ported",
               "the cap calls the component's: v8.4.2's (no mushy term, a reference pressure), not "
               "E3SM's (a mushy term). A negative open-ocean reference pressure is refused (the export's "
               "pressure is 0, so v8.4.2's offset is 0 for any other)",
               "config_open_ocean_freezing_temperature_reference_pressure"),
    Divergence("the cap's module-level land-ice boundary layer (ocn_diagnostics_variables)",
               "ocn_comp_mct.F:3470-3478", "ported",
               "E3SM keeps landIceBoundaryLayerTracers and landIceTracerTransferVelocities in a module; "
               "v8.4.2 keeps them in its diagnostics pool, which the hybrid hands the export"),
    Divergence("ocn_constants: MPAS_ESM_SHR_CONST", "mpas_ocn_constants.F:153-167 (both)", "ported",
               "E3SM builds MPAS with shr_const_mod's constants (T0_Kelvin 273.15, rho_sw 1026); the "
               "default here. v8.4.2's standalone build (273.16, config_density0) is constants='mpas'"),
)


def divergence(name: str) -> Divergence:
    for d in DIVERGENCES:
        if d.name == name:
            return d
    raise KeyError(name)


def _refuse(knob: str, value, reason: str, declaration: str):
    raise ConfigurationRefusal(knob=knob, value=value, reason=f"{HYBRID_NAME}: {reason}", declaration=declaration)


#: (knob, refused value test, reason, declaration)
_REFUSED_CONFIGS = (
    ("config_use_precip_scaling", True, "E3SM's precipitation scaling (ocn_update_scaling_factor) is E3SM-only",
     ".false."),
    ("config_scale_dismf_by_removed_ice_runoff", True,
     "E3SM's scaled ice-shelf melt (ocn_update_scaled_dismf) is E3SM-only", ".false."),
    ("config_remove_ais_river_runoff", True, "v8.4.2 has no removedRiverRunoffFlux", ".false."),
    ("config_remove_ais_ice_runoff", True, "v8.4.2 has no removedIceRunoffFlux", ".false."),
    ("config_cvmix_kpp_use_theory_wave", True, "v8.4.2 has no windSpeed10m", ".false."),
    ("config_eke_correction_enable", True, "v8.4.2 has no windSpeed10m", ".false."),
    ("config_momentum_use_active_wave", True, "v8.4.2 has no wave fields", ".false."),
    ("config_submesoscale_enable", True, "v8.4.2 has no submesoscale eddies", ".false."),
    ("config_use_ecosysTracers", True, "the ecosystem coupling is not ported", ".false."),
    ("config_use_DMSTracers", True, "the DMS coupling is not ported", ".false."),
    ("config_use_MacroMoleculesTracers", True, "the MacroMolecules coupling is not ported", ".false."),
    ("config_use_CFCTracers", True, "the CFC coupling is not ported", ".false."),
)


def check(configs: dict, *, ocn_cpl_dt: int, constants: str = "e3sm", wav_ocn_coup: str = "none",
          glc_nzoc: int = 0) -> None:
    """Refuse, by name, what the hybrid does not carry (:data:`DIVERGENCES`)."""
    from woof.hex.mpas_ocean import forward_v842 as fw
    from woof.hex.mpas_ocean import land_ice_v842
    if constants not in CONSTANTS:
        raise ValueError(f"constants must be one of {CONSTANTS}")
    for knob, value, reason, declaration in _REFUSED_CONFIGS:
        if configs.get(knob, False) == value:
            _refuse(knob, value, reason, f"{knob} = {declaration}")
    if wav_ocn_coup.rstrip() != "none":
        _refuse("wav_ocn_coup", wav_ocn_coup, "v8.4.2 has no wave fields", "wav_ocn_coup = 'none'")
    mode = land_ice_v842.mode(configs)
    if mode in ("standalone", "data"):
        _refuse("config_land_ice_flux_mode", mode,
                "v8.4.2 makes no avgLandIceFreshwaterFlux or avgLandIceHeatFlux, which E3SM's export sends "
                "in 'standalone' and 'data'", "config_land_ice_flux_mode = 'off' | 'pressure_only' | 'coupled'")
    tf = str(configs.get("config_glc_thermal_forcing_coupling_mode", "off")).rstrip()
    if tf != "off" or glc_nzoc:
        _refuse("config_glc_thermal_forcing_coupling_mode", tf, "v8.4.2 has no avgThermalForcingAtZLevels",
                "config_glc_thermal_forcing_coupling_mode = 'off', glc_nzoc = 0")
    ref = float(configs.get("config_open_ocean_freezing_temperature_reference_pressure", 0.0))
    if not ref >= 0.0:
        _refuse("config_open_ocean_freezing_temperature_reference_pressure", ref,
                "the export's freezing point is v8.4.2's at pressure 0, held for a reference pressure >= 0",
                "config_open_ocean_freezing_temperature_reference_pressure >= 0")
    if constants == "e3sm" and float(configs["config_density0"]) != SHR_RHO_SW:
        _refuse("config_density0", configs["config_density0"],
                "under E3SM's constants rho_sw is shr_const_mod's 1026 while the v8.4.2 port takes rho_sw from "
                "config_density0", "config_density0 = 1026.0, or constants='mpas'")
    if configs.get("config_use_standardGM", False):
        _refuse("config_use_standardGM", True, "GM and Redi are not ported", "config_use_standardGM = .false.")
    dt = fw.interval_seconds(configs["config_dt"])
    if int(ocn_cpl_dt) != ocn_cpl_dt or ocn_cpl_dt <= 0 or ocn_cpl_dt % dt:
        _refuse("ocn_cpl_dt", ocn_cpl_dt, "the ocean steps until its coupling alarm rings; the interval must be "
                "a whole number of steps", f"ocn_cpl_dt a positive multiple of config_dt ({dt} s)")


def time_average_init(forcing: dict, configs: dict) -> None:
    """v8.4.2's ``ocn_time_average_coupled_init``: the surface means zeroed
    over the forcing pool's cells, the land-ice means too in 'coupled', the
    count to 0 (the ecosystem means are refused with their packages)."""
    from woof.hex.mpas_ocean import land_ice_v842
    n = int(forcing["nCells"])
    names = ["avgSurfaceVelocity", "avgTracersSurfaceValue", "avgSSHGradient"]
    if land_ice_v842.mode(configs) == "coupled":
        names += ["avgLandIceBoundaryLayerTracers", "avgLandIceTracerTransferVelocities",
                  "avgEffectiveDensityInLandIce"]
    for name in names:
        forcing[name][:n] = 0
    forcing["nAccumulatedCoupled"] = 0


#: ocn_diagnostics_variables' indices (E3SM) and v8.4.2's diagnostics pool's
_BL_INDICES = (("indexBLT", "index_landIceBoundaryLayerTemperature"),
               ("indexBLS", "index_landIceBoundaryLayerSalinity"),
               ("indexHeatTrans", "index_landIceHeatTransferVelocity"),
               ("indexSaltTrans", "index_landIceSaltTransferVelocity"))
#: the x2o fields v8.4.2 has no target for and the coupler always sends
DROPPED_X2O = ("PFioi_bergh", "PFioi_bergw")


class HybridOcean:
    """The MPAS v8.4.2 ocean under E3SM v3.2.0's coupler (:data:`HYBRID_NAME`).

    ``pools`` and ``configs`` are a v8.4.2 forward-mode start
    (:mod:`woof.hex.mpas_ocean.forward_v842`, real(8)); ``x2o_fields`` and
    ``o2x_fields`` the coupler's lists; ``ocn_cpl_dt`` the coupling interval
    in seconds. Construction is ``ocn_init_mct`` for an initial run."""

    name = HYBRID_NAME

    def __init__(self, *, pools: dict, configs: dict, x2o_fields, o2x_fields, ocn_cpl_dt: int,
                 constants: str = "e3sm", rof2ocn_nutrients: bool = False, wav_ocn_coup: str = "none",
                 ocn_lnd_one_way: bool = False, glc_nzoc: int = 0) -> None:
        import cupy as cp
        from woof.hex.mpas_ocean import forward_v842 as fw
        from woof.hex.mpas_ocean.tend_v842 import _precision_of

        from . import ocn_coupling_e3sm as cpl
        check(configs, ocn_cpl_dt=ocn_cpl_dt, constants=constants, wav_ocn_coup=wav_ocn_coup, glc_nzoc=glc_nzoc)
        if _precision_of(pools["state"]["layerThickness"]) != "r8":
            raise ValueError(f"{HYBRID_NAME}: real(8) only (E3SM runs MPAS at RKIND 8)")
        self.pools, self.configs, self.constants = pools, configs, constants
        self.idx = cpl.indices(x2o_fields, o2x_fields, rof2ocn_nutrients=rof2ocn_nutrients,
                               wav_ocn_coup=wav_ocn_coup, ocn_lnd_one_way=ocn_lnd_one_way, glc_nzoc=glc_nzoc)
        c = configs
        from woof.hex.mpas_ocean import frazil_v842
        # the export's freezing point is v8.4.2's: E3SM's formula with no mushy term (1/inf = 0, see check)
        self.opts = cpl.Options(
            ocn_cpl_dt=int(ocn_cpl_dt), config_density0=float(c["config_density0"]),
            frazil_ice_active=frazil_v842.is_on(c),
            config_land_ice_flux_mode=str(c.get("config_land_ice_flux_mode", "off")),
            config_frazil_heat_of_fusion=float(frazil_v842._config(c, "config_frazil_heat_of_fusion")),
            open_ocean_freezing=tuple(float(c.get(f"config_open_ocean_freezing_temperature_{k}", 0.0))
                                      for k in ("coeff_0", "coeff_S", "coeff_p", "coeff_pS")) + (math.inf,))
        self.opts.check()
        if constants == "e3sm":
            self.t0_kelvin, self.export_constants = SHR_T0_KELVIN, (SHR_RHO_SW, SHR_CP_SW, SHR_T0_KELVIN)
        else:
            self.t0_kelvin = MPAS_T0_KELVIN
            self.export_constants = (float(c["config_density0"]), MPAS_CP_SW, MPAS_T0_KELVIN)
        f = pools["forcing"]
        # Faoo_h2otemp's source, which v8.4.2 lacks (DIVERGENCES)
        if f.get("avgTotalFreshWaterTemperatureFlux") is None:
            f["avgTotalFreshWaterTemperatureFlux"] = cp.zeros(int(pools["mesh"]["nCells"]) + 1, dtype=np.float64)
        self.n_steps = int(ocn_cpl_dt) // fw.interval_seconds(c["config_dt"])
        self.n_btr = fw.btr_subcycle_count(c["config_dt"], c["config_btr_dt"])
        self.n_cells_solve = int(pools["mesh"]["nCellsSolve"])
        self._o2x_vector = cp.zeros((self.n_cells_solve, self.idx.no2x), dtype=np.float64)
        # ocn_init_mct: the core's init (v8.4.2's), then the forcing arrays built again
        self.clock = fw.cold_start(pools=pools, configs=c)
        self._build_forcing()
        # runtype 'initial': the clock to the coupling alarm, no step; the means reset and taken once
        self.clock.now += int(ocn_cpl_dt)
        time_average_init(f, c)
        self._accumulate(time_level=1)
        self._o2x = self._export()
        # the init's last import: x2o as mct_aVect_zero left it (every array it writes the first window
        # writes again)
        self._import(cp.zeros((self.n_cells_solve, self.idx.nx2o), dtype=np.float64))
        self.windows = 0

    # ------------------------------------------------------------ pieces

    def _build_forcing(self) -> None:
        from woof.hex.mpas_ocean import forcing_v842, frazil_v842, land_ice_v842
        from woof.hex.mpas_ocean import forward_v842 as fw
        p, c = self.pools, self.configs
        st, d, m, f = p["state"], p["diagnostics"], p["mesh"], p["forcing"]
        dt = float(self.clock.step)
        forcing_v842.fraction_absorbed(mesh=m, layer_thickness=st["layerThickness"], diagnostics=d, forcing=f)
        frazil_v842.build_arrays(dt=dt, mesh=m, state=st, tracers=fw._tracers(p, 1), diagnostics=d, forcing=f,
                                 configs=c)
        land_ice_v842.build_arrays(dt=dt, mesh=m, state=st, diagnostics=d, forcing=f, configs=c)

    def _accumulate(self, *, time_level: int) -> None:
        from woof.hex.mpas_ocean import split_step_v842
        st = self.pools["state"]
        rho = st.get("effectiveDensityInLandIce" if time_level == 1 else "effectiveDensityInLandIce@2")
        split_step_v842.time_average_accumulate(diagnostics=self.pools["diagnostics"], forcing=self.pools["forcing"],
                                                effective_density=rho, configs=self.configs,
                                                t0_kelvin=self.t0_kelvin)

    def _bl_diagnostics(self) -> dict:
        d = self.pools["diagnostics"]
        out = {k: d.get(k) for k in ("landIceBoundaryLayerTracers", "landIceTracerTransferVelocities")}
        for e3sm, v842 in _BL_INDICES:
            if d.get(v842) is not None:
                out[e3sm] = int(d[v842])
        return {k: v for k, v in out.items() if v is not None}

    def _export(self):
        import cupy as cp

        from . import ocn_coupling_e3sm as cpl
        f = self.pools["forcing"]
        o2x = self._o2x_vector  # MCT's o2x_o: zeroed once at init, then only written
        dims = {k: int(f[k]) for k in ("index_avgTemperatureSurfaceValue", "index_avgSalinitySurfaceValue",
                                        "index_avgSurfaceVelocityZonal", "index_avgSurfaceVelocityMeridional",
                                        "index_avgSSHGradientZonal", "index_avgSSHGradientMeridional")}
        state = {"ssh": self.pools["state"]["ssh"], "layerThickness": self.pools["state"]["layerThickness"]}
        if self.opts.frazil_ice_active:
            state["accumulatedFrazilIceMass"] = self.pools["state"]["accumulatedFrazilIceMass"]
        cpl.export_o2x(self.idx, self.opts, o2x, n_cells_solve=self.n_cells_solve, forcing=f, state=state,
                       dims=dims, diagnostics=self._bl_diagnostics(), constants=self.export_constants)
        return cp.array(o2x, copy=True)

    def _import(self, x2o) -> None:
        import cupy as cp

        from . import ocn_coupling_e3sm as cpl
        for name in DROPPED_X2O:
            col = self.idx.x(name)
            if col >= 0 and bool(cp.any(x2o[:, col] != 0)):
                _refuse(f"x2o:{name}", "nonzero", "v8.4.2 has no iceberg forcing to take it",
                        "no data icebergs in the sea ice (the field zero)")
        f = self.pools["forcing"]
        forcing = {k: v for k, v in f.items() if isinstance(v, cp.ndarray) and v.dtype == np.float64}
        cpl.import_x2o(self.idx, self.opts, x2o, n_cells_solve=self.n_cells_solve,
                       lat_cell=self.pools["mesh"]["latCell"], forcing=forcing)

    # ------------------------------------------------------------ the run

    def initial_o2x(self):
        """The export ``ocn_init_mct`` sends."""
        return self._o2x

    def run_window(self, x2o):
        """``ocn_run_mct``: one coupling window from ``x2o`` ``(nCellsSolve,
        nx2o)`` (float64, device); returns the window's ``o2x``."""
        import cupy as cp
        from woof.hex.mpas_ocean import forward_v842 as fw
        from woof.hex.mpas_ocean import split_step_v842
        from woof.hex.mpas_ocean.tend_v842 import PRECISIONS
        p, c = self.pools, self.configs
        self._import(x2o)
        time_average_init(p["forcing"], c)
        real = PRECISIONS["r8"]
        start = self.clock.seconds(c["diagnostics:simulationStartTime"], "diagnostics:simulationStartTime")
        days_per_second = real(real(1.0) / real(86400.0))
        for _ in range(self.n_steps):
            self._build_forcing()
            self.clock.now += self.clock.step
            split_step_v842.split_step(dt=float(self.clock.step), n_btr_subcycles=self.n_btr, pools=p, configs=c,
                                       t0_kelvin=self.t0_kelvin)
            p["diagnostics"]["xtime"] = self.clock.time_stamp
            p["diagnostics"]["daysSinceStartOfSim"][...] = cp.asarray(
                real(real(self.clock.now - start) * days_per_second))
            fw._shift(p)
        self._o2x = self._export()
        self.windows += 1
        return self._o2x

    # ------------------------------------------------------------ on the spine (run_e3sm.Components)

    def ocean_start(self, step: int, x2o):
        """The ocean's run alarm: E3SM's ``ocn_run_mct`` runs the whole window
        from this import; its export is sent back at :meth:`ocean_finish`."""
        del step
        self.run_window(x2o)
        return None

    def ocean_finish(self, step: int):
        del step
        return self._o2x
