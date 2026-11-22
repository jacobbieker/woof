"""Cases and the stream layout for the MPAS-Seaice column-package oracle.

``tools/mpas_seaice_oracle/run_column.F90`` runs MPAS-Model v8.4.2's
``seaice_init_column_physics_package_parameters`` on one block, then the
stage a case names (``oracle_stage``):

- ``init``: ``initialize_coupler_fields``,
  ``seaice_init_column_physics_package_variables``, ``init_ice_state`` and
  ``seaice_reset_coupler_fluxes``, in ``seaice_init``'s order;
- ``aggregate``: ``seaice_column_aggregate``;
- ``update_state``: ``seaice_column_update_state`` (``oracle_update_type``,
  the stream's dt, ``oracle_ice_age_time_offset``);
- ``vertical_thermodynamics``: ``colpkg_init_thermo`` (the column package's
  ``heat_capacity`` and ``l_brine``, as seaice_init sets them), then
  ``column_vertical_thermodynamics`` at the clock's start
  (``config_start_time``), its ierr logged as ``TOBJ IERR``. Its cases carry
  a physically consistent BL99 column state (:func:`_thermo_state`);
- ``itd_thermodynamics``: ``colpkg_init_thermo`` and ``init_column_itd``
  (``categoryThicknessLimits`` and ``hi_min``), then
  ``column_itd_thermodynamics`` at the clock's start, its ierr logged the
  same way. Its cases carry a thickness distribution before and after a
  thermodynamic step (:func:`_itd_state`);
- ``ridging``: ``colpkg_init_thermo`` and ``init_column_itd``, then
  ``column_ridging``, its ierr logged the same way. Its cases carry a
  thickness distribution after transport and the ridging forcing
  (:func:`_ridge_state`);
- ``column_step``: ``colpkg_init_thermo``, ``init_column_itd`` and, for the
  delta-Eddington shortwave, ``colpkg_init_orbit``, then ``oracle_n_steps``
  times the column package's part of ``seaice_timestep`` (the
  re-initialisations, the pre-dynamics, each dynamics subcycle's column
  dynamics and the post-dynamics, each step after the first preceded by
  ``seaice_timestep_finalize``'s ``seaice_column_reinitialize_fluxes``), its
  ierr logged the same way. Its cases carry a BL99 column state and forcing
  (:func:`_step_state`);
- ``biogeochemistry``: ``column_biogeochemistry``, its ierr logged the same
  way; its abort logs the cell at ``MPAS_LOG_CRIT``, which terminates MPAS
  (the stub records it, :func:`run_oracle`'s ``crash``). Its cases (:data:`BGC_CASES`, with the init's under the column
  biogeochemistry) carry a brine-height state (:func:`_bgc_state`); the
  vertical salinity's (:data:`BGC_ZSAL_CASES`, inert as in v8.4.2, and
  :data:`BGC_ZSAL_SOLVE_CASES` under ``zsal_solve``, against the oracle built
  with ``WOOF_ORACLE_PATCH=zsal_solve``) add :func:`_zsal_state`;
- ``seaice_step``: the column step's setup and ``run_seaice_step.F90``'s
  (the velocity solver's time steps, ``seaice_init_evp``,
  ``seaice_init_advection``, ``seaice_initialize_time_diagnostics``), then
  ``oracle_n_steps`` times the clock, ``seaice_timestep`` with the column
  package and the velocity solver compiled verbatim and
  ``seaice_timestep_finalize`` (the last step's under
  ``oracle_finalize_last``), stopping on an abort; the ierr and the step are
  logged (``TOBJ IERR``, ``TOBJ STEP``). Its cases (:data:`COUPLED_CASES`)
  carry the column step's state on a culled hexagonal torus with the
  time-step oracle's dynamics state (:func:`_coupled_dynamics`);
- ``ocean_mixed_layer``: ``seaice_column_ocean_mixed_layer``. Its cases
  carry the open ocean and the air above it (:func:`_oml_state`).

``mpas_seaice_column.F``, ``mpas_seaice_initialize.F``'s and
``mpas_seaice_forcing.F``'s routines and the column package are compiled
verbatim with gfortran -O0 -ffp-contract=off at RKIND 8 (the column package
is dbl_kind; the wrappers do not compile at RKIND 4). The stream is
:mod:`woof.verify.mpas_ocean_stream`'s. The field table is the Registry's
(``woof/hex/mpas_seaice/data/registry-fields.json``): every real and integer
field of the pools the routines read, a field whose packages are off absent,
as MPAS allocates them.

The fixture keeps, per case, every output field the oracle changed and the
tracer object the oracle logs (``column-logs.json``).
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from woof.hex.mpas_seaice import column_bgc_v842 as cbgc
from woof.hex.mpas_seaice import column_itd_v842 as citd
from woof.hex.mpas_seaice import column_radiation_v842 as crad
from woof.hex.mpas_seaice import column_ridge_v842 as crdg
from woof.hex.mpas_seaice import column_step_v842 as cstep
from woof.hex.mpas_seaice import column_therm_v842 as ct
from woof.hex.mpas_seaice import column_v842 as cv
from woof.hex.mpas_seaice import ocean_mixed_layer_v842 as oml
from woof.verify import mpas_ocean_stream as _stream

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mpas_seaice_column"
#: the vertical thermodynamics' fixture (VT_CASES)
VT_FIXTURE_DIR = FIXTURE_DIR.parent / "mpas_seaice_column_therm"
#: the ITD thermodynamics' fixture (ITD_CASES)
ITD_FIXTURE_DIR = FIXTURE_DIR.parent / "mpas_seaice_column_itd"
#: the ridging's fixture (RIDGE_CASES)
RIDGE_FIXTURE_DIR = FIXTURE_DIR.parent / "mpas_seaice_column_ridge"
#: the shortwave's fixture (RADIATION_CASES)
RADIATION_FIXTURE_DIR = FIXTURE_DIR.parent / "mpas_seaice_column_radiation"
#: the delta-Eddington shortwave's fixture (DEDD_CASES)
DEDD_FIXTURE_DIR = FIXTURE_DIR.parent / "mpas_seaice_column_dedd"
#: the column step's fixture (STEP_CASES)
STEP_FIXTURE_DIR = FIXTURE_DIR.parent / "mpas_seaice_column_step"
#: the time step with the column package (COUPLED_CASES)
COUPLED_FIXTURE_DIR = FIXTURE_DIR.parent / "mpas_seaice_coupled_step"
#: the ocean mixed layer's fixture (OML_CASES and OML_STEP_CASES)
OML_FIXTURE_DIR = FIXTURE_DIR.parent / "mpas_seaice_ocean_mixed_layer"
#: the column biogeochemistry's (BGC_CASES)
BGC_FIXTURE_DIR = FIXTURE_DIR.parent / "mpas_seaice_bgc_brine"
#: the brine fraction carried through the ITD thermodynamics, the ridging and the steps
BGC_CARRY_FIXTURE_DIR = FIXTURE_DIR.parent / "mpas_seaice_bgc_brine_carry"
#: the vertical salinity's (BGC_ZSAL_CASES)
BGC_ZSAL_FIXTURE_DIR = FIXTURE_DIR.parent / "mpas_seaice_bgc_zsal"
#: the vertical salinity under zsal_solve (BGC_ZSAL_SOLVE_CASES), from the patched oracle
BGC_ZSAL_SOLVE_FIXTURE_DIR = FIXTURE_DIR.parent / "mpas_seaice_bgc_zsal_solve"
#: the inert vertical salinity through the ITD, the ridging and the steps (BGC_ZSAL_CARRY_CASES)
BGC_ZSAL_CARRY_FIXTURE_DIR = FIXTURE_DIR.parent / "mpas_seaice_bgc_zsal_carry"
#: the same under zsal_solve (BGC_ZSAL_SOLVE_CARRY_CASES), from the patched oracle
BGC_ZSAL_SOLVE_CARRY_FIXTURE_DIR = FIXTURE_DIR.parent / "mpas_seaice_bgc_zsal_solve_carry"
#: the vertical tracers' nonreactive tracer in the init and column_biogeochemistry
BGC_NONREACTIVE_FIXTURE_DIR = FIXTURE_DIR.parent / "mpas_seaice_bgc_nonreactive"
BGC_NONREACTIVE_CARRY_FIXTURE_DIR = FIXTURE_DIR.parent / "mpas_seaice_bgc_nonreactive_carry"
PRECISIONS = {"r8": np.float64}
REAL, INT, DIM = _stream.REAL, _stream.INT, _stream.DIM
#: the Registry structs the column routines read and write, as oracle_stream pools
POOLS = ("mesh", "tracers", "tracers_aggregate", "icestate", "boundary", "velocity_solver", "atmos_coupling",
         "atmos_forcing", "ocean_coupling", "ridging", "melt_growth_rates",
         "atmos_fluxes", "ocean_fluxes", "ocean_atmosphere", "shortwave", "drag", "ponds", "aerosols",
         "biogeochemistry", "initial", "diagnostics")
#: and the vertical thermodynamics' coupled surface fluxes (a pool only its cases carry,
#: so the other cases' bytes are unchanged)
VT_POOLS = POOLS + ("alternative_atmos_forcing",)
#: and the variational principal stresses seaice_column_reinitialize_diagnostics_dynamics zeroes
STEP_POOLS = VT_POOLS + ("velocity_variational",)
#: the Registry's name_in_code, the name the routines look a field up by, where it is not the var's name
NAME_IN_CODE = {("velocity_variational", f"{base}{tail}var"): f"{base}{tail}"
                for base in ("strain", "stress") for tail in ("11", "22", "12")}
NAME_IN_CODE.update({("velocity_variational", f"{name}Var"): name
                     for name in ("principalStress1", "principalStress2", "replacementPressure")})
#: the mesh block: one block's cells, owned and halo
MESH = {"nCells": 12, "nCellsSolve": 9, "nEdges": 30, "nEdgesSolve": 24, "nVertices": 20, "nVerticesSolve": 16,
        "maxEdges": 6, "maxEdges2": 12, "vertexDegree": 3}
_EXTENTS = {"nCells": "C1", "nEdges": "E1", "nVertices": "V1"}
BASE_CONFIG = {"config_dt": 3600.0, "oracle_clock_step": "0000_01:00:00", "oracle_update_type": "thermodynamics",
               "oracle_ice_age_time_offset": 0.0}
_BL99 = {"config_thermodynamics_type": "BL99", "config_sea_freezing_temperature_type": "linear_salt"}
_NO_PONDS = {"config_use_level_meltponds": False}
#: (name, seed, stage, config overrides)
CASES = (
    ("init_default", 20261901, "init", {}),
    ("init_bl99", 20261902, "init", _BL99),
    ("init_bl99_fixed_surface_temperature", 20261903, "init", {**_BL99,
                                                               "config_calc_surface_temperature": False}),
    ("init_zero_layer", 20261904, "init", {"config_thermodynamics_type": "zero layer",
                                           "config_sea_freezing_temperature_type": "linear_salt"}),
    ("init_bounds_new", 20261905, "init", {"config_category_bounds_type": "new"}),
    ("init_bounds_wmo6", 20261906, "init", {"config_category_bounds_type": "WMO", "config_nCategories": 6}),
    ("init_bounds_asymptotic", 20261907, "init", {"config_category_bounds_type": "asymptotic",
                                                  "config_nCategories": 4}),
    ("init_original_delta_function", 20261908, "init", {"config_itd_conversion_type": "delta function",
                                                        "config_nCategories": 3}),
    ("init_single_category", 20261909, "init", {"config_category_bounds_type": "single category",
                                                "config_itd_conversion_type": "delta function",
                                                "config_nCategories": 1}),
    ("init_bl99_three_layers", 20261910, "init", {**_BL99, "config_nIceLayers": 3}),
    ("init_bl99_sixteen_layers", 20261911, "init", {**_BL99, "config_nIceLayers": 16}),
    ("init_uniform", 20261912, "init", {"config_initial_condition_type": "uniform",
                                        "config_initial_velocity_type": "none", "config_initial_ice_area": 0.25,
                                        "config_initial_ice_volume": 0.5, "config_initial_snow_volume": 0.1}),
    ("init_no_ice_topo_ponds", 20261913, "init", {"config_initial_condition_type": "no_ice",
                                                  "config_use_level_meltponds": False,
                                                  "config_use_topo_meltponds": True}),
    ("init_cesm_ponds_aerosols", 20261914, "init", {"config_use_level_meltponds": False,
                                                    "config_use_cesm_meltponds": True,
                                                    "config_pond_refreezing_type": "cesm",
                                                    "config_use_aerosols": True}),
    ("aggregate_default", 20261921, "aggregate", {}),
    ("aggregate_cesm_ponds", 20261922, "aggregate", {"config_use_level_meltponds": False,
                                                     "config_use_cesm_meltponds": True,
                                                     "config_pond_refreezing_type": "cesm"}),
    ("aggregate_topo_ponds", 20261923, "aggregate", {"config_use_level_meltponds": False,
                                                     "config_use_topo_meltponds": True}),
    ("aggregate_bare", 20261924, "aggregate", {**_NO_PONDS, "config_use_ice_age": False,
                                               "config_use_first_year_ice": False, "config_use_level_ice": False,
                                               "config_nCategories": 3, "config_nIceLayers": 2}),
    ("update_state_thermodynamics", 20261931, "update_state", {}),
    ("update_state_transport_age_offset", 20261932, "update_state", {
        "oracle_update_type": "transport", "oracle_ice_age_time_offset": 1800.0}),
    ("update_state_no_age", 20261933, "update_state", {"config_use_ice_age": False}),
)
_VT = {**_BL99, **_NO_PONDS}
#: the vertical thermodynamics (colpkg stage 2): name, seed, stage, overrides
VT_CASES = (
    ("vt_bl99_bubbly", 20261941, "vertical_thermodynamics", _VT),
    ("vt_bl99_mu71_highfreq", 20261942, "vertical_thermodynamics", {
        **_VT, "config_heat_conductivity_type": "MU71", "config_use_high_frequency_coupling": True, "config_boundary_layer_iteration_number": 3,
        "config_nIceLayers": 4, "config_nCategories": 3}),
    ("vt_bl99_coupled_surface_temperature", 20261943, "vertical_thermodynamics", {
        **_VT, "config_calc_surface_temperature": False}),
    ("vt_bl99_melting", 20261944, "vertical_thermodynamics", {**_VT, "config_start_time": "2000-07-10_00:00:00"}),
    ("vt_bl99_first_year_north", 20261945, "vertical_thermodynamics", {
        **_VT, "config_start_time": "2000-09-15_00:00:00"}),
    ("vt_bl99_first_year_south", 20261946, "vertical_thermodynamics", {
        **_VT, "config_start_time": "2001-03-16_00:00:00", "config_nIceLayers": 9}),
    ("vt_bl99_one_layer_bare", 20261947, "vertical_thermodynamics", {
        **_VT, "config_nIceLayers": 1, "config_nCategories": 1, "config_category_bounds_type": "single category",
        "config_itd_conversion_type": "delta function", "config_use_ice_age": False,
        "config_use_first_year_ice": False, "config_use_level_ice": False}),
    ("vt_bl99_abort_salinity", 20261948, "vertical_thermodynamics", _VT),
    # a leap year after February on the default gregorian_noleap calendar: 2000-09-16 is
    # day 259 of the 365-day year (the north's reset), 260 on the Gregorian count
    ("vt_bl99_first_year_north_leap_year", 20261949, "vertical_thermodynamics", {
        **_VT, "config_start_time": "2000-09-16_00:00:00"}),
    ("vt_bl99_first_year_north_gregorian", 20261950, "vertical_thermodynamics", {
        **_VT, "config_calendar_type": "gregorian", "config_start_time": "2000-09-15_00:00:00"}),
)
_ITD = "itd_thermodynamics"
_DELTA_NEW = {"config_itd_conversion_type": "delta function", "config_category_bounds_type": "new"}
#: the ITD thermodynamics (colpkg stage 3): name, seed, stage, overrides; the Registry
#: default is mushy with level ponds and config_update_ocean_fluxes = .false.
ITD_CASES = (
    ("itd_mushy_level_ponds", 20261951, _ITD, {}),
    ("itd_mushy_three_layers_update_ocean", 20261952, _ITD, {"config_nIceLayers": 3,
                                                             "config_update_ocean_fluxes": True}),
    ("itd_bl99_update_ocean", 20261953, _ITD, {**_BL99, "config_update_ocean_fluxes": True}),
    ("itd_bl99_cesm_ponds", 20261954, _ITD, {**_BL99, "config_use_level_meltponds": False,
                                             "config_use_cesm_meltponds": True,
                                             "config_pond_refreezing_type": "cesm"}),
    ("itd_mushy_topo_ponds_bare", 20261955, _ITD, {"config_use_level_meltponds": False,
                                                   "config_use_topo_meltponds": True,
                                                   "config_use_ice_age": False,
                                                   "config_use_first_year_ice": False}),
    ("itd_bl99_delta_function", 20261956, _ITD, {**_BL99, **_DELTA_NEW, "config_nCategories": 4,
                                                 "config_use_level_meltponds": False,
                                                 "config_use_level_ice": False}),
    ("itd_bl99_single_category", 20261957, _ITD, {**_BL99, "config_nCategories": 1,
                                                  "config_category_bounds_type": "single category",
                                                  "config_itd_conversion_type": "delta function"}),
    ("itd_abort_shift_ice", 20261958, _ITD, {}),
    ("itd_abort_area_bounds", 20261959, _ITD, {**_BL99}),
    ("itd_delta_abort_zap_negative", 20261960, _ITD, {**_DELTA_NEW}),
)
_RDG = "ridging"
_THORNDIKE, _HIBLER = ({"config_ridging_participation_function": "Thorndike75"},
                       {"config_ridging_redistribution_function": "Hibler80"})
_TOPO = {"config_use_level_meltponds": False, "config_use_topo_meltponds": True}
#: the ridging (colpkg stage 4): name, seed, stage, overrides; the Registry default is
#: the exponential participation and redistribution, e-folding scale 3, one dynamics subcycle
RIDGE_CASES = (
    ("ridge_mushy_level_ponds", 20261971, _RDG, {}),
    ("ridge_bl99_thorndike_hibler", 20261972, _RDG, {**_BL99, **_THORNDIKE, **_HIBLER}),
    ("ridge_thorndike_exponential_cesm_ponds", 20261973, _RDG, {
        **_THORNDIKE, "config_use_level_meltponds": False, "config_use_cesm_meltponds": True,
        "config_pond_refreezing_type": "cesm"}),
    ("ridge_exponential_hibler_topo_ponds", 20261974, _RDG, {**_BL99, **_HIBLER, **_TOPO}),
    ("ridge_subcycles_topo_ponds", 20261975, _RDG, {**_TOPO, "config_dynamics_subcycle_number": 3,
                                                    "config_ridiging_efolding_scale": 4.5,
                                                    "config_nIceLayers": 3}),
    ("ridge_bl99_delta_function_bare", 20261976, _RDG, {**_BL99, **_DELTA_NEW, "config_nCategories": 4,
                                                        "config_use_level_meltponds": False,
                                                        "config_use_level_ice": False,
                                                        "config_use_ice_age": False}),
    ("ridge_bl99_single_category", 20261977, _RDG, {**_BL99, "config_nCategories": 1,
                                                    "config_category_bounds_type": "single category",
                                                    "config_itd_conversion_type": "delta function"}),
    ("ridge_negative_opening_dies", 20261978, _RDG, {}),
    ("ridge_abort_iterations", 20261979, _RDG, {**_HIBLER}),
    ("ridge_abort_cleanup", 20261980, _RDG, {**_BL99}),
)
_CCSM3 = {"config_shortwave_type": "ccsm3", **_NO_PONDS}
_CONSTANT = {**_CCSM3, "config_albedo_type": "constant"}
_COUPLED = {"config_calc_surface_temperature": False}
#: the shortwave without the delta-Eddington solution (colpkg stage 5a): name, seed,
#: stage, overrides. The Registry default is the delta-Eddington shortwave with the
#: ccsm3 albedos; MPAS refuses ponds with the CCSM3 shortwave and a computed surface
#: temperature. ccsm3 albedos need the opt-in atan (RADIATION_TRIG_CASES).
RADIATION_CASES = (
    ("rad_ccsm3_constant_albedos", 20261981, "radiation", {**_CONSTANT, **_BL99}),
    ("rad_ccsm3_constant_mushy_five_layers", 20261982, "radiation", {**_CONSTANT, "config_nIceLayers": 5}),
    ("rad_ccsm3_constant_zero_layer", 20261983, "radiation", {
        **_CONSTANT, "config_thermodynamics_type": "zero layer",
        "config_sea_freezing_temperature_type": "linear_salt", "config_nCategories": 3}),
    ("rad_ccsm3_albedos", 20261984, "radiation", {**_CCSM3, **_BL99}),
    ("rad_ccsm3_albedos_thickness_limit", 20261985, "radiation", {
        **_CCSM3, "config_variable_albedo_thickness_limit": 0.7, "config_visible_ice_albedo": 0.75,
        "config_infrared_snow_albedo": 0.68, "config_nCategories": 2, "config_category_bounds_type": "new"}),
    ("rad_coupled_dedd_topo_ponds", 20261986, "radiation", {**_COUPLED, **_BL99, **_TOPO}),
    ("rad_coupled_ccsm3_level_ponds", 20261987, "radiation", {**_COUPLED, **_BL99, "config_shortwave_type": "ccsm3"}),
    ("prep_radiation_default", 20261988, "prep_radiation", {}),
    ("prep_radiation_three_categories", 20261989, "prep_radiation", {"config_nCategories": 3,
                                                                     "config_nIceLayers": 4}),
    ("init_shortwave_ccsm3_constant", 20261990, "init_shortwave", {**_CONSTANT}),
    ("init_shortwave_ccsm3_albedos", 20261991, "init_shortwave", {**_CCSM3, **_BL99}),
    ("init_shortwave_coupled_topo_ponds", 20261992, "init_shortwave", {**_COUPLED, **_BL99, **_TOPO}),
)
#: the cases that call glibc's atan (the ccsm3 albedos with a computed surface temperature)
RADIATION_TRIG_CASES = ("rad_ccsm3_albedos", "rad_ccsm3_albedos_thickness_limit", "init_shortwave_ccsm3_albedos")
_RADIATION_STAGES = ("radiation", "prep_radiation", "init_shortwave")
_CESM = {"config_use_level_meltponds": False, "config_use_cesm_meltponds": True, "config_pond_refreezing_type": "cesm"}
_TOPO_ONLY = {"config_use_level_meltponds": False, "config_use_topo_meltponds": True}
#: the delta-Eddington shortwave with a computed surface temperature (colpkg stage
#: 5b; the Registry default): name, seed, stage, overrides. Every case calls glibc's
#: sin, cos, atan and asin (the orbit and the solar zenith angle).
DEDD_CASES = (
    ("dedd_default_level_ponds", 20261993, "radiation", {}),
    ("dedd_bl99_level_ponds_noon", 20261994, "radiation", {
        **_BL99, "config_calendar_type": "gregorian", "config_start_time": "2000-06-21_12:00:00"}),
    ("dedd_cesm_ponds_snow_transition", 20261995, "radiation", {
        **_CESM, "config_snow_to_ice_transition_depth": 0.03, "config_start_time": "2001-07-15_06:30:15"}),
    ("dedd_topo_ponds", 20261996, "radiation", {**_BL99, **_TOPO_ONLY, "config_start_time": "2001-12-21_23:59:59"}),
    ("dedd_no_ponds_snow_transition", 20261997, "radiation", {
        **_NO_PONDS, "config_snow_to_ice_transition_depth": 0.05, "config_start_time": "2001-03-20_18:00:00"}),
    ("dedd_negative_tuning", 20261998, "radiation", {
        "config_ice_shortwave_tuning_parameter": -1.5, "config_pond_shortwave_tuning_parameter": -2.5,
        "config_snow_shortwave_tuning_parameter": -1.0, "config_start_time": "2001-05-01_09:00:00"}),
    ("dedd_positive_tuning_no_algae", 20261999, "radiation", {
        **_CESM, "config_ice_shortwave_tuning_parameter": 1.0, "config_pond_shortwave_tuning_parameter": 1.5,
        "config_snow_shortwave_tuning_parameter": 2.0, "config_algae_absorption_coefficient": 0.0,
        "config_temp_change_snow_grain_radius_change": 3.0, "config_max_melting_snow_grain_radius": 900.0,
        "config_start_time": "2001-08-10_15:45:00"}),
    ("dedd_bl99_one_layer_three_categories", 20262000, "radiation", {
        **_BL99, "config_nIceLayers": 1, "config_nCategories": 3, "config_start_time": "2001-06-01_03:00:00"}),
    ("dedd_bl99_twelve_layers", 20262001, "radiation", {
        **_BL99, "config_nIceLayers": 12, "config_start_time": "2001-04-15_21:00:00"}),
    ("dedd_zero_layer", 20262002, "radiation", {
        "config_thermodynamics_type": "zero layer", "config_sea_freezing_temperature_type": "linear_salt",
        "config_start_time": "2001-06-21_00:00:00"}),
    ("dedd_zero_layer_topo_ponds", 20262003, "radiation", {
        **_TOPO_ONLY, "config_thermodynamics_type": "zero layer", "config_sea_freezing_temperature_type": "linear_salt",
        "config_start_time": "2000-01-10_12:00:00"}),
    # a leap year after February on the default gregorian_noleap calendar: 2000-03-01 is
    # day 60 of the 365-day year, 61 on the Gregorian count
    ("dedd_default_noleap_leap_year", 20262006, "radiation", {"config_start_time": "2000-03-01_12:00:00"}),
    ("init_shortwave_dedd_level_ponds", 20262004, "init_shortwave", {"config_start_time": "2001-06-21_12:00:00"}),
    ("init_shortwave_dedd_topo_ponds", 20262005, "init_shortwave", {
        **_BL99, **_TOPO_ONLY, "config_calendar_type": "gregorian", "config_start_time": "2000-02-29_07:00:00"}),
)
_STEP = "column_step"
#: the column step runs coupled to an ocean (the mixed layer is not ported)
_STEP_BASE = {**_VT, "config_use_ocean_mixed_layer": False}
#: the column package's part of seaice_timestep over several steps: name, seed, stage,
#: overrides. The vertical thermodynamics are BL99 without ponds; every case with
#: the computed surface temperature and the delta-Eddington shortwave calls glibc's
#: atan, sin, cos and asin.
STEP_CASES = (
    ("step_bl99_dedd", 20262011, _STEP, {**_STEP_BASE, "oracle_n_steps": 3,
                                         "config_start_time": "2001-06-01_10:00:00"}),
    ("step_bl99_dedd_subcycles_gregorian", 20262012, _STEP, {
        **_STEP_BASE, "oracle_n_steps": 3, "config_dynamics_subcycle_number": 2, "config_nIceLayers": 4,
        "config_nCategories": 3, "config_calendar_type": "gregorian", "config_start_time": "2000-07-10_06:00:00"}),
    ("step_bl99_ccsm3_constant_winter", 20262013, _STEP, {
        **_STEP_BASE, **_CONSTANT, "oracle_n_steps": 4, "config_start_time": "2001-01-15_00:00:00"}),
    ("step_bl99_coupled_surface_temperature", 20262014, _STEP, {**_STEP_BASE, **_COUPLED, "oracle_n_steps": 3}),
    ("step_mushy_level_ponds_no_vertical_thermodynamics", 20262015, _STEP, {
        "config_use_ocean_mixed_layer": False, "config_use_column_vertical_thermodynamics": False,
        "oracle_n_steps": 2, "config_start_time": "2001-07-20_12:00:00"}),
    ("step_bl99_dedd_melting", 20262016, _STEP, {**_STEP_BASE, "oracle_n_steps": 2,
                                                 "config_start_time": "2001-07-10_12:00:00"}),
    ("step_bl99_abort_salinity", 20262017, _STEP, {**_STEP_BASE, "oracle_n_steps": 2}),
)

_SEAICE_STEP = "seaice_step"
#: run_gpu's output for a ``seaice_step`` case: the step it stopped at (the oracle's ``TOBJ STEP``)
SEAICE_STEP_KEY = "oracle/step"
#: the velocity solver's settings of the time-step oracle's cases (mpas_seaice_step_oracle)
_DYNAMICS = {
    "config_use_velocity_solver": True, "config_stress_divergence_scheme": "variational",
    "config_revised_evp": False, "config_use_air_stress": True, "config_use_ocean_stress": True,
    "config_use_surface_tilt": True, "config_geostrophic_surface_tilt": True, "config_use_advection": False,
    "config_elastic_subcycle_number": 30, "config_use_forcing": False, "config_do_restart": False,
    "config_use_halo_exch": False, "config_perform_unit_test": False, "config_check_state": False,
    "config_use_column_package": True, "oracle_finalize_last": True,
}
_COUPLED_BASE = {**_STEP_BASE, **_DYNAMICS, "config_dt": 1800.0, "oracle_clock_step": "0000_00:30:00"}
#: the time step with the column package and the velocity solver: name, seed, stage,
#: overrides, (nq, nr, owned columns) of the culled hexagonal torus
COUPLED_CASES = (
    ("seaice_bl99_dedd_rothrock", 20262101, _SEAICE_STEP, {
        **_COUPLED_BASE, "oracle_n_steps": 3, "config_start_time": "2001-06-01_10:00:00"}, (12, 6, 3)),
    ("seaice_bl99_dedd_two_subcycles_exponential_gregorian", 20262102, _SEAICE_STEP, {
        **_COUPLED_BASE, "oracle_n_steps": 2, "config_dynamics_subcycle_number": 2,
        "config_elastic_subcycle_number": 12, "config_revised_evp": True,
        "config_ridging_participation_function": "exponential",
        "config_ridging_redistribution_function": "exponential", "config_nCategories": 3, "config_nIceLayers": 4,
        "config_calendar_type": "gregorian", "config_start_time": "2000-07-10_06:00:00"}, (12, 6, 3)),
    ("seaice_bl99_hibler_strength_ccsm3_winter", 20262103, _SEAICE_STEP, {
        **_COUPLED_BASE, **_CONSTANT, "oracle_n_steps": 3, "config_ice_strength_formulation": "Hibler79",
        "config_use_surface_tilt": False, "config_start_time": "2001-01-15_00:00:00"}, (10, 5, 3)),
    ("seaice_mushy_level_ponds_no_vertical_thermodynamics", 20262104, _SEAICE_STEP, {
        **{k: v for k, v in _COUPLED_BASE.items() if k not in _VT}, "config_use_column_vertical_thermodynamics": False,
        "oracle_n_steps": 2, "config_start_time": "2001-07-20_12:00:00"}, (10, 5, 3)),
    ("seaice_bl99_dedd_last_step_unfinalized", 20262105, _SEAICE_STEP, {
        **_COUPLED_BASE, "oracle_n_steps": 2, "oracle_finalize_last": False,
        "config_start_time": "2001-07-10_12:00:00"}, (10, 5, 3)),
    ("seaice_bl99_abort_salinity", 20262106, _SEAICE_STEP, {**_COUPLED_BASE, "oracle_n_steps": 2},
     (10, 5, 3)),
)
_OML = "ocean_mixed_layer"
_OML_BASE = {**_VT, "config_use_ocean_mixed_layer": True}
#: seaice_column_ocean_mixed_layer alone: name, seed, stage, overrides
OML_CASES = (
    ("oml_registry_defaults", 20262201, _OML, _OML_BASE),
    ("oml_no_surface_stresses", 20262202, _OML, {**_OML_BASE, "config_calc_surface_stresses": False}),
    ("oml_high_frequency_one_iteration", 20262203, _OML, {
        **_OML_BASE, "config_use_high_frequency_coupling": True, "config_boundary_layer_iteration_number": 1}),
    ("oml_test_ice_shelf_ten_iterations", 20262204, _OML, {
        **_OML_BASE, "config_use_test_ice_shelf": True, "config_boundary_layer_iteration_number": 10}),
)
#: the column step with the mixed layer on (seaice_column_coupling_prep runs it first)
OML_STEP_CASES = (
    ("oml_step_bl99_dedd", 20262211, _STEP, {**_OML_BASE, "oracle_n_steps": 3,
                                             "config_start_time": "2001-06-01_10:00:00"}),
    ("oml_step_registry_defaults_no_vertical_thermodynamics", 20262212, _STEP, {
        "config_use_column_vertical_thermodynamics": False, "oracle_n_steps": 2,
        "config_start_time": "2001-07-20_12:00:00"}),
)

_BGC_STAGE = "biogeochemistry"
#: the column biogeochemistry with the brine height
_BGC = {"config_use_column_biogeochemistry": True, "config_use_brine": True}
#: v8.4.2 cannot step it without (column_step_v842.check_config)
_BGC_INACTIVE = "bgc_inactive_diagnostics"
_BGC_STEP = {**_BGC, "woof_v842_corrections": _BGC_INACTIVE}
#: distinct carbon to nitrogen ratios, so colpkg_init_zbgc's R_C2N(1) shows which it took
_C2N = {"config_ratio_C_to_N_diatoms": 6.5, "config_ratio_C_to_N_small_plankton": 7.25,
        "config_ratio_C_to_N_phaeocystis": 7.75, "config_ratio_C_to_N_proteins": 5.5}
#: the column biogeochemistry (the init and column_biogeochemistry's brine height): name, seed, stage, overrides
BGC_CASES = (
    ("init_bgc_brine", 20262201, "init", _BGC),
    ("init_bgc_brine_bl99_c2n", 20262202, "init", {**_BGC, **_BL99, **_C2N}),
    ("init_bgc_brine_restart_hbrine", 20262203, "init", {**_BGC, "config_do_restart_hbrine": True}),
    ("init_bgc_no_brine", 20262204, "init", {"config_use_column_biogeochemistry": True}),
    ("init_bgc_brine_three_layers_cesm_ponds", 20262205, "init", {
        **_BGC, "config_nIceLayers": 3, "config_use_level_meltponds": False, "config_use_cesm_meltponds": True,
        "config_pond_refreezing_type": "cesm"}),
    ("bgc_brine", 20262211, _BGC_STAGE, _BGC),
    ("bgc_brine_bl99_half_hour", 20262212, _BGC_STAGE, {**_BGC, **_BL99, "config_dt": 1800.0}),
    ("bgc_brine_short_drainage_length", 20262213, _BGC_STAGE, {
        **_BGC, "config_bio_gravity_drainage_length_scale": 0.5}),
    ("bgc_brine_four_layers_three_categories", 20262214, _BGC_STAGE, {
        **_BGC, "config_nIceLayers": 4, "config_nCategories": 3}),
    ("bgc_brine_abort_remap", 20262215, _BGC_STAGE, _BGC),
    ("bgc_brine_remap_stop_cleared", 20262216, _BGC_STAGE, _BGC),
)


#: the brine fraction carried through the ITD thermodynamics and the ridging: name, seed,
#: stage, overrides (each an ITD or ridging case's, with the brine height on)
BGC_CARRY_CASES = (
    ("itd_bgc_brine_mushy_level_ponds", 20262301, _ITD, _BGC),
    ("itd_bgc_brine_bl99_update_ocean", 20262302, _ITD, {**_BGC, **_BL99, "config_update_ocean_fluxes": True}),
    ("itd_bgc_brine_bl99_delta_function", 20262303, _ITD, {
        **_BGC, **_BL99, **_DELTA_NEW, "config_nCategories": 4, "config_use_level_meltponds": False,
        "config_use_level_ice": False}),
    ("itd_bgc_brine_bl99_single_category", 20262304, _ITD, {
        **_BGC, **_BL99, "config_nCategories": 1, "config_category_bounds_type": "single category",
        "config_itd_conversion_type": "delta function"}),
    ("itd_bgc_brine_three_layers_topo_ponds", 20262305, _ITD, {
        **_BGC, "config_nIceLayers": 3, "config_use_level_meltponds": False, "config_use_topo_meltponds": True}),
    ("ridge_bgc_brine_mushy_level_ponds", 20262311, _RDG, _BGC),
    ("ridge_bgc_brine_bl99_thorndike_hibler", 20262312, _RDG, {**_BGC, **_BL99, **_THORNDIKE, **_HIBLER}),
    ("ridge_bgc_brine_subcycles_topo_ponds", 20262313, _RDG, {
        **_BGC, **_TOPO, "config_dynamics_subcycle_number": 3, "config_ridiging_efolding_scale": 4.5}),
    ("step_bgc_brine_bl99_dedd", 20262321, _STEP, {**_STEP_BASE, **_BGC_STEP, "oracle_n_steps": 3,
                                                   "config_start_time": "2001-06-01_10:00:00"}),
    ("step_bgc_brine_bl99_dedd_subcycles_gregorian", 20262322, _STEP, {
        **_STEP_BASE, **_BGC_STEP, "oracle_n_steps": 3, "config_dynamics_subcycle_number": 2,
        "config_nIceLayers": 4,
        "config_nCategories": 3, "config_calendar_type": "gregorian", "config_start_time": "2000-07-10_06:00:00"}),
    ("step_bgc_brine_mushy_level_ponds_no_vertical_thermodynamics", 20262323, _STEP, {
        **_BGC_STEP, "config_use_ocean_mixed_layer": False, "config_use_column_vertical_thermodynamics": False,
        "oracle_n_steps": 3, "config_start_time": "2001-07-20_12:00:00"}),
)
#: the brine fraction through the time step with the column package and the velocity solver
BGC_CARRY_COUPLED_CASES = (
    ("seaice_bgc_brine_bl99_dedd_rothrock", 20262331, _SEAICE_STEP, {
        **_COUPLED_BASE, **_BGC_STEP, "oracle_n_steps": 3, "config_start_time": "2001-06-01_10:00:00"},
     (12, 6, 3)),
    ("seaice_bgc_brine_mushy_level_ponds_no_vertical_thermodynamics", 20262332, _SEAICE_STEP, {
        **{k: v for k, v in _COUPLED_BASE.items() if k not in _VT}, **_BGC_STEP,
        "config_use_column_vertical_thermodynamics": False, "oracle_n_steps": 2,
        "config_start_time": "2001-07-20_12:00:00"}, (10, 5, 3)),
)


#: the column biogeochemistry with the brine height and the vertical salinity (BL99 only)
_BGC_ZSAL = {**_BGC, **_BL99, "config_use_vertical_zsalinity": True}
#: the vertical salinity in the init, column_biogeochemistry and the vertical
#: thermodynamics, as v8.4.2 runs it (inert: colpkg's solve_zsal is never set and
#: the pool's vertical salinity is copied only with the vertical tracers):
#: name, seed, stage, overrides
BGC_ZSAL_CASES = (
    ("init_bgc_zsal", 20262501, "init", _BGC_ZSAL),
    ("init_bgc_zsal_three_layers_four_categories", 20262503, "init", {
        **_BGC_ZSAL, "config_nIceLayers": 3, "config_nCategories": 4}),
    ("bgc_zsal", 20262511, _BGC_STAGE, _BGC_ZSAL),
    ("bgc_zsal_four_layers_three_categories", 20262514, _BGC_STAGE, {
        **_BGC_ZSAL, "config_nIceLayers": 4, "config_nCategories": 3}),
    ("bgc_zsal_abort_remap", 20262515, _BGC_STAGE, _BGC_ZSAL),
    ("vt_bgc_zsal_bl99", 20262521, "vertical_thermodynamics", {**_VT, **_BGC_ZSAL}),
)
#: the opt-in correction that solves the vertical salinity (column_v842.CORRECTIONS)
ZSAL_SOLVE = "zsal_solve"
_ZSAL_SOLVE = {**_BGC_ZSAL, "woof_v842_corrections": ZSAL_SOLVE}
#: the same under ``zsal_solve``, against the oracle built with
#: tools/mpas_seaice_oracle/zsal_solve.patch (WOOF_ORACLE_PATCH=zsal_solve)
BGC_ZSAL_SOLVE_CASES = (
    ("init_zsal_solve", 20262531, "init", _ZSAL_SOLVE),
    ("init_zsal_solve_restart_zsalinity", 20262532, "init", {**_ZSAL_SOLVE, "config_do_restart_zsalinity": True}),
    ("init_zsal_solve_three_layers_four_categories", 20262533, "init", {
        **_ZSAL_SOLVE, "config_nIceLayers": 3, "config_nCategories": 4}),
    ("bgc_zsal_solve", 20262541, _BGC_STAGE, _ZSAL_SOLVE),
    ("bgc_zsal_solve_half_hour_sublayer_drainage_scale", 20262542, _BGC_STAGE, {
        **_ZSAL_SOLVE, "config_dt": 1800.0, "config_zsalinity_molecular_sublayer": 0.01,
        "config_zsalinity_gravity_drainage_scale": 0.05, "config_bio_gravity_drainage_length_scale": 0.5}),
    ("bgc_zsal_solve_short_step", 20262543, _BGC_STAGE, {**_ZSAL_SOLVE, "config_dt": 200.0}),
    ("bgc_zsal_solve_four_layers_three_categories", 20262544, _BGC_STAGE, {
        **_ZSAL_SOLVE, "config_nIceLayers": 4, "config_nCategories": 3}),
    ("bgc_zsal_solve_abort_remap", 20262545, _BGC_STAGE, _ZSAL_SOLVE),
    ("bgc_zsal_solve_abort_max_salinity", 20262546, _BGC_STAGE, _ZSAL_SOLVE),
    ("vt_zsal_solve_bl99", 20262551, "vertical_thermodynamics", {**_VT, **_ZSAL_SOLVE}),
    ("vt_zsal_solve_bl99_melting", 20262552, "vertical_thermodynamics", {
        **_VT, **_ZSAL_SOLVE, "config_start_time": "2000-07-10_00:00:00"}),
)
_ZSAL_STEP = {**_BGC_ZSAL, "woof_v842_corrections": _BGC_INACTIVE}
#: v8.4.2's inert vertical salinity carried through the ITD thermodynamics, the ridging
#: and the steps (its tracers counted, never solved or copied; the remap advects the
#: pool's): name, seed, stage, overrides
BGC_ZSAL_CARRY_CASES = (
    ("itd_bgc_zsal_bl99_update_ocean", 20262601, _ITD, {**_BGC_ZSAL, "config_update_ocean_fluxes": True}),
    ("itd_bgc_zsal_bl99_delta_function_four_layers", 20262602, _ITD, {
        **_BGC_ZSAL, **_DELTA_NEW, "config_nIceLayers": 4, "config_nCategories": 4,
        "config_use_level_meltponds": False, "config_use_level_ice": False}),
    ("ridge_bgc_zsal_bl99", 20262611, _RDG, _BGC_ZSAL),
    ("ridge_bgc_zsal_bl99_thorndike_hibler_subcycles", 20262612, _RDG, {
        **_BGC_ZSAL, **_THORNDIKE, **_HIBLER, "config_dynamics_subcycle_number": 2}),
    ("step_bgc_zsal_bl99_dedd", 20262621, _STEP, {**_STEP_BASE, **_ZSAL_STEP, "oracle_n_steps": 3,
                                                 "config_start_time": "2001-06-01_10:00:00"}),
    ("step_bgc_zsal_bl99_dedd_three_categories_gregorian", 20262622, _STEP, {
        **_STEP_BASE, **_ZSAL_STEP, "oracle_n_steps": 3, "config_nIceLayers": 4, "config_nCategories": 3,
        "config_calendar_type": "gregorian", "config_start_time": "2000-07-10_06:00:00"}),
)
#: the inert vertical salinity through the time step with the velocity solver (no
#: advection, as every column-package time-step case here)
BGC_ZSAL_CARRY_COUPLED_CASES = (
    ("seaice_bgc_zsal_bl99_dedd_rothrock", 20262631, _SEAICE_STEP, {
        **_COUPLED_BASE, **_ZSAL_STEP, "oracle_n_steps": 3, "config_start_time": "2001-06-01_10:00:00"},
     (12, 6, 3)),
)
_ZSAL_SOLVE_STEP = {**_BGC_ZSAL, "woof_v842_corrections": (_BGC_INACTIVE, ZSAL_SOLVE)}
#: the solved vertical salinity (zsal_solve) through the ITD thermodynamics, the ridging
#: and the steps, against the patched oracle: name, seed, stage, overrides
BGC_ZSAL_SOLVE_CARRY_CASES = (
    ("itd_zsal_solve_bl99_update_ocean", 20262701, _ITD, {**_ZSAL_SOLVE, "config_update_ocean_fluxes": True}),
    ("itd_zsal_solve_bl99_delta_function_four_layers", 20262702, _ITD, {
        **_ZSAL_SOLVE, **_DELTA_NEW, "config_nIceLayers": 4, "config_nCategories": 4,
        "config_use_level_meltponds": False, "config_use_level_ice": False}),
    ("itd_zsal_solve_bl99_single_category", 20262703, _ITD, {
        **_ZSAL_SOLVE, "config_nCategories": 1, "config_category_bounds_type": "single category",
        "config_itd_conversion_type": "delta function"}),
    ("itd_zsal_solve_abort_regrid", 20262704, _ITD, _ZSAL_SOLVE),
    ("ridge_zsal_solve_bl99", 20262711, _RDG, _ZSAL_SOLVE),
    ("ridge_zsal_solve_bl99_thorndike_hibler_subcycles", 20262712, _RDG, {
        **_ZSAL_SOLVE, **_THORNDIKE, **_HIBLER, "config_dynamics_subcycle_number": 2}),
    ("step_zsal_solve_bl99_dedd", 20262721, _STEP, {**_STEP_BASE, **_ZSAL_SOLVE_STEP, "oracle_n_steps": 3,
                                                   "config_start_time": "2001-06-01_10:00:00"}),
    ("step_zsal_solve_bl99_dedd_three_categories_gregorian", 20262722, _STEP, {
        **_STEP_BASE, **_ZSAL_SOLVE_STEP, "oracle_n_steps": 3, "config_nIceLayers": 4, "config_nCategories": 3,
        "config_calendar_type": "gregorian", "config_start_time": "2000-07-10_06:00:00"}),
)
BGC_ZSAL_SOLVE_CARRY_COUPLED_CASES = (
    ("seaice_zsal_solve_bl99_dedd_rothrock", 20262731, _SEAICE_STEP, {
        **_COUPLED_BASE, **_ZSAL_SOLVE_STEP, "oracle_n_steps": 3, "config_start_time": "2001-06-01_10:00:00"},
     (12, 6, 3)),
)


#: the column biogeochemistry with the brine height and the vertical tracers' nonreactive tracer
_BGC_NR = {**_BGC, "config_use_vertical_tracers": True, "config_use_nonreactive": True}
#: the nonreactive tracer in the init and column_biogeochemistry, by its transport type
#: (config_mobility_type_nitrate: below 0 not mobile, then the retention and release
#: time bands): name, seed, stage, overrides
BGC_NONREACTIVE_CASES = (
    ("init_bgc_nonreactive", 20262601, "init", _BGC_NR),
    ("init_bgc_nonreactive_bl99_mobile_frazil", 20262602, "init", {
        **_BGC_NR, **_BL99, "config_mobility_type_nitrate": 0.2, "config_fraction_biotracer_in_frazil": 0.6}),
    ("init_bgc_nonreactive_restart_bgc", 20262603, "init", {**_BGC_NR, "config_do_restart_bgc": True}),
    ("init_bgc_nonreactive_three_layers_four_categories", 20262604, "init", {
        **_BGC_NR, "config_nIceLayers": 3, "config_nCategories": 4, "config_new_ice_fraction_biotracer": 0.7}),
    ("bgc_nonreactive", 20262611, _BGC_STAGE, _BGC_NR),
    ("bgc_nonreactive_bl99_retained", 20262612, _BGC_STAGE, {
        **_BGC_NR, **_BL99, "config_mobility_type_nitrate": 0.2}),
    ("bgc_nonreactive_rapid", 20262613, _BGC_STAGE, {
        **_BGC_NR, "config_mobility_type_nitrate": 0.7, "config_rapid_mobile_to_stationary_time": 1800.0}),
    ("bgc_nonreactive_released_half_hour", 20262614, _BGC_STAGE, {
        **_BGC_NR, "config_mobility_type_nitrate": 1.5, "config_dt": 1800.0,
        "config_long_mobile_to_stationary_time": 86400.0}),
    ("bgc_nonreactive_stationary_four_layers_three_categories", 20262615, _BGC_STAGE, {
        **_BGC_NR, "config_mobility_type_nitrate": 2.5, "config_nIceLayers": 4, "config_nCategories": 3,
        "config_biogrid_bottom_molecular_sublayer": 0.01}),
    ("bgc_nonreactive_bl99_retained_two_hours", 20262616, _BGC_STAGE, {
        **_BGC_NR, **_BL99, "config_mobility_type_nitrate": 0.2, "config_dt": 7200.0}),
    ("bgc_nonreactive_abort_init", 20262617, _BGC_STAGE, _BGC_NR),
    ("bgc_nonreactive_abort_negative", 20262618, _BGC_STAGE, _BGC_NR),
)
#: the nonreactive tracer through the ITD thermodynamics and the ridging: name, seed, stage,
#: overrides
BGC_NONREACTIVE_CARRY_CASES = (
    ("itd_nonreactive_mushy_level_ponds", 20262801, _ITD, _BGC_NR),
    ("itd_nonreactive_bl99_update_ocean", 20262802, _ITD, {**_BGC_NR, **_BL99, "config_update_ocean_fluxes": True}),
    ("itd_nonreactive_bl99_delta_function_four_layers", 20262803, _ITD, {
        **_BGC_NR, **_BL99, **_DELTA_NEW, "config_nIceLayers": 4, "config_nCategories": 4,
        "config_use_level_meltponds": False, "config_use_level_ice": False}),
    ("itd_nonreactive_single_category", 20262804, _ITD, {
        **_BGC_NR, "config_nCategories": 1, "config_category_bounds_type": "single category",
        "config_itd_conversion_type": "delta function"}),
    ("itd_nonreactive_mobile_three_layers", 20262805, _ITD, {
        **_BGC_NR, "config_nIceLayers": 3, "config_mobility_type_nitrate": 0.7,
        "config_fraction_biotracer_in_frazil": 0.6}),
    ("itd_nonreactive_abort_regrid", 20262806, _ITD, _BGC_NR),
    ("ridge_nonreactive_mushy_level_ponds", 20262811, _RDG, _BGC_NR),
    ("ridge_nonreactive_bl99_thorndike_hibler_subcycles", 20262812, _RDG, {
        **_BGC_NR, **_BL99, **_THORNDIKE, **_HIBLER, "config_dynamics_subcycle_number": 2}),
)


def configs_of(overrides: dict, stage: str) -> dict:
    return {**cv.default_configs(), **BASE_CONFIG, **overrides, "oracle_stage": stage}


def field_rows(config: dict, mesh: dict = MESH) -> tuple[list, dict[str, int], dict[str, bool]]:
    """The stream's field table, the dimensions and the packages for ``config``
    (on ``mesh``'s dimensions)."""
    dims = cv.dimensions(config, {k: mesh[k] for k in ("nCells", "nEdges", "maxEdges", "maxEdges2", "nVertices",
                                                        "vertexDegree")})
    active = cv.packages(config)
    rows = [("mesh", name, DIM, ()) for name in sorted(dims)]
    rows += [("mesh", name, DIM, ()) for name in ("nCellsSolve", "nEdgesSolve", "nVerticesSolve")]
    rows += [("tracers_aggregate", "nCellsSolve", DIM, ())]
    if config.get("oracle_stage") in (_STEP, _SEAICE_STEP):
        # seaice_column_coupling_prep reads its dimensions from the tracers pool
        rows += [("tracers", "nCellsSolve", DIM, ()), ("tracers", "nCategories", DIM, ())]
    stage = config.get("oracle_stage")
    if stage in (_STEP, _OML) and config.get("config_use_ocean_mixed_layer"):
        # seaice_column_ocean_mixed_layer reads its nCellsSolve from the ocean_coupling pool
        rows += [("ocean_coupling", "nCellsSolve", DIM, ())]
    pools = (VT_POOLS if stage in ("vertical_thermodynamics", _OML) else STEP_POOLS if stage in (_STEP, _SEAICE_STEP)
             else POOLS)
    # The Registry allocates a var by its own packages: gen_inc.c's parse_var never reads
    # the var_struct's (biogeochemistry's newlyFormedIce is the column package's).  The
    # earlier stages' cases keep the struct gate, which only drops fields their routines
    # never read, so their bytes (and fixtures) stay; the ITD stage reads newlyFormedIce.
    # Under the bgc_inactive_diagnostics correction the fields v8.4.2's
    # seaice_column_reinitialize_diagnostics_bgc zeroes without an array get one,
    # as a build whose zeroing asked would leave them: nothing else reads them, so
    # every other word is v8.4.2's (without, v8.4.2 writes to address 0).
    allocated = stand_ins(config)
    var_packages_only = config.get("oracle_stage") in ("itd_thermodynamics", "ridging", *_RADIATION_STAGES, _STEP,
                                                       _SEAICE_STEP)
    for struct in cv.registry_fields()["var_structs"]:
        if struct["name"] not in pools:
            continue
        struct_on = var_packages_only or all(active[f"{p}Active"] for p in struct["packages"])
        for name, kind, dimension_text, package_text in struct["vars"]:
            if kind not in ("real", "integer"):
                continue
            names = [d for d in dimension_text.split() if d != "Time"]
            fshape = tuple(_EXTENTS.get(d, dims.get(d)) for d in names)
            own = [p for p in package_text.split(";") if p]
            on = struct_on and (not own or any(active[f"{p}Active"] for p in own))
            if (struct["name"], name) in allocated:
                on = True
            name = NAME_IN_CODE.get((struct["name"], name), name)
            rows.append((struct["name"], name, REAL if kind == "real" else INT, fshape, on))
    return rows, dims, active


def _uniform(rs):
    """``rs.uniform`` with the scaling done in numpy: RandomState.uniform's C
    ``low + scale * r`` is contracted into an FMA on arm64 builds and not on
    x86-64, so its words differ by host; separate ufuncs round each step."""
    def u(low, high, size=None):
        r = rs.random_sample(size)
        return np.add(np.multiply(np.subtract(high, low), r), low)
    return u


def _values(rs, name: str, shp: tuple, dims: dict, stage: str):
    """Plausible values per field; the rest uniform in [-1, 1)."""
    nc = dims["nCells"]
    u = _uniform(rs)
    if name == "latCell":
        # mostly polar, a temperate cell, and cells exactly on the bounds' degrees
        lat = np.radians(np.where(u(0.0, 1.0, shp) < 0.5, u(70.5, 89.0, shp), u(-89.0, -60.5, shp)))
        lat[:4] = np.radians([75.0, -65.0, 10.0, 70.0])
        return lat
    if name == "seaSurfaceSalinity":
        s = u(28.0, 36.0, shp)
        s[2], s[5] = 130.0, 0.0
        return s
    if name == "seaSurfaceTemperature":
        # mostly at or below freezing, two warm cells
        t = u(-2.2, -1.6, shp)
        t[[4, 8]] = 0.5
        return t
    if name == "airTemperature":
        t = u(240.0, 280.0, shp)
        t[0] = 278.0
        return t
    if name in ("uAirVelocity", "vAirVelocity"):
        return u(-20.0, 20.0, shp)
    if name in ("landIceMask", "landIceMaskVertex"):
        m = np.zeros(shp, np.int32)
        m[min(6, shp[0] - 1)] = 1
        return m
    if name == "interiorVertex":
        m = np.ones(shp, np.int32)
        m[::7] = 0
        return m
    return None


def _tracers(rs, pools: dict, dims: dict):
    """The category state: some ice-free cells and categories, a cell of
    sub-puny area, ice without volume, ice ages and level ice that are zero."""
    nc, ncat = dims["nCells"], dims["nCategories"]
    t = pools["tracers"]
    u = _uniform(rs)
    area = u(0.0, 1.0 / ncat, (nc + 1, ncat, 1))
    area[1] = 0.0
    area[3] = 2.0e-12 / ncat
    area[4, 0] = 0.0
    volume = area * u(0.0, 4.0, area.shape)
    volume[5, -1] = 0.0
    snow = volume * u(0.0, 0.3, area.shape)
    t["iceAreaCategory"], t["iceVolumeCategory"], t["snowVolumeCategory"] = area, volume, snow
    ranges = {"surfaceTemperature": (-30.0, 0.0), "iceEnthalpy": (-3.3e8, -1.0e8),
              "snowEnthalpy": (-1.2e8, -1.0e8), "iceSalinity": (0.0, 10.0), "iceAge": (0.0, 3.0e7),
              "firstYearIceArea": (0.0, 1.0), "levelIceArea": (0.0, 1.0), "levelIceVolume": (0.0, 1.0),
              "pondArea": (0.0, 1.0), "pondDepth": (0.0, 0.5), "pondLidThickness": (0.0, 0.1)}
    for name, (lo, hi) in ranges.items():
        if t.get(name) is not None:
            t[name] = u(lo, hi, t[name].shape)
    if t.get("iceAge") is not None:
        t["iceAge"][7] = 0.0
    if t.get("levelIceArea") is not None:
        t["levelIceArea"][8, :] = 0.0
    if t.get("pondArea") is not None:
        t["pondArea"][6, 1 % ncat] = 0.0


_RHOI, _RHOS, _CP_ICE, _CP_OCN, _LFRESH, _DEPRESST = 917.0, 330.0, 2106.0, 4218.0, 2.835e6 - 2.501e6, 0.054


def _thermo_state(rs, case: dict) -> None:
    """A BL99 column state and forcing the solver converges on: category
    thicknesses 0.2-4 m, snow up to 0.4 m (a snow-free cell and one below
    hs_min), salinities above min_salin, temperatures from the surface's down
    to the freezing point and below each layer's melting point, enthalpies
    consistent with them, an ice-free cell and a sub-puny one, day and night
    shortwave. Every step is a separate numpy ufunc (no reductions), so the
    words are the same on every host."""
    pools, dims, name = case["pools"], case["dims"], case["name"]
    nc, ncat, nil, nsl = dims["nCells"], dims["nCategories"], dims["nIceLayers"], dims["nSnowLayers"]
    n1 = nc + 1
    u = _uniform(rs)
    melting = "melting" in name
    t = pools["tracers"]
    share = u(0.05, 1.0, (n1, ncat))
    total = np.zeros((n1, 1))
    for n in range(ncat):
        total[:, 0] = total[:, 0] + share[:, n]
    area = share / total * u(0.3, 0.99, (n1, 1))
    area[1] = 0.0
    area[3] = 2.0e-12 / ncat
    area[4, 0] = 0.0
    hin = u(0.2, 4.0, (n1, ncat))
    hsn = u(0.0, 0.4, (n1, ncat))
    hsn[2] = 0.0
    hsn[6] = 5.0e-5
    hin[7], hsn[7] = 0.3, 0.35           # snow below the freeboard: snow-ice forms
    tsfc = u(-0.6, 0.0, (n1, ncat)) if melting else u(-30.0, -2.0, (n1, ncat))
    sal = u(0.5, 4.0, (n1, ncat, nil))
    tmlt = -sal * _DEPRESST
    tin = np.empty((n1, ncat, nil))
    top = tsfc * 0.8 if not melting else tsfc - 1.5
    for k in range(nil):
        z = (k + 0.5) / nil
        tin[:, :, k] = top + (-1.85 - top) * z
    tin = np.minimum(tin, tmlt - 0.02)
    zqin = -_RHOI * (_CP_ICE * (tmlt - tin) + _LFRESH * (1.0 - tmlt / tin) - _CP_OCN * tmlt)
    tsn = np.minimum(tsfc * 0.6 + tin[:, :, :1][:, :, 0] * 0.4, -0.05)
    zqsn = np.empty((n1, ncat, nsl))
    for k in range(nsl):
        zqsn[:, :, k] = -_RHOS * (_LFRESH - _CP_ICE * tsn)
    if "abort" in name:
        sal[5, 0, nil - 1] = 0.05          # below min_salin: init_vertical_profile stops
        sal[7, 0, 0] = 0.02                # a later cell, which MPAS never reaches
    t["iceAreaCategory"] = area[:, :, None]
    t["iceVolumeCategory"] = (area * hin)[:, :, None]
    t["snowVolumeCategory"] = (area * hsn)[:, :, None]
    t["surfaceTemperature"] = tsfc[:, :, None]
    t["iceSalinity"], t["iceEnthalpy"], t["snowEnthalpy"] = sal, zqin, zqsn
    if t.get("iceAge") is not None:
        t["iceAge"] = u(0.0, 3.0e7, t["iceAge"].shape)
    if t.get("firstYearIceArea") is not None:
        t["firstYearIceArea"] = u(0.0, 1.0, t["firstYearIceArea"].shape)
    aggregate = np.zeros(n1)
    for n in range(ncat):
        aggregate = aggregate + area[:, n]
    pools["tracers_aggregate"]["iceAreaCell"] = aggregate
    mesh = pools["mesh"]
    mesh["indexToCellID"] = np.arange(1, n1 + 1, dtype=np.int32)
    ac, af = pools["atmos_coupling"], pools["atmos_forcing"]
    ac["airPotentialTemperature"] = u(271.0, 278.0, (n1,)) if melting else u(240.0, 272.0, (n1,))
    ac["airTemperature"] = ac["airPotentialTemperature"] - 0.1
    ac["airSpecificHumidity"] = u(2.0e-4, 3.0e-3, (n1,))
    ac["airDensity"] = u(1.2, 1.4, (n1,))
    ac["airLevelHeight"] = u(5.0, 30.0, (n1,))
    ac["uAirVelocity"], ac["vAirVelocity"] = u(-15.0, 15.0, (n1,)), u(-15.0, 15.0, (n1,))
    ac["uAirVelocity"][0], ac["vAirVelocity"][0] = 0.3, -0.2      # below umin
    af["windSpeed"] = np.sqrt(ac["uAirVelocity"] * ac["uAirVelocity"] + ac["vAirVelocity"] * ac["vAirVelocity"])
    ac["longwaveDown"] = u(150.0, 320.0, (n1,))
    snowfall = u(0.0, 2.0e-5, (n1,))
    snowfall[::3] = 0.0
    ac["snowfallRate"] = snowfall
    sw = pools["shortwave"]
    day = np.ones((n1, ncat))
    if not melting:
        day[::2] = 0.0
    sw["surfaceShortwaveFlux"] = u(0.0, 300.0 if melting else 150.0, (n1, ncat)) * day
    sw["penetratingShortwaveFlux"] = u(0.0, 5.0, (n1, ncat)) * day
    # the interior flux is what the layers absorb (the solver's energy balance)
    layers = u(0.0, 30.0 / nil, (n1, ncat, nil)) * day[:, :, None]
    snow_layers = u(0.0, 10.0, (n1, ncat, nsl)) * day[:, :, None]
    snow_layers[hsn <= 1.0e-4] = 0.0     # no snow layer below hs_min absorbs (l_snow)
    interior = np.zeros((n1, ncat))
    for k in range(nsl):
        interior = interior + snow_layers[:, :, k]
    for k in range(nil):
        interior = interior + layers[:, :, k]
    sw["interiorShortwaveFlux"] = interior
    sw["absorbedShortwaveIceLayer"] = layers
    sw["absorbedShortwaveSnowLayer"] = snow_layers
    oc = pools["ocean_coupling"]
    oc["seaSurfaceSalinity"] = u(30.0, 35.0, (n1,))
    oc["seaFreezingTemperature"] = -_DEPRESST * oc["seaSurfaceSalinity"]
    oc["seaSurfaceTemperature"] = oc["seaFreezingTemperature"] + u(0.0, 1.5, (n1,))
    frzmlt = u(-200.0, 50.0, (n1,))
    frzmlt[8] = 0.0
    oc["freezingMeltingPotential"] = frzmlt
    vs = pools["velocity_solver"]
    vs["oceanStressCellU"], vs["oceanStressCellV"] = u(-0.3, 0.3, (n1,)), u(-0.3, 0.3, (n1,))
    vs["uVelocity"] = u(-0.5, 0.5, vs["uVelocity"].shape)
    pools["drag"]["oceanDragCoefficient"] = u(0.004, 0.007, (n1,))
    dg = pools["diagnostics"]
    for key in ("meltOnset", "freezeOnset"):
        onset = u(0.0, 365.0, (n1,))
        onset[::2] = 0.0
        dg[key] = onset
    alt = pools["alternative_atmos_forcing"]
    alt["latentHeatFluxCouple"] = u(-20.0, 5.0, (n1, ncat))
    alt["sensibleHeatFluxCouple"] = u(-30.0, 30.0, (n1, ncat))
    # colpkg_step_therm1 hands set_sfcflux the two swapped (fsurfn is the coupled
    # conductive flux, fcondtopn the coupled surface flux): fsurfn >= fcondtopn here
    alt["surfaceHeatFluxCouple"] = u(-30.0, 10.0, (n1, ncat))
    alt["surfaceConductiveFluxCouple"] = alt["surfaceHeatFluxCouple"] + u(0.0, 5.0, (n1, ncat))


#: approximate category bounds (hin_max(0:ncat-1), and a finite top) for the case
#: generator only: colpkg_init_itd's own words come from the oracle and the GPU
_APPROX_BOUNDS = {
    ("original", 5): (0.0, 0.6445, 1.3914, 2.4702, 4.5673, 9.3354),
    ("single category", 1): (0.0, 3.0),
}


def _approx_bounds(config: dict, ncat: int) -> tuple[float, ...]:
    kind = config["config_category_bounds_type"]
    if kind == "new":
        # n * (3/ncat + (n-1) * 0.5/ncat), its top widened like the others'
        d1, d2 = 3.0 / ncat, 0.5 / ncat
        b = [0.0] + [n * (d1 + (n - 1) * d2) for n in range(1, ncat + 1)]
        return tuple(b)
    return _APPROX_BOUNDS[(kind, ncat)]


def _itd_state(rs, case: dict) -> None:
    """A thickness distribution before (``*CategoryInitial``) and after a
    thermodynamic step, each category's thickness inside its bounds, and the
    owned cells' branches (row r is indexToCellID r + 1):

    0. a plain cell: lateral melt, ocean melting potential;
    1. open water only, freezing: new ice in category 1;
    2. 0.5 % open water and a large freezing potential: the frazil thicker
       than hi0max spreads over every category (hsurp);
    3. sub-puny categories: zap_small_areas;
    4. snow too warm and too cold in two categories, a third below hs_min:
       zap_snow_temperature (its warnings);
    5. growth past the bounds: linear_itd gives up its remap, rebin moves ice;
    6. category 1 thinned below hi_min (its area removed);
    7. a total area in (1, 1 + puny): zap_small_areas' excess-area branch;
    8. melting: thinner categories and lateral melt.

    The abort cases change one or two cells (the later never reached). Every
    step is a separate numpy ufunc (no reductions), so the words are the same
    on every host."""
    pools, dims, name, config = case["pools"], case["dims"], case["name"], case["config"]
    nc, ncat, nil, nsl = dims["nCells"], dims["nCategories"], dims["nIceLayers"], dims["nSnowLayers"]
    n1 = nc + 1
    u = _uniform(rs)
    bounds = np.array(_approx_bounds(config, ncat))
    lo, hi = bounds[:-1][None, :], bounds[1:][None, :]
    share = u(0.3, 1.0, (n1, ncat))
    total = np.zeros((n1, 1))
    for n in range(ncat):
        total[:, 0] = total[:, 0] + share[:, n]
    area_init = share / total * u(0.5, 0.95, (n1, 1))
    h_init = lo + (hi - lo) * u(0.2, 0.8, (n1, ncat))
    dh = u(-0.12, 0.12, (n1, ncat))
    area_init[1] = 0.0
    area_init[2] = share[2] / total[2] * 0.995
    if config["config_use_brine"] and ncat >= 2:
        # an empty category under the spread frazil: add_new_ice_bgc's brine fraction 1
        area_init[2, 0] = area_init[2, 0] + area_init[2, 1]
        area_init[2, 1] = 0.0
    area_init[3] = 2.0e-12 / ncat
    dh[5] = 0.9
    dh[7] = np.abs(dh[7])
    dh[8] = -0.1
    area = area_init.copy()
    h = h_init + dh
    h[6, 0] = 0.005
    area[7] = 1.0 / ncat
    area[7, -1] = area[7, -1] + 4.0e-12
    area_init[7] = area[7]
    if name == "itd_abort_area_bounds":
        area[0] = share[0] / total[0] * 1.3
    hsn = u(0.0, 0.3, (n1, ncat))
    if ncat >= 3:
        hsn[4, 2] = 5.0e-5
    volume, snow = area * h, area * hsn
    volume_init = area_init * h_init
    if name == "itd_abort_shift_ice":
        # a negative category between empty ones: linear_itd's donor takes all of it
        area_init[5], h_init[5] = 0.0, 0.0
        area_init[5, 0], h_init[5, 0] = 0.5, 0.3
        volume_init[5] = area_init[5] * h_init[5]
        area[5], volume[5], snow[5] = 0.0, 0.0, 0.0
        area[5, 0], volume[5, 0], snow[5, 0] = 0.5, 0.5 * 0.32, 0.5 * 0.1
        area[5, 2] = -1.0e-6
    if name in ("itd_abort_shift_ice", "itd_delta_abort_zap_negative"):
        area[7] = area[7] * 1.3           # out of bounds, in a cell MPAS never reaches
    if name == "itd_delta_abort_zap_negative":
        area[6, 1], volume[6, 1], snow[6, 1] = -1.0e-6, 0.0, 0.0
    t, ist = pools["tracers"], pools["icestate"]
    t["iceAreaCategory"], t["iceVolumeCategory"] = area[:, :, None], volume[:, :, None]
    t["snowVolumeCategory"] = snow[:, :, None]
    ist["iceAreaCategoryInitial"], ist["iceVolumeCategoryInitial"] = area_init, volume_init
    aggregate = np.zeros(n1)
    for n in range(ncat):
        aggregate = aggregate + area[:, n]
    pools["tracers_aggregate"]["iceAreaCell"] = aggregate
    t["surfaceTemperature"] = u(-20.0, -1.0, (n1, ncat, 1))
    tsn = u(-20.0, -1.0, (n1, ncat, nsl))
    tsn[4, 0] = 0.5
    zqsn = -_RHOS * (_LFRESH - _CP_ICE * tsn)
    if ncat >= 2:
        zqsn[4, 1] = -_RHOS * (_LFRESH + _CP_ICE * 150.0)
    sal = u(2.0, 8.0, (n1, ncat, nil))
    tmlt = -sal * _DEPRESST
    tin = u(-15.0, -2.0, (n1, ncat, nil))
    t["iceSalinity"], t["snowEnthalpy"] = sal, zqsn
    t["iceEnthalpy"] = -_RHOI * (_CP_ICE * (tmlt - tin) + _LFRESH * (1.0 - tmlt / tin) - _CP_OCN * tmlt)
    for key, (a, b) in (("iceAge", (0.0, 3.0e7)), ("firstYearIceArea", (0.0, 1.0)),
                        ("levelIceArea", (0.2, 1.0)), ("levelIceVolume", (0.2, 1.0)),
                        ("pondArea", (0.0, 0.5)), ("pondDepth", (0.0, 0.3)),
                        ("pondLidThickness", (0.0, 0.05))):
        if t.get(key) is not None:
            t[key] = u(a, b, t[key].shape)
    pools["mesh"]["indexToCellID"] = np.arange(1, n1 + 1, dtype=np.int32)
    pools["initial"]["initialSalinityProfile"] = u(0.5, 3.5, (n1, nil + 1))
    oc = pools["ocean_coupling"]
    oc["seaSurfaceSalinity"] = u(28.0, 35.0, (n1,))
    oc["seaFreezingTemperature"] = -_DEPRESST * oc["seaSurfaceSalinity"]
    frzmlt = u(-150.0, 400.0, (n1,))
    frzmlt[1], frzmlt[2], frzmlt[3], frzmlt[7] = 300.0, 1000.0, 0.0, -50.0
    oc["freezingMeltingPotential"] = frzmlt
    rside = u(0.0, 0.04, (n1,))
    rside[[2, 3, 7]] = 0.0
    pools["melt_growth_rates"]["lateralIceMeltFraction"] = rside
    pools["atmos_coupling"]["rainfallRate"] = u(0.0, 2.0e-5, (n1,))
    onset = u(0.0, 365.0, (n1,))
    onset[::2] = 0.0
    pools["diagnostics"]["freezeOnset"] = onset


def _brine_fraction(rs, case: dict) -> None:
    """The brine fraction the ITD and ridging cases with the brine height
    carry: 0.6-1.1 (brine above and below the ice top)."""
    pools, dims = case["pools"], case["dims"]
    n1, ncat = dims["nCells"] + 1, dims["nCategories"]
    pools["tracers"]["brineFraction"] = _uniform(rs)(0.6, 1.1, (n1, ncat, 1))


def _brine_step_state(rs, case: dict) -> None:
    """On top of :func:`_step_state`, what the brine height's step reads that
    no earlier stage of the step writes: the brine fraction (:func:`_brine_fraction`),
    the thickness of the last brine height step (``iceThicknessCategoryInitial``,
    near the ice's), the Rayleigh criterion, the melt and growth (which the
    vertical thermodynamics, when on, overwrites) and ``colpkg_init_hbrine``'s
    grids."""
    _brine_fraction(rs, case)
    pools, dims = case["pools"], case["dims"]
    n1, ncat, nil, nbl = dims["nCells"] + 1, dims["nCategories"], dims["nIceLayers"], dims["nBioLayers"]
    u = _uniform(rs)
    t, ist, mg, bgc = pools["tracers"], pools["icestate"], pools["melt_growth_rates"], pools["biogeochemistry"]
    area, volume = t["iceAreaCategory"][:, :, 0], t["iceVolumeCategory"][:, :, 0]
    hin = volume / np.where(area > 0.0, area, 1.0)
    ist["iceThicknessCategoryInitial"] = hin * u(0.9, 1.1, (n1, ncat))
    bgc["rayleighCriteriaReal"] = u(0.0, 1.0, (n1,))
    for key, hi in (("basalIceMeltCategory", 0.01), ("surfaceIceMeltCategory", 0.01),
                    ("congelationCategory", 0.01), ("snowiceFormationCategory", 0.005),
                    ("snowMeltCategory", 0.005)):
        mg[key] = u(0.0, hi, (n1, ncat))
    for key, value in cv.hbrine_grids(nbl, nil).items():
        bgc[key] = value


def _ridge_state(rs, case: dict) -> None:
    """A thickness distribution after transport (``openWaterArea`` and the
    categories summing to ``asum``, not always 1), the ridging forcing
    (``ridgeConvergence``, ``ridgeShear``) and ``dynamicsTimeStep``, with the
    owned cells' branches (row r is indexToCellID r + 1):

    0. a plain cell: convergence and shear, 2 % open water;
    1. open water only;
    2. transport convergence (asum 1.3), little open water and a thin, small
       first category: the closing is cut back and ridge_ice repeats;
    3. divergence (asum 0.85): opening;
    4. a sub-puny category (not ridging; zapped) and snow too warm in another
       (zap_snow_temperature's warning);
    5. thick ice past the bounds in the top category, with snow;
    6. strong transport convergence (asum 1.6) of thin ice, no open water;
    7. no forcing;
    8. ice without volume in category 1.

    The abort cases change one cell (and a later one MPAS never reaches).
    Every step is a separate numpy ufunc (no reductions), so the words are
    the same on every host."""
    pools, dims, name, config = case["pools"], case["dims"], case["name"], case["config"]
    nc, ncat, nil, nsl = dims["nCells"], dims["nCategories"], dims["nIceLayers"], dims["nSnowLayers"]
    n1 = nc + 1
    u = _uniform(rs)
    bounds = np.array(_approx_bounds(config, ncat))
    lo, hi = bounds[:-1][None, :], bounds[1:][None, :]
    h = lo + (hi - lo) * u(0.2, 0.8, (n1, ncat))
    h[6] = lo[0] + (hi[0] - lo[0]) * 0.1
    h[5, -1] = 8.0
    share = u(0.3, 1.0, (n1, ncat))
    total = np.zeros((n1, 1))
    for n in range(ncat):
        total[:, 0] = total[:, 0] + share[:, n]
    ice = u(0.6, 0.95, (n1, 1))
    asum = np.ones((n1, 1))
    asum[2], asum[3], asum[6] = 1.3, 0.85, 1.6
    aice0 = (1.0 - ice[:, 0]) * asum[:, 0]
    aice0[0], aice0[1], aice0[2], aice0[6] = 0.02, 1.0, 1.0e-3, 0.0
    area = share / total * (asum[:, 0] - aice0)[:, None]
    area[2, 0] = 0.01
    if ncat >= 2:
        area[4, 1] = 2.0e-12
    conv = u(0.0, 3.0e-6, (n1,))
    shear = u(0.0, 6.0e-6, (n1,))
    conv[0] = 5.0e-6
    conv[3] = 0.0
    conv[7], shear[7] = 0.0, 0.0
    volume = area * h
    volume[8, 0] = 0.0
    hsn = u(0.0, 0.3, (n1, ncat))
    hsn[5, -1] = 0.3
    snow = area * hsn
    if name == "ridge_negative_opening_dies":
        # an opening rate below zero (a negative convergence) and no open water to give
        for r in (3, 6):
            aice0[r], conv[r] = 0.0, -1.0e-3
    if name == "ridge_abort_iterations":
        # ridges barely thicker than the 10 km ice they come from: twenty iterations
        # cannot bring the area down to 1
        for r in (5, 7):
            area[r], volume[r], snow[r], aice0[r] = 0.0, 0.0, 0.0, 0.0
            area[r, -1], volume[r, -1] = 3.0, 3.0e4
    if name == "ridge_abort_cleanup":
        # a negative category in the cell without forcing (no ridges land on it):
        # zap_small_areas stops
        for r in (7, 8):
            k = 1 % ncat
            area[r, k], volume[r, k], snow[r, k] = -1.0e-6, 0.0, 0.0
    t, ist = pools["tracers"], pools["icestate"]
    t["iceAreaCategory"], t["iceVolumeCategory"] = area[:, :, None], volume[:, :, None]
    t["snowVolumeCategory"] = snow[:, :, None]
    ist["openWaterArea"] = aice0
    aggregate = np.zeros(n1)
    for n in range(ncat):
        aggregate = aggregate + area[:, n]
    pools["tracers_aggregate"]["iceAreaCell"] = aggregate
    t["surfaceTemperature"] = u(-20.0, -1.0, (n1, ncat, 1))
    tsn = u(-20.0, -1.0, (n1, ncat, nsl))
    tsn[4, -1] = 0.5
    t["snowEnthalpy"] = -_RHOS * (_LFRESH - _CP_ICE * tsn)
    sal = u(2.0, 8.0, (n1, ncat, nil))
    tmlt = -sal * _DEPRESST
    tin = u(-15.0, -2.0, (n1, ncat, nil))
    t["iceSalinity"] = sal
    t["iceEnthalpy"] = -_RHOI * (_CP_ICE * (tmlt - tin) + _LFRESH * (1.0 - tmlt / tin) - _CP_OCN * tmlt)
    for key, (a, b) in (("iceAge", (0.0, 3.0e7)), ("firstYearIceArea", (0.0, 1.0)),
                        ("levelIceArea", (0.2, 1.0)), ("levelIceVolume", (0.2, 1.0)),
                        ("pondArea", (0.0, 0.5)), ("pondDepth", (0.0, 0.3)),
                        ("pondLidThickness", (0.0, 0.05))):
        if t.get(key) is not None:
            t[key] = u(a, b, t[key].shape)
    pools["mesh"]["indexToCellID"] = np.arange(1, n1 + 1, dtype=np.int32)
    rg = pools["ridging"]
    rg["ridgeConvergence"], rg["ridgeShear"] = conv, shear
    pools["velocity_solver"]["dynamicsTimeStep"] = np.array(
        float(config["config_dt"]) / config["config_dynamics_subcycle_number"])


def _radiation_state(rs, case: dict) -> None:
    """The shortwave's inputs on _tracers' categories (ice-free cells and
    categories, a sub-puny cell, ice without volume), with the owned cells'
    branches (row r is indexToCellID r + 1):

    0. a category at the melting point (the warm albedos, fT = 0);
    1. no ice;
    2. a category just above -2 puny (warm) and one just below (cold);
    3. sub-puny ice;
    4. bare ice (no snow), the first category ice-free;
    5. ice without volume in the last category (atan(0));
    6. thick ice and deep snow;
    7. no sun;
    8. for prep_radiation: no previous net shortwave (scaling factor 0) and
       no ice area; row 6 a sub-puny previous net shortwave.

    The topo ponds' area and lid thickness straddle puny and
    config_critical_pond_ice_thickness."""
    pools, dims, config = case["pools"], case["dims"], case["config"]
    nc, ncat = dims["nCells"], dims["nCategories"]
    n1 = nc + 1
    u = _uniform(rs)
    t = pools["tracers"]
    tsfc = u(-30.0, -0.5, (n1, ncat, 1))
    tsfc[0, 0] = 0.0
    if ncat >= 2:
        tsfc[2, 0], tsfc[2, 1] = -1.0e-11, -3.0e-11
    t["surfaceTemperature"] = tsfc
    snow = t["snowVolumeCategory"]
    snow[4] = 0.0
    t["iceAreaCategory"][4, 0] = 0.0
    if ncat >= 2:
        t["iceVolumeCategory"][6, -1] = t["iceAreaCategory"][6, -1] * 6.0
        snow[6, -1] = t["iceAreaCategory"][6, -1] * 1.5
    ac = pools["atmos_coupling"]
    for key, hi in (("shortwaveVisibleDirectDown", 250.0), ("shortwaveVisibleDiffuseDown", 120.0),
                    ("shortwaveIRDirectDown", 200.0), ("shortwaveIRDiffuseDown", 100.0)):
        value = u(0.0, hi, (n1,))
        value[7] = 0.0
        ac[key] = value
    sw = pools["shortwave"]
    for key in ("albedoVisibleDirectArea", "albedoVisibleDiffuseArea", "albedoIRDirectArea", "albedoIRDiffuseArea"):
        sw[key] = u(0.0, 0.9, (n1,))
    scale = u(50.0, 400.0, (n1,))
    scale[8], scale[6] = 0.0, 5.0e-12
    sw["shortwaveScalingFactor"] = scale
    aggregate = np.zeros(n1)
    for n in range(ncat):
        aggregate = aggregate + t["iceAreaCategory"][:, n, 0]
    aggregate[8] = 0.0
    pools["tracers_aggregate"]["iceAreaCell"] = aggregate
    coszen = u(0.05, 0.9, (n1,))
    coszen[7] = -0.3
    sw["solarZenithAngleCosine"] = coszen
    for key in ("surfaceShortwaveFlux", "interiorShortwaveFlux", "penetratingShortwaveFlux",
                "shortwaveLayerPenetration", "absorbedShortwaveSnowLayer", "absorbedShortwaveIceLayer"):
        sw[key] = u(0.0, 200.0, sw[key].shape)
    hp1 = float(config["config_critical_pond_ice_thickness"])
    if t.get("pondArea") is not None:
        pond = u(0.0, 0.4, t["pondArea"].shape)
        pond[0, 0] = 5.0e-12
        t["pondArea"] = pond
    if t.get("pondLidThickness") is not None:
        lid = u(0.0, 2.0 * hp1, t["pondLidThickness"].shape)
        lid[2] = hp1
        t["pondLidThickness"] = lid
    pools["mesh"]["indexToCellID"] = np.arange(1, n1 + 1, dtype=np.int32)
    if case["stage"] != "prep_radiation" and config["config_shortwave_type"] == "dEdd" \
            and config["config_calc_surface_temperature"]:
        _dedd_state(rs, case)


def _dedd_state(rs, case: dict) -> None:
    """The delta-Eddington shortwave's inputs on _radiation_state's (its draws
    only for these cases, so the other cases' bytes stay):

    - ice with volume everywhere: run_dEdd's ice without volume has no layer
      thickness and its bottom layer's chlorophyll absorption kalg*(0.5/0)
      makes NaN, whose payload is the host's;
    - longitudes over [0, 2 pi) with one at pi and two above it (the wrap);
    - a category just above and one just below hs_min, a surface between
      -dT_mlt and the melting point (melting snow grains);
    - ponds shallower than hpmin, between hpmin and hp0 and deeper; level
      ponds under deep snow (rp < 0.15) and lids with and without snow on
      them; topo lids straddling config_critical_pond_ice_thickness."""
    pools, dims = case["pools"], case["dims"]
    nc, ncat = dims["nCells"], dims["nCategories"]
    n1 = nc + 1
    u = _uniform(rs)
    t = pools["tracers"]
    area, volume, snow = t["iceAreaCategory"], t["iceVolumeCategory"], t["snowVolumeCategory"]
    volume[5, -1] = area[5, -1] * 1.2
    snow[2, 0] = area[2, 0] * 5.0e-5
    if ncat >= 2:
        snow[2, 1] = area[2, 1] * 2.0e-4
    t["surfaceTemperature"][6, 0] = -0.7
    t["surfaceTemperature"][8, :] = u(-1.4, 0.0, (ncat, 1))
    lon = u(0.0, 2.0 * np.pi, (n1,))
    lon[0], lon[1], lon[2] = np.pi, 1.5 * np.pi, 2.0 * np.pi - 1.0e-9
    pools["mesh"]["lonCell"] = lon
    fsnow = u(0.0, 1.0e-6, (n1,))
    fsnow[5] = 0.0
    pools["atmos_coupling"]["snowfallRate"] = fsnow
    if t.get("pondDepth") is not None:
        depth = t["pondDepth"]
        depth[2] = 0.003
        depth[4] = u(0.005, 0.2, depth[4].shape)
        depth[6] = 0.01
    ponds = pools["ponds"]
    if ponds.get("pondSnowDepthDifference") is not None:
        dhs = u(0.0, 0.05, (n1, ncat))
        dhs[:3] = 0.0
        ponds["pondSnowDepthDifference"] = dhs
    if ponds.get("pondLidMeltFluxFraction") is not None:
        ponds["pondLidMeltFluxFraction"] = u(0.0, 1.0, (n1, ncat))


def _step_state(rs, case: dict) -> None:
    """The column step's inputs: :func:`_thermo_state`'s BL99 column (its
    draws first) and, on top, what the other stages read: the open water
    (one minus the ice area), the thickness distribution before the
    thermodynamics (the state itself, for a case without the vertical
    thermodynamics, which sets it), the ridging forcing and
    ``dynamicsTimeStep``, the down shortwave (a cell without sun), the area
    albedos and the previous net shortwave prep_radiation scales by, the
    longitudes, the rain and the initial salinity profile new ice takes. Ice has volume everywhere (the
    delta-Eddington shortwave's ice without volume, see :func:`_dedd_state`)."""
    _thermo_state(rs, case)
    pools, dims, config = case["pools"], case["dims"], case["config"]
    nc, ncat = dims["nCells"], dims["nCategories"]
    n1 = nc + 1
    u = _uniform(rs)
    t, ist, ta = pools["tracers"], pools["icestate"], pools["tracers_aggregate"]
    ist["openWaterArea"] = 1.0 - ta["iceAreaCell"]
    volume = np.zeros(n1)
    for n in range(ncat):
        volume = volume + t["iceVolumeCategory"][:, n, 0]
    ta["iceVolumeCell"] = volume
    if not config["config_use_column_vertical_thermodynamics"]:
        ist["iceAreaCellInitial"] = ta["iceAreaCell"].copy()
        for key in ("iceAreaCategory", "iceVolumeCategory", "snowVolumeCategory"):
            ist[f"{key}Initial"] = t[key][:, :, 0].copy()
    rg = pools["ridging"]
    conv, shear = u(0.0, 3.0e-6, (n1,)), u(0.0, 6.0e-6, (n1,))
    conv[3] = 0.0
    rg["ridgeConvergence"], rg["ridgeShear"] = conv, shear
    pools["velocity_solver"]["dynamicsTimeStep"] = np.array(
        float(config["config_dt"]) / config["config_dynamics_subcycle_number"])
    ac = pools["atmos_coupling"]
    for key, hi in (("shortwaveVisibleDirectDown", 250.0), ("shortwaveVisibleDiffuseDown", 120.0),
                    ("shortwaveIRDirectDown", 200.0), ("shortwaveIRDiffuseDown", 100.0)):
        value = u(0.0, hi, (n1,))
        value[7] = 0.0
        ac[key] = value
    ac["rainfallRate"] = u(0.0, 2.0e-5, (n1,))
    sw = pools["shortwave"]
    for key in ("albedoVisibleDirectArea", "albedoVisibleDiffuseArea", "albedoIRDirectArea", "albedoIRDiffuseArea"):
        sw[key] = u(0.0, 0.9, (n1,))
    sw["shortwaveScalingFactor"] = u(50.0, 400.0, (n1,))
    sw["solarZenithAngleCosine"] = u(0.05, 0.9, (n1,))
    pools["mesh"]["lonCell"] = u(0.0, 2.0 * np.pi, (n1,))
    # seaice_init_column_physics_package_variables' profiles, which the new ice takes
    nil = dims["nIceLayers"]
    initial = pools["initial"]
    initial["initialSalinityProfile"] = u(0.5, 3.5, (n1, nil + 1))
    initial["initialMeltingTemperatureProfile"] = -_DEPRESST * initial["initialSalinityProfile"]
    for key, (a, b) in (("iceAge", (0.0, 3.0e7)), ("firstYearIceArea", (0.0, 1.0)),
                        ("levelIceArea", (0.2, 1.0)), ("levelIceVolume", (0.2, 1.0)),
                        ("pondArea", (0.0, 0.3)), ("pondDepth", (0.0, 0.2)),
                        ("pondLidThickness", (0.0, 0.05))):
        if t.get(key) is not None:
            t[key] = u(a, b, t[key].shape)


def _bgc_state(rs, case: dict) -> None:
    """The brine height's inputs: categories of 0.3-3 m ice (and below the
    0.05 m ``thinS`` in one cell), the thickness and area before the
    thermodynamics, the melt and growth, mushy enthalpies and salinities (a
    layer saltier than the liquidus break, a nearly melted category), the
    brine fraction above sea level with little snow (runoff) and below it
    under heavy snow (flooding), non-positive brine fractions, new ice (an
    initial area at or below puny), an ice-free cell, and ``colpkg_init_hbrine``'s
    grids. ``remap_stop`` in the name: a negative thickness in a middle
    category (a stop the next category clears); ``abort``: that, and the same
    in two cells' last category.
    Separate numpy ufuncs only, so the words are the same on every host."""
    pools, dims, name = case["pools"], case["dims"], case["name"]
    nc, ncat, nil, nbl = dims["nCells"], dims["nCategories"], dims["nIceLayers"], dims["nBioLayers"]
    n1 = nc + 1
    u = _uniform(rs)
    t, ist, mg, oc, bgc = (pools["tracers"], pools["icestate"], pools["melt_growth_rates"],
                           pools["ocean_coupling"], pools["biogeochemistry"])
    area = u(0.02, 0.9 / ncat, (n1, ncat))
    hin = u(0.3, 3.0, (n1, ncat))
    hsn = u(0.0, 0.3, (n1, ncat))
    fbri = u(0.85, 1.2, (n1, ncat))
    hsn[0] = 0.01                                  # brine above sea level: runoff
    fbri[0] = u(1.05, 1.2, (ncat,))
    hsn[1] = 0.6                                   # brine below sea level: flooding
    fbri[1] = u(0.7, 0.9, (ncat,))
    area[2] = 0.0                                  # no ice
    hin[3] = u(0.005, 0.045, (ncat,))              # below thinS
    fbri[5, ::2] = 0.0                             # reset to 1
    fbri[5, 1 % ncat] = -0.5
    area_init = area * u(0.9, 1.1, (n1, ncat))
    area_init[4, 0] = 0.0                          # new ice in the first category: no Rayleigh criterion
    area_init[6, 1 % ncat] = 5.0e-12               # at most puny
    hin_init = hin * u(0.9, 1.1, (n1, ncat))
    hin_init[6, 0] = 1.0e-13                       # an old thickness below puny
    if "abort_remap" in name or "remap_stop" in name:
        # remap_zbgc stops on a negative thickness, which colpkg_biogeochemistry does not
        # check: the next ice category's preflushing_changes clears it (cell 4)
        hin[4, 2 % ncat] = -0.4
        hin_init[4, 2 % ncat] = -0.4
    if "abort_remap" in name:
        # only a stop in a cell's last category aborts (cell 7)
        hin[7, ncat - 1] = -0.2
        hin_init[7, ncat - 1] = -0.2
        hin[8, ncat - 1] = -0.3                    # a later cell, which MPAS never reaches
        hin_init[8, ncat - 1] = -0.3
    t["iceAreaCategory"] = area[:, :, None]
    t["iceVolumeCategory"] = (area * hin)[:, :, None]
    t["snowVolumeCategory"] = (area * hsn)[:, :, None]
    t["brineFraction"] = fbri[:, :, None]
    sal = u(2.0, 12.0, (n1, ncat, nil))
    sal[7, 0, nil // 2] = 130.0                    # above the liquidus break salinity
    q = u(-3.4e8, -1.0e8, (n1, ncat, nil))
    q[7, 1 % ncat] = u(-5.0e5, -1.0e4, (nil,))     # nearly melted
    t["iceSalinity"], t["iceEnthalpy"] = sal, q
    ist["iceAreaCategoryInitial"] = area_init
    ist["iceVolumeCategoryInitial"] = area_init * hin_init
    ist["iceThicknessCategoryInitial"] = u(0.0, 3.0, (n1, ncat))
    total = np.zeros(n1)
    for n in range(ncat):
        total = total + area[:, n]
    ist["openWaterArea"] = 1.0 - total
    for key, hi in (("basalIceMeltCategory", 0.01), ("surfaceIceMeltCategory", 0.01),
                    ("congelationCategory", 0.01), ("snowiceFormationCategory", 0.005),
                    ("snowMeltCategory", 0.005)):
        value = u(0.0, hi, (n1, ncat))
        value[::2, 0] = 0.0
        mg[key] = value
    oc["seaSurfaceSalinity"] = u(30.0, 35.0, (n1,))
    sst = u(-1.9, -1.0, (n1,))
    sst[8] = 0.5
    oc["seaSurfaceTemperature"] = sst
    bgc["newlyFormedIce"] = rs.randint(0, 2, (n1, ncat)).astype(np.int32)
    bgc["rayleighCriteriaReal"] = u(0.0, 1.0, (n1,))
    bgc["netBrineHeight"] = u(0.0, 0.5, (n1,))
    for key, value in cv.hbrine_grids(nbl, nil).items():
        bgc[key] = value
    pools["mesh"]["indexToCellID"] = np.arange(1, n1 + 1, dtype=np.int32)


def _nonreactive_state(rs, case: dict) -> None:
    """On top of :func:`_bgc_state`, the vertical tracers' inputs: the
    nonreactive tracer's ice and snow layers (positive, one below puny) and
    mobile fractions, the ocean concentrations, snowfall (none in every third
    cell), the snow before the step (none in cell 0, deep in cell 1), the
    shortwave through the ice layers, and the fluxes and integrals the stage
    accumulates into. update_snow_bgc divides by the area before the step
    unguarded, which the step never leaves at 0 under ice: cell 4's new ice
    gets its area back. ``abort_init`` in the name: a negative tracer in cell
    6's first category, old ice (z_biogeochemistry's initialization error),
    and in cell 8's, which MPAS never reaches; ``abort_negative``: a negative
    ocean concentration under cell 3's thin old ice, which the thin-ice flux
    adds as the brine grows. Separate numpy ufuncs only."""
    pools, dims, name = case["pools"], case["dims"], case["name"]
    n1, ncat, nil, nbl, nz = (dims["nCells"] + 1, dims["nCategories"], dims["nIceLayers"], dims["nBioLayers"],
                              dims["nZBGCTracers"])
    u = _uniform(rs)
    t, ist, bgc = pools["tracers"], pools["icestate"], pools["biogeochemistry"]
    area = t["iceAreaCategory"][:, :, 0]
    ist["iceAreaCategoryInitial"][4, 0] = area[4, 0]
    hsn = u(0.0, 0.3, (n1, ncat))
    hsn[0] = 0.0
    hsn[1] = u(0.3, 0.6, (ncat,))
    ist["snowVolumeCategoryInitial"] = np.multiply(ist["iceAreaCategoryInitial"], hsn)
    conc = u(0.5, 20.0, (n1, ncat, nbl + 3))
    conc[:, :, nbl + 1:] = u(0.0, 5.0, (n1, ncat, 2))
    conc[7, 0, 1] = 1.0e-12
    t["verticalNonreactiveConc"] = conc
    t["mobileFraction"] = u(0.0, 1.0, (n1, ncat, nz))
    bgc["oceanBioConcentrations"] = u(0.5, 15.0, (n1, nz))
    snowfall = u(0.0, 3.0e-5, (n1,))
    snowfall[::3] = 0.0
    pools["atmos_coupling"]["snowfallRate"] = snowfall
    pools["shortwave"]["shortwaveLayerPenetration"] = u(0.0, 40.0, (n1, ncat, nil + 1))
    bgc["bioShortwaveFlux"] = u(0.0, 40.0, (n1, ncat, nbl + 1))
    for key, lo, hi in (("oceanBioFluxes", -1.0e-4, 1.0e-4), ("snowIceBioFluxes", 0.0, 1.0e-4),
                        ("atmosIceBioFluxes", 0.0, 1.0e-4), ("totalVerticalBiologyIce", 0.0, 50.0),
                        ("totalVerticalBiologySnow", 0.0, 10.0)):
        bgc[key] = u(lo, hi, (n1, nz))
    if "abort_init" in name:
        for cell in (6, 8):
            bgc["newlyFormedIce"][cell, 0] = 0
            conc[cell, 0, 2] = -1.0
    if "abort_negative" in name:
        bgc["newlyFormedIce"][3] = 0
        bgc["oceanBioConcentrations"][3] = -50.0


def _nonreactive_carry_state(rs, case: dict) -> None:
    """What the ITD thermodynamics and the ridging read of the nonreactive
    tracer: its ice and snow layers (positive), mobile fractions, the ocean
    concentration and flux, and ``colpkg_init_hbrine``'s grids.
    ``abort_regrid`` in the name: a slightly negative brine fraction in cell
    2's first category, under :func:`_itd_state`'s spread frazil, so the
    tracer's ``regrid_stationary`` stops on a negative old brine height.
    Separate numpy ufuncs only."""
    pools, dims, name = case["pools"], case["dims"], case["name"]
    n1, ncat, nil, nbl, nz = (dims["nCells"] + 1, dims["nCategories"], dims["nIceLayers"], dims["nBioLayers"],
                              dims["nZBGCTracers"])
    u = _uniform(rs)
    t, bgc = pools["tracers"], pools["biogeochemistry"]
    conc = u(0.5, 20.0, (n1, ncat, nbl + 3))
    conc[:, :, nbl + 1:] = u(0.0, 5.0, (n1, ncat, 2))
    t["verticalNonreactiveConc"] = conc
    t["mobileFraction"] = u(0.0, 1.0, (n1, ncat, nz))
    bgc["oceanBioConcentrations"] = u(0.5, 15.0, (n1, nz))
    bgc["oceanBioFluxes"] = u(-1.0e-4, 1.0e-4, (n1, nz))
    for key, value in cv.hbrine_grids(nbl, nil).items():
        bgc[key] = value
    if "abort_regrid" in name:
        t["brineFraction"][2, 0, 0] = -1.0e-3


def _zsal_dtemp_band(case: dict) -> None:
    """Cell 5's ice 0.05 degC below each layer's melting point, in daylight:
    inside BL99's absorbed-shortwave band, which ``temperature_changes``'
    dTemp (0.02, or 0.1 with ``solve_zsal``, ice_therm_bl99.F90:282-287)
    moves. Separate numpy ufuncs only."""
    t = case["pools"]["tracers"]
    tmlt = np.multiply(np.negative(t["iceSalinity"][5]), _DEPRESST)
    tin = np.subtract(tmlt, 0.05)
    t["iceEnthalpy"][5] = -_RHOI * (_CP_ICE * (tmlt - tin) + _LFRESH * (1.0 - tmlt / tin) - _CP_OCN * tmlt)


def _zsal_state(rs, case: dict) -> None:
    """On top of :func:`_bgc_state`, the vertical salinity's inputs: the
    salinity on the biology grid, the surface temperature, a brine height
    above ``maxhbr`` times the ice (cell 8's first category, old ice), thin
    first-category ice with the Rayleigh criterion met (cell 1) and not
    (cell 3), and the diagnostics it accumulates into.
    ``max_salinity`` in the name: cold, old, thick ice above ``max_salin``
    and below its melting salinity in cell 0's first category, where
    ``solve_S_dt`` stops."""
    pools, dims, name = case["pools"], case["dims"], case["name"]
    nc, ncat, nil, nbl = dims["nCells"], dims["nCategories"], dims["nIceLayers"], dims["nBioLayers"]
    n1 = nc + 1
    u = _uniform(rs)
    t, ist, bgc = pools["tracers"], pools["icestate"], pools["biogeochemistry"]
    t["verticalSalinity"] = u(2.0, 12.0, (n1, ncat, nbl))
    t["surfaceTemperature"] = u(-20.0, -1.0, (n1, ncat, 1))
    t["brineFraction"][8, 0, 0] = 1.4               # hbr_old > maxhbr*hice_old
    bgc["newlyFormedIce"][8, 0] = 0
    area = t["iceAreaCategory"][:, :, 0]
    for cell, real in ((1, 0.8), (3, 0.2)):
        bgc["rayleighCriteriaReal"][cell] = real
        bgc["newlyFormedIce"][cell, 0] = 0
        thin = np.multiply(area[cell, 0], 0.04)
        t["iceVolumeCategory"][cell, 0, 0] = thin
        ist["iceVolumeCategoryInitial"][cell, 0] = np.multiply(ist["iceAreaCategoryInitial"][cell, 0], 0.045)
    for key, lo, hi in (("totalVerticalSalinity", 0.0, 50.0), ("zSalinityFlux", -1.0e-5, 1.0e-5),
                        ("zSalinityGDFlux", -1.0e-5, 1.0e-5)):
        bgc[key] = u(lo, hi, (n1,))
    if "max_salinity" in name:
        bgc["newlyFormedIce"][0, 0] = 0
        t["brineFraction"][0, 0, 0] = 1.0
        t["iceEnthalpy"][0, 0] = -3.4e8
        t["iceSalinity"][0, 0] = 4.0
        t["verticalSalinity"][0, 0] = 250.0
        t["surfaceTemperature"][0, 0, 0] = -25.0


def _zsal_solve_carry_state(rs, case: dict) -> None:
    """Under ``zsal_solve``, what the ITD thermodynamics, the ridging and the
    steps read of the vertical salinity: its profile (positive), the salt
    flux and, for the ITD and the ridging, ``colpkg_init_hbrine``'s grids.
    ``abort_regrid`` in the name: a slightly negative brine fraction in cell
    2's first category, under :func:`_itd_state`'s spread frazil, so
    ``regrid_stationary``'s remap stops on a negative old brine height."""
    pools, dims, name = case["pools"], case["dims"], case["name"]
    n1, ncat, nil, nbl = dims["nCells"] + 1, dims["nCategories"], dims["nIceLayers"], dims["nBioLayers"]
    u = _uniform(rs)
    t, bgc = pools["tracers"], pools["biogeochemistry"]
    t["verticalSalinity"] = u(2.0, 12.0, (n1, ncat, nbl))
    bgc["zSalinityFlux"] = u(-1.0e-5, 1.0e-5, (n1,))
    if case["stage"] in (_ITD, _RDG):
        for key, value in cv.hbrine_grids(nbl, nil).items():
            bgc[key] = value
    if "abort_regrid" in name:
        t["brineFraction"][2, 0, 0] = -1.0e-3


def _coupled_dynamics(name: str, seed: int, torus: tuple, config: dict) -> tuple[dict, dict]:
    """The time-step oracle's dynamics state (:mod:`woof.verify.mpas_seaice_step_oracle`)
    on a culled hexagonal torus of ``torus = (nq, nr, owned columns)``: the
    mesh's dimensions, and that case."""
    from woof.verify import mpas_seaice_step_oracle as so
    from woof.verify.mpas_ocean_hexmesh import hex_mesh
    nq, nr, owned = torus
    mesh = hex_mesh(nq, nr, 0.08, 1, np.random.RandomState(seed + 1), min_levels=1, owned_columns=owned)
    dynamics = so.make_case(name, seed + 2, (nq, nr), owned, int(config["oracle_n_steps"]),
                            float(config["config_dt"]), {}, precision="r8", mesh=mesh,
                            n_categories=int(config["config_nCategories"]))
    pools = dynamics["pools"]["mesh"]
    dims = {k: int(pools[k]) for k in ("nCells", "nEdges", "nVertices", "nCellsSolve", "nEdgesSolve",
                                       "nVerticesSolve", "maxEdges", "vertexDegree")}
    dims["maxEdges2"] = 2 * dims["maxEdges"]
    return dims, dynamics


#: the column state the dynamics state does not replace
_COLUMN_OWNED = ("tracers", "tracers_aggregate", "atmos_coupling", "ridging")


def _overlay_dynamics(case: dict, dynamics: dict) -> None:
    """The dynamics state over the column's: the mesh, the masks, the
    variational basis, the velocities and the ocean forcing (not the
    category state, the aggregates, the atmosphere, the ridging forcing, the
    open water or the initial ice area, which are the column step's)."""
    from woof.verify import mpas_seaice_step_oracle as so
    pools = case["pools"]
    for pool, field, kind, _ in so.FIELDS:
        if kind == DIM or pool in _COLUMN_OWNED or (pool, field) in (
                ("icestate", "openWaterArea"), ("icestate", "iceAreaCellInitial"),
                ("velocity_solver", "dynamicsTimeStep"), ("velocity_solver", "elasticTimeStep")):
            continue
        value = np.asarray(dynamics["pools"][pool][field])
        mine = pools[pool].get(field)
        if mine is None or np.shape(mine) != value.shape:
            raise ValueError(f"{pool}/{field}: the column stream carries {np.shape(mine)}, the dynamics {value.shape}")
        pools[pool][field] = value.astype(np.asarray(mine).dtype)
    nc = case["dims"]["nCells"]
    pools["mesh"]["indexToCellID"] = np.arange(1, nc + 2, dtype=np.int32)


def _oml_state(rs, case: dict) -> None:
    """The open ocean under :func:`_thermo_state`'s air (its draws first): the
    down shortwave (a cell without sun), the mixed layer's depth (a shallow
    cell whose freezing/melting potential the +-1000 W m-2 clamp bounds), a
    sea surface at, below and well above freezing, a warm-air cell (a stable
    boundary layer over the sea), the deep heat flux convergence (positive
    where the mixed layer is at freezing, so it is zeroed), the fluxes
    through the ice and the drag ratio the call passes."""
    _thermo_state(rs, case)
    n1 = case["dims"]["nCells"] + 1
    u = _uniform(rs)
    ac = case["pools"]["atmos_coupling"]
    for key, hi in (("shortwaveVisibleDirectDown", 250.0), ("shortwaveVisibleDiffuseDown", 120.0),
                    ("shortwaveIRDirectDown", 200.0), ("shortwaveIRDiffuseDown", 100.0)):
        value = u(0.0, hi, (n1,))
        value[7] = 0.0
        ac[key] = value
    _oml_ocean(rs, case)


def _oml_ocean(rs, case: dict) -> None:
    """:func:`_oml_state`'s ocean (and the column step's, with the mixed layer on)."""
    pools, dims = case["pools"], case["dims"]
    n1 = dims["nCells"] + 1
    u = _uniform(rs)
    pools["atmos_coupling"]["airPotentialTemperature"][5] = 285.0
    oc = pools["ocean_coupling"]
    tf = oc["seaFreezingTemperature"]
    sst = tf + u(0.0, 1.5, (n1,))
    sst[1], sst[2] = tf[1] - 0.3, tf[2]
    sst[4], sst[8] = 3.0, 6.0
    oc["seaSurfaceTemperature"] = sst
    hmix = u(5.0, 40.0, (n1,))
    hmix[3] = 0.05
    oc["oceanMixedLayerDepth"] = hmix
    qdp = u(-30.0, 30.0, (n1,))
    qdp[1], qdp[2] = 15.0, 25.0
    oc["oceanHeatFluxConvergence"] = qdp
    of = pools["ocean_fluxes"]
    of["oceanHeatFlux"] = u(-60.0, 10.0, (n1,))
    of["oceanShortwaveFlux"] = u(0.0, 20.0, (n1,))
    pools["drag"]["airOceanDragCoefficientRatio"] = u(0.5, 1.5, (n1,))


def make_case(name, seed, stage, overrides, torus=None) -> dict:
    rs = np.random.RandomState(seed)
    config = configs_of(overrides, stage)
    mesh, dynamics = MESH, None
    if torus is not None:
        mesh, dynamics = _coupled_dynamics(name, seed, torus, config)
        config["diagnostics:xtime"] = config["config_start_time"]
        config["diagnostics:simulationStartTime"] = config["config_start_time"]
    rows, dims, active = field_rows(config, mesh)
    case = {"name": name, "precision": "r8", "stage": stage, "config": config,
            "dt": 1800.0 if stage == "update_state" else float(config["config_dt"]),
            "dims": {**dims, "nCells": mesh["nCells"], "nCellsSolve": mesh["nCellsSolve"],
                     "nVertLevels": 0}, "pools": {}}
    for pool, fname, kind, fshape, *on in rows:
        p = case["pools"].setdefault(pool, {})
        if kind == DIM:
            p[fname] = mesh.get(fname, dims.get(fname))
            continue
        if not on[0]:
            p[fname] = None
            continue
        shp = _stream.shape(case, fshape)
        value = _values(rs, fname, shp, case["dims"], stage)
        if value is None:
            value = rs.randint(0, 2, shp).astype(np.int32) if kind == INT else _uniform(rs)(-1.0, 1.0, shp)
        p[fname] = value
    _tracers(rs, case["pools"], case["dims"])
    if stage == "vertical_thermodynamics":
        _thermo_state(rs, case)
        if config["config_use_vertical_zsalinity"]:
            _zsal_dtemp_band(case)
    if stage in (_STEP, _SEAICE_STEP):
        _step_state(rs, case)
        if config["config_use_brine"]:
            _brine_step_state(rs, case)
        if config["config_use_ocean_mixed_layer"]:
            _oml_ocean(rs, case)
    if dynamics is not None:
        _overlay_dynamics(case, dynamics)
    if stage == "itd_thermodynamics":
        _itd_state(rs, case)
    if stage == "ridging":
        _ridge_state(rs, case)
    if stage in ("itd_thermodynamics", "ridging") and config["config_use_brine"]:
        _brine_fraction(rs, case)
    if stage in _RADIATION_STAGES:
        _radiation_state(rs, case)
    if stage == _OML:
        _oml_state(rs, case)
    if stage == _BGC_STAGE:
        _bgc_state(rs, case)
        if config["config_use_vertical_zsalinity"]:
            _zsal_state(rs, case)
        if cv.vertical_tracers(config):
            _nonreactive_state(rs, case)
    if stage in (_ITD, _RDG, _STEP, _SEAICE_STEP) and cv.solves_zsal(config):
        _zsal_solve_carry_state(rs, case)
    if stage in (_ITD, _RDG) and cv.vertical_tracers(config):
        _nonreactive_carry_state(rs, case)
    for key, on in active.items():
        config[f"package:{key}"] = on
    case["fields"] = [r[:4] for r in rows]
    return case


def cases(precision: str = "r8", rows=CASES) -> list[dict]:
    if precision != "r8":
        raise ValueError("the column oracle is RKIND 8 only")
    return [make_case(*row) for row in rows]


def input_bytes(case: dict) -> bytes:
    names = case["config"].get("woof_v842_corrections")
    if names is not None and not isinstance(names, str):
        # several corrections (which the oracle does not read) go into the stream as one string
        case = {**case, "config": {**case["config"], "woof_v842_corrections": " ".join(names)}}
    return _stream.input_bytes(case["fields"], case)


def input_digest(case: dict) -> str:
    return hashlib.sha256(input_bytes(case)).hexdigest()


def read_outputs(raw: bytes, case: dict) -> dict[str, np.ndarray]:
    return _stream.read_outputs(case["fields"], raw, case)


#: column_biogeochemistry's MPAS_LOG_ERR line, and the column package's char_len
_BGC_ABORT_LINE = "ERR column_biogeochemistry: "
_CHAR_LEN = 80


def parse_log(text: str) -> dict:
    """The tracer object (``TOBJ`` rows, a repeated name's rows concatenated)
    and every other line the routines logged. column_biogeochemistry's
    abortMessage (strKIND long) is never initialised and the column
    package's stop labels are ``char_len`` dummies, so past ``char_len`` its
    abort line is whatever the stack held: only the label is kept."""
    tobj: dict[str, list[int]] = {}
    other = []
    for line in text.split("\n"):
        if line.startswith("TOBJ "):
            parts = line.split()
            tobj.setdefault(parts[1], []).extend(int(x) for x in parts[2:])
        elif line.startswith(_BGC_ABORT_LINE):
            other.append(_BGC_ABORT_LINE + line[len(_BGC_ABORT_LINE):][:_CHAR_LEN].rstrip())
        elif line.strip():
            other.append(line.rstrip())
    return {"tracer_object": tobj, "log": other}


def _runtime_error(stderr: str) -> dict | None:
    """A gfortran runtime error's file, line and message (``At line N of file
    F`` / ``Fortran runtime error: M``), or None."""
    at = re.search(r"At line (\d+) of file (\S+)", stderr)
    error = re.search(r"Fortran runtime error: (.+)", stderr)
    if not at or not error:
        return None
    return {"file": Path(at.group(2)).name, "line": int(at.group(1)), "error": error.group(1).strip()}


def _critical(parsed: dict) -> dict | None:
    """An ``MPAS_LOG_CRIT`` abort's record (the stub's ``CRIT`` line, its
    integers and the ``ERR`` line before it), or None."""
    lines = parsed["log"]
    at = next((i for i, line in enumerate(lines) if line.startswith("CRIT ")), None)
    if at is None:
        return None
    error = next((line[len("ERR "):] for line in reversed(lines[:at]) if line.startswith("ERR ")), None)
    ints = (lines[at + 1].split()[1:] if at + 1 < len(lines) and lines[at + 1].startswith("INTS") else [])
    return {"error": error, "critical": lines[at][len("CRIT "):], "ints": [int(x) for x in ints]}


def run_oracle(build_dir: Path, case: dict) -> tuple[dict[str, np.ndarray] | None, dict]:
    """The oracle's outputs and log. A case on which v8.4.2 itself dies, with
    a Fortran runtime error or an ``MPAS_LOG_CRIT`` message (which aborts
    MPAS), has no outputs; its log carries ``crash`` (the error's file, line
    and message, or the critical message, its integers and the error logged
    before it)."""
    with tempfile.TemporaryDirectory() as tmp:
        src, dst, log = Path(tmp) / "in.bin", Path(tmp) / "out.bin", Path(tmp) / "log.txt"
        src.write_bytes(input_bytes(case))
        run = subprocess.run([str(Path(build_dir) / "run_column_r8"), str(src), str(dst), str(log)],
                             capture_output=True, text=True)
        crash = _runtime_error(run.stderr) if run.returncode else None
        parsed = parse_log(log.read_text(encoding="utf-8"))
        if run.returncode == 7 and crash is None:
            crash = _critical(parsed)
        if run.returncode and crash is None:
            raise subprocess.CalledProcessError(run.returncode, run.args, run.stdout, run.stderr)
        if crash is not None:
            return None, {**parsed, "crash": crash}
        return read_outputs(dst.read_bytes(), case), parsed


word_mismatches = _stream.word_mismatches


def changed_fields(case: dict, outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    changed = {}
    for key, value in outputs.items():
        pool, name = key.split("/", 1)
        if word_mismatches(value, np.asarray(case["pools"][pool][name])):
            changed[key] = value
    return changed


def load_fixture(fixture_dir: Path = FIXTURE_DIR) -> dict[str, dict[str, np.ndarray]]:
    data = np.load(fixture_dir / "oracle-r8.npz")
    out: dict[str, dict[str, np.ndarray]] = {}
    for key in data.files:
        case, field = key.split("/", 1)
        out.setdefault(case, {})[field] = data[key]
    return out


def load_logs(fixture_dir: Path = FIXTURE_DIR) -> dict:
    return json.loads((fixture_dir / "column-logs.json").read_text(encoding="utf-8"))


def stand_ins(config: dict) -> set[tuple[str, str]]:
    """The fields the oracle gives an array v8.4.2 does not have (see
    :func:`field_rows`): not v8.4.2's words, so not compared."""
    if _BGC_INACTIVE in cv.corrections(config) and config.get("oracle_stage") in (_STEP, _SEAICE_STEP):
        return {("biogeochemistry", name) for name in cstep.inactive_bgc_diagnostics(config)}
    return set()


def expected_outputs(case: dict, fixture: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    skip = stand_ins(case["config"])
    for pool, name, kind, _ in case["fields"]:
        if kind not in (REAL, INT) or case["pools"][pool][name] is None or (pool, name) in skip:
            continue
        key = f"{pool}/{name}"
        out[key] = fixture[key] if key in fixture else np.asarray(case["pools"][pool][name])
    return out


def on_device(case: dict) -> dict[str, dict]:
    return _stream.on_device(case["fields"], case)


def run_gpu(case: dict) -> tuple[dict[str, np.ndarray], "cv.ColumnState"]:
    """Run woof.hex.mpas_seaice.column_v842 on ``case`` as run_column.F90 drives the Fortran."""
    import cupy as cp

    dev = on_device(case)
    config = case["config"]
    stage = case["stage"]
    # the stages that carry the brine height (each refuses the rest itself)
    state = cv.init_parameters(dev, config, biogeochemistry=stage not in ("aggregate", "update_state", _OML),
                               z_tracers=stage in ("init", _BGC_STAGE, _ITD, _RDG))
    if stage == "init":
        cv.initialize_coupler_fields(dev, config, state)
        cv.init_variables(dev, config, state)
        cv.init_ice_state(dev, config, state)
        cv.reset_coupler_fluxes(dev, config, state)
    elif stage == "aggregate":
        cv.aggregate(dev, config, state)
    elif stage == "update_state":
        cv.update_state(dev, config, state, config["oracle_update_type"], case["dt"],
                        config["oracle_ice_age_time_offset"])
    elif stage == "vertical_thermodynamics":
        state.abort = ct.vertical_thermodynamics(dev, config, state)
    elif stage == "itd_thermodynamics":
        cv.init_column_itd(dev, state)
        state.abort = citd.itd_thermodynamics(dev, config, state)
    elif stage == "ridging":
        cv.init_column_itd(dev, state)
        state.abort = crdg.ridging(dev, config, state)
    elif stage == _BGC_STAGE:
        state.abort = cbgc.biogeochemistry(dev, config, state)
    elif stage == "prep_radiation":
        crad.prep_radiation(dev, config, state)
    elif stage == "radiation":
        crad.radiation(dev, config, state)
    elif stage == "init_shortwave":
        crad.init_shortwave(dev, config, state)
    elif stage == _STEP:
        cv.init_column_itd(dev, state)
        state.abort = cstep.run(dev, config, state, int(config["oracle_n_steps"]))
    elif stage == _SEAICE_STEP:
        state.abort, step = _run_seaice_step(dev, config)
    elif stage == _OML:
        oml.ocean_mixed_layer(dev, config, int(case["dims"]["nCellsSolve"]))
    else:
        raise ValueError(f"unknown stage {stage}")
    outputs = {f"{pool}/{name}": cp.asnumpy(dev[pool][name]) for pool, name, kind, _ in case["fields"]
               if kind in (REAL, INT) and dev[pool][name] is not None}
    if stage == _SEAICE_STEP:
        outputs[SEAICE_STEP_KEY] = np.array([step])
    return outputs, state


def _run_seaice_step(dev: dict, config: dict) -> tuple[tuple, int]:
    """:func:`woof.hex.mpas_seaice.timestep_v842.advance` as the ``seaice_step``
    stage drives ``seaice_timestep``: the abort (``(0, None, None)`` without
    one) and the step it stopped at."""
    from woof.hex.mpas_seaice import timestep_v842 as ts
    dev["diagnostics"]["xtime"] = config["diagnostics:xtime"]
    dev["diagnostics"]["simulationStartTime"] = config["diagnostics:simulationStartTime"]
    clock = ts.initialize(pools=dev, configs=config, basis="input")
    n_steps = int(config["oracle_n_steps"])
    try:
        ts.advance(pools=dev, configs=config, clock=clock, n_steps=n_steps,
                   finalize_last=bool(config["oracle_finalize_last"]))
    except ts.ColumnAbort as abort:
        return (abort.ierr, abort.cell, abort.message), abort.step
    return (0, None, None), n_steps


def measure_coupled(reference: dict | None = None, logs: dict | None = None, *, rows=COUPLED_CASES,
                    fixture_dir: Path = COUPLED_FIXTURE_DIR) -> dict[str, dict[str, int]]:
    """Per time-step case with the column package: the words the GPU gets
    wrong per field (an aborting case's too), whether its abort (ierr, first
    aborting cell, message) or the step it stopped at differs from the
    oracle's, and the tracer object's integers that differ."""
    reference = load_fixture(fixture_dir) if reference is None else reference
    logs = load_logs(fixture_dir) if logs is None else logs
    report: dict[str, dict[str, int]] = {}
    for case in cases("r8", rows):
        log = logs[case["name"]]
        got, state = run_gpu(case)
        row = {"abort": int(tuple(state.abort) != oracle_step_abort(log)),
               "step": int(got[SEAICE_STEP_KEY][0] != log["tracer_object"]["STEP"][0])}
        want = expected_outputs(case, reference.get(case["name"], {}))
        row.update({key: word_mismatches(got[key], want[key]) for key in want})
        mine = state.tracers.log_rows()
        row["tracer_object"] = sum(int(mine[k] != log["tracer_object"].get(k)) for k in mine)
        report[case["name"]] = row
    return report


def measure(reference: dict | None = None, logs: dict | None = None, *, rows=CASES,
            fixture_dir: Path = FIXTURE_DIR) -> dict[str, dict[str, int]]:
    """Per case and field, the words the GPU gets wrong, and the tracer
    object's integers that differ from the oracle's."""
    reference = load_fixture(fixture_dir) if reference is None else reference
    logs = load_logs(fixture_dir) if logs is None else logs
    report: dict[str, dict[str, int]] = {}
    for case in cases("r8", rows):
        got, state = run_gpu(case)
        want = expected_outputs(case, reference.get(case["name"], {}))
        report[case["name"]] = {key: word_mismatches(got[key], want[key]) for key in want}
        oracle = logs[case["name"]]["tracer_object"]
        mine = state.tracers.log_rows()
        report[case["name"]]["tracer_object"] = sum(int(mine[k] != oracle.get(k)) for k in mine)
    return report


def oracle_abort(log: dict, routine: str = "column_vertical_thermodynamics") -> tuple[int, int | None, str | None]:
    """The oracle's ``routine`` outcome: ``(ierr, cell, message)`` as
    :func:`woof.hex.mpas_seaice.column_therm_v842.vertical_thermodynamics` (and
    :func:`~woof.hex.mpas_seaice.column_itd_v842.itd_thermodynamics`) return it
    (the cell is ``indexToCellID``, the message the logged error)."""
    ierr = log["tracer_object"]["IERR"][0]
    if not ierr:
        return 0, None, None
    lines = log["log"]
    prefix = f"ERR {routine}: "
    message = next(line for line in lines if line.startswith(prefix))[len("ERR "):]
    cell = int(lines[lines.index("ERR iCell: $i") + 1].split()[1])
    return ierr, cell, message


def measure_bgc(reference: dict | None = None, logs: dict | None = None, *, rows=BGC_CASES,
                fixture_dir: Path = BGC_FIXTURE_DIR) -> dict[str, dict[str, int]]:
    """Per column-biogeochemistry case: the words the GPU gets wrong per
    field, the tracer object's integers that differ and, for the brine
    height's step, whether it reports an abort the oracle does not; for a
    case v8.4.2 dies on (``MPAS_LOG_CRIT``), whether the port fails to stop
    at the same cell with the same message."""
    reference = load_fixture(fixture_dir) if reference is None else reference
    logs = load_logs(fixture_dir) if logs is None else logs
    report: dict[str, dict[str, int]] = {}
    for case in cases("r8", rows):
        log = logs[case["name"]]
        if "crash" in log:
            # v8.4.2 dies: the port must stop at the same cell with the same message
            try:
                run_gpu(case)
            except cbgc.CriticalAbort as error:
                report[case["name"]] = {"crash": int(error.crash != log["crash"])}
            else:
                report[case["name"]] = {"crash": 1}
            continue
        got, state = run_gpu(case)
        row = {}
        if case["stage"] == _BGC_STAGE:
            row["abort"] = int(tuple(state.abort) != oracle_abort(log, "column_biogeochemistry"))
        want = expected_outputs(case, reference.get(case["name"], {}))
        row.update({key: word_mismatches(got[key], want[key]) for key in want})
        mine = state.tracers.log_rows()
        row["tracer_object"] = sum(int(mine[k] != log["tracer_object"].get(k)) for k in mine)
        report[case["name"]] = row
    return report


def measure_zsal(reference: dict | None = None, logs: dict | None = None, *, rows=BGC_ZSAL_CASES,
                 fixture_dir: Path = BGC_ZSAL_FIXTURE_DIR) -> dict[str, dict[str, int]]:
    """The vertical salinity's cases (``BGC_ZSAL_CASES`` by default): the
    init's and column_biogeochemistry's as :func:`measure_bgc`, the vertical
    thermodynamics' as :func:`measure_vt`."""
    reference = load_fixture(fixture_dir) if reference is None else reference
    logs = load_logs(fixture_dir) if logs is None else logs
    vt = tuple(row for row in rows if row[2] == "vertical_thermodynamics")
    rest = tuple(row for row in rows if row[2] != "vertical_thermodynamics")
    report: dict[str, dict[str, int]] = {}
    if rest:
        report.update(measure_bgc(reference, logs, rows=rest, fixture_dir=fixture_dir))
    if vt:
        report.update(measure_vt(reference, logs, rows=vt, fixture_dir=fixture_dir))
    return report


def measure_carry(reference: dict | None = None, logs: dict | None = None, *,
                  rows=BGC_CARRY_CASES + BGC_CARRY_COUPLED_CASES,
                  fixture_dir: Path = BGC_CARRY_FIXTURE_DIR) -> dict[str, dict[str, int]]:
    """The brine fraction's carry-through cases (``BGC_CARRY_CASES`` and
    ``BGC_CARRY_COUPLED_CASES`` by default), each measured as its stage's
    own (:func:`measure_itd`, :func:`measure_ridge`, :func:`measure_step`,
    :func:`measure_coupled`)."""
    reference = load_fixture(fixture_dir) if reference is None else reference
    logs = load_logs(fixture_dir) if logs is None else logs
    by_stage = {_ITD: measure_itd, _RDG: measure_ridge, _STEP: measure_step, _SEAICE_STEP: measure_coupled}
    report: dict[str, dict[str, int]] = {}
    for stage, measure_stage in by_stage.items():
        mine = tuple(row for row in rows if row[2] == stage)
        if mine:
            report.update(measure_stage(reference, logs, rows=mine, fixture_dir=fixture_dir))
    return report


def measure_vt(reference: dict | None = None, logs: dict | None = None, *, rows=VT_CASES,
               fixture_dir: Path = VT_FIXTURE_DIR) -> dict[str, dict[str, int]]:
    """Per vertical-thermodynamics case: the words the GPU gets wrong per
    field (an aborting case's too), whether its abort (ierr, first
    aborting cell, message) differs from the oracle's, and the tracer
    object's integers that differ."""
    reference = load_fixture(fixture_dir) if reference is None else reference
    logs = load_logs(fixture_dir) if logs is None else logs
    report: dict[str, dict[str, int]] = {}
    for case in cases("r8", rows):
        got, state = run_gpu(case)
        log = logs[case["name"]]
        oracle = oracle_abort(log)
        row = {"abort": int(tuple(state.abort) != oracle)}
        # an aborting case's too: the cells after the abort keep their words
        want = expected_outputs(case, reference.get(case["name"], {}))
        row.update({key: word_mismatches(got[key], want[key]) for key in want})
        mine = state.tracers.log_rows()
        row["tracer_object"] = sum(int(mine[k] != log["tracer_object"].get(k)) for k in mine)
        report[case["name"]] = row
    return report


def measure_itd(reference: dict | None = None, logs: dict | None = None, *, rows=ITD_CASES,
                fixture_dir: Path = ITD_FIXTURE_DIR) -> dict[str, dict[str, int]]:
    """Per ITD-thermodynamics case: the words the GPU gets wrong per field
    (an aborting case's too: the cells after the abort keep their words),
    whether its abort (ierr, first aborting cell, message) differs from the
    oracle's, whether the number of zap_snow_temperature warnings does, and
    the tracer object's integers that differ."""
    reference = load_fixture(fixture_dir) if reference is None else reference
    logs = load_logs(fixture_dir) if logs is None else logs
    report: dict[str, dict[str, int]] = {}
    for case in cases("r8", rows):
        got, state = run_gpu(case)
        log = logs[case["name"]]
        row = {"abort": int(tuple(state.abort) != oracle_abort(log, "column_itd_thermodynamics")),
               "warnings": int(state.warnings != log["log"].count(citd.ZAP_WARNING))}
        want = expected_outputs(case, reference.get(case["name"], {}))
        row.update({key: word_mismatches(got[key], want[key]) for key in want})
        mine = state.tracers.log_rows()
        row["tracer_object"] = sum(int(mine[k] != log["tracer_object"].get(k)) for k in mine)
        report[case["name"]] = row
    return report


def oracle_ridge_warnings(log: dict) -> list[str]:
    """The oracle's warning lines :func:`woof.hex.mpas_seaice.column_ridge_v842.ridging`
    grades (``state.warnings``): ridge_ice's ``Repeat ridging`` lines and
    zap_snow_temperature's out-of-bounds lines, in log order."""
    return [line for line in log["log"]
            if line.startswith("WARN  Repeat ridging, niter =") or line == citd.ZAP_WARNING]


def measure_ridge(reference: dict | None = None, logs: dict | None = None, *, rows=RIDGE_CASES,
                  fixture_dir: Path = RIDGE_FIXTURE_DIR) -> dict[str, dict[str, int]]:
    """Per ridging case: the words the GPU gets wrong per field (an aborting
    case's too), whether its abort (ierr, first aborting cell, message)
    differs from the oracle's, whether its graded warning lines do, and the
    tracer object's integers that differ; for a case v8.4.2 dies on, whether
    the port fails to stop with the same runtime error."""
    reference = load_fixture(fixture_dir) if reference is None else reference
    logs = load_logs(fixture_dir) if logs is None else logs
    report: dict[str, dict[str, int]] = {}
    for case in cases("r8", rows):
        log = logs[case["name"]]
        if "crash" in log:
            # v8.4.2 dies: the port must stop with the same runtime error, nothing to compare
            try:
                run_gpu(case)
            except crdg.FortranRuntimeError as error:
                report[case["name"]] = {"crash": int(error.crash != log["crash"])}
            else:
                report[case["name"]] = {"crash": 1}
            continue
        got, state = run_gpu(case)
        row = {"abort": int(tuple(state.abort) != oracle_abort(log, "column_ridging")),
               "warnings": int(state.warnings != oracle_ridge_warnings(log))}
        want = expected_outputs(case, reference.get(case["name"], {}))
        row.update({key: word_mismatches(got[key], want[key]) for key in want})
        mine = state.tracers.log_rows()
        row["tracer_object"] = sum(int(mine[k] != log["tracer_object"].get(k)) for k in mine)
        report[case["name"]] = row
    return report


def measure_radiation(reference: dict | None = None, logs: dict | None = None, *, rows=RADIATION_CASES,
                      fixture_dir: Path = RADIATION_FIXTURE_DIR) -> dict[str, dict[str, int]]:
    """Per shortwave case and field, the words the GPU gets wrong, and the
    tracer object's integers that differ from the oracle's."""
    return measure(reference, logs, rows=rows, fixture_dir=fixture_dir)


def measure_dedd(reference: dict | None = None, logs: dict | None = None, *, rows=DEDD_CASES,
                 fixture_dir: Path = DEDD_FIXTURE_DIR) -> dict[str, dict[str, int]]:
    """Per delta-Eddington case and field, the words the GPU gets wrong, and
    the tracer object's integers that differ from the oracle's."""
    return measure(reference, logs, rows=rows, fixture_dir=fixture_dir)


#: colpkg_init_thermo's profiles are graded for every ice-layer count up to this
THERMO_LAYERS = cv.MAX_ICE_LAYERS


def oracle_thermo_profiles(build_dir: Path, n_max: int = THERMO_LAYERS) -> np.ndarray:
    """``run_init_thermo_r8``'s profiles for 1..n_max ice layers, concatenated."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "thermo.bin"
        subprocess.run([str(Path(build_dir) / "run_init_thermo_r8"), str(n_max), str(out)], check=True)
        return np.frombuffer(out.read_bytes(), "<f8").astype(np.float64)


def gpu_thermo_profiles(n_max: int = THERMO_LAYERS) -> np.ndarray:
    """:mod:`~woof.hex.mpas_seaice.column_v842`'s ``colpkg_init_thermo`` (BL99) for 1..n_max layers."""
    return np.concatenate([cv.init_thermo_profile(n, ktherm=1) for n in range(1, n_max + 1)])


def thermo_profile_mismatches(want: np.ndarray, n_max: int = THERMO_LAYERS) -> dict[int, int]:
    """Per ice-layer count, the profile words the GPU gets wrong."""
    got = gpu_thermo_profiles(n_max)
    out, off = {}, 0
    for n in range(1, n_max + 1):
        out[n] = word_mismatches(got[off:off + n + 1], want[off:off + n + 1])
        off += n + 1
    return out


def load_thermo_profiles(fixture_dir: Path = FIXTURE_DIR) -> np.ndarray:
    return np.load(fixture_dir / "init-thermo-r8.npy")


def oracle_step_abort(log: dict) -> tuple[int, int | None, str | None]:
    """The oracle's column step outcome as :func:`woof.hex.mpas_seaice.column_step_v842.run`
    returns it: the ierr, and for an abort the first aborting cell's
    ``indexToCellID`` and the stage's logged error."""
    ierr = log["tracer_object"]["IERR"][0]
    if not ierr:
        return 0, None, None
    lines = log["log"]
    message = next(line for line in lines if line.startswith("ERR column_"))[len("ERR "):]
    cell = int(lines[lines.index("ERR iCell: $i") + 1].split()[1])
    return ierr, cell, message


def measure_step(reference: dict | None = None, logs: dict | None = None, *, rows=STEP_CASES,
                 fixture_dir: Path = STEP_FIXTURE_DIR) -> dict[str, dict[str, int]]:
    """Per column-step case: the words the GPU gets wrong per field (an
    aborting case's too), whether its abort (ierr, first aborting cell,
    message) differs from the oracle's, whether its graded warning lines
    (the ITD's and the ridging's) do, and the tracer object's integers that
    differ."""
    reference = load_fixture(fixture_dir) if reference is None else reference
    logs = load_logs(fixture_dir) if logs is None else logs
    report: dict[str, dict[str, int]] = {}
    for case in cases("r8", rows):
        log = logs[case["name"]]
        got, state = run_gpu(case)
        row = {"abort": int(tuple(state.abort) != oracle_step_abort(log)),
               "warnings": int(state.warnings != oracle_ridge_warnings(log))}
        want = expected_outputs(case, reference.get(case["name"], {}))
        row.update({key: word_mismatches(got[key], want[key]) for key in want})
        mine = state.tracers.log_rows()
        row["tracer_object"] = sum(int(mine[k] != log["tracer_object"].get(k)) for k in mine)
        report[case["name"]] = row
    return report
