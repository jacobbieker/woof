! The head of the module the sea-ice time-step oracle cuts out of
! mpas_seaice_column.F (build.sh): the time-integration wrappers and the
! re-initialisations seaice_timestep calls, verbatim.  Without the column
! package (config_use_column_package = .false., the oracle's runs) they touch
! only the atmospheric reference fields; the column routines they would call
! otherwise are stand-ins after them (column_cut_tail.F90) that stop by name.
module seaice_column
  use mpas_derived_types
  use mpas_pool_routines
  use mpas_timekeeping
  use mpas_timer
  use mpas_log, only: mpas_log_write
  implicit none
  private
  public :: seaice_column_predynamics_time_integration, seaice_column_dynamics_time_integration, &
            seaice_column_postdynamics_time_integration, seaice_column_reinitialize_fluxes, &
            seaice_column_reinitialize_diagnostics_thermodynamics, seaice_column_reinitialize_diagnostics_bgc, &
            seaice_column_reinitialize_diagnostics_dynamics
contains
