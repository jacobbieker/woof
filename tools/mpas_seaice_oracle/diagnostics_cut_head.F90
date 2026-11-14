! The head of the module the sea-ice time-step oracle cuts out of
! mpas_seaice_diagnostics.F (build.sh): the time diagnostics, the state check
! and the load-balance timers, verbatim, after the module's USE list.
module seaice_diagnostics
  use mpas_derived_types
  use mpas_pool_routines
  use mpas_timekeeping
  use mpas_io_units
  use mpas_log, only: mpas_log_write
  implicit none
  private
  save
  public :: seaice_initialize_time_diagnostics, seaice_set_time_diagnostics, seaice_check_state, &
            seaice_load_balance_timers
contains
