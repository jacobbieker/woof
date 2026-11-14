! The head of the module the sea-ice time-step oracle cuts out of
! mpas_seaice_time_integration.F (build.sh): seaice_timestep,
! store_timestep_counter and seaice_timestep_finalize, verbatim (their USE
! statements are inside them).
module seaice_time_integration
  use mpas_derived_types
  use mpas_pool_routines
  use mpas_timekeeping
  use mpas_timer
  implicit none
  private
  save
  public :: seaice_timestep, seaice_timestep_finalize
contains
