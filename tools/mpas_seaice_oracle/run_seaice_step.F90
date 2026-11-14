! Drive MPAS-Seaice v8.4.2's forward model over N time steps on one block and
! write every field word back: the reference the GPU time-step driver is
! held against.
!
!   run_seaice_step INPUT.bin OUTPUT.bin
!
! INPUT.bin is written by woof.verify.mpas_seaice_step_oracle in
! tools/mpas_ocean_oracle/oracle_stream.F90's layout (the shared framework
! stand-ins are tools/mpas_ocean_oracle/stubs.F90).  The driver follows
! seaice_core_init and seaice_core_run (mpas_seaice_core.F) for what changes
! the state:
!
!   init: the velocity solver's time steps (seaice_init_velocity_solver:
!   dynamicsTimeStep = config_dt / config_dynamics_subcycle_number,
!   elasticTimeStep = dynamicsTimeStep / config_elastic_subcycle_number),
!   seaice_init_evp (verbatim) and seaice_initialize_time_diagnostics (cut
!   verbatim), seaice_init_advection (verbatim: under config_use_advection the
!   incremental remap's init, with coeffs_reconstruct the input's,
!   remap_standins.F90); the basis functions, masks and the rest of the mesh
!   setup are the input's;
!   then oracle_n_steps times: the clock advanced, seaice_timestep (cut
!   verbatim: the time diagnostics, the step counter, the column wrappers, the
!   dynamics subcycles with seaice_run_velocity_solver and, under
!   config_use_advection, seaice_run_advection, both compiled verbatim) and
!   seaice_timestep_finalize.
!
! Forcing (config_use_forcing) and the column package are off, and advection is
! incremental remap or off (upwind cannot run in v8.4.2): the GPU driver
! refuses the rest by name.  Analysis members, streams and
! seaice_check_critical_error change no state and are not run.  The time
! manager is the stubs' (whole seconds).

program run_seaice_step
  use mpas_kind_types
  use mpas_derived_types
  use mpas_pool_routines
  use mpas_timekeeping
  use seaice_velocity_solver_constitutive_relation, only: seaice_init_evp
  use seaice_diagnostics, only: seaice_initialize_time_diagnostics
  use seaice_advection, only: seaice_init_advection
  use seaice_time_integration
  use oracle_stream
  implicit none
  type(oracle_pools) :: pools
  type(oracle_field), allocatable :: fields(:)
  type(domain_type), pointer :: domain
  real(kind=RKIND) :: inputDt
  real(kind=RKIND), pointer :: config_dt, dynamicsTimeStep, elasticTimeStep
  integer, pointer :: nSteps, config_dynamics_subcycle_number, config_elastic_subcycle_number
  character(len=StrKIND), pointer :: config_start_time, clockStep
  integer :: itimestep, ierr, p
  character(len=24), parameter :: structNames(24) = [character(len=24) :: 'mesh', 'boundary', 'velocity_solver', &
       'velocity_variational', 'velocity_weak', 'icestate', 'tracers', 'tracers_aggregate', 'atmos_coupling', &
       'ocean_coupling', 'ridging', 'diagnostics', 'scratch', 'tracer_tendencies', 'tracer_edge_fluxes', &
       'tracer_conservation', 'incremental_remap', 'rotated_mesh', 'tracer_masks', 'tracer_products', &
       'tracer_triangles', 'tracer_barycenter', 'tracer_reconstruction', 'tracer_monotonicity']
  character(len=1024) :: inPath, outPath

  call get_command_argument(1, inPath)
  call get_command_argument(2, outPath)
  call read_stream(inPath, inputDt, pools, fields)
  allocate(domain)
  allocate(domain % blocklist)
  allocate(domain % blocklist % structs)
  allocate(domain % clock)
  allocate(domain % dminfo)
  domain % configs => pools % configs
  domain % blocklist % configs => pools % configs
  domain % blocklist % dimensions => pool_named(pools, 'mesh')
  domain % blocklist % packages => pools % packages
  do p = 1, size(structNames)
     call stub_pool_put_subpool(domain % blocklist % structs, trim(structNames(p)), &
          pool_named(pools, trim(structNames(p))))
  end do

  call mpas_pool_get_config(pools % configs, 'config_start_time', config_start_time)
  call mpas_pool_get_config(pools % configs, 'oracle_clock_step', clockStep)
  call mpas_set_clock(domain % clock, config_start_time, clockStep)

  ! seaice_init_velocity_solver's time steps (mpas_seaice_velocity_solver.F:126-127)
  call mpas_pool_get_config(pools % configs, 'config_dt', config_dt)
  call mpas_pool_get_config(pools % configs, 'config_dynamics_subcycle_number', config_dynamics_subcycle_number)
  call mpas_pool_get_config(pools % configs, 'config_elastic_subcycle_number', config_elastic_subcycle_number)
  call mpas_pool_get_array(pool_named(pools, 'velocity_solver'), 'dynamicsTimeStep', dynamicsTimeStep)
  call mpas_pool_get_array(pool_named(pools, 'velocity_solver'), 'elasticTimeStep', elasticTimeStep)
  dynamicsTimeStep = config_dt / real(config_dynamics_subcycle_number,RKIND)
  elasticTimeStep = dynamicsTimeStep / real(config_elastic_subcycle_number,RKIND)
  call seaice_init_evp(domain)
  call seaice_init_advection(domain)
  call seaice_initialize_time_diagnostics(domain)

  ! seaice_core_run's loop
  call mpas_pool_get_config(pools % configs, 'oracle_n_steps', nSteps)
  do itimestep = 1, nSteps
     call mpas_advance_clock(domain % clock)
     ierr = 0
     call seaice_timestep(domain, domain % clock, itimestep, ierr)
     if (ierr /= 0) stop 9
     call seaice_timestep_finalize(domain)
  end do
  call write_stream(outPath, fields)
end program run_seaice_step
