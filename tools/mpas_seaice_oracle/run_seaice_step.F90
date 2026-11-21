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
!   rbf_standins.F90); under the config oracle_compute_basis (present at
!   all) the velocity solver's set-up after seaice_init_evp, as
!   seaice_init_velocity_solver runs it (verbatim): the variational basis,
!   or under config_stress_divergence_scheme = 'weak' the weak form's normal
!   vectors; else they are the input's; the masks and the rest of the mesh setup are the input's;
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
  use seaice_velocity_solver_variational, only: seaice_init_velocity_solver_variational
  use seaice_velocity_solver_weak, only: seaice_init_velocity_solver_weak
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
  character(len=StrKIND), pointer :: config_start_time, clockStep, config_calendar_type
  integer :: itimestep, ierr, p
  logical, pointer :: config_rotate_cartesian_grid, config_include_metric_terms
  character(len=StrKIND), pointer :: config_variational_basis, config_wachspress_integration_type
  character(len=StrKIND), pointer :: config_stress_divergence_scheme
  integer, pointer :: config_wachspress_integration_order
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

  ! the framework sets the calendar before the clock (mpas_framework.F:111-116)
  call mpas_pool_get_config(pools % configs, 'config_calendar_type', config_calendar_type)
  call mpas_timekeeping_init(config_calendar_type)
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
  ! under oracle_compute_basis, seaice_init_velocity_solver's basis set-up
  ! (mpas_seaice_velocity_solver.F:160-171, compiled verbatim), as v8.4.2 always
  ! runs it; without it the basis is the input's
  if (stub_pool_has(pools % configs, 'oracle_compute_basis')) then
     call mpas_pool_get_config(pools % configs, 'config_rotate_cartesian_grid', config_rotate_cartesian_grid)
     call mpas_pool_get_config(pools % configs, 'config_include_metric_terms', config_include_metric_terms)
     call mpas_pool_get_config(pools % configs, 'config_variational_basis', config_variational_basis)
     call mpas_pool_get_config(pools % configs, 'config_wachspress_integration_type', &
          config_wachspress_integration_type)
     call mpas_pool_get_config(pools % configs, 'config_wachspress_integration_order', &
          config_wachspress_integration_order)
     call mpas_pool_get_config(pools % configs, 'config_stress_divergence_scheme', &
          config_stress_divergence_scheme)
     if (trim(config_stress_divergence_scheme) == "weak") then
        call seaice_init_velocity_solver_weak(pool_named(pools, 'mesh'), pool_named(pools, 'boundary'), &
             pool_named(pools, 'velocity_weak'), config_rotate_cartesian_grid)
     else if (trim(config_stress_divergence_scheme) == "variational") then
        call seaice_init_velocity_solver_variational(pool_named(pools, 'mesh'), &
             pool_named(pools, 'velocity_variational'), pool_named(pools, 'boundary'), &
             config_rotate_cartesian_grid, config_include_metric_terms, config_variational_basis, &
             config_wachspress_integration_type, config_wachspress_integration_order)
     end if
  end if
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
