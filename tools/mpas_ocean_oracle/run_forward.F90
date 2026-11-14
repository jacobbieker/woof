! Drive MPAS-Ocean v8.4.2's forward mode over N time steps on one block and
! write every field word back: the reference the GPU forward driver is held
! against.
!
!   run_forward INPUT.bin OUTPUT.bin
!
! INPUT.bin is written by woof.verify.mpas_ocean_forward_oracle in
! oracle_stream.F90's layout (time level 2 is NAME@2; xtime and
! simulationStartTime are the diagnostics pool's character fields).  The
! driver follows ocn_forward_mode_init and ocn_forward_mode_run
! (mpas_ocn_forward_mode.F) for what changes the state:
!
!   the _init routine of every module the step reaches, ocn_timestep_init,
!   ocn_time_integration_split_init (verbatim: the barotropic subcycle count
!   from config_dt and config_btr_dt, the cold start's barotropic velocity);
!   the state part of ocn_init_routines_block (mpas_ocn_init_routines.F,
!   transcribed below: ocn_diagnostic_solve, the missing cell's and the
!   land levels' sentinels, normalTransportVelocity, the time levels
!   initialised); the mesh part (signs, derivTwo, advection coefficients,
!   mesh scaling, reconstruction coefficients) is the input's;
!   then oracle_n_steps times: the clock advanced, the absorbed fractions
!   (ocn_forcing_build_fraction_absorbed_array), ocn_timestep (verbatim),
!   the state's time levels shifted.
!
! The land-ice and frazil array builders return at once when off (the
! configurations the port refuses otherwise); analysis members, streams and
! ocn_validate_state change no state and are not run.  The time manager is
! stubs.F90's (whole seconds).

program run_forward
  use mpas_kind_types
  use mpas_derived_types
  use mpas_pool_routines
  use mpas_timekeeping
  use ocn_constants
  use ocn_tendency
  use ocn_forcing
  use ocn_surface_bulk_forcing
  use ocn_thick_hadv
  use ocn_thick_vadv
  use ocn_thick_surface_flux
  use ocn_vel_coriolis
  use ocn_vel_vadv
  use ocn_vel_pressure_grad
  use ocn_vel_hmix
  use ocn_vel_forcing
  use ocn_tracer_advection
  use ocn_tracer_hmix
  use ocn_tracer_hmix_redi
  use ocn_tracer_surface_flux_to_tend
  use ocn_tracer_short_wave_absorption
  use ocn_equation_of_state
  use ocn_diagnostics
  use ocn_vmix
  use ocn_split_step_cut
  use ocn_timestep_cut
  use oracle_stream
  implicit none
  type(oracle_pools) :: pools
  type(oracle_field), allocatable :: fields(:)
  type(domain_type), pointer :: domain
  type(MPAS_Time_type) :: currTime
  type(MPAS_TimeInterval_type) :: timeStep
  type(MPAS_pool_type), pointer :: meshPool, statePool, tracersPool, diagnosticsPool, forcingPool
  real(kind=RKIND) :: dt, inputDt
  integer :: err, itimestep, iCell, iEdge
  integer, pointer :: nSteps, nCells, nEdges, nVertLevels
  integer, dimension(:), pointer :: maxLevelCell, maxLevelEdgeTop, maxLevelEdgeBot
  real(kind=RKIND), dimension(:), pointer :: areaCell
  real(kind=RKIND), dimension(:,:), pointer :: layerThickness, normalVelocity, normalTransportVelocity
  real(kind=RKIND), dimension(:,:,:), pointer :: activeTracers
  character(len=StrKIND), pointer :: config_start_time, config_dt
  character(len=StrKIND) :: timeStamp
  character(len=1024) :: inPath, outPath

  call get_command_argument(1, inPath)
  call get_command_argument(2, outPath)
  call read_stream(inPath, inputDt, pools, fields)
  allocate(domain)
  allocate(domain % blocklist)
  allocate(domain % blocklist % structs)
  allocate(domain % clock)
  domain % configs => pools % configs
  domain % blocklist % configs => pools % configs
  domain % blocklist % dimensions => pool_named(pools, 'mesh')
  call stub_pool_put_subpool(domain % blocklist % structs, 'state', pool_named(pools, 'state'))
  call stub_pool_put_subpool(domain % blocklist % structs, 'diagnostics', pool_named(pools, 'diagnostics'))
  call stub_pool_put_subpool(domain % blocklist % structs, 'mesh', pool_named(pools, 'mesh'))
  call stub_pool_put_subpool(domain % blocklist % structs, 'verticalMesh', pool_named(pools, 'verticalMesh'))
  call stub_pool_put_subpool(domain % blocklist % structs, 'tend', pool_named(pools, 'tend'))
  call stub_pool_put_subpool(domain % blocklist % structs, 'scratch', pool_named(pools, 'scratch'))
  call stub_pool_put_subpool(domain % blocklist % structs, 'forcing', pool_named(pools, 'forcing'))
  call stub_pool_put_subpool(domain % blocklist % structs, 'shortwave', pool_named(pools, 'swForcing'))
  meshPool => pool_named(pools, 'mesh')
  statePool => pool_named(pools, 'state')
  tracersPool => pool_named(pools, 'tracers')
  diagnosticsPool => pool_named(pools, 'diagnostics')
  forcingPool => pool_named(pools, 'forcing')

  ! the clock: config_start_time, config_dt; dt as ocn_forward_mode_run takes it
  call mpas_pool_get_config(pools % configs, 'config_start_time', config_start_time)
  call mpas_pool_get_config(pools % configs, 'config_dt', config_dt)
  call mpas_set_clock(domain % clock, config_start_time, config_dt)
  call mpas_set_timeInterval(timeStep, timeString=config_dt)
  call mpas_get_timeInterval(timeStep, dt=dt)

  call ocn_constants_init(pools % configs, pools % packages)
  err = 0
  call ocn_timestep_init(err)
  call ocn_tendency_init(err)
  call ocn_forcing_init(err)
  call ocn_surface_bulk_forcing_init(err)
  call ocn_thick_hadv_init(err)
  call ocn_thick_vadv_init(err)
  call ocn_thick_surface_flux_init(err)
  call ocn_vel_coriolis_init(err)
  call ocn_vel_vadv_init(err)
  call ocn_vel_pressure_grad_init(err)
  call ocn_vel_hmix_init(err)
  call ocn_vel_forcing_init(err)
  if (err /= 0) stop 13
  call ocn_tracer_advection_init(err)
  if (err /= 0) stop 13
  call ocn_tracer_hmix_init(err)
  if (err /= 0) stop 13
  call ocn_tracer_hmix_redi_init(err)
  if (err /= 0) stop 13
  call ocn_tracer_surface_flux_init(err)
  if (err /= 0) stop 13
  call ocn_tracer_short_wave_absorption_init(domain, err)
  if (err /= 0) stop 13
  call ocn_equation_of_state_init(err)
  if (err /= 0) stop 13
  call ocn_diagnostics_init(err)
  if (err /= 0) stop 13
  call ocn_vmix_init(domain, err)
  if (err /= 0) stop 13
  call ocn_time_integration_split_init(domain)

  ! ocn_init_routines_block's state part (mpas_ocn_init_routines.F:621-650,690)
  call ocn_diagnostic_solve(dt, statePool, forcingPool, meshPool, diagnosticsPool, pool_named(pools, 'scratch'), &
       tracersPool)
  call mpas_pool_get_dimension(meshPool, 'nCells', nCells)
  call mpas_pool_get_dimension(meshPool, 'nEdges', nEdges)
  call mpas_pool_get_dimension(meshPool, 'nVertLevels', nVertLevels)
  call mpas_pool_get_array(meshPool, 'areaCell', areaCell)
  call mpas_pool_get_array(meshPool, 'maxLevelCell', maxLevelCell)
  call mpas_pool_get_array(meshPool, 'maxLevelEdgeTop', maxLevelEdgeTop)
  call mpas_pool_get_array(meshPool, 'maxLevelEdgeBot', maxLevelEdgeBot)
  call mpas_pool_get_array(statePool, 'layerThickness', layerThickness, 1)
  call mpas_pool_get_array(statePool, 'normalVelocity', normalVelocity, 1)
  call mpas_pool_get_array(tracersPool, 'activeTracers', activeTracers, 1)
  call mpas_pool_get_array(diagnosticsPool, 'normalTransportVelocity', normalTransportVelocity)
  areaCell(nCells+1) = -1.0e34_RKIND
  layerThickness(:, nCells+1) = 0.0_RKIND
  do iEdge=1, nEdges
     normalVelocity(maxLevelEdgeTop(iEdge)+1:maxLevelEdgeBot(iEdge), iEdge) = 0.0_RKIND
     normalVelocity(maxLevelEdgeBot(iEdge)+1:nVertLevels,iEdge) = -1.0e34_RKIND
  end do
  do iCell=1,nCells
     activeTracers(:, maxLevelCell(iCell)+1:nVertLevels,iCell) =  -1.0e34_RKIND
  end do
  do iEdge = 1, nEdges
     normalTransportVelocity(:, iEdge) = normalVelocity(:, iEdge)
  end do
  call mpas_pool_initialize_time_levels(statePool)

  ! ocn_forward_mode_run's loop
  call mpas_pool_get_config(pools % configs, 'oracle_n_steps', nSteps)
  do itimestep = 1, nSteps
     call mpas_advance_clock(domain % clock)
     currTime = mpas_get_clock_time(domain % clock, MPAS_NOW, err)
     call mpas_get_time(curr_time=currTime, dateTimeString=timeStamp, ierr=err)
     call ocn_forcing_build_fraction_absorbed_array(meshPool, statePool, diagnosticsPool, forcingPool, err, 1)
     call ocn_timestep(domain, dt, timeStamp)
     call mpas_pool_shift_time_levels(statePool)
  end do
  call write_stream(outPath, fields)
end program run_forward
