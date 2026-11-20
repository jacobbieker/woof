! Drive MPAS-Ocean v8.4.2's frazil ice formation on one block and write every
! field word back: the reference the GPU port is held against.
!
!   run_frazil INPUT.bin OUTPUT.bin
!
! INPUT.bin is written by woof.verify.mpas_ocean_frazil_oracle in
! oracle_stream.F90's layout (time level 2 is NAME@2).  mpas_ocn_frazil_forcing.F
! is compiled verbatim with the real equation of state.  The driver makes the
! calls the forward mode and the tendencies make, in their order:
!
!   ocn_frazil_forcing_init(err)
!   ocn_frazil_forcing_build_arrays(domain, mesh, forcing, diagnostics, state, err)
!     (dt from the clock: config_start_time, config_dt)
!   ocn_frazil_forcing_layer_thickness(mesh, forcing, tend % layerThickness, err)
!   ocn_frazil_forcing_tracers(mesh, tracers, 'activeTracers', forcing,
!                              tracersTend % activeTracersTend, err)

program run_frazil
  use mpas_kind_types
  use mpas_derived_types
  use mpas_pool_routines
  use mpas_timekeeping
  use ocn_constants
  use ocn_equation_of_state
  use ocn_frazil_forcing
  use oracle_stream
  implicit none
  type(oracle_pools) :: pools
  type(oracle_field), allocatable :: fields(:)
  type(domain_type), pointer :: domain
  type(MPAS_pool_type), pointer :: meshPool, statePool, diagnosticsPool, forcingPool
  real(kind=RKIND) :: inputDt
  integer :: err
  real(kind=RKIND), dimension(:,:), pointer :: tendThickness
  real(kind=RKIND), dimension(:,:,:), pointer :: tendTracers
  character(len=StrKIND), pointer :: config_start_time, config_dt, config_calendar_type
  character(len=1024) :: inPath, outPath

  call get_command_argument(1, inPath)
  call get_command_argument(2, outPath)
  call read_stream(inPath, inputDt, pools, fields)
  allocate(domain)
  allocate(domain % clock)
  domain % configs => pools % configs
  meshPool => pool_named(pools, 'mesh')
  statePool => pool_named(pools, 'state')
  diagnosticsPool => pool_named(pools, 'diagnostics')
  forcingPool => pool_named(pools, 'forcing')
  ! the framework sets the calendar before the clock (mpas_framework.F:111-116)
  call mpas_pool_get_config(pools % configs, 'config_calendar_type', config_calendar_type)
  call mpas_timekeeping_init(config_calendar_type)
  call mpas_pool_get_config(pools % configs, 'config_start_time', config_start_time)
  call mpas_pool_get_config(pools % configs, 'config_dt', config_dt)
  call mpas_set_clock(domain % clock, config_start_time, config_dt)

  call ocn_constants_init(pools % configs, pools % packages)
  call ocn_equation_of_state_init(err)
  if (err /= 0) stop 13
  call ocn_frazil_forcing_init(err)
  if (err /= 0) stop 13
  call ocn_frazil_forcing_build_arrays(domain, meshPool, forcingPool, diagnosticsPool, statePool, err)
  if (err /= 0) stop 9
  call mpas_pool_get_array(pool_named(pools, 'tend'), 'layerThickness', tendThickness)
  call ocn_frazil_forcing_layer_thickness(meshPool, forcingPool, tendThickness, err)
  if (err /= 0) stop 9
  call mpas_pool_get_array(pool_named(pools, 'tracersTend'), 'activeTracersTend', tendTracers)
  call ocn_frazil_forcing_tracers(meshPool, pool_named(pools, 'tracers'), 'activeTracers', forcingPool, tendTracers, &
       err)
  if (err /= 0) stop 9
  call write_stream(outPath, fields)
end program run_frazil
