! Drive MPAS-Ocean v8.4.2's tracer advection (ocn_tracer_advection_tend:
! the standard or the monotone scheme) on one block and write every field
! word back: the reference the GPU port is held against.
!
!   run_tracer_advection INPUT.bin OUTPUT.bin
!
! INPUT.bin is written by woof.verify.mpas_ocean_tracer_advection_oracle in
! oracle_stream.F90's self-describing layout.  The tracer group is the
! tracers pool's activeTracers, its tendency the tracersTend pool's
! activeTracersTend; the thickness flux and the vertical transport are the
! diagnostics pool's normalThicknessFlux and vertAleTransportTop.  The driver
! makes the calls the model makes:
!
!   ocn_constants_init(configs, packages)
!   ocn_tracer_advection_init(err)     the orders, coef_3rd_order, config_monotonic
!   ocn_tracer_advection_tend(activeTracers, normalThicknessFlux, vertAleTransportTop, layerThickness, dt,
!                             mesh, scratch, diagnostics, activeTracersTend, 'activeTracers')
!
! OUTPUT.bin holds every real and integer field's words in the input's order.

program run_tracer_advection
  use mpas_kind_types
  use mpas_derived_types
  use mpas_pool_routines
  use ocn_constants
  use ocn_tracer_advection
  use oracle_stream
  implicit none
  type(oracle_pools) :: pools
  type(oracle_field), allocatable :: fields(:)
  real(kind=RKIND) :: dt
  integer :: err
  real(kind=RKIND), dimension(:,:,:), pointer :: tracers, tend
  real(kind=RKIND), dimension(:,:), pointer :: flux, w, h
  character(len=1024) :: inPath, outPath

  call get_command_argument(1, inPath)
  call get_command_argument(2, outPath)
  call read_stream(inPath, dt, pools, fields)
  call ocn_constants_init(pools % configs, pools % packages)
  err = 0
  call ocn_tracer_advection_init(err)
  if (err /= 0) stop 13
  call mpas_pool_get_array(pool_named(pools, 'tracers'), 'activeTracers', tracers)
  call mpas_pool_get_array(pool_named(pools, 'tracersTend'), 'activeTracersTend', tend)
  call mpas_pool_get_array(pool_named(pools, 'diagnostics'), 'normalThicknessFlux', flux)
  call mpas_pool_get_array(pool_named(pools, 'diagnostics'), 'vertAleTransportTop', w)
  call mpas_pool_get_array(pool_named(pools, 'state'), 'layerThickness', h)
  call ocn_tracer_advection_tend(tracers, flux, w, h, dt, pool_named(pools, 'mesh'), pool_named(pools, 'scratch'), &
       pool_named(pools, 'diagnostics'), tend, 'activeTracers')
  call write_stream(outPath, fields)
end program run_tracer_advection
