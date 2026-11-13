! Drive MPAS-Ocean v8.4.2's horizontal tracer mixing (ocn_tracer_hmix_tend:
! del2, del4, and Redi, which is off here) on one block and write every field
! word back: the reference the GPU port is held against.
!
!   run_tracer_hmix INPUT.bin OUTPUT.bin
!
! INPUT.bin is written by woof.verify.mpas_ocean_tracer_hmix_oracle in
! oracle_stream.F90's layout.  The driver makes the calls the model makes
! (ocn_forward_mode_init calls the two _init routines):
!
!   ocn_constants_init(configs, packages)
!   ocn_tracer_hmix_init(err)        del2 and del4 from their configs
!   ocn_tracer_hmix_redi_init(err)   Redi from config_use_standardGM
!   ocn_tracer_hmix_tend(mesh, scratch, layerThicknessEdge, zMid, activeTracers, relativeSlopeTopOfEdge,
!                        relativeSlopeTapering, relativeSlopeTaperingCell, activeTracersTend, err)

program run_tracer_hmix
  use mpas_kind_types
  use mpas_derived_types
  use mpas_pool_routines
  use ocn_constants
  use ocn_tracer_hmix
  use ocn_tracer_hmix_redi
  use oracle_stream
  implicit none
  type(oracle_pools) :: pools
  type(oracle_field), allocatable :: fields(:)
  real(kind=RKIND) :: dt
  integer :: err
  real(kind=RKIND), dimension(:,:,:), pointer :: tracers, tend
  real(kind=RKIND), dimension(:,:), pointer :: hEdge, zMid, slope, tapering, taperingCell
  character(len=1024) :: inPath, outPath

  call get_command_argument(1, inPath)
  call get_command_argument(2, outPath)
  call read_stream(inPath, dt, pools, fields)
  call ocn_constants_init(pools % configs, pools % packages)
  call ocn_tracer_hmix_init(err)
  if (err /= 0) stop 13
  call ocn_tracer_hmix_redi_init(err)
  if (err /= 0) stop 13
  call mpas_pool_get_array(pool_named(pools, 'tracers'), 'activeTracers', tracers)
  call mpas_pool_get_array(pool_named(pools, 'tracersTend'), 'activeTracersTend', tend)
  call mpas_pool_get_array(pool_named(pools, 'diagnostics'), 'layerThicknessEdge', hEdge)
  call mpas_pool_get_array(pool_named(pools, 'diagnostics'), 'zMid', zMid)
  call mpas_pool_get_array(pool_named(pools, 'diagnostics'), 'relativeSlopeTopOfEdge', slope)
  call mpas_pool_get_array(pool_named(pools, 'diagnostics'), 'relativeSlopeTapering', tapering)
  call mpas_pool_get_array(pool_named(pools, 'diagnostics'), 'relativeSlopeTaperingCell', taperingCell)
  call ocn_tracer_hmix_tend(pool_named(pools, 'mesh'), pool_named(pools, 'scratch'), hEdge, zMid, tracers, &
       slope, tapering, taperingCell, tend, err)
  if (err /= 0) stop 9
  call write_stream(outPath, fields)
end program run_tracer_hmix
