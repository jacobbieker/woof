! Drive MPAS-Ocean v8.4.2's ocn_vert_transport_velocity_top (with
! ocn_ALE_thickness) on one block and write every field word back: the
! reference the GPU port is held against.
!
!   run_vert_transport INPUT.bin OUTPUT.bin
!
! INPUT.bin is written by woof.verify.mpas_ocean_vert_transport_oracle in
! oracle_stream.F90's layout.  The routine is cut verbatim out of
! mpas_ocn_diagnostics.F (diagnostics_cut_head.F90); mpas_ocn_thick_ale.F is
! compiled verbatim.  The driver makes the call the split-explicit step
! makes, on the state pool's layerThickness, normalVelocity and ssh and the
! diagnostics' layerThicknessEdge:
!
!   ocn_vert_transport_velocity_top(mesh, verticalMesh, scratch, layerThickness, layerThicknessEdge,
!                                   normalVelocity, ssh, dt, vertAleTransportTop, err)

program run_vert_transport
  use mpas_kind_types
  use mpas_derived_types
  use mpas_pool_routines
  use ocn_constants
  use ocn_diagnostics
  use oracle_stream
  implicit none
  type(oracle_pools) :: pools
  type(oracle_field), allocatable :: fields(:)
  real(kind=RKIND) :: dt
  integer :: err
  real(kind=RKIND), dimension(:,:), pointer :: h, hEdge, u, w
  real(kind=RKIND), dimension(:), pointer :: ssh
  character(len=1024) :: inPath, outPath

  call get_command_argument(1, inPath)
  call get_command_argument(2, outPath)
  call read_stream(inPath, dt, pools, fields)
  call ocn_constants_init(pools % configs, pools % packages)
  call mpas_pool_get_array(pool_named(pools, 'state'), 'layerThickness', h)
  call mpas_pool_get_array(pool_named(pools, 'state'), 'normalVelocity', u)
  call mpas_pool_get_array(pool_named(pools, 'state'), 'ssh', ssh)
  call mpas_pool_get_array(pool_named(pools, 'diagnostics'), 'layerThicknessEdge', hEdge)
  call mpas_pool_get_array(pool_named(pools, 'diagnostics'), 'vertAleTransportTop', w)
  call ocn_vert_transport_velocity_top(pool_named(pools, 'mesh'), pool_named(pools, 'verticalMesh'), &
       pool_named(pools, 'scratch'), h, hEdge, u, ssh, dt, w, err)
  if (err /= 0) stop 9
  call write_stream(outPath, fields)
end program run_vert_transport
