! Drive MPAS-Ocean v8.4.2's Gent-McWilliams bolus velocity on one block and
! write every field word back: the reference the GPU port is held against.
!
!   run_gm INPUT.bin OUTPUT.bin
!
! INPUT.bin is written by woof.verify.mpas_ocean_gm_oracle in
! oracle_stream.F90's layout.  mpas_ocn_gm.F is compiled verbatim (at RKIND 4
! with -fno-range-check: it assigns huge(0D0) to its RKIND scratch fields).
! The driver makes the calls the model makes (ocn_forward_mode_init calls
! ocn_gm_init; the split step and ocn_init_routines_block call the bolus
! velocity under config_use_standardGM):
!
!   ocn_constants_init(configs, packages)
!   ocn_gm_init(err)
!   ocn_gm_compute_Bolus_velocity(diagnostics, mesh, scratch)

program run_gm
  use mpas_kind_types
  use mpas_derived_types
  use mpas_pool_routines
  use ocn_constants
  use ocn_gm
  use oracle_stream
  implicit none
  type(oracle_pools) :: pools
  type(oracle_field), allocatable :: fields(:)
  real(kind=RKIND) :: dt
  integer :: err
  character(len=1024) :: inPath, outPath

  call get_command_argument(1, inPath)
  call get_command_argument(2, outPath)
  call read_stream(inPath, dt, pools, fields)
  call ocn_constants_init(pools % configs, pools % packages)
  call ocn_gm_init(err)
  if (err /= 0) stop 13
  call ocn_gm_compute_Bolus_velocity(pool_named(pools, 'diagnostics'), pool_named(pools, 'mesh'), &
       pool_named(pools, 'scratch'))
  call write_stream(outPath, fields)
end program run_gm
