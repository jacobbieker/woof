! Drive MPAS-Ocean v8.4.2's implicit vertical mixing (ocn_vmix_implicit) on
! one block and write every field word back: the reference the GPU port is
! held against.
!
!   run_vmix INPUT.bin OUTPUT.bin
!
! INPUT.bin is written by woof.verify.mpas_ocean_vmix_oracle in
! oracle_stream.F90's layout.  mpas_ocn_vmix.F and its coefficient modules
! (const, tanh, rich, Redi) are compiled verbatim, with the real equation of
! state; stubs_vmix.F90 stands in for CVMix (nothing when config_use_cvmix
! is off).  The driver makes the calls the model makes:
!
!   ocn_constants_init, ocn_equation_of_state_init, ocn_vmix_init(domain, err)
!   ocn_vmix_implicit(dt, mesh, diagnostics, state, forcing, scratch, err, timeLevelIn=1)

program run_vmix
  use mpas_kind_types
  use mpas_derived_types
  use mpas_pool_routines
  use ocn_constants
  use ocn_equation_of_state
  use ocn_vmix
  use oracle_stream
  implicit none
  type(oracle_pools) :: pools
  type(oracle_field), allocatable :: fields(:)
  type(domain_type) :: domain
  real(kind=RKIND) :: dt
  integer :: err
  character(len=1024) :: inPath, outPath

  call get_command_argument(1, inPath)
  call get_command_argument(2, outPath)
  call read_stream(inPath, dt, pools, fields)
  call ocn_constants_init(pools % configs, pools % packages)
  call ocn_equation_of_state_init(err)
  if (err /= 0) stop 13
  call ocn_vmix_init(domain, err)
  if (err /= 0) stop 13
  call ocn_vmix_implicit(dt, pool_named(pools, 'mesh'), pool_named(pools, 'diagnostics'), pool_named(pools, 'state'), &
       pool_named(pools, 'forcing'), pool_named(pools, 'scratch'), err, timeLevelIn=1)
  if (err /= 0) stop 9
  call write_stream(outPath, fields)
end program run_vmix
