! Drive MPAS-Ocean v8.4.2's ocn_diagnostic_solve on one block and write every
! field word back: the reference the GPU port is held against.
!
!   run_diagnostics INPUT.bin OUTPUT.bin
!
! INPUT.bin is written by woof.verify.mpas_ocean_diagnostics_oracle in
! oracle_stream.F90's self-describing layout.  The driver makes the calls the
! model makes:
!
!   ocn_constants_init(configs, packages)   rho_sw = config_density0
!   ocn_equation_of_state_init(err)
!   ocn_diagnostics_init(err)               ke_vertex_flag from config_include_KE_vertex
!   ocn_diagnostic_solve(dt, state, forcing, mesh, diagnostics, scratch, tracers, timeLevelIn=1)
!
! OUTPUT.bin holds every real and integer field's words in the input's order.
! Scratch fields are allocated by the routine and filled with huge() by the
! stub, so a word read before it is written would show.

program run_diagnostics
  use mpas_kind_types
  use mpas_derived_types
  use mpas_pool_routines
  use ocn_constants
  use ocn_equation_of_state
  use ocn_diagnostics
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
  call ocn_equation_of_state_init(err)
  if (err /= 0) stop 13
  call ocn_diagnostics_init(err)
  if (err /= 0) stop 13
  call ocn_diagnostic_solve(dt, pool_named(pools, 'state'), pool_named(pools, 'forcing'), &
       pool_named(pools, 'mesh'), pool_named(pools, 'diagnostics'), pool_named(pools, 'scratch'), &
       pool_named(pools, 'tracers'), timeLevelIn=1)
  call write_stream(outPath, fields)
end program run_diagnostics
