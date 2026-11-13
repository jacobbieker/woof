! Drive MPAS-Ocean v8.4.2's split-explicit barotropic stage (stage 2 of
! ocn_time_integrator_split) on one block and write every field word back:
! the reference the GPU port is held against.
!
!   run_btr_subcycle INPUT.bin OUTPUT.bin
!
! INPUT.bin is written by woof.verify.mpas_ocean_btr_subcycle_oracle in
! oracle_stream.F90's layout; a field's time level 2 is the entry NAME@2.
! The stage is cut verbatim out of mpas_ocn_time_integration_split.F
! (btr_subcycle_cut_head.F90).  The driver builds the one block the stage
! walks (its structs the state, diagnostics, mesh, tend and scratch pools,
! its dimensions the mesh pool's), sets nBtrSubcycles from the integer
! config 'nBtrSubcycles' and calls the stage.

program run_btr_subcycle
  use mpas_kind_types
  use mpas_derived_types
  use mpas_pool_routines
  use ocn_btr_stage_cut
  use oracle_stream
  implicit none
  type(oracle_pools) :: pools
  type(oracle_field), allocatable :: fields(:)
  type(domain_type), pointer :: domain
  real(kind=RKIND) :: dt
  integer, pointer :: n
  character(len=1024) :: inPath, outPath

  call get_command_argument(1, inPath)
  call get_command_argument(2, outPath)
  call read_stream(inPath, dt, pools, fields)
  allocate(domain)
  allocate(domain % blocklist, domain % blocklist % structs)
  domain % configs => pools % configs
  domain % blocklist % configs => pools % configs
  domain % blocklist % dimensions => pool_named(pools, 'mesh')
  call stub_pool_put_subpool(domain % blocklist % structs, 'state', pool_named(pools, 'state'))
  call stub_pool_put_subpool(domain % blocklist % structs, 'diagnostics', pool_named(pools, 'diagnostics'))
  call stub_pool_put_subpool(domain % blocklist % structs, 'mesh', pool_named(pools, 'mesh'))
  call stub_pool_put_subpool(domain % blocklist % structs, 'tend', pool_named(pools, 'tend'))
  call stub_pool_put_subpool(domain % blocklist % structs, 'scratch', pool_named(pools, 'scratch'))
  call mpas_pool_get_config(pools % configs, 'nBtrSubcycles', n)
  nBtrSubcycles = n
  call ocn_btr_stage(domain, dt)
  call write_stream(outPath, fields)
end program run_btr_subcycle
