! Drive MPAS-Seaice v8.4.2's coupling preparation on one block and write every
! field word back: the reference the GPU port is held against.
!
!   run_coupling_prep INPUT.bin OUTPUT.bin
!
! INPUT.bin is written by woof.verify.mpas_seaice_coupling_prep_oracle in
! tools/mpas_ocean_oracle/oracle_stream.F90's layout.  seaice_column_coupling_prep
! (which ends by calling seaice_column_scale_fluxes) is cut verbatim out of
! mpas_seaice_column.F (build.sh, coupling_cut_head.F90) and called once, as
! seaice_column_postdynamics_time_integration calls it.

program run_coupling_prep
  use mpas_kind_types
  use mpas_derived_types
  use mpas_pool_routines
  use seaice_column_coupling, only: seaice_column_coupling_prep
  use oracle_stream
  implicit none
  type(oracle_pools) :: pools
  type(oracle_field), allocatable :: fields(:)
  type(domain_type), pointer :: domain
  real(kind=RKIND) :: inputDt
  integer :: p
  character(len=24), parameter :: structNames(12) = [character(len=24) :: 'mesh', 'tracers', &
       'tracers_aggregate', 'diagnostics', 'shortwave', 'atmos_coupling', 'ocean_coupling', 'ponds', &
       'ocean_fluxes', 'biogeochemistry', 'velocity_solver', 'atmos_fluxes']
  character(len=1024) :: inPath, outPath

  call get_command_argument(1, inPath)
  call get_command_argument(2, outPath)
  call read_stream(inPath, inputDt, pools, fields)
  allocate(domain)
  allocate(domain % blocklist)
  allocate(domain % blocklist % structs)
  domain % configs => pools % configs
  domain % blocklist % configs => pools % configs
  domain % blocklist % dimensions => pool_named(pools, 'mesh')
  domain % blocklist % packages => pools % packages
  do p = 1, size(structNames)
     call stub_pool_put_subpool(domain % blocklist % structs, trim(structNames(p)), &
          pool_named(pools, trim(structNames(p))))
  end do
  call seaice_column_coupling_prep(domain)
  call write_stream(outPath, fields)
end program run_coupling_prep
