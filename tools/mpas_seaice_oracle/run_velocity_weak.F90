! Drive MPAS-Seaice v8.4.2's weak-form stress divergence on one block and
! write every field word back: the reference the port
! (woof.hex.mpas_seaice.weak_v842) is held against.
!
!   run_velocity_weak INPUT.bin OUTPUT.bin
!
! INPUT.bin is written by woof.verify.mpas_seaice_weak_oracle in
! tools/mpas_ocean_oracle/oracle_stream.F90's layout.  The driver runs, all
! compiled verbatim (mpas_seaice_velocity_solver_weak.F, the constitutive
! relation, seaice_mesh's seaice_normal_vectors cut verbatim; build.sh):
! seaice_init_evp, seaice_init_velocity_solver_weak (the normal vectors and
! rotated latitudes), oracle_n_iterations passes of
! seaice_internal_stress_weak (strain tensor, stress tensor, stress
! divergence), then seaice_final_divergence_shear_weak.

program run_velocity_weak
  use mpas_kind_types
  use mpas_derived_types
  use mpas_pool_routines
  use seaice_velocity_solver_constitutive_relation, only: seaice_init_evp
  use seaice_velocity_solver_weak, only: seaice_init_velocity_solver_weak, seaice_internal_stress_weak, &
       seaice_final_divergence_shear_weak
  use oracle_stream
  implicit none
  type(oracle_pools) :: pools
  type(oracle_field), allocatable :: fields(:)
  type(domain_type), pointer :: domain
  real(kind=RKIND) :: inputDt
  logical, pointer :: config_rotate_cartesian_grid
  integer, pointer :: nIterations
  integer :: i, p
  character(len=24), parameter :: structNames(5) = [character(len=24) :: 'mesh', 'boundary', &
       'velocity_solver', 'velocity_weak', 'ridging']
  character(len=1024) :: inPath, outPath

  call get_command_argument(1, inPath)
  call get_command_argument(2, outPath)
  call read_stream(inPath, inputDt, pools, fields)
  allocate(domain)
  allocate(domain % blocklist)
  allocate(domain % blocklist % structs)
  allocate(domain % dminfo)
  domain % configs => pools % configs
  domain % blocklist % configs => pools % configs
  domain % blocklist % dimensions => pool_named(pools, 'mesh')
  domain % blocklist % packages => pools % packages
  do p = 1, size(structNames)
     call stub_pool_put_subpool(domain % blocklist % structs, trim(structNames(p)), &
          pool_named(pools, trim(structNames(p))))
  end do

  call seaice_init_evp(domain)
  call mpas_pool_get_config(pools % configs, 'config_rotate_cartesian_grid', config_rotate_cartesian_grid)
  call seaice_init_velocity_solver_weak(pool_named(pools, 'mesh'), pool_named(pools, 'boundary'), &
       pool_named(pools, 'velocity_weak'), config_rotate_cartesian_grid)
  call mpas_pool_get_config(pools % configs, 'oracle_n_iterations', nIterations)
  do i = 1, nIterations
     call seaice_internal_stress_weak(domain)
  end do
  call seaice_final_divergence_shear_weak(domain % blocklist)
  call write_stream(outPath, fields)
end program run_velocity_weak
