! Drive MPAS-Ocean v8.4.2's land-ice fluxes and effective density on one block
! and write every field word back: the reference the GPU port is held against.
!
!   run_land_ice INPUT.bin OUTPUT.bin
!
! INPUT.bin is written by woof.verify.mpas_ocean_land_ice_oracle in
! oracle_stream.F90's layout (time level 2 is NAME@2).
! mpas_ocn_surface_land_ice_fluxes.F and mpas_ocn_effective_density_in_land_ice.F
! are compiled verbatim with the real equation of state.  The driver makes
! the calls the forward mode, the tendencies and the split step make, in
! their order:
!
!   ocn_surface_land_ice_fluxes_init(err)
!   ocn_surface_land_ice_fluxes_build_arrays(mesh, diagnostics, forcing, scratch, state, dt, err)
!   ocn_surface_land_ice_fluxes_thick(mesh, forcing, forcing % surfaceThicknessFlux, err)
!   ocn_surface_land_ice_fluxes_vel(mesh, diagnostics, forcing % surfaceStress,
!                                   forcing % surfaceStressMagnitude, err)
!   ocn_surface_land_ice_fluxes_tracers(mesh, 'activeTracers', forcing,
!                                       tracersSurfaceFlux % activeTracersSurfaceFlux, err)
!   ocn_effective_density_in_land_ice_update(mesh, forcing, state, scratch, err)
!
! dt is the stream's.

program run_land_ice
  use mpas_kind_types
  use mpas_derived_types
  use mpas_pool_routines
  use ocn_constants
  use ocn_equation_of_state
  use ocn_surface_land_ice_fluxes
  use ocn_effective_density_in_land_ice
  use oracle_stream
  implicit none
  type(oracle_pools) :: pools
  type(oracle_field), allocatable :: fields(:)
  type(MPAS_pool_type), pointer :: meshPool, statePool, diagnosticsPool, forcingPool, scratchPool
  real(kind=RKIND) :: dt
  integer :: err
  real(kind=RKIND), dimension(:), pointer :: surfaceThicknessFlux, surfaceStress, surfaceStressMagnitude
  real(kind=RKIND), dimension(:,:), pointer :: tracersSurfaceFlux
  character(len=1024) :: inPath, outPath

  call get_command_argument(1, inPath)
  call get_command_argument(2, outPath)
  call read_stream(inPath, dt, pools, fields)
  meshPool => pool_named(pools, 'mesh')
  statePool => pool_named(pools, 'state')
  diagnosticsPool => pool_named(pools, 'diagnostics')
  forcingPool => pool_named(pools, 'forcing')
  scratchPool => pool_named(pools, 'scratch')

  call ocn_constants_init(pools % configs, pools % packages)
  call ocn_equation_of_state_init(err)
  if (err /= 0) stop 13
  call ocn_surface_land_ice_fluxes_init(err)
  if (err /= 0) stop 13
  call ocn_surface_land_ice_fluxes_build_arrays(meshPool, diagnosticsPool, forcingPool, scratchPool, statePool, &
       dt, err)
  if (err /= 0) stop 9
  call mpas_pool_get_array(forcingPool, 'surfaceThicknessFlux', surfaceThicknessFlux)
  call ocn_surface_land_ice_fluxes_thick(meshPool, forcingPool, surfaceThicknessFlux, err)
  if (err /= 0) stop 9
  call mpas_pool_get_array(forcingPool, 'surfaceStress', surfaceStress)
  call mpas_pool_get_array(forcingPool, 'surfaceStressMagnitude', surfaceStressMagnitude)
  call ocn_surface_land_ice_fluxes_vel(meshPool, diagnosticsPool, surfaceStress, surfaceStressMagnitude, err)
  if (err /= 0) stop 9
  call mpas_pool_get_array(pool_named(pools, 'tracersSurfaceFlux'), 'activeTracersSurfaceFlux', tracersSurfaceFlux)
  call ocn_surface_land_ice_fluxes_tracers(meshPool, 'activeTracers', forcingPool, tracersSurfaceFlux, err)
  if (err /= 0) stop 9
  call ocn_effective_density_in_land_ice_update(meshPool, forcingPool, statePool, scratchPool, err)
  if (err /= 0) stop 9
  call write_stream(outPath, fields)
end program run_land_ice
