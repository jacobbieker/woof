! The stand-in for mpas_ocn_surface_land_ice_fluxes.F that the tendency
! oracles of the parent build directory (run_oracle, run_pgrad_hmix,
! run_tendency) compile mpas_ocn_tendency.F against.  The oracles in eos/
! (run_land_ice, run_tend_tracer, run_split_step, run_forward) compile the
! real module verbatim instead.  Its routines return at once when their
! switch is off (config_land_ice_flux_mode neither 'standalone' nor
! 'coupled'), as the real ones do, and stop, naming the routine, when it is
! on.  The dummy declarations are copied from the v8.4.2 source.  Nothing
! here carries arithmetic.

module ocn_surface_land_ice_fluxes
  use mpas_kind_types
  use mpas_pool_routines
  use ocn_constants
  implicit none
  private
  public :: ocn_surface_land_ice_fluxes_vel, ocn_surface_land_ice_fluxes_thick, ocn_surface_land_ice_fluxes_tracers

contains

  ! landIceFluxesOn as ocn_surface_land_ice_fluxes_init sets it
  subroutine stop_if_on(name)
    character(len=*), intent(in) :: name
    character(len=StrKIND), pointer :: config_land_ice_flux_mode
    call mpas_pool_get_config(ocnConfigs, 'config_land_ice_flux_mode', config_land_ice_flux_mode)
    if (trim(config_land_ice_flux_mode) == 'standalone' .or. trim(config_land_ice_flux_mode) == 'coupled') then
      print *, 'oracle stub: ', name, ' is not compiled here'
      stop 8
    end if
  end subroutine stop_if_on

  subroutine ocn_surface_land_ice_fluxes_vel(meshPool, diagnosticsPool, surfaceStress, surfaceStressMagnitude, err)
    type (mpas_pool_type), intent(in) :: meshPool
    type (mpas_pool_type), intent(in) :: diagnosticsPool
    real (kind=RKIND), dimension(:), intent(inout) :: surfaceStress, surfaceStressMagnitude
    integer, intent(out) :: err
    err = 0
    call stop_if_on('ocn_surface_land_ice_fluxes_vel')
  end subroutine ocn_surface_land_ice_fluxes_vel

  subroutine ocn_surface_land_ice_fluxes_thick(meshPool, forcingPool, surfaceThicknessFlux, err)
    type (mpas_pool_type), intent(in) :: meshPool
    type (mpas_pool_type), intent(inout) :: forcingPool
    real (kind=RKIND), dimension(:), intent(inout) :: surfaceThicknessFlux
    integer, intent(out) :: err
    err = 0
    call stop_if_on('ocn_surface_land_ice_fluxes_thick')
  end subroutine ocn_surface_land_ice_fluxes_thick

  subroutine ocn_surface_land_ice_fluxes_tracers(meshPool, groupName, forcingPool, tracersSurfaceFlux, err)
    type (mpas_pool_type), intent(in) :: meshPool
    character (len=*) :: groupName
    type (mpas_pool_type), intent(inout) :: forcingPool
    real (kind=RKIND), dimension(:,:), intent(inout) :: tracersSurfaceFlux
    integer, intent(out) :: err
    err = 0
    call stop_if_on('ocn_surface_land_ice_fluxes_tracers')
  end subroutine ocn_surface_land_ice_fluxes_tracers

end module ocn_surface_land_ice_fluxes
