! The stand-in for mpas_ocn_frazil_forcing.F that the tendency oracles of the
! parent build directory (run_oracle, run_pgrad_hmix, run_tendency) compile
! mpas_ocn_tendency.F against.  The oracles in eos/ (run_frazil,
! run_tend_tracer, run_split_step, run_forward) compile the real module
! verbatim instead.  Its routines return at once when
! config_use_frazil_ice_formation is false, as the real ones do, and stop,
! naming the routine, when it is on.  Nothing here carries arithmetic.

module ocn_frazil_forcing
  use mpas_kind_types
  use mpas_pool_routines
  use ocn_constants
  implicit none
  private
  public :: ocn_frazil_forcing_layer_thickness, ocn_frazil_forcing_tracers

contains

  ! frazilFormationOn as ocn_frazil_forcing_init sets it
  subroutine ocn_frazil_forcing_layer_thickness(meshPool, forcingPool, layerThicknessTend, err)
    type (mpas_pool_type), intent(in) :: meshPool
    type (mpas_pool_type), intent(in) :: forcingPool
    real (kind=RKIND), intent(inout), dimension(:,:) :: layerThicknessTend
    integer, intent(out) :: err
    logical, pointer :: config_use_frazil_ice_formation
    err = 0
    call mpas_pool_get_config(ocnConfigs, 'config_use_frazil_ice_formation', config_use_frazil_ice_formation)
    if (config_use_frazil_ice_formation) then
      print *, 'oracle stub: ocn_frazil_forcing_layer_thickness is not compiled here'
      stop 8
    end if
  end subroutine ocn_frazil_forcing_layer_thickness

  subroutine ocn_frazil_forcing_tracers(meshPool, tracersPool, groupName, forcingPool, tracersTend, err)
    character (len=*) :: groupName
    type (mpas_pool_type), intent(inout) :: meshPool
    type (mpas_pool_type), intent(inout) :: tracersPool
    type (mpas_pool_type), intent(inout) :: forcingPool
    real (kind=RKIND), dimension(:,:,:), intent(inout) :: tracersTend
    integer, intent(out) :: err
    logical, pointer :: config_use_frazil_ice_formation
    err = 0
    call mpas_pool_get_config(ocnConfigs, 'config_use_frazil_ice_formation', config_use_frazil_ice_formation)
    if (config_use_frazil_ice_formation) then
      print *, 'oracle stub: ocn_frazil_forcing_tracers is not compiled here'
      stop 8
    end if
  end subroutine ocn_frazil_forcing_tracers

end module ocn_frazil_forcing
