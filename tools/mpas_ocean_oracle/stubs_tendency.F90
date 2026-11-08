! Stand-ins for the MPAS-Ocean modules mpas_ocn_tendency.F USEs but the
! tendency oracle does not compile: tracer tendencies, diagnostics, vertical
! mixing, land-ice fluxes, frazil and the equation of state.
!
! ocn_tend_vel and ocn_tend_thick, the routines graded, call only
! ocn_surface_land_ice_fluxes_{vel,thick} and ocn_frazil_forcing_layer_thickness
! from these.  Those routines return at once when their switch is off
! (config_land_ice_flux_mode neither 'standalone' nor 'coupled';
! config_use_frazil_ice_formation false), and the stand-ins do the same: they
! read the switch by name and stop, naming the routine, when it is on.  Every
! other routine here is reached only from ocn_tend_tracer or
! ocn_tend_freq_filtered_thickness and stops by name.  The dummy declarations
! are copied from the v8.4.2 sources so that mpas_ocn_tendency.F compiles
! against them unmodified.  Nothing here carries arithmetic.

module ocn_equation_of_state
  use mpas_kind_types
  implicit none
  private
  public :: ocn_freezing_temperature

contains

  real (kind=RKIND) function ocn_freezing_temperature(salinity, pressure, inLandIceCavity)
    real (kind=RKIND), intent(in) :: salinity
    real (kind=RKIND), intent(in) :: pressure
    logical, intent(in) :: inLandIceCavity
    print *, 'oracle stub: ocn_freezing_temperature is not compiled here'
    stop 8
  end function ocn_freezing_temperature

end module ocn_equation_of_state

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
    print *, 'oracle stub: ocn_surface_land_ice_fluxes_tracers is not compiled here'
    stop 8
  end subroutine ocn_surface_land_ice_fluxes_tracers

end module ocn_surface_land_ice_fluxes

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
    print *, 'oracle stub: ocn_frazil_forcing_tracers is not compiled here'
    stop 8
  end subroutine ocn_frazil_forcing_tracers

end module ocn_frazil_forcing

module ocn_diagnostics
  use mpas_kind_types
  use mpas_pool_routines
  implicit none
  private
  public :: ocn_compute_KPP_input_fields

contains

  subroutine ocn_compute_KPP_input_fields(statePool, forcingPool, meshPool, diagnosticsPool, scratchPool, timeLevelIn)
    type (mpas_pool_type), intent(in) :: statePool
    type (mpas_pool_type), intent(in) :: forcingPool
    type (mpas_pool_type), intent(in) :: meshPool
    type (mpas_pool_type), intent(inout) :: diagnosticsPool
    type (mpas_pool_type), intent(in) :: scratchPool
    integer, intent(in), optional :: timeLevelIn
    print *, 'oracle stub: ocn_compute_KPP_input_fields is not compiled here'
    stop 8
  end subroutine ocn_compute_KPP_input_fields

end module ocn_diagnostics

module ocn_tracer_hmix
  use mpas_kind_types
  use mpas_pool_routines
  implicit none
  private
  public :: ocn_tracer_hmix_tend

contains

  subroutine ocn_tracer_hmix_tend(meshPool, scratchPool, layerThicknessEdge, zMid, tracers, relativeSlopeTopOfEdge, relativeSlopeTapering, relativeSlopeTaperingCell, tend, err)
    type (mpas_pool_type), intent(in) :: meshPool
    type (mpas_pool_type), intent(in) :: scratchPool
    real (kind=RKIND), dimension(:,:), intent(in) :: layerThicknessEdge, zMid
    real (kind=RKIND), dimension(:,:,:), intent(in) :: tracers
    real (kind=RKIND), dimension(:,:), intent(in) :: relativeSlopeTopOfEdge, relativeSlopeTapering, relativeSlopeTaperingCell
    real (kind=RKIND), dimension(:,:,:), intent(inout) :: tend
    integer, intent(out) :: err
    print *, 'oracle stub: ocn_tracer_hmix_tend is not compiled here'
    stop 8
  end subroutine ocn_tracer_hmix_tend

end module ocn_tracer_hmix

module ocn_high_freq_thickness_hmix_del2
  use mpas_kind_types
  use mpas_pool_routines
  implicit none
  private
  public :: ocn_high_freq_thickness_hmix_del2_tend

contains

  subroutine ocn_high_freq_thickness_hmix_del2_tend(meshPool, highFreqThickness, tend_highFreqThickness, err)
    type (mpas_pool_type), intent(in) :: meshPool
    real (kind=RKIND), dimension(:,:), intent(in) :: highFreqThickness
    real (kind=RKIND), dimension(:,:), intent(inout) :: tend_highFreqThickness
    integer, intent(out) :: err
    print *, 'oracle stub: ocn_high_freq_thickness_hmix_del2_tend is not compiled here'
    stop 8
  end subroutine ocn_high_freq_thickness_hmix_del2_tend

end module ocn_high_freq_thickness_hmix_del2

module ocn_tracer_advection
  use mpas_kind_types
  use mpas_pool_routines
  implicit none
  private
  public :: ocn_tracer_advection_tend

contains

  subroutine ocn_tracer_advection_tend(tracers, normalThicknessFlux, w, layerThickness, dt, meshPool, scratchPool, diagnosticsPool, tend, tracerGroupName)
    real (kind=RKIND), dimension(:,:,:), intent(inout) :: tend
    real (kind=RKIND), dimension(:,:,:), intent(in) :: tracers
    real (kind=RKIND), dimension(:,:), intent(in) :: normalThicknessFlux
    real (kind=RKIND), dimension(:,:), intent(in) :: w
    real (kind=RKIND), dimension(:,:), intent(in) :: layerThickness
    real (kind=RKIND), intent(in) :: dt
    type (mpas_pool_type), intent(in) :: meshPool
    type (mpas_pool_type), intent(in) :: scratchPool
    type (mpas_pool_type), intent(in) :: diagnosticsPool
    character (len=*), intent(in) :: tracerGroupName
    print *, 'oracle stub: ocn_tracer_advection_tend is not compiled here'
    stop 8
  end subroutine ocn_tracer_advection_tend

end module ocn_tracer_advection

module ocn_tracer_short_wave_absorption
  use mpas_kind_types
  use mpas_pool_routines
  implicit none
  private
  public :: ocn_tracer_short_wave_absorption_tend

contains

  subroutine ocn_tracer_short_wave_absorption_tend(meshPool, swForcingPool, forcingPool, index_temperature, layerThickness, penetrativeTemperatureFlux, penetrativeTemperatureFluxOBL, tend, err)
    type (mpas_pool_type), intent(in) :: meshPool, swForcingPool, forcingPool
    real (kind=RKIND), dimension(:), intent(in) :: penetrativeTemperatureFlux
    real (kind=RKIND), dimension(:), intent(inout) :: penetrativeTemperatureFluxOBL
    real (kind=RKIND), dimension(:,:), intent(in) :: layerThickness
    integer, intent(in) :: index_temperature
    real (kind=RKIND), dimension(:,:,:), intent(inout) :: tend
    integer, intent(out) :: err
    print *, 'oracle stub: ocn_tracer_short_wave_absorption_tend is not compiled here'
    stop 8
  end subroutine ocn_tracer_short_wave_absorption_tend

end module ocn_tracer_short_wave_absorption

module ocn_tracer_nonlocalflux
  use mpas_kind_types
  use mpas_pool_routines
  implicit none
  private
  public :: ocn_tracer_nonlocalflux_tend

contains

  subroutine ocn_tracer_nonlocalflux_tend(meshPool, vertNonLocalFlux, surfaceTracerFlux, tend, err)
    type (mpas_pool_type), intent(in) :: meshPool
    real (kind=RKIND), dimension(:,:), intent(in) :: surfaceTracerFlux
    real (kind=RKIND), dimension(:,:,:), intent(in) :: vertNonLocalFlux
    real (kind=RKIND), dimension(:,:,:), intent(inout) :: tend
    integer, intent(out) :: err
    print *, 'oracle stub: ocn_tracer_nonlocalflux_tend is not compiled here'
    stop 8
  end subroutine ocn_tracer_nonlocalflux_tend

end module ocn_tracer_nonlocalflux

module ocn_tracer_surface_restoring
  use mpas_kind_types
  use mpas_pool_routines
  implicit none
  private
  public :: ocn_tracer_surface_restoring_compute

contains

  subroutine ocn_tracer_surface_restoring_compute(groupName, nTracers, nCells, tracers, pistonVelocity, tracersSurfaceRestoringValue, tracersSurfaceFlux, indexSalinity, use_surface_salinity_monthly_restoring, salinity_restoring_constant_piston_velocity, err)
    character (len=*), intent(in) :: groupName
    integer, intent(in) :: nTracers, indexSalinity, nCells
    real (kind=RKIND), intent(in) :: salinity_restoring_constant_piston_velocity
    logical, intent(in) :: use_surface_salinity_monthly_restoring
    real (kind=RKIND), dimension(:,:,:), intent(in) :: tracers
    real (kind=RKIND), dimension(:,:), intent(inout) :: pistonVelocity, tracersSurfaceRestoringValue
    real (kind=RKIND), dimension(:,:), intent(inout) :: tracersSurfaceFlux
    integer, intent(out) :: err
    print *, 'oracle stub: ocn_tracer_surface_restoring_compute is not compiled here'
    stop 8
  end subroutine ocn_tracer_surface_restoring_compute

end module ocn_tracer_surface_restoring

module ocn_tracer_interior_restoring
  use mpas_kind_types
  use mpas_pool_routines
  implicit none
  private
  public :: ocn_tracer_interior_restoring_compute

contains

  subroutine ocn_tracer_interior_restoring_compute(nTracers, nCellsSolve, maxLevelCell, layerThickness, tracers, tracersInteriorRestoringRate, tracersInteriorRestoringValue, tracer_tend, err)
    integer, dimension(:), intent(in) :: maxLevelCell
    real (kind=RKIND), dimension(:,:), intent(in) :: layerThickness
    real (kind=RKIND), dimension(:,:,:), intent(in) :: tracers, tracersInteriorRestoringRate, tracersInteriorRestoringValue
    integer, intent(in) :: nTracers, nCellsSolve
    real (kind=RKIND), dimension(:,:,:), intent(inout) :: tracer_tend
    integer, intent(out) :: err
    print *, 'oracle stub: ocn_tracer_interior_restoring_compute is not compiled here'
    stop 8
  end subroutine ocn_tracer_interior_restoring_compute

end module ocn_tracer_interior_restoring

module ocn_tracer_exponential_decay
  use mpas_kind_types
  use mpas_pool_routines
  implicit none
  private
  public :: ocn_tracer_exponential_decay_compute

contains

  subroutine ocn_tracer_exponential_decay_compute(nTracers, nCellsSolve, maxLevelCell, layerThickness, tracers, tracersExponentialDecayRate, tracer_tend, err)
    integer, dimension(:), intent(in) :: maxLevelCell
    real (kind=RKIND), dimension(:), intent(in) :: tracersExponentialDecayRate
    real (kind=RKIND), dimension(:,:), intent(in) :: layerThickness
    real (kind=RKIND), dimension(:,:,:), intent(in) :: tracers
    integer, intent(in) :: nTracers, nCellsSolve
    real (kind=RKIND), dimension(:,:,:), intent(inout) :: tracer_tend
    integer, intent(out) :: err
    print *, 'oracle stub: ocn_tracer_exponential_decay_compute is not compiled here'
    stop 8
  end subroutine ocn_tracer_exponential_decay_compute

end module ocn_tracer_exponential_decay

module ocn_tracer_ideal_age
  use mpas_kind_types
  use mpas_pool_routines
  implicit none
  private
  public :: ocn_tracer_ideal_age_compute

contains

  subroutine ocn_tracer_ideal_age_compute(nTracers, nCellsSolve, maxLevelCell, layerThickness, idealAgeMask, tracers, tracer_tend, err)
    integer, dimension(:), intent(in) :: maxLevelCell
    real (kind=RKIND), dimension(:,:), intent(in) :: layerThickness, idealAgeMask
    integer, intent(in) :: nTracers, nCellsSolve
    real (kind=RKIND), dimension(:,:,:), intent(inout) :: tracers, tracer_tend
    integer, intent(out) :: err
    print *, 'oracle stub: ocn_tracer_ideal_age_compute is not compiled here'
    stop 8
  end subroutine ocn_tracer_ideal_age_compute

end module ocn_tracer_ideal_age

module ocn_tracer_TTD
  use mpas_kind_types
  use mpas_pool_routines
  implicit none
  private
  public :: ocn_tracer_TTD_compute

contains

  subroutine ocn_tracer_TTD_compute(nTracers, nCellsSolve, maxLevelCell, layerThickness, TTDMask, tracers, err)
    integer, dimension(:), intent(in) :: maxLevelCell
    real (kind=RKIND), dimension(:,:), intent(in) :: layerThickness, TTDMask
    integer, intent(in) :: nTracers, nCellsSolve
    real (kind=RKIND), dimension(:,:,:), intent(inout) :: tracers
    integer, intent(out) :: err
    print *, 'oracle stub: ocn_tracer_TTD_compute is not compiled here'
    stop 8
  end subroutine ocn_tracer_TTD_compute

end module ocn_tracer_TTD

module ocn_tracer_surface_flux_to_tend
  use mpas_kind_types
  use mpas_pool_routines
  implicit none
  private
  public :: ocn_tracer_surface_flux_tend

contains

  subroutine ocn_tracer_surface_flux_tend(meshPool, fractionAbsorbed, fractionAbsorbedRunoff, layerThickness, surfaceTracerFlux, surfaceTracerFluxRunoff, tend, err)
    type (mpas_pool_type), intent(in) :: meshPool
    real (kind=RKIND), dimension(:,:), intent(in) :: layerThickness
    real (kind=RKIND), dimension(:,:), intent(in) :: surfaceTracerFlux
    real (kind=RKIND), dimension(:,:), intent(in) :: fractionAbsorbed
    real (kind=RKIND), dimension(:,:), intent(in), pointer :: surfaceTracerFluxRunoff
    real (kind=RKIND), dimension(:,:), intent(in) :: fractionAbsorbedRunoff
    real (kind=RKIND), dimension(:,:,:), intent(inout) :: tend
    integer, intent(out) :: err
    print *, 'oracle stub: ocn_tracer_surface_flux_tend is not compiled here'
    stop 8
  end subroutine ocn_tracer_surface_flux_tend

end module ocn_tracer_surface_flux_to_tend

module ocn_tracer_ecosys
  use mpas_kind_types
  use mpas_pool_routines
  implicit none
  private
  public :: ocn_tracer_ecosys_compute, ocn_tracer_ecosys_surface_flux_compute

contains

  subroutine ocn_tracer_ecosys_compute(activeTracers, ecosysTracers, forcingPool, nTracers, nCellsSolve, latCell, maxLevelCell, nVertLevels, layerThickness, zMid, indexTemperature, indexSalinity, ecosysTracersTend, err)
    integer, dimension(:), intent(in) :: maxLevelCell
    real (kind=RKIND), dimension(:), intent(in) :: latCell
    real (kind=RKIND), dimension(:,:), intent(in) :: zMid, layerThickness
    real (kind=RKIND), dimension(:,:,:), intent(in) :: ecosysTracers
    real (kind=RKIND), dimension(:,:,:), intent(in) :: activeTracers
    integer, intent(in) :: nTracers, nCellsSolve, nVertLevels, indexTemperature, indexSalinity
    type (mpas_pool_type), intent(inout) :: forcingPool
    real (kind=RKIND), dimension(:,:,:), intent(inout) :: ecosysTracersTend
    integer, intent(out) :: err
    print *, 'oracle stub: ocn_tracer_ecosys_compute is not compiled here'
    stop 8
  end subroutine ocn_tracer_ecosys_compute

  subroutine ocn_tracer_ecosys_surface_flux_compute(activeTracers, ecosysTracers, forcingPool, nTracers, nCellsSolve, zMid, indexTemperature, indexSalinity, ecosysSurfaceFlux, err)
    real (kind=RKIND), dimension(:,:), intent(in) :: zMid
    real (kind=RKIND), dimension(:,:), intent(inout) :: ecosysSurfaceFlux
    real (kind=RKIND), dimension(:,:,:), intent(in) :: ecosysTracers
    real (kind=RKIND), dimension(:,:,:), intent(in) :: activeTracers
    integer, intent(in) :: nTracers, nCellsSolve, indexTemperature, indexSalinity
    type (mpas_pool_type), intent(inout) :: forcingPool
    integer, intent(out) :: err
    print *, 'oracle stub: ocn_tracer_ecosys_surface_flux_compute is not compiled here'
    stop 8
  end subroutine ocn_tracer_ecosys_surface_flux_compute

end module ocn_tracer_ecosys

module ocn_tracer_DMS
  use mpas_kind_types
  use mpas_pool_routines
  implicit none
  private
  public :: ocn_tracer_DMS_compute, ocn_tracer_DMS_surface_flux_compute

contains

  subroutine ocn_tracer_DMS_compute(activeTracers, DMSTracers, nTracersDMS, ecosysTracers, nTracersEcosys, forcingPool, nCellsSolve, maxLevelCell, nVertLevels, layerThickness, indexTemperature, indexSalinity, DMSTracersTend, err)
    integer, dimension(:), intent(in) :: maxLevelCell
    real (kind=RKIND), dimension(:,:), intent(in) :: layerThickness
    real (kind=RKIND), dimension(:,:,:), intent(in) :: DMSTracers
    real (kind=RKIND), dimension(:,:,:), intent(in) :: ecosysTracers
    real (kind=RKIND), dimension(:,:,:), intent(in) :: activeTracers
    type (mpas_pool_type), intent(in) :: forcingPool
    integer, intent(in) :: nTracersDMS, nTracersEcosys, nCellsSolve, nVertLevels
    integer, intent(in) :: indexTemperature, indexSalinity
    real (kind=RKIND), dimension(:,:,:), intent(inout) :: DMSTracersTend
    integer, intent(out) :: err
    print *, 'oracle stub: ocn_tracer_DMS_compute is not compiled here'
    stop 8
  end subroutine ocn_tracer_DMS_compute

  subroutine ocn_tracer_DMS_surface_flux_compute(activeTracers, DMSTracers, forcingPool, nTracers, nCellsSolve, zMid, indexTemperature, indexSalinity, DMSSurfaceFlux, DMSSurfaceFluxRemoved, err)
    real (kind=RKIND), dimension(:,:), intent(in) :: zMid
    real (kind=RKIND), dimension(:,:), intent(inout) :: DMSSurfaceFlux, DMSSurfaceFluxRemoved
    real (kind=RKIND), dimension(:,:,:), intent(in) :: DMSTracers
    real (kind=RKIND), dimension(:,:,:), intent(in) :: activeTracers
    integer, intent(in) :: nTracers, nCellsSolve, indexTemperature, indexSalinity
    type (mpas_pool_type), intent(inout) :: forcingPool
    integer, intent(out) :: err
    print *, 'oracle stub: ocn_tracer_DMS_surface_flux_compute is not compiled here'
    stop 8
  end subroutine ocn_tracer_DMS_surface_flux_compute

end module ocn_tracer_DMS

module ocn_tracer_MacroMolecules
  use mpas_kind_types
  use mpas_pool_routines
  implicit none
  private
  public :: ocn_tracer_MacroMolecules_compute, ocn_tracer_MacroMolecules_surface_flux_compute

contains

  subroutine ocn_tracer_MacroMolecules_compute(MacroMoleculesTracers, nTracersMacroMolecules, ecosysTracers, nTracersEcosys, forcingPool, nCellsSolve, maxLevelCell, nVertLevels, layerThickness, MacroMoleculesTracersTend, err)
    integer, dimension(:), intent(in) :: maxLevelCell
    real (kind=RKIND), dimension(:,:), intent(in) :: layerThickness
    real (kind=RKIND), dimension(:,:,:), intent(in) :: MacroMoleculesTracers
    real (kind=RKIND), dimension(:,:,:), intent(in) :: ecosysTracers
    type (mpas_pool_type), intent(in) :: forcingPool
    integer, intent(in) :: nTracersMacroMolecules, nTracersEcosys, nCellsSolve, nVertLevels
    real (kind=RKIND), dimension(:,:,:), intent(inout) :: MacroMoleculesTracersTend
    integer, intent(out) :: err
    print *, 'oracle stub: ocn_tracer_MacroMolecules_compute is not compiled here'
    stop 8
  end subroutine ocn_tracer_MacroMolecules_compute

  subroutine ocn_tracer_MacroMolecules_surface_flux_compute(activeTracers, MacroMoleculesTracers, forcingPool, nTracers, nCellsSolve, zMid, indexTemperature, indexSalinity, MacroMoleculesSurfaceFlux, err)
    real (kind=RKIND), dimension(:,:), intent(in) :: zMid
    real (kind=RKIND), dimension(:,:), intent(inout) :: MacroMoleculesSurfaceFlux
    real (kind=RKIND), dimension(:,:,:), intent(in) :: MacroMoleculesTracers
    real (kind=RKIND), dimension(:,:,:), intent(in) :: activeTracers
    integer, intent(in) :: nTracers, nCellsSolve, indexTemperature, indexSalinity
    type (mpas_pool_type), intent(inout) :: forcingPool
    integer, intent(out) :: err
    print *, 'oracle stub: ocn_tracer_MacroMolecules_surface_flux_compute is not compiled here'
    stop 8
  end subroutine ocn_tracer_MacroMolecules_surface_flux_compute

end module ocn_tracer_MacroMolecules

module ocn_vmix
  use mpas_kind_types
  use mpas_pool_routines
  implicit none
  private

contains

end module ocn_vmix
