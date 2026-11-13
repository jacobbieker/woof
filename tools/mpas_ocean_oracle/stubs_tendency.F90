! Stand-ins for the MPAS-Ocean modules mpas_ocn_tendency.F USEs that the
! tendency oracle (run_tendency: ocn_tend_vel, ocn_tend_thick) does not
! compile, though other oracles compile them for real: the tracer tendencies'
! advection, mixing, surface flux, short-wave absorption and non-local flux,
! the diagnostics and the equation of state.  ocn_tend_vel and ocn_tend_thick
! never reach them, so every routine here stops by name.  The modules no
! oracle compiles for real are stubs_unported.F90's, which every harness that
! compiles mpas_ocn_tendency.F links.  The dummy declarations are copied from
! the v8.4.2 sources so that mpas_ocn_tendency.F compiles against them
! unmodified.  Nothing here carries arithmetic.

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
