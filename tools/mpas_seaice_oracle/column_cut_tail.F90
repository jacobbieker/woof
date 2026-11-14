  subroutine column_stand_in(name)
    character(len=*), intent(in) :: name
    print *, 'oracle stub: ', name, ' (the column package) is not compiled here'
    stop 6
  end subroutine column_stand_in

  function seaice_column_initial_air_drag_coefficient() result(airDragCoefficient)
    real(kind=RKIND) :: airDragCoefficient
    airDragCoefficient = 0.0_RKIND
    call column_stand_in('seaice_column_initial_air_drag_coefficient')
  end function seaice_column_initial_air_drag_coefficient

  subroutine column_prep_radiation(domain)
    type(domain_type), intent(inout) :: domain
    call column_stand_in('column_prep_radiation')
  end subroutine column_prep_radiation

  subroutine column_vertical_thermodynamics(domain, clock, ierr)
    type(domain_type), intent(inout) :: domain
    type(MPAS_Clock_type), intent(in) :: clock
    integer, intent(inout) :: ierr
    call column_stand_in('column_vertical_thermodynamics')
  end subroutine column_vertical_thermodynamics

  subroutine column_biogeochemistry(domain, ierr)
    type(domain_type), intent(inout) :: domain
    integer, intent(inout) :: ierr
    call column_stand_in('column_biogeochemistry')
  end subroutine column_biogeochemistry

  subroutine column_itd_thermodynamics(domain, clock, ierr)
    type(domain_type), intent(inout) :: domain
    type(MPAS_Clock_type), intent(in) :: clock
    integer, intent(inout) :: ierr
    call column_stand_in('column_itd_thermodynamics')
  end subroutine column_itd_thermodynamics

  subroutine seaice_column_update_state(domain, stateUpdateType, dt, iceAgeTimeOffset)
    type(domain_type), intent(inout) :: domain
    character(len=*), intent(in) :: stateUpdateType
    real(kind=RKIND), intent(in) :: dt, iceAgeTimeOffset
    call column_stand_in('seaice_column_update_state')
  end subroutine seaice_column_update_state

  subroutine column_separate_snow_ice_tracers(domain)
    type(domain_type), intent(inout) :: domain
    call column_stand_in('column_separate_snow_ice_tracers')
  end subroutine column_separate_snow_ice_tracers

  subroutine column_combine_snow_ice_tracers(domain)
    type(domain_type), intent(inout) :: domain
    call column_stand_in('column_combine_snow_ice_tracers')
  end subroutine column_combine_snow_ice_tracers

  subroutine column_ridging(domain, ierr)
    type(domain_type), intent(inout) :: domain
    integer, intent(inout) :: ierr
    call column_stand_in('column_ridging')
  end subroutine column_ridging

  subroutine column_radiation(domain, clock, lInitialization)
    type(domain_type), intent(inout) :: domain
    type(MPAS_Clock_type), intent(in) :: clock
    logical, intent(in) :: lInitialization
    call column_stand_in('column_radiation')
  end subroutine column_radiation

  subroutine seaice_column_coupling_prep(domain)
    type(domain_type), intent(inout) :: domain
    call column_stand_in('seaice_column_coupling_prep')
  end subroutine seaice_column_coupling_prep
end module seaice_column
