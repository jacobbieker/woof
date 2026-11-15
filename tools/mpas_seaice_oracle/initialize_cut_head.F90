! The module head the column oracle's cut of mpas_seaice_initialize.F
! (MPAS-Seaice v8.4.2) is compiled in: build.sh cuts init_ice_state,
! init_ice_state_uniform_ice, init_ice_cice_default,
! initial_category_areas_and_volumes, init_ice_velocity_uniform and
! initialize_coupler_fields verbatim between this head and
! initialize_cut_tail.F90.  The module's own USE list is cut to what those
! routines need (mpas_stream_manager, mpas_io_units and mpas_abort are not).
!
! seaice_testing: init_ice_state USEs its square test cases, which are not
! part of this oracle; they stop by name.
module seaice_testing
  use mpas_derived_types
  use mpas_pool_routines
  implicit none
  private
  public :: seaice_init_square_test_case_hex, seaice_init_square_point_test_case_hex

contains

  subroutine seaice_init_square_test_case_hex(block, configs)
    type(block_type), intent(inout) :: block
    type(MPAS_pool_type), pointer, intent(in) :: configs
    print *, 'oracle stub: the square test case is not part of the column oracle'
    stop 6
  end subroutine seaice_init_square_test_case_hex

  subroutine seaice_init_square_point_test_case_hex(block, configs)
    type(block_type), intent(inout) :: block
    type(MPAS_pool_type), pointer, intent(in) :: configs
    print *, 'oracle stub: the square point test case is not part of the column oracle'
    stop 6
  end subroutine seaice_init_square_point_test_case_hex

end module seaice_testing

module seaice_initialize

  use mpas_dmpar
  use mpas_derived_types
  use mpas_pool_routines
  use mpas_timekeeping
  use mpas_log, only: mpas_log_write

  implicit none

  private
  save

  public :: &
       init_ice_state, &
       initialize_coupler_fields

contains

