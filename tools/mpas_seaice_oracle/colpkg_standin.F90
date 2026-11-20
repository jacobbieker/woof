! The column package's ice strength for the time-step oracle
! (run_seaice_step), which compiles the velocity solver at RKIND 4 and 8
! without the column package: it stops by name if reached.  The column
! oracle (run_column's seaice_step stage) links the column package's own
! colpkg_ice_strength instead.
module ice_colpkg
  use mpas_kind_types
  implicit none
  private
  public :: colpkg_ice_strength

contains

  subroutine colpkg_ice_strength(ncat, aice, vice, aice0, aicen, vicen, strength)
    ! The real colpkg routine takes dbl_kind arguments, so an RKIND 4 build
    ! of the velocity solver links only against a stand-in.  ice_strength
    ! calls it only under the column package's vertical thermodynamics, which
    ! run_presubcycle.F90 drives without ice_strength (the GPU port refuses
    ! that configuration's ice strength by name).
    integer, intent(in) :: ncat
    real(kind=RKIND), intent(in) :: aice, vice, aice0
    real(kind=RKIND), dimension(:), intent(in) :: aicen, vicen
    real(kind=RKIND), intent(inout) :: strength
    print *, 'oracle stub: colpkg_ice_strength is not part of this oracle'
    stop 6
  end subroutine colpkg_ice_strength

end module ice_colpkg
