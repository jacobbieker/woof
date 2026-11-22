! Stand-ins for the MPAS-Seaice modules the time-step oracle (run_seaice_step)
! compiles against but does not drive: the unit tests.  It stops by name if
! reached.  (The variational basis set-up, Wachspress and PWL, and the
! weak-form solver are compiled verbatim (build.sh).  Incremental-remap
! advection is compiled verbatim; its stand-ins are remap_standins.F90 and
! rbf_standins.F90.  The column package's ice strength is
! colpkg_standin.F90's, which the column oracle replaces with the column
! package itself.)  The framework is tools/mpas_ocean_oracle/stubs.F90,
! shared with the ocean.
module seaice_unit_test
  use mpas_derived_types
  implicit none
  private
  public :: seaice_perform_unit_test

contains

  subroutine seaice_perform_unit_test(domain)
    type(domain_type), intent(inout) :: domain
    print *, 'oracle stub: seaice_perform_unit_test is not compiled here'
    stop 6
  end subroutine seaice_perform_unit_test

end module seaice_unit_test
