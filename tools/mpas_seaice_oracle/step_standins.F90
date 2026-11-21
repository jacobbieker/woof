! Stand-ins for the MPAS-Seaice modules the time-step oracle (run_seaice_step)
! compiles against but does not drive: the weak-form solver and the unit
! tests.  Each stops by name if reached.  (The variational basis set-up,
! Wachspress and PWL, is compiled verbatim (build.sh).  Incremental-remap
! advection is compiled verbatim; its stand-ins are remap_standins.F90 and
! rbf_standins.F90.  The column package's ice strength is
! colpkg_standin.F90's, which the column oracle replaces with the column
! package itself.)  The framework is tools/mpas_ocean_oracle/stubs.F90,
! shared with the ocean.
module seaice_velocity_solver_weak
  use mpas_derived_types
  implicit none
  private
  public :: seaice_init_velocity_solver_weak, seaice_internal_stress_weak, seaice_final_divergence_shear_weak

contains

  subroutine seaice_init_velocity_solver_weak(mesh, boundary, velocity_weak, rotateCartesianGrid)
    type(MPAS_pool_type), pointer :: mesh, boundary, velocity_weak
    logical, intent(in) :: rotateCartesianGrid
    print *, 'oracle stub: the weak-form solver is not part of this oracle'
    stop 6
  end subroutine seaice_init_velocity_solver_weak

  subroutine seaice_internal_stress_weak(domain)
    type(domain_type), intent(inout) :: domain
    print *, 'oracle stub: the weak-form solver is not part of this oracle'
    stop 6
  end subroutine seaice_internal_stress_weak

  subroutine seaice_final_divergence_shear_weak(block)
    type(block_type), intent(inout) :: block
    print *, 'oracle stub: the weak-form solver is not part of this oracle'
    stop 6
  end subroutine seaice_final_divergence_shear_weak

end module seaice_velocity_solver_weak

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
