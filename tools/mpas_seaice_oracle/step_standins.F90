! Stand-ins for the MPAS-Seaice modules the time-step oracle (run_seaice_step)
! compiles against but does not drive: the variational basis setup
! (Wachspress, PWL: the driver gives the basis arrays), the weak-form solver,
! the column package's ice strength and the unit tests.  Each stops by name
! if reached.  (Incremental-remap advection is compiled verbatim; its
! stand-ins are remap_standins.F90.)  The
! framework is tools/mpas_ocean_oracle/stubs.F90, shared with the ocean.
module seaice_velocity_solver_wachspress
  use mpas_derived_types
  implicit none
  private
  public :: seaice_init_velocity_solver_wachspress

contains

  subroutine seaice_init_velocity_solver_wachspress(mesh, velocity_variational, boundary, &
       rotateCartesianGrid, includeMetricTerms, integrationType, integrationOrder)
    type(MPAS_pool_type), pointer, intent(in) :: mesh
    type(MPAS_pool_type), pointer :: velocity_variational, boundary
    logical, intent(in) :: rotateCartesianGrid, includeMetricTerms
    character(len=*), intent(in) :: integrationType
    integer, intent(in) :: integrationOrder
    print *, 'oracle stub: the Wachspress basis setup is not part of this oracle'
    stop 6
  end subroutine seaice_init_velocity_solver_wachspress

end module seaice_velocity_solver_wachspress

module seaice_velocity_solver_pwl
  use mpas_derived_types
  implicit none
  private
  public :: seaice_init_velocity_solver_pwl

contains

  subroutine seaice_init_velocity_solver_pwl(mesh, velocity_variational, boundary, &
       rotateCartesianGrid, includeMetricTerms)
    type(MPAS_pool_type), pointer, intent(in) :: mesh
    type(MPAS_pool_type), pointer :: velocity_variational, boundary
    logical, intent(in) :: rotateCartesianGrid, includeMetricTerms
    print *, 'oracle stub: the PWL basis setup is not part of this oracle'
    stop 6
  end subroutine seaice_init_velocity_solver_pwl

end module seaice_velocity_solver_pwl

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
