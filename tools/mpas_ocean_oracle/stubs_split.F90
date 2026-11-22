! A stand-in for RK4, which ocn_timestep calls for config_time_integrator
! 'RK4' (it stops by name), and the threading barrier.  GM (mpas_ocn_gm.F)
! and ocn_reconstruct_gm_vectors are the real routines (build.sh).

! RK4 (mpas_ocn_time_integration_rk4.F) is not driven: ocn_timestep's other branch
module ocn_time_integration_rk4
  use mpas_kind_types
  use mpas_derived_types
  implicit none
  private
  public :: ocn_time_integrator_rk4

contains

  subroutine ocn_time_integrator_rk4(domain, dt)
    type (domain_type), intent(inout) :: domain
    real (kind=RKIND), intent(in) :: dt
    print *, 'oracle stub: ocn_time_integrator_rk4 is not compiled here'
    stop 8
  end subroutine ocn_time_integrator_rk4

end module ocn_time_integration_rk4

! mpas_ocn_time_average_coupled.F calls the barrier without USE-ing
! mpas_threading: the external routine of that name (one thread)
subroutine mpas_threading_barrier()
end subroutine mpas_threading_barrier
