! Stand-ins for the routines the split-explicit step calls only under
! config_use_standardGM, which run_split_step does not drive: GM's bolus
! velocity (mpas_ocn_gm.F) and the GM vector reconstruction
! (ocn_reconstruct_gm_vectors, mpas_ocn_diagnostics.F); and for RK4, which
! ocn_timestep calls for config_time_integrator 'RK4'.  Each stops by name.
module ocn_gm
  use mpas_kind_types
  use mpas_pool_routines
  implicit none
  private
  public :: ocn_gm_compute_Bolus_velocity

contains

  subroutine ocn_gm_compute_Bolus_velocity(diagnosticsPool, meshPool, scratchPool)
    type (mpas_pool_type), intent(inout) :: diagnosticsPool
    type (mpas_pool_type), intent(in) :: meshPool
    type (mpas_pool_type), intent(inout) :: scratchPool
    print *, 'oracle stub: ocn_gm_compute_Bolus_velocity is not compiled here'
    stop 8
  end subroutine ocn_gm_compute_Bolus_velocity

end module ocn_gm

module split_gm_vectors
  use mpas_kind_types
  use mpas_pool_routines
  implicit none
  private
  public :: ocn_reconstruct_gm_vectors

contains

  subroutine ocn_reconstruct_gm_vectors(diagnosticsPool, meshPool)
    type (mpas_pool_type), intent(inout) :: diagnosticsPool
    type (mpas_pool_type), intent(in) :: meshPool
    print *, 'oracle stub: ocn_reconstruct_gm_vectors is not compiled here'
    stop 8
  end subroutine ocn_reconstruct_gm_vectors

end module split_gm_vectors

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
