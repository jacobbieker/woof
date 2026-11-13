! Stand-ins for the routines the split-explicit step calls only under
! config_use_standardGM, which run_split_step does not drive: GM's bolus
! velocity (mpas_ocn_gm.F) and the GM vector reconstruction
! (ocn_reconstruct_gm_vectors, mpas_ocn_diagnostics.F).  Each stops by name.
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
