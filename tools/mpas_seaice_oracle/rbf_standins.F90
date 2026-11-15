! Stand-ins for the RBF set-up MPAS-Seaice v8.4.2's remap init calls
! (mpas_rbf_interp_initialize and mpas_init_reconstruct fill the mesh pool's
! coeffs_reconstruct).  In run_incremental_remap_{r4,r8}, run_seaice_step_{r4,r8}
! those coefficients are the input's (as the variational basis is the velocity
! oracles'), so both are no-ops.  run_incremental_remap_init_{r4,r8} links
! MPAS's own routines instead (build.sh cuts them verbatim out of
! src/operators/mpas_vector_operations.F, mpas_rbf_interpolation.F and
! mpas_vector_reconstruction.F).
module mpas_rbf_interpolation
  use mpas_derived_types
  use mpas_pool_routines
  implicit none
  private
  public :: mpas_rbf_interp_initialize

contains

  subroutine mpas_rbf_interp_initialize(meshPool)
    type(MPAS_pool_type), intent(inout) :: meshPool
    ! the input carries coeffs_reconstruct
  end subroutine mpas_rbf_interp_initialize

end module mpas_rbf_interpolation

module mpas_vector_reconstruction
  use mpas_derived_types
  use mpas_pool_routines
  implicit none
  private
  public :: mpas_init_reconstruct

contains

  subroutine mpas_init_reconstruct(meshPool, includeHalos)
    type(MPAS_pool_type), intent(in) :: meshPool
    logical, optional, intent(in) :: includeHalos
    ! the input carries coeffs_reconstruct
  end subroutine mpas_init_reconstruct

end module mpas_vector_reconstruction
