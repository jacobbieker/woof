! Stand-ins for the modules MPAS-Seaice v8.4.2's incremental remap
! (mpas_seaice_advection_incremental_remap.F, compiled verbatim) and the
! advection dispatcher (mpas_seaice_advection.F, verbatim) USE but whose work
! is outside this oracle:
!
! - mpas_rbf_interpolation / mpas_vector_reconstruction: the remap's init
!   calls mpas_rbf_interp_initialize and mpas_init_reconstruct to fill the
!   mesh pool's coeffs_reconstruct.  Here those coefficients are the input's
!   (as the variational basis is the velocity oracles'), so both are no-ops.
! - seaice_error: its error codes (verbatim values) and
!   seaice_critical_error_write_block, which writes the abort stream (no state).
! - seaice_advection_upwind: upwind advection cannot run in v8.4.2 (it reads
!   configs the sea-ice Registry does not declare, config_convert_volume_to_thickness
!   and others, and mesh fields it does not carry, derivTwo, advCoefs and
!   others; MPAS_pool_get_config hands back a null pointer that it then
!   dereferences), so it stops by name.
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

module seaice_error
  use mpas_derived_types
  implicit none
  private
  save

  ! mpas_seaice_error.F:39-50
  integer, parameter, public :: &
       SEAICE_ERROR_COL_VERT_THERM = 10, &
       SEAICE_ERROR_COL_ITD_THERM  = 11, &
       SEAICE_ERROR_COL_RIDGING    = 12, &
       SEAICE_ERROR_COL_BGC        = 13

  integer, parameter, public :: &
       SEAICE_ERROR_IR_NEG_AREA    = 20, &
       SEAICE_ERROR_IR_NEG_MASS    = 21, &
       SEAICE_ERROR_IR_MONO        = 22, &
       SEAICE_ERROR_IR_CONS        = 23

  public :: seaice_critical_error_write_block

contains

  subroutine seaice_critical_error_write_block(domain, block)
    type(domain_type), intent(in) :: domain
    type(block_type), intent(in) :: block
    ! the abort_block stream: output only
  end subroutine seaice_critical_error_write_block

end module seaice_error

module seaice_advection_upwind
  use mpas_derived_types
  use mpas_timekeeping
  implicit none
  private
  public :: seaice_init_advection_upwind, seaice_run_advection_upwind

contains

  subroutine seaice_init_advection_upwind(domain)
    type(domain_type), intent(inout) :: domain
    print *, 'oracle stub: upwind advection cannot run in v8.4.2 (undeclared configs and mesh fields)'
    stop 6
  end subroutine seaice_init_advection_upwind

  subroutine seaice_run_advection_upwind(domain, clock)
    type(domain_type), intent(inout) :: domain
    type(MPAS_Clock_type), intent(in) :: clock
    print *, 'oracle stub: upwind advection cannot run in v8.4.2 (undeclared configs and mesh fields)'
    stop 6
  end subroutine seaice_run_advection_upwind

end module seaice_advection_upwind
