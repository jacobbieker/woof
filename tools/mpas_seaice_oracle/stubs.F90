! Framework stand-ins for the MPAS-Seaice EVP internal-stress oracle.
!
! The real mpas_derived_types / mpas_pool_routines / mpas_timer / mpas_dmpar
! pull in PIO, SMIOLf and MPI (mpas_derived_types USEs pio/smiolf, mpas_log
! USEs mpi_f08), none of which the routines under test need.  Those routines
! (mpas_seaice_velocity_solver_constitutive_relation.F and
! mpas_seaice_velocity_solver_variational.F, compiled verbatim) only walk a
! single-block domain and look fields up in pools by name.  These stand-ins
! hold named fields and the one-block domain; they carry no arithmetic.
! A lookup of a name the driver did not put is a stop that names it, so a
! field the routines read can never be silently absent.
!
! run_subcycle.F90 also drives mpas_seaice_velocity_solver.F (compiled
! verbatim; build.sh exposes its private routines with -Dprivate=public, a
! visibility change only), which walks block % dimensions and domain %
! configs, reads character/integer/real configs and names the halo-exchange,
! logging, mesh-interpolation, weak-form and column-package routines.  Those
! stand-ins stop with their name if reached (the driver sets
! config_use_halo_exch false and the variational scheme), except the timers
! and seaice_load_balance_timers, which time nothing and compute nothing.
!
! The two basis modules (wachspress, pwl) are USEd only by
! seaice_init_velocity_solver_variational, which the oracle does not call:
! the driver supplies the basis arrays as inputs.  Their stand-ins stop if
! reached.

module mpas_pool_routines
  use mpas_kind_types
  implicit none
  private
  public :: MPAS_pool_type, mpas_pool_get_subpool, mpas_pool_get_array, &
            mpas_pool_get_dimension, mpas_pool_get_config, &
            stub_pool_put_subpool, stub_pool_put_array, &
            stub_pool_put_dimension, stub_pool_put_config

  integer, parameter :: maxEntries = 64

  type pool_entry
    character(len=64) :: name = ''
    real(kind=RKIND), pointer :: r0 => null()
    character(len=StrKIND), pointer :: c0 => null()
    real(kind=RKIND), dimension(:), pointer :: r1 => null()
    real(kind=RKIND), dimension(:,:), pointer :: r2 => null()
    real(kind=RKIND), dimension(:,:,:), pointer :: r3 => null()
    integer, pointer :: i0 => null()
    integer, dimension(:), pointer :: i1 => null()
    integer, dimension(:,:), pointer :: i2 => null()
    logical, pointer :: l0 => null()
    type(MPAS_pool_type), pointer :: sub => null()
  end type pool_entry

  type MPAS_pool_type
    type(pool_entry), dimension(maxEntries) :: e
    integer :: n = 0
  end type MPAS_pool_type

  interface mpas_pool_get_config
    module procedure get_config_l, get_config_c, get_config_i, get_config_r
  end interface mpas_pool_get_config

  interface stub_pool_put_config
    module procedure put_config_l, put_config_c, put_config_i
  end interface stub_pool_put_config

  interface mpas_pool_get_array
    module procedure get_r0, get_r1, get_r2, get_r3, get_i1, get_i2
  end interface mpas_pool_get_array

  interface stub_pool_put_array
    module procedure put_r0, put_r1, put_r2, put_r3, put_i1, put_i2
  end interface stub_pool_put_array

contains

  integer function find(pool, name)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    integer :: i
    do i = 1, pool % n
      if (trim(pool % e(i) % name) == trim(name)) then
        find = i
        return
      end if
    end do
    print *, 'oracle stub: no pool entry named ', trim(name)
    stop 4
  end function find

  integer function slot(pool, name)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    pool % n = pool % n + 1
    if (pool % n > maxEntries) stop 5
    pool % e(pool % n) % name = name
    slot = pool % n
  end function slot

  subroutine mpas_pool_get_subpool(pool, name, sub)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    type(MPAS_pool_type), pointer :: sub
    sub => pool % e(find(pool, name)) % sub
  end subroutine mpas_pool_get_subpool

  subroutine mpas_pool_get_dimension(pool, name, dim)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    integer, pointer :: dim
    dim => pool % e(find(pool, name)) % i0
  end subroutine mpas_pool_get_dimension

  subroutine get_config_l(pool, name, value)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    logical, pointer :: value
    value => pool % e(find(pool, name)) % l0
  end subroutine get_config_l

  subroutine get_config_c(pool, name, value)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    character(len=StrKIND), pointer :: value
    value => pool % e(find(pool, name)) % c0
  end subroutine get_config_c

  subroutine get_config_i(pool, name, value)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    integer, pointer :: value
    value => pool % e(find(pool, name)) % i0
  end subroutine get_config_i

  subroutine get_config_r(pool, name, value)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    real(kind=RKIND), pointer :: value
    value => pool % e(find(pool, name)) % r0
  end subroutine get_config_r

  subroutine get_r0(pool, name, a, timeLevel)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    integer, intent(in), optional :: timeLevel
    real(kind=RKIND), pointer :: a
    a => pool % e(find(pool, name)) % r0
  end subroutine get_r0

  subroutine get_r1(pool, name, a, timeLevel)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    integer, intent(in), optional :: timeLevel
    real(kind=RKIND), dimension(:), pointer :: a
    a => pool % e(find(pool, name)) % r1
  end subroutine get_r1

  subroutine get_r2(pool, name, a, timeLevel)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    integer, intent(in), optional :: timeLevel
    real(kind=RKIND), dimension(:,:), pointer :: a
    a => pool % e(find(pool, name)) % r2
  end subroutine get_r2

  subroutine get_r3(pool, name, a, timeLevel)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    integer, intent(in), optional :: timeLevel
    real(kind=RKIND), dimension(:,:,:), pointer :: a
    a => pool % e(find(pool, name)) % r3
  end subroutine get_r3

  subroutine get_i1(pool, name, a, timeLevel)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    integer, intent(in), optional :: timeLevel
    integer, dimension(:), pointer :: a
    a => pool % e(find(pool, name)) % i1
  end subroutine get_i1

  subroutine get_i2(pool, name, a, timeLevel)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    integer, intent(in), optional :: timeLevel
    integer, dimension(:,:), pointer :: a
    a => pool % e(find(pool, name)) % i2
  end subroutine get_i2

  subroutine stub_pool_put_subpool(pool, name, sub)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    type(MPAS_pool_type), pointer :: sub
    pool % e(slot(pool, name)) % sub => sub
  end subroutine stub_pool_put_subpool

  subroutine stub_pool_put_dimension(pool, name, dim)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    integer, pointer :: dim
    pool % e(slot(pool, name)) % i0 => dim
  end subroutine stub_pool_put_dimension

  subroutine put_config_l(pool, name, value)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    logical, pointer :: value
    pool % e(slot(pool, name)) % l0 => value
  end subroutine put_config_l

  subroutine put_config_c(pool, name, value)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    character(len=StrKIND), pointer :: value
    pool % e(slot(pool, name)) % c0 => value
  end subroutine put_config_c

  subroutine put_config_i(pool, name, value)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    integer, pointer :: value
    pool % e(slot(pool, name)) % i0 => value
  end subroutine put_config_i

  subroutine put_r0(pool, name, a)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    real(kind=RKIND), pointer :: a
    pool % e(slot(pool, name)) % r0 => a
  end subroutine put_r0

  subroutine put_r1(pool, name, a)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    real(kind=RKIND), dimension(:), pointer :: a
    pool % e(slot(pool, name)) % r1 => a
  end subroutine put_r1

  subroutine put_r2(pool, name, a)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    real(kind=RKIND), dimension(:,:), pointer :: a
    pool % e(slot(pool, name)) % r2 => a
  end subroutine put_r2

  subroutine put_r3(pool, name, a)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    real(kind=RKIND), dimension(:,:,:), pointer :: a
    pool % e(slot(pool, name)) % r3 => a
  end subroutine put_r3

  subroutine put_i1(pool, name, a)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    integer, dimension(:), pointer :: a
    pool % e(slot(pool, name)) % i1 => a
  end subroutine put_i1

  subroutine put_i2(pool, name, a)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    integer, dimension(:,:), pointer :: a
    pool % e(slot(pool, name)) % i2 => a
  end subroutine put_i2

end module mpas_pool_routines

module mpas_derived_types
  use mpas_kind_types
  use mpas_pool_routines
  implicit none
  public

  integer, parameter :: MPAS_LOG_ERR = 3, MPAS_LOG_CRIT = 4
  integer, parameter :: MPAS_DMPAR_NOERR = 0

  type mpas_dminfo_type
    integer :: nprocs = 1
  end type mpas_dminfo_type

  type MPAS_Clock_type
    integer :: unused = 0
  end type MPAS_Clock_type

  type block_type
    type(MPAS_pool_type), pointer :: structs => null()
    type(MPAS_pool_type), pointer :: configs => null()
    type(MPAS_pool_type), pointer :: dimensions => null()
    type(block_type), pointer :: next => null()
  end type block_type

  type domain_type
    type(block_type), pointer :: blocklist => null()
    type(MPAS_pool_type), pointer :: configs => null()
    type(mpas_dminfo_type) :: dminfo
  end type domain_type

end module mpas_derived_types

module mpas_dmpar
  use mpas_kind_types
  use mpas_derived_types
  implicit none
  private
  public :: mpas_dmpar_min_real, mpas_dmpar_exch_group_create, mpas_dmpar_exch_group_add_field, &
            mpas_dmpar_exch_group_build_reusable_buffers, mpas_dmpar_exch_group_full_halo_exch, &
            mpas_dmpar_exch_group_reuse_halo_exch, mpas_dmpar_field_halo_exch

contains

  subroutine unreached(name)
    character(len=*), intent(in) :: name
    print *, 'oracle stub: ', name, ' is not part of this oracle'
    stop 6
  end subroutine unreached

  subroutine mpas_dmpar_exch_group_create(domain, groupName, iErr)
    type(domain_type), intent(inout) :: domain
    character(len=*), intent(in) :: groupName
    integer, optional, intent(out) :: iErr
    call unreached('mpas_dmpar_exch_group_create')
  end subroutine mpas_dmpar_exch_group_create

  subroutine mpas_dmpar_exch_group_add_field(domain, groupName, fieldName, iErr)
    type(domain_type), intent(inout) :: domain
    character(len=*), intent(in) :: groupName, fieldName
    integer, optional, intent(out) :: iErr
    call unreached('mpas_dmpar_exch_group_add_field')
  end subroutine mpas_dmpar_exch_group_add_field

  subroutine mpas_dmpar_exch_group_build_reusable_buffers(domain, groupName, iErr)
    type(domain_type), intent(inout) :: domain
    character(len=*), intent(in) :: groupName
    integer, optional, intent(out) :: iErr
    call unreached('mpas_dmpar_exch_group_build_reusable_buffers')
  end subroutine mpas_dmpar_exch_group_build_reusable_buffers

  subroutine mpas_dmpar_exch_group_full_halo_exch(domain, groupName, iErr)
    type(domain_type), intent(inout) :: domain
    character(len=*), intent(in) :: groupName
    integer, optional, intent(out) :: iErr
    call unreached('mpas_dmpar_exch_group_full_halo_exch')
  end subroutine mpas_dmpar_exch_group_full_halo_exch

  subroutine mpas_dmpar_exch_group_reuse_halo_exch(domain, groupName, iErr)
    type(domain_type), intent(inout) :: domain
    character(len=*), intent(in) :: groupName
    integer, optional, intent(out) :: iErr
    call unreached('mpas_dmpar_exch_group_reuse_halo_exch')
  end subroutine mpas_dmpar_exch_group_reuse_halo_exch

  subroutine mpas_dmpar_field_halo_exch(domain, fieldName)
    type(domain_type), intent(inout) :: domain
    character(len=*), intent(in) :: fieldName
    call unreached('mpas_dmpar_field_halo_exch')
  end subroutine mpas_dmpar_field_halo_exch

  subroutine mpas_dmpar_min_real(dminfo, inValue, outValue)
    type(mpas_dminfo_type), intent(in) :: dminfo
    real(kind=RKIND), intent(in) :: inValue
    real(kind=RKIND), intent(out) :: outValue
    ! one rank: the reduction is the identity
    outValue = inValue
  end subroutine mpas_dmpar_min_real

end module mpas_dmpar

module mpas_timekeeping
  ! MPAS_Clock_type lives in the mpas_derived_types stand-in; the velocity
  ! solver reads no time manager routine
  implicit none
end module mpas_timekeeping

module mpas_log
  use mpas_derived_types, only: MPAS_LOG_ERR, MPAS_LOG_CRIT
  implicit none
  private
  public :: mpas_log_write

contains

  subroutine mpas_log_write(message, messageType)
    character(len=*), intent(in) :: message
    integer, optional, intent(in) :: messageType
    print *, 'oracle stub: mpas_log_write: ', trim(message)
    stop 7
  end subroutine mpas_log_write

end module mpas_log

module mpas_timer
  implicit none
  private
  public :: mpas_timer_start, mpas_timer_stop

contains

  subroutine mpas_timer_start(timer_name, clear_timer_in)
    character(len=*), intent(in) :: timer_name
    logical, optional, intent(in) :: clear_timer_in
  end subroutine mpas_timer_start

  subroutine mpas_timer_stop(timer_name)
    character(len=*), intent(in) :: timer_name
  end subroutine mpas_timer_stop

end module mpas_timer

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

module seaice_diagnostics
  use mpas_derived_types
  implicit none
  private
  public :: seaice_load_balance_timers

contains

  subroutine seaice_load_balance_timers(domain, label)
    ! MPI barrier timing only: on one rank with no timers it does nothing
    type(domain_type), intent(in) :: domain
    character(len=*), intent(in) :: label
  end subroutine seaice_load_balance_timers

end module seaice_diagnostics

module seaice_mesh
  use mpas_derived_types
  use mpas_dmpar, only: mpas_dmpar_field_halo_exch
  implicit none
  private
  public :: seaice_interpolate_cell_to_vertex, seaice_interpolate_vertex_to_cell

contains

  subroutine seaice_interpolate_cell_to_vertex(mesh, variableVertex, variableCell)
    type(MPAS_pool_type), pointer, intent(in) :: mesh
    real(kind=RKIND), dimension(:), intent(out) :: variableVertex
    real(kind=RKIND), dimension(:), intent(in) :: variableCell
    print *, 'oracle stub: seaice_interpolate_cell_to_vertex is not part of this oracle'
    stop 6
  end subroutine seaice_interpolate_cell_to_vertex

  subroutine seaice_interpolate_vertex_to_cell(mesh, boundary, variableCell, variableVertex)
    type(MPAS_pool_type), pointer, intent(in) :: mesh, boundary
    real(kind=RKIND), dimension(:), intent(out) :: variableCell
    real(kind=RKIND), dimension(:), intent(in) :: variableVertex
    print *, 'oracle stub: seaice_interpolate_vertex_to_cell is not part of this oracle'
    stop 6
  end subroutine seaice_interpolate_vertex_to_cell

end module seaice_mesh

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
    ! of the velocity solver links only against a stand-in.  ice_strength is
    ! pre-subcycle work this oracle does not call.
    integer, intent(in) :: ncat
    real(kind=RKIND), intent(in) :: aice, vice, aice0
    real(kind=RKIND), dimension(:), intent(in) :: aicen, vicen
    real(kind=RKIND), intent(inout) :: strength
    print *, 'oracle stub: colpkg_ice_strength is not part of this oracle'
    stop 6
  end subroutine colpkg_ice_strength

end module ice_colpkg
