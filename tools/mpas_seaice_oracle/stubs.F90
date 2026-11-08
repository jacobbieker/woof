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

  subroutine mpas_pool_get_config(pool, name, value)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    logical, pointer :: value
    value => pool % e(find(pool, name)) % l0
  end subroutine mpas_pool_get_config

  subroutine get_r0(pool, name, a)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    real(kind=RKIND), pointer :: a
    a => pool % e(find(pool, name)) % r0
  end subroutine get_r0

  subroutine get_r1(pool, name, a)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    real(kind=RKIND), dimension(:), pointer :: a
    a => pool % e(find(pool, name)) % r1
  end subroutine get_r1

  subroutine get_r2(pool, name, a)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    real(kind=RKIND), dimension(:,:), pointer :: a
    a => pool % e(find(pool, name)) % r2
  end subroutine get_r2

  subroutine get_r3(pool, name, a)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    real(kind=RKIND), dimension(:,:,:), pointer :: a
    a => pool % e(find(pool, name)) % r3
  end subroutine get_r3

  subroutine get_i1(pool, name, a)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    integer, dimension(:), pointer :: a
    a => pool % e(find(pool, name)) % i1
  end subroutine get_i1

  subroutine get_i2(pool, name, a)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
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

  subroutine stub_pool_put_config(pool, name, value)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    logical, pointer :: value
    pool % e(slot(pool, name)) % l0 => value
  end subroutine stub_pool_put_config

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

  type mpas_dminfo_type
    integer :: nprocs = 1
  end type mpas_dminfo_type

  type block_type
    type(MPAS_pool_type), pointer :: structs => null()
    type(MPAS_pool_type), pointer :: configs => null()
    type(block_type), pointer :: next => null()
  end type block_type

  type domain_type
    type(block_type), pointer :: blocklist => null()
    type(mpas_dminfo_type) :: dminfo
  end type domain_type

end module mpas_derived_types

module mpas_dmpar
  use mpas_kind_types
  use mpas_derived_types
  implicit none
  private
  public :: mpas_dmpar_min_real

contains

  subroutine mpas_dmpar_min_real(dminfo, inValue, outValue)
    type(mpas_dminfo_type), intent(in) :: dminfo
    real(kind=RKIND), intent(in) :: inValue
    real(kind=RKIND), intent(out) :: outValue
    ! one rank: the reduction is the identity
    outValue = inValue
  end subroutine mpas_dmpar_min_real

end module mpas_dmpar

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
