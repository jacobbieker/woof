! Framework stand-ins for the MPAS-Ocean tendency oracle.
!
! The real mpas_derived_types / mpas_pool_routines / mpas_timer pull in PIO,
! SMIOLf and MPI, none of which the routines under test need.  The routines
! (compiled verbatim) only look fields, dimensions and configs up in pools by
! name.  These stand-ins hold the named entries and carry no arithmetic.  A
! lookup of a name the driver did not put is a stop that names it.
!
! The same name-lookup pool as tools/mpas_seaice_oracle/stubs.F90, with the
! entry kinds the ocean routines also read: character and real configs, the
! 1-D dimension arrays (nCellsArray, nEdgesArray, nVerticesArray), and the
! scratch fields hmix del4 allocates (field2DReal, field3DReal).
!
! The operator modules (mpas_vector/matrix/tensor_operations) are USEd by the
! hmix modules only for the tensor forms (config_use_mom_del2_tensor,
! config_use_mom_del4_tensor), which these oracles do not drive: their
! stand-ins stop by name if called.

module mpas_log
  ! ocn_vel_pressure_grad_init logs the pressure-gradient type: an
  ! informational line is dropped. An error or critical message stops,
  ! as MPAS_LOG_CRIT aborts the model.  As in MPAS, mpas_pool_routines (and
  ! through it mpas_derived_types) re-exports these names.
  use mpas_kind_types
  implicit none
  private
  public :: mpas_log_write, MPAS_LOG_WARN, MPAS_LOG_ERR, MPAS_LOG_CRIT

  integer, parameter :: MPAS_LOG_WARN = 2, MPAS_LOG_ERR = 3, MPAS_LOG_CRIT = 4

contains

  subroutine mpas_log_write(message, messageType, masterOnly, flushNow, intArgs, realArgs, logicArgs, err)
    character(len=*), intent(in) :: message
    integer, optional, intent(in) :: messageType
    logical, optional, intent(in) :: masterOnly, flushNow
    integer, dimension(:), optional, intent(in) :: intArgs
    real(kind=RKIND), dimension(:), optional, intent(in) :: realArgs
    logical, dimension(:), optional, intent(in) :: logicArgs
    integer, optional, intent(out) :: err
    if (present(err)) err = 0
    if (present(messageType)) then
      if (messageType == MPAS_LOG_ERR .or. messageType == MPAS_LOG_CRIT) then
        print *, 'oracle stub: mpas_log_write: ', trim(message)
        stop 7
      end if
    end if
  end subroutine mpas_log_write

end module mpas_log

module mpas_pool_routines
  use mpas_kind_types
  use mpas_log
  implicit none
  private
  public :: mpas_log_write, MPAS_LOG_WARN, MPAS_LOG_ERR, MPAS_LOG_CRIT
  public :: MPAS_pool_type, mpas_pool_get_subpool, mpas_pool_get_array, &
            mpas_pool_get_dimension, mpas_pool_get_config, &
            stub_pool_put_subpool, stub_pool_put_array, &
            stub_pool_put_dimension, stub_pool_put_config, &
            field1DReal, field2DReal, field3DReal, mpas_pool_get_field, stub_pool_put_field, &
            mpas_allocate_scratch_field, mpas_deallocate_scratch_field, &
            mpas_pool_iterator_type, MPAS_POOL_FIELD, mpas_pool_begin_iteration, mpas_pool_get_next_member

  integer, parameter :: maxEntries = 192

  ! a scratch field: the driver sets dimSizes, mpas_allocate_scratch_field
  ! allocates array to them
  type field1DReal
    real(kind=RKIND), dimension(:), pointer :: array => null()
    integer, dimension(1) :: dimSizes = 0
  end type field1DReal

  type field2DReal
    real(kind=RKIND), dimension(:,:), pointer :: array => null()
    integer, dimension(2) :: dimSizes = 0
  end type field2DReal

  type field3DReal
    real(kind=RKIND), dimension(:,:,:), pointer :: array => null()
    integer, dimension(3) :: dimSizes = 0
  end type field3DReal

  ! ocn_tend_tracer walks the tracer groups with a pool iterator; the
  ! oracles do not call it, and the iteration stops by name
  integer, parameter :: MPAS_POOL_FIELD = 1004

  type mpas_pool_iterator_type
    character(len=StrKIND) :: memberName = ''
    integer :: memberType = 0
  end type mpas_pool_iterator_type

  type pool_entry
    character(len=64) :: name = ''
    real(kind=RKIND), pointer :: r0 => null()
    real(kind=RKIND), dimension(:), pointer :: r1 => null()
    real(kind=RKIND), dimension(:,:), pointer :: r2 => null()
    real(kind=RKIND), dimension(:,:,:), pointer :: r3 => null()
    integer, pointer :: i0 => null()
    integer, dimension(:), pointer :: i1 => null()
    integer, dimension(:,:), pointer :: i2 => null()
    integer, dimension(:), pointer :: d1 => null()
    logical, pointer :: l0 => null()
    character(len=StrKIND), pointer :: c0 => null()
    type(MPAS_pool_type), pointer :: sub => null()
    type(field1DReal), pointer :: f1 => null()
    type(field2DReal), pointer :: f2 => null()
    type(field3DReal), pointer :: f3 => null()
  end type pool_entry

  type MPAS_pool_type
    type(pool_entry), dimension(maxEntries) :: e
    integer :: n = 0
    integer :: cursor = 0   ! mpas_pool_begin_iteration / mpas_pool_get_next_member
  end type MPAS_pool_type

  interface mpas_pool_get_config
    module procedure get_config_l, get_config_c, get_config_r, get_config_i
  end interface mpas_pool_get_config

  interface mpas_pool_get_dimension
    module procedure get_dimension_0d, get_dimension_1d
  end interface mpas_pool_get_dimension

  interface stub_pool_put_config
    module procedure put_config_l, put_config_c, put_config_r, put_config_i
  end interface stub_pool_put_config

  interface stub_pool_put_dimension
    module procedure put_dimension_0d, put_dimension_1d
  end interface stub_pool_put_dimension

  interface mpas_pool_get_array
    module procedure get_r0, get_r1, get_r2, get_r3, get_i1, get_i2
  end interface mpas_pool_get_array

  interface mpas_pool_get_field
    module procedure get_field_1d, get_field_2d, get_field_3d
  end interface mpas_pool_get_field

  interface stub_pool_put_field
    module procedure put_field_1d, put_field_2d, put_field_3d
  end interface stub_pool_put_field

  interface mpas_allocate_scratch_field
    module procedure allocate_scratch_1d, allocate_scratch_2d, allocate_scratch_3d
  end interface mpas_allocate_scratch_field

  interface mpas_deallocate_scratch_field
    module procedure deallocate_scratch_1d, deallocate_scratch_2d, deallocate_scratch_3d
  end interface mpas_deallocate_scratch_field

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

  ! a pool holds one time level: a lookup of another stops by name
  subroutine one_time_level(name, timeLevel)
    character(len=*), intent(in) :: name
    integer, intent(in), optional :: timeLevel
    if (present(timeLevel)) then
      if (timeLevel /= 1) then
        print *, 'oracle stub: time level ', timeLevel, ' of ', trim(name)
        stop 6
      end if
    end if
  end subroutine one_time_level

  subroutine mpas_pool_get_subpool(pool, name, sub)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    type(MPAS_pool_type), pointer :: sub
    sub => pool % e(find(pool, name)) % sub
  end subroutine mpas_pool_get_subpool

  subroutine get_dimension_0d(pool, name, dim)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    integer, pointer :: dim
    dim => pool % e(find(pool, name)) % i0
  end subroutine get_dimension_0d

  subroutine get_dimension_1d(pool, name, dim)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    integer, dimension(:), pointer :: dim
    dim => pool % e(find(pool, name)) % d1
  end subroutine get_dimension_1d

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

  subroutine put_config_i(pool, name, value)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    integer, pointer :: value
    integer :: i
    i = slot(pool, name)
    pool % e(i) % i0 => value
  end subroutine put_config_i

  subroutine get_config_r(pool, name, value)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    real(kind=RKIND), pointer :: value
    value => pool % e(find(pool, name)) % r0
  end subroutine get_config_r

  subroutine get_r0(pool, name, a)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    real(kind=RKIND), pointer :: a
    a => pool % e(find(pool, name)) % r0
  end subroutine get_r0

  subroutine get_r1(pool, name, a, timeLevel)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    real(kind=RKIND), dimension(:), pointer :: a
    integer, intent(in), optional :: timeLevel
    call one_time_level(name, timeLevel)
    a => pool % e(find(pool, name)) % r1
  end subroutine get_r1

  subroutine get_r2(pool, name, a, timeLevel)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    real(kind=RKIND), dimension(:,:), pointer :: a
    integer, intent(in), optional :: timeLevel
    call one_time_level(name, timeLevel)
    a => pool % e(find(pool, name)) % r2
  end subroutine get_r2

  subroutine get_r3(pool, name, a, timeLevel)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    real(kind=RKIND), dimension(:,:,:), pointer :: a
    integer, intent(in), optional :: timeLevel
    call one_time_level(name, timeLevel)
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
    integer :: i
    i = slot(pool, name)
    pool % e(i) % sub => sub
  end subroutine stub_pool_put_subpool

  subroutine put_dimension_0d(pool, name, dim)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    integer, pointer :: dim
    integer :: i
    i = slot(pool, name)
    pool % e(i) % i0 => dim
  end subroutine put_dimension_0d

  subroutine put_dimension_1d(pool, name, dim)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    integer, dimension(:), pointer :: dim
    integer :: i
    i = slot(pool, name)
    pool % e(i) % d1 => dim
  end subroutine put_dimension_1d

  subroutine put_config_l(pool, name, value)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    logical, pointer :: value
    integer :: i
    i = slot(pool, name)
    pool % e(i) % l0 => value
  end subroutine put_config_l

  subroutine put_config_c(pool, name, value)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    character(len=StrKIND), pointer :: value
    integer :: i
    i = slot(pool, name)
    pool % e(i) % c0 => value
  end subroutine put_config_c

  subroutine put_config_r(pool, name, value)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    real(kind=RKIND), pointer :: value
    integer :: i
    i = slot(pool, name)
    pool % e(i) % r0 => value
  end subroutine put_config_r

  subroutine put_r0(pool, name, a)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    real(kind=RKIND), pointer :: a
    integer :: i
    i = slot(pool, name)
    pool % e(i) % r0 => a
  end subroutine put_r0

  subroutine put_r1(pool, name, a)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    real(kind=RKIND), dimension(:), pointer :: a
    integer :: i
    i = slot(pool, name)
    pool % e(i) % r1 => a
  end subroutine put_r1

  subroutine put_r2(pool, name, a)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    real(kind=RKIND), dimension(:,:), pointer :: a
    integer :: i
    i = slot(pool, name)
    pool % e(i) % r2 => a
  end subroutine put_r2

  subroutine put_r3(pool, name, a)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    real(kind=RKIND), dimension(:,:,:), pointer :: a
    integer :: i
    i = slot(pool, name)
    pool % e(i) % r3 => a
  end subroutine put_r3

  subroutine put_i1(pool, name, a)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    integer, dimension(:), pointer :: a
    integer :: i
    i = slot(pool, name)
    pool % e(i) % i1 => a
  end subroutine put_i1

  subroutine put_i2(pool, name, a)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    integer, dimension(:,:), pointer :: a
    integer :: i
    i = slot(pool, name)
    pool % e(i) % i2 => a
  end subroutine put_i2

  ! Iteration visits every entry in the order it was put; a 3-D real array
  ! (a tracer group) is an MPAS_POOL_FIELD member, anything else another type.
  ! ocn_tend_tracer iterates the tracers pool this way.
  subroutine mpas_pool_begin_iteration(pool)
    type(MPAS_pool_type), intent(in) :: pool
    type(MPAS_pool_type), pointer :: p
    p => pool_target(pool)
    p % cursor = 0
  end subroutine mpas_pool_begin_iteration

  logical function mpas_pool_get_next_member(pool, iterator)
    type(MPAS_pool_type), intent(in) :: pool
    type(mpas_pool_iterator_type), intent(inout) :: iterator
    type(MPAS_pool_type), pointer :: p
    p => pool_target(pool)
    mpas_pool_get_next_member = .false.
    if (p % cursor >= p % n) return
    p % cursor = p % cursor + 1
    iterator % memberName = p % e(p % cursor) % name
    iterator % memberType = 0
    if (associated(p % e(p % cursor) % r3)) iterator % memberType = MPAS_POOL_FIELD
    mpas_pool_get_next_member = .true.
  end function mpas_pool_get_next_member

  ! the pool object itself, so the cursor can move under an intent(in) dummy
  function pool_target(pool) result(p)
    type(MPAS_pool_type), intent(in), target :: pool
    type(MPAS_pool_type), pointer :: p
    p => pool
  end function pool_target

  subroutine get_field_1d(pool, name, f)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    type(field1DReal), pointer :: f
    f => pool % e(find(pool, name)) % f1
  end subroutine get_field_1d

  subroutine put_field_1d(pool, name, f)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    type(field1DReal), pointer :: f
    integer :: i
    i = slot(pool, name)
    pool % e(i) % f1 => f
  end subroutine put_field_1d

  subroutine get_field_2d(pool, name, f, timeLevel)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    type(field2DReal), pointer :: f
    integer, intent(in), optional :: timeLevel
    call one_time_level(name, timeLevel)
    f => pool % e(find(pool, name)) % f2
  end subroutine get_field_2d

  subroutine get_field_3d(pool, name, f)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    type(field3DReal), pointer :: f
    f => pool % e(find(pool, name)) % f3
  end subroutine get_field_3d

  subroutine put_field_2d(pool, name, f)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    type(field2DReal), pointer :: f
    integer :: i
    i = slot(pool, name)
    pool % e(i) % f2 => f
  end subroutine put_field_2d

  subroutine put_field_3d(pool, name, f)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    type(field3DReal), pointer :: f
    integer :: i
    i = slot(pool, name)
    pool % e(i) % f3 => f
  end subroutine put_field_3d

  ! The array is filled with huge(): a word the routine reads before it
  ! writes it would show in the output.
  subroutine allocate_scratch_1d(f, single_block_in, init_array_in)
    type(field1DReal), pointer :: f
    logical, intent(in), optional :: single_block_in, init_array_in
    allocate(f % array(f % dimSizes(1)))
    f % array = huge(1.0_RKIND)
  end subroutine allocate_scratch_1d

  subroutine allocate_scratch_2d(f, single_block_in, init_array_in)
    type(field2DReal), pointer :: f
    logical, intent(in), optional :: single_block_in, init_array_in
    allocate(f % array(f % dimSizes(1), f % dimSizes(2)))
    f % array = huge(1.0_RKIND)
  end subroutine allocate_scratch_2d

  subroutine allocate_scratch_3d(f, single_block_in, init_array_in)
    type(field3DReal), pointer :: f
    logical, intent(in), optional :: single_block_in, init_array_in
    allocate(f % array(f % dimSizes(1), f % dimSizes(2), f % dimSizes(3)))
    f % array = huge(1.0_RKIND)
  end subroutine allocate_scratch_3d

  ! A field freed twice (ocn_diagnostic_solve frees its two vertex vorticity
  ! fields again at its end) is freed once, as MPAS's own routine does.
  subroutine deallocate_scratch_1d(f, single_block_in)
    type(field1DReal), pointer :: f
    logical, intent(in), optional :: single_block_in
    if (associated(f % array)) deallocate(f % array)
  end subroutine deallocate_scratch_1d

  subroutine deallocate_scratch_2d(f, single_block_in)
    type(field2DReal), pointer :: f
    logical, intent(in), optional :: single_block_in
    if (associated(f % array)) deallocate(f % array)
  end subroutine deallocate_scratch_2d

  subroutine deallocate_scratch_3d(f, single_block_in)
    type(field3DReal), pointer :: f
    logical, intent(in), optional :: single_block_in
    deallocate(f % array)
  end subroutine deallocate_scratch_3d

end module mpas_pool_routines

module mpas_derived_types
  use mpas_kind_types
  use mpas_pool_routines
  implicit none
  public

  type mpas_dminfo_type
    integer :: nprocs = 1
  end type mpas_dminfo_type

  ! the equation of state declares a dm_info it never uses
  type dm_info
    integer :: nprocs = 1
  end type dm_info

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


module mpas_threading
  implicit none
  private
  public :: mpas_threading_barrier

contains

  subroutine mpas_threading_barrier()
    ! one thread
  end subroutine mpas_threading_barrier

end module mpas_threading


module mpas_vector_operations
  use mpas_kind_types
  use mpas_pool_routines
  implicit none
  private
  public :: mpas_vector_R3Cell_to_normalVectorEdge, mpas_vector_R3Cell_to_2DEdge

contains

  subroutine mpas_vector_R3Cell_to_normalVectorEdge(vectorR3Cell, meshPool, includeHalo, normalVectorEdge)
    real(kind=RKIND), dimension(:,:,:), intent(in) :: vectorR3Cell
    type(MPAS_pool_type), intent(in) :: meshPool
    logical, intent(in) :: includeHalo
    real(kind=RKIND), dimension(:,:), intent(out) :: normalVectorEdge
    print *, 'oracle stub: mpas_vector_R3Cell_to_normalVectorEdge is not compiled here'
    stop 8
  end subroutine mpas_vector_R3Cell_to_normalVectorEdge

  subroutine mpas_vector_R3Cell_to_2DEdge(vectorR3Cell, meshPool, edgeTangentVectors, includeHalo, &
                                          normalVectorEdge, tangentialVectorEdge)
    real(kind=RKIND), dimension(:,:,:), intent(in) :: vectorR3Cell
    type(MPAS_pool_type), intent(in) :: meshPool
    real(kind=RKIND), dimension(:,:), intent(in) :: edgeTangentVectors
    logical, intent(in) :: includeHalo
    real(kind=RKIND), dimension(:,:), intent(out) :: normalVectorEdge, tangentialVectorEdge
    print *, 'oracle stub: mpas_vector_R3Cell_to_2DEdge is not compiled here'
    stop 8
  end subroutine mpas_vector_R3Cell_to_2DEdge

end module mpas_vector_operations

module mpas_matrix_operations
  use mpas_kind_types
  use mpas_pool_routines
  implicit none
  private
  public :: mpas_matrix_cell_to_edge

contains

  subroutine mpas_matrix_cell_to_edge(matrixCell, meshPool, includeHalo, matrixEdge)
    real(kind=RKIND), dimension(:,:,:), intent(in) :: matrixCell
    type(MPAS_pool_type), intent(in) :: meshPool
    logical, intent(in) :: includeHalo
    real(kind=RKIND), dimension(:,:,:), intent(out) :: matrixEdge
    print *, 'oracle stub: mpas_matrix_cell_to_edge is not compiled here'
    stop 8
  end subroutine mpas_matrix_cell_to_edge

end module mpas_matrix_operations

module mpas_tensor_operations
  use mpas_kind_types
  use mpas_pool_routines
  implicit none
  private
  public :: mpas_strain_rate_R3Cell, mpas_divergence_of_tensor_R3Cell

contains

  subroutine mpas_strain_rate_R3Cell(normalVelocity, tangentialVelocity, meshPool, edgeSignOnCell, &
                                     edgeTangentVectors, includeHalo, outerProductEdge, strainRateR3Cell)
    real(kind=RKIND), dimension(:,:), intent(in) :: normalVelocity, tangentialVelocity
    type(MPAS_pool_type), intent(in) :: meshPool
    integer, dimension(:,:), intent(in) :: edgeSignOnCell
    real(kind=RKIND), dimension(:,:), intent(in) :: edgeTangentVectors
    logical, intent(in) :: includeHalo
    real(kind=RKIND), dimension(:,:,:), intent(out) :: outerProductEdge, strainRateR3Cell
    print *, 'oracle stub: mpas_strain_rate_R3Cell is not compiled here'
    stop 8
  end subroutine mpas_strain_rate_R3Cell

  subroutine mpas_divergence_of_tensor_R3Cell(strainRateR3Edge, meshPool, edgeSignOnCell, includeHalo, &
                                              divTensorR3Cell)
    real(kind=RKIND), dimension(:,:,:), intent(in) :: strainRateR3Edge
    type(MPAS_pool_type), intent(in) :: meshPool
    integer, dimension(:,:), intent(in) :: edgeSignOnCell
    logical, intent(in) :: includeHalo
    real(kind=RKIND), dimension(:,:,:), intent(out) :: divTensorR3Cell
    print *, 'oracle stub: mpas_divergence_of_tensor_R3Cell is not compiled here'
    stop 8
  end subroutine mpas_divergence_of_tensor_R3Cell

end module mpas_tensor_operations

module mpas_timekeeping
  ! ocn_forcing and ocn_surface_bulk_forcing USE it; the routines the oracles
  ! call read no time manager routine
  implicit none
end module mpas_timekeeping

! modules the ocean units USE without calling anything in them here
module mpas_sort
end module mpas_sort

module mpas_hash
end module mpas_hash

module mpas_io_units
end module mpas_io_units
