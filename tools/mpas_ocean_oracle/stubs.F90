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
            mpas_pool_iterator_type, MPAS_POOL_FIELD, mpas_pool_begin_iteration, mpas_pool_get_next_member, &
            mpas_pool_get_package, mpas_pool_shift_time_levels, mpas_pool_initialize_time_levels

  integer, parameter :: maxEntries = 320

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
    module procedure get_r0, get_r1, get_r2, get_r3, get_i0, get_i1, get_i2, get_c0
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

  ! A field's time level 1 is its name; level N > 1 is the entry NAME@N
  ! (the split-explicit stage's two levels).  A lookup of a level the driver
  ! did not put stops by name.
  function level_name(name, timeLevel) result(key)
    character(len=*), intent(in) :: name
    integer, intent(in), optional :: timeLevel
    character(len=64) :: key
    key = name
    if (present(timeLevel)) then
      if (timeLevel < 1 .or. timeLevel > 9) then
        print *, 'oracle stub: time level ', timeLevel, ' of ', trim(name)
        stop 6
      end if
      if (timeLevel > 1) key = trim(name) // '@' // achar(48 + timeLevel)
    end if
  end function level_name

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

  ! a package's switch, which the driver puts in the packages pool by name
  subroutine mpas_pool_get_package(pool, name, value)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    logical, pointer :: value
    value => pool % e(find(pool, name)) % l0
  end subroutine mpas_pool_get_package

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

  ! a character field (xtime): the stream puts it as a character config of the pool
  subroutine get_c0(pool, name, a)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    character(len=StrKIND), pointer :: a
    a => pool % e(find(pool, name)) % c0
  end subroutine get_c0

  ! MPAS's mpas_pool_shift_time_levels for two levels: each field NAME and its
  ! level NAME@2 trade arrays, in the pool and its subpools
  recursive subroutine mpas_pool_shift_time_levels(pool)
    type(MPAS_pool_type), intent(inout) :: pool
    integer :: i, j
    real(kind=RKIND), dimension(:), pointer :: t1
    real(kind=RKIND), dimension(:,:), pointer :: t2
    real(kind=RKIND), dimension(:,:,:), pointer :: t3
    do i = 1, pool % n
      if (associated(pool % e(i) % sub)) then
        call mpas_pool_shift_time_levels(pool % e(i) % sub)
        cycle
      end if
      j = level_two(pool, i)
      if (j == 0) cycle
      t1 => pool % e(i) % r1; pool % e(i) % r1 => pool % e(j) % r1; pool % e(j) % r1 => t1
      t2 => pool % e(i) % r2; pool % e(i) % r2 => pool % e(j) % r2; pool % e(j) % r2 => t2
      t3 => pool % e(i) % r3; pool % e(i) % r3 => pool % e(j) % r3; pool % e(j) % r3 => t3
    end do
  end subroutine mpas_pool_shift_time_levels

  ! MPAS's mpas_pool_initialize_time_levels for two levels: NAME@2 takes NAME's words
  recursive subroutine mpas_pool_initialize_time_levels(pool)
    type(MPAS_pool_type), intent(inout) :: pool
    integer :: i, j
    do i = 1, pool % n
      if (associated(pool % e(i) % sub)) then
        call mpas_pool_initialize_time_levels(pool % e(i) % sub)
        cycle
      end if
      j = level_two(pool, i)
      if (j == 0) cycle
      if (associated(pool % e(i) % r1) .and. associated(pool % e(j) % r1)) pool % e(j) % r1 = pool % e(i) % r1
      if (associated(pool % e(i) % r2) .and. associated(pool % e(j) % r2)) pool % e(j) % r2 = pool % e(i) % r2
      if (associated(pool % e(i) % r3) .and. associated(pool % e(j) % r3)) pool % e(j) % r3 = pool % e(i) % r3
    end do
  end subroutine mpas_pool_initialize_time_levels

  ! the entry NAME@2 of entry i's field, or 0
  integer function level_two(pool, i)
    type(MPAS_pool_type), intent(in) :: pool
    integer, intent(in) :: i
    integer :: j
    level_two = 0
    if (index(pool % e(i) % name, '@') /= 0) return
    do j = 1, pool % n
      if (trim(pool % e(j) % name) == trim(pool % e(i) % name) // '@2') then
        level_two = j
        return
      end if
    end do
  end function level_two

  ! an integer scalar field (nAccumulatedCoupled): the stream puts it as a dimension
  subroutine get_i0(pool, name, a)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    integer, pointer :: a
    a => pool % e(find(pool, name)) % i0
  end subroutine get_i0

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
    a => pool % e(find(pool, level_name(name, timeLevel))) % r1
  end subroutine get_r1

  subroutine get_r2(pool, name, a, timeLevel)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    real(kind=RKIND), dimension(:,:), pointer :: a
    integer, intent(in), optional :: timeLevel
    a => pool % e(find(pool, level_name(name, timeLevel))) % r2
  end subroutine get_r2

  subroutine get_r3(pool, name, a, timeLevel)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    real(kind=RKIND), dimension(:,:,:), pointer :: a
    integer, intent(in), optional :: timeLevel
    a => pool % e(find(pool, level_name(name, timeLevel))) % r3
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
    ! a field's level NAME@N is not a member of its own
    if (associated(p % e(p % cursor) % r3) .and. index(p % e(p % cursor) % name, '@') == 0) &
         iterator % memberType = MPAS_POOL_FIELD
    mpas_pool_get_next_member = .true.
  end function mpas_pool_get_next_member

  ! the pool object itself, so the cursor can move under an intent(in) dummy
  function pool_target(pool) result(p)
    type(MPAS_pool_type), intent(in), target :: pool
    type(MPAS_pool_type), pointer :: p
    p => pool
  end function pool_target

  subroutine get_field_1d(pool, name, f, timeLevel)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    type(field1DReal), pointer :: f
    integer, intent(in), optional :: timeLevel
    f => pool % e(find(pool, level_name(name, timeLevel))) % f1
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
    f => pool % e(find(pool, level_name(name, timeLevel))) % f2
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

module mpas_timekeeping_types
  ! the time manager's types (whole seconds; mpas_timekeeping's routines)
  use mpas_kind_types
  implicit none
  public
  type MPAS_Time_type
    integer(kind=I8KIND) :: s = 0
  end type MPAS_Time_type
  type MPAS_TimeInterval_type
    integer(kind=I8KIND) :: s = 0
  end type MPAS_TimeInterval_type
  type MPAS_Clock_type
    integer(kind=I8KIND) :: start = 0, now = 0, step = 0
  end type MPAS_Clock_type
end module mpas_timekeeping_types

module mpas_derived_types
  use mpas_kind_types
  use mpas_timekeeping_types
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
    type(MPAS_pool_type), pointer :: dimensions => null()
    type(MPAS_pool_type), pointer :: configs => null()
    type(block_type), pointer :: next => null()
  end type block_type

  type domain_type
    type(block_type), pointer :: blocklist => null()
    type(MPAS_pool_type), pointer :: configs => null()
    type(MPAS_Clock_type), pointer :: clock => null()
    type(mpas_dminfo_type) :: dminfo
  end type domain_type

end module mpas_derived_types

module mpas_dmpar
  use mpas_kind_types
  use mpas_derived_types
  implicit none
  private
  public :: mpas_dmpar_min_real, mpas_dmpar_exch_group_create, mpas_dmpar_exch_group_add_field, &
            mpas_dmpar_exch_group_reuse_halo_exch, mpas_dmpar_exch_group_full_halo_exch, &
            mpas_dmpar_exch_group_destroy, mpas_dmpar_field_halo_exch, mpas_dmpar_exch_halo_field

  interface mpas_dmpar_exch_halo_field
    module procedure exch_halo_field1d, exch_halo_field2d, exch_halo_field3d
  end interface mpas_dmpar_exch_halo_field

contains

  subroutine mpas_dmpar_field_halo_exch(domain, fieldName, timeLevel, haloLayers)
    type(domain_type), intent(inout) :: domain
    character(len=*), intent(in) :: fieldName
    integer, optional, intent(in) :: timeLevel
    integer, dimension(:), optional, intent(in) :: haloLayers
  end subroutine mpas_dmpar_field_halo_exch

  subroutine exch_halo_field1d(field, haloLayersIn)
    type(field1DReal), pointer :: field
    integer, dimension(:), intent(in), optional :: haloLayersIn
  end subroutine exch_halo_field1d

  subroutine exch_halo_field2d(field, haloLayersIn)
    type(field2DReal), pointer :: field
    integer, dimension(:), intent(in), optional :: haloLayersIn
  end subroutine exch_halo_field2d

  subroutine exch_halo_field3d(field, haloLayersIn)
    type(field3DReal), pointer :: field
    integer, dimension(:), intent(in), optional :: haloLayersIn
  end subroutine exch_halo_field3d

  ! One block on one rank: a halo exchange is the identity.
  subroutine mpas_dmpar_exch_group_create(domain, groupName, iErr)
    type(domain_type), intent(inout) :: domain
    character(len=*), intent(in) :: groupName
    integer, optional, intent(out) :: iErr
    if (present(iErr)) iErr = 0
  end subroutine mpas_dmpar_exch_group_create

  subroutine mpas_dmpar_exch_group_add_field(domain, groupName, fieldName, timeLevel, haloLayers, iErr)
    type(domain_type), intent(inout) :: domain
    character(len=*), intent(in) :: groupName, fieldName
    integer, optional, intent(in) :: timeLevel
    integer, dimension(:), optional, intent(in) :: haloLayers
    integer, optional, intent(out) :: iErr
    if (present(iErr)) iErr = 0
  end subroutine mpas_dmpar_exch_group_add_field

  subroutine mpas_dmpar_exch_group_reuse_halo_exch(domain, groupName, timeLevel, haloLayers, iErr)
    type(domain_type), intent(inout) :: domain
    character(len=*), intent(in) :: groupName
    integer, optional, intent(in) :: timeLevel
    integer, dimension(:), optional, intent(in) :: haloLayers
    integer, optional, intent(out) :: iErr
    if (present(iErr)) iErr = 0
  end subroutine mpas_dmpar_exch_group_reuse_halo_exch

  subroutine mpas_dmpar_exch_group_full_halo_exch(domain, groupName, withGPUAwareMPI, iErr)
    type(domain_type), intent(inout) :: domain
    character(len=*), intent(in) :: groupName
    logical, optional, intent(in) :: withGPUAwareMPI
    integer, optional, intent(out) :: iErr
    if (present(iErr)) iErr = 0
  end subroutine mpas_dmpar_exch_group_full_halo_exch

  subroutine mpas_dmpar_exch_group_destroy(domain, groupName, iErr)
    type(domain_type), intent(inout) :: domain
    character(len=*), intent(in) :: groupName
    integer, optional, intent(out) :: iErr
    if (present(iErr)) iErr = 0
  end subroutine mpas_dmpar_exch_group_destroy

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
  ! The time manager the forward-mode oracle reaches, on whole seconds: a time
  ! is the seconds since 0001-01-01_00:00:00 on the proleptic Gregorian
  ! calendar (MPAS's default), an interval a count of seconds.  It parses and
  ! writes MPAS's strings ('YYYY-MM-DD_hh:mm:ss', '[DDDD_]hh:mm:ss').
  ! mpas_get_timeInterval's dt is the seconds as a real, as MPAS gives a whole
  ! interval.  ocn_forcing and ocn_surface_bulk_forcing USE the module and
  ! call nothing in it.
  use mpas_kind_types
  use mpas_timekeeping_types
  implicit none
  private
  public :: MPAS_Time_type, MPAS_TimeInterval_type, MPAS_Clock_type, MPAS_NOW, MPAS_START_TIME, &
            mpas_get_clock_time, mpas_set_timeInterval, mpas_interval_division, mpas_set_time, mpas_get_time, &
            mpas_get_timeInterval, mpas_advance_clock, mpas_set_clock, operator(>), operator(-)

  integer, parameter :: MPAS_NOW = 1, MPAS_START_TIME = 2

  interface operator(>)
    module procedure interval_gt
  end interface
  interface operator(-)
    module procedure time_minus
  end interface

contains

  logical function interval_gt(a, b)
    type(MPAS_TimeInterval_type), intent(in) :: a, b
    interval_gt = a % s > b % s
  end function interval_gt

  function time_minus(a, b) result(d)
    type(MPAS_Time_type), intent(in) :: a, b
    type(MPAS_TimeInterval_type) :: d
    d % s = a % s - b % s
  end function time_minus

  function mpas_get_clock_time(clock, whichTime, ierr) result(t)
    type(MPAS_Clock_type), intent(in) :: clock
    integer, intent(in) :: whichTime
    integer, intent(out), optional :: ierr
    type(MPAS_Time_type) :: t
    t % s = clock % now
    if (whichTime == MPAS_START_TIME) t % s = clock % start
    if (present(ierr)) ierr = 0
  end function mpas_get_clock_time

  subroutine mpas_set_clock(clock, startString, stepString)
    type(MPAS_Clock_type), intent(inout) :: clock
    character(len=*), intent(in) :: startString, stepString
    type(MPAS_Time_type) :: t
    type(MPAS_TimeInterval_type) :: dt
    call mpas_set_time(t, dateTimeString=startString)
    call mpas_set_timeInterval(dt, timeString=stepString)
    clock % start = t % s
    clock % now = t % s
    clock % step = dt % s
  end subroutine mpas_set_clock

  subroutine mpas_advance_clock(clock)
    type(MPAS_Clock_type), intent(inout) :: clock
    clock % now = clock % now + clock % step
  end subroutine mpas_advance_clock

  subroutine mpas_set_timeInterval(interval, S, timeString, ierr)
    type(MPAS_TimeInterval_type), intent(out) :: interval
    integer, intent(in), optional :: S
    character(len=*), intent(in), optional :: timeString
    integer, intent(out), optional :: ierr
    integer :: u, d, h, m, sec
    interval % s = 0
    if (present(S)) interval % s = S
    if (present(timeString)) then
      u = index(timeString, '_')
      d = 0
      if (u > 0) read(timeString(1:u-1), *) d
      read(timeString(u+1:u+2), *) h
      read(timeString(u+4:u+5), *) m
      read(timeString(u+7:u+8), *) sec
      interval % s = ((int(d, I8KIND) * 24 + h) * 60 + m) * 60 + sec
    end if
    if (present(ierr)) ierr = 0
  end subroutine mpas_set_timeInterval

  subroutine mpas_interval_division(ref_time, num, den, n, rem)
    type(MPAS_Time_type), intent(in) :: ref_time
    type(MPAS_TimeInterval_type), intent(in) :: num, den
    integer(kind=I8KIND), intent(out) :: n
    type(MPAS_TimeInterval_type), intent(out) :: rem
    n = num % s / den % s
    rem % s = num % s - n * den % s
  end subroutine mpas_interval_division

  subroutine mpas_get_timeInterval(interval, dt, ierr)
    type(MPAS_TimeInterval_type), intent(in) :: interval
    real(kind=RKIND), intent(out), optional :: dt
    integer, intent(out), optional :: ierr
    if (present(dt)) dt = real(interval % s, RKIND)
    if (present(ierr)) ierr = 0
  end subroutine mpas_get_timeInterval

  ! days since 0001-01-01 on the proleptic Gregorian calendar
  integer(kind=I8KIND) function civil_days(y, mo, d)
    integer, intent(in) :: y, mo, d
    integer(kind=I8KIND) :: yy, era, yoe, doy, doe
    yy = y
    if (mo <= 2) yy = yy - 1
    era = yy / 400
    if (yy < 0 .and. mod(yy, 400_I8KIND) /= 0) era = era - 1
    yoe = yy - era * 400
    doy = (153 * (mo + merge(-3, 9, mo > 2)) + 2) / 5 + d - 1
    doe = yoe * 365 + yoe / 4 - yoe / 100 + doy
    civil_days = era * 146097 + doe - 306
  end function civil_days

  subroutine mpas_set_time(curr_time, dateTimeString, ierr)
    type(MPAS_Time_type), intent(out) :: curr_time
    character(len=*), intent(in) :: dateTimeString
    integer, intent(out), optional :: ierr
    integer :: y, mo, d, h, m, sec
    read(dateTimeString(1:4), *) y
    read(dateTimeString(6:7), *) mo
    read(dateTimeString(9:10), *) d
    read(dateTimeString(12:13), *) h
    read(dateTimeString(15:16), *) m
    read(dateTimeString(18:19), *) sec
    curr_time % s = ((civil_days(y, mo, d) * 24 + h) * 60 + m) * 60 + sec
    if (present(ierr)) ierr = 0
  end subroutine mpas_set_time

  subroutine mpas_get_time(curr_time, dateTimeString, ierr)
    type(MPAS_Time_type), intent(in) :: curr_time
    character(len=*), intent(out) :: dateTimeString
    integer, intent(out), optional :: ierr
    integer(kind=I8KIND) :: days, rest, z, era, doe, yoe, y, doy, mp, d, mo
    days = curr_time % s / 86400
    rest = curr_time % s - days * 86400
    z = days + 306
    era = z / 146097
    if (z < 0 .and. mod(z, 146097_I8KIND) /= 0) era = era - 1
    doe = z - era * 146097
    yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe / 4 - yoe / 100)
    mp = (5 * doy + 2) / 153
    d = doy - (153 * mp + 2) / 5 + 1
    mo = mp + 3
    if (mo > 12) mo = mo - 12
    if (mo <= 2) y = y + 1
    write(dateTimeString, '(i4.4,"-",i2.2,"-",i2.2,"_",i2.2,":",i2.2,":",i2.2)') y, mo, d, rest / 3600, &
         mod(rest, 3600_I8KIND) / 60, mod(rest, 60_I8KIND)
    if (present(ierr)) ierr = 0
  end subroutine mpas_get_time

end module mpas_timekeeping

! modules the ocean units USE without calling anything in them here
module mpas_sort
end module mpas_sort

module mpas_hash
end module mpas_hash

module mpas_io_units
end module mpas_io_units
