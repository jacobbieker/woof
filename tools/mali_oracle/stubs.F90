! Framework stand-ins for the MPAS land-ice (MALI) oracle.
!
! The real mpas_derived_types / mpas_pool_routines / mpas_timer pull in PIO,
! SMIOLf and MPI, none of which the routines under test need.  The routines
! (compiled verbatim) only look fields, dimensions and configs up in pools by
! name.  These stand-ins hold the named entries and carry no arithmetic.  A
! lookup of a name the driver did not put is a stop that names it.
!
! The name-lookup pool of tools/mpas_ocean_oracle/stubs.F90 (character, real
! and logical configs, 1-D dimension arrays), plus the logging, abort and
! timekeeping modules the land-ice modules USE.  mpas_log_write prints
! errors and records that one was written; it never alters a value.  With
! stub_log_capture set (run_forward.F90) it also keeps every other message's
! text, for li_compute_statistics's report.

module mpas_log
  use mpas_kind_types
  implicit none
  private
  public :: mpas_log_write, MPAS_LOG_OUT, MPAS_LOG_WARN, MPAS_LOG_ERR, MPAS_LOG_CRIT, stub_log_errors, &
            stub_log_capture, stub_log_lines, stub_log_count

  integer, parameter :: MPAS_LOG_OUT = 1, MPAS_LOG_WARN = 2, MPAS_LOG_ERR = 3, MPAS_LOG_CRIT = 4
  integer, save :: stub_log_errors = 0
  logical, save :: stub_log_capture = .false.
  integer, save :: stub_log_count = 0
  character(len=StrKIND), dimension(:), allocatable, save :: stub_log_lines

contains

  recursive subroutine mpas_log_write(message, messageType, masterOnly, flushNow, &
       intArgs, int8Args, realArgs, logicArgs, err)
    character(len=*), intent(in) :: message
    integer, intent(in), optional :: messageType
    logical, intent(in), optional :: masterOnly, flushNow
    integer, dimension(:), intent(in), optional :: intArgs
    integer(kind=I8KIND), dimension(:), intent(in), optional :: int8Args
    real(kind=RKIND), dimension(:), intent(in), optional :: realArgs
    logical, dimension(:), intent(in), optional :: logicArgs
    integer, intent(out), optional :: err
    character(len=StrKIND), dimension(:), allocatable :: grown
    if (present(err)) err = 0
    if (present(messageType)) then
      if (messageType >= MPAS_LOG_ERR) then
        stub_log_errors = stub_log_errors + 1
        print *, 'oracle log: ', trim(message)
      end if
    end if
    if (stub_log_capture) then
      if (present(messageType)) then
        if (messageType /= MPAS_LOG_OUT) return
      end if
      if (.not. allocated(stub_log_lines)) allocate(stub_log_lines(256))
      if (stub_log_count == size(stub_log_lines)) then
        allocate(grown(2 * size(stub_log_lines)))
        grown(1:stub_log_count) = stub_log_lines
        call move_alloc(grown, stub_log_lines)
      end if
      stub_log_count = stub_log_count + 1
      stub_log_lines(stub_log_count) = message
    end if
  end subroutine mpas_log_write

end module mpas_log

module mpas_pool_routines
  use mpas_kind_types
  implicit none
  private
  public :: MPAS_pool_type, mpas_pool_get_subpool, mpas_pool_get_array, &
            mpas_pool_get_dimension, mpas_pool_get_config, &
            stub_pool_put_subpool, stub_pool_put_array, &
            stub_pool_put_dimension, stub_pool_put_config, &
            mpas_pool_add_config, mpas_pool_print_summary, MPAS_POOL_CONFIG, &
            field1DInteger, field1DReal, field2DReal, field3DReal, mpas_pool_get_field, &
            mpas_allocate_scratch_field, mpas_deallocate_scratch_field, &
            mpas_pool_initialize_time_levels, mpas_pool_shift_time_levels, &
            mpas_pool_iterator_type, mpas_pool_begin_iteration, mpas_pool_get_next_member, &
            stub_pool_has, stub_pool_get_logical, mpas_pool_create_pool, mpas_pool_get_package

  integer, parameter :: maxEntries = 192
  integer, parameter :: MPAS_POOL_CONFIG = 1004

  ! the field types the land-ice modules declare; only scratch-field and
  ! halo routines reach them.  Scratch fields resolve to arrays the driver
  ! allocates and puts in the pool under the field's name (run_advection.F90,
  ! run_thermal_solver.F90, run_time_integration_fe.F90)
  type field1DInteger
    integer, dimension(:), pointer :: array => null()
    type(field1DInteger), pointer :: next => null()
  end type field1DInteger

  type field1DReal
    real(kind=RKIND), dimension(:), pointer :: array => null()
    type(field1DReal), pointer :: next => null()
  end type field1DReal

  type field2DReal
    real(kind=RKIND), dimension(:,:), pointer :: array => null()
    type(field2DReal), pointer :: next => null()
  end type field2DReal

  type field3DReal
    real(kind=RKIND), dimension(:,:,:), pointer :: array => null()
    type(field3DReal), pointer :: next => null()
  end type field3DReal

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
  end type pool_entry

  type MPAS_pool_type
    type(pool_entry), dimension(maxEntries) :: e
    integer :: n = 0
    integer :: cursor = 0   ! mpas_pool_get_next_member's place: entries in the order put
  end type MPAS_pool_type

  ! the iterator the analysis-member driver reads (memberName only)
  type mpas_pool_iterator_type
    character(len=StrKIND) :: memberName = ''
    integer :: memberType = 0
  end type mpas_pool_iterator_type

  interface mpas_pool_get_field
    module procedure get_field_i1, get_field_r1, get_field_r2, get_field_r3
  end interface mpas_pool_get_field

  interface mpas_allocate_scratch_field
    module procedure scratch_i1, scratch_r1, scratch_r2, scratch_r3
  end interface mpas_allocate_scratch_field

  interface mpas_deallocate_scratch_field
    module procedure scratch_i1, scratch_r1, scratch_r2, scratch_r3
  end interface mpas_deallocate_scratch_field

  interface mpas_pool_add_config
    module procedure add_config_l, add_config_r, add_config_i
  end interface mpas_pool_add_config

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

  interface stub_pool_put_array
    module procedure put_r0, put_r1, put_r2, put_r3, put_i0, put_i1, put_i2, put_c0
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

  integer function level(pool, name, timeLevel)
    ! a field's entry at timeLevel: time level 2 is the entry the driver put
    ! as NAME@2 (run_forward.F90); a pool without one has a single level
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    integer, intent(in), optional :: timeLevel
    integer :: i
    level = 0
    if (present(timeLevel)) then
      if (timeLevel == 2) then
        do i = 1, pool % n
          if (trim(pool % e(i) % name) == trim(name) // '@2') level = i
        end do
      end if
    end if
    if (level == 0) level = find(pool, name)
  end function level

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

  subroutine get_field_i1(pool, name, field, timeLevel)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    type(field1DInteger), pointer :: field
    integer, intent(in), optional :: timeLevel
    allocate(field)
    field % array => pool % e(find(pool, name)) % i1
  end subroutine get_field_i1

  subroutine scratch_i1(field, single_block_in)
    type(field1DInteger), pointer :: field
    logical, intent(in), optional :: single_block_in
    if (.not. associated(field % array)) then
      print *, 'oracle stub: a scratch field without a driver array'
      stop 6
    end if
  end subroutine scratch_i1

  subroutine get_field_r1(pool, name, field, timeLevel)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    type(field1DReal), pointer :: field
    integer, intent(in), optional :: timeLevel
    allocate(field)
    field % array => pool % e(find(pool, name)) % r1
  end subroutine get_field_r1

  subroutine scratch_r1(field, single_block_in)
    type(field1DReal), pointer :: field
    logical, intent(in), optional :: single_block_in
    if (.not. associated(field % array)) then
      print *, 'oracle stub: a scratch field without a driver array'
      stop 6
    end if
  end subroutine scratch_r1

  subroutine get_field_r2(pool, name, field, timeLevel)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    type(field2DReal), pointer :: field
    integer, intent(in), optional :: timeLevel
    ! a field wrapping the array the driver put under this name
    allocate(field)
    field % array => pool % e(find(pool, name)) % r2
  end subroutine get_field_r2

  subroutine scratch_r2(field, single_block_in)
    ! allocating and freeing a scratch field: the driver owns its array
    type(field2DReal), pointer :: field
    logical, intent(in), optional :: single_block_in
    if (.not. associated(field % array)) then
      print *, 'oracle stub: a scratch field without a driver array'
      stop 6
    end if
  end subroutine scratch_r2

  subroutine get_field_r3(pool, name, field, timeLevel)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    type(field3DReal), pointer :: field
    integer, intent(in), optional :: timeLevel
    allocate(field)
    field % array => pool % e(find(pool, name)) % r3
  end subroutine get_field_r3

  subroutine scratch_r3(field, single_block_in)
    type(field3DReal), pointer :: field
    logical, intent(in), optional :: single_block_in
    if (.not. associated(field % array)) then
      print *, 'oracle stub: a scratch field without a driver array'
      stop 6
    end if
  end subroutine scratch_r3

  subroutine add_config_l(pool, name, value)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    logical, intent(in) :: value
    print *, 'oracle stub: mpas_pool_add_config is not part of this oracle: ', trim(name)
    stop 6
  end subroutine add_config_l

  subroutine add_config_r(pool, name, value)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    real(kind=RKIND), intent(in) :: value
    print *, 'oracle stub: mpas_pool_add_config is not part of this oracle: ', trim(name)
    stop 6
  end subroutine add_config_r

  subroutine add_config_i(pool, name, value)
    ! li_analysis_setup_packages builds its member list with it
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    integer, intent(in) :: value
    integer :: k
    k = slot(pool, name)
    allocate(pool % e(k) % i0)
    pool % e(k) % i0 = value
  end subroutine add_config_i

  subroutine mpas_pool_get_package(pool, name, package)
    ! a package the driver put (run_forward.F90: the analysis members'); any
    ! other is not part of these oracles.  A module procedure: its pointer
    ! argument needs the explicit interface
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    logical, pointer :: package
    call stub_pool_get_logical(pool, name, package)
    if (associated(package)) return
    print *, 'oracle stub: mpas_pool_get_package (', trim(name), ') is not part of this oracle'
    stop 6
  end subroutine mpas_pool_get_package

  subroutine mpas_pool_create_pool(pool)
    ! li_analysis_setup_packages's member list (an empty pool)
    type(MPAS_pool_type), pointer :: pool
    allocate(pool)
  end subroutine mpas_pool_create_pool

  subroutine mpas_pool_begin_iteration(pool)
    type(MPAS_pool_type), intent(inout) :: pool
    pool % cursor = 0
  end subroutine mpas_pool_begin_iteration

  logical function mpas_pool_get_next_member(pool, iterator)
    type(MPAS_pool_type), intent(inout) :: pool
    type(mpas_pool_iterator_type), intent(inout) :: iterator
    pool % cursor = pool % cursor + 1
    mpas_pool_get_next_member = pool % cursor <= pool % n
    if (mpas_pool_get_next_member) iterator % memberName = pool % e(pool % cursor) % name
  end function mpas_pool_get_next_member

  logical function stub_pool_has(pool, name)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    integer :: i
    stub_pool_has = .false.
    do i = 1, pool % n
      if (trim(pool % e(i) % name) == trim(name)) stub_pool_has = .true.
    end do
  end function stub_pool_has

  subroutine stub_pool_get_logical(pool, name, value)
    ! a logical entry, or null when the pool has none by that name
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    logical, pointer :: value
    integer :: i
    nullify(value)
    do i = 1, pool % n
      if (trim(pool % e(i) % name) == trim(name)) value => pool % e(i) % l0
    end do
  end subroutine stub_pool_get_logical

  subroutine mpas_pool_print_summary(pool, poolMemberType, recurseSubpools)
    type(MPAS_pool_type), intent(in) :: pool
    integer, intent(in) :: poolMemberType
    logical, intent(in), optional :: recurseSubpools
  end subroutine mpas_pool_print_summary

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
    a => pool % e(level(pool, name, timeLevel)) % r0
  end subroutine get_r0

  subroutine get_r1(pool, name, a, timeLevel)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    integer, intent(in), optional :: timeLevel
    real(kind=RKIND), dimension(:), pointer :: a
    a => pool % e(level(pool, name, timeLevel)) % r1
  end subroutine get_r1

  subroutine get_r2(pool, name, a, timeLevel)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    integer, intent(in), optional :: timeLevel
    real(kind=RKIND), dimension(:,:), pointer :: a
    a => pool % e(level(pool, name, timeLevel)) % r2
  end subroutine get_r2

  subroutine get_r3(pool, name, a, timeLevel)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    integer, intent(in), optional :: timeLevel
    real(kind=RKIND), dimension(:,:,:), pointer :: a
    a => pool % e(level(pool, name, timeLevel)) % r3
  end subroutine get_r3

  subroutine get_i0(pool, name, a, timeLevel)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    integer, intent(in), optional :: timeLevel
    integer, pointer :: a
    a => pool % e(level(pool, name, timeLevel)) % i0
  end subroutine get_i0

  subroutine get_i1(pool, name, a, timeLevel)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    integer, intent(in), optional :: timeLevel
    integer, dimension(:), pointer :: a
    a => pool % e(level(pool, name, timeLevel)) % i1
  end subroutine get_i1

  subroutine get_i2(pool, name, a, timeLevel)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    integer, intent(in), optional :: timeLevel
    integer, dimension(:,:), pointer :: a
    a => pool % e(level(pool, name, timeLevel)) % i2
  end subroutine get_i2

  subroutine get_c0(pool, name, a, timeLevel)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    integer, intent(in), optional :: timeLevel
    character(len=StrKIND), pointer :: a
    a => pool % e(level(pool, name, timeLevel)) % c0
  end subroutine get_c0

  subroutine mpas_pool_initialize_time_levels(pool)
    ! every level from level 1, as the model's: NAME@2 takes NAME's words
    type(MPAS_pool_type), intent(inout) :: pool
    integer :: i, j
    do i = 1, pool % n
      j = len_trim(pool % e(i) % name)
      if (j < 3) cycle
      if (pool % e(i) % name(j-1:j) /= '@2') cycle
      associate (two => pool % e(i), one => pool % e(find(pool, pool % e(i) % name(1:j-2))))
        if (associated(two % i1)) two % i1 = one % i1
        if (associated(two % r1)) two % r1 = one % r1
        if (associated(two % r2)) two % r2 = one % r2
      end associate
    end do
  end subroutine mpas_pool_initialize_time_levels

  subroutine mpas_pool_shift_time_levels(pool)
    ! two levels: NAME and NAME@2 trade arrays
    type(MPAS_pool_type), intent(inout) :: pool
    integer :: i, j, k
    integer, dimension(:), pointer :: i1
    real(kind=RKIND), dimension(:), pointer :: r1
    real(kind=RKIND), dimension(:,:), pointer :: r2
    do i = 1, pool % n
      j = len_trim(pool % e(i) % name)
      if (j < 3) cycle
      if (pool % e(i) % name(j-1:j) /= '@2') cycle
      k = find(pool, pool % e(i) % name(1:j-2))
      i1 => pool % e(i) % i1; pool % e(i) % i1 => pool % e(k) % i1; pool % e(k) % i1 => i1
      r1 => pool % e(i) % r1; pool % e(i) % r1 => pool % e(k) % r1; pool % e(k) % r1 => r1
      r2 => pool % e(i) % r2; pool % e(i) % r2 => pool % e(k) % r2; pool % e(k) % r2 => r2
    end do
  end subroutine mpas_pool_shift_time_levels

  subroutine stub_pool_put_subpool(pool, name, sub)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    type(MPAS_pool_type), pointer :: sub
    integer :: k
    k = slot(pool, name)  ! before the pointer assignment: slot changes pool
    pool % e(k) % sub => sub
  end subroutine stub_pool_put_subpool

  subroutine put_dimension_0d(pool, name, dim)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    integer, pointer :: dim
    integer :: k
    k = slot(pool, name)  ! before the pointer assignment: slot changes pool
    pool % e(k) % i0 => dim
  end subroutine put_dimension_0d

  subroutine put_dimension_1d(pool, name, dim)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    integer, dimension(:), pointer :: dim
    integer :: k
    k = slot(pool, name)  ! before the pointer assignment: slot changes pool
    pool % e(k) % d1 => dim
  end subroutine put_dimension_1d

  subroutine put_config_l(pool, name, value)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    logical, pointer :: value
    integer :: k
    k = slot(pool, name)  ! before the pointer assignment: slot changes pool
    pool % e(k) % l0 => value
  end subroutine put_config_l

  subroutine put_config_c(pool, name, value)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    character(len=StrKIND), pointer :: value
    integer :: k
    k = slot(pool, name)  ! before the pointer assignment: slot changes pool
    pool % e(k) % c0 => value
  end subroutine put_config_c

  subroutine put_config_i(pool, name, value)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    integer, pointer :: value
    integer :: k
    k = slot(pool, name)  ! before the pointer assignment: slot changes pool
    pool % e(k) % i0 => value
  end subroutine put_config_i

  subroutine put_config_r(pool, name, value)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    real(kind=RKIND), pointer :: value
    integer :: k
    k = slot(pool, name)  ! before the pointer assignment: slot changes pool
    pool % e(k) % r0 => value
  end subroutine put_config_r

  subroutine put_c0(pool, name, a)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    character(len=StrKIND), pointer :: a
    integer :: k
    k = slot(pool, name)  ! before the pointer assignment: slot changes pool
    pool % e(k) % c0 => a
  end subroutine put_c0

  subroutine put_r0(pool, name, a)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    real(kind=RKIND), pointer :: a
    integer :: k
    k = slot(pool, name)  ! before the pointer assignment: slot changes pool
    pool % e(k) % r0 => a
  end subroutine put_r0

  subroutine put_r1(pool, name, a)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    real(kind=RKIND), dimension(:), pointer :: a
    integer :: k
    k = slot(pool, name)  ! before the pointer assignment: slot changes pool
    pool % e(k) % r1 => a
  end subroutine put_r1

  subroutine put_r2(pool, name, a)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    real(kind=RKIND), dimension(:,:), pointer :: a
    integer :: k
    k = slot(pool, name)  ! before the pointer assignment: slot changes pool
    pool % e(k) % r2 => a
  end subroutine put_r2

  subroutine put_r3(pool, name, a)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    real(kind=RKIND), dimension(:,:,:), pointer :: a
    integer :: k
    k = slot(pool, name)  ! before the pointer assignment: slot changes pool
    pool % e(k) % r3 => a
  end subroutine put_r3

  subroutine put_i0(pool, name, a)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    integer, pointer :: a
    integer :: k
    k = slot(pool, name)  ! before the pointer assignment: slot changes pool
    pool % e(k) % i0 => a
  end subroutine put_i0

  subroutine put_i1(pool, name, a)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    integer, dimension(:), pointer :: a
    integer :: k
    k = slot(pool, name)  ! before the pointer assignment: slot changes pool
    pool % e(k) % i1 => a
  end subroutine put_i1

  subroutine put_i2(pool, name, a)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    integer, dimension(:,:), pointer :: a
    integer :: k
    k = slot(pool, name)  ! before the pointer assignment: slot changes pool
    pool % e(k) % i2 => a
  end subroutine put_i2

end module mpas_pool_routines

#ifndef MALI_REAL_TIMEKEEPING
module mpas_timekeeping
  ! Times and intervals in whole seconds: all the land-ice time step reads of
  ! them is set_timestep's interval to the next forced time, which the driver
  ! sets.  mpas_get_timeInterval(dt=) converts the integer count of seconds as
  ! MPAS does for an interval without a fraction; the intervals the step makes
  ! from real time steps only feed log strings, so they hold nothing.
  use mpas_kind_types
  implicit none
  private
  public :: MPAS_Time_type, MPAS_TimeInterval_type, MPAS_Clock_type, MPAS_NOW, operator(-), &
            mpas_get_clock_time, mpas_alarm_get_next_ring_time, mpas_set_timeInterval, mpas_get_timeInterval

  integer, parameter :: MPAS_NOW = 0

  type MPAS_Time_type
    integer(kind=I8KIND) :: seconds = 0
  end type MPAS_Time_type

  type MPAS_TimeInterval_type
    integer(kind=I8KIND) :: seconds = 0
  end type MPAS_TimeInterval_type

  type MPAS_Clock_type
    type(MPAS_Time_type) :: now
    type(MPAS_Time_type) :: nextForceTime   ! 'adaptiveTimestepForceInterval''s next ring
  end type MPAS_Clock_type

  interface operator(-)
    module procedure time_minus
  end interface operator(-)

contains

  type(MPAS_TimeInterval_type) function time_minus(t1, t2)
    type(MPAS_Time_type), intent(in) :: t1, t2
    time_minus % seconds = t1 % seconds - t2 % seconds
  end function time_minus

  type(MPAS_Time_type) function mpas_get_clock_time(clock, whichTime, ierr)
    type(MPAS_Clock_type), intent(in) :: clock
    integer, intent(in) :: whichTime
    integer, intent(out), optional :: ierr
    if (present(ierr)) ierr = 0
    mpas_get_clock_time = clock % now
  end function mpas_get_clock_time

  type(MPAS_Time_type) function mpas_alarm_get_next_ring_time(clock, alarmID)
    type(MPAS_Clock_type), intent(in) :: clock
    character(len=*), intent(in) :: alarmID
    if (trim(alarmID) /= 'adaptiveTimestepForceInterval') then
      print *, 'oracle stub: no alarm named ', trim(alarmID)
      stop 4
    end if
    mpas_alarm_get_next_ring_time = clock % nextForceTime
  end function mpas_alarm_get_next_ring_time

  subroutine mpas_set_timeInterval(interval, dt, ierr)
    type(MPAS_TimeInterval_type), intent(inout) :: interval
    real(kind=RKIND), intent(in), optional :: dt
    integer, intent(out), optional :: ierr
    if (present(ierr)) ierr = 0
    interval % seconds = 0
  end subroutine mpas_set_timeInterval

  subroutine mpas_get_timeInterval(interval, dt, timeString, ierr)
    type(MPAS_TimeInterval_type), intent(in) :: interval
    real(kind=RKIND), intent(out), optional :: dt
    character(len=*), intent(out), optional :: timeString
    integer, intent(out), optional :: ierr
    if (present(ierr)) ierr = 0
    if (present(dt)) dt = real(interval % seconds, RKIND)
    if (present(timeString)) timeString = 'oracle'
  end subroutine mpas_get_timeInterval

end module mpas_timekeeping
#endif

module mpas_derived_types
  use mpas_kind_types
  use mpas_log
  use mpas_pool_routines
#ifdef MALI_REAL_TIMEKEEPING
  ! run_forward.F90's build: v8.4.2's own mpas_timekeeping and the ESMF time
  ! library it wraps (build_forward.sh), with the types the framework's
  ! mpas_derived_types carries for them
  use ESMF
#else
  use mpas_timekeeping, only: MPAS_Clock_type
#endif
  implicit none
  public

#ifdef MALI_REAL_TIMEKEEPING
#include "mpas_timekeeping_types.inc"

  ! the streams the run declares (run_forward.F90: the analysis members'),
  ! by name with their reference time; stubs_forward.F90's manager reads them
  type MPAS_streamManager_type
    integer :: n = 0, cursor = 0
    character(len=StrKIND), dimension(8) :: names = '', refTimes = ''
  end type MPAS_streamManager_type

  ! the members of v8.4.2's stream list (mpas_stream_list_types.inc) that
  ! mpas_timekeeping's mpas_minimum_alarm_interval reads; nothing calls it
  type MPAS_stream_list_type
    character(len=StrKIND) :: name
    type(MPAS_stream_list_type), pointer :: next => null()
  end type MPAS_stream_list_type

  type decomp_list
    integer :: unused = 0
  end type decomp_list
#endif

  type dm_info
    integer :: nprocs = 1
    integer :: my_proc_id = 0
  end type dm_info

  type block_type
    integer :: localBlockID = 0
    type(MPAS_pool_type), pointer :: structs => null()
    type(MPAS_pool_type), pointer :: dimensions => null()
    type(MPAS_pool_type), pointer :: configs => null()
    type(block_type), pointer :: next => null()
  end type block_type

  type domain_type
    type(block_type), pointer :: blocklist => null()
    type(MPAS_pool_type), pointer :: configs => null()
    type(MPAS_pool_type), pointer :: packages => null()
    type(dm_info), pointer :: dminfo => null()
    type(MPAS_Clock_type), pointer :: clock => null()
#ifdef MALI_REAL_TIMEKEEPING
    type(MPAS_streamManager_type), pointer :: streamManager => null()
    type(decomp_list), pointer :: decompositions => null()
#endif
  end type domain_type

end module mpas_derived_types

module mpas_dmpar
  ! one rank, one block: every reduction is the identity and there is no halo.
  ! The reductions li_compute_statistics makes (sums, min/maxloc, broadcasts)
  ! also record each word they return, in call order: stub_trace_r (reals)
  ! and stub_trace_i (integers), which run_forward.F90 writes out
  use mpas_kind_types
  use mpas_derived_types
  implicit none
  private
  public :: mpas_dmpar_min_real, mpas_dmpar_max_real, mpas_dmpar_max_int, mpas_dmpar_field_halo_exch, &
            mpas_dmpar_sum_real, mpas_dmpar_sum_int, mpas_dmpar_minloc_real, mpas_dmpar_maxloc_real, &
            mpas_dmpar_bcast_real, mpas_dmpar_bcast_int, mpas_dmpar_sum_real_array, IO_NODE, &
            stub_trace_r, stub_trace_i, stub_trace_nr, stub_trace_ni, stub_trace_paused

  integer, parameter :: IO_NODE = 0
  real(kind=RKIND), dimension(:), allocatable, save :: stub_trace_r
  integer, dimension(:), allocatable, save :: stub_trace_i
  integer, save :: stub_trace_nr = 0, stub_trace_ni = 0
  ! set while an analysis member computes (run_forward.F90's timer hook): its
  ! reductions are recorded as the pool words it leaves, not here
  logical, save :: stub_trace_paused = .false.

contains

  subroutine trace_r(value)
    real(kind=RKIND), intent(in) :: value
    real(kind=RKIND), dimension(:), allocatable :: grown
    if (stub_trace_paused) return
    if (.not. allocated(stub_trace_r)) allocate(stub_trace_r(256))
    if (stub_trace_nr == size(stub_trace_r)) then
      allocate(grown(2 * size(stub_trace_r)))
      grown(1:stub_trace_nr) = stub_trace_r
      call move_alloc(grown, stub_trace_r)
    end if
    stub_trace_nr = stub_trace_nr + 1
    stub_trace_r(stub_trace_nr) = value
  end subroutine trace_r

  subroutine trace_i(value)
    integer, intent(in) :: value
    integer, dimension(:), allocatable :: grown
    if (stub_trace_paused) return
    if (.not. allocated(stub_trace_i)) allocate(stub_trace_i(256))
    if (stub_trace_ni == size(stub_trace_i)) then
      allocate(grown(2 * size(stub_trace_i)))
      grown(1:stub_trace_ni) = stub_trace_i
      call move_alloc(grown, stub_trace_i)
    end if
    stub_trace_ni = stub_trace_ni + 1
    stub_trace_i(stub_trace_ni) = value
  end subroutine trace_i

  subroutine mpas_dmpar_sum_real(dminfo, r, rout)
    type(dm_info), intent(in) :: dminfo
    real(kind=RKIND), intent(in) :: r
    real(kind=RKIND), intent(out) :: rout
    rout = r
    call trace_r(rout)
  end subroutine mpas_dmpar_sum_real

  subroutine mpas_dmpar_sum_int(dminfo, i, iout)
    type(dm_info), intent(in) :: dminfo
    integer, intent(in) :: i
    integer, intent(out) :: iout
    iout = i
    call trace_i(iout)
  end subroutine mpas_dmpar_sum_int

  subroutine mpas_dmpar_minloc_real(dminfo, r, rmin, procout)
    type(dm_info), intent(in) :: dminfo
    real(kind=RKIND), intent(in) :: r
    real(kind=RKIND), intent(out) :: rmin
    integer, intent(out) :: procout
    rmin = r
    procout = IO_NODE
    call trace_r(rmin)
  end subroutine mpas_dmpar_minloc_real

  subroutine mpas_dmpar_maxloc_real(dminfo, r, rmax, procout)
    type(dm_info), intent(in) :: dminfo
    real(kind=RKIND), intent(in) :: r
    real(kind=RKIND), intent(out) :: rmax
    integer, intent(out) :: procout
    rmax = r
    procout = IO_NODE
    call trace_r(rmax)
  end subroutine mpas_dmpar_maxloc_real

  subroutine mpas_dmpar_bcast_real(dminfo, r, proc)
    type(dm_info), intent(in) :: dminfo
    real(kind=RKIND), intent(inout) :: r
    integer, intent(in), optional :: proc
    call trace_r(r)
  end subroutine mpas_dmpar_bcast_real

  subroutine mpas_dmpar_bcast_int(dminfo, i, proc)
    type(dm_info), intent(in) :: dminfo
    integer, intent(inout) :: i
    integer, intent(in), optional :: proc
    call trace_i(i)
  end subroutine mpas_dmpar_bcast_int

  subroutine mpas_dmpar_sum_real_array(dminfo, nElements, inArray, outArray)
    type(dm_info), intent(in) :: dminfo
    integer, intent(in) :: nElements
    real(kind=RKIND), dimension(nElements), intent(in) :: inArray
    real(kind=RKIND), dimension(nElements), intent(out) :: outArray
    integer :: i
    outArray = inArray
    do i = 1, nElements
      call trace_r(outArray(i))
    end do
  end subroutine mpas_dmpar_sum_real_array

  subroutine mpas_dmpar_min_real(dminfo, inValue, outValue)
    type(dm_info), intent(in) :: dminfo
    real(kind=RKIND), intent(in) :: inValue
    real(kind=RKIND), intent(out) :: outValue
    outValue = inValue
  end subroutine mpas_dmpar_min_real

  subroutine mpas_dmpar_max_real(dminfo, inValue, outValue)
    type(dm_info), intent(in) :: dminfo
    real(kind=RKIND), intent(in) :: inValue
    real(kind=RKIND), intent(out) :: outValue
    outValue = inValue
  end subroutine mpas_dmpar_max_real

  subroutine mpas_dmpar_max_int(dminfo, inValue, outValue)
    type(dm_info), intent(in) :: dminfo
    integer, intent(in) :: inValue
    integer, intent(out) :: outValue
    outValue = inValue
  end subroutine mpas_dmpar_max_int

  subroutine mpas_dmpar_field_halo_exch(domain, fieldName, timeLevel, haloLayers)
    type(domain_type), intent(inout) :: domain
    character(len=*), intent(in) :: fieldName
    integer, intent(in), optional :: timeLevel
    integer, dimension(:), intent(in), optional :: haloLayers
  end subroutine mpas_dmpar_field_halo_exch

end module mpas_dmpar

module mpas_abort
  implicit none
  private
  public :: mpas_dmpar_global_abort

contains

  subroutine mpas_dmpar_global_abort(mesg, deferredAbort)
    character(len=*), intent(in) :: mesg
    logical, intent(in), optional :: deferredAbort
    print *, 'oracle abort: ', trim(mesg)
    stop 7
  end subroutine mpas_dmpar_global_abort

end module mpas_abort

module mpas_threading
  implicit none
  private
  public :: mpas_threading_barrier, mpas_threading_get_thread_num

contains

  subroutine mpas_threading_barrier()
    ! one thread
  end subroutine mpas_threading_barrier

  integer function mpas_threading_get_thread_num()
    mpas_threading_get_thread_num = 0
  end function mpas_threading_get_thread_num

end module mpas_threading

module mpas_timer
  ! no timing; a driver may hook the starts and stops (run_forward.F90 records
  ! what each analysis member leaves when its 'compute_' timer stops)
  implicit none
  private
  public :: mpas_timer_start, mpas_timer_stop, stub_timer_hook

  abstract interface
    subroutine timer_hook(timer_name, starting)
      character(len=*), intent(in) :: timer_name
      logical, intent(in) :: starting
    end subroutine timer_hook
  end interface
  procedure(timer_hook), pointer, save :: stub_timer_hook => null()

contains

  subroutine mpas_timer_start(timer_name, clear_timer_in)
    character(len=*), intent(in) :: timer_name
    logical, optional, intent(in) :: clear_timer_in
    if (associated(stub_timer_hook)) call stub_timer_hook(timer_name, .true.)
  end subroutine mpas_timer_start

  subroutine mpas_timer_stop(timer_name)
    character(len=*), intent(in) :: timer_name
    if (associated(stub_timer_hook)) call stub_timer_hook(timer_name, .false.)
  end subroutine mpas_timer_stop

end module mpas_timer


! mpas_geometry_utils' Wachspress unit tests call this as an external
! procedure; the oracle never runs those tests
subroutine mpas_pool_destroy_pool(pool)
  use mpas_pool_routines
  type(MPAS_pool_type), pointer :: pool
  print *, 'oracle stub: mpas_pool_destroy_pool is not part of this oracle'
  stop 6
end subroutine mpas_pool_destroy_pool

module li_velocity_external
  ! The higher-order velocity solvers (Albany FO, L1L2, Stokes) are an
  ! external C++ dycore that the port refuses by name; li_velocity's case
  ! for them is never reached by these oracles.
  use mpas_derived_types
  implicit none
  private
  public :: li_velocity_external_init, li_velocity_external_block_init, li_velocity_external_solve, &
            li_velocity_external_finalize, li_velocity_external_write_albany_mesh

contains

  subroutine reached(name)
    character(len=*), intent(in) :: name
    print *, 'oracle stub: ', name, ' (the external velocity solver) is not part of this oracle'
    stop 6
  end subroutine reached

  subroutine li_velocity_external_init(domain, err)
    type(domain_type), intent(inout) :: domain
    integer, intent(out) :: err
    err = 0
    call reached('li_velocity_external_init')
  end subroutine li_velocity_external_init

  subroutine li_velocity_external_block_init(block, err)
    type(block_type), intent(inout) :: block
    integer, intent(out) :: err
    err = 0
    call reached('li_velocity_external_block_init')
  end subroutine li_velocity_external_block_init

  subroutine li_velocity_external_solve(meshPool, geometryPool, thermalPool, scratchPool, velocityPool, err)
    type(MPAS_pool_type), intent(in) :: meshPool
    type(MPAS_pool_type), intent(inout) :: geometryPool, thermalPool, scratchPool, velocityPool
    integer, intent(out) :: err
    err = 0
    call reached('li_velocity_external_solve')
  end subroutine li_velocity_external_solve

  subroutine li_velocity_external_finalize(err)
    integer, intent(out) :: err
    err = 0
    call reached('li_velocity_external_finalize')
  end subroutine li_velocity_external_finalize

  subroutine li_velocity_external_write_albany_mesh(domain)
    ! li_core_init calls it on every start; v8.4.2's returns at once unless
    ! config_write_albany_ascii_mesh (refused by name), as this one does
    use mpas_pool_routines, only: mpas_pool_get_config
    type(domain_type), intent(inout) :: domain
    logical, pointer :: config_write_albany_ascii_mesh
    call mpas_pool_get_config(domain % configs, 'config_write_albany_ascii_mesh', config_write_albany_ascii_mesh)
    if (.not. config_write_albany_ascii_mesh) return
    call reached('li_velocity_external_write_albany_mesh')
  end subroutine li_velocity_external_write_albany_mesh

end module li_velocity_external

! li_subglacial_hydro, li_velocity and the analysis members call these
! without an interface
subroutine mpas_dmpar_max_real_array(dminfo, nElements, inArray, outArray)
  use mpas_derived_types
  type(dm_info), intent(in) :: dminfo
  integer, intent(in) :: nElements
  real(kind=RKIND), dimension(nElements), intent(in) :: inArray
  real(kind=RKIND), dimension(nElements), intent(out) :: outArray
  ! one rank: the reduction is the identity
  outArray = inArray
end subroutine mpas_dmpar_max_real_array

subroutine mpas_dmpar_min_real_array(dminfo, nElements, inArray, outArray)
  use mpas_derived_types
  type(dm_info), intent(in) :: dminfo
  integer, intent(in) :: nElements
  real(kind=RKIND), dimension(nElements), intent(in) :: inArray
  real(kind=RKIND), dimension(nElements), intent(out) :: outArray
  ! one rank: the reduction is the identity
  outArray = inArray
end subroutine mpas_dmpar_min_real_array
