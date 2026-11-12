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
! errors and records that one was written; it never alters a value.

module mpas_log
  use mpas_kind_types
  implicit none
  private
  public :: mpas_log_write, MPAS_LOG_OUT, MPAS_LOG_WARN, MPAS_LOG_ERR, MPAS_LOG_CRIT, stub_log_errors

  integer, parameter :: MPAS_LOG_OUT = 1, MPAS_LOG_WARN = 2, MPAS_LOG_ERR = 3, MPAS_LOG_CRIT = 4
  integer, save :: stub_log_errors = 0

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
    if (present(err)) err = 0
    if (present(messageType)) then
      if (messageType >= MPAS_LOG_ERR) then
        stub_log_errors = stub_log_errors + 1
        print *, 'oracle log: ', trim(message)
      end if
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
            mpas_allocate_scratch_field, mpas_deallocate_scratch_field

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
  end type MPAS_pool_type

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
    module procedure add_config_l, add_config_r
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
    module procedure get_r0, get_r1, get_r2, get_r3, get_i0, get_i1, get_i2
  end interface mpas_pool_get_array

  interface stub_pool_put_array
    module procedure put_r0, put_r1, put_r2, put_r3, put_i0, put_i1, put_i2
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

  subroutine get_i0(pool, name, a, timeLevel)
    type(MPAS_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    integer, intent(in), optional :: timeLevel
    integer, pointer :: a
    a => pool % e(find(pool, name)) % i0
  end subroutine get_i0

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

  subroutine put_dimension_0d(pool, name, dim)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    integer, pointer :: dim
    pool % e(slot(pool, name)) % i0 => dim
  end subroutine put_dimension_0d

  subroutine put_dimension_1d(pool, name, dim)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    integer, dimension(:), pointer :: dim
    pool % e(slot(pool, name)) % d1 => dim
  end subroutine put_dimension_1d

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

  subroutine put_config_r(pool, name, value)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    real(kind=RKIND), pointer :: value
    pool % e(slot(pool, name)) % r0 => value
  end subroutine put_config_r

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

  subroutine put_i0(pool, name, a)
    type(MPAS_pool_type), intent(inout) :: pool
    character(len=*), intent(in) :: name
    integer, pointer :: a
    pool % e(slot(pool, name)) % i0 => a
  end subroutine put_i0

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

module mpas_derived_types
  use mpas_kind_types
  use mpas_log
  use mpas_pool_routines
  use mpas_timekeeping, only: MPAS_Clock_type
  implicit none
  public

  type dm_info
    integer :: nprocs = 1
    integer :: my_proc_id = 0
  end type dm_info

  type block_type
    type(MPAS_pool_type), pointer :: structs => null()
    type(MPAS_pool_type), pointer :: configs => null()
    type(block_type), pointer :: next => null()
  end type block_type

  type domain_type
    type(block_type), pointer :: blocklist => null()
    type(MPAS_pool_type), pointer :: configs => null()
    type(MPAS_pool_type), pointer :: packages => null()
    type(dm_info), pointer :: dminfo => null()
    type(MPAS_Clock_type), pointer :: clock => null()
  end type domain_type

end module mpas_derived_types

module mpas_dmpar
  ! one rank, one block: every reduction is the identity and there is no halo
  use mpas_kind_types
  use mpas_derived_types
  implicit none
  private
  public :: mpas_dmpar_min_real, mpas_dmpar_max_real, mpas_dmpar_max_int, mpas_dmpar_field_halo_exch

contains

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
  public :: mpas_threading_barrier

contains

  subroutine mpas_threading_barrier()
    ! one thread
  end subroutine mpas_threading_barrier

end module mpas_threading

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


! mpas_geometry_utils' Wachspress unit tests call these as external
! procedures; the oracle never runs those tests
subroutine mpas_pool_create_pool(pool)
  use mpas_pool_routines
  type(MPAS_pool_type), pointer :: pool
  print *, 'oracle stub: mpas_pool_create_pool is not part of this oracle'
  stop 6
end subroutine mpas_pool_create_pool

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
    type(domain_type), intent(inout) :: domain
    call reached('li_velocity_external_write_albany_mesh')
  end subroutine li_velocity_external_write_albany_mesh

end module li_velocity_external

! li_subglacial_hydro and li_velocity call these without an interface
subroutine mpas_dmpar_min_real_array(dminfo, nElements, inArray, outArray)
  use mpas_derived_types
  type(dm_info), intent(in) :: dminfo
  integer, intent(in) :: nElements
  real(kind=RKIND), dimension(nElements), intent(in) :: inArray
  real(kind=RKIND), dimension(nElements), intent(out) :: outArray
  ! one rank: the reduction is the identity
  outArray = inArray
end subroutine mpas_dmpar_min_real_array

subroutine mpas_pool_get_package(pool, name, package)
  use mpas_pool_routines
  type(MPAS_pool_type), intent(in) :: pool
  character(len=*), intent(in) :: name
  logical, pointer :: package
  print *, 'oracle stub: mpas_pool_get_package (', trim(name), ') is not part of this oracle'
  stop 6
end subroutine mpas_pool_get_package
