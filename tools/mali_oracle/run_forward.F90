! Drive MALI v8.4.2's forward mode on one block for a whole run and write
! every field word back: the reference the GPU port's forward mode
! (woof.hex.mpas_ice.forward_v842) is held against.
!
!   run_forward INPUT.bin OUTPUT.bin
!
! INPUT.bin is written by woof.verify.mali_forward_oracle.  It is
! self-describing: every config (name, type, value), then every field (pool,
! name, type, rank, shape, words), as run_time_integration_fe.F90's, plus
! text fields (xtime, simulationStartTime) and second time levels (a field
! put as NAME@2).  The driver puts the configs in liConfigs (and
! 'on_a_sphere' and 'sphere_radius' in the mesh pool, where the mesh
! routines look for them), the fields in the pools they name, and then
! makes the calls MPAS's driver makes for the land-ice core, every one of
! them v8.4.2's own, compiled verbatim (mpas_li_core.F,
! mpas_li_time_integration.F, mpas_timekeeping.F and the ESMF time library):
!
!   mpas_timekeeping_init(config_calendar_type)
!   li_simulation_clock_init(clock, configs, err)
!   li_core_init(domain, startTimeStamp)
!   li_core_run(domain)                  every step to the clock's stop time
!
! The input stream's fields stand for the 'input' stream li_core_init
! reads; no stream is read or written during the run (stubs_forward.F90).
!
! The analysis members are v8.4.2's own too (mpas_li_analysis_driver.F and
! its globalStats and regionalStats): the driver makes MPAS's package-setup
! call, li_analysis_setup_packages(configs, packages), before li_core_init,
! with each member's configs at their Registry defaults when the input has
! none, so a run without them is the run it was.  A config input as
! 'stream:NAME' (text) declares the stream NAME with that reference time,
! for the members that name it (stubs_forward.F90's stream manager).
!
! OUTPUT.bin holds the error flags (clock init, core init, core run), the
! count of error-level log lines, the clock's final time as text (StrKIND
! characters), then every field's words in the input's order, then the
! record li_compute_statistics leaves (stubs.F90): the counts of its real
! and integer reduction words and of the log lines, the real words, the
! integer words, and every log message (StrKIND characters each) but the
! warnings and errors; then the analysis members' record: the number of
! computes and of their words, each compute's member (1 globalStats, 2
! regionalStats), the xtime it computed at (StrKIND characters each), and
! the words it left: every field of its pool (NAME'AM') in the input's
! order, taken when its 'compute_' timer stops.

program run_forward
  use mpas_kind_types
  use mpas_derived_types
  use mpas_pool_routines
  use mpas_timekeeping
  use mpas_log, only: stub_log_errors, stub_log_capture, stub_log_lines, stub_log_count
  use mpas_dmpar, only: stub_trace_r, stub_trace_i, stub_trace_nr, stub_trace_ni, stub_trace_paused
  use mpas_timer, only: stub_timer_hook
  use li_core, only: li_simulation_clock_init, li_core_init, li_core_run
  use li_analysis_driver, only: li_analysis_setup_packages
  implicit none

  integer, parameter :: maxFields = 512
  type field_ref
    character(len=16) :: pool
    character(len=64) :: name
    integer :: kind = 0, rank = 0
    real(kind=RKIND), pointer :: r0 => null()
    real(kind=RKIND), dimension(:), pointer :: r1 => null()
    real(kind=RKIND), dimension(:,:), pointer :: r2 => null()
    real(kind=RKIND), dimension(:,:,:), pointer :: r3 => null()
    integer, pointer :: i0 => null()
    integer, dimension(:), pointer :: i1 => null()
    integer, dimension(:,:), pointer :: i2 => null()
    character(len=StrKIND), pointer :: c0 => null()
  end type field_ref
  type pool_ref
    type(MPAS_pool_type), pointer :: p => null()
  end type pool_ref

  type(domain_type) :: domain
  type(block_type), pointer :: block
  type(MPAS_pool_type), pointer :: structs, configs
  type(pool_ref) :: pools(9)
  character(len=16), parameter :: poolNames(9) = [character(len=16) :: 'mesh', 'geometry', 'thermal', &
       'velocity', 'scratch', 'hydro', 'globalStatsAM', 'regionalStatsAM', 'regions']
  character(len=16), parameter :: memberNames(2) = [character(len=16) :: 'globalStats', 'regionalStats']
  ! the analysis members' record (see above)
  integer :: nComputes = 0, nWords = 0, member = 0, errPackages
  integer, allocatable :: computeMembers(:)
  character(len=StrKIND), allocatable :: computeTimes(:)
  real(kind=RKIND), allocatable :: computeWords(:)
  character(len=StrKIND), pointer :: xtime => null()
  type(field_ref), allocatable :: fields(:)
  integer :: nConfigs, nFields, i, u, ctype, dims(3), errClock, errInit, errRun, p
  character(len=StrKIND) :: startTimeStamp, calendar, endTimeStamp
  type(MPAS_Time_type) :: endTime
  character(len=64) :: cname
  character(len=StrKIND) :: cvalue
  real(kind=RKIND) :: rvalue
  integer :: ivalue
  real(kind=RKIND), pointer :: rc
  integer, pointer :: ic
  logical, pointer :: lc
  character(len=StrKIND), pointer :: cc
  character(len=1024) :: inPath, outPath

  call get_command_argument(1, inPath)
  call get_command_argument(2, outPath)

  allocate(structs, configs)
  do p = 1, size(poolNames)
     allocate(pools(p) % p)
  end do
  allocate(domain % dminfo, domain % clock, domain % packages, domain % streamManager)
  calendar = 'gregorian_noleap'

  open(newunit=u, file=trim(inPath), access='stream', form='unformatted', status='old')
  read(u) nConfigs, nFields

  do i = 1, nConfigs
     read(u) cname, ctype
     select case (ctype)
     case (1)
        read(u) rvalue
        allocate(rc)
        rc = rvalue
        if (trim(cname) == 'sphere_radius') then
           call stub_pool_put_config(pools(1) % p, trim(cname), rc)
        else
           call stub_pool_put_config(configs, trim(cname), rc)
        end if
     case (2)
        read(u) ivalue
        allocate(ic)
        ic = ivalue
        call stub_pool_put_config(configs, trim(cname), ic)
     case (3)
        read(u) ivalue
        allocate(lc)
        lc = ivalue /= 0
        if (trim(cname) == 'on_a_sphere') then
           call stub_pool_put_config(pools(1) % p, trim(cname), lc)
        else
           call stub_pool_put_config(configs, trim(cname), lc)
        end if
     case (4)
        read(u) cvalue
        if (cname(1:7) == 'stream:') then
           if (domain % streamManager % n == size(domain % streamManager % names)) stop 13
           domain % streamManager % n = domain % streamManager % n + 1
           domain % streamManager % names(domain % streamManager % n) = cname(8:)
           domain % streamManager % refTimes(domain % streamManager % n) = cvalue
           cycle
        end if
        allocate(cc)
        cc = cvalue
        call stub_pool_put_config(configs, trim(cname), cc)
        if (trim(cname) == 'config_calendar_type') calendar = cvalue
     case default
        stop 8
     end select
  end do

  if (nFields > maxFields) stop 9
  allocate(fields(nFields))
  do i = 1, nFields
     read(u) fields(i) % pool, fields(i) % name, fields(i) % kind, fields(i) % rank, dims
     p = findloc(poolNames, fields(i) % pool, 1)
     if (p == 0) stop 10
     associate (f => fields(i), pool => pools(p) % p)
       select case (f % kind * 10 + f % rank)
       case (10)
          allocate(f % r0)
          read(u) f % r0
          call stub_pool_put_array(pool, trim(f % name), f % r0)
       case (11)
          allocate(f % r1(dims(1)))
          read(u) f % r1
          call stub_pool_put_array(pool, trim(f % name), f % r1)
       case (12)
          allocate(f % r2(dims(1), dims(2)))
          read(u) f % r2
          call stub_pool_put_array(pool, trim(f % name), f % r2)
       case (13)
          allocate(f % r3(dims(1), dims(2), dims(3)))
          read(u) f % r3
          call stub_pool_put_array(pool, trim(f % name), f % r3)
       case (20)
          allocate(f % i0)
          read(u) f % i0
          call stub_pool_put_array(pool, trim(f % name), f % i0)
       case (21)
          allocate(f % i1(dims(1)))
          read(u) f % i1
          call stub_pool_put_array(pool, trim(f % name), f % i1)
       case (22)
          allocate(f % i2(dims(1), dims(2)))
          read(u) f % i2
          call stub_pool_put_array(pool, trim(f % name), f % i2)
       case (30)
          ! a dimension
          allocate(f % i0)
          read(u) f % i0
          call stub_pool_put_dimension(pool, trim(f % name), f % i0)
       case (40)
          ! text (xtime, simulationStartTime)
          allocate(f % c0)
          read(u) f % c0
          call stub_pool_put_array(pool, trim(f % name), f % c0)
          if (trim(f % pool) == 'mesh' .and. trim(f % name) == 'xtime') xtime => f % c0
       case default
          stop 11
       end select
     end associate
  end do
  close(u)

  do p = 1, size(poolNames)
     call stub_pool_put_subpool(structs, trim(poolNames(p)), pools(p) % p)
  end do

  ! the analysis members' configs at their Registry defaults, and their
  ! packages (off until li_analysis_setup_packages turns one on)
  do p = 1, size(memberNames)
     cname = 'config_AM_' // trim(memberNames(p))
     if (.not. stub_pool_has(configs, trim(cname) // '_enable')) then
        allocate(lc)
        lc = .false.
        call stub_pool_put_config(configs, trim(cname) // '_enable', lc)
        allocate(cc)
        cc = 'output_interval'
        call stub_pool_put_config(configs, trim(cname) // '_compute_interval', cc)
        allocate(cc)
        cc = trim(memberNames(p)) // 'Output'
        call stub_pool_put_config(configs, trim(cname) // '_stream_name', cc)
        allocate(lc)
        lc = .true.
        call stub_pool_put_config(configs, trim(cname) // '_compute_on_startup', lc)
        allocate(lc)
        lc = .true.
        call stub_pool_put_config(configs, trim(cname) // '_write_on_startup', lc)
     end if
     allocate(lc)
     lc = .false.
     call stub_pool_put_config(domain % packages, trim(memberNames(p)) // 'AMPKGActive', lc)
  end do

  allocate(block)
  block % structs => structs
  block % dimensions => pools(1) % p
  block % configs => configs
  block % next => null()
  domain % blocklist => block
  domain % configs => configs

  stub_log_capture = .true.
  allocate(computeMembers(0), computeTimes(0), computeWords(0))
  stub_timer_hook => record_member
  call li_analysis_setup_packages(configs, domain % packages, errPackages)
  if (errPackages /= 0) stop 14
  call mpas_timekeeping_init(calendar)
  call li_simulation_clock_init(domain % clock, configs, errClock)
  errInit = li_core_init(domain, startTimeStamp)
  errRun = li_core_run(domain)
  endTime = mpas_get_clock_time(domain % clock, MPAS_NOW)
  call mpas_get_time(endTime, dateTimeString=endTimeStamp)

  open(newunit=u, file=trim(outPath), access='stream', form='unformatted', status='replace')
  write(u) errClock, errInit, errRun, stub_log_errors
  write(u) endTimeStamp
  do i = 1, nFields
     associate (f => fields(i))
       select case (f % kind * 10 + f % rank)
       case (10)
          write(u) f % r0
       case (11)
          write(u) f % r1
       case (12)
          write(u) f % r2
       case (13)
          write(u) f % r3
       case (20, 30)
          write(u) f % i0
       case (21)
          write(u) f % i1
       case (22)
          write(u) f % i2
       case (40)
          write(u) f % c0
       end select
     end associate
  end do
  write(u) stub_trace_nr, stub_trace_ni, stub_log_count
  if (stub_trace_nr > 0) write(u) stub_trace_r(1:stub_trace_nr)
  if (stub_trace_ni > 0) write(u) stub_trace_i(1:stub_trace_ni)
  do i = 1, stub_log_count
     write(u) stub_log_lines(i)
  end do
  write(u) nComputes, nWords
  if (nComputes > 0) write(u) computeMembers, computeTimes
  if (nWords > 0) write(u) computeWords
  close(u)

contains

  subroutine record_member(timer_name, starting)
    ! an analysis member's 'compute_' timer: its reductions are not traced,
    ! and when it stops, the words it left are recorded
    character(len=*), intent(in) :: timer_name
    logical, intent(in) :: starting
    integer :: m, k
    character(len=16) :: poolName
    m = 0
    do k = 1, size(memberNames)
       if (timer_name == 'compute_' // trim(memberNames(k))) m = k
    end do
    if (m == 0) return
    stub_trace_paused = starting
    if (starting) return
    if (.not. associated(xtime)) stop 15
    nComputes = nComputes + 1
    computeMembers = [computeMembers, m]
    computeTimes = [character(len=StrKIND) :: computeTimes, xtime]
    poolName = trim(memberNames(m)) // 'AM'
    do k = 1, nFields
       if (fields(k) % pool /= poolName) cycle
       select case (fields(k) % kind * 10 + fields(k) % rank)
       case (10)
          computeWords = [computeWords, fields(k) % r0]
          nWords = nWords + 1
       case (11)
          computeWords = [computeWords, fields(k) % r1]
          nWords = nWords + size(fields(k) % r1)
       case default
          stop 16
       end select
    end do
  end subroutine record_member

end program run_forward
