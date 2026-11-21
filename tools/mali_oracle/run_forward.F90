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
! OUTPUT.bin holds the error flags (clock init, core init, core run), the
! count of error-level log lines, the clock's final time as text (StrKIND
! characters), then every field's words in the input's order.

program run_forward
  use mpas_kind_types
  use mpas_derived_types
  use mpas_pool_routines
  use mpas_timekeeping
  use mpas_log, only: stub_log_errors
  use li_core, only: li_simulation_clock_init, li_core_init, li_core_run
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
  type(pool_ref) :: pools(6)
  character(len=16), parameter :: poolNames(6) = [character(len=16) :: 'mesh', 'geometry', 'thermal', &
       'velocity', 'scratch', 'hydro']
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
       case default
          stop 11
       end select
     end associate
  end do
  close(u)

  do p = 1, size(poolNames)
     call stub_pool_put_subpool(structs, trim(poolNames(p)), pools(p) % p)
  end do

  allocate(block)
  block % structs => structs
  block % configs => configs
  block % next => null()
  domain % blocklist => block
  domain % configs => configs

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
  close(u)

end program run_forward
