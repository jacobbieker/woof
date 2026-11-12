! Drive one MALI v8.4.2 forward-Euler time step on one block and write every
! field word back: the reference the GPU port's step is held against.
!
!   run_time_integration_fe INPUT.bin OUTPUT.bin
!
! INPUT.bin is written by woof.verify.mali_time_integration_fe_oracle.  It is
! self-describing: the clock, then every config (name, type, value), then
! every field (pool, name, type, rank, shape, words).  The driver puts the
! configs in liConfigs (and 'on_a_sphere' in the mesh pool, where
! mpas_reconstruct looks for it), the fields in the pools they name, and the
! pools in one block, and then makes the calls the model makes:
!
!   li_thermal_init(domain, err)    as on a restart (config_do_restart, the
!                                   air temperature and heat flux from file),
!                                   where it only copies config_ice_density
!                                   and config_ocean_density into the module
!                                   and builds its vertical grid terms
!   li_time_integrator_forwardeuler(domain, err)
!
! OUTPUT.bin holds the two error flags and the count of error-level log
! lines, then every field's words in the input's order.  Scratch fields the
! step allocates resolve to fields the driver put in the scratch pool, filled
! with the case's words: the model leaves them uninitialised, so no output
! may depend on them.

program run_time_integration_fe
  use mpas_kind_types
  use mpas_derived_types
  use mpas_pool_routines
  use mpas_timekeeping, only: MPAS_Clock_type
  use mpas_log, only: stub_log_errors
  use li_setup, only: liConfigs
  use li_thermal, only: li_thermal_init
  use li_time_integration_fe, only: li_time_integrator_forwardeuler
  implicit none

  integer, parameter :: maxFields = 256
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
  integer :: nConfigs, nFields, i, u, ctype, dims(3), errInit, errStep, p
  integer(kind=I8KIND) :: clockNow, clockNextForce
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
  allocate(domain % dminfo, domain % clock)

  open(newunit=u, file=trim(inPath), access='stream', form='unformatted', status='old')
  read(u) clockNow, clockNextForce, nConfigs, nFields
  domain % clock % now % seconds = clockNow
  domain % clock % nextForceTime % seconds = clockNextForce

  do i = 1, nConfigs
     read(u) cname, ctype
     select case (ctype)
     case (1)
        read(u) rvalue
        allocate(rc)
        rc = rvalue
        call stub_pool_put_config(configs, trim(cname), rc)
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
       case default
          stop 11
       end select
     end associate
  end do
  close(u)

  do p = 1, size(poolNames)
     call stub_pool_put_subpool(structs, trim(poolNames(p)), pools(p) % p)
  end do
  liConfigs => configs

  allocate(block)
  block % structs => structs
  block % configs => configs
  block % next => null()
  domain % blocklist => block
  domain % configs => configs

  call li_thermal_init(domain, errInit)
  call li_time_integrator_forwardeuler(domain, errStep)

  open(newunit=u, file=trim(outPath), access='stream', form='unformatted', status='replace')
  write(u) errInit, errStep, stub_log_errors
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
       end select
     end associate
  end do
  close(u)

end program run_time_integration_fe
