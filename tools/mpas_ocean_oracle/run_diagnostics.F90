! Drive MPAS-Ocean v8.4.2's ocn_diagnostic_solve on one block and write every
! field word back: the reference the GPU port is held against.
!
!   run_diagnostics INPUT.bin OUTPUT.bin
!
! INPUT.bin is written by woof.verify.mpas_ocean_diagnostics_oracle.  It is
! self-describing: dt, then every config (name, type, value), then every
! field (pool, name, kind, rank, shape, words).  Kinds: 1 real array, 2
! integer array, 3 a dimension, 4 a dimension array (nCellsArray, ...), 5 an
! absent field (a package that is off: an unassociated pointer, which the
! routine tests with associated()), 6 a scratch field (its shape only).  The
! driver puts the configs in ocnConfigs, the fields in the pools they name
! ('tracers' is the state pool's subpool) and then makes the calls the model
! makes:
!
!   ocn_constants_init(configs, packages)   rho_sw = config_density0
!   ocn_equation_of_state_init(err)
!   ocn_diagnostics_init(err)               ke_vertex_flag from config_include_KE_vertex
!   ocn_diagnostic_solve(dt, state, forcing, mesh, diagnostics, scratch, tracers, timeLevelIn=1)
!
! OUTPUT.bin holds every real and integer field's words in the input's order.
! Scratch fields are allocated by the routine and filled with huge() by the
! stub, so a word read before it is written would show.

program run_diagnostics
  use mpas_kind_types
  use mpas_derived_types
  use mpas_pool_routines
  use ocn_constants
  use ocn_equation_of_state
  use ocn_diagnostics
  implicit none

  integer, parameter :: maxFields = 256
  type field_ref
    character(len=16) :: pool
    character(len=64) :: name
    integer :: kind = 0, rank = 0
    real(kind=RKIND), dimension(:), pointer :: r1 => null()
    real(kind=RKIND), dimension(:,:), pointer :: r2 => null()
    real(kind=RKIND), dimension(:,:,:), pointer :: r3 => null()
    integer, dimension(:), pointer :: i1 => null()
    integer, dimension(:,:), pointer :: i2 => null()
  end type field_ref
  type pool_ref
    type(MPAS_pool_type), pointer :: p => null()
  end type pool_ref

  character(len=16), parameter :: poolNames(6) = [character(len=16) :: 'mesh', 'state', 'tracers', &
       'diagnostics', 'forcing', 'scratch']
  type(pool_ref) :: pools(6)
  type(MPAS_pool_type), pointer :: configs, packages
  type(field_ref), allocatable :: fields(:)
  integer :: nConfigs, nFields, i, u, ctype, dims(3), p, err
  character(len=64) :: cname
  character(len=StrKIND) :: cvalue
  real(kind=RKIND) :: rvalue, dt
  integer :: ivalue
  real(kind=RKIND), pointer :: rc
  integer, pointer :: ic
  logical, pointer :: lc
  character(len=StrKIND), pointer :: cc
  integer, pointer :: d0
  integer, dimension(:), pointer :: d1
  real(kind=RKIND), dimension(:), pointer :: null1
  real(kind=RKIND), dimension(:,:), pointer :: null2
  real(kind=RKIND), dimension(:,:,:), pointer :: null3
  integer, dimension(:), pointer :: nullI1
  type(field1DReal), pointer :: s1
  type(field2DReal), pointer :: s2
  character(len=1024) :: inPath, outPath

  call get_command_argument(1, inPath)
  call get_command_argument(2, outPath)
  allocate(configs, packages)
  do p = 1, size(poolNames)
     allocate(pools(p) % p)
  end do
  call stub_pool_put_subpool(pools(2) % p, 'tracers', pools(3) % p)
  nullify(null1, null2, null3, nullI1)

  open(newunit=u, file=trim(inPath), access='stream', form='unformatted', status='old')
  read(u) dt, nConfigs, nFields
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
        call stub_pool_put_config(configs, trim(cname), lc)
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
       case (21)
          allocate(f % i1(dims(1)))
          read(u) f % i1
          call stub_pool_put_array(pool, trim(f % name), f % i1)
       case (22)
          allocate(f % i2(dims(1), dims(2)))
          read(u) f % i2
          call stub_pool_put_array(pool, trim(f % name), f % i2)
       case (30)
          allocate(d0)
          read(u) d0
          call stub_pool_put_dimension(pool, trim(f % name), d0)
       case (41)
          allocate(d1(dims(1)))
          read(u) d1
          call stub_pool_put_dimension(pool, trim(f % name), d1)
       case (51)
          call stub_pool_put_array(pool, trim(f % name), null1)
       case (52)
          call stub_pool_put_array(pool, trim(f % name), null2)
       case (53)
          call stub_pool_put_array(pool, trim(f % name), null3)
       case (54)
          ! an absent integer field (rank 1)
          call stub_pool_put_array(pool, trim(f % name), nullI1)
       case (61)
          allocate(s1)
          s1 % dimSizes = dims(1:1)
          call stub_pool_put_field(pool, trim(f % name), s1)
       case (62)
          allocate(s2)
          s2 % dimSizes = dims(1:2)
          call stub_pool_put_field(pool, trim(f % name), s2)
       case default
          stop 11
       end select
     end associate
  end do
  read(u, iostat=err) ivalue
  if (err == 0) stop 12   ! trailing input words
  close(u)

  call ocn_constants_init(configs, packages)
  call ocn_equation_of_state_init(err)
  if (err /= 0) stop 13
  call ocn_diagnostics_init(err)
  if (err /= 0) stop 13
  call ocn_diagnostic_solve(dt, pools(2) % p, pools(5) % p, pools(1) % p, pools(4) % p, pools(6) % p, &
       pools(3) % p, timeLevelIn=1)

  open(newunit=u, file=trim(outPath), access='stream', form='unformatted', status='replace')
  do i = 1, nFields
     associate (f => fields(i))
       select case (f % kind * 10 + f % rank)
       case (11)
          write(u) f % r1
       case (12)
          write(u) f % r2
       case (13)
          write(u) f % r3
       case (21)
          write(u) f % i1
       case (22)
          write(u) f % i2
       end select
     end associate
  end do
  close(u)

end program run_diagnostics
