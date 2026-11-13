! The self-describing input and output streams the generic MPAS-Ocean oracle
! drivers (run_diagnostics, run_tracer_advection, ...) share.
!
! INPUT: dt, the count of configs and of fields, then every config (name,
! type: 1 real, 2 integer, 3 logical, 4 character; value) and every field (pool, name,
! kind, rank, Fortran shape, words).  Field kinds: 1 real array, 2 integer
! array, 3 a dimension, 4 a dimension array (nCellsArray, ...), 5 an absent
! field (a package that is off: an unassociated pointer, which the routines
! test with associated(); rank 4 marks an integer one), 6 a scratch field
! (its shape only).  The configs go in one pool (ocnConfigs once
! ocn_constants_init has run), the fields in the pools they name.
!
! Pools: mesh, state, tracers (the state pool's 'tracers' subpool),
! diagnostics, forcing, scratch, tend and tracersTend (the tend pool's
! 'tracersTend' subpool), tracersSurfaceFlux (the forcing pool's), swForcing,
! verticalMesh.  A logical config named 'package:NAME' is the package
! switch NAME, put in the packages pool (ocnPackages); one named 'mesh:NAME'
! is the mesh pool's config NAME (on_a_sphere).
!
! OUTPUT: every real and integer field's words, in the input's order.
module oracle_stream
  use mpas_kind_types
  use mpas_derived_types
  use mpas_pool_routines
  implicit none
  private
  public :: oracle_pools, oracle_field, poolNames, read_stream, write_stream, pool_named

  integer, parameter :: maxFields = 256
  character(len=24), parameter :: poolNames(11) = [character(len=24) :: 'mesh', 'state', 'tracers', &
       'diagnostics', 'forcing', 'scratch', 'tend', 'tracersTend', 'tracersSurfaceFlux', 'swForcing', 'verticalMesh']

  type oracle_field
    character(len=24) :: pool
    character(len=64) :: name
    integer :: kind = 0, rank = 0
    real(kind=RKIND), dimension(:), pointer :: r1 => null()
    real(kind=RKIND), dimension(:,:), pointer :: r2 => null()
    real(kind=RKIND), dimension(:,:,:), pointer :: r3 => null()
    integer, dimension(:), pointer :: i1 => null()
    integer, dimension(:,:), pointer :: i2 => null()
  end type oracle_field

  type pool_ref
    type(MPAS_pool_type), pointer :: p => null()
  end type pool_ref

  type oracle_pools
    type(pool_ref) :: pool(11)
    type(MPAS_pool_type), pointer :: configs => null(), packages => null()
  end type oracle_pools

contains

  function pool_named(pools, name) result(p)
    type(oracle_pools), intent(in) :: pools
    character(len=*), intent(in) :: name
    type(MPAS_pool_type), pointer :: p
    integer :: i
    i = findloc(poolNames, name, 1)
    if (i == 0) stop 10
    p => pools % pool(i) % p
  end function pool_named

  subroutine read_stream(path, dt, pools, fields)
    character(len=*), intent(in) :: path
    real(kind=RKIND), intent(out) :: dt
    type(oracle_pools), intent(inout) :: pools
    type(oracle_field), allocatable, intent(out) :: fields(:)
    integer :: nConfigs, nFields, i, u, ctype, dims(3), p, err, ivalue
    character(len=64) :: cname
    character(len=StrKIND) :: cvalue
    real(kind=RKIND) :: rvalue
    real(kind=RKIND), pointer :: rc
    logical, pointer :: lc
    integer, pointer :: ic
    character(len=StrKIND), pointer :: cc
    integer, pointer :: d0
    integer, dimension(:), pointer :: d1
    real(kind=RKIND), dimension(:), pointer :: null1
    real(kind=RKIND), dimension(:,:), pointer :: null2
    real(kind=RKIND), dimension(:,:,:), pointer :: null3
    integer, dimension(:), pointer :: nullI1
    type(field1DReal), pointer :: s1
    type(field2DReal), pointer :: s2
    type(field3DReal), pointer :: s3

    allocate(pools % configs, pools % packages)
    do p = 1, size(poolNames)
       allocate(pools % pool(p) % p)
    end do
    call stub_pool_put_subpool(pool_named(pools, 'state'), 'tracers', pool_named(pools, 'tracers'))
    call stub_pool_put_subpool(pool_named(pools, 'tend'), 'tracersTend', pool_named(pools, 'tracersTend'))
    call stub_pool_put_subpool(pool_named(pools, 'forcing'), 'tracersSurfaceFlux', &
         pool_named(pools, 'tracersSurfaceFlux'))
    nullify(null1, null2, null3, nullI1)

    open(newunit=u, file=trim(path), access='stream', form='unformatted', status='old')
    read(u) dt, nConfigs, nFields
    do i = 1, nConfigs
       read(u) cname, ctype
       select case (ctype)
       case (1)
          read(u) rvalue
          allocate(rc)
          rc = rvalue
          call stub_pool_put_config(pools % configs, trim(cname), rc)
       case (2)
          read(u) ivalue
          allocate(ic)
          ic = ivalue
          call stub_pool_put_config(pools % configs, trim(cname), ic)
       case (3)
          read(u) ivalue
          allocate(lc)
          lc = ivalue /= 0
          if (cname(1:8) == 'package:') then
             call stub_pool_put_config(pools % packages, trim(cname(9:)), lc)
          else if (cname(1:5) == 'mesh:') then
             call stub_pool_put_config(pool_named(pools, 'mesh'), trim(cname(6:)), lc)
          else
             call stub_pool_put_config(pools % configs, trim(cname), lc)
          end if
       case (4)
          read(u) cvalue
          allocate(cc)
          cc = cvalue
          call stub_pool_put_config(pools % configs, trim(cname), cc)
       case default
          stop 8
       end select
    end do

    if (nFields > maxFields) stop 9
    allocate(fields(nFields))
    do i = 1, nFields
       read(u) fields(i) % pool, fields(i) % name, fields(i) % kind, fields(i) % rank, dims
       associate (f => fields(i), pool => pool_named(pools, trim(fields(i) % pool)))
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
            call stub_pool_put_array(pool, trim(f % name), nullI1)
         case (61)
            allocate(s1)
            s1 % dimSizes = dims(1:1)
            call stub_pool_put_field(pool, trim(f % name), s1)
         case (62)
            allocate(s2)
            s2 % dimSizes = dims(1:2)
            call stub_pool_put_field(pool, trim(f % name), s2)
         case (63)
            allocate(s3)
            s3 % dimSizes = dims(1:3)
            call stub_pool_put_field(pool, trim(f % name), s3)
         case default
            stop 11
         end select
       end associate
    end do
    read(u, iostat=err) ivalue
    if (err == 0) stop 12   ! trailing input words
    close(u)
  end subroutine read_stream

  subroutine write_stream(path, fields)
    character(len=*), intent(in) :: path
    type(oracle_field), intent(in) :: fields(:)
    integer :: i, u
    open(newunit=u, file=trim(path), access='stream', form='unformatted', status='replace')
    do i = 1, size(fields)
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
  end subroutine write_stream

end module oracle_stream
