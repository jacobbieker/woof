! Drive E3SM v3.2.0's MPAS-Seaice coupler import and export (ice_import_mct
! and ice_export_mct, cut verbatim out of components/mpas-seaice/driver/
! ice_comp_mct.F) over pools read from a stream.
!
!   run_seaice_coupling INPUT OUTPUT
!
! INPUT (little-endian):
!   wav_nfreq, ktherm (int32); wav_ice_coup (8 characters); tfrz_option (16
!     characters);
!   the x2i and i2x field lists (length, characters); lsize; x2i (nx2i,
!     lsize) and i2x (ni2x, lsize) as r8;
!   nEntries, then per entry: its pool (32 characters: configs, dimensions,
!     or a subpool of the block's structs by name), kind (1 logical, 2 real,
!     3 character, 4 integer dimension, 5 real 1-D, 6 real 2-D, 8 an absent
!     real 1-D field, 9 an absent real 2-D field), its name (64 characters),
!     and its value: int32 0/1, r8, 64 characters, int32, or the extents
!     (int32) and the array.
! OUTPUT: i2x after ice_export_mct, then every array entry (kinds 5, 6) in the
! order it was read, after ice_import_mct and ice_export_mct, then the
! rotation's trigonometric words (8, nCellsSolve): per cell sin and cos of
! latCell, of lonCell, and of the rotated grid's latitude and longitude, which
! E3SM's own seaice_grid_rotation_forward and seaice_latlon_from_xyz give
! (asin, atan2). They are the host libm's (glibc's) words, as the export's
! seaice_latlon_vector_rotation_backward computes them.
!
! Icepack's parameters are set as E3SM's MPAS-Seaice sets those the two
! routines reach (mpas_seaice_icepack.F's icepack_init_parameters call, from
! seaice_constants): puny, rhos, rhoi, rhow, cp_ice, cp_ocn, depressT,
! Tocnfrz, Tffresh, Lsub, Lvap (so Lfresh = Lsub - Lvap), ktherm and
! tfrz_option.
program run_seaice_coupling
  use shr_kind_mod, only: r8 => SHR_KIND_R8
  use seq_flds_mod, only: wav_ice_coup, wav_nfreq, seq_flds_x2i_fields, seq_flds_i2x_fields
  use mpassi_cpl_indices, only: mpassi_cpl_indices_set
  use mpas_kind_types
  use mpas_derived_types
  use mpas_pool_routines
  use seaice_constants
  use icepack_intfc, only: icepack_init_parameters, icepack_warnings_aborted
  use ice_comp_mct_cut, only: ice_import_mct, ice_export_mct, domain
  use seaice_mesh, only: seaice_grid_rotation_forward, seaice_latlon_from_xyz
  implicit none
  integer, parameter :: maxArrays = 256, maxPools = 32
  character(len=4096) :: path_in, path_out
  character(len=8) :: wav
  character(len=16) :: tfrz
  character(len=32) :: pname, poolNames(maxPools)
  character(len=64) :: name
  character(len=:), allocatable :: rlist
  integer :: uin, uout, i, n, kind, n1, n2, lsize, nx2i, ni2x, nArrays, nPools, ierr, ktherm, ival
  real(r8), allocatable :: x2i(:,:), i2x(:,:)
  type(MPAS_pool_type), pointer :: sub, p
  type pp
    type(MPAS_pool_type), pointer :: p => null()
  end type pp
  type(pp) :: pools(maxPools)
  integer, allocatable :: arrayKind(:)
  type r1p
    real(kind=RKIND), dimension(:), pointer :: a => null()
  end type r1p
  type r2p
    real(kind=RKIND), dimension(:,:), pointer :: a => null()
  end type r2p
  type(r1p) :: r1(maxArrays)
  type(r2p) :: r2(maxArrays)
  logical, pointer :: lval
  real(kind=RKIND), pointer :: rval
  character(len=StrKIND), pointer :: cval
  integer, pointer :: dval
  type(MPAS_pool_type), pointer :: meshPool
  integer, pointer :: nCellsSolve
  logical, pointer :: rotate
  real(kind=RKIND), pointer :: radius
  real(kind=RKIND), dimension(:), pointer :: latCell, lonCell, xCell, yCell, zCell
  real(kind=RKIND) :: xp, yp, zp, latp, lonp

  call get_command_argument(1, path_in)
  call get_command_argument(2, path_out)
  open(newunit=uin, file=trim(path_in), access='stream', form='unformatted', status='old')
  read(uin) wav_nfreq, ktherm
  read(uin) wav, tfrz
  wav_ice_coup = trim(wav)
  read(uin) n
  allocate(character(len=n) :: rlist)
  read(uin) rlist
  seq_flds_x2i_fields = rlist
  deallocate(rlist)
  read(uin) n
  allocate(character(len=n) :: rlist)
  read(uin) rlist
  seq_flds_i2x_fields = rlist
  deallocate(rlist)
  read(uin) lsize, nx2i, ni2x
  allocate(x2i(nx2i, lsize), i2x(ni2x, lsize))
  read(uin) x2i, i2x

  call mpassi_cpl_indices_set()
  call icepack_init_parameters( &
       puny_in = seaicePuny, rhos_in = seaiceDensitySnow, rhoi_in = seaiceDensityIce, &
       rhow_in = seaiceDensitySeaWater, cp_ice_in = seaiceFreshIceSpecificHeat, &
       cp_ocn_in = seaiceSeaWaterSpecificHeat, depressT_in = seaiceMeltingTemperatureDepression, &
       Tocnfrz_in = seaiceFreezingTemperatureConstant, Tffresh_in = seaiceFreshWaterFreezingPoint, &
       Lsub_in = seaiceLatentHeatSublimation, Lvap_in = seaiceLatentHeatVaporization, &
       ktherm_in = ktherm, tfrz_option_in = trim(tfrz))
  if (icepack_warnings_aborted()) stop 11

  ! one block; the domain's configs are the block's
  allocate(domain)
  allocate(domain % blocklist)
  allocate(domain % blocklist % structs, domain % blocklist % dimensions, domain % configs, domain % packages)
  domain % blocklist % configs => domain % configs
  domain % blocklist % packages => domain % packages
  nPools = 0

  read(uin) n
  allocate(arrayKind(n))
  nArrays = 0
  do i = 1, n
     read(uin) pname, kind, name
     p => target_pool(pname)
     select case (kind)
     case (1)
        read(uin) ival
        allocate(lval)
        lval = ival /= 0
        call stub_pool_put_config(p, trim(name), lval)
     case (2)
        allocate(rval)
        read(uin) rval
        call stub_pool_put_config(p, trim(name), rval)
     case (3)
        allocate(cval)
        cval = ''
        read(uin) cval(1:64)
        call stub_pool_put_config(p, trim(name), cval)
     case (4)
        allocate(dval)
        read(uin) dval
        call stub_pool_put_dimension(p, trim(name), dval)
     case (5, 8)
        nArrays = nArrays + 1
        if (kind == 5) then
           read(uin) n1
           allocate(r1(nArrays) % a(n1))
           read(uin) r1(nArrays) % a
        end if
        call stub_pool_put_array(p, trim(name), r1(nArrays) % a)
        if (kind == 8) call stub_pool_mark_absent(p, trim(name))
        arrayKind(nArrays) = kind
     case (6, 9)
        nArrays = nArrays + 1
        if (kind == 6) then
           read(uin) n1, n2
           allocate(r2(nArrays) % a(n1, n2))
           read(uin) r2(nArrays) % a
        end if
        call stub_pool_put_array(p, trim(name), r2(nArrays) % a)
        if (kind == 9) call stub_pool_mark_absent(p, trim(name))
        arrayKind(nArrays) = kind
     case default
        stop 9
     end select
  end do
  close(uin)

  call ice_import_mct(x2i, ierr)
  if (ierr /= 0) stop 12
  call ice_export_mct(i2x, ierr)
  if (ierr /= 0) stop 13

  open(newunit=uout, file=trim(path_out), access='stream', form='unformatted', status='replace')
  write(uout) i2x
  do i = 1, nArrays
     select case (arrayKind(i))
     case (5)
        write(uout) r1(i) % a
     case (6)
        write(uout) r2(i) % a
     end select
  end do
  call mpas_pool_get_subpool(domain % blocklist % structs, 'mesh', meshPool)
  call mpas_pool_get_dimension(meshPool, 'nCellsSolve', nCellsSolve)
  call mpas_pool_get_config(meshPool, 'sphere_radius', radius)
  call mpas_pool_get_config(domain % configs, 'config_rotate_cartesian_grid', rotate)
  call mpas_pool_get_array(meshPool, 'latCell', latCell)
  call mpas_pool_get_array(meshPool, 'lonCell', lonCell)
  call mpas_pool_get_array(meshPool, 'xCell', xCell)
  call mpas_pool_get_array(meshPool, 'yCell', yCell)
  call mpas_pool_get_array(meshPool, 'zCell', zCell)
  do i = 1, nCellsSolve
     call seaice_grid_rotation_forward(xp, yp, zp, xCell(i), yCell(i), zCell(i), rotate)
     call seaice_latlon_from_xyz(latp, lonp, xp, yp, zp, radius)
     write(uout) sin(latCell(i)), cos(latCell(i)), sin(lonCell(i)), cos(lonCell(i)), &
                 sin(latp), cos(latp), sin(lonp), cos(lonp)
  end do
  close(uout)

contains

  ! configs, dimensions, or the structs' subpool of that name (made on first use)
  function target_pool(pool_name) result(q)
    character(len=*), intent(in) :: pool_name
    type(MPAS_pool_type), pointer :: q
    integer :: k
    if (trim(pool_name) == 'configs') then
       q => domain % configs
       return
    end if
    if (trim(pool_name) == 'dimensions') then
       q => domain % blocklist % dimensions
       return
    end if
    do k = 1, nPools
       if (trim(poolNames(k)) == trim(pool_name)) then
          q => pools(k) % p
          return
       end if
    end do
    nPools = nPools + 1
    if (nPools > maxPools) stop 10
    poolNames(nPools) = pool_name
    allocate(sub)
    pools(nPools) % p => sub
    call stub_pool_put_subpool(domain % blocklist % structs, trim(pool_name), sub)
    q => sub
  end function target_pool

end program run_seaice_coupling
