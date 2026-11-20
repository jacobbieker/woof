! Drive E3SM v3.2.0's MPAS-Ocean coupler import and export (ocn_import_mct
! and ocn_export_mct, cut verbatim out of components/mpas-ocean/driver/
! ocn_comp_mct.F) over pools read from a stream.
!
!   run_ocean_coupling INPUT OUTPUT
!
! INPUT (little-endian):
!   ocn_cpl_dt, ocn_c2_glctf, rof2ocn_nutrients, ocn_lnd_one_way, glc_nzoc,
!     config_momentum_use_active_wave,
!     config_use_ecosysTracers_river_inputs_from_coupler,
!     config_use_ecosysTracers_iron_solubility_file,
!     config_use_ecosysTracers_atm_dust_from_coupler (int32 each);
!   precip_fact, config_frazil_heat_of_fusion, the land-ice cavity freezing
!     coefficients 0, S, p, pS and the open-ocean ones 0, S, p, pS,
!     mushy_az1_liq (r8 each); wav_ocn_coup (8 characters);
!   indexBLT, indexBLS, indexHeatTrans, indexSaltTrans, nBL (int32), then
!     landIceBoundaryLayerTracers and landIceTracerTransferVelocities
!     (2, nBL) as r8;
!   the x2o and o2x field lists (length, characters); lsize; x2o (nx2o,
!     lsize) and o2x (no2x, lsize) as r8;
!   nEntries, then per entry: pool (1 configs, 2 packages, 3 mesh, 4 forcing,
!     5 state), kind (1 logical, 2 real, 3 character, 4 integer dimension,
!     5 real 1-D, 6 real 2-D, 7 integer 1-D, 8 an absent real 1-D field,
!     9 an absent real 2-D field, 10 an absent integer 1-D array, which
!     mpas_pool_get_array returns disassociated), its name (64 characters),
!     and its value:
!     int32 0/1, r8, 64 characters, int32, or the extents (int32) and the
!     array.
! OUTPUT: o2x after ocn_export_mct, then every array entry (kinds 5, 6, 7) in
! the order it was read, after ocn_import_mct and ocn_export_mct.
program run_ocean_coupling
  use shr_kind_mod, only: r8 => SHR_KIND_R8
  use seq_flds_mod, only: wav_ocn_coup, rof2ocn_nutrients, ocn_lnd_one_way, &
                          seq_flds_x2o_fields, seq_flds_o2x_fields
  use seq_infodata_mod, only: seq_infodata_type
  use glc_zocnclass_mod, only: glc_zocnclass_init
  use mpaso_cpl_indices, only: mpaso_cpl_indices_set
  use mpas_kind_types
  use mpas_derived_types
  use mpas_pool_routines
  use ocn_config
  use ocn_diagnostics_variables
  use ocn_constants, only: ocn_constants_init
  use ocn_comp_mct_cut, only: ocn_import_mct, ocn_export_mct, domain, infodata, ocn_cpl_dt
  implicit none
  integer, parameter :: maxArrays = 256
  character(len=4096) :: path_in, path_out
  character(len=8) :: wav
  character(len=64) :: name
  character(len=:), allocatable :: rlist
  integer :: uin, uout, i, n, pool, kind, n1, n2, lsize, nx2o, no2x, nBL, glc_nzoc, nArrays, ierr
  integer :: c2_glctf, nutrients, one_way, active_wave, river_inputs, iron_file, dust
  integer :: ival
  real(r8) :: precip_fact
  real(r8), allocatable :: x2o(:,:), o2x(:,:)
  type(MPAS_pool_type), pointer :: pools(:)
  type(MPAS_pool_type), pointer :: sub
  integer, allocatable :: arrayPool(:), arrayKind(:)
  type r1p
    real(kind=RKIND), dimension(:), pointer :: a => null()
  end type r1p
  type r2p
    real(kind=RKIND), dimension(:,:), pointer :: a => null()
  end type r2p
  type i1p
    integer, dimension(:), pointer :: a => null()
  end type i1p
  type(r1p) :: r1(maxArrays)
  type(r2p) :: r2(maxArrays)
  type(i1p) :: i1(maxArrays)
  logical, pointer :: lval
  real(kind=RKIND), pointer :: rval
  character(len=StrKIND), pointer :: cval
  integer, pointer :: dval
  integer, dimension(:), pointer :: inull => null()

  call get_command_argument(1, path_in)
  call get_command_argument(2, path_out)
  open(newunit=uin, file=trim(path_in), access='stream', form='unformatted', status='old')
  read(uin) ocn_cpl_dt, c2_glctf, nutrients, one_way, glc_nzoc, active_wave, river_inputs, iron_file, dust
  read(uin) precip_fact, config_frazil_heat_of_fusion, &
       config_land_ice_cavity_freezing_temperature_coeff_0, config_land_ice_cavity_freezing_temperature_coeff_S, &
       config_land_ice_cavity_freezing_temperature_coeff_p, config_land_ice_cavity_freezing_temperature_coeff_pS, &
       config_open_ocean_freezing_temperature_coeff_0, config_open_ocean_freezing_temperature_coeff_S, &
       config_open_ocean_freezing_temperature_coeff_p, config_open_ocean_freezing_temperature_coeff_pS, &
       config_open_ocean_freezing_temperature_coeff_mushy_az1_liq
  read(uin) wav
  config_momentum_use_active_wave = active_wave /= 0
  config_use_ecosysTracers_river_inputs_from_coupler = river_inputs /= 0
  config_use_ecosysTracers_iron_solubility_file = iron_file /= 0
  config_use_ecosysTracers_atm_dust_from_coupler = dust /= 0
  rof2ocn_nutrients = nutrients /= 0
  ocn_lnd_one_way = one_way /= 0
  wav_ocn_coup = trim(wav)
  allocate(infodata)
  infodata % precip_fact = precip_fact
  infodata % ocn_c2_glctf = c2_glctf /= 0
  read(uin) indexBLT, indexBLS, indexHeatTrans, indexSaltTrans, nBL
  allocate(landIceBoundaryLayerTracers(2, nBL), landIceTracerTransferVelocities(2, nBL))
  read(uin) landIceBoundaryLayerTracers, landIceTracerTransferVelocities

  read(uin) n
  allocate(character(len=n) :: rlist)
  read(uin) rlist
  seq_flds_x2o_fields = rlist
  deallocate(rlist)
  read(uin) n
  allocate(character(len=n) :: rlist)
  read(uin) rlist
  seq_flds_o2x_fields = rlist
  deallocate(rlist)
  read(uin) lsize, nx2o, no2x
  allocate(x2o(nx2o, lsize), o2x(no2x, lsize))
  read(uin) x2o, o2x

  call glc_zocnclass_init(glc_nzoc)
  call mpaso_cpl_indices_set()

  ! one block: structs holds mesh, forcing and state (state holds tracers)
  allocate(domain)
  allocate(domain % blocklist)
  allocate(domain % blocklist % structs, domain % configs, domain % packages)
  allocate(pools(5))
  sub => pools(3)
  call stub_pool_put_subpool(domain % blocklist % structs, 'mesh', sub)
  sub => pools(4)
  call stub_pool_put_subpool(domain % blocklist % structs, 'forcing', sub)
  sub => pools(5)
  call stub_pool_put_subpool(domain % blocklist % structs, 'state', sub)
  allocate(sub)
  call stub_pool_put_subpool(pools(5), 'tracers', sub)

  read(uin) n
  allocate(arrayPool(n), arrayKind(n))
  nArrays = 0
  do i = 1, n
     read(uin) pool, kind, name
     select case (kind)
     case (1)
        read(uin) ival
        allocate(lval)
        lval = ival /= 0
        if (pool == 1) call stub_pool_put_config(domain % configs, trim(name), lval)
        if (pool == 2) then
           call stub_pool_put_config(domain % packages, trim(name), lval)
        end if
     case (2)
        allocate(rval)
        read(uin) rval
        call stub_pool_put_config(domain % configs, trim(name), rval)
     case (3)
        allocate(cval)
        cval = ''
        read(uin) cval(1:64)
        call stub_pool_put_config(domain % configs, trim(name), cval)
     case (4)
        allocate(dval)
        read(uin) dval
        call stub_pool_put_dimension(target_pool(pool), trim(name), dval)
     case (5, 8)
        nArrays = nArrays + 1
        if (kind == 5) then
           read(uin) n1
           allocate(r1(nArrays) % a(n1))
           read(uin) r1(nArrays) % a
        end if
        call stub_pool_put_array(target_pool(pool), trim(name), r1(nArrays) % a)
        if (kind == 8) call stub_pool_mark_absent(target_pool(pool), trim(name))
        arrayPool(nArrays) = pool
        arrayKind(nArrays) = kind
     case (6, 9)
        nArrays = nArrays + 1
        if (kind == 6) then
           read(uin) n1, n2
           allocate(r2(nArrays) % a(n1, n2))
           read(uin) r2(nArrays) % a
        end if
        call stub_pool_put_array(target_pool(pool), trim(name), r2(nArrays) % a)
        if (kind == 9) call stub_pool_mark_absent(target_pool(pool), trim(name))
        arrayPool(nArrays) = pool
        arrayKind(nArrays) = kind
     case (7)
        nArrays = nArrays + 1
        read(uin) n1
        allocate(i1(nArrays) % a(n1))
        read(uin) i1(nArrays) % a
        call stub_pool_put_array(target_pool(pool), trim(name), i1(nArrays) % a)
        arrayPool(nArrays) = pool
        arrayKind(nArrays) = kind
     case (10)
        call stub_pool_put_array(target_pool(pool), trim(name), inull)
     case default
        stop 9
     end select
  end do
  close(uin)

  call ocn_constants_init(domain % configs, domain % packages)
  call ocn_import_mct(x2o, ierr)
  call ocn_export_mct(o2x, ierr)

  open(newunit=uout, file=trim(path_out), access='stream', form='unformatted', status='replace')
  write(uout) o2x
  do i = 1, nArrays
     select case (arrayKind(i))
     case (5)
        write(uout) r1(i) % a
     case (6)
        write(uout) r2(i) % a
     case (7)
        write(uout) i1(i) % a
     end select
  end do
  close(uout)

contains

  function target_pool(code) result(p)
    integer, intent(in) :: code
    type(MPAS_pool_type), pointer :: p
    p => pools(code)
  end function target_pool

end program run_ocean_coupling
