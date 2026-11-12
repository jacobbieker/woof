! WRF-Chem's idealized background gas profile for chem_opt = 1 / 2 (RADM2,
! RADM2SORG) with chem_in_opt = 0, and the RADM2 lateral boundary value used
! when have_bcs_chem = .false., from chem/module_input_chem_data.F.
!
! tables      the module DATA (xl, zfa, zfa_bdy, dens, so4vaptoaer), grav as
!             module_input_chem_data sees it (module_data_sorgam's), and the
!             iref / fracref maps setup_gasprofile_maps builds for chem_opt 1.
! ic_c<o>_nz<n>  real.exe's call (main/real_em.F:553): input_chem_profile on a
!             6 x 2 grid with kde = n + 1 w levels, chem_opt = o.  ph_1 and phb
!             are written, then the whole chem array after the call.  Columns
!             run from below sea level to 4.5 km terrain, and tops from 300 m
!             to 32 km, so mass levels fall below the first profile level
!             (42.5 m), between every pair, and above the last (18.5 km).
! mk_c<o>     make_chem_profile on one level of hand-picked mass heights:
!             every profile level exactly, the float32 neighbours of each,
!             midpoints, and heights far below and above the profile.
! bdy         bdy_chem_value(chem, z, nch, numgas) for nch = p_so2 ... p_ho2
!             at the same heights, for chem_opt 1 and 2 (p_nu0 > 1 only for 2).
! pnnl        gas_ic_opt = gas_bc_opt = 101: input_chem_profile with
!             gas_ic_opt = 101 on the nz = 20 grid, and bdy_chem_value after
!             gasprofile_init_pnnl (CASENAME 4 rewrites only xl(iref(p_sulf-1),:)).
!             Runs last because it changes the module's xl.
program run_gasprofile
  use oracle_io
  use module_domain
  use module_data_sorgam, only: grav
  use module_chem_share, only: get_last_gas
  use module_input_chem_data
  implicit none
  integer, parameter :: nx = 6, ny = 2, nlev = 5
  integer, parameter :: nzs(nlev) = (/ 1, 3, 20, 44, 60 /)
  real, parameter :: surf(nx * ny) = (/ -30., 0., 15., 250., 1500., 4500., &
                                        0., 42.5, 120., 800., 2600., 3300. /)
  real, parameter :: tops(nx * ny) = (/ 300., 20000., 18500., 26000., 32000., 21000., &
                                        9000., 18543., 30000., 15000., 25000., 27500. /)
  real :: zm(400), bvals(400, 42, 2)
  integer :: nzm, l, o, n, opts(2) = (/ radm2, radm2sorg /)
  character(len=1024) :: root
  character(len=32) :: label
  type(domain) :: grid

  call oracle_root(root)

  ! tables
  call set_package(radm2)
  n = get_last_gas(radm2)
  call setup_gasprofile_maps(radm2, n)
  call oracle_open('tables')
  call oracle_put('xl', xl)
  call oracle_put('zfa', zfa)
  call oracle_put('zfa_bdy', zfa_bdy)
  call oracle_put('dens', dens)
  call oracle_put('so4vaptoaer', so4vaptoaer)
  call oracle_put('grav', grav)
  call oracle_put('iref', iref(1:50))
  call oracle_put('fracref', fracref(1:50))
  call oracle_put('numgas_radm2', n)
  call set_package(radm2sorg)
  call oracle_put('numgas_radm2sorg', get_last_gas(radm2sorg))
  call oracle_put('p_nu0_radm2sorg', p_nu0)
  call oracle_close()

  ! ic
  do o = 1, 2
    do l = 1, nlev
      if (o == 2 .and. nzs(l) /= 20 .and. nzs(l) /= 3) cycle
      write(label, '(a,i1,a,i2.2)') 'ic_c', opts(o), '_nz', nzs(l)
      call run_ic(opts(o), nzs(l), gas_ic_default, trim(label))
    end do
  end do

  ! mass heights for mk and bdy
  call edge_heights(zm, nzm)
  do o = 1, 2
    write(label, '(a,i1)') 'mk_c', opts(o)
    call run_mk(opts(o), trim(label))
  end do
  call run_bdy(bvals)
  call oracle_open('bdy')
  call oracle_put('z', zm(1:nzm))
  call oracle_put('chem', bvals(1:nzm, :, :))
  call oracle_close()

  ! pnnl, last: gasprofile_init_pnnl rewrites the module's xl.
  call run_ic(radm2, 20, gas_ic_pnnl, 'pnnl_ic_c1_nz20')
  call set_package(radm2sorg)
  call setup_gasprofile_maps(radm2sorg, get_last_gas(radm2sorg))
  call gasprofile_init_pnnl(radm2sorg)
  call run_bdy(bvals)
  call oracle_open('pnnl_bdy')
  call oracle_put('z', zm(1:nzm))
  call oracle_put('chem', bvals(1:nzm, :, :))
  call oracle_put('xl', xl)
  call oracle_close()

contains

  subroutine run_ic(chem_opt, nz, gas_ic_opt, case)
    integer, intent(in) :: chem_opt, nz, gas_ic_opt
    character(len=*), intent(in) :: case
    integer :: i, j, k, c, kde
    real :: zf, frac
    kde = nz + 1
    call set_package(chem_opt)
    grid%chem_opt = chem_opt
    grid%gas_ic_opt = gas_ic_opt
    ! ids..kpe: domain, memory and patch all 1:nx, 1:kde, 1:ny.
    grid%dims = (/ 1, nx + 1, 1, ny + 1, 1, kde, 1, nx, 1, ny, 1, kde, &
                   1, nx, 1, ny, 1, kde /)
    if (associated(grid%ph_1)) deallocate(grid%ph_1, grid%phb, grid%chem)
    allocate(grid%ph_1(nx, kde, ny), grid%phb(nx, kde, ny), grid%chem(nx, kde, ny, num_chem))
    do j = 1, ny
      do i = 1, nx
        c = i + (j - 1) * nx
        do k = 1, kde
          frac = real(k - 1) / real(nz)
          zf = surf(c) + (tops(c) - surf(c)) * frac * sqrt(frac)
          grid%phb(i, k, j) = 9.81 * zf
          grid%ph_1(i, k, j) = real(mod(7 * c + 3 * k, 11) - 5) * 0.37
        end do
      end do
    end do
    grid%chem = -7.
    call input_chem_profile(grid)
    call oracle_open(case)
    call oracle_put('chem_opt', chem_opt)
    call oracle_put('gas_ic_opt', gas_ic_opt)
    call oracle_put('ph_1', grid%ph_1)
    call oracle_put('phb', grid%phb)
    call oracle_put('chem', grid%chem)
    call oracle_close()
  end subroutine run_ic

  subroutine edge_heights(z, n)
    real, intent(out) :: z(:)
    integer, intent(out) :: n
    real :: zp, zq
    integer :: k
    n = 0
    call add_z(z, n, -500.); call add_z(z, n, -1.e-3); call add_z(z, n, 0.); call add_z(z, n, 1.)
    do k = 1, kx
      zp = 0.5 * (zfa(k) + zfa(k + 1))
      call add_z(z, n, nearest(zp, -1.)); call add_z(z, n, zp); call add_z(z, n, nearest(zp, 1.))
      if (k < kx) then
        zq = 0.5 * (zfa(k + 1) + zfa(k + 2))
        call add_z(z, n, 0.5 * (zp + zq)); call add_z(z, n, zp + 0.1 * (zq - zp))
      end if
    end do
    call add_z(z, n, 18500.5); call add_z(z, n, 21000.); call add_z(z, n, 35000.); call add_z(z, n, 1.e5)
  end subroutine edge_heights

  subroutine add_z(z, n, v)
    real, intent(inout) :: z(:)
    integer, intent(inout) :: n
    real, intent(in) :: v
    n = n + 1
    z(n) = v
  end subroutine add_z

  subroutine run_mk(chem_opt, case)
    integer, intent(in) :: chem_opt
    character(len=*), intent(in) :: case
    real, allocatable :: zg(:, :, :), ch(:, :, :, :)
    integer :: numgas
    call set_package(chem_opt)
    numgas = get_last_gas(chem_opt)
    call setup_gasprofile_maps(chem_opt, numgas)
    allocate(zg(nzm, 1, 1), ch(nzm, 1, 1, num_chem))
    zg(:, 1, 1) = zm(1:nzm)
    ch = -7.
    call make_chem_profile(1, nzm, 1, 1, 1, 1, num_chem, numgas, chem_opt, zg, ch)
    call oracle_open(case)
    call oracle_put('chem_opt', chem_opt)
    call oracle_put('z', zg)
    call oracle_put('chem', ch)
    call oracle_close()
  end subroutine run_mk

  subroutine run_bdy(vals)
    real, intent(out) :: vals(:, :, :)
    integer :: o, nch, i, numgas
    real :: v
    do o = 1, 2
      call set_package(opts(o))
      numgas = get_last_gas(opts(o))
      ! wrf.exe: chemics_init calls setup_gasprofile_maps (chem/chemics_init.F:657).
      call setup_gasprofile_maps(opts(o), numgas)
      do nch = p_so2, p_ho2
        do i = 1, nzm
          call bdy_chem_value(v, zm(i), nch, numgas)
          vals(i, nch - 1, o) = v
        end do
      end do
    end do
  end subroutine run_bdy

end program run_gasprofile
