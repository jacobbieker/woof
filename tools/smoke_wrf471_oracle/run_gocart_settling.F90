! GOCART gravitational settling: WRF v4.7.1's settling
! (chem/module_gocart_settling.F:206-367, REAL*8, cut verbatim by
! extract.py), fed one column at a time exactly as gocart_settling_driver
! feeds it for dust (idust = 1, iseas = 0, module_gocart_settling.F:89-122):
!
!   p_mid(kk)  = .01*p_phy(kte-k+kts)    real(4) product, then REAL*8
!   delz(kk)   = dz8w(kte-k+kts)         (both reversed: top first)
!   airden(kk) = rho_phy(k), tmp(kk) = t_phy(k), tc(kk,n) = chem(k,n)
!
! with g = 9.81 (module_model_constants) and dyn_visc = 1.5E-5
! (phys/module_data_gocart_dust.F).  rh is not read on the dust path.
!
! Cases: PM2.5 smoke bins (radius 0.1-1.25 um, 1400 kg m-3), the five GOCART
! dust bins with and without the UoC density override (uoc = 1), time steps
! from below one second (ntdt = 0) to an hour (the 12-substep cap), and
! columns holding negative mixing ratios (reset to 1e-32).
!
! Writes fixtures/gocart_settling/<case>/: inputs (real(4) WRF fields,
! REAL*8 bin radii and densities) and the routine's REAL*8 outputs
!   tc_out (nz, nbin, ncol), graset (nbin, ncol), grasetvel (nbin, ncol).
program run_gocart_settling
  use oracle_io
  use smoke_cut_gocart_settling
  implicit none
  character(len=1024) :: root
  real(8) :: smoke_r(3), smoke_d(3), dust_r(5), dust_d(5)

  smoke_r = (/ 0.1d-6, 0.35d-6, 1.25d-6 /)
  smoke_d = (/ 1400.d0, 1400.d0, 1400.d0 /)
  dust_r = (/ 0.73D-6, 1.4D-6, 2.4D-6, 4.5D-6, 8.0D-6 /)
  dust_d = (/ 2500.d0, 2650.d0, 2650.d0, 2650.d0, 2650.d0 /)

  call oracle_root(root)
  call one_case('gocart_settling/smoke_dt60', 3, smoke_r, smoke_d, 60.0, 0, 40, 6, 25.0)
  call one_case('gocart_settling/smoke_dt900', 3, smoke_r, smoke_d, 900.5, 0, 40, 6, 25.0)
  call one_case('gocart_settling/dust_dt300', 5, dust_r, dust_d, 300.0, 0, 35, 5, 20.0)
  call one_case('gocart_settling/dust_uoc_dt3600', 5, dust_r, dust_d, 3600.0, 1, 35, 5, 6.0)
  call one_case('gocart_settling/dust_dt_subsecond', 5, dust_r, dust_d, 0.75, 0, 30, 4, 20.0)
  call one_case('gocart_settling/smoke_thin_dt3600', 3, smoke_r, smoke_d, 3600.0, 0, 50, 4, 0.5)
  call oracle_close()

contains

  subroutine one_case(name, nbin, reff, den, dt, uoc, nz, ncol, dz0)
    character(len=*), intent(in) :: name
    integer, intent(in) :: nbin, uoc, nz, ncol
    real(8), intent(in) :: reff(nbin), den(nbin)
    real, intent(in) :: dt, dz0
    real, parameter :: g = 9.81, dyn_visc = 1.5E-5
    real :: chem(nz, nbin, ncol), t_phy(nz, ncol), p_phy(nz, ncol)
    real :: dz8w(nz, ncol), rho_phy(nz, ncol)
    real(8) :: tmp(1, 1, nz), airden(1, 1, nz), p_mid(1, 1, nz)
    real(8) :: delz(1, 1, nz), rh(1, 1, nz), tc(1, 1, nz, nbin)
    real(8) :: graset(1, 1, nbin), grasetvel(1, 1, nbin)
    real(8) :: tc_out(nz, nbin, ncol), graset_out(nbin, ncol)
    real(8) :: vel_out(nbin, ncol)
    real :: z, dz, frac
    integer :: i, k, kk, n, kts, kte

    ! Columns: a stretched grid from dz0 upward, a lapse rate and a scale
    ! height that differ per column, and a tracer profile with a few
    ! negative words.
    do i = 1, ncol
      z = 0.0
      do k = 1, nz
        dz = dz0 * (1.0 + 0.11 * real(k - 1)) * (1.0 + 0.07 * real(i - 1))
        z = z + 0.5 * dz
        dz8w(k, i) = dz
        t_phy(k, i) = 301.3 - (6.1 + 0.4 * real(i)) * 1.0e-3 * z
        p_phy(k, i) = 101325.0 * exp(-z / (8100.0 + 150.0 * real(i)))
        rho_phy(k, i) = p_phy(k, i) / (287.0 * t_phy(k, i))
        z = z + 0.5 * dz
        do n = 1, nbin
          frac = real(mod(k * 7 + n * 3 + i * 5, 11)) / 11.0
          chem(k, n, i) = (3.0 + 40.0 * frac) * exp(-z / 2500.0) &
                          * real(n) + 0.25 * real(i)
          if (mod(k + 2 * n + i, 13) == 0) chem(k, n, i) = -0.5 * frac
        end do
      end do
    end do

    kts = 1
    kte = nz
    do i = 1, ncol
      kk = 0
      do k = kts, kte
        kk = kk + 1
        p_mid(1, 1, kk) = .01 * p_phy(kte - k + kts, i)
        delz(1, 1, kk) = dz8w(kte - k + kts, i)
        airden(1, 1, kk) = rho_phy(k, i)
        tmp(1, 1, kk) = t_phy(k, i)
        rh(1, 1, kk) = .95
        do n = 1, nbin
          tc(1, 1, kk, n) = chem(k, n, i)
        end do
      end do
      graset = 0.
      call settling(1, 1, nz, nbin, g, dyn_visc, tc, tmp, p_mid, delz, &
                    1, graset, grasetvel, uoc, den, reff, dt, rh, 1, 0, &
                    airden)
      do n = 1, nbin
        do k = 1, nz
          tc_out(k, n, i) = tc(1, 1, k, n)
        end do
        graset_out(n, i) = graset(1, 1, n)
        vel_out(n, i) = grasetvel(1, 1, n)
      end do
    end do

    call oracle_open(name)
    call oracle_put('dt', dt)
    call oracle_put('g', g)
    call oracle_put('dyn_visc', dyn_visc)
    call oracle_put('uoc', uoc)
    call oracle_put('reff', reff)
    call oracle_put('den', den)
    call oracle_put('chem', chem)
    call oracle_put('t_phy', t_phy)
    call oracle_put('p_phy', p_phy)
    call oracle_put('dz8w', dz8w)
    call oracle_put('rho_phy', rho_phy)
    call oracle_put('tc_out', tc_out)
    call oracle_put('graset', graset_out)
    call oracle_put('grasetvel', vel_out)
  end subroutine one_case

end program run_gocart_settling
