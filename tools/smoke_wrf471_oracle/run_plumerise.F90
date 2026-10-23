! Oracle driver for WRF v4.7.1's Freitas et al. (2007) plume-rise model,
! chem/module_chem_plumerise_scalar.F, compiled byte-unmodified.
!
!   ./run_plumerise --count          print the number of cases
!   ./run_plumerise ROOT N           run case N, write ROOT/plumerise/caseNN/
!
! The plume model keeps its whole state in module variables and zeroes them
! only on its first call (plumerise:117-120 calls zero_plumegen_coms once;
! get_env_condition:297-302 builds the plume grid once), so each case runs
! in a fresh process: build.sh starts one per case, and the plume tops are
! taken by a second fresh process this driver starts on itself (mode 'tops').
!
! Mode 'run' (the default) calls plumerise exactly as chem/module_plumerise1.F
! does for one column (eburn_out zeroed first, nspecies = 1) and writes the
! column inputs and eburn_out.  Mode 'tops' repeats plumerise's own sequence
! of calls -- zero_plumegen_coms, the column copy, get_env_condition,
! get_fire_properties and makeplume per biome and heating rate, the
! grassland shortcut and set_flam_vert -- and writes the plume tops ztopmax
! and the injection layer k1, k2 of every biome, which plumerise keeps local.
!
! The cases are synthetic soundings chosen to reach the model's branches:
! every biome (the grassland imm = 2 shortcut included), dry and moist
! columns (condensation, rain, ice, melting), calm and sheared wind,
! weak and strong stability, a fire too weak to lift the plume, and a column
! that holds several biomes at once.  Every column top is above 20 km, so
! the plume grid's kmt is 199 and no plume comes near kmt - deltak, where
! the model would read levels it never set.
program run_plumerise
  use oracle_io
  use module_zero_plumegen_coms, only: nkp, zero_plumegen_coms, ucon, vcon, &
      thtcon, picon, rvcon, zcon, zzcon, w_vmd, vmd, w, t, qv, qc, qh, qi, &
      rho, radius, vel_p, rad_p, wc, qsat, time, dt, mintime, wbar, dqsdz
  use module_chem_plumerise_scalar, only: plumerise, get_env_condition, &
      get_fire_properties, makeplume, set_flam_vert
  implicit none

  integer, parameter :: ncase = 8
  integer, parameter :: m1 = 48
  character(len=1024) :: root, arg, exe, mode
  character(len=16) :: case
  integer :: icase, n, stat

  real :: up(m1,1,1), vp(m1,1,1), wp(m1,1,1), theta(m1,1,1), pp(m1,1,1)
  real :: dn0(m1,1,1), rv(m1,1,1), zt_rams(m1), zm_rams(m1)
  real :: firesize(4), mean_fct(4), eburn_in(1), eburn_out(m1,1)

  call get_command_argument(1, arg, length=n)
  if (trim(arg) == '--count') then
    write(*, '(I0)') ncase
    stop
  end if
  call oracle_root(root)
  call get_command_argument(2, arg, length=n)
  if (n == 0) error stop 'usage: run_plumerise ROOT N [tops]'
  read(arg, *) icase
  if (icase < 1 .or. icase > ncase) error stop 'run_plumerise: no such case'
  mode = 'run'
  call get_command_argument(3, arg, length=n)
  if (n > 0) mode = arg
  write(case, '(A,I2.2)') 'plumerise/case', icase

  call make_case(icase)

  if (trim(mode) == 'run') then
    eburn_out = 0.
    call plumerise(m1, 1, 1, 1, 1, 1, 1, firesize, mean_fct, 1, eburn_in, &
                   eburn_out, up, vp, wp, theta, pp, dn0, rv, zt_rams, &
                   zm_rams)
    call oracle_open(trim(case))
    call oracle_put('u', up(:,1,1))
    call oracle_put('v', vp(:,1,1))
    call oracle_put('theta', theta(:,1,1))
    call oracle_put('pi', pp(:,1,1))
    call oracle_put('qv', rv(:,1,1))
    call oracle_put('zt', zt_rams)
    call oracle_put('zm', zm_rams)
    call oracle_put('firesize', firesize)
    call oracle_put('mean_fct', mean_fct)
    call oracle_put('eburn_in', eburn_in(1))
    call oracle_put('eburn_out', eburn_out(:,1))
    call oracle_close()
    ! The plume tops, from a fresh process of its own.
    call get_command_argument(0, exe)
    write(arg, '(I0)') icase
    call execute_command_line(trim(exe) // ' "' // trim(root) // '" ' // &
                              trim(arg) // ' tops', exitstat=stat)
    if (stat /= 0) error stop 'run_plumerise: the tops pass failed'
    if (icase == 1) call literals()
  else if (trim(mode) == 'tops') then
    call tops()
  else
    error stop 'run_plumerise: unknown mode'
  end if

contains

  ! gfortran's own words for every real(4) literal the port spells (in the
  ! order of woof.smoke.plume_rise.LITERALS) and for the constant
  ! expressions the routines fold: module_model_constants' cp, rcp and
  ! module_chem_plumerise_scalar's cpor, g*2. (get_env_condition), G*R,
  ! 1./3., 5./3. (lbound), the microphysics' HEATCOND/CP, HEATSUBL/CP,
  ! HEATFUS/CP, 2./3.1416, (2./3.1416)*0.5, 1./(1.+gama), F0**(-0.42),
  ! 0.0366*ANBASE and MOIST/100. (get_fire_properties' integer MOIST).
  subroutine literals()
    use module_model_constants, only: g, cp, rcp
    use module_chem_plumerise_scalar, only: cpor
    real, parameter :: f0 = 0.75, gama = 0.5, anbase = 100000.
    integer, parameter :: moist = 10
    real, parameter :: folded(16) = (/ g*2., cp, rcp, cpor, &
        9.80796*287.04, 1./3., 5./3., 2.5E3/1.004, 2834./1.004, &
        334./1.004, 2./3.1416, (2./3.1416)*0.5, 1./(1.+gama), &
        f0**(-0.42), 0.0366*anbase, moist/100. /)
    real :: lit(59)
      lit = (/ .61, 1e-8, 1.e-3, 3483.8, 0.622, 9.80796, 287.04, 1004.6, &
             3.14159, 19.3e6, 6.1115, 22.542, 273.48, 6.1121, 18.729, &
             257.87, 227.3, 273.3, 2.5E3, 1.004, 2834., 334., 5.44E-4, &
             269.3, 3.1416, 0.05, 0.55, 0.1, 0.0001, 1.0E-10, 1.E6, 0.475, &
             0.525, 1.13, 7.E5, 4.1E6, 0.025, 2.27, 1.E-6, 332.27, 0.0052, &
             .875, 0.146, 0.0366, 0.000001, 0.9, 1.2, 9.81, 1004., 9.8, &
             1.6, 0.57E-3, -0.42, .65, .2, 0.75, 1.5, 1.e-6, -40.0 /)
    call oracle_open('plumerise_literals')
    call oracle_put('literals', lit)
    call oracle_put('folded', folded)
    call oracle_close()
  end subroutine literals

  ! plumerise:96-197, the calls in plumerise's own order, keeping what it
  ! keeps local: ztopmax(imm) and k1, k2 of every biome, and the plume
  ! model's state as each makeplume returns (the module arrays on the plume
  ! grid and the scalars time, dt, mintime, wbar, dqsdz).
  subroutine tops()
    integer :: k, kmt, iveg_ag, imm, k1, k2, ixx
    real :: ztopmax(2), burnt_area
    real :: ztop_out(2,4)
    integer :: k1_out(4), k2_out(4)
    integer, parameter :: nst = 12
    real :: state(nkp, nst, 8), scal(4, 8)
    integer :: minute(8)
    state = 0.
    scal = 0.
    minute = 0
    call zero_plumegen_coms
    do k = 1, m1
      ucon(k) = up(k,1,1)
      vcon(k) = vp(k,1,1)
      thtcon(k) = theta(k,1,1)
      picon(k) = pp(k,1,1)
      rvcon(k) = rv(k,1,1)
      zcon(k) = zt_rams(k)
      zzcon(k) = zm_rams(k)
    end do
    call get_env_condition(1, m1, kmt, 1)
    ztop_out = 0.
    k1_out = 0
    k2_out = 0
    ixx = 0
    do iveg_ag = 1, 4
      if (mean_fct(iveg_ag) < 1.e-6) cycle
      burnt_area = firesize(iveg_ag)
      do imm = 1, 2
        call get_fire_properties(imm, iveg_ag, burnt_area, 0.)
        if (iveg_ag == 4 .and. imm == 2) then
          ztopmax(2) = ztopmax(1)
          ztopmax(1) = zzcon(1)
          cycle
        end if
        call makeplume(kmt, ztopmax(imm), ixx, imm)
        state(:, 1, imm + 2*(iveg_ag - 1)) = w
        state(:, 2, imm + 2*(iveg_ag - 1)) = t
        state(:, 3, imm + 2*(iveg_ag - 1)) = qv
        state(:, 4, imm + 2*(iveg_ag - 1)) = qc
        state(:, 5, imm + 2*(iveg_ag - 1)) = qh
        state(:, 6, imm + 2*(iveg_ag - 1)) = qi
        state(:, 7, imm + 2*(iveg_ag - 1)) = rho
        state(:, 8, imm + 2*(iveg_ag - 1)) = radius
        state(:, 9, imm + 2*(iveg_ag - 1)) = vel_p
        state(:, 10, imm + 2*(iveg_ag - 1)) = rad_p
        state(:, 11, imm + 2*(iveg_ag - 1)) = wc
        state(:, 12, imm + 2*(iveg_ag - 1)) = qsat
        scal(:, imm + 2*(iveg_ag - 1)) = (/ time, dt, wbar, dqsdz /)
        minute(imm + 2*(iveg_ag - 1)) = mintime
      end do
      call set_flam_vert(ztopmax, k1, k2, nkp, zzcon, W_VMD, VMD)
      ztop_out(:, iveg_ag) = ztopmax
      k1_out(iveg_ag) = k1
      k2_out(iveg_ag) = k2
    end do
    call oracle_open(trim(case) // '/tops')
    call oracle_put('ztopmax', ztop_out)
    call oracle_put('k1', k1_out)
    call oracle_put('k2', k2_out)
    call oracle_put('kmt', kmt)
    call oracle_put('state', state)
    call oracle_put('scalars', scal)
    call oracle_put('mintime', minute)
    call oracle_close()
  end subroutine tops

  ! One synthetic sounding per case.  Interfaces z_w(1) = 0 < ... < z_w(m1+1)
  ! stretch from 40 m to 900 m; zm_rams(k) = z_w(k) and zt_rams(k) is the
  ! layer middle (module_plumerise1.F:249-250).  theta: a mixed layer of
  ! depth zi, an inversion of dth_inv over 200 m, gam_ft K/km above and
  ! 15 K/km above 12 km.  The exner function is the hydrostatic integral of
  ! theta from 1000 hPa (pi_in = cp*(p/p1000mb)**rcp, line 247); qv falls
  ! off with a 2.5 km scale height from qv0 and is capped at 90 % of
  ! saturation.  The wind turns from (u0, v0) by (du, dv) per km.
  subroutine make_case(ic)
    integer, intent(in) :: ic
    real :: zw(m1+1), dz, zi, th0, dth_inv, gam_ft, qv0, u0, v0, du, dv
    real :: z, th, tk, pk, es, qs, p, pik, dpik
    real, parameter :: cp = 1004.5, g = 9.81, rcp = 287./1004.5
    integer :: k

    zw(1) = 0.
    do k = 2, m1 + 1
      dz = min(900., 40. * 1.11**(k - 2))
      zw(k) = zw(k - 1) + dz
    end do
    if (zw(m1 + 1) < 20200.) then
      dz = 20200. - zw(m1 + 1)
      do k = 2, m1 + 1
        zw(k) = zw(k) + dz * real(k - 1) / real(m1)
      end do
    end if
    do k = 1, m1
      zm_rams(k) = zw(k)
      zt_rams(k) = 0.5 * (zw(k) + zw(k + 1))
    end do

    firesize = 0.
    mean_fct = 0.
    eburn_in = 1.0e-6
    select case (ic)
    case (1)   ! tropical forest, dry, moderate wind
      zi = 1500.; th0 = 302.; dth_inv = 2.; gam_ft = 4.; qv0 = 0.006
      u0 = 4.; v0 = 1.; du = 1.0; dv = 0.2
      firesize(1) = 2.0e5; mean_fct(1) = 0.6
    case (2)   ! boreal forest, moist, large fire: cloud, rain and ice
      zi = 2000.; th0 = 296.; dth_inv = 1.; gam_ft = 3.5; qv0 = 0.013
      u0 = 2.; v0 = 0.; du = 0.5; dv = 0.5
      firesize(2) = 8.0e5; mean_fct(2) = 1.0
    case (3)   ! savanna under a strong inversion
      zi = 800.; th0 = 300.; dth_inv = 6.; gam_ft = 6.; qv0 = 0.010
      u0 = 6.; v0 = -2.; du = 0.8; dv = 0.
      firesize(3) = 5.0e4; mean_fct(3) = 0.4
    case (4)   ! grassland, calm
      zi = 1200.; th0 = 298.; dth_inv = 1.5; gam_ft = 4.; qv0 = 0.008
      u0 = 0.; v0 = 0.; du = 0.; dv = 0.
      firesize(4) = 1.0e5; mean_fct(4) = 0.8
    case (5)   ! all four biomes in one sheared column
      zi = 1800.; th0 = 303.; dth_inv = 2.; gam_ft = 3.8; qv0 = 0.011
      u0 = 3.; v0 = 3.; du = 2.; dv = -1.
      firesize = (/ 1.5e5, 3.0e5, 6.0e4, 2.0e4 /)
      mean_fct = (/ 0.3, 0.2, 0.25, 0.15 /)
    case (6)   ! a fire too small to lift its plume far
      zi = 600.; th0 = 290.; dth_inv = 4.; gam_ft = 7.; qv0 = 0.004
      u0 = 8.; v0 = 0.; du = 1.5; dv = 0.
      firesize(3) = 200.; mean_fct(3) = 1.0
    case (7)   ! very moist, deep convective column, tropical forest
      zi = 2500.; th0 = 300.; dth_inv = 0.5; gam_ft = 3.2; qv0 = 0.017
      u0 = 1.; v0 = 1.; du = 0.3; dv = 0.3
      firesize(1) = 1.0e6; mean_fct(1) = 1.0
    case (8)   ! weakly stable, windy, boreal and grassland together
      zi = 1000.; th0 = 288.; dth_inv = 3.; gam_ft = 5.; qv0 = 0.005
      u0 = 10.; v0 = 4.; du = 1.; dv = 0.5
      firesize(2) = 4.0e5; mean_fct(2) = 0.7
      firesize(4) = 5.0e4; mean_fct(4) = 0.3
    end select

    ! theta, then the exner function by hydrostatic integration.
    do k = 1, m1
      z = zt_rams(k)
      if (z <= zi) then
        th = th0
      else if (z <= zi + 200.) then
        th = th0 + dth_inv * (z - zi) / 200.
      else if (z <= 12000.) then
        th = th0 + dth_inv + gam_ft * 1.e-3 * (z - zi - 200.)
      else
        th = th0 + dth_inv + gam_ft * 1.e-3 * (12000. - zi - 200.) + &
             15.e-3 * (z - 12000.)
      end if
      theta(k,1,1) = th
    end do
    pik = cp
    do k = 1, m1
      if (k == 1) then
        dpik = g * zt_rams(1) / theta(1,1,1)
      else
        dpik = g * (zt_rams(k) - zt_rams(k - 1)) &
               / (0.5 * (theta(k,1,1) + theta(k - 1,1,1)))
      end if
      pik = pik - dpik
      pp(k,1,1) = pik
      p = 1.e5 * (pik / cp)**(1. / rcp)
      tk = theta(k,1,1) * pik / cp
      es = 611.2 * exp(17.67 * (tk - 273.15) / (tk - 29.65))
      qs = 0.622 * es / (p - es)
      rv(k,1,1) = min(qv0 * exp(-zt_rams(k) / 2500.), 0.9 * qs)
      up(k,1,1) = u0 + du * 1.e-3 * zt_rams(k)
      vp(k,1,1) = v0 + dv * 1.e-3 * zt_rams(k)
    end do
    wp = 0.
    dn0 = 1.
  end subroutine make_case

end program run_plumerise
