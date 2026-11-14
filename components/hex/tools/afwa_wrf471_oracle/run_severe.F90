! Drives WRF v4.7.1's byte-unmodified severe_wx_diagnostics (phys/module_diag_afwa.F, after the driver) on the mesh
! columns of columns.bin and severe_columns.bin (both written by make_columns.py) for three steps and writes the
! severe package's fields after each step.  One mesh cell is one WRF column (i = cell, j = 1); the memory has a
! one-cell zero halo on every side (ims = 0 .. ime = nc + 1, jms = 0 .. jme = 2), as WRF's tiles have, because the
! tornado duration reads tornado_mask_prev(i-1:i+1, j-1:j+1).
!
! The mass-point state is built as run_afwa.F90 builds it.  The glue around the call is afwa_diagnostics_driver's own
! text (module_diag_afwa.F:328-414 the gust and 0-2000 ft shear maxima and their reset, :440-460 the reset of the
! severe maxima after a history dump, :462-466 do_buoy_calc), written out with the same expressions.  The steps:
!
!   step 1: after a history dump (reset), first time step (do_buoy_calc)
!   step 2: neither
!   step 3: after a history dump (reset), an output time step (do_buoy_calc)
!
! every step on the same columns.  midrh_min and midrh_min_old start at 0 (the hex module's and the WRF route's
! zeroed carriers).
!
!   run_severe columns.bin severe_columns.bin severe.bin
! severe.bin: float32 [3][nfield][nc] in the order of the write statement below.
program run_severe
  use diag_functions
  use module_diag_afwa_cut
  implicit none
  integer :: nc, nz, i, j, k, step, unit_in, unit_out, kme
  real :: ccn, melt, dt, wind_vel, prate_mm_per_hr, u1km, v1km, factor, ublend, vblend, wind_blend, u2000, v2000, us, vs
  logical :: is_target_level, reset, do_buoy_calc
  real, parameter :: g = 9.81
  real, allocatable :: zgrid(:,:), rho_dry(:,:), tt(:,:), pp(:,:), qv(:,:), qc(:,:), qi(:,:), qr(:,:), qs(:,:), qg(:,:), uu(:,:), vv(:,:)
  real, allocatable :: ww(:,:), uh_in(:)
  real, allocatable :: t_phy(:,:,:), ptot(:,:,:), qvapr(:,:,:), qcloud(:,:,:), qice(:,:,:), qrain(:,:,:), qsnow(:,:,:), qgrpl(:,:,:), ngrpl(:,:,:)
  real, allocatable :: u_phy(:,:,:), v_phy(:,:,:), rho(:,:,:), dz8w(:,:,:), z(:,:,:), z_at_w(:,:,:), rh(:,:,:), zagl(:,:,:), w_2(:,:,:)
  real, allocatable :: psfc(:), t2in(:), q2in(:), u10in(:), v10in(:), swdown(:), raincv(:), rainncv(:), rainc(:), rainnc(:)
  real, allocatable :: u10(:,:), v10(:,:), t2(:,:), ht(:,:), uh(:,:), afwa_precip(:,:), pwat2d(:,:)
  real, allocatable :: wspd10max(:,:), afwa_llws(:,:), llws(:,:), w_up_max(:,:), w_dn_max(:,:), up_heli_max(:,:), tcoli_max(:,:)
  real, allocatable :: grpl_flx_max(:,:), midrh_min(:,:), midrh_min_old(:,:), afwa_hail(:,:), afwa_tornado(:,:)
  real, allocatable :: tornado_mask(:,:), tornado_dur(:,:), cape(:,:), cin(:,:), zlfc(:,:), plfc(:,:), lidx(:,:)
  character(len=512) :: infile, severefile, outfile
  call get_command_argument(1, infile)
  call get_command_argument(2, severefile)
  call get_command_argument(3, outfile)
  open(newunit=unit_in, file=trim(infile), access='stream', form='unformatted', status='old')
  read(unit_in) nc, nz
  read(unit_in) ccn, melt
  allocate(zgrid(nc,nz+1), rho_dry(nc,nz), tt(nc,nz), pp(nc,nz), qv(nc,nz), qc(nc,nz), qi(nc,nz), qr(nc,nz), qs(nc,nz), qg(nc,nz), uu(nc,nz), vv(nc,nz))
  read(unit_in) zgrid
  read(unit_in) rho_dry, tt, pp, qv, qc, qi, qr, qs, qg, uu, vv
  allocate(psfc(nc), t2in(nc), q2in(nc), u10in(nc), v10in(nc), swdown(nc), raincv(nc), rainncv(nc), rainc(nc), rainnc(nc))
  read(unit_in) psfc, t2in, q2in, u10in, v10in, swdown, raincv, rainncv, rainc, rainnc
  close(unit_in)
  allocate(ww(nc,nz+1), uh_in(nc))
  open(newunit=unit_in, file=trim(severefile), access='stream', form='unformatted', status='old')
  read(unit_in) dt
  read(unit_in) ww, uh_in
  close(unit_in)
  kme = nz + 1
  allocate(t_phy(0:nc+1,kme,0:2), ptot(0:nc+1,kme,0:2), qvapr(0:nc+1,kme,0:2), qcloud(0:nc+1,kme,0:2), qice(0:nc+1,kme,0:2))
  allocate(qrain(0:nc+1,kme,0:2), qsnow(0:nc+1,kme,0:2), qgrpl(0:nc+1,kme,0:2), ngrpl(0:nc+1,kme,0:2))
  allocate(u_phy(0:nc+1,kme,0:2), v_phy(0:nc+1,kme,0:2), rho(0:nc+1,kme,0:2), dz8w(0:nc+1,kme,0:2), z(0:nc+1,kme,0:2))
  allocate(z_at_w(0:nc+1,kme,0:2), rh(0:nc+1,kme,0:2), zagl(0:nc+1,kme,0:2), w_2(0:nc+1,kme,0:2))
  ! the halo and the extra mass row: a finite, inert column (no level reaches 3500 m above ground there)
  t_phy = 250.; ptot = 50000.; qvapr = 0.; qcloud = 0.; qice = 0.; qrain = 0.; qsnow = 0.; qgrpl = 0.; ngrpl = 0.
  u_phy = 0.; v_phy = 0.; rho = 1.; dz8w = 0.; z = 0.; z_at_w = 0.; rh = 0.; zagl = 0.; w_2 = 0.
  allocate(u10(0:nc+1,0:2), v10(0:nc+1,0:2), t2(0:nc+1,0:2), ht(0:nc+1,0:2), uh(0:nc+1,0:2), afwa_precip(0:nc+1,0:2), pwat2d(0:nc+1,0:2))
  allocate(wspd10max(0:nc+1,0:2), afwa_llws(0:nc+1,0:2), llws(0:nc+1,0:2), w_up_max(0:nc+1,0:2), w_dn_max(0:nc+1,0:2))
  allocate(up_heli_max(0:nc+1,0:2), tcoli_max(0:nc+1,0:2), grpl_flx_max(0:nc+1,0:2), midrh_min(0:nc+1,0:2), midrh_min_old(0:nc+1,0:2))
  allocate(afwa_hail(0:nc+1,0:2), afwa_tornado(0:nc+1,0:2), tornado_mask(0:nc+1,0:2), tornado_dur(0:nc+1,0:2))
  allocate(cape(0:nc+1,0:2), cin(0:nc+1,0:2), zlfc(0:nc+1,0:2), plfc(0:nc+1,0:2), lidx(0:nc+1,0:2))
  u10 = 0.; v10 = 0.; t2 = 0.; ht = 0.; uh = 0.; afwa_precip = 0.; pwat2d = 0.
  wspd10max = 0.; afwa_llws = 0.; llws = 0.; w_up_max = 0.; w_dn_max = 0.; up_heli_max = 0.; tcoli_max = 0.
  grpl_flx_max = 0.; midrh_min = 0.; midrh_min_old = 0.; afwa_hail = 0.; afwa_tornado = 0.; tornado_mask = 0.; tornado_dur = 0.
  cape = 0.; cin = 0.; zlfc = 0.; plfc = 0.; lidx = 0.
  j = 1
  do i = 1, nc
    do k = 1, nz + 1
      z_at_w(i,k,j) = (0. + g * zgrid(i,k)) / g
    end do
    do k = 1, nz
      dz8w(i,k,j) = z_at_w(i,k+1,j) - z_at_w(i,k,j)
      z(i,k,j) = 0.5 * (z_at_w(i,k,j) + z_at_w(i,k+1,j))
      rho(i,k,j) = 1. / (1. / rho_dry(i,k))
      t_phy(i,k,j) = tt(i,k); ptot(i,k,j) = pp(i,k); qvapr(i,k,j) = qv(i,k); qcloud(i,k,j) = qc(i,k); qice(i,k,j) = qi(i,k)
      qrain(i,k,j) = qr(i,k); qsnow(i,k,j) = qs(i,k); qgrpl(i,k,j) = qg(i,k); u_phy(i,k,j) = uu(i,k); v_phy(i,k,j) = vv(i,k)
      w_2(i,k,j) = ww(i,k)
    end do
    w_2(i,kme,j) = ww(i,nz+1)
    ht(i,j) = z_at_w(i,1,j)
    u10(i,j) = u10in(i); v10(i,j) = v10in(i); t2(i,j) = t2in(i); uh(i,j) = uh_in(i)
    afwa_precip(i,j) = raincv(i) + rainncv(i)
  end do
  ! the driver's zagl and rh, over the whole memory
  do i = 0, nc + 1
    do k = 1, kme
      do j = 0, 2
        zagl(i,k,j) = z(i,k,j) - ht(i,j)
        rh(i,k,j) = calc_rh(ptot(i,k,j), t_phy(i,k,j), qvapr(i,k,j))
      end do
    end do
  end do
  open(newunit=unit_out, file=trim(outfile), access='stream', form='unformatted', status='replace')
  do step = 1, 3
    reset = step /= 2
    do_buoy_calc = step /= 2
    ! module_diag_afwa.F:328-339
    if (reset) then
      wspd10max = 0.
      afwa_llws = 0.
    end if
    ! :350-391, the 10 m gust maximum blended with the 1 km wind under heavy precipitation
    do j = 0, 2
      do i = 0, nc + 1
        wind_vel = uv_wind(u10(i,j), v10(i,j))
        prate_mm_per_hr = (afwa_precip(i,j) / dt) * 3600.
        if (prate_mm_per_hr .gt. 50.) then
          is_target_level = .false.
          do k = 1, kme
            if ((zagl(i,k,j) >= 1000.) .and. (.not. is_target_level) .and. (k .ne. 1)) then
              is_target_level = .true.
              u1km = u_phy(i,k-1,j) + (1000. - (zagl(i,k-1,j))) * ((u_phy(i,k,j) - u_phy(i,k-1,j))/(zagl(i,k,j)))
              v1km = v_phy(i,k-1,j) + (1000. - (zagl(i,k-1,j))) * ((v_phy(i,k,j) - v_phy(i,k-1,j))/(zagl(i,k,j)))
              exit
            end if
          end do
          factor = max(((150. - prate_mm_per_hr) / 100.), 0.)
          ublend = u10(i,j) * factor + u1km * (1. - factor)
          vblend = v10(i,j) * factor + v1km * (1. - factor)
          wind_blend = uv_wind(ublend, vblend)
          if (wind_blend .gt. wind_vel) then
            wind_vel = wind_blend
          end if
        end if
        if (wind_vel .gt. wspd10max(i,j)) then
          wspd10max(i,j) = wind_vel
        end if
      end do
    end do
    ! :395-416, the 0-2000 ft shear
    j = 1
    do i = 1, nc
      is_target_level = .false.
      do k = 1, kme
        if ((zagl(i,k,j) >= 609.6) .and. (.not. is_target_level) .and. (k .ne. 1)) then
          is_target_level = .true.
          u2000 = u_phy(i,k-1,j) + (609.6 - (zagl(i,k-1,j))) * ((u_phy(i,k,j) - u_phy(i,k-1,j))/(zagl(i,k,j)))
          v2000 = v_phy(i,k-1,j) + (609.6 - (zagl(i,k-1,j))) * ((v_phy(i,k,j) - v_phy(i,k-1,j))/(zagl(i,k,j)))
          us = u2000 - u10(i,j)
          vs = v2000 - v10(i,j)
          llws(i,j) = uv_wind(us, vs)
          if (llws(i,j) .gt. afwa_llws(i,j)) then
            afwa_llws(i,j) = llws(i,j)
          end if
          exit
        end if
      end do
    end do
    ! :447-460
    if (reset) then
      w_up_max = 0.; w_dn_max = 0.; tcoli_max = 0.; grpl_flx_max = 0.; up_heli_max = 0.; afwa_tornado = 0.
      midrh_min_old = midrh_min
      midrh_min = 999.
      afwa_hail = 0.
    end if
    call severe_wx_diagnostics(wspd10max, w_up_max, w_dn_max, up_heli_max, tcoli_max, midrh_min_old, midrh_min, &
                               afwa_hail, cape, cin, zlfc, plfc, lidx, llws, afwa_tornado, grpl_flx_max, u10, v10, &
                               w_2, uh, t_phy, t2, z, ht, tornado_mask, tornado_dur, dt, pwat2d, u_phy, v_phy, ptot, &
                               qice, qsnow, qgrpl, ngrpl, qvapr, qrain, qcloud, rho, dz8w, rh, do_buoy_calc, &
                               0, nc + 1, 0, 2, 1, kme, 1, nc, 1, 1, 1, nz, 1, 1, 1, nc)
    write(unit_out) wspd10max(1:nc,1), afwa_llws(1:nc,1), w_up_max(1:nc,1), w_dn_max(1:nc,1), tcoli_max(1:nc,1)
    write(unit_out) grpl_flx_max(1:nc,1), afwa_hail(1:nc,1), afwa_tornado(1:nc,1)
    write(unit_out) cape(1:nc,1), cin(1:nc,1), zlfc(1:nc,1), plfc(1:nc,1), lidx(1:nc,1)
    write(unit_out) midrh_min(1:nc,1), midrh_min_old(1:nc,1), tornado_mask(1:nc,1), tornado_dur(1:nc,1)
  end do
  close(unit_out)
end program run_severe
