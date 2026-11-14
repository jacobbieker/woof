! Drives WRF v4.7.1's byte-unmodified AFWA routines (phys/module_diag_afwa.F, everything after the driver, and
! phys/module_diag_functions.F) on the mesh columns of columns.bin (written by make_columns.py) and writes the AFWA
! fields.  One mesh cell is one WRF column (i = cell, j = 1).
!
! The mass-point state is built as afwa_diagnostics_driver's callers build it: PH = 0, PHB = g * zgrid,
! z_at_w = (ph + phb) / g, dz8w = z_at_w(k+1) - z_at_w(k), z = 0.5 * (z_at_w(k) + z_at_w(k+1)) (module_big_step_utilities_em.F
! :4877-4893), alt = 1 / rho_dry, rho = 1 / alt.  The glue between the routines (rh, ptot, rh2m, tv2m, wind10m, the
! 20 m and 125 m interpolations, the buoyancy calls and the AGL / lifted-index fix-ups) is the driver's own text
! (module_diag_afwa.F:262-290, 584-720, 884-947), written out here with the same expressions.
!
!   run_afwa columns.bin out.bin
! out.bin: float32 [nc] in the order of FIELDS below.
program run_afwa
  use diag_functions
  use module_diag_afwa_cut
  implicit none
  integer, parameter :: nfield = 26
  integer :: nc, nz, i, j, k, unit_in, unit_out, ostat, kde, kme
  integer :: ids, ide, jds, jde, kds, ims, ime, jms, jme, kms, ips, ipe, jps, jpe, kps, kpe
  real :: ccn, melt, zlfc_msl, dum1, dum2, dum3
  real, parameter :: g = 9.81
  real, allocatable :: zgrid(:,:), rho_dry(:,:), tt(:,:), pp(:,:), qv(:,:), qc(:,:), qi(:,:), qr(:,:), qs(:,:), qg(:,:), uu(:,:), vv(:,:)
  real, allocatable :: t_phy(:,:,:), ptot(:,:,:), qvapr(:,:,:), qcloud(:,:,:), qice(:,:,:), qrain(:,:,:), qsnow(:,:,:), qgrpl(:,:,:)
  real, allocatable :: u_phy(:,:,:), v_phy(:,:,:), rho(:,:,:), dz8w(:,:,:), z(:,:,:), z_at_w(:,:,:), alt(:,:,:), rh(:,:,:), zagl(:,:,:), z_e(:,:,:)
  real, allocatable :: psfc(:,:), t2(:,:), q2(:,:), u10(:,:), v10(:,:), swdown(:,:), raincv(:,:), rainncv(:,:), rainc(:,:), rainnc(:,:), ht(:,:)
  real, allocatable :: afwa_precip(:,:), afwa_totprecip(:,:), afwa_pwat(:,:), afwa_mslp(:,:)
  real, allocatable :: heatidx(:,:), wchill(:,:), fits(:,:), rain(:,:), snow(:,:), ice(:,:), frz(:,:), snowfall(:,:)
  real, allocatable :: vil(:,:), radarvil(:,:), refd(:,:), refd_com(:,:), echotop(:,:), vis(:,:), vis_dust(:,:), vis_alpha(:,:)
  real, allocatable :: cape(:,:), cin(:,:), zlfc(:,:), plfc(:,:), lidx(:,:), cape_mu(:,:), cin_mu(:,:)
  real, allocatable :: rh2m(:,:), rh20m(:,:), tv2m(:,:), tv20m(:,:), wind10m(:,:), wind125m(:,:), dustc(:,:,:)
  real, allocatable :: q1(:,:), q2d(:,:), q3(:,:), q4(:,:), q5(:,:), rho1(:,:)
  character(len=512) :: infile, outfile
  call get_command_argument(1, infile)
  call get_command_argument(2, outfile)
  open(newunit=unit_in, file=trim(infile), access='stream', form='unformatted', status='old')
  read(unit_in) nc, nz
  read(unit_in) ccn, melt
  kme = nz + 1; kde = nz + 1
  ids = 1; ide = nc + 1; jds = 1; jde = 2; kds = 1
  ims = 1; ime = nc; jms = 1; jme = 1; kms = 1
  ips = 1; ipe = nc; jps = 1; jpe = 1; kps = 1; kpe = nz
  allocate(zgrid(nc,nz+1), rho_dry(nc,nz), tt(nc,nz), pp(nc,nz), qv(nc,nz), qc(nc,nz), qi(nc,nz), qr(nc,nz), qs(nc,nz), qg(nc,nz), uu(nc,nz), vv(nc,nz))
  read(unit_in) zgrid
  read(unit_in) rho_dry, tt, pp, qv, qc, qi, qr, qs, qg, uu, vv
  allocate(psfc(nc,1), t2(nc,1), q2(nc,1), u10(nc,1), v10(nc,1), swdown(nc,1), raincv(nc,1), rainncv(nc,1), rainc(nc,1), rainnc(nc,1), ht(nc,1))
  read(unit_in) psfc(:,1), t2(:,1), q2(:,1), u10(:,1), v10(:,1), swdown(:,1), raincv(:,1), rainncv(:,1), rainc(:,1), rainnc(:,1)
  close(unit_in)
  ! (numpy C order (level, cell): the cell index runs fastest, which is Fortran's (nc, nz) order)
  allocate(t_phy(nc,kme,1), ptot(nc,kme,1), qvapr(nc,kme,1), qcloud(nc,kme,1), qice(nc,kme,1), qrain(nc,kme,1), qsnow(nc,kme,1), qgrpl(nc,kme,1))
  allocate(u_phy(nc,kme,1), v_phy(nc,kme,1), rho(nc,kme,1), dz8w(nc,kme,1), z(nc,kme,1), z_at_w(nc,kme,1), alt(nc,kme,1), rh(nc,kme,1), zagl(nc,kme,1), z_e(nc,kme,1))
  t_phy = 0.; ptot = 1.; qvapr = 0.; qcloud = 0.; qice = 0.; qrain = 0.; qsnow = 0.; qgrpl = 0.; u_phy = 0.; v_phy = 0.
  rho = 1.; dz8w = 0.; z = 0.; z_at_w = 0.; alt = 1.; rh = 0.; zagl = 0.; z_e = 0.
  do i = 1, nc
    do k = 1, nz + 1
      z_at_w(i,k,1) = (0. + g * zgrid(i,k)) / g
    end do
    do k = 1, nz
      dz8w(i,k,1) = z_at_w(i,k+1,1) - z_at_w(i,k,1)
      z(i,k,1) = 0.5 * (z_at_w(i,k,1) + z_at_w(i,k+1,1))
      alt(i,k,1) = 1. / rho_dry(i,k)
      rho(i,k,1) = 1. / alt(i,k,1)
      t_phy(i,k,1) = tt(i,k); ptot(i,k,1) = pp(i,k); qvapr(i,k,1) = qv(i,k); qcloud(i,k,1) = qc(i,k); qice(i,k,1) = qi(i,k)
      qrain(i,k,1) = qr(i,k); qsnow(i,k,1) = qs(i,k); qgrpl(i,k,1) = qg(i,k); u_phy(i,k,1) = uu(i,k); v_phy(i,k,1) = vv(i,k)
    end do
    ht(i,1) = z_at_w(i,1,1)
  end do
  do i = 1, nc
    do k = 1, nz
      zagl(i,k,1) = z(i,k,1) - ht(i,1)
      rh(i,k,1) = calc_rh(ptot(i,k,1), t_phy(i,k,1), qvapr(i,k,1))
    end do
  end do
  allocate(afwa_precip(nc,1), afwa_totprecip(nc,1), afwa_pwat(nc,1), afwa_mslp(nc,1))
  allocate(heatidx(nc,1), wchill(nc,1), fits(nc,1), rain(nc,1), snow(nc,1), ice(nc,1), frz(nc,1), snowfall(nc,1))
  allocate(vil(nc,1), radarvil(nc,1), refd(nc,1), refd_com(nc,1), echotop(nc,1), vis(nc,1), vis_dust(nc,1), vis_alpha(nc,1))
  allocate(cape(nc,1), cin(nc,1), zlfc(nc,1), plfc(nc,1), lidx(nc,1), cape_mu(nc,1), cin_mu(nc,1))
  allocate(rh2m(nc,1), rh20m(nc,1), tv2m(nc,1), tv20m(nc,1), wind10m(nc,1), wind125m(nc,1), dustc(nc,1,5))
  allocate(q1(nc,1), q2d(nc,1), q3(nc,1), q4(nc,1), q5(nc,1), rho1(nc,1))
  heatidx = 0.; wchill = 0.; fits = 0.; rain = 0.; snow = 0.; ice = 0.; frz = 0.; snowfall = 0.
  vil = 0.; radarvil = 0.; refd = 0.; refd_com = 0.; echotop = 0.; vis = 0.; vis_dust = 0.; vis_alpha = 0.
  cape = 0.; cin = 0.; zlfc = 0.; plfc = 0.; lidx = 0.; cape_mu = 0.; cin_mu = 0.; dustc = 0.

  ! afwa_diagnostics_driver: time-step precipitation, precipitable water
  do i = 1, nc
    afwa_precip(i,1) = raincv(i,1) + rainncv(i,1)
    afwa_totprecip(i,1) = rainc(i,1) + rainnc(i,1)
    afwa_pwat(i,1) = Pwat(nz, qvapr(i,1:nz,1), qcloud(i,1:nz,1), dz8w(i,1:nz,1), rho(i,1:nz,1))
  end do
  ! precipitation type (grid%itimestep == 1: the buckets start at zero)
  call precip_type_diagnostics(t_phy, t2, rh, z, dz8w, ht, afwa_precip, swdown, rain, snow, ice, frz, snowfall, ccn, melt, &
                               ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ips, ipe, jps, jpe, kps, kpe)
  ! output timestep: MSLP, 10 m wind, 2 m RH and Tv
  do i = 1, nc
    afwa_mslp(i,1) = MSLP(ht(i,1), psfc(i,1), z(i,1,1), qvapr(i,1,1), t_phy(i,1,1))
    wind10m(i,1) = uv_wind(u10(i,1), v10(i,1))
    rh2m(i,1) = calc_rh(psfc(i,1), t2(i,1), q2(i,1))
    tv2m(i,1) = t2(i,1) * (1 + 0.61 * q2(i,1))
  end do
  do i = 1, nc
    ostat = Buoyancy(nz, t_phy(i,1:nz,1), rh(i,1:nz,1), ptot(i,1:nz,1), z(i,1:nz,1), 1, cape(i,1), cin(i,1), zlfc_msl, plfc(i,1), lidx(i,1), 3)
    if (zlfc_msl .ge. ht(i,1)) then
      zlfc(i,1) = zlfc_msl - ht(i,1)
    else
      zlfc(i,1) = -1.
    end if
    if (lidx(i,1) .ne. 999.) lidx(i,1) = lidx(i,1) + 273.15
    ostat = Buoyancy(nz, t_phy(i,1:nz,1), rh(i,1:nz,1), ptot(i,1:nz,1), z(i,1:nz,1), 1, cape_mu(i,1), cin_mu(i,1), dum1, dum2, dum3, 1)
  end do
  call thermal_diagnostics(t2, psfc, rh2m, wind10m, heatidx, wchill, fits, &
                           ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ips, ipe, jps, jpe, kps, kpe)
  call wrf_dbzcalc(rho, t_phy, qrain, qsnow, qgrpl, z_e, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ips, ipe, jps, jpe, kps, kpe)
  call radar_diagnostics(refd, refd_com, echotop, z, z_e, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ips, ipe, jps, jpe, kps, kpe)
  call vert_int_liquid_diagnostics(vil, radarvil, t_phy, qrain, qsnow, qgrpl, z_e, dz8w, rho, &
                                   ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ips, ipe, jps, jpe, kps, kpe)
  ! visibility: the driver's 20 m interpolation and 125 m wind, then vis_diagnostics
  do i = 1, nc
    j = 1
    tv20m(i,j) = -999.
    rh20m(i,j) = -999.
    do k = 1, min(nz, kde-1)
      if (tv20m(i,j) .eq. -999. .and. zagl(i,k,j) .ge. 20.) then
        if (k .eq. 1) then
          tv20m(i,j) = tv2m(i,j) + (20. - 2.) / (zagl(i,k,j) - 2.) * (t_phy(i,k,j) * (1 + 0.61 * qvapr(i,k,j)) - tv2m(i,j))
          rh20m(i,j) = rh2m(i,j) + (20. - 2.) / (zagl(i,k,j) - 2.) * (rh(i,k,j) - rh2m(i,j))
        else
          tv20m(i,j) = t_phy(i,k-1,j) * (1 + 0.61 * qvapr(i,k-1,j)) + ((20. - zagl(i,k-1,j)) / (zagl(i,k,j) - zagl(i,k-1,j))) * &
                       (t_phy(i,k,j) * (1 + 0.61 * qvapr(i,k,j)) - t_phy(i,k-1,j) * (1 + 0.61 * qvapr(i,k-1,j)))
          rh20m(i,j) = rh(i,k-1,j) + ((20. - zagl(i,k-1,j)) / (zagl(i,k,j) - zagl(i,k-1,j))) * (rh(i,k,j) - rh(i,k-1,j))
        end if
      end if
    end do
    wind125m(i,j) = -999.
    do k = 1, min(nz, kde-1)
      if (wind125m(i,j) .eq. -999. .and. zagl(i,k,j) .ge. 125.) then
        if (k .eq. 1) then
          wind125m(i,j) = uv_wind(u_phy(i,k,j), v_phy(i,k,j))
        else
          wind125m(i,j) = uv_wind(u_phy(i,k-1,j), v_phy(i,k-1,j)) + ((125. - zagl(i,k-1,j)) / (zagl(i,k,j) - zagl(i,k-1,j))) * &
                          (uv_wind(u_phy(i,k,j), v_phy(i,k,j)) - uv_wind(u_phy(i,k-1,j), v_phy(i,k-1,j)))
        end if
      end if
    end do
  end do
  q1(:,1) = qcloud(:,1,1); q2d(:,1) = qrain(:,1,1); q3(:,1) = qice(:,1,1); q4(:,1) = qsnow(:,1,1); q5(:,1) = qgrpl(:,1,1)
  rho1(:,1) = rho(:,1,1)
  call vis_diagnostics(q1, q2d, q3, q4, q5, rho1, wind10m, wind125m, afwa_pwat, q2, rh2m, rh20m, tv2m, tv20m, dustc, &
                       vis, vis_dust, vis_alpha, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ips, ipe, jps, jpe, kps, kpe)
  open(newunit=unit_out, file=trim(outfile), access='stream', form='unformatted', status='replace')
  write(unit_out) afwa_mslp(:,1), afwa_pwat(:,1), afwa_totprecip(:,1), afwa_precip(:,1)
  write(unit_out) heatidx(:,1), wchill(:,1), fits(:,1)
  write(unit_out) rain(:,1), snow(:,1), ice(:,1), frz(:,1), snowfall(:,1)
  write(unit_out) vil(:,1), radarvil(:,1)
  write(unit_out) refd(:,1), refd_com(:,1), echotop(:,1)
  write(unit_out) vis(:,1), vis_dust(:,1), vis_alpha(:,1)
  write(unit_out) cape(:,1), cin(:,1), zlfc(:,1), plfc(:,1), lidx(:,1), cape_mu(:,1), cin_mu(:,1)
  close(unit_out)
end program run_afwa
