! Column driver for WRF v4.7.1 module_diag_solar.F solar_diag, called one
! column at a time with the dimensions module_diagnostics_driver.F:1184-1212
! gives it: kms:kme = 1:nz+1 (memory, w-staggered), kts = 1, kte = nz (the
! mass-level count, min(k_end, kde-1)).
!
! Input  columns.bin (stream, little endian): int ncol; per column
!   int nz, flags(7) = f_qv f_qc f_qi f_qs has_reqc has_reqi has_reqs,
!   real dt, coszen, swdnb, swdnt, prior,
!   real rho, dz8w, qv, qc, qi, qs, qc_tot, qi_tot, re_cloud, re_ice,
!        re_snow, cldfra (nz each), ph, phb (nz+1 each)
! A species that is absent has f_qx = .false. and P_QX = 1 (WRF's index of
! an unallocated moist species, below PARAM_FIRST_SCALAR = 2); present,
! P_QX = 3.  ``prior`` is the value every 2-D output holds before the call
! (GHI_ACCUM's running sum, and what WRF leaves in an output it does not
! assign).
! Output solar_diag.out.bin: per column the 26 outputs in the order of
! woof/core/solar_diagnostics.py SOLAR_OUTPUTS.
program run_solar_diag
  use module_diag_solar, only: solar_diag
  implicit none
  integer :: ncol, nz, c, u, v, flags(7), pqc, pqi, pqs
  real :: dt, prior
  real :: coszen(1,1), swdnb(1,1), swdnt(1,1)
  real, allocatable :: rho(:,:,:), dz8w(:,:,:), qv(:,:,:), qc(:,:,:), qi(:,:,:), &
       qs(:,:,:), qc_tot(:,:,:), qi_tot(:,:,:), re_cloud(:,:,:), re_ice(:,:,:), &
       re_snow(:,:,:), cldfra(:,:,:), ph(:,:,:), phb(:,:,:)
  real :: clrnidx(1,1), sza(1,1), ghi_accum(1,1), cldfrac2d(1,1), wvp(1,1), &
       lwp(1,1), iwp(1,1), swp(1,1), wp_sum(1,1), lwp_tot(1,1), iwp_tot(1,1), &
       wp_tot_sum(1,1), re_qc(1,1), re_qi(1,1), re_qs(1,1), re_qc_tot(1,1), &
       re_qi_tot(1,1), tau_qc(1,1), tau_qi(1,1), tau_qs(1,1), tau_qc_tot(1,1), &
       tau_qi_tot(1,1), cbase(1,1), ctop(1,1), cbase_tot(1,1), ctop_tot(1,1)
  integer :: i_start(1), i_end(1), j_start(1), j_end(1)

  open(newunit=u, file='columns.bin', access='stream', form='unformatted', &
       status='old', convert='little_endian')
  open(newunit=v, file='solar_diag.out.bin', access='stream', form='unformatted', &
       status='replace', convert='little_endian')
  read(u) ncol
  i_start = 1; i_end = 1; j_start = 1; j_end = 1
  do c = 1, ncol
    read(u) nz, flags, dt, coszen, swdnb, swdnt, prior
    allocate(rho(1,nz+1,1), dz8w(1,nz+1,1), qv(1,nz+1,1), qc(1,nz+1,1), &
             qi(1,nz+1,1), qs(1,nz+1,1), qc_tot(1,nz+1,1), qi_tot(1,nz+1,1), &
             re_cloud(1,nz+1,1), re_ice(1,nz+1,1), re_snow(1,nz+1,1), &
             cldfra(1,nz+1,1), ph(1,nz+1,1), phb(1,nz+1,1))
    ! the memory level above kte: benign values the routine reads into
    ! rhodz(kme) and never uses
    rho = 1.; dz8w = 1.; qv = 0.; qc = 0.; qi = 0.; qs = 0.; qc_tot = 0.
    qi_tot = 0.; re_cloud = 0.; re_ice = 0.; re_snow = 0.; cldfra = 0.
    read(u) rho(1,1:nz,1), dz8w(1,1:nz,1), qv(1,1:nz,1), qc(1,1:nz,1), &
            qi(1,1:nz,1), qs(1,1:nz,1), qc_tot(1,1:nz,1), qi_tot(1,1:nz,1), &
            re_cloud(1,1:nz,1), re_ice(1,1:nz,1), re_snow(1,1:nz,1), &
            cldfra(1,1:nz,1), ph(1,:,1), phb(1,:,1)
    clrnidx = prior; sza = prior; ghi_accum = prior; cldfrac2d = prior
    wvp = prior; lwp = prior; iwp = prior; swp = prior; wp_sum = prior
    lwp_tot = prior; iwp_tot = prior; wp_tot_sum = prior; re_qc = prior
    re_qi = prior; re_qs = prior; re_qc_tot = prior; re_qi_tot = prior
    tau_qc = prior; tau_qi = prior; tau_qs = prior; tau_qc_tot = prior
    tau_qi_tot = prior; cbase = prior; ctop = prior; cbase_tot = prior
    ctop_tot = prior
    pqc = merge(3, 1, flags(2) == 1)
    pqi = merge(4, 1, flags(3) == 1)
    pqs = merge(5, 1, flags(4) == 1)
    call solar_diag(dt, rho, dz8w, ph, phb, cldfra, coszen, swdnb, swdnt, &
         2, pqc, pqi, pqs, qv, qc, qi, qs, &
         qc_tot, qi_tot, flags(5), flags(6), flags(7), flags(1) == 1, &
         flags(2) == 1, flags(3) == 1, flags(4) == 1, &
         re_cloud, re_ice, re_snow, clrnidx, sza, ghi_accum, cldfrac2d, wvp, &
         lwp, iwp, swp, wp_sum, lwp_tot, iwp_tot, wp_tot_sum, re_qc, re_qi, &
         re_qs, re_qc_tot, re_qi_tot, tau_qc, tau_qi, tau_qs, tau_qc_tot, &
         tau_qi_tot, cbase, ctop, cbase_tot, ctop_tot, &
         1, 2, 1, 2, 1, nz+1, 1, 1, 1, 1, 1, nz+1, 1, 1, 1, 1, 1, nz+1, &
         i_start, i_end, j_start, j_end, 1, nz, 1)
    write(v) clrnidx, sza, ghi_accum, cldfrac2d, wvp, lwp, iwp, swp, wp_sum, &
         lwp_tot, iwp_tot, wp_tot_sum, re_qc, re_qi, re_qs, re_qc_tot, &
         re_qi_tot, tau_qc, tau_qi, tau_qs, tau_qc_tot, tau_qi_tot, cbase, &
         ctop, cbase_tot, ctop_tot
    deallocate(rho, dz8w, qv, qc, qi, qs, qc_tot, qi_tot, re_cloud, re_ice, &
               re_snow, cldfra, ph, phb)
  end do
  close(u)
  close(v)
end program run_solar_diag
