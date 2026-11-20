! WRFPLUS 4.7.1 oracle driver for woof's strict small_step_init and
! small_step_finish (docs/dev/P3A-WRF-DRYCORE-ADJOINT.md section 4).
! Built against WRFPLUS's libwrflib.a (configure wrfplus, real*8, -O0
! -ffp-contract=off) by tools/wrfplus_oracle/build.sh; driven by
! tools/wrfplus_oracle/fixture.py small_step.
!
! init:   g_small_step_prep, then g_calc_p_rho (non-hydrostatic, step 0)
!         with solve_em's arguments (t_1 = t_save, muts); the adjoint is
!         a_calc_p_rho on the post-TL base, then a_small_step_prep on the
!         pre-TL copies, with c2a, muts, t_save and alt's adjoints passed
!         between them.
! finish: G_SMALL_STEP_FINISH, then A_SMALL_STEP_FINISH on the pre-TL
!         copies.
!
! The halo columns of mu_1/mub (ims, ime; jms, jme) carry woof's clamped
! or wrapped neighbours, so WRF's face masses at the outer faces are the
! ones woof forms.
!
! Input and output (stream, native binary64/int32): see fixture.py
! _ss_record and _ss_expected.
program wrfplus_small_step
  use g_module_small_step_em, only : g_small_step_prep, g_calc_p_rho, g_small_step_finish
  use a_module_small_step_em, only : a_small_step_prep, a_calc_p_rho, a_small_step_finish
  implicit none
  integer :: nx, ny, nz, rk_step, rk_order, iu
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
  real :: dts
  real, allocatable, dimension(:,:,:) :: u_1, g_u_1, u_2, g_u_2, v_1, g_v_1, v_2, g_v_2, &
       w_1, g_w_1, w_2, g_w_2, t_1, g_t_1, t_2, g_t_2, ph_1, g_ph_1, ph_2, g_ph_2, &
       p, g_p, alt, g_alt, s_u, s_v, s_w, s_t, s_ph, s_al, s_pp, s_pm1
  real, allocatable, dimension(:,:,:) :: fu2, fu2d, fv2, fv2d, fw2, fw2d, ft2, ft2d, &
       fph2, fph2d, fus, fusd, fvs, fvsd, fws, fwsd, fts, ftsd, fphs, fphsd, fh, fhd, &
       fs_u, fs_v, fs_w, fs_t, fs_ph
  real, allocatable, dimension(:,:) :: mub, mu_1, g_mu_1, mu_2, g_mu_2, muu, g_muu, &
       muv, g_muv, mut, g_mut, msfu, msfv_inv, msft, s_mu
  real, allocatable, dimension(:,:) :: fmu2, fmu2d, fmus, fmusd, fmut, fmutd, fmuts, &
       fmutsd, fmuu, fmuud, fmuus, fmuusd, fmuv, fmuvd, fmuvs, fmuvsd, fmsfv, fs_mu
  real, allocatable, dimension(:) :: rdnw, dnw, znu
  ! prep work arrays
  real, allocatable, dimension(:,:,:) :: pb, ww, g_ww, ww_save, g_ww_save, c2a, g_c2a, &
       u_save, g_u_save, v_save, g_v_save, w_save, g_w_save, t_save, g_t_save, &
       ph_save, g_ph_save, al, g_al, pp, g_pp, pm1, g_pm1
  real, allocatable, dimension(:,:) :: one, muus, g_muus, muvs, g_muvs, muts, g_muts, &
       mudf, g_mudf, mu_save, g_mu_save
  real, allocatable, dimension(:,:,:) :: u_10, u_20, v_10, v_20, w_10, w_20, t_10, t_20, &
       ph_10, ph_20
  real, allocatable, dimension(:,:) :: mu_10, mu_20
  ! adjoints
  real, allocatable, dimension(:,:,:) :: a_u_1, a_v_1, a_w_1, a_t_1, a_ph_1, a_p, a_alt, &
       a_c2a, a_t_save, a_z3, a_ww, a_ww_save, a_u_save, a_v_save, a_w_save, a_ph_save
  real, allocatable, dimension(:,:) :: a_mu_1, a_muu, a_muv, a_mut, a_muts, a_muus, &
       a_muvs, a_mudf, a_mu_save
  ! finish
  real, allocatable, dimension(:,:,:) :: fu20, fv20, fw20, ft20, fph20, fww, fwwd, fww1, &
       fww1d, z3
  real, allocatable, dimension(:,:) :: fmu20, z2
  real, allocatable, dimension(:,:,:) :: b_us, b_vs, b_ws, b_ts, b_phs, b_h
  real, allocatable, dimension(:,:) :: b_mus, b_mut, b_muts, b_muu, b_muus, b_muv, b_muvs
  character(len=512) :: fin, fout

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(newunit=iu, file=trim(fin), access='stream', form='unformatted', status='old')
  read(iu) nx, ny, nz, rk_step, rk_order
  read(iu) dts
  ids = 1; ide = nx + 1; jds = 1; jde = ny + 1; kds = 1; kde = nz + 1
  ims = 0; ime = nx + 2; jms = 0; jme = ny + 2; kms = 1; kme = nz + 1
  call a3(u_1); call a3(g_u_1); call a3(u_2); call a3(g_u_2)
  call a3(v_1); call a3(g_v_1); call a3(v_2); call a3(g_v_2)
  call a3(w_1); call a3(g_w_1); call a3(w_2); call a3(g_w_2)
  call a3(t_1); call a3(g_t_1); call a3(t_2); call a3(g_t_2)
  call a3(ph_1); call a3(g_ph_1); call a3(ph_2); call a3(g_ph_2)
  call a3(p); call a3(g_p); call a3(alt); call a3(g_alt)
  call a3(s_u); call a3(s_v); call a3(s_w); call a3(s_t); call a3(s_ph)
  call a3(s_al); call a3(s_pp); call a3(s_pm1)
  call a3(fu2); call a3(fu2d); call a3(fv2); call a3(fv2d); call a3(fw2); call a3(fw2d)
  call a3(ft2); call a3(ft2d); call a3(fph2); call a3(fph2d)
  call a3(fus); call a3(fusd); call a3(fvs); call a3(fvsd); call a3(fws); call a3(fwsd)
  call a3(fts); call a3(ftsd); call a3(fphs); call a3(fphsd); call a3(fh); call a3(fhd)
  call a3(fs_u); call a3(fs_v); call a3(fs_w); call a3(fs_t); call a3(fs_ph)
  call a2(mub); call a2(mu_1); call a2(g_mu_1); call a2(mu_2); call a2(g_mu_2)
  call a2(muu); call a2(g_muu); call a2(muv); call a2(g_muv); call a2(mut); call a2(g_mut)
  call a2(msfu); call a2(msfv_inv); call a2(msft); call a2(s_mu)
  call a2(fmu2); call a2(fmu2d); call a2(fmus); call a2(fmusd); call a2(fmut); call a2(fmutd)
  call a2(fmuts); call a2(fmutsd); call a2(fmuu); call a2(fmuud); call a2(fmuus)
  call a2(fmuusd); call a2(fmuv); call a2(fmuvd); call a2(fmuvs); call a2(fmuvsd)
  call a2(fmsfv); call a2(fs_mu)
  allocate(rdnw(kms:kme), dnw(kms:kme), znu(kms:kme))
  read(iu) rdnw, dnw
  close(iu)
  znu = 0.

  ! ------------------------------------------------------------ init TL
  allocate(one, muus, g_muus, muvs, g_muvs, muts, g_muts, mudf, g_mudf, mu_save, &
       g_mu_save, mold=mub)
  one = 1.; muus = 0; g_muus = 0; muvs = 0; g_muvs = 0; muts = 0; g_muts = 0
  mudf = 0; g_mudf = 0; mu_save = 0; g_mu_save = 0
  allocate(pb, ww, g_ww, ww_save, g_ww_save, c2a, g_c2a, u_save, g_u_save, v_save, &
       g_v_save, w_save, g_w_save, t_save, g_t_save, ph_save, g_ph_save, al, g_al, pp, &
       g_pp, pm1, g_pm1, mold=u_1)
  pb = 0; ww = 0; g_ww = 0; ww_save = 0; g_ww_save = 0; c2a = 0; g_c2a = 0
  u_save = 0; g_u_save = 0; v_save = 0; g_v_save = 0; w_save = 0; g_w_save = 0
  t_save = 0; g_t_save = 0; ph_save = 0; g_ph_save = 0; al = 0; g_al = 0; pp = 0
  g_pp = 0; pm1 = 0; g_pm1 = 0
  allocate(u_10, source=u_1); allocate(u_20, source=u_2); allocate(v_10, source=v_1)
  allocate(v_20, source=v_2); allocate(w_10, source=w_1); allocate(w_20, source=w_2)
  allocate(t_10, source=t_1); allocate(t_20, source=t_2); allocate(ph_10, source=ph_1)
  allocate(ph_20, source=ph_2); allocate(mu_10, source=mu_1); allocate(mu_20, source=mu_2)

  call g_small_step_prep(u_1, g_u_1, u_2, g_u_2, v_1, g_v_1, v_2, g_v_2, &
       w_1, g_w_1, w_2, g_w_2, t_1, g_t_1, t_2, g_t_2, ph_1, g_ph_1, ph_2, g_ph_2, &
       mub, mu_1, g_mu_1, mu_2, g_mu_2, muu, g_muu, muus, g_muus, muv, g_muv, muvs, &
       g_muvs, mut, g_mut, muts, g_muts, mudf, g_mudf, u_save, g_u_save, v_save, &
       g_v_save, w_save, g_w_save, t_save, g_t_save, ph_save, g_ph_save, mu_save, &
       g_mu_save, ww, g_ww, ww_save, g_ww_save, c2a, g_c2a, pb, p, g_p, alt, g_alt, &
       one, msfu, one, msfv_inv, one, one, msft, 1., 1., rk_step, ids, ide, jds, jde, &
       kds, kde, ims, ime, jms, jme, kms, kme, ids, ide, jds, jde, kds, kde)
  call g_calc_p_rho(al, g_al, pp, g_pp, ph_2, g_ph_2, alt, g_alt, t_2, g_t_2, &
       t_save, g_t_save, c2a, g_c2a, pm1, g_pm1, mu_2, g_mu_2, muts, g_muts, znu, &
       300., rdnw, dnw, 0.1, .true., 0, ids, ide, jds, jde, kds, kde, ims, ime, jms, &
       jme, kms, kme, ids, ide, jds, jde, kds, kde)

  open(newunit=iu, file=trim(fout), access='stream', form='unformatted', status='replace')
  write(iu) u_2, g_u_2, v_2, g_v_2, w_2, g_w_2, t_2, g_t_2, ph_2, g_ph_2, al, g_al, &
       pp, g_pp, pm1, g_pm1
  write(iu) mu_2, g_mu_2

  ! ------------------------------------------------------------ init AD
  allocate(a_u_1, a_v_1, a_w_1, a_t_1, a_ph_1, a_p, a_alt, a_c2a, a_t_save, a_z3, &
       a_ww, a_ww_save, a_u_save, a_v_save, a_w_save, a_ph_save, mold=u_1)
  a_u_1 = 0; a_v_1 = 0; a_w_1 = 0; a_t_1 = 0; a_ph_1 = 0; a_p = 0; a_alt = 0
  a_c2a = 0; a_t_save = 0; a_z3 = 0; a_ww = 0; a_ww_save = 0; a_u_save = 0
  a_v_save = 0; a_w_save = 0; a_ph_save = 0
  allocate(a_mu_1, a_muu, a_muv, a_mut, a_muts, a_muus, a_muvs, a_mudf, a_mu_save, &
       mold=mub)
  a_mu_1 = 0; a_muu = 0; a_muv = 0; a_mut = 0; a_muts = 0; a_muus = 0; a_muvs = 0
  a_mudf = 0; a_mu_save = 0
  ! s_pp is the adjoint of calc_p_rho's p output, s_pm1 of pm1's.
  call a_calc_p_rho(al, s_al, pp, s_pp, ph_2, s_ph, alt, a_alt, t_2, s_t, &
       t_save, a_t_save, c2a, a_c2a, pm1, s_pm1, mu_2, s_mu, muts, a_muts, znu, &
       300., rdnw, dnw, 0.1, .true., 0, ids, ide, jds, jde, kds, kde, ims, ime, jms, &
       jme, kms, kme, ids, ide, jds, jde, kds, kde)
  call a_small_step_prep(u_10, a_u_1, u_20, s_u, v_10, a_v_1, v_20, s_v, &
       w_10, a_w_1, w_20, s_w, t_10, a_t_1, t_20, s_t, ph_10, a_ph_1, ph_20, s_ph, &
       mub, mu_10, a_mu_1, mu_20, s_mu, muu, a_muu, muus, a_muus, muv, a_muv, muvs, &
       a_muvs, mut, a_mut, muts, a_muts, mudf, a_mudf, u_save, a_u_save, v_save, &
       a_v_save, w_save, a_w_save, t_save, a_t_save, ph_save, a_ph_save, mu_save, &
       a_mu_save, ww, a_ww, ww_save, a_ww_save, c2a, a_c2a, pb, p, a_p, alt, a_alt, &
       one, msfu, one, msfv_inv, one, one, msft, 1., 1., rk_step, ids, ide, jds, jde, &
       kds, kde, ims, ime, jms, jme, kms, kme, ids, ide, jds, jde, kds, kde)
  write(iu) a_u_1, s_u, a_v_1, s_v, a_w_1, s_w, a_t_1, s_t, a_ph_1, s_ph, a_p, a_alt
  write(iu) a_mu_1, s_mu, a_muu, a_muv, a_mut

  ! ---------------------------------------------------------- finish TL
  allocate(fu20, source=fu2); allocate(fv20, source=fv2); allocate(fw20, source=fw2)
  allocate(ft20, source=ft2); allocate(fph20, source=fph2); allocate(fmu20, source=fmu2)
  allocate(fww, fwwd, fww1, fww1d, z3, mold=u_1)
  fww = 0; fwwd = 0; fww1 = 0; fww1d = 0; z3 = 0
  allocate(z2, mold=mub); z2 = 0
  call g_small_step_finish(fu2, fu2d, z3, fv2, fv2d, z3, fw2, fw2d, z3, ft2, ft2d, &
       z3, fph2, fph2d, z3, fww, fwwd, fww1, fww1d, fmu2, fmu2d, z2, fmut, fmutd, &
       fmuts, fmutsd, fmuu, fmuud, fmuus, fmuusd, fmuv, fmuvd, fmuvs, fmuvsd, fus, &
       fusd, fvs, fvsd, fws, fwsd, fts, ftsd, fphs, fphsd, fmus, fmusd, one, msfu, &
       fmsfv, one, one, msft, fh, fhd, 1, dts, rk_step, rk_order, ids, ide, jds, &
       jde, kds, kde, ims, ime, jms, jme, kms, kme, ids, ide, jds, jde, kds, kde)
  write(iu) fu2, fu2d, fv2, fv2d, fw2, fw2d, ft2, ft2d, fph2, fph2d
  write(iu) fmu2, fmu2d

  ! ---------------------------------------------------------- finish AD
  allocate(b_us, b_vs, b_ws, b_ts, b_phs, b_h, mold=u_1)
  b_us = 0; b_vs = 0; b_ws = 0; b_ts = 0; b_phs = 0; b_h = 0
  allocate(b_mus, b_mut, b_muts, b_muu, b_muus, b_muv, b_muvs, mold=mub)
  b_mus = 0; b_mut = 0; b_muts = 0; b_muu = 0; b_muus = 0; b_muv = 0; b_muvs = 0
  fww = 0; fwwd = 0; fww1d = 0
  call a_small_step_finish(fu20, fs_u, z3, fv20, fs_v, z3, fw20, fs_w, z3, ft20, &
       fs_t, z3, fph20, fs_ph, z3, fww, fwwd, fww1, fww1d, fmu20, fs_mu, z2, fmut, &
       b_mut, fmuts, b_muts, fmuu, b_muu, fmuus, b_muus, fmuv, b_muv, fmuvs, b_muvs, &
       fus, b_us, fvs, b_vs, fws, b_ws, fts, b_ts, fphs, b_phs, fmus, b_mus, one, &
       msfu, fmsfv, one, one, msft, fh, b_h, 1, dts, rk_step, rk_order, ids, ide, &
       jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ids, ide, jds, jde, kds, kde)
  write(iu) fs_u, b_us, fs_v, b_vs, fs_w, b_ws, fs_t, b_ts, fs_ph, b_phs, b_h
  write(iu) fs_mu, b_mus, b_mut, b_muts, b_muu, b_muus, b_muv, b_muvs
  close(iu)

contains

  subroutine a3(x)
    real, allocatable, intent(out) :: x(:,:,:)
    allocate(x(ims:ime, kms:kme, jms:jme))
    read(iu) x
  end subroutine a3

  subroutine a2(x)
    real, allocatable, intent(out) :: x(:,:)
    allocate(x(ims:ime, jms:jme))
    read(iu) x
  end subroutine a2

end program wrfplus_small_step
