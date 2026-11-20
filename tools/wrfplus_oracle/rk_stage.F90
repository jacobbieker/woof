! WRFPLUS 4.7.1 oracle driver for woof's strict RK-stage glue
! (docs/dev/P3A-WRF-DRYCORE-ADJOINT.md section 5):
!
!   rk_step_prep    G_CALC_MU_UV, G_COUPLE_MOMENTUM, g_calc_ww_cp (the
!                   dry part of g_rk_step_prep, in its order); the adjoint
!                   a_calc_ww_cp, A_COUPLE_MOMENTUM, A_CALC_MU_UV;
!   calc_p_rho_phi  G_/A_CALC_P_RHO_PHI (dry, non-hydrostatic);
!   rk_addtend_dry  G_/A_RK_ADDTEND_DRY at rk_step 2 (no save fold);
!   set_w_surface   g_/a_set_w_surface (fill_w_flag false).
!
! Each operator reads its own arrays from one record; the adjoints run on
! pre-TL copies of the base state.  The halo columns of mu, mub and ht
! (ims, ime; jms, jme) carry woof's clamped or wrapped neighbours.
! Built against WRFPLUS's libwrflib.a (configure wrfplus, real*8, -O0
! -ffp-contract=off) by tools/wrfplus_oracle/build.sh; driven by
! tools/wrfplus_oracle/fixture.py rk_stage.
!
! Input and output (stream, native binary64/int32): see fixture.py
! _rk_record and _rk_expected.
program wrfplus_rk_stage
  use module_configure, only : grid_config_rec_type
  use g_module_big_step_utilities_em, only : g_calc_mu_uv, g_couple_momentum, &
       g_calc_ww_cp, g_calc_p_rho_phi
  use a_module_big_step_utilities_em, only : a_calc_mu_uv, a_couple_momentum, &
       a_calc_ww_cp, a_calc_p_rho_phi
  use g_module_em, only : g_rk_addtend_dry
  use a_module_em, only : a_rk_addtend_dry
  use g_module_bc_em, only : g_set_w_surface
  use a_module_bc_em, only : a_set_w_surface
  implicit none
  integer :: nx, ny, nz, hypso, px, py, iu
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
  real :: rdx, rdy, ptop, cf1, cf2, cf3
  type(grid_config_rec_type) :: cf
  real, allocatable, dimension(:) :: dnw, rdnw, znu, znw
  ! rk_step_prep
  real, allocatable, dimension(:,:,:) :: ru_u, ru_gu, ru_v, ru_gv, ru_sru, ru_srv, ru_sww, &
       ru, gru, rv, grv, ww, gww, rw, grw, z3, a_u, a_v, a_w3
  real, allocatable, dimension(:,:) :: ru_mu, ru_gmu, ru_mub, ru_msfu, ru_msfvi, &
       ru_msft, muu, gmuu, muv, gmuv, a_mu, a_muu, a_muv, a_mut, z2, one
  ! calc_p_rho_phi
  real, allocatable, dimension(:,:,:) :: pr_t, pr_gt, pr_ph, pr_gph, pr_phb, pr_alb, &
       pr_pb, pr_sal, pr_sp, al, gal, p, gp, a_t, a_ph
  real, allocatable, dimension(:,:) :: pr_mu, pr_gmu, pr_muts, pr_gmuts, a_muts
  real, allocatable, dimension(:,:,:,:) :: moist, gmoist
  ! rk_addtend_dry
  real, allocatable, dimension(:,:,:) :: at_t1, at_t2, at_t3, at_t4, at_t5, &
       at_g1, at_g2, at_g3, at_g4, at_g5, at_f1, at_f2, at_f3, at_f4, at_f5, &
       at_d1, at_d2, at_d3, at_d4, at_d5, at_h, at_gh, at_s1, at_s2, at_s3, at_s4, &
       at_s5, b_f1, b_f2, b_f3, b_f4, b_f5, b_h
  real, allocatable, dimension(:,:) :: at_mt, at_gmt, at_mtf, at_gmtf, at_smt, at_mut, &
       at_gmut, at_msfu, at_msfvi, at_msft, b_mtf, b_mut
  ! set_w_surface
  real, allocatable, dimension(:,:,:) :: sw_w, sw_gw, sw_u, sw_gu, sw_v, sw_gv, sw_sw
  real, allocatable, dimension(:,:) :: sw_ht, sw_msft
  character(len=512) :: fin, fout

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(newunit=iu, file=trim(fin), access='stream', form='unformatted', status='old')
  read(iu) nx, ny, nz, hypso, px, py
  read(iu) rdx, rdy, ptop, cf1, cf2, cf3
  ids = 1; ide = nx + 1; jds = 1; jde = ny + 1; kds = 1; kde = nz + 1
  ims = 0; ime = nx + 2; jms = 0; jme = ny + 2; kms = 1; kme = nz + 1
  allocate(dnw(kms:kme), rdnw(kms:kme), znu(kms:kme), znw(kms:kme))
  read(iu) dnw, rdnw, znu, znw
  call a3(ru_u); call a3(ru_gu); call a3(ru_v); call a3(ru_gv)
  call a3(ru_sru); call a3(ru_srv); call a3(ru_sww)
  call a2(ru_mu); call a2(ru_gmu); call a2(ru_mub); call a2(ru_msfu); call a2(ru_msfvi)
  call a2(ru_msft)
  call a3(pr_t); call a3(pr_gt); call a3(pr_ph); call a3(pr_gph); call a3(pr_phb)
  call a3(pr_alb); call a3(pr_pb); call a3(pr_sal); call a3(pr_sp)
  call a2(pr_mu); call a2(pr_gmu); call a2(pr_muts); call a2(pr_gmuts)
  call a3(at_t1); call a3(at_t2); call a3(at_t3); call a3(at_t4); call a3(at_t5)
  call a3(at_g1); call a3(at_g2); call a3(at_g3); call a3(at_g4); call a3(at_g5)
  call a3(at_f1); call a3(at_f2); call a3(at_f3); call a3(at_f4); call a3(at_f5)
  call a3(at_d1); call a3(at_d2); call a3(at_d3); call a3(at_d4); call a3(at_d5)
  call a3(at_h); call a3(at_gh)
  call a3(at_s1); call a3(at_s2); call a3(at_s3); call a3(at_s4); call a3(at_s5)
  call a2(at_mt); call a2(at_gmt); call a2(at_mtf); call a2(at_gmtf); call a2(at_smt)
  call a2(at_mut); call a2(at_gmut); call a2(at_msfu); call a2(at_msfvi); call a2(at_msft)
  call a3(sw_w); call a3(sw_gw); call a3(sw_u); call a3(sw_gu); call a3(sw_v)
  call a3(sw_gv); call a3(sw_sw)
  call a2(sw_ht); call a2(sw_msft)
  close(iu)

  cf%specified = .false.;  cf%nested = .false.
  cf%periodic_x = px /= 0; cf%periodic_y = py /= 0
  cf%open_xs = .false.;    cf%open_xe = .false.;  cf%open_ys = .false.
  cf%open_ye = .false.;    cf%symmetric_xs = .false.; cf%symmetric_xe = .false.
  cf%symmetric_ys = .false.; cf%symmetric_ye = .false.; cf%polar = .false.

  allocate(z3, ru, gru, rv, grv, ww, gww, rw, grw, a_u, a_v, a_w3, mold=ru_u)
  z3 = 0; ru = 0; gru = 0; rv = 0; grv = 0; ww = 0; gww = 0; rw = 0; grw = 0
  a_u = 0; a_v = 0; a_w3 = 0
  allocate(z2, one, muu, gmuu, muv, gmuv, a_mu, a_muu, a_muv, a_mut, mold=ru_mu)
  z2 = 0; one = 1; muu = 0; gmuu = 0; muv = 0; gmuv = 0; a_mu = 0; a_muu = 0
  a_muv = 0; a_mut = 0

  open(newunit=iu, file=trim(fout), access='stream', form='unformatted', status='replace')

  ! ------------------------------------------------------- rk_step_prep
  call g_calc_mu_uv(cf, ru_mu, ru_gmu, ru_mub, muu, gmuu, muv, gmuv, &
       ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
       ids, ide, jds, jde, kds, kde)
  call g_couple_momentum(muu, gmuu, ru, gru, ru_u, ru_gu, ru_msfu, muv, gmuv, rv, &
       grv, ru_v, ru_gv, one, ru_msfvi, z2, z2, rw, grw, z3, z3, one, &
       ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
       ids, ide, jds, jde, kds, kde)
  call g_calc_ww_cp(ru_u, ru_gu, ru_v, ru_gv, ru_mu, ru_gmu, ru_mub, ww, gww, rdx, &
       rdy, ru_msft, ru_msft, one, ru_msfu, one, ru_msfvi, one, dnw, &
       ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
       ids, ide, jds, jde, kds, kde)
  write(iu) ru, gru, rv, grv, ww, gww
  call a_calc_ww_cp(ru_u, a_u, ru_v, a_v, ru_mu, a_mu, ru_mub, ww, ru_sww, rdx, &
       rdy, ru_msft, ru_msft, one, ru_msfu, one, ru_msfvi, one, dnw, &
       ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
       ids, ide, jds, jde, kds, kde)
  call a_couple_momentum(muu, a_muu, ru_sru, ru_u, a_u, ru_msfu, muv, a_muv, &
       ru_srv, ru_v, a_v, one, ru_msfvi, z2, a_mut, a_w3, z3, z3, one, &
       ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
       ids, ide, jds, jde, kds, kde)
  call a_calc_mu_uv(cf, a_mu, a_muu, a_muv, &
       ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
       ids, ide, jds, jde, kds, kde)
  write(iu) a_u, a_v, ru_sru, ru_srv, ru_sww
  write(iu) a_mu

  ! ----------------------------------------------------- calc_p_rho_phi
  allocate(moist(ims:ime, kms:kme, jms:jme, 1), gmoist(ims:ime, kms:kme, jms:jme, 1))
  moist = 0; gmoist = 0
  allocate(al, gal, p, gp, a_t, a_ph, mold=ru_u)
  al = 0; gal = 0; p = 0; gp = 0; a_t = 0; a_ph = 0
  allocate(a_muts, mold=ru_mu); a_muts = 0
  call g_calc_p_rho_phi(moist, gmoist, 1, hypso, al, gal, pr_alb, pr_mu, pr_gmu, &
       pr_muts, pr_gmuts, pr_ph, pr_gph, pr_phb, p, gp, pr_pb, pr_t, pr_gt, &
       1.e5, 300., ptop, znu, znw, dnw, rdnw, rdnw, .true., &
       ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
       ids, ide, jds, jde, kds, kde)
  write(iu) al, gal, p, gp
  a_mu = 0
  call a_calc_p_rho_phi(moist, gmoist, 1, hypso, al, pr_sal, pr_alb, pr_mu, a_mu, &
       pr_muts, a_muts, pr_ph, a_ph, pr_phb, p, pr_sp, pr_pb, pr_t, a_t, &
       1.e5, 300., ptop, znu, znw, dnw, rdnw, rdnw, .true., &
       ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
       ids, ide, jds, jde, kds, kde)
  write(iu) a_t, a_ph, pr_sal, pr_sp
  write(iu) a_mu, a_muts

  ! ----------------------------------------------------- rk_addtend_dry
  call g_rk_addtend_dry(at_t1, at_g1, at_t2, at_g2, at_t3, at_g3, at_t4, at_g4, &
       at_t5, at_g5, at_f1, at_d1, at_f2, at_d2, at_f3, at_d3, at_f4, at_d4, &
       at_f5, at_d5, z3, z3, z3, z3, z3, z3, z3, z3, z3, z3, at_mt, at_gmt, &
       at_mtf, at_gmtf, 2, at_h, at_gh, at_mut, at_gmut, at_msft, at_msft, one, &
       at_msfu, one, at_msfvi, one, &
       ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
       ids, ide, jds, jde, kds, kde, ids, ide, jds, jde, kds, kde)
  write(iu) at_t1, at_g1, at_t2, at_g2, at_t3, at_g3, at_t4, at_g4, at_t5, at_g5
  write(iu) at_mt, at_gmt
  allocate(b_f1, b_f2, b_f3, b_f4, b_f5, b_h, mold=ru_u)
  b_f1 = 0; b_f2 = 0; b_f3 = 0; b_f4 = 0; b_f5 = 0; b_h = 0
  allocate(b_mtf, b_mut, mold=ru_mu); b_mtf = 0; b_mut = 0
  ! at_t*/at_f*/at_mt are past the TL now; the adjoint reads only the
  ! base h_diabatic and mut, which the TL left alone.
  call a_rk_addtend_dry(at_t1, at_s1, at_t2, at_s2, at_t3, at_s3, at_t4, at_s4, &
       at_t5, at_s5, at_f1, b_f1, at_f2, b_f2, at_f3, b_f3, at_f4, b_f4, &
       at_f5, b_f5, z3, z3, z3, z3, z3, z3, z3, z3, z3, z3, at_mt, at_smt, &
       at_mtf, b_mtf, 2, at_h, b_h, at_mut, b_mut, at_msft, at_msft, one, &
       at_msfu, one, at_msfvi, one, &
       ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
       ids, ide, jds, jde, kds, kde, ids, ide, jds, jde, kds, kde)
  write(iu) b_f1, b_f2, b_f3, b_f4, b_f5, b_h, at_s1, at_s2, at_s3, at_s4, at_s5
  write(iu) b_mtf, at_smt, b_mut

  ! ------------------------------------------------------ set_w_surface
  call g_set_w_surface(cf, znw, .false., sw_w, sw_gw, sw_ht, sw_u, sw_gu, sw_v, &
       sw_gv, cf1, cf2, cf3, rdx, rdy, sw_msft, sw_msft, &
       ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
       ids, ide, jds, jde, kds, kde)
  write(iu) sw_w, sw_gw
  a_u = 0; a_v = 0
  call a_set_w_surface(cf, znw, .false., sw_w, sw_sw, sw_ht, sw_u, a_u, sw_v, &
       a_v, cf1, cf2, cf3, rdx, rdy, sw_msft, sw_msft, &
       ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
       ids, ide, jds, jde, kds, kde)
  write(iu) a_u, a_v, sw_sw
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

end program wrfplus_rk_stage
