! Driver for the use_theta_m = 1 / do_avgflx_em = 1 oracle over WRF v4.7.1's
! unmodified routine bodies (thetam_exact.F90, extracted by thetam_build.py;
! module_avgflx_em.F compiled whole).  One patch, memory = tile =
! (1:nx+1, 1:nz+1, 1:ny+1) with ids = jds = kds = 1, ide = nx+1, jde = ny+1,
! kde = nz+1: every mass loop runs i = 1..nx, j = 1..ny, k = 1..nz.
!
! Input thetam_inputs.bin (stream, little endian, Fortran order):
!   int nx, ny, nz, n_moist
!   real p_top, dt, lim_a, lim_b
!   3-D (nx+1, nz+1, ny+1): t_dry qv qc ph phb p pb alb alt u v
!                           th_post qv_post qc_post t_tendf moist_tend
!                           avg_rum avg_rvm avg_wwm ru_m rv_m ww_m
!   2-D (nx+1, ny+1): mu mub
!   1-D (nz+1): c1h c2h c3h c4h c1f c2f c3f c4f znu znw dnw rdnw rdn fnm fnp
! Output thetam_outputs.bin: the records named in OUTPUTS of
! woof/verify/thetam_oracle.py, each a full (nx+1, nz+1, ny+1) memory array.
program thetam_run
  use oracle_thetam
  use module_avgflx_em, only: zero_avgflx, upd_avgflx
  implicit none
  integer :: nx, ny, nz, n_moist, ids, ide, jds, jde, kds, kde
  integer :: ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte
  integer :: u_in, u_out, c, counts(4)
  real :: p_top, dt, lim_a, lim_b
  type(grid_config_rec_type) :: config
  type(oracle_grid) :: grid
  real, allocatable, dimension(:,:,:) :: t_dry, qv, qc, ph, phb, p, pb, alb, &
       alt, u, v, th_post, qv_post, qc_post, t_tendf, moist_tend, avg_rum, &
       avg_rvm, avg_wwm, ru_m, rv_m, ww_m
  real, allocatable :: moist(:,:,:,:), mu(:,:), mub(:,:), muts(:,:), mut(:,:)
  real, allocatable, dimension(:) :: c1h, c2h, c3h, c4h, c1f, c2f, c3f, c4f, &
       znu, znw, dnw, rdnw, rdn, fnm, fnp
  real, allocatable, dimension(:,:,:) :: t_m, al, pp, rho, th_phy, th0, p_phy, &
       pi_phy, u_phy, v_phy, p8w, t_phy, t8w, z, z_at_w, dz8w, p_hyd, p_hyd_w, &
       h_diab, qv_diab, qc_diab, pii, pf, t_new, t_old, work, rum, rvm, wwm

  open(newunit=u_in, file='thetam_inputs.bin', access='stream', &
       form='unformatted', status='old', convert='little_endian')
  read(u_in) nx, ny, nz, n_moist
  read(u_in) p_top, dt, lim_a, lim_b
  ids = 1; ide = nx + 1; jds = 1; jde = ny + 1; kds = 1; kde = nz + 1
  ims = 1; ime = nx + 1; jms = 1; jme = ny + 1; kms = 1; kme = nz + 1
  its = 1; ite = nx + 1; jts = 1; jte = ny + 1; kts = 1; kte = nz + 1
  call alloc3(t_dry); call alloc3(qv); call alloc3(qc); call alloc3(ph)
  call alloc3(phb); call alloc3(p); call alloc3(pb); call alloc3(alb)
  call alloc3(alt); call alloc3(u); call alloc3(v); call alloc3(th_post)
  call alloc3(qv_post); call alloc3(qc_post); call alloc3(t_tendf)
  call alloc3(moist_tend); call alloc3(avg_rum); call alloc3(avg_rvm)
  call alloc3(avg_wwm); call alloc3(ru_m); call alloc3(rv_m); call alloc3(ww_m)
  read(u_in) t_dry, qv, qc, ph, phb, p, pb, alb, alt, u, v, th_post, qv_post, &
       qc_post, t_tendf, moist_tend, avg_rum, avg_rvm, avg_wwm, ru_m, rv_m, ww_m
  allocate(mu(ims:ime, jms:jme), mub(ims:ime, jms:jme), muts(ims:ime, jms:jme), &
           mut(ims:ime, jms:jme))
  read(u_in) mu, mub
  allocate(c1h(kms:kme), c2h(kms:kme), c3h(kms:kme), c4h(kms:kme), &
           c1f(kms:kme), c2f(kms:kme), c3f(kms:kme), c4f(kms:kme), &
           znu(kms:kme), znw(kms:kme), dnw(kms:kme), rdnw(kms:kme), &
           rdn(kms:kme), fnm(kms:kme), fnp(kms:kme))
  read(u_in) c1h, c2h, c3h, c4h, c1f, c2f, c3f, c4f, znu, znw, dnw, rdnw, rdn, &
       fnm, fnp
  close(u_in)
  muts = mub + mu
  mut = muts
  allocate(moist(ims:ime, kms:kme, jms:jme, n_moist))
  moist = 0.
  moist(:, :, :, p_qv) = qv
  moist(:, :, :, p_qc) = qc

  call alloc3(t_m); call alloc3(al); call alloc3(pp); call alloc3(rho)
  call alloc3(th_phy); call alloc3(th0); call alloc3(p_phy); call alloc3(pi_phy)
  call alloc3(u_phy); call alloc3(v_phy); call alloc3(p8w); call alloc3(t_phy)
  call alloc3(t8w); call alloc3(z); call alloc3(z_at_w); call alloc3(dz8w)
  call alloc3(p_hyd); call alloc3(p_hyd_w); call alloc3(h_diab)
  call alloc3(qv_diab); call alloc3(qc_diab); call alloc3(pii); call alloc3(pf)
  call alloc3(t_new); call alloc3(t_old); call alloc3(work)
  call alloc3(rum); call alloc3(rvm); call alloc3(wwm)

  open(newunit=u_out, file='thetam_outputs.bin', access='stream', &
       form='unformatted', status='replace', convert='little_endian')

  ! A. real.exe's theta -> theta_m (module_initialize_real.F:4923-4931).
  config%use_theta_m = 1
  allocate(grid%t_2(ims:ime, kms:kme, jms:jme))
  grid%t_2 = t_dry
  call real_theta_m(config, grid, moist, n_moist, ids, ide, jds, jde, kds, &
       kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte)
  t_m = grid%t_2
  write(u_out) t_m

  ! B. calc_p_rho_phi: theta_m hypsometric 1 and 2, and the dry control.
  do c = 1, 3
    al = -987654.; pp = -987654.
    if (c == 3) then
      call calc_p_rho_phi(moist, n_moist, 1, al, alb, mu, muts, c1h, c2h, &
           c3h, c4h, c3f, c4f, ph, phb, pp, pb, t_dry, p1000mb, t0, p_top, &
           znu, znw, dnw, rdnw, rdn, .true., 0, ids, ide, jds, jde, kds, &
           kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte)
    else
      call calc_p_rho_phi(moist, n_moist, c, al, alb, mu, muts, c1h, c2h, &
           c3h, c4h, c3f, c4f, ph, phb, pp, pb, t_m, p1000mb, t0, p_top, &
           znu, znw, dnw, rdnw, rdn, .true., 1, ids, ide, jds, jde, kds, &
           kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte)
    end if
    write(u_out) al, pp
  end do

  ! C. phy_prep, use_theta_m = 1.
  th_phy = -987654.; th0 = -987654.
  call phy_prep(config, mut, mut, mut, c1h, c2h, c1f, c2f, u, v, p, pb, alt, &
       ph, phb, t_m, moist, n_moist, rho, th_phy, th0, p_phy, pi_phy, u_phy, &
       v_phy, p8w, t_phy, t8w, z, z_at_w, dz8w, p_hyd, p_hyd_w, dnw, fnm, &
       fnp, znw, p_top, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, &
       kms, kme, its, ite, jts, jte, kts, kte)
  write(u_out) th_phy, th0, t_phy, rho

  ! D. moist_physics_prep_em, use_theta_m = 1.
  t_new = t_m; t_old = t_m
  th_phy = -987654.; h_diab = -987654.; qv_diab = -987654.
  call moist_physics_prep_em(t_new, t_old, t0, rho, al, alb, p, p8w, &
       p1000mb, pb, ph, phb, th_phy, pii, pf, z, z_at_w, dz8w, dt, h_diab, &
       qv, qv_diab, qc, qc_diab, config, fnm, fnp, ids, ide, jds, jde, kds, &
       kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte)
  write(u_out) th_phy, h_diab, qv_diab

  ! E. moist_physics_finish_em, use_theta_m = 1, unclamped and clamped.
  ! th_post is the scheme's theta increment: the post-scheme th_phy is the
  ! prep's th_phy plus it, written with the prep's h_diabatic as the
  ! finish's own inputs.
  do c = 1, 2
    config%mp_tend_lim = merge(lim_a, lim_b, c == 1)
    t_new = t_m
    call moist_physics_prep_em(t_new, t_old, t0, rho, al, alb, p, p8w, &
         p1000mb, pb, ph, phb, th_phy, pii, pf, z, z_at_w, dz8w, dt, &
         h_diab, qv, qv_diab, qc, qc_diab, config, fnm, fnp, ids, ide, jds, &
         jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, &
         kts, kte)
    th_phy = th_phy + th_post
    write(u_out) th_phy, h_diab
    th0 = -987654.
    call moist_physics_finish_em(t_new, t_old, t0, mut, th_phy, h_diab, dt, &
         qv_post, qv_diab, qc_post, qc_diab, th0, config, ids, ide, jds, &
         jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, &
         kts, kte)
    write(u_out) t_new, th0, h_diab
  end do

  ! F. conv_t_tendf_to_moist (moist_old from initialize_moist_old).
  work = -987654.
  call initialize_moist_old(work, qv, ids, ide, jds, jde, kds, kde, ims, &
       ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte)
  call conv_t_tendf_to_moist(t_m, work, t_tendf, moist_tend, ids, ide, jds, &
       jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, &
       kte)
  write(u_out) t_tendf

  ! G. upd_avgflx: zero_avgflx then count 0, and counts 1, 5, 359.
  counts = (/ 0, 1, 5, 359 /)
  do c = 1, 4
    rum = avg_rum; rvm = avg_rvm; wwm = avg_wwm
    if (c == 1) call zero_avgflx(rum, rvm, wwm, ids, ide, jds, jde, kds, &
         kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte, &
         .false.)
    call upd_avgflx(counts(c), rum, rvm, wwm, ru_m, rv_m, ww_m, ids, ide, &
         jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, &
         jte, kts, kte, .false.)
    write(u_out) rum, rvm, wwm
  end do
  close(u_out)

contains

  subroutine alloc3(a)
    real, allocatable, intent(inout) :: a(:,:,:)
    allocate(a(ims:ime, kms:kme, jms:jme))
    a = 0.
  end subroutine alloc3

end program thetam_run
