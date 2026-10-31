! Driver for the second use_theta_m = 1 oracle over WRF v4.7.1's unmodified
! statements (thetam_ext_exact.F90, extracted by thetam_ext_build.py).  One
! patch, memory = tile = (1:nx+1, 1:nz+1, 1:ny+1) with ids = jds = kds = 1,
! ide = nx+1, jde = ny+1, kde = nz+1: every mass loop runs i = 1..nx,
! j = 1..ny, k = 1..nz.
!
! Input thetam_ext_inputs.bin (stream, little endian, Fortran order):
!   int nx, ny, nz, n_moist
!   real p_top, heat_flux
!   3-D (nx+1, nz+1, ny+1): t_m qv qc ph phb p pb alt u v t_tendf
!                           rthften rqvften
!   2-D (nx+1, ny+1): mu mub hfx qfx
!   1-D (nz+1): c1h c2h c1f c2f dnw fnm fnp znw
! Output thetam_ext_outputs.bin: the records named in RECORDS of
! woof/verify/thetam_ext_oracle.py, each a full memory array.
program thetam_ext_run
  use oracle_thetam_ext
  implicit none
  integer :: nx, ny, nz, n_moist
  integer :: ids, ide, jds, jde, kds, kde
  integer :: ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte
  integer :: u_in, u_out, c
  real :: p_top, heat_flux
  type(grid_config_rec_type) :: config
  real, allocatable, dimension(:,:,:) :: t_m, qv, qc, ph, phb, p, pb, alt, &
       u, v, t_tendf, rthften, rqvften
  real, allocatable :: moist(:,:,:,:), mu(:,:), mub(:,:), mut(:,:), &
       hfx(:,:), qfx(:,:), hfx_w(:,:), qfx_w(:,:)
  real, allocatable, dimension(:) :: c1h, c2h, c1f, c2f, dnw, fnm, fnp, znw, &
       zero_k, one_k
  real, allocatable, dimension(:,:,:) :: rho, th_phy, th0, p_phy, pi_phy, &
       u_phy, v_phy, p8w, t_phy, t8w, z, z_at_w, dz8w, p_hyd, p_hyd_w, &
       rth, rqv, rt, dummy
  real, allocatable :: dummy2(:,:)

  open(newunit=u_in, file='thetam_ext_inputs.bin', access='stream', &
       form='unformatted', status='old', convert='little_endian')
  read(u_in) nx, ny, nz, n_moist
  read(u_in) p_top, heat_flux
  ids = 1; ide = nx + 1; jds = 1; jde = ny + 1; kds = 1; kde = nz + 1
  ims = 1; ime = nx + 1; jms = 1; jme = ny + 1; kms = 1; kme = nz + 1
  its = 1; ite = nx + 1; jts = 1; jte = ny + 1; kts = 1; kte = nz + 1
  call alloc3(t_m); call alloc3(qv); call alloc3(qc); call alloc3(ph)
  call alloc3(phb); call alloc3(p); call alloc3(pb); call alloc3(alt)
  call alloc3(u); call alloc3(v); call alloc3(t_tendf); call alloc3(rthften)
  call alloc3(rqvften)
  read(u_in) t_m, qv, qc, ph, phb, p, pb, alt, u, v, t_tendf, rthften, rqvften
  allocate(mu(ims:ime, jms:jme), mub(ims:ime, jms:jme), mut(ims:ime, jms:jme), &
           hfx(ims:ime, jms:jme), qfx(ims:ime, jms:jme), &
           hfx_w(ims:ime, jms:jme), qfx_w(ims:ime, jms:jme), &
           dummy2(ims:ime, jms:jme))
  read(u_in) mu, mub, hfx, qfx
  allocate(c1h(kms:kme), c2h(kms:kme), c1f(kms:kme), c2f(kms:kme), &
           dnw(kms:kme), fnm(kms:kme), fnp(kms:kme), znw(kms:kme), &
           zero_k(kms:kme), one_k(kms:kme))
  read(u_in) c1h, c2h, c1f, c2f, dnw, fnm, fnp, znw
  close(u_in)
  mut = mub + mu
  zero_k = 0.
  one_k = 1.
  dummy2 = 0.
  allocate(moist(ims:ime, kms:kme, jms:jme, n_moist))
  moist = 0.
  moist(:, :, :, p_qv) = qv
  moist(:, :, :, p_qc) = qc

  call alloc3(rho); call alloc3(th_phy); call alloc3(th0); call alloc3(p_phy)
  call alloc3(pi_phy); call alloc3(u_phy); call alloc3(v_phy); call alloc3(p8w)
  call alloc3(t_phy); call alloc3(t8w); call alloc3(z); call alloc3(z_at_w)
  call alloc3(dz8w); call alloc3(p_hyd); call alloc3(p_hyd_w); call alloc3(rth)
  call alloc3(rqv); call alloc3(rt); call alloc3(dummy)

  open(newunit=u_out, file='thetam_ext_outputs.bin', access='stream', &
       form='unformatted', status='replace', convert='little_endian')

  ! A. phy_prep, use_theta_m = 1: the th_phy and rho vertical_diffusion_2
  ! reads (module_first_rk_step_part2.F:1072).
  config%use_theta_m = 1
  th_phy = -987654.; th0 = -987654.
  call phy_prep(config, mut, mut, mut, c1h, c2h, c1f, c2f, u, v, p, pb, alt, &
       ph, phb, t_m, moist, n_moist, rho, th_phy, th0, p_phy, pi_phy, u_phy, &
       v_phy, p8w, t_phy, t8w, z, z_at_w, dz8w, p_hyd, p_hyd_w, dnw, fnm, &
       fnp, znw, p_top, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, &
       kms, kme, its, ite, jts, jte, kts, kte)
  write(u_out) th_phy, rho

  ! B. phy_prep_part2, cu_physics = GFSCHEME, use_theta_m = 1: the coupled
  ! pair decoupled by c1h*mut + c2h, then RTHFTEN's dry-theta conversion
  ! (c = 1); and the conversion alone on an already decoupled pair, with
  ! c1 = 0, c2 = 1 so the decoupling divides by exactly 1 (c = 2).
  config%cu_physics = gfscheme
  do c = 1, 2
    rth = rthften; rqv = rqvften; th0 = -987654.
    dummy = 0.
    if (c == 1) then
      call part2(c1h, c2h)
    else
      call part2(zero_k, one_k)
    end if
    write(u_out) rth, rqv
  end do
  config%cu_physics = 0

  ! C. vertical_diffusion_2's hflux, use_theta_m = 1, isfflx = 0, 1, 2.
  do c = 0, 2
    config%isfflx = c
    config%tke_heat_flux = heat_flux
    rt = t_tendf; hfx_w = hfx; qfx_w = qfx
    call vd2_hflux(config, rt, moist, n_moist, th_phy, hfx_w, qfx_w, rho, &
         dnw, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
         its, ite, jts, jte, kts, kte)
    dummy = 0.
    dummy(:, 1, :) = hfx_w
    write(u_out) rt, dummy
  end do
  close(u_out)

contains

  subroutine alloc3(a)
    real, allocatable, intent(inout) :: a(:,:,:)
    allocate(a(ims:ime, kms:kme, jms:jme))
    a = 0.
  end subroutine alloc3

  subroutine part2(c1, c2)
    real, intent(in) :: c1(kms:kme), c2(kms:kme)
    real, dimension(ims:ime, kms:kme, jms:jme) :: d1, d2, d3, d4, d5, d6, &
         d7, d8, d9, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, &
         d21, d22, d23, d24, d25, d26, d27, d28, d29, d30
    d1 = 0.; d2 = 0.; d3 = 0.; d4 = 0.; d5 = 0.; d6 = 0.; d7 = 0.; d8 = 0.
    d9 = 0.; d10 = 0.; d11 = 0.; d12 = 0.; d13 = 0.; d14 = 0.; d15 = 0.
    d16 = 0.; d17 = 0.; d18 = 0.; d19 = 0.; d20 = 0.; d21 = 0.; d22 = 0.
    d23 = 0.; d24 = 0.; d25 = 0.; d26 = 0.; d27 = 0.; d28 = 0.; d29 = 0.
    d30 = 0.
    call phy_prep_part2(config, mut, mut, mut, c1, c2, c1f, c2f, &
         d1, &                                   ! rthraten
         d2, d3, d4, d5, d6, d7, &               ! rthblten..rqiblten
         d8, d9, d10, d11, d12, d13, d14, d15, & ! rucuten..rqscuten
         d16, d17, d18, d19, d20, d21, d22, d23, d24, & ! rushten..rqgshten
         rth, rqv, &
         d25, d26, d27, d28, d29, dummy2, &      ! fdda (inactive)
         t_m, th0, qv, &
         ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
         its, ite, jts, jte, kts, kte)
  end subroutine part2

end program thetam_ext_run
