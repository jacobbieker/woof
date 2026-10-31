! Driver for WRF v4.7.1's km_opt = 5 (SMS-3DTKE) routines, called on one
! single-tile patch in the order first_rk_step_part1/part2 call them:
!
!   phy_prep                       (module_first_rk_step_part1.F)
!   compute_diff_metrics + its set_physical_bc3d/2d calls
!                                  (module_first_rk_step_part2.F:433-500)
!   cal_deform_and_div             (:506-526)
!   nonlocal_flux, free_atmos_length, meso_length_scale, calculate_km_kh
!                                  (:582-645)
!   phy_bc                         (:660-678)
!   tke_rhs                        (:904-933)
!   vertical_diffusion_implicit    (:1028-1052)
!
! tke_shear is also called once on its own (into a separate zeroed array):
! it is the first term tke_rhs accumulates, and woof's production path
! takes it from the km_opt = 2 kernel, so the oracle records the word the
! km_opt = 5 remainder starts from.  It is a pure function of its inputs.
!
! The domain is the patch (its..ite = ids..ide, staggered extents
! included); the memory carries a 5-point halo.  The raw state's halo is
! the periodic copy (the halo exchange a periodic domain makes); every
! computed array's halo is WRF's own set_physical_bc3d/2d.  Input
! cases.bin and output oracle.bin are little-endian streams described in
! make_cases.py.
program run_sms3dtke
  use module_oracle_config
  use module_model_constants
  use sms_cut
  implicit none
  integer :: u_in, u_out, ncase, c
  open(newunit=u_in, file='cases.bin', access='stream', form='unformatted', &
       status='old', convert='little_endian')
  open(newunit=u_out, file='oracle.bin', access='stream', form='unformatted', &
       status='replace', convert='little_endian')
  read(u_in) ncase
  write(u_out) ncase
  do c = 1, ncase
    call run_case(u_in, u_out)
  end do
  close(u_in)
  close(u_out)
contains

  subroutine fill3(a, ims, ime, kms, kme, jms, jme, ni, nj, ilast, jlast, k1, k2)
    ! Periodic halo of a raw input: points outside 1..ilast / 1..jlast take
    ! the value ni (nj) columns (rows) away.
    integer, intent(in) :: ims, ime, kms, kme, jms, jme, ni, nj, ilast, jlast, k1, k2
    real, intent(inout) :: a(ims:ime, kms:kme, jms:jme)
    integer :: i, j, k, ii, jj
    do j = jms, jme
      do k = k1, k2
        do i = ims, ime
          ii = i; jj = j
          do while (ii < 1); ii = ii + ni; end do
          do while (ii > ilast); ii = ii - ni; end do
          do while (jj < 1); jj = jj + nj; end do
          do while (jj > jlast); jj = jj - nj; end do
          a(i, k, j) = a(ii, k, jj)
        end do
      end do
    end do
  end subroutine fill3

  subroutine read2(u, a, ims, ime, jms, jme, nx, ny)
    integer, intent(in) :: u, ims, ime, jms, jme, nx, ny
    real, intent(inout) :: a(ims:ime, jms:jme)
    integer :: i, j
    read(u) ((a(i, j), i=1,nx), j=1,ny)
  end subroutine read2

  subroutine read3(u, a, ims, ime, kms, kme, jms, jme, nx, ny, nz)
    integer, intent(in) :: u, ims, ime, kms, kme, jms, jme, nx, ny, nz
    real, intent(inout) :: a(ims:ime, kms:kme, jms:jme)
    integer :: i, k, j
    read(u) (((a(i, k, j), i=1,nx), k=1,nz), j=1,ny)
  end subroutine read3

  subroutine fill2(a, ims, ime, jms, jme, nx, ny)
    integer, intent(in) :: ims, ime, jms, jme, nx, ny
    real, intent(inout) :: a(ims:ime, jms:jme)
    real :: b(ims:ime, 1:1, jms:jme)
    b(:, 1, :) = a
    call fill3(b, ims, ime, 1, 1, jms, jme, nx, ny, nx, ny, 1, 1)
    a = b(:, 1, :)
  end subroutine fill2

  subroutine write2(u, a, ims, ime, jms, jme, nx, ny)
    integer, intent(in) :: u, ims, ime, jms, jme, nx, ny
    real, intent(in) :: a(ims:ime, jms:jme)
    integer :: i, j
    write(u) ((a(i, j), i=1,nx), j=1,ny)
  end subroutine write2

  subroutine write3(u, a, ims, ime, kms, kme, jms, jme, nx, ny, nz)
    integer, intent(in) :: u, ims, ime, kms, kme, jms, jme, nx, ny, nz
    real, intent(in) :: a(ims:ime, kms:kme, jms:jme)
    integer :: i, k, j
    write(u) (((a(i, k, j), i=1,nx), k=1,nz), j=1,ny)
  end subroutine write3

  subroutine run_case(u_in, u_out)
    integer, intent(in) :: u_in, u_out
    integer :: nx, ny, nz, bnd, isfflx, mixfull, thetam
    real :: dx, dy, dt, c_s, c_k, heat, drag, cf1, cf2, cf3, p_top, rdx, rdy
    integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
    integer :: its, ite, jts, jte, kts, kte, i, j, k, ii
    real, allocatable, dimension(:) :: fnm, fnp, dn, dnw, c1h, c2h, c1f, c2f, znw, &
                                       t_base, qv_base, u_base, v_base
    real, allocatable, dimension(:,:) :: msf, mut, ust, ustm, hfx, qfx, br, rmol, &
                                         wspd, u10, v10, gamu, gamv, pblh, muu, muv
    integer, allocatable :: kpbl(:,:)
    real, allocatable, dimension(:,:,:) :: u, v, w, ph, phb, t, p, pb, alt, tke
    real, allocatable :: moist(:,:,:,:), moist_tend(:,:,:,:), nba(:,:,:,:)
    real, allocatable :: chem(:,:,:,:), chem_tend(:,:,:,:), scal(:,:,:,:), &
                         scal_tend(:,:,:,:), trac(:,:,:,:), trac_tend(:,:,:,:)
    real, allocatable, dimension(:,:,:) :: rho, th_phy, th_phy_m_t0, p_phy, pi_phy, &
         u_phy, v_phy, p8w, t_phy, t8w, z, z_at_w, dz8w, p_hyd, p_hyd_w, &
         rdz, rdzw, zx, zy, zdm, div, d11, d22, d33, d12, d13, d23, &
         nlflux, elmin, dlk, xkmh, xkmv, xkhh, xkhv, xkmv_meso, bn2, &
         shear, tke_tend, l_diss, ru, rv, rw, rt, dummy, rhs
    real, allocatable :: hfx1(:,:), hfx2(:,:)
    type(grid_config_rec_type) :: cfg

    read(u_in) nx, ny, nz, bnd, isfflx, mixfull, thetam
    read(u_in) dx, dy, dt, c_s, c_k, heat, drag, cf1, cf2, cf3, p_top
    ids = 1; ide = nx + 1; jds = 1; jde = ny + 1; kds = 1; kde = nz + 1
    ims = -4; ime = nx + 6; jms = -4; jme = ny + 6; kms = 1; kme = nz + 1
    its = ids; ite = ide; jts = jds; jte = jde; kts = kds; kte = kde
    rdx = 1. / dx; rdy = 1. / dy

    cfg%km_opt = 5; cfg%diff_opt = 2; cfg%bl_pbl_physics = 0
    cfg%isfflx = isfflx; cfg%c_s = c_s; cfg%c_k = c_k
    cfg%tke_heat_flux = heat; cfg%tke_drag_coefficient = drag
    cfg%mix_full_fields = (mixfull /= 0); cfg%use_theta_m = thetam
    cfg%periodic_x = (bnd == 0); cfg%periodic_y = (bnd == 0)
    cfg%specified = (bnd /= 0)

    allocate(fnm(kms:kme), fnp(kms:kme), dn(kms:kme), dnw(kms:kme), c1h(kms:kme), &
             c2h(kms:kme), c1f(kms:kme), c2f(kms:kme), znw(kms:kme), t_base(kms:kme), &
             qv_base(kms:kme), u_base(kms:kme), v_base(kms:kme))
    read(u_in) fnm, fnp, dn, dnw, c1h, c2h, c1f, c2f, znw, t_base, qv_base, u_base, v_base

    allocate(msf(ims:ime,jms:jme), mut(ims:ime,jms:jme), ust(ims:ime,jms:jme), &
             ustm(ims:ime,jms:jme), hfx(ims:ime,jms:jme), qfx(ims:ime,jms:jme), &
             br(ims:ime,jms:jme), rmol(ims:ime,jms:jme), wspd(ims:ime,jms:jme), &
             u10(ims:ime,jms:jme), v10(ims:ime,jms:jme), gamu(ims:ime,jms:jme), &
             gamv(ims:ime,jms:jme), pblh(ims:ime,jms:jme), kpbl(ims:ime,jms:jme), &
             muu(ims:ime,jms:jme), muv(ims:ime,jms:jme), hfx1(ims:ime,jms:jme), &
             hfx2(ims:ime,jms:jme))
    msf = 1.; mut = 0.; ust = 0.; ustm = 0.; hfx = 0.; qfx = 0.; br = 0.; rmol = 0.
    wspd = 0.; u10 = 0.; v10 = 0.; gamu = 0.; gamv = 0.; pblh = 0.; kpbl = 0
    muu = 0.; muv = 0.
    call read2(u_in, msf, ims, ime, jms, jme, nx, ny)
    call read2(u_in, mut, ims, ime, jms, jme, nx, ny)
    call read2(u_in, ust, ims, ime, jms, jme, nx, ny)
    call read2(u_in, ustm, ims, ime, jms, jme, nx, ny)
    call read2(u_in, hfx, ims, ime, jms, jme, nx, ny)
    call read2(u_in, qfx, ims, ime, jms, jme, nx, ny)
    call read2(u_in, br, ims, ime, jms, jme, nx, ny)
    call read2(u_in, rmol, ims, ime, jms, jme, nx, ny)
    call read2(u_in, wspd, ims, ime, jms, jme, nx, ny)
    call read2(u_in, u10, ims, ime, jms, jme, nx, ny)
    call read2(u_in, v10, ims, ime, jms, jme, nx, ny)
    call read2(u_in, gamu, ims, ime, jms, jme, nx, ny)
    call read2(u_in, gamv, ims, ime, jms, jme, nx, ny)
    call read2(u_in, pblh, ims, ime, jms, jme, nx, ny)

    allocate(u(ims:ime,kms:kme,jms:jme), v(ims:ime,kms:kme,jms:jme), &
             w(ims:ime,kms:kme,jms:jme), ph(ims:ime,kms:kme,jms:jme), &
             phb(ims:ime,kms:kme,jms:jme), t(ims:ime,kms:kme,jms:jme), &
             p(ims:ime,kms:kme,jms:jme), pb(ims:ime,kms:kme,jms:jme), &
             alt(ims:ime,kms:kme,jms:jme), tke(ims:ime,kms:kme,jms:jme), &
             moist(ims:ime,kms:kme,jms:jme,4))
    u = 0.; v = 0.; w = 0.; ph = 0.; phb = 0.; t = 0.; p = 0.; pb = 0.; alt = 0.
    tke = 0.; moist = 0.
    read(u_in) (((u(i,k,j), i=1,nx+1), k=1,nz), j=1,ny)
    read(u_in) (((v(i,k,j), i=1,nx), k=1,nz), j=1,ny+1)
    read(u_in) (((w(i,k,j), i=1,nx), k=1,nz+1), j=1,ny)
    read(u_in) (((ph(i,k,j), i=1,nx), k=1,nz+1), j=1,ny)
    read(u_in) (((phb(i,k,j), i=1,nx), k=1,nz+1), j=1,ny)
    call read3(u_in, t, ims, ime, kms, kme, jms, jme, nx, ny, nz)
    call read3(u_in, p, ims, ime, kms, kme, jms, jme, nx, ny, nz)
    call read3(u_in, pb, ims, ime, kms, kme, jms, jme, nx, ny, nz)
    call read3(u_in, alt, ims, ime, kms, kme, jms, jme, nx, ny, nz)
    read(u_in) (((moist(i,k,j,p_qv), i=1,nx), k=1,nz), j=1,ny)
    read(u_in) (((moist(i,k,j,p_qc), i=1,nx), k=1,nz), j=1,ny)
    read(u_in) (((moist(i,k,j,p_qi), i=1,nx), k=1,nz), j=1,ny)
    call read3(u_in, tke, ims, ime, kms, kme, jms, jme, nx, ny, nz)

    ! Raw halos: the periodic copy.
    call fill3(u, ims, ime, kms, kme, jms, jme, nx, ny, nx+1, ny, 1, nz)
    call fill3(v, ims, ime, kms, kme, jms, jme, nx, ny, nx, ny+1, 1, nz)
    call fill3(w, ims, ime, kms, kme, jms, jme, nx, ny, nx, ny, 1, nz+1)
    call fill3(ph, ims, ime, kms, kme, jms, jme, nx, ny, nx, ny, 1, nz+1)
    call fill3(phb, ims, ime, kms, kme, jms, jme, nx, ny, nx, ny, 1, nz+1)
    call fill3(t, ims, ime, kms, kme, jms, jme, nx, ny, nx, ny, 1, nz)
    call fill3(p, ims, ime, kms, kme, jms, jme, nx, ny, nx, ny, 1, nz)
    call fill3(pb, ims, ime, kms, kme, jms, jme, nx, ny, nx, ny, 1, nz)
    call fill3(alt, ims, ime, kms, kme, jms, jme, nx, ny, nx, ny, 1, nz)
    call fill3(tke, ims, ime, kms, kme, jms, jme, nx, ny, nx, ny, 1, nz)
    do i = 2, 4
      call fill3(moist(:,:,:,i), ims, ime, kms, kme, jms, jme, nx, ny, nx, ny, 1, nz)
    end do
    call fill2(msf, ims, ime, jms, jme, nx, ny)
    call fill2(mut, ims, ime, jms, jme, nx, ny)
    call fill2(ust, ims, ime, jms, jme, nx, ny)
    call fill2(ustm, ims, ime, jms, jme, nx, ny)
    call fill2(hfx, ims, ime, jms, jme, nx, ny)
    call fill2(qfx, ims, ime, jms, jme, nx, ny)
    call fill2(br, ims, ime, jms, jme, nx, ny)
    call fill2(rmol, ims, ime, jms, jme, nx, ny)
    call fill2(wspd, ims, ime, jms, jme, nx, ny)
    call fill2(gamu, ims, ime, jms, jme, nx, ny)
    call fill2(gamv, ims, ime, jms, jme, nx, ny)
    call fill2(pblh, ims, ime, jms, jme, nx, ny)

    allocate(rho, th_phy, th_phy_m_t0, p_phy, pi_phy, u_phy, v_phy, p8w, t_phy, &
             t8w, z, z_at_w, dz8w, p_hyd, p_hyd_w, rdz, rdzw, zx, zy, zdm, div, &
             d11, d22, d33, d12, d13, d23, nlflux, elmin, dlk, xkmh, xkmv, xkhh, &
             xkhv, xkmv_meso, bn2, shear, tke_tend, l_diss, ru, rv, rw, rt, dummy, &
             rhs, mold=u)
    rho = 0.; th_phy = 0.; th_phy_m_t0 = 0.; p_phy = 0.; pi_phy = 0.; u_phy = 0.
    v_phy = 0.; p8w = 0.; t_phy = 0.; t8w = 0.; z = 0.; z_at_w = 0.; dz8w = 0.
    p_hyd = 0.; p_hyd_w = 0.; rdz = 0.; rdzw = 0.; zx = 0.; zy = 0.; zdm = 0.
    div = 0.; d11 = 0.; d22 = 0.; d33 = 0.; d12 = 0.; d13 = 0.; d23 = 0.
    nlflux = 0.; elmin = 0.; dlk = 0.; xkmh = 0.; xkmv = 0.; xkhh = 0.; xkhv = 0.
    xkmv_meso = 0.; bn2 = 0.; shear = 0.; tke_tend = 0.; l_diss = 0.
    ru = 0.; rv = 0.; rw = 0.; rt = 0.; dummy = 0.
    allocate(moist_tend(ims:ime,kms:kme,jms:jme,4), nba(ims:ime,kms:kme,jms:jme,3), &
             chem(ims:ime,kms:kme,jms:jme,1), chem_tend(ims:ime,kms:kme,jms:jme,1), &
             scal(ims:ime,kms:kme,jms:jme,1), scal_tend(ims:ime,kms:kme,jms:jme,1), &
             trac(ims:ime,kms:kme,jms:jme,1), trac_tend(ims:ime,kms:kme,jms:jme,1))
    moist_tend = 0.; nba = 0.; chem = 0.; chem_tend = 0.; scal = 0.; scal_tend = 0.
    trac = 0.; trac_tend = 0.

    call phy_prep(cfg, mut, muu, muv, c1h, c2h, c1f, c2f, u, v, p, pb, alt, ph, &
                  phb, t, moist, 4, rho, th_phy, th_phy_m_t0, p_phy, pi_phy, &
                  u_phy, v_phy, p8w, t_phy, t8w, z, z_at_w, dz8w, p_hyd, p_hyd_w, &
                  dnw, fnm, fnp, znw, p_top, &
                  ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
                  its, ite, jts, jte, kts, kte)

    call compute_diff_metrics(cfg, ph, phb, zdm, rdz, rdzw, zx, zy, rdx, rdy, &
                  ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
                  its, ite, jts, jte, kts, kte)
    call set_physical_bc3d(rdzw, 'w', cfg, ids, ide, jds, jde, kds, kde, ims, ime, &
         jms, jme, kms, kme, ids, ide, jds, jde, kds, kde, its, ite, jts, jte, kts, kte)
    call set_physical_bc3d(rdz, 'w', cfg, ids, ide, jds, jde, kds, kde, ims, ime, &
         jms, jme, kms, kme, ids, ide, jds, jde, kds, kde, its, ite, jts, jte, kts, kte)
    call set_physical_bc3d(zdm, 'w', cfg, ids, ide, jds, jde, kds, kde, ims, ime, &
         jms, jme, kms, kme, ids, ide, jds, jde, kds, kde, its, ite, jts, jte, kts, kte)
    call set_physical_bc3d(zx, 'e', cfg, ids, ide, jds, jde, kds, kde, ims, ime, &
         jms, jme, kms, kme, ids, ide, jds, jde, kds, kde, its, ite, jts, jte, kts, kte)
    call set_physical_bc3d(zy, 'f', cfg, ids, ide, jds, jde, kds, kde, ims, ime, &
         jms, jme, kms, kme, ids, ide, jds, jde, kds, kde, its, ite, jts, jte, kts, kte)
    call set_physical_bc2d(ustm, 't', cfg, ids, ide, jds, jde, ims, ime, jms, jme, &
         ids, ide, jds, jde, its, ite, jts, jte)
    call set_physical_bc2d(ust, 't', cfg, ids, ide, jds, jde, ims, ime, jms, jme, &
         ids, ide, jds, jde, its, ite, jts, jte)

    call cal_deform_and_div(cfg, u, v, w, div, d11, d22, d33, d12, d13, d23, &
                  nba, 3, u_base, v_base, msf, msf, msf, msf, msf, msf, &
                  rdx, rdy, dn, dnw, rdz, rdzw, fnm, fnp, cf1, cf2, cf3, zx, zy, &
                  ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
                  its, ite, jts, jte, kts, kte)

    call nonlocal_flux(cfg, nlflux, gamu, gamv, pblh, kpbl, dx, dy, dt, ust, hfx, &
                  qfx, br, ep_1, ep_2, karman, u_phy, v_phy, th_phy, rho, moist, 4, &
                  msf, msf, rdzw, u10, v10, wspd, &
                  ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
                  its, ite, jts, jte, kts, kte)
    hfx1 = hfx
    call free_atmos_length(cfg, dx, dy, rdzw, rdz, tke, th_phy, elmin, hfx, qfx, &
                  moist, 4, &
                  ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
                  its, ite, jts, jte, kts, kte)
    call meso_length_scale(cfg, dx, dy, rdzw, rdz, tke, p8w, t8w, th_phy, dlk, &
                  pblh, elmin, rmol, rho, hfx, qfx, moist, 4, &
                  ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
                  its, ite, jts, jte, kts, kte)
    hfx2 = hfx
    call calculate_km_kh(cfg, dt, 0., 0., 0, xkmh, xkmv, xkhh, xkhv, bn2, 0., 0., &
                  div, d11, d22, d33, d12, d13, d23, tke, p8w, t8w, th_phy, t_phy, &
                  p_phy, moist, dn, dnw, dx, dy, rdz, rdzw, 0, 4, cf1, cf2, cf3, &
                  .false., 0.1, msf, msf, zx, zy, pblh, dlk, xkmv_meso, &
                  ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
                  its, ite, jts, jte, kts, kte)
    call phy_bc(cfg, div, d11, d22, d33, d12, d13, d23, xkmh, xkmv, xkhh, xkhv, &
                  tke, rho, dummy, dummy, dummy, dummy, dummy, dummy, &
                  gamu, gamv, xkmv_meso, &
                  ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
                  ids, ide, jds, jde, kds, kde, its, ite, jts, jte, kts, kte)

    call tke_shear(shear, cfg, d11, d22, d33, d12, d13, d23, u, v, w, tke, ustm, &
                  mut, c1h, c2h, fnm, fnp, cf1, cf2, cf3, msf, msf, xkmh, xkmv, &
                  rdx, rdy, zx, zy, rdz, rdzw, dn, dnw, &
                  ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
                  its, ite, jts, jte, kts, kte)
    call tke_rhs(tke_tend, bn2, cfg, d11, d22, d33, d12, d13, d23, u, v, w, div, &
                  tke, mut, c1h, c2h, th_phy, p_phy, p8w, t8w, zdm, fnm, fnp, &
                  cf1, cf2, cf3, msf, msf, xkmh, xkmv, xkhv, rdx, rdy, dx, dy, dt, &
                  zx, zy, rdz, rdzw, dn, dnw, 0, hfx, qfx, moist(ims,kms,jms,p_qv), &
                  ustm, rho, l_diss, nlflux, pblh, dlk, &
                  ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
                  its, ite, jts, jte, kts, kte)
    rhs = tke_tend
    call vertical_diffusion_implicit(ru, rv, rw, rt, tke_tend, moist_tend, 4, &
                  chem_tend, 1, scal_tend, 1, trac_tend, 1, u, v, w, dt, t, &
                  u_base, v_base, t_base, qv_base, mut, tke, th_phy, cfg, moist, &
                  chem, scal, trac, xkmv, xkhv, xkmh, 5, fnm, fnp, dn, dnw, rdz, &
                  rdzw, hfx, qfx, ustm, rho, nlflux, gamu, gamv, xkmv_meso, l_diss, &
                  c1h, c2h, c1f, c2f, &
                  ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
                  its, ite, jts, jte, kts, kte)

    ! Output, in this order (make_cases.py ORACLE_FIELDS).
    write(u_out) (((rdzw(i,k,j), i=1,nx), k=1,nz), j=1,ny)
    write(u_out) (((rdz(i,k,j), i=1,nx), k=1,nz), j=1,ny)
    write(u_out) (((zx(i,k,j), i=1,nx+1), k=1,nz+1), j=1,ny)
    write(u_out) (((zy(i,k,j), i=1,nx), k=1,nz+1), j=1,ny+1)
    call write3(u_out, th_phy, ims, ime, kms, kme, jms, jme, nx, ny, nz)
    call write3(u_out, rho, ims, ime, kms, kme, jms, jme, nx, ny, nz)
    call write3(u_out, u_phy, ims, ime, kms, kme, jms, jme, nx, ny, nz)
    call write3(u_out, v_phy, ims, ime, kms, kme, jms, jme, nx, ny, nz)
    write(u_out) ((p8w(i,1,j), i=1,nx), j=1,ny)
    write(u_out) ((t8w(i,1,j), i=1,nx), j=1,ny)
    write(u_out) ((p8w(i,kde,j), i=1,nx), j=1,ny)
    write(u_out) ((t8w(i,kde,j), i=1,nx), j=1,ny)
    call write3(u_out, d11, ims, ime, kms, kme, jms, jme, nx, ny, nz)
    call write3(u_out, d22, ims, ime, kms, kme, jms, jme, nx, ny, nz)
    call write3(u_out, d12, ims, ime, kms, kme, jms, jme, nx, ny, nz)
    write(u_out) (((nlflux(i,k,j), i=1,nx), k=1,nz+1), j=1,ny)
    call write2(u_out, gamu, ims, ime, jms, jme, nx, ny)
    call write2(u_out, gamv, ims, ime, jms, jme, nx, ny)
    call write2(u_out, pblh, ims, ime, jms, jme, nx, ny)
    write(u_out) ((kpbl(i,j), i=1,nx), j=1,ny)
    call write2(u_out, hfx1, ims, ime, jms, jme, nx, ny)
    call write3(u_out, elmin, ims, ime, kms, kme, jms, jme, nx, ny, nz)
    call write3(u_out, dlk, ims, ime, kms, kme, jms, jme, nx, ny, nz)
    call write2(u_out, hfx2, ims, ime, jms, jme, nx, ny)
    call write3(u_out, xkmh, ims, ime, kms, kme, jms, jme, nx, ny, nz)
    call write3(u_out, xkmv, ims, ime, kms, kme, jms, jme, nx, ny, nz)
    call write3(u_out, xkhh, ims, ime, kms, kme, jms, jme, nx, ny, nz)
    call write3(u_out, xkhv, ims, ime, kms, kme, jms, jme, nx, ny, nz)
    call write3(u_out, xkmv_meso, ims, ime, kms, kme, jms, jme, nx, ny, nz)
    call write3(u_out, bn2, ims, ime, kms, kme, jms, jme, nx, ny, nz)
    call write3(u_out, shear, ims, ime, kms, kme, jms, jme, nx, ny, nz)
    call write3(u_out, rhs, ims, ime, kms, kme, jms, jme, nx, ny, nz)
    call write3(u_out, l_diss, ims, ime, kms, kme, jms, jme, nx, ny, nz)
    write(u_out) (((ru(i,k,j), i=1,nx+1), k=1,nz), j=1,ny)
    write(u_out) (((rv(i,k,j), i=1,nx), k=1,nz), j=1,ny+1)
    write(u_out) (((rw(i,k,j), i=1,nx), k=1,nz+1), j=1,ny)
    call write3(u_out, rt, ims, ime, kms, kme, jms, jme, nx, ny, nz)
    call write3(u_out, tke_tend, ims, ime, kms, kme, jms, jme, nx, ny, nz)
    do i = 2, 4
      write(u_out) (((moist_tend(ii, k, j, i), ii=1,nx), k=1,nz), j=1,ny)
    end do
    call write2(u_out, hfx, ims, ime, jms, jme, nx, ny)

  end subroutine run_case
end program run_sms3dtke
