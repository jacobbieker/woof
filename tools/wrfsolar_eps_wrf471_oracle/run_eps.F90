! Driver for WRF v4.7.1's WRF-Solar EPS scheme hooks (multi_perturb = 1),
! called as the drivers call them:
!   FARMS     module_radiation_driver.F :3052-3082  Add -> farms_driver -> Remove
!   Deng      module_shallowcu_driver.F :632-672    Add -> deng_shcu_driver -> Remove
!   Thompson  module_microphysics_driver.F :1120-1213 (mp 28), :1314-1381 (mp 8)
!                                                   Add -> mp_gt_driver -> Remove
!   icloud=3  module_radiation_driver.F :1472-1478  t_1d, qv_1d for cal_cldfra3
! farms_driver and deng_shcu_driver write none of the perturbed arrays, so
! their Remove runs on Add's output; mp_gt_driver does, so the Thompson
! Remove runs on supplied post-microphysics values (mp_*), as WRF's would on
! the scheme's output.
!
! One tile, j = 1: i = 1..ni, mass levels k = 1..nk (kts..kte), memory
! levels 1..nk+1 (kms..kme; WRF's pert3d and w carry the staggered top).
!
! Input inputs.bin (stream, little endian):
!   int ni, nk, ncase
!   per case: real farms(7) [albedo aod angexp aerasy qv qc qs],
!             deng(4) [qv qc t w], thom(5) [qv qc qi qs ni], cld3(2) [qv t]
!   real(ni, nk+1) perts: albedo aod angstrom assymfac qvapor qcloud qice
!                         qsnow ni th w
!   real(ni, nk+1) state: qv qc qi qs ni th t w
!   real(ni)       state: albedo aod angexp aerasy
!   real(ni, nk+1) post-microphysics: qv qc qi qs ni
! Output eps.out.bin, per case:
!   FARMS after Add:    albedo aod angexp aerasy (ni), qv qc qs, qv_tmp qc_tmp qs_tmp
!   FARMS after Remove: albedo aod angexp aerasy (ni), qv qc qs
!   Deng after Add:     qv qc th w;  after Remove: qv qc th w
!   Thompson after Add: qv qc qi qs ni, five tmp;  after Remove: qv qc qi qs ni
!   cld3:               t_1d qv_1d
! 3D arrays are written over the mass levels (ni, nk), w over (ni, nk+1);
! the _tmp arrays are WRF's (its:ite, kts:kte, jts:jte) ones.
program run_eps
  use eps_rad_cut
  use eps_shcu_cut
  use eps_mp_cut
  use eps_cld3_cut
  implicit none
  integer :: ni, nk, ncase, c, u, v, kme
  real, allocatable :: fac(:, :)
  real, allocatable :: p_alb(:,:,:), p_aod(:,:,:), p_ang(:,:,:), p_asy(:,:,:), &
      p_qv(:,:,:), p_qc(:,:,:), p_qi(:,:,:), p_qs(:,:,:), p_ni(:,:,:), p_th(:,:,:), p_w(:,:,:)
  real, allocatable :: s_qv(:,:,:), s_qc(:,:,:), s_qi(:,:,:), s_qs(:,:,:), s_ni(:,:,:), &
      s_th(:,:,:), s_t(:,:,:), s_w(:,:,:)
  real, allocatable :: s_alb(:,:), s_aod(:,:), s_ang(:,:), s_asy(:,:)
  real, allocatable :: m_qv(:,:,:), m_qc(:,:,:), m_qi(:,:,:), m_qs(:,:,:), m_ni(:,:,:)
  real, allocatable :: qv(:,:,:), qc(:,:,:), qi(:,:,:), qs(:,:,:), qn(:,:,:), th(:,:,:), w(:,:,:)
  real, allocatable :: alb(:,:), aod(:,:), ang(:,:), asy(:,:)
  real, allocatable :: t1(:,:,:), t2(:,:,:), t3(:,:,:), t4(:,:,:), t5(:,:,:)
  real, allocatable :: to(:,:,:), qo(:,:,:)

  open(newunit=u, file='inputs.bin', access='stream', form='unformatted', &
       status='old', convert='little_endian')
  open(newunit=v, file='eps.out.bin', access='stream', form='unformatted', &
       status='replace', convert='little_endian')
  read(u) ni, nk, ncase
  kme = nk + 1
  allocate(fac(18, ncase))
  read(u) fac
  allocate(p_alb(ni,kme,1), p_aod(ni,kme,1), p_ang(ni,kme,1), p_asy(ni,kme,1), &
      p_qv(ni,kme,1), p_qc(ni,kme,1), p_qi(ni,kme,1), p_qs(ni,kme,1), p_ni(ni,kme,1), &
      p_th(ni,kme,1), p_w(ni,kme,1))
  read(u) p_alb, p_aod, p_ang, p_asy, p_qv, p_qc, p_qi, p_qs, p_ni, p_th, p_w
  allocate(s_qv(ni,kme,1), s_qc(ni,kme,1), s_qi(ni,kme,1), s_qs(ni,kme,1), &
      s_ni(ni,kme,1), s_th(ni,kme,1), s_t(ni,kme,1), s_w(ni,kme,1))
  read(u) s_qv, s_qc, s_qi, s_qs, s_ni, s_th, s_t, s_w
  allocate(s_alb(ni,1), s_aod(ni,1), s_ang(ni,1), s_asy(ni,1))
  read(u) s_alb, s_aod, s_ang, s_asy
  allocate(m_qv(ni,kme,1), m_qc(ni,kme,1), m_qi(ni,kme,1), m_qs(ni,kme,1), m_ni(ni,kme,1))
  read(u) m_qv, m_qc, m_qi, m_qs, m_ni
  close(u)
  allocate(qv(ni,kme,1), qc(ni,kme,1), qi(ni,kme,1), qs(ni,kme,1), qn(ni,kme,1), &
      th(ni,kme,1), w(ni,kme,1), alb(ni,1), aod(ni,1), ang(ni,1), asy(ni,1), &
      t1(ni,nk,1), t2(ni,nk,1), t3(ni,nk,1), t4(ni,nk,1), t5(ni,nk,1), &
      to(ni,kme,1), qo(ni,kme,1))

  do c = 1, ncase
    ! ---- FARMS ----
    alb = s_alb; aod = s_aod; ang = s_ang; asy = s_asy
    qv = s_qv; qc = s_qc; qs = s_qs
    call Add_multi_perturb_swrad_perturbations (p_alb, p_aod, p_ang, &
        p_asy, p_qv, p_qc, p_qs, fac(1,c), fac(2,c), &
        fac(3,c), fac(4,c), fac(5,c), fac(6,c), fac(7,c), &
        alb, aod, ang, asy, qv, qc, qs, t1, t2, t3, 1, ni, 1, 1, &
        1, ni, 1, 1, 1, kme, 1, nk)
    write(v) alb, aod, ang, asy, qv(:,1:nk,:), qc(:,1:nk,:), qs(:,1:nk,:), t1, t2, t3
    call Remove_multi_perturb_swrad_perturbations (p_alb, p_aod, p_ang, &
        p_asy, p_qv, p_qc, p_qs, fac(1,c), fac(2,c), &
        fac(3,c), fac(4,c), fac(5,c), fac(6,c), fac(7,c), &
        alb, aod, ang, asy, qv, qc, qs, t1, t2, t3, 1, ni, 1, 1, &
        1, ni, 1, 1, 1, kme, 1, nk)
    write(v) alb, aod, ang, asy, qv(:,1:nk,:), qc(:,1:nk,:), qs(:,1:nk,:)

    ! ---- Deng ----
    qv = s_qv; qc = s_qc; th = s_th; w = s_w
    call Add_multi_perturb_shcu_perturbations (p_qv, p_qc, &
        p_th, p_w, fac(8,c), fac(9,c), fac(10,c), fac(11,c), &
        th, qv, qc, w, 1, ni, 1, 1, 1, ni, 1, 1, 1, kme, &
        1, nk)
    write(v) qv(:,1:nk,:), qc(:,1:nk,:), th(:,1:nk,:), w
    call Remove_multi_perturb_shcu_perturbations (p_qv, p_qc, &
        p_th, p_w, fac(8,c), fac(9,c), fac(10,c), fac(11,c), &
        th, qv, qc, w, 1, ni, 1, 1, 1, ni, 1, 1, 1, kme, &
        1, nk)
    write(v) qv(:,1:nk,:), qc(:,1:nk,:), th(:,1:nk,:), w

    ! ---- Thompson ----
    qv = s_qv; qc = s_qc; qi = s_qi; qs = s_qs; qn = s_ni
    call Add_multi_perturb_mp_perturbations (p_qv, p_qc, p_qi, &
        p_qs, p_ni, fac(12,c), fac(13,c), fac(14,c), fac(15,c), &
        fac(16,c), qv, qc, qi, qs, qn, t1, t2, t3, &
        t4, t5, 1, ni, 1, 1, 1, ni, 1, 1, 1, kme, 1, nk)
    write(v) qv(:,1:nk,:), qc(:,1:nk,:), qi(:,1:nk,:), qs(:,1:nk,:), qn(:,1:nk,:), &
        t1, t2, t3, t4, t5
    qv = m_qv; qc = m_qc; qi = m_qi; qs = m_qs; qn = m_ni
    call Remove_multi_perturb_mp_perturbations (p_qv, p_qc, p_qi, &
        p_qs, p_ni, fac(12,c), fac(13,c), fac(14,c), fac(15,c), &
        fac(16,c), qv, qc, qi, qs, qn, t1, t2, t3, &
        t4, t5, 1, ni, 1, 1, 1, ni, 1, 1, 1, kme, 1, nk)
    write(v) qv(:,1:nk,:), qc(:,1:nk,:), qi(:,1:nk,:), qs(:,1:nk,:), qn(:,1:nk,:)

    ! ---- icloud = 3 inputs ----
    call cld3_inputs(s_t, s_qv, p_th, p_qv, .true., 1, fac(18,c), fac(17,c), &
        to, qo, 1, ni, 1, nk, 1, 1)
    write(v) to(:,1:nk,:), qo(:,1:nk,:)
  end do
  close(v)
end program run_eps
