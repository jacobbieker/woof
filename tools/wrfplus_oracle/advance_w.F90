! WRFPLUS 4.7.1 oracle driver: g_advance_w then g_calc_p_rho (step 0,
! non-hydrostatic), and a_calc_p_rho then a_advance_w, on one input record
! (docs/dev/P3A-WRF-DRYCORE-ADJOINT.md section 4).  Built against WRFPLUS's
! libwrflib.a (configure wrfplus, real*8, -O0 -ffp-contract=off) by
! tools/wrfplus_oracle/build.sh; driven by tools/wrfplus_oracle/fixture.py.
!
! Input  (stream, native binary64/int32): see fixture.py _w_record.
! Output (stream): TL w, wd, ph, phd, p, pd, al, ald; then the AD
! adjoints in fixture.py W_AD3 / W_AD2 order, then w_save's adjoint.
! damp_opt = 3 (WRF's implicit w damper) when the record's dampcoef > 0,
! with w_save (and its perturbation) from the record.
program wrfplus_advance_w
  use module_configure, only : grid_config_rec_type
  use g_module_small_step_em, only : g_advance_w, g_calc_p_rho
  use a_module_small_step_em, only : a_advance_w, a_calc_p_rho
  implicit none
  integer :: nx, ny, nz, itop, iu
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
  real :: rdx, rdy, dts, cf1, cf2, cf3, epssm, dampcoef, zdamp
  real, parameter :: t0 = 300.
  type(grid_config_rec_type) :: cf
  logical :: top_lid
  real, allocatable, dimension(:,:,:) :: w, wd, ph, phd, rw, rwd, ww, wwd, th, thd, &
       thold, tholdd, thp, thpd, php, phpd, phb, rph, rphd, c2a, c2ad, cqw, cqwd, &
       alt, altd, a, ad, alpha, alphad, gam, gamd, u, ud, v, vd, sw, sph, sp, sal
  real, allocatable, dimension(:,:,:) :: p, pd, al, ald, pm1, pm1d, wsave, wsaved, alb, &
       w0, ph0, thold0, p1, al1, ph1
  real, allocatable, dimension(:,:,:) :: a_rw, a_ww, a_wsave, a_th, a_thold, a_thp, &
       a_php, a_rph, a_c2a, a_cqw, a_alt, a_a, a_alpha, a_gam, a_u, a_v, a_pm1
  real, allocatable, dimension(:,:) :: mut, mutd, muts, mutsd, muave, muaved, mu, mud, &
       ht, msft, a_mut, a_muts, a_muave, a_mu, a_mu1
  real, allocatable, dimension(:) :: fnm, fnp, rdnw, rdn, dnw, znu
  character(len=512) :: fin, fout

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(newunit=iu, file=trim(fin), access='stream', form='unformatted', status='old')
  read(iu) nx, ny, nz, itop
  read(iu) rdx, rdy, dts, cf1, cf2, cf3, epssm
  top_lid = itop /= 0
  ids = 1; ide = nx + 1; jds = 1; jde = ny + 1; kds = 1; kde = nz + 1
  ims = 0; ime = nx + 2; jms = 0; jme = ny + 2; kms = 1; kme = nz + 1
  call a3(w); call a3(wd); call a3(ph); call a3(phd); call a3(rw); call a3(rwd)
  call a3(ww); call a3(wwd); call a3(th); call a3(thd); call a3(thold); call a3(tholdd)
  call a3(thp); call a3(thpd); call a3(php); call a3(phpd); call a3(phb); call a3(rph)
  call a3(rphd); call a3(c2a); call a3(c2ad); call a3(cqw); call a3(cqwd); call a3(alt)
  call a3(altd); call a3(a); call a3(ad); call a3(alpha); call a3(alphad); call a3(gam)
  call a3(gamd); call a3(u); call a3(ud); call a3(v); call a3(vd)
  call a3(sw); call a3(sph); call a3(sp); call a3(sal)
  call a2(mut); call a2(mutd); call a2(muts); call a2(mutsd); call a2(muave)
  call a2(muaved); call a2(mu); call a2(mud); call a2(ht); call a2(msft)
  allocate(fnm(kms:kme), fnp(kms:kme), rdnw(kms:kme), rdn(kms:kme))
  read(iu) fnm, fnp, rdnw, rdn
  read(iu) dampcoef, zdamp
  call a3(wsave); call a3(wsaved)
  close(iu)
  allocate(dnw(kms:kme), znu(kms:kme))
  dnw = 0; where (rdnw /= 0) dnw = 1. / rdnw
  znu = 0

  cf%specified = .true.;   cf%nested = .false.;   cf%periodic_x = .false.
  cf%open_xs = .false.;    cf%open_xe = .false.;  cf%open_ys = .false.
  cf%open_ye = .false.;    cf%symmetric_xs = .false.; cf%symmetric_xe = .false.
  cf%symmetric_ys = .false.; cf%symmetric_ye = .false.; cf%polar = .false.
  cf%damp_opt = 0; cf%dampcoef = dampcoef; cf%zdamp = zdamp
  if (dampcoef > 0.) cf%damp_opt = 3

  ! msftx = msfty = msft: woof carries one mass-point factor (ones
  ! without map factors).
  allocate(p, pd, al, ald, pm1, pm1d, alb, mold=w)
  p = 0; pd = 0; al = 0; ald = 0; pm1 = 0; pm1d = 0; alb = 0
  allocate(w0, source=w); allocate(ph0, source=ph); allocate(thold0, source=thold)

  call g_advance_w(w, wd, rw, rwd, ww, wwd, wsave, wsaved, u, ud, v, vd, mu, mut, &
       mutd, muave, muaved, muts, mutsd, thold, tholdd, th, thd, thp, thpd, ph, phd, &
       php, phpd, phb, rph, rphd, ht, c2a, c2ad, cqw, cqwd, alt, altd, alb, a, ad, &
       alpha, alphad, gam, gamd, rdx, rdy, dts, t0, epssm, dnw, fnm, fnp, rdnw, rdn, &
       cf1, cf2, cf3, msft, msft, cf, top_lid, ids, ide, jds, jde, kds, kde, ims, &
       ime, jms, jme, kms, kme, ids, ide, jds, jde, kds, kde)
  call g_calc_p_rho(al, ald, p, pd, ph, phd, alt, altd, th, thd, thp, thpd, c2a, c2ad, &
       pm1, pm1d, mu, mud, muts, mutsd, znu, t0, rdnw, dnw, 0., .true., 0, ids, ide, &
       jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ids, ide, jds, jde, kds, kde)

  open(newunit=iu, file=trim(fout), access='stream', form='unformatted', status='replace')
  write(iu) w, wd, ph, phd, p, pd, al, ald

  ! Adjoint: a_calc_p_rho reads the post-advance_w ph; a_advance_w the
  ! inputs (w0, ph0, thold0).  sw/sph/sp/sal are the output adjoints.
  allocate(ph1, source=ph); allocate(p1, source=p); allocate(al1, source=al)
  allocate(a_rw, a_ww, a_wsave, a_th, a_thold, a_thp, a_php, a_rph, a_c2a, a_cqw, &
       a_alt, a_a, a_alpha, a_gam, a_u, a_v, a_pm1, mold=w)
  a_rw = 0; a_ww = 0; a_wsave = 0; a_th = 0; a_thold = 0; a_thp = 0; a_php = 0
  a_rph = 0; a_c2a = 0; a_cqw = 0; a_alt = 0; a_a = 0; a_alpha = 0; a_gam = 0
  a_u = 0; a_v = 0; a_pm1 = 0
  allocate(a_mut, a_muts, a_muave, a_mu, a_mu1, mold=mut)
  a_mut = 0; a_muts = 0; a_muave = 0; a_mu = 0; a_mu1 = 0
  call a_calc_p_rho(al1, sal, p1, sp, ph1, sph, alt, a_alt, th, a_th, thp, a_thp, c2a, &
       a_c2a, pm1, a_pm1, mu, a_mu, muts, a_muts, znu, t0, rdnw, dnw, 0., .true., 0, &
       ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ids, ide, jds, jde, &
       kds, kde)
  call a_advance_w(w0, sw, rw, a_rw, ww, a_ww, wsave, a_wsave, u, a_u, v, a_v, mu, &
       a_mu1, mut, a_mut, muave, a_muave, muts, a_muts, thold0, a_thold, th, a_th, &
       thp, a_thp, ph0, sph, php, a_php, phb, rph, a_rph, ht, c2a, a_c2a, cqw, a_cqw, &
       alt, a_alt, alb, a, a_a, alpha, a_alpha, gam, a_gam, rdx, rdy, dts, t0, epssm, &
       dnw, fnm, fnp, rdnw, rdn, cf1, cf2, cf3, msft, msft, cf, top_lid, ids, ide, &
       jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ids, ide, jds, jde, kds, kde)

  write(iu) sw, sph, sp, sal, a_rw, a_ww, a_th, a_thold, a_thp, a_php, a_rph, a_c2a, &
       a_cqw, a_alt, a_a, a_alpha, a_gam, a_u, a_v
  write(iu) a_mut, a_muts, a_muave, a_mu, a_mu1
  write(iu) a_wsave
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

end program wrfplus_advance_w
