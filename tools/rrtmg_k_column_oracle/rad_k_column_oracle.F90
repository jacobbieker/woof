! Column oracle for WRF 4.7.1 RRTMG-K (ra_lw_physics = ra_sw_physics = 14)
! driven the way phys/module_radiation_driver.F drives it on one radiation
! step, over ncol independent columns (i = 1..ncol, j = 1).
!
! Physics comes only from unmodified WRF text, compiled whole with
! -DBUILD_RRTMK=1:
!   rrtmg_lwinit_k                     phys/module_ra_rrtmg_lwk.F
!   rrtmg_swinit_k, rad_rrtmg_driver   phys/module_ra_rrtmg_swk.F
!   effectRad, rgmma                   phys/module_ra_effective_radius.F
!   radconst, calc_coszen, cal_cldfra1, ozn_time_int, ozn_p_int, oznini,
!   lin_interpol2                      extracted verbatim (build_units.py of
!                                      tools/rrtmg_legacy_column_oracle)
!
! The glue statements are the driver's own lines, in its order (the same
! glue as the legacy column oracle, rad_column_oracle.F90):
!   phy_prep t8w, radconst/coszen, CLDFRA zero + cal_cldfra1 (ICLOUD = 1),
!   ozone (o3input 2), GSW/GLW/SWDOWN/RTHRATEN zero,
!   CASE (RRTMK_LWSCHEME): rad_rrtmg_driver(RTHRATEN, RTHRATENSW, ...,
!       coszen, solcon, albedo, emiss, t, t8w, tsk, rho, p, p8w, cldfra,
!       r_d, g, qnc_curr, xland, f_qnc, ...)  module_radiation_driver.F:2104-2131
!   RTHRATENLW = RTHRATEN                      :2252-2262
!   CASE (RRTMK_SWSCHEME): RTHRATEN += RTHRATENSW   :2702-2710
!   RTHRATENSW = RTHRATEN - RTHRATENLW, SWDOWN = GSW/(1-ALBEDO)
!
! Diagnostics beyond the driver's fields (never fed back): cdm2 =
! rgmma(5/3), effectRad's radii on the same inputs after the driver's
! clamps, and the K shortwave liquid optics tables.
!
! Input (little-endian stream): the legacy oracle's layout, then qnc.
! hdr(23) = f_qnc.  Output: named records (char*32 name, int32 count,
! float32 data) until EOF.
program rad_k_column_oracle
  use module_ra_rrtmg_lwk, only: rrtmg_lwinit_k
  use module_ra_rrtmg_swk, only: rrtmg_swinit_k, rad_rrtmg_driver
  use module_ra_effective_radius, only: effectRad, rgmma
  use rrsw_cld_k, only: extliq1, ssaliq1, asyliq1
  use wrf461_raddrv_extract, only: radconst, calc_coszen, cal_cldfra1, &
                                   ozn_time_int, ozn_p_int, oznini
  implicit none

  integer, parameter :: levsiz = 59, num_months = 13
  integer :: hdr(24)
  real :: rhdr(8)
  integer :: ncol, nz, mp_physics
  logical :: f_qv, f_qc, f_qr, f_qi, f_qs, f_qg, f_qnc
  integer :: o3input, aer_opt, julday, yr, icloud
  real :: julian, xtime, gmt, radt, p_top
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
  integer :: its, ite, jts, jte, kts, kte, i, k, j, kde_, k_end
  real :: declin, solcon, degrad, dpd, z0, z1, z2, w1, w2, cdm2
  real, parameter :: piconst = 3.1415926535897932384626433
  character(len=1024) :: inpath, outpath
  integer :: iu, ou

  real, allocatable, dimension(:,:,:) :: p3d, t3d, pi3d, dz8w, rho, qv, qc, qr, &
       qi, qs, qg, re_cloud, re_ice, re_snow, nwfa, nifa, qnc, p8w, z_at_w, t8w, &
       zhalf, cldfra, o3rad, rthraten, rthratenlw, rthratensw, &
       rthratenlwc, rthratenswc, f_ice_phy, f_rain_phy, rqc, rqi, rqs
  integer, allocatable :: cldfra1_flag(:,:,:)
  real, allocatable, dimension(:,:) :: tsk, emiss, albedo, xland, xice, snow, &
       xlat, xlong, ht, coszen, hrang, glw, gsw, swdown, olr, lwcf, swcf, &
       lwupt, lwuptc, lwdnt, lwdntc, lwupb, lwupbc, lwdnb, lwdnbc, &
       swupt, swuptc, swdnt, swdntc, swupb, swupbc, swdnb, swdnbc
  real, allocatable :: fnm(:), fnp(:), qccps(:), re1(:), re2(:), re3(:)
  real, allocatable :: ozmixm(:,:,:,:), ozmixt(:,:,:), pin(:)

  call get_command_argument(1, inpath)
  call get_command_argument(2, outpath)
  open(newunit=iu, file=trim(inpath), access='stream', form='unformatted', &
       status='old', convert='little_endian')
  read(iu) hdr
  read(iu) rhdr
  ncol = hdr(1); nz = hdr(2); mp_physics = hdr(3)
  f_qv = hdr(7) /= 0; f_qc = hdr(8) /= 0; f_qr = hdr(9) /= 0
  f_qi = hdr(10) /= 0; f_qs = hdr(11) /= 0; f_qg = hdr(12) /= 0
  o3input = hdr(14); aer_opt = hdr(15)
  julday = hdr(17); yr = hdr(18); icloud = hdr(19)
  f_qnc = hdr(23) /= 0
  julian = rhdr(1); xtime = rhdr(2); gmt = rhdr(3); radt = rhdr(4); p_top = rhdr(5)

  ids = 1; ide = ncol + 1; jds = 1; jde = 2; kds = 1; kde = nz + 1
  ims = 1; ime = ncol; jms = 1; jme = 1; kms = 1; kme = nz + 1
  its = 1; ite = ncol; jts = 1; jte = 1; kts = 1; kte = nz

  call alloc3(p3d); call alloc3(t3d); call alloc3(pi3d); call alloc3(dz8w)
  call alloc3(rho); call alloc3(qv); call alloc3(qc); call alloc3(qr)
  call alloc3(qi); call alloc3(qs); call alloc3(qg); call alloc3(re_cloud)
  call alloc3(re_ice); call alloc3(re_snow); call alloc3(nwfa); call alloc3(nifa)
  call alloc3(qnc); call alloc3(p8w); call alloc3(z_at_w); call alloc3(t8w)
  call alloc3(zhalf); call alloc3(cldfra); call alloc3(o3rad)
  call alloc3(rthraten); call alloc3(rthratenlw); call alloc3(rthratensw)
  call alloc3(rthratenlwc); call alloc3(rthratenswc); call alloc3(f_ice_phy)
  call alloc3(f_rain_phy); call alloc3(rqc); call alloc3(rqi); call alloc3(rqs)
  allocate(cldfra1_flag(ims:ime, kms:kme, jms:jme)); cldfra1_flag = 0
  call alloc2(tsk); call alloc2(emiss); call alloc2(albedo); call alloc2(xland)
  call alloc2(xice); call alloc2(snow); call alloc2(xlat); call alloc2(xlong)
  call alloc2(ht); call alloc2(coszen); call alloc2(hrang); call alloc2(glw)
  call alloc2(gsw); call alloc2(swdown); call alloc2(olr); call alloc2(lwcf)
  call alloc2(swcf); call alloc2(lwupt); call alloc2(lwuptc)
  call alloc2(lwdnt); call alloc2(lwdntc); call alloc2(lwupb)
  call alloc2(lwupbc); call alloc2(lwdnb); call alloc2(lwdnbc)
  call alloc2(swupt); call alloc2(swuptc); call alloc2(swdnt); call alloc2(swdntc)
  call alloc2(swupb); call alloc2(swupbc); call alloc2(swdnb); call alloc2(swdnbc)
  allocate(fnm(kms:kme), fnp(kms:kme)); fnm = 0.; fnp = 0.

  call rd3(p3d, nz); call rd3(t3d, nz); call rd3(pi3d, nz); call rd3(dz8w, nz)
  call rd3(rho, nz); call rd3(qv, nz); call rd3(qc, nz); call rd3(qr, nz)
  call rd3(qi, nz); call rd3(qs, nz); call rd3(qg, nz); call rd3(re_cloud, nz)
  call rd3(re_ice, nz); call rd3(re_snow, nz); call rd3(nwfa, nz); call rd3(nifa, nz)
  call rd3(p8w, nz + 1); call rd3(z_at_w, nz + 1)
  read(iu) fnm(1:nz); read(iu) fnp(1:nz)
  call rd2(tsk); call rd2(emiss); call rd2(albedo); call rd2(xland); call rd2(xice)
  call rd2(snow); call rd2(xlat); call rd2(xlong); call rd2(ht)
  call rd3(qnc, nz)
  close(iu)

  ! ---- phy_prep: t8w (dyn_em/module_big_step_utilities_em.F) ----
  kde_ = kde
  k_end = kde_ - 1
  do j = jts, jte
  do k = kts, k_end
  do i = its, ite
    zhalf(i,k,j) = 0.5*(z_at_w(i,k,j) +z_at_w(i,k+1,j) )
  enddo
  enddo
  enddo
  do j = jts, jte
  do k = 2, k_end
  do i = its, ite
    t8w(i,k,j) = fnm(k)*t3d(i,k,j)+fnp(k)*t3d(i,k-1,j)
  enddo
  enddo
  enddo
  do j = jts, jte
  do i = its, ite
    z0 = z_at_w(i,1,j)
    z1 = zhalf(i,1,j)
    z2 = zhalf(i,2,j)
    w1 = (z0 - z2)/(z1 - z2)
    w2 = 1. - w1
    t8w(i,1,j) = w1*t3d(i,1,j)+w2*t3d(i,2,j)
    z0 = z_at_w(i,kde_,j)
    z1 = zhalf(i,k_end,j)
    z2 = zhalf(i,k_end-1,j)
    w1 = (z0 - z2)/(z1 - z2)
    w2 = 1. - w1
    t8w(i,kde_,j) = w1*t3d(i,kde_-1,j)+w2*t3d(i,kde_-2,j)
  enddo
  enddo

  ! ---- radconst + calc_coszen ----
  degrad = piconst/180.
  dpd = 360./365.
  call radconst(xtime, declin, solcon, julian, degrad, dpd)
  call calc_coszen(ims,ime,jms,jme,its,ite,jts,jte, &
                   julian,xtime+radt*0.5,gmt, &
                   declin,degrad,xlong,xlat,coszen,hrang)

  ! ---- cloud fraction ----
  cldfra = 0.
  if (icloud == 1) then
     if (f_qc .or. f_qi) then
        call cal_cldfra1(cldfra, qv, qc, qi, qs,                 &
                         f_qv, f_qc, f_qi, f_qs, t3d, p3d,        &
                         f_ice_phy, f_rain_phy,                   &
                         mp_physics, cldfra1_flag,                &
                         ids,ide, jds,jde, kds,kde,               &
                         ims,ime, jms,jme, kms,kme,               &
                         its,ite, jts,jte, kts,kte)
     end if
  end if

  ! ---- ozone, o3input = 2 ----
  o3rad = 0.
  if (o3input == 2) then
     allocate(ozmixm(ims:ime, levsiz, jms:jme, num_months), pin(levsiz))
     allocate(ozmixt(ims:ime, levsiz, jms:jme))
     call oznini(ozmixm, pin, levsiz, num_months, xlat, &
                 ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, &
                 its,ite, jts,jte, kts,kte)
     call ozn_time_int(julday, julian, ozmixm, ozmixt, levsiz, num_months, &
                       ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, &
                       its,ite, jts,jte, kts,kte)
     call ozn_p_int(p3d, pin, levsiz, ozmixt, o3rad, &
                    ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, &
                    its,ite, jts,jte, kts,kte)
  end if

  ! ---- driver zeroing ----
  gsw = 0.; glw = 0.; swdown = 0.
  rthraten = 0.; rthratenlw = 0.; rthratensw = 0.
  rthratenlwc = 0.; rthratenswc = 0.
  olr = 0.; lwcf = 0.; swcf = 0.
  lwupt = 0.; lwuptc = 0.; lwdnt = 0.; lwdntc = 0.
  lwupb = 0.; lwupbc = 0.; lwdnb = 0.; lwdnbc = 0.
  swupt = 0.; swuptc = 0.; swdnt = 0.; swdntc = 0.
  swupb = 0.; swupbc = 0.; swdnb = 0.; swdnbc = 0.

  call rrtmg_lwinit_k(.true., ids,ide, jds,jde, kds,kde, &
                      ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte)
  call rrtmg_swinit_k(.true., ids,ide, jds,jde, kds,kde, &
                      ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte)

  ! CASE (RRTMK_LWSCHEME), module_radiation_driver.F:2104-2131.  aerod is
  ! the driver's optional dummy, absent here (aer_opt /= 1).
  call rad_rrtmg_driver(                                 &
                rthraten, rthratensw,                    &
                rthratenlwc, rthratenswc,                &
                lwupt=lwupt, lwuptc=lwuptc, lwdnt=lwdnt, lwdntc=lwdntc, &
                lwupb=lwupb, lwupbc=lwupbc, lwdnb=lwdnb, lwdnbc=lwdnbc, &
                glw=glw, olr=olr, lwcf=lwcf,             &
                swupt=swupt, swuptc=swuptc, swdnt=swdnt, swdntc=swdntc, &
                swupb=swupb, swupbc=swupbc, swdnb=swdnb, swdnbc=swdnbc, &
                gsw=gsw, swcf=swcf,                      &
                cosz=coszen, solcon=solcon, albedo=albedo, emiss=emiss, &
                t3d=t3d, t8w=t8w, tsk=tsk, rho3d=rho, p3d=p3d, p8w=p8w, &
                cldfra3d=cldfra, r=287., g=9.81, nc3d=qnc, xland=xland, &
                f_qnc=f_qnc, f_qv=f_qv, f_qc=f_qc, f_qr=f_qr, f_qi=f_qi, &
                f_qs=f_qs, f_qg=f_qg,                    &
                qv3d=qv, qc3d=qc, qr3d=qr, qi3d=qi, qs3d=qs, qg3d=qg, &
                o3input=o3input, o33d=o3rad,             &
                aer_opt=aer_opt, no_src=1,               &
                ids=ids, ide=ide, jds=jds, jde=jde, kds=kds, kde=kde, &
                ims=ims, ime=ime, jms=jms, jme=jme, kms=kms, kme=kme, &
                its=its, ite=ite, jts=jts, jte=jte, kts=kts, kte=kte)
  ! RTHRATENLW = RTHRATEN (:2252-2262)
  do j = jts, jte
  do k = kts, kte
  do i = its, ite
     rthratenlw(i,k,j) = rthraten(i,k,j)
  end do
  end do
  end do
  ! CASE (RRTMK_SWSCHEME): RTHRATEN = RTHRATEN + RTHRATENSW (:2702-2710)
  do j = jts, jte
  do k = kts, kte
  do i = its, ite
     rthraten(i,k,j) = rthraten(i,k,j) + rthratensw(i,k,j)
  end do
  end do
  end do

  open(newunit=ou, file=trim(outpath), access='stream', form='unformatted', &
       status='replace', convert='little_endian')
  call put3('rthratensw_wrapper', rthratensw, nz)
  do j = jts, jte
  do k = kts, kte
  do i = its, ite
     rthratensw(i,k,j) = rthraten(i,k,j) - rthratenlw(i,k,j)
  end do
  end do
  end do
  do j = jts, jte
  do i = its, ite
     swdown(i,j) = gsw(i,j)/(1.-albedo(i,j))
  end do
  end do

  ! Diagnostics: effectRad on the driver's arguments with its clamps
  ! (module_ra_rrtmg_swk.F rad_rrtmg_driver 5-1); never fed back.
  allocate(qccps(kts:kte), re1(kts:kte), re2(kts:kte), re3(kts:kte))
  do i = its, ite
     qccps = 0.e0
     re1 = 5.0e-6; re2 = 10.0e-6; re3 = 25.0e-6
     call effectRad(t3d(i,kts:kte,1), qc(i,kts:kte,1), qnc(i,kts:kte,1), &
                    qi(i,kts:kte,1), qs(i,kts:kte,1), qg(i,kts:kte,1),   &
                    rho(i,kts:kte,1), 0., 273.15, qccps, f_qnc,          &
                    re1, re2, re3, kts, kte)
     re1 = re1 * 1.e+6; re2 = re2 * 1.e+6; re3 = re3 * 1.e+6
     rqc(i,kts:kte,1) = max(2.51,  min(re1, 50.))
     rqi(i,kts:kte,1) = max(10.01, min(re2, 125.))
     rqs(i,kts:kte,1) = max(25.,   min(re3, 999.))
  end do
  cdm2 = rgmma(5./3.)

  call put1('declin', declin); call put1('solcon', solcon); call put1('cdm2', cdm2)
  call put2('coszen', coszen); call put2('hrang', hrang)
  call put3('t8w', t8w, nz + 1); call put3('cldfra', cldfra, nz)
  call put3('o3rad', o3rad, nz)
  call put3('re_qc', rqc, nz); call put3('re_qi', rqi, nz); call put3('re_qs', rqs, nz)
  call put3('rthratenlw', rthratenlw, nz); call put3('rthratenlwc', rthratenlwc, nz)
  call put2('glw', glw); call put2('olr', olr); call put2('lwcf', lwcf)
  call put2('lwupt', lwupt); call put2('lwuptc', lwuptc); call put2('lwdnt', lwdnt)
  call put2('lwdntc', lwdntc); call put2('lwupb', lwupb); call put2('lwupbc', lwupbc)
  call put2('lwdnb', lwdnb); call put2('lwdnbc', lwdnbc)
  call put3('rthratensw', rthratensw, nz); call put3('rthratenswc', rthratenswc, nz)
  call put3('rthraten', rthraten, nz)
  call put2('gsw', gsw); call put2('swdown', swdown)
  call put2('swcf', swcf); call put2('swupt', swupt); call put2('swuptc', swuptc)
  call put2('swdnt', swdnt); call put2('swdntc', swdntc); call put2('swupb', swupb)
  call put2('swupbc', swupbc); call put2('swdnb', swdnb); call put2('swdnbc', swdnbc)
  call header('extliq1', size(extliq1)); write(ou) extliq1
  call header('ssaliq1', size(ssaliq1)); write(ou) ssaliq1
  call header('asyliq1', size(asyliq1)); write(ou) asyliq1
  close(ou)
  write(*,'(A,I0,A)') 'rad_k_column_oracle: OK (', ncol, ' columns)'

contains

  subroutine alloc3(a)
    real, allocatable, intent(inout) :: a(:,:,:)
    allocate(a(ims:ime, kms:kme, jms:jme)); a = 0.
  end subroutine alloc3

  subroutine alloc2(a)
    real, allocatable, intent(inout) :: a(:,:)
    allocate(a(ims:ime, jms:jme)); a = 0.
  end subroutine alloc2

  subroutine rd3(a, nk)
    real, intent(inout) :: a(ims:ime, kms:kme, jms:jme)
    integer, intent(in) :: nk
    real :: tmp(ncol, nk)
    read(iu) tmp
    a(1:ncol, 1:nk, 1) = tmp
  end subroutine rd3

  subroutine rd2(a)
    real, intent(inout) :: a(ims:ime, jms:jme)
    read(iu) a(1:ncol, 1)
  end subroutine rd2

  subroutine header(name, n)
    character(len=*), intent(in) :: name
    integer, intent(in) :: n
    character(len=32) :: tag
    tag = name
    write(ou) tag, n
  end subroutine header

  subroutine put1(name, v)
    character(len=*), intent(in) :: name
    real, intent(in) :: v
    call header(name, 1)
    write(ou) v
  end subroutine put1

  subroutine put2(name, a)
    character(len=*), intent(in) :: name
    real, intent(in) :: a(ims:ime, jms:jme)
    call header(name, ncol)
    write(ou) a(1:ncol, 1)
  end subroutine put2

  subroutine put3(name, a, nk)
    character(len=*), intent(in) :: name
    real, intent(in) :: a(ims:ime, kms:kme, jms:jme)
    integer, intent(in) :: nk
    call header(name, ncol*nk)
    write(ou) a(1:ncol, 1:nk, 1)
  end subroutine put3

end program rad_k_column_oracle
