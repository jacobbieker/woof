! WRFPLUS 4.7.1 oracle driver for P3a item 7b
! (docs/dev/P3A-WRF-DRYCORE-ADJOINT.md section 5, item 7):
!   mode 0: g_SPEC_BDY_FINAL on one field, then a_SPEC_BDY_FINAL;
!   mode 1: g_spec_bdyupdate_ph, g_zero_grad_bdy (w) and g_calc_p_rho
!           (step 0, non-hydrostatic), then a_calc_p_rho, a_zero_grad_bdy
!           and a_spec_bdyupdate_ph.
! Built against WRFPLUS's libwrflib.a (configure wrfplus, real*8, -O0
! -ffp-contract=off) by tools/wrfplus_oracle/build.sh; driven by
! tools/wrfplus_oracle/fixture.py (lbc_final).
!
! Input (stream, native binary64/int32): mode, nx, ny, nz (mass grid),
! var (mode 0: 0 'u', 1 'v', 2 't', 3 'h', 4 'm', 5 'w'), width,
! spec_zone, nk (level slots: nz+1, or 1 for 'm'); dt (dtbc or dtau).
! 3-D arrays on (-3:nx+4, 1:nk, -3:ny+4), 2-D on (-3:nx+4, -3:ny+4),
! tables (j or i, 1:nk, 1:width) on -3:ny+4 (x) or -3:nx+4 (y).
!   mode 0: field, tangent, adjoint seed; mu, its tangent, msf; xs, xe,
!     ys, ye and their tendencies; the sixteen table tangents.
!     Output: TL field, tangent; field adjoint, mu adjoint, the eight
!     table adjoints.
!   mode 1: ph, w, rph, php, t_2, t_1, alt, c2a, their tangents (same
!     order), the seeds of ph, w, p, al; mu_tend, muts, mu and tangents;
!     rdnw, dnw, znu (nz).
!     Output: TL ph, phd, w, wd, p, pd, al, ald; adjoints of ph, w, p,
!     al, rph, php, t_2, t_1, alt, c2a (3-D), mu_tend, muts, mu (2-D).
program wrfplus_lbc_final
  use module_configure, only : grid_config_rec_type
  use g_module_bc, only : g_spec_bdy_final, g_zero_grad_bdy
  use a_module_bc, only : a_spec_bdy_final, a_zero_grad_bdy
  use g_module_bc_em, only : g_spec_bdyupdate_ph
  use a_module_bc_em, only : a_spec_bdyupdate_ph
  use g_module_small_step_em, only : g_calc_p_rho
  use a_module_small_step_em, only : a_calc_p_rho
  implicit none
  integer :: mode, nx, ny, nz, var, width, spec_zone, nk, iu, hdr(8)
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, kte
  real :: dt, sc(1)
  real, parameter :: t0 = 300.
  type(grid_config_rec_type) :: cf
  character(len=1) :: vc
  character(len=512) :: fin, fout
  real, allocatable, dimension(:,:,:) :: f, fd, fs, f2, fd2, fb
  real, allocatable, dimension(:,:,:) :: xs, xe, ys, ye, txs, txe, tys, tye
  real, allocatable, dimension(:,:,:) :: dxs, dxe, dys, dye, dtxs, dtxe, dtys, dtye
  real, allocatable, dimension(:,:,:) :: axs, axe, ays, aye, atxs, atxe, atys, atye
  real, allocatable, dimension(:,:) :: mu, mud, msf, mub
  real, allocatable, dimension(:,:,:) :: ph, w, rph, php, th, thp, alt, c2a
  real, allocatable, dimension(:,:,:) :: phd, wd, rphd, phpd, thd, thpd, altd, c2ad
  real, allocatable, dimension(:,:,:) :: sph, sw, sp, sal, p, pd, al, ald, pm1, pm1d
  real, allocatable, dimension(:,:,:) :: ph0, ph1, p1, al1
  real, allocatable, dimension(:,:,:) :: a_rph, a_php, a_th, a_thp, a_alt, a_c2a, a_pm1
  real, allocatable, dimension(:,:) :: rmu, rmud, muts, mutsd, mupp, muppd
  real, allocatable, dimension(:,:) :: a_rmu, a_muts, a_mupp
  real, allocatable :: rdnw(:), dnw(:), znu(:)

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(newunit=iu, file=trim(fin), access='stream', form='unformatted', status='old')
  read(iu) hdr
  mode = hdr(1); nx = hdr(2); ny = hdr(3); nz = hdr(4); var = hdr(5)
  width = hdr(6); spec_zone = hdr(7); nk = hdr(8)
  read(iu) sc
  dt = sc(1)
  ids = 1; ide = nx + 1; jds = 1; jde = ny + 1; kds = 1; kde = nk
  ims = -3; ime = nx + 4; jms = -3; jme = ny + 4; kms = 1; kme = nk
  kte = nk
  cf%periodic_x = .false.; cf%periodic_y = .false.

  if (mode == 0) then
    vc = 'uvthmw'(var + 1:var + 1)
    call a3(f); call a3(fd); call a3(fs)
    call a2(mu); call a2(mud); call a2(msf)
    call ax(xs); call ax(xe); call ay(ys); call ay(ye)
    call ax(txs); call ax(txe); call ay(tys); call ay(tye)
    call ax(dxs); call ax(dxe); call ay(dys); call ay(dye)
    call ax(dtxs); call ax(dtxe); call ay(dtys); call ay(dtye)
    close(iu)
    allocate(f2, source=f); allocate(fd2, source=fd)
    call g_spec_bdy_final(f2, fd2, mu, mud, msf, xs, dxs, xe, dxe, ys, dys, ye, dye, &
         txs, dtxs, txe, dtxe, tys, dtys, tye, dtye, vc, cf, width, spec_zone, dt, &
         ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
         ids, ide, jds, jde, kds, kde, ids, ide, jds, jde, kds, kte)
    allocate(fb, source=fs); allocate(mub, mold=mu); mub = 0
    allocate(axs, axe, atxs, atxe, mold=xs)
    allocate(ays, aye, atys, atye, mold=ys)
    axs = 0; axe = 0; atxs = 0; atxe = 0; ays = 0; aye = 0; atys = 0; atye = 0
    call a_spec_bdy_final(f, fb, mu, mub, msf, xs, axs, xe, axe, ys, ays, ye, aye, &
         txs, atxs, txe, atxe, tys, atys, tye, atye, vc, cf, width, spec_zone, dt, &
         ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
         ids, ide, jds, jde, kds, kde, ids, ide, jds, jde, kds, kte)
    open(newunit=iu, file=trim(fout), access='stream', form='unformatted', status='replace')
    write(iu) f2, fd2, fb, mub, axs, axe, ays, aye, atxs, atxe, atys, atye
    close(iu)
  else
    call a3(ph); call a3(w); call a3(rph); call a3(php); call a3(th); call a3(thp)
    call a3(alt); call a3(c2a)
    call a3(phd); call a3(wd); call a3(rphd); call a3(phpd); call a3(thd); call a3(thpd)
    call a3(altd); call a3(c2ad)
    call a3(sph); call a3(sw); call a3(sp); call a3(sal)
    call a2(rmu); call a2(muts); call a2(mupp)
    call a2(rmud); call a2(mutsd); call a2(muppd)
    allocate(rdnw(nz + 1), dnw(nz + 1), znu(nz + 1))
    rdnw = 0; dnw = 0; znu = 0
    read(iu) rdnw(1:nz), dnw(1:nz), znu(1:nz)
    close(iu)
    allocate(ph0, source=ph)
    allocate(p, pd, al, ald, pm1, pm1d, mold=ph)
    p = 0; pd = 0; al = 0; ald = 0; pm1 = 0; pm1d = 0
    call g_spec_bdyupdate_ph(php, phpd, ph, phd, rph, rphd, rmu, rmud, muts, mutsd, dt, &
         'h', cf, spec_zone, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
         ids, ide, jds, jde, kds, kde, ids, ide, jds, jde, kds, kte)
    call g_zero_grad_bdy(w, wd, 'w', cf, spec_zone, ids, ide, jds, jde, kds, kde, &
         ims, ime, jms, jme, kms, kme, ids, ide, jds, jde, kds, kde, &
         ids, ide, jds, jde, kds, kte)
    call g_calc_p_rho(al, ald, p, pd, ph, phd, alt, altd, th, thd, thp, thpd, c2a, c2ad, &
         pm1, pm1d, mupp, muppd, muts, mutsd, znu, t0, rdnw, dnw, 0., .true., 0, ids, ide, &
         jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ids, ide, jds, jde, kds, kte)
    open(newunit=iu, file=trim(fout), access='stream', form='unformatted', status='replace')
    write(iu) ph, phd, w, wd, p, pd, al, ald

    allocate(ph1, source=ph); allocate(p1, source=p); allocate(al1, source=al)
    allocate(a_rph, a_php, a_th, a_thp, a_alt, a_c2a, a_pm1, mold=ph)
    a_rph = 0; a_php = 0; a_th = 0; a_thp = 0; a_alt = 0; a_c2a = 0; a_pm1 = 0
    allocate(a_rmu, a_muts, a_mupp, mold=muts)
    a_rmu = 0; a_muts = 0; a_mupp = 0
    call a_calc_p_rho(al1, sal, p1, sp, ph1, sph, alt, a_alt, th, a_th, thp, a_thp, c2a, &
         a_c2a, pm1, a_pm1, mupp, a_mupp, muts, a_muts, znu, t0, rdnw, dnw, 0., .true., 0, &
         ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ids, ide, jds, jde, &
         kds, kte)
    call a_zero_grad_bdy(sw, 'w', cf, spec_zone, ids, ide, jds, jde, kds, kde, &
         ims, ime, jms, jme, kms, kme, ids, ide, jds, jde, kds, kde, &
         ids, ide, jds, jde, kds, kte)
    call a_spec_bdyupdate_ph(php, a_php, ph0, sph, rph, a_rph, rmu, a_rmu, muts, a_muts, &
         dt, 'h', cf, spec_zone, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, &
         kme, ids, ide, jds, jde, kds, kde, ids, ide, jds, jde, kds, kte)
    write(iu) sph, sw, sp, sal, a_rph, a_php, a_th, a_thp, a_alt, a_c2a, a_rmu, a_muts, a_mupp
    close(iu)
  end if

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

  subroutine ax(x)
    real, allocatable, intent(out) :: x(:,:,:)
    allocate(x(jms:jme, kms:kme, width))
    read(iu) x
  end subroutine ax

  subroutine ay(x)
    real, allocatable, intent(out) :: x(:,:,:)
    allocate(x(ims:ime, kms:kme, width))
    read(iu) x
  end subroutine ay

end program wrfplus_lbc_final
