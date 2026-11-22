! WRFPLUS 4.7.1 oracle driver: the diff_opt 1 / km_opt 4 coefficients
! (docs/dev/P3A-WRF-DRYCORE-ADJOINT.md section 5, item 5).
! Built against WRFPLUS's libwrflib.a (configure wrfplus, real*8, -O0
! -ffp-contract=off) by tools/wrfplus_oracle/build.sh; driven by
! tools/wrfplus_oracle/fixture.py.
!
! Mode 1 (deformation), as first_rk_step_part2_tl/_ad: g_compute_diff_metrics,
! g_set_physical_bc3d('w') on rdzw, rdz, z, zx, zy, g_cal_deform_and_div;
! then a_cal_deform_and_div, the a_set_physical_bc3d in reverse and
! a_compute_diff_metrics.
! Mode 2 (coefficients): G_SMAG2D_KM (diff_opt 1, km_opt 4) then
! g_phy_bc's diff_opt 1 part (set_physical_bc3d 't' on xkmh and xkhh);
! then its adjoint, A_SMAG2D_KM.  xkmh and xkhh start at their cold zero.
!
! Input (stream, native binary64/int32): nx, ny, nz, flags (bit 0
! periodic_x, 1 periodic_y, 2 specified), mode; rdx, rdy, cf1, cf2, cf3,
! dx, dy, c_s; then on (-3:nx+4, 1:nz+1, -3:ny+4):
!   mode 1: u, ud, v, vd, ph, phd, phb, w, and the defor11, defor22,
!           defor12 adjoint seeds; on (-3:nx+4, -3:ny+4) msfu, msfv, msft;
!           on 1:nz+1 fnm, fnp, dn, dnw;
!   mode 2: defor11, its tangent, defor22, its tangent, defor12, its
!           tangent, the xkmh and xkhh adjoint seeds; msft.
! Output (stream):
!   mode 1: defor11, its tangent, defor22, its tangent, defor12, its
!           tangent; the defor11, defor22, defor12 seeds after the sweep;
!           the u, v and ph adjoints;
!   mode 2: xkmh, its tangent, xkhh, its tangent; the xkmh and xkhh seeds
!           after the sweep; the defor11, defor22, defor12 adjoints.
program wrfplus_diffusion
  use module_configure, only : grid_config_rec_type
  use g_module_diffusion_em, only : g_compute_diff_metrics, g_cal_deform_and_div, g_smag2d_km
  use a_module_diffusion_em, only : a_compute_diff_metrics, a_cal_deform_and_div, a_smag2d_km
  use g_module_bc, only : g_set_physical_bc3d
  use a_module_bc, only : a_set_physical_bc3d
  implicit none
  integer :: nx, ny, nz, iu, hdr(5), flags, mode
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
  real :: sc(8), rdx, rdy, cf1, cf2, cf3, dx, dy
  type(grid_config_rec_type) :: cf
  real, allocatable, dimension(:,:,:) :: u, ud, v, vd, ph, phd, phb, w, s11, s22, s12
  real, allocatable, dimension(:,:,:) :: d11, d11d, d22, d22d, d12, d12d, skm, skh
  real, allocatable, dimension(:,:,:) :: z, zd, rdz, rdzd, rdzw, rdzwd, zx, zxd, zy, zyd
  real, allocatable, dimension(:,:,:) :: wd, dv, dvd, d33, d33d, d13, d13d, d23, d23d
  real, allocatable, dimension(:,:,:) :: au, av, aw, aph, az, ardz, ardzw, azx, azy
  real, allocatable, dimension(:,:,:) :: a11, a22, a33, a12, a13, a23, adv
  real, allocatable, dimension(:,:,:) :: km, kmd, kv, kvd, kh, khd, khv, khvd
  real, allocatable, dimension(:,:,:) :: akm, akv, akh, akhv
  real, allocatable, dimension(:,:,:,:) :: nba, nbad
  real, allocatable, dimension(:,:) :: msfu, msfv, msft
  real, allocatable, dimension(:) :: fnm, fnp, dn, dnw, ubase, vbase
  character(len=512) :: fin, fout

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(newunit=iu, file=trim(fin), access='stream', form='unformatted', status='old')
  read(iu) hdr
  nx = hdr(1); ny = hdr(2); nz = hdr(3); flags = hdr(4); mode = hdr(5)
  read(iu) sc
  rdx = sc(1); rdy = sc(2); cf1 = sc(3); cf2 = sc(4); cf3 = sc(5); dx = sc(6); dy = sc(7)
  ids = 1; ide = nx + 1; jds = 1; jde = ny + 1; kds = 1; kde = nz + 1
  ims = -3; ime = nx + 4; jms = -3; jme = ny + 4; kms = 1; kme = nz + 1

  cf%periodic_x = btest(flags, 0); cf%periodic_y = btest(flags, 1)
  cf%specified = btest(flags, 2);  cf%nested = .false.
  cf%open_xs = .false.; cf%open_xe = .false.; cf%open_ys = .false.; cf%open_ye = .false.
  cf%symmetric_xs = .false.; cf%symmetric_xe = .false.
  cf%symmetric_ys = .false.; cf%symmetric_ye = .false.; cf%polar = .false.
  cf%diff_opt = 1; cf%km_opt = 4; cf%sfs_opt = 0; cf%c_s = sc(8)
  cf%bl_pbl_physics = 0; cf%cu_physics = 0; cf%shcu_physics = 0

  if (mode == 1) then
    call a3(u); call a3(ud); call a3(v); call a3(vd); call a3(ph); call a3(phd)
    call a3(phb); call a3(w); call a3(s11); call a3(s22); call a3(s12)
    call a2(msfu); call a2(msfv); call a2(msft)
    allocate(fnm(kms:kme), fnp(kms:kme), dn(kms:kme), dnw(kms:kme))
    read(iu) fnm, fnp, dn, dnw
    close(iu)
    allocate(z, zd, rdz, rdzd, rdzw, rdzwd, zx, zxd, zy, zyd, mold=u)
    allocate(wd, dv, dvd, d11, d11d, d22, d22d, d33, d33d, d12, d12d, d13, d13d, &
             d23, d23d, mold=u)
    allocate(au, av, aw, aph, az, ardz, ardzw, azx, azy, mold=u)
    allocate(a11, a22, a33, a12, a13, a23, adv, mold=u)
    allocate(nba(ims:ime, kms:kme, jms:jme, 1), nbad(ims:ime, kms:kme, jms:jme, 1))
    allocate(ubase(kms:kme), vbase(kms:kme))
    z = 0; zd = 0; rdz = 0; rdzd = 0; rdzw = 0; rdzwd = 0; zx = 0; zxd = 0; zy = 0; zyd = 0
    wd = 0; dv = 0; dvd = 0; d11 = 0; d11d = 0; d22 = 0; d22d = 0; d33 = 0; d33d = 0
    d12 = 0; d12d = 0; d13 = 0; d13d = 0; d23 = 0; d23d = 0; nba = 0; nbad = 0
    ubase = 0; vbase = 0
    call g_compute_diff_metrics(cf, ph, phd, phb, z, zd, rdz, rdzd, rdzw, rdzwd, &
         zx, zxd, zy, zyd, rdx, rdy, ids, ide, jds, jde, kds, kde, &
         ims, ime, jms, jme, kms, kme, ids, ide, jds, jde, kds, kde)
    call gbc(rdzw, rdzwd); call gbc(rdz, rdzd); call gbc(z, zd)
    call gbc(zx, zxd); call gbc(zy, zyd)
    call g_cal_deform_and_div(cf, u, ud, v, vd, w, wd, dv, dvd, d11, d11d, d22, d22d, &
         d33, d33d, d12, d12d, d13, d13d, d23, d23d, nba, nbad, 1, ubase, vbase, &
         msfu, msfu, msfv, msfv, msft, msft, rdx, rdy, dn, dnw, rdz, rdzd, rdzw, rdzwd, &
         fnm, fnp, cf1, cf2, cf3, zx, zxd, zy, zyd, ids, ide, jds, jde, kds, kde, &
         ims, ime, jms, jme, kms, kme, ids, ide, jds, jde, kds, kde)

    au = 0; av = 0; aw = 0; aph = 0; az = 0; ardz = 0; ardzw = 0; azx = 0; azy = 0
    a11 = s11; a22 = s22; a12 = s12; a33 = 0; a13 = 0; a23 = 0; adv = 0; nbad = 0
    call a_cal_deform_and_div(cf, u, au, v, av, w, aw, dv, adv, d11, a11, d22, a22, &
         d33, a33, d12, a12, d13, a13, d23, a23, nba, nbad, 1, ubase, vbase, &
         msfu, msfu, msfv, msfv, msft, msft, rdx, rdy, dn, dnw, rdz, ardz, rdzw, ardzw, &
         fnm, fnp, cf1, cf2, cf3, zx, azx, zy, azy, ids, ide, jds, jde, kds, kde, &
         ims, ime, jms, jme, kms, kme, ids, ide, jds, jde, kds, kde)
    call abc(azy); call abc(azx); call abc(az); call abc(ardz); call abc(ardzw)
    call a_compute_diff_metrics(cf, ph, aph, phb, z, az, rdz, ardz, rdzw, ardzw, &
         zx, azx, zy, azy, rdx, rdy, ids, ide, jds, jde, kds, kde, &
         ims, ime, jms, jme, kms, kme, ids, ide, jds, jde, kds, kde)

    open(newunit=iu, file=trim(fout), access='stream', form='unformatted', status='replace')
    write(iu) d11, d11d, d22, d22d, d12, d12d, a11, a22, a12, au, av, aph
    close(iu)
  else
    call a3(d11); call a3(d11d); call a3(d22); call a3(d22d); call a3(d12); call a3(d12d)
    call a3(skm); call a3(skh)
    call a2(msft)
    close(iu)
    allocate(km, kmd, kv, kvd, kh, khd, khv, khvd, akm, akv, akh, akhv, mold=d11)
    allocate(rdzw, rdzwd, zx, zxd, zy, zyd, ardzw, azx, azy, a11, a22, a12, mold=d11)
    km = 0; kmd = 0; kv = 0; kvd = 0; kh = 0; khd = 0; khv = 0; khvd = 0
    rdzw = 0; rdzwd = 0; zx = 0; zxd = 0; zy = 0; zyd = 0
    call g_smag2d_km(cf, km, kmd, kv, kvd, kh, khd, khv, khvd, d11, d11d, d22, d22d, &
         d12, d12d, rdzw, rdzwd, dx, dy, msft, msft, zx, zxd, zy, zyd, &
         ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
         ids, ide, jds, jde, kds, kde)
    call gbct(km, kmd); call gbct(kh, khd)

    akm = skm; akh = skh; akv = 0; akhv = 0; ardzw = 0; azx = 0; azy = 0
    a11 = 0; a22 = 0; a12 = 0
    call abct(akh); call abct(akm)
    call a_smag2d_km(cf, km, akm, kv, akv, kh, akh, khv, akhv, d11, a11, d22, a22, &
         d12, a12, rdzw, ardzw, dx, dy, msft, msft, zx, azx, zy, azy, &
         ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
         ids, ide, jds, jde, kds, kde)

    open(newunit=iu, file=trim(fout), access='stream', form='unformatted', status='replace')
    write(iu) km, kmd, kh, khd, akm, akh, a11, a22, a12
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

  subroutine gbc(x, xd)
    real, intent(inout) :: x(ims:ime, kms:kme, jms:jme), xd(ims:ime, kms:kme, jms:jme)
    call g_set_physical_bc3d(x, xd, 'w', cf, ids, ide, jds, jde, kds, kde, &
         ims, ime, jms, jme, kms, kme, ids, ide, jds, jde, kds, kde, &
         ids, ide, jds, jde, kds, kde)
  end subroutine gbc

  subroutine abc(xa)
    real, intent(inout) :: xa(ims:ime, kms:kme, jms:jme)
    call a_set_physical_bc3d(xa, 'w', cf, ids, ide, jds, jde, kds, kde, &
         ims, ime, jms, jme, kms, kme, ids, ide, jds, jde, kds, kde, &
         ids, ide, jds, jde, kds, kde)
  end subroutine abc

  subroutine gbct(x, xd)
    real, intent(inout) :: x(ims:ime, kms:kme, jms:jme), xd(ims:ime, kms:kme, jms:jme)
    call g_set_physical_bc3d(x, xd, 't', cf, ids, ide, jds, jde, kds, kde, &
         ims, ime, jms, jme, kms, kme, ids, ide, jds, jde, kds, kde, &
         ids, ide, jds, jde, kds, kde)
  end subroutine gbct

  subroutine abct(xa)
    real, intent(inout) :: xa(ims:ime, kms:kme, jms:jme)
    call a_set_physical_bc3d(xa, 't', cf, ids, ide, jds, jde, kds, kde, &
         ims, ime, jms, jme, kms, kme, ids, ide, jds, jde, kds, kde, &
         ids, ide, jds, jde, kds, kde)
  end subroutine abct

end program wrfplus_diffusion
