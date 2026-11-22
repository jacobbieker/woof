! WRFPLUS 4.7.1 oracle driver: g_relax_bdytend then g_spec_bdytend on one
! field (relax_bdy_dry's relaxation and spec_bdy_dry's specified zone),
! and a_spec_bdytend then a_relax_bdytend
! (docs/dev/P3A-WRF-DRYCORE-ADJOINT.md section 5, item 7).  Built against
! WRFPLUS's libwrflib.a (configure wrfplus, real*8, -O0 -ffp-contract=off)
! by tools/wrfplus_oracle/build.sh; driven by tools/wrfplus_oracle/fixture.py.
!
! Input (stream, native binary64/int32): nx, ny, nz (mass grid), var
! (0 'u', 1 'v', 2 't', 3 'h', 4 'm'), width, spec_zone, relax_zone, nk
! (the field's level slots: nz+1, or 1 for 'm'); dtbc; fcx, gcx
! (width); then on (-3:nx+4, 1:nk, -3:ny+4) the coupled field, its
! tangent, the tendency and its adjoint seed; then the tables xs, xe,
! ys, ye, their tendencies xs..ye (j or i, 1:nk, 1:width: x tables on
! -3:ny+4, y tables on -3:nx+4), and the same sixteen tangents.
! Output (stream): TL tendency, its tangent; AD tendency adjoint (after
! the sweep), the field adjoint, the eight table adjoints.
program wrfplus_lbc
  use module_configure, only : grid_config_rec_type
  use g_module_bc, only : g_relax_bdytend, g_spec_bdytend
  use a_module_bc, only : a_relax_bdytend, a_spec_bdytend
  implicit none
  integer :: nx, ny, nz, var, width, spec_zone, relax_zone, nk, iu, hdr(8), n
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, kte
  real :: dtbc, sc(1)
  real, allocatable :: fcx(:), gcx(:)
  type(grid_config_rec_type) :: cf
  real, allocatable, dimension(:,:,:) :: f, fd, t, ta, t2, td, fa
  real, allocatable, dimension(:,:,:) :: xs, xe, ys, ye, txs, txe, tys, tye
  real, allocatable, dimension(:,:,:) :: dxs, dxe, dys, dye, dtxs, dtxe, dtys, dtye
  real, allocatable, dimension(:,:,:) :: axs, axe, ays, aye, atxs, atxe, atys, atye
  character(len=1) :: vc
  character(len=512) :: fin, fout

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(newunit=iu, file=trim(fin), access='stream', form='unformatted', status='old')
  read(iu) hdr
  nx = hdr(1); ny = hdr(2); nz = hdr(3); var = hdr(4); width = hdr(5)
  spec_zone = hdr(6); relax_zone = hdr(7); nk = hdr(8)
  read(iu) sc
  dtbc = sc(1)
  allocate(fcx(width), gcx(width))
  read(iu) fcx, gcx
  vc = 'uvthm'(var + 1:var + 1)
  ids = 1; ide = nx + 1; jds = 1; jde = ny + 1; kds = 1; kde = nk
  ims = -3; ime = nx + 4; jms = -3; jme = ny + 4; kms = 1; kme = nk
  kte = nk
  if (var <= 2) then
    kde = nz + 1
  end if
  call a3(f); call a3(fd); call a3(t); call a3(ta)
  call ax(xs); call ax(xe); call ay(ys); call ay(ye)
  call ax(txs); call ax(txe); call ay(tys); call ay(tye)
  call ax(dxs); call ax(dxe); call ay(dys); call ay(dye)
  call ax(dtxs); call ax(dtxe); call ay(dtys); call ay(dtye)
  close(iu)

  cf%periodic_x = .false.; cf%periodic_y = .false.

  allocate(t2, td, fa, mold=t)
  t2 = t; td = 0; fa = 0
  call g_relax_bdytend(f, fd, t2, td, xs, dxs, xe, dxe, ys, dys, ye, dye, &
       txs, dtxs, txe, dtxe, tys, dtys, tye, dtye, vc, cf, &
       width, spec_zone, relax_zone, dtbc, fcx, gcx, &
       ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
       ids, ide, jds, jde, kds, kde, ids, ide, jds, jde, kds, kte)
  call g_spec_bdytend(t2, td, xs, dxs, xe, dxe, ys, dys, ye, dye, &
       txs, dtxs, txe, dtxe, tys, dtys, tye, dtye, vc, cf, width, spec_zone, &
       ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
       ids, ide, jds, jde, kds, kde, ids, ide, jds, jde, kds, kte)
  allocate(axs, axe, atxs, atxe, mold=xs)
  allocate(ays, aye, atys, atye, mold=ys)
  axs = 0; axe = 0; atxs = 0; atxe = 0; ays = 0; aye = 0; atys = 0; atye = 0
  call a_spec_bdytend(ta, atxs, atxe, atys, atye, vc, cf, width, spec_zone, &
       ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
       ids, ide, jds, jde, kds, kde, ids, ide, jds, jde, kds, kte)
  call a_relax_bdytend(fa, ta, axs, axe, ays, aye, atxs, atxe, atys, atye, vc, cf, &
       width, spec_zone, relax_zone, dtbc, fcx, gcx, &
       ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
       ids, ide, jds, jde, kds, kde, ids, ide, jds, jde, kds, kte)

  open(newunit=iu, file=trim(fout), access='stream', form='unformatted', status='replace')
  write(iu) t2, td, ta, fa, axs, axe, ays, aye, atxs, atxe, atys, atye
  close(iu)

contains

  subroutine a3(x)
    real, allocatable, intent(out) :: x(:,:,:)
    allocate(x(ims:ime, kms:kme, jms:jme))
    read(iu) x
  end subroutine a3

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

end program wrfplus_lbc
