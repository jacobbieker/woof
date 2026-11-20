! WRFPLUS 4.7.1 oracle driver: G_CORIOLIS then G_CURVATURE, and
! A_CURVATURE then A_CORIOLIS, on one input record
! (docs/dev/P3A-WRF-DRYCORE-ADJOINT.md section 5, item 5).  Built against
! WRFPLUS's libwrflib.a (configure wrfplus, real*8, -O0 -ffp-contract=off)
! by tools/wrfplus_oracle/build.sh; driven by tools/wrfplus_oracle/fixture.py.
!
! Input (stream, native binary64/int32): nx, ny, nz, flags (bit 0
! periodic_x, 1 periodic_y, 2 specified, 4-7 open_xs/xe/ys/ye); rdx, rdy;
! then on (-3:nx+4, 1:nz+1, -3:ny+4) ru_tend, its adjoint seed, rv_tend,
! its seed, rw_tend, its seed, ru, rud, rv, rvd, rw, rwd, u, ud, v, vd;
! on (-3:nx+4, -3:ny+4) msfu, msfv, msft, f, e, sina, cosa; on 1:nz+1
! fzm, fzp.  msfux = msfuy = msfu, msfvx = msfvy = msfv, msftx = msfty =
! msft; map_proj 1 (the vxgm branch of curvature).  rw is the coupled w
! (the fixture forms it and its tangent and adjoint from w and mut).
! Output (stream): TL ru_tend, its tangent, rv_tend, its tangent,
! rw_tend, its tangent; AD ru_tend, rv_tend, rw_tend (after the sweep),
! ru, rv, rw, u, v adjoints.
program wrfplus_coriolis
  use module_configure, only : grid_config_rec_type
  use g_module_big_step_utilities_em, only : g_coriolis, g_curvature
  use a_module_big_step_utilities_em, only : a_coriolis, a_curvature
  implicit none
  integer :: nx, ny, nz, flags, iu, hdr(4)
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
  real :: rdx, rdy, sc(2)
  type(grid_config_rec_type) :: cf
  real, allocatable, dimension(:,:,:) :: rut, rua_t, rvt, rva_t, rwt, rwa_t, ru, rud, rv, rvd, &
       rw, rwd, u, ud, v, vd, w, rut2, rutd, rvt2, rvtd, rwt2, rwtd, rua, rva, rwa, ua, va
  real, allocatable, dimension(:,:) :: msfu, msfv, msft, f, e, sina, cosa, xlat
  real, allocatable, dimension(:) :: fzm, fzp
  character(len=512) :: fin, fout

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(newunit=iu, file=trim(fin), access='stream', form='unformatted', status='old')
  read(iu) hdr
  nx = hdr(1); ny = hdr(2); nz = hdr(3); flags = hdr(4)
  read(iu) sc
  rdx = sc(1); rdy = sc(2)
  ids = 1; ide = nx + 1; jds = 1; jde = ny + 1; kds = 1; kde = nz + 1
  ims = -3; ime = nx + 4; jms = -3; jme = ny + 4; kms = 1; kme = nz + 1
  call a3(rut); call a3(rua_t); call a3(rvt); call a3(rva_t); call a3(rwt); call a3(rwa_t)
  call a3(ru); call a3(rud); call a3(rv); call a3(rvd); call a3(rw); call a3(rwd)
  call a3(u); call a3(ud); call a3(v); call a3(vd)
  call a2(msfu); call a2(msfv); call a2(msft); call a2(f); call a2(e); call a2(sina)
  call a2(cosa)
  allocate(fzm(kms:kme), fzp(kms:kme))
  read(iu) fzm, fzp
  close(iu)

  cf%periodic_x = btest(flags, 0); cf%periodic_y = btest(flags, 1)
  cf%specified = btest(flags, 2);  cf%nested = .false.
  cf%open_xs = btest(flags, 4);    cf%open_xe = btest(flags, 5)
  cf%open_ys = btest(flags, 6);    cf%open_ye = btest(flags, 7)
  cf%symmetric_xs = .false.; cf%symmetric_xe = .false.
  cf%symmetric_ys = .false.; cf%symmetric_ye = .false.; cf%polar = .false.
  cf%map_proj = 1

  allocate(w, rut2, rutd, rvt2, rvtd, rwt2, rwtd, rua, rva, rwa, ua, va, mold=ru)
  allocate(xlat, mold=f)
  w = 0; xlat = 0
  rut2 = rut; rvt2 = rvt; rwt2 = rwt; rutd = 0; rvtd = 0; rwtd = 0
  rua = 0; rva = 0; rwa = 0; ua = 0; va = 0
  call g_coriolis(ru, rud, rv, rvd, rw, rwd, rut2, rutd, rvt2, rvtd, rwt2, rwtd, cf, &
       msft, msft, msfu, msfu, msfv, msfv, f, e, sina, cosa, fzm, fzp, &
       ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
       ids, ide, jds, jde, kds, kde)
  call g_curvature(ru, rud, rv, rvd, rw, rwd, u, ud, v, vd, w, rut2, rutd, rvt2, rvtd, &
       rwt2, rwtd, cf, msfu, msfu, msfv, msfv, msft, msft, xlat, fzm, fzp, rdx, rdy, &
       ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
       ids, ide, jds, jde, kds, kde)
  call a_curvature(ru, rua, rv, rva, rw, rwa, u, ua, v, va, w, rut, rua_t, rvt, rva_t, &
       rwt, rwa_t, cf, msfu, msfu, msfv, msfv, msft, msft, xlat, fzm, fzp, rdx, rdy, &
       ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
       ids, ide, jds, jde, kds, kde)
  call a_coriolis(ru, rua, rv, rva, rw, rwa, rut, rua_t, rvt, rva_t, rwt, rwa_t, cf, &
       msft, msft, msfu, msfu, msfv, msfv, f, e, sina, cosa, fzm, fzp, &
       ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
       ids, ide, jds, jde, kds, kde)

  open(newunit=iu, file=trim(fout), access='stream', form='unformatted', status='replace')
  write(iu) rut2, rutd, rvt2, rvtd, rwt2, rwtd, rua_t, rva_t, rwa_t, rua, rva, rwa, ua, va
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

end program wrfplus_coriolis
