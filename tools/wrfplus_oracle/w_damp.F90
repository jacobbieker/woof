! WRFPLUS 4.7.1 oracle driver: G_W_DAMP and A_W_DAMP (w_damping = 1) on
! one input record (docs/dev/P3A-WRF-DRYCORE-ADJOINT.md section 5, item 5).
! Built against WRFPLUS's libwrflib.a (configure wrfplus, real*8, -O0
! -ffp-contract=off) by tools/wrfplus_oracle/build.sh; driven by
! tools/wrfplus_oracle/fixture.py.
!
! Input (stream, native binary64/int32): nx, ny, nz; dt; then on
! (-3:nx+4, 1:nz+1, -3:ny+4) rw_tend, its adjoint seed, ww, wwd, w; on
! (-3:nx+4, -3:ny+4) mut, mutd; on 1:nz+1 rdnw.  Map factors 1, polar
! off.  WRFPLUS uses the column mass mut (no hybrid c1f/c2f) and w_beta
! (1) as both the onset and the excess reference.
! Output (stream): TL rw_tend, its tangent; AD rw_tend seed (after the
! sweep), ww and mut adjoints.
program wrfplus_w_damp
  use module_configure, only : grid_config_rec_type
  use g_module_big_step_utilities_em, only : g_w_damp
  use a_module_big_step_utilities_em, only : a_w_damp
  implicit none
  integer :: nx, ny, nz, iu, hdr(3)
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
  real :: dt, sc(1), vmax, hmax, r1
  type(grid_config_rec_type) :: cf
  real, allocatable, dimension(:,:,:) :: rwt, rwa_t, ww, wwd, w, uv, rwt2, rwtd, wwa, zuv
  real, allocatable, dimension(:,:) :: mut, mutd, muta, one
  real, allocatable, dimension(:) :: rdnw
  character(len=512) :: fin, fout

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(newunit=iu, file=trim(fin), access='stream', form='unformatted', status='old')
  read(iu) hdr
  nx = hdr(1); ny = hdr(2); nz = hdr(3)
  read(iu) sc
  dt = sc(1)
  ids = 1; ide = nx + 1; jds = 1; jde = ny + 1; kds = 1; kde = nz + 1
  ims = -3; ime = nx + 4; jms = -3; jme = ny + 4; kms = 1; kme = nz + 1
  call a3(rwt); call a3(rwa_t); call a3(ww); call a3(wwd); call a3(w)
  call a2(mut); call a2(mutd)
  allocate(rdnw(kms:kme))
  read(iu) rdnw
  close(iu)

  cf%w_damping = 1; cf%polar = .false.
  allocate(uv, rwt2, rwtd, wwa, zuv, mold=ww)
  allocate(muta, one, mold=mut)
  uv = 0; zuv = 0; one = 1; r1 = 1
  rwt2 = rwt; rwtd = 0; wwa = 0; muta = 0
  call g_w_damp(rwt2, rwtd, vmax, hmax, uv, zuv, uv, zuv, ww, wwd, w, zuv, mut, mutd, &
       rdnw, r1, r1, one, one, one, one, dt, cf, &
       ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
       ids, ide, jds, jde, kds, kde)
  zuv = 0
  call a_w_damp(rwt, rwa_t, vmax, hmax, uv, zuv, uv, zuv, ww, wwa, w, zuv, mut, muta, &
       rdnw, r1, r1, one, one, one, one, dt, cf, &
       ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
       ids, ide, jds, jde, kds, kde)

  open(newunit=iu, file=trim(fout), access='stream', form='unformatted', status='replace')
  write(iu) rwt2, rwtd, rwa_t, wwa, muta
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

end program wrfplus_w_damp
