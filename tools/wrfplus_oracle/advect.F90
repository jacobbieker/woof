! WRFPLUS 4.7.1 oracle driver: G_ADVECT_{SCALAR,U,V,W} and
! A_ADVECT_{SCALAR,U,V,W} on one input record
! (docs/dev/P3A-WRF-DRYCORE-ADJOINT.md section 5, item 6).  Built against
! WRFPLUS's libwrflib.a (configure wrfplus, real*8, -O0 -ffp-contract=off)
! by tools/wrfplus_oracle/build.sh; driven by tools/wrfplus_oracle/fixture.py.
!
! Input (stream, native binary64/int32): mode (1 scalar, 2 u, 3 v, 4 w),
! nx, ny, nz, flags (bit 0 periodic_x, 1 periodic_y, 2 specified,
! 4-7 open_xs/xe/ys/ye), h_mom, v_mom, h_sca, v_sca; rdx, rdy; then on
! (-3:nx+4, 1:nz+1, -3:ny+4) the field, its perturbation, ru, rud, rv,
! rvd, rom, romd, the tendency and the tendency adjoint seed; on
! (-3:nx+4, -3:ny+4) mut, msfux, msfuy, msfvx, msfvy, msftx, msfty; on
! 1:nz+1 fzm, fzp and rdzw (rdzu for w).
! Output (stream): TL tendency, tendencyd; AD field, ru, rv, rom adjoints.
program wrfplus_advect
  use module_configure, only : grid_config_rec_type
  use g_module_advect_em, only : g_advect_scalar, g_advect_u, g_advect_v, g_advect_w
  use a_module_advect_em, only : a_advect_scalar, a_advect_u, a_advect_v, a_advect_w
  implicit none
  integer :: mode, nx, ny, nz, flags, iu, hdr(9)
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
  real :: rdx, rdy
  type(grid_config_rec_type) :: cf
  real, allocatable, dimension(:,:,:) :: f, fd, ru, rud, rv, rvd, rom, romd, &
       tend, tendb, td, fb, rub, rvb, romb, oldb
  real, allocatable, dimension(:,:) :: mut, mutd, mutb, msfux, msfuy, msfvx, msfvy, &
       msftx, msfty
  real, allocatable, dimension(:) :: fzm, fzp, rdz
  character(len=512) :: fin, fout

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(newunit=iu, file=trim(fin), access='stream', form='unformatted', status='old')
  read(iu) hdr
  mode = hdr(1); nx = hdr(2); ny = hdr(3); nz = hdr(4); flags = hdr(5)
  read(iu) rdx, rdy
  ids = 1; ide = nx + 1; jds = 1; jde = ny + 1; kds = 1; kde = nz + 1
  ims = -3; ime = nx + 4; jms = -3; jme = ny + 4; kms = 1; kme = nz + 1
  call a3(f); call a3(fd); call a3(ru); call a3(rud); call a3(rv); call a3(rvd)
  call a3(rom); call a3(romd); call a3(tend); call a3(tendb)
  call a2(mut); call a2(msfux); call a2(msfuy); call a2(msfvx); call a2(msfvy)
  call a2(msftx); call a2(msfty)
  allocate(fzm(kms:kme), fzp(kms:kme), rdz(kms:kme))
  read(iu) fzm, fzp, rdz
  close(iu)

  cf%periodic_x = btest(flags, 0); cf%periodic_y = btest(flags, 1)
  cf%specified = btest(flags, 2);  cf%nested = .false.
  cf%open_xs = btest(flags, 4);    cf%open_xe = btest(flags, 5)
  cf%open_ys = btest(flags, 6);    cf%open_ye = btest(flags, 7)
  cf%symmetric_xs = .false.; cf%symmetric_xe = .false.
  cf%symmetric_ys = .false.; cf%symmetric_ye = .false.; cf%polar = .false.
  cf%h_mom_adv_order = hdr(6); cf%v_mom_adv_order = hdr(7)
  cf%h_sca_adv_order = hdr(8); cf%v_sca_adv_order = hdr(9)

  allocate(td, fb, rub, rvb, romb, oldb, mold=f)
  allocate(mutd, mutb, mold=mut)
  td = 0; mutd = 0
  select case (mode)
  case (1)
    call g_advect_scalar(f, fd, f, fd, tend, td, ru, rud, rv, rvd, rom, romd, mut, 1, cf, &
         msfux, msfuy, msfvx, msfvy, msftx, msfty, fzm, fzp, rdx, rdy, rdz, &
         ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
         ids, ide, jds, jde, kds, kde)
  case (2)
    call g_advect_u(f, fd, f, fd, tend, td, ru, rud, rv, rvd, rom, romd, mut, mutd, 1, cf, &
         msfux, msfuy, msfvx, msfvy, msftx, msfty, fzm, fzp, rdx, rdy, rdz, &
         ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
         ids, ide, jds, jde, kds, kde)
  case (3)
    call g_advect_v(f, fd, f, fd, tend, td, ru, rud, rv, rvd, rom, romd, mut, mutd, 1, cf, &
         msfux, msfuy, msfvx, msfvy, msftx, msfty, fzm, fzp, rdx, rdy, rdz, &
         ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
         ids, ide, jds, jde, kds, kde)
  case (4)
    call g_advect_w(f, fd, f, fd, tend, td, ru, rud, rv, rvd, rom, romd, mut, 1, cf, &
         msfux, msfuy, msfvx, msfvy, msftx, msfty, fzm, fzp, rdx, rdy, rdz, &
         ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
         ids, ide, jds, jde, kds, kde)
  case default
    error stop 'unknown mode'
  end select

  fb = 0; rub = 0; rvb = 0; romb = 0; oldb = 0; mutb = 0
  select case (mode)
  case (1)
    call a_advect_scalar(f, fb, f, oldb, tend, tendb, ru, rub, rv, rvb, rom, romb, mut, 1, &
         cf, msfux, msfuy, msfvx, msfvy, msftx, msfty, fzm, fzp, rdx, rdy, rdz, &
         ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
         ids, ide, jds, jde, kds, kde)
  case (2)
    call a_advect_u(f, fb, f, oldb, tend, tendb, ru, rub, rv, rvb, rom, romb, mut, mutb, 1, &
         cf, msfux, msfuy, msfvx, msfvy, msftx, msfty, fzm, fzp, rdx, rdy, rdz, &
         ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
         ids, ide, jds, jde, kds, kde)
  case (3)
    call a_advect_v(f, fb, f, oldb, tend, tendb, ru, rub, rv, rvb, rom, romb, mut, mutb, 1, &
         cf, msfux, msfuy, msfvx, msfvy, msftx, msfty, fzm, fzp, rdx, rdy, rdz, &
         ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
         ids, ide, jds, jde, kds, kde)
  case (4)
    call a_advect_w(f, fb, f, oldb, tend, tendb, ru, rub, rv, rvb, rom, romb, mut, 1, &
         cf, msfux, msfuy, msfvx, msfvy, msftx, msfty, fzm, fzp, rdx, rdy, rdz, &
         ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
         ids, ide, jds, jde, kds, kde)
  end select

  open(newunit=iu, file=trim(fout), access='stream', form='unformatted', status='replace')
  write(iu) tend, td, fb + oldb, rub, rvb, romb
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

end program wrfplus_advect
