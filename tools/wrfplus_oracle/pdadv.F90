! WRFPLUS 4.7.1 oracle driver: positive-definite scalar advection
! (moist_adv_opt = 1; docs/dev/P3A-WRF-DRYCORE-ADJOINT.md section 5,
! item 13).  Built against WRFPLUS's libwrflib.a (configure wrfplus,
! real*8, -O0 -ffp-contract=off) by tools/wrfplus_oracle/build.sh; driven
! by tools/wrfplus_oracle/fixture.py.
!
! Mode 1: G_ADVECT_SCALAR_PD then A_ADVECT_SCALAR_PD (WRFPLUS's TL/AD,
! which linearise the pre-hybrid routine: column masses mut and
! mub + mu_old without c1/c2).  Mode 2: WRF 4.7.1's nonlinear
! advect_scalar_pd (dyn_em/module_advect_em.F, with c1/c2) on the base
! fields only; the fixture differences it.
!
! Input (stream, native binary64/int32): mode, nx, ny, nz, flags (bit 0
! periodic_x, 1 periodic_y, 2 specified), h_sca, v_sca, time_step, 0, 0;
! rdx, rdy, dt; on (-3:nx+4, 1:nz+1, -3:ny+4) field, fieldd, field_old,
! field_oldd, ru, rud, rv, rvd, rom, romd and the tendency adjoint seed;
! on (-3:nx+4, -3:ny+4) mut, mutd, mub, mu_old, mu_oldd, msfux, msfuy,
! msfvx, msfvy, msftx, msfty; on 1:nz+1 fzm, fzp, rdzw, c1, c2.
! Output (stream): mode 1 the TL tendency and tendencyd (from zero), the
! AD field, field_old, ru, rv, rom, mut and mu_old adjoints; mode 2 the
! tendency (from zero).
program wrfplus_pdadv
  use module_configure, only : grid_config_rec_type
  use module_advect_em, only : advect_scalar_pd
  use g_module_advect_em, only : g_advect_scalar_pd
  use a_module_advect_em, only : a_advect_scalar_pd
  implicit none
  integer :: mode, nx, ny, nz, flags, iu, hdr(10), tstep
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
  real :: rdx, rdy, dt
  type(grid_config_rec_type) :: cf
  real, allocatable, dimension(:,:,:) :: f, fd, fo, fod, ru, rud, rv, rvd, rom, romd, &
       tendb, tend, td, ht, htd, zt, ztd, fb, fob, rub, rvb, romb
  real, allocatable, dimension(:,:) :: mut, mutd, mub, muo, muod, msfux, msfuy, msfvx, &
       msfvy, msftx, msfty, mutb, muob
  real, allocatable, dimension(:) :: fzm, fzp, rdz, c1, c2
  character(len=512) :: fin, fout

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(newunit=iu, file=trim(fin), access='stream', form='unformatted', status='old')
  read(iu) hdr
  mode = hdr(1); nx = hdr(2); ny = hdr(3); nz = hdr(4); flags = hdr(5); tstep = hdr(8)
  read(iu) rdx, rdy, dt
  ids = 1; ide = nx + 1; jds = 1; jde = ny + 1; kds = 1; kde = nz + 1
  ims = -3; ime = nx + 4; jms = -3; jme = ny + 4; kms = 1; kme = nz + 1
  call a3(f); call a3(fd); call a3(fo); call a3(fod); call a3(ru); call a3(rud)
  call a3(rv); call a3(rvd); call a3(rom); call a3(romd); call a3(tendb)
  call a2(mut); call a2(mutd); call a2(mub); call a2(muo); call a2(muod)
  call a2(msfux); call a2(msfuy); call a2(msfvx); call a2(msfvy); call a2(msftx)
  call a2(msfty)
  allocate(fzm(kms:kme), fzp(kms:kme), rdz(kms:kme), c1(kms:kme), c2(kms:kme))
  read(iu) fzm, fzp, rdz, c1, c2
  close(iu)

  cf%periodic_x = btest(flags, 0); cf%periodic_y = btest(flags, 1)
  cf%specified = btest(flags, 2);  cf%nested = .false.
  cf%open_xs = .false.; cf%open_xe = .false.; cf%open_ys = .false.; cf%open_ye = .false.
  cf%symmetric_xs = .false.; cf%symmetric_xe = .false.
  cf%symmetric_ys = .false.; cf%symmetric_ye = .false.; cf%polar = .false.
  cf%h_sca_adv_order = hdr(6); cf%v_sca_adv_order = hdr(7)
  cf%h_mom_adv_order = 5; cf%v_mom_adv_order = 3

  allocate(tend, td, ht, htd, zt, ztd, fb, fob, rub, rvb, romb, mold=f)
  allocate(mutb, muob, mold=mut)
  tend = 0; td = 0; ht = 0; htd = 0; zt = 0; ztd = 0
  open(newunit=iu, file=trim(fout), access='stream', form='unformatted', status='replace')
  select case (mode)
  case (1)
    call g_advect_scalar_pd(f, fd, fo, fod, tend, td, ht, htd, zt, ztd, ru, rud, rv, rvd, &
         rom, romd, mut, mutd, mub, muo, muod, tstep, cf, .false., msfux, msfuy, msfvx, &
         msfvy, msftx, msfty, fzm, fzp, rdx, rdy, rdz, dt, &
         ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
         ids, ide, jds, jde, kds, kde)
    write(iu) tend, td
    fb = 0; fob = 0; rub = 0; rvb = 0; romb = 0; mutb = 0; muob = 0; htd = 0; ztd = 0
    tend = 0
    call a_advect_scalar_pd(f, fb, fo, fob, tend, tendb, ht, htd, zt, ztd, ru, rub, rv, &
         rvb, rom, romb, mut, mutb, mub, muo, muob, tstep, cf, .false., msfux, msfuy, &
         msfvx, msfvy, msftx, msfty, fzm, fzp, rdx, rdy, rdz, dt, &
         ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
         ids, ide, jds, jde, kds, kde)
    write(iu) fb, fob, rub, rvb, romb, mutb, muob
  case (2)
    call advect_scalar_pd(f, fo, tend, ht, zt, ru, rv, rom, c1, c2, mut, mub, muo, tstep, &
         cf, .false., msfux, msfuy, msfvx, msfvy, msftx, msfty, fzm, fzp, rdx, rdy, &
         rdz, dt, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
         ids, ide, jds, jde, kds, kde)
    write(iu) tend
  case default
    error stop 'unknown mode'
  end select
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

end program wrfplus_pdadv
