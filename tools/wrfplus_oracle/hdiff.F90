! WRFPLUS 4.7.1 oracle driver: the diff_opt 1 horizontal-diffusion rows
! (docs/dev/P3A-WRF-DRYCORE-ADJOINT.md section 5, item 5).
! Built against WRFPLUS's libwrflib.a (configure wrfplus, real*8, -O0
! -ffp-contract=off) by tools/wrfplus_oracle/build.sh; driven by
! tools/wrfplus_oracle/fixture.py.
!
! As rk_tendency_tl/_ad at diff_opt 1: G_HORIZONTAL_DIFFUSION ('u', 'v',
! 'w' or 'm', a scalar) or g_horizontal_diffusion_3dmp ('m', theta
! minus t_init), then A_HORIZONTAL_DIFFUSION or
! a_horizontal_diffusion_3dmp.  WRFPLUS's rows take mu as the mass (no
! c1/c2).  The map factors are msfux = msfuy = msfu, msfvx = msfvy = msfv,
! msfvx_inv = 1/msfv, msftx = msfty = msft.
!
! Input (stream, native binary64/int32): nx, ny, nz, flags (bit 0
! periodic_x, 1 periodic_y, 2 specified; bits 4-6 the row: 1 u, 2 v,
! 3 w, 4 theta (3dmp), 5 scalar), 0; rdx, rdy; then on
! (-3:nx+4, 1:nz+1, -3:ny+4): field, its tangent, the coefficient, its
! tangent, the base field (3dmp), the tendency, its tangent, the tendency
! adjoint seed; on (-3:nx+4, -3:ny+4) mu, its tangent, msfu, msfv, msft.
! Output (stream): the tendency, its tangent, the tendency adjoint after
! the sweep, the field and coefficient adjoints, then the mu adjoint.
program wrfplus_hdiff
  use module_configure, only : grid_config_rec_type
  use g_module_big_step_utilities_em, only : g_horizontal_diffusion, g_horizontal_diffusion_3dmp
  use a_module_big_step_utilities_em, only : a_horizontal_diffusion, a_horizontal_diffusion_3dmp
  implicit none
  integer :: nx, ny, nz, iu, hdr(5), flags, row
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
  real :: sc(2), rdx, rdy, khdif
  character(len=1) :: name
  type(grid_config_rec_type) :: cf
  real, allocatable, dimension(:,:,:) :: f, fd, xk, xkd, base, tend, tendd, atend, af, axk
  real, allocatable, dimension(:,:) :: mu, mud, amu, msfu, msfv, msft, msfvi
  character(len=512) :: fin, fout

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(newunit=iu, file=trim(fin), access='stream', form='unformatted', status='old')
  read(iu) hdr
  nx = hdr(1); ny = hdr(2); nz = hdr(3); flags = hdr(4)
  row = iand(ishft(flags, -4), 7)
  read(iu) sc
  rdx = sc(1); rdy = sc(2); khdif = 0.
  ids = 1; ide = nx + 1; jds = 1; jde = ny + 1; kds = 1; kde = nz + 1
  ims = -3; ime = nx + 4; jms = -3; jme = ny + 4; kms = 1; kme = nz + 1

  cf%periodic_x = btest(flags, 0); cf%periodic_y = btest(flags, 1)
  cf%specified = btest(flags, 2);  cf%nested = .false.
  cf%open_xs = .false.; cf%open_xe = .false.; cf%open_ys = .false.; cf%open_ye = .false.
  cf%symmetric_xs = .false.; cf%symmetric_xe = .false.
  cf%symmetric_ys = .false.; cf%symmetric_ye = .false.; cf%polar = .false.
  cf%diff_opt = 1; cf%km_opt = 4

  call a3(f); call a3(fd); call a3(xk); call a3(xkd); call a3(base)
  call a3(tend); call a3(tendd); call a3(atend)
  call a2(mu); call a2(mud); call a2(msfu); call a2(msfv); call a2(msft)
  close(iu)
  allocate(msfvi, mold=msfv)
  msfvi = 1. / msfv
  allocate(af, axk, mold=f)
  allocate(amu, mold=mu)
  af = 0; axk = 0; amu = 0
  select case (row)
  case (1); name = 'u'
  case (2); name = 'v'
  case (3); name = 'w'
  case default; name = 'm'
  end select

  if (row == 4) then
    call g_horizontal_diffusion_3dmp(name, f, fd, tend, tendd, mu, mud, cf, base, &
         msfu, msfu, msfv, msfvi, msfv, msft, msft, khdif, xk, xkd, rdx, rdy, &
         ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
         ids, ide, jds, jde, kds, kde)
    call a_horizontal_diffusion_3dmp(name, f, af, tend, atend, mu, amu, cf, base, &
         msfu, msfu, msfv, msfvi, msfv, msft, msft, khdif, xk, axk, rdx, rdy, &
         ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
         ids, ide, jds, jde, kds, kde)
  else
    call g_horizontal_diffusion(name, f, fd, tend, tendd, mu, mud, cf, &
         msfu, msfu, msfv, msfvi, msfv, msft, msft, khdif, xk, xkd, rdx, rdy, &
         ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
         ids, ide, jds, jde, kds, kde)
    call a_horizontal_diffusion(name, f, af, tend, atend, mu, amu, cf, &
         msfu, msfu, msfv, msfvi, msfv, msft, msft, khdif, xk, axk, rdx, rdy, &
         ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
         ids, ide, jds, jde, kds, kde)
  end if

  open(newunit=iu, file=trim(fout), access='stream', form='unformatted', status='replace')
  write(iu) tend, tendd, atend, af, axk, amu
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

end program wrfplus_hdiff
