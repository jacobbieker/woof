! WRFPLUS 4.7.1 oracle driver: g_calc_coef_w and a_calc_coef_w on one input
! record (docs/dev/P3A-WRF-DRYCORE-ADJOINT.md section 4).  Built against
! WRFPLUS's libwrflib.a (configure wrfplus, real*8, -O0 -ffp-contract=off)
! by tools/wrfplus_oracle/build.sh; driven by tools/wrfplus_oracle/fixture.py.
!
! Input  (stream, native binary64/int32): see fixture.py _cw_record.
! Output (stream): TL a, ad, alpha, alphad, gamma, gammad; then the AD
! a_a, a_alpha, a_gamma, a_c2a, a_cqw and a_mut.
program wrfplus_calc_coef_w
  use g_module_small_step_em, only : g_calc_coef_w
  use a_module_small_step_em, only : a_calc_coef_w
  implicit none
  integer :: nx, ny, nz, itop, iu
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
  real :: dts, epssm
  real, parameter :: g = 9.81
  logical :: top_lid
  real, allocatable, dimension(:,:,:) :: a, ad, alpha, alphad, gam, gamd, cqw, cqwd, &
       c2a, c2ad, sa, salpha, sgam, a_c2a, a_cqw, a0, alpha0, gam0
  real, allocatable, dimension(:,:) :: mut, mutd, a_mut
  real, allocatable, dimension(:) :: rdn, rdnw
  character(len=512) :: fin, fout

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(newunit=iu, file=trim(fin), access='stream', form='unformatted', status='old')
  read(iu) nx, ny, nz, itop
  read(iu) dts, epssm
  top_lid = itop /= 0
  ids = 1; ide = nx + 1; jds = 1; jde = ny + 1; kds = 1; kde = nz + 1
  ims = 0; ime = nx + 2; jms = 0; jme = ny + 2; kms = 1; kme = nz + 1
  call a3(a); call a3(ad); call a3(alpha); call a3(alphad); call a3(gam); call a3(gamd)
  call a3(cqw); call a3(cqwd); call a3(c2a); call a3(c2ad)
  call a3(sa); call a3(salpha); call a3(sgam)
  call a2(mut); call a2(mutd)
  allocate(rdn(kms:kme), rdnw(kms:kme))
  read(iu) rdn, rdnw
  close(iu)

  allocate(a0, source=a); allocate(alpha0, source=alpha); allocate(gam0, source=gam)
  call g_calc_coef_w(a, ad, alpha, alphad, gam, gamd, mut, mutd, cqw, cqwd, rdn, rdnw, &
       c2a, c2ad, dts, g, epssm, top_lid, ids, ide, jds, jde, kds, kde, ims, ime, &
       jms, jme, kms, kme, ids, ide, jds, jde, kds, kde)

  allocate(a_c2a, a_cqw, mold=a); a_c2a = 0; a_cqw = 0
  allocate(a_mut, mold=mut); a_mut = 0
  call a_calc_coef_w(a0, sa, alpha0, salpha, gam0, sgam, mut, a_mut, cqw, a_cqw, rdn, &
       rdnw, c2a, a_c2a, dts, g, epssm, top_lid, ids, ide, jds, jde, kds, kde, ims, &
       ime, jms, jme, kms, kme, ids, ide, jds, jde, kds, kde)

  open(newunit=iu, file=trim(fout), access='stream', form='unformatted', status='replace')
  write(iu) a, ad, alpha, alphad, gam, gamd
  write(iu) sa, salpha, sgam, a_c2a, a_cqw
  write(iu) a_mut
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

end program wrfplus_calc_coef_w
