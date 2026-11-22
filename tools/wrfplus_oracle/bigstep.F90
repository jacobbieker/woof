! WRFPLUS 4.7.1 oracle driver: G_/A_HORIZONTAL_PRESSURE_GRADIENT and
! g_/a_pg_buoy_w on one input record
! (docs/dev/P3A-WRF-DRYCORE-ADJOINT.md section 5, item 5).  Built against
! WRFPLUS's libwrflib.a (configure wrfplus, real*8, -O0 -ffp-contract=off)
! by tools/wrfplus_oracle/build.sh; driven by tools/wrfplus_oracle/fixture.py.
!
! Input (stream, native binary64/int32): mode (1 pressure gradient,
! 2 pg_buoy_w), nx, ny, nz, flags (bit 0 periodic_x, 1 periodic_y,
! 2 specified, 4-7 open_xs/xe/ys/ye), top_lid; rdx, rdy, cf1, cf2, cf3,
! cfn, cfn1; then on (-3:nx+4, 1:nz+1, -3:ny+4) ph, phd, alt, altd, p, pd,
! pb, al, ald, php, phpd, ru_tend, its adjoint seed, rv_tend, its seed,
! rw_tend, its seed; on (-3:nx+4, -3:ny+4) muu, muud, muv, muvd, mu, mud,
! mub, msft; on 1:nz+1 fnm, fnp, rdnw, rdn.  cqu = cqv = 1, cqw = 0 (dry),
! the u/v map-factor ratios 1.  In A_HORIZONTAL_PRESSURE_GRADIENT pb0 is
! the adjoint of p (Tapenade renamed it past the base pb) and phb, mub
! the adjoints of ph, mu.
! Output (stream): TL ru, ru_d, rv, rv_d, rw, rw_d; AD ph, alt, p, al,
! php, muu, muv, mu adjoints.
program wrfplus_bigstep
  use module_configure, only : grid_config_rec_type
  use g_module_big_step_utilities_em, only : g_horizontal_pressure_gradient, g_pg_buoy_w
  use a_module_big_step_utilities_em, only : a_horizontal_pressure_gradient, a_pg_buoy_w
  implicit none
  integer :: mode, nx, ny, nz, flags, iu, hdr(6)
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
  real :: rdx, rdy, cf1, cf2, cf3, cfn, cfn1, sc(7)
  logical :: top_lid
  type(grid_config_rec_type) :: cf
  real, allocatable, dimension(:,:,:) :: ph, phd, alt, altd, p, pd, pb, al, ald, php, phpd, &
       ru, rua, rv, rva, rw, rwa, rud, rvd, rwd, one3, zero3, cqw, &
       phb, altb, pbar, alb, phpb, cqb1, cqb2, cqwb
  real, allocatable, dimension(:,:) :: muu, muud, muv, muvd, mu, mud, mub, msft, one2, &
       muub, muvb, mubar
  real, allocatable, dimension(:) :: fnm, fnp, rdnw, rdn
  character(len=512) :: fin, fout

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(newunit=iu, file=trim(fin), access='stream', form='unformatted', status='old')
  read(iu) hdr
  mode = hdr(1); nx = hdr(2); ny = hdr(3); nz = hdr(4); flags = hdr(5)
  top_lid = hdr(6) /= 0
  read(iu) sc
  rdx = sc(1); rdy = sc(2); cf1 = sc(3); cf2 = sc(4); cf3 = sc(5); cfn = sc(6); cfn1 = sc(7)
  ids = 1; ide = nx + 1; jds = 1; jde = ny + 1; kds = 1; kde = nz + 1
  ims = -3; ime = nx + 4; jms = -3; jme = ny + 4; kms = 1; kme = nz + 1
  call a3(ph); call a3(phd); call a3(alt); call a3(altd); call a3(p); call a3(pd)
  call a3(pb); call a3(al); call a3(ald); call a3(php); call a3(phpd)
  call a3(ru); call a3(rua); call a3(rv); call a3(rva); call a3(rw); call a3(rwa)
  call a2(muu); call a2(muud); call a2(muv); call a2(muvd); call a2(mu); call a2(mud)
  call a2(mub); call a2(msft)
  allocate(fnm(kms:kme), fnp(kms:kme), rdnw(kms:kme), rdn(kms:kme))
  read(iu) fnm, fnp, rdnw, rdn
  close(iu)

  cf%periodic_x = btest(flags, 0); cf%periodic_y = btest(flags, 1)
  cf%specified = btest(flags, 2);  cf%nested = .false.
  cf%open_xs = btest(flags, 4);    cf%open_xe = btest(flags, 5)
  cf%open_ys = btest(flags, 6);    cf%open_ye = btest(flags, 7)
  cf%symmetric_xs = .false.; cf%symmetric_xe = .false.
  cf%symmetric_ys = .false.; cf%symmetric_ye = .false.; cf%polar = .false.

  allocate(rud, rvd, rwd, one3, zero3, cqw, phb, altb, pbar, alb, phpb, cqb1, cqb2, cqwb, mold=ph)
  allocate(one2, muub, muvb, mubar, mold=mu)
  rud = 0; rvd = 0; rwd = 0; one3 = 1; zero3 = 0; one2 = 1
  phb = 0; altb = 0; pbar = 0; alb = 0; phpb = 0; cqb1 = 0; cqb2 = 0; cqwb = 0
  muub = 0; muvb = 0; mubar = 0
  select case (mode)
  case (1)
    call g_horizontal_pressure_gradient(ru, rud, rv, rvd, ph, phd, alt, altd, p, pd, pb, &
         al, ald, php, phpd, one3, zero3, one3, zero3, muu, muud, muv, muvd, mu, mud, &
         fnm, fnp, rdnw, cf1, cf2, cf3, cfn, cfn1, rdx, rdy, one2, one2, one2, one2, &
         one2, one2, cf, .true., top_lid, &
         ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
         ids, ide, jds, jde, kds, kde)
    call a_horizontal_pressure_gradient(ru, rua, rv, rva, ph, phb, alt, altb, p, pbar, pb, &
         al, alb, php, phpb, one3, cqb1, one3, cqb2, muu, muub, muv, muvb, mu, mubar, &
         fnm, fnp, rdnw, cf1, cf2, cf3, cfn, cfn1, rdx, rdy, one2, one2, one2, one2, &
         one2, one2, cf, .true., top_lid, &
         ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
         ids, ide, jds, jde, kds, kde)
  case (2)
    cqw = 0
    call g_pg_buoy_w(rw, rwd, p, pd, cqw, zero3, mu, mud, mub, rdnw, rdn, 9.81, msft, msft, &
         ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
         ids, ide, jds, jde, kds, kde)
    cqw = 0
    call a_pg_buoy_w(rw, rwa, p, pbar, cqw, cqwb, mu, mubar, mub, rdnw, rdn, 9.81, msft, msft, &
         ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
         ids, ide, jds, jde, kds, kde)
  case default
    error stop 'unknown mode'
  end select

  open(newunit=iu, file=trim(fout), access='stream', form='unformatted', status='replace')
  write(iu) ru, rud, rv, rvd, rw, rwd, phb, altb, pbar, alb, phpb, muub, muvb, mubar
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

end program wrfplus_bigstep
