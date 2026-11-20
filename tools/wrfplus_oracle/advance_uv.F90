! WRFPLUS 4.7.1 oracle driver: g_advance_uv and a_advance_uv on one input
! record (docs/dev/P3A-WRF-DRYCORE-ADJOINT.md section 4).  Built against
! WRFPLUS's libwrflib.a (configure wrfplus, real*8, -O0 -ffp-contract=off)
! by tools/wrfplus_oracle/build.sh; driven by tools/wrfplus_oracle/fixture.py.
!
! Input  (stream, native binary64/int32): see fixture.py UV_FIELDS.
! Output (stream): TL u, ud, v, vd, then the AD adjoints in UV_ADJOINTS order.
program wrfplus_advance_uv
  use module_configure, only : grid_config_rec_type
  use g_module_small_step_em, only : g_advance_uv
  use a_module_small_step_em, only : a_advance_uv
  implicit none
  integer :: nx, ny, nz, spec_zone, itop, iu
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
  real :: rdx, rdy, dts, cf1, cf2, cf3, emdiv
  type(grid_config_rec_type) :: cf
  logical :: top_lid
  real, allocatable, dimension(:,:,:) :: u, ud, ru, rud, v, vd, rv, rvd, p, pd, pb, &
       ph, phd, php, phpd, alt, altd, al, ald, cqu, cqud, cqv, cqvd, au, av
  real, allocatable, dimension(:,:,:) :: u0, v0, a_ru, a_rv, a_p, a_ph, a_php, a_alt, &
       a_al, a_cqu, a_cqv
  real, allocatable, dimension(:,:) :: mu, mud, muu, muud, muv, muvd, mudf, mudfd, &
       msfux, msfuy, msfvx, msfvx_inv, msfvy, a_mu, a_muu, a_muv, a_mudf
  real, allocatable, dimension(:) :: fnm, fnp, rdnw
  character(len=512) :: fin, fout

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(newunit=iu, file=trim(fin), access='stream', form='unformatted', status='old')
  read(iu) nx, ny, nz, spec_zone, itop
  read(iu) rdx, rdy, dts, cf1, cf2, cf3, emdiv
  top_lid = itop /= 0
  ids = 1; ide = nx + 1; jds = 1; jde = ny + 1; kds = 1; kde = nz + 1
  ims = 0; ime = nx + 2; jms = 0; jme = ny + 2; kms = 1; kme = nz + 1
  call a3(u); call a3(ud); call a3(ru); call a3(rud); call a3(v); call a3(vd)
  call a3(rv); call a3(rvd); call a3(p); call a3(pd); call a3(pb); call a3(ph)
  call a3(phd); call a3(php); call a3(phpd); call a3(alt); call a3(altd)
  call a3(al); call a3(ald); call a3(cqu); call a3(cqud); call a3(cqv); call a3(cqvd)
  call a3(au); call a3(av)
  call a2(mu); call a2(mud); call a2(muu); call a2(muud); call a2(muv); call a2(muvd)
  call a2(mudf); call a2(mudfd); call a2(msfux); call a2(msfuy); call a2(msfvx)
  call a2(msfvx_inv); call a2(msfvy)
  allocate(fnm(kms:kme), fnp(kms:kme), rdnw(kms:kme))
  read(iu) fnm, fnp, rdnw
  close(iu)

  cf%specified = .true.;   cf%nested = .false.;   cf%periodic_x = .false.
  cf%open_xs = .false.;    cf%open_xe = .false.;  cf%open_ys = .false.
  cf%open_ye = .false.;    cf%symmetric_xs = .false.; cf%symmetric_xe = .false.
  cf%symmetric_ys = .false.; cf%symmetric_ye = .false.; cf%polar = .false.

  allocate(u0, source=u); allocate(v0, source=v)
  call g_advance_uv(u, ud, ru, rud, v, vd, rv, rvd, p, pd, pb, ph, phd, php, phpd, &
       alt, altd, al, ald, mu, mud, muu, muud, cqu, cqud, muv, muvd, cqv, cqvd, &
       mudf, mudfd, msfux, msfuy, msfvx, msfvx_inv, msfvy, rdx, rdy, dts, cf1, cf2, &
       cf3, fnm, fnp, emdiv, rdnw, cf, spec_zone, .true., top_lid, ids, ide, jds, jde, &
       kds, kde, ims, ime, jms, jme, kms, kme, ids, ide, jds, jde, kds, kde)

  allocate(a_ru, a_rv, a_p, a_ph, a_php, a_alt, a_al, a_cqu, a_cqv, mold=u)
  allocate(a_mu, a_muu, a_muv, a_mudf, mold=mu)
  a_ru = 0; a_rv = 0; a_p = 0; a_ph = 0; a_php = 0; a_alt = 0; a_al = 0
  a_cqu = 0; a_cqv = 0; a_mu = 0; a_muu = 0; a_muv = 0; a_mudf = 0
  call a_advance_uv(u0, au, ru, a_ru, v0, av, rv, a_rv, p, a_p, pb, ph, a_ph, php, &
       a_php, alt, a_alt, al, a_al, mu, a_mu, muu, a_muu, cqu, a_cqu, muv, a_muv, &
       cqv, a_cqv, mudf, a_mudf, msfux, msfuy, msfvx, msfvx_inv, msfvy, rdx, rdy, &
       dts, cf1, cf2, cf3, fnm, fnp, emdiv, rdnw, cf, spec_zone, .true., top_lid, &
       ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ids, ide, jds, &
       jde, kds, kde)

  open(newunit=iu, file=trim(fout), access='stream', form='unformatted', status='replace')
  write(iu) u, ud, v, vd
  write(iu) au, av, a_ru, a_rv, a_p, a_ph, a_php, a_alt, a_al, a_cqu, a_cqv
  write(iu) a_mu, a_muu, a_muv, a_mudf
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

end program wrfplus_advance_uv
