! WRFPLUS 4.7.1 oracle driver: g_advance_mu_t then g_spec_bdyupdate of t_2,
! mu_2 and muts (solve_em's order), and a_spec_bdyupdate (muts, mu_2, t_2)
! then a_advance_mu_t, on one input record
! (docs/dev/P3A-WRF-DRYCORE-ADJOINT.md section 4).  Built against WRFPLUS's
! libwrflib.a (configure wrfplus, real*8, -O0 -ffp-contract=off) by
! tools/wrfplus_oracle/build.sh; driven by tools/wrfplus_oracle/fixture.py.
!
! Input  (stream, native binary64/int32): see fixture.py _mt_record.
! Output (stream): TL ww, wwd, t, td, tave, taved; mu, mud, muts, mutsd,
! mudf, mudfd; then the AD a_ww, a_ww1, a_u, a_u1, a_v, a_v1, a_t, a_t1,
! a_tave, a_ft; a_mu, a_mut, a_muave, a_muts, a_muu, a_muv, a_mudf,
! a_mu_tend.
program wrfplus_advance_mu_t
  use module_configure, only : grid_config_rec_type
  use g_module_small_step_em, only : g_advance_mu_t
  use a_module_small_step_em, only : a_advance_mu_t
  use g_module_bc, only : g_spec_bdyupdate
  use a_module_bc, only : a_spec_bdyupdate
  implicit none
  integer :: nx, ny, nz, spec_zone, iu
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
  real :: rdx, rdy, dts, epssm
  type(grid_config_rec_type) :: cf
  real, allocatable, dimension(:,:,:) :: ww, wwd, ww1, ww1d, u, ud, u1, u1d, v, vd, &
       v1, v1d, t, td, t1, t1d, ft, ftd, sww, st, stave, tave, taved, uam, vam, wwam, &
       uamd, vamd, wwamd, ww0, t0, tave0
  real, allocatable, dimension(:,:,:) :: a_ww1, a_u, a_u1, a_v, a_v1, a_t1, a_ft, &
       a_uam, a_vam, a_wwam
  real, allocatable, dimension(:,:) :: mu, mud, mut, mutd, muu, muud, muv, muvd, muts, &
       mutsd, mu_tend, mu_tendd, msft, msfu, msfv, smu, smuts, smudf, one, muave, &
       muaved, mudf, mudfd, mu0, muts0, muave0, mudf0
  real, allocatable, dimension(:,:) :: a_mut, a_muave, a_muu, a_muv, a_mu_tend
  real, allocatable, dimension(:) :: fnm, fnp, dnw, rdnw
  character(len=512) :: fin, fout

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(newunit=iu, file=trim(fin), access='stream', form='unformatted', status='old')
  read(iu) nx, ny, nz, spec_zone
  read(iu) rdx, rdy, dts, epssm
  ids = 1; ide = nx + 1; jds = 1; jde = ny + 1; kds = 1; kde = nz + 1
  ims = 0; ime = nx + 2; jms = 0; jme = ny + 2; kms = 1; kme = nz + 1
  call a3(ww); call a3(wwd); call a3(ww1); call a3(ww1d); call a3(u); call a3(ud)
  call a3(u1); call a3(u1d); call a3(v); call a3(vd); call a3(v1); call a3(v1d)
  call a3(t); call a3(td); call a3(t1); call a3(t1d); call a3(ft); call a3(ftd)
  call a3(sww); call a3(st); call a3(stave)
  call a2(mu); call a2(mud); call a2(mut); call a2(mutd); call a2(muu); call a2(muud)
  call a2(muv); call a2(muvd); call a2(muts); call a2(mutsd); call a2(mu_tend)
  call a2(mu_tendd); call a2(msft); call a2(msfu); call a2(msfv)
  call a2(smu); call a2(smuts); call a2(smudf)
  allocate(fnm(kms:kme), fnp(kms:kme), dnw(kms:kme), rdnw(kms:kme))
  read(iu) fnm, fnp, dnw, rdnw
  close(iu)

  cf%specified = .true.;   cf%nested = .false.;   cf%periodic_x = .false.
  cf%open_xs = .false.;    cf%open_xe = .false.;  cf%open_ys = .false.
  cf%open_ye = .false.;    cf%symmetric_xs = .false.; cf%symmetric_xe = .false.
  cf%symmetric_ys = .false.; cf%symmetric_ye = .false.; cf%polar = .false.

  allocate(one, mold=mu); one = 1.
  allocate(muave, muaved, mudf, mudfd, mold=mu)
  muave = 0; muaved = 0; mudf = 0; mudfd = 0
  allocate(tave, taved, uam, vam, wwam, uamd, vamd, wwamd, mold=ww)
  tave = 0; taved = 0; uam = 0; vam = 0; wwam = 0; uamd = 0; vamd = 0; wwamd = 0
  allocate(ww0, source=ww); allocate(t0, source=t); allocate(tave0, source=tave)
  allocate(mu0, source=mu); allocate(muts0, source=muts); allocate(muave0, source=muave)
  allocate(mudf0, source=mudf)

  call g_advance_mu_t(ww, wwd, ww1, ww1d, u, ud, u1, u1d, v, vd, v1, v1d, mu, mud, &
       mut, mutd, muave, muaved, muts, mutsd, muu, muud, muv, muvd, mudf, mudfd, &
       uam, uamd, vam, vamd, wwam, wwamd, t, td, t1, t1d, tave, taved, ft, ftd, &
       mu_tend, mu_tendd, rdx, rdy, dts, epssm, dnw, fnm, fnp, rdnw, one, msfu, one, &
       msfv, one, msft, msft, 1, cf, ids, ide, jds, jde, kds, kde, ims, ime, jms, &
       jme, kms, kme, ids, ide, jds, jde, kds, kde)
  call g_spec_bdyupdate(t, td, ft, ftd, dts, 't', cf, spec_zone, ids, ide, jds, jde, &
       kds, kde, ims, ime, jms, jme, kms, kme, ids, ide, jds, jde, kds, kde, ids, ide, &
       jds, jde, kds, kde)
  call g_spec_bdyupdate(mu, mud, mu_tend, mu_tendd, dts, 'm', cf, spec_zone, ids, ide, &
       jds, jde, 1, 1, ims, ime, jms, jme, 1, 1, ids, ide, jds, jde, 1, 1, ids, ide, &
       jds, jde, 1, 1)
  call g_spec_bdyupdate(muts, mutsd, mu_tend, mu_tendd, dts, 'm', cf, spec_zone, ids, &
       ide, jds, jde, 1, 1, ims, ime, jms, jme, 1, 1, ids, ide, jds, jde, 1, 1, ids, &
       ide, jds, jde, 1, 1)

  open(newunit=iu, file=trim(fout), access='stream', form='unformatted', status='replace')
  write(iu) ww, wwd, t, td, tave, taved
  write(iu) mu, mud, muts, mutsd, mudf, mudfd

  ! Adjoint, from the pre-TL base state; sww/st/stave/smu/smuts/smudf are
  ! the output adjoints (muave's is zero: woof has no muave output).
  allocate(a_ww1, a_u, a_u1, a_v, a_v1, a_t1, a_ft, a_uam, a_vam, a_wwam, mold=ww)
  a_ww1 = 0; a_u = 0; a_u1 = 0; a_v = 0; a_v1 = 0; a_t1 = 0; a_ft = 0
  a_uam = 0; a_vam = 0; a_wwam = 0
  allocate(a_mut, a_muave, a_muu, a_muv, a_mu_tend, mold=mu)
  a_mut = 0; a_muave = 0; a_muu = 0; a_muv = 0; a_mu_tend = 0
  call a_spec_bdyupdate(smuts, a_mu_tend, dts, 'm', cf, spec_zone, ids, ide, jds, jde, &
       1, 1, ims, ime, jms, jme, 1, 1, ids, ide, jds, jde, 1, 1, ids, ide, jds, jde, 1, 1)
  call a_spec_bdyupdate(smu, a_mu_tend, dts, 'm', cf, spec_zone, ids, ide, jds, jde, &
       1, 1, ims, ime, jms, jme, 1, 1, ids, ide, jds, jde, 1, 1, ids, ide, jds, jde, 1, 1)
  call a_spec_bdyupdate(st, a_ft, dts, 't', cf, spec_zone, ids, ide, jds, jde, kds, &
       kde, ims, ime, jms, jme, kms, kme, ids, ide, jds, jde, kds, kde, ids, ide, jds, &
       jde, kds, kde)
  call a_advance_mu_t(ww0, sww, ww1, a_ww1, u, a_u, u1, a_u1, v, a_v, v1, a_v1, mu0, &
       smu, mut, a_mut, muave0, a_muave, muts0, smuts, muu, a_muu, muv, a_muv, mudf0, &
       smudf, uam, a_uam, vam, a_vam, wwam, a_wwam, t0, st, t1, a_t1, tave0, stave, &
       ft, a_ft, mu_tend, a_mu_tend, rdx, rdy, dts, epssm, dnw, fnm, fnp, rdnw, one, &
       msfu, one, msfv, one, msft, msft, 1, cf, ids, ide, jds, jde, kds, kde, ims, &
       ime, jms, jme, kms, kme, ids, ide, jds, jde, kds, kde)

  write(iu) sww, a_ww1, a_u, a_u1, a_v, a_v1, st, a_t1, stave, a_ft
  write(iu) smu, a_mut, a_muave, smuts, a_muu, a_muv, smudf, a_mu_tend
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

end program wrfplus_advance_mu_t
