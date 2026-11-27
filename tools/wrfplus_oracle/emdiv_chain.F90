! WRFPLUS 4.7.1 oracle driver for the external-mode filter's chain
! (emdiv > 0; docs/dev/P3A-WRF-DRYCORE-ADJOINT.md section 5): nsub
! acoustic substeps of g_advance_uv, then g_advance_mu_t with
! g_spec_bdyupdate of t_2, mu_2 and muts (solve_em's order), carrying u,
! v, mu, t and MUDF between them, so each substep's advance_uv reads the
! MUDF the previous substep's advance_mu_t wrote.  The adjoint runs the
! chain backwards (a_spec_bdyupdate, a_advance_mu_t, a_advance_uv per
! substep) on the bases the TL saved before each call.  advance_w is not
! in the chain: the pressure, geopotential and alpha advance_uv reads stay
! the record's.  Built against WRFPLUS's libwrflib.a (configure wrfplus,
! real*8, -O0 -ffp-contract=off) by tools/wrfplus_oracle/build.sh; driven
! by tools/wrfplus_oracle/fixture.py emdiv_chain.
!
! Input (stream, native binary64/int32): advance_uv.F90's record, then
! advance_mu_t.F90's (its u, v, mu and mudf unread: the chain's are
! advance_uv's), then nsub.  See fixture.py _chain_record.
! Output (stream): TL u, ud, v, vd; mu, mud, mudf, mudfd; t, td; then the
! AD a_u, a_v, a_t, a_ru, a_rv, a_ft; a_mu, a_mudf, a_muu, a_muv (advance_uv's
! faces), a_mut, a_muu, a_muv (advance_mu_t's faces), a_mu_tend.
program wrfplus_emdiv_chain
  use module_configure, only : grid_config_rec_type
  use g_module_small_step_em, only : g_advance_uv, g_advance_mu_t
  use a_module_small_step_em, only : a_advance_uv, a_advance_mu_t
  use g_module_bc, only : g_spec_bdyupdate
  use a_module_bc, only : a_spec_bdyupdate
  implicit none
  integer :: nx, ny, nz, spec_zone, itop, iu, nsub, s, mnx, mny, mnz, mzone
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
  real :: rdx, rdy, dts, cf1, cf2, cf3, emdiv, mrdx, mrdy, mdts, epssm
  type(grid_config_rec_type) :: cf
  logical :: top_lid
  ! advance_uv's record
  real, allocatable, dimension(:,:,:) :: u, ud, ru, rud, v, vd, rv, rvd, p, pd, pb, &
       ph, phd, php, phpd, alt, altd, al, ald, cqu, cqud, cqv, cqvd, au, av
  real, allocatable, dimension(:,:) :: mu, mud, muu, muud, muv, muvd, mudf, mudfd, &
       msfux, msfuy, msfvx, msfvx_inv, msfvy
  real, allocatable, dimension(:) :: fnm, fnp, rdnw
  ! advance_mu_t's record (m_: its copies of names advance_uv also has)
  real, allocatable, dimension(:,:,:) :: ww, wwd, ww1, ww1d, m_u, m_ud, u1, u1d, m_v, &
       m_vd, v1, v1d, t, td, t1, t1d, ft, ftd, sww, st, stave
  real, allocatable, dimension(:,:) :: m_mu, m_mud, mut, mutd, m_muu, m_muud, m_muv, &
       m_muvd, muts, mutsd, mu_tend, mu_tendd, msft, msfu, msfv, smu, smuts, smudf
  real, allocatable, dimension(:) :: m_fnm, m_fnp, dnw, m_rdnw
  ! chain work and the bases each call's adjoint reruns from
  real, allocatable, dimension(:,:,:) :: tave, taved, uam, vam, wwam, uamd, vamd, wwamd
  real, allocatable, dimension(:,:) :: one, muave, muaved
  real, allocatable, dimension(:,:,:,:) :: b_u, b_v, b_mu_u, b_mu_v, b_ww, b_t, b_tave
  real, allocatable, dimension(:,:,:) :: b_mu, b_mudf, b_mmu, b_mmudf, b_muts, b_muave
  ! adjoints
  real, allocatable, dimension(:,:,:) :: a_ru, a_rv, a_p, a_ph, a_php, a_alt, a_al, &
       a_cqu, a_cqv, a_ww1, a_u1, a_v1, a_t1, a_ft, a_uam, a_vam, a_wwam
  real, allocatable, dimension(:,:) :: a_mu, a_muu, a_muv, a_mudf, a_mut, a_muave, &
       a_mmuu, a_mmuv, a_mu_tend
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
  read(iu) mnx, mny, mnz, mzone
  read(iu) mrdx, mrdy, mdts, epssm
  if (mnx /= nx .or. mny /= ny .or. mnz /= nz .or. mzone /= spec_zone .or. &
      mrdx /= rdx .or. mrdy /= rdy .or. mdts /= dts) stop 'records disagree'
  call a3(ww); call a3(wwd); call a3(ww1); call a3(ww1d); call a3(m_u); call a3(m_ud)
  call a3(u1); call a3(u1d); call a3(m_v); call a3(m_vd); call a3(v1); call a3(v1d)
  call a3(t); call a3(td); call a3(t1); call a3(t1d); call a3(ft); call a3(ftd)
  call a3(sww); call a3(st); call a3(stave)
  call a2(m_mu); call a2(m_mud); call a2(mut); call a2(mutd); call a2(m_muu)
  call a2(m_muud); call a2(m_muv); call a2(m_muvd); call a2(muts); call a2(mutsd)
  call a2(mu_tend); call a2(mu_tendd); call a2(msft); call a2(msfu); call a2(msfv)
  call a2(smu); call a2(smuts); call a2(smudf)
  allocate(m_fnm(kms:kme), m_fnp(kms:kme), dnw(kms:kme), m_rdnw(kms:kme))
  read(iu) m_fnm, m_fnp, dnw, m_rdnw
  read(iu) nsub
  close(iu)

  cf%specified = .true.;   cf%nested = .false.;   cf%periodic_x = .false.
  cf%open_xs = .false.;    cf%open_xe = .false.;  cf%open_ys = .false.
  cf%open_ye = .false.;    cf%symmetric_xs = .false.; cf%symmetric_xe = .false.
  cf%symmetric_ys = .false.; cf%symmetric_ye = .false.; cf%polar = .false.

  allocate(one, muave, muaved, mold=mu); one = 1.; muave = 0; muaved = 0
  allocate(tave, taved, uam, vam, wwam, uamd, vamd, wwamd, mold=ww)
  tave = 0; taved = 0; uam = 0; vam = 0; wwam = 0; uamd = 0; vamd = 0; wwamd = 0
  allocate(b_u(ims:ime, kms:kme, jms:jme, nsub), b_v, b_mu_u, b_mu_v, b_ww, b_t, &
           b_tave, mold=b_u)
  allocate(b_mu(ims:ime, jms:jme, nsub), b_mudf, b_mmu, b_mmudf, b_muts, b_muave, &
           mold=b_mu)

  do s = 1, nsub
    b_u(:,:,:,s) = u; b_v(:,:,:,s) = v; b_mu(:,:,s) = mu; b_mudf(:,:,s) = mudf
    call g_advance_uv(u, ud, ru, rud, v, vd, rv, rvd, p, pd, pb, ph, phd, php, phpd, &
         alt, altd, al, ald, mu, mud, muu, muud, cqu, cqud, muv, muvd, cqv, cqvd, &
         mudf, mudfd, msfux, msfuy, msfvx, msfvx_inv, msfvy, rdx, rdy, dts, cf1, cf2, &
         cf3, fnm, fnp, emdiv, rdnw, cf, spec_zone, .true., top_lid, ids, ide, jds, &
         jde, kds, kde, ims, ime, jms, jme, kms, kme, ids, ide, jds, jde, kds, kde)
    b_mu_u(:,:,:,s) = u; b_mu_v(:,:,:,s) = v; b_ww(:,:,:,s) = ww; b_t(:,:,:,s) = t
    b_tave(:,:,:,s) = tave; b_mmu(:,:,s) = mu; b_mmudf(:,:,s) = mudf
    b_muts(:,:,s) = muts; b_muave(:,:,s) = muave
    call g_advance_mu_t(ww, wwd, ww1, ww1d, u, ud, u1, u1d, v, vd, v1, v1d, mu, mud, &
         mut, mutd, muave, muaved, muts, mutsd, m_muu, m_muud, m_muv, m_muvd, mudf, &
         mudfd, uam, uamd, vam, vamd, wwam, wwamd, t, td, t1, t1d, tave, taved, ft, &
         ftd, mu_tend, mu_tendd, rdx, rdy, dts, epssm, dnw, m_fnm, m_fnp, m_rdnw, one, &
         msfu, one, msfv, one, msft, msft, 1, cf, ids, ide, jds, jde, kds, kde, ims, &
         ime, jms, jme, kms, kme, ids, ide, jds, jde, kds, kde)
    call g_spec_bdyupdate(t, td, ft, ftd, dts, 't', cf, spec_zone, ids, ide, jds, &
         jde, kds, kde, ims, ime, jms, jme, kms, kme, ids, ide, jds, jde, kds, kde, &
         ids, ide, jds, jde, kds, kde)
    call g_spec_bdyupdate(mu, mud, mu_tend, mu_tendd, dts, 'm', cf, spec_zone, ids, &
         ide, jds, jde, 1, 1, ims, ime, jms, jme, 1, 1, ids, ide, jds, jde, 1, 1, ids, &
         ide, jds, jde, 1, 1)
    call g_spec_bdyupdate(muts, mutsd, mu_tend, mu_tendd, dts, 'm', cf, spec_zone, &
         ids, ide, jds, jde, 1, 1, ims, ime, jms, jme, 1, 1, ids, ide, jds, jde, 1, 1, &
         ids, ide, jds, jde, 1, 1)
  end do

  open(newunit=iu, file=trim(fout), access='stream', form='unformatted', status='replace')
  write(iu) u, ud, v, vd
  write(iu) mu, mud, mudf, mudfd
  write(iu) t, td

  ! Adjoint: au, av, smu (a_mu), smudf (a_mudf) and st (a_t) are the
  ! output adjoints; sww, stave and smuts come in as the fixture's zeros.
  allocate(a_ru, a_rv, a_p, a_ph, a_php, a_alt, a_al, a_cqu, a_cqv, a_ww1, a_u1, &
           a_v1, a_t1, a_ft, a_uam, a_vam, a_wwam, mold=u)
  a_ru = 0; a_rv = 0; a_p = 0; a_ph = 0; a_php = 0; a_alt = 0; a_al = 0; a_cqu = 0
  a_cqv = 0; a_ww1 = 0; a_u1 = 0; a_v1 = 0; a_t1 = 0; a_ft = 0; a_uam = 0; a_vam = 0
  a_wwam = 0
  allocate(a_mu, a_muu, a_muv, a_mudf, a_mut, a_muave, a_mmuu, a_mmuv, a_mu_tend, &
           mold=mu)
  a_mu = smu; a_mudf = smudf; a_muu = 0; a_muv = 0; a_mut = 0; a_muave = 0
  a_mmuu = 0; a_mmuv = 0; a_mu_tend = 0
  do s = nsub, 1, -1
    call a_spec_bdyupdate(smuts, a_mu_tend, dts, 'm', cf, spec_zone, ids, ide, jds, &
         jde, 1, 1, ims, ime, jms, jme, 1, 1, ids, ide, jds, jde, 1, 1, ids, ide, jds, &
         jde, 1, 1)
    call a_spec_bdyupdate(a_mu, a_mu_tend, dts, 'm', cf, spec_zone, ids, ide, jds, &
         jde, 1, 1, ims, ime, jms, jme, 1, 1, ids, ide, jds, jde, 1, 1, ids, ide, jds, &
         jde, 1, 1)
    call a_spec_bdyupdate(st, a_ft, dts, 't', cf, spec_zone, ids, ide, jds, jde, kds, &
         kde, ims, ime, jms, jme, kms, kme, ids, ide, jds, jde, kds, kde, ids, ide, &
         jds, jde, kds, kde)
    call a_advance_mu_t(b_ww(:,:,:,s), sww, ww1, a_ww1, b_mu_u(:,:,:,s), au, u1, a_u1, &
         b_mu_v(:,:,:,s), av, v1, a_v1, b_mmu(:,:,s), a_mu, mut, a_mut, &
         b_muave(:,:,s), a_muave, b_muts(:,:,s), smuts, m_muu, a_mmuu, m_muv, a_mmuv, &
         b_mmudf(:,:,s), a_mudf, uam, a_uam, vam, a_vam, wwam, a_wwam, b_t(:,:,:,s), &
         st, t1, a_t1, b_tave(:,:,:,s), stave, ft, a_ft, mu_tend, a_mu_tend, rdx, rdy, &
         dts, epssm, dnw, m_fnm, m_fnp, m_rdnw, one, msfu, one, msfv, one, msft, msft, &
         1, cf, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, ids, ide, &
         jds, jde, kds, kde)
    call a_advance_uv(b_u(:,:,:,s), au, ru, a_ru, b_v(:,:,:,s), av, rv, a_rv, p, a_p, &
         pb, ph, a_ph, php, a_php, alt, a_alt, al, a_al, b_mu(:,:,s), a_mu, muu, &
         a_muu, cqu, a_cqu, muv, a_muv, cqv, a_cqv, b_mudf(:,:,s), a_mudf, msfux, &
         msfuy, msfvx, msfvx_inv, msfvy, rdx, rdy, dts, cf1, cf2, cf3, fnm, fnp, emdiv, &
         rdnw, cf, spec_zone, .true., top_lid, ids, ide, jds, jde, kds, kde, ims, ime, &
         jms, jme, kms, kme, ids, ide, jds, jde, kds, kde)
  end do

  write(iu) au, av, st, a_ru, a_rv, a_ft
  write(iu) a_mu, a_mudf, a_muu, a_muv, a_mut, a_mmuu, a_mmuv, a_mu_tend
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

end program wrfplus_emdiv_chain
