! WRFPLUS 4.7.1 oracle driver: g_rhs_ph and a_rhs_ph on one input record
! (docs/dev/P3A-WRF-DRYCORE-ADJOINT.md section 5, item 5).  Built against
! WRFPLUS's libwrflib.a (configure wrfplus, real*8, -O0 -ffp-contract=off)
! by tools/wrfplus_oracle/build.sh; driven by tools/wrfplus_oracle/fixture.py.
!
! Input (stream, native binary64/int32): nx, ny, nz, flags (bit 0
! periodic_x, 1 periodic_y, 2 specified, 4-7 open_xs/xe/ys/ye, 8 TL only),
! h_sca_adv_order; rdx, rdy, cfn, cfn1; then on (-3:nx+4, 1:nz+1, -3:ny+4)
! ph_tend, its adjoint seed, u, ud, v, vd, ww, wwd, ph, phd, phb, w, wd;
! on (-3:nx+4, -3:ny+4) mut, mutd, muu, muud, muv, muvd, msfu, msfv, msft;
! on 1:nz+1 fnm, fnp, rdnw.  ph_old is ph (and its tangent ph's), as woof
! advects one geopotential; msfux = msfuy = msfu, msfvx = msfvy = msfv,
! msftx = msfty = msft; non-hydrostatic.
! Output (stream): TL ph_tend, its tangent; AD ph_tend (after the sweep),
! u, v, ww, ph, ph_old, w adjoints; mut, muu, muv adjoints (zero with TL
! only).
!
! TL only is for h_sca_adv_order 3-4 on a periodic y axis or a specified
! x axis: a_rhs_ph stores its order-3/4 trajectory in arrays declared
! from jts+2 (Tmpv4012-4015) and its+2 (Tmpv4016-4019), which assumes the
! open/specified start j_start = jts+2, i_start = its+2.  A periodic y
! axis starts at jts and WRF's order-3/4 x loop moves its start only for
! open_xs, so the stores run below those bounds (gfortran -fcheck=bounds:
! "Index '1' of dimension 3 of array 'tmpv4012' below lower bound of 3")
! and the heap is corrupted (SIGSEGV in free).
program wrfplus_rhsph
  use module_configure, only : grid_config_rec_type
  use g_module_big_step_utilities_em, only : g_rhs_ph
  use a_module_big_step_utilities_em, only : a_rhs_ph
  implicit none
  integer :: nx, ny, nz, flags, iu, hdr(5)
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
  real :: rdx, rdy, cfn, cfn1, sc(4)
  type(grid_config_rec_type) :: cf
  real, allocatable, dimension(:,:,:) :: pht, pha, u, ud, v, vd, ww, wwd, ph, phd, phb, w, wd, &
       pht2, phtd, ua, va, wwa, pha3, phoa, wa
  real, allocatable, dimension(:,:) :: mut, mutd, muu, muud, muv, muvd, msfu, msfv, msft, &
       msfvi, muta, muua, muva
  real, allocatable, dimension(:) :: fnm, fnp, rdnw
  character(len=512) :: fin, fout

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(newunit=iu, file=trim(fin), access='stream', form='unformatted', status='old')
  read(iu) hdr
  nx = hdr(1); ny = hdr(2); nz = hdr(3); flags = hdr(4)
  read(iu) sc
  rdx = sc(1); rdy = sc(2); cfn = sc(3); cfn1 = sc(4)
  ids = 1; ide = nx + 1; jds = 1; jde = ny + 1; kds = 1; kde = nz + 1
  ims = -3; ime = nx + 4; jms = -3; jme = ny + 4; kms = 1; kme = nz + 1
  call a3(pht); call a3(pha); call a3(u); call a3(ud); call a3(v); call a3(vd)
  call a3(ww); call a3(wwd); call a3(ph); call a3(phd); call a3(phb); call a3(w); call a3(wd)
  call a2(mut); call a2(mutd); call a2(muu); call a2(muud); call a2(muv); call a2(muvd)
  call a2(msfu); call a2(msfv); call a2(msft)
  allocate(fnm(kms:kme), fnp(kms:kme), rdnw(kms:kme))
  read(iu) fnm, fnp, rdnw
  close(iu)

  cf%periodic_x = btest(flags, 0); cf%periodic_y = btest(flags, 1)
  cf%specified = btest(flags, 2);  cf%nested = .false.
  cf%open_xs = btest(flags, 4);    cf%open_xe = btest(flags, 5)
  cf%open_ys = btest(flags, 6);    cf%open_ye = btest(flags, 7)
  cf%symmetric_xs = .false.; cf%symmetric_xe = .false.
  cf%symmetric_ys = .false.; cf%symmetric_ye = .false.; cf%polar = .false.
  cf%h_sca_adv_order = hdr(5)

  allocate(pht2, phtd, ua, va, wwa, pha3, phoa, wa, mold=ph)
  allocate(msfvi, muta, muua, muva, mold=mut)
  msfvi = 1.0 / msfv
  pht2 = pht; phtd = 0
  ua = 0; va = 0; wwa = 0; pha3 = 0; phoa = 0; wa = 0; muta = 0; muua = 0; muva = 0
  call g_rhs_ph(pht2, phtd, u, ud, v, vd, ww, wwd, ph, phd, ph, phd, phb, w, wd, mut, mutd, &
       muu, muud, muv, muvd, fnm, fnp, rdnw, cfn, cfn1, rdx, rdy, msfu, msfu, msfv, msfvi, &
       msfv, msft, msft, .true., cf, &
       ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
       ids, ide, jds, jde, kds, kde)
  if (btest(flags, 8)) then
    pha = 0
  else
  call a_rhs_ph(pht, pha, u, ua, v, va, ww, wwa, ph, pha3, ph, phoa, phb, w, wa, mut, muta, &
       muu, muua, muv, muva, fnm, fnp, rdnw, cfn, cfn1, rdx, rdy, msfu, msfu, msfv, msfvi, &
       msfv, msft, msft, .true., cf, &
       ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
       ids, ide, jds, jde, kds, kde)
  end if

  open(newunit=iu, file=trim(fout), access='stream', form='unformatted', status='replace')
  write(iu) pht2, phtd, pha, ua, va, wwa, pha3, phoa, wa, muta, muua, muva
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

end program wrfplus_rhsph
