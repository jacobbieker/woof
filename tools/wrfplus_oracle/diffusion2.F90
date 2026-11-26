! WRF 4.7.1 oracle driver: the diff_opt 2 horizontal-diffusion rows
! (horizontal_diffusion_2: horizontal_diffusion_u_2, _v_2, _w_2 and
! horizontal_diffusion_s) with their metrics (compute_diff_metrics), the
! density rho = 1/alt (1 + qv) and, for the w row, defor13 and defor23
! from cal_deform_and_div (docs/dev/P3A-WRF-DRYCORE-ADJOINT.md, item 2b);
! and vertical_diffusion_2 itself (its u, v, w, theta, moisture and
! scalar rows with the isfflx surface fluxes), with defor13/23/33 from
! cal_deform_and_div (item 2b-2).
!
! WRFPLUS 4.7.1's own G_/A_ diff_opt 2 routines differentiate the older
! mu-coupled formulation (a finding), so this driver links the WRF
! 4.7.1 nonlinear routines from the same libwrflib.a (real*8, -O0
! -ffp-contract=off) and tools/wrfplus_oracle/fixture.py takes the
! TL/AD from fourth-order central differences of them: it sends a batch
! of records, each adding one step to a set of memory words (a woof word
! and its halo images), and gets one tendency per record.
!
! The map factors are msfux = msfuy = msfu, msfvx = msfvy = msfv and
! msftx = msfty = msft.  Halos of the derived fields (rdz, rdzw, zx, zy,
! defor13, defor23) are filled as WRF's periodic and specified boundary
! conditions leave them for the rows: periodic images, else copies of the
! nearest domain word.  compute_diff_metrics leaves zx/zy on the end
! faces of the top w level unset (its face loops stop at kde-2); the
! driver sets them by its own interior formula (periodic) or to zero
! (specified) before the fill (a finding; woof computes them).
!
! Input (stream, native binary64/int32): nx, ny, nz, flags (bit 0
! periodic_x, 1 periodic_y, 2 specified, 3 moist; bits 4-6 the row:
! 1 u, 2 v, 3 w, 5 scalar, 6 vertical_diffusion_2), nrec; rdx, rdy, cf1,
! cf2, cf3; on (kms:kme)
! fnm, fnp, dn, dnw; on (ims:ime, jms:jme) msfu, msfv, msft; on
! (ims:ime, kms:kme, jms:jme) phb, then the active fields php, alt, qv,
! xk, defor11, defor22, defor12, u, v, w, field (ids 1-11) and the
! tendency.  Then nrec records: n, the step h, n field ids, n 1-based
! linear memory words.  Output (stream): nrec tendencies on the memory.
!
! Row 6 reads, after the tendency: vrow (1 ru_tendf, 2 rv_tendf,
! 3 rw_tendf, 4 rt_tendf, 5 moist_tendf(P_QV), 6 scalar_tendf), isfflx,
! mix_full_fields (0/1); cd0 and tke_heat_flux; ust, hfx, qfx on
! (ims:ime, jms:jme) and thb on the 3-D memory.  The field (id 11) is
! thp (plus thb when mix_full_fields, t_base = 0), moist(P_QV) for vrow 5
! or scalar 2; moist(P_QV) is qv otherwise; xkmv = xkhv = xkmh = xk.
program wrfplus_diffusion2
  use module_configure, only : grid_config_rec_type
  use module_diffusion_em, only : compute_diff_metrics, cal_deform_and_div, &
       horizontal_diffusion_u_2, horizontal_diffusion_v_2, horizontal_diffusion_w_2, &
       horizontal_diffusion_s, vertical_diffusion_2
  use module_state_description, only : P_qv, PARAM_FIRST_SCALAR
  implicit none
  integer, parameter :: nact = 11
  integer :: nx, ny, nz, iu, ou, hdr(5), flags, row, nrec, r, n, q
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
  logical :: px, py, moist
  real :: sc(5), rdx, rdy, cf1, cf2, cf3, h
  type(grid_config_rec_type) :: cf
  real, allocatable, dimension(:) :: fnm, fnp, dn, dnw, ubase, vbase
  real, allocatable, dimension(:,:) :: msfu, msfv, msft
  real, allocatable, dimension(:,:,:) :: phb, tend0, tend, rho, z, rdz, rdzw, zx, zy, &
       div, e11, e22, e33, e12, e13, e23, tke
  real, allocatable :: act(:,:,:,:), nba(:,:,:,:)
  integer :: vhdr(3), vrow
  real :: vsc(2)
  real, allocatable, dimension(:,:) :: ust, hfx, qfx
  real, allocatable, dimension(:,:,:) :: thb, ru, rv, rw, rt, tket, thp, theta
  real, allocatable, dimension(:,:,:,:) :: mt, sct, moist, scal, chem, chemt, trac, tract
  integer, allocatable :: fid(:), word(:)
  character(len=512) :: fin, fout

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(newunit=iu, file=trim(fin), access='stream', form='unformatted', status='old')
  read(iu) hdr
  nx = hdr(1); ny = hdr(2); nz = hdr(3); flags = hdr(4); nrec = hdr(5)
  row = iand(ishft(flags, -4), 7)
  px = btest(flags, 0); py = btest(flags, 1); moist = btest(flags, 3)
  read(iu) sc
  rdx = sc(1); rdy = sc(2); cf1 = sc(3); cf2 = sc(4); cf3 = sc(5)
  ids = 1; ide = nx + 1; jds = 1; jde = ny + 1; kds = 1; kde = nz + 1
  ims = -3; ime = nx + 4; jms = -3; jme = ny + 4; kms = 1; kme = nz + 1

  cf%periodic_x = px; cf%periodic_y = py
  cf%specified = btest(flags, 2);  cf%nested = .false.
  cf%open_xs = .false.; cf%open_xe = .false.; cf%open_ys = .false.; cf%open_ye = .false.
  cf%symmetric_xs = .false.; cf%symmetric_xe = .false.
  cf%symmetric_ys = .false.; cf%symmetric_ye = .false.; cf%polar = .false.
  cf%diff_opt = 2; cf%km_opt = 4; cf%m_opt = 0; cf%sfs_opt = 0

  allocate(fnm(kms:kme), fnp(kms:kme), dn(kms:kme), dnw(kms:kme))
  read(iu) fnm, fnp, dn, dnw
  allocate(ubase(kms:kme), vbase(kms:kme))
  ubase = 0; vbase = 0
  allocate(msfu(ims:ime, jms:jme), msfv(ims:ime, jms:jme), msft(ims:ime, jms:jme))
  read(iu) msfu, msfv, msft
  allocate(phb(ims:ime, kms:kme, jms:jme))
  read(iu) phb
  allocate(act(ims:ime, kms:kme, jms:jme, nact))
  read(iu) act
  allocate(tend0, tend, rho, z, rdz, rdzw, zx, zy, div, e11, e22, e33, e12, e13, e23, tke, &
           mold=phb)
  allocate(nba(ims:ime, kms:kme, jms:jme, 1))
  read(iu) tend0
  if (row == 6) then
    P_qv = PARAM_FIRST_SCALAR
    read(iu) vhdr, vsc
    vrow = vhdr(1)
    cf%isfflx = vhdr(2); cf%mix_full_fields = vhdr(3) == 1
    cf%tke_drag_coefficient = vsc(1); cf%tke_heat_flux = vsc(2); cf%use_theta_m = 0
    allocate(ust(ims:ime, jms:jme), hfx(ims:ime, jms:jme), qfx(ims:ime, jms:jme))
    read(iu) ust, hfx, qfx
    allocate(thb, ru, rv, rw, rt, tket, thp, theta, mold=phb)
    read(iu) thb
    allocate(mt(ims:ime, kms:kme, jms:jme, 2), sct(ims:ime, kms:kme, jms:jme, 2))
    allocate(moist(ims:ime, kms:kme, jms:jme, 2), scal(ims:ime, kms:kme, jms:jme, 2))
    allocate(chem(ims:ime, kms:kme, jms:jme, 1), chemt(ims:ime, kms:kme, jms:jme, 1))
    allocate(trac(ims:ime, kms:kme, jms:jme, 1), tract(ims:ime, kms:kme, jms:jme, 1))
  end if

  open(newunit=ou, file=trim(fout), access='stream', form='unformatted', status='replace')
  do r = 1, nrec
    read(iu) n, h
    allocate(fid(n), word(n))
    read(iu) fid, word
    do q = 1, n
      call bump(act(:,:,:,fid(q)), word(q), h)
    end do
    call evaluate()
    write(ou) tend
    do q = 1, n
      call bump(act(:,:,:,fid(q)), word(q), -h)
    end do
    deallocate(fid, word)
  end do
  close(iu)
  close(ou)

contains

  subroutine bump(x, at, step)
    real, intent(inout) :: x(*)
    integer, intent(in) :: at
    real, intent(in) :: step
    x(at) = x(at) + step
  end subroutine bump

  subroutine evaluate()
    integer :: i, k, j
    rho = 0; z = 0; rdz = 0; rdzw = 0; zx = 0; zy = 0
    div = 0; e11 = 0; e22 = 0; e33 = 0; e12 = 0; e13 = 0; e23 = 0; tke = 0; nba = 0
    do j = jms, jme
      do k = kms, kme
        do i = ims, ime
          if (moist) then
            rho(i,k,j) = 1./act(i,k,j,2)*(1.+act(i,k,j,3))
          else
            rho(i,k,j) = 1./act(i,k,j,2)
          end if
        end do
      end do
    end do
    call compute_diff_metrics(cf, act(:,:,:,1), phb, z, rdz, rdzw, zx, zy, rdx, rdy, &
         ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
         ids, ide, jds, jde, kds, kde)
    call top_faces()
    call fill(rdz, 0, 0); call fill(rdzw, 0, 0)
    call fill(zx, 1, 0); call fill(zy, 0, 1)
    tend = tend0
    select case (row)
    case (1)
      call horizontal_diffusion_u_2(tend, cf, act(:,:,:,5), act(:,:,:,7), div, nba, 1, tke, &
           msfu, msfu, act(:,:,:,4), rdx, rdy, fnm, fnp, dnw, zx, zy, rdzw, rho, &
           ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
           ids, ide, jds, jde, kds, kde)
    case (2)
      call horizontal_diffusion_v_2(tend, cf, act(:,:,:,7), act(:,:,:,6), div, nba, 1, tke, &
           msfv, msfv, act(:,:,:,4), rdx, rdy, fnm, fnp, dnw, zx, zy, rdzw, rho, &
           ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
           ids, ide, jds, jde, kds, kde)
    case (3)
      call cal_deform_and_div(cf, act(:,:,:,8), act(:,:,:,9), act(:,:,:,10), div, &
           e11, e22, e33, e12, e13, e23, nba, 1, ubase, vbase, &
           msfu, msfu, msfv, msfv, msft, msft, rdx, rdy, dn, dnw, rdz, rdzw, &
           fnm, fnp, cf1, cf2, cf3, zx, zy, &
           ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
           ids, ide, jds, jde, kds, kde)
      call fill(e13, 1, 0); call fill(e23, 0, 1)
      call horizontal_diffusion_w_2(tend, cf, e13, e23, div, nba, 1, tke, &
           msft, msft, act(:,:,:,4), rdx, rdy, fnm, fnp, dn, zx, zy, rdz, rho, &
           ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
           ids, ide, jds, jde, kds, kde)
    case (6)
      call cal_deform_and_div(cf, act(:,:,:,8), act(:,:,:,9), act(:,:,:,10), div, &
           e11, e22, e33, e12, e13, e23, nba, 1, ubase, vbase, &
           msfu, msfu, msfv, msfv, msft, msft, rdx, rdy, dn, dnw, rdz, rdzw, &
           fnm, fnp, cf1, cf2, cf3, zx, zy, &
           ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
           ids, ide, jds, jde, kds, kde)
      call fill(e13, 1, 0); call fill(e23, 0, 1)
      call vertical()
    case default
      call horizontal_diffusion_s(tend, cf, act(:,:,:,11), msft, msft, msfu, msfu, &
           msfv, msfv, act(:,:,:,4), rdx, rdy, fnm, fnp, cf1, cf2, cf3, &
           zx, zy, rdz, rdzw, dnw, dn, rho, .false., &
           ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
           ids, ide, jds, jde, kds, kde)
    end select
  end subroutine evaluate

  subroutine vertical()
    real :: zb(kms:kme)
    ru = 0; rv = 0; rw = 0; rt = 0; tket = 0; mt = 0; sct = 0
    chem = 0; chemt = 0; trac = 0; tract = 0; zb = 0
    moist = 0; scal = 0
    moist(:,:,:,P_qv) = act(:,:,:,3)
    if (vrow == 5) moist(:,:,:,P_qv) = act(:,:,:,11)
    if (vrow == 6) scal(:,:,:,2) = act(:,:,:,11)
    thp = act(:,:,:,11)
    if (cf%mix_full_fields) thp = act(:,:,:,11) + thb
    theta = 0
    select case (vrow)
    case (1); ru = tend0
    case (2); rv = tend0
    case (3); rw = tend0
    case (4); rt = tend0
    case (5); mt(:,:,:,P_qv) = tend0
    case default; sct(:,:,:,2) = tend0
    end select
    call vertical_diffusion_2(ru, rv, rw, rt, tket, mt, 2, chemt, 1, sct, 2, tract, 1, &
         act(:,:,:,8), act(:,:,:,9), thp, zb, zb, zb, zb, tket, theta, &
         cf, e13, e23, e33, nba, 1, div, moist, chem, scal, trac, &
         act(:,:,:,4), act(:,:,:,4), act(:,:,:,4), 4, &
         fnm, fnp, dn, dnw, rdz, rdzw, hfx, qfx, ust, rho, &
         ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
         ids, ide, jds, jde, kds, kde)
    select case (vrow)
    case (1); tend = ru
    case (2); tend = rv
    case (3); tend = rw
    case (4); tend = rt
    case (5); tend = mt(:,:,:,P_qv)
    case default; tend = sct(:,:,:,2)
    end select
  end subroutine vertical

  ! zx/zy on the end faces of the top w level (compute_diff_metrics's
  ! face loops stop at kde-2).
  subroutine top_faces()
    integer :: i, j, k
    k = kde
    do j = jds, jde - 1
      do i = ids, ide, ide - ids
        if (px) then
          zx(i,k,j) = rdx * (phb(i,k,j) - phb(i-1,k,j)) / 9.81
          zx(i,k,j) = zx(i,k,j) + rdx * (act(i,k,j,1) - act(i-1,k,j,1)) / 9.81
        else
          zx(i,k,j) = 0.
        end if
      end do
    end do
    do j = jds, jde, jde - jds
      do i = ids, ide - 1
        if (py) then
          zy(i,k,j) = rdy * (phb(i,k,j) - phb(i,k,j-1)) / 9.81
          zy(i,k,j) = zy(i,k,j) + rdy * (act(i,k,j,1) - act(i,k,j-1,1)) / 9.81
        else
          zy(i,k,j) = 0.
        end if
      end do
    end do
  end subroutine top_faces

  ! Halo fill: periodic images on a periodic axis, else the nearest
  ! domain word; sx/sy = 1 for a field on faces.
  subroutine fill(x, sx, sy)
    real, intent(inout) :: x(ims:ime, kms:kme, jms:jme)
    integer, intent(in) :: sx, sy
    integer :: i, j, si, sj
    do j = jms, jme
      do i = ims, ime
        si = image(i, ids, nx, sx, px)
        sj = image(j, jds, ny, sy, py)
        if (si /= i .or. sj /= j) x(i,:,j) = x(si,:,sj)
      end do
    end do
  end subroutine fill

  integer function image(i, lo, n, stag, per)
    integer, intent(in) :: i, lo, n, stag
    logical, intent(in) :: per
    if (per) then
      image = lo + modulo(i - lo, n)
      if (stag == 1 .and. i == lo + n) image = i
    else
      image = max(lo, min(i, lo + n - 1 + stag))
    end if
  end function image

end program wrfplus_diffusion2
