! WOOF-OPTIN: wrfchem_pnnl
! Abdul-Razzak and Ghan activation (phys/module_mixactivate.F, WRF v4.7.1):
! `activate` on a grid of cases that reaches its branches -- modal and
! quasi-sectional (msectional > 0), the single-updraft path (sigw <= 1e-5)
! with rising and sinking air, the Gaussian updraft spectrum for cloud base
! (wbar > 0) and cloud top (wbar < 0), smax prescribed (> 0 and <= 0) or
! computed by maxsat, empty and non-hygroscopic modes, and the single tiny
! mode's early return.  Inputs are built from literals with exact arithmetic
! only (no libm), so the fixture's inputs are the same words everywhere.
!
! Every case is stacked along a last (case) dimension in one directory, cases/
! (oracle_io's format): the inputs, then fn, fs, fm, fluxn, fluxs, fluxm,
! flux_fullact as activate left them (each output starts at -7 where activate
! does not write).
program run_activate
  use oracle_io
  use activate_extract, only: activate
  implicit none
  integer, parameter :: maxd_atype = 2, maxd_asize = 8
  real :: na(maxd_asize, maxd_atype), volc(maxd_asize, maxd_atype)
  real :: dlo(maxd_asize, maxd_atype), dhi(maxd_asize, maxd_atype)
  real :: sigman(maxd_asize, maxd_atype), hygro(maxd_asize, maxd_atype)
  real :: fn(maxd_asize, maxd_atype), fs(maxd_asize, maxd_atype), fm(maxd_asize, maxd_atype)
  real :: fluxn(maxd_asize, maxd_atype), fluxs(maxd_asize, maxd_atype), fluxm(maxd_asize, maxd_atype)
  real :: flux_fullact, wbar, sigw, wdiab, wminf, wmaxf, tair, rhoair, smax
  integer :: nsize(maxd_atype), ntype, msectional, ismax, icase, iw, it, ia, m, n
  integer, parameter :: ncmax = 128, nscal = 9
  ! per case: the (maxd_asize, maxd_atype) arrays in the order of anames, the scalars of snames, the integers
  real :: arrays(maxd_asize, maxd_atype, 12, ncmax), scal(nscal, ncmax)
  integer :: ints(7, ncmax)
  character(len=1024) :: root
  ! updraft (wbar, sigw) pairs: single updraft up/down, spectrum at base/top
  real, parameter :: wb(6) = (/ 0.5, -0.3, 0.2, -0.4, 1.5, 0.0 /)
  real, parameter :: sw(6) = (/ 0.0, 0.0, 0.6, 0.8, 0.25, 1.0 /)
  real, parameter :: ta(2) = (/ 285.0, 268.5 /)
  real, parameter :: ra(2) = (/ 1.18, 0.82 /)

  call oracle_root(root)
  icase = 0
  do ia = 1, 7           ! aerosol setups
    do iw = 1, 6         ! updrafts
      do it = 1, 2       ! thermodynamic states
        do ismax = 0, 2  ! 0: absent, 1: prescribed > 0, 2: prescribed <= 0
          if (ismax > 0 .and. (iw > 2 .or. it > 1)) cycle
          icase = icase + 1
          call setup(ia)
          wbar = wb(iw); sigw = sw(iw); wdiab = 0.0
          if (iw == 6) wdiab = 0.05
          wminf = wbar - 4.0 * sigw
          wmaxf = wbar + 4.0 * sigw
          if (iw <= 2) then
            wminf = wbar; wmaxf = wbar
          end if
          tair = ta(it); rhoair = ra(it)
          fn = -7.0; fs = -7.0; fm = -7.0
          fluxn = -7.0; fluxs = -7.0; fluxm = -7.0; flux_fullact = -7.0
          smax = 0.0
          if (ismax == 1) smax = 0.003
          if (ismax == 2) smax = 0.0
          if (ismax == 0) then
            call activate(wbar, sigw, wdiab, wminf, wmaxf, tair, rhoair, msectional, &
                 maxd_atype, ntype, maxd_asize, nsize, na, volc, dlo, dhi, sigman, hygro, &
                 fn, fs, fm, fluxn, fluxs, fluxm, flux_fullact, 1, 7, 2, 3, 4)
          else
            call activate(wbar, sigw, wdiab, wminf, wmaxf, tair, rhoair, msectional, &
                 maxd_atype, ntype, maxd_asize, nsize, na, volc, dlo, dhi, sigman, hygro, &
                 fn, fs, fm, fluxn, fluxs, fluxm, flux_fullact, 1, 7, 2, 3, 4, smax)
          end if
          arrays(:, :, 1, icase) = na; arrays(:, :, 2, icase) = volc
          arrays(:, :, 3, icase) = dlo; arrays(:, :, 4, icase) = dhi
          arrays(:, :, 5, icase) = sigman; arrays(:, :, 6, icase) = hygro
          arrays(:, :, 7, icase) = fn; arrays(:, :, 8, icase) = fs; arrays(:, :, 9, icase) = fm
          arrays(:, :, 10, icase) = fluxn; arrays(:, :, 11, icase) = fluxs; arrays(:, :, 12, icase) = fluxm
          scal(:, icase) = (/ wbar, sigw, wdiab, wminf, wmaxf, tair, rhoair, smax, flux_fullact /)
          ints(:, icase) = (/ ia, ismax, msectional, ntype, nsize(1), nsize(2), icase /)
        end do
      end do
    end do
  end do
  ! arrays: na volc dlo_sect dhi_sect sigman hygro fn fs fm fluxn fluxs fluxm
  ! scalars: wbar sigw wdiab wminf wmaxf tair rhoair smax flux_fullact
  ! ints: setup ismax(0 absent, 1 > 0, 2 <= 0) msectional ntype_aer nsize_aer(1:2) case
  call oracle_open('cases')
  call oracle_put('arrays', arrays(:, :, :, 1:icase))
  call oracle_put('scalars', scal(:, 1:icase))
  call oracle_put('ints', ints(:, 1:icase))
  call oracle_close()

contains

  subroutine setup(ia)
    integer, intent(in) :: ia
    na = 0.0; volc = 0.0; dlo = 0.0; dhi = 0.0; sigman = 1.0; hygro = 0.0; nsize = 0
    msectional = 0
    select case (ia)
    case (1)  ! SORGAM-like: Aitken and accumulation modes of one type
      ntype = 1; nsize(1) = 2
      na(1, 1) = 8.0e9; na(2, 1) = 1.2e9
      volc(1, 1) = 2.0e-12; volc(2, 1) = 6.0e-11
      sigman(1, 1) = 1.7; sigman(2, 1) = 2.0
      hygro(1, 1) = 0.5; hygro(2, 1) = 0.6
    case (2)  ! two types, the second non-hygroscopic and one mode empty
      ntype = 2; nsize(1) = 2; nsize(2) = 1
      na(1, 1) = 3.0e8; na(2, 1) = 1.0e-25
      volc(1, 1) = 4.0e-12; volc(2, 1) = 1.0e-40
      sigman(1, 1) = 1.8; sigman(2, 1) = 1.9
      hygro(1, 1) = 0.9; hygro(2, 1) = 0.4
      na(1, 2) = 5.0e8; volc(1, 2) = 3.0e-11; sigman(1, 2) = 2.2; hygro(1, 2) = 1.0e-12
    case (3)  ! a single tiny mode: activate's early return
      ntype = 1; nsize(1) = 1
      na(1, 1) = 1.0e-21; volc(1, 1) = 1.0e-30; sigman(1, 1) = 1.8; hygro(1, 1) = 0.5
    case (4)  ! a clean marine-like accumulation mode, high hygroscopicity
      ntype = 1; nsize(1) = 1
      na(1, 1) = 6.0e7; volc(1, 1) = 9.0e-12; sigman(1, 1) = 1.6; hygro(1, 1) = 1.1
    case (5)  ! quasi-sectional: eight sections spanning 0.04 to 10.24 um (cm here)
      ntype = 1; nsize(1) = 8; msectional = 20
      do m = 1, 8
        dlo(m, 1) = 4.0e-6 * 2.0**(m - 1)
        dhi(m, 1) = 2.0 * dlo(m, 1)
        sigman(m, 1) = 1.25
        hygro(m, 1) = 0.45 + 0.05 * real(m)
      end do
      na(1:8, 1) = (/ 2.0e9, 1.5e9, 9.0e8, 4.0e8, 1.0e8, 2.0e7, 3.0e6, 1.0e-25 /)
      ! volume per particle near each section's centre, (pi/6) d^3 in m3
      do m = 1, 7
        volc(m, 1) = na(m, 1) * 0.5236 * (1.5e-2 * dlo(m, 1))**3
      end do
      volc(8, 1) = 1.0e-40
    case (6)  ! sectional with sections whose mean sits at their edges (one type: the sectional path hands
              ! maxsat a one-element nsize_aer, (/1/), so a second type would read past it)
      ntype = 1; nsize(1) = 4; msectional = 20
      do m = 1, 4
        dlo(m, 1) = 1.0e-5 * 2.0**(m - 1)
        dhi(m, 1) = 2.0 * dlo(m, 1)
        sigman(m, 1) = 1.3
        hygro(m, 1) = 0.3 + 0.1 * real(m)
        na(m, 1) = 4.0e8 / real(m)
      end do
      ! mean diameter at the lower edge, the upper edge, inside, and just above the lower edge
      volc(1, 1) = na(1, 1) * 0.5236 * (1.0e-2 * dlo(1, 1))**3
      volc(2, 1) = na(2, 1) * 0.5236 * (1.0e-2 * dhi(2, 1))**3
      volc(3, 1) = na(3, 1) * 0.5236 * (1.3e-2 * dlo(3, 1))**3
      volc(4, 1) = na(4, 1) * 0.5236 * (1.004e-2 * dlo(4, 1))**3
    case (7)  ! sectional edge cases: a narrow section, means near the upper and lower edges, a mean high in
              ! its section (the dn/dy reshaping), and a high-hygroscopicity tail
      ntype = 1; nsize(1) = 5; msectional = 20
      do m = 1, 5
        dlo(m, 1) = 2.0e-5 * 2.0**(m - 1)
        dhi(m, 1) = 2.0 * dlo(m, 1)
        sigman(m, 1) = 1.2
        hygro(m, 1) = 0.2 + 0.2 * real(m)
        na(m, 1) = 3.0e8 / real(m * m)
      end do
      dhi(1, 1) = 1.005 * dlo(1, 1)
      volc(1, 1) = na(1, 1) * 0.5236 * (1.002e-2 * dlo(1, 1))**3
      volc(2, 1) = na(2, 1) * 0.5236 * (1.995e-2 * dlo(2, 1))**3
      volc(3, 1) = na(3, 1) * 0.5236 * (1.004e-2 * dlo(3, 1))**3
      volc(4, 1) = na(4, 1) * 0.5236 * (1.85e-2 * dlo(4, 1))**3
      volc(5, 1) = na(5, 1) * 0.5236 * (1.7e-2 * dlo(5, 1))**3
    end select
  end subroutine setup

end program run_activate
