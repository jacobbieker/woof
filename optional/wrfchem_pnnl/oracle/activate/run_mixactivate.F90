! WOOF-OPTIN: wrfchem_pnnl
! Aerosol activation and droplet-number mixing (phys/module_mixactivate.F,
! WRF v4.7.1): `mixactivate` over one 4 x 2 tile of 20-level columns, two
! consecutive calls (the second with the first's cloud fraction as the old
! one), for a SORGAM-like modal description (msectional = 0: two types,
! Aitken + accumulation and coarse; interstitial and cloud-borne phases) and
! a MOSAIC-like sectional one (msectional = 20, four sections).  The columns
! reach cloud forming (in place and from below), dissipating, persisting,
! clear air, ice and the surface-deposition limits.  Memory dimensions carry
! a halo (ims = its - 1 ...), so the arrays are indexed as WRF indexes them.
! Inputs are built from literals with exact arithmetic only.
!
! Each configuration writes one directory (modal, sectional) holding the
! inputs and, per call n = 1, 2, chem_n, qndrop3d_n, cldfra_n, qsrflx_n,
! ccn_n (ccn1..ccn6 stacked), nsource_n.
program run_mixactivate
  use oracle_io
  use activate_extract, only: mixactivate
  implicit none
  integer, parameter :: maxd_atype = 2, maxd_asize = 8, maxd_acomp = 20, maxd_aphase = 2
  integer, parameter :: num_chem = 40
  integer, parameter :: ids = 1, ide = 5, jds = 1, jde = 3, kds = 1, kde = 21
  integer, parameter :: its = 1, ite = 4, jts = 1, jte = 2, kts = 1, kte = 20
  integer, parameter :: ims = 0, ime = 5, jms = 0, jme = 3, kms = 1, kme = 21
  real :: chem(ims:ime, kms:kme, jms:jme, num_chem)
  real, dimension(ims:ime, kms:kme, jms:jme) :: qv, qc, qi, qndrop3d, temp, w, rho, zm, dz8w, &
       p_at_w, t_at_w, kvh, cldfra, cldfra_old, nsource, ccn1, ccn2, ccn3, ccn4, ccn5, ccn6
  real :: ccn(ims:ime, kms:kme, jms:jme, 6)
  real :: ddvel(its:ite, jts:jte, num_chem), qsrflx(ims:ime, jms:jme, num_chem)
  real :: hygro(its:ite, kts:kte, jts:jte, maxd_asize, maxd_atype)
  integer :: ncomp_aer(maxd_atype), nsize_aer(maxd_atype)
  integer :: massptr_aer(maxd_acomp, maxd_asize, maxd_atype, maxd_aphase)
  integer :: numptr_aer(maxd_asize, maxd_atype, maxd_aphase), waterptr_aer(maxd_asize, maxd_atype)
  real, dimension(maxd_asize, maxd_atype) :: dlo_sect, dhi_sect, sigmag_aer, dpvolmean_aer
  real, dimension(maxd_acomp, maxd_atype) :: dens_aer, mw_aer
  integer :: msectional, ntype_aer, nphase_aer, iconf, icall, i, j, k, l, m, n, ic
  real :: dtstep
  logical :: f_qc, f_qi
  character(len=1024) :: root
  character(len=16) :: tag

  call oracle_root(root)
  do iconf = 1, 2
    call describe(iconf)
    call fields()
    dtstep = 90.0
    f_qc = .true.; f_qi = .true.
    call oracle_open(merge('modal    ', 'sectional', iconf == 1))
    call oracle_put('chem_0', chem); call oracle_put('qndrop3d_0', qndrop3d)
    call oracle_put('qv', qv); call oracle_put('qc', qc); call oracle_put('qi', qi)
    call oracle_put('temp', temp); call oracle_put('w', w); call oracle_put('rho', rho)
    call oracle_put('zm', zm); call oracle_put('dz8w', dz8w); call oracle_put('p_at_w', p_at_w)
    call oracle_put('t_at_w', t_at_w); call oracle_put('kvh', kvh); call oracle_put('cldfra_old_0', cldfra_old)
    call oracle_put('ddvel', ddvel); call oracle_put('hygro', reshape(hygro, (/ 4, 20, 2, 16 /)))
    call oracle_put('ints', (/ msectional, ntype_aer, nphase_aer, num_chem /))
    call oracle_put('ncomp_aer', ncomp_aer); call oracle_put('nsize_aer', nsize_aer)
    call oracle_put('numptr_aer', numptr_aer); call oracle_put('massptr_aer', reshape(massptr_aer, (/ maxd_acomp, maxd_asize, maxd_atype * maxd_aphase /)))
    call oracle_put('waterptr_aer', waterptr_aer)
    call oracle_put('dlo_sect', dlo_sect); call oracle_put('dhi_sect', dhi_sect)
    call oracle_put('sigmag_aer', sigmag_aer); call oracle_put('dpvolmean_aer', dpvolmean_aer)
    call oracle_put('dens_aer', dens_aer); call oracle_put('mw_aer', mw_aer)
    call oracle_put('dtstep', dtstep)
    do icall = 1, 2
      cldfra = -9.0
      nsource = -9.0; ccn1 = -9.0; ccn2 = -9.0; ccn3 = -9.0; ccn4 = -9.0; ccn5 = -9.0; ccn6 = -9.0
      qsrflx = -9.0
      call mixactivate(msectional, chem, num_chem, qv, qc, qi, qndrop3d, temp, w, ddvel, 1, &
           maxd_acomp, maxd_asize, maxd_atype, maxd_aphase, ncomp_aer, nsize_aer, ntype_aer, nphase_aer, &
           numptr_aer, massptr_aer, dlo_sect, dhi_sect, sigmag_aer, dpvolmean_aer, dens_aer, mw_aer, &
           waterptr_aer, hygro, 1, 2, &
           ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte, &
           rho, zm, dz8w, p_at_w, t_at_w, kvh, cldfra, cldfra_old, qsrflx, &
           ccn1, ccn2, ccn3, ccn4, ccn5, ccn6, nsource, 1, icall, dtstep, f_qc, f_qi)
      ccn(:, :, :, 1) = ccn1; ccn(:, :, :, 2) = ccn2; ccn(:, :, :, 3) = ccn3
      ccn(:, :, :, 4) = ccn4; ccn(:, :, :, 5) = ccn5; ccn(:, :, :, 6) = ccn6
      write(tag, '(I0)') icall
      call oracle_put('chem_' // trim(tag), chem); call oracle_put('qndrop3d_' // trim(tag), qndrop3d)
      call oracle_put('cldfra_' // trim(tag), cldfra); call oracle_put('qsrflx_' // trim(tag), qsrflx)
      call oracle_put('ccn_' // trim(tag), ccn); call oracle_put('nsource_' // trim(tag), nsource)
      ! the next call sees this call's cloud as the old one, and new cloud elsewhere
      cldfra_old = cldfra
      do j = jts, jte
        do i = its, ite
          do k = kts, kte
            if (mod(i + j + k, 3) == 0) qc(i, k, j) = qc(i, k, j) + 2.0e-5
            if (mod(i + 2 * j + k, 5) == 0) qc(i, k, j) = 0.0
          end do
        end do
      end do
    end do
    call oracle_close()
  end do

contains

  subroutine describe(iconf)
    integer, intent(in) :: iconf
    ncomp_aer = 0; nsize_aer = 0; massptr_aer = -999888777; numptr_aer = -999888777; waterptr_aer = 0
    dlo_sect = 0.0; dhi_sect = 0.0; sigmag_aer = 1.0; dpvolmean_aer = 0.0; dens_aer = 0.0; mw_aer = 0.0
    nphase_aer = 2
    if (iconf == 1) then
      ! SORGAM-like: type 1 Aitken + accumulation (so4, nh4, orgpa, ec), type 2 coarse (soil, seas)
      msectional = 0; ntype_aer = 2; nsize_aer(1) = 2; nsize_aer(2) = 1
      ncomp_aer(1) = 4; ncomp_aer(2) = 2
      numptr_aer(1, 1, 1) = 1; numptr_aer(2, 1, 1) = 2; numptr_aer(1, 2, 1) = 3
      numptr_aer(1, 1, 2) = 14; numptr_aer(2, 1, 2) = 15; numptr_aer(1, 2, 2) = 16
      do l = 1, 4
        massptr_aer(l, 1, 1, 1) = 3 + l;  massptr_aer(l, 2, 1, 1) = 7 + l
        massptr_aer(l, 1, 1, 2) = 16 + l; massptr_aer(l, 2, 1, 2) = 20 + l
      end do
      do l = 1, 2
        massptr_aer(l, 1, 2, 1) = 11 + l; massptr_aer(l, 1, 2, 2) = 24 + l
      end do
      waterptr_aer(1, 1) = 27; waterptr_aer(2, 1) = 28
      sigmag_aer(1, 1) = 1.7; sigmag_aer(2, 1) = 2.0; sigmag_aer(1, 2) = 2.2
      dpvolmean_aer(1, 1) = 2.0e-6; dpvolmean_aer(2, 1) = 1.5e-5; dpvolmean_aer(1, 2) = 2.5e-4
      dlo_sect(1, 1) = 1.0e-7; dhi_sect(1, 1) = 1.0e-5
      dlo_sect(2, 1) = 1.0e-5; dhi_sect(2, 1) = 2.5e-4
      dlo_sect(1, 2) = 2.5e-4; dhi_sect(1, 2) = 1.0e-3
      dens_aer(1:4, 1) = (/ 1.8, 1.8, 1.4, 1.7 /); dens_aer(1:2, 2) = (/ 2.6, 2.2 /)
      mw_aer(1:4, 1) = (/ 96.0, 18.0, 250.0, 12.0 /); mw_aer(1:2, 2) = (/ 100.0, 58.5 /)
    else
      ! MOSAIC-like sectional: one type, four sections (so4, nh4, ec), quasi-sectional activation
      msectional = 20; ntype_aer = 1; nsize_aer(1) = 4; ncomp_aer(1) = 3
      do m = 1, 4
        numptr_aer(m, 1, 1) = m; numptr_aer(m, 1, 2) = 20 + m
        do l = 1, 3
          massptr_aer(l, m, 1, 1) = 4 + 3 * (m - 1) + l
          massptr_aer(l, m, 1, 2) = 24 + 3 * (m - 1) + l
        end do
        waterptr_aer(m, 1) = 0
        dlo_sect(m, 1) = 3.9e-6 * 2.0**(2 * (m - 1))
        dhi_sect(m, 1) = 4.0 * dlo_sect(m, 1)
        dpvolmean_aer(m, 1) = 2.0 * dlo_sect(m, 1)
        sigmag_aer(m, 1) = 1.2
      end do
      dens_aer(1:3, 1) = (/ 1.8, 1.8, 1.7 /); mw_aer(1:3, 1) = (/ 96.0, 18.0, 12.0 /)
    end if
  end subroutine describe

  subroutine fields()
    real :: z, cloud
    chem = 0.0
    qv = 0.0; qc = 0.0; qi = 0.0; qndrop3d = 0.0; temp = 0.0; w = 0.0; rho = 0.0; zm = 0.0
    dz8w = 0.0; p_at_w = 0.0; t_at_w = 0.0; kvh = 0.0; cldfra_old = 0.0
    do j = jms, jme
      do i = ims, ime
        do k = kms, kme
          z = 50.0 + 400.0 * real(k - 1) + 10.0 * real(i + j)
          zm(i, k, j) = z
          dz8w(i, k, j) = 400.0 + 5.0 * real(i)
          temp(i, k, j) = 290.0 - 0.0065 * z
          t_at_w(i, k, j) = temp(i, k, j) + 1.3
          rho(i, k, j) = 1.2 - 0.04 * real(k - 1)
          p_at_w(i, k, j) = 1.0e5 - 450.0 * real(k - 1)
          qv(i, k, j) = 0.012 - 0.0005 * real(k - 1)
          kvh(i, k, j) = 0.004 + 3.0 * real(mod(k * (i + 1), 7))
          w(i, k, j) = 0.05 * real(mod(i + 3 * j + k, 9) - 2)
          do l = 1, num_chem
            ! numbers (#/kg) and masses (ug/kg), different per species, level and column
            chem(i, k, j, l) = (1.0 + 0.1 * real(mod(l * k + i, 11))) * merge(4.0e8, 1.5, &
                 (msectional == 0 .and. any(l == (/ 1, 2, 3, 14, 15, 16 /))) .or. &
                 (msectional > 0 .and. (l <= 4 .or. (l > 20 .and. l <= 24))))
            if (l > 13 .and. l < 27 .and. msectional == 0) chem(i, k, j, l) = 0.1 * chem(i, k, j, l)
            ! sectional: fewer, heavier particles, so the larger sections activate
            if (msectional > 0) chem(i, k, j, l) = chem(i, k, j, l) * merge(0.05, 20.0, &
                 l <= 4 .or. (l > 20 .and. l <= 24))
          end do
        end do
      end do
    end do
    do j = jts, jte
      do i = its, ite
        do k = kts, kte
          ! column (i, j) cloud: forming low (cl = 1), dissipating (2), persistent (3), clear (4) ...
          ic = i + 4 * (j - 1)
          cloud = 0.0
          select case (ic)
          case (1)  ! forming in place, levels 4..9
            if (k >= 4 .and. k <= 9) cloud = 3.0e-4
          case (2)  ! dissipating: old cloud, none now
            if (k >= 3 .and. k <= 8) cldfra_old(i, k, j) = 1.0
          case (3)  ! persistent
            if (k >= 5 .and. k <= 12) then
              cloud = 2.0e-4; cldfra_old(i, k, j) = 1.0
            end if
          case (4)  ! clear
          case (5)  ! deepening upward under updraft (cloud above the old top)
            if (k >= 2 .and. k <= 14) cloud = 1.0e-4
            if (k >= 2 .and. k <= 7) cldfra_old(i, k, j) = 1.0
            w(i, k, j) = 1.5
          case (6)  ! ice only, and liquid with ice
            if (k >= 10 .and. k <= 16) qi(i, k, j) = 5.0e-5
            if (k >= 12 .and. k <= 14) cloud = 5.0e-5
          case (7)  ! thin cloud just above the threshold and partial old cloud
            if (k >= 6 .and. k <= 10) cloud = 1.5e-6
            if (k >= 6 .and. k <= 8) cldfra_old(i, k, j) = 0.5
          case (8)  ! cloud at the surface (deposition of droplets)
            if (k <= 3) cloud = 4.0e-4
            if (k <= 2) cldfra_old(i, k, j) = 1.0
          end select
          qc(i, k, j) = cloud
          qndrop3d(i, k, j) = merge(1.0e8, 0.0, cldfra_old(i, k, j) > 0.0)
          do n = 1, maxd_atype
            do m = 1, maxd_asize
              hygro(i, k, j, m, n) = 0.2 + 0.1 * real(m) + 0.05 * real(n) + 0.01 * real(k)
            end do
          end do
        end do
        do l = 1, num_chem
          ddvel(i, j, l) = 0.001 * real(mod(l, 5)) + merge(2.0, 0.0, l == 3)
        end do
      end do
    end do
  end subroutine fields

end program run_mixactivate
