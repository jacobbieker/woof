! Fast-TUV photolysis (ftuv_driver, chem/module_ftuv_driver.F) on a 6 x 2
! column grid, 30 levels, after ftuv_init (photo_inti + aer_init) for
! domain 1, dumped after every call.
!
! Columns span latitudes from -62 to 74 degrees and longitudes from -170 to
! 160, so one call holds high sun, low sun, twilight just under the 95 degree
! cut-off and night (photo returns zero rates).  Columns 3, 7 and 10 carry
! cloud water (setcld's optical depth, the 1e-6 g/m3 floor), column 5 is
! saturated aloft (the RH clamp), column 11 is very dry.  Terrain heights
! differ per column (zh is height above the lowest w level).
!
! Cases (one ftuv_driver call each, inputs written beside the outputs):
!   case1  chem_opt = 2, aer_ra_feedback = 0, julday 172, gmt 12, t = 0
!   case2  chem_opt = 2, aer_ra_feedback = 0, julday 172, gmt 12, t = 5400 s
!          (xhour/xmin carry: gmtp = 13.5)
!   case3  chem_opt = 2, aer_ra_feedback = 1 (tauaer/waer/gaer fed to
!          setaer), julday 15, gmt 6
!   case4  chem_opt = 1 (RADM2 without aerosol: isorg = 0), julday 300,
!          gmt 18
!   case5  chem_opt = 2, aer_ra_feedback = 1, julday 1, gmt 23, t = 3000 s
!          (gmtp wraps through 24: mod(xhour, 24))
program run_ftuv
  use oracle_io
  use module_state_description
  use module_configure
  use module_ftuv_driver, only: ftuv_init, ftuv_driver
  implicit none
  integer, parameter :: nx = 6, ny = 2, nz = 30, kme = nz + 1, nph = 47
  integer, parameter :: nlev0 = 51, nbin0 = 17, njmax0 = 58
  real :: chem(nx, kme, ny, num_chem), moist(nx, kme, ny, num_moist)
  real :: t_phy(nx, kme, ny), p_phy(nx, kme, ny), rho(nx, kme, ny)
  real :: p8w(nx, kme, ny), t8w(nx, kme, ny), z_at_w(nx, kme, ny)
  real :: tau(nx, kme, ny, 4), waer(nx, kme, ny, 4), gaer(nx, kme, ny, 4)
  real :: xlat(nx, ny), xlong(nx, ny), gd_cloud(nx, kme, ny)
  real :: ph(nx, kme, ny, nph)
  real :: radfld(nx, nlev0, ny, nbin0), adjcoe(nx, nlev0, ny, njmax0)
  real :: prate(nx, nlev0, ny, njmax0), wc_x(nbin0), zref_x(nlev0)
  integer :: ivgtyp(nx, ny)
  real :: dtstep, gmt, zw, zm, tsfc, hgt, ps, qv0, lapse
  real(kind=8) :: curr_secs
  integer :: i, j, k, c, n, icase, julday
  integer, parameter :: chem_opts(5) = (/ 2, 2, 2, 1, 2 /)
  integer, parameter :: feedbacks(5) = (/ 0, 0, 1, 0, 1 /)
  integer, parameter :: juldays(5) = (/ 172, 172, 15, 300, 1 /)
  real, parameter :: gmts(5) = (/ 12., 12., 6., 18., 23. /)
  real(kind=8), parameter :: secs(5) = (/ 0._8, 5400._8, 0._8, 0._8, 3000._8 /)
  real, parameter :: lats(12) = (/ 35., 0., 51., -33., 74., 12., &
                                   -62., 45., 60., 20., 28., -5. /)
  real, parameter :: lons(12) = (/ -100., 10., 0., 150., -40., 80., &
                                   -170., 160., 25., -60., 45., 120. /)
  real, parameter :: hgts(12) = (/ 300., 0., 50., 800., 1500., 10., &
                                   0., 2500., 150., 0., 1100., 30. /)
  real, parameter :: qvs(12) = (/ 1.e-2, 1.8e-2, 8.e-3, 6.e-3, 2.e-3, 2.e-2, &
                                  4.e-3, 5.e-3, 7.e-3, 1.6e-2, 5.e-4, 1.9e-2 /)
  type(grid_config_rec_type) :: config_flags
  character(len=1024) :: root
  character(len=32) :: name

  call oracle_root(root)
  gd_cloud = 0.
  config_flags%chem_opt = 2
  call ftuv_init(1, 1, nx, 1, ny, nz, nx + 1, ny + 1, config_flags, &
                 config_flags%num_land_cat, 'USGS')

  do j = 1, ny
    do i = 1, nx
      c = i + (j - 1) * nx
      xlat(i, j) = lats(c)
      xlong(i, j) = lons(c)
      ivgtyp(i, j) = 1 + mod(c, 24)
      hgt = hgts(c)
      tsfc = 300. - 0.4 * abs(lats(c)) - 6.5e-3 * hgt
      ps = 1.013e5 * exp(-hgt / 8000.)
      qv0 = qvs(c)
      ! w levels: 40 m first layer, stretching to 1 km layers, top ~17.5 km
      zw = hgt
      do k = 1, kme
        z_at_w(i, k, j) = zw
        zw = zw + min(1000., 40. * 1.2**(k - 1))
      end do
      do k = 1, kme
        if (k <= nz) then
          zm = 0.5 * (z_at_w(i, k, j) + z_at_w(i, k + 1, j))
        else
          zm = z_at_w(i, k, j)
        end if
        lapse = 6.5e-3
        t_phy(i, k, j) = max(tsfc - lapse * (zm - hgt), 205. + 0.1 * real(c))
        t8w(i, k, j) = max(tsfc - lapse * (z_at_w(i, k, j) - hgt), 205. + 0.1 * real(c))
        p_phy(i, k, j) = ps * exp(-(zm - hgt) / 7600.)
        p8w(i, k, j) = ps * exp(-(z_at_w(i, k, j) - hgt) / 7600.)
        rho(i, k, j) = p_phy(i, k, j) / (287. * t_phy(i, k, j))
        moist(i, k, j, :) = 0.
        moist(i, k, j, p_qv) = qv0 * exp(-(zm - hgt) / 2300.)
        if (c == 5 .and. k >= 8 .and. k <= 12) moist(i, k, j, p_qv) = 5.e-2
        if ((c == 3 .or. c == 7 .or. c == 10) .and. k >= 6 .and. k <= 14) &
          moist(i, k, j, p_qc) = 2.e-4 * real(k - 5) * real(1 + mod(c, 3))
        if (c == 10 .and. k == 15) moist(i, k, j, p_qc) = 1.e-9   ! under the floor
        chem(i, k, j, :) = 0.
        chem(i, k, j, p_o3) = 3.e-2 + 2.e-2 * real(mod(c, 4)) &
                              + 6. / (1. + exp(-(zm - 20000.) / 3000.))
        chem(i, k, j, p_sulf) = 1.e-4
        chem(i, k, j, p_ecj) = 0.8 * exp(-(zm - hgt) / 1500.) * real(1 + mod(c, 2))
        chem(i, k, j, p_orgpaj) = 2.0 * exp(-(zm - hgt) / 1500.)
        chem(i, k, j, p_no3aj) = 1.5 * exp(-(zm - hgt) / 1200.)
        chem(i, k, j, p_nh4aj) = 1.1 * exp(-(zm - hgt) / 1200.)
        chem(i, k, j, p_so4aj) = 3.0 * exp(-(zm - hgt) / 1800.)
        chem(i, k, j, p_seas) = 4.0 * exp(-(zm - hgt) / 800.) * real(mod(c, 3))
        chem(i, k, j, p_naaj) = 0.5 * exp(-(zm - hgt) / 800.)
        chem(i, k, j, p_claj) = 0.6 * exp(-(zm - hgt) / 800.)
        do n = 1, 4
          tau(i, k, j, n) = 0.05 * exp(-(zm - hgt) / 1500.) * (1.2 - 0.25 * real(n)) &
                            * real(1 + mod(c, 4))
          waer(i, k, j, n) = 0.97 - 0.02 * real(n) - 0.01 * real(mod(c, 5))
          gaer(i, k, j, n) = 0.72 - 0.04 * real(n) + 0.005 * real(mod(c, 3))
        end do
      end do
    end do
  end do

  do icase = 1, 5
    config_flags%chem_opt = chem_opts(icase)
    config_flags%aer_ra_feedback = feedbacks(icase)
    julday = juldays(icase)
    gmt = gmts(icase)
    curr_secs = secs(icase)
    dtstep = 120.
    ph = -999.
    call ftuv_driver(1, curr_secs, dtstep, config_flags, gmt, julday,          &
         p_phy, t_phy, rho, p8w, t8w, xlat, xlong, z_at_w, moist, chem,        &
         gd_cloud, gd_cloud,                                                   &
         ph(:,:,:,1), ph(:,:,:,2), ph(:,:,:,3), ph(:,:,:,4), ph(:,:,:,5),      &
         ph(:,:,:,6), ph(:,:,:,7), ph(:,:,:,8), ph(:,:,:,9), ph(:,:,:,10),     &
         ph(:,:,:,11), ph(:,:,:,12), ph(:,:,:,13), ph(:,:,:,14), ph(:,:,:,15), &
         ph(:,:,:,16), ph(:,:,:,17), ph(:,:,:,18), ph(:,:,:,19), ph(:,:,:,20), &
         ph(:,:,:,21), ph(:,:,:,22), ph(:,:,:,23), ph(:,:,:,24), ph(:,:,:,25), &
         ph(:,:,:,26), ph(:,:,:,27), ph(:,:,:,28), ph(:,:,:,29), ph(:,:,:,30), &
         ph(:,:,:,31), ph(:,:,:,32), ph(:,:,:,33), ph(:,:,:,34), ph(:,:,:,35), &
         ph(:,:,:,36), ph(:,:,:,37), ph(:,:,:,38), ph(:,:,:,39), ph(:,:,:,40), &
         ph(:,:,:,41), ph(:,:,:,42), ph(:,:,:,43), ph(:,:,:,44), ph(:,:,:,45), &
         ph(:,:,:,46), ph(:,:,:,47),                                           &
         ivgtyp, nlev0, nbin0, njmax0, radfld, adjcoe, prate, wc_x, zref_x,    &
         tau(:,:,:,1), tau(:,:,:,2), tau(:,:,:,3), tau(:,:,:,4),               &
         waer(:,:,:,1), waer(:,:,:,2), waer(:,:,:,3), waer(:,:,:,4),           &
         gaer(:,:,:,1), gaer(:,:,:,2), gaer(:,:,:,3), gaer(:,:,:,4),           &
         1, nx, 1, ny, 1, kme, 1, nx, 1, ny, 1, kme, 1, nx, 1, ny, 1, nz)
    write(name, '(A,I0)') 'case', icase
    call oracle_open(trim(name))
    call oracle_put('chem_opt', chem_opts(icase))
    call oracle_put('aer_ra_feedback', feedbacks(icase))
    call oracle_put('julday', julday)
    call oracle_put('gmt', gmt)
    call oracle_put('curr_secs', real(curr_secs))
    call oracle_put('dtstep', dtstep)
    call oracle_put('xlat', xlat); call oracle_put('xlong', xlong)
    call oracle_put('p_phy', p_phy); call oracle_put('t_phy', t_phy)
    call oracle_put('rho_phy', rho); call oracle_put('p8w', p8w)
    call oracle_put('t8w', t8w); call oracle_put('z_at_w', z_at_w)
    call oracle_put('moist', moist); call oracle_put('chem', chem)
    call oracle_put('tauaer', tau); call oracle_put('waer', waer)
    call oracle_put('gaer', gaer)
    call oracle_put('ph', ph)
    call oracle_close()
  end do
end program run_ftuv
