! MADE/SORGAM dry deposition (sorgam_depdriver with VDVG_2, aer_drydep_opt
! = 1, and VDVG, aer_drydep_opt = 11) and the PM sums (sum_pm_sorgam) on 8 x 2
! columns, chem/module_aerosols_sorgam.F.
!
! Columns cover friction velocity under and over sorgam_depdriver's 0.1 m/s
! floor, stable, neutral and unstable surface layers (rmol), smooth and rough
! surfaces (znt), shallow and deep boundary layers, small and large
! aerodynamic resistance, warm and cold, dry and humid air, clean and
! polluted aerosol.  aerosols_sorgam_init runs first for the module
! constants (chem_in_opt = 1 returns before touching chem).
program run_sorgam_dep
  use oracle_io
  use module_state_description
  use module_configure
  use module_data_sorgam
  use sorgam_extract, only: sorgam_depdriver, sum_pm_sorgam, aerosols_sorgam_init
  implicit none
  integer, parameter :: nx = 8, ny = 2, nz = 4, numgas = 43, numaer = num_chem - numgas
  real :: chem(nx, nz, ny, num_chem), moist(nx, nz, ny, num_moist)
  real, dimension(nx, nz, ny) :: t_phy, p_phy, rho, alt, dz8w, z, z_at_w, t8w, p8w, &
       h2oaj, h2oai, nu3, ac3, cor3, asulf, ahno3, anh3, cvaro1, cvaro2, cvalk1, &
       cvole1, cvapi1, cvapi2, cvlim1, cvlim2, convfac, pm25, pm25w, pm25ec, pm10
  real :: ust(nx, ny), rmol(nx, ny), znt(nx, ny), pbl(nx, ny), aer_res(nx, ny)
  real :: vgsa(nx, ny, numaer), ddflx(nx, ny, numgas + 1:num_chem), base(num_chem)
  real, parameter :: usts(16) = (/ .05, .3, .6, 1.2, .2, .08, .4, .9, &
                                   .15, .5, .25, .7, .1, .35, 1.5, .45 /)
  real, parameter :: rmols(16) = (/ 0., .02, -.02, 1.e-7, .1, -.1, -.005, .005, &
                                    0., -.5, .3, -1.e-3, 2.e-3, 0., -.05, .05 /)
  real, parameter :: znts(16) = (/ 1.e-4, .01, .1, 1., .05, 2.e-4, .3, .02, &
                                   .5, 1.e-3, .08, .2, .002, .9, .03, .15 /)
  real, parameter :: pbls(16) = (/ 50., 300., 1200., 2500., 800., 100., 1500., 600., &
                                   200., 2000., 400., 900., 150., 3000., 700., 1000. /)
  real, parameter :: ress(16) = (/ 10., 50., 200., 5., 80., 300., 20., 120., &
                                   1., 60., 150., 30., 400., 15., 90., 45. /)
  real, parameter :: polls(16) = (/ 1., 10., .1, 3., 1., 30., .3, 1., &
                                    1., 5., .05, 1., 2., 1., 1., 20. /)
  real, parameter :: qvs(16) = (/ 1.e-2, 5.e-3, 1.e-4, 2.e-2, 1.5e-2, 8.e-3, 1.e-6, 3.e-3, &
                                  1.2e-2, 2.e-3, 2.5e-2, 6.e-3, 9.e-3, 4.e-3, 1.e-2, 1.8e-2 /)
  logical :: is_aerosol(num_chem)
  integer :: i, j, k, c, n, opt
  type(grid_config_rec_type) :: config_flags
  character(len=1024) :: root
  character(len=32) :: name

  call oracle_root(root)
  base = 0.
  base(p_sulf) = 1.e-4; base(p_hno3) = 1.e-3; base(p_nh3) = 5.e-3; base(p_hcl) = 1.e-4
  base(p_so4aj) = 5.;   base(p_so4ai) = .5;   base(p_nh4aj) = 1.5;  base(p_nh4ai) = .1
  base(p_no3aj) = 1.;   base(p_no3ai) = .05;  base(p_naaj) = .1;    base(p_naai) = .01
  base(p_claj) = .1;    base(p_clai) = .01
  base(p_orgaro1j) = .1;  base(p_orgaro1i) = .01; base(p_orgaro2j) = .1; base(p_orgaro2i) = .01
  base(p_orgalk1j) = .05; base(p_orgalk1i) = .005; base(p_orgole1j) = .05; base(p_orgole1i) = .005
  base(p_orgba1j) = .2; base(p_orgba1i) = .02; base(p_orgba2j) = .2; base(p_orgba2i) = .02
  base(p_orgba3j) = .1; base(p_orgba3i) = .01; base(p_orgba4j) = .1; base(p_orgba4i) = .01
  base(p_orgpaj) = 2.;  base(p_orgpai) = .2;   base(p_ecj) = .5;     base(p_eci) = .05
  base(p_p25j) = 1.;    base(p_p25i) = .1;     base(p_antha) = 3.;   base(p_seas) = 2.
  base(p_soila) = 3.;   base(p_nu0) = 1.e9;    base(p_ac0) = 1.e8;   base(p_corn) = 1.e5
  do j = 1, ny
    do i = 1, nx
      c = i + (j - 1) * nx
      ust(i, j) = usts(c); rmol(i, j) = rmols(c); znt(i, j) = znts(c)
      pbl(i, j) = pbls(c); aer_res(i, j) = ress(c)
      do k = 1, nz
        z(i, k, j) = 25. + 50. * real(k - 1); z_at_w(i, k, j) = 50. * real(k - 1)
        dz8w(i, k, j) = 50.
        t_phy(i, k, j) = 300. - 6.5e-3 * z(i, k, j) + 3. * real(i - 4) - 12. * real(j - 1)
        if (c == 7) t_phy(i, k, j) = t_phy(i, k, j) - 70.
        t8w(i, k, j) = t_phy(i, k, j)
        p_phy(i, k, j) = 1.e5 * exp(-z(i, k, j) / 7400.)
        p8w(i, k, j) = p_phy(i, k, j)
        rho(i, k, j) = p_phy(i, k, j) / (287. * t_phy(i, k, j))
        alt(i, k, j) = 1. / rho(i, k, j)
        moist(i, k, j, :) = 0.
        moist(i, k, j, p_qv) = qvs(c)
        do n = 1, num_chem
          chem(i, k, j, n) = base(n) * polls(c)
        end do
        h2oaj(i, k, j) = 3. * polls(c); h2oai(i, k, j) = .2 * polls(c)
        nu3(i, k, j) = 0.; ac3(i, k, j) = 0.; cor3(i, k, j) = 0.
        asulf(i, k, j) = 0.; ahno3(i, k, j) = 0.; anh3(i, k, j) = 0.
        cvaro1(i, k, j) = .1; cvaro2(i, k, j) = .05; cvalk1(i, k, j) = .02; cvole1(i, k, j) = .02
        cvapi1(i, k, j) = 0.; cvapi2(i, k, j) = 0.; cvlim1(i, k, j) = 0.; cvlim2(i, k, j) = 0.
        convfac(i, k, j) = p_phy(i, k, j) / (8.314510 * t_phy(i, k, j))
      end do
      if (c == 11) chem(i, :, j, p_so4ai) = 0.
    end do
  end do
  call aerosols_sorgam_init(chem, convfac, z_at_w, pm25, pm25w, pm25ec, 1, 1, is_aerosol, &
       1, nx, 1, ny, 1, nz, 1, nx, 1, ny, 1, nz, 1, nx, 1, ny, 1, nz, config_flags)
  cw_phase = -999888777
  call oracle_open('init')
  call oracle_put('chem', chem)
  call oracle_put('t_phy', t_phy); call oracle_put('p_phy', p_phy)
  call oracle_put('alt', alt); call oracle_put('qv', moist(:, :, :, p_qv))
  call oracle_put('h2oaj', h2oaj); call oracle_put('h2oai', h2oai)
  call oracle_put('cvaro1', cvaro1); call oracle_put('cvaro2', cvaro2)
  call oracle_put('cvalk1', cvalk1); call oracle_put('cvole1', cvole1)
  call oracle_put('ust', ust); call oracle_put('rmol', rmol); call oracle_put('znt', znt)
  call oracle_put('pbl', pbl); call oracle_put('aer_res', aer_res)
  call oracle_close()
  do opt = 1, 2
    config_flags%aer_drydep_opt = merge(1, 11, opt == 1)
    vgsa = -1.; ddflx = 0.
    call sorgam_depdriver(1, config_flags, 1, 120., ust, t_phy, moist, p8w, t8w, rmol, znt, &
         pbl, alt, p_phy, chem, rho, dz8w, z, z_at_w, h2oaj, h2oai, nu3, ac3, cor3, asulf,  &
         ahno3, anh3, cvaro1, cvaro2, cvalk1, cvole1, cvapi1, cvapi2, cvlim1, cvlim2,       &
         aer_res, vgsa, numgas, ddflx, numaer,                                              &
         1, nx, 1, ny, 1, nz, 1, nx, 1, ny, 1, nz, 1, nx, 1, ny, 1, nz)
    write(name, '(A,I0)') 'drydep_opt', config_flags%aer_drydep_opt
    call oracle_open(trim(name))
    call oracle_put('vgsa', vgsa)
    call oracle_put('aer_res', aer_res)
    call oracle_close()
  end do
  ! ide/jde are WRF's staggered domain ends (mass points 1..nx, ide = nx + 1),
  ! so sum_pm_sorgam's ii = min(ide-1, i) is i itself.
  call sum_pm_sorgam(alt, chem, h2oaj, h2oai, pm25, pm25w, pm25ec, pm10, 0,     &
       1, nx + 1, 1, ny + 1, 1, nz + 1, 1, nx, 1, ny, 1, nz, 1, nx, 1, ny, 1, nz)
  call oracle_open('sum_pm')
  call oracle_put('pm2_5_dry', pm25); call oracle_put('pm2_5_water', pm25w)
  call oracle_put('pm2_5_dry_ec', pm25ec); call oracle_put('pm10', pm10)
  call oracle_close()
end program run_sorgam_dep
