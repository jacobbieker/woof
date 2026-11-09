! MADE/SORGAM (sorgam_driver, chem/module_aerosols_sorgam.F, isorop = 0
! branch) on 8 x 2 columns of 10 levels, dumped after every call.
!
! aerosols_sorgam_init runs first with chem_in_opt = 1: it computes the module
! constants (mode-width powers, minimum numbers, Gauss-Hermite nodes) and
! returns before touching chem.  cw_phase takes the value
! aerosols_sorgam_init_aercld_ptrs gives a package without cloud-borne
! aerosol (module_prep_wetscav_sorgam.F:37).
!
! Cells cover clean and polluted air, the eql4 skip (T < 233.15 K, RH < 0.1),
! sulfate production from zero to strong (so4rat), new-particle formation,
! RPMARES's ammonia-rich, sulfate-rich (cubic) and sulfate-free branches,
! SOA from the RADM2 precursor increments, number floors (nu0/ac0 below
! 1e7), and zero species (the epsilc floors).  Two schedules carry every
! array across calls: dtstep = 120 s (3 calls) and 600 s (2 calls); a third,
! 300 s (2 calls), runs with cw_phase = 2.
program run_sorgam
  use oracle_io
  use module_state_description
  use module_configure
  use module_data_sorgam
  use sorgam_extract, only: sorgam_driver, aerosols_sorgam_init
  implicit none
  integer, parameter :: nx = 8, ny = 2, nz = 10
  real :: chem(nx, nz, ny, num_chem), moist(nx, nz, ny, num_moist)
  real :: t_phy(nx, nz, ny), p_phy(nx, nz, ny), rho(nx, nz, ny), alt(nx, nz, ny)
  real :: dz8w(nx, nz, ny), z(nx, nz, ny), z_at_w(nx, nz, ny), t8w(nx, nz, ny)
  real :: p8w(nx, nz, ny), aerwrf(nx, nz, ny), convfac(nx, nz, ny)
  real :: pm25(nx, nz, ny), pm25w(nx, nz, ny), pm25ec(nx, nz, ny)
  real, dimension(nx, nz, ny) :: h2oaj, h2oai, nu3, ac3, cor3, asulf, ahno3, anh3, &
       cvaro1, cvaro2, cvalk1, cvole1, cvapi1, cvapi2, cvlim1, cvlim2, vcsulf_old
  real :: vdrog3(nx, nz, ny, ldrog), base(num_chem), poll, zk, dtstep
  real, parameter :: polls(16) = (/ 1., 10., .1, 3., 1., 30., .3, 1., &
                                    1., 5., .05, 1., 2., 1., 1., 20. /)
  real, parameter :: qvs(16) = (/ 1.e-2, 5.e-3, 1.e-4, 2.e-2, 1.5e-2, 8.e-3, 1.e-6, 3.e-3, &
                                  1.2e-2, 2.e-3, 2.5e-2, 6.e-3, 9.e-3, 4.e-3, 1.e-2, 1.8e-2 /)
  real, parameter :: so4p(16) = (/ 1., 1., 1., 0., 5., 1., 1., 20., &
                                   1., .1, 1., 1., 0., 3., 1., 1. /)
  logical :: is_aerosol(num_chem)
  integer :: i, j, k, c, n, sched, step, nstep
  type(grid_config_rec_type) :: config_flags
  character(len=1024) :: root
  character(len=32) :: name

  call oracle_root(root)
  base = 0.
  base(p_sulf) = 1.e-4; base(p_hno3) = 1.e-3; base(p_nh3) = 5.e-3; base(p_hcl) = 1.e-4
  base(p_so2) = 5.e-3;  base(p_o3) = 4.e-2
  ! aerosol mass in ug/kg dry air, number in #/kg
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

  do sched = 1, 3
    do j = 1, ny
      do i = 1, nx
        c = i + (j - 1) * nx
        poll = polls(c)
        do k = 1, nz
          zk = 50. + 600. * real(k - 1)
          z(i, k, j) = zk; z_at_w(i, k, j) = zk - 25.; dz8w(i, k, j) = 600.
          t_phy(i, k, j) = 300. - 6.5e-3 * zk + 3. * real(i - 4) - 12. * real(j - 1)
          t8w(i, k, j) = t_phy(i, k, j)
          p_phy(i, k, j) = 1.e5 * exp(-zk / 7400.)
          p8w(i, k, j) = p_phy(i, k, j)
          rho(i, k, j) = p_phy(i, k, j) / (287. * t_phy(i, k, j))
          alt(i, k, j) = 1. / rho(i, k, j)
          moist(i, k, j, :) = 0.
          moist(i, k, j, p_qv) = qvs(c) * exp(-zk / 2500.)
          do n = 1, num_chem
            chem(i, k, j, n) = base(n) * poll * exp(-zk / 3000.)
          end do
          chem(i, k, j, p_o3) = base(p_o3)
          vcsulf_old(i, k, j) = chem(i, k, j, p_sulf) / (1. + .01 * so4p(c))
          h2oaj(i, k, j) = 0.; h2oai(i, k, j) = 0.
          nu3(i, k, j) = 0.; ac3(i, k, j) = 0.; cor3(i, k, j) = 0.
          asulf(i, k, j) = 0.; ahno3(i, k, j) = 0.; anh3(i, k, j) = 0.
          cvaro1(i, k, j) = .1 * poll; cvaro2(i, k, j) = .05 * poll
          cvalk1(i, k, j) = .02 * poll; cvole1(i, k, j) = .02 * poll
          cvapi1(i, k, j) = 0.; cvapi2(i, k, j) = 0.; cvlim1(i, k, j) = 0.; cvlim2(i, k, j) = 0.
          do n = 1, ldrog
            vdrog3(i, k, j, n) = 1.e-5 * poll * real(n) / real(ldrog)
          end do
          aerwrf(i, k, j) = 0.
          convfac(i, k, j) = p_phy(i, k, j) / (8.314510 * t_phy(i, k, j))
        end do
        if (c == 3) chem(i, :, j, p_nu0) = 1.e5
        if (c == 3) chem(i, :, j, p_ac0) = 1.e4
        if (c == 11) then
          chem(i, :, j, p_so4ai) = 0.; chem(i, :, j, p_nh4aj) = 0.; chem(i, :, j, p_hcl) = 0.
        end if
        if (c == 7) t_phy(i, :, j) = t_phy(i, :, j) - 70.
        ! RPMARES's sulfate-rich branches (total ammonium / sulfate < 2):
        ! the cubic for nitrate (13), nearly no ammonia (12).
        if (c == 12 .or. c == 13) then
          chem(i, :, j, p_nh3) = 1.e-6
          chem(i, :, j, p_nh4aj) = .02 * real(c - 11)**3
          chem(i, :, j, p_nh4ai) = .002 * real(c - 11)**3
          chem(i, :, j, p_so4aj) = 8.
        end if
        ! no sulfate at all (tso4 under minso4), with nitrate (14) and without (15)
        if (c == 14 .or. c == 15) then
          chem(i, :, j, p_so4aj) = 0.; chem(i, :, j, p_so4ai) = 0.
          chem(i, :, j, p_sulf) = 0.; vcsulf_old(i, :, j) = 0.
        end if
        if (c == 15) then
          chem(i, :, j, p_hno3) = 0.; chem(i, :, j, p_no3aj) = 0.; chem(i, :, j, p_no3ai) = 0.
        end if
      end do
    end do
    call aerosols_sorgam_init(chem, convfac, z_at_w, pm25, pm25w, pm25ec, 1, 1, is_aerosol, &
         1, nx, 1, ny, 1, nz, 1, nx, 1, ny, 1, nz, 1, nx, 1, ny, 1, nz, config_flags)
    ! schedule 3: a package with cloud-borne aerosol (cw_phase = 2,
    ! module_prep_wetscav_sorgam.F:43), which turns on modpar's diameter limits
    cw_phase = -999888777
    if (sched == 3) cw_phase = 2
    if (sched == 1) then
      dtstep = 120.; nstep = 3
    else if (sched == 2) then
      dtstep = 600.; nstep = 2
    else
      dtstep = 300.; nstep = 2
    end if
    write(name, '(A,I0,A)') 'sched', sched, '_init'
    call oracle_open(trim(name))
    call oracle_put('chem', chem)
    call oracle_put('t_phy', t_phy); call oracle_put('p_phy', p_phy)
    call oracle_put('alt', alt); call oracle_put('qv', moist(:, :, :, p_qv))
    call oracle_put('vcsulf_old', vcsulf_old); call oracle_put('vdrog3', vdrog3)
    call oracle_put('cvaro1', cvaro1); call oracle_put('cvaro2', cvaro2)
    call oracle_put('cvalk1', cvalk1); call oracle_put('cvole1', cvole1)
    call oracle_put('dtstep', dtstep)
    call oracle_close()
    do step = 1, nstep
      call sorgam_driver(1, step, dtstep, t_phy, moist, aerwrf, p8w, t8w, alt, p_phy, &
           chem, rho, dz8w, z, z_at_w, h2oaj, h2oai, nu3, ac3, cor3, asulf, ahno3,  &
           anh3, cvaro1, cvaro2, cvalk1, cvole1, cvapi1, cvapi2, cvlim1, cvlim2,     &
           vcsulf_old, vdrog3, 1,                                                    &
           1, nx, 1, ny, 1, nz, 1, nx, 1, ny, 1, nz, 1, nx, 1, ny, 1, nz)
      write(name, '(A,I0,A,I0)') 'sched', sched, '_step', step
      call oracle_open(trim(name))
      call oracle_put('chem', chem)
      call oracle_put('h2oaj', h2oaj); call oracle_put('h2oai', h2oai)
      call oracle_put('nu3', nu3); call oracle_put('ac3', ac3); call oracle_put('cor3', cor3)
      call oracle_put('cvaro1', cvaro1); call oracle_put('cvaro2', cvaro2)
      call oracle_put('cvalk1', cvalk1); call oracle_put('cvole1', cvole1)
      call oracle_put('cvapi1', cvapi1); call oracle_put('cvapi2', cvapi2)
      call oracle_put('cvlim1', cvlim1); call oracle_put('cvlim2', cvlim2)
      ! the next call's sulfate production starts from this call's sulfate
      vcsulf_old = chem(:, :, :, p_sulf) / 1.01
      call oracle_put('vcsulf_next', vcsulf_old)
      call oracle_close()
    end do
  end do
end program run_sorgam
