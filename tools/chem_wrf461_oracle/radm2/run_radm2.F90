! RADM2 gas-phase chemistry (radm_driver, chem/module_radm.F) on a 6 x 2
! column grid, 24 levels, dumped after every call.
!
! Columns cover: night (all photolysis zero), weak/normal/strong sun, clean
! and polluted air, NO and other species at zero (the 1e-6 and epsilc input
! floors), RH under 0.1 and over 0.95 (the RH clamp) and over 0.7 (the
! heterogeneous N2O5 switch, rk(137)), cold and warm columns.
!
! Two schedules: dtstep = 120 s at curr_secs = 0, 120, 240 (the first call has
! xtime - xtimin = 0, so dtcmax is 0 and every substep is dtcmin = 0.5 min /
! 60), and dtstep = 600 s at curr_secs = 3600, 4200 (dt60 = 10 > dtcmax = 5,
! several adaptive substeps).  The species and vdrog3 carry across calls in
! each schedule, as in WRF.
program run_radm2
  use oracle_io
  use module_state_description
  use module_configure
  use module_data_sorgam, only: ldrog
  use module_radm, only: radm_driver
  implicit none
  integer, parameter :: nx = 6, ny = 2, nz = 24, nj = 21
  real :: chem(nx, nz, ny, num_chem), moist(nx, nz, ny, num_moist)
  real :: t_phy(nx, nz, ny), p_phy(nx, nz, ny), rho(nx, nz, ny), dz8w(nx, nz, ny)
  real :: z(nx, nz, ny), z_at_w(nx, nz, ny), t8w(nx, nz, ny), p8w(nx, nz, ny)
  real :: ph(nx, nz, ny, nj), vdrog3(nx, nz, ny, ldrog)
  real :: base(num_chem), jbase(nj), dtstep, gmt, poll, sun, zk
  real, parameter :: polls(12) = (/ 1., 10., .1, 1., 3., 1., 1., .3, 1., 30., 1., 1. /)
  real, parameter :: suns(12) = (/ 1., 1., 1., 0., .3, 1.5, 1., 1., 0., 1., .05, 1. /)
  real, parameter :: qvs(12) = (/ 1.e-2, 5.e-3, 1.e-4, 2.e-2, 1.5e-2, 8.e-3, &
                                  1.2e-2, 2.e-3, 2.5e-2, 6.e-3, 9.e-3, 4.e-3 /)
  real(kind=8) :: curr_secs
  integer :: i, j, k, n, c, sched, step, nstep
  type(grid_config_rec_type) :: config_flags
  character(len=1024) :: root
  character(len=32) :: name

  call oracle_root(root)
  ! ppmv, package order p_so2 .. p_ho2 (index 1 is the dummy row)
  base = 0.
  base(p_so2) = 5.e-3;  base(p_sulf) = 1.e-4; base(p_no2) = 1.e-2
  base(p_no) = 5.e-3;   base(p_o3) = 4.e-2;   base(p_hno3) = 1.e-3
  base(p_h2o2) = 1.e-3; base(p_ald) = 1.e-3;  base(p_hcho) = 2.e-3
  base(p_op1) = 5.e-4;  base(p_op2) = 1.e-4;  base(p_paa) = 1.e-4
  base(p_ora1) = 1.e-3; base(p_ora2) = 1.e-3; base(p_nh3) = 5.e-3
  base(p_n2o5) = 1.e-5; base(p_no3) = 1.e-6;  base(p_pan) = 5.e-4
  base(p_hc3) = 1.e-2;  base(p_hc5) = 3.e-3;  base(p_hc8) = 1.e-3
  base(p_eth) = 2.e-3;  base(p_co) = 1.5e-1;  base(p_ol2) = 2.e-3
  base(p_olt) = 5.e-4;  base(p_oli) = 3.e-4;  base(p_tol) = 2.e-3
  base(p_xyl) = 1.e-3;  base(p_aco3) = 1.e-6; base(p_tpan) = 1.e-5
  base(p_hono) = 1.e-4; base(p_hno4) = 1.e-5; base(p_ket) = 1.e-3
  base(p_gly) = 1.e-4;  base(p_mgly) = 1.e-4; base(p_dcb) = 1.e-4
  base(p_onit) = 1.e-4; base(p_csl) = 1.e-5;  base(p_iso) = 1.e-3
  base(p_hcl) = 1.e-4;  base(p_ho) = 1.e-7;   base(p_ho2) = 1.e-5
  base(p_nu0) = 1.e3
  ! 1/min, radm_driver's rj order is NOT this order; these are the ph_ fields
  ! in the argument order of radm_driver (o31d, o33p, no2, no3o2, no3o, hno2,
  ! hno3, hno4, h2o2, ch2or, ch2om, ch3cho, ch3coch3, ch3coc2h5, hcocho,
  ! ch3cocho, hcochest, ch3o2h, ch3coo2h, ch3ono2, hcochob)
  jbase = (/ 1.8e-3, 2.4e-2, 5.e-1, 1.2, 1.e1, 1.e-1, 3.6e-5, 3.e-4, 4.2e-4, &
             1.8e-3, 2.4e-3, 3.e-4, 6.e-6, 6.e-5, 3.e-3, 6.e-3, 1.e-3, 3.e-4, &
             2.e-4, 4.e-5, 2.e-3 /)

  do sched = 1, 2
    do j = 1, ny
      do i = 1, nx
        c = i + (j - 1) * nx
        poll = polls(c)
        sun = suns(c)
        do k = 1, nz
          zk = 50. + 400. * real(k - 1)
          z(i, k, j) = zk
          z_at_w(i, k, j) = zk - 25.
          dz8w(i, k, j) = 400.
          t_phy(i, k, j) = 300. - 6.5e-3 * zk + 4. * real(i - 3) - 8. * real(j - 1)
          t8w(i, k, j) = t_phy(i, k, j)
          p_phy(i, k, j) = 1.e5 * exp(-zk / 7400.)
          p8w(i, k, j) = p_phy(i, k, j)
          rho(i, k, j) = p_phy(i, k, j) / (287. * t_phy(i, k, j))
          moist(i, k, j, :) = 0.
          moist(i, k, j, p_qv) = qvs(c) * exp(-zk / 2500.)
          moist(i, k, j, p_qc) = 1.e-4
          do n = 1, num_chem
            chem(i, k, j, n) = base(n) * poll * exp(-zk / 3000.)
          end do
          chem(i, k, j, p_o3) = base(p_o3) * (1. + zk / 4000.)
          chem(i, k, j, p_co) = base(p_co) * (1. + .5 * poll * exp(-zk / 3000.))
          do n = 1, nj
            ph(i, k, j, n) = jbase(n) * sun * (1. + .02 * real(k))
          end do
        end do
        if (c == 7) chem(i, :, j, p_no) = 0.
        if (c == 8) then
          chem(i, :, j, p_hono) = 0.; chem(i, :, j, p_iso) = 0.
          chem(i, :, j, p_ho) = 0.;   chem(i, :, j, p_dcb) = 0.
        end if
        if (c == 11) chem(i, 1:6, j, p_no2) = 1.e-20
      end do
    end do
    vdrog3 = 0.
    gmt = 12.
    if (sched == 1) then
      dtstep = 120.; nstep = 3
    else
      dtstep = 600.; nstep = 2
    end if
    write(name, '(A,I0,A)') 'sched', sched, '_init'
    call oracle_open(trim(name))
    call oracle_put('chem', chem)
    call oracle_put('t_phy', t_phy); call oracle_put('p_phy', p_phy)
    call oracle_put('qv', moist(:, :, :, p_qv))
    call oracle_put('ph', ph)
    call oracle_put('dtstep', dtstep); call oracle_put('gmt', gmt)
    call oracle_close()
    do step = 1, nstep
      if (sched == 1) then
        curr_secs = real(step - 1, 8) * 120._8
      else
        curr_secs = 3600._8 + real(step - 1, 8) * 600._8
      end if
      call radm_driver(1, curr_secs, dtstep, config_flags, gmt, 1, t_phy, moist, &
           p8w, t8w, p_phy, chem, rho, dz8w, z, z_at_w, vdrog3,                  &
           ph(:,:,:,1), ph(:,:,:,2), ph(:,:,:,3), ph(:,:,:,4), ph(:,:,:,5),      &
           ph(:,:,:,6), ph(:,:,:,7), ph(:,:,:,8), ph(:,:,:,9), ph(:,:,:,10),     &
           ph(:,:,:,11), ph(:,:,:,12), ph(:,:,:,13), ph(:,:,:,14), ph(:,:,:,15), &
           ph(:,:,:,16), ph(:,:,:,17), ph(:,:,:,18), ph(:,:,:,19), ph(:,:,:,20), &
           ph(:,:,:,21),                                                         &
           1, nx, 1, ny, 1, nz, 1, nx, 1, ny, 1, nz, 1, nx, 1, ny, 1, nz)
      write(name, '(A,I0,A,I0)') 'sched', sched, '_step', step
      call oracle_open(trim(name))
      call oracle_put('curr_secs_hi', real(curr_secs))
      call oracle_put('chem', chem)
      call oracle_put('vdrog3', vdrog3)
      call oracle_close()
    end do
  end do
end program run_radm2
