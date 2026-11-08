program run_wrf_hydro_grid_transform_oracle
  ! Drive WRF-Hydro's disaggregateDomain (LSM grid -> routing grid), change
  ! the routing-grid head and soil moisture the way a routing step would, then
  ! aggregateDomain (routing grid -> LSM grid), over a 6 x 5 LSM grid with
  ! AGGFACTRT = 4, and dump every input and output word as CSV.  Inputs come
  ! from integer hashes, never libm, plus named columns: partly frozen soil,
  ! a fully frozen layer with no liquid water (the "SMC = SICE" path), weights
  ! that over-fill layers (the excess redistribution and its spill into
  ! infiltration excess), a water cell off the channel network (lake mask),
  ! and ocean cells (ELRT < -9998).
  use grid_transform_harness
  implicit none

  integer, parameter :: ix = 6, jx = 5, agg = 4, nsoil = 4, iswater = 17
  integer, parameter :: ixrt = ix * agg, jxrt = jx * agg
  real, dimension(nsoil) :: sldpth = (/ 0.1, 0.3, 0.6, 1.0 /)
  real, dimension(ix, jx) :: area_lsm, smcmax1, smcref1, smcwlt1, lksat, nexp, infxsrt
  integer, dimension(ix, jx) :: vegtyp, soiltyp
  real, dimension(ix, jx, nsoil) :: smc, sh2ox, sice
  real, dimension(ixrt, jxrt, 9) :: dist
  real, dimension(ixrt, jxrt) :: lksatfac, elrt, impervfrac, infxswgt, infxsubrt, lksatrt, nexprt
  integer, dimension(ixrt, jxrt) :: ch_netrt, lake_mskrt, soiltyprt
  real, dimension(ixrt, jxrt, nsoil) :: sh2owgt, smcrefrt, smcmaxrt, smcwltrt, smcrt
  integer :: i, j, k, u
  character(len=1024) :: dir

  call get_command_argument(1, dir)
  if (len_trim(dir) == 0) error stop 'usage: run_grid_transform OUTDIR'

  do j = 1, jx
    do i = 1, ix
      area_lsm(i, j) = 1.0e6 * (1.0 + 0.01 * frac(i + 3 * j, 7))
      smcmax1(i, j) = 0.40 + 0.08 * frac(i * 3 + j, 11)
      smcref1(i, j) = 0.25 + 0.08 * frac(i + j * 5, 13)
      smcwlt1(i, j) = 0.05 + 0.06 * frac(i * 7 + j, 17)
      lksat(i, j) = 1.0e-4 * (1.0 + 5.0 * frac(i + 2 * j, 19))
      nexp(i, j) = 1.0 + 2.0 * frac(2 * i + j, 23)
      infxsrt(i, j) = 3.0 * frac(i * 5 + j * 3, 29)
      vegtyp(i, j) = 1 + mod(i * 3 + j, 12)
      soiltyp(i, j) = 1 + mod(i + j * 2, 9)
      do k = 1, nsoil
        sh2ox(i, j, k) = smcwlt1(i, j) + (smcmax1(i, j) - smcwlt1(i, j)) &
                         * (0.3 + 0.65 * frac(i + 2 * j + 5 * k, 31))
        smc(i, j, k) = sh2ox(i, j, k)
      end do
    end do
  end do
  ! partly frozen column, and a layer frozen solid with no liquid water
  smc(2, 2, 1:2) = sh2ox(2, 2, 1:2) + 0.12
  smc(5, 3, 1) = sh2ox(5, 3, 1) + 0.30
  sh2ox(3, 4, 1) = 0.0
  smc(3, 4, 1) = 0.33
  vegtyp(4, 2) = iswater

  do j = 1, jxrt
    do i = 1, ixrt
      dist(i, j, :) = (/ 250., 353.55339, 250., 353.55339, 250., 353.55339, 250., &
                         353.55339, 62500. /)
      lksatfac(i, j) = 1000.0 * (0.5 + frac(i + j, 37))
      elrt(i, j) = 100.0 + frac(i * 3 + j * 7, 41)
      impervfrac(i, j) = 0.0
      infxswgt(i, j) = 1.0 / real(agg * agg)
      ch_netrt(i, j) = merge(1, -9999, mod(i + j, 9) == 0)
      lake_mskrt(i, j) = -9999 + 0 * i
      lake_mskrt(i, j) = merge(5, -9999, mod(i * j, 13) == 0)
      do k = 1, nsoil
        sh2owgt(i, j, k) = 0.8 + 0.45 * frac(i + 2 * j + 3 * k, 43)
      end do
    end do
  end do
  elrt(1:3, 1:2) = -9999.0          ! ocean

  call disaggregateDomain(ix, jx, nsoil, ixrt, jxrt, agg, &
      sice, smc, sh2ox, infxsrt, area_lsm, smcmax1, smcref1, &
      smcwlt1, vegtyp, lksat, nexp, dist, infxswgt, &
      lksatfac, ch_netrt, sh2owgt, smcrefrt, infxsubrt, smcmaxrt, &
      smcwltrt, smcrt, lake_mskrt, lksatrt, nexprt, &
      sldpth, soiltyprt, soiltyp, elrt, iswater, impervfrac, 0)

  open(newunit=u, file=trim(dir)//'/disagg-lsm.csv', status='replace')
  write(u, '(A)') 'i,j,vegtyp,soiltyp,area,smcmax,smcref,smcwlt,lksat,nexp,infxsrt,' // &
      'smc1,smc2,smc3,smc4,sh2o1,sh2o2,sh2o3,sh2o4,sice1,sice2,sice3,sice4'
  do j = 1, jx
    do i = 1, ix
      write(u, '(4(I0,","),ES24.16E3,18(",",ES24.16E3))') i, j, vegtyp(i, j), soiltyp(i, j), &
          area_lsm(i, j), smcmax1(i, j), smcref1(i, j), smcwlt1(i, j), lksat(i, j), nexp(i, j), &
          infxsrt(i, j), smc(i, j, :), sh2ox(i, j, :), sice(i, j, :)
    end do
  end do
  close(u)
  open(newunit=u, file=trim(dir)//'/disagg-rt.csv', status='replace')
  write(u, '(A)') 'i,j,ch_netrt,lake_in,lake,soiltyprt,dist9,lksatfac,elrt,infxswgt,' // &
      'w1,w2,w3,w4,infxsubrt,lksatrt,nexprt,smcmax1,smcmax2,smcmax3,smcmax4,' // &
      'smcref1,smcref2,smcref3,smcref4,smcwlt1,smcwlt2,smcwlt3,smcwlt4,smc1,smc2,smc3,smc4'
  do j = 1, jxrt
    do i = 1, ixrt
      write(u, '(6(I0,","),ES24.16E3,26(",",ES24.16E3))') i, j, ch_netrt(i, j), &
          merge(5, -9999, mod(i * j, 13) == 0), lake_mskrt(i, j), soiltyprt(i, j), &
          dist(i, j, 9), lksatfac(i, j), elrt(i, j), infxswgt(i, j), sh2owgt(i, j, :), &
          infxsubrt(i, j), lksatrt(i, j), nexprt(i, j), smcmaxrt(i, j, :), smcrefrt(i, j, :), &
          smcwltrt(i, j, :), smcrt(i, j, :)
    end do
  end do
  close(u)

  ! --- what a routing step leaves behind, then aggregateDomain ------------
  rt_domain(1)%ix = ix; rt_domain(1)%jx = jx
  rt_domain(1)%ixrt = ixrt; rt_domain(1)%jxrt = jxrt
  nlst(1)%aggfactrt = agg; nlst(1)%nsoil = nsoil
  allocate(rt_domain(1)%sh2oaggrt(nsoil))
  rt_domain(1)%infxswgt = infxswgt
  rt_domain(1)%sh2owgt = sh2owgt
  rt_domain(1)%sh2ox = sh2ox
  rt_domain(1)%sice = sice
  rt_domain(1)%smc = smc
  rt_domain(1)%overland%properties%distance_to_neighbor = dist
  rt_domain(1)%subsurface%properties%sldpth = sldpth
  rt_domain(1)%subsurface%grid_transform%smcmaxrt = smcmaxrt
  rt_domain(1)%subsurface%grid_transform%smcrt = smcrt
  allocate(rt_domain(1)%overland%control%surface_water_head_routing(ixrt, jxrt))
  allocate(rt_domain(1)%overland%control%surface_water_head_lsm(ix, jx))
  do j = 1, jxrt
    do i = 1, ixrt
      ! a head on most cells, none on whole LSM cells (the LSMVOL = 0 arm)
      rt_domain(1)%overland%control%surface_water_head_routing(i, j) = &
          merge(0.0, 20.0 * frac(i * 7 + j * 11, 47), (i - 1) / agg == 4 .and. (j - 1) / agg == 3)
      do k = 1, nsoil
        rt_domain(1)%subsurface%grid_transform%smcrt(i, j, k) = &
            max(0.0, min(smcmaxrt(i, j, k), smcrt(i, j, k) * (0.9 + 0.2 * frac(i + j + k, 53))))
      end do
    end do
  end do
  call aggregateDomain(1)

  open(newunit=u, file=trim(dir)//'/agg-rt.csv', status='replace')
  write(u, '(A)') 'i,j,head,smc1,smc2,smc3,smc4,infxswgt,w1,w2,w3,w4'
  do j = 1, jxrt
    do i = 1, ixrt
      write(u, '(2(I0,","),ES24.16E3,9(",",ES24.16E3))') i, j, &
          rt_domain(1)%overland%control%surface_water_head_routing(i, j), &
          rt_domain(1)%subsurface%grid_transform%smcrt(i, j, :), &
          rt_domain(1)%infxswgt(i, j), rt_domain(1)%sh2owgt(i, j, :)
    end do
  end do
  close(u)
  open(newunit=u, file=trim(dir)//'/agg-lsm.csv', status='replace')
  write(u, '(A)') 'i,j,head_lsm,sh2o1,sh2o2,sh2o3,sh2o4,smc1,smc2,smc3,smc4'
  do j = 1, jx
    do i = 1, ix
      write(u, '(2(I0,","),ES24.16E3,8(",",ES24.16E3))') i, j, &
          rt_domain(1)%overland%control%surface_water_head_lsm(i, j), &
          rt_domain(1)%sh2ox(i, j, :), rt_domain(1)%smc(i, j, :)
    end do
  end do
  close(u)

contains

  real function frac(m, p)
    integer, intent(in) :: m, p
    frac = real(mod(m * p * 7919, 1009)) / 1009.0
  end function frac

end program run_wrf_hydro_grid_transform_oracle
