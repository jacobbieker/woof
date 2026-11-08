program run_wrf_hydro_diffusive_oracle
  ! Drive WRF-Hydro's diffusive-wave channel routing (the channel_option = 3
  ! branch of drive_CHANNEL, cut byte for byte into diffusive_harness.F90)
  ! for eight model steps of DT = 300 s over a gridded node network: a main
  ! stem of 12 nodes ending in a pour point (TYPEL = 1) and two tributaries
  ! of 4 nodes joining it, on a 14 x 10 routing grid.  An adverse reach
  ! exercises DIFFUSION's minimum-slope guard, dry nodes the retention depth,
  ! and a storm pulse in step 3 the Courant halving of DTCT.  No lakes, no
  ! groundwater-channel exchange.  Every input and every output word, and
  ! DTCT after each step, are dumped.
  use iso_fortran_env, only: int64
  use diffusive_harness
  implicit none
  integer, parameter :: ixrt = 14, jxrt = 10, nl = 20, nstep = 8
  real, dimension(ixrt, jxrt) :: qstrmvolrt, qinflowbase, lakeinflort, qsubrt
  integer, dimension(ixrt, jxrt) :: ch_netrt, lake_mskrt
  integer(kind=int64), dimension(ixrt, jxrt) :: ch_netlnk
  integer, dimension(nl) :: chanxi, chanyj, typel
  real, dimension(nl) :: node_area, zelev, chanlen, mannn, so, chsslp, bw, hlink, cvol, qlateral
  integer(kind=int64), dimension(nl) :: from_node, to_node
  real :: qlink(nl, 2), dt, dtct, dtrt_ch, retdep, z, h, pulse
  integer :: i, k, u, un, uf, it
  character(len=1024) :: dir

  call get_command_argument(1, dir)
  ch_netrt = -9999; lake_mskrt = -9999; ch_netlnk = -9999
  ! main stem: nodes 1..12 at (2..13, 5), flowing east; node 12 is the pour point
  do i = 1, 12
    chanxi(i) = i + 1; chanyj(i) = 5
    to_node(i) = i + 1
  end do
  to_node(12) = 0
  ! tributary A: nodes 13..16 at (5, 1..4), flowing south into node 4 (5,5)
  do i = 13, 16
    chanxi(i) = 5; chanyj(i) = i - 12
    to_node(i) = i + 1
  end do
  to_node(16) = 4
  ! tributary B: nodes 17..20 at (9, 9..6), flowing north into node 8 (9,5)
  do i = 17, 20
    chanxi(i) = 9; chanyj(i) = 26 - i
    to_node(i) = i + 1
  end do
  to_node(20) = 8
  do i = 1, nl
    from_node(i) = i
    typel(i) = 0
    ch_netrt(chanxi(i), chanyj(i)) = 0
    ch_netlnk(chanxi(i), chanyj(i)) = i
    node_area(i) = 250. * 250. * (1. + 0.05 * frac(i, 41))
    chanlen(i) = 250. + 100. * frac(i * 3, 43)
    mannn(i) = 0.04 + 0.03 * frac(i * 5, 47)
    bw(i) = 2. + 6. * frac(i * 7, 53)
    chsslp(i) = 1. + 0.5 * frac(i * 11, 59)
    so(i) = 0.001 + 0.009 * frac(i * 13, 61)
  end do
  typel(12) = 1
  ! bed elevations: falling downstream, with an adverse step at node 6
  do i = 1, 12
    zelev(i) = 120. - 1.1 * real(i)
  end do
  zelev(6) = zelev(5) + 0.2
  do i = 13, 16
    zelev(i) = zelev(4) + 1.5 * real(17 - i)
  end do
  do i = 17, 20
    zelev(i) = zelev(8) + 1.2 * real(21 - i)
  end do
  ! initial depths (some below the retention depth) and the volumes they hold
  retdep = 0.001
  do i = 1, nl
    h = 0.4 * frac(i * 17, 67)
    if (mod(i, 5) == 0) h = 0.0005
    hlink(i) = h
    z = 1.0 / chsslp(i)
    cvol(i) = (bw(i) + z * h) * h * chanlen(i)
  end do
  qlink = 0.
  dt = 300.; dtrt_ch = 10.; dtct = 10.

  open(newunit=un, file=trim(dir)//'/diffusive-network.csv', status='replace')
  write(un, '(A)') 'node,chanxi,chanyj,from_node,to_node,typel,node_area,zelev,chanlen,mannn,so,chsslp,bw,hlink0,cvol0'
  do i = 1, nl
    write(un, '(6(I0,","),ES16.8E2,8(",",ES16.8E2))') i, chanxi(i), chanyj(i), from_node(i), to_node(i), &
        typel(i), node_area(i), zelev(i), chanlen(i), mannn(i), so(i), chsslp(i), bw(i), hlink(i), cvol(i)
  end do
  close(un)
  open(newunit=uf, file=trim(dir)//'/diffusive-forcing.csv', status='replace')
  write(uf, '(A)') 'step,node,qstrmvolrt,qinflowbase'
  open(newunit=u, file=trim(dir)//'/diffusive-steps.csv', status='replace')
  write(u, '(A)') 'step,node,dtct,hlink,cvol,qlink1,qlateral'
  qinflowbase = 0.; lakeinflort = 0.; qsubrt = 0.; qstrmvolrt = 0.
  do it = 1, nstep
    pulse = merge(60., 1., it == 3)
    do i = 1, nl
      qstrmvolrt(chanxi(i), chanyj(i)) = pulse * frac(i * 19 + it * 7, 71)
      qinflowbase(chanxi(i), chanyj(i)) = 0.05 * frac(i * 23 + it, 73)
      write(uf, '(2(I0,","),ES16.8E2,",",ES16.8E2)') it, i, qstrmvolrt(chanxi(i), chanyj(i)), &
          qinflowbase(chanxi(i), chanyj(i))
    end do
    call diffusive_route(ixrt, jxrt, nl, dt, dtct, dtrt_ch, retdep, qstrmvolrt, qinflowbase, &
        lakeinflort, qsubrt, ch_netrt, lake_mskrt, ch_netlnk, chanxi, chanyj, node_area, &
        from_node, to_node, typel, zelev, chanlen, mannn, so, chsslp, bw, qlink, hlink, cvol, qlateral)
    do i = 1, nl
      write(u, '(2(I0,","),ES16.8E2,4(",",ES16.8E2))') it, i, dtct, hlink(i), cvol(i), qlink(i, 1), &
          qlateral(i)
    end do
  end do
  close(u); close(uf)

contains

  real function frac(m, p)
    integer, intent(in) :: m, p
    frac = real(mod(m * p * 7919, 1009)) / 1009.
  end function frac

end program run_wrf_hydro_diffusive_oracle
