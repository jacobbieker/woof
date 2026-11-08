program run_wrf_hydro_diffusive_oracle
  ! Drive WRF-Hydro's diffusive-wave channel routing (the channel_option = 3
  ! branch of drive_CHANNEL, cut byte for byte into diffusive_harness.F90)
  ! for eight model steps of DT = 300 s over a gridded node network: a main
  ! stem of 12 nodes ending in a pour point (TYPEL = 1) and two tributaries
  ! of 4 nodes joining it, on a 14 x 10 routing grid.  An adverse reach
  ! exercises DIFFUSION's minimum-slope guard, dry nodes the retention depth,
  ! and the first step the Courant halving of DTCT.  Tributary A runs through
  ! a level-pool lake: node 14 flows into it (TYPEL = 2), node 15 is a lake
  ! node fed by LAKEINFLORT, node 16 takes the lake's outflow (CH_NETRT = 1),
  ! and LAKENODE marks 14 to 16, so the lake sees face inflow, lateral
  ! inflow and its own level-pool outflow (orifice, weir and overtopping over
  ! the run).  Then four more steps with the groundwater-channel exchange
  ! (GWBASESWCRT = 3): a groundwater head under every node, above the water
  ! surface at some nodes (flux in) and below it at others (flux out, some
  ! clipped at -0.005 m3/s), with gwChanCondSw = 1 and, on the last step,
  ! 0.  Every input and every output word, and DTCT and the lake state after
  ! each step, are dumped.
  use iso_fortran_env, only: int64
  use diffusive_harness
  implicit none
  integer, parameter :: ixrt = 14, jxrt = 10, nl = 20, nstep = 8, nlakes = 1
  real, dimension(ixrt, jxrt) :: qstrmvolrt, qinflowbase, lakeinflort, qsubrt
  integer, dimension(ixrt, jxrt) :: ch_netrt, lake_mskrt
  integer(kind=int64), dimension(ixrt, jxrt) :: ch_netlnk
  integer, dimension(nl) :: chanxi, chanyj, typel
  real, dimension(nl) :: node_area, zelev, chanlen, mannn, so, chsslp, bw, hlink, cvol, qlateral
  integer(kind=int64), dimension(nl) :: from_node, to_node
  real :: qlink(nl, 2), dt, dtct, dtrt_ch, retdep, z, h, pulse
  integer(kind=int64), dimension(nl) :: lakenode
  real, dimension(nlakes) :: resht, qlakei, qlakeo, lake_area, weir_elev, weir_coef, weir_len
  real, dimension(nlakes) :: dam_len, orifice_elev, orifice_coef, orifice_area, max_depth
  integer :: i, k, u, un, uf, ul, it, ug, ugl, ugf, gwsw
  real, dimension(:,:), allocatable :: gwhead, qgw_chanrt
  real, parameter :: cond_in = 2.0e-5, cond_out = 4.0e-5
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
  ! the lake on tributary A
  typel(14) = 2
  lake_mskrt(chanxi(15), chanyj(15)) = 1
  ch_netrt(chanxi(15), chanyj(15)) = -9999
  ch_netrt(chanxi(16), chanyj(16)) = 1
  lakenode = 0
  lakenode(14) = 1; lakenode(15) = 1; lakenode(16) = 1
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
  lake_area = 0.05; weir_elev = zelev(4) + 6.5; weir_coef = 0.4; weir_len = 10.
  dam_len = 60.; orifice_elev = zelev(4) + 4.5; orifice_coef = 0.1; orifice_area = 1.
  max_depth = zelev(4) + 7.0
  resht = zelev(4) + 6.3; qlakei = 0.; qlakeo = 0.
  call init_lakes(nlakes, resht, lake_area, weir_elev, weir_coef, weir_len, dam_len, &
      orifice_elev, orifice_coef, orifice_area, max_depth)
  ! initial depths (some below the retention depth) and the volumes they hold
  retdep = 0.001
  do i = 1, nl
    h = 0.4 * frac(i * 17, 67)
    if (mod(i, 5) == 0 .and. i /= 15) h = 0.0005
    hlink(i) = h
    z = 1.0 / chsslp(i)
    cvol(i) = (bw(i) + z * h) * h * chanlen(i)
  end do
  qlink = 0.
  dt = 300.; dtrt_ch = 10.; dtct = 10.

  open(newunit=un, file=trim(dir)//'/diffusive-network.csv', status='replace')
  write(un, '(A)') 'node,chanxi,chanyj,from_node,to_node,typel,ch_netrt,lake_mskrt,lakenode,node_area,' // &
      'zelev,chanlen,mannn,so,chsslp,bw,hlink0,cvol0'
  do i = 1, nl
    write(un, '(9(I0,","),ES16.8E2,8(",",ES16.8E2))') i, chanxi(i), chanyj(i), from_node(i), to_node(i), &
        typel(i), ch_netrt(chanxi(i), chanyj(i)), lake_mskrt(chanxi(i), chanyj(i)), lakenode(i), &
        node_area(i), zelev(i), chanlen(i), mannn(i), so(i), chsslp(i), bw(i), hlink(i), cvol(i)
  end do
  close(un)
  open(newunit=uf, file=trim(dir)//'/diffusive-forcing.csv', status='replace')
  write(uf, '(A)') 'step,node,qstrmvolrt,qinflowbase,lakeinflort'
  open(newunit=u, file=trim(dir)//'/diffusive-steps.csv', status='replace')
  write(u, '(A)') 'step,node,dtct,hlink,cvol,qlink1,qlateral'
  open(newunit=ul, file=trim(dir)//'/diffusive-lakes.csv', status='replace')
  write(ul, '(A)') 'step,lake,lake_area,weir_elev,weir_coef,weir_len,dam_len,orifice_elev,' // &
      'orifice_coef,orifice_area,max_depth,resht,qlakei,qlakeo'
  do i = 1, nlakes
    write(ul, '(2(I0,","),ES16.8E2,11(",",ES16.8E2))') 0, i, lake_area(i), weir_elev(i), weir_coef(i), &
        weir_len(i), dam_len(i), orifice_elev(i), orifice_coef(i), orifice_area(i), max_depth(i), &
        resht(i), qlakei(i), qlakeo(i)
  end do
  qinflowbase = 0.; lakeinflort = 0.; qsubrt = 0.; qstrmvolrt = 0.
  allocate(gwhead(ixrt, jxrt), qgw_chanrt(ixrt, jxrt))
  gwhead = 0.; qgw_chanrt = 0.
  do it = 1, nstep
    pulse = merge(60., 1., it == 3)
    do i = 1, nl
      qstrmvolrt(chanxi(i), chanyj(i)) = pulse * frac(i * 19 + it * 7, 71)
      qinflowbase(chanxi(i), chanyj(i)) = 0.05 * frac(i * 23 + it, 73)
      lakeinflort(chanxi(i), chanyj(i)) = merge(40. * frac(it * 29, 79), 0., i == 15)
      write(uf, '(2(I0,","),ES16.8E2,2(",",ES16.8E2))') it, i, qstrmvolrt(chanxi(i), chanyj(i)), &
          qinflowbase(chanxi(i), chanyj(i)), lakeinflort(chanxi(i), chanyj(i))
    end do
    call diffusive_route(ixrt, jxrt, nl, dt, dtct, dtrt_ch, retdep, qstrmvolrt, qinflowbase, &
        lakeinflort, qsubrt, ch_netrt, lake_mskrt, ch_netlnk, chanxi, chanyj, node_area, &
        from_node, to_node, typel, zelev, chanlen, mannn, so, chsslp, bw, qlink, hlink, cvol, qlateral, &
        nlakes, lakenode, resht, qlakei, qlakeo, 0, 0, 0., 0., gwhead, qgw_chanrt)
    do i = 1, nl
      write(u, '(2(I0,","),ES16.8E2,4(",",ES16.8E2))') it, i, dtct, hlink(i), cvol(i), qlink(i, 1), &
          qlateral(i)
    end do
    do i = 1, nlakes
      write(ul, '(2(I0,","),ES16.8E2,11(",",ES16.8E2))') it, i, lake_area(i), weir_elev(i), weir_coef(i), &
          weir_len(i), dam_len(i), orifice_elev(i), orifice_coef(i), orifice_area(i), max_depth(i), &
          resht(i), qlakei(i), qlakeo(i)
    end do
  end do
  close(u); close(uf); close(ul)

  ! the groundwater-channel exchange, continuing from step 8
  do i = 1, nl
    gwhead(chanxi(i), chanyj(i)) = zelev(i) + hlink(i) - 0.35 + 0.7 * frac(i * 31, 83)
  end do
  open(newunit=ugf, file=trim(dir)//'/diffusive-gw-forcing.csv', status='replace')
  write(ugf, '(A)') 'step,node,qstrmvolrt,qinflowbase,lakeinflort,gwhead,gwchancondsw,cond_in,cond_out'
  open(newunit=ug, file=trim(dir)//'/diffusive-gw-steps.csv', status='replace')
  write(ug, '(A)') 'step,node,dtct,hlink,cvol,qlink1,qlateral,qgw_chanrt'
  open(newunit=ugl, file=trim(dir)//'/diffusive-gw-lakes.csv', status='replace')
  write(ugl, '(A)') 'step,lake,resht,qlakei,qlakeo'
  do it = nstep + 1, nstep + 4
    gwsw = merge(0, 1, it == nstep + 4)
    do i = 1, nl
      qstrmvolrt(chanxi(i), chanyj(i)) = frac(i * 19 + it * 7, 71)
      qinflowbase(chanxi(i), chanyj(i)) = 0.05 * frac(i * 23 + it, 73)
      lakeinflort(chanxi(i), chanyj(i)) = merge(40. * frac(it * 29, 79), 0., i == 15)
      write(ugf, '(2(I0,","),ES16.8E2,3(",",ES16.8E2),",",I0,2(",",ES16.8E2))') it, i, &
          qstrmvolrt(chanxi(i), chanyj(i)), qinflowbase(chanxi(i), chanyj(i)), &
          lakeinflort(chanxi(i), chanyj(i)), gwhead(chanxi(i), chanyj(i)), gwsw, cond_in, cond_out
    end do
    call diffusive_route(ixrt, jxrt, nl, dt, dtct, dtrt_ch, retdep, qstrmvolrt, qinflowbase, &
        lakeinflort, qsubrt, ch_netrt, lake_mskrt, ch_netlnk, chanxi, chanyj, node_area, &
        from_node, to_node, typel, zelev, chanlen, mannn, so, chsslp, bw, qlink, hlink, cvol, qlateral, &
        nlakes, lakenode, resht, qlakei, qlakeo, 3, gwsw, cond_in, cond_out, gwhead, qgw_chanrt)
    do i = 1, nl
      write(ug, '(2(I0,","),ES16.8E2,5(",",ES16.8E2))') it, i, dtct, hlink(i), cvol(i), qlink(i, 1), &
          qlateral(i), qgw_chanrt(chanxi(i), chanyj(i))
    end do
    do i = 1, nlakes
      write(ugl, '(2(I0,","),ES16.8E2,2(",",ES16.8E2))') it, i, resht(i), qlakei(i), qlakeo(i)
    end do
  end do
  close(ug); close(ugl); close(ugf)

contains

  real function frac(m, p)
    integer, intent(in) :: m, p
    frac = real(mod(m * p * 7919, 1009)) / 1009.
  end function frac

end program run_wrf_hydro_diffusive_oracle
