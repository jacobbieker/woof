program run_wrf_hydro_channel_oracle
  ! Drive the reach branch of WRF-Hydro's drive_CHANNEL with channel_option 2
  ! (Muskingum-Cunge, four calls) and then 1 (Muskingum, two calls) over a
  ! 12-reach dendritic network fed by 60 channel pixels on a 20 x 16 routing
  ! grid, one pixel in a lake, and dump every input and output word as CSV.
  ! Reach geometry spans a plain trapezoid, a compound channel that floods
  ! (TwCC, nCC > 0), Bw = Tw, Bw > Tw, a zero side slope (Cs = 0), channel
  ! loss (ChannK > 0) and a reservoir reach (TYPEL = 1, which the reach branch
  ! leaves alone).  Every reach receives lateral inflow at every call:
  ! SUBMUSKINGCUNGE reads WPk and D uninitialised when a reach has no water.
  use channel_harness
  implicit none

  integer, parameter :: ixrt = 20, jxrt = 16, nlinks = 60, nlinksl = 12
  real, parameter :: dt = 300.0, dtrt_ch = 60.0
  real, dimension(ixrt, jxrt) :: qstrmvolrt, qinflowbase, lakeinflort
  integer, dimension(ixrt, jxrt) :: lake_mskrt
  integer(kind=int64), dimension(ixrt, jxrt) :: ch_lnkrt_sl
  integer, dimension(nlinks) :: chanxi, chanyj, typel
  real, dimension(nlinks) :: node_area, musk, musx, so, chanlen, mannn, chsslp, bw, tw
  real, dimension(nlinks) :: tw_cc, n_cc, channk, hlink, qlateral, velocity, qloss
  integer(kind=int64), dimension(nlinks) :: linkid, to_node
  real :: qlink(nlinks, 2)
  integer :: parent(nlinksl) = (/ 3, 3, 6, 5, 6, 9, 8, 9, 12, 11, 12, 0 /)
  integer :: k, jj, n, u, v, opt, call_no
  character(len=1024) :: dir

  call get_command_argument(1, dir)
  if (len_trim(dir) == 0) error stop 'usage: run_channel OUTDIR'

  linkid = 0; to_node = 0; typel = 0
  musk = 0.; musx = 0.; so = 0.; chanlen = 0.; mannn = 0.; chsslp = 0.
  bw = 0.; tw = 0.; tw_cc = 0.; n_cc = 0.; channk = 0.; hlink = 0.
  do k = 1, nlinksl
    linkid(k) = 1000 + k
    if (parent(k) > 0) to_node(k) = 1000 + parent(k)
    so(k) = 0.001 + 0.02 * frac(k, 7)
    chanlen(k) = 800.0 + 2000.0 * frac(k, 11)
    mannn(k) = 0.03 + 0.05 * frac(k, 13)
    chsslp(k) = 0.5 + 2.0 * frac(k, 17)
    bw(k) = 2.0 + 10.0 * frac(k, 19)
    tw(k) = bw(k) + 4.0 + 6.0 * frac(k, 23)
    musk(k) = 0.5 + 2.0 * frac(k, 29)
    musx(k) = 0.1 + 0.3 * frac(k, 31)
    hlink(k) = 0.3 * frac(k, 37)
  end do
  tw_cc(4) = 3.0 * tw(4); n_cc(4) = 0.1; tw(4) = bw(4) + 0.4   ! floods its banks
  tw(5) = bw(5)                                               ! Bw = Tw
  tw(7) = bw(7) - 1.0                                         ! Bw > Tw
  chsslp(8) = 0.0                                             ! Cs = 0
  channk(2) = 1.0e-5; channk(9) = 4.0e-6                      ! channel loss
  typel(10) = 1                                               ! reservoir reach

  ch_lnkrt_sl = 0
  lake_mskrt = -9999
  do jj = 1, nlinks
    chanxi(jj) = 1 + mod(jj * 7, ixrt)
    chanyj(jj) = 1 + mod(jj * 3 + jj / ixrt, jxrt)
    node_area(jj) = 62500.0
  end do
  do jj = 1, nlinks
    if (jj == 37) then
      lake_mskrt(chanxi(jj), chanyj(jj)) = 6     ! a lake pixel feeding reach 6
    else
      ch_lnkrt_sl(chanxi(jj), chanyj(jj)) = 1 + mod(jj, nlinksl)
    end if
  end do

  open(newunit=u, file=trim(dir)//'/channel-network.csv', status='replace')
  write(u, '(A)') 'k,linkid,to_node,typel,so,chanlen,mannn,chsslp,bw,tw,tw_cc,n_cc,channk,musk,musx,hlink0'
  do k = 1, nlinksl
    write(u, '(I0,",",I0,",",I0,",",I0,12(",",ES24.16E3))') k, linkid(k), to_node(k), typel(k), &
        so(k), chanlen(k), mannn(k), chsslp(k), bw(k), tw(k), tw_cc(k), n_cc(k), channk(k), &
        musk(k), musx(k), hlink(k)
  end do
  close(u)
  open(newunit=u, file=trim(dir)//'/channel-pixels.csv', status='replace')
  write(u, '(A)') 'jj,chanxi,chanyj,link,lake,node_area'
  do jj = 1, nlinks
    write(u, '(5(I0,","),ES24.16E3)') jj, chanxi(jj), chanyj(jj), &
        ch_lnkrt_sl(chanxi(jj), chanyj(jj)), lake_mskrt(chanxi(jj), chanyj(jj)), node_area(jj)
  end do
  close(u)

  open(newunit=v, file=trim(dir)//'/channel-steps.csv', status='replace')
  write(v, '(A)') 'call,option,k,qlink1,qlink2,hlink,qlateral,velocity,qloss'
  open(newunit=u, file=trim(dir)//'/channel-forcing.csv', status='replace')
  write(u, '(A)') 'call,jj,qstrmvol,qinflowbase,lakeinflo'
  qlink = 0.0; qlateral = 0.0; velocity = 0.0; qloss = 0.0
  do k = 1, nlinksl
    qlink(k, :) = 0.5 * frac(k, 41)
  end do
  call_no = 0
  do n = 1, 6
    opt = merge(2, 1, n <= 4)
    call_no = call_no + 1
    qstrmvolrt = 0.0; qinflowbase = 0.0; lakeinflort = 0.0
    do jj = 1, nlinks
      qstrmvolrt(chanxi(jj), chanyj(jj)) = 0.5 + 8.0 * frac(jj * 5 + n * 3, 43)
      qinflowbase(chanxi(jj), chanyj(jj)) = 0.2 * frac(jj + n, 47)
      lakeinflort(chanxi(jj), chanyj(jj)) = 5.0 * frac(jj * 3 + n, 53)
      write(u, '(I0,",",I0,3(",",ES24.16E3))') n, jj, qstrmvolrt(chanxi(jj), chanyj(jj)), &
          qinflowbase(chanxi(jj), chanyj(jj)), lakeinflort(chanxi(jj), chanyj(jj))
    end do
    call reach_route(opt, ixrt, jxrt, nlinks, nlinksl, dt, dtrt_ch, &
        qstrmvolrt, qinflowbase, lakeinflort, lake_mskrt, ch_lnkrt_sl, chanxi, chanyj, &
        node_area, linkid, to_node, typel, musk, musx, so, chanlen, mannn, chsslp, &
        bw, tw, tw_cc, n_cc, channk, qlink, hlink, qlateral, velocity, qloss)
    do k = 1, nlinksl
      write(v, '(I0,",",I0,",",I0,6(",",ES24.16E3))') n, opt, k, qlink(k, 1), qlink(k, 2), &
          hlink(k), qlateral(k), velocity(k), qloss(k)
    end do
  end do
  close(u); close(v)

contains

  real function frac(m, p)
    integer, intent(in) :: m, p
    frac = real(mod(m * p * 7919, 1009)) / 1009.0
  end function frac

end program run_wrf_hydro_channel_oracle
