! WRF v4.7.1 wind-farm layout oracle: one deck directory in, fixtures out.
!
!   cd DECK && run_windfarm_layout OUTDIR
!
! DECK holds what a WRF run of the farm reads (windturbines.txt and its
! wind-turbine-N.tbl files, written by woof) plus two harness files:
!
!   grid.txt     map_proj truelat1 truelat2 stand_lon dx  (config_flags;
!                map_proj 1 Lambert, 2 polar, 3 Mercator)
!                its jts ids ide                           (tile bounds)
!                xlat(its,jts) xlong(its,jts)              (float32 words)
!   queries.txt  correction_factor nq, then nq lines "kt speedhub"
!
! The work is WRF's own statements, byte-unmodified, cut by extract.py:
! wf_place.inc (init_module_wind_fitch:399-477) per turbine, wf_tables.inc
! (:491-510), dragcof (:277-334) and wf_power1.inc (dragforce:190-191) per
! query.  This file only opens, counts (as :369-377 does), allocates (as
! :383-386 does), calls dragcof with dragforce's argument list (:184-186)
! and writes what came out.
program run_windfarm_layout
  use module_llxy
  use wf_fitch
  use oracle_io
  implicit none

  type config_t
    integer :: map_proj
    real :: truelat1, truelat2, stand_lon, dx
  end type config_t

  type(config_t) :: config_flags
  TYPE (PROJ_INFO) :: ts_proj
  CHARACTER*256 num,input,message_wind
  real lat,lon,ts_rx,ts_ry
  REAL :: known_lat, known_lon
  INTEGER i,j,nval,k,id
  integer :: its, jts, ids, ide, nq, q, kt
  real, allocatable :: xlat(:,:), xlong(:,:)
  real, allocatable :: rx(:), ry(:), rlat(:), rlon(:)
  real, allocatable :: qspeed(:), qthr(:), qpow(:), qtke(:), qp1(:)
  integer, allocatable :: qkt(:)
  real :: tkecof, powcof, thrcof, speedhub, area, power1, x0, y0, pdlon
  integer :: ntypes
  character(len=1024) :: root

  call oracle_root(root)
  id = 1

  open(71, file='grid.txt', form='formatted', status='old')
  read(71,*) config_flags%map_proj, config_flags%truelat1, &
             config_flags%truelat2, config_flags%stand_lon, config_flags%dx
  read(71,*) its, jts, ids, ide
  allocate(xlat(its:its, jts:jts), xlong(its:its, jts:jts))
  read(71,*) x0, y0
  xlat(its,jts) = x0
  xlong(its,jts) = y0
  close(71)

  open(70,file='windturbines.txt',form='formatted',status='old')
  nt=0
10 read(70,*,end=100)
  nt=nt+1
  goto 10
100 continue
  rewind (70)

  allocate (nkind(nt),ival(nt,max_domains),jval(nt,max_domains))
  allocate (hubheight(nt),stc(nt),stc2(nt),cutin(nt),cutout(nt),diameter(nt),npower(nt))
  ival=-9999
  jval=-9999
  allocate(rx(nt), ry(nt), rlat(nt), rlon(nt))

  do k=1,nt
    include 'wf_place.inc'
    rx(k) = ts_rx
    ry(k) = ts_ry
    rlat(k) = lat
    rlon(k) = lon
  enddo
  close(70)

  include 'wf_tables.inc'

  open(72, file='queries.txt', form='formatted', status='old')
  read(72,*) correction_factor, nq
  allocate(qkt(nq), qspeed(nq), qthr(nq), qpow(nq), qtke(nq), qp1(nq))
  do q = 1, nq
    read(72,*) kt, speedhub
    CALL dragcof(tkecof,powcof,thrcof,               &
                 speedhub,cutin(kt),cutout(kt),   &
                 npower(kt),diameter(kt),stc(kt),stc2(kt),nkind(kt))
    include 'wf_power1.inc'
    qkt(q) = kt
    qspeed(q) = speedhub
    qthr(q) = thrcof
    qpow(q) = powcof
    qtke(q) = tkecof
    qp1(q) = power1
  enddo
  close(72)

  call oracle_open('placement')
  call oracle_put('lat_lon_known', (/ xlat(its,jts), xlong(its,jts) /))
  call oracle_put('nkind', nkind)
  call oracle_put('lat', rlat)
  call oracle_put('lon', rlon)
  call oracle_put('ival', ival(:,1))
  call oracle_put('jval', jval(:,1))
  call oracle_put('ts_rx', rx)
  call oracle_put('ts_ry', ry)
  ! proj_info after the last map_set; dlon is set by set_merc alone
  ! (map_init leaves it undefined), so it is written as 0 otherwise
  pdlon = 0.
  if (config_flags%map_proj == PROJ_MERC) pdlon = ts_proj%dlon
  call oracle_put('proj', (/ ts_proj%cone, ts_proj%rebydx, ts_proj%rsw, &
                             ts_proj%polei, ts_proj%polej, ts_proj%hemi, &
                             pdlon, ts_proj%stdlon, ts_proj%truelat1, &
                             ts_proj%truelat2, ts_proj%lat1, ts_proj%lon1, &
                             ts_proj%knowni, ts_proj%knownj, ts_proj%dx /))
  call oracle_open('tables')
  call oracle_put('hubheight', hubheight)
  call oracle_put('diameter', diameter)
  call oracle_put('stc', stc)
  call oracle_put('stc2', stc2)
  call oracle_put('npower', npower)
  call oracle_put('cutin', cutin)
  call oracle_put('cutout', cutout)
  ! the types the deck uses (rows past them are WRF's zero fill)
  ntypes = maxval(nkind)
  call oracle_put('turbws', turbws(1:ntypes,:))
  call oracle_put('turbtc', turbtc(1:ntypes,:))
  call oracle_put('turbpw', turbpw(1:ntypes,:))
  call oracle_open('dragcof')
  call oracle_put('correction_factor', correction_factor)
  call oracle_put('kt', qkt)
  call oracle_put('speedhub', qspeed)
  call oracle_put('thrcof', qthr)
  call oracle_put('powcof', qpow)
  call oracle_put('tkecof', qtke)
  call oracle_put('power1', qp1)
  call oracle_close()
end program run_windfarm_layout
