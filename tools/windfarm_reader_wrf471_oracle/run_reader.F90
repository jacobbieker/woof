! WRF v4.7.1's wind-farm input reads, run on one deck.
!
!   run_reader OUTROOT CASE MODE        (run from the deck's directory)
!
! MODE is 1 (windfarm_ij = 1, windturbines-ij.txt) or 0 (windturbines.txt).
! The four INCLUDEd blocks are module_wind_fitch.F's own statements, cut
! byte for byte by extract.py; everything else here is WRF's surrounding
! structure (the open, the allocation, the per-turbine loop) restated, plus
! the fixture writer.  The lat/lon a 0-mode row reads are kept per turbine
! here (WRF projects them at once; the placement is pinned elsewhere).
program run_reader
  use oracle_io
  implicit none
  integer, parameter :: MAXVALS = 100, MAXVALS2 = 100, max_domains = 1
  integer :: nt, id, i, k, nval, mode, n
  integer, dimension(:), allocatable :: nkind
  integer, dimension(:, :), allocatable :: ival, jval
  real, dimension(:), allocatable :: hubheight, diameter, stc, stc2, cutin, &
                                     cutout, npower, lats, lons
  real :: turbws(maxvals, maxvals2), turbtc(maxvals, maxvals2), &
          turbpw(maxvals, maxvals2)
  real :: lat, lon
  character*256 num, input
  character(len=1024) :: root, case, arg

  call oracle_root(root)
  call get_command_argument(2, case)
  call get_command_argument(3, arg, length=n)
  read(arg(1:n), *) mode
  id = 1

  if (mode .eq. 1) then
    open(70, file='windturbines-ij.txt', form='formatted', status='old')
  else
    open(70, file='windturbines.txt', form='formatted', status='old')
  end if
  include 'cut_count.inc'
  allocate (nkind(nt), ival(nt, max_domains), jval(nt, max_domains))
  allocate (hubheight(nt), stc(nt), stc2(nt), cutin(nt), cutout(nt), &
            diameter(nt), npower(nt), lats(nt), lons(nt))
  ival = -9999
  jval = -9999
  lats = 0.
  lons = 0.
  do k = 1, nt
    if (mode .eq. 1) then
      include 'cut_read_ij.inc'
    else
      include 'cut_read_ll.inc'
      lats(k) = lat
      lons(k) = lon
    end if
  end do
  close(70)
  hubheight = -1.; diameter = -1.; stc = -1.; npower = -1.
  cutin = -1.; cutout = -1.; stc2 = -1.
  include 'cut_tables.inc'

  call oracle_open(trim(case))
  call oracle_put('nt', nt)
  call oracle_put('nkind', nkind)
  if (mode .eq. 1) then
    call oracle_put('ival', ival(:, 1))
    call oracle_put('jval', jval(:, 1))
  else
    call oracle_put('lat', lats)
    call oracle_put('lon', lons)
  end if
  call oracle_put('hubheight', hubheight)
  call oracle_put('diameter', diameter)
  call oracle_put('stc', stc)
  call oracle_put('npower', npower)
  call oracle_put('cutin', cutin)
  call oracle_put('cutout', cutout)
  call oracle_put('stc2', stc2)
  call oracle_put('turbws', turbws)
  call oracle_put('turbtc', turbtc)
  call oracle_put('turbpw', turbpw)
  call oracle_close()
end program run_reader
