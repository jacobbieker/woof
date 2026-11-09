! WOOF-OPTIN: wrf-hydro
! Oracle harness for lake_option 3: WRF 4.7.1's reservoir modules
! (hydro/Routing/Reservoirs: the level pool, the persistence/level-pool
! hybrid, the RFC-forecast reservoir, their readers and utilities) compiled
! whole and driven as LandRT_ini and drive_CHANNEL drive them: init with
! the lake's LAKEPARM properties, the reservoir parameter file, the start
! date and the timeslice/time-series directories; then run() each call with
! the previous inflow, inflow, lateral inflow, the lake elevation (RESHT,
! inout) and the routing period.  Part of the WRF-Hydro opt-in (../LICENSE);
! no WRF-Hydro source is copied here.

program reservoir_da_oracle
  use iso_fortran_env, only: int64
  use module_levelpool, only: levelpool
  use module_persistence_levelpool_hybrid, only: persistence_levelpool_hybrid
  use module_rfc_forecasts, only: rfc_forecasts
  implicit none
  integer :: n, ncalls, lookback, update_s, rfc_lookback, u, o, i, c, b, rtype
  character(len=19) :: start
  character(len=256) :: usgs_path, usace_path, rfc_path
  integer(kind=int64), allocatable :: ids(:)
  integer, allocatable :: types(:)
  real, allocatable :: p(:,:), resht(:), prev(:), inflow(:), lateral(:)
  type(levelpool), allocatable :: lp(:)
  type(persistence_levelpool_hybrid), allocatable :: hy(:)
  type(rfc_forecasts), allocatable :: rf(:)
  real :: period, outflow, assim
  integer :: dyn
  character(len=256) :: assim_file

  open(newunit=u, file="reservoir-inputs.txt", status="old")
  read(u, *) n, ncalls, lookback, update_s, rfc_lookback, start
  allocate(ids(n), types(n), p(10, n), resht(n), prev(n), inflow(n), lateral(n))
  allocate(lp(n), hy(n), rf(n))
  usgs_path = "usgs"; usace_path = "usace"; rfc_path = "rfc"
  do i = 1, n
     read(u, *) ids(i), types(i)
     do c = 1, 10
        read(u, *) b
        p(c, i) = transfer(b, 1.0)
     end do
     ! p: lake_area, weir_e, weir_c, weir_l, dam_l, orifice_e, orifice_c, orifice_a, max_h, ifd
     resht(i) = p(6, i) + ((p(9, i) - p(6, i)) * p(10, i))   ! LandRT_ini :743-746
     select case (types(i))
     case (2, 3)
        call hy(i)%init(resht(i), p(1, i), p(2, i), p(3, i), p(4, i), p(5, i), p(6, i), p(7, i), &
             p(8, i), p(9, i), p(10, i), ids(i), types(i), "reservoir_parm.nc", start, &
             usgs_path, usace_path, lookback, update_s)
     case (4, 5)
        call rf(i)%init(resht(i), p(1, i), p(2, i), p(3, i), p(4, i), p(5, i), p(6, i), p(7, i), &
             p(8, i), p(9, i), p(10, i), ids(i), types(i), "reservoir_parm.nc", start, &
             rfc_path, rfc_lookback)
     case default
        call lp(i)%init(resht(i), p(1, i), p(2, i), p(3, i), p(4, i), p(5, i), p(6, i), p(7, i), &
             p(8, i), p(9, i), ids(i), 1)
     end select
  end do
  open(newunit=o, file="reservoir-steps.csv", status="replace")
  write(o, '(A)') "call,lake,outflow,water_elevation,dynamic_type,assimilated"
  do c = 1, ncalls
     read(u, *) b
     period = transfer(b, 1.0)
     do i = 1, n
        read(u, *) b
        prev(i) = transfer(b, 1.0)
     end do
     do i = 1, n
        read(u, *) b
        inflow(i) = transfer(b, 1.0)
     end do
     do i = 1, n
        read(u, *) b
        lateral(i) = transfer(b, 1.0)
     end do
     do i = 1, n
        select case (types(i))
        case (2, 3)
           call hy(i)%run(prev(i), inflow(i), lateral(i), resht(i), outflow, period, dyn, assim, assim_file)
        case (4, 5)
           call rf(i)%run(prev(i), inflow(i), lateral(i), resht(i), outflow, period, dyn, assim, assim_file)
        case default
           call lp(i)%run(prev(i), inflow(i), lateral(i), resht(i), outflow, period, dyn, assim, assim_file)
        end select
        write(o, '(I0,",",I0,",",I0,",",I0,",",I0,",",I0)') c, i, transfer(outflow, 1), &
             transfer(resht(i), 1), dyn, transfer(assim, 1)
     end do
  end do
  close(o)
  close(u)
end program reservoir_da_oracle
