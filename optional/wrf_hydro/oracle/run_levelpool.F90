program run_wrf_hydro_levelpool_oracle
  ! Drive WRF-Hydro's level-pool reservoir, LEVELPOOL_PHYSICS through
  ! run_levelpool_reservoir (hydro/Routing/Reservoirs/Level_Pool/
  ! module_levelpool.F90, compiled whole with module_reservoir.F90 and the
  ! level-pool property and state modules), for 240 reservoirs whose pools
  ! sit in the dead pool, on the orifice, over the weir and over the dam, each
  ! routed for six periods with changing inflows.  Every input and output
  ! word is dumped.
  use iso_fortran_env, only: int64
  use module_reservoir, only: reservoir_container
  use module_levelpool, only: levelpool
  implicit none
  integer, parameter :: nres = 240, nper = 6
  type(reservoir_container) :: res(nres)
  real :: h, area, we, wc, wl, dl, oe, oc, oa, maxh, qi0, qi1, ql, qo, dt, aval
  integer :: i, k, u, rtype
  integer(kind=int64) :: ln
  character(len=256) :: src
  character(len=1024) :: dir
  real :: h0(nres), ar(nres), wev(nres), wcv(nres), wlv(nres), dlv(nres), oev(nres), ocv(nres)
  real :: oav(nres), mhv(nres), qip(nres)

  call get_command_argument(1, dir)
  open(newunit=u, file=trim(dir)//'/levelpool-cases.csv', status='replace')
  write(u, '(A)') 'res,period,lake_area,weir_elev,weir_coef,weir_len,dam_len,orifice_elev,' // &
      'orifice_coef,orifice_area,max_depth,dt,qi0,qi1,ql,h_in,h_out,qo'
  do i = 1, nres
    oev(i) = 100. + 2. * frac(i, 41)
    wev(i) = oev(i) + 1. + 3. * frac(i * 3, 43)
    mhv(i) = wev(i) + 0.5 + 2. * frac(i * 5, 47)
    ! starting level: dead pool, orifice, weir, overtopping by quarters
    select case (mod(i, 4))
    case (0); h0(i) = oev(i) - 0.5 * frac(i * 7, 53)
    case (1); h0(i) = oev(i) + (wev(i) - oev(i)) * frac(i * 7, 53)
    case (2); h0(i) = wev(i) + (mhv(i) - wev(i)) * frac(i * 7, 53)
    case default; h0(i) = mhv(i) + 0.4 * frac(i * 7, 53)
    end select
    ar(i) = 0.3 + 2. * frac(i * 11, 59)
    if (mod(i, 37) == 0) ar(i) = 0.0
    wcv(i) = 0.3 + 0.3 * frac(i * 13, 61)
    wlv(i) = 5. + 40. * frac(i * 17, 67)
    dlv(i) = 20. + 200. * frac(i * 19, 71)
    ocv(i) = 0.05 + 0.15 * frac(i * 23, 73)
    oav(i) = 0.5 + 5. * frac(i * 29, 79)
    allocate(levelpool :: res(i)%ptr)
    ln = i
    select type (r => res(i)%ptr)
    type is (levelpool)
      call r%init(h0(i), ar(i), wev(i), wcv(i), wlv(i), dlv(i), oev(i), ocv(i), oav(i), mhv(i), ln)
    end select
    qip(i) = 50. * frac(i * 31, 83)
  end do
  do k = 1, nper
    dt = merge(300., 3600., mod(k, 2) == 1)
    do i = 1, nres
      qi0 = qip(i)
      qi1 = 80. * frac(i * 37 + k * 5, 89)
      ql = 10. * frac(i * 41 + k, 97)
      h = h0(i)
      call res(i)%ptr%run(qi0, qi1, ql, h, qo, dt, rtype, aval, src)
      write(u, '(2(I0,","),ES16.8E2,15(",",ES16.8E2))') i, k, ar(i), wev(i), wcv(i), wlv(i), dlv(i), &
          oev(i), ocv(i), oav(i), mhv(i), dt, qi0, qi1, ql, h0(i), h, qo
      h0(i) = h
      qip(i) = qi1
    end do
  end do
  close(u)

contains

  real function frac(m, p)
    integer, intent(in) :: m, p
    frac = real(mod(m * p * 7919, 1009)) / 1009.
  end function frac

end program run_wrf_hydro_levelpool_oracle
