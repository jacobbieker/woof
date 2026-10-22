program run_fire_ros
  ! WRF v4.7.1 module_fr_fire_phys.F, byte-unmodified: init_fuel_cats (the
  ! fuel tables), set_fire_params (the per-cell Rothermel coefficients) and
  ! fire_ros (the normal-direction spread split into base, wind and slope
  ! terms), on a fire grid whose i index is the fuel category and whose j
  ! index is the surface fuel moisture.
  !
  !   run_fire_ros ROOT CONFIG
  !
  ! CONFIG is "namelist" when the working directory holds the pinned
  ! test/em_fire/namelist.fire (13 Anderson categories, WRF-SFIRE's
  ! namelist defaults) and "defaults" when it holds none (the module's
  ! compiled DATA: 13 Anderson + the Scott & Burgan 2005 categories through
  ! set_fire_params' ksb map).  Writes ROOT/CONFIG/ through oracle_io.
  !
  ! The sweep reaches every branch of set_fire_params and fire_ros:
  !   * every category, the no-fuel category 14 and numbers ksb maps to it;
  !   * moisture 0, low, high, and each moisture of extinction with its two
  !     float32 neighbours (at, just below, just above extinction);
  !   * calm, along, against, across and oblique wind, up to and beyond
  !     fire_ros' 30 m/s cap, so the 6 m/s excess branch runs;
  !   * flat, up-, down- and cross-slope terrain, positive and negative;
  !   * unit normals in four directions and one non-unit normal (WRF v4.7.1
  !     does not normalise propx/propy: scale = 1);
  !   * fire_advection = 0 and 1.
  !
  ! fire_ros does not assign ros_base on its chaparral branch (ichap = 1,
  ! category 4) although the excess test reads it.  The driver passes 0.0 in
  ! ros_base on every call, so on that branch the recorded ros_base is the
  ! driver's 0.0 and the excess is computed from it; the port fixes
  ! ros_base = 0 there, and PROVENANCE.md holds that word out.
  use oracle_io
  use module_fr_fire_util, only: fire_fmc_read, fire_advection
  use module_fr_fire_phys
  implicit none

  integer, parameter :: nmce = 7, nbase = 7, nm = nbase + 3 * nmce
  real, parameter :: mce(nmce) = (/ 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40 /)
  real, parameter :: base(nbase) = (/ 0.0, 0.02, 0.05, 0.08, 0.10, 0.5, 1.0 /)
  real, parameter :: wind(2, 8) = reshape((/ 0., 0., 3., 0., -3., 0., 0., -2., &
       4., 4., 10., -3., 25., 6., 40., 0. /), (/ 2, 8 /))
  real, parameter :: slope(2, 5) = reshape((/ 0., 0., 0.25, 0., -0.25, 0., &
       0.15, -0.4, 1.5, 0.8 /), (/ 2, 5 /))
  real, parameter :: normal(2, 5) = reshape((/ 1., 0., -1., 0., 0., 1., &
       0.6, -0.8, 0.3, 0.2 /), (/ 2, 5 /))
  integer, parameter :: cats_nl(16) = (/ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, &
       12, 13, 14, 15, 91 /)
  ! 204 (SB4) is left out: ksb(204) = 54 > the compiled nfuelcats = 53, and
  ! set_fire_params crashes on it ("fuel category out of bounds").
  integer, parameter :: cats_df(56) = (/ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, &
       12, 13, 14, 101, 102, 103, 104, 105, 106, 107, 108, 109, 121, 122, &
       123, 124, 141, 142, 143, 144, 145, 146, 147, 148, 149, 161, 162, 163, &
       164, 165, 181, 182, 183, 184, 185, 186, 187, 188, 189, 201, 202, 203, &
       50, 91, 99 /)

  character(len=1024) :: root, config
  integer, allocatable :: cats(:), iw(:), is(:), in(:)
  real, allocatable :: fm(:), nfuel_cat(:, :), fuel_time(:, :)
  real, allocatable :: rb(:, :, :, :), rw(:, :, :, :), rs(:, :, :, :)
  real, allocatable :: case_v(:, :)
  type(fire_params) :: fp
  integer :: nc, ncase, i, j, k, c, adv, n
  real :: ros_base, ros_wind, ros_slope

  call oracle_root(root)
  call get_command_argument(2, config)

  if (trim(config) == 'namelist') then
    allocate(cats(size(cats_nl))); cats = cats_nl
    ! every wind x slope x the first three normals
    ncase = 8 * 5 * 3
  else
    allocate(cats(size(cats_df))); cats = cats_df
    ncase = 4 * 3 * 3
  end if
  nc = size(cats)

  allocate(fm(nm))
  fm(1:nbase) = base
  do k = 1, nmce
    fm(nbase + 3 * k - 2) = nearest(mce(k), -1.0)
    fm(nbase + 3 * k - 1) = mce(k)
    fm(nbase + 3 * k) = nearest(mce(k), 1.0)
  end do

  call init_fuel_cats(.false.)

  allocate(nfuel_cat(nc, nm), fuel_time(nc, nm))
  allocate(fp%vx(nc, nm), fp%vy(nc, nm), fp%zsf(nc, nm), fp%dzdxf(nc, nm))
  allocate(fp%dzdyf(nc, nm), fp%bbb(nc, nm), fp%betafl(nc, nm), fp%phiwc(nc, nm))
  allocate(fp%r_0(nc, nm), fp%fgip(nc, nm), fp%ischap(nc, nm), fp%iboros(nc, nm))
  allocate(fp%fuel_time(nc, nm), fp%fmc_g(nc, nm))
  do j = 1, nm
    do i = 1, nc
      nfuel_cat(i, j) = real(cats(i))
      fp%fmc_g(i, j) = fm(j)
    end do
  end do
  fp%zsf = 0.

  ! fire_fmc_read = 0: keep the per-cell fmc_g instead of fuelmc_g.
  fire_fmc_read = 0
  call set_fire_params(1, nc, 1, nm, 1, nc, 1, nm, 1, nc, 1, nm, &
                       60., 60., 0, nfuel_cat, fuel_time, fp)

  ! the (wind, slope, normal) case list
  allocate(iw(ncase), is(ncase), in(ncase), case_v(6, ncase))
  n = 0
  if (trim(config) == 'namelist') then
    do c = 1, 3
      do k = 1, 5
        do i = 1, 8
          n = n + 1; iw(n) = i; is(n) = k; in(n) = c
        end do
      end do
    end do
  else
    do c = 1, 4, 3      ! normals 1 and 4
      do k = 1, 5, 2    ! slopes 1, 3, 5
        do i = 1, 8, 2  ! winds 1, 3, 5, 7
          n = n + 1; iw(n) = i; is(n) = k; in(n) = c
        end do
      end do
    end do
    do k = 1, 5, 2
      do i = 1, 8, 2
        n = n + 1; iw(n) = i; is(n) = k; in(n) = 5
      end do
    end do
  end if
  if (n /= ncase) error stop 'case count'
  do n = 1, ncase
    case_v(1:2, n) = wind(:, iw(n))
    case_v(3:4, n) = slope(:, is(n))
    case_v(5:6, n) = normal(:, in(n))
  end do

  allocate(rb(nc, nm, ncase, 2), rw(nc, nm, ncase, 2), rs(nc, nm, ncase, 2))
  do adv = 0, 1
    fire_advection = adv
    do n = 1, ncase
      fp%vx = case_v(1, n); fp%vy = case_v(2, n)
      fp%dzdxf = case_v(3, n); fp%dzdyf = case_v(4, n)
      do j = 1, nm
        do i = 1, nc
          ros_base = 0.0
          ros_wind = 0.0
          ros_slope = 0.0
          call fire_ros(ros_base, ros_wind, ros_slope, case_v(5, n), &
                        case_v(6, n), i, j, fp)
          rb(i, j, n, adv + 1) = ros_base
          rw(i, j, n, adv + 1) = ros_wind
          rs(i, j, n, adv + 1) = ros_slope
        end do
      end do
    end do
  end do

  call oracle_open(trim(config))
  call oracle_put('nfuel_cat', cats)
  call oracle_put('fmc_g', fm)
  call oracle_put('cases', case_v)
  call oracle_put('fuel_time', fuel_time)
  call oracle_put('fgip', fp%fgip)
  call oracle_put('ischap', fp%ischap)
  call oracle_put('betafl', fp%betafl)
  call oracle_put('bbb', fp%bbb)
  call oracle_put('phiwc', fp%phiwc)
  call oracle_put('r_0', fp%r_0)
  call oracle_put('iboros', fp%iboros)
  call oracle_put('ros_base', rb)
  call oracle_put('ros_wind', rw)
  call oracle_put('ros_slope', rs)
  call oracle_close()
end program run_fire_ros
