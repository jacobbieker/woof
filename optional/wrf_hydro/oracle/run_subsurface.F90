program run_wrf_hydro_subsurface_oracle
  ! Drive WRF-Hydro's subsurfaceRouting (FINDZWAT, ROUTE_SUBSURFACE1 and
  ! SUBSFC_RTNG) over a small routing grid for NSTEP calls, resetting QSUBRT
  ! after each as HYDRO_exe does (module_HYDRO_drv.F90:767), and dump every
  ! input and every output word as CSV.  Inputs are spread by integer hashes,
  ! never libm.  The terrain slopes and steepest-neighbour indices are formed
  ! from an elevation field the way the routing initialisation forms them
  ! (slope = drop / distance, steepest = largest slope); they are inputs here.
  use subsurface_harness
  implicit none

  integer, parameter :: xx = 20, yy = 16, nsoil = 4, nstep = 6
  real, parameter :: dx = 250.0, dt = 900.0
  type(subsurface_struct) :: sub
  type(subsurface_static_interface) :: static
  type(subsurface_input_interface) :: input
  type(subsurface_output_interface) :: output
  real, target :: slope(xx, yy, 8), dist(xx, yy, 9), sx(xx, yy), sy(xx, yy)
  integer, target :: sdir(xx, yy, 3)
  real, pointer :: infx(:,:)
  real :: elev(xx, yy), diag
  integer :: i, j, k, n, ni(8), nj(8), u, v, w, best
  character(len=1024) :: dir

  call get_command_argument(1, dir)
  if (len_trim(dir) == 0) error stop 'usage: run_subsurface OUTDIR'

  allocate(sub%state, sub%properties, sub%grid_transform)
  call sub%state%init(xx, yy)
  call sub%grid_transform%init(xx, yy, nsoil)
  allocate(sub%properties%lksatrt(xx, yy), sub%properties%zwattablrt(xx, yy))
  allocate(sub%properties%soldeprt(xx, yy), sub%properties%sldpth(nsoil))
  allocate(sub%properties%nexprt(xx, yy))
  sub%properties%surface_slope => slope
  sub%properties%max_surface_slope_index => sdir
  sub%properties%distance_to_neighbor => dist
  sub%properties%surface_slope_x => sx
  sub%properties%surface_slope_y => sy
  call static%init(xx, yy, nsoil, dt, 1)
  allocate(infx(xx, yy))
  call input%init(xx, yy, infx)

  sub%properties%sldpth = (/ 0.1, 0.3, 0.6, 1.0 /)
  sub%properties%zsoil = 0.0
  sub%properties%zsoil(1:nsoil) = (/ -0.1, -0.4, -1.0, -2.0 /)
  ! module_RT.F90:1652: SOLDEPRT = -ZSOIL(NSOIL)
  sub%properties%soldeprt = -1.0 * sub%properties%zsoil(nsoil)
  sub%properties%zwattablrt = 0.0
  sub%state%qsubrt = 0.0
  sub%state%qsubbdryrt = 0.0
  sub%state%qsubbdrytrt = 0.0

  diag = sqrt(2.0) * dx
  do j = 1, yy
    do i = 1, xx
      ! a tilted plane with a valley along j = 8 and hashed bumps
      elev(i, j) = 100.0 + 0.4 * real(xx - i) + 0.6 * abs(real(j - 8)) &
                   + 1.5 * frac(i * 31 + j * 17, 11)
      dist(i, j, :) = (/ dx, diag, dx, diag, dx, diag, dx, diag, dx * dx /)
      sub%properties%lksatrt(i, j) = 1.0e-4 + 9.0e-4 * frac(i * 13 + j * 7, 3)
      sub%properties%nexprt(i, j) = 1.0 + 2.0 * frac(i * 5 + j * 11, 19)
      infx(i, j) = 2.0 * frac(i * 29 + j * 3, 23)
      do k = 1, nsoil
        sub%grid_transform%smcmaxrt(i, j, k) = 0.40 + 0.08 * frac(i + 3 * j + 5 * k, 29)
        sub%grid_transform%smcrefrt(i, j, k) = 0.25 + 0.08 * frac(2 * i + j + 7 * k, 31)
        sub%grid_transform%smcwltrt(i, j, k) = 0.05 + 0.08 * frac(i + 2 * j + 3 * k, 37)
        ! from below wilting to exactly saturated
        sub%grid_transform%smcrt(i, j, k) = sub%grid_transform%smcwltrt(i, j, k) &
            + (sub%grid_transform%smcmaxrt(i, j, k) - sub%grid_transform%smcwltrt(i, j, k)) &
            * min(1.0, 1.25 * frac(3 * i + 5 * j + 11 * k, 41) + 0.1 * real(k))
        ! WLT + (MAX - WLT) can round one word above MAX; SUBSFC_RTNG stops
        ! the run on any SMC above SMCMAX, so clamp the input.
        sub%grid_transform%smcrt(i, j, k) = min(sub%grid_transform%smcrt(i, j, k), &
            sub%grid_transform%smcmaxrt(i, j, k))
      end do
    end do
  end do
  ! a column saturated to the surface and one with no saturated layer
  sub%grid_transform%smcrt(6, 8, :) = sub%grid_transform%smcmaxrt(6, 8, :)
  ! a saturated block at the foot of the valley with fast lateral flow into
  ! it, so inflow finds every layer full and joins the infiltration excess
  do j = 7, 9
    do i = 16, 18
      sub%grid_transform%smcrt(i, j, :) = sub%grid_transform%smcmaxrt(i, j, :)
    end do
  end do
  do j = 5, 11
    do i = 10, 18
      sub%properties%lksatrt(i, j) = 2.0e-2 * (1.0 + frac(i + j, 43))
    end do
  end do
  sub%grid_transform%smcrt(14, 4, :) = sub%grid_transform%smcwltrt(14, 4, :)

  do j = 1, yy
    do i = 1, xx
      ni = (/ i, i+1, i+1, i+1, i, i-1, i-1, i-1 /)
      nj = (/ j+1, j+1, j, j-1, j-1, j-1, j, j+1 /)
      best = 1
      do k = 1, 8
        if (ni(k) < 1 .or. ni(k) > xx .or. nj(k) < 1 .or. nj(k) > yy) then
          slope(i, j, k) = 0.0
        else
          slope(i, j, k) = (elev(i, j) - elev(ni(k), nj(k))) / dist(i, j, k)
        end if
        if (slope(i, j, k) > slope(i, j, best)) best = k
      end do
      sdir(i, j, :) = (/ ni(best), nj(best), best /)
      if (i == 1 .or. i == xx .or. j == 1 .or. j == yy) sdir(i, j, :) = (/ i, j, 1 /)
      sx(i, j) = 0.0; sy(i, j) = 0.0
    end do
  end do

  open(newunit=u, file=trim(dir)//'/subsurface-inputs.csv', status='replace')
  write(u, '(A)') 'i,j,lksat,nexp,soldep,infxs,' // &
      'smcmax1,smcmax2,smcmax3,smcmax4,smcref1,smcref2,smcref3,smcref4,' // &
      'smcwlt1,smcwlt2,smcwlt3,smcwlt4,smc1,smc2,smc3,smc4,' // &
      'slope1,slope2,slope3,slope4,slope5,slope6,slope7,slope8,' // &
      'dist1,dist2,dist3,dist4,dist5,dist6,dist7,dist8,dist9,sdir_i,sdir_j,sdir_k'
  do j = 1, yy
    do i = 1, xx
      write(u, '(I0,",",I0)', advance='no') i, j
      write(u, '(37(",",ES24.16E3))', advance='no') sub%properties%lksatrt(i, j), &
          sub%properties%nexprt(i, j), sub%properties%soldeprt(i, j), infx(i, j), &
          sub%grid_transform%smcmaxrt(i, j, :), sub%grid_transform%smcrefrt(i, j, :), &
          sub%grid_transform%smcwltrt(i, j, :), sub%grid_transform%smcrt(i, j, :), &
          slope(i, j, :), dist(i, j, :)
      write(u, '(3(",",I0))') sdir(i, j, :)
    end do
  end do
  close(u)

  open(newunit=v, file=trim(dir)//'/subsurface-steps.csv', status='replace')
  write(v, '(A)') 'step,i,j,smc1,smc2,smc3,smc4,zwattablrt,qsubrt,qsubbdryrt,infxs'
  open(newunit=w, file=trim(dir)//'/subsurface-totals.csv', status='replace')
  write(w, '(A)') 'step,qsubbdrytrt'
  do n = 1, nstep
    call subsurfaceRouting(sub, static, input, output)
    do j = 1, yy
      do i = 1, xx
        write(v, '(I0,",",I0,",",I0)', advance='no') n, i, j
        write(v, '(8(",",ES24.16E3))') sub%grid_transform%smcrt(i, j, :), &
            sub%properties%zwattablrt(i, j), sub%state%qsubrt(i, j), &
            sub%state%qsubbdryrt(i, j), infx(i, j)
      end do
    end do
    write(w, '(I0,",",ES24.16E3)') n, sub%state%qsubbdrytrt
    ! module_HYDRO_drv.F90:767, at the end of every HYDRO_exe
    sub%state%qsubrt = 0.0
  end do
  close(v); close(w)

contains

  real function frac(m, p)
    integer, intent(in) :: m, p
    frac = real(mod(m * p * 7919, 1009)) / 1009.0
  end function frac

end program run_wrf_hydro_subsurface_oracle
