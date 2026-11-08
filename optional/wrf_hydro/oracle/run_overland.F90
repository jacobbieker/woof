program run_wrf_hydro_overland_oracle
  ! Drive WRF-Hydro's OverlandRouting (rt_option = 1) for three calls of
  ! DT = 120 s at DTRT_TER = 20 s over a 20 x 16 routing grid, setting the
  ! infiltration excess before each call from the last head plus a hashed
  ! rain, and dump every input and output word as CSV.  Terrain slopes and
  ! steepest-neighbour indices come from an elevation field as in the routing
  ! initialisation (inputs here); a channel line, a lake and a dry patch
  ! exercise the retention, channel, lake and GETMAX8DIR arms.
  use overland_harness
  implicit none

  integer, parameter :: xx = 20, yy = 16, ncall = 3
  real, parameter :: dx = 250.0, dt = 120.0, dtrt = 20.0
  type(overland_struct) :: ov
  real :: elev(xx, yy), diag, qx(xx, yy), qy(xx, yy)
  integer :: i, j, k, n, ni(8), nj(8), best, u, v, w
  character(len=1024) :: dir

  call get_command_argument(1, dir)
  if (len_trim(dir) == 0) error stop 'usage: run_overland OUTDIR'

  call ov%init(4, 4, xx, yy)
  if (.not. associated(ov%control%infiltration_excess)) allocate(ov%control%infiltration_excess(xx, yy))
  diag = sqrt(2.0) * dx
  do j = 1, yy
    do i = 1, xx
      elev(i, j) = 100.0 + 0.3 * real(xx - i) + 0.5 * abs(real(j - 8)) + 0.8 * frac(i * 31 + j * 17, 11)
      ov%properties%distance_to_neighbor(i, j, :) = (/ dx, diag, dx, diag, dx, diag, dx, diag, dx * dx /)
      ov%properties%roughness(i, j) = 0.03 + 0.17 * frac(i * 7 + j * 3, 13)
      ov%properties%retention_depth(i, j) = 1.0 + 2.0 * frac(i + 5 * j, 17)
      ov%properties%surface_slope_x(i, j) = 0.0
      ov%properties%surface_slope_y(i, j) = 0.0
      ov%streams_and_lakes%ch_netrt(i, j) = merge(0, -9999, j == 8 .and. i >= 6)
      ov%streams_and_lakes%lake_mask(i, j) = merge(3, -9999, i >= 3 .and. i <= 5 .and. j >= 11 .and. j <= 13)
      ov%streams_and_lakes%surface_water_to_channel(i, j) = 0.0
      ov%streams_and_lakes%surface_water_to_lake(i, j) = 0.0
      ov%control%boundary_flux(i, j) = 0.0
      ov%control%infiltration_excess(i, j) = 40.0 * frac(i * 13 + j * 19, 23)
    end do
  end do
  ! a dry patch below retention, so no water moves there
  ov%control%infiltration_excess(10:12, 3:4) = 0.5
  ! a pit: a cell lower than every neighbour, so its terrain slope is never
  ! positive and GETMAX8DIR must find (or fail to find) a head-driven outlet
  ! A deep pit (no outlet even for a large head) and a shallow one 1 cm below
  ! its lowest neighbour with a large head (an outlet the head makes).
  elev(15, 12) = elev(15, 12) - 3.0
  elev(9, 5) = minval(elev(8:10, 4:6), mask=reshape((/ .true., .true., .true., &
      .true., .false., .true., .true., .true., .true. /), (/ 3, 3 /))) - 0.01
  ov%control%infiltration_excess(9, 5) = 80.0
  ov%streams_and_lakes%accumulated_surface_water_to_channel = 0.0
  ov%streams_and_lakes%accumulated_surface_water_to_lake = 0.0
  ov%control%boundary_flux_total = 0.0
  do j = 1, yy
    do i = 1, xx
      ni = (/ i, i+1, i+1, i+1, i, i-1, i-1, i-1 /)
      nj = (/ j+1, j+1, j, j-1, j-1, j-1, j, j+1 /)
      best = 1
      do k = 1, 8
        if (ni(k) < 1 .or. ni(k) > xx .or. nj(k) < 1 .or. nj(k) > yy) then
          ov%properties%surface_slope(i, j, k) = 0.0
        else
          ov%properties%surface_slope(i, j, k) = (elev(i, j) - elev(ni(k), nj(k))) &
              / ov%properties%distance_to_neighbor(i, j, k)
        end if
        if (ov%properties%surface_slope(i, j, k) > ov%properties%surface_slope(i, j, best)) best = k
      end do
      ov%properties%max_surface_slope_index(i, j, :) = (/ ni(best), nj(best), best /)
      if (i == 1 .or. i == xx .or. j == 1 .or. j == yy) &
          ov%properties%max_surface_slope_index(i, j, :) = (/ i, j, 1 /)
    end do
  end do

  ! A 1.5 m mound on cell (16, 3)'s steepest terrain neighbour, so that
  ! neighbour's head reverses the slope and GETMAX8DIR finds another outlet.
  ov%control%infiltration_excess(ov%properties%max_surface_slope_index(16, 3, 1), &
                                 ov%properties%max_surface_slope_index(16, 3, 2)) = 1500.0
  ov%control%infiltration_excess(16, 3) = 30.0

  open(newunit=u, file=trim(dir)//'/overland-inputs.csv', status='replace')
  write(u, '(A)') 'i,j,ch_netrt,lake,roughness,retention,infxs,' // &
      'slope1,slope2,slope3,slope4,slope5,slope6,slope7,slope8,dist1,dist2,dist3,dist4,' // &
      'dist5,dist6,dist7,dist8,dist9,sdir_i,sdir_j,sdir_k'
  do j = 1, yy
    do i = 1, xx
      write(u, '(4(I0,","),ES24.16E3,19(",",ES24.16E3))', advance='no') i, j, &
          ov%streams_and_lakes%ch_netrt(i, j), ov%streams_and_lakes%lake_mask(i, j), &
          ov%properties%roughness(i, j), ov%properties%retention_depth(i, j), &
          ov%control%infiltration_excess(i, j), ov%properties%surface_slope(i, j, :), &
          ov%properties%distance_to_neighbor(i, j, :)
      write(u, '(3(",",I0))') ov%properties%max_surface_slope_index(i, j, :)
    end do
  end do
  close(u)

  open(newunit=v, file=trim(dir)//'/overland-steps.csv', status='replace')
  write(v, '(A)') 'call,i,j,infxs,head,retention,qx,qy,to_channel,to_lake,boundary_flux'
  open(newunit=w, file=trim(dir)//'/overland-totals.csv', status='replace')
  write(w, '(A)') 'call,acc_channel,acc_lake,boundary_flux_total'
  do n = 1, ncall
    if (n > 1) then
      do j = 1, yy
        do i = 1, xx
          ov%control%infiltration_excess(i, j) = ov%control%surface_water_head_routing(i, j) &
              + 6.0 * frac(i * 3 + j * 29 + n * 7, 31)
        end do
      end do
    end if
    qx = 0.0; qy = 0.0
    call OverlandRouting(ov, dt, dtrt, 1, xx, yy, qx, qy)
    do j = 1, yy
      do i = 1, xx
        write(v, '(3(I0,","),ES24.16E3,7(",",ES24.16E3))') n, i, j, &
            ov%control%infiltration_excess(i, j), ov%control%surface_water_head_routing(i, j), &
            ov%properties%retention_depth(i, j), qx(i, j), qy(i, j), &
            ov%streams_and_lakes%surface_water_to_channel(i, j), &
            ov%streams_and_lakes%surface_water_to_lake(i, j), ov%control%boundary_flux(i, j)
      end do
    end do
    write(w, '(I0,3(",",ES24.16E3))') n, ov%streams_and_lakes%accumulated_surface_water_to_channel, &
        ov%streams_and_lakes%accumulated_surface_water_to_lake, ov%control%boundary_flux_total
  end do
  close(v); close(w)

contains

  real function frac(m, p)
    integer, intent(in) :: m, p
    frac = real(mod(m * p * 7919, 1009)) / 1009.0
  end function frac

end program run_wrf_hydro_overland_oracle
