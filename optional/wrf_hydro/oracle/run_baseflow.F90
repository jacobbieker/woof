program run_wrf_hydro_baseflow_oracle
  ! Drive WRF-Hydro's simp_gw_buck over a 6 x 5 LSM grid with four basins
  ! and a 24 x 20 routing grid: four calls with the exponential bucket
  ! (GWBASESWCRT = 1; one basin's bucket small enough to spill; surface runoff
  ! counted in calls 1-2 with OVRTSWCRT = 0), then two pass-through calls
  ! (GWBASESWCRT = 2).  Drainage and runoff carry values below 0 and above
  ! 1000 mm, which the routine zeroes.  Every input and output word is dumped.
  use baseflow_harness
  implicit none
  integer, parameter :: ix = 6, jx = 5, ixrt = 24, jxrt = 20, nb = 4
  real, parameter :: dt = 3600.0
  integer :: gwsubbasmsk(ix, jx), gw_strm_msk(ixrt, jxrt), gw_strm_msk_lind(ixrt, jxrt)
  real :: runoff1(ix, jx), runoff2(ix, jx), dist(ixrt, jxrt, 9), qinflowbase(ixrt, jxrt)
  real :: basns_area(nb), c(nb), ex(nb), z_mx(nb), z(nb), qin(nb), qout(nb), pixct(nb)
  integer(kind=int64) :: basnsind(nb) = (/ 11, 22, 33, 44 /)
  integer :: i, j, n, b, sw, ov, u, v
  character(len=1024) :: dir

  call get_command_argument(1, dir)
  if (len_trim(dir) == 0) error stop 'usage: run_baseflow OUTDIR'
  do j = 1, jx
    do i = 1, ix
      gwsubbasmsk(i, j) = int(basnsind(1 + mod(i / 2 + j / 3 * 2, nb)))
    end do
  end do
  gw_strm_msk = 0
  dist = 0.0
  dist(:, :, 9) = 62500.0
  do j = 1, jxrt
    do i = 1, ixrt
      if (mod(i + 2 * j, 5) == 0) gw_strm_msk(i, j) = 1 + mod(i + j, nb)
      dist(i, j, 9) = 62500.0 * (1.0 + 0.01 * frac(i + j, 7))
    end do
  end do
  gw_strm_msk_lind = gw_strm_msk
  ! module_RT.F90:1383-1394, without MPP_LAND
  pixct = 0.
  do i = 1, ixrt
    do j = 1, jxrt
      if (gw_strm_msk(i, j) > 0) pixct(gw_strm_msk(i, j)) = pixct(gw_strm_msk(i, j)) + 1.0
    end do
  end do
  do b = 1, nb
    basns_area(b) = 1.0e6 * (1.0 + frac(b, 11))
    c(b) = 0.5 + 2.0 * frac(b, 13)
    ex(b) = 1.0 + 4.0 * frac(b, 17)
    z_mx(b) = 30.0 + 50.0 * frac(b, 19)
    z(b) = 0.5 * z_mx(b) * frac(b, 23)
  end do
  z_mx(3) = 4.0                 ! small enough to spill

  open(newunit=u, file=trim(dir)//'/baseflow-static.csv', status='replace')
  write(u, '(A)') 'b,basnsind,area,c,ex,z_mx,z0,pixct'
  do b = 1, nb
    write(u, '(I0,",",I0,6(",",ES24.16E3))') b, basnsind(b), basns_area(b), c(b), ex(b), &
        z_mx(b), z(b), pixct(b)
  end do
  close(u)
  open(newunit=u, file=trim(dir)//'/baseflow-grids.csv', status='replace')
  write(u, '(A)') 'grid,i,j,mask,dist9'
  do j = 1, jx
    do i = 1, ix
      write(u, '(A,3(",",I0),",",ES24.16E3)') 'lsm', i, j, gwsubbasmsk(i, j), 0.0
    end do
  end do
  do j = 1, jxrt
    do i = 1, ixrt
      write(u, '(A,3(",",I0),",",ES24.16E3)') 'rt', i, j, gw_strm_msk_lind(i, j), dist(i, j, 9)
    end do
  end do
  close(u)

  open(newunit=u, file=trim(dir)//'/baseflow-forcing.csv', status='replace')
  write(u, '(A)') 'call,sw,ov,i,j,runoff1,runoff2'
  open(newunit=v, file=trim(dir)//'/baseflow-steps.csv', status='replace')
  write(v, '(A)') 'call,b,z,qin,qout'
  do n = 1, 6
    sw = merge(1, 2, n <= 4)
    ov = merge(0, 1, n <= 2)
    do j = 1, jx
      do i = 1, ix
        runoff1(i, j) = 12.0 * frac(i * 3 + j * 5 + n, 29)
        runoff2(i, j) = 4.0 * frac(i * 7 + j + n * 3, 31)
      end do
    end do
    runoff2(2, 3) = -1.0; runoff1(5, 1) = 1500.0; runoff2(6, 5) = 2000.0
    do j = 1, jx
      do i = 1, ix
        write(u, '(I0,4(",",I0),2(",",ES24.16E3))') n, sw, ov, i, j, runoff1(i, j), runoff2(i, j)
      end do
    end do
    call simp_gw_buck(ix, jx, ixrt, jxrt, nb, nb, basns_area, basnsind, gw_strm_msk_lind, &
        gwsubbasmsk, runoff1, runoff2, z, qin, qout, qinflowbase, gw_strm_msk, pixct, dist, dt, &
        c, ex, z_mx, sw, ov)
    do b = 1, nb
      write(v, '(I0,",",I0,3(",",ES24.16E3))') n, b, z(b), qin(b), qout(b)
    end do
    write(v, '(A)') '#qinflowbase'
    do j = 1, jxrt
      do i = 1, ixrt
        if (gw_strm_msk_lind(i, j) > 0) write(v, '(A,I0,2(",",I0),",",ES24.16E3)') 'q', n, i, j, qinflowbase(i, j)
      end do
    end do
  end do
  close(u); close(v)

contains

  real function frac(m, p)
    integer, intent(in) :: m, p
    frac = real(mod(m * p * 7919, 1009)) / 1009.0
  end function frac

end program run_wrf_hydro_baseflow_oracle
