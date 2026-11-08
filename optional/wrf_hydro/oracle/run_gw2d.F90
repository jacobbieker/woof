program run_wrf_hydro_gw2d_oracle
  ! Drive WRF-Hydro's 2-D groundwater model on a 12 x 14 routing grid
  ! (AGGFACTRT = 2 over a 6 x 7 LSM grid): gw2d_ini with two pre-cycles
  ! (one hydraulic conductivity at the 100 sentinel), then four calls of the
  ! GWBASESWCRT = 3 arm of HYDRO_drv's groundwater call, the first three with
  ! the soil coupling (gwsoilcpl = 1: heads below, within and above the soil
  ! column, soil types past 12 skipped) and the last without.  Heads start
  ! in the aquifer, at its bottom (the re-wetting fix) and above the surface
  ! (excess), with ocean cells (ltype 2).  Every input and output word is
  ! dumped.  ix <= jx: gwstep allocates zfac(1:ydim) and sub_tri_solv's
  ! z-sweep indexes it 1..xdim.
  use gw2d_harness
  implicit none
  integer, parameter :: ixrt = 12, jxrt = 14, agg = 2, nsoil = 4
  integer :: i, j, k, n, u, v, w
  character(len=1024) :: dir
  real, parameter :: dx = 250.0, dt = 3600.0

  call get_command_argument(1, dir)
  if (len_trim(dir) == 0) error stop 'usage: run_gw2d OUTDIR'
  nlst(1)%nsoil = nsoil
  nlst(1)%aggfactRT = agg
  nlst(1)%GwPreCycles = 2
  nlst(1)%GwPreDiagInterval = 1
  nlst(1)%igrid = 1
  nlst(1)%output_gw = 0
  nlst(1)%dt = dt
  allocate(nlst(1)%zsoil8(nsoil))
  nlst(1)%zsoil8 = (/ -0.10, -0.40, -1.00, -2.00 /)
  rt_domain(1)%ixrt = ixrt; rt_domain(1)%jxrt = jxrt
  rt_domain(1)%ix = ixrt / agg; rt_domain(1)%jx = jxrt / agg
  allocate(rt_domain(1)%elrt(ixrt, jxrt), rt_domain(1)%soiltypRT(ixrt, jxrt))
  allocate(rt_domain(1)%subsurface%grid_transform%smcrt(ixrt, jxrt, nsoil))
  allocate(rt_domain(1)%subsurface%grid_transform%smcmaxrt(ixrt, jxrt, nsoil))
  allocate(rt_domain(1)%subsurface%properties%sldpth(nsoil))
  allocate(rt_domain(1)%overland%control%surface_water_head_routing(ixrt, jxrt))
  rt_domain(1)%subsurface%properties%sldpth = (/ 0.10, 0.30, 0.60, 1.00 /)
  call gw2d_allocate(1, ixrt, jxrt, nsoil)
  gw2d(1)%ebot = 0.; gw2d(1)%eocn = 0.
  do j = 1, jxrt
    do i = 1, ixrt
      rt_domain(1)%elrt(i, j) = 100.0 + 0.8 * i + 0.5 * j + 3.0 * frac(i * 13 + j * 7, 31)
      gw2d(1)%elev(i, j) = rt_domain(1)%elrt(i, j)
      gw2d(1)%bot(i, j) = gw2d(1)%elev(i, j) - 15.0 - 10.0 * frac(i * 5 + j * 11, 37)
      gw2d(1)%poros(i, j) = 0.1 + 0.3 * frac(i * 17 + j * 3, 41)
      gw2d(1)%hycond(i, j) = 1.0e-5 * (1.0 + 99.0 * frac(i * 19 + j * 23, 43))
      gw2d(1)%ltype(i, j) = 1
      select case (mod(i * 3 + j * 7, 9))
      case (0)       ! above the surface: excess
        gw2d(1)%ho(i, j) = gw2d(1)%elev(i, j) + 0.3 * frac(i + j, 47)
      case (1)       ! at the aquifer bottom: the re-wetting fix
        gw2d(1)%ho(i, j) = gw2d(1)%bot(i, j) + 0.002
      case (2)       ! within the soil column
        gw2d(1)%ho(i, j) = gw2d(1)%elev(i, j) - 1.2 * frac(i * 7 + j, 53)
      case default
        gw2d(1)%ho(i, j) = gw2d(1)%elev(i, j) - 2.5 - 8.0 * frac(i * 29 + j * 31, 59)
      end select
      rt_domain(1)%soiltypRT(i, j) = 1 + mod(i * 5 + j * 3, 14)
      do k = 1, nsoil
        rt_domain(1)%subsurface%grid_transform%smcmaxrt(i, j, k) = 0.40 + 0.08 * frac(i + j * k, 61)
        rt_domain(1)%subsurface%grid_transform%smcrt(i, j, k) = &
            rt_domain(1)%subsurface%grid_transform%smcmaxrt(i, j, k) * (0.35 + 0.6 * frac(i * k + j, 67))
      end do
    end do
  end do
  gw2d(1)%ltype(1, :) = 2                      ! an ocean edge
  gw2d(1)%ltype(ixrt, 1:3) = 0                 ! cells outside the aquifer
  gw2d(1)%hycond(4, 6) = 100.0                 ! gw2d_ini's sentinel
  gw2d(1)%hycond(9, 3) = 100.0
  gw2d(1)%compres = 0.
  gw2d(1)%h = gw2d(1)%ho
  rt_domain(1)%overland%control%surface_water_head_routing = 0.

  open(newunit=u, file=trim(dir)//'/gw2d-static.csv', status='replace')
  write(u, '(A)') 'i,j,ltype,soiltyp,elev,bot,poros,hycond,ho0,smcmax1,smcmax2,smcmax3,smcmax4'
  do j = 1, jxrt
    do i = 1, ixrt
      write(u, '(4(I0,","),ES24.16E3,8(",",ES24.16E3))') i, j, gw2d(1)%ltype(i, j), &
          rt_domain(1)%soiltypRT(i, j), gw2d(1)%elev(i, j), gw2d(1)%bot(i, j), &
          gw2d(1)%poros(i, j), gw2d(1)%hycond(i, j), gw2d(1)%ho(i, j), &
          rt_domain(1)%subsurface%grid_transform%smcmaxrt(i, j, :)
    end do
  end do
  close(u)
  open(newunit=v, file=trim(dir)//'/gw2d-steps.csv', status='replace')
  write(v, '(A)') 'step,i,j,ho,h,convgw,excess,qsgwrt,qdarcyrt,sfchead,hycond,smc1,smc2,smc3,smc4'
  open(newunit=w, file=trim(dir)//'/gw2d-totals.csv', status='replace')
  write(w, '(A)') 'step,istep,ebot,eocn'
  open(newunit=u, file=trim(dir)//'/gw2d-qsgw.csv', status='replace')
  write(u, '(A)') 'step,i,j,qsgw'

  call gw2d_ini(1, dt, dx)
  call dump(0)
  do n = 1, 4
    nlst(1)%gwsoilcpl = merge(1, 0, n <= 3)
    call gw_hydro_step(1)
    call dump(n)
    do j = 1, rt_domain(1)%jx
      do i = 1, rt_domain(1)%ix
        write(u, '(3(I0,","),ES24.16E3)') n, i, j, gw2d(1)%qsgw(i, j)
      end do
    end do
  end do
  close(u); close(v); close(w)

contains

  subroutine dump(n)
    integer, intent(in) :: n
    integer :: i, j
    do j = 1, jxrt
      do i = 1, ixrt
        write(v, '(3(I0,","),ES24.16E3,11(",",ES24.16E3))') n, i, j, gw2d(1)%ho(i, j), &
            gw2d(1)%h(i, j), gw2d(1)%convgw(i, j), gw2d(1)%excess(i, j), &
            gw2d(1)%qsgwrt(i, j), gw2d(1)%qdarcyRT(i, j), &
            rt_domain(1)%overland%control%surface_water_head_routing(i, j), &
            gw2d(1)%hycond(i, j), rt_domain(1)%subsurface%grid_transform%smcrt(i, j, :)
      end do
    end do
    write(w, '(2(I0,","),ES24.16E3,",",ES24.16E3)') n, gw2d(1)%istep, gw2d(1)%ebot, gw2d(1)%eocn
  end subroutine dump

  real function frac(m, p)
    integer, intent(in) :: m, p
    frac = real(mod(m * p * 7919, 1009)) / 1009.0
  end function frac

end program run_wrf_hydro_gw2d_oracle
