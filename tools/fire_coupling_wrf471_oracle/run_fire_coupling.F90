! WRF v4.7.1 fire-coupling oracle: the whole two-way fire driver.
!
! Everything that computes is WRF's own, byte-unmodified and pinned
! (SOURCES.sha256): module_fr_fire_driver_wrf.F's fire_driver_em_init (the
! fire_driver_em passes 1 and 2 start_domain_em makes) and
! fire_driver_em_step (passes 3 to 6, then fire_tendency), with everything
! they call -- fire_ignition_convert, set_flags, interpolate_atm2fire
! (interpolate_2d, continue_at_boundary), interpolate_z2fire,
! set_ideal_coord, set_nfuel_cat, set_fire_params, init_no_fire,
! fire_model (prop_ls_rk3, tign_update, reinit_ls_rk3, ignite_fire,
! fuel_left, heat_fluxes), sum_2d_cells and the driver's flux scaling.
! This program only fills a stub TYPE(domain) the way a WRF run would
! find it, advances grid%itimestep as solve_em does before the physics,
! and dumps.  stub_wrf.F90 declares the types; nothing there computes.
!
! Cases (all serial, one patch, one tile, as woof runs):
!   ideal   ideal coordinates, uniform fuel (fire_fuel_read = 0), two
!           ignitions (a walked line and a later point), sheared wind
!           varying in space and time, terrain and roughness varying.
!   real    longitude/latitude ignitions (fire_ignition_longlat = 1 via
!           the _lon/_lat keys, unit_fxlong from cen_lat), a fuel map with
!           no-fuel cells (fire_fuel_read = -1), fire_atm_feedback 0.7,
!           fire_wind_height 10 m, other extinction depths and crown height.
!   edge    a shallow model whose top u level lies below fire_wind_height
!           (the kdmax branch), roughness above fire_wind_height in some
!           columns (the zero-wind branch), fuel by altitude
!           (fire_fuel_read = 1), fire_advection = 0, no reinitialization.
!
! fire_tendency is also called directly (case tendency) on fluxes with
! canopy parts, so both branches of the crown-height test carry weight.
!
! tracer_opt = 0: add_fire_tracer_emissions is held to WRF by
! tools/smoke_wrf471_oracle; woof writes the flux it is the source of.
module coupling_cases
  use module_domain
  use module_configure
  use module_state_description, only: num_tracer
  use module_model_constants, only: g
  use module_fr_fire_driver_wrf, only: fire_driver_em_init, fire_driver_em_step
  use module_fr_fire_atm, only: fire_tendency
  use oracle_io
  implicit none

  integer :: nx, ny, nz, ir, jr, nxf, nyf
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
  integer :: ips, ipe, jps, jpe, kps, kpe
  real, allocatable :: rho(:,:,:), z_at_w(:,:,:), dz8w(:,:,:)

contains

  subroutine set_dims(nx_, ny_, nz_, ir_, jr_)
    integer, intent(in) :: nx_, ny_, nz_, ir_, jr_
    nx = nx_; ny = ny_; nz = nz_; ir = ir_; jr = jr_
    nxf = nx * ir; nyf = ny * jr
    ids = 1; ide = nx + 1; jds = 1; jde = ny + 1; kds = 1; kde = nz + 1
    ims = -2; ime = nx + 4; jms = -2; jme = ny + 4; kms = 1; kme = nz + 1
    ips = ids; ipe = ide; jps = jds; jpe = jde; kps = kds; kpe = kde
  end subroutine set_dims

  subroutine a3(p)
    real, pointer :: p(:,:,:)
    allocate(p(ims:ime, kms:kme, jms:jme)); p = 0.
  end subroutine a3

  subroutine a2(p)
    real, pointer :: p(:,:)
    allocate(p(ims:ime, jms:jme)); p = 0.
  end subroutine a2

  subroutine ac(p, n)
    real, pointer :: p(:,:,:)
    integer, intent(in) :: n
    allocate(p(ims:ime, n, jms:jme)); p = 0.
  end subroutine ac

  subroutine af(p)
    real, pointer :: p(:,:)
    integer :: ifms, ifme, jfms, jfme
    ifms = (ims - 1) * ir + 1; ifme = ime * ir
    jfms = (jms - 1) * jr + 1; jfme = jme * jr
    allocate(p(ifms:ifme, jfms:jfme)); p = 0.
  end subroutine af

  subroutine allocate_grid(grid, cf, dx, dt)
    type(domain), intent(inout) :: grid
    type(grid_config_rec_type), intent(in) :: cf
    real, intent(in) :: dx, dt
    grid%ids = ids; grid%ide = ide; grid%jds = jds; grid%jde = jde
    grid%kds = kds; grid%kde = kde
    grid%ims = ims; grid%ime = ime; grid%jms = jms; grid%jme = jme
    grid%kms = kms; grid%kme = kme
    grid%ips = ips; grid%ipe = ipe; grid%jps = jps; grid%jpe = jpe
    grid%kps = kps; grid%kpe = kpe
    grid%sr_x = ir; grid%sr_y = jr; grid%num_tiles = 1
    grid%dx = dx; grid%dy = dx; grid%dt = dt; grid%itimestep = 0
    allocate(grid%i_start(1), grid%i_end(1), grid%j_start(1), grid%j_end(1))
    grid%i_start = ids; grid%i_end = ide; grid%j_start = jds; grid%j_end = jde
    call a3(grid%u_2); call a3(grid%v_2); call a3(grid%ph_2); call a3(grid%phb)
    call a3(grid%rthfrten); call a3(grid%rqvfrten)
    allocate(grid%tracer(ims:ime, kms:kme, jms:jme, num_tracer)); grid%tracer = 0.
    call a2(grid%z0); call a2(grid%ht); call a2(grid%xlong); call a2(grid%xlat)
    call a2(grid%avg_fuel_frac); call a2(grid%grnhfx); call a2(grid%grnqfx)
    call a2(grid%canhfx); call a2(grid%canqfx); call a2(grid%grnhfx_fu)
    call a2(grid%grnqfx_fu); call a2(grid%uah); call a2(grid%vah)
    call a2(grid%mut); call a2(grid%rainc); call a2(grid%rainnc)
    call a2(grid%t2); call a2(grid%q2); call a2(grid%psfc)
    call a2(grid%rain_old); call a2(grid%t2_old); call a2(grid%q2_old)
    call a2(grid%psfc_old); call a2(grid%rh_fire)
    call ac(grid%fmc_gc, cf%nfmc); call ac(grid%fmep, 2)
    call ac(grid%fmc_equi, cf%nfmc); call ac(grid%fmc_lag, cf%nfmc)
    allocate(grid%c1h(kms:kme), grid%c2h(kms:kme)); grid%c1h = 0.; grid%c2h = 0.
    call af(grid%uf); call af(grid%vf); call af(grid%zsf); call af(grid%dzdxf)
    call af(grid%dzdyf); call af(grid%bbb); call af(grid%betafl)
    call af(grid%phiwc); call af(grid%r_0); call af(grid%fgip)
    call af(grid%ischap); call af(grid%iboros); call af(grid%fmc_g)
    call af(grid%lfn); call af(grid%lfn_hist); call af(grid%lfn_0)
    call af(grid%lfn_1); call af(grid%lfn_2); call af(grid%lfn_s0)
    call af(grid%lfn_s1); call af(grid%lfn_s2); call af(grid%lfn_s3)
    call af(grid%flame_length); call af(grid%ros_front); call af(grid%tign_g)
    call af(grid%fuel_frac); call af(grid%fire_area)
    call af(grid%burnt_area_dt); call af(grid%fgrnhfx); call af(grid%fgrnqfx)
    call af(grid%fcanhfx); call af(grid%fcanqfx); call af(grid%ros)
    call af(grid%fxlong); call af(grid%fxlat); call af(grid%fz0)
    call af(grid%nfuel_cat); call af(grid%fuel_time)
    if (allocated(rho)) deallocate(rho, z_at_w, dz8w)
    allocate(rho(ims:ime, kms:kme, jms:jme), z_at_w(ims:ime, kms:kme, jms:jme), &
             dz8w(ims:ime, kms:kme, jms:jme))
    rho = 0.; z_at_w = 0.; dz8w = 0.
  end subroutine allocate_grid

  ! The atmosphere: geopotential on w levels over the terrain, phy_prep's
  ! z_at_w and dz8w from it, a density profile, dry column mass and the
  ! hybrid weights.  zw(k) is the height of w level k above the ground.
  subroutine set_atmosphere(grid, zw)
    type(domain), intent(inout) :: grid
    real, intent(in) :: zw(:)
    integer :: i, j, k
    do j = jds, jde
      do k = kds, kde
        do i = ids, ide
          grid%phb(i,k,j) = g * (grid%ht(i,j) + zw(k) * (1. + 0.01 * real(i)))
          grid%ph_2(i,k,j) = 0.3 * real(k - 1) * sin(0.4 * real(i + 2 * j))
        end do
      end do
    end do
    do j = jds, jde
      do k = kds, kde
        do i = ids, ide
          z_at_w(i,k,j) = (grid%phb(i,k,j) + grid%ph_2(i,k,j)) / g
        end do
      end do
      do k = kds, kde - 1
        do i = ids, ide
          dz8w(i,k,j) = z_at_w(i,k+1,j) - z_at_w(i,k,j)
          rho(i,k,j) = 1.18 - 0.012 * real(k) + 0.001 * real(i - j)
        end do
      end do
      do i = ids, ide
        dz8w(i,kde,j) = dz8w(i,kde-1,j)
        rho(i,kde,j) = rho(i,kde-1,j)
        grid%mut(i,j) = 9.2e4 + 15. * real(i) - 9. * real(j)
      end do
    end do
    do k = kds, kde
      grid%c1h(k) = 1. - 0.1 * real(k - 1)
      grid%c2h(k) = 120. * real(k - 1)
    end do
  end subroutine set_atmosphere

  ! A wind that shears with height, varies across the domain and turns
  ! slowly with the step number.
  subroutine set_wind(grid, istep, u0, v0)
    type(domain), intent(inout) :: grid
    integer, intent(in) :: istep
    real, intent(in) :: u0, v0
    integer :: i, j, k
    real :: s
    s = 0.02 * real(istep)
    do j = jds, jde
      do k = kds, kde - 1
        do i = ids, ide
          grid%u_2(i,k,j) = u0 + 0.35 * real(k) + 0.04 * real(j) - s
          grid%v_2(i,k,j) = v0 - 0.15 * real(k) + 0.03 * real(i) + s
        end do
      end do
    end do
  end subroutine set_wind

  subroutine put_inputs(grid)
    type(domain), intent(in) :: grid
    call oracle_put('in_ht', grid%ht(ids:ide-1, jds:jde-1))
    call oracle_put('in_z0', grid%z0(ids:ide-1, jds:jde-1))
    call oracle_put('in_ph', grid%ph_2(ids:ide-1, kds:kde, jds:jde-1))
    call oracle_put('in_phb', grid%phb(ids:ide-1, kds:kde, jds:jde-1))
    call oracle_put('in_rho', rho(ids:ide-1, kds:kde-1, jds:jde-1))
    call oracle_put('in_z_at_w', z_at_w(ids:ide-1, kds:kde, jds:jde-1))
    call oracle_put('in_dz8w', dz8w(ids:ide-1, kds:kde-1, jds:jde-1))
    call oracle_put('in_mut', grid%mut(ids:ide-1, jds:jde-1))
    call oracle_put('in_c1h', grid%c1h(kds:kde-1))
    call oracle_put('in_c2h', grid%c2h(kds:kde-1))
    call oracle_put('in_xlat', grid%xlat(ids:ide-1, jds:jde-1))
    call oracle_put('in_xlong', grid%xlong(ids:ide-1, jds:jde-1))
    call oracle_put('in_nfuel_cat', grid%nfuel_cat(1:nxf, 1:nyf))
    call oracle_put('in_zsf', grid%zsf(1:nxf, 1:nyf))
    call oracle_put('in_dzdxf', grid%dzdxf(1:nxf, 1:nyf))
    call oracle_put('in_dzdyf', grid%dzdyf(1:nxf, 1:nyf))
  end subroutine put_inputs

  subroutine put_init(grid)
    type(domain), intent(in) :: grid
    call oracle_put('init_nfuel_cat', grid%nfuel_cat(1:nxf, 1:nyf))
    call oracle_put('init_fuel_time', grid%fuel_time(1:nxf, 1:nyf))
    call oracle_put('init_fgip', grid%fgip(1:nxf, 1:nyf))
    call oracle_put('init_r_0', grid%r_0(1:nxf, 1:nyf))
    call oracle_put('init_bbb', grid%bbb(1:nxf, 1:nyf))
    call oracle_put('init_phiwc', grid%phiwc(1:nxf, 1:nyf))
    call oracle_put('init_betafl', grid%betafl(1:nxf, 1:nyf))
    call oracle_put('init_fmc_g', grid%fmc_g(1:nxf, 1:nyf))
    call oracle_put('init_fxlong', grid%fxlong(1:nxf, 1:nyf))
    call oracle_put('init_fxlat', grid%fxlat(1:nxf, 1:nyf))
    call oracle_put('init_lfn', grid%lfn(1:nxf, 1:nyf))
    call oracle_put('init_tign', grid%tign_g(1:nxf, 1:nyf))
    call oracle_put('init_fuel_frac', grid%fuel_frac(1:nxf, 1:nyf))
  end subroutine put_init

  subroutine put_step(grid, s)
    type(domain), intent(in) :: grid
    integer, intent(in) :: s
    character(len=16) :: p
    write(p, '(a,i0,a)') 's', s, '_'
    call oracle_put(trim(p) // 'in_u', grid%u_2(ids:ide, kds:kde-1, jds:jde-1))
    call oracle_put(trim(p) // 'in_v', grid%v_2(ids:ide-1, kds:kde-1, jds:jde))
    call oracle_put(trim(p) // 'uf', grid%uf(1:nxf, 1:nyf))
    call oracle_put(trim(p) // 'vf', grid%vf(1:nxf, 1:nyf))
    call oracle_put(trim(p) // 'uah', grid%uah(ids:ide, jds:jde))
    call oracle_put(trim(p) // 'vah', grid%vah(ids:ide, jds:jde))
    call oracle_put(trim(p) // 'lfn', grid%lfn(1:nxf, 1:nyf))
    call oracle_put(trim(p) // 'tign', grid%tign_g(1:nxf, 1:nyf))
    call oracle_put(trim(p) // 'ros', grid%ros(1:nxf, 1:nyf))
    call oracle_put(trim(p) // 'fuel_frac', grid%fuel_frac(1:nxf, 1:nyf))
    call oracle_put(trim(p) // 'fire_area', grid%fire_area(1:nxf, 1:nyf))
    call oracle_put(trim(p) // 'burnt_area_dt', grid%burnt_area_dt(1:nxf, 1:nyf))
    call oracle_put(trim(p) // 'fgrnhfx', grid%fgrnhfx(1:nxf, 1:nyf))
    call oracle_put(trim(p) // 'fgrnqfx', grid%fgrnqfx(1:nxf, 1:nyf))
    call oracle_put(trim(p) // 'grnhfx', grid%grnhfx(ids:ide-1, jds:jde-1))
    call oracle_put(trim(p) // 'grnqfx', grid%grnqfx(ids:ide-1, jds:jde-1))
    call oracle_put(trim(p) // 'canhfx', grid%canhfx(ids:ide-1, jds:jde-1))
    call oracle_put(trim(p) // 'canqfx', grid%canqfx(ids:ide-1, jds:jde-1))
    call oracle_put(trim(p) // 'avg_fuel_frac', grid%avg_fuel_frac(ids:ide-1, jds:jde-1))
    call oracle_put(trim(p) // 'rthfrten', grid%rthfrten(ids:ide-1, kds:kde-1, jds:jde-1))
    call oracle_put(trim(p) // 'rqvfrten', grid%rqvfrten(ids:ide-1, kds:kde-1, jds:jde-1))
  end subroutine put_step

  subroutine run_case(grid, cf, nsteps, dumps, u0, v0)
    type(domain), intent(inout) :: grid
    type(grid_config_rec_type), intent(in) :: cf
    integer, intent(in) :: nsteps, dumps(:)
    real, intent(in) :: u0, v0
    integer :: istep
    call oracle_put('nx', nx); call oracle_put('ny', ny); call oracle_put('nz', nz)
    call oracle_put('ir', ir); call oracle_put('jr', jr)
    call oracle_put('dx', grid%dx); call oracle_put('dt', grid%dt)
    call oracle_put('nsteps', nsteps); call oracle_put('dumps', dumps)
    call put_config(cf)
    call put_inputs(grid)
    grid%itimestep = 0
    call fire_driver_em_init(grid, cf, ids, ide, kds, kde, jds, jde, &
                             ims, ime, kms, kme, jms, jme, &
                             ips, ipe, kps, kpe, jps, jpe)
    call put_init(grid)
    do istep = 1, nsteps
      grid%itimestep = istep          ! solve_em.F:376, before the physics
      call set_wind(grid, istep, u0, v0)
      call fire_driver_em_step(grid, cf, ids, ide, kds, kde, jds, jde, &
                               ims, ime, kms, kme, jms, jme, &
                               ips, ipe, kps, kpe, jps, jpe, &
                               rho, z_at_w, dz8w)
      if (any(dumps == istep)) call put_step(grid, istep)
    end do
    call oracle_close()
  end subroutine run_case

  ! The &fire values a case runs with, in the port's order.
  subroutine put_config(cf)
    type(grid_config_rec_type), intent(in) :: cf
    real :: ig(16, 5)
    integer :: n
    call oracle_put('cfg_ints', (/ cf%fire_num_ignitions, cf%fire_fuel_read, &
        cf%fire_fuel_cat, cf%fire_advection, merge(1, 0, cf%fire_lsm_reinit), &
        cf%fire_lsm_reinit_iter, cf%fire_lsm_band_ngp, cf%fire_viscosity_ngp /))
    call oracle_put('cfg_reals', (/ cf%fire_wind_height, cf%fire_ext_grnd, &
        cf%fire_ext_crwn, cf%fire_crwn_hgt, cf%fire_atm_feedback, &
        cf%fire_viscosity, cf%fire_viscosity_bg, cf%fire_viscosity_band, &
        cf%fire_lfn_ext_up, cf%fire_slope_factor, cf%cen_lat /))
    ! per ignition: start_x, start_y, end_x, end_y (m), start_lon,
    ! start_lat, end_lon, end_lat, radius, ros, start_time, end_time
    ig = 0.
    ig(:, 1) = (/ cf%fire_ignition_start_x1, cf%fire_ignition_start_y1, &
        cf%fire_ignition_end_x1, cf%fire_ignition_end_y1, &
        cf%fire_ignition_start_lon1, cf%fire_ignition_start_lat1, &
        cf%fire_ignition_end_lon1, cf%fire_ignition_end_lat1, &
        cf%fire_ignition_radius1, cf%fire_ignition_ros1, &
        cf%fire_ignition_start_time1, cf%fire_ignition_end_time1, 0., 0., 0., 0. /)
    ig(:, 2) = (/ cf%fire_ignition_start_x2, cf%fire_ignition_start_y2, &
        cf%fire_ignition_end_x2, cf%fire_ignition_end_y2, &
        cf%fire_ignition_start_lon2, cf%fire_ignition_start_lat2, &
        cf%fire_ignition_end_lon2, cf%fire_ignition_end_lat2, &
        cf%fire_ignition_radius2, cf%fire_ignition_ros2, &
        cf%fire_ignition_start_time2, cf%fire_ignition_end_time2, 0., 0., 0., 0. /)
    ig(:, 3) = (/ cf%fire_ignition_start_x3, cf%fire_ignition_start_y3, &
        cf%fire_ignition_end_x3, cf%fire_ignition_end_y3, &
        cf%fire_ignition_start_lon3, cf%fire_ignition_start_lat3, &
        cf%fire_ignition_end_lon3, cf%fire_ignition_end_lat3, &
        cf%fire_ignition_radius3, cf%fire_ignition_ros3, &
        cf%fire_ignition_start_time3, cf%fire_ignition_end_time3, 0., 0., 0., 0. /)
    n = 3
    call oracle_put('cfg_ignitions', ig(1:12, 1:n))
  end subroutine put_config

end module coupling_cases


program run_fire_coupling
  use coupling_cases
  use module_fr_fire_atm, only: fire_tendency
  implicit none
  character(len=1024) :: root
  type(domain), target :: grid
  type(grid_config_rec_type) :: cf
  real :: zw(9), zw_edge(4)
  integer :: i, j, ifr, jfr
  real :: xf, yf

  call oracle_root(root)
  zw = (/ 0., 4., 10., 18., 30., 50., 80., 120., 170. /)

  ! ---- ideal ---------------------------------------------------------------
  call oracle_open('ideal')
  call set_dims(12, 10, 8, 4, 4)
  cf = grid_config_rec_type()
  cf%ifire = 2
  cf%fire_boundary_guard = -1   ! PROVENANCE.md: woof does not stop the run
  cf%fire_fuel_read = 0
  cf%fire_fuel_cat = 3
  cf%fire_num_ignitions = 2
  cf%fire_ignition_start_x1 = 120.; cf%fire_ignition_start_y1 = 110.
  cf%fire_ignition_end_x1 = 150.;   cf%fire_ignition_end_y1 = 170.
  cf%fire_ignition_radius1 = 12.;   cf%fire_ignition_ros1 = 0.8
  cf%fire_ignition_start_time1 = 0.5; cf%fire_ignition_end_time1 = 6.
  cf%fire_ignition_start_x2 = 250.; cf%fire_ignition_start_y2 = 200.
  cf%fire_ignition_radius2 = 9.;   cf%fire_ignition_ros2 = 0.6
  cf%fire_ignition_start_time2 = 4.
  call allocate_grid(grid, cf, 30., 0.5)
  do j = jds, jde
    do i = ids, ide
      grid%ht(i,j) = 40. + 1.5 * real(i) - 0.8 * real(j)
      grid%z0(i,j) = 0.05 + 0.02 * real(mod(i + 2 * j, 5))
    end do
  end do
  do jfr = 1, nyf
    do ifr = 1, nxf
      xf = real(ifr); yf = real(jfr)
      grid%zsf(ifr,jfr) = 40. + 0.4 * xf - 0.2 * yf
      grid%dzdxf(ifr,jfr) = 0.05 + 0.01 * sin(0.3 * xf)
      grid%dzdyf(ifr,jfr) = -0.03 + 0.01 * cos(0.2 * yf)
    end do
  end do
  call set_atmosphere(grid, zw)
  call run_case(grid, cf, 40, (/ 1, 12, 26, 40 /), 2.5, 1.0)

  ! ---- real ----------------------------------------------------------------
  call oracle_open('real')
  call set_dims(12, 10, 8, 5, 4)
  cf = grid_config_rec_type()
  cf%ifire = 2
  cf%fire_boundary_guard = -1   ! PROVENANCE.md: woof does not stop the run
  cf%fire_fuel_read = -1
  cf%fire_num_ignitions = 3
  cf%cen_lat = 40.0012; cf%cen_lon = -105.003
  cf%fire_ignition_start_lon1 = -105.0016; cf%fire_ignition_start_lat1 = 40.0011
  cf%fire_ignition_end_lon1 = -105.0010;   cf%fire_ignition_end_lat1 = 40.0016
  cf%fire_ignition_radius1 = 15.;  cf%fire_ignition_ros1 = 0.8
  cf%fire_ignition_start_time1 = 1.; cf%fire_ignition_end_time1 = 5.
  cf%fire_ignition_start_lon2 = -104.9993; cf%fire_ignition_start_lat2 = 40.0006
  cf%fire_ignition_radius2 = 10.;  cf%fire_ignition_start_time2 = 3.; cf%fire_ignition_ros2 = 0.5
  ! radius 0: WRF counts ignitions up to the last with radius > 0
  cf%fire_ignition_start_lon3 = -105.0003; cf%fire_ignition_start_lat3 = 40.0002
  cf%fire_ignition_radius3 = 0.
  cf%fire_atm_feedback = 0.7
  cf%fire_wind_height = 10.
  cf%fire_ext_grnd = 30.; cf%fire_ext_crwn = 40.; cf%fire_crwn_hgt = 8.
  call allocate_grid(grid, cf, 25., 0.6)
  do j = jds, jde
    do i = ids, ide
      grid%ht(i,j) = 1600. + 4. * real(i) + 2.5 * real(j)
      grid%z0(i,j) = 0.1 + 0.03 * real(mod(3 * i + j, 4))
      grid%xlat(i,j) = 39.9997 + 0.000225 * real(j) + 0.000004 * real(i)
      grid%xlong(i,j) = -105.0026 + 0.000293 * real(i) - 0.000003 * real(j)
    end do
  end do
  do jfr = 1, nyf
    do ifr = 1, nxf
      xf = real(ifr); yf = real(jfr)
      grid%nfuel_cat(ifr,jfr) = real(1 + mod(ifr / 7 + jfr / 9, 3))
      if (ifr > 44 .and. jfr < 12) grid%nfuel_cat(ifr,jfr) = 14.
      if (mod(ifr + jfr, 17) == 0) grid%nfuel_cat(ifr,jfr) = 8.
      grid%zsf(ifr,jfr) = 1600. + 0.8 * xf + 0.6 * yf
      grid%dzdxf(ifr,jfr) = 0.12 * cos(0.15 * xf)
      grid%dzdyf(ifr,jfr) = 0.08 * sin(0.1 * yf)
    end do
  end do
  call set_atmosphere(grid, zw)
  call run_case(grid, cf, 30, (/ 1, 10, 30 /), 3.5, -1.2)

  ! ---- edge ----------------------------------------------------------------
  call oracle_open('edge')
  call set_dims(12, 10, 3, 4, 4)
  cf = grid_config_rec_type()
  cf%ifire = 2
  cf%fire_boundary_guard = -1   ! PROVENANCE.md: woof does not stop the run
  cf%fire_fuel_read = 1
  cf%fire_num_ignitions = 1
  cf%fire_ignition_start_x1 = 180.; cf%fire_ignition_start_y1 = 150.
  cf%fire_ignition_radius1 = 14.;  cf%fire_ignition_ros1 = 0.6
  cf%fire_advection = 0
  cf%fire_lsm_reinit = .false.
  call allocate_grid(grid, cf, 30., 0.5)
  zw_edge = (/ 0., 1.2, 2.6, 4.4 /)
  do j = jds, jde
    do i = ids, ide
      grid%ht(i,j) = 1300. + 60. * real(i) + 25. * real(j)
      grid%z0(i,j) = 0.2
      if (mod(i + j, 6) == 0) grid%z0(i,j) = 15.
    end do
  end do
  do jfr = 1, nyf
    do ifr = 1, nxf
      xf = real(ifr); yf = real(jfr)
      grid%zsf(ifr,jfr) = 1300. + 15. * xf + 6. * yf
      grid%dzdxf(ifr,jfr) = 0.02
      grid%dzdyf(ifr,jfr) = -0.01
    end do
  end do
  call set_atmosphere(grid, zw_edge)
  call run_case(grid, cf, 40, (/ 1, 40 /), 1.5, 2.0)

  ! ---- fire_tendency alone -------------------------------------------------
  call tendency_case()

contains

  subroutine tendency_case()
    real, allocatable :: ghf(:,:), gqf(:,:), chf(:,:), cqf(:,:), mu(:,:), zs(:,:)
    real, allocatable :: rth(:,:,:), rqv(:,:,:)
    integer :: k
    call oracle_open('tendency')
    call set_dims(9, 7, 10, 1, 1)
    cf = grid_config_rec_type()
    cf%fire_ext_grnd = 35.; cf%fire_ext_crwn = 45.; cf%fire_crwn_hgt = 12.
    call allocate_grid(grid, cf, 50., 1.)
    allocate(ghf(ims:ime, jms:jme), gqf(ims:ime, jms:jme), chf(ims:ime, jms:jme), &
             cqf(ims:ime, jms:jme), mu(ims:ime, jms:jme), zs(ims:ime, jms:jme))
    allocate(rth(ims:ime, kms:kme, jms:jme), rqv(ims:ime, kms:kme, jms:jme))
    ghf = 0.; gqf = 0.; chf = 0.; cqf = 0.; mu = 0.; zs = 0.
    rth = -7.; rqv = -7.
    do j = jds, jde
      do i = ids, ide
        grid%ht(i,j) = 200. + 3. * real(i * j)
        ghf(i,j) = 1.5e4 * real(mod(i * 3 + j, 7))
        gqf(i,j) = 2.0e3 * real(mod(i + 5 * j, 4))
        chf(i,j) = 3.0e3 * real(mod(i + j, 3))
        cqf(i,j) = 4.0e2 * real(mod(2 * i + j, 5))
      end do
    end do
    call set_atmosphere(grid, (/ 0., 3., 7., 12., 18., 26., 36., 50., 70., 95., 130. /))
    call oracle_put('nx', nx); call oracle_put('ny', ny); call oracle_put('nz', nz)
    call oracle_put('cfg_reals', (/ cf%fire_ext_grnd, cf%fire_ext_crwn, cf%fire_crwn_hgt /))
    call oracle_put('in_grnhfx', ghf(ids:ide-1, jds:jde-1))
    call oracle_put('in_grnqfx', gqf(ids:ide-1, jds:jde-1))
    call oracle_put('in_canhfx', chf(ids:ide-1, jds:jde-1))
    call oracle_put('in_canqfx', cqf(ids:ide-1, jds:jde-1))
    call put_inputs(grid)
    ! module_fr_fire_driver_wrf.F:115-133's call, one tile.
    call fire_tendency(ids, ide-1, kds, kde, jds, jde-1, &
        ims, ime, kms, kme, jms, jme, &
        ids, ide-1, kds, kde, jds, jde-1, &
        ghf, gqf, chf, cqf, &
        cf%fire_ext_grnd, cf%fire_ext_crwn, cf%fire_crwn_hgt, &
        cf%fire_sfc_flx, cf%fire_heat_peak, cf%fire_tg_ub, &
        grid%ht, z_at_w, dz8w, grid%mut, grid%c1h, grid%c2h, rho, &
        rth, rqv)
    call oracle_put('rthfrten', rth(ids:ide-1, kds:kde-1, jds:jde-1))
    call oracle_put('rqvfrten', rqv(ids:ide-1, kds:kde-1, jds:jde-1))
    call oracle_close()
    k = 0
  end subroutine tendency_case

end program run_fire_coupling
