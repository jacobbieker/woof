! WRF v4.7.1 fire-model oracle: fire_model passes 4, 5 and 6 and the
! driver's aggregation, step after step, on synthetic fire meshes.
!
! Everything that computes is WRF's own, byte-unmodified and pinned
! (SOURCES.sha256): init_no_fire, set_ideal_coord, prop_ls_rk3 (tend_ls,
! fire_ros, the selectors, continue_at_boundary), tign_update,
! reinit_ls_rk3 (advance_ls_reinit), ignite_fire (nearest), fuel_left
! (fuel_left_cell_1), heat_fluxes and sum_2d_cells.  This program only
! sets the module flags to their Registry defaults (what set_flags copies
! in a WRF run), fills the fire_params arrays, and calls the routines in
! module_fr_fire_model.F's order:
!
!   pass 4  prop_ls_rk3; tign_update; reinit_ls_rk3 if fire_lsm_reinit
!   pass 5  lfn = lfn_out; ignite_fire for each ignition line
!   pass 6  fuel_left; fuel_frac_burnt = fuel_frac - fuel_frac_end;
!           fuel_frac = fuel_frac_end; heat_fluxes
!   driver  sum_2d_cells of fuel_frac, fgrnhfx, fgrnqfx, then
!           module_fr_fire_driver.F:941-953's scaling (copied as
!           driver_941_953.inc and checked against the pinned file).
!
! A case with nsub > 1 takes nsub fire steps of dt/nsub per outer step and
! also writes the outer step's mean atmospheric flux (sum in step order,
! then / nsub) -- the harness's sub-stepping, which WRF does not do.
!
! Two deliberate settings, both written down in PROVENANCE.md:
! boundary_guard = -1 (the Registry's 8 stops the run when the fire nears
! the edge; that is a run control, not a field) and ischap = 0 everywhere
! (fire_ros's chaparral arm leaves ros_base undefined).
module fire_cases
  use module_fr_fire_util
  use module_fr_fire_phys, only: fire_params, heat_fluxes
  use module_fr_fire_core
  use module_domain, only: domain
  use oracle_io
  implicit none

  integer, parameter :: KIND_ISO = 1, KIND_WIND = 2, KIND_CAP = 3

contains

  subroutine set_registry_flags(adv, reinit)
    integer, intent(in) :: adv
    logical, intent(in) :: reinit
    fire_print_msg = 0
    fire_print_file = 0
    fuel_left_method = 1
    fuel_left_irl = 2
    fuel_left_jrl = 2
    fire_const_time = -1.
    fire_const_grnhfx = 0.
    fire_const_grnqfx = 0.
    fire_atm_feedback = 1.
    boundary_guard = -1
    fire_grows_only = 1
    fire_upwinding = 9
    fire_upwind_split = 0
    fire_viscosity = 0.4
    fire_lfn_ext_up = 1.0
    fire_test_steps = 0
    fire_advection = adv
    fire_lsm_reinit = reinit
    fire_lsm_reinit_iter = 1
    fire_upwinding_reinit = 4
    fire_lsm_band_ngp = 4
    fire_viscosity_bg = 0.4
    fire_viscosity_band = 0.5
    fire_viscosity_ngp = 2
    fire_slope_factor = 1.0
  end subroutine set_registry_flags

  ! variant 1 moves every numeric switch the port exposes off its Registry
  ! value, so each one's arithmetic is held to WRF and not only to itself.
  subroutine set_variant(variant)
    integer, intent(in) :: variant
    if (variant == 1) then
      fire_viscosity = 0.2
      fire_viscosity_bg = 0.1
      fire_viscosity_band = 0.8
      fire_viscosity_ngp = 3
      fire_lsm_band_ngp = 5
      fire_slope_factor = 1.5
      fire_atm_feedback = 0.5
      fire_lsm_reinit_iter = 2
      fire_lfn_ext_up = 0.5
    end if
  end subroutine set_variant

  subroutine run_case(cname, kind, nxa, nya, ir, jr, dxa, dt, nsteps, nsub, &
                      adv, reinit, weight, nign, ign, ndump, dumps, variant)
    character(len=*), intent(in) :: cname
    integer, intent(in) :: kind, nxa, nya, ir, jr, nsteps, nsub, adv, nign
    integer, intent(in) :: variant
    integer, intent(in) :: ndump, dumps(ndump)
    logical, intent(in) :: reinit
    real, intent(in) :: dxa, dt, weight
    real, intent(in) :: ign(8, nign)  ! start x,y end x,y start,end time, r, ros

    type(fire_params) :: fp
    type(domain) :: grid
    type(ignition_line_type) :: line(nign)
    integer :: nx, ny, i, j, istep, isub, ig, ignited, k
    integer :: ifds, ifde, jfds, jfde, ifms, ifme, jfms, jfme
    real :: fdx, fdy, time_start, h, tbound, x, y, s
    real, allocatable, target, dimension(:, :) :: vx, vy, zsf, dzdxf, dzdyf, &
        bbb, betafl, phiwc, r_0, fgip, ischap, iboros, fuel_time, fmc_g
    real, allocatable, dimension(:, :) :: lfn, tign, lfn_0, lfn_1, lfn_2, &
        lfn_s0, lfn_s1, lfn_s2, lfn_s3, lfn_out, ros, fuel_frac, fire_area, &
        grnhfx, grnqfx, coord_xf, coord_yf
    real, allocatable, dimension(:, :) :: fuel_frac_end, fuel_frac_burnt
    real, allocatable, dimension(:, :) :: grnhfx_a, grnqfx_a, canhfx_a, &
        canqfx_a, avg_ff, mean_h, mean_q
    character(len=32) :: tag

    nx = nxa * ir
    ny = nya * jr
    ifds = 1; ifde = nx; jfds = 1; jfde = ny
    ifms = 0; ifme = nx + 1; jfms = 0; jfme = ny + 1
    fdx = dxa / ir
    fdy = dxa / jr
    call set_registry_flags(adv, reinit)
    call set_variant(variant)

    allocate(vx(ifms:ifme, jfms:jfme), vy(ifms:ifme, jfms:jfme), &
             zsf(ifms:ifme, jfms:jfme), dzdxf(ifms:ifme, jfms:jfme), &
             dzdyf(ifms:ifme, jfms:jfme), bbb(ifms:ifme, jfms:jfme), &
             betafl(ifms:ifme, jfms:jfme), phiwc(ifms:ifme, jfms:jfme), &
             r_0(ifms:ifme, jfms:jfme), fgip(ifms:ifme, jfms:jfme), &
             ischap(ifms:ifme, jfms:jfme), iboros(ifms:ifme, jfms:jfme), &
             fuel_time(ifms:ifme, jfms:jfme), fmc_g(ifms:ifme, jfms:jfme))
    allocate(lfn(ifms:ifme, jfms:jfme), tign(ifms:ifme, jfms:jfme), &
             lfn_0(ifms:ifme, jfms:jfme), lfn_1(ifms:ifme, jfms:jfme), &
             lfn_2(ifms:ifme, jfms:jfme), lfn_s0(ifms:ifme, jfms:jfme), &
             lfn_s1(ifms:ifme, jfms:jfme), lfn_s2(ifms:ifme, jfms:jfme), &
             lfn_s3(ifms:ifme, jfms:jfme), lfn_out(ifms:ifme, jfms:jfme), &
             ros(ifms:ifme, jfms:jfme), fuel_frac(ifms:ifme, jfms:jfme), &
             fire_area(ifms:ifme, jfms:jfme), grnhfx(ifms:ifme, jfms:jfme), &
             grnqfx(ifms:ifme, jfms:jfme), coord_xf(ifms:ifme, jfms:jfme), &
             coord_yf(ifms:ifme, jfms:jfme))
    allocate(fuel_frac_end(1:nx, 1:ny), fuel_frac_burnt(1:nx, 1:ny))
    allocate(grnhfx_a(nxa, nya), grnqfx_a(nxa, nya), canhfx_a(nxa, nya), &
             canqfx_a(nxa, nya), avg_ff(nxa, nya), mean_h(nxa, nya), &
             mean_q(nxa, nya))
    ! WRF's Registry fill: every state and work array starts at zero.
    vx = 0.; vy = 0.; zsf = 0.; dzdxf = 0.; dzdyf = 0.; bbb = 0.
    betafl = 0.; phiwc = 0.; r_0 = 0.; fgip = 0.; ischap = 0.; iboros = 0.
    fuel_time = 0.; fmc_g = 0.
    lfn = 0.; tign = 0.; lfn_0 = 0.; lfn_1 = 0.; lfn_2 = 0.; lfn_s0 = 0.
    lfn_s1 = 0.; lfn_s2 = 0.; lfn_s3 = 0.; lfn_out = 0.; ros = 0.
    fuel_frac = 0.; fire_area = 0.; grnhfx = 0.; grnqfx = 0.
    coord_xf = 0.; coord_yf = 0.

    ! the given fire_params arrays (no fuel model)
    do j = 1, ny
      do i = 1, nx
        x = (i - 0.5) * fdx
        y = (j - 0.5) * fdy
        select case (kind)
        case (KIND_ISO)
          r_0(i, j) = 59.0
          bbb(i, j) = 1.
          phiwc(i, j) = 0.
          betafl(i, j) = 1.
          fgip(i, j) = 0.5
          fmc_g(i, j) = 0.08
          fuel_time(i, j) = weight / 0.85
        case (KIND_WIND)
          vx(i, j) = 4.0 + 2.0 * sin(y / 90.)
          vy(i, j) = 1.5 * cos(x / 70.)
          dzdxf(i, j) = 0.15 * sin(x / 120.)
          dzdyf(i, j) = 0.1 * cos(y / 150.)
          r_0(i, j) = 20. + 10. * cos((x + y) / 200.)
          bbb(i, j) = 1.2 + 0.3 * sin(x / 50.)
          phiwc(i, j) = 1.5e-3 * (1. + 0.5 * cos(y / 60.))
          betafl(i, j) = 0.003 * (1. + 0.3 * sin((x - y) / 80.))
          fgip(i, j) = 0.4 + 0.3 * cos(x / 80.) * sin(y / 90.)
          fmc_g(i, j) = 0.08 + 0.05 * sin(x / 60.)
          fuel_time(i, j) = (weight + 23. * (1. + sin(y / 100.))) / 0.85
          if (x > 280. .and. x < 330. .and. y > 60. .and. y < 140.) then
            ! set_fire_params's no-fuel cell
            r_0(i, j) = 0.; fgip(i, j) = 0.; betafl(i, j) = 1.
            bbb(i, j) = 1.; phiwc(i, j) = 0.
          end if
        case (KIND_CAP)
          vx(i, j) = 9.0
          vy(i, j) = 3.0 * sin(x / 60.)
          dzdxf(i, j) = 0.3
          r_0(i, j) = 300.
          bbb(i, j) = 1.5
          phiwc(i, j) = 4.0e-4
          betafl(i, j) = 0.002
          fgip(i, j) = 1.2
          fmc_g(i, j) = 0.12
          fuel_time(i, j) = weight / 0.85
        end select
      end do
    end do
    fp%vx => vx; fp%vy => vy; fp%zsf => zsf; fp%dzdxf => dzdxf
    fp%dzdyf => dzdyf; fp%bbb => bbb; fp%betafl => betafl
    fp%phiwc => phiwc; fp%r_0 => r_0; fp%fgip => fgip; fp%ischap => ischap
    fp%iboros => iboros; fp%fuel_time => fuel_time; fp%fmc_g => fmc_g

    do ig = 1, nign
      line(ig)%ros = ign(8, ig)
      line(ig)%stop_time = 0.; line(ig)%wind_red = 0.
      line(ig)%wrdist = 0.; line(ig)%wrupwind = 0.
      line(ig)%start_x = ign(1, ig); line(ig)%start_y = ign(2, ig)
      line(ig)%end_x = ign(3, ig); line(ig)%end_y = ign(4, ig)
      line(ig)%start_time = ign(5, ig); line(ig)%end_time = ign(6, ig)
      line(ig)%radius = ign(7, ig)
    end do

    call oracle_open(cname)
    call oracle_put('nx', nx); call oracle_put('ny', ny)
    call oracle_put('nxa', nxa); call oracle_put('nya', nya)
    call oracle_put('ir', ir); call oracle_put('jr', jr)
    call oracle_put('dxa', dxa); call oracle_put('dt', dt)
    call oracle_put('nsteps', nsteps); call oracle_put('nsub', nsub)
    call oracle_put('advection', adv)
    call oracle_put('reinit', merge(1, 0, reinit))
    call oracle_put('options', (/ fire_viscosity, fire_viscosity_bg, &
                    fire_viscosity_band, real(fire_viscosity_ngp), &
                    real(fire_lsm_band_ngp), fire_slope_factor, &
                    fire_atm_feedback, real(fire_lsm_reinit_iter), &
                    fire_lfn_ext_up /))
    call oracle_put('ignitions', ign)
    call oracle_put('dumps', dumps)
    call oracle_put('in_vx', vx(1:nx, 1:ny))
    call oracle_put('in_vy', vy(1:nx, 1:ny))
    call oracle_put('in_dzdxf', dzdxf(1:nx, 1:ny))
    call oracle_put('in_dzdyf', dzdyf(1:nx, 1:ny))
    call oracle_put('in_r_0', r_0(1:nx, 1:ny))
    call oracle_put('in_bbb', bbb(1:nx, 1:ny))
    call oracle_put('in_phiwc', phiwc(1:nx, 1:ny))
    call oracle_put('in_betafl', betafl(1:nx, 1:ny))
    call oracle_put('in_fgip', fgip(1:nx, 1:ny))
    call oracle_put('in_fuel_time', fuel_time(1:nx, 1:ny))
    call oracle_put('in_fmc_g', fmc_g(1:nx, 1:ny))

    time_start = 0.
    call set_ideal_coord(fdx, fdy, ifds, ifde, jfds, jfde, ifms, ifme, &
                         jfms, jfme, ifds, ifde, jfds, jfde, coord_xf, coord_yf)
    call init_no_fire(ifds, ifde, jfds, jfde, ifms, ifme, jfms, jfme, &
                      ifds, ifde, jfds, jfde, fdx, fdy, time_start, &
                      fuel_frac, fire_area, lfn, tign)
    call oracle_put('init_lfn', lfn(1:nx, 1:ny))
    call oracle_put('init_tign', tign(1:nx, 1:ny))

    h = dt / real(nsub)
    k = 1
    do istep = 1, nsteps
      mean_h = 0.; mean_q = 0.
      do isub = 1, nsub
        ! pass 4
        call prop_ls_rk3(1, ifds, ifde, jfds, jfde, ifms, ifme, jfms, jfme, &
                         ifds, ifde, jfds, jfde, ifds, ifde, jfds, jfde, &
                         time_start, h, fdx, fdy, tbound, lfn, lfn_0, &
                         lfn_1, lfn_2, lfn_out, tign, ros, fp, grid, &
                         1, nxa, 1, 1, 1, nya, 1, nxa, 1, 1, 1, nya, &
                         1, nxa, 1, 1, 1, nya)
        call tign_update(ifds, ifde, jfds, jfde, ifms, ifme, jfms, jfme, &
                         ifds, jfds, ifde, jfde, time_start, h, lfn, &
                         lfn_out, tign)
        if (fire_lsm_reinit) then
          call reinit_ls_rk3(1, ifds, ifde, jfds, jfde, ifms, ifme, jfms, &
                             jfme, ifds, ifde, jfds, jfde, ifds, ifde, jfds, &
                             jfde, time_start, h, fdx, fdy, lfn, lfn_2, &
                             lfn_s0, lfn_s1, lfn_s2, lfn_s3, lfn_out, tign, &
                             grid, 1, nxa, 1, 1, 1, nya, 1, nxa, 1, 1, 1, &
                             nya, 1, nxa, 1, 1, 1, nya)
        end if
        ! pass 5
        do j = jfds, jfde
          do i = ifds, ifde
            lfn(i, j) = lfn_out(i, j)
          end do
        end do
        do ig = 1, nign
          call ignite_fire(ifds, ifde, jfds, jfde, ifms, ifme, jfms, jfme, &
                           ifds, ifde, jfds, jfde, line(ig), time_start, &
                           time_start + h, coord_xf, coord_yf, 1., 1., lfn, &
                           tign, ignited)
        end do
        ! pass 6
        call fuel_left(ifms, ifme, jfms, jfme, ifds, ifde, jfds, jfde, &
                       ifds, ifde, jfds, jfde, lfn, tign, fuel_time, &
                       time_start + h, fuel_frac_end, fire_area)
        do j = jfds, jfde
          do i = ifds, ifde
            fuel_frac_burnt(i, j) = fuel_frac(i, j) - fuel_frac_end(i, j)
            fuel_frac(i, j) = fuel_frac_end(i, j)
          end do
        end do
        call heat_fluxes(h, fp, ifms, ifme, jfms, jfme, ifds, ifde, jfds, &
                         jfde, ifds, ifde, jfds, jfde, fp%fgip, &
                         fuel_frac_burnt, grnhfx, grnqfx)
        ! the driver: sum_2d_cells and module_fr_fire_driver.F:941-953
        call sum_2d_cells(ifms, ifme, jfms, jfme, ifds, ifde, jfds, jfde, &
                          fuel_frac, 1, nxa, 1, nya, 1, nxa, 1, nya, avg_ff)
        call sum_2d_cells(ifms, ifme, jfms, jfme, ifds, ifde, jfds, jfde, &
                          grnhfx, 1, nxa, 1, nya, 1, nxa, 1, nya, grnhfx_a)
        call sum_2d_cells(ifms, ifme, jfms, jfme, ifds, ifde, jfds, jfde, &
                          grnqfx, 1, nxa, 1, nya, 1, nxa, 1, nya, grnqfx_a)
        s = 1. / (ir * jr)
        do j = 1, nya
          do i = 1, nxa
            avg_ff(i, j) = avg_ff(i, j) * s
            grnhfx_a(i, j) = fire_atm_feedback * grnhfx_a(i, j) * s
            grnqfx_a(i, j) = fire_atm_feedback * grnqfx_a(i, j) * s
            canhfx_a(i, j) = 0
            canqfx_a(i, j) = 0
          end do
        end do
        do j = 1, nya
          do i = 1, nxa
            mean_h(i, j) = mean_h(i, j) + grnhfx_a(i, j)
            mean_q(i, j) = mean_q(i, j) + grnqfx_a(i, j)
          end do
        end do
        time_start = time_start + h
      end do
      do j = 1, nya
        do i = 1, nxa
          mean_h(i, j) = mean_h(i, j) / real(nsub)
          mean_q(i, j) = mean_q(i, j) / real(nsub)
        end do
      end do
      if (k <= ndump) then
        if (dumps(k) == istep) then
          write(tag, '(A,I0,A)') 's', istep, '_'
          call oracle_put(trim(tag) // 'time', time_start)
          call oracle_put(trim(tag) // 'tbound', tbound)
          call oracle_put(trim(tag) // 'lfn', lfn(1:nx, 1:ny))
          call oracle_put(trim(tag) // 'tign', tign(1:nx, 1:ny))
          call oracle_put(trim(tag) // 'ros', ros(1:nx, 1:ny))
          call oracle_put(trim(tag) // 'fire_area', fire_area(1:nx, 1:ny))
          call oracle_put(trim(tag) // 'fuel_frac', fuel_frac(1:nx, 1:ny))
          call oracle_put(trim(tag) // 'fuel_frac_burnt', fuel_frac_burnt)
          call oracle_put(trim(tag) // 'fgrnhfx', grnhfx(1:nx, 1:ny))
          call oracle_put(trim(tag) // 'fgrnqfx', grnqfx(1:nx, 1:ny))
          call oracle_put(trim(tag) // 'GRNHFX', grnhfx_a)
          call oracle_put(trim(tag) // 'GRNQFX', grnqfx_a)
          call oracle_put(trim(tag) // 'CANHFX', canhfx_a)
          call oracle_put(trim(tag) // 'CANQFX', canqfx_a)
          call oracle_put(trim(tag) // 'AVG_FUEL_FRAC', avg_ff)
          call oracle_put(trim(tag) // 'mean_GRNHFX', mean_h)
          call oracle_put(trim(tag) // 'mean_GRNQFX', mean_q)
          k = k + 1
        end if
      end if
    end do
    call oracle_close()
  end subroutine run_case

end module fire_cases

program run_fire
  use fire_cases
  use oracle_io
  implicit none
  character(len=1024) :: root
  real :: one(8, 1), two(8, 2)
  real :: weights(4) = (/ 7., 30., 180., 900. /)
  integer :: w
  character(len=32) :: name

  call oracle_root(root)

  ! a point, isotropic spread, Anderson's grass weight
  one(:, 1) = (/ 200., 200., 200., 200., 0., 0., 15., 1. /)
  call run_case('point_iso', KIND_ISO, 5, 5, 8, 8, 80., 2.0, 60, 1, 1, &
                .true., 7., 1, one, 3, (/ 1, 10, 60 /), 0)

  ! a line walked from inside the ENO1 edge band, in a varying wind on a
  ! varying slope, running into the no-fuel patch and out of the mesh
  one(:, 1) = (/ 30., 60., 250., 120., 3., 30., 12., 0.5 /)
  call run_case('line_wind', KIND_WIND, 6, 6, 8, 8, 80., 1.5, 140, 1, 1, &
                .true., 7., 1, one, 5, (/ 1, 20, 60, 100, 140 /), 0)

  ! the same fire in 6 s outer steps of four 1.5 s fire steps each
  call run_case('substep', KIND_WIND, 6, 6, 8, 8, 80., 6.0, 35, 4, 1, &
                .true., 7., 1, one, 3, (/ 1, 10, 35 /), 0)

  ! fire_advection = 0, no reinitialization, two ignitions (one later,
  ! held for 5 s)
  two(:, 1) = (/ 150., 150., 150., 150., 0., 0., 10., 2. /)
  two(:, 2) = (/ 330., 330., 330., 330., 20., 25., 10., 2. /)
  call run_case('options', KIND_WIND, 6, 6, 8, 8, 80., 1.5, 50, 1, 0, &
                .false., 7., 2, two, 4, (/ 1, 14, 17, 50 /), 0)

  ! fire_ros's 6 m/s cap taken out of the wind and slope parts
  one(:, 1) = (/ 120., 200., 120., 200., 0., 0., 15., 5. /)
  call run_case('cap', KIND_CAP, 5, 5, 8, 8, 80., 0.4, 25, 1, 1, &
                .true., 7., 1, one, 3, (/ 1, 10, 25 /), 0)

  ! a non-square mesh with dx /= dy and every numeric switch off its
  ! Registry value (set_variant): viscosity, its band and ramp, the WENO
  ! band, the slope factor, the feedback, two reinit iterations, and a
  ! half-biased boundary extrapolation
  one(:, 1) = (/ 40., 50., 300., 260., 2., 25., 12., 0.5 /)
  call run_case('asym', KIND_WIND, 6, 4, 8, 10, 80., 0.6, 120, 1, 1, &
                .true., 7., 1, one, 4, (/ 1, 30, 80, 120 /), 1)

  ! the fuel_left quadrature across WRF's fuel weights (grass to timber)
  one(:, 1) = (/ 200., 200., 200., 200., 0., 0., 15., 1. /)
  do w = 1, 4
    write(name, '(A,I0)') 'weight_', nint(weights(w))
    call run_case(trim(name), KIND_ISO, 5, 5, 8, 8, 80., 2.0, 40, 1, 1, &
                  .true., weights(w), 1, one, 2, (/ 20, 40 /), 0)
  end do
end program run_fire
