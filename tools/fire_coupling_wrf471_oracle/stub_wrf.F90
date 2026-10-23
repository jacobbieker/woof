! Service-only stubs for the standalone WRF v4.7.1 fire-coupling oracle.
!
! NOTHING here computes a physical quantity.  The fire driver
! (module_fr_fire_driver.F, module_fr_fire_driver_wrf.F) reads its inputs
! from WRF's TYPE(domain) and TYPE(grid_config_rec_type), which in a WRF
! build are generated from the Registry.  Here they are declared by hand,
! holding exactly the members the pinned fire sources reference, with the
! names, ranks and Registry defaults of Registry/registry.fire (pinned).
! The arrays are POINTERs, as in WRF's generated type, so the driver's
! ``fp%vx => grid%uf`` aliases them.  get_ijk_from_subgrid is
! frame/module_domain.F:287-326's arithmetic on the stub's own atmospheric
! bounds.  The rest is what `nm -u` reports undefined outside libm and
! libgfortran: the single-rank shims and the WRF abort.

module module_domain
  implicit none
  type domain
    ! atmospheric bounds (WRF's get_ijk_from_grid); single patch, single tile
    integer :: ids, ide, jds, jde, kds, kde
    integer :: ims, ime, jms, jme, kms, kme
    integer :: ips, ipe, jps, jpe, kps, kpe
    integer :: itimestep = 0, sr_x = 0, sr_y = 0, num_tiles = 1
    real :: dt = 0., dx = 0., dy = 0., u_frame = 0., v_frame = 0.
    real :: fmoist_lasttime = 0., fmoist_nexttime = 0.
    integer, pointer :: i_start(:) => null(), i_end(:) => null()
    integer, pointer :: j_start(:) => null(), j_end(:) => null()
    ! atmosphere, (ims:ime, kms:kme, jms:jme)
    real, pointer, dimension(:,:,:) :: u_2 => null(), v_2 => null(), &
        ph_2 => null(), phb => null(), rthfrten => null(), rqvfrten => null()
    real, pointer :: tracer(:,:,:,:) => null()
    ! atmosphere, (ims:ime, jms:jme)
    real, pointer, dimension(:,:) :: z0 => null(), ht => null(), &
        xlong => null(), xlat => null(), avg_fuel_frac => null(), &
        grnhfx => null(), grnqfx => null(), canhfx => null(), &
        canqfx => null(), grnhfx_fu => null(), grnqfx_fu => null(), &
        uah => null(), vah => null(), mut => null(), rainc => null(), &
        rainnc => null(), t2 => null(), q2 => null(), psfc => null(), &
        rain_old => null(), t2_old => null(), q2_old => null(), &
        psfc_old => null(), rh_fire => null()
    ! moisture-model classes, (ims:ime, n, jms:jme)
    real, pointer, dimension(:,:,:) :: fmc_gc => null(), fmep => null(), &
        fmc_equi => null(), fmc_lag => null()
    ! hybrid coordinate weights, (kms:kme)
    real, pointer, dimension(:) :: c1h => null(), c2h => null()
    ! fire mesh, (ifms:ifme, jfms:jfme)
    real, pointer, dimension(:,:) :: uf => null(), vf => null(), &
        zsf => null(), dzdxf => null(), dzdyf => null(), bbb => null(), &
        betafl => null(), phiwc => null(), r_0 => null(), fgip => null(), &
        ischap => null(), iboros => null(), fmc_g => null(), lfn => null(), &
        lfn_hist => null(), lfn_0 => null(), lfn_1 => null(), &
        lfn_2 => null(), lfn_s0 => null(), lfn_s1 => null(), &
        lfn_s2 => null(), lfn_s3 => null(), flame_length => null(), &
        ros_front => null(), tign_g => null(), fuel_frac => null(), &
        fire_area => null(), burnt_area_dt => null(), fgrnhfx => null(), &
        fgrnqfx => null(), fcanhfx => null(), fcanqfx => null(), &
        ros => null(), fxlong => null(), fxlat => null(), fz0 => null(), &
        nfuel_cat => null(), fuel_time => null()
  end type domain
contains
  ! frame/module_domain.F:287-326, on the stub's atmospheric bounds.
  subroutine get_ijk_from_subgrid(grid, ids0, ide0, jds0, jde0, kds0, kde0, &
                                  ims0, ime0, jms0, jme0, kms0, kme0, &
                                  ips0, ipe0, jps0, jpe0, kps0, kpe0)
    type(domain), intent(in) :: grid
    integer, intent(out) :: ids0, ide0, jds0, jde0, kds0, kde0, &
                            ims0, ime0, jms0, jme0, kms0, kme0, &
                            ips0, ipe0, jps0, jpe0, kps0, kpe0
    ids0 = grid%ids
    ide0 = grid%ide * grid%sr_x
    ims0 = (grid%ims - 1) * grid%sr_x + 1
    ime0 = grid%ime * grid%sr_x
    ips0 = (grid%ips - 1) * grid%sr_x + 1
    ipe0 = grid%ipe * grid%sr_x
    jds0 = grid%jds
    jde0 = grid%jde * grid%sr_y
    jms0 = (grid%jms - 1) * grid%sr_y + 1
    jme0 = grid%jme * grid%sr_y
    jps0 = (grid%jps - 1) * grid%sr_y + 1
    jpe0 = grid%jpe * grid%sr_y
    kds0 = grid%kds
    kde0 = grid%kde
    kms0 = grid%kms
    kme0 = grid%kme
    kps0 = grid%kps
    kpe0 = grid%kpe
  end subroutine get_ijk_from_subgrid
end module module_domain

! Registry/registry.fire's &fire defaults (and the three non-fire members
! the driver reads: restart, tracer_opt, cen_lat/cen_lon).
module module_configure
  implicit none
  type grid_config_rec_type
    logical :: restart = .false.
    integer :: tracer_opt = 0
    real :: cen_lat = 0., cen_lon = 0.
    integer :: nfmc = 5
    logical :: fmoist_run = .false., fmoist_interp = .false.
    logical :: fmoist_only = .false.
    integer :: fmoist_freq = 0
    real :: fmoist_dt = 600., fmep_decay_tlag = 999999.
    integer :: ifire = 0, fire_boundary_guard = 8, fire_num_ignitions = 0
    real :: fire_ignition_ros1 = 0.01, fire_ignition_ros2 = 0.01, &
            fire_ignition_ros3 = 0.01, fire_ignition_ros4 = 0.01, &
            fire_ignition_ros5 = 0.01
    real :: fire_ignition_start_lon1 = 0., fire_ignition_start_lat1 = 0., &
            fire_ignition_end_lon1 = 0., fire_ignition_end_lat1 = 0., &
            fire_ignition_radius1 = 0., fire_ignition_start_time1 = 0., &
            fire_ignition_end_time1 = 0.
    real :: fire_ignition_start_lon2 = 0., fire_ignition_start_lat2 = 0., &
            fire_ignition_end_lon2 = 0., fire_ignition_end_lat2 = 0., &
            fire_ignition_radius2 = 0., fire_ignition_start_time2 = 0., &
            fire_ignition_end_time2 = 0.
    real :: fire_ignition_start_lon3 = 0., fire_ignition_start_lat3 = 0., &
            fire_ignition_end_lon3 = 0., fire_ignition_end_lat3 = 0., &
            fire_ignition_radius3 = 0., fire_ignition_start_time3 = 0., &
            fire_ignition_end_time3 = 0.
    real :: fire_ignition_start_lon4 = 0., fire_ignition_start_lat4 = 0., &
            fire_ignition_end_lon4 = 0., fire_ignition_end_lat4 = 0., &
            fire_ignition_radius4 = 0., fire_ignition_start_time4 = 0., &
            fire_ignition_end_time4 = 0.
    real :: fire_ignition_start_lon5 = 0., fire_ignition_start_lat5 = 0., &
            fire_ignition_end_lon5 = 0., fire_ignition_end_lat5 = 0., &
            fire_ignition_radius5 = 0., fire_ignition_start_time5 = 0., &
            fire_ignition_end_time5 = 0.
    real :: fire_ignition_start_x1 = 0., fire_ignition_start_y1 = 0., &
            fire_ignition_end_x1 = 0., fire_ignition_end_y1 = 0.
    real :: fire_ignition_start_x2 = 0., fire_ignition_start_y2 = 0., &
            fire_ignition_end_x2 = 0., fire_ignition_end_y2 = 0.
    real :: fire_ignition_start_x3 = 0., fire_ignition_start_y3 = 0., &
            fire_ignition_end_x3 = 0., fire_ignition_end_y3 = 0.
    real :: fire_ignition_start_x4 = 0., fire_ignition_start_y4 = 0., &
            fire_ignition_end_x4 = 0., fire_ignition_end_y4 = 0.
    real :: fire_ignition_start_x5 = 0., fire_ignition_start_y5 = 0., &
            fire_ignition_end_x5 = 0., fire_ignition_end_y5 = 0.
    real :: fire_crwn_hgt = 15., fire_ext_grnd = 50., fire_ext_crwn = 50.
    integer :: fire_sfc_flx = 0
    real :: fire_heat_peak = 0., fire_tg_ub = 1000.
    integer :: fire_smk_scheme = 0
    real :: fire_smk_peak = 0., fire_smk_ext = 50.
    real :: fire_wind_height = 6.096
    integer :: fire_fuel_read = -1, fire_fuel_cat = 1, fire_fmc_read = 1
    integer :: fire_print_msg = 0, fire_print_file = 0
    integer :: fire_fuel_left_method = 1, fire_fuel_left_irl = 2, &
               fire_fuel_left_jrl = 2, fire_grows_only = 1, &
               fire_upwinding = 9, fire_upwind_split = 0
    real :: fire_viscosity = 0.4, fire_lfn_ext_up = 1.0
    integer :: fire_advection = 1, fire_test_steps = 0
    real :: fire_const_time = -1., fire_const_grnhfx = 0., &
            fire_const_grnqfx = 0., fire_atm_feedback = 1.
    logical :: fire_lsm_reinit = .true.
    integer :: fire_lsm_reinit_iter = 1, fire_upwinding_reinit = 4
    logical :: fire_is_real_perim = .false.
    integer :: fire_lsm_band_ngp = 4
    logical :: fire_lsm_zcoupling = .false.
    real :: fire_lsm_zcoupling_ref = 50., fire_tracer_smoke = 0.02
    real :: fire_viscosity_bg = 0.4, fire_viscosity_band = 0.5
    integer :: fire_viscosity_ngp = 2
    real :: fire_slope_factor = 1.0
  end type grid_config_rec_type
end module module_configure

! The tracer index of fire smoke (tracer_opt = 3 adds fire_smoke as the first
! tracer after the Registry's PARAM_FIRST_SCALAR = 2).
module module_state_description
  implicit none
  integer, parameter :: param_first_scalar = 2
  integer, parameter :: num_tracer = 2, p_fire_smoke = 2
end module module_state_description

! One rank: the maximum over ranks is the rank's own value.
module module_dm
  implicit none
  interface wrf_dm_maxval
    module procedure wrf_dm_maxval_integer
  end interface wrf_dm_maxval
contains
  subroutine wrf_dm_maxval_integer(val, idex, jdex)
    integer, intent(inout) :: val, idex, jdex
  end subroutine wrf_dm_maxval_integer
end module module_dm

subroutine wrf_abort()
  implicit none
  error stop 1
end subroutine wrf_abort

subroutine wrf_debug(level, msg)
  implicit none
  integer, intent(in) :: level
  character(len=*), intent(in) :: msg
end subroutine wrf_debug

logical function wrf_dm_on_monitor()
  implicit none
  wrf_dm_on_monitor = .true.
end function wrf_dm_on_monitor

subroutine wrf_dm_bcast_real(values, n)
  real :: values
  integer, intent(in) :: n
end subroutine wrf_dm_bcast_real

subroutine wrf_dm_bcast_integer(values, n)
  integer :: values
  integer, intent(in) :: n
end subroutine wrf_dm_bcast_integer

subroutine wrf_dm_bcast_string(value, n)
  character(len=*) :: value
  integer, intent(in) :: n
end subroutine wrf_dm_bcast_string

subroutine wrf_get_nproc(nproc)
  implicit none
  integer, intent(out) :: nproc
  nproc = 1
end subroutine wrf_get_nproc

subroutine wrf_get_myproc(myproc)
  implicit none
  integer, intent(out) :: myproc
  myproc = 0
end subroutine wrf_get_myproc
