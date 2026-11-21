! Drive the MPAS-Model v8.4.2 sea ice under E3SM v3.2.0's sea-ice coupling:
! the hybrid "E3SM v3.2.0 coupler with MPAS v8.4.2 components, not an E3SM
! configuration" (woof.hex.coupled.hybrid_seaice).
!
!   run_hybrid_seaice START.bin COUPLING.bin OUTPUT.bin LOG
!
! START.bin is a v8.4.2 sea-ice start in oracle_stream.F90's layout
! (woof.verify.e3sm_hybrid_seaice_oracle): the column oracle's seaice_step
! pools (tools/mpas_seaice_oracle/run_column.F90) plus what E3SM's import
! and export look up that v8.4.2 lacks: frazilMassAdjust (a zero array), the
! nFrequencies dimension (0), the configs config_column_physics_type,
! config_use_data_icebergs, config_use_floe_size_distribution and
! config_use_column_waves, and the mesh's sphere_radius.
!
! COUPLING.bin (little-endian): ice_cpl_dt, nWindows, lsize, nx2i, ni2x
! (int32); the x2i and i2x field lists (int32 length, characters); nextsw_cday
! at start and per window (nWindows + 1, r8); then nWindows x2i arrays
! (nx2i, lsize) as r8.
!
! OUTPUT.bin: i2x after ice_init_mct's export, then after each window's
! (ni2x, lsize) as r8; then the rotation's words (8, nCells): per cell sin
! and cos of latCell, lonCell and the rotated grid's latitude and longitude
! (v8.4.2's seaice_grid_rotation_forward and seaice_latlon_from_xyz; glibc's
! sin, cos, asin and atan2).  OUTPUT.bin.pools: every field's words
! (oracle_stream's write_stream).  LOG: the column's log, then 'TOBJ IERR n'
! and 'TOBJ STEP w s' (the window and step it stopped at).
!
! The coupler's side, ice_import_mct, ice_export_mct, basal_pressure and
! frazil_mass, is cut verbatim out of E3SM's
! components/mpas-seaice/driver/ice_comp_mct.F (build.sh) and compiled
! against v8.4.2's modules (hybrid_seaice_shims.F90: the constants and
! Icepack's three functions from v8.4.2's column package; hybrid_seaice_mesh:
! v8.4.2's rotation routines, verbatim).  The component's side is v8.4.2, as
! run_column.F90's seaice_step stage sets it up (the column package's
! parameters, colpkg_init_thermo, init_column_itd, the orbit, the time steps,
! seaice_init_evp, seaice_init_advection, seaice_initialize_time_diagnostics),
! in the order E3SM's driver calls it:
!
!   ice_init_mct (ice_comp_mct.F:659-663, 780-930; runtype 'initial'):
!   nextsw_cday into dayOfNextShortwaveCalculation, the core's init, the
!   coupling preparation (v8.4.2's seaice_column_coupling_prep where E3SM
!   calls seaice_icepack_coupling_prep), the export, the import of the
!   still-zero x2i;
!
!   ice_run_mct (ice_comp_mct.F:1147-1417), per window: the fluxes
!   re-initialised (v8.4.2's seaice_column_reinitialize_fluxes for E3SM's
!   seaice_icepack_reinitialize_fluxes), the import, nextsw_cday, v8.4.2's
!   post_atmospheric_coupling and post_oceanic_coupling (cut verbatim out of
!   mpas_seaice_forcing.F), the step counter reset; until the alarm rings: the
!   clock advanced, on the run's first step seaice_init_post_clock_advance
!   (cut verbatim out of mpas_seaice_initialize.F), seaice_timestep with no
!   seaice_timestep_finalize; the export.
!
! woof.hex.coupled.hybrid_seaice.DIVERGENCES lists what E3SM's loop does
! beyond that, each ported, refused by name or left out with its reason.
program run_hybrid_seaice
  use mpas_kind_types
  use mpas_derived_types
  use mpas_pool_routines
  use mpas_log, only: stub_log_warn_unit, stub_log_err_unit
  use mpas_timekeeping
  use seaice_column
  use ice_colpkg, only: colpkg_init_thermo, colpkg_init_orbit
  use seaice_velocity_solver_constitutive_relation, only: seaice_init_evp
  use seaice_diagnostics, only: seaice_initialize_time_diagnostics
  use seaice_advection, only: seaice_init_advection
  use seaice_time_integration, only: seaice_timestep
  use oracle_stream
  use seq_flds_mod, only: seq_flds_x2i_fields, seq_flds_i2x_fields
  use mpassi_cpl_indices, only: mpassi_cpl_indices_set
  use ice_comp_mct_cut, only: ice_import_mct, ice_export_mct, domain
  use hybrid_seaice_forcing, only: post_atmospheric_coupling, post_oceanic_coupling, &
                                   seaice_init_post_clock_advance
  use hybrid_seaice_mesh, only: seaice_grid_rotation_forward, seaice_latlon_from_xyz
  implicit none
  type(oracle_pools) :: pools
  type(oracle_field), allocatable :: fields(:)
  real(kind=RKIND) :: inputDt
  character(len=StrKIND), pointer :: config_start_time, clockStep, config_calendar_type, shortwaveType
  logical :: orbitAbort, first
  character(len=StrKIND) :: orbitMessage
  integer :: p, logUnit, ierr, uc, uo, n, lsize, nx2i, ni2x, nWindows, ice_cpl_dt, nSteps, w, s, itimestep, iCell
  integer, pointer :: nSubcycles, nElasticSubcycles, nIceLayers, nCells
  real(kind=RKIND), pointer :: config_dt, dynamicsTimeStep, elasticTimeStep, dayOfNextShortwaveCalculation, radius
  logical, pointer :: useShortwave, rotate
  real(kind=RKIND), allocatable :: sprofile(:), x2i(:,:,:), i2x(:,:), x2iZero(:,:), nextsw(:), trig(:,:)
  real(kind=RKIND), dimension(:), pointer :: latCell, lonCell, xCell, yCell, zCell
  real(kind=RKIND) :: xp, yp, zp, latp, lonp
  type(MPAS_pool_type), pointer :: meshPool
  character(len=:), allocatable :: rlist
  character(len=1024) :: inPath, cplPath, outPath, logPath

  call get_command_argument(1, inPath)
  call get_command_argument(2, cplPath)
  call get_command_argument(3, outPath)
  call get_command_argument(4, logPath)
  call read_stream(inPath, inputDt, pools, fields)
  open(newunit=logUnit, file=trim(logPath), status='replace', action='write')
  stub_log_warn_unit = logUnit
  stub_log_err_unit = logUnit

  open(newunit=uc, file=trim(cplPath), access='stream', form='unformatted', status='old')
  read(uc) ice_cpl_dt, nWindows, lsize, nx2i, ni2x
  read(uc) n
  allocate(character(len=n) :: rlist)
  read(uc) rlist
  seq_flds_x2i_fields = rlist
  deallocate(rlist)
  read(uc) n
  allocate(character(len=n) :: rlist)
  read(uc) rlist
  seq_flds_i2x_fields = rlist
  deallocate(rlist)
  allocate(nextsw(0:nWindows), x2i(nx2i, lsize, nWindows), i2x(ni2x, lsize), x2iZero(nx2i, lsize))
  read(uc) nextsw
  read(uc) x2i
  close(uc)
  call mpassi_cpl_indices_set()

  ! one block, as run_column.F90 sets it up; the cap's module-level domain
  allocate(domain)
  allocate(domain % blocklist)
  allocate(domain % blocklist % structs)
  allocate(domain % clock)
  allocate(domain % dminfo)
  domain % configs => pools % configs
  domain % packages => pools % packages
  domain % blocklist % configs => pools % configs
  domain % blocklist % dimensions => pool_named(pools, 'mesh')
  domain % blocklist % packages => pools % packages
  do p = 1, size(poolNames)
     call stub_pool_put_subpool(domain % blocklist % structs, trim(poolNames(p)), pool_named(pools, trim(poolNames(p))))
  end do
  call stub_pool_put_subpool(domain % blocklist % structs, 'alternative_atmos_forcing', &
       pool_named(pools, 'alternative_atmos_forcin'))
  meshPool => pool_named(pools, 'mesh')

  ! ice_init_mct: nextsw_cday (659-663), before the core's init
  call mpas_pool_get_array(pool_named(pools, 'shortwave'), 'dayOfNextShortwaveCalculation', &
       dayOfNextShortwaveCalculation)
  dayOfNextShortwaveCalculation = nextsw(0)

  ! the core's init and the clock at the start (EClock's time is config_start_time)
  call mpas_pool_get_config(pools % configs, 'config_calendar_type', config_calendar_type)
  call mpas_timekeeping_init(config_calendar_type)
  call mpas_pool_get_config(pools % configs, 'config_start_time', config_start_time)
  call mpas_pool_get_config(pools % configs, 'oracle_clock_step', clockStep)
  call mpas_set_clock(domain % clock, config_start_time, clockStep)
  call seaice_init_column_physics_package_parameters(domain)
  call mpas_pool_get_dimension(meshPool, 'nIceLayers', nIceLayers)
  allocate(sprofile(nIceLayers + 1))
  call colpkg_init_thermo(nIceLayers, sprofile)
  call init_column_itd(domain)
  call mpas_pool_get_config(pools % configs, 'config_shortwave_type', shortwaveType)
  call mpas_pool_get_config(pools % configs, 'config_use_column_shortwave', useShortwave)
  if (useShortwave .and. trim(shortwaveType) == 'dEdd') then
     call colpkg_init_orbit(orbitAbort, orbitMessage)
     if (orbitAbort) then
        print *, 'run_hybrid_seaice: colpkg_init_orbit: ', trim(orbitMessage)
        stop 3
     end if
  end if
  call mpas_pool_get_config(pools % configs, 'config_dt', config_dt)
  call mpas_pool_get_config(pools % configs, 'config_dynamics_subcycle_number', nSubcycles)
  call mpas_pool_get_config(pools % configs, 'config_elastic_subcycle_number', nElasticSubcycles)
  call mpas_pool_get_array(pool_named(pools, 'velocity_solver'), 'dynamicsTimeStep', dynamicsTimeStep)
  call mpas_pool_get_array(pool_named(pools, 'velocity_solver'), 'elasticTimeStep', elasticTimeStep)
  dynamicsTimeStep = config_dt / real(nSubcycles,RKIND)
  elasticTimeStep = dynamicsTimeStep / real(nElasticSubcycles,RKIND)
  call seaice_init_evp(domain)
  call seaice_init_advection(domain)
  call seaice_initialize_time_diagnostics(domain)

  ! the coupling alarm (797-841): ice_cpl_dt a whole number of time steps
  if (nint(config_dt) <= 0 .or. ice_cpl_dt < nint(config_dt) .or. mod(ice_cpl_dt, nint(config_dt)) /= 0) stop 14
  nSteps = ice_cpl_dt / nint(config_dt)

  ! the coupling preparation (844-856), the export (895) into i2x as mct_aVect_zero left it,
  ! the import of the zeroed x2i (917)
  call seaice_column_coupling_prep(domain)
  open(newunit=uo, file=trim(outPath), access='stream', form='unformatted', status='replace')
  i2x = 0.0_RKIND
  call ice_export_mct(i2x, ierr)
  write(uo) i2x
  x2iZero = 0.0_RKIND
  call ice_import_mct(x2iZero, ierr)

  ! ice_run_mct, per window
  first = .true.
  ierr = 0
  windows: do w = 1, nWindows
     call seaice_column_reinitialize_fluxes(domain)
     call ice_import_mct(x2i(:, :, w), ierr)
     dayOfNextShortwaveCalculation = nextsw(w)
     call post_atmospheric_coupling(domain % blocklist)
     call post_oceanic_coupling(domain % blocklist)
     itimestep = 0
     do s = 1, nSteps
        itimestep = itimestep + 1
        call mpas_advance_clock(domain % clock)
        if (first) then
           call seaice_init_post_clock_advance(domain, domain % clock)
           first = .false.
        end if
        call seaice_timestep(domain, domain % clock, itimestep, ierr)
        if (ierr > 0) exit windows
     end do
     call ice_export_mct(i2x, ierr)
     write(uo) i2x
  end do windows
  write(logUnit, '(a,1x,i0)') 'TOBJ IERR', ierr
  write(logUnit, '(a,2(1x,i0))') 'TOBJ STEP', min(w, nWindows), itimestep

  ! the rotation's words
  call mpas_pool_get_dimension(meshPool, 'nCells', nCells)
  call mpas_pool_get_config(pools % configs, 'config_rotate_cartesian_grid', rotate)
  call mpas_pool_get_config(meshPool, 'sphere_radius', radius)
  call mpas_pool_get_array(meshPool, 'latCell', latCell)
  call mpas_pool_get_array(meshPool, 'lonCell', lonCell)
  call mpas_pool_get_array(meshPool, 'xCell', xCell)
  call mpas_pool_get_array(meshPool, 'yCell', yCell)
  call mpas_pool_get_array(meshPool, 'zCell', zCell)
  allocate(trig(8, nCells))
  do iCell = 1, nCells
     call seaice_grid_rotation_forward(xp, yp, zp, xCell(iCell), yCell(iCell), zCell(iCell), rotate)
     call seaice_latlon_from_xyz(latp, lonp, xp, yp, zp, radius)
     trig(:, iCell) = [sin(latCell(iCell)), cos(latCell(iCell)), sin(lonCell(iCell)), cos(lonCell(iCell)), &
                       sin(latp), cos(latp), sin(lonp), cos(lonp)]
  end do
  write(uo) trig
  close(uo)
  call write_stream(trim(outPath) // '.pools', fields)
  close(logUnit)
end program run_hybrid_seaice
