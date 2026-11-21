! Drive the MPAS-Model v8.4.2 ocean under E3SM v3.2.0's ocean coupling: the
! hybrid "E3SM v3.2.0 coupler with MPAS v8.4.2 components, not an E3SM
! configuration" (woof.hex.coupled.hybrid_ocean).
!
!   run_hybrid_ocean FORWARD.bin COUPLING.bin OUTPUT.bin
!
! FORWARD.bin is a v8.4.2 forward-mode start in oracle_stream.F90's layout
! (woof.verify.e3sm_hybrid_ocean_oracle): the forward oracle's pools plus
! the fields E3SM's import and export look up.  A field v8.4.2's Registry
! lacks (the iceberg fluxes, the removed runoff, the waves, the 10 m wind)
! is an absent field: the import finds it inactive and drops it.
! avgTotalFreshWaterTemperatureFlux, which v8.4.2 lacks too and the export
! reads unconditionally, is a zero array.
!
! COUPLING.bin (little-endian): ocn_cpl_dt, nWindows, lsize, nx2o, no2x
! (int32); the x2o and o2x field lists (int32 length, characters); then
! nWindows x2o arrays (nx2o, lsize) as r8.
!
! OUTPUT.bin: o2x after ocn_init_mct's export, then after each window's
! (no2x, lsize) as r8; then every field's words (oracle_stream's
! write_stream).
!
! The coupler's side, ocn_import_mct and ocn_export_mct, is cut verbatim out
! of E3SM's components/mpas-ocean/driver/ocn_comp_mct.F (build.sh) and
! compiled against the component's modules: v8.4.2's ocn_constants (with or
! without MPAS_ESM_SHR_CONST: two executables), ocn_equation_of_state (its
! ocn_freezing_temperature) and mpas_constants.  ocn_config and
! ocn_diagnostics_variables are ocean_standins.F90's; the land-ice boundary
! layer pointers are pointed at v8.4.2's diagnostics pool, where v8.4.2 keeps
! those arrays.  The component's side is v8.4.2, as
! tools/mpas_ocean_oracle/run_forward.F90 drives it (its init, verbatim), in
! the order E3SM's driver calls it:
!
!   ocn_init_mct (ocn_comp_mct.F:781-791, 833-846, 908, 992; runtype
!   'initial'): the absorbed fractions, frazil and land-ice arrays built; the
!   clock advanced to the coupling alarm with no step; the coupler's means
!   reset and accumulated once (v8.4.2's ocn_time_average_coupled_init and
!   _accumulate); the export; the import of the still-zero x2o;
!
!   ocn_run_mct (ocn_comp_mct.F:1139-1411), per window: the import; the
!   means reset; until the alarm rings: the absorbed fractions, frazil, the
!   land-ice fluxes (E3SM's order), the clock advanced, ocn_timestep, the
!   state's time levels shifted; the export.
!
! What E3SM's loop does beyond that (the land-ice flux accumulation, the
! mixed-layer depth, GM and the submesoscale eddies, the scaled ice-shelf
! melt, the precipitation scaling, analysis, streams, validation) either does
! nothing under the configurations the hybrid accepts or is refused by name;
! woof.hex.coupled.hybrid_ocean.DIVERGENCES lists each with its reason.
program run_hybrid_ocean
  use mpas_kind_types
  use mpas_derived_types
  use mpas_pool_routines
  use mpas_timekeeping
  use ocn_constants
  use ocn_tendency
  use ocn_forcing
  use ocn_surface_bulk_forcing
  use ocn_thick_hadv
  use ocn_thick_vadv
  use ocn_thick_surface_flux
  use ocn_vel_coriolis
  use ocn_vel_vadv
  use ocn_vel_pressure_grad
  use ocn_vel_hmix
  use ocn_vel_forcing
  use ocn_tracer_advection
  use ocn_tracer_hmix
  use ocn_tracer_hmix_redi
  use ocn_tracer_surface_flux_to_tend
  use ocn_tracer_short_wave_absorption
  use ocn_equation_of_state
  use ocn_diagnostics
  use ocn_vmix
  use ocn_frazil_forcing
  use ocn_surface_land_ice_fluxes
  use ocn_time_average_coupled
  use ocn_split_step_cut
  use ocn_timestep_cut
  use oracle_stream
  use seq_flds_mod, only: seq_flds_x2o_fields, seq_flds_o2x_fields
  use glc_zocnclass_mod, only: glc_zocnclass_init
  use mpaso_cpl_indices, only: mpaso_cpl_indices_set
  use ocn_diagnostics_variables, only: landIceBoundaryLayerTracers, landIceTracerTransferVelocities, &
                                       indexBLT, indexBLS, indexHeatTrans, indexSaltTrans
  use ocn_comp_mct_cut, only: ocn_import_mct, ocn_export_mct, domain, infodata, ocn_cpl_dt
  use ocn_config, only: config_frazil_heat_of_fusion
  implicit none
  type(oracle_pools) :: pools
  type(oracle_field), allocatable :: fields(:)
  type(MPAS_Time_type) :: currTime
  type(MPAS_TimeInterval_type) :: timeStep
  type(MPAS_pool_type), pointer :: meshPool, statePool, tracersPool, diagnosticsPool, forcingPool, scratchPool
  real(kind=RKIND) :: dt, inputDt
  integer :: err, iCell, iEdge, n, uc, uo, lsize, nx2o, no2x, nWindows, w, s, nSteps, dtSeconds
  integer, pointer :: nCells, nEdges, nVertLevels
  integer, pointer :: iBLT, iBLS, iHT, iST
  integer, dimension(:), pointer :: maxLevelCell, maxLevelEdgeTop, maxLevelEdgeBot
  real(kind=RKIND), dimension(:), pointer :: areaCell
  real(kind=RKIND), dimension(:,:), pointer :: layerThickness, normalVelocity, normalTransportVelocity
  real(kind=RKIND), dimension(:,:,:), pointer :: activeTracers
  real(kind=RKIND), allocatable :: x2o(:,:,:), o2x(:,:), x2oZero(:,:)
  character(len=StrKIND), pointer :: config_start_time, config_dt, config_calendar_type, config_land_ice_flux_mode
  real(kind=RKIND), pointer :: frazilHeatOfFusion
  character(len=StrKIND) :: timeStamp
  character(len=:), allocatable :: rlist
  character(len=1024) :: inPath, cplPath, outPath

  call get_command_argument(1, inPath)
  call get_command_argument(2, cplPath)
  call get_command_argument(3, outPath)
  call read_stream(inPath, inputDt, pools, fields)

  open(newunit=uc, file=trim(cplPath), access='stream', form='unformatted', status='old')
  read(uc) ocn_cpl_dt, nWindows, lsize, nx2o, no2x
  read(uc) n
  allocate(character(len=n) :: rlist)
  read(uc) rlist
  seq_flds_x2o_fields = rlist
  deallocate(rlist)
  read(uc) n
  allocate(character(len=n) :: rlist)
  read(uc) rlist
  seq_flds_o2x_fields = rlist
  deallocate(rlist)
  allocate(x2o(nx2o, lsize, nWindows), o2x(no2x, lsize), x2oZero(nx2o, lsize))
  read(uc) x2o
  close(uc)
  call glc_zocnclass_init(0)
  call mpaso_cpl_indices_set()
  allocate(infodata)

  allocate(domain)
  allocate(domain % blocklist)
  allocate(domain % blocklist % structs)
  allocate(domain % clock)
  domain % configs => pools % configs
  domain % packages => pools % packages
  domain % blocklist % configs => pools % configs
  domain % blocklist % dimensions => pool_named(pools, 'mesh')
  call stub_pool_put_subpool(domain % blocklist % structs, 'state', pool_named(pools, 'state'))
  call stub_pool_put_subpool(domain % blocklist % structs, 'diagnostics', pool_named(pools, 'diagnostics'))
  call stub_pool_put_subpool(domain % blocklist % structs, 'mesh', pool_named(pools, 'mesh'))
  call stub_pool_put_subpool(domain % blocklist % structs, 'verticalMesh', pool_named(pools, 'verticalMesh'))
  call stub_pool_put_subpool(domain % blocklist % structs, 'tend', pool_named(pools, 'tend'))
  call stub_pool_put_subpool(domain % blocklist % structs, 'scratch', pool_named(pools, 'scratch'))
  call stub_pool_put_subpool(domain % blocklist % structs, 'forcing', pool_named(pools, 'forcing'))
  call stub_pool_put_subpool(domain % blocklist % structs, 'shortwave', pool_named(pools, 'swForcing'))
  meshPool => pool_named(pools, 'mesh')
  statePool => pool_named(pools, 'state')
  tracersPool => pool_named(pools, 'tracers')
  diagnosticsPool => pool_named(pools, 'diagnostics')
  forcingPool => pool_named(pools, 'forcing')
  scratchPool => pool_named(pools, 'scratch')

  ! run_forward.F90's init, verbatim: the clock, every module's _init, the split init,
  ! ocn_init_routines_block's state part
  call mpas_pool_get_config(pools % configs, 'config_calendar_type', config_calendar_type)
  call mpas_timekeeping_init(config_calendar_type)
  call mpas_pool_get_config(pools % configs, 'config_start_time', config_start_time)
  call mpas_pool_get_config(pools % configs, 'config_dt', config_dt)
  call mpas_set_clock(domain % clock, config_start_time, config_dt)
  call mpas_set_timeInterval(timeStep, timeString=config_dt)
  call mpas_get_timeInterval(timeStep, dt=dt)

  call ocn_constants_init(pools % configs, pools % packages)
  err = 0
  call ocn_timestep_init(err)
  call ocn_tendency_init(err)
  call ocn_forcing_init(err)
  call ocn_surface_bulk_forcing_init(err)
  call ocn_thick_hadv_init(err)
  call ocn_thick_vadv_init(err)
  call ocn_thick_surface_flux_init(err)
  call ocn_vel_coriolis_init(err)
  call ocn_vel_vadv_init(err)
  call ocn_vel_pressure_grad_init(err)
  call ocn_vel_hmix_init(err)
  call ocn_vel_forcing_init(err)
  if (err /= 0) stop 13
  call ocn_tracer_advection_init(err)
  if (err /= 0) stop 13
  call ocn_tracer_hmix_init(err)
  if (err /= 0) stop 13
  call ocn_tracer_hmix_redi_init(err)
  if (err /= 0) stop 13
  call ocn_tracer_surface_flux_init(err)
  if (err /= 0) stop 13
  call ocn_tracer_short_wave_absorption_init(domain, err)
  if (err /= 0) stop 13
  call ocn_equation_of_state_init(err)
  if (err /= 0) stop 13
  call ocn_diagnostics_init(err)
  if (err /= 0) stop 13
  call ocn_vmix_init(domain, err)
  if (err /= 0) stop 13
  call ocn_surface_land_ice_fluxes_init(err)
  if (err /= 0) stop 13
  call ocn_frazil_forcing_init(err)
  if (err /= 0) stop 13
  call ocn_time_integration_split_init(domain)

  call ocn_diagnostic_solve(dt, statePool, forcingPool, meshPool, diagnosticsPool, scratchPool, tracersPool)
  call mpas_pool_get_dimension(meshPool, 'nCells', nCells)
  call mpas_pool_get_dimension(meshPool, 'nEdges', nEdges)
  call mpas_pool_get_dimension(meshPool, 'nVertLevels', nVertLevels)
  call mpas_pool_get_array(meshPool, 'areaCell', areaCell)
  call mpas_pool_get_array(meshPool, 'maxLevelCell', maxLevelCell)
  call mpas_pool_get_array(meshPool, 'maxLevelEdgeTop', maxLevelEdgeTop)
  call mpas_pool_get_array(meshPool, 'maxLevelEdgeBot', maxLevelEdgeBot)
  call mpas_pool_get_array(statePool, 'layerThickness', layerThickness, 1)
  call mpas_pool_get_array(statePool, 'normalVelocity', normalVelocity, 1)
  call mpas_pool_get_array(tracersPool, 'activeTracers', activeTracers, 1)
  call mpas_pool_get_array(diagnosticsPool, 'normalTransportVelocity', normalTransportVelocity)
  areaCell(nCells+1) = -1.0e34_RKIND
  layerThickness(:, nCells+1) = 0.0_RKIND
  do iEdge=1, nEdges
     normalVelocity(maxLevelEdgeTop(iEdge)+1:maxLevelEdgeBot(iEdge), iEdge) = 0.0_RKIND
     normalVelocity(maxLevelEdgeBot(iEdge)+1:nVertLevels,iEdge) = -1.0e34_RKIND
  end do
  do iCell=1,nCells
     activeTracers(:, maxLevelCell(iCell)+1:nVertLevels,iCell) =  -1.0e34_RKIND
  end do
  do iEdge = 1, nEdges
     normalTransportVelocity(:, iEdge) = normalVelocity(:, iEdge)
  end do
  call mpas_pool_initialize_time_levels(statePool)
  call ocn_forcing_build_fraction_absorbed_array(meshPool, statePool, diagnosticsPool, forcingPool, err, 1)
  if (err /= 0) stop 9
  call ocn_surface_land_ice_fluxes_build_arrays(meshPool, diagnosticsPool, forcingPool, scratchPool, &
       statePool, dt, err)
  if (err /= 0) stop 9

  ! the cap's module-level boundary layer, where v8.4.2 keeps it (its diagnostics pool)
  call mpas_pool_get_config(pools % configs, 'config_land_ice_flux_mode', config_land_ice_flux_mode)
  if (trim(config_land_ice_flux_mode) == 'coupled') then
     call mpas_pool_get_array(diagnosticsPool, 'landIceBoundaryLayerTracers', landIceBoundaryLayerTracers)
     call mpas_pool_get_array(diagnosticsPool, 'landIceTracerTransferVelocities', landIceTracerTransferVelocities)
     call mpas_pool_get_dimension(diagnosticsPool, 'index_landIceBoundaryLayerTemperature', iBLT)
     call mpas_pool_get_dimension(diagnosticsPool, 'index_landIceBoundaryLayerSalinity', iBLS)
     call mpas_pool_get_dimension(diagnosticsPool, 'index_landIceHeatTransferVelocity', iHT)
     call mpas_pool_get_dimension(diagnosticsPool, 'index_landIceSaltTransferVelocity', iST)
     indexBLT = iBLT
     indexBLS = iBLS
     indexHeatTrans = iHT
     indexSaltTrans = iST
  end if

  ! E3SM's ocn_config_init sets the module's namelist options from the configs: the export's
  ! frazil heat of fusion (Fioo_q) is the component's
  call mpas_pool_get_config(pools % configs, 'config_frazil_heat_of_fusion', frazilHeatOfFusion)
  config_frazil_heat_of_fusion = frazilHeatOfFusion

  open(newunit=uo, file=trim(outPath), access='stream', form='unformatted', status='replace')

  ! ocn_init_mct: the forcing arrays built (ocn_comp_mct.F:781-791)
  call ocn_forcing_build_fraction_absorbed_array(meshPool, statePool, diagnosticsPool, forcingPool, err, 1)
  call ocn_frazil_forcing_build_arrays(domain, meshPool, forcingPool, diagnosticsPool, statePool, err)
  call ocn_surface_land_ice_fluxes_build_arrays(meshPool, diagnosticsPool, forcingPool, scratchPool, &
       statePool, dt, err)
  ! runtype 'initial' (833-846): the clock to the coupling alarm, no step; the means reset and taken once
  dtSeconds = nint(dt)
  if (dtSeconds <= 0 .or. mod(ocn_cpl_dt, dtSeconds) /= 0) stop 14
  nSteps = ocn_cpl_dt / dtSeconds
  do s = 1, nSteps
     call mpas_advance_clock(domain % clock)
  end do
  call ocn_time_average_coupled_init(forcingPool)
  call ocn_time_average_coupled_accumulate(diagnosticsPool, statePool, forcingPool, 1)
  ! the export (908) into o2x as mct_aVect_zero left it, then the import of the zeroed x2o (992)
  o2x = 0.0_RKIND
  call ocn_export_mct(o2x, err)
  write(uo) o2x
  x2oZero = 0.0_RKIND
  call ocn_import_mct(x2oZero, err)

  ! ocn_run_mct, per window
  do w = 1, nWindows
     call ocn_import_mct(x2o(:, :, w), err)
     call ocn_time_average_coupled_init(forcingPool)
     do s = 1, nSteps
        call ocn_forcing_build_fraction_absorbed_array(meshPool, statePool, diagnosticsPool, forcingPool, err, 1)
        call ocn_frazil_forcing_build_arrays(domain, meshPool, forcingPool, diagnosticsPool, statePool, err)
        call ocn_surface_land_ice_fluxes_build_arrays(meshPool, diagnosticsPool, forcingPool, scratchPool, &
             statePool, dt, err)
        call mpas_advance_clock(domain % clock)
        currTime = mpas_get_clock_time(domain % clock, MPAS_NOW, err)
        call mpas_get_time(curr_time=currTime, dateTimeString=timeStamp, ierr=err)
        call ocn_timestep(domain, dt, timeStamp)
        call mpas_pool_shift_time_levels(statePool)
     end do
     call ocn_export_mct(o2x, err)
     write(uo) o2x
  end do
  close(uo)
  call write_stream(trim(outPath) // '.pools', fields)
end program run_hybrid_ocean
