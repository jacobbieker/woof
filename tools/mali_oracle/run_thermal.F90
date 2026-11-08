! Drive MALI v8.4.2's thermal initialisation and its temperature, enthalpy and
! pressure-melting-point conversions on one block and write every output
! word: the reference the GPU port is held against.
!
!   run_thermal INPUT.bin OUTPUT.bin
!
! INPUT.bin is written by woof.verify.mali_thermal_oracle.  The driver calls,
! in order:
!
!   li_thermal_init(domain, err)              (the model's init call; it also
!                                              sets the module's rhoi)
!   li_compute_pressure_melting_point_fields  (all cells)
!   li_temperature_to_enthalpy                (each cell, on the case's
!                                              temperature/waterfrac columns)
!   li_enthalpy_to_temperature                (each cell, on the case's
!                                              enthalpy columns, with surface
!                                              and basal enthalpy)

program run_thermal
  use mpas_kind_types
  use mpas_derived_types
  use mpas_pool_routines
  use mpas_log, only: stub_log_errors
  use li_setup, only: liConfigs
  use li_thermal
  implicit none

  type(domain_type) :: domain
  type(block_type), pointer :: block
  type(MPAS_pool_type), pointer :: structs, mesh, geometry, thermal, configs
  integer, pointer :: nCells, nCellsSolve, nVertLevels, statsCellID
  integer :: initOption, airOption, err, u, i
  real(kind=RKIND), pointer :: rhoi, rhoo, airValue, lapseRate, heatFluxValue
  character(len=StrKIND), pointer :: thermalSolver, temperatureInit, airSource, heatFluxSource
  logical, pointer :: printThermalInfo, doRestart

  integer, dimension(:), pointer :: indexToCellID
  real(kind=RKIND), dimension(:), pointer :: layerCenterSigma, layerInterfaceSigma, thickness, upperSurface, &
       surfaceAirTemperature, surfaceTemperature, basalTemperature, basalHeatFlux, &
       basalPmpTemperature, convSurfaceEnthalpy, convSurfaceTemperature, convBasalEnthalpy, convBasalTemperature
  real(kind=RKIND), dimension(:,:), pointer :: temperature, waterfrac, enthalpy, pmpTemperature, &
       inTemperature, inWaterfrac, outEnthalpy, convEnthalpy, convTemperature, convWaterfrac
  character(len=1024) :: inPath, outPath

  call get_command_argument(1, inPath)
  call get_command_argument(2, outPath)

  allocate(nCells, nCellsSolve, nVertLevels, statsCellID)
  allocate(rhoi, rhoo, airValue, lapseRate, heatFluxValue)
  allocate(thermalSolver, temperatureInit, airSource, heatFluxSource, printThermalInfo, doRestart)

  open(newunit=u, file=trim(inPath), access='stream', form='unformatted', status='old')
  read(u) nCells, nCellsSolve, nVertLevels, initOption, airOption
  read(u) rhoi, rhoo, airValue, lapseRate, heatFluxValue
  thermalSolver = 'temperature'
  select case (initOption)
  case (1)
     temperatureInit = 'sfc_air_temperature'
  case (2)
     temperatureInit = 'linear'
  case default
     temperatureInit = 'file'
  end select
  airSource = 'constant'
  if (airOption /= 1) airSource = 'lapse'
  heatFluxSource = 'constant'
  printThermalInfo = .false.
  doRestart = .false.
  statsCellID = 1

  allocate(layerCenterSigma(nVertLevels), layerInterfaceSigma(nVertLevels+1), thickness(nCells), &
       upperSurface(nCells), indexToCellID(nCells))
  indexToCellID = (/ (i, i = 1, nCells) /)
  read(u) layerCenterSigma, layerInterfaceSigma, thickness, upperSurface
  ! the prior contents of the init's outputs
  allocate(temperature(nVertLevels,nCells), waterfrac(nVertLevels,nCells), enthalpy(nVertLevels,nCells), &
       surfaceAirTemperature(nCells), surfaceTemperature(nCells), basalTemperature(nCells), basalHeatFlux(nCells))
  read(u) temperature, waterfrac, enthalpy, surfaceAirTemperature, surfaceTemperature, basalTemperature, &
       basalHeatFlux
  ! the conversions' inputs
  allocate(inTemperature(nVertLevels,nCells), inWaterfrac(nVertLevels,nCells), convEnthalpy(nVertLevels,nCells), &
       convSurfaceEnthalpy(nCells), convBasalEnthalpy(nCells))
  read(u) inTemperature, inWaterfrac, convEnthalpy, convSurfaceEnthalpy, convBasalEnthalpy
  close(u)

  allocate(structs, mesh, geometry, thermal, configs)
  call stub_pool_put_dimension(mesh, "nCells", nCells)
  call stub_pool_put_dimension(mesh, "nCellsSolve", nCellsSolve)
  call stub_pool_put_dimension(mesh, "nVertLevels", nVertLevels)
  call stub_pool_put_array(mesh, "layerCenterSigma", layerCenterSigma)
  call stub_pool_put_array(mesh, "layerInterfaceSigma", layerInterfaceSigma)
  call stub_pool_put_array(mesh, "indexToCellID", indexToCellID)
  call stub_pool_put_array(geometry, "thickness", thickness)
  call stub_pool_put_array(geometry, "upperSurface", upperSurface)
  call stub_pool_put_array(thermal, "temperature", temperature)
  call stub_pool_put_array(thermal, "waterfrac", waterfrac)
  call stub_pool_put_array(thermal, "enthalpy", enthalpy)
  call stub_pool_put_array(thermal, "surfaceAirTemperature", surfaceAirTemperature)
  call stub_pool_put_array(thermal, "surfaceTemperature", surfaceTemperature)
  call stub_pool_put_array(thermal, "basalTemperature", basalTemperature)
  call stub_pool_put_array(thermal, "basalHeatFlux", basalHeatFlux)
  call stub_pool_put_subpool(structs, "mesh", mesh)
  call stub_pool_put_subpool(structs, "geometry", geometry)
  call stub_pool_put_subpool(structs, "thermal", thermal)

  call stub_pool_put_config(configs, "config_thermal_solver", thermalSolver)
  call stub_pool_put_config(configs, "config_temperature_init", temperatureInit)
  call stub_pool_put_config(configs, "config_surface_air_temperature_source", airSource)
  call stub_pool_put_config(configs, "config_surface_air_temperature_value", airValue)
  call stub_pool_put_config(configs, "config_surface_air_temperature_lapse_rate", lapseRate)
  call stub_pool_put_config(configs, "config_basal_heat_flux_source", heatFluxSource)
  call stub_pool_put_config(configs, "config_basal_heat_flux_value", heatFluxValue)
  call stub_pool_put_config(configs, "config_print_thermal_info", printThermalInfo)
  call stub_pool_put_config(configs, "config_do_restart", doRestart)
  call stub_pool_put_config(configs, "config_stats_cell_ID", statsCellID)
  call stub_pool_put_config(configs, "config_ice_density", rhoi)
  call stub_pool_put_config(configs, "config_ocean_density", rhoo)
  liConfigs => configs

  allocate(block)
  block % structs => structs
  block % configs => configs
  block % next => null()
  domain % blocklist => block
  domain % configs => configs

  call li_thermal_init(domain, err)

  allocate(pmpTemperature(nVertLevels,nCells), basalPmpTemperature(nCells))
  call li_compute_pressure_melting_point_fields(nCells, thickness, layerCenterSigma, pmpTemperature, &
       basalPmpTemperature)

  allocate(outEnthalpy(nVertLevels,nCells))
  do i = 1, nCells
     call li_temperature_to_enthalpy(layerCenterSigma, thickness(i), inTemperature(:,i), inWaterfrac(:,i), &
          outEnthalpy(:,i))
  enddo

  allocate(convTemperature(nVertLevels,nCells), convWaterfrac(nVertLevels,nCells), &
       convSurfaceTemperature(nCells), convBasalTemperature(nCells))
  do i = 1, nCells
     call li_enthalpy_to_temperature(layerCenterSigma, thickness(i), convEnthalpy(:,i), convTemperature(:,i), &
          convWaterfrac(:,i), convSurfaceEnthalpy(i), convSurfaceTemperature(i), convBasalEnthalpy(i), &
          convBasalTemperature(i))
  enddo

  open(newunit=u, file=trim(outPath), access='stream', form='unformatted', status='replace')
  write(u) err, stub_log_errors
  write(u) temperature, waterfrac, enthalpy, surfaceAirTemperature, surfaceTemperature, basalTemperature, &
       basalHeatFlux
  write(u) pmpTemperature, basalPmpTemperature, outEnthalpy
  write(u) convEnthalpy, convTemperature, convWaterfrac, convSurfaceEnthalpy, convSurfaceTemperature, &
       convBasalEnthalpy, convBasalTemperature
  close(u)

end program run_thermal
