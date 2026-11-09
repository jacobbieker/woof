! Drive MALI v8.4.2's thermal solver on one block and write every output
! word: the reference the GPU port is held against.
!
!   run_thermal_solver INPUT.bin OUTPUT.bin
!
! INPUT.bin is written by woof.verify.mali_thermal_solver_oracle.  The driver
! makes the calls li_time_integration_fe makes, in this order:
!
!   li_thermal_init(domain, err)             as on a restart (config_do_restart,
!                                            the basal heat flux from file), where
!                                            it copies config_ice_density and
!                                            config_ocean_density into the module,
!                                            builds dsigmaTerm and, with 'lapse',
!                                            sets the surface air temperature
!   li_basal_melt_floating_ice(domain, err)
!   li_thermal_solver(domain, err)
!
! The scratch fields they allocate (iceCellMask, workLevelEdge) are the
! driver's arrays, filled with the case's words: the model leaves them
! uninitialised, and the outputs must not depend on them.

program run_thermal_solver
  use mpas_kind_types
  use mpas_derived_types
  use mpas_pool_routines
  use mpas_log, only: stub_log_errors
  use li_setup, only: liConfigs
  use li_thermal, only: li_thermal_init, li_basal_melt_floating_ice, li_thermal_solver
  implicit none

  type(domain_type) :: domain
  type(block_type), pointer :: block
  type(MPAS_pool_type), pointer :: structs, mesh, geometry, thermal, velocity, scratch, configs
  integer, pointer :: nCells, nEdges, nVertices, nVertLevels, nVertInterfaces, nCellsSolve, nEdgesSolve, &
       vertexDegree, statsCellID
  integer :: maxEdges, solverFlag, thermalFlag, airFlag, floatFlag, bmbFlag, errInit, errFloat, errSolver, u, i
  real(kind=RKIND), pointer :: deltat, daysSinceStart, rhoi, rhoo, seaLevel, dynamicThickness, thermalThickness, &
       flowLawExponent, airValue, lapseRate, heatFluxValue, bmltFlux, bmltXlimit, serAmplitude, serPeriod, serPhase
  character(len=StrKIND), pointer :: velocitySolver, thermalSolver, temperatureInit, airSource, heatFluxSource, &
       basalMassBalFloat
  logical, pointer :: printThermalInfo, doRestart, calculateBmb

  integer, dimension(:), pointer :: nEdgesOnCell, cellMask, vertexMask, edgeMask, indexToCellID, indexToEdgeID, &
       iceCellMask
  integer, dimension(:,:), pointer :: cellsOnCell, cellsOnVertex, cellsOnEdge, edgesOnCell
  real(kind=RKIND), dimension(:), pointer :: layerCenterSigma, layerInterfaceSigma, dcEdge, dvEdge, areaCell, &
       xCell, thickness, bedTopography, upperSurface, lowerSurface, basalWaterThickness, slopeEdge, basalSpeed, &
       betaSolve, basalHeatFlux, surfaceTemperature, basalTemperature, surfaceAirTemperature, &
       surfaceConductiveFlux, basalConductiveFlux, basalFrictionFlux, groundedBasalMassBal, floatingBasalMassBal
  real(kind=RKIND), dimension(:,:), pointer :: flowParamA, temperature, waterfrac, enthalpy, heatDissipation, &
       workLevelEdge
  character(len=1024) :: inPath, outPath

  call get_command_argument(1, inPath)
  call get_command_argument(2, outPath)

  allocate(nCells, nEdges, nVertices, nVertLevels, nVertInterfaces, nCellsSolve, nEdgesSolve, vertexDegree, &
       statsCellID)
  allocate(deltat, daysSinceStart, rhoi, rhoo, seaLevel, dynamicThickness, thermalThickness, flowLawExponent, &
       airValue, lapseRate, heatFluxValue, bmltFlux, bmltXlimit, serAmplitude, serPeriod, serPhase)
  allocate(velocitySolver, thermalSolver, temperatureInit, airSource, heatFluxSource, basalMassBalFloat)
  allocate(printThermalInfo, doRestart, calculateBmb)

  open(newunit=u, file=trim(inPath), access='stream', form='unformatted', status='old')
  read(u) nCells, nEdges, nVertices, nVertLevels, maxEdges, nCellsSolve, solverFlag, thermalFlag, airFlag, &
       floatFlag, bmbFlag
  read(u) deltat, daysSinceStart, rhoi, rhoo, seaLevel, dynamicThickness, thermalThickness, flowLawExponent, &
       airValue, lapseRate, bmltFlux, bmltXlimit, serAmplitude, serPeriod, serPhase
  nVertInterfaces = nVertLevels + 1
  nEdgesSolve = nEdges
  vertexDegree = 3
  velocitySolver = 'sia'
  if (solverFlag /= 1) velocitySolver = 'none'
  select case (thermalFlag)
  case (1)
     thermalSolver = 'temperature'
  case (2)
     thermalSolver = 'enthalpy'
  case default
     thermalSolver = 'none'
  end select
  airSource = 'file'
  if (airFlag == 1) airSource = 'lapse'
  select case (floatFlag)
  case (1)
     basalMassBalFloat = 'file'
  case (2)
     basalMassBalFloat = 'constant'
  case (3)
     basalMassBalFloat = 'mismip'
  case (4)
     basalMassBalFloat = 'seroussi'
  case default
     basalMassBalFloat = 'none'
  end select
  temperatureInit = 'file'
  heatFluxSource = 'file'
  heatFluxValue = 0.0_RKIND
  printThermalInfo = .false.
  doRestart = .true.
  calculateBmb = bmbFlag /= 0
  statsCellID = 0

  allocate(nEdgesOnCell(nCells), cellsOnCell(maxEdges,nCells), cellsOnVertex(3,nVertices), &
       cellsOnEdge(2,nEdges), edgesOnCell(maxEdges,nCells), cellMask(nCells), vertexMask(nVertices), &
       edgeMask(nEdges), indexToCellID(nCells), indexToEdgeID(nEdges))
  read(u) nEdgesOnCell, cellsOnCell, cellsOnVertex, cellsOnEdge, edgesOnCell, cellMask, vertexMask, edgeMask
  indexToCellID = (/ (i, i = 1, nCells) /)
  indexToEdgeID = (/ (i, i = 1, nEdges) /)

  allocate(layerCenterSigma(nVertLevels), layerInterfaceSigma(nVertInterfaces), dcEdge(nEdges), dvEdge(nEdges), &
       areaCell(nCells), xCell(nCells), thickness(nCells), bedTopography(nCells), upperSurface(nCells), &
       lowerSurface(nCells), basalWaterThickness(nCells), slopeEdge(nEdges), flowParamA(nVertLevels,nCells), &
       basalSpeed(nCells), betaSolve(nCells), basalHeatFlux(nCells))
  read(u) layerCenterSigma, layerInterfaceSigma, dcEdge, dvEdge, areaCell, xCell, thickness, bedTopography, &
       upperSurface, lowerSurface, basalWaterThickness, slopeEdge, flowParamA, basalSpeed, betaSolve, basalHeatFlux

  ! the prior contents of every output, then the scratch arrays' words
  allocate(temperature(nVertLevels,nCells), waterfrac(nVertLevels,nCells), enthalpy(nVertLevels,nCells), &
       surfaceTemperature(nCells), basalTemperature(nCells), surfaceAirTemperature(nCells), &
       surfaceConductiveFlux(nCells), basalConductiveFlux(nCells), heatDissipation(nVertLevels,nCells), &
       basalFrictionFlux(nCells), groundedBasalMassBal(nCells), floatingBasalMassBal(nCells))
  read(u) temperature, waterfrac, enthalpy, surfaceTemperature, basalTemperature, surfaceAirTemperature, &
       surfaceConductiveFlux, basalConductiveFlux, heatDissipation, basalFrictionFlux, groundedBasalMassBal, &
       floatingBasalMassBal
  allocate(workLevelEdge(nVertLevels,nEdges), iceCellMask(nCells))
  read(u) workLevelEdge, iceCellMask
  close(u)

  allocate(structs, mesh, geometry, thermal, velocity, scratch, configs)
  call stub_pool_put_dimension(mesh, "nCells", nCells)
  call stub_pool_put_dimension(mesh, "nEdges", nEdges)
  call stub_pool_put_dimension(mesh, "nVertices", nVertices)
  call stub_pool_put_dimension(mesh, "nVertLevels", nVertLevels)
  call stub_pool_put_dimension(mesh, "nVertInterfaces", nVertInterfaces)
  call stub_pool_put_dimension(mesh, "nCellsSolve", nCellsSolve)
  call stub_pool_put_dimension(mesh, "nEdgesSolve", nEdgesSolve)
  call stub_pool_put_dimension(mesh, "vertexDegree", vertexDegree)
  call stub_pool_put_array(mesh, "nEdgesOnCell", nEdgesOnCell)
  call stub_pool_put_array(mesh, "cellsOnCell", cellsOnCell)
  call stub_pool_put_array(mesh, "cellsOnVertex", cellsOnVertex)
  call stub_pool_put_array(mesh, "cellsOnEdge", cellsOnEdge)
  call stub_pool_put_array(mesh, "edgesOnCell", edgesOnCell)
  call stub_pool_put_array(mesh, "layerCenterSigma", layerCenterSigma)
  call stub_pool_put_array(mesh, "layerInterfaceSigma", layerInterfaceSigma)
  call stub_pool_put_array(mesh, "dcEdge", dcEdge)
  call stub_pool_put_array(mesh, "dvEdge", dvEdge)
  call stub_pool_put_array(mesh, "areaCell", areaCell)
  call stub_pool_put_array(mesh, "xCell", xCell)
  call stub_pool_put_array(mesh, "indexToCellID", indexToCellID)
  call stub_pool_put_array(mesh, "indexToEdgeID", indexToEdgeID)
  call stub_pool_put_array(mesh, "deltat", deltat)
  call stub_pool_put_array(mesh, "daysSinceStart", daysSinceStart)

  call stub_pool_put_dimension(geometry, "nCells", nCells)
  call stub_pool_put_array(geometry, "thickness", thickness)
  call stub_pool_put_array(geometry, "bedTopography", bedTopography)
  call stub_pool_put_array(geometry, "upperSurface", upperSurface)
  call stub_pool_put_array(geometry, "lowerSurface", lowerSurface)
  call stub_pool_put_array(geometry, "basalWaterThickness", basalWaterThickness)
  call stub_pool_put_array(geometry, "slopeEdge", slopeEdge)
  call stub_pool_put_array(geometry, "cellMask", cellMask)
  call stub_pool_put_array(geometry, "vertexMask", vertexMask)
  call stub_pool_put_array(geometry, "edgeMask", edgeMask)
  call stub_pool_put_array(geometry, "groundedBasalMassBal", groundedBasalMassBal)
  call stub_pool_put_array(geometry, "floatingBasalMassBal", floatingBasalMassBal)

  call stub_pool_put_array(velocity, "flowParamA", flowParamA)
  call stub_pool_put_array(velocity, "basalSpeed", basalSpeed)
  call stub_pool_put_array(velocity, "betaSolve", betaSolve)

  call stub_pool_put_array(thermal, "temperature", temperature)
  call stub_pool_put_array(thermal, "waterfrac", waterfrac)
  call stub_pool_put_array(thermal, "enthalpy", enthalpy)
  call stub_pool_put_array(thermal, "surfaceAirTemperature", surfaceAirTemperature)
  call stub_pool_put_array(thermal, "surfaceTemperature", surfaceTemperature)
  call stub_pool_put_array(thermal, "basalTemperature", basalTemperature)
  call stub_pool_put_array(thermal, "basalHeatFlux", basalHeatFlux)
  call stub_pool_put_array(thermal, "surfaceConductiveFlux", surfaceConductiveFlux)
  call stub_pool_put_array(thermal, "basalConductiveFlux", basalConductiveFlux)
  call stub_pool_put_array(thermal, "basalFrictionFlux", basalFrictionFlux)
  call stub_pool_put_array(thermal, "heatDissipation", heatDissipation)

  call stub_pool_put_array(scratch, "workLevelEdge", workLevelEdge)
  call stub_pool_put_array(scratch, "iceCellMask", iceCellMask)

  call stub_pool_put_subpool(structs, "mesh", mesh)
  call stub_pool_put_subpool(structs, "geometry", geometry)
  call stub_pool_put_subpool(structs, "thermal", thermal)
  call stub_pool_put_subpool(structs, "velocity", velocity)
  call stub_pool_put_subpool(structs, "scratch", scratch)

  call stub_pool_put_config(configs, "config_ice_density", rhoi)
  call stub_pool_put_config(configs, "config_ocean_density", rhoo)
  call stub_pool_put_config(configs, "config_sea_level", seaLevel)
  call stub_pool_put_config(configs, "config_dynamic_thickness", dynamicThickness)
  call stub_pool_put_config(configs, "config_velocity_solver", velocitySolver)
  call stub_pool_put_config(configs, "config_flowLawExponent", flowLawExponent)
  call stub_pool_put_config(configs, "config_stats_cell_ID", statsCellID)
  call stub_pool_put_config(configs, "config_thermal_solver", thermalSolver)
  call stub_pool_put_config(configs, "config_thermal_thickness", thermalThickness)
  call stub_pool_put_config(configs, "config_thermal_calculate_bmb", calculateBmb)
  call stub_pool_put_config(configs, "config_temperature_init", temperatureInit)
  call stub_pool_put_config(configs, "config_surface_air_temperature_source", airSource)
  call stub_pool_put_config(configs, "config_surface_air_temperature_value", airValue)
  call stub_pool_put_config(configs, "config_surface_air_temperature_lapse_rate", lapseRate)
  call stub_pool_put_config(configs, "config_basal_heat_flux_source", heatFluxSource)
  call stub_pool_put_config(configs, "config_basal_heat_flux_value", heatFluxValue)
  call stub_pool_put_config(configs, "config_basal_mass_bal_float", basalMassBalFloat)
  call stub_pool_put_config(configs, "config_bmlt_float_flux", bmltFlux)
  call stub_pool_put_config(configs, "config_bmlt_float_xlimit", bmltXlimit)
  call stub_pool_put_config(configs, "config_basal_mass_bal_seroussi_amplitude", serAmplitude)
  call stub_pool_put_config(configs, "config_basal_mass_bal_seroussi_period", serPeriod)
  call stub_pool_put_config(configs, "config_basal_mass_bal_seroussi_phase", serPhase)
  call stub_pool_put_config(configs, "config_print_thermal_info", printThermalInfo)
  call stub_pool_put_config(configs, "config_do_restart", doRestart)
  liConfigs => configs

  allocate(block)
  block % structs => structs
  block % configs => configs
  block % next => null()
  domain % blocklist => block
  domain % configs => configs

  call li_thermal_init(domain, errInit)
  call li_basal_melt_floating_ice(domain, errFloat)
  call li_thermal_solver(domain, errSolver)

  open(newunit=u, file=trim(outPath), access='stream', form='unformatted', status='replace')
  write(u) errInit, errFloat, errSolver, stub_log_errors
  write(u) cellMask, vertexMask, edgeMask
  write(u) temperature, waterfrac, enthalpy, surfaceTemperature, basalTemperature, surfaceAirTemperature, &
       surfaceConductiveFlux, basalConductiveFlux, heatDissipation, basalFrictionFlux, groundedBasalMassBal, &
       floatingBasalMassBal
  close(u)

end program run_thermal_solver
