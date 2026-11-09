! Drive MALI v8.4.2's thickness and tracer advection on one block and write
! every output word: the reference the GPU port is held against.
!
!   run_advection INPUT.bin OUTPUT.bin
!
! INPUT.bin is written by woof.verify.mali_advection_oracle.  The driver
! makes the calls the model makes, in this order:
!
!   li_thermal_init(domain, err)    as on a restart (config_do_restart, the
!                                   air temperature and heat flux from file),
!                                   where it only copies config_ice_density
!                                   and config_ocean_density into the module
!                                   and builds its vertical grid terms
!   li_advection_thickness_tracers(dt, meshPool, velocityPool, geometryPool,
!                                  thermalPool, scratchPool, err,
!                                  advectTracersIn)
!
! the second as li_time_integration_fe calls it: advectTracersIn true with
! config_tracer_advection 'fo', false with 'none'.  The scratch fields it
! allocates (workLevelCell, workTracerCell[2], workTracerLevelCell[2]) are
! the driver's arrays, filled with the case's words: the model leaves them
! uninitialised, and the outputs must not depend on them.

program run_advection
  use mpas_kind_types
  use mpas_derived_types
  use mpas_pool_routines
  use mpas_log, only: stub_log_errors
  use li_setup, only: liConfigs
  use li_thermal, only: li_thermal_init
  use li_advection, only: li_advection_thickness_tracers
  implicit none

  type(domain_type) :: domain
  type(block_type), pointer :: block
  type(MPAS_pool_type), pointer :: structs, mesh, geometry, thermal, velocity, scratch, configs
  integer, pointer :: nCells, nEdges, nVertices, nVertLevels, nVertInterfaces, nCellsSolve, nEdgesSolve, &
       vertexDegree, statsCellID
  integer :: maxEdges, solverFlag, thermalFlag, tracerFlag, advectionFlag, errInit, errAdvect, u, i
  real(kind=RKIND) :: dt
  real(kind=RKIND), pointer :: rhoi, rhoo, seaLevel, dynamicThickness, airValue, lapseRate, heatFluxValue
  character(len=StrKIND), pointer :: velocitySolver, thermalSolver, thicknessAdvection, temperatureInit, &
       airSource, heatFluxSource
  logical, pointer :: printAdvectionInfo, printThermalInfo, doRestart

  integer, dimension(:), pointer :: nEdgesOnCell, cellMask, vertexMask, edgeMask, indexToCellID, indexToEdgeID
  integer, dimension(:,:), pointer :: cellsOnCell, cellsOnVertex, cellsOnEdge, edgesOnCell, edgeSignOnCell
  real(kind=RKIND), dimension(:), pointer :: layerThicknessFractions, layerCenterSigma, layerInterfaceSigma, &
       dvEdge, dcEdge, areaCell, bedTopography, groundedBasalMassBal, floatingBasalMassBal, &
       surfaceAirTemperature, basalTemperature, upperSurface, surfaceTemperature, basalHeatFlux, &
       thickness, sfcMassBal, basalMassBal, dynamicThickening
  real(kind=RKIND), dimension(:,:), pointer :: layerNormalVelocity, layerThicknessEdge, &
       layerThickness, temperature, waterfrac, enthalpy, workLevelCell, workTracerCell, workTracerCell2
  real(kind=RKIND), dimension(:,:,:), pointer :: workTracerLevelCell, workTracerLevelCell2
  character(len=1024) :: inPath, outPath

  call get_command_argument(1, inPath)
  call get_command_argument(2, outPath)

  allocate(nCells, nEdges, nVertices, nVertLevels, nVertInterfaces, nCellsSolve, nEdgesSolve, vertexDegree, &
       statsCellID)
  allocate(rhoi, rhoo, seaLevel, dynamicThickness, airValue, lapseRate, heatFluxValue)
  allocate(velocitySolver, thermalSolver, thicknessAdvection, temperatureInit, airSource, heatFluxSource)
  allocate(printAdvectionInfo, printThermalInfo, doRestart)

  open(newunit=u, file=trim(inPath), access='stream', form='unformatted', status='old')
  read(u) nCells, nEdges, nVertices, nVertLevels, maxEdges, nCellsSolve, solverFlag, thermalFlag, tracerFlag, &
       advectionFlag
  read(u) dt, rhoi, rhoo, seaLevel, dynamicThickness
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
  thicknessAdvection = 'fo'
  if (advectionFlag /= 1) thicknessAdvection = 'none'
  temperatureInit = 'file'
  airSource = 'file'
  heatFluxSource = 'file'
  airValue = 273.15_RKIND
  lapseRate = 0.0_RKIND
  heatFluxValue = 0.0_RKIND
  printAdvectionInfo = .false.
  printThermalInfo = .false.
  doRestart = .true.
  statsCellID = 0

  allocate(nEdgesOnCell(nCells), cellsOnCell(maxEdges,nCells), cellsOnVertex(3,nVertices), &
       cellsOnEdge(2,nEdges), edgesOnCell(maxEdges,nCells), cellMask(nCells), vertexMask(nVertices), &
       edgeMask(nEdges), indexToCellID(nCells), indexToEdgeID(nEdges), edgeSignOnCell(maxEdges,nCells))
  read(u) nEdgesOnCell, cellsOnCell, cellsOnVertex, cellsOnEdge, edgesOnCell, edgeSignOnCell, cellMask, &
       vertexMask, edgeMask
  indexToCellID = (/ (i, i = 1, nCells) /)
  indexToEdgeID = (/ (i, i = 1, nEdges) /)

  allocate(layerThicknessFractions(nVertLevels), layerCenterSigma(nVertLevels), &
       layerInterfaceSigma(nVertInterfaces), dvEdge(nEdges), dcEdge(nEdges), areaCell(nCells), &
       layerNormalVelocity(nVertLevels,nEdges), &
       layerThicknessEdge(nVertLevels,nEdges), bedTopography(nCells), groundedBasalMassBal(nCells), &
       floatingBasalMassBal(nCells), surfaceAirTemperature(nCells), basalTemperature(nCells))
  read(u) layerThicknessFractions, layerCenterSigma, layerInterfaceSigma, dvEdge, dcEdge, areaCell, &
       layerNormalVelocity, layerThicknessEdge, bedTopography, groundedBasalMassBal, &
       floatingBasalMassBal, surfaceAirTemperature, basalTemperature

  ! the prior contents of every output, then the scratch arrays' words
  allocate(thickness(nCells), sfcMassBal(nCells), basalMassBal(nCells), dynamicThickening(nCells), &
       layerThickness(nVertLevels,nCells), temperature(nVertLevels,nCells), waterfrac(nVertLevels,nCells), &
       enthalpy(nVertLevels,nCells))
  read(u) thickness, sfcMassBal, basalMassBal, dynamicThickening, layerThickness, temperature, waterfrac, enthalpy
  allocate(workLevelCell(nVertLevels,nCells), workTracerCell(1,nCells), workTracerCell2(1,nCells), &
       workTracerLevelCell(1,nVertLevels,nCells), workTracerLevelCell2(1,nVertLevels,nCells))
  read(u) workLevelCell, workTracerCell, workTracerCell2, workTracerLevelCell, workTracerLevelCell2
  close(u)
  ! li_thermal_init's fields it leaves alone on a restart
  allocate(upperSurface(nCells), surfaceTemperature(nCells), basalHeatFlux(nCells))
  upperSurface = 0.0_RKIND
  surfaceTemperature = 0.0_RKIND
  basalHeatFlux = 0.0_RKIND

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
  call stub_pool_put_array(mesh, "edgeSignOnCell", edgeSignOnCell)
  call stub_pool_put_array(mesh, "layerThicknessFractions", layerThicknessFractions)
  call stub_pool_put_array(mesh, "layerCenterSigma", layerCenterSigma)
  call stub_pool_put_array(mesh, "layerInterfaceSigma", layerInterfaceSigma)
  call stub_pool_put_array(mesh, "dvEdge", dvEdge)
  call stub_pool_put_array(mesh, "dcEdge", dcEdge)
  call stub_pool_put_array(mesh, "areaCell", areaCell)
  call stub_pool_put_array(mesh, "indexToCellID", indexToCellID)
  call stub_pool_put_array(mesh, "indexToEdgeID", indexToEdgeID)

  call stub_pool_put_dimension(geometry, "nCells", nCells)
  call stub_pool_put_array(geometry, "thickness", thickness)
  call stub_pool_put_array(geometry, "bedTopography", bedTopography)
  call stub_pool_put_array(geometry, "sfcMassBal", sfcMassBal)
  call stub_pool_put_array(geometry, "basalMassBal", basalMassBal)
  call stub_pool_put_array(geometry, "groundedBasalMassBal", groundedBasalMassBal)
  call stub_pool_put_array(geometry, "floatingBasalMassBal", floatingBasalMassBal)
  call stub_pool_put_array(geometry, "layerThickness", layerThickness)
  call stub_pool_put_array(geometry, "layerThicknessEdge", layerThicknessEdge)
  call stub_pool_put_array(geometry, "cellMask", cellMask)
  call stub_pool_put_array(geometry, "vertexMask", vertexMask)
  call stub_pool_put_array(geometry, "edgeMask", edgeMask)
  call stub_pool_put_array(geometry, "dynamicThickening", dynamicThickening)
  call stub_pool_put_array(geometry, "upperSurface", upperSurface)

  call stub_pool_put_array(velocity, "layerNormalVelocity", layerNormalVelocity)

  call stub_pool_put_array(thermal, "temperature", temperature)
  call stub_pool_put_array(thermal, "waterfrac", waterfrac)
  call stub_pool_put_array(thermal, "enthalpy", enthalpy)
  call stub_pool_put_array(thermal, "surfaceAirTemperature", surfaceAirTemperature)
  call stub_pool_put_array(thermal, "surfaceTemperature", surfaceTemperature)
  call stub_pool_put_array(thermal, "basalTemperature", basalTemperature)
  call stub_pool_put_array(thermal, "basalHeatFlux", basalHeatFlux)

  call stub_pool_put_array(scratch, "workLevelCell", workLevelCell)
  call stub_pool_put_array(scratch, "workTracerCell", workTracerCell)
  call stub_pool_put_array(scratch, "workTracerCell2", workTracerCell2)
  call stub_pool_put_array(scratch, "workTracerLevelCell", workTracerLevelCell)
  call stub_pool_put_array(scratch, "workTracerLevelCell2", workTracerLevelCell2)

  call stub_pool_put_subpool(structs, "mesh", mesh)
  call stub_pool_put_subpool(structs, "geometry", geometry)
  call stub_pool_put_subpool(structs, "thermal", thermal)

  call stub_pool_put_config(configs, "config_ice_density", rhoi)
  call stub_pool_put_config(configs, "config_ocean_density", rhoo)
  call stub_pool_put_config(configs, "config_sea_level", seaLevel)
  call stub_pool_put_config(configs, "config_dynamic_thickness", dynamicThickness)
  call stub_pool_put_config(configs, "config_velocity_solver", velocitySolver)
  call stub_pool_put_config(configs, "config_thickness_advection", thicknessAdvection)
  call stub_pool_put_config(configs, "config_print_thickness_advection_info", printAdvectionInfo)
  call stub_pool_put_config(configs, "config_stats_cell_ID", statsCellID)
  call stub_pool_put_config(configs, "config_thermal_solver", thermalSolver)
  call stub_pool_put_config(configs, "config_temperature_init", temperatureInit)
  call stub_pool_put_config(configs, "config_surface_air_temperature_source", airSource)
  call stub_pool_put_config(configs, "config_surface_air_temperature_value", airValue)
  call stub_pool_put_config(configs, "config_surface_air_temperature_lapse_rate", lapseRate)
  call stub_pool_put_config(configs, "config_basal_heat_flux_source", heatFluxSource)
  call stub_pool_put_config(configs, "config_basal_heat_flux_value", heatFluxValue)
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
  call li_advection_thickness_tracers(dt, mesh, velocity, geometry, thermal, scratch, errAdvect, &
       advectTracersIn = tracerFlag /= 0)

  open(newunit=u, file=trim(outPath), access='stream', form='unformatted', status='replace')
  write(u) errInit, errAdvect, stub_log_errors
  write(u) cellMask, vertexMask, edgeMask
  write(u) thickness, sfcMassBal, basalMassBal, dynamicThickening, layerThickness, temperature, waterfrac, enthalpy
  close(u)

end program run_advection
