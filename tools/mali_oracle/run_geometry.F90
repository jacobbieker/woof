! Drive MALI v8.4.2's geometry and advection diagnostics on one block and
! write every output word: the reference the GPU port is held against.
!
!   run_geometry INPUT.bin OUTPUT.bin
!
! INPUT.bin is written by woof.verify.mali_geometry_oracle.  The driver makes
! the calls the model makes before thickness advection, in this order:
!
!   li_calculate_mask(meshPool, velocityPool, geometryPool, err)
!   li_update_geometry(geometryPool)
!   li_calculate_layerThickness(meshPool, thickness, layerThickness)
!   li_layer_normal_velocity(meshPool, normalVelocity, layerNormalVelocity,
!                            minOfMaxAllowableDt, err)

program run_geometry
  use mpas_kind_types
  use mpas_derived_types
  use mpas_pool_routines
  use mpas_log, only: stub_log_errors
  use li_setup, only: liConfigs, li_calculate_layerThickness
  use li_mask, only: li_calculate_mask
  use li_advection, only: li_update_geometry, li_layer_normal_velocity
  implicit none

  type(MPAS_pool_type), pointer :: mesh, geometry, velocity, configs
  integer, pointer :: nCells, nEdges, nVertices, nVertLevels, nVertInterfaces, nEdgesSolve, vertexDegree
  integer :: maxEdges, solverFlag, errMask, errVelocity, u, i
  real(kind=RKIND), pointer :: rhoi, rhoo, seaLevel, dynamicThickness
  real(kind=RKIND) :: minOfMaxAllowableDt
  character(len=StrKIND), pointer :: velocitySolver
  logical, pointer :: printAdvectionInfo

  integer, dimension(:), pointer :: nEdgesOnCell, cellMask, vertexMask, edgeMask, indexToEdgeID
  integer, dimension(:,:), pointer :: cellsOnCell, cellsOnVertex, cellsOnEdge
  real(kind=RKIND), dimension(:), pointer :: thickness, bedTopography, layerThicknessFractions, dcEdge, &
       upperSurface, lowerSurface
  real(kind=RKIND), dimension(:,:), pointer :: normalVelocity, layerThickness, layerNormalVelocity
  character(len=1024) :: inPath, outPath

  call get_command_argument(1, inPath)
  call get_command_argument(2, outPath)

  allocate(nCells, nEdges, nVertices, nVertLevels, nVertInterfaces, nEdgesSolve, vertexDegree)
  allocate(rhoi, rhoo, seaLevel, dynamicThickness, velocitySolver, printAdvectionInfo)

  open(newunit=u, file=trim(inPath), access='stream', form='unformatted', status='old')
  read(u) nCells, nEdges, nVertices, nVertLevels, maxEdges, solverFlag, nEdgesSolve
  read(u) rhoi, rhoo, seaLevel, dynamicThickness
  nVertInterfaces = nVertLevels + 1
  vertexDegree = 3
  velocitySolver = 'sia'
  if (solverFlag /= 1) velocitySolver = 'none'
  printAdvectionInfo = .false.

  allocate(nEdgesOnCell(nCells), cellsOnCell(maxEdges,nCells), cellsOnVertex(3,nVertices), &
       cellsOnEdge(2,nEdges), cellMask(nCells), vertexMask(nVertices), edgeMask(nEdges), indexToEdgeID(nEdges))
  read(u) nEdgesOnCell, cellsOnCell, cellsOnVertex, cellsOnEdge, cellMask, vertexMask, edgeMask
  indexToEdgeID = (/ (i, i = 1, nEdges) /)

  allocate(thickness(nCells), bedTopography(nCells), layerThicknessFractions(nVertLevels), dcEdge(nEdges), &
       normalVelocity(nVertInterfaces,nEdges))
  read(u) thickness, bedTopography, layerThicknessFractions, dcEdge, normalVelocity

  allocate(upperSurface(nCells), lowerSurface(nCells), layerThickness(nVertLevels,nCells), &
       layerNormalVelocity(nVertLevels,nEdges))
  read(u) upperSurface, lowerSurface, layerThickness, layerNormalVelocity
  close(u)

  allocate(mesh, geometry, velocity, configs)
  call stub_pool_put_dimension(mesh, "nCells", nCells)
  call stub_pool_put_dimension(mesh, "nEdges", nEdges)
  call stub_pool_put_dimension(mesh, "nVertices", nVertices)
  call stub_pool_put_dimension(mesh, "nVertLevels", nVertLevels)
  call stub_pool_put_dimension(mesh, "nVertInterfaces", nVertInterfaces)
  call stub_pool_put_dimension(mesh, "nEdgesSolve", nEdgesSolve)
  call stub_pool_put_dimension(mesh, "vertexDegree", vertexDegree)
  call stub_pool_put_array(mesh, "nEdgesOnCell", nEdgesOnCell)
  call stub_pool_put_array(mesh, "cellsOnCell", cellsOnCell)
  call stub_pool_put_array(mesh, "cellsOnVertex", cellsOnVertex)
  call stub_pool_put_array(mesh, "cellsOnEdge", cellsOnEdge)
  call stub_pool_put_array(mesh, "layerThicknessFractions", layerThicknessFractions)
  call stub_pool_put_array(mesh, "dcEdge", dcEdge)
  call stub_pool_put_array(mesh, "indexToEdgeID", indexToEdgeID)

  call stub_pool_put_dimension(geometry, "nCells", nCells)
  call stub_pool_put_array(geometry, "thickness", thickness)
  call stub_pool_put_array(geometry, "bedTopography", bedTopography)
  call stub_pool_put_array(geometry, "cellMask", cellMask)
  call stub_pool_put_array(geometry, "vertexMask", vertexMask)
  call stub_pool_put_array(geometry, "edgeMask", edgeMask)
  call stub_pool_put_array(geometry, "upperSurface", upperSurface)
  call stub_pool_put_array(geometry, "lowerSurface", lowerSurface)

  call stub_pool_put_config(configs, "config_ice_density", rhoi)
  call stub_pool_put_config(configs, "config_ocean_density", rhoo)
  call stub_pool_put_config(configs, "config_sea_level", seaLevel)
  call stub_pool_put_config(configs, "config_dynamic_thickness", dynamicThickness)
  call stub_pool_put_config(configs, "config_velocity_solver", velocitySolver)
  call stub_pool_put_config(configs, "config_print_thickness_advection_info", printAdvectionInfo)
  liConfigs => configs

  call li_calculate_mask(mesh, velocity, geometry, errMask)
  call li_update_geometry(geometry)
  call li_calculate_layerThickness(mesh, thickness, layerThickness)
  call li_layer_normal_velocity(mesh, normalVelocity, layerNormalVelocity, minOfMaxAllowableDt, errVelocity)

  open(newunit=u, file=trim(outPath), access='stream', form='unformatted', status='replace')
  write(u) errMask, errVelocity, stub_log_errors
  write(u) minOfMaxAllowableDt
  write(u) cellMask, vertexMask, edgeMask
  write(u) upperSurface, lowerSurface, layerThickness, layerNormalVelocity
  close(u)

end program run_geometry
