! Drive MPAS land-ice (MALI) v8.4.2's shallow-ice velocity solver on one block
! and write every output word: the reference the GPU port is held against.
!
!   run_oracle INPUT.bin OUTPUT.bin
!
! INPUT.bin is written by woof.verify.mali_sia_oracle (raw little-endian
! words, Fortran array order).  The driver puts the fields in the mesh,
! geometry, thermal and velocity pools and the configs in liConfigs, then
! makes the model's call, li_sia_solve(meshPool, geometryPool, thermalPool,
! velocityPool, err).  That one call runs li_calculate_flowParamA, the
! upper-surface interpolation to vertices for the configured tangent-slope
! method, the slope magnitude and the SIA velocity profile on every edge.

program run_oracle
  use mpas_kind_types
  use mpas_derived_types
  use mpas_pool_routines
  use mpas_log, only: stub_log_errors
  use li_setup, only: liConfigs
  use li_sia
  implicit none

  type(MPAS_pool_type), pointer :: mesh, geometry, thermal, velocity, configs
  integer, pointer :: nCells, nEdges, nVertices, nVertLevels, nVertInterfaces, nEdgesSolve, vertexDegree
  integer :: maxEdges2, tangentOption, flowAOption, err, u
  real(kind=RKIND), pointer :: rhoi, flowLawExponent, enhancementFactor, defaultFlowParamA, dynamicThickness
  character(len=StrKIND), pointer :: tangentSlopeCalculation, flowParamACalculation

  integer, dimension(:), pointer :: nEdgesOnEdge, edgeMask, cellMask
  integer, dimension(:,:), pointer :: cellsOnEdge, verticesOnEdge, baryCellsOnVertex, cellsOnVertex, edgesOnEdge
  real(kind=RKIND), dimension(:), pointer :: layerInterfaceSigma, layerCenterSigma, dvEdge, thickness, &
       upperSurface, normalSlopeEdge, slopeEdge, tangentSlopeEdge, upperSurfaceVertex
  real(kind=RKIND), dimension(:,:), pointer :: baryWeightsOnVertex, kiteAreasOnVertex, weightsOnEdge, &
       temperature, normalVelocity, flowParamA
  character(len=1024) :: inPath, outPath

  call get_command_argument(1, inPath)
  call get_command_argument(2, outPath)

  allocate(nCells, nEdges, nVertices, nVertLevels, nVertInterfaces, nEdgesSolve, vertexDegree)
  allocate(rhoi, flowLawExponent, enhancementFactor, defaultFlowParamA, dynamicThickness)
  allocate(tangentSlopeCalculation, flowParamACalculation)

  open(newunit=u, file=trim(inPath), access='stream', form='unformatted', status='old')
  read(u) nCells, nEdges, nVertices, nVertLevels, maxEdges2, nEdgesSolve, tangentOption, flowAOption
  read(u) rhoi, flowLawExponent, enhancementFactor, defaultFlowParamA, dynamicThickness
  nVertInterfaces = nVertLevels + 1
  vertexDegree = 3
  select case (tangentOption)
  case (1)
     tangentSlopeCalculation = 'from_vertex_barycentric'
  case (2)
     tangentSlopeCalculation = 'from_vertex_barycentric_kiteareas'
  case default
     tangentSlopeCalculation = 'from_normal_slope'
  end select
  select case (flowAOption)
  case (1)
     flowParamACalculation = 'constant'
  case (2)
     flowParamACalculation = 'PB1982'
  case default
     flowParamACalculation = 'CP2010'
  end select

  allocate(cellsOnEdge(2,nEdges), verticesOnEdge(2,nEdges), baryCellsOnVertex(3,nVertices), &
       cellsOnVertex(3,nVertices), nEdgesOnEdge(nEdges), edgesOnEdge(maxEdges2,nEdges), &
       edgeMask(nEdges), cellMask(nCells))
  read(u) cellsOnEdge, verticesOnEdge, baryCellsOnVertex, cellsOnVertex, nEdgesOnEdge, edgesOnEdge, &
       edgeMask, cellMask

  allocate(layerInterfaceSigma(nVertInterfaces), layerCenterSigma(nVertLevels), dvEdge(nEdges), &
       baryWeightsOnVertex(3,nVertices), kiteAreasOnVertex(3,nVertices), weightsOnEdge(maxEdges2,nEdges), &
       thickness(nCells), upperSurface(nCells), normalSlopeEdge(nEdges), temperature(nVertLevels,nCells))
  read(u) layerInterfaceSigma, layerCenterSigma, dvEdge, baryWeightsOnVertex, kiteAreasOnVertex, &
       weightsOnEdge, thickness, upperSurface, normalSlopeEdge, temperature

  ! the prior contents of every output
  allocate(normalVelocity(nVertInterfaces,nEdges), flowParamA(nVertLevels,nCells), slopeEdge(nEdges), &
       tangentSlopeEdge(nEdges), upperSurfaceVertex(nVertices))
  read(u) normalVelocity, flowParamA, slopeEdge, tangentSlopeEdge, upperSurfaceVertex
  close(u)

  allocate(mesh, geometry, thermal, velocity, configs)
  call stub_pool_put_dimension(mesh, "nCells", nCells)
  call stub_pool_put_dimension(mesh, "nEdges", nEdges)
  call stub_pool_put_dimension(mesh, "nVertices", nVertices)
  call stub_pool_put_dimension(mesh, "nVertLevels", nVertLevels)
  call stub_pool_put_dimension(mesh, "nVertInterfaces", nVertInterfaces)
  call stub_pool_put_dimension(mesh, "nEdgesSolve", nEdgesSolve)
  call stub_pool_put_dimension(mesh, "vertexDegree", vertexDegree)
  call stub_pool_put_array(mesh, "cellsOnEdge", cellsOnEdge)
  call stub_pool_put_array(mesh, "verticesOnEdge", verticesOnEdge)
  call stub_pool_put_array(mesh, "baryCellsOnVertex", baryCellsOnVertex)
  call stub_pool_put_array(mesh, "cellsOnVertex", cellsOnVertex)
  call stub_pool_put_array(mesh, "nEdgesOnEdge", nEdgesOnEdge)
  call stub_pool_put_array(mesh, "edgesOnEdge", edgesOnEdge)
  call stub_pool_put_array(mesh, "layerInterfaceSigma", layerInterfaceSigma)
  call stub_pool_put_array(mesh, "layerCenterSigma", layerCenterSigma)
  call stub_pool_put_array(mesh, "dvEdge", dvEdge)
  call stub_pool_put_array(mesh, "baryWeightsOnVertex", baryWeightsOnVertex)
  call stub_pool_put_array(mesh, "kiteAreasOnVertex", kiteAreasOnVertex)
  call stub_pool_put_array(mesh, "weightsOnEdge", weightsOnEdge)

  call stub_pool_put_array(geometry, "thickness", thickness)
  call stub_pool_put_array(geometry, "edgeMask", edgeMask)
  call stub_pool_put_array(geometry, "cellMask", cellMask)
  call stub_pool_put_array(geometry, "slopeEdge", slopeEdge)
  call stub_pool_put_array(geometry, "normalSlopeEdge", normalSlopeEdge)
  call stub_pool_put_array(geometry, "tangentSlopeEdge", tangentSlopeEdge)
  call stub_pool_put_array(geometry, "upperSurfaceVertex", upperSurfaceVertex)
  call stub_pool_put_array(geometry, "upperSurface", upperSurface)

  call stub_pool_put_array(thermal, "temperature", temperature)
  call stub_pool_put_array(velocity, "normalVelocity", normalVelocity)
  call stub_pool_put_array(velocity, "flowParamA", flowParamA)

  call stub_pool_put_config(configs, "config_ice_density", rhoi)
  call stub_pool_put_config(configs, "config_flowLawExponent", flowLawExponent)
  call stub_pool_put_config(configs, "config_enhancementFactor", enhancementFactor)
  call stub_pool_put_config(configs, "config_default_flowParamA", defaultFlowParamA)
  call stub_pool_put_config(configs, "config_dynamic_thickness", dynamicThickness)
  call stub_pool_put_config(configs, "config_sia_tangent_slope_calculation", tangentSlopeCalculation)
  call stub_pool_put_config(configs, "config_flowParamA_calculation", flowParamACalculation)
  liConfigs => configs

  call li_sia_solve(mesh, geometry, thermal, velocity, err)

  open(newunit=u, file=trim(outPath), access='stream', form='unformatted', status='replace')
  write(u) err, stub_log_errors
  write(u) normalVelocity, flowParamA, slopeEdge, tangentSlopeEdge, upperSurfaceVertex
  close(u)

end program run_oracle
