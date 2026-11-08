! Drive MPAS-Ocean v8.4.2's ocn_tend_vel and ocn_tend_thick (the full
! momentum and thickness tendencies of mpas_ocn_tendency.F) on one block, and
! the velocity forcing and thickness surface flux on their own, and write every
! output word: the reference the GPU port is held against.
!
!   run_tendency INPUT.bin OUTPUT.bin
!
! INPUT.bin is written by woof.verify.mpas_ocean_tendency_oracle (raw
! little-endian words, Fortran array order; reals are RKIND wide).  The driver
! puts the fields in the pools ocn_tend_vel and ocn_tend_thick read, calls the
! _init routine of every term they call, then
!
!   ocn_tend_vel   -> tend % normalVelocity, forcing % surfaceStress,
!                     surfaceStressMagnitude, diagnostics % viscosity
!   ocn_tend_thick -> tend % layerThickness, forcing % surfaceThicknessFlux,
!                     surfaceThicknessFluxRunoff
!
! and, so the forcing terms are graded with a nonzero stress and flux whatever
! the bulk forcing switches are,
!
!   ocn_vel_forcing_tend(surfaceStressIn)          -> tendVelForcing
!   ocn_thick_surface_flux_tend(fluxIn, runoffIn)  -> tendThickSflux
!
! Every output array starts from the case's sentinel words.  Land-ice fluxes
! and frazil are off (their stand-ins stop if asked to run).

program run_tendency
  use mpas_kind_types
  use mpas_derived_types
  use mpas_pool_routines
  use ocn_constants
  use ocn_tendency
  use ocn_vel_coriolis
  use ocn_vel_vadv
  use ocn_thick_hadv
  use ocn_thick_vadv
  use ocn_vel_pressure_grad
  use ocn_vel_hmix
  use ocn_vel_forcing
  use ocn_thick_surface_flux
  use ocn_surface_bulk_forcing
  use ocn_forcing
  implicit none

  character(len=*), parameter :: pgradTypes(5) = [character(len=32) :: 'pressure_and_zmid', &
       'MontgomeryPotential', 'MontgomeryPotential_and_density', 'Jacobian_from_density', 'Jacobian_from_TS']
  integer, parameter :: nFlags = 14, nReals = 11

  type(MPAS_pool_type), pointer :: mesh, configs, packages, scratch, state, tracersPool, diag, forcing, tendP
  integer, pointer :: nVertLevels, vertexDegree, nCellsDim, nEdgesDim, indexT, indexS
  integer, dimension(:), pointer :: nCellsArray, nEdgesArray, nVerticesArray
  integer :: nCells, nEdges, nVertices, maxEdges, maxEdges2, nTracers, pgradType, err, u, i
  integer :: flagWords(nFlags)
  real(kind=RKIND) :: realWords(nReals)
  logical, pointer :: flagConfig, falseConfig
  real(kind=RKIND), pointer :: realValue
  character(len=StrKIND), pointer :: pgradTypeConfig, timeIntegrator, vertCoordMovement, landIceFluxMode

  integer, dimension(:), pointer :: maxLevelEdgeTop, maxLevelEdgeBot, maxLevelCell, maxLevelVertexTop, &
       nEdgesOnEdge, nEdgesOnCell
  integer, dimension(:,:), pointer :: cellsOnEdge, verticesOnEdge, edgesOnEdge, edgeMask, edgesOnVertex, &
       edgesOnCell, edgeSignOnVertex, edgeSignOnCell
  real(kind=RKIND), dimension(:), pointer :: dcEdge, dvEdge, areaCell, areaTriangle, meshScalingDel2, &
       meshScalingDel4, meshScaling, angleEdge
  real(kind=RKIND), dimension(:,:), pointer :: weightsOnEdge
  real(kind=RKIND), dimension(:,:), pointer :: normalVelocity
  real(kind=RKIND), dimension(:,:,:), pointer :: activeTracers
  real(kind=RKIND), dimension(:,:), pointer :: kineticEnergyCell, layerThicknessEdge, vertAleTransportTop, zMid, &
       relativeVorticity, normalizedRelativeVorticityEdge, normalizedPlanetaryVorticityEdge, divergence, &
       viscosity, montgomeryPotential, pressure, vertViscTopOfEdge, density, potentialDensity, &
       tangentialVelocity, thermalExpansion, salineContraction, normalTransportVelocity
  real(kind=RKIND), dimension(:), pointer :: surfaceFluxAttenuationCoefficient
  real(kind=RKIND), dimension(:), pointer :: windStressZonal, windStressMeridional, evaporationFlux, snowFlux, &
       seaIceFreshWaterFlux, riverRunoffFlux, iceRunoffFlux, rainFlux, surfaceStress, surfaceStressMagnitude, &
       surfaceThicknessFlux, surfaceThicknessFluxRunoff, surfaceStressIn, fluxIn, runoffIn
  real(kind=RKIND), dimension(:,:), pointer :: fractionAbsorbed, fractionAbsorbedRunoff, layerThickness
  real(kind=RKIND), dimension(:,:), pointer :: tendNormalVelocity, tendLayerThickness, tendVelForcing, tendThickSflux
  type(field2DReal), pointer :: delsqU, delsqDivergence, delsqRelativeVorticity
  character(len=1024) :: inPath, outPath
  character(len=48), parameter :: flagNames(nFlags) = [character(len=48) :: &
       'config_disable_vel_all_tend', 'config_disable_thick_all_tend', 'config_use_mom_del2', &
       'config_use_Leith_del2', 'config_use_mom_del4', 'config_use_bulk_wind_stress', &
       'config_use_bulk_thickness_flux', 'config_disable_vel_surface_stress', 'config_use_explicit_bottom_drag', &
       'config_Rayleigh_friction', 'config_Rayleigh_bottom_friction', 'config_disable_thick_sflux', &
       'config_disable_vel_hmix', 'config_disable_vel_pgrad']
  character(len=48), parameter :: realNames(nReals) = [character(len=48) :: &
       'config_density0', 'config_common_level_weight', 'config_mom_del2', 'config_mom_del4', &
       'config_mom_del4_div_factor', 'config_Leith_parameter', 'config_Leith_dx', 'config_Leith_visc2_max', &
       'config_explicit_bottom_drag_coeff', 'config_Rayleigh_damping_coeff', 'config_Rayleigh_bottom_damping_coeff']
  character(len=48), parameter :: falseNames(9) = [character(len=48) :: &
       'config_disable_vel_coriolis', 'config_disable_vel_vadv', 'config_disable_thick_hadv', &
       'config_disable_thick_vadv', 'config_use_mom_del2_tensor', 'config_use_mom_del4_tensor', &
       'config_disable_vel_explicit_bottom_drag', 'config_use_frazil_ice_formation', &
       'config_use_highFreqThick_restore']

  call get_command_argument(1, inPath)
  call get_command_argument(2, outPath)

  allocate(nVertLevels, vertexDegree, nCellsDim, nEdgesDim, indexT, indexS)
  allocate(nCellsArray(4), nEdgesArray(4), nVerticesArray(4))
  allocate(falseConfig, pgradTypeConfig, timeIntegrator, vertCoordMovement, landIceFluxMode)

  open(newunit=u, file=trim(inPath), access='stream', form='unformatted', status='old')
  read(u) nCells, nEdges, nVertices, maxEdges, maxEdges2, vertexDegree, nVertLevels, nTracers, indexT, indexS, &
       pgradType, i
  read(u) flagWords
  read(u) realWords
  ! one block, every element owned: each halo extent is the owned count
  nCellsDim = nCells
  nEdgesDim = nEdges
  nCellsArray = nCells
  nEdgesArray = nEdges
  nVerticesArray = nVertices
  pgradTypeConfig = pgradTypes(pgradType)
  timeIntegrator = 'split_explicit'
  if (i /= 0) timeIntegrator = 'RK4'
  vertCoordMovement = 'uniform_stretching'
  landIceFluxMode = 'off'
  falseConfig = .false.

  allocate(maxLevelEdgeTop(nEdges), maxLevelEdgeBot(nEdges), maxLevelCell(nCells), maxLevelVertexTop(nVertices), &
       nEdgesOnEdge(nEdges), nEdgesOnCell(nCells), cellsOnEdge(2,nEdges), verticesOnEdge(2,nEdges), &
       edgesOnEdge(maxEdges2,nEdges), edgeMask(nVertLevels,nEdges), edgesOnVertex(vertexDegree,nVertices), &
       edgesOnCell(maxEdges,nCells), edgeSignOnVertex(vertexDegree,nVertices), edgeSignOnCell(maxEdges,nCells))
  read(u) maxLevelEdgeTop, maxLevelEdgeBot, maxLevelCell, maxLevelVertexTop, nEdgesOnEdge, nEdgesOnCell, &
       cellsOnEdge, verticesOnEdge, edgesOnEdge, edgeMask, edgesOnVertex, edgesOnCell, edgeSignOnVertex, &
       edgeSignOnCell

  allocate(dcEdge(nEdges), dvEdge(nEdges), areaCell(nCells), areaTriangle(nVertices), &
       weightsOnEdge(maxEdges2,nEdges), meshScalingDel2(nEdges), meshScalingDel4(nEdges), meshScaling(nEdges), &
       angleEdge(nEdges))
  read(u) dcEdge, dvEdge, areaCell, areaTriangle, weightsOnEdge, meshScalingDel2, meshScalingDel4, meshScaling, &
       angleEdge

  allocate(normalVelocity(nVertLevels,nEdges), activeTracers(nTracers,nVertLevels,nCells))
  read(u) normalVelocity, activeTracers

  allocate(kineticEnergyCell(nVertLevels,nCells), layerThicknessEdge(nVertLevels,nEdges), &
       vertAleTransportTop(nVertLevels+1,nCells), zMid(nVertLevels,nCells), relativeVorticity(nVertLevels,nVertices), &
       normalizedRelativeVorticityEdge(nVertLevels,nEdges), normalizedPlanetaryVorticityEdge(nVertLevels,nEdges), &
       divergence(nVertLevels,nCells), viscosity(nVertLevels,nEdges), montgomeryPotential(nVertLevels,nCells), &
       pressure(nVertLevels,nCells), vertViscTopOfEdge(nVertLevels+1,nEdges), density(nVertLevels,nCells), &
       potentialDensity(nVertLevels,nCells), tangentialVelocity(nVertLevels,nEdges), &
       thermalExpansion(nVertLevels,nCells), salineContraction(nVertLevels,nCells), &
       normalTransportVelocity(nVertLevels,nEdges), surfaceFluxAttenuationCoefficient(nCells))
  read(u) kineticEnergyCell, layerThicknessEdge, vertAleTransportTop, zMid, relativeVorticity, &
       normalizedRelativeVorticityEdge, normalizedPlanetaryVorticityEdge, divergence, montgomeryPotential, &
       pressure, density, potentialDensity, tangentialVelocity, thermalExpansion, salineContraction, &
       normalTransportVelocity, surfaceFluxAttenuationCoefficient
  vertViscTopOfEdge = 0.0_RKIND

  allocate(windStressZonal(nCells), windStressMeridional(nCells), evaporationFlux(nCells), snowFlux(nCells), &
       seaIceFreshWaterFlux(nCells), riverRunoffFlux(nCells), iceRunoffFlux(nCells), rainFlux(nCells), &
       fractionAbsorbed(nVertLevels,nCells), fractionAbsorbedRunoff(nVertLevels,nCells), &
       surfaceStressIn(nEdges), fluxIn(nCells), runoffIn(nCells))
  read(u) windStressZonal, windStressMeridional, evaporationFlux, snowFlux, seaIceFreshWaterFlux, riverRunoffFlux, &
       iceRunoffFlux, rainFlux, fractionAbsorbed, fractionAbsorbedRunoff, surfaceStressIn, fluxIn, runoffIn

  allocate(tendNormalVelocity(nVertLevels,nEdges), surfaceStress(nEdges), surfaceStressMagnitude(nCells), &
       tendLayerThickness(nVertLevels,nCells), surfaceThicknessFlux(nCells), surfaceThicknessFluxRunoff(nCells), &
       tendVelForcing(nVertLevels,nEdges), tendThickSflux(nVertLevels,nCells), layerThickness(nVertLevels,nCells))
  read(u) tendNormalVelocity, surfaceStress, surfaceStressMagnitude, viscosity, tendLayerThickness, &
       surfaceThicknessFlux, surfaceThicknessFluxRunoff, tendVelForcing, tendThickSflux
  close(u)
  layerThickness = 0.0_RKIND

  allocate(mesh, configs, packages, scratch, state, tracersPool, diag, forcing, tendP)
  call stub_pool_put_dimension(mesh, "nVertLevels", nVertLevels)
  call stub_pool_put_dimension(mesh, "vertexDegree", vertexDegree)
  call stub_pool_put_dimension(mesh, "nCells", nCellsDim)
  call stub_pool_put_dimension(mesh, "nEdges", nEdgesDim)
  call stub_pool_put_dimension(mesh, "nCellsArray", nCellsArray)
  call stub_pool_put_dimension(mesh, "nEdgesArray", nEdgesArray)
  call stub_pool_put_dimension(mesh, "nVerticesArray", nVerticesArray)
  call stub_pool_put_array(mesh, "maxLevelEdgeTop", maxLevelEdgeTop)
  call stub_pool_put_array(mesh, "maxLevelEdgeBot", maxLevelEdgeBot)
  call stub_pool_put_array(mesh, "maxLevelCell", maxLevelCell)
  call stub_pool_put_array(mesh, "maxLevelVertexTop", maxLevelVertexTop)
  call stub_pool_put_array(mesh, "nEdgesOnEdge", nEdgesOnEdge)
  call stub_pool_put_array(mesh, "nEdgesOnCell", nEdgesOnCell)
  call stub_pool_put_array(mesh, "cellsOnEdge", cellsOnEdge)
  call stub_pool_put_array(mesh, "verticesOnEdge", verticesOnEdge)
  call stub_pool_put_array(mesh, "edgesOnEdge", edgesOnEdge)
  call stub_pool_put_array(mesh, "edgeMask", edgeMask)
  call stub_pool_put_array(mesh, "edgesOnVertex", edgesOnVertex)
  call stub_pool_put_array(mesh, "edgesOnCell", edgesOnCell)
  call stub_pool_put_array(mesh, "edgeSignOnVertex", edgeSignOnVertex)
  call stub_pool_put_array(mesh, "edgeSignOnCell", edgeSignOnCell)
  call stub_pool_put_array(mesh, "dcEdge", dcEdge)
  call stub_pool_put_array(mesh, "dvEdge", dvEdge)
  call stub_pool_put_array(mesh, "areaCell", areaCell)
  call stub_pool_put_array(mesh, "areaTriangle", areaTriangle)
  call stub_pool_put_array(mesh, "weightsOnEdge", weightsOnEdge)
  call stub_pool_put_array(mesh, "meshScalingDel2", meshScalingDel2)
  call stub_pool_put_array(mesh, "meshScalingDel4", meshScalingDel4)
  call stub_pool_put_array(mesh, "meshScaling", meshScaling)
  call stub_pool_put_array(mesh, "angleEdge", angleEdge)

  call stub_pool_put_array(state, "normalVelocity", normalVelocity)
  call stub_pool_put_subpool(state, "tracers", tracersPool)
  call stub_pool_put_array(tracersPool, "activeTracers", activeTracers)
  call stub_pool_put_dimension(tracersPool, "index_temperature", indexT)
  call stub_pool_put_dimension(tracersPool, "index_salinity", indexS)

  call stub_pool_put_array(diag, "kineticEnergyCell", kineticEnergyCell)
  call stub_pool_put_array(diag, "layerThicknessEdge", layerThicknessEdge)
  call stub_pool_put_array(diag, "vertAleTransportTop", vertAleTransportTop)
  call stub_pool_put_array(diag, "zMid", zMid)
  call stub_pool_put_array(diag, "relativeVorticity", relativeVorticity)
  call stub_pool_put_array(diag, "normalizedRelativeVorticityEdge", normalizedRelativeVorticityEdge)
  call stub_pool_put_array(diag, "normalizedPlanetaryVorticityEdge", normalizedPlanetaryVorticityEdge)
  call stub_pool_put_array(diag, "divergence", divergence)
  call stub_pool_put_array(diag, "viscosity", viscosity)
  call stub_pool_put_array(diag, "montgomeryPotential", montgomeryPotential)
  call stub_pool_put_array(diag, "pressure", pressure)
  call stub_pool_put_array(diag, "vertViscTopOfEdge", vertViscTopOfEdge)
  call stub_pool_put_array(diag, "density", density)
  call stub_pool_put_array(diag, "potentialDensity", potentialDensity)
  call stub_pool_put_array(diag, "tangentialVelocity", tangentialVelocity)
  call stub_pool_put_array(diag, "surfaceFluxAttenuationCoefficient", surfaceFluxAttenuationCoefficient)
  call stub_pool_put_array(diag, "inSituThermalExpansionCoeff", thermalExpansion)
  call stub_pool_put_array(diag, "inSituSalineContractionCoeff", salineContraction)
  call stub_pool_put_array(diag, "normalTransportVelocity", normalTransportVelocity)

  call stub_pool_put_array(forcing, "windStressZonal", windStressZonal)
  call stub_pool_put_array(forcing, "windStressMeridional", windStressMeridional)
  call stub_pool_put_array(forcing, "evaporationFlux", evaporationFlux)
  call stub_pool_put_array(forcing, "snowFlux", snowFlux)
  call stub_pool_put_array(forcing, "seaIceFreshWaterFlux", seaIceFreshWaterFlux)
  call stub_pool_put_array(forcing, "riverRunoffFlux", riverRunoffFlux)
  call stub_pool_put_array(forcing, "iceRunoffFlux", iceRunoffFlux)
  call stub_pool_put_array(forcing, "rainFlux", rainFlux)
  call stub_pool_put_array(forcing, "fractionAbsorbed", fractionAbsorbed)
  call stub_pool_put_array(forcing, "fractionAbsorbedRunoff", fractionAbsorbedRunoff)
  call stub_pool_put_array(forcing, "surfaceStress", surfaceStress)
  call stub_pool_put_array(forcing, "surfaceStressMagnitude", surfaceStressMagnitude)
  call stub_pool_put_array(forcing, "surfaceThicknessFlux", surfaceThicknessFlux)
  call stub_pool_put_array(forcing, "surfaceThicknessFluxRunoff", surfaceThicknessFluxRunoff)

  call stub_pool_put_array(tendP, "normalVelocity", tendNormalVelocity)
  call stub_pool_put_array(tendP, "layerThickness", tendLayerThickness)

  allocate(delsqU, delsqDivergence, delsqRelativeVorticity)
  delsqU % dimSizes = [nVertLevels, nEdges]
  delsqDivergence % dimSizes = [nVertLevels, nCells]
  delsqRelativeVorticity % dimSizes = [nVertLevels, nVertices]
  call stub_pool_put_field(scratch, "delsq_u", delsqU)
  call stub_pool_put_field(scratch, "delsq_divergence", delsqDivergence)
  call stub_pool_put_field(scratch, "delsq_relativeVorticity", delsqRelativeVorticity)

  ! each config is its own target, as the pool hands out pointers
  do i = 1, nFlags
    allocate(flagConfig)
    flagConfig = flagWords(i) /= 0
    call stub_pool_put_config(configs, trim(flagNames(i)), flagConfig)
  end do
  do i = 1, nReals
    allocate(realValue)
    realValue = realWords(i)
    call stub_pool_put_config(configs, trim(realNames(i)), realValue)
  end do
  do i = 1, size(falseNames)
    call stub_pool_put_config(configs, trim(falseNames(i)), falseConfig)
  end do
  call stub_pool_put_config(configs, "config_pressure_gradient_type", pgradTypeConfig)
  call stub_pool_put_config(configs, "config_time_integrator", timeIntegrator)
  call stub_pool_put_config(configs, "config_vert_coord_movement", vertCoordMovement)
  call stub_pool_put_config(configs, "config_land_ice_flux_mode", landIceFluxMode)

  call ocn_constants_init(configs, packages)
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

  call ocn_tend_vel(tendP, state, forcing, diag, mesh, scratch)
  call ocn_tend_thick(tendP, forcing, diag, mesh)

  call ocn_vel_forcing_tend(mesh, normalVelocity, surfaceFluxAttenuationCoefficient, surfaceStressIn, &
       kineticEnergyCell, layerThicknessEdge, tendVelForcing, err)
  call ocn_thick_surface_flux_tend(mesh, fractionAbsorbed, fractionAbsorbedRunoff, layerThickness, fluxIn, &
       runoffIn, tendThickSflux, err)

  open(newunit=u, file=trim(outPath), access='stream', form='unformatted', status='replace')
  write(u) tendNormalVelocity, surfaceStress, surfaceStressMagnitude, viscosity, tendLayerThickness, &
       surfaceThicknessFlux, surfaceThicknessFluxRunoff, tendVelForcing, tendThickSflux
  close(u)

end program run_tendency
