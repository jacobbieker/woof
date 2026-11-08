! Drive four MPAS-Ocean v8.4.2 tendency routines on one block and write every
! output word: the reference the GPU port is held against.
!
!   run_oracle INPUT.bin OUTPUT.bin
!
! INPUT.bin is written by woof.verify.mpas_ocean_tend_oracle (raw
! little-endian words, Fortran array order; reals are RKIND wide).  Each
! routine's init reads its configs from ocnConfigs, then each tendency
! routine adds to its own tendency array, which starts from the case's
! sentinel words:
!
!   ocn_vel_coriolis_tend   -> tendVelCoriolis(nVertLevels, nEdges)
!   ocn_vel_vadv_tend       -> tendVelVadv(nVertLevels, nEdges)
!   ocn_thick_hadv_tend     -> tendThickHadv(nVertLevels, nCells)
!   ocn_thick_vadv_tend     -> tendThickVadv(nVertLevels, nCells)
!
! (mpas_ocn_tendency.F calls them from ocn_tend_vel and ocn_tend_thick; this
! grades each routine on its own, so a mismatch names the routine.)

program run_oracle
  use mpas_kind_types
  use mpas_derived_types
  use mpas_pool_routines
  use ocn_constants
  use ocn_vel_coriolis
  use ocn_vel_vadv
  use ocn_thick_hadv
  use ocn_thick_vadv
  implicit none

  type(MPAS_pool_type), pointer :: mesh, configs
  integer, pointer :: nVertLevels
  integer, dimension(:), pointer :: nCellsArray, nEdgesArray
  integer :: nCells, nEdges, maxEdges, maxEdges2, rk4Flag, err, u
  logical, pointer :: disableCoriolis, disableVelVadv, disableThickHadv, disableThickVadv
  character(len=StrKIND), pointer :: timeIntegrator, vertCoordMovement

  integer, dimension(:), pointer :: maxLevelEdgeTop, maxLevelEdgeBot, maxLevelCell, nEdgesOnEdge, nEdgesOnCell
  integer, dimension(:,:), pointer :: cellsOnEdge, edgesOnEdge, edgeMask, edgesOnCell, edgeSignOnCell
  real(kind=RKIND), dimension(:), pointer :: dcEdge, dvEdge, areaCell
  real(kind=RKIND), dimension(:,:), pointer :: weightsOnEdge
  real(kind=RKIND), dimension(:,:), pointer :: relVortEdge, planVortEdge, layerThicknessEdge, normalVelocity, &
       kineticEnergyCell, vertAleTransportTop
  real(kind=RKIND), dimension(:,:), pointer :: tendVelCoriolis, tendVelVadv, tendThickHadv, tendThickVadv
  character(len=1024) :: inPath, outPath

  call get_command_argument(1, inPath)
  call get_command_argument(2, outPath)

  allocate(nVertLevels, nCellsArray(1), nEdgesArray(1))
  allocate(disableCoriolis, disableVelVadv, disableThickHadv, disableThickVadv, timeIntegrator, vertCoordMovement)

  open(newunit=u, file=trim(inPath), access='stream', form='unformatted', status='old')
  read(u) nCells, nEdges, maxEdges, maxEdges2, nVertLevels, rk4Flag
  nCellsArray(1) = nCells
  nEdgesArray(1) = nEdges
  timeIntegrator = 'split_explicit'
  if (rk4Flag /= 0) timeIntegrator = 'RK4'
  vertCoordMovement = 'uniform_stretching'
  disableCoriolis = .false.
  disableVelVadv = .false.
  disableThickHadv = .false.
  disableThickVadv = .false.

  allocate(maxLevelEdgeTop(nEdges), maxLevelEdgeBot(nEdges), maxLevelCell(nCells), nEdgesOnEdge(nEdges), &
       nEdgesOnCell(nCells), cellsOnEdge(2,nEdges), edgesOnEdge(maxEdges2,nEdges), &
       edgeMask(nVertLevels,nEdges), edgesOnCell(maxEdges,nCells), edgeSignOnCell(maxEdges,nCells))
  read(u) maxLevelEdgeTop, maxLevelEdgeBot, maxLevelCell, nEdgesOnEdge, nEdgesOnCell, cellsOnEdge, &
       edgesOnEdge, edgeMask, edgesOnCell, edgeSignOnCell

  allocate(dcEdge(nEdges), dvEdge(nEdges), areaCell(nCells), weightsOnEdge(maxEdges2,nEdges))
  read(u) dcEdge, dvEdge, areaCell, weightsOnEdge

  allocate(relVortEdge(nVertLevels,nEdges), planVortEdge(nVertLevels,nEdges), &
       layerThicknessEdge(nVertLevels,nEdges), normalVelocity(nVertLevels,nEdges), &
       kineticEnergyCell(nVertLevels,nCells), vertAleTransportTop(nVertLevels+1,nCells))
  read(u) relVortEdge, planVortEdge, layerThicknessEdge, normalVelocity, kineticEnergyCell, vertAleTransportTop

  allocate(tendVelCoriolis(nVertLevels,nEdges), tendVelVadv(nVertLevels,nEdges), &
       tendThickHadv(nVertLevels,nCells), tendThickVadv(nVertLevels,nCells))
  read(u) tendVelCoriolis, tendVelVadv, tendThickHadv, tendThickVadv
  close(u)

  allocate(mesh, configs)
  call stub_pool_put_dimension(mesh, "nVertLevels", nVertLevels)
  call stub_pool_put_dimension(mesh, "nCellsArray", nCellsArray)
  call stub_pool_put_dimension(mesh, "nEdgesArray", nEdgesArray)
  call stub_pool_put_array(mesh, "maxLevelEdgeTop", maxLevelEdgeTop)
  call stub_pool_put_array(mesh, "maxLevelEdgeBot", maxLevelEdgeBot)
  call stub_pool_put_array(mesh, "maxLevelCell", maxLevelCell)
  call stub_pool_put_array(mesh, "nEdgesOnEdge", nEdgesOnEdge)
  call stub_pool_put_array(mesh, "nEdgesOnCell", nEdgesOnCell)
  call stub_pool_put_array(mesh, "cellsOnEdge", cellsOnEdge)
  call stub_pool_put_array(mesh, "edgesOnEdge", edgesOnEdge)
  call stub_pool_put_array(mesh, "edgeMask", edgeMask)
  call stub_pool_put_array(mesh, "edgesOnCell", edgesOnCell)
  call stub_pool_put_array(mesh, "edgeSignOnCell", edgeSignOnCell)
  call stub_pool_put_array(mesh, "dcEdge", dcEdge)
  call stub_pool_put_array(mesh, "dvEdge", dvEdge)
  call stub_pool_put_array(mesh, "areaCell", areaCell)
  call stub_pool_put_array(mesh, "weightsOnEdge", weightsOnEdge)

  call stub_pool_put_config(configs, "config_disable_vel_coriolis", disableCoriolis)
  call stub_pool_put_config(configs, "config_disable_vel_vadv", disableVelVadv)
  call stub_pool_put_config(configs, "config_disable_thick_hadv", disableThickHadv)
  call stub_pool_put_config(configs, "config_disable_thick_vadv", disableThickVadv)
  call stub_pool_put_config(configs, "config_time_integrator", timeIntegrator)
  call stub_pool_put_config(configs, "config_vert_coord_movement", vertCoordMovement)
  ocnConfigs => configs

  call ocn_vel_coriolis_init(err)
  call ocn_vel_vadv_init(err)
  call ocn_thick_hadv_init(err)
  call ocn_thick_vadv_init(err)

  call ocn_vel_coriolis_tend(mesh, relVortEdge, planVortEdge, layerThicknessEdge, normalVelocity, &
       kineticEnergyCell, tendVelCoriolis, err)
  call ocn_vel_vadv_tend(mesh, normalVelocity, layerThicknessEdge, vertAleTransportTop, tendVelVadv, err)
  call ocn_thick_hadv_tend(mesh, normalVelocity, layerThicknessEdge, tendThickHadv, err)
  call ocn_thick_vadv_tend(mesh, vertAleTransportTop, tendThickVadv, err)

  open(newunit=u, file=trim(outPath), access='stream', form='unformatted', status='replace')
  write(u) tendVelCoriolis, tendVelVadv, tendThickHadv, tendThickVadv
  close(u)

end program run_oracle
