! Drive MPAS-Ocean v8.4.2's horizontal pressure gradient and horizontal
! momentum mixing on one block and write every output word: the reference the
! GPU port is held against.
!
!   run_pgrad_hmix INPUT.bin OUTPUT.bin
!
! INPUT.bin is written by woof.verify.mpas_ocean_pgrad_hmix_oracle (raw
! little-endian words, Fortran array order; reals are RKIND wide).
! ocn_constants_init sets rho_sw from config_density0, the two _init routines
! read their configs from ocnConfigs, then:
!
!   ocn_vel_pressure_grad_tend -> tendPgrad(nVertLevels, nEdges)
!   ocn_vel_hmix_tend          -> tendHmix(nVertLevels, nEdges), viscosity
!
! each adding into its own tendency array, which starts from the case's
! sentinel words.  ocn_vel_hmix_tend zeroes viscosity and calls del2,
! del2_tensor, Leith, del4 and del4_tensor in that order; the tensor forms are
! off (config_use_mom_del{2,4}_tensor false).  The pressure gradient is passed
! the in-situ expansion coefficients whatever its type, as ocn_tend_vel does.

program run_pgrad_hmix
  use mpas_kind_types
  use mpas_derived_types
  use mpas_pool_routines
  use ocn_constants
  use ocn_vel_pressure_grad
  use ocn_vel_hmix
  implicit none

  character(len=*), parameter :: pgradTypes(5) = [character(len=32) :: 'pressure_and_zmid', &
       'MontgomeryPotential', 'MontgomeryPotential_and_density', 'Jacobian_from_density', 'Jacobian_from_TS']

  type(MPAS_pool_type), pointer :: mesh, configs, packages, scratch
  integer, pointer :: nVertLevels, vertexDegree
  integer, dimension(:), pointer :: nCellsArray, nEdgesArray, nVerticesArray
  integer :: nCells, nEdges, nVertices, maxEdges, nTracers, indexT, indexS, pgradType, err, u
  integer :: flags(6)
  logical, pointer :: disablePgrad, disableHmix, useDel2, useDel4, useLeith, useDel2Tensor, useDel4Tensor
  character(len=StrKIND), pointer :: pgradTypeConfig
  real(kind=RKIND), pointer :: density0, commonLevelWeight, momDel2, momDel4, momDel4DivFactor, &
       leithParameter, leithDx, leithVisc2Max
  real(kind=RKIND) :: reals(8)

  integer, dimension(:), pointer :: maxLevelEdgeTop, maxLevelCell, maxLevelVertexTop, nEdgesOnCell
  integer, dimension(:,:), pointer :: cellsOnEdge, verticesOnEdge, edgeMask, edgesOnVertex, edgesOnCell, &
       edgeSignOnVertex, edgeSignOnCell
  real(kind=RKIND), dimension(:), pointer :: dcEdge, dvEdge, areaCell, areaTriangle, meshScalingDel2, &
       meshScalingDel4, meshScaling
  real(kind=RKIND), dimension(:,:), pointer :: pressure, montgomeryPotential, zMid, density, potentialDensity, &
       thermalExpansion, salineContraction, divergence, relativeVorticity, normalVelocity, tangentialVelocity
  real(kind=RKIND), dimension(:,:,:), pointer :: tracers
  real(kind=RKIND), dimension(:,:), pointer :: tendPgrad, tendHmix, viscosity
  type(field2DReal), pointer :: delsqU, delsqDivergence, delsqRelativeVorticity
  character(len=1024) :: inPath, outPath

  call get_command_argument(1, inPath)
  call get_command_argument(2, outPath)

  allocate(nVertLevels, vertexDegree, nCellsArray(4), nEdgesArray(4), nVerticesArray(4))
  allocate(disablePgrad, disableHmix, useDel2, useDel4, useLeith, useDel2Tensor, useDel4Tensor, pgradTypeConfig)
  allocate(density0, commonLevelWeight, momDel2, momDel4, momDel4DivFactor, leithParameter, leithDx, leithVisc2Max)

  open(newunit=u, file=trim(inPath), access='stream', form='unformatted', status='old')
  read(u) nCells, nEdges, nVertices, maxEdges, vertexDegree, nVertLevels, nTracers, indexT, indexS, pgradType
  ! flags: disable_vel_pgrad, disable_vel_hmix, use_mom_del2, use_Leith_del2, use_mom_del4, (unused)
  read(u) flags
  read(u) reals
  ! one block, every element owned: each halo extent is the owned count
  nCellsArray = nCells
  nEdgesArray = nEdges
  nVerticesArray = nVertices
  pgradTypeConfig = pgradTypes(pgradType)
  disablePgrad = flags(1) /= 0
  disableHmix = flags(2) /= 0
  useDel2 = flags(3) /= 0
  useLeith = flags(4) /= 0
  useDel4 = flags(5) /= 0
  useDel2Tensor = .false.
  useDel4Tensor = .false.
  density0 = reals(1)
  commonLevelWeight = reals(2)
  momDel2 = reals(3)
  momDel4 = reals(4)
  momDel4DivFactor = reals(5)
  leithParameter = reals(6)
  leithDx = reals(7)
  leithVisc2Max = reals(8)

  allocate(maxLevelEdgeTop(nEdges), maxLevelCell(nCells), maxLevelVertexTop(nVertices), nEdgesOnCell(nCells), &
       cellsOnEdge(2,nEdges), verticesOnEdge(2,nEdges), edgeMask(nVertLevels,nEdges), &
       edgesOnVertex(vertexDegree,nVertices), edgesOnCell(maxEdges,nCells), &
       edgeSignOnVertex(vertexDegree,nVertices), edgeSignOnCell(maxEdges,nCells))
  read(u) maxLevelEdgeTop, maxLevelCell, maxLevelVertexTop, nEdgesOnCell, cellsOnEdge, verticesOnEdge, &
       edgeMask, edgesOnVertex, edgesOnCell, edgeSignOnVertex, edgeSignOnCell

  allocate(dcEdge(nEdges), dvEdge(nEdges), areaCell(nCells), areaTriangle(nVertices), meshScalingDel2(nEdges), &
       meshScalingDel4(nEdges), meshScaling(nEdges))
  read(u) dcEdge, dvEdge, areaCell, areaTriangle, meshScalingDel2, meshScalingDel4, meshScaling

  allocate(pressure(nVertLevels,nCells), montgomeryPotential(nVertLevels,nCells), zMid(nVertLevels,nCells), &
       density(nVertLevels,nCells), potentialDensity(nVertLevels,nCells), thermalExpansion(nVertLevels,nCells), &
       salineContraction(nVertLevels,nCells), tracers(nTracers,nVertLevels,nCells), &
       divergence(nVertLevels,nCells), relativeVorticity(nVertLevels,nVertices), &
       normalVelocity(nVertLevels,nEdges), tangentialVelocity(nVertLevels,nEdges))
  read(u) pressure, montgomeryPotential, zMid, density, potentialDensity, thermalExpansion, salineContraction, &
       tracers, divergence, relativeVorticity, normalVelocity, tangentialVelocity

  allocate(tendPgrad(nVertLevels,nEdges), tendHmix(nVertLevels,nEdges), viscosity(nVertLevels,nEdges))
  read(u) tendPgrad, tendHmix, viscosity
  close(u)

  allocate(mesh, configs, packages, scratch)
  call stub_pool_put_dimension(mesh, "nVertLevels", nVertLevels)
  call stub_pool_put_dimension(mesh, "vertexDegree", vertexDegree)
  call stub_pool_put_dimension(mesh, "nCellsArray", nCellsArray)
  call stub_pool_put_dimension(mesh, "nEdgesArray", nEdgesArray)
  call stub_pool_put_dimension(mesh, "nVerticesArray", nVerticesArray)
  call stub_pool_put_array(mesh, "maxLevelEdgeTop", maxLevelEdgeTop)
  call stub_pool_put_array(mesh, "maxLevelCell", maxLevelCell)
  call stub_pool_put_array(mesh, "maxLevelVertexTop", maxLevelVertexTop)
  call stub_pool_put_array(mesh, "nEdgesOnCell", nEdgesOnCell)
  call stub_pool_put_array(mesh, "cellsOnEdge", cellsOnEdge)
  call stub_pool_put_array(mesh, "verticesOnEdge", verticesOnEdge)
  call stub_pool_put_array(mesh, "edgeMask", edgeMask)
  call stub_pool_put_array(mesh, "edgesOnVertex", edgesOnVertex)
  call stub_pool_put_array(mesh, "edgesOnCell", edgesOnCell)
  call stub_pool_put_array(mesh, "edgeSignOnVertex", edgeSignOnVertex)
  call stub_pool_put_array(mesh, "edgeSignOnCell", edgeSignOnCell)
  call stub_pool_put_array(mesh, "dcEdge", dcEdge)
  call stub_pool_put_array(mesh, "dvEdge", dvEdge)
  call stub_pool_put_array(mesh, "areaCell", areaCell)
  call stub_pool_put_array(mesh, "areaTriangle", areaTriangle)
  call stub_pool_put_array(mesh, "meshScalingDel2", meshScalingDel2)
  call stub_pool_put_array(mesh, "meshScalingDel4", meshScalingDel4)
  call stub_pool_put_array(mesh, "meshScaling", meshScaling)

  allocate(delsqU, delsqDivergence, delsqRelativeVorticity)
  delsqU % dimSizes = [nVertLevels, nEdges]
  delsqDivergence % dimSizes = [nVertLevels, nCells]
  delsqRelativeVorticity % dimSizes = [nVertLevels, nVertices]
  call stub_pool_put_field(scratch, "delsq_u", delsqU)
  call stub_pool_put_field(scratch, "delsq_divergence", delsqDivergence)
  call stub_pool_put_field(scratch, "delsq_relativeVorticity", delsqRelativeVorticity)

  call stub_pool_put_config(configs, "config_density0", density0)
  call stub_pool_put_config(configs, "config_pressure_gradient_type", pgradTypeConfig)
  call stub_pool_put_config(configs, "config_common_level_weight", commonLevelWeight)
  call stub_pool_put_config(configs, "config_disable_vel_pgrad", disablePgrad)
  call stub_pool_put_config(configs, "config_disable_vel_hmix", disableHmix)
  call stub_pool_put_config(configs, "config_mom_del2", momDel2)
  call stub_pool_put_config(configs, "config_use_mom_del2", useDel2)
  call stub_pool_put_config(configs, "config_use_mom_del2_tensor", useDel2Tensor)
  call stub_pool_put_config(configs, "config_use_Leith_del2", useLeith)
  call stub_pool_put_config(configs, "config_Leith_parameter", leithParameter)
  call stub_pool_put_config(configs, "config_Leith_dx", leithDx)
  call stub_pool_put_config(configs, "config_Leith_visc2_max", leithVisc2Max)
  call stub_pool_put_config(configs, "config_mom_del4", momDel4)
  call stub_pool_put_config(configs, "config_mom_del4_div_factor", momDel4DivFactor)
  call stub_pool_put_config(configs, "config_use_mom_del4", useDel4)
  call stub_pool_put_config(configs, "config_use_mom_del4_tensor", useDel4Tensor)

  call ocn_constants_init(configs, packages)
  call ocn_vel_pressure_grad_init(err)
  call ocn_vel_hmix_init(err)

  call ocn_vel_pressure_grad_tend(mesh, pressure, montgomeryPotential, zMid, density, potentialDensity, &
       indexT, indexS, tracers, tendPgrad, err, thermalExpansion, salineContraction)
  if (err /= 0) stop 9
  call ocn_vel_hmix_tend(mesh, scratch, divergence, relativeVorticity, normalVelocity, tangentialVelocity, &
       viscosity, tendHmix, err)
  if (err /= 0) stop 9

  open(newunit=u, file=trim(outPath), access='stream', form='unformatted', status='replace')
  write(u) tendPgrad, tendHmix, viscosity
  close(u)

end program run_pgrad_hmix
