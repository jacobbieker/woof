! Drive MPAS-Seaice v8.4.2's variational EVP internal stress on one block and
! write every output word: the reference the GPU port is held against.
!
!   run_oracle INPUT.bin OUTPUT.bin
!
! INPUT.bin is written by make_inputs.py (raw little-endian int32/real words,
! Fortran array order, the layout documented there).  The driver puts the
! fields in the pools the routines read, then calls the routines exactly as
! the model does each elastic subcycle:
!
!   seaice_init_evp(domain)                          once
!   seaice_internal_stress_variational(domain)       nIter times
!   seaice_final_divergence_shear_variational(block) once
!
! (mpas_seaice_velocity_solver.F: seaice_init_evp at init, the internal
! stress inside the elastic subcycle, the final divergence/shear after it.)
! Velocities are held fixed across the nIter calls, so the stress history the
! constitutive relation integrates is the only state carried between them.

program run_oracle
  use mpas_kind_types
  use mpas_derived_types
  use mpas_pool_routines
  use seaice_velocity_solver_constitutive_relation
  use seaice_velocity_solver_variational
  implicit none

  type(domain_type) :: domain
  type(block_type), pointer :: block
  type(MPAS_pool_type), pointer :: structs, configs, mesh, solver, variational, ridging

  integer, pointer :: nCells, nVertices, nEdges, maxEdges, vertexDegree, nEdgesSolve
  logical, pointer :: revisedEVP, useColumn
  integer :: revisedFlag, columnFlag, nIter, iter, u
  real(kind=RKIND), pointer :: dynamicsTimeStep, elasticTimeStep

  integer, dimension(:), pointer :: nEdgesOnCell, solveStress, solveVelocity
  integer, dimension(:,:), pointer :: verticesOnCell, cellsOnVertex, cellVerticesAtVertex
  real(kind=RKIND), dimension(:), pointer :: dvEdge, areaCell, areaTriangle, &
       uVelocity, vVelocity, icePressure, tanLat, stressDivergenceU, stressDivergenceV, &
       divergence, shear, ridgeConvergence, ridgeShear
  real(kind=RKIND), dimension(:,:), pointer :: strain11, strain22, strain12, &
       stress11, stress22, stress12, replacementPressure
  real(kind=RKIND), dimension(:,:,:), pointer :: basisGradientU, basisGradientV, &
       basisIntegralsU, basisIntegralsV, basisIntegralsMetric
  character(len=1024) :: inPath, outPath

  call get_command_argument(1, inPath)
  call get_command_argument(2, outPath)

  allocate(nCells, nVertices, nEdges, maxEdges, vertexDegree, nEdgesSolve)
  allocate(revisedEVP, useColumn, dynamicsTimeStep, elasticTimeStep)

  open(newunit=u, file=trim(inPath), access='stream', form='unformatted', status='old')
  read(u) nCells, nVertices, nEdges, maxEdges, vertexDegree, revisedFlag, columnFlag, nIter
  read(u) dynamicsTimeStep, elasticTimeStep
  revisedEVP = revisedFlag /= 0
  useColumn = columnFlag /= 0
  nEdgesSolve = nEdges

  allocate(nEdgesOnCell(nCells), verticesOnCell(maxEdges,nCells), &
       cellsOnVertex(vertexDegree,nVertices), cellVerticesAtVertex(vertexDegree,nVertices), &
       solveStress(nCells), solveVelocity(nVertices))
  read(u) nEdgesOnCell, verticesOnCell, cellsOnVertex, cellVerticesAtVertex, solveStress, solveVelocity

  allocate(dvEdge(nEdges), areaCell(nCells), areaTriangle(nVertices), uVelocity(nVertices), &
       vVelocity(nVertices), icePressure(nCells), tanLat(nVertices))
  read(u) dvEdge, areaCell, areaTriangle, uVelocity, vVelocity, icePressure, tanLat

  allocate(basisGradientU(maxEdges,maxEdges,nCells), basisGradientV(maxEdges,maxEdges,nCells), &
       basisIntegralsU(maxEdges,maxEdges,nCells), basisIntegralsV(maxEdges,maxEdges,nCells), &
       basisIntegralsMetric(maxEdges,maxEdges,nCells))
  read(u) basisGradientU, basisGradientV, basisIntegralsU, basisIntegralsV, basisIntegralsMetric

  ! the prior contents of every output: the routines leave the entries they
  ! do not solve (solveStress/solveVelocity = 0) as they were
  allocate(strain11(maxEdges,nCells), strain22(maxEdges,nCells), strain12(maxEdges,nCells), &
       stress11(maxEdges,nCells), stress22(maxEdges,nCells), stress12(maxEdges,nCells), &
       replacementPressure(maxEdges,nCells), stressDivergenceU(nVertices), &
       stressDivergenceV(nVertices), divergence(nCells), shear(nCells), &
       ridgeConvergence(nCells), ridgeShear(nCells))
  read(u) strain11, strain22, strain12, stress11, stress22, stress12, replacementPressure, &
       stressDivergenceU, stressDivergenceV, divergence, shear, ridgeConvergence, ridgeShear
  close(u)

  allocate(structs, configs, mesh, solver, variational, ridging)
  call stub_pool_put_config(configs, "config_revised_evp", revisedEVP)
  call stub_pool_put_config(configs, "config_use_column_package", useColumn)

  call stub_pool_put_dimension(mesh, "nCells", nCells)
  call stub_pool_put_dimension(mesh, "nVertices", nVertices)
  call stub_pool_put_dimension(mesh, "maxEdges", maxEdges)
  call stub_pool_put_dimension(mesh, "vertexDegree", vertexDegree)
  call stub_pool_put_dimension(mesh, "nEdgesSolve", nEdgesSolve)
  call stub_pool_put_array(mesh, "nEdgesOnCell", nEdgesOnCell)
  call stub_pool_put_array(mesh, "verticesOnCell", verticesOnCell)
  call stub_pool_put_array(mesh, "cellsOnVertex", cellsOnVertex)
  call stub_pool_put_array(mesh, "dvEdge", dvEdge)
  call stub_pool_put_array(mesh, "areaCell", areaCell)
  call stub_pool_put_array(mesh, "areaTriangle", areaTriangle)

  call stub_pool_put_array(solver, "dynamicsTimeStep", dynamicsTimeStep)
  call stub_pool_put_array(solver, "elasticTimeStep", elasticTimeStep)
  call stub_pool_put_array(solver, "solveStress", solveStress)
  call stub_pool_put_array(solver, "solveVelocity", solveVelocity)
  call stub_pool_put_array(solver, "uVelocity", uVelocity)
  call stub_pool_put_array(solver, "vVelocity", vVelocity)
  call stub_pool_put_array(solver, "icePressure", icePressure)
  call stub_pool_put_array(solver, "stressDivergenceU", stressDivergenceU)
  call stub_pool_put_array(solver, "stressDivergenceV", stressDivergenceV)
  call stub_pool_put_array(solver, "divergence", divergence)
  call stub_pool_put_array(solver, "shear", shear)

  call stub_pool_put_array(variational, "strain11", strain11)
  call stub_pool_put_array(variational, "strain22", strain22)
  call stub_pool_put_array(variational, "strain12", strain12)
  call stub_pool_put_array(variational, "stress11", stress11)
  call stub_pool_put_array(variational, "stress22", stress22)
  call stub_pool_put_array(variational, "stress12", stress12)
  call stub_pool_put_array(variational, "replacementPressure", replacementPressure)
  call stub_pool_put_array(variational, "cellVerticesAtVertex", cellVerticesAtVertex)
  call stub_pool_put_array(variational, "tanLatVertexRotatedOverRadius", tanLat)
  call stub_pool_put_array(variational, "basisGradientU", basisGradientU)
  call stub_pool_put_array(variational, "basisGradientV", basisGradientV)
  call stub_pool_put_array(variational, "basisIntegralsU", basisIntegralsU)
  call stub_pool_put_array(variational, "basisIntegralsV", basisIntegralsV)
  call stub_pool_put_array(variational, "basisIntegralsMetric", basisIntegralsMetric)

  call stub_pool_put_array(ridging, "ridgeConvergence", ridgeConvergence)
  call stub_pool_put_array(ridging, "ridgeShear", ridgeShear)

  call stub_pool_put_subpool(structs, "mesh", mesh)
  call stub_pool_put_subpool(structs, "velocity_solver", solver)
  call stub_pool_put_subpool(structs, "velocity_variational", variational)
  call stub_pool_put_subpool(structs, "ridging", ridging)

  allocate(block)
  block % structs => structs
  block % configs => configs
  block % next => null()
  domain % blocklist => block

  call seaice_init_evp(domain)
  do iter = 1, nIter
     call seaice_internal_stress_variational(domain)
  end do
  call seaice_final_divergence_shear_variational(block)

  open(newunit=u, file=trim(outPath), access='stream', form='unformatted', status='replace')
  write(u) numericalInertiaCoefficient
  write(u) strain11, strain22, strain12, stress11, stress22, stress12, replacementPressure, &
       stressDivergenceU, stressDivergenceV, divergence, shear, ridgeConvergence, ridgeShear
  close(u)

end program run_oracle
