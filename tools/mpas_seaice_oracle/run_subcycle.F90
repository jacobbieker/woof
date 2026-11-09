! Drive MPAS-Seaice v8.4.2's EVP velocity solve on one block and write every
! output word: the reference the GPU port is held against.
!
!   run_subcycle_r4 INPUT.bin OUTPUT.bin      (RKIND 4)
!   run_subcycle_r8 INPUT.bin OUTPUT.bin      (RKIND 8)
!
! INPUT.bin is written by woof.verify.mpas_seaice_velocity_oracle (raw
! little-endian int32 words and RKIND reals, Fortran array order).  The driver
! puts the fields in the pools mpas_seaice_velocity_solver.F reads.  build.sh
! compiles that file verbatim with -Dprivate=public, so its private routines
! can be called here.  It then calls them in the model's order
! (seaice_run_velocity_solver):
!
!   full = 1, the whole elastic subcycle:
!     seaice_init_evp                      (seaice_init_velocity_solver)
!     coriolis_force_coefficient           (velocity_solver_pre_subcycle)
!     ocean_stress
!     init_subcycle_variables
!     subcycle_velocity_solver             config_elastic_subcycle_number times:
!                                          internal stress (variational),
!                                          ocean_stress_coefficient,
!                                          solve_velocity or solve_velocity_revised
!     velocity_solver_post_subcycle        final_divergence_shear,
!                                          principal_stresses_driver,
!                                          ocean_stress_final
!
!   full = 0, the velocity solve without the internal stress:
!     the same pre-subcycle calls, then nSub times: stressDivergenceU/V set
!     to the case's forcing (standing in for the internal stress, which the
!     full mode covers), ocean_stress_coefficient, solve_velocity or
!     solve_velocity_revised; then principal_stresses_driver and
!     ocean_stress_final.
!     numericalInertiaCoefficient (seaice_init_evp's, read by the revised
!     solve) is the case's word.
!
! The pre-subcycle routines this oracle does not call (aggregate_mass_and_area,
! calculation_masks, new_ice_velocities, ice_strength, air_stress,
! surface_tilt) produce fields the case supplies instead: totalMassVertex,
! iceAreaVertex, solveStress/solveVelocity, icePressure, airStressVertexU/V,
! surfaceTiltForceU/V (run_presubcycle.F90 drives them).  ocean_stress_final
! interpolates with seaice_mesh's seaice_interpolate_vertex_to_cell, which
! build.sh compiles verbatim out of mpas_seaice_mesh.F.

program run_subcycle
  use mpas_kind_types
  use mpas_derived_types
  use mpas_pool_routines
  use seaice_velocity_solver_constitutive_relation
  use seaice_velocity_solver
  implicit none

  type(domain_type) :: domain
  type(block_type), pointer :: block
  type(MPAS_clock_type) :: clock
  type(MPAS_pool_type), pointer :: structs, configs, mesh, solver, variational, ridging, icestate, boundary

  integer, pointer :: nCells, nVertices, nEdges, maxEdges, vertexDegree, nEdgesSolve, &
       nVerticesSolve, nCellsSolve, nSub
  logical, pointer :: revisedEVP, useColumn, useOceanStress, useHaloExch
  character(len=StrKIND), pointer :: scheme
  integer :: revisedFlag, columnFlag, oceanFlag, full, iter, u
  real(kind=RKIND) :: inertia
  real(kind=RKIND), pointer :: dynamicsTimeStep, elasticTimeStep

  integer, dimension(:), pointer :: nEdgesOnCell, solveStress, solveVelocity, interiorVertex
  integer, dimension(:,:), pointer :: verticesOnCell, cellsOnVertex, cellVerticesAtVertex
  real(kind=RKIND), dimension(:), pointer :: dvEdge, areaCell, areaTriangle, tanLat, fVertex, &
       icePressure, totalMassVertex, iceAreaVertex, uOcean, vOcean, uInitial, vInitial, &
       airU, airV, tiltU, tiltV, forcingU, forcingV, &
       uVelocity, vVelocity, massF, oceanStressU, oceanStressV, oceanStressCoeff, &
       stressDivergenceU, stressDivergenceV, divergence, shear, ridgeConvergence, ridgeShear, &
       oceanStressCellU, oceanStressCellV
  real(kind=RKIND), dimension(:,:), pointer :: strain11, strain22, strain12, &
       stress11, stress22, stress12, replacementPressure, principalStress1, principalStress2
  real(kind=RKIND), dimension(:,:,:), pointer :: basisGradientU, basisGradientV, &
       basisIntegralsU, basisIntegralsV, basisIntegralsMetric
  character(len=1024) :: inPath, outPath

  call get_command_argument(1, inPath)
  call get_command_argument(2, outPath)

  allocate(nCells, nVertices, nEdges, maxEdges, vertexDegree, nEdgesSolve, nVerticesSolve, nCellsSolve, nSub)
  allocate(revisedEVP, useColumn, useOceanStress, useHaloExch, scheme, dynamicsTimeStep, elasticTimeStep)

  open(newunit=u, file=trim(inPath), access='stream', form='unformatted', status='old')
  read(u) nCells, nVertices, nEdges, maxEdges, vertexDegree, nVerticesSolve, nCellsSolve, &
       revisedFlag, columnFlag, oceanFlag, full, nSub
  read(u) dynamicsTimeStep, elasticTimeStep, inertia
  revisedEVP = revisedFlag /= 0
  useColumn = columnFlag /= 0
  useOceanStress = oceanFlag /= 0
  useHaloExch = .false.
  scheme = 'variational'
  nEdgesSolve = nEdges

  allocate(nEdgesOnCell(nCells), verticesOnCell(maxEdges,nCells), &
       cellsOnVertex(vertexDegree,nVertices), cellVerticesAtVertex(vertexDegree,nVertices), &
       solveStress(nCells), solveVelocity(nVertices), interiorVertex(nVertices))
  read(u) nEdgesOnCell, verticesOnCell, cellsOnVertex, cellVerticesAtVertex, solveStress, solveVelocity, &
       interiorVertex

  allocate(dvEdge(nEdges), areaCell(nCells), areaTriangle(nVertices), tanLat(nVertices), &
       fVertex(nVertices), icePressure(nCells), totalMassVertex(nVertices), iceAreaVertex(nVertices), &
       uOcean(nVertices), vOcean(nVertices), uInitial(nVertices), vInitial(nVertices), &
       airU(nVertices), airV(nVertices), tiltU(nVertices), tiltV(nVertices), &
       forcingU(nVertices), forcingV(nVertices))
  read(u) dvEdge, areaCell, areaTriangle, tanLat, fVertex, icePressure, totalMassVertex, &
       iceAreaVertex, uOcean, vOcean, uInitial, vInitial, airU, airV, tiltU, tiltV, forcingU, forcingV

  allocate(basisGradientU(maxEdges,maxEdges,nCells), basisGradientV(maxEdges,maxEdges,nCells), &
       basisIntegralsU(maxEdges,maxEdges,nCells), basisIntegralsV(maxEdges,maxEdges,nCells), &
       basisIntegralsMetric(maxEdges,maxEdges,nCells))
  read(u) basisGradientU, basisGradientV, basisIntegralsU, basisIntegralsV, basisIntegralsMetric

  ! the prior contents of every output: entries a routine does not reach keep
  ! their words
  allocate(uVelocity(nVertices), vVelocity(nVertices), massF(nVertices), oceanStressU(nVertices), &
       oceanStressV(nVertices), oceanStressCoeff(nVertices), &
       strain11(maxEdges,nCells), strain22(maxEdges,nCells), strain12(maxEdges,nCells), &
       stress11(maxEdges,nCells), stress22(maxEdges,nCells), stress12(maxEdges,nCells), &
       replacementPressure(maxEdges,nCells), stressDivergenceU(nVertices), &
       stressDivergenceV(nVertices), divergence(nCells), shear(nCells), &
       ridgeConvergence(nCells), ridgeShear(nCells), &
       principalStress1(maxEdges,nCells), principalStress2(maxEdges,nCells), &
       oceanStressCellU(nCells), oceanStressCellV(nCells))
  read(u) uVelocity, vVelocity, massF, oceanStressU, oceanStressV, oceanStressCoeff, &
       strain11, strain22, strain12, stress11, stress22, stress12, replacementPressure, &
       stressDivergenceU, stressDivergenceV, divergence, shear, ridgeConvergence, ridgeShear, &
       principalStress1, principalStress2, oceanStressCellU, oceanStressCellV
  close(u)

  allocate(structs, configs, mesh, solver, variational, ridging, icestate, boundary)
  call stub_pool_put_config(configs, "config_revised_evp", revisedEVP)
  call stub_pool_put_config(configs, "config_use_column_package", useColumn)
  call stub_pool_put_config(configs, "config_use_ocean_stress", useOceanStress)
  call stub_pool_put_config(configs, "config_use_halo_exch", useHaloExch)
  call stub_pool_put_config(configs, "config_stress_divergence_scheme", scheme)
  call stub_pool_put_config(configs, "config_elastic_subcycle_number", nSub)

  ! mesh doubles as block % dimensions
  call stub_pool_put_dimension(mesh, "nCells", nCells)
  call stub_pool_put_dimension(mesh, "nVertices", nVertices)
  call stub_pool_put_dimension(mesh, "maxEdges", maxEdges)
  call stub_pool_put_dimension(mesh, "vertexDegree", vertexDegree)
  call stub_pool_put_dimension(mesh, "nEdgesSolve", nEdgesSolve)
  call stub_pool_put_dimension(mesh, "nVerticesSolve", nVerticesSolve)
  call stub_pool_put_dimension(mesh, "nCellsSolve", nCellsSolve)
  call stub_pool_put_array(mesh, "nEdgesOnCell", nEdgesOnCell)
  call stub_pool_put_array(mesh, "verticesOnCell", verticesOnCell)
  call stub_pool_put_array(mesh, "cellsOnVertex", cellsOnVertex)
  call stub_pool_put_array(mesh, "dvEdge", dvEdge)
  call stub_pool_put_array(mesh, "areaCell", areaCell)
  call stub_pool_put_array(mesh, "areaTriangle", areaTriangle)
  call stub_pool_put_array(mesh, "fVertex", fVertex)

  call stub_pool_put_array(boundary, "interiorVertex", interiorVertex)

  call stub_pool_put_array(icestate, "totalMassVertex", totalMassVertex)
  call stub_pool_put_array(icestate, "iceAreaVertex", iceAreaVertex)

  call stub_pool_put_array(solver, "dynamicsTimeStep", dynamicsTimeStep)
  call stub_pool_put_array(solver, "elasticTimeStep", elasticTimeStep)
  call stub_pool_put_array(solver, "solveStress", solveStress)
  call stub_pool_put_array(solver, "solveVelocity", solveVelocity)
  call stub_pool_put_array(solver, "uVelocity", uVelocity)
  call stub_pool_put_array(solver, "vVelocity", vVelocity)
  call stub_pool_put_array(solver, "uVelocityInitial", uInitial)
  call stub_pool_put_array(solver, "vVelocityInitial", vInitial)
  call stub_pool_put_array(solver, "icePressure", icePressure)
  call stub_pool_put_array(solver, "totalMassVertexfVertex", massF)
  call stub_pool_put_array(solver, "oceanStressU", oceanStressU)
  call stub_pool_put_array(solver, "oceanStressV", oceanStressV)
  call stub_pool_put_array(solver, "oceanStressCoeff", oceanStressCoeff)
  call stub_pool_put_array(solver, "uOceanVelocityVertex", uOcean)
  call stub_pool_put_array(solver, "vOceanVelocityVertex", vOcean)
  call stub_pool_put_array(solver, "airStressVertexU", airU)
  call stub_pool_put_array(solver, "airStressVertexV", airV)
  call stub_pool_put_array(solver, "surfaceTiltForceU", tiltU)
  call stub_pool_put_array(solver, "surfaceTiltForceV", tiltV)
  call stub_pool_put_array(solver, "stressDivergenceU", stressDivergenceU)
  call stub_pool_put_array(solver, "stressDivergenceV", stressDivergenceV)
  call stub_pool_put_array(solver, "oceanStressCellU", oceanStressCellU)
  call stub_pool_put_array(solver, "oceanStressCellV", oceanStressCellV)
  call stub_pool_put_array(solver, "divergence", divergence)
  call stub_pool_put_array(solver, "shear", shear)

  call stub_pool_put_array(variational, "strain11", strain11)
  call stub_pool_put_array(variational, "strain22", strain22)
  call stub_pool_put_array(variational, "strain12", strain12)
  call stub_pool_put_array(variational, "stress11", stress11)
  call stub_pool_put_array(variational, "stress22", stress22)
  call stub_pool_put_array(variational, "stress12", stress12)
  call stub_pool_put_array(variational, "replacementPressure", replacementPressure)
  call stub_pool_put_array(variational, "principalStress1", principalStress1)
  call stub_pool_put_array(variational, "principalStress2", principalStress2)
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
  call stub_pool_put_subpool(structs, "boundary", boundary)
  call stub_pool_put_subpool(structs, "icestate", icestate)
  call stub_pool_put_subpool(structs, "velocity_solver", solver)
  call stub_pool_put_subpool(structs, "velocity_variational", variational)
  call stub_pool_put_subpool(structs, "ridging", ridging)

  allocate(block)
  block % structs => structs
  block % configs => configs
  block % dimensions => mesh
  block % next => null()
  domain % blocklist => block
  domain % configs => configs

  if (full /= 0) then
     call seaice_init_evp(domain)
  else
     numericalInertiaCoefficient = inertia
  end if
  call coriolis_force_coefficient(domain)
  call ocean_stress(domain)
  call init_subcycle_variables(domain)
  if (full /= 0) then
     call subcycle_velocity_solver(domain, clock)
     call velocity_solver_post_subcycle(domain)
  else
     do iter = 1, nSub
        stressDivergenceU = forcingU
        stressDivergenceV = forcingV
        call ocean_stress_coefficient(domain)
        if (revisedEVP) then
           call solve_velocity_revised(domain)
        else
           call solve_velocity(domain)
        end if
     end do
     call principal_stresses_driver(domain)
     call ocean_stress_final(domain)
  end if

  open(newunit=u, file=trim(outPath), access='stream', form='unformatted', status='replace')
  write(u) numericalInertiaCoefficient
  write(u) uVelocity, vVelocity, massF, oceanStressU, oceanStressV, oceanStressCoeff, &
       strain11, strain22, strain12, stress11, stress22, stress12, replacementPressure, &
       stressDivergenceU, stressDivergenceV, divergence, shear, ridgeConvergence, ridgeShear, &
       principalStress1, principalStress2, oceanStressCellU, oceanStressCellV
  close(u)

end program run_subcycle
