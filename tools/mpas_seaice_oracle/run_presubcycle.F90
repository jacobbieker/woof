! Drive MPAS-Seaice v8.4.2's velocity-solver pre-subcycle routines on one
! block and write every output word: the reference the GPU port is held
! against.
!
!   run_presubcycle_r4 INPUT.bin OUTPUT.bin      (RKIND 4)
!   run_presubcycle_r8 INPUT.bin OUTPUT.bin      (RKIND 8)
!
! INPUT.bin is written by woof.verify.mpas_seaice_presubcycle_oracle (raw
! little-endian int32 words and RKIND reals, Fortran array order).  The driver
! puts the fields in the pools mpas_seaice_velocity_solver.F reads; build.sh
! compiles that file verbatim with -Dprivate=public, and seaice_mesh's two
! interpolation routines verbatim out of mpas_seaice_mesh.F.
!
! Without the column package's vertical thermodynamics it calls
! velocity_solver_pre_subcycle itself:
!
!   aggregate_mass_and_area, calculation_masks, new_ice_velocities,
!   ice_strength, air_stress, coriolis_force_coefficient, ocean_stress,
!   surface_tilt, init_subcycle_variables
!
! With it (config_use_column_package and
! config_use_column_vertical_thermodynamics both true) ice_strength calls the
! column package's colpkg_ice_strength, which this oracle does not build (the
! GPU port refuses that configuration by name).  The driver then calls the
! same routines in the same order without ice_strength.

program run_presubcycle
  use mpas_kind_types
  use mpas_derived_types
  use mpas_pool_routines
  use seaice_velocity_solver
  implicit none

  type(domain_type) :: domain
  type(block_type), pointer :: block
  type(MPAS_pool_type), pointer :: structs, configs, mesh, solver, variational, icestate, tracers, &
       aggregate, ocean, atmos, boundary

  integer, pointer :: nCells, nVertices, maxEdges, vertexDegree, nCategories, nVerticesSolve, nCellsSolve
  logical, pointer :: useColumn, useVThermo, useAirStress, useOceanStress, useTilt, geostrophicTilt, &
       useHaloExch
  character(len=StrKIND), pointer :: scheme
  integer :: columnFlag, vthermoFlag, airFlag, oceanFlag, tiltFlag, u

  integer, dimension(:), pointer :: nEdgesOnCell, interiorVertex, landIceMask, landIceMaskVertex, &
       solveStress, solveVelocity, solveVelocityPrevious
  integer, dimension(:,:), pointer :: cellsOnCell, cellsOnVertex, verticesOnCell
  real(kind=RKIND), dimension(:), pointer :: areaCell, areaTriangle, fVertex, uOcean, vOcean, &
       uAir, vAir, airDensity, tiltCellU, tiltCellV, openWaterArea, &
       iceAreaCell, iceVolumeCell, snowVolumeCell, totalMassCell, iceAreaCellInitial, icePressure, &
       airCellU, airCellV, iceAreaVertex, totalMassVertex, uOceanVertex, vOceanVertex, &
       uVelocity, vVelocity, uInitial, vInitial, stressDivergenceU, stressDivergenceV, &
       oceanStressU, oceanStressV, oceanStressCoeff, massF, airVertexU, airVertexV, &
       tiltVertexU, tiltVertexV, tiltForceU, tiltForceV
  real(kind=RKIND), dimension(:,:), pointer :: kiteAreasOnVertex, &
       strain11, strain22, strain12, stress11, stress22, stress12
  real(kind=RKIND), dimension(:,:,:), pointer :: iceAreaCategory, iceVolumeCategory, snowVolumeCategory
  character(len=1024) :: inPath, outPath

  call get_command_argument(1, inPath)
  call get_command_argument(2, outPath)

  allocate(nCells, nVertices, maxEdges, vertexDegree, nCategories, nVerticesSolve, nCellsSolve)
  allocate(useColumn, useVThermo, useAirStress, useOceanStress, useTilt, geostrophicTilt, useHaloExch, scheme)

  open(newunit=u, file=trim(inPath), access='stream', form='unformatted', status='old')
  read(u) nCells, nVertices, maxEdges, vertexDegree, nCategories, nVerticesSolve, nCellsSolve, &
       columnFlag, vthermoFlag, airFlag, oceanFlag, tiltFlag
  useColumn = columnFlag /= 0
  useVThermo = vthermoFlag /= 0
  useAirStress = airFlag /= 0
  useOceanStress = oceanFlag /= 0
  useTilt = tiltFlag /= 0
  geostrophicTilt = tiltFlag == 1
  useHaloExch = .false.
  scheme = 'variational'

  allocate(nEdgesOnCell(nCells), cellsOnCell(maxEdges,nCells), verticesOnCell(maxEdges,nCells), &
       cellsOnVertex(vertexDegree,nVertices), interiorVertex(nVertices), landIceMask(nCells), &
       landIceMaskVertex(nVertices), solveStress(nCells), solveVelocity(nVertices), &
       solveVelocityPrevious(nVertices))
  read(u) nEdgesOnCell, cellsOnCell, verticesOnCell, cellsOnVertex, interiorVertex, landIceMask, &
       landIceMaskVertex, solveStress, solveVelocity, solveVelocityPrevious

  allocate(areaCell(nCells), areaTriangle(nVertices), kiteAreasOnVertex(vertexDegree,nVertices), &
       fVertex(nVertices), iceAreaCategory(1,nCategories,nCells), iceVolumeCategory(1,nCategories,nCells), &
       snowVolumeCategory(1,nCategories,nCells), uOcean(nCells), vOcean(nCells), uAir(nCells), vAir(nCells), &
       airDensity(nCells), tiltCellU(nCells), tiltCellV(nCells), openWaterArea(nCells))
  read(u) areaCell, areaTriangle, kiteAreasOnVertex, fVertex, iceAreaCategory, iceVolumeCategory, &
       snowVolumeCategory, uOcean, vOcean, uAir, vAir, airDensity, tiltCellU, tiltCellV, openWaterArea

  ! the prior contents of every output: entries a routine does not reach keep
  ! their words
  allocate(iceAreaCell(nCells), iceVolumeCell(nCells), snowVolumeCell(nCells), totalMassCell(nCells), &
       iceAreaCellInitial(nCells), icePressure(nCells), airCellU(nCells), airCellV(nCells), &
       iceAreaVertex(nVertices), totalMassVertex(nVertices), uOceanVertex(nVertices), vOceanVertex(nVertices), &
       uVelocity(nVertices), vVelocity(nVertices), uInitial(nVertices), vInitial(nVertices), &
       stressDivergenceU(nVertices), stressDivergenceV(nVertices), oceanStressU(nVertices), &
       oceanStressV(nVertices), oceanStressCoeff(nVertices), massF(nVertices), airVertexU(nVertices), &
       airVertexV(nVertices), tiltVertexU(nVertices), tiltVertexV(nVertices), tiltForceU(nVertices), &
       tiltForceV(nVertices), strain11(maxEdges,nCells), strain22(maxEdges,nCells), strain12(maxEdges,nCells), &
       stress11(maxEdges,nCells), stress22(maxEdges,nCells), stress12(maxEdges,nCells))
  read(u) iceAreaCell, iceVolumeCell, snowVolumeCell, totalMassCell, iceAreaCellInitial, icePressure, &
       airCellU, airCellV, iceAreaVertex, totalMassVertex, uOceanVertex, vOceanVertex, uVelocity, vVelocity, &
       uInitial, vInitial, stressDivergenceU, stressDivergenceV, oceanStressU, oceanStressV, &
       oceanStressCoeff, massF, airVertexU, airVertexV, tiltVertexU, tiltVertexV, tiltForceU, tiltForceV, &
       strain11, strain22, strain12, stress11, stress22, stress12
  close(u)

  allocate(structs, configs, mesh, solver, variational, icestate, tracers, aggregate, ocean, atmos, boundary)
  call stub_pool_put_config(configs, "config_use_column_package", useColumn)
  call stub_pool_put_config(configs, "config_use_column_vertical_thermodynamics", useVThermo)
  call stub_pool_put_config(configs, "config_use_air_stress", useAirStress)
  call stub_pool_put_config(configs, "config_use_ocean_stress", useOceanStress)
  call stub_pool_put_config(configs, "config_use_surface_tilt", useTilt)
  call stub_pool_put_config(configs, "config_geostrophic_surface_tilt", geostrophicTilt)
  call stub_pool_put_config(configs, "config_use_halo_exch", useHaloExch)
  call stub_pool_put_config(configs, "config_stress_divergence_scheme", scheme)

  ! mesh doubles as block % dimensions
  call stub_pool_put_dimension(mesh, "nCells", nCells)
  call stub_pool_put_dimension(mesh, "nVertices", nVertices)
  call stub_pool_put_dimension(mesh, "maxEdges", maxEdges)
  call stub_pool_put_dimension(mesh, "vertexDegree", vertexDegree)
  call stub_pool_put_dimension(mesh, "nCategories", nCategories)
  call stub_pool_put_dimension(mesh, "nVerticesSolve", nVerticesSolve)
  call stub_pool_put_dimension(mesh, "nCellsSolve", nCellsSolve)
  call stub_pool_put_array(mesh, "nEdgesOnCell", nEdgesOnCell)
  call stub_pool_put_array(mesh, "cellsOnCell", cellsOnCell)
  call stub_pool_put_array(mesh, "verticesOnCell", verticesOnCell)
  call stub_pool_put_array(mesh, "cellsOnVertex", cellsOnVertex)
  call stub_pool_put_array(mesh, "areaCell", areaCell)
  call stub_pool_put_array(mesh, "areaTriangle", areaTriangle)
  call stub_pool_put_array(mesh, "kiteAreasOnVertex", kiteAreasOnVertex)
  call stub_pool_put_array(mesh, "fVertex", fVertex)

  call stub_pool_put_array(boundary, "interiorVertex", interiorVertex)

  call stub_pool_put_array(tracers, "iceAreaCategory", iceAreaCategory)
  call stub_pool_put_array(tracers, "iceVolumeCategory", iceVolumeCategory)
  call stub_pool_put_array(tracers, "snowVolumeCategory", snowVolumeCategory)

  call stub_pool_put_array(aggregate, "iceAreaCell", iceAreaCell)
  call stub_pool_put_array(aggregate, "iceVolumeCell", iceVolumeCell)
  call stub_pool_put_array(aggregate, "snowVolumeCell", snowVolumeCell)

  call stub_pool_put_array(icestate, "totalMassCell", totalMassCell)
  call stub_pool_put_array(icestate, "iceAreaCellInitial", iceAreaCellInitial)
  call stub_pool_put_array(icestate, "iceAreaVertex", iceAreaVertex)
  call stub_pool_put_array(icestate, "totalMassVertex", totalMassVertex)
  call stub_pool_put_array(icestate, "openWaterArea", openWaterArea)

  call stub_pool_put_array(ocean, "landIceMask", landIceMask)
  call stub_pool_put_array(ocean, "landIceMaskVertex", landIceMaskVertex)
  call stub_pool_put_array(ocean, "uOceanVelocity", uOcean)
  call stub_pool_put_array(ocean, "vOceanVelocity", vOcean)
  call stub_pool_put_array(ocean, "seaSurfaceTiltU", tiltCellU)
  call stub_pool_put_array(ocean, "seaSurfaceTiltV", tiltCellV)

  call stub_pool_put_array(atmos, "uAirVelocity", uAir)
  call stub_pool_put_array(atmos, "vAirVelocity", vAir)
  call stub_pool_put_array(atmos, "airDensity", airDensity)

  call stub_pool_put_array(solver, "solveStress", solveStress)
  call stub_pool_put_array(solver, "solveVelocity", solveVelocity)
  call stub_pool_put_array(solver, "solveVelocityPrevious", solveVelocityPrevious)
  call stub_pool_put_array(solver, "uVelocity", uVelocity)
  call stub_pool_put_array(solver, "vVelocity", vVelocity)
  call stub_pool_put_array(solver, "uVelocityInitial", uInitial)
  call stub_pool_put_array(solver, "vVelocityInitial", vInitial)
  call stub_pool_put_array(solver, "icePressure", icePressure)
  call stub_pool_put_array(solver, "totalMassVertexfVertex", massF)
  call stub_pool_put_array(solver, "oceanStressU", oceanStressU)
  call stub_pool_put_array(solver, "oceanStressV", oceanStressV)
  call stub_pool_put_array(solver, "oceanStressCoeff", oceanStressCoeff)
  call stub_pool_put_array(solver, "uOceanVelocityVertex", uOceanVertex)
  call stub_pool_put_array(solver, "vOceanVelocityVertex", vOceanVertex)
  call stub_pool_put_array(solver, "airStressCellU", airCellU)
  call stub_pool_put_array(solver, "airStressCellV", airCellV)
  call stub_pool_put_array(solver, "airStressVertexU", airVertexU)
  call stub_pool_put_array(solver, "airStressVertexV", airVertexV)
  call stub_pool_put_array(solver, "seaSurfaceTiltVertexU", tiltVertexU)
  call stub_pool_put_array(solver, "seaSurfaceTiltVertexV", tiltVertexV)
  call stub_pool_put_array(solver, "surfaceTiltForceU", tiltForceU)
  call stub_pool_put_array(solver, "surfaceTiltForceV", tiltForceV)
  call stub_pool_put_array(solver, "stressDivergenceU", stressDivergenceU)
  call stub_pool_put_array(solver, "stressDivergenceV", stressDivergenceV)

  call stub_pool_put_array(variational, "strain11", strain11)
  call stub_pool_put_array(variational, "strain22", strain22)
  call stub_pool_put_array(variational, "strain12", strain12)
  call stub_pool_put_array(variational, "stress11", stress11)
  call stub_pool_put_array(variational, "stress22", stress22)
  call stub_pool_put_array(variational, "stress12", stress12)

  call stub_pool_put_subpool(structs, "mesh", mesh)
  call stub_pool_put_subpool(structs, "boundary", boundary)
  call stub_pool_put_subpool(structs, "tracers", tracers)
  call stub_pool_put_subpool(structs, "tracers_aggregate", aggregate)
  call stub_pool_put_subpool(structs, "icestate", icestate)
  call stub_pool_put_subpool(structs, "ocean_coupling", ocean)
  call stub_pool_put_subpool(structs, "atmos_coupling", atmos)
  call stub_pool_put_subpool(structs, "velocity_solver", solver)
  call stub_pool_put_subpool(structs, "velocity_variational", variational)

  allocate(block)
  block % structs => structs
  block % configs => configs
  block % dimensions => mesh
  block % next => null()
  domain % blocklist => block
  domain % configs => configs

  if (.not. (useColumn .and. useVThermo)) then
     call velocity_solver_pre_subcycle(domain)
  else
     call aggregate_mass_and_area(domain)
     call calculation_masks(domain)
     call new_ice_velocities(domain)
     call air_stress(domain)
     call coriolis_force_coefficient(domain)
     call ocean_stress(domain)
     call surface_tilt(domain)
     call init_subcycle_variables(domain)
  end if

  open(newunit=u, file=trim(outPath), access='stream', form='unformatted', status='replace')
  write(u) solveStress, solveVelocity, solveVelocityPrevious
  write(u) iceAreaCell, iceVolumeCell, snowVolumeCell, totalMassCell, iceAreaCellInitial, icePressure, &
       airCellU, airCellV, iceAreaVertex, totalMassVertex, uOceanVertex, vOceanVertex, uVelocity, vVelocity, &
       uInitial, vInitial, stressDivergenceU, stressDivergenceV, oceanStressU, oceanStressV, &
       oceanStressCoeff, massF, airVertexU, airVertexV, tiltVertexU, tiltVertexV, tiltForceU, tiltForceV, &
       strain11, strain22, strain12, stress11, stress22, stress12
  close(u)

end program run_presubcycle
