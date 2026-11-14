! Drive MPAS-Seaice v8.4.2's incremental remap (mpas_seaice_advection_incremental_remap.F
! and its tracers module, both compiled verbatim) on one block and write every
! field word back: the reference the GPU remap is held against.
!
!   run_incremental_remap INPUT.bin OUTPUT.bin LOG.txt
!
! INPUT.bin is written by woof.verify.mpas_seaice_remap_oracle in
! tools/mpas_ocean_oracle/oracle_stream.F90's layout, on the framework
! stand-ins of tools/mpas_ocean_oracle/stubs.F90.  The remap module is built
! with -Dprivate=public, which changes only which of its routines this driver
! may call.  Every run starts with seaice_init_advection_incremental_remap
! (verbatim: the tracer linked list from the packages, the halo check, the
! remap geometry; the RBF reconstruction coefficients, coeffs_reconstruct, are
! the input's, see remap_standins.F90).  Then, by oracle_stage:
!
!   0  nothing more (the tracer tree and the geometry);
!   1  seaice_set_tracer_array_pointers and the volume-to-thickness loop of
!      seaice_run_advection_incremental_remap (its lines 2510-2552), then,
!      under oracle_round_trip, its thickness-to-volume loop (2619-2661);
!   2  the volume-to-thickness loop, then find_departure_points,
!      find_departure_triangles and get_triangle_quadrature_points with the
!      arguments incremental_remap_block passes them (3082-3158);
!   3  the volume-to-thickness loop, then make_masks (threshold eps11) and
!      construct_linear_tracer_fields as incremental_remap_block calls them
!      (3035-3069);
!   4  seaice_run_advection_incremental_remap itself, oracle_n_steps times.
!
! LOG.txt gets the ordered tracer list ('TRACER name parent nParents
! hasChild'), each warning the routines log (the mean-recovery check), and the
! error code ('IERR n') of stage 4.
program run_incremental_remap
  use mpas_kind_types
  use mpas_derived_types
  use mpas_pool_routines
  use mpas_log, only: stub_log_warn_unit
  use mpas_timekeeping
  use seaice_advection_incremental_remap_tracers, only: tracer_type, tracersHead, &
       seaice_add_tracers_to_linked_list, seaice_set_tracer_array_pointers
  use seaice_advection_incremental_remap
  use oracle_stream
  implicit none
  type(oracle_pools) :: pools
  type(oracle_field), allocatable :: fields(:)
  type(domain_type), pointer :: domain
  type(block_type), pointer :: block
  type(tracer_type), pointer :: thisTracer
  type(MPAS_pool_type), pointer :: meshPool, velocityPool, irPool
  type(geometric_avg_cell_type), pointer :: geomAvgCell
  real(kind=RKIND) :: inputDt
  integer, pointer :: stage, nSteps, nCells, nVertices, nEdges, maxEdges, vertexDegree, nTriPerEdgeRemap
  logical, pointer :: roundTrip, on_a_sphere, config_rotate_cartesian_grid
  real(kind=RKIND), pointer :: dynamicsTimeStep
  real(kind=RKIND), dimension(:), pointer :: uVelocity, vVelocity, minLengthEdgesOnVertex, dcEdge
  real(kind=RKIND), dimension(:,:), pointer :: workCategoryCell, departurePoint, triangleArea, &
       xVertexOnCell, yVertexOnCell, xVertexOnEdge, yVertexOnEdge
  real(kind=RKIND), dimension(:,:,:), pointer :: xTriangle, yTriangle, transGlobalToCell, coeffsReconstruct
  integer, dimension(:), pointer :: indexToVertexID, indexToCellID, indexToEdgeID, nEdgesOnCell, remapEdge, &
       maskEdge, maskCell
  integer, dimension(:,:), pointer :: verticesOnCell, verticesOnEdge, edgesOnCell, cellsOnEdge, cellsOnCell, &
       cellsOnEdgeRemap, edgesOnEdgeRemap, iCellTriangle
  integer :: p, ierr, iStep, logUnit
  character(len=24), parameter :: structNames(13) = [character(len=24) :: 'mesh', 'velocity_solver', 'tracers', &
       'incremental_remap', 'rotated_mesh', 'tracer_masks', 'tracer_edge_fluxes', 'tracer_products', &
       'tracer_triangles', 'tracer_barycenter', 'tracer_reconstruction', 'tracer_conservation', &
       'tracer_monotonicity']
  character(len=1024) :: inPath, outPath, logPath

  call get_command_argument(1, inPath)
  call get_command_argument(2, outPath)
  call get_command_argument(3, logPath)
  call read_stream(inPath, inputDt, pools, fields)
  open(newunit=logUnit, file=trim(logPath), status='replace', action='write')
  stub_log_warn_unit = logUnit
  allocate(domain)
  allocate(domain % blocklist)
  allocate(domain % blocklist % structs)
  allocate(domain % clock)
  allocate(domain % dminfo)
  block => domain % blocklist
  domain % configs => pools % configs
  block % configs => pools % configs
  block % packages => pools % packages
  block % dimensions => pool_named(pools, 'mesh')
  do p = 1, size(structNames)
     call stub_pool_put_subpool(block % structs, trim(structNames(p)), pool_named(pools, trim(structNames(p))))
  end do
  meshPool => pool_named(pools, 'mesh')
  velocityPool => pool_named(pools, 'velocity_solver')
  irPool => pool_named(pools, 'incremental_remap')

  call mpas_pool_get_config(pools % configs, 'oracle_stage', stage)
  if (stage == 0) then
     call seaice_add_tracers_to_linked_list(domain)
  else
     call seaice_init_advection_incremental_remap(domain)
  end if
  thisTracer => tracersHead
  do while (associated(thisTracer))
     write(logUnit, '(a,1x,a,1x,a,1x,i0,1x,l1)') 'TRACER', trim(thisTracer % tracerName), &
          trim(thisTracer % parentName), thisTracer % nParents, thisTracer % hasChild
     thisTracer => thisTracer % next
  end do

  call mpas_pool_get_array(velocityPool, 'dynamicsTimeStep', dynamicsTimeStep)
  select case (stage)
  case (0)
  case (1)
     call volume_to_thickness_loop()
     call mpas_pool_get_config(pools % configs, 'oracle_round_trip', roundTrip)
     if (roundTrip) call thickness_to_volume_loop()
  case (2)
     call volume_to_thickness_loop()
     call get_block_arrays()
     ! incremental_remap_block:3083-3157
     call find_departure_points(&
          domain,                              &
          nVertices,                           &
          dynamicsTimeStep,                    &
          uVelocity,    vVelocity,             &
          minLengthEdgesOnVertex,              &
          indexToVertexID,                     &
          departurePoint,                      &
          on_a_sphere)
     call find_departure_triangles(&
          nEdges,                                    &
          nCells,                                    &
          maxEdges,                                  &
          nTriPerEdgeRemap,                          &
          nEdgesOnCell,                              &
          xVertexOnEdge,    yVertexOnEdge,           &
          xVertexOnCell,    yVertexOnCell,           &
          verticesOnCell,                            &
          verticesOnEdge,                            &
          edgesOnCell,                               &
          cellsOnEdge,                               &
          remapEdge,                                 &
          cellsOnEdgeRemap,                          &
          edgesOnEdgeRemap,                          &
          vertexDegree,                              &
          departurePoint,                            &
          maskEdge,                                  &
          xTriangle,        yTriangle,               &
          iCellTriangle,                             &
          triangleArea,                              &
          on_a_sphere,                               &
          transGlobalToCell,                         &
          indexToCellID,                             &
          indexToEdgeID,                             &
          indexToVertexID,                           &
          block)
     call get_triangle_quadrature_points (&
          nEdges,             &
          nTriPerEdgeRemap,   &
          maskEdge,          &
          xTriangle, yTriangle)
  case (3)
     call volume_to_thickness_loop()
     call get_block_arrays()
     ! incremental_remap_block:3036-3068
     call make_masks(&
          nCells,      &
          maskCell,    &
          tracersHead, &
          threshold_in = eps11)
     call construct_linear_tracer_fields(&
          nCells,                        &
          maxEdges,                      &
          nEdgesOnCell,                  &
          edgesOnCell,                   &
          cellsOnCell,                   &
          cellsOnEdge,                   &
          xVertexOnCell, yVertexOnCell,  &
          dcEdge,                        &
          on_a_sphere,                   &
          config_rotate_cartesian_grid,  &
          transGlobalToCell,             &
          coeffsReconstruct,             &
          maskCell,                      &
          geomAvgCell,                   &
          tracersHead,                   &
          indexToCellID,                 &
          indexToEdgeID,                 &
          block)
  case (4)
     call mpas_pool_get_config(pools % configs, 'oracle_n_steps', nSteps)
     ierr = 0
     do iStep = 1, nSteps
        call seaice_run_advection_incremental_remap(domain, domain % clock, ierr)
        if (ierr > 0) exit
     end do
     write(logUnit, '(a,1x,i0)') 'IERR', ierr
  case default
     stop 13
  end select
  close(logUnit)
  call write_stream(outPath, fields)

contains

  ! seaice_run_advection_incremental_remap:2510-2552, for this one block
  subroutine volume_to_thickness_loop()
    call mpas_pool_get_array(irPool, 'workCategoryCell', workCategoryCell)
    call seaice_set_tracer_array_pointers(tracersHead, block, 1)
    thisTracer => tracersHead
    do while (associated(thisTracer))
       if (trim(thisTracer % tracerName) == 'iceAreaCategory') then
          workCategoryCell(:,:) = thisTracer % array2D(:,:)
       elseif (trim(thisTracer % tracerName) == 'iceVolumeCategory' .or. &
               trim(thisTracer % tracerName) == 'snowVolumeCategory') then
          call volume_to_thickness(&
               workCategoryCell(:,:),  &
               thisTracer % array2D(:,:))
       endif
       thisTracer => thisTracer % next
    enddo
  end subroutine volume_to_thickness_loop

  ! seaice_run_advection_incremental_remap:2619-2661, for this one block
  subroutine thickness_to_volume_loop()
    call mpas_pool_get_array(irPool, 'workCategoryCell', workCategoryCell)
    call seaice_set_tracer_array_pointers(tracersHead, block, 1)
    thisTracer => tracersHead
    do while (associated(thisTracer))
       if (trim(thisTracer % tracerName) == 'iceAreaCategory') then
          workCategoryCell(:,:) = thisTracer % array2D(:,:)
       elseif (trim(thisTracer % tracerName) == 'iceVolumeCategory' .or. &
               trim(thisTracer % tracerName) == 'snowVolumeCategory') then
          call thickness_to_volume(&
               workCategoryCell(:,:),  &
               thisTracer % array2D(:,:))
       endif
       thisTracer => thisTracer % next
    enddo
  end subroutine thickness_to_volume_loop

  ! the arrays incremental_remap_block gets from the pools (2905-2971)
  subroutine get_block_arrays()
    call mpas_pool_get_config(block % configs, 'config_rotate_cartesian_grid', config_rotate_cartesian_grid)
    call mpas_pool_get_config(meshPool, 'on_a_sphere', on_a_sphere)
    call mpas_pool_get_array(velocityPool, 'uVelocity', uVelocity)
    call mpas_pool_get_array(velocityPool, 'vVelocity', vVelocity)
    call mpas_pool_get_dimension(meshPool, 'nCells', nCells)
    call mpas_pool_get_dimension(meshPool, 'nVertices', nVertices)
    call mpas_pool_get_dimension(meshPool, 'nEdges', nEdges)
    call mpas_pool_get_dimension(meshPool, 'maxEdges', maxEdges)
    call mpas_pool_get_dimension(meshPool, 'vertexDegree', vertexDegree)
    call mpas_pool_get_dimension(meshPool, 'nTriPerEdgeRemap', nTriPerEdgeRemap)
    call mpas_pool_get_array(meshPool, 'nEdgesOnCell', nEdgesOnCell)
    call mpas_pool_get_array(meshPool, 'edgesOnCell', edgesOnCell)
    call mpas_pool_get_array(meshPool, 'cellsOnEdge', cellsOnEdge)
    call mpas_pool_get_array(meshPool, 'cellsOnCell', cellsOnCell)
    call mpas_pool_get_array(meshPool, 'verticesOnCell', verticesOnCell)
    call mpas_pool_get_array(meshPool, 'verticesOnEdge', verticesOnEdge)
    call mpas_pool_get_array(meshPool, 'dcEdge', dcEdge)
    call mpas_pool_get_array(meshPool, 'coeffs_reconstruct', coeffsReconstruct)
    call mpas_pool_get_array(meshPool, 'indexToCellID', indexToCellID)
    call mpas_pool_get_array(meshPool, 'indexToVertexID', indexToVertexID)
    call mpas_pool_get_array(meshPool, 'indexToEdgeID', indexToEdgeID)
    allocate(geomAvgCell)
    call get_incremental_remap_geometry_pointers(geomAvgCell, block)
    call mpas_pool_get_array(irPool, 'xVertexOnCell', xVertexOnCell)
    call mpas_pool_get_array(irPool, 'yVertexOnCell', yVertexOnCell)
    call mpas_pool_get_array(irPool, 'xVertexOnEdge', xVertexOnEdge)
    call mpas_pool_get_array(irPool, 'yVertexOnEdge', yVertexOnEdge)
    call mpas_pool_get_array(irPool, 'departurePoint', departurePoint)
    call mpas_pool_get_array(irPool, 'remapEdge', remapEdge)
    call mpas_pool_get_array(irPool, 'cellsOnEdgeRemap', cellsOnEdgeRemap)
    call mpas_pool_get_array(irPool, 'edgesOnEdgeRemap', edgesOnEdgeRemap)
    call mpas_pool_get_array(irPool, 'xTriangle', xTriangle)
    call mpas_pool_get_array(irPool, 'yTriangle', yTriangle)
    call mpas_pool_get_array(irPool, 'iCellTriangle', iCellTriangle)
    call mpas_pool_get_array(irPool, 'triangleArea', triangleArea)
    call mpas_pool_get_array(irPool, 'transGlobalToCell', transGlobalToCell)
    call mpas_pool_get_array(irPool, 'minLengthEdgesOnVertex', minLengthEdgesOnVertex)
    call mpas_pool_get_array(irPool, 'maskEdge', maskEdge)
    call mpas_pool_get_array(irPool, 'maskCell', maskCell)
  end subroutine get_block_arrays

end program run_incremental_remap
