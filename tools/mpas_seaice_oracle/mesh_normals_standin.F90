  ! seaice_normal_vectors_polygon, a stand-in (the time-step oracle's mesh
  ! setup is the input's): it leaves the normal vectors the driver was given
  subroutine seaice_normal_vectors_polygon(mesh, normalVectorPolygon, rotateCartesianGrid, removeMetricTerms, &
                                           latCellRotated)
    type(MPAS_pool_type), pointer, intent(in) :: mesh
    real(kind=RKIND), dimension(:,:,:), intent(inout) :: normalVectorPolygon
    logical, intent(in) :: rotateCartesianGrid, removeMetricTerms
    real(kind=RKIND), dimension(:), optional, intent(out) :: latCellRotated
    if (present(latCellRotated)) latCellRotated = 0.0_RKIND
  end subroutine seaice_normal_vectors_polygon
