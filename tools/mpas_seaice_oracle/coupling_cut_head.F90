! The head of the module the coupling-preparation oracle cuts out of
! mpas_seaice_column.F (build.sh): seaice_column_coupling_prep and
! seaice_column_scale_fluxes follow it verbatim.  The rest of that file (the
! column package's physics) is not compiled here; what the two routines name
! from it is stood in for below.  ciceTracerObject is read only under
! config_use_column_biogeochemistry, and seaice_column_ocean_mixed_layer runs
! only under config_use_ocean_mixed_layer: the oracle's runs have both off,
! and the stand-in stops by name if called.
module seaice_column_coupling
  use mpas_kind_types
  use mpas_derived_types
  use mpas_pool_routines
  implicit none
  private
  public :: seaice_column_coupling_prep, seaice_column_scale_fluxes

  type :: cice_tracer_object_stand_in
     integer :: nBioTracers = 0
     integer, dimension(:), allocatable :: index_LayerIndexToDataArray
  end type cice_tracer_object_stand_in
  type(cice_tracer_object_stand_in) :: ciceTracerObject

contains

  subroutine seaice_column_ocean_mixed_layer(domain)
    type(domain_type) :: domain
    print *, 'oracle stub: seaice_column_ocean_mixed_layer (the ocean mixed layer) is not compiled here'
    stop 6
  end subroutine seaice_column_ocean_mixed_layer

