! The head of the module build.sh cuts out of the pinned
! src/core_ocean/shared/mpas_ocn_diagnostics.F: ocn_reconstruct_gm_vectors
! (mpas_reconstruct of the transport and bolus velocities, the relative slope
! and GM's stream function), verbatim.  The step and the forward driver's
! cold start call it under config_use_standardGM.
module split_gm_vectors
  use mpas_derived_types
  use mpas_pool_routines
  use mpas_vector_reconstruction
  use mpas_timer
  implicit none
  private
  public :: ocn_reconstruct_gm_vectors
contains
