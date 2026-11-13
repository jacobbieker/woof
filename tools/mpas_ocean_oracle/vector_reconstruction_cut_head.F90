! The head of the module build.sh cuts out of the pinned
! src/operators/mpas_vector_reconstruction.F: its generic mpas_reconstruct
! and the two routines behind it (mpas_reconstruct_1d, mpas_reconstruct_2d),
! verbatim.  The module's init routine (which needs the RBF interpolation
! and the vector operations) is not cut: the oracles give coeffs_reconstruct.
module mpas_vector_reconstruction
  use mpas_derived_types
  use mpas_pool_routines
  use mpas_constants
  use mpas_threading
  use mpas_timer, only : mpas_timer_start, mpas_timer_stop
  implicit none
  private
  public :: mpas_reconstruct
  interface mpas_reconstruct
     module procedure mpas_reconstruct_1d
     module procedure mpas_reconstruct_2d
  end interface
contains
