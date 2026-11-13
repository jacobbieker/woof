! The harness around mpas_tracer_advection_vflux4 and _vflux3, which build.sh
! cuts verbatim (awk ranges) out of the pinned
! src/operators/mpas_tracer_advection_helpers.F into
! mpas_tracer_advection_helpers_cut.F.  The module's other routines (the
! advection coefficients, derived from the mesh at init) need mpas_sort,
! mpas_hash and the geometry utilities, and the tendency never calls them.
module mpas_tracer_advection_helpers
   use mpas_kind_types
   implicit none
   private
   public :: mpas_tracer_advection_vflux4, mpas_tracer_advection_vflux3
contains
