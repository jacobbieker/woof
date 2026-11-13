! The harness around the routines build.sh cuts verbatim (awk ranges) out of
! the pinned src/core_ocean/shared/mpas_ocn_diagnostics.F into
! ocn_diagnostics_cut.F: ocn_diagnostic_solve, ocn_diagnostics_init and
! ocn_compute_land_ice_flux_input_fields.  This header repeats the module's
! own USE list (less the modules only its other routines need: the vector
! reconstruction, stream manager, IO units, GM and ALE thickness) and its
! module variables; the routines follow it unchanged.
module ocn_diagnostics
   use mpas_timer
   use mpas_derived_types
   use mpas_pool_routines
   use mpas_constants
   use mpas_threading
   use ocn_constants
   use ocn_equation_of_state
   use ocn_diagnostics_routines
   implicit none
   private
   save
   public :: ocn_diagnostic_solve, &
             ocn_diagnostics_init
   integer :: ke_cell_flag, ke_vertex_flag
   real (kind=RKIND) ::  fCoef
   real (kind=RKIND), pointer ::  coef_3rd_order
contains
