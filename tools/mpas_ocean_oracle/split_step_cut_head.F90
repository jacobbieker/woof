! The head of the module the split-step oracle cuts out of
! mpas_ocn_time_integration_split.F (build.sh): ocn_time_integrator_split,
! verbatim, after its module's USE list and variables.  The module's init
! routine (the time manager's barotropic step count) is not cut: the driver
! sets nBtrSubcycles.  GM's two routines come from stubs_split.F90 (GM is not
! driven).
module ocn_split_step_cut
   use mpas_derived_types
   use mpas_pool_routines
   use mpas_constants
   use mpas_dmpar
   use mpas_vector_reconstruction
   use mpas_timer
   use mpas_threading
   use mpas_log
   use ocn_tendency
   use ocn_diagnostics
   use split_gm_vectors
   use ocn_gm
   use ocn_equation_of_state
   use ocn_vmix
   use ocn_time_average_coupled
   use ocn_effective_density_in_land_ice
   implicit none
   private
   save
   public :: ocn_time_integrator_split, nBtrSubcycles
   character (len=*), parameter :: subcycleGroupName = 'subcycleFields'
   character (len=*), parameter :: finalBtrGroupName = 'finalBtrFields'
   integer :: nBtrSubcycles
   contains
