! The head of the module the split-step oracle cuts out of
! mpas_ocn_time_integration_split.F (build.sh): ocn_time_integrator_split,
! and ocn_time_integration_split_init (the barotropic subcycle count from
! config_dt and config_btr_dt, and the cold start's barotropic velocity),
! verbatim, after its module's USE list and variables.  run_split_step sets
! nBtrSubcycles itself; run_forward calls the init.  GM's two routines come from stubs_split.F90 (GM is not
! driven).
module ocn_split_step_cut
   use mpas_derived_types
   use mpas_pool_routines
   use mpas_constants
   use mpas_dmpar
   use mpas_timekeeping
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
   public :: ocn_time_integrator_split, ocn_time_integration_split_init, nBtrSubcycles
   character (len=*), parameter :: subcycleGroupName = 'subcycleFields'
   character (len=*), parameter :: finalBtrGroupName = 'finalBtrFields'
   integer :: nBtrSubcycles
   contains
