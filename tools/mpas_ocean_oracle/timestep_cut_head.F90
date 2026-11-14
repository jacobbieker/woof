! The head of the module the forward-mode oracle cuts out of
! mpas_ocn_time_integration.F (build.sh): ocn_timestep and
! ocn_timestep_init, verbatim, after the module's USE list (less the vector
! reconstruction and spline modules it does not call) and its variables.
! ocn_time_integrator_split comes from the split cut, RK4 from a stand-in.
module ocn_timestep_cut
   use mpas_derived_types
   use mpas_pool_routines
   use mpas_constants
   use mpas_timekeeping
   use mpas_dmpar
   use mpas_timer
   use mpas_log
   use ocn_constants
   use ocn_time_integration_rk4
   use ocn_split_step_cut
   implicit none
   private
   save
   public :: ocn_timestep, &
             ocn_timestep_init
    logical :: rk4On, splitOn
   contains
