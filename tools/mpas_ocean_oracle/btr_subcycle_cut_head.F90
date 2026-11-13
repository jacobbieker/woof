! The head of the module the barotropic oracle cuts out of
! mpas_ocn_time_integration_split.F (build.sh): ocn_time_integrator_split's
! declarations and config lookups, then its stage 2 ("Barotropic velocity
! (2D) prediction, explicitly subcycled"), each an awk range of the verbatim
! source, inside a routine of their own.  nBtrSubcycles is the module's
! variable (ocn_time_integration_split_init sets it from the time steps; the
! driver sets it here).
module ocn_btr_stage_cut
   use mpas_derived_types
   use mpas_pool_routines
   use mpas_constants
   use mpas_dmpar
   use mpas_timer
   use mpas_threading
   use mpas_log
   implicit none
   private
   save
   public :: ocn_btr_stage, nBtrSubcycles
   character (len=*), parameter :: subcycleGroupName = 'subcycleFields'
   character (len=*), parameter :: finalBtrGroupName = 'finalBtrFields'
   integer :: nBtrSubcycles
   contains
    subroutine ocn_btr_stage(domain, dt)
