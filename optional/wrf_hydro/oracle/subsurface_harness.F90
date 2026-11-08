! Service module for WRF-Hydro's subsurface routing as WRF 4.6.1 ships it
! (hydro/Routing/Noah_distr_routing_subsurface.F90, and GETSUB8/GETSUB8DIR
! from hydro/Routing/Noah_distr_routing.F90).
!
! WRF-Hydro builds those files with MPP_LAND, where each routine USEs the
! subsurface data modules itself.  Without MPP_LAND the USE lines are
! preprocessed away, so build.sh #includes the routines here, inside a module
! that USEs the same pinned modules: the routines reach the types by host
! association and their text is compiled unchanged.  Nothing in this file
! computes anything.
module subsurface_harness
  use module_subsurface_data
  use module_subsurface_static_data
  use module_subsurface_input
  use module_subsurface_output
  use module_hydro_stop
  implicit none
contains
#include "getsub8.inc"

#include "Noah_distr_routing_subsurface.F90"
end module subsurface_harness
