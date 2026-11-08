! Service module for WRF-Hydro's overland routing as WRF 4.7.1 ships it
! (hydro/Routing/Noah_distr_routing_overland.F90: OverlandRouting, OV_RTNG,
! ROUTE_OVERLAND1, ROUTE_OVERLAND2; GETMAX8DIR and GET8DIR from
! Noah_distr_routing.F90).  Without MPP_LAND, OverlandRouting has no USE of
! the overland data module, so the routines are #included in a module that
! USEs the same pinned modules.  Nothing in this file computes anything.
module overland_harness
  use overland_data
  use module_hydro_stop
  implicit none
contains
#include "getmax8dir.inc"

#include "Noah_distr_routing_overland.F90"
end module overland_harness
