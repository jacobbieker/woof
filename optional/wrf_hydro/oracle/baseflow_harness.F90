! Service module for WRF-Hydro's groundwater bucket as WRF 4.6.1 ships it,
! simp_gw_buck in hydro/Routing/module_GW_baseflow.F90.  build.sh cuts the
! routine byte for byte and #includes it here, in a module that USEs what
! module_GW_baseflow does without MPP_LAND (iso_fortran_env's int64).
module baseflow_harness
  use iso_fortran_env, only: int64
  implicit none
contains
#include "simp_gw_buck.inc"
end module baseflow_harness
