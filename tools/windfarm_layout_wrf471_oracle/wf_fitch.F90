! WRF's module data and dragcof, cut from phys/module_wind_fitch.F by
! extract.py and held here exactly as module_wind_fitch holds them.
! module_wind_fitch.F:44-66, less the USEs of module_driver_constants,
! module_dm and module_configure (max_domains is 1 here: one domain).
module wf_fitch
  USE module_model_constants, ONLY :  piconst
  USE module_llxy
  IMPLICIT NONE
  integer, parameter :: max_domains = 1
  include 'wf_decls.inc'
CONTAINS
  include 'wf_dragcof.inc'
end module wf_fitch
