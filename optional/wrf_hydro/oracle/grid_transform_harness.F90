! Service module for WRF-Hydro's LSM <-> routing grid transforms as WRF 4.6.1
! ships them: disaggregateDomain (hydro/Routing/Noah_distr_routing.F90) and
! aggregateDomain (hydro/HYDRO_drv/module_HYDRO_drv.F90).
!
! build.sh cuts both routines byte for byte from the pinned files and
! #includes them here.  aggregateDomain reads WRF-Hydro's global rt_domain(did)
! and nlst(did); this module declares objects of those names whose types carry
! exactly the components the routine reads, at WRF-Hydro's precision
! (module_rt_inc.F90:207-228: REAL).  Nothing in this file computes anything.
module grid_transform_harness
  use module_hydro_stop
  implicit none

  type gt_control
    real, allocatable :: surface_water_head_routing(:,:), surface_water_head_lsm(:,:)
  end type gt_control
  type gt_oprops
    real, allocatable :: distance_to_neighbor(:,:,:)
  end type gt_oprops
  type gt_overland
    type(gt_control) :: control
    type(gt_oprops) :: properties
  end type gt_overland
  type gt_transform
    real, allocatable :: smcrt(:,:,:), smcmaxrt(:,:,:)
  end type gt_transform
  type gt_sprops
    real, allocatable :: sldpth(:)
  end type gt_sprops
  type gt_subsurface
    type(gt_transform) :: grid_transform
    type(gt_sprops) :: properties
  end type gt_subsurface
  type gt_domain
    integer :: ix, jx, ixrt, jxrt
    real :: sfcheadaggrt, lsmvol
    real, allocatable :: sh2oaggrt(:), infxswgt(:,:), sh2owgt(:,:,:)
    real, allocatable :: sh2ox(:,:,:), sice(:,:,:), smc(:,:,:)
    type(gt_overland) :: overland
    type(gt_subsurface) :: subsurface
  end type gt_domain
  type gt_nlst
    integer :: aggfactrt, nsoil
  end type gt_nlst
  type(gt_domain) :: rt_domain(1)
  type(gt_nlst) :: nlst(1)

contains

#include "disaggregate.inc"

#include "aggregate.inc"

end module grid_transform_harness
