! Service modules for WRF-Hydro's 2-D groundwater model (GWBASESWCRT = 3) as
! WRF 4.7.1 ships it, hydro/Routing/module_gw_gw2d.F90.
!
! build_gw2d.sh cuts gw2d_ini, gw2d_allocate, gwstep, SCOPY, gwSoilFlux,
! darcyGwSoil, aggregateQsgw and sub_tri_solv byte for byte from
! module_gw_gw2d.F90, and the GWBASESWCRT = 3 arm of module_HYDRO_drv.F90's
! groundwater call (:849-874), and #includes them below; module_gw_gw2d_data
! is compiled whole.  gwstep is compiled with MPP_LAND, as WRF-Hydro always
! is, against mpp_gw_single's one-process exchanges: there are no neighbours
! (left/right/up/down_id = -1, np_up_down = np_left_right = 1), so the
! halo exchange MPP_LAND_COM_REAL leaves its argument as it is and each ADI
! sweep is sub_tri_solv's; MPI_Reduce/MPI_Bcast/MPI_Comm_size reduce one
! process (delcur = delcur/1; with itermax = itermin = 1 it does not steer
! anything).  sub_n_form, parysolv1 and parxsolv1 (several processes along
! a sweep) are never reached and stop if they are (gw2d_mpp_stubs.f90).
! config_single and rt_single hold the pieces of nlst and rt_domain the cut
! code reads, with WRF-Hydro's names, kinds and shapes.  Nothing in this file
! computes anything.
module mpp_gw_single
  implicit none
  integer :: my_id = 0, IO_id = 0
  integer :: left_id = -1, right_id = -1, up_id = -1, down_id = -1
  integer :: np_up_down = 1, np_left_right = 1, p_up_down = 0, p_left_right = 0
  integer :: HYDRO_COMM_WORLD = 0, MPI_REAL = 0, MPI_SUM = 0
contains
  subroutine MPP_LAND_COM_REAL(arr, ix, jx, flag)
    integer :: ix, jx, flag
    real, dimension(ix, jx) :: arr
  end subroutine
  subroutine MPI_Reduce(sendbuf, recvbuf, n, dtype, op, root, comm, ierr)
    real :: sendbuf, recvbuf
    integer :: n, dtype, op, root, comm, ierr
    recvbuf = sendbuf
    ierr = 0
  end subroutine
  subroutine MPI_Comm_size(comm, nsize, ierr)
    integer :: comm, nsize, ierr
    nsize = 1
    ierr = 0
  end subroutine
  subroutine MPI_Bcast(buf, n, dtype, root, comm, ierr)
    real :: buf
    integer :: n, dtype, root, comm, ierr
    ierr = 0
  end subroutine
end module mpp_gw_single

module config_single
  ! the members of config_base's nlst the cut code reads (config.F90 :84-155)
  implicit none
  type nlst_field
    integer :: nsoil, aggfactRT, gwsoilcpl, GwPreCycles, GwPreDiagInterval, igrid, output_gw
    logical :: GwPreDiag = .false.
    real, allocatable, dimension(:) :: zsoil8
    real :: dt
    character(len=19) :: olddate = "123456", startdate = "123456"
    character(len=256) :: geo_finegrid_flnm = ""
  end type
  type(nlst_field), target :: nlst(1)
end module config_single

module rt_single
  ! the members of module_RT_data's rt_domain the cut code reads
  implicit none
  type grid_transform_field
    real, allocatable, dimension(:,:,:) :: smcrt, smcmaxrt
  end type
  type subsurface_properties_field
    real, allocatable, dimension(:) :: sldpth
  end type
  type subsurface_field
    type(grid_transform_field) :: grid_transform
    type(subsurface_properties_field) :: properties
  end type
  type overland_control_field
    real, allocatable, dimension(:,:) :: surface_water_head_routing
  end type
  type overland_properties_field
    real, allocatable, dimension(:,:,:) :: distance_to_neighbor
  end type
  type overland_field
    type(overland_control_field) :: control
    type(overland_properties_field) :: properties
  end type
  type rt_field
    integer :: ix, jx, ixrt, jxrt
    real, allocatable, dimension(:,:) :: elrt, latval, lonval
    integer, allocatable, dimension(:,:) :: soiltypRT
    type(subsurface_field) :: subsurface
    type(overland_field) :: overland
  end type
  type(rt_field), target :: rt_domain(1)
end module rt_single

module module_HYDRO_io
  ! output_gw_spinup writes a diagnostics file; GwPreDiag is off here
  implicit none
contains
  subroutine output_gw_spinup(igrid, split_output_count, ixrt, jxrt, startdate, date, &
      head, convgw, excess, geo_finegrid_flnm, dt, latval, lonval, dist, output_gw)
    integer :: igrid, split_output_count, ixrt, jxrt, output_gw
    character(len=*) :: startdate, date, geo_finegrid_flnm
    real, dimension(:,:) :: head, convgw, excess, latval, lonval
    real, dimension(:,:,:) :: dist
    real :: dt
    error stop 'output_gw_spinup is not part of the oracle'
  end subroutine
end module module_HYDRO_io

module gw2d_harness
  use mpp_gw_single
  use module_gw_gw2d_data, only: gw2d, gw_field
  use rt_single
  use config_single
  implicit none
  integer, private :: ierr
  integer, parameter :: rowshift = 0
  integer, parameter :: colshift = 1
contains
#include "gw2d_ini.inc"

#include "gw2d_allocate.inc"

#include "gwstep.inc"

#include "scopy.inc"

#include "gwsoilflux.inc"

#include "darcygwsoil.inc"

#include "aggregateqsgw.inc"

#include "sub_tri_solv.inc"

  ! module_HYDRO_drv.F90 :849-874, the GWBASESWCRT = 3 arm of the
  ! groundwater call, as the cut has it
  subroutine gw_hydro_step(did)
    integer, intent(in) :: did
#include "drv_gw2d.inc"
  end subroutine gw_hydro_step
end module gw2d_harness
