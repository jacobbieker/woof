! Service module for WRF-Hydro's diffusive-wave channel routing on nodes
! (channel_option = 3) as WRF 4.6.1 ships it,
! hydro/Routing/module_channel_routing.F90.
!
! build_diffusive.sh cuts DIFFUSION, AREAf, HEAD and SGNf byte for byte, and
! the diffusive branch of drive_CHANNEL (its "elseif(channel_option .eq. 3)"
! block, :873-1281, through "CVOL = CVOLTMP") byte for byte, and #includes
! them here.  drive_CHANNEL itself cannot be compiled outside the full routing
! build, so diffusive_route below declares, with drive_CHANNEL's own names,
! kinds and shapes (:571-686), the variables the cut block reads and writes,
! runs drive_CHANNEL's entry initialisation (:688-703), opens the IF the cut
! block continues and closes it.  The reservoir objects the block drives
! (rt_domain(did)%reservoirs) come from reservoir_stub: with NLAKES = 0 they
! are never called.  Nothing in this file computes anything.
module reservoir_stub
  implicit none
  type reservoir_ptr
    class(reservoir_base), pointer :: ptr => null()
  end type
  type, abstract :: reservoir_base
  contains
    procedure(run_iface), deferred :: run
  end type
  abstract interface
    subroutine run_iface(this, qlakeip, qlakei, qllake, resht, qlakeo, dtct, ftype, aval, asrc)
      import :: reservoir_base
      class(reservoir_base), intent(inout) :: this
      real, intent(inout) :: qlakeip, qlakei, qllake, resht, qlakeo
      real, intent(in) :: dtct
      integer, intent(inout) :: ftype
      real, intent(inout) :: aval
      character(len=256), intent(inout) :: asrc
    end subroutine
  end interface
  type rt_field
    type(reservoir_ptr), allocatable :: reservoirs(:)
    integer, allocatable :: final_reservoir_type(:)
    real, allocatable :: reservoir_assimilated_value(:)
    character(len=256), allocatable :: reservoir_assimilated_source_file(:)
  end type
  type(rt_field), target :: rt_domain(1)
end module reservoir_stub

module diffusive_harness
  use iso_fortran_env, only: int64
  use module_hydro_stop
  use reservoir_stub
  implicit none
contains
#include "diffusion.inc"

#include "areaf.inc"

#include "head.inc"

#include "sgnf.inc"

  subroutine diffusive_route(ixrt, jxrt, nlinks, dt, dtct, dtrt_ch, retdep_chan, qstrmvolrt, &
      qinflowbase, lakeinflort, qsubrt, ch_netrt, lake_mskrt, ch_netlnk, chanxi, chanyj, &
      node_area, from_node, to_node, typel, zelev, chanlen, mannn, so, chsslp, bw, &
      qlink, hlink, cvol, qlateral)
    integer, intent(in) :: ixrt, jxrt, nlinks
    real, intent(in) :: dt, dtrt_ch
    real, intent(inout) :: dtct
    real :: retdep_chan
    real, intent(in), dimension(ixrt, jxrt) :: qstrmvolrt, qinflowbase, lakeinflort, qsubrt
    integer, intent(in), dimension(ixrt, jxrt) :: ch_netrt, lake_mskrt
    integer(kind=int64), intent(in), dimension(ixrt, jxrt) :: ch_netlnk
    integer, intent(in), dimension(nlinks) :: chanxi, chanyj, typel
    real, intent(in), dimension(nlinks) :: node_area, zelev, chanlen, mannn, so, chsslp, bw
    integer(kind=int64), intent(in), dimension(nlinks) :: from_node, to_node
    real, intent(inout), dimension(:,:) :: qlink
    real, intent(inout), dimension(nlinks) :: hlink, cvol, qlateral
    ! drive_CHANNEL's declarations (:571-686) for the names the block uses
    integer, parameter :: did = 1, nlakes = 0, channel_option = 3
    integer :: gwBaseSwCRT = 0, gwChanCondSw = 0
    real :: gwChanCondConstIn = 0., gwChanCondConstOut = 0.
    real, dimension(ixrt, jxrt) :: latval, lonval
    real, dimension(:,:), allocatable :: gwHead, qgw_chanrt
    real, dimension(nlinks) :: dzGwChanHead, Q_GW_CHAN_FLUX
    real, dimension(nlakes) :: resht, qlakei, qlakeip, qlakeo, qllake
    real*8, dimension(nlakes) :: qlakei8, qllake8
    integer(kind=int64), dimension(nlinks) :: lakenode
    real, dimension(nlinks) :: qsum, hlinktmp, cvoltmp, cd
    real*8, dimension(nlinks) :: qsum8
    real :: minDTCT
    integer :: i, krt, dt_steps, flag
    lakenode = 0
    latval = 0.; lonval = 0.
    ! drive_CHANNEL :688-703
    QLAKEIP = 0
    QLAKEI8 = 0
    HLINKTMP = 0
    CVOLTMP = 0
    CD = 0
    QLateral = 0
    QSUM     = 0
    QLLAKE   = 0
    dzGwChanHead = 0.
    if (channel_option .ne. 3) then
      error stop 'the harness runs channel_option 3 only'
#include "diffusive_block.inc"
    endif
  end subroutine diffusive_route
end module diffusive_harness
