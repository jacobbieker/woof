! Service module for WRF-Hydro's reach (Muskingum and Muskingum-Cunge)
! channel routing as WRF 4.6.1 ships it, hydro/Routing/module_channel_routing.F90.
!
! build.sh cuts MUSKING and SUBMUSKINGCUNGE byte for byte, and the reach
! branch of drive_CHANNEL (its "IF(channel_option .ne. 3)" block, :704-865,
! up to and including the substep loop) byte for byte, and #includes them
! here.  drive_CHANNEL itself cannot be compiled outside the full routing
! build (its diffusive branch drives the reservoir objects), so reach_route
! below declares, with drive_CHANNEL's own names, kinds and shapes
! (:571-686), exactly the variables the cut block reads and writes, and
! closes the block's IF.  my_id and linkls_s exist only for SUBMUSKINGCUNGE's
! non-convergence message.  Nothing in this file computes anything.
module channel_harness
  use iso_fortran_env, only: int64
  use module_hydro_stop
  implicit none
  integer :: my_id = 0
  integer :: linkls_s(1) = (/ 1 /)
contains
#include "musking.inc"

#include "submuskingcunge.inc"

  subroutine reach_route(channel_option, ixrt, jxrt, nlinks, nlinksl, dt, dtrt_ch, &
      qstrmvolrt, qinflowbase, lakeinflort, lake_mskrt, ch_lnkrt_sl, chanxi, chanyj, &
      node_area, linkid, to_node, typel, musk, musx, so, chanlen, mannn, chsslp, &
      bw, tw, tw_cc, n_cc, channk, qlink, hlink, qlateral, velocity, qloss)
    integer, intent(in) :: channel_option, ixrt, jxrt, nlinks, nlinksl
    real, intent(in) :: dt, dtrt_ch
    real, intent(in), dimension(ixrt, jxrt) :: qstrmvolrt, qinflowbase, lakeinflort
    integer, intent(in), dimension(ixrt, jxrt) :: lake_mskrt
    integer(kind=int64), intent(in), dimension(ixrt, jxrt) :: ch_lnkrt_sl
    integer, intent(in), dimension(nlinks) :: chanxi, chanyj, typel
    real, intent(in), dimension(nlinks) :: node_area, musk, musx, so, chanlen, mannn
    real, intent(in), dimension(nlinks) :: chsslp, bw, tw, tw_cc, n_cc, channk
    integer(kind=int64), intent(in), dimension(nlinks) :: linkid, to_node
    real, intent(inout), dimension(:,:) :: qlink
    real, intent(inout), dimension(nlinks) :: hlink, qlateral
    real, dimension(:), intent(inout) :: velocity, qloss
    real :: km, x, qup, quc
    real, dimension(nlinks, 2) :: tmpqlink
    real*8, dimension(nlinks) :: lqlateral
    integer :: k, m, jj, nt, nsteps
#include "reach_block.inc"
    endif
  end subroutine reach_route
end module channel_harness
