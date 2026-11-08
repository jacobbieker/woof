! Service module for WRF-Hydro's reach (Muskingum and Muskingum-Cunge)
! channel routing as WRF 4.7.1 ships it, hydro/Routing/module_channel_routing.F90.
!
! build_channel.sh cuts MUSKING and SUBMUSKINGCUNGE byte for byte, and the
! reach branch of drive_CHANNEL (its "IF(channel_option .ne. 3)" block,
! :705-885, up to and including the substep loop) byte for byte, and
! #includes them here.  drive_CHANNEL itself cannot be compiled outside the
! full routing build, so reach_route below declares, with drive_CHANNEL's own
! names, kinds and shapes (:543-679), exactly the variables the cut block
! reads and writes, allocates the lake work arrays as drive_CHANNEL's entry
! does (:680-688), and closes the block's IF.  The block is compiled with
! MPP_LAND, as WRF-Hydro always is, against mpp_reach_single's one-process
! exchanges, and drives the reservoir objects of rt_data_single (since 4.7.0
! a reach with TYPEL = 1 routes its lake through them).  Nothing in this file
! computes anything.
module mpp_reach_single
  ! WRF-Hydro is built with MPP_LAND.  On one process the exchanges the reach
  ! block calls reduce to the paths below, transcribed: there are no
  ! neighbours (left/right/up/down_id = -1) and every pixel is local
  ! (yw_mpp_nlinks = NLINKS, nlinks_index the identity); updateLinkV
  ! (updateLinkV8_mem, module_mpp_ReachLS.F90 :63-127) gathers one process's
  ! REAL*8 lateral inflow and rounds it to REAL; gbcastReal2 (:296-312)
  ! gathers QLINK and picks gbuf(index(j)); updateLake_seq (mpp_land.F90
  ! :2618-2650) on the I/O process reduces its own values (in = in).
  use iso_fortran_env, only: int64
  implicit none
  integer :: my_id = 0, io_id = 0
  integer :: left_id = -1, right_id = -1, up_id = -1, down_id = -1
contains
  subroutine updateLinkV(LinkV, outV)
    real*8, dimension(:) :: LinkV
    real, dimension(:) :: outV
    outV = LinkV
  end subroutine
  subroutine gbcastReal2(index, size1, inV, insize, outV)
    integer :: size1, insize
    integer, dimension(:) :: index
    real, dimension(:) :: inV, outV
    integer :: j
    outV = 0
    do j = 1, size1
      outV(j) = inV(index(j))
    end do
  end subroutine
  subroutine updateLake_seq(in, nsize, in0)
    integer :: nsize
    real, dimension(:) :: in
    real, dimension(:) :: in0
  end subroutine
end module mpp_reach_single

module rt_data_single
  ! The pieces of module_RT_data's rt_domain the reach block reads: the
  ! reservoir objects (module_reservoir's reservoir_container, level pool
  ! from Level_Pool/module_levelpool.F90, both compiled whole) and the
  ! per-lake diagnostics run() writes.
  use module_reservoir, only: reservoir_container
  implicit none
  type rt_field
    type(reservoir_container), allocatable :: reservoirs(:)
    integer, allocatable :: final_reservoir_type(:)
    real, allocatable :: reservoir_assimilated_value(:)
    character(len=256), allocatable :: reservoir_assimilated_source_file(:)
  end type
  type(rt_field), target :: rt_domain(1)
end module rt_data_single

module channel_harness
  use iso_fortran_env, only: int64
  use module_hydro_stop
  use mpp_reach_single
  use rt_data_single
  use module_levelpool, only: levelpool
  implicit none
  integer :: linkls_s(1) = (/ 1 /)
contains
#include "musking.inc"

#include "submuskingcunge.inc"

  ! levelpool_init for every lake, as module_RT's lake setup does
  subroutine init_lakes(nlakes, resht, lake_area, weir_elev, weir_coef, weir_len, dam_len, &
      orifice_elev, orifice_coef, orifice_area, max_depth, lake_opt)
    integer, intent(in) :: nlakes
    real, intent(inout) :: resht(nlakes)
    real, intent(in), dimension(nlakes) :: lake_area, weir_elev, weir_coef, weir_len, dam_len
    real, intent(in), dimension(nlakes) :: orifice_elev, orifice_coef, orifice_area, max_depth
    integer, intent(in) :: lake_opt(nlakes)
    integer :: i
    integer(kind=int64) :: ln
    allocate(rt_domain(1)%reservoirs(nlakes), rt_domain(1)%final_reservoir_type(nlakes))
    allocate(rt_domain(1)%reservoir_assimilated_value(nlakes))
    allocate(rt_domain(1)%reservoir_assimilated_source_file(nlakes))
    do i = 1, nlakes
      allocate(levelpool :: rt_domain(1)%reservoirs(i)%ptr)
      ln = i
      select type (r => rt_domain(1)%reservoirs(i)%ptr)
      type is (levelpool)
        call r%init(resht(i), lake_area(i), weir_elev(i), weir_coef(i), weir_len(i), &
            dam_len(i), orifice_elev(i), orifice_coef(i), orifice_area(i), max_depth(i), ln, &
            lake_opt(i))
      end select
    end do
  end subroutine init_lakes

  subroutine reach_route(channel_option, ixrt, jxrt, nlinks, nlinksl, dt, dtrt_ch, &
      qstrmvolrt, qinflowbase, lakeinflort, lake_mskrt, ch_lnkrt_sl, chanxi, chanyj, &
      node_area, linkid, to_node, typel, musk, musx, so, chanlen, mannn, chsslp, &
      bw, tw, tw_cc, n_cc, channk, qlink, hlink, qlateral, velocity, qloss, &
      nlakes, lake_lookup, resht, qlakei, qlakeo, gtonode, tonodeind, ntonodeind)
    integer, intent(in) :: channel_option, ixrt, jxrt, nlinks, nlinksl, nlakes
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
    integer, intent(in), dimension(:) :: lake_lookup
    real, intent(inout), dimension(nlakes) :: resht, qlakei, qlakeo
    integer, dimension(:) :: tonodeind
    integer(kind=int64), dimension(:,:) :: gtonode
    integer :: ntonodeind
    integer, parameter :: did = 1
    real :: km, x, qup, quc
    real, dimension(nlinks, 2) :: tmpqlink
    real(kind=8), dimension(nlinksl) :: lqlateral
    real, dimension(ntonodeind, 2) :: gqlink
    real, allocatable, dimension(:) :: tmpqlakeo, tmpqlakei, tmpresht
    integer :: k, m, n, jj, nt, nsteps, l_idx, iyw, yw_mpp_nlinks
    integer :: nlinks_index(nlinks)
    yw_mpp_nlinks = nlinks
    do jj = 1, nlinks
      nlinks_index(jj) = jj
    end do
    allocate(tmpqlakeo(nlakes), tmpqlakei(nlakes), tmpresht(nlakes))
#include "reach_block.inc"
    endif
    deallocate(tmpqlakeo, tmpqlakei, tmpresht)
  end subroutine reach_route
end module channel_harness
