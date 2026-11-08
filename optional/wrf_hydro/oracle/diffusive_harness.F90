! Service module for WRF-Hydro's diffusive-wave channel routing on nodes
! (channel_option = 3) as WRF 4.7.1 ships it,
! hydro/Routing/module_channel_routing.F90.
!
! build_diffusive.sh cuts DIFFUSION, AREAf, HEAD and SGNf byte for byte, and
! the diffusive branch of drive_CHANNEL (its "elseif(channel_option .eq. 3)"
! block, :893-1301, through "CVOL = CVOLTMP") byte for byte, and #includes
! them here.  drive_CHANNEL itself cannot be compiled outside the full routing
! build, so diffusive_route below declares, with drive_CHANNEL's own names,
! kinds and shapes (:543-679), the variables the cut block reads and writes,
! runs drive_CHANNEL's entry initialisation (:689-704), opens the IF the cut
! block continues and closes it.  The block is compiled with MPP_LAND, as
! WRF-Hydro always is, against mpp_single's one-process exchanges, and drives
! the level-pool reservoirs of rt_data_single.  Nothing in this file computes
! anything.
module mpp_single
  ! WRF-Hydro is built with MPP_LAND (hydro/CMakeLists.txt, every
  ! hydro/arc/macros.mpp.*).  On one process mpp_land.F90's channel and lake
  ! exchanges reduce to the paths below, transcribed: MPP_CHANNEL_COM_REAL(8),
  ! updateLake_grid and mpp_same_int1 leave their arguments as they are
  ! (left_id = right_id = -1, numprocs = 1), and sum_real8 (:2303-2324)
  ! rounds its REAL*8 sums through REAL (v = vin; vin = v).
  use iso_fortran_env, only: int64
  implicit none
contains
  subroutine MPP_CHANNEL_COM_REAL(Link_location, ix, jy, Link_V, size, flag)
    integer :: ix, jy, size, flag
    integer(kind=int64) :: Link_location(ix, jy)
    real :: Link_V(size)
  end subroutine
  subroutine MPP_CHANNEL_COM_REAL8(Link_location, ix, jy, Link_V, size, flag)
    integer :: ix, jy, size, flag
    integer(kind=int64) :: Link_location(ix, jy)
    real*8 :: Link_V(size)
  end subroutine
  subroutine sum_real8(vin, nsize)
    integer :: nsize
    real*8, dimension(nsize) :: vin
    real, dimension(nsize) :: v
    v = vin
    vin = v
  end subroutine
  subroutine updateLake_grid(in, nsize, lake_index)
    integer :: nsize
    real, dimension(nsize) :: in
    integer, dimension(nsize) :: lake_index
  end subroutine
  subroutine mpp_same_int1(v)
    integer :: v
  end subroutine
end module mpp_single

module rt_data_single
  ! The pieces of module_RT_data's rt_domain the diffusive block reads: the
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

module diffusive_harness
  use iso_fortran_env, only: int64
  use module_hydro_stop
  use mpp_single
  use rt_data_single
  use module_levelpool, only: levelpool
  implicit none
contains
  ! levelpool_init for every lake, as module_RT's lake setup does
  subroutine init_lakes(nlakes, resht, lake_area, weir_elev, weir_coef, weir_len, dam_len, &
      orifice_elev, orifice_coef, orifice_area, max_depth)
    integer, intent(in) :: nlakes
    real, intent(inout) :: resht(nlakes)
    real, intent(in), dimension(nlakes) :: lake_area, weir_elev, weir_coef, weir_len, dam_len
    real, intent(in), dimension(nlakes) :: orifice_elev, orifice_coef, orifice_area, max_depth
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
            dam_len(i), orifice_elev(i), orifice_coef(i), orifice_area(i), max_depth(i), ln, 1)
      end select
    end do
  end subroutine init_lakes

#include "diffusion.inc"

#include "areaf.inc"

#include "head.inc"

#include "sgnf.inc"

  subroutine diffusive_route(ixrt, jxrt, nlinks, dt, dtct, dtrt_ch, retdep_chan, qstrmvolrt, &
      qinflowbase, lakeinflort, qsubrt, ch_netrt, lake_mskrt, ch_netlnk, chanxi, chanyj, &
      node_area, from_node, to_node, typel, zelev, chanlen, mannn, so, chsslp, bw, &
      qlink, hlink, cvol, qlateral, nlakes, lakenode, resht, qlakei, qlakeo, &
      gwBaseSwCRT, gwChanCondSw, gwChanCondConstIn, gwChanCondConstOut, gwHead, qgw_chanrt)
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
    ! drive_CHANNEL's declarations (:543-679) for the names the block uses
    integer, intent(in) :: nlakes
    integer(kind=int64), intent(in), dimension(nlinks) :: lakenode
    real, intent(inout), dimension(nlakes) :: resht, qlakei, qlakeo
    integer, parameter :: did = 1, channel_option = 3
    ! the groundwater-channel exchange (GWBASESWCRT = 3): drive_CHANNEL's
    ! gwHead is gw2d's ho and qgw_chanrt gw2d's qgw_chanrt (HYDRO_drv :980)
    integer, intent(in) :: gwBaseSwCRT, gwChanCondSw
    real, intent(in) :: gwChanCondConstIn, gwChanCondConstOut
    real, intent(inout), dimension(:,:), allocatable :: gwHead, qgw_chanrt
    real, dimension(ixrt, jxrt) :: latval, lonval
    real, dimension(nlinks) :: dzGwChanHead, Q_GW_CHAN_FLUX
    real, dimension(nlakes) :: qlakeip, qllake
    real*8, dimension(nlakes) :: qlakei8, qllake8
    ! MPP_LAND arguments of drive_CHANNEL on one process (module_RT's
    ! single-task setup): every link and every lake is local
    integer(kind=int64), dimension(ixrt, jxrt) :: link_location
    integer, dimension(nlinks) :: nlinks_index
    integer, dimension(nlakes) :: lake_index
    integer :: mpp_nlinks, iyw, yw_mpp_nlinks
    real, dimension(nlinks) :: qsum, hlinktmp, cvoltmp, cd
    real*8, dimension(nlinks) :: qsum8
    real :: minDTCT
    integer :: i, krt, dt_steps, flag
    latval = 0.; lonval = 0.
    link_location = ch_netlnk
    nlinks_index = (/ (i, i = 1, nlinks) /)
    yw_mpp_nlinks = nlinks; mpp_nlinks = nlinks
    lake_index = (/ (i, i = 1, nlakes) /)
    ! drive_CHANNEL :689-704
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
