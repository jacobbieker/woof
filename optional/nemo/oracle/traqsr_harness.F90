! Service module for NEMO 5.0.2's penetrative solar radiation,
! src/OCE/TRA/traqsr.F90: tra_qsr, qsr_2BD and qsr_ext_lev.
!
! traqsr.F90 USEs most of NEMO (dom_oce, sbc_oce, iom, lib_mpp, ...), so
! build.sh cuts the three routines out of the pinned file, byte for byte, and
! #includes them here after NEMO's own do_loop_substitute.h90 and
! domzgr_substitute.h90, built with key_linssh and key_vco_3d (a linear free
! surface on a 3-D z/s grid).  This module declares only what those routines
! read and write, with NEMO's names, and stubs NEMO's I/O, timing, MPP and
! diagnostics services and the four other light schemes as error stops.
! Nothing in it computes a flux.  wp is NEMO's own par_kind (double).
module traqsr_harness
  use par_kind
  implicit none

  integer :: jpi, jpj, jpk, jpkm1
  integer, parameter :: jpts = 2, jpt = 3, jp_tem = 1
  integer :: ntsi, ntei, ntsj, ntej
  integer :: ntile = 1, nijtile = 1, nit000 = 1, numout = 6, numror = 0
  integer :: numrow = 0, nitrst = 0, jpdom_auto = 0, jptra_qsr = 0
  logical :: ln_timing = .false., l_istiled = .false., lwp = .false.
  logical :: l_trdtra = .false., ln_rstart = .false., l_1st_euler = .true.
  logical :: lrst_oce = .false., lk_top = .false.
  type ctl_t
    logical :: l_prtctl = .false.
  end type ctl_t
  type(ctl_t) :: sn_cfctl

  ! traqsr.F90 module variables (:56-98)
  integer, parameter :: np_RGB = 1, np_RGBc = 2, np_2BD = 3, np_5BD = 4
  integer, parameter :: np_5BDc = 5, np_BIO = 6
  integer :: nqsr, nk0, nkV, nksr
  real(wp) :: rn_abs, rn_si0, rn_si1, r1_si0, r1_si1, zz0

  ! the ocean state and grid the routines read (dom_oce, oce, sbc_oce, trc_oce)
  real(wp), allocatable :: e3t_3d(:,:,:), gdepw_3d(:,:,:), tmask(:,:,:), wmask(:,:,:)
  real(wp), allocatable :: qsr(:,:), qsr_hc(:,:,:), qsr_hc_b(:,:,:), fraqsr_1lev(:,:)
  real(wp), allocatable :: etot3(:,:,:)
  real(wp) :: rho0_rcp, r1_rho0_rcp, rDt

#  include "do_loop_substitute.h90"
#  include "domzgr_substitute.h90"

contains

  subroutine timing_start(c)
    character(*), intent(in) :: c
  end subroutine timing_start
  subroutine timing_stop(c)
    character(*), intent(in) :: c
  end subroutine timing_stop
  logical function iom_use(c)
    character(*), intent(in) :: c
    iom_use = .false.
  end function iom_use
  subroutine iom_get(a, b, c, d)
    integer, intent(in) :: a, b
    character(*), intent(in) :: c
    real(wp) :: d(:,:,:)
    error stop 'iom_get: restart reading is outside the harness'
  end subroutine iom_get
  subroutine iom_put(c, d)
    character(*), intent(in) :: c
    real(wp) :: d(:,:,:)
  end subroutine iom_put
  subroutine iom_rstput(a, b, c, d, e)
    integer, intent(in) :: a, b, c
    character(*), intent(in) :: d
    real(wp) :: e(..)
  end subroutine iom_rstput
  subroutine trd_tra(a, b, c, d, e, f, g)
    integer, intent(in) :: a, b, c
    character(*), intent(in) :: d
    integer, intent(in) :: e, f
    real(wp) :: g(:,:,:)
  end subroutine trd_tra
  subroutine prt_ctl(tab3d_1, clinfo1, mask1, clinfo3)
    real(wp) :: tab3d_1(:,:,:), mask1(:,:,:)
    character(*) :: clinfo1, clinfo3
  end subroutine prt_ctl
  subroutine mpp_max(c, k)        ! one process: the local value is the max
    character(*), intent(in) :: c
    integer, intent(inout) :: k
  end subroutine mpp_max
  subroutine qsr_RGBc(kt, Kmm, pts, Krhs)
    integer :: kt, Kmm, Krhs
    real(wp) :: pts(:,:,:,:,:)
    error stop 'qsr_RGBc is not part of this oracle'
  end subroutine qsr_RGBc
  subroutine qsr_RGB(kt, Kmm, pts, Krhs)
    integer :: kt, Kmm, Krhs
    real(wp) :: pts(:,:,:,:,:)
    error stop 'qsr_RGB is not part of this oracle'
  end subroutine qsr_RGB
  subroutine qsr_5BDc(kt, Kmm, pts, Krhs)
    integer :: kt, Kmm, Krhs
    real(wp) :: pts(:,:,:,:,:)
    error stop 'qsr_5BDc is not part of this oracle'
  end subroutine qsr_5BDc
  subroutine qsr_5BD(kt, Kmm, pts, Krhs)
    integer :: kt, Kmm, Krhs
    real(wp) :: pts(:,:,:,:,:)
    error stop 'qsr_5BD is not part of this oracle'
  end subroutine qsr_5BD

#include "tra_qsr.inc"

#include "qsr_2bd.inc"

#include "qsr_ext_lev.inc"

end module traqsr_harness
