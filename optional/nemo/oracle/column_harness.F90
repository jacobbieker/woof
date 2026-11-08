! sbccpl.F90's sbc_cpl_rcv (NEMO 5.0.2), cut byte for byte by build_column.sh
! and #included here.  sbccpl.F90 drives OASIS itself, so the module below
! declares, with sbccpl.F90's own names, kinds and shapes, the coupling-field
! registry the routine reads (srcv, frcv, nrcvinfo, the jpr_* indices, the
! receive switches) and replaces the OASIS get (cpl_rcv) by a copy from the
! fields the driver fills.  The configuration is WRF's: the ocean receives
! O_QsrOce, O_QnsOce, OOEvaMPr ('oce only'), O_OTaux1/O_OTauy1 on the local
! grid and O_TauMod; nothing else is active.  Nothing here computes a flux.
module sbccpl_harness
  use par_kind
  use dom_oce
  use sbc_oce
  use phycst
  use oce
  use eosbn2
  use lbclnk
  use lib_mpp
  use sbcwave
  use in_out_manager
  use iom
  implicit none
  integer, parameter :: jpr_otx1 = 1, jpr_oty1 = 2, jpr_otz1 = 3, jpr_taum = 4, jpr_w10m = 5
  integer, parameter :: jpr_qsroce = 6, jpr_qsrmix = 7, jpr_qnsoce = 8, jpr_qnsmix = 9
  integer, parameter :: jpr_oemp = 10, jpr_rain = 11, jpr_snow = 12, jpr_tevp = 13
  integer, parameter :: jpr_rnf = 14, jpr_cal = 15, jpr_icb = 16, jpr_isf = 17, jpr_sflx = 18
  integer, parameter :: jpr_fice = 19, jpr_co2 = 20, jpr_mslp = 21, jpr_sdrftx = 22, jpr_sdrfty = 23
  integer, parameter :: jpr_wper = 24, jpr_hsig = 25, jpr_wnum = 26, jpr_wstrf = 27, jpr_wdrag = 28
  integer, parameter :: jpr_charn = 29, jpr_tawx = 30, jpr_tawy = 31, jpr_twox = 32, jpr_twoy = 33
  integer, parameter :: jpr_phioc = 34, jpr_bhd = 35, jpr_tusd = 36, jpr_tvsd = 37, jpr_soce = 38
  integer, parameter :: jpr_toce = 39, jpr_ssh = 40, jpr_ocx1 = 41, jpr_ocy1 = 42, jpr_e3t1st = 43
  integer, parameter :: jpr_fraqsr = 44, jprcv = 44
  integer, parameter :: OASIS_idle = 0, OASIS_Rcv = 1
  type FLD_C
    character(len=32) :: cldes = '', clvgrd = '', clvor = '', clvref = ''
  end type
  type DYNARR
    real(wp), pointer :: z3(:,:,:) => null()
  end type
  type CPLFLD
    logical :: laction = .false.
    character(len=32) :: clname = ''
  end type
  type(CPLFLD) :: srcv(jprcv)
  type(DYNARR) :: frcv(jprcv)
  type(FLD_C) :: sn_rcv_tau, sn_rcv_emp
  integer :: nrcvinfo(jprcv) = 0, ncpl_qsr_freq = 0
  integer :: ncpl_sec = 1                 ! the coupling period, s (namcouple)
  real(wp), allocatable :: xcplmask(:,:,:), incoming(:,:,:)
  real(wp) :: pp_cldf = 0.81_wp, rpref = 101000._wp
  real(wp), allocatable, dimension(:,:) :: atm_co2, ssh_ib, ssh_ibb, apr, wmp, wnum, tauoc_wave, cdn_wave
  real(wp), allocatable, dimension(:,:) :: charn, tawx, tawy, twox, twoy, bhd_wave, tusd, tvsd
  real(wp), allocatable, dimension(:,:) :: ssh_m, ssu_m, ssv_m, e3t_m, frq_m, fwficb, fwfisf_oasis
  logical :: ln_tauoc = .false., ln_cdgw = .false., ln_charn = .false., ln_taw = .false.
  logical :: ln_bern_srfc = .false., ln_breivikFV_2016 = .false.
#  include "do_loop_substitute.h90"
#  include "domzgr_substitute.h90"
contains
  integer function cpl_freq(c)
    character(len=*), intent(in) :: c
    cpl_freq = 0
  end function
  subroutine cpl_rcv(kid, kstep, pdata, pmask, kinfo)
    integer, intent(in) :: kid, kstep
    real(wp), intent(inout) :: pdata(:,:,:)
    real(wp), intent(in) :: pmask(:,:,:)
    integer, intent(out) :: kinfo
    ! cpl_oasis3.F90's cpl_rcv: oasis_get delivers, and pdata changes, only
    ! at a coupling time; between them the field keeps its last value
    if (mod(kstep, ncpl_sec) == 0) then
      pdata(:,:,1) = incoming(:,:,kid)   ! the field the exchange delivered
      kinfo = OASIS_Rcv
    else
      kinfo = OASIS_idle
    end if
  end subroutine
  subroutine geo2oce(px, py, pz, cd, pte, ptn)
    real(wp) :: px(:,:), py(:,:), pz(:,:), pte(:,:), ptn(:,:)
    character(len=*) :: cd
    error stop 'geo2oce: cartesian stress is outside the harness'
  end subroutine
  subroutine rot_rep(pxin, pyin, cd_type, cdtodo, prot)
    real(wp) :: pxin(:,:), pyin(:,:), prot(:,:)
    character(len=*) :: cd_type, cdtodo
    error stop 'rot_rep: eastward-northward stress is outside the harness'
  end subroutine
  function sbc_dcy(pqsrin) result(zqsrout)
    real(wp) :: pqsrin(:,:)
    real(wp) :: zqsrout(size(pqsrin,1), size(pqsrin,2))
    zqsrout = pqsrin
    error stop 'sbc_dcy: the diurnal cycle is outside the harness'
  end function
  subroutine sbc_stokes(Kmm)
    integer :: Kmm
  end subroutine
#include "sbc_cpl_rcv.inc"
end module sbccpl_harness
