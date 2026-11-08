! Service module for WRF v4.7.1 frame/module_cpl.F's cpl_snd2 and cpl_rcv.
!
! build.sh cuts the two subroutines, byte for byte, out of the pinned
! module_cpl.F (from their SUBROUTINE line to their END SUBROUTINE line) and
! #includes them below, so the arithmetic compiled here is WRF's own text.
! This module supplies only what those two bodies reference and the pinned
! module cannot provide without WRF's Registry-generated module_domain:
!
!   * TYPE(domain) with exactly the grid components the two bodies read
!     (QFX, RAINCV, RAINNCV, DT, GSW, GLW, EMISS, SST, LH, HFX, u_phy, v_phy,
!     uoce, voce, rho, ust, id);
!   * cpl_get_fldid, cpl_tosend, cpl_toreceive: every field is coupled with
!     every external domain (what a namcouple naming all nine fields does);
!   * cpl_sndfield and cpl_oasis_rcv: the OASIS put/get, replaced by a copy
!     into or out of a harness array.  Nothing here computes a flux.
!
! STBOLT comes from share/module_model_constants.F compiled from the pinned
! tree, as module_cpl.F uses it.
module cpl_harness
  use module_model_constants, only: stbolt
  implicit none

  integer, parameter :: nfld = 9
  character(len=5), parameter :: coupler_name = 'oasis'
  integer :: nsecrun = 0

  type domain
    integer :: id = 1
    real :: dt
    real, allocatable, dimension(:,:) :: qfx, raincv, rainncv, gsw, glw
    real, allocatable, dimension(:,:) :: emiss, sst, lh, hfx, uoce, voce, ust
    real, allocatable, dimension(:,:,:) :: u_phy, v_phy, rho
  end type domain

  ! sent(:, :, kfldid): the field cpl_snd2 handed to cpl_sndfield.
  real, allocatable :: sent(:,:,:)
  ! received(:, :, kdomext): what the external domain's oasis_get returns.
  real, allocatable :: received(:,:,:)

contains

  function cpl_get_fldid( cdsuffix )
    character(*), intent(in) :: cdsuffix
    integer :: cpl_get_fldid
    select case (trim(cdsuffix))
    case ('EVAP-PRECIP');        cpl_get_fldid = 1
    case ('SURF_NET_SOLAR');     cpl_get_fldid = 2
    case ('SURF_NET_NON-SOLAR'); cpl_get_fldid = 3
    case ('TAUX');               cpl_get_fldid = 4
    case ('TAUY');               cpl_get_fldid = 5
    case ('TAUMOD');             cpl_get_fldid = 6
    case ('SST');                cpl_get_fldid = 7
    case ('UOCE');               cpl_get_fldid = 8
    case ('VOCE');               cpl_get_fldid = 9
    case default
      error stop 'cpl_get_fldid: unknown suffix'
    end select
  end function cpl_get_fldid

  function cpl_tosend( kdomwrf, kfldid, max_edom )
    integer, intent(in) :: kdomwrf, kfldid, max_edom
    logical, dimension(max_edom) :: cpl_tosend
    cpl_tosend(:) = .true.
  end function cpl_tosend

  function cpl_toreceive( kdomwrf, kdomext, kfldid )
    integer, intent(in) :: kdomwrf, kdomext, kfldid
    logical :: cpl_toreceive
    cpl_toreceive = .true.
  end function cpl_toreceive

  subroutine cpl_sndfield( kdomwrf, ldtosend, kfldid, pdata )
    integer, intent(in) :: kdomwrf
    logical, dimension(:), intent(in) :: ldtosend
    integer, intent(in) :: kfldid
    real, dimension(:,:), intent(in) :: pdata
    if (any(ldtosend)) sent(:, :, kfldid) = pdata(:, :)
  end subroutine cpl_sndfield

  subroutine cpl_oasis_rcv( kdomwrf, kdomext, kfldid, ksec, pcplrcv )
    integer, intent(in) :: kdomwrf, kdomext, kfldid, ksec
    real, dimension(:,:), intent(out) :: pcplrcv
    pcplrcv(:, :) = received(:, :, kdomext)
  end subroutine cpl_oasis_rcv

#include "cpl_snd2.inc"

#include "cpl_rcv.inc"

end module cpl_harness
