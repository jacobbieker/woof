! Service modules for compiling NEMO 5.0.2's src/OCE/TRA/eosbn2.F90 whole,
! outside NEMO.  Each module below carries, under NEMO's own module name, only
! the names eosbn2.F90 USEs from it, as variables the driver sets (the grid,
! the masks, the namelist text) or as services with no arithmetic (timing,
! control prints, the namelist error check).  phycst's constants: eos_init
! sets rho0, rcp and the derived four itself (eosbn2.F90:1898-1899,
! 2330-2333); grav is phycst.F90:38's literal.
module dom_oce
  use par_kind
  implicit none
  integer :: jpi, jpj, jpk, jpkm1, nn_hls = 0
  integer :: ntsi, ntei, ntsj, ntej, Nis0, Nie0, Njs0, Nje0
  integer :: ntile = 0, nijtile = 1
  logical :: l_istiled = .false.
  integer, parameter :: jpts = 2, jp_tem = 1, jp_sal = 2
  real(wp), allocatable :: tmask(:,:,:), wmask(:,:,:), gdept_3d(:,:,:), gdepw_3d(:,:,:)
  real(wp), allocatable :: e3t_3d(:,:,:), e3w_3d(:,:,:)
  real(wp) :: rn_Dt = 1800._wp
end module dom_oce

module domutl
  implicit none
  interface lbnd_ij
    module procedure lbnd_ij_2d, lbnd_ij_3d, lbnd_ij_4d, lbnd_ij_5d
  end interface
contains
  function lbnd_ij_2d(pt) result(kbnd)
    use par_kind
    real(wp), intent(in) :: pt(:,:)
    integer :: kbnd(2)
    kbnd = 1
  end function
  function lbnd_ij_3d(pt) result(kbnd)
    use par_kind
    real(wp), intent(in) :: pt(:,:,:)
    integer :: kbnd(2)
    kbnd = 1
  end function
  function lbnd_ij_4d(pt) result(kbnd)
    use par_kind
    real(wp), intent(in) :: pt(:,:,:,:)
    integer :: kbnd(2)
    kbnd = 1
  end function
  function lbnd_ij_5d(pt) result(kbnd)
    use par_kind
    real(wp), intent(in) :: pt(:,:,:,:,:)
    integer :: kbnd(2)
    kbnd = 1
  end function
end module domutl

module phycst
  use par_kind
  implicit none
  real(wp) :: rho0, rcp, r1_rho0, r1_rcp, rho0_rcp, r1_rho0_rcp
  real(wp) :: grav = 9.80665_wp
  real(wp) :: rt0 = 273.15_wp
end module phycst

module stopar
  implicit none
  logical :: ln_sto_eos = .false.
  integer :: nn_sto_eos = 1
end module stopar

module stopts
  use par_kind
  implicit none
  real(wp), allocatable :: pts_ran(:,:,:,:,:)
end module stopts

module in_out_manager
  implicit none
  character(len=:), allocatable :: numnam_ref, numnam_cfg
  integer :: numond = 6, numout = 6
  character(len=200) :: ctmp1, ctmp2, ctmp3
  logical :: lwm = .false., lwp = .false., ln_timing = .false.
  type ctl_t
    logical :: l_prtctl = .false.
  end type
  type(ctl_t) :: sn_cfctl
end module in_out_manager

module lib_mpp
  implicit none
contains
  subroutine ctl_nam(ios, cd, ld)
    integer, intent(inout) :: ios
    character(len=*), intent(in) :: cd
    logical, intent(in) :: ld
    if (ios > 0) error stop 'namelist read error'
  end subroutine
  subroutine ctl_stop(c1, c2, c3, c4, c5)
    character(len=*), intent(in), optional :: c1, c2, c3, c4, c5
    error stop 'ctl_stop'
  end subroutine
  subroutine ctl_warn(c1, c2, c3, c4, c5)
    character(len=*), intent(in), optional :: c1, c2, c3, c4, c5
  end subroutine
end module lib_mpp

module prtctl
  implicit none
contains
  subroutine prt_ctl(tab2d_1, tab3d_1, tab4d_1, tab2d_2, tab3d_2, mask1, mask2, clinfo, clinfo1, clinfo2, clinfo3, kdim)
    use par_kind
    real(wp), optional :: tab2d_1(:,:), tab3d_1(:,:,:), tab4d_1(:,:,:,:), tab2d_2(:,:), tab3d_2(:,:,:)
    real(wp), optional :: mask1(:,:,:), mask2(:,:,:)
    character(len=*), optional :: clinfo, clinfo1, clinfo2, clinfo3
    integer, optional :: kdim
  end subroutine
end module prtctl

module timing
  implicit none
contains
  subroutine timing_start(c)
    character(len=*), intent(in) :: c
  end subroutine
  subroutine timing_stop(c)
    character(len=*), intent(in) :: c
  end subroutine
end module timing
