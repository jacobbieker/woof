! Service modules for compiling NEMO 5.0.2's src/OCE/ZDF/zdftke.F90 and
! zdfsh2.F90 whole, outside NEMO.  Each module carries, under NEMO's module
! name, only the names those two files USE: state and grid arrays the driver
! fills, switches it sets, and I/O, timing, MPP and print services with no
! arithmetic.  phycst's constants are phycst.F90's values (rpi = 3.141592653589793_wp,
! vkarmn = 0.4_wp, rsmall = 0.5 * EPSILON(1.e0), grav = 9.80665_wp), and rho0 is
! eos_init's 1026._wp.
module oce
  use par_kind
  implicit none
  real(wp), allocatable :: uu(:,:,:,:), vv(:,:,:,:), rn2(:,:,:), rn2b(:,:,:)
end module oce

module dom_oce
  use par_kind
  implicit none
  integer :: jpi, jpj, jpk, jpkm1, nn_hls = 1
  integer :: ntsi, ntei, ntsj, ntej, Nis0, Nie0, Njs0, Nje0
  integer :: ntile = 0, nijtile = 1, nit000 = 1
  logical :: l_istiled = .false., ln_isfcav = .false., ln_rstart = .false.
  real(wp) :: rn_Dt
  real(wp), allocatable, dimension(:,:,:) :: tmask, umask, vmask, wmask, wumask, wvmask
  real(wp), allocatable, dimension(:,:,:) :: e3t_3d, e3w_3d, e3uw_3d, e3vw_3d, gdepw_3d, gdept_3d
  real(wp), allocatable :: ssmask(:,:), gphit(:,:)
  integer, allocatable :: mbkt(:,:), mikt(:,:)
end module dom_oce

module phycst
  use par_kind
  implicit none
  real(wp) :: rpi = 3.141592653589793_wp
  real(wp) :: vkarmn = 0.4_wp
  real(wp) :: grav = 9.80665_wp
  real(wp) :: rsmall = 0.5 * EPSILON( 1.e0 )
  real(wp) :: rho0 = 1026._wp
end module phycst

module sbc_oce
  use par_kind
  implicit none
  real(wp), allocatable :: utau(:,:), vtau(:,:), taum(:,:), fr_i(:,:)
  logical :: ln_cpl = .false., ln_stshear = .false.
  integer :: nn_ice = 0
end module sbc_oce

module zdfdrg
  use par_kind
  implicit none
  logical :: ln_drg_OFF = .false.
  real(wp), allocatable :: rCdU_bot(:,:), rCdU_top(:,:)
end module zdfdrg

module zdfmxl
  implicit none
  integer, allocatable :: nmln(:,:)
end module zdfmxl

module zdf_oce
  use par_kind
  implicit none
  real(wp), allocatable :: en(:,:,:), avt_k(:,:,:), avm_k(:,:,:), avtb_2d(:,:), avtb(:), avmb(:)
  logical :: ln_zdfiwm = .false.
end module zdf_oce

module sbcwave
  use par_kind
  implicit none
  logical :: cpl_sdrftx = .false., cpl_phioc = .false., ln_phioc = .false., ln_sdw = .false.
  real(wp), allocatable :: ut0sd(:,:), vt0sd(:,:), hsw(:,:), phioc(:,:), usd(:,:,:), vsd(:,:,:)
end module sbcwave

module in_out_manager
  implicit none
  character(len=:), allocatable :: numnam_ref, numnam_cfg
  integer :: numond = 6, numout = 6, numror = 0, numrow = 0, nitrst = 0
  logical :: lwm = .false., lwp = .false., ln_timing = .false.
  character(len=200) :: ctmp1, ctmp2, ctmp3
  type ctl_t
    logical :: l_prtctl = .false.
  end type
  type(ctl_t) :: sn_cfctl
end module in_out_manager

module iom
  use par_kind
  implicit none
  integer :: jpdom_auto = 0
contains
  logical function iom_use(c)
    character(len=*), intent(in) :: c
    iom_use = .false.
  end function
  subroutine iom_put(c, d)
    character(len=*), intent(in) :: c
    real(wp) :: d(:,:,:)
  end subroutine
  integer function iom_varid(k, c, ldstop)
    integer, intent(in) :: k
    character(len=*), intent(in) :: c
    logical, optional :: ldstop
    iom_varid = 0
  end function
  subroutine iom_get(k, d, c, a)
    integer, intent(in) :: k, d
    character(len=*), intent(in) :: c
    real(wp) :: a(:,:,:)
    error stop 'iom_get: restarts are outside the harness'
  end subroutine
  subroutine iom_rstput(k, n, u, c, a)
    integer, intent(in) :: k, n, u
    character(len=*), intent(in) :: c
    real(wp) :: a(:,:,:)
  end subroutine
end module iom

module lib_mpp
  implicit none
contains
  subroutine mpp_sum(c, k)
    character(len=*), intent(in) :: c
    integer, intent(inout) :: k
  end subroutine
  subroutine ctl_nam(ios, cd)
    integer, intent(inout) :: ios
    character(len=*), intent(in) :: cd
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
  subroutine prt_ctl(tab3d_1, clinfo1, tab3d_2, clinfo2)
    use par_kind
    real(wp), optional :: tab3d_1(:,:,:), tab3d_2(:,:,:)
    character(len=*), optional :: clinfo1, clinfo2
  end subroutine
end module prtctl
