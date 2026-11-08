! Service modules for compiling NEMO 5.0.2's tracer column whole, outside
! NEMO: eosbn2.F90, traqsr.F90, zdfsh2.F90, zdfdrg.F90, zdftke.F90,
! trazdf.F90, trasbc.F90 and traatf_qco.F90, plus sbccpl.F90's sbc_cpl_rcv (cut, see
! column_harness.F90).  Each module below carries, under NEMO's module name,
! only the names those files USE: state and grid arrays the driver fills,
! switches it sets, and I/O, timing, MPP, trend and print services with no
! arithmetic.  The switches select the configuration the port takes (linear
! free surface, leapfrog, no sea ice, no runoff, no ice shelves, no lateral
! slopes, no trend diagnostics, no stochastic EOS, no waves).  phycst's values
! are phycst.F90's literals; eos_init sets rho0, rcp and their derived four.
module oce
  use par_kind
  implicit none
  real(wp), allocatable :: uu(:,:,:,:), vv(:,:,:,:), rn2(:,:,:), rn2b(:,:,:), wi(:,:,:)
  real(wp), allocatable :: ts(:,:,:,:,:), ssh(:,:,:), fraqsr_1lev(:,:)
end module oce

module dom_oce
  use par_kind
  implicit none
  integer :: jpi, jpj, jpk, jpkm1, nn_hls = 1
  integer :: ntsi, ntei, ntsj, ntej, Nis0, Nie0, Njs0, Nje0
  integer :: ntile = 0, nijtile = 1, nit000 = 1
  integer, parameter :: jpts = 2, jp_tem = 1, jp_sal = 2, jpt = 3
  logical :: l_istiled = .false., ln_isfcav = .false., ln_rstart = .false., l_1st_euler = .true.
  logical, parameter :: lk_linssh = .true.
  real(wp) :: rn_Dt = 1800._wp, rDt = 3600._wp, r1_Dt = 1._wp / 3600._wp, rn_atfp = 0.1_wp
  real(wp), allocatable, dimension(:,:,:) :: tmask, umask, vmask, wmask, wumask, wvmask
  real(wp), allocatable, dimension(:,:,:) :: e3t_3d, e3w_3d, e3uw_3d, e3vw_3d, gdepw_3d, gdept_3d
  real(wp), allocatable, dimension(:,:,:) :: e3t_0, gdepw_0, e3u_3d, e3v_3d
  real(wp), allocatable :: ssmask(:,:), gphit(:,:), r3t(:,:,:), gdepw_1d(:), r1_e1e2t(:,:), r3t_f(:,:)
  integer, allocatable :: mbkt(:,:), mikt(:,:), mbku(:,:), mbkv(:,:), miku(:,:), mikv(:,:)
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

module domtile
  implicit none
  logical :: ln_tile = .false.
contains
  subroutine dom_tile(ktsi, ktsj, ktei, ktej, ktile, ldhold, cstr)
    integer :: ktsi, ktsj, ktei, ktej
    integer, optional :: ktile
    logical, optional :: ldhold
    character(len=*), optional :: cstr
  end subroutine
  subroutine dom_tile_start(ldhold)
    logical, optional :: ldhold
  end subroutine
  subroutine dom_tile_stop(ldhold)
    logical, optional :: ldhold
  end subroutine
end module domtile

module phycst
  use par_kind
  implicit none
  real(wp) :: rho0, rcp, r1_rho0, r1_rcp, rho0_rcp, r1_rho0_rcp
  real(wp) :: rpi = 3.141592653589793_wp, vkarmn = 0.4_wp, grav = 9.80665_wp
  real(wp) :: rsmall = 0.5 * EPSILON( 1.e0 ), rt0 = 273.15_wp
  real(wp) :: rLfus = 3.34e+5_wp
end module phycst

module sbc_oce
  use par_kind
  implicit none
  real(wp), allocatable :: utau(:,:), vtau(:,:), taum(:,:), wndm(:,:), fr_i(:,:)
  real(wp), allocatable :: qns(:,:), qsr(:,:), emp(:,:), sfx(:,:), rnf(:,:), sst_m(:,:), sss_m(:,:)
  real(wp), allocatable :: sbc_tsc(:,:,:), sbc_tsc_b(:,:,:), cloud_fra(:,:)
  real(wp), allocatable :: qsr_hc(:,:,:), qsr_hc_b(:,:,:)
  logical :: ln_cpl = .true., ln_stshear = .false., ln_mixcpl = .false., ln_dm2dc = .false.
  logical :: ln_wave = .false., ln_rnf = .false., ln_apr_dyn = .false.
  integer :: nn_ice = 0, nn_components = 0, nn_cplmodel = 1
  integer, parameter :: jp_iam_nemo = 0, jp_iam_oce = 1, jp_iam_sas = 2
end module sbc_oce

module trc_oce
  use par_kind
  implicit none
  real(wp), allocatable :: etot3(:,:,:)
  logical :: lk_top = .false.
  real(wp) :: r_si2
contains
  subroutine trc_oce_tab(ptab)
    real(wp) :: ptab(:,:)
    error stop 'trc_oce_tab: the RGB tables are outside the harness'
  end subroutine
  integer function trc_oce_ext_lev(prldex, pqsr_frc)
    real(wp) :: prldex, pqsr_frc
    trc_oce_ext_lev = 0
    error stop 'trc_oce_ext_lev: bio light is outside the harness'
  end function
end module trc_oce

module zdfmxl
  implicit none
  integer, allocatable :: nmln(:,:)
end module zdfmxl

module zdf_oce
  use par_kind
  implicit none
  real(wp), allocatable :: en(:,:,:), avt_k(:,:,:), avm_k(:,:,:), avtb_2d(:,:), avtb(:), avmb(:)
  real(wp), allocatable :: avt(:,:,:), avs(:,:,:), avm(:,:,:)
  logical :: ln_zdfiwm = .false., ln_zdfddm = .false., ln_zad_Aimp = .false., ln_zdfswm = .false.
end module zdf_oce

module zdfmfc
  use par_kind
  implicit none
  logical :: ln_zdfmfc = .false.
  real(wp), allocatable :: edmfm(:,:,:), edmftra(:,:,:,:)
end module zdfmfc

module ldftra
  implicit none
  logical :: l_ldfslp = .false., ln_traldf_msc = .false., ln_traldf_iso = .false.
end module ldftra

module ldfslp
  use par_kind
  implicit none
  real(wp), allocatable :: akz(:,:,:), ah_wslp2(:,:,:)
end module ldfslp

module sbcwave
  use par_kind
  implicit none
  logical :: cpl_sdrftx = .false., cpl_phioc = .false., ln_phioc = .false., ln_sdw = .false.
  real(wp), allocatable :: ut0sd(:,:), vt0sd(:,:), hsw(:,:), phioc(:,:), usd(:,:,:), vsd(:,:,:)
end module sbcwave

module sbcmod
end module sbcmod

module sbcrnf
  use par_kind
  implicit none
  real(wp), allocatable :: rnf_tsc(:,:,:), rnf_tsc_b(:,:,:), h_rnf(:,:)
  integer, allocatable :: nk_rnf(:,:)
  logical :: l_rnfcpl = .false.
end module sbcrnf

module isf_oce
  use par_kind
  implicit none
  logical :: ln_isf = .false., ln_isfcav_mlt = .false., ln_isfpar_mlt = .false., ln_isfcpl = .false.
  integer, allocatable :: misfkb_cav(:,:), misfkb_par(:,:), misfkt_cav(:,:), misfkt_par(:,:)
  real(wp), allocatable :: rfrac_tbl_cav(:,:), rfrac_tbl_par(:,:), rhisf_tbl_cav(:,:), rhisf_tbl_par(:,:)
  real(wp), allocatable :: risf_cav_tsc(:,:,:), risf_cav_tsc_b(:,:,:), risf_par_tsc(:,:,:)
  real(wp), allocatable :: risf_par_tsc_b(:,:,:), risfcpl_tsc(:,:,:,:)
end module isf_oce

module bdy_oce
  implicit none
  logical :: ln_bdy = .false.
end module bdy_oce

module bdytra
end module bdytra

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

module trd_oce
  implicit none
  logical :: l_trdtra = .false.
  logical :: l_trdtrc = .false.
  logical :: l_trddyn = .false.
  integer :: jpdyn_bfr = 0
  integer :: jptra_zdf = 0, jptra_qsr = 0, jptra_nsr = 0, jptra_atf = 0, jptra_tot = 0, jptra_zdfp = 0
end module trd_oce

module trddyn
  use par_kind
  implicit none
contains
  subroutine trd_dyn(pu, pv, ktrd, kt, Kmm)
    real(wp) :: pu(:,:,:), pv(:,:,:)
    integer :: ktrd, kt, Kmm
  end subroutine
end module trddyn

module sbc_ice
  use par_kind
  implicit none
  real(wp), allocatable :: rCdU_ice(:,:)
end module sbc_ice

module trdtra
  use par_kind
  implicit none
contains
  subroutine trd_tra(kt, Kmm, Krhs, c, jn, jt, t)
    integer :: kt, Kmm, Krhs, jn, jt
    character(len=*) :: c
    real(wp) :: t(:,:,:)
  end subroutine
end module trdtra

module in_out_manager
  implicit none
  character(len=:), allocatable :: numnam_ref, numnam_cfg
  integer :: numond = 6, numout = 6, numror = 0, numrow = 0, nitrst = 0, no_print = 0
  logical :: lwm = .false., lwp = .false., ln_timing = .false., lrst_oce = .false.
  character(len=200) :: ctmp1, ctmp2, ctmp3
  type ctl_t
    logical :: l_prtctl = .false.
  end type
  type(ctl_t) :: sn_cfctl
end module in_out_manager

module iom
  use par_kind
  implicit none
  integer :: jpdom_auto = 0, jpdom_global = 0
  interface iom_put
    module procedure iom_put_2d, iom_put_3d
  end interface
  interface iom_rstput
    module procedure iom_rstput_2d, iom_rstput_3d
  end interface
  interface iom_get
    module procedure iom_get_2d, iom_get_3d, iom_get_file2d
  end interface
contains
  logical function iom_use(c)
    character(len=*), intent(in) :: c
    iom_use = .false.
  end function
  subroutine iom_put_2d(c, d)
    character(len=*), intent(in) :: c
    real(wp) :: d(:,:)
  end subroutine
  subroutine iom_put_3d(c, d)
    character(len=*), intent(in) :: c
    real(wp) :: d(:,:,:)
  end subroutine
  subroutine iom_open(c, k)
    character(len=*), intent(in) :: c
    integer :: k
    error stop 'iom_open: boost files are outside the harness'
  end subroutine
  subroutine iom_close(k)
    integer :: k
  end subroutine
  integer function iom_varid(k, c, ldstop)
    integer, intent(in) :: k
    character(len=*), intent(in) :: c
    logical, optional :: ldstop
    iom_varid = 0
  end function
  subroutine iom_get_2d(k, d, c, a)
    integer, intent(in) :: k, d
    character(len=*), intent(in) :: c
    real(wp) :: a(:,:)
    error stop 'iom_get: restarts are outside the harness'
  end subroutine
  subroutine iom_get_3d(k, d, c, a)
    integer, intent(in) :: k, d
    character(len=*), intent(in) :: c
    real(wp) :: a(:,:,:)
    error stop 'iom_get: restarts are outside the harness'
  end subroutine
  subroutine iom_get_file2d(k, d, c, a, n)
    integer, intent(in) :: k, d, n
    character(len=*), intent(in) :: c
    real(wp) :: a(:,:)
    error stop 'iom_get: boost files are outside the harness'
  end subroutine
  subroutine iom_rstput_2d(k, n, u, c, a)
    integer, intent(in) :: k, n, u
    character(len=*), intent(in) :: c
    real(wp) :: a(:,:)
  end subroutine
  subroutine iom_rstput_3d(k, n, u, c, a)
    integer, intent(in) :: k, n, u
    character(len=*), intent(in) :: c
    real(wp) :: a(:,:,:)
  end subroutine
end module iom

module fldread
  use par_kind
  implicit none
  type FLD_N                    ! fldread.F90's layout, so namtra_qsr's sn_chl reads
    character(len=256) :: clname = ''
    real(wp) :: freqh = 0._wp
    character(len=34) :: clvar = ''
    logical :: ln_tint = .false., ln_clim = .false.
    character(len=8) :: clftyp = ''
    character(len=256) :: wname = ''
    character(len=34) :: vcomp = '', lname = ''
  end type
  type FLD
    real(wp), allocatable :: fnow(:,:,:), fdta(:,:,:,:)
  end type
contains
  subroutine fld_read(kt, kn, sd)
    integer :: kt, kn
    type(FLD) :: sd(:)
    error stop 'fld_read: chlorophyll input is outside the harness'
  end subroutine
  subroutine fld_fill(sdf, sdf_n, cdir, cdcaller, cdtitle, cdnam, knoprint)
    type(FLD) :: sdf(:)
    type(FLD_N) :: sdf_n(:)
    character(len=*) :: cdir, cdcaller, cdtitle, cdnam
    integer, optional :: knoprint
    error stop 'fld_fill: chlorophyll input is outside the harness'
  end subroutine
end module fldread

module restart
end module restart

module lib_mpp
  implicit none
  interface mpp_max
    module procedure mpp_max_i
  end interface
contains
  subroutine mpp_sum(c, k)
    character(len=*), intent(in) :: c
    integer, intent(inout) :: k
  end subroutine
  subroutine mpp_max_i(c, k)
    character(len=*), intent(in) :: c
    integer, intent(inout) :: k
  end subroutine
  subroutine ctl_nam(ios, cd, ld)
    integer, intent(inout) :: ios
    character(len=*), intent(in) :: cd
    logical, intent(in), optional :: ld
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

module lbclnk
  use par_kind
  implicit none
  interface lbc_lnk
    module procedure lbc_lnk_2d, lbc_lnk_2d2, lbc_lnk_3d2
  end interface
contains
  subroutine lbc_lnk_2d(c, a, t, s, ldfull)
    character(len=*) :: c, t
    real(wp) :: a(:,:), s
    logical, optional :: ldfull
  end subroutine
  subroutine lbc_lnk_2d2(c, a, t, s, b, t2, s2, ldfull)
    character(len=*) :: c, t, t2
    real(wp) :: a(:,:), b(:,:), s, s2
    logical, optional :: ldfull
  end subroutine
  subroutine lbc_lnk_3d2(c, a, t, s, b, t2, s2, ldfull)
    character(len=*) :: c, t, t2
    real(wp) :: a(:,:,:), b(:,:,:), s, s2
    logical, optional :: ldfull
  end subroutine
end module lbclnk

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
