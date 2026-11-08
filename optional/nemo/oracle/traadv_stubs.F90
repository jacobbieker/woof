! Service modules for compiling NEMO 5.0.2's src/OCE/TRA/traadv_fct.F90 whole,
! outside NEMO.  Each module carries, under NEMO's module name, only the
! names traadv_fct.F90 USEs: the grid arrays and switches the driver sets,
! and trend, diagnostic, I/O and MPP services with no arithmetic.  The one
! service that changes numbers is lbc_lnk: it does what NEMO's
! lbc_lnk_pt2pt does on one process with closed (non-periodic) boundaries,
! fill the whole nn_hls halo with the constant zland = 0 (jpfillcst).
module oce
  use par_kind
  implicit none
  real(wp), allocatable :: wi(:,:,:)
end module oce

module dom_oce
  use par_kind
  implicit none
  integer :: jpi, jpj, jpk, jpkm1, nn_hls = 2
  integer :: ntsi, ntei, ntsj, ntej, Nis0, Nie0, Njs0, Nje0
  integer :: ntile = 0, nijtile = 1
  integer, parameter :: jpt = 3
  logical :: l_istiled = .false., ln_isfcav = .false.
  logical, parameter :: lk_linssh = .true.
  real(wp), allocatable, dimension(:,:,:) :: tmask, umask, vmask, wmask, e3t_3d
  real(wp), allocatable :: e1e2t(:,:), r1_e1e2t(:,:)
  integer, allocatable :: mbkt(:,:), mikt(:,:)
end module dom_oce

module trc_oce
end module trc_oce

module trd_oce
  implicit none
  logical :: l_trdtra = .false., l_trdtrc = .false.
  integer :: jptra_xad = 0, jptra_yad = 0, jptra_zad = 0
end module trd_oce

module trdtra
  use par_kind
  implicit none
contains
  subroutine trd_tra(kt, Kmm, Krhs, c, jn, jt, t, pu, ptn)
    integer :: kt, Kmm, Krhs, jn, jt
    character(len=*) :: c
    real(wp) :: t(:,:,:)
    real(wp), optional :: pu(:,:,:), ptn(:,:,:)
    error stop 'trd_tra: trend diagnostics are outside the harness'
  end subroutine
end module trdtra

module diaptr
  use par_kind
  implicit none
  logical :: l_diaptr = .false.
contains
  subroutine dia_ptr_hst(jn, c, p)
    integer :: jn
    character(len=*) :: c
    real(wp) :: p(:,:,:)
    error stop 'dia_ptr_hst: transport diagnostics are outside the harness'
  end subroutine
end module diaptr

module diaar5
  use par_kind
  implicit none
  logical :: l_diaar5 = .false.
contains
  subroutine dia_ar5_hst(jn, c, px, py)
    integer :: jn
    character(len=*) :: c
    real(wp) :: px(:,:,:), py(:,:,:)
    error stop 'dia_ar5_hst: transport diagnostics are outside the harness'
  end subroutine
end module diaar5

module phycst
  use par_kind
  implicit none
  real(wp) :: rho0_rcp = 1026._wp * 3991.86795711963_wp
end module phycst

module zdf_oce
  implicit none
  logical :: ln_zad_Aimp = .false.
end module zdf_oce

module in_out_manager
  implicit none
  integer :: numout = 6
  logical :: lwp = .false., ln_timing = .false.
end module in_out_manager

module iom
  implicit none
contains
  logical function iom_use(c)
    character(len=*), intent(in) :: c
    iom_use = .false.
  end function
end module iom

module lib_mpp
  implicit none
contains
  subroutine ctl_stop(c1, c2, c3, c4, c5)
    character(len=*), intent(in), optional :: c1, c2, c3, c4, c5
    error stop 'ctl_stop'
  end subroutine
end module lib_mpp

module lib_fortran
end module lib_fortran

module lbclnk
  use par_kind
  use dom_oce, only : nn_hls
  implicit none
  interface lbc_lnk
    module procedure lbc_lnk_3d3, lbc_lnk_3d4
  end interface
contains
  subroutine zero_halo(a)
    real(wp), intent(inout) :: a(:,:,:)
    integer :: ni, nj
    ni = size(a, 1); nj = size(a, 2)
    a(1:nn_hls,:,:) = 0._wp
    a(ni-nn_hls+1:ni,:,:) = 0._wp
    a(:,1:nn_hls,:) = 0._wp
    a(:,nj-nn_hls+1:nj,:) = 0._wp
  end subroutine
  subroutine lbc_lnk_3d3(c, a, t1, s1, b, t2, s2, d, t3, s3)
    character(len=*) :: c, t1, t2, t3
    real(wp) :: a(:,:,:), b(:,:,:), d(:,:,:), s1, s2, s3
    call zero_halo(a); call zero_halo(b); call zero_halo(d)
  end subroutine
  subroutine lbc_lnk_3d4(c, a, t1, s1, b, t2, s2, d, t3, s3, e, t4, s4)
    character(len=*) :: c, t1, t2, t3, t4
    real(wp) :: a(:,:,:), b(:,:,:), d(:,:,:), e(:,:,:), s1, s2, s3, s4
    call zero_halo(a); call zero_halo(b); call zero_halo(d); call zero_halo(e)
  end subroutine
end module lbclnk
