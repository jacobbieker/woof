! Service modules for compiling NEMO 5.0.2's src/OCE/TRA/trazdf.F90 whole,
! outside NEMO: NEMO's module names carrying only the names trazdf.F90 USEs
! (the tracer and grid arrays, avt/avs, the switches the driver sets) and
! services with no arithmetic.  The switches select the configuration the
! port takes: no iso-neutral slopes (l_ldfslp = F), no adaptive-implicit
! vertical advection, no mass-flux convection, no double diffusion, no
! trend diagnostics.
module oce
  use par_kind
  implicit none
  real(wp), allocatable :: wi(:,:,:)
end module oce

module dom_oce
  use par_kind
  implicit none
  integer :: jpi, jpj, jpk, jpkm1, nn_hls = 1
  integer :: ntsi, ntei, ntsj, ntej, Nis0, Nie0, Njs0, Nje0
  integer :: ntile = 0, nijtile = 1, nit000 = 1
  integer, parameter :: jpts = 2, jp_tem = 1, jp_sal = 2, jpt = 3
  logical :: l_istiled = .false.
  real(wp) :: rDt
  real(wp), allocatable, dimension(:,:,:) :: tmask, e3t_3d, e3w_3d
end module dom_oce

module phycst
end module phycst

module sbc_oce
end module sbc_oce

module zdf_oce
  use par_kind
  implicit none
  real(wp), allocatable :: avt(:,:,:), avs(:,:,:)
  logical :: ln_zdfddm = .false., ln_zad_Aimp = .false.
end module zdf_oce

module zdfmfc
  use par_kind
  implicit none
  logical :: ln_zdfmfc = .false.
  real(wp), allocatable :: edmfm(:,:,:), edmftra(:,:,:,:)
end module zdfmfc

module ldftra
  implicit none
  logical :: l_ldfslp = .false., ln_traldf_msc = .false.
end module ldftra

module ldfslp
  use par_kind
  implicit none
  real(wp), allocatable :: akz(:,:,:), ah_wslp2(:,:,:)
end module ldfslp

module trd_oce
  implicit none
  logical :: l_trdtra = .false.
  integer :: jptra_zdf = 0
end module trd_oce

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

module eosbn2
  use par_kind
  implicit none
  logical :: ln_SEOS = .false.
  real(wp) :: rn_b0 = 7.6554e-1_wp
end module eosbn2

module in_out_manager
  implicit none
  integer :: numout = 6
  logical :: lwp = .false., ln_timing = .false.
  type ctl_t
    logical :: l_prtctl = .false.
  end type
  type(ctl_t) :: sn_cfctl
end module in_out_manager

module prtctl
  implicit none
contains
  subroutine prt_ctl(tab3d_1, clinfo1, mask1, tab3d_2, clinfo2, mask2, clinfo3)
    use par_kind
    real(wp), optional :: tab3d_1(:,:,:), tab3d_2(:,:,:), mask1(:,:,:), mask2(:,:,:)
    character(len=*), optional :: clinfo1, clinfo2, clinfo3
  end subroutine
end module prtctl

module lbclnk
end module lbclnk

module lib_mpp
end module lib_mpp

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
