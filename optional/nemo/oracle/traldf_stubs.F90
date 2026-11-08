! Service modules for compiling NEMO 5.0.2's src/OCE/TRA/traldf_lev.F90 whole,
! outside NEMO: the grid arrays and eddy diffusivities the driver sets, and
! diagnostic services with no arithmetic, under NEMO's module names.
module oce
end module oce

module dom_oce
  use par_kind
  implicit none
  integer :: jpi, jpj, jpk, jpkm1, nn_hls = 2
  integer :: ntsi, ntei, ntsj, ntej, Nis0, Nie0, Njs0, Nje0
  real(wp), allocatable, dimension(:,:,:) :: tmask, umask, vmask, e3t_3d, e3u_3d, e3v_3d
  real(wp), allocatable :: e2_e1u(:,:), e1_e2v(:,:), r1_e1e2t(:,:)
end module dom_oce

module domutl
  implicit none
contains
  logical function is_tile(pt)
    use par_kind
    real(wp), intent(in) :: pt(:,:,:)
    is_tile = .false.
  end function
end module domutl

module ldftra
  use par_kind
  implicit none
  real(wp), allocatable :: ahtu(:,:,:), ahtv(:,:,:)
end module ldftra

module diaptr
  use par_kind
  implicit none
  logical :: l_diaptr = .false.
contains
  subroutine dia_ptr_hst(jn, c, p)
    integer :: jn
    character(len=*) :: c
    real(wp) :: p(:,:)
    error stop 'dia_ptr_hst: transport diagnostics are outside the harness'
  end subroutine
end module diaptr

module diaar5
  use par_kind
  implicit none
contains
  subroutine dia_ar5_hst(jn, c, px, py, ldfin)
    integer :: jn
    character(len=*) :: c
    real(wp) :: px(:,:), py(:,:)
    logical, optional :: ldfin
    error stop 'dia_ar5_hst: transport diagnostics are outside the harness'
  end subroutine
end module diaar5

module in_out_manager
  implicit none
  integer :: numout = 6
  logical :: lwp = .false.
end module in_out_manager
