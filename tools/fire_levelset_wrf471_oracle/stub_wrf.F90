! Service-only stubs for the standalone WRF v4.7.1 fire-model oracle.
!
! NOTHING here computes a physical quantity.  module_fr_fire_core.F's
! prop_ls_rk3 and reinit_ls_rk3 take a TYPE(domain) argument and USE
! module_configure for halo exchanges that exist only under DM_PARALLEL;
! this build is serial, so the types carry nothing and are never read.
! module_wrf_error and module_model_constants are compiled from the pinned
! tree.  What is left is what `nm -u` reports undefined outside libm and
! libgfortran once module_fr_fire_{util,phys,core}.F are built: the WRF
! abort, the single-rank broadcast shims read_namelist_fire would call and
! the rank queries of the debug file writers (the oracle calls none of
! them; the fuel parameters are given arrays).

module module_domain
  implicit none
  type domain
    integer :: unused = 0
  end type domain
end module module_domain

module module_configure
  implicit none
  type grid_config_rec_type
    integer :: unused = 0
  end type grid_config_rec_type
end module module_configure

subroutine wrf_abort()
  implicit none
  error stop 1
end subroutine wrf_abort

subroutine wrf_debug(level, msg)
  implicit none
  integer, intent(in) :: level
  character(len=*), intent(in) :: msg
end subroutine wrf_debug

logical function wrf_dm_on_monitor()
  implicit none
  wrf_dm_on_monitor = .true.
end function wrf_dm_on_monitor

subroutine wrf_dm_bcast_real(values, n)
  real :: values
  integer, intent(in) :: n
end subroutine wrf_dm_bcast_real

subroutine wrf_dm_bcast_integer(values, n)
  integer :: values
  integer, intent(in) :: n
end subroutine wrf_dm_bcast_integer

subroutine wrf_dm_bcast_string(value, n)
  character(len=*) :: value
  integer, intent(in) :: n
end subroutine wrf_dm_bcast_string

! read_array_2d_real / write_array_m3 name their files by rank; one rank.
subroutine wrf_get_nproc(nproc)
  implicit none
  integer, intent(out) :: nproc
  nproc = 1
end subroutine wrf_get_nproc

subroutine wrf_get_myproc(myproc)
  implicit none
  integer, intent(out) :: myproc
  myproc = 0
end subroutine wrf_get_myproc
