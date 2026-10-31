! Services for the use_theta_m = 1 / do_avgflx_em = 1 oracle
! (tools/bigstep_wrf471_oracle/thetam_build.sh).
!
! module_configure and module_state_description are the two modules WRF's
! Registry GENERATES; this oracle has no WRF build tree to take them from, so
! they are written here with exactly the names the extracted routines USE,
! with the types WRF's generated modules give them (Registry.EM_COMMON
! rconfig types; moist scalar indices are INTEGER variables the package
! setup assigns, PARAM_FIRST_SCALAR is the PARAMETER 2).  The driver
! assigns every value it reads.  The scheme ids are the Registry package
! values (Registry.EM_COMMON:3200-3212).  Logging services are stubs.
module module_configure
  implicit none
  type grid_config_rec_type
    integer :: use_theta_m = 1
    integer :: no_mp_heating = 0
    real :: mp_tend_lim = 10.
    integer :: cu_physics = 0
    integer :: bl_pbl_physics = 0
    integer :: shcu_physics = 0
    integer :: ra_lw_physics = 0
    integer :: ra_sw_physics = 0
    integer :: grid_fdda = 0
  end type grid_config_rec_type
end module module_configure

module module_state_description
  implicit none
  integer, parameter :: param_first_scalar = 2
  integer :: p_qv = 2, p_qc = 3, p_qr = 4, p_qi = 1, p_qs = 1, p_qg = 1
  integer, parameter :: kfetascheme = 1, gfscheme = 3, g3scheme = 5, &
       tiedtkescheme = 6, mskfscheme = 11, ntiedtkescheme = 16, &
       gdscheme = 93, dfi_fwd = 1
end module module_state_description

module module_wrf_error
  implicit none
  character(len=512) :: wrf_err_message
contains
  subroutine wrf_debug(level, message)
    integer, intent(in) :: level
    character(len=*), intent(in) :: message
  end subroutine
  subroutine wrf_message(message)
    character(len=*), intent(in) :: message
  end subroutine
  subroutine wrf_error_fatal(message)
    character(len=*), intent(in) :: message
    error stop 'WRF fatal error'
  end subroutine
end module module_wrf_error

! module_avgflx_em USEs module_bc for nothing it calls.
module module_bc
end module module_bc
