! Declarations-only stand-ins for the Registry-generated WRF modules that
! chem/module_radm.F USEs (radm_driver, module_radm.F:18-20).
!
! WRF generates module_state_description and module_configure from the
! Registry at build time, so they are not files of the WRF tree.  Everything
! below is an index constant with the value the Registry gives it for the
! radm2sorg package (chem_opt = 2, Registry/registry.chem:3930: the chem array
! in package order after the PARAM_FIRST_SCALAR = 2 dummy), or a service
! routine.  Nothing here computes a physical quantity.
module module_state_description
  implicit none
  integer, parameter :: param_first_scalar = 2
  integer, parameter :: p_so2 = 2, p_sulf = 3, p_no2 = 4, p_no = 5,         &
                        p_o3 = 6, p_hno3 = 7, p_h2o2 = 8, p_ald = 9,        &
                        p_hcho = 10, p_op1 = 11, p_op2 = 12, p_paa = 13,    &
                        p_ora1 = 14, p_ora2 = 15, p_nh3 = 16, p_n2o5 = 17,  &
                        p_no3 = 18, p_pan = 19, p_hc3 = 20, p_hc5 = 21,     &
                        p_hc8 = 22, p_eth = 23, p_co = 24, p_ol2 = 25,      &
                        p_olt = 26, p_oli = 27, p_tol = 28, p_xyl = 29,     &
                        p_aco3 = 30, p_tpan = 31, p_hono = 32, p_hno4 = 33, &
                        p_ket = 34, p_gly = 35, p_mgly = 36, p_dcb = 37,    &
                        p_onit = 38, p_csl = 39, p_iso = 40, p_hcl = 41,    &
                        p_ho = 42, p_ho2 = 43
  ! The package's aerosol rows follow (so4aj ... corn); radm_driver reads only
  ! p_nu0 > 1 (module_radm.F:95) and clamps the whole array (module_radm.F:98).
  ! num_chem keeps the 43 gas-phase rows plus nu0 as the one aerosol row.
  integer, parameter :: p_nu0 = 44, num_chem = 44
  integer, parameter :: p_qv = 2, p_qc = 3, num_moist = 3
end module module_state_description

module module_configure
  use module_state_description
  implicit none
  type grid_config_rec_type
    integer :: chem_opt = 2
  end type grid_config_rec_type
end module module_configure

subroutine wrf_debug(level, str)
  integer, intent(in) :: level
  character(len=*), intent(in) :: str
end subroutine wrf_debug

subroutine wrf_message(str)
  character(len=*), intent(in) :: str
  write(0, '(A)') trim(str)
end subroutine wrf_message

subroutine wrf_error_fatal(str)
  character(len=*), intent(in) :: str
  write(0, '(A)') trim(str)
  error stop 'wrf_error_fatal'
end subroutine wrf_error_fatal
