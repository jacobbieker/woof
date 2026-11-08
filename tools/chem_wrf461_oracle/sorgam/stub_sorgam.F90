! Declarations-only stand-ins for the Registry-generated WRF modules the
! extracted SORGAM routines USE, for the radm2sorg package (chem_opt = 2,
! Registry/registry.chem:3930: chem in package order after the
! PARAM_FIRST_SCALAR = 2 dummy).  Species absent from the package carry
! index 1, as WRF gives them.  Nothing here computes a physical quantity.
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
  integer, parameter :: p_so4aj = 44, p_so4ai = 45, p_nh4aj = 46,           &
                        p_nh4ai = 47, p_no3aj = 48, p_no3ai = 49,           &
                        p_naaj = 50, p_naai = 51, p_claj = 52, p_clai = 53, &
                        p_orgaro1j = 54, p_orgaro1i = 55, p_orgaro2j = 56,  &
                        p_orgaro2i = 57, p_orgalk1j = 58, p_orgalk1i = 59,  &
                        p_orgole1j = 60, p_orgole1i = 61, p_orgba1j = 62,   &
                        p_orgba1i = 63, p_orgba2j = 64, p_orgba2i = 65,     &
                        p_orgba3j = 66, p_orgba3i = 67, p_orgba4j = 68,     &
                        p_orgba4i = 69, p_orgpaj = 70, p_orgpai = 71,       &
                        p_ecj = 72, p_eci = 73, p_p25j = 74, p_p25i = 75,   &
                        p_antha = 76, p_seas = 77, p_soila = 78, p_nu0 = 79,&
                        p_ac0 = 80, p_corn = 81
  integer, parameter :: num_chem = 81
  ! not in radm2sorg
  integer, parameter :: p_lim = 1, p_so4cwj = 1, p_p25cwi = 1, p_anthcw = 1, &
                        p_soilcw = 1, p_seascw = 1
  integer, parameter :: p_qv = 2, p_qc = 3, num_moist = 3
  ! aer_ic_opt package values (Registry/registry.chem:4170-4171)
  integer, parameter :: aer_ic_default = 1, aer_ic_pnnl = 101
end module module_state_description

module module_configure
  use module_state_description
  implicit none
  type grid_config_rec_type
    integer :: chem_opt = 2
    logical :: restart = .false.
    integer :: aer_drydep_opt = 1     ! Registry/registry.chem:3836 default
    integer :: diagnostic_dep = 0     ! Registry/registry.chem:3866 default
  end type grid_config_rec_type
end module module_configure

! aerosols_sorgam_init calls this only past its chem_in_opt == 1 return,
! which the oracle never passes.
subroutine sorgam_init_aer_ic_pnnl()
  error stop 'stub sorgam_init_aer_ic_pnnl reached'
end subroutine sorgam_init_aer_ic_pnnl

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

! aeroproc's isorop = 1 branch (ISORROPIA); the oracle builds the isorop = 0
! branch (build.sh), so this is never reached.
subroutine eql3()
  error stop 'stub eql3 reached: the ISORROPIA branch is not part of this oracle'
end subroutine eql3
