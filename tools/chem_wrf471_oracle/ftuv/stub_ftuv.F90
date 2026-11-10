! Declarations-only stand-ins for the Registry-generated WRF modules that
! chem/module_ftuv_driver.F USEs, for the radm2 (chem_opt = 1) and radm2sorg
! (chem_opt = 2) packages with phot_opt = 3 (FTUV).  The species indices are
! radm2sorg's (Registry/registry.chem, package order after the
! PARAM_FIRST_SCALAR = 2 dummy, as tools/chem_wrf471_oracle/sorgam's stub);
! species absent from the package carry index 1, as WRF gives them.  The
! package constants are Registry/registry.chem's chem_opt values.  Nothing
! here computes a physical quantity.
module module_state_description
  implicit none
  integer, parameter :: param_first_scalar = 2
  integer, parameter :: radm2 = 1, radm2sorg = 2, racmsorg_aq = 12,          &
                        cbmzsorg = 30, cbmzsorg_aq = 35,                     &
                        radm2sorg_aqchem = 41, racmsorg_aqchem_kpp = 42,     &
                        racm_esrlsorg_aqchem_kpp = 43, racmsorg_kpp = 105,   &
                        radm2sorg_kpp = 106, racm_esrlsorg_kpp = 107,        &
                        mozart_kpp = 111, mozcart_kpp = 112,                 &
                        t1_mozcart_kpp = 114, cb05_sorg_aq_kpp = 131,        &
                        cb05_sorg_vbs_aq_kpp = 132,                          &
                        mozart_mosaic_4bin_kpp = 201,                        &
                        mozart_mosaic_4bin_aq_kpp = 202
  integer, parameter :: ftuv = 3
  integer, parameter :: p_sulf = 3, p_o3 = 6
  integer, parameter :: p_so4aj = 44, p_nh4aj = 46, p_no3aj = 48,           &
                        p_naaj = 50, p_claj = 52, p_orgpaj = 70,            &
                        p_ecj = 72, p_seas = 77
  integer, parameter :: num_chem = 81
  ! not in radm2sorg
  integer, parameter :: p_bc1 = 1, p_bc2 = 1, p_oc1 = 1, p_oc2 = 1,         &
                        p_seas_1 = 1, p_seas_2 = 1, p_seas_3 = 1,           &
                        p_seas_4 = 1, p_asoa1j = 1, p_asoa2j = 1,           &
                        p_asoa3j = 1, p_asoa4j = 1, p_bsoa1j = 1,           &
                        p_bsoa2j = 1, p_bsoa3j = 1, p_bsoa4j = 1
  integer, parameter :: p_qv = 2, p_qc = 3, num_moist = 3
end module module_state_description

module module_configure
  use module_state_description
  implicit none
  type grid_config_rec_type
    integer :: chem_opt = 2
    integer :: phot_opt = 3
    integer :: aer_ra_feedback = 0
    integer :: num_land_cat = 24
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

subroutine wrf_abort()
  write(0, '(A)') 'wrf_abort'
  stop 1
end subroutine wrf_abort
