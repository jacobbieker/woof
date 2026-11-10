! Declarations-only stand-ins for the Registry-generated WRF modules that
! chem/module_emissions_anthropogenics.F and the extracted SORGAM routines
! USE, for the radm2sorg chem package (chem_opt = 2, Registry/registry.chem:
! 3930) and the eradmsorg emission package (emiss_opt = 3, :4055), each array
! in package order after the PARAM_FIRST_SCALAR = 2 dummy.  Species absent
! from a package carry index 1, as WRF gives them; package selectors carry
! their Registry values.  Nothing here computes a physical quantity.
module module_state_description
  implicit none
  integer, parameter :: param_first_scalar = 2
  ! radm2sorg chem rows
  integer, parameter :: &
      p_so2 = 2, p_sulf = 3, p_no2 = 4, p_no = 5,  &
      p_o3 = 6, p_hno3 = 7, p_h2o2 = 8, p_ald = 9,  &
      p_hcho = 10, p_op1 = 11, p_op2 = 12, p_paa = 13,  &
      p_ora1 = 14, p_ora2 = 15, p_nh3 = 16, p_n2o5 = 17,  &
      p_no3 = 18, p_pan = 19, p_hc3 = 20, p_hc5 = 21,  &
      p_hc8 = 22, p_eth = 23, p_co = 24, p_ol2 = 25,  &
      p_olt = 26, p_oli = 27, p_tol = 28, p_xyl = 29,  &
      p_aco3 = 30, p_tpan = 31, p_hono = 32, p_hno4 = 33,  &
      p_ket = 34, p_gly = 35, p_mgly = 36, p_dcb = 37,  &
      p_onit = 38, p_csl = 39, p_iso = 40, p_hcl = 41,  &
      p_ho = 42, p_ho2 = 43, p_so4aj = 44, p_so4ai = 45,  &
      p_nh4aj = 46, p_nh4ai = 47, p_no3aj = 48, p_no3ai = 49,  &
      p_naaj = 50, p_naai = 51, p_claj = 52, p_clai = 53,  &
      p_orgaro1j = 54, p_orgaro1i = 55, p_orgaro2j = 56, p_orgaro2i = 57,  &
      p_orgalk1j = 58, p_orgalk1i = 59, p_orgole1j = 60, p_orgole1i = 61,  &
      p_orgba1j = 62, p_orgba1i = 63, p_orgba2j = 64, p_orgba2i = 65,  &
      p_orgba3j = 66, p_orgba3i = 67, p_orgba4j = 68, p_orgba4i = 69,  &
      p_orgpaj = 70, p_orgpai = 71, p_ecj = 72, p_eci = 73,  &
      p_p25j = 74, p_p25i = 75, p_antha = 76, p_seas = 77,  &
      p_soila = 78, p_nu0 = 79, p_ac0 = 80, p_corn = 81
  integer, parameter :: num_chem = 81
  ! eradmsorg emis_ant rows
  integer, parameter :: &
      p_e_iso = 2, p_e_so2 = 3, p_e_no = 4, p_e_no2 = 5,  &
      p_e_co = 6, p_e_ch4 = 7, p_e_eth = 8, p_e_hc3 = 9,  &
      p_e_hc5 = 10, p_e_hc8 = 11, p_e_xyl = 12, p_e_ol2 = 13,  &
      p_e_olt = 14, p_e_oli = 15, p_e_tol = 16, p_e_csl = 17,  &
      p_e_hcho = 18, p_e_ald = 19, p_e_ket = 20, p_e_ora2 = 21,  &
      p_e_nh3 = 22, p_e_terp = 23, p_e_hono = 24, p_e_co2 = 25,  &
      p_e_pm25i = 26, p_e_pm25j = 27, p_e_pm_10 = 28, p_e_eci = 29,  &
      p_e_ecj = 30, p_e_orgi = 31, p_e_orgj = 32, p_e_so4i = 33,  &
      p_e_so4j = 34, p_e_no3i = 35, p_e_no3j = 36, p_e_naaj = 37,  &
      p_e_naai = 38, p_e_orgi_a = 39, p_e_orgj_a = 40, p_e_orgi_bb = 41,  &
      p_e_orgj_bb = 42, p_e_hcl = 43, p_e_cli = 44, p_e_clj = 45,  &
      p_e_ch3cl = 46
  integer, parameter :: num_emis_ant = 46
  ! not in these packages
  integer, parameter :: &
      p_acet = 1, p_api = 1, p_apin = 1, p_apinene = 1,  &
      p_bc1 = 1, p_benzene = 1, p_bigalk = 1, p_bigene = 1,  &
      p_bpinene = 1, p_c10h16 = 1, p_c2h2 = 1, p_c2h4 = 1,  &
      p_c2h5cho = 1, p_c2h5oh = 1, p_c2h6 = 1, p_c3h6 = 1,  &
      p_c3h8 = 1, p_c5h8 = 1, p_ch3cho = 1, p_ch3cl = 1,  &
      p_ch3co2h = 1, p_ch3oh = 1, p_ch4 = 1, p_co2 = 1,  &
      p_dime35eb = 1, p_e_apin = 1, p_e_bc = 1, p_e_benzene = 1,  &
      p_e_bigalk = 1, p_e_bigene = 1, p_e_c10h16 = 1, p_e_c2h2 = 1,  &
      p_e_c2h4 = 1, p_e_c2h5cho = 1, p_e_c2h5oh = 1, p_e_c2h6 = 1,  &
      p_e_c3h6 = 1, p_e_c3h8 = 1, p_e_c5h8 = 1, p_e_ch2o = 1,  &
      p_e_ch3cho = 1, p_e_ch3co2h = 1, p_e_ch3coch3 = 1, p_e_ch3oh = 1,  &
      p_e_co_a = 1, p_e_co_bb = 1, p_e_dime35eb = 1, p_e_gly = 1,  &
      p_e_hcooh = 1, p_e_isop = 1, p_e_macr = 1, p_e_mek = 1,  &
      p_e_methtol = 1, p_e_mgly = 1, p_e_mvk = 1, p_e_nc4h10 = 1,  &
      p_e_oc = 1, p_e_oethtol = 1, p_e_oxyl = 1, p_e_pethtol = 1,  &
      p_e_pm_25 = 1, p_e_sulf = 1, p_e_tbut2ene = 1, p_e_tm123b = 1,  &
      p_e_tm124b = 1, p_e_tm135b = 1, p_e_toluene = 1, p_e_xylene = 1,  &
      p_eac_ch4 = 1, p_eac_co = 1, p_eac_no = 1, p_eac_so2 = 1,  &
      p_ebu_bc = 1, p_ebu_oc = 1, p_ebu_pm10 = 1, p_ebu_pm25 = 1,  &
      p_ete = 1, p_hcooh = 1, p_isopr = 1, p_lim = 1,  &
      p_macr = 1, p_mbo = 1, p_mek = 1, p_methtol = 1,  &
      p_mvk = 1, p_nc4h10 = 1, p_oc1 = 1, p_oethtol = 1,  &
      p_oxyl = 1, p_p10 = 1, p_p25 = 1, p_par = 1,  &
      p_pethtol = 1, p_sesq = 1, p_tbut2ene = 1, p_tm123b = 1,  &
      p_tm124b = 1, p_tm135b = 1, p_toluene = 1, p_voca = 1,  &
      p_vocbb = 1, p_xylenes = 1
  integer, parameter :: num_emis_aircraft = 1, num_ebu = 1
  ! package selectors (Registry/registry.chem)
  integer, parameter :: radm2 = 1, radm2sorg = 2, mozart_kpp = 111,         &
                        mozcart_kpp = 112, t1_mozcart_kpp = 114,            &
                        mozart_mosaic_4bin_kpp = 201,                       &
                        mozart_mosaic_4bin_aq_kpp = 202, gocartracm_kpp = 301, &
                        gocartradm2 = 303, crimech_kpp = 600,               &
                        cri_mosaic_8bin_aq_kpp = 601,                       &
                        cri_mosaic_4bin_aq_kpp = 611
  integer, parameter :: eradmsorg = 3, mozem = 7, mozcem = 8, mozmem = 10,  &
                        mozc_t1_em = 11
  integer, parameter :: gunther1 = 1, beis314 = 2, megan2 = 3, megan2_clm = 4
  integer, parameter :: p_qv = 2, p_qc = 3, num_moist = 3
  integer, parameter :: aer_ic_default = 1, aer_ic_pnnl = 101
end module module_state_description

module module_configure
  use module_state_description
  implicit none
  type grid_config_rec_type
    integer :: chem_opt = 2
    integer :: emiss_opt = 3
    integer :: kemit = 9               ! Registry/registry.chem:3780 default
    integer :: kemit_aircraft = 1      ! Registry/registry.chem:3784 default
    integer :: aircraft_emiss_opt = 0  ! Registry/registry.chem:3869 default
    integer :: bio_emiss_opt = 0       ! Registry/registry.chem:3852 default
    logical :: restart = .false.
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

! Branches of the extracted routines this oracle never takes (sea salt and
! dust inside sorgam_addemiss, the PNNL aerosol ICs past chem_in_opt = 1).
subroutine sorgam_seasalt_emiss()
  error stop 'stub sorgam_seasalt_emiss reached'
end subroutine sorgam_seasalt_emiss
subroutine sorgam_dust_emiss()
  error stop 'stub sorgam_dust_emiss reached'
end subroutine sorgam_dust_emiss
subroutine sorgam_dust_gocartemis()
  error stop 'stub sorgam_dust_gocartemis reached'
end subroutine sorgam_dust_gocartemis
subroutine sorgam_init_aer_ic_pnnl()
  error stop 'stub sorgam_init_aer_ic_pnnl reached'
end subroutine sorgam_init_aer_ic_pnnl
