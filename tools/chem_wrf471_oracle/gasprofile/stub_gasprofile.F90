! Declarations-only stand-ins for the WRF modules that the specification part
! of chem/module_input_chem_data.F and the routines extracted from it USE.
!
! module_state_description: the Registry chem_opt / gas_ic_opt package values
! (Registry/registry.chem:3927-3999, 4164-4167) and the chem species indices.
! The indices are module VARIABLES, as in WRF, so the driver can switch between
! the radm2 (chem_opt = 1, registry.chem:3929) and radm2sorg (chem_opt = 2,
! :3930) packages; set_package mirrors what set_scalar_indices_from_config does
! for those two package lists: each row in package order after the
! PARAM_FIRST_SCALAR = 2 dummy, every row absent from the package at 1.
! module_domain: a TYPE(domain) carrying only the fields input_chem_profile
! reads, WRFU_Time and max_domains for the module's last_chem_time, and
! get_ijk_from_grid.  module_io_domain and module_get_file_names: only the
! names the module's USE lines import.  Nothing here computes a physical
! quantity.
module module_state_description
  implicit none
  integer, parameter :: param_first_scalar = 2
  integer, parameter :: &
      prescribe_aerosol = 0, radm2 = 1, radm2sorg = 2,  &
      cbmz = 5, cbmz_bb = 6, cbmz_mosaic_4bin = 7,  &
      cbmz_mosaic_8bin = 8, cbmz_mosaic_4bin_aq = 9, cbmz_mosaic_8bin_aq = 10,  &
      radm2sorg_aq = 11, racmsorg_aq = 12, chem_tracer = 13,  &
      chem_trace2 = 14, chem_trace_ens = 15, co2_tracer = 16,  &
      ghg_tracer = 17, cbmzsorg = 30, cbmz_mosaic_dms_4bin = 31,  &
      cbmz_mosaic_dms_4bin_aq = 32, cbmz_mosaic_dms_8bin = 33, cbmz_mosaic_dms_8bin_aq = 34,  &
      cbmzsorg_aq = 35, radm2sorg_aqchem = 41, racmsorg_aqchem_kpp = 42,  &
      racm_esrlsorg_aqchem_kpp = 43, radm2_kpp = 101, racm_mim_kpp = 102,  &
      racm_kpp = 103, racmpm_kpp = 104, racmsorg_kpp = 105,  &
      radm2sorg_kpp = 106, racm_esrlsorg_kpp = 107, racm_soa_vbs_kpp = 108,  &
      racm_soa_vbs_het_kpp = 100, racm_soa_vbs_aqchem_kpp = 109, cbm4_kpp = 110,  &
      mozart_kpp = 111, mozcart_kpp = 112, t1_mozcart_kpp = 114,  &
      cbmz_bb_kpp = 120, cb05_sorg_aq_kpp = 131, cb05_sorg_vbs_aq_kpp = 132,  &
      cbmz_mosaic_kpp = 170, saprc99_kpp = 195, saprc99_mosaic_4bin_vbs2_kpp = 198,  &
      mozart_mosaic_4bin_kpp = 201, mozart_mosaic_4bin_aq_kpp = 202, saprc99_mosaic_8bin_vbs2_aq_kpp = 203,  &
      saprc99_mosaic_8bin_vbs2_kpp = 204, nmhc9_kpp = 200, gocart_simple = 300,  &
      gocartracm_kpp = 301, gocartradm2 = 303, chem_vash = 400,  &
      dust = 401, chem_volc = 402, chem_volc_4bin = 403,  &
      cbmz_cam_mam3_noaq = 501, cbmz_cam_mam7_noaq = 502, cbmz_cam_mam3_aq = 503,  &
      cbmz_cam_mam7_aq = 504, crimech_kpp = 600, cri_mosaic_8bin_aq_kpp = 601,  &
      cri_mosaic_4bin_aq_kpp = 611
  integer, parameter :: gas_ic_default = 1, gas_ic_ghg = 16, gas_ic_pnnl = 101, &
                        gas_ic_cbm4 = 102
  integer, parameter :: gas_bc_default = 1, gas_bc_ghg = 16, gas_bc_pnnl = 101, &
                        gas_bc_cbm4 = 102
  ! radm2 / radm2sorg rows, set by set_package
  integer :: &
      p_so2 = 1, p_sulf = 1, p_no2 = 1, p_no = 1, p_o3 = 1, p_hno3 = 1,  &
      p_h2o2 = 1, p_ald = 1, p_hcho = 1, p_op1 = 1, p_op2 = 1, p_paa = 1,  &
      p_ora1 = 1, p_ora2 = 1, p_nh3 = 1, p_n2o5 = 1, p_no3 = 1, p_pan = 1,  &
      p_hc3 = 1, p_hc5 = 1, p_hc8 = 1, p_eth = 1, p_co = 1, p_ol2 = 1,  &
      p_olt = 1, p_oli = 1, p_tol = 1, p_xyl = 1, p_aco3 = 1, p_tpan = 1,  &
      p_hono = 1, p_hno4 = 1, p_ket = 1, p_gly = 1, p_mgly = 1, p_dcb = 1,  &
      p_onit = 1, p_csl = 1, p_iso = 1, p_hcl = 1, p_ho = 1, p_ho2 = 1,  &
      p_nu0 = 1, num_chem = 1
  ! rows in neither package (index 1)
  integer :: &
      p_aacd = 1, p_ald2 = 1, p_aldx = 1, p_bc1 = 1, p_bc2 = 1, p_ch4 = 1,  &
      p_co2 = 1, p_dms = 1, p_ic3h7no3 = 1, p_iole = 1, p_isopo2 = 1,  &
      p_meko2 = 1, p_msa = 1, p_mtf = 1, p_oc1 = 1, p_oc2 = 1, p_ole = 1,  &
      p_p25 = 1, p_par = 1, p_soag = 1, p_tracer_1 = 1, p_xylolo2 = 1
contains
  subroutine set_package(chem_opt)
    integer, intent(in) :: chem_opt
    p_so2 = 2; p_sulf = 3; p_no2 = 4; p_no = 5; p_o3 = 6; p_hno3 = 7
    p_h2o2 = 8; p_ald = 9; p_hcho = 10; p_op1 = 11; p_op2 = 12; p_paa = 13
    p_ora1 = 14; p_ora2 = 15; p_nh3 = 16; p_n2o5 = 17; p_no3 = 18; p_pan = 19
    p_hc3 = 20; p_hc5 = 21; p_hc8 = 22; p_eth = 23; p_co = 24; p_ol2 = 25
    p_olt = 26; p_oli = 27; p_tol = 28; p_xyl = 29; p_aco3 = 30; p_tpan = 31
    p_hono = 32; p_hno4 = 33; p_ket = 34; p_gly = 35; p_mgly = 36; p_dcb = 37
    p_onit = 38; p_csl = 39; p_iso = 40; p_hcl = 41; p_ho = 42; p_ho2 = 43
    if (chem_opt == radm2) then
      p_nu0 = 1; num_chem = 43
    else if (chem_opt == radm2sorg) then
      ! so4aj ... corn follow ho2; nu0 is the 36th of the 38 aerosol rows.
      p_nu0 = 79; num_chem = 81
    else
      error stop 'set_package: only radm2 and radm2sorg'
    end if
  end subroutine set_package
end module module_state_description

module module_domain
  use module_state_description
  implicit none
  integer, parameter :: max_domains = 21
  type WRFU_Time
    integer :: unused = 0
  end type WRFU_Time
  type domain
    integer :: chem_opt = radm2, gas_ic_opt = gas_ic_default
    integer :: dims(18)
    real, pointer :: ph_1(:, :, :) => null(), phb(:, :, :) => null()
    real, pointer :: chem(:, :, :, :) => null()
  end type domain
contains
  subroutine get_ijk_from_grid(grid, ids, ide, jds, jde, kds, kde, &
                               ims, ime, jms, jme, kms, kme, &
                               ips, ipe, jps, jpe, kps, kpe)
    type(domain), intent(in) :: grid
    integer, intent(out) :: ids, ide, jds, jde, kds, kde, &
                            ims, ime, jms, jme, kms, kme, &
                            ips, ipe, jps, jpe, kps, kpe
    ids = grid%dims(1); ide = grid%dims(2); jds = grid%dims(3)
    jde = grid%dims(4); kds = grid%dims(5); kde = grid%dims(6)
    ims = grid%dims(7); ime = grid%dims(8); jms = grid%dims(9)
    jme = grid%dims(10); kms = grid%dims(11); kme = grid%dims(12)
    ips = grid%dims(13); ipe = grid%dims(14); jps = grid%dims(15)
    jpe = grid%dims(16); kps = grid%dims(17); kpe = grid%dims(18)
  end subroutine get_ijk_from_grid
end module module_domain

module module_io_domain
end module module_io_domain

module module_get_file_names
  implicit none
  character(len=256), allocatable :: eligible_file_name(:)
  integer :: number_of_eligible_files = 0
contains
  subroutine unix_ls()
    error stop 'stub unix_ls reached'
  end subroutine unix_ls
end module module_get_file_names

subroutine wrf_debug(level, str)
  integer, intent(in) :: level
  character(len=*), intent(in) :: str
end subroutine wrf_debug

subroutine wrf_message(str)
  character(len=*), intent(in) :: str
end subroutine wrf_message

subroutine wrf_error_fatal(str)
  character(len=*), intent(in) :: str
  write(0, '(A)') trim(str)
  error stop 'wrf_error_fatal'
end subroutine wrf_error_fatal

subroutine nl_get_debug_level(id, level)
  integer, intent(in) :: id
  integer, intent(out) :: level
  level = 0
end subroutine nl_get_debug_level

subroutine set_wrf_debug_level(level)
  integer, intent(in) :: level
end subroutine set_wrf_debug_level
