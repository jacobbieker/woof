! Stand-ins for the E3SM v3.2.0 MPAS-Ocean modules the ocean's coupler import
! and export (ocn_comp_mct.F's ocn_import_mct and ocn_export_mct) reach.
!
! None of them does arithmetic.  In E3SM, ocn_config is generated from
! Registry.xml (a pointer per namelist option, pointed at the config pool by
! ocn_config_init) and ocn_diagnostics_variables points its land-ice arrays
! and indices at the forcing pool's.  Here they are the variables those two
! modules declare, set by the driver from the stream: the namelist options
! the two routines and ocn_freezing_temperature read at module scope, and the
! land-ice boundary-layer arrays and indices the export reads.

module ocn_config
  use mpas_kind_types
  implicit none
  public
  logical :: config_momentum_use_active_wave = .false.
  logical :: config_use_ecosysTracers_river_inputs_from_coupler = .false.
  logical :: config_use_ecosysTracers_iron_solubility_file = .false.
  logical :: config_use_ecosysTracers_atm_dust_from_coupler = .false.
  real(kind=RKIND) :: config_frazil_heat_of_fusion = 3.34e5_RKIND
  real(kind=RKIND) :: config_land_ice_cavity_freezing_temperature_coeff_0 = 0.0_RKIND
  real(kind=RKIND) :: config_land_ice_cavity_freezing_temperature_coeff_S = 0.0_RKIND
  real(kind=RKIND) :: config_land_ice_cavity_freezing_temperature_coeff_p = 0.0_RKIND
  real(kind=RKIND) :: config_land_ice_cavity_freezing_temperature_coeff_pS = 0.0_RKIND
  real(kind=RKIND) :: config_open_ocean_freezing_temperature_coeff_0 = 0.0_RKIND
  real(kind=RKIND) :: config_open_ocean_freezing_temperature_coeff_S = 0.0_RKIND
  real(kind=RKIND) :: config_open_ocean_freezing_temperature_coeff_p = 0.0_RKIND
  real(kind=RKIND) :: config_open_ocean_freezing_temperature_coeff_pS = 0.0_RKIND
  real(kind=RKIND) :: config_open_ocean_freezing_temperature_coeff_mushy_az1_liq = -18.48_RKIND
end module ocn_config

module ocn_diagnostics_variables
  use mpas_kind_types
  implicit none
  public
  real (kind=RKIND), dimension(:,:), pointer :: landIceBoundaryLayerTracers => null()
  real (kind=RKIND), dimension(:,:), pointer :: landIceTracerTransferVelocities => null()
  integer :: indexBLT, indexBLS, &
             indexHeatTrans, indexSaltTrans
end module ocn_diagnostics_variables
