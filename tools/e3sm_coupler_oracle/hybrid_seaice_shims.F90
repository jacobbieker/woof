! What E3SM v3.2.0's sea-ice cap reads that MPAS-Model v8.4.2's sea ice
! names differently, for the hybrid "E3SM v3.2.0 coupler with MPAS v8.4.2
! components, not an E3SM configuration" (woof.hex.coupled.hybrid_seaice;
! run_hybrid_seaice.F90).  Renames only: no arithmetic.
!
! - hybrid_seaice_constants: v8.4.2's seaice_constants, plus the three
!   constants E3SM's seaice_constants adds, taken from v8.4.2's column
!   package (the stand-alone build's cice constants): seaiceReferenceSalinity
!   is ice_ref_salinity (4), seaiceFrazilSalinityReduction dSin0_frazil (3)
!   and seaiceFrazilIcePorosity phi_init (0.75), the values E3SM's
!   non-CCSMCOUPLED seaice_constants gives them.
! - icepack_intfc: the three Icepack functions the cap calls are v8.4.2's
!   column package's of the same arithmetic (colpkg_sea_freezing_temperature,
!   colpkg_liquidus_temperature, ice_mushy_physics' enthalpy_mush), on the
!   column package's own tfrz_option and ktherm.
module hybrid_seaice_constants
  use seaice_constants
  use ice_constants_colpkg, only: seaiceReferenceSalinity => ice_ref_salinity
  use ice_colpkg_shared, only: seaiceFrazilSalinityReduction => dSin0_frazil, &
                               seaiceFrazilIcePorosity => phi_init
  implicit none
  public
end module hybrid_seaice_constants

module icepack_intfc
  use ice_colpkg, only: icepack_sea_freezing_temperature => colpkg_sea_freezing_temperature, &
                        icepack_liquidus_temperature => colpkg_liquidus_temperature
  use ice_mushy_physics, only: icepack_enthalpy_mush => enthalpy_mush
  implicit none
  public
end module icepack_intfc
