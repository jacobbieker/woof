! Stand-ins for the E3SM v3.2.0 coupler modules one coupler interval reaches
! that need MCT (after mct_mod and seq_map_type_mod; chain_standins_base.F90
! holds the rest), with no arithmetic: component_type_mod's components (the
! domain, the segment map and the two attribute vectors each exchanges),
! seq_nlmap_mod (no nonlinear map here), shr_mct_mod's map-file readers (not
! reached) and the mapper getters seq_frac_mct names from the components
! this interval does not have (prep_atm_mod holds the ice-to-atmosphere
! mapper the driver builds). prep_ocn_mod and prep_ice_mod are E3SM's own,
! cut verbatim.

module component_type_mod
  use mct_mod, only: mct_aVect, mct_gGrid, mct_gsMap
  implicit none
  private
  public :: component_type, component_get_dom_cx, component_get_gsmap_cx, component_get_c2x_cx
  public :: component_get_x2c_cx, ocn, atm, ice, rof, wav, glc
  type component_type
    type(mct_gGrid), pointer :: dom_cx => null()
    type(mct_gsMap), pointer :: gsmap_cx => null()
    type(mct_aVect), pointer :: c2x_cx => null()
    type(mct_aVect), pointer :: x2c_cx => null()
  end type component_type
  type(component_type), target :: ocn(1), atm(1), ice(1), rof(1), wav(1), glc(1)
contains
  function component_get_dom_cx(comp) result(p)
    type(component_type), intent(in) :: comp
    type(mct_gGrid), pointer :: p
    p => comp % dom_cx
  end function component_get_dom_cx

  function component_get_gsmap_cx(comp) result(p)
    type(component_type), intent(in) :: comp
    type(mct_gsMap), pointer :: p
    p => comp % gsmap_cx
  end function component_get_gsmap_cx

  function component_get_c2x_cx(comp) result(p)
    type(component_type), intent(in) :: comp
    type(mct_aVect), pointer :: p
    p => comp % c2x_cx
  end function component_get_c2x_cx

  function component_get_x2c_cx(comp) result(p)
    type(component_type), intent(in) :: comp
    type(mct_aVect), pointer :: p
    p => comp % x2c_cx
  end function component_get_x2c_cx
end module component_type_mod

module shr_mct_mod
  implicit none
  private
  public :: shr_mct_queryConfigFile, shr_mct_sMatReaddnc, shr_mct_sMatPInitnc
contains
  subroutine shr_mct_queryConfigFile()
    error stop 9
  end subroutine shr_mct_queryConfigFile

  subroutine shr_mct_sMatReaddnc()
    error stop 9
  end subroutine shr_mct_sMatReaddnc

  subroutine shr_mct_sMatPInitnc()
    error stop 9
  end subroutine shr_mct_sMatPInitnc
end module shr_mct_mod

module seq_nlmap_mod
  use seq_map_type_mod, only: seq_map
  use mct_mod, only: mct_aVect
  implicit none
  private
  public :: seq_nlmap_avNormArr
contains
  subroutine seq_nlmap_avNormArr(mapper, av_i, av_o, lnorm, a2s_cons)
    type(seq_map), intent(inout) :: mapper
    type(mct_aVect), intent(in) :: av_i
    type(mct_aVect), intent(inout) :: av_o
    logical, intent(inout) :: lnorm
    logical, intent(in) :: a2s_cons
    error stop 8
  end subroutine seq_nlmap_avNormArr
end module seq_nlmap_mod

module prep_atm_mod
  use seq_map_type_mod, only: seq_map
  implicit none
  private
  type(seq_map), pointer, public :: mapper_Fi2a => null()
  public :: prep_atm_get_mapper_Fo2a, prep_atm_get_mapper_Fi2a, prep_atm_get_mapper_Fl2a, prep_atm_get_mapper_Fz2a
contains
  function prep_atm_get_mapper_Fi2a() result(p)
    type(seq_map), pointer :: p
    p => mapper_Fi2a
  end function prep_atm_get_mapper_Fi2a

  function prep_atm_get_mapper_Fo2a() result(p)
    type(seq_map), pointer :: p
    p => null()
    error stop 8
  end function prep_atm_get_mapper_Fo2a

  function prep_atm_get_mapper_Fl2a() result(p)
    type(seq_map), pointer :: p
    p => null()
    error stop 8
  end function prep_atm_get_mapper_Fl2a

  function prep_atm_get_mapper_Fz2a() result(p)
    type(seq_map), pointer :: p
    p => null()
    error stop 8
  end function prep_atm_get_mapper_Fz2a
end module prep_atm_mod

module prep_lnd_mod
  use seq_map_type_mod, only: seq_map
  implicit none
  private
  public :: prep_lnd_get_mapper_Fa2l, prep_lnd_get_mapper_Sz2l
contains
  function prep_lnd_get_mapper_Fa2l() result(p)
    type(seq_map), pointer :: p
    p => null()
    error stop 8
  end function prep_lnd_get_mapper_Fa2l

  function prep_lnd_get_mapper_Sz2l() result(p)
    type(seq_map), pointer :: p
    p => null()
    error stop 8
  end function prep_lnd_get_mapper_Sz2l
end module prep_lnd_mod

module prep_rof_mod
  use seq_map_type_mod, only: seq_map
  implicit none
  private
  public :: prep_rof_get_mapper_Fl2r
contains
  function prep_rof_get_mapper_Fl2r() result(p)
    type(seq_map), pointer :: p
    p => null()
    error stop 8
  end function prep_rof_get_mapper_Fl2r
end module prep_rof_mod

module prep_glc_mod
  use seq_map_type_mod, only: seq_map
  implicit none
  private
  public :: prep_glc_get_mapper_Fl2g, prep_glc_calculate_subshelf_boundary_fluxes
contains
  subroutine prep_glc_calculate_subshelf_boundary_fluxes()
    error stop 8
  end subroutine prep_glc_calculate_subshelf_boundary_fluxes

  function prep_glc_get_mapper_Fl2g() result(p)
    type(seq_map), pointer :: p
    p => null()
    error stop 8
  end function prep_glc_get_mapper_Fl2g
end module prep_glc_mod

module prep_iac_mod
  use seq_map_type_mod, only: seq_map
  implicit none
  private
  public :: prep_iac_get_mapper_Sl2z, prep_iac_get_mapper_Sa2z
contains
  function prep_iac_get_mapper_Sl2z() result(p)
    type(seq_map), pointer :: p
    p => null()
    error stop 8
  end function prep_iac_get_mapper_Sl2z

  function prep_iac_get_mapper_Sa2z() result(p)
    type(seq_map), pointer :: p
    p => null()
    error stop 8
  end function prep_iac_get_mapper_Sa2z
end module prep_iac_mod
