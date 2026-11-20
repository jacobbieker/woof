! Stand-ins for the E3SM v3.2.0 modules the atmosphere-ocean flux reaches
! that need MCT (after mct_mod; flux_standins_base.F90 has the rest).
! component_type_mod holds the flux grid's domain and segment map;
! shr_mct_mod names the map-file readers seq_flux_mct's use statement
! imports and only the exchange grid's init (not cut) calls.  No arithmetic.

module component_type_mod
  use mct_mod, only: mct_gGrid, mct_gsMap
  implicit none
  private
  public :: component_type, component_get_dom_cx, component_get_gsmap_cx
  type component_type
    type(mct_gGrid), pointer :: dom_cx => null()
    type(mct_gsMap), pointer :: gsmap_cx => null()
  end type component_type
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
end module component_type_mod

module shr_mct_mod
  implicit none
  private
  public :: shr_mct_queryConfigFile, shr_mct_sMatReaddnc
contains
  subroutine shr_mct_queryConfigFile()
    error stop 9
  end subroutine shr_mct_queryConfigFile

  subroutine shr_mct_sMatReaddnc()
    error stop 9
  end subroutine shr_mct_sMatReaddnc
end module shr_mct_mod
