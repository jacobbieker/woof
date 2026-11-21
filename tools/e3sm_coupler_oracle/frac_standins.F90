! Stand-ins for the E3SM v3.2.0 coupler modules seq_frac_set and
! seq_map_map reach (driver-mct/main/seq_frac_mct.F90, seq_map_mod.F90).
!
! None of them does arithmetic.  shr_sys_abort stops the oracle (status 7)
! with its message; the coupler's log is the standard error, so
! seq_frac_check's lines reach the fixture; the MPI reductions
! seq_frac_check makes for its log are a single task's value (the oracle is
! one task); seq_infodata_mod holds the three switches seq_frac_set reads;
! component_type_mod the sea ice's domain and i2x pointers; the prep_*_mod
! modules the two mappers seq_frac_set takes (the driver builds them) and
! the getters seq_frac_mct names in its use statements and never calls
! here; seq_nlmap_mod the nonlinear map seq_map_avNormArr calls only when a
! mapper has one (none here).
!
! frac_standins_base.F90 is everything that does not need MCT;
! frac_standins.F90 (this file, after mct_mod and seq_map_type_mod) the rest.

module component_type_mod
  use mct_mod, only: mct_aVect, mct_gGrid
  implicit none
  private
  public :: component_type, component_get_dom_cx, component_get_c2x_cx
  type component_type
    type(mct_gGrid), pointer :: dom_cx => null()
    type(mct_aVect), pointer :: c2x_cx => null()
  end type component_type
contains
  function component_get_dom_cx(comp) result(p)
    type(component_type), intent(in) :: comp
    type(mct_gGrid), pointer :: p
    p => comp % dom_cx
  end function component_get_dom_cx

  function component_get_c2x_cx(comp) result(p)
    type(component_type), intent(in) :: comp
    type(mct_aVect), pointer :: p
    p => comp % c2x_cx
  end function component_get_c2x_cx
end module component_type_mod

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

module prep_ocn_mod
  use seq_map_type_mod, only: seq_map
  implicit none
  private
  type(seq_map), pointer, public :: mapper_SFi2o => null()
  public :: prep_ocn_get_mapper_Fa2o, prep_ocn_get_mapper_SFi2o
contains
  function prep_ocn_get_mapper_Fa2o() result(p)
    type(seq_map), pointer :: p
    p => null()
    error stop 8
  end function prep_ocn_get_mapper_Fa2o

  function prep_ocn_get_mapper_SFi2o() result(p)
    type(seq_map), pointer :: p
    p => mapper_SFi2o
  end function prep_ocn_get_mapper_SFi2o
end module prep_ocn_mod

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

module prep_ice_mod
  use seq_map_type_mod, only: seq_map
  implicit none
  private
  public :: prep_ice_get_mapper_SFo2i
contains
  function prep_ice_get_mapper_SFo2i() result(p)
    type(seq_map), pointer :: p
    p => null()
    error stop 8
  end function prep_ice_get_mapper_SFo2i
end module prep_ice_mod

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
  public :: prep_glc_get_mapper_Fl2g
contains
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
