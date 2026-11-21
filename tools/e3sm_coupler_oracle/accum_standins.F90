! Stand-ins for the E3SM v3.2.0 coupler modules prep_ocn_accum and
! prep_ocn_accum_avg reach (driver-mct/main/prep_ocn_mod.F90).
!
! None of them does arithmetic: seq_comm_mct's instance counts (one ocean)
! and the coupler's id; t_drv_timers_mod's timers (nothing); and
! component_type_mod's components with the ocean's x2o on the coupler's
! processes (component_get_x2c_cx), the getters prep_ocn_mod names.
module seq_comm_mct
  implicit none
  integer, parameter :: num_inst_glc = 1, num_inst_wav = 1, num_inst_ocn = 1
  integer, parameter :: CPLID = 1
end module seq_comm_mct

module t_drv_timers_mod
  implicit none
contains
  subroutine t_drvstartf(string, cplrun, cplcom, budget, barrier, hashint)
    character(len=*), intent(in) :: string
    logical, intent(in), optional :: cplrun, cplcom, budget
    integer, intent(in), optional :: barrier
    integer, intent(inout), optional :: hashint
  end subroutine t_drvstartf

  subroutine t_drvstopf(string, cplrun, cplcom, budget, hashint)
    character(len=*), intent(in) :: string
    logical, intent(in), optional :: cplrun, cplcom, budget
    integer, intent(inout), optional :: hashint
  end subroutine t_drvstopf
end module t_drv_timers_mod

module component_type_mod
  use mct_mod, only: mct_aVect
  implicit none
  private
  public :: component_type, component_get_x2c_cx, component_get_c2x_cx, ocn, atm, ice, rof, wav, glc
  type component_type
    type(mct_aVect), pointer :: x2c_cx => null()
    type(mct_aVect), pointer :: c2x_cx => null()
  end type component_type
  type(component_type), target :: ocn(1), atm(1), ice(1), rof(1), wav(1), glc(1)
contains
  function component_get_x2c_cx(comp) result(p)
    type(component_type), intent(in) :: comp
    type(mct_aVect), pointer :: p
    p => comp % x2c_cx
  end function component_get_x2c_cx

  function component_get_c2x_cx(comp) result(p)
    type(component_type), intent(in) :: comp
    type(mct_aVect), pointer :: p
    p => comp % c2x_cx
  end function component_get_c2x_cx
end module component_type_mod
