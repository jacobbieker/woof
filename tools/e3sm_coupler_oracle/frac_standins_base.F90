! Stand-ins for the E3SM v3.2.0 coupler modules seq_frac_set and
! seq_map_map reach that do not need MCT (see frac_standins.F90).

module shr_sys_mod
  implicit none
  private
  public :: shr_sys_abort, shr_sys_flush
contains
  subroutine shr_sys_abort(string)
    character(*), optional, intent(in) :: string
    if (present(string)) then
      write(0, '(a)') 'shr_sys_abort: '//trim(string)
    else
      write(0, '(a)') 'shr_sys_abort'
    end if
    error stop 7
  end subroutine shr_sys_abort

  subroutine shr_sys_flush(unit)
    integer, intent(in) :: unit
    flush(unit)
  end subroutine shr_sys_flush
end module shr_sys_mod

! one task: a reduction is the task's own value
module shr_mpi_mod
  use shr_kind_mod, only: r8 => SHR_KIND_R8
  implicit none
  private
  public :: shr_mpi_min, shr_mpi_max, shr_mpi_sum
  interface shr_mpi_max
    module procedure max_r8, max_int
  end interface shr_mpi_max
contains
  subroutine shr_mpi_min(lvalue, gvalue, comm, string, all)
    real(r8), intent(in) :: lvalue
    real(r8), intent(out) :: gvalue
    integer, intent(in) :: comm
    character(*), optional, intent(in) :: string
    logical, optional, intent(in) :: all
    gvalue = lvalue
  end subroutine shr_mpi_min

  subroutine max_r8(lvalue, gvalue, comm, string, all)
    real(r8), intent(in) :: lvalue
    real(r8), intent(out) :: gvalue
    integer, intent(in) :: comm
    character(*), optional, intent(in) :: string
    logical, optional, intent(in) :: all
    gvalue = lvalue
  end subroutine max_r8

  subroutine max_int(lvalue, gvalue, comm, string, all)
    integer, intent(in) :: lvalue
    integer, intent(out) :: gvalue
    integer, intent(in) :: comm
    character(*), optional, intent(in) :: string
    logical, optional, intent(in) :: all
    gvalue = lvalue
  end subroutine max_int

  subroutine shr_mpi_sum(lvalue, gvalue, comm, string, all)
    integer, intent(in) :: lvalue
    integer, intent(out) :: gvalue
    integer, intent(in) :: comm
    character(*), optional, intent(in) :: string
    logical, optional, intent(in) :: all
    gvalue = lvalue
  end subroutine shr_mpi_sum
end module shr_mpi_mod

! the coupler's ID, its log (the standard error) and its one-task communicator
module seq_comm_mct
  implicit none
  private
  integer, parameter, public :: CPLID = 1
  integer, public :: logunit = 0
  integer, public :: loglevel = 1
  public :: seq_comm_mpicom, seq_comm_iamroot
contains
  integer function seq_comm_mpicom(ID)
    integer, intent(in) :: ID
    seq_comm_mpicom = 0
  end function seq_comm_mpicom

  logical function seq_comm_iamroot(ID)
    integer, intent(in) :: ID
    seq_comm_iamroot = .true.
  end function seq_comm_iamroot
end module seq_comm_mct

! the three switches seq_frac_set reads
module seq_infodata_mod
  implicit none
  private
  public :: seq_infodata_type, seq_infodata_GetData
  type seq_infodata_type
    logical :: atm_present = .true.
    logical :: ice_present = .true.
    logical :: ocn_present = .true.
  end type seq_infodata_type
contains
  subroutine seq_infodata_GetData(infodata, atm_present, ice_present, ocn_present)
    type(seq_infodata_type), intent(in) :: infodata
    logical, optional, intent(out) :: atm_present, ice_present, ocn_present
    if (present(atm_present)) atm_present = infodata % atm_present
    if (present(ice_present)) ice_present = infodata % ice_present
    if (present(ocn_present)) ocn_present = infodata % ocn_present
  end subroutine seq_infodata_GetData
end module seq_infodata_mod
