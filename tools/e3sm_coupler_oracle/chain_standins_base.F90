! Stand-ins for the E3SM v3.2.0 coupler modules one coupler interval reaches
! (run_coupler_interval.F90) that do not need MCT; chain_standins.F90 holds
! the rest. They are the union of the single-routine oracles' stand-ins
! (frac_, flux_, accum_ and e3sm_standins.F90) and do no arithmetic:
! shr_sys_abort stops with status 7 and its message; the log is the standard
! error; one task's reductions are its own value; one instance of each
! component; seq_flds_mod's and seq_infodata_mod's switches as the driver
! sets them; the timers do nothing; shr_orb_mod's routines (the orbital
! albedo, not part of the oracle) stop with status 9.

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

module shr_log_mod
  implicit none
  private
  integer, public :: shr_log_Level = 0
  integer, public :: shr_log_Unit = 0
  public :: shr_log_errMsg
contains
  function shr_log_errMsg(file, line) result(msg)
    character(*), intent(in) :: file
    integer, intent(in) :: line
    character(len=512) :: msg
    character(len=16) :: num
    write(num, '(i0)') line
    msg = 'ERROR in '//trim(file)//' at line '//trim(num)
  end function shr_log_errMsg
end module shr_log_mod

module seq_comm_mct
  implicit none
  private
  integer, parameter, public :: CPLID = 1, OCNID = 2, ICEID = 3
  integer, parameter, public :: num_inst_atm = 1, num_inst_ocn = 1, num_inst_ice = 1, num_inst_rof = 1
  integer, parameter, public :: num_inst_glc = 1, num_inst_wav = 1, num_inst_xao = 1, num_inst_frc = 1
  integer, parameter, public :: num_inst_max = 1
  integer, public :: logunit = 0
  integer, public :: loglevel = 1
  public :: seq_comm_mpicom, seq_comm_iamroot, seq_comm_setptrs
contains
  integer function seq_comm_mpicom(ID)
    integer, intent(in) :: ID
    seq_comm_mpicom = 0
  end function seq_comm_mpicom

  logical function seq_comm_iamroot(ID)
    integer, intent(in) :: ID
    seq_comm_iamroot = .true.
  end function seq_comm_iamroot

  subroutine seq_comm_setptrs(ID, iamroot, mpicom)
    integer, intent(in) :: ID
    logical, optional, intent(out) :: iamroot
    integer, optional, intent(out) :: mpicom
    if (present(iamroot)) iamroot = (ID == CPLID)
    if (present(mpicom)) mpicom = 0
  end subroutine seq_comm_setptrs
end module seq_comm_mct

module seq_flds_mod
  ! as E3SM's (public) seq_flds_mod: its use of seq_comm_mct reaches prep_ocn_merge
  use seq_comm_mct, only: seq_comm_iamroot, seq_comm_setptrs, logunit
  implicit none
  public
  character(len=80) :: atm_flux_method = 'explicit'
  logical :: atm_gustiness = .false.
  character(len=80) :: wav_ocn_coup = 'none'
  character(len=80) :: wav_atm_coup = 'none'
  character(len=80) :: wav_ice_coup = 'none'
  logical :: rof2ocn_nutrients = .false.
  logical :: seq_flds_i2o_per_cat = .false.
end module seq_flds_mod

module seq_infodata_mod
  use shr_kind_mod, only: r8 => SHR_KIND_R8
  implicit none
  private
  public :: seq_infodata_type, seq_infodata_GetData
  type seq_infodata_type
    logical :: atm_present = .true., ice_present = .true., ocn_present = .true.
    real(r8) :: flux_epbalfact = 1.0_r8
    logical :: read_restart = .false.
    logical :: flux_albav = .false.
    real(r8) :: nextsw_cday = -1.0_r8
    real(r8) :: orb_eccen = 0.0_r8, orb_mvelpp = 0.0_r8, orb_lambm0 = 0.0_r8, orb_obliqr = 0.0_r8
    logical :: dead_comps = .false.
    logical :: ocn_prognostic = .true.
    logical :: flux_diurnal = .false.
    integer :: ocn_surface_flux_scheme = 0
    logical :: coldair_outbreak_mod = .false.
    real(r8) :: flux_convergence = 0.0_r8
    integer :: flux_max_iteration = 2
  end type seq_infodata_type
contains
  subroutine seq_infodata_GetData(infodata, atm_present, ice_present, ocn_present, flux_epbalfact, read_restart, &
                                  flux_albav, dead_comps, ocn_prognostic, flux_diurnal, ocn_surface_flux_scheme, &
                                  coldair_outbreak_mod, flux_convergence, flux_max_iteration, nextsw_cday, &
                                  orb_eccen, orb_mvelpp, orb_lambm0, orb_obliqr)
    type(seq_infodata_type), intent(in) :: infodata
    logical, optional, intent(out) :: atm_present, ice_present, ocn_present
    real(r8), optional, intent(out) :: flux_epbalfact
    logical, optional, intent(out) :: read_restart, flux_albav, dead_comps, ocn_prognostic, flux_diurnal
    logical, optional, intent(out) :: coldair_outbreak_mod
    integer, optional, intent(out) :: ocn_surface_flux_scheme, flux_max_iteration
    real(r8), optional, intent(out) :: flux_convergence
    real(r8), optional, intent(out) :: nextsw_cday, orb_eccen, orb_mvelpp, orb_lambm0, orb_obliqr
    if (present(atm_present)) atm_present = infodata % atm_present
    if (present(ice_present)) ice_present = infodata % ice_present
    if (present(ocn_present)) ocn_present = infodata % ocn_present
    if (present(flux_epbalfact)) flux_epbalfact = infodata % flux_epbalfact
    if (present(read_restart)) read_restart = infodata % read_restart
    if (present(flux_albav)) flux_albav = infodata % flux_albav
    if (present(nextsw_cday)) nextsw_cday = infodata % nextsw_cday
    if (present(orb_eccen)) orb_eccen = infodata % orb_eccen
    if (present(orb_mvelpp)) orb_mvelpp = infodata % orb_mvelpp
    if (present(orb_lambm0)) orb_lambm0 = infodata % orb_lambm0
    if (present(orb_obliqr)) orb_obliqr = infodata % orb_obliqr
    if (present(dead_comps)) dead_comps = infodata % dead_comps
    if (present(ocn_prognostic)) ocn_prognostic = infodata % ocn_prognostic
    if (present(flux_diurnal)) flux_diurnal = infodata % flux_diurnal
    if (present(ocn_surface_flux_scheme)) ocn_surface_flux_scheme = infodata % ocn_surface_flux_scheme
    if (present(coldair_outbreak_mod)) coldair_outbreak_mod = infodata % coldair_outbreak_mod
    if (present(flux_convergence)) flux_convergence = infodata % flux_convergence
    if (present(flux_max_iteration)) flux_max_iteration = infodata % flux_max_iteration
  end subroutine seq_infodata_GetData
end module seq_infodata_mod

module shr_orb_mod
  use shr_kind_mod, only: r8 => SHR_KIND_R8
  implicit none
  private
  public :: shr_orb_params, shr_orb_cosz, shr_orb_decl
contains
  subroutine shr_orb_params()
    error stop 9
  end subroutine shr_orb_params

  real(r8) pure function shr_orb_cosz(jday, lat, lon, declin, dt_avg, uniform_angle)
    real(r8), intent(in) :: jday, lat, lon, declin
    real(r8), optional, intent(in) :: dt_avg
    logical, optional, intent(in) :: uniform_angle
    shr_orb_cosz = 0.0_r8
    error stop 9
  end function shr_orb_cosz

  subroutine shr_orb_decl(calday, eccen, mvelpp, lambm0, obliqr, delta, eccf)
    real(r8), intent(in) :: calday, eccen, mvelpp, lambm0, obliqr
    real(r8), intent(out) :: delta, eccf
    delta = 0.0_r8
    eccf = 0.0_r8
    error stop 9
  end subroutine shr_orb_decl
end module shr_orb_mod

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

module perf_mod
  implicit none
end module perf_mod

module shr_assert_mod
  implicit none
end module shr_assert_mod
