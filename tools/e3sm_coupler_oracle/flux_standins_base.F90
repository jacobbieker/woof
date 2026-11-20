! Stand-ins for the E3SM v3.2.0 modules the atmosphere-ocean flux
! (driver-mct/main/seq_flux_mct.F90 over share/util/shr_flux_mod.F90)
! reaches that do not need MCT (see flux_standins.F90).
!
! None of them does arithmetic.  shr_sys_abort stops the oracle (status 7)
! with its message; the logs are the standard error; seq_flds_mod holds the
! four coupling switches the flux reads (set by the driver), and
! seq_infodata_mod the infodata fields; shr_orb_mod names the orbital
! routines seq_flux_mct's use statement imports and the albedo (not cut)
! calls.

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

! mct_mod uses it whole; nothing here calls it
module shr_mpi_mod
  implicit none
end module shr_mpi_mod

module shr_log_mod
  implicit none
  private
  integer, public :: shr_log_Level = 0
  integer, public :: shr_log_Unit = 0
  public :: shr_log_errMsg
contains
  ! as shr_log_mod's: 'ERROR in '//trim(file)//' at line '//toString(line)
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
  integer, parameter, public :: CPLID = 1
  integer, public :: logunit = 0
  integer, public :: loglevel = 1
end module seq_comm_mct

! seq_flds_mod's coupling switches (seq_flds_set reads them from the namelist)
module seq_flds_mod
  implicit none
  private
  character(len=80), public :: atm_flux_method = 'explicit'
  logical, public :: atm_gustiness = .false.
  character(len=80), public :: wav_ocn_coup = 'none'
  character(len=80), public :: wav_atm_coup = 'none'
end module seq_flds_mod

! the infodata fields seq_flux_atmocn_mct reads
module seq_infodata_mod
  use shr_kind_mod, only: r8 => SHR_KIND_R8
  implicit none
  private
  public :: seq_infodata_type, seq_infodata_GetData
  type seq_infodata_type
    logical :: read_restart = .false.
    logical :: flux_albav = .false.
    logical :: dead_comps = .false.
    logical :: ocn_prognostic = .true.
    logical :: flux_diurnal = .false.
    integer :: ocn_surface_flux_scheme = 0
    logical :: coldair_outbreak_mod = .false.
    real(r8) :: flux_convergence = 0.0_r8
    integer :: flux_max_iteration = 2
  end type seq_infodata_type
contains
  subroutine seq_infodata_GetData(infodata, read_restart, flux_albav, dead_comps, ocn_prognostic, flux_diurnal, &
                                  ocn_surface_flux_scheme, coldair_outbreak_mod, flux_convergence, &
                                  flux_max_iteration)
    type(seq_infodata_type), intent(in) :: infodata
    logical, optional, intent(out) :: read_restart, flux_albav, dead_comps, ocn_prognostic, flux_diurnal
    logical, optional, intent(out) :: coldair_outbreak_mod
    integer, optional, intent(out) :: ocn_surface_flux_scheme, flux_max_iteration
    real(r8), optional, intent(out) :: flux_convergence
    if (present(read_restart)) read_restart = infodata % read_restart
    if (present(flux_albav)) flux_albav = infodata % flux_albav
    if (present(dead_comps)) dead_comps = infodata % dead_comps
    if (present(ocn_prognostic)) ocn_prognostic = infodata % ocn_prognostic
    if (present(flux_diurnal)) flux_diurnal = infodata % flux_diurnal
    if (present(ocn_surface_flux_scheme)) ocn_surface_flux_scheme = infodata % ocn_surface_flux_scheme
    if (present(coldair_outbreak_mod)) coldair_outbreak_mod = infodata % coldair_outbreak_mod
    if (present(flux_convergence)) flux_convergence = infodata % flux_convergence
    if (present(flux_max_iteration)) flux_max_iteration = infodata % flux_max_iteration
  end subroutine seq_infodata_GetData
end module seq_infodata_mod

! named by seq_flux_mct's use statement; only the ocean albedo (not cut) calls them
module shr_orb_mod
  use shr_kind_mod, only: r8 => SHR_KIND_R8
  implicit none
  private
  public :: shr_orb_params, shr_orb_cosz, shr_orb_decl
contains
  subroutine shr_orb_params()
    error stop 9
  end subroutine shr_orb_params

  real(r8) function shr_orb_cosz()
    shr_orb_cosz = 0.0_r8
    error stop 9
  end function shr_orb_cosz

  subroutine shr_orb_decl()
    error stop 9
  end subroutine shr_orb_decl
end module shr_orb_mod
