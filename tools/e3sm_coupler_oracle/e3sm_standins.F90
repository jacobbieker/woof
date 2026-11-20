! Stand-ins for the E3SM v3.2.0 coupler modules prep_ocn_merge reaches.
!
! None of them does arithmetic.  shr_sys_abort stops the oracle with its
! message; seq_comm_setptrs reports the root task (the oracle is one task);
! seq_flds_mod holds the coupler field flags and lists the merge and the
! ocean's index set read, set by the driver; seq_infodata_mod the precipitation
! factor and the ocean-to-glc thermal forcing switch the ocean's import and
! export read; shr_log_mod and shr_assert_mod what glc_zocnclass_mod uses
! (built with NDEBUG, as its assertions only check an array's bound);
! prep_glc_mod's routine is named in a use statement and never called.
! mct_mod is not a stand-in: build.sh writes it from the pinned
! share/util/mct_mod.F90's own m_AttrVect renames, over the pinned MCT.

module shr_sys_mod
  implicit none
  private
  public :: shr_sys_abort, shr_sys_flush
contains
  subroutine shr_sys_abort(string)
    character(*), optional, intent(in) :: string
    if (present(string)) write(0, '(a)') 'shr_sys_abort: '//trim(string)
    error stop 7
  end subroutine shr_sys_abort

  subroutine shr_sys_flush(unit)
    integer, intent(in) :: unit
    flush(unit)
  end subroutine shr_sys_flush
end module shr_sys_mod

module seq_comm_mct
  implicit none
  private
  integer, parameter, public :: CPLID = 1
  integer, public :: logunit = 6
  public :: seq_comm_setptrs
contains
  subroutine seq_comm_setptrs(ID, iamroot)
    integer, intent(in) :: ID
    logical, optional, intent(out) :: iamroot
    if (present(iamroot)) iamroot = (ID == CPLID)
  end subroutine seq_comm_setptrs
end module seq_comm_mct

module seq_flds_mod
  implicit none
  character(len=80) :: wav_ocn_coup = 'none'
  logical :: rof2ocn_nutrients = .false.
  logical :: seq_flds_i2o_per_cat = .false.
  ! the component field lists and the ocean-to-land switch mpaso_cpl_indices reads
  character(len=16384) :: seq_flds_x2o_fields = '', seq_flds_o2x_fields = ''
  logical :: ocn_lnd_one_way = .false.
  ! the sea ice's: its field lists, the wave coupling and the wave spectrum's length
  character(len=80) :: wav_ice_coup = 'none'
  integer :: wav_nfreq = 0
  character(len=16384) :: seq_flds_x2i_fields = '', seq_flds_i2x_fields = ''
end module seq_flds_mod

! the two coupler settings the ocean's import and export read
module seq_infodata_mod
  use shr_kind_mod, only: SHR_KIND_R8
  implicit none
  private
  public :: seq_infodata_type, seq_infodata_GetData
  type seq_infodata_type
    real(SHR_KIND_R8) :: precip_fact = 1.0_SHR_KIND_R8
    logical :: ocn_c2_glctf = .false.
  end type seq_infodata_type
contains
  subroutine seq_infodata_GetData(infodata, precip_fact, ocn_c2_glctf)
    type(seq_infodata_type), intent(in) :: infodata
    real(SHR_KIND_R8), optional, intent(out) :: precip_fact
    logical, optional, intent(out) :: ocn_c2_glctf
    if (present(precip_fact)) precip_fact = infodata % precip_fact
    if (present(ocn_c2_glctf)) ocn_c2_glctf = infodata % ocn_c2_glctf
  end subroutine seq_infodata_GetData
end module seq_infodata_mod

! glc_zocnclass_mod's error-message helper and the module its (NDEBUG-empty)
! assertion macros use
module shr_log_mod
  implicit none
  private
  public :: shr_log_errMsg
contains
  function shr_log_errMsg(file, line) result(msg)
    character(len=*), intent(in) :: file
    integer, intent(in) :: line
    character(len=512) :: msg
    write(msg, '(a, i0)') trim(file)//':', line
  end function shr_log_errMsg
end module shr_log_mod

module shr_assert_mod
  implicit none
end module shr_assert_mod

module prep_glc_mod
  implicit none
  private
  public :: prep_glc_calculate_subshelf_boundary_fluxes
contains
  subroutine prep_glc_calculate_subshelf_boundary_fluxes()
    error stop 8
  end subroutine prep_glc_calculate_subshelf_boundary_fluxes
end module prep_glc_mod
