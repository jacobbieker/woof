! Stand-ins for the E3SM v3.2.0 coupler modules prep_ocn_merge reaches.
!
! None of them does arithmetic.  shr_sys_abort stops the oracle with its
! message; seq_comm_setptrs reports the root task (the oracle is one task);
! seq_flds_mod holds the three coupler field flags the merge reads, set by the
! driver; prep_glc_mod's routine is named in a use statement and never called.
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
end module seq_flds_mod

module prep_glc_mod
  implicit none
  private
  public :: prep_glc_calculate_subshelf_boundary_fluxes
contains
  subroutine prep_glc_calculate_subshelf_boundary_fluxes()
    error stop 8
  end subroutine prep_glc_calculate_subshelf_boundary_fluxes
end module prep_glc_mod
