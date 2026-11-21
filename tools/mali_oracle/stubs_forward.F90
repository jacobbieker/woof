! Framework stand-ins the MALI forward-mode oracle (run_forward.F90) adds to
! stubs.F90, compiled after v8.4.2's own mpas_timekeeping.
!
! mpas_li_core.F is compiled verbatim, so it USEs the stream manager, the
! RBF and reconstruction initialisation, the analysis-member driver and the
! statistics.  None of them changes the state the port is held to:
!
! - the streams: the driver puts the input stream's fields in the pools
!   before li_core_init, and no stream is read or written while the run is
!   under test (output, restart and forcing streams belong to the caller);
! - mpas_rbf_interp_initialize and mpas_init_reconstruct: the mesh's
!   initialisation, whose coefficients (coeffs_reconstruct) are the input
!   mesh's, as the hex atmosphere's are (build_forward.sh's reconstruction
!   cut carries the no-op mpas_init_reconstruct);
! - the analysis members (every one is off by default, and the port refuses
!   enabling one by name).
!
! li_compute_statistics is v8.4.2's own (mpas_li_statistics.F, compiled
! verbatim): its reductions and its report are recorded by stubs.F90's
! mpas_dmpar and mpas_log.

module mpas_framework
  ! li_core USEs it; everything it calls comes from the modules below
  implicit none
end module mpas_framework

module mpas_io_streams
  implicit none
  integer, parameter :: MPAS_STREAM_LATEST_BEFORE = 4
end module mpas_io_streams

module mpas_stream_manager
  use mpas_kind_types
  use mpas_derived_types
  implicit none
  private
  public :: mpas_stream_mgr_read, mpas_stream_mgr_write, mpas_stream_mgr_reset_alarms, &
            mpas_stream_mgr_ringing_alarms, MPAS_STREAM_INPUT, MPAS_STREAM_OUTPUT

  integer, parameter :: MPAS_STREAM_INPUT = 1, MPAS_STREAM_OUTPUT = 2

contains

  subroutine mpas_stream_mgr_read(manager, streamID, timeLevel, mgLevel, rightNow, when, whence, &
                                  actualWhen, saveActualWhen, ierr)
    type(MPAS_streamManager_type), pointer :: manager
    character(len=*), intent(in), optional :: streamID
    integer, intent(in), optional :: timeLevel, mgLevel, whence
    logical, intent(in), optional :: rightNow, saveActualWhen
    character(len=*), intent(in), optional :: when
    character(len=*), intent(out), optional :: actualWhen
    integer, intent(out), optional :: ierr
    if (present(ierr)) ierr = 0
  end subroutine mpas_stream_mgr_read

  subroutine mpas_stream_mgr_write(manager, streamID, timeLevel, mgLevel, forceWriteNow, writeTime, ierr)
    type(MPAS_streamManager_type), pointer :: manager
    character(len=*), intent(in), optional :: streamID
    integer, intent(in), optional :: timeLevel, mgLevel
    logical, intent(in), optional :: forceWriteNow
    character(len=*), intent(in), optional :: writeTime
    integer, intent(out), optional :: ierr
    if (present(ierr)) ierr = 0
  end subroutine mpas_stream_mgr_write

  subroutine mpas_stream_mgr_reset_alarms(manager, streamID, direction, ierr)
    type(MPAS_streamManager_type), pointer :: manager
    character(len=*), intent(in), optional :: streamID
    integer, intent(in), optional :: direction
    integer, intent(out), optional :: ierr
    if (present(ierr)) ierr = 0
  end subroutine mpas_stream_mgr_reset_alarms

  logical function mpas_stream_mgr_ringing_alarms(manager, streamID, direction, ierr)
    type(MPAS_streamManager_type), pointer :: manager
    character(len=*), intent(in), optional :: streamID
    integer, intent(in), optional :: direction
    integer, intent(out), optional :: ierr
    if (present(ierr)) ierr = 0
    mpas_stream_mgr_ringing_alarms = .false.
  end function mpas_stream_mgr_ringing_alarms

end module mpas_stream_manager

module mpas_io_units
  implicit none
  private
  public :: mpas_new_unit, mpas_release_unit

contains

  subroutine mpas_new_unit(unit)
    integer, intent(out) :: unit
    print *, 'oracle stub: mpas_new_unit (the restart stream) is not part of this oracle'
    stop 6
  end subroutine mpas_new_unit

  subroutine mpas_release_unit(unit)
    integer, intent(in) :: unit
  end subroutine mpas_release_unit

end module mpas_io_units

module mpas_rbf_interpolation
  use mpas_pool_routines
  implicit none
  private
  public :: mpas_rbf_interp_initialize

contains

  subroutine mpas_rbf_interp_initialize(meshPool)
    ! the mesh's initialisation: not under test
    type(MPAS_pool_type), intent(inout) :: meshPool
  end subroutine mpas_rbf_interp_initialize

end module mpas_rbf_interpolation

module mpas_decomp
  use mpas_derived_types
  implicit none
  private
  public :: mpas_decomp_destroy_decomp_list

contains

  subroutine mpas_decomp_destroy_decomp_list(decompList)
    type(decomp_list), pointer :: decompList
  end subroutine mpas_decomp_destroy_decomp_list

end module mpas_decomp

module li_analysis_driver
  ! every analysis member is off (their Registry default; the port refuses
  ! enabling one by name): v8.4.2's driver then does nothing
  use mpas_derived_types
  implicit none
  private
  public :: li_analysis_init, li_analysis_compute_startup, li_analysis_compute, li_analysis_restart, &
            li_analysis_write, li_analysis_finalize

contains

  subroutine li_analysis_init(domain, err)
    type(domain_type), intent(inout) :: domain
    integer, intent(out) :: err
    err = 0
  end subroutine li_analysis_init

  subroutine li_analysis_compute_startup(domain, err)
    type(domain_type), intent(inout) :: domain
    integer, intent(out) :: err
    err = 0
  end subroutine li_analysis_compute_startup

  subroutine li_analysis_compute(domain, err)
    type(domain_type), intent(inout) :: domain
    integer, intent(out) :: err
    err = 0
  end subroutine li_analysis_compute

  subroutine li_analysis_restart(domain, err)
    type(domain_type), intent(inout) :: domain
    integer, intent(out) :: err
    err = 0
  end subroutine li_analysis_restart

  subroutine li_analysis_write(domain, err)
    type(domain_type), intent(inout) :: domain
    integer, intent(out) :: err
    err = 0
  end subroutine li_analysis_write

  subroutine li_analysis_finalize(domain, err)
    type(domain_type), intent(inout) :: domain
    integer, intent(out) :: err
    err = 0
  end subroutine li_analysis_finalize

end module li_analysis_driver
