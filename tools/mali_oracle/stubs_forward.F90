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
! - the streams' own alarms: no output stream rings here, so an analysis
!   member computed at 'output_interval' computes on startup only (when
!   asked to); the streams an analysis member names are declared by the
!   driver with their reference time.
!
! li_compute_statistics is v8.4.2's own (mpas_li_statistics.F, compiled
! verbatim): its reductions and its report are recorded by stubs.F90's
! mpas_dmpar and mpas_log.  So are the analysis-member driver and the
! globalStats and regionalStats members (mpas_li_analysis_driver.F,
! mpas_li_global_stats.F, mpas_li_regional_stats.F): run_forward.F90
! records the words each leaves in its pool.

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
            mpas_stream_mgr_ringing_alarms, MPAS_STREAM_INPUT, MPAS_STREAM_OUTPUT, &
            mpas_stream_mgr_begin_iteration, mpas_stream_mgr_get_next_stream, &
            MPAS_stream_mgr_get_property, MPAS_STREAM_PROPERTY_REF_TIME

  integer, parameter :: MPAS_STREAM_INPUT = 1, MPAS_STREAM_OUTPUT = 2
  integer, parameter :: MPAS_STREAM_PROPERTY_REF_TIME = 4

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

  subroutine mpas_stream_mgr_begin_iteration(manager, streamID, ierr)
    type(MPAS_streamManager_type), pointer :: manager
    character(len=*), intent(in), optional :: streamID
    integer, intent(out), optional :: ierr
    if (present(ierr)) ierr = 0
    manager % cursor = 0
  end subroutine mpas_stream_mgr_begin_iteration

  logical function mpas_stream_mgr_get_next_stream(manager, streamID, directionProperty, activeProperty, &
                                                   immutableProperty)
    ! the streams the driver declared, in the order declared
    type(MPAS_streamManager_type), pointer :: manager
    character(len=StrKIND), intent(out) :: streamID
    integer, intent(out), optional :: directionProperty
    logical, intent(out), optional :: activeProperty, immutableProperty
    manager % cursor = manager % cursor + 1
    mpas_stream_mgr_get_next_stream = manager % cursor <= manager % n
    streamID = ''
    if (mpas_stream_mgr_get_next_stream) streamID = manager % names(manager % cursor)
  end function mpas_stream_mgr_get_next_stream

  subroutine MPAS_stream_mgr_get_property(manager, streamID, propertyName, propertyValue, direction, ierr)
    ! a declared stream's reference time (MPAS's default for a stream without
    ! one is the run's start; the driver declares it either way)
    type(MPAS_streamManager_type), pointer :: manager
    character(len=*), intent(in) :: streamID
    integer, intent(in) :: propertyName
    character(len=*), intent(out) :: propertyValue
    integer, intent(in), optional :: direction
    integer, intent(out), optional :: ierr
    integer :: i
    if (present(ierr)) ierr = 0
    if (propertyName /= MPAS_STREAM_PROPERTY_REF_TIME) stop 12
    do i = 1, manager % n
      if (trim(manager % names(i)) == trim(streamID)) then
        propertyValue = manager % refTimes(i)
        return
      end if
    end do
    print *, 'oracle stub: no stream named ', trim(streamID)
    stop 12
  end subroutine MPAS_stream_mgr_get_property

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
