! The two module_wrf_error entries module_fr_fire_util.F's message and crash
! call, so the module compiles whole outside the WRF tree.  Messages are
! dropped (continue_at_boundary reports every call); a crash stops the run.
module module_wrf_error
  implicit none
contains
  subroutine wrf_message(msg)
    character(len=*), intent(in) :: msg
  end subroutine wrf_message

  subroutine wrf_error_fatal(msg)
    character(len=*), intent(in) :: msg
    write(0, '(A)') trim(msg)
    error stop 'wrf_error_fatal'
  end subroutine wrf_error_fatal
end module module_wrf_error

! The process-count queries the module's array writers make (never called
! here): one process.
subroutine wrf_get_nproc(nprocs)
  integer, intent(out) :: nprocs
  nprocs = 1
end subroutine wrf_get_nproc

subroutine wrf_get_myproc(myproc)
  integer, intent(out) :: myproc
  myproc = 0
end subroutine wrf_get_myproc
