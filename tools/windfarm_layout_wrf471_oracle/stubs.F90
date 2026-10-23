! The error/message entry points share/module_llxy.F and the cuts call.
! Nothing here computes.

module module_wrf_error
  implicit none
contains
  subroutine wrf_error_fatal(msg)
    character(len=*), intent(in) :: msg
    write (0, '(A,A)') 'wrf_error_fatal: ', trim(msg)
    stop 9
  end subroutine wrf_error_fatal
  subroutine wrf_debug(level, msg)
    integer, intent(in) :: level
    character(len=*), intent(in) :: msg
    if (level < -huge(level)) write (0, '(A)') trim(msg)
  end subroutine wrf_debug
  subroutine wrf_message(msg)
    character(len=*), intent(in) :: msg
    if (len_trim(msg) < 0) write (0, '(A)') trim(msg)
  end subroutine wrf_message
end module module_wrf_error
