! The WRF framework calls the module_stoch cut and FFTPACK5 make: messages
! only on an FFTPACK error, which the oracle treats as fatal.
subroutine wrf_debug(level, str)
  implicit none
  integer, intent(in) :: level
  character(len=*), intent(in) :: str
  write(0, '(a)') trim(str)
  error stop 8
end subroutine wrf_debug

subroutine wrf_message(str)
  implicit none
  character(len=*), intent(in) :: str
  write(0, '(a)') trim(str)
end subroutine wrf_message

subroutine wrf_error_fatal(str)
  implicit none
  character(len=*), intent(in) :: str
  write(0, '(a)') trim(str)
  error stop 7
end subroutine wrf_error_fatal
