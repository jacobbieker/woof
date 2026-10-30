! The two WRF framework calls cal_cldfra3/find_cloudLayers make.
! wrf_debug: the driver passes debug_flag = .false. and every remaining call
! is a level-150 debug message or the level-0 warning before the fatal stop;
! nothing is printed.  wrf_error_fatal: the column driver never feeds the
! fatal profile (make_columns.py screens it), so reaching it is an error.
subroutine wrf_debug(level, str)
  implicit none
  integer, intent(in) :: level
  character(len=*), intent(in) :: str
end subroutine wrf_debug

subroutine wrf_error_fatal(str)
  implicit none
  character(len=*), intent(in) :: str
  write(0, '(a)') trim(str)
  error stop 7
end subroutine wrf_error_fatal
