! The WRF framework calls module_madwrf makes.  wrf_debug: level-150
! messages (Calc_tropo_height's very-high tropopause) and the level-0 lines
! before find_cloudLayers' stop; nothing is printed.  wrf_error_fatal: the
! fixtures screen WRF's fatal profiles, so reaching it is an error.
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
