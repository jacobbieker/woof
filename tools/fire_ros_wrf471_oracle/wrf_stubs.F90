! Serial stand-ins for the WRF framework routines module_fr_fire_util.F and
! module_fr_fire_phys.F call, so both compile byte-unmodified outside WRF.
! A serial (non-DM_PARALLEL) WRF build links the same behaviour: one rank,
! which is the monitor, and broadcasts that change nothing.  Nothing here
! computes a value the oracle records.
module module_wrf_error
  implicit none
contains
  subroutine wrf_message(msg)
    character(len=*), intent(in) :: msg
    write(0, '(A)') trim(msg)
  end subroutine wrf_message

  subroutine wrf_error_fatal(msg)
    character(len=*), intent(in) :: msg
    write(0, '(A)') trim(msg)
    error stop 'wrf_error_fatal'
  end subroutine wrf_error_fatal
end module module_wrf_error

logical function wrf_dm_on_monitor()
  wrf_dm_on_monitor = .true.
end function wrf_dm_on_monitor

subroutine wrf_dm_bcast_real(buf, n)
  integer, intent(in) :: n
  real, intent(inout) :: buf(*)
end subroutine wrf_dm_bcast_real

subroutine wrf_dm_bcast_integer(buf, n)
  integer, intent(in) :: n
  integer, intent(inout) :: buf(*)
end subroutine wrf_dm_bcast_integer

subroutine wrf_get_nproc(n)
  integer, intent(out) :: n
  n = 1
end subroutine wrf_get_nproc

subroutine wrf_get_myproc(m)
  integer, intent(out) :: m
  m = 0
end subroutine wrf_get_myproc
