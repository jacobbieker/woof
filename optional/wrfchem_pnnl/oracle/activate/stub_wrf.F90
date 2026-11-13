! WOOF-OPTIN: wrfchem_pnnl
! The one WRF service activate references outside its module.
subroutine wrf_error_fatal(message)
  character(len=*), intent(in) :: message
  write(0, '(A)') trim(message)
  error stop 'wrf_error_fatal'
end subroutine wrf_error_fatal
