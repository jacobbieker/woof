! The WRF framework symbols phys/module_fr_fire_atm.F reaches for, and
! nothing else.  module_fr_fire_atm.F is compiled byte-unmodified against
! these; none of them takes part in add_fire_tracer_emissions' arithmetic.
!
!   module_state_description  num_tracer and p_fire_smoke: one passive
!                             tracer, fire_smoke, as tracer_opt = 3 registers
!                             it (Registry/registry.fire:337-338).
!   module_fr_fire_util       used wholesale by module_fr_fire_atm.F; the
!                             three routines it calls (wrf_error_fatal,
!                             print_2d_stats, print_3d_stats) are external
!                             procedures below.
module module_state_description
  implicit none
  integer, parameter :: num_tracer = 1
  integer, parameter :: p_fire_smoke = 1
end module module_state_description

module module_fr_fire_util
  implicit none
end module module_fr_fire_util

subroutine wrf_error_fatal(message)
  character(len=*), intent(in) :: message
  write(0, '(A)') trim(message)
  error stop 'wrf_error_fatal'
end subroutine wrf_error_fatal

subroutine print_2d_stats(its, ite, jts, jte, ims, ime, jms, jme, a, name)
  integer, intent(in) :: its, ite, jts, jte, ims, ime, jms, jme
  real, intent(in) :: a(ims:ime, jms:jme)
  character(len=*), intent(in) :: name
end subroutine print_2d_stats

subroutine print_3d_stats(its, ite, kts, kte, jts, jte, ims, ime, kms, kme, &
                          jms, jme, a, name)
  integer, intent(in) :: its, ite, kts, kte, jts, jte, ims, ime, kms, kme
  integer, intent(in) :: jms, jme
  real, intent(in) :: a(ims:ime, kms:kme, jms:jme)
  character(len=*), intent(in) :: name
end subroutine print_3d_stats
