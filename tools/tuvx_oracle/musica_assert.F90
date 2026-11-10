! Copyright (C) 2026 WOOF contributors. Apache-2.0.
!
! A clean-room stand-in for TUV-x v0.17.0's src/util/assert.F90, which carries copyleft-licensed
! portions and is never compiled into a WOOF build (tools/tuvx_oracle/build.sh overwrites it).  It is
! written from the call sites alone: TUV-x calls assert(code, condition), assert_msg(code, condition,
! message), die(code), die_msg(code, message) with character or string_t messages, and
! almost_equal(a, b) on reals.  A failed check prints its code and message and stops with status 3.
! almost_equal holds when a == b, when the relative difference 2|a - b| / (|a| + |b|) is below
! 1e-10, or when |a - b| is at most 1e-30 (TUV-x's documented default tolerances); the oracle
! calls it only to match grid wavelengths, where the values are equal or far apart.
module musica_assert

  use musica_constants, only : musica_rk, musica_dk

  implicit none
  private

  public :: assert, assert_msg, die, die_msg, almost_equal

  interface assert_msg
    module procedure :: check_with_chars
    module procedure :: check_with_string
  end interface

  interface die_msg
    module procedure :: stop_with_chars
    module procedure :: stop_with_string
  end interface

  interface almost_equal
    module procedure :: near_single
    module procedure :: near_double
  end interface

contains

  subroutine fail( code, text )
    integer, intent(in) :: code
    character(len=*), intent(in) :: text
    write(0,'(a,i0,a,a)') 'TUV-x check ', code, ' failed: ', text
    stop 3
  end subroutine fail

  subroutine assert( code, condition )
    integer, intent(in) :: code
    logical, intent(in) :: condition
    if( .not. condition ) call fail( code, '(no message)' )
  end subroutine assert

  subroutine check_with_chars( code, condition, error_message )
    integer, intent(in) :: code
    logical, intent(in) :: condition
    character(len=*), intent(in) :: error_message
    if( .not. condition ) call fail( code, error_message )
  end subroutine check_with_chars

  subroutine check_with_string( code, condition, error_message )
    use musica_string, only : string_t
    integer, intent(in) :: code
    logical, intent(in) :: condition
    type(string_t), intent(in) :: error_message
    if( .not. condition ) call fail( code, error_message%val_ )
  end subroutine check_with_string

  subroutine die( code )
    integer, intent(in) :: code
    call fail( code, '(internal error)' )
  end subroutine die

  subroutine stop_with_chars( code, error_message )
    integer, intent(in) :: code
    character(len=*), intent(in) :: error_message
    call fail( code, error_message )
  end subroutine stop_with_chars

  subroutine stop_with_string( code, error_message )
    use musica_string, only : string_t
    integer, intent(in) :: code
    type(string_t), intent(in) :: error_message
    call fail( code, error_message%val_ )
  end subroutine stop_with_string

  logical function near_double( a, b ) result( near )
    real(kind=musica_dk), intent(in) :: a, b
    real(kind=musica_dk) :: gap
    gap = abs( b - a )
    near = a == b
    if( .not. near ) near = gap <= 1.0e-30_musica_dk
    if( .not. near ) near = gap * 2.0_musica_dk < 1.0e-10_musica_dk * ( abs( a ) + abs( b ) )
  end function near_double

  logical function near_single( a, b ) result( near )
    real(kind=musica_rk), intent(in) :: a, b
    near = near_double( real( a, musica_dk ), real( b, musica_dk ) )
  end function near_single

end module musica_assert
