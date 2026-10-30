! Stub modules the extracted WRF v4.7.1 routines USE (extract.py).
!
! module_configure: only what interp_fcn_sint, copy_fcn*, smoother/sm121/
! smdsm, relax_bdytend_core and spec_bdytend touch -- the namelist getters
! nl_get_spec_zone, nl_get_feedback and nl_get_smooth_option (values set
! by the driver through the module variables) and grid_config_rec_type's
! periodic_x.  module_wrf_error: the error/message entry points.
MODULE module_configure
  IMPLICIT NONE
  TYPE grid_config_rec_type
    LOGICAL :: periodic_x = .FALSE.
  END TYPE grid_config_rec_type
  INTEGER :: stub_spec_zone = 1, stub_feedback = 1, stub_smooth_option = 0
CONTAINS
  SUBROUTINE nl_get_spec_zone(id, v)
    INTEGER, INTENT(IN) :: id
    INTEGER, INTENT(OUT) :: v
    v = stub_spec_zone
  END SUBROUTINE nl_get_spec_zone
  SUBROUTINE nl_get_feedback(id, v)
    INTEGER, INTENT(IN) :: id
    INTEGER, INTENT(OUT) :: v
    v = stub_feedback
  END SUBROUTINE nl_get_feedback
  SUBROUTINE nl_get_smooth_option(id, v)
    INTEGER, INTENT(IN) :: id
    INTEGER, INTENT(OUT) :: v
    v = stub_smooth_option
  END SUBROUTINE nl_get_smooth_option
END MODULE module_configure

MODULE module_wrf_error
  IMPLICIT NONE
CONTAINS
  SUBROUTINE wrf_message(text)
    CHARACTER(LEN=*), INTENT(IN) :: text
    WRITE(*,'(A)') TRIM(text)
  END SUBROUTINE wrf_message
  SUBROUTINE wrf_error_fatal(text)
    CHARACTER(LEN=*), INTENT(IN) :: text
    WRITE(*,'(A)') TRIM(text)
    ERROR STOP 1
  END SUBROUTINE wrf_error_fatal
END MODULE module_wrf_error
