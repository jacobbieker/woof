! Minimal stand-ins for the WRF framework pieces that the byte-unmodified
! phys/module_shcu_deng.F and phys/module_shallowcu_driver.F (WRF v4.7.1)
! need to compile and link.  Nothing here carries physics.
!
!   module_wrf_error          deng USEs it (no procedure of it is called)
!   module_dm, module_domain  the shallow driver USEs them (ONLY: domain)
!   module_state_description  the package constants the driver selects on
!                             (Registry.EM_COMMON:3227-3232)
!   get_wrf_debug_level, wrf_message, wrf_debug, wrf_error_fatal,
!   wrf_error_fatal3, wrf_dm_on_monitor
!                             frame/ externals.  Debug level 0, as WRF's
!                             namelist default: deng prints nothing below 10.
!
! wrf_error_fatal does NOT stop: it counts, so the driver can flag the run.
! The Deng path reaches it only for an unsupported bl_pbl_physics, which the
! fixtures never use.
MODULE module_wrf_error
  IMPLICIT NONE
  INTEGER :: deng_oracle_fatal = 0
  INTEGER :: deng_oracle_messages = 0
END MODULE module_wrf_error

MODULE module_dm
END MODULE module_dm

MODULE module_domain
  TYPE domain
    INTEGER :: unused
  END TYPE domain
END MODULE module_domain

MODULE module_state_description
  INTEGER, PARAMETER :: G3SHCUSCHEME = 1, CAMUWSHCUSCHEME = 2, &
                        GRIMSSHCUSCHEME = 3, NSCVSHCUSCHEME = 4, &
                        DENGSHCUSCHEME = 5, CAMMGMPSCHEME = 11
  INTEGER, PARAMETER :: param_first_scalar = 2
END MODULE module_state_description

SUBROUTINE get_wrf_debug_level(level)
  INTEGER, INTENT(OUT) :: level
  level = 0
END SUBROUTINE get_wrf_debug_level

SUBROUTINE wrf_message(str)
  USE module_wrf_error
  CHARACTER(LEN=*), INTENT(IN) :: str
  deng_oracle_messages = deng_oracle_messages + 1
END SUBROUTINE wrf_message

SUBROUTINE wrf_debug(level, str)
  INTEGER, INTENT(IN) :: level
  CHARACTER(LEN=*), INTENT(IN) :: str
END SUBROUTINE wrf_debug

SUBROUTINE wrf_error_fatal(str)
  USE module_wrf_error
  CHARACTER(LEN=*), INTENT(IN) :: str
  deng_oracle_fatal = deng_oracle_fatal + 1
  WRITE(0, '(A,A)') 'DENG ORACLE: wrf_error_fatal: ', TRIM(str)
END SUBROUTINE wrf_error_fatal

SUBROUTINE wrf_error_fatal3(file_str, line, str)
  USE module_wrf_error
  CHARACTER(LEN=*), INTENT(IN) :: file_str, str
  INTEGER, INTENT(IN) :: line
  deng_oracle_fatal = deng_oracle_fatal + 1
  WRITE(0, '(A,A)') 'DENG ORACLE: wrf_error_fatal3: ', TRIM(str)
END SUBROUTINE wrf_error_fatal3

LOGICAL FUNCTION wrf_dm_on_monitor()
  wrf_dm_on_monitor = .TRUE.
END FUNCTION wrf_dm_on_monitor
