! The cloud WRF v4.7.1's radiation driver hands its schemes when a cumulus
! scheme feeds it (cu_rad_feedback's ICLOUD_CU, and shcu_physics = 5), in
! module_radiation_driver.F's own order, on columns from rad_cloud_cases.py.
! The arithmetic is wrf_rad_cloud_slices (gen_slices.py): verbatim WRF lines.
!
! Input  (stream): 'RCLDIN1 ', int32 ncase, then per case
!   int32 ncol, nz, icloud, icloud_cu, shcu_physics, mp_physics, f_qc, f_qi, f_qs
!   real32 qv, qc, qi, qs, t, p, qc_cu, qi_cu, cldfra_dp, cldfra_sh, cw_rad (ncol,nz)
! Output (stream): 'RCLDOU1 ', int32 ncase, then per case
!   real32 cldfra, qc, qi (ncol,nz)
PROGRAM run_rad_cloud
  USE wrf_rad_cloud_slices
  IMPLICIT NONE
  CHARACTER(LEN=8) :: magic
  CHARACTER(LEN=512) :: inpath, outpath
  INTEGER :: ncase, icase, ncol, nz, icloud, icloud_cu, shcu, mp, iqc, iqi, iqs
  LOGICAL :: F_QV, F_QC, F_QI, F_QS
  REAL, ALLOCATABLE, DIMENSION(:,:,:) :: qv, qc, qi, qs, t, p, qc_cu, qi_cu, &
       cldfra_dp, cldfra_sh, cw_rad, cldfra, f_ice, f_rain
  INTEGER, ALLOCATABLE :: flag(:,:,:)

  CALL get_command_argument(1, inpath)
  CALL get_command_argument(2, outpath)
  OPEN(10, FILE=TRIM(inpath), ACCESS='stream', FORM='unformatted', STATUS='old')
  OPEN(11, FILE=TRIM(outpath), ACCESS='stream', FORM='unformatted', STATUS='replace')
  READ(10) magic, ncase
  IF (magic /= 'RCLDIN1 ') STOP 'bad input magic'
  WRITE(11) 'RCLDOU1 ', ncase
  DO icase = 1, ncase
    READ(10) ncol, nz, icloud, icloud_cu, shcu, mp, iqc, iqi, iqs
    F_QV = .TRUE.
    F_QC = iqc /= 0
    F_QI = iqi /= 0
    F_QS = iqs /= 0
    ALLOCATE(qv(ncol,nz,1), qc(ncol,nz,1), qi(ncol,nz,1), qs(ncol,nz,1), &
             t(ncol,nz,1), p(ncol,nz,1), qc_cu(ncol,nz,1), qi_cu(ncol,nz,1), &
             cldfra_dp(ncol,nz,1), cldfra_sh(ncol,nz,1), cw_rad(ncol,nz,1), &
             cldfra(ncol,nz,1), f_ice(ncol,nz,1), f_rain(ncol,nz,1), &
             flag(ncol,nz,1))
    READ(10) qv, qc, qi, qs, t, p, qc_cu, qi_cu, cldfra_dp, cldfra_sh, cw_rad
    f_ice = 0.
    f_rain = 0.
    ! :1264-1281, before the cloud fraction
    CALL slice_icloud_cu1(qc, qi, qc_cu, qi_cu, F_QC, F_QI, icloud_cu, &
                          1,ncol, 1,1, 1,nz, 1,ncol, 1,1, 1,nz)
    ! :1310-1316
    cldfra = 0.
    IF (icloud == 1) THEN
      IF (F_QC .OR. F_QI) THEN
        ! :1326-1332
        CALL cal_cldfra1(cldfra, qv, qc, qi, qs, F_QV, F_QC, F_QI, F_QS, t, p, &
                         f_ice, f_rain, mp, flag, 1,ncol+1, 1,2, 1,nz+1, &
                         1,ncol, 1,1, 1,nz, 1,ncol, 1,1, 1,nz)
        ! :1333-1348 (CLDFRA_DP present, aercu_opt = 0)
        IF (icloud_cu == 2) &
          CALL slice_kf_merge(cldfra, qc, qi, qc_cu, qi_cu, cldfra_dp, cldfra_sh, &
                              1,ncol, 1,1, 1,nz, 1,ncol, 1,1, 1,nz)
      END IF
    ELSE IF (icloud == 2) THEN
      IF (F_QC .OR. F_QI) &
        CALL cal_cldfra2(cldfra, qc, qi, F_QC, F_QI, 1,ncol+1, 1,2, 1,nz+1, &
                         1,ncol, 1,1, 1,nz, 1,ncol, 1,1, 1,nz)
    END IF
    ! :1566-1575
    CALL slice_deng(cldfra, qc, cldfra_sh, cw_rad, shcu, &
                    1,ncol, 1,1, 1,nz, 1,ncol, 1,1, 1,nz)
    WRITE(11) cldfra, qc, qi
    DEALLOCATE(qv, qc, qi, qs, t, p, qc_cu, qi_cu, cldfra_dp, cldfra_sh, &
               cw_rad, cldfra, f_ice, f_rain, flag)
  END DO
  CLOSE(10)
  CLOSE(11)
END PROGRAM run_rad_cloud
