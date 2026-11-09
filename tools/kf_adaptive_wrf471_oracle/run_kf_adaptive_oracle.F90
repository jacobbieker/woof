! Kain-Fritsch (cu_physics=1) under a VARYING model step, WRF v4.7.1.
!
! The scheme is the byte-unmodified phys/module_cu_kfeta.F (KF_eta_CPS).
! Around it, per model step, this driver does the two pieces of WRF that
! decide what KF's held state does when dt changes from step to step:
!
!   1. module_cumulus_driver.F:803-857, the run_param decision with the
!      adaptive arm: cudtacttime is armed at curr_secs + cudt*60 and the
!      scheme runs when curr_secs >= cudtacttime (and on itimestep 1);
!   2. module_physics_addtendc.F:2289-2293 and :2346-2380, advance_ppt on
!      every step: RAINC += PRATEC*DT, then the KFETASCHEME NCA expiry
!      (NINT(NCA/DT) .LE. 1 zeroes the stored tendencies) and NCA -= DT.
!
! Both are transcribed statement for statement below (they sit inside
! routines that need the whole WRF registry to compile); build.sh pins the
! two source files they come from.  The atmospheric state is frozen: the
! case grades the clock, not the dynamics, so every column sees the same
! input on every step and only DT, CURR_SECS and the held state move.
!
! Input (stream, little endian): 'KFADT1  ', int32 ncol, nz, nsteps,
!   warm_rain, f_qr, f_qi, f_qs, trigger, stepcu; real32 dx, cudt (minutes);
!   real32 dt(nsteps); then per column real32 u, v, t, qv, p, dz8w, rho, pi
!   (nz each) and w (nz+1).
! Output: 'KFADO1  ', int32 ncol, nz, nsteps, then per step
!   int32 run_param, fatal; and per column
!   real32 nca, rainc, pratec, raincv (after advance_ppt),
!          rthcuten as the step's RK forcing sees it (before advance_ppt),
!          rthcuten, rqvcuten held after advance_ppt (nz each).
PROGRAM run_kf_adaptive_oracle
  USE module_wrf_error, ONLY: kf_oracle_fatal
  USE module_cu_kfeta, ONLY: KF_eta_CPS, kf_lutab
  IMPLICIT NONE
  ! share/module_model_constants.F (v4.7.1, unchanged from v4.6.1)
  REAL, PARAMETER :: g = 9.81, r_d = 287., cp = 7.*r_d/2., r_v = 461.6
  REAL, PARAMETER :: XLV0 = 3.15E6, XLV1 = 2370., XLS0 = 2.905E6, XLS1 = 259.532
  REAL, PARAMETER :: SVP1 = 0.6112, SVP2 = 17.67, SVP3 = 29.65, SVPT0 = 273.15
  REAL, PARAMETER :: EP_1 = R_v/R_d-1., EP_2 = R_d/R_v
  CHARACTER(LEN=8) :: magic
  CHARACTER(LEN=256) :: inpath, outpath
  INTEGER :: ncol, nz, nsteps, iwarm, ifqr, ifqi, ifqs, trig, stepcu
  INTEGER :: n, i, k, itimestep, irun, ifatal
  REAL :: dx, cudt, dt, curr_secs, cudtacttime
  LOGICAL :: warm_rain, f_qr, f_qi, f_qs, decided, run_param
  REAL, ALLOCATABLE :: dts(:)
  REAL, ALLOCATABLE, DIMENSION(:,:,:) :: u3, v3, t3, qv3, p3, dz3, rho3, &
       pi3, w03, w3, rth, rqv, rqc, rqr, rqi, rqs, rqvften, cdp, csh, qck, &
       qik, udr, ddr, uer, der
  REAL, ALLOCATABLE, DIMENSION(:,:) :: raincv, pratec, nca, cutop, cubot, &
       shall, timec_kf, rainc
  LOGICAL, ALLOCATABLE, DIMENSION(:,:) :: cu_act_flag
  REAL, ALLOCATABLE :: col(:)

  CALL get_command_argument(1, inpath)
  CALL get_command_argument(2, outpath)
  OPEN(10, FILE=TRIM(inpath), ACCESS='stream', FORM='unformatted', STATUS='old')
  OPEN(11, FILE=TRIM(outpath), ACCESS='stream', FORM='unformatted', STATUS='replace')
  READ(10) magic, ncol, nz, nsteps, iwarm, ifqr, ifqi, ifqs, trig, stepcu
  IF (magic /= 'KFADT1  ') STOP 'bad input magic'
  READ(10) dx, cudt
  ALLOCATE(dts(nsteps))
  READ(10) dts
  warm_rain = iwarm /= 0
  f_qr = ifqr /= 0
  f_qi = ifqi /= 0
  f_qs = ifqs /= 0
  WRITE(11) 'KFADO1  ', ncol, nz, nsteps

  ALLOCATE(u3(ncol,nz+1,1), v3(ncol,nz+1,1), t3(ncol,nz+1,1), &
           qv3(ncol,nz+1,1), p3(ncol,nz+1,1), dz3(ncol,nz+1,1), &
           rho3(ncol,nz+1,1), pi3(ncol,nz+1,1), w03(ncol,nz+1,1), &
           w3(ncol,nz+1,1), rth(ncol,nz+1,1), rqv(ncol,nz+1,1), &
           rqc(ncol,nz+1,1), rqr(ncol,nz+1,1), rqi(ncol,nz+1,1), &
           rqs(ncol,nz+1,1), rqvften(ncol,nz+1,1), cdp(ncol,nz+1,1), &
           csh(ncol,nz+1,1), qck(ncol,nz+1,1), qik(ncol,nz+1,1), &
           udr(ncol,nz+1,1), ddr(ncol,nz+1,1), uer(ncol,nz+1,1), &
           der(ncol,nz+1,1), col(nz+1))
  ALLOCATE(raincv(ncol,1), pratec(ncol,1), nca(ncol,1), cutop(ncol,1), &
           cubot(ncol,1), shall(ncol,1), timec_kf(ncol,1), rainc(ncol,1), &
           cu_act_flag(ncol,1))
  u3 = 0.; v3 = 0.; t3 = 0.; qv3 = 0.; p3 = 0.; dz3 = 0.; rho3 = 0.
  pi3 = 0.; w3 = 0.
  DO i = 1, ncol
    CALL rd(u3, i); CALL rd(v3, i); CALL rd(t3, i); CALL rd(qv3, i)
    CALL rd(p3, i); CALL rd(dz3, i); CALL rd(rho3, i); CALL rd(pi3, i)
    READ(10) col(1:nz+1)
    w3(i,1:nz+1,1) = col(1:nz+1)
  END DO
  CLOSE(10)

  ! WRF's start-of-run state: kf_eta_init (module_cu_kfeta.F:3084-3170 on
  ! a cold start) zeroes the tendencies and W0AVG and sets NCA = -100;
  ! RAINC, RAINCV and PRATEC start at zero.
  CALL kf_lutab(SVP1, SVP2, SVP3, SVPT0)
  w03 = 0.; rth = 0.; rqv = 0.; rqc = 0.; rqr = 0.; rqi = 0.; rqs = 0.
  rqvften = 0.; cdp = 0.; csh = 0.; qck = 0.; qik = 0.
  udr = 0.; ddr = 0.; uer = 0.; der = 0.
  nca = -100.; rainc = 0.; raincv = 0.; pratec = 0.
  cutop = 0.; cubot = 0.; shall = 0.; timec_kf = 0.
  cudtacttime = 0.
  curr_secs = 0.

  DO n = 1, nsteps
    itimestep = n
    dt = dts(n)
    kf_oracle_fatal = .FALSE.

    ! ---- module_cumulus_driver.F:803-857 (adapt_step_flag = .TRUE.) ----
    IF ( cudtacttime .EQ. 0. ) THEN
       cudtacttime = curr_secs + cudt*60.
    END IF
    decided = .FALSE.
    run_param = .FALSE.
    IF ( ( .NOT. decided ) .AND. &
         ( itimestep .EQ. 1 ) ) THEN
       run_param   = .TRUE.
       decided     = .TRUE.
    END IF
    IF ( ( .NOT. decided ) .AND. &
         ( ( cudt .EQ. 0. ) .OR. ( stepcu .EQ. 1 ) ) ) THEN
       run_param   = .TRUE.
       decided     = .TRUE.
    END IF
    ! (the fixed-step MOD(itimestep,stepcu) arm is skipped when adaptive)
    IF ( ( .NOT. decided ) .AND. &
         ( curr_secs .GE. cudtacttime ) ) THEN
       run_param   = .TRUE.
       decided     = .TRUE.
       cudtacttime = curr_secs + cudt*60
    END IF

    IF ( run_param ) THEN
       cu_act_flag = .TRUE.
       ! module_cumulus_driver.F:1017-1048, PRATEC=tmppratec (= PRATEC on
       ! entry, :779-784, and copied back at :1659-1661).
       CALL KF_eta_CPS( &
            ids=1, ide=ncol+1, jds=1, jde=2, kds=1, kde=nz+1, &
            ims=1, ime=ncol, jms=1, jme=1, kms=1, kme=nz+1, &
            its=1, ite=ncol, jts=1, jte=1, kts=1, kte=nz, &
            trigger=trig, &
            DT=dt, KTAU=itimestep, DX=dx, CUDT=cudt, ADAPT_STEP_FLAG=.TRUE., &
            rho=rho3, RAINCV=raincv, PRATEC=pratec, NCA=nca, &
            U=u3, V=v3, TH=t3, T=t3, W=w3, dz8w=dz3, Pcps=p3, pi=pi3, &
            W0AVG=w03, XLV0=XLV0, XLV1=XLV1, XLS0=XLS0, XLS1=XLS1, CP=cp, &
            R=r_d, G=g, EP1=EP_1, EP2=EP_2, SVP1=SVP1, SVP2=SVP2, SVP3=SVP3, &
            SVPT0=SVPT0, STEPCU=stepcu, CU_ACT_FLAG=cu_act_flag, &
            warm_rain=warm_rain, CUTOP=cutop, CUBOT=cubot, QV=qv3, &
            shall=shall, &
            F_QV=.TRUE., F_QC=.TRUE., F_QR=f_qr, F_QI=f_qi, F_QS=f_qs, &
            RTHCUTEN=rth, RQVCUTEN=rqv, RQCCUTEN=rqc, RQRCUTEN=rqr, &
            RQICUTEN=rqi, RQSCUTEN=rqs, RQVFTEN=rqvften, &
            cldfra_dp_KF=cdp, cldfra_sh_KF=csh, qc_KF=qck, qi_KF=qik, &
            UDR_KF=udr, DDR_KF=ddr, UER_KF=uer, DER_KF=der, &
            TIMEC_KF=timec_kf, KF_EDRATES=0)
    END IF
    irun = 0
    IF (run_param) irun = 1
    ifatal = 0
    IF (kf_oracle_fatal) ifatal = 1
    WRITE(11) irun, ifatal
    ! The step's RK forcing reads the held rates before advance_ppt.
    DO i = 1, ncol
      col(1:nz) = rth(i,1:nz,1)
      WRITE(11) col(1:nz)
    END DO

    ! ---- module_physics_addtendc.F:2289-2293, advance_ppt ----
    DO i = 1, ncol
       RAINC(I,1)  = RAINC(I,1) + PRATEC(I,1)*DT
    END DO
    ! ---- module_physics_addtendc.F:2346-2380, CASE (KFETASCHEME) ----
    DO i = 1, ncol
       IF ( NCA(I,1) .GT. 0 ) THEN
          IF ( NINT(NCA(I,1) / DT) .LE. 1 ) THEN
             DO k = 1, nz
                RTH(i,k,1)=0.
                RQV(i,k,1)=0.
                RQC(i,k,1)=0.
                RQR(i,k,1)=0.
                RQI(i,k,1)=0.
                RQS(i,k,1)=0.
             ENDDO
          ENDIF
          NCA(I,1)=NCA(I,1)-DT ! Decrease NCA
       ENDIF
    END DO

    DO i = 1, ncol
      WRITE(11) nca(i,1), rainc(i,1), pratec(i,1), raincv(i,1)
      col(1:nz) = rth(i,1:nz,1)
      WRITE(11) col(1:nz)
      col(1:nz) = rqv(i,1:nz,1)
      WRITE(11) col(1:nz)
    END DO
    curr_secs = curr_secs + dt
  END DO
  CLOSE(11)
CONTAINS
  SUBROUTINE rd(a, i)
    REAL, INTENT(INOUT) :: a(:,:,:)
    INTEGER, INTENT(IN) :: i
    READ(10) col(1:nz)
    a(i,1:nz,1) = col(1:nz)
  END SUBROUTINE rd
END PROGRAM run_kf_adaptive_oracle
