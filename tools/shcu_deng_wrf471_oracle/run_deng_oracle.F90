! Column-tile driver for WRF v4.7.1 Deng shallow cumulus (shcu_physics = 5),
! byte-unmodified phys/module_shcu_deng.F through the byte-unmodified
! phys/module_shallowcu_driver.F.
!
! Each case is one horizontal tile of NCOL columns (i = 1..NCOL, j = 1) run
! for NSTEPS consecutive model steps.  Every step is the call
! dyn_em/module_first_rk_step_part1.F:1570-1663 makes, keyword for keyword
! for every argument the Deng path reads, so the driver's own work is graded
! too: the TKE/KTH/BBLS selection by bl_pbl_physics (MYJ: TKE_PBL; MYNN:
! 0.5*QKE), and the closing RAINSHV = PRATESH*DT.  The Deng state the
! Registry declares (package dengshcuscheme, Registry.EM_COMMON:3232) lives
! in this program between steps exactly as it lives in the WRF grid.
!
! Memory dims are WRF's: ims:ime = 1:NCOL, kms:kme = 1:NZ+1, jms:jme = 1:1;
! ids:ide = 1:NCOL+1 and kds:kde = 1:NZ+1, so the driver's
! min(i_end, ide-1) and min(k_end, kde-1) give the tile 1:NCOL x 1:NZ.
!
! Input  (stream, little endian): 'DENGIN1 ', int32 ncase, then per case
!   int32  ncol, nz, nsteps, bl_pbl_physics, adapt_step_flag
!   real32 dt, dx, gmt, dnw(nz)
!   real32 ht(ncol), xlong(ncol)
!   state:  real32 rainsh, rainshv, rainshvb, pblmax, capesave, xtime1,
!                  radsave, clddpthb, cldtopb, mavail, pblhavg, pratesh (ncol)
!           int32  ltopb, kdcldtop, kdcldbas (ncol)
!           real32 ainckfsa(ncol,100)
!           real32 w0avg, tkeavg, cldareaa, cldareab, cldliqa, cldliqb,
!                  wub (ncol,nz)
!   per step: int32 itimestep; real32 xtime; real32 pblh(ncol)
!           real32 u, v, th, t, qv, qc, qr, p, pi, rho, dz8w, tke_pbl, qke,
!                  el_pbl, exch_h, rthratenlw, rthratensw (ncol,nz)
!           real32 z_at_w, w (ncol,nz+1)
! Output: 'DENGOU1 ', int32 ncase, then per case int32 ncol, nz, nsteps and
!   per step: int32 fatal, messages
!           real32 rthshten, rqvshten, rqcshten, rqrshten, rushten, rvshten,
!                  rdcashten, rqcdcshten, cldfra_sh, ca_rad, cw_rad, w0avg,
!                  tkeavg, cldareaa, cldareab, cldliqa, cldliqb, wub (ncol,nz)
!           real32 rainsh, rainshv, rainshvb, pblmax, capesave, xtime1,
!                  radsave, clddpthb, cldtopb, mavail, pblhavg, pratesh (ncol)
!           int32  ltopb, kdcldtop, kdcldbas (ncol)
!           real32 ainckfsa(ncol,100)
! The INTENT(OUT) cloud fields are refilled with -999. before every call, so
! a word the scheme leaves unwritten shows up as the sentinel.
PROGRAM run_deng_oracle
  USE module_wrf_error, ONLY: deng_oracle_fatal, deng_oracle_messages
  USE module_shallowcu_driver, ONLY: shallowcu_driver
  IMPLICIT NONE
  CHARACTER(LEN=8) :: magic
  CHARACTER(LEN=512) :: inpath, outpath
  INTEGER :: ncase, icase, ncol, nz, nsteps, ibl, iadapt, istep, itimestep
  INTEGER :: ide, kde, kme
  REAL :: dt, dx, gmt, xtime, cudt, curr_secs
  LOGICAL :: adapt, pert_deng, is_cammgmp_used
  INTEGER, DIMENSION(1) :: i_start, i_end, j_start, j_end
  REAL, ALLOCATABLE, DIMENSION(:) :: dnw, znu
  REAL, ALLOCATABLE, DIMENSION(:,:) :: ht, xlong, pblh, rainsh, rainshv, &
       rainshvb, pblmax, capesave, xtime1, radsave, clddpthb, cldtopb, &
       mavail, pblhavg, pratesh, nca, htop, hbot, xland, dx2d, area2d
  INTEGER, ALLOCATABLE, DIMENSION(:,:) :: ltopb, kdcldtop, kdcldbas
  REAL, ALLOCATABLE, DIMENSION(:,:,:) :: ainckfsa
  REAL, ALLOCATABLE, DIMENSION(:,:,:) :: u, v, th, t, qv, qc, qr, p, pii, &
       rho, dz8w, tke_pbl, qke, el_pbl, exch_h, rthratenlw, rthratensw, &
       z_at_w, w, z, p8w, w0avg, tkeavg, cldareaa, cldareab, cldliqa, &
       cldliqb, wub, rthshten, rqvshten, rqcshten, rqrshten, rushten, &
       rvshten, rdcashten, rqcdcshten, cldfra_sh, ca_rad, cw_rad, cldfra, &
       cldfra_old, cldfra_old_mp, cldfra_conv, dlf2, shfrc3d, rthraten, &
       rthblten
  REAL, ALLOCATABLE, DIMENSION(:,:,:,:) :: moist

  CALL get_command_argument(1, inpath)
  CALL get_command_argument(2, outpath)
  OPEN(10, FILE=TRIM(inpath), ACCESS='stream', FORM='unformatted', STATUS='old')
  OPEN(11, FILE=TRIM(outpath), ACCESS='stream', FORM='unformatted', STATUS='replace')
  READ(10) magic, ncase
  IF (magic /= 'DENGIN1 ') STOP 'bad input magic'
  WRITE(11) 'DENGOU1 ', ncase

  DO icase = 1, ncase
    READ(10) ncol, nz, nsteps, ibl, iadapt
    READ(10) dt, dx, gmt
    adapt = iadapt /= 0
    ide = ncol + 1
    kde = nz + 1
    kme = nz + 1
    i_start = 1; i_end = ncol; j_start = 1; j_end = 1
    ALLOCATE(dnw(kme), znu(kme))
    dnw = 0.; znu = 0.
    READ(10) dnw(1:nz)

    ALLOCATE(ht(ncol,1), xlong(ncol,1), pblh(ncol,1), rainsh(ncol,1), &
             rainshv(ncol,1), rainshvb(ncol,1), pblmax(ncol,1), capesave(ncol,1), &
             xtime1(ncol,1), radsave(ncol,1), clddpthb(ncol,1), cldtopb(ncol,1), &
             mavail(ncol,1), pblhavg(ncol,1), pratesh(ncol,1), nca(ncol,1), &
             htop(ncol,1), hbot(ncol,1), xland(ncol,1), dx2d(ncol,1), area2d(ncol,1))
    ALLOCATE(ltopb(ncol,1), kdcldtop(ncol,1), kdcldbas(ncol,1))
    ALLOCATE(ainckfsa(ncol,100,1))
    ALLOCATE(u(ncol,kme,1), v(ncol,kme,1), th(ncol,kme,1), t(ncol,kme,1), &
             qv(ncol,kme,1), qc(ncol,kme,1), qr(ncol,kme,1), p(ncol,kme,1), &
             pii(ncol,kme,1), rho(ncol,kme,1), dz8w(ncol,kme,1), &
             tke_pbl(ncol,kme,1), qke(ncol,kme,1), el_pbl(ncol,kme,1), &
             exch_h(ncol,kme,1), rthratenlw(ncol,kme,1), rthratensw(ncol,kme,1), &
             z_at_w(ncol,kme,1), w(ncol,kme,1), z(ncol,kme,1), p8w(ncol,kme,1), &
             w0avg(ncol,kme,1), tkeavg(ncol,kme,1), cldareaa(ncol,kme,1), &
             cldareab(ncol,kme,1), cldliqa(ncol,kme,1), cldliqb(ncol,kme,1), &
             wub(ncol,kme,1), rthshten(ncol,kme,1), rqvshten(ncol,kme,1), &
             rqcshten(ncol,kme,1), rqrshten(ncol,kme,1), rushten(ncol,kme,1), &
             rvshten(ncol,kme,1), rdcashten(ncol,kme,1), rqcdcshten(ncol,kme,1), &
             cldfra_sh(ncol,kme,1), ca_rad(ncol,kme,1), cw_rad(ncol,kme,1), &
             cldfra(ncol,kme,1), cldfra_old(ncol,kme,1), cldfra_old_mp(ncol,kme,1), &
             cldfra_conv(ncol,kme,1), dlf2(ncol,kme,1), shfrc3d(ncol,kme,1), &
             rthraten(ncol,kme,1), rthblten(ncol,kme,1))
    ALLOCATE(moist(ncol,kme,1,1))
    ! Fields the Deng path never reads: zero, as WRF's grid would hold them.
    z = 0.; p8w = 0.; cldfra = 0.; cldfra_old = 0.; cldfra_old_mp = 0.
    cldfra_conv = 0.; moist = 0.; nca = 0.; htop = 0.; hbot = 0.; xland = 1.
    dx2d = dx; area2d = dx*dx; rthraten = 0.; rthblten = 0.
    ! Tendencies start at deng_shcu_init's zero.
    rthshten = 0.; rqvshten = 0.; rqcshten = 0.; rqrshten = 0.; rushten = 0.
    rvshten = 0.; rdcashten = 0.; rqcdcshten = 0.
    w0avg = 0.; tkeavg = 0.; cldareaa = 0.; cldareab = 0.; cldliqa = 0.
    cldliqb = 0.; wub = 0.
    ! Level kme of every field is memory WRF allocates and the scheme never
    ! reads, except Z_AT_W and W, which are read to kte+1.
    u = 0.; v = 0.; th = 0.; t = 0.; qv = 0.; qc = 0.; qr = 0.; p = 0.
    pii = 0.; rho = 0.; dz8w = 0.; tke_pbl = 0.; qke = 0.; el_pbl = 0.
    exch_h = 0.; rthratenlw = 0.; rthratensw = 0.

    READ(10) ht(:,1), xlong(:,1)
    READ(10) rainsh(:,1), rainshv(:,1), rainshvb(:,1), pblmax(:,1), &
             capesave(:,1), xtime1(:,1), radsave(:,1), clddpthb(:,1), &
             cldtopb(:,1), mavail(:,1), pblhavg(:,1), pratesh(:,1)
    READ(10) ltopb(:,1), kdcldtop(:,1), kdcldbas(:,1)
    READ(10) ainckfsa(:,:,1)
    READ(10) w0avg(:,1:nz,1), tkeavg(:,1:nz,1), cldareaa(:,1:nz,1), &
             cldareab(:,1:nz,1), cldliqa(:,1:nz,1), cldliqb(:,1:nz,1), &
             wub(:,1:nz,1)

    WRITE(11) ncol, nz, nsteps
    cudt = 0.
    pert_deng = .FALSE.
    is_cammgmp_used = .FALSE.
    DO istep = 1, nsteps
      READ(10) itimestep, xtime
      READ(10) pblh(:,1)
      READ(10) u(:,1:nz,1), v(:,1:nz,1), th(:,1:nz,1), t(:,1:nz,1), &
               qv(:,1:nz,1), qc(:,1:nz,1), qr(:,1:nz,1), p(:,1:nz,1), &
               pii(:,1:nz,1), rho(:,1:nz,1), dz8w(:,1:nz,1), &
               tke_pbl(:,1:nz,1), qke(:,1:nz,1), el_pbl(:,1:nz,1), &
               exch_h(:,1:nz,1), rthratenlw(:,1:nz,1), rthratensw(:,1:nz,1)
      READ(10) z_at_w(:,1:nz+1,1), w(:,1:nz+1,1)
      curr_secs = xtime*60.
      cldfra_sh = -999.; ca_rad = -999.; cw_rad = -999.
      dlf2 = -999.; shfrc3d = -999.
      deng_oracle_fatal = 0
      deng_oracle_messages = 0

      ! module_first_rk_step_part1.F:1570-1663, every argument the Deng path
      ! (and the driver around it) reads; the optional CAM/GRIMS/NSCV-only
      ! arguments are absent, as they are irrelevant to shcu_physics = 5.
      CALL shallowcu_driver(                                              &
                    IDS=1,IDE=ide, JDS=1,JDE=2, KDS=1,KDE=kde             &
                   ,IMS=1,IME=ncol, JMS=1,JME=1, KMS=1,KME=kme            &
                   ,IPS=1,IPE=ncol, JPS=1,JPE=1, KPS=1,KPE=kde            &
                   ,I_START=i_start, I_END=min(i_end, ide-1)              &
                   ,J_START=j_start, J_END=min(j_end, 2-1)                &
                   ,KTS=1, KTE=min(nz, kde-1)                             &
                   ,NUM_TILES=1                                           &
                   ,U=u, V=v, TH=th, T=t                                  &
                   ,P=p, PI=pii, RHO=rho, MOIST=moist                     &
                   ,NUM_MOIST=1                                           &
                   ,ITIMESTEP=itimestep, DT=dt, DX=dx                     &
                   ,DX2D=dx2d, AREA2D=area2d                              &
                   ,CUDT=cudt                                             &
                   ,CURR_SECS=curr_secs, ADAPT_STEP_FLAG=adapt            &
                   ,RAINSH=rainsh, PRATESH=pratesh, NCA=nca               &
                   ,RAINSHV=rainshv                                       &
                   ,Z=z, Z_AT_W=z_at_w, DZ8W=dz8w                         &
                   ,MAVAIL=mavail, PBLH=pblh, P8W=p8w                     &
                   ,TKE_PBL=tke_pbl                                       &
                   ,CLDFRA=cldfra, CLDFRA_OLD=cldfra_old                  &
                   ,CLDFRA_OLD_MP=cldfra_old_mp                           &
                   ,CLDFRA_CONV=cldfra_conv                               &
                   ,HTOP=htop, HBOT=hbot                                  &
                   ,SHCU_PHYSICS=5                                        &
                   ,QV_CURR=qv, QC_CURR=qc, QR_CURR=qr                    &
                   ,DLF2=dlf2                                             &
                   ,RQVSHTEN=rqvshten, RQCSHTEN=rqcshten                  &
                   ,RQRSHTEN=rqrshten                                     &
                   ,RUSHTEN=rushten, RVSHTEN=rvshten                      &
                   ,RTHSHTEN=rthshten, RTHRATEN=rthraten                  &
                   ,RTHBLTEN=rthblten                                     &
                   ,HT=ht                                                 &
                   ,SHFRC3D=shfrc3d                                       &
                   ,IS_CAMMGMP_USED=is_cammgmp_used                       &
                   ,ZNU=znu                                               &
                   ,W=w, XLAND=xland                                      &
                   ,RDCASHTEN=rdcashten, RQCDCSHTEN=rqcdcshten            &
                   ,W0AVG=w0avg                                           &
                   ,clddpthb=clddpthb, cldtopb=cldtopb                    &
                   ,cldareaa=cldareaa, cldareab=cldareab                  &
                   ,cldliqa=cldliqa, cldliqb=cldliqb                      &
                   ,cldfra_sh=cldfra_sh, ca_rad=ca_rad, cw_rad=cw_rad     &
                   ,wub=wub, pblmax=pblmax, xlong=xlong                   &
                   ,rainshvb=rainshvb, capesave=capesave                  &
                   ,radsave=radsave, ainckfsa=ainckfsa                    &
                   ,ltopb=ltopb, kdcldtop=kdcldtop                        &
                   ,kdcldbas=kdcldbas                                     &
                   ,el_pbl=el_pbl                                         &
                   ,rthratenlw=rthratenlw, rthratensw=rthratensw          &
                   ,exch_h=exch_h                                         &
                   ,dnw=dnw, XTIME=xtime, XTIME1=xtime1                   &
                   ,GMT=gmt                                               &
                   ,qke=qke                                               &
                   ,PBLHAVG=pblhavg, TKEAVG=tkeavg                        &
                   ,BL_PBL_PHYSICS=ibl                                    &
                   ,multi_perturb=0                                       &
                   ,pert_deng=pert_deng                                   &
                   ,pert_deng_qv=0., pert_deng_qc=0.                      &
                   ,pert_deng_t=0., pert_deng_w=0.                        &
                                                                          )

      WRITE(11) deng_oracle_fatal, deng_oracle_messages
      WRITE(11) rthshten(:,1:nz,1), rqvshten(:,1:nz,1), rqcshten(:,1:nz,1), &
                rqrshten(:,1:nz,1), rushten(:,1:nz,1), rvshten(:,1:nz,1), &
                rdcashten(:,1:nz,1), rqcdcshten(:,1:nz,1), cldfra_sh(:,1:nz,1), &
                ca_rad(:,1:nz,1), cw_rad(:,1:nz,1), w0avg(:,1:nz,1), &
                tkeavg(:,1:nz,1), cldareaa(:,1:nz,1), cldareab(:,1:nz,1), &
                cldliqa(:,1:nz,1), cldliqb(:,1:nz,1), wub(:,1:nz,1)
      WRITE(11) rainsh(:,1), rainshv(:,1), rainshvb(:,1), pblmax(:,1), &
                capesave(:,1), xtime1(:,1), radsave(:,1), clddpthb(:,1), &
                cldtopb(:,1), mavail(:,1), pblhavg(:,1), pratesh(:,1)
      WRITE(11) ltopb(:,1), kdcldtop(:,1), kdcldbas(:,1)
      WRITE(11) ainckfsa(:,:,1)
    END DO

    DEALLOCATE(dnw, znu, ht, xlong, pblh, rainsh, rainshv, rainshvb, pblmax, &
               capesave, xtime1, radsave, clddpthb, cldtopb, mavail, pblhavg, &
               pratesh, nca, htop, hbot, xland, dx2d, area2d, ltopb, kdcldtop, &
               kdcldbas, ainckfsa, u, v, th, t, qv, qc, qr, p, pii, rho, dz8w, &
               tke_pbl, qke, el_pbl, exch_h, rthratenlw, rthratensw, z_at_w, w, &
               z, p8w, w0avg, tkeavg, cldareaa, cldareab, cldliqa, cldliqb, wub, &
               rthshten, rqvshten, rqcshten, rqrshten, rushten, rvshten, &
               rdcashten, rqcdcshten, cldfra_sh, ca_rad, cw_rad, cldfra, &
               cldfra_old, cldfra_old_mp, cldfra_conv, dlf2, shfrc3d, rthraten, &
               rthblten, moist)
  END DO
  CLOSE(10)
  CLOSE(11)
END PROGRAM run_deng_oracle
