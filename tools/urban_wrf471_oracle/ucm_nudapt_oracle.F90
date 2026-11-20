! Gridded urban morphology (NUDAPT / WUDAPT URB_PARAM) oracle, WRF v4.7.1.
!
! Drives the byte-unmodified phys/module_sf_urban.F, as build_ucm.sh does,
! in two modes:
!
!   run_ucm_nudapt column TABLE VARIANT OUT_PREFIX [CH TS]
!     `urban` with MH_URB > 0, i.e. its NUDAPT arm (module_sf_urban.F:
!     649-788): building and street width from LP/LB/HGT/FRC, ZR/R/RW/HGT/
!     SIGMA_ZED replaced, lambda_f chosen by wind sector, Macdonald (1998)
!     ZDC/Z0C/Z0R/Z0HC and the sky-view-factor loop.  Every urban category
!     of the table, eight forcing scenarios of ucm_column_oracle.F90, four
!     morphologies (VARIANT "probe": four degenerate ones), six carried
!     steps whose wind turns through every lambda_f sector.
!
!     WRF UNDEFINED READ, given defined behaviour.  The sector reads
!     ATAN2(U10,V10) at :705, but U10 and V10 are `urban`'s INTENT(OUT)
!     arguments, written only at its end (:1640-1641).  module_sf_noahdrv.F
!     passes its locals U10_URB/V10_URB (:1448), so WRF reads the 10 m wind
!     the PREVIOUS urban column of the loop left there (undefined on the
!     first).  This driver stores the column's own first-level wind U1/V1
!     into U10/V10 before each call, and the port reads ATAN2(U1,V1): the
!     direction the column's own U10 takes (U10 = U1*PSIX10/PSIX, :1640,
!     with PSIX10/PSIX > 0).
!
!   run_ucm_nudapt init OPT LCZ OUT_PREFIX
!     urban_param_init + urban_var_init (:2557-3052) over one row of
!     columns carrying gridded morphology, for sf_urban_physics OPT: the
!     morphology keep/zero rule (:2788-2833) and the FRC_URB2D default.
!
! Writes OUT_PREFIX.csv (and, for column, OUT_PREFIX-table.csv and
! OUT_PREFIX-switches.csv, as ucm_column_oracle.F90 does).

program run_ucm_nudapt
  use module_sf_urban
  implicit none

  integer, parameter :: nsoil = 4
  integer, parameter :: nstep = 6
  integer, parameter :: nscen = 8
  integer, parameter :: nmorph = 4
  real :: dzr(nsoil), dzb(nsoil), dzg(nsoil)
  character(len=16) :: mode, table, variant
  character(len=512) :: prefix, arg
  integer :: use_lcz, ut, sc, st, im, k, u, nrow, i
  integer :: ov(2)

  logical :: lsolar
  integer :: utype, jmonth
  real :: ta, qa, ua, u1, v1, ssg, ssgd, ssgq, llg, rain, rhoo, za
  real :: declin, cosz, omg, xlat, delt, znt, chs, chs2
  real :: tr, tb, tg, tc, qc, uc
  real :: trl(nsoil), tbl(nsoil), tgl(nsoil)
  real :: xxxr, xxxb, xxxg, xxxc
  real :: ts, qs, sh, lh, lh_kinematic, sw, alb, lw, g, rn, psim, psih
  real :: gz1oz0, u10, v10, th2, q2, ust
  real :: cmr_urb, chr_urb, cmc_urb, chc_urb, cmgr_urb, chgr_urb
  real :: mh_urb, stdh_urb, lf_urb(4), lp_urb, hgt_urb, frc_urb, lb_urb
  real :: zo_check
  real :: cmcr, tgr, tgrl(nsoil), smr(nsoil)
  real :: drelr, drelb, drelg, flxhumr, flxhumb, flxhumg
  real :: lf_urb_s, z0_urb, vegfrac
  real :: tsurf0, tlay0(3)
  real :: m_mh, m_stdh, m_lf(4), m_lp, m_hgt, m_frc, m_lb
  ! the wind's heading per step, degrees clockwise from north (u = sin)
  real, parameter :: heading(12) = (/ 0., 20., 45., 80., 90., 135., &
                                      170., 180., 200., 250., 290., 330. /)

  if (command_argument_count() < 4) then
    write(*, '(A)') 'usage: run_ucm_nudapt column TABLE VARIANT OUT_PREFIX [CH TS]'
    write(*, '(A)') '       run_ucm_nudapt init OPT LCZ OUT_PREFIX'
    stop 2
  end if
  call get_command_argument(1, mode)
  if (trim(mode) == 'init') then
    call init_mode()
    stop
  end if
  call get_command_argument(2, table)
  call get_command_argument(3, variant)
  call get_command_argument(4, prefix)
  ov = -9
  do i = 1, 2
    if (command_argument_count() >= 4 + i) then
      call get_command_argument(4 + i, arg)
      read(arg, *) ov(i)
    end if
  end do

  if (trim(table) == 'nlcd') then
    use_lcz = 0
  else if (trim(table) == 'lcz') then
    use_lcz = 1
  else
    write(*, '(A)') 'TABLE must be nlcd or lcz'
    stop 2
  end if

  dzr = 0.
  dzb = 0.
  dzg = 0.
  call urban_param_init(dzr, dzb, dzg, nsoil, 1, use_lcz, .false.)
  if (ov(1) > -8) ch_scheme_data = ov(1)
  if (ov(2) > -8) ts_scheme_data = ov(2)

  call write_table()

  open(newunit=u, file=trim(prefix)//'.csv', status='replace', action='write')
  call write_header()
  nrow = 0
  lsolar = .false.

  do ut = 1, icate
    do im = 1, nmorph
      do sc = 1, nscen
        call morphology()
        call scenario_init()
        do st = 1, nstep
          call scenario_forcing()
          ! the defined U10/V10 read (header): the column's own wind
          u10 = u1
          v10 = v1
          call write_inputs()
          call urban(lsolar, nsoil, nsoil, nsoil, dzr, dzb, dzg,           &
                     utype, ta, qa, ua, u1, v1, ssg, ssgd, ssgq, llg, rain,  &
                     rhoo, za, declin, cosz, omg, xlat, delt, znt,           &
                     chs, chs2,                                              &
                     tr, tb, tg, tc, qc, uc,                                 &
                     trl, tbl, tgl,                                          &
                     xxxr, xxxb, xxxg, xxxc,                                 &
                     ts, qs, sh, lh, lh_kinematic,                           &
                     sw, alb, lw, g, rn, psim, psih,                         &
                     gz1oz0,                                                 &
                     cmr_urb, chr_urb, cmc_urb, chc_urb,                     &
                     u10, v10, th2, q2, ust, mh_urb, stdh_urb, lf_urb,       &
                     lp_urb, hgt_urb, frc_urb, lb_urb, zo_check,             &
                     cmcr, tgr, tgrl, smr, cmgr_urb, chgr_urb, jmonth,       &
                     drelr, drelb, drelg, flxhumr, flxhumb, flxhumg,         &
                     lf_urb_s, z0_urb, vegfrac)
          call write_outputs()
          flush(u)
          nrow = nrow + 1
        end do
      end do
    end do
  end do
  close(u)
  write(*, '(A,A,A,I0,A)') 'run_ucm_nudapt ', trim(variant), ': ', nrow, ' rows'

contains

  ! The column's gridded morphology.  "main": four realistic cells (the
  ! BEP width formula, LB <= LP falling back to the table widths, a tall
  ! core and a sparse suburb).  "probe": four degenerate cells whose WRF
  ! arithmetic the port's ingest has to judge (woof/core/urban_param.py).
  subroutine morphology()
    if (trim(variant) == 'probe') then
      select case (im)
      case (1)          ! LP > FRC: the street width goes negative
        m_mh = 10.; m_stdh = 3.; m_lp = 0.7; m_hgt = 9.; m_lb = 1.6
        m_lf = (/ 0.20, 0.25, 0.22, 0.18 /); m_frc = 0.6
      case (2)          ! STDH = 0: lambda_fr = 0, Z0R = 0
        m_mh = 10.; m_stdh = 0.; m_lp = 0.3; m_hgt = 9.; m_lb = 1.1
        m_lf = (/ 0.20, 0.20, 0.20, 0.20 /); m_frc = 0.8
      case (3)          ! one sector with LF = 0: Z0C = 0 there
        m_mh = 10.; m_stdh = 3.; m_lp = 0.3; m_hgt = 9.; m_lb = 1.1
        m_lf = (/ 0.00, 0.20, 0.00, 0.20 /); m_frc = 0.8
      case (4)          ! LP = 0 below LB: 0*Inf in the street width
        m_mh = 10.; m_stdh = 3.; m_lp = 0.0; m_hgt = 9.; m_lb = 0.5
        m_lf = (/ 0.20, 0.20, 0.20, 0.20 /); m_frc = 0.8
      end select
    else
      select case (im)
      case (1)
        m_mh = 12.; m_stdh = 4.; m_lp = 0.35; m_hgt = 10.; m_lb = 1.2
        m_lf = (/ 0.15, 0.22, 0.30, 0.18 /); m_frc = 0.8
      case (2)
        m_mh = 45.; m_stdh = 20.; m_lp = 0.5; m_hgt = 40.; m_lb = 2.8
        m_lf = (/ 0.60, 0.45, 0.50, 0.70 /); m_frc = 0.95
      case (3)          ! LB <= LP: BUILDING_WIDTH(1), STREET_WIDTH(1)
        m_mh = 8.; m_stdh = 2.; m_lp = 0.4; m_hgt = 7.; m_lb = 0.3
        m_lf = (/ 0.10, 0.10, 0.12, 0.09 /); m_frc = 0.6
      case (4)
        m_mh = 6.; m_stdh = 1.5; m_lp = 0.12; m_hgt = 5.; m_lb = 0.35
        m_lf = (/ 0.05, 0.07, 0.04, 0.06 /); m_frc = 0.5
      end select
    end if
  end subroutine morphology

  ! urban_var_init's cold start (as ucm_column_oracle.F90), with the
  ! morphology above.
  subroutine scenario_init()
    utype = ut
    jmonth = 7
    select case (sc)
    case (1)
      tsurf0 = 303.
      tlay0 = (/ 301., 298., 295. /)
    case (2)
      tsurf0 = 281.
      tlay0 = (/ 284., 287., 289. /)
    case (3)
      tsurf0 = 296.
      tlay0 = (/ 295., 294., 293. /)
    case (4)
      tsurf0 = 294.
      tlay0 = (/ 294., 293., 292. /)
    case (5)
      tsurf0 = 289.
      tlay0 = (/ 290., 291., 291. /)
    case (6)
      tsurf0 = 299.
      tlay0 = (/ 298., 296., 294. /)
    case (7)
      tsurf0 = 262.
      tlay0 = (/ 264., 268., 272. /)
      jmonth = 1
    case (8)
      tsurf0 = 286.
      tlay0 = (/ 288., 289., 290. /)
    end select

    tr = tsurf0 + 0.
    tb = tsurf0 + 0.
    tg = tsurf0 + 0.
    tc = tsurf0 + 0.
    qc = 0.01
    uc = 0.
    xxxr = 0.
    xxxb = 0.
    xxxg = 0.
    xxxc = 0.
    trl(1) = tlay0(1) + 0.
    trl(2) = 0.5 * (tlay0(1) + tlay0(2))
    trl(3) = tlay0(2) + 0.
    trl(4) = tlay0(2) + (tlay0(3) - tlay0(2)) * 0.29
    tbl = trl
    tgl(1) = tlay0(1)
    tgl(2) = tlay0(2)
    tgl(3) = tlay0(3)
    tgl(4) = tlay0(3) + 1.
    drelr = 0.
    drelb = 0.
    drelg = 0.
    flxhumr = 0.
    flxhumb = 0.
    flxhumg = 0.
    cmcr = 0.
    tgr = tsurf0 + 0.
    tgrl = trl
    smr(1) = 0.2
    smr(2) = 0.2
    smr(3) = 0.2
    smr(4) = 0.
    cmr_urb = 0.
    chr_urb = 0.
    cmc_urb = 0.
    chc_urb = 0.
    cmgr_urb = 0.
    chgr_urb = 0.
    mh_urb = m_mh
    stdh_urb = m_stdh
    lf_urb = m_lf
    lp_urb = m_lp
    hgt_urb = m_hgt
    lb_urb = m_lb
    frc_urb = m_frc
    zo_check = 0.
    lf_urb_s = 0.
    z0_urb = 0.
    vegfrac = 0.
    sw = 0.
  end subroutine scenario_init

  ! ucm_column_oracle.F90's forcing for scenarios 1-8, the first level well
  ! above the gridded canopy, and a wind that turns step by step.
  subroutine scenario_forcing()
    real :: s, th
    integer :: ih
    s = real(st - 1)
    delt = 60.
    declin = 0.35
    xlat = 38.5
    za = max(35., m_mh + 20.)
    chs = 0.02
    chs2 = 0.015
    select case (sc)
    case (1)
      ta = 299. + 0.3 * s
      qa = 0.012
      ua = 3.2
      ssg = 820. - 20. * s
      llg = 390.
      rain = 0.
      rhoo = 1.14
      omg = 0.21 + 0.004 * s
      cosz = 0.85
    case (2)
      ta = 286. - 0.2 * s
      qa = 0.007
      ua = 1.0
      ssg = 0.
      llg = 300.
      rain = 0.
      rhoo = 1.21
      omg = -2.9
      cosz = -0.4
      chs = 0.004
      chs2 = 0.003
    case (3)
      ta = 294.
      qa = 0.014
      ua = 4.5
      ssg = 250.
      llg = 400.
      rain = 0.5
      rhoo = 1.17
      omg = 0.6
      cosz = 0.6
    case (4)
      ta = 292.
      qa = 0.015
      ua = 6.
      ssg = 120.
      llg = 410.
      rain = 4.0
      rhoo = 1.18
      omg = -0.5
      cosz = 0.5
      delt = 180.
    case (5)
      ta = 290.
      qa = 0.011
      ua = 2.
      ssg = 0.
      llg = 350.
      if (st <= 2) then
        rain = 3.0
      else
        rain = 0.
      end if
      rhoo = 1.19
      omg = 2.4
      cosz = -0.3
    case (6)
      ta = 297.
      qa = 0.009
      ua = 14.
      ssg = 600.
      llg = 370.
      rain = 0.
      rhoo = 1.16
      omg = -0.8
      cosz = 0.7
    case (7)
      ta = 266.
      qa = 0.0015
      ua = 2.5
      ssg = 280.
      llg = 230.
      rain = 0.
      rhoo = 1.33
      omg = 0.05
      cosz = 0.3
      declin = -0.38
    case (8)
      ta = 291.
      qa = 0.0165
      ua = 1.5
      ssg = 0.
      llg = 360.
      rain = 0.
      rhoo = 1.19
      omg = 2.6
      cosz = -0.5
    end select
    ih = mod((st - 1) + 3 * (sc - 1) + 5 * (im - 1) + 7 * (ut - 1), 12) + 1
    th = heading(ih)
    if (th == 0.) then
      u1 = 0.
      v1 = ua
    else if (th == 90.) then
      u1 = ua
      v1 = 0.
    else if (th == 180.) then
      u1 = 0.
      v1 = -ua
    else
      u1 = ua * sin(th * 3.14159265 / 180.)
      v1 = ua * cos(th * 3.14159265 / 180.)
    end if
    ssgd = 0.8 * ssg
    ssgq = ssg - ssgd
    znt = 0.5
  end subroutine scenario_forcing

  subroutine write_table()
    integer :: ut2, v
    real :: zr, sigma_zed, z0c, z0hc, zdc, svf, r, rw, hgt, ah, capr, capb, capg
    real :: aksr, aksb, aksg, albr, albb, albg, epsr, epsb, epsg
    real :: z0r, z0b, z0g, z0hb, z0hg, betr, betb, betg
    real :: trlend, tblend, tglend, akanda, alh
    integer :: numdir, numhgt, boundr, boundb, boundg, chs_, tss_
    real :: street_direction(maxdirs), street_width(maxdirs), building_width(maxdirs)
    real :: height_bin(maxhgts), hpercent_bin(maxhgts)
    open(newunit=v, file=trim(prefix)//'-table.csv', status='replace', action='write')
    write(v, '(A)') 'utype,zr,sigma_zed,z0c,z0hc,zdc,svf,r,rw,hgt,ah,alh,' //         &
         'capr,capb,capg,aksr,aksb,aksg,albr,albb,albg,epsr,epsb,epsg,' //          &
         'z0r,z0b,z0g,z0hb,z0hg,betr,betb,betg,trlend,tblend,tglend,akanda_urban,' // &
         'frc_urb,building_width1,street_width1'
    do ut2 = 1, icate
      call read_param(ut2, zr, sigma_zed, z0c, z0hc, zdc, svf, r, rw, hgt,    &
                      ah, capr, capb, capg, aksr, aksb, aksg, albr, albb,     &
                      albg, epsr, epsb, epsg, z0r, z0b, z0g, z0hb, z0hg,      &
                      betr, betb, betg, trlend, tblend, tglend,               &
                      numdir, street_direction, street_width,                 &
                      building_width, numhgt, height_bin, hpercent_bin,       &
                      boundr, boundb, boundg, chs_, tss_, akanda, alh)
      write(v, '(I0)', advance='no') ut2
      write(v, '(38(",",ES16.8E3))') zr, sigma_zed, z0c, z0hc, zdc, svf, r, rw, &
           hgt, ah, alh, capr, capb, capg, aksr, aksb, aksg, albr, albb, albg,  &
           epsr, epsb, epsg, z0r, z0b, z0g, z0hb, z0hg, betr, betb, betg,       &
           trlend, tblend, tglend, akanda, frc_urb_tbl(ut2), building_width(1), &
           street_width(1)
    end do
    close(v)
    open(newunit=v, file=trim(prefix)//'-switches.csv', status='replace', action='write')
    write(v, '(A)') 'name,value'
    write(v, '(A,I0)') 'boundr,', boundr_data
    write(v, '(A,I0)') 'boundb,', boundb_data
    write(v, '(A,I0)') 'boundg,', boundg_data
    write(v, '(A,I0)') 'ch_scheme,', ch_scheme_data
    write(v, '(A,I0)') 'ts_scheme,', ts_scheme_data
    write(v, '(A,I0)') 'ahoption,', ahoption
    write(v, '(A,I0)') 'alhoption,', alhoption
    write(v, '(A,I0)') 'imp_scheme,', imp_scheme
    write(v, '(A,I0)') 'iri_scheme,', iri_scheme
    write(v, '(A,I0)') 'groption,', groption
    write(v, '(A,ES16.8E3)') 'fgr,', fgr
    write(v, '(A,ES16.8E3)') 'oasis,', oasis
    do k = 1, nsoil
      write(v, '(A,I0,A,ES16.8E3)') 'dzr', k, ',', dzr(k)
    end do
    do k = 1, nsoil
      write(v, '(A,I0,A,ES16.8E3)') 'dzb', k, ',', dzb(k)
    end do
    do k = 1, nsoil
      write(v, '(A,I0,A,ES16.8E3)') 'dzg', k, ',', dzg(k)
    end do
    do k = 1, 4
      write(v, '(A,I0,A,ES16.8E3)') 'dzgr', k, ',', dzgr(k)
    end do
    do k = 1, 3
      write(v, '(A,I0,A,ES16.8E3)') 'porimp', k, ',', porimp(k)
    end do
    do k = 1, 3
      write(v, '(A,I0,A,ES16.8E3)') 'dengimp', k, ',', dengimp(k)
    end do
    do k = 1, 24
      write(v, '(A,I0,A,ES16.8E3)') 'ahdiuprf', k, ',', ahdiuprf(k)
    end do
    do k = 1, 4
      write(v, '(A,I0,A,ES16.8E3)') 'alhseason', k, ',', alhseason(k)
    end do
    do k = 1, 48
      write(v, '(A,I0,A,ES16.8E3)') 'alhdiuprf', k, ',', alhdiuprf(k)
    end do
    close(v)
  end subroutine write_table

  subroutine write_header()
    character(len=4096) :: h
    h = 'variant,utype,scenario,step,jmonth,morph,ta,qa,ua,u1,v1,ssg,llg,rain,rhoo,za,' // &
        'declin,cosz,omg,xlat,delt,znt_in,chs,chs2,frc_urb' // &
        ',mh_urb,stdh_urb,lf_urb1,lf_urb2,lf_urb3,lf_urb4,lp_urb,hgt_urb,lb_urb'
    h = trim(h) // ',tr_in,tb_in,tg_in,tc_in,qc_in,uc_in' //                    &
        ',trl1_in,trl2_in,trl3_in,trl4_in,tbl1_in,tbl2_in,tbl3_in,tbl4_in' //   &
        ',tgl1_in,tgl2_in,tgl3_in,tgl4_in,xxxr_in,xxxb_in,xxxg_in,xxxc_in' //   &
        ',cmr_in,chr_in,cmc_in,chc_in,cmgr_in,chgr_in,cmcr_in,tgr_in' //        &
        ',tgrl1_in,tgrl2_in,tgrl3_in,tgrl4_in,smr1_in,smr2_in,smr3_in,smr4_in' // &
        ',drelr_in,drelb_in,drelg_in,flxhumr_in,flxhumb_in,flxhumg_in'
    h = trim(h) // ',ts,qs,sh,lh,lh_kinematic,sw,alb,lw,g,rn,psim,psih,gz1oz0' // &
        ',u10,v10,th2,q2,ust,znt'
    h = trim(h) // ',tr,tb,tg,tc,qc,uc' //                                      &
        ',trl1,trl2,trl3,trl4,tbl1,tbl2,tbl3,tbl4' //                           &
        ',tgl1,tgl2,tgl3,tgl4,xxxr,xxxb,xxxg,xxxc' //                           &
        ',cmr,chr,cmc,chc,cmgr,chgr,cmcr,tgr' //                                &
        ',tgrl1,tgrl2,tgrl3,tgrl4,smr1,smr2,smr3,smr4' //                       &
        ',drelr,drelb,drelg,flxhumr,flxhumb,flxhumg'
    write(u, '(A)') trim(h)
  end subroutine write_header

  subroutine write_inputs()
    write(u, '(A,",",I0,",",I0,",",I0,",",I0,",",I0)', advance='no') trim(variant), &
         utype, sc, st, jmonth, im
    write(u, '(19(",",ES16.8E3))', advance='no') ta, qa, ua, u1, v1, ssg, llg, rain, &
         rhoo, za, declin, cosz, omg, xlat, delt, znt, chs, chs2, frc_urb
    write(u, '(9(",",ES16.8E3))', advance='no') mh_urb, stdh_urb, lf_urb, lp_urb, &
         hgt_urb, lb_urb
    write(u, '(6(",",ES16.8E3))', advance='no') tr, tb, tg, tc, qc, uc
    write(u, '(12(",",ES16.8E3))', advance='no') trl, tbl, tgl
    write(u, '(4(",",ES16.8E3))', advance='no') xxxr, xxxb, xxxg, xxxc
    write(u, '(8(",",ES16.8E3))', advance='no') cmr_urb, chr_urb, cmc_urb, chc_urb, &
         cmgr_urb, chgr_urb, cmcr, tgr
    write(u, '(8(",",ES16.8E3))', advance='no') tgrl, smr
    write(u, '(6(",",ES16.8E3))', advance='no') drelr, drelb, drelg, flxhumr, flxhumb, flxhumg
  end subroutine write_inputs

  subroutine write_outputs()
    write(u, '(19(",",ES16.8E3))', advance='no') ts, qs, sh, lh, lh_kinematic, sw, &
         alb, lw, g, rn, psim, psih, gz1oz0, u10, v10, th2, q2, ust, znt
    write(u, '(6(",",ES16.8E3))', advance='no') tr, tb, tg, tc, qc, uc
    write(u, '(12(",",ES16.8E3))', advance='no') trl, tbl, tgl
    write(u, '(4(",",ES16.8E3))', advance='no') xxxr, xxxb, xxxg, xxxc
    write(u, '(8(",",ES16.8E3))', advance='no') cmr_urb, chr_urb, cmc_urb, chc_urb, &
         cmgr_urb, chgr_urb, cmcr, tgr
    write(u, '(8(",",ES16.8E3))', advance='no') tgrl, smr
    write(u, '(6(",",ES16.8E3))') drelr, drelb, drelg, flxhumr, flxhumb, flxhumg
  end subroutine write_outputs

  ! urban_var_init (module_physics_init.F:3294-3345's call) over one row of
  ! columns that carry gridded morphology, for sf_urban_physics = opt.
  subroutine init_mode()
    integer, parameter :: ncol = 8, kms = 1, kme = 5, num_urban_hi = 15
    ! MODIFIED_IGBP_MODIS_NOAH: ISURBAN 13, VEGPARM LCZ_1..LCZ_11 = 51..61
    integer, parameter :: isurban = 13
    integer, parameter :: lcz(11) = (/ 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61 /)
    integer :: opt, wudapt, v, j
    integer :: ndm, nz, ng, nwr, nf, ngb, nbui, ngr
    integer :: zrd, zwd, gd, zd, zdf, bd, wd, gbd, fbd, zgrd
    integer :: ivgtyp(ncol, 1)
    real :: tsk(ncol, 1), tmn(ncol, 1), frc(ncol, 1), frc_in(ncol, 1)
    real :: tslb(ncol, nsoil, 1), smois(ncol, nsoil, 1)
    real, dimension(ncol, 1) :: xxxr2, xxxb2, xxxg2, xxxc2, tr2, tb2, tg2, tc2, qc2
    real, dimension(ncol, 1) :: sh2, lh2, g2, rn2, ts2, lf_ac, sf_ac, cm_ac
    real, dimension(ncol, 1) :: sfvent, lfvent, ep_pv, qgr3, tgr3, draingr
    real, dimension(ncol, 1) :: lp, lb, hgt, mh, stdh, cmcr2, tgr2
    real, dimension(ncol, 1) :: lp0, lb0, hgt0, mh0, stdh0
    real, dimension(ncol, 1) :: drelr2, drelb2, drelg2, flxhumr2, flxhumb2, flxhumg2
    real, dimension(ncol, nsoil, 1) :: trl2, tbl2, tgl2, tgrl2, smr2
    real, dimension(ncol, 4, 1) :: lf, lf0
    real, dimension(ncol, num_urban_hi, 1) :: hi, hi0
    real, dimension(ncol, kms:kme, 1) :: a_u, a_v, a_t, a_q, a_e, b_u, b_v
    real, dimension(ncol, kms:kme, 1) :: b_t, b_q, b_e, vl, dlg, sf, dl_u
    integer :: utype2(ncol, 1)
    real, allocatable, dimension(:, :, :) :: trb, tw1, tw2, tgb, tlev, qlev
    real, allocatable, dimension(:, :, :) :: tw1lev, tw2lev, tglev, tflev
    real, allocatable, dimension(:, :, :) :: sfwin1, sfwin2, sfw1, sfw2, sfr
    real, allocatable, dimension(:, :, :) :: sfg, t_pv, trv, qr4, drain
    real, allocatable, dimension(:, :, :) :: sfrv, lfrv, dgr, dg, lfr, lfg

    call get_command_argument(2, arg)
    read(arg, *) opt
    call get_command_argument(3, arg)
    read(arg, *) wudapt
    call get_command_argument(4, prefix)
    ! WRF_URBAN_DIMENSIONS (woof/core/urban_state.py), check_a_mundo's maps
    ndm = 1; nz = 1; ng = 1; nwr = 1; nf = 1; ngb = 1; nbui = 1; ngr = 1
    if (opt == 2) then
      ndm = 2; nz = 18; ng = 10; nwr = 10
    else if (opt == 3) then
      ndm = 2; nz = 18; ng = 10; nwr = 10; nf = 10; ngb = 10; nbui = 15; ngr = 10
    end if
    zrd = ndm * nwr * nz
    zwd = ndm * nwr * nz * nbui
    gd = ndm * ng
    zd = ndm * nz * nbui
    zdf = ndm * nz
    bd = nz * nbui
    wd = ndm * nz * nbui
    gbd = ndm * ngb * nbui
    fbd = ndm * (nz - 1) * nf * nbui
    zgrd = ndm * ngr * nz
    allocate(trb(ncol, zrd, 1), tw1(ncol, zwd, 1), tw2(ncol, zwd, 1))
    allocate(tgb(ncol, gd, 1), tlev(ncol, bd, 1), qlev(ncol, bd, 1))
    allocate(tw1lev(ncol, wd, 1), tw2lev(ncol, wd, 1), tglev(ncol, gbd, 1))
    allocate(tflev(ncol, fbd, 1), sfwin1(ncol, wd, 1), sfwin2(ncol, wd, 1))
    allocate(sfw1(ncol, zd, 1), sfw2(ncol, zd, 1), sfr(ncol, zdf, 1))
    allocate(sfg(ncol, ndm, 1), t_pv(ncol, zdf, 1), trv(ncol, zgrd, 1))
    allocate(qr4(ncol, zgrd, 1), drain(ncol, zdf, 1), sfrv(ncol, zdf, 1))
    allocate(lfrv(ncol, zdf, 1), dgr(ncol, zdf, 1), dg(ncol, ndm, 1))
    allocate(lfr(ncol, zdf, 1), lfg(ncol, ndm, 1))

    dzr = 0.0; dzb = 0.0; dzg = 0.0
    call urban_param_init(dzr, dzb, dzg, nsoil, opt, wudapt, .false.)

    ! Columns: urban with HGT_URB2D > 0 (kept), urban with HGT_URB2D = 0 but
    ! the rest nonzero (zeroed: "USING DEFAULT URBAN MORPHOLOGY"), urban with
    ! a negative HGT (zeroed), and non-urban columns carrying morphology
    ! (zeroed).
    if (wudapt == 0) then
      ivgtyp(:, 1) = (/ isurban, isurban, isurban, lcz(1), lcz(3), 10, 14, 17 /)
    else
      ivgtyp(:, 1) = (/ isurban, lcz(2), lcz(5), lcz(1), lcz(10), 10, 14, 17 /)
    end if
    frc_in(:, 1) = (/ 0.8, 0.0, 0.95, 0.6, 1.0, 0.4, 0.0, 0.3 /)
    hgt0(:, 1) = (/ 10., 0., 40., -2., 7.5, 12., 9., 5. /)
    do i = 1, ncol
      lp0(i, 1) = 0.1 + 0.05 * real(i)
      lb0(i, 1) = 0.4 + 0.2 * real(i)
      mh0(i, 1) = 6. + 3. * real(i)
      stdh0(i, 1) = 1. + 0.5 * real(i)
      do j = 1, 4
        lf0(i, j, 1) = 0.02 * real(i) + 0.03 * real(j)
      end do
      do j = 1, num_urban_hi
        hi0(i, j, 1) = 0.01 * real(mod(i + j, 7)) + 0.001 * real(j)
      end do
      tsk(i, 1) = 283.17 + 1.37 * real(i)
      tmn(i, 1) = 281.3 + 0.11 * real(i)
      tslb(i, 1, 1) = 284.63 + 0.73 * real(i)
      tslb(i, 2, 1) = 283.29 + 0.41 * real(i)
      tslb(i, 3, 1) = 282.07 + 0.19 * real(i)
      tslb(i, 4, 1) = 281.51 + 0.07 * real(i)
      smois(i, 1, 1) = 0.213 + 0.017 * real(i)
      smois(i, 2, 1) = 0.227 + 0.013 * real(i)
      smois(i, 3, 1) = 0.241 + 0.011 * real(i)
      smois(i, 4, 1) = 0.263 + 0.007 * real(i)
    end do
    ! real.exe's split: options 2/3 carry no MH/STDH/LF, option 1 no HI
    ! (module_initialize_real.F:3190-3215); the others are Registry zero.
    lp = lp0; lb = lb0; hgt = hgt0; frc = frc_in
    mh = 0.; stdh = 0.; lf = 0.; hi = 0.
    if (opt == 1) then
      mh = mh0; stdh = stdh0; lf = lf0
    else
      hi = hi0
    end if
    utype2 = -7

    call urban_var_init(ISURBAN=isurban, TSURFACE0_URB=tsk, &
      TLAYER0_URB=tslb, TDEEP0_URB=tmn, IVGTYP=ivgtyp, &
      ims=1, ime=ncol, jms=1, jme=1, kms=kms, kme=kme, &
      num_soil_layers=nsoil, &
      LCZ_1=lcz(1), LCZ_2=lcz(2), LCZ_3=lcz(3), LCZ_4=lcz(4), &
      LCZ_5=lcz(5), LCZ_6=lcz(6), LCZ_7=lcz(7), LCZ_8=lcz(8), &
      LCZ_9=lcz(9), LCZ_10=lcz(10), LCZ_11=lcz(11), &
      restart=.false., sf_urban_physics=opt, &
      XXXR_URB2D=xxxr2, XXXB_URB2D=xxxb2, XXXG_URB2D=xxxg2, XXXC_URB2D=xxxc2, &
      TR_URB2D=tr2, TB_URB2D=tb2, TG_URB2D=tg2, TC_URB2D=tc2, QC_URB2D=qc2, &
      TRL_URB3D=trl2, TBL_URB3D=tbl2, TGL_URB3D=tgl2, &
      SH_URB2D=sh2, LH_URB2D=lh2, G_URB2D=g2, RN_URB2D=rn2, TS_URB2D=ts2, &
      num_urban_ndm=ndm, urban_map_zrd=zrd, urban_map_zwd=zwd, &
      urban_map_gd=gd, urban_map_zd=zd, urban_map_zdf=zdf, &
      urban_map_bd=bd, urban_map_wd=wd, urban_map_gbd=gbd, &
      urban_map_fbd=fbd, urban_map_zgrd=zgrd, num_urban_hi=num_urban_hi, &
      TRB_URB4D=trb, TW1_URB4D=tw1, TW2_URB4D=tw2, TGB_URB4D=tgb, &
      TLEV_URB3D=tlev, QLEV_URB3D=qlev, TW1LEV_URB3D=tw1lev, &
      TW2LEV_URB3D=tw2lev, TGLEV_URB3D=tglev, TFLEV_URB3D=tflev, &
      SF_AC_URB3D=sf_ac, LF_AC_URB3D=lf_ac, CM_AC_URB3D=cm_ac, &
      SFVENT_URB3D=sfvent, LFVENT_URB3D=lfvent, &
      SFWIN1_URB3D=sfwin1, SFWIN2_URB3D=sfwin2, &
      SFW1_URB3D=sfw1, SFW2_URB3D=sfw2, SFR_URB3D=sfr, SFG_URB3D=sfg, &
      EP_PV_URB3D=ep_pv, T_PV_URB3D=t_pv, &
      TRV_URB4D=trv, QR_URB4D=qr4, QGR_URB3D=qgr3, TGR_URB3D=tgr3, &
      DRAIN_URB4D=drain, DRAINGR_URB3D=draingr, SFRV_URB3D=sfrv, &
      LFRV_URB3D=lfrv, DGR_URB3D=dgr, DG_URB3D=dg, LFR_URB3D=lfr, &
      LFG_URB3D=lfg, SMOIS_URB=smois, &
      LP_URB2D=lp, HI_URB2D=hi, LB_URB2D=lb, HGT_URB2D=hgt, MH_URB2D=mh, &
      STDH_URB2D=stdh, LF_URB2D=lf, &
      CMCR_URB2D=cmcr2, TGR_URB2D=tgr2, TGRL_URB3D=tgrl2, SMR_URB3D=smr2, &
      DRELR_URB2D=drelr2, DRELB_URB2D=drelb2, DRELG_URB2D=drelg2, &
      FLXHUMR_URB2D=flxhumr2, FLXHUMB_URB2D=flxhumb2, &
      FLXHUMG_URB2D=flxhumg2, &
      A_U_BEP=a_u, A_V_BEP=a_v, A_T_BEP=a_t, A_Q_BEP=a_q, A_E_BEP=a_e, &
      B_U_BEP=b_u, B_V_BEP=b_v, B_T_BEP=b_t, B_Q_BEP=b_q, B_E_BEP=b_e, &
      DLG_BEP=dlg, DL_U_BEP=dl_u, SF_BEP=sf, VL_BEP=vl, &
      FRC_URB2D=frc, UTYPE_URB2D=utype2, USE_WUDAPT_LCZ=wudapt)

    open(newunit=v, file=trim(prefix)//'.csv', status='replace', action='write')
    write(v, '(A)', advance='no') 'col,ivgtyp,utype,frc_in,frc,lp_in,lb_in,hgt_in,mh_in,stdh_in,' // &
         'lp,lb,hgt,mh,stdh'
    do j = 1, 4
      write(v, '(A,I0,A,I0)', advance='no') ',lf', j, '_in,lf', j
    end do
    do j = 1, num_urban_hi
      write(v, '(A,I0,A,I0)', advance='no') ',hi', j, '_in,hi', j
    end do
    write(v, '(A)') ''
    do i = 1, ncol
      write(v, '(I0,",",I0,",",I0)', advance='no') i, ivgtyp(i, 1), utype2(i, 1)
      write(v, '(12(",",ES16.8E3))', advance='no') frc_in(i, 1), frc(i, 1), &
           lp0(i, 1), lb0(i, 1), hgt0(i, 1), mh0(i, 1), stdh0(i, 1), &
           lp(i, 1), lb(i, 1), hgt(i, 1), mh(i, 1), stdh(i, 1)
      do j = 1, 4
        write(v, '(2(",",ES16.8E3))', advance='no') merge(lf0(i, j, 1), 0., opt == 1), lf(i, j, 1)
      end do
      do j = 1, num_urban_hi
        write(v, '(2(",",ES16.8E3))', advance='no') merge(hi0(i, j, 1), 0., opt /= 1), hi(i, j, 1)
      end do
      write(v, '(A)') ''
    end do
    close(v)
  end subroutine init_mode

end program run_ucm_nudapt
