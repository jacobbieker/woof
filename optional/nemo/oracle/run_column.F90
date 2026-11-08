program run_nemo_column_oracle
  ! Drive NEMO 5.0.2's tracer column through stpmlf.F90's step, in its order,
  ! for six steps of rn_Dt = 900 s with the atmosphere coupled every two
  ! steps: sbc (sbc_ssm's nn_fsbc = 1 lines, then sbc_cpl_rcv), eos_rab and
  ! bn2 for the before and now states, zdf_phy's TKE path (zdf_sh2, zdf_drg
  ! 'BOTTOM', zdf_tke, the avt/avm/avs copies), tra_sbc, tra_qsr, tra_zdf,
  ! tra_atf_qco, the time-level swap, sbc_cpl_snd's sea-surface temperature
  ! and the Euler-to-leapfrog switch.  The NEMO routines are the pinned files
  ! compiled whole (sbc_cpl_rcv is cut byte for byte, see
  ! column_harness.F90); this program only sets the state up, transcribes the
  ! glue lines of stpmlf.F90, sbcssm.F90, zdfphy.F90 and sbccpl.F90 named at
  ! each step, and dumps every word.  The namelists are namelist_ref's blocks
  ! (extracted by the build) plus the overrides passed as the third argument;
  ! the fourth argument is ln_traqsr (T or F).  The grid is a z-coordinate
  ! (1-D levels, full steps) on 5 x 4 columns with NEMO 5.0's two-cell halo
  ! repeating the edge columns, one land column and one shelf column; the ocean
  ! is at rest (no momentum is stepped).
  use par_kind
  use oce
  use dom_oce
  use sbc_oce
  use phycst
  use zdf_oce
  use zdfdrg
  use in_out_manager
  use eosbn2
  use traqsr
  use zdfsh2
  use zdftke
  use trazdf
  use trasbc
  use traatf_qco
  use sbccpl_harness
  implicit none
  integer, parameter :: nx = 9, ny = 8, nz = 21, nstep = 6, ncpl = 2
  real(wp), allocatable :: rab_b(:,:,:,:), rab_n(:,:,:,:), sh2(:,:,:), zts(:,:,:), ztmp1(:,:)
  real(wp), allocatable :: e3t_1d(:), e3w_1d(:), gdept_1d(:)
  real(4), allocatable :: f4(:,:,:), sst_out(:,:)
  integer, allocatable :: mbot(:,:)
  integer :: ji, jj, jk, kt, jw, u, ug, us, uf, n, ic, jc
  integer :: Nbb, Nnn, Naa, Nrhs
  character(len=1024) :: dir, cfg, ref, qsrflag
  character(len=*), parameter :: fr = '(ES25.17E3)'

  call get_command_argument(1, dir)
  call get_command_argument(2, ref)
  call get_command_argument(3, cfg)
  call get_command_argument(4, qsrflag)
  if (len_trim(qsrflag) == 0) error stop 'usage: run_column OUTDIR NAMELIST_REF_BLOCKS "&nam... /" T|F'
  open(newunit=u, file=trim(ref), status='old', access='stream', form='unformatted')
  inquire(unit=u, size=n)
  allocate(character(len=n) :: numnam_ref)
  read(u) numnam_ref
  close(u)
  numnam_cfg = trim(cfg)
  ln_traqsr = trim(qsrflag) == 'T'

  jpi = nx; jpj = ny; jpk = nz; jpkm1 = nz - 1
  ! NEMO 5.0 runs with a two-cell halo (mppini.F90 :185, nn_hls = MAX(2, nn_hls))
  nn_hls = 2
  Nis0 = 3; Nie0 = nx - 2; Njs0 = 3; Nje0 = ny - 2
  ntsi = Nis0; ntei = Nie0; ntsj = Njs0; ntej = Nje0
  ! domain.F90 dom_nam (:288-289): MLF sets rDt = 2 rn_Dt before the first step
  rn_Dt = 900._wp
  rDt = 2._wp * rn_Dt
  r1_Dt = 1._wp / rDt
  l_1st_euler = .true.
  rn_atfp = 0.1_wp
  Nbb = 1; Nnn = 2; Naa = 3; Nrhs = Naa

  allocate(tmask(nx,ny,nz), umask(nx,ny,nz), vmask(nx,ny,nz), wmask(nx,ny,nz), wumask(nx,ny,nz), wvmask(nx,ny,nz))
  allocate(e3t_3d(nx,ny,nz), e3w_3d(nx,ny,nz), e3uw_3d(nx,ny,nz), e3vw_3d(nx,ny,nz), gdepw_3d(nx,ny,nz), gdept_3d(nx,ny,nz))
  allocate(e3t_0(nx,ny,nz), gdepw_0(nx,ny,nz), gdepw_1d(nz), e3t_1d(nz), e3w_1d(nz), gdept_1d(nz))
  allocate(ssmask(nx,ny), gphit(nx,ny), mbkt(nx,ny), mikt(nx,ny), mbot(nx,ny))
  allocate(r3t(nx,ny,3), r1_e1e2t(nx,ny), r3t_f(nx,ny))
  allocate(uu(nx,ny,nz,3), vv(nx,ny,nz,3), rn2(nx,ny,nz), rn2b(nx,ny,nz), wi(nx,ny,nz))
  allocate(ts(nx,ny,nz,jpts,jpt), ssh(nx,ny,3), fraqsr_1lev(nx,ny))
  allocate(utau(nx,ny), vtau(nx,ny), taum(nx,ny), wndm(nx,ny), fr_i(nx,ny), qns(nx,ny), qsr(nx,ny))
  allocate(emp(nx,ny), sfx(nx,ny), rnf(nx,ny), sst_m(nx,ny), sss_m(nx,ny), cloud_fra(nx,ny))
  allocate(sbc_tsc(nx,ny,jpts), sbc_tsc_b(nx,ny,jpts), qsr_hc(nx,ny,nz), qsr_hc_b(nx,ny,nz))
  allocate(rab_b(nx,ny,nz,jpts), rab_n(nx,ny,nz,jpts), zts(nx,ny,jpts), ztmp1(nx,ny))
  allocate(en(Nis0:Nie0,Njs0:Nje0,nz), avt_k(Nis0:Nie0,Njs0:Nje0,nz), avm_k(nx,ny,nz))
  allocate(avtb_2d(nx,ny), avtb(nz), avmb(nz), avt(nx,ny,nz), avs(nx,ny,nz), avm(nx,ny,nz))
  allocate(sh2(Nis0:Nie0,Njs0:Nje0,nz), f4(Nis0:Nie0,Njs0:Nje0,6), sst_out(nx,ny))
  allocate(xcplmask(Nis0:Nie0,Njs0:Nje0,0:nn_cplmodel), incoming(Nis0:Nie0,Njs0:Nje0,jprcv))

  ! z-coordinate levels (usrdef_zgr style: 1-D, full steps)
  gdepw_1d(1) = 0._wp
  do jk = 1, nz
    e3t_1d(jk) = 2._wp + 0.6_wp * real(jk - 1, wp)**2
    if (jk > 1) gdepw_1d(jk) = gdepw_1d(jk-1) + e3t_1d(jk-1)
    gdept_1d(jk) = gdepw_1d(jk) + 0.5_wp * e3t_1d(jk)
  end do
  e3w_1d(1) = 0.5_wp * e3t_1d(1)
  do jk = 2, nz
    e3w_1d(jk) = gdept_1d(jk) - gdept_1d(jk-1)
  end do
  ! bottom level per interior column, then the halo repeats the edge
  do jj = Njs0, Nje0
    do ji = Nis0, Nie0
      mbot(ji,jj) = nz - 2 - mod(ji * 3 + jj * 5, 7)
      gphit(ji,jj) = -50._wp + 23._wp * real(jj, wp) + 0.7_wp * real(ji, wp)
    end do
  end do
  mbot(4,3) = 3                                   ! shelf
  mbot(3,4) = 0                                   ! land
  do jj = 1, ny
    do ji = 1, nx
      ic = min(max(ji, Nis0), Nie0); jc = min(max(jj, Njs0), Nje0)
      mbot(ji,jj) = mbot(ic,jc); gphit(ji,jj) = gphit(ic,jc)
      mbkt(ji,jj) = max(mbot(ji,jj), 1)
      mikt(ji,jj) = 1
      do jk = 1, nz
        tmask(ji,jj,jk) = merge(1._wp, 0._wp, jk <= mbot(ji,jj))
        e3t_3d(ji,jj,jk) = e3t_1d(jk); e3t_0(ji,jj,jk) = e3t_1d(jk)
        e3w_3d(ji,jj,jk) = e3w_1d(jk); e3uw_3d(ji,jj,jk) = e3w_1d(jk); e3vw_3d(ji,jj,jk) = e3w_1d(jk)
        gdept_3d(ji,jj,jk) = gdept_1d(jk)
        gdepw_3d(ji,jj,jk) = gdepw_1d(jk); gdepw_0(ji,jj,jk) = gdepw_1d(jk)
      end do
    end do
  end do
  do jj = 1, ny
    do ji = 1, nx
      ssmask(ji,jj) = tmask(ji,jj,1)
      do jk = 1, nz
        umask(ji,jj,jk) = tmask(ji,jj,jk) * tmask(min(ji+1,nx),jj,jk)
        vmask(ji,jj,jk) = tmask(ji,jj,jk) * tmask(ji,min(jj+1,ny),jk)
      end do
      wmask(ji,jj,1) = tmask(ji,jj,1); wumask(ji,jj,1) = umask(ji,jj,1); wvmask(ji,jj,1) = vmask(ji,jj,1)
      do jk = 2, nz
        wmask(ji,jj,jk) = tmask(ji,jj,jk) * tmask(ji,jj,jk-1)
        wumask(ji,jj,jk) = umask(ji,jj,jk) * umask(ji,jj,jk-1)
        wvmask(ji,jj,jk) = vmask(ji,jj,jk) * vmask(ji,jj,jk-1)
      end do
    end do
  end do
  r3t = 0._wp; r3t_f = 0._wp; r1_e1e2t = 1._wp

  ! istate_init (:81-87, 135): the after level, rn2 and rab at 0; the
  ! initial state (usr_def_istate style, masked) in before and now
  ts = 0._wp; rn2 = 0._wp; rn2b = 0._wp; rab_b = 0._wp; rab_n = 0._wp
  uu = 0._wp; vv = 0._wp; wi = 0._wp; ssh = 0._wp
  do jj = 1, ny
    do ji = 1, nx
      ic = min(max(ji, Nis0), Nie0); jc = min(max(jj, Njs0), Nje0)
      do jk = 1, nz
        ts(ji,jj,jk,jp_tem,Nbb) = (19._wp - 12._wp * gdept_1d(jk) / (gdept_1d(jk) + 150._wp) &
            + 1.5_wp * frac(ic * 5 + jc * 3 + jk, 53) - 0.4_wp * real(ic, wp)) * tmask(ji,jj,jk)
        ts(ji,jj,jk,jp_sal,Nbb) = (34.2_wp + 0.8_wp * gdept_1d(jk) / (gdept_1d(jk) + 300._wp) &
            + 0.3_wp * frac(ic + jc * 7 + jk * 3, 59)) * tmask(ji,jj,jk)
      end do
      ! a cold, salty layer at level 2 in two columns: statically unstable
      if (ic == 3 .or. jc == 5) then
        ts(ji,jj,2,jp_tem,Nbb) = (ts(ji,jj,3,jp_tem,Nbb) - 1.2_wp) * tmask(ji,jj,2)
      end if
    end do
  end do
  ts(:,:,:,:,Nnn) = ts(:,:,:,:,Nbb)

  ! surface fields: sbc_init / sbc_oce_alloc zero them; no sea ice, no runoff
  utau = 0._wp; vtau = 0._wp; taum = 0._wp; wndm = 0._wp; fr_i = 0._wp; qns = 0._wp; qsr = 0._wp
  emp = 0._wp; sfx = 0._wp; rnf = 0._wp; sst_m = 0._wp; sss_m = 0._wp; cloud_fra = 0._wp
  sbc_tsc = 0._wp; sbc_tsc_b = 0._wp; qsr_hc = 0._wp; qsr_hc_b = 0._wp; fraqsr_1lev = 1._wp
  xcplmask = 1._wp; incoming = 0._wp

  ! sbc_cpl_init's registry for WRF's six fields (namsbc_cpl: O_OTaux1/O_OTauy1
  ! on the local grid, O_TauMod, O_QsrOce, O_QnsOce, OOEvaMPr 'oce only')
  sn_rcv_tau%cldes = 'oce only'; sn_rcv_tau%clvor = 'local grid'; sn_rcv_tau%clvref = 'spherical'
  sn_rcv_tau%clvgrd = 'T'
  sn_rcv_emp%cldes = 'oce only'
  srcv(jpr_otx1)%laction = .true.; srcv(jpr_oty1)%laction = .true.; srcv(jpr_taum)%laction = .true.
  srcv(jpr_qsroce)%laction = .true.; srcv(jpr_qnsoce)%laction = .true.; srcv(jpr_oemp)%laction = .true.
  do n = 1, jprcv
    allocate(frcv(n)%z3(Nis0:Nie0,Njs0:Nje0,1))
    frcv(n)%z3 = 0._wp
  end do
  ncpl_sec = ncpl * nint(rn_Dt)

  ! nemo_init order: eos_init, then zdf_phy_init (:151-179 with nn_avb =
  ! nn_havtb = 0, zdf_tke_init, zdf_drg_init, tke_rst 'READ'), tra_qsr_init
  call eos_init
  avmb(:) = 1.2e-4_wp
  avtb(:) = 1.2e-5_wp
  avtb_2d(:,:) = 1._wp
  do jk = 1, jpk
    avt_k(:,:,jk) = avtb_2d(Nis0:Nie0,Njs0:Nje0) * avtb(jk) * wmask(Nis0:Nie0,Njs0:Nje0,jk)
    avm_k(:,:,jk) = avmb(jk) * wmask(:,:,jk)
  end do
  avt = 0._wp; avs = 0._wp; avm = 0._wp
  call zdf_tke_init
  call zdf_drg_init
  call tke_rst(nit000, 'READ')
  if (ln_traqsr) call tra_qsr_init
  sh2 = 0._wp

  open(newunit=ug, file=trim(dir)//'/column-grid.csv', status='replace')
  write(ug, '(A)') 'ji,jj,jk,mbkt,tmask,gphit,e3t,e3w,gdept,gdepw,t0,s0'
  do jj = Njs0, Nje0
    do ji = Nis0, Nie0
      do jk = 1, nz
        write(ug, '(4(I0,","),ES25.17E3,7(",",ES25.17E3))') ji, jj, jk, mbot(ji,jj), tmask(ji,jj,jk), &
            gphit(ji,jj), e3t_1d(jk), e3w_1d(jk), gdept_1d(jk), gdepw_1d(jk), &
            ts(ji,jj,jk,jp_tem,Nbb), ts(ji,jj,jk,jp_sal,Nbb)
      end do
    end do
  end do
  close(ug)
  open(newunit=uf, file=trim(dir)//'/column-forcing.csv', status='replace')
  write(uf, '(A)') 'window,ji,jj,evap_precip,surf_net_solar,surf_net_non_solar,taux,tauy,taumod'
  open(newunit=u, file=trim(dir)//'/column-steps.csv', status='replace')
  write(u, '(A)') 'kt,ji,jj,jk,tb,sb,tn,sn,en,avt,avm_k,rn2'
  open(newunit=us, file=trim(dir)//'/column-surface.csv', status='replace')
  write(us, '(A)') 'kt,ji,jj,sst_m,sss_m,utau,vtau,taum,wndm,qns,qsr,emp,sbc_tsc_t,sbc_tsc_s,rcdu_bot,sst_out'

  do kt = 1, nstep
    ! the exchange window this step opens: WRF's cpl_snd2 fields, REAL(4)
    if (mod(kt - 1, ncpl) == 0) then
      jw = (kt - 1) / ncpl + 1
      do jj = Njs0, Nje0
        do ji = Nis0, Nie0
          f4(ji,jj,1) = real(1.e-4_wp * (frac(ji * 3 + jj + jw * 7, 61) - 0.4_wp), 4)
          f4(ji,jj,2) = real(650._wp * frac(ji + jj * 5 + jw * 3, 67) * merge(0._wp, 1._wp, jw == 2 .and. ji == 3), 4)
          f4(ji,jj,3) = real(40._wp - 320._wp * frac(ji * 7 + jj * 2 + jw, 71), 4)
          f4(ji,jj,4) = real(0.5_wp * frac(ji * 11 + jj + jw * 5, 73) - 0.2_wp, 4)
          f4(ji,jj,5) = real(0.4_wp * frac(ji + jj * 13 + jw * 2, 79) - 0.2_wp, 4)
          f4(ji,jj,6) = real(sqrt(real(f4(ji,jj,4), wp)**2 + real(f4(ji,jj,5), wp)**2) &
              * (1._wp + 0.2_wp * frac(ji + jj + jw, 83)), 4)
          if (ji == 5 .and. jj == 3 .and. jw == 1) f4(ji,jj,6) = -1.e-3   ! MAX(0, taum) at work
          write(uf, '(3(I0,","),ES16.8E2,5(",",ES16.8E2))') jw, ji, jj, f4(ji,jj,:)
        end do
      end do
      ! the coupler hands NEMO REAL(8) fields: the REAL(4) values, exactly
      incoming(:,:,jpr_oemp) = real(f4(:,:,1), wp)
      incoming(:,:,jpr_qsroce) = real(f4(:,:,2), wp)
      incoming(:,:,jpr_qnsoce) = real(f4(:,:,3), wp)
      incoming(:,:,jpr_otx1) = real(f4(:,:,4), wp)
      incoming(:,:,jpr_oty1) = real(f4(:,:,5), wp)
      incoming(:,:,jpr_taum) = real(f4(:,:,6), wp)
    end if

    ! stpmlf.F90 :114-116
    if (l_1st_euler) then
      rDt = rn_Dt
      r1_Dt = 1._wp / rDt
    end if
    ! sbc (sbcmod.F90): sbc_ssm with nn_fsbc = 1 (sbcssm.F90 :70-76), then
    ! the pure coupled formulation's sbc_cpl_rcv
    zts(:,:,jp_tem) = ts(:,:,1,jp_tem,Nnn)
    zts(:,:,jp_sal) = ts(:,:,1,jp_sal,Nnn)
    if (l_useCT) then
      call eos_pt_from_ct(zts(:,:,jp_tem), zts(:,:,jp_sal), sst_m(:,:))
    else
      sst_m(:,:) = zts(:,:,jp_tem)
    end if
    sss_m(:,:) = zts(:,:,jp_sal)
    call sbc_cpl_rcv(kt, 1, nn_ice, Nbb, Nnn)
    call eos_rab(ts(:,:,:,:,Nbb), rab_b, Nnn)
    call eos_rab(ts(:,:,:,:,Nnn), rab_n, Nnn)
    call bn2(ts(:,:,:,:,Nbb), rab_b, rn2b, Nnn)
    call bn2(ts(:,:,:,:,Nnn), rab_n, rn2, Nnn)
    ! zdf_phy (zdfphy.F90 :268-330), TKE path, no ddm/evd/swm/iwm/rnf mouth
    call zdf_sh2(Nbb, Nnn, avm_k, sh2)
    call zdf_drg(kt, Nnn, 'BOTTOM')
    call zdf_tke(kt, Nbb, Nnn, sh2, avm_k, avt_k)
    do jk = 2, jpkm1
      do jj = Njs0, Nje0
        do ji = Nis0, Nie0
          avt(ji,jj,jk) = avt_k(ji,jj,jk)
          avm(ji,jj,jk) = avm_k(ji,jj,jk)
        end do
      end do
    end do
    do jk = 1, jpkm1
      do jj = Njs0, Nje0
        do ji = Nis0, Nie0
          avs(ji,jj,jk) = avt(ji,jj,jk)
        end do
      end do
    end do
    ! stpmlf.F90 :333-337, the TRA tile loop
    ts(Nis0:Nie0,Njs0:Nje0,1:jpkm1,:,Nrhs) = 0._wp
    call tra_sbc(kt, Nnn, ts, Nrhs)
    if (ln_traqsr) call tra_qsr(kt, Nnn, ts, Nrhs)
    call tra_zdf(kt, Nbb, Nnn, Nrhs, ts, Naa)
    call tra_atf_qco(kt, Nbb, Nnn, Naa, ts)
    ! stpmlf.F90 :403-406
    Nrhs = Nbb
    Nbb = Nnn
    Nnn = Naa
    Naa = Nrhs
    ! sbc_cpl_snd (sbccpl.F90 :2228-2236), nn_components = 0, 'oce only';
    ! the coupler hands WRF the REAL(8) field as REAL(4)
    if (l_useCT) then
      call eos_pt_from_ct(ts(:,:,1,jp_tem,Nnn), ts(:,:,1,jp_sal,Nnn), ztmp1(:,:))
    else
      ztmp1(:,:) = ts(:,:,1,jp_tem,Nnn)
    end if
    ztmp1(:,:) = ztmp1(:,:) + rt0
    sst_out(:,:) = real(ztmp1(:,:), 4)
    ! stpmlf.F90 :467-470
    if (l_1st_euler) then
      rDt = 2._wp * rn_Dt
      r1_Dt = 1._wp / rDt
      l_1st_euler = .false.
    end if

    do jj = Njs0, Nje0
      do ji = Nis0, Nie0
        write(us, '(3(I0,","),ES25.17E3,12(",",ES25.17E3),",",ES16.8E2)') kt, ji, jj, sst_m(ji,jj), &
            sss_m(ji,jj), utau(ji,jj), vtau(ji,jj), taum(ji,jj), wndm(ji,jj), qns(ji,jj), qsr(ji,jj), &
            emp(ji,jj), sbc_tsc(ji,jj,jp_tem), sbc_tsc(ji,jj,jp_sal), rCdU_bot(ji,jj), sst_out(ji,jj)
        do jk = 1, nz
          write(u, '(4(I0,","),ES25.17E3,7(",",ES25.17E3))') kt, ji, jj, jk, ts(ji,jj,jk,jp_tem,Nbb), &
              ts(ji,jj,jk,jp_sal,Nbb), ts(ji,jj,jk,jp_tem,Nnn), ts(ji,jj,jk,jp_sal,Nnn), en(ji,jj,jk), &
              avt(ji,jj,jk), avm_k(ji,jj,jk), rn2(ji,jj,jk)
        end do
      end do
    end do
  end do
  close(u); close(us); close(uf)

contains

  real(wp) function frac(m, p)
    integer, intent(in) :: m, p
    frac = real(mod(m * p * 7919, 1009), wp) / 1009._wp
  end function frac

end program run_nemo_column_oracle
