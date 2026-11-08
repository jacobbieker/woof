program run_nemo_tke_oracle
  ! Drive NEMO 5.0.2's zdf_sh2 and zdf_tke (zdfsh2.F90 and zdftke.F90 compiled
  ! whole against tke_stubs.F90) for three steps over a 9 x 7 x 21 grid with a
  ! one-cell halo, land, a shelf column and unstable layers, after
  ! zdf_tke_init has read namzdf_tke: NEMO's reference block (namelist_ref,
  ! extracted by the build) plus the configuration overrides passed as the
  ! second argument.  Every input and output word is dumped.  No sea-ice model
  ! (nn_ice = 0), so zdf_tke_init sets nn_eice = 0 as NEMO does.
  use par_kind
  use oce
  use dom_oce
  use sbc_oce
  use zdfdrg
  use zdf_oce
  use in_out_manager
  use zdfsh2
  use zdftke
  implicit none
  integer, parameter :: nx = 9, ny = 7, nz = 21, nstep = 3
  real(wp), allocatable :: avm(:,:,:), avt(:,:,:), sh2(:,:,:)
  integer :: ji, jj, jk, kt, mbot, u, n
  character(len=1024) :: dir, cfg, ref
  character(len=4096) :: buf

  call get_command_argument(1, dir)
  call get_command_argument(2, ref)
  call get_command_argument(3, cfg)
  if (len_trim(ref) == 0) error stop 'usage: run_tke OUTDIR NAMELIST_REF_FILE "&namzdf_tke ... /"'
  open(newunit=u, file=trim(ref), status='old', access='stream', form='unformatted')
  inquire(unit=u, size=n)
  allocate(character(len=n) :: numnam_ref)
  read(u) numnam_ref
  close(u)
  numnam_cfg = trim(cfg)

  jpi = nx; jpj = ny; jpk = nz; jpkm1 = nz - 1
  Nis0 = 2; Nie0 = nx - 1; Njs0 = 2; Nje0 = ny - 1
  ntsi = Nis0; ntei = Nie0; ntsj = Njs0; ntej = Nje0
  rn_Dt = 900._wp
  allocate(tmask(nx,ny,nz), umask(nx,ny,nz), vmask(nx,ny,nz), wmask(nx,ny,nz), wumask(nx,ny,nz), wvmask(nx,ny,nz))
  allocate(e3t_3d(nx,ny,nz), e3w_3d(nx,ny,nz), e3uw_3d(nx,ny,nz), e3vw_3d(nx,ny,nz), gdepw_3d(nx,ny,nz), gdept_3d(nx,ny,nz))
  allocate(ssmask(nx,ny), gphit(nx,ny), mbkt(nx,ny), mikt(nx,ny))
  allocate(uu(nx,ny,nz,3), vv(nx,ny,nz,3), rn2(nx,ny,nz), rn2b(nx,ny,nz))
  allocate(utau(nx,ny), vtau(nx,ny), taum(nx,ny), fr_i(nx,ny), rCdU_bot(nx,ny), rCdU_top(nx,ny))
  allocate(en(Nis0:Nie0,Njs0:Nje0,nz), avtb_2d(nx,ny), avtb(nz), avmb(nz))
  allocate(avm(nx,ny,nz), avt(Nis0:Nie0,Njs0:Nje0,nz), sh2(Nis0:Nie0,Njs0:Nje0,nz))

  do jj = 1, ny
    do ji = 1, nx
      mbot = nz - 2 - mod(ji * 3 + jj * 5, 6)
      if (ji == 5 .and. jj == 4) mbot = 3            ! shelf
      if (ji == 3 .and. jj == 3) mbot = 0            ! land
      mbkt(ji,jj) = max(mbot, 1)
      mikt(ji,jj) = 1
      gphit(ji,jj) = -60._wp + 20._wp * real(jj, wp) + 0.5_wp * real(ji, wp)
      gdepw_3d(ji,jj,1) = 0._wp
      do jk = 1, nz
        e3t_3d(ji,jj,jk) = 2._wp + 0.6_wp * real(jk - 1, wp)**2 + 0.05_wp * frac(ji + 3 * jj + jk, 53)
        if (jk > 1) gdepw_3d(ji,jj,jk) = gdepw_3d(ji,jj,jk-1) + e3t_3d(ji,jj,jk-1)
        gdept_3d(ji,jj,jk) = gdepw_3d(ji,jj,jk) + 0.5_wp * e3t_3d(ji,jj,jk)
        tmask(ji,jj,jk) = merge(1._wp, 0._wp, jk <= mbot)
        uu(ji,jj,jk,1) = 0.4_wp * frac(ji * 7 + jj + jk * 3, 59) - 0.2_wp
        uu(ji,jj,jk,2) = uu(ji,jj,jk,1) + 0.05_wp * frac(ji + jj * 5 + jk, 61)
        vv(ji,jj,jk,1) = 0.3_wp * frac(ji + jj * 11 + jk * 5, 67) - 0.15_wp
        vv(ji,jj,jk,2) = vv(ji,jj,jk,1) - 0.04_wp * frac(ji * 3 + jj + jk * 7, 71)
        rn2(ji,jj,jk) = 1.e-4_wp * (frac(ji * 13 + jj * 3 + jk, 73) - 0.15_wp)
        rn2b(ji,jj,jk) = rn2(ji,jj,jk) * (1._wp + 0.1_wp * frac(jk, 79))
      end do
      e3w_3d(ji,jj,1) = 0.5_wp * e3t_3d(ji,jj,1)
      do jk = 2, nz
        e3w_3d(ji,jj,jk) = gdept_3d(ji,jj,jk) - gdept_3d(ji,jj,jk-1)
      end do
      utau(ji,jj) = 0.2_wp * frac(ji * 5 + jj * 7, 83) - 0.05_wp
      vtau(ji,jj) = 0.15_wp * frac(ji * 11 + jj, 89) - 0.03_wp
      taum(ji,jj) = sqrt(utau(ji,jj)**2 + vtau(ji,jj)**2) * (1._wp + 0.1_wp * frac(ji + jj, 97))
      fr_i(ji,jj) = 0._wp
      rCdU_bot(ji,jj) = -1.e-3_wp * (0.5_wp + frac(ji * 3 + jj * 2, 101))
      rCdU_top(ji,jj) = 0._wp
      avtb_2d(ji,jj) = 1._wp
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
      do jk = 1, nz
        e3uw_3d(ji,jj,jk) = 0.5_wp * (e3w_3d(ji,jj,jk) + e3w_3d(min(ji+1,nx),jj,jk))
        e3vw_3d(ji,jj,jk) = 0.5_wp * (e3w_3d(ji,jj,jk) + e3w_3d(ji,min(jj+1,ny),jk))
      end do
    end do
  end do
  avmb = 1.2e-4_wp
  avtb = 1.2e-5_wp
  do jk = 1, nz
    avm(:,:,jk) = avmb(jk) * wmask(:,:,jk)
    avt(:,:,jk) = avtb(jk) * wmask(Nis0:Nie0,Njs0:Nje0,jk)
  end do
  sh2 = 0._wp

  open(newunit=u, file=trim(dir)//'/tke-grid.csv', status='replace')
  write(u, '(A)') 'ji,jj,jk,mbkt,tmask,umask,vmask,wmask,wumask,wvmask,e3t,e3w,e3uw,e3vw,gdepw,' // &
      'u1,u2,v1,v2,rn2,rn2b,utau,vtau,taum,rcdu_bot,gphit,ssmask'
  do jj = 1, ny
    do ji = 1, nx
      do jk = 1, nz
        write(u, '(4(I0,","),ES25.17E3,22(",",ES25.17E3))') ji, jj, jk, mbkt(ji,jj), tmask(ji,jj,jk), &
            umask(ji,jj,jk), vmask(ji,jj,jk), wmask(ji,jj,jk), wumask(ji,jj,jk), wvmask(ji,jj,jk), &
            e3t_3d(ji,jj,jk), e3w_3d(ji,jj,jk), e3uw_3d(ji,jj,jk), e3vw_3d(ji,jj,jk), gdepw_3d(ji,jj,jk), &
            uu(ji,jj,jk,1), uu(ji,jj,jk,2), vv(ji,jj,jk,1), vv(ji,jj,jk,2), rn2(ji,jj,jk), rn2b(ji,jj,jk), &
            utau(ji,jj), vtau(ji,jj), taum(ji,jj), rCdU_bot(ji,jj), gphit(ji,jj), ssmask(ji,jj)
      end do
    end do
  end do
  close(u)

  call zdf_tke_init
  open(newunit=u, file=trim(dir)//'/tke-steps.csv', status='replace')
  write(u, '(A)') 'kt,ji,jj,jk,sh2,en,avm,avt'
  do kt = 1, nstep
    call zdf_sh2(1, 2, avm, sh2)
    call zdf_tke(kt, 1, 2, sh2, avm, avt)
    do jj = Njs0, Nje0
      do ji = Nis0, Nie0
        do jk = 1, nz
          write(u, '(4(I0,","),ES25.17E3,3(",",ES25.17E3))') kt, ji, jj, jk, sh2(ji,jj,jk), &
              en(ji,jj,jk), avm(ji,jj,jk), avt(ji,jj,jk)
        end do
      end do
    end do
  end do
  close(u)

contains

  real(wp) function frac(m, p)
    integer, intent(in) :: m, p
    frac = real(mod(m * p * 7919, 1009), wp) / 1009._wp
  end function frac

end program run_nemo_tke_oracle
