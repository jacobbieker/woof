program run_nemo_eos_oracle
  ! Drive NEMO 5.0.2's eosbn2.F90, compiled whole against eos_stubs.F90, for
  ! each equation of state nameos offers (TEOS-10, EOS-80, S-EOS with its
  ! reference coefficients): eos_init from the namelist text, then the in
  ! situ and potential density (eos_insitu_pot_New), the thermal and haline
  ! expansion ratios (rab_3d) and the Brunt-Vaisala frequency (bn2), over a
  ! 7 x 5 x 31 grid with land, a shelf column, fresh and near-freezing water.
  ! Every input and output word is dumped.  Inputs come from integer hashes.
  use par_kind
  use dom_oce
  use in_out_manager
  use eosbn2
  implicit none
  integer, parameter :: nx = 7, ny = 5, nz = 31
  real(wp), allocatable :: pts(:,:,:,:,:), prd(:,:,:), prhop(:,:,:), pab(:,:,:,:), pn2(:,:,:)
  integer :: ji, jj, jk, mbot, u, ieos
  character(len=1024) :: dir
  character(len=8) :: names(3) = (/ 'teos10  ', 'eos80   ', 'seos    ' /)

  call get_command_argument(1, dir)
  if (len_trim(dir) == 0) error stop 'usage: run_eos OUTDIR'
  jpi = nx; jpj = ny; jpk = nz; jpkm1 = nz - 1
  ntsi = 1; ntei = nx; ntsj = 1; ntej = ny
  Nis0 = 1; Nie0 = nx; Njs0 = 1; Nje0 = ny
  allocate(tmask(nx, ny, nz), wmask(nx, ny, nz), gdept_3d(nx, ny, nz), gdepw_3d(nx, ny, nz))
  allocate(e3t_3d(nx, ny, nz), e3w_3d(nx, ny, nz))
  allocate(pts(nx, ny, nz, jpts, 3), prd(nx, ny, nz), prhop(nx, ny, nz), pab(nx, ny, nz, jpts), pn2(nx, ny, nz))
  do jj = 1, ny
    do ji = 1, nx
      mbot = nz - 1 - mod(ji * 3 + jj * 5, 9)
      if (ji == 4 .and. jj == 3) mbot = 4
      if (ji == 1 .and. jj == 1) mbot = 0       ! land
      gdepw_3d(ji, jj, 1) = 0._wp
      do jk = 1, nz
        e3t_3d(ji, jj, jk) = 1._wp + 0.18_wp * real(jk - 1, wp)**2 + 0.01_wp * frac(ji + 7 * jj + 3 * jk, 53)
        if (jk > 1) gdepw_3d(ji, jj, jk) = gdepw_3d(ji, jj, jk - 1) + e3t_3d(ji, jj, jk - 1)
        gdept_3d(ji, jj, jk) = gdepw_3d(ji, jj, jk) + 0.5_wp * e3t_3d(ji, jj, jk)
        tmask(ji, jj, jk) = merge(1._wp, 0._wp, jk <= mbot)
        ! temperature falls and salinity rises with depth, with hashed structure
        pts(ji, jj, jk, jp_tem, 1) = 28._wp * exp_free(jk) - 1.5_wp + 2._wp * frac(ji * 5 + jj * 3 + jk, 59)
        pts(ji, jj, jk, jp_sal, 1) = 33._wp + 3._wp * frac(ji + jj * 11 + jk * 7, 61)
      end do
      e3w_3d(ji, jj, 1) = 0.5_wp * e3t_3d(ji, jj, 1)
      do jk = 2, nz
        e3w_3d(ji, jj, jk) = gdept_3d(ji, jj, jk) - gdept_3d(ji, jj, jk - 1)
      end do
      wmask(ji, jj, 1) = tmask(ji, jj, 1)
      do jk = 2, nz
        wmask(ji, jj, jk) = tmask(ji, jj, jk) * tmask(ji, jj, jk - 1)
      end do
    end do
  end do
  pts(2, 4, 1:5, jp_sal, 1) = 0.2_wp        ! fresh river plume
  pts(6, 2, :, jp_tem, 1) = -1.9_wp         ! near freezing column
  pts(:, :, :, :, 2) = pts(:, :, :, :, 1)
  pts(:, :, :, :, 3) = pts(:, :, :, :, 1)

  open(newunit=u, file=trim(dir)//'/eos-grid.csv', status='replace')
  write(u, '(A)') 'ji,jj,jk,tmask,wmask,gdept,gdepw,e3w,t,s'
  do jj = 1, ny
    do ji = 1, nx
      do jk = 1, nz
        write(u, '(3(I0,","),ES25.17E3,6(",",ES25.17E3))') ji, jj, jk, tmask(ji, jj, jk), &
            wmask(ji, jj, jk), gdept_3d(ji, jj, jk), gdepw_3d(ji, jj, jk), e3w_3d(ji, jj, jk), &
            pts(ji, jj, jk, jp_tem, 1), pts(ji, jj, jk, jp_sal, 1)
      end do
    end do
  end do
  close(u)

  do ieos = 1, 3
    select case (ieos)
    case (1); numnam_ref = '&nameos ln_TEOS10=.true., ln_EOS80=.false., ln_SEOS=.false. /'
    case (2); numnam_ref = '&nameos ln_TEOS10=.false., ln_EOS80=.true., ln_SEOS=.false. /'
    case (3); numnam_ref = '&nameos ln_TEOS10=.false., ln_EOS80=.false., ln_SEOS=.true. /'
    end select
    numnam_cfg = '&nameos /'
    call eos_init
    prd = 0._wp; prhop = 0._wp; pab = 0._wp; pn2 = 0._wp
    call eos(pts, 1, prd, prhop)
    call eos_rab(pts(:, :, :, :, 1), pab, 1)
    call bn2(pts(:, :, :, :, 1), pab, pn2, 1)
    open(newunit=u, file=trim(dir)//'/eos-'//trim(names(ieos))//'.csv', status='replace')
    write(u, '(A)') 'ji,jj,jk,prd,prhop,alpha,beta,n2'
    do jj = 1, ny
      do ji = 1, nx
        do jk = 1, nz
          write(u, '(3(I0,","),ES25.17E3,4(",",ES25.17E3))') ji, jj, jk, prd(ji, jj, jk), &
              prhop(ji, jj, jk), pab(ji, jj, jk, jp_tem), pab(ji, jj, jk, jp_sal), pn2(ji, jj, jk)
        end do
      end do
    end do
    close(u)
  end do

contains

  real(wp) function frac(m, p)
    integer, intent(in) :: m, p
    frac = real(mod(m * p * 7919, 1009), wp) / 1009._wp
  end function frac
  ! a decreasing profile with no libm: 1 / (1 + k/6)
  real(wp) function exp_free(k)
    integer, intent(in) :: k
    exp_free = 1._wp / (1._wp + real(k, wp) / 6._wp)
  end function exp_free

end program run_nemo_eos_oracle
