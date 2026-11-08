program run_nemo_trazdf_oracle
  ! Drive NEMO 5.0.2's tra_zdf (trazdf.F90 compiled whole against
  ! trazdf_stubs.F90) for three leapfrog-style calls over a 9 x 7 x 21 grid
  ! with a halo, land and a shelf column: before tracers, a right-hand-side
  ! trend and avt/avs from hashed profiles, a fresh layer that the solve drives
  ! below zero salinity (NEMO resets it to 0.1).  Every word is dumped.
  use par_kind
  use oce
  use dom_oce
  use zdf_oce
  use trazdf
  implicit none
  integer, parameter :: nx = 9, ny = 7, nz = 21, nstep = 3
  real(wp), allocatable :: pts(:,:,:,:,:)
  integer :: ji, jj, jk, kt, mbot, u, jn
  character(len=1024) :: dir

  call get_command_argument(1, dir)
  if (len_trim(dir) == 0) error stop 'usage: run_trazdf OUTDIR'
  jpi = nx; jpj = ny; jpk = nz; jpkm1 = nz - 1
  Nis0 = 2; Nie0 = nx - 1; Njs0 = 2; Nje0 = ny - 1
  ntsi = Nis0; ntei = Nie0; ntsj = Njs0; ntej = Nje0
  rDt = 1800._wp
  allocate(tmask(nx,ny,nz), e3t_3d(nx,ny,nz), e3w_3d(nx,ny,nz), avt(nx,ny,nz), avs(nx,ny,nz))
  allocate(pts(nx,ny,nz,jpts,jpt))
  pts = 0._wp
  do jj = 1, ny
    do ji = 1, nx
      mbot = nz - 2 - mod(ji * 3 + jj * 5, 6)
      if (ji == 5 .and. jj == 4) mbot = 3
      if (ji == 3 .and. jj == 3) mbot = 0
      do jk = 1, nz
        e3t_3d(ji,jj,jk) = 2._wp + 0.6_wp * real(jk - 1, wp)**2 + 0.05_wp * frac(ji + 3 * jj + jk, 53)
        tmask(ji,jj,jk) = merge(1._wp, 0._wp, jk <= mbot)
        pts(ji,jj,jk,jp_tem,1) = (20._wp - 0.8_wp * real(jk, wp) + 2._wp * frac(ji * 5 + jj + jk, 59)) * tmask(ji,jj,jk)
        pts(ji,jj,jk,jp_sal,1) = (34._wp + 0.05_wp * real(jk, wp) + frac(ji + jj * 7 + jk, 61)) * tmask(ji,jj,jk)
        pts(ji,jj,jk,jp_tem,3) = 1.e-5_wp * (frac(ji * 3 + jj * 11 + jk, 67) - 0.5_wp) * tmask(ji,jj,jk)
        pts(ji,jj,jk,jp_sal,3) = 1.e-6_wp * (frac(ji * 13 + jj + jk * 3, 71) - 0.5_wp) * tmask(ji,jj,jk)
        avt(ji,jj,jk) = 1.e-5_wp + 5.e-3_wp * frac(ji + jj + jk * 5, 73)
        avs(ji,jj,jk) = avt(ji,jj,jk) * (0.5_wp + frac(jk, 79))
      end do
      e3w_3d(ji,jj,1) = 0.5_wp * e3t_3d(ji,jj,1)
      do jk = 2, nz
        e3w_3d(ji,jj,jk) = 0.5_wp * (e3t_3d(ji,jj,jk) + e3t_3d(ji,jj,jk-1))
      end do
    end do
  end do
  pts(4, 2, 1, jp_sal, 1) = 0.01_wp                     ! nearly fresh surface
  pts(4, 2, 1, jp_sal, 3) = -1.e-3_wp                    ! strongly freshening trend
  avt(4, 2, :) = 1.e-7_wp; avs(4, 2, :) = 1.e-7_wp       ! barely mixed, so it goes below 0
  pts(:,:,:,:,2) = pts(:,:,:,:,1)

  open(newunit=u, file=trim(dir)//'/trazdf-grid.csv', status='replace')
  write(u, '(A)') 'ji,jj,jk,tmask,e3t,e3w,avt,avs,t0,s0,trhs,srhs'
  do jj = 1, ny
    do ji = 1, nx
      do jk = 1, nz
        write(u, '(3(I0,","),ES25.17E3,8(",",ES25.17E3))') ji, jj, jk, tmask(ji,jj,jk), &
            e3t_3d(ji,jj,jk), e3w_3d(ji,jj,jk), avt(ji,jj,jk), avs(ji,jj,jk), &
            pts(ji,jj,jk,jp_tem,1), pts(ji,jj,jk,jp_sal,1), pts(ji,jj,jk,jp_tem,3), pts(ji,jj,jk,jp_sal,3)
      end do
    end do
  end do
  close(u)
  open(newunit=u, file=trim(dir)//'/trazdf-steps.csv', status='replace')
  write(u, '(A)') 'kt,ji,jj,jk,t,s'
  do kt = 1, nstep
    ! Kbb = 1 (before), Kmm = 2, Krhs = 3, Kaa = 2: the after field
    ! overwrites the now level, as NEMO's leapfrog arrays do once swapped
    call tra_zdf(kt, 1, 2, 3, pts, 2)
    do jj = Njs0, Nje0
      do ji = Nis0, Nie0
        do jk = 1, nz
          write(u, '(4(I0,","),ES25.17E3,",",ES25.17E3)') kt, ji, jj, jk, &
              pts(ji,jj,jk,jp_tem,2), pts(ji,jj,jk,jp_sal,2)
        end do
      end do
    end do
    pts(:,:,:,:,1) = pts(:,:,:,:,2)      ! the after field becomes the next before
  end do
  close(u)

contains

  real(wp) function frac(m, p)
    integer, intent(in) :: m, p
    frac = real(mod(m * p * 7919, 1009), wp) / 1009._wp
  end function frac

end program run_nemo_trazdf_oracle
