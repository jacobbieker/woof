program run_nemo_traldf_oracle
  ! Drive NEMO 5.0.2's iso-level lateral diffusion, traldf_lev_lap and
  ! traldf_lev_blp (traldf_lev.F90 compiled whole against traldf_stubs.F90),
  ! on a 10 x 9 x 15 grid with NEMO 5.0's two-cell halo (6 x 5 interior
  ! columns), closed boundaries (land halo), an interior land column, a
  ! shelf and varying scale factors.  Three calls for each operator with
  ! different diffusivities and tracer fields; every input, halo included,
  ! and the resulting RHS words are dumped.  Linear free surface.
  use par_kind
  use dom_oce
  use ldftra
  use traldf_lev
  implicit none
  integer, parameter :: nx = 10, ny = 9, nz = 15, ncall = 3
  integer, parameter :: Kbb = 1, Kmm = 2, Krhs = 3
  real(wp), allocatable :: pt(:,:,:,:,:), r0(:,:,:,:)
  integer, allocatable :: mbot(:,:)
  integer :: ji, jj, jk, jn, ic, iop, u, ug
  character(len=1024) :: dir
  character(len=3) :: op

  call get_command_argument(1, dir)
  jpi = nx; jpj = ny; jpk = nz; jpkm1 = nz - 1
  Nis0 = 3; Nie0 = nx - 2; Njs0 = 3; Nje0 = ny - 2
  ntsi = Nis0; ntei = Nie0; ntsj = Njs0; ntej = Nje0
  allocate(tmask(nx,ny,nz), umask(nx,ny,nz), vmask(nx,ny,nz), e3t_3d(nx,ny,nz), e3u_3d(nx,ny,nz))
  allocate(e3v_3d(nx,ny,nz), e2_e1u(nx,ny), e1_e2v(nx,ny), r1_e1e2t(nx,ny), mbot(nx,ny))
  allocate(ahtu(nx,ny,nz), ahtv(nx,ny,nz), pt(nx,ny,nz,2,3), r0(nx,ny,nz,2))
  mbot = 0
  do jj = Njs0, Nje0
    do ji = Nis0, Nie0
      mbot(ji,jj) = nz - 1 - mod(ji * 3 + jj * 5, 5)
    end do
  end do
  mbot(6,4) = 3          ! shelf
  mbot(4,6) = 0          ! interior land
  do jj = 1, ny
    do ji = 1, nx
      e2_e1u(ji,jj) = 1._wp + 0.3_wp * (frac(ji * 5 + jj, 41) - 0.5_wp)
      e1_e2v(ji,jj) = 1._wp + 0.3_wp * (frac(ji + jj * 7, 43) - 0.5_wp)
      r1_e1e2t(ji,jj) = 1._wp / (4.e6_wp * (1._wp + 0.2_wp * frac(ji * 3 + jj * 2, 47)))
      do jk = 1, nz
        tmask(ji,jj,jk) = merge(1._wp, 0._wp, jk <= mbot(ji,jj))
        e3t_3d(ji,jj,jk) = 3._wp + 1.5_wp * real(jk - 1, wp)**1.5_wp + 0.1_wp * frac(ji + jj * 5 + jk, 53)
      end do
    end do
  end do
  do jj = 1, ny
    do ji = 1, nx
      do jk = 1, nz
        umask(ji,jj,jk) = tmask(ji,jj,jk) * tmask(min(ji+1,nx),jj,jk)
        vmask(ji,jj,jk) = tmask(ji,jj,jk) * tmask(ji,min(jj+1,ny),jk)
        e3u_3d(ji,jj,jk) = min(e3t_3d(ji,jj,jk), e3t_3d(min(ji+1,nx),jj,jk))
        e3v_3d(ji,jj,jk) = min(e3t_3d(ji,jj,jk), e3t_3d(ji,min(jj+1,ny),jk))
      end do
    end do
  end do

  open(newunit=ug, file=trim(dir)//'/traldf-grid.csv', status='replace')
  write(ug, '(A)') 'ji,jj,jk,mbkt,tmask,umask,vmask,e3t,e3u,e3v,e2_e1u,e1_e2v,r1_e1e2t'
  do jj = 1, ny
    do ji = 1, nx
      do jk = 1, nz
        write(ug, '(4(I0,","),ES25.17E3,8(",",ES25.17E3))') ji, jj, jk, mbot(ji,jj), tmask(ji,jj,jk), &
            umask(ji,jj,jk), vmask(ji,jj,jk), e3t_3d(ji,jj,jk), e3u_3d(ji,jj,jk), e3v_3d(ji,jj,jk), &
            e2_e1u(ji,jj), e1_e2v(ji,jj), r1_e1e2t(ji,jj)
      end do
    end do
  end do
  close(ug)
  open(newunit=u, file=trim(dir)//'/traldf-calls.csv', status='replace')
  write(u, '(A)') 'op,call,ji,jj,jk,ahtu,ahtv,tb,sb,trhs0,srhs0,trhs,srhs'
  do iop = 1, 2
    op = merge('lap', 'blp', iop == 1)
    do ic = 1, ncall
      do jj = 1, ny
        do ji = 1, nx
          do jk = 1, nz
            ! laplacian ~ 1e2 m2/s; bilaplacian ~ -1e9 m4/s (ldf_tra's sign convention)
            ahtu(ji,jj,jk) = (frac(ji * 7 + jj + jk + ic * 3, 59) + 0.2_wp) * umask(ji,jj,jk)
            ahtv(ji,jj,jk) = (frac(ji + jj * 3 + jk * 2 + ic, 61) + 0.2_wp) * vmask(ji,jj,jk)
            if (iop == 1) then
              ahtu(ji,jj,jk) = 200._wp * ahtu(ji,jj,jk); ahtv(ji,jj,jk) = 200._wp * ahtv(ji,jj,jk)
            else
              ahtu(ji,jj,jk) = -2.e9_wp * ahtu(ji,jj,jk); ahtv(ji,jj,jk) = -2.e9_wp * ahtv(ji,jj,jk)
            end if
            pt(ji,jj,jk,1,Kbb) = (14._wp - 0.3_wp * real(jk, wp) + 3._wp * frac(ji * 5 + jj * 3 + jk + ic, 67) &
                - 1.5_wp) * tmask(ji,jj,jk)
            pt(ji,jj,jk,2,Kbb) = (34.6_wp + 0.04_wp * real(jk, wp) + 0.4_wp * frac(ji + jj * 7 + jk + ic, 71) &
                - 0.2_wp) * tmask(ji,jj,jk)
            pt(ji,jj,jk,:,Kmm) = pt(ji,jj,jk,:,Kbb)
            do jn = 1, 2
              pt(ji,jj,jk,jn,Krhs) = 1.e-5_wp * (frac(ji * jn + jj + jk * 3 + ic, 73) - 0.5_wp) * tmask(ji,jj,jk)
            end do
          end do
        end do
      end do
      r0 = pt(:,:,:,:,Krhs)
      if (iop == 1) then
        call traldf_lev_lap(ic, Kbb, Kmm, pt, Krhs)
      else
        call traldf_lev_blp(ic, Kbb, Kmm, pt, Krhs)
      end if
      do jj = 1, ny
        do ji = 1, nx
          do jk = 1, nz
            write(u, '(A,",",4(I0,","),ES25.17E3,7(",",ES25.17E3))') op, ic, ji, jj, jk, ahtu(ji,jj,jk), &
                ahtv(ji,jj,jk), pt(ji,jj,jk,1,Kbb), pt(ji,jj,jk,2,Kbb), r0(ji,jj,jk,1), r0(ji,jj,jk,2), &
                pt(ji,jj,jk,1,Krhs), pt(ji,jj,jk,2,Krhs)
          end do
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

end program run_nemo_traldf_oracle
