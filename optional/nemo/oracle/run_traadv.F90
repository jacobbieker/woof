program run_nemo_traadv_oracle
  ! Drive NEMO 5.0.2's tra_adv_fct (traadv_fct.F90 compiled whole against
  ! traadv_stubs.F90) on a 10 x 9 x 15 grid with NEMO 5.0's two-cell halo
  ! (6 x 5 interior columns), closed boundaries (the halo is land), an
  ! interior land column, a shelf, and fronts sharp enough for the limiter to
  ! act.  For each of the four (kn_fct_h, kn_fct_v) pairs from {2, 4}, three
  ! calls with different transports and tracer fields; every input and the
  ! resulting RHS words are dumped, halo included (masking leaves signed
  ! zeros there, and NEMO's signed zeros reach the RHS below the sea floor).
  ! MLF (no key_RK3), linear free surface, no adaptive-implicit vertical
  ! advection, no trend diagnostics.
  use par_kind
  use oce
  use dom_oce
  use traadv_fct
  implicit none
  integer, parameter :: nx = 10, ny = 9, nz = 15, ncall = 3
  integer, parameter :: Kbb = 1, Kmm = 2, Kaa = 3, Krhs = 3
  real(wp), allocatable :: pt(:,:,:,:,:), pU(:,:,:), pV(:,:,:), pW(:,:,:)
  integer, allocatable :: mbot(:,:)
  integer :: ji, jj, jk, jn, ic, kh, kv, u, ug, ord(2)
  real(wp) :: p2dt, front
  character(len=1024) :: dir

  call get_command_argument(1, dir)
  jpi = nx; jpj = ny; jpk = nz; jpkm1 = nz - 1
  Nis0 = 3; Nie0 = nx - 2; Njs0 = 3; Nje0 = ny - 2
  ntsi = Nis0; ntei = Nie0; ntsj = Njs0; ntej = Nje0
  allocate(tmask(nx,ny,nz), umask(nx,ny,nz), vmask(nx,ny,nz), wmask(nx,ny,nz), e3t_3d(nx,ny,nz))
  allocate(e1e2t(nx,ny), r1_e1e2t(nx,ny), mbkt(nx,ny), mikt(nx,ny), mbot(nx,ny), wi(nx,ny,nz))
  allocate(pt(nx,ny,nz,2,3), pU(nx,ny,nz), pV(nx,ny,nz), pW(nx,ny,nz))
  wi = 0._wp
  mbot = 0
  do jj = Njs0, Nje0
    do ji = Nis0, Nie0
      mbot(ji,jj) = nz - 1 - mod(ji * 5 + jj * 3, 5)
    end do
  end do
  mbot(5,4) = 4          ! shelf
  mbot(6,6) = 0          ! interior land
  do jj = 1, ny
    do ji = 1, nx
      mbkt(ji,jj) = max(mbot(ji,jj), 1)
      mikt(ji,jj) = 1
      e1e2t(ji,jj) = 4.e6_wp * (1._wp + 0.2_wp * frac(ji * 3 + jj * 7, 41))
      r1_e1e2t(ji,jj) = 1._wp / e1e2t(ji,jj)
      do jk = 1, nz
        tmask(ji,jj,jk) = merge(1._wp, 0._wp, jk <= mbot(ji,jj))
        e3t_3d(ji,jj,jk) = 3._wp + 1.5_wp * real(jk - 1, wp)**1.5_wp + 0.1_wp * frac(ji + jj * 5 + jk, 43)
      end do
    end do
  end do
  do jj = 1, ny
    do ji = 1, nx
      do jk = 1, nz
        umask(ji,jj,jk) = tmask(ji,jj,jk) * tmask(min(ji+1,nx),jj,jk)
        vmask(ji,jj,jk) = tmask(ji,jj,jk) * tmask(ji,min(jj+1,ny),jk)
      end do
      wmask(ji,jj,1) = tmask(ji,jj,1)
      do jk = 2, nz
        wmask(ji,jj,jk) = tmask(ji,jj,jk) * tmask(ji,jj,jk-1)
      end do
    end do
  end do
  p2dt = 1800._wp

  open(newunit=ug, file=trim(dir)//'/traadv-grid.csv', status='replace')
  write(ug, '(A)') 'ji,jj,jk,mbkt,tmask,e3t,e1e2t'
  do jj = 1, ny
    do ji = 1, nx
      do jk = 1, nz
        write(ug, '(4(I0,","),ES25.17E3,2(",",ES25.17E3))') ji, jj, jk, mbot(ji,jj), tmask(ji,jj,jk), &
            e3t_3d(ji,jj,jk), e1e2t(ji,jj)
      end do
    end do
  end do
  close(ug)
  open(newunit=u, file=trim(dir)//'/traadv-calls.csv', status='replace')
  write(u, '(A)') 'kn_fct_h,kn_fct_v,call,ji,jj,jk,pu,pv,pw,tb,sb,tn,sn,trhs0,srhs0,trhs,srhs'
  do kh = 2, 4, 2
    do kv = 2, 4, 2
      do ic = 1, ncall
        ! transports (m3/s), masked; the halo is land and carries none
        do jj = 1, ny
          do ji = 1, nx
            do jk = 1, nz
              pU(ji,jj,jk) = 6.e3_wp * (frac(ji * 7 + jj + jk * 3 + ic * 11, 47) - 0.45_wp) * umask(ji,jj,jk)
              pV(ji,jj,jk) = 5.e3_wp * (frac(ji + jj * 11 + jk * 5 + ic * 3, 53) - 0.5_wp) * vmask(ji,jj,jk)
              pW(ji,jj,jk) = 4.e2_wp * (frac(ji * 3 + jj * 2 + jk * 7 + ic, 59) - 0.5_wp) * wmask(ji,jj,jk)
              front = merge(4._wp, 0._wp, ji + jj + ic > 11)
              pt(ji,jj,jk,1,Kbb) = (15._wp - 0.4_wp * real(jk, wp) + front &
                  + frac(ji * 5 + jj + jk + ic * 7, 61)) * tmask(ji,jj,jk)
              pt(ji,jj,jk,2,Kbb) = (34.5_wp + 0.05_wp * real(jk, wp) - 0.3_wp * front &
                  + 0.2_wp * frac(ji + jj * 3 + jk * 2 + ic, 67)) * tmask(ji,jj,jk)
              do jn = 1, 2
                pt(ji,jj,jk,jn,Kmm) = (pt(ji,jj,jk,jn,Kbb) + 0.1_wp * (frac(ji + jj + jk * jn + ic, 71) &
                    - 0.5_wp)) * tmask(ji,jj,jk)
                pt(ji,jj,jk,jn,Krhs) = 1.e-5_wp * (frac(ji * jn + jj * 2 + jk + ic, 73) - 0.5_wp) &
                    * tmask(ji,jj,jk)
              end do
            end do
          end do
        end do
        ! a near-surface level where advection must not create extrema
        pt(4:7,4:6,2,1,Kbb) = pt(4:7,4:6,2,1,Kbb) * 1.3_wp
        pt(4:7,4:6,2,1,Kmm) = pt(4:7,4:6,2,1,Kbb)
        ord = [kh, kv]
        call dump(.true.)
        call tra_adv_fct(ic, 1, 'TRA', p2dt, pU, pV, pW, Kbb, Kmm, Kaa, pt, 2, Krhs, kh, kv, 1)
        call dump(.false.)
      end do
    end do
  end do
  close(u)

contains

  subroutine dump(before)
    logical, intent(in) :: before
    real(wp), save, allocatable :: r0(:,:,:,:)
    if (before) then
      if (.not. allocated(r0)) allocate(r0(nx,ny,nz,2))
      r0 = pt(:,:,:,:,Krhs)
      return
    end if
    do jj = 1, ny
      do ji = 1, nx
        do jk = 1, nz
          write(u, '(6(I0,","),ES25.17E3,10(",",ES25.17E3))') ord(1), ord(2), ic, ji, jj, jk, &
              pU(ji,jj,jk), pV(ji,jj,jk), pW(ji,jj,jk), pt(ji,jj,jk,1,Kbb), pt(ji,jj,jk,2,Kbb), &
              pt(ji,jj,jk,1,Kmm), pt(ji,jj,jk,2,Kmm), r0(ji,jj,jk,1), r0(ji,jj,jk,2), &
              pt(ji,jj,jk,1,Krhs), pt(ji,jj,jk,2,Krhs)
        end do
      end do
    end do
  end subroutine dump

  real(wp) function frac(m, p)
    integer, intent(in) :: m, p
    frac = real(mod(m * p * 7919, 1009), wp) / 1009._wp
  end function frac

end program run_nemo_traadv_oracle
