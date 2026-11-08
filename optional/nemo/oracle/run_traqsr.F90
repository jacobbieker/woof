program run_nemo_traqsr_oracle
  ! Drive NEMO 5.0.2's tra_qsr with the 2-band scheme (ln_qsr_2bd) over a
  ! small linear-free-surface grid for three leapfrog steps and dump every
  ! input and output word as CSV.  The tra_qsr_init lines for np_2BD
  ! (traqsr.F90:1281-1283, 1318-1322, 1374-1379) are restated below because
  ! the routine itself reads a namelist; they are the only lines not compiled
  ! from the pinned text.  Inputs come from integer hashes, never libm.
  use traqsr_harness
  implicit none

  integer, parameter :: nx = 7, ny = 5, nz = 31, nstep = 3
  real(wp), allocatable :: pts(:,:,:,:,:)
  real(wp) :: zvlp, rho0, rcp
  integer :: ji, jj, jk, kt, mbot, u, v
  character(len=1024) :: dir

  call get_command_argument(1, dir)
  if (len_trim(dir) == 0) error stop 'usage: run_traqsr OUTDIR'

  jpi = nx; jpj = ny; jpk = nz; jpkm1 = nz - 1
  ntsi = 1; ntei = nx; ntsj = 1; ntej = ny
  allocate(e3t_3d(nx, ny, nz), gdepw_3d(nx, ny, nz), tmask(nx, ny, nz))
  allocate(wmask(nx, ny, nz), qsr(nx, ny), qsr_hc(nx, ny, nz))
  allocate(qsr_hc_b(nx, ny, nz), fraqsr_1lev(nx, ny), etot3(nx, ny, nz))
  allocate(pts(nx, ny, nz, jpts, jpt))

  ! phycst.F90 (NEMO 5.0.2): rho0 = 1026, rcp = 3991.86795711963
  rho0 = 1026._wp
  rcp = 3991.86795711963_wp
  rho0_rcp = rho0 * rcp
  r1_rho0_rcp = 1._wp / rho0_rcp
  rDt = 1800._wp
  ! namtra_qsr reference values
  rn_abs = 0.58_wp; rn_si0 = 0.35_wp; rn_si1 = 23.0_wp

  ! levels: 1 m at the surface, stretching with depth; some columns end early
  do jj = 1, ny
    do ji = 1, nx
      mbot = nz - 1 - mod(ji * 3 + jj * 5, 9)
      if (ji == 4 .and. jj == 3) mbot = 4       ! a shelf column
      gdepw_3d(ji, jj, 1) = 0._wp
      do jk = 1, nz
        e3t_3d(ji, jj, jk) = 1._wp + 0.18_wp * real(jk - 1, wp)**2 &
                             + 0.01_wp * frac(ji + 7 * jj + 3 * jk, 53)
        if (jk > 1) gdepw_3d(ji, jj, jk) = gdepw_3d(ji, jj, jk - 1) + e3t_3d(ji, jj, jk - 1)
        tmask(ji, jj, jk) = merge(1._wp, 0._wp, jk <= mbot)
      end do
      wmask(ji, jj, 1) = tmask(ji, jj, 1)
      do jk = 2, nz
        wmask(ji, jj, jk) = tmask(ji, jj, jk) * tmask(ji, jj, jk - 1)
      end do
    end do
  end do

  nqsr = np_2BD
  r1_si0 = 1._wp / rn_si0
  nk0 = qsr_ext_lev(rn_si0, rn_abs)
  r1_si1 = 1._wp / rn_si1
  zvlp = (1._wp - rn_abs)
  nkV = qsr_ext_lev(rn_si1, zvlp)
  nksr = nkV
  qsr_hc(:,:,:) = 0._wp
  qsr_hc_b(:,:,:) = 0._wp
  fraqsr_1lev(:,:) = 1._wp

  open(newunit=u, file=trim(dir)//'/traqsr-grid.csv', status='replace')
  write(u, '(A,4(",",I0))') '# nk0,nkV,nksr,nz', nk0, nkV, nksr, nz
  write(u, '(A)') 'ji,jj,jk,e3t,gdepw,tmask,wmask'
  do jj = 1, ny
    do ji = 1, nx
      do jk = 1, nz
        write(u, '(3(I0,","),ES25.17E3,3(",",ES25.17E3))') ji, jj, jk, &
            e3t_3d(ji, jj, jk), gdepw_3d(ji, jj, jk), tmask(ji, jj, jk), wmask(ji, jj, jk)
      end do
    end do
  end do
  close(u)

  open(newunit=v, file=trim(dir)//'/traqsr-steps.csv', status='replace')
  write(v, '(A)') 'kt,ji,jj,jk,qsr,ta_rhs,qsr_hc,qsr_hc_b,fraqsr_1lev'
  do kt = nit000, nit000 + nstep - 1
    do jj = 1, ny
      do ji = 1, nx
        ! noon-ish to night: one column dark, one at full sun
        qsr(ji, jj) = 900._wp * frac(ji * 11 + jj * 13 + kt * 17, 59)
      end do
    end do
    qsr(2, 2) = 0._wp
    qsr(6, 4) = 1000._wp
    pts(:,:,:,:,:) = 0._wp
    call tra_qsr(kt, 1, pts, 3)
    do jj = 1, ny
      do ji = 1, nx
        do jk = 1, nz
          write(v, '(4(I0,","),ES25.17E3,4(",",ES25.17E3))') kt, ji, jj, jk, &
              qsr(ji, jj), pts(ji, jj, jk, jp_tem, 3), qsr_hc(ji, jj, jk), &
              qsr_hc_b(ji, jj, jk), fraqsr_1lev(ji, jj)
        end do
      end do
    end do
  end do
  close(v)

contains

  real(wp) function frac(m, p)
    integer, intent(in) :: m, p
    frac = real(mod(m * p * 7919, 1009), wp) / 1009._wp
  end function frac

end program run_nemo_traqsr_oracle
