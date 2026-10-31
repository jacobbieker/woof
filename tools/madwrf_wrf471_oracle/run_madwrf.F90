! Driver for WRF v4.7.1's MAD-WRF routines, called as real.exe and
! solve_em call them.
!
! Init_madwrf_clouds (module_initialize_real.F:4861-4866) on one column per
! call: i = 1 of ids = 1, ide = 2; j = 1 of jds = 1, jde = 2; kts = 1,
! kte = nz + 1 (real.exe's tile is the staggered patch); hold_ups = .true.
! (the first time, :436), em_width = 5; p00 = 1e5.
!
! Cloud_tracer_nudge (solve_em.F:3646-3662) on an (ni, nk, nj) block with a
! tile that leaves a spec-zone ring out, as scalar_tile_loop_1a's its..ite.
!
! Input madwrf_columns.bin (stream, little endian):
!   int ncol; per column: int nz, flag_cldmask, flag_cldtopz, flag_cldbasez,
!   flag_brtemp; real dx, dy, xland, ht, cldmask, cldtopz, cldbasez, brtemp;
!   real t2(nz), p(nz), ph(nz+1), phb(nz+1), qv(nz), qc(nz), qi(nz), qs(nz)
!   int ncase; per case: int ni, nk, nj, its, ite, jts, jte;
!   real dt, dt_relax, dt_nudge, xtime; real qc, qi, qs, trqc, trqi, trqs
!   (ni, nk+1, nj) each
! Output madwrf.out.bin: per column real qv, qc, qi, qs (nz), cldfra (nz+1),
!   cldmask, cldtopz; per nudge case real qc, qi, qs (ni, nk+1, nj).
program run_madwrf
  use module_madwrf, only: Init_madwrf_clouds
  use nudge_cut, only: Cloud_tracer_nudge
  implicit none
  integer :: ncol, c, nz, u, v, fm, ft, fb, fbr, kme, ncase, ni, nk, nj
  integer :: its, ite, jts, jte
  real :: dx, dy, xland1, ht1, cm1, ct1, cb1, br1, dt, dt_relax, dt_nudge, xtime
  real, allocatable :: moist(:,:,:,:), t(:,:,:), p(:,:,:), ph(:,:,:), phb(:,:,:), &
      alt(:,:,:), cldfra(:,:,:)
  real :: xland(2,2), ht(2,2), cldmask(2,2), cldtopz(2,2), cldbasez(2,2), brtemp(2,2)
  real, allocatable :: q(:,:,:,:)

  open(newunit=u, file='madwrf_columns.bin', access='stream', form='unformatted', &
       status='old', convert='little_endian')
  open(newunit=v, file='madwrf.out.bin', access='stream', form='unformatted', &
       status='replace', convert='little_endian')
  read(u) ncol
  do c = 1, ncol
    read(u) nz, fm, ft, fb, fbr
    read(u) dx, dy, xland1, ht1, cm1, ct1, cb1, br1
    kme = nz + 1
    allocate(moist(2, kme, 2, 4), t(2, kme, 2), p(2, kme, 2), ph(2, kme, 2), &
             phb(2, kme, 2), alt(2, kme, 2), cldfra(2, kme, 2))
    moist = 0.; t = 0.; p = 0.; ph = 0.; phb = 0.; alt = 1.; cldfra = 0.
    read(u) t(1, 1:nz, 1), p(1, 1:nz, 1), ph(1, 1:kme, 1), phb(1, 1:kme, 1), &
            moist(1, 1:nz, 1, 1), moist(1, 1:nz, 1, 2), moist(1, 1:nz, 1, 3), &
            moist(1, 1:nz, 1, 4)
    xland = xland1; ht = ht1; cldmask = cm1; cldtopz = ct1; cldbasez = cb1; brtemp = br1
    call Init_madwrf_clouds (moist, 1, 2, 3, 4, 1.0E5, t, p, ph, phb, &
        alt, xland, cldmask, cldtopz, cldbasez, &
        brtemp, ht, dx, dy, fm, ft, fb, fbr, 5, .true., 1, 2, 1, 2, &
        1, 1, 2, 1, 2, 1, kme, 1, 1, 1, 1, kme, cldfra)
    write(v) moist(1, 1:nz, 1, 1), moist(1, 1:nz, 1, 2), moist(1, 1:nz, 1, 3), &
             moist(1, 1:nz, 1, 4), cldfra(1, 1:kme, 1), cldmask(1, 1), cldtopz(1, 1)
    deallocate(moist, t, p, ph, phb, alt, cldfra)
  end do
  read(u) ncase
  do c = 1, ncase
    read(u) ni, nk, nj, its, ite, jts, jte
    read(u) dt, dt_relax, dt_nudge, xtime
    allocate(q(ni, nk + 1, nj, 6))
    read(u) q
    call Cloud_tracer_nudge (dt, dt_relax, dt_nudge, xtime, q(:,:,:,1), q(:,:,:,2), &
        q(:,:,:,3), q(:,:,:,4), q(:,:,:,5), q(:,:,:,6), 1, ni + 1, 1, nj + 1, &
        1, nk + 1, 1, ni, 1, nj, 1, nk + 1, its, ite, jts, jte, 1, nk)
    write(v) q(:,:,:,1), q(:,:,:,2), q(:,:,:,3)
    deallocate(q)
  end do
  close(u)
  close(v)
end program run_madwrf
