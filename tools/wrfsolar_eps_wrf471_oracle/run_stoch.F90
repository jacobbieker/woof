! Driver for WRF v4.7.1's multi_perturb pattern arithmetic (module_stoch.F),
! one STOCHPERT.TBL field on one tile spanning the domain, as
! INITIALIZE_STOCH (:375-445) and first_rk_step_part2 (:267-317) call it:
!
!   SETUP_RAND_PERTURB('R', vertstruc, ...)  with INITIALIZE_STOCH's
!     permuted band arguments (KMINFORCT = 1, KMAXFORCTH = 1000000,
!     LMINFORCT = 1, LMAXFORCTH = 1000000 -- the namelist defaults) and
!     itime_step = stepsp*time_step;
!   per update: UPDATE_STOCH's AR(1) step on the supplied draws (ar1_cut),
!     then RAND_PERT_UPDATE 'T': the fill (fill_cut), the vertical
!     structure (vert_cut), the serial-build backward FFTs
!     (do_fftback_along_x then _y over kps..min(kpe, num_stoch_levels)),
!     and the clip (clip_cut) over kts..min(kte, num_stoch_levels).
!
! ids = 1, ide = LMAX, jds = 1, jde = KMAX; levels: kte = nz (k_end, the
! mass top), kme = kpe = num_stoch_levels = nz + 1 (e_vert).
!
! Input stoch_inputs.bin (stream, little endian):
!   int ncase; per case: int lmax, kmax, nz, itime_step, vertstruc, nstep;
!   real dx, dy, std, lengthscale, timescale, cutoff;
!   then per step real noss(lmax, kmax), nosc(lmax, kmax)
! Output stoch.out.bin, per case: real alph, sp_amp(lmax, kmax); per step:
!   spforcc, spforcs (lmax, kmax); RAND_REAL and RAND_IMAG after the
!   vertical structure, level by level (lmax, kmax) for k = 1..nz;
!   RAND_REAL after the FFTs, likewise; RAND_PERT, likewise.
program run_stoch
  use stoch_cut
  implicit none
  integer :: ncase, c, s, k, u, v
  integer :: lmax, kmax, nz, itime_step, vertstruc, nstep, kme
  integer :: kmx_out, lmx_out
  real :: dx, dy, std, lscale, tscale, cutoff, alph_r
  real, allocatable :: sp_amp(:,:), spforcc(:,:), spforcs(:,:), noss(:,:), nosc(:,:)
  real, allocatable :: vsc(:,:,:), vss(:,:,:), vamp(:), rand_pert(:,:,:)
  type(stoch_grid) :: grid

  open(newunit=u, file='stoch_inputs.bin', access='stream', form='unformatted', &
       status='old', convert='little_endian')
  open(newunit=v, file='stoch.out.bin', access='stream', form='unformatted', &
       status='replace', convert='little_endian')
  read(u) ncase
  do c = 1, ncase
    read(u) lmax, kmax, nz, itime_step, vertstruc, nstep
    read(u) dx, dy, std, lscale, tscale, cutoff
    kme = nz + 1
    allocate(sp_amp(lmax, kmax), spforcc(lmax, kmax), spforcs(lmax, kmax), &
             noss(lmax, kmax), nosc(lmax, kmax), vsc(lmax, kme, kmax), &
             vss(lmax, kme, kmax), vamp(kme), rand_pert(lmax, kme, kmax))
    allocate(grid%RAND_REAL(lmax, kme, kmax), grid%RAND_IMAG(lmax, kme, kmax))
    grid%i_start(1) = 1; grid%i_end(1) = lmax
    grid%j_start(1) = 1; grid%j_end(1) = kmax
    grid%num_stoch_levels = kme
    spforcc = 0.; spforcs = 0.; sp_amp = 0.; vsc = 0.; vss = 0.
    grid%RAND_REAL = 0.; grid%RAND_IMAG = 0.; rand_pert = 0.
    call SETUP_RAND_PERTURB('R', vertstruc, .false., sp_amp, spforcc, spforcs, alph_r, &
        vsc, vss, vamp, 1, 1000000, 1, 1000000, kmx_out, lmx_out, &
        itime_step, dx, dy, std, lscale, tscale, 1.0E-05, 10800.0, -1.83, &
        1, lmax, 1, kmax, 1, kme, 1, lmax, 1, kmax, 1, kme, 1, lmax, 1, kmax, 1, kme)
    write(v) alph_r, sp_amp
    do s = 1, nstep
      read(u) noss, nosc
      call ar1_cut(spforcs, spforcc, sp_amp, alph_r, noss, nosc, &
                   1, lmax, 1, kmax, 1, lmax, 1, kmax, 1, lmax, 1, kmax)
      write(v) spforcc, spforcs
      call fill_cut(grid, spforcc, spforcs, 1, kme, 1, lmax, 1, kmax)
      call vert_cut(grid, vertstruc, vsc, vss, vamp, 1, kme, 1, lmax, 1, kme, 1, kmax)
      do k = 1, nz
        write(v) grid%RAND_REAL(:, k, :)
      end do
      do k = 1, nz
        write(v) grid%RAND_IMAG(:, k, :)
      end do
      call do_fftback_along_x(grid%RAND_REAL, grid%RAND_IMAG, 1, lmax, 1, kmax, &
          1, lmax, 1, kmax, 1, min(kme, grid%num_stoch_levels), &
          1, lmax, 1, kmax, 1, min(kme, grid%num_stoch_levels))
      call do_fftback_along_y(grid%RAND_REAL, grid%RAND_IMAG, 1, lmax, 1, kmax, &
          1, lmax, 1, kmax, 1, min(kme, grid%num_stoch_levels), &
          1, lmax, 1, kmax, 1, min(kme, grid%num_stoch_levels))
      do k = 1, nz
        write(v) grid%RAND_REAL(:, k, :)
      end do
      call clip_cut(grid, rand_pert, cutoff, std, 1, nz, 1, lmax, kme, 1, kmax)
      do k = 1, nz
        write(v) rand_pert(:, k, :)
      end do
    end do
    deallocate(sp_amp, spforcc, spforcs, noss, nosc, vsc, vss, vamp, rand_pert)
    deallocate(grid%RAND_REAL, grid%RAND_IMAG)
  end do
  close(u)
  close(v)
end program run_stoch
