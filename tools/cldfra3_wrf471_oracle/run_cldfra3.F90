! Column driver for WRF v4.7.1 cal_cldfra3 (icloud = 3), called as the
! radiation driver calls it (module_radiation_driver.F:1462-1510): copies of
! one column, modify_qvapor = .false., max_relh = 1.5, debug_flag = .false.
!
! Input  columns.bin (stream, little endian):
!   int ncol; then per column: int nz, real dx, dy, xland,
!   real qv(nz), qc(nz), qi(nz), qs(nz), dz8w(nz), p(nz), t(nz)
! Output cldfra3.out.bin: per column real gridkm, cldfra(nz), qc(nz), qi(nz)
!
! gridkm is the driver's own statement for the dx branch (:1055-1056):
! first_rk_step_part1 passes dx and dy, never dxkm.
program run_cldfra3
  use cldfra3_cut, only: cal_cldfra3
  implicit none
  integer :: ncol, nz, c, u, v
  real :: dx, dy, xland, gridkm
  real, allocatable :: qv(:), qc(:), qi(:), qs(:), dz(:), p(:), t(:), cf(:)

  open(newunit=u, file='columns.bin', access='stream', form='unformatted', &
       status='old', convert='little_endian')
  open(newunit=v, file='cldfra3.out.bin', access='stream', form='unformatted', &
       status='replace', convert='little_endian')
  read(u) ncol
  do c = 1, ncol
    read(u) nz, dx, dy, xland
    allocate(qv(nz), qc(nz), qi(nz), qs(nz), dz(nz), p(nz), t(nz), cf(nz))
    read(u) qv, qc, qi, qs, dz, p, t
    gridkm = SQRT(dx*0.001*dx*0.001 + dy*0.001*dy*0.001)
    cf = 0.
    call cal_cldfra3(cf, qv, qc, qi, qs, dz, p, t, xland, gridkm, &
                     .false., 1.5, 1, nz, .false.)
    write(v) gridkm, cf, qc, qi
    deallocate(qv, qc, qi, qs, dz, p, t, cf)
  end do
  close(u)
  close(v)
end program run_cldfra3
