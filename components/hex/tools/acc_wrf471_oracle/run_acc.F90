! Drives WRF v4.7.1's phys/module_diag_misc.F accumulation blocks on the per-step inputs of inputs.bin
! (written by make_inputs.py) and writes the accumulators after every step.
!
! The two blocks are WRF's own lines, cut from the byte-unmodified file by build.sh (acc_phy_tend: lines 223-286,
! prec_acc_dt: lines 432-462) and #included below; only the declarations and the step loop are written here.  One
! mesh cell is one WRF column (i = cell, j = 1, one tile).  curr_secs2 is the START time of the step, as WRF's.
!
!   run_acc inputs.bin out.bin
! out.bin: per step, float32: prec_acc_c[nc], prec_acc_nc[nc], snow_acc_nc[nc], then the sixteen (nl, nc)
! accumulators athmp aqvmp athcu aqvcu aucu avcu athsh aqvsh aush avsh athbl aqvbl aubl avbl athralw athrasw.
subroutine wrf_debug(level, message)
  integer, intent(in) :: level
  character(len=*), intent(in) :: message
end subroutine wrf_debug

program run_acc
  implicit none
  integer :: nc, nl, ns, cu_used, shcu_used, history_interval, itimestep, step
  integer :: acc_phy_tend, num_tiles, ij, i, j, k, kps, kpe, hmin, unit_in, unit_out
  integer :: i_start(1), i_end(1), j_start(1), j_end(1)
  real :: dt, prec_acc_dt, curr_secs2
  character(len=256) :: outstring, infile, outfile
  real, allocatable :: raincv(:,:), rainncv(:,:), snowncv(:,:), t2(:,:)
  real, allocatable :: prec_acc_c(:,:), prec_acc_nc(:,:), snow_acc_nc(:,:)
  real, allocatable :: h_diabatic(:,:,:), qv_diabatic(:,:,:), rthcuten(:,:,:), rqvcuten(:,:,:), rucuten(:,:,:), rvcuten(:,:,:)
  real, allocatable :: rthshten(:,:,:), rqvshten(:,:,:), rushten(:,:,:), rvshten(:,:,:)
  real, allocatable :: rthblten(:,:,:), rqvblten(:,:,:), rublten(:,:,:), rvblten(:,:,:), rthratenlw(:,:,:), rthratensw(:,:,:)
  real, allocatable :: athmp(:,:,:), aqvmp(:,:,:), athcu(:,:,:), aqvcu(:,:,:), aucu(:,:,:), avcu(:,:,:)
  real, allocatable :: athsh(:,:,:), aqvsh(:,:,:), aush(:,:,:), avsh(:,:,:)
  real, allocatable :: athbl(:,:,:), aqvbl(:,:,:), aubl(:,:,:), avbl(:,:,:), athralw(:,:,:), athrasw(:,:,:)
  call get_command_argument(1, infile)
  call get_command_argument(2, outfile)
  open(newunit=unit_in, file=trim(infile), access='stream', form='unformatted', status='old')
  open(newunit=unit_out, file=trim(outfile), access='stream', form='unformatted', status='replace')
  read(unit_in) nc, nl, ns, cu_used
  read(unit_in) dt, prec_acc_dt
  read(unit_in) hmin
  history_interval = hmin
  shcu_used = 0
  acc_phy_tend = 1
  num_tiles = 1; ij = 1
  i_start = 1; i_end = nc; j_start = 1; j_end = 1; kps = 1; kpe = nl
  allocate(raincv(nc,1), rainncv(nc,1), snowncv(nc,1), t2(nc,1), prec_acc_c(nc,1), prec_acc_nc(nc,1), snow_acc_nc(nc,1))
  allocate(h_diabatic(nc,nl,1), qv_diabatic(nc,nl,1), rthcuten(nc,nl,1), rqvcuten(nc,nl,1), rucuten(nc,nl,1), rvcuten(nc,nl,1))
  allocate(rthshten(nc,nl,1), rqvshten(nc,nl,1), rushten(nc,nl,1), rvshten(nc,nl,1))
  allocate(rthblten(nc,nl,1), rqvblten(nc,nl,1), rublten(nc,nl,1), rvblten(nc,nl,1), rthratenlw(nc,nl,1), rthratensw(nc,nl,1))
  allocate(athmp(nc,nl,1), aqvmp(nc,nl,1), athcu(nc,nl,1), aqvcu(nc,nl,1), aucu(nc,nl,1), avcu(nc,nl,1))
  allocate(athsh(nc,nl,1), aqvsh(nc,nl,1), aush(nc,nl,1), avsh(nc,nl,1))
  allocate(athbl(nc,nl,1), aqvbl(nc,nl,1), aubl(nc,nl,1), avbl(nc,nl,1), athralw(nc,nl,1), athrasw(nc,nl,1))
  rthshten = 0.; rqvshten = 0.; rushten = 0.; rvshten = 0.
  prec_acc_c = 0.; prec_acc_nc = 0.; snow_acc_nc = 0.
  athmp = 0.; aqvmp = 0.; athcu = 0.; aqvcu = 0.; aucu = 0.; avcu = 0.; athsh = 0.; aqvsh = 0.; aush = 0.; avsh = 0.
  athbl = 0.; aqvbl = 0.; aubl = 0.; avbl = 0.; athralw = 0.; athrasw = 0.
  do step = 1, ns
    itimestep = step
    curr_secs2 = real(step - 1) * dt
    read(unit_in) raincv(:,1), rainncv(:,1), snowncv(:,1), t2(:,1)
    read(unit_in) h_diabatic, qv_diabatic, rthcuten, rqvcuten, rucuten, rvcuten, rthblten, rqvblten, rublten, rvblten, &
                  rthratenlw, rthratensw
#include "block_acc_phy_tend.inc"
#include "block_prec_acc_dt.inc"
    write(unit_out) prec_acc_c, prec_acc_nc, snow_acc_nc
    write(unit_out) athmp, aqvmp, athcu, aqvcu, aucu, avcu, athsh, aqvsh, aush, avsh, athbl, aqvbl, aubl, avbl, athralw, athrasw
  end do
  close(unit_in); close(unit_out)
end program run_acc
