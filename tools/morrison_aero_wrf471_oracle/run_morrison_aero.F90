program run_morrison_aero
  ! Drive WRF v4.7.1's public MP_MORR_TWO_MOMENT_AERO (mp_physics = 40)
  ! over the columns written by aero_columns.py, for NSTEPS consecutive
  ! calls (the output state of one call is the input of the next; P, PII,
  ! RHO, DZ, W, KZH and the prescribed aerosol stay fixed, as they do across
  ! one model step's microphysics call).  Every output written here is a
  ! value WRF returned; this program never reimplements Morrison arithmetic.
  !
  ! usage: run_morrison_aero IN.bin OUT.bin NSTEPS
  ! IN  (stream, little endian): int32 ncol, nz, mode, aercu_opt; real32 dt,
  !     aercu_fct; then the 19 level inputs th qv qc qr qi qs qg ni ns nr ng
  !     nc rho pii p dz qrcuten qscuten qicuten, each (ncol, nz) Fortran
  !     order; then w and kzh, each (ncol, nz+1) (WRF's w-level staggering:
  !     the driver reads W(K) and KZH(K+1)); then aerocu (ncol, nz, 10) in
  !     WRF's species order (dust1..4, seasalt, sulfate, phobcar, phibcar,
  !     phoocar, phiocar); then the 7 surface inputs rainnc rainncv snownc
  !     snowncv graupelnc graupelncv sr, each (ncol).
  ! OUT: the 25 level outputs th qv qc qr qi qs qg ni ns nr ng nc refl_10cm
  !     mskf_refl_10cm ccn1_gs..ccn7_gs efcg efig efsg wact, each (ncol, nz),
  !     then the 7 surface outputs (ncol), after the last call.
  !
  ! The cumulus hydrometeors MSKF hands the scheme (NR_CU, QR_CU, NS_CU,
  ! QS_CU, CU_UAF) are zero: woof has no cu_physics = 11.
  use module_mp_morr_two_moment_aero, only: morr_two_moment_init_aero, &
                                             mp_morr_two_moment_aero
  implicit none
  integer, parameter :: nsrc = 10
  integer :: ncol, nz, mode, aercu_opt, uin, uout, nsteps, step, kme
  real :: dt, aercu_fct
  real, allocatable, dimension(:,:,:) :: th, qv, qc, qr, qi, qs, qg, ni,  &
       ns, nr, ng, nc, rho, pii, p, dz, qrcu, qscu, qicu, w, kzh, refl,    &
       mskf_refl, ccn1, ccn2, ccn3, ccn4, ccn5, ccn6, ccn7, efcg, efig,    &
       efsg, wact, nr_cu, qr_cu, ns_cu, qs_cu
  real, allocatable, dimension(:,:,:,:) :: aerocu
  real, allocatable, dimension(:,:) :: ht, cu_uaf, rainnc, rainncv,      &
       snownc, snowncv, graupelnc, graupelncv, sr
  character(len=1024) :: inpath, outpath, arg

  call get_command_argument(1, inpath)
  call get_command_argument(2, outpath)
  call get_command_argument(3, arg)
  read(arg, *) nsteps
  if (nsteps < 1) stop 'NSTEPS must be >= 1'
  open(newunit=uin, file=trim(inpath), access='stream', form='unformatted', &
       status='old', action='read')
  read(uin) ncol, nz, mode, aercu_opt, dt, aercu_fct
  kme = nz + 1
  allocate(th(ncol,kme,1), qv(ncol,kme,1), qc(ncol,kme,1), qr(ncol,kme,1),  &
           qi(ncol,kme,1), qs(ncol,kme,1), qg(ncol,kme,1), ni(ncol,kme,1),  &
           ns(ncol,kme,1), nr(ncol,kme,1), ng(ncol,kme,1), nc(ncol,kme,1),  &
           rho(ncol,kme,1), pii(ncol,kme,1), p(ncol,kme,1), dz(ncol,kme,1), &
           qrcu(ncol,kme,1), qscu(ncol,kme,1), qicu(ncol,kme,1),            &
           w(ncol,kme,1), kzh(ncol,kme,1), refl(ncol,kme,1),                &
           mskf_refl(ncol,kme,1), ccn1(ncol,kme,1), ccn2(ncol,kme,1),       &
           ccn3(ncol,kme,1), ccn4(ncol,kme,1), ccn5(ncol,kme,1),            &
           ccn6(ncol,kme,1), ccn7(ncol,kme,1), efcg(ncol,kme,1),            &
           efig(ncol,kme,1), efsg(ncol,kme,1), wact(ncol,kme,1),            &
           nr_cu(ncol,kme,1), qr_cu(ncol,kme,1), ns_cu(ncol,kme,1),         &
           qs_cu(ncol,kme,1), aerocu(ncol,kme,1,nsrc))
  allocate(ht(ncol,1), cu_uaf(ncol,1), rainnc(ncol,1), rainncv(ncol,1),    &
           snownc(ncol,1), snowncv(ncol,1), graupelnc(ncol,1),             &
           graupelncv(ncol,1), sr(ncol,1))
  ! The memory level kme = nz + 1 of a mass-level field is never read.
  th = 0.0; qv = 0.0; qc = 0.0; qr = 0.0; qi = 0.0; qs = 0.0; qg = 0.0
  ni = 0.0; ns = 0.0; nr = 0.0; ng = 0.0; nc = 0.0; rho = 0.0; pii = 0.0
  p = 0.0; dz = 0.0; qrcu = 0.0; qscu = 0.0; qicu = 0.0; aerocu = 0.0
  call rd(th); call rd(qv); call rd(qc); call rd(qr); call rd(qi)
  call rd(qs); call rd(qg); call rd(ni); call rd(ns); call rd(nr)
  call rd(ng); call rd(nc); call rd(rho); call rd(pii); call rd(p)
  call rd(dz); call rd(qrcu); call rd(qscu); call rd(qicu)
  read(uin) w(:, 1:kme, 1)
  read(uin) kzh(:, 1:kme, 1)
  read(uin) aerocu(:, 1:nz, 1, :)
  read(uin) rainnc, rainncv, snownc, snowncv, graupelnc, graupelncv, sr
  close(uin)
  ht = 0.0
  cu_uaf = 0.0
  nr_cu = 0.0; qr_cu = 0.0; ns_cu = 0.0; qs_cu = 0.0
  refl = sign(0.0, -1.0)
  mskf_refl = sign(0.0, -1.0)
  ccn1 = 0.0; ccn2 = 0.0; ccn3 = 0.0; ccn4 = 0.0; ccn5 = 0.0; ccn6 = 0.0
  ccn7 = 0.0; efcg = 0.0; efig = 0.0; efsg = 0.0; wact = 0.0

  call morr_two_moment_init_aero(mode)
  do step = 1, nsteps
     call mp_morr_two_moment_aero(                                         &
          itimestep=step, th=th, qv=qv, qc=qc, qr=qr, qi=qi, qs=qs, qg=qg, &
          ni=ni, ns=ns, nr=nr, ng=ng, nc=nc, rho=rho, pii=pii, p=p,        &
          dt_in=dt, dz=dz, ht=ht, w=w, kzh=kzh,                            &
          rainnc=rainnc, rainncv=rainncv,                                  &
          sr=sr, snownc=snownc, snowncv=snowncv, graupelnc=graupelnc,      &
          graupelncv=graupelncv, refl_10cm=refl,                           &
          mskf_refl_10cm=mskf_refl, diagflag=.true.,                       &
          do_radar_ref=1, qrcuten=qrcu, qscuten=qscu, qicuten=qicu,        &
          ids=1, ide=ncol, jds=1, jde=1, kds=1, kde=kme,                   &
          ims=1, ime=ncol, jms=1, jme=1, kms=1, kme=kme,                   &
          its=1, ite=ncol, jts=1, jte=1, kts=1, kte=nz,                    &
          pbl=1, aerocu=aerocu, aercu_opt=aercu_opt,                       &
          aercu_fct=aercu_fct, no_src_types_cu=nsrc,                       &
          efcg=efcg, efig=efig, efsg=efsg, wact=wact,                      &
          ccn1_gs=ccn1, ccn2_gs=ccn2, ccn3_gs=ccn3, ccn4_gs=ccn4,          &
          ccn5_gs=ccn5, ccn6_gs=ccn6, ccn7_gs=ccn7, nr_cu=nr_cu,           &
          qr_cu=qr_cu, ns_cu=ns_cu, qs_cu=qs_cu, cu_uaf=cu_uaf)
  end do

  open(newunit=uout, file=trim(outpath), access='stream',                 &
       form='unformatted', status='replace', action='write')
  call wo(th); call wo(qv); call wo(qc); call wo(qr); call wo(qi)
  call wo(qs); call wo(qg); call wo(ni); call wo(ns); call wo(nr)
  call wo(ng); call wo(nc); call wo(refl); call wo(mskf_refl)
  call wo(ccn1); call wo(ccn2); call wo(ccn3); call wo(ccn4)
  call wo(ccn5); call wo(ccn6); call wo(ccn7)
  call wo(efcg); call wo(efig); call wo(efsg); call wo(wact)
  write(uout) rainnc, rainncv, snownc, snowncv, graupelnc, graupelncv, sr
  close(uout)
contains
  subroutine rd(a)
    real, intent(inout) :: a(:,:,:)
    read(uin) a(:, 1:nz, 1)
  end subroutine rd
  subroutine wo(a)
    real, intent(in) :: a(:,:,:)
    write(uout) a(:, 1:nz, 1)
  end subroutine wo
end program run_morrison_aero
