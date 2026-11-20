#!/bin/bash
# WRFDA 4.7.1 FGAT (num_fgat_time = 7) on the tutorial's 4D-Var data
# (2008-02-05 12-18Z, hourly ob01..ob07.ascii), inside the oracle container
# (the plan's V5.1 lane).  Two steps, each into $W/run/NAME:
#
#   fc  WRF 4.7.1 (wrf.exe of $WRF, the em_real dmpar build) from the
#       testdata's 12Z wrfinput/wrfbdy to 18Z with the 4D-Var namelist's
#       physics, writing a wrfinput-format file every hour (write_input).
#       These are the first guesses: fg01 is the 12Z wrfinput itself,
#       fg02..fg07 are 13..18Z.  NetCDF classic (use_netcdf_classic), as
#       the testdata's files are: woof's analysis bundle patches a classic
#       template and refuses NetCDF-4.
#   da  da_wrfvar.exe (TREE, default the dump build) with
#         &wrfvar3  num_fgat_time=7
#         analysis_date 15Z, time_window_min 12Z, time_window_max 18Z
#       fg is the 15Z file (= fg04), obNN.ascii are the testdata's hourly
#       files, be.dat the tutorial's.  The forecast's files are WRF V4
#       files of a V3 run (hybrid_opt = 0, use_theta_m = 0), so the
#       namelist says so too.
#
# Layout under $W (default /d/scratch/wrfda) as run_tutorial.sh; the WRF
# tree under $WRF (default /d/fortran/src/WRF-4.7.1).
# usage: run_fgat.sh fc NAME   then   run_fgat.sh da NAME
set -eux
STEP=${1:?usage: run_fgat.sh fc|da NAME}
NAME=${2:?usage: run_fgat.sh fc|da NAME}
case "$NAME" in */*|.*) echo "NAME must be a plain directory name" >&2; exit 2;; esac
W=${W:-/d/scratch/wrfda}; W=$(cd "$W" && pwd); D=$W/data; R=$W/run/$NAME
WRF=${WRF:-/d/fortran/src/WRF-4.7.1}
export HOME=/tmp OMPI_MCA_plm=isolated OMPI_MCA_btl_vader_single_copy_mechanism=none \
  OMPI_MCA_rmaps_base_oversubscribe=yes
H="2008-02-05_"
case "$STEP" in
fc)
  rm -rf "$R/fc"; mkdir -p "$R/fc"; cd "$R/fc"
  for f in "$WRF"/run/*; do b=$(basename "$f")
    case $b in namelist.input|*.exe|wrfinput_d01|wrfbdy_d01) ;; *) ln -sf "$f" "$b";; esac
  done
  cp "$WRF/main/wrf.exe" .
  cp "$D/rc/2008020512/wrfinput_d01" "$D/rc/2008020512/wrfbdy_d01" .
  cat > namelist.input <<'EOF'
&time_control
 run_days = 0, run_hours = 6, run_minutes = 0, run_seconds = 0,
 start_year = 2008, start_month = 02, start_day = 05, start_hour = 12, start_minute = 00, start_second = 00,
 end_year = 2008, end_month = 02, end_day = 05, end_hour = 18, end_minute = 00, end_second = 00,
 interval_seconds = 21600,
 input_from_file = .true.,
 history_interval = 60, frames_per_outfile = 1,
 restart = .false., restart_interval = 100000,
 write_input = .true., inputout_interval = 60, inputout_begin_h = 0, inputout_end_h = 6,
 input_outname = "fgat_input_d<domain>_<date>",
 io_form_history = 2, io_form_restart = 2, io_form_input = 2, io_form_boundary = 2,
 debug_level = 0,
 force_use_old_data = .true.,
 use_netcdf_classic = .true.,
/
&domains
 time_step = 360, max_dom = 1,
 e_we = 90, e_sn = 60, e_vert = 41,
 p_top_requested = 5000,
 dx = 60000, dy = 60000,
 grid_id = 1, parent_id = 0, i_parent_start = 1, j_parent_start = 1, parent_grid_ratio = 1, parent_time_step_ratio = 1,
 feedback = 0, smooth_option = 0,
 interp_type = 1, t_extrap_type = 1,
/
&physics
 mp_physics = 3, ra_lw_physics = 1, ra_sw_physics = 1, radt = 30,
 sf_sfclay_physics = 1, sf_surface_physics = 1, bl_pbl_physics = 1, bldt = 0,
 cu_physics = 1, cudt = 5, num_soil_layers = 5, mp_zero_out = 2, co2tf = 0,
 isfflx = 1, ifsnow = 0, icloud = 1,
/
&fdda
/
&dynamics
 hybrid_opt = 0, use_theta_m = 0,
 w_damping = 1, diff_opt = 1, km_opt = 4, diff_6th_opt = 0,
 damp_opt = 0, zdamp = 5000., dampcoef = 0.01,
 khdif = 0, kvdif = 0, non_hydrostatic = .true.,
 moist_adv_opt = 1, scalar_adv_opt = 1,
 time_step_sound = 4,
 base_temp = 290.0, use_baseparam_fr_nml = .true.,
/
&bdy_control
 spec_bdy_width = 5, spec_zone = 1, relax_zone = 4,
 specified = .true., nested = .false.,
/
&grib2
/
&namelist_quilt
 nio_tasks_per_group = 0, nio_groups = 1,
/
EOF
  mpirun -np "${NP:-4}" ./wrf.exe > wrf.log 2>&1
  grep -q "SUCCESS COMPLETE WRF" rsl.error.0000 || { echo "wrf.exe failed: see $R/fc/rsl.error.0000" >&2; exit 1; }
  for h in 13 14 15 16 17 18; do [ -s "fgat_input_d01_${H}$h:00:00" ]; done
  sha256sum wrfinput_d01 fgat_input_d01_* > fg.sha256
  ;;
da)
  S=$W/src/${TREE:-WRFV4.7.1-dump}
  [ -s "$R/fc/fg.sha256" ] || { echo "run 'run_fgat.sh fc $NAME' first" >&2; exit 2; }
  rm -rf "$R/da"; mkdir -p "$R/da"; cd "$R/da"
  ln -sf "$S/var/build/da_wrfvar.exe" .
  ln -sf "$S/run/LANDUSE.TBL" .
  cp "${BE:-$D/be/be.dat}" be.dat
  for n in 1 2 3 4 5 6 7; do
    h=$((11 + n)); cp "$D/ob/20080205$h/ob0$n.ascii" "ob0$n.ascii"
    if [ "$n" = 1 ]; then cp "$R/fc/wrfinput_d01" fg01; else cp "$R/fc/fgat_input_d01_${H}$h:00:00" "fg0$n"; fi
  done
  cp fg04 fg
  NL=$S/var/test/tutorial/namelist.input
  for l in "^ob_format=2," "^analysis_date=" "^time_window_min=" "^time_window_max=" "^start_hour=12," "^end_hour=12,"; do
    grep -q "$l" "$NL" || { echo "$NL has no $l line" >&2; exit 2; }
  done
  sed "s/^ob_format=2,/ob_format=2,\nnum_fgat_time=7,/" "$NL" \
    | sed "s/^analysis_date=.*/analysis_date=\"${H}15:00:00.0000\",/" \
    | sed "s/^time_window_min=.*/time_window_min=\"${H}12:00:00.0000\",/" \
    | sed "s/^time_window_max=.*/time_window_max=\"${H}18:00:00.0000\",/" \
    | sed "s/^start_hour=12,/start_hour=15,/; s/^end_hour=12,/end_hour=15,/" \
    | sed "s/^start_year=2008,/force_use_old_data=.true.,\nstart_year=2008,/" \
    | sed "s/^&dynamics\$/\&dynamics\nhybrid_opt=0,\nuse_theta_m=0,/" > namelist.input
  grep -q "^num_fgat_time=7," namelist.input && grep -q "^use_theta_m=0," namelist.input \
    || { echo "num_fgat_time or use_theta_m did not reach namelist.input" >&2; exit 2; }
  status=0
  /usr/bin/time -v ./da_wrfvar.exe > wrfda.out 2> time.txt || status=$?
  grep -E "Elapsed|Maximum resident" time.txt
  grep -q "WRF-Var completed successfully" rsl.out.0000 wrfda.out 2>/dev/null \
    || { echo "da_wrfvar.exe did not complete (exit $status)" >&2; exit 1; }
  grep -E "Reading first guess|Restore to first guess|Final cost" wrfda.out | tail -12
  sha256sum fg fg0? ob0?.ascii woof_*.bin > files.sha256
  ;;
*) echo "STEP must be fc or da" >&2; exit 2;;
esac
