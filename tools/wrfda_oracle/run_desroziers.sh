#!/bin/bash
# WRFDA's randomised-trace Desroziers workflow on the tutorial case (inside the oracle container):
# an unperturbed run (writes unpert_obs, gts_omb_oma, jo), a run with omb_add_noise (writes pert_obs and
# rand_obs_error), then da_tune_obs_desroziers on them.  Needs run_tutorial.sh from this directory
# (with WRFVAR1/WRFVAR5 support) at $W/scripts/run_tutorial.sh and the pinned build's
# var/build/da_tune_obs_desroziers.exe.  Results: $W/run/des-tool/tool.out.
# usage: run_desroziers.sh        (W defaults to /d/scratch/wrfda)
set -eux -o pipefail
W=${W:-/d/scratch/wrfda}; : "${W:?}"; R=$W/run; S=$W/src/WRFV4.7.1
WRFVAR1='write_unpert_obs=.true.,' bash "$W/scripts/run_tutorial.sh" des-unpert
WRFVAR1='write_unpert_obs=.true.,' WRFVAR5='omb_add_noise=.true.,' bash "$W/scripts/run_tutorial.sh" des-pert
rm -rf "$R/des-tool"; mkdir "$R/des-tool"; cd "$R/des-tool"
cp ../des-pert/rand_obs_error fort.45
cp ../des-pert/pert_obs.0000 fort.46
cp ../des-unpert/unpert_obs.0000 fort.47
(cat ../des-unpert/jo; echo "*****") > fort.48
(cat ../des-unpert/wrfda.out; echo "*****") > fort.49
printf '&rtminit\n rtminit_nsensor=0,\n/\n' > namelist.radiance
"$S/var/build/da_tune_obs_desroziers.exe" > tool.out 2>&1
for f in rand_obs_error pert_obs.0000 unpert_obs.0000 gts_omb_oma_01.0000 tool.out; do
  for dd in "$R/des-tool" "$R/des-pert" "$R/des-unpert"; do [ -s "$dd/$f" ] && continue 2; done
  echo "missing or empty output $f" >&2; exit 1
done
grep -q "Total Jo factor" "$R/des-tool/tool.out" || { echo "da_tune_obs_desroziers did not finish (no Total Jo factor)" >&2; exit 1; }
