#!/usr/bin/env bash
# Build and run the second use_theta_m = 1 oracle: WRF v4.7.1's phy_prep and
# phy_prep_part2 (its use_theta_m RTHFTEN block) and vertical_diffusion_2's
# hflux block (its use_theta_m surface heat flux), extracted unchanged by
# thetam_ext_build.py, CPU only.
#
#   thetam_ext_build.sh WRF_SOURCE_ROOT STATE_NPZ OUT_DIR
#
# Reference of record: gfortran -O0 -ffp-contract=off against glibc 2.43's
# libm (Ubuntu 26.04), as thetam_build.sh.  phy_prep's only libm call is its
# (p/p1000mb)**rcp (powf); the script refuses any other libm.  On pb the
# record was made as
#
#   E=$(pixi exec -s gfortran sh -c 'echo $CONDA_PREFIX')
#   docker run --rm -u $(id -u):$(id -g) -v $E:$E:ro -v $WRF:/wrf:ro \
#       -v $PWD:$PWD -w $PWD -e PATH=$E/bin:/usr/bin:/bin \
#       -e PYTHON=$PWD/.pixi/envs/default/bin/python ubuntu:26.04 \
#       bash tools/bigstep_wrf471_oracle/thetam_ext_build.sh /wrf \
#       tests/data/wrf471_bigstep/state-real.npz OUT_DIR
#
# Writes OUT_DIR/{thetam_ext_inputs.bin, thetam_ext_inputs.npz,
# thetam_ext_outputs.bin, receipt.txt}.
# tests/test_thetam_ext_wrf471_parity.py grades the port.
set -euo pipefail
[[ $# -eq 3 ]] || { echo "usage: thetam_ext_build.sh WRF_SOURCE_ROOT STATE_NPZ OUT_DIR" >&2; exit 2; }
root=$(realpath "$1")
state=$(realpath "$2")
mkdir -p "$3"
out=$(realpath "$3")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python=${PYTHON:-python3}
glibc=$(ldd --version 2>/dev/null | sed -n 1p || true)
case "$glibc" in
    *" 2.43"*) ;;
    *) [[ -n "${THETAM_ANY_LIBM:-}" ]] || { echo "libm is not glibc 2.43 ($glibc); see the header (THETAM_ANY_LIBM=1 builds a non-record run)" >&2; exit 4; } ;;
esac

cd "$out"
"$python" "$here/thetam_ext_build.py" "$root" .
"$python" "$here/thetam_ext_cases.py" "$state" .
flags="-O0 -ffp-contract=off -fno-fast-math -cpp -ffree-form -ffree-line-length-none"
gfortran $flags -c "$here/thetam_ext_services.F90" -o thetam_ext_services.o
gfortran $flags -c "$root/share/module_model_constants.F" -o module_model_constants.o
# libmassv.F is fixed-form F77 (WRF's IBM MASS compatibility library).
gfortran -O0 -ffp-contract=off -fno-fast-math -cpp -c "$root/frame/libmassv.F" -o libmassv.o
gfortran $flags -c thetam_ext_exact.F90 -o thetam_ext_exact.o
gfortran $flags -c "$here/thetam_ext_run.F90" -o thetam_ext_run.o
gfortran thetam_ext_services.o module_model_constants.o libmassv.o \
    thetam_ext_exact.o thetam_ext_run.o -o thetam_ext_run
./thetam_ext_run
{
    echo "wrf v4.7.1 commit f52c197ed39d12e087d02c50f412d90d418f6186"
    gfortran --version | sed -n 1p
    echo "flags: $flags"
    echo "${glibc:-libm: host}"
    sha256sum "$root/dyn_em/module_big_step_utilities_em.F" \
        "$root/dyn_em/module_diffusion_em.F" \
        "$root/frame/libmassv.F" "$root/share/module_model_constants.F" \
        "$here/thetam_ext_build.py" "$here/thetam_ext_cases.py" \
        "$here/thetam_cases.py" "$here/thetam_ext_run.F90" \
        "$here/thetam_ext_services.F90" "$here/thetam_ext_build.sh" \
        thetam_ext_exact.F90 thetam_ext_inputs.bin thetam_ext_outputs.bin \
        | sed "s#$root/##; s#$here/#tools/bigstep_wrf471_oracle/#"
} > receipt.txt
cat receipt.txt
