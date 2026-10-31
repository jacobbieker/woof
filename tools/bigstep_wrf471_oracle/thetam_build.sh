#!/usr/bin/env bash
# Build and run the use_theta_m = 1 / do_avgflx_em = 1 oracle: WRF v4.7.1's
# use_theta_m branches (calc_p_rho_phi, phy_prep, moist_physics_prep_em,
# moist_physics_finish_em, initialize_moist_old, conv_t_tendf_to_moist and
# real.exe's theta -> theta_m block, extracted unchanged by thetam_build.py)
# and dyn_em/module_avgflx_em.F compiled whole, CPU only.
#
#   thetam_build.sh WRF_SOURCE_ROOT STATE_NPZ OUT_DIR
#
# Reference of record: gfortran -O0 -ffp-contract=off against glibc 2.43's
# libm (Ubuntu 26.04), the C library of the repository's WRF v4.7.1 column
# oracles.  The only libm call is calc_p_rho_phi's vspow (temp**cpovcv,
# powf); the script refuses any other libm.  On pb the record was made as
#
#   E=$(pixi exec -s gfortran sh -c 'echo $CONDA_PREFIX')
#   docker run --rm -u $(id -u):$(id -g) -v $E:$E:ro -v $WRF:/wrf:ro \
#       -v $PWD:$PWD -w $PWD -e PATH=$E/bin:/usr/bin:/bin \
#       -e PYTHON=$PWD/.pixi/envs/default/bin/python ubuntu:26.04 \
#       bash tools/bigstep_wrf471_oracle/thetam_build.sh /wrf \
#       tests/data/wrf471_bigstep/state-real.npz OUT_DIR
#
# Writes OUT_DIR/{thetam_inputs.bin, thetam_inputs.npz, thetam_outputs.bin,
# receipt.txt}.  tests/test_bigstep_thetam_wrf471_parity.py grades the port.
set -euo pipefail
[[ $# -eq 3 ]] || { echo "usage: thetam_build.sh WRF_SOURCE_ROOT STATE_NPZ OUT_DIR" >&2; exit 2; }
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
"$python" "$here/thetam_build.py" "$root" .
"$python" "$here/thetam_cases.py" "$state" .
flags="-O0 -ffp-contract=off -fno-fast-math -cpp -ffree-form -ffree-line-length-none"
gfortran $flags -c "$here/thetam_services.F90" -o thetam_services.o
gfortran $flags -c "$root/share/module_model_constants.F" -o module_model_constants.o
# libmassv.F is fixed-form F77 (WRF's IBM MASS compatibility library).
gfortran -O0 -ffp-contract=off -fno-fast-math -cpp -c "$root/frame/libmassv.F" -o libmassv.o
gfortran $flags -c "$root/dyn_em/module_avgflx_em.F" -o module_avgflx_em.o
gfortran $flags -c thetam_exact.F90 -o thetam_exact.o
gfortran $flags -c "$here/thetam_run.F90" -o thetam_run.o
gfortran thetam_services.o module_model_constants.o libmassv.o \
    module_avgflx_em.o thetam_exact.o thetam_run.o -o thetam_run
./thetam_run
{
    echo "wrf v4.7.1 commit f52c197ed39d12e087d02c50f412d90d418f6186"
    gfortran --version | sed -n 1p
    echo "flags: $flags"
    echo "${glibc:-libm: host}"
    sha256sum "$root/dyn_em/module_big_step_utilities_em.F" \
        "$root/dyn_em/module_initialize_real.F" "$root/dyn_em/module_avgflx_em.F" \
        "$root/frame/libmassv.F" "$root/share/module_model_constants.F" \
        "$here/thetam_build.py" "$here/thetam_cases.py" "$here/thetam_run.F90" \
        "$here/thetam_services.F90" "$here/thetam_build.sh" \
        thetam_exact.F90 thetam_inputs.bin thetam_outputs.bin \
        | sed "s#$root/##; s#$here/#tools/bigstep_wrf471_oracle/#"
} > receipt.txt
cat receipt.txt
