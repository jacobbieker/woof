#!/usr/bin/env bash
# Build and run the WRF-Solar EPS (multi_perturb = 1) scheme-hook oracle from
# byte-unmodified WRF v4.7.1, CPU only.
#
#   build.sh WRF_SOURCE_ROOT OUT_DIR
#
# The hooks are the drivers' own Add_/Remove_multi_perturb_* pairs (FARMS,
# Deng, Thompson) and the icloud = 3 input statements (extract.py).  They
# are float32 multiplies, divides, subtractions and MAX/MIN only: no libm
# call, so the C library does not move the record.  The compiler's MAX/MIN
# does, on a signed-zero tie: gfortran -O0 on x86-64 emits maxss/minss,
# which return their source operand, in an operand order GCC picks per
# statement (Add_'s albedo clamp keeps +0.0, Remove_'s keeps -0.0), while
# aarch64's fmaxnm/fminnm order -0.0 below +0.0 everywhere.  The reference
# of record is x86-64, the platform WRF's operational builds run on; the
# script refuses any other.  On pb it was made in the repository's WRF
# v4.7.1 oracle container,
#
#   E=$(pixi exec -s gfortran sh -c 'echo $CONDA_PREFIX')
#   docker run --rm -u $(id -u):$(id -g) -v $E:$E:ro -v $WRF:/wrf:ro \
#       -v $PWD:$PWD -w $PWD -e PATH=$E/bin:/usr/bin:/bin \
#       -e PYTHON=$PWD/.pixi/envs/default/bin/python ubuntu:26.04 \
#       bash tools/wrfsolar_eps_wrf471_oracle/build.sh /wrf OUT_DIR
#
# Writes OUT_DIR/{inputs.bin, eps.out.bin, receipt.txt} and the cut sources.
# tests/test_wrfsolar_eps_wrf471_oracle.py grades the port against it.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "usage: build.sh WRF_SOURCE_ROOT OUT_DIR" >&2; exit 2; }
root=$(realpath "$1")
mkdir -p "$2"
out=$(realpath "$2")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python=${PYTHON:-python3}
nm=${NM:-$(command -v nm || command -v x86_64-conda-linux-gnu-nm)}
[[ -n "$nm" ]] || { echo "no nm on PATH" >&2; exit 2; }

pinned=f52c197ed39d12e087d02c50f412d90d418f6186
for f in phys/module_radiation_driver.F phys/module_shallowcu_driver.F \
         phys/module_microphysics_driver.F; do
    want=$(grep -F " $f" "$here/wrf471_sha256.txt" | cut -d' ' -f1)
    have=$(sha256sum "$root/$f" | cut -d' ' -f1)
    [[ "$want" == "$have" ]] || { echo "$f differs from WRF v4.7.1 ($pinned)" >&2; exit 3; }
done
[[ "$(uname -m)" == "x86_64" ]] || { echo "not x86-64; see the header" >&2; exit 4; }

cd "$out"
"$python" "$here/extract.py" "$root" .
"$python" "$here/make_inputs.py" .
flags="-O0 -ffp-contract=off -fno-fast-math -ffree-form -ffree-line-length-none"
for m in eps_rad_cut eps_shcu_cut eps_mp_cut eps_cld3_cut; do
    gfortran $flags -c $m.F90 -o $m.o
done
gfortran $flags -c "$here/run_eps.F90" -o run_eps.o
gfortran eps_rad_cut.o eps_shcu_cut.o eps_mp_cut.o eps_cld3_cut.o run_eps.o -o run_eps
for m in eps_rad_cut eps_shcu_cut eps_mp_cut eps_cld3_cut; do "$nm" -u $m.o; done > undefined.txt
if grep -qE '(powf|expf|logf|log10f|sqrtf|sinf|cosf|_ZGV)' undefined.txt; then echo "libm symbol in a hook" >&2; cat undefined.txt >&2; exit 5; fi
./run_eps
{
    echo "wrf v4.7.1 commit $pinned"
    gfortran --version | sed -n 1p
    echo "flags: $flags"
    echo "machine: $(uname -m)"
    echo "undefined symbols in the hooks: $(tr -s ' ' < undefined.txt | tr '\n' ' ')"
    sha256sum "$root/phys/module_radiation_driver.F" "$root/phys/module_shallowcu_driver.F" \
        "$root/phys/module_microphysics_driver.F" \
        eps_rad_cut.F90 eps_shcu_cut.F90 eps_mp_cut.F90 eps_cld3_cut.F90 \
        "$here/extract.py" "$here/run_eps.F90" "$here/make_inputs.py" "$here/build.sh" \
        inputs.bin eps.out.bin | sed "s#$root/##; s#$here/#tools/wrfsolar_eps_wrf471_oracle/#"
} > receipt.txt
cat receipt.txt
