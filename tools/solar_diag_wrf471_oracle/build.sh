#!/usr/bin/env bash
# Build and run the solar_diagnostics = 1 column oracle: WRF v4.7.1
# phys/module_diag_solar.F (and share/module_model_constants.F for G),
# compiled byte-unmodified, CPU only.
#
#   build.sh WRF_SOURCE_ROOT OUT_DIR
#
# Reference of record: gfortran -O0 -ffp-contract=off against glibc 2.43's
# libm (Ubuntu 26.04, 2.43-2ubuntu2.4), the C library of the repository's
# WRF v4.7.1 column oracles; solar_diag's only libm call is ACOS (acosf,
# CORE-MATH and correctly rounded from glibc 2.42; glibc 2.39's is FDLIBM's),
# so the script refuses any other libm.  On pb the record was made as
#
#   E=$(pixi exec -s gfortran sh -c 'echo $CONDA_PREFIX')
#   docker run --rm -u $(id -u):$(id -g) -v $E:$E:ro -v $WRF:/wrf:ro \
#       -v $PWD:$PWD -w $PWD -e PATH=$E/bin:/usr/bin:/bin \
#       -e PYTHON=$PWD/.pixi/envs/default/bin/python ubuntu:26.04 \
#       bash tools/solar_diag_wrf471_oracle/build.sh /wrf OUT_DIR
#
# Writes OUT_DIR/{columns.bin, columns_meta.json, solar_diag.out.bin,
# receipt.txt}.  tests/test_solar_diag_wrf471_oracle.py grades the port.
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
for f in phys/module_diag_solar.F share/module_model_constants.F; do
    want=$(grep -F " $f" "$here/wrf471_sha256.txt" | cut -d' ' -f1)
    have=$(sha256sum "$root/$f" | cut -d' ' -f1)
    [[ "$want" == "$have" ]] || { echo "$f differs from WRF v4.7.1 ($pinned)" >&2; exit 3; }
done
glibc=$(ldd --version | sed -n 1p)
case "$glibc" in
    *" 2.43"*) ;;
    *) echo "libm is not glibc 2.43 ($glibc); see the header" >&2; exit 4 ;;
esac

cd "$out"
"$python" "$here/make_columns.py" .
flags="-O0 -ffp-contract=off -fno-fast-math -cpp -DEM_CORE=1 -ffree-form -ffree-line-length-none"
gfortran $flags -c "$root/share/module_model_constants.F" -o module_model_constants.o
gfortran $flags -c "$root/phys/module_diag_solar.F" -o module_diag_solar.o
gfortran $flags -c "$here/run_solar_diag.F90" -o run_solar_diag.o
gfortran module_model_constants.o module_diag_solar.o run_solar_diag.o -o run_solar_diag
"$nm" -u module_diag_solar.o > undefined.txt
if grep -q _ZGV undefined.txt; then echo "libmvec symbol at -O0" >&2; exit 5; fi
./run_solar_diag
{
    echo "wrf v4.7.1 commit $pinned"
    gfortran --version | sed -n 1p
    echo "flags: $flags"
    echo "$glibc"
    echo "libm symbols (module_diag_solar.o): $(grep -E 'acosf|atanf|powf|expf|logf' undefined.txt | tr -s ' ' | tr '\n' ' ')"
    sha256sum "$root/phys/module_diag_solar.F" "$root/share/module_model_constants.F" \
        "$here/run_solar_diag.F90" "$here/make_columns.py" "$here/build.sh" \
        columns.bin columns_meta.json solar_diag.out.bin \
        | sed "s#$root/##; s#$here/#tools/solar_diag_wrf471_oracle/#"
} > receipt.txt
cat receipt.txt
