#!/usr/bin/env bash
# Build and run the MAD-WRF oracle from byte-unmodified WRF v4.7.1, CPU only.
#
#   build.sh WRF_SOURCE_ROOT OUT_DIR
#
# Init_madwrf_clouds calls powf (``**`` with a real exponent) and the
# reference of record is gfortran -O0 -ffp-contract=off against glibc 2.43's
# libm on x86-64 (Ubuntu 26.04), the C library of the repository's other WRF
# v4.7.1 column oracles; the script refuses any other.  On pb:
#
#   E=$(pixi exec -s gfortran sh -c 'echo $CONDA_PREFIX')
#   docker run --rm -u $(id -u):$(id -g) -v $E:$E:ro -v $WRF:/wrf:ro \
#       -v $PWD:$PWD -w $PWD -e PATH=$E/bin:/usr/bin:/bin \
#       -e PYTHON=$PWD/.pixi/envs/default/bin/python ubuntu:26.04 \
#       bash tools/madwrf_wrf471_oracle/build.sh /wrf OUT_DIR
#
# Writes OUT_DIR/{madwrf_columns.bin, madwrf.out.bin, receipt.txt}.
# tests/test_madwrf_wrf471_oracle.py grades the port against it.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "usage: build.sh WRF_SOURCE_ROOT OUT_DIR" >&2; exit 2; }
root=$(realpath "$1")
mkdir -p "$2"
out=$(realpath "$2")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python=${PYTHON:-python3}
nm=${NM:-$(command -v nm || command -v x86_64-conda-linux-gnu-nm)}

pinned=f52c197ed39d12e087d02c50f412d90d418f6186
while read -r want f; do
    have=$(sha256sum "$root/$f" | cut -d' ' -f1)
    [[ "$want" == "$have" ]] || { echo "$f differs from WRF v4.7.1 ($pinned)" >&2; exit 3; }
done < "$here/wrf471_sha256.txt"
[[ "$(uname -m)" == "x86_64" ]] || { echo "not x86-64" >&2; exit 4; }
glibc=$(ldd --version | sed -n 1p)
case "$glibc" in
    *" 2.43"*) ;;
    *) echo "libm is not glibc 2.43 ($glibc)" >&2; exit 4 ;;
esac

cd "$out"
"$python" "$here/extract.py" "$root" .
"$python" "$here/make_columns.py" .
flags="-O0 -ffp-contract=off -fno-fast-math -ffree-form -ffree-line-length-none"
gfortran $flags -c "$here/stubs.F90" -o stubs.o
gfortran $flags -cpp -c module_model_constants.F90 -o module_model_constants.o
gfortran $flags -c module_soil_pre_cut.F90 -o module_soil_pre_cut.o
gfortran $flags -c module_mp_thompson_cut.F90 -o module_mp_thompson_cut.o
gfortran $flags -cpp -c module_madwrf.F90 -o module_madwrf.o
gfortran $flags -c nudge_cut.F90 -o nudge_cut.o
gfortran $flags -c "$here/run_madwrf.F90" -o run_madwrf.o
gfortran stubs.o module_model_constants.o module_soil_pre_cut.o module_mp_thompson_cut.o \
    module_madwrf.o nudge_cut.o run_madwrf.o -o run_madwrf
"$nm" -u module_madwrf.o > undefined.txt
if grep -q _ZGV undefined.txt; then echo "libmvec symbol at -O0" >&2; exit 5; fi
./run_madwrf
{
    echo "wrf v4.7.1 commit $pinned"
    gfortran --version | sed -n 1p
    echo "flags: $flags"
    echo "machine: $(uname -m)"
    echo "$glibc"
    echo "libm symbols (module_madwrf.o): $(grep -E 'powf|sqrtf|expf|logf' undefined.txt | tr -s ' ' | tr '\n' ' ')"
    while read -r _ f; do sha256sum "$root/$f" | sed "s#$root/##"; done < "$here/wrf471_sha256.txt"
    sha256sum "$here/extract.py" "$here/stubs.F90" "$here/run_madwrf.F90" \
        "$here/make_columns.py" "$here/build.sh" \
        "$here/../cldfra3_wrf471_oracle/make_columns.py" \
        madwrf_columns.bin madwrf.out.bin | sed "s#$here/#tools/madwrf_wrf471_oracle/#"
} > receipt.txt
cat receipt.txt
