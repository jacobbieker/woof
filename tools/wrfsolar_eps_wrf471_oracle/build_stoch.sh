#!/usr/bin/env bash
# Build and run the multi_perturb pattern-arithmetic oracle: WRF v4.7.1's
# module_stoch.F routines (extract_stoch.py) and WRF's own FFTPACK5
# (external/fftpack/fftpack5), byte-unmodified, CPU only.
#
#   build_stoch.sh WRF_SOURCE_ROOT OUT_DIR
#
# SETUP_RAND_PERTURB calls EXP and SQRT, the vertical structure COS and
# SIN, and FFTPACK5's twiddles COS/SIN/ATAN: the record is glibc 2.43's
# libm on x86-64 (Ubuntu 26.04), the C library of the repository's other
# WRF v4.7.1 oracles; the script refuses any other.  Made on pb as
#
#   E=$(pixi exec -s gfortran sh -c 'echo $CONDA_PREFIX')
#   docker run --rm -u $(id -u):$(id -g) -v $E:$E:ro -v $WRF:/wrf:ro \
#       -v $PWD:$PWD -w $PWD -e PATH=$E/bin:/usr/bin:/bin \
#       -e PYTHON=$PWD/.pixi/envs/default/bin/python ubuntu:26.04 \
#       bash tools/wrfsolar_eps_wrf471_oracle/build_stoch.sh /wrf OUT_DIR
#
# Writes OUT_DIR/{stoch_inputs.bin, stoch.out.bin, stoch_receipt.txt}.
# tests/test_wrfsolar_eps_pattern_oracle.py grades woof's generator.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "usage: build_stoch.sh WRF_SOURCE_ROOT OUT_DIR" >&2; exit 2; }
root=$(realpath "$1")
mkdir -p "$2"
out=$(realpath "$2")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python=${PYTHON:-python3}

pinned=f52c197ed39d12e087d02c50f412d90d418f6186
want=$(grep -F " dyn_em/module_stoch.F" "$here/wrf471_sha256.txt" | cut -d' ' -f1)
have=$(sha256sum "$root/dyn_em/module_stoch.F" | cut -d' ' -f1)
[[ "$want" == "$have" ]] || { echo "module_stoch.F differs from WRF v4.7.1" >&2; exit 3; }
[[ "$(uname -m)" == "x86_64" ]] || { echo "not x86-64" >&2; exit 4; }
glibc=$(ldd --version | sed -n 1p)
case "$glibc" in
    *" 2.43"*) ;;
    *) echo "libm is not glibc 2.43 ($glibc)" >&2; exit 4 ;;
esac

cd "$out"
"$python" "$here/extract_stoch.py" "$root" .
"$python" "$here/make_stoch_inputs.py" .
flags="-O0 -ffp-contract=off -fno-fast-math -ffree-form -ffree-line-length-none"
fft=(cfft1i cfft1b c1fm1b c1f2kb c1f3kb c1f4kb c1f5kb c1fgkb r4_mcfti1 r4_factor r4_tables xerfft)
objs=()
for f in "${fft[@]}"; do
    # WRF builds FFTPACK5 with -fallow-argument-mismatch (configure, gfortran >= 10)
    gfortran $flags -fallow-argument-mismatch -cpp -c "$root/external/fftpack/fftpack5/$f.F" -o "fft_$f.o"
    objs+=("fft_$f.o")
done
gfortran $flags -c "$here/stubs_stoch.F90" -o stubs_stoch.o
gfortran $flags -c stoch_cut.F90 -o stoch_cut.o
gfortran $flags -c "$here/run_stoch.F90" -o run_stoch.o
gfortran stubs_stoch.o stoch_cut.o run_stoch.o "${objs[@]}" -o run_stoch
./run_stoch > run_stoch.log
{
    echo "wrf v4.7.1 commit $pinned"
    gfortran --version | sed -n 1p
    echo "flags: $flags"
    echo "machine: $(uname -m)"
    echo "$glibc"
    sha256sum "$root/dyn_em/module_stoch.F" | sed "s#$root/##"
    for f in "${fft[@]}"; do sha256sum "$root/external/fftpack/fftpack5/$f.F" | sed "s#$root/##"; done
    sha256sum stoch_cut.F90 "$here/extract_stoch.py" "$here/run_stoch.F90" "$here/stubs_stoch.F90" \
        "$here/make_stoch_inputs.py" "$here/build_stoch.sh" \
        stoch_inputs.bin stoch.out.bin | sed "s#$here/#tools/wrfsolar_eps_wrf471_oracle/#"
} > stoch_receipt.txt
cat stoch_receipt.txt
