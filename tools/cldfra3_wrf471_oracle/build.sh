#!/usr/bin/env bash
# Build and run the icloud = 3 (cal_cldfra3) column oracle from
# byte-unmodified WRF v4.7.1, CPU only.
#
#   build.sh WRF_SOURCE_ROOT OUT_DIR
#
# Reference of record: gfortran -O0 -ffp-contract=off against glibc 2.43's
# libm (Ubuntu 26.04, 2.43-2ubuntu2.4) -- the C library of the repository's
# WRF v4.7.1 column oracles (tools/glibc_trig_flt32_proof,
# tools/noah_mosaic_wrf471_oracle).  cal_cldfra3 calls log10f and powf:
# glibc 2.42+ log10f is CORE-MATH's correctly rounded one, glibc 2.39's is
# FDLIBM's (29,787,060 float32 arguments differ), and macOS libm moves the
# record too, so the script refuses any other libm.  On pb the record was
# made as
#
#   E=$(pixi exec -s gfortran sh -c 'echo $CONDA_PREFIX')
#   docker run --rm -u $(id -u):$(id -g) -v $E:$E:ro -v $WRF:/wrf:ro \
#       -v $PWD:$PWD -w $PWD -e PATH=$E/bin:/usr/bin:/bin \
#       -e PYTHON=$PWD/.pixi/envs/default/bin/python ubuntu:26.04 \
#       bash tools/cldfra3_wrf471_oracle/build.sh /wrf OUT_DIR
# (the woof checkout mounted at its own path, so its pixi python runs).
#
# Writes OUT_DIR/{columns.bin, cldfra3.out.bin, receipt.txt} and the cut
# sources.  tests/test_cldfra3_wrf471_oracle.py grades the port against it.
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
for f in phys/module_radiation_driver.F phys/module_mp_thompson.F; do
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
"$python" "$here/extract.py" "$root" .
"$python" "$here/make_columns.py" .
flags="-O0 -ffp-contract=off -fno-fast-math -ffree-form -ffree-line-length-none"
gfortran $flags -c "$here/stubs.F90" -o stubs.o
gfortran $flags -c thompson_cut.F90 -o thompson_cut.o
gfortran $flags -c cldfra3_cut.F90 -o cldfra3_cut.o
gfortran $flags -c "$here/run_cldfra3.F90" -o run_cldfra3.o
gfortran stubs.o thompson_cut.o cldfra3_cut.o run_cldfra3.o -o run_cldfra3
# -O0 must call the scalar libm: no vector symbols.
"$nm" -u cldfra3_cut.o > undefined.txt
if grep -q _ZGV undefined.txt; then echo "libmvec symbol at -O0" >&2; exit 5; fi
./run_cldfra3
{
    echo "wrf v4.7.1 commit $pinned"
    gfortran --version | sed -n 1p
    echo "flags: $flags"
    echo "$glibc"
    echo "libm symbols (cldfra3_cut.o): $(grep -E 'log10f|powf' undefined.txt | tr -s ' ' | tr '\n' ' ')"
    sha256sum "$root/phys/module_radiation_driver.F" "$root/phys/module_mp_thompson.F" \
        cldfra3_cut.F90 thompson_cut.F90 "$here/extract.py" "$here/stubs.F90" \
        "$here/run_cldfra3.F90" "$here/make_columns.py" "$here/build.sh" \
        columns.bin cldfra3.out.bin | sed "s#$root/##; s#$here/#tools/cldfra3_wrf471_oracle/#"
} > receipt.txt
cat receipt.txt
