#!/usr/bin/env bash
# Build and run the km_opt = 5 (SMS-3DTKE) tile oracle from byte-unmodified
# WRF v4.7.1, CPU only.
#
#   build.sh WRF_SOURCE_ROOT OUT_DIR
#
# Reference of record: gfortran -O0 -ffp-contract=off against glibc 2.43's
# libm (Ubuntu 26.04), the C library of the repository's WRF v4.7.1 tile and
# column oracles.  The routines call powf, expf, logf and tanhf; glibc 2.41+
# tanhf is CORE-MATH's correctly rounded one and older glibc's is not, and
# macOS's libm moves every family, so the script refuses any other libm.  On
# pb the record was made as
#
#   E=$(pixi exec -s gfortran sh -c 'echo $CONDA_PREFIX')
#   docker run --rm -u $(id -u):$(id -g) -v $E:$E:ro -v $WRF:/wrf:ro \
#       -v $PWD:$PWD -w $PWD -e PATH=$E/bin:/usr/bin:/bin \
#       -e PYTHON=$PWD/.pixi/envs/default/bin/python ubuntu:26.04 \
#       bash tools/sms3dtke_wrf471_oracle/build.sh /wrf OUT_DIR
#
# Writes OUT_DIR/{cases.bin, cases.json, oracle.bin, receipt.txt} and the
# cut source.  tests/test_sms3dtke_wrf471_parity.py grades the port against
# it.  SMS_ALLOW_ANY_LIBM=1 skips the libm check (a dry run elsewhere; never
# the record).
set -euo pipefail
[[ $# -eq 2 ]] || { echo "usage: build.sh WRF_SOURCE_ROOT OUT_DIR" >&2; exit 2; }
root=$(realpath "$1")
mkdir -p "$2"
out=$(realpath "$2")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python=${PYTHON:-python3}
pinned=f52c197ed39d12e087d02c50f412d90d418f6186

if [[ "${SMS_ALLOW_ANY_LIBM:-0}" != 1 ]]; then
    glibc=$(ldd --version | sed -n 1p)
    case "$glibc" in
        *" 2.43"*) ;;
        *) echo "libm is not glibc 2.43 ($glibc); see the header" >&2; exit 4 ;;
    esac
else
    glibc="unchecked libm (SMS_ALLOW_ANY_LIBM=1)"
fi

cd "$out"
"$python" "$here/extract.py" "$root" .
"$python" "$here/make_cases.py" .
flags="-O0 -ffp-contract=off -fno-fast-math -cpp -ffree-form -ffree-line-length-none -fcheck=bounds"
gfortran $flags -c "$here/config.F90" -o config.o
gfortran $flags -c "$root/share/module_model_constants.F" -o constants.o
gfortran $flags -c sms_cut.F90 -o sms_cut.o
gfortran $flags -c "$here/run_sms3dtke.F90" -o run_sms3dtke.o
gfortran config.o constants.o sms_cut.o run_sms3dtke.o -o run_sms3dtke
nm=${NM:-$(command -v nm || command -v x86_64-conda-linux-gnu-nm || true)}
if [[ -n "$nm" ]]; then
    "$nm" -u sms_cut.o > undefined.txt
    if grep -q _ZGV undefined.txt; then echo "libmvec symbol at -O0" >&2; exit 5; fi
fi
./run_sms3dtke
{
    echo "wrf v4.7.1 commit $pinned"
    gfortran --version | sed -n 1p
    echo "flags: $flags"
    echo "$glibc"
    [[ -f undefined.txt ]] && echo "libm symbols (sms_cut.o): $(grep -E 'powf|expf|logf|tanhf|powisf2' undefined.txt | tr -s ' ' | tr '\n' ' ')"
    sha256sum "$root/dyn_em/module_diffusion_em.F" "$root/dyn_em/module_big_step_utilities_em.F" \
        "$root/share/module_bc.F" "$root/share/module_model_constants.F" \
        sms_cut.F90 "$here/extract.py" "$here/config.F90" "$here/run_sms3dtke.F90" \
        "$here/make_cases.py" "$here/build.sh" cases.bin oracle.bin \
        | sed "s#$root/##; s#$here/#tools/sms3dtke_wrf471_oracle/#"
} > receipt.txt
cat receipt.txt
