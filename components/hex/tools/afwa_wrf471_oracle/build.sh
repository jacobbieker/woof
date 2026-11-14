#!/usr/bin/env bash
# Build and run the AFWA column oracle from byte-unmodified WRF v4.7.1 (CPU only).
#
#   build.sh WRF_SOURCE_ROOT OUT_DIR     (WRF_SOURCE_ROOT: a WRF v4.7.1 tree, or tools/diag_wrf471_oracle/upstream)
#
# Compiles phys/module_diag_functions.F as it is, and the part of phys/module_diag_afwa.F after
# afwa_diagnostics_driver (severe_wx, vil, icing, radar, dbzcalc, precip_type, vis, cloud, thermal, turbulence
# diagnostics: the lines between "END SUBROUTINE afwa_diagnostics_driver" and "END MODULE module_diag_afwa", cut by
# marker) wrapped in a module of its own, sha256 of both files in wrf471_sha256.txt.  The driver itself needs the
# WRF domain, configuration and clock modules, so run_afwa.F90 writes out its glue.  Runs on the columns make_columns.py
# writes and leaves OUT_DIR/{columns.bin,oracle.bin,receipt.txt}.  The committed copies live in
# components/hex/tests/afwa_oracle; components/hex/tests/test_mpas_afwa_diagnostics.py grades the hex module against
# them.  gfortran -O0 -ffp-contract=off; the libm (exp, log10, powf) is the build machine's, which the receipt records,
# so the test grades with a tolerance rather than bitwise.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "usage: build.sh WRF_SOURCE_ROOT OUT_DIR" >&2; exit 2; }
root=$(cd "$1" && pwd)
mkdir -p "$2"
out=$(cd "$2" && pwd)
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python=${PYTHON:-python3}
src() { local f=phys/$1; [[ -f "$root/$f" ]] || f=$1; echo "$root/$f"; }
for name in module_diag_afwa.F module_diag_functions.F; do
    want=$(grep -F "phys/$name" "$here/wrf471_sha256.txt" | cut -d' ' -f1)
    have=$( (sha256sum "$(src $name)" 2>/dev/null || shasum -a 256 "$(src $name)") | cut -d' ' -f1)
    [[ "$want" == "$have" ]] || { echo "$name differs from WRF v4.7.1" >&2; exit 3; }
done
cd "$out"
"$python" "$here/make_columns.py" .
afwa=$(src module_diag_afwa.F)
first=$(grep -n 'END SUBROUTINE afwa_diagnostics_driver' "$afwa" | cut -d: -f1)
last=$(grep -n '^END MODULE module_diag_afwa' "$afwa" | tail -1 | cut -d: -f1)
{ printf 'MODULE module_diag_afwa_cut\n    USE diag_functions\nCONTAINS\n'; sed -n "$((first+1)),$((last-1))p" "$afwa"; printf 'END MODULE module_diag_afwa_cut\n'; } > afwa_cut.F90
flags="-O0 -ffp-contract=off -fno-fast-math -ffree-form -ffree-line-length-none -cpp -DNMM_CORE=0 -DEM_CORE=1"
gfortran $flags -c "$(src module_diag_functions.F)"
gfortran $flags -c afwa_cut.F90
gfortran $flags -c "$here/run_afwa.F90"
gfortran -o run_afwa run_afwa.o afwa_cut.o module_diag_functions.o
./run_afwa columns.bin oracle.bin
{ echo "gfortran: $(gfortran --version | head -1)"; echo "host: $(uname -sm)"; echo "libm: ${LIBM_NOTE:-system}"; } > receipt.txt
