#!/usr/bin/env bash
# Build and run the p_lev / z_lev column oracle from byte-unmodified WRF v4.7.1 (CPU only).
#
#   build.sh WRF_SOURCE_ROOT OUT_DIR
#
# Compiles WRF's share/module_model_constants.F, phys/module_diag_pld.F and phys/module_diag_zld.F as they are
# (sha256 in wrf471_sha256.txt), runs them on the columns make_columns.py writes, and leaves
# OUT_DIR/{columns_extrap1.bin,columns_extrap2.bin,oracle_extrap1.bin,oracle_extrap2.bin,receipt.txt}.
# The committed copies live in components/hex/tests/plevel_oracle; components/hex/tests/test_mpas_plevel_diagnostics.py grades the
# port against them.  gfortran -O0 -ffp-contract=off; the libm (log, exp, powf) is the build machine's, which
# the receipt records, so the test grades with a few-ulp tolerance rather than bitwise.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "usage: build.sh WRF_SOURCE_ROOT OUT_DIR" >&2; exit 2; }
root=$(cd "$1" && pwd)
mkdir -p "$2"
out=$(cd "$2" && pwd)
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../../../.." && pwd)
python=${PYTHON:-python3}
for f in share/module_model_constants.F phys/module_diag_pld.F phys/module_diag_zld.F; do
    want=$(grep -F " $f" "$here/wrf471_sha256.txt" | cut -d' ' -f1)
    have=$( (sha256sum "$root/$f" 2>/dev/null || shasum -a 256 "$root/$f") | cut -d' ' -f1)
    [[ "$want" == "$have" ]] || { echo "$f differs from WRF v4.7.1" >&2; exit 3; }
done
cd "$out"
PYTHONPATH="$repo" "$python" "$here/make_columns.py" .
flags="-O0 -ffp-contract=off -fno-fast-math -ffree-form -ffree-line-length-none -cpp -DEM_CORE=1 -DNMM_CORE=0"
gfortran $flags -c "$root/share/module_model_constants.F"
gfortran $flags -c "$root/phys/module_diag_pld.F"
gfortran $flags -c "$root/phys/module_diag_zld.F"
gfortran $flags -c "$here/run_plevel.F90"
gfortran -o run_plevel run_plevel.o module_diag_pld.o module_diag_zld.o module_model_constants.o
for e in 1 2; do ./run_plevel columns_extrap$e.bin oracle_extrap$e.bin; done
{ echo "gfortran: $(gfortran --version | head -1)"; echo "host: $(uname -sm)"; echo "libm: ${LIBM_NOTE:-system}"; } > receipt.txt
