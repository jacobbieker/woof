#!/usr/bin/env bash
# Build and run the prec_acc_dt / acc_phy_tend oracle from byte-unmodified WRF v4.7.1 (CPU only).
#
#   build.sh WRF_SOURCE_ROOT OUT_DIR     (WRF_SOURCE_ROOT: a WRF v4.7.1 tree, or tools/diag_wrf471_oracle/upstream)
#
# Cuts WRF's own accumulation blocks out of phys/module_diag_misc.F (sha256 in wrf471_sha256.txt; lines 223-286 are
# the acc_phy_tend block and 432-462 the prec_acc_dt block of diagnostic_output_calc), compiles them into run_acc.F90
# with gfortran -O0 -ffp-contract=off, runs them on the inputs make_inputs.py writes, and leaves
# OUT_DIR/{inputs.bin,oracle.bin,receipt.txt}.  The committed copies live in components/hex/tests/acc_oracle;
# components/hex/tests/test_mpas_acc_diagnostics.py grades the port against them word for word (the arithmetic is
# one float32 product and sum per word, so it is exact on every host).
set -euo pipefail
[[ $# -eq 2 ]] || { echo "usage: build.sh WRF_SOURCE_ROOT OUT_DIR" >&2; exit 2; }
root=$(cd "$1" && pwd)
mkdir -p "$2"
out=$(cd "$2" && pwd)
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python=${PYTHON:-python3}
f=phys/module_diag_misc.F
# a WRF tree, or the repository's byte-copy (tools/diag_wrf471_oracle/upstream) which holds the file flat
[[ -f "$root/$f" ]] || f=module_diag_misc.F
want=$(grep -F "module_diag_misc.F" "$here/wrf471_sha256.txt" | cut -d' ' -f1)
have=$( (sha256sum "$root/$f" 2>/dev/null || shasum -a 256 "$root/$f") | cut -d' ' -f1)
[[ "$want" == "$have" ]] || { echo "$f differs from WRF v4.7.1" >&2; exit 3; }
cd "$out"
sed -n '223,286p' "$root/$f" > block_acc_phy_tend.inc
sed -n '432,462p' "$root/$f" > block_prec_acc_dt.inc
head -1 block_acc_phy_tend.inc | grep -q 'IF ( acc_phy_tend .gt. 0 ) THEN' || { echo "block lines moved" >&2; exit 4; }
head -1 block_prec_acc_dt.inc | grep -q 'IF (prec_acc_dt .gt. 0.) THEN' || { echo "block lines moved" >&2; exit 4; }
"$python" "$here/make_inputs.py" .
gfortran -O0 -ffp-contract=off -fno-fast-math -ffree-form -ffree-line-length-none -cpp -I. -o run_acc "$here/run_acc.F90"
./run_acc inputs.bin oracle.bin
{ echo "gfortran: $(gfortran --version | head -1)"; echo "host: $(uname -sm)"; } > receipt.txt
