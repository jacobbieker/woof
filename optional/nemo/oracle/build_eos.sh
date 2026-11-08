#!/usr/bin/env bash
# Build and run the NEMO 5.0.2 equation-of-state oracle: src/OCE/TRA/eosbn2.F90
# compiled WHOLE, unchanged, against eos_stubs.F90 (NEMO's module names
# carrying only what eosbn2 USEs), with NEMO's own macro headers.  Part of the
# NEMO opt-in (CeCILL v2.0, see ../LICENSE): NEMO's source is read at build
# time and none of it is copied into the tree.  Also regenerates
# woof_nemo/eos_coeffs.py from the pinned file and refuses if the committed
# copy differs.  -O0, key_linssh + key_vco_3d, no libmvec.
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "usage: build_eos.sh NEMO_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
pinned_commit="dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796"   # tag 5.0.2
sources=(src/OCE/par_kind.F90 src/OCE/TRA/eosbn2.F90 src/OCE/do_loop_substitute.h90
         src/OCE/DOM/domzgr_substitute.h90 src/OCE/read_nml_substitute.h90)
if [[ "$(git -C "${source_root}" rev-parse HEAD)" != "${pinned_commit}" ]]; then
    echo "NEMO tree is not at the pinned ${pinned_commit}" >&2
    exit 3
fi
for rel in "${sources[@]}"; do test -f "${source_root}/${rel}"; done
if ! git -C "${source_root}" diff --quiet HEAD -- "${sources[@]}"; then
    echo "one of the NEMO sources differs from the pinned commit" >&2
    exit 3
fi
mkdir -p "${build_dir}"
cd "${build_dir}"
F="-O0 -cpp -ffree-form -ffree-line-length-none"
gfortran -c ${F} "${source_root}/src/OCE/par_kind.F90"
gfortran -c ${F} "${script_dir}/eos_stubs.F90"
gfortran -c ${F} -Dkey_linssh -Dkey_vco_3d -I "${source_root}/src/OCE" -I "${source_root}/src/OCE/DOM" \
    "${source_root}/src/OCE/TRA/eosbn2.F90"
gfortran -c -O0 -ffree-form -ffree-line-length-none "${script_dir}/run_eos.F90"
gfortran -o run_eos par_kind.o eos_stubs.o eosbn2.o run_eos.o
nm -u eosbn2.o | sed 's/^ *//' | sort > undefined-O0-eosbn2.txt
if grep -q '_ZGV' undefined-O0-eosbn2.txt; then
    echo "-O0 eosbn2.o pulled in a libmvec symbol" >&2
    exit 4
fi
./run_eos . > run-eos-stdout.txt
python3 -I "${script_dir}/gen_eos_coeffs.py" "${source_root}" eos_coeffs.py > /dev/null
if ! cmp -s eos_coeffs.py "${script_dir}/../woof_nemo/eos_coeffs.py"; then
    echo "woof_nemo/eos_coeffs.py differs from what the pinned eosbn2.F90 generates" >&2
    exit 8
fi
sha256sum "${sources[@]/#/${source_root}/}" "${script_dir}/eos_stubs.F90" \
    "${script_dir}/run_eos.F90" "${script_dir}/build_eos.sh" "${script_dir}/gen_eos_coeffs.py" \
    eos_coeffs.py eos-grid.csv eos-teos10.csv eos-eos80.csv eos-seos.csv \
    | sed "s#${source_root}/#<nemo-5.0.2-checkout>/#; s#${script_dir}/#<oracle-workspace>/optional/nemo/oracle/#" \
    > eos-sha256sums.txt
echo "oracle built in ${build_dir}"
