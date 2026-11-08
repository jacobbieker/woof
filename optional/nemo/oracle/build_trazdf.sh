#!/usr/bin/env bash
# Build and run the NEMO 5.0.2 tra_zdf oracle: src/OCE/TRA/trazdf.F90 compiled
# WHOLE, unchanged, against trazdf_stubs.F90, with NEMO's macro headers.  Part
# of the NEMO opt-in (CeCILL v2.0, see ../LICENSE): NEMO's source is read at
# build time and none of it is copied into the tree.  -O0, key_linssh +
# key_vco_3d, no libmvec.
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "usage: build_trazdf.sh NEMO_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
pinned_commit="dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796"   # tag 5.0.2
sources=(src/OCE/par_kind.F90 src/OCE/TRA/trazdf.F90 src/OCE/do_loop_substitute.h90
         src/OCE/DOM/domzgr_substitute.h90)
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
gfortran -c ${F} "${script_dir}/trazdf_stubs.F90"
gfortran -c ${F} -Dkey_linssh -Dkey_vco_3d -I "${source_root}/src/OCE" -I "${source_root}/src/OCE/DOM" \
    "${source_root}/src/OCE/TRA/trazdf.F90"
gfortran -c -O0 -ffree-form -ffree-line-length-none "${script_dir}/run_trazdf.F90"
gfortran -o run_trazdf par_kind.o trazdf_stubs.o trazdf.o run_trazdf.o
./run_trazdf . > run-trazdf-stdout.txt
sha256sum "${sources[@]/#/${source_root}/}" "${script_dir}/trazdf_stubs.F90" \
    "${script_dir}/run_trazdf.F90" "${script_dir}/build_trazdf.sh" trazdf-grid.csv trazdf-steps.csv \
    | sed "s#${source_root}/#<nemo-5.0.2-checkout>/#; s#${script_dir}/#<oracle-workspace>/optional/nemo/oracle/#" \
    > trazdf-sha256sums.txt
echo "oracle built in ${build_dir}"
