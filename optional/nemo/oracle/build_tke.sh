#!/usr/bin/env bash
# Build and run the NEMO 5.0.2 TKE oracle: src/OCE/ZDF/zdfsh2.F90 and
# zdftke.F90 compiled WHOLE, unchanged, against tke_stubs.F90, with NEMO's
# macro headers; &namzdf_tke is NEMO's own block from cfgs/SHARED/namelist_ref
# plus four configuration overrides.  Part of the NEMO opt-in (CeCILL v2.0,
# see ../LICENSE): NEMO's source is read at build time and none of it is
# copied into the tree.  -O0, key_linssh + key_vco_3d, no libmvec.
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "usage: build_tke.sh NEMO_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
pinned_commit="dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796"   # tag 5.0.2
sources=(src/OCE/par_kind.F90 src/OCE/ZDF/zdfsh2.F90 src/OCE/ZDF/zdftke.F90
         src/OCE/do_loop_substitute.h90 src/OCE/DOM/domzgr_substitute.h90
         cfgs/SHARED/namelist_ref)
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
gfortran -c ${F} "${script_dir}/tke_stubs.F90"
for f in zdfsh2 zdftke; do
    gfortran -c ${F} -Dkey_linssh -Dkey_vco_3d -I "${source_root}/src/OCE" \
        -I "${source_root}/src/OCE/DOM" "${source_root}/src/OCE/ZDF/${f}.F90"
done
gfortran -c -O0 -ffree-form -ffree-line-length-none "${script_dir}/run_tke.F90"
gfortran -o run_tke par_kind.o tke_stubs.o zdfsh2.o zdftke.o run_tke.o
nm -u zdftke.o | sed 's/^ *//' | sort > undefined-O0-zdftke.txt
if grep -q '_ZGV' undefined-O0-zdftke.txt; then
    echo "-O0 zdftke.o pulled in a libmvec symbol" >&2
    exit 4
fi
awk '/^&namzdf_tke/,/^\//' "${source_root}/cfgs/SHARED/namelist_ref" > namzdf_tke_ref.txt
declare -A cfgs=(
    [ref]='&namzdf_tke /'
    [mxl2]='&namzdf_tke nn_mxl=2, nn_pdl=0, ln_mxl0=.false., ln_lc=.false., nn_etau=0 /'
    [mxl0]='&namzdf_tke nn_mxl=0, nn_htau=0 /'
    [mxl1]='&namzdf_tke nn_mxl=1 /'
)
outs=()
for name in ref mxl2 mxl0 mxl1; do
    mkdir -p "${name}"
    ./run_tke "${name}" namzdf_tke_ref.txt "${cfgs[$name]}" > "${name}/stdout.txt"
    mv "${name}/tke-steps.csv" "tke-${name}.csv"
    outs+=("tke-${name}.csv")
done
cp ref/tke-grid.csv tke-grid.csv
sha256sum "${sources[@]/#/${source_root}/}" "${script_dir}/tke_stubs.F90" \
    "${script_dir}/run_tke.F90" "${script_dir}/build_tke.sh" tke-grid.csv "${outs[@]}" \
    | sed "s#${source_root}/#<nemo-5.0.2-checkout>/#; s#${script_dir}/#<oracle-workspace>/optional/nemo/oracle/#" \
    > tke-sha256sums.txt
echo "oracle built in ${build_dir}"
