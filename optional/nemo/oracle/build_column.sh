#!/usr/bin/env bash
# Build and run the NEMO 5.0.2 tracer-column oracle: eosbn2.F90, traqsr.F90,
# zdfsh2.F90, zdfdrg.F90, zdftke.F90, trazdf.F90, trasbc.F90 and
# traatf_qco.F90 compiled WHOLE, unchanged, against column_stubs.F90, and
# sbccpl.F90's sbc_cpl_rcv cut byte for byte into column_harness.F90; then
# run_column.F90 steps them in stpmlf.F90's order.  Part of the NEMO opt-in
# (CeCILL v2.0, see ../LICENSE): NEMO's source is read at build time and none
# of it is copied into the tree.  -O0, key_linssh + key_vco_3d, no libmvec.
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "usage: build_column.sh NEMO_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
pinned_commit="dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796"   # tag 5.0.2
whole=(TRA/eosbn2 TRA/traqsr ZDF/zdfsh2 ZDF/zdfdrg ZDF/zdftke TRA/trazdf TRA/trasbc TRA/traatf_qco)
sources=(src/OCE/par_kind.F90 src/OCE/SBC/sbccpl.F90 src/OCE/do_loop_substitute.h90
         src/OCE/DOM/domzgr_substitute.h90 cfgs/SHARED/namelist_ref)
for f in "${whole[@]}"; do sources+=("src/OCE/${f}.F90"); done
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
awk '/^   SUBROUTINE sbc_cpl_rcv\(/,/^   END SUBROUTINE sbc_cpl_rcv/' \
    "${source_root}/src/OCE/SBC/sbccpl.F90" > sbc_cpl_rcv.inc
if [[ "$(wc -l < sbc_cpl_rcv.inc)" -ne 443 ]]; then
    echo "the sbc_cpl_rcv cut is not the pinned 443 lines" >&2
    exit 3
fi
F="-O0 -cpp -ffree-form -ffree-line-length-none"
N="-Dkey_linssh -Dkey_vco_3d -I ${source_root}/src/OCE -I ${source_root}/src/OCE/DOM"
gfortran -c ${F} "${source_root}/src/OCE/par_kind.F90"
gfortran -c ${F} "${script_dir}/column_stubs.F90"
objs=(par_kind.o column_stubs.o)
for f in "${whole[@]}"; do
    gfortran -c ${F} ${N} "${source_root}/src/OCE/${f}.F90"
    objs+=("$(basename "${f}").o")
done
gfortran -c ${F} ${N} -I . "${script_dir}/column_harness.F90"
gfortran -c -O0 -ffree-form -ffree-line-length-none "${script_dir}/run_column.F90"
gfortran -o run_column "${objs[@]}" column_harness.o run_column.o
for o in "${objs[@]}" column_harness.o; do nm -u "${o}"; done | sed 's/^ *//' | sort -u > undefined-O0-column.txt
if grep -q '_ZGV' undefined-O0-column.txt; then
    echo "-O0 objects pulled in a libmvec symbol" >&2
    exit 4
fi
for nml in nameos namtra_qsr namzdf_tke namdrg namdrg_bot; do
    awk "/^&${nml} /,/^\\//" "${source_root}/cfgs/SHARED/namelist_ref"
done > column_namelist_ref.txt
# a case, not an associative array: the script also runs under bash 3.2 (macOS)
cfg_of() {
    case "$1" in
        teos10) echo '&nameos ln_teos10=.true. / &namtra_qsr ln_qsr_rgb=.false., ln_qsr_2bd=.true. / &namzdf_tke / &namdrg ln_non_lin=.true. / &namdrg_bot /' ;;
        eos80)  echo '&nameos ln_eos80=.true. / &namtra_qsr / &namzdf_tke nn_mxl=2, nn_etau=0 / &namdrg ln_loglayer=.true. / &namdrg_bot /' ;;
    esac
}
qsr_of() { case "$1" in teos10) echo T ;; eos80) echo F ;; esac; }
outs=()
for name in teos10 eos80; do
    mkdir -p "${name}"
    ./run_column "${name}" column_namelist_ref.txt "$(cfg_of "${name}")" "$(qsr_of "${name}")" > "${name}/stdout.txt"
    for part in grid forcing steps surface; do
        mv "${name}/column-${part}.csv" "column-${name}-${part}.csv"
        outs+=("column-${name}-${part}.csv")
    done
done
sha256sum "${sources[@]/#/${source_root}/}" "${script_dir}/column_stubs.F90" \
    "${script_dir}/column_harness.F90" "${script_dir}/run_column.F90" "${script_dir}/build_column.sh" \
    "${outs[@]}" \
    | sed "s#${source_root}/#<nemo-5.0.2-checkout>/#; s#${script_dir}/#<oracle-workspace>/optional/nemo/oracle/#" \
    > column-sha256sums.txt
echo "oracle built in ${build_dir}"
