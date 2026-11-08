#!/usr/bin/env bash
# Build and run the NEMO 5.0.2 traqsr (2-band penetrative solar) oracle.
#
# Part of the NEMO opt-in (optional/nemo, CeCILL v2.0, see ../LICENSE): it
# reads NEMO's source at build time and copies none of it into the tree.
# tra_qsr, qsr_2BD and qsr_ext_lev are cut byte for byte from the pinned
# src/OCE/TRA/traqsr.F90 and #included in traqsr_harness.F90 after NEMO's own
# do_loop_substitute.h90 and domzgr_substitute.h90; par_kind.F90 is compiled
# from the pinned tree.  -O0, key_linssh + key_vco_3d, no key_RK3 (the
# leapfrog MLF path), and a refusal on any libmvec symbol.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build.sh NEMO_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
pinned_commit="dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796"   # tag 5.0.2

sources=(src/OCE/par_kind.F90 src/OCE/TRA/traqsr.F90
         src/OCE/do_loop_substitute.h90 src/OCE/DOM/domzgr_substitute.h90)
head_commit=$(git -C "${source_root}" rev-parse HEAD)
if [[ "${head_commit}" != "${pinned_commit}" ]]; then
    echo "NEMO tree is at ${head_commit}, not the pinned ${pinned_commit}" >&2
    exit 3
fi
for rel in "${sources[@]}"; do test -f "${source_root}/${rel}"; done
if ! git -C "${source_root}" diff --quiet HEAD -- "${sources[@]}"; then
    echo "one of the NEMO sources differs from the pinned commit" >&2
    exit 3
fi

mkdir -p "${build_dir}"
cd "${build_dir}"
cut_routine() {   # $1 = kind, $2 = name (case-insensitive end)
    awk -v kind="$1" -v name="$2" '
        { low = tolower($0) }
        !on && low ~ "^ *" tolower(kind) " +" tolower(name) "\\(" { on = 1 }
        on { print }
        on && low ~ "^ *end +" tolower(kind) " +" tolower(name) " *$" { exit }
    ' "${source_root}/src/OCE/TRA/traqsr.F90"
}
cut_routine SUBROUTINE tra_qsr > tra_qsr.inc
cut_routine SUBROUTINE qsr_2BD > qsr_2bd.inc
cut_routine FUNCTION qsr_ext_lev > qsr_ext_lev.inc
lines="$(grep -c '' tra_qsr.inc) $(grep -c '' qsr_2bd.inc) $(grep -c '' qsr_ext_lev.inc)"
if [[ "${lines}" != "148 86 67" ]]; then
    echo "cut lengths ${lines}, expected 148 86 67" >&2
    exit 6
fi

free="-ffree-form -ffree-line-length-none"
defines="-Dkey_linssh -Dkey_vco_3d"
gfortran -c -O0 -cpp ${free} "${source_root}/src/OCE/par_kind.F90"
gfortran -c -O0 -cpp ${free} ${defines} -I "${build_dir}" \
    -I "${source_root}/src/OCE" -I "${source_root}/src/OCE/DOM" \
    "${script_dir}/traqsr_harness.F90"
gfortran -c -O0 ${free} -I "${build_dir}" "${script_dir}/run_traqsr.F90"
gfortran -o run_traqsr par_kind.o traqsr_harness.o run_traqsr.o

nm -u traqsr_harness.o | sed 's/^ *//' | sort > undefined-O0-harness.txt
if grep -q '_ZGV' undefined-O0-harness.txt; then
    echo "-O0 harness pulled in a libmvec symbol" >&2
    exit 4
fi
{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# -O0 traqsr_harness.o math symbols:"
    grep -E 'exp|log|pow|_ZGV' undefined-O0-harness.txt || true
} > libm-report.txt
cat libm-report.txt

./run_traqsr . > run-stdout.txt

sha256sum "${sources[@]/#/${source_root}/}" tra_qsr.inc qsr_2bd.inc qsr_ext_lev.inc \
    "${script_dir}/traqsr_harness.F90" "${script_dir}/run_traqsr.F90" \
    "${script_dir}/build.sh" traqsr-grid.csv traqsr-steps.csv \
    | sed "s#${source_root}/#<nemo-5.0.2-checkout>/#; s#${script_dir}/#<oracle-workspace>/optional/nemo/oracle/#" \
    > oracle-sha256sums.txt
echo "oracle built in ${build_dir}"
