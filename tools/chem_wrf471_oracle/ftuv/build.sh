#!/usr/bin/env bash
# Build and run the Fast-TUV photolysis (phot_opt = 3) oracle against the
# byte-unmodified WRF v4.7.1 sources.
#
#   bash tools/chem_wrf471_oracle/ftuv/build.sh WRF_SRC BUILD_DIR [FIXTURE_ROOT]
#
# WRF_SRC is a WRF v4.7.1 tree (or any folder holding the files in
# SOURCES.sha256 at their WRF paths).  Every compiled WRF file is checked
# against its sha256 first; a mismatch stops the build.  The six files are
# compiled WHOLE and unchanged; stub_ftuv.F90 supplies only the
# Registry-generated module_state_description / module_configure and the
# wrf_debug / wrf_message / wrf_abort entry points.
#
# NETCDF is left undefined.  In module_ftuv_driver.F it guards only the
# MOZART branches (ftuv_init's column-density read, ftuv_timestep_init,
# p_interp), none of which chem_opt = 1 or 2 reaches: photo_inti, aer_init
# and ftuv_driver compile identically either way.
#
# -O0, no vectoriser, and the build fails if any -O0 object imports a libmvec
# _ZGV* symbol (a positive control proves the check can fire).
set -euo pipefail
if [[ $# -lt 2 ]]; then
    echo "usage: build.sh WRF_SRC BUILD_DIR [FIXTURE_ROOT]" >&2
    exit 2
fi
src=$(realpath "$1")
build=$(realpath -m "$2")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fixtures=$(realpath -m "${3:-${build}/fixtures}")

rels=()
while read -r sum rel; do
    [[ -z "${sum}" || "${sum}" == \#* ]] && continue
    got=$(sha256sum "${src}/${rel}" | cut -d' ' -f1)
    if [[ "${got}" != "${sum}" ]]; then
        echo "${rel}: sha256 ${got}, pinned ${sum}" >&2
        exit 3
    fi
    rels+=("${rel}")
done < "${here}/SOURCES.sha256"

mkdir -p "${build}" "${fixtures}"
cd "${build}"
defines="-DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4 -DWRF_CHEM=1"
free="-ffree-form -ffree-line-length-none -fallow-argument-mismatch"
fc="gfortran -c -O0 -cpp ${defines} ${free} -I ${build}"

: > undefined-O0.txt
guard() {
    nm -u "$1" | sed 's/^ *//' | sort > "undefined-$1.txt"
    cat "undefined-$1.txt" >> undefined-O0.txt
    if grep -q '_ZGV' "undefined-$1.txt"; then
        echo "-O0 object $1 pulled in a libmvec vector symbol" >&2
        exit 4
    fi
}
${fc} "${here}/stub_ftuv.F90"; guard stub_ftuv.o
${fc} "${here}/oracle_io.F90"; guard oracle_io.o
objs=(stub_ftuv.o oracle_io.o)
for rel in "${rels[@]}"; do
    obj=$(basename "${rel}" .F).o
    ${fc} -o "${obj}" "${src}/${rel}"
    guard "${obj}"
    objs+=("${obj}")
done

printf 'subroutine control(x, n)\n integer :: n, i\n real :: x(n)\n do i = 1, n\n  x(i) = exp(x(i))\n end do\nend subroutine\n' > control.f90
gfortran -c -Ofast -ftree-vectorize -o control.o control.f90
nm -u control.o | grep -q '_ZGV' || {
    echo "libmvec positive control produced no _ZGV symbol; the guard is vacuous" >&2; exit 5; }

for driver in "${here}"/run_*.F90; do
    name=$(basename "${driver}" .F90)
    ${fc} -o "${name}.o" "${driver}"; guard "${name}.o"
    gfortran -o "${name}" "${objs[@]}" "${name}.o"
    rm -rf "${fixtures:?}/${name#run_}"
    mkdir -p "${fixtures}/${name#run_}"
    "./${name}" "${fixtures}/${name#run_}" > "${name}-stdout.txt"
done

{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version 2>&1 | head -1 || true)"
    echo "# -O0 libm symbols the reference calls:"
    grep -E 'exp|pow|log|sin|cos|tan|atan|acos|_ZGV' undefined-O0.txt | sort -u || true
} > libm-report.txt
# The build's inputs (WRF pins, harness sources) stay in the build directory;
# the fixture directory gets the checksums of its own files, the list the
# parity test verifies (tests/data/oracles/chem/ftuv/oracle-sha256sums.txt).
cat "${here}/SOURCES.sha256" > oracle-source-sha256sums.txt
(cd "${here}" && sha256sum ./*.F90 ./build.sh | sed 's#  \./#  tools/chem_wrf471_oracle/ftuv/#') >> oracle-source-sha256sums.txt
for driver in "${here}"/run_*.F90; do
    name=$(basename "${driver}" .F90)
    (cd "${fixtures}/${name#run_}" && find . \( -name '*.bin' -o -name MANIFEST.txt \) \
        | LC_ALL=C sort | xargs sha256sum > oracle-sha256sums.txt)
done
echo "ftuv oracle built in ${build}; fixtures in ${fixtures}"
