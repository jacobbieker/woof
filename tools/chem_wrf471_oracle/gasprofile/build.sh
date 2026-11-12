#!/usr/bin/env bash
# Build and run the WRF-Chem background gas profile oracle (chem_opt = 1/2,
# chem_in_opt = 0) against WRF v4.7.1.
#
#   bash tools/chem_wrf471_oracle/gasprofile/build.sh WRF_SRC BUILD_DIR [FIXTURE_ROOT]
#
# share/module_model_constants.F, chem/module_data_radm2.F,
# chem/module_data_sorgam.F and share/module_chem_share.F compile whole and
# unchanged.  chem/module_input_chem_data.F is rebuilt as module
# module_input_chem_data from the file's own lines 1 to CONTAINS (the FSL
# notice, `#define CASENAME 4`, the USE lines and every module DATA statement)
# and the routines in routines.list, each copied BYTE FOR BYTE from its header
# line to its END line.  Nothing is edited.
#
# stub_gasprofile.F90 supplies only the Registry-generated
# module_state_description (package values; radm2 / radm2sorg species indices),
# a module_domain with the fields input_chem_profile reads, and the names the
# other USE lines import.  -O0, and the build fails if any -O0 object imports a
# libmvec _ZGV* symbol (positive control included).
set -euo pipefail
if [[ $# -lt 2 ]]; then
    echo "usage: build.sh WRF_SRC BUILD_DIR [FIXTURE_ROOT]" >&2
    exit 2
fi
src=$(realpath "$1")
build=$(realpath -m "$2")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fixtures=$(realpath -m "${3:-${build}/fixtures}")

while read -r sum rel; do
    [[ -z "${sum}" || "${sum}" == \#* ]] && continue
    got=$(sha256sum "${src}/${rel}" | cut -d' ' -f1)
    if [[ "${got}" != "${sum}" ]]; then
        echo "${rel}: sha256 ${got}, pinned ${sum}" >&2
        exit 3
    fi
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
${fc} "${here}/stub_gasprofile.F90"; guard stub_gasprofile.o
${fc} "${here}/oracle_io.F90"; guard oracle_io.o
objs=(stub_gasprofile.o oracle_io.o)
for rel in share/module_model_constants.F chem/module_data_radm2.F chem/module_data_sorgam.F \
           share/module_chem_share.F; do
    obj=$(basename "${rel}" .F).o
    ${fc} -o "${obj}" "${src}/${rel}"
    guard "${obj}"
    objs+=("${obj}")
done

extract() {  # print routine $2 of file $1 from its header line to its END line
    awk -v want="$2" '
        function norm(s) { s = tolower(s); gsub(/\r/, "", s); gsub(/^[ \t]+/, "", s); return s }
        {
            l = norm($0)
            if (!inside && l ~ /^((real|integer)[^ \t]*[ \t]+)?(subroutine|function)[ \t]+/) {
                t = l; sub(/^.*(subroutine|function)[ \t]+/, "", t); sub(/[ \t(].*$/, "", t)
                if (t == want) inside = 1
            }
            if (inside) {
                print $0
                if (l ~ /^end[ \t]*(subroutine|function)/) { found = 1; exit }
            }
        }
        END { if (!found) exit 7 }' "$1"
}
mod="${src}/chem/module_input_chem_data.F"
out=module_input_chem_data_extract.F
{
    # The file's own lines 1 .. CONTAINS, unchanged.
    awk '{ print } /^CONTAINS[ \t\r]*$/ { found = 1; exit } END { if (!found) exit 7 }' "${mod}"
} > "${out}"
while read -r name; do
    name="${name%%#*}"; name="$(echo "${name}" | xargs)"
    [[ -z "${name}" ]] && continue
    extract "${mod}" "${name}" >> "${out}" || {
        echo "routine ${name} not found" >&2; exit 7; }
done < "${here}/routines.list"
echo "END MODULE module_input_chem_data" >> "${out}"
${fc} -o module_input_chem_data.o "${out}"; guard module_input_chem_data.o
objs+=(module_input_chem_data.o)

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
    (cd "${fixtures}/${name#run_}" && find . \( -name '*.bin' -o -name MANIFEST.txt \) \
        | LC_ALL=C sort | xargs sha256sum > oracle-sha256sums.txt)
done

{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version 2>&1 | head -1 || true)"
    echo "# -O0 libm symbols the reference calls:"
    grep -E 'expf|powf|logf|log10f|sqrtf|exp$|log$|pow$|__powi|_ZGV' undefined-O0.txt | sort -u || true
} > libm-report.txt
cat "${here}/SOURCES.sha256" > oracle-source-sha256sums.txt
(cd "${here}" && sha256sum ./*.F90 ./*.list ./build.sh | sed 's#  \./#  tools/chem_wrf471_oracle/gasprofile/#') >> oracle-source-sha256sums.txt
echo "gasprofile oracle built in ${build}; fixtures in ${fixtures}"
