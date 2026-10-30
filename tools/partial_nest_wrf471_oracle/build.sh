#!/usr/bin/env bash
# Build and run the partial-depth nest coupling-operator oracle against
# WRF v4.7.1.
#
#   bash tools/partial_nest_wrf471_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR
#
# WRF_SOURCE_ROOT holds the files SOURCES.sha256 pins (fetch_sources.sh lays
# them out); every one is checked before anything compiles.  extract.py cuts
# WRF's own statements (share/sint.F, interp_fcn.F's interp_fcn_sint,
# copy_fcn, copy_fcnm, copy_fcni, smoother/sm121/smdsm, and module_bc.F's
# relax_bdytend_core and spec_bdytend) byte for byte; stubs.F90 supplies
# the namelist getters and config record they USE; run_partial_nest.F90
# generates the inputs, calls the routines and writes the fixture with
# ../urban_wrf471_oracle/oracle_io.F90.
#
# Reference build: -O0 -ffp-contract=off, REAL is real(4) (no -r8).
# GFORTRAN defaults to `pixi exec -s gfortran gfortran`.  The libmvec guard
# (an -O0 object must not reference glibc's _ZGV* vector libm) is the
# vert_nest oracle's; on a host without glibc (macOS) there is no libmvec,
# the positive control cannot produce a _ZGV symbol, and the guard records
# that instead of failing.  Copy BUILD_DIR/fixtures to
# tests/data/oracles/partial_nest/ to publish it.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi
source_root=$(cd "$1" && pwd)
mkdir -p "$2"
build_dir=$(cd "$2" && pwd)
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
read -r -a fc <<< "${GFORTRAN:-pixi exec -s gfortran gfortran}"
sha() { if command -v sha256sum >/dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }

( cd "${source_root}" && grep -v '^#' "${script_dir}/SOURCES.sha256" \
    | sha -c --quiet - ) || {
    echo "a WRF source differs from SOURCES.sha256; refusing to build" >&2
    exit 3
}

cd "${build_dir}"
rm -rf fixtures ./*.o ./*.mod run_partial_nest
python3 "${script_dir}/extract.py" "${source_root}" "${build_dir}"

flags=(-c -O0 -ffp-contract=off -ffree-form -ffree-line-length-none)
"${fc[@]}" "${flags[@]}" "${script_dir}/stubs.F90"
"${fc[@]}" "${flags[@]}" wrf_sint.F90
"${fc[@]}" "${flags[@]}" wrf_interp_fcn.F90
"${fc[@]}" "${flags[@]}" wrf_bc.F90
"${fc[@]}" "${flags[@]}" "${script_dir}/../urban_wrf471_oracle/oracle_io.F90"
"${fc[@]}" "${flags[@]}" "${script_dir}/run_partial_nest.F90"

: > libmvec-report.txt
for obj in wrf_sint.o wrf_interp_fcn.o wrf_bc.o run_partial_nest.o; do
    if nm -u "${obj}" | grep -q '_ZGV'; then
        echo "-O0 object ${obj} pulled in a libmvec vector symbol" >&2
        exit 4
    fi
    echo "# ${obj} -O0 libm symbols:" >> libmvec-report.txt
    nm -u "${obj}" | grep -E 'expf|powf|logf|sqrtf|sinf|cosf' \
        >> libmvec-report.txt || true
done
cat > libmvec_positive_control.F90 <<'PCF'
subroutine positive_control(x, n)
  integer, intent(in) :: n
  real, intent(inout) :: x(n)
  integer :: i
  do i = 1, n
    x(i) = exp(x(i))
  end do
end subroutine positive_control
PCF
"${fc[@]}" -c -Ofast -ftree-vectorize -o libmvec_positive_control.o \
    libmvec_positive_control.F90
if nm -u libmvec_positive_control.o | grep -q '_ZGV'; then
    echo "# positive control (plain expf loop at -Ofast):" >> libmvec-report.txt
    nm -u libmvec_positive_control.o | grep -E 'expf|_ZGV' >> libmvec-report.txt
elif [[ "$(uname -s)" == "Linux" ]]; then
    echo "libmvec positive control produced no _ZGV symbol; the guard is" \
         "vacuous on this toolchain" >&2
    exit 5
else
    echo "# positive control: no _ZGV symbol on $(uname -s) (no glibc libmvec);" \
         "the guard is informational here" >> libmvec-report.txt
fi
# The operators call no libm function at all (SINT, copy_fcn, the
# smoothers and the relaxation are + - * / and MIN/MAX/SIGN only).

"${fc[@]}" -o run_partial_nest run_partial_nest.o wrf_sint.o wrf_interp_fcn.o \
    wrf_bc.o stubs.o oracle_io.o
mkdir -p fixtures
./run_partial_nest fixtures

{
    echo "gfortran: $("${fc[@]}" --version | head -1)"
    echo "flags: -O0 -ffp-contract=off -ffree-form (REAL = real(4))"
    echo "host: $(uname -srm)"
} > fixtures/TOOLCHAIN.txt
cp libmvec-report.txt fixtures/
( cd "${source_root}" && grep -v '^#' "${script_dir}/SOURCES.sha256" ) \
    > fixtures/SOURCES.sha256
( cd "${script_dir}" && sha build.sh fetch_sources.sh extract.py stubs.F90 \
    run_partial_nest.F90 ../urban_wrf471_oracle/oracle_io.F90 ) > fixtures/HARNESS.sha256
( cd fixtures && find . -type f \( -name '*.bin' -o -name 'MANIFEST.txt' \) \
    | LC_ALL=C sort | while read -r f; do sha "$f"; done ) > fixtures/FIXTURES.sha256
echo "fixtures in ${build_dir}/fixtures"
