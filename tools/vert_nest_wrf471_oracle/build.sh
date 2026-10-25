#!/usr/bin/env bash
# Build and run the vertical-nesting oracle against WRF v4.7.1.
#
#   bash tools/vert_nest_wrf471_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR
#
# WRF_SOURCE_ROOT holds the files SOURCES.sha256 pins (fetch_sources.sh lays
# them out); every one is checked before anything compiles.  extract.py cuts
# WRF's own statements (share/interp_fcn.F's vertical-nesting routines,
# dyn_em/nest_init_utils.F's coefficient code, and force_domain_em_part2.F's
# reference heights and rebalance) byte for byte; run_vert_nest.F90 expands
# them inside subroutines whose arguments carry WRF's names, and writes the
# fixture with ../urban_wrf471_oracle/oracle_io.F90.
#
# Reference build: -O0, REAL is real(4) (no -r8).  WRF builds share/ and
# dyn_em/ at -O2 -ftree-vectorize, where gfortran may call glibc's 4-ULP
# vector logf/powf (libmvec).  The script fails if an -O0 object references
# a _ZGV* symbol, and proves with a positive control that the grep can see
# one on this toolchain.  Copy BUILD_DIR/fixtures to
# tests/data/oracles/vert_nest/ to publish it.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

( cd "${source_root}" && grep -v '^#' "${script_dir}/SOURCES.sha256" \
    | sha256sum -c --quiet - ) || {
    echo "a WRF source differs from SOURCES.sha256; refusing to build" >&2
    exit 3
}

mkdir -p "${build_dir}"
cd "${build_dir}"
rm -rf fixtures *.o *.mod run_vert_nest
python3 "${script_dir}/extract.py" "${source_root}" "${build_dir}"
cp "${script_dir}/nest_forcedown_interp_vert.inc" .

ff=(-ffree-form -ffree-line-length-none)
gfortran -c -O0 "${ff[@]}" wrf_vert_nest_interp.F90
gfortran -c -O0 "${ff[@]}" wrf_vcoord.F90
gfortran -c -O0 "${ff[@]}" "${script_dir}/../urban_wrf471_oracle/oracle_io.F90"
gfortran -c -O0 -cpp -I "${build_dir}" "${ff[@]}" \
    "${script_dir}/run_vert_nest.F90"

: > libmvec-report.txt
for obj in wrf_vert_nest_interp.o wrf_vcoord.o run_vert_nest.o; do
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
gfortran -c -Ofast -ftree-vectorize -o libmvec_positive_control.o \
    libmvec_positive_control.F90
if ! nm -u libmvec_positive_control.o | grep -q '_ZGV'; then
    echo "libmvec positive control produced no _ZGV symbol; the guard is" \
         "vacuous on this toolchain" >&2
    exit 5
fi
echo "# positive control (plain expf loop at -Ofast):" >> libmvec-report.txt
nm -u libmvec_positive_control.o | grep -E 'expf|_ZGV' >> libmvec-report.txt

gfortran -o run_vert_nest run_vert_nest.o wrf_vert_nest_interp.o \
    wrf_vcoord.o oracle_io.o
mkdir -p fixtures
./run_vert_nest fixtures

{
    echo "gfortran: $(gfortran --version | head -1)"
    echo "glibc: $(ldd --version | head -1)"
    echo "host: $(uname -srm)"
} > fixtures/TOOLCHAIN.txt
cp libmvec-report.txt fixtures/
( cd "${source_root}" && grep -v '^#' "${script_dir}/SOURCES.sha256" ) \
    > fixtures/SOURCES.sha256
( cd "${script_dir}" && sha256sum build.sh fetch_sources.sh extract.py \
    run_vert_nest.F90 nest_forcedown_interp_vert.inc \
    ../urban_wrf471_oracle/oracle_io.F90 ) > fixtures/HARNESS.sha256
( cd fixtures && find . -type f \( -name '*.bin' -o -name 'MANIFEST.txt' \) \
    | LC_ALL=C sort | xargs sha256sum ) > fixtures/FIXTURES.sha256
echo "fixtures in ${build_dir}/fixtures"
