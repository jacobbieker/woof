#!/usr/bin/env bash
# Build and run the Fitch wind-farm column oracle against WRF v4.7.1.
#
#   bash tools/windfarm_fitch_wrf471_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR
#
# WRF_SOURCE_ROOT holds the files SOURCES.sha256 pins (fetch_sources.sh lays
# them out); every one is checked before anything compiles, so a fixture can
# never come from an edited file.  What is compiled, byte-unmodified:
#
#   phys/module_wind_fitch.F:52-63,67-334   the turbine state, dragforce,
#                                            turbine_area, dragcof
#   share/module_model_constants.F:72        piconst
#
# cut by extract.py into one module (the file's initialisation routine needs
# WRF's map projections and MPI layer, which the drag never touches), plus
# ../urban_wrf471_oracle/oracle_io.F90 (the fixture writer) and run_fitch.F90
# here.  The driver writes its fixture under BUILD_DIR/fixtures; copy that
# directory to woof/data/windfarm_fitch/oracle/ to publish it.
#
# REAL is real(4), WRF's RWORDSIZE=4 default; no -r8 or -fdefault-real-8.
#
# Reference build: -O0.  WRF builds phys/ at -O2 -ftree-vectorize, where
# gfortran may call glibc's vector powf/asinf (libmvec).  The script fails if
# an -O0 object references a _ZGV* symbol, proves with a positive control that
# the grep can see one on this toolchain, and records which vector symbols
# WRF's own -O2 flags would pull in (libmvec-report.txt).
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# --- byte identity to the pinned sources ------------------------------------
( cd "${source_root}" && grep -v '^#' "${script_dir}/SOURCES.sha256" \
    | sha256sum -c --quiet - ) || {
    echo "a WRF source differs from SOURCES.sha256; refusing to build" >&2
    exit 3
}

mkdir -p "${build_dir}"
cd "${build_dir}"
rm -rf fixtures *.o *.mod run_fitch
python3 "${script_dir}/extract.py" "${source_root}" "${build_dir}"

ff=(-ffree-form -ffree-line-length-none)
gfortran -c -O0 "${ff[@]}" -o wrf_wind_fitch_inline.o wrf_wind_fitch_inline.F90
gfortran -c -O0 "${ff[@]}" "${script_dir}/../urban_wrf471_oracle/oracle_io.F90"

# --- libmvec receipts --------------------------------------------------------
: > libmvec-report.txt
if nm -u wrf_wind_fitch_inline.o | grep -q '_ZGV'; then
    echo "-O0 object wrf_wind_fitch_inline.o pulled in a libmvec vector symbol" >&2
    nm -u wrf_wind_fitch_inline.o | grep '_ZGV' >&2
    exit 4
fi
echo "# wrf_wind_fitch_inline.o -O0 libm symbols:" >> libmvec-report.txt
nm -u wrf_wind_fitch_inline.o | grep -E 'powf|asinf|sqrtf|expf|logf' \
    >> libmvec-report.txt || true
gfortran -c -O2 -ftree-vectorize -funroll-loops "${ff[@]}" -o O2vec.o \
    wrf_wind_fitch_inline.F90
echo "# wrf_wind_fitch_inline.F90 at WRF's -O2 -ftree-vectorize -funroll-loops:" \
    >> libmvec-report.txt
nm -u O2vec.o | grep -E 'powf|asinf|sqrtf|_ZGV' >> libmvec-report.txt || true
rm -f O2vec.o
cat > libmvec_positive_control.F90 <<'EOF'
subroutine positive_control(x, n)
  integer, intent(in) :: n
  real, intent(inout) :: x(n)
  integer :: i
  do i = 1, n
    x(i) = exp(x(i))
  end do
end subroutine positive_control
EOF
gfortran -c -Ofast -ftree-vectorize -o libmvec_positive_control.o \
    libmvec_positive_control.F90
if ! nm -u libmvec_positive_control.o | grep -q '_ZGV'; then
    echo "libmvec positive control produced no _ZGV symbol; the guard is" \
         "vacuous on this toolchain" >&2
    exit 5
fi
echo "# positive control (plain expf loop at -Ofast):" >> libmvec-report.txt
nm -u libmvec_positive_control.o | grep -E 'expf|_ZGV' >> libmvec-report.txt

# --- driver --------------------------------------------------------------------
gfortran -c -O0 "${ff[@]}" "${script_dir}/run_fitch.F90"
gfortran -o run_fitch run_fitch.o wrf_wind_fitch_inline.o oracle_io.o
mkdir -p fixtures
./run_fitch fixtures

# --- receipts -------------------------------------------------------------------
{
    echo "gfortran: $(gfortran --version | head -1)"
    echo "glibc: $(ldd --version | head -1)"
    echo "host: $(uname -srm)"
} > fixtures/TOOLCHAIN.txt
cp libmvec-report.txt fixtures/
( cd "${source_root}" && grep -v '^#' "${script_dir}/SOURCES.sha256" ) \
    > fixtures/SOURCES.sha256
( cd "${script_dir}" && sha256sum build.sh fetch_sources.sh extract.py run_fitch.F90 \
    ../urban_wrf471_oracle/oracle_io.F90 ) > fixtures/HARNESS.sha256
( cd fixtures && find . -type f \( -name '*.bin' -o -name 'MANIFEST.txt' \) \
    | LC_ALL=C sort | xargs sha256sum ) \
    > fixtures/FIXTURES.sha256
echo "fixtures in ${build_dir}/fixtures"
