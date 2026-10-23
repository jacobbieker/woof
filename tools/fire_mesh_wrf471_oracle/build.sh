#!/usr/bin/env bash
# Build and run the fire-mesh oracle against WRF v4.7.1.
#
#   bash tools/fire_mesh_wrf471_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR
#
# WRF_SOURCE_ROOT holds the files SOURCES.sha256 pins (fetch_sources.sh lays
# them out); every one is checked before anything compiles, so a fixture can
# never come from an edited file.  What is compiled, all byte-unmodified:
#
#   phys/module_fr_fire_util.F          whole: interpolate_z2fire,
#                                       continue_at_boundary, interpolate_2d
#   phys/module_fr_fire_model.F:194-208 ZSF continuation and the centred
#                                       slope (cut by extract.py)
#   phys/module_fr_fire_driver.F:566-567 dxf = dx/ir (cut by extract.py)
#
# share/module_llxy.F is pinned here too and compiled by build_llxy_fire.py
# (the fire-cell placement fixture, llxy_fire_mesh/).
#
# plus wrf_error_stub.F90 (module_wrf_error's two entries),
# ../urban_wrf471_oracle/oracle_io.F90 (the fixture writer) and
# run_fire_mesh.F90.  The fixture lands in BUILD_DIR/fixtures; copy that
# directory to woof/data/fire/oracle/ to publish it.
#
# Reference build: -O0, default REAL(4) (WRF's RWORDSIZE=4), no OpenMP.
# The routines call no libm, but the build still fails if an -O0 object
# references a _ZGV* (libmvec) symbol, and proves with a positive control
# that the grep can see one on this toolchain.
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
rm -rf fixtures ./*.o ./*.mod run_fire_mesh
python3 "${script_dir}/extract.py" "${source_root}" "${build_dir}"

ff=(-ffree-form -ffree-line-length-none)
gfortran -c -O0 "${ff[@]}" "${script_dir}/wrf_error_stub.F90"
gfortran -c -O0 -cpp "${ff[@]}" -o module_fr_fire_util.o \
    "${source_root}/phys/module_fr_fire_util.F"
gfortran -c -O0 "${ff[@]}" wrf_fire_slope_cut.F90
gfortran -c -O0 "${ff[@]}" "${script_dir}/../urban_wrf471_oracle/oracle_io.F90"
gfortran -c -O0 "${ff[@]}" "${script_dir}/run_fire_mesh.F90"

wrf_objects=(module_fr_fire_util.o wrf_fire_slope_cut.o)

# --- libmvec receipts --------------------------------------------------------
: > libmvec-report.txt
for obj in "${wrf_objects[@]}"; do
    if nm -u "${obj}" | grep -q '_ZGV'; then
        echo "-O0 object ${obj} pulled in a libmvec vector symbol" >&2
        nm -u "${obj}" | grep '_ZGV' >&2
        exit 4
    fi
    echo "# ${obj} -O0 libm symbols:" >> libmvec-report.txt
    nm -u "${obj}" | grep -E 'expf|powf|logf|sqrtf|tanhf|atan2f|sinf|cosf' \
        >> libmvec-report.txt || true
done
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

# --- driver -------------------------------------------------------------------
gfortran -o run_fire_mesh run_fire_mesh.o wrf_fire_slope_cut.o \
    module_fr_fire_util.o wrf_error_stub.o oracle_io.o
mkdir -p fixtures
./run_fire_mesh fixtures

# --- receipts -------------------------------------------------------------------
{
    echo "gfortran: $(gfortran --version | head -1)"
    echo "glibc: $(ldd --version | head -1)"
    echo "host: $(uname -srm)"
} > fixtures/TOOLCHAIN.txt
cp libmvec-report.txt fixtures/
( cd "${source_root}" && grep -v '^#' "${script_dir}/SOURCES.sha256" ) \
    > fixtures/SOURCES.sha256
( cd "${script_dir}" && sha256sum build.sh fetch_sources.sh extract.py \
    run_fire_mesh.F90 wrf_error_stub.F90 ../urban_wrf471_oracle/oracle_io.F90 ) \
    > fixtures/HARNESS.sha256
( cd fixtures && find . -type f \( -name '*.bin' -o -name 'MANIFEST.txt' \) \
    | LC_ALL=C sort | xargs sha256sum ) \
    > fixtures/FIXTURES.sha256
echo "fixtures in ${build_dir}/fixtures"
