#!/usr/bin/env bash
# Build and run the fire-spread oracle against WRF v4.7.1.
#
#   bash tools/fire_ros_wrf471_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR
#
# WRF_SOURCE_ROOT holds the files SOURCES.sha256 pins (fetch_sources.sh lays
# them out); every one is checked before anything compiles, so a fixture can
# never come from an edited file.  What is compiled, all byte-unmodified:
#
#   share/module_model_constants.F   (cp, xlv for module_fr_fire_phys)
#   phys/module_fr_fire_util.F       (flags, message/crash; serial build)
#   phys/module_fr_fire_phys.F       init_fuel_cats, set_fire_params, fire_ros
#
# plus wrf_stubs.F90 (serial stand-ins for the WRF framework calls),
# ../urban_wrf471_oracle/oracle_io.F90 (the fixture writer) and
# run_fire_ros.F90.  The driver runs twice: beside the pinned
# test/em_fire/namelist.fire ("namelist") and with no namelist.fire at all
# ("defaults", the module's compiled DATA).  Copy BUILD_DIR/fixtures to
# woof/data/fire/oracle/ to publish.
#
# Reference build: -O0, default real(4) as WRF's RWORDSIZE=4 build.  WRF
# builds phys/ at -O2 -ftree-vectorize, where gfortran may call glibc's
# 4-ULP vector expf/powf (libmvec).  The script fails if an -O0 object
# references a _ZGV* symbol, proves with a positive control that the grep
# can see one on this toolchain, and records which vector symbols WRF's own
# -O2 flags would pull in (libmvec-report.txt).
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
rm -rf fixtures run_namelist run_defaults *.o *.mod run_fire_ros

ff=(-ffree-form -ffree-line-length-none)
gfortran -c -O0 "${ff[@]}" "${script_dir}/wrf_stubs.F90"
gfortran -c -O0 -cpp "${ff[@]}" -o module_model_constants.o \
    "${source_root}/share/module_model_constants.F"
gfortran -c -O0 -cpp "${ff[@]}" -o module_fr_fire_util.o \
    "${source_root}/phys/module_fr_fire_util.F"
# WRF passes scalars and arrays to the same wrf_dm_bcast_* externals; WRF's
# own configure adds -fallow-argument-mismatch for gfortran >= 10, and so
# does this build (it changes no generated arithmetic).
gfortran -c -O0 -cpp -fallow-argument-mismatch "${ff[@]}" \
    -o module_fr_fire_phys.o "${source_root}/phys/module_fr_fire_phys.F"
gfortran -c -O0 "${ff[@]}" "${script_dir}/../urban_wrf471_oracle/oracle_io.F90"

wrf_objects=(module_fr_fire_util.o module_fr_fire_phys.o)

# --- libmvec receipts --------------------------------------------------------
: > libmvec-report.txt
for obj in "${wrf_objects[@]}"; do
    if nm -u "${obj}" | grep -q '_ZGV'; then
        echo "-O0 object ${obj} pulled in a libmvec vector symbol" >&2
        nm -u "${obj}" | grep '_ZGV' >&2
        exit 4
    fi
    echo "# ${obj} -O0 libm symbols:" >> libmvec-report.txt
    nm -u "${obj}" | grep -E 'expf|powf|logf|sqrtf|exp$|pow$' \
        >> libmvec-report.txt || true
done
gfortran -c -O2 -ftree-vectorize -funroll-loops -cpp -fallow-argument-mismatch \
    "${ff[@]}" -I "${build_dir}" -o O2vec.o "${source_root}/phys/module_fr_fire_phys.F"
echo "# phys/module_fr_fire_phys.F at WRF's -O2 -ftree-vectorize -funroll-loops:" \
    >> libmvec-report.txt
nm -u O2vec.o | grep -E 'expf|powf|logf|sqrtf|_ZGV' >> libmvec-report.txt || true
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

# --- driver ----------------------------------------------------------------------
gfortran -c -O0 "${ff[@]}" "${script_dir}/run_fire_ros.F90"
gfortran -o run_fire_ros run_fire_ros.o module_fr_fire_phys.o \
    module_fr_fire_util.o module_model_constants.o wrf_stubs.o oracle_io.o

mkdir -p fixtures run_namelist run_defaults
cp "${source_root}/test/em_fire/namelist.fire" run_namelist/
( cd run_namelist && "${build_dir}/run_fire_ros" "${build_dir}/fixtures" namelist \
    2> init.log )
( cd run_defaults && "${build_dir}/run_fire_ros" "${build_dir}/fixtures" defaults \
    2> init.log )
cp run_namelist/namelist.fire.output fixtures/namelist/namelist.fire.output
cp run_defaults/namelist.fire.output fixtures/defaults/namelist.fire.output

# --- receipts -------------------------------------------------------------------
{
    echo "gfortran: $(gfortran --version | head -1)"
    echo "glibc: $(ldd --version | head -1)"
    echo "host: $(uname -srm)"
} > fixtures/TOOLCHAIN.txt
cp libmvec-report.txt fixtures/
( cd "${source_root}" && grep -v '^#' "${script_dir}/SOURCES.sha256" ) \
    > fixtures/SOURCES.sha256
( cd "${script_dir}" && sha256sum build.sh fetch_sources.sh wrf_stubs.F90 \
    run_fire_ros.F90 ../urban_wrf471_oracle/oracle_io.F90 ) > fixtures/HARNESS.sha256
( cd fixtures && find . -type f \( -name '*.bin' -o -name 'MANIFEST.txt' \) \
    | LC_ALL=C sort | xargs sha256sum ) \
    > fixtures/FIXTURES.sha256
echo "fixtures in ${build_dir}/fixtures"
