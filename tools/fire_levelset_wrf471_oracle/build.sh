#!/usr/bin/env bash
# Build and run the fire-model oracle against WRF v4.7.1.
#
#   bash tools/fire_levelset_wrf471_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR
#
# WRF_SOURCE_ROOT holds the files SOURCES.sha256 pins (fetch_sources.sh lays
# them out); every one is checked before anything compiles, so a fixture can
# never come from an edited file.  What is compiled, all byte-unmodified:
#
#   share/module_model_constants.F   XLV for heat_fluxes
#   frame/module_wrf_error.F         wrf_message / wrf_error_fatal
#   phys/module_fr_fire_util.F       flags, continue_at_boundary,
#                                    sum_2d_cells, set_ideal_coord
#   phys/module_fr_fire_phys.F       fire_params, fire_ros, heat_fluxes
#   phys/module_fr_fire_core.F       init_no_fire, ignite_fire, nearest,
#                                    fuel_left, prop_ls_rk3, tend_ls,
#                                    reinit_ls_rk3, tign_update
#
# with -DRWORDSIZE=4 (WRF's REAL is REAL(4)), plus stub_wrf.F90 (types and
# single-rank shims, no physics), ../urban_wrf471_oracle/oracle_io.F90 (the
# fixture writer) and run_fire.F90 (the driver).  The driver's
# module_fr_fire_driver.F:941-953 scaling is checked against
# driver_941_953.inc here.  Copy BUILD_DIR/fixtures to
# tests/data/oracles/fire_levelset/ to publish it.
#
# Reference build: -O0.  WRF builds phys/ at -O2 -ftree-vectorize, where
# gfortran may call glibc's vector expf/powf (libmvec).  The script fails if
# an -O0 object references a _ZGV* symbol, proves with a positive control
# that the grep can see one on this toolchain, and records which vector
# symbols WRF's own -O2 flags would pull in (libmvec-report.txt).
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
if [[ "${build_dir}" == "${source_root}" || "${build_dir}" == "${source_root}/"* ]]; then
    echo "BUILD_DIR inside the WRF source would modify the reference tree" >&2
    exit 7
fi
sed -n '941,953p' "${source_root}/phys/module_fr_fire_driver.F" \
    | cmp - "${script_dir}/driver_941_953.inc" || {
    echo "module_fr_fire_driver.F:941-953 no longer reads as" \
         "driver_941_953.inc; the driver's scaling moved" >&2
    exit 8
}

mkdir -p "${build_dir}"
cd "${build_dir}"
rm -rf fixtures ./*.o ./*.mod run_fire

defines="-DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8"
defines="${defines} -DLWORDSIZE=4"
ff=(-ffree-form -ffree-line-length-none)
fc() { gfortran -c -O0 -cpp ${defines} "${ff[@]}" -fallow-argument-mismatch \
    -I "${build_dir}" "$@"; }

gfortran -c -O0 "${ff[@]}" "${script_dir}/stub_wrf.F90"
fc -o module_model_constants.o "${source_root}/share/module_model_constants.F"
fc -o module_wrf_error.o "${source_root}/frame/module_wrf_error.F"
fc -o module_fr_fire_util.o "${source_root}/phys/module_fr_fire_util.F"
fc -o module_fr_fire_phys.o "${source_root}/phys/module_fr_fire_phys.F"
fc -o module_fr_fire_core.o "${source_root}/phys/module_fr_fire_core.F"
gfortran -c -O0 "${ff[@]}" "${script_dir}/../urban_wrf471_oracle/oracle_io.F90"
gfortran -c -O0 "${ff[@]}" -I "${build_dir}" "${script_dir}/run_fire.F90"

wrf_objects=(module_model_constants.o module_wrf_error.o
             module_fr_fire_util.o module_fr_fire_phys.o module_fr_fire_core.o)

# --- libmvec receipts --------------------------------------------------------
: > libmvec-report.txt
for obj in "${wrf_objects[@]}"; do
    if nm -u "${obj}" | grep -q '_ZGV'; then
        echo "-O0 object ${obj} pulled in a libmvec vector symbol" >&2
        nm -u "${obj}" | grep '_ZGV' >&2
        exit 4
    fi
    echo "# ${obj} -O0 libm symbols:" >> libmvec-report.txt
    nm -u "${obj}" | grep -E 'expf|powf|logf|sqrtf|sinf|cosf' \
        >> libmvec-report.txt || true
done
for src in phys/module_fr_fire_phys.F phys/module_fr_fire_core.F; do
    gfortran -c -O2 -ftree-vectorize -funroll-loops -cpp ${defines} "${ff[@]}" \
        -fallow-argument-mismatch \
        -I "${build_dir}" -o O2vec.o "${source_root}/${src}"
    echo "# ${src} at WRF's -O2 -ftree-vectorize -funroll-loops:" >> libmvec-report.txt
    nm -u O2vec.o | grep -E 'expf|powf|logf|sqrtf|_ZGV' >> libmvec-report.txt || true
    rm -f O2vec.o
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

# --- the driver ----------------------------------------------------------------
gfortran -o run_fire run_fire.o oracle_io.o "${wrf_objects[@]}" stub_wrf.o
mkdir -p fixtures
./run_fire fixtures

# --- receipts -------------------------------------------------------------------
{
    echo "gfortran: $(gfortran --version | head -1)"
    echo "glibc: $(ldd --version | head -1)"
    echo "host: $(uname -srm)"
} > fixtures/TOOLCHAIN.txt
cp libmvec-report.txt fixtures/
( cd "${source_root}" && grep -v '^#' "${script_dir}/SOURCES.sha256" ) \
    > fixtures/SOURCES.sha256
( cd "${script_dir}" && sha256sum build.sh fetch_sources.sh stub_wrf.F90 \
    run_fire.F90 driver_941_953.inc ../urban_wrf471_oracle/oracle_io.F90 ) \
    > fixtures/HARNESS.sha256
( cd fixtures && find . -type f \( -name '*.bin' -o -name 'MANIFEST.txt' \) \
    | LC_ALL=C sort | xargs sha256sum ) > fixtures/FIXTURES.sha256
echo "fixtures in ${build_dir}/fixtures"
