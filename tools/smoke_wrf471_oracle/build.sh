#!/usr/bin/env bash
# Build and run the smoke-physics oracles against WRF v4.7.1.
#
#   bash tools/smoke_wrf471_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR
#
# WRF_SOURCE_ROOT holds the files SOURCES.sha256 pins (fetch_sources.sh lays
# them out); every one is checked before anything compiles, so a fixture can
# never come from an edited file.  What is compiled, all byte-unmodified:
#
#   share/module_model_constants.F      WRF's constants (cp, xlv, g, r_d ...)
#   chem/module_zero_plumegen_coms.F    the plume model's common state
#   chem/module_chem_plumerise_scalar.F the Freitas et al. (2007) 1-D plume
#                                       rise model (plumerise, makeplume ...)
#   phys/module_fr_fire_atm.F           WRF-SFIRE's add_fire_tracer_emissions
#                                       (tracer_opt = 3, the fire_smoke tracer)
#   chem/module_aer_drydep.F:1650-1941  zhang2001_aer_drydepvel_1 (cut by
#                                       extract.py: the module's drivers need
#                                       the whole WRF-Chem framework)
#   chem/module_gocart_settling.F:206-367  settling (cut by extract.py)
#
# plus stub_wrf.F90 (module_state_description's two tracer indices and the
# two framework calls module_fr_fire_atm.F names), oracle_io.F90 (the fixture
# writer) and the run_*.F90 drivers here.  Every WRF routine is real(4)
# except settling, which WRF declares REAL*8.  Each driver writes its cases
# under BUILD_DIR/fixtures; copy that directory to tests/data/oracles/smoke/
# to publish it.
#
# Reference build: -O0.  WRF builds chem/ and phys/ at -O2 -ftree-vectorize,
# where gfortran may call glibc's 4-ULP vector expf/powf/logf (libmvec).  The
# script fails if an -O0 object references a _ZGV* symbol, proves with a
# positive control that the grep can see one on this toolchain, and records
# which vector symbols WRF's own -O2 flags would pull in (libmvec-report.txt).
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
rm -rf fixtures ./*.o ./*.mod run_*
python3 "${script_dir}/extract.py" "${source_root}" "${build_dir}" \
    > extract-report.txt

ff=(-ffree-form -ffree-line-length-none)
gfortran -c -O0 -cpp "${ff[@]}" "${source_root}/share/module_model_constants.F"
gfortran -c -O0 -cpp "${ff[@]}" "${source_root}/chem/module_zero_plumegen_coms.F"
gfortran -c -O0 -cpp "${ff[@]}" \
    "${source_root}/chem/module_chem_plumerise_scalar.F"
gfortran -c -O0 -cpp "${ff[@]}" "${script_dir}/stub_wrf.F90"
gfortran -c -O0 -cpp "${ff[@]}" "${source_root}/phys/module_fr_fire_atm.F"
gfortran -c -O0 -cpp "${ff[@]}" cut_zhang2001.F90
gfortran -c -O0 -cpp "${ff[@]}" cut_gocart_settling.F90
gfortran -c -O0 "${ff[@]}" "${script_dir}/oracle_io.F90"

wrf_objects=(module_model_constants.o module_zero_plumegen_coms.o
             module_chem_plumerise_scalar.o module_fr_fire_atm.o
             cut_zhang2001.o cut_gocart_settling.o)

# --- libmvec receipts --------------------------------------------------------
: > libmvec-report.txt
for obj in "${wrf_objects[@]}"; do
    if nm -u "${obj}" | grep -q '_ZGV'; then
        echo "-O0 object ${obj} pulled in a libmvec vector symbol" >&2
        nm -u "${obj}" | grep '_ZGV' >&2
        exit 4
    fi
    echo "# ${obj} -O0 libm symbols:" >> libmvec-report.txt
    nm -u "${obj}" | grep -E 'exp|pow|log|sqrt|tanh' \
        >> libmvec-report.txt || true
done
for src in "${source_root}/chem/module_chem_plumerise_scalar.F" \
           cut_zhang2001.F90 cut_gocart_settling.F90; do
    gfortran -c -O2 -ftree-vectorize -funroll-loops -cpp "${ff[@]}" \
        -I "${build_dir}" -o O2vec.o "${src}"
    echo "# $(basename "${src}") at WRF's -O2 -ftree-vectorize -funroll-loops:" \
        >> libmvec-report.txt
    nm -u O2vec.o | grep -E 'exp|pow|log|sqrt|tanh|_ZGV' \
        >> libmvec-report.txt || true
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

# --- drivers -------------------------------------------------------------------
drivers=(run_emission run_plumerise run_zhang2001 run_gocart_settling)
for drv in "${drivers[@]}"; do
    gfortran -c -O0 "${ff[@]}" "${script_dir}/${drv}.F90"
done
gfortran -o run_emission run_emission.o module_fr_fire_atm.o stub_wrf.o \
    module_model_constants.o oracle_io.o
gfortran -o run_plumerise run_plumerise.o module_chem_plumerise_scalar.o \
    module_zero_plumegen_coms.o module_model_constants.o oracle_io.o
gfortran -o run_zhang2001 run_zhang2001.o cut_zhang2001.o oracle_io.o
gfortran -o run_gocart_settling run_gocart_settling.o cut_gocart_settling.o \
    oracle_io.o

mkdir -p fixtures
./run_emission fixtures
./run_zhang2001 fixtures
./run_gocart_settling fixtures
# The plume model keeps its state in module variables that it zeroes only on
# its first call (module_chem_plumerise_scalar.F:117-120, 297-302), so every
# later column starts from whatever the one before it left.  One process per
# column gives every case WRF's first-call state; see PROVENANCE.md.
ncase=$(./run_plumerise --count)
for icase in $(seq 1 "${ncase}"); do
    ./run_plumerise fixtures "${icase}"
done

# --- receipts -------------------------------------------------------------------
{
    echo "gfortran: $(gfortran --version | head -1)"
    echo "glibc: $(ldd --version | head -1)"
    echo "host: $(uname -srm)"
} > fixtures/TOOLCHAIN.txt
cp libmvec-report.txt extract-report.txt fixtures/
( cd "${source_root}" && grep -v '^#' "${script_dir}/SOURCES.sha256" ) \
    > fixtures/SOURCES.sha256
( cd "${script_dir}" && sha256sum build.sh fetch_sources.sh extract.py \
    stub_wrf.F90 oracle_io.F90 run_*.F90 ) > fixtures/HARNESS.sha256
( cd fixtures && find . -type f \( -name '*.bin' -o -name 'MANIFEST.txt' \) \
    | LC_ALL=C sort | xargs sha256sum ) \
    > fixtures/FIXTURES.sha256
echo "fixtures in ${build_dir}/fixtures"
