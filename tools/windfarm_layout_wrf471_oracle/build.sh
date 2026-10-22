#!/usr/bin/env bash
# Build and run the wind-farm layout oracle against WRF v4.7.1.
#
#   bash tools/windfarm_layout_wrf471_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR
#
# WRF_SOURCE_ROOT holds the files SOURCES.sha256 pins (fetch_sources.sh lays
# them out); every one is checked before anything compiles, so a fixture can
# never come from an edited file.  What is compiled, all byte-unmodified:
#
#   share/module_model_constants.F   piconst
#   share/module_llxy.F              map_init, map_set, set_lc, lc_cone,
#                                    latlon_to_ij, llij_lc
#   phys/module_wind_fitch.F         cut by extract.py: its module data,
#                                    dragcof, init_module_wind_fitch's
#                                    per-turbine placement and table read,
#                                    and dragforce's area/power1
#
# all default REAL (real(4), WRF's -DRWORDSIZE=4 build), plus stubs.F90 (the
# module_wrf_error entry points), wf_fitch.F90 (the module that holds the
# cuts), ../urban_wrf471_oracle/oracle_io.F90 (the fixture writer) and
# run_windfarm_layout.F90.  The decks are synthetic farms woof lays out and
# writes as WRF inputs (make_deck.py, run with ${PYTHON:-python3}); the
# driver runs inside each deck directory, as WRF runs beside its inputs,
# and writes BUILD_DIR/fixtures/<case>/{placement,tables,dragcof}.  Copy
# BUILD_DIR/fixtures to tests/data/oracles/windfarm_layout/ to publish.
#
# Reference build: -O0.  WRF builds phys/ at -O2 -ftree-vectorize, where
# gfortran may call glibc's 4-ULP vector expf/powf/logf (libmvec).  The script
# fails if an -O0 object references a _ZGV* symbol, proves with a positive
# control that the grep can see one on this toolchain, and records which
# libm symbols each object calls (libmvec-report.txt).
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)
python=${PYTHON:-python3}

# --- byte identity to the pinned sources ------------------------------------
( cd "${source_root}" && grep -v '^#' "${script_dir}/SOURCES.sha256" \
    | sha256sum -c --quiet - ) || {
    echo "a WRF source differs from SOURCES.sha256; refusing to build" >&2
    exit 3
}

mkdir -p "${build_dir}"
cd "${build_dir}"
rm -rf fixtures decks *.o *.mod *.inc run_windfarm_layout
"${python}" "${script_dir}/extract.py" "${source_root}" "${build_dir}"

ff=(-ffree-form -ffree-line-length-none)
gfortran -c -O0 -cpp "${ff[@]}" -o module_model_constants.o \
    "${source_root}/share/module_model_constants.F"
gfortran -c -O0 "${ff[@]}" "${script_dir}/stubs.F90"
gfortran -c -O0 -cpp "${ff[@]}" -o module_llxy.o \
    "${source_root}/share/module_llxy.F"
gfortran -c -O0 "${ff[@]}" -I "${build_dir}" -o wf_fitch.o \
    "${script_dir}/wf_fitch.F90"
gfortran -c -O0 "${ff[@]}" "${script_dir}/../urban_wrf471_oracle/oracle_io.F90"
gfortran -c -O0 "${ff[@]}" -I "${build_dir}" -o run_windfarm_layout.o \
    "${script_dir}/run_windfarm_layout.F90"

wrf_objects=(module_llxy.o wf_fitch.o run_windfarm_layout.o)

# --- libmvec receipts --------------------------------------------------------
: > libmvec-report.txt
for obj in "${wrf_objects[@]}"; do
    if nm -u "${obj}" | grep -q '_ZGV'; then
        echo "-O0 object ${obj} pulled in a libmvec vector symbol" >&2
        nm -u "${obj}" | grep '_ZGV' >&2
        exit 4
    fi
    echo "# ${obj} -O0 libm symbols:" >> libmvec-report.txt
    nm -u "${obj}" | grep -E 'expf|powf|logf|log10f|sqrtf|tanf|atan2f|sinf|cosf' \
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

gfortran -o run_windfarm_layout run_windfarm_layout.o wf_fitch.o \
    module_llxy.o module_model_constants.o stubs.o oracle_io.o

# --- decks and fixtures --------------------------------------------------------
( cd "${repo_root}" && "${python}" -m tools.windfarm_layout_wrf471_oracle.make_deck \
    "${build_dir}/decks" )
mkdir -p fixtures
for deck in decks/*/; do
    case=$(basename "${deck}")
    ( cd "${deck}" && "${build_dir}/run_windfarm_layout" \
        "${build_dir}/fixtures/${case}" )
    mkdir -p "fixtures/${case}/deck"
    cp "${deck}"/* "fixtures/${case}/deck/"
done

# --- receipts -------------------------------------------------------------------
{
    echo "gfortran: $(gfortran --version | head -1)"
    echo "glibc: $(ldd --version | head -1)"
    echo "host: $(uname -srm)"
    echo "flags: -O0 ${ff[*]} (default REAL: real(4))"
} > fixtures/TOOLCHAIN.txt
cp libmvec-report.txt fixtures/
( cd "${source_root}" && grep -v '^#' "${script_dir}/SOURCES.sha256" ) \
    > fixtures/SOURCES.sha256
( cd "${script_dir}" && sha256sum build.sh fetch_sources.sh extract.py \
    make_deck.py stubs.F90 wf_fitch.F90 run_windfarm_layout.F90 \
    ../urban_wrf471_oracle/oracle_io.F90 ) > fixtures/HARNESS.sha256
( cd fixtures && find . -type f ! -name 'FIXTURES.sha256' \
    | LC_ALL=C sort | xargs sha256sum ) > fixtures/FIXTURES.sha256
echo "fixtures in ${build_dir}/fixtures"
