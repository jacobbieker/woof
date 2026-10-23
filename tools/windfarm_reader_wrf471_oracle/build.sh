#!/usr/bin/env bash
# Build and run WRF v4.7.1's wind-farm input reads on the oracle decks.
#
#   bash tools/windfarm_reader_wrf471_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR
#
# WRF_SOURCE_ROOT holds the files SOURCES.sha256 pins (fetch_sources.sh lays
# them out); every one is checked before anything compiles.  What is
# compiled, byte-unmodified, as INCLUDE files in run_reader.F90:
#
#   phys/module_wind_fitch.F:369-375   the turbine-list record count
#   phys/module_wind_fitch.F:393       the windfarm_ij = 1 row read
#   phys/module_wind_fitch.F:399       the windfarm_ij = 0 row read
#   phys/module_wind_fitch.F:491-510   the wind-turbine-N.tbl reads
#
# plus ../urban_wrf471_oracle/oracle_io.F90 (the fixture writer).  The decks
# are written by make_deck.py; the fixtures land in BUILD_DIR/fixtures (one
# directory per deck, the decks themselves under fixtures/decks).  Copy that
# directory to tests/data/oracles/windfarm_reader/ to publish it.
#
# REAL is real(4), WRF's RWORDSIZE=4 default.  -O0, and the libmvec guard of
# the other wind-farm oracles: no object may reference a _ZGV* symbol (the
# reads call no libm at all; the guard proves it).
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
rm -rf fixtures *.o *.mod *.inc run_reader
python3 "${script_dir}/extract.py" "${source_root}" "${build_dir}"

ff=(-ffree-form -ffree-line-length-none)
gfortran -c -O0 "${ff[@]}" "${script_dir}/../urban_wrf471_oracle/oracle_io.F90"
gfortran -c -O0 "${ff[@]}" -I"${build_dir}" "${script_dir}/run_reader.F90"
for object in oracle_io.o run_reader.o; do
    if nm -u "${object}" | grep -q '_ZGV'; then
        echo "-O0 object ${object} pulled in a libmvec vector symbol" >&2
        exit 4
    fi
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
{
    echo "# run_reader.o -O0 undefined libm symbols (none expected):"
    nm -u run_reader.o | grep -E 'powf|expf|logf|strtof|_ZGV' || true
    echo "# positive control (plain expf loop at -Ofast):"
    nm -u libmvec_positive_control.o | grep -E 'expf|_ZGV'
} > libmvec-report.txt
gfortran -o run_reader run_reader.o oracle_io.o

mkdir -p fixtures/decks
python3 "${script_dir}/make_deck.py" fixtures/decks
for deck in ij latlon; do
    mode=0
    [[ "${deck}" == ij ]] && mode=1
    ( cd "fixtures/decks/${deck}" && "${build_dir}/run_reader" \
        "${build_dir}/fixtures" "${deck}" "${mode}" )
done

{
    echo "gfortran: $(gfortran --version | head -1)"
    echo "glibc: $(ldd --version | head -1)"
    echo "host: $(uname -srm)"
} > fixtures/TOOLCHAIN.txt
cp libmvec-report.txt fixtures/
( cd "${source_root}" && grep -v '^#' "${script_dir}/SOURCES.sha256" ) \
    > fixtures/SOURCES.sha256
( cd "${script_dir}" && sha256sum build.sh fetch_sources.sh extract.py \
    make_deck.py run_reader.F90 ../urban_wrf471_oracle/oracle_io.F90 ) \
    > fixtures/HARNESS.sha256
( cd fixtures && find . -type f \( -name '*.bin' -o -name 'MANIFEST.txt' \
    -o -name '*.txt' -path './decks/*' -o -name '*.tbl' \) \
    | LC_ALL=C sort | xargs sha256sum ) > fixtures/FIXTURES.sha256
echo "fixtures in ${build_dir}/fixtures"
