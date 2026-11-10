#!/usr/bin/env bash
# The TUV-x CPU oracle WOOF's TUV-x photolysis port is graded against.
#
# Builds NCAR TUV-x v0.17.0 (commit ebec1cb4, Apache-2.0) from its tagged source and links
# harness.F90, a host-column driver (TUV-x's from-host height grid and temperature, air, O2 and O3
# profiles) over photolysis.json, the TUV 5.4 configuration cut to the photolysis reactions WOOF's
# MICM mechanisms use.  TUV-x's src/util/assert.F90 carries GPL-2.0-or-later portions: it is
# deleted before the build and replaced by musica_assert.F90, a clean-room module written from the
# call sites.  Nothing it builds or fetches is committed.  Build on an x86-64 glibc host in the
# mpas-oracle pixi environment (gfortran, netcdf-fortran, cmake); the committed fixtures are glibc
# words, and the build refuses a library that contracts multiply-adds.
#
#   pixi run -e mpas-oracle tools/tuvx_oracle/build.sh <work-dir>   # leaves <work-dir>/harness
#   pixi run -e default python tools/tuvx_oracle/generate.py <work-dir>
set -euo pipefail
work=$(realpath -m "${1:?usage: build.sh <work-dir>}")
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$work"
cd "$work"
[ -d tuv-x ] || git clone --quiet https://github.com/NCAR/tuv-x.git tuv-x
git -C tuv-x checkout --quiet ebec1cb47084fd102e338fb896c6d2b80fbcd292
test "$(git -C tuv-x rev-parse HEAD)" = ebec1cb47084fd102e338fb896c6d2b80fbcd292
# The GPL file never takes part: the clean-room module takes its path in the source list.
cp "$here/musica_assert.F90" tuv-x/src/util/assert.F90
if grep -rl "General Public License" tuv-x/src tuv-x/include >/dev/null; then
  echo "a GPL source remains in the TUV-x tree" >&2; exit 1
fi
cp "$here/photolysis.json" tuv-x/woof_photolysis.json
export PKG_CONFIG_PATH="${CONDA_PREFIX:?run inside the mpas-oracle pixi environment}/lib/pkgconfig"
# The environment carries no C++ compiler: TUV-x's config.cpp and yaml-cpp build with the host's,
# so the link takes the host sysroot (whose libstdc++ needs the host glibc).
cmake -S tuv-x -B build -DCMAKE_BUILD_TYPE=Release -DTUVX_ENABLE_TESTS=OFF \
  -DTUVX_ENABLE_REGRESSION_TESTS=OFF -DTUVX_BUILD_CLI=OFF -DCMAKE_Fortran_FLAGS= \
  -DCMAKE_Fortran_FLAGS_RELEASE="-O2 -ffp-contract=off" -DCMAKE_EXE_LINKER_FLAGS=--sysroot=/ > cmake.log
make -C build -j4 > make.log
gcc_lib=$(dirname "$(/usr/bin/g++ -print-file-name=libstdc++.so)")
gfortran --sysroot=/ -O2 -ffp-contract=off -I build/include "$here/harness.F90" -o harness \
  build/lib/libtuvx.a build/_deps/yaml-cpp-build/libyaml-cpp.a -L"$CONDA_PREFIX/lib" -lnetcdff -lnetcdf \
  -L"$gcc_lib" -lstdc++ -Wl,-rpath,"$CONDA_PREFIX/lib"
if objdump -d harness | grep -qE 'vfn?m(add|sub)'; then
  echo "the oracle contracts multiply-adds; the fixtures assume none" >&2; exit 1
fi
echo "$work/harness"
