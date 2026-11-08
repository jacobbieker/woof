#!/usr/bin/env bash
# The MICM CPU oracle WOOF's MICM port (woof/chem_micm.py) is graded against.
#
# Builds MUSICA v0.17.2 (NCAR, Apache-2.0; it pins MICM e15898ce and MechanismConfiguration) from its
# tagged source with only MICM enabled -- MIEM, MIAM, TUV-x and CARMA off, so the unlicensed MIEM is
# never fetched -- and links harness.cpp, a one-grid-cell RosenbrockStandardOrder driver.  Nothing it
# builds is committed.  Build on an x86-64 glibc host: the committed fixtures are glibc words, and a
# libm that is not glibc's (macOS) gives other words for exp/pow/log10.
#
#   tools/micm_oracle/build.sh <work-dir>      # leaves <work-dir>/harness
#   pixi run -e default python tools/micm_oracle/generate.py <work-dir>/harness
set -euo pipefail
work=$(realpath -m "${1:?usage: build.sh <work-dir>}")
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$work"
cd "$work"
[ -d musica ] || git clone --quiet --depth 1 --branch v0.17.2 https://github.com/NCAR/musica.git musica
test "$(git -C musica rev-parse HEAD)" = 875ef7337cef3cc67dddf9e7d0787c5fd49e3222
cmake -S musica -B build -DCMAKE_BUILD_TYPE=Release -DMUSICA_ENABLE_TESTS=OFF -DMUSICA_ENABLE_TUVX=OFF \
  -DMUSICA_ENABLE_CARMA=OFF -DMUSICA_ENABLE_MIEM=OFF -DMUSICA_ENABLE_MIAM=OFF \
  -DMUSICA_BUILD_FORTRAN_INTERFACE=OFF -DMUSICA_ENABLE_INSTALL=OFF > cmake.log
make -C build -j4 > make.log
B=build
g++ -std=c++23 -O2 -I musica/include -I $B/_deps/micm-src/include \
  -I $B/_deps/mechanism_configuration-src/include -I $B/_deps/yaml-cpp-src/include \
  "$here/harness.cpp" -o harness $B/lib/libmusica.a $B/lib/libmechanism_configuration.a \
  $B/_deps/yaml-cpp-build/libyaml-cpp.a -lfmt
if objdump -d $B/lib/libmusica.a | grep -qE 'vfn?m(add|sub)'; then
  echo "the oracle library contracts multiply-adds; the fixtures assume none" >&2; exit 1
fi
echo "$work/harness"
