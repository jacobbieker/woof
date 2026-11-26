#!/bin/bash
# build481.sh STEP -- WRF 4.8.1 pristine oracle builds. Run inside the wrf-oracle:u2404 image with the fortran work dir mounted at /f;
# src/WRF-4.8.1 and src/WRF-4.8.1-scalar are copies (no .git) of a recursive clone of the tag. Verify each with
# python -m tools.wrf481_source.manifest check --tree TREE before building.
# Source: tag v4.8.1 commit e209f84a1a811424990a8f2f373a0b77c0d35f0d + submodule gitlinks (recursive clone);
# the release asset has no submodules. Option 32 = GNU serial.
set -ex
F=/f; S=$F/src; J=${J:-4}
export HOME=/tmp NETCDF=/usr JASPERLIB=$F/build/jasper/lib JASPERINC=$F/build/jasper/include
case $1 in
wrf481)
  cd $S/WRF-4.8.1; ./clean -a >/dev/null 2>&1 || true
  printf "32\n1\n" | ./configure
  grep -E "^(DMPARALLEL|FCOPTIM|SFC)" configure.wrf
  ./compile -j $J em_real > $F/build/wrf481-compile.log 2>&1 || true
  ls -la main/*.exe ;;
scalar481)
  cd $S/WRF-4.8.1-scalar; ./clean -a >/dev/null 2>&1 || true
  printf "32\n1\n" | ./configure
  sed -i "s/^FCOPTIM .*/FCOPTIM         =       -O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize/" configure.wrf
  grep -E "^(DMPARALLEL|FCOPTIM|SFC)" configure.wrf
  ./compile -j $J em_real > $F/build/wrf481-scalar-compile.log 2>&1 || true
  ls -la main/*.exe ;;
esac
