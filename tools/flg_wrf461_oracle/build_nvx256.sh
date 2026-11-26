#!/usr/bin/env bash
# Patched-nvx FLG oracle: WRF 4.6.1 module_ra_flg.F with only PARA_FILE's
# nvx raised from 100 to 256 (nvx256.patch), for columns of 101 to 256
# layers. build.sh refuses an edited FLG source; this script never edits the
# WRF tree: it copies the pristine file, applies the recorded patch to the
# copy and refuses any other source or patch bytes.
set -euo pipefail
[[ $# == 2 ]] || { echo 'usage: build_nvx256.sh WRF_SOURCE_ROOT BUILD_DIR' >&2; exit 2; }
wrf=$(cd "$1" && pwd); mkdir -p "$2"; out=$(cd "$2" && pwd); here=$(cd "$(dirname "$0")" && pwd)
pin=d66e442fccc04111067e29274c9f9eaccc3cef28
pristine=c4977cdf2ddf3f2279c9be7e1fc659ef23bbb37376e5cf11f9a98a73d33b0266
patch_sha=b1e28863f9d089db2574252f268a2b8c507c8a8f8df3550b0d00117d2ca0d788
patched=a6b183cbc1ba103ff585ea9a5f80291c1cdfafb0879b6644b3613f00263e5a48
sha() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }
[[ $(git -C "$wrf" rev-parse HEAD) == "$pin" ]] || { echo 'wrong WRF revision' >&2; exit 3; }
[[ $(sha "$wrf/phys/module_ra_flg.F") == "$pristine" ]] || { echo 'edited FLG source' >&2; exit 3; }
[[ $(sha "$here/nvx256.patch") == "$patch_sha" ]] || { echo 'nvx256.patch differs from the recorded patch' >&2; exit 3; }
mkdir -p "$out/pristine/phys" "$out/nvx256/phys"
cp "$wrf/phys/module_ra_flg.F" "$out/pristine/phys/module_ra_flg.F"
cp "$wrf/phys/module_ra_flg.F" "$out/nvx256/phys/module_ra_flg.F"
patch -s -p1 -d "$out/nvx256" < "$here/nvx256.patch"
[[ $(sha "$out/nvx256/phys/module_ra_flg.F") == "$patched" ]] || { echo 'patched FLG source differs' >&2; exit 3; }
flags=(-O0 -fPIC -cpp -ffree-form -ffree-line-length-none -fno-tree-vectorize -ffp-contract=off)
for build in pristine nvx256; do
    mkdir -p "$out/$build/obj"
    (cd "$out/$build/obj"
     gfortran "${flags[@]}" -c "$out/$build/phys/module_ra_flg.F" -o module_ra_flg.o
     gfortran "${flags[@]}" -c "$here/stub_wrf_error.F90" -o stub_wrf_error.o
     gfortran -shared -o "$out/libflg_oracle_$build.so" module_ra_flg.o stub_wrf_error.o)
done
printf '%s\n' "gfortran ${flags[*]}" > "$out/compiler-flags.txt"
gfortran --version | sed -n 1p > "$out/compiler.txt"
if ldd --version >/dev/null 2>&1; then ldd --version | sed -n 1p; else uname -sr; fi > "$out/libc.txt"
{ for f in "$out/pristine/phys/module_ra_flg.F" "$out/nvx256/phys/module_ra_flg.F" "$here/nvx256.patch" \
           "$here/build_nvx256.sh" "$out/libflg_oracle_pristine.so" "$out/libflg_oracle_nvx256.so"; do
      printf '%s  %s\n' "$(sha "$f")" "${f#"$out/"}"; done; } > "$out/sha256sums.txt"
echo "patched-nvx FLG oracle built in $out"
