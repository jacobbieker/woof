#!/usr/bin/env bash
# Line coverage of the byte-unmodified module_shcu_deng.F by an oracle input:
# the same build as build.sh's O0 with --coverage, then gcov.  Writes
# BUILD_DIR/cov/module_shcu_deng.F.gcov and prints the never-executed
# executable lines of deng_shcu_driver/deng_shcu and the helpers it calls.
#
#   bash coverage.sh WRF_SOURCE_ROOT BUILD_DIR INPUT.bin
set -euo pipefail
source_root=$(cd "$1" && pwd)
mkdir -p "$2/cov"
build_dir=$(cd "$2" && pwd)
input=$(cd "$(dirname "$3")" && pwd)/$(basename "$3")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
kernels_dir=$(cd "${script_dir}/../../woof/core/kernels" && pwd)
cd "${build_dir}/cov"
python3 "${script_dir}/gen_camuw_stub.py" "${source_root}" camuw_stub.F
fl="-O0 -ffree-form -ffree-line-length-none -ffp-contract=off -cpp -DEM_CORE=1 -DWRF_CHEM=0"
c++ -std=c++17 -O0 -ffp-contract=off -fno-builtin -I "${kernels_dir}" \
    -c "${script_dir}/libm_shim.cpp" -o libm_shim.o
cp "${source_root}/phys/module_shcu_deng.F" .
gfortran -c ${fl} "${script_dir}/stub_wrf.F90"
gfortran -c ${fl} "${source_root}/share/module_model_constants.F"
gfortran -c ${fl} camuw_stub.F
gfortran -c ${fl} "${source_root}/phys/module_shcu_grims.F"
gfortran -c ${fl} "${source_root}/phys/module_shcu_nscv.F"
gfortran -c ${fl} --coverage module_shcu_deng.F
gfortran -c ${fl} "${source_root}/phys/module_shallowcu_driver.F"
gfortran -c ${fl} "${script_dir}/run_deng_oracle.F90"
gfortran --coverage -o run_cov run_deng_oracle.o module_shallowcu_driver.o \
    module_shcu_deng.o module_shcu_nscv.o module_shcu_grims.o camuw_stub.o \
    module_model_constants.o stub_wrf.o libm_shim.o
rm -f module_shcu_deng.gcda
./run_cov "${input}" cov.out
gcov module_shcu_deng.F > /dev/null
python3 - <<'EOF'
import re
lines = open("module_shcu_deng.F.gcov").read().splitlines()
miss = []
for l in lines:
    m = re.match(r"\s*([^:]+):\s*(\d+):(.*)", l)
    if not m:
        continue
    cnt, no, src = m.group(1).strip(), int(m.group(2)), m.group(3)
    if cnt.startswith("#####") and no < 4430:
        miss.append((no, src.strip()))
execd = sum(1 for l in lines if re.match(r"\s*\d+\*?:\s*\d+:", l))
print(f"executed lines {execd}, never-executed (lines < 4430) {len(miss)}")
for no, src in miss:
    print(f"{no:5d}: {src}")
EOF
