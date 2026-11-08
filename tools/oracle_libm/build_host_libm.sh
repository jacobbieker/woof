#!/usr/bin/env bash
# Print the objects an MPAS oracle links for glibc's libm words, building them
# into the current directory when the host libm is not glibc.
#
#   libm_objects=($(bash tools/oracle_libm/build_host_libm.sh))
#
# On a glibc host nothing is built and nothing printed: gfortran's calls reach
# glibc itself.  Elsewhere glibc_host.cpp (the repository's MIT glibc
# transcriptions) is compiled by CXX (default c++) and its object printed.
# The second line on stderr says which libm the oracle uses, for TOOLCHAIN.txt.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "${here}/../.." && pwd)
if getconf GNU_LIBC_VERSION >/dev/null 2>&1; then
    echo "libm: host glibc ($(getconf GNU_LIBC_VERSION))" >&2
    exit 0
fi
read -r -a cxx <<< "${CXX:-c++}"
"${cxx[@]}" -c -O0 -ffp-contract=off -std=c++17 -fno-exceptions -fno-rtti \
    -I"${root}/woof/backend/host" -I"${root}/woof/core/kernels" "${here}/glibc_host.cpp" -o glibc_host.o
echo "libm: tools/oracle_libm/glibc_host.cpp (woof/core/kernels glibc transcriptions; host libm is not glibc)" >&2
echo "glibc_host.o"
