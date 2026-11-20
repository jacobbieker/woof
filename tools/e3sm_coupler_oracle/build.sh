#!/usr/bin/env bash
# Build the E3SM v3.2.0 coupler oracle: the ocean merge, prep_ocn_merge.
#
#   bash tools/e3sm_coupler_oracle/build.sh E3SM_SOURCE_ROOT MCT_SOURCE_ROOT BUILD_DIR
#
# E3SM_SOURCE_ROOT is an E3SM tree at tag v3.2.0
# (648b167b7f95267a159a9fdee25faf604c2bfc4f); MCT_SOURCE_ROOT is MCT at the
# commit that tag pins as externals/mct (82b0071e69d14330b75d23b0bc68543ebea9aadc).
# Both are checked against SOURCES.sha256 before anything is built.
#
# - MCT is built as it is (configure --enable-mpiserial, its own mpi-serial),
#   with the same gfortran, at -O0 -ffp-contract=off.  It only moves data:
#   the attribute vectors, their field lists, the shared-index copy and the
#   zeroing.
# - prep_ocn_merge is cut byte for byte out of driver-mct/main/prep_ocn_mod.F90
#   into a module (prep_ocn_merge_cut) whose head names the modules it uses.
#   shr_kind_mod is compiled verbatim.  mct_mod is written from the pinned
#   share/util/mct_mod.F90's own `use m_AttrVect ... =>` renames and its
#   mct_usevector flag; the rest of that file pulls in the MPI share layer.
#   e3sm_standins.F90 holds the rest (no arithmetic).
# - Compiled with gfortran -O0 -ffp-contract=off; the merge is real(r8).  The
#   objects must contain no FMA instruction (checked with objdump when it is
#   present).
#
# gfortran (and the C compiler MCT's mpi-serial needs) are the mpas-oracle
# pixi environment's unless GFORTRAN / CC name others.
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo "usage: build.sh E3SM_SOURCE_ROOT MCT_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi
e3sm=$(cd "$1" && pwd)
mct=$(cd "$2" && pwd)
mkdir -p "$3"
build_dir=$(cd "$3" && pwd)
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
read -r -a fc <<< "${GFORTRAN:-gfortran}"
cc="${CC:-gcc}"

check() {
    ( cd "$1" && grep "^[0-9a-f]\{64\}  $2/" "${script_dir}/SOURCES.sha256" | sed "s#  $2/#  #" \
        | { command -v sha256sum >/dev/null && sha256sum -c --quiet - || shasum -a 256 -c --quiet -; } ) || {
        echo "a $2 source differs from SOURCES.sha256; refusing to build" >&2
        exit 3
    }
}
check "${e3sm}" e3sm
check "${mct}" mct

# MCT, out of tree: a copy of the pinned sources, configured and built here
if [[ ! -f "${build_dir}/mct/mct/libmct.a" ]]; then
    rm -rf "${build_dir}/mct" && cp -R "${mct}" "${build_dir}/mct"
    ( cd "${build_dir}/mct" \
        && ./configure --enable-mpiserial FC="${fc[*]}" CC="${cc}" FCFLAGS="-O0 -ffp-contract=off" \
               CFLAGS="-O0 -std=gnu89" > configure.log 2>&1 \
        && make > make.log 2>&1 ) || { echo "MCT did not build; see ${build_dir}/mct/*.log" >&2; exit 4; }
fi
mctinc=(-I"${build_dir}/mct/mct" -I"${build_dir}/mct/mpeu" -I"${build_dir}/mct/mpi-serial")
mctlib=("${build_dir}/mct/mct/libmct.a" "${build_dir}/mct/mpeu/libmpeu.a" "${build_dir}/mct/mpi-serial/libmpi-serial.a")

obj="${build_dir}/merge"
rm -rf "${obj}" && mkdir -p "${obj}" && cd "${obj}"
# mct_mod: the pinned wrapper's own renames of MCT's attribute-vector routines
{
    printf 'module mct_mod\n'
    grep -E '^ *use m_AttrVect +,only:' "${e3sm}/share/util/mct_mod.F90"
    printf '  implicit none\n'
    grep -E '^ *logical,public :: mct_usevector = \.false\.$' "${e3sm}/share/util/mct_mod.F90"
    printf 'end module mct_mod\n'
} > mct_mod.F90
# prep_ocn_merge, byte for byte, inside a module of its own
{
    printf 'module prep_ocn_merge_cut\n'
    printf '  use shr_kind_mod, only: r8 => SHR_KIND_R8, cs => SHR_KIND_CS, cl => SHR_KIND_CL\n'
    printf '  use shr_sys_mod, only: shr_sys_abort, shr_sys_flush\n'
    printf '  use seq_comm_mct, only: CPLID, logunit, seq_comm_setptrs\n'
    printf '  use seq_flds_mod\n  use mct_mod\n  implicit none\n  private\n  public :: prep_ocn_merge\ncontains\n'
    awk '/^  subroutine prep_ocn_merge\(/{on=1} on{print} /^  end subroutine prep_ocn_merge$/{on=0}' \
        "${e3sm}/driver-mct/main/prep_ocn_mod.F90"
    printf 'end module prep_ocn_merge_cut\n'
} > prep_ocn_merge_cut.F90
flags=(-c -O0 -ffp-contract=off -ffree-line-length-none "${mctinc[@]}")
"${fc[@]}" "${flags[@]}" "${e3sm}/share/util/shr_kind_mod.F90"
"${fc[@]}" "${flags[@]}" "${script_dir}/e3sm_standins.F90"
"${fc[@]}" "${flags[@]}" mct_mod.F90
"${fc[@]}" "${flags[@]}" prep_ocn_merge_cut.F90
"${fc[@]}" "${flags[@]}" "${script_dir}/run_ocean_merge.F90"
"${fc[@]}" -o "${build_dir}/run_ocean_merge" run_ocean_merge.o prep_ocn_merge_cut.o mct_mod.o e3sm_standins.o \
    shr_kind_mod.o "${mctlib[@]}"
if command -v objdump >/dev/null && objdump -d ./*.o | grep -qiE '\bv?fn?m(add|sub)[0-9]'; then
    echo "an object contains an FMA instruction" >&2
    exit 5
fi
echo "built ${build_dir}/run_ocean_merge"
