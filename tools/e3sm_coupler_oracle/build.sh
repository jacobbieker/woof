#!/usr/bin/env bash
# Build the E3SM v3.2.0 coupler oracles: the ocean merge, prep_ocn_merge
# (run_ocean_merge), the sea ice's, prep_ice_merge (run_ice_merge), MPAS-Ocean's coupler import and export,
# ocn_import_mct and ocn_export_mct (run_ocean_coupling), and, given Icepack,
# MPAS-Seaice's, ice_import_mct and ice_export_mct (run_seaice_coupling),
# and the coupler's surface fractions, seq_frac_set (run_seq_frac).
#
#   bash tools/e3sm_coupler_oracle/build.sh E3SM_SOURCE_ROOT MCT_SOURCE_ROOT BUILD_DIR [ICEPACK_SOURCE_ROOT]
#
# ICEPACK_SOURCE_ROOT is Icepack at the commit v3.2.0 pins as
# components/mpas-seaice/src/icepack (cf1d460fbb6a13d28988f798b33734f46d70a9dc).
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
if [[ $# -ne 3 && $# -ne 4 ]]; then
    echo "usage: build.sh E3SM_SOURCE_ROOT MCT_SOURCE_ROOT BUILD_DIR [ICEPACK_SOURCE_ROOT]" >&2
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
icepack=""
if [[ $# -eq 4 ]]; then
    icepack=$(cd "$4" && pwd)
    check "${icepack}" icepack
fi

# MCT, out of tree: a copy of the pinned sources, configured and built here;
# rebuilt whenever the compilers, the flags or the sources' digests change
stamp="FC=${fc[*]} $("${fc[@]}" --version | head -1) CC=${cc} $("${cc}" --version | head -1)"
stamp+=" FCFLAGS=-O0 -ffp-contract=off CFLAGS=-O0 -std=gnu89 $(grep '  mct/' "${script_dir}/SOURCES.sha256" | cksum)"
if [[ ! -f "${build_dir}/mct/mct/libmct.a" || "$(cat "${build_dir}/mct.stamp" 2>/dev/null)" != "${stamp}" ]]; then
    rm -rf "${build_dir}/mct" "${build_dir}/mct.stamp" && cp -R "${mct}" "${build_dir}/mct"
    ( cd "${build_dir}/mct" && rm -f Makefile.conf config.h config.status config.log \
        && ./configure --enable-mpiserial FC="${fc[*]}" CC="${cc}" FCFLAGS="-O0 -ffp-contract=off" \
               CFLAGS="-O0 -std=gnu89" > configure.log 2>&1 \
        && make > make.log 2>&1 ) || { echo "MCT did not build; see ${build_dir}/mct/*.log" >&2; exit 4; }
    printf '%s' "${stamp}" > "${build_dir}/mct.stamp"
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
if [[ $(grep -c '^  subroutine prep_ocn_merge(' prep_ocn_merge_cut.F90) -ne 1 \
      || $(grep -c '^  end subroutine prep_ocn_merge$' prep_ocn_merge_cut.F90) -ne 1 ]]; then
    echo "prep_ocn_merge was not cut out of prep_ocn_mod.F90 whole" >&2
    exit 3
fi
flags=(-c -O0 -ffp-contract=off -ffree-line-length-none "${mctinc[@]}")
"${fc[@]}" "${flags[@]}" "${e3sm}/share/util/shr_kind_mod.F90"
"${fc[@]}" "${flags[@]}" "${script_dir}/e3sm_standins.F90"
"${fc[@]}" "${flags[@]}" mct_mod.F90
"${fc[@]}" "${flags[@]}" prep_ocn_merge_cut.F90
"${fc[@]}" "${flags[@]}" "${script_dir}/run_ocean_merge.F90"
"${fc[@]}" -o "${build_dir}/run_ocean_merge" run_ocean_merge.o prep_ocn_merge_cut.o mct_mod.o e3sm_standins.o \
    shr_kind_mod.o "${mctlib[@]}"
# the merge's own objects: no fused multiply-add (x86 vfmadd..., aarch64
# fmadd/fmsub/fnmadd/fnmsub) and no libm call (MCT's routines used here only
# zero and copy)
if command -v objdump >/dev/null; then
    objdump -d prep_ocn_merge_cut.o shr_kind_mod.o e3sm_standins.o run_ocean_merge.o > disassembly.txt
    if grep -iE '\s(v?fn?m(add|sub)[0-9a-z]*|fn?m(add|sub))\s' disassembly.txt > fma.txt; then
        echo "an object contains an FMA instruction (${obj}/fma.txt)" >&2
        exit 5
    fi
fi
nm -u prep_ocn_merge_cut.o > undefined.txt
if grep -E '\b(exp|log|pow|sqrt|sin|cos|tan|atan|asin|acos|cbrt|erf)[fl]?\b' undefined.txt; then
    echo "prep_ocn_merge calls libm" >&2
    exit 6
fi
echo "built ${build_dir}/run_ocean_merge"

# prep_ice_merge, byte for byte, inside a module of its own, in the same
# objects' directory (mct_mod, shr_kind_mod and e3sm_standins as above)
{
    printf 'module prep_ice_merge_cut\n'
    printf '  use shr_kind_mod, only: r8 => SHR_KIND_R8, cs => SHR_KIND_CS, cl => SHR_KIND_CL\n'
    printf '  use shr_sys_mod, only: shr_sys_abort, shr_sys_flush\n'
    printf '  use seq_comm_mct, only: CPLID, logunit, seq_comm_getData => seq_comm_setptrs\n'
    printf '  use mct_mod\n  implicit none\n  private\n  public :: prep_ice_merge\ncontains\n'
    awk '/^  subroutine prep_ice_merge\(/{on=1} on{print} /^  end subroutine prep_ice_merge$/{on=0}' \
        "${e3sm}/driver-mct/main/prep_ice_mod.F90"
    printf 'end module prep_ice_merge_cut\n'
} > prep_ice_merge_cut.F90
if [[ $(grep -c '^  subroutine prep_ice_merge(' prep_ice_merge_cut.F90) -ne 1 \
      || $(grep -c '^  end subroutine prep_ice_merge$' prep_ice_merge_cut.F90) -ne 1 ]]; then
    echo "prep_ice_merge was not cut out of prep_ice_mod.F90 whole" >&2
    exit 3
fi
"${fc[@]}" "${flags[@]}" prep_ice_merge_cut.F90
"${fc[@]}" "${flags[@]}" "${script_dir}/run_ice_merge.F90"
"${fc[@]}" -o "${build_dir}/run_ice_merge" run_ice_merge.o prep_ice_merge_cut.o mct_mod.o e3sm_standins.o \
    shr_kind_mod.o "${mctlib[@]}"
if command -v objdump >/dev/null; then
    objdump -d prep_ice_merge_cut.o run_ice_merge.o > disassembly_ice.txt
    if grep -iE '\s(v?fn?m(add|sub)[0-9a-z]*|fn?m(add|sub))\s' disassembly_ice.txt > fma_ice.txt; then
        echo "an object contains an FMA instruction (${obj}/fma_ice.txt)" >&2
        exit 5
    fi
fi
nm -u prep_ice_merge_cut.o > undefined_ice.txt
if grep -E '\b(exp|log|pow|sqrt|sin|cos|tan|atan|asin|acos|cbrt|erf)[fl]?\b' undefined_ice.txt; then
    echo "prep_ice_merge calls libm" >&2
    exit 6
fi
echo "built ${build_dir}/run_ice_merge"

# ---------------------------------------------------------------- the ocean's import and export
# ocn_import_mct and ocn_export_mct are cut byte for byte out of
# components/mpas-ocean/driver/ocn_comp_mct.F into a module
# (ocn_comp_mct_cut) whose head names the modules they use and carries the
# host module's own declarations of the variables they read (lines 107-109,
# 119, 125-127 and 129 of the pinned file).  ocn_freezing_temperature is cut
# the same way out of mpas_ocn_equation_of_state.F.  Compiled verbatim:
# shr_kind_mod, shr_const_mod, glc_zocnclass_mod (with NDEBUG), MPAS's
# mpas_kind_types and mpas_constants, mpaso_cpl_indices.F and
# mpas_ocn_constants.F (with MPAS_ESM_SHR_CONST, as E3SM builds MPAS).  The
# MPAS framework is tools/mpas_ocean_oracle/stubs.F90 (pools by name, one
# block, halo exchanges the identity); ocean_standins.F90 and
# e3sm_standins.F90 hold the rest.  real(RKIND) is real(8), as E3SM runs.
obj="${build_dir}/ocean"
rm -rf "${obj}" && mkdir -p "${obj}" && cd "${obj}"
src="${e3sm}/components/mpas-ocean"
oflags=(-c -cpp -O0 -ffp-contract=off -ffree-line-length-none -DNDEBUG -I"${e3sm}/share/include" "${mctinc[@]}")
mpas=(-ffree-form)
cp "${obj}/../merge/mct_mod.F90" mct_mod.F90
{
    printf 'module ocn_equation_of_state\n  use mpas_kind_types\n  use ocn_config\n  implicit none\n  private\n'
    printf '  public :: ocn_freezing_temperature\ncontains\n'
    awk '/^    real \(kind=RKIND\) function ocn_freezing_temperature\(/{on=1} on{print} /^    end function ocn_freezing_temperature!/{on=0}' \
        "${src}/src/shared/mpas_ocn_equation_of_state.F"
    printf 'end module ocn_equation_of_state\n'
} > ocn_freezing_cut.F90
host=$(sed -n '107,109p;119p;125,127p;129p' "${src}/driver/ocn_comp_mct.F")
{
    printf 'module ocn_comp_mct_cut\n'
    printf '  use seq_flds_mod\n  use seq_infodata_mod\n  use mpaso_cpl_indices\n  use mpas_derived_types\n'
    printf '  use mpas_pool_routines\n  use mpas_kind_types\n  use mpas_dmpar\n  use mpas_constants\n'
    printf '  use ocn_constants\n  use ocn_config\n  use ocn_equation_of_state\n  use ocn_diagnostics_variables\n'
    printf '  implicit none\n  public\n'
    printf '%s\n' "${host}"
    printf 'contains\n'
    awk '/^ subroutine ocn_import_mct\(/{on=1} on{print} /^ end subroutine ocn_import_mct!/{on=0}' "${src}/driver/ocn_comp_mct.F"
    awk '/^ subroutine ocn_export_mct\(/{on=1} on{print} /^ end subroutine ocn_export_mct!/{on=0}' "${src}/driver/ocn_comp_mct.F"
    printf 'end module ocn_comp_mct_cut\n'
} > ocn_comp_mct_cut.F90
for want in 'subroutine ocn_import_mct(' 'end subroutine ocn_import_mct!' 'subroutine ocn_export_mct(' \
            'end subroutine ocn_export_mct!' 'function ocn_freezing_temperature(' 'end function ocn_freezing_temperature!'; do
    if [[ $(cat ocn_comp_mct_cut.F90 ocn_freezing_cut.F90 | grep -cF "${want}") -ne 1 ]]; then
        echo "a routine was not cut out whole (${want})" >&2
        exit 3
    fi
done
if [[ $(printf '%s\n' "${host}" | grep -cE 'saved_accumulatedFrazilIceMass\(:\)$|saved_frazil_allocated_size = 0$|saved_frazil_initialized = \.false\.$|pointer :: infodata$|pointer:: domain$|itimestep, &|ocn_cpl_dt +!|:: precFactor $') -ne 8 ]]; then
    echo "ocn_comp_mct.F's module declarations moved" >&2
    exit 3
fi
"${fc[@]}" "${oflags[@]}" "${e3sm}/share/util/shr_kind_mod.F90"
"${fc[@]}" "${oflags[@]}" "${e3sm}/share/util/shr_const_mod.F90"
"${fc[@]}" "${oflags[@]}" "${mpas[@]}" "${e3sm}/components/mpas-framework/src/framework/mpas_kind_types.F"
"${fc[@]}" "${oflags[@]}" "${script_dir}/../mpas_ocean_oracle/stubs.F90"
"${fc[@]}" "${oflags[@]}" "${mpas[@]}" "${e3sm}/components/mpas-framework/src/framework/mpas_constants.F"
"${fc[@]}" "${oflags[@]}" "${script_dir}/e3sm_standins.F90"
"${fc[@]}" "${oflags[@]}" "${script_dir}/ocean_standins.F90"
"${fc[@]}" "${oflags[@]}" mct_mod.F90
"${fc[@]}" "${oflags[@]}" "${e3sm}/driver-mct/shr/glc_zocnclass_mod.F90"
"${fc[@]}" "${oflags[@]}" "${mpas[@]}" "${src}/driver/mpaso_cpl_indices.F"
"${fc[@]}" "${oflags[@]}" "${mpas[@]}" -DMPAS_ESM_SHR_CONST "${src}/src/shared/mpas_ocn_constants.F"
"${fc[@]}" "${oflags[@]}" ocn_freezing_cut.F90
"${fc[@]}" "${oflags[@]}" ocn_comp_mct_cut.F90
"${fc[@]}" "${oflags[@]}" "${script_dir}/run_ocean_coupling.F90"
"${fc[@]}" -o "${build_dir}/run_ocean_coupling" ./*.o "${mctlib[@]}"
if command -v objdump >/dev/null; then
    objdump -d ocn_comp_mct_cut.o ocn_freezing_cut.o mpas_ocn_constants.o run_ocean_coupling.o > disassembly.txt
    if grep -iE '\s(v?fn?m(add|sub)[0-9a-z]*|fn?m(add|sub))\s' disassembly.txt > fma.txt; then
        echo "an object contains an FMA instruction (${obj}/fma.txt)" >&2
        exit 5
    fi
fi
# the import's sqrt (the 10 m wind speed) is IEEE's correctly rounded square
# root, an instruction or glibc's; nothing else may reach libm
nm -u ocn_comp_mct_cut.o ocn_freezing_cut.o > undefined.txt
if grep -E '\b(exp|log|pow|sin|cos|tan|atan|asin|acos|cbrt|erf)[fl]?\b' undefined.txt; then
    echo "the ocean's import or export calls libm" >&2
    exit 6
fi
echo "built ${build_dir}/run_ocean_coupling"

# ---------------------------------------------------------------- the surface fractions
# seq_frac_set and seq_frac_check are cut byte for byte out of
# driver-mct/main/seq_frac_mct.F90 into a module of the same name whose head
# is the host module's own (lines 143-182 and 184-228: its use lines, its
# switches, eps_* and the mapper pointers; seq_frac_init's public line left
# out, as it is not cut) plus a harness setter for the private
# seq_frac_dead.  seq_map_map, seq_map_avNormAvF and seq_map_avNormArr are
# cut the same way out of seq_map_mod.F90 into a module of that name whose
# head is the host's use lines (14-17 and 19-23; shr_mct_mod's map-file
# readers are not reached) and its seq_map_avNorm interface (41-44).
# seq_map_type_mod is the pinned type seq_map, cut out of its file.
# mct_mod is written from the pinned share/util/mct_mod.F90's own `use m_*,
# only:` renames, its two flags and its mct_rearr_rearrange_fldlist (cut
# verbatim), over the pinned MCT and the share layer's stand-ins.
# Compiled verbatim: shr_kind_mod and shr_const_mod.  frac_standins_base.F90
# and frac_standins.F90 hold the rest (no arithmetic).
obj="${build_dir}/frac"
rm -rf "${obj}" && mkdir -p "${obj}" && cd "${obj}"
drv="${e3sm}/driver-mct/main"
cut() {  # cut FILE NAME KIND: a routine of that kind, from its first line to its end line
    awk -v r="$2" -v k="$3" '$0 ~ "^ *"k" "r"\\(" {on=1} on{print} $0 ~ "^ *end "k" "r"( |$)" {on=0}' "$1"
}
{
    printf 'module mct_mod\n'
    grep -E '^ *use (shr_mpi_mod|shr_sys_mod) ' "${e3sm}/share/util/mct_mod.F90"
    grep -E '^ *use m_[A-Za-z0-9_]+ +,only:' "${e3sm}/share/util/mct_mod.F90"
    printf '  use shr_kind_mod, only: IN => SHR_KIND_IN\n  implicit none\n'
    grep -E '^ *logical,public :: mct_use(alltoall|vector) = \.false\.$' "${e3sm}/share/util/mct_mod.F90"
    printf 'contains\n'
    cut "${e3sm}/share/util/mct_mod.F90" mct_rearr_rearrange_fldlist subroutine
    printf 'end module mct_mod\n'
} > mct_mod.F90
{
    printf 'module seq_map_type_mod\n'
    sed -n '/^ *use shr_kind_mod *, only: R8/p;/^ *use shr_kind_mod *, only: CL/p' "${drv}/seq_map_type_mod.F90"
    printf '  use mct_mod\n  implicit none\n'
    awk '/^ *type seq_map *$/{on=1} on{print} /^ *end type seq_map *$/{on=0}' "${drv}/seq_map_type_mod.F90"
    printf 'end module seq_map_type_mod\n'
} > seq_map_type_cut.F90
{
    printf 'module seq_map_mod\n'
    sed -n '14,17p;19,23p' "${drv}/seq_map_mod.F90"
    printf '  implicit none\n  save\n  private\n  public :: seq_map_map\n'
    sed -n '41,44p' "${drv}/seq_map_mod.F90"
    printf 'contains\n'
    for r in seq_map_map seq_map_avNormAvF seq_map_avNormArr; do cut "${drv}/seq_map_mod.F90" "${r}" subroutine; done
    printf 'end module seq_map_mod\n'
} > seq_map_cut.F90
{
    sed -n '143,182p;184,228p' "${drv}/seq_frac_mct.F90"
    printf '  public seq_frac_oracle_set_dead\n'
    sed -n '229p' "${drv}/seq_frac_mct.F90"
    for r in seq_frac_set seq_frac_check; do cut "${drv}/seq_frac_mct.F90" "${r}" subroutine; done
    printf '  ! the harness: seq_frac_init sets this from the coupler (dead models)\n'
    printf '  subroutine seq_frac_oracle_set_dead(dead)\n    logical, intent(in) :: dead\n'
    printf '    seq_frac_dead = dead\n  end subroutine seq_frac_oracle_set_dead\n'
    printf 'end module seq_frac_mct\n'
} > seq_frac_cut.F90
for want in 'subroutine mct_rearr_rearrange_fldlist:mct_mod.F90' 'type seq_map:seq_map_type_cut.F90' \
            'subroutine seq_map_map:seq_map_cut.F90' 'subroutine seq_map_avNormAvF:seq_map_cut.F90' \
            'subroutine seq_map_avNormArr:seq_map_cut.F90' 'subroutine seq_frac_set:seq_frac_cut.F90' \
            'subroutine seq_frac_check:seq_frac_cut.F90'; do
    if [[ $(grep -cE "^ *(end )?${want%%:*}( |\\(|$)" "${want##*:}") -ne 2 ]]; then
        echo "a routine was not cut out whole (${want%%:*})" >&2
        exit 3
    fi
done
if [[ "$(sed -n '143p;183p;229p' "${drv}/seq_frac_mct.F90" | tr -d ' ')" != $'moduleseq_frac_mct\npublicseq_frac_init\ncontains' \
      || "$(sed -n '18p;41p;44p' "${drv}/seq_map_mod.F90" | tr -s ' ')" != \
         $' use shr_mct_mod, only: shr_mct_sMatPInitnc, shr_mct_queryConfigFile\n interface seq_map_avNorm\n end interface seq_map_avNorm' ]]; then
    echo "seq_frac_mct.F90's or seq_map_mod.F90's module head moved" >&2
    exit 3
fi
fflags=(-c -cpp -O0 -ffp-contract=off -ffree-line-length-none "${mctinc[@]}")
"${fc[@]}" "${fflags[@]}" "${e3sm}/share/util/shr_kind_mod.F90"
"${fc[@]}" "${fflags[@]}" "${e3sm}/share/util/shr_const_mod.F90"
"${fc[@]}" "${fflags[@]}" "${script_dir}/frac_standins_base.F90"
"${fc[@]}" "${fflags[@]}" mct_mod.F90
"${fc[@]}" "${fflags[@]}" seq_map_type_cut.F90
"${fc[@]}" "${fflags[@]}" "${script_dir}/frac_standins.F90"
"${fc[@]}" "${fflags[@]}" seq_map_cut.F90
"${fc[@]}" "${fflags[@]}" seq_frac_cut.F90
"${fc[@]}" "${fflags[@]}" "${script_dir}/run_seq_frac.F90"
"${fc[@]}" -o "${build_dir}/run_seq_frac" ./*.o "${mctlib[@]}"
if command -v objdump >/dev/null; then
    objdump -d seq_frac_cut.o seq_map_cut.o mct_mod.o "${build_dir}/mct/mct/m_MatAttrVectMul.o" \
        "${build_dir}/mct/mct/m_Rearranger.o" > disassembly.txt
    if grep -iE '\s(v?fn?m(add|sub)[0-9a-z]*|fn?m(add|sub))\s' disassembly.txt > fma.txt; then
        echo "an object contains an FMA instruction (${obj}/fma.txt)" >&2
        exit 5
    fi
fi
nm -u seq_frac_cut.o seq_map_cut.o > undefined.txt
if grep -E '\b(exp|log|pow|sqrt|sin|cos|tan|atan|atan2|asin|acos|cbrt|erf)[fl]?\b' undefined.txt; then
    echo "seq_frac_set or seq_map_map calls libm" >&2
    exit 6
fi
echo "built ${build_dir}/run_seq_frac"

# ---------------------------------------------------------------- the sea ice's import and export
# ice_import_mct, ice_export_mct, basal_pressure and frazil_mass are cut byte
# for byte out of components/mpas-seaice/driver/ice_comp_mct.F into a module
# (ice_comp_mct_cut) whose head names the modules they use (the host
# module's own use lines for seaice_constants, seaice_mesh and icepack_intfc,
# lines 58-68, 75 and 78) and carries the host module's declaration of the
# domain (line 124).  seaice_latlon_vector_rotation_backward and the five
# routines it calls are cut the same way out of mpas_seaice_mesh.F into
# seaice_mesh.  Compiled verbatim: mpassi_cpl_indices.F,
# mpas_seaice_constants.F (with CCSMCOUPLED, as E3SM builds it, so its
# constants are shr_const_mod's) and Icepack's columnphysics (every module,
# as icepack_intfc uses them all).  The MPAS framework is
# tools/mpas_ocean_oracle/stubs.F90.
if [[ -z "${icepack}" ]]; then
    echo "no ICEPACK_SOURCE_ROOT: the sea ice's import and export oracle is not built"
    exit 0
fi
obj="${build_dir}/seaice"
rm -rf "${obj}" && mkdir -p "${obj}" && cd "${obj}"
src="${e3sm}/components/mpas-seaice"
cp "${obj}/../merge/mct_mod.F90" mct_mod.F90
{
    printf 'module seaice_mesh\n  use mpas_kind_types\n  implicit none\n  public\ncontains\n'
    for r in seaice_latlon_vector_rotation_backward seaice_grid_rotation_forward seaice_latlon_from_xyz \
             latlon_vector_to_xyz_vector xyz_vector_rotation_backward xyz_vector_to_latlon_vector; do
        awk -v r="${r}" '$0 ~ "^  subroutine "r"\\(" {on=1} on{print} $0 ~ "^  end subroutine "r"!" {on=0}' \
            "${src}/src/shared/mpas_seaice_mesh.F"
    done
    printf 'end module seaice_mesh\n'
} > seaice_mesh_cut.F90
host_use=$(sed -n '58,68p;75p;78p' "${src}/driver/ice_comp_mct.F")
host_decl=$(sed -n '124p' "${src}/driver/ice_comp_mct.F")
{
    printf 'module ice_comp_mct_cut\n'
    printf '  use seq_flds_mod\n  use mpassi_cpl_indices\n  use mpas_derived_types\n  use mpas_pool_routines\n'
    printf '  use mpas_kind_types\n  use mpas_dmpar\n'
    printf '%s\n' "${host_use}"
    printf '  implicit none\n  public\n'
    printf '%s\n' "${host_decl}"
    printf 'contains\n'
    awk '/^ subroutine ice_import_mct\(/{on=1} on{print} /^ end subroutine ice_import_mct!/{on=0}' "${src}/driver/ice_comp_mct.F"
    awk '/^ subroutine ice_export_mct\(/{on=1} on{print} /^ end subroutine ice_export_mct!/{on=0}' "${src}/driver/ice_comp_mct.F"
    awk '/^   subroutine basal_pressure\(/{on=1} on{print} /^   end subroutine basal_pressure!/{on=0}' "${src}/driver/ice_comp_mct.F"
    awk '/^   subroutine frazil_mass\(/{on=1} on{print} /^   end subroutine frazil_mass!/{on=0}' "${src}/driver/ice_comp_mct.F"
    printf 'end module ice_comp_mct_cut\n'
} > ice_comp_mct_cut.F90
for r in ice_import_mct ice_export_mct basal_pressure frazil_mass; do
    if [[ $(grep -cE "^ *subroutine ${r}\(" ice_comp_mct_cut.F90) -ne 1 \
          || $(grep -cE "^ *end subroutine ${r}!" ice_comp_mct_cut.F90) -ne 1 ]]; then
        echo "${r} was not cut out of ice_comp_mct.F whole" >&2
        exit 3
    fi
done
for r in seaice_latlon_vector_rotation_backward seaice_grid_rotation_forward seaice_latlon_from_xyz \
         latlon_vector_to_xyz_vector xyz_vector_rotation_backward xyz_vector_to_latlon_vector; do
    if [[ $(grep -cE "^  (end )?subroutine ${r}[(!]" seaice_mesh_cut.F90) -ne 2 ]]; then
        echo "${r} was not cut out of mpas_seaice_mesh.F whole" >&2
        exit 3
    fi
done
if [[ $(printf '%s\n' "${host_use}" | grep -cE 'use seaice_constants, only: coupleAlarmID|^ +pii$|use seaice_mesh, only: seaice_latlon_vector_rotation_backward$|use icepack_intfc, only: icepack_sea_freezing_temperature$') -ne 4 \
      || "${host_decl}" != "   type (domain_type), pointer :: domain" ]]; then
    echo "ice_comp_mct.F's module use lines or declarations moved" >&2
    exit 3
fi
# seaice_constants' coupleAlarmID is a character parameter of the host; nothing here reads it
"${fc[@]}" "${oflags[@]}" "${e3sm}/share/util/shr_kind_mod.F90"
"${fc[@]}" "${oflags[@]}" "${e3sm}/share/util/shr_const_mod.F90"
"${fc[@]}" "${oflags[@]}" "${mpas[@]}" "${e3sm}/components/mpas-framework/src/framework/mpas_kind_types.F"
"${fc[@]}" "${oflags[@]}" "${script_dir}/../mpas_ocean_oracle/stubs.F90"
"${fc[@]}" "${oflags[@]}" "${script_dir}/e3sm_standins.F90"
"${fc[@]}" "${oflags[@]}" mct_mod.F90
"${fc[@]}" "${oflags[@]}" "${mpas[@]}" "${src}/driver/mpassi_cpl_indices.F"
"${fc[@]}" "${oflags[@]}" "${mpas[@]}" -DCCSMCOUPLED "${src}/src/shared/mpas_seaice_constants.F"
# Icepack's columnphysics: compiled in dependency order (each pass compiles what it can)
pending=("${icepack}"/columnphysics/icepack_*.F90)
while ((${#pending[@]})); do
    left=()
    for f in "${pending[@]}"; do
        "${fc[@]}" "${oflags[@]}" "${f}" 2>/dev/null || left+=("${f}")
    done
    if ((${#left[@]} == ${#pending[@]})); then
        "${fc[@]}" "${oflags[@]}" "${left[0]}"
        echo "Icepack did not build" >&2
        exit 4
    fi
    pending=("${left[@]}")
done
"${fc[@]}" "${oflags[@]}" seaice_mesh_cut.F90
"${fc[@]}" "${oflags[@]}" ice_comp_mct_cut.F90
"${fc[@]}" "${oflags[@]}" "${script_dir}/run_seaice_coupling.F90"
"${fc[@]}" -o "${build_dir}/run_seaice_coupling" ./*.o "${mctlib[@]}"
if command -v objdump >/dev/null; then
    objdump -d ice_comp_mct_cut.o seaice_mesh_cut.o icepack_mushy_physics.o icepack_therm_shared.o \
        run_seaice_coupling.o > disassembly.txt
    if grep -iE '\s(v?fn?m(add|sub)[0-9a-z]*|fn?m(add|sub))\s' disassembly.txt > fma.txt; then
        echo "an object contains an FMA instruction (${obj}/fma.txt)" >&2
        exit 5
    fi
fi
# the import and export reach no libm; the rotation's sin, cos, asin and
# atan2 are glibc's (the port takes their words, which the driver records)
nm -u ice_comp_mct_cut.o > undefined.txt
if grep -E '\b(exp|log|pow|sqrt|sin|cos|tan|atan|atan2|asin|acos|cbrt|erf)[fl]?\b' undefined.txt; then
    echo "the sea ice's import or export calls libm" >&2
    exit 6
fi
echo "built ${build_dir}/run_seaice_coupling"
