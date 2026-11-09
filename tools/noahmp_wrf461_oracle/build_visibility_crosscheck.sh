#!/usr/bin/env bash
# Prove the leaf-visibility rewrite is inert across the WHOLE Noah-MP module,
# not just across the eight leaves the fixtures slice out of it.
#
# Why this exists
# ---------------
# `build_leaves.sh` compiles a visibility-patched copy of
# phys/module_sf_noahmplsm.F so the 50 `private ::` leaf routines can be called
# from a separate program unit.  The safety argument has two halves:
#
#   1. `check_visibility_patch.py` proves the *text* diff is nothing but
#      `private ::` -> `public ::`.
#   2. This script proves the *code gfortran emits* for every routine body is
#      unchanged by that rewrite -- all 85 module procedures, covering the
#      ~9,300 lines the leaf fixtures never reach (ENERGY, WATER, PHASECHANGE,
#      CARBON and everything under them).
#
# Half 2 is what the adjudication assumed but never measured: "visibility is
# not physics -- at -O0 it cannot change the codegen of routine bodies at all."
# That is now checked rather than asserted.
#
# What is NOT possible, and why
# -----------------------------
# The obvious cross-check -- run the whole-column driver against the patched
# module and diff against the pristine `noahmp-sflx.csv` -- CANNOT BE BUILT.
# `run_sflx.F90` needs `module_sf_noahmpdrv` (TRANSFER_MP_PARAMETERS,
# SNOW_INIT), and that file declares a dummy argument named `ALBEDO`
# (module_sf_noahmpdrv.F:227).  Lifting the leaf routine `ALBEDO` to public
# makes the name ambiguous and the driver stops compiling:
#
#   Error: Name 'albedo' at (1) is an ambiguous reference to 'albedo'
#          from current program unit
#
# Stage 4 below pins that failure deliberately.  It is a useful property, not
# just an obstacle: the visibility-patched module physically cannot be linked
# into a build that includes WRF's Noah-MP driver, so the patch cannot escape
# the oracle harness and reach a real forecast.  Object-code equivalence
# (stage 2) is the stronger check anyway -- it covers every routine, whereas a
# 4-column fixture only covers the paths those four columns happen to take.
#
# Usage: build_visibility_crosscheck.sh WRF_SOURCE_ROOT BUILD_DIR
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build_visibility_crosscheck.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)

patch_file="${repo_root}/patches/noahmp-lsm-leaf-visibility.patch"
constants_source="${source_root}/share/module_model_constants.F"
gecros_source="${source_root}/phys/module_sf_gecros.F"
lsm_source="${source_root}/phys/module_sf_noahmplsm.F"
drv_source="${source_root}/phys/module_sf_noahmpdrv.F"
glacier_source="${source_root}/phys/module_sf_noahmp_glacier.F"
groundwater_source="${source_root}/phys/module_sf_noahmp_groundwater.F"

# Pristine and patched digests come from whichever of the two pinned trees
# (noahmp_pins.sh) the source carries; the patch file is the same for both,
# because its hunks (lines 27-87) are byte-identical in the two trees.
source "${script_dir}/noahmp_pins.sh"
noahmp_pins "${lsm_source}"
PRISTINE_SHA=${PRISTINE_LSM_SHA}
PATCH_SHA=b16db586bbed70735b148618c794206ed0ff2988dfc8c9ee822b80added34a95

for f in "${constants_source}" "${gecros_source}" "${lsm_source}" \
         "${drv_source}" "${patch_file}"; do
    test -f "${f}"
done

check_sha() {
    local expect="$1" path="$2" label="$3" got
    got=$(sha256sum "${path}" | cut -d' ' -f1)
    if [[ "${got}" != "${expect}" ]]; then
        echo "crosscheck: ${label} sha256 mismatch" >&2
        echo "  path     ${path}" >&2
        echo "  expected ${expect}" >&2
        echo "  actual   ${got}" >&2
        exit 3
    fi
}

check_sha "${PRISTINE_SHA}" "${lsm_source}" "pinned pristine module"
# Stage 4 reads the driver's verdict, so the driver and the glacier module it
# USEs must be the same pinned tree's, not just any file of that name.
check_sha "${PRISTINE_DRV_SHA}" "${drv_source}" "pinned driver"
check_sha "${PRISTINE_GLA_SHA}" "${glacier_source}" "pinned glacier module"
check_sha "${PATCH_SHA}" "${patch_file}" "visibility patch"

rm -rf "${build_dir}"
mkdir -p "${build_dir}"
cd "${build_dir}"

fflags=(-O0 -cpp -ffree-form -ffree-line-length-none)
gfortran -c "${fflags[@]}" "${constants_source}" 2>/dev/null
gfortran -c "${fflags[@]}" "${gecros_source}" 2>/dev/null

# Compile a module_sf_noahmplsm.F into its own directory.  Same basename and
# same flags every time, so nothing but the source content can differ.
compile_module() {
    local tag="$1" src="$2"
    mkdir -p "${build_dir}/${tag}"
    cp "${src}" "${build_dir}/${tag}/module_sf_noahmplsm.F"
    ( cd "${build_dir}/${tag}" \
      && gfortran -c "${fflags[@]}" -I "${build_dir}" \
           module_sf_noahmplsm.F ) > "${build_dir}/${tag}.log" 2>&1
}

# --- stage 1: apply and audit the patch ------------------------------------
mkdir -p "${build_dir}/patched-src/phys"
cp "${lsm_source}" "${build_dir}/patched-src/phys/module_sf_noahmplsm.F"
( cd "${build_dir}/patched-src" \
  && patch -p1 --no-backup-if-mismatch < "${patch_file}" ) > /dev/null
patched_source="${build_dir}/patched-src/phys/module_sf_noahmplsm.F"
check_sha "${PATCHED_SHA}" "${patched_source}" "visibility-patched module"
python3 "${script_dir}/check_visibility_patch.py" \
    "${lsm_source}" "${patched_source}" "${patch_file}"

# --- stage 2: object-code equivalence --------------------------------------
compile_module pristine "${lsm_source}"
compile_module patched  "${patched_source}"

echo "--- stage 2: pristine vs visibility-patched object code"
if ! python3 "${script_dir}/compare_object_code.py" \
        "${build_dir}/pristine/module_sf_noahmplsm.o" \
        "${build_dir}/patched/module_sf_noahmplsm.o"; then
    echo "CROSSCHECK FAILED: the visibility rewrite changed emitted code" >&2
    exit 4
fi

# --- stage 3: negative controls --------------------------------------------
# Stage 2 is worth nothing unless it can fail.  Two perturbations of the
# patched source, each of a different kind, must both be caught.
echo "--- stage 3: negative controls"

# 3a. one FP32 literal: CSNOW's Stieglitz coefficient at line 2562.  Live code,
#     and the literal is unique in the file.  Shows up in .rodata.
mkdir -p "${build_dir}/mutant-literal"
literal_line=$(grep -n '3\.2217E-6' "${patched_source}" | cut -d: -f1)
if [[ $(printf '%s\n' "${literal_line}" | wc -l) -ne 1 ]]; then
    echo "crosscheck: expected exactly one 3.2217E-6 literal" >&2
    exit 5
fi
sed "${literal_line}s/3\.2217E-6/3.2218E-6/" "${patched_source}" \
    > "${build_dir}/mutant-literal.F"
cmp -s "${patched_source}" "${build_dir}/mutant-literal.F" \
    && { echo "crosscheck: literal mutation was a no-op" >&2; exit 5; }
compile_module mutant-literal "${build_dir}/mutant-literal.F"

if python3 "${script_dir}/compare_object_code.py" \
        "${build_dir}/patched/module_sf_noahmplsm.o" \
        "${build_dir}/mutant-literal/module_sf_noahmplsm.o" > /dev/null 2>&1
then
    echo "CROSSCHECK IS VACUOUS: a changed FP32 literal was not detected" >&2
    exit 6
fi
echo "  3a literal mutation: detected"

# 3b. one operator inside CSNOW: changes an instruction, not a constant.
sed 's/      EPORE(IZ)    = 1.0 - SNICEV(IZ)/      EPORE(IZ)    = 1.0 + SNICEV(IZ)/' \
    "${patched_source}" > "${build_dir}/mutant-operator.F"
cmp -s "${patched_source}" "${build_dir}/mutant-operator.F" \
    && { echo "crosscheck: operator mutation was a no-op" >&2; exit 5; }
compile_module mutant-operator "${build_dir}/mutant-operator.F"

if python3 "${script_dir}/compare_object_code.py" \
        "${build_dir}/patched/module_sf_noahmplsm.o" \
        "${build_dir}/mutant-operator/module_sf_noahmplsm.o" > /dev/null 2>&1
then
    echo "CROSSCHECK IS VACUOUS: a changed operator was not detected" >&2
    exit 6
fi
echo "  3b operator mutation: detected"

# --- stage 4: the patch must NOT be linkable with WRF's Noah-MP driver ------
# Pinning this failure keeps the exposure confined to the oracle harness.
echo "--- stage 4: patched module must break module_sf_noahmpdrv.F"
# The driver also USEs the glacier and groundwater modules and the WRF
# service stubs; build those against the patched module first, so the only
# thing that can stop the driver compiling is the module it is given.
if ! ( cd "${build_dir}/patched" \
       && gfortran -c "${fflags[@]}" -I "${build_dir}/patched" -I "${build_dir}" \
            "${script_dir}/stub_wrf.F90" "${glacier_source}" \
            "${groundwater_source}" \
     ) > "${build_dir}/drv-deps.log" 2>&1; then
    echo "UNEXPECTED: the driver's other dependencies failed to compile;" >&2
    echo "inspect ${build_dir}/drv-deps.log" >&2
    exit 7
fi
if ( cd "${build_dir}/patched" \
     && gfortran -c "${fflags[@]}" -DEM_CORE=0 -fallow-argument-mismatch \
          -I "${build_dir}/patched" -I "${build_dir}" "${drv_source}" \
     ) > "${build_dir}/drv-patched.log" 2>&1; then
    echo "UNEXPECTED: module_sf_noahmpdrv.F compiled against the patched" >&2
    echo "module.  The ALBEDO ambiguity that confines this patch to the" >&2
    echo "oracle harness is gone; re-adjudicate before trusting it." >&2
    exit 7
fi
if ! grep -q "ambiguous reference to .albedo." "${build_dir}/drv-patched.log"
then
    echo "UNEXPECTED: driver failed for a different reason than the known" >&2
    echo "ALBEDO ambiguity; inspect ${build_dir}/drv-patched.log" >&2
    exit 7
fi
echo "  driver rejects the patched module on the ALBEDO ambiguity, as pinned"

# --- evidence ---------------------------------------------------------------
{
    echo "tree                 ${NOAHMP_TREE} (phys/noahmp ${NOAHMP_COMMIT})"
    echo "pristine_module      ${PRISTINE_SHA}"
    echo "patched_module       ${PATCHED_SHA}"
    echo "visibility_patch     ${PATCH_SHA}"
    echo "object_code          identical over all common function bodies"
    echo "negative_control_3a  FP32 literal change detected (.rodata)"
    echo "negative_control_3b  operator change detected (csnow body)"
    echo "driver_linkage       refused: ALBEDO ambiguity (by design)"
    echo "compiler             $(gfortran --version | head -1)"
    echo "binutils             $(objdump --version | head -1)"
} > "${build_dir}/visibility-crosscheck.txt"
echo
cat "${build_dir}/visibility-crosscheck.txt"
