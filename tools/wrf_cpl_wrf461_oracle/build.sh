#!/usr/bin/env bash
# Build and run the WRF v4.6.1 ocean-coupler exchange oracle: cpl_snd2 and
# cpl_rcv of the byte-unmodified frame/module_cpl.F.
#
# module_cpl.F cannot be compiled whole outside WRF: it USEs module_domain,
# which WRF's Registry generates.  So build.sh cuts the two subroutines out of
# the pinned file from their SUBROUTINE line to their END SUBROUTINE line,
# checks the cut text against the digests recorded below, and #includes it
# in cpl_harness.F90, whose TYPE(domain) carries only the components the two
# bodies read.  Everything the oracle computes is that text.
#
# Same toolchain discipline as tools/noah_wrf461_oracle: -O0, WRF's own
# phys/frame defines, and a refusal if any _ZGV* (libmvec) symbol appears.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
pinned_commit="d66e442fccc04111067e29274c9f9eaccc3cef28"

sources=(share/module_model_constants.F frame/module_cpl.F)
for rel in "${sources[@]}"; do
    test -f "${source_root}/${rel}"
done
head_commit=$(git -C "${source_root}" rev-parse HEAD)
if [[ "${head_commit}" != "${pinned_commit}" ]]; then
    echo "WRF tree is at ${head_commit}, not the pinned ${pinned_commit}" >&2
    exit 3
fi
if ! git -C "${source_root}" diff --quiet HEAD -- "${sources[@]}"; then
    echo "one of the coupler sources differs from the pinned commit" >&2
    exit 3
fi

mkdir -p "${build_dir}"
cd "${build_dir}"

# --- the two subroutines, byte for byte -------------------------------------
cut_subroutine() {
    awk -v name="$1" '
        $0 ~ "^ *SUBROUTINE " name "\\(" { on = 1 }
        on { print }
        on && $0 ~ "END SUBROUTINE " name "$" { exit }
    ' "${source_root}/frame/module_cpl.F"
}
cut_subroutine cpl_snd2 > cpl_snd2.inc
cut_subroutine cpl_rcv > cpl_rcv.inc
sha256sum cpl_snd2.inc cpl_rcv.inc > cut-sha256sums.txt
# The cuts are pinned so a change to the awk or to the file cannot pass as
# the same oracle.  module_cpl.F:298-382 and :408-455.
for pin in "aa20302003dde633b097ad0fcf47cf094d1609593f6d75c6e740b122c454e46c  cpl_snd2.inc" \
           "1d74da88c615c769ea0762acda9dde5ba862064fa5a1f187d442e01bf61a29bf  cpl_rcv.inc"; do
    if ! grep -qxF "${pin}" cut-sha256sums.txt; then
        echo "cut ${pin##* } does not match its pinned digest" >&2
        exit 6
    fi
done
expected_snd2="$(grep -c '' cpl_snd2.inc)"
expected_rcv="$(grep -c '' cpl_rcv.inc)"
if [[ "${expected_snd2}" -ne 85 || "${expected_rcv}" -ne 48 ]]; then
    echo "cut lengths ${expected_snd2}/${expected_rcv}, expected 85/48" >&2
    exit 6
fi

defines="-DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4"
free="-ffree-form -ffree-line-length-none"

gfortran -c -O0 -cpp ${defines} ${free} \
    "${source_root}/share/module_model_constants.F"
gfortran -c -O0 -cpp ${defines} ${free} -I "${build_dir}" \
    "${script_dir}/cpl_harness.F90"
gfortran -c -O0 ${free} -I "${build_dir}" "${script_dir}/run_cpl.F90"
gfortran -o run_cpl module_model_constants.o cpl_harness.o run_cpl.o

nm -u cpl_harness.o | sed 's/^ *//' | sort > undefined-O0-cpl_harness.txt
if grep -q '_ZGV' undefined-O0-cpl_harness.txt; then
    echo "-O0 cpl_harness.o pulled in a libmvec vector symbol" >&2
    exit 4
fi
{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# -O0 cpl_harness.o math symbols:"
    grep -E 'powi|powf|sqrtf|_ZGV' undefined-O0-cpl_harness.txt || true
} > libm-report.txt
cat libm-report.txt

./run_cpl wrf-cpl.csv > run-cpl-stdout.txt

sha256sum "${source_root}/share/module_model_constants.F" \
    "${source_root}/frame/module_cpl.F" \
    cpl_snd2.inc cpl_rcv.inc \
    "${script_dir}/cpl_harness.F90" "${script_dir}/run_cpl.F90" \
    "${script_dir}/build.sh" wrf-cpl.csv \
    | sed "s#${source_root}/#<wrf-4.6.1-checkout>/#; s#${script_dir}/#<oracle-workspace>/tools/wrf_cpl_wrf461_oracle/#" \
    > oracle-sha256sums.txt
echo "oracle built in ${build_dir}"
