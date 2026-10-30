#!/usr/bin/env bash
# Fetch the WRF v4.7.1 files SOURCES.sha256 pins into DEST, laid out as in
# the WRF tree, and refuse any file whose sha256 differs from its pin.
set -euo pipefail
if [[ $# -ne 1 ]]; then
    echo "usage: fetch_sources.sh DEST" >&2
    exit 2
fi
mkdir -p "$1"
dest=$(cd "$1" && pwd)
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
wrf="https://raw.githubusercontent.com/wrf-model/WRF/v4.7.1"
grep -v '^#' "${script_dir}/SOURCES.sha256" | while read -r sum path; do
    [[ -z "${sum}" ]] && continue
    mkdir -p "${dest}/$(dirname "${path}")"
    [[ -f "${dest}/${path}" ]] && continue
    curl -sfL -o "${dest}/${path}" "${wrf}/${path}"
done
cd "${dest}"
grep -v '^#' "${script_dir}/SOURCES.sha256" | { command -v sha256sum >/dev/null && sha256sum -c --quiet - || shasum -a 256 -c --quiet -; }
echo "WRF v4.7.1 partial-nest coupling sources verified in ${dest}"
