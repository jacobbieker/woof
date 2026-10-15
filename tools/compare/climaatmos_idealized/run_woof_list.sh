#!/bin/bash
# Run a list of woof_driver.py runs, one per line: "NAME driver-args...".
# Blank lines and lines starting with # are skipped.  Each run takes the
# GPU lock separately (see common.sh woof_run).
#   bash run_woof_list.sh runs.txt
set -u
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
while read -r name args; do
    case "$name" in ''|'#'*) continue ;; esac
    # shellcheck disable=SC2086
    woof_run "$name" $args < /dev/null
done < "$1"
