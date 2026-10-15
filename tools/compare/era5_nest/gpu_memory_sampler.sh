#!/usr/bin/env bash
# Sample the device memory a process tree holds, every INTERVAL seconds,
# until the root process exits.  Writes "unix_seconds<TAB>MiB" lines and a
# final "# peak_mib<TAB>N" line.  On a shared card nvidia-smi's per-process
# figure is the only number that belongs to the run alone.
#
#   gpu_memory_sampler.sh --pid PID --out FILE [--interval 2]
set -euo pipefail
interval=2
while [ $# -gt 0 ]; do
    case $1 in
    --pid) root=$2; shift 2;;
    --out) out=$2; shift 2;;
    --interval) interval=$2; shift 2;;
    *) echo "unknown argument $1" >&2; exit 2;;
    esac
done
: "${root:?--pid is required}" "${out:?--out is required}"

tree() {
    local pids=$1 frontier=$1 next
    while [ -n "$frontier" ]; do
        next=$(for p in $frontier; do pgrep -P "$p" || true; done | tr '\n' ' ')
        pids="$pids $next"
        frontier=$next
    done
    echo "$pids"
}

peak=0
: > "$out"
while kill -0 "$root" 2>/dev/null; do
    pids=" $(tree "$root") "
    used=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
        | awk -F', *' -v pids="$pids" 'index(pids, " " $1 " ") {s += $2} END {print s + 0}' \
        || echo 0)
    printf '%s\t%s\n' "$(date +%s)" "$used" >> "$out"
    [ "$used" -gt "$peak" ] && peak=$used
    sleep "$interval"
done
printf '# peak_mib\t%s\n' "$peak" >> "$out"
