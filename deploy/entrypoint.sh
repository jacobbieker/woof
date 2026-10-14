#!/usr/bin/env bash
# Entry point of the woof energy worker container (deploy/Dockerfile).
#
#   woof-entrypoint [worker] [EXTRA WORKER FLAGS...]
#       woof energy worker --queue "$WOOF_ENERGY_QUEUE_URL"
#                          --gpus "${WOOF_ENERGY_GPUS:-all}" EXTRA...
#   woof-entrypoint COMMAND [ARGS...]   any command, e.g. the full
#                                       `woof energy worker ...` line the
#                                       renderers write
#   woof-entrypoint --sync-only         stage the static data, then exit
#   woof-entrypoint --healthcheck       `woof doctor`, quietly (no sync)
#
# Before the command runs, the static data a run reads is staged onto local
# scratch (`$WOOF_SCRATCH`, default /scratch: instance NVMe, a GCP scratch
# disk or a Kubernetes emptyDir):
#
#   $WOOF_GEOG_URL           -> $GPUWM_CASE_DATA_ROOT/WPS_GEOG
#                               (= $GPUWM_WPS_GEOG; woof's default geog root)
#   $WOOF_HIGHRES_CACHE_URL  -> $XDG_CACHE_HOME/woof/highres-cache
#
# URLs are s3://, gs://, file:// or an absolute path.  The copy is rclone
# with the ambient identity (instance role, ECS task role, IRSA, GKE
# Workload Identity, VM service account: env_auth), falling back to the
# aws/gcloud CLIs or cp.  Nothing here reads or writes a credential.  A
# copy finished on this node is recorded in `.woof-synced` and not
# repeated; concurrent workers on one node wait on a lock instead of
# copying twice.  WOOF_SYNC_ON_START=0 skips staging.
#
# Packing several jobs per GPU (docs/energy-hindcast/queue-worker.md), for a
# `woof energy worker` command (worker mode or the full command):
#
#   WOOF_SLOTS_PER_GPU=N|auto  adds --slots-per-gpu N|auto unless the command
#                              already carries it
#   WOOF_ENABLE_MPS=1          adds --mps --mps-dir $WOOF_SCRATCH/mps/<index>
#                              (the worker starts nvidia-cuda-mps-control -d
#                              with its pipe/ and log/ there, hands
#                              CUDA_MPS_PIPE_DIRECTORY/CUDA_MPS_LOG_DIRECTORY
#                              to its slots and stops it on exit; it refuses
#                              when the control binary is missing).  MPS runs
#                              packed slots' kernels concurrently instead of
#                              time-slicing them.
set -euo pipefail

log() { printf 'woof-entrypoint: %s\n' "$*" >&2; }

# The image's pixi environment (written at build time by `pixi shell-hook`).
if [ -f /opt/woof/activate.sh ]; then
  set +u
  # shellcheck disable=SC1091
  . /opt/woof/activate.sh
  set -u
fi

# Tables `woof fetch-tables` staged into the image at build time.  The image
# keeps them outside $HOME so a runtime that swaps $HOME (Apptainer) still
# finds them.
image_home=/opt/woof-home/.woof
if [ -z "${WOOF_THOMPSON_TABLE_ROOT:-}" ] && [ -d "$image_home/tables/thompson" ]; then
  export WOOF_THOMPSON_TABLE_ROOT="$image_home/tables/thompson"
fi
if [ -z "${WOOF_WIF_DATA_ROOT:-}" ] && [ -d "$image_home/wif" ]; then
  export WOOF_WIF_DATA_ROOT="$image_home/wif"
fi

mode="${1:-worker}"
if [ "$mode" = "--healthcheck" ]; then
  exec woof doctor >/dev/null
fi

WOOF_SCRATCH="${WOOF_SCRATCH:-/scratch}"
if ! mkdir -p "$WOOF_SCRATCH" 2>/dev/null || [ ! -w "$WOOF_SCRATCH" ]; then
  fallback="${HOME:-/tmp}/woof-scratch"
  log "scratch $WOOF_SCRATCH is not writable by uid $(id -u); using $fallback"
  WOOF_SCRATCH="$fallback"
  mkdir -p "$WOOF_SCRATCH"
fi
export WOOF_SCRATCH
export GPUWM_CASE_DATA_ROOT="${GPUWM_CASE_DATA_ROOT:-$WOOF_SCRATCH/case-data}"
export GPUWM_WPS_GEOG="${GPUWM_WPS_GEOG:-$GPUWM_CASE_DATA_ROOT/WPS_GEOG}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$WOOF_SCRATCH/cache}"
mkdir -p "$GPUWM_CASE_DATA_ROOT" "$XDG_CACHE_HOME"
# An inherited TMPDIR may name a host path the container cannot see
# (Apptainer and Pyxis forward the job's environment): use scratch instead.
if [ -z "${TMPDIR:-}" ] || ! mkdir -p "$TMPDIR" 2>/dev/null || [ ! -w "$TMPDIR" ]; then
  export TMPDIR="$WOOF_SCRATCH/tmp"
  mkdir -p "$TMPDIR"
fi

# One index name for logs, whichever scheduler started us.
for var in JOB_COMPLETION_INDEX AWS_BATCH_JOB_ARRAY_INDEX BATCH_TASK_INDEX \
           SLURM_ARRAY_TASK_ID; do
  if [ -n "${!var:-}" ]; then
    export WOOF_WORKER_INDEX="${!var}"
    break
  fi
done
export WOOF_WORKER_INDEX="${WOOF_WORKER_INDEX:-0}"

# rclone's on-the-fly remote for a URL (env_auth: the ambient identity).
rclone_source() {
  local url="$1" region="${AWS_REGION:-${AWS_DEFAULT_REGION:-}}"
  case "$url" in
    s3://*)
      if [ -n "$region" ]; then
        printf ':s3,provider=AWS,env_auth=true,region=%s:%s' "$region" "${url#s3://}"
      else
        printf ':s3,provider=AWS,env_auth=true:%s' "${url#s3://}"
      fi ;;
    gs://*) printf ':gcs,env_auth=true,bucket_policy_only=true:%s' "${url#gs://}" ;;
    file://*) printf '%s' "${url#file://}" ;;
    /*) printf '%s' "$url" ;;
    *) return 1 ;;
  esac
}

copy_tree() {
  local url="$1" dest="$2" source
  case "$url" in
    s3://*|gs://*|file://*|/*) ;;
    *) log "REFUSED: $url is not s3://, gs://, file:// or an absolute path"; exit 2 ;;
  esac
  if command -v rclone >/dev/null 2>&1; then
    source="$(rclone_source "$url")"
    rclone copy "$source" "$dest" \
      --transfers "${WOOF_SYNC_TRANSFERS:-32}" --checkers 64 --fast-list \
      --retries 5 --low-level-retries 20 --stats 60s --stats-one-line
  elif [ "${url#s3://}" != "$url" ] && command -v aws >/dev/null 2>&1; then
    aws s3 sync --only-show-errors "$url" "$dest"
  elif [ "${url#gs://}" != "$url" ] && command -v gcloud >/dev/null 2>&1; then
    gcloud storage rsync --recursive "$url" "$dest"
  elif [ "${url#gs://}" = "$url" ] && [ "${url#s3://}" = "$url" ]; then
    local path="${url#file://}"
    cp -a "$path/." "$dest/"
  else
    log "REFUSED: no rclone, aws or gcloud to copy $url"
    exit 2
  fi
}

# stage URL DEST LABEL: copy once per node; later workers reuse it.
stage() {
  local url="$1" dest="$2" label="$3"
  mkdir -p "$dest"
  exec 9>"$dest.lock"
  flock 9
  if [ -f "$dest/.woof-synced" ] && [ "$(cat "$dest/.woof-synced")" = "$url" ]; then
    log "$label already staged from $url"
  else
    log "staging $label: $url -> $dest"
    copy_tree "$url" "$dest"
    printf '%s\n' "$url" > "$dest/.woof-synced"
    log "staged $label ($(du -sh "$dest" 2>/dev/null | cut -f1))"
  fi
  flock -u 9
  exec 9>&-
}

if [ "${WOOF_SYNC_ON_START:-1}" != "0" ]; then
  if [ -n "${WOOF_GEOG_URL:-}" ]; then
    stage "$WOOF_GEOG_URL" "$GPUWM_WPS_GEOG" "WPS_GEOG"
  elif [ ! -d "$GPUWM_WPS_GEOG" ]; then
    log "no WOOF_GEOG_URL and no $GPUWM_WPS_GEOG; runs that need WPS_GEOG will fetch or refuse"
  fi
  if [ -n "${WOOF_HIGHRES_CACHE_URL:-}" ]; then
    stage "$WOOF_HIGHRES_CACHE_URL" "$XDG_CACHE_HOME/woof/highres-cache" "high-resolution cache"
  fi
fi

case "$mode" in
  --sync-only)
    exit 0 ;;
  worker)
    shift || true
    if [ -z "${WOOF_ENERGY_QUEUE_URL:-}" ]; then
      log "REFUSED: worker mode needs WOOF_ENERGY_QUEUE_URL (or pass a full command)"
      exit 2
    fi
    set -- woof energy worker --queue "$WOOF_ENERGY_QUEUE_URL" \
      --gpus "${WOOF_ENERGY_GPUS:-all}" "$@" ;;
esac

# has_arg FLAG ARGS...: whether ARGS carries FLAG (alone or as FLAG=...).
has_arg() {
  local flag="$1" arg
  shift
  for arg in "$@"; do
    case "$arg" in
      "$flag"|"$flag"=*) return 0 ;;
    esac
  done
  return 1
}

if [ "$#" -ge 3 ] && [ "$1" = woof ] && [ "$2" = energy ] && [ "$3" = worker ]; then
  if [ -n "${WOOF_SLOTS_PER_GPU:-}" ] && ! has_arg --slots-per-gpu "$@"; then
    set -- "$@" --slots-per-gpu "$WOOF_SLOTS_PER_GPU"
  fi
  case "${WOOF_ENABLE_MPS:-0}" in
    1|true|yes)
      if ! has_arg --mps "$@"; then
        set -- "$@" --mps
      fi
      if ! has_arg --mps-dir "$@"; then
        mps_dir="$WOOF_SCRATCH/mps/$WOOF_WORKER_INDEX"
        mkdir -p "$mps_dir"
        set -- "$@" --mps-dir "$mps_dir"
      fi ;;
    0|false|no|"") ;;
    *) log "REFUSED: WOOF_ENABLE_MPS=$WOOF_ENABLE_MPS is not 0 or 1"; exit 2 ;;
  esac
fi

log "worker index $WOOF_WORKER_INDEX on $(hostname): $*"
exec "$@"
