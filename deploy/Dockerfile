# Container for `woof energy worker`: the GPU worker that drains a
# `woof energy hindcast` job queue on AWS Batch, GCP Batch, Kubernetes or
# Slurm (`woof energy deploy render` writes those deployments).
#
# Build from the repository root (the context is the checkout; the
# Dockerfile-specific ignore file beside this one keeps .git, .pixi and
# cargo target directories out of it):
#
#   docker build -f deploy/Dockerfile -t woof-energy:1.0.3-cu12 .
#   docker build -f deploy/Dockerfile --build-arg CUDA=13 \
#     --build-arg CUDA_BASE=nvidia/cuda:13.0.1-base-ubuntu24.04 \
#     -t woof-energy:1.0.3-cu13 .
#
# Stages:
#   build    the locked pixi environment (`default` for CUDA 12, `cuda13`
#            for CUDA 13; CuPy brings its own CUDA toolkit wheels), the
#            native Rust bridges (`pixi run build-native`, vendored crates,
#            offline), and the small tables (`woof fetch-tables`, plus the
#            215 MiB WIF climatology unless PREFETCH_WIF=0).
#   runtime  the CUDA base image (driver injected by the NVIDIA container
#            toolkit), tini, rclone, the built checkout, a non-root user
#            (uid/gid 10001), and deploy/entrypoint.sh.
#
# Not in the image: WPS_GEOG (~30 GiB) and the high-resolution static cache.
# The entrypoint copies them from $WOOF_GEOG_URL / $WOOF_HIGHRES_CACHE_URL
# onto local scratch at start, or they are mounted at $GPUWM_WPS_GEOG.
#
# Pinned: the pixi lock (pixi.lock, `--locked`), the vendored crates
# (`cargo --locked --offline`), the table bytes (fetch-tables verifies
# SHA-256), and the tool versions below.  The CUDA base image is pinned by
# tag; pin it by digest for a reproducible fleet.

ARG CUDA_BASE=nvidia/cuda:12.6.3-base-ubuntu22.04
ARG PIXI_VERSION=0.77.0
ARG RCLONE_VERSION=1.71.1

FROM ghcr.io/prefix-dev/pixi:${PIXI_VERSION} AS pixi
FROM rclone/rclone:${RCLONE_VERSION} AS rclone

# ---------------------------------------------------------------------------
FROM ${CUDA_BASE} AS build
ARG CUDA=12
ARG PREFETCH_WIF=1
ENV DEBIAN_FRONTEND=noninteractive \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    HOME=/opt/woof-home \
    GPUWM_NO_LOCAL_GPU=1
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential ca-certificates git \
 && rm -rf /var/lib/apt/lists/*
COPY --from=pixi /usr/local/bin/pixi /usr/local/bin/pixi
WORKDIR /opt/woof
COPY . /opt/woof
# The pixi environment for the CUDA major, recorded for the later steps.
RUN case "${CUDA}" in \
      12) env=default ;; \
      13) env=cuda13 ;; \
      *) echo "CUDA must be 12 or 13, got ${CUDA}" >&2; exit 2 ;; \
    esac \
 && echo "${env}" > /opt/woof/.woof-pixi-env \
 && pixi install --locked -e "${env}" \
 && ln -s ".pixi/envs/${env}" /opt/woof/env \
 && pixi shell-hook -e "${env}" > /opt/woof/activate.sh \
 && rm -rf "${HOME}/.cache/rattler"
# Native bridges, then drop cargo's intermediates (the artifacts woof loads
# sit directly in target/release).
RUN pixi run -e "$(cat /opt/woof/.woof-pixi-env)" build-native \
 && for crate in tools/*/target/release; do \
      rm -rf "${crate}/deps" "${crate}/build" "${crate}/.fingerprint" "${crate}/incremental"; \
    done \
 && rm -rf "${HOME}/.cargo/registry"
# Small tables into /opt/woof-home/.woof (outside the runtime $HOME, so a
# runtime that replaces $HOME still finds them; the entrypoint exports
# WOOF_THOMPSON_TABLE_ROOT / WOOF_WIF_DATA_ROOT).
RUN env="$(cat /opt/woof/.woof-pixi-env)" \
 && pixi run -e "${env}" woof fetch-tables \
 && if [ "${PREFETCH_WIF}" = "1" ]; then pixi run -e "${env}" woof fetch-tables --wif --wif-only; fi

# ---------------------------------------------------------------------------
FROM ${CUDA_BASE} AS runtime
ENV DEBIAN_FRONTEND=noninteractive \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates tini \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --gid 10001 woof \
 && useradd --uid 10001 --gid 10001 --create-home --home-dir /home/woof --shell /bin/bash woof \
 && mkdir -p /scratch \
 && chown 10001:10001 /scratch
COPY --from=rclone /usr/local/bin/rclone /usr/local/bin/rclone
COPY --from=build --chown=10001:10001 /opt/woof /opt/woof
COPY --from=build --chown=10001:10001 /opt/woof-home /opt/woof-home
COPY --chmod=0755 deploy/entrypoint.sh /usr/local/bin/woof-entrypoint
ENV PATH=/opt/woof/env/bin:${PATH} \
    HOME=/home/woof \
    WOOF_SCRATCH=/scratch
USER 10001:10001
WORKDIR /home/woof
# `woof doctor` (no sync).  Docker and ECS honour this; Batch and Kubernetes
# Jobs judge a worker by its exit status instead.
HEALTHCHECK --interval=10m --timeout=5m --start-period=30m --retries=3 \
  CMD ["/usr/local/bin/woof-entrypoint", "--healthcheck"]
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/woof-entrypoint"]
CMD ["worker"]
