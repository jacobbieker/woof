# Deploying the wind-farm hindcast: container and cloud backends

The wind-farm reanalysis is a large set of independent GPU jobs.
`woof energy hindcast plan` writes them to a queue in object storage, and
`woof energy worker` claims jobs from that queue one at a time and runs one
per GPU. This page covers the two pieces that put those workers on
machines:

1. **The container** (`deploy/Dockerfile`, `deploy/entrypoint.sh`). It
   holds the locked pixi environment, the native Rust bridges and the small
   tables, and runs as a non-root user.
2. **`woof energy deploy render`**. From one typed configuration it writes
   the files that start N workers on AWS Batch, GCP Batch, Kubernetes or
   Slurm.

Rendering never calls a cloud API. You review the files it writes, then
run them yourself.

## The container

```bash
# CUDA 12 (pixi environment `default`)
docker build -f deploy/Dockerfile -t woof-energy:1.0.3-cu12 .
# CUDA 13 (pixi environment `cuda13`)
docker build -f deploy/Dockerfile --build-arg CUDA=13 \
  --build-arg CUDA_BASE=nvidia/cuda:13.0.1-base-ubuntu24.04 \
  -t woof-energy:1.0.3-cu13 .
```

Run the build from the repository root. `deploy/Dockerfile.dockerignore`
keeps `.git`, `.pixi` and the cargo `target/` directories out of the build
context.

| Stage | What it does |
|---|---|
| `build` | `pixi install --locked -e default\|cuda13`; `pixi run build-native` (vendored crates, offline), keeping only the release artifacts; `woof fetch-tables`, plus the 215 MiB WIF climatology unless `--build-arg PREFETCH_WIF=0` |
| `runtime` | the CUDA `base` image (the NVIDIA container toolkit injects the driver; CuPy ships its own CUDA wheels), `tini`, `rclone`, the built checkout, user `woof` (uid/gid 10001), and the entrypoint |

| Build arg | Default | Meaning |
|---|---|---|
| `CUDA` | `12` | `12` selects pixi env `default`, `13` selects `cuda13` |
| `CUDA_BASE` | `nvidia/cuda:12.6.3-base-ubuntu22.04` | base image for both stages; pin it by digest for a reproducible fleet |
| `PIXI_VERSION` | `0.77.0` | the pixi that reads `pixi.lock` |
| `RCLONE_VERSION` | `1.71.1` | the object-store copier the entrypoint uses |
| `PREFETCH_WIF` | `1` | also stage the WIF aerosol climatology |

`HEALTHCHECK` runs `woof doctor` without syncing anything. Docker and ECS
use it. Batch services and Kubernetes Jobs ignore it and judge a worker by
its exit status.

### What the entrypoint does

`/usr/local/bin/woof-entrypoint` runs under `tini`. Before it `exec`s the
command, it does the following:

1. Sets up scratch. `WOOF_SCRATCH` defaults to `/scratch`, which is the
   instance NVMe, a GCP scratch disk or a Kubernetes volume. If scratch is
   not writable, it falls back to `$HOME/woof-scratch`. It then exports
   `GPUWM_CASE_DATA_ROOT=$WOOF_SCRATCH/case-data`,
   `GPUWM_WPS_GEOG=$GPUWM_CASE_DATA_ROOT/WPS_GEOG`,
   `XDG_CACHE_HOME=$WOOF_SCRATCH/cache` and `TMPDIR`, unless they are
   already set.
2. Stages static data from object storage. WPS_GEOG (about 30 GiB) and the
   high-resolution static cache are not in the image:

   | Variable | Copied to |
   |---|---|
   | `WOOF_GEOG_URL` | `$GPUWM_WPS_GEOG`, woof's default geog root |
   | `WOOF_HIGHRES_CACHE_URL` | `$XDG_CACHE_HOME/woof/highres-cache` |

   URLs can be `s3://`, `gs://`, `file://` or an absolute path. The copy is
   `rclone copy` with `env_auth`, which uses the identity the machine
   already has: instance role, ECS task role, IRSA, GKE Workload Identity
   or VM service account. Without rclone it falls back to `aws s3 sync`,
   `gcloud storage rsync` or `cp`. Each node copies once and records the
   source in `.woof-synced`. Concurrent workers on one node wait on a
   `flock` rather than copying twice. Set `WOOF_SYNC_ON_START=0` to skip
   this step. Seed the bucket once from a machine that has the data, for
   example `woof fetch-geog --datasets wrf` and then
   `aws s3 sync "$GPUWM_CASE_DATA_ROOT/WPS_GEOG" s3://bucket/WPS_GEOG`.
3. Exports `WOOF_WORKER_INDEX`. It is taken from `JOB_COMPLETION_INDEX`,
   `AWS_BATCH_JOB_ARRAY_INDEX`, `BATCH_TASK_INDEX` or
   `SLURM_ARRAY_TASK_ID`, whichever the scheduler set.
4. Exports `WOOF_THOMPSON_TABLE_ROOT` and `WOOF_WIF_DATA_ROOT`, pointing at
   the tables staged in the image under `/opt/woof-home/.woof`. These sit
   outside `$HOME`, so Apptainer, which replaces `$HOME`, still finds them.

| Invocation | Runs |
|---|---|
| `woof-entrypoint` / `woof-entrypoint worker [FLAGS]` | `woof energy worker --queue "$WOOF_ENERGY_QUEUE_URL" --gpus "${WOOF_ENERGY_GPUS:-all}" FLAGS` |
| `woof-entrypoint COMMAND ...` | any command after staging (what the renderers use) |
| `woof-entrypoint --sync-only` | stage, then exit |
| `woof-entrypoint --healthcheck` | `woof doctor`, quietly |

A bad URL, or worker mode without `WOOF_ENERGY_QUEUE_URL`, exits 2.

## Rendering a deployment

```bash
woof energy deploy render --backend aws-batch --config deploy/examples/aws.toml -o deploy-out/aws
woof energy deploy render --backend gcp-batch --config deploy/examples/gcp.toml -o deploy-out/gcp
woof energy deploy render --backend k8s --config deploy/examples/aws.toml -o deploy-out/eks
woof energy deploy render --backend slurm --config deploy/examples/aws.toml --workers 8 -o deploy-out/slurm
```

The configuration is TOML, or JSON when the file ends in `.json`. The
flags `--name`, `--image`, `--queue`, `--output-url`, `--workers`,
`--parallelism`, `--gpus-per-worker`, `--geog-url`, `--highres-cache-url`
and `--spot`/`--on-demand` override the top-level keys. A configuration
can also be given by flags alone.

The command prints a JSON summary to stdout and writes notes to stderr.
The output directory always gets `deploy.json` (`woof-energy.deploy.v1`),
which records:

- the backend
- the normalized configuration and its SHA-256
- the worker command
- each file with its SHA-256
- any placeholders left and any notes

Rendering is deterministic. The same configuration and woof version give
byte-identical files. Rendering again into a directory this command wrote
removes the files the earlier render listed and this one does not. A
directory whose `deploy.json` is something else is refused.

### Top-level keys

| Key | Default | Meaning |
|---|---|---|
| `name` | required | 1-40 characters: lower-case letters, digits and `-`. Prefixes every cloud resource |
| `image` | required | worker image, pinned by tag or `@sha256:` digest (`:latest` is refused) |
| `queue_url` | required | the queue `woof energy hindcast plan` filled. Passed as `--queue` and as `WOOF_ENERGY_QUEUE_URL` |
| `output_url` | required | the Icechunk repository. Passed as `WOOF_ENERGY_OUTPUT_URL` |
| `workers` | 1 | array size / `taskCount` / `completions` |
| `parallelism` | `workers` | workers running at once, capped at `workers` |
| `gpus_per_worker` | the machine's GPU count on AWS and GCP; required on k8s and Slurm | GPUs each worker is given |
| `worker_gpus` | `all` | the worker's `--gpus` |
| `exit_when_empty` | true | add `--exit-when-empty`, so a worker exits and frees its machine once the queue is drained |
| `worker_args` | `[]` | extra worker flags. `--queue`, `--gpus` and `--exit-when-empty` are refused here |
| `spot` | false | spot or preemptible capacity |
| `geog_url`, `highres_cache_url` | none | sources the entrypoint stages (see above) |
| `max_runtime_hours` | 72 | per-attempt limit (Batch timeout, `maxRunDuration`, `activeDeadlineSeconds`, `--time`) |
| `retries` | 3 | retries after the first attempt (0-9). A worker that exits 2 (a refusal) is never retried |
| `env` | `{}` | extra environment variables (see the credential rules below) |
| `labels` | `{}` | resource labels/tags. Lower-case keys and values |

### `aws-batch`

| File | Consumed by |
|---|---|
| `launch-template.json` | `aws ec2 create-launch-template --cli-input-json`. MIME user data that RAID0s the instance-store NVMe drives to `/scratch` (or, with `nvme_scratch = false`, only creates it) and makes it writable for uid 10001. The readable copy is `scratch-userdata.sh` |
| `compute-environment.json` | `aws batch create-compute-environment`. `MANAGED`; `EC2` with `BEST_FIT_PROGRESSIVE`, or `SPOT` with `SPOT_PRICE_CAPACITY_OPTIMIZED`; `ECS_AL2023_NVIDIA` AMI; `minvCpus` 0, so it scales to zero |
| `job-queue.json` | `aws batch create-job-queue` |
| `job-definition.json` | `aws batch register-job-definition`. GPU, VCPU and MEMORY resource requirements: a worker given g of an instance's G GPUs asks for g/G of its vCPUs and of 90 % of its memory (the smallest listed type decides), so G/g workers pack onto one instance; `/scratch` host volume; up to 64 GiB `/dev/shm`; retry on `Host EC2*` (spot reclaim), exit on 2 |
| `submit-job.json` | `aws batch submit-job`. An array job of `workers` children (at most 10,000) |
| `submit.sh` | the five calls above, in order. It waits up to 15 minutes for the compute environment to be `VALID` and stops with its `statusReason` if it turns `INVALID` |

`[aws_batch]` keys:

- `region`
- `instance_types` (default `["p5.48xlarge"]`)
- `subnets` and `security_group_ids`
- `instance_role_arn`: the ECS instance profile
- `job_role_arn`: the role the worker uses for S3
- `service_role_arn`: optional
- `max_vcpus`: default `parallelism` × the largest type's vCPUs
- `nvme_scratch` (default true) and `root_volume_gib` (default 200)
- `instance_shapes`: declares a type outside the built-in table

The built-in table covers:

| Family | GPU |
|---|---|
| `p5`, `p5e`, `p5en` | H100 / H200 |
| `p4d`, `p4de` | A100 |
| `g6e` | L40S |
| `g5` | A10G |
| `g6` | L4 |

A type with fewer GPUs than `gpus_per_worker` is refused.

### `gcp-batch`

`job.json` is a Batch `Job`, submitted by `submit.sh` with
`gcloud batch jobs submit NAME --location REGION --config job.json`:

- `taskCount` = `workers`, `parallelism`, and `taskCountPerNode: 1`.
- A first script runnable makes the scratch disk writable for uid 10001.
  The container runnable then mounts it at `/scratch` with a 64 GiB
  `/dev/shm`.
- `installGpuDrivers: true`. `accelerators` matches the machine type
  (`a3-highgpu-*`, `a3-megagpu-8g`, `a2-highgpu-*`, `a2-ultragpu-*`,
  `g2-standard-*`).
- `provisioningModel` is `SPOT` or `STANDARD`.
- A lifecycle policy fails a task that exits 2 instead of retrying it.

`[gcp_batch]` keys:

- `project` and `region`
- `machine_type` (default `a3-highgpu-8g`)
- `service_account_email`: the identity the VMs and workers run as
- `network`, `subnetwork`, `no_external_ip`
- `boot_disk_gb`, `scratch_disk_gb`, `scratch_disk_type` (default
  `pd-balanced`, which every A3/A2/G2 machine supports)

A `gpus_per_worker` that differs from the machine's GPU count is refused,
because one worker runs per VM.

### `k8s`

`job.yaml` holds two documents:

- **ServiceAccount**: carries the workload-identity annotation.
  `cloud = "aws"` writes `eks.amazonaws.com/role-arn` (IRSA) and
  `cloud = "gcp"` writes `iam.gke.io/gcp-service-account`.
- **Indexed Job**:
  - `completions` = `workers`, plus `parallelism`
  - `backoffLimitPerIndex` = `retries`
  - a pod failure policy that fails an index on exit 2 and ignores spot
    disruptions (`DisruptionTarget`)
  - `nvidia.com/gpu` requests and limits, the `nvidia.com/gpu` toleration,
    and `node_selector` plus the provider's spot label when `spot = true`
  - `runAsNonRoot` as uid 10001, with all capabilities dropped
  - scratch on an `emptyDir` or `scratch_host_path` (local NVMe), and a
    memory-backed `/dev/shm`. `fsGroup` does not apply to a `hostPath`, so
    with `scratch_host_path` a root init container (only `FOWNER`) runs
    `chmod 1777 /scratch` first

`submit.sh` creates the namespace if it is missing, then runs
`kubectl apply`. Backoff per index and pod failure policies need
Kubernetes 1.29 or later.

`[k8s]` keys:

- `cloud`: `aws`, `gcp` or `generic`
- `namespace`, `service_account`, `create_service_account`
- `iam_role_arn` or `gcp_service_account`
- `node_selector`, `tolerations`
- `cpu`, `memory_gib`
- `scratch_host_path`, `scratch_size_gib`, `shm_gib`
- `ttl_seconds_after_finished`

`cloud = "aws"` needs `s3://` URLs and `cloud = "gcp"` needs `gs://`.
`file://` URLs are refused on Kubernetes, and so is `spot = true` with
`cloud = "generic"`: put your cluster's spot label in `node_selector`
instead.

### `slurm`

`worker.sbatch` is an array job, `--array=0-(N-1)%parallelism`. Each task
gets one node (`--nodes=1 --ntasks=1 --exclusive --mem=0`) and
`--gres=gpu:N`, and runs one worker that uses every GPU Slurm gave it.
`--requeue` and `--signal=B:TERM@300` let a preempted node hand its job
back to the queue's lease. Three runtimes are available:

| `runtime` | Launch |
|---|---|
| `apptainer` (default) | `apptainer exec --nv --bind SCRATCH:/scratch IMAGE woof-entrypoint woof energy worker ...`. Set `sif_path` to a SIF on shared storage (`apptainer pull`), or every task converts `docker://IMAGE` itself |
| `pyxis` | `srun --container-image=registry#repo:tag --container-mounts=SCRATCH:/scratch woof-entrypoint ...` |
| `native` | `pixi run --manifest-path WOOF_ROOT/pyproject.toml -e default bash WOOF_ROOT/deploy/entrypoint.sh ...` on a shared checkout (`woof_root`) |

Other `[slurm]` keys:

- `partition`, `account`
- `time_limit`: default derived from `max_runtime_hours`
- `cpus_per_task`
- `pixi_env`
- `scratch_dir`: node-local; shell expansions allowed (default
  `/tmp/woof-energy-${USER}`). It is the same directory for every job on a
  node, so WPS_GEOG is staged once per node and reused
- `log_dir`, `exclusive`
- `extra_directives`: whole `--option=value` lines

The entrypoint replaces an inherited `TMPDIR` that the container cannot
create (a host path Apptainer or Pyxis forwarded) with `$WOOF_SCRATCH/tmp`.
Slurm accepts `file://` queue and output URLs on a shared filesystem.
Arrays larger than Slurm's default `MaxArraySize` (1001) produce a note.

## Credentials and placeholders

No file this command writes contains a credential. Workers authenticate
with the identity of where they run:

| Backend | Identity |
|---|---|
| AWS Batch | `job_role_arn`, delivered as ECS task-role credentials; the instance profile only runs the ECS agent |
| GCP Batch | `service_account_email` on the VMs |
| Kubernetes | IRSA or GKE Workload Identity on the ServiceAccount |
| Slurm | the nodes' instance role or VM service account |

The configuration is refused if:

- an `env` name looks like a credential (`SECRET`, `TOKEN`, `PASSWORD`,
  `ACCESS_KEY`, `CREDENTIAL`, `API_KEY`, ...);
- an `env` value holds an AWS access key id or a PEM block;
- a URL carries userinfo;
- `env` sets one of the variables the renderer owns.

On `aws-batch` and `gcp-batch` every URL must use that cloud's own scheme.
A cross-cloud copy would need a key in the job, and these files never
carry one.

If an account-specific value is missing (subnet, security group, role ARN,
region, project, service account, `woof_root`), it is written as
`REPLACE_ME_<WHAT>` and listed in the summary and in `deploy.json`. Each
`submit.sh` refuses to run (exit 2) while any `REPLACE_ME_` remains in its
directory. `--strict` refuses at render time instead.

## Sizing for the year-long run

Each worker runs one queue job per GPU. Total throughput therefore scales
with the number of GPUs, and `workers` × GPUs per worker is the number of
jobs in flight.

- **Spot.** Use spot capacity with `exit_when_empty = true`. The queue's
  leases make a reclaimed worker's job reappear for another worker, and
  `retries` re-launches the worker itself.
- **Machine size.** An 80 GB H100/A100 node (p5, p4de, a3, a2-ultra) fits
  the 100-300 m wind-farm nests with the most headroom per job. L40S, A10G
  and L4 machines suit smaller farms at lower cost.
- **Scratch.** Put scratch on local NVMe. The entrypoint stages WPS_GEOG
  there once per node, and run directories are written there before the
  worker uploads to Icechunk.
