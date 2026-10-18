# Running a fleet of track runs on AWS

This page covers running many independent `woof go` forecasts on AWS Batch.
Each forecast is one *track run*, for example a moving nest that follows a
given path, and each writes its own Icechunk repository. Every run is a
**job document**. `woof go-job` runs one job document on one GPU. Two AWS
Batch fleets of `woof energy worker` drain the job queues:

- an **L40S tier** (`g6e.xlarge`, 48 GB) for most runs;
- an **H100 tier** (`p5.48xlarge`, 80 GB) for the few runs that need more
  than 44 GiB.

`YOUR-BUCKET`, the account id `123456789012`, the subnets, the security
group and the role names on this page and in
`deploy/examples/aws-track-runs.toml` are placeholders. Replace them with
your own. Put the fleet in the region of your bucket, so that no data
crosses a region. The examples use `us-west-2`.

```
job documents ──push──> s3://YOUR-BUCKET/woof-queue/track-runs/l40s/ ──> g6e.xlarge workers ─┐
                └─push──> s3://YOUR-BUCKET/woof-queue/track-runs/h100/ ──> p5.48xlarge workers ┤
                                                                                                v
                                                     s3://YOUR-BUCKET/woof-outputs/track-runs/<job_id>.icechunk
```

Each worker claims one job, runs
`woof go-job JOB --workdir DIR --result FILE`, and records the result in the
queue. `woof go-job` does these things in order:

1. It copies the config and its companion files onto the instance's NVMe.
2. It runs `woof go CONFIG --icechunk URL --products none`. The Icechunk
   export runs beside the forecast and commits one snapshot per history
   frame.
3. If the forecast finishes but the export does not, it resumes the export
   from the run folder on local disk. It tries up to three times, waiting
   30 s, then 2 min, then 5 min.
4. It writes a small result JSON. The JSON holds the status, the exit codes,
   the timings, the store URL, the last committed frame and the last
   snapshot.

None of the files, commands or examples on this page holds a credential.
Workers use the Batch job role. Commands you run from a workstation use
your own `AWS_PROFILE` or SSO session.

**Inputs never go through the repository.** Run names, input URLs and
output URLs come only from the job documents and the command-line
arguments you give. The configs, their companion files and any path data
reach the instances at run time, in one of the two ways in section 2.

## 1. Job documents

A job document is a JSON file with the schema `woof.go-job.v1`. Here is one
run:

```json
{
  "schema": "woof.go-job.v1",
  "job_id": "run-001",
  "config": "s3://YOUR-BUCKET/inputs/run-001.toml",
  "files": [
    {"url": "s3://YOUR-BUCKET/inputs/run-001.namelist.wps"},
    {"url": "s3://YOUR-BUCKET/inputs/eta-256.txt", "path": "eta-256.txt"}
  ],
  "icechunk_url": "s3://YOUR-BUCKET/woof-outputs/track-runs/run-001.icechunk",
  "result_url": "s3://YOUR-BUCKET/woof-outputs/track-runs/run-001.result.json",
  "resources": {"gpu_mem_gib": 38.5}
}
```

| Key | Required | Meaning |
|---|---|---|
| `schema` | yes | `woof.go-job.v1` |
| `job_id` | yes | the run's name. It becomes an object key, so it may use only `A-Z a-z 0-9 . _ -` |
| `config` | one of these two | an `s3://`, `https://` or `file://` URL, or a path. A relative path or URL is resolved against the job document's own location |
| `config_toml` | one of these two | the config TOML itself, inline |
| `config_name` | no | the file name the config is staged as. The default is the URL's file name, or `<job_id>.toml` for inline TOML. `woof go` looks for companions such as `<stem>.namelist.wps` beside the config, so name them to match |
| `files` | no | companion files. Each one is a URL string, a `{url, path}` table, or a `{path, content}` table that carries the file's text inline. Each file is staged beside the config under `path`, which defaults to the URL's file name. `path` must be relative, with no `..` |
| `icechunk_url` | no | the repository the run writes (`s3://`, `gs://` or `file://`). The default is `$WOOF_ENERGY_OUTPUT_URL/<job_id>.icechunk`. A rendered deployment sets that variable from `output_url` |
| `products` | no | the value passed to `woof go --products`. The default is `none` |
| `go_args` | no | extra `woof go` flags, from a fixed list: `--cycle`, `--transport`, `--keep-checkpoints`, `--devices`, `--late-after-minutes`, `--verify-inputs`, `--whole-cycle`, `--no-memory-gate` and `--no-verify-visuals`. The flags go-job sets itself (`--icechunk`, `--products` and `--outdir`) are refused, as is any other flag |
| `result_url` | no | where a copy of the result JSON is written (`s3://`, `gs://` or `file://`) |
| `resources.gpu_mem_gib` | no | the run's peak GPU memory in GiB. Workers pack jobs by this value. go-job refuses a job that declares more memory than the card holds |
| `depends_on`, `description`, `labels` | no | passed through. `depends_on` holds the queue's job ids |

Any other key is refused by name. A misspelt `icechunk` cannot silently
send a run to the default store.

A URL that carries `user:password@` is refused, and so is a URL with a
query string, such as a presigned URL. Either one would put a credential in
the queue, in the logs and in the result.

**Which tier.** Push a job with `resources.gpu_mem_gib` of 44 or less to
the L40S queue, and a job with more than 44 to the H100 queue. An idle
worker admits any one job, whatever its size. If a 52 GiB run lands on the
L40S queue, go-job refuses it on the L40S card (exit 2), and the queue
marks it dead without running it. Requeue it on the H100 queue.

## 2. Getting the inputs to the instances

### a) Through an S3 input prefix you name

Push a local folder of configs and companions to a prefix the job role can
read. Then reference the resulting `s3://` URLs from the job documents:

```bash
woof go-job-push ./my-runs s3://YOUR-BUCKET/inputs/ --dry-run   # the manifest only
woof go-job-push ./my-runs s3://YOUR-BUCKET/inputs/
```

`woof go-job-push` uploads every file under the folder (dot files are
skipped) to `PREFIX/<relative path>`, using your own identity through
boto3, rclone or the `aws` CLI. It prints a manifest with each file's URL,
size and SHA-256. `aws s3 sync ./my-runs s3://YOUR-BUCKET/inputs/` does
the same upload without the manifest.

### b) Directly from this machine

- **Inline.** Put the config in `config_toml` and each companion or path
  file in a `{path, content}` entry of `files`. The job document then
  carries everything, and the queue delivers it to the instance with the
  job. This suits text inputs of a few megabytes at most. S3 queues hold
  each job as one object.
- **Copied to an instance.** Copy the inputs onto a running GPU instance
  (`scp`, or `aws ssm start-session` with port forwarding), and run
  `woof go-job job.json` there with local paths in the job document.
  `woof remote` drives an SSH-reachable host whose inputs are already in
  place.

## 3. Build and push the image

The image is the worker image from `deploy/Dockerfile`. It holds the pixi
environment, the native bridges, the tables, `rclone` and a pinned `boto3`,
which an `s3://` queue needs. Build it from the repository root and push it
to ECR in your region:

```bash
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGISTRY=$ACCOUNT.dkr.ecr.us-west-2.amazonaws.com
aws ecr create-repository --region us-west-2 --repository-name woof-energy
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin "$REGISTRY"
docker build -f deploy/Dockerfile -t "$REGISTRY/woof-energy:1.0.3-cu12" .
docker push "$REGISTRY/woof-energy:1.0.3-cu12"
```

The ECR login uses your own session. The token goes straight to
`docker login` on stdin and is not stored on this page or in any file.

Then set `image` in your copy of `deploy/examples/aws-track-runs.toml` to
the tag you pushed. Use a version tag or a digest, never `latest`. The
renderer refuses an image without a tag, and it refuses `latest`.

## 4. Stage the static data once

The container entrypoint copies these onto each instance's NVMe before the
first job:

```bash
aws s3 sync "$GPUWM_WPS_GEOG" s3://YOUR-BUCKET/woof-static/WPS_GEOG/
aws s3 sync ~/.cache/woof/highres-cache s3://YOUR-BUCKET/woof-static/highres-cache/
```

## 5. IAM

There are two roles. You create them once.

**The instance profile** (`instance_role_arn`) runs the ECS agent. Attach
the AWS-managed `AmazonEC2ContainerServiceforEC2Role` to it. That policy
includes ECR pull.

**The job role** (`job_role_arn`) is what `woof go-job`, `woof go` and the
worker use for S3. Its trust policy allows `ecs-tasks.amazonaws.com`. Its
permissions policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {"Sid": "QueueAndOutputs", "Effect": "Allow",
     "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
     "Resource": ["arn:aws:s3:::YOUR-BUCKET/woof-queue/track-runs/*",
                  "arn:aws:s3:::YOUR-BUCKET/woof-outputs/track-runs/*"]},
    {"Sid": "Inputs", "Effect": "Allow", "Action": ["s3:GetObject"],
     "Resource": ["arn:aws:s3:::YOUR-BUCKET/woof-static/*",
                  "arn:aws:s3:::YOUR-BUCKET/inputs/*"]},
    {"Sid": "ListOwnPrefixes", "Effect": "Allow", "Action": ["s3:ListBucket"],
     "Resource": "arn:aws:s3:::YOUR-BUCKET",
     "Condition": {"StringLike": {"s3:prefix": [
       "woof-queue/track-runs/*", "woof-outputs/track-runs/*", "woof-static/*", "inputs/*"]}}},
    {"Sid": "PublicForcing", "Effect": "Allow",
     "Action": ["s3:GetObject", "s3:ListBucket"],
     "Resource": ["arn:aws:s3:::noaa-hrrr-bdp-pds", "arn:aws:s3:::noaa-hrrr-bdp-pds/*",
                  "arn:aws:s3:::noaa-gfs-bdp-pds", "arn:aws:s3:::noaa-gfs-bdp-pds/*",
                  "arn:aws:s3:::us-west-2.opendata.source.coop",
                  "arn:aws:s3:::us-west-2.opendata.source.coop/*"]}
  ]
}
```

Add any other public bucket your runs' sources read. A signed request to
another account's bucket needs an allow in your own policy as well.

The queue makes its writes safe with conditional writes (`If-None-Match`
and `If-Match`), which a general-purpose bucket supports. It deletes each
lease object as the job settles, so the policy grants `s3:DeleteObject` on
the queue prefix.

## 6. Render the two fleets

```bash
woof energy deploy render --backend aws-batch \
  --config deploy/examples/aws-track-runs.toml --tier l40s -o deploy-out/l40s
woof energy deploy render --backend aws-batch \
  --config deploy/examples/aws-track-runs.toml --tier h100 -o deploy-out/h100
```

`aws-track-runs.toml` holds the shared settings (image, region, subnets,
security groups, roles, the output prefix, `WOOF_GEOG_URL` and the worker
command) and two `[tiers.*]` tables:

| Tier | Instance | GPUs per worker | Workers | Capacity | Queue |
|---|---|---|---|---|---|
| `l40s` | `g6e.xlarge` (1 × L40S 48 GB, 4 vCPU, 32 GiB) | 1 | 40 | spot | `s3://YOUR-BUCKET/woof-queue/track-runs/l40s/` |
| `h100` | `p5.48xlarge` (8 × H100 80 GB) | 1 | 2 | on demand, at most one instance | `s3://YOUR-BUCKET/woof-queue/track-runs/h100/` |

Every worker runs this command:

```
woof energy worker --queue QUEUE --gpus all --exit-when-empty \
  --runner-cmd "{python} -m woof go-job {job} --workdir {workdir} --result {result}" \
  --timeout-s 172800
```

The worker fills in `{job}`, `{workdir}` and `{result}` for each claimed
job. `--timeout-s` stops a job after 48 hours. Each tier renders its own
compute environment, job queue, job definition and array job, with names
prefixed `woof-track-runs-l40s-` and `woof-track-runs-h100-`.

If you delete the placeholder identifiers instead of replacing them, the
renderer writes `REPLACE_ME_` placeholders that `submit.sh` refuses to run
with. `--strict` turns a missing value into a refusal at render time.
Review the JSON files before you submit them.

The host memory on `g6e.xlarge` is 32 GiB, and the job definition asks for
90 % of it. If a run's preparation needs more, change the L40S tier's
`instance_types` to `["g6e.2xlarge"]` (64 GiB) or `["g6e.4xlarge"]`
(128 GiB) and render again.

## 7. Create the queues, push the jobs, start the fleets

The workstation that pushes jobs needs `boto3` (`pip install boto3`) and an
AWS session.

```bash
for tier in l40s h100; do
  woof energy queue init s3://YOUR-BUCKET/woof-queue/track-runs/$tier/ \
    --max-attempts 4 --lease-s 900
done
woof energy queue push s3://YOUR-BUCKET/woof-queue/track-runs/l40s/ jobs/l40s/ --schema woof.go-job.v1
woof energy queue push s3://YOUR-BUCKET/woof-queue/track-runs/h100/ jobs/h100/ --schema woof.go-job.v1

deploy-out/l40s/submit.sh
deploy-out/h100/submit.sh
```

`push` is idempotent. Pushing the same job again does nothing, and pushing
a changed job under the same id is refused. `submit.sh` creates the launch
template, the compute environment, the job queue and the job definition,
then submits the array of workers. Push the jobs first. A worker that
starts on an empty queue exits at once.

## 8. Monitor

```bash
woof energy queue status s3://YOUR-BUCKET/woof-queue/track-runs/l40s/ --jobs
aws batch list-jobs --region us-west-2 --array-job-id ARRAY_ID
aws s3 ls s3://YOUR-BUCKET/woof-outputs/track-runs/
```

- **Per job**: each attempt's full log (go-job, `woof go` and the exporter)
  is uploaded to the queue under `logs/<job_id>/`. A completed job's
  `done/<job_id>.json` embeds go-job's result: `last_committed_frame`,
  `last_snapshot`, `frames_exported` and `timings_s`. The job's
  `result_url` gets a copy of the same JSON.
- **Per worker**: CloudWatch Logs, log group `/aws/batch/job`.
- **Retrying the dead**: `woof energy queue requeue URL --all-dead`. The
  workers have exited by then (`--exit-when-empty`), so submit the array
  again with `aws batch submit-job --cli-input-json file://deploy-out/l40s/submit-job.json`.

## 9. Exit codes and retries

| go-job exit | Meaning | Queue | AWS Batch |
|---|---|---|---|
| 0 | the forecast and its export are complete | done | — |
| 2 | refused: a bad job document, a URL go-job cannot use, an input the store says does not exist, a card too small, frames the Icechunk store refuses, or `woof go`'s own refusal | dead at once | not retried |
| 73 | the forecast finished, and the export is still incomplete after three resumes | retried (the whole run starts again) | retried |
| 75 | stopped by SIGTERM, an input that could not be fetched after four tries, or a host with no transfer tool | released (SIGTERM) or retried | retried |
| other | `woof go`'s own status, or 128+N for a signal | retried | retried |

The worker itself exits 0 when the queue is empty. It exits 75 after a spot
interruption, and Batch retries it on `Host EC2*` (spot reclaim).

**What a retry costs.** A run's checkpoints live on the instance's NVMe. A
run retried on another instance starts again from its first step. The
store is resumable and idempotent. A frame written again from a later run
replaces the earlier one. So a retried run ends with the same store a
single clean run would have written. On the H100 tier a restart is
expensive, which is why that tier runs on demand.

## 10. One AWS Batch array job, without a queue

A list of runs can also go as one array job, with no queue: child N runs
entry N. Write a list document and put it in your input prefix:

```json
{"schema": "woof.go-job-list.v1",
 "jobs": ["run-001.json", "run-002.json", {"schema": "woof.go-job.v1", "job_id": "run-003", "...": "..."}]}
```

Each entry is a job document, or a path or URL to one. A relative entry is
resolved against the list's own location. `woof go-job LIST` reads
`$AWS_BATCH_JOB_ARRAY_INDEX` (or `--index N`). Submit with the rendered job
definition and override its command:

```json
{
  "jobName": "woof-track-runs-l40s-array",
  "jobQueue": "woof-track-runs-l40s-queue",
  "jobDefinition": "woof-track-runs-l40s-worker",
  "arrayProperties": {"size": 38},
  "containerOverrides": {"command": [
    "woof", "go-job", "s3://YOUR-BUCKET/inputs/runs-l40s.json"]}
}
```

The job definition's retry strategy applies to each child: exit 2 is never
retried, and any other failure is retried up to `retries`. Give each job a
`result_url` so that every child's outcome lands next to its store.

## 11. Trying it locally

go-job takes `file://` stores, `file://` queues and local paths. You can
exercise the whole route on one workstation GPU without touching S3:

```bash
woof go-job job.json --workdir /tmp/run-test      # icechunk_url = "file:///tmp/run-test.icechunk"
woof energy queue init file:///tmp/q
woof energy queue push file:///tmp/q jobs/ --schema woof.go-job.v1
woof energy worker file:///tmp/q --gpus all --exit-when-empty \
  --runner-cmd "{python} -m woof go-job {job} --workdir {workdir} --result {result}"
```

## 12. Cost (approximate)

These are on-demand prices in `us-west-2`, as recorded when this page was
written. **Verify them on the EC2 pricing page before launch.** Spot prices
vary by hour and by availability zone.

| Instance | GPUs | On demand, approx. |
|---|---|---|
| `g6e.xlarge` | 1 × L40S 48 GB | ~$1.86 / h |
| `g6e.2xlarge` | 1 × L40S 48 GB | ~$2.24 / h |
| `p5.48xlarge` | 8 × H100 80 GB | ~$55 / h (the whole instance, whichever GPUs are used) |

A rough fleet cost is (runs per tier) × (hours per run) × (rate), plus the
S3 storage of the stores. Inside one region there is no transfer cost. The
two H100 runs share one `p5.48xlarge`, and its six idle GPUs are paid for.
A single-GPU P5 size, where the region offers one, would cost less for two
runs: declare it under `[aws_batch] instance_shapes` and check its price
and quota first.

**Quotas.** Check these before you submit:

- "All G and VT Spot Instance Requests": 40 × `g6e.xlarge` needs 160 vCPUs.
- "Running On-Demand P instances": one `p5.48xlarge` needs 192 vCPUs. P5
  capacity can be scarce. EC2 Capacity Blocks for ML are one way to
  reserve it.
