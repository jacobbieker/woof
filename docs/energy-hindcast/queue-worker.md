# Hindcast work queue and GPU worker

A year-long wind-farm reanalysis is tens of thousands of independent
jobs (one `woof-energy.hindcast-job.v1` document each, with a
deterministic `job_id`).  `woof energy queue` keeps them in an object
store and `woof energy worker` drains them, one loop per GPU, on as many
nodes as you start.  The bucket is the only thing the nodes share: no
database, no shared filesystem, no coordinator.

```text
woof energy queue init   s3://bkt/wf2024/queue          queue.json (settings)
woof energy queue push   s3://bkt/wf2024/queue jobs/    jobs/<id>.json
woof energy worker       s3://bkt/wf2024/queue          on every GPU node
woof energy queue status s3://bkt/wf2024/queue          JSON counts by state
```

Code: `woof/energy/workqueue.py` (queue and backends) and
`woof/energy/worker.py` (worker).  Tests: `tests/test_energy_hindcast_queue.py`.

## Backends

| URL | Create-only write (claim, completion) | Compare-and-swap (renew, settle) | Needs |
|---|---|---|---|
| `file:///abs/dir` | staged file `link(2)`ed into place under a per-key `flock` | `flock`, digest compare, `rename(2)` | a local or POSIX-shared directory |
| `s3://bucket/prefix` | `PutObject` with `If-None-Match: *` | `PutObject` with `If-Match: <ETag>` | `boto3`; AWS credentials from the environment |
| `gs://bucket/prefix` | upload with `ifGenerationMatch=0` | upload with `ifGenerationMatch=<generation>` | `google-cloud-storage`; Application Default Credentials |

A missing client library is a named refusal (`QueueBackendUnavailable`)
naming the package to install; nothing falls back to another backend.
S3-compatible stores work if they implement conditional writes
(`AWS_ENDPOINT_URL_S3` points boto3 at them).

## Layout

```text
<queue>/queue.json          woof-energy.queue.v1         max attempts, lease, heartbeat, backoff
<queue>/jobs/<id>.json      the job document as pushed   immutable
<queue>/leases/<id>.json    woof-energy.queue-lease.v1   state, owner, expiry, attempt, failures, preemptions
<queue>/done/<id>.json      woof-energy.queue-done.v1    written once; the runner's result.json embedded
<queue>/failed/<id>.json    woof-energy.queue-failed.v1  every charged failure with its error tail
<queue>/dead/<id>.json      woof-energy.queue-dead.v1    gave up; only requeue revives it
<queue>/logs/<id>/attempt-NNN-<claimed>-<worker>.log     the runner's output, per attempt
```

## Guarantees

- **One claim per lease version.**  Claiming a fresh job creates
  `leases/<id>.json` create-only; taking over an expired, released or
  backed-off lease is a compare-and-swap on the version the claimant
  read.  Two workers cannot both win either write.
- **Completion happens exactly once.**  `done/<id>.json` is create-only.
  If a lease expires under a slow or partitioned worker and a second
  worker runs the job too, whichever finishes first writes the marker;
  the other's record goes to `logs/<id>/duplicate-*.json`.  A heartbeat
  that sees the job done elsewhere cancels its own copy.
- **Leases expire.**  A worker renews every `heartbeat_s` (at most half
  of `lease_s`).  A worker that stops renewing -- crashed, killed, host
  lost -- loses the job `lease_s` after its last renewal; the next
  claimant charges one failure (`lease-expired`) and runs it again.
- **Bounded retries.**  A charged failure (runner exit not 0, timeout,
  expired lease) waits `min(backoff_max_s, backoff_base_s * 2^(failures-1))`
  before any worker takes the job again.  At `max_attempts` charged
  failures the job is dead.  Runner exit 2 (woof's refusal) is dead at
  once: retrying a refusal does not change it.
- **Preemption is free.**  SIGTERM (spot reclaim, preemptible VM,
  `systemctl stop`) makes the worker SIGTERM its runner, wait
  `--term-grace-s` (default 20 s, inside GCP's 30 s preemption notice;
  AWS gives 2 minutes), SIGKILL it, and release the lease uncharged.  The
  next worker takes it at once.
- **Clocks.**  Expiry uses each host's wall clock, so hosts need NTP;
  the defaults (15 min lease, 2 min heartbeat) leave minutes of margin.

## Setting up a queue

```bash
woof energy queue init s3://bkt/wf2024/queue --max-attempts 3 --lease-s 900
woof energy queue push s3://bkt/wf2024/queue jobs/
```

`init` is idempotent with the same settings and refuses different ones
(`QueueConfigConflict`).  `push` takes files or directories of `*.json`,
accepts only `woof-energy.hindcast-job.v1` documents unless `--schema`
names another, and is idempotent: a job already queued with the same
document is `already-queued`; a different document under the same id is
refused (exit 2), because a changed job needs a new id.

## Running workers

On each GPU node:

```bash
woof energy worker s3://bkt/wf2024/queue --gpus all --exit-when-empty
```

The worker starts one slot process per selected GPU (`--gpus all`, the
first `N`, or a comma list of UUIDs).  Each slot:

1. holds a slot lock for its GPU UUID (`energy-worker-gpu-<digest>.lock`
   under `WOOF_GPU_LOCK_ROOT`), so two workers never share a card;
2. runs with `CUDA_VISIBLE_DEVICES=<uuid>` and its own temp, CuPy and
   CUDA caches (the environment `woof multi-run` gives its children);
3. waits while another woof process holds the card's supervisor lock (a
   forecast started outside the queue);
4. claims a job, writes the job document to its work directory and runs
   the runner with the lease renewed in the background;
5. uploads the runner's log and marks the job done, failed or released.

The runner's own supervisor takes the card's UUID lock; the slot lock is
a separate file so it does not refuse its own job.

If another woof process takes the card between the claim and the
runner's start, the slot releases the job uncharged and waits.

Stop conditions: `--exit-when-empty` exits when every job is done or
dead (workers stay while other workers' leases could still expire);
`--max-idle-s` exits after that long with nothing to claim.  A slot whose
jobs fail 5 times in a row stops claiming (a broken runner or a sick GPU
should not burn through the queue).  A slot that crashes is restarted up
to 3 times; a slot whose parent worker died stops claiming after its
current job.  A second SIGTERM/SIGINT to the worker SIGKILLs the slots
and their runners; their leases then expire and are charged.

Queue writes that settle a job (done, failed, released) are retried
through transient store errors (up to 8 tries, waits capped at 30 s); a
retried completion that finds its own marker counts as completed.  Each
claim reads at most 64 held leases, sampled at random, so a fleet polling
a queue with thousands of running jobs does not read every lease on every
poll.

The summary on stdout is `woof-energy.worker-summary.v1` with per-slot
counts (`completed`, `duplicate`, `superseded`, `released`,
`failed-backoff`, `failed-dead`, `failed-lost`).  Exit codes: 0 drained
or idle, 1 a slot failed, 2 refused, 75 stopped by SIGTERM, 130 stopped
by SIGINT.

### The runner

The default runner is the hindcast door:
`{python} -m woof energy hindcast run-local {job}`.  A worker whose
install lacks that door refuses to start (exit 2) rather than let every
job exit 2 and go dead.  `--runner-cmd`
replaces it with any command template; placeholders are `{python}`,
`{job}` (local copy of the job document), `{job_id}`, `{attempt}`,
`{workdir}`, `{result}`, `{gpu_uuid}` and `{queue}`; any other `{name}`
is refused.  The runner also sees `WOOF_ENERGY_JOB_ID`,
`WOOF_ENERGY_JOB_ATTEMPT`, `WOOF_ENERGY_JOB_PATH`,
`WOOF_ENERGY_RESULT_PATH` and `WOOF_ENERGY_QUEUE`.  JSON the runner
writes to `{result}` is embedded in `done/<id>.json` (for example the
Icechunk snapshot it committed).  `--timeout-s` (default 24 h) bounds
each run.

Programs embedding the worker pass any callable taking a
`RunnerRequest` and returning a `RunnerOutcome` to `Slot(runner=...)`.

## Status and requeue

```bash
woof energy queue status s3://bkt/wf2024/queue --jobs
woof energy queue requeue s3://bkt/wf2024/queue --all-dead
woof energy queue requeue s3://bkt/wf2024/queue --job wf-hornsea-2024-03
```

`status` (`woof-energy.queue-status.v1`) counts `pending`, `running`,
`expired` (reclaimable), `backoff`, `done` and `dead`, lists the active
workers with their lease time left, and says whether the queue is
`drained`.  `requeue` revives dead jobs (failure count restarts;
`failed/` keeps the history) and clears a backoff wait; it refuses done
jobs (completion is exactly once) and running ones (that would start a
second copy).

## Testing without a GPU

`--no-gpu` runs CPU slots (`--slots N`) with `CUDA_VISIBLE_DEVICES=-1`
and `GPUWM_NO_LOCAL_GPU=1`.  `--fake-gpus N` stands in N GPU identities
without reading the device; point `WOOF_GPU_LOCK_ROOT` at a temporary
directory so the stand-in locks stay out of the machine's lock root.  The
end-to-end test starts three workers on a `file://` queue with a fake
runner, SIGKILLs the worker holding a lease, and checks that the lease
expires, the job is charged one `lease-expired` failure, runs again and
completes exactly once.
