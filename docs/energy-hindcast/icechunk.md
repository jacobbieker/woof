# Icechunk stores for the energy hindcast

The year-long wind farm hindcast runs as many independent jobs, each one
cluster or leaf domain over one time segment, on many GPUs and nodes. Each
job writes straight into a shared [Icechunk](https://icechunk.io)
repository on S3, GCS or a shared filesystem. Nothing is staged on the
node's disk, so local storage does not limit the run.
`woof/energy/icechunk_store.py` provides the storage factory, the two
writers and the verifier.

## Where a store lives

Every door that takes a repository accepts these locations:

| Location | Storage | Credentials |
| --- | --- | --- |
| `s3://bucket/prefix` | AWS S3 or S3-compatible (MinIO, Ceph) | `AWS_*` variables, shared profile, or instance/pod role |
| `gs://bucket/prefix` (or `gcs://`) | Google Cloud Storage | application-default credentials or workload identity |
| `file:///abs/path`, or a bare path | local or shared filesystem | none |
| `memory://name` | in-process, shared by name | none (tests, dry runs) |

- S3 region defaults to `AWS_REGION`, then `AWS_DEFAULT_REGION`.
- The S3 endpoint defaults to `AWS_ENDPOINT_URL_S3`, then
  `AWS_ENDPOINT_URL`. A custom endpoint turns on path-style addressing,
  and an `http://` endpoint turns on plain HTTP. MinIO needs both.
- `--anonymous` (`anonymous=True`) reads public buckets without
  credentials.
- A URL with a user or password in it is refused. Credentials come from
  the environment and are never written to a summary.

```python
from woof.energy.icechunk_store import storage_from_url, open_repository

storage = storage_from_url("s3://my-bucket/hindcast-2025",
                           endpoint_url="http://minio:9000")
repo = open_repository("gs://my-bucket/hindcast-2025")   # created if absent
```

Repositories this module creates split their manifests along `time` and
`Time` every 672 chunks (`MANIFEST_SPLIT_CHUNKS`). A commit therefore
rewrites only the manifests for the times its segment touched, not the
whole year's.

## Many writers, one branch

Every write runs through `commit_with_retry`:

1. The writes are applied to a fresh writable session at the branch tip.
2. If another job committed first, the commit raises `ConflictError`. The
   session is rebased with `ConflictDetector`, which succeeds when the two
   jobs wrote different chunks, and the commit is retried.
3. If the two jobs wrote the same chunk, the rebase fails. The session is
   thrown away and the writes are re-applied on a new session from the new
   tip. The read-modify-write of the shared chunk then sees the other
   job's values, so neither job's data is lost.
4. Before each retry the writer sleeps a random time between 0 and
   `min(20 s, 0.25 s × 2^n)`. After 12 attempts it gives up with
   `IcechunkStoreRefused`. These limits are set in `RetryPolicy`.

Object stores detect the race through their conditional writes, which
S3 and GCS provide. Icechunk's local filesystem storage does not: two
processes committing at once can both report success while one commit is
silently dropped. In a two-process test with icechunk 1.1, this happened
in 5 races out of 30. For a bare path or `file://` store, every writer in
this module therefore holds an exclusive `flock` on
`<repo>.woof-commit.lock`, beside the repository, while it commits. The
shared filesystem must support `flock`: local disks, NFSv4, and Lustre
mounted with `-o flock` do. Writers outside this module that commit to
the same filesystem repository must take the same lock.

Sessions that were thrown away leave chunks that no snapshot references.
Nothing can read them, and `Repository.garbage_collect` removes them.
Shared chunks are always safe but cost a re-write. Chunking that gives
each job whole chunks (see `--site-chunk` and `--time-chunk`) avoids the
re-writes.

## Site store: `(time, site, height)` for the year

`woof energy icechunk init` allocates the store once, before any job runs:

```bash
woof energy icechunk init s3://my-bucket/hindcast-2025 --sites sites.json --start 2025-01-01T00:00 --end 2025-12-31T23:45 --interval 15min --site-chunk 64
```

- `time` runs from `--start` to `--end`, both included, every
  `--interval`. The interval can be finer than 15 min.
- `site` is in `sites.v1` order, with the site coordinates from that
  document.
- `height` is the sites document's heights, or `--heights-m`.
- Every `forecast.v1` variable is created as a float32 array filled with
  NaN, or only those given with `--vars`. Only metadata is written; no
  data chunk exists until a job writes one.
- `written(time, site)` is 1 where a job has written.
- Chunks are one day of steps × `--site-chunk` sites × all heights.
  Compression is Blosc zstd with bitshuffle.

Running `init` again with the same axis, sites, heights and variables does
nothing (`"created": false`). Running it with any of those changed is
refused rather than replacing the store.

Each job then writes its own times and sites:

```bash
woof energy extract plan.json --format icechunk --icechunk-mode region --job-id leaf-17-2025-03 -o s3://my-bucket/hindcast-2025
```

`write_site_region` writes by orthogonal selection, so it touches only the
chunks that hold the job's points. It also does the following:

- sets `written` for those points;
- fills the per-site coordinates the job owns (`terrain_height`,
  `domain_id`, `dx_m`, `inside`);
- commits with metadata: job id, segment id, first and last time, host,
  and the job's notes. Notes go into the commit metadata, not into
  attributes, because two jobs changing the same attribute would conflict.

It refuses the write if:

- a time is not on the axis;
- a site is not in the store;
- the heights differ from the store's;
- a variable was not allocated at `init`.

Re-running a job is idempotent. `overwrite`, the default, writes the same
values again. `on_existing="skip"` writes nothing when every point is
already written.

Without `--icechunk-mode region`, `extract --format icechunk` behaves as
before: the whole forecast replaces the repository at a local path. It now
also accepts the URLs above.

## Gridded store: wrfout frames per domain

`tools/wrfout_to_netcdf.py` writes one segment of one domain's wrfout
frames into a group, one group per cluster/leaf domain:

```bash
python tools/wrfout_to_netcdf.py runs/leaf-17/seg-03 --icechunk-repo s3://my-bucket/hindcast-2025-grid --icechunk-group leaf-17/d02 --time-axis 2025-01-01T00:00,2025-12-31T23:45,15min --window 2025-03-01T00:00,2025-04-01T00:00 --compress-zstd-bitshuffle --job-id leaf-17-seg-03
```

`write_gridded_segment` does the following:

- **Allocates the axis on first use.** The first segment to reach a group
  creates it on the full `--time-axis`. Later segments must give the same
  axis or none.
- **Keeps the seed-and-NaN-fill rule of the old streaming export.** The
  variable set is the union over the creating segment's frames. A field
  WRF only starts writing after t=0 (such as `REFL_10CM`) is stored NaN
  where it is missing, not dropped. A field the store does not hold is
  dropped from later frames and listed in `dropped_variables`.
- **Replaces `Times` and `XTIME` with the `Time` coordinate.** `XTIME`
  counts from each segment's own start, so it has no single meaning on a
  shared axis.
- **Records progress.** `frame_written(Time)` is 1 for every frame
  written.
- **Filters by window.** `--window FIRST,END` keeps frames with
  `FIRST <= time < END`, which drops a segment's spin-up or its overlap
  with the next segment. The window is applied first; a frame off the
  axis, or a valid time found in two files inside the window, is then
  refused.
- **Makes re-runs idempotent.** Frames already written are skipped
  (`--on-existing skip`, the default) or rewritten (`overwrite`).
- **Compresses and chunks.** With `--compress-zstd-bitshuffle`, numeric
  arrays use Blosc zstd with bitshuffle (`--zstd-clevel`), as in the
  tool's other modes. The Python function compresses by default. Time chunks are `--time-chunk` frames (default 1, so
  two segments never share a chunk). Other dimensions are whole unless a
  chunk would exceed 64 MiB (`MAX_CHUNK_BYTES`). The time chunk is fixed
  when the group is created. The summary reports `chunk_aligned: false`
  when a segment shares time chunks with its neighbours under that
  chunking.

Without `--icechunk-group`, the whole-directory export and the
`--watch-icechunk` stream behave as before, and also take a URL in
`--icechunk-repo`.

## Checking a store

```bash
woof energy icechunk verify s3://my-bucket/hindcast-2025 --anonymous
```

`verify_repository` opens the repository, with credentials or anonymously,
and reports:

- the branch tip snapshot id;
- the last commit's message, time and metadata;
- for each site or gridded store group (or only `--group`): the axis, the
  number of steps written, and the first and last written times.

`--no-coverage` skips reading the written masks.

## Schemas

| Schema | What |
| --- | --- |
| `woof-energy.icechunk-sites.v1` | site store group attribute `schema` |
| `woof-energy.icechunk-gridded.v1` | gridded group attribute `schema` |
| `woof-energy.icechunk-write.v1` | commit metadata and writer summaries |
| `woof-energy.icechunk-verify.v1` | `verify` report |

## Tests

`tests/test_energy_hindcast_icechunk.py` runs offline against `memory://`
and `file://` stores. It covers:

- two processes racing to create a group and writing two segments;
- two processes forced to collide on a shared site chunk, with neither
  job's writes lost;
- idempotent re-runs;
- the CLI doors.

The S3 round trip runs only when `WOOF_S3_TEST_URL=s3://bucket/prefix` is
set. It uses credentials from the environment and writes under a fresh
random prefix.
