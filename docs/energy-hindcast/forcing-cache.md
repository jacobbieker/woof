# Forcing cache: fetch each cycle once

A year-long reanalysis of every public wind farm is about 440 regional
clusters times 1,460 six-hourly cycles: some 640,000 independent jobs.
Every cluster of a cycle forces its run from the same global cycle (Met
Office global, IFS, ICON, GFS) or, over the CONUS, the same HRRR/RAP
cycle. Left alone, each job's `woof go` downloads its own copy into its
own managed download folder, so one Met Office cycle (3.2 GB for a 6 h
window, 41 objects) is downloaded 440 times, about 2,000 TB over the
year. The forcing cache moves that download to one **prefetch** per
(source, cycle) into a shared object store, and each forecast job
**mirrors** only the objects its own fetch would download into the folder
`woof go` will use. `woof fetch` then finds every object verified and asks
no host.

Code: `woof/energy/forcing_cache.py`, wired into `woof/energy/hindcast.py`
(plan, run-local, `prefetch`, `mirror`), `woof/energy/workqueue.py`
(dependencies) and `woof/energy/cost.py` (the `forcing` section).
`woof.fetch_routes.adopt_verified_objects` writes the per-object receipts
`woof fetch` itself writes. Tests: `tests/test_energy_hindcast_forcing_cache.py`.

```text
woof energy hindcast plan ... --forcing-cache s3://bkt/wf2024/forcing -o hc
    -> jobs/prefetch-<source>-<YYYYMMDDTHH>Z-<sha12>/job.json   one per (source, cycle)
    -> jobs/<cluster>-<YYYYMMDDTHH>Z-<sha12>/job.json           forcing + depends_on
woof energy hindcast prefetch hc            # or: the queue runs prefetch jobs first
    -> s3://bkt/wf2024/forcing/entries/<source>/<YYYYMMDDTHH>Z/<key16>.json
    -> s3://bkt/wf2024/forcing/blobs/sha256/<aa>/<sha256>
woof energy hindcast run-local JOB          # mirrors, then woof energy run -> woof go
    -> plan/runs/<domain>/downloads/<request key>/...           exactly woof fetch's layout
    -> jobs/<job>/forcing-mirror.json
```

## What is shared, and what is not

| Source | Fetch | Shared? | Why |
|---|---|---|---|
| `met-office-global`, `ecmwf-open-data`, `hrrr-native`, `rap`, `icon-global`, `icon-global-ml`, `aifs`, `gem-gdps`, every other table route | whole published objects (or record subsets) named by the route table | **yes** | the request carries no area: every cluster of a cycle asks for byte-identical files |
| `gfs`, `gdas` | NOMADS grib filter cropped to the cluster's area; archive full-file objects for old cycles | no | the filter crops per cluster, and the full-file receipt binds the area (`woof.fetch.check_prior_request` refuses another cluster's folder) |
| `hrrr` (legacy transport) | `.idx` byte ranges | no | its receipt binds the requested area |
| `era5` | CDS retrieval cropped to the area; a declared input | no | the retrieval is per cluster |

`woof.energy.forcing_cache.shareability(source)` is the one answer. Jobs
of a source that is not shared carry `forcing.mode = "direct"` with the
reason, have no dependency, and fetch exactly as before. The default
source selector puts `hrrr-native`, `rap` and `met-office-global` (all
table routes) ahead of `gfs`, so in practice nearly every job is served
from the cache.

### No lat/lon cropping

The published objects are global GRIB2 or NetCDF-4 grids. A lat/lon window
of one is not a byte range of it: cutting a window out means decoding and
re-encoding, which changes what woof's decoders read. The cache does not do
it. What it does subset safely is by **object**: a cache entry may hold more
objects than a job needs (a longer window, another member's files), and the
job takes exactly the objects of its own resolved fetch plan
(`woof.fetch_routes.resolve_request` on the job's own `[fetch]` table).
Decoding is untouched: the job's preparation reads the same bytes at the
same relative paths it would have downloaded, and crops to its domain as it
always has.

## Planning

`--forcing-cache URL` on `woof energy hindcast plan` takes
`s3://bucket/prefix`, `gs://bucket/prefix`, `file:///abs/dir` or a bare
path (made an absolute `file://` URL; it only works across hosts on a
shared filesystem). Credentials in the URL are refused; the workers use
the instance role, workload identity or the environment.

For each job whose source is shared, the plan predicts the request its
segment's `[fetch]` table will make (the planner's own `fetch_window` and
default member, and the cycle through `parse_cycle`, the planner's own
calls) and groups jobs by request family (source, cycle, cadence,
forecast start hour, member). Each group gets one prefetch job for the
longest window any of its jobs needs:

```json
{"schema": "woof-energy.hindcast-job.v1", "kind": "prefetch",
 "job_id": "prefetch-ecmwf-open-data-20251101T00Z-f1039a3cfd69",
 "forcing": {"cache_url": "s3://bkt/wf2024/forcing",
             "entry": "entries/ecmwf-open-data/20251101T00Z/034d60b03d85e9b3.json",
             "request": {"source": "ecmwf-open-data", "cycle": "2025-11-01T00",
                         "hours": 6, "cadence": 3, "forecast_start_hour": 0,
                         "member": null, "transport": null},
             "argv": ["woof", "fetch", "--cadence", "3", "--cycle",
                      "2025-11-01T00", "--hours", "6", "--source",
                      "ecmwf-open-data"]},
 "segment": {"cycle": "2025-11-01T00:00:00Z"}, "resources": {"gpus": 0}}
```

and each forecast job of the group gets

```json
"forcing": {"mode": "shared", "cache_url": "...", "entry": "...", "key": "...",
            "request": {...}, "prefetch_job_id": "prefetch-ecmwf-open-data-..."},
"depends_on": ["prefetch-ecmwf-open-data-..."]
```

The prefetch id hashes the request and the cache URL, so re-planning
writes the same documents. `forcing` and `depends_on` are not part of a
forecast job's identity: its id is the same with or without a cache,
because the cache changes where forcing bytes come from, never what they
are. An existing `job.json` is never rewritten except for this
binding: planning an existing directory again with `--forcing-cache`
(`--resume`, since the request changed) rewrites `forcing` and
`depends_on` in the jobs already on disk whose every other field is what
the new plan computes, keeps a running job (or one planned from other
inputs) as it is, leaves it out of the prefetch jobs' `dependents` and
notes how many. A queue that already holds the old documents refuses the
changed ones (`JobConflict`); push into a new queue. The manifest keeps
prefetch jobs in their own `prefetch`
table (`source`, `cycle`, `entry`, `dependents`, `status`) with
`prefetch_counts`, so the forecast job table, its counts and its coverage
report are unchanged, and `status` adds a `prefetch` summary.

## Cache layout and the entry document

```text
<cache>/entries/<source>/<YYYYMMDDTHH>Z/<key16>.json   woof-energy.forcing-cache.v1
<cache>/blobs/sha256/<aa>/<sha256>                     file bytes, content-addressed
```

The key is the sha256 of the canonical JSON of the request. Blobs are
named by their content, so a blob is either the right bytes or absent:
re-runs skip blobs already present at the right size, a constant file
shared by several cycles is stored once, and two prefetchers of one cycle
cannot corrupt each other. The entry document is written create-only and
last, after every blob is uploaded; a reader that finds it can trust that
every blob it names was uploaded.

The entry (`woof-energy.forcing-cache.v1`) holds:

- `request`, `key`, `source`, `cycle`, `complete: true`
- `files`: each object the fetch moved (`kind: object`, with the route
  manifest's `fetch_entry`: name, bytes, sha256, url, endpoint, plus its
  role and lead), each file of a declared donor's `donor-<source>/` folder
  (`kind: donor`), and `fetch-manifest.json` and `SHA256SUMS` for
  provenance (`kind: receipt`); every one with `path`, `sha256`, `bytes`
  and `blob`
- `totals`: files, bytes, objects, object bytes
- `fetch`: the `woof fetch` argv, the route manifest schema and identity,
  the route table sha256, the endpoints that served it, the fetch seconds
- `provenance`: woof version, prefetch job id, host; `retrieved_utc`

The rest of a fetch's handoff (`inputs.txt`, `prep-arguments.json`,
`prep-command.txt`, `posting/`) names the prefetch's scratch folder and
is rewritten by each job's own fetch, so it is not cached. Cache entries
are not deleted by woof: give the bucket prefix a lifecycle rule (the cost
model assumes 7 days) once the cycles' jobs are done.

## Prefetch

`woof energy hindcast prefetch DIR` runs every prefetch job the manifest
does not call complete, one after another (`--cycle`, `--source`, `--job`
select; `--dry-run` prints each fetch, its entry and its measured download
size from `woof.download_budget`). Each one runs the `woof fetch` the
jobs' `woof go` fetch stage would run for that request (the argv
`woof.runplan._fetch_arguments_from_hints` spells from the same `[fetch]`
table) into a scratch folder in the job directory, checks the route
manifest it wrote (schema, source and cycle; every listed object present
with its recorded sha256), uploads, publishes the entry and deletes the
scratch. It writes `prefetch.json` (`woof-energy.forcing-prefetch.v1`:
`status`, files, bytes, uploaded and reused blobs, fetch seconds) and marks
the manifest record complete.

An entry already complete in the cache is not fetched again
(`status: cached`); `--refetch` publishes it again. An entry whose blobs
went missing is fetched again and only the missing blobs are uploaded. A
prefetch job is also an ordinary job for `run-local` and the queue's
default runner (`woof energy hindcast run-local {job}` dispatches on
`kind`), and it runs from a job document carried away from its hindcast
directory. `--fetcher MODULE:CALLABLE` replaces the transport
(`fetcher(request, out)` must leave a table-route fetch folder in `out`).

## Mirror

Before `woof energy run`, `run-local` mirrors a shared job's forcing
(`woof energy hindcast mirror JOB` does it alone, after materializing).
For each `woof go` step of the plan (`woof.energy.run.build_steps`):

1. Read the config's `[fetch]` table. A config with `[case_data]`, a
   local source root or a source that is not shared is left alone.
2. Compare its request with the cache entry's. Another source, cycle,
   cadence, start hour or member, or more hours than the entry holds, is
   `request-differs`: nothing is mirrored and the job fetches for itself.
   The host is not compared: a job whose fetch pins one (`[fetch]
   transport`, or `WOOF_FETCH_POLICY=aws` on a hosted fleet) asks for the
   same keys at the same relative paths, its receipts name its own host,
   and its fetch re-verifies every object's sha256.
3. Compute the download folder `woof go` will choose:
   `woof.go_cli.managed_download_dir(<run_dir>, pin_request(config_fetch_request(payload), pinned))`,
   the same call with the same arguments `woof go` makes, so the folder
   is `<plan>/runs/<domain>/downloads/<request sha256>/` and `woof go`
   picks it without a `-1` sibling.
4. Resolve the request's fetch plan offline and, for each planned object,
   copy its blob to `<folder>/<relpath>`: written to a `.cache-part`
   file (hashed as it is copied from a `file://` cache, after the
   download from S3 or GCS), size and sha256 checked, renamed only when
   both match; a file already there with the right sha256 is kept. A
   declared donor's `donor-<source>/` folder is copied whole the same
   way.
5. Write the receipts `woof fetch` writes after a transfer
   (`fetch-recovery-request.json` and one `.fetch-verified/<sha256(relpath)>.json`
   per object, binding source, cycle, host, member and the object key),
   through `woof.fetch_routes.adopt_verified_objects`.

`woof go`'s fetch stage then runs `woof fetch` on that folder:
`request_cached` is true, so no publication probe and no listing; every
object is re-verified (size, magic, GRIB end marker or HDF5 end address,
sha256) and reused; `fetch-manifest.json`, `SHA256SUMS`, `inputs.txt` and
`prep-arguments.json` are written for this folder. A folder that already
holds the whole request (a rerun) is `already-local` and nothing is copied.

Fallback is per file. An object the entry does not hold, a blob that is
absent, or one whose bytes do not hash to the recorded sha256 is not
placed; `woof fetch` downloads exactly those, and `forcing-mirror.json`
lists each with the reason (`status: partial`, `fallback: direct-fetch`).
An entry that does not exist (`entry-missing`) falls back for the whole
config; inside a hindcast directory `run-local` first runs the job's
prefetch job inline and mirrors again, so `run-local` over a directory
fetches each cycle once even without `prefetch`. A cache that cannot be
read at all is never fatal: the receipt says `error` and the run fetches
everything itself. `--no-forcing-cache` skips the mirror.

`forcing-mirror.json` (`woof-energy.forcing-mirror.v1`) records, per
config: the data folder, the request, objects planned, mirrored, already
local, missing (path and reason), donor files, bytes copied, receipts
written; and totals. The manifest record of the job gets a `forcing`
summary.

## Scheduling on the queue

Push the prefetch jobs with the forecast jobs (`woof energy queue push`
pushes `kind: prefetch` documents of a batch first). The queue claims a
job only when every dependency that is queued is `done` or `dead`
([queue-worker.md](queue-worker.md#dependencies)), and tries `prefetch-`
jobs before others, so the fleet fetches a cycle before it runs the
cycle's 440 jobs. A dead prefetch (a refused or unreachable cycle) does
not hold its cycle hostage: its jobs become claimable and fetch for
themselves, recording `entry-missing`. A prefetch holds a worker slot for
its download only (`resources.gpus = 0`); there are about 1,460 of them
per source in a year against 640,000 forecasts.

## What it saves

`woof energy estimate` adds a `forcing` section (see [cost.md](cost.md)):
one cycle's download measured per object by `woof.download_budget`, then
per job versus per cycle. For the worldwide year (440 clusters, 1,460
cycles, Met Office global, 3.246 GB per cycle, the default assumptions of
100 MB/s from a publisher and 400 MB/s from a same-region cache):

| | per job (no cache) | per cycle (cache) |
|---|---|---|
| bytes from publishers | 2,085 TB | 4.74 TB |
| GPU-slot hours waiting on downloads | 5,792 | 1,461 (1,448 reading the cache, 13 prefetching) |
| requests | 26.3 M publisher GETs | 59,860 cache PUTs, 26.3 M same-region GETs ($10.54) |
| cache storage (7 days per cycle) | - | 1,091 GB-months ($25) |
| p5.48xlarge on-demand slot time + cache requests and storage | $39.8k | $10.1k |
| a3-highgpu-8g on-demand | $63.9k | $16.1k |

The rates and the retention are assumptions, labelled in the document;
publisher egress is 0 for the public open-data buckets
(`--publisher-egress-usd-per-gb` otherwise). The cache does not change
decode or preparation time.

## Limits

- Legacy `gfs`/`gdas`/`hrrr`/`era5` fetches are not shared (table above).
  GFS archive full-file objects are area-independent bytes, but their
  receipt binds the area; sharing them needs a fetch-side change this
  unit does not make.
- A `file://` cache is shared only by hosts that see the same filesystem.
- `forcing` is bound at plan time; see "Planning" for adding a cache to
  an existing directory.
