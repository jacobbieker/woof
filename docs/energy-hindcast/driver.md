# Hindcast segment driver

`woof energy hindcast` turns a set of wind-farm sites and a date range into
a chain of forecast segments, one job per cluster per segment. Running every
job and keeping each one's post-spin-up window gives a continuous
reanalysis at 100-300 m. The code is in `woof/energy/hindcast.py`.

A year can't be one run. No boundary source forecasts a year ahead, and a
run restarted from its own checkpoint can't switch to a new forcing cycle
(only the HRRR route chains restarts across cycles). So each segment is a
fresh cold start from one source cycle, and its first hours are thrown away
as spin-up.

## Segments

| Flag | Default | Meaning |
|---|---|---|
| `--start` | required | first segment's day (`YYYY-MM-DD`, or `YYYY-MM-DDTHH` on `--cycle-hour`) |
| `--end` | required | exclusive bound on segment run starts |
| `--segment-hours` | 30 | run length of every segment |
| `--spinup-hours` | 6 | leading hours discarded |
| `--cycle-hour` | 0 | UTC hour each run starts from |

Segment `k` runs from `start + k * keep` for `segment-hours`, where
`keep = segment-hours - spinup-hours` (24 h by default). Its kept window
is `[run_start + spinup, run_end)`, which is half-open, so consecutive
segments tile time with no gap and no duplicate frame. With the defaults,
`--start 2025-11-01 --end 2025-11-04` makes three jobs: runs from 00Z on
the 1st, 2nd and 3rd, keeping 06Z to 06Z. To keep whole UTC days (00Z to
24Z), pass `--cycle-hour 18` and start the day before. A 2024 hindcast is
366 segments and a 2025 one is 365.

All times are naive UTC. Month ends, 29 February and daylight-saving
changes are just datetime arithmetic. An input with a non-UTC offset
(`+01:00`) is refused rather than converted. A `--start` hour that
disagrees with `--cycle-hour` is refused rather than shifted.

## Clusters

- `--sites SITES.json` plans the whole sites document as one cluster,
  named by `--cluster-id` or by the file's stem.
- `--partition PARTITION.json` reads a `woof-energy.partition.v1`
  document. Each cluster gives its `cluster_id`, `site_ids` and, if it
  has them, `region`, `leaf_dx_m`, `chain` and `topology`. `--clusters
  ID,ID` keeps only some clusters. The sites document is the partition's
  `sites.path`, or `--sites`.

Each cluster's sites are written to `clusters/<id>/sites-<sha12>.json`, named by content, so editing the sites never rewrites a file an older job uses. Every
job of that cluster binds the file by sha256.

## Boundary source per segment

The source is chosen per job by a selector, called as
`selector(region, cycle, hours, p_top_pa=..., priority=..., now=...)`. It
returns an object with `.source` and, if it has them, `.reasons`.

- The default selector is `woof.energy.source_select.select_source`, when
  that module is in the build.
- Without it, a built-in fallback takes the first source in
  `--source-priority` (default `met-office-global,gfs`) whose publication
  window covers the segment. Windows are judged by
  `woof.energy.source_window.window_refusal`. The Met Office archive is a
  rolling two years, so older segments fall back to GFS.
- `--source` pins one source for every job.

Each job records the selector, its reasons, the previous segment's source
and whether the source changed. A source change is a discontinuity in the
reanalysis, so the manifest notes it and `status` counts it. `--p-top-pa`
(default 5000 Pa, about 20 km) is passed to the selector. At materialize it
is checked against the `p_top` in the emitted configuration.

## History

| Topology | What the driver sets |
|---|---|
| `wrf-nests` | `history_begin_s = spinup * 3600` on every `[[domain]]`; leaves every `--leaf-history-interval-s` (default 900); parents every `--parent-history-interval-s` (default 3600) |
| `wrf-tiles` | the parent keeps every frame from t = 0 at the planner's cadence, because those frames force the tiles; leaf tiles get `output_interval_s = --leaf-history-interval-s` with `--child-config-sha256` recomputed; the kept window is applied at extraction |

`wrf-tiles` children read a legacy `[run]` table that has no history
window, and they start where the parent's history starts. That's why the
tile spin-up frames are written, then cut at extraction and handed to the
writer as `time_window`. `--parent-history-interval-s` is refused with
`wrf-tiles`.

Every interval must divide the kept window. The leaf interval must also
divide the spin-up, so that the first kept frame falls exactly on the
window's start. Finer cadences such as 300 s are allowed. Patched files are
read back through the door's own loader (`woof.experiment.load_experiment`,
`woof.offline_child.resolve_child_run_config`) before they replace the
planner's files.

`hex-swath` has no history window and is an experimental lane, so it is
refused.

## Jobs and the manifest

```text
hindcast-manifest.json           woof-energy.hindcast-manifest.v1
clusters/<cluster>/sites-<sha12>.json  sites.v1 subset, named by content
jobs/<job_id>/job.json           woof-energy.hindcast-job.v1
jobs/<job_id>/plan/plan.json     woof-energy.plan.v1 (after materialize)
jobs/<job_id>/materialized.json  woof-energy.hindcast-materialized.v1
jobs/<job_id>/sites-forecast.nc  forecast.v1, kept window (after run-local)
```

A job id looks like `<cluster>-<YYYYMMDDTHH>Z-<sha12>`. The hash covers
everything that changes what the job computes: the cluster's site content,
leaf spacing and chain, the segment window, the source, the planner
topology and arguments, the history settings and the output targets. The
same request writes byte-identical `job.json` files, because nothing in
them depends on when the plan was made.

A job document holds:

- `cluster`: its id and site file, plus the partition, region, leaf
  spacing and chain where a partition gave them
- `segment`: the run window, kept window and `history_begin_s`
- `source`: the source with the selector's reasons, and the previous
  segment's source
- `planner`: the topology, its `build_plan` arguments (leaf `dx_m`,
  `parent_dx_m`, `nz`, `card`/`vram_gib`, `start`, `hours`) and
  `p_top_pa`
- `history`
- `outputs`: the gridded store template with `{leaf}` still to fill, the
  site store URL and the `time_window`
- `resources`, `cost`, and `paths` relative to the job directory

`cost` is a hook. Before materialize it holds the run and kept hours.
After materialize it adds the plan's domain count and horizontal cells, and
the largest VRAM envelope the planner priced. `gpu_hours` stays empty
unless the caller injects a `cost_estimator`.

Output URLs are filled from `--output-root` (default `icechunk`, relative
to the hindcast directory; give `s3://bucket/prefix` for S3),
`--gridded-url` (default `{root}/gridded/{cluster_id}/{leaf}.icechunk`)
and `--sites-url` (default `{root}/sites/{cluster_id}.icechunk`). There is
one store per cluster leaf for the whole hindcast, and each job writes its
kept window into it.

Job statuses are `pending`, `materialized`, `running`, `complete` and
`failed` (with a reason). The manifest is written to `<name>.partial` and
then renamed, under an advisory `flock`, so several workers sharing one
filesystem don't lose each other's updates.

Running the same `plan` again changes nothing and keeps every status. A
different request into the same directory is refused unless you pass
`--resume`. With `--resume`, new jobs are added and jobs with the same id
keep their records. A job planned differently for the same cluster and
cycle replaces the old one, unless the old one is `complete` or
`running`, in which case the old one is kept and a note is added. An
existing `job.json` is never rewritten, and job directories are never
deleted.

## Commands

```bash
woof energy hindcast plan --sites tests/fixtures/energy/sites_wales.json --start 2025-11-01 --end 2025-11-04 --vram-gib 80 --dry-run -o hindcast
woof energy hindcast plan --sites tests/fixtures/energy/sites_wales.json --start 2025-11-01 --end 2025-11-04 --vram-gib 80 -o hindcast
woof energy hindcast status hindcast
woof energy hindcast materialize JOB_ID --dir hindcast
woof energy run hindcast/jobs/JOB_ID/plan/plan.json --dry-run
woof energy hindcast run-local JOB_ID --dir hindcast --dry-run
woof energy hindcast run-local JOB_ID --dir hindcast --writer mypackage.store:write_segment
```

- `plan --dry-run` prints the manifest it would write and one full job,
  and writes nothing. Everything up to `run-local` without `--dry-run`
  runs on the CPU.
- `materialize` calls the planner for the job's segment (`--start` = the
  segment's run start, `--hours` = its length). A partition cluster goes
  through `woof.energy.partition.plan_cluster(partition, cluster_id,
  **kwargs)` when that module is in the build; arguments its signature
  doesn't take are not passed. Otherwise the topology's `build_plan` plans
  the cluster's sites file (refused if its sha256 changed). Materialize then applies the history
  settings above and writes `materialized.json`. A plan already
  materialized for this job, whose `plan.json` still matches the recorded
  sha256, is reused. `--force`, or anything else left in `plan/`, moves
  the old directory to `plan.previous-<UTC>`; nothing is deleted.
- `run-local` materializes if needed, runs every domain with `woof energy
  run`'s `run_plan` (`--resume` is passed through), then runs the post
  step. The post step samples the sites, keeps the window and writes
  `sites-forecast.nc`. If `--writer MODULE:CALLABLE` is given, the writer
  (the Icechunk writer, for example) is then called as
  `writer(job=, job_dir=, plan_path=, forecast=, targets=)`.
  `--no-extract` skips the sampling. A job the manifest calls complete is
  skipped unless `--force`. The job is claimed as `running` under the
  manifest lock, so a second worker is refused. A run that died leaves
  `running` behind; `--force` takes the job over. `materialize` also
  refuses to re-plan a `running` or `complete` job without `--force`.
- `status` reports counts by status, and for each cluster: kept-window
  coverage with any gaps or overlaps, the sources used, and how many times
  the source changed. `--status failed` lists the failed jobs.

The Python entry points take the same injection hooks: `source_selector`,
`cost_estimator`, `cluster_planner`, `runner`, `post_step` and `writer`.
This lets a work queue or a test drive the driver without the sibling
modules.
