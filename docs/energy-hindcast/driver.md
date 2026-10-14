# Hindcast segment driver

`woof energy hindcast` turns a set of wind-farm sites and a date range into
a chain of forecast segments, one job per cluster per segment. Running every
job and keeping each one's post-spin-up window gives a continuous
reanalysis at 100-300 m. The code is in `woof/energy/hindcast.py`.

A year can't be one run. No boundary source forecasts a year ahead, and a
run restarted from its own checkpoint can't switch to a new forcing cycle
(only the HRRR route chains restarts across cycles). So each segment is a
fresh cold start from one source cycle. By default each global cycle
(00/06/12/18 UTC) gets one 9 h forecast at 15-minute output. Its first 3 h
are discarded as spin-up and hours 3 to 9 are kept, so the kept windows
`[init + 3 h, init + 9 h)` stitch into one continuous analysis (see
"Spin-up" below). By default the leaves are 300 m and every domain carries
the 49-level wind-energy column. These are the cost-efficient end of the
requirement; [cost.md](cost.md#cheapest-configuration) prices them against
the alternatives.

## Segments

| Flag | Default | Meaning |
|---|---|---|
| `--start` | required | first segment's day (`YYYY-MM-DD`, or `YYYY-MM-DDTHH` on `--cycle-hour`) |
| `--end` | required | exclusive bound on segment run starts |
| `--segment-hours` | 9 | run length of every segment |
| `--spinup-hours` | 3 | leading hours discarded; omitted with another `--segment-hours`, it is refused unless the kept window is still a whole number of 6-hourly cycles |
| `--cycle-hour` | 0 | UTC hour each run starts from |

Segment `k` runs from `start + k * keep` for `segment-hours`, where
`keep = segment-hours - spinup-hours` (6 h by default). Its kept window
is `[run_start + spinup, run_end)`, which is half-open, so consecutive
segments tile time with no gap and no duplicate frame: the 09:00 frame
comes from the 06Z init (3 h in), never the 00Z forecast's last step. With
the defaults, `--start 2025-11-01 --end 2025-11-04` makes twelve jobs, one
per 00/06/12/18Z init, kept from 03Z on the first day to 03Z after the
last, and a year is 1,460 segments (1,464 in a leap year). The stored
period is therefore `[start + 3 h, end + 3 h)`: to cover 00-03Z of the
first day, start the request a day earlier (or one cycle earlier with
`--cycle-hour 18`). Consecutive requests stitch with no gap.
`--segment-hours 6` on its own is refused: with the 3 h default it would
keep 3 h windows from runs at 03/09/15/21Z, so give `--spinup-hours`. The 3 h spin-up
costs 1.5x the GPU time of keeping each 6 h forecast whole
(`--segment-hours 6 --spinup-hours 0`, kept `[init, init + 6 h)`); 1 h and
2 h are `--segment-hours 7 --spinup-hours 1` and `8 / 2`. The daily layout
of earlier versions is `--segment-hours 30 --spinup-hours 6` (365 segments
a year, kept 06Z to 06Z).

### Spin-up

A cold start from a 9–13 km global analysis needs time before its fields
are in balance with the 100–300 m grid:

| What adjusts | Typical time |
|---|---|
| acoustic and gravity-wave noise from interpolating the analysis onto the fine terrain | 30–60 min |
| resolved boundary-layer turbulence on the LES leaves (from smooth initial fields) | 30–60 min by day, 1–2 h in stable nights |
| cloud and rain water (little or none in the global analyses; icing depends on it) | 2–6 h |
| mesoscale circulations the analysis does not resolve (sea breezes, valley flows, low-level jet timing) | 6–12 h |

Inflow turbulence also needs *fetch*: a 250 m child's inflow spin-up was
measured to cover most of its own domain
([GRAYZONE-NEST.md](../public/GRAYZONE-NEST.md)), and no amount of time
spin-up removes that until the inflow generator lands. The model is
numerically stable from the first step; spin-up is about realism, not
blow-ups (those are a time-step and terrain question). The 3 h default
covers the noise and most of the boundary-layer adjustment; it does not
cover cloud water or mesoscale circulations, which would need 6-12 h at 2x
to 3x the cost. Every stitch is a switch to a new init, so expect a step
change at the join; `--spinup-hours` shrinks it, and ramp detection should
treat join times as suspect.

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
(default 5500 Pa, 19.97 km in the standard atmosphere) is passed to the
selector and, as `column_request`, to the planner, so every domain carries the
wind-energy column of [the vertical page](vertical.md): explicit eta levels with
a 12 m lowest layer, at least 8 layers below 250 m, and a top refused above
20 km. At materialize it is checked against the `p_top` in the emitted
configuration.

## Grid defaults

| Flag | Default | Meaning |
|---|---|---|
| `--dx-m` | the cluster's `leaf_dx_m`, else 300 | leaf spacing, 100 to 300 m; a value that disagrees with the partition's is refused |
| `--nz` | 49 | levels of the wind-energy column on every domain: the smallest with at least 8 layers below 250 m, a 12 m lowest layer, a top at most 20 km and no layer deeper than 1000 m (`woof.energy.vertical.HINDCAST_NZ`) |

Both are the cheap end of the requirement. 300 m costs about a third of
200 m, and 49 levels cost about 0.76x of the 70-level preset with the same
10 rotor layers. Pass `--dx-m 200 --nz 70` (and a partition cut at 200 m)
for the finer grid.

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

**What a leaf writes.** `--leaf-output-preset` (default `windfarm`) sets
every leaf's `[output]`: a `wrf-nests` leaf `[[domain]]` gets
`output = { ... }` and a `wrf-tiles` leaf tile's child config gets its
`[output]` table replaced (and its `--child-config-sha256` rebound).

| `--leaf-output-preset` | leaf `[output]` |
|---|---|
| `windfarm` (default) | `preset = "windfarm"`, `hub_heights_m` = the cluster sites document's `heights_m`: only 2-D planes, the hub-height wind and scalars at exactly the heights the site store holds ([the windfarm preset](../energy-forecasts.md#wind-farm-output-without-3-d-volumes-the-windfarm-preset)) |
| `energy` | `preset = "energy"`, `hub_wind = true`: the 3-D site-forecast volumes plus hub wind at the default heights |
| `full` | `preset = "full"`, `hub_wind = true`: every variable |

`windfarm` refuses, at plan time, a cluster whose sites carry more than
32 heights or a height outside (0, 2000] m: rebuild the sites with a
coarser `--hub-height-step-m`, or use `energy`. The job records
`history.leaf_output_preset` and `history.leaf_hub_heights_m`, and both
are part of the job identity. A job document written before the option
existed has neither and keeps the `full` leaves it was planned with.

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
