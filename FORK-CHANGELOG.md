# Fork changelog

What this fork (README.md, "This fork") changes on top of the upstream
release it carries. It ships as `metal-woof`, versioned by its
`metal-vX.Y.Z.N` tag; CHANGELOG.md is upstream's record of `recast-woof`
releases and is kept as upstream wrote it.

## Unreleased

Merged upstream WOOF 1.1.5 (with 1.1.4). Where upstream built a feature the
fork already had, upstream's keys and flags are now the only ones:

- Wind farms: `wind_farm_scheme`, `wind_farms` (and `--wind-farms`),
  `windfarm_tke_factor`. The fork's WRF v4.7.1 Fitch scheme is
  `wind_farm_scheme = "fitch_wrf"`, reading `wind_farms` as a WRF
  `windturbines.txt`, a `windturbines-ij.txt` (WRF's `windfarm_ij = 1`) or a
  woof turbine table with turbine-library curves; it writes WRF's `POWER`,
  runs on hex's unequal-area cells (`--physics wind_farm_scheme=fitch_wrf
  --physics wind_farms=FILE`), and `woof import-namelist` selects it for
  `windfarm_opt = 1`. The `windfarm_opt`, `windfarm_ij` and
  `windfarm_turbine_file` config keys are gone.
- Icechunk: `woof go` exports through `--output-format netcdf,icechunk` /
  `--icechunk-store` (`[output] formats` / `icechunk_store`). `--icechunk URL`
  and `[output] icechunk = {...}` are gone; `[output]
  icechunk_delete_after_commit` (or `--icechunk-delete-after-commit`) and
  `icechunk_domains` carry the fork's delete-after-commit backpressure and
  domain subset, and `{name}` in the store is the `[experiment] name`.
- LES nests: upstream's `--les-below-m`, `--les-closure` and
  `--les-inflow-seeding`; a tree suite whose nests run PBL off is still
  counted as LES in the gray-zone advisory.

New:

- Noah-MP is recorded as following WRF 4.7.1's Noah-MP submodule (e5c0859),
  at the user's request. No forecast answer changes, because the default
  already computes what e5c0859 computes. Of e5c0859's edits to the code
  WOOF transcribes:
  - SNOW_AGE moving out of the sunlit branch (ALBEDO and the glacier albedo)
    is already the default, `noahmp_snow_age = "wrf_470"`;
  - NOAHMP_GLACIER returning HCPCT for the driver's SOILENERGY/SNOWENERGY
    integrals is what the glacier port already did; it was documented as a
    departure from 4.6.1's stale local and now matches e5c0859;
  - SFCDIF2's new PSLMS/PSLHS are dead, because `ILECH = 0`;
  - `calculate_soil` and the irrigation table reader change no arithmetic.
  `noahmp_snow_age = "wrf_461"` stays available for WRF 4.6.1's behaviour.
  The WRF-Hydro, Noah `-DWRF_HYDRO`, Noah-MP `WRF_HYDRO` and ocean-coupler
  lanes are pinned to WRF 4.7.1.

- A live nest can carry its own eta ladder (WRF `vert_refine_method = 2`): a
  `[[domain]]` names `eta_levels` under the tree's shared `p_top`,
  `hybrid_opt` and `etac`. The parent forces it every parent step through
  WRF's vertical interpolation and its rebalance of `t`, QV and `ph`, ported
  as one GPU unit and a binary32 CPU twin that both match WRF v4.7.1's Fortran
  word for word (`tools/vert_nest_wrf471_oracle`). A nest born from its parent
  is interpolated the same way. A nest prepared from the analysis is built on
  its own ladder directly. WRF namelists with `vert_refine_method = 2` and
  per-domain `e_vert`/`eta_levels` import. Feedback, moving, spawned, split
  and streamed edges, and a nest with a different microphysics scheme, are
  refused by name. A tree whose nests name no ladder of their own is
  unchanged: its RunConfigs, fingerprints, checkpoints and caches are the
  same.

- `woof downscale --child-levels` takes two more spellings: `eta:LIST|FILE`, an
  explicit child eta ladder (a comma list or a file such as one
  `python -m woof.vertical_bands --write` makes), and
  `bands:NZ,LO:HI:N[:agl],...`, a band-biased ladder laid out under the
  archived parent's model top and over its median ground. The child still
  shares the parent's `p_top`, `hybrid_opt` and `etac`, so the conservative
  remap is unchanged. `N,STRETCH` and runs that name no ladder are unchanged.

- `woof track splice` joins a stack of grids. The coarsest covers the whole box at
  every height; finer ones cover smaller footprints below cutoff heights. Each
  point is answered by the finest grid that covers it, laterally beyond its
  edge zone and below its cutoff, falling back to the next coarser grid, with
  optional linear blends across `--lateral-blend-cells` and `--blend-m`. Grids
  are given as `--parent`/`--child`/`--splice-height-m`, or as any number of
  `--tier PATH[,CUTOFF_M]`, coarsest first. `--path` samples every fix of a
  time/lat/lon/height path (CSV, JSON or an Icechunk group) and states each
  join's continuity, vertical and lateral. `--fields-out` writes each finer
  grid's frames merged with the coarser ones, plus `SPLICE_WEIGHT`, in wrfout
  layout for the Icechunk group writer. It reads finished history only; no run
  changes.

- `windfarm_opt = 1` runs WRF v4.7.1's Fitch wind-farm scheme in the
  forecast, so a farm's wakes reach the cells downstream. Each turbine of
  `windfarm_turbine_file` (a woof turbine table, or WRF's own
  `windturbines.txt`/`windturbines-ij.txt` and `wind-turbine-N.tbl`) is an
  elevated momentum sink and a TKE source over the levels its rotor
  sweeps, added after MYNN in the PBL slot as WRF's PBL driver adds it.
  The driver's PBL step matches the pure kernel word for word, and the
  kernel matches WRF's Fortran. History gains WRF's `POWER` (W per cell),
  also in the `windfarm` preset. `windfarm_ij` and `windfarm_tke_factor`
  import from WRF namelists. MYNN is required; moving, spawned, streamed
  and multi-device domains are refused by name. Checkpoints bind the
  turbine file's bytes. Off by default: nothing is built or launched, and
  checkpoints, fingerprints and prepared bundles are unchanged. The
  layout's duplicate GPU `dragcof` kernel is retired; the CPU table writer
  stays.

- `--source ukv`: the Met Office UKV (UK deterministic 2 km), 33 pressure
  levels to 10 hPa, from two anonymous stores. A window inside the Source
  Cooperative Icechunk archive (2024-01-01 00Z to 2025-08-13 22Z, each
  hour's own analysis) is read from it; any other window is read from ONE
  run of the Met Office's open-data bucket (a rolling two years), the
  newest run whose leads hold every valid time and that has posted them
  all. The 03Z and 15Z runs reach T+120, three-hourly past T+54, so a
  window longer than 54 h takes `--cadence 3`. `woof fetch --source ukv
  --area ...` regrids the window from the UK Lambert azimuthal equal-area
  grid to a regular 0.02-degree grid over the area plus 0.25 degrees,
  writes earth-relative wind components (the directions were measured to
  be true-north) and one CF NetCDF per hour; the packaged profile
  `ukv-sc-netcdf-v1` decodes them. Neither store has soil or sea ice: each
  file carries the same hour's ERA5 land-sea mask, sea ice and soil
  column (Google's ARCO copy), or GFS 0.25-degree's past ERA5's last
  published hour, on Noah's four layers by nearest index, and one terrain
  (a live run's orography; in the archive, which publishes none, the
  first hour's surface pressure on its own levels). Domains off the UKV
  grid, windows no store serves and runs still posting are refused by
  name, and an interrupted fetch resumes hour by hour. Archive values are
  float16, so temperature steps 0.125-0.25 K and height 16 m above about
  100 hPa; the live objects are float32.

- `woof track plan` writes a nested, track-following `woof go` run for
  any 3-D path (time, lat, lon, height). It reads the path with the
  `woof track inspect` column options. It builds a 1.5 km / 300 m / 100 m
  tree with the `woof domain` machinery: d01 covers the path's envelope
  plus a margin, and d03 is born from d02 an hour before launch and writes
  history only in flight. The column has 256 band-biased levels, and the
  model top sits 7 km above the path's peak. Only d03 is exported to
  Icechunk. Each run also gets a `plan.json` with cells, GPU memory,
  cell-steps, a wall-time range and output volume, and, with
  `--job-json`, a `woof.go-job.v1` document, inline or by URL under
  `--inputs-prefix`. Several paths, or a folder, plan in one call with
  `--names-from stem`. Where this woof has no track itinerary, LES-nest
  tree suite or lean track history preset, the plan uses static nests,
  explicit nest physics keys or the `minimal` preset, and its notes say
  so. When the source was picked automatically and its column stops below
  the model top, the plan falls back to GFS. A window the source serves
  only three-hourly (UKV past T+54) is planned and fetched on 3-hour
  boundaries. A path too fast for the relocation cadence is refused with
  a cadence that works.

- A delayed nest can now be born from its live parent. Set
  `start_from = "parent"` on a `[[domain]]` whose `start_time` is later
  than its parent's, and `woof go` interpolates the parent's state onto
  the nest's own grid and terrain at that instant (WRF's nest
  initialization from the parent, with the terrain blend, `adjust_tempqv`
  and `press_adj`), with the parent's land state and accumulated rain. A
  long coarse spin-up can then feed a short fine one, without fetching an
  analysis at the nest's start or landing that start on a forcing seam.
  Each birth is recorded in `nest_birth_receipts.jsonl`. A resume after
  the birth births the nest again before restoring the checkpoint.
  Measured on an RTX A6000 with a 1.5 km / 300 m / 100 m HRRR tree whose
  100 m nest started 10 minutes in: the nest wrote nothing before its
  start, its first frame matched its parent's window (mean T, MU, PSFC
  and T2 within 0.002 K, 0.002 Pa, 0.09 Pa and 0.002 K), d01 and d02 were
  bit-identical to the same run without the nest (one-way), and a resume
  from a checkpoint after the birth was bit-identical to the unbroken run.
  `woof run`, `woof stream`, the wrfinput/met_em doors, the stock-WRF
  export, streamed parents or nests, and moving nests refuse it by name.

- A nested tree split across cards with `[devices]` is no longer drained
  to the host at every parent step. The nest coupler now reads only its
  windows from the slabs: the nest's boundary frame and the parent's
  donors for forcing, and the nest's footprint for feedback. Two-way
  feedback writes the parent's restricted rectangle back to the slabs the
  same way. The next sweep no longer copies the whole store back to every
  slab. With `feedback = 0` the nest is never drained for coupling. This
  covers a resident parent over a split nest and a split parent over a
  split nest. On a stand-in for the one-card proof tree (288 x 288 parent,
  216 x 216 nest, 49 levels, two slabs on one RTX A6000), both roads gave
  bit-identical carriers and digests in all four shapes. Drains and copies
  back per parent step fell from one per split grid to zero. The
  split-parent, split-nest, `feedback = 1` run was about 1.4x faster
  (21.5 and 19.0 s down to 14.4 and 13.3 s). `WOOF_SPLIT_NEST_DRAIN=1`
  brings back the drain road for comparison. See
  [One forecast across cards](docs/public/MULTIGPU.md#nested-trees).

- The per-step health check no longer makes the CPU wait for each step
  to finish. Before, every step ended with a blocking read of its health
  result (NaN, a collapsed model layer, and each physics scheme's status
  word), so the next step could not be queued until the GPU had drained.
  Now the result is copied to page-locked memory asynchronously and
  checked once the next step is queued. Every pending result is still
  checked synchronously before a history frame, a checkpoint, a stop,
  history or restart boundary, a relocation, a delayed nest start, an
  adaptive step choice, and the end of the run, so a bad step never
  reaches a wrfout or a checkpoint. A failure names the step that went
  bad and says it was detected one step late. In strict mode the run
  stops before the failed step's clock advances; in lagged mode it stops
  at the end of that step. Results are bit-identical.
  Measured on an RTX A6000 that two other forecasts kept at 100%
  utilisation, with 5 interleaved runs per mode. Final state, every
  history instant and every checkpoint instant were bit-identical
  between jacob/main, strict and lagged. On a 100 x 100 x 50 WSM6
  domain the median step fell from 36.4 ms to 34.2 ms (6%). On
  40 x 40 x 30 (dry and WSM6) and 100 x 100 x 50 (dry) the differences
  were within run-to-run noise (+/-20%). There, a step takes about
  1.3 ms to queue but 4-13 ms to run on the shared card, so the CPU was
  not what limited them.
  `--health-check strict` on either prepared runner, or
  `WOOF_HEALTH_CHECK=strict`, checks every step synchronously as before.
  `--health-debug` and `[tiles]` streamed domains always check strictly.
  The receipt's `health_check` block records the mode and the counts.
  Also:
  - The every-fourth-step full-state check uploads its descriptor
    tables only when they change. It used to make seven blocking
    uploads each time. Away from a mandatory boundary it is lagged too.
    A lagged report has the same phase, field, index and status bits;
    only the first bad value is read a step later, and the report says
    so.
  - The adaptive controller of a nested tree reads every domain's CFL
    with one synchronisation per root step instead of one per domain.
  - While lagged, the scheme status words go through the existing
    health ledger, as under `[tiles]` graph capture. The carrier check
    reads the land/ice footprint only when it is about to refuse, so an
    all-water domain is checked exactly as in strict mode.
  - The single-domain `integrate_prepared_case` loop (`woof run` without
    an adaptive clock, and ensemble members on that route) is unchanged.
  See docs/public/PROGRESS.md.
- The model step no longer stops to read back each scheme's health check.
  The microphysics, YSU and Kain-Fritsch status words, validated scheme
  outputs, the surface-radiation carrier finiteness check and MYNN's input
  and output guards now record their verdicts on the GPU. The step reads them
  once, at its end. It raises the same exception type with the same message,
  plus a note that the check failed earlier in the step. An all-water domain
  is checked as before.
  - An ordinary step of the wind-farm suite (Thompson, MYNN, Noah, RRTMGP)
    now has 1 blocking read instead of 7, plus the run loop's stability
    check.
  - Measured on a shared RTX A6000, the step is 1.34x faster at 48,000
    cells, 1.19x at 0.49 M and 1.06x at 1.96 M cells.
  - Results are bit-identical.
  - RRTMGP's input guard, Grell-Freitas, WDM6 and the urban models still
    read straight away.
  - `WOOF_HEALTH_READBACK=immediate` restores the per-check reads.
  - Run receipts record the mode and counts under `health_readback`.
  - Inside the lagged health check above, the step uses that check's
    ledger and does not read at its own end. MYNN's guards now go through
    either ledger.

  The study behind this change is in
  docs/dev/perf-launch-overhead-20261015.md. It reports per-step launches,
  host syncs, GPU-busy time and the CUDA-graph headroom. It also proposes a
  new per-step cost for the hindcast cost model.
  `tools/profile_launch_overhead.py` and `tools/launch_profile/` are the
  CUPTI-based profiler used for it.
- History frames leave the model loop sooner. Every file stays
  byte-identical, and the output identity (SHA-256) is the same value.
  - Resident domains: the frame's fields are copied device-to-device into
    buffers the writer keeps, on the model's own stream. A side stream copies
    them to the host. The model no longer waits for the whole host copy of
    each frame. The buffers are used only inside the room the run's priced
    memory plan leaves on the card (at most 4 GiB per writer). Otherwise the
    old fenced copy is used, and the receipt names the reason (`unpriced`,
    `outside-priced-plan`, `over-cap`, `device-allocation-failed`,
    `disabled`).
  - `[tiles]` host-store domains: the frame is snapshotted with one host
    memcpy, inside the asynchronous checkpoint's host-RAM budget. The model
    sweeps on while the frame is written, instead of draining the netCDF
    write between two sweeps. Otherwise: views and a drain, with the reason
    `insufficient-host-ram`.
  - The Rust writer hashes each tape while writing it, so the output
    identity no longer reads every finished file back. This needs the
    rebuilt `netcdf-writer`; an older library falls back to the re-read, by
    name. `WOOF_OUTPUT_IDENTITY_VERIFY=1` re-reads anyway and refuses a
    disagreement.
  - The process-wide netCDF lock now covers only the netCDF4 validation
    reopen of a Rust-engine tape, so nested domains' tapes are written at
    the same time. The netCDF4 engine keeps its whole-session lock.
  - `[output] history_queue_depth` (or `WOOF_HISTORY_QUEUE_DEPTH`) lets a
    domain queue more frames, capped so their page-locked copies fit host
    RAM. The page-locked pool is released at the writers' drain points.
  - `woof hex forecast` writes its history files (netCDF, zlib, stamps,
    hash) on a background thread. The capture and its physical gate stay
    on the model thread, so refusals land on the same step.
    `woof.hex.output.AsyncHistoryWriter` is the reusable writer.
  - The run receipt's `history_writes` block records each domain's copy
    mode and reason, the queue depth, the identity mode, the lock scope and
    the per-frame model-loop stall. `WOOF_ASYNC_HISTORY=off` restores every
    old path. See OUTPUT-VARIABLES.md, "How a history frame leaves the
    model".
  - Measured, with other forecasts sharing the card, so read these as
    rough. The case is a nested 1 km + 333 m Greenland run (180x144x37 and
    90x72x37), 300 s, history every 30 s (22 files). Old and new runs were
    interleaved, three of each, and these are medians.
    - `[tiles]` host store: the model-loop time per 1 km frame fell from
      243 ms to 20 ms. History submits per run took 4.73 s, now 0.59 s.
    - Resident: frames are small here, so submits per run went only from
      0.35 s to 0.25 s. The 1 km producer-stream stall per frame fell from
      11 ms to 2.8 ms.
    - The loop wall time moved by less than the run-to-run noise.
    - Every wrfout was byte-identical between old and new. The streamed
      identities equal the SHA-256 of the files.
    - Hex: a harness on real 24k-cell x 80-level frames, not a full
      forecast (that needs more GPU memory than the test cap). Model-thread
      time per frame fell from 1.7-2.0 s to under 1 ms, with byte-identical
      files.

- Checkpoints no longer stop the forecast while they are written. The
  device state is copied into page-locked host memory on a side stream.
  The model goes on stepping, and a background thread writes, `fsync`s and
  renames the set. On a 294 x 506 x 256 domain each hourly checkpoint used
  to stall the loop for a median 386 s with the GPU idle (35% of the run).
  A checkpoint becomes the resume point (`last_checkpoint`, the step log,
  retirement of older sets) only once it is durable. At most one is in
  flight; if the next comes due first, the model waits and says so.
  Every exit, including a failure or a signal, finishes or removes the
  checkpoint in flight, and a failed write publishes nothing. History
  frames are no longer drained before each checkpoint; the root member
  is published only after the frames at or before its instant are
  durable. The archives are the synchronous archives (same members,
  order and header, apart from `created`), and results are bit-identical.
  `[output] async_checkpoints = false` or `WOOF_ASYNC_CHECKPOINTS=0` turns
  it off. A checkpoint is written synchronously, with the reason printed
  and recorded, for a `[tiles]` streamed domain, the sealed forcing
  extension, a state on several GPUs, too little host RAM, or no
  page-locked memory. The receipt's `checkpoint_writes` block records the
  mode, any fallback reasons, and each checkpoint's model-loop stall. See
  PREPARED-RESTART.md.
- The synchronous checkpoint writer copies one member at a time through
  one reusable page-locked buffer instead of copying the whole state to
  the host first. The files are byte-identical. On a 1.5 GiB test state,
  peak host RSS during the write fell from 1552 MiB to 78 MiB above the
  process baseline, and the write took 6 s instead of 12-17 s. The setup
  fingerprint and forcing-prefix rows in each header are cached per
  boundary inventory instead of rehashing every boundary table at every
  checkpoint, and the digests hash arrays in place instead of through a
  `tobytes()` copy.
- `woof go` can export history frames to Icechunk while the forecast runs.
  Set `[output] icechunk = { url = "s3://..." }` in the config (also
  `gs://`, `file://` or a directory), or pass `--icechunk URL`. Before,
  the export started only after the forecast had finished: for the 48 h
  UK run (289 frames of about 3 GB each) it took another 14 hours. A
  separate process, at `nice` 10, now polls the run's `ready/` markers.
  It writes each frame into group `dNN` of the repository (one group per
  domain, one commit per frame), compressed with Blosc zstd+bitshuffle.
  The model never waits on the exporter.
  `go` waits for the export to finish before it reports the run done, and
  prints progress and lag while it waits. If the export is incomplete
  (the exporter could not start, crashed, or a frame was refused), the
  forecast still completes. `go` then names what is missing, prints the
  command that finishes the export, and exits 73.
  `woof icechunk-export PATH` finishes such an export, or exports run
  folders after the fact; it exits 0, 1 (store failure) or 2 (refused).
  - Each group's time axis grows from its first frame.
  - A field first written after t=0 (`REFL_10CM`) is created on the frame
    that first carries it, and is NaN before.
  - A resumed run's folder continues the same groups. Where a valid time
    is in both runs, the newer file wins (`frame_published_unix_ms`), so
    there are no duplicates. Re-running an export commits nothing.
  - `s3://` and `gs://` URLs are committed directly, with conflict
    retries. A local repository can still be synced afterwards.
  - Progress, lag and the final receipt are in
    `<run>/icechunk-export/status.json`.
  - Measured on a 40 x 40 x 30 HRRR case (21 frames), alternating runs
    with and without the export: the median model step was 0.0525 s in
    both. The live store equalled an export of the same files made after
    the run.
  - `woof.energy.icechunk_store.write_gridded_segment` gains
    `extend_axis`, `add_variables`, `on_existing="newer"` and
    `write_threads`. With 4 threads, a 540 MB frame took 3-6.5 s instead
    of 11-15 s. Every gridded group now records
    `frame_published_unix_ms`. The writer's netCDF reads take the
    process-wide HDF5 lock.
  - A `[[domain]]` `output = { ... }` table refuses `icechunk` by name:
    one repository holds every domain.
  - See docs/public/PROGRESS.md and docs/energy-hindcast/icechunk.md.

- `woof multi-run` binds a prepared run to its preparation itself. A
  module-form run (`woof.prepared_single_domain_forecast` or
  `woof.prepared_domain_tree_forecast`) needs no hand-copied digests. At plan
  load the declared `--prepared-root` is bound as `woof sim` and
  `woof go --prepared-root` bind it: the sealed proof (or tree receipt and
  config digest), or the head of a chained preparation still being prepared.
  The digests are inserted into the argv, and the single-domain runner's
  absent `--source` comes from the bundle. Digests a plan gives are kept and
  must equal the derived ones; a mismatch refuses the plan, naming both. A
  head pinned on a sealed chained preparation is accepted when it is the head
  its proof records. A mapped single-domain preparation's declared
  experiment config and WPS namelist must be the bytes its
  `execution_inputs` record, or the plan is refused, naming the file that
  still matches. The summary records each
  run's `prepared_binding` (every digest marked `derived` or
  `given, verified`) and the plan's argv as written (`plan_arguments`).
  `woof multi-run --dry-run` prints the bound plan as JSON without touching a
  device or claiming anything. A child that fails now prints the last line of
  its log beside its exit code.
- GPU kernel caches are shared and persistent across `woof multi-run` runs
  and `woof energy worker` slots and jobs. Before, every child started from
  a new, empty CuPy and CUDA cache and recompiled every kernel (typically 1-3
  minutes per run, per job). By default children now use the cache an
  ordinary `woof run` uses, so only the first run on a machine compiles.
  `[cache] kernels = "shared" | "per-run" | "<dir>"` (multi-run plan),
  `woof multi-run --kernel-cache shared|per-run|DIR` and
  `woof energy worker --kernel-cache shared|per-slot|DIR` choose. `per-run`
  and `per-slot` restore the old isolation for cross-device comparisons and
  forensics. Temp directories, outputs and GPU locks stay per child.
  CuPy writes cache entries atomically (temporary file and `os.replace`,
  SHA-1 checked on read), so concurrent writers are safe; the receipts record
  the mode, the paths and this basis.

- `woof multi-run` can now pack several ordinary forecasts onto one GPU. Set
  `[devices] runs_per_gpu = N|"auto"` in the plan or pass
  `--runs-per-gpu N|auto`. `auto` allows at most 8 runs per card. The default
  of 1 is the old behaviour: a second run on a card is refused.
  Each run is priced before anything is claimed, the way `woof check` and its
  own `woof run` supervisor price it. A run starts when three things hold:
  its card has fewer than N of the plan's runs; the live siblings' priced
  peaks plus its own plus a margin fit the card's NVML total; and its priced
  peak plus the margin fit the card's measured free memory, less the growth
  its siblings may still make. The margin is `mem_margin_gib` or
  `--mem-margin-gib`, default 1 GiB.
  Memory used by other processes, such as a forecast already on the card,
  reduces what is admitted. A run that does not fit waits for a sibling to
  finish. A run that does not fit while none of the plan's runs hold its card
  is refused as `not_admitted`, and the other runs continue. A run whose priced
  peak can never fit its card refuses the whole plan before anything is
  claimed, naming the run.
  Each child keeps its own caches and gets its own `WOOF_GPU_LOCK_ROOT`, and
  its supervisor still runs the measured shared-GPU admission itself. The
  multi-run holds each packed card's machine-wide UUID lock, so no other woof
  run starts beside the packed runs. A card whose lock another forecast
  already holds is shared by measurement.
  `--mps` (or `[devices] mps = true`) runs the CUDA MPS control daemon.
  The receipt's `packing` block records card totals, priced peaks and their
  basis, and every admission event with measured used and free memory.
  The MPS daemon code now lives in `woof/gpu_packing.py`, shared with
  `woof energy worker --mps`. See docs/public/HARDWARE.md, "Several runs on
  one card".
- `woof energy worker --slots-per-gpu N|auto` runs several hindcast jobs
  on one GPU. A slot claims only a job whose declared
  `resources.gpu_mem_gib` fits beside the card's live reservations plus a
  margin (`--mem-margin-gib`, default 1 GiB). A job that declares nothing
  costs the whole card unless `--default-job-mem-gib` is given. An idle
  card admits any one job.
  `auto` = min(8, card memory / 3.4 GiB, CPUs per GPU). One worker still
  owns a card, a slot's reservation lives as long as its runner, and each
  runner's `woof run` locks under its own `WOOF_GPU_LOCK_ROOT`.
  The default of 1 is the old behaviour.
  `--mps` starts (or reuses) the CUDA MPS control daemon, so packed
  slots' kernels run concurrently instead of time-slicing. Its pipe and
  log directories go under `--mps-dir`, and it is stopped on exit. The
  worker refuses when `nvidia-cuda-mps-control` is missing.
  `woof energy hindcast plan` records each job's priced GPU memory: the
  wrf-nests planner's priced peak, the wrf-tiles fit ceiling, or `null`
  without a card. `woof energy estimate --slots-per-gpu N|auto [--mps]`
  prices packing with a stated utilisation curve and band. In the
  worldwide example, `auto --mps` on p5 cuts GPU-hours from 395k to 197k
  (106k-697k); this is an estimate, and nothing packed has been timed.
  The deploy renderers pass `slots_per_gpu`/`mps` to all four backends,
  and k8s also gets a commented device-plugin MPS sharing example. The
  container entrypoint reads `WOOF_SLOTS_PER_GPU` and `WOOF_ENABLE_MPS=1`.
  Queue claims gate on dependencies and then on memory; `kind: prefetch`
  jobs declare `gpu_mem_gib = 0` and never reserve or wait for a card.
  `woof energy worker` now also takes the queue as `--queue URL`, the form
  the renderers and the entrypoint already wrote.
  See docs/energy-hindcast/queue-worker.md.
- Hindcast forcing cache: `woof energy hindcast plan --forcing-cache URL`
  (`s3://`, `gs://`, `file://` or a path) fetches each (source, cycle)
  once instead of once per cluster. Every table-route source (Met Office
  global, ECMWF open data, HRRR native, RAP, ICON, ...) gets one prefetch
  job per cycle (`kind: prefetch` in `woof-energy.hindcast-job.v1`), and
  each forecast job of that cycle carries `forcing` and `depends_on`
  (job ids unchanged). `woof energy hindcast prefetch DIR` (or the queue)
  runs the cycle's `woof fetch` once and publishes it to a
  content-addressed cache (`woof-energy.forcing-cache.v1` entries, sha256
  blobs, idempotent, create-only). Before running, `run-local` (and
  `woof energy hindcast mirror JOB`) copies the objects the job's own
  fetch would download into the exact folder `woof go` picks, sha256
  verified, with the receipts `woof fetch` writes
  (`woof.fetch_routes.adopt_verified_objects`), so the job's fetch reuses
  every object and asks no host; anything the cache lacks is fetched by
  the job and listed in `forcing-mirror.json`. Sources whose fetch crops to
  the cluster's area (`gfs`, `gdas`, legacy `hrrr`, `era5`) are marked
  `direct` with the reason. `woof energy queue` claims a job only when its
  queued dependencies are done or dead, and pushes and tries prefetch jobs
  first. `woof energy estimate` adds a `forcing` section: downloads per
  job versus per cycle, measured per object, with GPU-slot time, requests
  and cache storage priced per offer (for the worldwide year about 2,080
  TB and 4,331 slot-hours avoided; every rate an assumption). See
  `docs/energy-hindcast/forcing-cache.md`.
- `[output] preset = "windfarm"` writes only 2-D planes for a wind-farm
  reanalysis: the hub-height wind and its interval MAX/MIN/MEAN, new
  hub-height scalars `T_HUB`, `P_HUB`, `RHO_HUB`, `QV_HUB`, `QC_HUB`,
  `QR_HUB` and `QNC_HUB` (interpolated by the in-model hub diagnostic on
  the GPU at each frame, linear in z like the wind; a CPU NumPy
  reference matches the kernel bit for bit except T/RHO, within 2 ulp),
  the near-surface state, boundary-layer and surface-flux fields,
  precipitation accumulators, the gust family and the georeference. It
  is the one preset without the structural `T`/`PHB` volumes, needs
  `hub_wind`/`hub_heights_m`, and lists a reason for every field
  (`woof.io.history_selection.WINDFARM_FIELD_REASONS`). At six hub
  heights a frame is 99 planes, against about 700 for `energy` and
  2,240 for `full` on the hindcast's 49-level leaf. The hub scalars are produced only
  when a selection names them, so existing `full`/`energy` hub-wind runs
  write what they did before
  (docs/energy-forecasts.md#wind-farm-output-without-3-d-volumes-the-windfarm-preset).
- `woof energy extract` builds `forecast.v1` from the hub planes when the
  history has no `PH`/`PHB`: winds from `WSPD_HUB`/`WDIR_HUB`,
  `air_temperature` and `air_density` from `T_HUB`/`RHO_HUB`, humidity
  and hydrometeors from the hub scalars, at the history's `HUB_HEIGHT`
  levels (a height it does not carry is refused). The 3-D path is
  unchanged and wins when both are present.
- `woof energy hindcast plan --leaf-output-preset windfarm|energy|full`
  (default `windfarm`, with `hub_heights_m` set to the cluster sites'
  heights) sets every leaf's `[output]` on `wrf-nests` and `wrf-tiles`,
  and records it in the job. `woof energy estimate --store-preset
  windfarm [--hub-heights N]` prices the store from the preset's own
  field list (the worldwide example: about 72 TB, against 0.63 PB for
  `energy` and 2.0 PB for `full`). `woof energy hindcast optimize` now
  prices `--store-preset windfarm` by default, the leaves the hindcast
  writes, and its recipe carries the preset to the estimate step. A
  re-plan that would switch the leaf preset under segments already
  written is refused.

- `woof energy plan --topology wrf-tiles|wrf-nests --source auto` picks
  the boundary source for the sites and the `--start` cycle from
  `--source-priority` (default
  `hrrr-native,hrrr,rap,met-office-global,ecmwf-open-data,gfs,era5`):
  the first that covers the sites' box plus the planner's margin, holds
  the cycle (archive start, retention, publication era), reaches
  `--hours` and the 5000 Pa model top. The choice and every passed-over
  candidate's reasons are recorded in `plan.json` and
  `source-choice.json` (`woof-energy.source-choice.v1`).
  `woof.energy.source_select.select_source` is the same choice as a
  function, read from the registry and route table with no network
  (docs/energy-hindcast/source-select.md).
- `woof energy partition` cuts a worldwide `woof-energy.sites.v1` set
  into independent regional clusters (`woof-energy.partition.v1`):
  great-circle single linkage (default 30 km, grid-hashed, about 3 s for
  400 000 sites), bisection of clusters over `--max-extent-km`/
  `--max-sites` or the card, stable hash ids, antimeridian-safe boxes,
  a suggested parent chain, projection, topology and tile estimate per
  cluster, sized for `--card`/`--vram-gib`/`--hardware-json` without a
  local GPU probe. `woof energy plan-cluster` plans one cluster with its
  own parent. The wrf-tiles parent region now spans the sites' shortest
  longitude arc, so a site set across the antimeridian is planned
  rather than refused. See `docs/energy-hindcast/partition.md`.
- `woof energy import` reads four more wind-farm sources: the U.S. Wind
  Turbine Database (`--format uswtdb`, CSV or API JSON), Global Energy
  Monitor's Global Wind Power Tracker (`--format gem`, `.xlsx` or CSV),
  EMODnet offshore wind farms (`--format emodnet`) and OpenStreetMap
  `.osm.pbf` extracts (`--format osm-pbf`, which needs pyosmium and is
  refused without it). These formats keep operating farms only unless
  `--status` asks for more. `woof energy inventory merge` folds several
  assets documents into one wind inventory. It matches turbines by
  distance and plants by distance, name and capacity. A plant whose
  turbines are mapped gives way to those turbines. Each asset records
  its sources and licences in `woof:provenance` and `woof:licenses`. The
  output is still `woof-energy.assets.v1`. `woof energy inventory
  download` fetches USWTDB and EMODnet. See
  `docs/energy-hindcast/inventory.md`.
- `woof energy queue init|push|status|requeue` keeps energy hindcast
  jobs in an object-store work queue (`file://`, `s3://` with
  conditional writes, `gs://` with generation preconditions), and
  `woof energy worker` drains it with one loop per GPU on any number of
  nodes: leases with heartbeat renewal and expiry, bounded retries with
  backoff, a dead-letter state, exactly-once completion markers, runner
  logs uploaded per attempt, and SIGTERM releasing the lease uncharged
  for spot and preemptible machines.  See
  `docs/energy-hindcast/queue-worker.md`.
- `woof energy deploy render --backend aws-batch|gcp-batch|k8s|slurm`
  writes the files that start a fleet of `woof energy worker` queue
  workers: an AWS Batch compute environment, job queue, GPU job
  definition and array job; a GCP Batch job; a Kubernetes Indexed Job
  with workload identity; or a Slurm sbatch array. It renders
  deterministically from a typed TOML/JSON configuration and records
  the result in `deploy.json` (`woof-energy.deploy.v1`). No credentials
  are written, and a secret-looking `env` entry is refused. Missing
  account identifiers become `REPLACE_ME_` placeholders, which
  `submit.sh` refuses to run with and `--strict` refuses to render.
  `deploy/Dockerfile` (CUDA 12, or CUDA 13 with `--build-arg CUDA=13`)
  builds the locked pixi environment, the native bridges and the tables
  into a non-root image with a `woof doctor` healthcheck. Its entrypoint
  copies WPS_GEOG and the high-resolution cache from `WOOF_GEOG_URL` and
  `WOOF_HIGHRES_CACHE_URL` onto local scratch. The guide is
  `docs/energy-hindcast/deploy.md`.
- Energy hindcast jobs can write into one shared Icechunk repository on
  S3, GCS (`s3://`, `gs://`, `file://`, `memory://`) or a shared
  filesystem, many nodes at once. Credentials come from the environment,
  a role or workload identity. `woof energy icechunk init` allocates a
  year-long `(time, site, height)` site store once, and `woof energy
  extract --format icechunk --icechunk-mode region` writes one job's
  times and sites into it. `tools/wrfout_to_netcdf.py --icechunk-group
  --time-axis` writes one segment's wrfout frames for one domain onto a
  shared time axis. Conflicting commits are rebased or re-applied with
  bounded, jittered retries, so no write is lost, and re-runs are
  idempotent. `woof energy icechunk verify` reports the snapshot and how
  much of the axis is written. See `docs/energy-hindcast/icechunk.md`.
- `woof energy rating --products turbine-icing` estimates wind-turbine
  icing at each turbine's hub height. It reports liquid water content, the
  droplet median volume diameter (from Thompson `QNCLOUD`, or a documented
  default droplet number) and meteorological icing. It accretes ice
  (Makkonen, with Finstad collision efficiency) on the ISO 12494 reference
  collector, which gives instrumental icing and the ISO class, and on a
  rotating-blade proxy, which gives rotor icing. It also reports the IEA
  Wind Task 19 ice class and an uncalibrated production-loss estimate.
  `woof.energy.icing_turbine.turbine_icing_dataset` runs the same model on
  wrfout or Icechunk grids (xarray in and out, dask- and CuPy-capable,
  chunked hindcasts continue exactly). `forecast.v1` gains
  `cloud_number_mixing_ratio`, sampled from `QNCLOUD`. See
  `docs/energy-hindcast/icing.md`.
- `woof energy rating --products wind-resource,ramps` adds wind-farm
  products. `wind-resource` gives the IEC 61400-12-1:2017 rotor-equivalent
  wind speed, the power-law shear exponent, veer across the rotor,
  turbulence intensity, density-corrected generic-curve power and capacity
  factor. `ramps` finds ramp events in the power at any time step, from
  15-minute down to 1-minute output, by two definitions: a change over a
  window (15 min, 1 h and 4 h by default) and swinging-door segmentation.
  It writes per-timestep ramp flags and rates and a validated event table
  (`woof.energy.contracts.load_ramp_events`). `--power-curve` chooses
  between new generic IEC class I, class III and offshore curves and the
  existing class II curve. The same calculations are available as NumPy
  and xarray functions in `woof.energy.ramps`. See
  `docs/energy-hindcast/ramps.md`.
- `woof energy power FORECAST.nc --assets ASSETS.geojson -o power.nc`
  estimates wind farm power from turbine-specific curves. Each turbine is
  matched to a curve by make and model (exact, then fuzzy), else by rating
  and rotor diameter, else a generic class curve, and the match is
  recorded (`--matches`). It uses the rotor-equivalent wind where the
  forecast heights span the rotor, the IEC density correction, high-wind
  shutdown with hysteresis (optionally on gusts), Jensen or Bastankhah
  wakes from the turbine layout (a fallback table for plants known only by
  capacity), and availability, icing and electrical losses. Output is
  `woof-energy.power.v1`: power, gross power, wake loss and capacity factor
  per plant, with optional per-turbine series. The turbine library
  (`woof/data/energy/turbine_library.json`, about 200 machines) derives from
  the Open Energy Platform Wind Turbine Library (ODbL-1.0, (c) Reiner
  Lemoine Institut) and is rebuilt by `tools/build_turbine_library.py`. See
  `docs/energy-hindcast/power.md`.
- Hub-height wind inside the model. `[output] hub_wind = true` or
  `hub_heights_m = [...]` writes `WSPD_HUB` and `WDIR_HUB` (wind speed and
  earth-relative direction at each configured height above ground,
  default 40 to 200 m) on a new `hub_height` axis. It also writes
  `WSPD_HUB_MAX`, `WSPD_HUB_MIN` and the time-weighted `WSPD_HUB_MEAN`,
  computed from every model step since the previous history frame, so a
  ramp inside a 15-minute file is recorded. `HUB_WINDOW_S` gives how many
  seconds those statistics cover. `hub_sites` with `hub_sample_interval_s`
  writes a time × site × height sidecar,
  `hubwind_d0N_<time>.nc`, for each history frame. The diagnostic
  changes no model value and is not saved in checkpoints. After a resume
  the statistics start again from the restart time, and `HUB_WINDOW_S`
  shows that. Streamed (`[tiles]`) domains, ensemble capture and nest
  relocation are refused by name. See docs/energy-hindcast/hub-wind.md.
- `woof energy estimate` prices a wind-farm hindcast before any GPU is
  rented. It reads a `wrf-tiles`/`wrf-nests` plan, a partition document, or
  summary statistics (`--clusters N --footprint-km2 A`) and reports
  GPU-hours, instance-hours, wall clock for `--instances` (or the instances
  a `--deadline-days` needs), on-demand and spot dollars per cloud instance,
  and the Icechunk store's size and monthly cost. The output is a
  `woof-energy.cost-estimate.v1` JSON on stdout and a table on stderr.
  Throughput is calibrated on two consumer-card timing receipts and scaled
  by memory bandwidth, so every figure carries a 0.5x-1.5x band. Prices
  come from a dated, editable `woof/data/energy/cloud_prices.json`
  (as of 2025-07-01) and are never fetched live.
  See docs/energy-hindcast/cost.md, including a worked estimate for every
  public wind farm worldwide at 200 m for one year.
- `--card` accepts cloud datacentre parts: `h100-80gb`/`h100-sxm`/
  `h100-pcie`, `h200`, `a100-80gb`, `a100-40gb`, `l40s`, `l4` and `a10g`.
  Each part sizes against the capacity it reports rather than its marketing
  size (an L40S at 45.0 GiB). These rows live in the new `woof.card_catalog`
  with SM count and bandwidth and are marked declared, not measured locally.
- `nwp_diagnostics = 1` now also writes the wind members of WRF's
  `nwp_diagnostics` family, `WSPD10MAX`, `W_UP_MAX` and `W_DN_MAX`, plus
  `U10MAX`/`V10MAX` (the 10 m wind at the instant of the speed maximum)
  and `GUST`, the interval-maximum surface gust from the NCEP UPP
  `CALGUST` method. Each is folded on every model step on the GPU, reset
  after each history frame and carried in checkpoints. The `energy`
  preset selects them. The method and where WOOF differs from WRF are in
  `docs/energy-hindcast/gusts.md`.

- Station verification reads Dynamical.org's ASOS Parquet archive of
  the Iowa Environmental Mesonet's reports: the United States and 14
  other countries, about 4,060 stations in 2026. `woof verify-visuals
  --station-source {auto,dynamical,iem}` or `WOOF_VERIFY_STATION_SOURCE`
  selects the route; `auto`, the default, takes Dynamical.org where its
  frozen station table has stations in the domain and the IEM's
  `rw_asos` elsewhere or when the read fails. Only the needed row groups
  and columns are fetched, by HTTP range requests, and cached under
  `~/.woof/cache/dynamical-asos/` (`WOOF_DYNAMICAL_ASOS_CACHE`). Both
  routes write the same station and surface files, so `rw_verify` scores
  2 m temperature, dewpoint and 10 m wind unchanged; precipitation is not
  read. `verification.json` and each hour's receipt name the source,
  any fallback reason and the data attribution. The archive is labelled
  experimental and states no licence. `pip install 'recast-woof[obs]'`
  adds `pyarrow`, which the Pixi environment carries.
- `woof obs dynamical-asos --bbox W,S,E,N --valid-time ISO --out DIR`
  writes those files for any box and hour, and `woof doctor` reports the
  route in a non-blocking row. The obs battery reads the archive through
  `DynamicalAsosSurfaceSource`.
  Coverage and limits are in `docs/international-obs.md`.
- `woof energy` produces high-resolution forecasts (typically 100 m or
  50 m) along power lines, substations, wind farms and solar farms.
  `woof energy fetch` reads OpenStreetMap power infrastructure through
  the Overpass API (ODbL 1.0; the attribution travels in each assets
  file), cached under `~/.woof/cache/energy-osm/`
  (`WOOF_ENERGY_OSM_CACHE`). `woof energy import` reads PyPSA-Eur
  network CSVs, the UK REPD, GeoJSON and CSV. `woof energy sites`
  samples lines every `--spacing-m` with the conductor bearing and adds
  substations, turbines at hub height and PV farms. `woof energy run`
  runs a plan, `woof energy extract` samples it at the sites, and
  `woof energy rating` computes IEEE 738 line ratings, Makkonen
  conductor icing, wind power and PV power. Four versioned files
  (`woof-energy.assets.v1`, `sites.v1`, `plan.v1` and `forecast.v1`)
  connect the stages.
- `woof energy plan` has three topologies. `wrf-nests` is one run with
  up to 20 sibling nests. `wrf-tiles` is one regional parent run plus
  any number of offline `woof downscale` tiles along the corridors: an
  irregular WRF topology for long lines and national grids. `hex-swath`
  is an MPAS mesh refined along the corridors, subject to the mesh
  generation gates.
- The `energy` history preset (`[output] preset = "energy"`) keeps the
  fields the extractor reads, including direct normal and diffuse
  shortwave irradiance (`SWDDNI`, `SWDDIF`) for the forecast's `dni` and
  `dhi`.
- The Rust site sampler (`librw_sitesample.so`, override with
  `WOOF_SITESAMPLE_BRIDGE`) interpolates wrfout history to sites and
  heights above ground. It interpolates bilinearly on mass points and
  linearly in height, and it never extrapolates.
- `woof doctor` reports `woof energy` in a non-blocking row: whether the
  package imports, the Overpass cache size, whether the site sampler is
  on disk, and whether the optional `xarray`, `zarr` and `icechunk`
  packages import. It contacts no host. The guide is
  `docs/energy-forecasts.md`.
- The hex (MPAS) model reaches 100 m and 50 m cells. An opt-in
  experimental timestep lane (`woof hex forecast --experimental-dt`,
  `WOOF_HEX_EXPERIMENTAL_DT=1`) admits whole-millisecond timesteps below
  the 5 s anchors, 0.5 s at 100 m and 0.25 s at 50 m, when Courant,
  radiation divisibility and binary64 clock closure hold. Every receipt
  and history file is stamped `experimental-unanchored`, and the anchor
  table is unchanged. Below the anchors the CUDA path recovers theta
  increments with compensated summation.
- The MPAS v8.4.1 LES closure is ported: `woof hex forecast --les-model
  {off,3d_smagorinsky,prognostic_tke}` with `--pbl off`. The column takes
  its level count from the vertical file, up to 256 levels, and the `les`
  vertical preset puts fine layers near the ground.
- Adaptive meshing: `woof hex density` writes `woof-hex.density.v1`
  spacing rasters from sites, assets, terrain or forecast fields, and
  `woof mesh --density-raster` generates from them. `woof mesh
  --regional-window` relaxes only inside a window, and generated meshes
  default to incremental triangulation. `woof hex remap` moves a state
  between meshes conservatively, and `woof hex adapt` turns forecast
  history into the next cycle's density raster.
- New hex doors: `woof hex register`, `woof hex vertical`, `woof hex
  static-highres` (GLO-30 terrain and CGLC land use), `woof hex
  intermediate --source {gfs,gdas,ecmwf-open-data,era5,aifs}` through the
  Rust `met_intermediate`, and `--source wrfout` to force a hex limited
  area from a WOOF WRF run. Hex history gains `SWDDNI`, `SWDDIF`,
  `COSZEN`, `zgrid` and `xtime`, and output-variable selection.
- `woof energy plan --topology hex-swath --hex-experimental-dt` emits the
  full hex command chain, and `wrf-tiles` tiles now carry the LES physics
  and the `energy` output preset through `woof downscale --child-config`.
- `components/hex/tools/mint_dt_anchor.py --regional-plan` plans a
  sub-second timestep mint on a limited-area cull without running it.
- Every physics scheme in the regional, hex (MPAS) and global models runs
  up to `nz = 256` vertical levels, which is also the acoustic solver's
  ceiling. Before, WSM6 and WDM6 stopped at 80 levels; Kain-Fritsch, YSU,
  Shin-Hong, MYJ, MYJ-urban and SASE at 128; and both longwave radiation
  arms at 128 layers, which is 115 levels at `p_top = 5000` Pa. Radiation
  now sizes its column for each run as `nz` plus the cap layers above the
  model top, so `p_top` no longer lowers the ceiling. WSM6, WDM6 and MYJ
  keep their columns in global workspaces. Device sizing prices each
  per-thread frame that grows with `nz` per level. Those frames were
  re-measured on sm_86 only. `woof check` refuses a deeper column and names
  the scheme. No full-physics real-data run above 192 levels has been
  integrated yet.
- The hex model's level ceiling now comes from the physics a run selects:
  the smallest bound of its schemes in `woof/physics_vertical_contract.py`,
  capped at the hex column's own 256. Before, it was a copy of WSM6's 80,
  which also refused P3 and aerosol-aware Thompson columns that those
  schemes can run. `woof hex forecast`, the mesh bind, `woof energy plan
  --nz` and the dt-mint plan use it. The forecast driver checks again with
  the run's own microphysics and model-top pressure before it seals the
  physics seam. The YSU gravity-wave-drag kernel compiles for the run's
  column up to 256 levels (was 160). Device admission prices it at 44 B
  per level: 11,392 B per thread at 256 levels. The `les` preset still
  defaults to 80 levels. Timestep anchors and regional classes are still
  measured at 55 levels only.
- `woof energy hindcast plan|status|materialize|run-local` runs a long
  reanalysis as a chain of cold-started segments. Each cluster and date
  range becomes daily `woof-energy.hindcast-job.v1` jobs: a 30 h run with
  6 h of spin-up, keeping 24 h. Lengths and cycle hour are configurable,
  and the job ids are deterministic. Every job records the boundary
  source chosen for its segment and whether it changed from the previous
  segment. On `wrf-nests`, `history_begin_s` is set on every domain and
  leaves write every 900 s (or finer, such as 300 s). Each job carries
  Icechunk targets and a cost hook. A `woof-energy.hindcast-manifest.v1`
  manifest tracks status and resume. See
  `docs/energy-hindcast/driver.md`.
- `woof domain` takes an explicit model column: `--p-top-pa`,
  `--eta-levels` (a comma list or a file) and `--dzbot`, `--dzstretch-s`,
  `--dzstretch-u` and `--max-dz` (WRF 4.6.1 `compute_eta` level
  generation). The column is written as explicit `eta_levels` and
  `p_top`. It must run strictly from 1 down to 0 with at most 256 levels,
  and its top must lie inside the source's reach. `woof energy plan
  --topology wrf-nests|wrf-tiles` takes the same flags. With any of them
  set, the planner builds the wind-energy column
  (`woof.energy.vertical`). Its defaults are a 5500 Pa top (19.97 km),
  70 levels and a 12 m lowest layer. It refuses a model top above 20 km
  in the standard atmosphere and any column with fewer than 8 layers
  below 250 m. The column record is written to `vertical.json`
  (`woof-energy.vertical.v1`). Without the flags, nothing changes. See
  `docs/energy-hindcast/vertical.md`.
- `woof energy hindcast optimize` ranks leaf spacing x levels x segment
  layout x cloud offer by estimated cost and keeps the rows that meet
  `--deadline-days` (and `--max-instances`). It re-partitions the sites
  in-process once per card memory and spacing, with the cuts cached under
  `--partition-dir`. Every row is marked an estimate with its throughput
  and utilisation band and the price table's age. The chosen row carries
  the partition, estimate and plan commands, and its cost at 0-3 h of
  spin-up. The output is `woof-energy.cost-optimize.v1`. See
  `docs/energy-hindcast/cost.md`.
- The cost model now has a documented utilisation term. Throughput is not
  linear in cells: an occupancy knee plus a fixed per-step cost, in three
  assumed pairs (optimistic, central, pessimistic). Each pair has `K`
  refitted on the receipts, and the band is the envelope over the pairs.
  Every offer reports its utilisation, so a small domain is no longer
  priced as if it filled a big card. These are assumptions to replace
  with a cells sweep on the rented card.
- The price table has AWS `g6.xlarge`/`g6.12xlarge` (L4 24 GB) and
  `g5.12xlarge`/`g5.48xlarge` (A10G) rows, with on-demand prices as of
  its 2025-07-01 date and illustrative spot prices.

Changed:

- Inputs are verified once, not on every re-run. Re-running `woof go`
  on a finished 33 GB ICON cycle spent about 66 minutes in the fetch
  stage re-hashing files it had already proven. The forecast preflight
  and the post-run check then read the prepared inputs again.
  - After a full read proves a file's or prepared-cache array's SHA-256,
    woof keeps a digest stamp of the file's device, inode, size, mtime and
    ctime in a per-user cache (`~/.cache/woof/verify-stamps`;
    `WOOF_VERIFY_STAMPS` moves it, or `off` disables it). A later check of
    a stat-identical file reuses the digest without reading it. Any
    differing field, or a missing or corrupt stamp, means a full read, and
    a file changed less than 2 s before its read is never stamped. This
    applies to `woof fetch`'s already-present check, `woof go`'s
    reused-preparation check, and the forecast preflight.
  - The post-run "inputs changed during execution" check is now a stat
    comparison against the preflight's reads. A rewrite always moves
    ctime, so a rewritten input is still refused. A change that keeps
    every stat field is not seen; `WOOF_VERIFY=full` or
    `--verify-inputs full` on `woof fetch`/`woof go` reads every byte at
    every check, as before. `full` is the default on Windows, where
    `st_ctime` is the creation time. An unknown `WOOF_VERIFY` is refused
    before anything is fetched or read.
  - Receipts say which way each input was proven. Fetch manifests have a
    `verification` block and a per-file `verified` (`hashed`, `memo` or
    `stamp`), `report.json` has `input_verification.preflight`/`post_run`,
    and a reused preparation's decision has `verification`.
  - Full reads are faster too. Files and arrays are read 8 at a time
    (`WOOF_VERIFY_WORKERS`) in 16 MiB blocks; prepared arrays are hashed
    without loading them into memory. A table-route fetch (ICON, the DWD
    tree) now asks once whether the request is already on disk. Before,
    it asked up to three times, re-reading every file each time. See
    DATA.md and PIPELINE-STAGES.md.
- Hindcast defaults are now the cost-efficient end of the requirement.
  On the US + EU-offshore partition (AWS spot, central) they bring the
  year from about $439k (200 m, 70 levels, 8 h runs with 2 h spin-up, on
  g5.xlarge) to about $133k:
  - `woof energy partition --leaf-dx-m` and the hindcast leaf spacing
    default to 300 m, not 200 m and 100 m. 100-250 m is on request.
  - `woof energy hindcast plan --nz` defaults to 49
    (`woof.energy.vertical.HINDCAST_NZ`). That is the smallest wind-energy
    column with at least 8 layers below 250 m, a 12 m lowest layer, a top
    at most 20 km and no layer deeper than 1000 m, and it keeps the
    70-level preset's 10 rotor layers at about 0.76x the cost.
  - Segments are 9 h runs from each 00/06/12/18 UTC init with 3 h of
    spin-up, keeping `[init + 3 h, init + 9 h)`. They were 6 h runs with
    no spin-up. `--segment-hours 6 --spinup-hours 0` restores the old
    layout at about 0.68x the cost. `--segment-hours 6` alone is refused
    rather than read as 3 h windows off the 6-hourly cycle.
- `woof energy estimate` defaults follow the driver: 9 h segments with
  3 h spin-up, 300 m leaves and 49 levels. Without `--segments`, the
  count stores one year (`ceil(8760 / kept hours)`, 1,460 by default).
  The old defaults were 365 x 30 h with 6 h spin-up, 200 m and 60 levels.

Fixed:

- A fine nest over steep ground could go non-finite within seconds. A
  100 m nest whose 30 m Copernicus terrain read a 1.56 slope (57 degrees)
  on a canyon wall went non-finite at its fourth step, two model seconds
  in. Halving its step, six acoustic substeps, `epssm` 0.9 and `diff_opt`
  1 did not help; smoothing the ground did. High-resolution terrain
  (`[static.highres]`, the default at 1 km and finer) now gets a
  steep-ground pass at preparation: ground steeper than 0.85, the
  steepest slope any measured acoustic substep count holds, is smoothed
  down to it by local 1-2-1 rounds, and only that ground. Other ground
  keeps its exact bytes, so flat and gentle domains are unchanged, and
  the run clock is untouched. The pass reaches at most 2.5 km and runs at
  500 m and finer (where its rounds leave ground still steeper, the
  preparation says so); the terrain window is read that much wider so a
  moving nest's corridor crop still equals its footprint. The static
  receipt records what moved under `steep_ground`. Re-prepare to pick it
  up; no configuration change is needed. Since the 1.1.0 merge this pass
  is off: 1.1.0's terrain auto-smoothing smooths every route's terrain to
  the run's measured slope limit and records it as `terrain_autosmooth`.
- Nested stability recovery never saw the most common blow-up. The
  per-step gate raised a plain `RuntimeError` for a non-finite state, so
  recovery only caught the every-fourth-step full-state validator and
  `stability-recovery.json` read `attempts: []`, `failures: []` and
  `FAILED` with no reason. The gate now raises
  `woof.core.lagged_health.StepStabilityError` (still a `RuntimeError`,
  same message). Recovery records it and either retries or says why it
  cannot under `refusal`: a moving or fixed-clock tree, or one with no
  checkpoint yet.
- A long moving-nest run could run out of GPU memory at a move. An
  asynchronous checkpoint kept references to the live state arrays it had
  already copied to host until it was published, so a nest move or a
  containment slide while a checkpoint was being written could not free
  the outgoing domain, and its rebuild put a second copy of that domain on
  the card. With a 256-level 240 x 240 / 324 x 324 nest pair that is about
  7 GB, and it outlasted the forecast's priced envelope on a 48 GB card.
  Measured at 64 levels: +691 MiB at a slide and +1,075 MiB at a move at
  every checkpoint boundary before the fix, nothing after it. The
  checkpoint now keeps only paths and sizes once its snapshot is queued.
  Before a relocation releases a domain, the in-flight snapshot and any
  queued `direct`-mode history frame drop their device references, and the
  domain's named scratch slots are released with the rest of its arrays.
  No output byte changes.
- A moving nest with a `[relocation.containment]` parent could die mid-run
  at a containment slide. The runner clamped the slide to
  `[relocation.containment] max_move_parent_cells`, then judged it against
  the mover's own `[relocation] max_move_parent_cells`, so any slide wider
  than the mover's cap refused with "the move is N parent cells, over the
  configured maximum". This hit `woof relocation from-track` itineraries,
  whose containment cap is wider than their row cap. A slide is now judged
  against the containment's cap.
- `woof go` refused a sealed mapped single-domain preparation (for example
  `icon-global-ml`) at the forecast stage, calling it "a multi-domain
  hierarchy product", because its proof records no top-level
  `input_manifest_sha256`. It now binds the evidence manifest, the one
  `woof sim` binds.
- `woof energy estimate --partition` silently ignored `--leaf-dx-m`, and a
  job's spacing, when the partition's clusters carried their own
  `leaf_dx_m`. A spacing that disagrees is now refused, and the refusal
  says to re-partition (`woof energy partition --leaf-dx-m`, or `woof
  energy hindcast optimize`, which re-cuts per spacing).

- `woof energy hindcast run-local --force` on a job the manifest called
  complete marked it `running` and then returned "skipped", leaving it
  stuck; a forced run now runs.
- `woof energy hindcast materialize` of a partition cluster handed
  `woof.energy.partition.plan_cluster` a `sites_path` it passed on to the
  planner (a second one on wrf-nests, an unknown one on wrf-tiles), so
  every partition job was refused with a TypeError; `plan_cluster` binds
  its own sites file.
- `woof go CONFIG --prepare-only` now stops once the prepared bundle is
  sealed on every prepared route, prints `{"status": "PREPARED",
  "prepared_root": ...}` and exits 0. Before, a config on the staged
  route (ICON, IFS and the other `woof prep` sources) or the native HRRR
  route was sent to the full run plan: it printed `fetch -> prepare ->
  forecast`, prepared, sealed and then ran the whole forecast. The run
  plan now carries the stop as `run_options.prepare_only`, so the launch
  keeps its gates, `launch.log`, events and shared download cache. The
  plan line reads `fetch -> prepare`, and `--dry-run` with it lists only
  those stages. `--prepare-only` with `[case_data]`, `--prepared-root`,
  `--restart`, `--wps-namelist` or `--restart-roster` is refused, since
  those inputs already exist. Under `--prepare-only` the forecast memory
  gate warns instead of refusing and the forecast-device check is
  skipped, since no forecast starts.
