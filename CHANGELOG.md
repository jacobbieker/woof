# Changelog

## 1.0.2

WRF-fidelity fixes correct moisture pressure terms, edge vertical velocity,
diffusion and map factors, and the 2 m humidity cap. Forecast answers change.

New:

- `[devices]` in a single-domain or tree config, or `woof go --devices N`,
  splits a forecast across cards, with per-card memory checks and an
  execution receipt. `domains` selects split grids, defaulting to all;
  split nests take forcing each parent step and return two-way feedback
  to split parents. `woof sim --devices N` or `--devices-table` sets a
  prepared forecast's split without changing its configuration.
  `woof check CONFIG --devices N` prices every card, grid and pinned host
  store, exiting 2 for an unknown VRAM budget. Frames reach the host while
  the model steps on. Split parents over resident nests, moving nests and
  late-starting split grids are refused. Split grids refuse Noah mosaic
  and slope radiation (`slope_rad = 1`) before allocation, naming the
  missing state; resident grids retain both. Splitting is off by default.
- `woof run-plan` starts GFS and native HRRR single domains and trees from
  the first posted hours, as `woof go` does. Both start single `hrrr-prs`,
  `rap`, `rrfs`, `icon-eu` and `gem-gdps` domains the same way. A `[tiles]`
  root, including `woof cyclone-setup` trees, waits at each boundary seam,
  instead of for the whole cycle. Output is unchanged. An earlier forecast
  failure keeps its own error beside any later input timeout.
- `WOOF_FETCH_POLICY=aws` pins sources with an AWS host to that host without
  NOMADS fall-through; sources with no AWS host retain their usual host.
  `latest` selects the newest cycle whose first hours are on AWS;
  `--whole-cycle` retains the whole-cycle rule. Forecasts still stream from
  their first posted hours. Unset, fetching is unchanged. `woof fetch`
  probes hosts refusing HEAD with a one-byte GET.
- `woof import-namelist` accepts `domain_wizard`, `mod_levs`, `plotfmt` and
  bare `30s` geography, and fills 17 of 33 formerly required keys from WRF
  and WPS defaults. Unknown sections are refused by name.
- `diff_opt = 1` runs WRF's coordinate-surface horizontal diffusion with
  `km_opt` 2 or 4 and either `mix_full_fields` setting, including restarts
  and streamed forecasts. It is off by default (`diff_opt = 2`).
- `topo_wind` runs under YSU, excluding BEP and BEP+BEM;
  `gwd_opt = 1` or `3` runs KIM or GSL gravity-wave drag under every PBL
  scheme. GPU column checks match WRF 4.7.1 bit for bit. Static preparation
  adds orographic statistics from WPS geography, and namelist imports keep
  both keys. Orographic statistics are byte-identical on Linux and Windows.
  A streamed `[tiles]` domain stops at its first drag call.
  Both options are off by default.
- `woof ml-export` converts regional histories, converted hex frames,
  global tapes or their ZIPs into one Zarr store per domain. It offers
  WeatherBench 2's 13 pressure levels, ERA5's 37 or model levels, ERA5
  and WeatherBench 2 names and units, earth-relative winds, a
  `below_ground` mask over ERA5's below-ground fill, an optional regular
  latitude-longitude grid and a ZIP that opens in place. Levels above the
  model top are omitted and listed. `xarray.open_zarr` works with
  zarr-python 2.18 and 3. On Windows, ZIP history files with WRF's colon
  filenames unpack with underscores; see `docs/ml-export.md`.
- GPU preparation rotates winds, interpolates vertically and builds the
  Thompson cold start with the CPU route's arithmetic, matching its
  prepared arrays. Forecasts prepared on the card change. A full HRRR
  forcing preparation took 154 s instead of 640 s.
- Acoustic, advection, horizontal diffusion and per-step glue kernels move
  less memory, with byte-identical output. The horizontal diffusion change
  alone reduced an hour of an 800 x 600 x 50 forecast from 95.7 to 88.4 s
  on an RTX 5090.

Fixed:

- Legacy RRTMG longwave radiation keeps the correct spacing above the
  model top. A pressure guard had compressed valid layers, changing their
  temperatures and heating. Forecast answers change.
- `woof go --cycle` accepts an exact match to a prepared bundle,
  checkpoint or declared forcing's actual initial time, recording a no-op.
  Conflicting or unreadable input times remain refused. Forcing fetched by
  the run is checked after acquisition and before preparation.
- Every shipped profile and configuration leaving `moist_cq` unset applies
  WRF's moisture correction to pressure terms, including passive vapour
  with microphysics off. Dry states bypass it; an explicit setting wins.
  Unmatched configurations had omitted it, and WSM6 and Kessler profiles
  had disabled it. Forecast answers change.
- The dynamical core corrects vertical-velocity damping and float32
  rounding, geopotential tendencies at open and specified outer rows,
  vertical advection at an open model top, mapped open-boundary map
  factors, and diffusion's sixth-order filtering, boundary deformation,
  outer-row TKE and prescribed surface heat flux. Routine output words
  are checked against compiled WRF 4.7.1. Forecast answers change.
- Surface vertical velocity at open and specified edges uses WRF's
  clamped terrain slope. Copying the interior slope outward had doubled
  its terrain-following term there. Periodic domains and interior cells
  are unchanged; edge answers change.
- Land points cap 2 m humidity at 1.05 times the lowest model level's
  vapour after the surface schemes, as WRF's surface driver does.
  Answers change for 2 m humidity.
- WRF `wrfinput` starts initialize unfrozen Noah soil liquid water from
  soil moisture and terrain-following vertical wind from surface flow,
  instead of retaining zeros. First-hour latent heat flux RMSE against WRF
  fell from 239 to 0.4 W/m2 on a 3 km comparison. With `usemonalb` off,
  Noah keeps WRF's seasonal background albedo instead of the file's monthly
  albedo. Answers change for these forecasts.
- `woof downscale` children of saved runs, including `--point`, build
  terrain, land use and soil at their own spacing, using Copernicus GLO-30
  terrain at 1 km or finer. Parent states and every boundary frame are
  blended and rebalanced as WRF's `ndown` does, within a few float32
  epsilons of compiled WRF 4.7.1 on column checks. A WPS_GEOG tree is
  required (`woof fetch-geog` or `--geog-root`); a missing tree is refused.
  `--parent-terrain` retains the old route, and `[static]` records the
  choice. Answers change for these children.
- Downscaled children and DA nests take steep-terrain acoustic substeps
  and the `epssm` floor from their own terrain, keeping automatic parent
  choices. Lower explicit `epssm` values are refused. Answers change when
  the child's terrain needs the rule and the parent's terrain did not.
- Adaptive steps land on history, radiation, cumulus, surface and
  boundary-layer times; history intervals need only be whole seconds.
  Prepared forecasts honor adaptive history alarms and delayed domains
  start on time. Answers change where those times fell between steps.
- Disk estimates count each domain's physics fields, `history_vars` and
  output window. Trimmed runs that fit start; insufficient space is
  refused before starting.
- BEP+BEM workspace estimates count urban columns, including forecasts
  starting while their source posts, keeping a forecast that fits on the
  card. Configuration checks explain their upper bound.
- A resident nest under a streamed or split parent restores from the
  shared scratch arena without a `KeyError`.
- Streamed final-state digests use the resident forecast's model time.
  Fractional adaptive steps no longer leave identical forecasts with
  different digests through a 0.27 ms time difference after 3 h.
- Resident, streamed and out-of-core checkpoint writers record the run
  configuration alike. Checkpoints of the same state no longer differ in
  their header.
- CPU-only validation and SASE resource accounting work without CuPy.
  Windows numerical receipts resolve their own paths.

Available since 1.0.1:

- `sf_urban_physics = 1` runs single-layer urban canopy under Noah or
  Noah-MP; `2` runs BEP with YSU or MYJ; `3` adds BEP's building energy
  model. Local Climate Zones and NLCD urban classes are retained. A
  single-layer canopy above the first model level is refused. Urban
  physics is off by default; BEP under YSU applies nonurban drag once.
- `sf_surface_mosaic = 1` runs Noah land-use tiles per cell. With the
  single-layer canopy, `mosaic_urban_canopy = "every_tile"` also treats
  town tiles in mostly rural cells. Mosaic is off by default.
- The UW moist-turbulence PBL (`bl_pbl_physics = 9`) runs on the GPU and
  imports from WRF namelists. `[shared] zadvect_implicit = 1` runs
  implicit-explicit vertical advection with `w_crit_cfl`, including the
  two corrected boundary terms. Implicit advection is off by default;
  answers change.
- Terrain accepts WPS GEOGRID.TBL smoothers or no smoothing through
  `woof domain --terrain-smoothing`, domain static settings or GEOGRID.TBL
  import. `smooth_precision = "wps-float32"` or
  `--terrain-smoothing-precision` matches default `geogrid.exe` smoothing;
  float64 remains the default.
- WRF namelists retain `history_begin`, `history_end`, `smooth_cg_topo`,
  `topo_shading` and `slope_rad`, and imports name all unsupported keys
  together. Restarts can change history windows where cadence can change.
  WUDAPT Local Climate Zones import with their urban land cover.
- Byte-identical physics speedups remain included: per call, legacy
  RRTMG 7x, RUC 7 to 22x, Noah-MP 5.8x, MYNN 4.5x, NSSL 1.9 to 3.1x,
  P3 1.5 to 2.2x, RTE-RRTMGP 1.6x, Thompson 1.3 to 1.6x, Morrison
  1.2 to 1.5x, and Milbrandt-Yau, Kain-Fritsch, New Tiedtke, Shin-Hong
  and Grell-Freitas 1.1 to 1.5x. Dycore bookkeeping and nest forcing
  use fewer launches.
- `[shared] adaptive_nest_lattice = true` selects the root step with the
  fewest nest cell-steps; the three-domain 250 m comparison was 6% faster.
  It is opt-in and changes answers.
- `woof fetch` takes posted leads in order and records their first
  availability or an already-up marker; late leads exit 75, AIGFS waits
  for its GDAS donor, and events carry every source's leads.
  GFS, native HRRR and mapped-source trees prepare their start and nests
  first, then stream later boundaries; storm-following trees start at
  that head too.
  `woof sources` shows posting schedules and lateness budgets; fetch
  `latest`, `--readiness` and `--whole-cycle` remain available, and
  `woof go --cycle` selects the cycle. HRRR DA uses expected posting
  times, and `latest` probes AIFS f360 from +5 h 26 min. Config preparation
  can pin its CPU or GPU backend with `woof run --preprocess-backend` or
  `[case_data] preprocess_backend`, and `--preprocess-workers` sets decode
  concurrency. A 48 h HRRR decode on 24 cores took 140 s instead of 480 s.

## 1.0.1

The engine's speed work, forecasts that start before their inputs finish
posting, and the fixes made since 1.0.0.

New:

- Faster physics, output byte-identical: legacy RRTMG radiation about 7x
  faster per call (product-suite forecasts up to 13% faster), RTE-RRTMGP
  1.6x faster per call, MYNN about 4.5x faster per call, and RUC land
  surface as six fused kernels instead of up to 3,000 launches (7 to 22x
  faster per call). The dycore step and nest forcing make fewer launches
  and host waits (nest forcing takes 25% less time).
- `WOOF_PHYSICS_MEGAKERNEL=1` runs the physics glue as a few fused kernels:
  a 100 x 80 default-suite step is 1.12x faster and forecasts are
  byte-identical. It is off by default, and a receipt names it when on.
- `[shared] adaptive_nest_lattice = true` picks the adaptive outer time step
  with the fewest nest cell-steps (6% faster on a three-domain 250 m tree).
  Opt-in: answers change.
- Sources are fetched as they post: `woof fetch` takes each forecast time as
  soon as a host has it, `--cycle latest` needs only the first ones, a time
  that passes its late limit exits 75, `--whole-cycle` keeps the old rule and
  `--readiness` says whether a run can start. `woof sources` shows each
  source's posting schedule and lateness budget.
- Domain trees from HRRR pressure levels, mapped sources and GFS prepare
  their start time and nests first, then one boundary interval at a time, and
  the forecast starts on that head, waiting at any interval not yet prepared.
  Output is identical to waiting for the whole preparation.
- The source decode uses every core (`woof prep --preprocess-workers` sets
  the count): a 48 h HRRR decode on 24 cores took 140 s instead of 480. Prepared arrays
  are unchanged.
- GEM GDPS, and every regular latitude/longitude source, decodes only the
  window its domains read: a 6 h GDPS preparation took 32 s and 1.5 GB of
  memory instead of 175 s and 12 GB.

Fixed:

- With `hypsometric_opt = 2`, which `woof domain` writes, a CPU-prepared
  state takes glibc's `log1pf` on every machine, so it no longer differs in
  the last bits on AVX-512 machines (the 1.0.0 known issue). The default
  hybrid coordinate and the preparation's exp, log and pow are also the
  same on every machine now.
- Under the adaptive clock, the steep-terrain cap on `max_time_step` comes
  from three-hour runs of the adaptive clock itself: a 2.25 km ridge under a
  24 m/s crest wind keeps 30 s and a 3 km CONUS domain keeps 24 s up to
  30 m/s, where 1.0.0 capped them shorter.
- `--cycle latest` waits each publisher's measured delay for that cycle hour
  and no longer skips posted cycles of seven sources. ECMWF's "slow down"
  answer is retried for up to 10 minutes.
- An AIFS forecast starting after f000, and a GEM GDPS window starting after
  f000, now prepare: the land mask and terrain come from the cycle's f000.
- A storm-following nest's 1 h rain draws at the hours it moved, and ground
  it moves onto starts from the parent's rain total.
- A nest that starts after the forecast draws live and through
  `woof render`, where every frame was refused.
- A forecast waiting for a boundary interval says so on its heartbeat,
  progress file and events, and `woof go` no longer stops it after 120 s.
- Two card preparations of the same inputs now publish identical prepared
  caches; per-run measurements stay in the receipt.
- In a cloud-cover picture, clear sky under the first level (10%) draws
  nothing, so the map shows through instead of a white sheet.
- Hex and global maps and cross-sections name their model, WOOF Hex and
  WOOF Global, in the line under the title, where they said WRF.
- The global model's quickstart configurations name the `woof` commands to
  fetch its analysis and run it.
- The identity texts name WOOF (the 1.0.0 known issue): the pin tables
  `woof global pins` and `python -m woof.globe.spectral pins` print, and the
  hex model's contract texts and messages. They carry new ids (global pins
  v3, Level-3 pins v2, hex adapter contract v3) and no arithmetic moved.
  1.0.1 reads the 1.0.0 ids as the same arithmetic, so a global checkpoint,
  export or receipt written by 1.0.0 resumes and validates.
- A global receipt's `libraries` block records the `recast-woof` version.
  1.0.0 receipts recorded `recast-woof-data` and not the package itself.

## 1.0.0

Recast WOOF 1.0.0 is the regional engine of release 2.8.0 and the fixes made
since, under the WOOF name, with the hex and global models inside it and the
next plots.

New:

- One install, `pip install recast-woof`, carries the regional model (`woof`),
  the hex model (`woof hex`, preview) and the global model (`woof global`,
  preview). One version covers all three.
- The Python package is `woof`, the command is `woof`, the table companion is
  `recast-woof-data` and settings are spelled `WOOF_*`. Settings spelled
  `GPUWM_*` keep working.
- Plots take the shape of their domain: a square or tall domain fills the
  picture instead of leaving white space, the header runs the full width, the
  colour bar matches the map's length, wind barbs are larger and place names
  are drawn on every domain. County lines are left off at continental scale.
- `woof render --diff A_RUN B_RUN` draws any product as run A minus run B on a
  zero-centred colour bar, frame by frame at each valid time both runs hold.
  Runs on different grids or times are refused by name.
- `woof hex` runs on any NVIDIA GPU that can run its kernels (compute
  capability 7.5 and newer, with a CUDA 13 runtime). Each run's receipt
  records whether the card is anchored, meaning its architecture's numerical
  contract was measured against the reference card's: the RTX 40 series and
  the L40S (sm_89) are anchored, as are sm_86 (the RTX 30 series) and sm_120
  (the RTX 50 series). Any other card runs unanchored, and its receipt says
  so. `woof hex doctor` names the card, its compute capability and whether
  it is anchored.
- The global model reads its trace-gas and ozone climatology from a shipped
  table instead of the RFMIP input file; its forecasts are unchanged, bit for
  bit.

Fixed:

- A forecast from HRRR (native or pressure levels), ICON-D2 or a mapping that
  declares cloud water, rain, ice, snow and graupel now brings them in through
  its outer edges, instead of water vapour alone, so the analysed cloud and
  snow no longer drain out of the domain. Sources that publish none, such as
  GFS and ERA5, are unchanged. A forecast started from WPS met_em files or
  from wrfinput and wrfbdy files keeps water vapour alone at its edges, as
  WRF's real.exe writes them.
- Snow-covered land whose analysed top soil is more than 30 K colder than the
  snow surface, as older HRRR analyses carry under western snowpack, now
  starts with its soil rebuilt from the skin temperature down to the deep soil
  temperature. On a 2017 Idaho case the 2 m temperature bias at snow-covered
  stations over the first 3 hours went from -2.5 K to -0.2 K.
- A nested domain on its parent's vertical levels, including one spawned or
  moved during a run, now gets the same pressure correction as the outer
  domain. Its base state was held at single precision, which zeroed that
  correction.
- A nest spawned during a run is checked against the card's free memory
  before it is built. A card that cannot hold it stops the run with a message
  naming the nest's sizes and the memory free, instead of a CUDA out-of-memory
  error part-way through. Moving a nest from Python now stages through host
  memory by default, instead of holding the old and the new nest on the card
  at once.
- A forecast prepared with engine release 2.8.0 still runs, on
  water-vapour-only edges (it says so in one line) and with the soil it was
  prepared with. Prepare it again to get the fixes above.
- A forecast too big for the card's free memory is refused before anything is
  allocated, naming what it needs and what the card has, instead of stopping
  in a CUDA out-of-memory error part-way through. `woof run` (one domain or
  a domain tree, before anything is downloaded), `woof downscale`, the
  prepared and domain-tree forecasts and the loaders price the state,
  physics and boundary tables first, and `--no-memory-gate` skips the check
  wherever a forecast runs. A pinned `[tiles]` tiling that does not fit
  names the largest tile that does.
- Every memory check prices the tables that carry a source's cloud and
  precipitation at the edges: the `[tiles]` decision, the check before a
  forecast is built, the card reservations of `woof stream` and `woof
  multi-run`, and the host memory a streamed forecast keeps its edges in.
  They priced water vapour alone, so a HRRR-forced run could be let onto a
  card or a host that could not hold it. `woof check`, `woof run-plan
  --estimate` and the sizing table of `woof domain` count the same tables in
  every figure they print.
- A `woof run` preparation on the card prices the memory pool's reserve at
  1.25 times its arrays instead of 1.20, which a measured run exceeded, and
  `woof check` prices it the same way.
- A `woof run` domain tree whose nest starts after the forecast and streams
  through `[tiles]` now runs past the nest's start, where it stopped at the
  nest's first step. A nest that starts late is checked against the card's
  free memory before it is rebuilt at its start, and no longer holds two
  copies of itself in memory there.
- A domain finer than 1 km made by `woof domain` with no physics named now
  runs Thompson microphysics with MYNN and RUC instead of the source's YSU
  and Noah default. Scored against stations and ceilometers on marine fog
  days at 750 m, that suite kept more coastal fog and low stratus (stratus
  CSI 0.71, against 0.52 with Noah and 0.55 with YSU and Noah). This is
  limited observation evidence: the selection record publishes neither a
  case count nor a reproducible score receipt. A source whose soil RUC
  cannot start from keeps its own default.
- `woof domain` refuses a RUC land-surface suite on a source whose soil RUC
  cannot start from (GEM GDPS publishes one soil layer and RUC needs two),
  instead of leaving the refusal to the preparation after the download. A
  `--source hrrr` domain with nests now runs a RUC suite.
- `woof local-da` on a rung finer than 1 km with no physics named now runs
  Thompson with MYNN and RUC, as every other door does at that spacing,
  instead of the source's YSU and Noah default. A published local DA case on
  `--source hrrr` now launches, where it was refused as an incomplete roster.
- A nested domain no longer gets stronger sixth-order damping than its
  parent. A suite that sets `diff_6th_factor` below 0.10 on the outer grid
  wrote 0.10 on its first nest.
- A forecast with nests that names its physics suite now runs from every
  source, with that suite on the outer grid and the nests as `woof domain`
  wrote them. On every source but GFS and ERA5 it was refused after the
  download and the preparation.
- A stock-WRF export handed no physics choice writes the schemes its
  preparation was made with into each wrfinput and wrfbdy, instead of the
  WSM6 and Dudhia values of the bundled reference file.
- Under the adaptive clock, a 3 km domain over steep, high ground under a
  strong crest-level wind has its `max_time_step` capped where a longer step
  was measured to stop. A 3 km CONUS domain keeps the default 24 s under a
  20 m/s crest-level wind and is capped at 15 s from 30 to 40 m/s. A cap
  only ever shortens the step.
- A native `hrrr` configuration with no route namelists beside it now runs:
  the run writes them from the configuration. One the route cannot carry is
  refused by `woof go --dry-run`, where it used to pass the dry run and fail
  seconds into the real run.
- `woof warm-kernels --all-profiles` warms all 28 shipped physics suites. A
  suite that still fails is named with its remedy and the pass goes on to the
  rest, where the first failure used to stop it.
- `woof cycle --parent-kind mpas-cuda` checks the hex mesh against the card's
  free memory before the first leg, and refuses a mesh the card cannot hold
  with both numbers, where it used to fail inside the forecast worker.
- `woof cycle --parent-kind mpas-cuda` runs against an MPAS port checkout in
  the port's current package layout. The forecast worker imported only the
  package's older name, so such a checkout passed the memory check and then
  stopped at its first import. A checkout holding neither package is refused
  by name.
- A prepared DA cycle's memory check (`python -m tools.da_cycle_prepared`,
  which the storm nowcast also drives) counts each leg's analysis, sized the
  way the solve sizes its chunks, and each member perturbation's FFT work
  areas, which are now released after each draw instead of staying on the
  card.
- A state prepared on the CPU and the Python map projections no longer
  depend on NumPy 2.5's AVX-512 math, which most cloud Linux machines run:
  the prepared pressure takes glibc's `powf` on every machine, and the
  projections take tan, atan, log, exp and pow from the C library, as the
  native tools do. With `hypsometric_opt = 2` one step still does (see
  Known issues).
- `woof render` imports each frame once instead of twice. Every picture is
  byte-identical; the nine long windows of a 37-frame run took a quarter of
  the CPU time.
- The 10 m wind maximum pictures are drawn from the stored maximum
  (`WSPD10MAX`) when a history carries it. From an hourly history without
  one they are titled as the largest hourly snapshot, where they called the
  top-of-hour wind a maximum.
- `simulated_ir_satellite` drawn from a wrfout shows the model's cloud tops.
  Its cloud optical depths were a thousand times too small, so most cloud
  drew the ground's temperature; it now matches WRF-Python's cloud-top
  temperature to 2e-5 K in every column.
- A domain tree whose configuration carries a `[perturbation]` block now
  prepares from native HRRR, HRRR pressure levels, the other mapped sources
  and ERA5 as it did from GFS, and the forecast applies the bubbles at its
  start. Those preparations used to refuse the block.
- `woof hex doctor` reports a bridge that is present but not executable as a
  gap and prints the `chmod` that fixes it, instead of reporting it found.
- `woof hex render` reads the renderer's catalog rows, which gained a sixth
  field in release 2.8.0. The hex model on its own refused every render
  against that renderer with "produced no catalog rows".
- A `woof hex` forecast's hour-0 frame carries a real 2 m humidity and
  surface pressure. It published a 2 m humidity of 0 and a surface pressure
  of 100000 Pa in every cell, so the hour-0 2 m dewpoint map was drawn from
  zero humidity while every later hour was right.
- Hex receipts name the machine by a salted digest, never its network name.
- The global model's level weights take pow and exp from the C library, so a
  configuration's identity, which checkpoints and receipts record, no longer
  depends on the CPU's vector math. With numpy 2.5 on an AVX-512 machine every
  shipped global configuration had a different identity.
- `woof global doctor` exits 0 on a correct install. It graded two engine
  functions that no documented command reaches as gaps and exited 1; both
  now print as optional notes, and calling either still refuses by name.
- The global model's six observation decoders (`rw_atms`, `rw_gnssro`,
  `rw_ndbc`, `rw_igra2`, `rw_amv`, `rw_wis2`) ship in WOOF's bridge bundle,
  which now carries 34 programs and libraries. `woof global doctor` grades
  all fourteen Rust doors against WOOF's own pins and exits 0 on a clean
  install, and `woof global fetch-doors` says there is nothing for it to
  stage.

Known issues, each fixed in 1.0.1 for the reason it names:

- A configuration with `hypsometric_opt = 2`, which `woof domain` writes,
  still takes NumPy's float32 `log1p` when it is prepared on the CPU, so on
  a machine with AVX-512 its prepared `al`, `alt` and `p` can differ from
  another machine's in the last bits. The fix is made in the engine and
  ships in 1.0.1: it changes prepared bytes, and 1.0.0's release checks ran
  on the engine before it, so it goes in with the next engine re-point
  and its checks.
- Two kinds of identity text still print the engine's earlier name: the
  pin table `woof global pins` prints, and the contract texts `woof hex
  forecast --preflight` prints. Their bytes are hashed into every global
  pin and every hex receipt, so a changed word changes every digest, and a
  global checkpoint written before would stop resuming. They keep their
  bytes in 1.0.0 and are reworded in 1.0.1.
