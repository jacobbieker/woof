# Choosing which variables the wrfout files carry

A forecast writes its whole history inventory every output period. On a
nested tree at forecast cadence that is what fills a disk: the 1 km child
writes an order of magnitude more bytes than its 12 km parent at the same
cadence, and most of them are three-dimensional volumes nobody on that run
is going to open.

WRF's answer is `iofields_filename`. WOOF's is the `[output]` table, and
it is a **selection over the inventory the run already produces** rather
than a second inventory of its own.

```toml
[output]
history_drop = ["QNRAIN", "QNSNOW", "QNGRAUPEL"]
```

That is the whole configuration. Everything else the run produces still
lands in the file, in the same order, with the same header.

## The default writes everything

`[output]` absent (or `preset = "full"` with no drop list) writes
exactly what every WOOF release before this one wrote: same inventory,
same variable order, same global attributes, byte for byte. Trimming is
something you ask for; it is never something a default does to you.
(The one addition to the full inventory is the [surface solar
fields](#surface-solar-fields), present when a radiation scheme
produces them.)

## Three ways to say it

### A named preset

```toml
[output]
preset = "minimal"
```

| preset | what the tape carries |
|---|---|
| `"full"` | **the default.** Every variable the run produces. |
| `"minimal"` | the 2-D surface state and the accumulators: `T2`, `Q2`, `TH2`, `PSFC`, `U10`/`V10`, `TSK`, the surface energy budget (`HFX`, `LH`, `QFX`, `GRDFLX`, `SWDOWN`, `GLW`, `OLR`), `PBLH`, the precipitation and snow accumulators, `UP_HELI_MAX` and the interval wind extremes (`WSPD10MAX`, `U10MAX`/`V10MAX`, `GUST`, `W_UP_MAX`/`W_DN_MAX`, produced under `nwp_diagnostics = 1`), the grid metadata and the land identity, plus the two structural volumes below. |
| `"severe"` | `minimal` plus the storm-scale volumes: `U`, `V`, `W`, `PH`, `P`, `PB`, the hydrometeor mixing ratios and number concentrations, and `REFL_10CM`. It sheds the restart-only scheme carriers no render product reads. |
| `"energy"` | what a forecast along power lines, substations and renewable sites is sampled from (`woof energy extract`): the column winds and theta (`U`, `V`, `W`, `T`), the height and pressure coordinate (`PH`, `PHB`, `HGT`, `P`, `PB`), `QVAPOR`, `QCLOUD`, `QRAIN`, `QICE`, `QSNOW`, `QGRAUP`, the near-surface state (`T2`, `Q2`, `U10`, `V10`, `PSFC`), the surface shortwave trio (`SWDOWN`, `SWDDNI`, `SWDDIF`, `COSZEN`), `RAINNC`, `RAINC`, `SINALPHA`/`COSALPHA`, the georeference, and the history-interval wind extremes `WSPD10MAX`, `U10MAX`/`V10MAX`, `GUST`, `W_UP_MAX`/`W_DN_MAX` when the run sets `nwp_diagnostics = 1` (see [Wind maxima and gusts](#wind-maxima-and-gusts)). |
| `"windfarm"` | **only 2-D planes**, for a wind-farm reanalysis whose reader is `woof energy extract`: the hub-height wind and its interval statistics plus the hub-height scalars (`T_HUB`, `P_HUB`, `RHO_HUB`, `QV_HUB`, `QC_HUB`, `QR_HUB`, `QNC_HUB`; see [Hub-height wind fields](#hub-height-wind-fields)), the near-surface state (`U10`, `V10`, `T2`, `Q2`, `PSFC`, `TSK`), the boundary layer and surface fluxes (`PBLH`, `UST`, `HFX`, `LH`, `SWDOWN`), `RAINNC`, `RAINC`, `SNOWNC`, the interval wind extremes (`WSPD10MAX`, `U10MAX`/`V10MAX`, `GUST`, `W_UP_MAX`/`W_DN_MAX`), `SINALPHA`/`COSALPHA`, `LANDMASK` and the georeference. **No 3-D volume**: it is the one preset that does not carry the structural `T` and `PHB` (see [What cannot be dropped](#what-cannot-be-dropped)). It needs `hub_wind = true` or `hub_heights_m`, and is refused without them. The reason for every field is the table `woof.io.history_selection.WINDFARM_FIELD_REASONS`. |

A preset is a filter over what the run produced, so a member the run
does not produce is simply absent from the tape, never zero-filled:
`severe` on a warm-rain scheme has no `QICE`, and `energy` has no
`SWDDNI`/`SWDDIF` unless the shortwave scheme computes a direct/diffuse
split (see [Surface solar fields](#surface-solar-fields)).

### What each preset buys, measured

Against one real 1 km frame of the proof run (49 levels, WSM6, RRTMG),
asking the real renderer how many of its catalog it can draw:

| tape | variables | `rw_wrfbatch` | products renderable |
|---|---|---|---|
| minimal, with no 3-D at all | 57 | **refuses the file** | - |
| minimal (`T` structural) | 58 | opens | 30 |
| + `P`, `PB` | 60 | opens | 38 |
| + `PH` | 61 | opens | 42 |
| + `QVAPOR` | 62 | opens | 66 |
| + `U`, `V`, `W` (≈ `severe`) | 65 | opens | 160 |
| the full inventory | 74 | opens | 162 |

`energy` was measured separately, on one 3 km frame of a 2-hour
HRRR-forced run (49 levels, Thompson, RTE+RRTMGP) with a newer renderer
whose catalog is larger, so its figure is its own pair and not a rung of
the ladder above: the full tape (82 variables) renders 207 products and
the same frame trimmed to `energy` (37 variables) renders 174. It keeps
every volume the ladder's large rungs need and sheds `REFL_10CM` (radar
products fall back to a generic hydrometeor dBZ), `MU`,
`UP_HELI_MAX`, `OLR` and the land-surface rows.

The first row is why `T` is structural: a tape with no `T` is not a
smaller wrfout, it is a file the estate's own reader will not open.

### An explicit keep list

```toml
[output]
history_vars = ["T2", "U10", "V10", "RAINNC", "REFL_10CM"]
```

The structural fields ride along automatically, so the list stays about
the science you want. See [What cannot be
dropped](#what-cannot-be-dropped).

### An explicit drop list

```toml
[output]
history_drop = ["QICE", "QSNOW", "QGRAUP"]
```

`history_vars` and `history_drop` are an include list and an exclude list
over the same inventory, and there is no rule for which would win, so
setting both is refused. `preset` and `history_vars` are two spellings of
the same include list and are likewise refused together. A preset **may**
be trimmed further with `history_drop`:

```toml
[output]
preset = "severe"
history_drop = ["QHAIL"]
```

## Per domain: keep the parent whole, trim the child

The same table sits inline on a `[[domain]]`, exactly as `[tiles]` does,
and it overrides the tree-wide one for that domain:

```toml
[output]
preset = "severe"           # the tree-wide default

[[domain]]
grid_id = 1
# ... no output table: this domain takes the tree-wide "severe"

[[domain]]
grid_id = 2
# ... the 1 km child, which writes the bytes
output = { preset = "minimal" }
```

A domain that speaks overrides the tree-wide table **entirely** rather
than merging key by key: "the preset from the tree, the drop list from
the domain" is a configuration nobody can read off the file.

## What cannot be dropped

Some fields are not merely important, they are what makes the file
readable at all: the horizontal georeference (`XLAT`, `XLONG`), the
terrain and the mass/vertical coordinate the eta levels are defined
against (`HGT`, `ZNU`, `ZNW`, `P_TOP`, `MUB`, `PHB`), the time
coordinate WRF tooling reads (`Times`, `XTIME`, `ITIMESTEP`), and the
perturbation potential temperature `T`. A wrfout missing one of those is
not a smaller wrfout; it is a file no reader in the estate can place a
value in. Naming one in `history_drop` is refused, and the refusal says
which set it belongs to and what to drop instead.

`T` is in that set for a measured reason rather than a stylistic one: a
tape written without it is rejected outright by `rw_wrfbatch`, *"Open
WRF ... failed and the file is not a supported post-processed WRF
archive"*: the whole file, not one product. It costs one 3-D volume;
see the table above.

The `windfarm` preset is the one exception, and it is deliberate: its
reader is `woof energy extract`, which reads the hub-height planes, not
a WRF post-processor, and the two structural volumes `T` and `PHB` would
be 99 planes on a 49-level leaf, as many as every plane the preset keeps
at six hub heights. A `windfarm` tape therefore does not open in
`rw_wrfbatch`, and the plan-time warning says so (0 render products).
Every other structural field stays forced.

`REFL_10CM` is deliberately **not** in that set. It is a large volume and
dropping it is a legitimate thing to want: you just lose the
reflectivity products, which is what the next section is about.

## What it costs you in pictures, said before the run

WOOF owns the render product catalog, so it can answer "which plots does
this selection cost me?" at **config resolution** rather than three hours
later in an empty render directory. One warning per domain, naming the
product and the variable:

```
warning: d02 of case.toml: [output] preset=minimal drops 132 variables,
  so these render products can no longer be drawn from this run's
  history: composite_reflectivity needs REFL_10CM; refl needs REFL_10CM.
  Keep the variable(s) if you want the product(s); otherwise this is the
  storage you asked for.
```

A short drop list is spelled out; a preset sheds more names than anyone
reads, so it is counted instead and the product-to-variable mapping --
the half you can act on -- stays in view.

It is a warning and not a refusal. Shedding a product deliberately is
exactly what the table is for.

## And again at the render front door

A trimmed run stamps what it did into every file it writes:

```
GPUWM_HISTORY_PRESET    = "full"
GPUWM_HISTORY_SELECTION = "preset=full history_drop=QCLOUD,QRAIN,REFL_10CM"
GPUWM_HISTORY_DROPPED   = "QCLOUD,QRAIN,REFL_10CM"
```

so `woof render` can name the cause instead of reporting an absence.
Asking for a product whose input this run dropped is refused at the front
door, before a run directory is claimed:

```
woof render: every product you asked for needs a variable this run did
not write.
  composite_reflectivity is drawn from REFL_10CM, which this run's
  [output] history_drop / preset did not write.
  remedy: re-run without dropping the variable(s) -- delete the name from
  [output] history_drop, or widen the preset (severe keeps the
  storm-scale volumes, full keeps everything).
  # to see what THIS file can still draw:
  #   woof render --list-products FILE
```

Ask for a mix and the survivors are still drawn; the casualties are named
in a `note:` line rather than quietly missing from the directory.
`--products all` is unaffected: it asks for whatever the file supports.

A full-default run stamps none of those attributes, so its header is
unchanged and nothing downstream can tell this feature exists.

## Checkpoints are a different stream

`restart_interval_s` writes `gpuwmrst_*.npz` **checkpoint** files from
model state; `[output]` selects what the **history** tape carries. They
are different files written by different code, and the selection is
deliberately kept out of the restart identity, so a run that filled its
disk can resume with a trimmed tape, and a trimmed run resumes a full
run's checkpoints. Nothing about the forecast changes: same trajectory,
same numbers, same fingerprint.

## How a history frame leaves the model

A history frame is written in the background. What the model waits for
depends on where the domain's numbers live. None of it changes a byte of
any file: every path below writes the same arrays, in the same order, through
the same writer.

**A resident domain** (no `[tiles]`). The frame's selected fields are
copied on the device, on the model's own stream, into buffers the writer
keeps. A side stream then copies those buffers to page-locked host memory,
and a writer thread writes the file. The model waits only for the device
copy. It used to wait for the whole copy over PCIe to the host. Before the
next frame reuses the buffers, the model waits for the previous host copy to
finish, which normally finished long before.

The buffers are extra device memory: one copy of the frame's selected
fields, held for the run. They are used only when the run's priced memory
plan (the resident admission `woof run`, `woof go` and the prepared runners
take before they build the domain) leaves that much room on the card. The
writers on a card share that room, and each writer is capped at 4 GiB.
Otherwise the frame is copied the old way, and the receipt says why:

| reason | when |
|---|---|
| `unpriced` | nothing priced this card (a harness, an unread card) |
| `outside-priced-plan` | the buffers do not fit what the plan leaves |
| `over-cap` | the frame is larger than 4 GiB |
| `device-allocation-failed` | the card refused the allocation |
| `disabled` | `WOOF_ASYNC_HISTORY=off` |

**A `[tiles]` host-store domain.** The frame is copied out of the host
store with one memcpy, then handed to the writer thread. The model goes on
sweeping. It used to wait between two sweeps for the whole netCDF write: a
5.24 GB frame took 81 s. The next frame of that domain waits for this one's
write to finish, because it refills the same copy. The copy is held only if
it fits host RAM beside the run. The reserve is the asynchronous
checkpoint's: the larger of 2 GiB and 5% of RAM. Otherwise the frame is
written from the store and drained before the next sweep, as before, with
the reason `insufficient-host-ram`.

**Queue depth.** Each domain's writer can hold `[output]
history_queue_depth` frames queued behind the one it is writing. The
default is 1. Each queued frame is one page-locked host copy, and a depth
whose copies would not fit host RAM is lowered.

**Locking.** The Rust writer (the default engine) takes the process-wide
netCDF lock only for the netCDF4 read-back that validates a finished tape.
Nested domains' tapes are therefore written at the same time. The netCDF4
engine (`WOOF_WRFOUT_WRITER=python`) still holds the lock for its whole
session.

**Output identity.** The Rust writer computes each tape's SHA-256 while it
writes the bytes, in file order. It is the same digest a re-read gives, and
the run's frame records carry it. The writer no longer reads every finished
file back to hash it. It still checks that the file it renames is the file
it hashed: same file, same length, and the address names it.
`WOOF_OUTPUT_IDENTITY_VERIFY=1` also re-reads every tape and refuses one
whose read-back digest differs. A tape written out of file order, a library
built before this change, and the netCDF4 engine all fall back to the
re-read, and the receipt names why.

**Hex.** `woof hex forecast` captures each frame and runs its physical
gate on the model thread, as before, so a refusal lands on the same step.
The netCDF write, its zlib compression, the stamps and the file hash then
run on a background thread. `WOOF_ASYNC_HISTORY=off` writes them inline
again.

The run receipt's `history_writes` block records, per domain:

* the copy mode and its reason;
* the requested and effective queue depth;
* the engine, and what the netCDF lock covers;
* the identity mode counts;
* each frame's model-loop cost: host seconds inside the submit, and
  producer-stream seconds between CUDA events around it;
* the `[tiles]` frame mode.

`WOOF_ASYNC_HISTORY` (`auto`, the default; `off`; `on`) switches all of it
off. `on` also allows the device buffers when no plan was priced, using
the card's live free memory; it is meant for benchmarks.

## Demo

A two-domain forecast that keeps the storm volumes on the 3 km parent and
writes surface-only on the 1 km child:

```toml
# case.toml -- the [output] parts only; the rest is an ordinary experiment
[output]
preset = "severe"

[[domain]]
grid_id = 1
# ... 3 km parent: takes the tree-wide "severe"

[[domain]]
grid_id = 2
# ... 1 km child
output = { preset = "minimal" }
```

```console
$ woof prep --source SOURCE ... --experiment-config case.toml \
      --output-root prepared
warning: d02 of case.toml: [output] preset=minimal drops ... so these
  render products can no longer be drawn from this run's history:
  composite_reflectivity needs REFL_10CM; refl needs REFL_10CM. ...

$ woof sim prepared --experiment-config case.toml --outdir out

$ woof render --engine rust --products 2m_temperature \
      --out png out/run-*/wrfout/wrfout_d02_*
render: 3 file(s) -> png

$ woof render --engine rust --products composite_reflectivity \
      --out png out/run-*/wrfout/wrfout_d02_*
woof render: every product you asked for needs a variable this run did
not write. ...
$ echo $?
2
```

To see what a given file can still draw:

```console
$ woof render --list-products out/run-*/wrfout/wrfout_d02_0*
```

## Where the vocabulary comes from

The names `[output]` accepts are the union of WOOF's wrfout schema
tables, the Registry-transcribed physics history rows
(`woof/io/wrf_output_schema.py`) plus the writer's own metadata table,
so a physics scheme that adds a history field joins the vocabulary with
its schema row and nothing here changes. An unknown name is refused with
the whole valid set printed, because there is no other place to read it:

```
[output] history_drop of case.toml names 'QCLOUDD', which is not a wrfout
history variable.  Did you mean QCLOUD?  The valid set is: ACRUNOFF,
ALBOLD, ... ZTOP_PLUME.
```

## Surface solar fields

Beside `SWDOWN` (global horizontal irradiance), a run with radiation on
publishes WRF's surface solar fields under their WRF names, exactly when
the active scheme computes them:

| variable | units | written when |
|---|---|---|
| `SWDDNI` | W m-2 | the shortwave scheme computes a direct/diffuse split from its own solve: `ra_sw_physics = 4` (RTE+RRTMGP, the default, or `ra_rrtmg_variant = "rrtmg_legacy"`). Direct-normal irradiance, WRF's `SWDDIR / COSZEN` on daylit columns, zero at night. |
| `SWDDIF` | W m-2 | same condition. Diffuse horizontal irradiance at the surface. |
| `COSZEN` | dimensionless | any radiation scheme is on. The cosine of the solar zenith angle at the radiation call (WRF's interval-midpoint sun), held between calls as WRF holds it. |

On a daylit column `SWDDNI * COSZEN + SWDDIF` is the scheme's surface
downward shortwave, which is `SWDOWN`. All three are held between
radiation calls, like `SWDOWN`; under `swint_opt = 1` the
per-step interpolated `SWDDNI`/`SWDDIF` are written instead.

Dudhia shortwave (`ra_sw_physics = 1`) and the analytic proxy compute no
direct/diffuse split. WRF fills one for Dudhia with the radiation
driver's empirical Ruiz-Arias model; WOOF does not publish an empirical
split as model output, so those runs carry `COSZEN` but neither
`SWDDNI` nor `SWDDIF`. WRF flags `SWDDNI`/`SWDDIF` restart-only, so a
stock WRF wrfout carries them only through an `iofields_filename` add;
WOOF writes them with WRF's own schema (name, units, description,
stagger). They make the default full inventory larger than releases
before this one: `COSZEN` on every run with radiation on, and
`SWDDNI`/`SWDDIF` as well on a run with `ra_sw_physics = 4`.

## Hub-height wind fields

Off by default. `[output] hub_wind = true` (the default heights 40, 80,
100, 120, 150 and 200 m above ground) or `hub_heights_m = [...]` turns on
an in-model diagnostic computed after every model step, with one
`hub_height` axis:

| variable | dimensions | meaning |
|---|---|---|
| `HUB_HEIGHT` | `Time, hub_height` | the configured heights, m above ground |
| `HUB_WINDOW_S` | `Time` | model seconds the MAX/MIN/MEAN window covers |
| `WSPD_HUB` | `Time, hub_height, south_north, west_east` | wind speed at the frame's valid time, m s-1 |
| `WDIR_HUB` | same | earth-relative direction the wind blows from, degrees |
| `WSPD_HUB_MAX` | same | maximum speed since the previous history frame |
| `WSPD_HUB_MIN` | same | minimum speed since the previous history frame |
| `WSPD_HUB_MEAN` | same | time-weighted mean speed since the previous history frame |

The window restarts after each history write, so a ramp within a
15-minute file shows up as `WSPD_HUB_MAX - WSPD_HUB_MIN`. `hub_sites`
also samples each site every `hub_sample_interval_s` (60 s by default)
into a `hubwind_d0N_<time>.nc` sidecar per history write. The `energy`,
`minimal`, `severe` and `windfarm` presets keep these fields.

The same diagnostic also writes the hub-height **scalars** when the
selection names them: the `windfarm` preset does, and so can
`history_vars`. `full` does not switch them on, because a full tape
already carries the volumes they are interpolated from. They are
instantaneous at the frame's valid time, computed when the frame is
written (after the step's microphysics), on the same `hub_height` axis:

| variable | units | meaning |
|---|---|---|
| `T_HUB` | K | air temperature: theta and pressure interpolated linear in z, then `theta (p / 1e5)^(R_d/c_p)` |
| `P_HUB` | Pa | full pressure |
| `RHO_HUB` | kg m-3 | moist air density `p / (R_d T (1 + 0.608 q))`, `q = qv / (1 + qv)` |
| `QV_HUB`, `QC_HUB`, `QR_HUB` | kg kg-1 | water vapour, cloud water and rain mixing ratios (moist runs) |
| `QNC_HUB` | kg-1 | cloud droplet number, where the microphysics carries it (the array `QNCLOUD` is written from) | The arithmetic, the
restart behaviour and what is not supported are in
[docs/energy-hindcast/hub-wind.md](../energy-hindcast/hub-wind.md).

## Wind maxima and gusts

With `nwp_diagnostics = 1` (`&time_control`, or the RunConfig key of the
same name) every domain folds six running maxima on every model step and
restarts them after each history write, so each frame carries the
extreme over the interval since the previous frame:

| variable | units | meaning |
|---|---|---|
| `WSPD10MAX` | m s-1 | maximum 10 m wind speed in the interval (WRF `nwp_diagnostics`). |
| `U10MAX` | m s-1 | 10 m U component at the instant of `WSPD10MAX`. |
| `V10MAX` | m s-1 | 10 m V component at the instant of `WSPD10MAX`. |
| `GUST` | m s-1 | maximum surface wind gust in the interval, NCEP UPP `CALGUST` method (PBL momentum mixed down). |
| `W_UP_MAX` | m s-1 | strongest updraft below 400 hPa in the interval (WRF `nwp_diagnostics`). |
| `W_DN_MAX` | m s-1 | strongest downdraft below 400 hPa in the interval (negative). |

They ride beside `UP_HELI_MAX`, are carried through checkpoints, and
are selected by the `minimal`, `severe` and `energy` presets (a preset selects them; it does
not switch the diagnostic on). The method, its sources and where WOOF
differs from WRF are in
[docs/energy-hindcast/gusts.md](../energy-hindcast/gusts.md).
