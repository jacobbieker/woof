# Real-data regional throughput: woof vs Breeze/NumericalEarth on ERA5 boundaries

**Question.** Breeze.jl, driven through NumericalEarth.jl's `nested_atmosphere_model`, publishes
two real-data limited-area runs with ERA5 boundaries. If woof runs the same domains with the
same boundary data, how fast is it, and what does the whole workflow cost (commands,
preparation time, data size)?

**Short answer.**

- **woof MC3E (measured).** On the 150 x 136 x 50 domain, woof ran at **345 simulated hours per
  wall hour** (6.3 M cell-steps/s). That was on a shared RTX A6000 that other jobs held at 100 %
  utilisation, under a 6 GiB device-memory cap.
- **Breeze MC3E, same card (measured).** **17.5x** real time (4.2 M cell-steps/s). This is
  NumericalEarth's own example, run on the same card under the same cap. The older published
  figure on an A100 is 8.3x.
- **AR, 540 x 315 x 50, 72 h.**
  - woof fetched and prepared the full domain. Its 72 h forecast needs about 9.5 GiB, so on the
    capped card it was run reduced to 180 x 105 (1,098x real time).
  - From the two measured per-cell costs, the full woof run would take about **41-122x** real
    time. That is an estimate.
  - Breeze's published figure is 19x on one A100-40GB, which scales by bandwidth to about
    9-10x on an A6000, also an estimate.
  - `run_timing.sh` measures both cases full-size on a clean card.
- **Why woof is faster.** It takes about 56 s per step where Breeze takes about 4 s. Each
  woof step is also cheaper per cell on this card (0.163 s against 0.241 s for the same
  1.02 M cells).
- **What this does not show.** Nothing here says which model is more accurate. The physics
  suites differ (see below), so accuracy is confounded by design. The ASOS table for the woof
  run is included as a sanity check, not as a comparison.

Every number below is labelled **measured** (on this machine, this branch), **documented**
(stated by a Breeze/NumericalEarth source, cited), or **estimate** (a documented number scaled,
with the scaling stated, or a measured number scaled to another size). All measured numbers
come from a shared card; see
[Caveats](#caveats-shared-card-memory-cap-and-what-is-not-matched).

The harness is in [`tools/compare/era5_nest/`](../../../tools/compare/era5_nest/README.md).
`run_timing.sh` there reruns everything full-size on a clean card.

## The two published cases

Both cases run on Breeze's terrain-following `LatitudeLongitudeGrid` at 1/9 degree, with 50
levels to about 20 km and hourly ERA5 pressure-level boundaries:

| | MC3E squall line | Atmospheric river |
|---|---|---|
| Source | [NumericalEarth.jl `examples/breeze_downscaling_era5.jl`](https://github.com/NumericalEarth/NumericalEarth.jl/blob/66112eac/examples/breeze_downscaling_era5.jl) | [glwagner/AtmosphericRivers.jl](https://github.com/glwagner/AtmosphericRivers.jl/blob/7cc5735d/README.md) |
| Box | 105.835-89.135 W, 29.055-44.155 N, centred on ARM SGP (lines 69-80) | 170-110 W, 25-60 N (README:53, case.jl:22) |
| Grid | 150 x 136 x 50 = 1.02 M cells | 540 x 315 x 50 = 8.5 M cells (README:56) |
| Spacing | dy 12.35 km; dx 10.8 km (south edge) to 8.9 km (north edge) | dy 12.35 km; dx 11.2 km (south edge) to **6.2 km** (north edge) |
| Window | 2011-05-20 00Z + 3 h (lines 64-65) | 2025-12-07 12Z + 72 h (README:57) |
| Time step | adaptive, CFL 0.5, at most 10 s (lines 214-219) | same; 47,600 iterations in 72 h, so 5.45 s mean (derived) |
| Lateral boundaries | open BCs + Davies relaxation, 5 cells, rate 1/300 s^-1 | 8 cells, 240 km terrain blend (the "production recipe", README:63) |
| Physics | 1-moment mixed-phase microphysics, all-sky RRTMGP every hour, `SlabLand`, no PBL, no cumulus, numerical diffusion only (lines 29-33) | the same nest defaults, with a static-SST `PrescribedOcean` |
| Published speed | "The simulated hour cost 7.2 minutes wall" on an A100 ([PR 454 comment](https://github.com/NumericalEarth/NumericalEarth.jl/pull/454#issuecomment-5012436787)) | "72 h ... in 3.77 h of stepping on one A100-40GB (47,600 iterations, ~19x real time)" (README:62-64); 0.267 s/iter on 1 GPU, 0.178 s/iter on 2, without output (README:77-78) |

Both models build the vertical column the same way: `ReferenceToStretchedDiscretization` with a
60 m surface cell, x1.15 stretching and 490 m at most. Using Oceananigans' own algorithm, the 50
interfaces reach **20,002.99 m** (`breeze_cases.breeze_z_interfaces`, unit-tested against the
documented first faces).

The nest's initial balancing is an adiabatic, FV3 `na_init`-style pass
(`ext/NumericalEarthBreezeExt/breeze_nested_atmosphere.jl:421`). It is not a digital filter: the
digital-filter PR is still open.

The AtmosphericRivers README states that only the interior more than **25 cells from the frame**
"stays physical for all 72 hours" (README:64-65). At 12 km that is a 300 km rind the analysis
masks out. On 540 x 315 the usable interior is 490 x 265, about 76 % of the cells. woof's 4-cell
relaxation zone plus 1-cell specified zone has no such published rind.

Open Breeze issues relevant to real-data LBCs:

- [Breeze.jl #825](https://github.com/NumericalEarth/Breeze.jl/issues/825) (open): spurious
  vertical velocity at the lateral boundaries. It comes from the per-substep boundary density
  relaxation that PR #747 added.
- [#793](https://github.com/NumericalEarth/Breeze.jl/issues/793) (open): a cold-start blow-up
  with divergence damping. This is why the nest runs `NoDivergenceDamping`.
- [#897](https://github.com/NumericalEarth/Breeze.jl/issues/897) (closed): implicit vertical
  advection NaNs on the AR nest.

## How woof reproduces them

`make_nest_configs.py` runs the domain wizard and then pins what the comparison needs:

```bash
python tools/compare/era5_nest/make_nest_configs.py --case mc3e --out-dir RUN --hours 3 --vram-gib 6 --no-restarts
#  -> woof domain --point=36.6050,-97.4850 --projection lambert --root-dx 12 \
#       --eta-levels <Breeze's 51 faces as eta> --p-top-pa 5472 --source era5 \
#       --era5-provider arco --cadence 1 --cycle 2011-05-20T00 --hours 3 \
#       --physics-profile thompson-mp8-ysu-mm5-noah-rte-rrtmgp-v1 ...
#    then sets nx/ny = 150/136 (e_we/e_sn 151/137), recomputes the fetch area for that
#    grid with the wizard's own _fetch_area, sets radt = 60 min and restarts off.
```

What matches the Breeze runs:

- the cell counts `Nx x Ny x Nz` exactly;
- the centre;
- the nominal 12 km spacing;
- Breeze's own 50 interface heights, mapped to eta through a US Standard Atmosphere column
  with p_top = p(20,003 m) = 5472 Pa;
- hourly ERA5 boundaries;
- hourly radiation;
- no cumulus scheme.

What does not match, and why it matters:

- **Grid.** woof's regional grids are map projections, so it uses Lambert conformal at a
  uniform 12 km. Breeze's lat-lon grid at the same counts covers less ground:
  - MC3E: woof covers 1800 x 1632 km, 1.18x Breeze's area.
  - AR: woof covers 6480 x 3780 km, 1.30x Breeze's area.

  The AR footprint is also wider east-west, so its ERA5 crop crosses the antimeridian
  (`18.45,168.87,61.50,-88.87`). woof fetched it without complaint.
- **Time step.** Each code takes the step its own scheme allows. woof uses WRF's RK3 with
  acoustic substeps and an adaptive dt of about 56 s at 12 km. Breeze is capped at 10 s and
  CFL 0.5 by its wizard, and its smallest zonal spacing is 6.2 km on the AR grid. That is the
  largest single reason for the gap in simulated time per wall time. cell-steps/s divides it
  out.
- **Physics.** Breeze runs the nest defaults. woof runs the verified suite closest to them,
  `thompson-mp8-ysu-mm5-noah-rte-rrtmgp-v1`, which does more work per column:

  | | Breeze nest | woof suite |
  |---|---|---|
  | microphysics | 1-moment mixed-phase (CloudMicrophysics) | Thompson (mp 8): 2-moment rain and ice |
  | radiation | all-sky RRTMGP, hourly | RTE+RRTMGP lw+sw, hourly (radt 60) |
  | surface | `SlabLand` bucket + Monin-Obukhov (MC3E); static SST (AR) | Noah 4-layer LSM + revised MM5 surface layer |
  | boundary layer | none; numerical diffusion only | YSU |
  | cumulus | none | none |
  | dynamics | compressible, split-explicit, WENO5 momentum, no divergence damping | ARW compressible, split-explicit RK3, 5th/3rd-order advection, 6th-order diffusion, w-Rayleigh damping |

  None of woof's suites drops the PBL at 12 km. The pbl-off suites exist for LES spacings.
- **Boundaries.** woof uses WRF's specified + relaxation zone (spec_zone 1, relax_zone 4, width
  5) on ERA5 interpolated through its WPS-equivalent preprocessing. Breeze uses open BCs plus
  Davies relaxation at 1/300 s^-1 over 5 cells (MC3E) or 8 cells (AR).

The ERA5 route is one of woof's 24 runnable driving models in `woof/source_adapters.py`, and one
of the 3 there certified against stock WRF (`woof/source_adapters.py:1798-1814`, the `era5`
row). It reads Google's keyless ARCO Zarr store or, with an account, the Copernicus CDS.

## Results

### MC3E, 150 x 136 x 50, 3 h

The woof run is measured. The 3 h woof run's wall time breaks down as follows (measured, from
the run's `events.jsonl` stage records):

| stage | wall | note |
|---|---|---|
| `woof fetch` ERA5, ARCO, 4 hourly times | 178 s | 77.7 MB on disk; ARCO serves global chunks per variable per hour, so the cost is per hour, not per area |
| prepare (terrain clock, schedule, static geography, ERA5 to WRF real) | 92.4 s | preprocessing fell back to the **CPU backend** because the card was 100 % busy |
| initialise writers | 1.6 s | |
| forecast, 189 steps, 3 h | 45.3 s | step 1 took 11.9 s including the GPU kernel compile |
| render (67 products per frame) | 20.9 s | |
| station verification (`woof verify-visuals`) | 98 s | ASOS from Dynamical.org's IEM parquet mirror |
| peak device memory, process tree | 3,080 MiB | `nvidia-smi` per-process sampling; the wizard estimated a 2.96 GiB envelope |

Throughput (`nest_throughput.py`):

| run | kind | simulated h / wall h | cell-steps/s | s / step | mean dt (s) |
|---|---|---|---|---|---|
| woof, RTX A6000 (shared, 6 GiB cap), steps 10-189 | measured | **344.9** | **6.27e+06** | 0.163 | 56.1 |
| woof, same run, whole forecast stage incl. compile | measured | 238.7 | -- | -- | 57.1 |
| Breeze 0.12 / NumericalEarth 66112eac, RTX A6000 (shared, 6 GiB cap), iterations 30-829 | measured | **17.5** | **4.23e+06** | 0.241 | 4.2 |
| Breeze, A100 (PR 454, with output, Breeze 0.9) | documented | 8.3 | -- | -- | -- |
| Breeze PR 454 scaled to an RTX A6000 by bandwidth, if the A100 was 40 GB (768/1555) | estimate | 4.1 | -- | -- | -- |
| the same, if it was 80 GB (768/2039) | estimate | 3.1 | -- | -- | -- |

The Breeze measurement (`breeze_mc3e_timing.jl`, 1 simulated hour, written to
`timing-1h-cap6.json`) gives the following figures:

- **Run shape.** 829 iterations. The adaptive step ran between 2.8 and 4.7 s, never reaching
  its 10 s cap.
- **Setup.** 903 s from launch to the first step: JIT compilation, ERA5 regridding, and the
  balancing pass.
- **First iterations.** `run!` took 459 s in all. Its first 200 iterations took 256 s,
  against about 48 s for each later 200, so roughly 210 s of that is first-call JIT.
- **Memory.** Peak sampled device memory was 2,122 MiB, against woof's 3,080 MiB for the same
  cell count with its heavier physics.
- **Sanity.** The fields stayed finite. max|w| reached 21 m/s at 58 min. Without fields on
  disk it cannot be said whether that is the squall line or the boundary w of
  [Breeze #825](https://github.com/NumericalEarth/Breeze.jl/issues/825).

The PR 454 figure (8.3x on an A100, older Breeze) is slower than this A6000 measurement
(17.5x). Breeze has evidently become faster since then, so the published number understates
current Breeze. The measured row is the one to compare.

**Workflow, side by side (MC3E):**

| step | woof | Breeze/NumericalEarth |
|---|---|---|
| install | `pip install` + `woof setup --with-geog` (WPS_GEOG, 28 GB here) | Julia 1.12 + `Pkg.instantiate` (3.6 GB depot, 228 s precompile, measured) |
| credentials | none (ARCO); CDS optional | CDS account + licences for ERA5 **and** `satellite-albedo` (CGLS) |
| data | 77.7 MB ERA5 (ARCO, 178 s) | 28 MB ERA5 (CDS) + 456 MB ETOPO2022 + CGLS albedo; prefetch 765 s on CPU including JIT (measured) |
| time to first step | prepare 92 s + initialise 2 s, then 11.9 s for step 1 (kernel compile) | 903 s setup + ~210 s of first-call JIT in the first iterations (measured, GPU) |
| configure | `woof domain ...` (one command) | edit the example script |
| run | `woof go CONFIG` (prepare, forecast, render, verify) | `julia --project example.jl` |
| products | wrfout, 67 rendered products per hour, ASOS/MRMS verification | JLD2 slices; plots in the example |

The CDS account on this machine had not accepted the CGLS `satellite-albedo` licence, so the
NumericalEarth example failed at its albedo download (HTTP 403, "required licences not
accepted"). The licence was not accepted on the user's behalf. Instead, the Breeze timing ran
with a constant 0.2 surface albedo (`BREEZE_ALBEDO=0.2`). That changes the surface energy
budget, but not the radiation solver's cost.

**Station verification (woof, MC3E, measured; ASOS via Dynamical.org, forecast minus
observation):**

| valid (UTC) | quantity | stations | bias (F-O) | RMSE |
|---|---|---|---|---|
| 2011-05-20 00:00Z | temperature_2m | 591 | +0.13 K | 1.78 K |
| 2011-05-20 00:00Z | dewpoint_2m | 591 | -0.18 K | 2.14 K |
| 2011-05-20 00:00Z | wind_speed_10m | 579 | -0.74 m s-1 | 2.04 m s-1 |
| 2011-05-20 01:00Z | temperature_2m | 595 | -0.20 K | 1.80 K |
| 2011-05-20 01:00Z | dewpoint_2m | 595 | +0.19 K | 2.02 K |
| 2011-05-20 01:00Z | wind_speed_10m | 575 | -0.25 m s-1 | 1.96 m s-1 |
| 2011-05-20 02:00Z | temperature_2m | 592 | -0.78 K | 2.08 K |
| 2011-05-20 02:00Z | dewpoint_2m | 592 | -0.10 K | 2.09 K |
| 2011-05-20 02:00Z | wind_speed_10m | 564 | -0.44 m s-1 | 2.18 m s-1 |
| 2011-05-20 03:00Z | temperature_2m | 590 | -0.82 K | 2.15 K |
| 2011-05-20 03:00Z | dewpoint_2m | 589 | -0.16 K | 2.04 K |
| 2011-05-20 03:00Z | wind_speed_10m | 558 | -0.01 m s-1 | 1.89 m s-1 |

How to read this table:

- **What it says.** The run is sane, and it is the same ERA5 analysis a few hours on. Hour 0
  is ERA5 interpolated to the grid.
- **What it does not say.** It ranks nothing against Breeze. Breeze has no station verifier,
  and the two physics suites differ.
- **Squall line.** Fields at 03Z are finite throughout. They include a developed squall line:
  REFL_10CM up to 61.7 dBZ, RAINNC up to 54.8 mm, W up to 9.1 m/s.
- **Arms not scored.** MRMS and the HRRR reference arm do not exist for 2011, so those arms
  stay pending.

### Atmospheric river, 540 x 315 x 50, 72 h

| | value | kind |
|---|---|---|
| ERA5 ARCO fetch, 4 hourly times, antimeridian-crossing crop | 222 s, 640 MB | measured |
| full 72 h fetch at hourly cadence (73 times) | ~68 min, ~11.7 GB | estimate (linear in hours) |
| `woof static` (geography for the 540 x 315 grid, CPU) | 218 s | measured |
| `woof ingest` (ERA5 to the full 540 x 315 x 50 grid, 4 boundary times; CPU preprocessing backend because the card was busy, device state under the 6 GiB cap) | 283 s, 269 MB initialised state | measured |
| forecast memory, woof's preflight | 9.46 GiB envelope | estimate (`woof check`); Breeze README: "~10 GB on GPU" |
| reduced-domain correctness run, 180 x 105 x 50 (1/3 per axis, same centre and spacing; ERA5 reused from the full fetch), 3 h, `woof go --products none` | prepare 58 s, forecast 13.5 s for 116 steps; 2,942 MiB peak | measured |

The reduced run's fields at 15Z are finite. T2 ranges 281-292 K (all ocean), the jet core
reaches 57 m/s, max W is 0.18 m/s, and RAINNC reaches 1.7 mm. Its steady window, steps 10-116:

| run | kind | simulated h / wall h | cell-steps/s | s / step | mean dt (s) |
|---|---|---|---|---|---|
| woof AR 180 x 105 x 50, RTX A6000 (shared, 6 GiB cap) | measured | 1097.8 | 1.09e+07 | 0.086 | 94.8 |

Over open ocean the adaptive clock ran near woof's 8 s/km ceiling: 95 s against 56 s over
MC3E's terrain and convection. Per cell-step this run cost 9.1e-8 s, and MC3E cost 1.6e-7 s on
the same contended card.

The full AR forecast does not fit the 6 GiB cap the shared card allows. Its 72 h timing is in
`run_timing.sh` for the clean-card phase. Until it runs, the woof figure is an estimate built
from the two measured runs:

- **Seconds per step.** The per-cell-step cost (9.1e-8 to 1.6e-7 s) times 8.5 M cells gives
  0.78-1.36 s per step.
- **Step count.** The mean dt (56-95 s) gives 2,700-4,600 steps for 72 h.
- **Result.** **0.6-1.7 h of stepping for 72 h, 41-122x real time.** Both inputs were
  measured on a contended card.

| run | kind | simulated h / wall h | cell-steps/s | s / step | mean dt (s) |
|---|---|---|---|---|---|
| Breeze, 1x A100-40GB, 72 h with output | documented | 19.1 | 2.98e+07 | 0.285 | 5.4 |
| Breeze, 1x A100-40GB, no output | documented | 20.4 | 3.19e+07 | 0.267 | 5.4 (borrowed) |
| Breeze, 2x A100-40GB, no output | documented | 30.6 | 4.78e+07 | 0.178 | 5.4 (borrowed) |
| Breeze, 1x A100-40GB scaled to an RTX A6000 by bandwidth | estimate | 9.4 | 1.47e+07 | 0.577 | 5.4 |
| woof 540 x 315 x 50, RTX A6000, from the two measured per-cell costs | estimate | 41-122 | 6.3e+06-1.09e+07 | 0.78-1.36 | 56-95 |

Reading the two kinds of number together:

- **Per step.** Breeze's documented AR rate on an A100 (2.98e7 cell-steps/s) is above woof's
  measured rates on a contended A6000 (6.3e6-1.09e7). Its bandwidth-scaled A6000 estimate
  (1.47e7) is too. At MC3E size on the same card the measured order is the other way round:
  woof 6.3e6, Breeze 4.2e6. Breeze's documented AR rate is 7x its own measured MC3E rate here,
  which points to card occupancy (8.5 M cells against 1 M) as well as the card.
- **Per simulated hour.** woof takes 10-17x longer steps, so in simulated time woof leads on
  both cases: estimated 41-122x against Breeze's documented 19x on an A100.
- **Two caveats.**
  - woof's cell-step carries a PBL, a land model and 2-moment microphysics.
  - Neither model's number is from a clean A6000 yet. `run_timing.sh` produces those.

## Is either faster, more accurate, more efficient?

- **Faster (simulated time per wall time).** woof, by about **20x** on the same card: 345x
  against 17.5x on MC3E. The time step is most of it.
  - Breeze's outer step is capped at 10 s and CFL 0.5. It averaged 4.2 s on MC3E here and
    5.45 s for the published AR run.
  - woof's WRF-lineage split-explicit step averaged 56 s on MC3E and 95 s on the reduced AR
    at 12 km.
  - Breeze's adaptive implicit vertical advection (#897, closed) targets the vertical part of
    that limit. Whether it lengthens the step on these grids is not measured here.
- **More efficient per cell-step.** On MC3E on the same card, woof: 6.3 M cell-steps/s
  against Breeze's 4.2 M, about 1.5x. That holds even though woof's step also runs a PBL, a
  land model and 2-moment microphysics.
  - Breeze's documented AR rate on an A100 (2.98e7 cell-steps/s) is higher than both. It is
    on a larger grid that fills the card better, with a bandwidth-scaled A6000 estimate of
    1.47e7.
  - So per-cell efficiency at AR size is open until `run_timing.sh` runs both codes full-size
    on a clean card.
- **Memory.** Breeze is leaner per cell: 2.1 GiB against 3.0 GiB at 1.02 M cells.
- **More accurate.** Not answerable from these runs. The physics differ in every family except
  radiation and the absence of cumulus. Breeze has no PBL or land model, and woof has both. Any
  skill difference would measure the suites, not the dynamical cores. Breeze's own README also
  masks a 25-cell rind as unphysical.
- **Boundary data from external sources.**
  - Both take ERA5.
  - woof takes it without an account, from the ARCO store. It validates and fingerprints it,
    crops it across the antimeridian, and also runs from 23 other driving models.
  - The NumericalEarth nest needs a CDS account plus a second licence (CGLS albedo) and ETOPO
    for its terrain. It supports ERA5 pressure levels as its parent dataset.

## Caveats: shared card, memory cap, and what is not matched

- **Contended card.** Every measured GPU number here ran on an RTX A6000 (48 GB, 768 GB/s)
  shared with other forecasts. `nvidia-smi` showed 100 % utilisation and 33.5 GB used by other
  processes before these runs. They also ran under a 6 GiB device-memory cap (`CUPY_GPU_MEMORY_LIMIT`
  / `JULIA_CUDA_HARD_MEMORY_LIMIT`). Read them as lower bounds on both codes' speed.
- **Bandwidth scaling.** It is a first-order estimate. It assumes both codes are
  memory-bandwidth bound and ignores clocks, cache and launch latency. The A100-40GB has
  1555 GB/s and the A6000 768 GB/s.
- **Breeze measurement provenance.** The Breeze measurement uses current NumericalEarth main
  (66112eac, v0.8.2) with Breeze 0.12.0. The AR publication used an unpublished NumericalEarth
  branch (v0.6.1) with Breeze 0.8.0, and the MC3E timing used registered Breeze 0.9.0.
- **Grid geometry.** Matching cell counts on a Lambert grid is not the same ground or the same
  smallest spacing; see above.

## Reproduce

See [`tools/compare/era5_nest/README.md`](../../../tools/compare/era5_nest/README.md). In short:

```bash
python tools/compare/era5_nest/make_nest_configs.py --case mc3e --out-dir RUN --hours 3 --vram-gib 6 --no-restarts
(cd RUN && woof fetch --source era5 --era5-provider arco --retrieve --cycle 2011-05-20T00 \
    --hours 3 --cadence 1 --area 26.78,-110.77,46.04,-84.20 --out data/mc3e_breeze_match)
CUPY_GPU_MEMORY_LIMIT=6442450944 tools/compare/era5_nest/woof_run.sh RUN/mc3e_breeze_match.toml RUN/go
python tools/compare/era5_nest/nest_throughput.py woof RUN/go --case mc3e
python tools/compare/era5_nest/station_table.py RUN/go/run-*
tools/compare/era5_nest/run_timing.sh        # clean card: MC3E 3 h, AR 72 h, Breeze MC3E 3 h
```
