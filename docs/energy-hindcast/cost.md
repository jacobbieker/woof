# Hindcast cost estimate

`woof energy estimate` prices a wind-farm hindcast before any GPU is rented.
It reports GPU-hours, billed instance-hours, wall clock for a number of
instances, and the on-demand and spot dollar cost for each cloud instance in
a dated price table. It also reports the size and monthly cost of the
Icechunk store the run would fill. The JSON document
(`woof-energy.cost-estimate.v1`) goes to stdout and a table goes to stderr.

**Every number it prints is an estimate.** The throughput model is fitted on
two timing receipts from consumer cards in this repository and scaled to
other cards by memory bandwidth. No datacentre card has been timed by this
tree. Each figure comes with a band (0.5x to 1.5x of the fitted throughput,
widened by an assumed utilisation curve, see [the model](#the-model)), and
the JSON lists the receipts and every assumption, so you can see what a
number rests on.

> **Command names.** The estimator is `woof energy estimate` and, with the
> same parser, `woof energy hindcast estimate`. `woof energy hindcast
> optimize` searches leaf spacing, levels, segment layout and offers for the
> cheapest hindcast that meets a deadline ([below](#cheapest-configuration)).

## Three ways to say what to price

| mode | flags | what is priced |
|---|---|---|
| plan | `--plan PLAN.json` | the domains of a `woof-energy.plan.v1` (`wrf-tiles` or `wrf-nests`), one segment each, repeated `--segments` times |
| partition | `--partition PARTITION.json` | each row of the document's `clusters` list, synthesised into parent and leaf tiles the way the `wrf-tiles` planner shapes them |
| summary | `--clusters N --footprint-km2 A` | `N` equal clusters sharing `A` km² of farm footprint, for when no partition exists yet |

`--job JOB.json` (a `woof-energy.hindcast-job.v1`-like document) supplies
the segment count, segment hours and spin-up when they are present. Flags
override it. With neither, a campaign is the hindcast driver's default:
9 h runs from each 00/06/12/18 UTC init with 3 h of spin-up, so 6 h of each
run is stored, and enough runs to store one year (1,460). Without
`--segments` the count is always one year of stored hours
(`ceil(8760 / (segment hours - spin-up))`, noted in `notes`). As in the
driver, a `--segment-hours` other than 9 with no spin-up given is refused
when the default 3 h would leave a window off the 6-hourly cycle. Synthesised
leaves default to 300 m with 49 levels (`--leaf-dx-m`, `--nz`), the
hindcast's own defaults.

Partition and job documents are read by duck typing, so a sibling's schema
can change without breaking the estimator:

- **cluster id**: `cluster_id` or `id`
- **area**: `leaf_area_km2`, `footprint_km2` or `area_km2` is farm footprint,
  multiplied by `--cover-factor`. A `region` (a `[W,S,E,N]` bbox, inline
  GeoJSON, or a path to a `.geojson`) is used as the leaf cover as drawn.
- **chain and tiles**: `leaf_dx_m`/`dx_m`, `chain_m`/`chain` (a list, or
  `"1000->200"`), and `est_tile_count`/`tile_count` (a floor on the number
  of leaf tiles)
- **job**: `segments` as a list or a count, `segment_hours`, `spinup_hours`,
  and `planner_args` (a mapping or an argv list) for `dx_m` and `nz`

A cluster with no area is refused, and the refusal names the keys it looked
for.

## The model

For each domain and segment, on a card with memory bandwidth `B` (GB/s) and
`S` streaming multiprocessors:

- `steps = ceil(run_seconds / dt)` and `cells = nx * ny * nz`.
- `throughput = K * B * cells / (cells + S * h)` cell-steps per second.
  `K` is the fitted constant. The second factor is an occupancy knee: a
  grid with few cells per SM cannot fill the card (`h` = 4,096 cells per
  SM reach half of it, central).
- `compute = steps * (n_sound + 4) / (4 + 4) * (cells / throughput + L)`.
  The receipts ran WRF's 4 acoustic substeps, and one large step is assumed
  to cost as much as 4 substeps. `L` (1 ms per step, central) is a fixed
  per-step cost, kernel launches and their tails, that no bandwidth
  removes.
- `output` and `checkpoint` are the bytes of each history frame and each
  hourly restart (60 3-D fields, assumed), moved at 10 GB/s device to host.
  That is the staged-copy rate measured in `tilestream/multigpu.py`. A leaf
  writes the `--store-preset` fields; parents and intermediates write the
  full tape their children are built from.
- `startup` is `14 s + 5.1 s per million cells`, a straight-line fit
  through the launch-to-first-step receipt in
  [docs/manual/08-operational-envelope.md](../manual/08-operational-envelope.md)
  section 8.3. It is charged to the GPU because the card is rented while
  preparation runs.

For each offer (provider and instance):

- **GPU-hours** are counted on that offer's card.
- **Instance-hours** = GPU-hours / (GPUs per instance × `--packing-efficiency`,
  default 0.85). The packing efficiency is the fraction of paid card time
  that a job holds the card at all. The rest is spent waiting on a parent,
  staging data or idling at the tail of a batch. Several jobs sharing one
  card are a separate question (`--slots-per-gpu`, below).
- **Wall clock** for `--instances` instances is the larger of instance-hours
  / instances and one segment's critical path. Segments and the tiles inside
  a segment run independently, but a segment's parent must finish before
  its tiles start. In `wrf-nests` plans every nest runs in one process, so
  the whole run is the critical path.
- **Spot** adds `spot_rework_fraction` (10%) for work lost to
  interruptions, to the cost, to `spot_instance_hours` and to
  `wall_clock_days_spot`. `--deadline-days` sizes the instance count on
  spot hours whenever spot is priced, and says which market it used.
- **Mode flags**: a flag the chosen mode would ignore is refused rather
  than dropped. `--plan` refuses `--footprint-km2`, `--chain-m`,
  `--leaf-dx-m`, `--nz`, `--parent-nz` and `--cover-factor`, and
  `--partition` refuses `--footprint-km2`. A `--chain-m` given with a
  partition applies only to clusters whose leaf it ends at.
- **Partition spacing**: a partition's clusters were cut at their own
  `leaf_dx_m`, so a `--leaf-dx-m` (or a job's spacing) that disagrees with
  any of them is **refused**, naming the partition's spacing. It used to be
  silently ignored. Re-partition at the spacing you want
  (`woof energy partition --leaf-dx-m`), or let `woof energy hindcast
  optimize` re-partition per spacing and card. Rows with no spacing of
  their own take the flag.
- **Band**: `low` is the cheapest utilisation pair (below) at 1.5x the
  fitted throughput and `high` the dearest at 0.5x, so the band covers both
  the bandwidth scaling and the utilisation curve.
- **Plan mode only**: a plan whose tiles were sized for a bigger card than
  the offer's is refused for that offer, with the `--card` to re-plan with.
  Summary and partition modes size tiles to each offer's card.

### Several jobs per GPU (`--slots-per-gpu`, `--mps`)

`woof energy worker --slots-per-gpu N|auto [--mps]` packs jobs onto each
card (docs/energy-hindcast/queue-worker.md). The estimate prices that with
the same flags. It is a model, not a measurement: no packed run has been
timed by this tree.

- **Copies.** `k = min(N, floor((card GiB - 1 GiB margin) / the domain's
  priced peak))`. With `auto`, N is 8, capped at the offer's vCPUs per GPU
  when the deploy machine tables know the instance, as the worker caps it
  at its CPUs. The cap is 4 on g5/g6e.xlarge and 8 on p5/p4d. A domain
  priced at most of the card packs once and gains nothing.
- **Utilisation curve.** Compute has two parts: the occupancy-bound part,
  under the utilisation pair's knee `u(c) = c / (c + S h)`, and the
  per-step cost `L`. Under **CUDA MPS** (`--mps`) kernels of separate
  processes run concurrently, so `k` copies of a grid of `c` cells fill the
  card like one grid of `k * c` cells. The ideal gain on the
  occupancy-bound part is `u(k c) / u(c)`. That is close to `k` for a grid
  far below the knee and close to 1 for a grid that already fills the card,
  so it saturates with cells. One copy's launches also overlap the other
  copies' kernels, so the ideal gain on `L` is `k`. **Without MPS** kernels
  time-slice and both gains are 1.
- **Overlap.** Either way, a job's startup, output and checkpoint time
  overlaps the other jobs' work. Ideally `1 - 1/k` of it is hidden.
- **Realisation.** A fraction `r` of each ideal is realised: 0.5
  pessimistic, 0.75 central, 1.0 optimistic. This is an assumption, and it
  stands for MPS contention, shared L2 and memory bandwidth, and CPU launch
  threads. The `high` band pairs the pessimistic `r` with the pessimistic
  throughput and utilisation pair, and `low` pairs the optimistic ones.
- **Card-seconds per copy** = `bandwidth / (1 + r (u(kc)/u(c) - 1)) +
  launch / (1 + r (k - 1)) + overheads × (1 - r (1 - 1/k))`. The critical
  path gets longer, because a packed copy's compute wall is `k` times its
  share of the card.

With `--slots-per-gpu 1` (the default) every figure is unchanged. When
packing, `gpu_hours` are card-hours with jobs sharing cards.
`gpu_hours_one_job_per_gpu` gives the unpacked figure, and each offer's
`packing` block gives the copies, the compute gain and the launch gain per
role. `--mps` without packing is refused.

In summary and partition modes, each cluster gets the planner's default
chain: a parent at 5x the leaf spacing, forced directly by the global
model, then the leaf. Use `--chain-m 2500,500,100` to set a longer chain,
which adds intermediate tiles. The domains are shaped as follows:

- **Parent**: covers the leaf cover plus 16 km on every side.
- **Leaf**: covers the leaf area, split into square tiles that fit the card.
  Tiles are sized at 0.84 KiB per cell plus a 3.4 GiB intercept, the upper
  ends of the fit in section 8.1 of the operational envelope. Each tile has
  an 11-cell lateral rim.
- **Clock**: 5 s per km of spacing.
- **Levels**: 49 for parents (`--parent-nz`); leaves and LES intermediates
  at 250 m or finer take `--nz`, default 49, the hindcast's wind-energy
  column. The optimizer prices parents at the leaf count too, because the
  hindcast's column is the whole model's.

### Utilisation: small domains do not saturate big cards

Throughput is not linear in cells. The knee `h` and the per-step cost `L`
make a small domain on a big, fast card pay for idle SMs and launch
latency. Without them, the ranking would favour an H100 for a 0.3 M-cell
tile. A large domain on a small card pays for the extra tiles and rims the
memory fit forces. **Both curves are assumptions, not measurements.** Three
pairs span them:

| pair | `h` (cells per SM at half throughput) | `L` (per step) | H100 half throughput at |
|---|---:|---:|---:|
| optimistic | 2,048 | 0.5 ms | 0.27 M cells |
| central | 4,096 | 1 ms | 0.54 M cells |
| pessimistic | 16,384 | 3 ms | 2.2 M cells |

`K` is refitted on the receipts for each pair, so each pair reproduces
them, and the band is the envelope over the pairs. Every offer reports its
work-weighted `utilisation` (saturated time / modelled time) under the
optimistic, central and pessimistic pairs. On the US + EU-offshore
partition at 300 m with 49 levels, the central utilisation is 0.59 on an
H100, 0.66 on an A100 40 GB, 0.74 on an A10G and 0.80 on an L4. The
pessimistic pair takes the H100 to about 0.3. Replace the pairs with a
measured curve before trusting a card ranking at either end: sweep one
domain from 0.1 M to 20 M cells on the card you will rent.

### Calibration receipts

| receipt | card | what was timed | K (cell-steps/s per GB/s) |
|---|---|---|---:|
| [FIRST-LIGHT.md](../public/FIRST-LIGHT.md) section 3 | RTX 4070, 504 GB/s, 46 SMs | 342x272x49, 2160 steps in 523 s, output included | 39,057 |
| `woof/live_products.py` docstring | RTX 5070 Ti, 896 GB/s, 70 SMs | 12/3/1 km stack (170², 336², 408x402) at 170 s per model hour; nz 49 and the 60/15/5 s clock are assumed | 49,480 |
| fitted `K` | | geometric mean | **43,961** |

One receipt was read and deliberately left out of the fit. The RTX 5090
acceptance run in FIRST-LIGHT.md section 5 (250x200x49, 361 steps in about
200 s) was a cold first run on Windows (WDDM), and its wall time includes
launch and first-frame compilation. It implies about 16x less throughput
per GB/s than the two fitted receipts. It is a reminder that a first run
on a fresh machine is much slower than steady state.

At saturation, `K` predicts about 147 M cell-steps/s on an H100 SXM and
13 M on an L4. Treat these as scaled guesses until a datacentre card has
been timed. The first hour on a rented card should be a timed Wales tile,
and its `K` should replace this table's.

## Cards: `--card` for cloud parts

`woof.card_catalog` holds a row for each GPU model with its memory, SM
count and bandwidth. A `provenance` field keeps the two kinds of row apart:

- **Datacentre rows** are **declared, not measured locally**. Every figure
  is the vendor's specification, and the capacity is the total that
  `nvidia-smi` commonly reports on that part.
- **Consumer rows** marked `measured` (RTX 5090, 5070 Ti, 3080) carry the
  capacity and SM count this project read off real devices. Their bandwidth
  is the vendor's nameplate figure.

| `--card` | part | memory | SMs | GB/s |
|---|---|---:|---:|---:|
| `h100-sxm-80gb`, `h100-sxm` | H100 SXM5 80GB (AWS p5, GCP a3-highgpu) | 81,559 MiB | 132 | 3,350 |
| `h100-pcie-80gb`, `h100-80gb`, `h100` | H100 PCIe 80GB | 81,559 MiB | 114 | 2,000 |
| `h200-141gb`, `h200` | H200 141GB | 143,771 MiB | 132 | 4,800 |
| `a100-80gb` | A100 SXM4 80GB (GCP a2-ultragpu) | 81,920 MiB | 108 | 2,039 |
| `a100-40gb`, `a100` | A100 SXM4 40GB (AWS p4d) | 40,960 MiB | 108 | 1,555 |
| `l40s-48gb`, `l40s` | L40S 48GB (AWS g6e) | 46,068 MiB | 142 | 864 |
| `l4-24gb`, `l4` | L4 24GB (GCP g2) | 23,034 MiB | 58 | 300 |
| `a10g-24gb`, `a10g` | A10G 24GB (AWS g5) | 23,028 MiB | 80 | 600 |

Rules for how these spellings resolve:

- **No variant named.** When a spelling names a model but not its variant,
  the conservative variant is chosen: bare `h100` is the PCIe part (same
  memory, less bandwidth) and bare `a100` is the 40 GB part.
- **Sizing doors.** `woof domain`, `woof energy plan` and every door that
  reads `woof.domain_wizard.card_capacity_gib` size a cloud spelling against
  the capacity the part reports. An L40S is sized at 45.0 GiB, not 48. A
  declared card never probes the local GPU.
- **Device profile.** Sizing a declared card keeps the wizard's 170-SM
  reference device profile. Every part above has fewer SMs, so the
  SM-proportional workspaces are over-priced, which is the safe direction.
- **Hex doors.** The hex mesh doors (`woof.hex.mesh_point.CARD_ALIASES`)
  have no rows for these parts yet, because nothing in the hex lane needs
  them.

Sizing with no GPU in the machine, for a declared H100 (writes a TOML; the
wizard reports "an estimate for a declared 79.6475 GiB card, not a
measurement of hardware in this machine"):

```bash
GPUWM_NO_LOCAL_GPU=1 woof domain --point 51.72,-4.26 --card h100-80gb \
    --cycle 2025-01-01T00 --hours 30 --source era5 --out wales-h100.toml
```

## Prices

`woof/data/energy/cloud_prices.json` (`woof-energy.cloud-prices.v1`) is a
dated, editable table:

- **Contents**: hourly on-demand and spot prices for AWS `p5.48xlarge`,
  `p4d.24xlarge`, `g6e.xlarge`/`12xlarge`/`48xlarge`,
  `g5.xlarge`/`12xlarge`/`48xlarge` (A10G 24 GB) and `g6.xlarge`/`12xlarge`
  (L4 24 GB), and GCP `a3-highgpu-8g`, `a2-ultragpu-1g` and
  `g2-standard-4`, with the region and the card each instance carries. It also has S3 and GCS rates per
  GB-month, per 1,000 PUTs and per GB of internet egress, and a `sources`
  URL for each block.
- **Date**: it is dated **2025-07-01**. On-demand rows are list prices as of
  that date, the G5/G6 multi-GPU rows included (they were not part of AWS's
  June 2025 price cut, which the P5/P4d rows reflect).
- **Spot rows** are illustrative typical values. Spot prices change hourly,
  and spot capacity for 8-GPU instances is often unavailable.
- **No live prices**: nothing is fetched. The estimator notes when the table
  is older than its `stale_after_days` (180).

Edit the file, or pass `--prices` with a copy holding current or negotiated
rates, before quoting a number. `--instance aws:p5.48xlarge` (repeatable),
or `--instance aws` for every AWS row, limits the offers. `--market spot`
or `--market on-demand` limits the markets.

## Worked example: the Wales fixture, one year

This is the unit's end-to-end check, with no GPU in the machine. First,
build a `wrf-tiles` plan sized for an H100 at 200 m leaves:

```bash
GPUWM_NO_LOCAL_GPU=1 woof energy plan tests/fixtures/energy/sites_wales.json \
    --topology wrf-tiles --card h100-80gb --dx-m 200 \
    --start 2025-01-01T00 --hours 30 -o plan-wales-200m
```

The plan has two domains:

- a 134x66x49 parent at 1 km with a 5 s step
- one 520x180x60 LES leaf tile at 200 m with a 1 s step, a 7.3 GiB peak and
  15-minute history

Then price 365 daily 30 h segments with 6 h of spin-up (the daily layout,
given explicitly; the default is the 6-hourly one):

```bash
woof energy estimate --plan plan-wales-200m/plan.json --segments 365 \
    --segment-hours 30 --spinup-hours 6 --instance aws:p5.48xlarge \
    --instance gcp:a3-highgpu-8g --instance aws:g6e.xlarge
```

```text
offer                      card                 GPU-h          GPU-h band    inst-h   wall d              on-demand                   spot  store TB
aws:p5.48xlarge            h100-sxm-80gb          493       320-1,140            73     3.02    $4.0k ($2.6k-$9.2k)    $2.2k ($1.4k-$5.1k)      10.6
gcp:a3-highgpu-8g          h100-sxm-80gb          493       320-1,140            73     3.02   $6.4k ($4.2k-$14.8k)    $1.8k ($1.1k-$4.1k)      10.6
aws:g6e.xlarge             l40s-48gb            1,866     1,204-4,383         2,195    91.48    $4.1k ($2.6k-$9.6k)    $1.8k ($1.2k-$4.3k)      10.6
```

The two H100 offers need the same GPU-hours and differ only in price. With
the 2025-07 table, GCP spot is cheaper than AWS spot, and AWS on-demand is
cheaper than GCP on-demand. Storing every variable on every level every
15 minutes is 10.6 TB a year for this one tile. With `--store-preset energy`
(the planner's leaf history preset) it is about 3.3 TB.

## Worked example: all public wind farms worldwide, 1 year, 200 m leaves

There is no worldwide partition yet, so this example prices summary
statistics. Rerun it with `--partition` on the real inventory. Every input
below is an assumption, stated with its uncertainty.

| input | central | range | basis |
|---|---:|---|---|
| wind farms | ~30,000 | ±30% | order of magnitude of farm phases in Global Energy Monitor's Global Wind Power Tracker; the US Wind Turbine Database (USWTDB) alone lists about 75,000 turbines in about 1,800 projects |
| installed capacity | ~1,136 GW | ±5% | GWEC Global Wind Report 2025, end-2024 cumulative capacity |
| project area per MW | 0.345 km² | 0.12-0.57 | Denholm et al. 2009 (NREL/TP-6A2-45834): 34.5 ± 22.4 ha/MW total project area |
| total farm footprint | **390,000 km²** | 140,000-650,000 | capacity × area per MW |
| clusters | **4,000** | 2,000-8,000 | about 7.5 farms per cluster; replaced by the partition's real count |
| leaf cover factor | 1.5 | | rectangle cover and corridor over large footprints (`--cover-factor`) |
| leaves | 200 m, 60 levels to 20 km | | the planner's LES leaf ladder |
| parent | 1 km, 49 levels | | the default ratio-5 chain |
| segments | 365 × 30 h | | 6 h spin-up, 24 h stored per day (the daily layout; the driver's default is now 6-hourly, below) |

The citations above are the author's reading of public totals. Check them
against the current GEM and USWTDB releases before quoting them.

```bash
woof energy estimate --clusters 4000 --footprint-km2 390000 \
    --cover-factor 1.5 --leaf-dx-m 200 --nz 60 --segments 365 \
    --segment-hours 30 --spinup-hours 6 \
    --instance aws:p5.48xlarge --instance aws:p4d.24xlarge \
    --instance aws:g6e.xlarge --instance aws:g6e.48xlarge \
    --instances 100 --deadline-days 30
```

```text
offer                      card                 GPU-h          GPU-h band    inst-h   wall d              on-demand                   spot  store TB
aws:p5.48xlarge            h100-sxm-80gb      394,962   198,213-1,778,525    58,083    24.20 $3.20M ($1.60M-$14.40M) $1.76M ($881.8k-$7.91M)   1,998.4
aws:p4d.24xlarge           a100-40gb          699,137   355,427-3,098,748   102,814    42.84 $2.26M ($1.15M-$10.01M) $995.2k ($506.0k-$4.41M)   1,998.4
aws:g6e.xlarge             l40s-48gb        1,398,232   673,632-6,805,020 1,644,979   685.41 $3.06M ($1.47M-$14.90M) $1.36M ($653.8k-$6.60M)   1,998.4
aws:g6e.48xlarge           l40s-48gb        1,398,232   673,632-6,805,020   205,622    85.68 $6.20M ($2.98M-$30.15M) $2.73M ($1.31M-$13.26M)   1,998.4
```

AWS compute for the central scenario:

| AWS instance | GPU-hours | on-demand | spot | spot instances to finish in 30 days | on-demand wall clock on 100 instances |
|---|---:|---:|---:|---:|---:|
| p5.48xlarge (8x H100) | 395k (198k-1.78M) | $3.2M ($1.6M-$14.4M) | $1.8M ($0.88M-$7.9M) | 89 (45-400) | 24 days |
| p4d.24xlarge (8x A100 40GB) | 699k (355k-3.10M) | $2.3M ($1.2M-$10.0M) | $1.0M ($0.51M-$4.4M) | 158 (80-697) | 43 days |
| g6e.xlarge (1x L40S) | 1.40M (0.67M-6.81M) | $3.1M ($1.5M-$14.9M) | $1.4M ($0.65M-$6.6M) | 2,514 | 685 days |
| g6e.48xlarge (8x L40S) | 1.40M | $6.2M ($3.0M-$30.2M) | $2.7M ($1.3M-$13.3M) | 315 | 86 days |

The instance counts are sized on spot hours, which include the 10% rework.
On-demand needs about 10% fewer instances.

The bands are wide here because the clusters are small: about 146 km² of
leaf cover each, one 0.4 M-cell tile, which the central utilisation curve
puts at 0.36 of an H100 and the pessimistic one at 0.14. That is the
utilisation term doing its job: a tiny domain on a big card is not priced
as if it saturated the card.

Bracketing scenarios (p5.48xlarge):

- **Low**: 2,000 clusters over 140,000 km². 184k GPU-hours, $1.49M
  on-demand, $0.82M spot, 42 spot instances for 30 days.
- **High**: 8,000 clusters over 650,000 km². 760k GPU-hours, $6.2M
  on-demand, $3.4M spot, 171 spot instances for 30 days.

Combining the scenario spread with the throughput and utilisation band,
**this daily-layout, 200 m, 60-level year costs roughly $0.4M to $28M in
AWS compute on p4d or p5 (spot at the low end, on-demand at the high end).
The central estimate is about $1.8M on p5 spot, or about $1.0M on p4d spot
if 40 GB tiles are acceptable.** The real inventory partition and the
cost-efficient defaults below bring the central estimate down by an order
of magnitude.

Storage is the other large bill:

- **Size**: every variable on every model level at 15-minute cadence is
  about **2.0 PB** compressed in the central scenario (0.7-3.4 PB across the
  scenarios). The `energy` field preset is about 0.63 PB.
- **S3 Standard**: about $46k per month to hold, at the first-tier rate
  (cheaper at volume tiers).
- **Egress**: about $180k to download once over the internet.

Storing derived hub-height fields instead of full model levels, or
quantising before compression, cuts the store by an order of magnitude.

What moves these numbers most:

- **Throughput on the real card.** The band is about 0.5x to 4x around a
  bandwidth-scaled guess with an assumed utilisation curve. Time one Wales
  tile, and a cells sweep, on the instance you will rent and update the
  calibration table and the utilisation pairs.
- **Cluster size.** The central scenario's 4,000 clusters average about
  146 km² of leaf cover. That is one roughly 82x82x60 tile, 0.4 M cells,
  which fills less than half an H100 under the occupancy knee. Fewer,
  larger clusters would reduce GPU-hours, and so would several small jobs
  sharing one GPU (next).
- **Packing.** The same command with `--slots-per-gpu auto --mps` packs 8
  copies per H100 or A100 (4 per L40S on g6e.xlarge, capped by its 4
  vCPUs). The model then gives the following. These are ESTIMATES: the
  realisation `r` is an assumption, and nothing packed has been timed.

  | AWS instance | GPU-hours, one job per GPU | packed + MPS | spot, one job per GPU | packed + MPS |
  |---|---:|---:|---:|---:|
  | p5.48xlarge | 395k | 197k (106k-697k) | $1.76M | $0.87M ($0.47M-$3.10M) |
  | p4d.24xlarge | 699k | 393k (219k-1.32M) | $1.00M | $0.56M ($0.31M-$1.88M) |
  | g6e.xlarge | 1.40M | 843k (438k-3.55M) | $1.36M | $0.82M ($0.43M-$3.45M) |

  The spot instances needed for 30 days on p5 fall from 89 to 45. The
  pessimistic end narrows the most (1.78M to 0.70M GPU-hours on p5),
  because packing hides the per-step cost that dominates small grids under
  the pessimistic utilisation pair. Without MPS (`--slots-per-gpu auto`
  alone) the model credits only the overlap of startup and output, about
  2% here. Time-sliced kernels never share the
  SMs, so MPS is the half of this that matters. Time one packed Wales job
  on the card you will rent before you rely on it.
- **Parent overhead.** Each cluster's 1 km parent carries a 16 km buffer.
  Across 4,000 small clusters that is about 10% of compute.
- **Spot capacity.** 8-GPU spot capacity (p5, a3) is often unavailable at
  the scale of 70+ instances. Plan on a mix, or on capacity blocks or
  committed-use pricing, which the table does not carry.

## Cheapest configuration

The requirement is a year of 100-300 m leaves under a model top of at most
20 km, with at least 8 model layers below 250 m. Four levers move the bill,
in this order:

1. **Leaf spacing.** A leaf's cost goes about as `dx^-3` (cells times
   steps). 300 m is the coarse end of the requirement and costs about a
   third of 200 m and a twentieth of 100 m.
2. **Spin-up.** Each run is `6 h + spin-up` long but stores 6 h, so 3 h of
   spin-up costs 1.5x the 0 h layout, 2 h costs 1.33x and 1 h 1.17x.
3. **Levels.** The smallest lean wind-energy column (`HINDCAST_NZ` = 49, see
   [vertical.md](vertical.md)) keeps the 70-level preset's 10 rotor layers
   and costs about 0.76x of it.
4. **Card.** At 300 m the leaf tiles are small, and the A10G (`g5.xlarge`)
   wins on spot price per useful GPU-hour. The H100 is the dearest despite
   being the fastest, because its utilisation on these tiles is about 0.59.

### Measured on the real partition

These figures come from the estimator and optimizer on the US +
EU-offshore inventory (78,002 sites, 440 clusters). They are AWS spot,
central with the band in brackets, 1,460 runs (one per 6-hourly init for a
year), the 2025-07-01 price table, and a re-partition at each spacing and
card. They are **estimates**: the band is the throughput and utilisation
band above, and spot prices are illustrative.

| leaf dx | levels | run / spin-up | g5.xlarge (A10G) | p4d.24xlarge (A100 40 GB) | g6e.xlarge (L40S) | p5.48xlarge (H100) |
|---:|---:|---|---:|---:|---:|---:|
| 200 m | 70 | 8 h / 2 h | $439k ($285k-$1.01M) | $466k | $581k | $725k |
| 100 m | 70 | 8 h / 2 h | $3.08M ($2.05M-$6.16M) | $3.10M | $3.82M | $4.55M |
| 300 m | 70 | 8 h / 2 h | $156k ($98k-$426k) | $176k | $221k | $293k |
| 300 m | 49 | 8 h / 2 h | $118k ($72k-$358k) | $138k | $176k | $238k |
| 300 m | 49 | 6 h / 0 h | $90k ($55k-$270k) | $106k | $133k | $185k |
| **300 m** | **49** | **9 h / 3 h (default)** | **$133k ($80k-$403k)** | $154k | $197k | $264k |

The first rows reproduce what was measured before the defaults changed:
$434k on g5 and $456k on p4d at 200 m / 70 levels / 8 h with 2 h spin-up,
$153k and $169k at 300 m, $118k and $135k at 300 m with 50 levels, and
$90k and $103k with 6 h runs and no spin-up. They differ by a few per cent
because the model now charges a per-step cost. 100 m was about $3.1M.

The **defaults are now the cost-efficient end of the requirement**:

- `woof energy partition --leaf-dx-m` and `woof energy hindcast plan`
  default to **300 m** (100-250 m on request).
- Every hindcast domain carries the **49-level** wind-energy column
  (`--nz` overrides; the column still refuses a count that breaks the
  rotor or top rules).
- Runs are **9 h from each 00/06/12/18 UTC init with 3 h spin-up**, keeping
  `[init + 3 h, init + 9 h)`, so the 6 h windows stitch continuously. That
  costs 1.48x the spin-up-free layout (`--segment-hours 6 --spinup-hours 0`)
  in exchange for no frame younger than 3 h into its run.
- `woof energy estimate` prices the same campaign by default.

At the default, the year costs **about $133k on g5.xlarge spot ($80k-$403k)
with about 440 instances for 30 days**, or $154k on 25 p4d.24xlarge. Storage
is separate: the `full` preset on every level is about 0.94 PB compressed
at this spacing (about $22k a month on S3 Standard); `--store-preset
energy` is about a third of that.

### The recipe

```bash
# 1. Let the optimizer pick spacing, levels and offer for the deadline,
#    keeping each cut partition so the chosen one can be planned.
woof energy hindcast optimize --sites sites.json --deadline-days 30 \
    --instance aws --partition-dir partitions -o optimize.json
# 2. Run the chosen row's recipe (optimize.json "chosen" -> "recipe"), e.g.
woof energy partition sites.json --leaf-dx-m 300 --card a10g-24gb \
    -o partitions/partition-300m-22gib.json
woof energy hindcast plan --partition partitions/partition-300m-22gib.json \
    --start 2025-01-01 --end 2026-01-01 --card a10g-24gb -o hindcast
```

Before committing the budget, time one tile on the chosen instance and a
cells sweep for the utilisation curve, and refresh the price table.

## `woof energy hindcast optimize`

The optimizer ranks configuration x offer rows by estimated cost and keeps
only those that meet a deadline.

**What it searches**: the product of these, each held to the requirement
or refused by name:

- `--leaf-dx-m` (default `300,250,200`): each must be within 100-300 m.
- `--nz` (default `49,60,70`): each must give a lean wind-energy column
  (at least 8 layers below 250 m, a 12 m lowest layer, top at most 20 km, no
  layer deeper than 1000 m). Below 49 is refused with the preset.
- `--spinup-hours` (default `3`, the driver's) and `--keep-hours` (default
  `6`, one run per init). Pass `--spinup-hours 0,1,2,3` to rank the
  spin-ups too.
- Every price-table offer, narrowed with `--instance aws` or
  `--instance aws:g5.xlarge`.

**Re-partitioning**: the clusters depend on the card's memory (a parent
chain must fit) and on the spacing. The optimizer therefore re-cuts the
sites in-process with `woof energy partition`'s own code, once per
(memory rounded to the GiB, spacing). A10G and L4 share a cut, and so do
H100 and A100 80 GB. A 78k-site cut takes about a second. `--sites` cuts a
sites document. `--partition` re-cuts the partition's own sites with its
linkage, extent, site-count and corridor parameters. `--partition
--no-repartition` prices the partition as cut, at its spacing only and on
cards with at least its memory. `--partition-dir` keeps each cut and reuses
it on the next search.

**Deadline**: a row meets `--deadline-days` when one run's critical path
fits and, with `--max-instances` (spot capacity), the instance count the
central estimate needs is within the cap. The count the pessimistic band
needs is reported beside it. When nothing meets the deadline, the refusal
names the closest row and why it misses.

**Ranking**: rows are ranked by `--rank-market` (default spot) central
cost, then pessimistic cost, then wall clock. The other market is reported
on every row and ranked separately in `top_by_other_market`. Storage is
reported but not ranked.

**Output**: `woof-energy.cost-optimize.v1` goes to stdout (and to `-o`,
written atomically), with a table on stderr:

- `inputs`: the sites or partition, with sha256.
- `search`: the candidate lists, the offers, every partition cut (spacing,
  card, GiB, clusters, path, built or reused, seconds) and the row count.
- `constraints` and `ranking`.
- `model`: the throughput band, the utilisation pairs with their refitted
  `K`, and the status "estimate".
- `prices`: `as_of`, `age_days`, `stale`, and the spot caveat.
- `top`: the ranked rows. Each has the configuration, offer, card,
  `gpu_hours`, `instance_hours`, `cost_usd` per market with the band,
  `deadline` (instances, instances needed central/high, wall clock),
  `utilisation`, storage, the partition used, and `"kind": "estimate"`.
- `chosen`: the first row plus a `recipe` (partition, estimate and plan
  commands).
- `spinup_sensitivity`: the chosen row re-priced at 0, 1, 2 and 3 h of
  spin-up, relative to the chosen one.
- `excluded` and `notes`.

A refusal exits 2 with `{"schema": "woof-energy.cost-optimize.v1",
"refused": "..."}`.

## Forcing downloads: per job versus once per cycle

The `forcing` section prices fetching the boundary data, with and without
the forcing cache ([forcing-cache.md](forcing-cache.md)). One cycle's
download is the bytes and objects `woof.download_budget` measures for
`--forcing-source` (default `met-office-global`) over the segment's fetch
window, or `--forcing-gb-per-cycle` (and `--forcing-objects-per-cycle`).
Cycles are the segments and jobs per cycle are the clusters, so a `--plan`
(one cluster) shows no saving and says so.

| | per job (no cache) | per cycle (cache) |
|---|---|---|
| downloads from the publisher | jobs | cycles |
| GPU-slot hours | jobs x cycle bytes / `--publisher-mbps` | cycles x cycle bytes / `--publisher-mbps` + jobs x cycle bytes / `--cache-read-mbps` |
| requests | jobs x objects publisher GETs | cycles x objects cache PUTs + jobs x objects cache GETs |
| storage | - | cycles x cycle bytes x `--forcing-cache-days` / 30.4 GB-months |

Assumptions, all labelled in `forcing.assumptions`: one job draws
`--publisher-mbps` (default 100 MB/s) from a publisher and
`--cache-read-mbps` (default 400 MB/s) from a cache in its own region; a
cycle stays in the cache `--forcing-cache-days` (default 7); the publisher
charges `--publisher-egress-usd-per-gb` (default 0, the public open-data
buckets) and same-region reads are free. Download time is priced as
GPU-slot time (the instance price over its GPUs: the card idles while its
job waits for bytes), on each priced market, in each offer's
`forcing_usd` (`per_job_fetch`, `per_cycle_cache`, `saving`); PUT, GET and
storage use the offer's provider rates from the price table
(`usd_per_1000_put`, `usd_per_1000_get`, `usd_per_gb_month`). A source the
cache does not share (`gfs`, `hrrr`, `era5`) shows the same figures on
both sides and the reason. The forcing figures are reported beside the
GPU-hours, not added to them, and the cache does not change decode or
preparation time. `--no-forcing` leaves the section out.

For the worldwide year (440 clusters x 1,460 cycles of Met Office global,
3.246 GB and 41 objects per cycle) the cache avoids 2,080 TB of downloads
and 4,331 GPU-slot hours: $29.8k on p5.48xlarge and $47.7k on
a3-highgpu-8g at on-demand rates.

## Output document

`woof-energy.cost-estimate.v1` holds:

- **Campaign and inputs**: `mode`, `inputs` (paths and sha256 of the plan or
  partition, the summary statistics, the job's facts) and `campaign`
  (segments, hours, spin-up, stored hours, clusters).
- **Work**: `work_per_segment`. In plan mode it lists the domains;
  otherwise it gives totals per role for the first offer, and each offer
  carries its own, because synthesised tiles are sized to each card.
- **Model and prices**: `model` (the calibration with its receipts and the
  excluded receipt, the throughput band, the `utilisation` pairs with each
  pair's refitted `K`, every assumption) and `prices` (the table's `as_of`,
  sources and path).
- **Storage**: `storage` (preset, fields per frame, stored leaf columns,
  frames per segment, raw and stored bytes, PUT count) for the offer it
  names in `offer`. Each offer also carries its own `storage` byte counts.
- **Offers**: each entry in `offers` has:
  - the card row and `fits` (or `refused`)
  - `gpu_hours` and `instance_hours`, each with `central`/`low`/`high`, and
    `gpu_hours_by_component`
  - `wall_clock_days` and `critical_path_hours_per_segment`
  - `utilisation`: work-weighted saturated / modelled time under the
    optimistic, central and pessimistic pairs
  - `cost_usd` per market; when spot is priced, also `spot_instance_hours`
    and `wall_clock_days_spot`
  - `instances_for_deadline` (with `--deadline-days`), `storage` and
    `storage_usd`
  - with `--slots-per-gpu` above 1: `packing` (slots, MPS, vCPUs per GPU,
    copies, compute gain and launch gain per role) and
    `gpu_hours_one_job_per_gpu`
  - `forcing_usd` per market: forcing downloads per job, through the
    cache, and the saving. When packing, its slot-hours are still priced at
    a whole GPU and carry a `note` saying they are an upper bound
  - in partition mode, a `clusters` row per cluster
- **Forcing**: `forcing` (source, shared or why not, one cycle's bytes and
  objects with their basis, `per_job_fetch`, `per_cycle_cache`, `saving`,
  `assumptions`), unless `--no-forcing`.
- **Notes**: `notes`, such as the price table's age.

A refusal exits 2 and prints `{"schema": ..., "refused": "..."}`.
