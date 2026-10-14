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
tree. Each figure comes with a band (0.5x to 1.5x of the fitted throughput),
and the JSON lists the receipts and every assumption, so you can see what a
number rests on.

> **Command name.** The estimator is registered as `woof energy estimate`.
> When the `woof energy hindcast` group lands, it mounts the same parser as
> `woof energy hindcast estimate` through `woof.energy.cli.register_estimate`.
> The flags and output are identical.

## Three ways to say what to price

| mode | flags | what is priced |
|---|---|---|
| plan | `--plan PLAN.json` | the domains of a `woof-energy.plan.v1` (`wrf-tiles` or `wrf-nests`), one segment each, repeated `--segments` times |
| partition | `--partition PARTITION.json` | each row of the document's `clusters` list, synthesised into parent and leaf tiles the way the `wrf-tiles` planner shapes them |
| summary | `--clusters N --footprint-km2 A` | `N` equal clusters sharing `A` km² of farm footprint, for when no partition exists yet |

`--job JOB.json` (a `woof-energy.hindcast-job.v1`-like document) supplies
the segment count, segment hours and spin-up when they are present. Flags
override it. With neither, a campaign is 365 segments of 30 h with 6 h of
spin-up, and 24 h of each segment is stored.

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
- `throughput = K * B * cells / (cells + S * 4096)` cell-steps per second.
  `K` is the fitted constant. The second factor is an occupancy knee: a
  grid with few cells per SM cannot fill the card.
- `compute = steps * cells * (n_sound + 4) / (4 + 4) / throughput`. The
  receipts ran WRF's 4 acoustic substeps, and one large step is assumed to
  cost as much as 4 substeps.
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
  default 0.85).
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
- **Band**: `low` uses 1.5x the fitted throughput and `high` uses 0.5x.
- **Plan mode only**: a plan whose tiles were sized for a bigger card than
  the offer's is refused for that offer, with the `--card` to re-plan with.
  Summary and partition modes size tiles to each offer's card.

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
- **Levels**: 49 for parents; 60 for LES leaves and intermediates at 250 m
  or finer (`--nz`, `--parent-nz`).

### Calibration receipts

| receipt | card | what was timed | K (cell-steps/s per GB/s) |
|---|---|---|---:|
| [FIRST-LIGHT.md](../public/FIRST-LIGHT.md) section 3 | RTX 4070, 504 GB/s, 46 SMs | 342x272x49, 2160 steps in 523 s, output included | 38,896 |
| `woof/live_products.py` docstring | RTX 5070 Ti, 896 GB/s, 70 SMs | 12/3/1 km stack (170², 336², 408x402) at 170 s per model hour; nz 49 and the 60/15/5 s clock are assumed | 49,183 |
| fitted `K` | | geometric mean | **43,738** |

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
  `p4d.24xlarge`, `g6e.xlarge`/`12xlarge`/`48xlarge` and `g5.xlarge`, and
  GCP `a3-highgpu-8g`, `a2-ultragpu-1g` and `g2-standard-4`, with the region
  and the card each instance carries. It also has S3 and GCS rates per
  GB-month, per 1,000 PUTs and per GB of internet egress, and a `sources`
  URL for each block.
- **Date**: it is dated **2025-07-01**. On-demand rows are list prices as of
  that date. The P5/P4d rows reflect AWS's June 2025 on-demand price cut.
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

Then price 365 daily 30 h segments:

```bash
woof energy estimate --plan plan-wales-200m/plan.json --segments 365 \
    --segment-hours 30 --instance aws:p5.48xlarge \
    --instance gcp:a3-highgpu-8g --instance aws:g6e.xlarge
```

```text
offer                      card                 GPU-h          GPU-h band    inst-h   wall d              on-demand                   spot  store TB
aws:p5.48xlarge            h100-sxm-80gb          482       324-957              71     2.96    $3.9k ($2.6k-$7.7k)    $2.1k ($1.4k-$4.3k)      10.6
gcp:a3-highgpu-8g          h100-sxm-80gb          482       324-957              71     2.96   $6.3k ($4.2k-$12.4k)    $1.7k ($1.2k-$3.4k)      10.6
aws:g6e.xlarge             l40s-48gb            1,862     1,244-3,717         2,191    91.30    $4.1k ($2.7k-$8.1k)    $1.8k ($1.2k-$3.6k)      10.6
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
| segments | 365 × 30 h | | 6 h spin-up, 24 h stored per day |

The citations above are the author's reading of public totals. Check them
against the current GEM and USWTDB releases before quoting them.

```bash
woof energy estimate --clusters 4000 --footprint-km2 390000 \
    --cover-factor 1.5 --leaf-dx-m 200 --segments 365 --segment-hours 30 \
    --instance aws:p5.48xlarge --instance aws:p4d.24xlarge \
    --instance aws:g6e.xlarge --instance aws:g6e.48xlarge \
    --instances 100 --deadline-days 30
```

```text
offer                      card                 GPU-h          GPU-h band    inst-h   wall d              on-demand                   spot  store TB
aws:p5.48xlarge            h100-sxm-80gb      344,076   233,857-674,734      50,599    21.08 $2.78M ($1.89M-$5.46M) $1.53M ($1.04M-$3.00M)   1,998.4
aws:p4d.24xlarge           a100-40gb          649,799   437,672-1,286,180    95,559    39.82 $2.10M ($1.41M-$4.15M) $925.0k ($623.0k-$1.83M)   1,998.4
aws:g6e.xlarge             l40s-48gb        1,352,452   906,108-2,691,487 1,591,120   662.97 $2.96M ($1.98M-$5.89M) $1.31M ($879.5k-$2.61M)   1,998.4
aws:g6e.48xlarge           l40s-48gb        1,352,452   906,108-2,691,487   198,890    82.87 $5.99M ($4.01M-$11.93M) $2.64M ($1.77M-$5.25M)   1,998.4
```

AWS compute for the central scenario:

| AWS instance | GPU-hours | on-demand | spot | spot instances to finish in 30 days | on-demand wall clock on 100 instances |
|---|---:|---:|---:|---:|---:|
| p5.48xlarge (8x H100) | 344k (234k-675k) | $2.8M ($1.9M-$5.5M) | $1.5M ($1.0M-$3.0M) | 78 (53-152) | 21 days |
| p4d.24xlarge (8x A100 40GB) | 650k (438k-1.29M) | $2.1M ($1.4M-$4.2M) | $0.93M ($0.62M-$1.8M) | 146 (99-289) | 40 days |
| g6e.xlarge (1x L40S) | 1.35M (0.91M-2.69M) | $3.0M ($2.0M-$5.9M) | $1.3M ($0.88M-$2.6M) | 2,431 | 663 days |
| g6e.48xlarge (8x L40S) | 1.35M | $6.0M ($4.0M-$11.9M) | $2.6M ($1.8M-$5.3M) | 304 | 83 days |

The instance counts are sized on spot hours, which include the 10% rework.
On-demand needs about 10% fewer instances (71 p5.48xlarge).

Bracketing scenarios (p5.48xlarge):

- **Low**: 2,000 clusters over 140,000 km². 159k GPU-hours, $1.29M
  on-demand, $0.71M spot, 36 spot instances for 30 days.
- **High**: 8,000 clusters over 650,000 km². 658k GPU-hours, $5.3M
  on-demand, $2.9M spot, 148 spot instances for 30 days.

Combining the scenario spread with the throughput band, **the year costs
roughly $0.3M to $10M in AWS compute on p4d or p5 (spot at the low end,
on-demand at the high end). The central estimate is about $1.5M on p5
spot, or about $0.9M on p4d spot if 40 GB tiles are acceptable.**

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

- **Throughput on the real card.** The band is ±2x around a bandwidth-scaled
  guess. Time one Wales tile on the instance you will rent and update the
  calibration table.
- **Cluster size.** The central scenario's 4,000 clusters average about
  146 km² of leaf cover. That is one roughly 82x82x60 tile, 0.4 M cells,
  which fills less than half an H100 under the occupancy knee. Fewer,
  larger clusters, or several small tiles sharing one GPU (CUDA MPS or
  several processes per card, which this model does not credit), would
  reduce GPU-hours.
- **Parent overhead.** Each cluster's 1 km parent carries a 16 km buffer.
  Across 4,000 small clusters that is about 10% of compute.
- **Spot capacity.** 8-GPU spot capacity (p5, a3) is often unavailable at
  the scale of 70+ instances. Plan on a mix, or on capacity blocks or
  committed-use pricing, which the table does not carry.

## Output document

`woof-energy.cost-estimate.v1` holds:

- **Campaign and inputs**: `mode`, `inputs` (paths and sha256 of the plan or
  partition, the summary statistics, the job's facts) and `campaign`
  (segments, hours, spin-up, stored hours, clusters).
- **Work**: `work_per_segment`. In plan mode it lists the domains;
  otherwise it gives totals per role for the first offer, and each offer
  carries its own, because synthesised tiles are sized to each card.
- **Model and prices**: `model` (the calibration with its receipts and the
  excluded receipt, the throughput band, every assumption) and `prices` (the
  table's `as_of`, sources and path).
- **Storage**: `storage` (preset, fields per frame, stored leaf columns,
  frames per segment, raw and stored bytes, PUT count) for the offer it
  names in `offer`. Each offer also carries its own `storage` byte counts.
- **Offers**: each entry in `offers` has:
  - the card row and `fits` (or `refused`)
  - `gpu_hours` and `instance_hours`, each with `central`/`low`/`high`, and
    `gpu_hours_by_component`
  - `wall_clock_days` and `critical_path_hours_per_segment`
  - `cost_usd` per market; when spot is priced, also `spot_instance_hours`
    and `wall_clock_days_spot`
  - `instances_for_deadline` (with `--deadline-days`), `storage` and
    `storage_usd`
  - in partition mode, a `clusters` row per cluster
- **Notes**: `notes`, such as the price table's age.

A refusal exits 2 and prints `{"schema": ..., "refused": "..."}`.
