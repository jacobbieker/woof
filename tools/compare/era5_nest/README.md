# Breeze ERA5 regional domains in woof: throughput harness

Reproduces the two real-data limited-area cases NumericalEarth.jl/Breeze.jl
publish, MC3E (20 May 2011, Oklahoma, 150 x 136 x 50) and the AtmosphericRivers.jl
NE Pacific case (7-10 Dec 2025, 540 x 315 x 50). Both are driven by ERA5 at
about 12 km, and the harness times woof on each. The write-up is
[docs/comparisons/clima-numericalearth/regional-benchmark.md](../../../docs/comparisons/clima-numericalearth/regional-benchmark.md).

| file | what it does |
|---|---|
| `breeze_cases.py` | Both cases' geometry, dates and published timings, each with its source line. Also builds Breeze's 50-level column and its eta equivalent. |
| `make_nest_configs.py` | Runs `woof domain`, then sets the exact `Nx x Ny` the case needs, the forcing area, `radt` and restarts. |
| `nest_throughput.py` | Computes simulated h / wall h, cell-steps/s and s/step from a woof `progress.jsonl`, the Breeze driver JSON, or the published figures (with an optional bandwidth-scaled estimate). |
| `station_table.py` | Builds a markdown table of the ASOS scores (count, bias, RMSE per hour and quantity) from a run's verify-visuals receipts. |
| `woof_run.sh` | Runs `woof go` under the shared-card lock and samples the device memory used by its process tree. |
| `breeze_setup.sh` | Installs Julia 1.12.6 and the Breeze 0.12 / NumericalEarth 66112eac environment under `$BENCH_ROOT/julia`. |
| `breeze_mc3e_timing.jl` | NumericalEarth's `examples/breeze_downscaling_era5.jl` model, without figures or writers, with a timed window. |
| `breeze_run.sh` | Two modes. `prefetch` (CPU, downloads only) and `time` (GPU, under the lock, 6 GiB cap by default). |
| `gpu_memory_sampler.sh` | Logs the per-process `nvidia-smi` memory of a process tree. |
| `run_timing.sh` | Full-length timings for a clean card: MC3E 3 h, AR 72 h, and Breeze MC3E 3 h. |

Commands assume the repository root as the working directory and `PY` set to the
environment's Python. Data, runs and the Julia install go under
`BENCH_ROOT=~/wf-scratch/regional-benchmark`.

## woof, one case end to end (what the shared-card run did)

```bash
export PYTHONPATH=$PWD TMPDIR=$BENCH_ROOT/tmp
# 1. config: wizard + exact counts + Breeze column + hourly ERA5 + hourly RRTMGP
$PY tools/compare/era5_nest/make_nest_configs.py --case mc3e \
    --out-dir $BENCH_ROOT/mc3e-e2e --hours 3 --vram-gib 6 --no-restarts
# 2. ERA5 from Google's public ARCO store (no account); area from the TOML's [fetch]
cd $BENCH_ROOT/mc3e-e2e
$PY -m woof fetch --source era5 --era5-provider arco --retrieve \
    --cycle 2011-05-20T00 --hours 3 --cadence 1 \
    --area 26.78,-110.77,46.04,-84.20 --out data/mc3e_breeze_match
# 3. prepare + forecast + render + ASOS verification, under the lock, 6 GiB cap
CUPY_GPU_MEMORY_LIMIT=6442450944 WOOF="$PY -m woof" \
    $OLDPWD/tools/compare/era5_nest/woof_run.sh mc3e_breeze_match.toml go
# 4. throughput from the run's events.jsonl, and the ASOS table
$PY $OLDPWD/tools/compare/era5_nest/nest_throughput.py woof go --case mc3e
$PY -m woof verify-visuals go/run-* --cycle 2011-05-20T00Z
$PY $OLDPWD/tools/compare/era5_nest/station_table.py go/run-*
```

`woof go` starts the station verification in the background after the forecast.
On the shared-card run that pass stayed `pending`, and running
`woof verify-visuals` to completion (step 4) scored every hour against ASOS
reports from Dynamical.org's parquet mirror of IEM. MRMS and the HRRR
reference do not exist for 2011, so those arms remain pending.

For the AR domain use `--case ar --vram-gib 48`. Its fetch area crosses the
antimeridian (`18.45,168.87,61.50,-88.87`). At 8.5 M cells the forecast does
not fit a 6 GiB cap. On the shared card it was therefore fetched and prepared
only, with `woof static CONFIG --output static.npz` and then
`woof ingest CONFIG --output ingest.npz`; ingest needs the device for its
final state.

A reduced-domain correctness run of the same case used
`make_nest_configs.py --case ar --shrink 3 --vram-gib 6`. That is 180 x 105
at the same centre and spacing; pass `nest_throughput.py woof ... --cells
945000` for its rates.

## Breeze MC3E on the same card

```bash
tools/compare/era5_nest/breeze_setup.sh          # ~4 min precompile, 3.6 GB depot
# CDS: a two-line rc (url: https://cds.climate.copernicus.eu/api, key: <token>)
#      at $BENCH_ROOT/home/.cdsapirc, or CDS_RC=/path/to/rc
tools/compare/era5_nest/breeze_run.sh prefetch   # ERA5 + ETOPO2022 + CGLS albedo, CPU
BREEZE_HOURS=1 tools/compare/era5_nest/breeze_run.sh time   # GPU, lock, 6 GiB cap
$PY tools/compare/era5_nest/nest_throughput.py breeze \
    "$(ls -t $BENCH_ROOT/breeze-mc3e/timing-*.json | head -1)" --case mc3e
```

`breeze_run.sh` sets `HOME=$BENCH_ROOT/home` for the Julia process. The CDS
client and the era5cli credential bootstrap therefore never read or write
the real home directory.

## Clean-card phase

```bash
PY=$PY WOOF="$PY -m woof" PYTHONPATH=$PWD tools/compare/era5_nest/run_timing.sh
```

The script writes `results-<stamp>/` under `$BENCH_ROOT`, containing:

- fetch logs and byte counts;
- `woof go` logs and their memory samples;
- per-case throughput JSON;
- the Breeze timing JSON;
- the published figures with their bandwidth-scaled estimates.

## Tests

`tests/test_compare_era5_nest.py` (CPU) covers the case geometry, the
Oceananigans column, the eta mapping, the config patches and the throughput
arithmetic.
