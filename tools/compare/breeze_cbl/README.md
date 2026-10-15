# woof vs Breeze: the FastEddy dry convective boundary layer

Harness for running one LES case in woof and in
[Breeze.jl](https://github.com/NumericalEarth/Breeze.jl) (NumericalEarth's GPU
atmosphere model on Oceananigans) at matched grid, time step, precision and
forcing, then comparing throughput and boundary-layer statistics. The write-up
is [docs/comparisons/clima-numericalearth/les-benchmark.md](../../../docs/comparisons/clima-numericalearth/les-benchmark.md).

The case is the dry CBL of Sauer & Muñoz-Esparza (2020, JAMES,
doi:10.1029/2020MS002100, §4.2). It is the case the Breeze CI benchmark uses
([BreezeBenchmarks](https://github.com/NumericalEarth/BreezeBenchmarks); Breeze
`benchmarking/src/convective_boundary_layer.jl`): a 12 × 12 × 3 km periodic box,
Ug = 9 m/s, f-plane at 33.5° N, θ = 309 K to 600 m and +4 K/km above, a
surface heat flux of 0.35 K m/s, ±0.25 K noise below 400 m, Δt = 0.5 s and
Float32. `case.py` is the only place these parameters are defined.

| file | role |
|---|---|
| `case.py` | case constants, drag coefficient, zero-order-jump bulk model, FastEddy figure readings |
| `run_woof.py` | woof driver; `--mode timing` or `--mode simulate` |
| `run_breeze.jl` | Breeze driver with the same modes and output schema |
| `julia/Project.toml`, `julia/Manifest.toml` | pinned Julia environment: Breeze 0.12.0, Oceananigans 0.113.6, CUDA.jl 6.4.2, JSON 1.10 |
| `analyze.py` | throughput table, profile and z_i statistics, figures, `summary.{md,json}` |
| `run_e2e_small.sh` | short correctness runs at 128² × 64 for 2 simulated hours, then analysis |
| `run_timing.sh` | full-size, interleaved timing for a clean card, plus optional long profile runs |

## Install (done once; nothing is installed system-wide)

Julia and its package depot live under `/home/jacob/wf-scratch/les-benchmark/`:

```bash
S=/home/jacob/wf-scratch/les-benchmark
curl -fsSL https://julialang-s3.julialang.org/bin/linux/x64/1.12/julia-1.12.7-linux-x86_64.tar.gz -o $S/tmp/julia.tar.gz
echo "4e7e9e776634d24835250de67cde39b0d4af15bc432eb20697e6be6c28ea69e8  $S/tmp/julia.tar.gz" | sha256sum -c
tar -C $S -xzf $S/tmp/julia.tar.gz
export JULIA_DEPOT_PATH=$S/depot
$S/julia-1.12.7/bin/julia --project=tools/compare/breeze_cbl/julia \
    -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' < /dev/null
```

CUDA.jl downloads its own CUDA toolkit as artifacts into the depot. Run the
precompile step with stdin set to `/dev/null`: when stdin was inherited from a
background shell, the precompile of the `REPL` stdlib hung, sleeping, for 10
minutes.

## Run

The examples below assume `PYTHONPATH` is the repository root and `python` has
woof's runtime installed (CuPy).

```bash
# one woof run: 2 simulated hours, profiles every 5 min
python tools/compare/breeze_cbl/run_woof.py --mode simulate --size 128x128x64 \
    --km-opt 3 --hours 2 --out OUT/woof_km3
# one Breeze run with the same operators as woof (see "Variants")
JULIA_DEPOT_PATH=$S/depot $S/julia-1.12.7/bin/julia --project=tools/compare/breeze_cbl/julia \
    tools/compare/breeze_cbl/run_breeze.jl --mode simulate --size 128x128x64 \
    --dynamics compressible_splitexplicit --advection Upwind53 --closure Smagorinsky \
    --drag bulk --substeps 12 --distribution wrf --forward-weight 0.55 --hours 2 --out OUT/breeze_se
# analysis
python tools/compare/breeze_cbl/analyze.py --profiles OUT/woof_km3 OUT/breeze_se --out OUT/report

# small end-to-end (shared card: holds the GPU lock and caps memory at 6 GiB;
# REUSE=1 keeps finished simulate outputs; report in $SCRATCH/out/small/report)
tools/compare/breeze_cbl/run_e2e_small.sh
# full-size timing on a clean card (see the header of the script for its knobs)
PY=python tools/compare/breeze_cbl/run_timing.sh
PY=python PROFILES=1 tools/compare/breeze_cbl/run_timing.sh   # adds 4 h profile runs at 256^2 x 128
```

Timing mode builds the case, runs `--warmup` steps that are not timed (Julia
JIT, NVRTC compilation, first-touch allocation), then `--reps` blocks of
`--steps` steps, each closed by a device synchronize. Nothing is written inside
a timed block. The reported figures are median s/step,
**Mpts·steps/s** (grid points × steps per wall-second / 10⁶) and simulated
seconds per wall-second.

## Variants (`run_timing.sh`)

| name | model | dynamics | advection | closure | surface drag | acoustic substeps |
|---|---|---|---|---|---|---|
| `woof_km3` | woof | compressible, split-explicit (WRF ARW) | WRF upwind, 5th order horizontal and 3rd vertical | `km_opt=3` Smagorinsky, c_s 0.18, Pr 1/3 | bulk Cd | ns = 12, WRF distribution (1, 6, 12) |
| `woof_km2` | woof | same | same | `km_opt=2` 1.5-order TKE, c_k 0.10 | bulk Cd | same |
| `breeze_anelastic_weno5_ci` | Breeze | anelastic | WENO5 | none (implicit LES) | constant u* = 0.4 | none |
| `breeze_se_weno5_ci` | Breeze | compressible, split-explicit (WS-RK3) | WENO5 | none | constant u* | 12, proportional (4, 6, 12), ω = 0.65 |
| `breeze_se_matched` | Breeze | compressible, split-explicit | `Upwind53` (UpwindBiased 5/5/3) | Lilly–Smagorinsky with C 0.18, Cb 1.5, Pr 1/3 (the WRF km_opt=3 form) | bulk Cd | 12, WRF distribution, ω = 0.55 (WRF epssm 0.1) |
| `breeze_se_weno5_smag` | Breeze | compressible, split-explicit | WENO5 | Lilly–Smagorinsky as above | bulk Cd | same as `_matched` |
| `breeze_anelastic_matched` | Breeze | anelastic | `Upwind53` | Lilly–Smagorinsky as above | bulk Cd | none |

The `_ci` rows are the exact BreezeBenchmarks CI configurations, so their
numbers can be compared with the published dashboard. The `_matched` rows use
woof's set of operators. The bulk drag is Cd = (κ / ln(z₁/z₀))², with z₀ =
0.05 m (FastEddy's value) and z₁ = Δz/2. woof's LES lower boundary (WRF
`isfflx=0`) offers only this form of drag.

## What the harness adds to woof

woof has no namelist control for a geostrophic pressure gradient in idealized
cases (WRF's `pert_coriolis` is not ported). `run_woof.py` therefore enables
woof's WRF Coriolis kernel on the full wind (`set_map_coriolis(f=...)`) and adds
the constant force (−f Vg, +f Ug)·Δt to the velocity once per full step,
operator-split. This costs one CuPy add per step, and the timed loop includes
it. The script also builds an eta ladder whose base-state layers are uniform in
height, to match Breeze's z grid. Everything else is
`woof.verify.cases.convective_boundary_layer.make_config`, unchanged.

## Shared-card note

Every number produced while this harness was being developed came from an RTX
A6000 that other jobs were also using, under `flock /tmp/woof-gpu-test.lock`,
with `CUPY_GPU_MEMORY_LIMIT=6442450944` and `JULIA_CUDA_HARD_MEMORY_LIMIT=6GiB`.
The CUDA.jl variable name was checked against CUDACore's
`src/memory.jl:223`. Those timings show only that the harness runs. Use the
output of `run_timing.sh` on an otherwise idle card for any speed comparison.
