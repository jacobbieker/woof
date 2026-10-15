# woof vs ClimaAtmos.jl: idealized benchmarks

This harness runs ClimaAtmos.jl's own idealized configs and woof's versions
of the same setups, then scores both against the same analytic or reference
solutions and tabulates their throughput.

| case | ClimaAtmos config (unchanged unless noted) | woof equivalent (`woof_driver.py`) | yardstick |
|---|---|---|---|
| Schär mountain wave | `config/model_configs/plane_schar_mountain_float32_test.yml`: 2-D periodic plane, 100 km, 100 x 100 elements (300 unique x nodes, dx ≈ 333 m), 21 km top, tanh-stretched levels with dz_bottom = 10 m, dt 0.7 s, 24 h, Float32 | `--case schar`: nx = 300, nz = 100 on ClimaAtmos's own face heights (`schar_z_faces.json`, mapped to woof's dry-hydrostatic eta), the same Schär profile (25 m, a = 5 km, λ = 4 km), U = 10 m/s, N = 0.01 1/s, T_sfc = 288 K, p_sfc = 101325 Pa, w-only sin² Rayleigh layer above 13 km at 0.1 1/s (`damp_opt=3`), w = 0 above the kinematic surface w at t = 0 (`--w-init wrf-fill` for WRF's fill_w start), dt 2 s | ClimaAtmos's linear steady-state solution, ported to `analysis.schar_steady_state` and checked against ClimaAtmos's `wapredicted` |
| density current, ClimaAtmos IC | `plane_density_current_test.yml`: 51.2 x 6.4 km, 80 x 90 elements (240 x 90, dx = 213 m, dz = 71 m), dt 0.3 s, 900 s, default 4th-order hyperdiffusion; θ′ = −15 K bubble at z = 2 km | `--case dc --ic clima`: 240 x 90, uniform dz, constant K = 75 m²/s, dt 1 s | model vs model (there is no published reference for this IC) |
| density current, Straka IC | the same config, run with `--straka-ic`: Straka et al. (1993) bubble (T′ = −15 K at z = 3 km, so θ′ = T′/π_b) | `--case dc --ic straka` (240 x 90), plus woof's gated 100 m grid (512 x 64, dt 0.5 s) | Straka et al. (1993) converged values: θ′_min ≈ −9.77 K, front ≈ 15.54 km at 900 s |

There is a caveat on the density currents. Straka's reference uses a
constant viscosity of 75 m²/s for momentum and heat. ClimaAtmos has no such
option for momentum: `constant_horizontal_diffusion` is horizontal and
applies to scalars only. The ClimaAtmos runs therefore use the config's
default hyperdiffusion. Expect a sharper and colder head than the reference
in ClimaAtmos, while woof runs the published K = 75.

## Environment

Everything for ClimaAtmos lives under `BENCH_ROOT`
(`/home/jacob/wf-scratch/idealized-benchmark` by default; see `env.sh`):

```
julia-1.11.7/         Julia tarball (julialang-s3 1.11.7 linux x86_64)
depot/                JULIA_DEPOT_PATH (packages, precompile cache)
ClimaAtmos.jl/        git clone https://github.com/CliMA/ClimaAtmos.jl (dda668d3d, v0.42.14)
env/                  run project: ClimaAtmos (dev path), CUDA, ClimaComms, ClimaCore, NCDatasets, YAML
runs/                 RUNS_DIR: one directory per run (result.json + fields)
tmp/                  TMPDIR (/tmp is RAM-backed on the benchmark host)
```

To recreate it:

```bash
B=/home/jacob/wf-scratch/idealized-benchmark; mkdir -p $B/tmp $B/env
curl -fsSL https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-1.11.7-linux-x86_64.tar.gz | tar -xz -C $B
git clone https://github.com/CliMA/ClimaAtmos.jl $B/ClimaAtmos.jl
cat > $B/env/setup.jl <<'EOF'
import Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = joinpath(@__DIR__, "..", "ClimaAtmos.jl"))
Pkg.add(["CUDA", "ClimaComms", "ClimaCore", "NCDatasets", "YAML"])
Pkg.instantiate(); Pkg.precompile()
EOF
TMPDIR=$B/tmp JULIA_DEPOT_PATH=$B/depot $B/julia-1.11.7/bin/julia $B/env/setup.jl   # about 30 min
```

For woof, use any Python environment that has woof's GPU dependencies. Set
`WOOF_PY` to it. `env.sh` puts the woof checkout on `PYTHONPATH`.

On the shared card, every GPU run takes `flock /tmp/woof-gpu-test.lock`.
`env.sh` sets `JULIA_CUDA_HARD_MEMORY_LIMIT=6GiB` and
`CUPY_GPU_MEMORY_LIMIT=6442450944`.

## Commands

```bash
cd tools/compare/climaatmos_idealized
export WOOF_PY=/path/to/python            # woof env

# small end-to-end runs (accuracy mode), then summary tables
bash run_e2e_small.sh                     # Schär 2 h (SCHAR_HOURS) + density currents 900 s
#   SKIP_CA=1 / SKIP_WOOF=1 run one side only

# clean-card phase: throughput (timing mode); ACCURACY=1 adds 24 h Schär + 900 s DCs
RUNS_DIR=/path/to/clean JULIA_CUDA_HARD_MEMORY_LIMIT=none CUPY_GPU_MEMORY_LIMIT=0 \
    REPEATS=3 ACCURACY=1 bash run_timing.sh

# score / tabulate any runs directory
$WOOF_PY -m tools.compare.climaatmos_idealized.analysis summarize $RUNS_DIR
```

You can also run each driver on its own:

```bash
source env.sh
$JULIA --project=$CA_ENV ca_driver.jl --config $CA_REPO/config/model_configs/plane_schar_mountain_float32_test.yml \
    --out $RUNS_DIR/ca_schar --set t_end=2hours --warm-steps 20 --timed-steps 200 [--mode timing] [--straka-ic]
$JULIA --project=$CA_ENV ca_driver.jl --batch runs.txt     # one session, one arg list per line
$WOOF_PY -m tools.compare.climaatmos_idealized.woof_driver --case schar --out $RUNS_DIR/woof_schar --run-seconds 7200
$JULIA --project=$CA_ENV make_schar_z_faces.jl             # regenerates schar_z_faces.json
```

## What is measured

- **Throughput.** Each run does a warm-up block of steps first. That block
  covers JIT and kernel compilation and allocator growth, and it is
  excluded. The driver then times a block of steps between two device
  synchronisations. In timing mode no diagnostics or output are written.
  The run reports s/step, point-steps/s and simulated seconds per wall
  second (`dt / (s/step)`).
  - Points are unique degrees of freedom. For ClimaAtmos that is
    `x_elem * nh_poly` (periodic GLL nodes) times the number of levels.
    `stored_point_steps_per_sec` also counts the duplicated element-edge
    nodes ClimaAtmos stores, which is (nh_poly+1)/nh_poly = 4/3 more.
  - Each model runs at its own stable dt: ClimaAtmos IMEX ARS343 at the
    config dt, and woof's split-explicit RK3 with 4 acoustic substeps.
    `run_timing.sh` also runs woof Schär at ClimaAtmos's dt.
- **Schär accuracy.** The error of w against the linear solution, on all
  interior w points below the 13 km sponge base. Reported as relative L2,
  correlation, RMSE and maximum error. For ClimaAtmos runs the summary also
  reports how far the Python port is from ClimaAtmos's own `wapredicted`
  (`port_vs_CA_prediction`), and ClimaAtmos's error against its own
  prediction.
- **Density current.** θ′_min over the domain and the front position: the
  rightmost x at which θ′ on the lowest level crosses −1 K, measured from
  the bubble centre. Also the left/right front asymmetry, and the
  differences from Straka's values for the Straka IC.

## Files

`ca_driver.jl` (ClimaAtmos runner), `woof_driver.py` (woof runner),
`analysis.py` (analytic solution, metrics, tables; CPU tests in
`tests/test_compare_climaatmos_idealized.py`), `env.sh`, `common.sh`,
`run_e2e_small.sh`, `run_timing.sh`, `make_schar_z_faces.jl`,
`schar_z_faces.json`.
