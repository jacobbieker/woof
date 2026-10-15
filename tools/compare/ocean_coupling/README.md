# Small regional atmosphere–ocean coupling experiment (NumericalEarth + Breeze + Oceananigans)

This directory backs `docs/comparisons/clima-numericalearth/ocean-coupling.md`. It runs a
small coupled atmosphere–ocean experiment with the released Julia packages and records which
regional configurations work today and which do not. Everything runs on the CPU and nothing is
downloaded: the parent atmosphere and the external ocean state are synthetic.

| File | What it is |
|---|---|
| `coupled_regional_box.jl` | The experiment: six configurations plus two `_unpatched` controls (eight cases), each wrapped in try/catch so a failure is logged, not fatal |
| `Project.toml`, `Manifest.toml` | Pinned environment: NumericalEarth 0.8.2, Breeze 0.11.3, Oceananigans 0.113.6, ClimaSeaIce 0.5.12, Julia 1.12.7 |
| `four_oceans_slab_check.jl` | One-off: the published `breeze_over_four_oceans.jl` slab configuration, unpatched, 12 steps with a progress line after each |
| `run.sh` | Sets the depot and TMPDIR, then runs the experiment script |
| `results/summary.csv`, `results/*.log`, `results/four_oceans_slab_check.log` | The measured run that the comparison document quotes |

## Cases

| Case | Atmosphere | Ocean | Question it answers |
|---|---|---|---|
| `periodic_slab` | Breeze anelastic LES, 2-D periodic, 32 × 32 | `SlabOcean`, 10 m | Does the published example's coupling path run at all? |
| `periodic_catke` | as above | Oceananigans hydrostatic, 20 levels over 50 m, CATKE | The same, with a real ocean column |
| `lam_slab` | Breeze compressible limited-area model: `nested_atmosphere_model` driven through open lateral boundaries by a synthetic `PrescribedAtmosphere` parent, 16 × 16 × 16 on a 2° box | `SlabOcean` on the same horizontal grid | Can the limited-area atmosphere drive an interactive ocean surface? |
| `lam_catke` | as above | Oceananigans hydrostatic, 16 × 16 × 10 over 100 m, closed walls, CATKE | Limited-area atmosphere over a 3-D regional ocean |
| `ocean_open` | none | as above, with open lateral boundaries from an external `FieldTimeSeries` (T/S Dirichlet, baroclinic u/v `PerturbationAdvection`, barotropic Flather, η Chapman) | Does a regional ocean with GLORYS-style open boundaries run? |
| `lam_catke_open` | limited-area, as above | the open-boundary ocean | Regional atmosphere over a regional open-boundary ocean |
| `periodic_slab_unpatched`, `lam_slab_unpatched` | as `periodic_slab` / `lam_slab` | as above | The same cases *without* the halo workaround in the script, to show the upstream defect (NumericalEarth issue #754). Any coupled case accepts the `_unpatched` suffix. |

For each case the script records the number of steps, the wall time of the first step
(which carries the JIT compilation) and of the rest, the start and end area-mean SST, the
time-mean turbulent heat flux (sensible + latent), and, for the closed cases, the relative
residual of the ocean heat budget: the change in ocean heat content against minus the time
integral of the temperature flux the coupler handed to the ocean.

## Exact commands

All installs live under one scratch directory, `$OC`; any writable directory will do (we used
`/home/jacob/wf-scratch/ocean-coupling`). `$REPO` is the woof checkout. Nothing is installed
system-wide. The Manifest was resolved with Julia 1.12.7; use that version to get the pinned
environment, even though `Project.toml` admits 1.11.9 and later.

```bash
OC=$HOME/scratch/ocean-coupling      # any writable directory
REPO=$HOME/woof                       # the woof checkout
mkdir -p $OC && cd $OC

# 1. Julia 1.12.7 (official tarball; no juliaup needed).
curl -sL https://julialang-s3.julialang.org/bin/linux/x64/1.12/julia-1.12.7-linux-x86_64.tar.gz | tar xz

# 2. Instantiate the pinned environment into a private depot (about 5 minutes of
#    precompilation on 32 cores; the depot is about 0.8 GB and Julia itself 1.1 GB).
JULIA_DEPOT_PATH=$OC/depot TMPDIR=$OC/tmp \
  $OC/julia-1.12.7/bin/julia --project=$REPO/tools/compare/ocean_coupling -e 'using Pkg; Pkg.instantiate()'

# 3. Run all eight cases (or name a subset), CPU only.
OC=$OC $REPO/tools/compare/ocean_coupling/run.sh
OC=$OC $REPO/tools/compare/ocean_coupling/run.sh lam_catke ocean_open

# 4. The one-off check of the published four-oceans slab configuration.
JULIA_DEPOT_PATH=$OC/depot TMPDIR=$OC/tmp JULIA_NUM_THREADS=4 \
  $OC/julia-1.12.7/bin/julia --project=$REPO/tools/compare/ocean_coupling \
  $REPO/tools/compare/ocean_coupling/four_oceans_slab_check.jl
```

`run.sh` writes to `$OC/results` unless `OC_OUT` is set. The committed `results/` is a copy
of one such run.

The GPU was not used. Running on the GPU would mean adding `CUDA` to the environment and
replacing `CPU()` with `GPU()` in the grid constructors; on the shared card that run would
have to sit under the lock (`flock /tmp/woof-gpu-test.lock`) with
`JULIA_CUDA_HARD_MEMORY_LIMIT=6GiB`.

## Measured result (CPU, 4 threads, 2026-10-15)

`results/summary.csv` is the default eight-case run. `results/summary_unpatched_extra.csv` adds
the remaining `_unpatched` variants. `results/run.log` holds the progress lines.

- **All six configurations ran end to end** with the halo workaround in the script. That
  includes the limited-area atmosphere over a slab ocean, over a closed CATKE ocean, and over an
  open-boundary CATKE ocean.
- **Without the workaround**, only the periodic slab configuration fails: NaN after step 1, both
  here and in the published four-oceans slab configuration (`four_oceans_slab_check.log`). Every
  other configuration also runs unpatched.
- **Ocean heat budget** (closed oceans): it closes to 1e-10 or better for the slab and periodic
  cases, and to 0.9 % for `lam_catke`.

The comparison document discusses the numbers.
