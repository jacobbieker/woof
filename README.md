# Recast WOOF

**Weather Observations and Open Forecasting: a GPU-native weather model you run on your own card.**

> **This is a fork.** WOOF is written and maintained by Recast Systems at
> [github.com/recastsystems/woof](https://github.com/recastsystems/woof),
> which is the original and main code, and the place to start. This fork,
> [github.com/jacobbieker/woof](https://github.com/jacobbieker/woof), merges
> each upstream release and adds the work listed under
> [This fork](#this-fork) below, most notably a backend that runs on
> Apple-silicon GPUs. On PyPI, Recast's release is `recast-woof`; this fork's
> Apple-silicon build is `metal-woof`, which is unofficial and not supported
> by Recast Systems.

WOOF is a regional weather model with WRF-class physics that runs on one
NVIDIA GPU, with its own preprocessing, data assimilation and plots. One
install carries two models:

| Model | Command | Status |
| --- | --- | --- |
| Regional (WRF-ARW class, nests, DA) | `woof` | Production |
| Hex (MPAS-Atmosphere v8.4.2 on the GPU, regional and global) | `woof hex`, `woof hex global` | Preview |

Preview means the model is in the package, documented and tested on CPU, and
its command says preview each time it runs; its config and checkpoint formats
may change in a minor release. The major-version promise covers the regional
model.

Production means the regional commands and their config and checkpoint
formats are covered by that promise. It is not a statement about forecast
accuracy; see Science and limits below.

## Install

Linux on x86_64, Python 3.11 or newer, an NVIDIA GPU and driver. WOOF has
no Windows build of the native tools yet.

With Pixi (Python and the Rust/build toolchain managed in one env):

```bash
pixi install
pixi run setup
pixi run doctor
# or: pixi run doctor-explain
```

The default Pixi environment installs this checkout in editable mode with the
`all-cu13` extra. CUDA 12 is an explicit choice:

```bash
pixi install -e cuda12
```

On a source checkout, `pixi run setup` builds the native Rust tools locally.
`woof fetch-bridges` may report that no bundle pins are present; that is normal
outside a published wheel install.

```bash
python3 -m venv ~/woof-env
~/woof-env/bin/python -m pip install 'recast-woof[all-cu13]'
~/woof-env/bin/woof fetch-tables
~/woof-env/bin/woof fetch-geog --datasets wrf
~/woof-env/bin/woof doctor
```

For a driver that serves only CUDA 12, install `all-cu12` explicitly
instead; `all` and `gpu` mean the CUDA 13 pair. The hex model runs on either extra:
`all-cu12` works on a CUDA 12 or CUDA 13 driver, `all-cu13` needs a CUDA 13
driver, and `woof hex doctor` names the one for your driver. Install one
CuPy/CUDA major per environment.
The Linux wheel carries the native tools (GRIB and NetCDF decoding, the
WPS-equivalent preprocessor, the renderer).

`fetch-geog` downloads the global geography every forecast builds its
terrain, land use and soil from: about 1.3 GB, 17 GB unpacked, once per
computer. `fetch-tables` adds two Thompson microphysics tables (about 314 MiB).
Reference tables and map assets arrive with the `recast-woof-data` package.
`woof doctor` reports each remedy as a command or a `#` comment explaining
the manual step.

For a developer installation from a clone of this repository, use
`bash install.sh`: it installs the checkout's `recast-woof-data` companion and
the package with its extras, builds the native tools (this needs Rust and a C
toolchain) and ends with `woof doctor`. The explicit `bash install.sh` form
also works when an extracted source archive has lost its executable bits; use
it if `./install.sh` reports `Permission denied`.

## Run a forecast

```bash
woof domain --point 35.3,-97.5 --source hrrr --cycle latest --hours 3 \
     --root-dx 3 --point-extent-km 300 --out forecast.toml
woof go forecast.toml --dry-run
woof go forecast.toml
```

`--dry-run` reviews the route, sizes it for your card and names anything
missing; the second line fetches, prepares, runs and draws. Pictures are drawn
as each history frame lands, beside the running forecast.

Coupled hydro is opt-in. `woof go forecast.toml --hydro on` couples NextGen
hydro wherever a hydrofabric covers the domain; a forecast without it has no
hydro.

Compare two runs, any product, as run A minus run B:

```bash
woof render --diff runA/ runB/ --out diff/
```

The pair must share the grid and the valid times; a mismatch is refused by
name.

## Documentation

- `docs/public/`: the CLI manual, physics, hardware guidance and data sources.
  The [code verification record](docs/public/VERIFICATION.md) documents
  historical WRF v4.6.1 comparisons and their configuration and coverage
  limits.
- [Energy forecasts](docs/energy-forecasts.md): `woof energy` builds 50-100 m
  forecasts along power lines, substations and renewable sites from
  OpenStreetMap or your own grid data, with line rating, icing and power
  products.
- [Stage contract](docs/public/PIPELINE-STAGES.md) for running preprocessing,
  the simulation and rendering on your own terms.
- [Smoke, dust and air quality](docs/public/SMOKE-AND-AIR-QUALITY.md): wildfire
  smoke, dust, sea salt, carbon and CAMS air-quality species are an opt-in
  product, off unless a configuration names `chem_sets`.
- `components/hex/`: the hex model's manual, tests and tools.
- `woof --help-all` lists every command; `woof hex --help` lists the hex
  model's commands.

## Science and limits

WOOF implements a WRF-ARW-class regional model and WRF-derived physics on the
GPU. Comparisons against WRF Fortran, ULP and byte-identity checks, and the
matched-run and ensemble consistency comparisons are **code verification**:
evidence that the tested code computes its intended equations. Each result
belongs to the build, reference version and configuration it tested. None
establishes general equivalence with WRF or skill for every combination of
physics, input data and hardware.

**Solution verification** estimates numerical error in a particular run.
Routine numerical-error budgets for weather forecasts remain a gap.
**Validation** compares model results with observations. Meteorology calls
that scoring forecast verification; in computational-science V&V terms it is
validation. Matching WRF is code verification and never validates a model.
WOOF can inherit WRF's published validation record only to the extent the
two are statistically indistinguishable for the configuration in question.

Published observation evidence is limited to specific runs and variables,
including the ASOS snow/soil comparison, the station/ceilometer fog-suite
comparison and an MRMS model-top comparison. The observation scoreboard is
in development. These do not establish a broad validation record across
regions, seasons or forecast lengths. The
[evidence page](docs/public/VERIFICATION.md) documents historical WRF v4.6.1
code comparisons and their configuration and coverage limits.

The hex model is a GPU port of MPAS-Atmosphere v8.4.2, with MPAS-A's own
physics suites as its default and this engine's physics selectable; it is not
MPAS and is not endorsed by UCAR. Use official
meteorological services for forecasts and warnings.

WOOF continues an engine released before under another name: WOOF 1.0 is
that engine's release 2.8.0 and the fixes made since, under the WOOF name,
with the hex model inside it. Settings spelled `GPUWM_*` keep
working beside the new `WOOF_*` spelling.

## This fork

Everything above describes Recast WOOF, and is kept as upstream wrote it.
This section is the fork's own. The fork carries upstream WOOF 1.1.6 and adds
the work below on top of it. It does not change upstream's defaults: each new
option is off until selected, and a configuration that selects none of them
runs as upstream's does. The section is grouped by area, and where
something is partial it says so. Every merge into the fork's `main` is released as a
new `metal-woof` build (`metal-vX.Y.Z.N`). The fork's own changes are recorded
in [FORK-CHANGELOG.md](FORK-CHANGELOG.md).

Upstream now covers several things this section once listed as the fork's:
storm-following nests and storm-following hex regions (with HAFS starts and
storm merge), coupled ocean waves and storm surge, a GPU mixed-layer ocean
and sea ice, coupled hydrology over a NextGen hydrofabric, HRRR-Smoke and
aerosol transport, and its own wiring of `use_theta_m`, FARMS, MAD-WRF and
Deng shallow cumulus. Where both existed, upstream's keys are the ones to use.

### What it adds

- **An Apple-silicon backend (MLX/Metal).** `WOOF_BACKEND=mlx` runs the GPU
  code on a Mac. The CUDA kernels are not rewritten by hand: each `.cu` unit
  is translated to Metal when it first loads, and `import cupy` resolves to a
  CuPy-compatible layer over MLX. CUDA runs exactly as before. Under strict
  numerics, binary32 `pow` on Metal is WOOF's own correctly rounded `powf`,
  so it carries no vendor code. Numerics come in two modes: strict
  (`GPUWM_WRF_EXACT=1`, bit-exact software binary64, with exact
  df64-seeded division and square root) and fast (the default, binary64
  libm computed in df64). The WRF regional model runs on a Mac, and so
  does the hex model, including its MPAS v8.4.2 suite route (R2); the hex
  helper libraries are built from source on macOS. A parity ledger
  (`tests/test_backend_parity_ledger.py`) records every MLX-against-CUDA
  gap by name. Design and details:
  [docs/dev/mlx-backend.md](docs/dev/mlx-backend.md); limits are under
  "Limitations of the MLX backend" below.
- **A host (CPU) backend.** `WOOF_BACKEND=host` runs the same GPU code on
  any CPU. Each `.cu` unit is compiled by the host C++ compiler, and
  `import cupy` resolves to a CuPy-compatible layer over NumPy. It is never
  selected by default. Design and measurements:
  [docs/dev/host-backend.md](docs/dev/host-backend.md).
- **WRF-Chem gas and aerosol chemistry,** each piece held word for word to
  WRF's chem code (v4.7.1; the earlier pieces v4.6.1, being re-pinned): RADM2
  gas phase, MADE/SORGAM aerosol dynamics, SORGAM initial and boundary
  values, equilibrium, Wesely and SORGAM dry deposition, and Fast-TUV
  photolysis (`phot_opt = 3`). Anthropogenic emissions (`emiss_opt = 3`)
  take EDGAR v8.1 AP inventory frames (CC-BY-4.0), speciated onto RADM2.
  Namelists with `chem_opt = 1, 2` import. Aerosol feedback on radiation
  (`aer_ra_feedback`, legacy RRTMG only) uses optical properties, and
  multimode aerosol activation uses a solver, both written clean-room from
  the published methods and checked black-box. `so2so4_qc_threshold` is an
  opt-in guard against nucleation bursts from round-off-sized cloud water;
  by default WRF's any-cloud rule applies. Where a WRF routine is not freely
  licensed, the default build uses a version written from the published
  method. The exact routine is either an explicit opt-in (the PNNL files
  under `optional/wrfchem_pnnl`: Abdul-Razzak-Ghan `activate` and the
  `mixactivate` column driver, bitwise against WRF v4.7.1,
  `WOOF_ENABLE_WRFCHEM_PNNL=1`, never in the wheels), or is generated from
  your own WRF tree (`WOOF_WRF_SOURCE`). Not ported yet: WRF's own optics
  and the other `chem_opt` packages, which are refused by name. Plan and
  licence table:
  [docs/chem-coupled-aerosol-roadmap.md](docs/chem-coupled-aerosol-roadmap.md).
- **Chemistry on the hex mesh.** Hex runs select a WRF-Chem species set with
  `chem_opt` and the `&chem` options under their WRF names. The WRF column
  processes run unchanged on hex columns, bitwise per column against the
  regional model: dust, sea salt, dry deposition, vertical mixing, settling,
  ageing, wet deposition and optics. RADM2 and RADM2SORG (`chem_opt = 1, 2`)
  run there with their emissions, input chemistry, regional boundary
  values, fire smoke and plume rise, aerosol feedback on radiation and
  Thompson aerosol coupling, also on the MPAS v8.4.2 suite route. An MICM
  gas-phase option (NCAR MUSICA's Rosenbrock solver) is word-exact to its
  oracle; its photolysis is the `j_max` form by default, or a port of NCAR
  TUV-x v0.17.0 under `"photolysis": "tuvx"` (input files are fetched,
  hash-verified and cached on first use). Not every combination of these has
  been run as a live forecast.
- **WRF-Hydro and NEMO coupling, as explicit opt-ins.** Both are under
  `optional/` and are never in the wheels, because their licences (UCAR
  WRF-Hydro, CeCILL NEMO) are not ones the default build carries.
  `WOOF_ENABLE_WRF_HYDRO=1` and `WOOF_ENABLE_NEMO=1` turn them on.
  - WRF-Hydro: Noah and Noah-MP under `WRF_HYDRO`, the transfer between the
    land-model grid and the routing grid, subsurface and overland routing,
    Muskingum-Cunge and diffusive-wave channels, the groundwater bucket,
    level-pool reservoirs and `hydro.namelist`. Each piece is bitwise against
    WRF 4.7.1's `hydro/`.
    Also ported: the 2-D groundwater model (`GWBASESWCRT = 3`),
    `rt_option = 2`, reservoir `lake_option 3` and Route_Link reaches. It is
    wired into the forecast route (`--wrf-hydro`, `[wrf_hydro]`), and into
    hex through a conservative cell-to-routing-grid exchange. Unported
    `hydro.namelist` options are refused by name.
  - NEMO 5.0.2: the equation of state, TKE mixing, vertical and lateral
    tracer diffusion, FCT advection and solar penetration. A coupled column
    ocean is driven by WRF's ocean-coupler exchange (SST only for now); a
    step splits at the coupling time on adaptive clocks.
  - Details: [docs/coupling/WRF-HYDRO-NEMO.md](docs/coupling/WRF-HYDRO-NEMO.md).
- **Noah-MP follows WRF 4.7.1's submodule (e5c0859), at the user's
  request.** This is not a default change: the default
  `noahmp_snow_age = "wrf_470"` already ages snow at night as e5c0859 does,
  the glacier energy integrals already use the glacier column's own heat
  capacity, and e5c0859's other edits are dead in WRF or arithmetic-neutral.
  `noahmp_snow_age = "wrf_461"` keeps WRF 4.6.1's daytime-only aging.
- **MPAS-Ocean, MPAS-Seaice and MALI land ice,** ported a piece at a time,
  each word-exact to MPAS v8.4.2 on CUDA and Metal (RKIND 4 and 8 unless
  noted), behind the MPAS coupling spine (`woof hex ice verify` and
  friends). So far:
  - MPAS-Ocean: the equation of state and freezing temperature,
    diagnostics, pressure gradient and horizontal mixing, momentum and
    thickness tendencies, tracer tendencies, monotone tracer advection,
    implicit vertical mixing, vertical transport velocity, the
    split-explicit barotropic stage and whole split-explicit step, and
    forward mode (clock, cold start, N steps, bitwise restart);
  - MPAS-Seaice: variational EVP internal stress, the velocity solve and
    elastic subcycles, the pre-subcycle routines, ocean stress, and a
    forward time step (dynamics only: N steps, bitwise restart);
  - MALI: the shallow-ice velocity solver, masks and geometry diagnostics,
    thermal initialisation, the thermal solver and basal melt, thickness
    and tracer advection, calving (every option, and front restoration),
    subglacial hydrology and one forward-Euler time step.

  Limits: Albany's higher-order solvers are refused by name. Sea-ice
  advection and thermodynamics, ocean KPP and GM/Redi, and the coupler
  import and export are not ported. A few RKIND 8 cases need the LGPL libm
  opt-in below. These are ported pieces and steps, not coupled forecasts.
  Plan: [docs/mpas-coupled-roadmap.md](docs/mpas-coupled-roadmap.md).
- **Further WRF physics,** each held word for word to compiled WRF Fortran
  where the port says so:
  - SMS-3DTKE (`km_opt = 5`);
  - `do_avgflx_em`;
  - WRF v4.7.1's Fitch wind-farm scheme word for word, as
    `wind_farm_scheme = "fitch_wrf"` beside upstream's `fitch`, also on hex's
    unequal-area cells;
  - coupled fire spread and smoke;
  - Morrison with CESM aerosol (`mp_physics = 40`, `aercu_opt = 0`).
- **Hex (MPAS) parity with WRF's options.** The goal is for the hex model
  to accept the options WRF does, reusing WRF's column implementations
  rather than copies, and to refuse by name what it cannot yet do. Landed:
  - `woof go` runs hex with `[model] engine = "hex"` (regional route);
  - surface inputs on the generic column seam: CLM lake, SLUCM urban on
    per-cell fractions, WUDAPT LCZ classes, Noah and RUC mosaic, Pleim-Xiu
    and the slab with their soil inits, irrigation and `topo_wind = 1`
    ([docs/ruc-mosaic-and-clm-lake.md](docs/ruc-mosaic-and-clm-lake.md));
  - CCPP TKE-EDMF on the column seam;
  - diagnostics: `nwp_diagnostics` maxima, `p_lev`/`z_lev` levels, `tslist`
    station series, `prec_acc_dt` windows and `acc_phy_tend`, AFWA
    (icing and some radar/VIL/visibility options are refused), CLWRF, and
    RASM and surface-energy reductions. `afwa_ptype` keeps its intended
    logic rather than WRF's literal operator precedence; the module
    documents the difference;
  - `aer_opt = 1, 2`, `w_damping`, and grid and spectral analysis nudging;
  - stochastic physics: SPP (`spp_conv`, `spp_pbl`, `spp_lsm`) on the WOOF
    column route, WRF-Solar EPS (`multi_perturb`) on regional meshes, and
    WRF's inflow perturbation on the regional relax rings
    (`woof hex forecast --inflow-perturbation`, which takes its depth from
    the run's own boundary-layer height since hex has no parent model);
  - `woof resume` and `woof branch` for hex runs
    ([docs/branch-from-checkpoint.md](docs/branch-from-checkpoint.md));
  - GRIB2 and Earth2 export ([docs/grib2-export.md](docs/grib2-export.md));
  - a terrain hook (`--terrain-source`, `[static.highres]`);
  - an option matrix that runs generic-seam column options through a real
    hex forecast on CUDA and Metal.

  Limits: the MPAS v8.4.2 suite route (R2) takes a subset of these, and
  refuses the rest by name. WRF's adaptive time step (`--adaptive-timestep`)
  runs on the WOOF column route (R1) only, and is refused with `--nudging`.
  Accuracy against source WRF and MPAS over day-long runs is not
  yet measured, and several options have unit and oracle evidence but no
  live forecast yet.
- **Terrain: GEM-15 (experimental).** `terrain_source = "gem15-dtm"` or
  `"gem15-dsm"` (WRF `[static.highres]`; hex `--terrain`) read Google
  Research's 15 m global elevation models by range reads of verified COG
  blocks, against a pinned cell index. Neither is ever the default. The data
  is CC-BY-4.0, and its attribution travels with the static. It is a
  preprint dataset with offsets of several metres, covers roughly 60S to 84N,
  and excluded regions are refused by name
  ([docs/high-resolution-static-geography.md](docs/high-resolution-static-geography.md),
  [docs/public/HIGHRES-TERRAIN.md](docs/public/HIGHRES-TERRAIN.md)).
- **Height-dependent resolution:** a partial-depth two-way nest, fine near
  the ground and coarser aloft, on WRF and (dry dynamics only) on MPAS; and
  WRF's live vertical nesting (`vert_refine_method = 2`).  On WRF the band
  (`partial_nest = { k_top = ... }` on a child `[[domain]]`) runs on the
  idealized tree and on `woof go`'s real-data domain tree: map factors, the
  child's land surface rebuilt on its truncated column, radiation on an
  extended column to the model top, and two-way feedback below the
  interface plus WRF's surface feedback.  The interface must be a
  pure-pressure level (at or above `etac`), above 12 km under a parent with
  deep cumulus, and below a `damp_opt = 3` layer.  The real-data band needs
  a fixed clock (`woof domain ... --clock fixed`); its child takes
  `diff_opt = 1` with `km_opt = 4` and `emdiv = 0`; it writes no
  checkpoints and cannot resume.
- **More data sources:** the Met Office global and UKV models, DWD ICON
  (global and the machine-learning variant) and additional mapped routes.
- **Applied tooling:** wind-energy siting, hindcasts and power products,
  storm-track planning, and delete-after-commit backpressure and domain
  selection for upstream's Icechunk store export
  (`[output] icechunk_delete_after_commit`, `icechunk_domains`).
- **Performance work.** Metal: out-of-line compilation of the largest units,
  compile prefetch and persisted translations, fewer host syncs per step,
  and hardware fast paths that keep the same words (a strict 3 km forecast
  went from 47.7 to 42.1 s per forecast hour); CUDA: a memory-capped compile
  pool and fewer boundary launches. `WOOF_METAL_GPU_TIMES` times each kernel.
- **Continuous integration and releases:** GitHub Actions on Linux and
  macOS, with a self-hosted CUDA job. Hosted runners skip a few heavyweight
  tests by name, and main's full run is periodic. Every merge to `main`
  tags, writes release notes and publishes `metal-woof` to PyPI
  ([docs/dev/ci.md](docs/dev/ci.md)).

### Licensing and opt-ins

The default build and the published wheels (`metal-woof`, `recast-woof`)
carry none of the following. Each is off until selected:

- **LGPL glibc 2.39 libm** (`WOOF_GLIBC239_LIBM64`, `WOOF_GLIBC239_TRIG64`):
  glibc's binary64 words on the GPU and host. It lifts some MALI and
  MPAS-Ocean RKIND 8 refusals.
- **NVIDIA-derived `powf`**: CUDA's transcription for Metal strict numerics.
  The default is WOOF's own correctly rounded `powf`.
- **WRF-Hydro** (`WOOF_ENABLE_WRF_HYDRO=1`), **NEMO**
  (`WOOF_ENABLE_NEMO=1`) and **WRF-Chem PNNL** (`WOOF_ENABLE_WRFCHEM_PNNL=1`),
  under `optional/`.
- WRF routines that are not freely licensed come from your own WRF tree
  (`WOOF_WRF_SOURCE`).
- GEM-15 terrain is CC-BY-4.0 and needs its attribution.

### Install on a Mac

```bash
pip install metal-woof        # macOS 14 or newer on Apple silicon, Python 3.11+
```

The `metal-woof` wheel carries the macOS builds of the Rust bridges and the
MLX stack, and selects the MLX backend on its own. The hex helper libraries
(host preparation and physics tables) are built from source on macOS. It installs the same
`woof` package and commands as `recast-woof`; install one or the other, not
both. From a checkout, `pixi run -e mlx ...` uses the same backend.

### Limitations of the MLX backend

The Metal backend is younger than the CUDA one and narrower. Before relying
on it:

- **Apple silicon only,** macOS 14 or newer, one GPU. Domain size is bounded
  by the Mac's unified memory. There is no multi-GPU or peer transport.
- **Double precision is emulated.** Apple GPUs have no fp64. Under strict
  numerics (`GPUWM_WRF_EXACT=1`) `double` is a bit-exact software binary64:
  correct to the last bit, much slower than native fp64, and slow to
  compile. Fast numerics (the default) keeps the same binary64 words and
  exact `+ - *`, division and square root, but computes CUDA's binary64
  libm (exp, log, pow, sin, ...) in df64, a pair of binary32 values: about
  2^-46 relative instead of correctly rounded, 20-150x faster, and units
  that call it compile up to 100x sooner (`docs/dev/mlx-backend.md`,
  "Binary64: soft-float and df64"). Units built on woof's own glibc ports
  keep them, and their words, in both modes (CLM lake still compiles 5.6x
  sooner, from the faster division and square root).
- **The answers are not CUDA's bits everywhere.**
  - Metal never contracts a multiply-add, so where a CUDA result carries
    NVRTC's fused multiply-adds, Metal gives CUDA's `--fmad=false` words
    instead. Those families are pinned separately for Metal, each only after
    a launch-by-launch trace shows the difference is contraction.
  - Single-precision libm calls that are not woof's own glibc ports use
    Metal's `precise::` functions, a recorded vendor-libm difference.
  - NaN payloads can differ.
- **The first run compiles.** Every kernel unit is translated and compiled on
  first load. Pipelines are then kept on disk (`~/.cache/woof/metal-pipelines`),
  but a cold start can take minutes, and the largest units (CLM lake's step
  kernel, for one) about twenty.
- **The hex (MPAS) model runs on Metal by default** on an Apple GPU
  (`WOOF_HEX_METAL=0` opts out; on a virtual Mac it refuses). Its words differ from CUDA's where binary32 libm does (`precise::`
  against libdevice), so a forecast drifts about as far from CUDA as CUDA's
  own run does with contraction off. Within hex:
  - the MPAS v8.4.2 suite route (R2) runs end to end on a Mac (a new
    12 km geometry for 1 h took about 10 minutes of wall time), but some
    kernel units are refused by name: TEMPO on macOS, Grell-Freitas
    `gfl_columns` and RRFS `mynnedmf_run_l1` on Metal;
  - the LTS executor's reserve census needs 64-bit atomic adds, which Apple
    GPUs lack;
  - the dynamics subcycle runs eager, because the facade has no CUDA graphs;
  - the MPAS WSM6 unit of the water-budget gates (binary64) once had not
    finished its cold compile after 1 h 45 min; it now compiles cold in
    about 4 minutes, but its gates have not been re-run on Metal.
- **Virtual Macs are not supported.** GitHub's hosted macOS runners expose an
  "Apple Paravirtual device" that computes wrong answers (NaN) where an
  M-series GPU matches WRF, so CI runs the Metal parity gates only on real
  Apple GPUs.
- **Some checks do not run on Metal.** A few test helpers capture CUDA-only
  compiler output. They are reported as failing on Metal rather than skipped.

### Limitations of the host backend

- **It is slow.** On a 128×128×40 moist bubble it runs about 100× slower
  than an RTX A6000. It is for checking results and for machines with no
  GPU, not for production forecasts.
- **The first run compiles.** Each unit is built once and then cached in
  `~/.cache/woof/host-kernels`.
- **The answers are CUDA's `--fmad=false` words, except for libm.** Nothing
  is contracted. libdevice calls (`expf`, `powf` …) become glibc's, and
  NaN payloads are x86's. Under strict numerics, a one-hour moist bubble is
  bit-identical to CUDA.
- **Not supported:** cooperative launches and grid sync, textures, `__half`
  arithmetic, partial warp masks, and inline PTX outside woof's idioms. The
  hex lane is not measured on this backend.

## Licence

Apache License 2.0. Third-party code, tables and datasets keep their own
terms; see [NOTICE](NOTICE) and [licenses/](licenses/).
