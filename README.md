# Recast WOOF

**Weather Observations and Open Forecasting: GPU-native weather models you run on your own card.**

> **This is a fork.** WOOF is written and maintained by Recast Systems at
> [github.com/recastsystems/woof](https://github.com/recastsystems/woof),
> which is the original and main code, and the place to start. This fork,
> [github.com/jacobbieker/woof](https://github.com/jacobbieker/woof), merges
> each upstream release and adds the work listed under
> [This fork](#this-fork) below, most notably a backend that runs on
> Apple-silicon GPUs. On PyPI, Recast's release is `recast-woof`; this fork's
> Apple-silicon build is `metal-woof`, which is unofficial and not supported
> by Recast Systems.

WOOF is two weather models that run on NVIDIA GPUs, in one install with
their own preprocessing and maps: WOOF-ArWen (the `woof` command) and
WOOF-HEX-A (hexagonal mesh; A for Atmosphere, as in MPAS-A), the `woof hex`
command.

| Model | What it is | Command |
| --- | --- | --- |
| WOOF-ArWen | The WRF engine: WRF-ARW on GPUs for regions you draw, with nests and storm-following runs. Fast numerics by default; an exact mode is checked against native WRF 4.8.0 on the cases its verification page names. | `woof` |
| WOOF-HEX-A | The MPAS engine: MPAS-Atmosphere v8.4.2 and NOAA's hrrrv5 (RRFS v2) configuration on GPUs, on hexagonal meshes: whole globes, zoom globes, limited-area and storm-following regions, with a tangent-linear and adjoint for sensitivity work. | `woof hex`, `woof hex global` |

WRF and MPAS are developed at NCAR; WOOF is not affiliated with or endorsed
by NCAR or UCAR.

Each engine keeps its own version number. WOOF-ArWen is also released on its
own as the `gpuwm` package, and WOOF-HEX-A as `gpuwm-hex`; WOOF 1.1.7
carries gpuwm 3.0.4 and gpuwm-hex 0.6.2. Install `recast-woof`, or `gpuwm`
and `gpuwm-hex`, not both in one environment: they install the same
top-level packages and the same `rw-wps` command.

The WOOF site's API keeps its own values for the two models:

| API `model` value | Model |
| --- | --- |
| `regional`, `cyclone` (and `kind: arwen` in run lists) | WOOF-ArWen |
| `hex` | WOOF-HEX-A |

Separate projects, not part of this package: woof-nam
([recastsystems/woof-nam](https://github.com/recastsystems/woof-nam)) and
ArWen Global
([FahrenheitResearch/gpuwm-global](https://github.com/FahrenheitResearch/gpuwm-global)).

WOOF-ArWen is the production model: its commands and their config and
checkpoint formats are covered by the major-version promise. That is not a
statement about forecast accuracy; see Science and limits below.

WOOF-HEX-A is a preview: it is in the package, documented and tested on CPU,
and its command says preview each time it runs; its config and checkpoint
formats may change in a minor release.

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
. ~/woof-env/bin/activate       # puts woof and its tools on PATH; again in each new shell
python -m pip install 'recast-woof[all-cu13]'
woof fetch-tables
woof fetch-geog --datasets wrf
woof doctor
```

For WOOF-HEX-A, add the geography its meshes read, then check it:

```bash
woof fetch-geog --datasets mesh
woof hex doctor
```

For a driver that serves only CUDA 12, install `all-cu12` explicitly
instead; in this fork `all` and `gpu` mean the CUDA 13 pair. WOOF-HEX-A runs on either extra:
`all-cu12` works on a CUDA 12 or CUDA 13 driver, `all-cu13` needs a CUDA 13
driver, and `woof hex doctor` names the one for your driver. Install one
CuPy/CUDA major per environment.
The Linux wheel carries the native tools (GRIB and NetCDF decoding, the
WPS-equivalent preprocessor, the renderer).

`fetch-geog --datasets wrf` downloads the global geography every forecast
builds its terrain, land use and soil from: about 2.6 GB, about 26 GiB
unpacked, once per computer. `fetch-tables` adds two Thompson microphysics tables (about 314 MiB).
Reference tables and map assets arrive with the `recast-woof-data` package.
`woof doctor` reports each remedy as a command or a `#` comment explaining
the manual step.

For a developer installation from a clone of this repository, use
`bash install.sh`: it installs the checkout's `recast-woof-data` companion and
the package with its extras, builds the native tools (this needs Rust, a C
toolchain and CMake) and ends with `woof doctor`. The explicit `bash install.sh` form
also works when an extracted source archive has lost its executable bits; use
it if `./install.sh` reports `Permission denied`.

## Run a forecast

WOOF-ArWen, a 3 km regional forecast:

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

WOOF-HEX-A, a 120 km global forecast in one command (it fetches its mesh,
start and kernels itself):

```bash
woof hex global --row global-120km --cycle 2026-10-11T12 --work-dir RUN
```

## The derivative commands (WOOF-HEX-A)

`woof hex tangent`, `adjoint-test`, `sensitivity` and `param-grad` run the
tangent-linear and adjoint of the exact-mode hrrrv5 forecast
(`--physics-mode exact`), not of the correct-mode forecast `woof hex
forecast` runs by default. They run on NVIDIA cards of compute capability
12.0 (GeForce RTX 50 series, RTX PRO Blackwell) and need the `ad` extra, in
a Python 3.14 environment of its own (the extra brings its own CuPy package,
and one environment holds one):

```bash
python3.14 -m venv ~/woof-ad
~/woof-ad/bin/python -m pip install 'recast-woof[ad]'
~/woof-ad/bin/woof hex adjoint-test --row hrrrv5-q12.96.2890.n40.60w91.30 --steps 15 \
     --at 40.6,-91.3 --radius-km 150 --out cut12
```

The last command is the 12 km example: one card with 28,700 MiB free (a 32 GB
GeForce RTX 5090 runs it) and 47,104 MiB of host memory. It fetches its own
1.20 GB of inputs and ends PASS when the adjoint passes its transpose check
over the window and the gradient test (the real model rerun from a nudged
start, against the 1 per cent bar) finds no test direction outside the bar
and two independent directions inside it; a direction it cannot resolve is
named. No PASS covers a measure of the storm itself (rain water, cloud water
or updraft), and a measure built on radar reflectivity is refused. What each example row holds, and every
measured limit, is in `components/hex/docs/differentiable/` (`LIMITS.md`).

The learned column correction (`WOOF_HEX_LEARNED_CORRECTION`, from a
checkpoint of your own; no weights ship) and the two network options on the
hrrrv5 suite (the surface-flux network and the neural boundary layer) need
PyTorch, from the `torch` extra, which adds it to the `ad` environment:
`~/woof-ad/bin/python -m pip install 'recast-woof[torch]'`. Each is off
unless named, and no skill is claimed for them; the User Manual's chapter 11
(`components/hex/docs/manual/11-new-in-0.6.2.md`) says what each does.

## Documentation

- [FAMILY.md](FAMILY.md): the two models, their packages, commands and
  versions, and which installs can share an environment.
- `docs/public/`: the CLI manual, physics, hardware guidance and data sources.
  The [WRF comparison](docs/public/README-EVIDENCE.md#wrf-comparison) names
  the grids, physics and fields of WOOF-ArWen's exact-mode cases, and the
  [code verification record](docs/public/VERIFICATION.md) keeps the earlier
  WRF v4.6.1 comparisons with their configuration and coverage limits.
- [Energy forecasts](docs/energy-forecasts.md): `woof energy` builds 50-100 m
  forecasts along power lines, substations and renewable sites from
  OpenStreetMap or your own grid data, with line rating, icing and power
  products.
- [Stage contract](docs/public/PIPELINE-STAGES.md) for running preprocessing,
  the simulation and rendering on your own terms.
- [Smoke, dust and air quality](docs/public/SMOKE-AND-AIR-QUALITY.md): wildfire
  smoke, dust, sea salt, carbon and CAMS air-quality species are an opt-in
  product, off unless a configuration names `chem_sets`.
- `components/hex/`: WOOF-HEX-A's manual, tests and tools.
- `woof --help-all` lists every command; `woof hex --help` lists WOOF-HEX-A's
  commands.

## Science and limits

WOOF-ArWen implements a WRF-ARW-class regional model and WRF-derived physics
on the GPU.
Comparisons against WRF Fortran, ULP and byte-identity checks, and the
matched-run and ensemble consistency comparisons are **code verification**:
evidence that the tested code computes its intended equations. Each result
belongs to the build, reference version and configuration it tested. None
establishes general equivalence with WRF or skill for every combination of
physics, input data and hardware.

In strict numerics, a build of gpuwm 3.0.3 (the release before gpuwm 3.0.4,
which WOOF-ArWen carries here) matched stock WRF 4.8.0 in every stored value on two one-hour, 3 km
cases (40 x 40 x 50 points), on one card and one process. With
`--physics-mode exact --numerics strict`, gpuwm-hex 0.6.0's one-hour hrrrv5
forecasts on a 12 km and a 3 km limited-area mesh matched the patched native
RRFS-MPAS 44c0849 reference in every stored value, on one card and one
process. The default forecasts (fast numerics, and correct mode on
WOOF-HEX-A) are not bit for bit, by design, and other meshes, dates and
lengths are not claimed by these lines.

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
regions, seasons or forecast lengths.

WOOF-HEX-A is a GPU port of MPAS-Atmosphere v8.4.2, with MPAS-A's own
physics suites as its default and ArWen physics selectable; it is not MPAS.
Use official meteorological services for forecasts and warnings.

WOOF 1.0 was the engine's release 2.8.0 and the fixes made since, under the
WOOF name, with WOOF-HEX-A inside it. Settings spelled `GPUWM_*` keep working
beside the new `WOOF_*` spelling.

## This fork

Everything above describes Recast WOOF, and is kept as upstream wrote it.
This section is the fork's own. The fork carries upstream WOOF 1.1.7 and adds
the work below on top of it. It does not change upstream's defaults: each new
option is off until selected, and a configuration that selects none of them
runs as upstream's does. The section is grouped by area, and where
something is partial it says so. Every merge into the fork's `main` is tagged and released on GitHub as a
new `metal-woof` build (`metal-vX.Y.Z.N`); PyPI gets the newest one daily. The fork's own changes are recorded
in [FORK-CHANGELOG.md](FORK-CHANGELOG.md).

Upstream now covers several things this section once listed as the fork's:
storm-following nests and storm-following hex regions (with HAFS starts and
storm merge), coupled ocean waves and storm surge, a GPU mixed-layer ocean
and sea ice, coupled hydrology over a NextGen hydrofabric, HRRR-Smoke and
aerosol transport, and its own wiring of `use_theta_m`, FARMS, MAD-WRF and
Deng shallow cumulus. Where both existed, upstream's keys are the ones to use.

Notes on the 1.1.7 sync:

- Upstream 1.1.7 removed `woof local-da`. The fork keeps it, with its
  tests, and it runs as before.
- The fork takes upstream 1.1.7's correct mode (`physics.mode`
  `correct`/`exact`) and its default-on corrections as upstream ships them.
  The fork's own quirk corrections follow the same switch: on in correct
  mode (the default), off in exact mode.
- The hex MPAS-route qsat pole guard is now correction RP9, not RP8. This
  is a user-visible ID change with no alias.

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
  - Hex's cell-to-grid exchange runs on the GPU by default. The overlap
    weights stay on the device and nothing goes through the host at the LSM
    step. Its words match the host route's on the same machine: it sums in
    scipy's order and copies that host's use of FMA.
    `--no-wrf-hydro-gpu-exchange` (`[wrf_hydro] gpu_exchange = false`) keeps
    the host route, which also runs, with a log line, when there is no GPU
    backend. Measured on a 128x128 1-km LSM grid with a 96k-cell
    hex mesh: on pb the exchange went from 230 ms (host) to 0.4 ms (A6000),
    and the whole coupling cycle from 528 to 196 ms. On the M3 Max it went
    from 15-27 ms to 4 ms; there the routing (about 650 ms) dominates.
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
    split-explicit barotropic stage and whole split-explicit step,
    forward mode (clock on `config_calendar_type`'s calendar, the 365-day
    `gregorian_noleap` by default or `gregorian`; cold start, N steps,
    bitwise restart), and frazil
    ice formation (`config_use_frazil_ice_formation`, off by default as in
    v8.4.2's Registry: open ocean and under land ice, its tendencies, the
    accumulated mass and the surface pressure; standalone and over forward
    steps, in fast and strict numerics), and the land-ice fluxes
    (`config_land_ice_flux_mode` 'standalone' and 'coupled', off by default:
    ISOMIP and Jenkins melt, the top drag, the boundary layer, the thickness,
    velocity and tracer forcing, the effective density in land ice and the
    coupler's land-ice means; standalone, in the diagnostics and over forward
    steps; 'HollandJenkins' is refused by name), and the Gent-McWilliams
    bolus velocity (`ocn_gm_compute_Bolus_velocity`: isoneutral slopes,
    their smoothing and tapering, `k33`, the stream function's solve) and
    Redi isoneutral mixing (`ocn_tracer_hmix_redi_tend`'s three horizontal
    terms, and `k33` in the vertical diffusivity, which v8.4.2 adds without
    `config_Redi_kappa`; `woof_redi_k33 = 'kappa'` scales it), standalone,
    in the split-explicit step (GM before the implicit mixing and after the
    last diagnostics, the bolus velocity in the transport, Redi in the
    tracer tendency, `ocn_reconstruct_gm_vectors`) and over forward steps
    (the cold start's bolus transport); off by default. Three v8.4.2 quirks keep
    their words by default and have corrected opt-ins, each tested on its
    own: the ISOMIP freezing point taken at the boundary layer's salinity
    rather than its temperature
    (`woof_land_ice_isomip_freezing_salinity = 'boundary_layer_salinity'`),
    the Rayleigh bottom friction with its own coefficient and without the
    interior friction (`woof_rayleigh_bottom_friction = 'corrected'`), and
    no bottom drag on a dry edge instead of the refusal
    (`woof_bottom_drag_dry_edges = 'skip'`). The step
    checks its barotropic halo copies against the mesh once per mesh, not
    with two device reads per stage: on an 8,300-cell, 60-level torus a
    step takes 36 instead of 130 ms on an A6000 and 110 instead of 245 ms
    on an M3 Max, with the same words. Its level-independent loops
    (Coriolis, thickness flux, pressure gradient, del2/Leith, vertical
    momentum advection, stability, reconstruction and the split step's
    edge updates) run a thread per edge or cell and level, so a warp reads
    a column's consecutive levels: a 33,000-cell, 60-level step runs about
    1.9 times as fast on a (shared) A6000, the same words. Each barotropic
    subcycle's velocity kernels also add the thickness flux, the
    corrector's velocity copy and the running sum, so in the default
    configuration (predictor, two corrector iterations, no SSH2) a
    subcycle launches 4 kernels instead of 8, the same words: on
    an M3 Max the 8,300-cell step takes about 44 instead of 52 ms. On CUDA
    the barotropic stage is captured as a CUDA graph and replayed (the
    same kernels and arguments, so the same words; switch:
    `WOOF_OCEAN_CUDA_GRAPHS=0`): 9.9 instead of 10.7 ms on an A6000 at
    8,300 cells. On CUDA the barotropic transport and baroclinic update
    also run a warp per edge (the level sums still added in level order)
    and monotone vertical advection a thread per cell and level, so a
    column's loads coalesce: the 33,000-cell step takes 42 instead of 58 ms
    on an A6000, the same words. Metal keeps a thread per column,
    which is faster there (switch: `WOOF_OCEAN_COLUMN_LANES=1` or `32`);
  - MPAS-Seaice: variational EVP internal stress with its basis set-up
    (Wachspress and PWL under every integration rule, computed on the host
    in the init; the RKIND 8 metric terms need `metric_libm` because
    glibc's binary64 asin and tan are not correctly rounded), the
    weak-form stress divergence (`config_stress_divergence_scheme =
    'weak'`, its normal vectors computed on the host in the init; on a
    sphere it needs `metric_libm` too), the velocity
    solve and elastic subcycles, the pre-subcycle routines, ocean stress,
    incremental-remap advection on planar and spherical meshes (the rotated
    grid, hexagons and quadrilaterals, variable resolution) with its
    optional conservation and monotonicity checks (one rank), a forward
    time step with advection (N steps, bitwise restart), and the column
    package's (colpkg's) initialisation and aggregation (RKIND 8: the
    tracer object, thickness categories, salinity profiles, initial ice
    state, coupler fields, aggregate and state update), and its BL99
    vertical thermodynamics (`ice_atmo`'s stability-corrected boundary
    layer, the temperature solve, growth and melt, snow ice). That
    thermodynamics runs only with the LGPL libm opt-in
    `WOOF_GLIBC239_LIBM64=1`, because it needs glibc's atan word for word.
    Also its ITD thermodynamics (BL99 or mushy: the linear remap of the
    thickness distribution or the delta function, frazil and new ice,
    lateral melt, rebinning and zapping, with level, CESM or topo ponds),
    and its ridging (both participation and both redistribution functions,
    the ridging iterations and the cleanup after them), neither of which
    needs an opt-in, and its shortwave: the delta-Eddington shortwave (the
    Registry default: the orbit, the solar zenith angle and `run_dEdd` with
    level, CESM, topo or no ponds, on BL99, mushy or zero-layer ice), the
    CCSM3 shortwave with constant or computed albedos, the coupled
    shortwave, `column_prep_radiation` and the initial shortwave. The
    computed albedos need the opt-in for glibc's atan, and the
    delta-Eddington shortwave needs it for glibc's sin, cos, atan and asin
    (the opt-in now carries asin). Also the coupling preparation
    (`seaice_column_coupling_prep` and `seaice_column_scale_fluxes`: cell
    albedos, pond freshwater feedback, shortwave scaling, fluxes per unit
    ice area; RKIND 4 and 8, no opt-in), with the stand-alone ocean mixed
    layer it runs first by default (`ocean_mixed_layer_v842`: the open
    ocean's boundary layer and `colpkg_ocn_mixed_layer`'s SST and
    freezing/melting potential; RKIND 8, and it needs the opt-in for
    glibc's atan). Four v8.4.2 column bugs keep their words by default and have
    named opt-in corrections (`woof_v842_corrections`): the swapped coupled
    surface and conductive fluxes, the ice velocity in the boundary layer
    (a vertex's u and no v), CCSM3's absorbed shortwave over `ncat`
    layers, and the delta-Eddington's infinite extinction in ice without
    volume. The column package's part of the time step chains these over
    several steps, word for word with v8.4.2: the diagnostics' and fluxes'
    re-initialisations, the pre-dynamics (shortwave preparation, vertical
    and ITD thermodynamics), the column dynamics (ridging), the
    post-dynamics (shortwave and coupling preparation) and the stop on an
    abort, with the day and time read from the step's clock
    (`column_step_v842`). The forward time step runs it with the velocity
    solver, as v8.4.2's `seaice_timestep` does under
    `config_use_column_package` (`timestep_v842`, RKIND 8): the solver's
    ice strength is the column package's `colpkg_ice_strength` (Rothrock75's
    ridging energy or Hibler79's), the fluxes are finalised after each step
    (or left for a coupler after the last), and an abort raises
    `ColumnAbort` with its step, cell and message, word for word with
    v8.4.2 on CUDA and Metal. The elastic subcycles check
    their arrays once per solve and work out their compile options once per numerics mode:
    120 subcycles on a 9,200-cell torus take about 34 instead of 39-41 ms a step on
    an M3 Max, with the same words. Each subcycle computes the ocean
    stress coefficient and solves the velocity in one launch (a vertex's
    solve reads only its own coefficient), so the step launches 120 fewer
    kernels: about 30 instead of 33 ms on an M3 Max, the same words.
    The column biogeochemistry's framework and brine height
    (`config_use_column_biogeochemistry` with `config_use_brine`, opt-in;
    `column_bgc_v842`): the brine fraction tracer, the init's grids,
    profiles and ocean concentrations, and `column_biogeochemistry`'s
    flushing, flooding and Darcy velocity, RKIND 8 with no opt-in, word for
    word with v8.4.2 on CUDA and Metal. v8.4.2's diatom carbon to nitrogen
    ratio (the proteins') has a named correction (`zbgc_c2n_proteins`).
    The brine fraction is carried through the ITD (new ice, the thickness
    surplus, zapped categories), ridging, the remap, the column step and
    the coupled step (the ocean biogeochemistry fluxes zeroed and scaled),
    word for word on CUDA and Metal. v8.4.2's step writes the algae's
    `totalChlorophyll` through a null pointer when the brine height runs
    alone (it dies on every step), so the step refuses that configuration
    unless the correction `bgc_inactive_diagnostics` is named. The vertical
    salinity (`config_use_vertical_zsalinity`, BL99) is inert in v8.4.2,
    which never sets the column package's `solve_zsal` and copies
    `verticalSalinity` only with the vertical tracers; every column stage
    (the init, the brine height's step, the vertical and ITD
    thermodynamics, the ridging, the column step and the coupled step)
    keeps that word for word, and the correction `zsal_solve` solves it
    (`compute_microS`, `solve_S_dt`, the salt fluxes, BL99's `dTemp`, and
    in the ITD and the ridging the new ice's salt loss, `adjust_tracer_profile`,
    `lateral_melt_bgc` and `zap_small_areas`' salt) in every one of those
    stages, word for word on CUDA and Metal with an oracle built from
    `zsal_solve.patch`. The vertical tracers (`config_use_vertical_tracers`)
    with the nonreactive tracer (`config_use_nonreactive`) run in the init,
    `column_biogeochemistry` (`update_snow_bgc`'s scavenging,
    `z_biogeochemistry`'s mobile and stationary phases and
    `merge_bgc_fluxes`), the ITD thermodynamics and the ridging
    (`add_new_ice_bgc`, `lateral_melt_bgc` and `zap_small_areas`' tracer
    into `oceanBioFluxes`) and the column step (the snow and ice layers
    separated after the pre-dynamics and combined before each ridging, the
    vertical thermodynamics, the shortwave and `coupling_prep`) and the
    coupled step with the velocity solver, word for
    word with v8.4.2 on CUDA and Metal. Three v8.4.2 quirks there have opt-in
    corrections: `zap_snow_area` (`zap_small_areas` hands `zap_snow` the area
    it has just zeroed, so the snow surface layer's tracer drops out of the
    flux; the correction passes the category's area),
    `subcycle_snow_ice_tracers` (v8.4.2 combines every dynamics
    subcycle but separates once a step, so an earlier subcycle's ridging of
    the tracers is lost) and `bgc_snow_new_category` (without the vertical
    thermodynamics `update_snow_bgc` divides 0/0 for a category the last
    step opened, and `oceanBioFluxes` is NaN; on Metal that NaN's sign bit
    is the opposite of x86's, the one word not compared bitwise there). The
    other biogeochemistry
    flags are
    refused by name until their phases land;
  - MALI: the shallow-ice velocity solver, masks and geometry diagnostics,
    thermal initialisation, the thermal solver and basal melt, thickness
    and tracer advection, calving (every option, and front restoration),
    subglacial hydrology, one forward-Euler time step, and the forward mode
    around it (the clock with its adaptive-step force alarm, the cold-start
    initialisation, the initial solve and the time-step loop, with a
    bitwise restart; `woof/hex/mpas_ice/forward_v842.py`), word-exact over
    the dome, EISMINT2 A, hydrology and enthalpy-benchmark cases on uniform
    and variable resolution meshes; `woof hex ice run --namelist
    namelist.landice --input landice_grid.nc --out DIR` runs it from an
    MPAS land-ice input file to the oracle's words (`output.nc`, and a
    `restart.npz` to continue from); MALI's side of the land-ice/ocean seam
    exports `landIcePressure`, `landIceFraction`, `landIceMask` and the
    liquid (grounding-line discharge) and solid (calving) runoff, and
    imports the ocean's floating basal mass balance
    (`woof/hex/mpas_ice/ocean_seam_v842.py`, graded by conservation); the
    land-ice core's registry status is now `ported_subset` (velocity
    solvers 'sia', 'simple' and 'none'), and its global statistics
    (`config_stats_interval`, `woof/hex/mpas_ice/statistics_v842.py`) match
    v8.4.2's log report character for character, with
    `woof_statistics_extrema = 'ice'` as an opt-in correction, and so do
    its analysis members `globalStats` and `regionalStats`
    (`woof/hex/mpas_ice/analysis_v842.py`: every word and every compute's
    time against v8.4.2's driver and members compiled verbatim, their
    alarms on the streams' reference times; the door writes each stream's
    records), with `woof_analysis_empty = 'corrected'` (0 for a mean over no
    ice, which v8.4.2 leaves NaN) as an opt-in correction. Two v8.4.2 quirks have opt-in corrections, off by
    default: `woof_sgh_init_mask = 'corrected'` and
    `woof_enthalpy_advection_units = 'corrected'` (the advection converts
    enthalpy in Celsius, as the thermal solver does, so the EISMINT2
    enthalpy ice no longer melts away);
  - the coupler's ocean import merge, E3SM v3.2.0's `prep_ocn_merge`
    (`woof/hex/coupled/prep_ocn_merge_e3sm.py`, opt-in, not called by the
    coupled step): x2o built from the atmosphere, sea ice, runoff, waves,
    land ice and the atmosphere/ocean fluxes, weighted by the ice and
    open-ocean fractions, with E3SM's own field matching and aborts.
    real(r8), word-exact on CUDA and Metal against the routine compiled
    verbatim on the pinned MCT (`tools/e3sm_coupler_oracle`); the water
    isotopes are refused by name;
  - the coupler's sea-ice import merge, E3SM v3.2.0's `prep_ice_merge`
    (`woof/hex/coupled/prep_ice_merge_e3sm.py`, opt-in): x2i from the
    ocean, atmosphere, land ice and waves by shared name, with rain, snow and
    the ice runoff summed and scaled by `flux_epbalfact`. real(r8),
    word-exact on CUDA and Metal against the routine compiled verbatim on
    MCT; the water isotopes are refused by name;
  - MPAS-Ocean's coupler import and export, E3SM v3.2.0's `ocn_import_mct`
    and `ocn_export_mct` with `mpaso_cpl_indices_set`
    (`woof/hex/coupled/ocn_coupling_e3sm.py`, opt-in): x2o into the forcing
    arrays (the removed Antarctic runoff, the precipitation factor, the
    sea-ice pressure cap, the waves coupled two ways, the 10 m wind) and o2x
    from the surface state, the frazil, the land ice and the 3-D thermal
    forcing. real(r8), word-exact on CUDA and Metal against the routines
    compiled verbatim; the BGC, DMS, MacroMolecules and CFC tracers are
    refused by name, and so is what E3SM writes or reads out of bounds.
  - MPAS-Seaice's coupler import and export, E3SM v3.2.0's `ice_import_mct`
    and `ice_export_mct` with `mpassi_cpl_indices_set`
    (`woof/hex/coupled/ice_coupling_e3sm.py`, opt-in): x2i into the ocean,
    atmosphere, aerosol, z-aerosol and wave arrays (Icepack's sea freezing
    temperature, the frazil mass) and i2x from the fraction, the basal
    pressure, the stresses rotated back to the geographic grid and the
    fluxes. real(r8), word-exact on CUDA and Metal against the routines
    compiled verbatim with the pinned Icepack; the rotation's libm words are
    an input. The coupled column biogeochemistry is refused by name, and so
    is what E3SM reads out of bounds or through an unset pointer.
  - The coupler's surface fractions update, E3SM v3.2.0's `seq_frac_set`
    (`woof/hex/coupled/seq_frac_e3sm.py`, opt-in): the ice grid's `ifrac`
    and `ofrac` from `Si_ifrac` and the domain fraction, mapped to the ocean
    and atmosphere grids (copy, rearrangement or MCT's sparse matrix
    multiply in its own element order), with `seq_frac_check`'s stops.
    real(r8), word-exact on CUDA and Metal against the routines compiled
    verbatim on MCT.
  - The coupler's atmosphere-ocean flux, E3SM v3.2.0's
    `seq_flux_atmocn_mct` over `shr_flux_atmOcn`
    (`woof/hex/coupled/flux_atmocn_e3sm.py`, opt-in): `seq_flux_init_mct`'s
    mask and the E3SMv1 scheme into `xao`, with the implicit stress,
    gustiness, the cold-air outbreak scaling, the convergence switches and
    dead models, and E3SM's and MCT's stops. real(r8), word-exact on CUDA
    and Metal against the routines compiled verbatim on MCT. It calls
    glibc's atan, so it runs only with `WOOF_GLIBC239_LIBM64=1` and is
    refused by name without it. COARE, UA, the diurnal cycle, the two-way
    wave drag and the water isotopes are refused by name.
  - The coupler's ocean albedos, E3SM v3.2.0's `seq_flux_ocnalb_mct`
    (`woof/hex/coupled/ocnalb_e3sm.py`, opt-in): `flux_albav`'s namelist
    albedos and radiative fractions, or the net shortwave from xao's
    albedos, with E3SM's and MCT's stops. real(r8), word-exact on CUDA and
    Metal against the routine compiled verbatim on MCT. The orbital albedo
    is refused by name.
  - The coupler's ocean-input accumulation, E3SM v3.2.0's `prep_ocn_accum`
    and `prep_ocn_accum_avg` over MCT's `mct_avect_accum`/`mct_aVect_avg`
    (`woof/hex/coupled/ocn_accum_e3sm.py`, opt-in): the merged x2o summed
    over the ocean's coupling window, averaged by E3SM's rounded `1/count`
    and copied back. real(r8), word-exact on CUDA and Metal against the
    routines compiled verbatim on MCT.
  - One coupler interval chained (`woof/hex/coupled/interval_e3sm.py`,
    opt-in): the ported routines above, run in E3SM v3.2.0's RASM_OPTION1
    order over several coupler steps (the i2x map, the flux, the ocean
    merge and accumulation, the albedos; the window's average on the
    ocean's alarm; the o2x and a2x maps and the sea-ice merge on the sea
    ice's; then the fractions), with the components' exports scripted.
    real(r8), word-exact on CUDA and Metal against E3SM's own routines cut
    verbatim, linked and driven in that order on MCT. Like the flux it needs
    `WOOF_GLIBC239_LIBM64=1`; the wave couplings are refused by name.
    Per-step inputs (`nextsw_cday`, `flux_epbalfact`, the runoff, land-ice
    and wave exports) go through `set_inputs`. A `seq_frac_set` stop
    comes through the chain.
  - The coupled step on the spine (`woof/hex/coupled/run_e3sm.py`,
    opt-in): the interval driven over a `CoupledClock`, one coupler step
    per clock interval. The ocean's and sea ice's alarms come from their
    coupling periods, and the components are called where RASM_OPTION1
    runs them. Each hand-over is a matched exchange, refused by its
    channel's name. Word-exact against the chained oracle with scripted
    components.
  - E3SM v3.2.0 coupler with MPAS v8.4.2 components, not an E3SM
    configuration (`woof/hex/coupled/hybrid_ocean.py`, opt-in): the MPAS
    v8.4.2 ocean (this fork's port) run as E3SM v3.2.0's `ocn_init_mct` and
    `ocn_run_mct` run their ocean (the clock moved to its first coupling
    alarm at start, the means reset each window, the forcing built in
    E3SM's order), with E3SM's import and export around it. E3SM's build's
    constants by default (`T0_Kelvin` 273.15, `rho_sw` 1026), or v8.4.2's
    own. real(r8), word-exact on CUDA and Metal against E3SM's import and
    export cut verbatim and compiled against v8.4.2's modules, around
    v8.4.2's forward mode. Each place E3SM's loop does something v8.4.2's
    does not is listed (`DIVERGENCES`): ported, refused by name (the
    removed Antarctic runoff, the waves, the 10 m wind, the land ice's
    'standalone' and 'data', the 3-D thermal forcing, nonzero iceberg
    fluxes, precipitation scaling, the BGC tracers) or left out with the
    reason (`Faoo_h2otemp` is sent as 0). E3SM's own MPAS-Ocean is not
    ported.
  - The same hybrid for the sea ice (`woof/hex/coupled/hybrid_seaice.py`,
    opt-in): the MPAS v8.4.2 sea ice (this fork's port, column package and
    velocity solver) run as E3SM v3.2.0's `ice_init_mct` and `ice_run_mct`
    run theirs. At start: `nextsw_cday`, the core's init, the coupling
    preparation, the export, then the zero import. Each window: the fluxes
    re-initialised once, the import, v8.4.2's post-atmospheric and
    post-oceanic coupling, then the steps with the counter restarted and no
    per-step finalize, then the export. The cap's constants are v8.4.2's
    stand-alone column package's. real(r8), word-exact on CUDA against
    E3SM's import and export cut verbatim and compiled against v8.4.2's
    modules. `DIVERGENCES` lists every difference. v8.4.2's ESM build
    (CCSMCOUPLED) is left out with its reason. Refused by name: Icepack,
    the forcing streams, prescribed ice, icebergs, floe sizes, waves, the
    scaled dib, the coupled BGC and aerosols, the slab ocean, restarts.
    E3SM's own MPAS-Seaice is not ported.
    are the MALI port's ocean seam: `landIcePressure`, `landIceFraction` and
    `landIceMask` (int32), liquid and solid runoff in place of one mass
    channel, and `ocn2li_floating_basal_mass_bal` back in MALI's sign
    (`landIceDraft` is a diagnostic only). Calving momentum is refused by
    name.

  Limits: Albany's higher-order solvers are refused by name. The sea-ice
  remap computes its init geometry (frames, moments, MPAS's RBF
  coefficients) itself, and refuses halo exchanges and periodic planar
  meshes by name. The sea-ice column physics' time step (mushy vertical thermodynamics,
  shortwave), ocean KPP and GM/Redi,
  and the coupler pieces other than the ocean merge and the ocean's import and export are not ported. A few RKIND 8 cases need the LGPL libm
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
  - Morrison with CESM aerosol (`mp_physics = 40`, `aercu_opt = 0`);
  - RRTMG-K (`ra_lw_physics = ra_sw_physics = 14`, only as the pair), held
    bit for bit to compiled WRF 4.7.1 on oracle columns. It reuses the
    legacy RRTMG kernels and ports only K's differences. `aer_opt`,
    `icloud = 2/3` and `rrtmg_spectral_output` are refused with K by name.
    Beyond WRF, whose K ignores it, `ghg_input = 1` gives K legacy RRTMG's
    time-varying CO2, N2O, CH4, CFC-11 and CFC-12 from the CAMtr table at
    every radiation call, in both spectra and on both routes; unset, K
    keeps WRF's fixed gases bit for bit. A K-only WRF namelist still
    imports without it, as wrf.exe ran; a WRF tree mixing K with another
    RRTMG-family domain under `ghg_input = 1` refuses at import. Hex runs it on its generic column seam (R1), bit for
    bit with the WRF route's adapter on the same columns; a 1 h forecast on
    the `edi-vr` variable-resolution mesh takes 0.184 s/step on CUDA (A6000)
    and 0.282 s/step on Metal (M3 Max).
- **Corrections of WRF/MPAS quirks, on by default (correct mode).**
  Each key changes only the arithmetic it names, where its scheme runs.
  Exact mode turns every one off, so the run is the WRF 4.7.1 / MPAS v8.4.2
  oracle word for word; parity tests run in exact mode. Exact mode is a
  strict `GPUWM_WRF_EXACT=1` process or every correct-mode key set false
  (WRF route), `--physics-mode exact` (hex door), or
  `config_physics_mode = 'exact'` (MPAS-Ocean, MALI and MPAS-Seaice config
  dicts).
  - `woof_quirk_corrections` is the umbrella. Unset (the default), the
    corrections follow the mode, and a correction's own key wins: that
    turns one off in correct mode, or on in exact mode. `"all"` / `"none"`
    turns every one on / off whatever the mode (the hex door takes
    `--quirk-corrections all|none`); a key set false next to `"all"` is
    refused by name, and a key set next to `"none"` wins. On the WRF route
    the keys are correct-mode findings FQ1-FQ6, so the run receipt's
    `correct_mode` block names them.
    It covers `gf_deep_rain_evaporation_booking`, `ysu_surface_heat_exner`,
    `mynn_sfclay_psih_zolzt`, `noah_smc_floor_booking`,
    `thompson_rain_snow_closure` and `rrtmg_lw_model_top` (WRF route);
    `--undriven-species-boundary wrf-flow-dependent` (hex, regional, where its drivers carry it; the only
    flag `--quirk-corrections` sets); `woof_land_ice_isomip_freezing_salinity`,
    `woof_bottom_drag_dry_edges`, `woof_rayleigh_bottom_friction` and
    `woof_redi_k33` (MPAS-Ocean); `woof_sgh_init_mask`, `woof_enthalpy_advection_units`,
    `woof_simple_velocity_bed`, `woof_statistics_extrema` and
    `woof_analysis_empty` (MALI); the MPAS-Seaice column's
    `set_sfcflux_order`, `ice_velocity_at_cell`,
    `ccsm3_absorbed_solar_layers`, `dedd_zero_volume_ice`,
    `zbgc_c2n_proteins`, `bgc_inactive_diagnostics`, `zsal_solve`,
    `bgc_snow_new_category`, `subcycle_snow_ice_tracers` and `zap_snow_area`;
    and the MPAS-Seaice velocity solver's `planar_metric_terms`,
    `periodic_local_coords` and `weak_ridge_delta`.
    The list and its values are in
    [CONFIGURATION.md](docs/public/CONFIGURATION.md) and
    `woof/quirk_corrections.py`.
  - `ysu_surface_heat_exner` (YSU, `bl_pbl_physics = 1`, WRF route). WRF's YSU adds the surface sensible heat flux to
    potential temperature as if it were a temperature flux and converts
    back with pi, so the column gets `HFX*pi` instead of `HFX`. That loses
    4.5% of the surface heating at 850 hPa and 14% at 600 hPa. The option
    uses `HFX/pi(1)` (not read with BEP/BEM, where YSU takes the urban
    heat sources instead; the receipt lists it as uncorrected there). Checked by the column heat budget, not yet scored
    against observations; the hex MPAS-suite YSU does not carry it yet.
  - `mynn_sfclay_psih_zolzt` (MYNN surface layer, `sf_sfclay_physics = 5`,
    WRF route). WRF integrates the 2 m and 10 m heat profile from `z0/L`
    while its log term and the first-level `psih` start at the thermal
    roughness `zt/L`. The option uses `zt/L` throughout. It moves only the
    T2/Q2 diagnostics; checked against the corrected CPU reference, not yet
    scored against observations.
  - `noah_smc_floor_booking` (Noah, `sf_surface_physics = 2`, WRF route).
    WRF's Noah raises a soil layer that dries below 0.02 to 0.02 and books
    the water it adds nowhere, so dry sandy columns gain water (up to 30 mm
    in a step on a column that starts below the floor). The option takes it
    from the layer below, and past the bottom layer from the drainage, so
    the column water budget closes. Not read with a WRF-Hydro coupling. Not
    yet scored against observations; the hex MPAS-suite Noah does not carry
    it yet.
  - `thompson_rain_snow_closure` (Thompson, `mp_physics = 8, 28, 38`, WRF
    route). Below 0 C, WRF's Thompson limits the rain and snow losses of
    rain-snow collection but keeps the graupel gain at its unlimited value,
    so the exchange creates graupel (and its freezing heat) when rain or
    snow runs out. The correction makes the gain equal what rain and snow
    lose, floored at zero (finding FQ5). Not yet scored against observations; the hex
    MPAS-suite Thompson and RCON (`mp_physics = 29`) do not carry it.
  - `rrtmg_lw_model_top` (legacy RRTMG longwave, `ra_lw_physics = 4` with
    `ra_rrtmg_variant = "rrtmg_legacy"`, WRF route). WRF's RRTMG longwave
    matches its standard profile to the column in a loop that starts one
    interface too low, so it overwrites the model-top interface temperature
    and the top layer's with the profile and anchors at the interface below
    the top. The option keeps both and anchors at the top interface (finding
    FQ6). Not read by RTE+RRTMGP; not yet scored against observations.
- **Observation-driven corrections, each opt-in and off by default** (GF's
  is also a quirk correction, so it is on in correct mode):
  - `gf_deep_rain_evaporation_booking` (Grell-Freitas, `cu_physics = 3`).
    WRF's deep GF counts the downdraft evaporation it returns to vapour as
    surface rain too, so deep `pret` is 1.29 to 1.66 times the column's
    water loss. The option closes that budget and leaves the tendencies
    unchanged. On five global 120 km cases scored against Stage-IV, the
    24 h bias goes from +0.49 to −0.11 mm and RMSE falls by 1.3 mm. ETS
    at 1 mm falls by 0.02. Evidence:
    [declared-divergences.md](components/hex/docs/declared-divergences.md)
    section 2.
  - `global-120km-high-top` (or `--vertical-spec` with
    `woof/hex/data/countries/s2s80-v1.json` on a `--physics-row arwen`
    global row) for the polar stratospheric drift. The default global rows
    use MPAS-A v8.4.2's 55 levels and 30 km lid, and native MPAS drifts the
    same way.
    On seven held-out 72 h cases scored against WIS2 radiosondes, the
    90-60S 20 hPa bias goes from −9.98 to −1.75 K and the RMSE from 11.44
    to 2.32 K. The cost is about +0.3 K at 100 hPa, already present at
    hour 0. UGWP non-orographic drag does not help. Evidence:
    [declared-divergences.md](components/hex/docs/declared-divergences.md)
    section 1.
  - On the MPAS suite route the 50 km lid needs correction RP9. MPAS
    v8.4.2's MYNN `qsat_blend` has a pole where p reaches the saturation
    vapour pressure, which happens above about 35 km. RP9 adds the
    NOAA-GSL MYNN-EDMF guards. It is on in correct mode (the route's
    default), and `--physics-correct-skip RP9` turns it off. This
    correction was RP8 before the 1.1.7 sync; upstream 1.1.7 uses RP8 for
    the BouLac sign fix, so the fork's ID changed to RP9 with no alias.
  - The MPAS suite route also runs the declaration's upper damping as
    MPAS v8.4.2's `&damping` options (`config_mpas_cam_coef`,
    `config_rayleigh_damp_u` and their level counts and timescale). The
    operators match MPAS-8.4.2's own lines, compiled natively, bit for
    bit.
- **Hex (MPAS) parity with WRF's options.** The goal is for the hex model
  to accept the options WRF does, reusing WRF's column implementations
  rather than copies, and to refuse by name what it cannot yet do. Landed:
  - `woof go` runs hex with `[model] engine = "hex"` (regional route);
    WRF's severe-weather keys there (`lightning_option`, `hailcast_opt`,
    `afwa_*_opt`, `nwp_diagnostics` and their knobs) become the MPAS suite
    route's `&diagnostics`. A `--physics-namelist` holding only
    `&diagnostics` keeps the default MPAS suite, from `woof hex forecast`
    and from `woof hex regional --forecast-arg`. The WOOF column route
    refuses `&diagnostics` by name and says how to fix the request;
  - surface inputs on the generic column seam: CLM lake, SLUCM urban on
    per-cell fractions, WUDAPT LCZ classes, Noah and RUC mosaic, Pleim-Xiu
    and the slab with their soil inits, irrigation and `topo_wind = 1`
    ([docs/ruc-mosaic-and-clm-lake.md](docs/ruc-mosaic-and-clm-lake.md)).
    `topo_wind = 1` forms WRF's terrain Laplacian from the mesh's own
    neighbours (`cellsOnCell`), which is WRF's 5-point form on a quad
    neighbourhood. It is not recommended on the 120 km globe: WRF's
    -30 m cut-off removes the surface drag in 32 % of land cells there,
    and against WIS2 stations it raised the 10 m wind bias by 0.5 m/s.
    The seam warns above 4 km
    ([TERRAIN-DRAG.md](docs/public/TERRAIN-DRAG.md)).
    `slope_rad = 1` on the WOOF column route builds SLOPE and SLP_AZI from
    the least-squares terrain plane through each cell's mesh neighbours, in
    place of WRF's grid-axis centred differences (`woof/hex/slope_statics.py`
    documents the difference); spherical meshes only, and `topo_shading`
    stays refused;
  - CCPP TKE-EDMF on the column seam;
  - diagnostics: `nwp_diagnostics` maxima, `p_lev`/`z_lev` levels, `tslist`
    station series, `prec_acc_dt` windows and `acc_phy_tend`, AFWA
    (icing and some radar/VIL/visibility options are refused), CLWRF, and
    RASM and surface-energy reductions. `afwa_ptype` keeps its intended
    logic rather than WRF's literal operator precedence; the module
    documents the difference;
  - severe weather on the MPAS v8.4.2 suite route (R2): lightning
    (`lightning_option = 1, 2, 3`, flash counts, rates and LPI) and HAILCAST.
    `woof hex products` draws these, the AFWA fields and the PSD hail sizes
    whenever a run writes them (HAILCAST and PSD hail on the renderer's hail
    maps; lightning and AFWA as plain field maps with generic colour scales),
    plus the -10 C reflectivity maximum (`REFDM10C_MAX`: the WRF route's gsd
    kernel on mesh columns, written when a stream asks for it; max vertical
    vorticity is not on hex because WRF's grid stencil does not carry over),
    beside the regional severe set (MLCAPE, SRH, shear, EHI, STP, lightning
    threat, gust);
  - `aer_opt = 1, 2`, `w_damping`, grid and spectral analysis nudging, and
    WRF observation nudging (`obs_nudge_opt = 1`, `--obs-nudging`), with
    WRF's weights and great-circle radii found through a cell KD-tree;
  - stochastic physics: SPP (`spp_conv`, `spp_pbl`, `spp_lsm`) on the WOOF
    column route, WRF-Solar EPS (`multi_perturb`) on regional meshes, and
    WRF's inflow perturbation on the regional relax rings
    (`woof hex forecast --inflow-perturbation`, which takes its depth from
    the run's own boundary-layer height since hex has no parent model).
    `--w-damping` and `--inflow-perturbation` run on both the WOOF column
    route and the MPAS v8.4.2 suite route (R2), where the depth is the
    suite's `hpbl`;
  - WRF's flow-dependent specified zone for the microphysics species a
    regional boundary stream does not drive (`woof hex forecast
    --undriven-species-boundary wrf-flow-dependent`: outflow copies the
    first cell inside the zone, inflow sets 0, or `ccn_conc` for the CCN
    number). The default, `free`, keeps MPAS's behaviour: on the WOOF
    column route those species get no boundary step, and on the MPAS
    v8.4.2 suite route (R2) they are driven from the boundary pool's
    allocated values. R2 carries the rule on one card: it then drives only
    the species the boundary files hold;
  - extra-species microphysics on regional hex runs: CAM-MG, Morrison and
    Morrison-aerosol, NSSL-2, Milbrandt-Yau, P3 (mp 50-53), NTU, ISHMAEL,
    UDM, WDM6, Goddard 4-ice, Ferrier-Aligo with advected rime (mp 15)
    and Fast-SBM run in the hex option matrix on
    CUDA and Metal (`components/hex/evidence/option-matrix.json`).
    Fast-SBM history frames carry the allocated reflectivity default,
    because WRF's Fast-SBM writes no `REFL_10CM`. ISHMAEL needs its WRF
    tables staged first (`woof fetch-tables --ishmael`);
  - `woof resume` and `woof branch` for hex runs
    ([docs/branch-from-checkpoint.md](docs/branch-from-checkpoint.md));
  - GRIB2 and Earth2 export ([docs/grib2-export.md](docs/grib2-export.md));
  - a terrain hook (`--terrain-source`, `[static.highres]`);
  - variable-resolution regional meshes: `woof hex regional` grades the cut
    over several doublings (`--ring-factor`, `--transition-factor`,
    `--cull-pad-scale`, e.g. a 3 km core to 20 km) or refines it by a
    `woof hex density` raster (`--density-raster`, e.g. steep terrain).
    `tools/hex_varres.py` holds the variable-resolution test and bench
    cases (graded, terrain-driven and a global 120-30 km mesh) and the
    transition-band diagnostics. The step is set by the smallest cell, so
    a graded mesh costs about what its finest cells cost;
  - an option matrix that runs generic-seam column options through a real
    hex forecast on CUDA and Metal. Every row runs, is refused by name
    with its reason, or records its failure: CAM radiation is refused on
    Metal, GFDL-Eta on the 149-level case (nominal eta), MAD-WRF
    (species row) and any shortwave-only or radiation-off run with Noah-MP
    (no GLW source). On the 149-level case the CUDA rows for WSM3, WSM5,
    WSM7, WDM5, WDM7, BouLac, QNSE (PBL and surface layer), Multiscale
    Kain-Fritsch and Fu-Liou-Gu now run within the baseline's budget
    tolerances, since each admits 256 levels (compile-time tiers or
    runtime-sized workspaces; words at the old level counts unchanged,
    `tests/test_cu_kfcup_mskf_256_levels_gpu.py` and each family's
    256-level test). KF-CuP now passes admission there too but its CUDA
    row stops with negative accumulated `rainc` (a binary32 rounding in
    the capped downdraft, not the level count); the row records that
    failure until the fix lands. The Metal entries, KF-CuP's 128-level
    refusal among them, predate the WRF
    4.8.0 sync: the case runs the default Grell-Freitas
    (`gfl_873b414`), which is refused on Metal, so they refresh after the
    GF Metal port lands.

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
- **Building morphology (opt-in input).** `woof.static.buildings` turns
  GEM-15 heights (surface minus terrain, with tree and noise filters) and
  open footprint files (Overture Maps, Microsoft, Google Open Buildings,
  OSM; GeoJSON or GeoParquet) into per-cell building statistics on the model
  grid. These are plan and frontal area indices, a height histogram, and
  WRF's NUDAPT `URB_PARAM` block. Each source's licence and attribution
  travel with the file. `woof buildings` builds the product for one
  domain of a WPS namelist. Footprints are read from local files, or
  fetched for the domain from the public Overture Maps release
  (`--overture`, anonymous S3, cached, ODbL-1.0 with per-record source
  licences recorded; `--offline` uses the cache only). With
  `urban_param_file` set, the SLUCM, BEP and BEP+BEM read that file's
  `URB_PARAM` (or a `geo_em`'s) as WRF 4.7.1's real.exe and
  `urban_var_init` do. This includes the SLUCM's NUDAPT arm, which matches
  WRF bit for bit. Files that cannot be matched are refused
  ([docs/public/BUILDINGS.md](docs/public/BUILDINGS.md)).
- **Resolved buildings in LES (`obstacle_opt = 1`, opt-in).** On a domain
  with spacing up to 500 m, the building product becomes porous immersed
  obstacles on the WRF route. Faces of solid cells are blocked (Brinkman
  penalisation), partly built cells get canopy drag, and walls and roofs
  get log-law stress. One fused CUDA/Metal kernel acts on the RK-stage
  momentum tendencies. A street-canyon benchmark gives the H/W = 1 vortex,
  and a cube benchmark gives the wake and horseshoe. Off, or with no
  buildings, nothing launches and runs are bit-identical. Momentum only,
  regional grid only. The hex (MPAS) dycore takes the same option on the
  WOOF-physics route (`woof hex forecast --obstacle-opt 1`), on the Voronoi
  edges and w faces: a cube on a test sphere at 15 cells per H reattaches
  at 1.80 H. A real-data check over Manhattan places the model's
  buildings at their footprint heights (r = 0.99 at 25 m). In lower and
  midtown Manhattan, 50 m resolves 59% of buildings and 25 m resolves 77%;
  50 m leaves only about a third of the street area fully open, against
  nearly half at 25 m. A two-hour rerun on Overture footprints finds a
  street-canyon vortex in 43-45% of canyons at 25 m, against 30-39% at
  50 m and none without buildings. The same case runs on Metal through
  `woof sim`, about 2.4 times slower than a shared A6000. A domain with
  buildings (or an urban canopy under LES) takes bare-earth 3DEP terrain instead of the sub-km
  GLO-30 default, which has the buildings in it, and is refused outside
  3DEP unless a terrain source is named.
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
- **GFS lake and snow choices on a downloaded run.** A `[case_data]` table
  that holds only `gfs_lake_skin`, `gfs_lake_ice` or `gfs_snow_convention`
  stays on `woof go`'s fetch route (including `--prepare-only`); upstream
  sends any `[case_data]` table to the declared-inputs route, where a GFS
  download cannot run. A table with any other key routes as upstream's does.
- **Applied tooling:** wind-energy siting, hindcasts and power products,
  storm-track planning, and delete-after-commit backpressure and domain
  selection for upstream's Icechunk store export
  (`[output] icechunk_delete_after_commit`, `icechunk_domains`), and
  store-only S3 credentials for a store in another account
  (`WOOF_ICECHUNK_AWS_*`, read by the writer alone).
- **Observation referees for MPAS-layout history.** These are for the hex
  model or native MPAS-A.
  - `woof.hex.sonde_profile_referee` scores temperature against radiosondes
    at the mandatory levels, by latitude band and lead, including the drift
    per 24 h. The sondes are IGRA v2 and WIS2, read from the Source
    Cooperative icechunk stores.
  - `woof.hex.station_wind_referee` scores the 10 m wind speed against
    WIS2 land SYNOP reports by latitude band and lead, globally.
  - `woof.hex.qpe_cell_referee` scores 24 h precipitation against Stage-IV,
    averaged onto the model's cells.
  - All three compare two arms with a case-paired bootstrap
    ([obs-referee.md](components/hex/docs/obs-referee.md)).
- **Performance work.** Metal: out-of-line compilation of the largest units,
  compile prefetch and persisted translations, fewer host syncs per step,
  and hardware fast paths that keep the same words (a strict 3 km forecast
  went from 47.7 to 42.1 s per forecast hour); CUDA: a memory-capped compile
  pool and fewer boundary launches. The sixth-order filter of every
  mass-point row (theta, moisture, chem, TKE) is one launch per step
  (`diff6_multi`, byte-identical: CUDA Haleakala 333 m LES about 9 % less
  wall per forecast hour, Edinburgh 3 km about 2 %; `WOOF_DIFF6_MULTI=0`
  restores the per-field launches). Advection folds each species' tendency
  zero-fill into its first accumulating launch (`zero_first`, word-identical
  on CUDA and Metal). On CUDA the four flux-divergence launches take 32 x 4
  thread blocks instead of 128-wide rows (word-identical; 6-14 % less kernel
  time at 150-260 points per row on the A6000; Metal keeps the rows). The
  positive-definite flux and renormalisation launches take 64 x 4 blocks
  on CUDA (word-identical in fast and strict; 3-11 % less kernel time,
  e.g. 257 -> 230 us at 50x150x186; `WOOF_PD_BLOCK=BXxBY` overrides, width
  a multiple of 32; Metal keeps the rows). Its advection receipts for the RTX 5090 and RTX PRO 6000
  (Blackwell) are **not re-pinned**: no Blackwell card was available, so
  `certified_unrepinned` in each fixture's `gpu-receipts.json` records the
  pins and the commit, and the gate warns rather than fails for those pins.
  They are marked stale pending a re-record on an RTX 5090; until then the
  RTX A6000 receipt stands in and gates kernel edits (`certified_stand_in`). An RK stage's lateral specified/relaxation
  rows (u, v, theta, phi, w, mu) are one launch that also absorbs their held
  zero-fills and interior ufuncs (`state_relaxation_multi`, byte-identical;
  about 33 launches, 15 memsets and 15 strided adds fewer per nested domain
  step on the LES, 34 launches fewer per step on a 3 km specified domain;
  `WOOF_LBC_RELAX_MULTI=0` restores the per-row launches). The launch walks
  each level's frame rings first and the interior after, so the relaxation
  runs in dense warps (word-identical; 243 -> 145 us per nested-stage
  launch at 49x120x150, 479 -> 121 us at 49x120x120; CUPTI kernel time on
  a shared A6000, default numerics, against the merge base). The step-end
  `spec_bdy_final` takes its forced fields (u, v, theta, phi, moist scalars,
  w on a nest) in one table-driven launch while every field has at most 1M
  cells (`finalize_state_multi`; each cell runs the per-field kernel's own
  body, byte-identical in fast, strict and WRF-exact on CUDA and Metal;
  10 fields at 50 levels, CUDA-event minimum on a shared A6000: 73 -> 46 us
  at 60x60, 160 -> 138 us at 100x120; larger domains keep the per-field
  launches, which are faster there; `WOOF_LBC_FINAL_MULTI=0` restores them,
  `WOOF_LBC_FINAL_MULTI_MAX_CELLS` moves the cut). WRF-Solar's
  FARMS surface irradiance (`swint_opt = 2`) is one launch per step: the
  column kernel composes the cloud fraction per cell, merges the MYNN
  boundary-layer cloud, scales the microphysics radii and writes the six
  fluxes and the published cloud fraction itself (byte-identical; about 20
  launches and copies fewer per step; `WOOF_FARMS_FUSED=0` restores the
  per-step glue; cloud options outside the shared composition keep it).
  The Smagorinsky scalar rows (theta, moisture, chem, TKE) take their
  horizontal flux and divergence launches four rows at a time with the face
  and cell metrics formed once (`wrf_smag_flux_s_multi`/`wrf_smag_hd_s_multi`,
  byte-identical; `WOOF_SMAG_MULTI=0` restores one launch pair per row).
  The u and v Smagorinsky stress divergences read the stress coefficients
  (rho K, rhoavg kavg) stored once per step (`smag2d_uv_faces`,
  byte-identical in fast and strict numerics) where each point re-evaluated
  up to ten of them: the u+v pair 1.3 -> 0.44 ms at 49x120x150 and
  9.5 -> 2.1 ms at 49x300x300 on an A6000, 4.0 -> 2.0 ms and 32 -> 18 ms on
  an M3 Max (`WOOF_SMAG_UV_FACES=0` restores the per-point kernels).
  `km_opt = 2`'s TKE source reads the defor13/defor23 faces stored once
  (`smag2d_tke_faces`, the `wrf_tke_rhs` statements on the stored words,
  byte-identical in fast and strict numerics) where each mass point
  re-evaluated up to six of them (`WOOF_TKE_FACES=0` restores the
  per-point kernel). The w stress divergence reads the defor13/defor23
  faces stored once and keeps the stresses' density and K averages at the
  reader (`smag2d_w_faces`, byte-identical in fast and strict numerics; the
  earlier stored-stress route moved fast-numerics words through FMA
  contraction): 9.0 -> 2.6 ms at 49x120x150 and 60 -> 17 ms at 49x300x300
  on an A6000, 18 -> 3.4 ms and 75 -> 22 ms on an M3 Max
  (`WOOF_SMAG_W_FACES=0` restores the per-point kernel).
  `WOOF_ACOUSTIC_KPAR=1` takes a level-parallel w''-phi'' solve (32 columns
  x 8 lanes per block; 1.7-2.2x the column kernel's speed per launch on an
  A6000): opt-in, because on CUDA its words are the column kernel's generic
  build's, not the shape-specialized build's that runs by default. The
  column w''-phi'' solve itself takes its forward sweep in the forcing loop
  and its back substitution, damper, phi'' update and p''/alpha''
  diagnosis in one top-down pass (byte-identical in fast and strict
  numerics on CUDA and Metal): 725 -> 655 us per launch at 49x228x276 and
  367 -> 329 us at 49x150x186 on an A6000 (CUPTI kernel time per launch,
  default numerics and shape-specialized build, this change against its
  merge base, on a shared GPU; output words compared equal). The specified/nested frame's
  phi'', w'' and p'' update launches over the ring's columns with level
  lanes instead of one thread per column of the domain
  (`advance_frame_phi_w_ring`, byte-identical in fast and strict
  numerics): 108 -> 25 us per substep at 49x150x186 and 116 -> 33 us at a
  nested 49x228x276 on an A6000 under the same conditions. On Metal the
  YSU boundary layer launches 64 column blocks per GPU core per tile
  (workspace capped at 512 MiB), so on a 40-core M3 Max a 228 x 276
  domain is one launch instead of four (byte-identical in fast,
  strict and WRF-exact numerics; Alps 3-1 km on an M3 Max: `ysu_column`
  46 -> 32 ms of isolated GPU time per d01 step, 10 -> 3 launches;
  `WOOF_METAL_YSU_TILE_BLOCKS=16` restores CUDA's width, and CUDA keeps
  its occupancy-bounded 16 per SM). On Metal the Smagorinsky deformation
  reads u/msfu and v/msfv stored once per face instead of dividing at
  every stencil point, in every K launcher (km_opt 1, 2, 3, 4 and 5;
  9-11 % off the 2-D launch, TKE K 15.5 -> 14.4 ms and 3-D Smagorinsky K
  23.0 -> 21.9 ms at 49x228x276 on an M3 Max, the same words;
  `WOOF_SMAG_DEFORM_HAT=0` restores the old kernel; CUDA keeps it,
  because the extra pass did not pay there). MRF, New SAS and the
  registry column cumulus (BMJ and its siblings) take the same Metal tile
  rule as YSU, 2,048 columns per core under the byte cap (byte-identical
  in every numerics mode; 49x228x276 on an M3 Max: MRF 8.4 -> 6.5 ms,
  New SAS 54.6 -> 33.1 ms per launch, BMJ 16.3 -> 12.5 ms on 63k
  columns, EEPS 14.4 -> 11.6 ms at 49 levels;
  `WOOF_METAL_COLUMN_TILE_BLOCKS` sets the width in 32-column units;
  CUDA keeps its occupancy-bounded tiles). Hex: CUDA
  subcycle graphs stay on under the adaptive time step, one graph per
  repeated step length. On a 4,565-cell variable-resolution mesh, a
  byte-identical adaptive run takes 0.47-0.64x the eager adaptive time and
  0.38-0.47x the fixed-step time (`WOOF_HEX_CUDA_GRAPHS_ADAPTIVE=0` restores
  the eager path). Hex on Metal: the regional garbage-column scrub passes
  its descriptors by value (`WOOF_HEX_SCRUB_WORDS=0` restores the device
  table), and a fresh storage's address waits only for its scheduling
  (`WOOF_METAL_ADDRESS_ASYNC=0` restores the eval). Together they remove
  about 96 % of the queue drains on that mesh, and the hour takes 0.59x the
  wall time on edi-vr and 0.64x on a uniform 12 km mesh. Both are
  byte-identical. Hex on CUDA: a forecast keeps its kernel images in the
  box's kernel cache (`<kernel cache root>/cupy`) instead of each run's
  `--scratch`, so a regional run in a fresh work dir no longer recompiles
  in its first step (edi-vr 1 h on an A6000: step 1 from 42.5 s to 3.6 s,
  byte-identical; `WOOF_HEX_SHARED_KERNEL_IMAGES=0` restores the per-run
  cache).
  Hex regional scalar transport reads each edge's advection stencil once
  per (edge, level chunk) and walks the tracers
  (`transport_edge_values_regional_hoist_v841`, byte-identical in fast and
  strict numerics on CUDA and Metal; Alps 3 km variable-resolution, 11
  tracers: 28 -> 2-3 ms per launch on an M3 Max, 15 -> 0.55 ms on an A6000;
  `WOOF_HEX_REGIONAL_EDGE_HOIST=0` restores the element kernel).
  Hex regional forecasts keep each LBC interval's expanded driving
  tendencies for the whole interval (up to 256 MiB per rank;
  `WOOF_HEX_LBC_TEND_CACHE_MB` sets the cap, 0 expands on every call), and
  MPAS physics producers that write a whole resident pool field write it in
  place instead of copying through a private buffer
  (`WOOF_HEX_POOL_DIRECT=0` restores the copy). Both are byte-identical in
  fast and strict numerics on CUDA and Metal (Alps VR 1 h: Metal median
  step 241 ms against 245-294 ms off).
  Hex MYNN on Metal gives every column its own worker (within the
  one-eighth-of-free-memory bound) instead of 128 per GPU core, so the
  column loop no longer strides: on the 14,751-cell Alps VR mesh
  `mynn_mmm_run` drops from 112 to 50 ms a step, byte-identical
  (`WOOF_HEX_MYNN_WORKERS_PER_PROCESSOR` sets the per-processor cap on
  either backend; CUDA keeps 128 per SM).
  Hex `woof hex moving`: the resident forecast process prepares the next
  region beside the running one only when the host has room for a second
  copy (`WOOF_HEX_MOVING_PREPARE_HOST_GATE=0` always prepares;
  `WOOF_HEX_MOVING_PREPARE_HOST_RESERVE_GB`, default 2, is the room kept
  free; the copy is priced at the last preparation's peak growth times
  `WOOF_HEX_MOVING_PREPARE_HOST_MARGIN`, default 1.25), and on Linux it gives the freed heap back at each move. A skipped
  preparation makes the move slower; the words are the same.
  The Metal facade's `get(out=)` reads the evaluated buffer in
  place from unified memory and copies once, into `out`: the history
  writers' per-field staging into page-locked blocks takes about half the
  time (40 Edinburgh 3 km fields: 51-86 ms against 111-143 ms median,
  byte-identical; `WOOF_METAL_DIRECT_OUT=0` restores the two copies).
  WRF on Metal: the Smagorinsky row kernels' by-value address
  tables beside their grid pointers, and the spec-zone ring copy's device
  table, are resolved to the buffers they name and ordered by MLX instead
  of a queue drain before and after each launch (Edinburgh 3 km: 26.5 to
  10.8 blocking drains per step, byte-identical;
  `WOOF_METAL_RESOLVE_DEVICE_TABLES=0` restores the drains). Hex on Metal:
  the solve-region checks, the physics seam's per-species tables and the
  MPAS sea-ice driver tables are resolved the same way (Edinburgh 3 km
  variable-resolution: 23 to 0 drains around those launches per step,
  byte-identical). A host upload (`cp.asarray`) is held as the array
  built from host memory, not as a lazy contiguous copy, so reading a
  freshly uploaded table no longer waits for the GPU queue (hex Edinburgh
  3 km variable-resolution: the WSM6 recovery table's read from 14 to
  0.1 ms per step, byte-identical; `WOOF_METAL_READY_UPLOADS=0` restores
  the old path).
  A device-to-device `copy_from_device(_async)` between facade arrays (the
  wrfout writer's per-field snapshot) is one ordered device copy instead of
  a read back and a write: the writer's host time per Edinburgh 3 km frame
  falls from 761-1061 ms to 470-576 ms, byte-identical
  (`WOOF_METAL_DEVICE_COPY=0` restores the host round trip).
  Metal: woof's glibc binary32 `logf`, `expf` and `powf` ports
  (`gfk_log`, `gfk_exp`, `gfk_pow`) compute in binary64, which Metal
  emulates in integer arithmetic. They now first evaluate the same
  formula in float pairs and keep that word only when no binary32 rounding
  boundary lies within its error bound, and otherwise run the binary64
  body. The words are identical by construction, and every binary32
  input (logf, expf, and powf at 20 exponents) plus 2^30 random powf
  pairs was checked under both subnormal modes
  (`tools/metal_gfk_fast_grade.py`). The surface-layer kernel
  `sfclayrev` is 5.5x faster (55x83: 11.70 to 2.14 ms). On the hex
  Edinburgh 3 km variable-resolution forecast, WSM6 drops from 22.5 to
  5.2 ms per step, the surface layer from 21.0 to 6.6 ms and Grell-Freitas
  from 8.3 to 4.6 ms; one forecast hour integrates in 44-47 s instead of
  50-57 s, with byte-identical history.
  `WOOF_METAL_GFK_FAST=0` restores the binary64-only path.
  The Metal facade's `ReductionKernel` folded each output on one thread.
  An exact, order-free and idempotent fold (`a | b`, `a || b`, `a & b`,
  `a && b`, a concrete reduce type and an `out = a` post-map) over 4096
  or more elements per output, with fewer than 2048 outputs, is now folded
  in slices and then across the slices, giving the same value (hex Edinburgh 3 km
  variable-resolution: Noah-MP's surface identity check from 3.6 ms to
  0.16 ms per step and the transport species-word check from 1.4 ms to
  0.18 ms, byte-identical; `WOOF_METAL_SPLIT_REDUCTIONS=0` restores the
  serial fold).
  `WOOF_METAL_GPU_TIMES` times each kernel. Its per-launch command
  buffers also hold the lazy MLX work the launch waited on, and their GPU
  intervals overlap, so a facade write (`woof_copy_4`) used to carry other
  work's time (on the Alps 3 km/1 km pair, 883 of 2047 timed ms per step).
  `WOOF_METAL_GPU_TIMES_ISOLATE=1` is a profiling mode with the same words:
  each launch runs alone, after its lazy inputs are evaluated and counted
  apart, and writes are keyed by their Python call site. There the facade's
  writes cost 45 ms of 1357 per step (Alps hex variable-resolution mesh:
  65 of 667 ms, where `mynn_mmm_run` is the top kernel at 107 ms).
  The open-boundary frame zero and copy (`zero_frame`, `copy_frame`)
  launch one thread per frame element instead of one per buffer element,
  storing the same words (Alps 1 km nest: about 13x faster on Metal and 7x
  less kernel time on CUDA).
  The `woof.radtran` spectral diagnostic (`radtran_*`) is cheaper, keeping
  the same words. `radtran_disort.cu` compiles in about 6 minutes in
  0.55 GB, against over an hour in over 17 GB before: its helpers are kept
  out of line on CUDA. Its workspace is interleaved across threads on CUDA,
  and on CUDA its chunk is sized from free device memory. Together these
  make the 16-stream solver about 9x faster on an A6000. The Hu & Stamnes cloud optics
  are evaluated vectorized, with `np.interp`'s own multiply-add (fused or
  not, as numpy was built). The layer optics run on host threads over
  blocks of columns. On the Alps 3 km/1 km pair, one 550 nm call fell from
  about 190 s to 11 s, and a three-wavelength call from about 570 s to
  28 s. With an output every 10 minutes, the hour costs 209 s instead of
  2,578 s; with no diagnostic it costs 61 s. Eight wavelengths in one call
  on such a domain used to hold over 13 GB of host memory. The
  representative wavelengths are now solved in groups bounded by
  `WOOF_RADTRAN_GROUP_BYTES` (default 2 GiB), with the same words: 6,000
  columns x 8 wavelengths peak at 2.6 GB instead of 3.6 GB.
  Each block of columns runs every wavelength of a group with one cache of
  what does not depend on wavelength. That cache holds the layer air and
  gas columns, the cloud radius selections and the REPTRAN pressure,
  temperature and vmr brackets. All eight REPTRAN lookups share one grid,
  so one bracket serves every species looked up without a vmr (H2O, looked
  up with its vmr, takes its own vmr-interpolated bracket). The blocks write straight into
  the solver's arrays, and the AFGL splice is written in place. The words
  are unchanged: 30,000 columns x 8 wavelengths take 15 s instead of 23 s
  on the A6000 host. Night columns are dropped before the AFGL splice,
  cloud and aerosol preparation, not only before the solve, and every day
  column keeps the AFGL levels of the whole set (the same words). That
  saves about 0.35 s of a 4.1 s call at 100,000 columns with 30 % in
  daylight (M3 Max).
  The RRTMG band and g-point output (`rrtmg_spectral_output = "bands"` or
  `"gpoints"`) keeps only the surface and top levels it publishes, with the
  same words: a 12,000-column shortwave call with g-points fell from about
  2.9 s to 0.3 s on an A6000, and with bands from about 0.5 s to 0.15 s.
  Multi-scale Kain-Fritsch (`cu_physics = 11`) launches 256 columns per SM
  instead of 64, so both of its resident blocks stay busy: `mskf_column`
  fell from 32.5 to 11.0 ms per step on Edinburgh 3 km (A6000),
  byte-identical.
  Hex: the composite step's scheme health checks (microphysics, YSU,
  Grell-Freitas, Noah-MP's surface identity, the lagged heating rate and
  the radiative-carrier finiteness) record on the device and are read
  once, just before the step commits, so a failure is still refused and
  rolled back by the step that produced it (Edinburgh 3 km on Metal: 257
  to 247 ms per step, byte-identical; `WOOF_HEALTH_READBACK=immediate`
  restores the per-check reads).
- **Lakes.** `lake_state_source = "era5"` or `"auto+era5"` (opt-in) starts
  lakes outside GLSEA from ERA5's FLake lake state (keyless, Google ARCO),
  including the CLM lake column's profile, instead of the forcing's skin (on
  the GFS route, the nearest GFS water: 286-291 K for the Alpine lakes on
  2026-11-09). The forcing's ice analysis stays in charge of which lakes are
  frozen. Against Eawag's Simstrat on six Swiss lakes, the error of the
  starting lake temperature falls from 4.5 K (warm) to 2.7 K (mostly cold),
  but ERA5 is worse at Constance and Maggiore
  ([docs/lake-state-initialization.md](docs/lake-state-initialization.md)).
  `woof go --lakes clm` turns on WRF's CLM lake model with WPS bathymetry and
  this initial state on every domain in one opt-in; `--lakes ic` sets only the
  initial state ([docs/lakes.md](docs/lakes.md)).
  `[static.highres] fields = "lakes"` (opt-in) takes the lake extent from
  CGLC-MODIS-LCZ 100 m land cover. WPS's 30-arc-second lakes cover glacier
  and rock and sit about 1 km north of the shoreline: on the Alps 1 km
  domain, 480 lake cells above 2000 m became 8 real reservoirs
  ([docs/high-resolution-static-geography.md](docs/high-resolution-static-geography.md)).
  **Recommended on the WRF route where lakes matter:** `--lakes clm` with
  `fields = "lakes"` (both stay opt-in). On the Alps 1 km case, lake-shore
  T2 bias/RMSE went from +1.66/2.34 K to +0.43/1.49 K and 10 m wind RMSE
  from 1.86 to 1.44 m/s (25 SMN stations, 24 h). Caveats
  ([docs/lakes.md](docs/lakes.md#recommended-settings)): the Great Lakes cool
  2-3 times as fast as GLSEA (WRF's CLM lake), lake-effect cores are
  over-intense (50 mm FSS .81 to .70 at 9 km on Buffalo, Nov 2022),
  `woof go` downloads `lake_depth` itself (2.4 MB, 1.9 GB unpacked; unless `use_lakedepth = 0`), and hex
  R2 lakes start about 1.4 K
  cold.
  On the hex MPAS-suite route, `--physics config_clm_lake_backend=hex-cuda`
  (opt-in) runs WRF's CLM lake on the LAKEMASK cells of `--surface-statics`,
  bitwise to WRF's lake kernel on the same inputs. It starts from lakeini and
  is refused with SST updates, OML, restarts, checkpoints, moving regions and
  multi-rank or LTS runs
  ([mpas-physics-seam.md](components/hex/docs/manual/mpas-physics-seam.md)).
- **Space weather (opt-in, beyond WRF/MPAS).** `solar_tsi_opt = 1` drives
  every shortwave scheme with a time-varying total solar irradiance (the NOAA
  NRLSSI2 CDR with `solar_tsi_source = "nrlssi2"`, or your own CSV/netCDF)
  instead of WRF's fixed 1370 W m-2. `solar_tsi_mode = "absolute"` uses the
  measured TSI (about -0.6 % shortwave); `"relative"` keeps 1370 and adds only
  the solar-cycle and sunspot variability (TSI fell by about 2 W m-2
  between 1 and 8 May 2024). It works on the WRF route and on hex's generic column seam
  (`--physics solar_tsi_opt=1`), but not yet on hex's MPAS-native
  radiation suite, and it is refused with FARMS (`swint_opt = 2`).
  `solar_ssi_opt = 1` adds the spectral change: the NOAA CDR's daily solar
  spectrum moves RRTMG SW's source between its 14 bands (200-263 nm about
  +2.5 % in 2024 against the cycle-24 mean) with the total unchanged, for
  legacy RRTMG and RRTMG-K; NRLSSI2's 200-263 nm band follows SORCE's observed
  one (daily correlation 0.79, the same 27-day amplitude). With RADM2
  chemistry it also scales Fast-TUV's extraterrestrial flux per wavelength
  bin: solar maximum against minimum moves J(O1D) by about 0.2 % and J(NO2)
  and J(HCHO) by about 0.1 % in the troposphere. Hex's generic column seam
  takes both, bit for bit the WRF route's, and hex's MICM chemistry with
  TUV-x photolysis takes the same factor on TUV-x's 156 bins (120-735 nm).
  `woof spacewx ionosphere` computes an IRI-type climatological ionosphere on
  the GPU on a global grid: foF2, hmF2, foE, F1 and vertical TEC, from the
  CCIR/URSI maps, IGRF-14 and F10.7 (GFZ). Against the IGS GNSS TEC maps it
  has a correlation of 0.90 on 1-3 May 2024 and 0.81 during the 10-12 May
  storm, and it reads about 25 % low at this solar maximum
  (`tools/spacewx_gim_validate.py`). `--storm` adds a storm-time foF2
  correction, woof's own fit of filtered ap on CODE GNSS maps (2015-2023).
  In the 2024 storms it raises the pattern correlation with GNSS TEC, and
  it lowers the storm-hour foF2 RMSE at 12 ionosondes from 2.46 to 2.14 MHz.
  It overshoots the May 2024 negative phase.
  `spacewx_ionosphere_opt = 1` writes the
  same ionosphere into every WRF-route history frame (`IONO_TEC`,
  `IONO_FOF2`, `IONO_HMF2`, `IONO_NMF2`, `IONO_FOE`) without changing the
  forecast, and `spacewx_ionosphere_opt = 2` applies the storm-time
  correction there too (plus its ratio `IONO_STORM`), from GFZ's ap with the
  latest observed ap persisted past the start; hex's generic column seam
  writes the same rows per cell, hex's MPAS v8.4.2 suite route writes them per
  cell from `--physics` (the aurora rows too), and streamed domains refuse both
  for now. `woof spacewx aurora`
  places the auroral oval (Starkov 1994 on the IGRF-14 dipole) on the GPU from
  observed Kp, OMNI2 AL or the solar wind (NASA OMNI2 or NOAA SWPC real time,
  through a Kp fit with test RMSE 0.73); `spacewx_aurora_opt = 1` writes that
  oval (`AURORA_ZONE`, `AURORA_DIST`, `MAG_LAT`, `MAG_LT`) into every WRF-route
  history frame from the start's Kp (on hex's generic column seam too, per
  cell). `woof spacewx aurora --energy` adds the
  precipitating electron and ion energy and number fluxes and the hemispheric
  power of OVATION Prime 2010 (a port of NOAA's public-domain IDL code, which
  reproduces its test outputs; the coefficients are fetched and pinned), and
  the Knipp et al. 2004 Joule power from OMNI2's PC(N) and Dst. Against POES
  TED electron flux over the 24-25 Oct 2011 storm, the per-pass correlation is
  0.48-0.61 and the model reads about 2.5 times higher. `spacewx_aurora_opt =
  2` adds OVATION's electron energy flux, mean energy and ion energy flux
  (`AURORA_EFLUX`, `AURORA_EMEAN`, `AURORA_IFLUX`) to those history frames,
  from the solar wind at the start, and `spacewx_aurora_opt = 3` also the
  Knipp et al. 2004 northern Joule power (`JOULE_POWER`, GW; one number held
  from OMNI2's PC(N) and Dst at the start, not a local heating rate).
  `woof spacewx particles` turns the GOES
  integral proton flux (OMNI2 or SWPC real time) into polar-cap ion-pair rates
  and NOx/HOx production during solar proton events (Bethe stopping in air,
  35 eV per ion pair). Against the CMIP7 rates, the ratio is 0.4-1.0 at 0.1-100
  hPa. `spacewx_epp_opt = 1` adds that NO, OH and HO2 to RADM2 chemistry in
  the polar caps, from the GOES flux at the start; on hex, the same keys in
  the chemistry config's `wrf_chem` block do it on both routes, word for word
  the WRF path's. With `--electron-energy-flux`
  and `--electron-mean-energy` (OVATION's eflux and mean energy) the command
  adds the auroral electrons' ion-pair rates and NOx (Fang et al. 2010). They
  deposit above about 85 km, so they are a diagnostic, not a forecast source
  ([docs/spacewx.md](docs/spacewx.md)).
- **A plan, not yet code, for gradients and hybrid physics.**
  [docs/dev/DIFFERENTIABLE-HYBRID-PLAN.md](docs/dev/DIFFERENTIABLE-HYBRID-PLAN.md)
  studies gradients of a forecast's fit to observations and opt-in
  neural-network column physics. woof has no adjoint or autodiff code today.
  The plan starts with adjoint-free sensitivities (finite-difference and
  ensemble), then one differentiable column scheme, NN physics through the
  existing column seams, and later a WRF dry-core adjoint checked against
  WRFPLUS. Default runs are not changed.
  The first tools exist (experimental, reached only through `woof grad`):
  `woof grad obs` reads a forecast window's surface, upper-air, aircraft
  and radio-occultation observations from the bkr/obs stores, and
  `woof grad cost` scores a WRF or hex history against them as an
  error-weighted observation cost `J_o`, pairing each observation with the
  nearest history frame and freezing the accepted set so perturbed runs
  are scored on the same observations. `woof grad direction`,
  `woof grad perturb` and `woof grad fd` give finite-difference
  sensitivities of `J_o`. They write a perturbed restart state, continued
  with `woof go --restart` (WRF route) or `--restart-from` with
  `--da-analysis` (hex), and turn the scored plus and minus runs into a
  derivative, per scale and per observation type. `woof grad members`,
  `woof grad ens`, `woof grad letkf` and `woof grad efso` give ensemble
  sensitivities and each observation's impact on the forecast cost
  (EFSO), using the LETKF's own weights ([docs/grad.md](docs/grad.md)).
  An opt-in `--mcica-seed-mode cell` (`woof go`, hex forecast door, or
  `WOOF_MCICA_SEED_MODE`) freezes RRTMG's McICA cloud seeds to a hash of column
  and call time, removing the +-5 J seed noise that one pressure ulp adds to a
  leg; the default keeps upstream's pressure-digit seeding byte for byte, and
  RRTMGP refuses the mode.
  On a regional case, the finite-difference and ensemble sensitivities
  agree in sign and pattern on the WRF route at 9 km and at 3 km, though
  ES and EFSO at the most-correlated points overstate magnitudes by about
  2-8x; an independent member set is within about 0.3-3x. On hex they are
  dominated by noise at 12 km as well as at 3-5 km, so the problem is the
  route, not the resolution. Later gradient work may use 3 km on the WRF
  route, and hex gradient work waits on a diagnosis (plan, gate G0 and
  P0.5).
  `woof.grad.bridge` hands woof's arrays to PyTorch and back: DLPack with
  no copy on CUDA (with a stream handoff), no copy from Metal into
  torch-MPS and one device copy back, and a refusal for any unnamed copy or
  any shared view that autograd could see overwritten
  ([docs/grad.md](docs/grad.md#the-tensor-bridge)).
  `woof.grad.kessler` is WRF's Kessler warm-rain scheme written in PyTorch,
  so autograd gives its tangent-linear and adjoint for a column batch. It
  is checked against woof's own `kessler_column` kernel on the WRF oracle
  fixture, passes the dot-product test (float64 to 1e-15, float32 to 1e-7
  on CUDA) and runs as a column model on the NN seat. Its backward pass
  costs 8x the kernel eager and 4-5x compiled on CUDA, so the 3x cost gate
  is not met. `tools/grad_kessler_g1.py` measures all of this
  ([docs/grad.md](docs/grad.md#a-differentiable-scheme-kessler)).
  The WRF dry-core adjoint has started, kernel by kernel
  ([docs/dev/P3A-WRF-DRYCORE-ADJOINT.md](docs/dev/P3A-WRF-DRYCORE-ADJOINT.md)).
  `woof.grad.wrfplus` has the tangent-linear and adjoint of the acoustic
  `advance_uv` in binary64 (CUDA, host) and binary32 (CUDA, Metal). They
  pass the dot-product test to 1e-15 and agree with WRFPLUS 4.7.1's
  `g_/a_advance_uv` to about 1e-16, using a driver in
  `tools/wrfplus_oracle`. `advance_w` with `calc_p_rho` (woof's
  `advance_w_phi`) follows the same pattern: dot-product test to about
  1e-14, WRFPLUS `g_/a_advance_w` and `g_/a_calc_p_rho` to about 1e-15,
  and the float build reproduces the strict primal's words, with map
  factors too (`advance_w_phi_msf`, against WRFPLUS with msftx = msfty =
  msft). The w damper
  (`damp_opt=3`) is refused, since WRFPLUS has none. `calc_coef_w`
  (woof's `calc_coefs`: c2a and the implicit w factors) matches WRFPLUS
  `g_/a_calc_coef_w` to about 1e-15. `advance_mu_t` (woof's
  `advance_mu_th` and the specified-frame mu/theta/muts update) matches
  WRFPLUS `g_/a_advance_mu_t` and `g_/a_spec_bdyupdate` to about 1e-15,
  with map factors. `small_step_prep` with `calc_p_rho` and
  `small_step_finish` (woof's `small_step_init_*`/`_finish_*`) match
  WRFPLUS within the oracle test's 1e-12, and a chained test checks whole acoustic
  substeps. The RK-stage glue (`calc_p_rho_phi`, `rk_step_prep`'s
  `couple_momentum` and `calc_ww_cp`, `rk_addtend_dry`, `set_w_surface`)
  matches WRFPLUS within 1e-12 and, in strict mode
  (`GPUWM_WRF_EXACT=1`), the primal word for word. A
  chained test runs one whole RK stage with the big-step tendencies held.
  Flux-form advection (woof's strict `flux_div_{scalar,u,v,w}`, WRF
  `advect_scalar/u/v/w`) has its TL/AD for periodic, open and specified
  boundaries, map factors, horizontal orders 3-6 and vertical orders 2-6.
  It matches WRFPLUS `g_/a_advect_*` to about 1e-16 on the points where
  woof and WRF compute the same thing (see the design note), and the float build
  reproduces the strict primal's words. The big-step pressure gradient
  and buoyancy (woof's strict `slow_pgf` and `slow_buoyancy`, WRF
  `horizontal_pressure_gradient` and `pg_buoy_w`) follow, with the lid,
  open and specified boundaries, map factors and hybrid levels, and so
  does the geopotential tendency (`slow_geopotential`, WRF `rhs_ph`) for
  every `h_sca_adv_order`, the Coriolis and curvature terms
  (`coriolis_curvature`, WRF `coriolis` and `curvature`) on rotated
  grids, the `w_damp` vertical-velocity limiter, and the lateral-boundary
  relaxation and specified zone (woof's strict `state_relaxation_multi`,
  WRF `relax_bdy_dry` and `spec_bdy_dry`, with the boundary tables'
  adjoint as an output) and its end-of-step and acoustic frame updates
  (`finalize_state_field` and `advance_specified_phi_w`, WRF
  `spec_bdy_final`, `spec_bdyupdate_ph` and `zero_grad_bdy`). In binary64 they match WRFPLUS to about 1e-15
  on non-hybrid cases (WRFPLUS has no hybrid coordinate; the exclusions
  are listed in the design note). The comparison found a lid bug in
  WRFPLUS's own pressure-gradient adjoint and, in its `rhs_ph`, a heap
  overrun in the order-3/4 adjoint and a short interior loop on bounded
  axes, all described in the design note.
  The binary32 adjoint costs 3.2-5.0x the forward acoustic kernels,
  0.85-3.2x the glue, 3.3-4.5x advection, 1.3-4.4x the big-step
  tendencies and 2.5-3.7x the lateral boundaries. The kernels chain into
  one strict dry step (`woof.grad.wrfplus.model`, periodic or with
  specified lateral boundaries, with Coriolis, curvature and map factors;
  the boundary tables are inputs whose adjoint it returns) whose binary32 base state reproduces
  `dycore.step` word for word, and whose binary64 dot-product and
  finite-difference tests pass. Measured at 200x200x49 on an A6000, the
  step's adjoint sweep costs 3.3x the forward step, and 4.8x with its
  taped forward recompute through a base-only build of the kernels
  (details in the design note). A window driver
  (`woof.grad.wrfplus.window.Window`) runs the forward, TL and adjoint
  over N steps with checkpoints, and takes adjoint forcing at
  intermediate states for 4D-Var. Its TL can start from part of the
  state, with the rest zero, as WRFDA's TL start does
  (`Window.tl(dx0, partial=True)`). It can also linearise about a
  supplied trajectory instead of its own dry rerun, as WRFPLUS does
  about the nonlinear run's states (`Window.forward(trajectory=...)`).
  The step also covers a moist state without microphysics (qv, qc and
  qr: calc_cq, the moist pressure gradient, buoyancy and diagnosis,
  sumflux and the non-PD scalar update, and qv's boundary rows with
  flow_dep_bdy for qc and qr). The forecast path does not import any of
  it.
  On a dry state the step also covers WRF's diff_opt 1 / km_opt 4
  horizontal mixing (`woof.grad.wrfplus.diffusion`, bound into the step
  on periodic and specified domains, word for word against
  `dycore.step`; the moist rows are next): the deformation (WRF
  `compute_diff_metrics` + `cal_deform_and_div`) and `smag2d_km`'s
  xkmh/xkhh with the 10 * mlen
  cap. Their binary32 base reproduces woof's strict `wrf_smag_deform` and
  `wrf_diff_opt1_km4` word for word, and they match WRFPLUS to about
  4e-16, except at the top level of a periodic seam, where WRF keeps a
  cold-zero slope and woof the true one. The rows that use them
  (`hd_tl/hd_ad`: WRF `horizontal_diffusion` for u, v, w and scalars,
  `horizontal_diffusion_3dmp` for theta) reproduce woof's strict
  `wrf_diff_opt1_horizontal` and match WRFPLUS to about 1e-16.
- **LETKF host solve on every core by default (Linux).** An analysis on host
  arrays (no `solve_namespace`, the dense reference route) now runs its
  chunks on forked worker processes once the work is large (2x10^8
  gridpoint-slot pairs), as the host-staged route already did. Workers are
  capped at 64, by the chunk count and by available memory (psutil, bounded
  by the cgroup v2 `memory.max`), because each worker holds one chunk's
  scratch. The automatic mode forks only on Linux. On macOS an analysis
  stays in one process unless `host_workers` or `WOOF_DA_HOST_WORKERS` asks
  for workers.
- Every host chunk solve runs with one BLAS thread, so the increments are
  the same bytes at any worker count and on any box. Before, a
  multi-threaded BLAS made them depend on the core count by up to 1e-14, so
  large host analyses change by that much once.
- The trade-off: `host_workers=1`, and mid-size analyses below the
  threshold, now solve with single-threaded BLAS. That is slower than a
  threaded BLAS but deterministic. Apple's Accelerate is not covered:
  threadpoolctl cannot limit it.
- **Variational data assimilation (planned; first pieces landing).**
  [docs/dev/WRFDA-3DVAR-PLAN.md](docs/dev/WRFDA-3DVAR-PLAN.md) scopes
  WRFDA-style 3D-Var on the GPU: WRFDA's own `be.dat` and control-variable
  transforms, a conventional-observation 3D-Var checked against compiled
  WRFDA 4.7.1, background-error statistics from woof forecasts, and hybrid
  3DEnVar with the LETKF members. Radar, GOES and radiances come later;
  radiances need an opt-in CRTM. It also scopes opt-in gradient and
  neural-network options inside the DA. 4D-Var waits for the adjoints in
  the plan above. So far `woof.da.var` reads WRFDA's `be.dat` and has a
  binary64 reference of the control-variable transform `U` and its
  adjoint; on WRFDA's tutorial case both equal a compiled WRFDA 4.7.1 word
  for word. `DeviceU` runs `U` and `U^T` as CUDA or Metal kernels: the
  binary64 oracle tier gives the reference's words, and the binary32
  production tier gives the same words on CUDA and Metal under strict
  numerics. It also reads OBSPROC's `obs_gts` ASCII the way WRFDA does:
  time window, domain check, wind rotation, `q` from `rh`, and the
  per-type error rules. On the tutorial case it keeps the same
  8,913 of the file's 9,061 reports as WRFDA, in the same order, with the
  same grid indices and errors. `VarCost` and `minimise_cg` add the cost
  function, its gradient and WRFDA's conjugate-gradient minimiser in
  WRFDA's operation order, printing WRFDA's iteration log.
  It computes WRFDA's innovations, background
  check and `Jo` for the surface, sounding, aircraft, wind-profile, AMV,
  scatterometer, GNSS precipitable water, GNSS-RO refractivity and satem
  thickness types, with the same accepted observations as WRFDA and its
  initial cost `J_0` on the tutorial case, and their linearised operators
  and adjoints.
  `woof var --be be.dat --obs ob.ascii --first-guess wrfinput_d01 --out
  BUNDLE` (experimental) runs the whole 3D-Var and writes a
  `wrf-analysis-bundle/1` that `woof run --analysis BUNDLE` starts from
  ([docs/var.md](docs/var.md)). On the tutorial case it takes the same 29
  iterations as WRFDA, with every logged cost and gradient within 1e-14
  relative, and its analysis rounds to WRFDA's `wrfvar_output` word for word.
  `--tier oracle` runs on CUDA or Metal with the same words;
  `--tier production` (binary32) gives one digest on both. The production
  tier's 50-iteration minimisation on a synthetic 200x200x50 case with
  ~10^5 observations takes 3.1 s on an M3 Max (Metal) and 3.2-4.5 s on a shared
  A6000 (targets 20 s and 5 s; the tutorial case: 1.4 s and 2.3 s), with
  H' and H'^T on the device as an ordered gather (no atomics).
  Hybrid 3DEnVar (experimental; `woof var --ensemble-member ... --je-factor`,
  or `woof.da.var.hybrid` from Python) adds WRFDA's: the alpha control variable with its `alpha_corr_scale`
  recursive filter and vertical localisation (`alpha_vertloc_opt`), blended
  with the static `B` through `je_factor`/`jb_factor`, from WRFDA `ep` files
  or ensemble members already on the device. On the tutorial case with ten
  members it follows a compiled WRFDA 4.7.1 hybrid run's 50 iterations to
  3.5e-14 relative; the oracle tier gives the reference's words on the
  device and the production tier one digest under fast and strict numerics.
  The `woof var` door makes the perturbations from members' history files
  as WRFDA's `gen_be_ep2` does (to 2.8e-13 K in `t`; the rest word for
  word). Its analysis matches WRFDA's `wrfvar_output` to O7's tolerances:
  max |difference| at most 4.7e-5 of the RMS.
  `woof.da.var.pairing` (Python API) pairs it with an LETKF cycle: the hybrid's
  ensemble is the LETKF's forecast ensemble, and the LETKF's analysis members
  are recentred on the variational analysis. Tested on synthetic grids only;
  no cycling skill claim, and writing the members back is the controller's.
  FGAT (experimental; `woof var --fgat-obs obNN.ascii --fgat-first-guess fgNN`
  per time slot) compares each observation with the first guess at its
  slot's time, as WRFDA's `num_fgat_time` does. On a 7-slot 12-18Z case
  with first guesses from a WRF 4.7.1 forecast, it takes WRFDA 4.7.1
  FGAT's 31 iterations to its final cost, and the analysis meets O7's
  tolerances.
  `woof.da.var.gen_be` (experimental, Python API, host binary64) makes a
  `be.dat` from woof or WRF history files (NMC 24 h minus 12 h pairs, or
  ensemble members minus their mean): WRFDA's gen_be stages for
  `cv_options` 5 and 7 with `bin_type` 5 only. On WRFDA's own gen_be sample
  forecasts it reproduces WRFDA 4.7.1's gen_be to 1e-7 relative (cv5; the
  residual is WRFDA's `pi = 3.1415926` sine-transform tables in psi and chi)
  and to 3e-12 for cv7, and WRFDA's 3D-Var runs the woof-made file as its own.
  Section 11 plans geostationary imager
  radiances for initial conditions: Google's SatWeave mosaic and the native
  GOES ABI, Himawari AHI and Meteosat imagers, through an opt-in CRTM into
  the LETKF first and into 3D-Var later. Default runs are not changed.
- **SatWeave imager reader (opt-in; in progress).** `woof fetch satweave`
  subsets and caches Google's SatWeave geostationary mosaic (17 bands at
  0.1°, 2016 onward; CC BY 4.0, attribution kept in every provenance
  record) over anonymous HTTPS. It reconstructs each pixel's satellite and
  view geometry, which SatWeave does not publish, from band availability
  and the sectors' sub-satellite longitudes
  ([docs/satellite-imagers.md](docs/satellite-imagers.md)). `woof fetch geo`
  reads native GOES ABI L1b (GOES-16 to 19) from the `bkr/geo` virtual
  Icechunk stores on Source Cooperative, which reference NOAA's own files.
  It keeps NOAA's calibration and quality flags and gives the true per-pixel
  geometry. The same door reads Himawari-8/9 AHI (`--satellite himawari9`)
  from NOAA/STAR's sectorized CMI stores, which hold brightness temperature
  and reflectance with no quality flag, and Meteosat SEVIRI (`--satellite msg`,
  `rss` or `odegree`) and MTG FCI (`--satellite fci`) from Satpy-calibrated
  stores. GK-2A is refused, and FCI cannot be simulated: CRTM v2.4 has no AMI
  or FCI coefficients. `woof.rtm` adds an opt-in clear-sky
  infrared operator. It uses JCSDA CRTM v2.4+ (CC0), which you build
  yourself with `python -m woof.rtm.crtm_build`, together with CRTM's own
  coefficients. It gives brightness temperatures and Jacobians from woof
  columns or from an isobaric analysis. `woof.da.obs_radiance` (opt-in) turns a scene
  into LETKF batches. It thins the scene, runs H(x) per member through
  CRTM, screens for clear sky, applies a static bias correction and a gross
  check, and places each observation at its weighting-function peak.
  `woof var --radiance-scene DIR` (opt-in) assimilates clear-sky imager
  radiances in 3D-Var. It uses WRFDA's channel selection and errors, its
  clear-sky QC, GSI thinning and the CRTM K-matrix, and adds VarBC. The bias
  coefficients join the control vector with WRFDA's predictors,
  preconditioning and `VARBC.in`/`VARBC.out` files
  (plan section 13). The adjoint holds to 1e-13 and the gradient check to
  1e-6, and the production-tier words are the same under fast and strict
  numerics. The microwave sounder (WIS2) records plug into the same term.
- **WIS2 observation reader (opt-in; in-situ families first).**
  `woof obs wis2 fetch` reads the WIS2 Global Cache archive
  (`bkr/obs/wis2.icechunk` on Source Cooperative, anonymous). It covers
  land stations (BUFR and SYNOP TAC), ships and buoys, radiosondes,
  aircraft and wind profilers in a time window and box. It applies the
  store's quality flags and keeps the latest correction of each report.
  It merges each sounding's TEMP parts, and it checks every part against
  the rest of its launch. A part whose pressures are ten times too large is
  detected and refused; the US Part D (`TEMP_DD`) objects in the store
  have this fault. Errors are WRFDA's OBSPROC table (`obserr.txt`). It
  writes a `gpuwm-obs.table.v2` table for the LETKF (`--table`) and
  OBSPROC's `obs_gts` ASCII for `woof.da.var` (`--obs-gts`). Profilers go
  only to `obs_gts`, because the LETKF's point operators are
  pressure-level only. The LETKF uses aircraft rows (WIS2, ACARS and
  PrepBUFR alike) only with `woof local-da --assimilate-aircraft` (opt-in);
  ship rows have no operator yet. WIS2 has no US AMDAR. Satellite point
  winds are opt-in families: `--families amv,scat` reads atmospheric
  motion vectors (WMO template 310077 only; quality index without forecast
  at least 80; geostationary, plus polar AVHRR, MODIS and VIIRS) and ASCAT
  10 m winds (the selected ambiguity; land, ice and cell-quality flags
  refused), with WRFDA's SATOB and QSCAT errors and an optional
  `--thin-km`. AMVs go to both outputs. Scatterometer winds go to
  `obs_gts`, and to the table under a label the LETKF has no operator for
  yet. WIS2 has no GOES AMVs. `--families ro` reads GNSS-RO refractivity
  profiles (template 310026: COSMIC-2, FY-3, PAZ and others relayed by JMA
  and CMA; EUMETSAT's products carry bending angles only and are counted,
  not used). Each level sits at its tangent point, 0-30 km, with WRFDA's
  gpsref error and the observed-gradient (super-refraction) checks of
  `da_qc_gpsref`. Profiles go to the table (`ro_refractivity_tangent_point`,
  the LETKF's existing refractivity operator) and to `obs_gts` as FM-116.
  `woof obs wis2 radiances` reads AMSU-A, MHS and ATMS footprints (Metop-B
  and -C, S-NPP, NOAA-20 and -21) to one `.npz` per instrument. Sensors
  CRTM has no coefficients for are counted and refused.
  `woof.da.obs_mw` (opt-in) turns them into LETKF batches. It uses CRTM
  v2.4 clear sky over open sea, WRFDA's channels, errors and screens
  (`da_qc_amsua`, `da_qc_mhs`, `da_qc_atms`: scan edges, scattering
  index, background cloud, 15 K and 3 sigma), and a static bias
  correction. Each channel sits at its Jacobian peak. Metop-B's channel 15
  is dead, so its scattering check uses the channel 1-2 cloud liquid
  retrieval instead. The 3D-Var radiance term comes later
  ([plan section 12](docs/dev/WRFDA-3DVAR-PLAN.md)). In one real case
  (GEFS ensemble, Europe and the North Atlantic, 2026-10-28 12Z) the
  WIS2 satellite and aircraft families reduce their own O-A, and
  MHS and ATMS departures fall on columns that were not assimilated as
  well. Against held-out radiosondes the analysis is neutral to slightly
  better without AMVs. With AMVs, upper-level winds are 3-4 % worse
  (plan 12.8 item 6). The 31-member solve runs on host-staged arrays
  (`solve_namespace=np`) with 12 forked workers, 237k active points: the
  transform took 65 s, against 20 minutes in one process at 4 members.
- **Neural-network column physics (opt-in, inference only; in progress).**
  `woof.nn` loads a column model (a manifest with weights hash, units,
  normalisation and trained input range) and runs it with PyTorch on CUDA
  (DLPack, no copies; `recast-woof[nn-torch]`, never a base dependency) or
  MLX on Metal (PyTorch artefacts also run there, on torch-MPS). A safety layer refuses, clips or falls back to the physics
  scheme outside the trained range, and limits outputs, keeps moisture
  positive and can fix column budgets. `woof run`/`woof go --nn NN.toml`
  with `mode = "tap"` records a WRF-route scheme's column inputs and
  physical-rate tendencies to Zarr or Icechunk as training data, leaving the
  forecast byte-identical. `mode = "residual"` or `"replace"` applies a
  model at the same seat through the couplers' own path, with a fallback to
  the scheme outside the trained range. Such runs are not WRF-identical:
  `GPUWM_WRF_EXACT=1` refuses them, and checkpoints bind the artefact.
  `python -m woof.nn.train TAP OUT` trains a small MLP on a tap with NumPy
  alone, and `tools/nn_seat_cost.py` measures the seat's per-step cost; a
  worked 24 h, 9 km example (PBL emulator, CUDA and Metal) is in
  [docs/nn-physics.md](docs/nn-physics.md). `woof hex forecast --nn`
  puts the same seat on the hex MPAS route. It taps or adjusts the held
  physics source (`scheme = "total"`, theta and moisture rates) before
  SPPT, IAU and nudging. The tap uses interior cells only, an applying
  model skips the specified zone, and the hex carrier binds the artefact.
  `woof.nn.hex_backend` is an example `woof.hex.rows` backend that adds a
  model to a column batch.
  On hex, 5-day emulator runs on a uniform and a variable-resolution mesh
  were stable on CUDA (the uniform one also on Metal) and kept the mass
  budget. Trained on six cycles over four seasons (P2.6), the model
  passes the plan's G2 gate. T2 is not worse than the physics run at 95 %
  on either mesh, and better on the uniform mesh on CUDA (-0.06 K).
  Most column calls (94-97 %) still fell back to the schemes in the
  evaluation week, because its stratosphere lay outside the training
  range. Training on the lowest 44 levels (P2.7) cuts the fallback to
  21 % on a fresh cycle (66 % on the variable mesh, whose `dx` is out of
  range), but T2 is then no better than the physics run (+0.04 K, CI
  spanning zero): no evidence of skill. Results and per-step costs are in
  [docs/nn-physics.md](docs/nn-physics.md).
- **Continuous integration and releases:** GitHub Actions on Linux and
  macOS, with a self-hosted CUDA job. Hosted runners skip a few heavyweight
  tests by name, and main's full run is periodic. Every merge to `main`
  tags and writes release notes; PyPI gets the newest `metal-woof` tag once a day
  ([docs/dev/ci.md](docs/dev/ci.md)).
- **256-level columns for WSM3, WSM5, WSM7, WDM5 and WDM7.** These five
  microphysics units (`mp_physics` 3, 4, 24, 14 and 26) keep their level
  arrays per thread, sized by a compile-time `*_KMAX` tier, and the ladder
  ended at 80 levels. It now has 128 and 256 tiers: a column of 80 levels or
  fewer compiles the unchanged source (every output word is the same), a
  deeper column compiles the same source at the next tier, and preflight
  prices the deeper per-thread frame (WSM3 and WSM5 22,560 B, WSM7 28,720 B,
  WDM5 36,896 B and WDM7 38,960 B at 256, against 7,000-12,000 B at 80).
  Past 256 is refused by name. Hex runs these through the generic column
  seam and follows the same bound. WSM6 and WDM6 already ran 256 levels on
  a column workspace; the five here keep the per-thread arrays, so a
  256-level run reserves several GiB of local memory on a large card
  ([the frame table](docs/kernel_local_memory_bounds.md)). On the hex
  option matrix's 149-level case the WSM3, WSM5, WSM7, WDM5 and WDM7 rows
  run on CUDA.
- **256-level columns for the GSFC shortwave scheme.** `ra_sw_physics=2`
  was refused above 128 layers: its per-column arena (every array's extent
  and offset) and the entry point's `zero[]` were literals for a 128-layer
  column. They are now expressions of `GSFC_KMAX`, whose source default is
  128, so a column of 128 layers or fewer compiles the unchanged-meaning
  source (the oracle fixture words and a raw-bit comparison at nz = 49 and
  128 are identical) and a deeper column adds `-D GSFC_KMAX=256`. The
  local frame stays 0 B at both tiers; the arena grows from 32,677 to 62,885
  words per column (535 MB to 1.03 GB for the 4,096-column chunk) and
  preflight prices it by tier. Past 256 is refused by name. Hex does not
  offer `ra_sw_physics=2`, so only the WRF route is affected
  ([the record](docs/kernel_local_memory_bounds.md)).
- **256-level columns for GFDL Eta radiation.** `ra_lw_physics` and
  `ra_sw_physics` 99 were refused above 81 layers. That was a limit of this
  port, not of WRF. Stock `module_ra_gfdleta.F` sizes its model-level arrays
  from `kte`, and its `NL=81` is the ozone climatology's fixed pressure grid,
  which it interpolates to any model grid. The port's transcription gives
  every per-column array fixed storage, generated at 81. The same lowering
  is now regenerated at 256 (`tools/gfdleta_wrf461_oracle/build_tier.py`)
  into `gfdleta_column_l256` and `gfdleta_co2_init_l256`:
  - A column of 81 layers or fewer runs the unchanged units.
  - A deeper column runs the 256-layer units.
  - Past 256 is refused by name.

  Both tiers are tested word for word:
  - Forcing the deep tier onto the 12- to 60-layer WRF fixtures reproduces
    every word.
  - A 256-layer column matches the byte-unmodified WRF 4.6.1 module run at
    256 layers in strict numerics, for every CO2/ozone table word and every
    output word.

  The deep column workspace is 1.30 MB per slot, against 234 KB. Preflight
  prices it by tier, and `column_chunk` is the lever at both tiers.
- **256-level columns for Fu-Liou-Gu radiation.** `ra_lw_physics=7`
  (with `ra_sw_physics=7`) was refused above 100 layers, stock WRF's
  `nvx = 100` scratch bound. The generators now emit that bound as the
  macro `FLG_NVX`, whose source default is the stock 100: every column of
  100 layers or fewer compiles stock WRF's bound, and its SASS is identical
  to the previous kernel's on sm_86, sm_100 and sm_120 in both numerics
  modes. A 101 to 256 layer column adds `-D FLG_NVX=256` and is checked
  word for word in strict mode against WRF's `module_ra_flg.F` with only
  `nvx` raised to 256 (the patch, its hash and why it changes no
  arithmetic are in `woof/data/wrf_radiation/flg/PROVENANCE.md`). In the
  forecast column the per-level storage was already a global workspace
  sized by the column, so the local frame does not grow (928 B on sm_86 at
  both tiers); the workspace grows from 30,301 to 77,101 words per column
  at 100 and 256 layers, and admission prices it per domain. Past 256 is
  refused by name. The hex option-matrix row runs on CUDA at 149
  levels ([the record](docs/kernel_local_memory_bounds.md)).
- **256-level hex member init and TC potential intensity.** The hex GPU
  member init (`woof/hex/init_gpu/physics.cu`) and the TC diagnostics'
  potential-intensity sounding (`tc_climate/potential_intensity.cu`) held
  per-thread columns of 128 levels and refused deeper meshes. Both bounds
  are now macros whose default is 128 (`MAX_FG`/`MAX_Z`, `PI_MAX_LEV`): a
  mesh of 128 levels or fewer compiles the unchanged text (the PI source
  preprocesses to the old literal source, and the deep builds reproduce the
  shipped words at 64 levels), and a deeper mesh prepends a 256 define.
  Past 256 is refused by name. A 200-level resampling of the PI reference
  sounding gives the reference PI to within 0.4 m/s.

### Licensing and opt-ins

The default build and the published wheels (`metal-woof`, `recast-woof`)
carry none of the following. Each is off until selected:

- **LGPL glibc 2.39 libm** (`WOOF_GLIBC239_LIBM64`, `WOOF_GLIBC239_TRIG64`):
  glibc's binary64 words on the GPU and host. It lifts some MALI,
  MPAS-Ocean and MPAS-Seaice RKIND 8 refusals, and E3SM's
  atmosphere-ocean flux.
- **NVIDIA-derived `powf`**: CUDA's transcription for Metal strict numerics.
  The default is WOOF's own correctly rounded `powf`.
- **WRF-Hydro** (`WOOF_ENABLE_WRF_HYDRO=1`), **NEMO**
  (`WOOF_ENABLE_NEMO=1`) and **WRF-Chem PNNL** (`WOOF_ENABLE_WRFCHEM_PNNL=1`),
  under `optional/`.
- WRF routines that are not freely licensed come from your own WRF tree
  (`WOOF_WRF_SOURCE`).
- GEM-15 terrain is CC-BY-4.0 and needs its attribution.
- Building footprints from Overture, Microsoft and OSM are ODbL-1.0; a
  derived buildings file must carry their attribution.

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
  - the LTS executor's scalar kernels run without 64-bit atomics on Metal:
    the reserve census counts with two 32-bit atomics and a carry (the same
    words as CUDA), and the binary64 receipt sums are gathered per element
    and summed in a fixed order (deterministic on Metal; CUDA's atomic sums
    vary in the last bits from run to run). So `woof hex moving` and LTS
    regional runs work on an Apple GPU;
  - the dynamics subcycle runs eager, because the facade has no CUDA graphs;
  - the MPAS WSM6 unit of the water-budget gates (binary64) once had not
    finished its cold compile after 1 h 45 min; it now compiles cold in
    about 4 minutes, but its gates have not been re-run on Metal.
- **WRF v4.8.0 Grell-Freitas-Li is refused on Metal for now.** `cu_physics = 3`
  with the default `cu_gf_version = 'gfl_873b414'` (also the hex suites'
  default cumulus) stops at setup on Metal with a message naming the
  Metal-ready `cu_gf_version = 'wrf_461'` (WRF v4.6.1 GF). Nothing switches
  to it silently. Its lane-interleaved CUDA unit does not translate to Metal
  yet; the port is in progress.
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
