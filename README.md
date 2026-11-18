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
    split-explicit barotropic stage and whole split-explicit step, and
    forward mode (clock, cold start, N steps, bitwise restart). The step
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
  - MPAS-Seaice: variational EVP internal stress, the velocity solve and
    elastic subcycles, the pre-subcycle routines, ocean stress,
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
    needs an opt-in. The elastic subcycles check their arrays once per
    solve and work out their compile options once per numerics mode:
    120 subcycles on a 9,200-cell torus take about 34 instead of 39-41 ms a step on
    an M3 Max, with the same words. Each subcycle computes the ocean
    stress coefficient and solves the velocity in one launch (a vertex's
    solve reads only its own coefficient), so the step launches 120 fewer
    kernels: about 30 instead of 33 ms on an M3 Max, the same words;
  - MALI: the shallow-ice velocity solver, masks and geometry diagnostics,
    thermal initialisation, the thermal solver and basal melt, thickness
    and tracer advection, calving (every option, and front restoration),
    subglacial hydrology and one forward-Euler time step.

  Limits: Albany's higher-order solvers are refused by name. The sea-ice
  remap computes its init geometry (frames, moments, MPAS's RBF
  coefficients) itself, and refuses halo exchanges and periodic planar
  meshes by name. The sea-ice column physics' time step (mushy vertical thermodynamics,
  shortwave), ocean KPP and GM/Redi,
  and the coupler import and export are not ported. A few RKIND 8 cases need the LGPL libm
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
- **Observation-driven corrections, each opt-in and off by default:**
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
  - On the MPAS suite route the 50 km lid needs correction RP8. MPAS
    v8.4.2's MYNN `qsat_blend` has a pole where p reaches the saturation
    vapour pressure, which happens above about 35 km. RP8 adds the
    NOAA-GSL MYNN-EDMF guards. It is on in correct mode (the route's
    default), and `--physics-correct-skip RP8` turns it off.
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
    ([TERRAIN-DRAG.md](docs/public/TERRAIN-DRAG.md));
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
    the run's own boundary-layer height since hex has no parent model);
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
- **Building morphology (opt-in input).** `woof.static.buildings` turns
  GEM-15 heights (surface minus terrain, with tree and noise filters) and
  open footprint files (Overture Maps, Microsoft, Google Open Buildings,
  OSM; GeoJSON or GeoParquet) into per-cell building statistics on the model
  grid. These are plan and frontal area indices, a height histogram, and
  WRF's NUDAPT `URB_PARAM` block. Each source's licence and attribution
  travel with the file. `woof buildings` builds the product for one
  domain of a WPS namelist. Footprints are read from a local file, not
  downloaded, and the urban schemes do not read `URB_PARAM` yet
  ([docs/public/BUILDINGS.md](docs/public/BUILDINGS.md)).
- **Resolved buildings in LES (`obstacle_opt = 1`, opt-in).** On a domain
  with spacing up to 500 m, the building product becomes porous immersed
  obstacles on the WRF route. Faces of solid cells are blocked (Brinkman
  penalisation), partly built cells get canopy drag, and walls and roofs
  get log-law stress. One fused CUDA/Metal kernel acts on the RK-stage
  momentum tendencies. A street-canyon benchmark gives the H/W = 1 vortex,
  and a cube benchmark gives the wake and horseshoe. Off, or with no
  buildings, nothing launches and runs are bit-identical. Momentum only,
  regional grid only. A real-data check over Manhattan places the model's
  buildings at their footprint heights (r = 0.99 at 25 m). In lower and
  midtown Manhattan, 50 m resolves 59% of buildings and 25 m resolves 77%;
  50 m does not open the streets.
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
  (`[output] icechunk_delete_after_commit`, `icechunk_domains`).
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
  its occupancy-bounded 16 per SM). Hex: CUDA
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
  `lake_depth` needs `woof fetch-geog` (unless `use_lakedepth = 0`), and hex
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
  forecast; hex and streamed domains refuse it for now. `woof spacewx aurora`
  places the auroral oval (Starkov 1994 on the IGRF-14 dipole) on the GPU from
  observed Kp, OMNI2 AL or the solar wind (NASA OMNI2 or NOAA SWPC real time,
  through a Kp fit with test RMSE 0.73); `spacewx_aurora_opt = 1` writes that
  oval (`AURORA_ZONE`, `AURORA_DIST`, `MAG_LAT`, `MAG_LT`) into every WRF-route
  history frame from the start's Kp. `woof spacewx aurora --energy` adds the
  precipitating electron and ion energy and number fluxes and the hemispheric
  power of OVATION Prime 2010 (a port of NOAA's public-domain IDL code, which
  reproduces its test outputs; the coefficients are fetched and pinned), and
  the Knipp et al. 2004 Joule power from OMNI2's PC(N) and Dst. Against POES
  TED electron flux over the 24-25 Oct 2011 storm, the per-pass correlation is
  0.48-0.61 and the model reads about 2.5 times higher. `spacewx_aurora_opt =
  2` adds OVATION's electron energy flux, mean energy and ion energy flux
  (`AURORA_EFLUX`, `AURORA_EMEAN`, `AURORA_IFLUX`) to those history frames,
  from the solar wind at the start. `woof spacewx particles` turns the GOES
  integral proton flux (OMNI2 or SWPC real time) into polar-cap ion-pair rates
  and NOx/HOx production during solar proton events (Bethe stopping in air,
  35 eV per ion pair). Against the CMIP7 rates, the ratio is 0.4-1.0 at 0.1-100
  hPa. `spacewx_epp_opt = 1` adds that NO, OH and HO2 to RADM2 chemistry in
  the polar caps, from the GOES flux at the start
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
- **A plan, not yet code, for variational data assimilation.**
  [docs/dev/WRFDA-3DVAR-PLAN.md](docs/dev/WRFDA-3DVAR-PLAN.md) scopes
  WRFDA-style 3D-Var on the GPU: WRFDA's own `be.dat` and control-variable
  transforms, a conventional-observation 3D-Var checked against compiled
  WRFDA 4.7.1, background-error statistics from woof forecasts, and hybrid
  3DEnVar with the LETKF members. Radar, GOES and radiances come later;
  radiances need an opt-in CRTM. It also scopes opt-in gradient and
  neural-network options inside the DA. 4D-Var waits for the adjoints in
  the plan above. woof has no variational DA today, and default runs are
  not changed.
- **Neural-network column physics (opt-in, inference only; in progress).**
  `woof.nn` loads a column model (a manifest with weights hash, units,
  normalisation and trained input range) and runs it with PyTorch on CUDA
  (DLPack, no copies; `recast-woof[nn-torch]`, never a base dependency) or
  MLX on Metal. A safety layer refuses, clips or falls back to the physics
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
