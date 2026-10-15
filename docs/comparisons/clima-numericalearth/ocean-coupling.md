# Regional ocean and atmosphere–ocean coupling: woof, CliMA and NumericalEarth

This page compares what woof, the CliMA stack and the NumericalEarth stack can do for a regional
ocean, and for coupling an ocean to an atmosphere. It also reports a small coupled experiment
that was run for this comparison. It ends with a concrete design for giving woof an interactive
ocean, with the woof code each option would touch.

**Evidence labels.** Every claim is one of three kinds:

- **documented**: a README, docstring, example or paper says it. The source is cited.
- **measured**: we ran it. The command and output are in
  [`tools/compare/ocean_coupling/`](../../../tools/compare/ocean_coupling/README.md).
- **estimated**: our engineering judgement. Effort and risk figures are all of this kind.

**Versions read and run.**

| Package | Version run (registered) | Source read | Licence (file text) |
|---|---|---|---|
| woof | `jacob/main` b4a3783a | this tree | this repository's |
| NumericalEarth.jl | 0.8.2 (tag `v0.8.2`, 5a7d131) | v0.8.2 and `main` 66112ea (2026-10-15) | MIT |
| Breeze.jl | 0.11.3 (tag `v0.11.3`, 781b641) | v0.11.3 and `main` 6eabb89 | Apache-2.0 |
| Oceananigans.jl | 0.113.6 (tag `v0.113.6`, 91038a7) | v0.113.6 and `main` fec351f (0.114.0) | MIT |
| ClimaSeaIce.jl | 0.5.12 | `main` 4e7958d | Apache-2.0 |
| ClimaOcean.jl | not run | `main` 940e551 (0.10.0) | MIT |
| ClimaCoupler.jl | not run | `main` e3355c8 (0.2.3) | Apache-2.0 |
| WRF | not run | tag `v4.8.0` | public domain |

GitHub's licence badge shows NOASSERTION for NumericalEarth.jl and ClimaOcean.jl. The licence
text in both repositories is MIT.

Upstream paths below are relative to each repository at the version in the table. NumericalEarth
paths are at `v0.8.2`; where `main` differs, the text says so. woof paths are relative to this
repository.

## Bottom line

1. **woof has no ocean of any kind** (documented in this tree). Water skin temperature is set once
   from the analysis and never changes during a run. Only three things move a water-like skin
   temperature: the CLM lake on lake cells, RUC's sea-ice column, and the global path's lake slab.
   Details are in [woof today](#woof-today).
2. **CliMA** (Oceananigans + ClimaOcean + ClimaSeaIce + ClimaCoupler) has a strong GPU ocean and
   sea-ice stack (documented).
   - Oceananigans' hydrostatic and non-hydrostatic models carry CATKE and other TKE closures and a
     full kit of open-boundary schemes: Flather, Chapman, perturbation advection, radiation,
     tracer reservoirs, tides.
   - ClimaOcean runs global and near-global ocean–sea-ice configurations under a *prescribed*
     atmosphere.
   - ClimaCoupler couples to the ClimaAtmos *global* atmosphere only.
   - No CliMA component couples a *limited-area* atmosphere to an ocean.
3. **NumericalEarth** (NumericalEarth.jl + Breeze.jl) is the only one of the three whose coupler
   can, by construction, put a limited-area atmosphere over an ocean (documented).
   - Its `NestedModel` drives a Breeze limited-area atmosphere through open lateral boundaries.
   - Its `EarthSystemModel` exchanges Monin–Obukhov fluxes with a prescribed, slab, hydrostatic
     or non-hydrostatic ocean every coupled step.
   - The published limited-area example couples to *land*. The only published Breeze–ocean
     demonstration is a periodic, idealized 2-D LES (`examples/breeze_over_four_oceans.jl`).
   - Its own downstream user (AtmosphericRivers.jl) lists "an `AtmosphereOceanModel` around a
     `NestedModel`" as untested.
   - Breeze nests start from an adiabatic balance in the style of FV3's `na_init`
     (`ext/NumericalEarthBreezeExt/breeze_nested_atmosphere.jl:421-453`). The reversible-leapfrog
     digital-filter version is an open pull request
     ([Breeze.jl#834](https://github.com/NumericalEarth/Breeze.jl/pull/834)).
4. **Measured here** (CPU, small boxes; see [Experiment](#experiment-what-runs-today)):
   - With the released packages, a Breeze atmosphere over a `SlabOcean` on a **periodic** grid
     is NaN after its first coupled step. That is true both of our periodic slab case and of a
     one-off run of the published `breeze_over_four_oceans.jl` slab configuration.
   - The cause is a halo-initialisation defect. It is already reported upstream for
     `PrescribedOcean`
     ([NumericalEarth.jl#754](https://github.com/NumericalEarth/NumericalEarth.jl/issues/754),
     open). A short workaround in our script (a halo fill after the state exchange) fixes it.
   - Every other configuration ran end to end, with or without the workaround: the periodic
     CATKE ocean, and the three limited-area cases.
   - **The limited-area cases are the headline.** A limited-area Breeze atmosphere ran for one
     hour over each of:
     - a slab ocean;
     - a closed 3-D CATKE ocean;
     - a 3-D CATKE ocean with open lateral boundaries driven by an external state.

     This is the path AtmosphericRivers.jl calls untested. It works at this toy size on the CPU.
   - The results table is in [Experiment](#experiment-what-runs-today).
5. **Nobody has turnkey regional ocean boundaries from GLORYS** (documented by absence).
   - NumericalEarth can download GLORYS/ECCO/EN4/WOA and restore toward them.
   - Oceananigans has the open-boundary numerics.
   - No published example or function wires reanalysis data into a regional ocean's open
     boundaries. The Mediterranean example restores toward ECCO inside a closed basin, and at
     v0.8.2 it calls a function the package no longer defines.
   - A "nested" ocean model meant for "open boundary conditions with realistic data" is an open,
     unmerged pull request:
     [NumericalEarth.jl#676](https://github.com/NumericalEarth/NumericalEarth.jl/pull/676),
     opened 2026-09-11, still open on 2026-10-15.
6. **Wave coupling exists in none of the three.** All use wind-dependent Charnock-type roughness.
7. **For woof** we recommend this staged path, which keeps the order of
   [gaps-and-roadmap.md](gaps-and-roadmap.md) (added by PR #76):
   - (a0) time-varying SST (`sst_update = 1`);
   - (a1) WRF's mixed-layer ocean (`sf_ocean_physics = 1`), a small single-kernel port;
   - (a2) WRF's 3-D PWP ocean (`sf_ocean_physics = 2`), only if (a1) proves too limited;
   - (b) offline, cycle-level coupling to a regional Oceananigans ocean;
   - (c) online two-way coupling, only after (b) shows the ocean feedback matters for the target
     cases.
   - The [design](#design-an-interactive-ocean-for-woof) section gives the code each step
     touches, its effort and its risk.

## Ocean capability

| Capability | woof | CliMA (Oceananigans, ClimaOcean, ClimaSeaIce) | NumericalEarth (NumericalEarth.jl + Breeze) |
|---|---|---|---|
| Ocean dynamics | None. The skin temperature over water is fixed for the run (`sst_update`, `sst_skin` and `tmn_update` are pinned to 0, `docs/public/CONFIGURATION.md:843`; `woof/namelist_import.py:980`). | Oceananigans `HydrostaticFreeSurfaceModel` (split-explicit or implicit free surface, z-star) and `NonhydrostaticModel` (documented, Oceananigans `src/Models`). ClimaOcean wraps the hydrostatic model with realistic setups (`src/OceanConfigurations/`: lat-lon, ½°, ⅙°, ⅒° tripolar, ORCA). | Uses Oceananigans. `ocean_simulation(grid; model = :hydrostatic)` defaults to `SplitExplicitFreeSurface`, CATKE, quadratic bottom drag 0.003 and two-band penetrating shortwave (`src/Oceans/ocean_simulation.jl:117`, `:163-166`, `:201`, `:264`, `:286`). There is also `nonhydrostatic_ocean_simulation` (`src/Oceans/nonhydrostatic_ocean_simulation.jl:45`), a `SlabOcean` (`src/Oceans/slab_ocean.jl:4`), a `PrescribedOcean`, and a Veros (Python) ocean through PythonCall (`ext/NumericalEarthVerosExt/`). |
| Vertical mixing | n/a | CATKE (`catke_vertical_diffusivity.jl:4`), k–ε (`TKEDissipationVerticalDiffusivity`), Richardson-number (`RiBasedVerticalDiffusivity`) in Oceananigans `src/TurbulenceClosures/`. KPP lives in ClimaOcean `src/OMIPConfigurations/KPP/` (OMIP use). | CATKE by default. Anything Oceananigans offers can be passed through `closure`. |
| Regional open boundaries | n/a (woof's lateral boundaries are atmospheric only) | Oceananigans 0.113.6 has Flather (`GravityWaveRadiation`) and Chapman (`SurfaceWaveRadiation`) for the barotropic mode (`src/BoundaryConditions/open_boundary_schemes/gravity_wave_schemes.jl:56`, `:124`). It also has `PerturbationAdvection` (`perturbation_advection.jl:5`), `NormalRadiation`, `ObliqueRadiation`, MOM6-style `TracerReservoir` (`tracer_reservoir.jl:37`), `TidalHarmonics` (`tidal_forcing.jl:5`), target-transport pinning, and validation cases (`validation/open_boundaries/`). All are documented and in the released version. | Generic `parent_boundary_conditions(grid; variables, sides, bc_types, schemes)` turns any parent `FieldTimeSeries` into lateral boundary conditions (`src/NestedModels/parent_boundary_conditions.jl:53`). It is written for the atmosphere nest, but nothing in it is atmosphere-specific. We used it for the ocean in the experiment. |
| External ocean data | n/a. Initial SST only, through the water-temperature overlay (`woof/ingest/water_overlay.py:77`). | ClimaOcean remains its own package; NumericalEarth is taking over its generic coupling and data-wrangling code, and ClimaOcean depends on NumericalEarth for it (ClimaOcean `README.md:14`, `Project.toml`). | `src/DataWrangling/`: ECCO2/ECCO4, GLORYS (static, daily, monthly, including `thetao`, `uo`, `zos`, `siconc`, `sithick`; `GLORYS/GLORYS.jl:37-39`, `:75-84`), EN4, WOA, JRA55, ERA5, ETOPO, GEBCO, IBCAO, IBCSO. `DatasetRestoring` (`DataWrangling/restoring.jl:72`) nudges T/S toward a dataset in a mask. There is no helper that feeds GLORYS into open boundaries; `examples/download_glorys_data.jl` only downloads and reads it. |
| Bathymetry | n/a | `ImmersedBoundaryGrid` with `GridFittedBottom` / partial cells (Oceananigans) | `regrid_bathymetry(grid; dataset = ETOPO2022())` (`src/Bathymetry/regrid_bathymetry.jl:74`, `:174`), with basin connectivity and smoothing options |
| Resolution / speed (documented) | n/a | "approximately 8 simulated years per day" at ¼°, 10 levels on one H100 (ClimaOcean `README.md:80`). Near-global 488 m at 15 simulated days per day on 768 A100s, and 9.9 SYPD at 10 km on 68 A100s ([arXiv:2309.06662](https://arxiv.org/abs/2309.06662)). GPU dynamical core paper: Silvestri et al. 2025, JAMES, [doi:10.1029/2024MS004465](https://doi.org/10.1029/2024MS004465). | Same ocean. The Mediterranean example is 1/15° (≈ 7 km) with a stretched 5000 m grid (`examples/mediterranean_simulation_with_ecco_restoring.jl:39-50`). |
| Sea ice | Fixed `xice` for the run. RUC has a prognostic sea-ice column (`woof/core/ruc_gpu.py:1273` `ruc_sea_ice_step_cuda`). Noah and Noah-MP skip ice columns (`woof/core/noahmp_runtime.py:630`). MYNN has a WRF-style ice skin wrapper (`woof/core/physics.py:3126`). | ClimaSeaIce: thermodynamics, snow, elasto-visco-plastic dynamics (README). Coupled to the ocean in ClimaOcean and ClimaCoupler. | `sea_ice_simulation(grid, ocean)` with EVP rheology by default (`src/SeaIces/sea_ice_simulation.jl:95`, `:183-186`). There is a three-equation ice–ocean heat flux, and atmosphere–sea-ice similarity fluxes. |
| Lakes | CLM lake, prognostic, on GPU (`woof/core/lake.py:28`, called at `woof/core/physics.py:5307`) | — | — |
| Waves | Charnock / Davis / Taylor-Yelland roughness options (`isftcflx`, `woof/core/kernels/mynn_surface.cu:315`, `woof/core/kernels/sfclay.cu:486`). No wave model. | No wave model. Oceananigans has prescribed Stokes drift for LES (`src/StokesDrifts.jl`). | No wave model. Wind-dependent Charnock (Edson et al. 2013), COARE 3.6 scalar roughness (`InterfaceComputations/roughness_lengths.jl:25-41`, `:65`). |
| Demonstrated *regional* ocean | — | Repository description says "Regional-to-global". Published ClimaOcean examples are global or near-global (`examples/`). | Mediterranean basin with ECCO restoring and closed boundaries (`examples/mediterranean_simulation_with_ecco_restoring.jl`). At v0.8.2 that example calls `ECCO_restoring_forcing` (`:79-80`), which the package no longer defines (grep finds no definition), so it would not run as published. `examples/single_column_os_papa_simulation.jl` is a column. |

## Coupling architecture

| Aspect | woof | CliMA (ClimaCoupler; ClimaOcean `OceanSeaIceModel`) | NumericalEarth (`EarthSystemModel`) |
|---|---|---|---|
| What couples | Atmosphere ↔ fixed lower boundary. Land models, the CLM lake and RUC sea ice are the only interactive surfaces. | ClimaAtmos (global cubed sphere, `h_elem`) with the bucket or ClimaLand, a prescribed, slab or Oceananigans ocean, and a prescribed or ClimaSeaIce ice (`src/Input.jl:264-267`; `config/ci_configs/cmip_oceananigans_climaseaice.yml`). ClimaOcean couples ocean and sea ice to a *prescribed* atmosphere (JRA55 in the README example, `README.md:75`), ocean side only. | Any atmosphere (prescribed, Breeze, SpeedyWeather) with any ocean (prescribed, slab, Oceananigans hydrostatic or non-hydrostatic, Veros), sea ice (ClimaSeaIce), land (`SlabLand`, prescribed) and radiation (`src/NumericalEarth.jl` exports; `ext/`). |
| Limited-area atmosphere | woof is a limited-area model | No. ClimaAtmos in ClimaCoupler is global. | Yes. `nested_atmosphere_model(parent, child_grid; …)` builds a Breeze compressible limited-area model with parent-driven open boundaries and Davies relaxation, wrapped in a `NestedModel` (`ext/NumericalEarthBreezeExt/breeze_nested_atmosphere.jl:218`; `src/NestedModels/nested_model.jl:39-95`). The ERA5 → 12 km example couples it to `SlabLand` (`examples/breeze_downscaling_era5.jl:1-35`). Its downstream user lists "`AtmosphereOceanModel` around a `NestedModel` atmosphere is an untested path" and "Mixed land+ocean surface … assumes one surface type per cell" ([glwagner/AtmosphericRivers.jl README](https://github.com/glwagner/AtmosphericRivers.jl), Known risks and Scale-up ladder, commit 7cc5735). |
| Flux algorithm | woof's own surface layers compute fluxes over fixed SST: MM5 (`woof/core/kernels/sfclay.cu:212`), MYNN (`woof/core/kernels/mynn_surface.cu:351`), MYJ (`woof/core/kernels/myjsfc.cu:104`) | SurfaceFluxes.jl Monin–Obukhov (`ClimaCoupler src/FluxCalculator.jl:11`, `:265`) | `SimilarityTheoryFluxes`: Edson stability functions, Edson 2013 / COARE 3.6 roughness, Beljaars convective gustiness (`InterfaceComputations/similarity_theory_turbulent_fluxes.jl:169-181`, `:633-635`). Or `CoefficientBasedFluxes`, including Large & Yeager 2004 (`coefficient_based_turbulent_fluxes.jl:64-101`). Fluxes are computed by the coupler, not by the atmosphere's own surface layer. |
| Exchange frequency | n/a | `dt_cpl` with component steps that divide it. The CMIP CI config uses `dt_cpl` 6 min, atmosphere 2 min, ocean and ice 30 min (`cmip_oceananigans_climaseaice.yml:9-11`). Fluxes are accumulated between ocean steps (`FluxCalculator.jl:120-229`). | Every coupled step, sequentially, with one Δt for all components: radiation, atmosphere, land, ocean, sea ice, then `update_state!` recomputes all interface fluxes (`src/EarthSystemModels/time_step_earth_system_model.jl:14-84`). There is no sub-cycling or flux averaging in the generic model. |
| Two-way | — | Yes | Yes. SST feeds the flux solve, and the fluxes feed both atmosphere and ocean boundary conditions (`breeze_atmosphere_interface.jl:178-210`). |
| Grids / regridding | — | Remapping between the cubed-sphere atmosphere and the ocean grid (`use_intersection_grid`, `ext/ClimaCouplerCMIPExt/exchange_grid.jl`) | The exchange grid is the ocean grid (`src/EarthSystemModels/components.jl:23`). The Breeze state is copied **index for index** onto it (`breeze_atmosphere_interface.jl:108-121`), so the atmosphere and ocean must share the horizontal grid. A prescribed atmosphere is interpolated instead. |
| Conservation | — | `ConservationChecker.jl` | `examples/coupled_conservation.jl` closes volume, salt and energy for ocean + sea ice under a prescribed atmosphere. |
| Restarts | woof checkpoints (`woof/io/restart.py`) | `Checkpointer.jl` | Oceananigans `Checkpointer`; `NestedModel` checkpoints only the child (`nested_model.jl:98-99`); `test/test_checkpointer.jl` |
| Wave coupling | none | none | none |
| GPU | CuPy, one process | CUDA via ClimaComms; MPI | CUDA via Oceananigans; distributed (NCCL in AtmosphericRivers) |

## Experiment: what runs today

**What we ran (measured).** We wrote
[`tools/compare/ocean_coupling/coupled_regional_box.jl`](../../../tools/compare/ocean_coupling/coupled_regional_box.jl)
and ran it with the registered NumericalEarth 0.8.2, Breeze 0.11.3 and Oceananigans 0.113.6 on
Julia 1.12.7.

- It ran **on the CPU** (4 threads). We did not use the GPU because the card was shared.
- It downloads nothing. The parent atmosphere is a synthetic `PrescribedAtmosphere`, and the
  "external" ocean state is a synthetic `FieldTimeSeries` standing in for GLORYS.
- The limited-area box is 2° × 2° at 36.6°N with 16 × 16 columns (≈ 12 km), 16 atmospheric
  levels to 16 km, and 10 ocean levels to 100 m.
- The initial SST is 20 °C under 288 K air, so heat flows from ocean to atmosphere.
- The exact commands are in the
  [README](../../../tools/compare/ocean_coupling/README.md). The raw outputs are
  `tools/compare/ocean_coupling/results/summary.csv`, `summary_unpatched_extra.csv`, `run.log`,
  and one log per case.

### What the columns mean

| Column | Meaning |
|---|---|
| Steps / simulated | Number of coupled steps (Δt = 5 s) and the simulated time they cover |
| SST start → end | Area-mean top-level SST, °C |
| Mean Q | Time- and area-mean turbulent heat flux into the atmosphere (sensible + latent), W m⁻² |
| Budget residual | Relative mismatch between the change in ocean heat content and −ρc ∫ Jᵀ dt, where Jᵀ is the temperature flux the coupler handed to the ocean. Only for closed oceans. |
| Wall: first / rest | Seconds for the first coupled step (JIT compilation; near zero when an earlier case in the same process compiled the same code) and for the remaining steps |

### Results (measured; `results/summary.csv`)

| Case | Status | Steps / simulated | SST start → end (°C) | Mean Q (W m⁻²) | Budget residual | Wall: first / rest (s) |
|---|---|---|---|---|---|---|
| `periodic_slab_unpatched` | **NaN** (interface flux first non-finite after step 1) | 100 / 500 s (stopped by Oceananigans' NaN checker) | 20.00 → NaN | NaN | — | 4.1 / 0.4 |
| `periodic_slab` | OK | 360 / 30 min | 20.00 → 19.98 | 474 | 1.2e-12 | 0.0 / 0.8 |
| `periodic_catke` | OK | 360 / 30 min | 20.00 → 19.98 | 474 | 1.3e-10 | 17.3 / 1.4 |
| `lam_slab_unpatched` | OK | 720 / 1 h | 20.00 → 19.95 | 589 | 6.1e-12 | 41.5 / 98.4 |
| `lam_slab` | OK | 720 / 1 h | 20.00 → 19.95 | 589 | 6.1e-12 | 0.1 / 41.1 |
| `lam_catke` | OK | 720 / 1 h | 20.00 → 19.81 | 577 | 8.6e-3 | 48.2 / 92.5 |
| `ocean_open` (no atmosphere) | OK, max\|u\| 0.065 m/s against a 0.05 m/s external current | 720 / 1 h | 20.00 → 20.00 | — | — (open) | 18.6 / 6.7 |
| `lam_catke_open` | OK | 720 / 1 h | 20.00 → 19.81 | 577 | — (open; ΔH = −1.87 MJ m⁻² vs −∫flux = −2.08 MJ m⁻², the difference being heat carried in through the boundaries) | 2.7 / 136.3 |

The same harness was also run without the workaround on the other configurations
(`results/summary_unpatched_extra.csv`): `periodic_catke_unpatched`, `lam_catke_unpatched` and
`lam_catke_open_unpatched` were all OK, with the same numbers to four figures. The published
four-oceans slab configuration, unpatched (`four_oceans_slab_check.jl`,
`results/four_oceans_slab_check.log`), had a finite state at step 0 and was NaN in the
atmosphere at step 1.

### What it shows

- **The defect is real and narrow.**
  - The interface flux solve runs over halo columns
    (`InterfaceComputations/InterfaceComputations.jl:139-142`). The `SlabOcean` temperature's
    halos are never filled, so the halo column holds 0 K.
  - We also measured that the Breeze atmosphere's exchange state is written interior-only, with
    its halo left at T = 0, p = 0.
  - The resulting NaN is interpolated into the first ρu face
    (`breeze_atmosphere_interface.jl:185`).
  - It only matters where that face is prognostic: periodic grids, as in the published example.
  - Hydrostatic Oceananigans oceans fill their own halos.
  - Upstream issue #754 reports the same mechanism for `PrescribedOcean`, and reports a
    limited-area NaN about 20 steps in with that component.
- **A limited-area atmosphere over an interactive regional ocean works** at toy size: slab,
  closed CATKE, and open-boundary CATKE.
  - The coupling is two-way. SST falls by 0.05 K (10 m slab) and 0.19 K (top 10 m cell of the
    CATKE ocean) in one hour under about 580 W m⁻² of mean heat loss.
  - The surface flux the atmosphere receives is not steady (`results/run.log`). It is 380 W m⁻²
    at the start, rises to 800–820 W m⁻² at 30 minutes as the nest adjusts, and falls to
    265–280 W m⁻² at one hour. One hour of a synthetic case says nothing about realism; it shows
    the exchange is live in both directions.
  - The 3-D ocean cools its top cell faster than the slab because CATKE confines the loss to the
    mixed layer. That is the physics an OML-type scheme approximates.
- **The coupler is energy-consistent on the ocean side.**
  - For the slab and the periodic column, the ocean's heat-content change matches the integrated
    flux it was handed to 1e-10 or better.
  - For `lam_catke` the residual is 0.9 %. We have not isolated it. Advection on, plus a
    non-conserving linear free surface (the w·T flux through z = 0), is the likely cause
    (estimated); the periodic column with advection off closes to 1e-10.
- **Open boundaries from an external state work.**
  - The generic `parent_boundary_conditions` (written for the atmosphere nest) also builds an
    ocean's lateral boundary conditions from an external `FieldTimeSeries`: T/S Dirichlet,
    baroclinic `PerturbationAdvection`.
  - Combined with Oceananigans' Flather/Chapman barotropic conditions, it ran stably with the
    atmosphere on top.
  - This is a synthetic stand-in. Wiring real GLORYS fields (time interpolation, vertical
    regridding, `zos` and the transport to the Flather condition) is still to be written by
    whoever builds it.
- **Cost.**
  - On the CPU, about 1–3 minutes of compilation per new component combination.
  - Then 40–140 s per simulated hour at 16 × 16 columns. That is meaningless for GPU
    performance, which we did not measure.

### Limits of the experiment

- Toy size. Synthetic, horizontally uniform parent atmosphere. No radiation, no land, no sea
  ice, no bathymetry (flat 100 m).
- One hour of simulation. CPU only.
- The halo workaround is ours, not upstream's.
- The atmosphere and ocean share one horizontal grid. That is required, because the Breeze
  exchange copies index for index.

## What WRF offers (the reference woof ports from)

WRF v4.8.0 has three in-model options for the ocean below the atmosphere (documented,
`run/README.namelist:1007-1037`).

- **`sst_update = 1`.** `real.exe` writes SST, XICE, albedo and VEGFRA to `wrflowinp_d0N`, and WRF
  reads them at the `auxinput4_interval`. This is prescribed, time-varying SST, not an ocean.
- **`sf_ocean_physics = 1`: OML.**
  - A 1-D mixed-layer slab after Pollard, Rhines and Thompson (1973) (`phys/module_sf_oml.F:8-23`).
  - Seven 2-D fields: mixed-layer temperature and depth, their initial values, the two
    mixed-layer transports, and the top-200 m mean temperature.
  - Parameters: `oml_hml0` (initial depth; a value below 0 means real-time mixed-layer depth from
    the input, 0 means climatology), `oml_gamma` (deep lapse rate) and `oml_relaxation_time`.
  - It is called after the land surface model on every water point (`XLAND > 1.5`) from
    `phys/module_sf_ocean_driver.F:113-131`, using that step's HFX, LH, GSW, GLW and UST.
  - It writes `TSK = TML`.
  - The whole scheme is about 90 lines of point-wise arithmetic (`module_sf_oml.F:80-165`).
- **`sf_ocean_physics = 2`: 3-D PWP.**
  - A 3-D Price–Weller–Pinkel ocean (Price et al. 1994) with horizontal advection and pressure
    gradients (`phys/module_sf_3dpwp.F:10-24`, `:352-401`). WRF's notes say it has "no bathymetry".
  - It reads `ocean_levels` (default 30) of T/S/U/V profiles from `wrfinput`, or from the
    namelist profiles `ocean_z`, `ocean_t` and `ocean_s`.
  - It is stepped every `omdt` minutes.
  - It runs only where Δx ≥ 3 km. Finer domains just copy the top ocean temperature into TSK
    (`module_sf_ocean_driver.F:135-160`).

Coupling WRF to a full regional ocean is done outside WRF. COAWST couples WRF, ROMS, SWAN and
WW3 through MCT (Warner et al. 2010, Ocean Modelling,
[doi:10.1016/j.ocemod.2010.07.010](https://doi.org/10.1016/j.ocemod.2010.07.010)). SKRIPS couples
WRF and MITgcm through ESMF/NUOPC (Sun et al. 2019, GMD,
[doi:10.5194/gmd-12-4221-2019](https://doi.org/10.5194/gmd-12-4221-2019)). Both are separate
coupler layers around unmodified-ish component models. Neither is a namelist option.

## woof today

All of these were checked in this tree (documented by the code).

- **State.** The surface lives in `PhysicsDriver.fields`, a dict of CuPy arrays built by
  `initialize_physics` (`woof/core/physics.py:5720`, the dict at `:5975-6000`), and attached to
  the domain at `woof/core/physics.py:6588`.
  - SST is not a field of its own. Over water, `tsk = where(terrestrial | sea_ice, skin, water_temperature)`
    at preparation (`woof/ingest/soil.py:1660`), and nothing changes it afterwards.
  - Under RUC with fractional ice, `tsk_sea` carries the open-water part (`woof/core/physics.py:6052-6063`).
- **Surface layers.** All read `tsk` over water and only update the roughness length.
  - MM5: `ground_t = tsk`, water roughness (Charnock 0.0185 or `isftcflx`), fluxes, at
    `woof/core/kernels/sfclay.cu:242-244`, `:486-515`.
  - MYNN: `woof/core/kernels/mynn_surface.cu:397`, `:473-474`, `:616-622`.
  - MYJ: `woof/core/kernels/myjsfc.cu:150`.
- **Land surface models skip water.**
  - Noah returns on open water and on sea ice (`woof/core/kernels/noah.cu:994-998`).
  - Noah-MP classifies and skips sea ice and open water (`woof/core/noahmp_runtime.py:630`).
  - RUC only recomputes the surface humidity from `tsk` on water (`woof/core/ruc.py:9704-9741`).
- **Physics order** in `PhysicsDriver.compute` (`woof/core/physics.py:5084`):
  - radiation;
  - the surface-layer slot (`resolve_physics_dispatch`, `:5185`);
  - the land-model slot (`:5297`);
  - SFCDIAGS and urban;
  - **the CLM lake step, which overwrites `tsk`, `hfx`, `qfx` and `lh` on lake cells
    (`woof/core/physics.py:5307-5310`)**;
  - the planetary boundary layer (`:5341`).

  An ocean step belongs exactly where the lake step is.
- **Ocean currents.** MYNN's PBL passes zero `UOCE`/`VOCE` columns, with the comment "woof has no
  ocean-current coupling" (`woof/core/mynn_pbl_runtime.py:315-317`).
- **Time-varying SST is fetched, then dropped.**
  - Source snapshots carry SST at every forcing time.
  - The overlay is applied to all of them (`overlay_snapshots`, `woof/ingest/water_overlay.py:532`).
  - Only the analysis time reaches `tsk`.
  - The physics registry records why `sst_update` is unimplemented: it "require[s] a
    time-varying lower-boundary input stream, interpolation/cadence state, restart position, and
    surface ownership" (`tools/build_registry.py:1124-1128`).
- **The global path** says the same thing ("no ocean model", `woof/globe/surface_seeding.py:347`).
  It already integrates a slab skin for lakes with a 4e7 J m⁻² K⁻¹ capacity (a 10 m mixed layer;
  `woof/globe/physics/native_runtime.py:1305-1345`). That is the closest thing in the tree to
  WRF's OML.
- **Restart.** Everything in `driver.fields` is checkpointed automatically
  (`woof/io/restart.py:3436`). Restore demands that the stored and live field sets match exactly.
  The lake precedent for adding default-off state without breaking old checkpoints is at
  `woof/io/restart.py:2469-2476`.

## Design: an interactive ocean for woof

The options are ordered by cost and match the sequence in
[gaps-and-roadmap.md](gaps-and-roadmap.md): time-varying SST, then the mixed layer, then PWP, then
regional Oceananigans/ClimaOcean coupling. Each later option reuses the earlier ones.

**What every option needs.** Every option that changes SST during a run needs:

- an owner for "water skin temperature" that is not the analysis;
- a restart entry for that state;
- a configuration identity so old checkpoints refuse to resume under a new ocean setting.

Those three are common to all options. They are priced once, in (a0).

### (a0) Time-varying prescribed SST (`sst_update = 1`)

**What it is.** Interpolate SST and sea-ice fraction between forcing times, as WRF's `wrflowinp`
does, and write them into `tsk` (and `tsk_sea`, `xice`) on water cells.

**woof code it touches:**

- Lift the pin: `woof/namelist_import.py:980` (the `PHYSICS_PINS` row) and its enforcement at
  `:3737-3745`; the contract rule at `woof/namelist_contract.py:280`; the
  `docs/public/CONFIGURATION.md:843` row; and the `UNIMPLEMENTED_LEDGER` entry at
  `tools/build_registry.py:1124`.
- Keep the per-time SST that `overlay_snapshots` (`woof/ingest/water_overlay.py:532`) already
  computes, and carry it through preparation the way the lateral boundary file is carried.
- Bind a clock and interpolate in time on the lateral-boundary template:
  `woof/ingest/lateral_bc.py:1903` `bind_lateral_boundary_clock` and `:1998`
  `apply_state_lateral_boundaries`.
- Write `tsk` on water before the surface layer in `woof/core/physics.py:5084` (`compute`), or
  widen the external-forcing door `set_forcing` (`woof/core/physics.py:5397`, the allowed set at
  `:5412`).
- Restart position: the interval index joins the driver manifest (`woof/io/restart.py:3368`).
- Pins elsewhere: `woof/hrrr_hierarchy_direct.py:366` and `woof/hrrr_route_inputs.py:860` emit or
  require `sst_update = 0`. They stay as they are for HRRR, which has no SST.

**Effort and risk (estimated).** 1–2 weeks. Low risk: no new physics, and a WRF parity test is
possible.

**Value.** Multi-day runs and cycling get the analysis's own SST evolution. It is a prerequisite
for (b).

### (a1) Mixed-layer ocean inside woof (WRF `sf_ocean_physics = 1`)

**What it is.** Port `OML1D`. It is about 90 lines of point-wise arithmetic, so one CuPy
`ElementwiseKernel` or one small `.cu` kernel on water cells (`xland > 1.5`, not lake, not ice).

**woof code it touches:**

- **Config.** New `RunConfig` fields `sf_ocean_physics`, `oml_hml0`, `oml_gamma` and
  `oml_relaxation_time`, appended at the end of `woof/config.py:57` `RunConfig` the way
  `isftcflx` was (`woof/config.py:271-279`). Validation goes in `validate_run_config` beside the
  `isftcflx` checks (`woof/config.py:4538`).
  - Namelist import goes through `PHYSICS_CHOICES` (`woof/namelist_import.py:1029`) and the parse
    at `:3702`. Today an unknown `sf_ocean_physics` fails as an unmapped key (`:1511-1517`).
  - The wrfout global attribute that now reads "unimplemented" is at
    `woof/io/wrf_output_schema.py:850`.
- **State.** Seven new 2-D arrays (`tml`, `t0ml`, `hml`, `h0ml`, `huml`, `hvml`, `tmoml`) in the
  fields dict (`woof/core/physics.py:5975-6000`). Checkpointing is then automatic
  (`woof/io/restart.py:3436`).
  - The default-off drop in `_drop_default_off_run_keys` follows the lake precedent
    (`woof/io/restart.py:2469-2476`).
  - VRAM pricing: `woof/core/preflight.py:4293` `physics_field_names_2d` and `:4365`
    `physics_array_shapes`.
  - Scheme identity: `woof/checkpoint_identity.py:214`.
- **Call site.** After the land model and before the planetary boundary layer, exactly where the
  lake steps (`woof/core/physics.py:5307-5310`). WRF calls it at the same point
  (`module_surface_driver.F:4486-4506`).
  - Inputs already in `fields`: `hfx`, `lh`, `swdown`/`gsw`, `glw`, `ust`, the lowest-level wind,
    and the Coriolis parameter.
  - Output: `tsk` on water. Under RUC fractional ice, `tsk_sea`.
- **Initialisation.** `oml_hml0 ≥ 0` needs nothing new. `oml_hml0 < 0` needs a mixed-layer-depth
  field from the source (for example HYCOM or GLORYS), which is a new preparation input.

**Effort and risk (estimated).**

- About 1–2 weeks including a column parity test against the Fortran.
- Low risk. The physics is simple and its limits are known: no advection, the mixed layer only
  deepens, and it is best for tropical-cyclone cold wakes (the AHW use case).
- **Validation.** Bit-for-bit or near parity against WRF on a column, and SST cooling under a
  synthetic hurricane-force wind.

### (a2) 3-D PWP ocean (WRF `sf_ocean_physics = 2`)

**What it is.** Port `DPWP` and `PWP` (`phys/module_sf_3dpwp.F`, about 980 lines). It adds an
ocean vertical axis (`ocean_levels`) and horizontal advection and pressure gradients on woof's
mass grid. It needs no bathymetry, and WRF skips it below Δx = 3 km.

**woof code it touches:**

- Everything in (a1).
- A new 3-D ocean state (T, S, U, V, density, depth on `ocean_levels`) in the driver.
- A halo or stencil treatment on woof's grid for the horizontal terms.
- Preparation of `OM_TMP`, `OM_S`, `OM_U`, `OM_V`, `OM_DEPTH` and `OM_ML` from an ocean source.
  This is new: woof's ingest carries no ocean profiles today (`woof/ingest/metem.py:67` lists
  surface fields only).

**Effort and risk (estimated).** 3–5 weeks, medium risk. **Value** is limited. With no
bathymetry, no open boundaries and the 3 km gate, it is a WRF-parity feature rather than a
regional ocean. We would do it only if (a1) proves too crude and (b) is out of reach.

### (b) One-way (cycle-level) offline coupling to a regional Oceananigans ocean

**What it is.**

1. woof runs a forecast as today.
2. Its surface output drives an Oceananigans/NumericalEarth regional ocean for the same window.
   - Fields used: 10 m wind or stress, `t2`, `q2`, `psfc`, `swdown`, `glw`, rain.
   - The ocean has GLORYS initial state and open boundaries, ETOPO bathymetry and CATKE.
3. The ocean's SST is handed back as the water-temperature overlay for the **next** cycle.

**How it splits.**

- The atmosphere forces the ocean within a cycle.
- The ocean feeds the atmosphere across cycles.
- Neither waits on the other.

**woof code it touches:**

- **Export.** The live Icechunk export already streams frames to a separate process without the
  model waiting on it (`woof/icechunk_export.py:859` `LiveExport`). Frames are marked ready by
  `woof/progress_log.py:411` `write_frame_marker`. The ocean driver consumes the same frames.
- **Fields.** A surface-forcing field list must include the fluxes. `hfx`, `qfx`, `lh`, `ust` and
  `u10`/`v10` live in `driver.fields`. There is no `taux`/`tauy` array, so stress is rebuilt from
  `ust` and the wind direction, or recomputed by the ocean's own bulk formula from the 10 m state.
- **Exchange format.** The two directions differ.
  - woof → ocean: woof's export writes Icechunk/Zarr, which the Julia side can read directly.
    Oceananigans also has a `ZarrWriter` (`src/OutputWriters/zarr_writer.jl:8`, implemented in its
    Zarr extension).
  - ocean → woof: the overlay loader reads only netCDF or GRIB2 today
    (`woof/ingest/water_overlay.py:361-388`). So the ocean must write its SST as netCDF (it can,
    through Oceananigans' NetCDF writer, which is the `OceananigansNCDatasetsExt` loaded in our
    environment), or woof's loader must learn Zarr.
- **Feedback.** The `external_overlay` water-temperature policy (`woof/ingest/water_temperature.py:107-112`)
  and `load_water_temperature_overlay` (`woof/ingest/water_overlay.py:361`) already accept an
  external SST field at initialisation. The ocean writes that file.

**Julia side.**

- A `PrescribedAtmosphere` built from woof's frames (NumericalEarth computes the bulk fluxes), or
  flux boundary conditions taken directly from woof's `hfx`/`lh`.
- The ocean itself:
  - `ocean_simulation` on a `LatitudeLongitudeGrid` covering woof's domain;
  - `regrid_bathymetry`;
  - GLORYS `Metadatum`s for initial T/S/u/v/η;
  - `parent_boundary_conditions` and Flather/Chapman for the boundaries, as in our experiment's
    `ocean_open` case.

**Effort and risk (estimated).**

- 2–4 weeks for a research workflow.
- Low risk to woof: no change to the dynamics core or the physics.
- Medium risk on the Julia side. The GLORYS-to-open-boundary wiring has no upstream example; our
  experiment proves only the synthetic version.

**What it buys.** Ocean memory between cycles: cold wakes, upwelling and coastal mixing, seen by
the next forecast. It does not give within-forecast feedback.

### (c) Online two-way coupling with a regional Oceananigans/ClimaOcean ocean

**What it is.** The ocean steps alongside woof's forecast. Every N woof steps, the coupling
interval (for example 10–30 min, the way ClimaCoupler sub-cycles):

1. woof passes time-averaged surface fluxes to the ocean.
2. The ocean steps.
3. It returns SST, sea-ice fraction and surface currents.

The choices are listed one at a time.

1. **Who computes the fluxes.**
   - Keep woof's surface layer (MM5/MYNN/MYJ) and give the ocean woof's `hfx`, `qfx`, `lh` and
     stress. The ocean is then forced exactly by what the atmosphere felt, which conserves energy
     at the interface.
   - The alternative, letting NumericalEarth's similarity theory compute fluxes, would put a
     second surface layer under woof that is inconsistent with its PBL.
   - **Recommended:** woof computes fluxes.
2. **Accumulators.**
   - Sum `hfx`, `qfx`, the shortwave and longwave absorbed at the surface, and stress over the
     coupling interval in new `fields` arrays (checkpointed automatically, `woof/io/restart.py:3436`).
   - Hand over the means.
   - Hold the ocean's SST fixed within the interval.
3. **Grids.**
   - woof's mass grid is Lambert conformal, polar stereographic or Mercator, and is orthogonal.
     Oceananigans can represent it either as an `OrthogonalSphericalShellGrid` built from woof's
     cell corners and map factors (exact, so no regridding), or as a separate
     `LatitudeLongitudeGrid` with conservative remapping. NumericalEarth already has a weak
     dependency on ConservativeRegridding.jl.
   - Our experiment shows NumericalEarth's own Breeze path assumes identical grids: an
     index-for-index copy (`breeze_atmosphere_interface.jl:108-121`).
   - Land–sea masks must agree. A woof water cell that is land in the ocean's bathymetry needs a
     fallback, either the analysis SST or (a1).
4. **SST and sea-ice feedback.**
   - Write the returned SST into `tsk` (and `tsk_sea`) on water cells at the lake's slot
     (`woof/core/physics.py:5307`).
   - Write the ice fraction into `xice` if the ocean runs ClimaSeaIce.
   - Feed surface currents into MYNN's `UOCE`/`VOCE`, which are zero today
     (`woof/core/mynn_pbl_runtime.py:315-317`).
5. **Process and GPU co-residence.** Three ways to run the two models together:
   - **c1, in process, woof drives Julia through juliacall.**
     - CuPy and CUDA.jl share the device's primary context, so device arrays could be wrapped
       without copies: CUDA.jl `unsafe_wrap(CuArray, …)` on a CuPy pointer, and back through
       `__cuda_array_interface__`.
     - Risks: two allocators on one card (CuPy's pool and CUDA.jl's, which need hard caps), Julia
       JIT latency at start-up (minutes; our CPU build of one coupled case took about 2–3
       minutes), Julia GC pauses inside woof's step, and packaging a Julia depot with woof's
       wheels.
     - The precedent runs the other way: NumericalEarth already drives a *Python* ocean (Veros)
       through PythonCall (`ext/NumericalEarthVerosExt/`).
   - **c2, Julia drives woof** as a NumericalEarth atmosphere component through PythonCall. This
     reuses the `EarthSystemModel` loop but makes NumericalEarth compute the fluxes, which is
     choice 1's rejected alternative. **Not recommended.**
   - **c3, two processes** exchanging coupling-interval fields through shared files or Zarr, or
     CUDA IPC handles, at the coupling boundary.
     - Mechanism: woof writes fluxes and waits on an SST "ready" marker, mirroring
       `woof/progress_log.py:411`.
     - Pros: each model keeps its own allocator, failure domain and restart. It is how COAWST and
       SKRIPS couple WRF.
     - Cost: a host round trip per interval, which is negligible at 10–30 min intervals for
       regional sizes.
     - **Recommended first.**
6. **Hook in woof's loop.**
   - The per-domain post-step hook in `execute_experiment` (`woof/core/model.py:1552`).
     `spectral_seam.after_step` at `:2004-2018` is the existing pattern for "do something after
     each committed step".
   - `on_period_commit` (`:2240`) for synchronized boundaries.
   - The ocean step itself runs at the lake's slot or after the step commits. The SST it returns
     applies from the next step.
7. **Restarts.**
   - woof checkpoints only at a coupling boundary.
   - The ocean writes an Oceananigans `Checkpointer` file at the same simulated time.
   - woof's restart manifest records the ocean checkpoint path and configuration hash, beside
     `_driver_manifest` (`woof/io/restart.py:3368`), so a resume refuses mismatched pairs.
   - `on_restart` (`woof/core/model.py:2184`) is the trigger.
8. **Conservation.**
   - With woof-computed fluxes, averaged over the interval and applied once, the ocean's heat
     change equals minus the integrated flux. Our closed-ocean cases check that on the ocean side.
   - The atmosphere side is exact by construction, because it is woof's own flux.
   - Regridding must be conservative if the grids differ.
9. **Nests.**
   - Couple one domain (the outer or a fixed inner domain).
   - Moving or spawned nests are out of scope, as for the CLM lake
     (`docs/ruc-mosaic-and-clm-lake.md:68-77`).

**Effort and risk (estimated).**

- c3 prototype: 4–8 weeks, medium–high risk.
- c1: add 2–4 weeks and higher risk: GPU memory contention, JIT, packaging.
- Either needs (a0)'s SST ownership, and (b)'s regional ocean setup, first.

### Summary of options

| Option | What changes SST | woof code touched (main sites) | Effort (est.) | Risk (est.) |
|---|---|---|---|---|
| (a0) `sst_update = 1` | Analysis SST interpolated in time | `woof/namelist_import.py:980`, `woof/ingest/water_overlay.py:532`, `woof/ingest/lateral_bc.py:1903`, `woof/core/physics.py:5397` | 1–2 weeks | low |
| (a1) OML (`sf_ocean_physics = 1`) | Prognostic mixed layer, per water column | `woof/config.py:57`, `woof/namelist_import.py:1029`, `woof/core/physics.py:5307`, `woof/io/restart.py:2469`, `woof/core/preflight.py:4293` | 1–2 weeks | low |
| (a2) 3-D PWP (`sf_ocean_physics = 2`) | 3-D PWP ocean, no bathymetry, Δx ≥ 3 km | (a1) plus new ocean state and ingest of ocean profiles | 3–5 weeks | medium |
| (b) Offline cycle coupling | Regional Oceananigans SST as next cycle's overlay | `woof/icechunk_export.py:859`, `woof/ingest/water_temperature.py:107`, `woof/ingest/water_overlay.py:361` | 2–4 weeks | low (woof), medium (Julia side) |
| (c) Online two-way | Regional Oceananigans SST every coupling interval | (a0) plus `woof/core/model.py:2004`, `woof/core/physics.py:5307`, `woof/core/mynn_pbl_runtime.py:315`, `woof/io/restart.py:3368` | 4–8 weeks (c3), +2–4 (c1) | medium–high |

**Recommendation.**

- Do (a0) then (a1). Together they give woof WRF's capability, with tests, in under a month.
- Run (b) as a research workflow on cases where the ocean plausibly matters: tropical-cyclone
  wakes, coastal upwelling, multi-day runs.
- Take on (c) only if (b) shows that within-forecast feedback changes the atmosphere enough to
  justify it.
- Do not plan (c) around NumericalEarth's `EarthSystemModel` as the driver. Its fluxes,
  same-grid copy and step-locked exchange do not fit woof's own surface layer. Use Oceananigans
  (with NumericalEarth's data and boundary tools) as the ocean component, driven by woof.

## Not verified here

- GPU behaviour and performance of any coupled configuration. We ran on the CPU only.
- Whether NumericalEarth `main` fixes the halo defect. Issue #754 was open on 2026-10-15, and
  `main`'s `breeze_atmosphere_interface.jl` assembly kernel is unchanged from v0.8.2.
- ClimaOcean and ClimaCoupler were read, not run.
- WRF's OML and PWP were read, not run. The effort figures for porting them are estimates.
