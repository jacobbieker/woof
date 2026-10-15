# woof against CliMA and NumericalEarth: gaps and roadmap

This page lists what woof lacks compared with the CliMA stack
([github.com/CliMA](https://github.com/CliMA)) and the NumericalEarth stack
([github.com/NumericalEarth](https://github.com/NumericalEarth)) for regional
and limited-area modelling. Each gap has candidate approaches, an effort
size, the main risk, and the external components that could be reused, with
their licences. The page then lists what woof does that the other two stacks
do not. It ends with a prioritised top five for two goals: regional
ocean-plus-atmosphere modelling, and wind accuracy over rough terrain.

Snapshot: woof `jacob/main` at `b4a3783a`. External repositories were read on
2026-10-15. Their code links are permalinks to the commits read that day:

| Repository | Commit read |
|---|---|
| [CliMA/ClimaAtmos.jl](https://github.com/CliMA/ClimaAtmos.jl) | `dda668d3` |
| [CliMA/ClimaCore.jl](https://github.com/CliMA/ClimaCore.jl) | `d7f5b795` |
| [CliMA/CloudMicrophysics.jl](https://github.com/CliMA/CloudMicrophysics.jl) | `542dd849` |
| [CliMA/RRTMGP.jl](https://github.com/CliMA/RRTMGP.jl) | `19c6355a` |
| [CliMA/ClimaOcean.jl](https://github.com/CliMA/ClimaOcean.jl) | `940e5515` |
| [CliMA/Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl) | `fec351f0` |
| [NumericalEarth/Breeze.jl](https://github.com/NumericalEarth/Breeze.jl) | `6eabb893` |
| [NumericalEarth/NumericalEarth.jl](https://github.com/NumericalEarth/NumericalEarth.jl) | `66112eac` |
| [wrf-model/WRF](https://github.com/wrf-model/WRF) | `06d4240a` |
| [erf-model/ERF](https://github.com/erf-model/ERF) | `92410d57` |

## How to read this page

Each statement has one of three kinds of support:

- **Documented**: a page or a source file says it. The citation is a URL, or
  a `path:line` in this repository at the snapshot above.
- **Measured**: a receipt or a test in this repository recorded it. The
  citation names the page that reports the measurement.
- **Judgement**: the author's estimate. Every effort size and every risk
  rating on this page is a judgement. None has been costed against a plan
  or a team.

Effort sizes (judgement):

| Size | Meaning |
|---|---|
| **S** | Up to about one engineer-month. A configuration change, a new data fetcher behind an existing seam, or a port of one WRF routine using the existing oracle pattern. |
| **M** | One to four engineer-months. A new component behind an existing interface, with its own verification. |
| **L** | More than four engineer-months, or a change to the dynamical core, the restart contract or the multi-domain coupler. |

Risk ratings (judgement) say how likely the work is to stall or to fail its
own acceptance test. They do not measure the risk to existing users, because
every item below would be off by default.

## Summary

| # | Gap | Who has it | Effort (judgement) | Risk (judgement) |
|---|---|---|---|---|
| 1 | Ocean and sea-ice coupling | CliMA (ClimaOcean, ClimaSeaIce, ClimaCoupler). NumericalEarth (`EarthSystemModel` with prescribed, slab, CATKE-column and LES oceans) | S for time-varying SST or a WRF-style mixed-layer ocean. L for a 3-D regional ocean | Low for the slab. High for two-way 3-D coupling |
| 2 | In-run adaptive mesh refinement | Neither stack ships it for a real-data atmosphere. ERF (AMReX) does | M for more in-run nest triggers. L for true AMR | Medium. High for AMR |
| 3 | Sub-10 m terrain and 10 m land use | NumericalEarth (10 m ESA WorldCover, 30 m GLO-30) | S per source | Low technically. Medium scientifically (surface vs bare-earth models) |
| 4 | Roughness and canopy from building and vegetation height | NumericalEarth (GHS-BUILT-H, 3D-GloBFP building morphometry) | M | Medium |
| 5 | Differentiability and data assimilation | CliMA (ForwardDiff-compatible microphysics). NumericalEarth (Reactant and Enzyme) | S to M for ensemble parameter estimation; M for a supported LETKF. L for adjoint-based DA | Low for ensembles. Very high for an adjoint |
| 6 | Verification depth | Neither stack publishes a regional multi-season campaign either | M | Medium (observation access, not code) |
| 7 | Stratospheric and middle-atmosphere physics | CliMA (non-orographic gravity-wave drag) | S to M | Low to medium |
| 8 | Multi-GPU scaling | CliMA (published weak scaling to 256 A100s) | M to L | Medium |
| 9 | Calibrated parameterisations | CliMA (EDMF calibrated with ensemble Kalman methods) | M | Medium |

Section 10 lists smaller gaps found while reading: a wind-farm drag scheme
and steep-terrain numerics.

## 1. Ocean and sea-ice coupling

**woof today (documented).**

- Sea-surface temperature is fixed for the whole run. `sst_update`,
  `sst_skin` and `tmn_update` are pinned to 0: "SST update cycling is not
  implemented (single-analysis case runs)" (`woof/namelist_import.py:980-985`,
  `docs/public/CONFIGURATION.md:843`).
- The initial water temperature is chosen carefully. One provider is used
  per connected body of water, and an optional user-supplied
  high-resolution analysis can replace ERA5 SST
  (`docs/water-temperature-overlay.md:3-11`).
- WRF's ocean options (`sf_ocean_physics` 1, a 1-D mixed layer, and 2, a 3-D
  Price-Weller-Pinkel ocean;
  [README.namelist:347](https://github.com/wrf-model/WRF/blob/06d4240ae989cc3e50af412bb472df3d9048783c/run/README.namelist#L347))
  appear in woof only in the packaged WRF registry
  (`woof/data/wrf_namelist/registry_v4.7.1.json`). No woof module implements
  them.
- Sea ice enters as a fixed surface fraction (for example RUC's
  fractional-sea-ice blend, `docs/public/PHYSICS.md:1302-1303`). There is no
  prognostic sea-ice model.
  Lakes can run WRF's CLM lake column (`docs/public/PHYSICS.md:1254-1255`).

**The other stacks (documented).**

- NumericalEarth's `EarthSystemModel` "encapsulates the component
  models—ocean, sea ice, atmosphere"
  ([README.md:72](https://github.com/NumericalEarth/NumericalEarth.jl/blob/66112eac639e6163639b89f3fc1c684b146002d2/README.md#L72)).
  An `AtmosphereOceanModel(atmosphere, ocean; land, radiation)` constructor
  exists
  ([earth_system_model.jl:459](https://github.com/NumericalEarth/NumericalEarth.jl/blob/66112eac639e6163639b89f3fc1c684b146002d2/src/EarthSystemModels/earth_system_model.jl#L459)).
  One example couples a Breeze LES to four oceans: prescribed SST, a 10 m
  slab, a 50 m hydrostatic ocean with CATKE mixing, and a 50 m
  nonhydrostatic LES ocean
  ([breeze_over_four_oceans.jl:1-13](https://github.com/NumericalEarth/NumericalEarth.jl/blob/66112eac639e6163639b89f3fc1c684b146002d2/examples/breeze_over_four_oceans.jl#L1-L13)).
  The slab ocean is
  [src/Oceans/slab_ocean.jl](https://github.com/NumericalEarth/NumericalEarth.jl/blob/66112eac639e6163639b89f3fc1c684b146002d2/src/Oceans/slab_ocean.jl).
  A regional ocean example (Mediterranean, ECCO restoring) is
  [mediterranean_simulation_with_ecco_restoring.jl](https://github.com/NumericalEarth/NumericalEarth.jl/blob/66112eac639e6163639b89f3fc1c684b146002d2/examples/mediterranean_simulation_with_ecco_restoring.jl).
- ClimaOcean's repository description is "Regional-to-global coupled ocean
  and sea ice simulations based on Oceananigans"
  ([github.com/CliMA/ClimaOcean.jl](https://github.com/CliMA/ClimaOcean.jl)).
  Its README says its coupling framework and data utilities "are
  being moved to NumericalEarth.jl", and ClimaOcean stays a focused ocean and
  sea-ice package
  ([README.md:14-15](https://github.com/CliMA/ClimaOcean.jl/blob/940e551532aaaea8995e83ac5a2eddc799054119/README.md#L14-L15)).
- Oceananigans has regional open boundary schemes. They include Flather's
  characteristic condition (named `GravityWaveRadiation`;
  [gravity_wave_schemes.jl:2-10](https://github.com/CliMA/Oceananigans.jl/blob/fec351f0d3657ec3421adfedde30418ee6c0ac3d/src/BoundaryConditions/open_boundary_schemes/gravity_wave_schemes.jl#L2-L10))
  and perturbation advection, in the same
  [open_boundary_schemes](https://github.com/CliMA/Oceananigans.jl/tree/fec351f0d3657ec3421adfedde30418ee6c0ac3d/src/BoundaryConditions/open_boundary_schemes)
  directory.

**Candidate approaches**, in order of increasing cost:

| Step | What | Effort (judgement) | Risk (judgement) |
|---|---|---|---|
| 1a | Time-varying SST and sea ice from the forcing source at the boundary interval: WRF's `sst_update = 1` behaviour. This is not coupling, but it removes the fixed-SST error from multi-day runs. | S | Low. Global forcing sources such as ERA5 and GFS publish SST or skin temperature at each output time; which woof adapters map it at every boundary time needs checking per source. |
| 1b | Port WRF's 1-D ocean mixed layer (`sf_ocean_physics = 1`, used in WRF hurricane forecasting by [Davis et al. 2008](https://doi.org/10.1175/2007MWR2085.1)). Use the same column-oracle pattern the land and PBL ports use. WRF is public domain (`licenses/LICENSE-WRF-public-domain.txt`). | S | Low. It is a single-column scheme with a WRF oracle. |
| 1c | A slab or mixed-layer ocean with prescribed deep-ocean heat flux (Q-flux), shaped like NumericalEarth's `slab_ocean.jl`. It overlaps 1b. Choose one of the two. | S | Low |
| 1d | WRF's 3-D PWP ocean (`sf_ocean_physics = 2`, "no bathymetry" per README.namelist:347). | M | Medium. It is a 3-D ocean grid inside the atmosphere's device memory budget. |
| 1e | Two-way coupling to a regional Oceananigans or ClimaOcean ocean. | L | High. See below. |

Step 1e is designed in the companion page
[ocean-coupling.md](ocean-coupling.md), written as a separate change in this
series. At this level the main points are
(judgement):

- woof is Python and CuPy. Oceananigans is Julia. The coupler needs either an
  in-process bridge, for example [PythonCall.jl / juliacall](https://github.com/JuliaPy/PythonCall.jl),
  or two processes that exchange surface fluxes and SST through files or
  shared memory. Two CUDA runtimes on one card compete for memory, and woof
  sizes its plans to the card.
- The ocean needs its own lateral boundaries and initial state. Oceananigans
  provides the boundary schemes. ClimaOcean and NumericalEarth provide data
  wranglers for ocean reanalyses such as ECCO (the Mediterranean example
  above).
- A coupled run's restart must carry the ocean state, so woof's restart
  contract would need an ocean section.
- [COAWST](https://doi.org/10.1016/j.ocemod.2010.07.010) (WRF with ROMS,
  SWAN and sediment transport) is the established reference for this shape
  of system. It is a useful benchmark for flux-exchange choices, not a
  component to reuse.

## 2. In-run adaptive mesh refinement

**woof today (documented).**

- The WRF-grid path has nests that are created, moved and retired *during*
  a run. A dormant nest is declared in full and materialises when a trigger
  fires mid-run, then follows its feature
  (`docs/nest-spawn-at-trigger.md:1-8`). Retire, re-arm and follow rules are
  in `docs/nest-lifecycle.md:1-11`. The dormant nest is priced in the memory
  plan as if it existed (`docs/nest-spawn-at-trigger.md:10-16`), so this is
  relocatable fixed-shape refinement, not AMR.
- WRF's own moving-nest namelist keys are refused by the loader
  (`woof/experiment.py:94-98`, refusal at `:1976-1990`). The refusal names the
  replacement: a following nest is expressed as discrete relocation by whole
  parent cells, so each move can be checked.
- The hex (MPAS-derived) core has variable-resolution meshes and a
  limited-area lane (`components/hex/README.md:17-26`). It adapts *between*
  cycles: a coarse parent, fine corridors placed inside it, and re-detection
  six hours on (`components/hex/README.md:82-86`). Within a cycle the mesh is
  fixed. A corridor that moves far needs a regenerated parent, which "is not
  built" (`components/hex/README.md:983-985`).

**The other stacks (documented).** Neither has in-run AMR for a real-data
atmosphere. ClimaAtmos offers `sphere`, `column`, `box` and `plane`
configurations
([default_config.yml:138-140](https://github.com/CliMA/ClimaAtmos.jl/blob/dda668d3d49296beb7543aace1bde482b96ee294/config/default_configs/default_config.yml#L138-L140)),
with no lateral-boundary or nesting configuration. NumericalEarth's ERA5
downscaling example drives one Breeze child through open lateral boundaries
and Davies relaxation, at about 12 km
([breeze_downscaling_era5.jl:1-13](https://github.com/NumericalEarth/NumericalEarth.jl/blob/66112eac639e6163639b89f3fc1c684b146002d2/examples/breeze_downscaling_era5.jl#L1-L13)).
For comparison, WRF has specified and vortex-following moving nests, the
latter "experimental"
([README.namelist:370-393](https://github.com/wrf-model/WRF/blob/06d4240ae989cc3e50af412bb472df3d9048783c/run/README.namelist#L370-L393)).
ERF, built on AMReX, "allows both dynamic and static mesh refinement with
sub-cycling in time"
([MeshRefinement.rst:10](https://github.com/erf-model/ERF/blob/92410d578f612ab9e844c450bf8e900194becfb5/Docs/sphinx_doc/MeshRefinement.rst#L10)).

**Candidate approaches.**

| Option | What | Effort (judgement) | Risk (judgement) |
|---|---|---|---|
| 2a | More triggers for the existing spawn and follow machinery: low-level jet, ridge-top wind, frontal gradient, icing indices. This gives most of the value of AMR for energy sites at a fraction of the cost. | M | Medium. Trigger tuning needs verification (section 6). |
| 2b | Hex in-run remeshing: regenerate or re-cull the limited-area mesh mid-run and remap state. The cycle loop already re-detects and regenerates between cycles. Doing it mid-run needs conservative remapping of the full prognostic state. | L | High. Remap conservation and the known hex upper-level warm drift (`components/hex/README.md:909-915`) would compound. |
| 2c | Block-structured AMR in the WRF-grid core, in the style of ERF/AMReX. | L | Very high. It would replace the nest coupler, the memory planner and the restart contract. |

Recommendation (judgement): do 2a. Defer 2b until the hex warm drift is
explained. Do not attempt 2c. If block AMR is ever needed, ERF
(BSD-3-Clause-LBNL,
[LICENSE.md](https://github.com/erf-model/ERF/blob/92410d578f612ab9e844c450bf8e900194becfb5/LICENSE.md))
is the reference to study.

## 3. Sub-10 m terrain and 10 m land use

**woof today (documented).** Domains at 1 km or finer take Copernicus DEM
GLO-30 terrain by default (`docs/public/HIGHRES-TERRAIN.md:14-20`). The
source table (`docs/public/HIGHRES-TERRAIN.md:67-72`, code in
`woof/static/highres_fetch.py:375` and `:517`) is:

- terrain: GLO-30 (~30 m), SRTM 1 arc-second (~30 m), USGS 3DEP 1/3
  arc-second (~10 m, conterminous US);
- land cover: CGLC-MODIS-LCZ (100 m) or Annual NLCD (30 m, US);
- soil: SoilGrids v2 (250 m).

woof already documents and measures the surface-versus-bare-earth issue.
"Copernicus DEM and SRTM are **surface** models: they see forest canopy and
buildings. USGS 3DEP 1/3 arc-second is **bare earth**"
(`docs/public/HIGHRES-TERRAIN.md:385-387`). Over the Colorado Front Range,
GLO-30 reads +3.34 m above 3DEP below 3000 m, +5.43 m in dense subalpine
forest, and −0.07 m above treeline (measured;
`docs/public/HIGHRES-TERRAIN.md:399-409`).

**The other stacks (documented).** NumericalEarth reads GLO-30 and labels it
a "Digital Surface Model (DSM), representing the Earth's surface including
buildings, infrastructure, and vegetation"
([CopernicusDEM.jl:27-29](https://github.com/NumericalEarth/NumericalEarth.jl/blob/66112eac639e6163639b89f3fc1c684b146002d2/src/DataWrangling/CopernicusDEM/CopernicusDEM.jl#L27-L29)).
It also reads ESA WorldCover at 10 m
([WorldCover.jl:90](https://github.com/NumericalEarth/NumericalEarth.jl/blob/66112eac639e6163639b89f3fc1c684b146002d2/src/DataWrangling/WorldCover/WorldCover.jl#L90)).
Its default aggregates 12 x 12 pixels to about 110 m cells
([WorldCover.jl:95-96](https://github.com/NumericalEarth/NumericalEarth.jl/blob/66112eac639e6163639b89f3fc1c684b146002d2/src/DataWrangling/WorldCover/WorldCover.jl#L95-L96)).
No sub-10 m DEM reader was found in either stack.

**Data that would close the gap (documented).**

| Source | Resolution and type | Coverage | Licence |
|---|---|---|---|
| [USGS 3DEP 1 m DEM](https://catalog.data.gov/dataset/usgs-1-meter-digital-elevation-model) | 1 m, bare earth | US, project by project: "seamless within collection projects, but not necessarily seamless across projects" | Public domain |
| [Environment Agency LiDAR Composite DTM 2022](https://developers.google.com/earth-engine/datasets/catalog/UK_EA_ENGLAND_1M_TERRAIN_2022), and the matching [DSM](https://www.data.gov.uk/dataset/cf3f1137-c12b-44a1-a835-e80fe4a60b92/lidar-composite-digital-surface-model-dsm-1m) | 1 m, DTM and DSM | About 99 % of England | Open Government Licence v3 |
| Other national DTMs (Scotland, Wales, the Netherlands, Denmark and others) | 0.5 to 2 m | National | Varies; check each |
| [FABDEM](https://doi.org/10.1088/1748-9326/ac4d4f) (GLO-30 with forests and buildings removed) | 30 m, bare-earth estimate | Global | [CC BY-NC-SA 4.0](https://gee-community-catalog.org/projects/fabdem/): non-commercial only, [commercial licence from Fathom](https://www.fathom.global/insight/fabdem-download/) |
| [ESA WorldCover](https://esa-worldcover.org/en/data-access) | 10 m land cover | Global | CC BY 4.0 |

**Pitfalls (judgement, from the documented measurements above).**

- *DSM vs DTM.* A surface model raises forested and built-up terrain by the
  canopy or building height. A model that also applies canopy roughness
  then counts the obstacle twice: once as terrain, once as roughness. The
  rule should be a bare-earth DTM for terrain, with canopy and buildings
  entering only through roughness and displacement height (section 4).
- *Datums.* GLO-30 is on EGM2008, 3DEP on NAVD88, and the EA LiDAR composite is
  "referenced to Ordinance Survey Newlyn" (its
  [catalogue page](https://developers.google.com/earth-engine/datasets/catalog/UK_EA_ENGLAND_1M_TERRAIN_2022)). woof already records the datum per run
  (`docs/public/HIGHRES-TERRAIN.md:379-383`). A new source must do the same.
- *Grid scale.* A 1 m DEM does not give 1 m terrain on a 50 m grid. The
  aggregation (block mean, plus the sub-grid statistics that `topo_wind` and
  `gwd_opt` need, `docs/public/TERRAIN-DRAG.md:9-10`) decides what the model
  sees. The value of 1 m data at 50 to 100 m grid spacing is mostly better
  sub-grid statistics and an exact bare-earth surface, not finer resolved
  terrain.
- *Steepness.* Finer terrain is steeper terrain. woof's terrain clock already
  reduces the time step or adds acoustic substeps on steep ground
  (`docs/public/CONFIGURATION.md:428-430`). Beyond some slope the
  terrain-following coordinate itself fails (section 10).

**Candidate approaches.**

| Option | What | Effort (judgement) | Risk (judgement) |
|---|---|---|---|
| 3a | ESA WorldCover 10 m as a `landcover_source`, with a class crosswalk to MODIS-21 categories. | S | Low. The crosswalk loses detail (11 WorldCover classes against 21 MODIS classes), and it has no urban LCZ split, so CGLC-MODIS-LCZ should stay the default where urban physics runs. |
| 3b | EA LiDAR DTM and 3DEP 1 m as `terrain_source` values, aggregated to the model grid, with sub-grid variance written for `topo_wind`. | S per source | Low technically. Medium on volume: a 50 x 50 km domain at 1 m is 2.5 billion cells, about 10 GB as uncompressed float32 (arithmetic). |
| 3c | FABDEM as the default bare-earth terrain outside lidar coverage. | S | Licence: non-commercial only unless licensed from Fathom. |

## 4. Roughness and canopy from building and vegetation height

**woof today (documented).** Roughness comes from land-use category tables.
Urban physics (single-layer UCM, BEP, BEP+BEM) is ported
(`docs/public/PHYSICS.md:185-187`), with heights from the URBPARM tables.
Gridded urban morphology is not ingested: "the gridded morphology
(URB_PARAM) that alone could fill another bin count is not ingested"
(`woof/config.py:1431-1436`). Two RUC roughness constants (`ruc.z0.tall`, `ruc.z0.short`) are among the
eight tunable physics constants (`woof/physics_params_registry_v1.json:94`,
`:113`; roughness rows described at `docs/PHYSICS-PARAMETERS.md:21-22`). There is no canopy-height input.

**The other stacks (documented).** NumericalEarth reads GHS-BUILT-H average
building height at 100 m
([GHSL.jl:40-42](https://github.com/NumericalEarth/NumericalEarth.jl/blob/66112eac639e6163639b89f3fc1c684b146002d2/src/DataWrangling/GHSL/GHSL.jl#L40-L42)).
It also rasterises the 3D-GloBFP building footprints and reduces them to
mean height, height standard deviation, maximum height, plan-area index and
frontal-area index: "the height-heterogeneity inputs an urban
aerodynamic-roughness closure (Kanda et al. 2013) is designed for"
([globfp3d_building_morphometry.jl:1-12](https://github.com/NumericalEarth/NumericalEarth.jl/blob/66112eac639e6163639b89f3fc1c684b146002d2/examples/globfp3d_building_morphometry.jl#L1-L12)).

**Data (documented).**

| Source | What | Licence |
|---|---|---|
| [GHS-BUILT-H R2023A](https://developers.google.com/earth-engine/datasets/catalog/JRC_GHSL_P2023A_GHS_BUILT_H) | Building height, 100 m, epoch 2018 | European Commission reuse notice, attribution |
| [Overture buildings](https://docs.overturemaps.org/attribution/) | Footprints and heights | ODbL (share-alike on derived databases) |
| [Microsoft Global ML Building Footprints](https://github.com/microsoft/GlobalMLBuildingFootprints) | Footprints | CDLA-Permissive-2.0 |
| [Meta/WRI global canopy height](https://registry.opendata.aws/dataforgood-fb-forests/) ([Tolan et al. 2024](https://doi.org/10.1016/j.rse.2023.113888)) | Canopy height, 1 m | CC BY 4.0, plus Maxar imagery attribution |
| [ETH global canopy height](https://doi.org/10.1038/s41559-023-02206-6) (Lang et al. 2023) | Canopy height, 10 m, 2020 | The paper is CC BY 4.0. No explicit data licence was found; check before use. |
| EA LiDAR DSM minus DTM (section 3) | Obstacle height, 1 m, England | Open Government Licence v3 |

**Candidate approaches.**

| Option | What | Effort (judgement) | Risk (judgement) |
|---|---|---|---|
| 4a | Gridded vegetation roughness z0 and displacement height d from canopy height and LAI, using [Raupach (1994)](https://doi.org/10.1007/BF00709229), replacing the table z0 where a canopy map exists. | M | Medium. Each land surface scheme (Noah, Noah-MP, RUC) and the MYNN and YSU surface layers read z0 differently. A gridded override must enter each, and the WRF parity tests must stay green with the override off. |
| 4b | Ingest WRF's gridded urban morphology (`URB_PARAM`: building-height histogram, plan-area and frontal-area indices) from building data, so BEP and BEP+BEM run on real morphology. The WRF format already exists, so this is ingestion, not new physics. | M | Medium. The BEP kernels index a fixed 15-bin histogram (`woof/config.py:1431-1436`). |
| 4c | An urban aerodynamic closure from building statistics ([Kanda et al. 2013](https://doi.org/10.1007/s10546-013-9818-x)) for the bulk-urban case without BEP. | S after 4b | Medium. Building heights from machine learning are biased low; the NumericalEarth example states RMSE 1.9–14.6 m. |

## 5. Differentiability and data assimilation

**woof today (documented).** woof has ensemble data assimilation, not
gradients. A batched GPU LETKF exists but is "EXPERIMENTAL. Nothing in the
forecast path calls this module" (`woof/da/letkf.py:1-4`). A radar DA nowcast
demo ran ten members and was scored against the operational HRRR on one case
(`docs/da-vs-wofs.md:1-25`). The CUDA kernels are hand-written and have no
tangent-linear or adjoint code.

**The other stacks (documented).** CloudMicrophysics.jl is "AD-compatible
(ForwardDiff.jl) for differentiable physics"
([README.md:63](https://github.com/CliMA/CloudMicrophysics.jl/blob/542dd84950373248eb53e49528a08b0fdb00f9b7/README.md#L63)).
Breeze has a Reactant extension
([ext/BreezeReactantExt](https://github.com/NumericalEarth/Breeze.jl/tree/6eabb8937670f61ad3bfd0ca2ad400a747681646/ext/BreezeReactantExt))
and a reverse-mode Enzyme gradient test through an advection-diffusion model
([advection_diffusion_ad.jl:12, 105-109](https://github.com/NumericalEarth/Breeze.jl/blob/6eabb8937670f61ad3bfd0ca2ad400a747681646/test/reactant/advection_diffusion_ad.jl#L105-L109)).
NumericalEarth has a Reactant extension for its coupled model
([NumericalEarthReactantExt.jl](https://github.com/NumericalEarth/NumericalEarth.jl/blob/66112eac639e6163639b89f3fc1c684b146002d2/ext/NumericalEarthReactantExt.jl)).
These show gradients through parts of the model. This review did not find a
published 4D-Var or full-physics adjoint in either stack.

**Candidate approaches.**

| Option | What | Effort (judgement) | Risk (judgement) |
|---|---|---|---|
| 5a | Promote the LETKF to a supported path for surface, tall-tower, lidar and radar observations relevant to wind. | M | Medium. The algorithm is in place. Observation operators and quality control are the work. |
| 5b | Parameter estimation with augmented-state or ensemble Kalman inversion (section 9), using the existing physics parameter sets and several runs per card (`docs/PHYSICS-PARAMETERS.md:1-6`; `runs_per_gpu`, `docs/public/HARDWARE.md:351`). | S to M | Low. No gradients are needed. |
| 5c | A differentiable column for one scheme (for example the surface layer), written in a framework with automatic differentiation and checked against the CUDA kernel. This is useful for sensitivity studies. | M | Medium. A second implementation can drift from the first. woof's oracle tests are the control. |
| 5d | Tangent-linear and adjoint of the full model for 4D-Var. | L | Very high. Every CUDA kernel would need an adjoint. Not recommended. |

## 6. Verification depth

**woof today (documented).** Verification against WRF is extensive.
Verification against observations is thin, and the project says so. "No
published run-specific numerical-error budget" exists for weather forecasts
(`docs/public/VERIFICATION.md:49`). Observation scoring is limited to some
ASOS case scoring, a small fog comparison and one MRMS reflectivity A/B
(`docs/public/VERIFICATION.md:50`). "A broad observation campaign with
stated uncertainty and coverage remains a gap"
(`docs/public/VERIFICATION.md:57-59`).

**The other stacks.** Neither publishes a regional, multi-season,
observation-based verification campaign for its limited-area configuration
in the repositories read here. The ERA5 downscaling example in
NumericalEarth is a single case (section 2). This is not a gap relative to
them. It is the gap that most limits claims about woof's accuracy over
rough terrain.

**Candidate approach.** A campaign designed around the target uses
(judgement):

- *Sites*: tall towers and lidar profilers in complex terrain, for example
  the [Perdigão](https://doi.org/10.1175/BAMS-D-17-0227.1) and
  [WFIP2](https://doi.org/10.1175/BAMS-D-18-0036.1) field campaigns, plus offshore masts and
  buoys for the coupled-ocean case.
- *Seasons*: at least four seasons, with stable winter nights over-sampled,
  because they are where 100 m grids do least well
  (`docs/energy-forecasts.md:711-713`).
- *Design*: preregistered scores (hub-height wind speed error, ramp
  detection, direction error) with a fixed control configuration. Report
  every arm.

Effort: M (judgement). Most of the cost is observation access and cleaning,
not code. Risk: medium. A campaign can return an unflattering answer, and
that answer is the point.

## 7. Stratospheric and middle-atmosphere physics

**woof today (documented).** woof runs up to 256 levels. Columns of 256
levels to a 1 hPa top (roughly 48 km) run through the shipped physics
launchers with finite results (`docs/kernel_local_memory_bounds.md:736-738`,
`docs/public/PHYSICS.md:2241-2246`). The UK forecast configurations added in
commit `4adb152b` (later removed from the tree) used 115 levels to a 50 hPa
top. This review found no receipt in the tree for a real-data run at a 1 hPa
top. Other relevant settings:

- Ozone is climatological on both radiation arms. The default RTE+RRTMGP
  arm interpolates one median RFMIP ozone profile in log pressure, the same
  profile at every latitude and date (`woof/core/rrtmgp.py:3142-3145`,
  `:3305-3306`). The legacy RRTMG arm uses the CAM climatology by
  default (`o3input = 2`, `docs/public/CONFIGURATION.md:836`), or the RRTMG
  wrapper's built-in profile with `o3input = 0` (`woof/config.py:4583-4586`). Neither reads ozone
  from the forcing source, although sources such as ERA5 carry an ozone
  mixing ratio
  ([ERA5 data documentation, Table 9: "Ozone mass mixing ratio", paramId 203](https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation)).
- Gravity-wave drag is orographic only. `gwd_opt` 1 and 3 are ported
  (`docs/public/TERRAIN-DRAG.md:10`). WRF's option 3 is also orographic: "NOAA/GSL
  gravity wave drag and turbulent orographic form drag"
  ([README.namelist:1741-1743](https://github.com/wrf-model/WRF/blob/06d4240ae989cc3e50af412bb472df3d9048783c/run/README.namelist#L1741-L1743)).
  WRF has no non-orographic option to port.
- The upper sponge is Rayleigh damping of `w` (`damp_opt = 3`,
  `docs/public/CONFIGURATION.md:450-452`).
- The hex core warms above level 45 by +0.019 K/h, reaching +0.46 K at 24 h
  (measured, `components/hex/README.md:909-915`).

Note: the "Fixed by WOOF" table row `gwd_opt | 0 | no gravity-wave drag`
(`docs/public/CONFIGURATION.md:839`) disagrees with
`docs/public/TERRAIN-DRAG.md`, which documents options 1 and 3. That row
appears to be stale.

**The other stacks (documented).** ClimaAtmos has orographic and
non-orographic gravity-wave drag
([default_config.yml:346-351](https://github.com/CliMA/ClimaAtmos.jl/blob/dda668d3d49296beb7543aace1bde482b96ee294/config/default_configs/default_config.yml#L346-L351)),
and its AMIP target configuration turns on the non-orographic scheme with a
convective (Beres) source
([amip_target.yml:21-23](https://github.com/CliMA/ClimaAtmos.jl/blob/dda668d3d49296beb7543aace1bde482b96ee294/config/longrun_configs/amip_target.yml#L21-L23)).
Its finest sphere numerics configuration has 63 vertical elements to a 60 km
top with a 30 m bottom layer
([numerics_sphere_he30ze63.yml:1-6](https://github.com/CliMA/ClimaAtmos.jl/blob/dda668d3d49296beb7543aace1bde482b96ee294/config/common_configs/numerics_sphere_he30ze63.yml#L1-L6)).
woof has more levels and a lower top. ClimaAtmos has the parameterisation a
high top needs.

**Candidate approaches.**

| Option | What | Effort (judgement) | Risk (judgement) |
|---|---|---|---|
| 7a | Ozone from the forcing source where it exists, with a latitude- and season-dependent climatology as fallback on the RTE+RRTMGP arm. Ozone heating is the main shortwave heating of the stratosphere, so a single global profile matters more as the top rises (judgement). | S to M | Low. The RRTMGP arm must move from one 1-D profile to a 3-D field, and a source field must be carried through the boundary updates. |
| 7b | A non-orographic GWD scheme (convective source and a background spectrum), using ClimaAtmos's implementation (Apache-2.0) as the reference. | M | Medium. Its effect on 1 to 3 day regional runs is small, because the lateral boundaries constrain the stratosphere. Its value grows with run length and model top height (judgement). |
| 7c | Document and test the sponge layer for tops above 10 hPa, before 7b. | S | Low |

## 8. Multi-GPU scaling

**woof today (measured).** On 8 consumer RTX 4090 cards without peer-to-peer
links, the halo exchange was bound by the number of transfers, not by
bandwidth. Two GPUs were slower than one (0.87x) at 9.63 Mcell
(`tilestream/EXCHANGE-IS-COUNT-BOUND.md:9-16`). Eight-card efficiency was
18.8 % at 9.63 Mcell and 52.6 % at 154.1 Mcell
(`tilestream/EXCHANGE-IS-COUNT-BOUND.md:54-55`). A packed exchange is
now the default. It beats the unpacked exchange by about 1.4x on the
exchange itself on two cards (`tilestream/EXCHANGE-PACK-DISPOSITION.md:13-22`).
End to end on four RTX 3090 cards, the packed default measured 2 to 4 %
*slower* than the direct path, a difference the page leaves unattributed,
because the exchange there is only 0.2 to 0.7 % of a step
(`tilestream/EXCHANGE-PACK-DISPOSITION.md:24-32`). So on that box the
exchange is no longer the limit, and packing has not shown an end-to-end
gain. The remaining limits are:

- the Python interpreter lock (radiation steps took 13–25 s under the lock
  and 5–8 s without it; `docs/public/MULTIGPU.md:116-121`);
- the nest coupler draining split grids through the host every parent step
  (`docs/public/MULTIGPU.md:52-57`);
- hex multi-GPU "built and proven but not shipped"
  (`components/hex/README.md:999-1001`).

No current-engine weak or strong scaling curve is published.

**The other stacks (documented).** ClimaCore reports "weak-scaling
efficiency above 92% on GPUs"
([README.md:55](https://github.com/CliMA/ClimaCore.jl/blob/d7f5b7954be71f034d5133a8ab8dc98d2c3bfab9/README.md#L55)).
Its performance page places that on Derecho A100s, from 103 km on 1 GPU to
6 km on 256 GPUs. The 0.20 simulated years per day at 6 km figure is on 256
H100s
([performance page](https://clima.github.io/ClimaCore.jl/stable/explanation/performance/)).
These are global runs on data-centre GPUs with NVLink and MPI. woof's
numbers are regional runs on consumer cards over PCIe. The two are not
directly comparable.

**Candidate approaches.**

| Option | What | Effort (judgement) | Risk (judgement) |
|---|---|---|---|
| 8a | Publish a current-engine weak and strong scaling curve on one consumer box and one data-centre box. | S | Low |
| 8b | Make the free-threaded Python build the supported multi-card runtime, then measure. | S to M | Medium. It depends on CuPy, NumPy and netCDF4 support for the free-threaded build. |
| 8c | Read only the coupler's windows from split slabs instead of draining whole grids (named as "the next step for nested speed", `docs/public/MULTIGPU.md:56-57`). | M | Medium |
| 8d | Multi-node (MPI or NCCL) for the WRF-grid core. | L | High |

## 9. Calibrated parameterisations

**woof today (documented).** woof runs WRF's parameterisations with WRF's
constants. Eight constants (five MYNN, three RUC;
`woof/physics_params_registry_v1.json:6-132`) can be changed through parameter
sets, and "No tuned set is a new default" (`docs/PHYSICS-PARAMETERS.md:6`).
Stochastic physics (SPPT, SKEBS, SPP) is refused "until their amplitudes are
calibrated against observations" (`docs/ensemble-wrf-stochastic-import.md:3-8`).

**The other stacks (documented).** CliMA calibrates its EDMF scheme with
gradient-free ensemble Kalman methods against LES
([Lopez-Gomez et al. 2022](https://doi.org/10.1029/2022MS003105); scheme:
[Cohen et al. 2020](https://doi.org/10.1029/2020MS002162)). The tool is
[EnsembleKalmanProcesses.jl](https://github.com/CliMA/EnsembleKalmanProcesses.jl)
([Dunbar et al. 2022](https://doi.org/10.21105/joss.04869)), with
[CalibrateEmulateSample.jl](https://github.com/CliMA/CalibrateEmulateSample.jl)
for uncertainty quantification. Both are Apache-2.0.

**Candidate approach (judgement).** Use ensemble Kalman inversion over the
existing parameter-set mechanism. Each ensemble member is one woof run with
a parameter file. The scoring target is either woof's own LES at 50 to
100 m (`docs/public/LES.md:9-14`, `:106-110`) or the observations from
section 6. EnsembleKalmanProcesses.jl can drive this from outside the model,
because it needs only parameters in and scores out. Start with roughness
and surface-layer constants for rough terrain, then the MYNN mixing length.
Effort: M. Risk: medium. Calibration against one region or season can
overfit. The section 6 campaign is the guard.

## 10. Smaller gaps found while reading

- **No wind-farm parameterisation.** WRF has `windfarm_opt` 1
  ([Fitch et al. 2012](https://doi.org/10.1175/MWR-D-11-00352.1)) and 2
  ([README.namelist:1155](https://github.com/wrf-model/WRF/blob/06d4240ae989cc3e50af412bb472df3d9048783c/run/README.namelist#L1155)).
  woof's `windfarm` name is a history-output preset only
  (`woof/energy/cost.py:341-343`). No turbine-drag scheme was found in
  either CliMA or NumericalEarth. Porting Fitch is S to M (judgement), with WRF as the
  oracle. It matters for wakes in farm clusters.
- **Steep terrain.** woof, WRF and Breeze all use terrain-following
  coordinates (Breeze:
  [README.md:89](https://github.com/NumericalEarth/Breeze.jl/blob/6eabb8937670f61ad3bfd0ca2ad400a747681646/README.md#L89)).
  At 50 m spacing over real terrain, slopes exceed what terrain-following
  coordinates handle well. WRF has a research immersed-boundary method for
  this ([Lundquist et al. 2010](https://doi.org/10.1175/2009MWR2990.1),
  [2012](https://doi.org/10.1175/MWR-D-11-00311.1)). Oceananigans has
  immersed-boundary grids
  ([src/ImmersedBoundaries](https://github.com/CliMA/Oceananigans.jl/tree/fec351f0d3657ec3421adfedde30418ee6c0ac3d/src/ImmersedBoundaries)).
  An immersed boundary in woof is L with high risk (judgement). The terrain
  clock already guards stability (`docs/public/CONFIGURATION.md:428-430`).

## Reusable external components

Licences were read from each repository's LICENSE file on 2026-10-15. woof
is Apache-2.0 (`pyproject.toml:11`). This is not legal advice.

| Component | Licence | What woof could reuse | How (judgement) |
|---|---|---|---|
| [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl/blob/fec351f0d3657ec3421adfedde30418ee6c0ac3d/LICENSE) | MIT | Regional ocean, open boundaries, immersed boundaries | Run as a coupled component (section 1, step 1e) |
| [ClimaOcean.jl](https://github.com/CliMA/ClimaOcean.jl/blob/940e551532aaaea8995e83ac5a2eddc799054119/LICENSE) | MIT (file text; GitHub's API reports NOASSERTION) | Realistic ocean and sea-ice set-ups | Coupled component |
| [ClimaSeaIce.jl](https://github.com/CliMA/ClimaSeaIce.jl) | Apache-2.0 | Prognostic sea ice | Coupled component, after the ocean |
| [NumericalEarth.jl](https://github.com/NumericalEarth/NumericalEarth.jl/blob/66112eac639e6163639b89f3fc1c684b146002d2/LICENSE) | MIT (file text, inherited from ClimaOcean; GitHub's API reports NOASSERTION) | Coupling framework, slab ocean, data wranglers for WorldCover, GLO-30, GHSL and 3D-GloBFP | Study, or call for static preprocessing |
| [Breeze.jl](https://github.com/NumericalEarth/Breeze.jl/blob/6eabb8937670f61ad3bfd0ca2ad400a747681646/LICENSE) | Apache-2.0 | LES reference for intercomparison | Independent check on woof's LES |
| [RRTMGP.jl](https://github.com/CliMA/RRTMGP.jl) | Apache-2.0 | Independent RTE+RRTMGP implementation ([Pincus et al. 2019](https://doi.org/10.1029/2019MS001621)) | Cross-check for woof's own RTE+RRTMGP coupling (`docs/public/PHYSICS.md:1348`) |
| [CloudMicrophysics.jl](https://github.com/CliMA/CloudMicrophysics.jl) | Apache-2.0 | 0-moment, 1-moment, Seifert-Beheng 2-moment and P3 ([README.md:43-46](https://github.com/CliMA/CloudMicrophysics.jl/blob/542dd84950373248eb53e49528a08b0fdb00f9b7/README.md#L43-L46)) | Cross-check for woof's WRF P3 port (`docs/public/PHYSICS.md:263`) |
| [ClimaAtmos.jl](https://github.com/CliMA/ClimaAtmos.jl) | Apache-2.0 | Non-orographic GWD | Reference for a CUDA port (section 7, 7b) |
| [EnsembleKalmanProcesses.jl](https://github.com/CliMA/EnsembleKalmanProcesses.jl), [CalibrateEmulateSample.jl](https://github.com/CliMA/CalibrateEmulateSample.jl) | Apache-2.0 | Ensemble Kalman inversion and uncertainty quantification | External driver over woof parameter sets (section 9) |
| [ERF](https://github.com/erf-model/ERF/blob/92410d578f612ab9e844c450bf8e900194becfb5/LICENSE.md) | BSD-3-Clause-LBNL | Block AMR design | Study only |
| [WRF](https://github.com/wrf-model/WRF) | Public domain (`licenses/LICENSE-WRF-public-domain.txt`) | Ocean mixed layer, PWP ocean, wind-farm schemes | Port with the existing oracle pattern |

MIT, Apache-2.0 and BSD licences are compatible with woof's Apache-2.0
licence for both calling and porting, provided notices are kept. Data
licences are more restrictive: FABDEM is non-commercial, and Overture's ODbL
is share-alike on derived databases. Check these before shipping derived
static fields.

## What woof does that the others do not

| Capability | woof (documented) | CliMA and NumericalEarth (documented) |
|---|---|---|
| Real-data lateral boundaries from many sources | 23 named sources: HRRR, RAP, RRFS, GFS, GDAS, GEFS, ECMWF open data, AIFS, ICON global, ICON-EU and ICON-D2, GEM, Met Office global, ERA5 and ERA5 model levels, 20CRv3 and others (`docs/public/SOURCES.md:21-43`), plus a generic mapped-source adapter (`docs/public/SOURCES.md:150`) | ClimaAtmos: no lateral boundaries (configurations at `default_config.yml:138-140`). NumericalEarth: ERA5 for the Breeze nest (`breeze_downscaling_era5.jl:1-13`) |
| Nesting | Static nests, one-way and experimental two-way (`docs/public/CONFIGURATION.md:140`); trigger-spawned and following nests (`docs/nest-spawn-at-trigger.md:1-8`); offline downscaling (`docs/public/DOWNSCALE.md:1-6`); out-of-core tiles (`docs/public/TILES.md:1-3`); hex variable-resolution meshes (`components/hex/README.md:17-26`) | One nest in the NumericalEarth example (above) |
| WRF physics suites | Ported WRF microphysics, PBL, surface layer, land surface (Noah, Noah-MP, RUC, CLM lake), urban and radiation, each with its evidence tier (`docs/public/PHYSICS.md`) | Own physics: CloudMicrophysics, RRTMGP.jl, EDMF, TKE closures |
| Wind-energy products | Hub-height wind every step with interval statistics (`docs/energy-hindcast/hub-wind.md:1-8`), power, ramps, gusts and icing pages under `docs/energy-hindcast/`, and an asset-driven planner for 100 m and 50 m domains (`docs/energy-forecasts.md:1-20`) | None found in either stack |
| Deep columns | Up to 256 levels; 1 hPa top exercised at column level (`docs/kernel_local_memory_bounds.md:736-738`) | ClimaAtmos's finest sphere configuration: 63 elements to 60 km (section 7) |
| High-resolution static data | GLO-30, SRTM, 3DEP, CGLC-MODIS-LCZ, Annual NLCD and SoilGrids, fetched per domain and hash-recorded (`docs/high-resolution-static-geography.md:16-27`) | NumericalEarth: GLO-30, WorldCover, GHSL, 3D-GloBFP (sections 3 and 4). It has the 10 m land cover and building data that woof lacks; woof has 3DEP, soil texture and the per-run provenance |
| LES | 100 m and 50 m idealised receipts, and a 250 m child in a real terrain-following tree (`docs/public/LES.md:9-14`, `:106-110`) | Breeze: LES to mesoscale, anelastic and compressible ([README.md:87-94](https://github.com/NumericalEarth/Breeze.jl/blob/6eabb8937670f61ad3bfd0ca2ad400a747681646/README.md#L87-L94)) |

The comparison is not one-sided. CliMA and NumericalEarth lead on coupling,
differentiability, calibration and scaling. woof leads on real-data regional
forecasting: boundary sources, nesting, WRF physics and energy outputs.

## Prioritised roadmap: top five

Ranked by value per unit effort for the two goals: regional
ocean-plus-atmosphere modelling, and rough-terrain wind accuracy. The
ranking is a judgement.

| Rank | Item | Goal | Effort | Why this rank |
|---|---|---|---|---|
| 1 | **Observation verification campaign for rough-terrain wind** (section 6): towers and lidar in complex terrain and offshore, four seasons, preregistered scores | Both | M | Without it, no improvement in items 2–5 can be measured. It also produces the calibration target for item 4. |
| 2 | **Bare-earth terrain and gridded roughness** (sections 3 and 4): EA LiDAR and 3DEP 1 m DTMs, FABDEM where licensed, WorldCover 10 m, canopy-height z0 and d, then `URB_PARAM` from building data | Rough-terrain wind | S for the data sources; M for roughness | It removes the known DSM double-counting (+3 to +5 m over forest, measured) and replaces category-table roughness, which is likely the largest controllable error source for near-surface wind over forest and towns (judgement). |
| 3 | **Time-varying SST, then a mixed-layer ocean** (section 1, steps 1a and 1b) | Ocean plus atmosphere | S, then S | Cheapest move from fixed SST to an ocean that responds. It is a prerequisite for, and a control arm against, the full coupling. |
| 4 | **Ensemble Kalman calibration of surface-layer and PBL constants** (section 9) | Rough-terrain wind | M | Uses existing parameter sets and several runs per card. Scores against item 1 and woof's own LES. |
| 5 | **Regional 3-D ocean coupling via Oceananigans or ClimaOcean** (section 1, step 1e; design in [ocean-coupling.md](ocean-coupling.md)) | Ocean plus atmosphere | L | Largest value for coastal and offshore work, and largest risk. It comes after item 3 so there is a slab-ocean control to beat. |

Deliberately left out of the top five (judgement): in-run AMR (the
spawn-and-follow nests cover most of the energy use case), adjoint-based DA
(very high cost), non-orographic GWD (small effect in short regional runs),
and multi-node scaling (the single-box limits in section 8 come first).
