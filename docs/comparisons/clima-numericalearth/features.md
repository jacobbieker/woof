# WOOF, CliMA and NumericalEarth: feature comparison for regional and limited-area modelling

This page compares WOOF with two GPU-native Julia stacks, criterion by
criterion:

- **CliMA**: ClimaAtmos.jl, with ClimaCoupler.jl, ClimaOcean.jl,
  ClimaSeaIce.jl and ClimaLand.jl.
- **NumericalEarth**: Breeze.jl and NumericalEarth.jl, built on
  Oceananigans.jl.

The question it answers is which of the three can run a **regional or
limited-area** forecast. It covers:

- multi-scale grids;
- deep vertical columns into the stratosphere;
- 50 to 100 m runs;
- metre-scale terrain and land use;
- a coupled regional ocean;
- cloud and radiation physics;
- the outputs an energy user needs.

Every cell cites its source. External sources are pinned to a commit or a
paper. WOOF sources are given as `path:line` on the snapshot below.

This page records **capability and maturity**. It does not measure speed or
accuracy. Its sibling pages do that:

| Page | Covers | State at this snapshot |
| --- | --- | --- |
| [les-benchmark.md](les-benchmark.md) | 50 to 100 m LES cases run on more than one stack | woof and Breeze run the FastEddy dry CBL at 128²×64 for 2 h; full-size clean-card timing pending (`tools/compare/breeze_cbl/run_timing.sh`) |
| [idealized-benchmark.md](idealized-benchmark.md) | idealized dynamical-core cases | planned |
| [regional-benchmark.md](regional-benchmark.md) | a real-data regional case, timed and scored | planned |
| [ocean-coupling.md](ocean-coupling.md) | regional atmosphere and ocean coupling in depth | planned |
| [gaps-and-roadmap.md](gaps-and-roadmap.md) | what WOOF would need to close the gaps found here | drafted in a separate pull request (#76) |

"Planned" means another piece of work in the same comparison is writing that
page. The link resolves once that page lands.

## Snapshot

Taken on 2026-10-15. "HEAD" is the default branch on that date. Every
file link below is pinned to one of these commits. Issue and pull-request
links, the GitHub repository description and the BreezeBenchmarks dashboard
are live pages; their state is recorded as of the snapshot date.

| Project | Release | HEAD commit | Licence |
| --- | --- | --- | --- |
| WOOF (fork `jacobbieker/woof`, `main`) | 1.0.3 line | `b4a3783a5e13` (2026-10-15) | Apache-2.0 |
| [CliMA/ClimaAtmos.jl](https://github.com/CliMA/ClimaAtmos.jl) | v0.42.14 (2026-10-10) | `dda668d3d492` | Apache-2.0 |
| [CliMA/ClimaCoupler.jl](https://github.com/CliMA/ClimaCoupler.jl) | v0.2.3 (2026-08-18) | `e3355c8ac9dd` | Apache-2.0 |
| [CliMA/ClimaOcean.jl](https://github.com/CliMA/ClimaOcean.jl) | v0.10.0 (2026-06-10) | `940e551532aa` | MIT |
| [CliMA/ClimaLand.jl](https://github.com/CliMA/ClimaLand.jl) | v1.13.0 (2026-10-10) | `692452613025` | Apache-2.0 |
| [CliMA/ClimaSeaIce.jl](https://github.com/CliMA/ClimaSeaIce.jl) | v0.5.12 (2026-10-12) | `4e7958d35c66` | Apache-2.0 |
| [CliMA/RRTMGP.jl](https://github.com/CliMA/RRTMGP.jl) | v1.1.0 (2026-10-10) | `19c6355a9629` | Apache-2.0 |
| [CliMA/CloudMicrophysics.jl](https://github.com/CliMA/CloudMicrophysics.jl) | v0.45.0 (2026-10-12) | `542dd8495037` | Apache-2.0 |
| [CliMA/ClimaDiagnostics.jl](https://github.com/CliMA/ClimaDiagnostics.jl) | v0.3.11 (2026-09-16) | `7fc5f050c927` | Apache-2.0 |
| [CliMA/ClimaInitialConditions.jl](https://github.com/CliMA/ClimaInitialConditions.jl) | v0.1.0 (2026-09-27) | `6c1c1588c784` | Apache-2.0 |
| [CliMA/Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl) | v0.114.0 (2026-10-14) | `fec351f0d365` | MIT |
| [NumericalEarth/Breeze.jl](https://github.com/NumericalEarth/Breeze.jl) | v0.12.0 (2026-10-10) | `6eabb8937670` | Apache-2.0 |
| [NumericalEarth/NumericalEarth.jl](https://github.com/NumericalEarth/NumericalEarth.jl) | v0.8.2 (2026-10-09) | `66112eac639e` | MIT (the LICENSE file opens with the MIT text and then carries the Apache-2.0 text; GitHub reports NOASSERTION) |
| [NumericalEarth/Terrarium.jl](https://github.com/NumericalEarth/Terrarium.jl) | v0.1.8 (2026-10-07) | `a8a0c35dae8a` | EUPL-1.2 |
| [NumericalEarth/BreezeBenchmarks](https://github.com/NumericalEarth/BreezeBenchmarks) | none | `626d58016751` | none stated |
| [glwagner/AtmosphericRivers.jl](https://github.com/glwagner/AtmosphericRivers.jl) | none | `7cc5735d2ffe` (`main`); `af7b94768b02` (branch `glw/precip-sensitivity-adjoint`, 2026-10-13) | none stated |

Notes on the snapshot:

- Oceananigans is still canonical at `CliMA/Oceananigans.jl`.
  `NumericalEarth/Oceananigans.jl` is a stale fork.
- ClimaOcean itself stays at CliMA. Its README says its generic coupling
  framework and data-wrangling utilities "are being moved to NumericalEarth.jl"
  [C19]. The two orgs therefore share a numerical base, and NumericalEarth.jl is
  taking over ClimaOcean's coupling and data code.
- Licences come from the GitHub API and the LICENSE files at these commits.

## How to read the labels

| Label | Meaning here |
| --- | --- |
| **verified** | The project publishes a comparison for this capability against a reference code or observations, and the comparison is cited. The label covers the compared configuration, not the whole capability. |
| **implemented** | The capability is in released code and documented, with no published comparison for it. WOOF's own term for this is "implemented-unverified" ([W30]). |
| **experimental** | The project itself calls the capability experimental, preview, "under development", "not run in CI" or "not production-ready", or it has open defects that block it, or it exists only in an example or an open pull request. |
| **absent** | Nothing implements the capability at the snapshot. The cell says what was searched. |

Numbers carry their kind:

- *documented*: a project states the number in its own README, docs or
  paper.
- *measured here*: computed for this page from a run record; the method is
  given.
- *estimated*: derived arithmetic; the inputs are given.

Speed numbers from different stacks use different grids, physics and cards.
**They are not comparable with each other.** Like-for-like timing belongs to
the benchmark pages above.

## What "regional" means in each stack

The three stacks use "regional" and "limited-area" for different things. The
matrix below cannot be read correctly without this distinction.

**WOOF: real-data limited area with lateral boundaries from an outside model.**

- WOOF is a WRF-ARW-class regional model. Its Production model is the
  regional one ([W1]).
- A run has a map-projected outer domain. Its edges are driven by a parent
  forecast or reanalysis through WRF-style specified and relaxation zones.
  The UK run below uses `spec_zone = 1`, `relax_zone = 4` ([W16]).
- 24 runnable source models (3 of them certified) plus a generic mapped
  door can supply initial and boundary data ([W24]).
- Finer resolution comes from:
  - one-way nests inside the same run ([W3]);
  - offline children re-run from an archived parent ([W8]);
  - variable-resolution hex meshes in the preview hex model ([W10]).

**CliMA (ClimaAtmos): global or doubly periodic. No lateral boundary
conditions exist.**

- ClimaAtmos describes itself as "a GPU-capable global atmosphere model" [C1].
- Its grids are `SphereGrid`, `ColumnGrid`, `BoxGrid` and `PlaneGrid` [C4].
- On the YAML configuration path, box and plane grids are hard-coded
  periodic: `periodic_x = true, periodic_y = true` [C5].
- The script API accepts non-periodic flags, and ClimaCore can tag box edges.
  However, ClimaAtmos has no lateral boundary tendency, relaxation zone or
  boundary forcing to put on those edges.
- Searches of ClimaAtmos source, docs and configs, and of its issues, PRs and
  Discussions, for "lateral boundary", "limited area", "nesting", "regional"
  and "Davies" found no limited-area work [C5].
- "Regional" in the CliMA stack means ClimaLand's "regional boxes", which are
  land-only [C21].
- A CliMA atmosphere runs on a real region only as part of a global run. That
  run is about 100 km in the weather configurations [C7].

**NumericalEarth (Breeze): real-data limited area, from ERA5 only, as one
experimental nest.**

- Breeze's GitHub description is "Limited area LES-to-mesoscale atmosphere
  simulations based on Oceananigans" [N1].
- Its LES cases are doubly periodic and flat: `(Periodic, Periodic, Bounded)`
  [N10].
- Real-data limited-area runs go through NumericalEarth.jl's
  `nested_atmosphere_model` [N4]. In that setup:
  - An ERA5 `PrescribedAtmosphere` on its native 0.25° pressure-level grid is
    the parent.
  - It drives a Breeze child "through open lateral boundary conditions +
    interior Davies relaxation" [N5].
  - Davies relaxation is opt-in. Its default is `relaxation_rate = nothing`
    [N4].
- The one shipped example says "Single nest only (ERA5 → 12 km …)". It also
  says there is "no boundary-layer or cumulus parameterization" [N5].
- Open issues remain on the open-boundary path:
  - spurious vertical velocity at the lateral boundaries (#825, open);
  - split-explicit cold-start blow-ups (#793, open).
  - The terrain-nest NaN issue (#897) was closed on 2026-09-08 [N7].
- ERA5 is the only boundary source. NumericalEarth has no GRIB, GFS, HRRR,
  ICON or IFS reader [N14].

## The matrix

Cells are short. The [notes](#notes-by-criterion) below the matrix carry the
detail and the corrections. W = WOOF, C = CliMA, N = NumericalEarth.
[Sources](#sources) lists every ID.

| # | Criterion | WOOF | CliMA (ClimaAtmos + ClimaCoupler) | NumericalEarth (Breeze + NumericalEarth.jl) | Bottom line |
| --- | --- | --- | --- | --- | --- |
| 1 | Real-data limited-area runs with lateral boundaries | **implemented** (Production model): WRF-style specified and relaxation zones on a projected domain; 24 runnable driving models (3 certified) plus a generic mapped door [W1][W24]. WRF code comparison exists, but the t=0 state comparison FAILs ([W40]). | **absent**: global sphere, or periodic box/plane/column only [C4][C5]. | **experimental**: ERA5 parent to one Breeze nest through open boundaries and opt-in Davies relaxation [N4][N5]. Open LBC defects #825 and #793 [N7]. | Only WOOF runs regional real-data forecasts as its main product. Breeze can, experimentally, from ERA5. CliMA cannot. |
| 2 | Multi-scale grids: nesting and variable resolution, as used by the energy planners | **implemented** one-way static nests. 21 domains is the stock-WRF export cap, not an engine limit ([W5]). **experimental** two-way feedback ([W3]). Offline downscale children with their own level ladder ([W8]). Energy planners `wrf-nests`, `wrf-tiles` and `hex-swath` ([W6][W7]). **experimental** (preview) hex variable-resolution meshes ([W10]). | **absent**: uniform cubed sphere (`h_elem` per panel); stretching is vertical only [C4]. | **experimental**: one nest, offline from ERA5 [N5]. No in-run nesting or two-way nesting; two-way is a roadmap goal [N25]. A Lambert conformal child grid is in an open PR [N19]. | WOOF is the only one with nest trees and an energy planner. Neither Julia stack has in-run multi-scale grids. |
| 3 | Adaptive meshes | **absent** in-run AMR. Nearest equivalents: discrete nest relocation at cycle boundaries ([W4]), pre-declared nests that activate on a trigger ([W13]), and hex re-meshing between cycles ([W12]). | **absent** (no AMR in ClimaAtmos or ClimaCore) [C4]. | **absent** (none in Breeze, NumericalEarth or Oceananigans) [N3]. | None of the three has in-run AMR. WOOF's cycle-to-cycle re-meshing and moving nests are the closest. |
| 4 | Many vertical levels and a stratospheric top | **implemented**: up to 256 levels; explicit eta levels; Rayleigh `w` damping; rigid lid by default ([W14][W15]). *Measured locally, no in-tree receipt*: a real-data 48 h UK run at 294×506×256 with `p_top` 100 Pa (`ztop = 48000.0` in its config) ([W16]). In-tree evidence for a 1 hPa top is column-level only ([W45]), and the docs still say 192 levels is the most run ([W14]). | **implemented** global: 63 levels to 60 km is the common sphere setup, with tanh stretching, SLEVE terrain and Rayleigh/viscous sponges [C8][C9][C10]. ClimaAtmos's own finest sphere configuration is 63 elements to 60 km (`numerics_sphere_he30ze63.yml`). The 130 levels to 70 km setup is a **ClimaCoupler** weather config marked "Under development — not run in CI yet" [C7]. | **implemented** grid stretching. Real-data examples use 50 levels to about 20 km (about 50 hPa) with an `UpperSponge` [N4][N5]. | All three stretch their columns. WOOF has a local, unpublished real-data regional run to 1 hPa. CliMA's global configurations reach 60 km (ClimaAtmos) and 70 km (one ClimaCoupler weather config). Breeze's real-data examples stop near 20 km. |
| 5 | 50 to 100 m runs | **verified** for idealized 100 m LES only (WOOF's own label stays implemented-unverified [W17]): LES closures `km_opt` 2/3 at 100 m match WRF `em_les` to within 0.11–0.32 % on four bulk metrics, idealized ([W19]). 50 m idealized runs ([W18]). **implemented** otherwise: one real-terrain 250 m LES child, passing ([W20]). *Measured here*: a real-terrain 100 m ERA5 run at about 1× realtime, using a PBL scheme rather than LES ([W21]). | **implemented** LES boxes: ISDAC about 107 m node spacing, 167 m vertical, `dt` 0.05 s, 2-minute config [C11]. Schär plane with 1 km elements [C12]. No real-terrain LES. | **implemented** LES, periodic and flat: neutral ABL 96³, BOMEX and RICO at 100 m horizontal / 40 m vertical [N10]. Terrain-following grids exist [N3], but no real-terrain LES example. | Breeze has the most developed LES code and benchmark suite. WOOF is the only one with real-terrain sub-km runs off real data. |
| 6 | Metre-scale terrain and land use (winds over rough terrain) | **implemented**: GLO-30 (about 30 m), SRTM 30 m and 3DEP (about 10 m) terrain; CGLC-MODIS-LCZ 100 m and NLCD 30 m land cover; SoilGrids 250 m ([W22]). Automatic GLO-30 at ≤1 km grid spacing. WRF terrain drag (`gwd_opt` 1/3, `topo_wind`) ([W32]). | **absent** for user data: ETOPO2022 at **60″** (not 30″), always diffusively smoothed [C18]. No DEM or land-cover input keys. | **implemented** readers: ETOPO2022 (default), Copernicus GLO-30 (needs a DestinE token), GLO-90, ESA WorldCover 10 m, GHSL, GloBFP3D, SoilGrids [N14][N15][N16]. The run uses a horizontally uniform terrain-following grid [N3]. No land model consumes land cover in the regional example (slab land) [N5]. | WOOF and NumericalEarth both read 10–30 m DEMs. Only WOOF feeds land cover into a land-surface model and roughness. CliMA uses a smoothed 60″ global DEM. |
| 7 | Boundary and initial condition sources | **implemented**: 24 runnable models: GFS, GDAS, GEFS, HRRR (3 forms), RAP (2 forms), RRFS, ECMWF open data and ENS, AIFS, AIGFS, AIGEFS, ICON global (2 forms)/EU/D2, Met Office global, GEM, ERA5, ERA5 L137, 20CRv3, plus a generic mapped door. Certified: `hrrr`, `gfs`, `era5`; the other 21 are runnable-not-certified ([W24]). | **implemented** for global ICs: ERA5 model levels through ClimaInitialConditions.jl, which ClimaAtmos v0.42.14 now depends on [C6][C24]. No boundary sources (no LBCs). | **implemented**: ERA5 (CDS) and ERA5-Land for the atmosphere; JRA55, ECCO, GLORYS, EN4 and WOA for the ocean. **absent**: GRIB, GFS, IFS, ICON, HRRR [N14]. | WOOF has by far the widest operational-source coverage. NumericalEarth is reanalysis-only. CliMA has initial conditions only. |
| 8 | Ocean coupled to the atmosphere, regional | **absent**: no ocean model, no 1-D mixed layer, SST fixed through the run (`sst_update` pinned 0), sea ice diagnostic only ([W26][W27][W29]). A water-temperature overlay at preparation is the only ocean input ([W28]). | **implemented** global only: ClimaCoupler CMIP mode with an Oceananigans ocean and ClimaSeaIce; the shipped configuration is a 3 h CI case at `h_elem` 8 [C20]. ClimaOcean is ocean-only, driven by a prescribed atmosphere [C19]. No regional coupled configuration. | **experimental**: `EarthSystemModel` and `AtmosphereOceanModel` couple Breeze to prescribed, slab, hydrostatic or non-hydrostatic oceans [N17]. Demonstrated only in a periodic idealized LES [N18]. The regional AtmosphericRivers case uses a prescribed ocean and calls the coupled-nest path "untested" [N8]. A nested ocean is in an open PR [N19]. Oceananigans has regional open ocean boundaries [N20]. | NumericalEarth has the components for a regional coupled model but no regional coupled case yet. CliMA couples globally only. WOOF has no ocean at all. This is WOOF's largest gap against the brief. |
| 9 | Land surface | **implemented**: Noah, RUC and Noah-MP; CLM lake; urban SLUCM/BEP/BEM ([W30]). | **implemented**: ClimaLand through ClimaCoupler (`land_model: "integrated"` in the CMIP configuration [C20]); ClimaLand "can simulate single columns, regional boxes, and global runs" [C21]. | **implemented** `SlabLand` only. Terrarium.jl is "not production-ready" and is not wired into NumericalEarth.jl [N17][N23]. | WOOF and CliMA have full land models. Breeze has a slab. |
| 10 | Clouds: microphysics and convection | **implemented**, with one scheme **verified** (Thompson mp 8, a historical matched run against WRF): Kessler, WSM6, Thompson, Thompson aerosol, Morrison, Milbrandt-Yau, WDM6, NSSL-2, P3 ([W30]). Cumulus: KF, Grell-Freitas, New Tiedtke. 27 of 45 physics options are implemented-unverified ([W31]). | **implemented**: CloudMicrophysics.jl 0M, 1M, 2M (Seifert–Beheng) and P3 [C14]. Prognostic EDMF (convection and turbulence) calibrated with EKI [C15]. | **implemented**: saturation adjustment, Kessler, CloudMicrophysics.jl 1M/2M, native P3 [N12]. **absent**: cumulus or shallow-convection scheme [N5]. | WOOF has the WRF scheme menu. CliMA has a smaller, calibrated, unified set. Breeze relies on resolving convection. |
| 11 | Radiation | **implemented**: RTE+RRTMGP 4/4 (default, supported), legacy RRTMG (verification tier), Dudhia SW ([W30]). The default RTE+RRTMGP path uses one global median RFMIP ozone profile; the CAM ozone climatology applies to legacy RRTMG only ([W46]). | **implemented**: RRTMGP.jl (the RTE+RRTMGP scheme of Pincus et al. 2019 [C25]). *Documented*: 86,400 columns × 63 layers in 0.95 s clear-sky and 1.40 s all-sky on an A100 [C13]. | **implemented**: RRTMGP.jl gray, clear-sky and all-sky through an extension [N12]. | All three use RRTMGP lineage. CliMA's RRTMGP.jl publishes its GPU cost. |
| 12 | Boundary layer and turbulence | **implemented**: YSU, MYJ, MYNN, UW, Shin-Hong, SASE (WOOF-only, unvalidated); LES `km_opt` 2/3; MM5, Eta and MYNN surface layers ([W30]). | **implemented**: prognostic EDMF; Smagorinsky–Lilly and AMD for LES; Monin–Obukhov through SurfaceFluxes.jl; orographic and non-orographic gravity-wave drag [C15][C16][C17]. WOOF's `gwd_opt` 1/3 are orographic only, and WRF has no non-orographic scheme ([W32]). | **implemented**: a 1.5-order TKE closure with constant stability functions (calibration PR open) [N13]; Smagorinsky–Lilly, dynamic Smagorinsky, AMD [N12]. The regional example runs with no PBL scheme [N5]. | WOOF has the operational PBL menu. CliMA has a calibrated EDMF. Breeze's mesoscale PBL is immature. |
| 13 | Simulation speed (published numbers; not comparable) | *Measured locally, no in-tree receipt*: UK 294×506×256 mass points at 2.45 km, full physics, one card (the workstation's RTX A6000; the record does not name it), about 3.2× realtime ([W16]). *Documented*: 3 km, 551×551×50 staggered (14.82 M mass cells), on an RTX 5090 at 0.871 s per simulated minute, about 69× realtime, measured with the predecessor engine's 1.5.0 wheel ([W33]). HRRR 1797×1057×50 on 4 × RTX PRO 6000 at 130 s per forecast hour ([W34]). | *Documented*: 0.20 SYPD at 6 km global on 256 H100 (idealized moist baroclinic wave with 1M microphysics); 223 ms per step weak-scaled over 1→256 Derecho A100 [C2]. | *Documented*: 12 km 540×315×50, 72 h in 3.77 h on one A100-40GB (about 19× realtime) [N8]. L4 Float32 512²×256: anelastic 0.518 s per step, compressible split-explicit 2.38 s per step [N11]. | Not comparable as published. See the regional, LES and idealized benchmark pages. |
| 14 | Parallelism and precision | **implemented**: one domain split across cards in one host (`[devices]`), byte-identical across 2, 4 and 8 cards ([W34][W35]). No multi-node. FP32 ([W44]). | **implemented**: MPI multi-GPU (weak scaling above 92 % to 256 Derecho A100) [C2]. Float32 by default [C3]. | **implemented**: multi-GPU through Oceananigans' distributed architecture (`NCCLDistributed` in AtmosphericRivers); 2 A100s gave 1.50× on a 12 km nest [N8]. Float32 in benchmarks [N11]. | CliMA is the most proven at scale. WOOF is limited to one host. |
| 15 | Wind-energy outputs | **implemented**: hub-height wind, gusts (`WSPD10MAX`, UPP `GUST`), icing, ramps, turbine and farm power with wakes, line rating ([W36]). Icechunk live export ([W37]). | **absent**: no hub-height or power diagnostics. `uas`/`vas` are the lowest-cell wind, not a 10 m diagnostic [C23]. | **absent**: no turbine or actuator code [N12]. | Only WOOF produces energy products. |
| 16 | Output formats | **implemented**: wrfout NetCDF, Icechunk, Zarr (ML export) ([W37][W38][W39]). | **implemented**: NetCDF and HDF5 through ClimaDiagnostics [C22]. | **implemented**: NetCDF and JLD2 [N12]. Oceananigans also has a `ZarrWriter`, not re-exported by Breeze [N20]. | WOOF writes WRF-compatible files that downstream tools already read. |
| 17 | Evidence of accuracy | **verified** (code verification only) against WRF Fortran per scheme and in matched runs; the t=0 full-state comparison FAILs ([W40]). Observation evidence is "limited to specific runs and variables" ([W40]). *Measured here*: UK 48 h station scores ([W41]). | **verified** for the dynamical core on standard test cases (dry and moist baroclinic waves, topographic gravity waves; paper §6) [C2]. EDMF is calibrated against LES profiles [C15]. No regional weather verification exists (no regional capability). | **implemented**: the LES cases follow published intercomparison setups (BOMEX after Siebesma 2003) [N10]. No published quantitative comparison was found by this page; [les-benchmark.md](les-benchmark.md) now compares Breeze and woof on the FastEddy dry CBL. No regional verification. | None of the three has a broad regional validation record. WOOF has the most regional evidence, and it is still thin. |

## Notes by criterion

### 1. Real-data limited area

- **WOOF.**
  - The regional model is Production in the sense of a stable interface,
    not a claim about forecast accuracy ([W1]).
  - The t=0 full-state comparison against WRF fails on all four domains of
    the reference case ([W40]).
- **Breeze/NumericalEarth.**
  - The regional route is the `nested_atmosphere_model` function and one
    example [N4][N5].
  - Defaults: split-explicit compressible core, `UpperSponge` lid, Davies
    relaxation off.
  - The example balances the initial state with an FV3 `na_init`-style
    adiabatic forward/backward excursion [N6]. It is not a classical digital
    filter; the digital-filter PR (#834) is still open [N7].
  - Breeze open boundaries are Oceananigans `OpenBoundaryCondition` /
    `PerturbationAdvection` [N12].
  - The earlier "open boundaries with mass-flux correction" PR (#287) was
    closed without merging.
  - The merged #747 adds per-substep relaxation of ρ′ and (ρθ)′ at the
    outermost cell [N7]. That relaxation is what #825 reports as kicking the
    boundary-adjacent momentum.
  - The AtmosphericRivers 72 h run says only the interior "(> 25 cells from
    the frame) stays physical", and its analysis masks that rind [N8]. A
    25-cell rind at 12 km is a 300 km band at every edge.
- **CliMA.** See [What "regional" means](#what-regional-means-in-each-stack).
  CliMA has no regional atmosphere at this snapshot.

### 2. Multi-scale grids

- **WOOF: the domain count.**
  - "21 domains" is a constant for stock-WRF export (`MAX_DOMAINS = 21`).
  - `ANALYSIS_MAX_DOMAINS = 256` only caps the namelist report's own
    per-domain expansion; the source calls it "Nothing physical and no engine
    limit" ([W5]). No engine domain cap is documented.
  - The energy `wrf-nests` planner caps a plan at 20 children plus the root
    ([W6]).
- **WOOF: offline children.** `wrf-tiles` emits any number of offline child
  tiles, each re-run by `woof downscale`, with no domain-count ceiling
  ([W7]). `woof downscale --child-levels` gives an offline child its own
  deeper level ladder ([W8]).
- **WOOF: one-way vertical nesting.** A live nest may name its own
  `eta_levels` (WRF `vert_refine_method = 2`); without it, sub-km children
  share their parent's level count ([W9]).
- **WOOF: documentation conflict.** The energy topology table says
  `wrf-nests` is "Two-way inside one run" (`docs/energy-forecasts.md:435`),
  but the plan preamble says "sibling nests of one run (feedback = 0)"
  (`woof/energy/plan_wrf_nests.py:751`). The configuration default is
  `feedback = 0` (`woof/experiment.py:1361`). The code is one-way. The doc row
  is wrong.
- **WOOF: hex meshes.**
  - The hex model is a port of the MPAS-A v8.4.1 dynamical core and runs on
    CUDA 13 only ([W10]).
  - The finest graded hex mesh with a forecast measurement is 0.75 km.
  - A 100 m corridor mesh has been generated, but no 100 to 400 m hex
    forecast is measured.
  - Sub-km hex steps run on an "experimental-unanchored" time step ([W11]).

### 4. Vertical levels and the stratosphere

- **The UK run** is the deepest real-data WOOF run on record. It is measured
  locally and has no in-tree receipt ([W16]):
  - 294×506 at 2.45 km, 256 levels, `p_top` 100 Pa (`ztop = 48000.0` in the
    config).
  - 48 h, ICON global (`icon-global-ml`) boundaries.
  - Thompson mp 28, MYNN, RUC, legacy RRTMG.
  - Upper damping: `damp_opt = 3`, `zdamp` 5 km.
- **WOOF's docs are behind this run.**
  - `docs/public/LES.md:382-384` says no full-physics real-data column above
    192 levels has been integrated.
  - `docs/manual/02-dynamics-grids-nesting.md:88` still says 128.
  - The run record postdates both. It sits outside the repository (see
    [W16]), so the docs are stale but no published receipt replaces them yet.
  - The only in-tree 1 hPa evidence is column-level: "256 levels to a 1 hPa
    top run through the shipped launchers and come back finite" ([W45]). The
    UK configurations committed earlier used 115 levels to 50 hPa (commit
    `4adb152b`, `uk-1530m-115l-15min-rad.toml`: `nz = 115`,
    `p_top = 5000.0`; the files have since left the tree).
- **CliMA** has run deep, stretched global columns with sponges. The 63-level
  / 60 km setup is the common one [C8].
- **Breeze**'s real-data example uses 50 levels to about 20 km [N5]. Nothing
  in the code limits the column, but no deeper real-data case is published.

### 5. 50 to 100 m

- **WOOF: LES maturity.** LES "enters WOOF's existing maturity ladder as
  implemented-unverified" ([W17]). The 100 m agreement with WRF `em_les` is
  "measurement, not a maturity promotion" ([W19]).
- **WOOF: real terrain.** The real-terrain 250 m LES child is one case. Per
  the doc it "demonstrates the closure works there: it does not verify it
  there", and "No idealized terrain LES exists at all" ([W20]).
- **WOOF: the 100 m Greenland run** ([W21]):
  - 200×160×115 at `dt` 0.5 s, ERA5, `p_top` 50 hPa.
  - It ran MYNN PBL with `km_opt = 4`. It is a gray-zone mesoscale run, not
    LES.
  - *Measured here*: 1 h of model time took 62.6 min of wall clock, about
    0.96× realtime.
- **CliMA.** The ISDAC box is 3.2 km wide with 10 spectral elements per side
  (cubic, so about 107 m between nodes) and 15 uniform vertical elements over
  2.5 km. It runs 2 minutes from a restart on a cluster path [C11]. The Schär
  case is 1 km elements, about 333 m between nodes [C12].
- **Breeze.** BOMEX is 64×64×75 over 6.4 × 6.4 × 3 km. That is 100 m by
  **40 m**, not 25 m as the brief had it [N10].

### 6. Terrain and land use

- **CliMA's DEM.**
  - The atmosphere reads `ETOPO_2022_v1_60s_N90W180_surface.nc`, which is
    60″.
  - A 30″ artifact path exists, but nothing in `src` calls it [C18].
  - Smoothing is always applied. Its strength is set by
    `topography_damping_factor`: 32 iterations at the default of 5, about 3
    at the weather configs' 1.15 [C18][C7].
- **WOOF.** GLO-30 is "marginal" at 50 m, by WOOF's own energy page ([W23]).

### 8. Ocean

This is the criterion the brief stresses. The detail will be on
[ocean-coupling.md](ocean-coupling.md) (planned).

**WOOF.**

- WOOF has no ocean model of any kind. `sf_ocean_physics` is recorded as
  "unimplemented" ([W27]), and `sst_update`, `sst_skin` and `tmn_update` are
  pinned to 0 ([W26]).
- The water-temperature overlay replaces ERA5 SST over water before
  interpolation ([W28]). With `sst_update = 0`, the model SST stays at its
  initial value through the run.

**CliMA.**

- ClimaOcean is "ocean-only and coupled ocean + sea-ice simulations driven by
  prescribed atmospheres" [C19].
- The atmosphere-ocean coupling is ClimaCoupler's global CMIP mode
  (`cmip_oceananigans_climaseaice`, `h_elem` 8) [C20].

**NumericalEarth.**

- `EarthSystemModel{radiation, atmosphere, land, sea_ice, ocean}` and
  `AtmosphereOceanModel` exist [N17].
- The four ocean types are exercised together only in
  `breeze_over_four_oceans.jl`, a periodic 64×64 idealized LES [N18].
- In AtmosphericRivers, `AtmosphereOceanModel` around a nested atmosphere is
  "an untested path" [N8].
- The open PRs that would make a regional coupled case possible are on the
  NumericalEarth side:
  - #676, a nested ocean;
  - #753, Breeze to `PrescribedOcean` fixes;
  - #664, a fractional land-sea surface [N19].
- Oceananigans itself has regional open boundaries [N20]:
  - Flather-type `GravityWaveRadiation`;
  - `PerturbationAdvection`;
  - tidal forcing.

### 13. Speed (the published numbers and their conditions)

These are recorded so the benchmark pages can reproduce them. They are not a
ranking.

**WOOF.**

- UK run, *measured locally, no in-tree receipt*: 129,000 s of model time (12:10 on 13 October to
  00:00 on 15 October), between the first and last history commits of the
  restarted segment. That took 40,769 s of wall clock, which is 3.16×
  realtime.
  - Card: one card. The run record does not name it; the workstation's
    card is an RTX A6000.
  - Grid: 38.1 M mass cells.
  - Physics: full, with 10-minute history output and adaptive `dt` (mean
    11.5 s over 15,073 steps).
  - Source: the `output_committed` events in the run's `events.jsonl`
    ([W16]).
- Reference envelope: receipt dated 2026-08-03, measured with the
  predecessor engine's published 1.5.0 wheel ([W33]). WOOF 1.0 continues that
  engine from its release 2.8.0, so 1.5.0 is older than WOOF 1.0, not newer
  ([W1], `README.md:134-136`).
- Multi-card: CHANGELOG 1.0.3 ([W34]).

**CliMA.**

- 0.20 SYPD is the paper's §7.3 figure: dynamical core plus 1-moment
  microphysics, 6 km, 43 levels, idealized moist baroclinic wave, 256 Google
  Cloud H100s.
- 223 ms per step is the weak-scaling figure (about 103 km on 1 GPU to 6 km
  on 256) on NCAR Derecho
  A100s [C2].
- No real-data, full-physics regional timing exists to compare.

**NumericalEarth.**

- AtmosphericRivers, `main`: 8.5 M cells, 47,600 iterations in 3.77 h on one
  A100-40GB, about 19× realtime. Strong scaling from 1 to 2 GPUs gave 1.50×,
  measured on 6 simulated hours [N8].
- A newer branch reports a 324×162×50 nest on one H100 at about 380×
  realtime (26 ms per step, `dt` 10 s) [N9].
- BreezeBenchmarks figures are per-commit JSONs from an NVIDIA L4 [N11].

### 17. Accuracy

None of the three stacks publishes a multi-case regional forecast
verification. WOOF's evidence is the strongest of the three, and it is thin.

**WOOF.** It has:

- per-scheme WRF Fortran comparisons;
- historical matched runs;
- a handful of observation comparisons (ASOS snow and soil, a fog suite,
  MRMS model top) ([W40]).

The UK 48 h run was scored against 94 METAR stations (21,770 pairs), *measured
here* ([W41]). This is one case, one season, on a local run record:

| Variable | Bias | RMSE |
| --- | --- | --- |
| 2 m temperature | −0.80 K | 1.45 K |
| 2 m dewpoint | +0.21 K | 1.39 K |
| 10 m wind speed | −0.22 m/s | 1.81 m/s |

**CliMA.** The dynamical-core paper checks standard test cases (§6) and
reports scaling [C2]. EDMF calibration uses EnsembleKalmanProcesses against
LES profiles from forced single-column cases [C15].

**Breeze.** The LES cases are set up from published intercomparisons [N10].
Their quantitative scoring is the job of [les-benchmark.md](les-benchmark.md)
(planned).

## Corrections to the starting brief

Verifying the brief against primary sources changed these points. They are
already reflected in the matrix.

1. **CliMA topography.** It is ETOPO2022 at **60″**, not 30″ [C18].
2. **CliMA resolution claims.**
   - The 6 km global run is a scaling benchmark of an idealized moist
     baroclinic wave with 43 levels. No 6 km configuration is in the
     repository [C2].
   - The weather configurations are about 100 km [C7].
3. **CliMA initial conditions.** ClimaInitialConditions.jl v0.1.0 is now a
   dependency of released ClimaAtmos [C24]. It is no longer separate
   tooling.
4. **Breeze's tagline.** "Limited area LES-to-mesoscale…" is the GitHub repo
   description. The README tagline is "Fast and friendly Julia software for
   atmospheric fluid dynamics on CPUs and GPUs" [N1].
5. **Breeze open-boundary PRs.**
   - #287 (mass-flux correction) was closed without merging.
   - #747 (merged) is a per-substep density relaxation, not a mass-flux
     correction [N7].
6. **Breeze issue #897** is closed, not open [N7].
7. **Breeze initialization balance.** It is FV3-style `na_init` adiabatic
   balancing. The digital-filter PR is open [N6][N7].
8. **BOMEX vertical spacing.** It is 40 m, not 25 m [N10].
9. **Zarr output.** Oceananigans has a `ZarrWriter`, so "Zarr input only" no
   longer holds for the stack. Breeze does not re-export it [N20].
10. **Oceananigans ocean speed claims.**
    - The 8 SYPD figure is from the NumericalEarth README. Its configuration
      is 1440×560 with **10 levels**, 3000 m deep [N21].
    - The arXiv paper says "global" at 488 m, 15 simulated days per day, on
      768 A100s [N22].
11. **WOOF domain count.** 21 is the stock-WRF export cap, not an engine
    limit ([W5]).
12. **WOOF hex at 100 to 400 m.** It is generated meshes plus an
    experimental time-step lane. The finest hex forecast measured is 0.75 km
    ([W11]).
13. **WOOF Greenland 100 m.** It is a MYNN gray-zone run, not LES ([W21]).
14. **CliMA vertical extent.** ClimaAtmos's own finest sphere configuration
    is 63 elements to 60 km. The 130-level, 70 km setup belongs to a
    ClimaCoupler weather configuration that is not run in CI [C7][C8].
15. **CliMA weak scaling.** It was measured on NCAR Derecho A100s (about
    103 km on 1 GPU to 6 km on 256). The 0.20 SYPD figure is on 256 H100s
    [C2].
16. **WOOF 1 hPa top.** No in-tree receipt shows a real-data run at a 1 hPa
    top. The in-tree evidence is column-level ([W45]). The 256-level UK run
    is measured locally only ([W16]).
17. **WOOF ozone.** The default RTE+RRTMGP path uses one global median RFMIP
    ozone profile ([W46]).
18. **WOOF multi-GPU.** The README's "runs on one NVIDIA GPU" ([W2]) predates
    the multi-card `[devices]` split ([W35]).

## Sources

### WOOF (fork `main`, `b4a3783a`)

Paths are relative to the repository root. Line numbers are at the snapshot
commit.

- **W1** `README.md:9-13`: model table; Regional is Production, Hex and
  Global are Preview. `README.md:20-22`: Production "is not a statement about
  forecast accuracy".
- **W2** `README.md:5-6`: "runs on one NVIDIA GPU".
- **W3** `docs/manual/02-dynamics-grids-nesting.md:110-118`: "Static one-way
  nesting is the supported default"; two-way "ships as an experimental path".
  Also `docs/manual/09-limits-and-roadmap.md:39-40`.
- **W4** `docs/manual/02-dynamics-grids-nesting.md:140-142`: moving nests are
  "a sequence of static nests joined by a re-grid at cycle boundaries".
- **W5** `woof/namelist_compat.py:40-58`: `MAX_DOMAINS = 21` "belongs to the
  UNCHANGED WRF EXECUTABLE"; `ANALYSIS_MAX_DOMAINS = 256`.
- **W6** `docs/energy-forecasts.md:429-436`: planner topology table
  (`wrf-nests`, `wrf-tiles`, `hex-swath`). `woof/energy/plan_wrf_nests.py:751`:
  "(feedback = 0)" in the plan preamble. `woof/experiment.py:1361`: the
  `feedback` default of 0.
- **W7** `docs/energy-forecasts.md:447-456`: `wrf-tiles`, "no domain-count
  ceiling".
- **W8** `docs/public/DOWNSCALE.md:3-4`: offline `ndown` equivalent.
  `docs/public/DOWNSCALE.md:1018-1028`: `--child-levels`.
- **W9** `docs/manual/02-dynamics-grids-nesting.md:73-77`: no vertical
  nesting in a live tree.
- **W10** `README.md:130-132`: MPAS-A v8.4.1 port. `components/hex/README.md:155`:
  CUDA 13 only. `components/hex/README.md:495-502`: limited area on a culled
  mesh. `docs/mpas-mesh-generation.md:229`: regional window "(experimental)".
- **W11** `docs/energy-forecasts.md:520-521`: finest measured graded mesh
  0.75 km. `docs/energy-forecasts.md:523-547`: "experimental-unanchored"
  sub-km time steps. `docs/mpas-mesh-generation.md:312-320`: a 100 m corridor
  mesh generated.
- **W12** `docs/energy-forecasts.md:643-647`: hex adapt plans the next cycle's
  mesh from one cycle's history.
- **W13** `docs/nest-spawn-at-trigger.md:10-24`: dormant nests,
  "ACTIVATION, not allocation".
- **W14** `docs/public/LES.md:331`: "`nz <= 192` has been RUN and receipted;
  the solver admits 256". `docs/public/LES.md:382-384`: no full-physics
  real-data column above 192 levels. `docs/manual/02-dynamics-grids-nesting.md:88`:
  the older 128 statement.
- **W15** `docs/public/CONFIGURATION.md:450-453`: `damp_opt`, `zdamp`,
  `dampcoef`, `w_damping`. `docs/public/CONFIGURATION.md:463`: `top_lid`
  defaults to true.
- **W16** Local run record, not in the repository. It was produced by
  `woof go` on 2026-10-13 and 14 and is kept read-only under
  `/data/uk-l256/`:
  - config: `uk-2450m-256l-1hpa-10min-hrrr.toml`, which uses source
    `icon-global-ml` despite its file name;
  - progress: `run-20261013-192214Z_i202610130000Z/run-progress.json`
    (status `complete`, `outer_step` 15073, `model_elapsed_seconds` 172800);
  - timing: `run-20261013-192214Z_i202610130000Z/events.jsonl`.
- **W17** `docs/public/LES.md:19`: LES is implemented-unverified.
- **W18** `docs/public/LES.md:105-110`: idealized 50 m CBL runs.
- **W19** `docs/public/LES.md:152-164`: 100 m comparison with WRF `em_les`.
  `docs/public/LES.md:509-518`: "measurement, not a maturity promotion".
- **W20** `docs/public/LES.md:207-212`: 250 m `km_opt = 3` child in a HRRR
  tree. `docs/public/LES.md:527-531`: one case; no idealized terrain LES.
- **W21** Local run record, not in the repository: the
  `greenland-100m-115l-15min-rad-run/` directory inside the main working
  checkout, untracked.
  - `run-progress.json`: started 2026-10-02T15:44:15Z, updated 16:46:53Z,
    3600 s model time, 7200 steps.
  - The captured config gives nx 200, ny 160, nz 115, dx 100 m,
    `bl_pbl_physics = 5`, `km_opt = 4`, source `era5`.
  - The log does not name the card. The workstation's card is an RTX A6000.
- **W22** `docs/public/HIGHRES-TERRAIN.md:12-21`: automatic GLO-30 at ≤1 km.
  `docs/public/HIGHRES-TERRAIN.md:67-72`: source table.
- **W23** `docs/energy-forecasts.md:723`: GLO-30 marginal at 50 m.
- **W24** `woof/source_adapters.py:1-8`: runnable and certified definitions.
  `docs/manual/05-initialization.md:44-49`: certified rows and registered rows.
  `docs/public/SOURCES.md:21-43`: source table.
  - `docs/manual/09-limits-and-roadmap.md:14-16` says "13 runnable model
    rows". At the snapshot the registry has 21 runnable-not-certified model
    rows besides the 3 certified ones and `mapped`, so that line is
    stale.
- **W26** `woof/namelist_import.py:980-986`: `sst_update`, `sst_skin` and
  `tmn_update` pinned to 0, "not implemented".
- **W27** `woof/io/wrf_output_schema.py:850-851`: `SF_OCEAN_PHYSICS`
  "unimplemented".
- **W28** `docs/water-temperature-overlay.md:1-6`: the overlay.
- **W29** `docs/manual/01-arwen-and-wrf.md:137`: Noah sea-ice thermodynamics
  not ported.
- **W30** `docs/public/PHYSICS.md:16-37`: maturity vocabulary.
  - `docs/public/PHYSICS.md:251-263`: microphysics.
  - `docs/public/PHYSICS.md:1089-1094`: PBL.
  - `docs/public/PHYSICS.md:1249-1258`: land surface and lake.
  - `docs/public/PHYSICS.md:1344-1352`: radiation.
  - `docs/public/PHYSICS.md:1495-1503`: cumulus.
  - `docs/public/PHYSICS.md:185-187`: urban.
- **W31** `docs/manual/09-limits-and-roadmap.md:52-57`: 27 of 45 options
  implemented-unverified.
- **W32** `docs/public/TERRAIN-DRAG.md:1-15`: `topo_wind`, `gwd_opt`.
- **W33** `docs/manual/08-operational-envelope.md:4-9`: the receipt is dated
  2026-08-03 and used the predecessor engine's 1.5.0 wheel.
  `docs/manual/08-operational-envelope.md:21-24`:
  RTX 5090 and RTX 5070 Ti throughput.
- **W34** `CHANGELOG.md:746-751`: `[devices]` timings and byte identity
  across 2, 4 and 8 cards.
- **W35** `docs/public/MULTIGPU.md:1-6`: `[devices]` slabs and halo exchange.
- **W36** `docs/energy-forecasts.md:359-364`: hub-height and gust fields.
  `docs/energy-forecasts.md:842-854`: products and their caveat.
- **W37** `docs/public/PROGRESS.md:297-319`: live Icechunk export.
- **W38** `docs/public/OUTPUT-VARIABLES.md:1-20`: wrfout variables.
- **W39** `docs/ml-export.md:1-8`: Zarr ML export.
- **W40** `README.md:106-128`: code verification against validation; the
  limits of the observation evidence. `docs/manual/09-limits-and-roadmap.md:49-51`:
  t=0 FAIL.
- **W41** Local verification record, not in the repository:
  `/data/uk-l256/verify/summary.md` (289 frames, 94 stations, 21,770 pairs).
- **W44** `docs/manual/02-dynamics-grids-nesting.md:8`: FP32 on CUDA.
- **W45** `docs/kernel_local_memory_bounds.md:736-738`: "256 levels to a 1 hPa
  top run through the shipped launchers and come back finite" (column-level
  kernel evidence, not a forecast).
- **W46** `woof/core/rrtmgp.py:3142-3145`: "ozone is interpolated from the
  median RFMIP climatological profile".

### CliMA

Prefixes used below:

- `A` = `https://github.com/CliMA/ClimaAtmos.jl/blob/dda668d3d49296beb7543aace1bde482b96ee294`
- `CC` = `https://github.com/CliMA/ClimaCoupler.jl/blob/e3355c8ac9dd0f4787cbdafa93ed737a61a460a1`

| ID | Source | Supports |
| --- | --- | --- |
| C1 | [A/README.md#L7-L9][C1] | "a GPU-capable global atmosphere model" |
| C2 | Yatunin et al. (2026), "The Climate Modeling Alliance Atmosphere Dynamical Core: Concepts, Numerics, and Scaling", JAMES 18(3), e2025MS005014, [doi:10.1029/2025MS005014][C2doi] ([open PDF][C2]) | SEM horizontal, staggered finite-difference vertical, HEVI; §6 test cases; §7.1 weak scaling (above 92 %, 223 ms per step, A100); §7.3 0.20 SYPD at 6 km on 256 H100 |
| C3 | [A/config/default_configs/default_config.yml#L141-L143][C3] | `FLOAT_TYPE` Float32; ARS343 IMEX default at L71-L73 |
| C4 | [A/src/simulation/grids.jl#L7][C4] | `SphereGrid, ColumnGrid, BoxGrid, PlaneGrid` |
| C5 | [A/src/config/type_getters.jl#L535-L546][C5] | periodic box and plane on the YAML path |
| C6 | [A/src/setups/WeatherModel.jl#L1-L33][C6] | "ERA5-derived initial condition" |
| C7 | [CC/config/weather_configs/weatherbench_progedmf_1m.yml#L1-L23][C7] | "Under development — not run in CI yet"; `h_elem` 30, `z_elem` 130, `z_max` 70000 |
| C8 | [A/config/common_configs/numerics_sphere_he16ze63.yml#L1-L11][C8] | 63 levels, 60 km, sponges on |
| C9 | [A/config/default_configs/default_config.yml#L40-L48][C9] | SLEVE terrain warp |
| C10 | [A/src/types.jl#L829-L900][C10] | `ViscousSponge`, `RayleighSponge` |
| C11 | [A/config/model_configs/les_isdac_box.yml#L12-L23][C11] | ISDAC LES box, `dt` 0.05 s |
| C12 | [A/config/model_configs/plane_schar_mountain_float32_test.yml#L1-L11][C12] | Schär plane |
| C13 | [RRTMGP.jl docs/src/howto/gpu.md#L120-L139][C13] | 86,400 columns × 63 layers on an A100 |
| C14 | [CloudMicrophysics.jl README#L41-L46][C14] | 0M, 1M, 2M, P3 |
| C15 | [A/config/default_configs/default_config.yml#L469-L471][C15], [A/calibration/experiments/gcm_driven_scm/run_calibration.jl#L3][C15b] and [its README#L3][C15c] | prognostic EDMF; EKI calibration "to LES profiles" |
| C16 | [A/docs/src/les_sgs.md#L5-L10][C16] | Smagorinsky–Lilly, AMD |
| C17 | [A/config/default_configs/default_config.yml#L237-L239][C17] and [#L346-L351][C17b] | Monin–Obukhov surface; gravity-wave drag |
| C18 | [A/src/utils/AtmosArtifacts.jl#L108-L133][C18], [A/src/simulation/grids.jl#L313-L337][C18b], [A/docs/src/topography.md#L55-L71][C18c] | 60″ ETOPO2022, always smoothed; 30″ path unused |
| C19 | [ClimaOcean.jl README#L8-L16][C19] | "driven by prescribed atmospheres"; moving to NumericalEarth.jl |
| C20 | [CC/README.md#L10-L15][C20] and [CC/config/ci_configs/cmip_oceananigans_climaseaice.yml][C20b] | global CMIP/AMIP/slab; Oceananigans ocean, ClimaSeaIce, integrated land, `h_elem` 8, `t_end` 3 hours |
| C21 | [ClimaLand.jl README#L28][C21] | "single columns, regional boxes, and global runs" |
| C22 | [ClimaDiagnostics.jl netcdf_writer.jl#L22][C22] and [hdf5_writer.jl#L18][C22b] | NetCDF and HDF5 writers |
| C23 | [A/src/diagnostics/core_diagnostics.jl#L469-L490][C23] | `uas`/`vas` at the bottom cell centre |
| C25 | Pincus, Mlawer and Delamere (2019), "Balancing accuracy, efficiency, and flexibility in radiation calculations for dynamical models", JAMES 11, [doi:10.1029/2019MS001621][C25] | the RTE+RRTMGP scheme that RRTMGP.jl implements |
| C24 | [ClimaInitialConditions.jl README#L7][C24] and [A/Project.toml][C24b] | ERA5 from CDS; dependency of ClimaAtmos |

### NumericalEarth

Prefixes used below:

- `B` = `https://github.com/NumericalEarth/Breeze.jl/blob/6eabb8937670f61ad3bfd0ca2ad400a747681646`
- `NE` = `https://github.com/NumericalEarth/NumericalEarth.jl/blob/66112eac639e6163639b89f3fc1c684b146002d2`

| ID | Source | Supports |
| --- | --- | --- |
| N1 | [B/README.md#L8][N1] and the [GitHub repository description][N1b] | README tagline; "Limited area LES-to-mesoscale…" |
| N2 | [B/src/TimeSteppers/acoustic_runge_kutta_3.jl#L28-L64][N2] | Wicker–Skamarock RK3 with acoustic substepping; anelastic core at [B/src/AnelasticEquations/anelastic_dynamics.jl#L5][N2b] |
| N3 | [B/docs/src/terrain_following_coordinates.md#L3-L9][N3] | "The horizontal grid stays uniform — a design decision"; Gal-Chen–Somerville at L190, SLEVE at L220-L256 |
| N4 | [NE/ext/NumericalEarthBreezeExt/breeze_nested_atmosphere.jl#L111-L240][N4] | `nested_atmosphere_model`; `UpperSponge` default; `relaxation_rate = nothing` |
| N5 | [NE/examples/breeze_downscaling_era5.jl#L9-L91][N5] | ERA5 parent, single nest, 12 km, 150×136×50, 50 levels to about 20 km, no PBL or cumulus |
| N6 | [B/src/AtmosphereModels/adiabatic_balance.jl#L1-L34][N6] | `na_init`-style adiabatic balance |
| N7 | Breeze [#287][N7a] (closed, not merged), [#747][N7b] (merged 2026-06-01), [#825][N7c] (open), [#793][N7d] (open), [#897][N7e] (closed 2026-09-08), [#834][N7f] (open) | open-boundary history and defects |
| N8 | [glwagner/AtmosphericRivers.jl README#L56-L107][N8] | 540×315×50, 72 h in 3.77 h on an A100-40GB; 1.50× on 2 GPUs; coupled nest "untested" |
| N9 | [AtmosphericRivers.jl README on branch `glw/precip-sensitivity-adjoint`, L57-L69][N9] | 324×162×50 on an H100 at about 380× realtime |
| N10 | [B/examples/bomex.jl#L42-L50][N10], [rico.jl#L45-L53][N10b], [neutral_atmospheric_boundary_layer.jl#L38-L44][N10c] | periodic flat LES grids; Siebesma 2003 setup |
| N11 | [BreezeBenchmarks benchmark_results.json#L175-L194 and #L605-L624][N11] ([dashboard][N11b]) | L4, Float32: 0.518 s per step (129.6 Mpts/s); 2.38 s per step (28.2 Mpts/s) |
| N12 | [B/src/Breeze.jl#L115-L233][N12] | microphysics, closures and output writer exports |
| N13 | [B/src/TurbulenceClosures/tke_based_turbulence_closure.jl#L5-L10][N13] | constant stability functions |
| N14 | [NE/src/DataWrangling/DataWrangling.jl#L409-L430][N14] | dataset list (no GRIB, GFS, IFS, ICON or HRRR) |
| N15 | [NE/src/DataWrangling/CopernicusDEM/CopernicusDEM.jl#L25-L44][N15] and [NE/src/DataWrangling/WorldCover/WorldCover.jl][N15b] | GLO-30 needs `DESTINE_ACCESS_TOKEN`; WorldCover 10 m |
| N16 | [NE/src/Bathymetry/regrid_bathymetry.jl#L172-L188][N16] | ETOPO2022 default |
| N17 | [NE/src/EarthSystemModels/earth_system_model.jl#L6-L15][N17] | `EarthSystemModel`; `AtmosphereOceanModel` at L452-L459 |
| N18 | [NE/examples/breeze_over_four_oceans.jl#L1-L18][N18] | four oceans, periodic idealized LES |
| N19 | NumericalEarth.jl open PRs [#676][N19a] (nested ocean), [#753][N19b], [#664][N19c], [#695][N19d] (Lambert conformal child) | regional coupling work in progress |
| N20 | [Oceananigans gravity_wave_schemes.jl#L1-L56][N20] and [zarr_writer.jl#L8][N20b] | Flather-type open boundaries; `ZarrWriter` |
| N21 | [NE/README.md#L86-L110][N21] | 8 SYPD on an H100, 1440×560×10 |
| N22 | [arXiv:2309.06662][N22] | Oceananigans at 488 m global, 15 SDPD on 768 A100 |
| N23 | [Terrarium.jl README#L22][N23] | "not production-ready" |
| N25 | [B/docs/src/index.md#L72][N25] | two-way nesting as a roadmap goal |

[W1]: ../../../README.md
[W2]: ../../../README.md
[W3]: ../../manual/02-dynamics-grids-nesting.md
[W4]: ../../manual/02-dynamics-grids-nesting.md
[W5]: ../../../woof/namelist_compat.py
[W6]: ../../energy-forecasts.md
[W7]: ../../energy-forecasts.md
[W8]: ../../public/DOWNSCALE.md
[W9]: ../../manual/02-dynamics-grids-nesting.md
[W10]: ../../../components/hex/README.md
[W11]: ../../energy-forecasts.md
[W12]: ../../energy-forecasts.md
[W13]: ../../nest-spawn-at-trigger.md
[W14]: ../../public/LES.md
[W15]: ../../public/CONFIGURATION.md
[W16]: #sources
[W17]: ../../public/LES.md
[W18]: ../../public/LES.md
[W19]: ../../public/LES.md
[W20]: ../../public/LES.md
[W21]: #sources
[W22]: ../../public/HIGHRES-TERRAIN.md
[W23]: ../../energy-forecasts.md
[W24]: ../../../woof/source_adapters.py
[W26]: ../../../woof/namelist_import.py
[W27]: ../../../woof/io/wrf_output_schema.py
[W28]: ../../water-temperature-overlay.md
[W29]: ../../manual/01-arwen-and-wrf.md
[W30]: ../../public/PHYSICS.md
[W31]: ../../manual/09-limits-and-roadmap.md
[W32]: ../../public/TERRAIN-DRAG.md
[W33]: ../../manual/08-operational-envelope.md
[W34]: ../../../CHANGELOG.md
[W35]: ../../public/MULTIGPU.md
[W36]: ../../energy-forecasts.md
[W37]: ../../public/PROGRESS.md
[W38]: ../../public/OUTPUT-VARIABLES.md
[W39]: ../../ml-export.md
[W40]: ../../../README.md
[W41]: #sources
[W44]: ../../manual/02-dynamics-grids-nesting.md
[W45]: ../../kernel_local_memory_bounds.md
[W46]: ../../../woof/core/rrtmgp.py

[C1]: https://github.com/CliMA/ClimaAtmos.jl/blob/dda668d3d49296beb7543aace1bde482b96ee294/README.md#L7-L9
[C2]: https://clima.caltech.edu/wp-content/uploads/2026/03/J-Adv-Model-Earth-Syst-2026-Yatunin-The-Climate-Modeling-Alliance-Atmosphere-Dynamical-Core.pdf
[C2doi]: https://doi.org/10.1029/2025MS005014
[C3]: https://github.com/CliMA/ClimaAtmos.jl/blob/dda668d3d49296beb7543aace1bde482b96ee294/config/default_configs/default_config.yml#L141-L143
[C4]: https://github.com/CliMA/ClimaAtmos.jl/blob/dda668d3d49296beb7543aace1bde482b96ee294/src/simulation/grids.jl#L7
[C5]: https://github.com/CliMA/ClimaAtmos.jl/blob/dda668d3d49296beb7543aace1bde482b96ee294/src/config/type_getters.jl#L535-L546
[C6]: https://github.com/CliMA/ClimaAtmos.jl/blob/dda668d3d49296beb7543aace1bde482b96ee294/src/setups/WeatherModel.jl#L1-L33
[C7]: https://github.com/CliMA/ClimaCoupler.jl/blob/e3355c8ac9dd0f4787cbdafa93ed737a61a460a1/config/weather_configs/weatherbench_progedmf_1m.yml#L1-L23
[C8]: https://github.com/CliMA/ClimaAtmos.jl/blob/dda668d3d49296beb7543aace1bde482b96ee294/config/common_configs/numerics_sphere_he16ze63.yml#L1-L11
[C9]: https://github.com/CliMA/ClimaAtmos.jl/blob/dda668d3d49296beb7543aace1bde482b96ee294/config/default_configs/default_config.yml#L40-L48
[C10]: https://github.com/CliMA/ClimaAtmos.jl/blob/dda668d3d49296beb7543aace1bde482b96ee294/src/types.jl#L829-L900
[C11]: https://github.com/CliMA/ClimaAtmos.jl/blob/dda668d3d49296beb7543aace1bde482b96ee294/config/model_configs/les_isdac_box.yml#L12-L23
[C12]: https://github.com/CliMA/ClimaAtmos.jl/blob/dda668d3d49296beb7543aace1bde482b96ee294/config/model_configs/plane_schar_mountain_float32_test.yml#L1-L11
[C13]: https://github.com/CliMA/RRTMGP.jl/blob/19c6355a962940d8e6a2957f120c4c7299042da3/docs/src/howto/gpu.md#L120-L139
[C14]: https://github.com/CliMA/CloudMicrophysics.jl/blob/542dd84950373248eb53e49528a08b0fdb00f9b7/README.md#L41-L46
[C15]: https://github.com/CliMA/ClimaAtmos.jl/blob/dda668d3d49296beb7543aace1bde482b96ee294/config/default_configs/default_config.yml#L469-L471
[C15b]: https://github.com/CliMA/ClimaAtmos.jl/blob/dda668d3d49296beb7543aace1bde482b96ee294/calibration/experiments/gcm_driven_scm/run_calibration.jl#L3
[C15c]: https://github.com/CliMA/ClimaAtmos.jl/blob/dda668d3d49296beb7543aace1bde482b96ee294/calibration/experiments/gcm_driven_scm/README.md#L3
[C16]: https://github.com/CliMA/ClimaAtmos.jl/blob/dda668d3d49296beb7543aace1bde482b96ee294/docs/src/les_sgs.md#L5-L10
[C17]: https://github.com/CliMA/ClimaAtmos.jl/blob/dda668d3d49296beb7543aace1bde482b96ee294/config/default_configs/default_config.yml#L237-L239
[C17b]: https://github.com/CliMA/ClimaAtmos.jl/blob/dda668d3d49296beb7543aace1bde482b96ee294/config/default_configs/default_config.yml#L346-L351
[C18]: https://github.com/CliMA/ClimaAtmos.jl/blob/dda668d3d49296beb7543aace1bde482b96ee294/src/utils/AtmosArtifacts.jl#L108-L133
[C18b]: https://github.com/CliMA/ClimaAtmos.jl/blob/dda668d3d49296beb7543aace1bde482b96ee294/src/simulation/grids.jl#L313-L337
[C18c]: https://github.com/CliMA/ClimaAtmos.jl/blob/dda668d3d49296beb7543aace1bde482b96ee294/docs/src/topography.md#L55-L71
[C19]: https://github.com/CliMA/ClimaOcean.jl/blob/940e551532aaaea8995e83ac5a2eddc799054119/README.md#L8-L16
[C20]: https://github.com/CliMA/ClimaCoupler.jl/blob/e3355c8ac9dd0f4787cbdafa93ed737a61a460a1/README.md#L10-L15
[C20b]: https://github.com/CliMA/ClimaCoupler.jl/blob/e3355c8ac9dd0f4787cbdafa93ed737a61a460a1/config/ci_configs/cmip_oceananigans_climaseaice.yml
[C21]: https://github.com/CliMA/ClimaLand.jl/blob/69245261302509b2ea07ee63872b25f17bc0d92f/README.md#L28
[C22]: https://github.com/CliMA/ClimaDiagnostics.jl/blob/7fc5f050c9272fa043de1469513ddb4930038f89/src/netcdf_writer.jl#L22
[C22b]: https://github.com/CliMA/ClimaDiagnostics.jl/blob/7fc5f050c9272fa043de1469513ddb4930038f89/src/hdf5_writer.jl#L18
[C23]: https://github.com/CliMA/ClimaAtmos.jl/blob/dda668d3d49296beb7543aace1bde482b96ee294/src/diagnostics/core_diagnostics.jl#L469-L490
[C24]: https://github.com/CliMA/ClimaInitialConditions.jl/blob/6c1c1588c7840e0df7da3f16346f36838e4424ef/README.md#L7
[C25]: https://doi.org/10.1029/2019MS001621
[C24b]: https://github.com/CliMA/ClimaAtmos.jl/blob/dda668d3d49296beb7543aace1bde482b96ee294/Project.toml

[N1]: https://github.com/NumericalEarth/Breeze.jl/blob/6eabb8937670f61ad3bfd0ca2ad400a747681646/README.md#L8
[N1b]: https://github.com/NumericalEarth/Breeze.jl
[N2]: https://github.com/NumericalEarth/Breeze.jl/blob/6eabb8937670f61ad3bfd0ca2ad400a747681646/src/TimeSteppers/acoustic_runge_kutta_3.jl#L28-L64
[N2b]: https://github.com/NumericalEarth/Breeze.jl/blob/6eabb8937670f61ad3bfd0ca2ad400a747681646/src/AnelasticEquations/anelastic_dynamics.jl#L5
[N3]: https://github.com/NumericalEarth/Breeze.jl/blob/6eabb8937670f61ad3bfd0ca2ad400a747681646/docs/src/terrain_following_coordinates.md#L3-L9
[N4]: https://github.com/NumericalEarth/NumericalEarth.jl/blob/66112eac639e6163639b89f3fc1c684b146002d2/ext/NumericalEarthBreezeExt/breeze_nested_atmosphere.jl#L111-L240
[N5]: https://github.com/NumericalEarth/NumericalEarth.jl/blob/66112eac639e6163639b89f3fc1c684b146002d2/examples/breeze_downscaling_era5.jl#L9-L91
[N6]: https://github.com/NumericalEarth/Breeze.jl/blob/6eabb8937670f61ad3bfd0ca2ad400a747681646/src/AtmosphereModels/adiabatic_balance.jl#L1-L34
[N7]: #numericalearth
[N7a]: https://github.com/NumericalEarth/Breeze.jl/pull/287
[N7b]: https://github.com/NumericalEarth/Breeze.jl/pull/747
[N7c]: https://github.com/NumericalEarth/Breeze.jl/issues/825
[N7d]: https://github.com/NumericalEarth/Breeze.jl/issues/793
[N7e]: https://github.com/NumericalEarth/Breeze.jl/issues/897
[N7f]: https://github.com/NumericalEarth/Breeze.jl/pull/834
[N8]: https://github.com/glwagner/AtmosphericRivers.jl/blob/7cc5735d2ffe8ed6709433c60693d46f170aafc5/README.md#L56-L107
[N9]: https://github.com/glwagner/AtmosphericRivers.jl/blob/af7b94768b02c4e1984edf66157612a807f14804/README.md#L57-L69
[N10]: https://github.com/NumericalEarth/Breeze.jl/blob/6eabb8937670f61ad3bfd0ca2ad400a747681646/examples/bomex.jl#L42-L50
[N10b]: https://github.com/NumericalEarth/Breeze.jl/blob/6eabb8937670f61ad3bfd0ca2ad400a747681646/examples/rico.jl#L45-L53
[N10c]: https://github.com/NumericalEarth/Breeze.jl/blob/6eabb8937670f61ad3bfd0ca2ad400a747681646/examples/neutral_atmospheric_boundary_layer.jl#L38-L44
[N11]: https://github.com/NumericalEarth/BreezeBenchmarks/blob/626d58016751e28a1abd303c5a088f311e849e0b/benchmarks/by_branch/main/6eabb8937670f61ad3bfd0ca2ad400a747681646/benchmark_results.json#L175-L194
[N11b]: https://numericalearth.github.io/BreezeBenchmarks/
[N12]: https://github.com/NumericalEarth/Breeze.jl/blob/6eabb8937670f61ad3bfd0ca2ad400a747681646/src/Breeze.jl#L115-L233
[N13]: https://github.com/NumericalEarth/Breeze.jl/blob/6eabb8937670f61ad3bfd0ca2ad400a747681646/src/TurbulenceClosures/tke_based_turbulence_closure.jl#L5-L10
[N14]: https://github.com/NumericalEarth/NumericalEarth.jl/blob/66112eac639e6163639b89f3fc1c684b146002d2/src/DataWrangling/DataWrangling.jl#L409-L430
[N15]: https://github.com/NumericalEarth/NumericalEarth.jl/blob/66112eac639e6163639b89f3fc1c684b146002d2/src/DataWrangling/CopernicusDEM/CopernicusDEM.jl#L25-L44
[N15b]: https://github.com/NumericalEarth/NumericalEarth.jl/blob/66112eac639e6163639b89f3fc1c684b146002d2/src/DataWrangling/WorldCover/WorldCover.jl
[N16]: https://github.com/NumericalEarth/NumericalEarth.jl/blob/66112eac639e6163639b89f3fc1c684b146002d2/src/Bathymetry/regrid_bathymetry.jl#L172-L188
[N17]: https://github.com/NumericalEarth/NumericalEarth.jl/blob/66112eac639e6163639b89f3fc1c684b146002d2/src/EarthSystemModels/earth_system_model.jl#L6-L15
[N18]: https://github.com/NumericalEarth/NumericalEarth.jl/blob/66112eac639e6163639b89f3fc1c684b146002d2/examples/breeze_over_four_oceans.jl#L1-L18
[N19]: #numericalearth
[N19a]: https://github.com/NumericalEarth/NumericalEarth.jl/pull/676
[N19b]: https://github.com/NumericalEarth/NumericalEarth.jl/pull/753
[N19c]: https://github.com/NumericalEarth/NumericalEarth.jl/pull/664
[N19d]: https://github.com/NumericalEarth/NumericalEarth.jl/pull/695
[N20]: https://github.com/CliMA/Oceananigans.jl/blob/fec351f0d3657ec3421adfedde30418ee6c0ac3d/src/BoundaryConditions/open_boundary_schemes/gravity_wave_schemes.jl#L1-L56
[N20b]: https://github.com/CliMA/Oceananigans.jl/blob/fec351f0d3657ec3421adfedde30418ee6c0ac3d/src/OutputWriters/zarr_writer.jl#L8
[N21]: https://github.com/NumericalEarth/NumericalEarth.jl/blob/66112eac639e6163639b89f3fc1c684b146002d2/README.md#L86-L110
[N22]: https://arxiv.org/abs/2309.06662
[N23]: https://github.com/NumericalEarth/Terrarium.jl/blob/a8a0c35dae8a6d74298c30a98d2358a44844dedf/README.md#L22
[N25]: https://github.com/NumericalEarth/Breeze.jl/blob/6eabb8937670f61ad3bfd0ca2ad400a747681646/docs/src/index.md#L72
