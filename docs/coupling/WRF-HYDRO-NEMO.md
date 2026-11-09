# WRF-Hydro and WRF-NEMO coupling: what is ported, the licences, and the roadmap

Status as of 2026-11-08.  WOOF holds WRF physics to compiled WRF reference
builds; the WRF-Hydro, Noah `-DWRF_HYDRO`, Noah-MP `WRF_HYDRO` (submodule
`e5c0859`) and ocean-coupler lanes pin WRF 4.7.1 (upstream's baseline,
`f52c197e`); NEMO pins 5.0.2, the newest NEMO release.  This page records which parts of WRF's hydrology and ocean
couplings are ported to that standard, which are not and why, and what each
remaining step needs.

## What is in the tree

Numerics: the opt-in units follow the rest of WOOF (`woof/numerics.py`).
In the default `fast` mode the compile boundary drops their `--fmad=false`, so
NVRTC fuses multiply-adds and results can sit a few ULP from WRF-Hydro (the
diffusive channel's DIFFUSION does).  `strict` (`GPUWM_NUMERICS=strict` or
`GPUWM_WRF_EXACT=1`, which the test battery's conftest sets) keeps the
no-contraction request, and every WRF-Hydro and NEMO parity test's bitwise
claim holds there.


| Piece | Source | Where | Gate |
| --- | --- | --- | --- |
| Noah LSM built with `-DWRF_HYDRO`: ponded head `sfcheadrt` in; infiltration excess `infxsrt` and soil drainage `soldrain` out; the NDHMS edits to SMFLX, SRT and SSTEP | WRF 4.7.1 `phys/module_sf_noahdrv.F`, `phys/module_sf_noahlsm.F` (public domain; byte-identical to 4.6.1's) | `launch_noah(wrf_hydro=...)` -> `noah_column_hydro` (`woof/core/kernels/noah.cu`) | `tests/test_noah_wrf_hydro_parity.py`: 0 ULP in every field on CUDA (sm_86) and Metal (M3 Max), default and strict numerics |
| WRF's side of the OASIS coupling with an ocean model: `cpl_snd2` (EVAP-PRECIP, SURF_NET_SOLAR, SURF_NET_NON-SOLAR, TAUX, TAUY, TAUMOD) and `cpl_rcv` (SST, UOCE, VOCE merged under CPLMASK) | WRF 4.7.1 `frame/module_cpl.F` (public domain; byte-identical to 4.6.1's) | `woof/ocean/wrf_cpl.py`, `woof/ocean/wrf_cpl.cu` | `tests/test_wrf_cpl_wrf461_parity.py` (pending: #159) |

| WRF-Hydro subsurface routing (`rt_option = 1`): FINDZWAT, ROUTE_SUBSURFACE1 with GETSUB8, SUBSFC_RTNG | WRF 4.7.1 `hydro/Routing/Noah_distr_routing_subsurface.F90`, `Noah_distr_routing.F90` (UCAR WRF-Hydro licence) | **opt-in**: `optional/wrf_hydro/` via `woof.optin.load("wrf_hydro")` with `WOOF_ENABLE_WRF_HYDRO=1` | `tests/test_wrf_hydro_subsurface_parity.py`: six routing calls bitwise on CUDA and Metal; skips by named rule when the opt-in is off |

| NEMO 2-band penetrative solar radiation (`ln_qsr_2bd`, linear free surface, leapfrog): `qsr_ext_lev`, `qsr_2BD`, `tra_qsr` | NEMO 5.0.2 `src/OCE/TRA/traqsr.F90` (CeCILL v2.0) | **opt-in**: `optional/nemo/` via `woof.optin.load("nemo")` with `WOOF_ENABLE_NEMO=1` | `tests/test_nemo_traqsr_parity.py`: three steps bitwise in double on CUDA and Metal; skips by named rule when the opt-in is off |

## Decision (2026-11-07)

The project owner approved both licence questions below: WRF-Hydro routing
(UCAR licence, no sale) and NEMO kernels (CeCILL v2.0) may enter the tree
**only as explicit opt-ins**, treated like the LGPL glibc 2.39 libm port and
`cuda_powf.cuh`.  For each opt-in:

* the source lives under `optional/<name>/`, outside every Python package,
  with its licence text (`LICENSE`) and attribution (`NOTICE.md`) beside it;
* it is off by default and loads only through `woof.optin.load(name)` when its
  environment variable is set (`WOOF_ENABLE_WRF_HYDRO=1`,
  `WOOF_ENABLE_NEMO=1`) in a source checkout;
* it is in neither published wheel (`recast-woof`, `metal-woof`):
  `tests/test_optin_wheel_contents.py` asks setuptools what each wheel would
  contain and fails on any file under `optional/`, any `WOOF-OPTIN:` marked
  file, or the opt-in's licence text; it is also excluded from the release
  snapshot (`RELEASE-EXCLUDE.txt`);
* its tests skip, naming the opt-in rule, when it is off.

## The 2026-11-06 drafts

Untracked drafts found on the Mac checkout (`woof/hex/wrf_hydro*.py`,
`woof/hex/wrf_hydro_noah.cu`, `woof/hex/wrf_nemo.py`, their "oracles" under
`components/hex/tests/` and four tests) were read and none was kept:

* `wrf_hydro_oracle/wrf_hydro_column.F90` and `wrf_nemo_oracle/wrf_nemo_column.F90`
  describe themselves as "exact Fortran replica[s] of the CUDA kernel": the
  oracle was written from the kernel, so it can only agree with it.  The
  physics in both is invented (a bucket with a slope term; bulk fluxes with
  fixed drag coefficients and a constant saturation humidity).  Neither is
  WRF-Hydro or NEMO.
* `wrf_hydro_noah_wrapper.F90` wraps a second copy of `module_sf_noahlsm.F`
  with hand-set parameter tables, and states that WRF-Hydro is Apache 2.0,
  which is false (below).  WOOF already ports Noah against the WRF driver
  with WRF's own table reader; the WRF_HYDRO arm above extends that port
  rather than adding a second one.

## Licences (read 2026-11-07)

**WRF 4.7.1** (as 4.6.1) (`LICENSE.txt`): "NCAR and UCAR ... consider WRF to be in the
public domain for use by any person or entity for any purpose without any fee
or charge."  Everything ported above is WRF code.

**WRF-Hydro**, NCAR/wrf_hydro_nwm_public, pinned candidate `v5.4.0` at
`49b0fba3e859ef5288b06ec430621be03193f49e`, `LICENSE.txt` sha256
`434cc6a8692405f600a3b7dd45d6c3d1af0848543d0ffd2e9cb886037b6be41b`.  A
UCAR licence, not an OSI one.  Clause 1 grants use, derivative works and
redistribution and then says: "You shall not sell, license or transfer for a
fee the Software, or any work that in any manner contains the Software."
Clause 9 requires the licence text and a UCAR notice with every copy.  The
no-fee clause binds anything that contains WRF-Hydro-derived code, which an
Apache-2.0 distribution cannot promise its recipients.  Two further facts:

* WRF 4.7.1's (as 4.6.1's) own tarball carries a copy of WRF-Hydro under `hydro/` with no
  separate licence file, under WRF's public-domain statement.  Whether that
  statement reaches code UCAR also publishes under the restrictive licence
  is not settled by either text.
* Upstream WOOF withdrew a WRF-Hydro port: `RELEASE-EXCLUDE.txt` excludes the
  "withdrawn WRF-Hydro product door", and `licenses/NOTICE-t-route-reservoirs.txt`,
  `woof/core/kernels/NOTICE-CFE.txt` and `docs/nextgen-products.md` each state
  that their own adaptation (t-route reservoirs, CFE, the NextGen products)
  uses no source, port or fixture from that withdrawn port; they are notices
  about those adaptations, not a provenance statement for the whole tree.  The NextGen hydrology
  stack (t-route Muskingum-Cunge, CFE, Noah-OWP) is the upstream replacement.

So a port of WRF-Hydro's routing could only be an opt-in in the sense the fork
uses for the glibc 2.39 libm (#145): source in a separately licensed directory
outside every package, built only by an explicit task, excluded from the
wheels and the release export.  Unlike LGPL it would also bar anyone from
selling a copy of the repository that contains it.  The project owner
decided for the opt-in (above).

**NEMO**, `https://forge.nemo-ocean.eu/nemo/nemo.git`, tag `5.0.2` (commit
`dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796`), `LICENSE` sha256
`2dcf0966f7eaae3af05f3e640215ac537c76d938d50ad02ad9d4260f1ca4bda5`: the CeCILL
Free Software License Agreement, version 2.0 (2006-09-05); source headers say
"Software governed by the CeCILL license".  CeCILL is a strong copyleft:
a modification (a CUDA port of a NEMO routine is one) is distributed only
under CeCILL (5.3.2), and an "Internal Module", code that runs in the same
address space as the Software, counts as a Contribution.  WOOF loading
NEMO-derived kernels into its own process is exactly that, so a distributed
build containing them would have to be CeCILL (or GPL, through 5.3.4) as a
whole.  Keeping such kernels in the repository is permissible as separately
licensed files that no default build compiles; shipping them is not
compatible with the Apache-2.0 wheels.  The opt-in shape would therefore be
the #145 one, with the extra condition that an opt-in build is never
redistributed except under CeCILL/GPL.  The project owner decided for the
opt-in (above), with that condition.

**OASIS3-MCT** (the coupler WRF and NEMO use) is LGPL-3.0.  Nothing of it is
used: WOOF's coupling-interval averaging is its own and on-grid only.

## Roadmap

Each step is one PR under 300 files, held to a compiled oracle, CUDA and
Metal, and refuses what it does not port by name.

### WRF-Hydro (steps 3-5 go in the opt-in; step 2 is WRF code)

1. **Done**: the LSM side of the exchange for Noah (`sf_surface_physics=2`).
2. **Done: Noah-MP under WRF_HYDRO** (WRF code, no licence question;
   default build): `noahmp_lsm_step(wrf_hydro=...)` reads `sfcheadrt` and
   writes `INFXSRT = RUNSF`, `SOLDRAIN = MAX(RUNSB, 0)` and
   `QTILEDRAIN = QTLDRN` on land columns (module_sf_noahmpdrv.F
   :1211-1218).  WATER adds the head to QINSUR (:6173-6175), graded word for
   word against the module built with `-DWRF_HYDRO`
   (`tools/noahmp_wrf461_oracle/build_water_hydro.sh`,
   `tests/test_noahmp_water_wrf_hydro.py`, CPU and device); glacier columns
   send it to surface runoff (module_sf_noahmp_glacier.F :2143-2145), graded
   by CUDA/CPython parity as the glacier port is; the water-budget stops are
   compiled out as under WRF_HYDRO.  It runs on the selected-options water
   and glacier kernels, so the default kernels are untouched.  Refused by
   name: Hooghoudt tile drainage with WRF-Hydro's water table
   (`opt_tdrn = 2`, ZWATBLE2D -> WATBLED) and the staged column path.
3. **The coupling driver** `hydro/CPL/WRF_cpl/module_wrf_HYDRO.F90` and
   `HYDRO_drv/module_HYDRO_drv.F90`: **done**.  `aggregate`/`disaggregate`
   between the LSM grid and the routing grid (`AGGFACTRT`) are in the opt-in
   (`woof_wrf_hydro.grid_transform`, `tests/test_wrf_hydro_grid_transform_parity.py`);
   HYDRO_ini (Fulldom_hires.nc, hydro2dtbl.nc, CHANPARM.TBL, LAKEPARM.nc,
   GWBASINS.nc, GWBUCKPARM.nc read with netCDF4; the gridded network,
   slopes, retention, lakes and buckets derived as WRF-Hydro does) and
   `wrf_cpl_HYDRO`/`HYDRO_exe` are `woof_wrf_hydro.routing_domain` and
   `woof_wrf_hydro.coupled`, graded against WRF 4.7.1's own `wrf_cpl_HYDRO`
   over the whole hydro/ tree compiled with MPP_LAND
   (`oracle/build_coupled.sh`, `tests/test_wrf_hydro_coupled_parity.py`:
   HYDRO_ini's network and terrain and four coupling cycles bitwise, strict
   numerics).  The forecast route reaches it through `woof.wrf_hydro_route`
   (default build, imports nothing from the opt-in unless asked):
   `--wrf-hydro HYDRO.NAMELIST` on `woof run`/`woof go` or a `[wrf_hydro]`
   table; after every LSM call (`HYDRO_dt = DTBL`) the routing reads SMOIS,
   SH2O, INFXSRT, SOLDRAIN and returns SMOIS, SH2O, SFCHEADRT, which the
   next Noah or Noah-MP call reads through its WRF_HYDRO hooks.  Output:
   `frxst_pts_out.txt`, CHRTOUT- and RTOUT-like NetCDF files, and SFCHEADRT,
   INFXSRT, SOLDRAIN in history.  GWBASESWCRT = 3 (the 2-D aquifer, read
   from `gwhires.nc`, coupled to the soil and the channel) runs in the
   coupled driver with AGGFACTRT = 1 and is graded by the coupled oracle's
   gw2d variant; WRF 4.7.1 itself faults in HYDRO_ini's get_basn_area for
   any GWBASESWCRT > 0 that is not the bucket (it writes the unallocated
   basns_area), so the oracle harness stands in zero-length bucket arrays
   first, and readGW2d reads only an IX x JX corner of each field under
   MPP_LAND when AGGFACTRT > 1, which is refused.  Refused by name: Route_Link
   reaches (channel_option 1/2), UDMP_OPT = 1, IMPERV_ADJ, routing restarts and checkpoints, more than one domain, Noah
   with urban physics, NextGen (`--hydro`) at the same time.  Two WRF 4.7.1
   facts the coupled run follows: ZSOIL8 is WRF's ZS (the layer centres,
   module_wrf_HYDRO.F90 :94-99), not the namelist's; read_rt_nlst opens
   `namelist.hrldas` even when coupled to WRF (config.F90 :724), so a
   WRF+WRF-Hydro run directory needs one (WOOF does not read it).
   `tests/test_wrf_hydro_coupled_forecast.py` runs short Noah and Noah-MP
   forecasts through the driver with a per-cycle water-budget check, and
   checks that a run without the table calls the LSM exactly as before.
4. **Routing**, in this order, each from the pinned WRF-Hydro source with its
   own oracle: subsurface (`Routing/Noah_distr_routing_subsurface.F90`,
   **done** in the opt-in), overland (`Routing/Overland/`,
   `Noah_distr_routing_overland.F90`: **done** in the opt-in for
   rt_option = 1 and 2 (ROUTE_OVERLAND2; the subsurface routes with
   ROUTE_SUBSURFACE1 for both, as WRF-Hydro hard-codes),
   `tests/test_wrf_hydro_overland_parity.py`), channel (`Routing/module_channel_routing.F90`:
   Muskingum and Muskingum-Cunge reach routing **done** in the opt-in, with
   4.7.0's reservoir reaches (TYPEL = 1 routes its lake through the
   reservoir; TYPEL = 2 is not committed), `tests/test_wrf_hydro_channel_parity.py`;
   the diffusive wave on nodes, channel_option 3, **done** in the opt-in with
   level-pool lakes (`Reservoirs/Level_Pool/`, lake_option 1, and 4.7.0's
   pass-through, lake_option 2), both compiled as WRF-Hydro always is, with
   MPP_LAND, on one process, `tests/test_wrf_hydro_diffusive_parity.py`; lake_option 3's
   persistence/level-pool hybrid (USGS type 2, USACE type 3) and RFC-forecast
   reservoirs (types 4 and 5), with their timeslice and RFC time-series
   readers on the host, are **done** in the opt-in (`woof_wrf_hydro.reservoir_da`,
   run inside the diffusive channel's lake loop): `tests/test_wrf_hydro_reservoir_da_parity.py`
   grades 900 calls of five lakes against Reservoirs/ compiled whole
   (`oracle/build_reservoir_da.sh`), and the coupled oracle's lake_option 3
   variant grades the hybrid inside HYDRO_exe), then the bucket baseflow (`module_GW_baseflow.F90`
   simp_gw_buck, GWBASESWCRT 1 and 2: **done** in the opt-in,
   `tests/test_wrf_hydro_baseflow_parity.py`; the 2-D groundwater model,
   GWBASESWCRT 3 (`module_gw_gw2d.F90`: gw2d_ini, gwstep's ADI sweeps as
   WRF-Hydro's one-process MPP_LAND build runs them, the soil coupling and
   the LSM-grid aggregate), **done** in the opt-in,
   `tests/test_wrf_hydro_gw2d_parity.py`, with its exchange with the
   diffusive channel (the gwOption block of drive_CHANNEL and HYDRO_drv's
   head correction, `tests/test_wrf_hydro_diffusive_parity.py`); 4 remains).
5. **Configuration** (**done** in the opt-in: `woof_wrf_hydro.namelist`,
   `tests/test_wrf_hydro_namelist.py`, held to WRF 4.7.1's template and
   read_rt_nlst): `hydro.namelist` (`SUBRTSWCRT`, `OVRTSWCRT`,
   `CHANRTSWCRT`, `channel_option`, `rt_option` (1 and 2), `GWBASESWCRT`, `lake_option`, `DXRT`,
   `AGGFACTRT`, `DTRT_TER`, `DTRT_CH`) mapped to a WOOF table, every unported
   value refused by name.  Until step 4 lands there is nothing to route to:
   `launch_noah(wrf_hydro=...)` is an API for the coupling, not a model run.

### WRF-NEMO

1. **Pending in #159**: WRF's exchange contract (`cpl_snd2`/`cpl_rcv`),
   on a shared grid, with a conserving coupling-interval average and the
   `ExternalOcean` protocol in `woof/ocean/interface.py`.
2. **Done (SST only)**: the exchange wired into the driver
   (`woof/ocean/external.py`, an `[external_ocean]` table in the
   `ocean_config` TOML): `cpl_store_input`'s SST_INPUT capture at
   itimestep 1 (WRF 4.7.1 `solve_em.F:379`); `cpl_rcv`'s SST merge at the top of every
   surface-driver call (`module_surface_driver.F:1507`), followed by
   `sst_update = 1`'s water lines (TSK and the top soil layer take SST in
   250-350 K, :1721-1728), without which the merged SST never reaches TSK;
   and `cpl_snd` once per completed step (`module_integrate.F:460`) with
   U_PHY/V_PHY/RHO as `phy_prep` left them.  At each coupling time the
   interval average goes to the ocean, which steps the interval (sequential
   coupling).  Refused by name: an ocean that returns currents (WOOF's
   surface layers do not read UOCE/VOCE), sea ice and the lake model under
   the coupling (sst_update's ice and lake lines are not wired), a coupling
   interval that is not a whole number of steps, and restart of a coupled
   run.  `tests/test_external_ocean_coupling.py` runs the real driver
   against a stub ocean, and the NEMO column through the same wiring.
3. **An ocean**: the NEMO opt-in has begun with `traqsr` (2-band penetrative
   solar, the consumer of SURF_NET_SOLAR) and the equation of state
   (`eosbn2.F90`, TEOS-10/EOS-80/S-EOS, `tests/test_nemo_eos_parity.py`)
   and TKE vertical mixing (`zdfsh2.F90`, `zdftke.F90`, no sea ice,
   `tests/test_nemo_tke_parity.py`) and the implicit vertical diffusion of
   T and S (`trazdf.F90`, `tests/test_nemo_trazdf_parity.py`), and those
   pieces stepped together as a coupled tracer column,
   `woof_nemo.column.NemoColumnOcean`, which implements `ExternalOcean`:
   stpmlf.F90's order with sbc_ssm, sbccpl.F90's `sbc_cpl_rcv` ocean-only
   mapping of WRF's six fields, the bottom drag of `zdfdrg.F90`, `tra_sbc`,
   `tra_atf_qco`'s Asselin filter and `sbc_cpl_snd`'s SST, word for word
   against those files compiled whole (`oracle/build_column.sh`,
   `tests/test_nemo_column_parity.py`, CUDA and Metal).  The column is an
   ocean at rest on a z-coordinate, so it couples SST only (UOCE/VOCE are
   zero); NEMO 5.0 runs with a two-cell halo (`mppini.F90`: `nn_hls =
   MAX(2, nn_hls)`), which the column oracle uses and which does not reach
   the interior columns.  FCT tracer advection (`traadv_fct.F90`:
   upstream step, 2nd/4th-order centred and 4th-order compact
   anti-diffusive fluxes, Zalesak's limiter; `tests/test_nemo_traadv_parity.py`)
   is ported given the transports; it does not enter the resting column.
   So is iso-level lateral diffusion (`traldf_lev.F90`, laplacian and
   bilaplacian, `tests/test_nemo_traldf_parity.py`) given ahtu/ahtv.  The
   other advection schemes, the iso-neutral diffusion operators (traldf_iso,
   traldf_triad), ldf_tra's diffusivity fields and the dynamics remain.  The
   alternative is an external NEMO through a real coupler (out of
   process; NEMO stays a separate program and its licence does not reach
   WOOF), or, after the licence decision, NEMO kernels behind an opt-in with
   an oracle built from the pinned NEMO tree (XIOS, NetCDF and MPI are the
   toolchain cost).  The fields NEMO's `sbccpl.F90` can send that WRF 4.7.1
   does not receive (sea-ice fraction, sea-surface height) stay refused by
   name in `wrf_cpl.NOT_IN_WRF_461` until a WRF version that receives them is
   the reference.
4. **Remapping** between different atmosphere and ocean grids (OASIS's SCRIP
   conservative remap) is refused by `wrf_cpl.require_shared_grid`; a port
   would need its own licence read (OASIS3-MCT and SCRIP).
