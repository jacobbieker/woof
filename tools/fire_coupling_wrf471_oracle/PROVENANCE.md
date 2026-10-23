# WRF v4.7.1 fire-coupling oracle provenance

These fixtures are WRF v4.7.1's own two-way fire driver (`ifire = 2`,
WRF-SFIRE's level set coupled to the atmosphere) for
`woof/fire/coupling.py` and `woof/core/kernels/fire_coupling.cu`, with the
level set of `woof/fire/levelset.py` underneath. The comparison is in
`tests/test_fire_coupling_wrf471_parity.py`. Every dumped float32 word must
equal WRF's: max ULP 0, raw-word equality, on the NumPy host reference and
on the device, and the device must equal the host. No output is held out.

Upstream tag: `v4.7.1`.
Upstream commit: `f52c197ed39d12e087d02c50f412d90d418f6186`.
Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0, run through
`pixi exec -s gfortran` (gfortran is not a project dependency).
C library: GNU libc 2.44 (the host's; `expf`/`powf`/`logf`/`cosf` are
glibc's).
Host: Linux 6.12 x86-64, little endian.

## Build

```sh
bash tools/fire_coupling_wrf471_oracle/fetch_sources.sh WRF_SOURCE_ROOT
pixi exec -s gfortran -- \
    bash tools/fire_coupling_wrf471_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR
cp -r BUILD_DIR/fixtures/. tests/data/oracles/fire_coupling/
```

`build.sh` checks every pinned source against `SOURCES.sha256` before it
compiles anything. It compiles byte-unmodified
`share/module_model_constants.F`, `frame/module_wrf_error.F` and
`phys/module_fr_fire_{util,phys,core,model,atm,driver,driver_wrf}.F` with:

```
-O0 -cpp -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8
-DLWORDSIZE=4 -ffree-form -ffree-line-length-none -fallow-argument-mismatch
```

and no `-DDM_PARALLEL`: one rank, one patch, one tile, as woof runs. The
program runs in a directory holding the pinned `test/em_fire/namelist.fire`,
which `init_fuel_cats` reads, as a WRF fire run does.

No -O0 object references a vector libm symbol. The -Ofast positive control
emitted `_ZGVbN4v_expf`. The WRF objects call glibc's scalar `expf`, `powf`,
`logf`, `cosf` and `tanhf` (`libmvec-report.txt`). At WRF's own
`-O2 -ftree-vectorize -funroll-loops`, `module_fr_fire_driver.F` pulls in
`_ZGVbN4v_logf` (the vectorised wind interpolation); the -O0 reference is
the scalar `logf` the device unit transcribes.

## The stubs

WRF's `TYPE(domain)` and `TYPE(grid_config_rec_type)` are generated from the
Registry, so a standalone build has to declare them. `stub_wrf.F90` does,
by hand, with exactly the members the pinned fire sources reference, under
`Registry/registry.fire`'s names, ranks and defaults. The arrays are
POINTERs, as in WRF's generated type, so the driver's `fp%vx => grid%uf`
aliases them. `get_ijk_from_subgrid` is `frame/module_domain.F:287-326`'s
arithmetic on the stub's own atmospheric bounds. The rest is what `nm -u`
reports undefined outside libm and libgfortran: `wrf_dm_maxval` on one
rank, the broadcast and rank-query shims, and the WRF abort. Nothing in
the stubs computes a physical quantity.

## What the driver does

`run_fire_coupling.F90` fills the stub domain the way a WRF run would find
it -- wind, geopotential, roughness, terrain, phy_prep's `rho`, `z_at_w` and
`dz8w`, the dry column mass and the hybrid weights, the fuel and fire-mesh
terrain -- and then calls WRF's own entry points in WRF's order:

- `fire_driver_em_init` at `itimestep = 0`, as `start_domain_em` does
  (passes 1 and 2: coordinates, `set_nfuel_cat`, `set_fire_params`,
  `init_no_fire`);
- for each step, `itimestep = itimestep + 1` (`solve_em.F:376`), a new
  wind (sheared, varying in space and turning with the step), then
  `fire_driver_em_step` (passes 3 to 6: `interpolate_atm2fire`, the level
  set, the ignitions, `fuel_left`, `heat_fluxes`, `sum_2d_cells` and the
  driver's scaling, then `fire_tendency`).

Each case dumps its configuration and inputs once, the state after
initialization, and every output at the listed steps:

| Case | Atmosphere (nx × ny × nz, dx) | Fire mesh | dt (s) × steps | What it reaches |
|---|---|---|---|---|
| `ideal` | 12 × 10 × 8, 30 m | 48 × 40 (4 × 4) | 0.5 × 40 | Ideal coordinates; uniform fuel (`fire_fuel_read = 0`, category 3); a walked ignition line and a later point; sheared wind turning with time; sloping terrain; varying roughness |
| `real` | 12 × 10 × 8, 25 m | 60 × 40 (5 × 4) | 0.6 × 30 | Longitude/latitude ignitions (`unit_fxlong` from `cen_lat`); a third line counted out by its zero radius; a fuel map with no-fuel and timber cells (`fire_fuel_read = -1`); `fire_atm_feedback = 0.7`; `fire_wind_height = 10`; other extinction depths and crown height |
| `edge` | 12 × 10 × 3, 30 m | 48 × 40 (4 × 4) | 0.5 × 40 | A model top below `fire_wind_height` (the top layer's wind); roughness above it (zero wind); fuel by altitude (`fire_fuel_read = 1`); `fire_advection = 0`; no reinitialization |
| `tendency` | 9 × 7 × 10, 50 m | -- | -- | `fire_tendency` alone, with canopy fluxes on both sides of the crown height |

`FIXTURES.sha256` has the hashes of every binary and manifest, about 2.2 MB
in all. `HARNESS.sha256` has the hashes of the harness that wrote them.

## Settings and differences from a default WRF run

- **`fire_boundary_guard = -1` in every case.** The Registry's 8 stops the
  run when the fire comes within eight fire-mesh nodes of the edge. That is
  a run control, not a field, and woof does not stop the run: the fire
  leaves the mesh.
- **`tracer_opt = 0`.** woof turns the fire's smoke into a flux for its own
  tracer's surface seam (kg m-2 s-1), not WRF's per-step g/kg increment of
  the `tracer` array, so there is no tracer word to compare.
  `add_fire_tracer_emissions` itself is held to WRF by
  `tools/smoke_wrf471_oracle`. WRF v4.7.1's loop in it is `do i=i_st,i_st`
  (`module_fr_fire_atm.F:81`), which reaches only the first interior
  column of a tile; woof's source reaches every interior column.
- **`fire_sfc_flx = 0`, `fire_smk_scheme = 0`.** The truncated-Gaussian
  schemes (1) are not ported: `tg_dist` reads `z_w` before it sets it
  (`module_fr_fire_atm.F:342`), so they have no defined answer.
- **One fire step per atmospheric step.** WRF takes no sub-steps; it only
  warns when `dt` exceeds the level set's CFL bound. The coupling calls
  `woof.fire.levelset.step` once per step and does not use `advance`'s
  sub-stepping.
- **The fire clock is `time_start = itimestep * dt` in REAL(4)**, with
  `itimestep` the domain's 1-based step number, as `fire_driver_em`
  computes it; WRF assumes `dt` does not change, and woof refuses the fire
  under an adaptive time step.

No output is held out: every array the oracle dumps is compared, word for
word.
