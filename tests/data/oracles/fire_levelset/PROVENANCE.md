# WRF v4.7.1 fire-model oracle provenance

These fixtures are WRF v4.7.1's own fire spread (WRF-SFIRE lineage,
`ifire = 2`) for `woof/fire/levelset.py`, `woof/fire/heat.py` and
`woof/core/kernels/fire_levelset.cu`. The comparison is in
`tests/test_fire_levelset_wrf471_parity.py`. Every dumped float32 word must
equal WRF's: max ULP 0, raw-word equality, on the NumPy host reference and
on the device. No output is held out.

Upstream tag: `v4.7.1`.
Upstream commit: `f52c197ed39d12e087d02c50f412d90d418f6186`.
Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0, run through
`pixi exec -s gfortran -s binutils` (gfortran is not a project dependency).
C library: GNU libc 2.44 (the host's; `expf`/`powf` are glibc's).
Host: Linux 6.12 x86-64, little endian.

WRF v4 renamed WRF-SFIRE's `module_fr_sfire_{core,phys,util}.F` to
`module_fr_fire_{core,phys,util}.F`. These are the files compiled here.

## Build

```sh
bash tools/fire_levelset_wrf471_oracle/fetch_sources.sh WRF_SOURCE_ROOT
bash tools/fire_levelset_wrf471_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR
cp -r BUILD_DIR/fixtures/. tests/data/oracles/fire_levelset/
```

`build.sh` checks every pinned source against `SOURCES.sha256` before it
compiles anything. It also checks that `module_fr_fire_driver.F:941-953`
still reads as `driver_941_953.inc`. It compiles byte-unmodified
`share/module_model_constants.F`, `frame/module_wrf_error.F` and
`phys/module_fr_fire_{util,phys,core}.F` with:

```
-O0 -cpp -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8
-DLWORDSIZE=4 -ffree-form -ffree-line-length-none -fallow-argument-mismatch
```

`-fallow-argument-mismatch` is needed because `read_namelist_fire` passes
scalars and arrays to the same broadcast shim. The oracle never calls it.

No -O0 object references a vector libm symbol. The -Ofast positive control
emitted `_ZGVbN4v_expf`. The WRF objects call glibc's scalar `expf` and
`powf` (`libmvec-report.txt`).

The stubs in `stub_wrf.F90` compute nothing. They provide the serial build's
empty `TYPE(domain)` and `grid_config_rec_type`, `wrf_abort`, and the
single-rank broadcast and rank-query shims.

## What the driver does

`run_fire.F90` sets the `module_fr_fire_util` flags to the Registry defaults
(`Registry/registry.fire`, which `set_flags` copies in a WRF run). It fills
the `fire_params` arrays with given values; no fuel model is evaluated. It
then calls WRF's routines in `module_fr_fire_model.F`'s order:

- **Pass 4:** `prop_ls_rk3` (`tend_ls`, `fire_ros`), `tign_update`, and
  `reinit_ls_rk3` when it is on.
- **Pass 5:** `lfn = lfn_out`, then `ignite_fire` for each ignition line.
- **Pass 6:** `fuel_left`, `fuel_frac_burnt`, then `heat_fluxes`.
- **Driver:** `sum_2d_cells` of `fuel_frac`, `fgrnhfx` and `fgrnqfx`, with
  the driver's `fire_atm_feedback * sum * (1/(ir*jr))` scaling.

`init_no_fire` and `set_ideal_coord` set the start state and the node
coordinates. All arrays start at zero, as WRF's Registry fill does. Each
case dumps its inputs once, then every field at the listed steps:

| Case | Mesh (nodes, dx) | dt (s) × steps | What it reaches |
|---|---|---|---|
| `point_iso` | 40 × 40, 10 m | 2.0 × 60 | Isotropic point fire: WENO5 band, viscosity ramp, reinit, ignition growing to its radius |
| `line_wind` | 48 × 48, 10 m | 1.5 × 140 | Line walked from inside the ENO1 edge band; varying wind and slope; no-fuel patch the fire stops at |
| `substep` | 48 × 48, 10 m | 6.0 × 35 | As `line_wind`, each outer step four fire steps of 1.5 s, with the outer mean `GRNHFX`/`GRNQFX` |
| `options` | 48 × 48, 10 m | 1.5 × 50 | `fire_advection = 0`, no reinitialization; two ignitions, one at 20 s held to 25 s |
| `cap` | 40 × 40, 10 m | 0.4 × 25 | `fire_ros`'s 6 m/s cap taken out of the wind and slope parts |
| `asym` | 48 × 40, dx 10 m, dy 8 m | 0.6 × 120 | Every numeric switch off its Registry value: viscosity 0.2 / bg 0.1 / band 0.8 / ngp 3, `fire_lsm_band_ngp` 5, slope factor 1.5, feedback 0.5, two reinit iterations, `fire_lfn_ext_up` 0.5 |
| `weight_7`, `weight_30`, `weight_180`, `weight_900` | 40 × 40, 10 m | 2.0 × 40 | `fuel_left` across WRF's fuel weights (`fuel_time = weight/0.85`) |

`FIXTURES.sha256` has the hashes of every binary and manifest, about 2.9 MB
in all. `HARNESS.sha256` has the hashes of the harness that wrote them.

## Settings that differ from a default WRF run

- **`boundary_guard = -1`.** The Registry's `fire_boundary_guard = 8` stops
  the run when the fire comes within eight cells of the edge. That is a run
  control, not a field. The port has no stop, so the fire leaves the mesh.
- **`ischap = 0` everywhere.** `fire_ros`'s chaparral arm leaves `ros_base`
  undefined, so there is no defined answer to reproduce. The port has no
  chaparral arm.
- **Sub-stepping is harness-side.** WRF only warns when `dt` exceeds the CFL
  bound. The `substep` case's outer-step mean (sum in step order, then
  `/ nsub`) is the harness's, and is what `woof.fire.levelset.advance`
  returns.

No output is held out: every array the oracle dumps is compared, word for
word.
