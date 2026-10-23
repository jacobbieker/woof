# Fire-spread oracle: WRF v4.7.1 `module_fr_fire_phys.F`

What these fixtures are, where their bytes came from, and the one recorded
word the ports are not held to.

## Source

WRF v4.7.1 (tag `v4.7.1`, commit `f52c197ed39d12e087d02c50f412d90d418f6186`),
fetched by `tools/fire_ros_wrf471_oracle/fetch_sources.sh` and refused by
`build.sh` unless every byte matches `SOURCES.sha256`:

| file | sha256 |
|---|---|
| `phys/module_fr_fire_phys.F` | `8185c11a706983db747c7e5059227d05e02185fa58f030126f3d07723acbab14` |
| `phys/module_fr_fire_util.F` | `ab9499f12b305257fd62a0c76f299ee308b7fabc98c6fae073ba25f132ec8d27` |
| `share/module_model_constants.F` | `5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062` |
| `test/em_fire/namelist.fire` | `2ad6471a8c2e8eb67e3e0ea01e0483f36a9f9915a5c87be7fd8cc98278f941fc` |

WRF v4.7.1 carries the WRF-SFIRE fuel and spread physics as
`phys/module_fr_fire_phys.F`.  There is no `module_fr_sfire_phys.F` in the
WRF tree; that file name belongs to the openwfm WRF-SFIRE fork, whose
`fire_ros` differs: it premultiplies the coefficients, normalises the
normal and splits into `fire_ros_cawfe`.  The ports transcribe WRF v4.7.1.

Routines exercised, all byte-unmodified: `init_fuel_cats` (with
`read_namelist_fire`), `set_fire_params` and `fire_ros` (`ibeh = 1`).
`fire_risk` does not exist in WRF v4.7.1, so it is not covered.

## Build

`tools/fire_ros_wrf471_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR`, with
gfortran from conda-forge (`pixi exec -s gfortran -s binutils -- bash
build.sh ...`):

- `TOOLCHAIN.txt`: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0, GNU libc
  2.44, Linux x86_64.
- `-O0`, default real(4).  This matches WRF's `RWORDSIZE=4` build of this
  physics.
- `-fallow-argument-mismatch` on `module_fr_fire_phys.F` only.  WRF passes
  scalars and arrays to the same `wrf_dm_bcast_*` externals.  WRF's own
  configure adds the flag for gfortran 10 and later, and it changes no
  arithmetic.
- `wrf_stubs.F90` supplies serial stand-ins for the framework calls: one rank
  that is the monitor, and broadcasts that do nothing.  They compute nothing
  the fixture records.
- libmvec receipts are in `libmvec-report.txt`.  The -O0 objects reference
  only the scalar `expf`, `powf` and `logf`.  The -O2 `-ftree-vectorize`
  build of the module also pulls in none of the vector ones.  The positive
  control does show `_ZGVbN4v_expf`, which proves the guard can see them.

`HARNESS.sha256` hashes the harness that built these bytes, and
`FIXTURES.sha256` hashes every `.bin` and `MANIFEST.txt`.  Both are checked
by `tests/test_fire_ros_wrf471_parity.py`.

## Fixtures

Each fixture is one case directory, written by
`tools/urban_wrf471_oracle/oracle_io.F90`.  Arrays are raw little-endian
float32 or int32 in Fortran order, described by `MANIFEST.txt`.
`namelist.fire.output` is WRF's own echo of the tables it ran with.

| case | how WRF is set up | fuel numbers | (wind, slope, normal) cases |
|---|---|---|---|
| `namelist/` | beside the pinned `namelist.fire` (13 Anderson categories, `nfuelcats = 13`) | 1-14, 15, 91 | 120: 8 winds x 5 slopes x 3 unit normals |
| `defaults/` | no `namelist.fire`, so the compiled DATA (13 Anderson + Scott & Burgan 40, `nfuelcats = 53`) | 1-14, all 40 S&B except 204, 50, 91, 99 | 36, including the non-unit normal (0.3, 0.2) |

Both cases cross their fuel numbers with 28 surface moistures: 0, 0.02,
0.05, 0.08, 0.10, 0.5 and 1.0, plus every moisture of extinction (0.12 to
0.40) with its two float32 neighbours.  Both run `fire_advection` = 0 and 1.
`tests/test_fire_ros_wrf471_parity.py::test_sweep_reaches_every_branch`
asserts that the sweep exercises each of these:

- backing, calm, oblique and beyond-30 m/s wind;
- up- and down-slope terrain;
- moisture above extinction, where `r_0 < 0`;
- chaparral;
- no fuel;
- the 6 m/s cap, including its sequential trim driving `ros_slope`
  negative.

## Parity

The float32 CPU port (`woof.fire.wrf_sfire`, NumPy with glibc's
`powf`/`expf` from `woof.core.noahmp_libm`) and the GPU port
(`kernels/fire_spread.cu` through direct NVRTC with `-fmad=false
--ftz=false` and `glibc_flt32.cuh`) both equal every compared word.  Max
ULP is 0:

| field | words compared | held out | CPU vs GPU | CPU vs oracle | GPU vs oracle |
|---|---:|---:|---:|---:|---:|
| `fuel_time` | 2016 | 0 | 0 | 0 | 0 |
| `fgip` | 2016 | 0 | 0 | 0 | 0 |
| `ischap` | 2016 | 0 | 0 | 0 | 0 |
| `betafl` | 2016 | 0 | 0 | 0 | 0 |
| `bbb` | 2016 | 0 | 0 | 0 | 0 |
| `phiwc` | 2016 | 0 | 0 | 0 | 0 |
| `r_0` | 2016 | 0 | 0 | 0 | 0 |
| `iboros` | 2016 | 0 | 0 | 0 | 0 |
| `ros_base` | 211680 | 8736 | 0 | 0 | 0 |
| `ros_wind` | 220416 | 0 | 0 | 0 | 0 |
| `ros_slope` | 220416 | 0 | 0 | 0 | 0 |

## Held out

`ros_base` is held out on chaparral cells (`ischap = 1`, category 4):
8736 words.  On its chaparral branch WRF v4.7.1's `fire_ros` never assigns
its `intent(out)` argument `ros_base`, but the excess test after the branch
reads it.  WRF therefore uses whatever its caller left there.  The driver
passes 0.0 on every call, and the port returns 0.0, so both evaluate the
excess from the same 0.  The recorded words are the driver's 0.0, which is
a choice of the harness and not WRF arithmetic.
`test_held_out_set_is_exactly_chaparral_ros_base` pins that the held-out
set is exactly those cells and that the recorded value there is 0.0.

## WRF behaviour the ports keep

These are WRF's choices, transcribed and documented rather than corrected:

- **`fgi` is treated as wet mass**: `fuelloadm = (1 - fmc/(1+fmc)) fgi`.
- **The moisture damping is not clipped.**  Above the moisture of extinction
  the cubic is negative, so `r_0` and the spread terms are negative.  The
  level set's `max(rr, 0)` (`fire_grows_only`, see
  `woof.fire.wrf_sfire.fire_ros_total`) stops the fire.
- **The normal is used as given** (`scale = 1`).
- **The 6 m/s cap is applied sequentially.**  The excess comes out of
  `ros_wind` first, and the slope share is then computed from the reduced
  wind.  This can drive `ros_slope` negative and leaves the total a few
  ULPs either side of 6.
- **The compiled defaults reject SB4.**  They declare `nfuelcats = 53` but
  place SB4 (204) at category 54, so `set_fire_params` crashes on it.
  `woof.fire.wrf_sfire.category_rows` raises the same way, and the fixture
  leaves 204 out.

## Scott & Burgan 40 and the full Rothermel model

WRF's compiled DATA hold the Scott & Burgan 40 only as one-particle lumped
categories (one SAV, one dead load, and a separate live-herb load used for
curing).  The `defaults` case holds the ports to WRF bit for bit on those.

The full multi-class Rothermel (1972) / Albini (1976) model of
`woof.fire.rothermel` has no Fortran routine in WRF for any of the 53
standard models.  It is tested two ways:

- **Against its own float64 NumPy reference.**  The reference uses the C
  library's `exp`/`pow` through `woof.core.host_libm`, and the GPU kernel
  uses `glibc_flt64.cuh`.  CPU and GPU differ in 0 words on every output
  (`tests/test_fire_rothermel.py::test_gpu_full_model_*`).
- **Against published values, at their published precision.**  These are
  BehavePlus's regression values (firelab/behave `testBehave.cpp`, six
  decimals), Andrews (2018) RMRS-GTR-371 Tables 8, 17, 21-24 and its worked
  examples, and Anderson (1982) INT-122's representative rates.

## Regenerating

```
bash tools/fire_ros_wrf471_oracle/fetch_sources.sh /tmp/wrf471
pixi exec -s gfortran -s binutils -- \
    bash tools/fire_ros_wrf471_oracle/build.sh /tmp/wrf471 /tmp/fire_build
cp -r /tmp/fire_build/fixtures/. tests/data/oracles/fire_ros/
```

Keep this file and update its tables if anything changes.
