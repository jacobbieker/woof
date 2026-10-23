# Smoke physics oracle: WRF v4.7.1, provenance

This harness holds `woof.smoke` to WRF's own Fortran, word for word. The
WRF sources are WRF v4.7.1 (tag `v4.7.1`, commit
`f52c197ed39d12e087d02c50f412d90d418f6186`), byte-unmodified and pinned in
`SOURCES.sha256`:

| file | sha256 | use |
|---|---|---|
| `chem/module_chem_plumerise_scalar.F` | `57d2cc62...3b8633` | compiled: the Freitas plume model |
| `chem/module_zero_plumegen_coms.F` | `98cac206...afcf` | compiled: its module state |
| `chem/module_plumerise1.F` | `42e8dd19...875d` | cited: how WRF-Chem calls `plumerise` |
| `chem/module_aer_drydep.F` | `9bd19b05...3d63` | cut: `zhang2001_aer_drydepvel_1` |
| `chem/module_gocart_settling.F` | `9ace2538...f9b0` | cut: `settling` |
| `phys/module_fr_fire_atm.F` | `61583834...a272` | compiled: `add_fire_tracer_emissions` |
| `share/module_model_constants.F` | `5b80377f...7062` | compiled: WRF's constants |
| `Registry/registry.fire` | `a4d146b5...e00e9b` | cited: `fire_tracer_smoke` default 0.02 |
| `phys/module_data_gocart_dust.F` | `86941008...2329` | cited: GOCART's `dyn_visc = 1.5E-5` |

Full digests are in `SOURCES.sha256`; `fetch_sources.sh DEST` fetches the
files and refuses any other bytes.

## Regenerating

```sh
bash tools/smoke_wrf471_oracle/fetch_sources.sh WRF_SOURCE_ROOT
bash tools/smoke_wrf471_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR
cp -r BUILD_DIR/fixtures/. tests/data/oracles/smoke/
```

`build.sh` checks every source against its pin, cuts the two routines with
`extract.py` (`extract-report.txt` records the line ranges and the sha256 of
each cut), compiles everything at `-O0 -ffree-form -ffree-line-length-none`
with real(4) defaults, and runs the four drivers. The plume driver runs one
process per case (see the plume-rise section). The fixtures carry their own
receipts, which `tests/test_smoke_physics.py` checks:

* `TOOLCHAIN.txt`: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0, GNU libc
  2.44, Linux 6.12 x86_64. gfortran is not a project dependency; the build
  used `pixi exec -s gfortran`.
* `libmvec-report.txt`: the libm symbols of every `-O0` object (scalar
  `expf`/`powf`/`exp`/`pow` only, no `_ZGV*` vector form), what WRF's own
  `-O2 -ftree-vectorize` flags would pull in (`_ZGVbN4vv_powf` for the plume
  model), and the `-Ofast` positive control that proves the guard can see a
  vector symbol.
* `SOURCES.sha256`, `HARNESS.sha256` (every harness file), `FIXTURES.sha256`
  (every fixture word file).

The fixtures were built twice, once per component and once by the shared
`build.sh`, and came out byte-identical.

## The ports

| WRF routine | woof (CuPy, production) | woof (NumPy reference) | kernel |
|---|---|---|---|
| `add_fire_tracer_emissions` | `emission.fire_tracer_emissions` | `fire_tracer_emissions_np` | `smoke_fire_tracer_emissions` |
| `plumerise` | `plume_rise.plumerise` | `plumerise_np` | `smoke_plumerise_columns` |
| `zhang2001_aer_drydepvel_1` | `deposition.zhang2001_depvel` | `zhang2001_depvel_np` | `smoke_deposition_zhang2001` |
| `settling` | `deposition.gocart_settling` | `gocart_settling_np` | `smoke_deposition_gocart_settling` |

All kernels are one NVRTC unit (`woof/smoke/_kernels.py`): `glibc_flt32.cuh`
and `glibc_flt64.cuh` followed by `smoke_emission.cu`, `smoke_plumerise.cu`
and `smoke_deposition.cu`, compiled directly with `-std=c++17 -fmad=false
--ftz=false`. Fortran's real(4) `EXP` and real `**` are glibc's `expf`/`powf`
(`gfk_exp`, `gfk_pow`; on the CPU, `woof.core.noahmp_libm`), REAL*8's are
glibc's `exp`/`pow` (`g64_exp::exp`, `g64_pow::pow`; on the CPU,
`woof.core.host_libm`). Comparison is raw-word equality
(`woof.verify.smoke_oracle.assert_words_equal`, where `-0` and `+0` are one
ULP apart).

## Fire smoke emission (`emission/`)

**Routine.** `add_fire_tracer_emissions`, `phys/module_fr_fire_atm.F:15-106`,
compiled whole and byte-unmodified (with `stub_wrf.F90` for
`module_state_description`'s `num_tracer`/`p_fire_smoke` and the three
framework calls the module names, none of which takes part in the
arithmetic).  `tracer_opt = 3`, `fire_smk_scheme = 0`.

**Driver.** `run_emission.F90`, four cases:

| case | atm grid (nx, ny, nz) | refinement (ir, jr) | fire_tracer_smoke | inputs |
|------|----------------------|---------------------|-------------------|--------|
| case01 | 6, 5, 4 | 4, 4 | 0.02 (WRF default) | zero initial tracer |
| case02 | 7, 6, 3 | 3, 2 | 0.015 | non-zero initial tracer |
| case03 | 5, 4, 5 | 10, 10 | 0.009 | most fire cells burning |
| case04 | 4, 4, 2 | 1, 1 | 0.02 | fuel loads over six decades |

Burnt fractions run from none through partial to whole cells.  Every word
of the output tracer, the columns WRF changes and the ones it leaves, is
compared.

**Held out, with reasons.**

1. *WRF emits into one column per tile.*  The i loop is `do i=i_st,i_st`
   (`module_fr_fire_atm.F:81`), so on a tile only the first interior column
   receives smoke.  woof applies the statement to every interior column.
   To hold every column to WRF's own arithmetic, the driver calls the
   routine once per interior column on a one-column tile (`its = ite = i`,
   with that column's fire-grid tile), which is a legitimate tiling the
   routine supports; the order of accumulation within the column (joff
   outer, ioff inner) is the routine's.
2. *`fire_smk_scheme = 1` is not ported.*  `tg_dist` evaluates
   `xi=(z_w-fire_peak_hgt)/...` with the local `z_w` never assigned
   (`module_fr_fire_atm.F:342`): WRF reads an undefined value.
3. *A fire-grid `fire_tracer_smoke`.*  woof also accepts a per-fire-cell
   factor so the PM2.5 factor can follow the fuel.  With a field equal to
   the scalar everywhere the result is the scalar's word for word
   (`test_a_field_factor_equal_to_the_scalar_is_the_scalar`); a varying
   field is graded CPU against GPU only.

**Result.** NumPy reference vs WRF, CuPy kernel vs WRF and CuPy vs NumPy:
max ULP 0 on every word of every case.

## Plume rise: the Freitas et al. (2007) model of WRF-Chem v4.7.1

Port: `woof/smoke/plume_rise.py` (CuPy entry `plumerise`, NumPy reference
`plumerise_np`), device code `woof/core/kernels/smoke_plumerise.cu`.
Oracle: `chem/module_chem_plumerise_scalar.F` and
`chem/module_zero_plumegen_coms.F`, byte-unmodified (SOURCES.sha256),
compiled at -O0 against `share/module_model_constants.F` (real(4), WRF's
default), driven by `run_plumerise.F90`.  Parity test:
`tests/test_smoke_plumerise_wrf471_parity.py`.

### What is compared

Eight synthetic soundings (`run_plumerise.F90:make_case`), 48 levels to
23.7 km, so the plume grid's `kmt` is 199 in every case:

| case | biomes (fire size m2, flaming fraction) | column | reaches |
|------|------------------------------------------|--------|---------|
| 01 | tropical forest (2e5, 0.6) | dry, 4 m/s | 2.0-2.4 km |
| 02 | boreal forest (8e5, 1.0) | moist | 10.0-10.3 km: condensation, rain, ice, melting |
| 03 | savanna (5e4, 0.4) | strong inversion, 6 m/s | 1.0 km |
| 04 | grassland (1e5, 0.8) | calm | 1.8 km, the imm = 2 shortcut |
| 05 | all four | sheared | 2.1-2.8 km |
| 06 | savanna (200 m2) | stable, windy | 0.1-0.8 km |
| 07 | tropical forest (1e6, 1.0) | very moist, deep | 13.4-13.5 km: the full microphysics |
| 08 | boreal + grassland | windy, cold | 1.2-1.6 km |

Per case, word for word (max ULP 0, raw-word equality):

* `eburn_out` (48 words): `plumerise`'s output, called as
  `module_plumerise1.F` calls it (eburn_out zeroed, nspecies = 1);
* `ztopmax` (8), `k1` (4), `k2` (4): the plume tops and injection layer
  `plumerise` keeps local, from a second fresh process (`run_plumerise ROOT
  N tops`) that repeats `plumerise`'s own calls (zero_plumegen_coms,
  get_env_condition, get_fire_properties, makeplume, the grassland
  shortcut, set_flam_vert);
* the plume model's module state as each `makeplume` returns: `w, t, qv,
  qc, qh, qi, rho, radius, vel_p, rad_p, wc, qsat` on all 200 plume levels
  (19,200 words per case) and `time, dt, wbar, dqsdz` (32), `mintime` (8);
* the 59 real(4) literals the port spells and 16 constant expressions the
  routines fold (`plumerise_literals`), so no literal is parsed through
  binary64 into a different word.

Measured on an RTX A6000 (sm_86, NVRTC 12.9), gfortran 16.2.0, glibc 2.44:

| field | words per side | CPU vs oracle | GPU vs oracle | GPU vs CPU |
|-------|---------------:|--------------:|--------------:|-----------:|
| eburn_out | 384 | 0 | 0 | 0 |
| ztopmax | 64 | 0 | 0 | 0 |
| k1, k2 | 32 + 32 | equal | equal | equal |
| plume state (12 fields x 200 levels x 8 runs) | 153,600 | 0 | 0 | 0 |
| time, dt, wbar, dqsdz | 256 | 0 | 0 | 0 |
| mintime | 64 | equal | equal | equal |
| literals, folded constants | 59 + 16 | 0 | (compiled in) | - |

GPU vs CPU is also compared on a (48, 2, 3) domain mixing burning and
skipped columns, under WRF's heat-flux table and under a 50 kW m-2 forest
table (every `PlumeriseResult` field, 0 ULP).

### Held out, and why

1. **WRF's cross-column state.**  The model keeps its state in module
   variables that `plumerise` zeroes on its first call only
   (`ncall`, line 117) and the plume grid is built once (`n_setgrid`,
   line 297).  A later column therefore starts with the scalars `WBAR` and
   `DQSDZ` its predecessor left (read by the first `LBOUND -> WATERBAL` of
   its `INITIAL`), with `zzcon` above its own top from a taller predecessor,
   and with levels above its `kmt` as the predecessor left them.  WRF's
   answer for a column depends on which column it computed before.  The
   port starts every column from the first-call state, which is what a
   fresh WRF process gives its first column; that is why `build.sh` runs
   one process per case.
2. **Plumes within 2 km of the plume grid's top.**  When `kkmax + deltak`
   reaches `kmt`, the advection reads `RHO(kmt+1)` and neighbours that no
   routine set for this column (first-call zeros here, the previous
   column's values in WRF).  The cases keep every plume below 13.5 km with
   `kmt = 199`; such a plume needs a column top below ~20 km or a top above
   ~18 km.
3. **A plume top at or above the column's highest w level.**
   `set_flam_vert` then finds no level above it (`k = nkp`) and `plumerise`
   reads `zzcon(k2+1)` past the column (or past `nkp`) and writes
   `eburn_out` past `m1`.  The port reads the first-call zeros up to
   `nkp + 1` and refuses (`RuntimeError`, status 3) when `k2 >= m1`, the
   first value at which `dz_flam` reads `zzcon(m1+1)`, on the CPU and the
   GPU alike; the
   `htint` and `get_env_condition` STOPs are refused the same way (status
   1, 2).  Not in the fixtures.
4. **Dead code for the output, not ported:** `set_flam_vert`'s version 2
   (`W_VMD`, `VMD`, the parabolic mass distribution; `plumerise` uses only
   `k1`, `k2`), `fallpart`'s `VTC` and `CVH`, `TXS`, `get_env_condition`'s
   `DNE`, `printout` (`izprint = 0`), `CONVERT2`, `STD_burnt_area`.
   None writes anything another routine reads.
5. **Another heat-flux table.**  `heat_flux` is `get_fire_properties`'
   DATA statement; the port takes another `(2, 4)` table (the same
   arithmetic on other data).  The oracle covers WRF's table only; another
   table is held CPU against GPU.

### Order of evaluation, the parallel form

The device runs one block of 128 threads per burning column with the
module arrays in shared memory (44 x 203 floats, 35,728 B); level `k`
belongs to thread `(k - 1) % 128` and `__syncthreads()` separates loops
that read a level another thread wrote.  WRF's loops over levels are
elementwise except: the prefix chains (htint's search, the Exner integral,
INITIAL's radius and viscosity) and the plume-top and `wmax` scans, run on
thread 0; `fallpart`'s read of `VTH(k+1)`, `VTI(k+1)` before the loop sets
them; and the microphysics loop, whose one carried dependence is
`QSAT(L-1)` (`WATERBAL(L)` touches level `L` only), so `QSAT`, `EST`, `RHO`
of every level are formed first and `DQSDZ(L)` from `QSAT(L+1)` as the step
left it.  The NumPy reference vectorises over levels at the same places
and runs the microphysics level by level.  The 153,600-word state
comparison is what holds both reorganisations to WRF's sequential loop.

## Deposition and settling (`zhang2001/`, `gocart_settling/`)

Part of `tools/smoke_wrf471_oracle`. Fixtures are in
`tests/data/oracles/smoke/zhang2001/` and `tests/data/oracles/smoke/gocart_settling/`.
They are checked by `tests/test_smoke_deposition_wrf471_parity.py`.

### What is compiled

Both routines are cut verbatim from the pinned WRF v4.7.1 files by `extract.py`.
The cut runs from the `SUBROUTINE` line to the `END SUBROUTINE` line, and
`extract-report.txt` records the line ranges and the sha256 of each cut. Each
cut is wrapped in a module of its own and compiled at `-O0` with
`-ffree-form -ffree-line-length-none`, the same flags as the rest of the oracle.

| fixture | WRF routine | lines | precision |
|---|---|---|---|
| `zhang2001` | `zhang2001_aer_drydepvel_1` | `chem/module_aer_drydep.F:1650-1941` | real(4) |
| `gocart_settling` | `settling` | `chem/module_gocart_settling.F:206-367` | REAL*8 (inputs real(4) as the driver passes them) |

The rest of each module is left out because its drivers `use` the WRF-Chem
framework. That framework is `module_configure`, `module_state_description`
and the MOSAIC, CAM-MAM and GOCART data modules. The two cut routines use
none of it.

### Drivers

- `run_zhang2001.F90` writes seven cases, one for each combination of
  `zhang_optaa` (1, 2), `modesurftype` (Zhang classes, USGS classes, and 0)
  and the moment switches `ido_mom0`/`ido_mom3` ((1,1), (1,0), (0,1), (0,0)).
  Within a case the cells cover:
  - every surface type, plus out-of-range ones (-1..16 for Zhang, -1..27 for USGS);
  - every season, plus out-of-range ones (-1..6);
  - dry and wet surfaces;
  - eight number-mode diameters, from below the 1 nm clamp to above the 1 cm one.

  The other inputs come from short lists chosen by an integer hash of the cell
  index. Each list straddles the routine's "reality check" clamps.
  `airkinvisc`/`freepath` are given (positive) on some cells and computed by the
  routine (zero or negative) on others. Both values are written back after the
  call, as the routine leaves them. That gives 20,352 cells per output field.
- `run_gocart_settling.F90` prepares each column exactly as
  `gocart_settling_driver` does for dust (`module_gocart_settling.F:89-122`):
  - `p_mid = .01*p_phy` as a real(4) product;
  - `p_mid` and `delz` reversed, top level first;
  - `airden`, `tmp` and `tc` in model order;
  - `g = 9.81` and `dyn_visc = 1.5E-5`;
  - `idust = 1`, `iseas = 0`.

  It then calls `settling` on six cases:
  - PM2.5 smoke bins (radius 0.1, 0.35 and 1.25 um, 1400 kg m-3);
  - the five GOCART dust bins (`reff_dust`/`den_dust`), with and without the
    UoC density override (`uoc = 1`);
  - time steps of 0.75 s (`INT(dt) = 0`), 60 s, 300 s, 900.5 s (a truncated
    `INT`) and 3600 s (the 12-substep cap);
  - columns holding negative mixing ratios, which are reset to 1e-32.

### Comparison

Every output word is compared, NumPy reference and device kernel alike. The
result is max ULP 0 for all of them:

| field | words | CPU vs oracle | GPU vs oracle | GPU vs CPU |
|---|---:|---:|---:|---:|
| depvel_0 | 20352 | 0 | 0 | 0 |
| depvel_3 | 20352 | 0 | 0 | 0 |
| resist_s0 | 20352 | 0 | 0 | 0 |
| resist_s3 | 20352 | 0 | 0 | 0 |
| vsettl_0 | 20352 | 0 | 0 | 0 |
| vsettl_3 | 20352 | 0 | 0 | 0 |
| airkinvisc (out) | 20352 | 0 | 0 | 0 |
| freepath (out) | 20352 | 0 | 0 | 0 |
| tc (REAL*8) | 4390 | 0 | 0 | 0 |
| graset (REAL*8) | 118 | 0 | 0 | 0 |
| grasetvel (REAL*8) | 118 | 0 | 0 | 0 |

How each implementation evaluates the libm functions:

- **NumPy reference.** real(4) uses glibc's `expf`/`powf` through
  `woof.core.noahmp_libm`, and `sqrtf` as a correctly rounded `np.sqrt`.
  REAL*8 uses the C library's `exp`/`pow` through `woof.core.host_libm`.
- **Device kernels.** real(4) uses `gfk_exp` and `gfk_pow`
  (`glibc_flt32.cuh`). REAL*8 uses `g64_exp::exp` and `g64_pow::pow`
  (`glibc_flt64.cuh`). They are compiled with the unit's `MODULE_OPTIONS`
  (`woof/smoke/_kernels.py`): no FMA contraction and no flush of tiny values.

### Details that matter for bit equality

- `x**beta` takes the real variable `beta = 2.0`, so it is `powf(x, 2.0)` and
  not a product. `tmpa**2` and `dp_mom**2` are products.
- gfortran folds `log(5.0)` once, correctly rounded, giving 0x1.9c0420p+0.
- `INT(ntdt/dtmax)` is x86-64 `cvttsd2si`, which truncates and returns INT_MIN
  outside the int32 range. Both ports reproduce it. Because of the `MAX(1, .)`
  that follows, an out-of-range quotient settles in one substep, not twelve.
- `REAL(ntdt)/REAL(ndt_settl)` is a real(4) quotient. `4.0/9.0*g0` and
  `2./9.*g0` are real(4) products that are promoted only at the next operand.
- Real(4) literals in REAL*8 expressions (`1.458E-6`, `110.4`, `1.1E-3`,
  `1.257`, `0.4`, `1.1`) enter as their float values.
- `settling` reads `p_mid` and `delz` at `l2 = lmx-l+1`, and that index lands
  back on model level `l`. The ports therefore read every field at the level
  being settled, in model order.

### Held out

- **The sea-salt path** (`iseas = 1`). It uses `rh`, `LOG10` and the Gerber
  growth. A PM2.5 smoke bin takes the dust path, so this path is not ported.
- **The driver's post-processing.** This covers `dustgraset_n += conver*graset*airden(1)*dt`,
  the real(4) store back to `chem` and the `chem_opt = 2/11` bin remapping.
  These steps belong to WRF's chemistry framework, not to the settling routine.
  The port returns the routine's REAL*8 `tc`, `graset` and `grasetvel`.
- **`dyn_visc = 1.5E-5`.** It comes from `phys/module_data_gocart_dust.F:16`
  (v4.7.1, sha256
  `8694100875645b54a58bcff38d462ad3c9fd115261fb39941358466c84b62329`). The
  file is cited only and pinned in `SOURCES.sha256`; nothing compiles it.

### Woof-own pieces (no WRF oracle)

- **`aerodynamic_resistance`.** This is the log law with Businger-Dyer
  `psi_h`, in real(4) with glibc `logf`, `zeta = z_ref/L` held at 1 (the
  top of the fitted stable range); an Obukhov length of 0 is refused.
- **`surface_deposition_tendency`.** This is `-v_d q0/dz0`, with the flux
  `rho0 v_d q0`.

For both, the NumPy and device forms are equal at 0 ULP. Each is also held to
a float64 evaluation of the same formula within a tolerance: rtol 2e-5 for the
resistance and 1e-6 for the tendency.
