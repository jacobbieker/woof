# `woof var`: 3D-Var analysis (experimental)

`woof var` runs the woof port of WRFDA 4.7.1's regional 3D-Var. It takes
WRFDA's background-error file, OBSPROC's observations and a WRF first guess.
It prints WRFDA's minimiser log and writes the analysis as a
[`wrf-analysis-bundle/1`](analysis-bundle-wrf.md). A forecast starts from
the bundle with `woof run --analysis BUNDLE`.

```
woof var --be be.dat --obs ob.ascii --first-guess wrfinput_d01 --out BUNDLE \
    [--tier reference|oracle|production] [--ntmax 50] [--eps 0.01] [--max-ext-its 1]
```

The port lives in `woof.da.var`. Only this command and its tests import it
(`tests/test_var_import_graph.py`), so no forecast path changes. The plan is
[docs/dev/WRFDA-3DVAR-PLAN.md](dev/WRFDA-3DVAR-PLAN.md); provenance is in
`PROVENANCE.md` under "EXPERIMENTAL WRFDA 3D-Var".

## What it runs

1. **`da_transfer_wrftoxb`** (`woof.da.var.analysis.wrftoxb`). The background
   comes from the first guess. WRFDA recomputes the base state (`mub`, `phb`,
   `pb`) from `P00`/`T00`/`TLP`/`TISO`. Roughness comes from `LANDUSE.TBL`
   (woof's packaged copy, or `--landuse-table`).
2. **Observations.** OBSPROC's `ob.ascii` is read within ±`--window-minutes`
   (default 60) of the first guess's time. Innovations and the background
   check follow WRFDA:
   - surface, sounding, aircraft, wind profiler, AMV and scatterometer;
   - GNSS PW and refractivity;
   - SATEM thickness.
3. **Cost and minimiser.** The cost is `J = Jb + Jo` with the cv5 control
   variable transform `U` from `be.dat`. WRFDA's `da_minimise_cg` minimises
   it:
   - `--ntmax` sets the iteration limit, `--eps` the stopping threshold and
     `--max-ext-its` the number of outer loops;
   - `orthonorm_gradient` is on by default; `--no-orthonorm-gradient` turns
     it off;
   - the log lines are WRFDA's (`minimize_cg  1  k  J  |g|  step`).
4. **`da_transfer_xatowrf`.** The increment becomes WRF's
   `U V T QVAPOR MU P PH PSFC`:
   - Guo's tangent-linear pressure;
   - hydrostatic `PH` (`HYPSOMETRIC_OPT` 1 or 2);
   - C-grid winds with WRFDA's boundary extrapolation;
   - `--check-rh 1` for WRFDA's `check_rh = 1`.
5. **The bundle.** The analysis file is the first guess with those eight
   fields replaced (`da_update_firstguess`), written through
   `woof.io.analysis_bundle`.
   - The bundle is `kind = control`, with `recipe_id woof-var-3dvar` and the
     door's settings digested into `recipe_sha256`.
   - Each outer loop's iteration count and final cost go into
     `notes.var`.
   - `--ph balance` lets the bundle writer rebuild `PH` instead of keeping
     WRFDA's.
   - `--iau-window-s` records an IAU window.

## Hybrid 3DEnVar (opt-in)

WRFDA's hybrid adds the alpha control variable to the cost. Jb is weighted
by `jb_factor = je/(je-1)` and the ensemble term Je by `je_factor`. The
increment is `U_s v_s + sum_n alpha_n o x'_n`. With an
ensemble, `woof var` runs it (the Python door is
`analyse(..., ensemble=, hybrid=, vertloc=)`):

```
woof var --be be.dat --obs ob.ascii --first-guess fg --out bundle \
    --ensemble-member m01/wrfout --ensemble-member m02/wrfout ... --je-factor 2
```

- **Members.**
  - `--ensemble-member` (repeated) takes the members' history files, valid
    at the analysis time. The perturbations are `gen_be_ep2`'s
    (`woof.da.var.hybrid.ep_from_wrf`): `u` and `v` at mass points, `t`
    from theta and pressure, `q` as specific humidity, and `PSFC`, each
    about the members' running mean.
  - `--ep-dir` with `--ensdim-alpha` (and `--ep-format` 1 or 11) instead
    reads WRFDA `ep` files.
  - Hydrometeor perturbations are not used.
- **`--je-factor` is required.** WRFDA's default of 1.0 means a pure
  ensemble `B` (`jb_factor` -999, no static term), so woof asks rather than
  assuming it. 2 weights the two equally.
- **Localisation.**
  - `--alpha-corr-scale` is the horizontal scale in km (default 200).
  - `--alpha-vertloc-opt` selects the vertical localisation:
    - 2 (default): WRFDA's log-p function from `znw` and `ptop`;
    - 1: a `be.vertloc.dat` given with `--be-vertloc`;
    - 0: none.
  - `--max-vert-var-alpha` sets the share of eigenvalues kept (default 99%).
- **The bundle.** The hybrid settings are digested into `recipe_sha256` and
  listed under `notes.var.hybrid`.

Checked against WRFDA 4.7.1 on the tutorial case, with ten members of
WRFDA's gen_be sample forecasts, `je_factor = 2` and `alpha_corr_scale =
200` (`tests/test_var_hybrid_o6.py`, `tests/test_var_hybrid_cli.py`):

- `ep_from_wrf` is `gen_be_ep2`'s files. `u`, `v`, `q` and `ps` match word
  for word; `t` is within 2.8e-13 K, from `**kappa`.
- The 50 iterations follow WRFDA's log to 3.5e-14 relative.
- The analysis increments against WRFDA's `wrfvar_output` have max
  |difference| at most 4.7e-5 of the RMS (PSFC; the other fields at most
  7.2e-6) and correlation of at least 0.9999999996.

## Tiers

| `--tier` | arithmetic | where |
|---|---|---|
| `reference` (default) | NumPy binary64 | CPU |
| `oracle` | binary64, WRFDA's serial dot product | CUDA or Metal |
| `production` | binary32, fixed-tree dot product | CUDA or Metal |

## Speed

Minimisation only (no I/O, no innovation step), the tutorial's namelist
(50 iterations at most, `orthonorm_gradient`), measured 2026-11-20. The
A6000 was shared with two or three other jobs, so its times are upper
bounds and vary with the load.

| case | tier | M3 Max (Metal) | A6000 (CUDA) |
|---|---|---|---|
| tutorial, 90x60x41, 41,857 observations, 29 iterations | `reference` | 10.5 s | 13.4 s |
| | `oracle` | 81 s | 29 s |
| | `production` | 1.4 s | 2.3-2.4 s |
| synthetic 200x200x50, 101,000 obs values, 50 iterations | `production` | 2.9-3.1 s | 3.2-4.5 s |

The plan's targets (section 4.12) are 20 s on the M3 Max and 5 s on the
A6000 for the 200x200x50 case. `WOOF_VAR_PERF=1 pytest
tests/test_var_perf.py` runs that gate; `tests/_var_perf_case.py` builds the
case (a random cv5 `be.dat` through the writer, a smooth first guess and
sound, synop, airep and geoamv innovations). Serial WRFDA takes 7.9 s for
the whole tutorial run.

- `H'` and `H'^T` run on the device (`kernels/var_obs.cu`). `H'` is a
  gather. `H'^T` is WRFDA's scatter written as a gather: one thread per
  model point adds its contributions in the serial loops' order, so there
  are no atomics. The words are the host operators', in both tiers.
  `gpspw` and `satem` keep their host code; they feed the same gather.
- Per iteration at 200x200x50, production: `U` 10-13 ms, `U^T` 12-18 ms,
  `H'` under 1 ms, `H'^T` about 4 ms. The re-orthonormalisation's dot
  products stay on the device, so an iteration waits on the host three
  times rather than once per stored gradient.
- The `oracle` tier keeps `da_dot`'s serial order: a block forms the
  products and one thread adds them. Its time is recorded, not gated.

## Single-observation tests

`woof.da.var.pseudo_wrfda.pseudo_obs` is WRFDA's pseudo observation (`num_pseudo = 1`, `pseudo_var`
`u`, `v`, `t`, `p` or `q`): one report at a fractional grid position and level with a given innovation and
error, and nothing else. `analyse` takes it in place of an `ob.ascii`. Tested against WRFDA on the tutorial
(`tests/test_var_pseudo.py`, `WOOF_WRFDA471_PSEUDO_DIR`; position 40.4, 30.6, level 15.3; `u`, `v`, `t`
with value and error 1, `q` with 0.001 and 0.0005):

- `J_0`, `|g_0|`, the final `J` and the one iteration match WRFDA's log (reference tier 1e-12 relative; the
  binary32 production tier 1e-6);
- all eight WRF fields of the analysis are `wrfvar_output`'s float32 words to at most 0.5 ulp of the field's
  largest value, on both tiers (the limit is 1 ulp of the field's maximum); where WRFDA's increment is
  non-zero ours is non-zero with correlation at least 0.9999.

## Checked against WRFDA 4.7.1

The reference is a serial GNU WRFDA 4.7.1 on the WRFDA tutorial case:
90x60x41, 41,857 observations, the tutorial's namelist. Every check needs
`WOOF_WRFDA471_DIR`; no WRFDA data is in the repository.

- **O6, the minimisation** (`tests/test_var_o6.py`):
  - 29 iterations, as in WRFDA;
  - every logged `J` and `|g|` within 1e-14 relative of WRFDA's log;
  - `xhat` within 6e-14 of WRFDA's.
  - The oracle tier gives the reference's words on Metal and CUDA.
  - The production tier ends within 7e-9 of WRFDA's final `J`. Its words
    hash to one digest under fast and strict numerics, on Metal and on CUDA.
- **O7, the analysis** (`tests/test_var_analysis.py`). Each field's
  increment (analysis minus first guess) is compared with WRFDA's
  (`wrfvar_output` minus first guess):
  - the largest difference is at most 1.3e-4 of the increment's RMS for
    `PSFC`, and at most 1.2e-5 for the other seven fields;
  - correlation is at least 0.999999997.
  - Rounded to the file's float32, all eight fields are `wrfvar_output`'s
    words.
  - The bundle's `analysis.nc` carries those words, and the
    `woof run --analysis` door accepts the bundle.
- **A forecast from the analysis** (`tests/test_var_forecast.py`). A 6 h woof forecast
  (the tutorial's physics, direct insertion) from the reference-tier analysis and one from WRFDA's
  `wrfvar_output` give byte-identical `wrfout` files at the three history times (12, 15, 18 UTC), and a run without the analysis differs from both. WRFDA's
  analysis enters as the first guess with its eight updated fields replaced, because `wrfvar_output`'s `TITLE`
  is not the audited pre-hybrid writer's and the bundle writer refuses its `HYBRID_OPT = -1`.
  A one-off measurement on Metal (12 h, not a test): the binary32 production tier differs at float32 noise at the
  start and the difference grows with the tutorial's convection: 3% of the no-assimilation difference at 6 h
  and 13-30% at 12 h. The forecast is
  deterministic, so that is the model's sensitivity.
- **Known residual in the background.** `wrftoxb` from the first guess
  is not WRFDA's `xb` word for word. Most fields are exact, but some differ
  by a few ulps, at most 9e-14 relative:
  - `p` differs at 22% of points; `t`, `es`, `qs` and `ref` follow from it;
  - `rho`, `psac` and `h` differ at about 0.1% of points.
  - The differences are the same on glibc (Linux) and macOS. They do not
    come from the maths library: glibc's scalar `pow`, `log` and `exp` per
    element give the same words. WRFDA's build does not use FMA. The source
    is not yet found.
  - The minimiser's `J` and `|g|` are within 1.5e-15 of WRFDA's. The float32
    analysis is still WRFDA's words.

## Desroziers diagnostics

`woof.da.var.desroziers` (plan gate G4a, tests in `tests/test_var_desroziers.py`). `run` solves the
cost twice, once as is and once with Gaussian noise `sigma z` on the observations, and gives:

- WRFDA's randomised-trace table (`da_tune_obs_desroziers`): per type and variable the trace
  `Tr(HK) = sum (y_p - y) z / sigma`, the expected `Jo = (n - Tr(HK)) / 2`, the actual `Jo` and the error
  factor `sqrt(Jo_actual / Jo_expected)`, with the totals. `format_report` prints it as the tool does.
- The classic O-B / O-A / A-B statistics (`classic`): `<(O-A)(O-B)>` against the specified error variance and
  `<(A-B)(O-B)>` for `H B H^T`.

On the tutorial (the noise is read from WRFDA's `rand_obs_error`, so both use one draw), against the tool
run on WRFDA's own runs (`tools/wrfda_oracle/run_desroziers.sh`):

- all 210 numbers of the table and the totals agree with the tool to 3e-5 absolute plus 1e-6 relative (it
  prints five decimals and sums in binary32; at most 2e-5 apart measured);
- `H'(xhat)` agrees with WRFDA's `unpert_obs` to the file's seven digits (4e-7 of the largest value);
- the classic statistics agree with those computed from WRFDA's own `gts_omb_oma` O-B and O-A, within the
  file's printed precision.

On the tutorial the specified errors are too small for the wind, too large for scatterometer, satellite
thickness and GNSS refractivity: the error factors range from 0.34 (satem) to 1.95 (ships v), and the total
`Jo` factor is 0.978. These tune `R` only; `Jb` has the tool's own `NaN` factor here because 4.7.1 does not
print `Jb factor used`.

## Not yet

- Radar and GOES (plan sections 6-11); hydrometeor perturbations in the hybrid
  (`alpha_hydrometeors`).
- `cv_options` other than 5.
- WRFDA's `A2C`, `var4d` and cloud control variables; pseudo-observations other than `u`, `v`, `t`, `q`.
- Nested domains.
