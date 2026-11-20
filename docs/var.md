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
  largest value, on both tiers; the fields a case leaves alone are exact.

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

## Not yet

- Radar, GOES, radiances and hybrid 3DEnVar (plan sections 6-11).
- Desroziers diagnostics (plan G4a).
- `cv_options` other than 5.
- WRFDA's `A2C`, `var4d` and cloud control variables; pseudo-observations other than `u`, `v`, `t`, `q`.
- Nested domains.
