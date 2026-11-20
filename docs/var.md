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
- **Platform maths.** Off glibc (macOS), the background fields that pass
  through real powers, `log` and `exp` can differ from WRFDA's by an ulp
  (at most 9e-14 relative).

## Not yet

- Radar, GOES, radiances and hybrid 3DEnVar (plan sections 6-11).
- Desroziers diagnostics (plan G4a).
- `cv_options` other than 5.
- WRFDA's `A2C`, `var4d`, cloud control variables and pseudo-observations.
- Nested domains.
