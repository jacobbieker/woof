# Fitch wind-farm oracle: provenance

The fixture under `woof/data/windfarm_fitch/oracle/` is WRF v4.7.1's own
Fitch et al. (2012) wind-farm drag (`windfarm_opt = 1`), compiled and run by
this directory's `build.sh`.  `tests/test_windfarm_fitch_wrf471_parity.py`
holds the device kernel (`woof/core/kernels/windfarm_fitch.cu`) and the
binary32 CPU mirror (`woof.verify.windfarm_fitch_ref.f32_fitch_column`) to
it at max ULP 0.

## Sources

WRF v4.7.1, tag `v4.7.1`, commit `f52c197ed39d12e087d02c50f412d90d418f6186`,
public domain.  `SOURCES.sha256` pins every file; `fetch_sources.sh` fetches
them from `raw.githubusercontent.com/wrf-model/WRF/v4.7.1` and `build.sh`
refuses any other bytes.

| file | sha256 | use |
|---|---|---|
| `phys/module_wind_fitch.F` | `83247add…60c3` | compiled: lines 52-63 (module state) and 67-334 (`dragforce`, `turbine_area`, `dragcof`) |
| `share/module_model_constants.F` | `5b80377f…7062` | compiled: line 72, `piconst` |
| `phys/module_pbl_driver.F` | `42ff5357…fb` | cited: the `dragforce` call, lines 2053-2063 |
| `Registry/Registry.EM_COMMON` | `6f3ee021…623a` | cited: `windfarm_tke_factor` default 0.25, line 3005 |

`extract.py` cuts the compiled lines byte for byte (each block's sha256 is
checked) into one module, because the file's `init_module_wind_fitch` needs
WRF's map projections, configuration type and MPI layer, none of which the
drag touches.  `run_fitch.F90` fills the module's turbine state where
`init_module_wind_fitch` would fill it from `windturbines-ij.txt` and
`wind-turbine-N.tbl` (including its `cutin`, `cutout`, `stc2` statements,
`module_wind_fitch.F:519-521`), and calls `dragforce` as the PBL driver does.

## Toolchain and flags

* gfortran: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0, through
  `pixi exec -s gfortran -s binutils`; not a project dependency.
* glibc 2.44 (the host's libm supplies `powf` and `asinf`); Linux 6.12 x86_64.
* `-O0 -ffree-form -ffree-line-length-none`, REAL is real(4) (WRF's
  `RWORDSIZE=4`): no `-r8`, no `-fdefault-real-8`.
* At -O0 every `x**2.` and `x**3.` is a call to `powf` (17 call sites) and
  every `asin` a call to `asinf` (6), read from `gfortran -S`; the kernel
  uses `gfk_pow` and `glibc_asinf` at exactly those sites.  No `_ZGV*`
  vector symbol is referenced (the build fails if one is), and a positive
  control proves the check can see one; `libmvec-report.txt` records both.

`fixtures/TOOLCHAIN.txt`, `SOURCES.sha256`, `HARNESS.sha256` (the harness
files) and `FIXTURES.sha256` (every published word) are written by
`build.sh` and checked by the parity test.

## Regenerating

    bash tools/windfarm_fitch_wrf471_oracle/fetch_sources.sh /tmp/wrf471
    pixi exec -s gfortran -s binutils -- \
        bash tools/windfarm_fitch_wrf471_oracle/build.sh /tmp/wrf471 /tmp/fitch-build
    rm -rf woof/data/windfarm_fitch/oracle
    cp -r /tmp/fitch-build/fixtures woof/data/windfarm_fitch/oracle

## Cases

Two cases (`default`: dx 1 km, dt 10 s, factor 0.25; `coarse`: dx 3 km,
dt 18 s, factor 1.0), each eleven columns: one turbine; three of one type;
a mixed cell (four 3 MW at 100 m and two 8 MW at 150 m); a rotor across
twenty-one 8 m levels; a rotor inside one 320 m level; below cut-in; above
cut-out; rated wind on the thrust taper; no turbine; a mixed cell of three
types at three hub heights; a uniform 8 m/s column.  DU, DV and QKE start
from nonzero PBL tendencies, as RUBLTEN, RVBLTEN and QKE do in WRF; POWER
starts at -1 and WRF zeroes it.  The turbine list interleaves the cells; a
cell's identical turbines are consecutive, and the layout groups them.

## Held out

* **Hub below the first mass level.**  `module_wind_fitch.F:163-164` scales
  the first level's speed by `hubheight/z1` with `z1` never assigned on that
  path, so WRF's value is whatever the stack held.  The port takes the first
  mass level's height.  No fixture column reaches it, which
  `test_fixture_reaches_every_case_and_none_of_the_held_out_paths` pins.
* **Hub above the top mass level.**  WRF leaves `kbot`/`ktop` unassigned;
  the port takes the top level's speed.  Unreached, pinned by the same test.
* **Clipping a rotor edge into its own layer** (`z1 > D`, `z2 < 0`), which
  WRF does not do and where it would take the square root of a negative
  number.  Unreached by any finite column the level search admits.
* **`overlap` and `thrust`.**  Diagnostics WRF does not form; they have no
  oracle word.  They are held GPU against the binary32 CPU mirror at max
  ULP 0 instead.

The float64 mirror `woof.verify.windfarm_fitch_ref.np_fitch_column` is a
secondary reference: it is held to the oracle within float32 rounding
(2e-5 of each field's scale), not word for word.
