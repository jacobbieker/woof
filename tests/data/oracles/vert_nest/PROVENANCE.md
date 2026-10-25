# Vertical grid nesting oracle: provenance

The fixture under `tests/data/oracles/vert_nest/` comes from WRF v4.7.1's
own vertical-grid-nesting statements (`vert_refine_method = 2`), compiled
and run by this directory's `build.sh`.
`tests/test_nest_vertical_wrf471_parity.py` holds two implementations of
`woof/core/nest_vertical.py` to it at max ULP 0, raw words equal:

- the device kernels in `woof/core/kernels/nest_vertical.cu`;
- the binary32 CPU twin, `VerticalNestPlan.interp_cpu` and
  `rebalance_cpu`.

## Sources

The sources are WRF v4.7.1, tag `v4.7.1`, commit
`f52c197ed39d12e087d02c50f412d90d418f6186`, which is public domain.
`SOURCES.sha256` pins every file. `fetch_sources.sh` fetches them from
`raw.githubusercontent.com/wrf-model/WRF/v4.7.1`, and `build.sh` refuses any
other bytes.

| file | lines | use |
|---|---|---|
| `share/interp_fcn.F` | 1910-2152 | compiled: `vert_interp_vert_nesting`, `_w`, `_1d`, `inter_wrf_copy`, `coeff_mon_wrf_copy` |
| `dyn_em/nest_init_utils.F` | 1033-1184 | compiled: `compute_vcoord_1d_coeffs` (both ladders' c1..c4) |
| `dyn_em/nest_init_utils.F` | 388-409 | compiled: dn, rdn, fnp, fnm and cf1..cfn1 of an arbitrary ladder |
| `external/RSL_LITE/force_domain_em_part2.F` | 73-109 | compiled: the reference log-pressure heights |
| `external/RSL_LITE/force_domain_em_part2.F` | 124-303 | compiled: the vertical-interpolation call site and the rebalance of t_2, QV and ph_2 |
| `external/RSL_LITE/interp_domain_em_part2.F` | -- | cited: the same interpolation at a nest's initialization |
| `share/module_model_constants.F` | -- | cited: the constants the driver declares as PARAMETERs |

`extract.py` cuts each block byte for byte and checks the sha256 of each
block. It does this because the blocks sit inside routines that need WRF's
whole domain type, communication layer and generated includes.
`run_vert_nest.F90` expands them inside subroutines whose dummy arguments
carry WRF's names, so the statements paste unmodified.
`nest_forcedown_interp_vert.inc` is the include WRF's registry tool writes
there (`tools/gen_interp.c:540-640`), restricted to the fields the oracle
carries: u, w, ph, t and QV.

## Cases

Each case builds two ladders, `znw = 1 - x**p` with `x = (k-1)/nz`:

| case | parent | child | hypsometric_opt |
|---|---|---|---|
| `refine_h2` | 40 levels, p = 1.4 | 79 levels, p = 2.2 | 2 |
| `refine_h1` | 40 levels, p = 1.4 | 79 levels, p = 2.2 | 1 |
| `coarsen_h2` | 60 levels | 30 levels | 2 |
| `same_h2` | 40 levels | the same ladder | 2 |

The fields are smooth synthetic profiles, WRF-coupled
(`x*(c1*mut + c2)`), on a 6 x 5 memory grid of parent columns. Coarsening
is a ladder WRF's own driver cannot allocate, because its intermediate grid
carries the nest's level count. The harness sizes the arrays for both, and
the routines are WRF's either way.

## Toolchain and flags

- gfortran: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0, through
  `pixi exec -s gfortran -s binutils`; not a project dependency.
- glibc 2.44: the host's libm supplies `logf`, `powf` and `expf`.
- Linux 6.12 x86_64.
- `-O0 -ffree-form -ffree-line-length-none`. REAL is real(4), WRF's
  `RWORDSIZE=4`.
- `build.sh` fails if an -O0 object references a libmvec `_ZGV` symbol. It
  proves the check can see one with a positive control at -Ofast.

## Divergences

There is one, and it is refused rather than reproduced. Where
`inter_wrf_copy` finds no interval for a child point, WRF leaves that value
undefined. This can only happen if the two reference columns do not share
their end points. `VerticalNestPlan` refuses such a pair of ladders by name.
