# MPAS-Ocean v8.4.2 CVMix vertical mixing oracle (RKIND 8)

Written by `tools/mpas_ocean_oracle/build.sh` (with `CVMIX_SOURCE`) and
`make_fixture.py --piece cvmix` on pb (Linux 6.12 x86_64, glibc 2.44),
2026-11-22.

- MPAS source: github.com/MPAS-Dev/MPAS-Model tag v8.4.2, commit
  `f34984b2c89353cf31c235b08d3b4acd96b37bb3`, byte-unmodified.
  `tools/mpas_ocean_oracle/SOURCES.sha256` covers
  `src/core_ocean/shared/mpas_ocn_vmix_cvmix.F` along with the other vmix modules
  and the equation of state.
- CVMix source: github.com/CVMix/CVMix-src tag v0.84-beta, commit
  `d83f582714e7f0f98d20efd8fac8fab01fa3bfe6`. This is the tree MPAS-Model
  v8.4.2's `get_cvmix.sh` clones. The ten modules of `src/shared` are
  byte-unmodified and checked against `tools/mpas_ocean_oracle/CVMIX_SOURCES.sha256`.
  build.sh refuses to build if any digest differs. Nothing from CVMix is
  committed: the build downloads it.
- These are compiled verbatim in the build's `cvmix/` directory, together with
  `mpas_ocn_vmix_cvmix.F`, the other vmix modules and the real equation of
  state. There is no stand-in for CVMix in this build. `stubs_vmix.F90` serves
  only the `run_vmix_*` drivers of `tests/data/oracles/mpas_ocean_vmix`.
- Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0 (`mpas-oracle` pixi
  environment), `-O0 -ffp-contract=off`, RKIND 8. No object contains an FMA
  instruction. Real powers call the host glibc's `pow`. `x**2` is `x*x`, while
  `x**3` and runtime integer powers go through libgcc's `__powidf2`.
- RKIND 8 only: `mpas_ocn_vmix_cvmix.F` does not compile with
  `-DSINGLE_PRECISION`. CVMix is `real(8)`, and gfortran reports a REAL(4) to
  REAL(8) type mismatch with no specific for `cvmix_put` and
  `cvmix_init_bkgnd`. `woof.hex.mpas_ocean.cvmix_v842.check_config` refuses
  CVMix at RKIND 4 by name with that reason.
- Driver: `run_vmix.F90` (stream: `oracle_stream.F90`), linked as
  `run_vmix_cvmix_r8`. It gives `ocn_vmix_init` a domain whose block carries
  the config pool and the mesh dimensions, because `ocn_vmix_cvmix_init` reads
  them. It then calls `ocn_vmix_implicit(timeLevelIn=1)`, which runs
  `ocn_vmix_coefs_cvmix_build`, and writes every field back.

## Licence

CVMix v0.84-beta ships no licence file. CVMix-src added LGPL-2.1 and LGPL-3
texts in April 2021 (`1fa42e0`, `5564f97`). In March 2025 it removed them and
was relicensed under the Apache License 2.0 with the LLVM exception
(`2183465`, "Remove LGPL, Add Apache2.0").

The GPU transcription (`woof/hex/mpas_ocean/cvmix_v842.cu`) is taken under
that Apache grant, as CVMix-src is licensed at commit
`fce422195a0c58f15a55946b5ed517ba4365e232`
(`licenses/LICENSE-CVMix-Apache-2.0-LLVM.txt`, NOTICE section 18). No
LGPL-licensed text is in the distribution.

Each transcribed routine was compared between v0.84-beta and `fce42219`,
ignoring comments and whitespace:

| result | routines |
|---|---|
| identical | `cvmix_kpp_compute_kOBL_depth`, `cvmix_kpp_compute_enhanced_diff`, `cvmix_kpp_compute_shape_function_coeffs`, `cvmix_math_cubic_root_find`, `cvmix_math_poly_interp` (renamed under a generic interface), `cvmix_coeffs_shear_wrap`, `cvmix_coeffs_conv_wrap` and `_low`, `cvmix_put_conv`/`cvmix_get_conv_real`, `cvmix_init_bkgnd_scalar`, `cvmix_update_wrap`, `cvmix_put_int` |
| differ, same result on MPAS's path | `cvmix_init_kpp`, `cvmix_kpp_compute_bulk_Richardson`, `cvmix_kpp_compute_turbulent_scales_0d`/`_1d_sigma`/`_1d_OBL`, `cvmix_kpp_compute_unresolved_shear` (master's default case is v0.84's formula), `compute_phi_inv`, `cvmix_put_kpp`/`cvmix_get_kpp_real`, `cvmix_put_real` (renamed keys only), `cvmix_math_evaluate_cubic` (its `fprime` is unused), `cvmix_coeffs_shear_low` and `cvmix_init_shear` (`Prandtl_shear` defaults to 1, `PP_kappa_b` to 0), `cvmix_init_conv` |
| v0.84-only logic, not ported | `cvmix_kpp_compute_OBL_depth_low` with `config_cvmix_kpp_EkmanOBL = .true.`; `cvmix_coeffs_kpp_low` with `config_cvmix_kpp_use_enhanced_diff = .false.` |

The branches master adds (Stokes MOST, Langmuir, XI, the CVt2/ER/LF17
parameters) are off in every configuration MPAS-Ocean v8.4.2 builds. So are
the v0.84 branches master drops (`llangmuirEF`, `lenhanced_entr`), because MPAS
never passes those flags.

The two v0.84-only behaviours:

1. **`config_cvmix_kpp_EkmanOBL = .true.`** v0.84 caps the boundary layer at
   the Ekman depth `0.7 u*/|f|` whenever `f` is non-zero, whatever the
   stability. Master caps it only under stable forcing. v0.84 also tests
   `lMonOb` in place of `lEkman` before using the Coriolis parameter; that has
   no effect, because MPAS always passes `f`.
2. **`config_cvmix_kpp_use_enhanced_diff = .false.`** When `ktup == kwup`,
   v0.84 zeroes the interface just below the boundary layer. Master restores
   the old diffusivity there.

Both are refused by name (`cvmix_v842.check_config`) with this reason. They
are queued as possible opt-ins.

## Cases

| case | nCells | nVertLevels | what it covers |
|---|---|---|---|
| `kpp_simple_shapes` | 48 | 24 | background, PP shear, convection, KPP (SimpleShapes, quadratic interpolation, BLD smoothing, two Ri smoothing loops) |
| `kpp_nonlocal_implicit_budgets` | 44 | 20 | `config_cvmix_kpp_nonlocal_with_implicit_mix`: the non-local flux in the implicit tracer solve; the budget |
| `kpp_match_both_cubic_kpp_shear_monob` | 48 | 24 | MatchBoth, cubic interpolation, KPP shear without background, the Monin-Obukhov limit |
| `kpp_match_both_quadratic` | 40 | 18 | MatchBoth, quadratic interpolation, the non-local flux in the solve |
| `kpp_linear_unsmoothed` | 40 | 16 | linear interpolation, no BLD or Ri smoothing, Ri_crit 0.3, surface layer 0.2 |
| `kpp_fixed_layer_ice` | 40 | 16 | the fixed 45 m boundary layer, sea ice |
| `kpp_ice_minimum` | 44 | 20 | the 150 m minimum under sea ice (`iceFraction > 0.15`), unsmoothed |
| `convection_negative_trigger` | 32 | 12 | convection alone, with a negative BVF trigger |
| `convection_shear_no_kpp` | 36 | 14 | convection from the top (no KPP), PP shear, background |
| `background_only` | 24 | 8 | the background coefficients alone |
| `kpp_shallow_calm` | 48 | 12 | a third of the columns one or two levels deep, a third of `u*` zero, Monin-Obukhov |
| `kpp_const_rich` | 40 | 16 | CVMix on top of the constant and Richardson coefficients, the budget |
| `kpp_halos` | 48 | 20 | distinct `nCellsArray`/`nEdgesArray` extents |
| `kpp_tidal_ddiff_flags` | 32 | 12 | the tidal and double-diffusion flags, which v8.4.2 accepts and ignores |
| `kpp_only_cubic` | 40 | 30 | KPP with no other CVMix term, cubic interpolation, five Ri smoothing loops |

The meshes have random connectivity on one block. `cellsOnCell` pads about
15% of the neighbours past the first. Potential density is stable below a
random mixed layer, with some inversions near the top. The Brunt-Vaisala
frequency takes both signs. Surface buoyancy forcing is mostly destabilising.
The Coriolis parameter comes from a cubic of the latitude, not the host libm,
so the inputs are the same bytes on every host (`inputs.sha256`).

## Grading

`tests/test_mpas_ocean_cvmix_parity.py` and `woof hex ocean verify`
(`cvmix_kpp_r8`) run `woof.hex.mpas_ocean.cvmix_v842` through
`vmix_v842.vmix_implicit`. Every word must be bit-identical, including the
implicitly mixed velocities and tracers.

The test also checks the following:

- the fixture reaches both kOBL fractions (`.25` and `.75`), the
  bottom, half-first-layer and under-ice limits, stable, unstable and calm
  forcing;
- the non-local flux is non-zero only under unstable forcing;
- a mutated kernel is caught;
- the unported options are refused by name.

Quirks of v8.4.2 kept word for word are listed in
`woof/hex/mpas_ocean/cvmix_v842.py`'s docstring.
