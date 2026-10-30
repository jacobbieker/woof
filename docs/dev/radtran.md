# woof.radtran: spectral radiative transfer on the GPU

woof.radtran computes spectral irradiances and radiances on atmospheric
columns. It uses REPTRAN representative wavelengths, a discrete-ordinate
solver (DISORT method) or a two-stream solver, Rayleigh scattering, gas
absorption, and water and ice clouds. Each GPU thread solves one
(column, representative wavelength) problem. It runs standalone
(`woof.radtran.driver.compute`) and as an output-only diagnostic on WRF domains
(RunConfig `radtran_*`) and on the MPAS suite route (`&radtran`).

## Clean-room statement

woof is Apache-2.0. libRadtran, including its C DISORT and twostr, is GPL-2.
**woof's implementation is clean-room.** No libRadtran, DISORT, cDISORT or
twostr source text was opened, read, searched or copied. Every component was
written from the published papers listed below and from our own derivations.

libRadtran is used in only two ways:

* **Its data files**: AFGL atmospheres, REPTRAN band and lookup tables, cross
  sections, and the Hu & Stamnes and Fu tables. woof reads them at run time
  from the user's own install (`woof/radtran/data.py`: `$WOOF_LIBRADTRAN_DATA`,
  the pixi `libradtran` environment, or `$CONDA_PREFIX`). They are never
  copied into the repository. The oracle fixtures store only uvspec's outputs
  and a digest of the inputs, not AFGL profiles.
* **uvspec as a black-box oracle**: `tools/uvspec_oracle` runs it as a
  separate process (`pixi run -e libradtran uvspec`), writes its input files
  and parses its standard output. Where uvspec's conventions mattered (the
  REPTRAN lookup interpolation, layer column integration, the delta-scaling
  conventions, the direct/diffuse split it reports), they were established by
  probing that binary with crafted inputs and its `verbose` output, never by
  reading its code.

| Component | File | Published source |
|---|---|---|
| Discrete ordinates: Fourier azimuth expansion, double-Gauss quadrature, beam particular solution, scaled-exponential boundary-value problem, intensities by source-function integration | `woof/core/kernels/radtran_disort.cu`, `woof/radtran/reference.py` | Stamnes, Tsay, Wiscombe & Jayaweera (1988) Appl. Opt. 27, 2502 |
| Eigenvalue reduction, (α−β)(α+β)v = k²v; symmetrised here via the W^½M^½ similarity, then Cholesky + cyclic Jacobi | same | Stamnes & Swanson (1981) J. Atmos. Sci. 38, 387 |
| delta-M scaling | same | Wiscombe (1977) J. Atmos. Sci. 34, 1408 |
| Pseudo-spherical beam: slant optical depth through homogeneous spherical shells, average layer secant | same | Dahlback & Stamnes (1991) Planet. Space Sci. 39, 671 |
| TMS radiance correction | same + host `ptrue` | Nakajima & Tanaka (1988) JQSRT 40, 51 |
| Two-stream as the order-1 discrete-ordinate problem (μ₁ = 1/√3, w = 1), delta-Eddington f = g² | same | Kylling, Stamnes & Tsay (1995) J. Atmos. Chem. 21, 115; Joseph, Wiscombe & Weinman (1976) |
| Rayleigh cross section, King factor, depolarisation | `rayleigh.py` | Bodhaine, Wood, Dutton & Slusser (1999), eqs. 18–23 |
| REPTRAN bands, weights, extraterrestrial flux | `reptran.py` | Gasteiger et al. (2014) JQSRT 148, 99 |
| Water clouds | `clouds.py` | Hu & Stamnes (1993) J. Climate 6, 728 |
| Ice clouds | `clouds.py` | Fu (1996) J. Climate 9, 2058 |
| Mie scattering (for OPAC, not yet wired) | `mie.py` | Bohren & Huffman (1983) ch. 4; Wiscombe (1980) Appl. Opt. 19, 1505 |
| Sun–Earth distance | `driver.py` | Spencer (1971) Search 2, 172 |
| Closed-form two-stream check | `tests/test_radtran_reference.py` | Meador & Weaver (1980) J. Atmos. Sci. 37, 630 |

## Design

**Optics (host, NumPy; `optics.py`, `atmosphere.py`, `clouds.py`).** Levels are
ordered top first.

* **Layer gas column**: inside each layer the air density varies exponentially
  and each gas's mixing ratio varies linearly. For air this is the log-mean
  density times the thickness.
* **Layer cross section**: the mean of the cross sections at the layer's two
  levels.
* **REPTRAN lookup**: linear in pressure; linear in the temperature offset from
  each pressure node's own reference temperature; linear in the H₂O mixing
  ratio for H₂O. The stored values ×1e-16 are cm².
* **Cross-section files** (used where the lookup has none):
  * O₃: Molina (`crs_o3_all.dat`), quadratic through 226/263/298 K;
  * NO₂: Burrows GOME;
  * O₄: Greenblatt, applied to the column of n(O₂)².
* **Rayleigh**: CO₂ is taken at the lowest level.

**Solver (`radtran_disort.cu`).** Arithmetic is binary64 throughout, with
`exp`/`cos` from woof's word-exact ports (`glibc_flt64.cuh`), IEEE `sqrt` and
division, and no FMA contraction (`_NO_FMAD_MODULES`). The same source
compiles under NVRTC, through the Metal translator, and as host C++
(`woof/radtran/_host.py`).

* Per Fourier mode m, it builds each layer's eigen-solution and particular
  solution, then solves the (2N·L) boundary-value system with banded LU
  (partial pivoting) in a global workspace.
* Fluxes and radiances are produced at any subset of levels.
* Conventions, matching uvspec:
  * `edir` is the unscaled direct beam, and `edn` gets the delta-M forward
    peak back;
  * the mean intensities keep the delta-M split;
  * single-scattering albedo is capped at 1 − 1e-9.

**Bands (`driver.py`).** For band b,
E_b = d · Σᵢ wᵢ E0ᵢ Eᵢ,
where wᵢ are the REPTRAN weights, E0ᵢ is REPTRAN's `extra`, and d is the
Spencer distance factor. Output is in mW m⁻² nm⁻¹; the model fields are in
W m⁻² nm⁻¹.

**Model columns (`model.py`).**

* **Levels and layers**: the RT levels are the model's interfaces and the RT
  layers its mass layers.
* **Interface T and qv**: means of the adjacent mass levels; at the ends, the
  adjacent mass level's value.
* **Above the model top**: U.S. Standard (afglus) levels more than 1 km above
  the highest column top are spliced on. Gases the model does not carry
  (O₃, CH₄, N₂O, CO, NO₂, O₂, N₂) come from afglus mixing ratios. CO₂ is
  400 ppm.
* **Clouds**: qc and qi × dry density give LWC and IWC. Effective radii come
  from the microphysics where available, otherwise 10 µm (water) and 30 µm
  (ice), clipped to each parameterisation's range.
* **MPAS interface pressures**: log-linear interpolation in height between mass
  levels, and hydrostatic extrapolation at the surface and model top.

**WRF wiring.** The `radtran_*` RunConfig keys are keyword-only, appended last,
and off by default.

* `PhysicsDriver.compute` runs the diagnostic on the first step and on each
  step that crosses a multiple of `radtran_interval_s`. It uses the step's own
  solar zenith angle, from `swint.calendar_scalars` and
  `solar_geometry_device`.
* History fields, on the `rt_wavelength` axis (one entry per *requested*
  wavelength):
  * `RT_EDIR`, `RT_EDN`, `RT_EUP` (surface) and `RT_EUP_TOA`;
  * `RT_RAD` on (`rt_wavelength`, `rt_direction`): umu > 0 at TOA, umu < 0 at
    the surface;
  * axis variables `RT_WAVELENGTH`, `RT_BAND_MIN`, `RT_BAND_MAX`, `RT_UMU`,
    `RT_PHI`.
* Like other output-only keys, it is excluded from the checkpoint digest,
  echo, identity, preflight, docs and per-domain overrides.

**MPAS wiring.** The `&radtran` group uses quoted lists, since MPAS namelists
refuse multi-value assignments.

* The run gets package-gated `diag_physics` fields `rt_*` on
  `nRadtranWavelengths`/`nRadtranDirections`, in the diagnostics stream.
* They are computed when a stream carrying them is written, so the cadence is
  that stream's output interval.

## Oracle method

`tools/uvspec_oracle/`:

* `cases.py` builds the inputs from the user's libRadtran data.
* `oracle.py` writes uvspec's inputs on exactly woof's levels. CH₄, N₂O, CO and
  N₂ go in as explicit `mol_file`s, so uvspec adds no levels of its own. It
  runs uvspec from a short temporary path, because uvspec truncates data file
  names near 100 characters.
* `generate.py` writes `tests/data/oracles/uvspec/<case>.json` and
  `receipt.json`. The receipt holds the uvspec version (2.0.6-MYSTIC), the
  conda-forge build `rubin-libradtran 2.0.6 py311pl5321h6e06971_1`, the
  platform, and the sha256 of every data file used.

The fixtures were generated on macOS arm64. uvspec on Linux x86-64 reproduces
them to ≤8e-7, its print precision.

`tests/test_radtran_kernel_words*.py` hold the host build, CUDA (RTX A6000) and
Metal to the same committed output words (`tests/data/radtran/kernel_words.npz`),
so CUDA and MLX agree word for word.

## Measured agreement with uvspec 2.0.6 (relative, max over bands and levels)

The tolerances in `tests/test_radtran_uvspec_oracle.py` are about 2× these
measured values. uvspec prints irradiance to 7 significant digits (a quantum of
5e-7) and radiance to 10.

| Case | Irradiances | Note |
|---|---|---|
| Rayleigh, single band, SZA 0/30/60 | ≤3.1e-7 | print precision |
| REPTRAN clear sky 400–405 nm (US), 548–552 nm (tropics) | ≤5.4e-7 | |
| O₃ Chappuis, H₂O 940, O₂ A-band, CO₂/CH₄ 1600 | ≤6.6e-7 | |
| O₃ Huggins 318–322 nm | 8.1e-6 | uvspec interpolates cross-section files at a float32 wavelength. With a float32 wavelength, woof agrees to 1.5e-6 |
| Water cloud (Hu & Stamnes), τ≈50 | 1.4e-5 (diffuse) | Thick-cloud sensitivity to uvspec's binary32 single-scattering albedo: rounding woof's ssa to binary32 moves its result by 1–2e-5 |
| Water cloud at 1600 nm; ice cloud (Fu) at 550/1600 nm | ≤7e-7 | |
| Plane-parallel SZA 85 | ≤6.3e-7 | |
| Pseudo-spherical, diffuse fields, SZA 0/60/85/89 | ≤4e-7 / 4e-7 / 4e-7 / 6.4e-6 | |
| Pseudo-spherical direct beam, SZA 60/85/89 | 1.4e-5 / 2.2e-3 / 2.8e-2 | **Differs.** uvspec's *reported* direct irradiance is not exp(−slant τ) of the beam its own solver deposits: its diffuse fields agree with woof's, and an energy check shows its reported direct beam is inconsistent with them. woof reports the beam its solver uses. No simple geometric model (effective radius, half-layer, exponential sub-layer) reproduced uvspec's number |
| Two-stream, clear sky and thin cloud (τ≈0.5) | ≤3.6e-7 | |
| Two-stream, thick cloud (τ≈50) | 1.7e-3 | **Differs.** The gap grows ~τ² (4e-6 at τ 2.5). woof reproduces the closed-form two-stream (Meador & Weaver) to ≤1e-10, so the departure is uvspec's twostr |
| Two-stream, pseudo-spherical SZA 85 | diffuse 1.3e-6; direct as above | |
| Model column spliced on afglus: clear, H₂O, cloud | 5e-7 / 2.7e-6 / 1.9e-5 | uvspec holds altitudes in float32; thick cloud as above |
| `day_of_year` 180 | 3.7e-7 | Spencer (1971) |

| Radiance case (16 streams) | Agreement | Note |
|---|---|---|
| Rayleigh | 1.4e-7 | |
| Cloud, `disort_intcor off` | 4.7e-7 | |
| Cloud, `disort_intcor moments` | ≤2.4e-6 outside the near-solar downward row, 1.6e-2 inside it | uvspec also applies Nakajima & Tanaka's IMS secondary-scattering correction; woof implements TMS only |
| Pseudo-spherical SZA 85 | 4.2e-7 | |

## Not done / limitations

* **OPAC aerosol is not wired.**
  * The woof Mie module (`mie.py`) agrees with libRadtran's `mie` tool, used
    as a black box, on OPAC water-soluble aerosol: ext 7e-6, ssa 5e-7, g 7e-6
    dry; ext per wet mass 7e-7 at 50 % RH.
  * The installed libRadtran has no OPAC species library. One can be generated
    with the `mie` tool (`aerosol_type <species>`, `basename <species>`, then
    `aerosol_species_library <dir>/`).
  * uvspec selects the *nearest* humidity node and treats mixture rows as
    layer properties.
  * Still to do: the mass-at-50 %-RH scaling, the RH formula, the woof
    pipeline, and the oracle cases.
* **IMS radiance correction** is not implemented (see above). The default
  `disort_intcor` is `moments`, i.e. TMS; uvspec's own default, `phase`
  (Buras et al. 2011), is not implemented either.
* **REPTRAN medium/fine**: the reader handles them, but the installed
  libRadtran ships lookup tables for `coarse` only. Medium and fine run only
  with the separately distributed tables, and they are not oracle-tested.
* **Out of scope**: thermal emission, Baum ice optics (no data installed),
  wavelengths below 240 nm (no O₂ Schumann–Runge treatment), and an SZA ≥ 90°
  sun.
* **Metal speed**: binary64 is soft-float on Metal, which is about 100× slower
  per problem than CUDA. A 16-stream radiance problem with 49 layers takes
  about 3 s per thread-launch batch.
