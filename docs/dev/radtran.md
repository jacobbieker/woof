# woof.radtran: spectral radiative transfer on the GPU

woof.radtran computes spectral irradiances and radiances on atmospheric
columns. It uses REPTRAN representative wavelengths, a discrete-ordinate
solver (DISORT method) or a two-stream solver, Rayleigh scattering, gas
absorption, water and ice clouds, and OPAC aerosol. Each GPU thread solves one
(column, representative wavelength) problem. It runs standalone
(`woof.radtran.driver.compute`) and as an output-only diagnostic on WRF domains
(RunConfig `radtran_*`) and on the MPAS suite route (`&radtran`).

## Clean-room statement

woof is Apache-2.0. libRadtran, including its C DISORT and twostr, is GPL-2.
**woof's implementation is clean-room.** No libRadtran, DISORT, cDISORT or
twostr source text was opened, read, searched or copied. Every component was
written from the published papers listed below and from our own derivations.

libRadtran is used in only these ways:

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
* **libRadtran's `mie` tool as a black box**: `tools/uvspec_oracle/opac_library.py`
  runs it (`aerosol_type <species>`) to make OPAC tables for uvspec, and woof's
  own tables are checked against them. Option names for `mie` and uvspec's
  aerosol inputs come from the libRadtran user's guide (its option
  descriptions), never from source.

| Component | File | Published source |
|---|---|---|
| Discrete ordinates: Fourier azimuth expansion, double-Gauss quadrature, beam particular solution, scaled-exponential boundary-value problem, intensities by source-function integration | `woof/core/kernels/radtran_disort.cu`, `woof/radtran/reference.py` | Stamnes, Tsay, Wiscombe & Jayaweera (1988) Appl. Opt. 27, 2502 |
| Eigenvalue reduction, (α−β)(α+β)v = k²v; symmetrised here via the W^½M^½ similarity, then Cholesky + cyclic Jacobi | same | Stamnes & Swanson (1981) J. Atmos. Sci. 38, 387 |
| delta-M scaling | same | Wiscombe (1977) J. Atmos. Sci. 34, 1408 |
| Pseudo-spherical beam: slant optical depth through homogeneous spherical shells, average layer secant | same | Dahlback & Stamnes (1991) Planet. Space Sci. 39, 671 |
| TMS radiance correction | same + host `ptrue` | Nakajima & Tanaka (1988) JQSRT 40, 51 |
| IMS radiance correction (secondary scattering through the forward peak) | `ims.py` | Nakajima & Tanaka (1988), eqs. 13, 20, 23, 24 |
| Two-stream as the order-1 discrete-ordinate problem (μ₁ = 1/√3, w = 1), delta-Eddington f = g² | same | Kylling, Stamnes & Tsay (1995) J. Atmos. Chem. 21, 115; Joseph, Wiscombe & Weinman (1976) |
| Rayleigh cross section, King factor, depolarisation | `rayleigh.py` | Bodhaine, Wood, Dutton & Slusser (1999), eqs. 18–23 |
| REPTRAN bands, weights, extraterrestrial flux | `reptran.py` | Gasteiger et al. (2014) JQSRT 148, 99 |
| Water clouds | `clouds.py` | Hu & Stamnes (1993) J. Climate 6, 728 |
| Ice clouds | `clouds.py` | Fu (1996) J. Climate 9, 2058 |
| Mie scattering | `mie.py` | Bohren & Huffman (1983) ch. 4; Wiscombe (1980) Appl. Opt. 19, 1505 |
| Exact Legendre moments of the size-integrated phase function: Clebsch-Gordan series of d-function products, 3j symbols by three-term recursion | `mie.py` | Edmonds (1957) *Angular Momentum in Quantum Mechanics* eq. 4.3.2; Schulten & Gordon (1975) J. Math. Phys. 16, 1961 |
| OPAC aerosol components and mixtures | `aerosol.py` | Hess, Koepke & Schult (1998) BAMS 79, 831 |
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

**OPAC aerosol (`aerosol.py`, `mie.py`).**

* **Tables**: for each OPAC species (`inso`, `waso`, `soot`, `ssam`, `sscm`,
  `minm`, `miam`, `micm`, `mitr`, `suso`), at OPAC's wavelengths 0.25–6 µm and
  its humidity nodes, woof computes the bulk optics with its own Mie code.
  The inputs are the user's libRadtran OPAC files (`size_distr.cfg`,
  `refractive_indices/`), read at run time:
  * a lognormal number distribution between OPAC's r_min and r_max, trapezoid
    in ln r with steps of 0.005 in ln r and at most 0.05 in size parameter
    (converged to ~2e-6);
  * extinction per gram of the particle at that node, single-scattering
    albedo, and Legendre moments 0–32.
  The moments are exact for the discretised distribution: the Clebsch–Gordan
  series of |S1|² + |S2|² needs only size-integrated products A_n A_m* with
  |n − m| ≤ l, so there is no angular quadrature however peaked the
  forward lobe. A phase function on 227 angles is computed only when TMS
  needs it. Nodes are computed on demand and cached under
  `$XDG_CACHE_HOME/woof/radtran-opac`.
* **uvspec's rules** (uvspec 2.0.6 with an `aerosol_species_file`; probed,
  tests/data/radtran/uvspec_aerosol_probes.json):
  * **Mass**: layer extinction = mass × the table's extinction per unit mass
    *at the selected node*. uvspec applies no further hygroscopic scaling.
    The profile masses are, by libRadtran's mixture convention, OPAC's
    masses at 50 % RH.
  * **RH at a level**: 100 e / e_s(T), with e = n_H₂O k T and
    ln(e_s/hPa) = 19.1484007 − 4055.15160/T − 185921.210/T². This law was
    fitted to uvspec's `output_user rh` over 182–335 K (residual 4e-7, its
    print precision). A layer's RH is the mean of its two levels' values.
  * **Humidity node**: the nearest node, except that a layer whose nearest
    node is the dry (0 %) one keeps the node of the layer above it. Layers
    are visited from the top, and the top layer starts at the dry node. So
    a dry layer under a humid one takes the humid layer's node. This holds
    on 900 probed layers, apart from exact midpoint ties, which go either way
    with uvspec's single precision.
  * **Profile rows**: as for clouds, the row at a level defines the layer
    above it.
  * **Wavelength**: linear interpolation between the bracketing table
    wavelengths of extinction, scattering coefficient, and scattering
    coefficient × moment. The moments are therefore scattering-weighted.
* **Mapping from the models** (`radtran_aerosol`, `model.aerosol_layers`):
  * **`thompson`**: the aerosol-aware Thompson microphysics' number
    concentrations (WRF `mp_physics = 28` `nwfa`/`nifa`; MPAS
    `mp_thompson_aerosols` scalars) times the dry density. Water-friendly
    aerosol becomes OPAC water-soluble (`waso`), at its mean particle mass at
    the 50 % node. Ice-friendly (dust) becomes OPAC mineral, accumulation
    mode (`miam`), dry. Model layers only.
  * **An OPAC mixture name**: libRadtran's mixture profile by height above
    the surface, on every layer, including the spliced standard atmosphere.
  * Each layer's node follows from the column's own T and water vapour.

**IMS (`ims.py`).** With `disort_intcor moments`, after the kernel's TMS,
Nakajima & Tanaka's eq. (23) adds the secondary scattering through the
forward peak:

du = −(F₀/4π)(1 − fω) ŵ² [2 P̂ − P̂₂] h, with ŵ = fω/(1 − fω).

* The terms:
  * P̂ = Σ(2n+1) ĝ_n P_n, where ĝ_n = 1 below the truncation order 2N and
    g_n/f beyond;
  * P̂₂ = Σ(2n+1) ĝ_n² P_n (eq. 13);
  * h = 1/(μ μ₀*) ∫₀^τ t e^{−t/μ₀*} e^{(t−τ)/μ} dt, in closed form, with
    μ₀* = μ₀/(1 − fω).
* What was fixed against uvspec:
  * the "optically averaged homogeneous atmosphere" above the output level:
    τ is the sum of the layer depths, ω and f are scattering-weighted
    averages, and ĝ_n is weighted by fω dτ;
  * h uses the *unscaled* τ;
  * downward directions within 10° of the solar zenith angle only.
* uvspec's own default, `disort_intcor phase`, is not reproduced (see
  limitations).
* Henyey–Greenstein clouds supply all moments analytically. The aerosol
  tables stop at moment 32, so their peak beyond that is not included.
  uvspec itself refuses `moments` with Mie tables lacking complete moments.
* The correction is host-side (NumPy), after the solver, so no kernel word
  changes.

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
  platform, and the sha256 of every data file used. It also embeds the
  receipts of the OPAC libraries.
* `opac_library.py` builds the two OPAC species libraries uvspec reads
  (`aerosol_species_library`), in a cache directory (`$WOOF_OPAC_LIBRARY`)
  and never in the repository. Both hold the species `inso`, `waso`,
  `soot`, `ssam`, `sscm` and `suso`. The `woof/` library covers 0.4–2 µm.
  The `mie/` library covers 0.45–0.6 µm, because the tool needs about
  20 min for coarse sea salt over 0.4–2 µm.
  * `mie/`: libRadtran's `mie` tool, run as a separate process. The receipt
    records every input text and the conda build: `aerosol_type`,
    `mie_program MIEV0`, `output_user netcdf`, OPAC's wavelengths.
  * `woof/`: woof's tables in the same netCDF layout.
  The aerosol cases record which library they used and its sha256.
  * Cases on `woof/` isolate uvspec's layer rules and match to print
    precision.
  * Cases on `mie/` are end to end and include both Mie codes' size
    integrations.
  * Its `fixture` mode writes `tests/data/radtran/opac_mie_tool.json`, the
    `mie` tool's optics at selected nodes, which woof's tables are held to.
* `probe_aerosol.py` probes uvspec's RH law, layer mean and node rule and
  writes `tests/data/radtran/uvspec_aerosol_probes.json`.
* uvspec's `disort_intcor moments` refuses Mie tables without complete
  moments, so the aerosol radiance cases use `disort_intcor off`.

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
| Cloud, `disort_intcor moments` (TMS + IMS) | 4.7e-7 | TMS alone left 1.6e-2 in the near-solar downward row |
| IMS window, SZA 60, umu −0.65…−0.35 | 3.0e-7 | corrected inside 10° of the sun, not outside |
| IMS, water + ice clouds, 4 levels | 5.3e-7 | layer averages above each level |
| Pseudo-spherical SZA 85 | 4.2e-7 | |
| OPAC urban RH 80, maritime RH 95 (`disort_intcor off`) | ≤3.0e-7 | woof tables |

| OPAC aerosol case (8 streams; mixtures from libRadtran's OPAC files) | Irradiances | Note |
|---|---|---|
| continental_average on afglus; RH 90 | ≤4.8e-7 | uvspec on woof's tables (mie-tool layout): isolates uvspec's layer rules |
| maritime_clean RH 70, 97; at 1600 nm RH 90 | ≤4.9e-7 | waso, ssam, sscm, suso: four hygroscopic species |
| urban RH 10, 80 | ≤3.0e-7 | RH 10: dry node below, the upper troposphere's node carried down |
| continental_average RH 50, mie-tool tables | 3.8e-6 | end to end: the `mie` tool's tables in uvspec, woof's own in woof |
| maritime_clean RH 90, mie-tool tables | 5.4e-6 | |
| urban RH 80, mie-tool tables | 5.1e-5 | soot: the `mie` tool's extinction is 2.9e-4 and its ssa 5.5e-4 off woof's converged integral |
| maritime_clean RH 95, 16-stream radiances, mie-tool tables | fluxes 5.2e-5, radiances 1.5e-4 | sea salt: the `mie` tool's extinction is 4.6e-4 and its moments 3.7e-4 off |

| woof's OPAC tables vs libRadtran's `mie` tool (0.5 and 0.55 µm, every humidity node) | extinction | ssa | moments χ₀…χ₁₆ (absolute) |
|---|---|---|---|
| inso | 6.5e-5 | 5.9e-7 | 6.4e-6 |
| waso | 9.5e-6 | 8.9e-7 | 5.0e-6 |
| soot | 2.9e-4 | 5.5e-4 | 6.5e-6 |
| ssam | 4.5e-4 | 4e-8 | 3.6e-4 |
| sscm | 4.6e-4 | 1.2e-6 | 3.7e-4 |
| suso | 4.9e-5 | 2e-8 | 3.5e-5 |

Single-sphere efficiencies from woof and the `mie` tool agree to its print
precision (≤3e-7). The table differences are therefore the `mie` tool's size
integration, a fixed grid that `dx_max` does not refine for `aerosol_type`.
woof's integral moves by ≤2e-6 when its size-parameter step is halved.

## Not done / limitations

* **OPAC aerosol**:
  * WRF's `aer_opt` 1/2 fields (Tegen climatology, user AOD) are not mapped;
    only Thompson's numbers and the OPAC mixtures are.
  * The phase-function moments stop at 32, so IMS misses the aerosol
    forward peak beyond that order.
  * The first use of a wavelength computes its table nodes. Coarse sea salt
    at 0.25 µm takes ~30 s per node; the visible takes seconds. Nodes are
    cached afterwards.
* **uvspec's default `disort_intcor phase`** (Buras, Dowling & Emde 2011) is
  not implemented: the paper is not openly available. With it, uvspec
  differs from woof's `moments` (TMS + IMS) by 6e-5–2.7e-4 at every
  direction on the cloud cases (its TMS part also changes), consistent with
  a tabulated-phase-function method.
* **REPTRAN medium/fine**: the reader handles them, but the installed
  libRadtran ships lookup tables for `coarse` only. Medium and fine run only
  with the separately distributed tables, and they are not oracle-tested.
* **Out of scope**: thermal emission, Baum ice optics (no data installed),
  wavelengths below 240 nm (no O₂ Schumann–Runge treatment), and an SZA ≥ 90°
  sun.
* **Metal speed**: binary64 is soft-float on Metal, which is about 100× slower
  per problem than CUDA.
  * One thread per problem is latency bound: 16-stream radiances on 30
    layers take ~7 s for any count from 64 to 1024 problems.
  * Radiances on up to 8192 threads (problems × streams) now run their
    Fourier modes on separate threads (`radtran_disort_modes`, then
    `radtran_disort_finish` sums the modes in order and adds TMS). This
    gives the same words: each radiance slot takes one addition per mode.
    Times: 1.1 s for 64 problems, 2.4 s for 128, 3.7 s for 256 and 5.5 s
    for 512.
  * Fluxes only need mode 0 and stay one thread per problem (0.12 s).
  * Large domains are throughput bound, and only less arithmetic would
    help there.
