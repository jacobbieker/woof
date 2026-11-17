# Space weather in woof

Space weather affects the atmosphere through three channels:
- photons: the total and spectral solar irradiance;
- particles: auroral and radiation-belt electrons, and solar protons, which ionise the middle
  atmosphere and make NOx and HOx;
- the ionosphere and thermosphere above the model top, where the solar wind and geomagnetic
  activity act.

WRF and MPAS model tops sit at about 20-80 km, while the ionosphere spans about 60-1000 km.
woof therefore adds space weather in layers. Every layer is opt-in, because each goes beyond
WRF 4.7.1 / MPAS v8.4.2 parity: the default runs execute no new code.

| layer | what it does | status |
|---|---|---|
| L1 total solar irradiance | `solar_tsi_opt`: a measured, time-varying TSI drives every shortwave scheme | available (WRF route, hex generic column seam) |
| L3a ionosphere diagnostic | IRI-type F2/E peaks (CCIR/URSI maps driven by F10.7), Epstein profile, vertical TEC, foF2, hmF2, on the GPU | available as `woof spacewx ionosphere` (global grid) and in WRF-route forecast history (`spacewx_ionosphere_opt`) |
| L3b high latitudes | auroral oval boundaries (Starkov 1994) from Kp, AL or the solar wind (OMNI2, SWPC real time), on the GPU; energy flux and Joule heating to follow | oval available as `woof spacewx aurora` (global grid); energy flux, Joule heating and forecast history planned |
| L3c thermosphere | exospheric temperature and density (a clean-room Jacchia-type model) | planned |
| L2 particle precipitation | ion-pair rates (CMIP6 EPP, Ap-driven MEE, GOES SPE) as NOx/HOx sources in chemistry, for high-top runs | planned |
| L1b spectral irradiance | per-band solar variability in RRTMG SW and photolysis | planned |

Couplings that stay speculative are not implemented, and none is planned. One example is
galactic cosmic rays acting on cloud nucleation: the CLOUD chamber finds that the response of
cloud condensation nuclei is about 1 % or less (Kirkby et al. 2011; Dunne et al. 2016), and
no robust Forbush-decrease cloud signal has been found (Laken et al. 2012).

## L1: total solar irradiance (`solar_tsi_opt`)

WRF's radiation driver uses `SOLCON = 1370 * ECCFAC`. The measured TSI is about 1361 W m-2
(SORCE/TIM, TSIS; Kopp & Lean 2011: 1360.8 +- 0.5 W m-2). It varies by about 1 W m-2 over the
11-year cycle, and by a few tenths of a percent as large sunspot groups cross the disc. For
example, the NOAA CDR has 1362.5 W m-2 on 1 May 2024 and 1360.7 W m-2 on 8 May 2024, just
before the Gannon storm.

With `solar_tsi_opt = 1`, every shortwave output of a radiation call is multiplied by
`TSI(t) / reference`. That covers the heating rate, SWDOWN, GSW, the direct and diffuse
surface beams, the top and bottom flux diagnostics, and the band/g-point outputs. Shortwave
radiative transfer is linear in the incident flux: RRTMG's `solvar(ib) = scon / rrsw_scon`
multiplies each band's source, and the other schemes work the same way. So this equals
changing SOLCON, to within one float32 rounding, for every shortwave scheme at once. The
longwave is untouched.

The factor is applied as soon as the scheme returns, before any consumer reads the result.
The consumers include `swint_opt = 1`, the eclipse overlay, the tendency coupling, the land
surface and history. See the [configuration reference](public/CONFIGURATION.md) for the keys.

The data come from the NOAA Climate Data Record "Total Solar Irradiance" (NRLSSI2 v03, daily;
Coddington et al. 2016, doi:10.7289/V56W985W). It is fetched on demand and never shipped.

**Limits.** The CDR lags real time by a few months, so a forecast for the coming days needs a
user CSV. The spectral (band-wise) variability is not carried: the whole spectrum scales by the
TSI ratio. The UV varies 5-10 times more than the TSI, which matters for stratospheric heating
and ozone, not for weather (layer L1b). The MPAS-native hex radiation suite refuses the keys by
name. So does FARMS (`swint_opt = 2`).

**Expected effect.** In absolute mode the shortwave drops by about 0.6 % (about 5 W m-2 at a
clear-sky noon of 800 W m-2), which removes WRF's constant-high bias. In relative mode the
change is the solar variability alone, at most about 0.15 %.

## L3a: climatological ionosphere (`woof spacewx ionosphere`)

The ionosphere sits above every WRF/MPAS model top, so this layer is a diagnostic: it does
not feed back on the weather. It computes an IRI-type electron density profile on each
column on the GPU (CUDA and Metal; one thread per column) and reports these quantities:
- foF2, NmF2 and hmF2;
- M(3000)F2;
- foE, and foF1/hmF1 where an F1 layer forms;
- the F2 bottomside and topside thicknesses;
- vertical TEC, from 60 to 2000 km.

The model is built from published formulas:
- foF2 and M(3000)F2 come from the CCIR (default) or URSI numerical maps (Jones & Gallet
  1965). foF2 is interpolated in IG12 and M(3000)F2 in R12, as CCIR Report 340 does. Months
  are blended linearly in time between mid-month maps.
- Each map is expanded in Rawer's modified dip latitude, computed from IGRF-14 at 300 km.
- foE uses the IRI effective zenith angle.
- hmF2 follows Bilitza et al. (1979).
- The bottomside thickness comes from the peak density gradient. The topside thickness
  uses the NeQuick-type height scaling.
- Each layer is an Epstein layer.

The formulas and references are listed at the top of `woof/core/kernels/spacewx_ionosphere.cu`.
PyIRI (Forsythe et al. 2023, MIT licence) was read for the arrangement of the map expansions,
and its coefficient files are fetched, pinned by SHA-256, as data. No IRI or PyIRI code is run
or copied.

Space weather enters through F10.7, taken from GFZ's index file:
- `--f107-mode 81day` (the default) uses the trailing 81-day mean of the observed flux;
- `--f107-mode daily` uses the previous day's value.

Both are available at forecast time. The maps are monthly medians, so the profile does not
respond to a geomagnetic storm. That response is layer L3b.

**Validation against GNSS TEC.** `python -m tools.spacewx_gim_validate` grades the model's
vertical TEC against IGS rapid global ionosphere maps (CDDIS). It compares every 2-hour map
node by node, on 71 x 73 nodes per map, with cos(latitude) weights. Bias is model minus GIM,
in TECU. The default 81-day driver and CCIR maps give:

| period | bias | RMSE | correlation | correlation, low / mid / high latitudes |
|---|---|---|---|---|
| 1-3 May 2024 (Ap 7, 42, 8) | -8.7 | 13.6 | 0.90 | 0.87 / 0.89 / 0.77 |
| 10-12 May 2024 (Gannon storm, Ap 105, 271, 53) | -10.6 | 19.1 | 0.81 | 0.78 / 0.66 / 0.54 |
| 11 May 2024 alone (main and recovery phase) | -6.4 | 18.2 | 0.71 | |

Three findings stand out:
- **The storm costs skill.** At mid and high latitudes the correlation falls from 0.89 to
  0.66 and from 0.77 to 0.54. This is the signal a storm-time layer has to capture.
- **The climatology is about 25 % low at this solar maximum.**
  - Only 0.8 TECU of the gap is the profile above 2000 km.
  - The rest is the F10.7-to-R12/IG12 conversion and the climatology itself.
  - With the previous day's flux (`daily`), the storm days come out at bias -2.1, RMSE
    16.2 and correlation 0.77. The quiet days, with a lower daily flux of 135-156 sfu,
    come out at bias -13.5 and RMSE 17.9.
- **The URSI maps change little:** on 1-3 May, bias -8.5, RMSE 13.9 and correlation 0.89.

### In forecast history (`spacewx_ionosphere_opt`)

`spacewx_ionosphere_opt = 1` evaluates the same kernel on a WRF-route domain's
columns (the radiation geometry's latitude and longitude) at each history
frame's valid time and writes five surface rows: `IONO_TEC`, `IONO_FOF2`,
`IONO_HMF2`, `IONO_NMF2` and `IONO_FOE`. Nothing the step reads changes, so
the forecast is identical with the option on or off.

- The F10.7 is the value the run's start would use (`spacewx_f107_mode`), held
  for the whole run. A forecast has no observed flux for its future days, and
  persistence is the usual stand-in. A number in `spacewx_f107_source` fixes it.
- The magnetic coordinates are taken at the start. The UT, month blend and
  solar zenith angle move with each frame.
- The rows are rebuilt from the configuration on restart (no checkpoint state),
  and the four keys are output-only for the restart and prepared-cache
  identities, so a resume may switch the rows on or off. The F10.7 is read
  again on resume; GFZ's observed flux rarely changes once released, and the
  `[spacewx ionosphere]` start-up line records the value and the file's SHA-256.
- Every monthly map the run will blend is loaded when the run starts, so a
  forecast never fetches one mid-run.
- Preflight prices the held arrays (14 column vectors and the five rows) and
  the kernel's 96 B local frame.
- A streamed domain refuses the option, because the rows are not a tile
  carrier, and hex refuses it until a hex history stream carries them.

## L3b: the auroral oval and the solar wind (`woof spacewx aurora`)

`woof spacewx aurora --time 2024-05-10T21:00Z --count 12` places the auroral
oval on a global grid on the GPU (CUDA or Metal; kernel
`woof/core/kernels/spacewx_aurora.cu`, model `woof/spacewx/aurora.py`). Each
column gets its geomagnetic latitude, magnetic local time, a zone (0
equatorward of the diffuse aurora, 1 diffuse aurora, 2 auroral oval, 3 polar
cap) and its distance in degrees from the diffuse aurora's equatorward edge,
which is the usual visibility line of an aurora forecast. `--out` writes the
fields and a JSON receipt.

- **Oval.** Starkov 1994 (Geomagn. Aeron. 34, 331): three boundaries (the polar
  cap edge, the oval's equatorward edge, the diffuse aurora's equatorward
  edge), each a three-harmonic series in MLT whose coefficients are cubic in
  log10|AL|. The coefficient table is Sigernes et al. 2011 (Optica Pura y
  Aplicada 44, 599, Table 2). ocbpy (BSD-3-Clause) carries the same table
  except that the oval's equatorward A0 has b0 = 1.16, where Sigernes and a
  third tabulation give 1.61; woof uses 1.61. |AL| is clamped to 10-2000 nT.
- **Geomagnetic frame.** The centred dipole of IGRF-14 (degree 1) at the
  run's date (2026: boreal pole at 80.9 N, 72.8 W). Starkov's boundaries are
  in corrected geomagnetic coordinates, which differ from the dipole by a few
  degrees in some sectors; that is this layer's main approximation.
- **Activity, three ways.**
  - `--kp` (fixed) or `--indices gfz` (observed Kp, GFZ, CC BY 4.0), converted
    with Starkov's `|AL| = 18 - 12.3 Kp + 27.2 Kp^2 - 2 Kp^3` nT. GFZ adds
    each day's Kp through the day, so the cached file is fetched again when it
    lacks the interval asked for. The cubic peaks near Kp 8.8 (about 653 nT).
  - `--solar-wind omni|swpc|FILE[:FILE]`: Kp estimated from the solar wind.
    `omni` fetches NASA SPDF's hourly OMNI2 years; `swpc` reads NOAA SWPC's
    real-time L1 feed (the active spacecraft only, 1-minute data averaged to
    hourly means with at least 10 minutes each). Each line also prints the
    Newell et al. 2007 coupling and the Boyle et al. 1997 polar cap potential
    (`null` in the JSON receipt for an hour of fill values). A cached OMNI2
    year is fetched again when it does not reach the window or is damaged,
    but not within 6 hours of the last fetch, since OMNI2 lags by weeks.
  - `--solar-wind omni --observed-al` drives the oval with OMNI2's observed AL;
    the Kp estimate is then informational and may be missing. SWPC carries no
    AL, so `swpc` with `--observed-al` is refused.
- **Kp from the solar wind** (`woof.spacewx.solarwind.KP_FIT`). Fitted here on
  OMNI2 against GFZ Kp, since published Kp regressions are not all
  reproducible from their papers. Each 3-hour Kp interval starting at T is
  regressed on the mean of the hourly values stamped T-1, T and T+1 (all three
  must lie in the record read):
  `Kp = a + b (1e-4 dPhi/dt)^0.6 + c (1e-7 sqrt(n) v^2)`, clipped to 0-9, where
  dPhi/dt is Newell's coupling and the second term is a viscous one. Trained
  on 2012-2021 (29,019 intervals), tested on 2022-2024 (8,658): RMSE 0.73,
  correlation 0.84, bias -0.16. For observed Kp >= 5 the RMSE is 1.43 and the
  bias -0.70, so strong storms are underestimated. Newell alone gives RMSE 0.92
  and correlation 0.72.

Checks:
- The device kernel matches the float64 NumPy reference to 0.002 deg in
  geomagnetic latitude with no zone mismatches away from the boundaries
  (tests/test_spacewx_aurora.py, CUDA and Metal).
- Starkov's Kp-to-AL relation against OMNI2's observed AL, 2012-2024 (median
  |AL| observed / Starkov, nT): Kp 2: 89 / 86, Kp 4: 290 / 276, Kp 6: 535 /
  491, Kp 8: 744 / 636. It holds to Kp 5 and reads low above that.
- Against NOAA SWPC's OVATION Prime nowcast (17 Nov 2026 16:52 UT, Kp 0.67):
  the diffuse aurora's equatorward edge (taken as OVATION's 10 % contour) is
  5.0 deg poleward of OVATION's in the north (MAE 6.0) and 5.4 deg in the
  south; with the solar-wind Kp of 1.35, 3.8 and 4.1 deg. The dipole frame and
  the choice of contour account for much of this.
- The storm of 10-11 May 2024: observed Kp 7.67-9 puts the diffuse aurora's
  edge at 44.5-52.5 N geographic. The solar-wind estimate saturates at Kp 9.
  The observed AL (peaking at 1636 nT) takes it to 36.5 N.

## Data sources and licences (all layers)

| input | source | licence |
|---|---|---|
| TSI, SSI | NOAA NCEI CDR (NRLSSI2) | NOAA CDR, open |
| Kp, ap, Ap, F10.7 | GFZ Potsdam (`kp.gfz.de`) | CC BY 4.0; the sunspot column (CC BY-NC) is not read |
| F10.7, solar wind real time (`json/rtsw`) | NOAA SWPC | public domain |
| solar wind historical (OMNI2 hourly, incl. Kp and AL) | NASA SPDF OMNIWeb | public |
| auroral oval coefficients (shipped in `woof/spacewx/aurora.py`) | Starkov 1994, as tabulated by Sigernes et al. 2011 and ocbpy | published coefficients; ocbpy BSD-3-Clause, notice kept in the module |
| Kp-from-solar-wind fit (shipped, 4 numbers) | fitted here on OMNI2 and GFZ Kp | woof's own; GFZ Kp CC BY 4.0 |
| CCIR/URSI ionospheric maps | PyIRI (MIT) at commit c0a50e0a, each file pinned by SHA-256 | ITU-R / URSI published coefficients |
| IGRF-14 | IAGA V-MOD via NOAA NCEI, pinned by SHA-256 | IAGA published coefficients |
| GNSS TEC maps (validation only) | NASA CDDIS IONEX (Earthdata login) | public |
| CMIP6 solar forcing (EPP ion-pair rates) | input4MIPs SOLARIS-HEPPA | CC BY 4.0 |

**Redistribution.** The coefficient sets listed above are fetched at run time into the
space-weather cache and checked against pinned sizes and SHA-256 sums. woof does not ship them:
none is in the wheel, the repository or the test fixtures, which use synthetic coefficients.
Attribution:
- The CCIR foF2 and M(3000)F2 maps are from CCIR Report 340 (ITU-R, 1966/1988), and the URSI
  foF2 maps from Rush et al. 1989 (Telecomm. J. 56, 179). Both are distributed with IRI and
  copied by PyIRI (Forsythe et al. 2023, Space Weather 21, e2022SW003352, MIT licence), whose
  repository at commit c0a50e0a is the download source.
- IGRF-14 is from IAGA Division V-MOD (Alken et al. 2025), downloaded from NOAA NCEI.
- Kp, ap and F10.7 are from GFZ Potsdam (Matzka et al. 2021, CC BY 4.0), with F10.7 from
  DRAO/NRCan. Cite these when publishing results.
- OMNI2 is from NASA SPDF (King & Papitashvili 2005, JGR 110, A02104), with AL from WDC Kyoto.
  The real-time solar wind is from NOAA SWPC.
- The auroral oval coefficients are Starkov's (1994) published fit, as tabulated by Sigernes et
  al. 2011 and by ocbpy (Copyright (c) 2017 Angeline G. Burrell and Gareth Chisham,
  BSD-3-Clause).
- The IONEX TEC maps are from IGS through NASA CDDIS (Johnston et al. 2017, the IGS). They are
  used for grading only.

NRLMSIS 2.x is not freely licensed (NRL restricts commercial use), so it is not bundled.
Neither are the GPL OVATION and Weimer ports. The thermosphere and aurora layers use
clean-room implementations of published formulas instead.

## Validation plan

The ionosphere layers will be checked against observations:
- GNSS TEC from the IGS/CODE/JPL global ionosphere maps at NASA CDDIS;
- foF2 and hmF2 from GIRO ionosondes;
- the storm of 10-12 May 2024, with 1-3 May 2024 as the quiet reference.

The particle layer will be checked against MLS/ACE NOx and ozone. L1 is checked by scaling
identity tests on CUDA and Metal. Its bias direction can be checked against NSRDB / source.coop
surface solar. Its 0.07-0.6 % signal is below the noise of single-station observations.
