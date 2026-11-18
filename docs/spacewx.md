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
| L3a ionosphere diagnostic | IRI-type F2/E peaks (CCIR/URSI maps driven by F10.7), Epstein profile, vertical TEC, foF2, hmF2, on the GPU; a storm-time foF2 correction from filtered ap (woof's own fit on CODE GNSS maps) | available as `woof spacewx ionosphere` (global grid, `--storm` for the correction) and in WRF-route forecast history (`spacewx_ionosphere_opt`) |
| L3b high latitudes | auroral oval boundaries (Starkov 1994) from Kp, AL or the solar wind (OMNI2, SWPC real time), on the GPU; precipitating energy and number flux (OVATION Prime 2010) and Joule power (Knipp et al. 2004) | oval available as `woof spacewx aurora` (global grid) and in WRF-route forecast history (`spacewx_aurora_opt = 1`); energy flux and Joule power available as `woof spacewx aurora --energy`, and the energy flux in history (`spacewx_aurora_opt = 2`) |
| L3c thermosphere | exospheric temperature and density (a clean-room Jacchia-type model) | planned |
| L2 particle precipitation | solar-proton ion-pair rates from the GOES integral flux (OMNI2 or SWPC), and the NOx/HOx they make; auroral-electron ion-pair rates (Fang et al. 2010) from OVATION's energy flux and mean energy | ion-pair rates available as `woof spacewx particles` (electrons with `--electron-energy-flux`, a diagnostic above every model top); the NO/OH/HO2 source in RADM2 chemistry as `spacewx_epp_opt = 1` (WRF route) |
| L1b spectral irradiance | `solar_ssi_opt`: the day's solar spectrum (NOAA CDR SSI) moves the source between RRTMG SW's 14 bands | available for RRTMG (`ra_sw_physics = 4`, `ra_rrtmg_variant = "rrtmg_legacy"`) and RRTMG-K (14) on the WRF route; photolysis not yet |

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
surface and history. The shortwave diagnostics a scheme writes into the driver's fields itself
are scaled by the same factor: RSWTOA for GSFC (`ra_sw_physics = 2`), and RSWTOA and SWDOWNC for
GFDL-Eta (`ra_sw_physics = 99`). See the [configuration reference](public/CONFIGURATION.md) for
the keys.

At start-up the run prints a `[spacewx tsi]` line: the source, the SHA-256 of the TSI files
read, the mode and the reference. A fetched CDR file is read as a TSI series before it is
installed in the cache; a truncated or wrong download is deleted and refused.

The data come from the NOAA Climate Data Record "Total Solar Irradiance" (NRLSSI2 v03, daily;
Coddington et al. 2016, doi:10.7289/V56W985W). It is fetched on demand and never shipped.

**Limits.** The CDR lags real time by a few months, so a forecast for the coming days needs a
user CSV. The spectral (band-wise) variability is not carried: the whole spectrum scales by the
TSI ratio. The UV varies 5-10 times more than the TSI, which matters for stratospheric heating
and ozone, not for weather; `solar_ssi_opt` (L1b, below) adds it for RRTMG. The MPAS-native hex radiation suite refuses the keys by
name. So does FARMS (`swint_opt = 2`).

**Expected effect.** In absolute mode the shortwave drops by about 0.6 % (about 5 W m-2 at a
clear-sky noon of 800 W m-2), which removes WRF's constant-high bias. In relative mode the
change is the solar variability alone, at most about 0.15 %.

## L1b: spectral irradiance (`solar_ssi_opt`)

RRTMG's shortwave source has a fixed spectral shape (Kurucz, in each band's `sfluxref`), and
`solar_tsi_opt` scales all of it by one number. Over the solar cycle the shape changes: the
NOAA CDR (NRLSSI2) puts the 200-263 nm band (RRTMG band 28) 3 % higher at the 2024 maximum
than at the 2019 minimum, 263-345 nm (band 27) 0.3 %, and the visible and near-infrared
bands 0.1 % or less, with the opposite sign.

With `solar_ssi_opt = 1` each of RRTMG SW's 14 bands (16-29) takes the factor

    f_b(t) = [S_b(t) / TSI(t)] / [S_b,ref / TSI_ref]

where S_b is the CDR's daily spectral irradiance integrated over the band and the reference is
its mean over solar cycle 24 (2009-2019, shipped as 14 numbers in
`woof/spacewx/solar_spectrum.py` and refitted by `tools/spacewx_ssi_reference.py`). The
factors are renormalised so that sum_b w_b f_b = sum_b w_b with w_b = S_b,ref / TSI_ref:
the band shares move, and the total stays what `solar_tsi_opt` (or WRF's 1370 W m-2) sets.
RRTMG's own (Kurucz) band shares are within 1.2 % of NRLSSI2's except 263-345 nm (+7 %), so
RRTMG's top-of-atmosphere total moves by about 1e-5 (1.3e-5 on 2024's largest UV day, measured
on a legacy RRTMG column). The factor multiplies the per-band `adjflux` of the batched SW engine
(`band_factor`; `None` leaves every output bitwise unchanged) at every radiation call, from the
series at the call's model time.

| band (RRTMG) | 2024 mean (min-max) | 2019 mean | 2014 mean |
|---|---|---|---|
| 28: 200-263 nm | 1.0246 (1.0110-1.0453) | 0.9931 | 1.0110 |
| 27: 263-345 nm | 1.0026 (1.0008-1.0046) | 0.9993 | 1.0011 |
| 26: 345-442 nm | 1.0012 (1.0000-1.0023) | 0.9997 | 1.0005 |
| 25: 442-625 nm | 0.9998 | 1.0001 | 0.9999 |
| 20: 1.63-1.94 um | 0.9990 (0.9982-1.0001) | 1.0003 | 0.9996 |

**Validation.** NRLSSI2 is a model of the spectrum (from sunspot and facular indices),
so its band 27-28 irradiances were compared with SORCE SSI level 3 (SOLSTICE below 310 nm,
SIM above; LASP LISIRD), on the 1836 days of 2009-2019 both cover:

| band | mean NRLSSI2 / SORCE | daily anomaly correlation | slope NRLSSI2 vs SORCE | 27-day std, SORCE / NRLSSI2 |
|---|---|---|---|---|
| 28: 200-263 nm | 1.003 | 0.79 | 0.73 | 0.285 % / 0.276 % |
| 27: 263-345 nm | 1.018 | 0.60 | 0.26 | 0.083 % / 0.044 % |

In band 28 the annual means rise and fall together (2014 +0.82 % / +1.17 %, 2019 -1.32 % /
-0.66 % against the cycle mean); NRLSSI2's rotational variability matches, and its cycle
amplitude is 0.7-0.85 of SORCE's (SOLSTICE's long-term degradation correction is the larger
uncertainty there). In band 27 SORCE varies two to three times more, mostly SIM above 310 nm,
whose cycle trends are disputed (Ermolli et al. 2013); NRLSSI2's smaller UV cycle is the
conservative choice, and both bands carry under 4 % of the TSI.

At start-up the run prints a `[spacewx ssi]` line: the files, their SHA-256 and the band
factors at the start. Hex refuses the key, as do RTE+RRTMGP (`ra_rrtmg_variant` other than
`"rrtmg_legacy"`, whose g-point solar source has its own bands and its own NRLSSI2 facular/sunspot
terms), every other shortwave scheme, FARMS (`swint_opt = 2`), and DFI (the filter captures the step in a CUDA graph,
which would freeze the first call's factor). The CDR lags real time by a few months; a forecast
for the coming days needs a netCDF in the CDR's layout. Photolysis (the chemistry's actinic
flux) does not take the factor yet.

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
  1965). foF2 is interpolated in IG12 and M(3000)F2 in R12, as CCIR Report 340 does. The
  quadratic R12-to-IG12 fit turns over at R12 of about 247 (F10.7 near 300 sfu), so R12 is
  held at that point above it rather than letting more flux give a lower foF2. Months
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
respond to a geomagnetic storm unless the storm-time correction below is switched on.

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

### Storm-time correction (`woof spacewx ionosphere --storm`)

`--storm` multiplies foF2 by a storm factor sqrt(R), where R = NmF2(storm) / NmF2(quiet),
on each column and at each time. hmF2, the thicknesses and TEC then follow from the
corrected peak, as usual. The fit is woof's own and is shipped as
`woof/data/spacewx/storm_fit_v1.json`. It follows the approach of STORM (Araujo-Pradere et
al. 2002), but no STORM code or coefficients are used.

- **Driver.** Wrenn's (1987) filtered ap,
  `ap_tau(t) = (1 - tau) * sum_i tau^i ap_i`, taken over the 3-hour GFZ ap intervals that
  have ended by t (40 intervals, 120 h). It never reads the future, so a forecast can use it
  up to its start. The latest two intervals must have ap, because a storm's onset is in
  them: before GFZ publishes them, `--storm` refuses the time rather than fall back to older
  ap. A cached GFZ file that lacks them is fetched again. An older missing interval is left
  out and the weights renormalised (at least 80 % of the weight must be present).
- **Model.** `log R = a x + b x^2`, with `x = ap_tau / 100`. There is no intercept, so a
  quiet day has R = 1. (a, b) are interpolated linearly between nodes of:
  - dipole |MLAT|: 0, 20, 35, 50, 62.5 and 75 deg;
  - local solar time: every 4 h, periodic;
  - season: winter, equinox and summer of the magnetic hemisphere, going linearly to
    equinox within 10 deg of the dip equator, so that the two hemispheres meet without a
    jump.

  x is held at the largest value in the training data, and R is clipped to the range of the
  training data.
- **Target.** `log(TEC / TEC_q)` on CODE's final global ionosphere maps (AIUB, University of
  Bern), 2-hourly and at nodes every 5 deg x 10 deg. TEC_q is the median of the 27 days
  before, at the same UT and node. This measures the storm's departure from the recent quiet
  ionosphere, independently of the climatology's own bias.
- **Fitting.** The fit was trained on 2015, 2017 and 2023, and tested on 2024 and 2025.
  - A nuisance term per |MLAT| node, in the day-before F10.7 over its 27-day mean, is fitted
    alongside but not shipped. It keeps flux changes that come with storms out of the ap
    coefficients.
  - The filter constant (tau = 0.6-0.98) and the shape (linear, or linear plus quadratic)
    were chosen by leave-one-year-out RMSE on the storm rows: those with ap_tau >= 30 for
    tau = 0.8, the same rows for every candidate. Training RMSE alone favours the slowest
    filter, which tracks solar rotation rather than storms.
  - The choice is tau = 0.93 with the quadratic shape.
- **Tools.** `tools/spacewx_storm_fit.py` refits and grades the model.
  `tools/spacewx_gim_validate.py --storm` grades the corrected climatology.

**Test years (2024-2025), against the quiet reference TEC_q** (RMSE of TEC in TECU; the
storm rows have ap_tau >= 30 or >= 80 for tau = 0.8):

| rows | |MLAT| 0-20 | 20-40 | 40-60 | 60-90 | all |
|---|---|---|---|---|---|
| all (10.2 million) | 9.63 -> 9.42 | 7.43 -> 7.34 | 5.92 -> 5.78 | 5.37 -> 5.34 | 7.18 -> 7.06 |
| ap_tau >= 30 | 13.70 -> 13.70 | 11.44 -> 11.62 | 8.58 -> 7.83 | 6.70 -> 6.07 | 10.26 -> 10.05 |
| ap_tau >= 80 | 18.46 -> 19.79 | 18.84 -> 18.34 | 13.03 -> 9.32 | 8.95 -> 7.14 | 15.06 -> 14.37 |

The gain is in the negative phase at middle and high magnetic latitudes. At low latitudes
the positive phase is not captured: 0-20 deg gets slightly worse in strong storms. The RMSE
of log R barely moves overall (0.298 -> 0.297), because storm-to-storm variability is much
larger than any mean storm response.

**On the climatology, against CODE TEC** (cos-latitude weighted; correlation and RMSE in
TECU, without -> with `--storm`):

| period | correlation all / mid / high | RMSE all / mid / high |
|---|---|---|
| 1-3 May 2024 (quiet) | 0.897 -> 0.901 / 0.864 -> 0.873 / 0.721 -> 0.747 | 13.2 -> 13.3 / 8.1 -> 8.2 / 6.7 -> 6.9 |
| 10-12 May 2024 (Gannon storm) | 0.798 -> 0.813 / 0.651 -> 0.713 / 0.522 -> 0.671 | 19.6 -> 20.5 / 14.3 -> 16.0 / 8.6 -> 10.8 |
| 10-12 Oct 2024 (storm) | 0.782 -> 0.817 / 0.768 -> 0.786 / 0.555 -> 0.672 | 25.2 -> 25.9 / 14.2 -> 16.5 / 10.4 -> 12.2 |

The correction improves the storm pattern (correlation), but it raises the TEC RMSE. The
reason is that the climatology already reads about 25 % low at this solar maximum, and a
negative phase lowers TEC further.

**Against ionosondes (foF2, GIRO).** Twenty station-periods are graded at hourly medians
(autoscaling confidence >= 70; storm hours by the fit's own ap_tau):
- stations: Juliusruh, Ebro, Athens, Pruhonice, Dourbes, Wallops Island, Austin, Point
  Arguello, Millstone Hill, Tromso, Grahamstown and Learmonth;
- periods: 12 Apr-20 May and 8 Sep-16 Oct 2024.

Here the climatology reads foF2 high during storms, and the correction helps:

| hours | n | RMSE, MHz | bias, MHz | correlation |
|---|---|---|---|---|
| all | 15,897 | 1.46 -> 1.40 | +0.64 -> +0.41 | 0.850 -> 0.859 |
| ap_tau >= 30 | 2,788 | 2.46 -> 2.14 | +1.69 -> +0.24 | 0.777 -> 0.723 |
| ap_tau >= 60 | 922 | 2.83 -> 2.36 | +2.05 -> -0.75 | 0.718 -> 0.637 |

The two storms differ:
- In October the correction helps at every mid-latitude station (storm RMSE 2.9-3.1 ->
  2.0-2.2 MHz).
- In May it overshoots: storm bias -1.3 to -2.2 MHz and higher RMSE at most stations, so the
  storm-hour correlation drops.

The GIRO data are CC BY-NC-SA 4.0. They were used for this grading only and are not shipped.
We thank the station operators and the Lowell GIRO Data Center.

The CLI reads ap from GFZ's file, which must reach the requested time, so `--storm` refuses
`--f107`. The dipole latitudes are taken at the first time. The receipt records the fit's
SHA-256 and tau, and gives each time's ap_tau. The `storm_factor` field is R.

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

### Energy flux and Joule heating (`woof spacewx aurora --energy`)

With a solar wind (`--solar-wind omni|swpc|FILE`), `--energy` adds the
precipitating particle fluxes of OVATION Prime 2010 to the same grid, and with
OMNI2 the Joule power. The model is `woof/spacewx/ovation.py`.

- **Model.** OVATION Prime 2010 (Newell, Sotirelis & Wing 2009; Newell et al.
  2010) regresses DMSP precipitation on the Newell coupling dPhi/dt, on a
  96 MLT x 0.5 deg grid from 50 deg poleward, for four seasons and four kinds
  of aurora: diffuse, monoenergetic and broadband electrons, and ions. woof
  ports NOAA's public-domain IDL distribution, ovation-prime 2.3.0 (P. Newell,
  JHU/APL; R. Redmon and J. Machol, NOAA/NGDC), routine by routine:
  - dPhi/dt is averaged over the 4 hours up to the time, with weights 0.65^k;
  - the four seasons are blended by day of year;
  - the probability of each aurora type is fitted;
  - the IDL's outlier clamps are kept;
  - the northern and southern bins at the same |MLAT| are combined, as the
    authors recommend.
- **Two departures from the IDL code, both documented in the module:**
  - The IDL's fallback table of probabilities is read into a subscripted array,
    which IDL passes by value, so it is never filled and the fallback gives 0.
    woof keeps that behaviour, and the distribution's test outputs confirm it.
  - For the southern hemisphere woof moves the day of year by half a year. The
    IDL's comment says `365 - doy`, which would leave the solstices in the same
    season.
- **The coefficients are not shipped.** The 24 MB distribution zip is fetched
  from SourceForge into `~/.woof/spacewx/ovation` and checked against its
  pinned size and SHA-256 (`--ovation` takes a local zip or `premodel`
  directory, and the receipt's `spacewx_ovation_pinned` says whether the
  tables are the pinned ones). OvationPyme (LGPL) is not read or used.
- **Outputs.** In the `.npz`:
  - `eflux`: the electron energy flux, diffuse + mono + wave, in erg cm-2 s-1
    (= mW m-2);
  - each part of it, and the ion energy flux (`eflux_ion`);
  - `nflux`: the electron number flux, in cm-2 s-1;
  - `mean_energy`: eflux / nflux, in keV.

  A time with fewer than 3 good solar-wind hours of the 4 has no coupling:
  its fields are NaN and its scalars `null`, and the other times are kept.

  Each line, and the receipt, also carries the northern and southern
  hemispheric power in GW (electrons, and ions separately). It is integrated as
  the IDL does it, leaving out the dayside strip at 7-19 MLT equatorward of
  63 deg. The grid is placed on the IGRF-14 dipole frame. OVATION's own MLAT is
  AACGM, so the dipole shifts it by a few degrees in some sectors, as it does
  for the oval.
- **Joule power.** Knipp et al. 2004 (Sol. Phys. 224, 495, Table I) give the
  northern hemisphere's Joule power as a seasonal quadratic in |PC(N)| and
  |Dst|, fitted to AMIE:
  - winter (21 Oct - 20 Feb): `13.36 PC + 5.08 PC^2 + 0.47 Dst + 0.0011 Dst^2` GW;
  - summer (21 Apr - 20 Aug): `29.27 PC + 8.18 PC^2 - 0.04 Dst + 0.0126 Dst^2` GW;
  - equinox (otherwise): `29.14 PC + 2.54 PC^2 + 0.21 Dst + 0.0023 Dst^2` GW.

  PC(N) and Dst come from OMNI2 (words 52 and 41), so SWPC runs have no Joule
  line. Knipp et al. double the northern value for the globe. During great
  storms the fit is extrapolated: on 10-11 May 2024 it gives 1.5-4 TW.

Checks (`tools/spacewx_ovation_validate.py`):
- **Against the IDL.** The distribution's test outputs for the storm of 24-25
  Oct 2011 were run with its own OMNI2 file.
  - All 63 energy-flux outputs agree with the port cell by cell to 0.02
    erg cm-2 s-1, where the IDL prints 0.01 and the largest difference is 0.019.
    Their hemispheric powers agree to 0.01 %.
  - Of the 60 number-flux outputs, 58 agree to 0.3 % of each field's maximum. The 03 UT wave output
    differs in one cell, where the two hemispheres straddle a clamp. The 12 UT
    diffuse output, from NOAA's later trunk build, has flux in 165 cells at
    50-52 deg where the regression itself is 0 in that day's seasons, so it
    was made with other tables.
  - The outputs show the 2.3.0 IDL's probability: 0 where the probability fit
    is absent, as woof does. The tabulated fallback the code intends (but
    never fills) would miss every electron output by 2-190 % of its peak.
  - A unit test repeats the energy-flux check whenever the pinned zip is cached.
- **Against observations.** The NOAA POES/MetOp TED electron energy flux (NCEI
  level 2, `ted x econtr`) was taken at the 120-km foot of each 16-s record.
  Six satellites over 24-25 Oct 2011 gave 29,403 records poleward of 50 deg
  and 338 polar passes. For both hemispheres:
  - log-flux correlation per record: 0.57 (north), 0.56 (south);
  - correlation of the flux summed over each pass: 0.48 (north), 0.61 (south);
  - median ratio of model to observed: 2.3 (north), 2.6 (south). With ions
    against TED's total, it is 2.4 and 2.7.

  The high ratio is consistent with TED's 20 keV upper limit, against DMSP's
  30 keV, and with TED viewing only near the local vertical. Single records
  scatter widely, because OVATION is a model of mean precipitation.
- **The storm of 10-11 May 2024** (OMNI2):
  - hemispheric power rises from about 12 GW (16 UT) to 110-115 GW (21-01 UT);
  - the northern Joule power is 1.5-4 TW.

### In forecast history (`spacewx_aurora_opt`)

`spacewx_aurora_opt = 1` evaluates the oval kernel on a WRF-route domain's
columns at each history frame's valid time and writes four surface rows:
`AURORA_ZONE`, `AURORA_DIST` (the visibility line), `MAG_LAT` and `MAG_LT`.
The forecast is identical with the option on or off.

- The Kp is the one the start sees (`spacewx_kp_source`), with nothing after
  the start read: the observed GFZ Kp of the latest 3-hour interval that has
  ended by the start, the solar-wind estimate (OMNI2 or SWPC) for the latest
  interval whose fitted hours T-1, T and T+1 have ended by the start, or a
  number. It is held for the run (persistence) and must lie within 24 h before
  the start. The receipt names the interval it came from. If GFZ cannot be
  reached, the cached file is used.
- The dipole frame is taken at the start; the MLT turns with each frame.
- `spacewx_aurora_opt = 2` adds three rows from OVATION Prime 2010:
  `AURORA_EFLUX` (the electron energy flux, diffuse + mono + wave, mW m-2),
  `AURORA_EMEAN` (its mean energy, keV) and `AURORA_IFLUX` (the ion energy
  flux, mW m-2). It needs `spacewx_kp_source = "omni"` or `"swpc"`. OVATION's
  dPhi/dt is taken on the hour of the start from the three hours before it,
  which must all have good solar wind; the hour in progress and anything later
  are not read. It is held for the run, and so is each hemisphere's season. Each frame looks the held grids up at the oval's
  MAG_LAT and MAG_LT, so the pattern turns with the Sun as the oval does. The
  `[spacewx aurora]` line adds dPhi/dt, the coefficients' SHA-256 and the
  hemispheric power. The dPhi/dt is kept with the held Kp (by option 1 too),
  so a resume may switch the energy rows on. A hold written before this
  option existed has no dPhi/dt: the solar wind for the start is then read
  again, and the resume is refused by name if the feed no longer covers it
  (SWPC keeps only 24 h).
- Rows are rebuilt from the configuration on restart (no checkpoint state),
  and the two keys are output-only for the restart and prepared-cache
  identities, so a resume may switch the rows on or off. A fetched Kp (gfz,
  omni, swpc) is kept per source and start in `~/.woof/spacewx/aurora-activity`,
  so a resume on the same host, and every domain of a nested run, holds the
  Kp the first read found, even after SWPC's 24-hour feed has moved on. The
  `[spacewx aurora]` start-up line records the value, the interval and the
  source. Preflight prices the held arrays and the kernel's
  32 B frame. Streamed domains and hex refuse it.

## L2: solar proton events (`woof spacewx particles`)

`woof spacewx particles` turns the integral proton flux that GOES measures into
ion-pair rates in the polar caps, and the NOx and HOx those ion pairs make:

```
woof spacewx particles --time 2012-01-24T12:00Z --count 6 --step-hours 1 --source omni \
    --levels 0.01 0.1 1 10 50 100 --out jan2012.json
```

- **Flux.** `--source omni` reads OMNI2's words 43-48 (J(>1, >2, >4, >10, >30,
  >60 MeV), cm-2 s-1 sr-1, GOES since the 1990s; after about 2010 only >10, >30
  and >60 MeV are filled). `--source swpc` reads SWPC's real-time GOES feed
  (`json/goes/primary/integral-protons-3-day.json`, >1 to >500 MeV, the last
  3 days), averaged to hours that hold at least 3 of their 12 values; where
  two spacecraft report the same time, the one with more records is kept. A path
  reads an OMNI2 file (several joined by `:`).
- **Event gate.** An hour counts only while J(>10 MeV) is at least 10 pfu (NOAA
  S1). Below that the channels hold mostly galactic cosmic rays and detector
  background (about 0.2 pfu at >10 MeV), which this layer leaves out.
- **Spectrum.** A power law in log J against log E between the channels
  present. Past the last channel the last pair's slope is kept, but never
  flatter than J ~ E^-3, up to 1 GeV (the protons above 1 GeV are counted at
  1 GeV).
- **Deposition.** Protons in air:
  - stopping power from the Bethe formula (Z/A 0.4992, I = 85.7 eV, within 3 %
    of NIST PSTAR at 10 and 100 MeV);
  - CSDA ranges;
  - an isotropic downward flux over 2π, with a 16-point Gauss-Legendre rule
    in the cosine of the zenith angle;
  - one ion pair per 35 eV.
  The depth is p/g, so the rates need no temperature. Energy is conserved: the
  column absorbs the whole flux.
- **Chemistry yield.** 1.25 NOx and 2 HOx per ion pair (Porter et al. 1976;
  Solomon et al. 1981; the CMIP6/CMIP7 convention). The tendency in mixing
  ratio is yield x q x M_air / N_A, with q in ion pairs per gram per second.
- **Where.** The polar caps, |magnetic latitude| >= 60 degrees (the CMIP7
  rates are uniform poleward of about 60 degrees). The command prints one
  profile per time. `--out` writes it with the flux, the event flag and a
  receipt (source, SHA-256, channels).

For the storm of 23-24 January 2012 at 12 UT (1310 pfu above 10 MeV), the rates
in ion pairs per gram per second are:

| pressure | ion pairs per gram per second | NOx made |
|---|---|---|
| 0.01 hPa | 7.6e9 | 40 ppbv/day |
| 0.1 hPa | 4.4e9 | 23 ppbv/day |
| 1 hPa | 1.4e8 | 0.7 ppbv/day |
| 10 hPa | 2.5e5 | 0.001 ppbv/day |

A model top at 10-50 hPa therefore sees almost nothing from a soft event like
this one. Hard events (October 1989, July 2000, January 2005) reach 10-100 hPa.

**Validation against CMIP7.** The rates are compared with the CMIP7 solar
forcing (input4MIPs SOLARIS-HEPPA v4.6, `iprp`, daily, CC BY 4.0) at 77.5
degrees N. That forcing is built from GOES fluxes with a full ionisation model,
so it serves as the reference here. Each number is the geometric mean, over the
event days, of the daily-mean ratio woof/CMIP7; hours outside an event count as
zero:

| pressure | training: 8 days, 2000-2005 | test: 7 days, 2012-2017 |
|---|---|---|
| 0.01 hPa | 0.57 | 0.14 |
| 0.1 hPa | 0.57 | 0.82 |
| 1 hPa | 0.55 | 0.82 |
| 3 hPa | 0.73 | 0.53 |
| 10 hPa | 0.65 | 0.48 |
| 31.6 hPa | 0.59 | 0.39 |
| 100 hPa | 0.96 | 0.59 |

The floor on the tail index was the only number chosen against CMIP7, and only
on the training days. With no floor (the 30-60 MeV slope carried to 1 GeV), the
training ratios are 1.8 at 10 hPa, 5.8 at 31.6 hPa and 38 at 100 hPa. A floor
of 2 gives 1.2, 2.1 and 7.3; a floor of 3 gives 0.65, 0.59 and 0.96; a floor of
4 gives 0.35, 0.16 and 0.13. Floor 3 is shipped.

Known limits:
- The spectrum starts at the lowest channel present, with nothing below it.
  After 2010, OMNI2 has no channel below 10 MeV, so the protons under 10 MeV,
  which ionise above about 0.03 hPa, are missing, and 0.01 hPa reads about 7
  times low. The SWPC feed has the >1 and >5 MeV channels. Before 1988 (IMP 8),
  the channels that OMNI2's word 49 marks as holding magnetospheric protons
  are dropped.
- OMNI2's proton words 43-48 are fill from 2021 on, so `--source omni` gives
  no flux (NaN, no event) for recent years; use `--source swpc` for the last 3
  days. Between 2021 and the SWPC window there is no source here.
- The last energy bin, at about 1 GeV, also holds the protons above 1 GeV.
  In the atmosphere they lose energy at close to the minimum-ionising rate, as
  a 1 GeV proton does, so they are placed there rather than dropped.
- Single days scatter by a factor of 2-5. The largest misses are the decay
  days of hard events: 2001-04-15 and 2005-01-20, where three channels cannot
  represent a spectrum hardening above 100 MeV.

### In RADM2 chemistry (`spacewx_epp_opt`)

`spacewx_epp_opt = 1` turns those ion pairs into chemistry on a WRF-route
domain that runs RADM2 (`chem_sets` `radm2` or `radm2sorg`).
- **What it adds.** At each chemistry step, inside `emission.radm2` (so the
  ledger books it as emitted), each polar-cap column gains:
  - 1.25 NO per ion pair;
  - one OH and one HO2 per ion pair.
  RADM2 has no N or H atoms. The N atoms are therefore counted as NO, as in
  models without odd-nitrogen atoms. The H atom is counted as HO2, which it
  becomes within seconds through H + O2 + M.
- **Where.** The polar caps, |dipole magnetic latitude| >= 60 deg at the start.
- **The flux.** `spacewx_epp_source` (`omni`, `swpc` or OMNI2 file(s) joined
  by `:`) gives the GOES flux of the last hour that has ended by the domain's
  start. That flux is held for the run (persistence, as the aurora's Kp is).
  Each domain holds the hour before its own start: a nest started with its
  parent holds the same hour, a nest started later holds its own. A fetched
  flux is kept per source and start in the space-weather cache, so a resume
  holds the same one. An hour below S1 adds nothing. An hour whose >10 MeV flux
  is missing is refused rather than taken as quiet; OMNI2's proton words are
  fill from 2021 on, so recent runs need `swpc` (the last 3 days).
- **On the device.** The deposition rate per unit depth is tabulated on the
  host, in float64, on the 241-point depth grid. Each layer reads it, log-log,
  at its mid depth sqrt(X_top X_bottom), with X = p8w / g. This is within 3 %
  of the layer means wherever the rate matters
  (`tests/test_spacewx_epp_chem.py`).
- **Refusals.** Streamed domains and hex refuse the option, for now, and so
  does a run with no radiation scheme (the caps are placed on the radiation
  geometry's xlat/xlong). A domain with no cell in a polar cap adds nothing.

**Validation.** The source is the ion-pair rate above multiplied by fixed
yields, so its validation is the CMIP7 comparison. A comparison with MLS (OH,
HO2) or SABER is not made here, because at WRF's usual model tops it would
measure nothing. At 10 hPa the January 2012 event makes about 0.001 ppbv of NO
per day, and at 50 hPa less. Those amounts are far below MLS's single-profile
precision for OH and HO2 at those heights. A high-top run reaching 1 hPa or
above would gain about 1 ppbv per day, which MLS could see in principle; that
comparison awaits such a run.

### Auroral electrons (`woof spacewx particles --electron-energy-flux`)

`--electron-energy-flux` (erg cm-2 s-1) and `--electron-mean-energy` (keV) add
the ion-pair rates of precipitating auroral electrons, and the NOx they make
(1.25 per ion pair), at the same levels. The inputs are OVATION Prime's
`eflux` and `mean_energy` (`woof spacewx aurora --energy`, L3b) at a point:

```
woof spacewx particles --time 2024-05-11T00:00Z --source none \
    --electron-energy-flux 10 --electron-mean-energy 4 --levels 1e-6 1e-5 1e-4 1e-3 0.01
```

- **Model.** Fang et al. 2010 (GRL 37, L22106): a fit to first-principles
  multi-stream and two-stream transport models, for isotropic monoenergetic
  electrons of 100 eV to 1 MeV. The rate per gram is Q f(y, E) / (35 eV X),
  with y = (2 / E) (X / 6e-6)^0.7 and X the column depth (g cm-2), so it needs
  no temperature. Layer means come from the energy absorbed above each
  interface, as for the protons.
- **Spectrum.** A Maxwellian number flux, E exp(-E / E0), whose mean energy
  2 E0 is OVATION's eflux / nflux (`--electron-spectrum maxwellian`, the
  default), on 160 log bins over 0.1-1000 keV; or monoenergetic
  (`--electron-spectrum mono`). A mean energy that puts more than 5 % of the
  energy flux outside the fit's 0.1-1000 keV (below about 0.25 keV or above
  about 300 keV) is refused, as is a negative or non-finite flux; a zero flux
  with a zero mean energy (OVATION outside the oval) gives zeros. The profile
  is one profile for the given inputs, recorded with `--time`; `--count`
  steps the protons only.
- **Checks** (`tests/test_spacewx_particles.py`):
  - The column total against the incident energy: 0.84 at 100 eV, 0.94 at
    1 keV, 0.99 at 10 keV, 1.01 at 100 keV and 1.05 at 1 MeV. This is the fit's
    own error, which the paper quotes as within 5 % above 1 keV. It is not
    renormalised.
  - The peaks of the monoenergetic rates per volume lie where the transport
    models put them: 10 keV at about 1.3e-4 hPa (about 105 km), 1 MeV at about
    0.15 hPa (about 60 km).
- **HOx is not given** for the electrons. HOx forms through water cluster
  ions, which fade above about 80 km where these electrons deposit.
- **Not a forecast source.** A Maxwellian of 10 keV mean energy, harder than
  OVATION's usual 1-5 keV, falls to 1e-6 of its peak rate by 6.5e-3 hPa
  (about 85 km). Even a 1 MeV monoenergetic beam peaks at 0.15 hPa. WRF and
  MPAS model tops sit at pressures of 1 hPa or more (about 48 km or lower), so
  `spacewx_epp_opt` carries the protons only. Coupling the electrons to a
  forecast would add exactly nothing inside the domain. The indirect effect,
  thermospheric NOx carried down inside the polar vortex in winter, needs a
  model whose top reaches the lower thermosphere.

Radiation-belt (medium-energy, 30 keV-1 MeV) electrons reach 60-90 km. They
need a flux source (POES MEPED) that woof does not read, and they too stay
above WRF tops. Galactic cosmic rays are left out.

## Data sources and licences (all layers)

| input | source | licence |
|---|---|---|
| TSI, SSI (and the shipped 14 cycle-24 band shares) | NOAA NCEI CDR (NRLSSI2 v03; Coddington et al. 2016) | NOAA CDR, open |
| SORCE SSI level 3 (validation only) | LASP LISIRD (SORCE SOLSTICE/SIM) | NASA, public |
| Kp, ap, Ap, F10.7 | GFZ Potsdam (`kp.gfz.de`) | CC BY 4.0; the sunspot column (CC BY-NC) is not read |
| F10.7, solar wind real time (`json/rtsw`) | NOAA SWPC | public domain |
| solar wind historical (OMNI2 hourly, incl. Kp and AL) | NASA SPDF OMNIWeb | public |
| auroral oval coefficients (shipped in `woof/spacewx/aurora.py`) | Starkov 1994, as tabulated by Sigernes et al. 2011 and ocbpy | published coefficients; ocbpy BSD-3-Clause, notice kept in the module |
| auroral precipitation (OVATION Prime 2010 coefficients) | NOAA's ovation-prime 2.3.0 distribution on SourceForge, pinned by SHA-256 | public domain (NOAA); fetched at run time, not shipped |
| Joule power fit (shipped, 16 numbers) | Knipp et al. 2004, Table I | published coefficients |
| POES/MetOp TED energy flux (validation only) | NOAA NCEI SEM-2 level 2 | public |
| Kp-from-solar-wind fit (shipped, 4 numbers) | fitted here on OMNI2 and GFZ Kp | woof's own; GFZ Kp CC BY 4.0 |
| CCIR/URSI ionospheric maps | PyIRI (MIT) at commit c0a50e0a, each file pinned by SHA-256 | ITU-R / URSI published coefficients |
| IGRF-14 | IAGA V-MOD via NOAA NCEI, pinned by SHA-256 | IAGA published coefficients |
| GNSS TEC maps (validation only) | NASA CDDIS IONEX (Earthdata login) | public |
| CODE final GIMs (storm-time fit input; validation) | AIUB, University of Bern, via `zhw-b.s3.cloud.switch.ch/aiub/CODE/` | open, cite Dach et al.; only woof's fitted coefficients are shipped |
| Ionosonde foF2 (validation only) | GIRO / Lowell DIDBase FastChar | CC BY-NC-SA 4.0: not shipped, used for grading only |
| CMIP6 solar forcing (EPP ion-pair rates) | input4MIPs SOLARIS-HEPPA | CC BY 4.0 |
| CMIP7 solar forcing v4.6 (proton ion-pair rates `iprp`; validation only) | input4MIPs SOLARIS-HEPPA | CC BY 4.0 |
| GOES integral proton flux, real time (`json/goes/primary/integral-protons-3-day.json`) | NOAA SWPC | public domain |
| GOES integral proton flux, historical (OMNI2 words 43-48) | NASA SPDF OMNIWeb | public |
| Auroral electron ionisation fit (shipped, 32 numbers) | Fang et al. 2010, GRL 37, L22106, Table 1 | published coefficients |

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
- OVATION Prime 2010 is Newell et al. (2009, 2010), distributed by NOAA NGDC/NCEI (ovation-prime
  2.3.0: P. Newell, JHU/APL; R. Redmon and J. Machol, NOAA). The POES TED data are from NOAA NCEI.

NRLMSIS 2.x is not freely licensed (NRL restricts commercial use), so it is not bundled.
Neither are the copyleft ports of OVATION (OvationPyme, LGPL) and of Weimer. The aurora layer
ports NOAA's public-domain OVATION IDL code instead. The thermosphere layer will use a clean-room
implementation of published formulas.

## Validation plan

The ionosphere layers will be checked against observations:
- GNSS TEC from the IGS/CODE/JPL global ionosphere maps at NASA CDDIS;
- foF2 and hmF2 from GIRO ionosondes;
- the storm of 10-12 May 2024, with 1-3 May 2024 as the quiet reference.

The proton ion-pair rates are checked against the CMIP7 rates (above). The chemistry source
(`spacewx_epp_opt`) is too small at WRF tops for MLS/ACE NOx and ozone to see (above); the
auroral electron rates are checked against the transport models Fang et al. 2010 fitted. L1 is checked by scaling
identity tests on CUDA and Metal, L1b by band-factor identity and linearity tests and against
SORCE SSI (above). L1's bias direction can be checked against NSRDB / source.coop
surface solar. Its 0.07-0.6 % signal is below the noise of single-station observations.
