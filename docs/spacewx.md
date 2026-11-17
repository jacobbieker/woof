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
| L3a ionosphere diagnostic | IRI-type F2/E peaks (CCIR/URSI maps driven by F10.7), Epstein profile, vertical TEC, foF2, hmF2, on the GPU | available as `woof spacewx ionosphere` (global grid); forecast history output planned |
| L3b high latitudes | auroral oval energy flux and Joule heating from Kp and the solar wind (OMNI, SWPC real time) | planned |
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

## Data sources and licences (all layers)

| input | source | licence |
|---|---|---|
| TSI, SSI | NOAA NCEI CDR (NRLSSI2) | NOAA CDR, open |
| Kp, ap, Ap, F10.7 | GFZ Potsdam (`kp.gfz.de`) | CC BY 4.0; the sunspot column (CC BY-NC) is not read |
| F10.7, solar wind real time | NOAA SWPC | public domain |
| solar wind historical | NASA SPDF OMNIWeb | public |
| CCIR/URSI ionospheric maps | PyIRI (MIT) at commit c0a50e0a, each file pinned by SHA-256 | ITU-R / URSI published coefficients |
| IGRF-14 | IAGA V-MOD via NOAA NCEI, pinned by SHA-256 | IAGA published coefficients |
| GNSS TEC maps (validation only) | NASA CDDIS IONEX (Earthdata login) | public |
| CMIP6 solar forcing (EPP ion-pair rates) | input4MIPs SOLARIS-HEPPA | CC BY 4.0 |

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
