# Wind farm power: `woof energy power`

`woof energy power` turns a `woof-energy.forecast.v1` file into power at
each wind farm, using turbine-specific power curves, air-density
correction, high-wind shutdown with hysteresis, wake losses and
electrical losses. It reads the forecast of a hindcast (a year-long
reanalysis, in shards) or of a forecast in exactly the same way, and works
on any time step: 15-minute, 10-minute or finer.

```bash
woof energy power forecast.nc --assets assets.geojson -o power.nc
woof energy power forecast.nc --assets assets.geojson --per-turbine --matches matches.json -o power.nc
woof energy power forecast.nc --assets assets.geojson --wake-model bastankhah --gust-shutdown --icing-loss products.nc -o power.nc
```

`woof energy rating --products wind-power` remains the quick product: one
generic curve for every turbine and no wakes. `woof energy power` is the
turbine-aware one.

## Inputs

| Input | Used for |
|---|---|
| `forecast.v1` `wind_speed` | Hub-height or rotor-equivalent wind |
| `air_density`, or `air_pressure` and `air_temperature` | IEC density correction (dry air from p and T, noted) |
| `u` and `v`, or `wind_from_direction` | Wind direction for the wake tables (without it the tables are averaged over direction, noted) |
| A `(time, site)` gust variable in the forecast file | `--gust-shutdown` (`--gust-var NAME`, or the first of `wind_speed_of_gust`, `wind_gust`, `gust`, `wind_gust_10m`), used as given |
| `--assets ASSETS.geojson` | Turbine make and model, rating, rotor, hub height, IEC class, offshore flag, plant membership and positions |
| `--icing-loss FILE.nc[:VAR]` | A `(time, site)` loss fraction in [0, 1] on the forecast's axes (default variable `icing_loss_fraction`) |

From each asset, `woof energy power` reads `capacity_mw` (one turbine's
rating for a generator, the plant total for a plant), `rotor_diameter_m`,
`hub_height_m` and these tags, first match wins:

| What | Tags |
|---|---|
| Manufacturer | `manufacturer`, `generator:manufacturer`, `turbine:manufacturer`, `t_manu`, `turbine_manufacturer` |
| Model | `model`, `generator:model`, `turbine:model`, `t_model`, `turbine_model`, `generator:type` |
| Rating (when `capacity_mw` is absent) | `generator:output:electricity` (`3 MW`, `3000 kW`), `t_cap`, `rated_power`, `turbine:rated_power` |
| Rotor diameter, hub height | `rotor:diameter`, `rotor_diameter`, `t_rd`; `height:hub`, `hub_height`, `t_hh` |
| IEC class | `iec_class`, `wind_class_iec`, `turbine:iec_class` |
| Offshore | `offshore` (`yes`/`no`), `location` or `plant:type` (`offshore`/`onshore`) |
| Plant | `plant`, `plant_id`, `plant_ref`, `p_name`, `project`; else the wind plant polygon containing the turbine; else the turbine is its own plant |
| Turbine count of a plant-level asset | `generator:count`, `turbine_count`, `turbines`, `p_tnum`, `n_turbines` |

Without `--assets`, every turbine is matched from the forecast's
`capacity_mw` alone (a generic curve) and is its own plant.

## The turbine library

`woof/data/energy/turbine_library.json` (`woof-energy.turbine-library.v1`)
holds about 200 machines: the 140 rows of the Open Energy Platform Wind
Turbine Library as snapshotted by windpowerlib (67 with a manufacturer power
curve), plus about 60 common models it lacks (GE 1.5 to 2.82 MW,
Haliade-X, Vestas V47 to V236, Siemens Gamesa SG, Gamesa G, Mitsubishi
MWT, Suzlon, Goldwind, Clipper, Enercon E-40 to E-138) with their
manufacturer-published rating and rotor diameter. Curves are kW against
wind speed at 1.225 kg/m3. An entry whose curve is not published carries a
parametric curve at its specific power and says so (`curve_source`
`parametric`):

`P / P_rated = s / (1 + s^8)^(1/8)`, `s = 0.5 rho0 cp v^3 / SP`, `cp = 0.45`,
zero below 3 m/s, cut-out 25 m/s.

Licence: the library is a derived database of the OEP Wind Turbine Library,
(c) Reiner Lemoine Institut, under ODbL-1.0, and is itself ODbL-1.0. The
attribution travels in the file and in every `power.v1` output
(`turbine_library_license`, `turbine_library_attribution`).

Regenerate it with `python -m tools.build_turbine_library --fetch` (the two
upstream CSVs at a pinned windpowerlib commit) or, offline,
`--upstream-dir DIR`; `--check` compares with the committed file. The
script corrects a rating that disagrees with the type suffix, clips curve
points above the rating, drops the zeros after cut-out (cut-out is applied
with hysteresis at run time), and records each correction in the entry's
`notes`.

`--library FILE.json` replaces the built-in library with your own document
of the same schema, for example manufacturer curves under NDA.

## Matching a turbine to a curve

Each turbine walks a fixed ladder; the rung that answered is
`match_quality`, and `--matches` writes the reason for every turbine
(`woof-energy.turbine-matches.v1`).

| `match_quality` | Rule |
|---|---|
| `exact` | Manufacturer and model, lower-cased and reduced to letters and digits (a trailing `MW`/`kW` dropped), equal one of an entry's names or aliases, with a compatible manufacturer (`Siemens Gamesa Renewable Energy`, `SGRE`, `Gamesa` and `Bonus` are one manufacturer; so are `GE Wind`, `General Electric` and `Alstom`; see `manufacturer_aliases` in the library) |
| `fuzzy` | The model string parses to the same rotor diameter (within 1.5 m) and rating (within 3 %, or the asset's rating within 10 % when the model names none), with a compatible family prefix (`SWT3.6/120 DD` is `SWT120/3600`); or a string similarity of at least 0.85 within the manufacturer |
| `size` | Rating within 15 % and rotor within 7 % of an entry; nearest in `(ln P / 0.10, dD / D / 0.05)`, with penalties of 0.5 for an IEC class or onshore/offshore disagreement and 0.25 for a parametric curve |
| `generic` | A parametric curve at the asset's specific power when its rotor is known; else the generic class curve (`iec_I` 400, `iec_II` 320, `iec_III` 250, `offshore` 350 W/m2; `iec_II` onshore by default), scaled to the asset's rating |
| `none` | No rating anywhere; power is NaN and noted |

Ties go to the nearest rating, then a measured curve, then the entry id, so
the result is deterministic.

The turbine's own rating (`capacity_mw`) scales the curve's shape; the
library rating is used only when the asset has none.

A plant known only by its outline and capacity stands for `count`
identical turbines: the count from a count tag, else from the matched
model's rating, else from a typical unit of 3000 kW onshore or 8000 kW
offshore, and the unit rating is `capacity / count`.

## Per-turbine power

1. **Inflow.** Hub-height wind by linear interpolation in ln(height). When
   at least 3 forecast heights lie inside the rotor disc, the IEC
   61400-12-1:2017 rotor-equivalent wind speed
   `(sum_i A_i / A v_i^3)^(1/3)` is used instead: each height owns the band
   of the disc between the midpoints to its neighbours, the outermost
   bands reaching the disc edge. `--hub-only` turns this off; the
   `speed_method` coordinate says which was used.
2. **Wakes**, below.
3. **Density.** `v_eq = v (rho / 1.225)^(1/3)` (IEC 61400-12-1, pitch
   regulated) on the waked speed, then the curve, linear between points.
4. **High-wind shutdown with hysteresis.** A turbine stops when its
   free-stream speed exceeds cut-out (the curve's last powered speed), or
   with `--gust-shutdown` when the gust exceeds cut-out + 7 m/s, and
   restarts only when the speed falls below cut-out - 3 m/s. The state is
   carried along time; a turbine is running at the first step unless that
   step is a shutdown. Free-stream speed decides, because above rated the
   thrust, and so the wake, is small.
5. **Losses.** `power = waked x availability x (1 - icing loss) x
   (1 - electrical losses)`; defaults `--availability 1` and
   `--electrical-losses 0.02`. NaN icing loss is no loss, counted in the
   notes.

## Wakes

A plant with two or more turbines at known point positions gets a
directional wake table: for 72 directions (5 degrees) and free-stream
speeds 0 to 30 m/s, each turbine's waked / free speed ratio, computed
upstream to downstream with each turbine's thrust coefficient taken from
its own curve at its own waked speed, deficits combined as the root sum of
squares. The table is smoothed over direction by a Gaussian of 5 degrees
and read bilinearly at the plant's mean free-stream speed and circular-mean
hub direction at every step.

| `--wake-model` | Deficit behind a turbine of radius R at distance x |
|---|---|
| `jensen` (default) | Jensen / Katic PARK top hat `(1 - sqrt(1 - Ct)) (R / (R + k x))^2` times the fraction of the downstream rotor inside the wake; `k` 0.075 onshore, 0.04 offshore |
| `bastankhah` | Bastankhah and Porte-Agel (2014) Gaussian, `sigma / D = k* x / D + 0.2 sqrt(beta)`; `k*` 0.04 onshore, 0.025 offshore |
| `none` | No wakes |

The thrust coefficient comes from 1-D momentum theory on the curve:
`Cp = P / (0.5 rho0 A v^3 0.94 0.80)` (drivetrain efficiency 0.94, a real
rotor's peak Cp 0.80 of Betz), `4 a (1 - a)^2 = Cp`, `Ct = 4 a (1 - a)`;
about 0.85 below rated, falling above rated.

A plant-level row, or a plant whose turbines lack positions, uses a
fallback wake loss `L` by turbine count, applied as a uniform speed deficit
`1 - (1 - L)^(1/3)` so that `L` is the loss in the cubic part of the curve
and vanishes above rated:

| Turbines | Wake loss L |
|---|---|
| 1 | 0 |
| 2 to 5 | 0.03 |
| 6 to 20 | 0.06 |
| 21 to 50 | 0.09 |
| 51 to 100 | 0.11 |
| more | 0.13 |

Offshore plants add 0.02 (lower ambient turbulence, slower wake recovery).
These are engineering rules of thumb in the range reported for onshore
(5 to 15 %) and offshore (10 to 20 %) farms, not a model of any layout.

Building a table is O(N^2) in the plant's turbines per direction and speed
bin; a farm of a few hundred turbines takes seconds.

## `woof-energy.power.v1` (`power.nc`)

A netCDF file with dimensions `time` (the forecast's), `plant` and
`turbine`. A `turbine` row is a turbine site, or a plant-level site
standing for `count` identical turbines. Plants sum their rows; a NaN row
(no hub height, no rating, or no forecast data) makes the plant NaN at that
step and is noted, never dropped silently.

| Variable | Dimensions | Units | Meaning |
|---|---|---|---|
| `power_mw` | time, plant | MW | net plant power |
| `gross_power_mw` | time, plant | MW | free-stream plant power without losses |
| `wake_loss_mw` | time, plant | MW | power lost to wakes |
| `capacity_factor` | time, plant | 1 | net power over capacity |
| `hub_wind_speed` | time, plant | m s-1 | mean free-stream hub-height wind speed |
| `running_fraction` | time, plant | 1 | capacity fraction not in high-wind shutdown |
| `turbine_power_mw` | time, turbine | MW | net turbine-row power |
| `turbine_wind_speed` | time, turbine | m s-1 | free-stream hub or rotor-equivalent wind speed |
| `turbine_waked_wind_speed` | time, turbine | m s-1 | waked wind speed |
| `turbine_equivalent_wind_speed` | time, turbine | m s-1 | waked density-corrected wind speed |
| `turbine_running` | time, turbine | 1 | 1 running, 0 in high-wind shutdown |

The `turbine_*` series are written only with `--per-turbine`.

| Coordinate | Dimensions | Units | Meaning |
|---|---|---|---|
| `time` | time | - | valid time (UTC) |
| `plant_id` | plant | - | plant identifier |
| `plant_name` | plant | - | plant name |
| `plant_lat` | plant | degree_north | plant mean latitude |
| `plant_lon` | plant | degree_east | plant mean longitude |
| `capacity_mw` | plant | MW | sum of turbine ratings |
| `n_turbines` | plant | 1 | turbines in the plant |
| `offshore` | plant | 1 | 1 offshore, 0 onshore, -1 unknown |
| `wake_method` | plant | - | wake model applied |
| `fallback_wake_loss` | plant | 1 | table wake loss (NaN unless used) |
| `turbine_id` | turbine | - | turbine row identifier |
| `site_id` | turbine | - | forecast site |
| `asset_id` | turbine | - | asset identifier |
| `plant_index` | turbine | 1 | index on the plant dimension |
| `count` | turbine | 1 | turbines the row stands for |
| `turbine_lat` | turbine | degree_north | latitude |
| `turbine_lon` | turbine | degree_east | longitude |
| `rated_mw` | turbine | MW | rating of one turbine |
| `rotor_diameter_m` | turbine | m | rotor diameter |
| `hub_height_m` | turbine | m | hub height |
| `curve_id` | turbine | - | power curve used |
| `curve_source` | turbine | - | oedb, parametric or generic |
| `match_quality` | turbine | - | exact, fuzzy, size, generic or none |
| `match_score` | turbine | 1 | match score |
| `speed_method` | turbine | - | hub or rews |

Global attributes bind the inputs (`forecast_sha256`, `assets_sha256`,
`turbine_library_sha256`) and record every option, the library licence and
attribution, and the notes. `woof.energy.power_farm.load_power` and
`load_matches` check a file against these tables.

The printed summary gives, per plant, the capacity, turbine count, wake
method, mean power, mean capacity factor, energy (MWh, each step held to
the next) and the wake loss fraction, plus the count of turbines at each
match quality.

## Refusals

`woof energy power` exits 2 with a `refused` message, writing nothing, when
the forecast is not `forecast.v1`, lacks `wind_speed` or any density
source, has no wind sites, when the assets document does not hold a wind
site's asset, when a requested gust or icing variable is missing, is not on
the forecast's `(time, site)` axes or (icing) lies outside [0, 1], or when
the library is malformed.

## What is not modelled

Turbulence-intensity or shear corrections of the curve beyond REWS,
curtailment, grid limits, wake-induced turbulence, blockage, deep-array
effects beyond the root-sum-square superposition, wakes between
neighbouring plants, and storm-control ramps (curves stop at cut-out).
