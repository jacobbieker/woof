# Ramps and wind-resource products

`woof energy rating` has two wind products besides `wind-power`:

- `wind-resource` gives the rotor-equivalent wind speed, wind shear, veer,
  turbulence intensity, generic-curve power and capacity factor at every
  turbine and wind-farm site.
- `ramps` finds ramp events in the power of those sites, at any time step.

Neither runs by default. Ask for them with `--products`:

```bash
woof energy rating hindcast/forecast.nc --products wind-power,wind-resource,ramps -o hindcast/wind-products.nc
```

The same calculations are available as NumPy and xarray functions in
`woof.energy.ramps`, so you can run them on a reanalysis store without a
`forecast.v1` file. The xarray functions are `ramps_dataset` and
`wind_resource_dataset`. They use `xarray.apply_ufunc(dask="parallelized")`,
so Icechunk- or dask-backed data is loaded lazily wherever a quantity
depends on only one time step. The ramp event table and the resolved
turbulence intensity need each site's whole time series.

## Which sites

These products use the same sites as `wind-power`: sites of kind `turbine`,
and `plant` sites that have a `hub_height_m`. A site without a hub height
gets NaN, and the notes say so.

**Rotor diameter.** The rotor diameter comes from the forecast's
`rotor_diameter_m` coordinate when there is one. `woof energy extract` does
not write that coordinate, so the default for a site without one is the
`--rotor-diameter-m` value. If that is not given either, the rotor diameter
is taken to be the hub height. On current turbines the hub height is
0.8 to 1.2 times the rotor diameter. The notes count how many sites used an
assumed diameter.

If the rotor would reach the ground (diameter at least twice the hub
height), every rotor quantity for that site is NaN.

## Definitions

**Rotor strips.** The rotor disc is divided into 10 horizontal strips of
equal height. Each strip takes the wind at its centre height, interpolated
linearly in ln(height) between the forecast heights. A strip outside the
forecast heights takes the nearest one, and the notes report this. Each
strip is weighted by its exact share of the disc area.

**Rotor-equivalent wind speed** (`rotor_equivalent_wind_speed`) follows
IEC 61400-12-1:2017, equation 4:

    REWS = (sum_i (v_i cos phi_i)^3 A_i / A)^(1/3)

Here `phi_i` is the wind direction of strip `i` relative to the wind at hub
height. It is taken from `u` and `v`, or from `wind_from_direction` when
those are absent. With no direction at all, `cos phi = 1`. `cos phi` is
never allowed below 0.

**Shear exponent** (`shear_exponent`) is `alpha` in
`U(z) = U_ref (z / z_ref)^alpha`. It is the least-squares slope of ln U
against ln z, fitted over the forecast heights between the lower and upper
rotor tips. If fewer than two forecast heights lie on the rotor, the two
heights nearest the hub (in ln z) are used instead. Heights where the wind
is 0 or less are left out of the fit.

**Veer** (`wind_veer`) is the wind direction at the upper tip minus the
direction at the lower tip, in degrees between -180 and 180. A positive
value means the wind turns clockwise with height (veering). It is NaN when
either tip is calm or the forecast has no wind direction.

**Turbulence intensity** (`turbulence_intensity`) is `sigma_u / U` at hub
height. It is NaN when the mean wind is below 1 m/s. It can come from two
sources:

- **TKE**, when the data carry it: `sigma_u = sqrt(2/3 TKE)`, assuming
  isotropic turbulence. In WRF this is the MYNN or UW scheme's subgrid
  `TKE_PBL`. Only `wind_resource_dataset` can use this route, because
  `forecast.v1` has no TKE variable.
- **Resolved**, otherwise: the standard deviation of the hub-height wind
  over a trailing window (`--ti-window-min`, default 10 minutes, at least
  3 samples), divided by its mean. This captures only the variance the
  model resolves at the output time step.

From a `forecast.v1` file the product is therefore NaN, with a note, unless
the time step gives at least 3 samples per window. A 15-minute hindcast does
not; 1- to 5-minute output does.

**Power** (`rews_power_mw`) is the normalised generic power curve evaluated
at the density-corrected REWS, `REWS (rho / 1.225)^(1/3)`, times
`capacity_mw`. The density correction is the IEC 61400-12-1 rule for
pitch-regulated turbines, with `rho` at the hub. Choose the curve with
`--power-curve`.

When a `turbine` site has both `capacity_mw` and `rotor_diameter_m`, the
curve is stretched between cut-in and rated speed so that it reaches rated
power at `(2 P / (A rho Cp))^(1/3)`, with `Cp = 0.45`. This is the
specific-power scaling the packaged curves follow. Cut-in and cut-out do not
change. No wake, availability, curtailment or turbulence losses are applied;
turbine-specific curves and wakes are separate work.

**Capacity factor** (`capacity_factor`) is the mean of `rews_power_mw` over
the times where it is finite, divided by `capacity_mw`.

## Packaged power curves

All four curves use the same cubic form between cut-in and rated speed:
`P/P_rated = (v^3 - v_in^3) / (v_rated^3 - v_in^3)`. Each is a generic
curve for its class, not the measured curve of the reference turbine.

| Curve | Cut-in, rated, cut-out (m/s) | Implied specific power (W/m2) | Source of the speeds |
|---|---|---|---|
| `generic_iec_class_i` | 3, 11.4, 25 | 408 | NREL 5 MW reference turbine (Jonkman et al. 2009, NREL/TP-500-38060), IEC class IB |
| `generic_iec_class_ii` | 3, 12, 25 | - | Textbook IEC class II curve (the default) |
| `generic_iec_class_iii` | 4, 9.8, 25 | 259 | IEA 3.4 MW onshore reference turbine (Bortolotti et al. 2019, NREL/TP-5000-73492), IEC class IIIA |
| `generic_offshore` | 3, 10.59, 25 | 327 | IEA 15 MW offshore reference turbine (Gaertner et al. 2020, NREL/TP-5000-75698), IEC class IB |

The implied specific power is `rho Cp v_rated^3 / 2` with `Cp = 0.45`. It is
within 3 % of each reference turbine's actual rated power over swept area.

## Ramps

The ramp signal (`ramp_signal`) is `rews_power_mw / capacity_mw`: power as a
fraction of capacity. It needs no capacity, so plant sites without one still
get ramps. Thresholds are fractions of capacity. Two definitions run side by
side.

### Threshold over a window

The windows default to 15 minutes, 1 hour and 4 hours, and the threshold to
0.2 of capacity. Set them with `--ramp-windows 15m,1h,4h` and
`--ramp-threshold 0.2`. To give one window its own threshold, write
`--ramp-windows 5m:0.1,1h:0.3`. A bare number means minutes; `s`, `m` and
`h` are accepted as units.

For each window `W` with threshold `X`, detection works like this:

1. Compute `ramp_delta(t) = P(t) - P(t - W)`. `P(t - W)` is interpolated
   linearly in time, so the time step does not need to divide `W`.
2. Each time where `|ramp_delta| >= X` marks the interval `[t - W, t]`. The
   interval starts at the sample at or before `t - W`.
3. Overlapping intervals in the same direction are merged.
4. Each merged interval is trimmed to its largest rise (up ramp) or fall
   (down ramp). The event ends at the first sample of that largest change,
   and starts at the last sample of the extreme it leaves.

Because of the merging, an event can last longer than `W`, for example when
a long ramp chains several windows together.

In a `woof energy rating` run, a window shorter than the forecast time step
is skipped, and the notes say so. If every window is too short, the product
is refused. The library function `detect_ramps` refuses any window shorter
than the median time step.

### Swinging door

This follows Florita, Hodge and Orwig (2013, NREL/CP-5500-58060), using the
swinging-door algorithm of Bristol (1990):

1. Segment the signal into straight lines through retained samples. Each
   line stays within `--sda-tolerance` (default 0.05 of capacity) of the
   samples it covers. A NaN ends a segment.
2. Merge consecutive segments whose slope has the same sign and a rate of at
   least 0.05 per hour.
3. A merged run is a ramp when its total change is at least
   `--ramp-threshold`.
4. Adjust the ends. A swinging door retains a sample only once the door has
   closed, a little after the true corner. So each end is first moved inward
   while the signal does not advance in the ramp direction (this trims
   plateaus), then outward while it strictly does.

No maximum duration is imposed.

### Per-timestep outputs and the event table

`ramp_flag` (one layer per window) and `ramp_flag_sda` are +1 inside an up
event, -1 inside a down event and 0 elsewhere. Where an up and a down event
overlap, the larger one wins. `ramp_rate` is the rate of change from the
previous sample, per hour.

Every event is one row along the `ramp_event` dimension.
`woof.energy.contracts.load_ramp_events(path_or_dataset)` reads the table
from a products file or from a `ramps_dataset` result and validates every
row. A file with no ramps has a `ramp_event` dimension of length zero, which
netCDF stores as unlimited.

**Time step.** Ramps are defined on whatever time step the signal has. A
15-minute series locates a ramp's start and end to within 15 minutes. To
time fast ramps, or to see the 15-minute window properly, write the
hindcast at 1 to 5 minutes. The definitions do not change.

## Product variables

| Variable | Dimensions | Units | Meaning |
|---|---|---|---|
| `rotor_equivalent_wind_speed` | time, site | m s-1 | rotor-equivalent wind speed |
| `shear_exponent` | time, site | 1 | power-law wind shear exponent across the rotor |
| `wind_veer` | time, site | degree | wind direction change from lower to upper rotor tip |
| `turbulence_intensity` | time, site | 1 | hub-height turbulence intensity |
| `rews_power_mw` | time, site | MW | generic-curve power at the density-corrected REWS |
| `capacity_factor` | site | 1 | mean rews_power_mw over capacity_mw |
| `ramp_window` | ramp_window | min | threshold ramp window |
| `ramp_threshold` | ramp_window | 1 | threshold ramp magnitude, fraction of capacity |
| `ramp_signal` | time, site | 1 | generic-curve power as a fraction of capacity |
| `ramp_rate` | time, site | h-1 | sample-to-sample rate of the ramp signal |
| `ramp_delta` | time, site, ramp_window | 1 | ramp signal change over the window ending at each time |
| `ramp_flag` | time, site, ramp_window | 1 | threshold ramp flag: +1 up, -1 down, 0 none |
| `ramp_flag_sda` | time, site | 1 | swinging-door ramp flag: +1 up, -1 down, 0 none |

## Ramp event table

| Column | Units | Meaning |
|---|---|---|
| `ramp_event_site_id` | - | site identifier |
| `ramp_event_method` | - | threshold or swinging_door |
| `ramp_event_window` | min | threshold window W; NaN for swinging_door |
| `ramp_event_direction` | 1 | +1 up ramp, -1 down ramp |
| `ramp_event_start` | time | sample time the ramp leaves its extreme |
| `ramp_event_end` | time | sample time the ramp reaches its extreme |
| `ramp_event_magnitude` | 1 | absolute signal change, end minus start |
| `ramp_event_duration` | min | end minus start |
| `ramp_event_max_rate` | h-1 | largest sample-to-sample rate in the ramp direction |

In a products file, start and end are stored as numbers in the forecast's
time units. In `ramps_dataset` they are `datetime64` values. Magnitude and
rate are in the units of the signal: a fraction of capacity in a products
file, or the input's own units when `ramps_dataset` is called without a
capacity (m/s for a hub-height wind series, for example).

## Reusing the helpers

These functions are public and work on NumPy arrays:

- `density_corrected_speed(speed, density)`
- `curve_power_fraction(speed, curve, rated_ms=None)`
- `rated_speed_for_specific_power(sp)` and `specific_power(capacity_mw, rotor_diameter_m)`
- `rotor_segments(hub, diameter, n)`
- `rotor_equivalent_wind_speed(speed, heights, hub, diameter, u=, v=)`
- `shear_exponent`, `wind_veer`, `turbulence_intensity_from_tke` and `resolved_turbulence_intensity`
- `capacity_factor`
- `detect_ramps(signal, time_s, windows=, threshold=, sda_tolerance=)`

Turbine-specific power code can call them directly. For example, to find
ramps in a wake-corrected farm power series, pass that series to
`detect_ramps` or `ramps_dataset`.
