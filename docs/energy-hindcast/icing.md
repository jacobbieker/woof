# Turbine icing at hub height

`woof energy rating --products turbine-icing` estimates icing on wind
turbines at their hub height. The `icing` product does not do this: it
accretes ice on line conductors and, at every other site, on the ISO 12494
30 mm reference collector at 10 m. A turbine ices at its hub and on its
rotating blades, so this product reads the weather at each turbine's
`hub_height_m` and accretes ice on two collectors there.

The same model runs on gridded model output, such as a year of wrfout files
or an Icechunk hindcast store, through `turbine_icing_dataset` (xarray in,
xarray out). The site path and the gridded path share one array kernel. A
grid point and a site with the same weather get the same numbers.

The code is `woof/energy/icing_turbine.py`. Every constant on this page is
a module constant, and each output variable's attributes name its method
and assumptions.

## What it reports

Each time-series variable has dimensions `(time, site)` in a products file,
or `(time, ...)` on a grid.

| Variable | Units | Meaning |
|---|---|---|
| `turbine_lwc` | kg m-3 | Liquid water content at hub height: cloud water, plus rain and drizzle when the forecast carries `rain_mixing_ratio` |
| `turbine_droplet_mvd` | m | Median volume diameter of the cloud droplets (NaN without cloud water) |
| `turbine_meteorological_icing` | flag | Meteorological icing: air below 0 degC and LWC above 0.01 g m-3 |
| `turbine_blade_relative_speed` | m s-1 | Air speed seen by the blade-proxy section |
| `turbine_reference_ice_mass` | kg m-1 | Ice on the ISO 12494 reference collector (30 mm cylinder) at hub height |
| `turbine_reference_icing_rate` | kg m-1 s-1 | Accretion rate on the reference collector |
| `turbine_instrumental_icing` | flag | Instrumental icing: at least 10 g/m of ice on the reference collector |
| `turbine_iso12494_class` | flag | ISO 12494 rime class R1 to R10 of the reference-collector ice |
| `turbine_blade_ice_mass` | kg m-1 | Ice on the blade-proxy section |
| `turbine_blade_icing_rate` | kg m-1 s-1 | Accretion rate on the blade-proxy section |
| `turbine_rotor_icing` | flag | Rotor icing: at least 10 g/m of ice on the blade proxy |
| `turbine_icing_loss_fraction` | 1 | Fraction of the un-iced power lost to blade ice |

Flags are `int8`: -1 is undefined (unknown weather, or not a turbine), 0 is
no and 1 is yes.

Each per-site (or per-grid-point) summary covers the whole period.

| Variable | Units | Meaning |
|---|---|---|
| `turbine_meteorological_icing_fraction` | 1 | Share of the period with meteorological icing |
| `turbine_instrumental_icing_fraction` | 1 | Share of the period with instrumental icing |
| `turbine_rotor_icing_fraction` | 1 | Share of the period with rotor icing |
| `turbine_iea_ice_class` | flag | IEA Wind Task 19 ice class, 1 to 5 (-1 undefined) |
| `turbine_icing_energy_loss_fraction` | 1 | Share of the period's un-iced energy lost to blade ice |

The gridded output also carries the end-of-period state
(`turbine_reference_ice_mass_end`, `turbine_reference_ice_volume_end`,
`turbine_blade_ice_mass_end`, `turbine_blade_ice_volume_end`). You pass it
back to continue a chunked hindcast.

The fractions count output times, so they are durations only when the
output interval is regular. The reanalysis writes every 15 minutes, so they
are.

## Inputs

| `forecast.v1` variable | Needed | Use |
|---|---|---|
| `air_temperature`, `wind_speed`, `cloud_liquid_mixing_ratio` | yes | Accretion |
| `air_density` | yes, or `air_pressure` | Accretion. Without it, density is `p / (R_d T)` for dry air, and a note says so |
| `hub_height_m` (coordinate) | yes | Where the weather is read |
| `rain_mixing_ratio` | no | Freezing rain and drizzle. Without it, LWC is cloud water only |
| `cloud_number_mixing_ratio` | no | Thompson `QNCLOUD`, for the droplet size. Without it, a default droplet number is used |
| `air_pressure` | no | Evaporative cooling, and sublimation |
| `relative_humidity` | no | Ambient vapour pressure, for sublimation. Without it, the air is taken as saturated |

Weather is read at `hub_height_m`. Wind is interpolated linearly in
ln(height) and everything else linearly in height, between the bracketing
forecast heights. Outside the forecast's height range the nearest height is
used, and a note says so. Turbines are sites of kind `turbine`, and `plant`
sites that carry a hub height. A turbine with no hub height is undefined,
and a note says so.

### The droplet number: `QNCLOUD`

`cloud_number_mixing_ratio` is new in `forecast.v1`. `woof energy extract`
writes it from `QNCLOUD` when the history carries it. Thompson
aerosol-aware microphysics (`mp_physics = 28`) does. Thompson
`mp_physics = 8` and WSM6 carry cloud water mass only, so the variable is
not written and the product uses the default droplet number.

The `energy` history preset (`[output] preset = "energy"`) does not include
`QNCLOUD`. Presets and `history_vars` are exclusive. To keep `QNCLOUD`,
list the preset's fields plus `QNCLOUD` in `[output] history_vars`.

## The model

### Liquid water content

The cloud liquid water content is `w_c = rho q_c`. Rain and drizzle add
`w_r = rho q_r`. The reported LWC is `w_c + w_r`.

### Droplet median volume diameter

The cloud droplets follow Thompson's gamma spectrum
`N(D) = N0 D^mu exp(-lambda D)`. Its shape is
`mu = clip(nint(1e9 / N_c) + 2, 2, 15)`, with `N_c` in m-3 (Thompson and
Eidhammer 2014). The liquid water content fixes the slope:

    lambda = [ (pi rho_w / 6) (mu + 1)(mu + 2)(mu + 3) N_c / w_c ]^(1/3)

The median volume diameter is the median of the mass spectrum, a gamma of
shape `mu + 4`, so `MVD = (mu + 3.67) / lambda`. It is clipped to
5 to 50 um.

`N_c = rho QNCLOUD` where the run carries the droplet number. Otherwise the
default `N_c = 1e8 m-3` is used, as it is at any cloudy point where
`QNCLOUD` is zero or missing (a note says so). That is the fixed droplet number of
Thompson `mp_physics = 8`, which gives `mu = 12`. For example,
`w_c = 0.25 g m-3` at the default number gives an MVD of 18.9 um.

### Accretion

Ice grows by the Makkonen (2000) accretion equation, per metre of
collector:

    dM/dt = a1 a2 a3 w v D

- `a1` for cloud droplets is the Finstad, Lozowski and Gates (1988)
  collision efficiency at the droplet MVD, the fit the `icing` product uses.
  Rain drops are larger than 100 um, so they are collected with `a1 = 1`.
  They hit at `sqrt(v^2 + 4^2)` m/s, with an assumed fall speed of 4 m/s
  (Jones 1998).
- `a2 = 1`.
- `a3` is the frozen fraction from the Makkonen (2000) heat balance at a
  0 degC surface. Heat leaves the surface by convection, `h pi D (0 - t_a)`,
  and by evaporation, `h pi D eps L_v (e_s(0) - e_a) / (c_p p)`. Evaporation
  needs `air_pressure`. Warming the collected water also takes heat,
  `F c_w (0 - t_a)`. Against these are the aerodynamic heating,
  `h pi D r v^2 / (2 c_p)` with recovery factor `r = 0.79`, the droplets'
  kinetic energy, `F v^2 / 2`, and the latent heat of freezing, `F L_f`.
  The heat-transfer coefficient is `h = Nu k_a / D` with Makkonen's
  `Nu = 0.032 Re^0.85`. Radiation is neglected.

Dry growth (`a3 = 1`) is stored as rime at 500 kg m-3, and wet growth as
glaze at 900 kg m-3. The collector diameter grows with the accreted
volume.

### Removal

- **Melting.** When the heat balance has a surplus, the surface cannot stay
  at 0 degC, and the surplus melts ice at `surplus / L_f`. This happens in
  air above freezing, and on a fast blade just below freezing, where
  aerodynamic heating wins.
- **Sublimation.** Below freezing, ice sublimates at
  `h pi D eps (e_si(t_a) - e_a) / (c_p p)`. This is the Chilton-Colburn
  analogy, with a Lewis number of 1 and the surface at air temperature. It
  is never negative, and it needs both `air_pressure` and
  `relative_humidity`. Only a frozen surface sublimates. During wet growth
  the surface is liquid at 0 degC, and its evaporation is already in the
  heat balance.
- **Shedding.** There is no mechanical shedding model. On a rotating blade
  this overstates how long ice stays, so rotor-icing durations and losses
  are upper bounds.

### Two collectors at hub height

**The reference collector** is the ISO 12494 30 mm cylinder, in the hub
wind. Its ice is IEA Wind Task 19's *instrumental icing*: ice present on a
structure or instrument. Here "present" means at least 10 g/m, about
0.2 mm of rime.

**The blade proxy** stands in for the leading edge of the outer blade. It
is a 0.10 m cylinder at 85 % of the rotor radius. While the turbine runs,
between the power curve's cut-in and cut-out speeds:

- the tip speed is `min(7 v, 80 m/s)`, a tip-speed ratio of 7 with an
  80 m/s tip-speed limit;
- the section sees `sqrt((0.85 v_tip)^2 + ((1 - a) v)^2)`, with the Betz
  axial induction `a = 1/3`.

Parked, outside cut-in to cut-out, it sees the hub wind `v`. Its ice is
*rotor icing*. At 8 m/s the section sees 47.9 m/s. At 12 m/s the tip speed
reaches its limit and the section sees 68.5 m/s.

These are the numbers of a generic 2 to 5 MW rotor, not of any particular
turbine. `BladeProxy` holds them. Because of aerodynamic heating
(`0.79 v^2 / 2 c_p`, about 0.9 K at 48 m/s), a running blade stays ice-free
just below 0 degC while the still reference collector ices.

### Meteorological icing and the IEA ice class

Meteorological icing is air below 0 degC with an LWC above 0.01 g m-3: the
conditions for active accretion.

The IEA Wind Task 19 ice class comes from the share of the period with
meteorological icing:

| IEA ice class | Meteorological icing |
|---|---|
| 1 | at most 0.5 % |
| 2 | 0.5 to 3 % |
| 3 | 3 to 5 % |
| 4 | 5 to 10 % |
| 5 | above 10 % |

The classes are defined on annual statistics. A shorter period is classed
as if it were a year, and a note says so.

### Production loss

This is a simple, documented model, not a calibrated one. It works from the
blade-proxy ice mass `M`:

| Blade ice | Loss fraction |
|---|---|
| below 10 g/m | 0 |
| 10 g/m up to 1.0 kg/m | `0.5 M / 1.0 kg m-1`: the power curve degrades, up to half |
| 1.0 kg/m and above | 1: the turbine stops (ice detection, or rotor imbalance) |

The period's energy loss is `sum(P L) / sum(P)`. `P` is the power curve at
the IEC 61400-12 density-corrected hub wind: the packaged generic curve,
or one you name. A stopped rotor does not feed back on accretion.

Before relying on the loss numbers, calibrate the three `IcingLossModel`
numbers against SCADA data from iced turbines. IEA Wind Task 19 reports
annual production losses of 20 % and more in the heaviest ice classes.

### Time stepping

A rate at time `t_n` is diagnosed from the weather at `t_n` and the ice
already on the collector. It is held over the interval `(t_{n-1}, t_n]`,
the convention `precipitation_rate` uses in `woof energy extract`. The mass
at the first time is the initial state: zero, or the state carried over
from a previous chunk. An unknown input (NaN) makes the ice unknown from
then on, never zero.

## Running it on sites

```bash
woof energy rating wales/forecast.nc --products turbine-icing,wind-power -o wales/products.nc
```

`turbine-icing` is in the default product list. It is skipped, with a note,
when the forecast has no turbines. It is refused when an input it needs is
missing.

## Running it on a grid

`turbine_icing_dataset` takes either of two kinds of dataset:

- **A wrfout history** (`PH`, `PHB`, `HGT`, `T`, `P`, `PB`, `QVAPOR`,
  `QCLOUD`, `U` and `V`; `QRAIN` and `QNCLOUD` are used when present). It
  is interpolated to `hub_height_m` above model terrain the way the site
  sampler measures heights: `(PH + PHB) / g - HGT` at mass levels. U and V
  are destaggered to mass points. The derivations of temperature, density
  and relative humidity are those of `woof energy extract`.
- **A dataset with `forecast.v1` variable names.** These are on a 1-D
  `height` dimension (interpolated to `hub_height_m`), or already at hub
  height.

```python
import xarray as xr
from woof.energy.icing_turbine import turbine_icing_dataset

ds = xr.open_mfdataset("run/wrfout_d02_*", combine="nested",
                       concat_dim="Time", chunks={"south_north": 128})
icing = turbine_icing_dataset(ds, hub_height_m=110.0)
icing.to_zarr("icing.zarr")
```

All other dimensions are kept. The time loop runs over whole spatial
arrays. Dask-backed inputs stay lazy and are mapped over spatial chunks.
Time must be a single chunk, and it is rechunked if it is not, so chunk
over space.

`device="gpu"` moves each chunk to CuPy and back. `device="auto"`, the
default, uses CuPy only when the data already are CuPy arrays.

### Chunked hindcasts

A year of 15-minute output need not be processed in one call. Start each
chunk at the previous chunk's last time, and pass the previous output as
`initial_state`:

```python
first = turbine_icing_dataset(ds.isel(Time=slice(0, 2881)), hub_height_m=110.0)
second = turbine_icing_dataset(ds.isel(Time=slice(2880, 5761)), hub_height_m=110.0,
                               initial_state=first)
```

The second chunk then continues the first exactly. The IEA class and the
fractions are per chunk, so combine them over the year yourself, for
example as duration-weighted means of the fractions.

## Limitations

- **The blade proxy is generic.** A cylinder stands in for an airfoil
  leading edge, at one radius, for one generic rotor. Real collection
  efficiency varies along the span and with the angle of attack.
- **No shedding.** Ice leaves the blade only by melting and sublimation.
- **Uncalibrated loss.** The production-loss numbers are assumptions. Fit
  them to SCADA data before quoting energy losses.
- **Model cloud water.** The answers are only as good as the microphysics'
  cloud water near the surface. Fog and low stratus at 100 m to 300 m grid
  spacing are hard for any mesoscale model.
- **Droplet size.** Without Thompson `QNCLOUD` the droplet size rests on a
  fixed droplet number.
- **Subsaturated cloud.** Inconsistent model states can have cloud water in
  air that is below saturation. Sublimation then competes with accretion.

## References

- Finstad, K. J., Lozowski, E. P. and Gates, E. M. (1988). A computational
  investigation of water droplet trajectories. *J. Atmos. Oceanic Technol.*
  5, 160-170.
- IEA Wind Task 19. *Wind energy projects in cold climates* (Expert Group
  Study on Recommended Practices 13), with the IEA ice classification.
- ISO 12494 (2017). *Atmospheric icing of structures.*
- Jones, K. F. (1998). A simple model for freezing rain ice loads.
  *Atmos. Res.* 46, 87-97.
- Makkonen, L. (2000). Models for the growth of rime, glaze, icicles and
  wet snow on structures. *Phil. Trans. R. Soc. Lond. A* 358, 2913-2939.
- Thompson, G., Field, P. R., Rasmussen, R. M. and Hall, W. D. (2008).
  Explicit forecasts of winter precipitation using an improved bulk
  microphysics scheme. Part II. *Mon. Wea. Rev.* 136, 5095-5115.
- Thompson, G. and Eidhammer, T. (2014). A study of aerosol impacts on
  clouds and precipitation development in a large winter cyclone.
  *J. Atmos. Sci.* 71, 3636-3658.
